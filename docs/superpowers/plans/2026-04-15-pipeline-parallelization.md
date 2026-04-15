# Pipeline Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce anomaly detection pipeline runtime from ~22-42s to ~1-2s by removing unnecessary LLM calls, vectorizing computation, and parallelizing interpretation.

**Architecture:** Three surgical changes: (1) `detection_agent.py` stops calling `Runner.run()` and returns deterministic results directly, (2) a new `process_batch()` method vectorizes sklearn scoring and EWMA z-score computation across all data points at once, (3) `main.py` collects all anomalies then fires interpretation LLM calls concurrently via `asyncio.gather` with a semaphore for rate limiting.

**Tech Stack:** Python asyncio, sklearn (IsolationForest batch scoring), pandas (vectorized EWMA), OpenAI Agents SDK (interpretation only)

---

### Task 1: Remove LLM from Detection Path

**Files:**
- Modify: `src/detection_agent.py:161-241`
- Test: `tests/test_detection.py`

The `process_single()` method already computes everything deterministically (lines 168-215) before calling the LLM (line 225). The LLM just re-does the same work via tool calls. Remove the LLM call and return the deterministic result directly.

- [ ] **Step 1: Modify `process_single()` to skip LLM**

In `src/detection_agent.py`, replace the try/except block (lines 216-241) with a direct return:

```python
    async def process_single(self, row: dict) -> DetectionResult:
        """Process a single telemetry data point through the detection pipeline.

        All computation is deterministic (sklearn + pandas). No LLM calls.
        """
        timestamp = row["timestamp"]
        latency = float(row["latency_ms"])
        packet_loss = float(row["packet_loss_pct"])
        rsrp = float(row["rsrp_dbm"])

        # Update EWMA histories
        self._ewma_histories["latency_ms"].append(latency)
        self._ewma_histories["packet_loss_pct"].append(packet_loss)
        self._ewma_histories["rsrp_dbm"].append(rsrp)

        # Pre-compute EWMA z-scores
        ewma_states = {}
        zscores = {}
        for metric in ["latency_ms", "packet_loss_pct", "rsrp_dbm"]:
            mean, std = self._compute_ewma_state(metric)
            ewma_states[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
            val = {"latency_ms": latency, "packet_loss_pct": packet_loss, "rsrp_dbm": rsrp}[metric]
            zscores[metric] = round((val - mean) / std, 4) if std > 0 else 0.0

        # Compute normalized isolation score
        point = np.array([[latency, packet_loss, rsrp]])
        scaled = _scaler.transform(point)
        raw_iso_score = float(_model.score_samples(scaled)[0])
        iso_score = self._normalize_iso_score(raw_iso_score)

        # Determine anomaly status
        max_abs_z = max(abs(zscores["latency_ms"]), abs(zscores["packet_loss_pct"]), abs(zscores["rsrp_dbm"]))
        is_anomaly = iso_score < -2.5 or max_abs_z > 2.5

        # Compute severity
        if iso_score < -4.0 and max_abs_z > 2.5:
            severity = SeverityLevel.CRITICAL
        elif iso_score < -4.0 or (iso_score < -2.5 and max_abs_z > 2.5):
            severity = SeverityLevel.HIGH
        elif iso_score < -2.5 or max_abs_z > 2.5:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW

        # Determine triggered metrics
        triggered = []
        if abs(zscores["latency_ms"]) > 2.5:
            triggered.append("latency_ms")
        if abs(zscores["packet_loss_pct"]) > 2.5:
            triggered.append("packet_loss_pct")
        if abs(zscores["rsrp_dbm"]) > 2.5:
            triggered.append("rsrp_dbm")

        return DetectionResult(
            timestamp=timestamp,
            latency_ms=latency,
            packet_loss_pct=packet_loss,
            rsrp_dbm=rsrp,
            isolation_score=round(iso_score, 6),
            ewma_zscore_latency=zscores["latency_ms"],
            ewma_zscore_packet_loss=zscores["packet_loss_pct"],
            ewma_zscore_rsrp=zscores["rsrp_dbm"],
            is_anomaly=is_anomaly,
            severity=severity,
            triggered_metrics=triggered,
        )
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `pytest tests/test_detection.py -v`
Expected: All tests PASS (the tests already exercise the deterministic path since the LLM fallback produces the same results)

- [ ] **Step 3: Commit**

```bash
git add src/detection_agent.py
git commit -m "perf: remove LLM from detection path, use deterministic computation directly"
```

---

### Task 2: Add Vectorized Batch Processing

**Files:**
- Modify: `src/detection_agent.py` (add `process_batch()` method)
- Modify: `tests/test_detection.py` (add batch tests)

Add a `process_batch()` method that scores all points at once using vectorized sklearn and pandas operations instead of looping one-at-a-time.

- [ ] **Step 1: Write failing test for `process_batch()`**

Add to `tests/test_detection.py`:

```python
class TestBatchProcessing:
    def test_batch_returns_list_of_detection_results(self, fitted_agent, telemetry_df):
        results = fitted_agent.process_batch(telemetry_df)
        assert isinstance(results, list)
        assert len(results) == len(telemetry_df)
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_batch_detects_handover_failure(self, fitted_agent, telemetry_df):
        results = fitted_agent.process_batch(telemetry_df)
        hf_anomalies = [r for i, r in enumerate(results) if 120 <= i <= 140 and r.is_anomaly]
        assert len(hf_anomalies) > 0, "No anomalies detected in handover failure window"

    def test_batch_detects_congestion_drift(self, fitted_agent, telemetry_df):
        results = fitted_agent.process_batch(telemetry_df)
        cd_anomalies = [r for i, r in enumerate(results) if 250 <= i <= 290 and r.is_anomaly]
        assert len(cd_anomalies) > 0, "No anomalies detected in congestion drift window"

    def test_batch_normal_points_mostly_not_anomalous(self, fitted_agent, telemetry_df):
        results = fitted_agent.process_batch(telemetry_df)
        normal_indices = telemetry_df.index[~telemetry_df["injected_anomaly"]].tolist()
        false_positives = sum(1 for i in normal_indices[:30] if results[i].is_anomaly)
        assert false_positives < 10, f"Too many false positives: {false_positives}/30"

    def test_batch_matches_single_processing(self, fitted_agent, telemetry_df):
        """Batch results must match sequential process_single results exactly."""
        import asyncio

        batch_results = fitted_agent.process_batch(telemetry_df)

        # Reset agent state and run sequentially for comparison
        agent2 = DetectionAgent()
        agent2.fit_warmup(telemetry_df.head(60))

        async def run_sequential():
            results = []
            for _, row in telemetry_df.iterrows():
                r = await agent2.process_single(row.to_dict())
                results.append(r)
            return results

        sequential_results = asyncio.run(run_sequential())

        for i in range(len(telemetry_df)):
            assert batch_results[i].is_anomaly == sequential_results[i].is_anomaly, \
                f"Mismatch at index {i}: batch={batch_results[i].is_anomaly}, seq={sequential_results[i].is_anomaly}"
            assert batch_results[i].severity == sequential_results[i].severity, \
                f"Severity mismatch at index {i}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detection.py::TestBatchProcessing -v`
Expected: FAIL with `AttributeError: 'DetectionAgent' object has no attribute 'process_batch'`

- [ ] **Step 3: Implement `process_batch()`**

Add this method to the `DetectionAgent` class in `src/detection_agent.py`:

```python
    def process_batch(self, df: pd.DataFrame) -> list[DetectionResult]:
        """Process all data points using vectorized operations. No LLM calls.

        Replaces the per-point process_single() loop with batch sklearn scoring
        and vectorized EWMA z-score computation.
        """
        n = len(df)
        timestamps = df["timestamp"].tolist()
        latency = df["latency_ms"].values.astype(float)
        packet_loss = df["packet_loss_pct"].values.astype(float)
        rsrp = df["rsrp_dbm"].values.astype(float)

        # --- Batch IsolationForest scoring ---
        features = np.column_stack([latency, packet_loss, rsrp])
        scaled = self.scaler.transform(features)
        raw_iso_scores = self.model.score_samples(scaled)
        iso_scores = (raw_iso_scores - self._baseline_mean) / self._baseline_std

        # --- Vectorized EWMA z-scores ---
        zscores = {}
        for col in ["latency_ms", "packet_loss_pct", "rsrp_dbm"]:
            # Prepend warm-up history so EWMA state is correct
            history = self._ewma_histories[col].copy()
            full_series = pd.Series(history + df[col].tolist())
            ewma_mean = full_series.ewm(span=self.ewma_span, adjust=False).mean()
            ewma_std = full_series.ewm(span=self.ewma_span, adjust=False).std()
            ewma_std = ewma_std.replace(0, 1e-6).fillna(1e-6)

            # Slice off the warm-up prefix to get z-scores for the actual data
            offset = len(history)
            values = full_series.values[offset:]
            means = ewma_mean.values[offset:]
            stds = ewma_std.values[offset:]
            zscores[col] = np.round((values - means) / stds, 4)

        # --- Vectorized anomaly flags + severity ---
        abs_z_latency = np.abs(zscores["latency_ms"])
        abs_z_packet_loss = np.abs(zscores["packet_loss_pct"])
        abs_z_rsrp = np.abs(zscores["rsrp_dbm"])
        max_abs_z = np.maximum(np.maximum(abs_z_latency, abs_z_packet_loss), abs_z_rsrp)

        is_anomaly = (iso_scores < -2.5) | (max_abs_z > 2.5)

        # --- Build results ---
        results = []
        for i in range(n):
            iso = round(float(iso_scores[i]), 6)
            maz = float(max_abs_z[i])

            if iso < -4.0 and maz > 2.5:
                severity = SeverityLevel.CRITICAL
            elif iso < -4.0 or (iso < -2.5 and maz > 2.5):
                severity = SeverityLevel.HIGH
            elif iso < -2.5 or maz > 2.5:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW

            triggered = []
            if abs_z_latency[i] > 2.5:
                triggered.append("latency_ms")
            if abs_z_packet_loss[i] > 2.5:
                triggered.append("packet_loss_pct")
            if abs_z_rsrp[i] > 2.5:
                triggered.append("rsrp_dbm")

            results.append(DetectionResult(
                timestamp=timestamps[i],
                latency_ms=float(latency[i]),
                packet_loss_pct=float(packet_loss[i]),
                rsrp_dbm=float(rsrp[i]),
                isolation_score=iso,
                ewma_zscore_latency=float(zscores["latency_ms"][i]),
                ewma_zscore_packet_loss=float(zscores["packet_loss_pct"][i]),
                ewma_zscore_rsrp=float(zscores["rsrp_dbm"][i]),
                is_anomaly=bool(is_anomaly[i]),
                severity=severity,
                triggered_metrics=triggered,
            ))

        # Update EWMA histories so subsequent calls have correct state
        for col in ["latency_ms", "packet_loss_pct", "rsrp_dbm"]:
            self._ewma_histories[col].extend(df[col].tolist())

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detection.py -v`
Expected: All tests PASS, including the new `TestBatchProcessing` tests and the `test_batch_matches_single_processing` equivalence test.

- [ ] **Step 5: Commit**

```bash
git add src/detection_agent.py tests/test_detection.py
git commit -m "perf: add vectorized process_batch() for batch sklearn scoring and EWMA"
```

---

### Task 3: Parallelize Interpretation + Restructure Main Pipeline

**Files:**
- Modify: `src/interpretation_agent.py` (add `interpret_anomalies_parallel()`)
- Modify: `main.py` (use `process_batch()` + parallel interpretation)

- [ ] **Step 1: Add `interpret_anomalies_parallel()` to `src/interpretation_agent.py`**

Add this function after the existing `interpret_anomaly()`:

```python
async def interpret_anomalies_parallel(
    detections: list[DetectionResult],
    max_concurrent: int = 10,
) -> list[AnomalyInterpretation]:
    """Interpret multiple anomalies concurrently with rate limiting.

    Uses asyncio.Semaphore to cap concurrent OpenAI API calls.
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_interpret(detection: DetectionResult) -> AnomalyInterpretation:
        async with semaphore:
            return await interpret_anomaly(detection)

    tasks = [rate_limited_interpret(d) for d in detections]
    return await asyncio.gather(*tasks)
```

- [ ] **Step 2: Rewrite `main.py` `run_pipeline()` to use batch detection + parallel interpretation**

Replace the `run_pipeline()` function in `main.py` with:

```python
async def run_pipeline():
    """Run the detection + interpretation pipeline."""
    LOG_PATH.write_text("")  # Clear previous run

    print("=" * 55)
    print("  SKYLO — Satellite RAN Anomaly Detection Engine")
    print("=" * 55)
    print()

    print("[1/4] Generating satellite RAN telemetry...")
    df = generate_telemetry()
    print(f"      {len(df)} data points (30 min at 5-sec intervals)")
    print(f"      Injected: handover_failure (t=120–140), congestion_drift (t=250–290)")
    print()

    print("[2/4] Fitting IsolationForest on warm-up window (first 60 points)...")
    agent = DetectionAgent()
    agent.fit_warmup(df.head(60))
    print("      Model fitted. Scaler + IF ready.")
    print()

    print("[3/4] Running batch detection pipeline...")
    t0 = time.time()
    detections = agent.process_batch(df)
    detection_ms = (time.time() - t0) * 1000
    anomaly_detections = [(i, d) for i, d in enumerate(detections) if d.is_anomaly]
    print(f"      {len(df)} points scored in {detection_ms:.1f}ms")
    print(f"      {len(anomaly_detections)} anomalies detected")
    print()

    # Build telemetry results for dashboard
    all_results = []
    for i, (_, row) in enumerate(df.iterrows()):
        result_row = row.to_dict()
        for k, v in list(result_row.items()):
            if hasattr(v, "item"):
                result_row[k] = v.item()
            try:
                if v != v:  # NaN check
                    result_row[k] = None
            except (TypeError, ValueError):
                pass
        result_row["isolation_score"] = detections[i].isolation_score
        result_row["is_anomaly"] = detections[i].is_anomaly
        all_results.append(result_row)

    # Parallel interpretation of all anomalies
    if anomaly_detections:
        print(f"[3b/4] Interpreting {len(anomaly_detections)} anomalies in parallel...")
        t1 = time.time()
        anomaly_dets = [d for _, d in anomaly_detections]
        interpretations = await interpret_anomalies_parallel(anomaly_dets)
        interp_ms = (time.time() - t1) * 1000
        print(f"       {len(interpretations)} interpretations in {interp_ms:.1f}ms")
        print()

        # Write anomaly log entries
        for (idx, detection), interpretation in zip(anomaly_detections, interpretations):
            severity_str = detection.severity.value if hasattr(detection.severity, "value") else detection.severity
            print(f"      t={idx:>3} [{severity_str.upper()}] {interpretation.likely_cause}")

            entry = AnomalyLogEntry(
                timestamp=detection.timestamp,
                affected_metrics=detection.triggered_metrics,
                severity=detection.severity.value if hasattr(detection.severity, "value") else detection.severity,
                reason=interpretation.reason,
                likely_cause=interpretation.likely_cause,
                operator_action=interpretation.operator_action,
                isolation_score=detection.isolation_score,
                raw_values={
                    "latency_ms": detection.latency_ms,
                    "packet_loss_pct": detection.packet_loss_pct,
                    "rsrp_dbm": detection.rsrp_dbm,
                },
                interpretation_model="gpt-5.4-mini",
                detection_latency_ms=round(detection_ms / len(df), 2),
            )
            with open(LOG_PATH, "a") as f:
                f.write(entry.model_dump_json() + "\n")

    # Write full telemetry for dashboard
    Path("logs/telemetry_run.json").write_text(
        json.dumps({"data": all_results}, default=str)
    )

    # Summary
    anomalies_detected = len(anomaly_detections)
    cost_estimate = anomalies_detected * 200 * 0.75 / 1_000_000
    print()
    print("[4/4] Pipeline complete")
    print("=" * 55)
    print(f"  Total data points:    {len(df)}")
    print(f"  Anomalies detected:   {anomalies_detected}")
    print(f"  Detection time:       {detection_ms:.1f}ms")
    print(f"  Estimated LLM cost:   ${cost_estimate:.6f}")
    print(f"  Log file:             logs/anomalies.jsonl")
    print(f"  Telemetry file:       logs/telemetry_run.json")
    print("=" * 55)

    return anomalies_detected
```

Also update the import at the top of `main.py`:

```python
from src.interpretation_agent import interpret_anomaly, interpret_anomalies_parallel
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_detection.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/interpretation_agent.py main.py
git commit -m "perf: parallelize interpretation with asyncio.gather, use batch detection in pipeline"
```

---

### Task 4: Clean Up Unused Agents SDK Code

**Files:**
- Modify: `src/detection_agent.py` (remove unused agent definition and tools)

After removing the LLM from the detection path, the `detection_agent` Agent definition (lines 85-101), the three `@function_tool` functions (lines 29-79), and the `from agents import Agent, Runner, function_tool` import are dead code.

- [ ] **Step 1: Remove unused imports and agent definition**

In `src/detection_agent.py`:
- Remove `from agents import Agent, Runner, function_tool` (line 12)
- Remove `import json` (line 7, only used by the tool functions)
- Remove the three `@function_tool` functions: `run_isolation_forest` (lines 29-44), `run_ewma_detection` (lines 47-59), `compute_severity` (lines 62-79)
- Remove the `detection_agent` Agent definition (lines 85-101)
- Remove the module-level state variables `_scaler`, `_model`, `_baseline_mean`, `_baseline_std` (lines 20-24) and update `fit_warmup()` to stop setting them
- Update `process_single()` to use `self.scaler`/`self.model` instead of the globals

The resulting file should only contain:

```python
"""Deterministic anomaly detection using IsolationForest + EWMA.

All computation is pure sklearn + pandas — no LLM in the detection path.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models import DetectionResult, SeverityLevel


class DetectionAgent:
    """Anomaly detection using IsolationForest + EWMA z-scores."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.ewma_span = 20
        self._ewma_histories: dict[str, list[float]] = {
            "latency_ms": [],
            "packet_loss_pct": [],
            "rsrp_dbm": [],
        }
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0

    def fit_warmup(self, warmup_df: pd.DataFrame) -> None:
        """Fit IsolationForest and StandardScaler on the warm-up window."""
        features = warmup_df[["latency_ms", "packet_loss_pct", "rsrp_dbm"]].values
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        self.model.fit(scaled)

        baseline_scores = self.model.score_samples(scaled)
        self._baseline_mean = float(np.mean(baseline_scores))
        self._baseline_std = float(np.std(baseline_scores))
        if self._baseline_std < 1e-10:
            self._baseline_std = 1.0

        for col in ["latency_ms", "packet_loss_pct", "rsrp_dbm"]:
            self._ewma_histories[col] = warmup_df[col].tolist()

    def _compute_ewma_state(self, metric: str) -> tuple[float, float]:
        """Compute current EWMA mean and std for a metric from its history."""
        series = pd.Series(self._ewma_histories[metric])
        ewma_mean = float(series.ewm(span=self.ewma_span, adjust=False).mean().iloc[-1])
        ewma_std = float(series.ewm(span=self.ewma_span, adjust=False).std().iloc[-1])
        if np.isnan(ewma_std) or ewma_std == 0:
            ewma_std = 1e-6
        return ewma_mean, ewma_std

    def _normalize_iso_score(self, raw_score: float) -> float:
        """Normalize IF score to z-score relative to warm-up baseline."""
        return (raw_score - self._baseline_mean) / self._baseline_std

    async def process_single(self, row: dict) -> DetectionResult:
        # ... (updated to use self.scaler/self.model instead of globals)

    def process_batch(self, df: pd.DataFrame) -> list[DetectionResult]:
        # ... (already uses self.scaler/self.model)
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_detection.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/detection_agent.py
git commit -m "refactor: remove unused Agents SDK code from detection path"
```
