"""Deterministic anomaly detection using IsolationForest + EWMA, orchestrated via OpenAI Agents SDK.

The Agents SDK handles tool registration, orchestration, and structured output.
All actual computation is pure sklearn + pandas — no LLM reasoning in the detection path.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from agents import Agent, Runner, function_tool

from src.models import DetectionResult, SeverityLevel


# ---------------------------------------------------------------------------
# Module-level state (populated by DetectionAgent.fit_warmup)
# ---------------------------------------------------------------------------
_scaler: StandardScaler | None = None
_model: IsolationForest | None = None
_baseline_mean: float = 0.0
_baseline_std: float = 1.0


# ---------------------------------------------------------------------------
# Agents SDK function tools — pure deterministic computation
# ---------------------------------------------------------------------------
@function_tool
def run_isolation_forest(
    latency_ms: float,
    packet_loss_pct: float,
    rsrp_dbm: float,
) -> str:
    """Run IsolationForest scoring on a single telemetry data point.
    Returns the normalized anomaly score (negative = more anomalous, 0 = baseline mean)."""
    if _scaler is None or _model is None:
        return json.dumps({"error": "Model not fitted yet"})
    point = np.array([[latency_ms, packet_loss_pct, rsrp_dbm]])
    scaled = _scaler.transform(point)
    raw_score = float(_model.score_samples(scaled)[0])
    # Normalize: z-score relative to warm-up baseline
    norm_score = (raw_score - _baseline_mean) / _baseline_std if _baseline_std > 0 else 0.0
    return json.dumps({"isolation_score": round(norm_score, 6)})


@function_tool
def run_ewma_detection(
    metric_name: str,
    current_value: float,
    ewma_mean: float,
    ewma_std: float,
) -> str:
    """Compute EWMA z-score for a single metric.
    Returns z-score — flag if |z| > 2.5."""
    if ewma_std == 0:
        return json.dumps({"zscore": 0.0})
    zscore = (current_value - ewma_mean) / ewma_std
    return json.dumps({"zscore": round(zscore, 4)})


@function_tool
def compute_severity(
    isolation_score: float,
    max_ewma_zscore: float,
) -> str:
    """Compute deterministic severity tier from normalized IF score and max EWMA z-score.
    isolation_score is z-scored: negative values are more anomalous than baseline.
    Returns: 'critical', 'high', 'medium', or 'low'."""
    # Severity uses both IF score and EWMA z-score as complementary signals.
    # Critical requires extreme IF outlier AND multi-metric EWMA trigger.
    if isolation_score < -4.0 and max_ewma_zscore > 2.5:
        return "critical"
    elif isolation_score < -4.0 or (isolation_score < -2.5 and max_ewma_zscore > 2.5):
        return "high"
    elif isolation_score < -2.5 or max_ewma_zscore > 2.5:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------------------------
# Agent definition — uses gpt-5.4-mini purely for tool orchestration
# ---------------------------------------------------------------------------
detection_agent = Agent(
    name="RAN Detection Agent",
    instructions="""You are a deterministic anomaly detection engine for satellite RAN telemetry.
You MUST use your tools to compute anomaly scores — do not reason about the data yourself.

Workflow:
1. Call run_isolation_forest with the current data point values (latency_ms, packet_loss_pct, rsrp_dbm)
2. Call run_ewma_detection for each of the three metrics (latency_ms, packet_loss_pct, rsrp_dbm) using the provided EWMA state
3. Call compute_severity with the isolation score and the maximum absolute z-score
4. Return a structured DetectionResult with all computed values

is_anomaly is True if isolation_score < -2.5 OR any |ewma_zscore| > 2.5.
triggered_metrics should list any metric where |ewma_zscore| > 2.5.""",
    tools=[run_isolation_forest, run_ewma_detection, compute_severity],
    output_type=DetectionResult,
    model="gpt-5.4-mini",
)


class DetectionAgent:
    """Wraps the Agents SDK detection agent with sklearn model state and EWMA tracking."""

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
        global _scaler, _model, _baseline_mean, _baseline_std
        features = warmup_df[["latency_ms", "packet_loss_pct", "rsrp_dbm"]].values
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        self.model.fit(scaled)
        _scaler = self.scaler
        _model = self.model

        # Compute baseline score distribution for normalization
        baseline_scores = self.model.score_samples(scaled)
        self._baseline_mean = float(np.mean(baseline_scores))
        self._baseline_std = float(np.std(baseline_scores))
        if self._baseline_std < 1e-10:
            self._baseline_std = 1.0
        _baseline_mean = self._baseline_mean
        _baseline_std = self._baseline_std

        # Seed EWMA histories with warm-up data
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

        # Pre-compute values for the agent (and for fallback)
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
