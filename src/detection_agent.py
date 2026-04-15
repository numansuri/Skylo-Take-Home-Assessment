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

        # Compute baseline score distribution for normalization
        baseline_scores = self.model.score_samples(scaled)
        self._baseline_mean = float(np.mean(baseline_scores))
        self._baseline_std = float(np.std(baseline_scores))
        if self._baseline_std < 1e-10:
            self._baseline_std = 1.0

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
        scaled = self.scaler.transform(point)
        raw_iso_score = float(self.model.score_samples(scaled)[0])
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
