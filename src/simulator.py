"""Satellite RAN telemetry simulator.

Generates 30 minutes of telemetry at 5-second intervals (360 data points)
with two injected anomaly scenarios: Handover Failure and Congestion Drift.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_telemetry(
    duration_minutes: int = 30,
    interval_seconds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_points = (duration_minutes * 60) // interval_seconds  # 360

    start_time = datetime(2025, 1, 15, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(n_points)]

    # --- Baseline telemetry ---
    latency_ms = rng.normal(loc=500, scale=30, size=n_points)
    packet_loss_pct = rng.beta(a=2, b=600, size=n_points) * 100  # stays <1% normally
    rsrp_dbm = rng.normal(loc=-100, scale=3, size=n_points)
    rsrp_dbm = np.clip(rsrp_dbm, -110, -90)

    injected_anomaly = np.zeros(n_points, dtype=bool)
    anomaly_scenario = np.array([None] * n_points, dtype=object)

    # --- Scenario 1: Handover Failure (t=120 to t=140) ---
    for i in range(120, min(141, n_points)):
        latency_ms[i] = rng.uniform(1800, 2500)
        rsrp_dbm[i] = rng.uniform(-135, -125)
        packet_loss_pct[i] = rng.uniform(8, 15)
        injected_anomaly[i] = True
        anomaly_scenario[i] = "handover_failure"

    # --- Scenario 2: Congestion Drift (t=250 to t=290) ---
    drift_len = min(290, n_points - 1) - 250 + 1
    for j, i in enumerate(range(250, min(291, n_points))):
        progress = j / max(drift_len - 1, 1)
        packet_loss_pct[i] = 0.3 + progress * rng.uniform(5.7, 8.7)
        latency_ms[i] = 500 + progress * rng.uniform(300, 400)
        # RSRP stays normal — this is the key distinguisher
        injected_anomaly[i] = True
        anomaly_scenario[i] = "congestion_drift"

    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "latency_ms": np.round(latency_ms, 2),
        "packet_loss_pct": np.round(packet_loss_pct, 4),
        "rsrp_dbm": np.round(rsrp_dbm, 2),
        "injected_anomaly": injected_anomaly,
        "anomaly_scenario": anomaly_scenario,
    })

    return df
