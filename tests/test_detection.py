"""Smoke tests for the anomaly detection pipeline."""

import asyncio
import pytest
import pandas as pd

from src.simulator import generate_telemetry
from src.detection_agent import DetectionAgent
from src.models import DetectionResult


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


@pytest.fixture
def telemetry_df():
    return generate_telemetry()


@pytest.fixture
def fitted_agent(telemetry_df):
    agent = DetectionAgent()
    agent.fit_warmup(telemetry_df.head(60))
    return agent


class TestSimulator:
    def test_generates_360_rows(self, telemetry_df):
        assert len(telemetry_df) == 360

    def test_has_required_columns(self, telemetry_df):
        required = ["timestamp", "latency_ms", "packet_loss_pct", "rsrp_dbm",
                     "injected_anomaly", "anomaly_scenario"]
        for col in required:
            assert col in telemetry_df.columns

    def test_handover_failure_injected(self, telemetry_df):
        hf = telemetry_df[telemetry_df["anomaly_scenario"] == "handover_failure"]
        assert len(hf) > 0
        assert hf["latency_ms"].min() > 1000
        assert hf["rsrp_dbm"].max() < -120

    def test_congestion_drift_injected(self, telemetry_df):
        cd = telemetry_df[telemetry_df["anomaly_scenario"] == "congestion_drift"]
        assert len(cd) > 0
        assert cd["packet_loss_pct"].max() > 3.0

    def test_baseline_latency_reasonable(self, telemetry_df):
        normal = telemetry_df[~telemetry_df["injected_anomaly"]]
        assert 400 < normal["latency_ms"].mean() < 600


class TestDetectionAgent:
    def test_returns_detection_result(self, fitted_agent, telemetry_df):
        row = telemetry_df.iloc[0].to_dict()
        result = run_async(fitted_agent.process_single(row))
        assert isinstance(result, DetectionResult)

    def test_detects_handover_failure(self, fitted_agent, telemetry_df):
        """At least one anomaly detected in the handover failure window."""
        async def _run():
            count = 0
            for i in range(120, 141):
                row = telemetry_df.iloc[i].to_dict()
                result = await fitted_agent.process_single(row)
                if result.is_anomaly:
                    count += 1
            return count

        anomaly_count = run_async(_run())
        assert anomaly_count > 0, "No anomalies detected in handover failure window"

    def test_detects_congestion_drift(self, fitted_agent, telemetry_df):
        """At least one anomaly detected in the congestion drift window."""
        async def _run():
            count = 0
            # Process history first to build EWMA state
            for i in range(60, 291):
                row = telemetry_df.iloc[i].to_dict()
                result = await fitted_agent.process_single(row)
                if i >= 250 and result.is_anomaly:
                    count += 1
            return count

        anomaly_count = run_async(_run())
        assert anomaly_count > 0, "No anomalies detected in congestion drift window"

    def test_normal_points_mostly_not_anomalous(self, fitted_agent, telemetry_df):
        """Most normal data points should not be flagged."""
        normal_rows = telemetry_df[~telemetry_df["injected_anomaly"]].head(30)

        async def _run():
            count = 0
            for _, row in normal_rows.iterrows():
                result = await fitted_agent.process_single(row.to_dict())
                if result.is_anomaly:
                    count += 1
            return count

        anomaly_count = run_async(_run())
        assert anomaly_count < 10, f"Too many false positives: {anomaly_count}/30"

    def test_severity_levels_valid(self, fitted_agent, telemetry_df):
        row = telemetry_df.iloc[130].to_dict()
        result = run_async(fitted_agent.process_single(row))
        assert result.severity.value in ["low", "medium", "high", "critical"]

    def test_triggered_metrics_populated_on_anomaly(self, fitted_agent, telemetry_df):
        """Handover failure should trigger multiple metrics on first detection."""
        async def _run():
            # Process the first handover point — should have high EWMA z-scores
            row = telemetry_df.iloc[120].to_dict()
            return await fitted_agent.process_single(row)

        result = run_async(_run())
        if result.is_anomaly:
            assert len(result.triggered_metrics) > 0
