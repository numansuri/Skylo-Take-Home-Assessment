"""Entry point — runs the full anomaly detection pipeline, then starts the API server.

Single command: python main.py
  1. Generates simulated satellite RAN telemetry
  2. Runs detection + interpretation pipeline
  3. Starts FastAPI server on http://localhost:8000
  4. Opens the dashboard in your default browser
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from src.simulator import generate_telemetry
from src.detection_agent import DetectionAgent
from src.interpretation_agent import interpret_anomalies_parallel
from src.models import AnomalyLogEntry

load_dotenv()

Path("logs").mkdir(exist_ok=True)
LOG_PATH = Path("logs/anomalies.jsonl")


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


def start_server():
    """Start the FastAPI server via uvicorn."""
    import uvicorn
    from src.api import app

    print()
    print("  Starting API server on http://localhost:8000 ...")
    print("  Dashboard: opening frontend/index.html in browser")
    print("  Press Ctrl+C to stop")
    print()

    # Open dashboard in browser after a short delay
    def open_browser():
        time.sleep(1.5)
        dashboard_path = Path("frontend/index.html").resolve()
        webbrowser.open(f"file://{dashboard_path}")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    # Step 1: Run the detection pipeline
    asyncio.run(run_pipeline())

    # Step 2: Start the API server + open dashboard
    start_server()
