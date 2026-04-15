"""Entry point — runs the full anomaly detection pipeline.

Generates simulated satellite RAN telemetry, runs each data point through
the deterministic detection agent, and sends confirmed anomalies to the
LLM interpretation agent. Results are logged to logs/anomalies.jsonl.
"""

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.simulator import generate_telemetry
from src.detection_agent import DetectionAgent
from src.interpretation_agent import interpret_anomaly
from src.models import AnomalyLogEntry

load_dotenv()

Path("logs").mkdir(exist_ok=True)
LOG_PATH = Path("logs/anomalies.jsonl")


async def main():
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

    anomalies_detected = 0
    all_results = []

    print("[3/4] Running detection pipeline...")
    for i, row in df.iterrows():
        t0 = time.time()
        detection = await agent.process_single(row.to_dict())
        detection_latency = (time.time() - t0) * 1000

        result_row = row.to_dict()
        # Convert numpy types and NaN/None to JSON-safe Python types
        for k, v in list(result_row.items()):
            if hasattr(v, "item"):
                result_row[k] = v.item()
            try:
                if v != v:  # NaN check
                    result_row[k] = None
            except (TypeError, ValueError):
                pass
        result_row["isolation_score"] = detection.isolation_score
        result_row["is_anomaly"] = detection.is_anomaly
        all_results.append(result_row)

        if detection.is_anomaly:
            anomalies_detected += 1
            interpretation = await interpret_anomaly(detection)

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
                detection_latency_ms=round(detection_latency, 2),
            )
            with open(LOG_PATH, "a") as f:
                f.write(entry.model_dump_json() + "\n")

            severity_str = detection.severity.value if hasattr(detection.severity, "value") else detection.severity
            print(f"      [{severity_str.upper():>8}] t={i:>3} {detection.timestamp} — {detection.triggered_metrics}")

    # Write full telemetry for dashboard
    Path("logs/telemetry_run.json").write_text(
        json.dumps({"data": all_results}, default=str)
    )

    # Summary
    cost_estimate = anomalies_detected * 200 * 0.75 / 1_000_000
    print()
    print("[4/4] Pipeline complete")
    print("=" * 55)
    print(f"  Total data points:    {len(df)}")
    print(f"  Anomalies detected:   {anomalies_detected}")
    print(f"  Estimated LLM cost:   ${cost_estimate:.6f}")
    print(f"  Log file:             logs/anomalies.jsonl")
    print(f"  Telemetry file:       logs/telemetry_run.json")
    print()
    print("  Dashboard:")
    print("    1. uvicorn src.api:app --reload")
    print("    2. Open frontend/index.html in browser")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
