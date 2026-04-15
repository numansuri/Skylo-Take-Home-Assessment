"""FastAPI query interface for anomaly logs and telemetry data."""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.models import AnomalyLogEntry

app = FastAPI(title="Skylo Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path("frontend/index.html")
LOG_PATH = Path("logs/anomalies.jsonl")
TELEMETRY_PATH = Path("logs/telemetry_run.json")


def load_anomalies() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    return FileResponse(FRONTEND_PATH, headers={"Cache-Control": "no-store"})


@app.get("/anomalies", response_model=list[AnomalyLogEntry])
async def get_anomalies(
    severity: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    anomalies = load_anomalies()
    if severity:
        anomalies = [a for a in anomalies if a["severity"] == severity.lower()]
    if start:
        anomalies = [a for a in anomalies if a["timestamp"] >= start]
    if end:
        anomalies = [a for a in anomalies if a["timestamp"] <= end]
    return anomalies


@app.get("/health")
async def health():
    anomalies = load_anomalies()
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in anomalies:
        counts[a.get("severity", "low")] += 1
    return {
        "status": "ok",
        "anomaly_count": len(anomalies),
        "by_severity": counts,
    }


@app.get("/telemetry")
async def get_telemetry():
    """Return full telemetry run summary for dashboard charts."""
    if not TELEMETRY_PATH.exists():
        return {"data": []}
    return json.loads(TELEMETRY_PATH.read_text())
