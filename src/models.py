from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionResult(BaseModel):
    timestamp: str
    latency_ms: float
    packet_loss_pct: float
    rsrp_dbm: float
    isolation_score: float
    ewma_zscore_latency: float
    ewma_zscore_packet_loss: float
    ewma_zscore_rsrp: float
    is_anomaly: bool
    severity: SeverityLevel
    triggered_metrics: list[str]


class AnomalyInterpretation(BaseModel):
    reason: str = Field(description="1-2 sentence RAN engineer explanation of the anomaly")
    likely_cause: str = Field(description="Most probable root cause: handover failure, congestion, interference, etc.")
    operator_action: str = Field(description="Recommended next step for a network operator")


class AnomalyLogEntry(BaseModel):
    timestamp: str
    affected_metrics: list[str]
    severity: str
    reason: str
    likely_cause: str
    operator_action: str
    isolation_score: float
    raw_values: dict
    interpretation_model: str
    detection_latency_ms: float
