"""LLM-powered anomaly interpretation agent.

Receives ONLY confirmed anomalies (is_anomaly=True) from the detection pipeline.
Uses GPT-5.4 mini via the OpenAI Agents SDK for grounded natural language interpretation.
"""

import asyncio

from agents import Agent, Runner
from src.models import DetectionResult, AnomalyInterpretation


SYSTEM_PROMPT = """You are a satellite RAN (Radio Access Network) anomaly analyst.
You receive structured telemetry anomaly data from a deterministic ML detection system.
Provide a concise, technically grounded interpretation for a network operations engineer.

Focus on:
- Which metrics are affected and by how much vs their baseline
- What RAN event this pattern is most consistent with (handover failure, congestion, interference, beam switch)
- What the operator should investigate first

Be specific. Reference actual metric values and deviations. Do not be vague.
Your output must be valid JSON matching the AnomalyInterpretation schema."""

interpretation_agent = Agent(
    name="RAN Interpretation Agent",
    instructions=SYSTEM_PROMPT,
    model="gpt-5.4-mini",
    output_type=AnomalyInterpretation,
)


async def interpret_anomaly(detection: DetectionResult) -> AnomalyInterpretation:
    """Send a confirmed anomaly to the interpretation agent for NL explanation."""
    context = (
        f"Timestamp: {detection.timestamp}\n"
        f"Raw values: latency={detection.latency_ms}ms, "
        f"packet_loss={detection.packet_loss_pct}%, rsrp={detection.rsrp_dbm}dBm\n"
        f"Baselines: latency~500ms, packet_loss~0.3%, rsrp~-100dBm\n"
        f"Isolation score: {detection.isolation_score:.3f} (threshold: -0.05)\n"
        f"EWMA z-scores: latency={detection.ewma_zscore_latency:.2f}, "
        f"packet_loss={detection.ewma_zscore_packet_loss:.2f}, "
        f"rsrp={detection.ewma_zscore_rsrp:.2f}\n"
        f"Triggered metrics: {detection.triggered_metrics}\n"
        f"Severity: {detection.severity}"
    )
    try:
        result = await Runner.run(interpretation_agent, input=context)
        return result.final_output
    except Exception as e:
        return AnomalyInterpretation(
            reason=f"[Interpretation unavailable — API error: {str(e)[:80]}]",
            likely_cause="Unknown",
            operator_action="Review raw telemetry manually",
        )


async def interpret_anomalies_parallel(
    detections: list[DetectionResult],
    max_concurrent: int = 10,
) -> list[AnomalyInterpretation]:
    """Interpret multiple anomalies concurrently with rate limiting.

    Uses asyncio.Semaphore to cap concurrent OpenAI API calls.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_interpret(detection: DetectionResult) -> AnomalyInterpretation:
        async with semaphore:
            return await interpret_anomaly(detection)

    tasks = [rate_limited_interpret(d) for d in detections]
    return await asyncio.gather(*tasks)
