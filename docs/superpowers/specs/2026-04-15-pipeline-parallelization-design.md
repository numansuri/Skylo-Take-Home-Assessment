# Pipeline Parallelization Design

**Goal:** Reduce anomaly detection pipeline runtime from ~22-42s to ~1-2s.

**Three optimizations:**
1. Remove LLM from detection path — use deterministic computation directly
2. Vectorize sklearn scoring + EWMA into batch operations
3. Parallelize interpretation LLM calls with `asyncio.gather` + Semaphore

**Constraints:** Output format, thresholds, severity logic, and API endpoints remain identical.
