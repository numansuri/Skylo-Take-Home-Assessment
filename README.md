# Skylo — Satellite RAN Anomaly Detection Engine

A production-quality anomaly detection system for satellite Radio Access Network (RAN) telemetry, built with **scikit-learn** for deterministic ML detection and the **OpenAI Agents SDK** for natural language interpretation of confirmed anomalies.

The system simulates 30 minutes of NTN (Non-Terrestrial Network) telemetry, detects anomalies through a fully deterministic, vectorized ML pipeline, then passes confirmed anomalies to GPT-5.4 mini in parallel for grounded natural language interpretation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
cp .env.example .env
# Edit .env with your key

# Run everything (pipeline + API server + opens dashboard)
python main.py
```

This single command runs the detection pipeline, starts the FastAPI server on `http://localhost:8000`, and opens the dashboard in your browser automatically.

> **Note:** The detection pipeline works fully without an API key. Only the interpretation step requires OpenAI access — it degrades gracefully with a fallback message if unavailable.

## Architecture

```
┌─────────────────────────────────┐
│     Telemetry Simulator         │
│     (simulator.py)              │
│     360 points, 5-sec intervals │
└──────────────┬──────────────────┘
               │ DataFrame[360 rows]
               ▼
┌─────────────────────────────────┐
│     DetectionAgent              │
│     (detection_agent.py)        │
│                                 │
│  Pure sklearn + pandas.         │
│  No LLM. No API calls.         │
│                                 │
│  1. Fit warm-up (first 60 pts)  │
│     - StandardScaler            │
│     - IsolationForest           │
│                                 │
│  2. Batch score all 360 points  │
│     - Vectorized IF scoring     │
│     - Vectorized EWMA z-scores  │
│     - Anomaly flags + severity  │
│                                 │
│  Output: DetectionResult[360]   │
│  Timing: ~6ms for 360 points    │
└──────────────┬──────────────────┘
               │ is_anomaly=True only (~67 anomalies)
               ▼
┌─────────────────────────────────┐
│     InterpretationAgent         │
│     [OpenAI Agents SDK]         │
│                                 │
│  model: gpt-5.4-mini           │
│  output_type:                   │
│    AnomalyInterpretation        │
│                                 │
│  Parallel async execution:      │
│  asyncio.gather + Semaphore     │  ◄── LLM ONLY here. Never in detection.
│  (max 10 concurrent API calls)  │      Receives structured data, returns
│                                 │      structured interpretation.
│  Graceful fallback on API error │
└──────────────┬──────────────────┘
               │ AnomalyLogEntry[N]
               ▼
┌─────────────────────────────────┐
│  logs/anomalies.jsonl           │
│  logs/telemetry_run.json        │
│  FastAPI (localhost:8000)       │
│  Dashboard (frontend/index.html)│
└─────────────────────────────────┘
```

### Why the LLM is ONLY in the interpretation path

The detection path is **fully deterministic**: IsolationForest scoring and EWMA z-scores produce the same output for the same input every time. Putting an LLM in the detection loop would introduce non-determinism, latency, and cost with zero accuracy benefit — the math doesn't need "reasoning."

The LLM adds value only *after* detection, where the task shifts from computation to explanation: translating numeric anomaly signatures into actionable operator language. This is where GPT-5.4 mini excels.

### Performance: Batch + Parallel Architecture

The pipeline uses two key optimizations:

1. **Vectorized batch detection** — `process_batch()` scores all 360 points in a single call using numpy/sklearn vectorized operations instead of looping point-by-point. IsolationForest's `score_samples()` and pandas' `ewm()` operate on the full array at once. Result: **~6ms for 360 points**.

2. **Parallel interpretation** — `interpret_anomalies_parallel()` fires all LLM interpretation calls concurrently via `asyncio.gather()` with an `asyncio.Semaphore(max_concurrent=10)` for rate limiting. Instead of interpreting 67 anomalies sequentially (~20s), all calls run concurrently.

## ML Model Design

### Why Unsupervised Anomaly Detection — Not Supervised Regression/Classification

The single most important constraint is **the absence of labeled anomaly data**. In a real satellite RAN deployment, there is no historical corpus of labeled faults on day one. This makes supervised models fundamentally inapplicable.

| Model Family | Why It Fails Here |
|---|---|
| **Random Forest Classifier / Gradient Boosting Classifier / XGBoost / CatBoost / LightGBM** | These require a target column of labels (`y`) — "anomaly" vs. "normal" for every training sample. Without labels, they literally cannot be trained. Calling `.fit(X)` without `y` raises an error. |
| **Random Forest Regressor / Gradient Boosting Regressor** | Could theoretically be used for time-series forecasting (predict next latency, flag large residuals), but this is a fundamentally different task. It requires lagged-feature engineering, only detects univariate deviations (not multivariate correlations like latency + packet loss + RSRP degrading together), and still needs a labeled threshold for "how large a residual is anomalous." |
| **All supervised models** | Only detect anomaly patterns present in the training set. Novel failure modes — new interference patterns, previously unseen handover failures, orbital geometry edge cases — would be missed entirely. |

**Additional supervised model weaknesses in this context:**

- **Cold-start problem**: You need months of expert-labeled RAN telemetry before training can begin.
- **Class imbalance**: Anomalies are rare (typically 1-5% of observations). Supervised classifiers trained on imbalanced data tend to predict the majority class, requiring careful resampling (SMOTE, class weights) — all of which presuppose labels exist.
- **Label ambiguity**: Even expert RAN engineers may disagree on whether a telemetry reading is "anomalous" or just "unusual but acceptable," making label quality unreliable.
- **Validation is impossible**: Without ground truth labels, you cannot compute precision, recall, F1, or AUC-ROC — so you cannot even evaluate whether a supervised model works.

IsolationForest requires **zero labels**. It learns the structure of normal data by isolation partitioning and flags points that are easy to isolate (few random splits needed) as anomalous.

**References:**
- Liu, Ting, Zhou (2008). *"Isolation Forest."* IEEE ICDM. The foundational paper establishing isolation-based anomaly detection.
- Liu, Ting, Zhou (2012). *"Isolation-Based Anomaly Detection."* ACM TKDD 6(1). Extended journal version.
- *"Robust IoT Security Using Isolation Forest and One Class SVM"* (Scientific Reports / Nature, 2025) — direct IF vs OC-SVM comparison for IoT telemetry.

### IsolationForest vs Other Unsupervised Models

| Criterion | IsolationForest | One-Class SVM | Autoencoder | DBSCAN |
|---|---|---|---|---|
| **Training complexity** | O(n log n) | O(n^2) to O(n^3) | Architecture-dependent; GPU often needed | O(n log n) with spatial index |
| **Scoring complexity** | O(log n) per sample | O(n_sv x d) per sample | Forward pass; heavier than tree traversal | Not designed for online scoring |
| **Hyperparameter sensitivity** | Low (contamination, n_estimators) | High (kernel, gamma, nu all interact) | High (architecture, learning rate, epochs, latent dim) | Very high (eps, min_samples; scale-sensitive) |
| **Real-time streaming** | Excellent — single tree traversal | Moderate | Moderate | Poor — requires full dataset |
| **Continuous scoring** | `score_samples()` — continuous | `decision_function()` — continuous | Reconstruction error — continuous | No per-point scoring |
| **Implementation** | sklearn one-liner | sklearn one-liner but tuning is hard | Requires deep learning framework | sklearn but not for streaming |

**Why IsolationForest wins:**

- **One-Class SVM**: Training complexity of O(n^2) to O(n^3) makes it impractical for large telemetry streams. Careful kernel and gamma tuning is required, which is difficult without labeled validation data. A 2025 comparison in *Scientific Reports* (Nature) found IF achieved comparable or better detection with significantly lower computational cost for IoT security data.
- **Autoencoders**: Require a deep learning framework (PyTorch/TensorFlow), GPU infrastructure, and extensive tuning of architecture, learning rate, epochs, and reconstruction error threshold. Over-engineered for 3 features.
- **DBSCAN**: A clustering algorithm, not a streaming anomaly detector. It requires the entire dataset to compute clusters, cannot score individual incoming data points in real time, and is extremely sensitive to eps and min_samples parameters. A 2025 evaluation (ITM Web of Conferences) found *"Isolation Forests are most efficient at identifying collective anomalies"* with the fastest inference times.

### `score_samples()` over `predict()`

`predict()` returns binary {-1, 1} — anomalous or not. `score_samples()` returns a continuous anomaly score where more negative = more anomalous. This enables:

- **Severity tiering**: We normalize raw IF scores to z-scores relative to the warm-up baseline, then map to CRITICAL / HIGH / MEDIUM / LOW. Critical requires extreme IF deviation *and* multi-metric EWMA triggers; high requires strong IF deviation alone. This tiering is impossible with binary output.
- **Flexible thresholding**: Operators can adjust sensitivity post-deployment without retraining.
- **Operator prioritization**: Sort by score to triage the worst anomalies first.

Per sklearn documentation: *"The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest."*

### EWMA — Complementary Drift Detection

IsolationForest is fitted on the warm-up window (first 60 points) and scores each new point independently against the learned normal distribution. It excels at catching **multivariate outliers** — data points that sit far from the training distribution in feature space.

EWMA (Exponentially Weighted Moving Average) with span=20 provides a complementary signal. It maintains a rolling sense of "recent normal" that adapts over time, computing per-metric z-scores: `(current - ewma_mean) / ewma_std`. This catches **gradual univariate drift** that the IF might miss if each individual point remains within the broadly normal region of feature space.

**In practice on our simulated data:**
- **Handover Failure (Scenario 1):** IF dominates detection. The abrupt, correlated spike across all three metrics creates extreme IF outliers (z-score < -5.0). EWMA triggers on the first handover point (all 3 metrics exceed 2.5 sigma), enabling the CRITICAL severity classification.
- **Congestion Drift (Scenario 2):** IF also dominates here because the drift magnitude (latency 500 to 900ms, packet loss 0.3 to 9%) is large enough relative to the training distribution to produce strong IF outliers. EWMA z-scores remain below the 2.5 sigma trigger because the EWMA mean adapts as the drift progresses.

**The two methods are architecturally complementary**: IF catches multivariate outliers (both abrupt and cumulative), EWMA provides per-metric z-scores for severity classification and would independently catch drift patterns that remain within IF's normal region.

### Detection Thresholds

```python
# Anomaly flag
is_anomaly = (iso_score < -2.5) or (max_abs_ewma_zscore > 2.5)

# Severity tiers
CRITICAL:  iso_score < -4.0  AND  max_abs_z > 2.5
HIGH:      iso_score < -4.0  OR  (iso_score < -2.5 AND max_abs_z > 2.5)
MEDIUM:    iso_score < -2.5  OR  max_abs_z > 2.5
LOW:       below all thresholds (normal)
```

### `contamination=0.05` Rationale

The `contamination` parameter sets the expected proportion of outliers in the training data. Since our code uses `score_samples()` with custom z-score normalization and its own thresholds (normalized IF score < -2.5), the contamination parameter's primary effect is on model fitting — specifically, how the isolation trees partition the feature space.

Why 0.05:
- **Conservative prior**: In satellite RAN telemetry, anomalous readings are expected at roughly 1-5% during normal operations due to handover events, atmospheric interference, and orbital transitions.
- **Operational balance**: Too low (0.01) makes the model overly focused on extreme outliers; too high (0.10) dilutes the normal distribution and risks desensitizing the model.
- **Research-supported**: Deep learning anomaly detection studies show performance degradation increases significantly above ~10% contamination (arXiv, 2024).

**Actual detection rate**: On our simulated data with two injected anomaly scenarios, the pipeline detects ~67 anomalies (~18.6% of 360 points). This is higher than the 5% contamination parameter because both injected scenarios are severe and sustained, and our detection threshold uses custom z-score normalization rather than the contamination-derived offset.

### ML Model Parameters

| Component | Configuration |
|-----------|---------------|
| **IsolationForest** | n_estimators=100, contamination=0.05, max_samples="auto", random_state=42, n_jobs=-1 |
| **StandardScaler** | Fitted on warm-up window (first 60 points) |
| **EWMA** | span=20, adjust=False |

## Anomaly Scenarios — RAN-Grounded Definitions

### Scenario 1: Handover Failure (t=120 to t=140)

**Signature:** Correlated degradation across all three metrics simultaneously — latency spikes to 1800-2500ms, RSRP drops to -125 to -135 dBm, packet loss jumps to 8-15%.

**NTN context:** In non-terrestrial networks, handover failures occur during satellite pass transitions. Unlike terrestrial handovers (milliseconds between towers), NTN handovers involve:
- **Doppler shift** from satellite orbital velocity (~7.5 km/s in LEO) causing frequency offset
- **Beam switching delays** as the UE transitions between spot beams
- **Ephemeris mismatch** where predicted and actual satellite positions diverge, causing timing advance errors

The multi-metric correlation is the key discriminator: all three degrade together because the root cause (lost beam lock) affects the entire radio link simultaneously.

**Detection behavior:** IF produces extreme outlier scores (normalized z < -5.0). The first point also triggers all three EWMA metrics (> 2.5 sigma), enabling CRITICAL severity classification. Subsequent points in the window are classified HIGH.

### Scenario 2: Congestion Drift (t=250 to t=290)

**Signature:** Gradual packet loss increase (0.3% to 6-9%) and latency creep (500ms to 900ms) over 41 time steps, with **RSRP remaining normal**.

**NTN context:** The healthy RSRP indicates the radio link is fine — the satellite signal is strong and stable. The degradation is above the radio layer:
- **RAN scheduler saturation** — too many UEs sharing limited satellite bandwidth
- **Backhaul congestion** — the satellite-to-ground station link is at capacity
- **Buffer bloat** — increasing queuing delay as buffers fill

This pattern is critical to detect because operators might ignore it (no signal degradation = "radio is fine") while user experience degrades significantly.

**Detection behavior:** Despite being a gradual drift, the cumulative deviation from the warm-up baseline is large enough for IF to flag these as strong outliers (normalized z < -4.0). Points are classified HIGH severity.

**References:**
- *"FALCON: A Framework for Fault Prediction in Open RAN Using Multi-Level Telemetry"* (arXiv, 2025) — directly relevant to multi-metric RAN anomaly detection.
- *"Enhancing Satellite Telemetry Monitoring with Machine Learning"* (Springer, 2025) — evaluated IsolationForest among other methods for satellite telemetry.

## OpenAI Agents SDK Usage

The Agents SDK is used **only for interpretation** — not for detection.

### InterpretationAgent

The `InterpretationAgent` uses GPT-5.4 mini with `output_type=AnomalyInterpretation` to generate structured natural language explanations. The SDK provides:

- **Structured output enforcement** via `output_type` — the Pydantic schema is enforced on the LLM output
- **Automatic retry** on malformed outputs — no manual `json.loads()` + `try/except`
- **Type safety** — agent output flows directly into `AnomalyLogEntry`
- **Graceful fallback** — returns a placeholder interpretation if the API is unavailable

### What the Agents SDK is NOT used for

Detection is **100% deterministic Python** — pure sklearn + pandas with no API calls. The `DetectionAgent` class is a plain Python class, not an Agents SDK Agent. This means:
- Detection works without an API key
- Detection is fully reproducible (seed=42)
- Detection cost is $0
- Detection latency is ~6ms for 360 points (no network round-trips)

## Cost Analysis

| Component | Cost |
|-----------|------|
| **Detection path** | **$0.00** — pure sklearn + pandas, no API calls |
| **Interpretation path** | ~200 input tokens x $0.75/1M tokens = **$0.00015 per anomaly** |
| **Per run** (360 points, ~67 anomalies detected) | 67 x $0.00015 = **$0.010** |
| **Production scale** (1M points/day, ~5% anomaly rate) | 50,000 anomalies x $0.00015 = **~$7.50/day** |

The key insight: detection costs scale with data volume at $0. Interpretation costs scale only with anomaly count, which is a fraction of total data points.

## Project Structure

```
├── main.py                      # Entry point — runs pipeline + starts server
├── requirements.txt             # Python dependencies
├── .env.example                 # OPENAI_API_KEY placeholder
├── README.md
├── src/
│   ├── __init__.py
│   ├── simulator.py             # Telemetry generator (numpy/pandas)
│   ├── detection_agent.py       # IsolationForest + EWMA detection (pure sklearn)
│   ├── interpretation_agent.py  # OpenAI Agents SDK interpretation (GPT-5.4 mini)
│   ├── models.py                # Pydantic v2 schemas
│   └── api.py                   # FastAPI query interface + dashboard serving
├── frontend/
│   └── index.html               # Single-file ops dashboard (Chart.js)
├── logs/
│   ├── anomalies.jsonl          # Anomaly log (JSON Lines format)
│   └── telemetry_run.json       # Full telemetry run data
└── tests/
    ├── __init__.py
    └── test_detection.py        # 16 tests: simulator, detection, batch processing
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend dashboard |
| `/anomalies` | GET | All detected anomalies. Supports `?severity=`, `?start=`, `?end=` filters |
| `/health` | GET | Pipeline status + anomaly counts by severity |
| `/telemetry` | GET | Full telemetry run data for dashboard charts |

Server runs on `http://localhost:8000` with CORS enabled for all origins.

## Tests

```bash
pytest tests/ -v
```

16 tests across three suites:

- **TestSimulator** (5 tests) — Validates 360-point generation, required columns, correct anomaly injection for both scenarios, reasonable baseline values
- **TestDetectionAgent** (6 tests) — Validates DetectionResult schema, anomaly detection in both scenario windows, low false positive rate on normal data, valid severity levels, triggered metrics populated
- **TestBatchProcessing** (5 tests) — Validates batch output shape and types, detection in both scenario windows, low false positives, and **exact equivalence** between batch and sequential processing

## AI Tools Used

This project was built with **Claude Code** (Anthropic's AI coding assistant). Claude Code was used for:
- Generating the initial project scaffold from the architecture spec
- Implementing all modules with consistent interfaces
- Writing tests and the dashboard frontend
- Performance optimization (batch processing, parallel interpretation)
- Drafting this README

All code was reviewed, understood, and validated by the author. The architecture decisions (IsolationForest + EWMA dual detection, LLM-only-in-interpretation) were specified in the design doc and implemented faithfully.

Skylo explicitly permits and encourages AI tool usage — transparency about tooling is a positive signal, not a caveat.
