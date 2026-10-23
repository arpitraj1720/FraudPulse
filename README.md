# FraudPulse

> **Catch the Pattern, Not Just the Transaction**  
> *FinTech Real-Time Fraud & Sequence Detection Engine powered by Scikit-Learn, Stateful Behavioral Analysis, and Gemma AI.*

---

## 1. Problem Statement & Differentiation

Most existing fraud detection systems evaluate incoming financial transactions **in isolation**. If an adversary initiates a small payment ($5.00), it passes through friction-free. A few minutes later, another nominal payment ($10.00) clears unnoticed. Moments later, a massive transfer ($45,000.00) is routed to an unknown payee.

- **Traditional rules & single-row classifiers** see harmless micro-transactions and often flag high-value legitimate transfers (causing customer-friction false positives).
- **FraudPulse detects transaction sequences**: It combines a point-in-time ML classifier with a stateful behavioral sequence engine, escalating high-risk sequence anomalies to **Gemma** for structured, human-interpretable investigation.

```
Incoming Transaction 
        │
        ├──► [ 1. Scikit-Learn ML Classifier ] ──► Static Risk Probability (P_ml)
        │
        ├──► [ 2. Stateful Sequence Engine ]   ──► Behavioral Anomaly Signals (V_seq)
        │           (Velocity, Micro-probing,
        │            Payee Novelty, Ratio Burst)
        │
        ▼
   [ Hybrid Risk Aggregator ] ── (Threshold >= 0.65 or Critical Pattern)
        │
        ├──► Low Risk: Auto-Approve / Monitor
        │
        └──► Elevated Risk ──► [ 3. Gemma AI Investigator ]
                                      │
                                      ▼
                               • Threat Rationale
                               • Supporting Evidence
                               • Mitigating Counter-Evidence
                               • Recommended Action (e.g. Step-up 2FA)
```

---

## 2. Architecture & Division of Responsibilities

| Component | Responsibility | Technology | Why This Choice? |
| :--- | :--- | :--- | :--- |
| **ML Fraud Classifier** | Scores single-transaction risk using baseline transaction features (`step`, `type`, `log_amount`, cyclical `hour`). Deliberately excludes post-transaction balances to prevent target leakage. | Scikit-Learn (Random Forest), Joblib | Lightweight, reproducible, runs in milliseconds locally without GPU. |
| **Sequence & Behavior Engine** | Maintains sliding-window state per account (`window_size=10`). Flags rapid velocity, micro-probing trial transfers, and sudden deviations from moving averages. | Python / In-Memory State Cache | Deterministic, transparent, and captures multi-step attacks that tabular single-row ML misses. |
| **Hybrid Risk Scorer** | Combines ML risk probability and sequence signals into a unified risk index ($0.0 - 1.0$) with confidence tags (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`). | Mathematical Weighted Heuristic | Prevents double-counting correlated evidence while ensuring sequence triggers take precedence. |
| **Gemma AI Investigator** | Synthesizes observed signals into plain-English investigation briefs with counter-evidence analysis and step-up verification actions. Fallback parser included. | Google GenAI SDK (`gemma-2-9b-it`) / Structured Fallback | Provides explainability for risk officers without delegating deterministic policy authority to an LLM. |
| **Dashboard & API** | Real-time transaction simulation, scenario replay, risk breakdown radar, and interactive alerts. | FastAPI + Modern Web UI | Zero external database or broker dependencies; single-command local run. |

---

## 3. Dataset & Synthetic Mode

FraudPulse supports two training pipelines:

1. **Synthetic PaySim-Aligned Mode (Instant Hackathon Mode)**:
   - Built-in generator creating 25,000 transactions matching PaySim schemas and fraud distributions.
   - Requires **no external downloads**; trains in ~1 second.
2. **Kaggle PaySim Full Dataset**:
   - Download `PS_20174392719_1491204439457_log.csv` (~470MB) from [Kaggle PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1).
   - Place into `data/` or provide path via CLI.

> **Data Leakage Safeguard:** Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) and `isFlaggedFraud` are **strictly excluded** during training, as they represent information unavailable at the moment of payment authorization.

---

## 4. Quick Start & Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended on macOS Apple Silicon)
- Node.js 18+ (for frontend dashboard)

### Step 1: Environment Setup
```bash
# Clone and enter directory
cd FraudPulse

# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the template and supply your Google AI Studio key (optional; deterministic fallback active if unset):
```bash
cp .env.example .env
# Edit .env and insert GEMINI_API_KEY
```

### Step 3: Train the Classifier
Run the training pipeline (automatically uses synthetic mode if full PaySim CSV is not downloaded):
```bash
python train_model.py
```
*Outputs `fraudpulse_model.joblib` and `training_metrics.json`.*

To train on the downloaded Kaggle PaySim CSV:
```bash
python train_model.py data/PS_20174392719_1491204439457_log.csv
```

---

## 5. Model Evaluation Metrics

Evaluated on held-out test data (never downsampled, stratified 20% split):

```
Dataset: PaySim Aligned Distribution
Test Rows: 5,000 (150 fraud cases)
PR-AUC (Average Precision): ~0.667
ROC-AUC: ~0.993
Threshold 0.50: Recall: 100.0%, Precision: 70.8%, FPR: 1.28%
```
*Full threshold trade-offs saved in `training_metrics.json`.*

---

## 6. API Endpoints

Once the backend is launched (`python -m uvicorn backend.main:app --port 8000`):

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, model status, active accounts, and graph node/edge counts. |
| `GET` | `/api/metrics` | Model metrics, ROC-AUC, PR-AUC, Brier score, and latency benchmarks. |
| `GET` | `/api/graph/stats` | In-memory directed graph topology metrics (nodes, edges, flagged entities). |
| `POST` | `/api/score` | Evaluates a single transaction through ML + Sequence + Graph Engine with Tree SHAP. |
| `POST` | `/api/investigate` | Triggers Gemma 27B AI Copilot investigation on an escalated transaction. |
| `GET` | `/api/scenarios` | Pre-built test scenarios (Micro-probing, High-Value Normal, Burst, PaySim Fraud, Mule Loop). |
| `POST` | `/api/scenarios/replay/{id}` | Replays scenario sequence step-by-step to show live multi-tier detection (< 3ms). |
| `GET` | `/api/stream/live` | **Server-Sent Events (SSE)** continuous push of scored transactions & live metrics. |
| `GET` | `/api/stream/stats` | Real-time stream telemetry: Throughput (TPS), False Positives Prevented, Precision/Recall. |
| `GET` | `/api/stream/batch?size=5` | Pulls next batch of scored stream transactions (polling fallback for SSE). |
| `POST` | `/api/stream/toggle` | Starts or pauses background stream generator (`{"active": true, "tps": 10}`). |
| `POST` | `/api/stream/step` | Advances the stream simulator by N transactions for interactive stepping. |
| `POST` | `/api/stream/reset` | Resets stream counters, ground-truth confusion matrix, and sequence window. |

---

## 7. Hackathon Limitations & Safety Notice

1. **Not for Production Payment Gateways:** PaySim is synthetic mobile-money data. Thresholds and weights are uncalibrated for real-world UPI, ACH, or credit card networks.
2. **Advisory Gemma Role:** Gemma acts strictly as an investigative advisor. Real payment holds must always follow deterministic compliance policies.
3. **No Real Banking Connections:** No actual accounts or funds are touched.
