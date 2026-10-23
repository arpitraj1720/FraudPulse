# FraudPulse: Machine Learning & Behavioral Detection Skills

This document provides a comprehensive technical breakdown of the machine learning skills, behavioral sequence patterns, graph topology engine, explainability mechanisms, and decision mathematics used by **FraudPulse** to detect, classify, and explain financial transaction fraud.

---

## 1. Architectural Philosophy: The Dual-Tier Defense

Single-row machine learning models fail in real-world payment networks because financial fraud is **inherently temporal, sequential, and relational**. A $10,000 transfer in isolation might look completely normal; whether it is fraudulent depends entirely on **what preceded it**, **how fast it happened**, and **who is receiving the funds in the wider transaction network**.

FraudPulse operates a strict **Dual-Tier architecture**:

```
[ Incoming Payment Authorization Request ]
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: REAL-TIME SYNCHRONOUS INLINE GATEWAY (< 3ms SLA)               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Vectorized NumPy Preprocessor (make_features_fast: ~3 µs)           │
│ 2. LightGBM GBDT Point Anomaly Classifier (C++ Booster: 0.22 ms)       │
│ 3. Native Tree SHAP Micro-Attributions (Pred Contrib: < 0.1 ms)        │
│ 4. Stateful Sliding-Window Sequence Engine (Velocity, Acceleration)    │
│ 5. In-Memory Graph Network Topology Engine (Fan-In/Out, Loops, Hops)   │
│ 6. Hybrid Risk Blending Formula & Policy Triage Matrix                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          │                                                   │
  Risk < 0.50 (LOW / MED)                             Risk >= 0.50 (HIGH / CRITICAL)
          │                                                   │
          ▼                                                   ▼
┌──────────────────┐                               ┌─────────────────────────────────┐
│  AUTO-APPROVE /  │                               │ TIER 2: ASYNC AI COPILOT        │
│  FLAG & MONITOR  │                               ├─────────────────────────────────┤
│ (Instant Permit) │                               │ Gemma 27B Multimodal Copilot    │
└──────────────────┘                               │ - Synthesizes 4-pillar evidence │
                                                   │ - Generates Counter-Evidence    │
                                                   │ - Auto-files Regulatory SARs    │
                                                   └─────────────────────────────────┘
```

---

## 2. Core Detection Skills & Feature Taxonomy

### Skill 1: Account Balance Drain & Liquidation Modeling
- **The Threat:** In Account Takeover (ATO) and unauthorized wire fraud, attackers attempt to extract the maximum possible capital before the victim notices or the bank locks the card.
- **Engineered Features:**
  - `balance_drain_ratio` $= \min\left(10.0, \frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}\right)$  
    Normal transactions typically utilize $2\%$ to $15\%$ of an account's liquidity. Fraudulent transfers exhibit a drain ratio approaching $1.00$ ($100\%$ account depletion).
  - `is_exact_wipeout` $= \mathbb{I}(|\text{oldbalanceOrg} - \text{amount}| < 1.0 \land \text{amount} > \$100)$  
    In the PaySim financial dataset, over $95\%$ of confirmed fraudulent transfers drain the exact balance down to $\$0.00$.

### Skill 2: Counterparty Entity Classification (Merchant vs. Peer)
- **The Threat:** Legitimate high-volume disbursements frequently flow to commercial accounts and utility merchants, while fraudulent exfiltrations flow to anonymous unverified individual accounts.
- **Engineered Features:**
  - `is_dest_merchant` $= \mathbb{I}(\text{nameDest starts with 'M' or contains 'merchant'})$  
    Commercial merchant terminals have rigorous KYC and verification requirements. Fraud transfers almost exclusively target consumer wallets (`C...`). Incorporating this feature eliminates thousands of false positives.

### Skill 3: Micro-Probing Sequence Recognition
- **The Threat:** Fraud syndicates test stolen credit card credentials or breached bank logins using nominal "probing" transfers ($\$1.00 - \$15.00$) to confirm validity and ensure no anti-fraud tripwires fire, followed minutes later by a catastrophic account drain ($\$40,000+$).
- **Engineered Logic:**
  - The Sequence Engine scans the account's rolling sliding window.
  - Flags `MICRO_PROBING_SEQUENCE` if $\ge 1$ transaction under $\$25.00$ occurs within 15 minutes of a transaction exceeding $\$5,000.00$ whose amount is $\ge 10\times$ the account's historical average.
  - Automatically elevates sequence risk score to $\ge 0.82$.

### Skill 4: High-Velocity Rapid Card Burst Detection
- **The Threat:** Automated bot scripts and credential stuffers submit rapid bursts of transactions across multiple distinct wallets within seconds.
- **Engineered Features:**
  - `tx_count_5m` & `tx_count_1h`: Sliding-window transaction counts.
  - `time_since_last_tx`: Temporal delta in seconds between sequential events.
  - `velocity_acceleration`: First derivative of transaction frequency ($\frac{\Delta \text{tx\_rate}}{\Delta t}$).
  - Triggers `HIGH_VELOCITY` when $\ge 4$ events occur within an hour, and `RAPID_SUCCESSION` when sequential transfers occur in $< 30$ seconds.

### Skill 5: Counterparty Novelty vs. Established Payee Trust Discounting
- **The Threat:** Traditional ML generates massive false alarm rates on legitimate high-value payments (e.g. paying quarterly rent or vendor invoices).
- **Engineered Logic:**
  - **Payee Novelty:** If `recipient` has never appeared in the account's verified history, a `+0.20` novelty penalty is applied.
  - **Payee Trust Discount:** If the account has transacted with the recipient $\ge 2$ times previously, an established payee trust discount (`-0.20`) is applied, capping the final risk at $\le 0.38$ to prevent legitimate user friction.

### Skill 6: Cyclical & Temporal Attack Modeling
- **The Threat:** Fraud rings intentionally conduct automated exfiltrations during nocturnal hours (2 AM – 5 AM) when account holders are asleep and fraud call centers operate with reduced staffing.
- **Engineered Features:**
  - $\text{hour\_sin} = \sin\left(2\pi \cdot \frac{\text{step} \pmod{24}}{24}\right)$
  - $\text{hour\_cos} = \cos\left(2\pi \cdot \frac{\text{step} \pmod{24}}{24}\right)$
  - Preserves continuous cyclical distance across the midnight boundary ($23:59 \to 00:01$).

### Skill 7: In-Memory Graph Network Topology (TigerGraph Style)
- **The Threat:** Organized crime syndicates, money mules, and wash traders do not act in isolation. They bounce illicit funds through intermediary nodes and funnel money into offshore cash-out hubs. Traditional row-level ML is completely blind to graph topology.
- **Engineered Graph Engine (`backend/graph_engine.py`):**
  - **In-Degree (Fan-In Mule Hub Spikes):** Detects when an unverified recipient wallet suddenly receives inflows from multiple distinct originators ($\ge 2$ unlinked senders). Flags `MULE_FAN_IN_SPIKE`.
  - **Out-Degree (Fan-Out Smurfing / Dispersion):** Detects when a compromised account disperses funds across $\ge 3$ distinct destination wallets. Flags `SMURFING_FAN_OUT`.
  - **Directed Circular Cycle Detection:** Uses NetworkX path finding to detect when money flows through a closed loop ($A \to B \to C \to A$) to layer funds and launder audit trails. Flags `CIRCULAR_WASH_CYCLE` (+0.40 risk contribution).
  - **Graph Contagion / Degrees of Separation:** Computes shortest-path distance to known blacklisted AML/fraud nodes (`Wallet_99x_Offshore`, `C553264065`). Distance 0 is a direct hit (`DIRECT_FLAGGED_DESTINATION`, +0.50 risk); Distance 1 represents 1-hop proximity (`PROXIMITY_FLAGGED_1HOP`, +0.30 risk).

### Skill 8: Native Tree SHAP Local Explainability
- **The Threat:** Black-box ML models are rejected by FinTech regulators (CFPB, FDIC, FCA) because they cannot provide adverse action explanations.
- **Engineered Engine:**
  - Evaluates LightGBM's native C++ `model.booster_.predict(X, pred_contrib=True)` in **< 0.1ms** without the heavy overhead of external libraries.
  - Decomposes the point prediction into exact feature marginal contributions:
    - **`+RISK` Factors:** Features pushing score towards fraud (e.g. `is_exact_wipeout: +15.73`, `balance_drain_ratio: +4.82`).
    - **`-SAFE` Factors:** Features reducing risk (e.g. `is_dest_merchant: -2.31`, `log_amount: -0.37`).
  - Passed directly into Gemma 27B's context to provide mathematical proof in suspicious activity reports.

### Skill 9: Vectorized Pure-NumPy Preprocessing Pipeline
- **The Engineering:** Python Pandas DataFrame construction creates significant heap allocation and indexing overhead (~2.0 - 3.5ms per transaction).
- **Optimization:**
  - `make_features_fast()` computes the 14 float32 inputs using pure scalar math and allocates a contiguous `(1, 14)` NumPy buffer in **~3 microseconds**.
  - Feeds directly into the compiled C++ LightGBM booster, slashing single transaction scoring time to **0.22ms**.

---

## 3. How FraudPulse Mathematically Defines Fraud

Fraud is not defined by a simple binary threshold. FraudPulse computes a **Hybrid Risk Score** ($S_{\text{final}} \in [0.0, 1.0]$) that dynamically blends supervised statistical risk, behavioral sequence signals, and graph topology:

### The Hybrid Scoring Formula

$$S_{\text{final}} = \begin{cases} 
\max(0.92, 0.20 \cdot P_{\text{ML}} + 0.80 \cdot S_{\text{seq}}), & \text{if circular cycle or direct AML blacklist hit detected} \\
\max(0.82, 0.20 \cdot P_{\text{ML}} + 0.80 \cdot S_{\text{seq}}), & \text{if micro-probing signature detected} \\
\max(0.80, 0.25 \cdot P_{\text{ML}} + 0.75 \cdot S_{\text{seq}}), & \text{if mule fan-in funnel spike detected} \\
\min(0.38, 0.20 \cdot P_{\text{ML}} + 0.80 \cdot S_{\text{seq}}), & \text{if established payee trust verified} \\
\max(S_{\text{seq}}, 0.30 \cdot P_{\text{ML}} + 0.70 \cdot S_{\text{seq}}), & \text{if } S_{\text{seq}} \ge 0.50 \text{ (high velocity burst)} \\
0.40 \cdot P_{\text{ML}} + 0.60 \cdot S_{\text{seq}}, & \text{standard baseline blending}
\end{cases}$$

Where:
- $P_{\text{ML}}$: Anomaly probability predicted by the trained LightGBM GBDT classifier with native Tree SHAP attributions.
- $S_{\text{seq}}$: Cumulative risk score calculated by the Sequence Engine, incorporating temporal features and graph metrics ($S_{\text{seq}} = \min(1.0, S_{\text{base}} + \Delta S_{\text{graph}})$).

---

## 4. Policy Triage & Action Decision Matrix

| Risk Tier | Score Range | Policy Decision | Gateway Action | AI Investigation |
|---|---|---|---|---|
| **CRITICAL** | $0.75 - 1.00$ | `HOLD_INVESTIGATE` | Immediate transaction block; freeze destination routing. | **Tier 2 Gemma 27B** auto-synthesizes SAR incident brief. |
| **HIGH** | $0.50 - 0.74$ | `STEP_UP_VERIFY` | Challenge user with step-up biometric / hardware 2FA. | **Tier 2 Gemma 27B** auto-synthesizes threat rationale. |
| **MEDIUM** | $0.25 - 0.49$ | `FLAG_MONITOR` | Authorize transaction; log telemetry for asynchronous review. | Offline queuing / logging. |
| **LOW** | $0.00 - 0.24$ | `AUTO_APPROVE` | Instant sub-50ms clearance without user friction. | Bypassed (Zero LLM overhead). |

---

## 5. Model Training & Evaluation Metrics

The supervised classifier uses **LightGBM Gradient Boosted Decision Trees (GBDT)** with cost-sensitive class balancing, trained on the **6,362,620-row Kaggle PaySim dataset** and evaluated on an independent, stratified, held-out test split of **1,272,524 transactions** (containing **1,643 ground-truth fraud cases**):

```
Dataset: PaySim Real Financial Mobile-Money Log
Model Architecture: LightGBM GBDT (150 estimators, max_depth=8, num_leaves=31)
Training Rows: 306,570 (balanced negative sampling + 100% of train fraud cases)
Held-out Evaluation Split: 1,272,524 rows (unaltered real-world distribution)
Ground-Truth Fraud Prevalence: 0.1291% (1 in 774 transactions)
Inference Engine: Native LightGBM C++ Booster + Vectorized Pure-NumPy Pipeline
```

### Full Evaluation Metrics Across Operating Thresholds

| Threshold ($\tau$) | Precision | Recall | F1 Score | True Positives | False Positives | False Negatives | False Positive Rate (FPR) |
|---|---|---|---|---|---|---|---|
| **0.50 (Balanced)** | **87.00%** | **99.76%** | **0.9294** | **1,639** | 245 | 4 | **0.0193%** (1 in 5,186) |
| **0.25 (Aggressive)** | 74.40% | 99.76% | 0.8523 | 1,639 | 564 | 4 | 0.0444% (1 in 2,252) |
| **0.10 (High Sensitivity)** | 53.63% | 99.76% | 0.6976 | 1,639 | 1,417 | 4 | 0.1115% (1 in 896) |

### Global Model Performance & Probability Calibration

| Evaluation Metric | Score | Industry Benchmark | Status |
|---|---|---|---|
| **Average Precision (PR-AUC)** | **0.9976** | $> 0.80$ | State of the art |
| **ROC-AUC** | **0.9997** | $> 0.95$ | Near-perfect separation |
| **Brier Score Calibration Loss** | **0.000683** | $< 0.05$ | Well-calibrated probabilities |
| **C++ Booster Prediction Time** | **0.22ms** | $< 5.0ms$ | Sub-millisecond pre-auth SLA |
| **Full Gateway Step Replay** | **1.92ms - 2.72ms** | $< 50ms$ | Zero authorization drag |

### Feature Importance Ranking (Tree Split & Gain):
1. **`log_amount` (30.6%):** Non-linear transaction magnitude scaling.
2. **`step` (19.6%):** Simulation timeframe and account age progression.
3. **`log_oldbalance` (14.0%):** Available pre-transaction liquidity context.
4. **`hour_cos` & `hour_sin` (16.2%):** Nocturnal vs. business-hour cyclical attack timing.
5. **`balance_drain_ratio` & `is_exact_wipeout` (12.4%):** Full-balance account depletion indicators.

---

## 6. Pre-Built Demonstration Scenarios

FraudPulse includes 5 production test scenarios in [`backend/scenarios.py`](file:///Users/arpitraj/Desktop/FraudPulse/backend/scenarios.py) to demonstrate the multi-tier defense:

1. **`micro_probing_attack` (ATO Account Takeover):**
   - Step 1: $12.00 test transfer to establish link.
   - Step 2: $8.50 micro-test transfer.
   - Step 3: $42,500.00 full account drain to unverified wallet.
   - *Result:* Caught by Skill 3 (Micro-probing sequence recognition) + Skill 1 (Exact wipeout). Risk: `1.00 (CRITICAL)`. Escalates to Gemma Copilot.
2. **`legitimate_high_value` (Quarterly Commercial Supplier Invoice):**
   - Step 1: $1,200.00 to merchant `M_Acme_Supplies`.
   - Step 2: $1,450.00 to merchant `M_Acme_Supplies`.
   - Step 3: $85,000.00 large quarterly invoice to same verified merchant.
   - *Result:* Cleared by Skill 2 (Merchant entity) + Skill 5 (Established payee trust discount). Risk: `0.04 (LOW)`. **Auto-approved with zero user friction.**
3. **`rapid_velocity_burst` (Credential Stuffing / Bot Exfiltration):**
   - Rapid-fire transactions across 4 different destination wallets in under 45 seconds.
   - *Result:* Caught by Skill 4 (Velocity acceleration) + Skill 7 (Smurfing fan-out dispersion). Risk: `0.895 (CRITICAL)`.
4. **`real_paysim_fraud` (Ground-Truth Kaggle Dataset Replay):**
   - Replays genuine ground-truth fraud row from the Kaggle dataset (`amount: $181,386.00` draining account with matching balance).
   - *Result:* Caught by Skill 1 (Exact wipeout Tree SHAP: `+15.73`). Risk: `0.92 (CRITICAL)`.
5. **`circular_mule_ring` (AML Wash-Trading Network Loop):**
   - Closed money movement cycle: Account A $\to$ Wallet B $\to$ Wallet C $\to$ Account A.
   - *Result:* Caught by Skill 7 (Graph circular loop path finder). Risk: `0.92 (CRITICAL)`.

---

## 7. Data Leakage & Regulatory Compliance Safeguards

To ensure realistic, production-grade validity, FraudPulse strictly adheres to the following safeguards:
1. **Pre-Authorization Features Only:** Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) and regulatory flags (`isFlaggedFraud`) are strictly barred from training and inference to prevent artificial data leakage.
2. **Deterministic Fallback Resiliency:** If third-party AI APIs experience rate limits or network downtime, Layer 2 deterministic heuristics seamlessly generate explainable threat rationales with Tree SHAP risk drivers without halting payment authorizations.
3. **Sub-3ms Gateway SLA:** GenAI inference is completely decoupled from payment authorization, ensuring real-time gateway throughput exceeds $1,000$ transactions per second.
