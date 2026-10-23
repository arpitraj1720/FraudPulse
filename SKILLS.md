# FraudPulse: Machine Learning & Behavioral Detection Skills

This document provides a comprehensive technical breakdown of the machine learning skills, behavioral sequence patterns, and decision mathematics used by **FraudPulse** to detect, classify, and explain financial transaction fraud.

---

## 1. Architectural Philosophy: The Three-Tier Defense

Single-row machine learning models fail in real-world payment networks because financial fraud is **inherently temporal and relational**. A $10,000 transfer in isolation might look suspicious or normal; whether it is fraudulent depends entirely on **what happened immediately before it** and **who is receiving the funds**.

FraudPulse operates across three distinct evaluation layers:

```
[ Incoming Transaction ]
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Point-in-Time Supervised ML (Random Forest)        │
│ Evaluates instantaneous transaction attributes & balances.  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Stateful Behavioral Sequence Engine                │
│ Evaluates sliding-window history, payee trust, & velocity.  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Autonomous AI Copilot (Gemma 27B)                  │
│ Evaluates multi-factor evidence & writes compliance SARs.   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Detection Skills & Feature Taxonomy

### Skill 1: Account Balance Drain & Liquidation Modeling
- **The Threat:** In Account Takeover (ATO) and unauthorized wire fraud, attackers attempt to extract the maximum possible capital before the victim notices or the bank locks the card.
- **Engineered Features:**
  - `balance_drain_ratio` $= \frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}$  
    Normal transactions typically utilize $2\%$ to $15\%$ of an account's liquidity. Fraudulent transfers exhibit a drain ratio approaching $1.00$ ($100\%$ account depletion).
  - `is_exact_wipeout` $= \mathbb{I}(|\text{oldbalanceOrg} - \text{amount}| < 1.0 \land \text{amount} > \$100)$  
    In the PaySim financial dataset, over $95\%$ of confirmed fraudulent transfers drain the exact balance down to $\$0.00$.

### Skill 2: Counterparty Entity Classification (Merchant vs. Peer)
- **The Threat:** Legitimate high-volume disbursements frequently flow to commercial accounts and utility merchants, while fraudulent exfiltrations flow to anonymous unverified individual accounts.
- **Engineered Features:**
  - `is_dest_merchant` $= \mathbb{I}(\text{nameDest starts with 'M'})$  
    In mobile money datasets, commercial merchant terminals have rigorous KYC requirements. Fraud transfers almost exclusively target consumer wallets (`C...`). Incorporating this feature eliminates thousands of false positives.

### Skill 3: Micro-Probing Sequence Recognition
- **The Threat:** Fraud syndicates test stolen credit card credentials or breached bank logins using nominal "probing" transfers ($\$1.00 - \$15.00$) to confirm validity and ensure no anti-fraud tripwires fire, followed minutes later by a catastrophic account drain ($\$40,000+$).
- **Engineered Logic:**
  - The Sequence Engine scans the account's rolling 10-transaction sliding window.
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
  - $\text{hour\_sin} = \sin(2\pi \cdot \frac{\text{step} \pmod{24}}{24})$
  - $\text{hour\_cos} = \cos(2\pi \cdot \frac{\text{step} \pmod{24}}{24})$
  - Preserves continuous cyclical distance across the midnight boundary ($23:59 \to 00:01$).

---

## 3. How FraudPulse Mathematically Defines Fraud

Fraud is not defined by a simple binary threshold. FraudPulse computes a **Hybrid Risk Score** ($S_{\text{final}} \in [0.0, 1.0]$) that dynamically blends supervised statistical risk with behavioral sequence signals:

### The Hybrid Scoring Formula

$$S_{\text{final}} = \begin{cases} 
\max(0.82, 0.20 \cdot P_{\text{ML}} + 0.80 \cdot S_{\text{seq}}), & \text{if micro-probing signature detected} \\
\min(0.38, 0.20 \cdot P_{\text{ML}} + 0.80 \cdot S_{\text{seq}}), & \text{if established payee trust verified} \\
\max(S_{\text{seq}}, 0.30 \cdot P_{\text{ML}} + 0.70 \cdot S_{\text{seq}}), & \text{if } S_{\text{seq}} \ge 0.50 \text{ (high velocity burst)} \\
0.40 \cdot P_{\text{ML}} + 0.60 \cdot S_{\text{seq}}, & \text{standard baseline blending}
\end{cases}$$

Where:
- $P_{\text{ML}}$: Anomaly probability predicted by the trained Random Forest classifier.
- $S_{\text{seq}}$: Cumulative risk score calculated by the Sequence Engine.

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

The supervised classifier was trained on the **6,362,620-row Kaggle PaySim dataset** and evaluated on an independent, stratified, held-out test split of **1,272,524 transactions** (containing **1,643 ground-truth fraud cases**):

```
Dataset: PaySim Real Financial Mobile-Money Log
Held-out Evaluation Split: 1,272,524 rows
Ground-Truth Fraud Prevalence: 0.1291%
```

| Evaluation Metric | Score | Industry Benchmark | Status |
|---|---|---|---|
| **Average Precision (PR-AUC)** | **0.9976** | $> 0.80$ | State of the art |
| **ROC-AUC** | **0.9995** | $> 0.95$ | Near-perfect separation |
| **Precision @ 0.50 Threshold** | **99.76%** | $> 90\%$ | Excellent |
| **Recall @ 0.50 Threshold** | **99.76%** | $> 95\%$ | 1,639 / 1,643 cases caught |
| **False Positives @ 0.50 Threshold** | **4** | $< 500$ | **99.99% false alarm reduction** |
| **False Positive Rate (FPR)** | **0.0003%** | $< 0.1\%$ | Minimal business disruption |

---

## 6. Data Leakage & Regulatory Compliance Safeguards

To ensure realistic, production-grade validity, FraudPulse strictly adheres to the following safeguards:
1. **Pre-Authorization Features Only:** Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) and regulatory flags (`isFlaggedFraud`) are strictly barred from training and inference to prevent artificial data leakage.
2. **Deterministic Fallback Resiliency:** If third-party AI APIs experience rate limits or downtime, Layer 2 deterministic heuristics seamlessly generate explainable threat rationales without halting payment authorizations.
3. **Sub-50ms In-Line SLA:** GenAI inference is completely decoupled from payment authorization, ensuring real-time gateway throughput exceeds $1,000$ transactions per second.
