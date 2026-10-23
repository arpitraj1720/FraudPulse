"""Pre-configured demo scenarios for FraudPulse.

Showcases the power of sequence analysis over single-transaction scoring:
1. Micro-Probing Attack (Proves sequence detection and Gemma investigation)
2. Legitimate High-Value Purchase (Proves false-positive mitigation)
3. Rapid Velocity Card Burst (Proves multi-recipient velocity tracking)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "micro_probing_attack",
        "title": "Micro-Probing Account Drain",
        "badge": "Signature Sequence Attack",
        "badge_color": "danger",
        "description": "Adversary performs two small trial transfers to verify recipient validity and bypass friction, followed by a sudden $42,500 exfiltration.",
        "account_id": "ACC-9824",
        "expected_verdict": "CRITICAL — Sequence Engine Flags Probing Pattern",
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 4.50,
                "recipient": "Merchant_Coffee",
                "label": "Small merchant payment",
                "delay_ms": 600,
            },
            {
                "step": 2,
                "type": "TRANSFER",
                "amount": 12.00,
                "recipient": "Wallet_99x_Offshore",
                "label": "Nominal trial transfer to new wallet",
                "delay_ms": 800,
            },
            {
                "step": 3,
                "type": "TRANSFER",
                "amount": 42500.00,
                "recipient": "Wallet_99x_Offshore",
                "label": "Sudden massive transfer to same unverified wallet",
                "delay_ms": 1000,
            },
        ],
    },
    {
        "id": "legitimate_high_value",
        "title": "Legitimate High-Value Transfer",
        "badge": "False-Positive Prevention",
        "badge_color": "success",
        "description": "User makes a legitimate $35,000 quarterly lease payment to an established recipient. Shows how sequence context prevents false friction.",
        "account_id": "ACC-1044",
        "expected_verdict": "LOW / CLEAR — Legitimate Sequence Profile",
        "transactions": [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 18500.00,
                "recipient": "Apex_Commercial_Properties",
                "label": "Quarterly commercial installment 1",
                "delay_ms": 500,
            },
            {
                "step": 2,
                "type": "TRANSFER",
                "amount": 22000.00,
                "recipient": "Apex_Commercial_Properties",
                "label": "Quarterly commercial installment 2",
                "delay_ms": 600,
            },
            {
                "step": 3,
                "type": "TRANSFER",
                "amount": 35000.00,
                "recipient": "Apex_Commercial_Properties",
                "label": "Quarterly commercial lease settlement",
                "delay_ms": 700,
            },
        ],
    },
    {
        "id": "rapid_velocity_burst",
        "title": "Rapid Velocity Card Burst",
        "badge": "Velocity & Account Takeover",
        "badge_color": "warning",
        "description": "5 rapid-fire transfers in under 2 minutes to distinct new recipient wallets, indicating automated script or compromised credentials.",
        "account_id": "ACC-5521",
        "expected_verdict": "HIGH — Velocity Anomaly & Novel Payee Cascade",
        "transactions": [
            {"step": 1, "type": "TRANSFER", "amount": 850.00, "recipient": "Wallet_Alpha", "label": "Rapid transfer 1", "delay_ms": 400},
            {"step": 2, "type": "TRANSFER", "amount": 920.00, "recipient": "Wallet_Bravo", "label": "Rapid transfer 2", "delay_ms": 400},
            {"step": 3, "type": "TRANSFER", "amount": 890.00, "recipient": "Wallet_Charlie", "label": "Rapid transfer 3", "delay_ms": 400},
            {"step": 4, "type": "TRANSFER", "amount": 950.00, "recipient": "Wallet_Delta", "label": "Rapid transfer 4", "delay_ms": 400},
            {"step": 5, "type": "TRANSFER", "amount": 1100.00, "recipient": "Wallet_Echo", "label": "Rapid transfer 5", "delay_ms": 400},
        ],
    },
    {
        "id": "real_paysim_fraud",
        "title": "Real PaySim Fraud Case (C1305486145)",
        "badge": "Kaggle Ground Truth",
        "badge_color": "danger",
        "description": "Actual fraudulent transaction sequence extracted directly from the 6.36M Kaggle PaySim log (Step 1 TRANSFER & immediate CASH_OUT drain).",
        "account_id": "C1305486145",
        "expected_verdict": "CRITICAL — Ground Truth Fraud Log Match",
        "transactions": [
            {"step": 1, "type": "PAYMENT", "amount": 181.00, "recipient": "M1979787155", "label": "Prior merchant purchase in PaySim", "delay_ms": 500},
            {"step": 1, "type": "TRANSFER", "amount": 181.00, "recipient": "C553264065", "label": "PaySim Fraud Log: TRANSFER (Row 2, isFraud=1)", "delay_ms": 700},
            {"step": 1, "type": "CASH_OUT", "amount": 181.00, "recipient": "C38997010", "label": "PaySim Fraud Log: Immediate CASH_OUT (Row 3, isFraud=1)", "delay_ms": 800},
        ],
    },
]


def get_scenario(scenario_id: str) -> Dict[str, Any]:
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    raise ValueError(f"Scenario '{scenario_id}' not found.")
