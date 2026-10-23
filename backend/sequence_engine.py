"""Stateful Behavioral and Sequence Detection Engine for FraudPulse.

Maintains sliding-window transaction history per account and computes dynamic
sequence risk signals (micro-probing, velocity bursts, recipient novelty, ratio jumps).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.graph_engine import graph_risk_engine


@dataclass
class TransactionRecord:
    timestamp: float
    amount: float
    tx_type: str
    recipient: str
    step: int = 1


class SequenceEngine:
    def __init__(self, window_size: int = 10, time_window_seconds: float = 3600.0):
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds
        # Mapping: account_id -> deque of recent TransactionRecord
        self.history: Dict[str, deque[TransactionRecord]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # Mapping: account_id -> set of known recipients
        self.known_recipients: Dict[str, set[str]] = defaultdict(set)

    def reset(self, account_id: Optional[str] = None) -> None:
        """Reset sequence history for a specific account or all accounts."""
        if account_id:
            if account_id in self.history:
                del self.history[account_id]
            if account_id in self.known_recipients:
                del self.known_recipients[account_id]
        else:
            self.history.clear()
            self.known_recipients.clear()
            graph_risk_engine.reset()

    def seed_account(self, account_id: str, records: List[Dict[str, Any]]) -> None:
        """Seed account history for demo scenarios."""
        self.reset(account_id)
        for r in records:
            tx = TransactionRecord(
                timestamp=r.get("timestamp", time.time()),
                amount=float(r.get("amount", 0.0)),
                tx_type=str(r.get("type", "PAYMENT")),
                recipient=str(r.get("recipient", "UNKNOWN")),
                step=int(r.get("step", 1)),
            )
            self.history[account_id].append(tx)
            self.known_recipients[account_id].add(tx.recipient)

    def evaluate_transaction(
        self,
        account_id: str,
        amount: float,
        tx_type: str,
        recipient: str,
        step: int = 1,
        tx_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate an incoming transaction against prior sequence history."""
        now = tx_timestamp if tx_timestamp is not None else time.time()
        recent = list(self.history[account_id])
        signals: List[Dict[str, Any]] = []
        sequence_risk = 0.0

        # Feature 1: Recipient Novelty vs Established Payee
        is_novel_recipient = recipient not in self.known_recipients[account_id] and len(recent) > 0
        is_established_payee = recipient in self.known_recipients[account_id] and len(recent) >= 2
        if is_novel_recipient:
            signals.append({
                "code": "NOVEL_RECIPIENT",
                "severity": "MEDIUM",
                "title": "First-Time Payee",
                "description": f"Account has never transacted with recipient '{recipient}' before.",
                "risk_contribution": 0.20,
            })
            sequence_risk += 0.20
        elif is_established_payee:
            signals.append({
                "code": "ESTABLISHED_PAYEE",
                "severity": "LOW",
                "title": "Verified Payee Trust",
                "description": f"Account has verified relationship with recipient '{recipient}'.",
                "risk_contribution": -0.20,
            })

        # Feature 2: Transaction Velocity
        recent_timestamps = [t.timestamp for t in recent if (now - t.timestamp) <= self.time_window_seconds]
        tx_count_in_window = len(recent_timestamps) + 1  # including current
        if tx_count_in_window >= 4:
            signals.append({
                "code": "HIGH_VELOCITY",
                "severity": "HIGH",
                "title": "Velocity Spike",
                "description": f"{tx_count_in_window} transactions within the recent time window.",
                "risk_contribution": 0.30,
            })
            sequence_risk += 0.30
        elif tx_count_in_window >= 2:
            sequence_risk += 0.05

        # Feature 3: Micro-Probing Followed by Major Exfiltration (The Signature Sequence)
        # Pattern: Previous 1-3 transactions were nominal (< $100), current transaction is significantly larger (> $5,000)
        # and represents a huge ratio jump.
        is_micro_probing = False
        if len(recent) >= 2:
            prev_amounts = [t.amount for t in recent[-3:]]
            avg_prev = sum(prev_amounts) / len(prev_amounts)
            if all(a < 500.0 for a in prev_amounts) and amount > 5000.0:
                jump_multiplier = amount / max(avg_prev, 1.0)
                if jump_multiplier >= 10.0:
                    is_micro_probing = True
                    signals.append({
                        "code": "MICRO_PROBING_SEQUENCE",
                        "severity": "CRITICAL",
                        "title": "Micro-Probing Sequence Detected",
                        "description": (
                            f"Current amount (${amount:,.2f}) is {jump_multiplier:.1f}x higher than recent "
                            f"nominal trial transactions (avg ${avg_prev:,.2f}). Classic fraud preparation signature."
                        ),
                        "risk_contribution": 0.65,
                    })
                    sequence_risk += 0.65

        # Feature 4: Historical Amount Deviation
        if len(recent) >= 2 and not is_micro_probing:
            all_prev_amounts = [t.amount for t in recent]
            avg_all = sum(all_prev_amounts) / len(all_prev_amounts)
            if amount > 5 * avg_all and amount > 2000.0:
                signals.append({
                    "code": "AMOUNT_DEVIATION",
                    "severity": "MEDIUM",
                    "title": "Abnormal Amount Deviation",
                    "description": f"Transaction is {amount / avg_all:.1f}x higher than account 10-tx baseline (${avg_all:,.2f}).",
                    "risk_contribution": 0.25,
                })
                sequence_risk += 0.25

        # Feature 5: Transfer / Cash-Out Sequence Affinity
        if tx_type in ("TRANSFER", "CASH_OUT") and sequence_risk > 0.15:
            sequence_risk += 0.15

        # If micro-probing is confirmed, set a high floor on sequence risk
        if is_micro_probing:
            sequence_risk = max(sequence_risk, 0.85)

        # 12 Engineered Behavioral Features (Aligned with Industry Standards)
        avg_amount = sum(t.amount for t in recent) / len(recent) if recent else amount
        amount_vs_avg_ratio = round(amount / max(avg_amount, 1.0), 2)
        tx_count_5m = sum(1 for t in recent if (now - t.timestamp) <= 300) + 1
        tx_count_1h = sum(1 for t in recent if (now - t.timestamp) <= self.time_window_seconds) + 1
        amount_sum_1h = round(sum(t.amount for t in recent if (now - t.timestamp) <= self.time_window_seconds) + amount, 2)
        time_since_last_tx = round(now - recent[-1].timestamp, 1) if recent else 0.0

        recent_10m_recips = set(t.recipient for t in recent if (now - t.timestamp) <= 600)
        recent_10m_recips.add(recipient)
        unique_recipients_10m = len(recent_10m_recips)
        new_recipient_flag = 1 if is_novel_recipient else 0
        recipient_tx_count_24h = sum(1 for t in recent if t.recipient == recipient)
        balance_drain_ratio = round(amount / max(amount_sum_1h, 1.0), 3)
        rapid_small_tx_count = sum(1 for t in recent if t.amount < 50.0 and (now - t.timestamp) <= self.time_window_seconds)
        max_prior_amount = max([t.amount for t in recent], default=amount)
        amount_escalation_ratio = round(amount / max(max_prior_amount, 1.0), 2)
        micro_tx_before_large_tx = bool(is_micro_probing)

        behavioral_features = {
            "amount_vs_avg_ratio": amount_vs_avg_ratio,
            "tx_count_5m": tx_count_5m,
            "tx_count_1h": tx_count_1h,
            "amount_sum_1h": amount_sum_1h,
            "time_since_last_tx": time_since_last_tx,
            "unique_recipients_10m": unique_recipients_10m,
            "new_recipient": new_recipient_flag,
            "recipient_tx_count_24h": recipient_tx_count_24h,
            "balance_drain_ratio": balance_drain_ratio,
            "rapid_small_tx_count": rapid_small_tx_count,
            "amount_escalation_ratio": amount_escalation_ratio,
            "micro_tx_before_large_tx": micro_tx_before_large_tx,
        }

        # Feature 6: Graph Topology & Counterparty Network Analysis
        graph_eval = graph_risk_engine.evaluate_topology(
            sender=account_id,
            recipient=recipient,
            amount=amount,
            step=step,
            tx_timestamp=now,
        )
        if graph_eval["signals"]:
            signals.extend(graph_eval["signals"])
            sequence_risk += graph_eval["graph_risk_score"]

        # Merge graph metrics into behavioral features
        for k, v in graph_eval["metrics"].items():
            behavioral_features[f"graph_{k}"] = v

        # Commit transaction to state
        current_tx = TransactionRecord(
            timestamp=now,
            amount=amount,
            tx_type=tx_type,
            recipient=recipient,
            step=step,
        )
        self.history[account_id].append(current_tx)
        self.known_recipients[account_id].add(recipient)

        return {
            "account_id": account_id,
            "sequence_risk_score": round(sequence_risk, 4),
            "recent_tx_count": len(recent),
            "signals": signals,
            "is_micro_probing": is_micro_probing,
            "is_novel_recipient": is_novel_recipient,
            "is_established_payee": is_established_payee,
            "graph_metrics": graph_eval["metrics"],
            "graph_risk_score": graph_eval["graph_risk_score"],
            "behavioral_features": behavioral_features,
            "history_summary": [
                {
                    "step": t.step,
                    "type": t.tx_type,
                    "amount": t.amount,
                    "recipient": t.recipient,
                    "timestamp": t.timestamp,
                }
                for t in list(self.history[account_id])
            ],
        }


def compute_hybrid_risk(ml_prob: float, sequence_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Combines static ML probability with dynamic sequence behavioral risk and graph topology."""
    seq_score = sequence_eval.get("sequence_risk_score", 0.0)
    signals = sequence_eval.get("signals", [])

    # Graph-specific critical overrides
    has_critical_graph_signal = any(
        s.get("code") in ("CIRCULAR_WASH_CYCLE", "DIRECT_FLAGGED_DESTINATION")
        for s in signals
    )
    has_mule_fan_in = any(s.get("code") == "MULE_FAN_IN_SPIKE" for s in signals)

    # If critical micro-probing pattern is flagged, give sequence engine strong dominance
    if sequence_eval.get("is_micro_probing"):
        combined_score = max(0.82, 0.20 * ml_prob + 0.80 * seq_score)
    elif has_critical_graph_signal:
        combined_score = max(0.92, 0.20 * ml_prob + 0.80 * seq_score)
    elif has_mule_fan_in:
        combined_score = max(0.80, 0.25 * ml_prob + 0.75 * seq_score)
    elif sequence_eval.get("is_established_payee"):
        # Verified recipient trust discounts static ML false alarms
        combined_score = min(0.38, 0.20 * ml_prob + 0.80 * seq_score)
    elif seq_score >= 0.50:
        # High velocity / multi-payee behavioral cascade dominates
        combined_score = max(seq_score, 0.30 * ml_prob + 0.70 * seq_score)
    else:
        combined_score = 0.40 * ml_prob + 0.60 * seq_score

    combined_score = min(1.0, max(0.0, combined_score))

    # Triage decision bands
    if combined_score >= 0.75:
        tier = "CRITICAL"
        decision = "HOLD_INVESTIGATE"
        action_label = "Block & Escalate to Fraud Desk"
    elif combined_score >= 0.50:
        tier = "HIGH"
        decision = "STEP_UP_VERIFY"
        action_label = "Trigger Step-Up Biometric / 2FA"
    elif combined_score >= 0.25:
        tier = "MEDIUM"
        decision = "FLAG_MONITOR"
        action_label = "Approve with Active Monitoring"
    else:
        tier = "LOW"
        decision = "AUTO_APPROVE"
        action_label = "Instant Clearance"

    return {
        "final_risk_score": round(combined_score, 4),
        "risk_tier": tier,
        "policy_decision": decision,
        "recommended_action_label": action_label,
        "ml_static_score": round(ml_prob, 4),
        "sequence_behavioral_score": round(seq_score, 4),
        "signals_count": len(signals),
        "signals": signals,
        "requires_investigation": combined_score >= 0.45,
    }
