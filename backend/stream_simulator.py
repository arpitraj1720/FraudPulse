"""Real-Time Transaction Stream Simulator for FraudPulse.

Samples realistic streaming transactions from the Kaggle PaySim financial dataset
interwoven with coordinated multi-step attack bursts (micro-probing, mule fan-ins,
wash trading circles), and executes real-time Tier 1 scoring with live ground-truth accuracy tracking.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "PS_20174392719_1491204439457_log.csv"


class StreamSimulator:
    def __init__(self, sample_size: int = 50_000) -> None:
        self.normal_pool: List[Dict[str, Any]] = []
        self.fraud_pool: List[Dict[str, Any]] = []
        self.is_active: bool = False
        self.target_tps: float = 5.0
        self.step_counter: int = 1

        # Real-time Telemetry Metrics
        self.total_processed: int = 0
        self.total_cleared: int = 0
        self.total_flagged: int = 0
        self.false_positives_prevented: int = 0

        # Ground-Truth Confusion Matrix
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

        # Latency & Throughput Tracking
        self.recent_latencies: Deque[float] = deque(maxlen=50)
        self.recent_scored: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.tps_timestamps: Deque[float] = deque(maxlen=100)

        # Queue for multi-step coordinated attack bursts
        self.attack_queue: Deque[Dict[str, Any]] = deque()

        self._load_dataset_sample(sample_size)

    def _load_dataset_sample(self, sample_size: int) -> None:
        """Pre-load a balanced in-memory pool from PaySim for sub-millisecond streaming."""
        if not DATA_PATH.exists():
            print(f"[!] Warning: PaySim dataset not found at {DATA_PATH}. Using synthetic fallback generator.")
            self._generate_synthetic_pool()
            return

        try:
            t0 = time.time()
            df = pd.read_csv(DATA_PATH, nrows=sample_size)
            fraud_df = df[df["isFraud"] == 1]
            normal_df = df[df["isFraud"] == 0].sample(n=min(len(df) - len(fraud_df), 10_000), random_state=42)

            for _, row in normal_df.iterrows():
                self.normal_pool.append({
                    "step": int(row["step"]),
                    "type": str(row["type"]),
                    "amount": round(float(row["amount"]), 2),
                    "account_id": str(row["nameOrig"]),
                    "oldbalanceOrg": round(float(row["oldbalanceOrg"]), 2),
                    "recipient": str(row["nameDest"]),
                    "is_ground_truth_fraud": 0,
                    "source": "PaySim Normal Sample",
                })

            for _, row in fraud_df.iterrows():
                self.fraud_pool.append({
                    "step": int(row["step"]),
                    "type": str(row["type"]),
                    "amount": round(float(row["amount"]), 2),
                    "account_id": str(row["nameOrig"]),
                    "oldbalanceOrg": round(float(row["oldbalanceOrg"]), 2),
                    "recipient": str(row["nameDest"]),
                    "is_ground_truth_fraud": 1,
                    "source": "PaySim Ground-Truth Fraud",
                })

            print(f"[✓] StreamSimulator initialized in {time.time()-t0:.2f}s: {len(self.normal_pool)} normal rows, {len(self.fraud_pool)} fraud rows.")
        except Exception as e:
            print(f"[!] Error loading PaySim stream sample: {e}. Falling back to synthetic.")
            self._generate_synthetic_pool()

    def _generate_synthetic_pool(self) -> None:
        """Fallback in-memory generator if CSV dataset is absent."""
        for i in range(2000):
            amt = round(random.uniform(5.0, 450.0), 2)
            bal = round(amt * random.uniform(3.0, 40.0), 2)
            self.normal_pool.append({
                "step": 1 + (i % 24),
                "type": random.choice(["PAYMENT", "TRANSFER", "CASH_OUT"]),
                "amount": amt,
                "account_id": f"ACC-USER-{1000 + (i % 100)}",
                "oldbalanceOrg": bal,
                "recipient": f"M_Merchant_{random.randint(100, 500)}" if random.random() < 0.6 else f"C{random.randint(10000, 99999)}",
                "is_ground_truth_fraud": 0,
                "source": "Synthetic Stream Normal",
            })

        for i in range(100):
            amt = round(random.uniform(15000.0, 250000.0), 2)
            self.fraud_pool.append({
                "step": 1 + (i % 24),
                "type": random.choice(["TRANSFER", "CASH_OUT"]),
                "amount": amt,
                "account_id": f"ACC-VICTIM-{2000 + i}",
                "oldbalanceOrg": amt,  # exact wipeout
                "recipient": f"C_MULE_{random.randint(9000, 9999)}",
                "is_ground_truth_fraud": 1,
                "source": "Synthetic Ground-Truth Fraud",
            })

    def _queue_random_attack_sequence(self, attack_type: Optional[str] = None) -> None:
        """Inject a realistic multi-step attack pattern into the stream."""
        if not attack_type or attack_type == "random":
            attack_type = random.choice(["micro_probing", "mule_fan_in", "wash_circle"])

        target_account = f"ACC-ATTACK-{random.randint(100, 999)}"
        mule_wallet = f"Wallet_Mule_{random.randint(50, 99)}"

        if attack_type == "micro_probing":
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": round(random.uniform(3.0, 9.0), 2),
                "account_id": target_account,
                "oldbalanceOrg": 45000.0,
                "recipient": mule_wallet,
                "is_ground_truth_fraud": 1,
                "attack_label": "Micro-Probing Probe 1",
            })
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": round(random.uniform(10.0, 18.0), 2),
                "account_id": target_account,
                "oldbalanceOrg": 44990.0,
                "recipient": mule_wallet,
                "is_ground_truth_fraud": 1,
                "attack_label": "Micro-Probing Probe 2",
            })
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": 44970.0,
                "account_id": target_account,
                "oldbalanceOrg": 44970.0,
                "recipient": mule_wallet,
                "is_ground_truth_fraud": 1,
                "attack_label": "Micro-Probing Account Drain",
            })
        elif attack_type == "mule_fan_in":
            hub = f"Mule_Hub_{random.randint(10, 50)}"
            for i in range(3):
                self.attack_queue.append({
                    "step": self.step_counter,
                    "type": "TRANSFER",
                    "amount": round(random.uniform(2500.0, 6000.0), 2),
                    "account_id": f"ACC-VICTIM-FAN-{random.randint(1000, 9999)}",
                    "oldbalanceOrg": 15000.0,
                    "recipient": hub,
                    "is_ground_truth_fraud": 1,
                    "attack_label": f"Mule Funnel Inflow {i+1}",
                })
        elif attack_type == "wash_circle":
            ring_nodes = [f"Wash_Node_{random.randint(10, 99)}" for _ in range(3)]
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": 12500.0,
                "account_id": ring_nodes[0],
                "oldbalanceOrg": 20000.0,
                "recipient": ring_nodes[1],
                "is_ground_truth_fraud": 1,
                "attack_label": "Wash Cycle Hop 1",
            })
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": 12400.0,
                "account_id": ring_nodes[1],
                "oldbalanceOrg": 12500.0,
                "recipient": ring_nodes[2],
                "is_ground_truth_fraud": 1,
                "attack_label": "Wash Cycle Hop 2",
            })
            self.attack_queue.append({
                "step": self.step_counter,
                "type": "TRANSFER",
                "amount": 12300.0,
                "account_id": ring_nodes[2],
                "oldbalanceOrg": 12400.0,
                "recipient": ring_nodes[0],
                "is_ground_truth_fraud": 1,
                "attack_label": "Wash Cycle Closure",
            })

    def next_raw_transaction(self) -> Dict[str, Any]:
        """Fetch the next raw transaction from the attack queue or PaySim pools."""
        self.step_counter += 1

        # Check attack queue first
        if self.attack_queue:
            tx = self.attack_queue.popleft()
            tx["step"] = self.step_counter
            return tx

        # With 3% probability, trigger an attack sequence burst (every ~35 txs)
        if random.random() < 0.035:
            self._queue_random_attack_sequence()
            if self.attack_queue:
                tx = self.attack_queue.popleft()
                tx["step"] = self.step_counter
                return tx

        # With 3% probability, pick an authentic ground-truth fraud row from PaySim
        if self.fraud_pool and random.random() < 0.03:
            base = random.choice(self.fraud_pool).copy()
            base["step"] = self.step_counter
            return base

        # Otherwise pick legitimate transaction
        base = random.choice(self.normal_pool).copy()
        base["step"] = self.step_counter
        return base

    def score_and_record(
        self,
        raw_tx: Dict[str, Any],
        predict_fn: Callable[..., Tuple[float, List[Dict[str, Any]]]],
        sequence_engine: Any,
        compute_risk_fn: Callable[..., Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score transaction using Tier 1 GBDT + Tree SHAP + Sequence Engine and update live metrics."""
        t0 = time.perf_counter()

        # 1. Tier 1 Fast ML + Tree SHAP
        ml_score, shap_attributions = predict_fn(
            tx_type=raw_tx["type"],
            amount=raw_tx["amount"],
            step=raw_tx["step"],
            recipient=raw_tx["recipient"],
            oldbalanceOrg=raw_tx.get("oldbalanceOrg"),
        )

        # 2. Tier 1 Stateful Sequence & Graph Network Evaluation
        seq_eval = sequence_engine.evaluate_transaction(
            account_id=raw_tx["account_id"],
            amount=raw_tx["amount"],
            tx_type=raw_tx["type"],
            recipient=raw_tx["recipient"],
            step=raw_tx["step"],
        )

        # 3. Hybrid Blending & Policy Matrix
        hybrid = compute_risk_fn(ml_score, seq_eval)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Update Live Telemetry
        self.total_processed += 1
        now = time.time()
        self.tps_timestamps.append(now)
        self.recent_latencies.append(elapsed_ms)

        final_risk = hybrid["final_risk_score"]
        is_cleared = final_risk < 0.50
        is_flagged = final_risk >= 0.50

        if is_cleared:
            self.total_cleared += 1
        else:
            self.total_flagged += 1

        # Check False Positive Prevented:
        # Case where point-in-time ML panicked (>=0.50), but sequence trust / merchant discounted to <0.38
        if ml_score >= 0.50 and final_risk < 0.38:
            self.false_positives_prevented += 1

        # Live Ground-Truth Confusion Matrix
        is_fraud = raw_tx.get("is_ground_truth_fraud", 0)
        if is_fraud == 1:
            if is_flagged:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if is_flagged:
                self.fp += 1
            else:
                self.tn += 1

        scored_item = {
            "id": f"tx_{self.total_processed}",
            "timestamp": now,
            "raw": raw_tx,
            "ml_risk": round(ml_score, 4),
            "seq_risk": round(seq_eval["sequence_risk_score"], 4),
            "final_risk": round(final_risk, 4),
            "risk_tier": hybrid["risk_tier"],
            "policy_decision": hybrid["policy_decision"],
            "action_label": hybrid["recommended_action_label"],
            "signals": [s["title"] for s in seq_eval.get("signals", [])],
            "tree_shap_top": shap_attributions[:2] if shap_attributions else [],
            "latency_ms": round(elapsed_ms, 2),
            "is_ground_truth_fraud": is_fraud,
            "graph_summary": {
                "in_degree": seq_eval.get("graph_metrics", {}).get("recipient_in_degree", 1),
                "sender_out_degree": seq_eval.get("graph_metrics", {}).get("sender_out_degree", 1),
                "is_mule": seq_eval.get("graph_metrics", {}).get("is_mule_fan_in", False),
                "is_smurfing": seq_eval.get("graph_metrics", {}).get("is_smurfing_fan_out", False),
                "cycle": seq_eval.get("graph_metrics", {}).get("cycle_detected", False),
                "cycle_path": seq_eval.get("graph_metrics", {}).get("cycle_path", []),
                "degrees_to_flagged": seq_eval.get("graph_metrics", {}).get("degrees_to_flagged_fraud", -1),
                "closest_flagged": seq_eval.get("graph_metrics", {}).get("closest_flagged_entity"),
            },
        }

        self.recent_scored.append(scored_item)
        return scored_item

    def get_stats(self) -> Dict[str, Any]:
        """Return real-time telemetry metrics and live accuracy rates."""
        now = time.time()
        # Compute TPS over last 3 seconds
        recent_events = [t for t in self.tps_timestamps if now - t <= 3.0]
        current_tps = len(recent_events) / 3.0 if recent_events else (self.target_tps if self.is_active else 0.0)

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 1.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
        avg_lat = sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0.25

        return {
            "is_active": self.is_active,
            "target_tps": self.target_tps,
            "current_tps": round(current_tps, 1),
            "total_processed": self.total_processed,
            "total_cleared": self.total_cleared,
            "total_flagged": self.total_flagged,
            "false_positives_prevented": self.false_positives_prevented,
            "ground_truth_metrics": {
                "true_positives": self.tp,
                "false_positives": self.fp,
                "false_negatives": self.fn,
                "true_negatives": self.tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            },
            "avg_latency_ms": round(avg_lat, 2),
            "recent_count": len(self.recent_scored),
        }

    def toggle(self, active: Optional[bool] = None, tps: Optional[float] = None) -> Dict[str, Any]:
        """Start or pause the stream simulator, optionally updating target TPS."""
        if active is not None:
            self.is_active = active
        else:
            self.is_active = not self.is_active

        if tps is not None:
            self.target_tps = max(0.5, min(50.0, float(tps)))

        return {
            "is_active": self.is_active,
            "target_tps": self.target_tps,
            "message": f"Stream simulator is now {'ACTIVE' if self.is_active else 'PAUSED'} at {self.target_tps} TPS",
        }

    def reset(self) -> Dict[str, Any]:
        """Reset live stream metrics and ring buffers."""
        self.total_processed = 0
        self.total_cleared = 0
        self.total_flagged = 0
        self.false_positives_prevented = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.recent_latencies.clear()
        self.recent_scored.clear()
        self.tps_timestamps.clear()
        self.attack_queue.clear()
        return {"status": "ok", "message": "Stream simulator counters and buffers successfully reset"}


# Global singleton instance
stream_simulator = StreamSimulator()
