"""In-Memory Graph Risk and Network Topology Engine for FraudPulse.

Implements high-speed topological fraud detection using NetworkX:
1. In-Degree / Fan-In Spikes (Mule Account detection)
2. Out-Degree / Fan-Out Velocity (Smurfing / Layering dispersion)
3. Circular Path & Cycle Detection (Money washing loops)
4. Degrees of Separation from Flagged Fraud Entities
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


class GraphRiskEngine:
    def __init__(self, time_window_seconds: float = 7200.0):
        self.time_window_seconds = time_window_seconds
        # Directed multi-graph representing transaction flows
        self.graph = nx.DiGraph()
        # Set of known fraudulent or flagged node IDs
        self.flagged_nodes: Set[str] = set()
        # Seed initial known flagged mule wallets for demo realism
        self.flagged_nodes.add("Wallet_99x_Offshore")
        self.flagged_nodes.add("C553264065")

    def reset(self) -> None:
        """Clear graph state and reset to baseline."""
        self.graph.clear()
        self.flagged_nodes = {"Wallet_99x_Offshore", "C553264065"}

    def flag_node(self, node_id: str) -> None:
        """Mark an account or wallet as confirmed fraudulent."""
        self.flagged_nodes.add(node_id)
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id]["is_fraud_flagged"] = True

    def add_transaction(
        self,
        sender: str,
        recipient: str,
        amount: float,
        step: int = 1,
        tx_timestamp: Optional[float] = None,
        tx_type: str = "TRANSFER",
    ) -> None:
        """Add directed transaction edge to the network graph."""
        now = tx_timestamp if tx_timestamp is not None else time.time()

        if not self.graph.has_node(sender):
            self.graph.add_node(
                sender,
                entity_type="merchant" if sender.startswith("M") else "customer",
                total_in=0.0,
                total_out=0.0,
                is_fraud_flagged=sender in self.flagged_nodes,
                first_seen=now,
            )

        if not self.graph.has_node(recipient):
            self.graph.add_node(
                recipient,
                entity_type="merchant" if recipient.startswith("M") else "customer",
                total_in=0.0,
                total_out=0.0,
                is_fraud_flagged=recipient in self.flagged_nodes,
                first_seen=now,
            )

        # Update node balance flows
        self.graph.nodes[sender]["total_out"] += amount
        self.graph.nodes[recipient]["total_in"] += amount

        # Add or update directed edge
        if self.graph.has_edge(sender, recipient):
            edge = self.graph[sender][recipient]
            edge["weight"] += amount
            edge["count"] += 1
            edge["last_timestamp"] = now
        else:
            self.graph.add_edge(
                sender,
                recipient,
                weight=amount,
                count=1,
                last_timestamp=now,
                tx_type=tx_type,
                step=step,
            )

    def evaluate_topology(
        self,
        sender: str,
        recipient: str,
        amount: float,
        step: int = 1,
        tx_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate graph topology metrics and signals prior to committing transaction."""
        now = tx_timestamp if tx_timestamp is not None else time.time()
        signals: List[Dict[str, Any]] = []
        graph_risk = 0.0

        # Pre-existing in-degree (unique originators paying this recipient)
        recipient_in_degree = self.graph.in_degree(recipient) if self.graph.has_node(recipient) else 0
        sender_out_degree = self.graph.out_degree(sender) if self.graph.has_node(sender) else 0

        # 1. Money Mule Detection: Fan-In Spike
        is_merchant = recipient.startswith("M") or "merchant" in recipient.lower()
        if not is_merchant:
            # If a peer/unverified wallet has multiple distinct senders feeding it
            if recipient_in_degree >= 2 and not self.graph.has_edge(sender, recipient):
                signals.append({
                    "code": "MULE_FAN_IN_SPIKE",
                    "severity": "HIGH",
                    "title": "Mule Funnel Pattern (Fan-In)",
                    "description": f"Destination wallet '{recipient}' is funneling incoming transfers from {recipient_in_degree + 1} distinct accounts.",
                    "risk_contribution": 0.25,
                })
                graph_risk += 0.25

        # 2. Smurfing / Layering Detection: Fan-Out Dispersion
        if sender_out_degree >= 3:
            signals.append({
                "code": "SMURFING_FAN_OUT",
                "severity": "MEDIUM",
                "title": "Fund Dispersion (Fan-Out)",
                "description": f"Originator has rapidly dispersed transfers to {sender_out_degree + 1} distinct destination wallets.",
                "risk_contribution": 0.20,
            })
            graph_risk += 0.20

        # 3. Circular Wash Trading / Routing Cycle Detection
        cycle_detected = False
        cycle_path: List[str] = []
        if self.graph.has_node(recipient) and self.graph.has_node(sender):
            try:
                # Check if a directed path already exists from recipient back to sender
                if nx.has_path(self.graph, recipient, sender):
                    cycle_path = nx.shortest_path(self.graph, recipient, sender)
                    cycle_detected = True
                    signals.append({
                        "code": "CIRCULAR_WASH_CYCLE",
                        "severity": "CRITICAL",
                        "title": "Circular Fund Routing Loop",
                        "description": f"Circular loop detected: money moves from {' -> '.join(cycle_path)} -> {recipient}.",
                        "risk_contribution": 0.40,
                    })
                    graph_risk += 0.40
            except Exception:
                pass

        # 4. Proximity to Flagged Fraud Entities (Degrees of Separation)
        min_separation = 999
        closest_flagged_node = None
        if self.graph.number_of_nodes() > 1:
            # Undirected view for contagion analysis
            undirected_view = self.graph.to_undirected(as_view=True)
            for flagged in self.flagged_nodes:
                if self.graph.has_node(flagged):
                    # Check separation from recipient or sender
                    target = recipient if recipient != flagged else sender
                    if self.graph.has_node(target) and nx.has_path(undirected_view, target, flagged):
                        dist = nx.shortest_path_length(undirected_view, target, flagged)
                        if dist < min_separation:
                            min_separation = dist
                            closest_flagged_node = flagged

        if min_separation == 0 or recipient in self.flagged_nodes:
            signals.append({
                "code": "DIRECT_FLAGGED_DESTINATION",
                "severity": "CRITICAL",
                "title": "Direct Interaction with Flagged Entity",
                "description": f"Recipient '{recipient}' is on the confirmed AML/Fraud blacklist.",
                "risk_contribution": 0.50,
            })
            graph_risk += 0.50
        elif min_separation == 1:
            signals.append({
                "code": "PROXIMITY_FLAGGED_1HOP",
                "severity": "HIGH",
                "title": "1-Hop Proximity to Flagged Fraud",
                "description": f"Recipient is directly connected to blacklisted entity '{closest_flagged_node}' (1 degree of separation).",
                "risk_contribution": 0.30,
            })
            graph_risk += 0.30
        elif min_separation == 2:
            signals.append({
                "code": "PROXIMITY_FLAGGED_2HOP",
                "severity": "LOW",
                "title": "2-Hop Network Association",
                "description": f"Counterparty is 2 degrees of separation from flagged node '{closest_flagged_node}'.",
                "risk_contribution": 0.10,
            })
            graph_risk += 0.10

        # Now commit the transaction edge to graph state
        self.add_transaction(sender, recipient, amount, step, now)

        # Graph topological feature summary
        metrics = {
            "recipient_in_degree": recipient_in_degree + 1,
            "sender_out_degree": sender_out_degree + 1,
            "is_mule_fan_in": recipient_in_degree >= 2 and not is_merchant,
            "is_smurfing_fan_out": sender_out_degree >= 3,
            "cycle_detected": cycle_detected,
            "cycle_path": cycle_path,
            "degrees_to_flagged_fraud": min_separation if min_separation < 999 else -1,
            "closest_flagged_entity": closest_flagged_node,
            "graph_risk_contribution": round(min(1.0, graph_risk), 4),
            "total_network_nodes": self.graph.number_of_nodes(),
            "total_network_edges": self.graph.number_of_edges(),
        }

        return {
            "signals": signals,
            "graph_risk_score": round(min(1.0, graph_risk), 4),
            "metrics": metrics,
        }


# Singleton Global Graph Risk Engine instance
graph_risk_engine = GraphRiskEngine()
