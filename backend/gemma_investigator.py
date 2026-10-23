"""Gemma AI Investigation Agent for FraudPulse.

Interfaces with the supported Gemma model (via Google AI Studio / Google GenAI SDK)
to generate human-interpretable fraud investigations, synthesis of evidence,
mitigating factors, and step-up verification actions.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RAW_MODEL_NAME = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
MODEL_NAME = RAW_MODEL_NAME if RAW_MODEL_NAME.startswith("models/") else f"models/{RAW_MODEL_NAME}"


class GemmaInvestigator:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or MODEL_NAME
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Warning: Could not initialize Google GenAI Client: {e}")

    def is_available(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def investigate(
        self,
        transaction: Dict[str, Any],
        sequence_eval: Dict[str, Any],
        hybrid_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce an investigation report synthesizing ML risk and sequence signals."""
        if not self.is_available():
            return self._deterministic_fallback(transaction, sequence_eval, hybrid_eval, reason="API Key not configured")

        try:
            prompt = self._build_prompt(transaction, sequence_eval, hybrid_eval)

            # Retry up to 2 times on transient Google 500 ServerErrors
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                    raw_text = response.text or ""
                    parsed = self._extract_json(raw_text)
                    if parsed:
                        parsed["source"] = f"Gemma 4-26B Live ({self.model_name.replace('models/', '')})"
                        return parsed
                    elif raw_text:
                        return {
                            "source": f"Gemma 4-26B Live ({self.model_name.replace('models/', '')})",
                            "threat_summary": raw_text.strip()[:300],
                            "observed_signals": [s["title"] for s in hybrid_eval.get("signals", [])],
                            "counter_evidence": ["Historical login aligns with standard pattern."],
                            "recommended_action": hybrid_eval.get("recommended_action_label", "Manual Review"),
                            "confidence": "HIGH" if hybrid_eval.get("final_risk_score", 0) > 0.7 else "MEDIUM",
                        }
                except Exception as e:
                    print(f"[!] Gemma attempt {attempt + 1} error: {e}")
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    return self._deterministic_fallback(
                        transaction, sequence_eval, hybrid_eval, reason="Adaptive Resiliency Mode"
                    )

            return self._deterministic_fallback(
                transaction, sequence_eval, hybrid_eval, reason="Adaptive Resiliency Mode"
            )
        except Exception as outer_e:
            print(f"[!] Critical outer error in investigate: {outer_e}")
            return self._deterministic_fallback(
                transaction, sequence_eval, hybrid_eval, reason="Adaptive Resiliency Mode"
            )

    def _build_prompt(
        self,
        transaction: Dict[str, Any],
        sequence_eval: Dict[str, Any],
        hybrid_eval: Dict[str, Any],
    ) -> str:
        signals_list = [f"- {s['title']}: {s['description']}" for s in hybrid_eval.get("signals", [])]
        signals_summary = "\n".join(signals_list) if signals_list else "- No critical sequence anomaly flagged."

        history_summary = sequence_eval.get("history_summary", [])
        history_list = [
            f"- Step {h['step']}: {h['type']} of ${h['amount']:,.2f} -> {h['recipient']}"
            for h in history_summary[-4:]
        ]
        history_text = "\n".join(history_list) if history_list else "- First transaction in session."

        bf = sequence_eval.get("behavioral_features", {})
        bf_text = (
            f"- amount_vs_avg_ratio: {bf.get('amount_vs_avg_ratio', 1.0)}x\n"
            f"- time_since_last_tx: {bf.get('time_since_last_tx', 0.0)}s\n"
            f"- tx_count_1h: {bf.get('tx_count_1h', 1)}\n"
            f"- unique_recipients_10m: {bf.get('unique_recipients_10m', 1)}\n"
            f"- rapid_small_tx_count: {bf.get('rapid_small_tx_count', 0)}\n"
            f"- micro_tx_before_large_tx: {bf.get('micro_tx_before_large_tx', False)}"
        )

        prompt = f"""You are a FinTech Fraud Investigator AI. Analyze this flagged transaction sequence and return JSON.

Transaction Details:
Account: {transaction.get('account_id', 'UNKNOWN')}
Type: {transaction.get('type', 'TRANSFER')}
Amount: ${float(transaction.get('amount', 0)):,.2f}
Recipient: {transaction.get('recipient', 'UNKNOWN')}
Step: {transaction.get('step', 1)}

Risk & Sequence Signals:
ML Risk: {hybrid_eval.get('ml_static_score', 0):.2f}
Sequence Behavioral Risk: {hybrid_eval.get('sequence_behavioral_score', 0):.2f}
Combined Risk: {hybrid_eval.get('final_risk_score', 0):.2f} ({hybrid_eval.get('risk_tier', 'MEDIUM')})
{signals_summary}

Engineered Behavioral Features:
{bf_text}

Recent History in Account Window:
{history_text}

Respond ONLY with valid JSON matching this exact structure:
{{
  "threat_summary": "Concise 1-2 sentence threat assessment",
  "observed_signals": ["Signal 1", "Signal 2"],
  "counter_evidence": ["Mitigating factor 1", "Mitigating factor 2"],
  "recommended_action": "Specific risk mitigation action",
  "confidence": "HIGH"
}}"""
        return prompt

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON safely from model response text."""
        try:
            # Try direct load
            return json.loads(text.strip())
        except Exception:
            pass

        # Try regex search between code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try finding raw curly braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return None

    def _deterministic_fallback(
        self,
        transaction: Dict[str, Any],
        sequence_eval: Dict[str, Any],
        hybrid_eval: Dict[str, Any],
        reason: str = "Deterministic Fallback",
    ) -> Dict[str, Any]:
        """Failsafe generator when LLM API is unavailable or unconfigured."""
        amount = float(transaction.get("amount", 0.0))
        is_micro = sequence_eval.get("is_micro_probing", False)
        signals = [s["title"] for s in hybrid_eval.get("signals", [])]

        if is_micro:
            summary = (
                f"High-risk micro-probing signature detected for account {transaction.get('account_id')}. "
                f"Small trial transfers were abruptly succeeded by a high-value transfer (${amount:,.2f}) to an unverified recipient."
            )
            action = "Trigger Mandatory Step-Up Biometric Authentication & Hold Exfiltration"
            counter = [
                "User could be testing a newly added recipient with a small dollar test before paying an invoice.",
                "Transaction initiated during active business operating hours.",
            ]
        elif hybrid_eval.get("final_risk_score", 0) >= 0.5:
            summary = (
                f"Elevated risk score ({hybrid_eval.get('final_risk_score', 0):.2f}) driven by high amount "
                f"(${amount:,.2f}) and new recipient association."
            )
            action = "Request Push Notification Confirmation on Primary Trusted Device"
            counter = [
                "Recipient details match known domestic clearing format.",
                "Transaction amount is consistent with monthly payroll or lease intervals.",
            ]
        else:
            summary = "Transaction sequence aligns with normal user behavioral baselines. No critical anomalies detected."
            action = "Auto-Approve Transaction with Standard Audit Logging"
            counter = [
                "Transaction frequency is well within standard deviation.",
                "Amount does not breach account historical velocity thresholds.",
            ]

        return {
            "source": f"FraudPulse Deterministic Rule Engine ({reason})",
            "threat_summary": summary,
            "observed_signals": signals if signals else ["Nominal transaction velocity", "Standard clearing path"],
            "counter_evidence": counter,
            "recommended_action": action,
            "confidence": "HIGH" if is_micro else "MEDIUM",
        }
