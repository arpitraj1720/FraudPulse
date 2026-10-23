"""FastAPI Main Server for FraudPulse.

Integrates:
- Supervised ML Fraud Classifier (Scikit-Learn Joblib Bundle)
- Stateful Behavioral Sequence Engine
- Gemma AI Investigation Agent (Google GenAI SDK)
- Live Scenario Replay Engine
- Modern Dashboard SPA Static Serving
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.gemma_investigator import GemmaInvestigator
from backend.scenarios import SCENARIOS, get_scenario
from backend.sequence_engine import SequenceEngine, compute_hybrid_risk
from train_model import PAY_SIM_TYPES, make_features

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "frontend" / "static"
MODEL_PATH = ROOT_DIR / "fraudpulse_model.joblib"
METRICS_PATH = ROOT_DIR / "training_metrics.json"

app = FastAPI(
    title="FraudPulse API",
    description="Real-Time FinTech Fraud Sequence Detection & Gemma AI Investigation Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory components
sequence_engine = SequenceEngine(window_size=12, time_window_seconds=3600.0)
gemma_investigator = GemmaInvestigator()
model_bundle: Optional[Dict[str, Any]] = None
model_metrics: Optional[Dict[str, Any]] = None


def load_artifacts() -> None:
    global model_bundle, model_metrics
    if MODEL_PATH.exists():
        try:
            model_bundle = joblib.load(MODEL_PATH)
            print(f"[*] Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            print(f"[!] Error loading model: {e}")

    if METRICS_PATH.exists():
        try:
            model_metrics = json.loads(METRICS_PATH.read_text())
            print(f"[*] Loaded training metrics from {METRICS_PATH}")
        except Exception as e:
            print(f"[!] Error loading metrics: {e}")


load_artifacts()


class TransactionPayload(BaseModel):
    account_id: str = Field(default="ACC-USER-1", description="Source Account ID")
    type: str = Field(default="TRANSFER", description="Transaction type (e.g. TRANSFER, PAYMENT, CASH_OUT)")
    amount: float = Field(default=500.0, description="Amount in transaction currency")
    recipient: str = Field(default="WALLET-EXT-1", description="Destination payee / wallet")
    step: int = Field(default=1, description="Simulation step / hour")
    auto_investigate: bool = Field(default=True, description="Whether to call Gemma if risk threshold reached")


def predict_ml_risk(
    tx_type: str,
    amount: float,
    step: int = 1,
    recipient: str = "",
    oldbalanceOrg: Optional[float] = None,
) -> float:
    """Predict point-in-time anomaly probability from trained Random Forest."""
    if not model_bundle or "model" not in model_bundle:
        # Fallback heuristic if model not trained
        base = 0.65 if tx_type in ("TRANSFER", "CASH_OUT") else 0.10
        if amount > 10000:
            base += 0.25
        return min(0.95, base)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    raw_dict: Dict[str, Any] = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "recipient": recipient,
        "nameDest": recipient,
    }
    if oldbalanceOrg is not None:
        raw_dict["oldbalanceOrg"] = oldbalanceOrg

    raw = pd.DataFrame([raw_dict])
    X = make_features(raw).reindex(columns=feature_columns, fill_value=0)
    prob = float(model.predict_proba(X)[0, 1])
    return prob


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "FraudPulse",
        "model_loaded": model_bundle is not None,
        "model_version": model_bundle.get("model_version", "none") if model_bundle else "none",
        "gemma_available": gemma_investigator.is_available(),
        "gemma_model": gemma_investigator.model_name,
        "active_monitored_accounts": len(sequence_engine.history),
    }


@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    if not model_metrics:
        raise HTTPException(status_code=404, detail="Training metrics not found. Run train_model.py first.")
    return model_metrics


@app.get("/api/scenarios")
def list_scenarios() -> List[Dict[str, Any]]:
    return SCENARIOS


@app.post("/api/reset")
def reset_engine(account_id: Optional[str] = None) -> Dict[str, Any]:
    sequence_engine.reset(account_id)
    return {"status": "ok", "message": f"Sequence state reset for {'all accounts' if not account_id else account_id}"}


class InvestigatePayload(BaseModel):
    transaction: Dict[str, Any]
    sequence_evaluation: Dict[str, Any]
    hybrid_risk: Dict[str, Any]


@app.post("/api/investigate")
def run_investigation(payload: InvestigatePayload) -> Dict[str, Any]:
    """Tier 2 Asynchronous Intelligence: Gemma 27B Autonomous Copilot investigation."""
    start_t = time.time()
    try:
        brief = gemma_investigator.investigate(
            payload.transaction,
            payload.sequence_evaluation,
            payload.hybrid_risk,
        )
    except Exception as e:
        print(f"[!] Error in investigation endpoint: {e}")
        brief = gemma_investigator._deterministic_fallback(
            payload.transaction,
            payload.sequence_evaluation,
            payload.hybrid_risk,
            reason="Adaptive Resiliency Mode",
        )
    brief["generation_time_ms"] = round((time.time() - start_t) * 1000, 2)
    return brief


@app.post("/api/score")
def score_transaction(payload: TransactionPayload) -> Dict[str, Any]:
    """Tier 1 Hot Path: Sub-5ms scoring via ML classifier and Sequence Engine."""
    start_t = time.time()

    # 1. ML Classifier
    ml_prob = predict_ml_risk(
        tx_type=payload.type,
        amount=payload.amount,
        step=payload.step,
        recipient=payload.recipient,
    )

    # 2. Sequence Engine
    seq_eval = sequence_engine.evaluate_transaction(
        account_id=payload.account_id,
        amount=payload.amount,
        tx_type=payload.type,
        recipient=payload.recipient,
        step=payload.step,
    )

    # 3. Hybrid Risk Aggregator
    hybrid_eval = compute_hybrid_risk(ml_prob, seq_eval)
    tier1_latency_ms = round((time.time() - start_t) * 1000, 2)

    tx_dict = {
        "account_id": payload.account_id,
        "type": payload.type,
        "amount": payload.amount,
        "recipient": payload.recipient,
        "step": payload.step,
    }

    return {
        "transaction": payload.dict(),
        "ml_risk": round(ml_prob, 4),
        "sequence_evaluation": seq_eval,
        "hybrid_risk": hybrid_eval,
        "tier1_latency_ms": tier1_latency_ms,
        "requires_investigation": hybrid_eval["requires_investigation"],
        "investigation_payload": {
            "transaction": tx_dict,
            "sequence_evaluation": seq_eval,
            "hybrid_risk": hybrid_eval,
        } if hybrid_eval["requires_investigation"] else None,
        "timestamp": time.time(),
    }


@app.post("/api/scenarios/replay/{scenario_id}")
def replay_scenario(scenario_id: str) -> Dict[str, Any]:
    """Tier 1 Hot Path: Replays entire scenario in sub-5ms, returning telemetry immediately."""
    start_time = time.time()
    try:
        scenario = get_scenario(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    account_id = scenario["account_id"]
    sequence_engine.reset(account_id)

    step_results = []

    for tx in scenario["transactions"]:
        ml_prob = predict_ml_risk(
            tx_type=tx["type"],
            amount=tx["amount"],
            step=tx["step"],
            recipient=tx.get("recipient", ""),
            oldbalanceOrg=tx.get("oldbalanceOrg"),
        )
        seq_eval = sequence_engine.evaluate_transaction(
            account_id=account_id,
            amount=tx["amount"],
            tx_type=tx["type"],
            recipient=tx["recipient"],
            step=tx["step"],
        )
        hybrid_eval = compute_hybrid_risk(ml_prob, seq_eval)

        step_results.append({
            "step": tx["step"],
            "type": tx["type"],
            "amount": tx["amount"],
            "recipient": tx["recipient"],
            "label": tx.get("label", ""),
            "ml_risk": round(ml_prob, 4),
            "sequence_score": round(seq_eval["sequence_risk_score"], 4),
            "final_risk_score": round(hybrid_eval["final_risk_score"], 4),
            "risk_tier": hybrid_eval["risk_tier"],
            "policy_decision": hybrid_eval["policy_decision"],
            "action_label": hybrid_eval["recommended_action_label"],
            "signals": seq_eval["signals"],
            "is_micro_probing": seq_eval["is_micro_probing"],
            "behavioral_features": seq_eval.get("behavioral_features", {}),
            "requires_investigation": hybrid_eval["requires_investigation"],
            "raw_seq_eval": seq_eval,
            "raw_hybrid_eval": hybrid_eval,
        })

    tier1_latency_ms = round((time.time() - start_time) * 1000, 2)

    # Identify the single most critical step that requires Tier 2 Gemma escalation
    critical_step = max(step_results, key=lambda s: s["final_risk_score"]) if step_results else None
    requires_inv = bool(critical_step and critical_step.get("requires_investigation"))

    investigation_payload = None
    if requires_inv and critical_step:
        investigation_payload = {
            "transaction": {
                "account_id": account_id,
                "type": critical_step["type"],
                "amount": critical_step["amount"],
                "recipient": critical_step["recipient"],
                "step": critical_step["step"],
            },
            "sequence_evaluation": critical_step["raw_seq_eval"],
            "hybrid_risk": critical_step["raw_hybrid_eval"],
        }

    # Clean up internal raw dicts before returning JSON
    for s in step_results:
        s.pop("raw_seq_eval", None)
        s.pop("raw_hybrid_eval", None)

    return {
        "scenario": scenario,
        "account_id": account_id,
        "steps": step_results,
        "final_step": critical_step,
        "tier1_latency_ms": tier1_latency_ms,
        "requires_investigation": requires_inv,
        "investigation_payload": investigation_payload,
    }


# Static files mount for the Frontend UI
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>FraudPulse Dashboard Loading...</h1>")
