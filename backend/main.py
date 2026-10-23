"""FastAPI Main Server for FraudPulse.

Integrates:
- Supervised ML Fraud Classifier (Scikit-Learn Joblib Bundle)
- Stateful Behavioral Sequence Engine
- Gemma AI Investigation Agent (Google GenAI SDK)
- Live Scenario Replay Engine
- Modern Dashboard SPA Static Serving
"""
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.gemma_investigator import GemmaInvestigator
from backend.graph_engine import graph_risk_engine
from backend.scenarios import SCENARIOS, get_scenario
from backend.sequence_engine import SequenceEngine, compute_hybrid_risk
from backend.stream_simulator import stream_simulator
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


def make_features_fast(
    step: int,
    tx_type: str,
    amount: float,
    recipient: str = "",
    oldbalanceOrg: Optional[float] = None,
) -> np.ndarray:
    """Ultra-fast NumPy feature generator bypassing Pandas DataFrame overhead.
    
    Constructs the exact 14 float32 input features in ~3 microseconds.
    """
    amount_f = max(0.0, float(amount))
    oldbal_f = max(0.0, float(oldbalanceOrg)) if oldbalanceOrg is not None else 0.0
    step_f = float(step)

    log_amt = math.log1p(amount_f)
    log_oldbal = math.log1p(oldbal_f)
    drain_ratio = min(10.0, amount_f / (oldbal_f + 1.0))
    is_exact_wipeout = 1.0 if (abs(oldbal_f - amount_f) < 1.0 and amount_f > 100.0) else 0.0

    r_str = recipient or ""
    r_lower = r_str.lower()
    is_merchant = 1.0 if (r_str.startswith("M") or "merchant" in r_lower) else 0.0
    is_round = 1.0 if (amount_f % 100 == 0 and amount_f >= 1000.0) else 0.0

    hour = (step_f % 24.0) / 24.0
    hour_sin = math.sin(2.0 * math.pi * hour)
    hour_cos = math.cos(2.0 * math.pi * hour)

    t_in = 1.0 if tx_type == "CASH_IN" else 0.0
    t_out = 1.0 if tx_type == "CASH_OUT" else 0.0
    t_debit = 1.0 if tx_type == "DEBIT" else 0.0
    t_pay = 1.0 if tx_type == "PAYMENT" else 0.0
    t_trans = 1.0 if tx_type == "TRANSFER" else 0.0

    return np.array([[
        step_f, log_amt, log_oldbal, drain_ratio, is_exact_wipeout,
        is_merchant, is_round, hour_sin, hour_cos,
        t_in, t_out, t_debit, t_pay, t_trans
    ]], dtype=np.float32)


def predict_ml_with_shap(
    tx_type: str,
    amount: float,
    step: int = 1,
    recipient: str = "",
    oldbalanceOrg: Optional[float] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Predict point-in-time anomaly probability and extract native Tree SHAP attributions in <0.3ms."""
    if not model_bundle or "model" not in model_bundle:
        base = 0.65 if tx_type in ("TRANSFER", "CASH_OUT") else 0.10
        if amount > 10000:
            base += 0.25
        return min(0.95, base), []

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    X_np = make_features_fast(step, tx_type, amount, recipient, oldbalanceOrg)

    # Use native C++ booster predict if available for sub-millisecond latency & zero pandas overhead
    attributions: List[Dict[str, Any]] = []
    if hasattr(model, "booster_"):
        try:
            prob = float(model.booster_.predict(X_np)[0])
            contribs = model.booster_.predict(X_np, pred_contrib=True)[0]
            shap_values = contribs[:-1]
            ranked = sorted(zip(feature_columns, shap_values), key=lambda x: abs(x[1]), reverse=True)
            for feat, val in ranked[:5]:
                attributions.append({
                    "feature": feat,
                    "shap_value": round(float(val), 3),
                    "impact": "+RISK" if val > 0 else "-SAFE",
                })
            return prob, attributions
        except Exception:
            pass

    # Fallback to standard scikit-learn API
    prob = float(model.predict_proba(X_np)[0, 1])
    return prob, attributions


def predict_ml_risk(
    tx_type: str,
    amount: float,
    step: int = 1,
    recipient: str = "",
    oldbalanceOrg: Optional[float] = None,
) -> float:
    """Predict point-in-time anomaly probability from trained GBDT."""
    prob, _ = predict_ml_with_shap(tx_type, amount, step, recipient, oldbalanceOrg)
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
        "graph_nodes_count": graph_risk_engine.graph.number_of_nodes(),
        "graph_edges_count": graph_risk_engine.graph.number_of_edges(),
        "graph_flagged_entities": len(graph_risk_engine.flagged_nodes),
    }


@app.get("/api/graph/stats")
def get_graph_stats() -> Dict[str, Any]:
    """Return topological graph network metrics for TigerGraph-style visualization."""
    return {
        "nodes_count": graph_risk_engine.graph.number_of_nodes(),
        "edges_count": graph_risk_engine.graph.number_of_edges(),
        "flagged_entities": list(graph_risk_engine.flagged_nodes),
        "active": True,
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
    return {"status": "ok", "message": f"Sequence and network graph state reset for {'all accounts' if not account_id else account_id}"}


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

    # 1. ML Classifier with Native Tree SHAP Attributions
    ml_prob, ml_attributions = predict_ml_with_shap(
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
        "ml_attributions": ml_attributions,
    }

    return {
        "transaction": payload.dict(),
        "ml_risk": round(ml_prob, 4),
        "ml_attributions": ml_attributions,
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
        sender_id = tx.get("sender", account_id)
        ml_prob, ml_attributions = predict_ml_with_shap(
            tx_type=tx["type"],
            amount=tx["amount"],
            step=tx["step"],
            recipient=tx.get("recipient", ""),
            oldbalanceOrg=tx.get("oldbalanceOrg"),
        )
        seq_eval = sequence_engine.evaluate_transaction(
            account_id=sender_id,
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
            "sender": sender_id,
            "recipient": tx["recipient"],
            "label": tx.get("label", ""),
            "ml_risk": round(ml_prob, 4),
            "ml_attributions": ml_attributions,
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
                "account_id": critical_step.get("sender", account_id),
                "type": critical_step["type"],
                "amount": critical_step["amount"],
                "recipient": critical_step["recipient"],
                "step": critical_step["step"],
                "ml_attributions": critical_step.get("ml_attributions", []),
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


# -------------------------------------------------------------
# Real-Time Transaction Stream Simulator & SSE Endpoints
# -------------------------------------------------------------

class StreamTogglePayload(BaseModel):
    active: Optional[bool] = None
    tps: Optional[float] = None


class StreamStepPayload(BaseModel):
    count: int = Field(default=1, ge=1, le=50)


@app.get("/api/stream/stats")
def get_stream_stats() -> Dict[str, Any]:
    """Return real-time stream simulator telemetry, throughput, and accuracy metrics."""
    return stream_simulator.get_stats()


@app.get("/api/stream/recent")
def get_recent_stream_transactions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recently scored stream transactions."""
    return list(stream_simulator.recent_scored)[-limit:]


@app.get("/api/stream/batch")
def get_stream_batch(size: int = 5) -> Dict[str, Any]:
    """Score and return the next batch of stream transactions on demand."""
    batch_size = max(1, min(50, size))
    results = []
    for _ in range(batch_size):
        raw = stream_simulator.next_raw_transaction()
        scored = stream_simulator.score_and_record(
            raw,
            predict_ml_with_shap,
            sequence_engine,
            compute_hybrid_risk,
        )
        results.append(scored)
    return {
        "items": results,
        "count": len(results),
        "stats": stream_simulator.get_stats(),
    }


@app.post("/api/stream/step")
def step_stream(payload: StreamStepPayload) -> Dict[str, Any]:
    """Execute 1 to 50 stream steps manually for step-by-step UI debugging."""
    return get_stream_batch(size=payload.count)


@app.post("/api/stream/toggle")
def toggle_stream(payload: StreamTogglePayload) -> Dict[str, Any]:
    """Start or pause the stream simulator, optionally updating target TPS."""
    return stream_simulator.toggle(active=payload.active, tps=payload.tps)


@app.post("/api/stream/reset")
def reset_stream() -> Dict[str, Any]:
    """Reset stream simulator counters, buffers, and sequence state."""
    res = stream_simulator.reset()
    sequence_engine.reset()
    return res


class StreamInjectPayload(BaseModel):
    attack_type: str = Field(default="random", description="micro_probing, mule_fan_in, wash_circle, or random")


@app.post("/api/stream/inject")
def inject_stream_attack(payload: StreamInjectPayload) -> Dict[str, Any]:
    """Inject a coordinated attack burst into the active transaction stream."""
    stream_simulator._queue_random_attack_sequence(attack_type=payload.attack_type)
    return {
        "status": "ok",
        "attack_type": payload.attack_type,
        "message": f"Injected '{payload.attack_type}' attack sequence into stream queue ({len(stream_simulator.attack_queue)} steps queued)",
        "queue_length": len(stream_simulator.attack_queue),
    }


@app.get("/api/stream/live")
async def live_stream_sse():
    """Server-Sent Events (SSE) streaming endpoint pushing live scored transactions at configured TPS.
    
    Usage in frontend JavaScript:
        const es = new EventSource('/api/stream/live');
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log(data.transaction, data.stats);
        };
    """
    async def event_generator():
        stream_simulator.is_active = True
        try:
            while True:
                if stream_simulator.is_active:
                    raw = stream_simulator.next_raw_transaction()
                    scored = stream_simulator.score_and_record(
                        raw,
                        predict_ml_with_shap,
                        sequence_engine,
                        compute_hybrid_risk,
                    )
                    stats = stream_simulator.get_stats()
                    payload = json.dumps({"transaction": scored, "stats": stats})
                    yield f"data: {payload}\n\n"
                    sleep_time = max(0.02, 1.0 / stream_simulator.target_tps)
                    await asyncio.sleep(sleep_time)
                else:
                    # Send heartbeat ping every second when paused
                    stats = stream_simulator.get_stats()
                    yield f": heartbeat {json.dumps(stats)}\n\n"
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.get("/stream", response_class=HTMLResponse)
@app.head("/stream", response_class=HTMLResponse)
def serve_stream_dashboard():
    stream_file = STATIC_DIR / "stream.html"
    if stream_file.exists():
        return FileResponse(stream_file)
    return HTMLResponse("<h1>FraudPulse Live Stream Loading...</h1>")
