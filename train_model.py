"""Train a first FraudPulse fraud classifier on the PaySim dataset.

Important: this is a hackathon prototype trained on synthetic mobile-money data.
It is not suitable for making real financial decisions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = ROOT / "fraudpulse_model.joblib"
METRICS_PATH = ROOT / "training_metrics.json"
RANDOM_STATE = 42
MAX_LEGITIMATE_TRAIN = 300_000  # cap training time; retain all fraud rows in train split


PAY_SIM_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create only transaction-time features; deliberately avoid post-transaction balances."""
    step = pd.to_numeric(raw["step"], errors="coerce").fillna(0).astype("float32")
    amount = pd.to_numeric(raw["amount"], errors="coerce").fillna(0).clip(lower=0).astype("float32")
    hour = (step % 24.0) / 24.0
    features = pd.DataFrame(index=raw.index)
    features["step"] = step
    features["log_amount"] = np.log1p(amount)
    features["hour_sin"] = np.sin(2 * np.pi * hour).astype("float32")
    features["hour_cos"] = np.cos(2 * np.pi * hour).astype("float32")
    raw_type = raw["type"].astype(str)
    for t in PAY_SIM_TYPES:
        features[f"type_{t}"] = (raw_type == t).astype("uint8")
    return features


def report_at_threshold(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype("uint8")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }
    return out


def generate_synthetic_data(n_samples: int = 15_000) -> pd.DataFrame:
    """Generate realistic PaySim-like synthetic data for instant hackathon testing without 470MB download."""
    np.random.seed(RANDOM_STATE)
    steps = np.random.randint(1, 744, size=n_samples)
    types = np.random.choice(
        PAY_SIM_TYPES,
        size=n_samples,
        p=[0.22, 0.35, 0.01, 0.34, 0.08],
    )
    # Amounts generally follow log-normal
    amounts = np.random.lognormal(mean=8.5, sigma=1.8, size=n_samples).round(2)
    # In PaySim, fraud is strictly TRANSFER and CASH_OUT, often high amounts
    is_fraud = np.zeros(n_samples, dtype=np.uint8)
    fraud_candidates = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amounts > 50_000))[0]
    if len(fraud_candidates) > 0:
        fraud_idx = np.random.choice(
            fraud_candidates,
            size=min(len(fraud_candidates), int(n_samples * 0.03)),
            replace=False,
        )
        is_fraud[fraud_idx] = 1

    df = pd.DataFrame({
        "step": steps.astype("int16"),
        "type": types,
        "amount": amounts.astype("float32"),
        "isFraud": is_fraud,
    })
    return df


def main() -> None:
    use_synthetic = "--synthetic" in sys.argv
    csv_arg = next((arg for arg in sys.argv[1:] if not arg.startswith("--")), None)
    data_path = Path(csv_arg) if csv_arg else DATA_PATH

    started = time.time()
    usecols = ["step", "type", "amount", "isFraud"]
    dtypes = {"step": "int16", "type": "category", "amount": "float32", "isFraud": "int8"}

    if use_synthetic or not data_path.exists():
        if not data_path.exists() and not use_synthetic:
            print(f"[!] Real PaySim dataset not found at: {data_path}")
            print("[*] Generating synthetic PaySim-aligned dataset for instant local training & validation...")
        else:
            print("[*] Using synthetic PaySim-aligned generation mode (--synthetic)...")
        data = generate_synthetic_data(25_000)
    else:
        print(f"Loading {data_path} ...")
        data = pd.read_csv(data_path, usecols=usecols, dtype=dtypes)
        data = data.dropna(subset=usecols).reset_index(drop=True)
        data["isFraud"] = data["isFraud"].astype("uint8")

    print(f"Rows loaded: {len(data):,}")
    print("Class counts (0=legitimate, 1=fraud):")
    print(data["isFraud"].value_counts().sort_index().to_string())
    print(f"Observed dataset fraud prevalence: {data['isFraud'].mean():.4%}")

    # Keep a representative test split from the original class distribution.
    train_raw, test_raw = train_test_split(
        data, test_size=0.20, random_state=RANDOM_STATE, stratify=data["isFraud"]
    )

    # Downsample only the training majority class; never downsample the held-out test set.
    fraud_train = train_raw[train_raw["isFraud"] == 1]
    legit_train = train_raw[train_raw["isFraud"] == 0]
    if len(legit_train) > MAX_LEGITIMATE_TRAIN:
        legit_train = legit_train.sample(n=MAX_LEGITIMATE_TRAIN, random_state=RANDOM_STATE)
    train_small = pd.concat([legit_train, fraud_train], axis=0).sample(
        frac=1, random_state=RANDOM_STATE
    )
    y_train = train_small["isFraud"].astype("uint8")
    y_test = test_raw["isFraud"].astype("uint8")

    print(f"Training rows: {len(train_small):,} ({int(y_train.sum()):,} fraud)")
    print(f"Held-out test rows: {len(test_raw):,} ({int(y_test.sum()):,} fraud)")
    print("Building transaction-time features (step, type, amount; no post-transaction balances)...")
    X_train = make_features(train_small)
    feature_columns = list(X_train.columns)
    X_test = make_features(test_raw).reindex(columns=feature_columns, fill_value=0)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    print("Training RandomForestClassifier ...")
    model.fit(X_train, y_train)
    print("Scoring held-out test set ...")
    prob = model.predict_proba(X_test)[:, 1]

    # Report multiple thresholds to expose the precision/recall trade-off.
    metrics = {
        "dataset": "PaySim (Synthetic Mode)" if (use_synthetic or not data_path.exists()) else "PaySim Full Dataset",
        "rows_loaded": int(len(data)),
        "fraud_prevalence_test": float(y_test.mean()),
        "training_rows": int(len(train_small)),
        "training_fraud_rows": int(y_train.sum()),
        "test_rows": int(len(test_raw)),
        "test_fraud_rows": int(y_test.sum()),
        "average_precision_pr_auc": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "threshold_reports": [report_at_threshold(y_test, prob, t) for t in (0.5, 0.25, 0.10)],
        "features": feature_columns,
        "warning": "Synthetic-data prototype. Thresholds are not calibrated for real-world deployment.",
        "elapsed_seconds": round(time.time() - started, 2),
    }
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "model_version": "fraudpulse-baseline-0.1",
        "input_fields": ["step", "type", "amount"],
        "types_supported": PAY_SIM_TYPES,
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(f"PR-AUC / Average precision: {metrics['average_precision_pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    for row in metrics["threshold_reports"]:
        print(
            f"threshold={row['threshold']:.2f} | precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} F1={row['f1']:.4f} "
            f"false_alerts={row['false_positives']:,} "
            f"missed_fraud={row['false_negatives']:,} "
            f"FPR={row['false_positive_rate']:.4%}"
        )
    print(f"\nSaved trained model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
