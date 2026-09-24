"""Train, evaluate, and export the churn model.

Run locally:

    python -m trainer.task --data-path data/raw/Telco-Customer-Churn.csv --model-dir artifacts/model

On Vertex AI custom training, omit --model-dir so the job writes to AIP_MODEL_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from trainer.data import load_raw_csv, clean_churn_frame, split_features_and_target
from trainer.model import build_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a telco churn classifier.")
    parser.add_argument(
        "--data-path",
        default=os.environ.get("AIP_TRAINING_DATA_URI", "data/raw/Telco-Customer-Churn.csv"),
        help="Path to the Telco Customer Churn CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("AIP_MODEL_DIR", "artifacts/model"),
        help="Directory where model.joblib and metrics.json are written.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Estimator used inside the sklearn pipeline.",
    )
    return parser.parse_args()


def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_and_export(data_path: str | Path, model_dir: str | Path, model_type: str) -> dict[str, float]:
    raw = load_raw_csv(data_path)
    cleaned = clean_churn_frame(raw)
    features, target = split_features_and_target(cleaned)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    pipeline = build_pipeline(model_type=model_type)
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = evaluate(y_test, predictions, probabilities)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_and_export(args.data_path, args.model_dir, args.model_type)
    print(json.dumps({"status": "ok", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
