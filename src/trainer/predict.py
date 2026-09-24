"""Load a saved pipeline and score one or more customer records."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from trainer.data import FEATURE_COLUMNS


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def instances_to_frame(instances: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(instances)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction payload is missing columns: {missing}")
    return frame[FEATURE_COLUMNS]


def predict_instances(model, instances: list[dict], threshold: float = 0.5) -> list[dict]:
    features = instances_to_frame(instances)
    probabilities = model.predict_proba(features)[:, 1]
    results = []
    for probability in probabilities:
        results.append(
            {
                "churn_probability": float(probability),
                "predicted_churn": int(probability >= threshold),
            }
        )
    return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score JSON instances with a saved churn model.")
    parser.add_argument("--model-path", default="artifacts/model/model.joblib")
    parser.add_argument("--instances-path", default="artifacts/sample_instance.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.instances_path).read_text())
    instances = payload if isinstance(payload, list) else [payload]
    results = predict_instances(load_model(args.model_path), instances)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
