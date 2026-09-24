"""Vertex AI compatible HTTP server for churn predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from trainer.predict import load_model, predict_instances

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/model.joblib"))
HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

app = FastAPI(title="Telco churn prediction", version="1.0.0")
model = load_model(MODEL_PATH)


async def health() -> dict[str, str]:
    """Report readiness after the model has loaded successfully."""
    return {"status": "healthy"}


async def predict(payload: dict[str, Any]) -> dict[str, list[dict[str, float | int]]]:
    """Validate Vertex's request envelope and return probability predictions."""
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise HTTPException(
            status_code=400,
            detail="'instances' must be a non-empty list of customer objects.",
        )

    parameters = payload.get("parameters") or {}
    threshold = parameters.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")

    try:
        predictions = predict_instances(model, instances, threshold=float(threshold))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"predictions": predictions}


app.add_api_route(HEALTH_ROUTE, health, methods=["GET"])
app.add_api_route(PREDICT_ROUTE, predict, methods=["POST"])
