FROM python:3.11-slim AS trainer

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/raw/Telco-Customer-Churn.csv ./data/raw/Telco-Customer-Churn.csv
RUN pip install --no-cache-dir .
RUN python -m trainer.task \
    --data-path data/raw/Telco-Customer-Churn.csv \
    --model-dir /model \
    --model-type logistic

FROM python:3.11-slim AS serving

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/model.joblib

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[serving]"
COPY --from=trainer /model/model.joblib ./model/model.joblib
COPY --from=trainer /model/metrics.json ./model/metrics.json

EXPOSE 8080
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
