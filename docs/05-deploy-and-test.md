# 05 — Deploy and test

The first live deployment uses a custom FastAPI container. This preserves named
JSON features and returns both probability and class predictions.

## Deployed workshop resources

- Project: `gcp-evening-batch-501811`
- Region: `us-central1`
- Image: `us-central1-docker.pkg.dev/gcp-evening-batch-501811/churn-models/telco-churn:v1`
- Model ID: `2319679813293441024`
- Endpoint ID: `8612769799241465856`
- Deployed model ID: `2276917607065976832`
- Machine: `n1-standard-2`, one replica

The exact resources are also recorded in `artifacts/deployment.env`.

## How it was deployed manually

1. `gcloud builds submit` built the multi-stage `Dockerfile`.
2. The build trained the logistic pipeline and copied `model.joblib` into the
   final serving image.
3. The image was pushed to Artifact Registry.
4. The custom container was registered in Vertex AI Model Registry with
   `/health` and `/predict` routes.
5. The model was deployed to the endpoint at 100% traffic.

## Prediction contract

Send the same 19 feature columns used in `src/trainer/data.py`. The complete
request is in `artifacts/vertex_request.json`:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

The request envelope contains `instances` and an optional
`parameters.threshold` from 0 to 1. The custom server validates missing fields.

## Live smoke test

```bash
export CLOUDSDK_CONFIG="$PWD/.gcloud"
GCLOUD=.tools/google-cloud-sdk/bin/gcloud

$GCLOUD ai endpoints predict 8612769799241465856 \
  --region=us-central1 \
  --json-request=artifacts/vertex_request.json
```

Validated response for the sample high-risk customer:

```text
[{'churn_probability': 0.8064134154783931, 'predicted_churn': 1}]
```

Further class tests:

- Long-tenure two-year customer → lower probability.
- Missing feature → HTTP 400 validation error.
- Multiple objects in `instances` → same-length predictions list.
- Change `parameters.threshold` and verify that the class can change while the
  probability remains constant.

## REST

The endpoint requires Google Cloud authentication. Do not make it public for
the workshop.
