# 05 — Deploy and test (later)

Only after the model exists in Model Registry.

## Deploy

1. Open the registered model → Deploy to endpoint.
2. New endpoint name: `churn-endpoint`.
3. Machine: `n1-standard-2` or `n1-standard-4`.
4. Min / max replicas: **1 / 1** for class (cost control).
5. Wait until the endpoint shows the model at 100% traffic.

## Prediction contract

Send the same 19 feature columns used in `src/trainer/data.py`. Example instance:

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

Prebuilt sklearn serving expects a list of instances (arrays or objects depending on how the model was saved). If object dicts fail, send values in `FEATURE_COLUMNS` order as a list. Practice this mismatch in class; it is the most common deploy bug.

## Tests to run live

- High-risk month-to-month customer → higher churn probability.
- Long-tenure two-year contract → lower probability.
- Missing field → error (good; do not hide it).
- Local `notebooks/03_local_prediction_tests.ipynb` already defined the expected response shape: `churn_probability` and `predicted_churn`. The raw Vertex response may only return class / probability arrays; map them in a notebook.

## REST

Use the Cloud Console “Test your model” panel first. Then optionally `gcloud ai endpoints predict` with a JSON file. Do not build a public unauthenticated API in class.
