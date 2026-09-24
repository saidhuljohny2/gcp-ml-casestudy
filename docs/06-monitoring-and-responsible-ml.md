# 06 — Monitoring and responsible ML (later)

## What to show in the console

- Vertex endpoint request count and latency.
- Cloud Logging for prediction errors.
- Concept of **training-serving skew** (missing category, different TotalCharges typing).
- Optional: create a Model Monitoring job with a 24h window. Do not wait for a real alert in a one-day class.

## Responsible use

- `gender` and `SeniorCitizen` can correlate with outcomes. Discuss whether the business should use them.
- A false negative (missed churn) loses revenue. A false positive (loyal customer scored as churn) may trigger an unnecessary discount.
- This dataset is fictional; production churn systems need consent, retention policy, and appeal paths.

## Retrain when

- Churn rate in production drifts far from ~26.5%.
- A new product (for example 5G home) appears as an unseen category.
- Offer policy changes so the operating threshold must move.

Local `metrics.json` is the baseline you compare against later cloud metrics.
