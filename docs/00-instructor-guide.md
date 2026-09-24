# Instructor guide

## Audience

Students who know basic Python and pandas. They do **not** need prior GCP experience for the local half. GCP is a second day / second half.

## Timing (6–8 hours)

| Block | Minutes | Goal |
| --- | --- | --- |
| 0. Context | 20 | Why churn is a real business problem |
| 1. Local setup | 20 | venv, install, download CSV |
| 2. Data quality | 50 | Schema, missing TotalCharges, leakage |
| 3. EDA | 50 | Churn rate, contract type, tenure |
| 4. Baseline model | 70 | Dummy vs logistic vs random forest |
| 5. Metrics and cost | 40 | Precision/recall vs retention offer cost |
| 6. Local scoring | 30 | `model.joblib` contract tests |
| 7. (Later) GCP setup | 40 | Project, APIs, bucket, BigQuery |
| 8. (Later) Vertex train | 50 | Custom job, logs, Model Registry |
| 9. (Later) Deploy + test | 40 | Endpoint, REST, smoke tests |
| 10. (Later) Monitor + cleanup | 30 | Drift concept, cost teardown |

If you only have a 3-hour slot, stop after local scoring.

## Teaching style

- Students type commands and inspect outputs. Do not skip to a finished notebook.
- Pause after every metric and ask: *would we rather miss a churner or annoy a loyal customer?*
- Keep class_weight="balanced" as a discussion, not a magic default.

## Checkpoints

1. `pytest` is green.
2. EDA shows ~26.5% churn and blank TotalCharges for tenure=0.
3. Local `metrics.json` has ROC-AUC clearly above 0.5 (expect ~0.82–0.85 for logistic).
4. Prediction JSON includes all 19 feature columns.
5. (Later) Endpoint returns a probability for a known high-risk month-to-month customer.

## Common mistakes

- Treating `TotalCharges` as a string and failing the pipeline.
- Including `customerID` as a feature.
- Using accuracy alone on an imbalanced target.
- Training on the full dataset then reporting test metrics from the same rows.
- For GCP: leaving an endpoint deployed overnight (largest surprise cost).

## Reset points

- Local: delete `artifacts/` and re-run `python -m trainer.task`.
- GCP: follow [07-cleanup.md](07-cleanup.md) then recreate the bucket/dataset.
