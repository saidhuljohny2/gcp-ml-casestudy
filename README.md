# Telco Churn on Google Cloud (local first)

Hands-on workshop: train a **customer churn** classifier on a laptop, then later deploy the same code on **Vertex AI**.

This is a **6–8 hour beginner workshop**. Every step is manual. There is no Terraform, Cloud Build, Vertex Pipelines, or CI/CD in this version.

## What students build

A telecom company wants to know **which customers are likely to leave**. You will:

1. Inspect a public fictional churn dataset.
2. Train a scikit-learn pipeline locally.
3. Evaluate precision, recall, F1, and ROC-AUC against a business threshold.
4. Score sample customers from a saved `model.joblib`.
5. (Later) Upload data, run custom training, deploy an endpoint, and test it on Google Cloud.

```text
CSV --> pandas EDA --> sklearn Pipeline --> artifacts/model/model.joblib
                                              |
                                              +--> local prediction tests
                                              +--> (next) Vertex AI training + endpoint
```

## Workshop order

| Module | When | Guide |
| --- | --- | --- |
| Local setup | Now | [docs/01-local-setup.md](docs/01-local-setup.md) |
| Data and business problem | Now | [docs/02-data-and-problem.md](docs/02-data-and-problem.md) |
| EDA notebook | Now | [notebooks/01_eda_and_data_quality.ipynb](notebooks/01_eda_and_data_quality.ipynb) |
| Train and evaluate | Now | [notebooks/02_train_and_evaluate.ipynb](notebooks/02_train_and_evaluate.ipynb) |
| Local scoring tests | Now | [notebooks/03_local_prediction_tests.ipynb](notebooks/03_local_prediction_tests.ipynb) |
| GCP project setup | Deployed | [docs/03-gcp-setup.md](docs/03-gcp-setup.md) |
| Vertex training | Later | [docs/04-vertex-training.md](docs/04-vertex-training.md) |
| Deploy and test | Live | [docs/05-deploy-and-test.md](docs/05-deploy-and-test.md) |
| Monitoring | Later | [docs/06-monitoring-and-responsible-ml.md](docs/06-monitoring-and-responsible-ml.md) |
| Cleanup | Later | [docs/07-cleanup.md](docs/07-cleanup.md) |
| Instructor notes | Teaching | [docs/00-instructor-guide.md](docs/00-instructor-guide.md) |

## Quick start (local)

Use Python 3.10 or 3.11 (not 3.14):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest
python -m trainer.task --data-path data/raw/Telco-Customer-Churn.csv --model-dir artifacts/model
```

Then open the notebooks with `jupyter notebook` or VS Code / Cursor.

## Dataset

IBM **Telco Customer Churn** (fictional). Attribution and download notes are in [data/README.md](data/README.md).

## What we are not doing yet

- Vertex AI Pipelines
- Automated Cloud Build triggers / GitHub Actions
- Terraform
- Automated retraining

Those belong in a follow-on MLOps module after students can explain the manual path.

## Current live deployment

The custom prediction container is deployed in project
`gcp-evening-batch-501811`, region `us-central1`. See
[docs/05-deploy-and-test.md](docs/05-deploy-and-test.md) for the resource IDs
and tested request, and [docs/07-cleanup.md](docs/07-cleanup.md) to stop endpoint
costs.
