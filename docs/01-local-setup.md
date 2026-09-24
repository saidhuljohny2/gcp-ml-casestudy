# 01 — Local setup

Do this on your laptop before any Google Cloud work.

## Prerequisites

- Python **3.10 or 3.11** (avoid 3.14; scikit-learn wheels may not exist yet)
- Git
- Ability to create a virtual environment

Confirm:

```bash
python3.10 --version
```

## Install

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` installs the `trainer` package from `src/` so notebooks and tests import `from trainer ...`.

## Dataset

```bash
python scripts/download_data.py
```

You should see `data/raw/Telco-Customer-Churn.csv` with a header plus 7,043 rows.

## Sanity checks

```bash
pytest
python -m trainer.task --data-path data/raw/Telco-Customer-Churn.csv --model-dir artifacts/model
```

Expected:

- Tests pass.
- `artifacts/model/model.joblib` is written.
- `artifacts/model/metrics.json` contains accuracy, precision, recall, f1, roc_auc.

## Open notebooks

```bash
jupyter notebook notebooks
```

Or open the `.ipynb` files in Cursor / VS Code and select the `.venv` kernel.

## What “done” looks like

You can load the CSV, explain the target column `Churn`, and train a model without using the Google Cloud console.
