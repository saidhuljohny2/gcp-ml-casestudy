# 04 — Vertex AI custom training (later)

Use the **same** `src/trainer` package you already ran locally.

## Why custom training (not AutoML) in this workshop

Students should see the exact sklearn pipeline, `model.joblib` contract, and logs. AutoML hides those steps.

## Package the trainer

From the repo root, after local tests pass:

```bash
python -m pip install build
python -m build --sdist
```

Upload `dist/telco_churn_trainer-0.1.0.tar.gz` to `gs://PROJECT_ID-churn-workshop/trainer/`.

Also upload the CSV to `gs://PROJECT_ID-churn-workshop/data/Telco-Customer-Churn.csv`.

## Console job (prebuilt sklearn container)

1. Vertex AI → Training → Custom jobs → Create.
2. Region: `us-central1`.
3. Worker pool 0:
   - Pre-built container: scikit-learn (match the version in `requirements.txt`, 1.5.x).
   - Package URI: the `.tar.gz` in Cloud Storage.
   - Python module: `trainer.task`.
   - Arguments:
     - `--data-path=gs://PROJECT_ID-churn-workshop/data/Telco-Customer-Churn.csv`
     - `--model-type=logistic`
4. Leave model directory unset so Vertex injects `AIP_MODEL_DIR`.
5. Machine type: `n1-standard-4` is enough.
6. Submit and open logs.

`trainer.task` already writes `model.joblib` to `AIP_MODEL_DIR` (or `--model-dir` locally).

If the training container cannot read `gs://` with pandas, copy the CSV into the job using a Cloud Storage download in a follow-up, or run training from a Workbench notebook that downloads the file first. The local CLI remains the source of truth.

## Register the model

After the job succeeds:

1. Find artifacts under the job’s output bucket / `model/` folder.
2. Vertex AI → Model Registry → Import.
3. Pre-built **sklearn** prediction container, same major.minor as training.
4. Artifact URI: the folder that contains `model.joblib` (filename must be exactly `model.joblib`).

## Notebook alternative

Students can also call `CustomTrainingJob` from a Workbench notebook after `pip install google-cloud-aiplatform`. Keep that optional so the console path stays visible.
