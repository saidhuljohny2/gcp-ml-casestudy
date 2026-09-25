# 07 — Cleanup (GCP)

Run this before students leave, even if the rest of the GCP lab is unfinished.

## Exact cleanup for the current deployment

The endpoint replica is the main ongoing cost. Undeploy it first:

```bash
export CLOUDSDK_CONFIG="$PWD/.gcloud"
GCLOUD=.tools/google-cloud-sdk/bin/gcloud

$GCLOUD ai endpoints undeploy-model 8612769799241465856 \
  --region=us-central1 \
  --deployed-model-id=2276917607065976832

$GCLOUD ai endpoints delete 8612769799241465856 \
  --region=us-central1 --quiet

$GCLOUD ai models delete 2319679813293441024 \
  --region=us-central1 --quiet

$GCLOUD artifacts docker images delete \
  us-central1-docker.pkg.dev/gcp-evening-batch-501811/churn-models/telco-churn:v1 \
  --delete-tags --quiet
```

Optionally delete the empty repository:

```bash
$GCLOUD artifacts repositories delete churn-models \
  --location=us-central1 --quiet
```

## General workshop cleanup

1. Vertex AI → Endpoints → Undeploy model → Delete endpoint.
2. Stop or delete Workbench / notebook VMs.
3. Delete custom training jobs only if you do not need logs; jobs stop computing when they finish.
4. Optional: delete the Model Registry entry.
5. Empty and delete the workshop Cloud Storage bucket.
6. Delete the BigQuery dataset `churn_workshop`.
7. Optional: shut down the entire student project.

Local files to reset:

```bash
rm -rf artifacts
```

The CSV in `data/raw/` can stay; it is not billed.
