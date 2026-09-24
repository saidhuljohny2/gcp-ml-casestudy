# 07 — Cleanup (GCP)

Run this before students leave, even if the rest of the GCP lab is unfinished.

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
