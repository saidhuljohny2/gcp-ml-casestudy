# 03 — Google Cloud setup (after the local model works)

Complete [01-local-setup.md](01-local-setup.md) and the three notebooks first.

## Goal

Create a project that can store data, run Vertex AI training, and host an endpoint.

## Manual console steps

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project. Record `PROJECT_ID`.
3. Confirm billing is enabled. Training and endpoints are not free.
4. Choose one region and stick to it, for example `us-central1`.
5. Enable APIs:
   - Vertex AI
   - Cloud Storage
   - BigQuery
   - Compute Engine (Workbench VMs)
6. Create a Cloud Storage bucket, for example `gs://PROJECT_ID-churn-workshop`.
7. Create a BigQuery dataset, for example `churn_workshop`.
8. (Optional) Create a Vertex AI Workbench instance. Students can also keep using local notebooks and only use the console for jobs.

## IAM (keep it simple for class)

Students typically need:

- Vertex AI User
- Storage Admin (or a tighter custom role on one bucket)
- BigQuery Job User + Data Editor on the workshop dataset

Do not grant Owner unless this is a throwaway student project.

## Ingest the CSV

1. In Cloud Storage, upload `data/raw/Telco-Customer-Churn.csv`.
2. In BigQuery, create table `churn_workshop.telco_customers` from that URI.
3. Run:

```sql
SELECT Churn, COUNT(*) AS n
FROM `PROJECT_ID.churn_workshop.telco_customers`
GROUP BY 1;
```

You should see the same ~26.5% churn rate as locally.

## Cost warning

The expensive part is usually a **deployed endpoint** left running. Plan to undeploy at the end of the GCP block. See [07-cleanup.md](07-cleanup.md).
