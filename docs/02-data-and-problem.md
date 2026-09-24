# 02 — Data and business problem

## Business story

A fictional California telecom offers phone and internet service. About one in four customers left last month. Retention offers (discounts, better support) cost money, so the company wants a model that ranks **who is likely to churn** before they leave.

Target:

- `Churn = Yes` → customer left
- `Churn = No` → customer stayed

This is **binary classification**.

## Why this dataset is a good teaching set

- Tabular and small (7,043 rows): trains in seconds on a laptop.
- Mix of numeric and categorical features.
- Real data-quality issue: `TotalCharges` is stored as text and is blank when `tenure = 0`.
- Clear business trade-off: missing a churner vs offering a discount to a loyal customer.

## Columns students should understand

| Column | Role |
| --- | --- |
| `customerID` | Identifier. **Never a model feature.** |
| `tenure` | Months as a customer |
| `Contract` | Month-to-month / one year / two year |
| `InternetService`, add-on flags | Product mix |
| `MonthlyCharges`, `TotalCharges` | Billing |
| `PaymentMethod` | Electronic check often correlates with churn |
| `Churn` | Label |

## Leakage and assumptions (discuss before modeling)

- This snapshot does **not** include a timestamp. You cannot do a true time-based backtest.
- `TotalCharges` is roughly `tenure * MonthlyCharges` with discounts/taxes. It is allowed as a feature, but students should notice the correlation with tenure.
- Do not use future information such as “churn reason” fields from other IBM sample files. This CSV does not include them.

## Load into pandas (local)

```python
import pandas as pd

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
df.shape
df.dtypes
df["Churn"].value_counts(normalize=True)
```

Expected churn rate is about **26.5% Yes**.

## Cleaning rule you will reuse in training code

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

The same logic lives in `src/trainer/data.py` so notebooks and Vertex jobs stay consistent.

## Later on GCP (not now)

The same CSV will be uploaded to Cloud Storage and loaded into BigQuery. Schema and cleaning do not change.
