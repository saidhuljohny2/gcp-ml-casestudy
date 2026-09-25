# Teaching guide: data, business, and the three notebooks

Use this file when you explain the workshop. It is written for **you**, the instructor. Students still run the notebooks themselves.

Related files:

- Data: [`data/raw/Telco-Customer-Churn.csv`](../data/raw/Telco-Customer-Churn.csv)
- Shared training code: [`src/trainer/data.py`](../src/trainer/data.py), [`src/trainer/model.py`](../src/trainer/model.py), [`src/trainer/task.py`](../src/trainer/task.py), [`src/trainer/predict.py`](../src/trainer/predict.py)
- Notebooks: [`notebooks/01_eda_and_data_quality.ipynb`](../notebooks/01_eda_and_data_quality.ipynb), [`notebooks/02_train_and_evaluate.ipynb`](../notebooks/02_train_and_evaluate.ipynb), [`notebooks/03_local_prediction_tests.ipynb`](../notebooks/03_local_prediction_tests.ipynb)

---

## 1. Business problem in one minute

A fictional California telecom sells **home phone and internet**. Last month, **1,869 of 7,043 customers left** (about **26.5%**). Leaving is called **churn**.

The company can try to keep people: a discount, a better plan, a support call. Those actions cost money. If you call everyone, you waste budget on loyal customers. If you call nobody, you lose revenue when people leave.

The model’s job is **not** to “know the future.” It is to **rank customers by risk of leaving soon**, using information the company already has: who they are, what they buy, how they pay, how long they have stayed, and what they are billed.

This is **binary classification**:

| Label in CSV | Meaning | Number in model |
| --- | --- | --- |
| `Churn = Yes` | Left last month | `1` (positive class) |
| `Churn = No` | Still a customer | `0` |

**Say this to students:** we care more about catching leavers than about looking accurate. A model that always says “No” is already 73.5% accurate and is useless for retention.

---

## 2. What this dataset is (and is not)

| Fact | Detail |
| --- | --- |
| Name | IBM Telco Customer Churn (sample / fictional) |
| Rows | 7,043 customers, one row per customer |
| Columns | 21 (20 attributes + `Churn`) |
| Time | One snapshot. There is **no date column**. You cannot do a true time-based backtest. |
| PII | Fictional. Treat it as production-like in class, but it is not real people. |
| File | `data/raw/Telco-Customer-Churn.csv` |

Source: [IBM telco churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

---

## 3. How to think about the columns

Group columns the way a business person would, then map each group to ML.

```text
Who they are          Product they buy         How they pay           Outcome
----------------      ----------------         -----------            -------
customerID            PhoneService             Contract               Churn
gender                MultipleLines            PaperlessBilling
SeniorCitizen         InternetService          PaymentMethod
Partner               add-on flags             MonthlyCharges
Dependents            (security, TV, ...)      TotalCharges
                      tenure
```

**Feature policy in this workshop** (see `FEATURE_COLUMNS` in `src/trainer/data.py`):

- **Never use** `customerID` as a feature.
- **Use** 19 columns: 4 numeric + 15 categorical.
- **Target** is `Churn` only. Do not invent extra labels.

Dependent values you must explain:

- If `PhoneService = No`, then `MultipleLines` is always `No phone service` (682 rows).
- If `InternetService = No`, then security / backup / protection / support / streaming columns are `No internet service` (1,526 rows). Those are **not** missing data. They mean “does not apply.”

---

## 4. Column-by-column dictionary

Counts below are from the CSV in this repo (7,043 rows).

### 4.1 Identity

#### `customerID`

| | |
| --- | --- |
| Raw type | String, for example `7590-VHVEG` |
| Unique | Yes. One ID per row. |
| Business | Account number. Used to join CRM, billing, tickets. |
| Modeling | **Drop.** IDs are unique noise. If the model “learns” an ID, it will not generalize. |
| Teaching line | “If you can look a customer up by this field, it is not a predictor.” |

### 4.2 Household / demographics

These describe the person or household. They can correlate with churn. They can also raise **fairness** questions. Discuss whether the business *should* use them.

#### `gender`

| | |
| --- | --- |
| Values | `Female` 3,488 · `Male` 3,555 |
| Business | Marketing segment. Weak churn signal in this file. |
| Modeling | Categorical. One-hot encoded. |
| Teaching line | “Included because it is in the CRM extract. In production, legal and fairness review may drop it.” |

#### `SeniorCitizen`

| | |
| --- | --- |
| Values | `0` = not senior, 5,901 · `1` = senior, 1,142 |
| Churn rate | Non-senior **23.6%** · senior **41.7%** |
| Raw type | Integer, but it is a **flag**, not a continuous number. |
| Modeling | Treated as **numeric** in our pipeline (0/1). That is valid for a binary flag. |
| Teaching line | “Seniors churn more here. That can be age, plan mix, or income. Do not claim the column *causes* churn.” |

#### `Partner`

| | |
| --- | --- |
| Values | `Yes` 3,402 · `No` 3,641 |
| Business | Household with a spouse/partner. Often more stable billing. |
| Modeling | Categorical Yes/No. |

#### `Dependents`

| | |
| --- | --- |
| Values | `No` 4,933 · `Yes` 2,110 |
| Business | Children or others on the account. Switching cost can be higher. |
| Modeling | Categorical Yes/No. |

### 4.3 Loyalty / time with the company

#### `tenure`

| | |
| --- | --- |
| Type | Integer months, **0 to 72** |
| Business | How long they have been a customer. The strongest simple loyalty signal. |
| Observed churn | First 6 months **~53%** · 49–72 months **~9.5%** |
| Special case | `tenure = 0` means brand-new, not yet billed. Matches **11 blank `TotalCharges`**. |
| Modeling | Numeric. Scaled. |
| Teaching line | “New customers are the retention emergency. Two-year veterans rarely leave next month.” |

### 4.4 Phone product

#### `PhoneService`

| | |
| --- | --- |
| Values | `Yes` 6,361 · `No` 682 |
| Business | Landline / voice on this account. |
| Modeling | Categorical. |

#### `MultipleLines`

| | |
| --- | --- |
| Values | `No` 3,390 · `Yes` 2,971 · `No phone service` 682 |
| Business | Extra voice lines. `No phone service` is **not** the same as `No`. |
| Modeling | Categorical with three levels. Do not recode `No phone service` to `No`. |

### 4.5 Internet product and add-ons

#### `InternetService`

| | |
| --- | --- |
| Values | `Fiber optic` 3,096 · `DSL` 2,421 · `No` 1,526 |
| Churn rate | Fiber **41.9%** · DSL **19.0%** · no internet **7.4%** |
| Business | Access technology. Fiber is often more expensive and more competitive. |
| Teaching line | “Fiber customers churn a lot in this sample. That may be price, competition, or who buys fiber — not ‘fiber is bad.’” |

The next six columns are **add-on products**. Each can be `Yes`, `No`, or `No internet service`.

| Column | What the business sells | Yes | No | No internet service |
| --- | --- | --- | --- | --- |
| `OnlineSecurity` | Extra security suite | 2,019 | 3,498 | 1,526 |
| `OnlineBackup` | Cloud backup | 2,429 | 3,088 | 1,526 |
| `DeviceProtection` | Device insurance / protection | 2,422 | 3,095 | 1,526 |
| `TechSupport` | Paid support | 2,044 | 3,473 | 1,526 |
| `StreamingTV` | TV streaming bundle | 2,707 | 2,810 | 1,526 |
| `StreamingMovies` | Movie streaming bundle | 2,732 | 2,785 | 1,526 |

**Teaching line:** add-ons are stickiness and margin. “No internet service” means the add-on question was never asked. Leave the third category; do not treat it as missing.

### 4.6 Contract and billing behavior

#### `Contract`

| | |
| --- | --- |
| Values | `Month-to-month` 3,875 · `Two year` 1,695 · `One year` 1,473 |
| Churn rate | Month-to-month **42.7%** · one year **11.3%** · two year **2.8%** |
| Business | How hard it is to leave. Month-to-month is the default high-risk segment. |
| Teaching line | “This is the first chart a retention VP would want. Locked contracts churn less because leaving is expensive.” |

#### `PaperlessBilling`

| | |
| --- | --- |
| Values | `Yes` 4,171 · `No` 2,872 |
| Business | Email invoices vs paper. Digital billing often correlates with digital-native, more mobile customers. |
| Modeling | Categorical. |

#### `PaymentMethod`

| | |
| --- | --- |
| Values | Electronic check 2,365 · mailed check 1,612 · bank transfer (automatic) 1,544 · credit card (automatic) 1,522 |
| Churn rate | Electronic check **45.3%** · mailed check **19.1%** · bank **16.7%** · card **15.2%** |
| Business | Autopay is sticky. Electronic check is often one-off, higher friction, higher churn. |
| Teaching line | “Payment method is a behavior, not just a finance field.” |

### 4.7 Money

#### `MonthlyCharges`

| | |
| --- | --- |
| Type | Float dollars per month, about **18.25 to 118.75** |
| Business | Recurring bill. Higher bills can mean more products *or* more price sensitivity. |
| Modeling | Numeric. Scaled. |

#### `TotalCharges`

| | |
| --- | --- |
| Raw type | **String** in the CSV (this is the data-quality lesson). |
| Meaning | Lifetime billed amount, roughly tenure × monthly, with promotions and taxes. |
| Problem | 11 blanks, all with `tenure = 0`. They have not received a bill yet. |
| Cleaning | `pd.to_numeric(..., errors="coerce")` then fill `0`. Same logic in `clean_churn_frame()`. |
| Modeling | Numeric. Correlated with `tenure`. Allowed, but do not pretend it is independent of tenure. |

### 4.8 Target

#### `Churn`

| | |
| --- | --- |
| Values | `No` 5,174 (73.46%) · `Yes` 1,869 (26.54%) |
| Business | Left within the last month (as defined by this sample). |
| Modeling | Mapped to 0/1. Positive class is **Yes**. |
| Leakage | This CSV has **no** `ChurnReason`. Good. A reason collected *after* they leave is not available at scoring time. |

---

## 5. What the model is allowed to know at prediction time

At scoring time, a customer has **not** left yet. You may only send fields CRM/billing would know **today**:

- identity flags, products, contract, payment, charges, tenure
- **not** “they called to cancel,” “churn reason,” or next month’s bill

That is why notebook 03 tests a JSON object with the 19 feature columns, the same contract Vertex uses.

---

## 6. Metrics you must be able to explain

Assume we predict `1` = will churn.

| Term | Meaning in this business |
| --- | --- |
| True positive | We flagged a leaver. Retention team can act. |
| False positive | We flagged a loyal customer. We may waste a discount. |
| False negative | We missed a leaver. Lost revenue. |
| True negative | We correctly left a loyal customer alone. |
| Accuracy | Overall % correct. Misleading when 73.5% are non-churn. |
| Precision | Of people we flag, how many actually leave. Cost of offers. |
| Recall | Of actual leavers, how many we catch. Cost of missed churn. |
| F1 | Balance of precision and recall. |
| ROC-AUC | Ranking quality across thresholds. Random is 0.5. Our logistic model is about **0.84**. |
| Threshold | Default 0.5 is **not** a business rule. Cheap offers → lower threshold (more recall). Expensive offers → higher threshold (more precision). |

---

## 7. Shared Python package (why notebooks import `trainer`)

Students should see this once before notebook 02:

| Module | Job |
| --- | --- |
| `trainer.data` | Load CSV, fix `TotalCharges`, map `Churn`, drop ID, define feature lists |
| `trainer.model` | `ColumnTransformer` + logistic or random forest |
| `trainer.task` | Train/test split, fit, metrics, write `model.joblib` |
| `trainer.predict` | Validate JSON columns and return probability + class |

Preprocessing inside `build_pipeline()`:

1. Numeric: median impute, then `StandardScaler`
2. Categorical: most-frequent impute, then `OneHotEncoder(handle_unknown="ignore")`
3. Classifier: `class_weight="balanced"` because 26.5% positives
4. `random_state=42` so class results match

The same pipeline is what Cloud Build trained into the Vertex image.

---

## 8. Notebook 01 — EDA and data quality

**File:** `notebooks/01_eda_and_data_quality.ipynb`  
**Goal:** trust the table before you train.  
**Kernel:** run from the `notebooks/` folder so `../data/raw/...` works.  
**Prerequisite:** `pip install -e ".[dev]"` is nice-to-have here; this notebook only needs pandas.

### Cell 0 (markdown) — title

Tell students: this notebook does **not** fit a model. If they skip EDA, they will mishandle `TotalCharges` and `customerID`.

### Cell 1 — load and shape

```python
from pathlib import Path
import pandas as pd

DATA_PATH = Path("../data/raw/Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)
df.shape
```

| What it does | Why |
| --- | --- |
| `Path` | Avoids OS-specific path strings |
| `read_csv` | One row per customer |
| `.shape` | Confirm **(7043, 21)** |

**Ask:** “If shape is (0, n) or FileNotFoundError, what did you forget?” Answer: run `python scripts/download_data.py` from repo root, or wrong working directory.

### Cell 2 — `df.head()`

Show the first rows. Point to mixed types: IDs, Yes/No, numbers.

**Ask:** “Can you predict churn from row 0 without a model?” Month-to-month, tenure 1, electronic check is intuitively risky. That is the intuition the model should recover.

### Cell 3 — `df.dtypes`

Expected teaching points:

- `TotalCharges` is **object** (string), not float.
- `SeniorCitizen` is int.
- `tenure` is int.
- `MonthlyCharges` is float.
- Almost everything else is object (categories).

**Say:** pandas guessed types from the file. Blank `TotalCharges` forced the whole column to string. That is why later `to_numeric` is required.

### Cell 4 (markdown) — target balance

Set up why accuracy is a trap.

### Cell 5 — value counts

```python
df["Churn"].value_counts()
df["Churn"].value_counts(normalize=True)
```

Expect about 5174 / 1869 and **0.735 / 0.265**.

Only the **last** expression prints in Jupyter unless they `print()` both. If students only see percentages, that is normal. Have them `print(df["Churn"].value_counts())` if you want both.

### Cell 6–7 — blank TotalCharges

```python
blank_total = df["TotalCharges"].astype(str).str.strip().eq("")
print("blank TotalCharges rows:", int(blank_total.sum()))
df.loc[blank_total, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
```

Expect **11** blanks, all `tenure = 0`.

**Walk the logic:**

1. Force string so blanks are visible (`NaN` vs `" "`).
2. `str.strip()` catches space-only cells.
3. Show billing columns together so the story is obvious: new customer, no invoice yet.

**Wrong student fix:** drop the 11 rows. Better: fill 0, because new customers still exist in production.

### Cell 8 — clean copy

```python
clean = df.copy()
clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
print(clean["TotalCharges"].isna().sum())
clean["TotalCharges"] = clean["TotalCharges"].fillna(0)
clean["ChurnFlag"] = clean["Churn"].map({"Yes": 1, "No": 0})
```

| Line | Meaning |
| --- | --- |
| `.copy()` | Do not overwrite the raw frame while exploring |
| `to_numeric(..., errors="coerce")` | Bad strings become NaN instead of crashing |
| Print NA count | Should equal 11 before fillna |
| `fillna(0)` | New customers have $0 billed |
| `ChurnFlag` | Numeric 0/1 so `groupby().mean()` is a churn **rate** |

This cell is the **manual version** of `clean_churn_frame()` (notebook 01 does not import `trainer` so EDA stays readable).

### Cell 9 (markdown) — business slices

Tell them we now answer: *who leaves?*

### Cell 10 — churn by contract

```python
clean.groupby("Contract")["ChurnFlag"].mean().sort_values(ascending=False)
```

Expect approximately:

- Month-to-month **0.427**
- One year **0.113**
- Two year **0.028**

**Classroom sentence:** “If you only had one rule, you would watch month-to-month accounts.”

### Cell 11 — churn by payment method

Expect electronic check **~0.453**, other methods **~0.15–0.19**.

**Classroom sentence:** “Autopay customers are harder to lose. Electronic check is a red flag, not a cause by itself.”

### Cell 12 — churn by tenure bucket

```python
clean.groupby(pd.cut(clean["tenure"], bins=[-0.1, 6, 12, 24, 48, 72]))["ChurnFlag"].mean()
```

`pd.cut` puts each tenure into a bin. `-0.1` includes tenure 0.

Expect a **downward** staircase: newest customers highest churn, oldest lowest.

### Cell 13–14 — leakage check

```python
assert "ChurnReason" not in clean.columns
print("identifier unique:", clean["customerID"].nunique() == len(clean))
print("columns:", list(clean.columns))
```

| Check | Pass means |
| --- | --- |
| No `ChurnReason` | We cannot cheat with post-churn text |
| Unique IDs | One row = one customer |
| Column list | Students see `ChurnFlag` was added only in this notebook |

**Checkpoint before notebook 02:** students can explain target imbalance, the 11 blanks, contract/tenure patterns, and why ID is dropped.

---

## 9. Notebook 02 — train and evaluate

**File:** `notebooks/02_train_and_evaluate.ipynb`  
**Goal:** beat a dummy baseline, compare two models, talk threshold, save `model.joblib`.  
**Prerequisite:** package installed so `from trainer...` works. Working directory `notebooks/`.

### Cell 0 (markdown)

Same code will later run on Vertex. Local and cloud must not drift.

### Cell 1 — load, clean, split

```python
from trainer.data import clean_churn_frame, load_raw_csv, split_features_and_target
from trainer.model import build_pipeline
from trainer.task import evaluate
# ...
raw = load_raw_csv(Path("../data/raw/Telco-Customer-Churn.csv"))
clean = clean_churn_frame(raw)
X, y = split_features_and_target(clean)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_train.mean(), y_test.mean()
```

Walk this slowly:

1. `load_raw_csv` — same CSV as notebook 01.
2. `clean_churn_frame` — numeric TotalCharges, Churn 0/1, schema check.
3. `split_features_and_target` — `X` has 19 columns, **no** `customerID`, `y` is 0/1.
4. `train_test_split`
   - `test_size=0.2` → about 1,409 test rows
   - `random_state=42` → reproducible
   - `stratify=y` → train and test both keep ~26.5% churn
5. `y_train.mean(), y_test.mean()` should both be **~0.265**. If they differ a lot, stratify was forgotten.

**Why not train on all rows?** Test metrics would be cheating. Students must never evaluate on rows the model already saw.

### Cell 2–3 — dummy baseline

```python
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(classification_report(y_test, dummy_pred, zero_division=0))
```

`most_frequent` always predicts **0** (stay).

Expect:

- Accuracy ~**0.73**
- Recall for class 1 = **0**
- Precision for class 1 = **0** (`zero_division=0` avoids warnings)

**Say:** “This is the score to beat. If your fancy model is only 74% accurate, you have not helped the business.”

### Cell 4 — logistic vs random forest

```python
results = {}
for name in ["logistic", "random_forest"]:
    pipe = build_pipeline(name)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {...}
```

| Step | Meaning |
| --- | --- |
| `build_pipeline(name)` | Preprocess + classifier |
| `fit` | Learn weights / trees **only on train** |
| `predict_proba(...)[:, 1]` | Probability of churn (class 1), not the 0/1 label |
| `>= 0.5` | Default cutoff we will challenge next |
| `evaluate(...)` | accuracy, precision, recall, f1, roc_auc |

**Expected ballpark** (logistic, held-out test, class_weight balanced):

- accuracy ~0.74 (can look similar to dummy!)
- precision ~0.50
- recall ~0.78
- f1 ~0.61
- roc_auc ~0.84

**Must say:** dummy also has ~0.74 accuracy. The win is **recall and ROC-AUC**, not accuracy. Logistic with balanced weights **flags more leavers**, so more false positives too. That is a business choice.

Random forest in this pipeline is a shallow forest (`max_depth=8`, `min_samples_leaf=10`) so it does not overfit a 7k-row table as easily as an unconstrained forest.

### Cell 5–6 — confusion matrix and ROC

Left plot: counts of TP / FP / FN / TN at threshold 0.5.

Right plot: ROC. The diagonal is a coin flip. A curve toward the top-left is better ranking. AUC ~0.84 means if you pick a random leaver and a random stayer, the model scores the leaver higher about 84% of the time.

**Discussion prompt:** “Is a missed churner worse than a wasted discount?” For a high-ARPU fiber customer, missed churn is usually worse → accept more false positives.

### Cell 7–8 — precision-recall and threshold

```python
precision, recall, thresholds = precision_recall_curve(y_test, results["logistic"]["proba"])
```

The PR curve shows the trade-off **without** picking 0.5.

**Teaching script:**

- Move threshold **down** (for example 0.3): more customers flagged, recall up, precision down, more call-center load.
- Move threshold **up** (for example 0.7): fewer flags, cleaner list, more missed leavers.

0.5 is a programming default, not a CFO default.

### Cell 9 — export the production artifact

```python
from trainer.task import train_and_export

metrics = train_and_export(
    Path("../data/raw/Telco-Customer-Churn.csv"),
    Path("../artifacts/model"),
    "logistic",
)
```

This **retrains** with the same split rules as `trainer.task` (`random_state=42`, 20% test) and writes:

- `artifacts/model/model.joblib` — full sklearn Pipeline
- `artifacts/model/metrics.json` — test metrics

Why call `train_and_export` instead of `joblib.dump(results["logistic"]["model"])`? So the file on disk is produced by the **same function** Cloud / CLI uses. Notebook 03 and Vertex must load that contract.

**Checkpoint:** ROC-AUC clearly above 0.5; `model.joblib` exists.

---

## 10. Notebook 03 — local prediction tests

**File:** `notebooks/03_local_prediction_tests.ipynb`  
**Goal:** prove the saved model scores JSON like an API.  
**Prerequisite:** notebook 02 cell 9 (or `python -m trainer.task ...`) already created `artifacts/model/model.joblib`.

If `FileNotFoundError` on `model.joblib`, they skipped export.

### Cell 0 (markdown)

Local scoring is the dress rehearsal for Vertex `instances: [...]`.

### Cell 1 — load model and feature list

```python
MODEL_PATH = Path("../artifacts/model/model.joblib")
model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS
```

Show the 19 names. **This is the API contract.** Vertex requests missing any of these fail.

`joblib.load` loads the **entire pipeline** (impute, scale, one-hot, classifier). Students must not load “just the logistic regression” without preprocess.

### Cell 2 — high risk vs low risk

```python
high_risk = X.loc[(X["Contract"] == "Month-to-month") & (X["tenure"] <= 2)].head(3)
low_risk = X.loc[(X["Contract"] == "Two year") & (X["tenure"] >= 60)].head(3)
predict_instances(model, high_risk.to_dict(orient="records"))
```

| Slice | Why we chose it |
| --- | --- |
| Month-to-month and tenure ≤ 2 | Matches EDA: newest flexible contracts |
| Two year and tenure ≥ 60 | Locked, long-lived |

`to_dict(orient="records")` makes a **list of JSON objects**, the same shape as `artifacts/vertex_request.json`.

`predict_instances` returns:

```json
{"churn_probability": 0.80, "predicted_churn": 1}
```

Expect **high_risk probabilities clearly above** low_risk (often ~0.7–0.9 vs ~0.05–0.2). If they overlap a lot, the model or filter is wrong.

`display()` needs a Jupyter kernel. In a plain script, use `print(...)`.

### Cell 3–4 — missing columns must fail

```python
try:
    predict_instances(model, [{"tenure": 1}])
except ValueError as exc:
    print("expected error:", exc)
```

One field is not a legal request. The error lists missing columns. That is **good**. Silent defaults would hide broken clients.

Same behavior was verified on the live Vertex endpoint (HTTP 400).

### Cell 5 — write a sample payload

```python
payload = high_risk.iloc[0].to_dict()
payload_path.write_text(json.dumps(payload, indent=2) + "\n")
```

Creates `artifacts/sample_instance.json` for CLI:

```bash
python -m trainer.predict \
  --model-path artifacts/model/model.joblib \
  --instances-path artifacts/sample_instance.json
```

On GCP, wrap the same object in `{"instances": [ ... ]}` as in `artifacts/vertex_request.json`.

**Checkpoint:** students can explain why high-risk JSON scores high, why a partial JSON errors, and that the endpoint is not a public website — it is an authenticated predict API.

---

## 11. Suggested 45-minute teaching order

1. Business story and 26.5% churn (5 min)  
2. Column dictionary, focusing on Contract, tenure, PaymentMethod, TotalCharges, customerID (15 min)  
3. Run notebook 01 live; pause on blanks and groupby rates (10 min)  
4. Notebook 02 dummy vs logistic; attack accuracy (10 min)  
5. Notebook 03 JSON contract; mention Vertex uses the same fields (5 min)

---

## 12. Common student mistakes (and the correction)

| Mistake | Correction |
| --- | --- |
| “Accuracy 74%, great model” | Compare to dummy 73.5%. Use recall / AUC. |
| Drop `No internet service` as missing | It is a real category. |
| Include `customerID` | Identifiers leak uniqueness, not signal. |
| Train and test on the same rows | Always hold out 20% with stratify. |
| Treat `TotalCharges` as string in the model | `to_numeric` + fill 0. |
| Threshold 0.5 forever | Draw PR curve; pick a cost. |
| Call the Vertex URL in Chrome like a website | Console “Test your model” or `gcloud ai endpoints predict`. |

---

## 13. One-page cheat sheet for the whiteboard

```text
Problem: will this customer leave?  Yes=1  No=0
Rows: 7043    Churn Yes: 26.5%

Drop: customerID
Fix:  TotalCharges blanks when tenure=0 → 0.0
Hot:  Contract=Month-to-month, low tenure, Electronic check, Fiber

Pipeline: impute → scale / one-hot → logistic (balanced)
Beat: DummyClassifier most_frequent
Judge: recall, precision, F1, ROC-AUC  (not accuracy alone)
Serve: 19 JSON fields → {churn_probability, predicted_churn}
```
