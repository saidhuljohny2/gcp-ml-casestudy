"""Load and prepare the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read the official IBM telco churn CSV."""
    return pd.read_csv(path)


def clean_churn_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, missing TotalCharges, and encode the target as 0/1."""
    frame = df.copy()
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    # Brand-new customers with tenure 0 have a blank TotalCharges value.
    frame["TotalCharges"] = frame["TotalCharges"].fillna(0.0)
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Churn column contains values other than Yes/No.")
    return frame


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features X and binary target y."""
    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].astype(int)
    return features, target
