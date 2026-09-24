from pathlib import Path

import pandas as pd
import pytest

from trainer.data import FEATURE_COLUMNS, TARGET_COLUMN, clean_churn_frame, load_raw_csv, split_features_and_target
from trainer.model import build_pipeline
from trainer.predict import instances_to_frame, predict_instances
from trainer.task import train_and_export

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")


@pytest.fixture(scope="module")
def cleaned_frame() -> pd.DataFrame:
    return clean_churn_frame(load_raw_csv(DATA_PATH))


def test_schema_has_expected_columns(cleaned_frame: pd.DataFrame) -> None:
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert column in cleaned_frame.columns


def test_total_charges_are_numeric_and_complete(cleaned_frame: pd.DataFrame) -> None:
    assert pd.api.types.is_numeric_dtype(cleaned_frame["TotalCharges"])
    assert cleaned_frame["TotalCharges"].isna().sum() == 0


def test_target_is_binary(cleaned_frame: pd.DataFrame) -> None:
    assert set(cleaned_frame[TARGET_COLUMN].unique()) <= {0, 1}


def test_split_drops_id_and_keeps_feature_order(cleaned_frame: pd.DataFrame) -> None:
    features, target = split_features_and_target(cleaned_frame)
    assert list(features.columns) == FEATURE_COLUMNS
    assert "customerID" not in features.columns
    assert len(target) == len(features)


def test_pipeline_fits_and_predicts_probability_shape(cleaned_frame: pd.DataFrame) -> None:
    features, target = split_features_and_target(cleaned_frame.sample(n=200, random_state=42))
    pipeline = build_pipeline("logistic")
    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features[:5])
    assert probabilities.shape == (5, 2)


def test_train_and_export_writes_artifacts(tmp_path) -> None:
    sample_path = tmp_path / "sample.csv"
    load_raw_csv(DATA_PATH).head(400).to_csv(sample_path, index=False)

    metrics = train_and_export(sample_path, tmp_path / "model", "logistic")
    assert (tmp_path / "model" / "model.joblib").exists()
    assert (tmp_path / "model" / "metrics.json").exists()
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_prediction_contract_requires_all_features() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        instances_to_frame([{"tenure": 1}])


def test_predict_instances_returns_probability_and_label(cleaned_frame: pd.DataFrame) -> None:
    features, target = split_features_and_target(cleaned_frame.head(250))
    pipeline = build_pipeline("logistic")
    pipeline.fit(features, target)
    payload = features.head(3).to_dict(orient="records")
    results = predict_instances(pipeline, payload)
    assert len(results) == 3
    assert set(results[0]) == {"churn_probability", "predicted_churn"}
    assert 0.0 <= results[0]["churn_probability"] <= 1.0
    assert results[0]["predicted_churn"] in {0, 1}
