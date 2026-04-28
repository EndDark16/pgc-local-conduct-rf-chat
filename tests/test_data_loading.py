from pathlib import Path

import pytest

from src.data_contract import analyze_data_contract
from src.data_loader import load_main_dataset
from src.questionnaire_loader import load_questionnaire


def _has_data_files() -> bool:
    root = Path(__file__).resolve().parent.parent / "data"
    dataset_ok = any(root.glob("hybrid_no_external_scores_dataset_ready*.csv"))
    questionnaire_ok = any(root.glob("questionnaire*.csv"))
    return dataset_ok and questionnaire_ok


@pytest.mark.skipif(not _has_data_files(), reason="No data files in data/")
def test_load_main_dataset():
    df, profile = load_main_dataset()
    assert df.shape[0] > 100
    assert "target_domain_conduct_final" in df.columns
    assert profile["rows"] == df.shape[0]


@pytest.mark.skipif(not _has_data_files(), reason="No data files in data/")
def test_load_questionnaire():
    q_df, profile, feature_map = load_questionnaire()
    assert "feature" in q_df.columns
    assert q_df.shape[0] > 50
    assert len(feature_map) > 50
    assert profile["features_count"] == q_df["feature"].nunique()


@pytest.mark.skipif(not _has_data_files(), reason="No data files in data/")
def test_data_contract_match():
    report = analyze_data_contract()
    assert report["matching_features_count"] >= 100
    assert report["selected_target"] in report["detected_target_columns"]
