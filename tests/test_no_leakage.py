from __future__ import annotations

from brfss_diabetes.config import TARGET
from brfss_diabetes.data_loading import load_brfss_data
from brfss_diabetes.feature_engineering import BRFSSFeatureEngineer


def test_target_and_nhanes_variables_do_not_enter_features() -> None:
    frame = load_brfss_data()
    raw_features = frame.drop(columns=TARGET)
    transformed = BRFSSFeatureEngineer().fit_transform(raw_features.head(50))
    forbidden = {
        "Diabetes_012",
        "Diabetes_binary",
        "HighRisk_binary",
        "LBXGH",
        "LBXGLU",
        "LBXGLT",
        "SEQN",
    }
    assert not forbidden.intersection(transformed.columns)
    assert all("nhanes" not in column.lower() for column in transformed.columns)
