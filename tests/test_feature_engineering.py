from __future__ import annotations

import pandas as pd

from brfss_diabetes.feature_engineering import BRFSSFeatureEngineer


def test_domain_features_are_created_correctly() -> None:
    raw = pd.DataFrame(
        [
            {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 32,
                "Smoker": 1,
                "Stroke": 0,
                "HeartDiseaseorAttack": 1,
                "PhysActivity": 0,
                "Fruits": 0,
                "Veggies": 1,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 0,
                "NoDocbcCost": 1,
                "GenHlth": 4,
                "MentHlth": 15,
                "PhysHlth": 20,
                "DiffWalk": 1,
                "Sex": 0,
                "Age": 10,
                "Education": 3,
                "Income": 2,
            }
        ]
    )
    transformed = BRFSSFeatureEngineer().fit_transform(raw)
    row = transformed.iloc[0]
    assert row["bmi_category"] == 3
    assert row["obese_flag"] == 1
    assert row["bp_cholesterol_combo"] == 1
    assert row["cardio_event_history"] == 1
    assert row["physical_inactivity_flag"] == 1
    assert row["healthcare_access_barrier"] == 1
    assert row["total_unhealthy_days"] == 35
    assert row["socioeconomic_risk_count"] == 2
    assert row["older_adult_flag"] == 1
