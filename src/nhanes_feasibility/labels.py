from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_VARIABLES


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def build_diabetes_risk_stage(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Create 0/1/2 stage using the highest-risk available label evidence."""
    diagnosis = _series_or_nan(frame, LABEL_VARIABLES["self_reported_diabetes"])
    prediabetes = _series_or_nan(
        frame, LABEL_VARIABLES["self_reported_prediabetes"]
    )
    hba1c = _series_or_nan(frame, LABEL_VARIABLES["glycohemoglobin_percent"])
    fasting = _series_or_nan(frame, LABEL_VARIABLES["fasting_glucose_mg_dl"])
    ogtt = _series_or_nan(frame, LABEL_VARIABLES["ogtt_2h_glucose_mg_dl"])

    class_2 = (
        diagnosis.eq(1)
        | hba1c.ge(6.5)
        | fasting.ge(126)
        | ogtt.ge(200)
    )
    class_1 = (
        diagnosis.eq(3)
        | prediabetes.eq(1)
        | hba1c.between(5.7, 6.4, inclusive="both")
        | fasting.between(100, 125, inclusive="both")
        | ogtt.between(140, 199, inclusive="both")
    )

    label = pd.Series(0, index=frame.index, dtype="int64")
    label.loc[class_1] = 1
    label.loc[class_2] = 2
    label.name = "diabetes_risk_stage"

    evidence = pd.DataFrame(
        {
            "label_source": [
                "self-reported doctor-diagnosed/borderline diabetes",
                "self-reported prediabetes",
                "HbA1c",
                "fasting plasma glucose",
                "2-hour OGTT glucose",
            ],
            "nhanes_variable": [
                LABEL_VARIABLES["self_reported_diabetes"],
                LABEL_VARIABLES["self_reported_prediabetes"],
                LABEL_VARIABLES["glycohemoglobin_percent"],
                LABEL_VARIABLES["fasting_glucose_mg_dl"],
                LABEL_VARIABLES["ogtt_2h_glucose_mg_dl"],
            ],
            "available": [
                LABEL_VARIABLES["self_reported_diabetes"] in frame.columns,
                LABEL_VARIABLES["self_reported_prediabetes"] in frame.columns,
                LABEL_VARIABLES["glycohemoglobin_percent"] in frame.columns,
                LABEL_VARIABLES["fasting_glucose_mg_dl"] in frame.columns,
                LABEL_VARIABLES["ogtt_2h_glucose_mg_dl"] in frame.columns,
            ],
            "class_1_rule": [
                "DIQ010 == 3 (borderline)",
                "DIQ160 == 1 (ever told prediabetes)",
                "5.7 <= LBXGH <= 6.4",
                "100 <= LBXGLU <= 125 mg/dL",
                "140 <= LBXGLT <= 199 mg/dL",
            ],
            "class_2_rule": [
                "DIQ010 == 1 (doctor diagnosed)",
                "not used for class 2",
                "LBXGH >= 6.5",
                "LBXGLU >= 126 mg/dL",
                "LBXGLT >= 200 mg/dL",
            ],
        }
    )
    return label, evidence


def has_label_evidence(frame: pd.DataFrame) -> pd.Series:
    diagnosis = _series_or_nan(frame, LABEL_VARIABLES["self_reported_diabetes"])
    prediabetes = _series_or_nan(
        frame, LABEL_VARIABLES["self_reported_prediabetes"]
    )
    diagnosis_observed = diagnosis.isin([1, 2, 3]) | prediabetes.isin([1, 2])
    biomarker_observed = pd.concat(
        [
            _series_or_nan(frame, LABEL_VARIABLES["glycohemoglobin_percent"]),
            _series_or_nan(frame, LABEL_VARIABLES["fasting_glucose_mg_dl"]),
            _series_or_nan(frame, LABEL_VARIABLES["ogtt_2h_glucose_mg_dl"]),
        ],
        axis=1,
    ).notna().any(axis=1)
    return diagnosis_observed | biomarker_observed
