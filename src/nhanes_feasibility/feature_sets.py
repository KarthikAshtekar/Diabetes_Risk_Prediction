from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAKAGE_VARIABLES

RAW_FEATURE_GROUPS = {
    "demographics": [
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "HIQ011",
    ],
    "anthropometrics": ["BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"],
    "vitals": [
        "BPXSY1",
        "BPXSY2",
        "BPXSY3",
        "BPXSY4",
        "BPXDI1",
        "BPXDI2",
        "BPXDI3",
        "BPXDI4",
        "BPXPLS",
    ],
    "medical_history": [
        "BPQ020",
        "BPQ080",
        "BPQ040A",
        "BPQ090D",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
    ],
    "lifestyle": [
        "SMQ020",
        "SMQ040",
        "PAQ605",
        "PAQ620",
        "PAQ650",
        "PAQ665",
        "PAD680",
        "ALQ101",
        "ALQ130",
        "ALQ141Q",
        "ALQ141U",
        "DBQ700",
        "DBD895",
        "DBD900",
        "DBD905",
        "DBD910",
    ],
    "socioeconomic_access": ["INQ020", "INQ012", "HIQ011"],
    # No clean, generally administered gestational-diabetes or PCOS history
    # variable was found in the 2015-2016 files used by this pilot.
    "female_specific": [],
}

CATEGORICAL_RAW_FEATURES = {
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "HIQ011",
    "BPQ020",
    "BPQ080",
    "BPQ040A",
    "BPQ090D",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "SMQ020",
    "SMQ040",
    "PAQ605",
    "PAQ620",
    "PAQ650",
    "PAQ665",
    "ALQ101",
    "DBQ700",
    "INQ020",
    "INQ012",
}

ENGINEERED_CATEGORICAL_FEATURES = {"bmi_category", "age_band"}

CATEGORICAL_MISSING_CODES = {
    "RIAGENDR": [7, 9],
    "DMDEDUC2": [7, 9],
    "HIQ011": [7, 9],
    "BPQ020": [7, 9],
    "BPQ080": [7, 9],
    "BPQ040A": [7, 9],
    "BPQ090D": [7, 9],
    "MCQ160B": [7, 9],
    "MCQ160C": [7, 9],
    "MCQ160D": [7, 9],
    "MCQ160E": [7, 9],
    "MCQ160F": [7, 9],
    "SMQ020": [7, 9],
    "SMQ040": [7, 9],
    "PAQ605": [7, 9],
    "PAQ620": [7, 9],
    "PAQ650": [7, 9],
    "PAQ665": [7, 9],
    "ALQ101": [7, 9],
    "DBQ700": [7, 9],
    "INQ020": [7, 9],
    "INQ012": [7, 9],
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _yes(frame: pd.DataFrame, column: str) -> pd.Series:
    return _numeric(frame, column).eq(1)


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    definitions: list[dict[str, str]] = []

    systolic_columns = [c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4") if c in result]
    diastolic_columns = [c for c in ("BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4") if c in result]
    if systolic_columns:
        result["avg_systolic_bp"] = result[systolic_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        definitions.append(
            {
                "feature": "avg_systolic_bp",
                "formula": f"row mean of {', '.join(systolic_columns)}",
                "group": "blood_pressure",
            }
        )
    if diastolic_columns:
        result["avg_diastolic_bp"] = result[diastolic_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        definitions.append(
            {
                "feature": "avg_diastolic_bp",
                "formula": f"row mean of {', '.join(diastolic_columns)}",
                "group": "blood_pressure",
            }
        )
    if {"avg_systolic_bp", "avg_diastolic_bp"}.issubset(result.columns):
        result["pulse_pressure"] = (
            result["avg_systolic_bp"] - result["avg_diastolic_bp"]
        )
        result["mean_arterial_pressure"] = (
            result["avg_diastolic_bp"] + result["pulse_pressure"] / 3
        )
        definitions.extend(
            [
                {
                    "feature": "pulse_pressure",
                    "formula": "avg_systolic_bp - avg_diastolic_bp",
                    "group": "blood_pressure",
                },
                {
                    "feature": "mean_arterial_pressure",
                    "formula": "avg_diastolic_bp + pulse_pressure / 3",
                    "group": "blood_pressure",
                },
            ]
        )

    age = _numeric(result, "RIDAGEYR")
    bmi = _numeric(result, "BMXBMI")
    waist = _numeric(result, "BMXWAIST")
    height = _numeric(result, "BMXHT")
    sex = _numeric(result, "RIAGENDR")

    if "BMXBMI" in result:
        result["bmi_category"] = pd.cut(
            bmi,
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obesity"],
            right=False,
        ).astype("object")
        definitions.append(
            {
                "feature": "bmi_category",
                "formula": "WHO-style BMI bands: <18.5, 18.5-24.9, 25-29.9, >=30",
                "group": "anthropometric",
            }
        )
    if "BMXWAIST" in result and "BMXHT" in result:
        result["waist_to_height_ratio"] = waist / height.replace(0, np.nan)
        definitions.append(
            {
                "feature": "waist_to_height_ratio",
                "formula": "BMXWAIST / BMXHT",
                "group": "anthropometric",
            }
        )
    if "BMXWAIST" in result and "RIAGENDR" in result:
        result["central_obesity_flag"] = np.where(
            waist.notna() & sex.notna(),
            ((sex.eq(1) & waist.ge(102)) | (sex.eq(2) & waist.ge(88))).astype(float),
            np.nan,
        )
        definitions.append(
            {
                "feature": "central_obesity_flag",
                "formula": "waist >=102 cm for men or >=88 cm for women",
                "group": "anthropometric",
            }
        )
    if "RIDAGEYR" in result:
        result["age_band"] = pd.cut(
            age,
            bins=[20, 35, 45, 55, 65, np.inf],
            labels=["20-34", "35-44", "45-54", "55-64", "65+"],
            right=False,
        ).astype("object")
        definitions.append(
            {
                "feature": "age_band",
                "formula": "20-34, 35-44, 45-54, 55-64, 65+",
                "group": "demographics",
            }
        )
    if {"RIDAGEYR", "BMXBMI"}.issubset(result.columns):
        result["bmi_x_age"] = bmi * age
        definitions.append(
            {
                "feature": "bmi_x_age",
                "formula": "BMXBMI * RIDAGEYR",
                "group": "anthropometric",
            }
        )
    if {"BMXBMI", "BMXWAIST"}.issubset(result.columns):
        result["bmi_x_waist"] = bmi * waist
        definitions.append(
            {
                "feature": "bmi_x_waist",
                "formula": "BMXBMI * BMXWAIST",
                "group": "anthropometric",
            }
        )

    measured_hypertension = pd.Series(False, index=result.index)
    measured_available = pd.Series(False, index=result.index)
    if "avg_systolic_bp" in result:
        measured_hypertension |= result["avg_systolic_bp"].ge(130)
        measured_available |= result["avg_systolic_bp"].notna()
    if "avg_diastolic_bp" in result:
        measured_hypertension |= result["avg_diastolic_bp"].ge(80)
        measured_available |= result["avg_diastolic_bp"].notna()
    history_available = _numeric(result, "BPQ020").notna()
    result["hypertension_flag"] = np.where(
        measured_available | history_available,
        (measured_hypertension | _yes(result, "BPQ020")).astype(float),
        np.nan,
    )
    definitions.append(
        {
            "feature": "hypertension_flag",
            "formula": "avg SBP >=130 or avg DBP >=80 or BPQ020 == 1",
            "group": "blood_pressure",
        }
    )

    smoking_ever = _numeric(result, "SMQ020")
    smoking_now = _numeric(result, "SMQ040")
    result["smoker_flag"] = np.where(
        smoking_ever.notna() | smoking_now.notna(),
        (smoking_ever.eq(1) & smoking_now.isin([1, 2])).astype(float),
        np.nan,
    )
    definitions.append(
        {
            "feature": "smoker_flag",
            "formula": "SMQ020 == 1 and SMQ040 in {1,2}",
            "group": "lifestyle",
        }
    )

    activity_columns = [c for c in ("PAQ605", "PAQ620", "PAQ650", "PAQ665") if c in result]
    if activity_columns:
        activity = result[activity_columns].apply(pd.to_numeric, errors="coerce")
        result["physical_inactivity_flag"] = np.where(
            activity.notna().any(axis=1),
            (~activity.eq(1).any(axis=1)).astype(float),
            np.nan,
        )
        definitions.append(
            {
                "feature": "physical_inactivity_flag",
                "formula": f"no reported moderate/vigorous activity across {', '.join(activity_columns)}",
                "group": "lifestyle",
            }
        )

    if "ALQ130" in result:
        drinks = _numeric(result, "ALQ130")
        result["heavy_alcohol_flag"] = np.where(
            drinks.notna() & sex.notna(),
            ((sex.eq(1) & drinks.gt(2)) | (sex.eq(2) & drinks.gt(1))).astype(float),
            np.nan,
        )
        definitions.append(
            {
                "feature": "heavy_alcohol_flag",
                "formula": "ALQ130 >2 drinks/day for men or >1 for women",
                "group": "lifestyle",
            }
        )

    result["cholesterol_history_flag"] = np.where(
        _numeric(result, "BPQ080").notna(),
        _yes(result, "BPQ080").astype(float),
        np.nan,
    )
    cardiovascular_columns = [
        c for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F") if c in result
    ]
    if cardiovascular_columns:
        cardiovascular = result[cardiovascular_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        result["cardiovascular_history_flag"] = np.where(
            cardiovascular.notna().any(axis=1),
            cardiovascular.eq(1).any(axis=1).astype(float),
            np.nan,
        )
    else:
        result["cardiovascular_history_flag"] = np.nan
    result["cardiometabolic_history_count"] = result[
        [
            "hypertension_flag",
            "cholesterol_history_flag",
            "cardiovascular_history_flag",
        ]
    ].sum(axis=1, min_count=1)
    definitions.extend(
        [
            {
                "feature": "cholesterol_history_flag",
                "formula": "BPQ080 == 1",
                "group": "medical_history",
            },
            {
                "feature": "cardiovascular_history_flag",
                "formula": "any of MCQ160B-F == 1",
                "group": "medical_history",
            },
            {
                "feature": "cardiometabolic_history_count",
                "formula": "sum of hypertension, cholesterol and cardiovascular flags",
                "group": "medical_history",
            },
        ]
    )

    insured = _numeric(result, "HIQ011")
    result["insurance_gap_flag"] = np.where(
        insured.notna(), insured.eq(2).astype(float), np.nan
    )
    pir = _numeric(result, "INDFMPIR")
    result["low_income_access_risk_flag"] = np.where(
        pir.notna() | insured.notna(),
        (pir.lt(1.3) | insured.eq(2)).astype(float),
        np.nan,
    )
    definitions.extend(
        [
            {
                "feature": "insurance_gap_flag",
                "formula": "HIQ011 == 2",
                "group": "socioeconomic_access",
            },
            {
                "feature": "low_income_access_risk_flag",
                "formula": "INDFMPIR < 1.3 or HIQ011 == 2",
                "group": "socioeconomic_access",
            },
        ]
    )

    return result, pd.DataFrame(definitions)


def select_tier1_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    candidates: list[str] = []
    group_lookup: dict[str, str] = {}
    for group, columns in RAW_FEATURE_GROUPS.items():
        for column in columns:
            if column not in candidates:
                candidates.append(column)
                group_lookup[column] = group

    engineered = [
        "bmi_category",
        "waist_to_height_ratio",
        "central_obesity_flag",
        "age_band",
        "bmi_x_age",
        "bmi_x_waist",
        "avg_systolic_bp",
        "avg_diastolic_bp",
        "pulse_pressure",
        "mean_arterial_pressure",
        "hypertension_flag",
        "smoker_flag",
        "physical_inactivity_flag",
        "heavy_alcohol_flag",
        "cholesterol_history_flag",
        "cardiovascular_history_flag",
        "cardiometabolic_history_count",
        "low_income_access_risk_flag",
        "insurance_gap_flag",
    ]
    for column in engineered:
        if column not in candidates:
            candidates.append(column)
            group_lookup[column] = "engineered"

    selected = [
        column
        for column in candidates
        if column in frame.columns
        and column not in LEAKAGE_VARIABLES
        and frame[column].notna().sum() > 0
    ]
    forbidden_selected = sorted(set(selected) & LEAKAGE_VARIABLES)
    if forbidden_selected:
        raise RuntimeError(f"Leakage variables entered X: {forbidden_selected}")

    categorical = sorted(
        set(selected)
        & (CATEGORICAL_RAW_FEATURES | ENGINEERED_CATEGORICAL_FEATURES)
    )
    numeric = sorted(set(selected) - set(categorical))
    feature_dictionary = pd.DataFrame(
        {
            "feature": selected,
            "source_group": [group_lookup[column] for column in selected],
            "type": [
                "categorical" if column in categorical else "numeric"
                for column in selected
            ],
            "non_missing": [int(frame[column].notna().sum()) for column in selected],
            "missing_pct": [float(frame[column].isna().mean()) for column in selected],
        }
    )
    features = frame[selected].copy()
    for column, codes in CATEGORICAL_MISSING_CODES.items():
        if column in features:
            features[column] = features[column].replace(codes, np.nan)
    for column in categorical:
        features[column] = features[column].astype("object")
    return features, numeric, categorical, feature_dictionary
