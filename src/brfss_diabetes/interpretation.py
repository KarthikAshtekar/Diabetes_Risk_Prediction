from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from .config import RANDOM_STATE
from .imbalance import balanced_sample_weights
from .models import tuned_xgboost_pipeline

FEATURE_FAMILIES = {
    "demographics": {"Sex", "Age", "age_band", "older_adult_flag"},
    "BMI/obesity": {
        "BMI",
        "bmi_category",
        "obese_flag",
        "overweight_or_obese_flag",
    },
    "cardiometabolic history": {
        "HighBP",
        "HighChol",
        "Stroke",
        "HeartDiseaseorAttack",
        "cardiometabolic_count",
        "bp_cholesterol_combo",
        "cardio_event_history",
    },
    "lifestyle": {
        "Smoker",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "unhealthy_lifestyle_count",
        "healthy_diet_flag",
        "physical_inactivity_flag",
    },
    "healthcare access": {
        "CholCheck",
        "AnyHealthcare",
        "NoDocbcCost",
        "healthcare_access_barrier",
        "preventive_screening_gap",
        "cholcheck_with_highchol_flag",
    },
    "general health burden": {
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "poor_general_health_flag",
        "high_mental_distress_flag",
        "high_physical_distress_flag",
        "total_unhealthy_days",
        "limited_functioning_flag",
        "health_burden_score",
    },
    "socioeconomic status": {
        "Education",
        "Income",
        "low_income_flag",
        "low_education_flag",
        "socioeconomic_risk_count",
    },
    "engineered interactions": {
        "bmi_age_interaction",
        "bmi_highbp_interaction",
        "bmi_highchol_interaction",
        "smoking_inactivity_combo",
        "diet_inactivity_combo",
        "income_education_interaction",
        "age_bmi_interaction",
        "age_cardiometabolic_interaction",
    },
}


def feature_family_for(feature: str) -> str:
    clean = feature.split("__")[-1]
    for family, features in FEATURE_FAMILIES.items():
        if clean in features:
            return family
    return "other"


def xgboost_feature_importance(model: object) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    frame = pd.DataFrame(
        {
            "feature": [name.split("__")[-1] for name in feature_names],
            "importance": importance,
        }
    )
    frame["feature_family"] = frame["feature"].map(feature_family_for)
    return frame.sort_values("importance", ascending=False)


def transformed_permutation_importance(
    pipeline: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    sample_size: int = 12_000,
) -> pd.DataFrame:
    if len(x_test) > sample_size:
        sample = x_test.sample(sample_size, random_state=RANDOM_STATE)
        target = y_test.loc[sample.index]
    else:
        sample = x_test
        target = y_test
    engineered = pipeline.named_steps["feature_engineering"].transform(sample)
    transformed = pipeline.named_steps["preprocessor"].transform(engineered)
    result = permutation_importance(
        pipeline.named_steps["model"],
        transformed,
        target,
        scoring="f1_macro",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    frame = pd.DataFrame(
        {
            "feature": [name.split("__")[-1] for name in feature_names],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    frame["feature_family"] = frame["feature"].map(feature_family_for)
    return frame.sort_values("importance_mean", ascending=False)


def feature_family_ablation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    all_feature_columns: list[str],
    best_params: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def fit_score(columns: list[str]) -> tuple[float, float]:
        model = tuned_xgboost_pipeline(columns, "multiclass", best_params)
        model.fit(
            x_train,
            y_train,
            model__sample_weight=balanced_sample_weights(y_train),
        )
        prediction = model.predict(x_validation)
        return (
            f1_score(y_validation, prediction, average="macro"),
            balanced_accuracy_score(y_validation, prediction),
        )

    baseline_f1, baseline_balanced = fit_score(all_feature_columns)
    rows.append(
        {
            "removed_feature_family": "none (baseline)",
            "remaining_feature_count": len(all_feature_columns),
            "macro_f1": baseline_f1,
            "balanced_accuracy": baseline_balanced,
            "macro_f1_drop": 0.0,
            "balanced_accuracy_drop": 0.0,
        }
    )
    for family, family_features in FEATURE_FAMILIES.items():
        remaining = [
            column for column in all_feature_columns if column not in family_features
        ]
        score, balanced = fit_score(remaining)
        rows.append(
            {
                "removed_feature_family": family,
                "remaining_feature_count": len(remaining),
                "macro_f1": score,
                "balanced_accuracy": balanced,
                "macro_f1_drop": baseline_f1 - score,
                "balanced_accuracy_drop": baseline_balanced - balanced,
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1_drop", ascending=False)


def error_analysis(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    multiclass_prediction: np.ndarray,
    binary_prediction: np.ndarray,
) -> pd.DataFrame:
    analysis = x_test.copy()
    analysis["actual_multiclass"] = y_test.to_numpy()
    analysis["predicted_multiclass"] = multiclass_prediction
    analysis["actual_binary"] = y_test.eq(2).astype(int).to_numpy()
    analysis["predicted_binary"] = binary_prediction
    groups = {
        "diabetes_false_negative": analysis[
            analysis["actual_binary"].eq(1) & analysis["predicted_binary"].eq(0)
        ],
        "diabetes_false_positive": analysis[
            analysis["actual_binary"].eq(0) & analysis["predicted_binary"].eq(1)
        ],
        "prediabetes_to_no_diabetes": analysis[
            analysis["actual_multiclass"].eq(1)
            & analysis["predicted_multiclass"].eq(0)
        ],
        "prediabetes_to_diabetes": analysis[
            analysis["actual_multiclass"].eq(1)
            & analysis["predicted_multiclass"].eq(2)
        ],
    }
    rows: list[dict[str, object]] = []
    diagnostic_columns = [
        column
        for column in ["BMI", "Age", "GenHlth", "HighBP", "HighChol", "Income"]
        if column in analysis
    ]
    for error_type, group in groups.items():
        row: dict[str, object] = {
            "error_type": error_type,
            "count": len(group),
            "percentage_of_test_set": len(group) / len(analysis),
        }
        for column in diagnostic_columns:
            row[f"mean_{column}"] = group[column].mean() if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_performance(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = x_test.copy()
    frame["actual"] = y_test.to_numpy()
    frame["predicted"] = prediction
    definitions = {
        "age_group": pd.cut(
            frame["Age"],
            bins=[0, 4, 8, 11, 13],
            labels=["18-34 approx.", "35-54 approx.", "55-69 approx.", "70+ approx."],
            include_lowest=True,
        ),
        "sex": frame["Sex"].map({0: "Female", 1: "Male"}).fillna("Unknown"),
        "income_group": pd.cut(
            frame["Income"],
            bins=[0, 3, 6, 8],
            labels=["Lower", "Middle", "Higher"],
            include_lowest=True,
        ),
        "education_group": pd.cut(
            frame["Education"],
            bins=[0, 3, 5, 6],
            labels=["High school or lower", "Some college", "College graduate"],
            include_lowest=True,
        ),
    }
    rows: list[dict[str, object]] = []
    for dimension, groups in definitions.items():
        for group_name in pd.Series(groups).dropna().unique():
            mask = pd.Series(groups, index=frame.index).eq(group_name)
            part = frame.loc[mask]
            if len(part) < 30:
                continue
            recalls = recall_score(
                part["actual"],
                part["predicted"],
                labels=[0, 1, 2],
                average=None,
                zero_division=0,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(group_name),
                    "n": len(part),
                    "macro_f1": f1_score(
                        part["actual"], part["predicted"], average="macro"
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        part["actual"], part["predicted"]
                    ),
                    "class_0_recall": recalls[0],
                    "class_1_recall": recalls[1],
                    "class_2_recall": recalls[2],
                }
            )
    return pd.DataFrame(rows)
