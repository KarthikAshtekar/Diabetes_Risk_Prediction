from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from brfss_diabetes.config import MODEL_DIR, PROJECT_ROOT, REPORT_DIR, TABLE_DIR
from brfss_diabetes.data_loading import load_brfss_data


def test_expected_outputs_exist_and_models_predict() -> None:
    expected = [
        PROJECT_ROOT / "README.md",
        REPORT_DIR / "FINAL_REPORT.md",
        REPORT_DIR / "MODEL_CARD.md",
        REPORT_DIR / "CV_SUMMARY.md",
        REPORT_DIR / "INTERVIEW_DEFENSE_NOTES.md",
        TABLE_DIR / "model_comparison_multiclass.csv",
        TABLE_DIR / "model_comparison_binary.csv",
        TABLE_DIR / "feature_family_ablation.csv",
        TABLE_DIR / "bootstrap_confidence_intervals.csv",
        TABLE_DIR / "repeated_cv_multiclass_summary.csv",
        TABLE_DIR / "grouped_profile_split_results.csv",
        TABLE_DIR / "advanced_multiclass_strategy_comparison.csv",
        TABLE_DIR / "paired_model_bootstrap_comparison.csv",
        MODEL_DIR / "final_multiclass_xgboost.joblib",
        MODEL_DIR / "final_binary_xgboost.joblib",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
    sample = load_brfss_data().drop(columns="Diabetes_012").head(8)
    multiclass = joblib.load(MODEL_DIR / "final_multiclass_xgboost.joblib")
    binary = joblib.load(MODEL_DIR / "final_binary_xgboost.joblib")
    assert len(multiclass.predict(sample)) == 8
    assert len(binary.predict(sample)) == 8


def test_robustness_results_have_expected_invariants() -> None:
    confidence = pd.read_csv(TABLE_DIR / "bootstrap_confidence_intervals.csv")
    assert (confidence["ci_lower_95"] <= confidence["estimate"]).all()
    assert (confidence["estimate"] <= confidence["ci_upper_95"]).all()

    grouped = pd.read_csv(TABLE_DIR / "grouped_profile_split_results.csv")
    grouped_overlap = grouped.loc[
        grouped["split_design"].eq("Profile-grouped holdout"),
        "profile_overlap_count",
    ]
    assert grouped_overlap.eq(0).all()

    repeated = pd.read_csv(TABLE_DIR / "repeated_cv_multiclass_summary.csv")
    assert repeated["folds"].eq(15).all()


def test_nhanes_reports_remain_separate() -> None:
    nhanes = PROJECT_ROOT / "reports" / "nhanes_feasibility"
    brfss = PROJECT_ROOT / "reports" / "brfss_final"
    assert nhanes.exists()
    assert brfss.exists()
    assert Path(nhanes).resolve() != Path(brfss).resolve()
