from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brfss_diabetes.config import MODEL_DIR, REPORT_DIR, TABLE_DIR
from brfss_diabetes.data_loading import load_brfss_data


def main() -> int:
    required = [
        ROOT / "README.md",
        REPORT_DIR / "FINAL_REPORT.md",
        REPORT_DIR / "MODEL_CARD.md",
        REPORT_DIR / "CV_SUMMARY.md",
        REPORT_DIR / "INTERVIEW_DEFENSE_NOTES.md",
        REPORT_DIR / "report.html",
        MODEL_DIR / "final_multiclass_xgboost.joblib",
        MODEL_DIR / "final_binary_xgboost.joblib",
        MODEL_DIR / "preprocessing_pipeline.joblib",
        TABLE_DIR / "final_test_metrics_multiclass.csv",
        TABLE_DIR / "final_test_metrics_binary.csv",
        TABLE_DIR / "bootstrap_confidence_intervals.csv",
        TABLE_DIR / "repeated_cv_multiclass_summary.csv",
        TABLE_DIR / "grouped_profile_split_results.csv",
        TABLE_DIR / "advanced_multiclass_strategy_comparison.csv",
        TABLE_DIR / "paired_model_bootstrap_comparison.csv",
        REPORT_DIR / "figures" / "bootstrap_confidence_intervals.png",
        REPORT_DIR / "figures" / "repeated_cv_model_comparison.png",
        REPORT_DIR / "figures" / "grouped_profile_split_comparison.png",
        REPORT_DIR / "figures" / "advanced_multiclass_strategy_comparison.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required BRFSS outputs missing: {missing}")

    frame = load_brfss_data()
    sample = frame.drop(columns="Diabetes_012").head(20)
    multiclass_model = joblib.load(
        MODEL_DIR / "final_multiclass_xgboost.joblib"
    )
    binary_model = joblib.load(MODEL_DIR / "final_binary_xgboost.joblib")
    multiclass_prediction = multiclass_model.predict(sample)
    binary_prediction = binary_model.predict(sample)
    if len(multiclass_prediction) != len(sample) or len(binary_prediction) != len(sample):
        raise AssertionError("Saved models could not score the sample batch.")

    multi_predictions = pd.read_csv(
        TABLE_DIR / "multiclass_test_predictions.csv"
    )
    multi_metrics = pd.read_csv(
        TABLE_DIR / "final_test_metrics_multiclass.csv"
    ).set_index("metric")["value"]
    recomputed_multi = {
        "macro_f1": f1_score(
            multi_predictions["actual"],
            multi_predictions["predicted"],
            average="macro",
        ),
        "balanced_accuracy": balanced_accuracy_score(
            multi_predictions["actual"], multi_predictions["predicted"]
        ),
    }
    binary_predictions = pd.read_csv(TABLE_DIR / "binary_test_predictions.csv")
    binary_metrics = pd.read_csv(
        TABLE_DIR / "final_test_metrics_binary.csv"
    ).set_index("metric")["value"]
    recomputed_binary = {
        "roc_auc": roc_auc_score(
            binary_predictions["actual"],
            binary_predictions["calibrated_probability"],
        ),
        "pr_auc": average_precision_score(
            binary_predictions["actual"],
            binary_predictions["calibrated_probability"],
        ),
    }
    for metric, value in recomputed_multi.items():
        if not np.isclose(value, multi_metrics.loc[metric], atol=1e-12):
            raise AssertionError(f"Multiclass metric mismatch: {metric}")
    for metric, value in recomputed_binary.items():
        if not np.isclose(value, binary_metrics.loc[metric], atol=1e-12):
            raise AssertionError(f"Binary metric mismatch: {metric}")

    summary = {
        "status": "ready_to_share_with_documented_limitations",
        "required_outputs": len(required),
        "saved_model_prediction_check": "passed",
        "metric_recomputation": "passed",
        "nhanes_artifact_isolation": all(
            "nhanes" not in column.lower() for column in sample.columns
        ),
        "robustness_outputs": "passed",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
