from __future__ import annotations

import gc
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brfss_diabetes.calibration import SigmoidProbabilityCalibrator  # noqa: E402
from brfss_diabetes.config import (  # noqa: E402
    ABLATION_TRAIN_SAMPLE,
    ABLATION_VALIDATION_SAMPLE,
    MODEL_COMPARISON_SAMPLE,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TABLE_DIR,
    TARGET,
    TEST_SIZE,
    TUNING_SAMPLE,
    ensure_directories,
)
from brfss_diabetes.data_loading import load_brfss_data, make_targets  # noqa: E402
from brfss_diabetes.evaluation import (  # noqa: E402
    compare_models_cv,
    evaluate_binary,
    evaluate_multiclass,
    high_risk_threshold_analysis,
    threshold_analysis,
)
from brfss_diabetes.feature_engineering import (  # noqa: E402
    BRFSSFeatureEngineer,
    engineered_feature_dictionary,
)
from brfss_diabetes.imbalance import (  # noqa: E402
    balanced_sample_weights,
    binary_scale_pos_weight,
)
from brfss_diabetes.interpretation import (  # noqa: E402
    error_analysis,
    feature_family_ablation,
    subgroup_performance,
    transformed_permutation_importance,
    xgboost_feature_importance,
)
from brfss_diabetes.models import (  # noqa: E402
    comparison_models,
    tuned_xgboost_pipeline,
)
from brfss_diabetes.reporting import (  # noqa: E402
    generate_figures,
    write_html_report,
    write_reports,
)
from brfss_diabetes.tuning import (  # noqa: E402
    tune_logistic,
    tune_random_forest,
    tune_xgboost,
)
from brfss_diabetes.utils import (  # noqa: E402
    configure_logging,
    stratified_sample_indices,
    write_json,
)
from brfss_diabetes.validation import validate_brfss_frame  # noqa: E402


def _save_cv_results(frame: pd.DataFrame, filename: str) -> None:
    selected = [
        column
        for column in frame.columns
        if column.startswith("param_")
        or column
        in {
            "mean_fit_time",
            "std_fit_time",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "params",
        }
    ]
    frame[selected].to_csv(TABLE_DIR / filename, index=False)


def main() -> int:
    ensure_directories()
    configure_logging()
    logger = logging.getLogger("brfss_full_pipeline")

    logger.info("Loading and validating BRFSS 2015 data")
    frame = load_brfss_data()
    schema, missingness, class_distribution, summary = validate_brfss_frame(frame)
    schema.to_csv(TABLE_DIR / "schema_validation.csv", index=False)
    missingness.to_csv(TABLE_DIR / "missingness_report.csv", index=False)
    class_distribution.to_csv(TABLE_DIR / "class_distribution.csv", index=False)
    summary.to_csv(TABLE_DIR / "basic_summary_statistics.csv", index=False)
    engineered_feature_dictionary().to_csv(
        TABLE_DIR / "engineered_feature_dictionary.csv", index=False
    )

    targets = make_targets(frame)
    x_raw = frame.drop(columns=TARGET)
    y_multi = targets["Diabetes_012"]
    y_binary = targets["Diabetes_binary"]
    feature_engineer = BRFSSFeatureEngineer()
    engineered_preview = feature_engineer.transform(x_raw.head(10))
    final_feature_columns = engineered_preview.columns.tolist()

    (
        x_train,
        x_test,
        y_multi_train,
        y_multi_test,
        y_binary_train,
        y_binary_test,
    ) = train_test_split(
        x_raw,
        y_multi,
        y_binary,
        test_size=TEST_SIZE,
        stratify=y_multi,
        random_state=RANDOM_STATE,
    )
    split_table = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(x_train),
                "class_0": int((y_multi_train == 0).sum()),
                "class_1": int((y_multi_train == 1).sum()),
                "class_2": int((y_multi_train == 2).sum()),
            },
            {
                "split": "test",
                "n": len(x_test),
                "class_0": int((y_multi_test == 0).sum()),
                "class_1": int((y_multi_test == 1).sum()),
                "class_2": int((y_multi_test == 2).sum()),
            },
        ]
    )
    split_table.to_csv(TABLE_DIR / "train_test_split_summary.csv", index=False)
    test_export = x_test.copy()
    test_export["Diabetes_012"] = y_multi_test
    test_export["Diabetes_binary"] = y_binary_test
    test_export.to_csv(
        PROCESSED_DATA_DIR / "brfss_test_set.csv.gz",
        index_label="row_index",
        compression="gzip",
    )

    logger.info("Running model comparison on a stratified training-only sample")
    comparison_indices = stratified_sample_indices(
        y_multi_train, MODEL_COMPARISON_SAMPLE
    )
    x_compare = x_train.loc[comparison_indices].reset_index(drop=True)
    y_multi_compare = y_multi_train.loc[comparison_indices].reset_index(drop=True)
    y_binary_compare = y_binary_train.loc[comparison_indices].reset_index(drop=True)

    multi_models = comparison_models(final_feature_columns, "multiclass")
    comparison_multiclass = compare_models_cv(
        multi_models, x_compare, y_multi_compare, "multiclass"
    )
    comparison_multiclass.to_csv(
        TABLE_DIR / "model_comparison_multiclass.csv", index=False
    )

    binary_models = comparison_models(final_feature_columns, "binary")
    binary_models["XGBoost"].set_params(
        model__scale_pos_weight=binary_scale_pos_weight(y_binary_compare)
    )
    comparison_binary = compare_models_cv(
        binary_models, x_compare, y_binary_compare, "binary"
    )
    comparison_binary.to_csv(
        TABLE_DIR / "model_comparison_binary.csv", index=False
    )

    logger.info("Comparing multiclass XGBoost imbalance strategies")
    unweighted = clone(multi_models["XGBoost"])
    weighted = clone(multi_models["XGBoost"])
    imbalance_models = {
        "XGBoost unweighted": unweighted,
        "XGBoost": weighted,
    }
    imbalance_comparison = compare_models_cv(
        imbalance_models, x_compare, y_multi_compare, "multiclass"
    )
    imbalance_comparison["strategy"] = imbalance_comparison["model"].map(
        {
            "XGBoost unweighted": "No weighting/resampling",
            "XGBoost": "Balanced sample weights inside each training fold",
        }
    )
    imbalance_comparison.loc[len(imbalance_comparison)] = {
        "model": "SMOTE-NC",
        "cv_folds": 0,
        "strategy": (
            "Not selected: synthetic coded survey profiles can distort ordinal/"
            "binary semantics and probability calibration."
        ),
    }
    imbalance_comparison.to_csv(
        TABLE_DIR / "imbalance_strategy_comparison.csv", index=False
    )
    del multi_models, binary_models, imbalance_models, unweighted, weighted
    gc.collect()

    logger.info("Tuning Logistic Regression, Random Forest and XGBoost")
    tuning_indices = stratified_sample_indices(y_multi_train, TUNING_SAMPLE)
    x_tune = x_train.loc[tuning_indices].reset_index(drop=True)
    y_multi_tune = y_multi_train.loc[tuning_indices].reset_index(drop=True)
    y_binary_tune = y_binary_train.loc[tuning_indices].reset_index(drop=True)

    logistic_multi_params, logistic_multi_results = tune_logistic(
        x_tune, y_multi_tune, final_feature_columns, "f1_macro"
    )
    _save_cv_results(
        logistic_multi_results, "logistic_tuning_results_multiclass.csv"
    )
    del logistic_multi_results
    gc.collect()
    rf_params, rf_results = tune_random_forest(
        x_tune, y_multi_tune, final_feature_columns
    )
    _save_cv_results(rf_results, "random_forest_tuning_results_multiclass.csv")
    del rf_results
    gc.collect()

    best_multi_params, multi_tuning_results = tune_xgboost(
        x_tune, y_multi_tune, final_feature_columns, "multiclass"
    )
    _save_cv_results(
        multi_tuning_results, "xgboost_tuning_results_multiclass.csv"
    )
    del multi_tuning_results
    gc.collect()

    logistic_binary_params, logistic_binary_results = tune_logistic(
        x_tune, y_binary_tune, final_feature_columns, "average_precision"
    )
    _save_cv_results(
        logistic_binary_results, "logistic_tuning_results_binary.csv"
    )
    del logistic_binary_results
    gc.collect()
    best_binary_params, binary_tuning_results = tune_xgboost(
        x_tune, y_binary_tune, final_feature_columns, "binary"
    )
    best_binary_params["scale_pos_weight"] = binary_scale_pos_weight(y_binary_train)
    _save_cv_results(
        binary_tuning_results, "xgboost_tuning_results_binary.csv"
    )
    del binary_tuning_results
    gc.collect()
    write_json(
        TABLE_DIR / "best_model_parameters.json",
        {
            "logistic_multiclass": logistic_multi_params,
            "random_forest_multiclass": rf_params,
            "xgboost_multiclass": best_multi_params,
            "logistic_binary": logistic_binary_params,
            "xgboost_binary": best_binary_params,
        },
    )

    logger.info("Selecting validation-only thresholds and binary calibration")
    (
        x_binary_fit,
        x_binary_validation,
        y_binary_fit,
        y_binary_validation,
    ) = train_test_split(
        x_train,
        y_binary_train,
        test_size=0.15,
        stratify=y_binary_train,
        random_state=RANDOM_STATE,
    )
    binary_validation_model = tuned_xgboost_pipeline(
        final_feature_columns, "binary", best_binary_params
    )
    binary_validation_model.fit(x_binary_fit, y_binary_fit)
    validation_raw_probability = binary_validation_model.predict_proba(
        x_binary_validation
    )[:, 1]
    calibrator = SigmoidProbabilityCalibrator().fit(
        validation_raw_probability, y_binary_validation.to_numpy()
    )
    validation_calibrated_probability = calibrator.predict(
        validation_raw_probability
    )
    threshold_table = threshold_analysis(
        y_binary_validation, validation_calibrated_probability
    )
    threshold_table.to_csv(
        TABLE_DIR / "binary_threshold_analysis.csv", index=False
    )
    selected_threshold = float(
        threshold_table.loc[
            threshold_table["selection"].str.contains("max_f1"), "threshold"
        ].iloc[0]
    )
    calibration_comparison = pd.DataFrame(
        [
            {
                "probability": "raw_xgboost",
                "validation_brier_score": brier_score_loss(
                    y_binary_validation, validation_raw_probability
                ),
            },
            {
                "probability": "sigmoid_calibrated",
                "validation_brier_score": brier_score_loss(
                    y_binary_validation, validation_calibrated_probability
                ),
            },
        ]
    )
    calibration_comparison.to_csv(
        TABLE_DIR / "binary_calibration_comparison.csv", index=False
    )

    (
        x_multi_fit,
        x_multi_validation,
        y_multi_fit,
        y_multi_validation,
    ) = train_test_split(
        x_train,
        y_multi_train,
        test_size=0.15,
        stratify=y_multi_train,
        random_state=RANDOM_STATE,
    )
    multi_validation_model = tuned_xgboost_pipeline(
        final_feature_columns, "multiclass", best_multi_params
    )
    multi_validation_model.fit(
        x_multi_fit,
        y_multi_fit,
        model__sample_weight=balanced_sample_weights(y_multi_fit),
    )
    multi_validation_probability = multi_validation_model.predict_proba(
        x_multi_validation
    )
    high_risk_table = high_risk_threshold_analysis(
        y_multi_validation,
        multi_validation_probability[:, 1] + multi_validation_probability[:, 2],
    )
    high_risk_table.to_csv(
        TABLE_DIR / "high_risk_threshold_analysis.csv", index=False
    )

    logger.info("Fitting final models on the complete training partition")
    final_multiclass = tuned_xgboost_pipeline(
        final_feature_columns, "multiclass", best_multi_params
    )
    final_multiclass.fit(
        x_train,
        y_multi_train,
        model__sample_weight=balanced_sample_weights(y_multi_train),
    )
    final_binary = tuned_xgboost_pipeline(
        final_feature_columns, "binary", best_binary_params
    )
    final_binary.fit(x_train, y_binary_train)

    test_raw_binary_probability = final_binary.predict_proba(x_test)[:, 1]
    test_calibrated_probability = calibrator.predict(test_raw_binary_probability)
    (
        multiclass_metrics,
        multiclass_report,
        multiclass_details,
        multiclass_predictions,
    ) = evaluate_multiclass(final_multiclass, x_test, y_multi_test)
    binary_metrics, binary_report, binary_details, binary_predictions = (
        evaluate_binary(
            final_binary,
            x_test,
            y_binary_test,
            threshold=selected_threshold,
            calibrated_probability=test_calibrated_probability,
        )
    )
    multiclass_metrics.to_csv(
        TABLE_DIR / "final_test_metrics_multiclass.csv", index=False
    )
    binary_metrics.to_csv(
        TABLE_DIR / "final_test_metrics_binary.csv", index=False
    )
    multiclass_report.to_csv(
        TABLE_DIR / "classification_report_multiclass.csv", index=False
    )
    binary_report.to_csv(
        TABLE_DIR / "classification_report_binary.csv", index=False
    )
    multiclass_predictions.to_csv(
        TABLE_DIR / "multiclass_test_predictions.csv", index=False
    )
    binary_predictions.to_csv(
        TABLE_DIR / "binary_test_predictions.csv", index=False
    )

    logger.info("Computing importance, ablation and error analysis")
    built_in_importance = xgboost_feature_importance(final_multiclass)
    built_in_importance.to_csv(
        TABLE_DIR / "xgboost_feature_importance.csv", index=False
    )
    permutation = transformed_permutation_importance(
        final_multiclass, x_test, y_multi_test
    )
    permutation.to_csv(TABLE_DIR / "permutation_importance.csv", index=False)

    ablation_total = ABLATION_TRAIN_SAMPLE + ABLATION_VALIDATION_SAMPLE
    ablation_indices = stratified_sample_indices(y_multi_train, ablation_total)
    x_ablation = x_train.loc[ablation_indices]
    y_ablation = y_multi_train.loc[ablation_indices]
    x_ablation_train, x_ablation_validation, y_ablation_train, y_ablation_validation = (
        train_test_split(
            x_ablation,
            y_ablation,
            test_size=ABLATION_VALIDATION_SAMPLE / ablation_total,
            stratify=y_ablation,
            random_state=RANDOM_STATE,
        )
    )
    ablation_params = best_multi_params.copy()
    ablation_params["n_estimators"] = min(
        int(ablation_params.get("n_estimators", 300)), 240
    )
    ablation = feature_family_ablation(
        x_ablation_train,
        y_ablation_train,
        x_ablation_validation,
        y_ablation_validation,
        final_feature_columns,
        ablation_params,
    )
    ablation.to_csv(TABLE_DIR / "feature_family_ablation.csv", index=False)

    errors = error_analysis(
        x_test,
        y_multi_test,
        np.asarray(multiclass_details["prediction"]),
        np.asarray(binary_details["prediction"]),
    )
    errors.to_csv(TABLE_DIR / "error_analysis.csv", index=False)
    subgroups = subgroup_performance(
        x_test, y_multi_test, np.asarray(multiclass_details["prediction"])
    )
    subgroups.to_csv(TABLE_DIR / "subgroup_performance.csv", index=False)

    logger.info("Saving models and generating reports")
    joblib.dump(
        final_multiclass, MODEL_DIR / "final_multiclass_xgboost.joblib"
    )
    joblib.dump(final_binary, MODEL_DIR / "final_binary_xgboost.joblib")
    joblib.dump(calibrator, MODEL_DIR / "binary_probability_calibrator.joblib")
    preprocessing = Pipeline(final_multiclass.steps[:-1])
    joblib.dump(preprocessing, MODEL_DIR / "preprocessing_pipeline.joblib")

    chart_map = generate_figures(
        class_distribution=class_distribution,
        comparison_multiclass=comparison_multiclass,
        comparison_binary=comparison_binary,
        multiclass_details=multiclass_details,
        binary_details=binary_details,
        y_binary_test=y_binary_test,
        raw_binary_probability=test_raw_binary_probability,
        calibrated_binary_probability=test_calibrated_probability,
        threshold_table=threshold_table,
        feature_importance=built_in_importance,
        permutation=permutation,
        ablation=ablation,
    )
    chart_map.to_csv(TABLE_DIR / "chart_map.csv", index=False)
    write_reports(
        dataset_shape=frame.shape,
        original_feature_count=len(x_raw.columns),
        engineered_feature_count=len(final_feature_columns),
        class_distribution=class_distribution,
        multiclass_comparison=comparison_multiclass,
        binary_comparison=comparison_binary,
        multiclass_metrics=multiclass_metrics,
        binary_metrics=binary_metrics,
        multiclass_report=multiclass_report,
        binary_report=binary_report,
        threshold_table=threshold_table,
        high_risk_table=high_risk_table,
        feature_importance=built_in_importance,
        permutation=permutation,
        ablation=ablation,
        error_table=errors,
        subgroup_table=subgroups,
        best_multiclass_params=best_multi_params,
        best_binary_params=best_binary_params,
    )
    html_path = write_html_report()

    multiclass_cv_leader = comparison_multiclass.sort_values(
        "macro_f1", ascending=False
    ).iloc[0]
    binary_cv_leader = comparison_binary.sort_values(
        "pr_auc", ascending=False
    ).iloc[0]
    summary_payload = {
        "dataset_shape": list(frame.shape),
        "target_distribution": {
            str(key): int(value)
            for key, value in y_multi.value_counts().sort_index().items()
        },
        "original_feature_count": len(x_raw.columns),
        "engineered_feature_count": len(final_feature_columns),
        "selected_multiclass_model": "XGBoost",
        "multiclass_cv_macro_f1_leader": {
            "model": str(multiclass_cv_leader["model"]),
            "macro_f1": float(multiclass_cv_leader["macro_f1"]),
        },
        "multiclass_metrics": dict(
            zip(multiclass_metrics["metric"], multiclass_metrics["value"])
        ),
        "selected_binary_model": "XGBoost",
        "binary_cv_pr_auc_leader": {
            "model": str(binary_cv_leader["model"]),
            "pr_auc": float(binary_cv_leader["pr_auc"]),
        },
        "binary_metrics": dict(
            zip(binary_metrics["metric"], binary_metrics["value"])
        ),
        "xgboost_multiclass_cv_macro_f1": float(
            comparison_multiclass.loc[
                comparison_multiclass["model"].eq("XGBoost"), "macro_f1"
            ].iloc[0]
        ),
        "strongest_non_xgboost_multiclass_cv_macro_f1": float(
            comparison_multiclass.loc[
                ~comparison_multiclass["model"].isin(
                    ["XGBoost", "DummyClassifier"]
                ),
                "macro_f1",
            ].max()
        ),
        "xgboost_beat_multiclass_macro_f1_baselines": bool(
            comparison_multiclass.loc[
                comparison_multiclass["model"].eq("XGBoost"), "macro_f1"
            ].iloc[0]
            >= multiclass_cv_leader["macro_f1"]
        ),
        "top_feature_families": ablation.loc[
            ablation["removed_feature_family"].ne("none (baseline)")
        ]
        .head(3)["removed_feature_family"]
        .tolist(),
        "report": str(html_path.relative_to(ROOT)),
    }
    write_json(REPORT_DIR / "run_summary.json", summary_payload)

    if os.environ.get("BRFSS_SKIP_ROBUSTNESS", "0") != "1":
        logger.info("Running statistical and advanced-ML robustness analyses")
        from run_brfss_robustness_analysis import main as run_robustness

        run_robustness()

    print("\nBRFSS final pipeline complete")
    print(json.dumps(summary_payload, indent=2))
    print("\nReproduce with:")
    print("python scripts/run_brfss_full_pipeline.py")
    print("python scripts/validate_brfss_project.py")
    print("pytest")
    print("ruff check .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
