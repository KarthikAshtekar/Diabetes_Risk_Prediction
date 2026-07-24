from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brfss_diabetes.config import (
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TABLE_DIR,
    TARGET,
    TEST_SIZE,
    ensure_directories,
)
from brfss_diabetes.data_loading import load_brfss_data, make_targets
from brfss_diabetes.evaluation import threshold_analysis
from brfss_diabetes.feature_engineering import BRFSSFeatureEngineer
from brfss_diabetes.imbalance import balanced_sample_weights
from brfss_diabetes.models import (
    comparison_models,
    tuned_xgboost_pipeline,
)
from brfss_diabetes.reporting import write_html_report
from brfss_diabetes.robustness import (
    MulticlassLogitCalibrator,
    apply_probability_multipliers,
    binary_metric_row,
    bootstrap_confidence_intervals,
    class_weight_sensitivity,
    custom_multiclass_sample_weights,
    default_ensemble_models,
    fit_ordinal_models,
    fit_two_stage_models,
    fitted_binary_logistic,
    fitted_extratrees,
    grouped_profile_split,
    mcnemar_exact_test,
    multiclass_metric_row,
    oof_ensemble_weight_search,
    ordinal_predict_proba,
    paired_bootstrap_difference,
    paired_bootstrap_probability_difference,
    repeated_model_comparison,
    smotenc_sensitivity_cv,
    tune_probability_multipliers,
    two_stage_predict_proba,
)
from brfss_diabetes.robustness_reporting import (
    append_robustness_chart_map,
    generate_robustness_figures,
    update_report_with_robustness,
)
from brfss_diabetes.utils import (
    configure_logging,
    stratified_sample_indices,
    write_json,
)


def _strategy_row(
    strategy: str,
    y_true: pd.Series,
    probability: np.ndarray,
    prediction: np.ndarray | None = None,
) -> dict[str, object]:
    predicted = probability.argmax(axis=1) if prediction is None else prediction
    return {
        "strategy": strategy,
        **multiclass_metric_row(y_true, predicted, probability),
    }


def _feature_ablation_table(
    repeated_folds: pd.DataFrame,
    repeated_summary: pd.DataFrame,
) -> pd.DataFrame:
    name_map = {
        "XGBoost — original features": "Original features only",
        "XGBoost — engineered features": "Original + engineered features",
    }
    selected = repeated_summary.loc[
        repeated_summary["model"].isin(name_map)
    ].copy()
    selected["configuration"] = selected["model"].map(name_map)
    columns = [
        "configuration",
        "folds",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "class_1_recall_mean",
        "class_2_recall_mean",
        "macro_roc_auc_ovr_mean",
    ]
    result = selected[columns].copy()
    pivot = repeated_folds.loc[
        repeated_folds["model"].isin(name_map)
    ].pivot(index="fold", columns="model", values="macro_f1")
    differences = (
        pivot["XGBoost — engineered features"]
        - pivot["XGBoost — original features"]
    )
    result["paired_macro_f1_difference_mean"] = np.nan
    result["paired_macro_f1_difference_std"] = np.nan
    engineered_mask = result["configuration"].eq(
        "Original + engineered features"
    )
    result.loc[
        engineered_mask, "paired_macro_f1_difference_mean"
    ] = differences.mean()
    result.loc[
        engineered_mask, "paired_macro_f1_difference_std"
    ] = differences.std(ddof=1)
    return result


def main() -> int:
    ensure_directories()
    configure_logging()
    logger = logging.getLogger("brfss_robustness")
    logger.info("Loading official BRFSS split and saved model parameters")

    frame = load_brfss_data()
    targets = make_targets(frame)
    x_raw = frame.drop(columns=TARGET)
    y_multi = targets["Diabetes_012"]
    y_binary = targets["Diabetes_binary"]
    final_feature_columns = (
        BRFSSFeatureEngineer().fit_transform(x_raw.head(10)).columns.tolist()
    )
    original_feature_columns = x_raw.columns.tolist()
    parameters = json.loads(
        (TABLE_DIR / "best_model_parameters.json").read_text(encoding="utf-8")
    )
    best_multiclass_params = parameters["xgboost_multiclass"]
    best_binary_params = parameters["xgboost_binary"]

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

    logger.info("Running 5x3 repeated cross-validation and feature-set ablation")
    repeated_indices = stratified_sample_indices(y_multi_train, 30_000)
    x_repeated = x_train.loc[repeated_indices].reset_index(drop=True)
    y_repeated = y_multi_train.loc[repeated_indices].reset_index(drop=True)
    comparison = comparison_models(final_feature_columns, "multiclass")
    repeated_models = {
        "Logistic Regression": comparison["Logistic Regression"],
        "ExtraTrees": comparison["ExtraTrees"],
        "XGBoost — engineered features": tuned_xgboost_pipeline(
            final_feature_columns, "multiclass", best_multiclass_params
        ),
        "XGBoost — original features": tuned_xgboost_pipeline(
            original_feature_columns, "multiclass", best_multiclass_params
        ),
    }
    repeated_folds, repeated_summary = repeated_model_comparison(
        repeated_models,
        x_repeated,
        y_repeated,
        xgboost_model_names={
            "XGBoost — engineered features",
            "XGBoost — original features",
        },
    )
    repeated_folds.to_csv(
        TABLE_DIR / "repeated_cv_multiclass_folds.csv", index=False
    )
    repeated_summary.to_csv(
        TABLE_DIR / "repeated_cv_multiclass_summary.csv", index=False
    )
    feature_ablation = _feature_ablation_table(
        repeated_folds, repeated_summary
    )
    feature_ablation.to_csv(
        TABLE_DIR / "feature_engineering_ablation_repeated_cv.csv",
        index=False,
    )

    logger.info("Evaluating a profile-grouped holdout")
    group_train_indices, group_test_indices, group_summary = (
        grouped_profile_split(x_raw, y_multi)
    )
    grouped_model = tuned_xgboost_pipeline(
        final_feature_columns, "multiclass", best_multiclass_params
    )
    grouped_model.fit(
        x_raw.iloc[group_train_indices],
        y_multi.iloc[group_train_indices],
        model__sample_weight=balanced_sample_weights(
            y_multi.iloc[group_train_indices]
        ),
    )
    grouped_probability = grouped_model.predict_proba(
        x_raw.iloc[group_test_indices]
    )
    grouped_prediction = grouped_probability.argmax(axis=1)
    grouped_metrics = multiclass_metric_row(
        y_multi.iloc[group_test_indices],
        grouped_prediction,
        grouped_probability,
    )

    final_multiclass = joblib.load(
        MODEL_DIR / "final_multiclass_xgboost.joblib"
    )
    official_probability = final_multiclass.predict_proba(x_test)
    official_prediction = official_probability.argmax(axis=1)
    official_metrics = multiclass_metric_row(
        y_multi_test, official_prediction, official_probability
    )
    grouped_rows = []
    for split_design, metrics, test_rows, overlap in (
        (
            "Official random holdout",
            official_metrics,
            len(x_test),
            len(
                set(
                    pd.util.hash_pandas_object(
                        x_train, index=False
                    ).to_numpy()
                )
                & set(
                    pd.util.hash_pandas_object(
                        x_test, index=False
                    ).to_numpy()
                )
            ),
        ),
        (
            "Profile-grouped holdout",
            grouped_metrics,
            len(group_test_indices),
            group_summary["profile_overlap_count"],
        ),
    ):
        for metric, value in metrics.items():
            grouped_rows.append(
                {
                    "split_design": split_design,
                    "metric": metric,
                    "value": value,
                    "test_rows": test_rows,
                    "profile_overlap_count": overlap,
                }
            )
    grouped_results = pd.DataFrame(grouped_rows)
    grouped_results.to_csv(
        TABLE_DIR / "grouped_profile_split_results.csv", index=False
    )
    write_json(TABLE_DIR / "grouped_profile_split_summary.json", group_summary)

    logger.info("Creating training-only strategy selection and evaluation splits")
    (
        x_strategy_train,
        x_strategy_temp,
        y_strategy_train,
        y_strategy_temp,
    ) = train_test_split(
        x_train,
        y_multi_train,
        test_size=0.30,
        stratify=y_multi_train,
        random_state=RANDOM_STATE,
    )
    (
        x_strategy_selection,
        x_strategy_evaluation,
        y_strategy_selection,
        y_strategy_evaluation,
    ) = train_test_split(
        x_strategy_temp,
        y_strategy_temp,
        test_size=0.50,
        stratify=y_strategy_temp,
        random_state=RANDOM_STATE,
    )

    baseline_strategy_model = tuned_xgboost_pipeline(
        final_feature_columns, "multiclass", best_multiclass_params
    )
    baseline_strategy_model.fit(
        x_strategy_train,
        y_strategy_train,
        model__sample_weight=balanced_sample_weights(y_strategy_train),
    )
    baseline_selection_probability = baseline_strategy_model.predict_proba(
        x_strategy_selection
    )
    baseline_evaluation_probability = baseline_strategy_model.predict_proba(
        x_strategy_evaluation
    )

    logger.info("Tuning multiclass probability multipliers")
    multiplier_results = tune_probability_multipliers(
        y_strategy_selection, baseline_selection_probability
    )
    multiplier_results.to_csv(
        TABLE_DIR / "multiclass_probability_threshold_tuning.csv",
        index=False,
    )
    selected_multiplier = multiplier_results.iloc[0]
    adjusted_prediction, adjusted_probability = apply_probability_multipliers(
        baseline_evaluation_probability,
        float(selected_multiplier["class_1_multiplier"]),
        float(selected_multiplier["class_2_multiplier"]),
    )

    logger.info("Fitting multiclass probability calibration")
    multiclass_calibrator = MulticlassLogitCalibrator().fit(
        baseline_selection_probability, y_strategy_selection
    )
    calibrated_evaluation_probability = multiclass_calibrator.predict_proba(
        baseline_evaluation_probability
    )
    calibration_rows = []
    for name, probability in (
        ("Raw weighted XGBoost", baseline_evaluation_probability),
        ("Multinomial log-probability calibration", calibrated_evaluation_probability),
    ):
        prediction = probability.argmax(axis=1)
        calibration_rows.append(
            {
                "probability": name,
                **multiclass_metric_row(
                    y_strategy_evaluation, prediction, probability
                ),
            }
        )
    calibration_comparison = pd.DataFrame(calibration_rows)
    calibration_comparison.to_csv(
        TABLE_DIR / "multiclass_calibration_comparison.csv", index=False
    )

    logger.info("Searching custom class-weight multipliers")
    weight_indices = stratified_sample_indices(y_strategy_train, 45_000)
    weight_results, selected_weights = class_weight_sensitivity(
        x_strategy_train.loc[weight_indices].reset_index(drop=True),
        y_strategy_train.loc[weight_indices].reset_index(drop=True),
        x_strategy_selection,
        y_strategy_selection,
        final_feature_columns,
        best_multiclass_params,
    )
    weight_results.to_csv(
        TABLE_DIR / "class_weight_sensitivity.csv", index=False
    )
    custom_weight_model = tuned_xgboost_pipeline(
        final_feature_columns, "multiclass", best_multiclass_params
    )
    custom_weight_model.fit(
        x_strategy_train,
        y_strategy_train,
        model__sample_weight=custom_multiclass_sample_weights(
            y_strategy_train, selected_weights[0], selected_weights[1]
        ),
    )
    custom_weight_probability = custom_weight_model.predict_proba(
        x_strategy_evaluation
    )

    logger.info("Fitting ordinal and two-stage multiclass alternatives")
    ordinal_models = fit_ordinal_models(
        x_strategy_train,
        y_strategy_train,
        final_feature_columns,
        best_binary_params,
    )
    ordinal_probability = ordinal_predict_proba(
        ordinal_models, x_strategy_evaluation
    )
    two_stage_models = fit_two_stage_models(
        x_strategy_train,
        y_strategy_train,
        final_feature_columns,
        best_binary_params,
    )
    two_stage_probability = two_stage_predict_proba(
        two_stage_models, x_strategy_evaluation
    )
    joblib.dump(
        two_stage_models,
        MODEL_DIR / "candidate_two_stage_multiclass_xgboost.joblib",
    )
    joblib.dump(
        multiclass_calibrator,
        MODEL_DIR / "candidate_multiclass_probability_calibrator.joblib",
    )
    write_json(
        MODEL_DIR / "candidate_probability_multipliers.json",
        {
            "class_0_multiplier": 1.0,
            "class_1_multiplier": float(
                selected_multiplier["class_1_multiplier"]
            ),
            "class_2_multiplier": float(
                selected_multiplier["class_2_multiplier"]
            ),
            "selection_split_macro_f1": float(
                selected_multiplier["macro_f1"]
            ),
        },
    )

    logger.info("Selecting an out-of-fold three-model ensemble")
    ensemble_indices = stratified_sample_indices(y_strategy_train, 30_000)
    ensemble_models = default_ensemble_models(
        final_feature_columns, best_multiclass_params
    )
    ensemble_search = oof_ensemble_weight_search(
        ensemble_models,
        x_strategy_train.loc[ensemble_indices].reset_index(drop=True),
        y_strategy_train.loc[ensemble_indices].reset_index(drop=True),
        xgboost_model_names={"XGBoost"},
    )
    ensemble_search.to_csv(
        TABLE_DIR / "multiclass_ensemble_weight_search.csv", index=False
    )
    best_ensemble = ensemble_search.iloc[0]
    fitted_ensemble = {}
    for name, estimator in ensemble_models.items():
        model = clone(estimator)
        fit_params: dict[str, object] = {}
        if name == "XGBoost":
            fit_params["model__sample_weight"] = balanced_sample_weights(
                y_strategy_train
            )
        model.fit(x_strategy_train, y_strategy_train, **fit_params)
        fitted_ensemble[name] = model.predict_proba(x_strategy_evaluation)
    ensemble_probability = sum(
        float(best_ensemble[f"weight_{name}"]) * probability
        for name, probability in fitted_ensemble.items()
    )

    advanced_results = pd.DataFrame(
        [
            _strategy_row(
                "Weighted XGBoost baseline",
                y_strategy_evaluation,
                baseline_evaluation_probability,
            ),
            _strategy_row(
                "Class-specific probability multipliers",
                y_strategy_evaluation,
                adjusted_probability,
                adjusted_prediction,
            ),
            _strategy_row(
                "Custom class weights",
                y_strategy_evaluation,
                custom_weight_probability,
            ),
            _strategy_row(
                "Multiclass probability calibration",
                y_strategy_evaluation,
                calibrated_evaluation_probability,
            ),
            _strategy_row(
                "Ordinal cumulative XGBoost",
                y_strategy_evaluation,
                ordinal_probability,
            ),
            _strategy_row(
                "Two-stage high-risk XGBoost",
                y_strategy_evaluation,
                two_stage_probability,
            ),
            _strategy_row(
                "OOF Logistic + ExtraTrees + XGBoost ensemble",
                y_strategy_evaluation,
                ensemble_probability,
            ),
        ]
    ).sort_values("macro_f1", ascending=False)
    advanced_results.to_csv(
        TABLE_DIR / "advanced_multiclass_strategy_comparison.csv",
        index=False,
    )

    logger.info("Running fold-safe moderate SMOTE-NC sensitivity analysis")
    smote_indices = stratified_sample_indices(y_strategy_train, 20_000)
    smote_results = smotenc_sensitivity_cv(
        x_strategy_train.loc[smote_indices].reset_index(drop=True),
        y_strategy_train.loc[smote_indices].reset_index(drop=True),
        final_feature_columns,
        best_multiclass_params,
    )
    smote_results.to_csv(
        TABLE_DIR / "smotenc_sensitivity.csv", index=False
    )

    logger.info("Computing official test-set bootstrap confidence intervals")
    final_binary = joblib.load(MODEL_DIR / "final_binary_xgboost.joblib")
    binary_calibrator = joblib.load(
        MODEL_DIR / "binary_probability_calibrator.joblib"
    )
    binary_threshold = float(
        pd.read_csv(TABLE_DIR / "final_test_metrics_binary.csv")
        .set_index("metric")
        .loc["threshold", "value"]
    )
    binary_raw_probability = final_binary.predict_proba(x_test)[:, 1]
    binary_probability = binary_calibrator.predict(binary_raw_probability)
    binary_prediction = (binary_probability >= binary_threshold).astype(int)
    confidence_intervals = bootstrap_confidence_intervals(
        y_multi_test,
        official_prediction,
        official_probability,
        y_binary_test,
        binary_prediction,
        binary_probability,
    )
    confidence_intervals.to_csv(
        TABLE_DIR / "bootstrap_confidence_intervals.csv", index=False
    )

    logger.info("Fitting paired comparison models on the official training split")
    extratrees = fitted_extratrees(final_feature_columns)
    extratrees.fit(x_train, y_multi_train)
    extratrees_probability = extratrees.predict_proba(x_test)
    extratrees_prediction = extratrees_probability.argmax(axis=1)

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
    binary_logistic_validation = fitted_binary_logistic(
        final_feature_columns
    )
    binary_logistic_validation.fit(x_binary_fit, y_binary_fit)
    logistic_validation_probability = (
        binary_logistic_validation.predict_proba(x_binary_validation)[:, 1]
    )
    logistic_threshold_table = threshold_analysis(
        y_binary_validation, logistic_validation_probability
    )
    logistic_threshold = float(
        logistic_threshold_table.loc[
            logistic_threshold_table["selection"]
            .fillna("")
            .str.contains("max_f1"),
            "threshold",
        ].iloc[0]
    )
    binary_logistic = fitted_binary_logistic(final_feature_columns)
    binary_logistic.fit(x_train, y_binary_train)
    logistic_probability = binary_logistic.predict_proba(x_test)[:, 1]
    logistic_prediction = (
        logistic_probability >= logistic_threshold
    ).astype(int)

    paired_rows = [
        paired_bootstrap_difference(
            y_multi_test,
            official_prediction,
            extratrees_prediction,
            lambda truth, prediction: f1_score(
                truth, prediction, average="macro"
            ),
            "Weighted XGBoost",
            "ExtraTrees",
            "multiclass_macro_f1",
        ),
        paired_bootstrap_difference(
            y_multi_test,
            official_prediction,
            extratrees_prediction,
            balanced_accuracy_score,
            "Weighted XGBoost",
            "ExtraTrees",
            "multiclass_balanced_accuracy",
        ),
        paired_bootstrap_difference(
            y_multi_test,
            official_prediction,
            extratrees_prediction,
            lambda truth, prediction: recall_score(
                truth,
                prediction,
                labels=[1],
                average=None,
                zero_division=0,
            )[0],
            "Weighted XGBoost",
            "ExtraTrees",
            "prediabetes_recall",
        ),
        paired_bootstrap_probability_difference(
            y_binary_test,
            binary_probability,
            logistic_probability,
            roc_auc_score,
            "Calibrated XGBoost",
            "Logistic Regression",
            "binary_roc_auc",
        ),
        paired_bootstrap_probability_difference(
            y_binary_test,
            binary_probability,
            logistic_probability,
            average_precision_score,
            "Calibrated XGBoost",
            "Logistic Regression",
            "binary_pr_auc",
        ),
    ]
    paired_comparison = pd.DataFrame(paired_rows)
    paired_comparison.to_csv(
        TABLE_DIR / "paired_model_bootstrap_comparison.csv", index=False
    )
    statistical_tests = pd.DataFrame(
        [
            mcnemar_exact_test(
                y_multi_test,
                official_prediction,
                extratrees_prediction,
                "Weighted XGBoost",
                "ExtraTrees",
            ),
            mcnemar_exact_test(
                y_binary_test,
                binary_prediction,
                logistic_prediction,
                "Calibrated XGBoost",
                "Threshold-tuned Logistic Regression",
            ),
        ]
    )
    statistical_tests.to_csv(
        TABLE_DIR / "paired_statistical_tests.csv", index=False
    )
    binary_logistic_comparison = pd.DataFrame(
        [
            {
                "model": "Calibrated XGBoost",
                "threshold": binary_threshold,
                **binary_metric_row(
                    y_binary_test, binary_prediction, binary_probability
                ),
            },
            {
                "model": "Threshold-tuned Logistic Regression",
                "threshold": logistic_threshold,
                **binary_metric_row(
                    y_binary_test,
                    logistic_prediction,
                    logistic_probability,
                ),
            },
        ]
    )
    binary_logistic_comparison.to_csv(
        TABLE_DIR / "binary_xgboost_logistic_comparison.csv", index=False
    )

    logger.info("Generating robustness figures and updating reports")
    chart_map = generate_robustness_figures(
        confidence_intervals,
        repeated_summary,
        grouped_results,
        advanced_results,
    )
    append_robustness_chart_map(chart_map)
    update_report_with_robustness(
        confidence_intervals=confidence_intervals,
        repeated_cv_summary=repeated_summary,
        feature_engineering_ablation=feature_ablation,
        grouped_split_results=grouped_results,
        advanced_strategy_results=advanced_results,
        calibration_comparison=calibration_comparison,
        smotenc_comparison=smote_results,
        paired_comparison=paired_comparison,
        statistical_tests=statistical_tests,
        threshold_tuning=multiplier_results,
        class_weight_results=weight_results,
    )
    html_path = write_html_report()

    best_advanced = advanced_results.iloc[0]
    robustness_summary = {
        "bootstrap_repetitions": 1000,
        "repeated_cv_folds": 15,
        "profile_grouped_test_rows": len(group_test_indices),
        "profile_group_overlap": group_summary["profile_overlap_count"],
        "profile_grouped_macro_f1": grouped_metrics["macro_f1"],
        "best_training_only_advanced_strategy": str(
            best_advanced["strategy"]
        ),
        "best_training_only_advanced_macro_f1": float(
            best_advanced["macro_f1"]
        ),
        "smotenc_macro_f1": float(
            smote_results.loc[
                smote_results["strategy"].eq("Moderate SMOTE-NC"),
                "macro_f1_mean",
            ].iloc[0]
        ),
        "report": str(html_path.relative_to(ROOT)),
    }
    write_json(REPORT_DIR / "robustness_summary.json", robustness_summary)
    run_summary_path = REPORT_DIR / "run_summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.exists()
        else {}
    )
    run_summary["robustness"] = robustness_summary
    write_json(run_summary_path, run_summary)

    print("\nBRFSS robustness analysis complete")
    print(json.dumps(robustness_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
