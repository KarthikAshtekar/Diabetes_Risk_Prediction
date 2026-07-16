from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .feature_engineering import BRFSSFeatureEngineer
from .imbalance import balanced_sample_weights, binary_scale_pos_weight
from .models import comparison_models, tuned_xgboost_pipeline


def multiclass_metric_row(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray | None = None,
) -> dict[str, float]:
    truth = np.asarray(y_true)
    if probability is not None:
        probability = np.clip(np.asarray(probability, dtype=float), 1e-12, None)
        probability = probability / probability.sum(axis=1, keepdims=True)
    recalls = recall_score(
        truth,
        prediction,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    precisions = precision_score(
        truth,
        prediction,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    row = {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "class_0_recall": recalls[0],
        "class_1_recall": recalls[1],
        "class_2_recall": recalls[2],
        "class_0_precision": precisions[0],
        "class_1_precision": precisions[1],
        "class_2_precision": precisions[2],
    }
    if probability is not None:
        row["macro_roc_auc_ovr"] = roc_auc_score(
            truth, probability, multi_class="ovr", average="macro"
        )
        row["multiclass_log_loss"] = log_loss(
            truth, probability, labels=[0, 1, 2]
        )
        row["multiclass_brier_score"] = multiclass_brier_score(
            truth, probability
        )
        row["multiclass_ece"] = multiclass_expected_calibration_error(
            truth, probability
        )
    return row


def binary_metric_row(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(y_true)
    return {
        "roc_auc": roc_auc_score(truth, probability),
        "pr_auc": average_precision_score(truth, probability),
        "recall": recall_score(truth, prediction, zero_division=0),
        "precision": precision_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "brier_score": brier_score_loss(truth, probability),
    }


def multiclass_brier_score(
    y_true: pd.Series | np.ndarray, probability: np.ndarray
) -> float:
    truth = np.asarray(y_true, dtype=int)
    one_hot = np.eye(3)[truth]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def multiclass_expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    truth = np.asarray(y_true, dtype=int)
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == truth
    boundaries = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


class MulticlassLogitCalibrator:
    """Multinomial calibration fitted to log probabilities."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)

    @staticmethod
    def _features(probability: np.ndarray) -> np.ndarray:
        return np.log(np.clip(np.asarray(probability), 1e-7, 1.0))

    def fit(
        self,
        probability: np.ndarray,
        target: pd.Series | np.ndarray,
    ) -> MulticlassLogitCalibrator:
        self.model.fit(self._features(probability), np.asarray(target))
        return self

    def predict_proba(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._features(probability))


def _stratified_bootstrap_indices(
    target: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    sampled = []
    for label in np.unique(target):
        indices = np.flatnonzero(target == label)
        sampled.append(generator.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(sampled)
    generator.shuffle(result)
    return result


def bootstrap_confidence_intervals(
    y_multiclass: pd.Series | np.ndarray,
    multiclass_prediction: np.ndarray,
    multiclass_probability: np.ndarray,
    y_binary: pd.Series | np.ndarray,
    binary_prediction: np.ndarray,
    binary_probability: np.ndarray,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    y_multi = np.asarray(y_multiclass)
    y_bin = np.asarray(y_binary)
    generator = np.random.default_rng(RANDOM_STATE)
    metric_values: dict[str, list[float]] = {
        "multiclass_macro_f1": [],
        "multiclass_balanced_accuracy": [],
        "prediabetes_recall": [],
        "diabetes_recall": [],
        "multiclass_macro_roc_auc_ovr": [],
        "binary_roc_auc": [],
        "binary_pr_auc": [],
        "binary_recall": [],
        "binary_precision": [],
        "binary_f1": [],
    }
    for _ in range(n_bootstrap):
        indices = _stratified_bootstrap_indices(y_multi, generator)
        truth_multi = y_multi[indices]
        prediction_multi = multiclass_prediction[indices]
        probability_multi = multiclass_probability[indices]
        truth_binary = y_bin[indices]
        prediction_binary = binary_prediction[indices]
        probability_binary = binary_probability[indices]
        recalls = recall_score(
            truth_multi,
            prediction_multi,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        metric_values["multiclass_macro_f1"].append(
            f1_score(truth_multi, prediction_multi, average="macro")
        )
        metric_values["multiclass_balanced_accuracy"].append(
            balanced_accuracy_score(truth_multi, prediction_multi)
        )
        metric_values["prediabetes_recall"].append(recalls[1])
        metric_values["diabetes_recall"].append(recalls[2])
        metric_values["multiclass_macro_roc_auc_ovr"].append(
            roc_auc_score(
                truth_multi,
                probability_multi,
                multi_class="ovr",
                average="macro",
            )
        )
        metric_values["binary_roc_auc"].append(
            roc_auc_score(truth_binary, probability_binary)
        )
        metric_values["binary_pr_auc"].append(
            average_precision_score(truth_binary, probability_binary)
        )
        metric_values["binary_recall"].append(
            recall_score(truth_binary, prediction_binary, zero_division=0)
        )
        metric_values["binary_precision"].append(
            precision_score(truth_binary, prediction_binary, zero_division=0)
        )
        metric_values["binary_f1"].append(
            f1_score(truth_binary, prediction_binary, zero_division=0)
        )
    point_metrics = {
        "multiclass_macro_f1": f1_score(
            y_multi, multiclass_prediction, average="macro"
        ),
        "multiclass_balanced_accuracy": balanced_accuracy_score(
            y_multi, multiclass_prediction
        ),
        "prediabetes_recall": recall_score(
            y_multi,
            multiclass_prediction,
            labels=[1],
            average=None,
            zero_division=0,
        )[0],
        "diabetes_recall": recall_score(
            y_multi,
            multiclass_prediction,
            labels=[2],
            average=None,
            zero_division=0,
        )[0],
        "multiclass_macro_roc_auc_ovr": roc_auc_score(
            y_multi,
            multiclass_probability,
            multi_class="ovr",
            average="macro",
        ),
        "binary_roc_auc": roc_auc_score(y_bin, binary_probability),
        "binary_pr_auc": average_precision_score(y_bin, binary_probability),
        "binary_recall": recall_score(
            y_bin, binary_prediction, zero_division=0
        ),
        "binary_precision": precision_score(
            y_bin, binary_prediction, zero_division=0
        ),
        "binary_f1": f1_score(y_bin, binary_prediction, zero_division=0),
    }
    rows = []
    for metric, values in metric_values.items():
        distribution = np.asarray(values)
        rows.append(
            {
                "metric": metric,
                "estimate": point_metrics[metric],
                "ci_lower_95": np.quantile(distribution, 0.025),
                "ci_upper_95": np.quantile(distribution, 0.975),
                "bootstrap_repetitions": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    y_true: pd.Series | np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    model_a: str,
    model_b: str,
    metric_name: str,
    n_bootstrap: int = 1000,
) -> dict[str, object]:
    truth = np.asarray(y_true)
    generator = np.random.default_rng(RANDOM_STATE + 11)
    differences = []
    for _ in range(n_bootstrap):
        indices = _stratified_bootstrap_indices(truth, generator)
        differences.append(
            metric(truth[indices], prediction_a[indices])
            - metric(truth[indices], prediction_b[indices])
        )
    distribution = np.asarray(differences)
    estimate = metric(truth, prediction_a) - metric(truth, prediction_b)
    return {
        "metric": metric_name,
        "model_a": model_a,
        "model_b": model_b,
        "difference_a_minus_b": estimate,
        "ci_lower_95": np.quantile(distribution, 0.025),
        "ci_upper_95": np.quantile(distribution, 0.975),
        "probability_difference_positive": float((distribution > 0).mean()),
        "bootstrap_repetitions": n_bootstrap,
    }


def paired_bootstrap_probability_difference(
    y_true: pd.Series | np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    model_a: str,
    model_b: str,
    metric_name: str,
    n_bootstrap: int = 1000,
) -> dict[str, object]:
    truth = np.asarray(y_true)
    generator = np.random.default_rng(RANDOM_STATE + 23)
    differences = []
    for _ in range(n_bootstrap):
        indices = _stratified_bootstrap_indices(truth, generator)
        differences.append(
            metric(truth[indices], probability_a[indices])
            - metric(truth[indices], probability_b[indices])
        )
    distribution = np.asarray(differences)
    estimate = metric(truth, probability_a) - metric(truth, probability_b)
    return {
        "metric": metric_name,
        "model_a": model_a,
        "model_b": model_b,
        "difference_a_minus_b": estimate,
        "ci_lower_95": np.quantile(distribution, 0.025),
        "ci_upper_95": np.quantile(distribution, 0.975),
        "probability_difference_positive": float((distribution > 0).mean()),
        "bootstrap_repetitions": n_bootstrap,
    }


def mcnemar_exact_test(
    y_true: pd.Series | np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    model_a: str,
    model_b: str,
) -> dict[str, object]:
    truth = np.asarray(y_true)
    correct_a = prediction_a == truth
    correct_b = prediction_b == truth
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = (
        float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "test": "exact_mcnemar",
        "model_a": model_a,
        "model_b": model_b,
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "p_value_display": "<1e-300" if p_value == 0 else f"{p_value:.3e}",
    }


def profile_group_ids(features: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


def grouped_profile_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    groups = profile_group_ids(features)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_indices, test_indices = next(
        splitter.split(features, target, groups=groups)
    )
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    summary = {
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "unique_profiles": len(np.unique(groups)),
        "train_unique_profiles": len(train_groups),
        "test_unique_profiles": len(test_groups),
        "profile_overlap_count": len(train_groups & test_groups),
    }
    return train_indices, test_indices, summary


def repeated_model_comparison(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    xgboost_model_names: set[str],
    n_splits: int = 5,
    n_repeats: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]
        for name, estimator in models.items():
            model = clone(estimator)
            fit_params: dict[str, object] = {}
            if name in xgboost_model_names:
                fit_params["model__sample_weight"] = balanced_sample_weights(
                    y_train
                )
            model.fit(x_train, y_train, **fit_params)
            probability = model.predict_proba(x_validation)
            prediction = probability.argmax(axis=1)
            row = {
                "model": name,
                "fold": fold,
                **multiclass_metric_row(
                    y_validation, prediction, probability
                ),
            }
            rows.append(row)
    folds = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in folds.columns
        if column not in {"model", "fold"}
    ]
    summary_rows = []
    for model, part in folds.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "folds": len(part),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_std"] = part[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    return folds, summary


def tune_probability_multipliers(
    y_true: pd.Series,
    probability: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for class_1_multiplier, class_2_multiplier in product(
        np.linspace(0.5, 3.0, 11),
        np.linspace(0.6, 1.6, 6),
    ):
        adjusted = probability * np.array(
            [1.0, class_1_multiplier, class_2_multiplier]
        )
        prediction = adjusted.argmax(axis=1)
        rows.append(
            {
                "class_0_multiplier": 1.0,
                "class_1_multiplier": class_1_multiplier,
                "class_2_multiplier": class_2_multiplier,
                **multiclass_metric_row(y_true, prediction),
            }
        )
    result = pd.DataFrame(rows)
    result["selection"] = ""
    result.loc[result["macro_f1"].idxmax(), "selection"] = "max_macro_f1"
    result.loc[result["class_1_recall"].idxmax(), "selection"] += (
        "|max_prediabetes_recall"
    )
    return result.sort_values("macro_f1", ascending=False)


def apply_probability_multipliers(
    probability: np.ndarray,
    class_1_multiplier: float,
    class_2_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = probability * np.array(
        [1.0, class_1_multiplier, class_2_multiplier]
    )
    normalized = adjusted / adjusted.sum(axis=1, keepdims=True)
    return normalized.argmax(axis=1), normalized


def custom_multiclass_sample_weights(
    target: pd.Series | np.ndarray,
    class_1_multiplier: float,
    class_2_multiplier: float,
) -> np.ndarray:
    target_array = np.asarray(target)
    weights = balanced_sample_weights(target_array)
    weights[target_array == 1] *= class_1_multiplier
    weights[target_array == 2] *= class_2_multiplier
    return weights / weights.mean()


def class_weight_sensitivity(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_selection: pd.DataFrame,
    y_selection: pd.Series,
    feature_columns: list[str],
    best_params: dict[str, object],
) -> tuple[pd.DataFrame, tuple[float, float]]:
    rows = []
    candidates = list(
        product(
            [0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
            [0.75, 1.0, 1.25],
        )
    )
    for class_1_multiplier, class_2_multiplier in candidates:
        model = tuned_xgboost_pipeline(
            feature_columns, "multiclass", best_params
        )
        model.fit(
            x_train,
            y_train,
            model__sample_weight=custom_multiclass_sample_weights(
                y_train, class_1_multiplier, class_2_multiplier
            ),
        )
        probability = model.predict_proba(x_selection)
        prediction = probability.argmax(axis=1)
        rows.append(
            {
                "class_1_multiplier": class_1_multiplier,
                "class_2_multiplier": class_2_multiplier,
                **multiclass_metric_row(
                    y_selection, prediction, probability
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result["selection"] = ""
    result.loc[result["macro_f1"].idxmax(), "selection"] = "max_macro_f1"
    best = result.iloc[0]
    return result, (
        float(best["class_1_multiplier"]),
        float(best["class_2_multiplier"]),
    )


def fit_ordinal_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    best_params: dict[str, object],
) -> tuple[Pipeline, Pipeline]:
    greater_than_zero = y_train.gt(0).astype(int)
    greater_than_one = y_train.gt(1).astype(int)
    first = tuned_xgboost_pipeline(feature_columns, "binary", best_params)
    second = tuned_xgboost_pipeline(feature_columns, "binary", best_params)
    first.set_params(
        model__scale_pos_weight=binary_scale_pos_weight(greater_than_zero)
    )
    second.set_params(
        model__scale_pos_weight=binary_scale_pos_weight(greater_than_one)
    )
    first.fit(x_train, greater_than_zero)
    second.fit(x_train, greater_than_one)
    return first, second


def ordinal_predict_proba(
    models: tuple[Pipeline, Pipeline],
    features: pd.DataFrame,
) -> np.ndarray:
    probability_gt_zero = models[0].predict_proba(features)[:, 1]
    probability_gt_one = models[1].predict_proba(features)[:, 1]
    probability_gt_one = np.minimum(probability_gt_one, probability_gt_zero)
    probability = np.column_stack(
        [
            1 - probability_gt_zero,
            probability_gt_zero - probability_gt_one,
            probability_gt_one,
        ]
    )
    return probability / probability.sum(axis=1, keepdims=True)


def fit_two_stage_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    best_params: dict[str, object],
) -> tuple[Pipeline, Pipeline]:
    high_risk = y_train.gt(0).astype(int)
    first = tuned_xgboost_pipeline(feature_columns, "binary", best_params)
    first.set_params(
        model__scale_pos_weight=binary_scale_pos_weight(high_risk)
    )
    first.fit(x_train, high_risk)

    high_risk_mask = y_train.gt(0)
    second_target = y_train.loc[high_risk_mask].eq(2).astype(int)
    second = tuned_xgboost_pipeline(feature_columns, "binary", best_params)
    second.set_params(
        model__scale_pos_weight=binary_scale_pos_weight(second_target)
    )
    second.fit(x_train.loc[high_risk_mask], second_target)
    return first, second


def two_stage_predict_proba(
    models: tuple[Pipeline, Pipeline],
    features: pd.DataFrame,
) -> np.ndarray:
    probability_high_risk = models[0].predict_proba(features)[:, 1]
    probability_diabetes_given_high_risk = models[1].predict_proba(features)[:, 1]
    probability = np.column_stack(
        [
            1 - probability_high_risk,
            probability_high_risk
            * (1 - probability_diabetes_given_high_risk),
            probability_high_risk * probability_diabetes_given_high_risk,
        ]
    )
    return probability / probability.sum(axis=1, keepdims=True)


def search_ensemble_weights(
    probabilities: dict[str, np.ndarray],
    target: pd.Series,
    step: float = 0.1,
) -> pd.DataFrame:
    names = list(probabilities)
    if len(names) != 3:
        raise ValueError("The ensemble search expects exactly three models.")
    rows = []
    units = round(1 / step)
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            weights = np.array([first, second, third], dtype=float) / units
            combined = sum(
                weight * probabilities[name]
                for weight, name in zip(weights, names, strict=True)
            )
            combined = np.clip(combined, 1e-12, None)
            combined = combined / combined.sum(axis=1, keepdims=True)
            prediction = combined.argmax(axis=1)
            rows.append(
                {
                    f"weight_{names[0]}": weights[0],
                    f"weight_{names[1]}": weights[1],
                    f"weight_{names[2]}": weights[2],
                    **multiclass_metric_row(target, prediction, combined),
                }
            )
    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result["selection"] = ""
    result.loc[result["macro_f1"].idxmax(), "selection"] = "max_macro_f1"
    return result


def oof_ensemble_weight_search(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    xgboost_model_names: set[str],
) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    probabilities = {
        name: np.zeros((len(features), 3), dtype=float) for name in models
    }
    for train_indices, validation_indices in splitter.split(features, target):
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        for name, estimator in models.items():
            model = clone(estimator)
            fit_params: dict[str, object] = {}
            if name in xgboost_model_names:
                fit_params["model__sample_weight"] = balanced_sample_weights(
                    y_train
                )
            model.fit(x_train, y_train, **fit_params)
            probabilities[name][validation_indices] = model.predict_proba(
                x_validation
            )
    return search_ensemble_weights(probabilities, target)


def moderate_smotenc_sampling_strategy(
    target: pd.Series | np.ndarray,
) -> dict[int, int]:
    values, counts = np.unique(np.asarray(target), return_counts=True)
    current = dict(zip(values.astype(int), counts.astype(int), strict=True))
    majority = max(current.values())
    desired = {
        1: max(current.get(1, 0), round(majority * 0.20)),
        2: max(current.get(2, 0), round(majority * 0.45)),
    }
    return {
        label: count
        for label, count in desired.items()
        if label in current and count > current[label]
    }


def smotenc_pipeline(
    feature_columns: list[str],
    best_params: dict[str, object],
) -> ImbalancedPipeline:
    continuous = {
        "BMI",
        "MentHlth",
        "PhysHlth",
        "bmi_age_interaction",
        "bmi_highbp_interaction",
        "bmi_highchol_interaction",
        "total_unhealthy_days",
        "health_burden_score",
        "income_education_interaction",
        "age_bmi_interaction",
        "age_cardiometabolic_interaction",
    }
    categorical_indices = [
        index
        for index, column in enumerate(feature_columns)
        if column not in continuous
    ]
    parameters = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
        "max_bin": 128,
        **best_params,
    }
    return ImbalancedPipeline(
        [
            ("feature_engineering", BRFSSFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "smotenc",
                SMOTENC(
                    categorical_features=categorical_indices,
                    sampling_strategy=moderate_smotenc_sampling_strategy,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("model", XGBClassifier(**parameters)),
        ]
    )


def smotenc_sensitivity_cv(
    features: pd.DataFrame,
    target: pd.Series,
    feature_columns: list[str],
    best_params: dict[str, object],
) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    baseline = tuned_xgboost_pipeline(
        feature_columns, "multiclass", best_params
    )
    smote = smotenc_pipeline(feature_columns, best_params)
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        y_validation = target.iloc[validation_indices]
        weighted = clone(baseline)
        weighted.fit(
            x_train,
            y_train,
            model__sample_weight=balanced_sample_weights(y_train),
        )
        for name, model in (
            ("Balanced sample weights", weighted),
            ("Moderate SMOTE-NC", clone(smote)),
        ):
            if name == "Moderate SMOTE-NC":
                model.fit(x_train, y_train)
            probability = model.predict_proba(x_validation)
            prediction = probability.argmax(axis=1)
            rows.append(
                {
                    "strategy": name,
                    "fold": fold,
                    **multiclass_metric_row(
                        y_validation, prediction, probability
                    ),
                }
            )
    folds = pd.DataFrame(rows)
    return (
        folds.groupby("strategy")
        .agg(
            folds=("fold", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            class_1_recall_mean=("class_1_recall", "mean"),
            class_1_precision_mean=("class_1_precision", "mean"),
            class_2_recall_mean=("class_2_recall", "mean"),
            multiclass_log_loss_mean=("multiclass_log_loss", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )


def default_ensemble_models(
    feature_columns: list[str],
    best_params: dict[str, object],
) -> dict[str, Pipeline]:
    comparison = comparison_models(feature_columns, "multiclass")
    return {
        "Logistic": comparison["Logistic Regression"],
        "ExtraTrees": comparison["ExtraTrees"],
        "XGBoost": tuned_xgboost_pipeline(
            feature_columns, "multiclass", best_params
        ),
    }


def fitted_extratrees(
    feature_columns: list[str],
) -> Pipeline:
    comparison = comparison_models(feature_columns, "multiclass")
    return comparison["ExtraTrees"]


def fitted_binary_logistic(
    feature_columns: list[str],
) -> Pipeline:
    return comparison_models(feature_columns, "binary")["Logistic Regression"]


def lightweight_extratrees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=90,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
