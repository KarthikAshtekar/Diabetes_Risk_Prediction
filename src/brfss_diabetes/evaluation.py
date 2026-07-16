from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from .config import CLASS_NAMES, CV_FOLDS, RANDOM_STATE
from .imbalance import balanced_sample_weights


def compare_models_cv(
    models: dict[str, object],
    x: pd.DataFrame,
    y: pd.Series,
    task: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for name, estimator in models.items():
        fold_rows: list[dict[str, float]] = []
        for train_index, validation_index in splitter.split(x, y):
            model = clone(estimator)
            x_train = x.iloc[train_index]
            y_train = y.iloc[train_index]
            x_valid = x.iloc[validation_index]
            y_valid = y.iloc[validation_index]
            fit_params: dict[str, object] = {}
            if name == "XGBoost" and task == "multiclass":
                fit_params["model__sample_weight"] = balanced_sample_weights(y_train)
            model.fit(x_train, y_train, **fit_params)
            prediction = model.predict(x_valid)
            probability = model.predict_proba(x_valid)
            if task == "multiclass":
                recalls = recall_score(
                    y_valid,
                    prediction,
                    labels=[0, 1, 2],
                    average=None,
                    zero_division=0,
                )
                try:
                    auc = roc_auc_score(
                        y_valid, probability, multi_class="ovr", average="macro"
                    )
                except ValueError:
                    auc = np.nan
                fold_rows.append(
                    {
                        "macro_f1": f1_score(y_valid, prediction, average="macro"),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_valid, prediction
                        ),
                        "class_0_recall": recalls[0],
                        "class_1_recall": recalls[1],
                        "class_2_recall": recalls[2],
                        "macro_roc_auc_ovr": auc,
                    }
                )
            else:
                positive_probability = probability[:, 1]
                fold_rows.append(
                    {
                        "pr_auc": average_precision_score(
                            y_valid, positive_probability
                        ),
                        "roc_auc": roc_auc_score(y_valid, positive_probability),
                        "f1": f1_score(y_valid, prediction, zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_valid, prediction
                        ),
                        "recall": recall_score(
                            y_valid, prediction, zero_division=0
                        ),
                        "precision": precision_score(
                            y_valid, prediction, zero_division=0
                        ),
                    }
                )
        metrics = pd.DataFrame(fold_rows)
        row: dict[str, object] = {"model": name, "cv_folds": CV_FOLDS}
        for column in metrics:
            row[column] = metrics[column].mean()
            row[f"{column}_std"] = metrics[column].std(ddof=1)
        rows.append(row)
    primary = "macro_f1" if task == "multiclass" else "pr_auc"
    return pd.DataFrame(rows).sort_values(primary, ascending=False)


def evaluate_multiclass(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)
    binarized = label_binarize(y_test, classes=[0, 1, 2])
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "macro_f1": f1_score(y_test, prediction, average="macro"),
        "weighted_f1": f1_score(y_test, prediction, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "macro_roc_auc_ovr": roc_auc_score(
            y_test, probability, multi_class="ovr", average="macro"
        ),
        "macro_pr_auc_ovr": float(
            np.mean(
                [
                    average_precision_score(binarized[:, i], probability[:, i])
                    for i in range(3)
                ]
            )
        ),
    }
    report = pd.DataFrame(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1, 2],
            target_names=[CLASS_NAMES[i] for i in range(3)],
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="class")
    metric_table = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    details = {
        "prediction": prediction,
        "probability": probability,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1, 2]),
    }
    predictions = pd.DataFrame(
        {
            "row_index": y_test.index,
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "probability_class_0": probability[:, 0],
            "probability_class_1": probability[:, 1],
            "probability_class_2": probability[:, 2],
            "probability_high_risk": probability[:, 1] + probability[:, 2],
        }
    )
    return metric_table, report, details, predictions


def evaluate_binary(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    calibrated_probability: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    raw_probability = model.predict_proba(x_test)[:, 1]
    probability = (
        np.asarray(calibrated_probability)
        if calibrated_probability is not None
        else raw_probability
    )
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, probability),
        "pr_auc": average_precision_score(y_test, probability),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "brier_score": brier_score_loss(y_test, probability),
    }
    report = pd.DataFrame(
        classification_report(
            y_test,
            prediction,
            target_names=["No diabetes", "Diabetes"],
            output_dict=True,
            zero_division=0,
        )
    ).T.reset_index(names="class")
    fpr, tpr, roc_thresholds = roc_curve(y_test, probability)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, probability)
    details = {
        "prediction": prediction,
        "probability": probability,
        "raw_probability": raw_probability,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]),
        "roc_curve": (fpr, tpr, roc_thresholds),
        "pr_curve": (precision, recall, pr_thresholds),
    }
    predictions = pd.DataFrame(
        {
            "row_index": y_test.index,
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "raw_probability": raw_probability,
            "calibrated_probability": probability,
        }
    )
    return (
        pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()]),
        report,
        details,
        predictions,
    )


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    minimum_precision: float = 0.35,
) -> pd.DataFrame:
    truth = np.asarray(y_true)
    rows: list[dict[str, object]] = []
    for threshold in np.linspace(0.05, 0.90, 86):
        prediction = (probability >= threshold).astype(int)
        precision = precision_score(truth, prediction, zero_division=0)
        recall = recall_score(truth, prediction, zero_division=0)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1_score(truth, prediction, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "predicted_positive_rate": prediction.mean(),
                "meets_minimum_precision": precision >= minimum_precision,
            }
        )
    result = pd.DataFrame(rows)
    result["selection"] = ""
    result.loc[result["f1"].idxmax(), "selection"] += "max_f1"
    result.loc[result["balanced_accuracy"].idxmax(), "selection"] += (
        "|balanced_precision_recall"
    )
    high_recall_candidates = result.loc[result["recall"].ge(0.85)]
    if not high_recall_candidates.empty:
        result.loc[
            high_recall_candidates["precision"].idxmax(), "selection"
        ] += "|high_recall"
    precision_candidates = result.loc[result["meets_minimum_precision"]]
    if not precision_candidates.empty:
        result.loc[
            precision_candidates["recall"].idxmax(), "selection"
        ] += "|minimum_precision"
    return result


def high_risk_threshold_analysis(
    y_multiclass: pd.Series | np.ndarray,
    high_risk_probability: np.ndarray,
) -> pd.DataFrame:
    truth = (np.asarray(y_multiclass) > 0).astype(int)
    rows = threshold_analysis(truth, high_risk_probability, minimum_precision=0.20)
    rows = rows.rename(
        columns={
            "recall": "high_risk_recall",
            "precision": "high_risk_precision",
        }
    )
    order = np.argsort(-high_risk_probability, kind="stable")
    for k in (0.10, 0.20, 0.30, 0.40, 0.50):
        tested = max(1, math.ceil(len(truth) * k))
        selected = truth[order[:tested]]
        rows.loc[len(rows)] = {
            "threshold": float(high_risk_probability[order[tested - 1]]),
            "high_risk_precision": float(selected.mean()),
            "high_risk_recall": float(selected.sum() / truth.sum()),
            "f1": np.nan,
            "balanced_accuracy": np.nan,
            "predicted_positive_rate": tested / len(truth),
            "meets_minimum_precision": True,
            "selection": f"top_{int(k * 100)}_percent",
        }
    return rows
