from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {
    0: "Normal / lower-risk",
    1: "Diabetes-prone / prediabetes-risk",
    2: "Diabetic / diabetes-range",
}


def _aligned_probabilities(model: object, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(getattr(model, "classes_", CLASS_LABELS), dtype=int)
    aligned = np.zeros((len(frame), 3), dtype=float)
    for index, class_value in enumerate(classes):
        aligned[:, int(class_value)] = probabilities[:, index]
    return aligned


def evaluate_models(
    models: dict[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    comparison_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    details: dict[str, dict[str, object]] = {}

    y_array = y_test.to_numpy(dtype=int)
    binary_truth = (y_array > 0).astype(int)

    for name, model in models.items():
        probabilities = _aligned_probabilities(model, x_test)
        prediction = np.argmax(probabilities, axis=1)
        high_risk_probability = probabilities[:, 1] + probabilities[:, 2]
        high_risk_prediction = (prediction > 0).astype(int)

        class_recall = recall_score(
            y_array,
            prediction,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )
        try:
            roc_auc = roc_auc_score(binary_truth, high_risk_probability)
        except ValueError:
            roc_auc = np.nan

        comparison_rows.append(
            {
                "model": name,
                "macro_f1": f1_score(
                    y_array,
                    prediction,
                    labels=CLASS_LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "class_0_recall": class_recall[0],
                "class_1_recall": class_recall[1],
                "class_2_recall": class_recall[2],
                "high_risk_recall": recall_score(
                    binary_truth, high_risk_prediction, zero_division=0
                ),
                "high_risk_precision": precision_score(
                    binary_truth, high_risk_prediction, zero_division=0
                ),
                "high_risk_pr_auc": average_precision_score(
                    binary_truth, high_risk_probability
                ),
                "high_risk_roc_auc": roc_auc,
                "high_risk_brier_score": brier_score_loss(
                    binary_truth, high_risk_probability
                ),
            }
        )
        for class_value, recall_value in zip(CLASS_LABELS, class_recall):
            recall_rows.append(
                {
                    "model": name,
                    "class": class_value,
                    "class_name": CLASS_NAMES[class_value],
                    "recall": recall_value,
                }
            )

        fraction_positive, mean_predicted = calibration_curve(
            binary_truth,
            high_risk_probability,
            n_bins=10,
            strategy="quantile",
        )
        details[name] = {
            "probabilities": probabilities,
            "prediction": prediction,
            "high_risk_probability": high_risk_probability,
            "confusion_matrix": confusion_matrix(
                y_array, prediction, labels=CLASS_LABELS
            ),
            "calibration_fraction_positive": fraction_positive,
            "calibration_mean_predicted": mean_predicted,
        }

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "high_risk_pr_auc", ascending=False
    )
    return comparison, pd.DataFrame(recall_rows), details
