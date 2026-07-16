from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from brfss_diabetes.config import TABLE_DIR


def test_saved_metrics_recompute() -> None:
    multi_predictions = pd.read_csv(
        TABLE_DIR / "multiclass_test_predictions.csv"
    )
    multi_metrics = pd.read_csv(
        TABLE_DIR / "final_test_metrics_multiclass.csv"
    ).set_index("metric")["value"]
    assert np.isclose(
        f1_score(
            multi_predictions["actual"],
            multi_predictions["predicted"],
            average="macro",
        ),
        multi_metrics["macro_f1"],
        atol=1e-12,
    )
    assert np.isclose(
        balanced_accuracy_score(
            multi_predictions["actual"], multi_predictions["predicted"]
        ),
        multi_metrics["balanced_accuracy"],
        atol=1e-12,
    )

    binary_predictions = pd.read_csv(TABLE_DIR / "binary_test_predictions.csv")
    binary_metrics = pd.read_csv(
        TABLE_DIR / "final_test_metrics_binary.csv"
    ).set_index("metric")["value"]
    probability = binary_predictions["calibrated_probability"]
    assert np.isclose(
        roc_auc_score(binary_predictions["actual"], probability),
        binary_metrics["roc_auc"],
        atol=1e-12,
    )
    assert np.isclose(
        average_precision_score(binary_predictions["actual"], probability),
        binary_metrics["pr_auc"],
        atol=1e-12,
    )
