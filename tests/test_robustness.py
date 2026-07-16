from __future__ import annotations

import numpy as np
import pandas as pd

from brfss_diabetes.robustness import (
    apply_probability_multipliers,
    grouped_profile_split,
    multiclass_brier_score,
)


def test_probability_multipliers_return_normalized_probabilities() -> None:
    probability = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
        ]
    )
    prediction, adjusted = apply_probability_multipliers(
        probability, class_1_multiplier=1.5, class_2_multiplier=0.8
    )
    assert prediction.shape == (2,)
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_grouped_split_has_no_profile_overlap() -> None:
    features = pd.DataFrame(
        {
            "a": np.repeat(np.arange(30), 2),
            "b": np.tile([0, 1], 30),
        }
    )
    features = pd.concat([features, features.iloc[:10]], ignore_index=True)
    target = pd.Series(np.tile([0, 1, 2, 0, 2], 14))
    train_indices, test_indices, summary = grouped_profile_split(
        features, target
    )
    assert len(train_indices) + len(test_indices) == len(features)
    assert summary["profile_overlap_count"] == 0


def test_multiclass_brier_is_zero_for_perfect_probabilities() -> None:
    target = np.array([0, 1, 2])
    probability = np.eye(3)
    assert multiclass_brier_score(target, probability) == 0.0
