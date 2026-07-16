from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight


def balanced_sample_weights(target: pd.Series | np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=np.asarray(target))


def binary_scale_pos_weight(target: pd.Series | np.ndarray) -> float:
    values = np.asarray(target)
    positives = int((values == 1).sum())
    negatives = int((values == 0).sum())
    return negatives / positives if positives else 1.0
