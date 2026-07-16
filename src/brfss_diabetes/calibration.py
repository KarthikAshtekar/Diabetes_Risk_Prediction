from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidProbabilityCalibrator:
    """Platt-style calibration fitted only on validation probabilities."""

    def __init__(self) -> None:
        self.model = LogisticRegression()

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(
        self, probability: np.ndarray, target: np.ndarray
    ) -> "SigmoidProbabilityCalibrator":
        self.model.fit(self._logit(probability), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probability))[:, 1]
