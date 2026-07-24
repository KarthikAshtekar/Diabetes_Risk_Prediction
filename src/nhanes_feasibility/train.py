from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocess import build_preprocessor


@dataclass
class RuleScoreModel:
    """Transparent points score with a multinomial calibration layer."""

    calibrator: LogisticRegression | None = None
    classes_: np.ndarray | None = None

    @staticmethod
    def score(frame: pd.DataFrame) -> np.ndarray:
        index = frame.index

        def numeric(column: str) -> pd.Series:
            if column in frame:
                return pd.to_numeric(frame[column], errors="coerce")
            return pd.Series(np.nan, index=index)

        age = numeric("RIDAGEYR")
        bmi = numeric("BMXBMI")
        waist = numeric("BMXWAIST")
        systolic = numeric("avg_systolic_bp")
        hypertension = numeric("hypertension_flag")
        inactivity = numeric("physical_inactivity_flag")
        smoker = numeric("smoker_flag")

        score = np.zeros(len(frame), dtype=float)
        score += np.select([age.ge(65), age.ge(55), age.ge(45), age.ge(35)], [4, 3, 2, 1], default=0)
        score += np.select([bmi.ge(35), bmi.ge(30), bmi.ge(25)], [3, 2, 1], default=0)
        score += np.select([waist.ge(110), waist.ge(100), waist.ge(90)], [3, 2, 1], default=0)
        score += np.select([systolic.ge(160), systolic.ge(140), systolic.ge(130)], [3, 2, 1], default=0)
        score += np.nan_to_num(hypertension.to_numpy(), nan=0.0) * 2
        score += np.nan_to_num(inactivity.to_numpy(), nan=0.0)
        score += np.nan_to_num(smoker.to_numpy(), nan=0.0)
        return score

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> RuleScoreModel:
        scores = self.score(frame).reshape(-1, 1)
        self.calibrator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        self.calibrator.fit(scores, target)
        self.classes_ = self.calibrator.classes_
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("RuleScoreModel must be fit before prediction.")
        return self.calibrator.predict_proba(self.score(frame).reshape(-1, 1))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(frame), axis=1)


def build_logistic_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_weight=3,
                    subsample=0.90,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
        ]
    )


def fit_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, BaseEstimator | RuleScoreModel]:
    models: dict[str, BaseEstimator | RuleScoreModel] = {}

    logistic = build_logistic_model(numeric_features, categorical_features)
    logistic.fit(x_train, y_train)
    models["Logistic Regression"] = logistic

    xgboost = build_xgboost_model(numeric_features, categorical_features)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    xgboost.fit(x_train, y_train, model__sample_weight=sample_weight)
    models["XGBoost"] = xgboost

    rule = RuleScoreModel().fit(x_train, y_train)
    models["Simple Rule Score"] = rule
    return models


def model_sanity_table(
    models: dict[str, BaseEstimator | RuleScoreModel],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        prediction = model.predict(x_train)
        rows.append(
            {
                "model": name,
                "train_balanced_accuracy": balanced_accuracy_score(y_train, prediction),
            }
        )
    return pd.DataFrame(rows)
