from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from .config import CV_FOLDS, RANDOM_STATE
from .imbalance import balanced_sample_weights, binary_scale_pos_weight
from .models import make_pipeline, tuned_xgboost_pipeline


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("model__"): value
        for key, value in params.items()
        if key.startswith("model__")
    }


def tune_logistic(
    x: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    scoring: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pipeline = make_pipeline(
        LogisticRegression(
            max_iter=1800,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        feature_columns,
        "linear",
    )
    search = GridSearchCV(
        pipeline,
        {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        scoring=scoring,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=2,
        refit=False,
        return_train_score=False,
    )
    search.fit(x, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return _clean_params(search.best_params_), results


def tune_random_forest(
    x: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    pipeline = make_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        feature_columns,
        "tree",
    )
    search = RandomizedSearchCV(
        pipeline,
        {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [10, 14, 18, None],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", 0.7, 1.0],
        },
        n_iter=3,
        scoring="f1_macro",
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=False,
        return_train_score=False,
    )
    search.fit(x, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return _clean_params(search.best_params_), results


def tune_xgboost(
    x: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    task: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pipeline = tuned_xgboost_pipeline(feature_columns, task)
    if task == "binary":
        pipeline.set_params(
            model__scale_pos_weight=binary_scale_pos_weight(y)
        )
    parameter_distributions = {
        "model__n_estimators": randint(180, 421),
        "model__max_depth": randint(3, 7),
        "model__learning_rate": loguniform(0.025, 0.14),
        "model__subsample": uniform(0.72, 0.28),
        "model__colsample_bytree": uniform(0.72, 0.28),
        "model__min_child_weight": randint(1, 8),
        "model__gamma": uniform(0.0, 1.5),
        "model__reg_alpha": loguniform(1e-4, 1.0),
        "model__reg_lambda": loguniform(0.5, 8.0),
    }
    scoring = "f1_macro" if task == "multiclass" else "average_precision"
    search = RandomizedSearchCV(
        pipeline,
        parameter_distributions,
        n_iter=5,
        scoring=scoring,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=False,
        return_train_score=False,
        verbose=1,
    )
    fit_params: dict[str, object] = {}
    if task == "multiclass":
        fit_params["model__sample_weight"] = balanced_sample_weights(y)
    search.fit(x, y, **fit_params)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best = _clean_params(search.best_params_)
    for key, value in list(best.items()):
        if isinstance(value, np.generic):
            best[key] = value.item()
    return best, results
