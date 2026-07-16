from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .feature_engineering import BRFSSFeatureEngineer
from .preprocessing import preprocess_for_linear, preprocess_for_tree


def make_pipeline(model: object, feature_columns: list[str], kind: str) -> Pipeline:
    preprocessor = (
        preprocess_for_linear(feature_columns)
        if kind == "linear"
        else preprocess_for_tree(feature_columns)
    )
    return Pipeline(
        [
            ("feature_engineering", BRFSSFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def comparison_models(
    feature_columns: list[str],
    task: str,
) -> dict[str, Pipeline]:
    if task not in {"multiclass", "binary"}:
        raise ValueError(f"Unknown task: {task}")
    objective = "multi:softprob" if task == "multiclass" else "binary:logistic"
    eval_metric = "mlogloss" if task == "multiclass" else "logloss"
    xgb_extra = {"num_class": 3} if task == "multiclass" else {}
    return {
        "DummyClassifier": make_pipeline(
            DummyClassifier(strategy="prior"), feature_columns, "tree"
        ),
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            feature_columns,
            "linear",
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=90,
                max_depth=16,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            feature_columns,
            "tree",
        ),
        "ExtraTrees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=90,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            feature_columns,
            "tree",
        ),
        "XGBoost": make_pipeline(
            XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                n_estimators=180,
                max_depth=4,
                learning_rate=0.08,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                max_bin=128,
                **xgb_extra,
            ),
            feature_columns,
            "tree",
        ),
    }


def tuned_xgboost_pipeline(
    feature_columns: list[str],
    task: str,
    params: dict[str, object] | None = None,
) -> Pipeline:
    objective = "multi:softprob" if task == "multiclass" else "binary:logistic"
    eval_metric = "mlogloss" if task == "multiclass" else "logloss"
    defaults: dict[str, object] = {
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
        "max_bin": 128,
    }
    if task == "multiclass":
        defaults["num_class"] = 3
    if params:
        defaults.update(params)
    return make_pipeline(
        XGBClassifier(**defaults),
        feature_columns,
        "tree",
    )
