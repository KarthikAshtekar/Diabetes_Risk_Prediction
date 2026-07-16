from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ENGINEERED_CATEGORICAL


def preprocess_for_linear(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in feature_columns if c in ENGINEERED_CATEGORICAL]
    numeric = [c for c in feature_columns if c not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def preprocess_for_tree(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "features",
                SimpleImputer(strategy="median"),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
