from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ORIGINAL_FEATURES, TARGET

EXPECTED_RANGES = {
    "Diabetes_012": (0, 2),
    "BMI": (10, 100),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}


def validate_brfss_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_values = set(pd.to_numeric(frame[TARGET], errors="coerce").dropna().astype(int))
    if not target_values.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected target values: {sorted(target_values)}")

    records: list[dict[str, object]] = []
    for column in [TARGET, *ORIGINAL_FEATURES]:
        available = column in frame.columns
        series = frame[column] if available else pd.Series(dtype=float)
        numeric = pd.to_numeric(series, errors="coerce")
        expected_range = EXPECTED_RANGES.get(column)
        if expected_range and available:
            range_valid = bool(
                numeric.dropna().between(*expected_range, inclusive="both").all()
            )
        elif available:
            unique = set(numeric.dropna().unique())
            range_valid = bool(unique.issubset({0, 1})) if column not in {
                "BMI",
                "MentHlth",
                "PhysHlth",
            } else True
        else:
            range_valid = False
        records.append(
            {
                "column": column,
                "available": available,
                "dtype": str(series.dtype) if available else "",
                "non_missing": int(series.notna().sum()) if available else 0,
                "unique_non_missing": int(series.nunique()) if available else 0,
                "minimum": float(numeric.min()) if available else np.nan,
                "maximum": float(numeric.max()) if available else np.nan,
                "range_valid": range_valid,
            }
        )
    schema = pd.DataFrame(records)

    missingness = pd.DataFrame(
        {
            "column": frame.columns,
            "missing": [int(frame[column].isna().sum()) for column in frame],
            "missing_pct": [float(frame[column].isna().mean()) for column in frame],
            "dtype": [str(frame[column].dtype) for column in frame],
        }
    )
    class_distribution = (
        frame[TARGET]
        .value_counts()
        .reindex([0, 1, 2], fill_value=0)
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = (
        class_distribution["count"] / len(frame)
    )
    class_distribution["class_name"] = class_distribution["class"].map(
        {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
    )

    summary = frame.describe(include="all").T.reset_index(names="column")
    summary["exact_duplicate_rows_in_dataset"] = int(frame.duplicated().sum())
    summary["duplicate_policy"] = (
        "Retained: no respondent identifier exists, so identical survey profiles "
        "cannot be proven to be duplicate people."
    )
    return schema, missingness, class_distribution, summary
