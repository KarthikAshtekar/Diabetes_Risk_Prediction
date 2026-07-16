from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import NHANES_FILES, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)


def load_available_components(
    raw_dir: Path = RAW_DATA_DIR,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    variable_records: list[dict[str, object]] = []

    for component, filename in NHANES_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            LOGGER.warning("Missing file: %s", path)
            continue
        try:
            frame = pd.read_sas(path, format="xport")
        except Exception as exc:  # malformed public download should not stop pilot
            LOGGER.exception("Could not read %s: %s", path, exc)
            continue
        frame.columns = [str(column).upper() for column in frame.columns]
        if "SEQN" not in frame.columns:
            LOGGER.warning("Skipping %s because SEQN is unavailable", filename)
            continue
        if frame["SEQN"].duplicated().any():
            LOGGER.warning(
                "%s contains duplicate SEQN values; retaining the first row", filename
            )
            frame = frame.drop_duplicates("SEQN", keep="first")
        frames[component] = frame
        for column in frame.columns:
            variable_records.append(
                {
                    "component": component,
                    "filename": filename,
                    "variable": column,
                    "dtype": str(frame[column].dtype),
                    "non_missing": int(frame[column].notna().sum()),
                    "missing_pct": float(frame[column].isna().mean()),
                }
            )

    return frames, pd.DataFrame(variable_records)


def merge_components(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("No readable NHANES XPT files were available.")
    if "demographics" in frames:
        base_name = "demographics"
    else:
        base_name = max(frames, key=lambda name: len(frames[name]))

    merged = frames[base_name].copy()
    for component, frame in frames.items():
        if component == base_name:
            continue
        overlapping = sorted((set(merged.columns) & set(frame.columns)) - {"SEQN"})
        if overlapping:
            LOGGER.info(
                "Dropping duplicate columns from %s before merge: %s",
                component,
                overlapping,
            )
            frame = frame.drop(columns=overlapping)
        before = len(merged)
        merged = merged.merge(
            frame,
            on="SEQN",
            how="left",
            validate="one_to_one",
        )
        if len(merged) != before:
            raise RuntimeError(f"Unexpected row multiplication while merging {component}")
    return merged


def build_missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "variable": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "non_missing": [int(frame[column].notna().sum()) for column in frame.columns],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_pct": [float(frame[column].isna().mean()) for column in frame.columns],
            "unique_non_missing": [
                int(frame[column].nunique(dropna=True)) for column in frame.columns
            ],
        }
    )
    return report.sort_values(["missing_pct", "variable"], ascending=[False, True])
