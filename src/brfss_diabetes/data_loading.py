from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, ORIGINAL_FEATURES, TARGET

LOGGER = logging.getLogger(__name__)


def find_brfss_csv(root: Path | None = None) -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    search_root = root or DATA_PATH.parent
    candidates = sorted(search_root.rglob("*BRFSS2015*.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find the BRFSS 2015 CSV in the project.")
    return candidates[0]


def load_brfss_data(path: Path | str | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else find_brfss_csv()
    if not data_path.exists():
        raise FileNotFoundError(f"BRFSS CSV not found: {data_path}")
    dtype_map = {
        TARGET: "int8",
        **{column: "int8" for column in ORIGINAL_FEATURES},
        "BMI": "int16",
    }
    frame = pd.read_csv(data_path, dtype=dtype_map, low_memory=False)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is unavailable.")
    missing = sorted(set(ORIGINAL_FEATURES) - set(frame.columns))
    if missing:
        LOGGER.warning("Expected BRFSS predictors unavailable: %s", missing)
    usable = [column for column in ORIGINAL_FEATURES if column in frame.columns]
    if len(usable) < 10:
        raise ValueError(
            f"Only {len(usable)} expected predictors are available; at least 10 are required."
        )
    return frame[[TARGET, *usable]].copy()


def make_targets(frame: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DataFrame(index=frame.index)
    targets["Diabetes_012"] = frame[TARGET].astype(int)
    targets["Diabetes_binary"] = frame[TARGET].eq(2).astype(int)
    targets["HighRisk_binary"] = frame[TARGET].isin([1, 2]).astype(int)
    return targets
