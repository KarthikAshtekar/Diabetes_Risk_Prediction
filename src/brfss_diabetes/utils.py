from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, REPORT_DIR


def configure_logging() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(REPORT_DIR / "pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )


def stratified_sample_indices(
    target: pd.Series,
    n: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    if len(target) <= n:
        return target.index.to_numpy()
    generator = np.random.default_rng(random_state)
    sampled: list[np.ndarray] = []
    for _, indices in target.groupby(target).groups.items():
        group_indices = np.asarray(list(indices))
        group_n = max(1, round(n * len(group_indices) / len(target)))
        sampled.append(
            generator.choice(group_indices, size=min(group_n, len(group_indices)), replace=False)
        )
    result = np.concatenate(sampled)
    if len(result) > n:
        result = generator.choice(result, size=n, replace=False)
    return np.sort(result)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
