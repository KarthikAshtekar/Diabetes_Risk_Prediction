from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nhanes_feasibility.config import NHANES_FILES, RAW_DATA_DIR  # noqa: E402


def main() -> int:
    rows = []
    for component, filename in NHANES_FILES.items():
        path = RAW_DATA_DIR / filename
        status = "missing"
        rows_count = None
        columns_count = None
        error = ""
        if path.exists():
            try:
                frame = pd.read_sas(path, format="xport")
                status = "readable"
                rows_count = len(frame)
                columns_count = len(frame.columns)
                if "SEQN" not in frame.columns:
                    status = "missing SEQN"
            except Exception as exc:
                status = "unreadable"
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "component": component,
                "filename": filename,
                "status": status,
                "rows": rows_count,
                "columns": columns_count,
                "error": error,
            }
        )
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return 1 if result["status"].isin(["unreadable", "missing SEQN"]).any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
