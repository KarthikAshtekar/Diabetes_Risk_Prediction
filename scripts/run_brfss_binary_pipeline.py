from __future__ import annotations

from run_brfss_full_pipeline import main


if __name__ == "__main__":
    print(
        "The reproducible full pipeline includes the binary task, calibration and "
        "threshold analysis. Running the complete pipeline."
    )
    raise SystemExit(main())
