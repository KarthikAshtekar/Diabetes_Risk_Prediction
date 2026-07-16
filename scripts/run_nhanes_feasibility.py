from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nhanes_feasibility.config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TABLE_DIR,
    TEST_SIZE,
    ensure_directories,
)
from nhanes_feasibility.download_nhanes import download_nhanes_files  # noqa: E402
from nhanes_feasibility.evaluate import CLASS_NAMES, evaluate_models  # noqa: E402
from nhanes_feasibility.feature_sets import (  # noqa: E402
    engineer_features,
    select_tier1_features,
)
from nhanes_feasibility.labels import (  # noqa: E402
    build_diabetes_risk_stage,
    has_label_evidence,
)
from nhanes_feasibility.load_merge_nhanes import (  # noqa: E402
    build_missingness_report,
    load_available_components,
    merge_components,
)
from nhanes_feasibility.reporting import (  # noqa: E402
    generate_figures,
    write_html_report,
    write_markdown_reports,
)
from nhanes_feasibility.threshold_analysis import (  # noqa: E402
    evaluate_acceptance_criteria,
    testing_burden_for_capture_targets,
    topk_testing_simulation,
)
from nhanes_feasibility.train import fit_models, model_sanity_table  # noqa: E402


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


def main() -> int:
    ensure_directories()
    configure_logging()
    logger = logging.getLogger("nhanes_feasibility")

    logger.info("Downloading/validating official NHANES 2015-2016 files")
    manifest = download_nhanes_files()
    manifest.to_csv(TABLE_DIR / "download_manifest.csv", index=False)
    missing_files = manifest.loc[manifest["status"].eq("missing")].copy()
    missing_files.to_csv(TABLE_DIR / "missing_files.csv", index=False)

    frames, available_variables = load_available_components()
    if not frames:
        raise RuntimeError(
            "No NHANES files could be loaded. See missing_files.csv and pipeline.log."
        )
    available_variables.to_csv(TABLE_DIR / "available_variables.csv", index=False)

    logger.info("Merging %d readable NHANES components on SEQN", len(frames))
    merged = merge_components(frames)
    build_missingness_report(merged).to_csv(
        TABLE_DIR / "missingness_report.csv", index=False
    )
    merged.to_csv(
        PROCESSED_DATA_DIR / "nhanes_2015_2016_merged.csv.gz",
        index=False,
        compression="gzip",
    )

    adult_mask = pd.to_numeric(merged.get("RIDAGEYR"), errors="coerce").ge(20)
    evidence_mask = has_label_evidence(merged)
    cohort = merged.loc[adult_mask & evidence_mask].copy()
    if cohort.empty:
        raise RuntimeError("No adults with usable label evidence were available.")

    target, label_evidence = build_diabetes_risk_stage(cohort)
    cohort["diabetes_risk_stage"] = target
    label_evidence.to_csv(TABLE_DIR / "label_variable_dictionary.csv", index=False)

    label_distribution = (
        target.value_counts()
        .reindex([0, 1, 2], fill_value=0)
        .rename_axis("class")
        .reset_index(name="count")
    )
    label_distribution["class_name"] = label_distribution["class"].map(CLASS_NAMES)
    label_distribution["percentage"] = (
        label_distribution["count"] / label_distribution["count"].sum()
    )
    label_distribution.to_csv(TABLE_DIR / "label_distribution.csv", index=False)
    if (label_distribution["count"] == 0).any():
        raise RuntimeError(
            "At least one target class is empty; multiclass feasibility modeling cannot run."
        )

    engineered, engineered_dictionary = engineer_features(cohort)
    engineered_dictionary.to_csv(
        TABLE_DIR / "engineered_features_dictionary.csv", index=False
    )
    x, numeric_features, categorical_features, feature_dictionary = (
        select_tier1_features(engineered)
    )
    feature_dictionary.to_csv(TABLE_DIR / "tier1_features_used.csv", index=False)
    x["SEQN"] = cohort["SEQN"].to_numpy()
    modeling_frame = x.copy()
    modeling_frame["diabetes_risk_stage"] = target.to_numpy()
    modeling_frame.to_csv(
        PROCESSED_DATA_DIR / "nhanes_2015_2016_modeling_cohort.csv.gz",
        index=False,
        compression="gzip",
    )
    x = x.drop(columns="SEQN")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(y_train),
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum()),
                "class_2": int((y_train == 2).sum()),
            },
            {
                "split": "test",
                "n": len(y_test),
                "class_0": int((y_test == 0).sum()),
                "class_1": int((y_test == 1).sum()),
                "class_2": int((y_test == 2).sum()),
            },
        ]
    )
    split_summary.to_csv(TABLE_DIR / "train_test_split_summary.csv", index=False)

    logger.info(
        "Fitting models with %d train rows and %d Tier-1 features",
        len(y_train),
        x.shape[1],
    )
    models = fit_models(
        x_train,
        y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model_sanity_table(models, x_train, y_train).to_csv(
        TABLE_DIR / "training_sanity_metrics.csv", index=False
    )
    joblib.dump(models["XGBoost"], PROCESSED_DATA_DIR / "xgboost_tier1_model.joblib")

    model_comparison, class_recall, details = evaluate_models(models, x_test, y_test)
    model_comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    class_recall.to_csv(TABLE_DIR / "classwise_recall.csv", index=False)

    xgb_probability = np.asarray(details["XGBoost"]["high_risk_probability"])
    rule_probability = np.asarray(details["Simple Rule Score"]["high_risk_probability"])
    topk = topk_testing_simulation(y_test, xgb_probability, "XGBoost")
    rule_topk = topk_testing_simulation(y_test, rule_probability, "Simple Rule Score")
    topk.to_csv(TABLE_DIR / "topk_testing_simulation.csv", index=False)
    rule_topk.to_csv(TABLE_DIR / "rule_topk_testing_simulation.csv", index=False)
    burden = testing_burden_for_capture_targets(y_test, xgb_probability, "XGBoost")
    burden.to_csv(TABLE_DIR / "testing_burden_reduction.csv", index=False)
    criteria, passed = evaluate_acceptance_criteria(
        topk, burden, model_comparison, rule_topk
    )
    criteria.to_csv(TABLE_DIR / "acceptance_criteria.csv", index=False)

    prediction_output = pd.DataFrame(
        {
            "SEQN": cohort.loc[x_test.index, "SEQN"].to_numpy(),
            "actual_stage": y_test.to_numpy(),
            "predicted_stage": details["XGBoost"]["prediction"],
            "probability_class_0": details["XGBoost"]["probabilities"][:, 0],
            "probability_class_1": details["XGBoost"]["probabilities"][:, 1],
            "probability_class_2": details["XGBoost"]["probabilities"][:, 2],
            "probability_high_risk": xgb_probability,
        }
    )
    prediction_output.to_csv(TABLE_DIR / "xgboost_test_predictions.csv", index=False)

    files_used = manifest.loc[
        manifest["status"].isin(["available", "downloaded"]),
        ["component", "filename", "status", "source_url"],
    ].copy()
    files_used.to_csv(TABLE_DIR / "dataset_files_used.csv", index=False)
    generate_figures(label_distribution, details["XGBoost"], topk)
    verdict = write_markdown_reports(
        files_used=files_used,
        final_sample_size=len(cohort),
        label_distribution=label_distribution,
        feature_dictionary=feature_dictionary,
        model_comparison=model_comparison,
        topk=topk,
        burden=burden,
        criteria=criteria,
        passed=passed,
        label_evidence=label_evidence,
    )
    html_report = write_html_report()

    run_summary = {
        "final_sample_size": len(cohort),
        "features_used": len(feature_dictionary),
        "failed_downloads": missing_files["filename"].tolist(),
        "verdict": verdict,
        "report": str(html_report.relative_to(ROOT)),
    }
    (REPORT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print("\nNHANES feasibility pilot complete")
    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
