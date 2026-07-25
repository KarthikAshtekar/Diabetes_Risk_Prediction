# End-to-End Project Validation

## Overall assessment: Ready to share with documented limitations

The maintained BRFSS pipeline, robustness suite, NHANES feasibility pilot, validation script, automated tests, lint checks, and two reader-facing notebooks were rerun successfully on 24 July 2026. Generated models, tables, figures, Markdown reports, HTML reports, and notebook outputs now reflect the pinned environment in `requirements.txt`.

The portable, source-backed technical audit is available at `reports/project_audit/diabetes_project_validation.html`; its validated manifest is preserved beside it as `artifact.json`.

The two identical Colab-era notebooks are preserved as historical evidence. They are not locally reproducible because they depend on `/content` paths, `google.colab`, Colab downloads, and notebook-only SHAP artifacts. The historical submitted PDF was therefore preserved rather than represented as a refreshed current report.

## Scope, data, and environment

- **Primary source:** `diabetes_012_health_indicators_BRFSS2015.csv`, containing 253,680 BRFSS 2015 rows, 21 predictors, and the three-class target.
- **Extension source:** 15 official NHANES 2015–2016 XPT components under `data/raw/nhanes/`; all were readable and contained `SEQN`.
- **Environment:** Python 3.13.7 with exact direct dependency versions pinned in `requirements.txt`.
- **Maintained notebooks:** `notebooks/01_brfss_final_pipeline_walkthrough.ipynb` and `notebooks/01_nhanes_feasibility_eda.ipynb`.
- **Historical notebooks:** `CODE.ipynb` and `notebooks/legacy_brfss_baseline.ipynb`; their SHA-256 hashes are identical.

## Reproduction and methodology review

1. Created a clean repo-local `.venv`.
2. Installed the declared dependencies after correcting the Windows long-path and missing-dependency issues.
3. Ran `scripts/run_brfss_full_pipeline.py`, including the chained robustness analysis.
4. Ran `scripts/run_nhanes_feasibility.py` against all 15 official source files.
5. Executed both maintained notebooks top-to-bottom with the repo-local `diabetes-project-venv` kernel.
6. Ran the project validator, NHANES file audit, 13 automated tests, Ruff, and `pip check`.
7. Recomputed saved headline metrics from prediction files and inspected representative regenerated figures.

The BRFSS design remains leakage-aware: the holdout is created before feature engineering, tuning, calibration, and threshold selection. Robustness alternatives remain training-only development evidence unless a result is explicitly identified as an official held-out estimate. The NHANES pilot continues to isolate glycemic and self-reported diabetes variables to label construction.

## Issues found and resolved

1. **Medium — clean installation failed on the deep Windows project path.** The umbrella `jupyter` package pulled a JupyterLab widget asset whose path exceeded Windows handling. It was replaced with the smaller execution stack: `nbconvert`, `nbformat`, and `ipykernel`.
2. **Medium — NHANES runtime dependencies were undeclared.** `requests` and `tqdm` were added to `requirements.txt`.
3. **Medium — dependency drift changed regenerated metrics.** Exact direct versions are now pinned so the refreshed artifacts have a defined reproduction environment.
4. **Low — Ruff reported 45 issues.** Stale import suppressions, import ordering, quoted annotations, redundant casts, and small iteration/logging issues were corrected; Ruff now passes.
5. **Documented limitation — legacy notebooks are Colab-specific.** Local execution stops in the first cell at `/content/diabetes_012_health_indicators_BRFSS2015.csv`. The maintained scripts and walkthrough notebooks are the supported local path.

## Calculation spot-checks

- **BRFSS multiclass:** macro-F1 0.4256, balanced accuracy 0.5187, prediabetes recall 0.2538, and diabetes recall 0.6454.
- **BRFSS binary:** ROC-AUC 0.8286, PR-AUC 0.4294, recall 0.5899, and precision 0.3865 at the validation-selected threshold 0.25.
- **Uncertainty:** the 1,000-repetition bootstrap 95% interval for multiclass macro-F1 is 0.4218–0.4296.
- **Profile sensitivity:** the grouped holdout contains zero identical-profile overlap and has macro-F1 0.4265.
- **Repeated validation:** all repeated-CV summaries contain 15 folds.
- **NHANES:** XGBoost high-risk PR-AUC is 0.809; testing the top 40% captures 59.1% of high-risk participants.
- **NHANES decision:** none of the three predefined acceptance criteria passes, so the verdict remains NO-GO.

Saved BRFSS macro-F1, balanced accuracy, ROC-AUC, and PR-AUC were independently recomputed from the prediction CSVs by both the validator and tests.

## Visualization and report review

All 21 regenerated PNG figures are non-empty with valid dimensions. Representative model-comparison, uncertainty, advanced-strategy, capture-curve, and confusion-matrix figures were inspected for label clipping, scale integrity, readable subtitles, and consistency with their source tables. Standard comparison bars start at zero, uncertainty intervals use a 0–1 metric scale, and the NHANES capture chart identifies the random-ranking reference.

Both generated HTML reports contain the refreshed Markdown content and local figure references. Current reader-facing metrics agree across `README.md`, the BRFSS report, model card, CV summary, interview notes, the NHANES verdict, feasibility summary, notebooks, and run-summary JSON files.

## Remaining limitations

- BRFSS predictors and labels are self-reported survey variables from one historical cycle.
- Prediabetes prevalence is only 1.83%, and its survey-feature separability remains weak.
- Internal holdout and profile-grouped results do not replace temporal or external validation.
- NHANES evaluation is an unweighted feasibility analysis, not an estimate of US population prevalence.
- The repo retains joblib model persistence; loading under NumPy 2.5 emits a deprecation warning but passed prediction and metric checks.
- The maintained project is locally reproducible; the historical Colab notebook and submitted PDF are archival, not current evidence.

## Recommended next steps

1. Validate the selected BRFSS model and the two-stage candidate on a later BRFSS cycle before making any deployment claim.
2. Consider native XGBoost model export alongside joblib for longer-lived cross-version portability.
3. Keep the NHANES pilot as negative feasibility evidence unless a new data cycle, survey-weighted design, or materially different operating criterion is approved.
