# Diabetes Risk Prediction using BRFSS Health Indicators

The official project builds leakage-safe multiclass and binary diabetes risk models on the existing BRFSS 2015 CSV. It is a screening and machine-learning project, not a medical diagnostic system.

Last verified by full local execution: **2026-07-24**. Install the pinned versions in `requirements.txt` before comparing regenerated metrics with the committed artifacts.

## Dataset and formulation

- 253,680 BRFSS rows and 21 original predictors
- Main target: `Diabetes_012` — no diabetes, prediabetes, diabetes
- Secondary target: diabetes versus no diabetes/prediabetes

## Pipeline

`CSV → validation → stratified holdout → in-pipeline feature engineering → preprocessing → imbalance handling → CV/tuning → validation calibration/thresholding → untouched test evaluation → interpretation/reporting`

## Final results

| Task | Main metrics |
| --- | --- |
| Multiclass XGBoost | Macro-F1 0.426; balanced accuracy 0.519; prediabetes recall 0.254; diabetes recall 0.645 |
| Binary XGBoost | ROC-AUC 0.829; PR-AUC 0.429; recall 0.590 at threshold 0.25 |

ExtraTrees led multiclass CV macro-F1 (0.444), while XGBoost was retained for its stronger prediabetes recall and balanced-accuracy trade-off. XGBoost led the primary binary CV metric, PR-AUC (0.416).

## Run on Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_brfss_full_pipeline.py
python scripts/run_nhanes_feasibility.py
python scripts/validate_brfss_project.py
pytest
ruff check .
```

The full BRFSS command includes the robustness stage. Run `python scripts/run_brfss_robustness_analysis.py` separately only to refresh robustness outputs after valid base outputs already exist.

## Run on PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_brfss_full_pipeline.py
python scripts/run_nhanes_feasibility.py
python scripts/validate_brfss_project.py
pytest
ruff check .
```

The two maintained notebooks were executed top-to-bottom with the repo-local environment. To reproduce that validation, install a local kernelspec and execute both:

```powershell
python -m ipykernel install --prefix .venv --name diabetes-project-venv --display-name "Python (diabetes project .venv)"
$env:JUPYTER_PATH = (Resolve-Path ".venv\share\jupyter").Path
jupyter-nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name=diabetes-project-venv notebooks\01_brfss_final_pipeline_walkthrough.ipynb
jupyter-nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name=diabetes-project-venv notebooks\01_nhanes_feasibility_eda.ipynb
```

## Key outputs

- `reports/project_audit/diabetes_project_validation.html`
- `reports/PROJECT_VALIDATION.md`
- `reports/brfss_final/FINAL_REPORT.md`
- `reports/brfss_final/report.html`
- `reports/brfss_final/MODEL_CARD.md`
- `reports/brfss_final/CV_SUMMARY.md`
- `reports/brfss_final/INTERVIEW_DEFENSE_NOTES.md`
- `notebooks/01_brfss_final_pipeline_walkthrough.ipynb`
- `reports/nhanes_feasibility/NHANES_PILOT_VERDICT.md`
- `notebooks/01_nhanes_feasibility_eda.ipynb`
- `models/brfss_final/`

## Repository structure

- `src/brfss_diabetes/`: reusable production logic
- `scripts/`: full, binary and validation entry points
- `tests/`: loading, leakage, feature and metric checks
- `notebooks/`: maintained walkthroughs plus an archived Colab baseline
- `reports/brfss_final/`: tables, figures and project documentation
- `reports/nhanes_feasibility/`: rejected research extension
- `reports/project_audit/`: validated portable audit report and source manifest

## Historical artifacts

`25BM6JP22_CDS_CODE.ipynb` and `notebooks/legacy_brfss_baseline.ipynb` are identical Colab-era snapshots with `/content` paths and Colab-only steps; they are preserved for history and are not the maintained local execution path. `25BM6JP22_CDS_Final_Report.pdf` is the corresponding historical submitted report. Current verified results are the generated artifacts under `reports/`.

<!-- ROBUSTNESS_ANALYSIS_START -->
## Robustness extensions

- 1,000-repetition bootstrap confidence intervals for held-out metrics
- 5×3 repeated stratified CV for model and feature-set stability
- Profile-grouped holdout with zero identical-profile overlap
- Validation-only multiclass threshold and class-weight searches
- Ordinal, two-stage, calibrated and out-of-fold ensemble comparisons
- Fold-safe moderate SMOTE-NC sensitivity analysis
- Paired bootstrap model differences and exact McNemar tests

The official headline metrics remain unchanged. Advanced strategies are reported as training-only development evidence unless explicitly identified as held-out inference.
<!-- ROBUSTNESS_ANALYSIS_END -->

## Limitations

BRFSS variables and labels are survey/self-reported. Prediabetes is severely imbalanced and weakly separable without biomarkers. Exact response profiles may repeat without respondent identifiers. Results are internal holdout estimates and should not guide clinical decisions.

## NHANES extension

The non-glycemic NHANES prioritisation pilot failed its predefined feasibility criteria. It remains documented research evidence and does not replace this BRFSS project.
