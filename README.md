# Diabetes Risk Prediction using BRFSS Health Indicators

The official project builds leakage-safe multiclass and binary diabetes risk models on the existing BRFSS 2015 CSV. It is a screening and machine-learning project, not a medical diagnostic system.

## Dataset and formulation

- 253,680 BRFSS rows and 21 original predictors
- Main target: `Diabetes_012` — no diabetes, prediabetes, diabetes
- Secondary target: diabetes versus no diabetes/prediabetes

## Pipeline

`CSV → validation → stratified holdout → in-pipeline feature engineering → preprocessing → imbalance handling → CV/tuning → validation calibration/thresholding → untouched test evaluation → interpretation/reporting`

## Final results

| Task | Main metrics |
| --- | --- |
| Multiclass XGBoost | Macro-F1 0.426; balanced accuracy 0.520; prediabetes recall 0.251; diabetes recall 0.651 |
| Binary XGBoost | ROC-AUC 0.829; PR-AUC 0.430; recall 0.575 at threshold 0.26 |

ExtraTrees led multiclass CV macro-F1 (0.444), while XGBoost was retained for its stronger prediabetes recall and balanced-accuracy trade-off. XGBoost led the primary binary CV metric, PR-AUC (0.418).

## Run on Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_brfss_full_pipeline.py
# Optional: rerun only the robustness stage after base outputs exist
python scripts/run_brfss_robustness_analysis.py
python scripts/validate_brfss_project.py
pytest
ruff check .
```

## Run on PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_brfss_full_pipeline.py
# Optional: rerun only the robustness stage after base outputs exist
python scripts/run_brfss_robustness_analysis.py
python scripts/validate_brfss_project.py
pytest
ruff check .
```

## Key outputs

- `reports/brfss_final/FINAL_REPORT.md`
- `reports/brfss_final/report.html`
- `reports/brfss_final/MODEL_CARD.md`
- `reports/brfss_final/CV_SUMMARY.md`
- `reports/brfss_final/INTERVIEW_DEFENSE_NOTES.md`
- `notebooks/01_brfss_final_pipeline_walkthrough.ipynb`
- `models/brfss_final/`

## Repository structure

- `src/brfss_diabetes/`: reusable production logic
- `scripts/`: full, binary and validation entry points
- `tests/`: loading, leakage, feature and metric checks
- `reports/brfss_final/`: tables, figures and project documentation
- `reports/nhanes_feasibility/`: rejected research extension

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
