# CV Summary

## Three concise bullet options

- Built BRFSS multiclass pipeline; macro-F1 0.426 and diabetes recall 0.651.
- Engineered 31 obesity, lifestyle, health, access and SES features.
- Calibrated binary XGBoost reached ROC-AUC 0.829 and PR-AUC 0.430.

## Four alternative bullet options

- Tuned XGBoost and tree/linear baselines on 253k BRFSS 2015 records.
- Built multiclass and binary BRFSS classifiers with leakage-safe CV and calibration.
- Achieved 0.251 prediabetes recall in a 1.83%-prevalence multiclass task.
- Audited feature families and subgroup errors; general health burden ranked highest.

<!-- ROBUSTNESS_ANALYSIS_START -->
## Robustness-focused bullet options

- Validated BRFSS models with 15-fold repeated CV and 1,000 bootstrap resamples.
- Built a two-stage diabetes-risk model with training-only macro-F1 0.464.
- Tested grouped splits, calibration, SMOTE-NC and paired model significance.
<!-- ROBUSTNESS_ANALYSIS_END -->

## Project description — compact

Built a reproducible BRFSS 2015 diabetes-risk pipeline for multiclass and binary prediction. Added domain-driven features, imbalance handling, cross-validated XGBoost tuning, calibration, threshold selection, feature-family ablation and subgroup error analysis.

## Project description — detailed

Developed an end-to-end diabetes risk-screening project on 253,680 BRFSS 2015 records. Compared linear and tree baselines with XGBoost, engineered clinically motivated survey features, protected an untouched test set, calibrated binary probabilities, tuned operating thresholds, and audited feature families and subgroup errors. The final multiclass test macro-F1 was 0.426; binary ROC-AUC was 0.829.

## Interview-ready summary

1. I used BRFSS 2015 because its large sample supports robust supervised-learning evaluation.
2. The main task preserves no-diabetes, prediabetes and diabetes as three classes.
3. Prediabetes was only 1.83%, so accuracy was not an adequate objective.
4. I engineered transparent obesity, health-burden, lifestyle, access and socioeconomic features.
5. All preprocessing, weighting and tuning stayed inside training/CV boundaries.
6. I compared Dummy, Logistic Regression, Random Forest, ExtraTrees and XGBoost.
7. I added binary calibration, validation-only threshold tuning and subgroup error diagnostics.
8. I position the model as risk screening, not diagnosis; NHANES remains a rejected feasibility extension.
