# Model Card — BRFSS Diabetes Risk Prediction

## Intended use

Educational risk-screening and portfolio demonstration using BRFSS 2015 health indicators.

## Non-intended use

Clinical diagnosis, treatment decisions, automated denial of care, or deployment without external validation and governance.

## Dataset and targets

- Source: local BRFSS 2015 health-indicator extract, 253,680 rows.
- Multiclass target: no diabetes, prediabetes, diabetes.
- Binary target: diabetes versus no diabetes/prediabetes.

## Features

21 original survey indicators plus deterministic domain features for obesity, cardiometabolic burden, lifestyle, healthcare access, general health and socioeconomic status. No NHANES or glycemic biomarker features enter this pipeline.

## Final model

XGBoost pipelines with training-only preprocessing and tuning. The multiclass model uses balanced sample weights and was selected for its minority-class trade-off; ExtraTrees had the higher multiclass CV macro-F1. The binary model uses validation-only sigmoid calibration and threshold selection.

## Held-out metrics

- Multiclass macro-F1: 0.426
- Multiclass balanced accuracy: 0.520
- Prediabetes recall: 0.251
- Diabetes recall: 0.651
- Binary ROC-AUC: 0.829
- Binary PR-AUC: 0.430
- Binary recall at threshold 0.26: 0.575

## Limitations and ethical considerations

Survey self-reporting, class imbalance, absent respondent identifiers, duplicate response profiles, one historical cycle, subgroup performance differences and prevalence-dependent calibration limit use. Outputs should support human review and preventive screening discussion, never replace clinical testing.
