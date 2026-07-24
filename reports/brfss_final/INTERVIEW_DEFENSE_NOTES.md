# Interview Defense Notes

## Why BRFSS remained the official project

The NHANES pilot asked a narrower operational question: whether non-glycemic indicators could reduce confirmatory testing enough to justify replacing BRFSS. It failed the predefined capture and burden-reduction criteria. BRFSS therefore remained the stronger, more complete ML project, while NHANES became evidence of disciplined feasibility testing.

## Core defense points

- **Why multiclass is hard:** prediabetes is rare (1.83%) and weakly separated by survey variables.
- **Why XGBoost:** it models nonlinearities and interactions in mixed coded health indicators while supporting weighted training.
- **Why imbalance matters:** majority-class accuracy can hide near-zero prediabetes recall.
- **Why accuracy is insufficient:** macro-F1, balanced accuracy, class recall and PR-AUC weight minority performance more appropriately.
- **Most important feature families:** general health burden, engineered interactions, lifestyle.
- **Domain features:** cardiometabolic count, health-burden score, BMI categories and access/lifestyle combinations encode transparent hypotheses.
- **Limitations:** self-reporting, profile duplicates, one cycle, absent clinical biomarkers and no external validation.
- **Future work:** temporal validation, grouped profile split, cost-sensitive learning and prevalence-shift recalibration.

<!-- ROBUSTNESS_ANALYSIS_START -->
## Additional robustness defense

- **Uncertainty:** macro-F1 95% bootstrap interval was [0.422, 0.430].
- **Repeated CV:** model rankings were checked over 15 folds rather than one split.
- **Duplicate profiles:** a grouped-profile holdout produced macro-F1 0.427 with zero profile overlap.
- **Feature engineering:** repeated-CV ablation compared the original 21 predictors against all 52 predictors.
- **Alternative designs:** ordinal, two-stage, ensemble, calibrated, threshold-adjusted and custom-weight models were evaluated on training-only partitions.
- **SMOTE-NC:** tested inside CV folds; its result is reported rather than assumed beneficial.
- **Statistical testing:** paired bootstrap intervals and exact McNemar tests distinguish uncertainty from practical model value.
<!-- ROBUSTNESS_ANALYSIS_END -->

## Ten likely interview questions

1. **Why did you use both multiclass and binary targets?**  
   Multiclass preserves prediabetes as the main scientific challenge; binary prediction provides a cleaner screening benchmark and supports calibration and threshold decisions.

2. **Why not optimize accuracy?**  
   A model could score highly by favoring the no-diabetes majority while failing on prediabetes. Macro-F1 and class recall expose that failure.

3. **How did you prevent leakage?**  
   I split raw rows first. Feature engineering, imputation, scaling, weighting, tuning, calibration and threshold choice were fitted only on training or validation partitions.

4. **Why use class weights instead of SMOTE?**  
   The dataset contains many coded and ordinal fields. Class weighting avoids synthetic survey profiles and generally preserves probability calibration better.

5. **Why did XGBoost help?**  
   It captured nonlinear age, BMI, general-health and cardiometabolic interactions that a linear model represents less naturally. The CV tables show whether that gain was material.

6. **What was the hardest class?**  
   Prediabetes, because it is rare and its survey profile overlaps both no diabetes and diagnosed diabetes.

7. **How did you interpret the model?**  
   I combined built-in gain importance, held-out permutation importance and feature-family ablation rather than relying on one explanation method.

8. **What do duplicate rows mean?**  
   They are identical response profiles, not confirmed duplicate people because respondent IDs are absent. I retained them and disclosed possible optimism from random splitting.

9. **Can this diagnose diabetes?**  
   No. The source is self-reported survey data and the model has no confirmatory biomarkers. It is a risk-screening demonstration.

10. **What would you do before deployment?**  
    Validate on later BRFSS cycles and local populations, group identical profiles, audit subgroup calibration, choose costs with clinicians, and establish monitoring and governance.
