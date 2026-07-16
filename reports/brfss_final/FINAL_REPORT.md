# Diabetes Risk Prediction using BRFSS Health Indicators

## Technical summary

The official project uses BRFSS 2015 survey indicators for two related screening tasks: three-class diabetes status and binary diabetes risk. The final multiclass XGBoost achieved **macro-F1 0.426**, **balanced accuracy 0.520**, prediabetes recall **0.251**, and diabetes recall **0.651** on the untouched test set. The calibrated binary model achieved **ROC-AUC 0.829**, **PR-AUC 0.430**, and recall **0.575** at threshold **0.26**.

**Model-selection conclusion:** ExtraTrees led multiclass CV macro-F1 (0.444 versus 0.423 for XGBoost). XGBoost remained the selected final multiclass model because its weighted operating point improved prediabetes recall (0.184 versus 0.013) and balanced accuracy (0.505 versus 0.474). XGBoost led the primary binary CV metric, PR-AUC (0.418). The model is suitable as a portfolio demonstration of reproducible risk-screening ML, not as a medical diagnostic system.

<!-- ROBUSTNESS_SUMMARY_START -->
**Robustness conclusion:** The official macro-F1 95% bootstrap interval is **[0.422, 0.430]**. A zero-overlap profile-grouped holdout produced macro-F1 **0.428**, close to the official **0.426**. A two-stage model reached training-only evaluation macro-F1 **0.464**, but reduced prediabetes recall and therefore remains a candidate for future external validation rather than the new official model.
<!-- ROBUSTNESS_SUMMARY_END -->

## The rare prediabetes class defines the multiclass difficulty

The dataset contains **253,680 rows**, **21 original predictors**, and no missing values. Prediabetes represents only **1.83%** of rows. Exact duplicate response rows were retained because the public extract has no respondent identifier; identical response profiles cannot be proven to be duplicate people.

| class | count | percentage | class_name |
| --- | --- | --- | --- |
| 0 | 213703 | 0.8424 | No diabetes |
| 1 | 4631 | 0.0183 | Prediabetes |
| 2 | 35346 | 0.1393 | Diabetes |

![BRFSS class distribution](figures/class_distribution.png)

This imbalance means raw accuracy would reward predicting the majority class. Macro-F1, balanced accuracy and class-wise recall therefore governed model selection.

## Multiclass and binary tasks answer different screening questions

The official task predicts `Diabetes_012`: 0 for no diabetes, 1 for prediabetes and 2 for diabetes. The secondary task predicts diabetes versus no diabetes/prediabetes. The multiclass task preserves the clinically meaningful intermediate group; the binary task provides a more separable benchmark and supports probability calibration and operating-threshold analysis.

## Data validation found no missingness but identified profile duplication

All expected columns were available and all coded ranges passed validation. There were 23,899 exact duplicate rows. Because no respondent key is present, rows were retained and split using reproducible stratification. This can make performance optimistic when identical profiles appear across partitions and is documented as a limitation.

## Domain features expanded the model from 21 to 52 predictors

Feature engineering retained all original variables and added BMI categories, cardiometabolic burden, lifestyle risk, healthcare access, general-health burden, socioeconomic risk and interactions. Every transformation is deterministic and occurs inside the fitted pipeline.

The highest-impact families in validation ablation were: **general health burden, lifestyle, engineered interactions**.

![Feature-family ablation](figures/feature_family_ablation.png)

Positive drops show that removing the family reduced macro-F1. Small or negative drops indicate redundancy or weak incremental signal, not clinical irrelevance.

## Leakage-safe preprocessing and imbalance handling

The untouched test set was split before feature engineering, imputation, scaling, tuning, calibration or threshold selection. Linear models used median imputation, scaling and one-hot encoding for engineered ordinal categories. Tree models retained ordinal codes with median imputation. Multiclass XGBoost used training-fold balanced sample weights; synthetic oversampling was not selected because the coded survey feature space and calibration objective made class weighting the lower-risk default.

## XGBoost was tuned against strong baselines

| model | cv_folds | macro_f1 | macro_f1_std | balanced_accuracy | balanced_accuracy_std | class_0_recall | class_0_recall_std | class_1_recall | class_1_recall_std | class_2_recall | class_2_recall_std | macro_roc_auc_ovr | macro_roc_auc_ovr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees | 3 | 0.4436 | 0.0032 | 0.4742 | 0.0052 | 0.8271 | 0.0074 | 0.0128 | 0.0083 | 0.5828 | 0.0165 | 0.7347 | 0.0046 |
| Random Forest | 3 | 0.4433 | 0.0039 | 0.4701 | 0.0033 | 0.8527 | 0.0058 | 0.0018 | 0.0032 | 0.5557 | 0.0102 | 0.7493 | 0.0029 |
| XGBoost | 3 | 0.4232 | 0.0066 | 0.5048 | 0.0069 | 0.6751 | 0.0137 | 0.1843 | 0.0078 | 0.6550 | 0.0148 | 0.7509 | 0.0029 |
| Logistic Regression | 3 | 0.4119 | 0.0049 | 0.4898 | 0.0114 | 0.6430 | 0.0139 | 0.2609 | 0.0297 | 0.5656 | 0.0295 | 0.7533 | 0.0086 |
| DummyClassifier | 3 | 0.3048 | 0.0000 | 0.3333 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |

![Multiclass model comparison](figures/model_comparison_multiclass.png)

| model | cv_folds | pr_auc | pr_auc_std | roc_auc | roc_auc_std | f1 | f1_std | balanced_accuracy | balanced_accuracy_std | recall | recall_std | precision | precision_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 3 | 0.4184 | 0.0102 | 0.8201 | 0.0051 | 0.4362 | 0.0051 | 0.7401 | 0.0052 | 0.7584 | 0.0095 | 0.3061 | 0.0036 |
| Logistic Regression | 3 | 0.4115 | 0.0126 | 0.8227 | 0.0054 | 0.4420 | 0.0018 | 0.7480 | 0.0020 | 0.7780 | 0.0076 | 0.3087 | 0.0021 |
| Random Forest | 3 | 0.3961 | 0.0051 | 0.8084 | 0.0049 | 0.4386 | 0.0052 | 0.6882 | 0.0015 | 0.5065 | 0.0032 | 0.3869 | 0.0101 |
| ExtraTrees | 3 | 0.3884 | 0.0048 | 0.8044 | 0.0028 | 0.4385 | 0.0070 | 0.6990 | 0.0047 | 0.5574 | 0.0084 | 0.3615 | 0.0072 |
| DummyClassifier | 3 | 0.1393 | 0.0001 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

![Binary model comparison](figures/model_comparison_binary.png)

Logistic regression regularization, Random Forest structure and XGBoost parameters were tuned using training-only cross-validation. XGBoost tuning covered estimators, depth, learning rate, subsampling, column sampling, child weight, gamma and L1/L2 regularization. Selected multiclass XGBoost parameters: `{'colsample_bytree': 0.7713532627589614, 'gamma': 0.4563633644393066, 'learning_rate': 0.061738558891618764, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 228, 'reg_alpha': 0.012563152773938664, 'reg_lambda': 1.5151324184839015, 'subsample': 0.7330663856998123}`. Selected binary XGBoost parameters: `{'colsample_bytree': 0.9926515452756086, 'gamma': 0.34915701064545634, 'learning_rate': 0.029223394319260826, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 287, 'reg_alpha': 0.011400863701127324, 'reg_lambda': 2.584093506411774, 'subsample': 0.7330061155615993, 'scale_pos_weight': 6.176998974431517}`.

## Final multiclass errors concentrate around prediabetes

| metric | value |
| --- | --- |
| accuracy | 0.6492 |
| macro_f1 | 0.4262 |
| weighted_f1 | 0.7187 |
| balanced_accuracy | 0.5198 |
| macro_roc_auc_ovr | 0.7615 |
| macro_pr_auc_ovr | 0.4650 |

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| No diabetes | 0.9526 | 0.6576 | 0.7781 | 42741.0000 |
| Prediabetes | 0.0302 | 0.2505 | 0.0539 | 926.0000 |
| Diabetes | 0.3398 | 0.6512 | 0.4465 | 7069.0000 |
| accuracy | 0.6492 | 0.6492 | 0.6492 | 0.6492 |
| macro avg | 0.4409 | 0.5198 | 0.4262 | 50736.0000 |
| weighted avg | 0.8504 | 0.6492 | 0.7187 | 50736.0000 |

![Multiclass confusion matrix](figures/multiclass_confusion_matrix.png)

Prediabetes has weaker separability than diabetes because it is rare and BRFSS indicators are survey/self-reported rather than glycemic biomarkers. Confusions with both no diabetes and diabetes are therefore reported directly rather than hidden by aggregate accuracy.

## The binary task supports calibrated risk screening

| metric | value |
| --- | --- |
| threshold | 0.2600 |
| roc_auc | 0.8287 |
| pr_auc | 0.4297 |
| recall | 0.5746 |
| precision | 0.3965 |
| f1 | 0.4692 |
| balanced_accuracy | 0.7165 |
| brier_score | 0.0970 |

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| No diabetes | 0.9257 | 0.8584 | 0.8908 | 43667.0000 |
| Diabetes | 0.3965 | 0.5746 | 0.4692 | 7069.0000 |
| accuracy | 0.8189 | 0.8189 | 0.8189 | 0.8189 |
| macro avg | 0.6611 | 0.7165 | 0.6800 | 50736.0000 |
| weighted avg | 0.8520 | 0.8189 | 0.8321 | 50736.0000 |

![Binary precision-recall curve](figures/binary_pr_curve.png)

![Binary calibration curve](figures/calibration_curve_binary.png)

The sigmoid calibrator and operating threshold were learned from validation probabilities only. The test set was used once for the final metric readout.

## Threshold choice changes the screening operating point

| threshold | precision | recall | f1 | balanced_accuracy | predicted_positive_rate | meets_minimum_precision | selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1100 | 0.2782 | 0.8522 | 0.4194 | 0.7471 | 0.4269 | False | \|high_recall |
| 0.1500 | 0.3121 | 0.7864 | 0.4469 | 0.7529 | 0.3511 | False | \|balanced_precision_recall |
| 0.2000 | 0.3530 | 0.6912 | 0.4674 | 0.7431 | 0.2728 | True | \|minimum_precision |
| 0.2600 | 0.4015 | 0.5750 | 0.4729 | 0.7181 | 0.1995 | True | max_f1 |

![Threshold trade-off](figures/threshold_precision_recall_tradeoff.png)

The saved binary report uses the validation-selected max-F1 threshold. Other rows support high-recall or minimum-precision operating points without pretending one threshold is universally correct.

The multiclass high-risk sensitivity analysis defines high risk as class 1 or 2:

| threshold | high_risk_precision | high_risk_recall | f1 | balanced_accuracy | predicted_positive_rate | meets_minimum_precision | selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2100 | 0.2014 | 0.9804 | 0.3342 | 0.6267 | 0.7669 | True | \|minimum_precision |
| 0.5900 | 0.2973 | 0.8537 | 0.4411 | 0.7382 | 0.4524 | True | \|high_recall |
| 0.6600 | 0.3287 | 0.7909 | 0.4644 | 0.7444 | 0.3791 | True | \|balanced_precision_recall |
| 0.7800 | 0.4164 | 0.5981 | 0.4910 | 0.7206 | 0.2263 | True | max_f1 |
| 0.8676 | 0.5110 | 0.3244 |  |  | 0.1000 | True | top_10_percent |
| 0.7997 | 0.4329 | 0.5495 |  |  | 0.2000 | True | top_20_percent |
| 0.7253 | 0.3672 | 0.6992 |  |  | 0.3000 | True | top_30_percent |
| 0.6402 | 0.3190 | 0.8099 |  |  | 0.4000 | True | top_40_percent |
| 0.5419 | 0.2800 | 0.8885 |  |  | 0.5000 | True | top_50_percent |

## Multiple interpretation methods agree on the main signal families

![XGBoost feature importance](figures/top_feature_importance_xgboost.png)

![Permutation importance](figures/permutation_importance_top20.png)

Built-in importance, held-out permutation importance and family ablation were all used. These methods describe predictive contribution; none establishes causal or clinical effect.

Top built-in features:

| feature | importance | feature_family |
| --- | --- | --- |
| age_cardiometabolic_interaction | 0.0794 | engineered interactions |
| cardiometabolic_count | 0.0780 | cardiometabolic history |
| GenHlth | 0.0510 | general health burden |
| poor_general_health_flag | 0.0374 | general health burden |
| cholcheck_with_highchol_flag | 0.0261 | healthcare access |
| health_burden_score | 0.0260 | general health burden |
| bmi_highbp_interaction | 0.0238 | engineered interactions |
| HighBP | 0.0232 | cardiometabolic history |
| bmi_age_interaction | 0.0231 | engineered interactions |
| HvyAlcoholConsump | 0.0197 | lifestyle |
| CholCheck | 0.0187 | healthcare access |
| physical_inactivity_flag | 0.0176 | lifestyle |
| age_bmi_interaction | 0.0173 | engineered interactions |
| bmi_highchol_interaction | 0.0169 | engineered interactions |
| age_band | 0.0166 | demographics |

Top permutation features:

| feature | importance_mean | importance_std | feature_family |
| --- | --- | --- | --- |
| bmi_highbp_interaction | 0.0051 | 0.0009 | engineered interactions |
| Sex | 0.0033 | 0.0009 | demographics |
| HvyAlcoholConsump | 0.0032 | 0.0013 | lifestyle |
| income_education_interaction | 0.0022 | 0.0014 | engineered interactions |
| Smoker | 0.0014 | 0.0007 | lifestyle |
| HighBP | 0.0014 | 0.0004 | cardiometabolic history |
| bmi_age_interaction | 0.0014 | 0.0015 | engineered interactions |
| GenHlth | 0.0012 | 0.0003 | general health burden |
| healthcare_access_barrier | 0.0012 | 0.0003 | healthcare access |
| HeartDiseaseorAttack | 0.0012 | 0.0004 | cardiometabolic history |
| PhysActivity | 0.0011 | 0.0004 | lifestyle |
| bmi_highchol_interaction | 0.0011 | 0.0008 | engineered interactions |
| bp_cholesterol_combo | 0.0009 | 0.0001 | cardiometabolic history |
| cardiometabolic_count | 0.0007 | 0.0008 | cardiometabolic history |
| CholCheck | 0.0006 | 0.0004 | healthcare access |

## Error and subgroup diagnostics expose where the model is weakest

| error_type | count | percentage_of_test_set | mean_BMI | mean_Age | mean_GenHlth | mean_HighBP | mean_HighChol | mean_Income |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diabetes_false_negative | 3007 | 0.0593 | 28.4905 | 9.0166 | 2.7034 | 0.5304 | 0.5028 | 5.9056 |
| diabetes_false_positive | 6182 | 0.1218 | 32.9502 | 9.7422 | 3.5542 | 0.8845 | 0.7454 | 4.9400 |
| prediabetes_to_no_diabetes | 250 | 0.0049 | 27.2760 | 7.6400 | 2.2800 | 0.2680 | 0.3560 | 6.0680 |
| prediabetes_to_diabetes | 444 | 0.0088 | 33.0991 | 9.6779 | 3.4392 | 0.8446 | 0.7185 | 4.9752 |

| dimension | group | n | macro_f1 | balanced_accuracy | class_0_recall | class_1_recall | class_2_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age_group | 35-54 approx. | 18711 | 0.4375 | 0.5309 | 0.7563 | 0.2674 | 0.5690 |
| age_group | 55-69 approx. | 17766 | 0.3965 | 0.4855 | 0.5037 | 0.2335 | 0.7194 |
| age_group | 70+ approx. | 6608 | 0.3232 | 0.4347 | 0.3404 | 0.3098 | 0.6539 |
| age_group | 18-34 approx. | 7651 | 0.4355 | 0.4494 | 0.9470 | 0.1000 | 0.3013 |
| sex | Male | 22527 | 0.4184 | 0.5152 | 0.6289 | 0.2346 | 0.6822 |
| sex | Female | 28209 | 0.4323 | 0.5218 | 0.6799 | 0.2630 | 0.6224 |
| income_group | Higher | 26762 | 0.4299 | 0.5083 | 0.7774 | 0.2114 | 0.5361 |
| income_group | Middle | 16547 | 0.4001 | 0.5045 | 0.5441 | 0.2806 | 0.6889 |
| income_group | Lower | 7427 | 0.3838 | 0.4766 | 0.4107 | 0.2639 | 0.7554 |
| education_group | College graduate | 21505 | 0.4302 | 0.5131 | 0.7670 | 0.2323 | 0.5399 |
| education_group | Some college | 26509 | 0.4128 | 0.5095 | 0.5862 | 0.2519 | 0.6904 |
| education_group | High school or lower | 2722 | 0.3740 | 0.4666 | 0.3694 | 0.2952 | 0.7351 |

Subgroup diagnostics are descriptive checks by age, sex, income and education. They are not a fairness certification because BRFSS coding, sample composition and outcome quality can differ across groups.

<!-- ROBUSTNESS_ANALYSIS_START -->
## Robustness analysis quantifies uncertainty and tests alternative ML designs

The official holdout estimates are now accompanied by stratified bootstrap intervals. Multiclass macro-F1 was **0.426** with a 95% interval of **[0.422, 0.430]**. Binary ROC-AUC was **0.829** with a 95% interval of **[0.824, 0.834]**. These intervals describe sampling uncertainty in the fixed test set; they do not cover temporal or population shift.

![Bootstrap confidence intervals](figures/bootstrap_confidence_intervals.png)

| metric | estimate | ci_lower_95 | ci_upper_95 | bootstrap_repetitions |
| --- | --- | --- | --- | --- |
| multiclass_macro_f1 | 0.4262 | 0.4224 | 0.4297 | 1000 |
| multiclass_balanced_accuracy | 0.5198 | 0.5107 | 0.5295 | 1000 |
| prediabetes_recall | 0.2505 | 0.2235 | 0.2786 | 1000 |
| diabetes_recall | 0.6512 | 0.6397 | 0.6609 | 1000 |
| multiclass_macro_roc_auc_ovr | 0.7615 | 0.7553 | 0.7682 | 1000 |
| binary_roc_auc | 0.8287 | 0.8238 | 0.8335 | 1000 |
| binary_pr_auc | 0.4297 | 0.4200 | 0.4420 | 1000 |
| binary_recall | 0.5746 | 0.5630 | 0.5865 | 1000 |
| binary_precision | 0.3965 | 0.3898 | 0.4038 | 1000 |
| binary_f1 | 0.4692 | 0.4615 | 0.4777 | 1000 |

### Repeated CV confirms the model-ranking trade-off

Five-fold cross-validation repeated across three seeds confirms that no single model dominates every objective. ExtraTrees remains competitive on macro-F1, while weighted XGBoost retains materially stronger prediabetes recall. The comparison is based only on a stratified training sample and does not reuse the official test set for selection.

![Repeated CV comparison](figures/repeated_cv_model_comparison.png)

| model | folds | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | class_1_recall_mean | class_2_recall_mean | macro_roc_auc_ovr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees | 15 | 0.4459 | 0.0054 | 0.4766 | 0.0280 | 0.5792 | 0.7370 |
| XGBoost — engineered features | 15 | 0.4321 | 0.0049 | 0.4924 | 0.0742 | 0.6704 | 0.7459 |
| XGBoost — original features | 15 | 0.4315 | 0.0053 | 0.5010 | 0.1174 | 0.6721 | 0.7435 |
| Logistic Regression | 15 | 0.4153 | 0.0054 | 0.5000 | 0.2829 | 0.5737 | 0.7589 |

### Engineered features provide limited incremental average performance

Using identical repeated-CV folds, the original-plus-engineered specification produced mean macro-F1 **0.432**, versus **0.432** for original features alone. The result should be interpreted as an ablation finding: engineered variables improve transparency and specific interactions, but they do not guarantee a large aggregate score increase.

| configuration | folds | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | class_1_recall_mean | class_2_recall_mean | macro_roc_auc_ovr_mean | paired_macro_f1_difference_mean | paired_macro_f1_difference_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original + engineered features | 15 | 0.4321 | 0.0049 | 0.4924 | 0.0742 | 0.6704 | 0.7459 | 0.0006 | 0.0024 |
| Original features only | 15 | 0.4315 | 0.0053 | 0.5010 | 0.1174 | 0.6721 | 0.7435 |  |  |

### Grouping identical profiles changes the evaluation population

The profile-grouped split had zero predictor-profile overlap and produced macro-F1 **0.428**, compared with **0.426** on the official random holdout. Because the grouped and random test populations differ, this is a robustness sensitivity rather than a paired estimate of leakage bias.

![Grouped split comparison](figures/grouped_profile_split_comparison.png)

| split_design | metric | value | test_rows | profile_overlap_count |
| --- | --- | --- | --- | --- |
| Official random holdout | accuracy | 0.6492 | 50736 | 5104 |
| Official random holdout | macro_f1 | 0.4262 | 50736 | 5104 |
| Official random holdout | balanced_accuracy | 0.5198 | 50736 | 5104 |
| Official random holdout | class_1_recall | 0.2505 | 50736 | 5104 |
| Official random holdout | class_2_recall | 0.6512 | 50736 | 5104 |
| Official random holdout | macro_roc_auc_ovr | 0.7615 | 50736 | 5104 |
| Profile-grouped holdout | accuracy | 0.6477 | 50735 | 0 |
| Profile-grouped holdout | macro_f1 | 0.4278 | 50735 | 0 |
| Profile-grouped holdout | balanced_accuracy | 0.5300 | 50735 | 0 |
| Profile-grouped holdout | class_1_recall | 0.2808 | 50735 | 0 |
| Profile-grouped holdout | class_2_recall | 0.6545 | 50735 | 0 |
| Profile-grouped holdout | macro_roc_auc_ovr | 0.7700 | 50735 | 0 |

### Advanced strategies did not automatically solve prediabetes separation

The strategy evaluation split was created entirely inside the original training partition. The best validation-only strategy was **Two-stage high-risk XGBoost**, with macro-F1 **0.464**, versus **0.426** for the weighted XGBoost baseline. Probability multipliers selected class-1 factor **0.50** and class-2 factor **0.60**. These are model-development results, not new independent test claims.

![Advanced strategy comparison](figures/advanced_multiclass_strategy_comparison.png)

| strategy | accuracy | macro_f1 | balanced_accuracy | class_1_recall | class_1_precision | class_2_recall | macro_roc_auc_ovr | multiclass_log_loss | multiclass_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Two-stage high-risk XGBoost | 0.7772 | 0.4641 | 0.4968 | 0.1025 | 0.0437 | 0.5602 | 0.7822 | 0.6202 | 0.1098 |
| Class-specific probability multipliers | 0.7704 | 0.4576 | 0.4974 | 0.0522 | 0.0355 | 0.6312 | 0.7697 | 0.6178 | 0.0982 |
| Custom class weights | 0.7398 | 0.4459 | 0.5008 | 0.0576 | 0.0282 | 0.6803 | 0.7672 | 0.6760 | 0.0897 |
| OOF Logistic + ExtraTrees + XGBoost ensemble | 0.7264 | 0.4425 | 0.5045 | 0.0647 | 0.0276 | 0.7046 | 0.7657 | 0.7299 | 0.1146 |
| Weighted XGBoost baseline | 0.6531 | 0.4259 | 0.5178 | 0.2284 | 0.0303 | 0.6647 | 0.7632 | 0.8011 | 0.0716 |
| Ordinal cumulative XGBoost | 0.7102 | 0.4171 | 0.5030 | 0.0000 | 0.0000 | 0.7977 | 0.7167 | 0.6501 | 0.0590 |
| Multiclass probability calibration | 0.8495 | 0.3938 | 0.3841 | 0.0000 | 0.0000 | 0.1724 | 0.7801 | 0.3940 | 0.0047 |

The comparison includes class-specific probability adjustment, custom class weighting, multinomial probability calibration, ordinal cumulative models, a two-stage high-risk model, and an out-of-fold Logistic/ExtraTrees/XGBoost ensemble. The two-stage model improved macro-F1 mainly by recovering majority-class precision; its prediabetes recall remained below the weighted baseline. It is therefore a candidate operating design, not an unconditional replacement.

### Calibration, SMOTE-NC and paired statistical comparisons

Multiclass calibration was evaluated using log loss, multiclass Brier score and expected calibration error. Calibration sharply improved probability quality but collapsed prediabetes recall to zero at the default argmax rule, demonstrating that calibrated probabilities still require a separate decision policy. Moderate SMOTE-NC was run inside each CV training fold and compared with balanced sample weights; it reduced macro-F1, balanced accuracy and diabetes recall, so class weighting remains the preferred imbalance treatment.

| probability | macro_f1 | balanced_accuracy | class_1_recall | class_2_recall | multiclass_log_loss | multiclass_brier_score | multiclass_ece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw weighted XGBoost | 0.4259 | 0.5178 | 0.2284 | 0.6647 | 0.8011 | 0.4524 | 0.0716 |
| Multinomial log-probability calibration | 0.3938 | 0.3841 | 0.0000 | 0.1724 | 0.3940 | 0.2208 | 0.0047 |

| strategy | folds | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | class_1_recall_mean | class_1_precision_mean | class_2_recall_mean | multiclass_log_loss_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced sample weights | 3 | 0.4387 | 0.0014 | 0.4852 | 0.0329 | 0.0190 | 0.6469 | 0.5890 |
| Moderate SMOTE-NC | 3 | 0.4337 | 0.0083 | 0.4236 | 0.0028 | 0.0476 | 0.3247 | 0.4214 |

Paired bootstrap intervals compare models on the same observations. An interval crossing zero indicates that the available test set does not establish a clear difference for that metric. Exact McNemar tests compare paired correctness, but statistical significance should not be confused with practical value in a dataset this large.

| metric | model_a | model_b | difference_a_minus_b | ci_lower_95 | ci_upper_95 | probability_difference_positive | bootstrap_repetitions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multiclass_macro_f1 | Weighted XGBoost | ExtraTrees | -0.0191 | -0.0234 | -0.0150 | 0.0000 | 1000 |
| multiclass_balanced_accuracy | Weighted XGBoost | ExtraTrees | 0.0230 | 0.0133 | 0.0333 | 1.0000 | 1000 |
| prediabetes_recall | Weighted XGBoost | ExtraTrees | 0.1803 | 0.1512 | 0.2095 | 1.0000 | 1000 |
| binary_roc_auc | Calibrated XGBoost | Logistic Regression | 0.0048 | 0.0038 | 0.0060 | 1.0000 | 1000 |
| binary_pr_auc | Calibrated XGBoost | Logistic Regression | 0.0177 | 0.0130 | 0.0224 | 1.0000 | 1000 |

| test | model_a | model_b | a_correct_b_wrong | a_wrong_b_correct | discordant_pairs | p_value | p_value_display |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_mcnemar | Weighted XGBoost | ExtraTrees | 1167 | 5698 | 6865 | 0.0000 | <1e-300 |
| exact_mcnemar | Calibrated XGBoost | Threshold-tuned Logistic Regression | 1685 | 648 | 2333 | 0.0000 | 1.252e-105 |

Detailed selection surfaces are saved in `multiclass_probability_threshold_tuning.csv` and `class_weight_sensitivity.csv`:

| class_0_multiplier | class_1_multiplier | class_2_multiplier | accuracy | macro_f1 | balanced_accuracy | class_0_recall | class_1_recall | class_2_recall | class_0_precision | class_1_precision | class_2_precision | selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.5000 | 0.6000 | 0.7715 | 0.4577 | 0.4955 | 0.8127 | 0.0576 | 0.6162 | 0.9248 | 0.0383 | 0.3696 | max_macro_f1 |
| 1.0000 | 3.0000 | 0.6000 | 0.3760 | 0.2510 | 0.4571 | 0.4160 | 0.8885 | 0.0669 | 0.9754 | 0.0260 | 0.5410 | \|max_prediabetes_recall |

| class_1_multiplier | class_2_multiplier | accuracy | macro_f1 | balanced_accuracy | class_0_recall | class_1_recall | class_2_recall | class_0_precision | class_1_precision | class_2_precision | macro_roc_auc_ovr | multiclass_log_loss | multiclass_brier_score | multiclass_ece | selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6000 | 0.7500 | 0.7476 | 0.4478 | 0.4966 | 0.7789 | 0.0629 | 0.6478 | 0.9295 | 0.0347 | 0.3460 | 0.7628 | 0.6155 | 0.3481 | 0.0521 | max_macro_f1 |
| 0.8000 | 0.7500 | 0.7325 | 0.4469 | 0.5007 | 0.7650 | 0.1205 | 0.6165 | 0.9330 | 0.0346 | 0.3497 | 0.7607 | 0.6455 | 0.3647 | 0.0480 |  |
| 1.0000 | 0.7500 | 0.7162 | 0.4432 | 0.5017 | 0.7501 | 0.1727 | 0.5823 | 0.9343 | 0.0328 | 0.3565 | 0.7589 | 0.6716 | 0.3796 | 0.0473 |  |
| 1.2500 | 0.7500 | 0.6955 | 0.4368 | 0.5030 | 0.7317 | 0.2410 | 0.5363 | 0.9373 | 0.0326 | 0.3605 | 0.7577 | 0.7015 | 0.3976 | 0.0474 |  |
| 1.0000 | 1.0000 | 0.6975 | 0.4358 | 0.5111 | 0.7114 | 0.1349 | 0.6869 | 0.9433 | 0.0360 | 0.3232 | 0.7585 | 0.7092 | 0.4028 | 0.0338 |  |
| 0.8000 | 1.0000 | 0.7096 | 0.4355 | 0.5044 | 0.7233 | 0.0809 | 0.7089 | 0.9413 | 0.0338 | 0.3197 | 0.7595 | 0.6847 | 0.3902 | 0.0295 |  |
| 0.6000 | 1.0000 | 0.7192 | 0.4318 | 0.4987 | 0.7324 | 0.0342 | 0.7296 | 0.9398 | 0.0287 | 0.3159 | 0.7603 | 0.6554 | 0.3762 | 0.0267 |  |
| 1.2500 | 1.0000 | 0.6813 | 0.4315 | 0.5101 | 0.6983 | 0.1888 | 0.6431 | 0.9448 | 0.0328 | 0.3293 | 0.7567 | 0.7372 | 0.4182 | 0.0410 |  |
<!-- ROBUSTNESS_ANALYSIS_END -->

## Limitations and robustness checks

- BRFSS is survey and self-reported data; labels and predictors may contain recall or reporting error.
- The model is a risk-screening exercise, not a diagnostic medical system.
- Prediabetes is difficult because of severe class imbalance and weak survey-feature separability.
- Exact profile duplicates were retained because respondent identifiers are absent. A profile-grouped sensitivity split produced similar aggregate performance with zero profile overlap, but it is still only internal validation.
- The single 2015 extract provides internal holdout validation, not temporal or external validation.
- Feature importance is associative and should not be interpreted as clinical causality.
- Calibration and threshold choices may shift under a different population prevalence.

## Recommended next steps

Use the BRFSS project as the official CV project. Position the contribution around leakage-safe pipeline design, domain features, imbalance-aware evaluation, repeated validation, bootstrap uncertainty, calibration, threshold selection and honest prediabetes limitations. Treat the two-stage model as the leading future candidate until it is confirmed on a fresh external or temporal test set. Keep the NHANES work as a rejected feasibility extension demonstrating evidence-based project selection.

## Further questions

- How does the model generalize to a later BRFSS cycle?
- Does the two-stage macro-F1 gain persist on a fresh independent test set?
- Can a decision policy preserve calibrated probabilities while recovering prediabetes recall?
