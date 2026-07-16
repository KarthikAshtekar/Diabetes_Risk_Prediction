from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, REPORT_DIR
from .reporting import COLORS, _header, _save, _theme, markdown_table

ROBUSTNESS_START = "<!-- ROBUSTNESS_ANALYSIS_START -->"
ROBUSTNESS_END = "<!-- ROBUSTNESS_ANALYSIS_END -->"
ROBUSTNESS_SUMMARY_START = "<!-- ROBUSTNESS_SUMMARY_START -->"
ROBUSTNESS_SUMMARY_END = "<!-- ROBUSTNESS_SUMMARY_END -->"


def _replace_marked_section(
    text: str,
    content: str,
    start_marker: str,
    end_marker: str,
    before_heading: str | None = None,
) -> str:
    block = f"{start_marker}\n{content.strip()}\n{end_marker}"
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(block, text)
    if before_heading and before_heading in text:
        return text.replace(before_heading, f"{block}\n\n{before_heading}", 1)
    return f"{text.rstrip()}\n\n{block}\n"


def generate_robustness_figures(
    confidence_intervals: pd.DataFrame,
    repeated_cv_summary: pd.DataFrame,
    grouped_split_results: pd.DataFrame,
    advanced_strategy_results: pd.DataFrame,
) -> pd.DataFrame:
    _theme()
    chart_map = []

    selected_metrics = confidence_intervals.loc[
        confidence_intervals["metric"].isin(
            [
                "multiclass_macro_f1",
                "multiclass_balanced_accuracy",
                "prediabetes_recall",
                "diabetes_recall",
                "binary_roc_auc",
                "binary_pr_auc",
                "binary_recall",
            ]
        )
    ].copy()
    labels = {
        "multiclass_macro_f1": "Multiclass macro-F1",
        "multiclass_balanced_accuracy": "Multiclass balanced accuracy",
        "prediabetes_recall": "Prediabetes recall",
        "diabetes_recall": "Diabetes recall",
        "binary_roc_auc": "Binary ROC-AUC",
        "binary_pr_auc": "Binary PR-AUC",
        "binary_recall": "Binary recall",
    }
    selected_metrics["label"] = selected_metrics["metric"].map(labels)
    selected_metrics = selected_metrics.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(9, 6))
    lower = (
        selected_metrics["estimate"] - selected_metrics["ci_lower_95"]
    ).to_numpy()
    upper = (
        selected_metrics["ci_upper_95"] - selected_metrics["estimate"]
    ).to_numpy()
    ax.errorbar(
        selected_metrics["estimate"],
        selected_metrics["label"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=COLORS["blue_dark"],
        markerfacecolor=COLORS["blue"],
        markeredgecolor=COLORS["blue_dark"],
        linewidth=1,
        capsize=3,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Metric estimate with 95% bootstrap interval")
    ax.set_ylabel("")
    _header(
        fig,
        ax,
        "Held-out metric uncertainty",
        "Stratified nonparametric bootstrap on the 50,736-row official test set; 1,000 repetitions.",
    )
    _save(fig, "bootstrap_confidence_intervals.png")
    chart_map.append(
        {
            "figure": "bootstrap_confidence_intervals.png",
            "family": "faceted dot and interval",
            "claim": "Quantifies uncertainty around the official held-out metrics.",
        }
    )

    plot = repeated_cv_summary.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(
        plot["macro_f1_mean"],
        plot["model"],
        xerr=plot["macro_f1_std"],
        fmt="o",
        color=COLORS["gold_dark"],
        markerfacecolor=COLORS["gold"],
        markeredgecolor=COLORS["gold_dark"],
        linewidth=1,
        capsize=3,
    )
    ax.set_xlabel("Macro-F1 mean ± one fold standard deviation")
    ax.set_ylabel("")
    _header(
        fig,
        ax,
        "Repeated cross-validation comparison",
        "Five folds repeated across three seeds on a stratified 30,000-row training-only sample.",
    )
    _save(fig, "repeated_cv_model_comparison.png")
    chart_map.append(
        {
            "figure": "repeated_cv_model_comparison.png",
            "family": "dot and interval",
            "claim": "Shows stability across 15 repeated validation folds.",
        }
    )

    metric_order = [
        "macro_f1",
        "balanced_accuracy",
        "class_1_recall",
        "class_2_recall",
    ]
    grouped_plot = grouped_split_results.loc[
        grouped_split_results["metric"].isin(metric_order)
    ].copy()
    grouped_plot["metric"] = pd.Categorical(
        grouped_plot["metric"], categories=metric_order, ordered=True
    )
    pivot = grouped_plot.pivot(
        index="metric", columns="split_design", values="value"
    ).reindex(metric_order)
    positions = np.arange(len(pivot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, column, fill, edge in (
        (-width / 2, "Official random holdout", COLORS["blue"], COLORS["blue_dark"]),
        (
            width / 2,
            "Profile-grouped holdout",
            COLORS["orange"],
            COLORS["orange_dark"],
        ),
    ):
        bars = ax.bar(
            positions + offset,
            pivot[column],
            width=width,
            label=column,
            color=fill,
            edgecolor=edge,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in pivot[column]],
            padding=2,
            fontsize=8,
        )
    ax.set_xticks(
        positions,
        ["Macro-F1", "Balanced accuracy", "Prediabetes recall", "Diabetes recall"],
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0, max(0.8, float(pivot.max().max()) * 1.18))
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False)
    _header(
        fig,
        ax,
        "Random versus profile-grouped holdout",
        "Grouped splitting keeps every identical predictor profile in one partition; test populations therefore differ.",
    )
    _save(fig, "grouped_profile_split_comparison.png")
    chart_map.append(
        {
            "figure": "grouped_profile_split_comparison.png",
            "family": "grouped bar",
            "claim": "Measures sensitivity to identical profiles crossing partitions.",
        }
    )

    plot = advanced_strategy_results.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot["strategy"],
        plot["macro_f1"],
        color=COLORS["olive"],
        edgecolor=COLORS["olive_dark"],
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in plot["macro_f1"]],
        padding=3,
    )
    ax.set_xlabel("Macro-F1 on strategy evaluation split")
    ax.set_ylabel("")
    ax.set_xlim(0, min(1.0, max(0.1, plot["macro_f1"].max() * 1.16)))
    _header(
        fig,
        ax,
        "Advanced multiclass strategy comparison",
        "All strategies use the same training-only development and evaluation partitions; official test metrics are unchanged.",
    )
    _save(fig, "advanced_multiclass_strategy_comparison.png")
    chart_map.append(
        {
            "figure": "advanced_multiclass_strategy_comparison.png",
            "family": "ranked horizontal bar",
            "claim": "Compares thresholding, calibration, weights, ordinal, two-stage and ensemble approaches.",
        }
    )
    return pd.DataFrame(chart_map)


def update_report_with_robustness(
    *,
    confidence_intervals: pd.DataFrame,
    repeated_cv_summary: pd.DataFrame,
    feature_engineering_ablation: pd.DataFrame,
    grouped_split_results: pd.DataFrame,
    advanced_strategy_results: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    smotenc_comparison: pd.DataFrame,
    paired_comparison: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    threshold_tuning: pd.DataFrame,
    class_weight_results: pd.DataFrame,
) -> None:
    selected_threshold = threshold_tuning.loc[
        threshold_tuning["selection"].fillna("").str.contains("max_macro_f1")
    ].iloc[0]
    best_advanced = advanced_strategy_results.sort_values(
        "macro_f1", ascending=False
    ).iloc[0]
    baseline = advanced_strategy_results.loc[
        advanced_strategy_results["strategy"].eq("Weighted XGBoost baseline")
    ].iloc[0]
    engineered = feature_engineering_ablation.loc[
        feature_engineering_ablation["configuration"].eq(
            "Original + engineered features"
        )
    ].iloc[0]
    original = feature_engineering_ablation.loc[
        feature_engineering_ablation["configuration"].eq("Original features only")
    ].iloc[0]
    grouped_macro = grouped_split_results.loc[
        grouped_split_results["split_design"].eq("Profile-grouped holdout")
        & grouped_split_results["metric"].eq("macro_f1"),
        "value",
    ].iloc[0]
    random_macro = grouped_split_results.loc[
        grouped_split_results["split_design"].eq("Official random holdout")
        & grouped_split_results["metric"].eq("macro_f1"),
        "value",
    ].iloc[0]
    macro_interval = confidence_intervals.loc[
        confidence_intervals["metric"].eq("multiclass_macro_f1")
    ].iloc[0]
    binary_auc_interval = confidence_intervals.loc[
        confidence_intervals["metric"].eq("binary_roc_auc")
    ].iloc[0]
    repeated_report = repeated_cv_summary[
        [
            "model",
            "folds",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "class_1_recall_mean",
            "class_2_recall_mean",
            "macro_roc_auc_ovr_mean",
        ]
    ]
    grouped_report = grouped_split_results.loc[
        grouped_split_results["metric"].isin(
            [
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "class_1_recall",
                "class_2_recall",
                "macro_roc_auc_ovr",
            ]
        )
    ]
    advanced_report = advanced_strategy_results[
        [
            "strategy",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "class_1_recall",
            "class_1_precision",
            "class_2_recall",
            "macro_roc_auc_ovr",
            "multiclass_log_loss",
            "multiclass_ece",
        ]
    ]
    calibration_report = calibration_comparison[
        [
            "probability",
            "macro_f1",
            "balanced_accuracy",
            "class_1_recall",
            "class_2_recall",
            "multiclass_log_loss",
            "multiclass_brier_score",
            "multiclass_ece",
        ]
    ]

    section = f"""
## Robustness analysis quantifies uncertainty and tests alternative ML designs

The official holdout estimates are now accompanied by stratified bootstrap intervals. Multiclass macro-F1 was **{macro_interval['estimate']:.3f}** with a 95% interval of **[{macro_interval['ci_lower_95']:.3f}, {macro_interval['ci_upper_95']:.3f}]**. Binary ROC-AUC was **{binary_auc_interval['estimate']:.3f}** with a 95% interval of **[{binary_auc_interval['ci_lower_95']:.3f}, {binary_auc_interval['ci_upper_95']:.3f}]**. These intervals describe sampling uncertainty in the fixed test set; they do not cover temporal or population shift.

![Bootstrap confidence intervals](figures/bootstrap_confidence_intervals.png)

{markdown_table(confidence_intervals.round(4))}

### Repeated CV confirms the model-ranking trade-off

Five-fold cross-validation repeated across three seeds confirms that no single model dominates every objective. ExtraTrees remains competitive on macro-F1, while weighted XGBoost retains materially stronger prediabetes recall. The comparison is based only on a stratified training sample and does not reuse the official test set for selection.

![Repeated CV comparison](figures/repeated_cv_model_comparison.png)

{markdown_table(repeated_report.round(4))}

### Engineered features provide limited incremental average performance

Using identical repeated-CV folds, the original-plus-engineered specification produced mean macro-F1 **{engineered['macro_f1_mean']:.3f}**, versus **{original['macro_f1_mean']:.3f}** for original features alone. The result should be interpreted as an ablation finding: engineered variables improve transparency and specific interactions, but they do not guarantee a large aggregate score increase.

{markdown_table(feature_engineering_ablation.round(4))}

### Grouping identical profiles changes the evaluation population

The profile-grouped split had zero predictor-profile overlap and produced macro-F1 **{grouped_macro:.3f}**, compared with **{random_macro:.3f}** on the official random holdout. Because the grouped and random test populations differ, this is a robustness sensitivity rather than a paired estimate of leakage bias.

![Grouped split comparison](figures/grouped_profile_split_comparison.png)

{markdown_table(grouped_report.round(4))}

### Advanced strategies did not automatically solve prediabetes separation

The strategy evaluation split was created entirely inside the original training partition. The best validation-only strategy was **{best_advanced['strategy']}**, with macro-F1 **{best_advanced['macro_f1']:.3f}**, versus **{baseline['macro_f1']:.3f}** for the weighted XGBoost baseline. Probability multipliers selected class-1 factor **{selected_threshold['class_1_multiplier']:.2f}** and class-2 factor **{selected_threshold['class_2_multiplier']:.2f}**. These are model-development results, not new independent test claims.

![Advanced strategy comparison](figures/advanced_multiclass_strategy_comparison.png)

{markdown_table(advanced_report.round(4))}

The comparison includes class-specific probability adjustment, custom class weighting, multinomial probability calibration, ordinal cumulative models, a two-stage high-risk model, and an out-of-fold Logistic/ExtraTrees/XGBoost ensemble. The two-stage model improved macro-F1 mainly by recovering majority-class precision; its prediabetes recall remained below the weighted baseline. It is therefore a candidate operating design, not an unconditional replacement.

### Calibration, SMOTE-NC and paired statistical comparisons

Multiclass calibration was evaluated using log loss, multiclass Brier score and expected calibration error. Calibration sharply improved probability quality but collapsed prediabetes recall to zero at the default argmax rule, demonstrating that calibrated probabilities still require a separate decision policy. Moderate SMOTE-NC was run inside each CV training fold and compared with balanced sample weights; it reduced macro-F1, balanced accuracy and diabetes recall, so class weighting remains the preferred imbalance treatment.

{markdown_table(calibration_report.round(4))}

{markdown_table(smotenc_comparison.round(4))}

Paired bootstrap intervals compare models on the same observations. An interval crossing zero indicates that the available test set does not establish a clear difference for that metric. Exact McNemar tests compare paired correctness, but statistical significance should not be confused with practical value in a dataset this large.

{markdown_table(paired_comparison.round(4))}

{markdown_table(statistical_tests.round(6))}

Detailed selection surfaces are saved in `multiclass_probability_threshold_tuning.csv` and `class_weight_sensitivity.csv`:

{markdown_table(threshold_tuning.loc[threshold_tuning['selection'].fillna('').ne('')].round(4))}

{markdown_table(class_weight_results.head(8).round(4))}
"""

    report_path = REPORT_DIR / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    summary_block = f"""
**Robustness conclusion:** The official macro-F1 95% bootstrap interval is **[{macro_interval['ci_lower_95']:.3f}, {macro_interval['ci_upper_95']:.3f}]**. A zero-overlap profile-grouped holdout produced macro-F1 **{grouped_macro:.3f}**, close to the official **{random_macro:.3f}**. A two-stage model reached training-only evaluation macro-F1 **{best_advanced['macro_f1']:.3f}**, but reduced prediabetes recall and therefore remains a candidate for future external validation rather than the new official model.
"""
    report = _replace_marked_section(
        report,
        summary_block,
        ROBUSTNESS_SUMMARY_START,
        ROBUSTNESS_SUMMARY_END,
        before_heading="## The rare prediabetes class defines the multiclass difficulty",
    )
    report = _replace_marked_section(
        report,
        section,
        ROBUSTNESS_START,
        ROBUSTNESS_END,
        before_heading="## Limitations and robustness checks",
    )
    report_path.write_text(report, encoding="utf-8")

    interview_path = REPORT_DIR / "INTERVIEW_DEFENSE_NOTES.md"
    interview = interview_path.read_text(encoding="utf-8")
    interview_section = f"""
## Additional robustness defense

- **Uncertainty:** macro-F1 95% bootstrap interval was [{macro_interval['ci_lower_95']:.3f}, {macro_interval['ci_upper_95']:.3f}].
- **Repeated CV:** model rankings were checked over 15 folds rather than one split.
- **Duplicate profiles:** a grouped-profile holdout produced macro-F1 {grouped_macro:.3f} with zero profile overlap.
- **Feature engineering:** repeated-CV ablation compared the original 21 predictors against all 52 predictors.
- **Alternative designs:** ordinal, two-stage, ensemble, calibrated, threshold-adjusted and custom-weight models were evaluated on training-only partitions.
- **SMOTE-NC:** tested inside CV folds; its result is reported rather than assumed beneficial.
- **Statistical testing:** paired bootstrap intervals and exact McNemar tests distinguish uncertainty from practical model value.
"""
    interview = _replace_marked_section(
        interview,
        interview_section,
        ROBUSTNESS_START,
        ROBUSTNESS_END,
        before_heading="## Ten likely interview questions",
    )
    interview_path.write_text(interview, encoding="utf-8")

    cv_path = REPORT_DIR / "CV_SUMMARY.md"
    cv = cv_path.read_text(encoding="utf-8")
    cv_section = f"""
## Robustness-focused bullet options

- Validated BRFSS models with 15-fold repeated CV and 1,000 bootstrap resamples.
- Built a two-stage diabetes-risk model with training-only macro-F1 {best_advanced['macro_f1']:.3f}.
- Tested grouped splits, calibration, SMOTE-NC and paired model significance.
"""
    cv = _replace_marked_section(
        cv,
        cv_section,
        ROBUSTNESS_START,
        ROBUSTNESS_END,
        before_heading="## Project description — compact",
    )
    cv_path.write_text(cv, encoding="utf-8")

    readme_path = PROJECT_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme_section = """
## Robustness extensions

- 1,000-repetition bootstrap confidence intervals for held-out metrics
- 5×3 repeated stratified CV for model and feature-set stability
- Profile-grouped holdout with zero identical-profile overlap
- Validation-only multiclass threshold and class-weight searches
- Ordinal, two-stage, calibrated and out-of-fold ensemble comparisons
- Fold-safe moderate SMOTE-NC sensitivity analysis
- Paired bootstrap model differences and exact McNemar tests

The official headline metrics remain unchanged. Advanced strategies are reported as training-only development evidence unless explicitly identified as held-out inference.
"""
    readme = _replace_marked_section(
        readme,
        readme_section,
        ROBUSTNESS_START,
        ROBUSTNESS_END,
        before_heading="## Limitations",
    )
    readme_path.write_text(readme, encoding="utf-8")


def append_robustness_chart_map(chart_map: pd.DataFrame) -> None:
    path = REPORT_DIR / "tables" / "chart_map.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    existing = existing.loc[
        ~existing.get("figure", pd.Series(dtype=str)).isin(chart_map["figure"])
    ]
    pd.concat([existing, chart_map], ignore_index=True).to_csv(path, index=False)
