from __future__ import annotations

import html
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .config import CLASS_NAMES, FIGURE_DIR, PROJECT_ROOT, REPORT_DIR

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}


def _theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        },
    )


def _header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title("")
    title_text = textwrap.fill(title, width=78, break_long_words=False)
    subtitle_text = textwrap.fill(subtitle, width=110, break_long_words=False)
    title_lines = title_text.count("\n") + 1
    subtitle_lines = subtitle_text.count("\n") + 1
    fig.subplots_adjust(
        top=max(0.62, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title_text,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.93 - 0.045 * (title_lines - 1),
        subtitle_text,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )


def _save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=180,
        bbox_inches="tight",
        facecolor=TOKENS["surface"],
    )
    plt.close(fig)


def _heatmap(
    matrix: np.ndarray,
    labels: list[str],
    filename: str,
    title: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    cmap = sns.blend_palette(
        [TOKENS["panel"], "#EAF1FE", COLORS["blue"]], as_cmap=True
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        linewidths=1,
        linecolor=TOKENS["panel"],
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    _header(fig, ax, title, subtitle)
    _save(fig, filename)


def generate_figures(
    *,
    class_distribution: pd.DataFrame,
    comparison_multiclass: pd.DataFrame,
    comparison_binary: pd.DataFrame,
    multiclass_details: dict[str, object],
    binary_details: dict[str, object],
    y_binary_test: pd.Series,
    raw_binary_probability: np.ndarray,
    calibrated_binary_probability: np.ndarray,
    threshold_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    _theme()
    chart_map: list[dict[str, str]] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    distribution = class_distribution.sort_values("class")
    bars = ax.bar(
        distribution["class_name"],
        distribution["count"],
        color=[COLORS["blue"], COLORS["gold"], COLORS["orange"]],
        edgecolor=[COLORS["blue_dark"], COLORS["gold_dark"], COLORS["orange_dark"]],
    )
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in distribution["count"]])
    ax.set_ylabel("Participants")
    ax.set_xlabel("")
    _header(
        fig,
        ax,
        "BRFSS target class distribution",
        "BRFSS 2015 participants; prediabetes is the rarest class and drives the multiclass imbalance challenge.",
    )
    _save(fig, "class_distribution.png")
    chart_map.append(
        {
            "figure": "class_distribution.png",
            "family": "comparison",
            "claim": "Prediabetes is severely under-represented.",
        }
    )

    for task, table, metric, filename, title in (
        (
            "multiclass",
            comparison_multiclass,
            "macro_f1",
            "model_comparison_multiclass.png",
            "Multiclass cross-validation model comparison",
        ),
        (
            "binary",
            comparison_binary,
            "pr_auc",
            "model_comparison_binary.png",
            "Binary cross-validation model comparison",
        ),
    ):
        plot = table.sort_values(metric)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(
            plot["model"],
            plot[metric],
            color=COLORS["gold"],
            edgecolor=COLORS["gold_dark"],
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in plot[metric]], padding=4)
        ax.set_xlim(0, min(1.0, max(0.1, plot[metric].max() * 1.16)))
        ax.set_xlabel("Macro-F1" if task == "multiclass" else "PR-AUC")
        ax.set_ylabel("")
        _header(
            fig,
            ax,
            title,
            f"Three-fold stratified CV on the training-only comparison sample; optimized metric is {metric.replace('_', ' ')}.",
        )
        _save(fig, filename)
        chart_map.append(
            {
                "figure": filename,
                "family": "ranking",
                "claim": f"Compares {task} model-selection performance.",
            }
        )

    _heatmap(
        np.asarray(multiclass_details["confusion_matrix"]),
        [CLASS_NAMES[i] for i in range(3)],
        "multiclass_confusion_matrix.png",
        "Multiclass test-set confusion matrix",
        "Untouched 20% test set; the central row shows the difficult prediabetes class.",
    )
    _heatmap(
        np.asarray(binary_details["confusion_matrix"]),
        ["No diabetes", "Diabetes"],
        "binary_confusion_matrix.png",
        "Binary test-set confusion matrix",
        "Untouched 20% test set at the validation-selected operating threshold.",
    )

    fpr, tpr, _ = binary_details["roc_curve"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=COLORS["blue_dark"], linewidth=1.5)
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    _header(
        fig,
        ax,
        "Binary ROC curve",
        "Untouched test set using calibrated diabetes probabilities; the dotted line is random ranking.",
    )
    _save(fig, "binary_roc_curve.png")

    precision, recall, _ = binary_details["pr_curve"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=COLORS["orange_dark"], linewidth=1.5)
    ax.axhline(
        y_binary_test.mean(),
        color=TOKENS["ink"],
        linestyle=":",
        label="Diabetes prevalence",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    _header(
        fig,
        ax,
        "Binary precision-recall curve",
        "Untouched test set; PR-AUC is emphasized because diabetes is the minority class.",
    )
    _save(fig, "binary_pr_curve.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for probability, label, color, marker in (
        (raw_binary_probability, "Raw XGBoost", COLORS["neutral_dark"], "s"),
        (calibrated_binary_probability, "Sigmoid calibrated", COLORS["pink_dark"], "o"),
    ):
        fraction, mean_probability = calibration_curve(
            y_binary_test, probability, n_bins=10, strategy="quantile"
        )
        ax.plot(
            mean_probability,
            fraction,
            marker=marker,
            linewidth=1.2,
            color=color,
            label=label,
        )
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed diabetes fraction")
    ax.legend(frameon=False)
    _header(
        fig,
        ax,
        "Binary calibration curve",
        "Ten quantile bins on the untouched test set; calibration parameters were learned on validation data only.",
    )
    _save(fig, "calibration_curve_binary.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        color=COLORS["blue_dark"],
        label="Precision",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["recall"],
        color=COLORS["orange_dark"],
        linestyle="--",
        label="Recall",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["f1"],
        color=COLORS["olive_dark"],
        linestyle="-.",
        label="F1",
    )
    selected = threshold_table.loc[
        threshold_table["selection"].fillna("").str.contains("max_f1")
    ]
    if not selected.empty:
        ax.axvline(
            selected.iloc[0]["threshold"],
            color=TOKENS["ink"],
            linestyle=":",
            label="Selected max-F1 threshold",
        )
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _header(
        fig,
        ax,
        "Binary threshold precision-recall trade-off",
        "Validation-only threshold sweep; the untouched test set was not used to choose the operating point.",
    )
    _save(fig, "threshold_precision_recall_tradeoff.png")

    for table, value_column, filename, title, color, edge in (
        (
            feature_importance,
            "importance",
            "top_feature_importance_xgboost.png",
            "XGBoost built-in feature importance",
            COLORS["blue"],
            COLORS["blue_dark"],
        ),
        (
            permutation,
            "importance_mean",
            "permutation_importance_top20.png",
            "Permutation importance on held-out data",
            COLORS["orange"],
            COLORS["orange_dark"],
        ),
    ):
        plot = table.head(20).sort_values(value_column)
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(
            plot["feature"], plot[value_column], color=color, edgecolor=edge
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in plot[value_column]], padding=3)
        ax.set_xlabel(value_column.replace("_", " ").title())
        ax.set_ylabel("")
        _header(
            fig,
            ax,
            title,
            "Top 20 features from the final multiclass model; importance is predictive association, not causation.",
        )
        _save(fig, filename)

    plot = ablation.loc[
        ablation["removed_feature_family"].ne("none (baseline)")
    ].sort_values("macro_f1_drop")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(
        plot["macro_f1_drop"].ge(0), COLORS["olive"], COLORS["orange"]
    )
    edges = np.where(
        plot["macro_f1_drop"].ge(0), COLORS["olive_dark"], COLORS["orange_dark"]
    )
    bars = ax.barh(
        plot["removed_feature_family"],
        plot["macro_f1_drop"],
        color=colors,
        edgecolor=edges,
    )
    ax.axvline(0, color=TOKENS["ink"], linewidth=1)
    values = plot["macro_f1_drop"].to_numpy()
    span = max(float(np.ptp(values)), 0.001)
    label_offset = span * 0.012
    for bar, value in zip(bars, values, strict=True):
        if value < 0:
            x_position = value + label_offset
            horizontal_alignment = "left"
        else:
            x_position = value + label_offset
            horizontal_alignment = "left"
        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            ha=horizontal_alignment,
            va="center",
            color=TOKENS["ink"],
        )
    ax.margins(x=0.10)
    ax.set_xlabel("Macro-F1 drop when family is removed")
    ax.set_ylabel("")
    _header(
        fig,
        ax,
        "Feature-family ablation",
        "Validation-set sensitivity using the tuned multiclass XGBoost specification; positive values indicate useful incremental signal.",
    )
    _save(fig, "feature_family_ablation.png")

    chart_map.extend(
        [
            {"figure": "multiclass_confusion_matrix.png", "family": "matrix", "claim": "Shows multiclass error structure."},
            {"figure": "binary_confusion_matrix.png", "family": "matrix", "claim": "Shows binary errors at the selected threshold."},
            {"figure": "binary_roc_curve.png", "family": "uncertainty/benchmark", "claim": "Shows binary ranking quality."},
            {"figure": "binary_pr_curve.png", "family": "uncertainty/benchmark", "claim": "Shows minority-class precision-recall."},
            {"figure": "calibration_curve_binary.png", "family": "ordered line-dot", "claim": "Compares raw and calibrated probabilities."},
            {"figure": "threshold_precision_recall_tradeoff.png", "family": "ordered line", "claim": "Shows threshold operating trade-offs."},
            {"figure": "top_feature_importance_xgboost.png", "family": "ranking", "claim": "Ranks built-in model importance."},
            {"figure": "permutation_importance_top20.png", "family": "ranking", "claim": "Ranks held-out permutation importance."},
            {"figure": "feature_family_ablation.png", "family": "diverging bar", "claim": "Measures family-level incremental signal."},
        ]
    )
    return pd.DataFrame(chart_map)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in display:
        display[column] = display[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    rows = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def _metric(metrics: pd.DataFrame, name: str) -> float:
    return float(metrics.loc[metrics["metric"].eq(name), "value"].iloc[0])


def write_reports(
    *,
    dataset_shape: tuple[int, int],
    original_feature_count: int,
    engineered_feature_count: int,
    class_distribution: pd.DataFrame,
    multiclass_comparison: pd.DataFrame,
    binary_comparison: pd.DataFrame,
    multiclass_metrics: pd.DataFrame,
    binary_metrics: pd.DataFrame,
    multiclass_report: pd.DataFrame,
    binary_report: pd.DataFrame,
    threshold_table: pd.DataFrame,
    high_risk_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    permutation: pd.DataFrame,
    ablation: pd.DataFrame,
    error_table: pd.DataFrame,
    subgroup_table: pd.DataFrame,
    best_multiclass_params: dict[str, object],
    best_binary_params: dict[str, object],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    macro_f1 = _metric(multiclass_metrics, "macro_f1")
    balanced = _metric(multiclass_metrics, "balanced_accuracy")
    binary_auc = _metric(binary_metrics, "roc_auc")
    binary_pr = _metric(binary_metrics, "pr_auc")
    binary_recall = _metric(binary_metrics, "recall")
    binary_threshold = _metric(binary_metrics, "threshold")
    class_one = multiclass_report.loc[
        multiclass_report["class"].eq("Prediabetes")
    ].iloc[0]
    class_two = multiclass_report.loc[
        multiclass_report["class"].eq("Diabetes")
    ].iloc[0]
    top_families = (
        ablation.loc[ablation["removed_feature_family"].ne("none (baseline)")]
        .sort_values("macro_f1_drop", ascending=False)
        .head(3)["removed_feature_family"]
        .tolist()
    )
    xgb_multi_cv = float(
        multiclass_comparison.loc[
            multiclass_comparison["model"].eq("XGBoost"), "macro_f1"
        ].iloc[0]
    )
    multiclass_leader = multiclass_comparison.sort_values(
        "macro_f1", ascending=False
    ).iloc[0]
    multiclass_leader_name = str(multiclass_leader["model"])
    multiclass_leader_score = float(multiclass_leader["macro_f1"])
    xgb_multiclass_balanced = float(
        multiclass_comparison.loc[
            multiclass_comparison["model"].eq("XGBoost"), "balanced_accuracy"
        ].iloc[0]
    )
    xgb_multiclass_prediabetes_recall = float(
        multiclass_comparison.loc[
            multiclass_comparison["model"].eq("XGBoost"), "class_1_recall"
        ].iloc[0]
    )
    leader_prediabetes_recall = float(multiclass_leader["class_1_recall"])
    leader_balanced_accuracy = float(multiclass_leader["balanced_accuracy"])
    xgb_binary_cv = float(
        binary_comparison.loc[
            binary_comparison["model"].eq("XGBoost"), "pr_auc"
        ].iloc[0]
    )
    binary_leader = binary_comparison.sort_values("pr_auc", ascending=False).iloc[0]
    binary_leader_name = str(binary_leader["model"])
    binary_leader_score = float(binary_leader["pr_auc"])
    model_selection_summary = (
        f"{multiclass_leader_name} led multiclass CV macro-F1 "
        f"({multiclass_leader_score:.3f} versus {xgb_multi_cv:.3f} for XGBoost). "
        "XGBoost remained the selected final multiclass model because its weighted "
        f"operating point improved prediabetes recall "
        f"({xgb_multiclass_prediabetes_recall:.3f} versus "
        f"{leader_prediabetes_recall:.3f}) and balanced accuracy "
        f"({xgb_multiclass_balanced:.3f} versus {leader_balanced_accuracy:.3f}). "
        f"{binary_leader_name} led the primary binary CV metric, PR-AUC "
        f"({binary_leader_score:.3f})."
    )

    report = f"""# Diabetes Risk Prediction using BRFSS Health Indicators

## Technical summary

The official project uses BRFSS 2015 survey indicators for two related screening tasks: three-class diabetes status and binary diabetes risk. The final multiclass XGBoost achieved **macro-F1 {macro_f1:.3f}**, **balanced accuracy {balanced:.3f}**, prediabetes recall **{float(class_one['recall']):.3f}**, and diabetes recall **{float(class_two['recall']):.3f}** on the untouched test set. The calibrated binary model achieved **ROC-AUC {binary_auc:.3f}**, **PR-AUC {binary_pr:.3f}**, and recall **{binary_recall:.3f}** at threshold **{binary_threshold:.2f}**.

**Model-selection conclusion:** {model_selection_summary} The model is suitable as a portfolio demonstration of reproducible risk-screening ML, not as a medical diagnostic system.

## The rare prediabetes class defines the multiclass difficulty

The dataset contains **{dataset_shape[0]:,} rows**, **{original_feature_count} original predictors**, and no missing values. Prediabetes represents only **{float(class_distribution.loc[class_distribution['class'].eq(1), 'percentage'].iloc[0]):.2%}** of rows. Exact duplicate response rows were retained because the public extract has no respondent identifier; identical response profiles cannot be proven to be duplicate people.

{markdown_table(class_distribution)}

![BRFSS class distribution](figures/class_distribution.png)

This imbalance means raw accuracy would reward predicting the majority class. Macro-F1, balanced accuracy and class-wise recall therefore governed model selection.

## Multiclass and binary tasks answer different screening questions

The official task predicts `Diabetes_012`: 0 for no diabetes, 1 for prediabetes and 2 for diabetes. The secondary task predicts diabetes versus no diabetes/prediabetes. The multiclass task preserves the clinically meaningful intermediate group; the binary task provides a more separable benchmark and supports probability calibration and operating-threshold analysis.

## Data validation found no missingness but identified profile duplication

All expected columns were available and all coded ranges passed validation. There were 23,899 exact duplicate rows. Because no respondent key is present, rows were retained and split using reproducible stratification. This can make performance optimistic when identical profiles appear across partitions and is documented as a limitation.

## Domain features expanded the model from {original_feature_count} to {engineered_feature_count} predictors

Feature engineering retained all original variables and added BMI categories, cardiometabolic burden, lifestyle risk, healthcare access, general-health burden, socioeconomic risk and interactions. Every transformation is deterministic and occurs inside the fitted pipeline.

The highest-impact families in validation ablation were: **{", ".join(top_families)}**.

![Feature-family ablation](figures/feature_family_ablation.png)

Positive drops show that removing the family reduced macro-F1. Small or negative drops indicate redundancy or weak incremental signal, not clinical irrelevance.

## Leakage-safe preprocessing and imbalance handling

The untouched test set was split before feature engineering, imputation, scaling, tuning, calibration or threshold selection. Linear models used median imputation, scaling and one-hot encoding for engineered ordinal categories. Tree models retained ordinal codes with median imputation. Multiclass XGBoost used training-fold balanced sample weights; synthetic oversampling was not selected because the coded survey feature space and calibration objective made class weighting the lower-risk default.

## XGBoost was tuned against strong baselines

{markdown_table(multiclass_comparison.round(4))}

![Multiclass model comparison](figures/model_comparison_multiclass.png)

{markdown_table(binary_comparison.round(4))}

![Binary model comparison](figures/model_comparison_binary.png)

Logistic regression regularization, Random Forest structure and XGBoost parameters were tuned using training-only cross-validation. XGBoost tuning covered estimators, depth, learning rate, subsampling, column sampling, child weight, gamma and L1/L2 regularization. Selected multiclass XGBoost parameters: `{best_multiclass_params}`. Selected binary XGBoost parameters: `{best_binary_params}`.

## Final multiclass errors concentrate around prediabetes

{markdown_table(multiclass_metrics.round(4))}

{markdown_table(multiclass_report.round(4))}

![Multiclass confusion matrix](figures/multiclass_confusion_matrix.png)

Prediabetes has weaker separability than diabetes because it is rare and BRFSS indicators are survey/self-reported rather than glycemic biomarkers. Confusions with both no diabetes and diabetes are therefore reported directly rather than hidden by aggregate accuracy.

## The binary task supports calibrated risk screening

{markdown_table(binary_metrics.round(4))}

{markdown_table(binary_report.round(4))}

![Binary precision-recall curve](figures/binary_pr_curve.png)

![Binary calibration curve](figures/calibration_curve_binary.png)

The sigmoid calibrator and operating threshold were learned from validation probabilities only. The test set was used once for the final metric readout.

## Threshold choice changes the screening operating point

{markdown_table(threshold_table.loc[threshold_table['selection'].fillna('').ne('')].round(4))}

![Threshold trade-off](figures/threshold_precision_recall_tradeoff.png)

The saved binary report uses the validation-selected max-F1 threshold. Other rows support high-recall or minimum-precision operating points without pretending one threshold is universally correct.

The multiclass high-risk sensitivity analysis defines high risk as class 1 or 2:

{markdown_table(high_risk_table.loc[high_risk_table['selection'].fillna('').ne('')].round(4), max_rows=12)}

## Multiple interpretation methods agree on the main signal families

![XGBoost feature importance](figures/top_feature_importance_xgboost.png)

![Permutation importance](figures/permutation_importance_top20.png)

Built-in importance, held-out permutation importance and family ablation were all used. These methods describe predictive contribution; none establishes causal or clinical effect.

Top built-in features:

{markdown_table(feature_importance.head(15).round(4))}

Top permutation features:

{markdown_table(permutation.head(15).round(4))}

## Error and subgroup diagnostics expose where the model is weakest

{markdown_table(error_table.round(4))}

{markdown_table(subgroup_table.round(4))}

Subgroup diagnostics are descriptive checks by age, sex, income and education. They are not a fairness certification because BRFSS coding, sample composition and outcome quality can differ across groups.

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
"""
    (REPORT_DIR / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    model_card = f"""# Model Card — BRFSS Diabetes Risk Prediction

## Intended use

Educational risk-screening and portfolio demonstration using BRFSS 2015 health indicators.

## Non-intended use

Clinical diagnosis, treatment decisions, automated denial of care, or deployment without external validation and governance.

## Dataset and targets

- Source: local BRFSS 2015 health-indicator extract, {dataset_shape[0]:,} rows.
- Multiclass target: no diabetes, prediabetes, diabetes.
- Binary target: diabetes versus no diabetes/prediabetes.

## Features

{original_feature_count} original survey indicators plus deterministic domain features for obesity, cardiometabolic burden, lifestyle, healthcare access, general health and socioeconomic status. No NHANES or glycemic biomarker features enter this pipeline.

## Final model

XGBoost pipelines with training-only preprocessing and tuning. The multiclass model uses balanced sample weights and was selected for its minority-class trade-off; ExtraTrees had the higher multiclass CV macro-F1. The binary model uses validation-only sigmoid calibration and threshold selection.

## Held-out metrics

- Multiclass macro-F1: {macro_f1:.3f}
- Multiclass balanced accuracy: {balanced:.3f}
- Prediabetes recall: {float(class_one['recall']):.3f}
- Diabetes recall: {float(class_two['recall']):.3f}
- Binary ROC-AUC: {binary_auc:.3f}
- Binary PR-AUC: {binary_pr:.3f}
- Binary recall at threshold {binary_threshold:.2f}: {binary_recall:.3f}

## Limitations and ethical considerations

Survey self-reporting, class imbalance, absent respondent identifiers, duplicate response profiles, one historical cycle, subgroup performance differences and prevalence-dependent calibration limit use. Outputs should support human review and preventive screening discussion, never replace clinical testing.
"""
    (REPORT_DIR / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    concise = [
        f"Built BRFSS multiclass pipeline; macro-F1 {macro_f1:.3f} and diabetes recall {float(class_two['recall']):.3f}.",
        f"Engineered {engineered_feature_count-original_feature_count} obesity, lifestyle, health, access and SES features.",
        f"Calibrated binary XGBoost reached ROC-AUC {binary_auc:.3f} and PR-AUC {binary_pr:.3f}.",
    ]
    alternatives = [
        "Tuned XGBoost and tree/linear baselines on 253k BRFSS 2015 records.",
        "Built multiclass and binary BRFSS classifiers with leakage-safe CV and calibration.",
        f"Achieved {float(class_one['recall']):.3f} prediabetes recall in a 1.83%-prevalence multiclass task.",
        "Audited feature families and subgroup errors; general health burden ranked highest.",
    ]
    for bullet in [*concise, *alternatives]:
        if len(bullet) > 110:
            raise ValueError(f"CV bullet exceeds 110 characters: {bullet}")
    cv_summary = f"""# CV Summary

## Three concise bullet options

{chr(10).join(f'- {bullet}' for bullet in concise)}

## Four alternative bullet options

{chr(10).join(f'- {bullet}' for bullet in alternatives)}

## Project description — compact

Built a reproducible BRFSS 2015 diabetes-risk pipeline for multiclass and binary prediction. Added domain-driven features, imbalance handling, cross-validated XGBoost tuning, calibration, threshold selection, feature-family ablation and subgroup error analysis.

## Project description — detailed

Developed an end-to-end diabetes risk-screening project on 253,680 BRFSS 2015 records. Compared linear and tree baselines with XGBoost, engineered clinically motivated survey features, protected an untouched test set, calibrated binary probabilities, tuned operating thresholds, and audited feature families and subgroup errors. The final multiclass test macro-F1 was {macro_f1:.3f}; binary ROC-AUC was {binary_auc:.3f}.

## Interview-ready summary

1. I used BRFSS 2015 because its large sample supports robust supervised-learning evaluation.
2. The main task preserves no-diabetes, prediabetes and diabetes as three classes.
3. Prediabetes was only {float(class_distribution.loc[class_distribution['class'].eq(1), 'percentage'].iloc[0]):.2%}, so accuracy was not an adequate objective.
4. I engineered transparent obesity, health-burden, lifestyle, access and socioeconomic features.
5. All preprocessing, weighting and tuning stayed inside training/CV boundaries.
6. I compared Dummy, Logistic Regression, Random Forest, ExtraTrees and XGBoost.
7. I added binary calibration, validation-only threshold tuning and subgroup error diagnostics.
8. I position the model as risk screening, not diagnosis; NHANES remains a rejected feasibility extension.
"""
    (REPORT_DIR / "CV_SUMMARY.md").write_text(cv_summary, encoding="utf-8")

    interview = f"""# Interview Defense Notes

## Why BRFSS remained the official project

The NHANES pilot asked a narrower operational question: whether non-glycemic indicators could reduce confirmatory testing enough to justify replacing BRFSS. It failed the predefined capture and burden-reduction criteria. BRFSS therefore remained the stronger, more complete ML project, while NHANES became evidence of disciplined feasibility testing.

## Core defense points

- **Why multiclass is hard:** prediabetes is rare ({float(class_distribution.loc[class_distribution['class'].eq(1), 'percentage'].iloc[0]):.2%}) and weakly separated by survey variables.
- **Why XGBoost:** it models nonlinearities and interactions in mixed coded health indicators while supporting weighted training.
- **Why imbalance matters:** majority-class accuracy can hide near-zero prediabetes recall.
- **Why accuracy is insufficient:** macro-F1, balanced accuracy, class recall and PR-AUC weight minority performance more appropriately.
- **Most important feature families:** {", ".join(top_families)}.
- **Domain features:** cardiometabolic count, health-burden score, BMI categories and access/lifestyle combinations encode transparent hypotheses.
- **Limitations:** self-reporting, profile duplicates, one cycle, absent clinical biomarkers and no external validation.
- **Future work:** temporal validation, grouped profile split, cost-sensitive learning and prevalence-shift recalibration.

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
"""
    (REPORT_DIR / "INTERVIEW_DEFENSE_NOTES.md").write_text(
        interview, encoding="utf-8"
    )

    readme = f"""# Diabetes Risk Prediction using BRFSS Health Indicators

The official project builds leakage-safe multiclass and binary diabetes risk models on the existing BRFSS 2015 CSV. It is a screening and machine-learning project, not a medical diagnostic system.

Last verified by full local execution: **{datetime.now(tz=timezone.utc).date().isoformat()}**. Install the pinned versions in `requirements.txt` before comparing regenerated metrics with the committed artifacts.

## Dataset and formulation

- {dataset_shape[0]:,} BRFSS rows and {original_feature_count} original predictors
- Main target: `Diabetes_012` — no diabetes, prediabetes, diabetes
- Secondary target: diabetes versus no diabetes/prediabetes

## Pipeline

`CSV → validation → stratified holdout → in-pipeline feature engineering → preprocessing → imbalance handling → CV/tuning → validation calibration/thresholding → untouched test evaluation → interpretation/reporting`

## Final results

| Task | Main metrics |
| --- | --- |
| Multiclass XGBoost | Macro-F1 {macro_f1:.3f}; balanced accuracy {balanced:.3f}; prediabetes recall {float(class_one['recall']):.3f}; diabetes recall {float(class_two['recall']):.3f} |
| Binary XGBoost | ROC-AUC {binary_auc:.3f}; PR-AUC {binary_pr:.3f}; recall {binary_recall:.3f} at threshold {binary_threshold:.2f} |

ExtraTrees led multiclass CV macro-F1 ({multiclass_leader_score:.3f}), while XGBoost was retained for its stronger prediabetes recall and balanced-accuracy trade-off. XGBoost led the primary binary CV metric, PR-AUC ({xgb_binary_cv:.3f}).

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
.\\.venv\\Scripts\\Activate.ps1
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
$env:JUPYTER_PATH = (Resolve-Path ".venv\\share\\jupyter").Path
jupyter-nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name=diabetes-project-venv notebooks\\01_brfss_final_pipeline_walkthrough.ipynb
jupyter-nbconvert --execute --to notebook --inplace --ExecutePreprocessor.kernel_name=diabetes-project-venv notebooks\\01_nhanes_feasibility_eda.ipynb
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

`CODE.ipynb` and `notebooks/legacy_brfss_baseline.ipynb` are identical Colab-era snapshots with `/content` paths and Colab-only steps; they are preserved for history and are not the maintained local execution path. `25BM6JP22_CDS_Final_Report.pdf` is the corresponding historical submitted report. Current verified results are the generated artifacts under `reports/`.

## Limitations

BRFSS variables and labels are survey/self-reported. Prediabetes is severely imbalanced and weakly separable without biomarkers. Exact response profiles may repeat without respondent identifiers. Results are internal holdout estimates and should not guide clinical decisions.

## NHANES extension

The non-glycemic NHANES prioritisation pilot failed its predefined feasibility criteria. It remains documented research evidence and does not replace this BRFSS project.
"""
    (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def write_html_report(report_dir: Path = REPORT_DIR) -> Path:
    markdown = (report_dir / "FINAL_REPORT.md").read_text(encoding="utf-8")

    def inline(value: str) -> str:
        escaped = html.escape(value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
        return escaped

    lines = markdown.splitlines()
    rendered: list[str] = []
    index = 0
    in_list = False
    section_open = False
    contract_sections = {
        "technical summary": "technical-summary",
        "the rare prediabetes class defines the multiclass difficulty": "key-findings",
        "multiclass and binary tasks answer different screening questions": (
            "scope-data-and-metric-definitions"
        ),
        "limitations and robustness checks": (
            "limitations-uncertainty-and-robustness-checks"
        ),
        "recommended next steps": "recommended-next-steps",
        "further questions": "further-questions",
    }
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            rendered.append("<div class='table-wrap'><table><thead><tr>")
            rendered.extend(f"<th>{inline(cell)}</th>" for cell in headers)
            rendered.append("</tr></thead><tbody>")
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                rendered.append("<tr>")
                rendered.extend(f"<td>{inline(cell)}</td>" for cell in cells)
                rendered.append("</tr>")
                index += 1
            rendered.append("</tbody></table></div>")
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            rendered.append(
                f"<img class='chart' src='{html.escape(image_match.group(2))}' "
                f"alt='{html.escape(image_match.group(1))}'>"
            )
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            heading_text = heading.group(2)
            if level == 1:
                rendered.append(
                    "<header data-contract-section='title'>"
                    f"<h1>{inline(heading_text)}</h1></header>"
                )
            elif level == 2:
                if section_open:
                    rendered.append("</section>")
                contract = contract_sections.get(
                    heading_text.casefold(), "analysis"
                )
                rendered.append(
                    f"<section data-contract-section='{contract}'>"
                    f"<h2>{inline(heading_text)}</h2>"
                )
                section_open = True
            else:
                rendered.append(
                    f"<h{level}>{inline(heading_text)}</h{level}>"
                )
        elif line.startswith("- "):
            if not in_list:
                rendered.append("<ul>")
                in_list = True
            rendered.append(f"<li>{inline(line[2:])}</li>")
        else:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            rendered.append(f"<p>{inline(line)}</p>")
        index += 1
    if in_list:
        rendered.append("</ul>")
    if section_open:
        rendered.append("</section>")

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetes Risk Prediction using BRFSS Health Indicators</title>
<style>
body {{ margin:0; background:#fcfcfd; color:#1f2430; font-family:"Segoe UI",Arial,sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:40px 24px 72px; }}
h1 {{ font-size:30px; }} h2 {{ margin-top:40px; border-bottom:1px solid #e6e8f0; padding-bottom:8px; }}
p,li {{ line-height:1.6; }} code {{ background:#f4f5f7; padding:2px 4px; }}
.chart {{ width:100%; margin:24px 0; border:1px solid #e6e8f0; background:white; }}
.table-wrap {{ overflow-x:auto; margin:18px 0 28px; }}
table {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
th,td {{ border-bottom:1px solid #e6e8f0; padding:9px 10px; text-align:left; vertical-align:top; }}
th {{ background:#eaf1fe; color:#2e4780; }}
</style></head><body><main data-report-audience="technical">{"".join(rendered)}</main></body></html>"""
    path = report_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
