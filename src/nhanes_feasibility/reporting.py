from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR, PROJECT_NAME, REPORT_DIR
from .evaluate import CLASS_NAMES

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


def _add_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.80)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        textwrap.fill(title, 76),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.90,
        textwrap.fill(subtitle, 105),
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def generate_figures(
    label_distribution: pd.DataFrame,
    xgb_details: dict[str, object],
    xgb_topk: pd.DataFrame,
    figure_dir: Path = FIGURE_DIR,
) -> None:
    _theme()

    fig, ax = plt.subplots(figsize=(8, 5))
    plot = label_distribution.sort_values("class")
    palette = [COLORS["blue"], COLORS["gold"], COLORS["orange"]]
    bars = ax.bar(
        plot["class_name"],
        plot["count"],
        color=palette,
        edgecolor=[COLORS["blue_dark"], COLORS["gold_dark"], COLORS["orange_dark"]],
        linewidth=1,
    )
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in plot["count"]], padding=3)
    ax.set_xlabel("")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=10)
    _add_header(
        fig,
        ax,
        "Label distribution",
        "Adults aged 20+ with labels constructed from available glycemic evidence; highest-risk evidence wins.",
    )
    _save(fig, figure_dir / "label_distribution.png")

    matrix = np.asarray(xgb_details["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = sns.blend_palette(
        [TOKENS["panel"], "#EAF1FE", COLORS["blue"]], as_cmap=True
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=1,
        linecolor=TOKENS["panel"],
        cbar=False,
        ax=ax,
    )
    labels = [CLASS_NAMES[i] for i in range(3)]
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Actual stage")
    _add_header(
        fig,
        ax,
        "XGBoost confusion matrix",
        "Held-out test participants; classes 1 and 2 are both treated as high-risk for the testing-prioritisation decision.",
    )
    _save(fig, figure_dir / "confusion_matrix_xgboost.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        xgb_topk["testing_percentage"],
        xgb_topk["high_risk_capture_rate"],
        marker="o",
        color=COLORS["blue_dark"],
        markerfacecolor=COLORS["blue"],
        linewidth=1.5,
        label="High-risk capture",
    )
    ax.plot(
        xgb_topk["testing_percentage"],
        xgb_topk["diabetic_capture_rate"],
        marker="s",
        linestyle="--",
        color=COLORS["orange_dark"],
        markerfacecolor=COLORS["orange"],
        linewidth=1.2,
        label="Diabetes-range capture",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color=TOKENS["ink"], label="Random ranking")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Share sent for confirmatory testing")
    ax.set_ylabel("Cases captured")
    ax.legend(loc="lower right", frameon=False)
    _add_header(
        fig,
        ax,
        "High-risk capture as testing volume increases",
        "XGBoost ranking on the held-out test set; the dotted line represents random selection.",
    )
    _save(fig, figure_dir / "high_risk_capture_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        xgb_topk["high_risk_capture_rate"],
        xgb_topk["precision_among_tested"],
        marker="o",
        color=COLORS["olive_dark"],
        markerfacecolor=COLORS["olive"],
        linewidth=1.5,
    )
    for _, row in xgb_topk.iterrows():
        ax.annotate(
            f"top {row['testing_percentage']:.0%}",
            (row["high_risk_capture_rate"], row["precision_among_tested"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            color=TOKENS["muted"],
        )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("High-risk recall / capture")
    ax.set_ylabel("Precision among tested")
    _add_header(
        fig,
        ax,
        "Top-k testing precision-recall trade-off",
        "Each point is an operational testing-volume cut on the held-out test set.",
    )
    _save(fig, figure_dir / "topk_precision_recall.png")

    mean_predicted = np.asarray(xgb_details["calibration_mean_predicted"])
    fraction_positive = np.asarray(xgb_details["calibration_fraction_positive"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        color=COLORS["pink_dark"],
        markerfacecolor=COLORS["pink"],
        linewidth=1.4,
        label="XGBoost",
    )
    ax.plot([0, 1], [0, 1], linestyle=":", color=TOKENS["ink"], label="Ideal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Mean predicted high-risk probability")
    ax.set_ylabel("Observed high-risk fraction")
    ax.legend(frameon=False)
    _add_header(
        fig,
        ax,
        "High-risk calibration curve",
        "Ten quantile bins on the held-out test set; ranking is the primary use, but calibration is shown for completeness.",
    )
    _save(fig, figure_dir / "calibration_curve.png")


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    view = frame.head(max_rows) if max_rows else frame
    display = view.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.3f}"
            if isinstance(value, float)
            else str(value)
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    rows = [
        [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        for row in display.itertuples(index=False, name=None)
    ]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def write_markdown_reports(
    *,
    files_used: pd.DataFrame,
    final_sample_size: int,
    label_distribution: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    topk: pd.DataFrame,
    burden: pd.DataFrame,
    criteria: pd.DataFrame,
    passed: bool,
    label_evidence: pd.DataFrame,
    report_dir: Path = REPORT_DIR,
) -> str:
    report_dir.mkdir(parents=True, exist_ok=True)
    recommendation = (
        "Proceed with the NHANES upgrade as a CV project, while describing it as a "
        "testing-prioritisation feasibility study rather than a diagnostic model."
        if passed
        else "NHANES Tier-1 pre-test prioritisation did not create enough incremental "
        "value. Recommended fallback: retain BRFSS as the official project and polish "
        "feature engineering, threshold tuning and reporting."
    )
    verdict_label = "GO — proceed with NHANES upgrade" if passed else "NO-GO — fall back to BRFSS"
    leakage_exclusions = pd.DataFrame(
        [
            ["DIQ010", "Used only to construct the label"],
            ["DIQ160", "Self-reported prediabetes; used only to construct the label"],
            ["DIQ170 / DIQ172", "Diabetes-risk awareness/perception; excluded to avoid leakage"],
            ["LBXGH", "HbA1c; used only to construct the label"],
            ["LBXGLU / LBDGLUSI", "Fasting glucose and derivative; label-only/excluded"],
            ["LBXGLT / LBDGLTSI", "2-hour OGTT glucose and derivative; label-only/excluded"],
            ["HOMA-IR / TyG / glucose-derived features", "Not constructed"],
        ],
        columns=["variable_or_family", "treatment"],
    )
    model_view = model_comparison.copy()
    numeric_columns = model_view.select_dtypes(include="number").columns
    model_view[numeric_columns] = model_view[numeric_columns].round(3)
    topk_view = topk.copy()
    for column in (
        "testing_percentage",
        "testing_volume_reduction",
        "high_risk_capture_rate",
        "diabetic_capture_rate",
        "diabetes_prone_capture_rate",
        "precision_among_tested",
    ):
        topk_view[column] = topk_view[column].map(lambda value: f"{value:.1%}")
    burden_view = burden.copy()
    for column in (
        "target_high_risk_recall",
        "minimum_testing_percentage",
        "testing_volume_reduction",
    ):
        burden_view[column] = burden_view[column].map(lambda value: f"{value:.1%}")

    verdict = f"""# NHANES Pilot Verdict

## Technical summary

**Decision: {verdict_label}.**

This pilot tested whether quickly collectable, non-glycemic Tier-1 indicators can rank adults for confirmatory HbA1c, fasting-glucose or OGTT testing. It does **not** diagnose diabetes. Glycemic variables and self-reported diabetes status were used only to construct the ground-truth stage.

**Recommendation:** {recommendation}

## Dataset files used

{_markdown_table(files_used)}

The analytical cohort contains **{final_sample_size:,} adults aged 20 years or older** after the one-row-per-participant merge and age restriction. NHANES complex survey weights were not used for model fitting; results describe this feasibility sample, not population prevalence.

## Target construction and label distribution

When label markers disagreed, the highest-risk class was assigned.

{_markdown_table(label_evidence)}

{_markdown_table(label_distribution)}

![Label distribution](figures/label_distribution.png)

## Tier-1 features used

The main model used **{len(feature_dictionary):,} available non-glycemic features**. Missing candidates were skipped rather than forced.

{_markdown_table(feature_dictionary[["feature", "source_group", "type", "missing_pct"]])}

No clean, generally administered gestational-diabetes or PCOS history variable was identified in the selected 2015–2016 components, so no female-specific feature was forced into the model.

## Features excluded due to leakage

{_markdown_table(leakage_exclusions)}

## Model comparison

{_markdown_table(model_view)}

![XGBoost confusion matrix](figures/confusion_matrix_xgboost.png)

The business decision is based primarily on high-risk ranking quality and top-k capture, not on three-class accuracy alone.

## Top-k testing simulation

{_markdown_table(topk_view)}

![High-risk capture curve](figures/high_risk_capture_curve.png)

![Top-k precision-recall](figures/topk_precision_recall.png)

## Testing required for target capture

{_markdown_table(burden_view)}

![Calibration curve](figures/calibration_curve.png)

## Acceptance criteria

Criterion C defines “materially beats” before inspecting results as at least +0.03 high-risk PR-AUC and +0.05 absolute high-risk capture at top 30% or top 40% versus the simple rule score.

{_markdown_table(criteria)}

## Final recommendation

**{recommendation}**

{"Suggested CV positioning: Built an NHANES 2015–2016 non-glycemic risk-ranking pipeline that simulates constrained confirmatory-testing capacity, compares statistical, boosted-tree and transparent rule baselines, and quantifies the testing volume needed to capture high-risk participants." if passed else "Do not position this pilot as the replacement project. Keep the BRFSS work as the official project and retain this NHANES pilot as documented negative feasibility evidence."}

## Limitations and robustness notes

- This is a held-out feasibility result from one NHANES cycle, not external validation.
- NHANES is a complex survey; unweighted predictive evaluation does not estimate US prevalence.
- Some fasting/OGTT labels are structurally missing because those tests apply to examination subsamples.
- Class 0 means no available criterion met the class 1 or class 2 thresholds; it does not prove absence of dysglycemia.
- Operational value depends on local prevalence, testing costs, capacity and acceptable miss rates.
"""
    (report_dir / "NHANES_PILOT_VERDICT.md").write_text(verdict, encoding="utf-8")

    top40_row = topk.loc[np.isclose(topk["testing_percentage"], 0.40)].iloc[0]
    at80 = burden.loc[np.isclose(burden["target_high_risk_recall"], 0.80)].iloc[0]
    summary = f"""# NHANES Feasibility Summary

## What problem we tested

We tested whether age, body measurements, blood pressure, medical history, lifestyle and access-to-care variables can help decide who should receive confirmatory glycemic testing first when testing capacity is limited.

## Why glycemic biomarkers were excluded from X

HbA1c, fasting glucose and OGTT glucose would make the prioritisation task circular. They were used only to define the outcome, never as model inputs. The model is a **pre-test ranking tool**, not a diabetes diagnosis.

## Was Tier-1 non-glycemic data enough?

**Decision: {verdict_label}.** The strongest evidence is the held-out testing simulation and the comparison against a transparent rule score. XGBoost's high-risk PR-AUC was **{float(model_comparison.loc[model_comparison['model'].eq('XGBoost'), 'high_risk_pr_auc'].iloc[0]):.3f}**.

## How much testing burden could be reduced?

Testing the top 40% captured **{float(top40_row['high_risk_capture_rate']):.1%}** of high-risk participants and reduced immediate testing volume by **{float(top40_row['testing_volume_reduction']):.1%}**. Reaching 80% high-risk capture required testing **{float(at80['minimum_testing_percentage']):.1%}** of participants, a **{float(at80['testing_volume_reduction']):.1%}** reduction versus universal testing.

## Does the project have logical and business value?

The concept has value only if ranking meaningfully reduces immediate testing while retaining an acceptable share of high-risk cases. The acceptance criteria make that trade-off explicit and compare the boosted model with a simple operational rule.

## Final go/no-go decision

**{recommendation}**
"""
    (report_dir / "FEASIBILITY_SUMMARY.md").write_text(summary, encoding="utf-8")
    return verdict_label


def write_html_report(report_dir: Path = REPORT_DIR) -> Path:
    """Create one portable technical report surface from the requested Markdown."""
    verdict_path = report_dir / "NHANES_PILOT_VERDICT.md"
    verdict_text = verdict_path.read_text(encoding="utf-8")

    def inline_markup(value: str) -> str:
        escaped = html.escape(value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        return escaped

    lines = verdict_text.splitlines()
    rendered: list[str] = []
    index = 0
    in_list = False
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
            rendered.extend(f"<th>{inline_markup(cell)}</th>" for cell in headers)
            rendered.append("</tr></thead><tbody>")
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                rendered.append("<tr>")
                rendered.extend(f"<td>{inline_markup(cell)}</td>" for cell in cells)
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
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            level = len(heading_match.group(1))
            rendered.append(
                f"<h{level}>{inline_markup(heading_match.group(2))}</h{level}>"
            )
        elif line.startswith("- "):
            if not in_list:
                rendered.append("<ul>")
                in_list = True
            rendered.append(f"<li>{inline_markup(line[2:])}</li>")
        else:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            rendered.append(f"<p>{inline_markup(line)}</p>")
        index += 1
    if in_list:
        rendered.append("</ul>")

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(PROJECT_NAME)}</title>
  <style>
    body {{ margin: 0; background: #fcfcfd; color: #1f2430; font-family: "Segoe UI", Arial, sans-serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 40px 24px 64px; }}
    h1 {{ font-size: 30px; margin: 0 0 24px; }}
    h2 {{ margin-top: 38px; border-bottom: 1px solid #e6e8f0; padding-bottom: 8px; }}
    h3 {{ margin-top: 28px; }}
    p, li {{ line-height: 1.58; }}
    .chart {{ width: 100%; background: white; margin: 24px 0; border: 1px solid #e6e8f0; }}
    .table-wrap {{ overflow-x: auto; margin: 18px 0 26px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6e8f0; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eaf1fe; color: #2e4780; position: sticky; top: 0; }}
  </style>
</head>
<body>
<main>
  {"".join(rendered)}
</main>
</body>
</html>
"""
    path = report_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
