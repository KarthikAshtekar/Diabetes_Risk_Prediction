from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CAPTURE_TARGETS, TOP_K_VALUES


def topk_testing_simulation(
    y_true: pd.Series | np.ndarray,
    high_risk_probability: np.ndarray,
    model_name: str = "XGBoost",
    k_values: tuple[float, ...] = TOP_K_VALUES,
) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(high_risk_probability, dtype=float)
    order = np.argsort(-probability, kind="stable")
    total = len(truth)
    total_high_risk = int((truth > 0).sum())
    total_diabetic = int((truth == 2).sum())
    total_prone = int((truth == 1).sum())
    rows: list[dict[str, object]] = []

    for k in k_values:
        tested_n = min(total, max(1, math.ceil(total * k)))
        selected = truth[order[:tested_n]]
        high_risk_captured = int((selected > 0).sum())
        diabetic_captured = int((selected == 2).sum())
        prone_captured = int((selected == 1).sum())
        rows.append(
            {
                "model": model_name,
                "testing_percentage": k,
                "tested_n": tested_n,
                "testing_volume_reduction": 1 - (tested_n / total),
                "high_risk_cases_captured": high_risk_captured,
                "high_risk_capture_rate": (
                    high_risk_captured / total_high_risk if total_high_risk else np.nan
                ),
                "diabetic_cases_captured": diabetic_captured,
                "diabetic_capture_rate": (
                    diabetic_captured / total_diabetic if total_diabetic else np.nan
                ),
                "diabetes_prone_cases_captured": prone_captured,
                "diabetes_prone_capture_rate": (
                    prone_captured / total_prone if total_prone else np.nan
                ),
                "number_needed_to_test": (
                    tested_n / high_risk_captured if high_risk_captured else np.nan
                ),
                "missed_high_risk_cases": total_high_risk - high_risk_captured,
                "precision_among_tested": high_risk_captured / tested_n,
            }
        )
    return pd.DataFrame(rows)


def testing_burden_for_capture_targets(
    y_true: pd.Series | np.ndarray,
    high_risk_probability: np.ndarray,
    model_name: str = "XGBoost",
    targets: tuple[float, ...] = CAPTURE_TARGETS,
) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(high_risk_probability, dtype=float)
    order = np.argsort(-probability, kind="stable")
    sorted_high_risk = (truth[order] > 0).astype(int)
    cumulative = np.cumsum(sorted_high_risk)
    total_high_risk = int(sorted_high_risk.sum())
    total = len(truth)
    rows: list[dict[str, object]] = []

    for target in targets:
        required_cases = math.ceil(target * total_high_risk)
        if required_cases == 0:
            tested_n = 0
        else:
            tested_n = int(np.searchsorted(cumulative, required_cases, side="left") + 1)
        testing_percentage = tested_n / total if total else np.nan
        rows.append(
            {
                "model": model_name,
                "target_high_risk_recall": target,
                "tested_n": tested_n,
                "minimum_testing_percentage": testing_percentage,
                "testing_volume_reduction": 1 - testing_percentage,
                "high_risk_cases_required": required_cases,
            }
        )
    return pd.DataFrame(rows)


def evaluate_acceptance_criteria(
    xgb_topk: pd.DataFrame,
    xgb_burden: pd.DataFrame,
    model_comparison: pd.DataFrame,
    rule_topk: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    top30 = float(
        xgb_topk.loc[
            np.isclose(xgb_topk["testing_percentage"], 0.30),
            "high_risk_capture_rate",
        ].iloc[0]
    )
    top40 = float(
        xgb_topk.loc[
            np.isclose(xgb_topk["testing_percentage"], 0.40),
            "high_risk_capture_rate",
        ].iloc[0]
    )
    at_80 = xgb_burden.loc[
        np.isclose(xgb_burden["target_high_risk_recall"], 0.80)
    ].iloc[0]
    xgb_pr = float(
        model_comparison.loc[
            model_comparison["model"].eq("XGBoost"), "high_risk_pr_auc"
        ].iloc[0]
    )
    rule_pr = float(
        model_comparison.loc[
            model_comparison["model"].eq("Simple Rule Score"), "high_risk_pr_auc"
        ].iloc[0]
    )
    rule_top30 = float(
        rule_topk.loc[
            np.isclose(rule_topk["testing_percentage"], 0.30),
            "high_risk_capture_rate",
        ].iloc[0]
    )
    rule_top40 = float(
        rule_topk.loc[
            np.isclose(rule_topk["testing_percentage"], 0.40),
            "high_risk_capture_rate",
        ].iloc[0]
    )

    criterion_a = top30 >= 0.70 or top40 >= 0.80
    criterion_b = float(at_80["testing_volume_reduction"]) >= 0.40
    # "Materially" is operationalized before reading results: at least +0.03
    # PR-AUC and +0.05 absolute capture at either the top-30% or top-40% cut.
    criterion_c = (xgb_pr - rule_pr >= 0.03) and (
        (top30 - rule_top30 >= 0.05) or (top40 - rule_top40 >= 0.05)
    )
    criteria = pd.DataFrame(
        [
            {
                "criterion": "A",
                "definition": "Top 30% captures >=70% OR top 40% captures >=80% of high-risk cases",
                "observed": f"top30={top30:.1%}; top40={top40:.1%}",
                "passed": criterion_a,
            },
            {
                "criterion": "B",
                "definition": "At 80% high-risk recall, testing volume is reduced by >=40%",
                "observed": (
                    f"testing={float(at_80['minimum_testing_percentage']):.1%}; "
                    f"reduction={float(at_80['testing_volume_reduction']):.1%}"
                ),
                "passed": criterion_b,
            },
            {
                "criterion": "C",
                "definition": "XGBoost exceeds rule score by >=0.03 PR-AUC and >=0.05 top-k capture",
                "observed": (
                    f"PR-AUC delta={xgb_pr-rule_pr:+.3f}; "
                    f"top30 delta={top30-rule_top30:+.1%}; "
                    f"top40 delta={top40-rule_top40:+.1%}"
                ),
                "passed": criterion_c,
            },
        ]
    )
    return criteria, bool(criteria["passed"].any())
