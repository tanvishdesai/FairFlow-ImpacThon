"""
Statistical summaries for multi-seed FairFlow experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "demographic_parity_ratio",
    "equalized_odds_gap",
    "intervention_rate",
    "rolling_fair_rate",
    "rolling_tail_avg_dpr",
    "rolling_tail_fair_rate",
)


@dataclass(frozen=True)
class AggregateSpec:
    group_cols: tuple[str, ...]
    metric_cols: tuple[str, ...] = DEFAULT_METRICS
    seed_col: str = "seed"


def _ci_half_width(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    n = len(values)
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def aggregate_with_intervals(frame: pd.DataFrame, spec: AggregateSpec) -> pd.DataFrame:
    """
    Aggregate per-run results into mean/std/95% CI tables.
    """
    if frame.empty:
        columns = list(spec.group_cols)
        for metric in spec.metric_cols:
            columns.extend(
                [
                    f"{metric}_mean",
                    f"{metric}_std",
                    f"{metric}_ci_low",
                    f"{metric}_ci_high",
                ]
            )
        columns.append("n_runs")
        return pd.DataFrame(columns=columns)

    grouped = frame.groupby(list(spec.group_cols), dropna=False)
    rows: list[dict] = []
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(spec.group_cols, keys))
        row["n_runs"] = int(len(part))
        for metric in spec.metric_cols:
            if metric not in part.columns:
                continue
            values = pd.to_numeric(part[metric], errors="coerce")
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci_half = _ci_half_width(values)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_ci_low"] = mean_value - ci_half
            row[f"{metric}_ci_high"] = mean_value + ci_half
        rows.append(row)
    return pd.DataFrame(rows)


def paired_win_summary(
    frame: pd.DataFrame,
    *,
    against_method: str,
    metric: str,
    higher_is_better: bool,
    group_cols: Iterable[str] = ("dataset", "model", "seed"),
    method_col: str = "method",
) -> pd.DataFrame:
    """
    Count method wins against a reference baseline on matched tasks.
    """
    if frame.empty:
        return pd.DataFrame(columns=["method", "wins", "losses", "ties", "metric", "against_method"])

    pivot = frame.pivot_table(
        index=list(group_cols),
        columns=method_col,
        values=metric,
        aggfunc="first",
    )
    if against_method not in pivot.columns:
        return pd.DataFrame(columns=["method", "wins", "losses", "ties", "metric", "against_method"])

    rows: list[dict] = []
    baseline = pivot[against_method]
    for method in pivot.columns:
        if method == against_method:
            continue
        comparison = pivot[method] - baseline
        if not higher_is_better:
            comparison = -comparison
        wins = int((comparison > 1e-12).sum())
        losses = int((comparison < -1e-12).sum())
        ties = int(comparison.shape[0] - wins - losses)
        rows.append(
            {
                "method": method,
                "against_method": against_method,
                "metric": metric,
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    return pd.DataFrame(rows)


def format_interval(mean_value: float, low_value: float, high_value: float, *, decimals: int = 3) -> str:
    """Format a mean and 95% confidence interval for tables."""
    template = f"{{:.{decimals}f}}"
    return f"{template.format(mean_value)} [{template.format(low_value)}, {template.format(high_value)}]"

