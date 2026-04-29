"""
Metrics and rolling-stream summaries for the FairFlow paper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_ratio(numerator: float, denominator: float, *, default: float = 1.0) -> float:
    """Return a stable ratio for fairness metrics."""
    if denominator == 0:
        return default if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_static_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    protected: Iterable[int],
    *,
    scores: Optional[Iterable[float]] = None,
    intervention_flags: Optional[Iterable[int]] = None,
    fairness_threshold: float = 0.8,
) -> Dict[str, float]:
    """Compute the main performance and fairness metrics used in the paper."""
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    protected = np.asarray(list(protected), dtype=int)

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    precision = safe_ratio(tp, tp + fp, default=0.0)
    recall = safe_ratio(tp, tp + fn, default=0.0)
    specificity = safe_ratio(tn, tn + fp, default=0.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = safe_ratio(2 * precision * recall, precision + recall, default=0.0)

    privileged_mask = protected == 1
    unprivileged_mask = protected == 0

    privileged_rate = float(np.mean(y_pred[privileged_mask] == 1)) if privileged_mask.any() else 0.0
    unprivileged_rate = float(np.mean(y_pred[unprivileged_mask] == 1)) if unprivileged_mask.any() else 0.0
    dp_ratio = safe_ratio(unprivileged_rate, privileged_rate)
    dp_difference = unprivileged_rate - privileged_rate

    def group_tpr(mask: np.ndarray) -> float:
        positives = (y_true[mask] == 1)
        if positives.sum() == 0:
            return 0.0
        return float(np.mean(y_pred[mask][positives] == 1))

    def group_fpr(mask: np.ndarray) -> float:
        negatives = (y_true[mask] == 0)
        if negatives.sum() == 0:
            return 0.0
        return float(np.mean(y_pred[mask][negatives] == 1))

    privileged_tpr = group_tpr(privileged_mask)
    unprivileged_tpr = group_tpr(unprivileged_mask)
    privileged_fpr = group_fpr(privileged_mask)
    unprivileged_fpr = group_fpr(unprivileged_mask)

    tpr_gap = unprivileged_tpr - privileged_tpr
    fpr_gap = unprivileged_fpr - privileged_fpr
    eo_gap = max(abs(tpr_gap), abs(fpr_gap))

    metrics: Dict[str, float] = {
        "n_samples": float(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "privileged_positive_rate": privileged_rate,
        "unprivileged_positive_rate": unprivileged_rate,
        "demographic_parity_ratio": dp_ratio,
        "demographic_parity_difference": dp_difference,
        "privileged_tpr": privileged_tpr,
        "unprivileged_tpr": unprivileged_tpr,
        "privileged_fpr": privileged_fpr,
        "unprivileged_fpr": unprivileged_fpr,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_gap": eo_gap,
        "fairness_pass": float(dp_ratio >= fairness_threshold),
    }

    if scores is not None:
        score_array = np.asarray(list(scores), dtype=float)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, score_array))
        else:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    if intervention_flags is not None:
        intervention_array = np.asarray(list(intervention_flags), dtype=int)
        metrics["intervention_rate"] = float(np.mean(intervention_array))
        metrics["total_interventions"] = float(np.sum(intervention_array))
    else:
        metrics["intervention_rate"] = 0.0
        metrics["total_interventions"] = 0.0

    return metrics


def _blank_counts() -> Dict[str, float]:
    return {
        "n": 0.0,
        "correct": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "tn": 0.0,
        "fn": 0.0,
        "privileged_n": 0.0,
        "unprivileged_n": 0.0,
        "privileged_positive": 0.0,
        "unprivileged_positive": 0.0,
        "privileged_tp": 0.0,
        "privileged_fp": 0.0,
        "privileged_tn": 0.0,
        "privileged_fn": 0.0,
        "unprivileged_tp": 0.0,
        "unprivileged_fp": 0.0,
        "unprivileged_tn": 0.0,
        "unprivileged_fn": 0.0,
    }


def _apply_event(counts: Dict[str, float], y_true: int, y_pred: int, protected: int, delta: float) -> None:
    counts["n"] += delta
    counts["correct"] += delta if y_true == y_pred else 0.0

    if y_true == 1 and y_pred == 1:
        counts["tp"] += delta
    elif y_true == 0 and y_pred == 1:
        counts["fp"] += delta
    elif y_true == 0 and y_pred == 0:
        counts["tn"] += delta
    else:
        counts["fn"] += delta

    prefix = "privileged" if protected == 1 else "unprivileged"
    counts[f"{prefix}_n"] += delta
    counts[f"{prefix}_positive"] += delta if y_pred == 1 else 0.0

    if y_true == 1 and y_pred == 1:
        counts[f"{prefix}_tp"] += delta
    elif y_true == 0 and y_pred == 1:
        counts[f"{prefix}_fp"] += delta
    elif y_true == 0 and y_pred == 0:
        counts[f"{prefix}_tn"] += delta
    else:
        counts[f"{prefix}_fn"] += delta


def _metrics_from_counts(counts: Dict[str, float], fairness_threshold: float) -> Dict[str, float]:
    if counts["n"] == 0:
        return {
            "accuracy": 0.0,
            "demographic_parity_ratio": 1.0,
            "tpr_gap": 0.0,
            "fpr_gap": 0.0,
            "equalized_odds_gap": 0.0,
            "fairness_pass": 1.0,
        }

    accuracy = safe_ratio(counts["correct"], counts["n"], default=0.0)
    privileged_rate = safe_ratio(counts["privileged_positive"], counts["privileged_n"], default=0.0)
    unprivileged_rate = safe_ratio(counts["unprivileged_positive"], counts["unprivileged_n"], default=0.0)
    dp_ratio = safe_ratio(unprivileged_rate, privileged_rate)

    privileged_tpr = safe_ratio(
        counts["privileged_tp"],
        counts["privileged_tp"] + counts["privileged_fn"],
        default=0.0,
    )
    unprivileged_tpr = safe_ratio(
        counts["unprivileged_tp"],
        counts["unprivileged_tp"] + counts["unprivileged_fn"],
        default=0.0,
    )
    privileged_fpr = safe_ratio(
        counts["privileged_fp"],
        counts["privileged_fp"] + counts["privileged_tn"],
        default=0.0,
    )
    unprivileged_fpr = safe_ratio(
        counts["unprivileged_fp"],
        counts["unprivileged_fp"] + counts["unprivileged_tn"],
        default=0.0,
    )

    tpr_gap = unprivileged_tpr - privileged_tpr
    fpr_gap = unprivileged_fpr - privileged_fpr

    return {
        "accuracy": accuracy,
        "privileged_positive_rate": privileged_rate,
        "unprivileged_positive_rate": unprivileged_rate,
        "demographic_parity_ratio": dp_ratio,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_gap": max(abs(tpr_gap), abs(fpr_gap)),
        "fairness_pass": float(dp_ratio >= fairness_threshold),
    }


@dataclass
class RollingFairnessTracker:
    """Track cumulative and rolling fairness metrics in O(n) time."""

    window_size: int = 50
    fairness_threshold: float = 0.8

    def __post_init__(self) -> None:
        self._window_events: deque[tuple[int, int, int]] = deque()
        self._window_counts = _blank_counts()
        self._cumulative_counts = _blank_counts()
        self.steps = 0

    def update(self, y_true: int, y_pred: int, protected: int) -> dict:
        event = (int(y_true), int(y_pred), int(protected))
        self.steps += 1

        _apply_event(self._window_counts, *event, 1.0)
        _apply_event(self._cumulative_counts, *event, 1.0)
        self._window_events.append(event)

        if len(self._window_events) > self.window_size:
            removed = self._window_events.popleft()
            _apply_event(self._window_counts, *removed, -1.0)

        cumulative = _metrics_from_counts(self._cumulative_counts, self.fairness_threshold)
        rolling = _metrics_from_counts(self._window_counts, self.fairness_threshold)
        return {
            "step": self.steps,
            "cumulative_accuracy": cumulative["accuracy"],
            "cumulative_dpr": cumulative["demographic_parity_ratio"],
            "rolling_accuracy": rolling["accuracy"],
            "rolling_dpr": rolling["demographic_parity_ratio"],
            "rolling_tpr_gap": rolling["tpr_gap"],
            "rolling_fpr_gap": rolling["fpr_gap"],
            "rolling_equalized_odds_gap": rolling["equalized_odds_gap"],
            "rolling_fairness_pass": rolling["fairness_pass"],
        }


def build_stream_trace(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    protected: Iterable[int],
    *,
    window_size: int = 50,
    fairness_threshold: float = 0.8,
) -> pd.DataFrame:
    """Build a rolling fairness trace for a decision stream."""
    tracker = RollingFairnessTracker(
        window_size=window_size,
        fairness_threshold=fairness_threshold,
    )
    rows = []
    for truth, pred, group in zip(y_true, y_pred, protected):
        rows.append(tracker.update(int(truth), int(pred), int(group)))
    return pd.DataFrame(rows)


def summarize_stream_trace(trace: pd.DataFrame) -> Dict[str, float]:
    """Summarize a rolling fairness trace into paper-friendly scalars."""
    if trace.empty:
        return {
            "rolling_avg_dpr": 1.0,
            "rolling_min_dpr": 1.0,
            "rolling_fair_rate": 1.0,
            "rolling_max_eo_gap": 0.0,
        }
    return {
        "rolling_avg_dpr": float(trace["rolling_dpr"].mean()),
        "rolling_min_dpr": float(trace["rolling_dpr"].min()),
        "rolling_fair_rate": float(trace["rolling_fairness_pass"].mean()),
        "rolling_max_eo_gap": float(trace["rolling_equalized_odds_gap"].max()),
    }


def paper_selection_score(
    metrics: Dict[str, float],
    *,
    fairness_threshold: float = 0.8,
    fairness_weight: float = 1.5,
    intervention_weight: float = 0.2,
) -> float:
    """
    A simple objective for validation-time selection of post-processing rules.

    Higher is better.
    """
    dpr = metrics["demographic_parity_ratio"]
    fairness_shortfall = max(0.0, fairness_threshold - dpr)
    eo_penalty = metrics["equalized_odds_gap"]
    intervention_penalty = metrics.get("intervention_rate", 0.0)
    return (
        metrics["accuracy"]
        - fairness_weight * fairness_shortfall
        - 0.35 * eo_penalty
        - intervention_weight * intervention_penalty
    )

