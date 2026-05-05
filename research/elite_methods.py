"""
Selective post-deployment fairness guards used in the upgraded paper pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from research.baselines import GroupThresholdOptimizer
from research.metrics import RollingFairnessTracker, build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.rl import RLTrainingConfig, build_state_from_history


@dataclass
class SelectiveGuardConfig:
    fairness_threshold: float = 0.8
    fairness_upper_target: float = 1.05
    window_size: int = 50
    activation_dpr_grid: tuple[float, ...] = (0.74, 0.80)
    release_dpr_grid: tuple[float, ...] = (0.88, 0.94)
    overshoot_activation_grid: tuple[float, ...] = (1.08,)
    overshoot_release_grid: tuple[float, ...] = (1.02,)
    warmup_grid: tuple[int, ...] = (25, 50)
    cooldown_grid: tuple[int, ...] = (4, 8)
    max_active_steps_grid: tuple[int, ...] = (16, 32)
    objective_fairness_weight: float = 1.9
    objective_overshoot_weight: float = 0.65
    objective_eo_weight: float = 0.35
    objective_intervention_weight: float = 0.30
    tag: str = "guard"


@dataclass
class AdaptiveGuardConfig:
    fairness_threshold: float = 0.8
    fairness_upper_target: float = 1.02
    window_size: int = 50
    warmup_grid: tuple[int, ...] = (10, 25)
    deficit_weight_grid: tuple[float, ...] = (4.0, 6.0, 8.0)
    overshoot_weight_grid: tuple[float, ...] = (0.75, 1.25)
    intervention_penalty_grid: tuple[float, ...] = (0.005, 0.015)
    deficit_utility_slack_grid: tuple[float, ...] = (0.05, 0.10, 0.15)
    safe_utility_slack_grid: tuple[float, ...] = (0.0, 0.02)
    safe_min_gain_grid: tuple[float, ...] = (0.02, 0.05)
    deficit_min_improvement_grid: tuple[float, ...] = (0.0, 0.01)
    objective_fairness_weight: float = 2.75
    objective_overshoot_weight: float = 0.50
    objective_eo_weight: float = 0.30
    objective_intervention_weight: float = 0.20
    tag: str = "adaptive_guard"


@dataclass
class PrimalDualGuardConfig:
    fairness_threshold: float = 0.8
    fairness_upper_target: float = 1.05
    window_size: int = 50
    warmup_grid: tuple[int, ...] = (10, 25)
    eta_grid: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00, 4.00)
    intervention_penalty_grid: tuple[float, ...] = (0.0, 0.01, 0.02)
    lambda_cap_grid: tuple[float, ...] = (8.0, 16.0, 32.0)
    min_projected_gain_grid: tuple[float, ...] = (0.0, 0.005)
    objective_fairness_weight: float = 3.00
    objective_overshoot_weight: float = 0.20
    objective_eo_weight: float = 0.25
    objective_intervention_weight: float = 0.15
    tag: str = "primal_dual_guard"


@dataclass
class PrimalDualOffsetConfig:
    fairness_threshold: float = 0.8
    target_dpr: float = 0.85
    fairness_upper_target: float = 1.05
    window_size: int = 50
    warmup_steps: int = 10
    lambda_lr_grid: tuple[float, ...] = (0.03, 0.05, 0.08)
    delta_lr_grid: tuple[float, ...] = (0.005, 0.01, 0.02)
    lambda_cap: float = 8.0
    lambda_decay: float = 0.98
    delta_decay: float = 0.995
    max_delta: float = 0.35
    aggressive_multiplier: float = 1.5
    utility_slack: float = 0.05
    mild_offset_base_cost: float = 0.005
    aggressive_offset_base_cost: float = 0.010
    offset_cost_scale: float = 0.02
    group_tolerance: float = 0.02
    objective_fairness_weight: float = 3.25
    objective_overshoot_weight: float = 0.20
    objective_eo_weight: float = 0.25
    objective_intervention_weight: float = 0.15
    tag: str = "primal_dual_offset"


def _tradeoff_objective(
    metrics: dict,
    *,
    fairness_threshold: float,
    fairness_upper_target: float,
    fairness_weight: float,
    overshoot_weight: float,
    eo_weight: float,
    intervention_weight: float,
) -> float:
    dpr = float(metrics["demographic_parity_ratio"])
    fairness_shortfall = max(0.0, fairness_threshold - dpr)
    overshoot = max(0.0, dpr - fairness_upper_target)
    return (
        float(metrics["accuracy"])
        - fairness_weight * fairness_shortfall
        - overshoot_weight * overshoot
        - eo_weight * float(metrics["equalized_odds_gap"])
        - intervention_weight * float(metrics.get("intervention_rate", 0.0))
    )


def _guard_objective(metrics: dict, config: SelectiveGuardConfig) -> float:
    return _tradeoff_objective(
        metrics,
        fairness_threshold=config.fairness_threshold,
        fairness_upper_target=config.fairness_upper_target,
        fairness_weight=config.objective_fairness_weight,
        overshoot_weight=config.objective_overshoot_weight,
        eo_weight=config.objective_eo_weight,
        intervention_weight=config.objective_intervention_weight,
    )


@dataclass
class _GuardParams:
    activation_dpr: float
    release_dpr: float
    overshoot_activation: float
    overshoot_release: float
    warmup_steps: int
    cooldown_steps: int
    max_active_steps: int


@dataclass
class GuardSimulationResult:
    predictions: np.ndarray
    interventions: np.ndarray
    trace: pd.DataFrame
    metrics: dict
    diagnostics: dict


@dataclass
class _AdaptiveParams:
    warmup_steps: int
    deficit_weight: float
    overshoot_weight: float
    intervention_penalty: float
    deficit_utility_slack: float
    safe_utility_slack: float
    safe_min_gain: float
    deficit_min_improvement: float


@dataclass
class _PrimalDualParams:
    warmup_steps: int
    eta: float
    intervention_penalty: float
    lambda_cap: float
    min_projected_gain: float


@dataclass
class _PrimalDualOffsetParams:
    lambda_lr: float
    delta_lr: float


def _blank_group_counts() -> dict[str, float]:
    return {
        "privileged_n": 0.0,
        "unprivileged_n": 0.0,
        "privileged_positive": 0.0,
        "unprivileged_positive": 0.0,
    }


def _apply_group_event(counts: dict[str, float], y_pred: int, protected: int, delta: float) -> None:
    prefix = "privileged" if int(protected) == 1 else "unprivileged"
    counts[f"{prefix}_n"] += delta
    counts[f"{prefix}_positive"] += delta if int(y_pred) == 1 else 0.0


def _dpr_from_group_counts(counts: dict[str, float]) -> float:
    privileged_rate = counts["privileged_positive"] / counts["privileged_n"] if counts["privileged_n"] > 0 else 0.5
    unprivileged_rate = counts["unprivileged_positive"] / counts["unprivileged_n"] if counts["unprivileged_n"] > 0 else 0.5
    if privileged_rate == 0:
        return 1.0 if unprivileged_rate == 0 else 0.0
    return unprivileged_rate / privileged_rate


def _approval_rates_from_buffers(decision_buffers: dict[int, deque[int]]) -> dict[int, float]:
    return {
        int(group): (float(np.mean(buffer)) if len(buffer) > 0 else 0.5)
        for group, buffer in decision_buffers.items()
    }


def _dpr_from_rates(approval_rates: dict[int, float]) -> float:
    if not approval_rates:
        return 1.0
    rates = list(approval_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)
    if max_rate == 0.0:
        return 1.0
    return float(min_rate / (max_rate + 1e-9))


def _project_group_counts(
    decision_history: deque[tuple[int, int]],
    current_counts: dict[str, float],
    *,
    window_size: int,
    proposed_pred: int,
    protected_value: int,
) -> dict[str, float]:
    projected = current_counts.copy()
    if len(decision_history) >= window_size and decision_history:
        oldest_pred, oldest_group = decision_history[0]
        _apply_group_event(projected, oldest_pred, oldest_group, -1.0)
    _apply_group_event(projected, proposed_pred, protected_value, 1.0)
    return projected


def _expected_correctness(decision: int, score: float) -> float:
    return float(score) if int(decision) == 1 else 1.0 - float(score)


class SelectiveThresholdGuard:
    """
    A conservative controller that only activates group-threshold corrections when
    rolling fairness drifts outside a desired operating band.
    """

    def __init__(self, *, threshold_baseline: GroupThresholdOptimizer, config: SelectiveGuardConfig):
        self.threshold_baseline = threshold_baseline
        self.config = config
        self.params_: Optional[_GuardParams] = None
        self.validation_metrics_: dict = {}

    def _candidate_grid(self) -> Iterable[_GuardParams]:
        for values in product(
            self.config.activation_dpr_grid,
            self.config.release_dpr_grid,
            self.config.overshoot_activation_grid,
            self.config.overshoot_release_grid,
            self.config.warmup_grid,
            self.config.cooldown_grid,
            self.config.max_active_steps_grid,
        ):
            params = _GuardParams(*values)
            if params.release_dpr < params.activation_dpr:
                continue
            if params.overshoot_release > params.overshoot_activation:
                continue
            yield params

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "SelectiveThresholdGuard":
        best_score = -1e18
        best_params: Optional[_GuardParams] = None
        best_metrics: dict = {}
        best_diagnostics: dict = {}

        for params in self._candidate_grid():
            result = self.simulate(
                y_true=y_true,
                base_preds=base_preds,
                base_scores=base_scores,
                protected=protected,
                params=params,
            )
            score = _guard_objective(result.metrics, self.config)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result.metrics
                best_diagnostics = result.diagnostics

        if best_params is None:
            raise RuntimeError("SelectiveThresholdGuard search did not evaluate any valid parameter sets.")
        self.params_ = best_params
        self.validation_metrics_ = {**best_metrics, **best_diagnostics, "selection_objective": best_score}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_GuardParams] = None,
    ) -> GuardSimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("Guard parameters are not fitted yet.")

        threshold_preds = self.threshold_baseline.predict(base_scores, protected).astype(int)
        tracker = RollingFairnessTracker(
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )

        final_preds = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)
        active_flags = np.zeros_like(base_preds, dtype=int)
        candidate_flags = (threshold_preds != base_preds).astype(int)
        accepted_flags = np.zeros_like(base_preds, dtype=int)
        activation_events = 0
        active = False
        active_steps = 0
        cooldown_remaining = 0

        for idx, (truth, base_pred, score, group, threshold_pred) in enumerate(
            zip(y_true, base_preds, base_scores, protected, threshold_preds)
        ):
            if cooldown_remaining > 0 and not active:
                cooldown_remaining -= 1

            snapshot = {
                "rolling_dpr": 1.0,
                "rolling_equalized_odds_gap": 0.0,
            }
            if tracker.steps > 0:
                rolling = tracker._window_counts  # pylint: disable=protected-access
                privileged_rate = (
                    rolling["privileged_positive"] / rolling["privileged_n"]
                    if rolling["privileged_n"] > 0
                    else 0.5
                )
                unprivileged_rate = (
                    rolling["unprivileged_positive"] / rolling["unprivileged_n"]
                    if rolling["unprivileged_n"] > 0
                    else 0.5
                )
                if privileged_rate == 0:
                    rolling_dpr = 1.0 if unprivileged_rate == 0 else 0.0
                else:
                    rolling_dpr = unprivileged_rate / privileged_rate
                privileged_tpr = (
                    rolling["privileged_tp"] / (rolling["privileged_tp"] + rolling["privileged_fn"])
                    if (rolling["privileged_tp"] + rolling["privileged_fn"]) > 0
                    else 0.5
                )
                unprivileged_tpr = (
                    rolling["unprivileged_tp"] / (rolling["unprivileged_tp"] + rolling["unprivileged_fn"])
                    if (rolling["unprivileged_tp"] + rolling["unprivileged_fn"]) > 0
                    else 0.5
                )
                privileged_fpr = (
                    rolling["privileged_fp"] / (rolling["privileged_fp"] + rolling["privileged_tn"])
                    if (rolling["privileged_fp"] + rolling["privileged_tn"]) > 0
                    else 0.5
                )
                unprivileged_fpr = (
                    rolling["unprivileged_fp"] / (rolling["unprivileged_fp"] + rolling["unprivileged_tn"])
                    if (rolling["unprivileged_fp"] + rolling["unprivileged_tn"]) > 0
                    else 0.5
                )
                snapshot = {
                    "rolling_dpr": rolling_dpr,
                    "rolling_equalized_odds_gap": max(
                        abs(unprivileged_tpr - privileged_tpr),
                        abs(unprivileged_fpr - privileged_fpr),
                    ),
                }

            low_alert = snapshot["rolling_dpr"] < params.activation_dpr
            high_alert = snapshot["rolling_dpr"] > params.overshoot_activation
            within_band = params.release_dpr <= snapshot["rolling_dpr"] <= params.overshoot_release

            if idx >= params.warmup_steps and not active and cooldown_remaining == 0 and (low_alert or high_alert):
                active = True
                active_steps = 0
                activation_events += 1

            if active and (within_band or active_steps >= params.max_active_steps):
                active = False
                active_steps = 0
                cooldown_remaining = params.cooldown_steps

            final_pred = int(base_pred)
            if active and threshold_pred != base_pred:
                final_pred = int(threshold_pred)
                accepted_flags[idx] = 1

            final_preds[idx] = final_pred
            interventions[idx] = int(final_pred != base_pred)
            active_flags[idx] = int(active)

            tracker.update(int(truth), final_pred, int(group))
            if active:
                active_steps += 1

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics.update(summarize_stream_trace(trace, warmup=self.config.window_size))
        diagnostics = {
            "guard_activation_rate": float(active_flags.mean()),
            "guard_activation_events": float(activation_events),
            "guard_candidate_rate": float(candidate_flags.mean()),
            "guard_accept_rate": float(accepted_flags.mean()),
            "guard_accept_given_candidate": float(
                accepted_flags.sum() / candidate_flags.sum()
            ) if candidate_flags.sum() > 0 else 0.0,
        }
        return GuardSimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
        )


class SelectiveAnchoredRLGuard:
    """
    A selective controller that uses a universal RL policy as an accept/reject gate
    over the candidate interventions suggested by a static group-threshold anchor.
    """

    def __init__(
        self,
        *,
        rl_model,
        threshold_baseline: GroupThresholdOptimizer,
        rl_config: RLTrainingConfig,
        guard_config: SelectiveGuardConfig,
        allow_direct_rl_override: bool = False,
    ):
        self.rl_model = rl_model
        self.threshold_baseline = threshold_baseline
        self.rl_config = rl_config
        self.guard_config = guard_config
        self.allow_direct_rl_override = allow_direct_rl_override
        self.params_: Optional[_GuardParams] = None
        self.validation_metrics_: dict = {}

    def _candidate_grid(self) -> Iterable[_GuardParams]:
        temp = SelectiveThresholdGuard(threshold_baseline=self.threshold_baseline, config=self.guard_config)
        return temp._candidate_grid()

    @staticmethod
    def _action_to_prediction(action: int, base_pred: int) -> int:
        if int(action) == 0:
            return int(base_pred)
        if int(action) == 1:
            return 0
        return 1

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "SelectiveAnchoredRLGuard":
        best_score = -1e18
        best_params: Optional[_GuardParams] = None
        best_metrics: dict = {}
        best_diagnostics: dict = {}

        for params in self._candidate_grid():
            result = self.simulate(
                y_true=y_true,
                base_preds=base_preds,
                base_scores=base_scores,
                protected=protected,
                params=params,
            )
            score = _guard_objective(result.metrics, self.guard_config)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result.metrics
                best_diagnostics = result.diagnostics

        if best_params is None:
            raise RuntimeError("SelectiveAnchoredRLGuard search did not evaluate any valid parameter sets.")
        self.params_ = best_params
        self.validation_metrics_ = {**best_metrics, **best_diagnostics, "selection_objective": best_score}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_GuardParams] = None,
    ) -> GuardSimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("Guard parameters are not fitted yet.")

        threshold_preds = self.threshold_baseline.predict(base_scores, protected).astype(int)
        tracker = RollingFairnessTracker(
            window_size=self.guard_config.window_size,
            fairness_threshold=self.guard_config.fairness_threshold,
        )

        final_preds = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)
        active_flags = np.zeros_like(base_preds, dtype=int)
        candidate_flags = (threshold_preds != base_preds).astype(int)
        accepted_flags = np.zeros_like(base_preds, dtype=int)
        rl_override_flags = np.zeros_like(base_preds, dtype=int)
        activation_events = 0
        active = False
        active_steps = 0
        cooldown_remaining = 0

        privileged_confidences: deque[float] = deque(maxlen=self.rl_config.window_size)
        unprivileged_confidences: deque[float] = deque(maxlen=self.rl_config.window_size)
        seen_group_history: deque[int] = deque(maxlen=self.rl_config.window_size)
        intervention_history: list[int] = []
        consecutive_same_group = 0
        last_group: Optional[int] = None
        protected_population_mean = float(np.mean(protected))

        for idx, (truth, base_pred, score, group, threshold_pred) in enumerate(
            zip(y_true, base_preds, base_scores, protected, threshold_preds)
        ):
            if cooldown_remaining > 0 and not active:
                cooldown_remaining -= 1

            snapshot = {"rolling_dpr": 1.0}
            if tracker.steps > 0:
                rolling = tracker._window_counts  # pylint: disable=protected-access
                privileged_rate = (
                    rolling["privileged_positive"] / rolling["privileged_n"]
                    if rolling["privileged_n"] > 0
                    else 0.5
                )
                unprivileged_rate = (
                    rolling["unprivileged_positive"] / rolling["unprivileged_n"]
                    if rolling["unprivileged_n"] > 0
                    else 0.5
                )
                if privileged_rate == 0:
                    snapshot["rolling_dpr"] = 1.0 if unprivileged_rate == 0 else 0.0
                else:
                    snapshot["rolling_dpr"] = unprivileged_rate / privileged_rate

            low_alert = snapshot["rolling_dpr"] < params.activation_dpr
            high_alert = snapshot["rolling_dpr"] > params.overshoot_activation
            within_band = params.release_dpr <= snapshot["rolling_dpr"] <= params.overshoot_release

            if idx >= params.warmup_steps and not active and cooldown_remaining == 0 and (low_alert or high_alert):
                active = True
                active_steps = 0
                activation_events += 1

            if active and (within_band or active_steps >= params.max_active_steps):
                active = False
                active_steps = 0
                cooldown_remaining = params.cooldown_steps

            if last_group == int(group):
                consecutive_same_group += 1
            else:
                consecutive_same_group = 1
            last_group = int(group)

            if int(group) == 1:
                privileged_confidences.append(float(score))
            else:
                unprivileged_confidences.append(float(score))
            seen_group_history.append(int(group))

            final_pred = int(base_pred)
            if active:
                state = build_state_from_history(
                    base_pred=int(base_pred),
                    base_prob=float(score),
                    protected_value=int(group),
                    tracker=tracker,
                    intervention_history=intervention_history,
                    seen_group_history=seen_group_history,
                    consecutive_same_group=consecutive_same_group,
                    privileged_confidences=privileged_confidences,
                    unprivileged_confidences=unprivileged_confidences,
                    protected_population_mean=protected_population_mean,
                    config=self.rl_config,
                )
                action, _ = self.rl_model.predict(state, deterministic=True)
                rl_pred = self._action_to_prediction(int(action), int(base_pred))

                if threshold_pred != base_pred:
                    if rl_pred == int(threshold_pred):
                        final_pred = int(threshold_pred)
                        accepted_flags[idx] = 1
                    elif self.allow_direct_rl_override and rl_pred != int(base_pred):
                        final_pred = int(rl_pred)
                        accepted_flags[idx] = 1
                        rl_override_flags[idx] = 1
                elif self.allow_direct_rl_override and rl_pred != int(base_pred):
                    final_pred = int(rl_pred)
                    accepted_flags[idx] = 1
                    rl_override_flags[idx] = 1

            final_preds[idx] = final_pred
            interventions[idx] = int(final_pred != base_pred)
            intervention_history.append(int(interventions[idx]))
            if len(intervention_history) > self.rl_config.window_size:
                intervention_history.pop(0)
            active_flags[idx] = int(active)

            tracker.update(int(truth), final_pred, int(group))
            if active:
                active_steps += 1

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=self.guard_config.window_size,
            fairness_threshold=self.guard_config.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=self.guard_config.fairness_threshold,
        )
        metrics.update(summarize_stream_trace(trace, warmup=self.guard_config.window_size))
        diagnostics = {
            "guard_activation_rate": float(active_flags.mean()),
            "guard_activation_events": float(activation_events),
            "guard_candidate_rate": float(candidate_flags.mean()),
            "guard_accept_rate": float(accepted_flags.mean()),
            "guard_accept_given_candidate": float(
                accepted_flags.sum() / candidate_flags.sum()
            ) if candidate_flags.sum() > 0 else 0.0,
            "guard_rl_override_rate": float(rl_override_flags.mean()),
        }
        return GuardSimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
        )


class ProjectedFairnessGuard:
    """
    A projected utility-aware controller that compares the base decision and the
    threshold-corrected candidate at each step and only intervenes when the
    fairness gain is worth the expected utility cost.
    """

    def __init__(self, *, threshold_baseline: GroupThresholdOptimizer, config: AdaptiveGuardConfig):
        self.threshold_baseline = threshold_baseline
        self.config = config
        self.params_: Optional[_AdaptiveParams] = None
        self.validation_metrics_: dict = {}

    def _candidate_grid(self) -> Iterable[_AdaptiveParams]:
        for values in product(
            self.config.warmup_grid,
            self.config.deficit_weight_grid,
            self.config.overshoot_weight_grid,
            self.config.intervention_penalty_grid,
            self.config.deficit_utility_slack_grid,
            self.config.safe_utility_slack_grid,
            self.config.safe_min_gain_grid,
            self.config.deficit_min_improvement_grid,
        ):
            yield _AdaptiveParams(*values)

    def _fairness_gain(
        self,
        *,
        base_dpr: float,
        candidate_dpr: float,
        params: _AdaptiveParams,
    ) -> float:
        base_deficit = max(0.0, self.config.fairness_threshold - base_dpr)
        candidate_deficit = max(0.0, self.config.fairness_threshold - candidate_dpr)
        base_overshoot = max(0.0, base_dpr - self.config.fairness_upper_target)
        candidate_overshoot = max(0.0, candidate_dpr - self.config.fairness_upper_target)
        deficit_gain = base_deficit - candidate_deficit
        overshoot_gain = base_overshoot - candidate_overshoot
        return params.deficit_weight * deficit_gain + params.overshoot_weight * overshoot_gain

    def _adaptive_objective(self, metrics: dict) -> float:
        return _tradeoff_objective(
            metrics,
            fairness_threshold=self.config.fairness_threshold,
            fairness_upper_target=self.config.fairness_upper_target,
            fairness_weight=self.config.objective_fairness_weight,
            overshoot_weight=self.config.objective_overshoot_weight,
            eo_weight=self.config.objective_eo_weight,
            intervention_weight=self.config.objective_intervention_weight,
        )

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "ProjectedFairnessGuard":
        best_score = -1e18
        best_params: Optional[_AdaptiveParams] = None
        best_metrics: dict = {}
        best_diagnostics: dict = {}

        for params in self._candidate_grid():
            result = self.simulate(
                y_true=y_true,
                base_preds=base_preds,
                base_scores=base_scores,
                protected=protected,
                params=params,
            )
            score = self._adaptive_objective(result.metrics)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result.metrics
                best_diagnostics = result.diagnostics

        if best_params is None:
            raise RuntimeError("ProjectedFairnessGuard search did not evaluate any valid parameter sets.")
        self.params_ = best_params
        self.validation_metrics_ = {**best_metrics, **best_diagnostics, "selection_objective": best_score}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_AdaptiveParams] = None,
    ) -> GuardSimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("Adaptive guard parameters are not fitted yet.")

        threshold_preds = self.threshold_baseline.predict(base_scores, protected).astype(int)
        decision_history: deque[tuple[int, int]] = deque()
        group_counts = _blank_group_counts()

        final_preds = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)
        candidate_flags = (threshold_preds != base_preds).astype(int)
        accepted_flags = np.zeros_like(base_preds, dtype=int)
        pressure_flags = np.zeros_like(base_preds, dtype=int)
        fairness_gain_values: list[float] = []

        for idx, (base_pred, score, group, threshold_pred) in enumerate(
            zip(base_preds, base_scores, protected, threshold_preds)
        ):
            base_pred = int(base_pred)
            threshold_pred = int(threshold_pred)
            group = int(group)
            projected_base_counts = _project_group_counts(
                decision_history,
                group_counts,
                window_size=self.config.window_size,
                proposed_pred=base_pred,
                protected_value=group,
            )
            projected_base_dpr = _dpr_from_group_counts(projected_base_counts)

            final_pred = base_pred
            if idx >= params.warmup_steps and threshold_pred != base_pred:
                projected_candidate_counts = _project_group_counts(
                    decision_history,
                    group_counts,
                    window_size=self.config.window_size,
                    proposed_pred=threshold_pred,
                    protected_value=group,
                )
                projected_candidate_dpr = _dpr_from_group_counts(projected_candidate_counts)
                utility_delta = (
                    _expected_correctness(threshold_pred, float(score))
                    - _expected_correctness(base_pred, float(score))
                )
                projected_dpr_delta = projected_candidate_dpr - projected_base_dpr
                fairness_gain = self._fairness_gain(
                    base_dpr=projected_base_dpr,
                    candidate_dpr=projected_candidate_dpr,
                    params=params,
                )
                total_gain = fairness_gain + utility_delta - params.intervention_penalty
                in_deficit = projected_base_dpr < self.config.fairness_threshold
                in_overshoot = projected_base_dpr > self.config.fairness_upper_target
                if fairness_gain > 0:
                    pressure_flags[idx] = 1
                if in_deficit:
                    dynamic_slack = params.deficit_utility_slack * (
                        1.0 + 2.0 * (self.config.fairness_threshold - projected_base_dpr)
                    )
                    should_accept = (
                        projected_candidate_dpr > projected_base_dpr + params.deficit_min_improvement
                        and total_gain >= -dynamic_slack
                    )
                elif in_overshoot:
                    should_accept = (
                        projected_candidate_dpr < projected_base_dpr
                        and total_gain >= -params.safe_utility_slack
                    )
                else:
                    near_upper_band = projected_base_dpr >= (self.config.fairness_upper_target - 0.01)
                    should_accept = (
                        near_upper_band
                        and projected_candidate_dpr < projected_base_dpr
                        and total_gain >= params.safe_min_gain
                    )

                if should_accept:
                    final_pred = threshold_pred
                    accepted_flags[idx] = 1
                    fairness_gain_values.append(projected_dpr_delta)

            final_preds[idx] = final_pred
            interventions[idx] = int(final_pred != base_pred)

            if len(decision_history) >= self.config.window_size and decision_history:
                oldest_pred, oldest_group = decision_history.popleft()
                _apply_group_event(group_counts, oldest_pred, oldest_group, -1.0)
            decision_history.append((int(final_pred), group))
            _apply_group_event(group_counts, int(final_pred), group, 1.0)

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics.update(summarize_stream_trace(trace, warmup=self.config.window_size))
        activation_events = 0
        active_prev = 0
        for accepted in accepted_flags:
            if int(accepted) == 1 and active_prev == 0:
                activation_events += 1
            active_prev = int(accepted)
        diagnostics = {
            "guard_activation_rate": float(np.mean(pressure_flags)),
            "guard_activation_events": float(activation_events),
            "guard_candidate_rate": float(np.mean(candidate_flags)),
            "guard_accept_rate": float(np.mean(accepted_flags)),
            "guard_accept_given_candidate": float(
                accepted_flags.sum() / candidate_flags.sum()
            ) if candidate_flags.sum() > 0 else 0.0,
            "guard_avg_projected_dpr_gain": float(np.mean(fairness_gain_values)) if fairness_gain_values else 0.0,
        }
        return GuardSimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
        )


class PrimalDualFairnessController:
    """
    A selective controller that treats fairness deficit as a dual variable.

    The controller observes the current rolling DPR, updates a non-negative
    Lagrange multiplier, and accepts threshold-based fairness corrections only
    when the dual-weighted projected fairness gain outweighs the expected
    accuracy cost plus a small intervention penalty.
    """

    def __init__(self, *, threshold_baseline: GroupThresholdOptimizer, config: PrimalDualGuardConfig):
        self.threshold_baseline = threshold_baseline
        self.config = config
        self.params_: Optional[_PrimalDualParams] = None
        self.validation_metrics_: dict = {}

    def _candidate_grid(self) -> Iterable[_PrimalDualParams]:
        for values in product(
            self.config.warmup_grid,
            self.config.eta_grid,
            self.config.intervention_penalty_grid,
            self.config.lambda_cap_grid,
            self.config.min_projected_gain_grid,
        ):
            yield _PrimalDualParams(*values)

    def _objective(self, metrics: dict) -> float:
        return _tradeoff_objective(
            metrics,
            fairness_threshold=self.config.fairness_threshold,
            fairness_upper_target=self.config.fairness_upper_target,
            fairness_weight=self.config.objective_fairness_weight,
            overshoot_weight=self.config.objective_overshoot_weight,
            eo_weight=self.config.objective_eo_weight,
            intervention_weight=self.config.objective_intervention_weight,
        )

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "PrimalDualFairnessController":
        best_score = -1e18
        best_params: Optional[_PrimalDualParams] = None
        best_metrics: dict = {}
        best_diagnostics: dict = {}

        for params in self._candidate_grid():
            result = self.simulate(
                y_true=y_true,
                base_preds=base_preds,
                base_scores=base_scores,
                protected=protected,
                params=params,
            )
            score = self._objective(result.metrics)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result.metrics
                best_diagnostics = result.diagnostics

        if best_params is None:
            raise RuntimeError("PrimalDualFairnessController search did not evaluate any valid parameter sets.")
        self.params_ = best_params
        self.validation_metrics_ = {**best_metrics, **best_diagnostics, "selection_objective": best_score}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_PrimalDualParams] = None,
    ) -> GuardSimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("Primal-dual guard parameters are not fitted yet.")

        threshold_preds = self.threshold_baseline.predict(base_scores, protected).astype(int)
        decision_history: deque[tuple[int, int]] = deque()
        group_counts = _blank_group_counts()

        final_preds = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)
        candidate_flags = (threshold_preds != base_preds).astype(int)
        accepted_flags = np.zeros_like(base_preds, dtype=int)
        active_flags = np.zeros_like(base_preds, dtype=int)
        lambda_history: list[float] = []
        fairness_gain_values: list[float] = []
        lambda_value = 0.0
        activation_events = 0
        active_prev = 0

        for idx, (base_pred, score, group, threshold_pred) in enumerate(
            zip(base_preds, base_scores, protected, threshold_preds)
        ):
            base_pred = int(base_pred)
            threshold_pred = int(threshold_pred)
            group = int(group)
            current_dpr = _dpr_from_group_counts(group_counts)

            if idx >= params.warmup_steps:
                lambda_value = max(
                    0.0,
                    lambda_value + params.eta * (self.config.fairness_threshold - current_dpr),
                )
                lambda_value = min(lambda_value, params.lambda_cap)

            is_active = int(lambda_value > 0.0)
            active_flags[idx] = is_active
            lambda_history.append(lambda_value)
            if is_active == 1 and active_prev == 0:
                activation_events += 1
            active_prev = is_active

            final_pred = base_pred
            if idx >= params.warmup_steps and threshold_pred != base_pred:
                projected_base_counts = _project_group_counts(
                    decision_history,
                    group_counts,
                    window_size=self.config.window_size,
                    proposed_pred=base_pred,
                    protected_value=group,
                )
                projected_candidate_counts = _project_group_counts(
                    decision_history,
                    group_counts,
                    window_size=self.config.window_size,
                    proposed_pred=threshold_pred,
                    protected_value=group,
                )
                projected_base_dpr = _dpr_from_group_counts(projected_base_counts)
                projected_candidate_dpr = _dpr_from_group_counts(projected_candidate_counts)
                projected_dpr_gain = max(0.0, projected_candidate_dpr - projected_base_dpr)
                utility_delta = (
                    _expected_correctness(threshold_pred, float(score))
                    - _expected_correctness(base_pred, float(score))
                )
                accuracy_cost = max(0.0, -utility_delta)
                fairness_value = lambda_value * projected_dpr_gain
                should_accept = (
                    projected_dpr_gain > params.min_projected_gain
                    and fairness_value >= accuracy_cost + params.intervention_penalty
                )
                if should_accept:
                    final_pred = threshold_pred
                    accepted_flags[idx] = 1
                    fairness_gain_values.append(projected_dpr_gain)

            final_preds[idx] = final_pred
            interventions[idx] = int(final_pred != base_pred)

            if len(decision_history) >= self.config.window_size and decision_history:
                oldest_pred, oldest_group = decision_history.popleft()
                _apply_group_event(group_counts, oldest_pred, oldest_group, -1.0)
            decision_history.append((int(final_pred), group))
            _apply_group_event(group_counts, int(final_pred), group, 1.0)

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics.update(summarize_stream_trace(trace, warmup=self.config.window_size))
        diagnostics = {
            "guard_activation_rate": float(np.mean(active_flags)),
            "guard_activation_events": float(activation_events),
            "guard_candidate_rate": float(np.mean(candidate_flags)),
            "guard_accept_rate": float(np.mean(accepted_flags)),
            "guard_accept_given_candidate": float(
                accepted_flags.sum() / candidate_flags.sum()
            ) if candidate_flags.sum() > 0 else 0.0,
            "guard_avg_projected_dpr_gain": float(np.mean(fairness_gain_values)) if fairness_gain_values else 0.0,
            "dual_lambda_mean": float(np.mean(lambda_history)) if lambda_history else 0.0,
            "dual_lambda_max": float(np.max(lambda_history)) if lambda_history else 0.0,
            "dual_lambda_final": float(lambda_history[-1]) if lambda_history else 0.0,
        }
        return GuardSimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
        )


class PrimalDualOffsetController:
    """
    Per-group primal-dual controller with learned threshold offsets.

    Each protected group maintains:
    - a dual variable that grows when the group becomes approval-deprived, and
    - a learned threshold offset that creates a richer intervention space than a
      single static threshold-corrected candidate.
    """

    def __init__(
        self,
        *,
        threshold_baseline: Optional[GroupThresholdOptimizer] = None,
        config: PrimalDualOffsetConfig,
    ):
        self.threshold_baseline = threshold_baseline
        self.config = config
        self.params_: Optional[_PrimalDualOffsetParams] = None
        self.validation_metrics_: dict = {}

    def _candidate_grid(self) -> Iterable[_PrimalDualOffsetParams]:
        for values in product(self.config.lambda_lr_grid, self.config.delta_lr_grid):
            yield _PrimalDualOffsetParams(*values)

    def _offset_objective(self, metrics: dict) -> float:
        return _tradeoff_objective(
            metrics,
            fairness_threshold=self.config.target_dpr,
            fairness_upper_target=self.config.fairness_upper_target,
            fairness_weight=self.config.objective_fairness_weight,
            overshoot_weight=self.config.objective_overshoot_weight,
            eo_weight=self.config.objective_eo_weight,
            intervention_weight=self.config.objective_intervention_weight,
        )

    @staticmethod
    def _build_candidate_specs(
        *,
        base_pred: int,
        score: float,
        delta: float,
        config: PrimalDualOffsetConfig,
    ) -> list[tuple[int, float, float]]:
        clipped_delta = min(max(delta, 0.0), config.max_delta)
        aggressive_delta = min(config.max_delta, clipped_delta * config.aggressive_multiplier)
        return [
            (int(base_pred), 0.0, 0.0),
            (int(score >= (0.5 - clipped_delta)), clipped_delta, config.mild_offset_base_cost),
            (int(score >= (0.5 - aggressive_delta)), aggressive_delta, config.aggressive_offset_base_cost),
        ]

    @staticmethod
    def _projected_dpr_gain(
        *,
        decision_buffers: dict[int, deque[int]],
        current_group: int,
        candidate_pred: int,
    ) -> float:
        current_rates = _approval_rates_from_buffers(decision_buffers)
        projected_buffers = {
            int(group): deque(buffer, maxlen=buffer.maxlen)
            for group, buffer in decision_buffers.items()
        }
        if current_group not in projected_buffers:
            projected_buffers[current_group] = deque(maxlen=next(iter(decision_buffers.values())).maxlen)
        projected_buffers[current_group].append(int(candidate_pred))
        projected_rates = _approval_rates_from_buffers(projected_buffers)
        return _dpr_from_rates(projected_rates) - _dpr_from_rates(current_rates)

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "PrimalDualOffsetController":
        best_score = -1e18
        best_params: Optional[_PrimalDualOffsetParams] = None
        best_metrics: dict = {}
        best_diagnostics: dict = {}

        for params in self._candidate_grid():
            result = self.simulate(
                y_true=y_true,
                base_preds=base_preds,
                base_scores=base_scores,
                protected=protected,
                params=params,
            )
            score = self._offset_objective(result.metrics)
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = result.metrics
                best_diagnostics = result.diagnostics

        if best_params is None:
            raise RuntimeError("PrimalDualOffsetController search did not evaluate any parameter sets.")
        self.params_ = best_params
        self.validation_metrics_ = {**best_metrics, **best_diagnostics, "selection_objective": best_score}
        return self

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_PrimalDualOffsetParams] = None,
    ) -> GuardSimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("Primal-dual offset parameters are not fitted yet.")

        groups = tuple(sorted(int(group) for group in np.unique(protected)))
        if not groups:
            groups = (0, 1)
        decision_buffers = {group: deque(maxlen=self.config.window_size) for group in groups}
        outcome_buffers = {group: deque(maxlen=self.config.window_size) for group in groups}
        lambda_g = {group: 0.0 for group in groups}
        delta_g = {group: 0.0 for group in groups}

        final_preds = np.zeros_like(base_preds, dtype=int)
        interventions = np.zeros_like(base_preds, dtype=int)
        candidate_flags = np.zeros_like(base_preds, dtype=int)
        accepted_flags = np.zeros_like(base_preds, dtype=int)
        active_flags = np.zeros_like(base_preds, dtype=int)
        projected_gain_values: list[float] = []
        lambda_mean_history: list[float] = []
        lambda_max_history: list[float] = []
        delta_mean_history: list[float] = []
        delta_max_history: list[float] = []
        activation_events = 0
        active_prev = 0

        for idx, (truth, base_pred, score, group) in enumerate(zip(y_true, base_preds, base_scores, protected)):
            base_pred = int(base_pred)
            group = int(group)
            score = float(score)
            current_active = int(max(lambda_g.values()) > 1e-8 or max(delta_g.values()) > 1e-8)
            active_flags[idx] = current_active
            if current_active == 1 and active_prev == 0:
                activation_events += 1
            active_prev = current_active

            candidate_specs = self._build_candidate_specs(
                base_pred=base_pred,
                score=score,
                delta=delta_g.get(group, 0.0),
                config=self.config,
            )
            candidate_flags[idx] = int(any(pred != base_pred for pred, _, _ in candidate_specs[1:]))

            best_pred = base_pred
            best_score = 0.0
            best_projected_gain = 0.0

            if idx >= self.config.warmup_steps and candidate_flags[idx] == 1:
                current_lambda = lambda_g.get(group, 0.0)
                base_correctness = _expected_correctness(base_pred, score)
                for candidate_pred, offset_used, base_cost in candidate_specs[1:]:
                    candidate_pred = int(candidate_pred)
                    if candidate_pred == base_pred:
                        continue
                    projected_gain = self._projected_dpr_gain(
                        decision_buffers=decision_buffers,
                        current_group=group,
                        candidate_pred=candidate_pred,
                    )
                    if projected_gain <= 0.0:
                        continue
                    candidate_correctness = _expected_correctness(candidate_pred, score)
                    accuracy_cost = max(0.0, base_correctness - candidate_correctness)
                    accuracy_cost += float(base_cost) + abs(float(offset_used)) * self.config.offset_cost_scale
                    if accuracy_cost > self.config.utility_slack:
                        continue
                    fairness_value = current_lambda * projected_gain
                    candidate_score = fairness_value - accuracy_cost
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pred = candidate_pred
                        best_projected_gain = projected_gain

            final_preds[idx] = best_pred
            interventions[idx] = int(best_pred != base_pred)
            accepted_flags[idx] = interventions[idx]
            if interventions[idx] == 1:
                projected_gain_values.append(best_projected_gain)

            if group not in decision_buffers:
                decision_buffers[group] = deque(maxlen=self.config.window_size)
                outcome_buffers[group] = deque(maxlen=self.config.window_size)
                lambda_g[group] = 0.0
                delta_g[group] = 0.0
            decision_buffers[group].append(int(best_pred))
            outcome_buffers[group].append(int(truth))

            approval_rates = _approval_rates_from_buffers(decision_buffers)
            current_dpr = _dpr_from_rates(approval_rates)
            max_rate = max(approval_rates.values()) if approval_rates else 0.5
            group_rate = approval_rates.get(group, 0.5)
            is_deprived = group_rate < (max_rate - self.config.group_tolerance)
            dpr_deficit = max(0.0, self.config.target_dpr - current_dpr)

            for other_group in lambda_g:
                if other_group == group and idx >= self.config.warmup_steps:
                    if is_deprived:
                        lambda_g[other_group] = min(
                            self.config.lambda_cap,
                            max(0.0, lambda_g[other_group] + params.lambda_lr * dpr_deficit),
                        )
                    else:
                        lambda_g[other_group] = max(0.0, lambda_g[other_group] * self.config.lambda_decay)
                else:
                    lambda_g[other_group] = max(0.0, lambda_g[other_group] * self.config.lambda_decay)

            for other_group in delta_g:
                if other_group == group and idx >= self.config.warmup_steps:
                    if current_dpr < self.config.target_dpr and is_deprived:
                        delta_g[other_group] = min(
                            self.config.max_delta,
                            delta_g[other_group] + params.delta_lr * lambda_g[other_group],
                        )
                    else:
                        delta_g[other_group] = max(0.0, delta_g[other_group] * self.config.delta_decay)
                else:
                    delta_g[other_group] = max(0.0, delta_g[other_group] * self.config.delta_decay)

            lambda_mean_history.append(float(np.mean(list(lambda_g.values()))))
            lambda_max_history.append(float(np.max(list(lambda_g.values()))))
            delta_mean_history.append(float(np.mean(list(delta_g.values()))))
            delta_max_history.append(float(np.max(list(delta_g.values()))))

        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=self.config.fairness_threshold,
        )
        metrics.update(summarize_stream_trace(trace, warmup=self.config.window_size))
        diagnostics = {
            "guard_activation_rate": float(np.mean(active_flags)),
            "guard_activation_events": float(activation_events),
            "guard_candidate_rate": float(np.mean(candidate_flags)),
            "guard_accept_rate": float(np.mean(accepted_flags)),
            "guard_accept_given_candidate": float(
                accepted_flags.sum() / candidate_flags.sum()
            ) if candidate_flags.sum() > 0 else 0.0,
            "guard_avg_projected_dpr_gain": float(np.mean(projected_gain_values)) if projected_gain_values else 0.0,
            "dual_lambda_mean": float(np.mean(lambda_mean_history)) if lambda_mean_history else 0.0,
            "dual_lambda_max": float(np.max(lambda_max_history)) if lambda_max_history else 0.0,
            "dual_lambda_final": float(lambda_mean_history[-1]) if lambda_mean_history else 0.0,
            "offset_delta_mean": float(np.mean(delta_mean_history)) if delta_mean_history else 0.0,
            "offset_delta_max": float(np.max(delta_max_history)) if delta_max_history else 0.0,
            "offset_delta_final": float(delta_mean_history[-1]) if delta_mean_history else 0.0,
        }
        return GuardSimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
        )
