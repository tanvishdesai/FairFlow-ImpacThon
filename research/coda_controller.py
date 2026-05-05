"""
CODA v2: Continuous Online Dual-Adaptive Thresholding.

A fully online per-group threshold controller that continuously adjusts
decision thresholds via a primal-dual gradient scheme.  Unlike the guard
methods that gate a static `group_threshold` baseline, CODA **is** the
thresholding itself — it learns per-group thresholds from the stream.

v2 improvements over v1
-----------------------
- Starts from GroupThresholdOptimizer's calibrated thresholds (not naive 0.5)
- Much tighter deviation/step bounds to prevent accuracy collapse
- Accuracy-aware damping: slows adaptation when rolling accuracy drops
- Group-aware warmup: waits for samples from BOTH groups before adapting
- Lower λ cap and stronger EMA smoothing options
- Earlier anti-overshoot activation (1.02 instead of 1.10)
- Asymmetric update: advantaged group threshold moves less
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from research.metrics import (
    build_stream_trace,
    compute_static_metrics,
    summarize_stream_trace,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CODAConfig:
    """All tuneable knobs for the CODA controller."""

    # ----- Fairness targets -----
    fairness_threshold: float = 0.8
    target_dpr: float = 0.88          # v2: lowered from 0.90 to reduce overshoot
    fairness_upper_target: float = 1.02   # v2: lowered from 1.10 — anti-overshoot activates earlier

    window_size: int = 50

    # ----- Grid parameters (searched during fit) -----
    warmup_grid: tuple[int, ...] = (15, 30, 50)
    eta_grid: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)          # v2: much lower range
    threshold_lr_grid: tuple[float, ...] = (0.003, 0.008, 0.015, 0.03)  # v2: much lower
    ema_alpha_grid: tuple[float, ...] = (0.10, 0.20, 0.35)           # v2: heavier smoothing

    # ----- Fixed hyper-parameters -----
    lambda_cap: float = 5.0           # v2: lowered from 20.0 to prevent runaway
    lambda_decay: float = 0.990       # v2: faster decay when constraint satisfied
    max_step: float = 0.015           # v2: halved from 0.04 — max per-step change
    max_deviation: float = 0.12       # v2: halved from 0.30 — max total drift
    min_threshold: float = 0.05
    max_threshold: float = 0.95
    advantaged_scale: float = 0.3     # v2: NEW — advantaged group moves at 30% of disadvantaged

    # ----- Accuracy safeguard -----
    accuracy_floor: float = 0.02      # v2: NEW — max allowed accuracy drop vs base model
    accuracy_damping: float = 0.1     # v2: NEW — when accuracy drops, multiply step by this

    # ----- Group-aware warmup -----
    min_group_samples: int = 10       # v2: NEW — require this many from EACH group

    # ----- Rate gap minimum for activation -----
    min_rate_gap: float = 0.03        # v2: raised from 0.01 — less twitchy

    # ----- Objective weights (for grid selection) -----
    objective_fairness_weight: float = 2.0     # v2: lowered from 3.0
    objective_overshoot_weight: float = 1.5    # v2: raised from 0.35 — penalise overshoot harder
    objective_eo_weight: float = 0.30
    objective_intervention_weight: float = 0.5  # v2: raised from 0.15
    objective_accuracy_weight: float = 1.5      # v2: NEW — penalise accuracy loss

    tag: str = "coda_v2"


# ---------------------------------------------------------------------------
# Internal parameter container (one point in the search grid)
# ---------------------------------------------------------------------------

@dataclass
class _CODAParams:
    warmup_steps: int
    eta: float
    threshold_lr: float
    ema_alpha: float


# ---------------------------------------------------------------------------
# Simulation result (same shape as GuardSimulationResult for compatibility)
# ---------------------------------------------------------------------------

@dataclass
class CODASimulationResult:
    predictions: np.ndarray
    interventions: np.ndarray
    trace: pd.DataFrame
    metrics: dict
    diagnostics: dict
    threshold_history: dict[int, list[float]]


# ---------------------------------------------------------------------------
# Helper: trade-off objective used for grid selection (v2: includes accuracy)
# ---------------------------------------------------------------------------

def _tradeoff_objective(
    metrics: dict,
    *,
    fairness_threshold: float,
    fairness_upper_target: float,
    fairness_weight: float,
    overshoot_weight: float,
    eo_weight: float,
    intervention_weight: float,
    accuracy_weight: float,
    base_accuracy: float,
) -> float:
    dpr = float(metrics["demographic_parity_ratio"])
    fairness_shortfall = max(0.0, fairness_threshold - dpr)
    overshoot = max(0.0, dpr - fairness_upper_target)
    accuracy_loss = max(0.0, base_accuracy - float(metrics["accuracy"]))
    return (
        float(metrics["accuracy"])
        - fairness_weight * fairness_shortfall
        - overshoot_weight * overshoot
        - eo_weight * float(metrics["equalized_odds_gap"])
        - intervention_weight * float(metrics.get("intervention_rate", 0.0))
        - accuracy_weight * accuracy_loss
    )


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class CODAController:
    """
    Continuous Online Dual-Adaptive Thresholding controller (v2).

    CODA maintains per-group decision thresholds and adapts them at every
    timestep via a primal-dual gradient scheme.

    v2 key improvements:
    - Starts from GroupThresholdOptimizer's calibrated thresholds
    - Accuracy-aware damping prevents accuracy collapse
    - Group-aware warmup prevents blind adaptation
    - Much tighter bounds prevent threshold runaway

    Parameters
    ----------
    config : CODAConfig
        All tuneable knobs.
    initial_thresholds : dict[int, float] | None
        Starting thresholds per group (group_id -> threshold).
        If None, both start at 0.5.
    """

    def __init__(
        self,
        *,
        config: CODAConfig,
        initial_thresholds: Optional[dict[int, float]] = None,
    ):
        self.config = config
        self.initial_thresholds = initial_thresholds or {}
        self.params_: Optional[_CODAParams] = None
        self.validation_metrics_: dict = {}
        self._base_accuracy: float = 0.0  # set during fit

    # ----- grid enumeration -----

    def _candidate_grid(self) -> Iterable[_CODAParams]:
        for values in product(
            self.config.warmup_grid,
            self.config.eta_grid,
            self.config.threshold_lr_grid,
            self.config.ema_alpha_grid,
        ):
            yield _CODAParams(*values)

    # ----- objective -----

    def _objective(self, metrics: dict) -> float:
        return _tradeoff_objective(
            metrics,
            fairness_threshold=self.config.target_dpr,
            fairness_upper_target=self.config.fairness_upper_target,
            fairness_weight=self.config.objective_fairness_weight,
            overshoot_weight=self.config.objective_overshoot_weight,
            eo_weight=self.config.objective_eo_weight,
            intervention_weight=self.config.objective_intervention_weight,
            accuracy_weight=self.config.objective_accuracy_weight,
            base_accuracy=self._base_accuracy,
        )

    # ----- fit (grid search on validation data) -----

    def fit(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
    ) -> "CODAController":
        # v2: compute base model accuracy for accuracy-aware objective
        self._base_accuracy = float(np.mean(y_true == base_preds))

        best_score = -1e18
        best_params: Optional[_CODAParams] = None
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
            raise RuntimeError(
                "CODAController grid search did not evaluate any parameter sets."
            )
        self.params_ = best_params
        self.validation_metrics_ = {
            **best_metrics,
            **best_diagnostics,
            "selection_objective": best_score,
            "selected_warmup": best_params.warmup_steps,
            "selected_eta": best_params.eta,
            "selected_threshold_lr": best_params.threshold_lr,
            "selected_ema_alpha": best_params.ema_alpha,
        }
        return self

    # ----- simulate (the core online loop) — v2 rewrite -----

    def simulate(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        base_scores: np.ndarray,
        protected: np.ndarray,
        params: Optional[_CODAParams] = None,
    ) -> CODASimulationResult:
        params = self.params_ if params is None else params
        if params is None:
            raise ValueError("CODA parameters are not fitted yet.")

        cfg = self.config
        n = len(y_true)
        groups = sorted(set(int(g) for g in np.unique(protected)))
        if not groups:
            groups = [0, 1]

        # --- Initialise per-group thresholds ---
        tau = {
            g: float(self.initial_thresholds.get(g, 0.5))
            for g in groups
        }
        tau_initial = {g: tau[g] for g in groups}

        # EMA-smoothed threshold (starts at same value)
        tau_ema = {g: tau[g] for g in groups}

        # Dual variable (non-negative)
        lambda_val = 0.0

        # Rolling window buffers per group (for DPR tracking)
        decision_buffers: dict[int, deque[int]] = {
            g: deque(maxlen=cfg.window_size) for g in groups
        }

        # v2: Rolling accuracy tracker
        accuracy_buffer: deque[int] = deque(maxlen=cfg.window_size)

        # v2: Per-group sample counters (for group-aware warmup)
        group_sample_counts: dict[int, int] = {g: 0 for g in groups}

        # Output arrays
        final_preds = np.zeros(n, dtype=int)
        interventions = np.zeros(n, dtype=int)

        # Diagnostics
        lambda_history: list[float] = []
        threshold_history: dict[int, list[float]] = {g: [] for g in groups}
        dpr_history: list[float] = []
        damping_events = 0
        threshold_updates_count = 0

        for idx in range(n):
            base_pred = int(base_preds[idx])
            score = float(base_scores[idx])
            group = int(protected[idx])
            truth = int(y_true[idx])

            # --- 1. Make decision using current group threshold ---
            current_tau = tau_ema.get(group, 0.5)
            coda_decision = int(score >= current_tau)

            final_preds[idx] = coda_decision
            interventions[idx] = int(coda_decision != base_pred)

            # --- 2. Update buffers ---
            if group not in decision_buffers:
                decision_buffers[group] = deque(maxlen=cfg.window_size)
            decision_buffers[group].append(coda_decision)
            group_sample_counts[group] = group_sample_counts.get(group, 0) + 1

            # v2: track rolling accuracy
            accuracy_buffer.append(int(coda_decision == truth))

            # --- 3. Compute rolling DPR ---
            approval_rates = {}
            for g in groups:
                buf = decision_buffers.get(g)
                if buf and len(buf) > 0:
                    approval_rates[g] = float(np.mean(list(buf)))
                else:
                    approval_rates[g] = 0.5

            max_rate = max(approval_rates.values())
            min_rate = min(approval_rates.values())
            rolling_dpr = (
                float(min_rate / (max_rate + 1e-9))
                if max_rate > 0
                else 1.0
            )
            dpr_history.append(rolling_dpr)

            # --- v2: Check group-aware warmup ---
            all_groups_ready = all(
                group_sample_counts.get(g, 0) >= cfg.min_group_samples
                for g in groups
            )
            past_warmup = idx >= params.warmup_steps and all_groups_ready

            # --- 4. Update dual variable ---
            if past_warmup:
                # Dual ascent: λ grows when DPR is below target
                constraint_violation = cfg.target_dpr - rolling_dpr
                lambda_val = max(
                    0.0,
                    lambda_val + params.eta * constraint_violation,
                )
                lambda_val = min(lambda_val, cfg.lambda_cap)

                # Decay λ when constraint is satisfied
                if rolling_dpr >= cfg.target_dpr:
                    lambda_val *= cfg.lambda_decay

            lambda_history.append(lambda_val)

            # --- 5. Update per-group thresholds (primal step) ---
            if past_warmup and lambda_val > 1e-8:
                # Identify disadvantaged/advantaged groups
                min_rate_group = min(approval_rates, key=approval_rates.get)
                max_rate_group = max(approval_rates, key=approval_rates.get)

                rate_gap = approval_rates[max_rate_group] - approval_rates[min_rate_group]

                if rate_gap > cfg.min_rate_gap:
                    # Scale the step by the dual variable and rate gap
                    step_magnitude = params.threshold_lr * lambda_val * rate_gap

                    # v2: Accuracy-aware damping
                    if len(accuracy_buffer) >= cfg.window_size // 2:
                        rolling_accuracy = float(np.mean(list(accuracy_buffer)))
                        accuracy_drop = self._base_accuracy - rolling_accuracy
                        if accuracy_drop > cfg.accuracy_floor:
                            step_magnitude *= cfg.accuracy_damping
                            damping_events += 1

                    # Clamp per-step change
                    step_magnitude = min(step_magnitude, cfg.max_step)

                    # Lower threshold for disadvantaged group (easier to approve)
                    new_tau_min = tau[min_rate_group] - step_magnitude
                    # v2: Advantaged group threshold moves MUCH less
                    new_tau_max = tau[max_rate_group] + step_magnitude * cfg.advantaged_scale

                    # --- Apply deviation bounds ---
                    new_tau_min = max(
                        new_tau_min,
                        tau_initial[min_rate_group] - cfg.max_deviation,
                    )
                    new_tau_min = max(cfg.min_threshold, new_tau_min)

                    new_tau_max = min(
                        new_tau_max,
                        tau_initial[max_rate_group] + cfg.max_deviation,
                    )
                    new_tau_max = min(cfg.max_threshold, new_tau_max)

                    tau[min_rate_group] = new_tau_min
                    tau[max_rate_group] = new_tau_max
                    threshold_updates_count += 1

                # v2: Anti-overshoot with tighter target
                if rolling_dpr > cfg.fairness_upper_target:
                    overshoot = rolling_dpr - cfg.fairness_upper_target
                    # Use a stronger correction for overshoot
                    overshoot_step = params.threshold_lr * overshoot * 2.0
                    overshoot_step = min(overshoot_step, cfg.max_step)

                    # Reverse: raise threshold for disadvantaged (was lowered)
                    rev_tau_min = tau[min_rate_group] + overshoot_step
                    rev_tau_min = min(rev_tau_min, tau_initial[min_rate_group] + cfg.max_deviation)
                    rev_tau_min = min(cfg.max_threshold, rev_tau_min)

                    # Reverse: lower threshold for advantaged (was raised)
                    rev_tau_max = tau[max_rate_group] - overshoot_step * cfg.advantaged_scale
                    rev_tau_max = max(rev_tau_max, tau_initial[max_rate_group] - cfg.max_deviation)
                    rev_tau_max = max(cfg.min_threshold, rev_tau_max)

                    tau[min_rate_group] = rev_tau_min
                    tau[max_rate_group] = rev_tau_max

            # --- 6. EMA smooth the thresholds ---
            for g in groups:
                tau_ema[g] = (
                    params.ema_alpha * tau[g]
                    + (1.0 - params.ema_alpha) * tau_ema[g]
                )

            # --- Record ---
            for g in groups:
                threshold_history[g].append(tau_ema[g])

        # --- Build trace and metrics ---
        trace = build_stream_trace(
            y_true=y_true,
            y_pred=final_preds,
            protected=protected,
            window_size=cfg.window_size,
            fairness_threshold=cfg.fairness_threshold,
        )
        metrics = compute_static_metrics(
            y_true,
            final_preds,
            protected,
            scores=base_scores,
            intervention_flags=interventions,
            fairness_threshold=cfg.fairness_threshold,
        )
        metrics.update(
            summarize_stream_trace(trace, warmup=cfg.window_size)
        )

        diagnostics = {
            "coda_threshold_updates": float(threshold_updates_count),
            "coda_damping_events": float(damping_events),
            "coda_lambda_mean": float(np.mean(lambda_history))
            if lambda_history
            else 0.0,
            "coda_lambda_max": float(np.max(lambda_history))
            if lambda_history
            else 0.0,
            "coda_lambda_final": float(lambda_history[-1])
            if lambda_history
            else 0.0,
            "coda_dpr_mean": float(np.mean(dpr_history))
            if dpr_history
            else 1.0,
            "coda_dpr_std": float(np.std(dpr_history))
            if dpr_history
            else 0.0,
        }
        for g in groups:
            if threshold_history[g]:
                diagnostics[f"coda_tau_initial_group_{g}"] = float(tau_initial[g])
                diagnostics[f"coda_tau_final_group_{g}"] = float(
                    threshold_history[g][-1]
                )
                diagnostics[f"coda_tau_mean_group_{g}"] = float(
                    np.mean(threshold_history[g])
                )
                diagnostics[f"coda_tau_range_group_{g}"] = float(
                    np.max(threshold_history[g])
                    - np.min(threshold_history[g])
                )

        return CODASimulationResult(
            predictions=final_preds,
            interventions=interventions,
            trace=trace,
            metrics=metrics,
            diagnostics=diagnostics,
            threshold_history=threshold_history,
        )


# ---------------------------------------------------------------------------
# Convenience: create a CODA controller seeded from GroupThresholdOptimizer
# ---------------------------------------------------------------------------

def build_coda_with_calibrated_init(
    *,
    config: CODAConfig,
    threshold_baseline,
    val_scores: np.ndarray,
    val_protected: np.ndarray,
    val_y_true: np.ndarray,
) -> CODAController:
    """
    Create a CODA controller whose initial per-group thresholds come from
    the fitted GroupThresholdOptimizer.

    v2: This is the recommended way to initialise CODA.  Starting from the
    same thresholds as group_threshold means CODA begins at a known-good
    operating point and then makes *conservative* online adjustments.

    Parameters
    ----------
    config : CODAConfig
        CODA configuration.
    threshold_baseline : GroupThresholdOptimizer
        A fitted GroupThresholdOptimizer that provides calibrated
        per-group thresholds.
    val_scores : np.ndarray
        Validation set scores (used only for fallback calibration).
    val_protected : np.ndarray
        Validation set protected attributes.
    val_y_true : np.ndarray
        Validation set true labels.
    """
    # v2: Use GroupThresholdOptimizer's calibrated thresholds directly
    if (
        hasattr(threshold_baseline, "threshold_privileged")
        and hasattr(threshold_baseline, "threshold_unprivileged")
    ):
        initial_thresholds = {
            1: float(threshold_baseline.threshold_privileged),   # privileged
            0: float(threshold_baseline.threshold_unprivileged), # unprivileged
        }
    else:
        # Fallback: calibrate from validation data
        groups = sorted(set(int(g) for g in np.unique(val_protected)))
        overall_pos_rate = float(np.mean(val_y_true))
        initial_thresholds = {}
        for g in groups:
            mask = val_protected == g
            group_scores = val_scores[mask]
            if len(group_scores) == 0:
                initial_thresholds[g] = 0.5
                continue
            best_tau = 0.5
            best_diff = 1e18
            for tau_candidate in np.linspace(0.1, 0.9, 17):
                pos_rate = float(np.mean(group_scores >= tau_candidate))
                diff = abs(pos_rate - overall_pos_rate)
                if diff < best_diff:
                    best_diff = diff
                    best_tau = float(tau_candidate)
            initial_thresholds[g] = best_tau

    return CODAController(
        config=config,
        initial_thresholds=initial_thresholds,
    )
