"""
High-quality benchmark runners for the upgraded FairFlow paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from research.baselines import (
    GroupThresholdOptimizer,
    OnlineGroupThresholdConfig,
    OnlineGroupThresholdController,
    RuleBasedController,
    evaluate_static_method,
    model_scores,
    train_base_models,
)
from research.dataset_catalog import dataset_catalog_rows
from research.datasets import load_dataset
from research.elite_methods import (
    AdaptiveGuardConfig,
    PrimalDualFairnessController,
    PrimalDualGuardConfig,
    PrimalDualOffsetConfig,
    PrimalDualOffsetController,
    ProjectedFairnessGuard,
    SelectiveAnchoredRLGuard,
    SelectiveGuardConfig,
    SelectiveThresholdGuard,
)
from research.metrics import build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.rl import RLTrainingConfig, evaluate_rl_controller, make_order_indices, train_dataset_specific_controller, train_universal_controller
from research.statistics import AggregateSpec, aggregate_with_intervals, paired_win_summary


@dataclass
class EliteExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: ["adult", "german_credit", "compas", "bank_marketing", "recruitment"])
    model_names: list[str] = field(default_factory=lambda: ["logistic_regression", "xgboost"])
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])
    fairness_threshold: float = 0.8
    window_size: int = 50
    order_protocol: str = "natural"
    universal_rl: RLTrainingConfig = field(
        default_factory=lambda: RLTrainingConfig(
            total_timesteps=60_000,
            accuracy_weight=0.55,
            fairness_weight=0.45,
            intervention_penalty=-0.08,
            fairness_upper_target=1.02,
            fairness_band_bonus=0.30,
            overshoot_penalty_weight=1.00,
            tag="elite_universal",
        )
    )
    dataset_specific_rl: RLTrainingConfig = field(
        default_factory=lambda: RLTrainingConfig(
            total_timesteps=25_000,
            n_steps=512,
            batch_size=128,
            accuracy_weight=0.60,
            fairness_weight=0.40,
            intervention_penalty=-0.08,
            fairness_upper_target=1.02,
            fairness_band_bonus=0.30,
            overshoot_penalty_weight=1.00,
            tag="elite_dataset_specific",
        )
    )
    guard: SelectiveGuardConfig = field(default_factory=SelectiveGuardConfig)
    adaptive_guard: AdaptiveGuardConfig = field(default_factory=AdaptiveGuardConfig)
    primal_dual_guard: PrimalDualGuardConfig = field(default_factory=PrimalDualGuardConfig)
    primal_dual_offset: PrimalDualOffsetConfig = field(default_factory=PrimalDualOffsetConfig)
    online_group_threshold: OnlineGroupThresholdConfig = field(default_factory=OnlineGroupThresholdConfig)
    include_online_group_threshold: bool = False


@dataclass
class OrderStressConfig:
    datasets: list[str] = field(default_factory=lambda: ["adult", "compas", "recruitment"])
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])
    model_name: str = "xgboost"
    protocols: list[str] = field(default_factory=lambda: ["natural", "alternating_groups", "privileged_first", "unprivileged_first"])
    benchmark: EliteExperimentConfig = field(default_factory=EliteExperimentConfig)


@dataclass
class GuardAblationConfig:
    datasets: list[str] = field(default_factory=lambda: ["adult", "compas", "recruitment"])
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])
    model_name: str = "xgboost"
    include_dataset_specific_rl: bool = False
    benchmark: EliteExperimentConfig = field(default_factory=EliteExperimentConfig)


def _ordered_arrays(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    scores: np.ndarray,
    order_protocol: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = make_order_indices(protected, protocol=order_protocol, seed=seed)
    return y_true[order], y_pred[order], protected[order], scores[order]


def _result_row(
    *,
    seed: int,
    dataset_name: str,
    model_name: str,
    method_name: str,
    order_protocol: str,
    metrics: dict,
    base_metrics: dict,
    extra: Optional[dict] = None,
) -> dict:
    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "order_protocol": order_protocol,
        **metrics,
        "accuracy_delta_vs_base": float(metrics["accuracy"] - base_metrics["accuracy"]),
        "dpr_delta_vs_base": float(metrics["demographic_parity_ratio"] - base_metrics["demographic_parity_ratio"]),
        "eo_gap_delta_vs_base": float(metrics["equalized_odds_gap"] - base_metrics["equalized_odds_gap"]),
        "intervention_delta_vs_base": float(metrics.get("intervention_rate", 0.0) - base_metrics.get("intervention_rate", 0.0)),
    }
    if extra:
        row.update(extra)
    return row


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _append_trace_rows(
    trace_rows: list[dict],
    *,
    seed: int,
    dataset_name: str,
    model_name: str,
    method_name: str,
    order_protocol: str,
    trace: pd.DataFrame,
) -> None:
    trace_rows.extend(
        {
            "seed": seed,
            "dataset": dataset_name,
            "model": model_name,
            "method": method_name,
            "order_protocol": order_protocol,
            **row,
        }
        for row in trace.to_dict(orient="records")
    )


def _write_elite_outputs(
    *,
    output_path: Path,
    per_run_results: pd.DataFrame,
    trace_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
) -> None:
    per_run_results.to_csv(output_path / "per_run_results.csv", index=False)
    trace_frame.to_csv(output_path / "rolling_traces.csv", index=False)
    diagnostics_frame.to_csv(output_path / "guard_diagnostics.csv", index=False)

    aggregate_spec = AggregateSpec(
        group_cols=("dataset", "model", "method", "order_protocol"),
        metric_cols=(
            "accuracy",
            "balanced_accuracy",
            "demographic_parity_ratio",
            "equalized_odds_gap",
            "intervention_rate",
            "rolling_fair_rate",
            "rolling_tail_avg_dpr",
            "rolling_tail_fair_rate",
            "guard_activation_rate",
            "guard_accept_rate",
        ),
    )
    aggregated = aggregate_with_intervals(per_run_results, aggregate_spec)
    aggregated.to_csv(output_path / "aggregated_results.csv", index=False)

    method_summary = aggregate_with_intervals(
        per_run_results,
        AggregateSpec(
            group_cols=("method",),
            metric_cols=(
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_fair_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ),
        ),
    )
    method_summary.to_csv(output_path / "method_summary.csv", index=False)

    accuracy_wins = paired_win_summary(
        per_run_results,
        against_method="group_threshold",
        metric="accuracy",
        higher_is_better=True,
    )
    dpr_wins = paired_win_summary(
        per_run_results,
        against_method="group_threshold",
        metric="demographic_parity_ratio",
        higher_is_better=True,
    )
    eo_wins = paired_win_summary(
        per_run_results,
        against_method="group_threshold",
        metric="equalized_odds_gap",
        higher_is_better=False,
    )
    win_summary = pd.concat([accuracy_wins, dpr_wins, eo_wins], ignore_index=True)
    win_summary.to_csv(output_path / "paired_win_summary.csv", index=False)


def run_elite_benchmark(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    config: EliteExperimentConfig,
    show_rl_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(parents=True, exist_ok=True)
    _save_json(output_path / "elite_config.json", asdict(config))
    _save_json(output_path / "dataset_catalog.json", {"datasets": dataset_catalog_rows()})

    result_rows: list[dict] = []
    trace_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    dataset_summary_rows: list[dict] = []

    for seed in tqdm(config.seeds, desc="Seeds"):
        universal_dir = output_path / "models" / "universal" / f"seed_{seed}"
        universal_cfg = RLTrainingConfig(**config.universal_rl.to_dict())
        universal_cfg.random_seed = seed
        universal_cfg.tag = f"{config.universal_rl.tag}_seed_{seed}"
        universal_model = train_universal_controller(
            output_dir=universal_dir,
            config=universal_cfg,
            curriculum=True,
            device="cpu",
            show_progress=show_rl_progress,
        )

        for dataset_name in tqdm(config.datasets, desc=f"Seed {seed} datasets", leave=False):
            dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=seed)
            dataset_summary_rows.append({"seed": seed, **dataset.summary})

            dataset_dir = output_path / "models" / dataset_name / f"seed_{seed}"
            models = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=config.model_names,
                seed=seed,
                output_dir=dataset_dir,
            )

            for model_name in tqdm(list(models.keys()), desc=f"{dataset_name} models", leave=False):
                model = models[model_name]
                val_scores = model_scores(model, dataset.X_val)
                test_scores = model_scores(model, dataset.X_test)
                val_base_preds = model.predict(dataset.X_val)
                test_base_preds = model.predict(dataset.X_test)

                threshold_baseline = GroupThresholdOptimizer(fairness_threshold=config.fairness_threshold).fit(
                    dataset.y_val.to_numpy(),
                    val_scores,
                    dataset.protected_val.to_numpy(),
                )
                rule_baseline = RuleBasedController(
                    fairness_threshold=config.fairness_threshold,
                    window_size=config.window_size,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=val_base_preds,
                    base_scores=val_scores,
                    protected=dataset.protected_val.to_numpy(),
                )
                online_threshold_baseline = None
                if config.include_online_group_threshold:
                    online_threshold_baseline = OnlineGroupThresholdController(
                        config=config.online_group_threshold,
                    ).fit(
                        y_true=dataset.y_val.to_numpy(),
                        base_scores=np.asarray(val_scores, dtype=float),
                        protected=dataset.protected_val.to_numpy(),
                    )

                threshold_guard = SelectiveThresholdGuard(
                    threshold_baseline=threshold_baseline,
                    config=config.guard,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                adaptive_guard = ProjectedFairnessGuard(
                    threshold_baseline=threshold_baseline,
                    config=config.adaptive_guard,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                primal_dual_guard = PrimalDualFairnessController(
                    threshold_baseline=threshold_baseline,
                    config=config.primal_dual_guard,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                primal_dual_offset = PrimalDualOffsetController(
                    threshold_baseline=threshold_baseline,
                    config=config.primal_dual_offset,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                anchored_guard = SelectiveAnchoredRLGuard(
                    rl_model=universal_model,
                    threshold_baseline=threshold_baseline,
                    rl_config=universal_cfg,
                    guard_config=config.guard,
                    allow_direct_rl_override=False,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                ordered_truth, ordered_base_preds, ordered_protected, ordered_scores = _ordered_arrays(
                    y_true=dataset.y_test.to_numpy(),
                    y_pred=np.asarray(test_base_preds, dtype=int),
                    protected=dataset.protected_test.to_numpy(),
                    scores=np.asarray(test_scores, dtype=float),
                    order_protocol=config.order_protocol,
                    seed=seed,
                )

                base_trace = build_stream_trace(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    window_size=config.window_size,
                    fairness_threshold=config.fairness_threshold,
                )
                base_metrics = compute_static_metrics(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    scores=ordered_scores,
                    fairness_threshold=config.fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace, warmup=config.window_size))
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="base_model",
                        order_protocol=config.order_protocol,
                        metrics=base_metrics,
                        base_metrics=base_metrics,
                    )
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="base_model",
                    order_protocol=config.order_protocol,
                    trace=base_trace,
                )

                threshold_preds = threshold_baseline.predict(ordered_scores, ordered_protected).astype(int)
                threshold_metrics = evaluate_static_method(
                    y_true=ordered_truth,
                    y_pred=threshold_preds,
                    protected=ordered_protected,
                    scores=ordered_scores,
                    intervention_flags=(threshold_preds != ordered_base_preds).astype(int),
                    fairness_threshold=config.fairness_threshold,
                    window_size=config.window_size,
                )
                threshold_trace = build_stream_trace(
                    ordered_truth,
                    threshold_preds,
                    ordered_protected,
                    window_size=config.window_size,
                    fairness_threshold=config.fairness_threshold,
                )
                threshold_metrics.update(summarize_stream_trace(threshold_trace, warmup=config.window_size))
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="group_threshold",
                        order_protocol=config.order_protocol,
                        metrics=threshold_metrics,
                        base_metrics=base_metrics,
                    )
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="group_threshold",
                    order_protocol=config.order_protocol,
                    trace=threshold_trace,
                )

                if online_threshold_baseline is not None:
                    online_threshold_preds, online_threshold_interventions, online_threshold_trace = online_threshold_baseline.simulate(
                        y_true=ordered_truth,
                        base_scores=ordered_scores,
                        protected=ordered_protected,
                    )
                    online_threshold_metrics = compute_static_metrics(
                        ordered_truth,
                        online_threshold_preds,
                        ordered_protected,
                        scores=ordered_scores,
                        intervention_flags=online_threshold_interventions,
                        fairness_threshold=config.fairness_threshold,
                    )
                    online_threshold_metrics.update(
                        summarize_stream_trace(online_threshold_trace, warmup=config.window_size)
                    )
                    result_rows.append(
                        _result_row(
                            seed=seed,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            method_name="group_threshold_online",
                            order_protocol=config.order_protocol,
                            metrics=online_threshold_metrics,
                            base_metrics=base_metrics,
                        )
                    )
                    _append_trace_rows(
                        trace_rows,
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="group_threshold_online",
                        order_protocol=config.order_protocol,
                        trace=online_threshold_trace,
                    )

                rule_preds, rule_interventions, rule_trace = rule_baseline.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                rule_metrics = compute_static_metrics(
                    ordered_truth,
                    rule_preds,
                    ordered_protected,
                    scores=ordered_scores,
                    intervention_flags=rule_interventions,
                    fairness_threshold=config.fairness_threshold,
                )
                rule_metrics.update(summarize_stream_trace(rule_trace, warmup=config.window_size))
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="rule_based",
                        order_protocol=config.order_protocol,
                        metrics=rule_metrics,
                        base_metrics=base_metrics,
                    )
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="rule_based",
                    order_protocol=config.order_protocol,
                    trace=rule_trace,
                )

                universal_metrics, universal_trace, _, _ = evaluate_rl_controller(
                    model=universal_model,
                    predictions=np.asarray(test_base_preds, dtype=int),
                    probabilities=np.asarray(test_scores, dtype=float),
                    true_labels=dataset.y_test.to_numpy(),
                    protected=dataset.protected_test.to_numpy(),
                    order_protocol=config.order_protocol,
                    config=universal_cfg,
                    seed=seed,
                )
                universal_metrics.update(summarize_stream_trace(universal_trace, warmup=config.window_size))
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="universal_rl",
                        order_protocol=config.order_protocol,
                        metrics=universal_metrics,
                        base_metrics=base_metrics,
                    )
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="universal_rl",
                    order_protocol=config.order_protocol,
                    trace=universal_trace,
                )

                threshold_guard_result = threshold_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                threshold_guard_metrics = {
                    **threshold_guard_result.metrics,
                    **threshold_guard_result.diagnostics,
                }
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="guard_threshold",
                        order_protocol=config.order_protocol,
                        metrics=threshold_guard_metrics,
                        base_metrics=base_metrics,
                    )
                )
                diagnostics_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "guard_threshold",
                        **threshold_guard_result.diagnostics,
                    }
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="guard_threshold",
                    order_protocol=config.order_protocol,
                    trace=threshold_guard_result.trace,
                )

                adaptive_guard_result = adaptive_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                adaptive_guard_metrics = {
                    **adaptive_guard_result.metrics,
                    **adaptive_guard_result.diagnostics,
                }
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="adaptive_guard",
                        order_protocol=config.order_protocol,
                        metrics=adaptive_guard_metrics,
                        base_metrics=base_metrics,
                    )
                )
                diagnostics_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "adaptive_guard",
                        **adaptive_guard_result.diagnostics,
                    }
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="adaptive_guard",
                    order_protocol=config.order_protocol,
                    trace=adaptive_guard_result.trace,
                )

                primal_dual_result = primal_dual_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                primal_dual_metrics = {
                    **primal_dual_result.metrics,
                    **primal_dual_result.diagnostics,
                }
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="primal_dual_guard",
                        order_protocol=config.order_protocol,
                        metrics=primal_dual_metrics,
                        base_metrics=base_metrics,
                    )
                )
                diagnostics_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "primal_dual_guard",
                        **primal_dual_result.diagnostics,
                    }
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="primal_dual_guard",
                    order_protocol=config.order_protocol,
                    trace=primal_dual_result.trace,
                )

                primal_dual_offset_result = primal_dual_offset.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                primal_dual_offset_metrics = {
                    **primal_dual_offset_result.metrics,
                    **primal_dual_offset_result.diagnostics,
                }
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="primal_dual_offset",
                        order_protocol=config.order_protocol,
                        metrics=primal_dual_offset_metrics,
                        base_metrics=base_metrics,
                    )
                )
                diagnostics_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "primal_dual_offset",
                        **primal_dual_offset_result.diagnostics,
                    }
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="primal_dual_offset",
                    order_protocol=config.order_protocol,
                    trace=primal_dual_offset_result.trace,
                )

                anchored_guard_result = anchored_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                anchored_guard_metrics = {
                    **anchored_guard_result.metrics,
                    **anchored_guard_result.diagnostics,
                }
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="fairflow_guard_rl",
                        order_protocol=config.order_protocol,
                        metrics=anchored_guard_metrics,
                        base_metrics=base_metrics,
                    )
                )
                diagnostics_rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "fairflow_guard_rl",
                        **anchored_guard_result.diagnostics,
                    }
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name="fairflow_guard_rl",
                    order_protocol=config.order_protocol,
                    trace=anchored_guard_result.trace,
                )

    per_run_results = pd.DataFrame(result_rows)
    trace_frame = pd.DataFrame(trace_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    pd.DataFrame(dataset_summary_rows).drop_duplicates().to_csv(output_path / "dataset_summaries.csv", index=False)
    _write_elite_outputs(
        output_path=output_path,
        per_run_results=per_run_results,
        trace_frame=trace_frame,
        diagnostics_frame=diagnostics_frame,
    )
    return per_run_results, trace_frame, diagnostics_frame


def run_guard_ablation(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    config: GuardAblationConfig,
    show_rl_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_json(output_path / "guard_ablation_config.json", asdict(config))

    result_rows: list[dict] = []
    trace_rows: list[dict] = []

    for seed in tqdm(config.seeds, desc="Ablation seeds"):
        benchmark_cfg = config.benchmark
        universal_cfg = RLTrainingConfig(**benchmark_cfg.universal_rl.to_dict())
        universal_cfg.random_seed = seed
        universal_cfg.tag = f"ablation_universal_seed_{seed}"
        universal_model = train_universal_controller(
            output_dir=output_path / "models" / "universal" / f"seed_{seed}",
            config=universal_cfg,
            curriculum=True,
            device="cpu",
            show_progress=show_rl_progress,
        )

        for dataset_name in tqdm(config.datasets, desc=f"Seed {seed} datasets", leave=False):
            dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=seed)
            base_model = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=[config.model_name],
                seed=seed,
            )[config.model_name]

            val_scores = model_scores(base_model, dataset.X_val)
            test_scores = model_scores(base_model, dataset.X_test)
            val_base_preds = base_model.predict(dataset.X_val)
            test_base_preds = base_model.predict(dataset.X_test)
            ordered_truth, ordered_base_preds, ordered_protected, ordered_scores = _ordered_arrays(
                y_true=dataset.y_test.to_numpy(),
                y_pred=np.asarray(test_base_preds, dtype=int),
                protected=dataset.protected_test.to_numpy(),
                scores=np.asarray(test_scores, dtype=float),
                order_protocol="natural",
                seed=seed,
            )

            base_trace = build_stream_trace(
                ordered_truth,
                ordered_base_preds,
                ordered_protected,
                window_size=benchmark_cfg.window_size,
                fairness_threshold=benchmark_cfg.fairness_threshold,
            )
            base_metrics = compute_static_metrics(
                ordered_truth,
                ordered_base_preds,
                ordered_protected,
                scores=ordered_scores,
                fairness_threshold=benchmark_cfg.fairness_threshold,
            )
            base_metrics.update(summarize_stream_trace(base_trace, warmup=benchmark_cfg.window_size))

            threshold_baseline = GroupThresholdOptimizer(fairness_threshold=benchmark_cfg.fairness_threshold).fit(
                dataset.y_val.to_numpy(),
                val_scores,
                dataset.protected_val.to_numpy(),
            )

            method_results: dict[str, tuple[dict, pd.DataFrame]] = {}

            threshold_preds = threshold_baseline.predict(ordered_scores, ordered_protected).astype(int)
            threshold_trace = build_stream_trace(
                ordered_truth,
                threshold_preds,
                ordered_protected,
                window_size=benchmark_cfg.window_size,
                fairness_threshold=benchmark_cfg.fairness_threshold,
            )
            threshold_metrics = compute_static_metrics(
                ordered_truth,
                threshold_preds,
                ordered_protected,
                scores=ordered_scores,
                intervention_flags=(threshold_preds != ordered_base_preds).astype(int),
                fairness_threshold=benchmark_cfg.fairness_threshold,
            )
            threshold_metrics.update(summarize_stream_trace(threshold_trace, warmup=benchmark_cfg.window_size))
            method_results["group_threshold"] = (threshold_metrics, threshold_trace)

            standard_guard = SelectiveThresholdGuard(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            standard_result = standard_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["guard_threshold"] = (
                {**standard_result.metrics, **standard_result.diagnostics},
                standard_result.trace,
            )

            adaptive_guard = ProjectedFairnessGuard(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.adaptive_guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            adaptive_result = adaptive_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["adaptive_guard"] = (
                {**adaptive_result.metrics, **adaptive_result.diagnostics},
                adaptive_result.trace,
            )

            primal_dual_guard = PrimalDualFairnessController(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.primal_dual_guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            primal_dual_result = primal_dual_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["primal_dual_guard"] = (
                {**primal_dual_result.metrics, **primal_dual_result.diagnostics},
                primal_dual_result.trace,
            )

            primal_dual_offset = PrimalDualOffsetController(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.primal_dual_offset,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            primal_dual_offset_result = primal_dual_offset.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["primal_dual_offset"] = (
                {**primal_dual_offset_result.metrics, **primal_dual_offset_result.diagnostics},
                primal_dual_offset_result.trace,
            )

            no_hysteresis_cfg = SelectiveGuardConfig(**asdict(benchmark_cfg.guard))
            no_hysteresis_cfg.release_dpr_grid = no_hysteresis_cfg.activation_dpr_grid
            no_hysteresis_cfg.overshoot_release_grid = no_hysteresis_cfg.overshoot_activation_grid
            no_hysteresis_cfg.cooldown_grid = (0,)
            no_hysteresis_guard = SelectiveThresholdGuard(
                threshold_baseline=threshold_baseline,
                config=no_hysteresis_cfg,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            no_hysteresis_result = no_hysteresis_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["guard_threshold_no_hysteresis"] = (
                {**no_hysteresis_result.metrics, **no_hysteresis_result.diagnostics},
                no_hysteresis_result.trace,
            )

            universal_metrics, universal_trace, _, _ = evaluate_rl_controller(
                model=universal_model,
                predictions=np.asarray(test_base_preds, dtype=int),
                probabilities=np.asarray(test_scores, dtype=float),
                true_labels=dataset.y_test.to_numpy(),
                protected=dataset.protected_test.to_numpy(),
                order_protocol="natural",
                config=universal_cfg,
                seed=seed,
            )
            universal_metrics.update(summarize_stream_trace(universal_trace, warmup=benchmark_cfg.window_size))
            method_results["universal_rl"] = (universal_metrics, universal_trace)

            anchored_guard = SelectiveAnchoredRLGuard(
                rl_model=universal_model,
                threshold_baseline=threshold_baseline,
                rl_config=universal_cfg,
                guard_config=benchmark_cfg.guard,
                allow_direct_rl_override=False,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            anchored_result = anchored_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["fairflow_guard_rl"] = (
                {**anchored_result.metrics, **anchored_result.diagnostics},
                anchored_result.trace,
            )

            free_guard = SelectiveAnchoredRLGuard(
                rl_model=universal_model,
                threshold_baseline=threshold_baseline,
                rl_config=universal_cfg,
                guard_config=benchmark_cfg.guard,
                allow_direct_rl_override=True,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            free_result = free_guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            method_results["guard_rl_no_anchor"] = (
                {**free_result.metrics, **free_result.diagnostics},
                free_result.trace,
            )

            if config.include_dataset_specific_rl:
                ds_cfg = RLTrainingConfig(**benchmark_cfg.dataset_specific_rl.to_dict())
                ds_cfg.random_seed = seed
                ds_cfg.tag = f"ablation_ds_seed_{seed}_{dataset_name}"
                ds_model = train_dataset_specific_controller(
                    predictions=base_model.predict(dataset.X_train),
                    probabilities=model_scores(base_model, dataset.X_train),
                    true_labels=dataset.y_train.to_numpy(),
                    protected=dataset.protected_train.to_numpy(),
                    output_dir=output_path / "models" / "dataset_specific" / f"seed_{seed}",
                    config=ds_cfg,
                    tag=f"{dataset_name}_{config.model_name}",
                    device="cpu",
                    show_progress=show_rl_progress,
                )
                ds_metrics, ds_trace, _, _ = evaluate_rl_controller(
                    model=ds_model,
                    predictions=np.asarray(test_base_preds, dtype=int),
                    probabilities=np.asarray(test_scores, dtype=float),
                    true_labels=dataset.y_test.to_numpy(),
                    protected=dataset.protected_test.to_numpy(),
                    order_protocol="natural",
                    config=ds_cfg,
                    seed=seed,
                )
                ds_metrics.update(summarize_stream_trace(ds_trace, warmup=benchmark_cfg.window_size))
                method_results["dataset_specific_rl"] = (ds_metrics, ds_trace)

            for method_name, (metrics, trace) in method_results.items():
                result_rows.append(
                    _result_row(
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=config.model_name,
                        method_name=method_name,
                        order_protocol="natural",
                        metrics=metrics,
                        base_metrics=base_metrics,
                    )
                )
                _append_trace_rows(
                    trace_rows,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_name=config.model_name,
                    method_name=method_name,
                    order_protocol="natural",
                    trace=trace,
                )

    per_run_results = pd.DataFrame(result_rows)
    traces = pd.DataFrame(trace_rows)
    per_run_results.to_csv(output_path / "per_run_results.csv", index=False)
    traces.to_csv(output_path / "rolling_traces.csv", index=False)
    aggregate_with_intervals(
        per_run_results,
        AggregateSpec(
            group_cols=("dataset", "method"),
            metric_cols=(
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "guard_activation_rate",
            ),
        ),
    ).to_csv(output_path / "aggregated_results.csv", index=False)
    return per_run_results, traces


def run_order_stress_benchmark(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    config: OrderStressConfig,
    show_rl_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_json(output_path / "order_stress_config.json", asdict(config))

    result_rows: list[dict] = []
    trace_rows: list[dict] = []

    for seed in tqdm(config.seeds, desc="Order-stress seeds"):
        benchmark_cfg = config.benchmark
        universal_cfg = RLTrainingConfig(**benchmark_cfg.universal_rl.to_dict())
        universal_cfg.random_seed = seed
        universal_cfg.tag = f"order_stress_seed_{seed}"
        universal_model = train_universal_controller(
            output_dir=output_path / "models" / "universal" / f"seed_{seed}",
            config=universal_cfg,
            curriculum=True,
            device="cpu",
            show_progress=show_rl_progress,
        )

        for dataset_name in tqdm(config.datasets, desc=f"Seed {seed} datasets", leave=False):
            dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=seed)
            base_model = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=[config.model_name],
                seed=seed,
            )[config.model_name]

            val_scores = model_scores(base_model, dataset.X_val)
            test_scores = model_scores(base_model, dataset.X_test)
            val_base_preds = base_model.predict(dataset.X_val)
            test_base_preds = base_model.predict(dataset.X_test)

            threshold_baseline = GroupThresholdOptimizer(fairness_threshold=benchmark_cfg.fairness_threshold).fit(
                dataset.y_val.to_numpy(),
                val_scores,
                dataset.protected_val.to_numpy(),
            )
            online_threshold_baseline = None
            if benchmark_cfg.include_online_group_threshold:
                online_threshold_baseline = OnlineGroupThresholdController(
                    config=benchmark_cfg.online_group_threshold,
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )
            threshold_guard = SelectiveThresholdGuard(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            adaptive_guard = ProjectedFairnessGuard(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.adaptive_guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            primal_dual_guard = PrimalDualFairnessController(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.primal_dual_guard,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )
            primal_dual_offset = PrimalDualOffsetController(
                threshold_baseline=threshold_baseline,
                config=benchmark_cfg.primal_dual_offset,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            anchored_guard = SelectiveAnchoredRLGuard(
                rl_model=universal_model,
                threshold_baseline=threshold_baseline,
                rl_config=universal_cfg,
                guard_config=benchmark_cfg.guard,
                allow_direct_rl_override=False,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )

            for protocol in config.protocols:
                ordered_truth, ordered_base_preds, ordered_protected, ordered_scores = _ordered_arrays(
                    y_true=dataset.y_test.to_numpy(),
                    y_pred=np.asarray(test_base_preds, dtype=int),
                    protected=dataset.protected_test.to_numpy(),
                    scores=np.asarray(test_scores, dtype=float),
                    order_protocol=protocol,
                    seed=seed,
                )
                base_trace = build_stream_trace(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    window_size=benchmark_cfg.window_size,
                    fairness_threshold=benchmark_cfg.fairness_threshold,
                )
                base_metrics = compute_static_metrics(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    scores=ordered_scores,
                    fairness_threshold=benchmark_cfg.fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace, warmup=benchmark_cfg.window_size))

                threshold_result = threshold_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                adaptive_result = adaptive_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                online_threshold_payload = None
                if online_threshold_baseline is not None:
                    online_threshold_preds, online_threshold_interventions, online_threshold_trace = online_threshold_baseline.simulate(
                        y_true=ordered_truth,
                        base_scores=ordered_scores,
                        protected=ordered_protected,
                    )
                    online_threshold_metrics = compute_static_metrics(
                        ordered_truth,
                        online_threshold_preds,
                        ordered_protected,
                        scores=ordered_scores,
                        intervention_flags=online_threshold_interventions,
                        fairness_threshold=benchmark_cfg.fairness_threshold,
                    )
                    online_threshold_metrics.update(
                        summarize_stream_trace(online_threshold_trace, warmup=benchmark_cfg.window_size)
                    )
                    online_threshold_payload = (online_threshold_metrics, online_threshold_trace)
                primal_dual_result = primal_dual_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                primal_dual_offset_result = primal_dual_offset.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                anchored_result = anchored_guard.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                universal_metrics, universal_trace, _, _ = evaluate_rl_controller(
                    model=universal_model,
                    predictions=np.asarray(test_base_preds, dtype=int),
                    probabilities=np.asarray(test_scores, dtype=float),
                    true_labels=dataset.y_test.to_numpy(),
                    protected=dataset.protected_test.to_numpy(),
                    order_protocol=protocol,
                    config=universal_cfg,
                    seed=seed,
                )
                universal_metrics.update(summarize_stream_trace(universal_trace, warmup=benchmark_cfg.window_size))

                method_payloads = {
                    "base_model": (base_metrics, base_trace),
                    "guard_threshold": ({**threshold_result.metrics, **threshold_result.diagnostics}, threshold_result.trace),
                    "adaptive_guard": ({**adaptive_result.metrics, **adaptive_result.diagnostics}, adaptive_result.trace),
                    "primal_dual_guard": ({**primal_dual_result.metrics, **primal_dual_result.diagnostics}, primal_dual_result.trace),
                    "primal_dual_offset": ({**primal_dual_offset_result.metrics, **primal_dual_offset_result.diagnostics}, primal_dual_offset_result.trace),
                    "fairflow_guard_rl": ({**anchored_result.metrics, **anchored_result.diagnostics}, anchored_result.trace),
                    "universal_rl": (universal_metrics, universal_trace),
                }
                if online_threshold_payload is not None:
                    method_payloads["group_threshold_online"] = online_threshold_payload
                for method_name, (metrics, trace) in method_payloads.items():
                    result_rows.append(
                        _result_row(
                            seed=seed,
                            dataset_name=dataset_name,
                            model_name=config.model_name,
                            method_name=method_name,
                            order_protocol=protocol,
                            metrics=metrics,
                            base_metrics=base_metrics,
                        )
                    )
                    _append_trace_rows(
                        trace_rows,
                        seed=seed,
                        dataset_name=dataset_name,
                        model_name=config.model_name,
                        method_name=method_name,
                        order_protocol=protocol,
                        trace=trace,
                    )

    per_run_results = pd.DataFrame(result_rows)
    traces = pd.DataFrame(trace_rows)
    per_run_results.to_csv(output_path / "per_run_results.csv", index=False)
    traces.to_csv(output_path / "rolling_traces.csv", index=False)
    aggregate_with_intervals(
        per_run_results,
        AggregateSpec(
            group_cols=("dataset", "method", "order_protocol"),
            metric_cols=(
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ),
        ),
    ).to_csv(output_path / "aggregated_results.csv", index=False)
    return per_run_results, traces
