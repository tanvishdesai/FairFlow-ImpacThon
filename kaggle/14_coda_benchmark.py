"""
Run the CODA (Continuous Online Dual-Adaptive Thresholding) benchmark.

This script compares CODA against all existing methods on focused datasets,
including order-stress testing for robustness evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import (
    GroupThresholdOptimizer,
    OnlineGroupThresholdConfig,
    OnlineGroupThresholdController,
    RuleBasedController,
    evaluate_static_method,
    model_scores,
    train_base_models,
)
from research.coda_controller import CODAConfig, CODAController, build_coda_with_calibrated_init
from research.datasets import load_dataset
from research.elite_methods import (
    SelectiveGuardConfig,
    SelectiveThresholdGuard,
    PrimalDualGuardConfig,
    PrimalDualFairnessController,
)
from research.metrics import build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.rl import RLTrainingConfig, evaluate_rl_controller, make_order_indices, train_universal_controller
from research.statistics import AggregateSpec, aggregate_with_intervals, paired_win_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CODA FairFlow benchmark.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/coda_v2_benchmark")
    parser.add_argument("--datasets", nargs="+", default=["adult", "compas", "german_credit", "bank_marketing", "recruitment"])
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--order-protocols",
        nargs="+",
        default=["natural", "privileged_first", "unprivileged_first"],
    )
    parser.add_argument("--skip-rl", action="store_true", help="Skip universal RL (faster runs)")
    parser.add_argument("--show-rl-progress", action="store_true")
    return parser.parse_args()


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
    extra: dict | None = None,
) -> dict:
    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "order_protocol": order_protocol,
        **metrics,
        "accuracy_delta_vs_base": float(metrics["accuracy"] - base_metrics["accuracy"]),
        "dpr_delta_vs_base": float(
            metrics["demographic_parity_ratio"] - base_metrics["demographic_parity_ratio"]
        ),
        "eo_gap_delta_vs_base": float(
            metrics["equalized_odds_gap"] - base_metrics["equalized_odds_gap"]
        ),
        "intervention_delta_vs_base": float(
            metrics.get("intervention_rate", 0.0) - base_metrics.get("intervention_rate", 0.0)
        ),
    }
    if extra:
        row.update(extra)
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "coda_benchmark_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    fairness_threshold = 0.8
    window_size = 50

    # ========================================================================
    # PHASE 1: Natural-order benchmark (main results)
    # ========================================================================
    print("=" * 70)
    print("PHASE 1: Main benchmark (natural order)")
    print("=" * 70)

    benchmark_rows: list[dict] = []
    trace_rows: list[dict] = []
    diagnostics_rows: list[dict] = []

    for seed in tqdm(args.seeds, desc="Seeds"):
        # Train universal RL once per seed (if not skipped)
        universal_model = None
        universal_cfg = None
        if not args.skip_rl:
            universal_cfg = RLTrainingConfig(
                total_timesteps=60_000,
                accuracy_weight=0.55,
                fairness_weight=0.45,
                intervention_penalty=-0.08,
                fairness_upper_target=1.02,
                fairness_band_bonus=0.30,
                overshoot_penalty_weight=1.00,
                random_seed=seed,
                tag=f"coda_universal_seed_{seed}",
            )
            universal_model = train_universal_controller(
                output_dir=output_dir / "models" / "universal" / f"seed_{seed}",
                config=universal_cfg,
                curriculum=True,
                device="cpu",
                show_progress=args.show_rl_progress,
            )

        for dataset_name in tqdm(args.datasets, desc=f"Seed {seed}", leave=False):
            dataset = load_dataset(dataset_name, search_roots=args.search_root, random_state=seed)

            for model_name in tqdm(args.models, desc=f"{dataset_name}", leave=False):
                models = train_base_models(
                    dataset.X_train,
                    dataset.y_train,
                    model_names=[model_name],
                    seed=seed,
                )
                model = models[model_name]

                val_scores = model_scores(model, dataset.X_val)
                test_scores = model_scores(model, dataset.X_test)
                val_base_preds = model.predict(dataset.X_val)
                test_base_preds = model.predict(dataset.X_test)

                # --- Fit baselines on validation ---
                threshold_baseline = GroupThresholdOptimizer(
                    fairness_threshold=fairness_threshold
                ).fit(
                    dataset.y_val.to_numpy(),
                    val_scores,
                    dataset.protected_val.to_numpy(),
                )

                guard = SelectiveThresholdGuard(
                    threshold_baseline=threshold_baseline,
                    config=SelectiveGuardConfig(),
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                # --- Fit CODA v2 on validation (seeded from group_threshold) ---
                coda = build_coda_with_calibrated_init(
                    config=CODAConfig(),
                    threshold_baseline=threshold_baseline,
                    val_scores=np.asarray(val_scores, dtype=float),
                    val_protected=dataset.protected_val.to_numpy(),
                    val_y_true=dataset.y_val.to_numpy(),
                ).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                # --- Evaluate on test (natural order) ---
                ordered_truth, ordered_base_preds_arr, ordered_protected, ordered_scores = (
                    _ordered_arrays(
                        y_true=dataset.y_test.to_numpy(),
                        y_pred=np.asarray(test_base_preds, dtype=int),
                        protected=dataset.protected_test.to_numpy(),
                        scores=np.asarray(test_scores, dtype=float),
                        order_protocol="natural",
                        seed=seed,
                    )
                )

                # --- Base model ---
                base_trace = build_stream_trace(
                    ordered_truth, ordered_base_preds_arr, ordered_protected,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                base_metrics = compute_static_metrics(
                    ordered_truth, ordered_base_preds_arr, ordered_protected,
                    scores=ordered_scores, fairness_threshold=fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace, warmup=window_size))
                benchmark_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="base_model", order_protocol="natural",
                    metrics=base_metrics, base_metrics=base_metrics,
                ))

                # --- group_threshold ---
                threshold_preds = threshold_baseline.predict(ordered_scores, ordered_protected).astype(int)
                threshold_metrics = evaluate_static_method(
                    y_true=ordered_truth, y_pred=threshold_preds,
                    protected=ordered_protected, scores=ordered_scores,
                    intervention_flags=(threshold_preds != ordered_base_preds_arr).astype(int),
                    fairness_threshold=fairness_threshold, window_size=window_size,
                )
                threshold_trace = build_stream_trace(
                    ordered_truth, threshold_preds, ordered_protected,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                threshold_metrics.update(summarize_stream_trace(threshold_trace, warmup=window_size))
                benchmark_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="group_threshold", order_protocol="natural",
                    metrics=threshold_metrics, base_metrics=base_metrics,
                ))

                # --- guard_threshold ---
                guard_result = guard.simulate(
                    y_true=ordered_truth, base_preds=ordered_base_preds_arr,
                    base_scores=ordered_scores, protected=ordered_protected,
                )
                guard_metrics = {**guard_result.metrics, **guard_result.diagnostics}
                benchmark_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="guard_threshold", order_protocol="natural",
                    metrics=guard_metrics, base_metrics=base_metrics,
                ))

                # --- universal_rl ---
                if universal_model is not None and universal_cfg is not None:
                    rl_metrics, rl_trace, _, _ = evaluate_rl_controller(
                        model=universal_model,
                        predictions=np.asarray(test_base_preds, dtype=int),
                        probabilities=np.asarray(test_scores, dtype=float),
                        true_labels=dataset.y_test.to_numpy(),
                        protected=dataset.protected_test.to_numpy(),
                        order_protocol="natural",
                        config=universal_cfg,
                        seed=seed,
                    )
                    rl_metrics.update(summarize_stream_trace(rl_trace, warmup=window_size))
                    benchmark_rows.append(_result_row(
                        seed=seed, dataset_name=dataset_name, model_name=model_name,
                        method_name="universal_rl", order_protocol="natural",
                        metrics=rl_metrics, base_metrics=base_metrics,
                    ))

                # --- CODA ---
                coda_result = coda.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds_arr,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                coda_metrics = {**coda_result.metrics, **coda_result.diagnostics}
                benchmark_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="coda", order_protocol="natural",
                    metrics=coda_metrics, base_metrics=base_metrics,
                ))
                diagnostics_rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": "coda",
                    "order_protocol": "natural",
                    **coda_result.diagnostics,
                })

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(output_dir / "per_run_results.csv", index=False)
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_df.to_csv(output_dir / "coda_diagnostics.csv", index=False)

    # --- Aggregate results ---
    method_summary = aggregate_with_intervals(
        benchmark_df,
        AggregateSpec(
            group_cols=("method",),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate", "rolling_fair_rate",
                "rolling_tail_avg_dpr", "rolling_tail_fair_rate",
            ),
        ),
    )
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)

    per_dataset = aggregate_with_intervals(
        benchmark_df,
        AggregateSpec(
            group_cols=("dataset", "method"),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate", "rolling_tail_avg_dpr", "rolling_tail_fair_rate",
            ),
        ),
    )
    per_dataset.to_csv(output_dir / "per_dataset_summary.csv", index=False)

    # --- Paired wins ---
    accuracy_wins = paired_win_summary(benchmark_df, against_method="group_threshold", metric="accuracy", higher_is_better=True)
    dpr_wins = paired_win_summary(benchmark_df, against_method="group_threshold", metric="demographic_parity_ratio", higher_is_better=True)
    eo_wins = paired_win_summary(benchmark_df, against_method="group_threshold", metric="equalized_odds_gap", higher_is_better=False)
    win_summary = pd.concat([accuracy_wins, dpr_wins, eo_wins], ignore_index=True)
    win_summary.to_csv(output_dir / "paired_win_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("METHOD SUMMARY (natural order)")
    print("=" * 70)
    print(method_summary.to_string(index=False))
    print()

    # ========================================================================
    # PHASE 2: Order-stress benchmark
    # ========================================================================
    print("=" * 70)
    print("PHASE 2: Order-stress benchmark")
    print("=" * 70)

    order_rows: list[dict] = []

    for seed in tqdm(args.seeds, desc="Order-stress seeds"):
        universal_model_os = None
        universal_cfg_os = None
        if not args.skip_rl:
            universal_cfg_os = RLTrainingConfig(
                total_timesteps=60_000,
                accuracy_weight=0.55,
                fairness_weight=0.45,
                intervention_penalty=-0.08,
                fairness_upper_target=1.02,
                fairness_band_bonus=0.30,
                overshoot_penalty_weight=1.00,
                random_seed=seed,
                tag=f"coda_os_universal_seed_{seed}",
            )
            universal_model_os = train_universal_controller(
                output_dir=output_dir / "models" / "os_universal" / f"seed_{seed}",
                config=universal_cfg_os,
                curriculum=True,
                device="cpu",
                show_progress=args.show_rl_progress,
            )

        for dataset_name in tqdm(args.datasets, desc=f"Seed {seed} OS", leave=False):
            dataset = load_dataset(dataset_name, search_roots=args.search_root, random_state=seed)
            # Use XGBoost for order-stress (consistent with prior experiments)
            os_model_name = "xgboost"
            models = train_base_models(
                dataset.X_train, dataset.y_train,
                model_names=[os_model_name], seed=seed,
            )
            model = models[os_model_name]

            val_scores = model_scores(model, dataset.X_val)
            test_scores = model_scores(model, dataset.X_test)
            val_base_preds = model.predict(dataset.X_val)
            test_base_preds = model.predict(dataset.X_test)

            threshold_baseline = GroupThresholdOptimizer(
                fairness_threshold=fairness_threshold
            ).fit(
                dataset.y_val.to_numpy(), val_scores, dataset.protected_val.to_numpy(),
            )

            guard = SelectiveThresholdGuard(
                threshold_baseline=threshold_baseline,
                config=SelectiveGuardConfig(),
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )

            coda = build_coda_with_calibrated_init(
                config=CODAConfig(),
                threshold_baseline=threshold_baseline,
                val_scores=np.asarray(val_scores, dtype=float),
                val_protected=dataset.protected_val.to_numpy(),
                val_y_true=dataset.y_val.to_numpy(),
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )

            for protocol in tqdm(args.order_protocols, desc=f"{dataset_name} orders", leave=False):
                ordered_truth, ordered_bp, ordered_prot, ordered_sc = _ordered_arrays(
                    y_true=dataset.y_test.to_numpy(),
                    y_pred=np.asarray(test_base_preds, dtype=int),
                    protected=dataset.protected_test.to_numpy(),
                    scores=np.asarray(test_scores, dtype=float),
                    order_protocol=protocol,
                    seed=seed,
                )

                # Base metrics for this order
                base_m = compute_static_metrics(
                    ordered_truth, ordered_bp, ordered_prot,
                    scores=ordered_sc, fairness_threshold=fairness_threshold,
                )
                base_tr = build_stream_trace(
                    ordered_truth, ordered_bp, ordered_prot,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                base_m.update(summarize_stream_trace(base_tr, warmup=window_size))
                order_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=os_model_name,
                    method_name="base_model", order_protocol=protocol,
                    metrics=base_m, base_metrics=base_m,
                ))

                # group_threshold
                tp = threshold_baseline.predict(ordered_sc, ordered_prot).astype(int)
                tm = evaluate_static_method(
                    y_true=ordered_truth, y_pred=tp,
                    protected=ordered_prot, scores=ordered_sc,
                    intervention_flags=(tp != ordered_bp).astype(int),
                    fairness_threshold=fairness_threshold, window_size=window_size,
                )
                tt = build_stream_trace(ordered_truth, tp, ordered_prot, window_size=window_size, fairness_threshold=fairness_threshold)
                tm.update(summarize_stream_trace(tt, warmup=window_size))
                order_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=os_model_name,
                    method_name="group_threshold", order_protocol=protocol,
                    metrics=tm, base_metrics=base_m,
                ))

                # guard_threshold
                gr = guard.simulate(
                    y_true=ordered_truth, base_preds=ordered_bp,
                    base_scores=ordered_sc, protected=ordered_prot,
                )
                gm = {**gr.metrics, **gr.diagnostics}
                order_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=os_model_name,
                    method_name="guard_threshold", order_protocol=protocol,
                    metrics=gm, base_metrics=base_m,
                ))

                # universal_rl
                if universal_model_os is not None and universal_cfg_os is not None:
                    rm, rt, _, _ = evaluate_rl_controller(
                        model=universal_model_os,
                        predictions=np.asarray(test_base_preds, dtype=int),
                        probabilities=np.asarray(test_scores, dtype=float),
                        true_labels=dataset.y_test.to_numpy(),
                        protected=dataset.protected_test.to_numpy(),
                        order_protocol=protocol,
                        config=universal_cfg_os,
                        seed=seed,
                    )
                    rm.update(summarize_stream_trace(rt, warmup=window_size))
                    order_rows.append(_result_row(
                        seed=seed, dataset_name=dataset_name, model_name=os_model_name,
                        method_name="universal_rl", order_protocol=protocol,
                        metrics=rm, base_metrics=base_m,
                    ))

                # CODA
                cr = coda.simulate(
                    y_true=ordered_truth, base_preds=ordered_bp,
                    base_scores=ordered_sc, protected=ordered_prot,
                )
                cm = {**cr.metrics, **cr.diagnostics}
                order_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=os_model_name,
                    method_name="coda", order_protocol=protocol,
                    metrics=cm, base_metrics=base_m,
                ))

    order_df = pd.DataFrame(order_rows)
    order_df.to_csv(output_dir / "order_stress_results.csv", index=False)

    order_summary = aggregate_with_intervals(
        order_df,
        AggregateSpec(
            group_cols=("dataset", "method", "order_protocol"),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate", "rolling_tail_avg_dpr", "rolling_tail_fair_rate",
            ),
        ),
    )
    order_summary.to_csv(output_dir / "order_stress_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("ORDER STRESS SUMMARY")
    print("=" * 70)
    print(order_summary.to_string(index=False))

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - per_run_results.csv ({len(benchmark_df)} rows)")
    print(f"  - method_summary.csv")
    print(f"  - per_dataset_summary.csv")
    print(f"  - paired_win_summary.csv")
    print(f"  - order_stress_results.csv ({len(order_df)} rows)")
    print(f"  - order_stress_summary.csv")
    print(f"  - coda_diagnostics.csv")


if __name__ == "__main__":
    main()
