"""
CODA v2 Ablation Study.

Runs CODA with each v2 component disabled one at a time to demonstrate
that every design decision is necessary.

Ablation variants:
  1. coda_full         — Full CODA v2 (control)
  2. coda_no_damping   — Accuracy-aware damping disabled
  3. coda_no_warmup    — Group-aware warmup disabled
  4. coda_no_ema       — EMA smoothing disabled (alpha=1.0)
  5. coda_naive_init   — Starts from 0.5 instead of calibrated thresholds
  6. coda_no_overshoot — Anti-overshoot mechanism disabled
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import (
    GroupThresholdOptimizer,
    evaluate_static_method,
    model_scores,
    train_base_models,
)
from research.coda_controller import CODAConfig, CODAController, build_coda_with_calibrated_init
from research.datasets import load_dataset
from research.metrics import build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.rl import make_order_indices
from research.statistics import AggregateSpec, aggregate_with_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CODA v2 ablation study.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/coda_ablation")
    parser.add_argument("--datasets", nargs="+", default=["adult", "compas", "german_credit", "bank_marketing", "recruitment"])
    parser.add_argument("--models", nargs="+", default=["xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    return parser.parse_args()


def _make_ablation_configs() -> dict[str, CODAConfig]:
    """Create all ablation variants."""
    full = CODAConfig()

    # 1. No accuracy damping
    no_damping = CODAConfig(accuracy_damping=1.0, accuracy_floor=999.0, tag="coda_no_damping")

    # 2. No group-aware warmup
    no_warmup = CODAConfig(min_group_samples=0, tag="coda_no_warmup")

    # 3. No EMA smoothing (alpha=1.0 means raw threshold, no smoothing)
    no_ema = CODAConfig(
        ema_alpha_grid=(1.0,),
        tag="coda_no_ema",
    )

    # 4. Naive init (handled separately in the loop — starts from 0.5)
    naive_init = CODAConfig(tag="coda_naive_init")

    # 5. No anti-overshoot (set upper target very high)
    no_overshoot = CODAConfig(fairness_upper_target=100.0, tag="coda_no_overshoot")

    return {
        "coda_full": full,
        "coda_no_damping": no_damping,
        "coda_no_warmup": no_warmup,
        "coda_no_ema": no_ema,
        "coda_naive_init": naive_init,
        "coda_no_overshoot": no_overshoot,
    }


def _result_row(
    *,
    seed: int,
    dataset_name: str,
    model_name: str,
    method_name: str,
    metrics: dict,
    base_metrics: dict,
    extra: dict | None = None,
) -> dict:
    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        **metrics,
        "accuracy_delta_vs_base": float(metrics["accuracy"] - base_metrics["accuracy"]),
        "dpr_delta_vs_base": float(
            metrics["demographic_parity_ratio"] - base_metrics["demographic_parity_ratio"]
        ),
    }
    if extra:
        row.update(extra)
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    fairness_threshold = 0.8
    window_size = 50
    ablation_configs = _make_ablation_configs()

    all_rows: list[dict] = []

    for seed in tqdm(args.seeds, desc="Seeds"):
        for dataset_name in tqdm(args.datasets, desc=f"Seed {seed}", leave=False):
            try:
                dataset = load_dataset(dataset_name, search_roots=args.search_root, random_state=seed)
            except FileNotFoundError:
                print(f"  [SKIP] {dataset_name} not found")
                continue

            for model_name in args.models:
                models = train_base_models(
                    dataset.X_train, dataset.y_train,
                    model_names=[model_name], seed=seed,
                )
                model = models[model_name]

                val_scores = model_scores(model, dataset.X_val)
                test_scores = model_scores(model, dataset.X_test)
                val_base_preds = model.predict(dataset.X_val)
                test_base_preds = model.predict(dataset.X_test)

                y_test = dataset.y_test.to_numpy()
                prot_test = dataset.protected_test.to_numpy()
                bp_test = np.asarray(test_base_preds, dtype=int)
                sc_test = np.asarray(test_scores, dtype=float)

                # Base model metrics
                base_metrics = compute_static_metrics(
                    y_test, bp_test, prot_test,
                    scores=sc_test, fairness_threshold=fairness_threshold,
                )
                base_trace = build_stream_trace(
                    y_test, bp_test, prot_test,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace, warmup=window_size))
                all_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="base_model", metrics=base_metrics, base_metrics=base_metrics,
                ))

                # group_threshold baseline
                threshold_baseline = GroupThresholdOptimizer(
                    fairness_threshold=fairness_threshold
                ).fit(
                    dataset.y_val.to_numpy(), val_scores, dataset.protected_val.to_numpy(),
                )
                tp = threshold_baseline.predict(sc_test, prot_test).astype(int)
                gt_metrics = evaluate_static_method(
                    y_true=y_test, y_pred=tp,
                    protected=prot_test, scores=sc_test,
                    intervention_flags=(tp != bp_test).astype(int),
                    fairness_threshold=fairness_threshold, window_size=window_size,
                )
                gt_trace = build_stream_trace(y_test, tp, prot_test, window_size=window_size, fairness_threshold=fairness_threshold)
                gt_metrics.update(summarize_stream_trace(gt_trace, warmup=window_size))
                all_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="group_threshold", metrics=gt_metrics, base_metrics=base_metrics,
                ))

                # Run each ablation variant
                for ablation_name, config in tqdm(ablation_configs.items(), desc="Ablations", leave=False):
                    if ablation_name == "coda_naive_init":
                        # Naive init: start from 0.5 for all groups
                        coda = CODAController(
                            config=config,
                            initial_thresholds={0: 0.5, 1: 0.5},
                        )
                    else:
                        # All other ablations: use calibrated init
                        coda = build_coda_with_calibrated_init(
                            config=config,
                            threshold_baseline=threshold_baseline,
                            val_scores=np.asarray(val_scores, dtype=float),
                            val_protected=dataset.protected_val.to_numpy(),
                            val_y_true=dataset.y_val.to_numpy(),
                        )

                    coda.fit(
                        y_true=dataset.y_val.to_numpy(),
                        base_preds=np.asarray(val_base_preds, dtype=int),
                        base_scores=np.asarray(val_scores, dtype=float),
                        protected=dataset.protected_val.to_numpy(),
                    )

                    result = coda.simulate(
                        y_true=y_test,
                        base_preds=bp_test,
                        base_scores=sc_test,
                        protected=prot_test,
                    )
                    coda_metrics = {**result.metrics, **result.diagnostics}
                    all_rows.append(_result_row(
                        seed=seed, dataset_name=dataset_name, model_name=model_name,
                        method_name=ablation_name, metrics=coda_metrics, base_metrics=base_metrics,
                    ))

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "ablation_per_run.csv", index=False)

    # Aggregate
    summary = aggregate_with_intervals(
        df,
        AggregateSpec(
            group_cols=("method",),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate", "rolling_tail_avg_dpr", "rolling_tail_fair_rate",
            ),
        ),
    )
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    per_dataset = aggregate_with_intervals(
        df,
        AggregateSpec(
            group_cols=("dataset", "method"),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate",
            ),
        ),
    )
    per_dataset.to_csv(output_dir / "ablation_per_dataset.csv", index=False)

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
