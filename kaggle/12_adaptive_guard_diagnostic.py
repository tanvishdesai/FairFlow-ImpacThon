"""
Run a targeted diagnostic sweep for the adaptive guard on the hardest datasets.
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

from research.baselines import GroupThresholdOptimizer, model_scores, train_base_models
from research.datasets import load_dataset
from research.elite_methods import AdaptiveGuardConfig, ProjectedFairnessGuard
from research.rl import make_order_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused adaptive-guard diagnostic sweep.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/adaptive_guard_diagnostic")
    parser.add_argument("--datasets", nargs="+", default=["adult", "compas"])
    parser.add_argument("--model-name", default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deficit-weights", nargs="+", type=float, default=[8.0, 12.0, 16.0])
    parser.add_argument("--deficit-utility-slacks", nargs="+", type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument("--order-protocol", default="natural")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    combos = [
        (float(weight), float(slack))
        for weight in args.deficit_weights
        for slack in args.deficit_utility_slacks
    ]

    for dataset_name in tqdm(args.datasets, desc="Diagnostic datasets"):
        dataset = load_dataset(dataset_name, search_roots=args.search_root, random_state=args.seed)
        model = train_base_models(
            dataset.X_train,
            dataset.y_train,
            model_names=[args.model_name],
            seed=args.seed,
        )[args.model_name]

        val_scores = model_scores(model, dataset.X_val)
        test_scores = model_scores(model, dataset.X_test)
        val_base_preds = model.predict(dataset.X_val)
        test_base_preds = model.predict(dataset.X_test)

        threshold_baseline = GroupThresholdOptimizer(fairness_threshold=0.8).fit(
            dataset.y_val.to_numpy(),
            val_scores,
            dataset.protected_val.to_numpy(),
        )

        order = make_order_indices(dataset.protected_test.to_numpy(), protocol=args.order_protocol, seed=args.seed)
        ordered_truth = dataset.y_test.to_numpy()[order]
        ordered_base_preds = np.asarray(test_base_preds, dtype=int)[order]
        ordered_protected = dataset.protected_test.to_numpy()[order]
        ordered_scores = np.asarray(test_scores, dtype=float)[order]

        for deficit_weight, utility_slack in tqdm(combos, desc=f"{dataset_name} grid", leave=False):
            config = AdaptiveGuardConfig(
                warmup_grid=(10, 25),
                deficit_weight_grid=(deficit_weight,),
                overshoot_weight_grid=(0.75,),
                intervention_penalty_grid=(0.005,),
                deficit_utility_slack_grid=(utility_slack,),
                safe_utility_slack_grid=(0.02,),
                safe_min_gain_grid=(0.02,),
                deficit_min_improvement_grid=(0.0,),
            )
            guard = ProjectedFairnessGuard(
                threshold_baseline=threshold_baseline,
                config=config,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=np.asarray(val_base_preds, dtype=int),
                base_scores=np.asarray(val_scores, dtype=float),
                protected=dataset.protected_val.to_numpy(),
            )
            result = guard.simulate(
                y_true=ordered_truth,
                base_preds=ordered_base_preds,
                base_scores=ordered_scores,
                protected=ordered_protected,
            )
            row = {
                "dataset": dataset_name,
                "model": args.model_name,
                "seed": args.seed,
                "order_protocol": args.order_protocol,
                "deficit_weight": deficit_weight,
                "deficit_utility_slack": utility_slack,
                **result.metrics,
                **result.diagnostics,
                **guard.validation_metrics_,
            }
            rows.append(row)

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "diagnostic_results.csv", index=False)

    summary = (
        result_frame.groupby(["dataset", "deficit_weight", "deficit_utility_slack"], dropna=False)[
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "guard_accept_rate",
                "guard_accept_given_candidate",
                "guard_avg_projected_dpr_gain",
                "rolling_tail_fair_rate",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["dataset", "demographic_parity_ratio", "guard_accept_given_candidate"], ascending=[True, False, False])
    )
    summary.to_csv(output_dir / "diagnostic_summary.csv", index=False)

    with open(output_dir / "diagnostic_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Saved {len(result_frame)} adaptive diagnostic rows to {output_dir / 'diagnostic_results.csv'}")
    print(f"Saved adaptive diagnostic summary to {output_dir / 'diagnostic_summary.csv'}")


if __name__ == "__main__":
    main()
