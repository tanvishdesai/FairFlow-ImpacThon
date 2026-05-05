"""
Run a focused tuning sweep for the primal-dual offset controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import model_scores, train_base_models
from research.datasets import load_dataset
from research.elite_methods import PrimalDualOffsetConfig, PrimalDualOffsetController
from research.metrics import summarize_stream_trace
from research.rl import make_order_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the primal-dual offset FairFlow controller.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/primal_dual_offset_tuning")
    parser.add_argument("--datasets", nargs="+", default=["adult", "compas", "recruitment"])
    parser.add_argument("--core-datasets", nargs="+", default=["adult", "compas"])
    parser.add_argument("--model-name", default="xgboost")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--lambda-lrs", nargs="+", type=float, default=[0.03, 0.05, 0.08])
    parser.add_argument("--delta-lrs", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--order-protocol", default="natural")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combos = [(float(lambda_lr), float(delta_lr)) for lambda_lr, delta_lr in product(args.lambda_lrs, args.delta_lrs)]
    rows: list[dict] = []

    for seed in tqdm(args.seeds, desc="Tuning seeds"):
        for dataset_name in tqdm(args.datasets, desc=f"Seed {seed} datasets", leave=False):
            dataset = load_dataset(dataset_name, search_roots=args.search_root, random_state=seed)
            model = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=[args.model_name],
                seed=seed,
            )[args.model_name]

            val_scores = model_scores(model, dataset.X_val)
            test_scores = model_scores(model, dataset.X_test)
            val_base_preds = model.predict(dataset.X_val)
            test_base_preds = model.predict(dataset.X_test)

            order = make_order_indices(
                dataset.protected_test.to_numpy(),
                protocol=args.order_protocol,
                seed=seed,
            )
            ordered_truth = dataset.y_test.to_numpy()[order]
            ordered_base_preds = np.asarray(test_base_preds, dtype=int)[order]
            ordered_protected = dataset.protected_test.to_numpy()[order]
            ordered_scores = np.asarray(test_scores, dtype=float)[order]

            for lambda_lr, delta_lr in tqdm(combos, desc=f"{dataset_name} offset grid", leave=False):
                config = PrimalDualOffsetConfig(
                    lambda_lr_grid=(lambda_lr,),
                    delta_lr_grid=(delta_lr,),
                )
                controller = PrimalDualOffsetController(config=config).fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )
                result = controller.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                trace_summary = summarize_stream_trace(result.trace, warmup=config.window_size)
                metrics = {
                    **result.metrics,
                    **trace_summary,
                }
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": args.model_name,
                        "seed": seed,
                        "order_protocol": args.order_protocol,
                        "lambda_lr": lambda_lr,
                        "delta_lr": delta_lr,
                        **metrics,
                        **result.diagnostics,
                        **{f"val_{key}": value for key, value in controller.validation_metrics_.items()},
                    }
                )

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "combo_results.csv", index=False)

    dataset_summary = (
        result_frame.groupby(["dataset", "lambda_lr", "delta_lr"], dropna=False)[
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "guard_accept_given_candidate",
                "dual_lambda_final",
                "offset_delta_final",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["dataset", "demographic_parity_ratio", "intervention_rate"], ascending=[True, False, True])
    )
    dataset_summary.to_csv(output_dir / "combo_by_dataset.csv", index=False)

    core_frame = result_frame[result_frame["dataset"].isin(args.core_datasets)]
    core_summary = (
        core_frame.groupby(["lambda_lr", "delta_lr"], dropna=False)[
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "core_accuracy_mean",
                "demographic_parity_ratio": "core_dpr_mean",
                "equalized_odds_gap": "core_eo_gap_mean",
                "intervention_rate": "core_intervention_mean",
                "rolling_tail_avg_dpr": "core_tail_dpr_mean",
                "rolling_tail_fair_rate": "core_tail_fair_mean",
            }
        )
    )

    overall_summary = (
        result_frame.groupby(["lambda_lr", "delta_lr"], dropna=False)[
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "overall_accuracy_mean",
                "demographic_parity_ratio": "overall_dpr_mean",
                "equalized_odds_gap": "overall_eo_gap_mean",
                "intervention_rate": "overall_intervention_mean",
                "rolling_tail_avg_dpr": "overall_tail_dpr_mean",
                "rolling_tail_fair_rate": "overall_tail_fair_mean",
            }
        )
    )

    recruitment_summary = (
        result_frame[result_frame["dataset"] == "recruitment"]
        .groupby(["lambda_lr", "delta_lr"], dropna=False)[
            ["accuracy", "demographic_parity_ratio", "intervention_rate"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "recruitment_accuracy_mean",
                "demographic_parity_ratio": "recruitment_dpr_mean",
                "intervention_rate": "recruitment_intervention_mean",
            }
        )
    )

    selection = (
        core_summary.merge(overall_summary, on=["lambda_lr", "delta_lr"], how="left")
        .merge(recruitment_summary, on=["lambda_lr", "delta_lr"], how="left")
    )
    selection["meets_intervention_budget"] = selection["core_intervention_mean"] <= 0.08
    selection["meets_fairness_target"] = selection["core_dpr_mean"] >= 0.85
    selection = selection.sort_values(
        [
            "meets_fairness_target",
            "meets_intervention_budget",
            "core_dpr_mean",
            "core_intervention_mean",
            "overall_accuracy_mean",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    selection["rank"] = np.arange(1, len(selection) + 1)
    selection.to_csv(output_dir / "selection_table.csv", index=False)

    best_row = selection.iloc[0].to_dict()
    with open(output_dir / "best_offset_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "recommended_lambda_lr": float(best_row["lambda_lr"]),
                "recommended_delta_lr": float(best_row["delta_lr"]),
                "selection_metrics": best_row,
                "datasets": args.datasets,
                "core_datasets": args.core_datasets,
                "model_name": args.model_name,
                "seeds": args.seeds,
                "order_protocol": args.order_protocol,
            },
            handle,
            indent=2,
        )

    with open(output_dir / "tuning_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Saved {len(result_frame)} tuning rows to {output_dir / 'combo_results.csv'}")
    print(f"Saved dataset-wise tuning summary to {output_dir / 'combo_by_dataset.csv'}")
    print(f"Saved ranking table to {output_dir / 'selection_table.csv'}")
    print(f"Saved recommended configuration to {output_dir / 'best_offset_config.json'}")


if __name__ == "__main__":
    main()
