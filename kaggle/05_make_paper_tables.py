"""
Create paper-ready tables from benchmark and ablation outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _save_table(frame: pd.DataFrame, path_prefix: Path) -> None:
    frame.to_csv(path_prefix.with_suffix(".csv"), index=False)
    path_prefix.with_suffix(".tex").write_text(frame.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-friendly result tables.")
    parser.add_argument("--main-results", default="research_outputs/main_benchmark/main_results.csv")
    parser.add_argument("--state-results", default="research_outputs/state_ablation/state_ablation_results.csv")
    parser.add_argument("--reward-results", default="research_outputs/reward_ablation/reward_ablation_results.csv")
    parser.add_argument("--order-results", default="research_outputs/order_stress/order_stress_results.csv")
    parser.add_argument("--output-dir", default="research_outputs/paper_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results = Path(args.main_results)
    if main_results.exists():
        frame = pd.read_csv(main_results)
        columns = [
            "dataset",
            "model",
            "method",
            "order_protocol",
            "accuracy",
            "demographic_parity_ratio",
            "equalized_odds_gap",
            "intervention_rate",
            "rolling_min_dpr",
            "rolling_fair_rate",
            "accuracy_delta_vs_base",
            "dpr_delta_vs_base",
        ]
        summary = frame[columns].sort_values(["dataset", "model", "method"])
        _save_table(summary, output_dir / "table_main_results")

        compact = (
            frame[frame["order_protocol"] == "natural"]
            .pivot_table(
                index=["dataset", "model"],
                columns="method",
                values=["accuracy", "demographic_parity_ratio", "intervention_rate"],
                aggfunc="first",
            )
            .reset_index()
        )
        compact.columns = ["_".join([str(part) for part in col if part]).strip("_") for col in compact.columns.to_flat_index()]
        _save_table(compact, output_dir / "table_compact_comparison")

    state_results = Path(args.state_results)
    if state_results.exists():
        frame = pd.read_csv(state_results)
        summary = frame[
            ["ablation", "dataset", "accuracy", "demographic_parity_ratio", "equalized_odds_gap", "intervention_rate", "rolling_min_dpr"]
        ].sort_values(["ablation", "dataset"])
        _save_table(summary, output_dir / "table_state_ablation")

    reward_results = Path(args.reward_results)
    if reward_results.exists():
        frame = pd.read_csv(reward_results)
        summary = frame[
            ["reward_variant", "dataset", "accuracy", "demographic_parity_ratio", "equalized_odds_gap", "intervention_rate", "rolling_min_dpr"]
        ].sort_values(["reward_variant", "dataset"])
        _save_table(summary, output_dir / "table_reward_ablation")

    order_results = Path(args.order_results)
    if order_results.exists():
        frame = pd.read_csv(order_results)
        summary = frame[
            ["dataset", "order_protocol", "accuracy", "demographic_parity_ratio", "equalized_odds_gap", "intervention_rate", "rolling_min_dpr", "rolling_fair_rate"]
        ].sort_values(["dataset", "order_protocol"])
        _save_table(summary, output_dir / "table_order_stress")

    print(f"Saved paper tables to {output_dir}")


if __name__ == "__main__":
    main()

