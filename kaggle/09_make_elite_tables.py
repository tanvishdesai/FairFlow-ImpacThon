"""
Create paper-friendly tables for the upgraded FairFlow experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.statistics import format_interval


def _save_table(frame: pd.DataFrame, path_prefix: Path) -> None:
    frame.to_csv(path_prefix.with_suffix(".csv"), index=False)
    path_prefix.with_suffix(".tex").write_text(frame.to_latex(index=False), encoding="utf-8")


def _intervalize(frame: pd.DataFrame, metric_names: list[str], decimals: int = 3) -> pd.DataFrame:
    result = frame.copy()
    for metric in metric_names:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if mean_col in result.columns and low_col in result.columns and high_col in result.columns:
            result[metric] = result.apply(
                lambda row: format_interval(row[mean_col], row[low_col], row[high_col], decimals=decimals),
                axis=1,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make tables for the upgraded FairFlow experiments.")
    parser.add_argument("--benchmark-aggregated", default="research_outputs/elite_benchmark/aggregated_results.csv")
    parser.add_argument("--benchmark-method-summary", default="research_outputs/elite_benchmark/method_summary.csv")
    parser.add_argument("--benchmark-wins", default="research_outputs/elite_benchmark/paired_win_summary.csv")
    parser.add_argument("--ablation-aggregated", default="research_outputs/elite_guard_ablation/aggregated_results.csv")
    parser.add_argument("--order-aggregated", default="research_outputs/elite_order_stress/aggregated_results.csv")
    parser.add_argument("--diagnostics", default="research_outputs/elite_benchmark/guard_diagnostics.csv")
    parser.add_argument("--output-dir", default="research_outputs/elite_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(args.benchmark_aggregated)
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        readable = _intervalize(
            benchmark,
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ],
        )
        main_table = readable[
            [
                "dataset",
                "model",
                "method",
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "n_runs",
            ]
        ].sort_values(["dataset", "model", "method"])
        _save_table(main_table, output_dir / "table_elite_main")

        compact = readable.pivot_table(
            index=["dataset", "model"],
            columns="method",
            values=["accuracy", "demographic_parity_ratio", "equalized_odds_gap"],
            aggfunc="first",
        ).reset_index()
        compact.columns = ["_".join([str(part) for part in col if part]).strip("_") for col in compact.columns.to_flat_index()]
        _save_table(compact, output_dir / "table_elite_compact")

    method_summary_path = Path(args.benchmark_method_summary)
    if method_summary_path.exists():
        method_summary = pd.read_csv(method_summary_path)
        readable = _intervalize(
            method_summary,
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_fair_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ],
        )
        overall = readable[
            [
                "method",
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_fair_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "n_runs",
            ]
        ].sort_values("method")
        _save_table(overall, output_dir / "table_elite_overall")

    wins_path = Path(args.benchmark_wins)
    if wins_path.exists():
        wins = pd.read_csv(wins_path).sort_values(["metric", "method"])
        _save_table(wins, output_dir / "table_elite_wins")

    ablation_path = Path(args.ablation_aggregated)
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        readable = _intervalize(
            ablation,
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ],
        )
        ablation_table = readable[
            [
                "dataset",
                "method",
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "n_runs",
            ]
        ].sort_values(["dataset", "method"])
        _save_table(ablation_table, output_dir / "table_guard_ablation")

    order_path = Path(args.order_aggregated)
    if order_path.exists():
        order_frame = pd.read_csv(order_path)
        readable = _intervalize(
            order_frame,
            [
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
            ],
        )
        order_table = readable[
            [
                "dataset",
                "method",
                "order_protocol",
                "accuracy",
                "demographic_parity_ratio",
                "equalized_odds_gap",
                "intervention_rate",
                "rolling_tail_avg_dpr",
                "rolling_tail_fair_rate",
                "n_runs",
            ]
        ].sort_values(["dataset", "method", "order_protocol"])
        _save_table(order_table, output_dir / "table_elite_order_stress")

    diagnostics_path = Path(args.diagnostics)
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostic_cols = [
            col
            for col in [
                "guard_activation_rate",
                "guard_activation_events",
                "guard_candidate_rate",
                "guard_accept_rate",
                "guard_accept_given_candidate",
                "guard_avg_projected_dpr_gain",
                "dual_lambda_mean",
                "dual_lambda_max",
                "dual_lambda_final",
                "offset_delta_mean",
                "offset_delta_max",
                "offset_delta_final",
                "guard_rl_override_rate",
            ]
            if col in diagnostics.columns
        ]
        summary = (
            diagnostics.groupby(["dataset", "model", "method"], dropna=False)[diagnostic_cols]
            .mean()
            .reset_index()
            .sort_values(["dataset", "model", "method"])
        )
        _save_table(summary, output_dir / "table_guard_diagnostics")

    print(f"Saved upgraded paper tables to {output_dir}")


if __name__ == "__main__":
    main()
