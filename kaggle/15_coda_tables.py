"""
Generate paper-quality comparison tables from CODA benchmark results.

Uses the same formatting conventions as 09_make_elite_tables.py for
consistency with the rest of the paper.
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
                lambda row, m=mean_col, l=low_col, h=high_col: format_interval(
                    row[m], row[l], row[h], decimals=decimals
                ),
                axis=1,
            )
    return result


CORE_METRICS = [
    "accuracy",
    "demographic_parity_ratio",
    "equalized_odds_gap",
    "intervention_rate",
    "rolling_tail_avg_dpr",
    "rolling_tail_fair_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CODA comparison tables.")
    parser.add_argument("--input-dir", default="research_outputs/coda_v2_benchmark")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating CODA tables from: {input_dir}")
    print(f"Saving tables to: {output_dir}")

    # ---- 1. Overall method summary ----
    method_summary_path = input_dir / "method_summary.csv"
    if method_summary_path.exists():
        method_summary = pd.read_csv(method_summary_path)
        readable = _intervalize(method_summary, CORE_METRICS + ["rolling_fair_rate"])
        display_cols = [c for c in ["method"] + CORE_METRICS + ["rolling_fair_rate", "n_runs"] if c in readable.columns]
        overall = readable[display_cols].sort_values("method")
        _save_table(overall, output_dir / "table_coda_overall")
        print("  [OK] table_coda_overall")
    else:
        print(f"  [SKIP] method_summary.csv not found")

    # ---- 2. Per-dataset summary ----
    per_dataset_path = input_dir / "per_dataset_summary.csv"
    if per_dataset_path.exists():
        per_dataset = pd.read_csv(per_dataset_path)
        readable = _intervalize(per_dataset, CORE_METRICS)
        display_cols = [c for c in ["dataset", "method"] + CORE_METRICS + ["n_runs"] if c in readable.columns]
        main_table = readable[display_cols].sort_values(["dataset", "method"])
        _save_table(main_table, output_dir / "table_coda_per_dataset")
        print("  [OK] table_coda_per_dataset")
    else:
        print(f"  [SKIP] per_dataset_summary.csv not found")

    # ---- 3. Paired win counts ----
    wins_path = input_dir / "paired_win_summary.csv"
    if wins_path.exists():
        wins = pd.read_csv(wins_path).sort_values(["metric", "method"])
        _save_table(wins, output_dir / "table_coda_wins")
        print("  [OK] table_coda_wins")
    else:
        print(f"  [SKIP] paired_win_summary.csv not found")

    # ---- 4. Order-stress table ----
    order_path = input_dir / "order_stress_summary.csv"
    if order_path.exists():
        order_frame = pd.read_csv(order_path)
        readable = _intervalize(order_frame, CORE_METRICS)
        display_cols = [
            c for c in ["dataset", "method", "order_protocol"] + CORE_METRICS + ["n_runs"]
            if c in readable.columns
        ]
        order_table = readable[display_cols].sort_values(["dataset", "method", "order_protocol"])
        _save_table(order_table, output_dir / "table_coda_order_stress")
        print("  [OK] table_coda_order_stress")
    else:
        print(f"  [SKIP] order_stress_summary.csv not found")

    # ---- 5. CODA diagnostics ----
    diag_path = input_dir / "coda_diagnostics.csv"
    if diag_path.exists():
        diag_df = pd.read_csv(diag_path)
        diagnostic_cols = [
            col for col in [
                "coda_threshold_updates",
                "coda_lambda_mean",
                "coda_lambda_max",
                "coda_lambda_final",
                "coda_dpr_mean",
                "coda_dpr_std",
                "coda_tau_final_group_0",
                "coda_tau_final_group_1",
                "coda_tau_mean_group_0",
                "coda_tau_mean_group_1",
                "coda_tau_range_group_0",
                "coda_tau_range_group_1",
            ]
            if col in diag_df.columns
        ]
        id_cols = [c for c in ["dataset", "model", "seed"] if c in diag_df.columns]
        summary = (
            diag_df.groupby(id_cols[:2] if len(id_cols) >= 2 else id_cols, dropna=False)[diagnostic_cols]
            .mean()
            .reset_index()
        )
        _save_table(summary, output_dir / "table_coda_diagnostics")
        print("  [OK] table_coda_diagnostics")
    else:
        print(f"  [SKIP] coda_diagnostics.csv not found")

    print(f"\nAll tables saved to {output_dir}")


if __name__ == "__main__":
    main()
