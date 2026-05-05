"""
Run the upgraded multi-seed FairFlow benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.elite_runner import EliteExperimentConfig, run_elite_benchmark
from research.elite_methods import PrimalDualOffsetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the upgraded FairFlow benchmark.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/elite_benchmark")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german_credit", "compas", "bank_marketing", "recruitment"])
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--offset-lambda-lr", type=float, default=None)
    parser.add_argument("--offset-delta-lr", type=float, default=None)
    parser.add_argument("--show-rl-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EliteExperimentConfig(
        datasets=args.datasets,
        model_names=args.models,
        seeds=args.seeds,
    )
    if args.offset_lambda_lr is not None or args.offset_delta_lr is not None:
        lambda_lr = args.offset_lambda_lr if args.offset_lambda_lr is not None else 0.05
        delta_lr = args.offset_delta_lr if args.offset_delta_lr is not None else 0.01
        config.primal_dual_offset = PrimalDualOffsetConfig(
            lambda_lr_grid=(float(lambda_lr),),
            delta_lr_grid=(float(delta_lr),),
        )
    results, traces, diagnostics = run_elite_benchmark(
        search_roots=args.search_root,
        output_dir=args.output_dir,
        config=config,
        show_rl_progress=args.show_rl_progress,
    )
    print(f"Saved {len(results)} elite benchmark rows to {args.output_dir}/per_run_results.csv")
    print(f"Saved {len(traces)} elite rolling-trace rows to {args.output_dir}/rolling_traces.csv")
    print(f"Saved {len(diagnostics)} guard diagnostic rows to {args.output_dir}/guard_diagnostics.csv")


if __name__ == "__main__":
    main()
