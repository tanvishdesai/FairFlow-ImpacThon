"""
Run the focused primal-dual FairFlow benchmark and robustness suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.elite_runner import (
    EliteExperimentConfig,
    GuardAblationConfig,
    OrderStressConfig,
    run_elite_benchmark,
    run_guard_ablation,
    run_order_stress_benchmark,
)
from research.elite_methods import PrimalDualOffsetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the focused primal-dual FairFlow benchmark.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-root", default="research_outputs/primal_dual_focus")
    parser.add_argument("--benchmark-datasets", nargs="+", default=["adult", "compas", "recruitment"])
    parser.add_argument("--ablation-datasets", nargs="+", default=["adult", "compas", "recruitment"])
    parser.add_argument("--order-datasets", nargs="+", default=["adult", "compas", "recruitment"])
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["natural", "privileged_first", "unprivileged_first"],
    )
    parser.add_argument("--offset-lambda-lr", type=float, default=None)
    parser.add_argument("--offset-delta-lr", type=float, default=None)
    parser.add_argument("--show-rl-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    benchmark_config = EliteExperimentConfig(
        datasets=args.benchmark_datasets,
        model_names=args.models,
        seeds=args.seeds,
        include_online_group_threshold=True,
    )
    if args.offset_lambda_lr is not None or args.offset_delta_lr is not None:
        lambda_lr = args.offset_lambda_lr if args.offset_lambda_lr is not None else 0.05
        delta_lr = args.offset_delta_lr if args.offset_delta_lr is not None else 0.01
        benchmark_config.primal_dual_offset = PrimalDualOffsetConfig(
            lambda_lr_grid=(float(lambda_lr),),
            delta_lr_grid=(float(delta_lr),),
        )
    benchmark_results, benchmark_traces, benchmark_diagnostics = run_elite_benchmark(
        search_roots=args.search_root,
        output_dir=output_root / "elite_benchmark",
        config=benchmark_config,
        show_rl_progress=args.show_rl_progress,
    )

    ablation_config = GuardAblationConfig(
        datasets=args.ablation_datasets,
        seeds=args.seeds,
        model_name="xgboost",
        benchmark=benchmark_config,
    )
    ablation_results, ablation_traces = run_guard_ablation(
        search_roots=args.search_root,
        output_dir=output_root / "elite_guard_ablation",
        config=ablation_config,
        show_rl_progress=args.show_rl_progress,
    )

    order_config = OrderStressConfig(
        datasets=args.order_datasets,
        seeds=args.seeds,
        model_name="xgboost",
        protocols=args.protocols,
        benchmark=benchmark_config,
    )
    order_results, order_traces = run_order_stress_benchmark(
        search_roots=args.search_root,
        output_dir=output_root / "elite_order_stress",
        config=order_config,
        show_rl_progress=args.show_rl_progress,
    )

    print(f"Focused primal-dual benchmark rows: {len(benchmark_results)}")
    print(f"Focused primal-dual benchmark traces: {len(benchmark_traces)}")
    print(f"Focused primal-dual benchmark diagnostics: {len(benchmark_diagnostics)}")
    print(f"Focused primal-dual ablation rows: {len(ablation_results)}")
    print(f"Focused primal-dual ablation traces: {len(ablation_traces)}")
    print(f"Focused primal-dual order-stress rows: {len(order_results)}")
    print(f"Focused primal-dual order-stress traces: {len(order_traces)}")


if __name__ == "__main__":
    main()
