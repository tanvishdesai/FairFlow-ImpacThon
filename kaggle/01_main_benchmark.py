"""
Main FairFlow paper benchmark for Kaggle or local runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research.runner import ExperimentConfig, RLTrainingConfig, run_main_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main FairFlow paper benchmark.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Directory to search for raw dataset files. Repeat this flag for multiple roots.",
    )
    parser.add_argument(
        "--output-dir",
        default="research_outputs/main_benchmark",
        help="Directory where results, traces, and trained models will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "german_credit", "compas", "bank_marketing", "recruitment"],
        help="Datasets to include.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "xgboost"],
        help="Base models to train.",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["natural"],
        help="Order protocols to evaluate in the main benchmark.",
    )
    parser.add_argument(
        "--universal-timesteps",
        type=int,
        default=60000,
        help="Training timesteps for the universal RL controller.",
    )
    parser.add_argument(
        "--dataset-specific-timesteps",
        type=int,
        default=30000,
        help="Training timesteps for dataset-specific RL controllers.",
    )
    parser.add_argument(
        "--dataset-specific-models",
        nargs="+",
        default=["xgboost"],
        help="Base-model families that should also get dataset-specific RL controllers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    search_roots = args.search_root or ["data/raw", "/kaggle/input", "."]
    config = ExperimentConfig(
        datasets=args.datasets,
        model_names=args.models,
        dataset_specific_models=args.dataset_specific_models,
        order_protocols=args.orders,
        universal_rl=RLTrainingConfig(
            total_timesteps=args.universal_timesteps,
            tag="main_universal",
        ),
        dataset_specific_rl=RLTrainingConfig(
            total_timesteps=args.dataset_specific_timesteps,
            n_steps=512,
            batch_size=128,
            tag="main_dataset_specific",
        ),
    )
    results, traces = run_main_benchmark(
        search_roots=search_roots,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"Saved {len(results)} benchmark rows to {Path(args.output_dir) / 'main_results.csv'}")
    print(f"Saved {len(traces)} rolling-trace rows to {Path(args.output_dir) / 'rolling_traces.csv'}")


if __name__ == "__main__":
    main()

