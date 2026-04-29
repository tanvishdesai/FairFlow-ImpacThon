"""
Order-sensitivity evaluation for the FairFlow paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research.runner import ExperimentConfig, RLTrainingConfig, run_order_stress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run order-sensitivity stress tests.")
    parser.add_argument("--search-root", action="append", default=[], help="Dataset search root.")
    parser.add_argument("--output-dir", default="research_outputs/order_stress", help="Output directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "german_credit", "compas", "bank_marketing", "recruitment"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["natural", "random_seed_0", "privileged_first", "unprivileged_first", "alternating_groups", "burst_priv_then_unpriv"],
        help="Order protocols to evaluate.",
    )
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps for the universal controller.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        universal_rl=RLTrainingConfig(total_timesteps=args.timesteps, tag="order_stress"),
    )
    protocols = [item if item != "random_seed_0" else "random_0" for item in args.orders]
    result = run_order_stress(
        search_roots=args.search_root or ["data/raw", "/kaggle/input", "."],
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        protocols=protocols,
        config=config,
    )
    print(f"Saved {len(result)} rows to {Path(args.output_dir) / 'order_stress_results.csv'}")


if __name__ == "__main__":
    main()

