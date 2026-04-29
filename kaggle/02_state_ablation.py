"""
State ablations for the FairFlow paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research.runner import ExperimentConfig, RLTrainingConfig, run_state_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state ablations for the universal controller.")
    parser.add_argument("--search-root", action="append", default=[], help="Dataset search root.")
    parser.add_argument("--output-dir", default="research_outputs/state_ablation", help="Output directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "german_credit", "compas", "bank_marketing", "recruitment"],
        help="Datasets to evaluate.",
    )
    parser.add_argument("--timesteps", type=int, default=40000, help="Training timesteps per ablation run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        universal_rl=RLTrainingConfig(total_timesteps=args.timesteps, tag="state_ablation"),
    )
    excluded_group_sets = {
        "full_state": [],
        "no_fairness_rates": ["fairness_rates"],
        "no_intervention_history": ["intervention_history"],
        "no_stream_context": ["stream_context"],
        "no_confidence_gap": ["confidence_gap"],
    }
    result = run_state_ablation(
        search_roots=args.search_root or ["data/raw", "/kaggle/input", "."],
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        excluded_group_sets=excluded_group_sets,
        config=config,
    )
    print(f"Saved {len(result)} rows to {Path(args.output_dir) / 'state_ablation_results.csv'}")


if __name__ == "__main__":
    main()

