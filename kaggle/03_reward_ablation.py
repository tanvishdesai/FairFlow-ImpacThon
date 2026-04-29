"""
Reward ablations for the FairFlow paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research.runner import ExperimentConfig, RLTrainingConfig, run_reward_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reward ablations for the universal controller.")
    parser.add_argument("--search-root", action="append", default=[], help="Dataset search root.")
    parser.add_argument("--output-dir", default="research_outputs/reward_ablation", help="Output directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "german_credit", "compas", "bank_marketing", "recruitment"],
        help="Datasets to evaluate.",
    )
    parser.add_argument("--timesteps", type=int, default=40000, help="Training timesteps per reward variant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        universal_rl=RLTrainingConfig(total_timesteps=args.timesteps, tag="reward_ablation"),
    )
    reward_settings = {
        "accuracy_heavy": {"accuracy_weight": 0.60, "fairness_weight": 0.40, "intervention_penalty": -0.03},
        "balanced": {"accuracy_weight": 0.50, "fairness_weight": 0.50, "intervention_penalty": -0.05},
        "fairness_heavy": {"accuracy_weight": 0.35, "fairness_weight": 0.65, "intervention_penalty": -0.05},
        "fairness_low_penalty": {"accuracy_weight": 0.35, "fairness_weight": 0.65, "intervention_penalty": -0.01},
    }
    result = run_reward_ablation(
        search_roots=args.search_root or ["data/raw", "/kaggle/input", "."],
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        reward_settings=reward_settings,
        config=config,
    )
    print(f"Saved {len(result)} rows to {Path(args.output_dir) / 'reward_ablation_results.csv'}")


if __name__ == "__main__":
    main()

