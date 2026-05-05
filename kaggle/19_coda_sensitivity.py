"""
CODA Hyperparameter Sensitivity Analysis.

Sweeps each of CODA's 6 key hyperparameters one-at-a-time (holding all others
at their validated defaults) and records DPR, Accuracy, and Intervention Rate.

Produces:
  - sensitivity_results.csv   (raw per-run data)
  - sensitivity_summary.csv   (aggregated mean ± CI per hyperparameter value)
  - sensitivity_plot.png      (6-panel figure for the paper)

Usage:
  python kaggle/19_coda_sensitivity.py --search-root data/raw
  python kaggle/19_coda_sensitivity.py --search-root /kaggle/input
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import (
    GroupThresholdOptimizer,
    model_scores,
    train_base_models,
)
from research.coda_controller import (
    CODAConfig,
    CODAController,
    _CODAParams,
    build_coda_with_calibrated_init,
)
from research.datasets import load_dataset
from research.metrics import (
    build_stream_trace,
    compute_static_metrics,
    summarize_stream_trace,
)

# ── Hyperparameter sweep definitions ─────────────────────────────────────────

SWEEP_DEFS: dict[str, dict] = {
    "eta": {
        "label": r"$\eta$ (dual learning rate)",
        "grid_field": "eta_grid",
        "values": (0.01, 0.05, 0.10, 0.25, 0.50, 1.00),
    },
    "ema_alpha": {
        "label": r"$\beta$ (EMA coefficient)",
        "grid_field": "ema_alpha_grid",
        "values": (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00),
    },
    "warmup": {
        "label": r"$w$ (warmup steps)",
        "grid_field": "warmup_grid",
        "values": (5, 15, 30, 50, 100, 200),
    },
    "max_deviation": {
        "label": r"$\delta_{\max}$ (max deviation)",
        "config_field": "max_deviation",
        "values": (0.02, 0.05, 0.08, 0.12, 0.15, 0.20),
    },
    "lambda_cap": {
        "label": r"$\lambda_{\mathrm{cap}}$ (dual cap)",
        "config_field": "lambda_cap",
        "values": (1.0, 2.0, 3.0, 5.0, 8.0, 10.0),
    },
    "accuracy_floor": {
        "label": r"$\alpha_{\mathrm{floor}}$ (accuracy floor)",
        "config_field": "accuracy_floor",
        "values": (0.001, 0.01, 0.02, 0.05, 0.10, 0.20),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CODA hyperparameter sensitivity analysis."
    )
    parser.add_argument(
        "--search-root", action="append",
        default=["/kaggle/input", "data/raw"],
    )
    parser.add_argument(
        "--output-dir",
        default="research_outputs/coda_sensitivity",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["adult", "compas"],
        help="Datasets to sweep on (default: adult, compas — the two with largest CODA impact).",
    )
    parser.add_argument(
        "--model", default="xgboost",
        help="Base model family (default: xgboost).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 52, 62],
    )
    return parser.parse_args()


def _run_coda_with_config(
    *,
    coda_config: CODAConfig,
    threshold_baseline,
    val_scores: np.ndarray,
    val_protected: np.ndarray,
    val_y_true: np.ndarray,
    val_base_preds: np.ndarray,
    test_y_true: np.ndarray,
    test_base_preds: np.ndarray,
    test_scores: np.ndarray,
    test_protected: np.ndarray,
    window_size: int = 50,
    fairness_threshold: float = 0.8,
) -> dict:
    """Fit CODA with a given config and return test metrics."""
    coda = build_coda_with_calibrated_init(
        config=coda_config,
        threshold_baseline=threshold_baseline,
        val_scores=val_scores,
        val_protected=val_protected,
        val_y_true=val_y_true,
    ).fit(
        y_true=val_y_true,
        base_preds=val_base_preds,
        base_scores=val_scores,
        protected=val_protected,
    )

    result = coda.simulate(
        y_true=test_y_true,
        base_preds=test_base_preds,
        base_scores=test_scores,
        protected=test_protected,
    )

    return {**result.metrics, **result.diagnostics}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "sensitivity_config.json", "w", encoding="utf-8") as f:
        json.dump({
            **vars(args),
            "sweeps": {k: list(v["values"]) for k, v in SWEEP_DEFS.items()},
        }, f, indent=2)

    fairness_threshold = 0.8
    window_size = 50
    all_rows: list[dict] = []

    total_runs = (
        len(args.seeds)
        * len(args.datasets)
        * sum(len(sd["values"]) for sd in SWEEP_DEFS.values())
    )
    pbar = tqdm(total=total_runs, desc="Sensitivity sweep")

    for seed in args.seeds:
        for dataset_name in args.datasets:
            dataset = load_dataset(
                dataset_name,
                search_roots=args.search_root,
                random_state=seed,
            )

            models = train_base_models(
                dataset.X_train, dataset.y_train,
                model_names=[args.model], seed=seed,
            )
            model = models[args.model]

            val_scores = model_scores(model, dataset.X_val)
            test_scores = model_scores(model, dataset.X_test)
            val_base_preds = model.predict(dataset.X_val)
            test_base_preds = model.predict(dataset.X_test)

            threshold_baseline = GroupThresholdOptimizer(
                fairness_threshold=fairness_threshold,
            ).fit(
                dataset.y_val.to_numpy(),
                val_scores,
                dataset.protected_val.to_numpy(),
            )

            # ── Sweep each hyperparameter ────────────────────────────────
            for hp_name, sweep_def in SWEEP_DEFS.items():
                for hp_value in sweep_def["values"]:
                    pbar.set_postfix_str(
                        f"{dataset_name}/{seed}/{hp_name}={hp_value}"
                    )

                    # Build a modified CODAConfig
                    cfg = CODAConfig()

                    if "config_field" in sweep_def:
                        # It's a fixed config field (e.g. max_deviation)
                        setattr(cfg, sweep_def["config_field"], hp_value)
                    elif "grid_field" in sweep_def:
                        # It's a grid-searched param — fix to single value
                        setattr(cfg, sweep_def["grid_field"], (hp_value,))

                    try:
                        metrics = _run_coda_with_config(
                            coda_config=cfg,
                            threshold_baseline=threshold_baseline,
                            val_scores=np.asarray(val_scores, dtype=float),
                            val_protected=dataset.protected_val.to_numpy(),
                            val_y_true=dataset.y_val.to_numpy(),
                            val_base_preds=np.asarray(val_base_preds, dtype=int),
                            test_y_true=dataset.y_test.to_numpy(),
                            test_base_preds=np.asarray(test_base_preds, dtype=int),
                            test_scores=np.asarray(test_scores, dtype=float),
                            test_protected=dataset.protected_test.to_numpy(),
                            window_size=window_size,
                            fairness_threshold=fairness_threshold,
                        )
                    except Exception as e:
                        print(
                            f"  [WARN] {hp_name}={hp_value}, "
                            f"{dataset_name}/{args.model}/seed={seed} failed: {e}"
                        )
                        metrics = {
                            "accuracy": float("nan"),
                            "demographic_parity_ratio": float("nan"),
                            "intervention_rate": float("nan"),
                        }

                    all_rows.append({
                        "hyperparameter": hp_name,
                        "hp_value": hp_value,
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": args.model,
                        "accuracy": float(metrics.get("accuracy", float("nan"))),
                        "demographic_parity_ratio": float(
                            metrics.get("demographic_parity_ratio", float("nan"))
                        ),
                        "intervention_rate": float(
                            metrics.get("intervention_rate", float("nan"))
                        ),
                    })
                    pbar.update(1)

    pbar.close()

    # ── Save raw results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_dir / "sensitivity_results.csv", index=False)

    # ── Aggregate: mean ± std per (hyperparameter, hp_value) ─────────────────
    summary_rows = []
    for (hp_name, hp_val), grp in results_df.groupby(
        ["hyperparameter", "hp_value"]
    ):
        n = len(grp)
        for metric in ["accuracy", "demographic_parity_ratio", "intervention_rate"]:
            vals = grp[metric].dropna()
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std())
            ci = 1.96 * std / (len(vals) ** 0.5) if len(vals) > 1 else 0
            summary_rows.append({
                "hyperparameter": hp_name,
                "hp_value": hp_val,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci_low": mean - ci,
                "ci_high": mean + ci,
                "n": int(len(vals)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "sensitivity_summary.csv", index=False)

    # ── Generate plot ────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            "CODA Hyperparameter Sensitivity Analysis",
            fontsize=16, fontweight="bold", y=0.98,
        )

        metric_labels = {
            "demographic_parity_ratio": ("DPR", "tab:blue"),
            "accuracy": ("Accuracy", "tab:green"),
            "intervention_rate": ("IR", "tab:orange"),
        }

        for ax, (hp_name, sweep_def) in zip(axes.flat, SWEEP_DEFS.items()):
            hp_data = summary_df[summary_df["hyperparameter"] == hp_name]

            for metric, (label, color) in metric_labels.items():
                mdata = hp_data[hp_data["metric"] == metric].sort_values("hp_value")
                if mdata.empty:
                    continue
                x = mdata["hp_value"].values
                y = mdata["mean"].values
                lo = mdata["ci_low"].values
                hi = mdata["ci_high"].values

                ax.plot(x, y, "o-", color=color, label=label, markersize=5, linewidth=1.5)
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)

            ax.set_xlabel(sweep_def["label"], fontsize=11)
            ax.set_ylabel("Value", fontsize=10)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

            # Mark default value
            default_cfg = CODAConfig()
            if "config_field" in sweep_def:
                default_val = getattr(default_cfg, sweep_def["config_field"])
                ax.axvline(default_val, color="red", linestyle="--", alpha=0.5, label="default")
            elif "grid_field" in sweep_def:
                # Mark the middle of the default grid
                default_grid = getattr(default_cfg, sweep_def["grid_field"])
                mid_val = default_grid[len(default_grid) // 2]
                ax.axvline(mid_val, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = output_dir / "sensitivity_plot.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSensitivity plot saved to: {plot_path}")

    except ImportError:
        print("\n[WARN] matplotlib not available — skipping plot generation.")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\nSensitivity analysis complete.")
    print(f"  Raw results: {output_dir / 'sensitivity_results.csv'} ({len(results_df)} rows)")
    print(f"  Summary:     {output_dir / 'sensitivity_summary.csv'} ({len(summary_df)} rows)")
    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY (DPR)")
    print("=" * 70)
    dpr_summary = summary_df[
        summary_df["metric"] == "demographic_parity_ratio"
    ][["hyperparameter", "hp_value", "mean", "std", "ci_low", "ci_high", "n"]]
    print(dpr_summary.to_string(index=False))


if __name__ == "__main__":
    main()
