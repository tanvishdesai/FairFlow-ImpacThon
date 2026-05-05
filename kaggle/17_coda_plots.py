"""
CODA Dynamics Visualization.

Generates publication-quality plots of CODA's internal state:
1. Per-group threshold trajectories over time
2. Dual variable (λ) evolution
3. Rolling DPR over time — CODA vs group_threshold vs base_model
4. Intervention timeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import (
    GroupThresholdOptimizer,
    model_scores,
    train_base_models,
)
from research.coda_controller import CODAConfig, CODAController, build_coda_with_calibrated_init
from research.datasets import load_dataset
from research.metrics import build_stream_trace


# ---------- Style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
COLORS = {
    "group_0": "#e63946",
    "group_1": "#457b9d",
    "lambda": "#f77f00",
    "dpr_coda": "#2a9d8f",
    "dpr_gt": "#264653",
    "dpr_base": "#adb5bd",
    "intervention": "#e76f51",
    "fairness_target": "#6c757d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CODA dynamics plots.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/coda_plots")
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _smooth(arr: np.ndarray, window: int = 25) -> np.ndarray:
    """Simple moving average for plot smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots for {args.dataset} / {args.model} / seed={args.seed}")
    dataset = load_dataset(args.dataset, search_roots=args.search_root, random_state=args.seed)

    models = train_base_models(
        dataset.X_train, dataset.y_train,
        model_names=[args.model], seed=args.seed,
    )
    model = models[args.model]

    val_scores = model_scores(model, dataset.X_val)
    test_scores = model_scores(model, dataset.X_test)
    val_base_preds = model.predict(dataset.X_val)
    test_base_preds = model.predict(dataset.X_test)

    y_test = dataset.y_test.to_numpy()
    prot_test = dataset.protected_test.to_numpy()
    bp_test = np.asarray(test_base_preds, dtype=int)
    sc_test = np.asarray(test_scores, dtype=float)

    # Fit baselines
    threshold_baseline = GroupThresholdOptimizer(fairness_threshold=0.8).fit(
        dataset.y_val.to_numpy(), val_scores, dataset.protected_val.to_numpy(),
    )

    coda = build_coda_with_calibrated_init(
        config=CODAConfig(),
        threshold_baseline=threshold_baseline,
        val_scores=np.asarray(val_scores, dtype=float),
        val_protected=dataset.protected_val.to_numpy(),
        val_y_true=dataset.y_val.to_numpy(),
    ).fit(
        y_true=dataset.y_val.to_numpy(),
        base_preds=np.asarray(val_base_preds, dtype=int),
        base_scores=np.asarray(val_scores, dtype=float),
        protected=dataset.protected_val.to_numpy(),
    )

    # Simulate CODA on test
    result = coda.simulate(
        y_true=y_test,
        base_preds=bp_test,
        base_scores=sc_test,
        protected=prot_test,
    )

    # Also get group_threshold and base model traces
    gt_preds = threshold_baseline.predict(sc_test, prot_test).astype(int)
    base_trace = build_stream_trace(y_test, bp_test, prot_test, window_size=50, fairness_threshold=0.8)
    gt_trace = build_stream_trace(y_test, gt_preds, prot_test, window_size=50, fairness_threshold=0.8)
    coda_trace = result.trace

    n = len(y_test)
    timesteps = np.arange(n)

    # ================================================================
    # PLOT 1: Per-group threshold trajectories
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = sorted(result.threshold_history.keys())
    for g in groups:
        tau_arr = np.array(result.threshold_history[g])
        label = f"τ (group {g}: {'privileged' if g == 1 else 'unprivileged'})"
        color = COLORS["group_1"] if g == 1 else COLORS["group_0"]
        ax.plot(timesteps, tau_arr, color=color, alpha=0.3, linewidth=0.5)
        ax.plot(timesteps, _smooth(tau_arr, 50), color=color, linewidth=2, label=label)

    # Add initial threshold lines
    for g in groups:
        init_val = coda.initial_thresholds.get(g, 0.5)
        color = COLORS["group_1"] if g == 1 else COLORS["group_0"]
        ax.axhline(init_val, color=color, linestyle="--", alpha=0.5, linewidth=1,
                    label=f"τ₀ (group {g}) = {init_val:.2f}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Decision Threshold τ")
    ax.set_title(f"CODA Threshold Trajectories — {args.dataset.title()}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"threshold_trajectories_{args.dataset}.png")
    plt.close(fig)
    print(f"  [OK] threshold_trajectories_{args.dataset}.png")

    # ================================================================
    # PLOT 2: Dual variable λ evolution
    # ================================================================
    # Re-simulate to capture lambda history (we need to access internal state)
    # We'll reconstruct it from the diagnostics and the simulate method
    # For this we need to re-run simulate and capture lambda
    # Actually, let's modify our approach: recalculate lambda from the DPR trace

    # Lambda reconstruction from DPR trace
    params = coda.params_
    config = coda.config
    dpr_values = coda_trace["rolling_dpr"].values if "rolling_dpr" in coda_trace.columns else np.ones(n)

    lambda_reconstructed = np.zeros(n)
    lam = 0.0
    for i in range(n):
        if i >= params.warmup_steps:
            dpr_val = dpr_values[i] if i < len(dpr_values) else 1.0
            if not np.isnan(dpr_val):
                violation = config.target_dpr - dpr_val
                lam = max(0.0, lam + params.eta * violation)
                lam = min(lam, config.lambda_cap)
                if dpr_val >= config.target_dpr:
                    lam *= config.lambda_decay
        lambda_reconstructed[i] = lam

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(timesteps, lambda_reconstructed, color=COLORS["lambda"], linewidth=1.5, alpha=0.4)
    ax.plot(timesteps, _smooth(lambda_reconstructed, 50), color=COLORS["lambda"], linewidth=2.5, label="λ(t) — dual variable")
    ax.axhline(config.lambda_cap, color=COLORS["fairness_target"], linestyle=":", alpha=0.7, label=f"λ cap = {config.lambda_cap}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Dual Variable λ")
    ax.set_title(f"CODA Dual Variable Evolution — {args.dataset.title()}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"lambda_evolution_{args.dataset}.png")
    plt.close(fig)
    print(f"  [OK] lambda_evolution_{args.dataset}.png")

    # ================================================================
    # PLOT 3: Rolling DPR comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 4))

    for trace_df, name, color, lw in [
        (base_trace, "Base Model", COLORS["dpr_base"], 1.5),
        (gt_trace, "Group Threshold", COLORS["dpr_gt"], 2.0),
        (coda_trace, "CODA v2", COLORS["dpr_coda"], 2.5),
    ]:
        if "rolling_dpr" in trace_df.columns:
            dpr = trace_df["rolling_dpr"].values
            ts = np.arange(len(dpr))
            ax.plot(ts, _smooth(dpr, 30), color=color, linewidth=lw, label=name)

    ax.axhline(0.8, color=COLORS["fairness_target"], linestyle="--", alpha=0.7, linewidth=1, label="DPR = 0.80 (threshold)")
    ax.axhline(1.0, color="#2d6a4f", linestyle=":", alpha=0.5, linewidth=1, label="DPR = 1.00 (perfect parity)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rolling DPR")
    ax.set_title(f"Rolling Demographic Parity Ratio — {args.dataset.title()}")
    ax.set_ylim(-0.05, 1.4)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"rolling_dpr_{args.dataset}.png")
    plt.close(fig)
    print(f"  [OK] rolling_dpr_{args.dataset}.png")

    # ================================================================
    # PLOT 4: Intervention timeline
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 2.5))
    intervention_points = np.where(result.interventions == 1)[0]
    ax.scatter(intervention_points, np.ones(len(intervention_points)),
               color=COLORS["intervention"], s=1, alpha=0.3, marker="|")

    # Rolling intervention rate
    window = 100
    rolling_rate = pd.Series(result.interventions).rolling(window, min_periods=1).mean().values
    ax.plot(timesteps, rolling_rate, color=COLORS["intervention"], linewidth=2,
            label=f"Rolling intervention rate (w={window})")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Intervention Rate")
    ax.set_title(f"CODA Intervention Timeline — {args.dataset.title()}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.5)
    fig.savefig(output_dir / f"intervention_timeline_{args.dataset}.png")
    plt.close(fig)
    print(f"  [OK] intervention_timeline_{args.dataset}.png")

    # ================================================================
    # PLOT 5: Combined 2x2 figure for the paper
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: Thresholds
    ax = axes[0, 0]
    for g in groups:
        tau_arr = np.array(result.threshold_history[g])
        color = COLORS["group_1"] if g == 1 else COLORS["group_0"]
        label = f"τ ({'priv.' if g == 1 else 'unpriv.'})"
        ax.plot(timesteps, _smooth(tau_arr, 50), color=color, linewidth=2, label=label)
        init_val = coda.initial_thresholds.get(g, 0.5)
        ax.axhline(init_val, color=color, linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("(a) Per-Group Threshold Trajectories")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Threshold τ")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: Lambda
    ax = axes[0, 1]
    ax.plot(timesteps, _smooth(lambda_reconstructed, 50), color=COLORS["lambda"], linewidth=2.5)
    ax.axhline(config.lambda_cap, color=COLORS["fairness_target"], linestyle=":", alpha=0.7)
    ax.set_title("(b) Dual Variable λ Evolution")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("λ(t)")
    ax.grid(True, alpha=0.3)

    # Panel C: Rolling DPR
    ax = axes[1, 0]
    for trace_df, name, color, lw in [
        (base_trace, "Base", COLORS["dpr_base"], 1.5),
        (gt_trace, "Group Threshold", COLORS["dpr_gt"], 2.0),
        (coda_trace, "CODA v2", COLORS["dpr_coda"], 2.5),
    ]:
        if "rolling_dpr" in trace_df.columns:
            dpr = trace_df["rolling_dpr"].values
            ax.plot(np.arange(len(dpr)), _smooth(dpr, 30), color=color, linewidth=lw, label=name)
    ax.axhline(0.8, color=COLORS["fairness_target"], linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(1.0, color="#2d6a4f", linestyle=":", alpha=0.3, linewidth=1)
    ax.set_title("(c) Rolling DPR Comparison")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rolling DPR")
    ax.set_ylim(-0.05, 1.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel D: Intervention
    ax = axes[1, 1]
    rolling_rate = pd.Series(result.interventions).rolling(100, min_periods=1).mean().values
    ax.plot(timesteps, rolling_rate, color=COLORS["intervention"], linewidth=2, label="CODA intervention rate")
    gt_interventions = (gt_preds != bp_test).astype(int)
    gt_rolling = pd.Series(gt_interventions).rolling(100, min_periods=1).mean().values
    ax.plot(timesteps, gt_rolling, color=COLORS["dpr_gt"], linewidth=2, linestyle="--", label="Group threshold rate")
    ax.set_title("(d) Rolling Intervention Rate")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Intervention Rate")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.5)

    fig.suptitle(f"CODA v2 Dynamics — {args.dataset.title()} ({args.model})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / f"coda_dynamics_combined_{args.dataset}.png")
    plt.close(fig)
    print(f"  [OK] coda_dynamics_combined_{args.dataset}.png")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
