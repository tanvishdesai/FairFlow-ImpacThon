"""
CODA Multi-Group Extension Benchmark.

Demonstrates that CODA generalises to k > 2 protected groups — a key
paper-level contribution that most fairness post-processing methods do not
address.

Compares:
  - base_model (no correction)
  - group_threshold (static per-group thresholds, extended to k groups)
  - CODA v2 (online adaptive, extended to k groups)

Datasets:
  - adult_race: Adult Census with race as 5-group protected attribute
  - recruitment_gender: Recruitment with gender as 3-group protected attribute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.baselines import model_scores, train_base_models
from research.coda_controller import CODAConfig, CODAController
from research.metrics import build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.multigroup_datasets import load_multigroup_dataset, MULTIGROUP_LOADERS
from research.statistics import AggregateSpec, aggregate_with_intervals


# -----------------------------------------------------------------------
# Multi-group group_threshold (grid search over k thresholds)
# -----------------------------------------------------------------------

class MultiGroupThresholdOptimizer:
    """
    Extends GroupThresholdOptimizer to k > 2 groups.

    Searches for per-group thresholds that maximise a fairness-accuracy
    trade-off objective.  For k groups the full grid is exponential, so we
    use a coordinate-descent approach: optimise one group's threshold at
    a time while keeping the others fixed, cycling until convergence.
    """

    def __init__(self, *, fairness_threshold: float = 0.8, n_cycles: int = 5):
        self.fairness_threshold = fairness_threshold
        self.n_cycles = n_cycles
        self.thresholds_: dict[int, float] = {}
        self.validation_metrics_: dict = {}

    def fit(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        protected: np.ndarray,
    ) -> "MultiGroupThresholdOptimizer":
        groups = sorted(set(int(g) for g in np.unique(protected)))
        grid = np.round(np.linspace(0.15, 0.85, 15), 2)

        # Initial: 0.5 for all groups
        thresholds = {g: 0.5 for g in groups}

        for _ in range(self.n_cycles):
            for target_group in groups:
                best_score = -1e18
                best_tau = 0.5
                for tau in grid:
                    thresholds[target_group] = tau
                    preds = self._predict(scores, protected, thresholds)
                    metrics = compute_static_metrics(
                        y_true, preds, protected, scores=scores,
                        fairness_threshold=self.fairness_threshold,
                    )
                    score = self._objective(metrics)
                    if score > best_score:
                        best_score = score
                        best_tau = tau
                thresholds[target_group] = best_tau

        self.thresholds_ = thresholds

        # Compute final metrics
        final_preds = self._predict(scores, protected, self.thresholds_)
        self.validation_metrics_ = compute_static_metrics(
            y_true, final_preds, protected, scores=scores,
            fairness_threshold=self.fairness_threshold,
        )
        return self

    def predict(self, scores: np.ndarray, protected: np.ndarray) -> np.ndarray:
        return self._predict(scores, protected, self.thresholds_)

    @staticmethod
    def _predict(scores, protected, thresholds):
        preds = np.zeros(len(scores), dtype=int)
        for g, tau in thresholds.items():
            mask = protected == g
            preds[mask] = (scores[mask] >= tau).astype(int)
        return preds

    def _objective(self, metrics: dict) -> float:
        dpr = float(metrics["demographic_parity_ratio"])
        acc = float(metrics["accuracy"])
        shortfall = max(0.0, self.fairness_threshold - dpr)
        return acc - 3.0 * shortfall


def _result_row(
    *,
    seed: int,
    dataset_name: str,
    model_name: str,
    method_name: str,
    metrics: dict,
    base_metrics: dict,
    extra: dict | None = None,
) -> dict:
    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "n_groups": int(extra.get("n_groups", 2)) if extra else 2,
        **metrics,
        "accuracy_delta_vs_base": float(metrics["accuracy"] - base_metrics["accuracy"]),
        "dpr_delta_vs_base": float(
            metrics["demographic_parity_ratio"] - base_metrics["demographic_parity_ratio"]
        ),
    }
    if extra:
        row.update(extra)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CODA multi-group extension benchmark.")
    parser.add_argument("--search-root", action="append", default=["/kaggle/input", "data/raw"])
    parser.add_argument("--output-dir", default="research_outputs/coda_multigroup")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(MULTIGROUP_LOADERS.keys()),
    )
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "multigroup_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    fairness_threshold = 0.8
    window_size = 50
    all_rows: list[dict] = []
    diag_rows: list[dict] = []

    for seed in tqdm(args.seeds, desc="Seeds"):
        for dataset_name in tqdm(args.datasets, desc=f"Seed {seed}", leave=False):
            try:
                dataset = load_multigroup_dataset(
                    dataset_name, search_roots=args.search_root, random_state=seed,
                )
            except (FileNotFoundError, KeyError) as exc:
                print(f"  [SKIP] {dataset_name}: {exc}")
                continue

            n_groups = dataset.metadata.get("n_groups", 2)
            group_names = dataset.metadata.get("group_names", {})
            print(f"  {dataset_name}: {n_groups} groups — {group_names}")

            for model_name in args.models:
                models = train_base_models(
                    dataset.X_train, dataset.y_train,
                    model_names=[model_name], seed=seed,
                )
                model = models[model_name]

                val_scores = model_scores(model, dataset.X_val)
                test_scores = model_scores(model, dataset.X_test)
                val_base_preds = model.predict(dataset.X_val)
                test_base_preds = model.predict(dataset.X_test)

                y_test = dataset.y_test.to_numpy()
                prot_test = dataset.protected_test.to_numpy()
                bp_test = np.asarray(test_base_preds, dtype=int)
                sc_test = np.asarray(test_scores, dtype=float)

                extra_info = {"n_groups": n_groups}

                # --- Base model ---
                base_metrics = compute_static_metrics(
                    y_test, bp_test, prot_test,
                    scores=sc_test, fairness_threshold=fairness_threshold,
                )
                base_trace = build_stream_trace(
                    y_test, bp_test, prot_test,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace, warmup=window_size))
                all_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="base_model", metrics=base_metrics, base_metrics=base_metrics,
                    extra=extra_info,
                ))

                # --- Multi-group group_threshold ---
                mg_threshold = MultiGroupThresholdOptimizer(
                    fairness_threshold=fairness_threshold,
                ).fit(
                    dataset.y_val.to_numpy(), val_scores, dataset.protected_val.to_numpy(),
                )
                tp = mg_threshold.predict(sc_test, prot_test).astype(int)
                gt_metrics = compute_static_metrics(
                    y_test, tp, prot_test,
                    scores=sc_test, fairness_threshold=fairness_threshold,
                    intervention_flags=(tp != bp_test).astype(int),
                )
                gt_trace = build_stream_trace(
                    y_test, tp, prot_test,
                    window_size=window_size, fairness_threshold=fairness_threshold,
                )
                gt_metrics.update(summarize_stream_trace(gt_trace, warmup=window_size))
                all_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="multigroup_threshold", metrics=gt_metrics, base_metrics=base_metrics,
                    extra={**extra_info, "thresholds": str(mg_threshold.thresholds_)},
                ))

                # --- CODA (multi-group) ---
                # Initialise CODA from multi-group threshold optimizer
                initial_thresholds = {int(g): float(t) for g, t in mg_threshold.thresholds_.items()}
                coda = CODAController(
                    config=CODAConfig(),
                    initial_thresholds=initial_thresholds,
                )
                coda.fit(
                    y_true=dataset.y_val.to_numpy(),
                    base_preds=np.asarray(val_base_preds, dtype=int),
                    base_scores=np.asarray(val_scores, dtype=float),
                    protected=dataset.protected_val.to_numpy(),
                )

                result = coda.simulate(
                    y_true=y_test,
                    base_preds=bp_test,
                    base_scores=sc_test,
                    protected=prot_test,
                )
                coda_metrics = {**result.metrics, **result.diagnostics}
                all_rows.append(_result_row(
                    seed=seed, dataset_name=dataset_name, model_name=model_name,
                    method_name="coda_multigroup", metrics=coda_metrics, base_metrics=base_metrics,
                    extra=extra_info,
                ))
                diag_rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_groups": n_groups,
                    **result.diagnostics,
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "multigroup_per_run.csv", index=False)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(output_dir / "multigroup_diagnostics.csv", index=False)

    # Aggregate
    summary = aggregate_with_intervals(
        df,
        AggregateSpec(
            group_cols=("dataset", "method"),
            metric_cols=(
                "accuracy", "demographic_parity_ratio", "equalized_odds_gap",
                "intervention_rate", "rolling_tail_avg_dpr", "rolling_tail_fair_rate",
            ),
        ),
    )
    summary.to_csv(output_dir / "multigroup_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("MULTI-GROUP SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
