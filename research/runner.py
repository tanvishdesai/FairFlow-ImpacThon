"""
Benchmark and ablation runners for the FairFlow paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from research.baselines import GroupThresholdOptimizer, RuleBasedController, evaluate_static_method, model_scores, train_base_models
from research.dataset_catalog import dataset_catalog_rows
from research.datasets import available_dataset_names, load_dataset
from research.metrics import build_stream_trace, compute_static_metrics, summarize_stream_trace
from research.rl import RLTrainingConfig, evaluate_rl_controller, make_order_indices, train_dataset_specific_controller, train_universal_controller


@dataclass
class ExperimentConfig:
    datasets: list[str] = field(default_factory=lambda: ["adult", "german_credit", "compas", "bank_marketing", "recruitment"])
    model_names: list[str] = field(default_factory=lambda: ["logistic_regression", "random_forest", "xgboost"])
    dataset_specific_models: list[str] = field(default_factory=lambda: ["xgboost"])
    fairness_threshold: float = 0.8
    window_size: int = 50
    random_seed: int = 42
    order_protocols: list[str] = field(default_factory=lambda: ["natural"])
    train_universal_controller: bool = True
    overwrite_existing: bool = False
    universal_rl: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    dataset_specific_rl: RLTrainingConfig = field(
        default_factory=lambda: RLTrainingConfig(
            total_timesteps=30_000,
            n_steps=512,
            batch_size=128,
            tag="dataset_specific",
        )
    )


def _ordered_arrays(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    scores: np.ndarray,
    order_protocol: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = make_order_indices(protected, protocol=order_protocol, seed=seed)
    return y_true[order], y_pred[order], protected[order], scores[order]


def _baseline_row(
    *,
    dataset_name: str,
    model_name: str,
    method_name: str,
    order_protocol: str,
    metrics: dict,
    base_metrics: dict,
) -> dict:
    row = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "order_protocol": order_protocol,
        **metrics,
    }
    row["accuracy_delta_vs_base"] = row["accuracy"] - base_metrics["accuracy"]
    row["dpr_delta_vs_base"] = row["demographic_parity_ratio"] - base_metrics["demographic_parity_ratio"]
    return row


def run_main_benchmark(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the main benchmark suite and save paper-friendly outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)

    with open(output_path / "dataset_catalog.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_catalog_rows(), handle, indent=2)
    with open(output_path / "benchmark_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)

    result_rows: list[dict] = []
    trace_rows: list[dict] = []
    universal_model = None

    if config.train_universal_controller:
        universal_dir = output_path / "models" / "universal"
        universal_model = train_universal_controller(
            output_dir=universal_dir,
            config=config.universal_rl,
            curriculum=True,
        )

    for dataset_name in config.datasets:
        try:
            dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=config.random_seed)
        except FileNotFoundError:
            continue

        dataset_dir = output_path / "models" / dataset_name
        base_models = train_base_models(
            dataset.X_train,
            dataset.y_train,
            model_names=config.model_names,
            seed=config.random_seed,
            output_dir=dataset_dir,
        )

        for model_name, model in base_models.items():
            val_scores = model_scores(model, dataset.X_val)
            test_scores = model_scores(model, dataset.X_test)
            val_base_preds = model.predict(dataset.X_val)
            test_base_preds = model.predict(dataset.X_test)

            threshold_baseline = GroupThresholdOptimizer(fairness_threshold=config.fairness_threshold).fit(
                dataset.y_val.to_numpy(),
                val_scores,
                dataset.protected_val.to_numpy(),
            )
            rule_baseline = RuleBasedController(
                fairness_threshold=config.fairness_threshold,
                window_size=config.window_size,
            ).fit(
                y_true=dataset.y_val.to_numpy(),
                base_preds=val_base_preds,
                base_scores=val_scores,
                protected=dataset.protected_val.to_numpy(),
            )

            dataset_specific_model = None
            if model_name in config.dataset_specific_models:
                dataset_specific_dir = output_path / "models" / "dataset_specific"
                dataset_specific_model = train_dataset_specific_controller(
                    predictions=model.predict(dataset.X_train),
                    probabilities=model_scores(model, dataset.X_train),
                    true_labels=dataset.y_train.to_numpy(),
                    protected=dataset.protected_train.to_numpy(),
                    output_dir=dataset_specific_dir,
                    config=config.dataset_specific_rl,
                    tag=f"{dataset_name}_{model_name}",
                )

            for order_protocol in config.order_protocols:
                ordered_truth, ordered_base_preds, ordered_protected, ordered_scores = _ordered_arrays(
                    y_true=dataset.y_test.to_numpy(),
                    y_pred=np.asarray(test_base_preds, dtype=int),
                    protected=dataset.protected_test.to_numpy(),
                    scores=np.asarray(test_scores, dtype=float),
                    order_protocol=order_protocol,
                    seed=config.random_seed,
                )

                base_trace = build_stream_trace(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    window_size=config.window_size,
                    fairness_threshold=config.fairness_threshold,
                )
                base_metrics = compute_static_metrics(
                    ordered_truth,
                    ordered_base_preds,
                    ordered_protected,
                    scores=ordered_scores,
                    fairness_threshold=config.fairness_threshold,
                )
                base_metrics.update(summarize_stream_trace(base_trace))
                result_rows.append(
                    _baseline_row(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="base_model",
                        order_protocol=order_protocol,
                        metrics=base_metrics,
                        base_metrics=base_metrics,
                    )
                )
                trace_rows.extend(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "base_model",
                        "order_protocol": order_protocol,
                        **row,
                    }
                    for row in base_trace.to_dict(orient="records")
                )

                threshold_preds = threshold_baseline.predict(ordered_scores, ordered_protected)
                threshold_metrics = evaluate_static_method(
                    y_true=ordered_truth,
                    y_pred=threshold_preds,
                    protected=ordered_protected,
                    scores=ordered_scores,
                    intervention_flags=(threshold_preds != ordered_base_preds).astype(int),
                    fairness_threshold=config.fairness_threshold,
                    window_size=config.window_size,
                )
                result_rows.append(
                    _baseline_row(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="group_threshold",
                        order_protocol=order_protocol,
                        metrics=threshold_metrics,
                        base_metrics=base_metrics,
                    )
                )

                rule_preds, rule_interventions, rule_trace = rule_baseline.simulate(
                    y_true=ordered_truth,
                    base_preds=ordered_base_preds,
                    base_scores=ordered_scores,
                    protected=ordered_protected,
                )
                rule_metrics = compute_static_metrics(
                    ordered_truth,
                    rule_preds,
                    ordered_protected,
                    scores=ordered_scores,
                    intervention_flags=rule_interventions,
                    fairness_threshold=config.fairness_threshold,
                )
                rule_metrics.update(summarize_stream_trace(rule_trace))
                result_rows.append(
                    _baseline_row(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name="rule_based",
                        order_protocol=order_protocol,
                        metrics=rule_metrics,
                        base_metrics=base_metrics,
                    )
                )
                trace_rows.extend(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": "rule_based",
                        "order_protocol": order_protocol,
                        **row,
                    }
                    for row in rule_trace.to_dict(orient="records")
                )

                if universal_model is not None:
                    universal_metrics, universal_trace, _, _ = evaluate_rl_controller(
                        model=universal_model,
                        predictions=np.asarray(test_base_preds, dtype=int),
                        probabilities=np.asarray(test_scores, dtype=float),
                        true_labels=dataset.y_test.to_numpy(),
                        protected=dataset.protected_test.to_numpy(),
                        order_protocol=order_protocol,
                        config=config.universal_rl,
                        seed=config.random_seed,
                    )
                    result_rows.append(
                        _baseline_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            method_name="universal_rl",
                            order_protocol=order_protocol,
                            metrics=universal_metrics,
                            base_metrics=base_metrics,
                        )
                    )
                    trace_rows.extend(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": "universal_rl",
                            "order_protocol": order_protocol,
                            **row,
                        }
                        for row in universal_trace.to_dict(orient="records")
                    )

                if dataset_specific_model is not None:
                    ds_metrics, ds_trace, _, _ = evaluate_rl_controller(
                        model=dataset_specific_model,
                        predictions=np.asarray(test_base_preds, dtype=int),
                        probabilities=np.asarray(test_scores, dtype=float),
                        true_labels=dataset.y_test.to_numpy(),
                        protected=dataset.protected_test.to_numpy(),
                        order_protocol=order_protocol,
                        config=config.dataset_specific_rl,
                        seed=config.random_seed,
                    )
                    result_rows.append(
                        _baseline_row(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            method_name="dataset_specific_rl",
                            order_protocol=order_protocol,
                            metrics=ds_metrics,
                            base_metrics=base_metrics,
                        )
                    )
                    trace_rows.extend(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": "dataset_specific_rl",
                            "order_protocol": order_protocol,
                            **row,
                        }
                        for row in ds_trace.to_dict(orient="records")
                    )

    results_df = pd.DataFrame(result_rows)
    traces_df = pd.DataFrame(trace_rows)
    results_df.to_csv(output_path / "main_results.csv", index=False)
    traces_df.to_csv(output_path / "rolling_traces.csv", index=False)
    return results_df, traces_df


def run_state_ablation(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    dataset_names: Iterable[str],
    excluded_group_sets: Dict[str, list[str]],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Train and evaluate universal controllers with state-feature ablations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ablation_name, excluded_groups in excluded_group_sets.items():
        ablation_config = RLTrainingConfig(**config.universal_rl.to_dict())
        ablation_config.state_mask = None
        from research.rl import state_mask_from_exclusions

        ablation_config.state_mask = state_mask_from_exclusions(excluded_groups)
        ablation_config.tag = f"state_{ablation_name}"
        model = train_universal_controller(
            output_dir=output_path / "models",
            config=ablation_config,
            curriculum=True,
        )

        for dataset_name in dataset_names:
            try:
                dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=config.random_seed)
            except FileNotFoundError:
                continue

            models = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=["xgboost"],
                seed=config.random_seed,
            )
            base_model = models["xgboost"]
            metrics, _, _, _ = evaluate_rl_controller(
                model=model,
                predictions=base_model.predict(dataset.X_test),
                probabilities=model_scores(base_model, dataset.X_test),
                true_labels=dataset.y_test.to_numpy(),
                protected=dataset.protected_test.to_numpy(),
                order_protocol="natural",
                config=ablation_config,
                seed=config.random_seed,
            )
            rows.append(
                {
                    "ablation": ablation_name,
                    "excluded_groups": ",".join(excluded_groups),
                    "dataset": dataset_name,
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "state_ablation_results.csv", index=False)
    return frame


def run_reward_ablation(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    dataset_names: Iterable[str],
    reward_settings: Dict[str, dict],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Train and evaluate universal controllers with different reward settings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tag, reward_kwargs in reward_settings.items():
        reward_config = RLTrainingConfig(**config.universal_rl.to_dict())
        reward_config.tag = f"reward_{tag}"
        for key, value in reward_kwargs.items():
            setattr(reward_config, key, value)
        model = train_universal_controller(
            output_dir=output_path / "models",
            config=reward_config,
            curriculum=True,
        )

        for dataset_name in dataset_names:
            try:
                dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=config.random_seed)
            except FileNotFoundError:
                continue

            models = train_base_models(
                dataset.X_train,
                dataset.y_train,
                model_names=["xgboost"],
                seed=config.random_seed,
            )
            base_model = models["xgboost"]
            metrics, _, _, _ = evaluate_rl_controller(
                model=model,
                predictions=base_model.predict(dataset.X_test),
                probabilities=model_scores(base_model, dataset.X_test),
                true_labels=dataset.y_test.to_numpy(),
                protected=dataset.protected_test.to_numpy(),
                order_protocol="natural",
                config=reward_config,
                seed=config.random_seed,
            )
            rows.append({"reward_variant": tag, "dataset": dataset_name, **metrics})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "reward_ablation_results.csv", index=False)
    return frame


def run_order_stress(
    *,
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
    dataset_names: Iterable[str],
    protocols: Iterable[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate the xgboost + universal-RL stack under different stream orders."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    universal_model = train_universal_controller(
        output_dir=output_path / "models",
        config=config.universal_rl,
        curriculum=True,
    )

    rows: list[dict] = []
    for dataset_name in dataset_names:
        try:
            dataset = load_dataset(dataset_name, search_roots=search_roots, random_state=config.random_seed)
        except FileNotFoundError:
            continue

        base_model = train_base_models(
            dataset.X_train,
            dataset.y_train,
            model_names=["xgboost"],
            seed=config.random_seed,
        )["xgboost"]

        for protocol in protocols:
            metrics, _, _, _ = evaluate_rl_controller(
                model=universal_model,
                predictions=base_model.predict(dataset.X_test),
                probabilities=model_scores(base_model, dataset.X_test),
                true_labels=dataset.y_test.to_numpy(),
                protected=dataset.protected_test.to_numpy(),
                order_protocol=protocol,
                config=config.universal_rl,
                seed=config.random_seed,
            )
            rows.append({"dataset": dataset_name, "order_protocol": protocol, **metrics})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path / "order_stress_results.csv", index=False)
    return frame

