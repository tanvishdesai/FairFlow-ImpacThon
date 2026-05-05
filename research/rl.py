"""
Reinforcement-learning components for the FairFlow paper experiments.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from research.metrics import RollingFairnessTracker, build_stream_trace, compute_static_metrics, summarize_stream_trace
from src.utils.synthetic_data import SyntheticScenarioGenerator

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - only relevant if stable-baselines3 is missing
    PPO = None


STATE_GROUPS: Dict[str, tuple[int, ...]] = {
    "base_model": (0, 1, 2),
    "fairness_rates": (3, 4, 5, 6, 7),
    "intervention_history": (8,),
    "stream_context": (9, 10),
    "confidence_gap": (11,),
}


def state_mask_from_exclusions(excluded_groups: Iterable[str]) -> np.ndarray:
    """Create a boolean mask for state ablations."""
    mask = np.ones(12, dtype=bool)
    for group_name in excluded_groups:
        if group_name not in STATE_GROUPS:
            raise KeyError(f"Unknown state group '{group_name}'. Available: {sorted(STATE_GROUPS)}")
        mask[list(STATE_GROUPS[group_name])] = False
    return mask


def make_order_indices(protected: np.ndarray, protocol: str, seed: int = 42) -> np.ndarray:
    """Generate stream orders for robustness and drift experiments."""
    n_samples = len(protected)
    indices = np.arange(n_samples)
    rng_seed = seed

    if protocol == "natural":
        return indices
    if protocol.startswith("random"):
        parts = protocol.split("_", maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit():
            rng_seed = seed + int(parts[1])
        rng = np.random.default_rng(rng_seed)
        return rng.permutation(indices)
    if protocol == "privileged_first":
        return np.concatenate([indices[protected == 1], indices[protected == 0]])
    if protocol == "unprivileged_first":
        return np.concatenate([indices[protected == 0], indices[protected == 1]])
    if protocol == "alternating_groups":
        privileged = list(indices[protected == 1])
        unprivileged = list(indices[protected == 0])
        merged: list[int] = []
        while privileged or unprivileged:
            if privileged:
                merged.append(privileged.pop(0))
            if unprivileged:
                merged.append(unprivileged.pop(0))
        return np.asarray(merged, dtype=int)
    if protocol == "burst_priv_then_unpriv":
        privileged = indices[protected == 1]
        unprivileged = indices[protected == 0]
        merged: list[int] = []
        chunk = 32
        for start in range(0, max(len(privileged), len(unprivileged)), chunk):
            merged.extend(privileged[start:start + chunk].tolist())
            merged.extend(unprivileged[start:start + chunk].tolist())
        return np.asarray(merged, dtype=int)

    raise KeyError(f"Unknown order protocol '{protocol}'")


@dataclass
class RLTrainingConfig:
    total_timesteps: int = 60_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    accuracy_weight: float = 0.45
    fairness_weight: float = 0.55
    intervention_penalty: float = -0.05
    fairness_threshold: float = 0.8
    fairness_upper_target: float = 1.05
    fairness_band_bonus: float = 0.45
    overshoot_penalty_weight: float = 0.75
    window_size: int = 50
    random_seed: int = 42
    synthetic_random_scenarios: int = 12
    state_mask: Optional[np.ndarray] = None
    tag: str = "default"

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        if self.state_mask is not None:
            data["state_mask"] = self.state_mask.astype(int).tolist()
        return data


def tracker_snapshot(tracker: RollingFairnessTracker, fairness_threshold: float) -> dict:
    """Read the current rolling metrics from a tracker without mutating it."""
    rolling = tracker._window_counts  # pylint: disable=protected-access

    privileged_rate = (
        rolling["privileged_positive"] / rolling["privileged_n"]
        if rolling["privileged_n"] > 0
        else 0.5
    )
    unprivileged_rate = (
        rolling["unprivileged_positive"] / rolling["unprivileged_n"]
        if rolling["unprivileged_n"] > 0
        else 0.5
    )
    if privileged_rate == 0:
        rolling_dpr = 1.0 if unprivileged_rate == 0 else 0.0
    else:
        rolling_dpr = unprivileged_rate / privileged_rate

    privileged_tpr = (
        rolling["privileged_tp"] / (rolling["privileged_tp"] + rolling["privileged_fn"])
        if (rolling["privileged_tp"] + rolling["privileged_fn"]) > 0
        else 0.5
    )
    unprivileged_tpr = (
        rolling["unprivileged_tp"] / (rolling["unprivileged_tp"] + rolling["unprivileged_fn"])
        if (rolling["unprivileged_tp"] + rolling["unprivileged_fn"]) > 0
        else 0.5
    )
    privileged_fpr = (
        rolling["privileged_fp"] / (rolling["privileged_fp"] + rolling["privileged_tn"])
        if (rolling["privileged_fp"] + rolling["privileged_tn"]) > 0
        else 0.5
    )
    unprivileged_fpr = (
        rolling["unprivileged_fp"] / (rolling["unprivileged_fp"] + rolling["unprivileged_tn"])
        if (rolling["unprivileged_fp"] + rolling["unprivileged_tn"]) > 0
        else 0.5
    )
    tpr_gap = unprivileged_tpr - privileged_tpr
    fpr_gap = unprivileged_fpr - privileged_fpr
    return {
        "rolling_dpr": rolling_dpr,
        "rolling_privileged_rate": privileged_rate,
        "rolling_unprivileged_rate": unprivileged_rate,
        "rolling_tpr_gap": tpr_gap,
        "rolling_fpr_gap": fpr_gap,
        "rolling_equalized_odds_gap": max(abs(tpr_gap), abs(fpr_gap)),
        "rolling_fairness_pass": float(rolling_dpr >= fairness_threshold),
    }


def build_state_from_history(
    *,
    base_pred: int,
    base_prob: float,
    protected_value: int,
    tracker: RollingFairnessTracker,
    intervention_history: list[int],
    seen_group_history: deque[int],
    consecutive_same_group: int,
    privileged_confidences: deque[float],
    unprivileged_confidences: deque[float],
    protected_population_mean: float,
    config: RLTrainingConfig,
) -> np.ndarray:
    """Build the 12-dimensional state used by the RL controller from external history."""
    snapshot = tracker_snapshot(tracker, config.fairness_threshold)
    intervention_rate = float(np.mean(intervention_history)) if intervention_history else 0.0
    group_ratio = (
        float(np.mean(np.asarray(seen_group_history) == 0))
        if seen_group_history
        else 1.0 - protected_population_mean
    )
    consecutive_norm = min(consecutive_same_group / max(config.window_size, 1), 1.0)
    privileged_conf = float(np.mean(privileged_confidences)) if privileged_confidences else 0.5
    unprivileged_conf = float(np.mean(unprivileged_confidences)) if unprivileged_confidences else 0.5
    confidence_gap = unprivileged_conf - privileged_conf

    state = np.array(
        [
            float(base_pred),
            float(base_prob),
            float(protected_value),
            np.clip(snapshot["rolling_dpr"] / 2.0, 0.0, 1.0),
            np.clip((snapshot["rolling_tpr_gap"] + 1.0) / 2.0, 0.0, 1.0),
            np.clip((snapshot["rolling_fpr_gap"] + 1.0) / 2.0, 0.0, 1.0),
            np.clip(snapshot["rolling_privileged_rate"], 0.0, 1.0),
            np.clip(snapshot["rolling_unprivileged_rate"], 0.0, 1.0),
            np.clip(intervention_rate, 0.0, 1.0),
            np.clip(group_ratio, 0.0, 1.0),
            np.clip(consecutive_norm, 0.0, 1.0),
            np.clip((confidence_gap + 1.0) / 2.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    if config.state_mask is not None:
        masked = StreamingFairnessEnv.neutral_state.copy()
        masked[config.state_mask] = state[config.state_mask]
        return masked
    return state


class StreamingFairnessEnv(gym.Env):
    """Streaming environment for fairness-aware intervention."""

    metadata = {"render_modes": []}
    neutral_state = np.full(12, 0.5, dtype=np.float32)

    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        protected: np.ndarray,
        *,
        config: RLTrainingConfig,
        order_indices: Optional[np.ndarray] = None,
        shuffle_on_reset: bool = True,
    ):
        super().__init__()
        self.base_predictions = np.asarray(predictions, dtype=int)
        self.base_probabilities = np.asarray(probabilities, dtype=float)
        self.true_labels = np.asarray(true_labels, dtype=int)
        self.protected = np.asarray(protected, dtype=int)
        self.config = config
        self.order_indices = order_indices
        self.shuffle_on_reset = shuffle_on_reset

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(self.config.random_seed)
        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.current_step = 0
        self.stream_indices = np.arange(len(self.base_predictions))
        self.tracker = RollingFairnessTracker(
            window_size=self.config.window_size,
            fairness_threshold=self.config.fairness_threshold,
        )
        self.interventions: list[int] = []
        self.privileged_confidences: deque[float] = deque(maxlen=self.config.window_size)
        self.unprivileged_confidences: deque[float] = deque(maxlen=self.config.window_size)
        self.seen_group_history: deque[int] = deque(maxlen=self.config.window_size)
        self.consecutive_same_group = 0
        self.last_group: Optional[int] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_internal_state()
        if self.order_indices is not None:
            self.stream_indices = np.asarray(self.order_indices, dtype=int)
        elif self.shuffle_on_reset:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self.stream_indices = self._rng.permutation(len(self.base_predictions))
        return self._build_observation(), {}

    def step(self, action: int):
        idx = self.stream_indices[self.current_step]
        base_pred = int(self.base_predictions[idx])
        base_prob = float(self.base_probabilities[idx])
        true_label = int(self.true_labels[idx])
        protected_value = int(self.protected[idx])

        if action == 0:
            final_decision = base_pred
        elif action == 1:
            final_decision = 0
        else:
            final_decision = 1

        intervened = int(final_decision != base_pred)
        self.interventions.append(intervened)

        if protected_value == 1:
            self.privileged_confidences.append(base_prob)
        else:
            self.unprivileged_confidences.append(base_prob)
        self.seen_group_history.append(protected_value)

        if self.last_group == protected_value:
            self.consecutive_same_group += 1
        else:
            self.consecutive_same_group = 1
        self.last_group = protected_value

        stream_metrics = self.tracker.update(true_label, final_decision, protected_value)

        reward = self._reward(
            final_decision=final_decision,
            true_label=true_label,
            protected_value=protected_value,
            intervened=intervened,
            stream_metrics=stream_metrics,
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.stream_indices)
        observation = self.neutral_state.copy() if terminated else self._build_observation()

        info = {
            "index": idx,
            "base_prediction": base_pred,
            "base_probability": base_prob,
            "true_label": true_label,
            "protected_value": protected_value,
            "final_decision": final_decision,
            "intervened": intervened,
            **stream_metrics,
        }
        return observation, reward, terminated, False, info

    def _build_observation(self) -> np.ndarray:
        idx = self.stream_indices[self.current_step]
        return build_state_from_history(
            base_pred=int(self.base_predictions[idx]),
            base_prob=float(self.base_probabilities[idx]),
            protected_value=int(self.protected[idx]),
            tracker=self.tracker,
            intervention_history=self.interventions,
            seen_group_history=self.seen_group_history,
            consecutive_same_group=self.consecutive_same_group,
            privileged_confidences=self.privileged_confidences,
            unprivileged_confidences=self.unprivileged_confidences,
            protected_population_mean=float(np.mean(self.protected)),
            config=self.config,
        )

    def _reward(
        self,
        *,
        final_decision: int,
        true_label: int,
        protected_value: int,
        intervened: int,
        stream_metrics: dict,
    ) -> float:
        accuracy_reward = 1.0 if final_decision == true_label else -1.0
        dpr = stream_metrics["rolling_dpr"]
        eo_gap = stream_metrics["rolling_equalized_odds_gap"]

        if dpr < self.config.fairness_threshold:
            fairness_reward = -(self.config.fairness_threshold - dpr)
        elif dpr <= self.config.fairness_upper_target:
            fairness_reward = self.config.fairness_band_bonus
        else:
            fairness_reward = self.config.fairness_band_bonus - (
                self.config.overshoot_penalty_weight * (dpr - self.config.fairness_upper_target)
            )

        fairness_reward -= 0.25 * eo_gap

        if protected_value == 0 and final_decision == 1 and dpr < self.config.fairness_threshold:
            fairness_reward += 0.25

        intervention_term = self.config.intervention_penalty if intervened else 0.0
        return (
            self.config.accuracy_weight * accuracy_reward
            + self.config.fairness_weight * fairness_reward
            + intervention_term
        )


class MultiScenarioStreamingEnv(StreamingFairnessEnv):
    """Environment that samples a new synthetic scenario each episode."""

    def __init__(self, scenarios: list[dict], *, config: RLTrainingConfig, curriculum: bool = True):
        self.scenarios = scenarios
        self.curriculum = curriculum
        self.episode_count = 0
        self._scenario_rng = np.random.default_rng(config.random_seed)

        first = scenarios[0]
        super().__init__(
            first["predictions"],
            first["probabilities"],
            first["true_labels"],
            first["protected"],
            config=config,
            order_indices=None,
            shuffle_on_reset=True,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.episode_count += 1
        scenario = self._pick_scenario()
        self.base_predictions = np.asarray(scenario["predictions"], dtype=int)
        self.base_probabilities = np.asarray(scenario["probabilities"], dtype=float)
        self.true_labels = np.asarray(scenario["true_labels"], dtype=int)
        self.protected = np.asarray(scenario["protected"], dtype=int)
        return super().reset(seed=seed, options=options)

    def _pick_scenario(self) -> dict:
        if not self.curriculum:
            return self.scenarios[self._scenario_rng.integers(0, len(self.scenarios))]
        progress = min(self.episode_count / 120.0, 1.0)
        max_index = max(1, int(progress * len(self.scenarios)))
        return self.scenarios[self._scenario_rng.integers(0, max_index)]


def ensure_sb3_available() -> None:
    if PPO is None:
        raise ImportError("stable-baselines3 is required for the RL experiment scripts.")


def train_universal_controller(
    *,
    output_dir: str | Path,
    config: RLTrainingConfig,
    curriculum: bool = True,
    device: str = "cpu",
    show_progress: bool = False,
) -> PPO:
    """Train a universal controller on synthetic scenarios."""
    ensure_sb3_available()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticScenarioGenerator()
    scenarios = generator.generate_all_training_data(
        base_seed=config.random_seed,
        augment_random=config.synthetic_random_scenarios,
        verbose=show_progress,
    )
    env = MultiScenarioStreamingEnv(scenarios=scenarios, config=config, curriculum=curriculum)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        seed=config.random_seed,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": [128, 128, 64]},
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=show_progress)

    model_path = output_path / f"universal_{config.tag}.zip"
    model.save(str(model_path.with_suffix("")))
    with open(output_path / f"universal_{config.tag}.json", "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    return model


def train_dataset_specific_controller(
    *,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    protected: np.ndarray,
    output_dir: str | Path,
    config: RLTrainingConfig,
    tag: str,
    device: str = "cpu",
    show_progress: bool = False,
) -> PPO:
    """Train a controller on one dataset-model stream only."""
    ensure_sb3_available()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = StreamingFairnessEnv(
        predictions=predictions,
        probabilities=probabilities,
        true_labels=true_labels,
        protected=protected,
        config=config,
        order_indices=None,
        shuffle_on_reset=True,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        seed=config.random_seed,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": [128, 128, 64]},
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=show_progress)

    model_path = output_path / f"{tag}.zip"
    model.save(str(model_path.with_suffix("")))
    return model


def evaluate_rl_controller(
    *,
    model: PPO,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    protected: np.ndarray,
    order_protocol: str,
    config: RLTrainingConfig,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Run a trained controller on a fixed stream and collect paper metrics."""
    order = make_order_indices(protected, protocol=order_protocol, seed=seed)
    env = StreamingFairnessEnv(
        predictions=predictions,
        probabilities=probabilities,
        true_labels=true_labels,
        protected=protected,
        config=config,
        order_indices=order,
        shuffle_on_reset=False,
    )
    obs, _ = env.reset(seed=seed)

    ordered_truth = true_labels[order]
    ordered_protected = protected[order]
    ordered_scores = probabilities[order]
    ordered_base_preds = predictions[order]

    final_decisions = np.zeros_like(ordered_base_preds, dtype=int)
    interventions = np.zeros_like(ordered_base_preds, dtype=int)

    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        final_decisions[step] = int(info["final_decision"])
        interventions[step] = int(info["intervened"])
        step += 1
        done = terminated or truncated

    summary = compute_static_metrics(
        ordered_truth,
        final_decisions,
        ordered_protected,
        scores=ordered_scores,
        intervention_flags=interventions,
        fairness_threshold=config.fairness_threshold,
    )
    trace = build_stream_trace(
        ordered_truth,
        final_decisions,
        ordered_protected,
        window_size=config.window_size,
        fairness_threshold=config.fairness_threshold,
    )
    summary.update(summarize_stream_trace(trace))
    summary["base_accuracy"] = float(np.mean(ordered_truth == ordered_base_preds))
    summary["base_demographic_parity_ratio"] = compute_static_metrics(
        ordered_truth,
        ordered_base_preds,
        ordered_protected,
        scores=ordered_scores,
        fairness_threshold=config.fairness_threshold,
    )["demographic_parity_ratio"]
    return summary, trace, final_decisions, interventions
