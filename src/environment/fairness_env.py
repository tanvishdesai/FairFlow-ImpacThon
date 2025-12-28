"""
FairFlow RL Environment

A custom Gymnasium environment for training the FairFlow bias mitigation agent.
The agent learns to override biased predictions to achieve a balance between
accuracy and fairness.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import joblib
from pathlib import Path


class FairnessEnv(gym.Env):
    """
    Custom Gymnasium environment for fairness-aware decision making.
    
    The agent acts as a "Gatekeeper" that can override base model predictions
    to improve fairness metrics while maintaining acceptable accuracy.
    
    Observation Space:
        - Base model's prediction (0 or 1)
        - Base model's confidence (0.0 to 1.0)
        - Applicant features (normalized)
        - Current demographic parity ratio (rolling window)
        
    Action Space:
        - 0: APPROVE (override to positive outcome)
        - 1: DENY (override to negative outcome)
        - 2: ACCEPT (accept base model's decision)
        
    Reward:
        Composite reward balancing accuracy and fairness.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        protected: np.ndarray,
        base_model,
        feature_names: list,
        accuracy_weight: float = 0.4,
        fairness_weight: float = 0.6,
        fairness_threshold: float = 0.8,
        window_size: int = 50,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the FairnessEnv.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_true: Ground truth labels
            protected: Protected attribute values (0=unprivileged, 1=privileged)
            base_model: The pre-trained base model (XGBoost)
            feature_names: List of feature names
            accuracy_weight: Weight for accuracy in reward
            fairness_weight: Weight for fairness in reward
            fairness_threshold: Target demographic parity ratio threshold
            window_size: Rolling window size for fairness calculation
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.X = np.asarray(X)
        self.y_true = np.asarray(y_true)
        self.protected = np.asarray(protected)
        self.base_model = base_model
        self.feature_names = feature_names
        
        self.accuracy_weight = accuracy_weight
        self.fairness_weight = fairness_weight
        self.fairness_threshold = fairness_threshold
        self.window_size = window_size
        self.render_mode = render_mode
        
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        
        # Pre-compute base model predictions
        self.base_predictions = base_model.predict(X)
        self.base_probabilities = base_model.predict_proba(X)[:, 1]
        
        # Observation space: [base_pred, base_conf, features..., current_dpr]
        # Total: 1 + 1 + n_features + 1 = n_features + 3
        obs_dim = self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: APPROVE=0, DENY=1, ACCEPT=2
        self.action_space = spaces.Discrete(3)
        self.action_names = ["APPROVE", "DENY", "ACCEPT"]
        
        # Episode state
        self.current_idx = 0
        self.sample_order = np.arange(self.n_samples)
        
        # Rolling history for fairness calculation
        self.decision_history = []  # List of (final_decision, protected_value)
        
        # Metrics tracking
        self.episode_rewards = []
        self.intervention_count = 0
        self.correct_count = 0
        self.total_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Shuffle sample order
        if seed is not None:
            np.random.seed(seed)
        self.sample_order = np.random.permutation(self.n_samples)
        self.current_idx = 0
        
        # Reset history
        self.decision_history = []
        self.episode_rewards = []
        self.intervention_count = 0
        self.correct_count = 0
        self.total_count = 0
        
        return self._get_observation(), {"sample_idx": self.sample_order[0]}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take (0=APPROVE, 1=DENY, 2=ACCEPT)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        idx = self.sample_order[self.current_idx]
        
        # Get current sample info
        base_pred = self.base_predictions[idx]
        true_label = self.y_true[idx]
        protected_val = self.protected[idx]
        
        # Determine final decision based on action
        if action == 0:  # APPROVE
            final_decision = 1
            intervened = (base_pred != 1)
        elif action == 1:  # DENY
            final_decision = 0
            intervened = (base_pred != 0)
        else:  # ACCEPT
            final_decision = base_pred
            intervened = False
        
        if intervened:
            self.intervention_count += 1
        
        # Track decision for fairness calculation
        self.decision_history.append((final_decision, protected_val))
        
        # Keep only recent history for rolling fairness
        if len(self.decision_history) > self.window_size:
            self.decision_history = self.decision_history[-self.window_size:]
        
        # Calculate reward
        reward, reward_breakdown = self._calculate_reward(
            final_decision, true_label, protected_val
        )
        
        self.episode_rewards.append(reward)
        self.total_count += 1
        if final_decision == true_label:
            self.correct_count += 1
        
        # Move to next sample
        self.current_idx += 1
        terminated = self.current_idx >= self.n_samples
        truncated = False
        
        # Get next observation (or final observation if done)
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "sample_idx": idx,
            "action_name": self.action_names[action],
            "base_prediction": int(base_pred),
            "final_decision": int(final_decision),
            "true_label": int(true_label),
            "protected_value": int(protected_val),
            "intervened": intervened,
            "reward_breakdown": reward_breakdown,
            "rolling_dpr": self._calculate_rolling_dpr(),
            "rolling_accuracy": self.correct_count / max(1, self.total_count),
            "intervention_rate": self.intervention_count / max(1, self.total_count)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        idx = self.sample_order[self.current_idx]
        
        base_pred = float(self.base_predictions[idx])
        base_conf = float(self.base_probabilities[idx])
        features = self.X[idx].astype(np.float32)
        current_dpr = self._calculate_rolling_dpr()
        
        observation = np.concatenate([
            [base_pred, base_conf],
            features,
            [current_dpr]
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_rolling_dpr(self) -> float:
        """Calculate demographic parity ratio from recent decisions."""
        if len(self.decision_history) < 10:
            return 1.0  # Default to fair if not enough data
        
        decisions = np.array([d[0] for d in self.decision_history])
        protected = np.array([d[1] for d in self.decision_history])
        
        privileged_mask = protected == 1
        unprivileged_mask = protected == 0
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return 1.0
        
        priv_rate = decisions[privileged_mask].mean()
        unpriv_rate = decisions[unprivileged_mask].mean()
        
        if priv_rate == 0:
            return 1.0 if unpriv_rate == 0 else 0.0
        
        return unpriv_rate / priv_rate
    
    def _calculate_reward(
        self,
        final_decision: int,
        true_label: int,
        protected_val: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the composite reward.
        
        The reward balances:
        1. Accuracy: Correct predictions are rewarded
        2. Fairness: Maintaining DPR close to 1.0 is rewarded
        
        Args:
            final_decision: The final decision (0 or 1)
            true_label: Ground truth (0 or 1)
            protected_val: Protected attribute value
            
        Returns:
            reward: Total reward
            breakdown: Dictionary with reward components
        """
        # Accuracy reward: +1 for correct, -1 for incorrect
        accuracy_reward = 1.0 if final_decision == true_label else -1.0
        
        # Fairness reward based on current DPR
        current_dpr = self._calculate_rolling_dpr()
        
        # Reward for being in the fair range [0.8, 1.25]
        if 0.8 <= current_dpr <= 1.25:
            fairness_reward = 1.0
        else:
            # Penalize based on distance from fair range
            if current_dpr < 0.8:
                distance = 0.8 - current_dpr
            else:
                distance = current_dpr - 1.25
            fairness_reward = -distance * 5.0  # Scaled penalty
        
        # Intervention penalty (small penalty for overriding)
        intervention_penalty = -0.1 if final_decision != self.base_predictions[
            self.sample_order[self.current_idx - 1] if self.current_idx > 0 else 0
        ] else 0.0
        
        # Composite reward
        reward = (
            self.accuracy_weight * accuracy_reward +
            self.fairness_weight * fairness_reward +
            intervention_penalty
        )
        
        breakdown = {
            "accuracy_reward": accuracy_reward,
            "fairness_reward": fairness_reward,
            "intervention_penalty": intervention_penalty,
            "current_dpr": current_dpr
        }
        
        return reward, breakdown
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            dpr = self._calculate_rolling_dpr()
            accuracy = self.correct_count / max(1, self.total_count)
            int_rate = self.intervention_count / max(1, self.total_count)
            
            print(f"Step {self.current_idx}/{self.n_samples} | "
                  f"DPR: {dpr:.3f} | Acc: {accuracy:.3f} | Int Rate: {int_rate:.3f}")
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary metrics for the episode."""
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "final_accuracy": self.correct_count / max(1, self.total_count),
            "final_dpr": self._calculate_rolling_dpr(),
            "intervention_rate": self.intervention_count / max(1, self.total_count),
            "total_interventions": self.intervention_count,
            "total_samples": self.total_count
        }


def create_fairness_env(
    data_dir: str = "data",
    model_path: str = "models/base_model/xgboost_biased.joblib",
    split: str = "train",
    **kwargs
) -> FairnessEnv:
    """
    Factory function to create a FairnessEnv from saved data and model.
    
    Args:
        data_dir: Path to data directory
        model_path: Path to trained base model
        split: Which data split to use ("train", "val", "test")
        **kwargs: Additional arguments passed to FairnessEnv
        
    Returns:
        Configured FairnessEnv instance
    """
    from src.utils.data_loader import load_adult_data
    
    # Load data
    data = load_adult_data(data_dir=data_dir, protected_attribute="sex")
    
    # Load model
    model = joblib.load(model_path)
    
    # Create environment
    env = FairnessEnv(
        X=data[f"X_{split}"].values,
        y_true=data[f"y_{split}"].values,
        protected=data[f"protected_{split}"].values,
        base_model=model,
        feature_names=data["feature_names"],
        **kwargs
    )
    
    return env


if __name__ == "__main__":
    # Test the environment
    print("Testing FairnessEnv...")
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y_true = np.random.randint(0, 2, n_samples)
    protected = np.random.randint(0, 2, n_samples)
    
    # Create a dummy model
    class DummyModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        def predict_proba(self, X):
            probs = np.random.rand(len(X), 2)
            return probs / probs.sum(axis=1, keepdims=True)
    
    # Create environment
    env = FairnessEnv(
        X=X,
        y_true=y_true,
        protected=protected,
        base_model=DummyModel(),
        feature_names=[f"feature_{i}" for i in range(n_features)],
        render_mode="human"
    )
    
    # Run a test episode
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            env.render()
        
        if terminated:
            break
    
    print(f"\n📊 Episode Summary:")
    summary = env.get_episode_summary()
    for k, v in summary.items():
        print(f"   {k}: {v:.4f}")
