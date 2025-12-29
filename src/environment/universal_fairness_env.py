"""
Universal FairFlow RL Environment

A dataset-agnostic Gymnasium environment for training the FairFlow bias mitigation agent.
The agent uses STATISTICAL SUMMARIES instead of raw features, making it transferable
across any dataset without retraining.

State Space (12 dimensions - FIXED regardless of dataset):
    0: base_prediction         - Base model's prediction (0 or 1)
    1: base_confidence        - Base model's confidence (0-1)
    2: protected_value        - Protected attribute value (0=unprivileged, 1=privileged)
    3: current_dpr           - Rolling Demographic Parity Ratio (normalized 0-1)
    4: current_tpr_diff      - Rolling TPR difference (normalized 0-1)
    5: current_fpr_diff      - Rolling FPR difference (normalized 0-1)
    6: privileged_approval   - Privileged group approval rate (0-1)
    7: unprivileged_approval - Unprivileged group approval rate (0-1)
    8: intervention_rate     - Recent intervention rate (0-1)
    9: group_ratio           - Unprivileged count / total (0-1)
    10: consecutive_same     - Consecutive same-group predictions (normalized 0-1)
    11: confidence_gap       - Model confidence gap between groups (normalized 0-1)

Action Space:
    0: APPROVE - Accept the base model's prediction
    1: DENY    - Override to rejection (0)
    2: ACCEPT  - Override to approval (1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Callable, Union
from collections import deque


class UniversalFairnessEnv(gym.Env):
    """
    Dataset-agnostic Gymnasium environment for fairness-aware decision making.
    
    This environment works with ANY dataset because it uses statistical fairness
    summaries instead of raw features. The agent learns fairness patterns that
    transfer across domains.
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Fixed state dimension - this never changes regardless of dataset
    STATE_DIM = 12
    
    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
        protected: np.ndarray,
        accuracy_weight: float = 0.4,
        fairness_weight: float = 0.5,
        fairness_threshold: float = 0.8,
        window_size: int = 50,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the UniversalFairnessEnv.
        
        Args:
            predictions: Base model predictions (n_samples,) - 0 or 1
            probabilities: Base model probabilities (n_samples,) - 0 to 1
            true_labels: Ground truth labels (n_samples,)
            protected: Protected attribute values (n_samples,) - 0=unprivileged, 1=privileged
            accuracy_weight: Weight for accuracy in reward (0-1)
            fairness_weight: Weight for fairness in reward (0-1)
            fairness_threshold: Minimum acceptable DPR (e.g., 0.8)
            window_size: Window for rolling metrics
            render_mode: Rendering mode ("human" or None)
        """
        super().__init__()
        
        # Store data
        self.predictions = np.array(predictions)
        self.probabilities = np.array(probabilities)
        self.true_labels = np.array(true_labels)
        self.protected = np.array(protected)
        self.n_samples = len(predictions)
        
        # Reward weights
        self.accuracy_weight = accuracy_weight
        self.fairness_weight = fairness_weight
        self.fairness_threshold = fairness_threshold
        
        # Window for rolling metrics
        self.window_size = window_size
        
        # Render mode
        self.render_mode = render_mode
        
        # ===== FIXED STATE SPACE (12 dimensions) =====
        # All values normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.STATE_DIM,),
            dtype=np.float32
        )
        
        # Action space: 0=APPROVE (use base), 1=DENY (force 0), 2=ACCEPT (force 1)
        self.action_space = spaces.Discrete(3)
        
        # Episode state
        self.current_idx = 0
        self.decisions_history = deque(maxlen=window_size)
        self.interventions = []
        self.episode_rewards = []
        
        # Rolling statistics per group
        self.privileged_decisions = deque(maxlen=window_size)
        self.unprivileged_decisions = deque(maxlen=window_size)
        self.privileged_confidences = deque(maxlen=window_size)
        self.unprivileged_confidences = deque(maxlen=window_size)
        
        # Track correct predictions for TPR/FPR
        self.privileged_tp = deque(maxlen=window_size)
        self.privileged_fp = deque(maxlen=window_size)
        self.privileged_tn = deque(maxlen=window_size)
        self.privileged_fn = deque(maxlen=window_size)
        self.unprivileged_tp = deque(maxlen=window_size)
        self.unprivileged_fp = deque(maxlen=window_size)
        self.unprivileged_tn = deque(maxlen=window_size)
        self.unprivileged_fn = deque(maxlen=window_size)
        
        # Track consecutive same-group predictions
        self.consecutive_same_group = 0
        self.last_protected_value = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Shuffle data order for varied training
        if seed is not None:
            np.random.seed(seed)
        self.indices = np.random.permutation(self.n_samples)
        
        # Reset state
        self.current_idx = 0
        self.decisions_history.clear()
        self.interventions = []
        self.episode_rewards = []
        
        # Reset rolling stats
        self.privileged_decisions.clear()
        self.unprivileged_decisions.clear()
        self.privileged_confidences.clear()
        self.unprivileged_confidences.clear()
        
        self.privileged_tp.clear()
        self.privileged_fp.clear()
        self.privileged_tn.clear()
        self.privileged_fn.clear()
        self.unprivileged_tp.clear()
        self.unprivileged_fp.clear()
        self.unprivileged_tn.clear()
        self.unprivileged_fn.clear()
        
        self.consecutive_same_group = 0
        self.last_protected_value = None
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: 0=APPROVE (use base), 1=DENY (force 0), 2=ACCEPT (force 1)
        """
        idx = self.indices[self.current_idx]
        
        base_pred = int(self.predictions[idx])
        base_prob = float(self.probabilities[idx])
        true_label = int(self.true_labels[idx])
        protected_val = int(self.protected[idx])
        
        # Determine final decision
        if action == 0:  # APPROVE - use base prediction
            final_decision = base_pred
            intervened = False
            intervention_type = "none"
        elif action == 1:  # DENY - force rejection
            final_decision = 0
            intervened = (base_pred != 0)
            intervention_type = "deny" if intervened else "none"
        else:  # ACCEPT - force approval
            final_decision = 1
            intervened = (base_pred != 1)
            intervention_type = "accept" if intervened else "none"
        
        # Update rolling statistics
        self._update_statistics(final_decision, true_label, protected_val, base_prob)
        
        # Calculate reward
        reward, reward_breakdown = self._calculate_reward(
            final_decision, true_label, protected_val
        )
        
        # Store history
        self.decisions_history.append({
            "decision": final_decision,
            "true_label": true_label,
            "protected": protected_val,
            "intervened": intervened
        })
        self.interventions.append(intervened)
        self.episode_rewards.append(reward)
        
        # Update consecutive tracking
        if self.last_protected_value == protected_val:
            self.consecutive_same_group += 1
        else:
            self.consecutive_same_group = 1
        self.last_protected_value = protected_val
        
        # Move to next sample
        self.current_idx += 1
        terminated = (self.current_idx >= self.n_samples)
        
        # Get next observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.STATE_DIM, dtype=np.float32)
        
        info = {
            "base_prediction": base_pred,
            "base_probability": base_prob,
            "final_decision": final_decision,
            "true_label": true_label,
            "protected_value": protected_val,
            "intervened": intervened,
            "intervention_type": intervention_type,
            "reward_breakdown": reward_breakdown
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the UNIVERSAL observation (12 dimensions).
        
        This is the key innovation: the state is dataset-agnostic.
        """
        idx = self.indices[self.current_idx]
        
        base_pred = float(self.predictions[idx])
        base_prob = float(self.probabilities[idx])
        protected_val = float(self.protected[idx])
        
        # Calculate rolling fairness metrics
        dpr = self._calculate_rolling_dpr()
        tpr_diff, fpr_diff = self._calculate_rolling_rates()
        priv_approval, unpriv_approval = self._calculate_approval_rates()
        intervention_rate = np.mean(self.interventions) if self.interventions else 0.0
        
        # Calculate group ratio
        group_ratio = np.mean(self.protected == 0)  # Proportion of unprivileged
        
        # Consecutive same-group (normalized by window size)
        consecutive_norm = min(self.consecutive_same_group / self.window_size, 1.0)
        
        # Confidence gap between groups
        confidence_gap = self._calculate_confidence_gap()
        
        # Build state vector (all normalized to [0, 1])
        state = np.array([
            base_pred,                          # 0: Base prediction
            base_prob,                          # 1: Base confidence
            protected_val,                      # 2: Protected value
            self._normalize_dpr(dpr),           # 3: DPR (normalized)
            self._normalize_diff(tpr_diff),     # 4: TPR diff (normalized)
            self._normalize_diff(fpr_diff),     # 5: FPR diff (normalized)
            priv_approval,                      # 6: Privileged approval rate
            unpriv_approval,                    # 7: Unprivileged approval rate
            intervention_rate,                  # 8: Intervention rate
            group_ratio,                        # 9: Group ratio
            consecutive_norm,                   # 10: Consecutive same-group
            self._normalize_diff(confidence_gap) # 11: Confidence gap
        ], dtype=np.float32)
        
        return state
    
    def _update_statistics(self, decision: int, true_label: int, protected: int, prob: float):
        """Update rolling statistics for a single sample."""
        if protected == 1:  # Privileged
            self.privileged_decisions.append(decision)
            self.privileged_confidences.append(prob)
            
            # Track confusion matrix components
            if decision == 1 and true_label == 1:
                self.privileged_tp.append(1)
            else:
                self.privileged_tp.append(0)
            
            if decision == 1 and true_label == 0:
                self.privileged_fp.append(1)
            else:
                self.privileged_fp.append(0)
                
            if decision == 0 and true_label == 0:
                self.privileged_tn.append(1)
            else:
                self.privileged_tn.append(0)
                
            if decision == 0 and true_label == 1:
                self.privileged_fn.append(1)
            else:
                self.privileged_fn.append(0)
        else:  # Unprivileged
            self.unprivileged_decisions.append(decision)
            self.unprivileged_confidences.append(prob)
            
            if decision == 1 and true_label == 1:
                self.unprivileged_tp.append(1)
            else:
                self.unprivileged_tp.append(0)
            
            if decision == 1 and true_label == 0:
                self.unprivileged_fp.append(1)
            else:
                self.unprivileged_fp.append(0)
                
            if decision == 0 and true_label == 0:
                self.unprivileged_tn.append(1)
            else:
                self.unprivileged_tn.append(0)
                
            if decision == 0 and true_label == 1:
                self.unprivileged_fn.append(1)
            else:
                self.unprivileged_fn.append(0)
    
    def _calculate_rolling_dpr(self) -> float:
        """Calculate Demographic Parity Ratio from recent decisions."""
        if len(self.privileged_decisions) < 5 or len(self.unprivileged_decisions) < 5:
            return 1.0  # Default to fair until we have enough data
        
        priv_rate = np.mean(list(self.privileged_decisions))
        unpriv_rate = np.mean(list(self.unprivileged_decisions))
        
        if priv_rate == 0:
            return 0.0 if unpriv_rate > 0 else 1.0
        
        return min(unpriv_rate / priv_rate, 2.0)  # Cap at 2.0
    
    def _calculate_rolling_rates(self):
        """Calculate TPR and FPR differences."""
        # TPR = TP / (TP + FN)
        priv_tp = sum(self.privileged_tp)
        priv_fn = sum(self.privileged_fn)
        unpriv_tp = sum(self.unprivileged_tp)
        unpriv_fn = sum(self.unprivileged_fn)
        
        priv_tpr = priv_tp / (priv_tp + priv_fn) if (priv_tp + priv_fn) > 0 else 0.5
        unpriv_tpr = unpriv_tp / (unpriv_tp + unpriv_fn) if (unpriv_tp + unpriv_fn) > 0 else 0.5
        tpr_diff = priv_tpr - unpriv_tpr
        
        # FPR = FP / (FP + TN)
        priv_fp = sum(self.privileged_fp)
        priv_tn = sum(self.privileged_tn)
        unpriv_fp = sum(self.unprivileged_fp)
        unpriv_tn = sum(self.unprivileged_tn)
        
        priv_fpr = priv_fp / (priv_fp + priv_tn) if (priv_fp + priv_tn) > 0 else 0.5
        unpriv_fpr = unpriv_fp / (unpriv_fp + unpriv_tn) if (unpriv_fp + unpriv_tn) > 0 else 0.5
        fpr_diff = priv_fpr - unpriv_fpr
        
        return tpr_diff, fpr_diff
    
    def _calculate_approval_rates(self):
        """Calculate approval rates per group."""
        priv_rate = np.mean(list(self.privileged_decisions)) if self.privileged_decisions else 0.5
        unpriv_rate = np.mean(list(self.unprivileged_decisions)) if self.unprivileged_decisions else 0.5
        return priv_rate, unpriv_rate
    
    def _calculate_confidence_gap(self) -> float:
        """Calculate average confidence gap between groups."""
        if not self.privileged_confidences or not self.unprivileged_confidences:
            return 0.0
        
        priv_conf = np.mean(list(self.privileged_confidences))
        unpriv_conf = np.mean(list(self.unprivileged_confidences))
        return priv_conf - unpriv_conf
    
    def _normalize_dpr(self, dpr: float) -> float:
        """Normalize DPR from [0, 2] to [0, 1]."""
        return min(max(dpr / 2.0, 0.0), 1.0)
    
    def _normalize_diff(self, diff: float) -> float:
        """Normalize difference from [-1, 1] to [0, 1]."""
        return min(max((diff + 1.0) / 2.0, 0.0), 1.0)
    
    def _calculate_reward(self, final_decision: int, true_label: int, protected: int):
        """
        Calculate composite reward balancing accuracy and fairness.
        """
        # Accuracy component: +1 for correct, -1 for incorrect
        accuracy_reward = 1.0 if final_decision == true_label else -1.0
        
        # Fairness component: Penalize when DPR deviates from 1.0
        current_dpr = self._calculate_rolling_dpr()
        
        # DPR penalty: penalize distance from 1.0
        if current_dpr >= self.fairness_threshold:
            fairness_reward = 0.5  # Small bonus for maintaining fairness
        else:
            # Penalty proportional to how far below threshold
            fairness_reward = -2.0 * (self.fairness_threshold - current_dpr)
        
        # Bonus for improving fairness when it was bad
        if protected == 0 and final_decision == 1:  # Approved unprivileged
            if current_dpr < self.fairness_threshold:
                fairness_reward += 0.5  # Bonus for helping unprivileged
        
        # Intervention penalty (small) - prefer minimal intervention
        intervention_penalty = -0.1 if len(self.interventions) > 0 and self.interventions[-1] else 0.0
        
        # Composite reward
        total_reward = (
            self.accuracy_weight * accuracy_reward +
            self.fairness_weight * fairness_reward +
            intervention_penalty
        )
        
        breakdown = {
            "accuracy_reward": accuracy_reward,
            "fairness_reward": fairness_reward,
            "intervention_penalty": intervention_penalty,
            "current_dpr": current_dpr,
            "total": total_reward
        }
        
        return total_reward, breakdown
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary metrics for the episode."""
        if not self.decisions_history:
            return {}
        
        decisions = [d["decision"] for d in self.decisions_history]
        true_labels = [d["true_label"] for d in self.decisions_history]
        protected = [d["protected"] for d in self.decisions_history]
        
        decisions = np.array(decisions)
        true_labels = np.array(true_labels)
        protected = np.array(protected)
        
        # Calculate final metrics
        accuracy = np.mean(decisions == true_labels)
        
        # DPR
        priv_mask = protected == 1
        unpriv_mask = protected == 0
        priv_rate = np.mean(decisions[priv_mask]) if np.sum(priv_mask) > 0 else 0.5
        unpriv_rate = np.mean(decisions[unpriv_mask]) if np.sum(unpriv_mask) > 0 else 0.5
        dpr = unpriv_rate / priv_rate if priv_rate > 0 else 1.0
        
        intervention_rate = np.mean(self.interventions) if self.interventions else 0.0
        
        return {
            "final_accuracy": accuracy,
            "final_dpr": dpr,
            "privileged_approval_rate": priv_rate,
            "unprivileged_approval_rate": unpriv_rate,
            "intervention_rate": intervention_rate,
            "total_samples": len(decisions),
            "total_interventions": sum(self.interventions),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        }
    
    def render(self):
        """Render current state."""
        if self.render_mode != "human":
            return
        
        summary = self.get_episode_summary()
        print(f"\n📊 Step {self.current_idx}/{self.n_samples}")
        print(f"   Accuracy: {summary.get('final_accuracy', 0):.4f}")
        print(f"   DPR: {summary.get('final_dpr', 1):.4f}")
        print(f"   Interventions: {summary.get('intervention_rate', 0):.2%}")


class UniversalFairnessEnvFactory:
    """
    Factory class to create UniversalFairnessEnv from any model and dataset.
    """
    
    @staticmethod
    def from_model_and_data(
        model,
        X: np.ndarray,
        y_true: np.ndarray,
        protected: np.ndarray,
        **kwargs
    ) -> UniversalFairnessEnv:
        """
        Create environment from a model and dataset.
        
        Args:
            model: Any model with predict and predict_proba methods
            X: Feature matrix
            y_true: True labels
            protected: Protected attribute values
            **kwargs: Additional arguments for UniversalFairnessEnv
        """
        # Get predictions from model
        predictions = model.predict(X)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if len(proba.shape) > 1:
                probabilities = proba[:, 1]  # Probability of class 1
            else:
                probabilities = proba
        else:
            # Use prediction as confidence
            probabilities = predictions.astype(float)
        
        return UniversalFairnessEnv(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=y_true,
            protected=protected,
            **kwargs
        )
    
    @staticmethod
    def from_synthetic(
        n_samples: int = 1000,
        bias_level: float = 0.3,
        minority_ratio: float = 0.3,
        base_accuracy: float = 0.8,
        seed: int = 42,
        **kwargs
    ) -> UniversalFairnessEnv:
        """
        Create environment from synthetic data for training.
        
        Args:
            n_samples: Number of samples
            bias_level: How biased the model is (0-1, higher = more biased)
            minority_ratio: Proportion of unprivileged group
            base_accuracy: Base model accuracy
            seed: Random seed
        """
        np.random.seed(seed)
        
        # Generate protected attribute
        protected = np.random.binomial(1, 1 - minority_ratio, n_samples)
        
        # Generate true labels (base rate may differ slightly by group)
        base_rate = 0.4
        true_labels = np.random.binomial(1, base_rate, n_samples)
        
        # Generate biased predictions
        predictions = true_labels.copy()
        
        # Add noise (model errors)
        error_mask = np.random.random(n_samples) > base_accuracy
        predictions[error_mask] = 1 - predictions[error_mask]
        
        # Add bias: unprivileged group gets denied more often
        unprivileged_mask = protected == 0
        bias_mask = np.random.random(n_samples) < bias_level
        predictions[unprivileged_mask & bias_mask] = 0
        
        # Generate probabilities (synthetic confidence)
        probabilities = np.where(
            predictions == 1,
            np.random.uniform(0.6, 0.95, n_samples),
            np.random.uniform(0.05, 0.4, n_samples)
        )
        
        return UniversalFairnessEnv(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=true_labels,
            protected=protected,
            **kwargs
        )


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing UniversalFairnessEnv")
    print("=" * 60)
    
    # Create synthetic environment
    env = UniversalFairnessEnvFactory.from_synthetic(
        n_samples=500,
        bias_level=0.4,
        minority_ratio=0.3,
        base_accuracy=0.75,
        render_mode="human"
    )
    
    print(f"\n📐 State space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Run a test episode
    obs, _ = env.reset()
    print(f"\n🔍 Initial observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 25 == 0:
            env.render()
        
        if terminated:
            break
    
    print("\n📊 Episode Summary:")
    summary = env.get_episode_summary()
    for k, v in summary.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
