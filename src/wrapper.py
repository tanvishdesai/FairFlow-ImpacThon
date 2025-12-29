"""
FairFlow Universal Wrapper

A plug-and-play wrapper that adds bias mitigation to ANY machine learning model.
The wrapper uses a pre-trained universal RL agent that works across all datasets.

Example Usage:
    ```python
    from fairflow.wrapper import FairFlowWrapper
    
    # Wrap any model
    fair_model = FairFlowWrapper(
        base_model=your_sklearn_model,
        protected_attribute_index=0,  # or column name
        fairness_threshold=0.8
    )
    
    # Single prediction
    result = fair_model.predict_single(features, protected_value=0)
    
    # Batch prediction
    results = fair_model.predict(X, protected_values)
    ```
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import warnings

import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None
    warnings.warn("stable_baselines3 not installed. Some features may not work.")


@dataclass
class FairFlowPrediction:
    """Result of a FairFlow prediction."""
    base_prediction: int
    base_probability: float
    final_decision: int
    intervened: bool
    intervention_type: str  # "none", "approve_override", "deny_override"
    fairness_contribution: str  # "positive", "negative", "neutral"
    current_dpr: float
    
    def to_dict(self) -> dict:
        return {
            "base_prediction": self.base_prediction,
            "base_probability": self.base_probability,
            "final_decision": self.final_decision,
            "intervened": self.intervened,
            "intervention_type": self.intervention_type,
            "fairness_contribution": self.fairness_contribution,
            "current_dpr": self.current_dpr
        }


class FairFlowWrapper:
    """
    Universal FairFlow Wrapper - Add bias mitigation to any model.
    
    This wrapper uses a pre-trained universal RL agent that can handle
    any dataset without retraining. It maintains rolling fairness statistics
    and intervenes on predictions to maintain fairness thresholds.
    
    Attributes:
        STATE_DIM: Fixed state dimension (12) used by the universal agent
    """
    
    STATE_DIM = 12
    
    def __init__(
        self,
        base_model: Any = None,
        protected_attribute_index: Union[int, str] = None,
        fairness_threshold: float = 0.8,
        fairness_metric: str = "demographic_parity",
        window_size: int = 100,
        agent_path: str = None,
        fallback_mode: str = "threshold"
    ):
        """
        Initialize the FairFlow wrapper.
        
        Args:
            base_model: The base ML model to wrap. Must have predict() and 
                       optionally predict_proba() methods.
            protected_attribute_index: Index or column name of protected attribute
                                      in the feature matrix
            fairness_threshold: Minimum acceptable DPR (default 0.8, legal standard)
            fairness_metric: "demographic_parity" or "equalized_odds"
            window_size: Window for rolling fairness metrics
            agent_path: Path to pre-trained universal agent. If None, will try
                       to load from default location.
            fallback_mode: What to do if RL agent unavailable:
                          "threshold" - Use rule-based thresholds
                          "passthrough" - Return base predictions
                          "strict" - Raise error
        """
        self.base_model = base_model
        self.protected_attribute_index = protected_attribute_index
        self.fairness_threshold = fairness_threshold
        self.fairness_metric = fairness_metric
        self.window_size = window_size
        self.fallback_mode = fallback_mode
        
        # Load RL agent
        self.agent = None
        self._load_agent(agent_path)
        
        # Rolling statistics for fairness tracking
        self._reset_statistics()
        
        # Statistics for reporting
        self.total_predictions = 0
        self.total_interventions = 0
    
    def _load_agent(self, agent_path: str = None):
        """Load the pre-trained universal RL agent."""
        if PPO is None:
            print("⚠️ stable_baselines3 not available. Using fallback mode.")
            return
        
        if agent_path is None:
            # Try default locations
            possible_paths = [
                Path(__file__).parent.parent / "models" / "rl_agent" / "ppo_universal_fairness_agent.zip",
                Path(__file__).parent.parent / "models" / "rl_agent" / "ppo_universal_fairness_agent",
                Path(__file__).parent.parent / "models" / "rl_agent" / "ppo_fairness_agent.zip",
                Path(__file__).parent.parent / "models" / "rl_agent" / "ppo_fairness_agent",
                Path(__file__).parent.parent / "models" / "rl_agent" / "best_model.zip",
            ]
            
            for path in possible_paths:
                if path.exists() or Path(str(path) + ".zip").exists():
                    agent_path = str(path)
                    break
        
        if agent_path:
            try:
                self.agent = PPO.load(agent_path)
                print(f"✅ Loaded universal agent from {agent_path}")
            except Exception as e:
                print(f"⚠️ Could not load agent: {e}")
                print(f"   Using {self.fallback_mode} mode.")
    
    def _reset_statistics(self):
        """Reset rolling statistics."""
        self.privileged_decisions = deque(maxlen=self.window_size)
        self.unprivileged_decisions = deque(maxlen=self.window_size)
        self.privileged_confidences = deque(maxlen=self.window_size)
        self.unprivileged_confidences = deque(maxlen=self.window_size)
        
        # For TPR/FPR tracking
        self.privileged_outcomes = deque(maxlen=self.window_size)
        self.unprivileged_outcomes = deque(maxlen=self.window_size)
        
        self.last_protected_value = None
        self.consecutive_same_group = 0
    
    def _get_state(
        self,
        base_pred: int,
        base_prob: float,
        protected_value: int
    ) -> np.ndarray:
        """
        Build the 12-dimensional universal state vector.
        """
        # Calculate rolling metrics
        dpr = self._calculate_dpr()
        tpr_diff, fpr_diff = self._calculate_rate_diffs()
        priv_approval, unpriv_approval = self._calculate_approval_rates()
        
        # Intervention rate (based on recent history)
        total_recent = len(self.privileged_decisions) + len(self.unprivileged_decisions)
        intervention_rate = self.total_interventions / max(self.total_predictions, 1)
        
        # Group ratio
        n_unpriv = len(self.unprivileged_decisions)
        group_ratio = n_unpriv / max(total_recent, 1)
        
        # Consecutive same-group
        consecutive_norm = min(self.consecutive_same_group / self.window_size, 1.0)
        
        # Confidence gap
        confidence_gap = self._calculate_confidence_gap()
        
        state = np.array([
            float(base_pred),                    # 0: Base prediction
            base_prob,                           # 1: Base confidence
            float(protected_value),              # 2: Protected value
            self._normalize_dpr(dpr),            # 3: DPR (normalized)
            self._normalize_diff(tpr_diff),      # 4: TPR diff (normalized)
            self._normalize_diff(fpr_diff),      # 5: FPR diff (normalized)
            priv_approval,                       # 6: Privileged approval rate
            unpriv_approval,                     # 7: Unprivileged approval rate
            min(intervention_rate, 1.0),         # 8: Intervention rate
            group_ratio,                         # 9: Group ratio
            consecutive_norm,                    # 10: Consecutive same-group
            self._normalize_diff(confidence_gap) # 11: Confidence gap
        ], dtype=np.float32)
        
        return state
    
    def _calculate_dpr(self) -> float:
        """Calculate current Demographic Parity Ratio."""
        if len(self.privileged_decisions) < 3 or len(self.unprivileged_decisions) < 3:
            return 1.0
        
        priv_rate = np.mean(list(self.privileged_decisions))
        unpriv_rate = np.mean(list(self.unprivileged_decisions))
        
        if priv_rate == 0:
            return 0.0 if unpriv_rate > 0 else 1.0
        
        return min(unpriv_rate / priv_rate, 2.0)
    
    def _calculate_rate_diffs(self):
        """Calculate TPR and FPR differences (placeholders without true labels)."""
        # Without true labels in real-time, we estimate based on approval patterns
        return 0.0, 0.0
    
    def _calculate_approval_rates(self):
        """Calculate approval rates per group."""
        priv_rate = np.mean(list(self.privileged_decisions)) if self.privileged_decisions else 0.5
        unpriv_rate = np.mean(list(self.unprivileged_decisions)) if self.unprivileged_decisions else 0.5
        return priv_rate, unpriv_rate
    
    def _calculate_confidence_gap(self) -> float:
        """Calculate confidence gap between groups."""
        if not self.privileged_confidences or not self.unprivileged_confidences:
            return 0.0
        
        priv_conf = np.mean(list(self.privileged_confidences))
        unpriv_conf = np.mean(list(self.unprivileged_confidences))
        return priv_conf - unpriv_conf
    
    def _normalize_dpr(self, dpr: float) -> float:
        """Normalize DPR from [0, 2] to [0, 1]."""
        return min(max(dpr / 2.0, 0.0), 1.0)
    
    def _normalize_diff(self, diff: float) -> float:
        """Normalize from [-1, 1] to [0, 1]."""
        return min(max((diff + 1.0) / 2.0, 0.0), 1.0)
    
    def _update_statistics(self, decision: int, protected: int, prob: float):
        """Update rolling statistics after a decision."""
        if protected == 1:
            self.privileged_decisions.append(decision)
            self.privileged_confidences.append(prob)
        else:
            self.unprivileged_decisions.append(decision)
            self.unprivileged_confidences.append(prob)
        
        # Update consecutive tracking
        if self.last_protected_value == protected:
            self.consecutive_same_group += 1
        else:
            self.consecutive_same_group = 1
        self.last_protected_value = protected
    
    def _get_fallback_decision(
        self,
        base_pred: int,
        base_prob: float,
        protected_value: int
    ) -> tuple:
        """
        Rule-based fallback when RL agent is unavailable.
        """
        current_dpr = self._calculate_dpr()
        
        if current_dpr >= self.fairness_threshold:
            # Fair enough, pass through
            return base_pred, "none"
        
        # DPR is low - need to approve more unprivileged
        if protected_value == 0 and base_pred == 0:
            # Unprivileged being denied - consider overriding
            # Override if probability is borderline (>0.35)
            if base_prob > 0.35:
                return 1, "approve_override"
        
        return base_pred, "none"
    
    def predict_single(
        self,
        features: np.ndarray,
        protected_value: int = None
    ) -> FairFlowPrediction:
        """
        Make a single prediction with fairness intervention.
        
        Args:
            features: Feature vector for the sample
            protected_value: Protected attribute value (0 or 1).
                           If None, will extract from features using
                           protected_attribute_index.
        
        Returns:
            FairFlowPrediction object with decision details
        """
        # Get base model prediction
        features = np.array(features).reshape(1, -1)
        base_pred = int(self.base_model.predict(features)[0])
        
        # Get probability
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(features)
            base_prob = float(proba[0, 1]) if len(proba.shape) > 1 else float(proba[0])
        else:
            base_prob = float(base_pred)
        
        # Extract protected value if not provided
        if protected_value is None:
            if self.protected_attribute_index is not None:
                if isinstance(self.protected_attribute_index, int):
                    protected_value = int(features[0, self.protected_attribute_index])
                else:
                    raise ValueError("protected_attribute_index must be int for array input")
            else:
                raise ValueError("Must provide protected_value or set protected_attribute_index")
        
        # Get decision
        if self.agent is not None:
            state = self._get_state(base_pred, base_prob, protected_value)
            action, _ = self.agent.predict(state, deterministic=True)
            
            if action == 0:  # APPROVE (use base)
                final_decision = base_pred
                intervention_type = "none"
            elif action == 1:  # DENY
                final_decision = 0
                intervention_type = "deny_override" if base_pred != 0 else "none"
            else:  # ACCEPT
                final_decision = 1
                intervention_type = "approve_override" if base_pred != 1 else "none"
        else:
            final_decision, intervention_type = self._get_fallback_decision(
                base_pred, base_prob, protected_value
            )
        
        intervened = intervention_type != "none"
        
        # Update statistics
        self._update_statistics(final_decision, protected_value, base_prob)
        self.total_predictions += 1
        if intervened:
            self.total_interventions += 1
        
        # Determine fairness contribution
        current_dpr = self._calculate_dpr()
        if protected_value == 0 and final_decision == 1:
            fairness_contribution = "positive"
        elif protected_value == 0 and final_decision == 0:
            fairness_contribution = "negative" if current_dpr < self.fairness_threshold else "neutral"
        else:
            fairness_contribution = "neutral"
        
        return FairFlowPrediction(
            base_prediction=base_pred,
            base_probability=base_prob,
            final_decision=final_decision,
            intervened=intervened,
            intervention_type=intervention_type,
            fairness_contribution=fairness_contribution,
            current_dpr=current_dpr
        )
    
    def predict(
        self,
        X: np.ndarray,
        protected_values: np.ndarray = None
    ) -> np.ndarray:
        """
        Batch prediction with fairness intervention.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            protected_values: Protected attribute values (n_samples,).
                            If None, will extract from X.
        
        Returns:
            Array of final decisions
        """
        X = np.array(X)
        n_samples = len(X)
        
        # Extract protected values if not provided
        if protected_values is None:
            if self.protected_attribute_index is not None:
                protected_values = X[:, self.protected_attribute_index].astype(int)
            else:
                raise ValueError("Must provide protected_values or set protected_attribute_index")
        
        protected_values = np.array(protected_values)
        
        # Get base predictions
        base_preds = self.base_model.predict(X)
        
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)
            base_probs = proba[:, 1] if len(proba.shape) > 1 else proba
        else:
            base_probs = base_preds.astype(float)
        
        # Process each sample
        final_decisions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            result = self._process_single(
                int(base_preds[i]),
                float(base_probs[i]),
                int(protected_values[i])
            )
            final_decisions[i] = result
        
        return final_decisions
    
    def _process_single(self, base_pred: int, base_prob: float, protected_value: int) -> int:
        """Process a single sample (internal, no base model call)."""
        if self.agent is not None:
            state = self._get_state(base_pred, base_prob, protected_value)
            action, _ = self.agent.predict(state, deterministic=True)
            
            if action == 0:
                final_decision = base_pred
            elif action == 1:
                final_decision = 0
            else:
                final_decision = 1
        else:
            final_decision, _ = self._get_fallback_decision(base_pred, base_prob, protected_value)
        
        intervened = (final_decision != base_pred)
        
        self._update_statistics(final_decision, protected_value, base_prob)
        self.total_predictions += 1
        if intervened:
            self.total_interventions += 1
        
        return final_decision
    
    def predict_with_details(
        self,
        X: np.ndarray,
        protected_values: np.ndarray = None
    ) -> List[FairFlowPrediction]:
        """
        Batch prediction with full details for each sample.
        """
        X = np.array(X)
        
        if protected_values is None:
            if self.protected_attribute_index is not None:
                protected_values = X[:, self.protected_attribute_index].astype(int)
            else:
                raise ValueError("Must provide protected_values")
        
        results = []
        for i in range(len(X)):
            result = self.predict_single(X[i], protected_values[i])
            results.append(result)
        
        return results
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """
        Get current fairness metrics report.
        """
        dpr = self._calculate_dpr()
        priv_rate, unpriv_rate = self._calculate_approval_rates()
        
        return {
            "total_predictions": self.total_predictions,
            "total_interventions": self.total_interventions,
            "intervention_rate": self.total_interventions / max(self.total_predictions, 1),
            "demographic_parity_ratio": dpr,
            "is_fair": dpr >= self.fairness_threshold,
            "fairness_threshold": self.fairness_threshold,
            "privileged_approval_rate": priv_rate,
            "unprivileged_approval_rate": unpriv_rate,
            "privileged_samples": len(self.privileged_decisions),
            "unprivileged_samples": len(self.unprivileged_decisions),
            "agent_loaded": self.agent is not None
        }
    
    def reset_session(self):
        """Reset statistics for a new session."""
        self._reset_statistics()
        self.total_predictions = 0
        self.total_interventions = 0


def create_fairflow_wrapper(
    model,
    protected_attribute: Union[int, str],
    fairness_threshold: float = 0.8,
    agent_path: str = None
) -> FairFlowWrapper:
    """
    Convenience factory function to create a FairFlow wrapper.
    
    Args:
        model: Any sklearn-compatible model
        protected_attribute: Index or name of protected attribute
        fairness_threshold: Minimum acceptable DPR
        agent_path: Path to RL agent (optional)
    
    Returns:
        Configured FairFlowWrapper
    """
    return FairFlowWrapper(
        base_model=model,
        protected_attribute_index=protected_attribute,
        fairness_threshold=fairness_threshold,
        agent_path=agent_path
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing FairFlowWrapper")
    print("=" * 60)
    
    # Create a dummy biased model for testing
    class DummyBiasedModel:
        """A model that's biased against protected_value=0."""
        def predict(self, X):
            preds = (X[:, 0] > 0).astype(int)  # Simple threshold
            # Add bias: deny more when protected=0
            protected = X[:, -1]  # Assume last column is protected
            bias_mask = (protected == 0) & (np.random.random(len(X)) < 0.3)
            preds[bias_mask] = 0
            return preds
        
        def predict_proba(self, X):
            preds = self.predict(X)
            probs = np.column_stack([1 - preds, preds])
            return probs
    
    # Generate test data
    np.random.seed(42)
    n_samples = 200
    X_test = np.random.randn(n_samples, 5)
    X_test[:, -1] = np.random.binomial(1, 0.7, n_samples)  # Protected attribute
    
    # Test wrapper
    model = DummyBiasedModel()
    wrapper = FairFlowWrapper(
        base_model=model,
        protected_attribute_index=-1,  # Last column
        fairness_threshold=0.8
    )
    
    print("\n📊 Running predictions...")
    predictions = wrapper.predict(X_test)
    
    print("\n📈 Fairness Report:")
    report = wrapper.get_fairness_report()
    for k, v in report.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    print("\n✅ Wrapper test complete!")
