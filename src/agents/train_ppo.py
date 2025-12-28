"""
Train PPO Agent Script

This script trains a PPO agent on the FairnessEnv to learn optimal
bias mitigation strategies.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.fairness_env import FairnessEnv
from src.utils.data_loader import load_adult_data
from src.utils.metrics import calculate_all_metrics


class FairnessMetricsCallback(BaseCallback):
    """
    Custom callback to log fairness metrics during training.
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            
            summary = self.eval_env.get_episode_summary()
            self.metrics_history.append({
                "step": self.n_calls,
                **summary
            })
            
            if self.verbose > 0:
                print(f"\n📊 Step {self.n_calls}:")
                print(f"   Accuracy: {summary['final_accuracy']:.4f}")
                print(f"   DPR: {summary['final_dpr']:.4f}")
                print(f"   Intervention Rate: {summary['intervention_rate']:.4f}")
        
        return True


def train_ppo_agent(
    data_dir: str = "data",
    base_model_path: str = "models/base_model/xgboost_biased.joblib",
    output_dir: str = "models/rl_agent",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    accuracy_weight: float = 0.4,
    fairness_weight: float = 0.6,
    seed: int = 42,
    verbose: int = 1
):
    """
    Train a PPO agent for fairness-aware decision making.
    
    Args:
        data_dir: Path to data directory
        base_model_path: Path to the biased base model
        output_dir: Directory to save trained agent
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        accuracy_weight: Weight for accuracy in reward
        fairness_weight: Weight for fairness in reward
        seed: Random seed
        verbose: Verbosity level
    """
    print("\n" + "=" * 60)
    print("🤖 TRAINING PPO FAIRNESS AGENT")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📥 Loading data...")
    data = load_adult_data(data_dir=data_dir, protected_attribute="sex")
    
    # Load base model
    print(f"📦 Loading base model from {base_model_path}...")
    base_model = joblib.load(base_model_path)
    
    # Create training environment
    print("🌍 Creating training environment...")
    train_env = FairnessEnv(
        X=data["X_train"].values,
        y_true=data["y_train"].values,
        protected=data["protected_train"].values,
        base_model=base_model,
        feature_names=data["feature_names"],
        accuracy_weight=accuracy_weight,
        fairness_weight=fairness_weight
    )
    train_env = Monitor(train_env)
    
    # Create evaluation environment (on validation set)
    print("🔍 Creating evaluation environment...")
    eval_env = FairnessEnv(
        X=data["X_val"].values,
        y_true=data["y_val"].values,
        protected=data["protected_val"].values,
        base_model=base_model,
        feature_names=data["feature_names"],
        accuracy_weight=accuracy_weight,
        fairness_weight=fairness_weight
    )
    
    # Initialize PPO agent
    print("\n🏗️ Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        seed=seed,
        tensorboard_log=str(output_path / "tensorboard")
    )
    
    # Set up callbacks
    metrics_callback = FairnessMetricsCallback(eval_env, eval_freq=5000, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path),
        log_path=str(output_path / "logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    print(f"\n🎓 Training for {total_timesteps} timesteps...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = output_path / "ppo_fairness_agent"
    model.save(str(final_model_path))
    print(f"\n💾 Final model saved to {final_model_path}")
    
    # Save training metrics
    import json
    metrics_file = output_path / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_callback.metrics_history, f, indent=2)
    print(f"📊 Training metrics saved to {metrics_file}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION")
    print("=" * 60)
    
    evaluate_agent(model, data, base_model)
    
    return model


def evaluate_agent(model, data: dict, base_model, split: str = "test"):
    """
    Evaluate the trained agent on a data split.
    
    Args:
        model: Trained PPO agent
        data: Data dictionary
        base_model: The biased base model
        split: Which split to evaluate on
    """
    # Create test environment
    test_env = FairnessEnv(
        X=data[f"X_{split}"].values,
        y_true=data[f"y_{split}"].values,
        protected=data[f"protected_{split}"].values,
        base_model=base_model,
        feature_names=data["feature_names"]
    )
    
    # Collect predictions
    obs, _ = test_env.reset()
    done = False
    
    base_decisions = []
    final_decisions = []
    true_labels = []
    protected_vals = []
    interventions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        base_decisions.append(info["base_prediction"])
        final_decisions.append(info["final_decision"])
        true_labels.append(info["true_label"])
        protected_vals.append(info["protected_value"])
        interventions.append(info["intervened"])
    
    # Convert to arrays
    base_decisions = np.array(base_decisions)
    final_decisions = np.array(final_decisions)
    true_labels = np.array(true_labels)
    protected_vals = np.array(protected_vals)
    
    # Calculate metrics for base model
    print("\n🔴 BASE MODEL (Before FairFlow):")
    base_metrics = calculate_all_metrics(true_labels, base_decisions, protected_vals)
    print(f"   Accuracy:  {base_metrics['accuracy']:.4f}")
    print(f"   DPR:       {base_metrics['demographic_parity_ratio']:.4f}")
    print(f"   TPR Diff:  {base_metrics['tpr_difference']:.4f}")
    print(f"   Fair:      {'✅' if base_metrics['is_fair'] else '❌'}")
    
    # Calculate metrics for FairFlow
    print("\n🟢 FAIRFLOW (After Intervention):")
    fair_metrics = calculate_all_metrics(true_labels, final_decisions, protected_vals)
    print(f"   Accuracy:  {fair_metrics['accuracy']:.4f}")
    print(f"   DPR:       {fair_metrics['demographic_parity_ratio']:.4f}")
    print(f"   TPR Diff:  {fair_metrics['tpr_difference']:.4f}")
    print(f"   Fair:      {'✅' if fair_metrics['is_fair'] else '❌'}")
    
    # Intervention stats
    print("\n🔧 INTERVENTION STATISTICS:")
    intervention_rate = np.mean(interventions)
    print(f"   Total Samples:      {len(true_labels)}")
    print(f"   Interventions:      {sum(interventions)}")
    print(f"   Intervention Rate:  {intervention_rate:.2%}")
    
    # Improvement summary
    print("\n📊 IMPROVEMENT SUMMARY:")
    acc_change = fair_metrics['accuracy'] - base_metrics['accuracy']
    dpr_change = fair_metrics['demographic_parity_ratio'] - base_metrics['demographic_parity_ratio']
    print(f"   Accuracy Change:    {acc_change:+.4f}")
    print(f"   DPR Change:         {dpr_change:+.4f}")
    
    return {
        "base_metrics": base_metrics,
        "fair_metrics": fair_metrics,
        "intervention_rate": intervention_rate
    }


def main():
    """Main training pipeline."""
    # Set up paths
    script_dir = Path(__file__).parent.parent.parent
    data_dir = script_dir / "data"
    base_model_path = script_dir / "models" / "base_model" / "xgboost_biased.joblib"
    output_dir = script_dir / "models" / "rl_agent"
    
    print("🚀 FairFlow RL Agent Training Pipeline")
    print(f"   Data directory: {data_dir}")
    print(f"   Base model: {base_model_path}")
    print(f"   Output directory: {output_dir}")
    
    # Check if base model exists
    if not base_model_path.exists():
        print("\n❌ Error: Base model not found!")
        print("   Please run 'python src/train_base_model.py' first.")
        return
    
    # Train agent - using more timesteps for larger Adult Census dataset
    model = train_ppo_agent(
        data_dir=str(data_dir),
        base_model_path=str(base_model_path),
        output_dir=str(output_dir),
        total_timesteps=100000,  # Increased for Adult dataset (~23k training samples)
        accuracy_weight=0.3,    # Lower accuracy weight to focus on fairness
        fairness_weight=0.7,    # Higher fairness weight
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
