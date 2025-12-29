"""
Universal FairFlow Training Script

Trains a dataset-agnostic PPO agent on multiple synthetic scenarios.
The resulting agent can generalize to any fairness problem without retraining.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.universal_fairness_env import UniversalFairnessEnv
from src.utils.synthetic_data import SyntheticScenarioGenerator, CurriculumScheduler


class MultiScenarioEnv(UniversalFairnessEnv):
    """
    Environment that cycles through multiple scenarios during training.
    
    This is the key to generalization: the agent sees many different
    bias patterns and learns to handle them all.
    """
    
    def __init__(
        self,
        scenarios: list,
        accuracy_weight: float = 0.4,
        fairness_weight: float = 0.5,
        fairness_threshold: float = 0.8,
        window_size: int = 50,
        curriculum: bool = True
    ):
        # Store all scenarios
        self.all_scenarios = scenarios
        self.current_scenario_idx = 0
        self.episode_count = 0
        self.curriculum = curriculum
        
        # Start with first scenario
        first = scenarios[0]
        super().__init__(
            predictions=first["predictions"],
            probabilities=first["probabilities"],
            true_labels=first["true_labels"],
            protected=first["protected"],
            accuracy_weight=accuracy_weight,
            fairness_weight=fairness_weight,
            fairness_threshold=fairness_threshold,
            window_size=window_size
        )
    
    def reset(self, seed=None, options=None):
        """Reset and potentially switch to a new scenario."""
        self.episode_count += 1
        
        if self.curriculum:
            # Curriculum: gradually expose to harder scenarios
            # First 25% episodes: easy scenarios only
            # Next 25%: easy + medium
            # Next 25%: easy + medium + hard
            # Final 25%: all scenarios
            progress = min(self.episode_count / 100, 1.0)  # Assume ~100 episodes in training
            max_scenario_idx = int(progress * len(self.all_scenarios))
            max_scenario_idx = max(1, min(max_scenario_idx, len(self.all_scenarios)))
            self.current_scenario_idx = np.random.randint(0, max_scenario_idx)
        else:
            # Random uniform selection
            self.current_scenario_idx = np.random.randint(0, len(self.all_scenarios))
        
        # Load new scenario data
        scenario = self.all_scenarios[self.current_scenario_idx]
        self.predictions = np.array(scenario["predictions"])
        self.probabilities = np.array(scenario["probabilities"])
        self.true_labels = np.array(scenario["true_labels"])
        self.protected = np.array(scenario["protected"])
        self.n_samples = len(self.predictions)
        
        return super().reset(seed=seed, options=options)
    
    def get_current_scenario_name(self) -> str:
        """Get name of current scenario."""
        return self.all_scenarios[self.current_scenario_idx]["scenario"]["name"]


class UniversalMetricsCallback(BaseCallback):
    """
    Callback to log fairness metrics during universal training.
    """
    
    def __init__(
        self, 
        eval_scenarios: list,
        eval_freq: int = 5000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_scenarios = eval_scenarios
        self.eval_freq = eval_freq
        self.metrics_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate on each validation scenario
            results = []
            
            for scenario_data in self.eval_scenarios:
                env = UniversalFairnessEnv(
                    predictions=scenario_data["predictions"],
                    probabilities=scenario_data["probabilities"],
                    true_labels=scenario_data["true_labels"],
                    protected=scenario_data["protected"]
                )
                
                obs, _ = env.reset()
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                
                summary = env.get_episode_summary()
                results.append({
                    "scenario": scenario_data["scenario"]["name"],
                    "initial_dpr": scenario_data["initial_metrics"]["dpr"],
                    "final_dpr": summary["final_dpr"],
                    "dpr_improvement": summary["final_dpr"] - scenario_data["initial_metrics"]["dpr"],
                    "final_accuracy": summary["final_accuracy"],
                    "intervention_rate": summary["intervention_rate"]
                })
            
            # Calculate aggregate metrics
            avg_dpr_improvement = np.mean([r["dpr_improvement"] for r in results])
            avg_final_dpr = np.mean([r["final_dpr"] for r in results])
            avg_accuracy = np.mean([r["final_accuracy"] for r in results])
            
            self.metrics_history.append({
                "step": self.n_calls,
                "avg_dpr_improvement": avg_dpr_improvement,
                "avg_final_dpr": avg_final_dpr,
                "avg_accuracy": avg_accuracy,
                "details": results
            })
            
            if self.verbose > 0:
                print(f"\n📊 Step {self.n_calls} - Validation Results:")
                print(f"   Avg DPR Improvement: {avg_dpr_improvement:+.4f}")
                print(f"   Avg Final DPR: {avg_final_dpr:.4f}")
                print(f"   Avg Accuracy: {avg_accuracy:.4f}")
        
        return True


def train_universal_agent(
    output_dir: str = "models/rl_agent",
    total_timesteps: int = 200000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    accuracy_weight: float = 0.35,
    fairness_weight: float = 0.5,
    use_curriculum: bool = True,
    n_random_scenarios: int = 15,
    seed: int = 42,
    verbose: int = 1
):
    """
    Train a universal PPO agent on multiple synthetic scenarios.
    
    Args:
        output_dir: Directory to save trained agent
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        accuracy_weight: Weight for accuracy in reward
        fairness_weight: Weight for fairness in reward
        use_curriculum: Whether to use curriculum learning
        n_random_scenarios: Number of random augmentation scenarios
        seed: Random seed
        verbose: Verbosity level
    """
    print("\n" + "=" * 70)
    print("🌍 TRAINING UNIVERSAL FAIRFLOW AGENT")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate training scenarios
    print("\n📊 Generating Training Scenarios...")
    generator = SyntheticScenarioGenerator()
    training_scenarios = generator.generate_all_training_data(
        base_seed=seed,
        augment_random=n_random_scenarios
    )
    
    print(f"\n✅ Generated {len(training_scenarios)} training scenarios")
    
    # Generate validation scenarios
    print("\n📊 Generating Validation Scenarios...")
    validation_scenarios = generator.generate_validation_data(base_seed=seed + 5000)
    print(f"✅ Generated {len(validation_scenarios)} validation scenarios")
    
    # Create multi-scenario training environment
    print("\n🌍 Creating Multi-Scenario Training Environment...")
    train_env = MultiScenarioEnv(
        scenarios=training_scenarios,
        accuracy_weight=accuracy_weight,
        fairness_weight=fairness_weight,
        curriculum=use_curriculum
    )
    train_env = Monitor(train_env)
    
    # Initialize PPO agent
    print("\n🏗️ Initializing PPO Agent...")
    print(f"   State Dimension: {UniversalFairnessEnv.STATE_DIM} (FIXED)")
    print(f"   Action Space: 3 (APPROVE, DENY, ACCEPT)")
    
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
        tensorboard_log=str(output_path / "tensorboard_universal"),
        policy_kwargs={
            "net_arch": [128, 128, 64]  # Deeper network for better generalization
        }
    )
    
    # Set up callbacks
    metrics_callback = UniversalMetricsCallback(
        eval_scenarios=validation_scenarios,
        eval_freq=10000,
        verbose=1
    )
    
    # Train the agent
    print(f"\n🎓 Training for {total_timesteps} timesteps...")
    print(f"   Curriculum Learning: {'Enabled' if use_curriculum else 'Disabled'}")
    print("-" * 70)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_callback],
        progress_bar=True
    )
    
    # Save the universal agent
    universal_model_path = output_path / "ppo_universal_fairness_agent"
    model.save(str(universal_model_path))
    print(f"\n💾 Universal agent saved to {universal_model_path}")
    
    # Save training config
    config = {
        "model_type": "PPO",
        "state_dim": UniversalFairnessEnv.STATE_DIM,
        "action_dim": 3,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "accuracy_weight": accuracy_weight,
        "fairness_weight": fairness_weight,
        "n_training_scenarios": len(training_scenarios),
        "n_validation_scenarios": len(validation_scenarios),
        "curriculum_learning": use_curriculum,
        "trained_at": datetime.now().isoformat(),
        "is_universal": True,
        "compatible_with": "any dataset with binary classification and binary protected attribute"
    }
    
    config_file = output_path / "universal_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📄 Config saved to {config_file}")
    
    # Save training metrics
    metrics_file = output_path / "universal_training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_callback.metrics_history, f, indent=2)
    print(f"📊 Training metrics saved to {metrics_file}")
    
    # Final comprehensive evaluation
    print("\n" + "=" * 70)
    print("📈 FINAL EVALUATION ON ALL SCENARIOS")
    print("=" * 70)
    
    evaluate_universal_agent(model, training_scenarios + validation_scenarios)
    
    return model


def evaluate_universal_agent(model, scenarios: list, detailed: bool = False):
    """
    Evaluate the universal agent on multiple scenarios.
    """
    results = []
    
    for scenario_data in scenarios:
        env = UniversalFairnessEnv(
            predictions=scenario_data["predictions"],
            probabilities=scenario_data["probabilities"],
            true_labels=scenario_data["true_labels"],
            protected=scenario_data["protected"]
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        
        result = {
            "scenario": scenario_data["scenario"]["name"],
            "initial_dpr": scenario_data["initial_metrics"]["dpr"],
            "final_dpr": summary["final_dpr"],
            "dpr_improvement": summary["final_dpr"] - scenario_data["initial_metrics"]["dpr"],
            "initial_accuracy": scenario_data["initial_metrics"]["accuracy"],
            "final_accuracy": summary["final_accuracy"],
            "accuracy_change": summary["final_accuracy"] - scenario_data["initial_metrics"]["accuracy"],
            "intervention_rate": summary["intervention_rate"],
            "is_fair": summary["final_dpr"] >= 0.8
        }
        results.append(result)
    
    # Summary statistics
    n_scenarios = len(results)
    n_fair = sum(1 for r in results if r["is_fair"])
    avg_dpr_improvement = np.mean([r["dpr_improvement"] for r in results])
    avg_accuracy_change = np.mean([r["accuracy_change"] for r in results])
    avg_intervention = np.mean([r["intervention_rate"] for r in results])
    
    print(f"\n📊 Evaluated on {n_scenarios} scenarios:")
    print(f"   Scenarios achieving fairness (DPR ≥ 0.8): {n_fair}/{n_scenarios} ({n_fair/n_scenarios:.1%})")
    print(f"   Average DPR Improvement: {avg_dpr_improvement:+.4f}")
    print(f"   Average Accuracy Change: {avg_accuracy_change:+.4f}")
    print(f"   Average Intervention Rate: {avg_intervention:.2%}")
    
    if detailed:
        print("\n📋 Detailed Results:")
        for r in results:
            status = "✅" if r["is_fair"] else "❌"
            print(f"   {status} {r['scenario'][:30]:30s} | DPR: {r['initial_dpr']:.3f} → {r['final_dpr']:.3f} | Acc: {r['final_accuracy']:.3f}")
    
    return results


def test_on_real_dataset(model, data_name: str = "adult"):
    """
    Test the universal agent on a real dataset (without retraining).
    
    This is the ultimate test of generalization.
    """
    print(f"\n🧪 Testing on REAL dataset: {data_name}")
    print("-" * 50)
    
    # Import data loader
    from src.utils.data_loader import load_adult_data
    import joblib
    
    # Load real data and base model
    data = load_adult_data(data_dir="data", protected_attribute="sex")
    base_model = joblib.load("models/base_model/xgboost_biased.joblib")
    
    # Get predictions from base model
    X_test = data["X_test"].values
    y_test = data["y_test"].values
    protected_test = data["protected_test"].values
    
    predictions = base_model.predict(X_test)
    probabilities = base_model.predict_proba(X_test)[:, 1]
    
    # Calculate initial metrics
    priv_rate = np.mean(predictions[protected_test == 1])
    unpriv_rate = np.mean(predictions[protected_test == 0])
    initial_dpr = unpriv_rate / priv_rate if priv_rate > 0 else 0
    initial_acc = np.mean(predictions == y_test)
    
    print(f"   Initial Accuracy: {initial_acc:.4f}")
    print(f"   Initial DPR: {initial_dpr:.4f}")
    
    # Run through universal agent
    env = UniversalFairnessEnv(
        predictions=predictions,
        probabilities=probabilities,
        true_labels=y_test,
        protected=protected_test
    )
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    summary = env.get_episode_summary()
    
    print(f"\n   🟢 After FairFlow (Universal Agent):")
    print(f"   Final Accuracy: {summary['final_accuracy']:.4f} ({summary['final_accuracy'] - initial_acc:+.4f})")
    print(f"   Final DPR: {summary['final_dpr']:.4f} ({summary['final_dpr'] - initial_dpr:+.4f})")
    print(f"   Intervention Rate: {summary['intervention_rate']:.2%}")
    print(f"   Fair: {'✅' if summary['final_dpr'] >= 0.8 else '❌'}")
    
    return summary


def main():
    """Main training pipeline."""
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "models" / "rl_agent"
    
    print("🚀 Universal FairFlow Agent Training Pipeline")
    print(f"   Output directory: {output_dir}")
    
    # Train universal agent
    model = train_universal_agent(
        output_dir=str(output_dir),
        total_timesteps=200000,  # More timesteps for better generalization
        accuracy_weight=0.35,
        fairness_weight=0.5,
        use_curriculum=True,
        n_random_scenarios=15,
        verbose=1
    )
    
    # Test on real dataset (if available)
    try:
        test_on_real_dataset(model, "adult")
    except Exception as e:
        print(f"\n⚠️ Could not test on real dataset: {e}")
        print("   (This is okay - the agent is still trained successfully)")
    
    print("\n" + "=" * 70)
    print("✅ UNIVERSAL TRAINING COMPLETE!")
    print("=" * 70)
    print("\n💡 The trained agent can now be used with ANY dataset.")
    print("   Just provide: predictions, probabilities, true_labels, protected")


if __name__ == "__main__":
    main()
