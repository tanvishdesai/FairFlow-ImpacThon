"""
Synthetic Data Generator for Universal FairFlow Training

Generates diverse fairness scenarios to train a dataset-agnostic RL agent.
The agent learns from many different bias patterns to generalize across domains.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class FairnessScenario:
    """A single fairness scenario configuration."""
    name: str
    n_samples: int
    minority_ratio: float  # Proportion of unprivileged group (0.1-0.5)
    bias_level: float      # How biased against minority (0-1)
    base_accuracy: float   # Base model accuracy (0.6-0.95)
    base_rate: float       # Base positive rate (0.2-0.6)
    
    def to_dict(self):
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "minority_ratio": self.minority_ratio,
            "bias_level": self.bias_level,
            "base_accuracy": self.base_accuracy,
            "base_rate": self.base_rate
        }


class SyntheticScenarioGenerator:
    """
    Generates synthetic fairness scenarios for RL training.
    
    The key insight is that to train a universal agent, we need to expose it
    to many different types of bias patterns during training.
    """
    
    # Pre-defined training scenarios covering the bias spectrum
    TRAINING_SCENARIOS = [
        # Mild bias scenarios
        FairnessScenario("mild_bias_balanced", 1000, 0.5, 0.2, 0.85, 0.4),
        FairnessScenario("mild_bias_minority_small", 1000, 0.3, 0.2, 0.85, 0.4),
        FairnessScenario("mild_bias_minority_tiny", 1000, 0.15, 0.2, 0.85, 0.4),
        
        # Moderate bias scenarios
        FairnessScenario("moderate_bias_balanced", 1000, 0.5, 0.4, 0.80, 0.35),
        FairnessScenario("moderate_bias_minority_small", 1000, 0.3, 0.4, 0.80, 0.35),
        FairnessScenario("moderate_bias_minority_tiny", 1000, 0.15, 0.4, 0.80, 0.35),
        
        # Severe bias scenarios
        FairnessScenario("severe_bias_balanced", 1000, 0.5, 0.6, 0.75, 0.3),
        FairnessScenario("severe_bias_minority_small", 1000, 0.3, 0.6, 0.75, 0.3),
        FairnessScenario("severe_bias_minority_tiny", 1000, 0.15, 0.6, 0.75, 0.3),
        
        # Extreme bias scenarios
        FairnessScenario("extreme_bias_balanced", 1000, 0.5, 0.8, 0.70, 0.3),
        FairnessScenario("extreme_bias_minority_small", 1000, 0.3, 0.8, 0.70, 0.3),
        
        # High accuracy model scenarios
        FairnessScenario("high_acc_mild_bias", 1000, 0.3, 0.25, 0.92, 0.4),
        FairnessScenario("high_acc_severe_bias", 1000, 0.3, 0.6, 0.92, 0.4),
        
        # Low accuracy model scenarios
        FairnessScenario("low_acc_mild_bias", 1000, 0.3, 0.2, 0.65, 0.4),
        FairnessScenario("low_acc_severe_bias", 1000, 0.3, 0.5, 0.65, 0.4),
        
        # Different base rates
        FairnessScenario("low_positive_rate", 1000, 0.3, 0.4, 0.80, 0.15),
        FairnessScenario("high_positive_rate", 1000, 0.3, 0.4, 0.80, 0.6),
    ]
    
    # Held-out scenarios for validation
    VALIDATION_SCENARIOS = [
        FairnessScenario("val_moderate_balanced", 500, 0.45, 0.35, 0.82, 0.38),
        FairnessScenario("val_severe_minority", 500, 0.25, 0.55, 0.78, 0.32),
        FairnessScenario("val_mild_high_acc", 500, 0.35, 0.25, 0.88, 0.42),
    ]
    
    @staticmethod
    def generate_scenario(scenario: FairnessScenario, seed: int = None) -> dict:
        """
        Generate synthetic data for a single scenario.
        
        Returns:
            dict with predictions, probabilities, true_labels, protected
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = scenario.n_samples
        
        # Generate protected attribute (0=unprivileged/minority, 1=privileged/majority)
        protected = np.random.binomial(1, 1 - scenario.minority_ratio, n)
        
        # Generate true labels with base rate
        true_labels = np.random.binomial(1, scenario.base_rate, n)
        
        # Start with predictions = true labels (perfect model)
        predictions = true_labels.copy()
        
        # Add random errors based on accuracy
        error_indices = np.random.choice(
            n, 
            size=int(n * (1 - scenario.base_accuracy)), 
            replace=False
        )
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Apply bias against unprivileged group
        unprivileged_mask = protected == 0
        n_unprivileged = np.sum(unprivileged_mask)
        
        # Bias: flip some unprivileged approvals to denials
        unprivileged_positive = unprivileged_mask & (predictions == 1)
        n_to_flip = int(np.sum(unprivileged_positive) * scenario.bias_level)
        
        if n_to_flip > 0:
            flip_indices = np.random.choice(
                np.where(unprivileged_positive)[0],
                size=min(n_to_flip, np.sum(unprivileged_positive)),
                replace=False
            )
            predictions[flip_indices] = 0
        
        # Generate probabilities (confidence scores)
        probabilities = np.zeros(n)
        
        # Positive predictions get high confidence
        pos_indices = predictions == 1
        probabilities[pos_indices] = np.random.uniform(0.55, 0.95, np.sum(pos_indices))
        
        # Negative predictions get low confidence
        neg_indices = predictions == 0
        probabilities[neg_indices] = np.random.uniform(0.05, 0.45, np.sum(neg_indices))
        
        # Add slight confidence bias (unprivileged get lower confidence on average)
        probabilities[unprivileged_mask] *= (1 - 0.1 * scenario.bias_level)
        probabilities = np.clip(probabilities, 0, 1)
        
        # Calculate initial DPR for logging
        priv_rate = np.mean(predictions[protected == 1])
        unpriv_rate = np.mean(predictions[protected == 0])
        initial_dpr = unpriv_rate / priv_rate if priv_rate > 0 else 0.0
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "true_labels": true_labels,
            "protected": protected,
            "scenario": scenario.to_dict(),
            "initial_metrics": {
                "accuracy": np.mean(predictions == true_labels),
                "dpr": initial_dpr,
                "privileged_approval_rate": priv_rate,
                "unprivileged_approval_rate": unpriv_rate,
                "n_privileged": int(np.sum(protected == 1)),
                "n_unprivileged": int(np.sum(protected == 0))
            }
        }
    
    @staticmethod
    def generate_random_scenario(
        n_samples: int = 1000,
        seed: int = None
    ) -> dict:
        """
        Generate a completely random scenario.
        Good for augmenting training data.
        """
        if seed is not None:
            np.random.seed(seed)
        
        scenario = FairnessScenario(
            name=f"random_{seed}",
            n_samples=n_samples,
            minority_ratio=np.random.uniform(0.1, 0.5),
            bias_level=np.random.uniform(0.1, 0.8),
            base_accuracy=np.random.uniform(0.6, 0.95),
            base_rate=np.random.uniform(0.15, 0.6)
        )
        
        return SyntheticScenarioGenerator.generate_scenario(scenario, seed)
    
    @staticmethod
    def generate_all_training_data(
        base_seed: int = 42,
        augment_random: int = 10
    ) -> List[dict]:
        """
        Generate all training scenarios.
        
        Args:
            base_seed: Base random seed
            augment_random: Number of random scenarios to add
        """
        all_data = []
        
        # Generate pre-defined scenarios
        for i, scenario in enumerate(SyntheticScenarioGenerator.TRAINING_SCENARIOS):
            data = SyntheticScenarioGenerator.generate_scenario(
                scenario, seed=base_seed + i
            )
            all_data.append(data)
            print(f"Generated: {scenario.name} | DPR: {data['initial_metrics']['dpr']:.3f}")
        
        # Generate random augmentation scenarios
        for i in range(augment_random):
            data = SyntheticScenarioGenerator.generate_random_scenario(
                n_samples=1000,
                seed=base_seed + 1000 + i
            )
            all_data.append(data)
            print(f"Generated: {data['scenario']['name']} | DPR: {data['initial_metrics']['dpr']:.3f}")
        
        return all_data
    
    @staticmethod
    def generate_validation_data(base_seed: int = 9999) -> List[dict]:
        """Generate validation scenarios."""
        all_data = []
        
        for i, scenario in enumerate(SyntheticScenarioGenerator.VALIDATION_SCENARIOS):
            data = SyntheticScenarioGenerator.generate_scenario(
                scenario, seed=base_seed + i
            )
            all_data.append(data)
        
        return all_data


class CurriculumScheduler:
    """
    Curriculum learning scheduler for training.
    
    Starts with easy scenarios (mild bias) and progressively
    introduces harder scenarios (severe bias, small minorities).
    """
    
    def __init__(self, scenarios: List[dict], n_stages: int = 4):
        self.all_scenarios = scenarios
        self.n_stages = n_stages
        
        # Sort scenarios by difficulty (bias_level * (1 - minority_ratio))
        self.sorted_scenarios = sorted(
            scenarios,
            key=lambda x: x['scenario']['bias_level'] * (1 - x['scenario']['minority_ratio'])
        )
        
        # Split into stages
        n = len(self.sorted_scenarios)
        self.stages = []
        for i in range(n_stages):
            start = 0
            end = int((i + 1) / n_stages * n)
            self.stages.append(self.sorted_scenarios[:end])
    
    def get_scenarios_for_stage(self, stage: int) -> List[dict]:
        """Get scenarios available at a given training stage."""
        stage = min(stage, self.n_stages - 1)
        return self.stages[stage]
    
    def get_stage_for_timestep(self, timestep: int, total_timesteps: int) -> int:
        """Determine which stage we're in based on progress."""
        progress = timestep / total_timesteps
        stage = int(progress * self.n_stages)
        return min(stage, self.n_stages - 1)


if __name__ == "__main__":
    print("=" * 60)
    print("🎲 Testing Synthetic Scenario Generator")
    print("=" * 60)
    
    # Generate a few test scenarios
    gen = SyntheticScenarioGenerator()
    
    print("\n📊 Generating Training Scenarios:")
    training_data = gen.generate_all_training_data(base_seed=42, augment_random=5)
    
    print(f"\n✅ Generated {len(training_data)} training scenarios")
    
    print("\n📊 Generating Validation Scenarios:")
    val_data = gen.generate_validation_data()
    
    print(f"\n✅ Generated {len(val_data)} validation scenarios")
    
    # Test curriculum scheduler
    print("\n📚 Testing Curriculum Scheduler:")
    scheduler = CurriculumScheduler(training_data, n_stages=4)
    
    for stage in range(4):
        scenarios = scheduler.get_scenarios_for_stage(stage)
        print(f"  Stage {stage}: {len(scenarios)} scenarios available")
    
    # Show example scenario
    print("\n🔍 Example Scenario Details:")
    example = training_data[5]
    print(f"  Name: {example['scenario']['name']}")
    print(f"  Minority Ratio: {example['scenario']['minority_ratio']:.2%}")
    print(f"  Bias Level: {example['scenario']['bias_level']:.2%}")
    print(f"  Initial Accuracy: {example['initial_metrics']['accuracy']:.2%}")
    print(f"  Initial DPR: {example['initial_metrics']['dpr']:.3f}")
