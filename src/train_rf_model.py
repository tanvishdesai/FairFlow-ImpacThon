"""
Train Random Forest Model Script

This script trains a deliberately biased Random Forest classifier on the Adult Census
dataset. This serves as an alternative "bad actor" to demonstrate FairFlow's
model-agnostic capabilities.
"""

import os
import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_adult_data
from src.utils.metrics import calculate_all_metrics, print_metrics_report


def train_biased_rf_model(data: dict, model_dir: str = "models/rf_model") -> RandomForestClassifier:
    """
    Train a deliberately biased Random Forest model.
    
    The model is trained WITHOUT any fairness constraints, allowing it to
    exploit correlations with protected attributes like sex and race.
    
    Args:
        data: Dictionary from load_adult_data
        model_dir: Directory to save the trained model
        
    Returns:
        Trained Random Forest classifier
    """
    print("\n" + "=" * 60)
    print("🎓 TRAINING BIASED RANDOM FOREST MODEL")
    print("=" * 60)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Features: {len(X_train.columns)}")
    
    # Train Random Forest classifier (no fairness constraints)
    # Using parameters that might allow for some overfitting/bias retention
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")
    
    # Save model
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    model_file = model_path / "random_forest_biased.joblib"
    joblib.dump(model, model_file)
    print(f"💾 Model saved to {model_file}")
    
    # Save feature names for later use
    feature_file = model_path / "feature_names.txt"
    with open(feature_file, "w") as f:
        f.write("\n".join(data["feature_names"]))
    
    # Save protected attribute info
    info_file = model_path / "model_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Dataset: Adult Census\n")
        f.write(f"Model: Random Forest\n")
        f.write(f"Protected Attribute: {data['protected_attribute']}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Features: {len(X_train.columns)}\n")
    
    return model


def evaluate_model(model, data: dict, split: str = "test") -> dict:
    """
    Evaluate the model on a data split and calculate fairness metrics.
    
    Args:
        model: Trained classifier
        data: Dictionary from load_adult_data
        split: Which split to evaluate on ("train", "val", "test")
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating on {split} set...")
    
    X = data[f"X_{split}"]
    y_true = data[f"y_{split}"]
    protected = data[f"protected_{split}"]
    
    y_pred = model.predict(X)
    
    # Calculate all metrics
    metrics = calculate_all_metrics(y_true.values, y_pred, protected.values)
    
    return metrics


def analyze_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Analyze feature importance to identify potential proxy discrimination.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    print("\n📈 Top 10 Feature Importances:")
    print("-" * 40)
    for i, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"   {row['feature'][:25]:<25} {row['importance']:.4f} {bar}")
    
    return importance_df


def main():
    """Main training pipeline."""
    # Set up paths relative to script location
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data"
    model_dir = script_dir / "models" / "rf_model"
    
    print("🚀 FairFlow Random Forest Model Training Pipeline")
    print(f"   Data directory: {data_dir}")
    print(f"   Model directory: {model_dir}")
    
    # Load data - using SEX as protected attribute for strong bias demonstration
    print("\n📥 Loading Adult Census dataset...")
    data = load_adult_data(
        data_dir=str(data_dir),
        protected_attribute="sex",  # Using sex for strong bias demonstration
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Train model
    model = train_biased_rf_model(data, model_dir=str(model_dir))
    
    # Evaluate on all splits
    for split in ["train", "val", "test"]:
        metrics = evaluate_model(model, data, split)
        print_metrics_report(metrics)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, data["feature_names"])
    importance_df.to_csv(model_dir / "feature_importance.csv", index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TRAINING SUMMARY")
    print("=" * 60)
    print(f"   ✅ Model trained on Adult Census dataset ({len(data['X_train']):,} samples)")
    print(f"   ✅ Model saved to: {model_dir / 'random_forest_biased.joblib'}")
    print(f"   ✅ Feature importance saved to: {model_dir / 'feature_importance.csv'}")
    print(f"\n   ⚠️  This model is DELIBERATELY BIASED!")
    print(f"   Protected attribute: {data['protected_attribute']}")
    print("   It will be used as the alternative 'bad actor' to demonstrate FairFlow's model agnosticism.")
    print("=" * 60)


if __name__ == "__main__":
    main()
