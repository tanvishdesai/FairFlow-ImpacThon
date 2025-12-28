"""
SHAP Explainer Module

This module provides SHAP-based explanations for model predictions and
FairFlow interventions.
"""

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class ShapExplainer:
    """
    SHAP-based explainer for XGBoost model predictions.
    
    Generates feature attribution explanations for individual predictions
    and intervention decisions.
    """
    
    def __init__(self, model, feature_names: List[str], X_background: Optional[np.ndarray] = None):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            X_background: Background data for SHAP (if None, uses model's internal data)
        """
        self.model = model
        self.feature_names = feature_names
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(model)
        
        # Store background data if provided
        self.X_background = X_background
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for given samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            SHAP values array
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """
        Generate a detailed explanation for a single prediction.
        
        Args:
            X: Feature values for the sample
            sample_idx: Index for identification
            
        Returns:
            Dictionary with explanation details
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Create feature contributions
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(
            self.feature_names, X[0], shap_values[0]
        )):
            contributions.append({
                "feature": name,
                "value": float(value),
                "shap_value": float(shap_val),
                "abs_shap_value": abs(float(shap_val)),
                "direction": "positive" if shap_val > 0 else "negative"
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)
        
        return {
            "sample_idx": sample_idx,
            "prediction": int(prediction),
            "probability_approve": float(probability[1]),
            "probability_deny": float(probability[0]),
            "base_value": float(self.explainer.expected_value),
            "contributions": contributions,
            "top_positive": [c for c in contributions if c["direction"] == "positive"][:3],
            "top_negative": [c for c in contributions if c["direction"] == "negative"][:3]
        }
    
    def generate_waterfall_plot(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        max_display: int = 10,
        show_plot: bool = False
    ) -> str:
        """
        Generate a SHAP waterfall plot as a base64 encoded image.
        
        Args:
            X: Feature values for the sample
            sample_idx: Index for identification
            max_display: Maximum number of features to display
            show_plot: Whether to display the plot interactively
            
        Returns:
            Base64 encoded PNG image string
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer(X)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create waterfall plot
        shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
        
        # Customize plot
        plt.title(f"SHAP Explanation for Sample {sample_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return image_base64
    
    def generate_force_plot(
        self,
        X: np.ndarray,
        sample_idx: int = 0
    ) -> str:
        """
        Generate a SHAP force plot as a base64 encoded image.
        
        Args:
            X: Feature values for the sample
            sample_idx: Index for identification
            
        Returns:
            Base64 encoded PNG image string
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            X[0],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def generate_intervention_explanation(
        self,
        X: np.ndarray,
        base_prediction: int,
        final_decision: int,
        sample_idx: int = 0
    ) -> Dict:
        """
        Generate an explanation for why FairFlow intervened.
        
        Args:
            X: Feature values for the sample
            base_prediction: Original model prediction
            final_decision: FairFlow's final decision
            sample_idx: Index for identification
            
        Returns:
            Dictionary with intervention explanation
        """
        explanation = self.explain_prediction(X, sample_idx)
        
        # Determine intervention type
        if base_prediction == final_decision:
            intervention_type = "ACCEPTED"
            intervention_reason = "Base model decision was accepted without modification."
        elif base_prediction == 0 and final_decision == 1:
            intervention_type = "OVERRIDE_TO_APPROVE"
            intervention_reason = "FairFlow overrode DENIAL to APPROVAL to improve fairness."
        elif base_prediction == 1 and final_decision == 0:
            intervention_type = "OVERRIDE_TO_DENY"
            intervention_reason = "FairFlow overrode APPROVAL to DENIAL (rare case)."
        else:
            intervention_type = "ESCALATED"
            intervention_reason = "Decision escalated to human review."
        
        # Find potentially problematic features (proxies for protected attributes)
        proxy_features = ["age", "foreign_worker", "personal_status", "housing", 
                         "checking_status", "savings_status"]
        
        problematic_contributions = [
            c for c in explanation["contributions"]
            if any(pf in c["feature"].lower() for pf in proxy_features)
            and c["abs_shap_value"] > 0.1
        ]
        
        # Generate natural language explanation
        if problematic_contributions:
            top_problem = problematic_contributions[0]
            detailed_reason = (
                f"Intervention triggered because '{top_problem['feature']}' "
                f"contributed {abs(top_problem['shap_value']):.2f} to the decision. "
                f"This feature may correlate with protected attributes."
            )
        else:
            detailed_reason = (
                "Intervention triggered to maintain demographic parity across groups. "
                "Rolling fairness metrics indicated potential bias."
            )
        
        # Generate plot
        waterfall_image = self.generate_waterfall_plot(X, sample_idx)
        
        return {
            **explanation,
            "base_prediction": int(base_prediction),
            "final_decision": int(final_decision),
            "intervention_type": intervention_type,
            "intervention_reason": intervention_reason,
            "detailed_reason": detailed_reason,
            "problematic_features": problematic_contributions,
            "waterfall_plot": waterfall_image
        }


def create_explainer_from_model(
    model_path: str,
    feature_names_path: Optional[str] = None
) -> ShapExplainer:
    """
    Factory function to create a ShapExplainer from saved model.
    
    Args:
        model_path: Path to saved XGBoost model
        feature_names_path: Path to feature names file
        
    Returns:
        Configured ShapExplainer instance
    """
    import joblib
    
    model = joblib.load(model_path)
    
    if feature_names_path:
        with open(feature_names_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        # Try to find feature names file next to model
        model_dir = Path(model_path).parent
        feature_file = model_dir / "feature_names.txt"
        if feature_file.exists():
            with open(feature_file, "r") as f:
                feature_names = [line.strip() for line in f.readlines()]
        else:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    
    return ShapExplainer(model, feature_names)


if __name__ == "__main__":
    # Test the explainer
    print("Testing ShapExplainer...")
    
    import xgboost as xgb
    from sklearn.datasets import make_classification
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    feature_names = [f"feature_{i}" for i in range(20)]
    
    # Train a simple model
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ShapExplainer(model, feature_names)
    
    # Test explanation
    sample = X[0]
    explanation = explainer.explain_prediction(sample)
    
    print(f"\n📊 Prediction: {explanation['prediction']}")
    print(f"   Probability (approve): {explanation['probability_approve']:.4f}")
    print(f"\n   Top 3 Positive Contributors:")
    for c in explanation['top_positive']:
        print(f"      {c['feature']}: {c['shap_value']:+.4f}")
    print(f"\n   Top 3 Negative Contributors:")
    for c in explanation['top_negative']:
        print(f"      {c['feature']}: {c['shap_value']:+.4f}")
    
    # Generate waterfall plot
    print("\n🖼️ Generating waterfall plot...")
    image_b64 = explainer.generate_waterfall_plot(sample)
    print(f"   Generated image ({len(image_b64)} bytes base64)")
    
    # Test intervention explanation
    print("\n🔧 Testing intervention explanation...")
    intervention = explainer.generate_intervention_explanation(
        sample, base_prediction=0, final_decision=1
    )
    print(f"   Type: {intervention['intervention_type']}")
    print(f"   Reason: {intervention['intervention_reason']}")
