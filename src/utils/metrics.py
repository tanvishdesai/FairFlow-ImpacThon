"""
Fairness Metrics Module

This module provides functions to calculate various fairness metrics
used in the FairFlow bias detection and mitigation system.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_demographic_parity(
    y_pred: np.ndarray,
    protected: np.ndarray,
    positive_label: int = 1
) -> Dict[str, float]:
    """
    Calculate Demographic Parity (Statistical Parity) metrics.
    
    Demographic Parity requires that the probability of positive prediction
    is equal across all groups.
    
    DP Ratio = P(Y_pred=1 | Protected=0) / P(Y_pred=1 | Protected=1)
    
    A ratio close to 1.0 indicates fairness. Legal thresholds often use 0.8-1.25.
    
    Args:
        y_pred: Predicted labels (0 or 1)
        protected: Protected attribute values (0 = unprivileged, 1 = privileged)
        positive_label: The label considered "positive" (usually 1 = approve)
        
    Returns:
        Dictionary with demographic parity metrics
    """
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)
    
    # Calculate approval rates for each group
    privileged_mask = protected == 1
    unprivileged_mask = protected == 0
    
    privileged_approval_rate = np.mean(y_pred[privileged_mask] == positive_label)
    unprivileged_approval_rate = np.mean(y_pred[unprivileged_mask] == positive_label)
    
    # Calculate ratio (avoid division by zero)
    if privileged_approval_rate == 0:
        dp_ratio = float('inf') if unprivileged_approval_rate > 0 else 1.0
    else:
        dp_ratio = unprivileged_approval_rate / privileged_approval_rate
    
    # Calculate difference
    dp_difference = unprivileged_approval_rate - privileged_approval_rate
    
    return {
        "demographic_parity_ratio": dp_ratio,
        "demographic_parity_difference": dp_difference,
        "privileged_approval_rate": privileged_approval_rate,
        "unprivileged_approval_rate": unprivileged_approval_rate,
        "is_dp_fair": 0.8 <= dp_ratio <= 1.25  # Common legal threshold for demographic parity
    }


def calculate_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Equalized Odds metrics.
    
    Equalized Odds requires that the True Positive Rate (TPR) and 
    False Positive Rate (FPR) are equal across groups.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        protected: Protected attribute values (0 = unprivileged, 1 = privileged)
        
    Returns:
        Dictionary with equalized odds metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected = np.asarray(protected)
    
    metrics = {}
    
    for group_name, group_val in [("privileged", 1), ("unprivileged", 0)]:
        group_mask = protected == group_val
        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]
        
        # True Positive Rate = TP / (TP + FN)
        positive_mask = group_true == 1
        if positive_mask.sum() > 0:
            tpr = np.mean(group_pred[positive_mask] == 1)
        else:
            tpr = 0.0
        
        # False Positive Rate = FP / (FP + TN)
        negative_mask = group_true == 0
        if negative_mask.sum() > 0:
            fpr = np.mean(group_pred[negative_mask] == 1)
        else:
            fpr = 0.0
        
        metrics[f"{group_name}_tpr"] = tpr
        metrics[f"{group_name}_fpr"] = fpr
    
    # Calculate differences
    metrics["tpr_difference"] = metrics["unprivileged_tpr"] - metrics["privileged_tpr"]
    metrics["fpr_difference"] = metrics["unprivileged_fpr"] - metrics["privileged_fpr"]
    
    # Calculate ratios
    if metrics["privileged_tpr"] > 0:
        metrics["tpr_ratio"] = metrics["unprivileged_tpr"] / metrics["privileged_tpr"]
    else:
        metrics["tpr_ratio"] = float('inf') if metrics["unprivileged_tpr"] > 0 else 1.0
    
    if metrics["privileged_fpr"] > 0:
        metrics["fpr_ratio"] = metrics["unprivileged_fpr"] / metrics["privileged_fpr"]
    else:
        metrics["fpr_ratio"] = float('inf') if metrics["unprivileged_fpr"] > 0 else 1.0
    
    # Equalized odds is fair if both TPR and FPR differences are small
    metrics["is_eo_fair"] = (abs(metrics["tpr_difference"]) < 0.1 and 
                              abs(metrics["fpr_difference"]) < 0.1)
    
    return metrics


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate overall accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all fairness and performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        protected: Protected attribute values
        
    Returns:
        Dictionary with all metrics
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    dp_metrics = calculate_demographic_parity(y_pred, protected)
    eo_metrics = calculate_equalized_odds(y_true, y_pred, protected)
    
    # Overall fairness check - primarily based on demographic parity (regulatory standard)
    is_fair = dp_metrics.get("is_dp_fair", False)
    
    return {
        "accuracy": accuracy,
        "is_fair": is_fair,  # Overall fairness based on DPR
        **dp_metrics,
        **eo_metrics
    }


def print_metrics_report(metrics: Dict[str, float]) -> None:
    """
    Print a formatted fairness metrics report.
    
    Args:
        metrics: Dictionary of calculated metrics
    """
    print("\n" + "=" * 60)
    print("📊 FAIRNESS METRICS REPORT")
    print("=" * 60)
    
    print(f"\n🎯 Performance:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    print(f"\n⚖️  Demographic Parity:")
    print(f"   Privileged Approval Rate:   {metrics.get('privileged_approval_rate', 0):.4f}")
    print(f"   Unprivileged Approval Rate: {metrics.get('unprivileged_approval_rate', 0):.4f}")
    print(f"   DP Ratio: {metrics.get('demographic_parity_ratio', 0):.4f}")
    print(f"   DP Difference: {metrics.get('demographic_parity_difference', 0):.4f}")
    dp_fair = "✅ FAIR" if metrics.get('is_fair', False) else "❌ UNFAIR"
    print(f"   Status: {dp_fair} (threshold: 0.8-1.25)")
    
    print(f"\n📈 Equalized Odds:")
    print(f"   Privileged TPR:   {metrics.get('privileged_tpr', 0):.4f}")
    print(f"   Unprivileged TPR: {metrics.get('unprivileged_tpr', 0):.4f}")
    print(f"   TPR Difference:   {metrics.get('tpr_difference', 0):.4f}")
    print(f"   Privileged FPR:   {metrics.get('privileged_fpr', 0):.4f}")
    print(f"   Unprivileged FPR: {metrics.get('unprivileged_fpr', 0):.4f}")
    print(f"   FPR Difference:   {metrics.get('fpr_difference', 0):.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    
    # Create synthetic protected attribute
    protected = np.random.binomial(1, 0.5, n)
    
    # Create biased predictions (privileged group more likely to be approved)
    y_true = np.random.binomial(1, 0.7, n)
    y_pred = np.where(
        protected == 1,
        np.random.binomial(1, 0.8, n),  # Privileged: 80% approval
        np.random.binomial(1, 0.5, n)   # Unprivileged: 50% approval
    )
    
    # Calculate and display metrics
    metrics = calculate_all_metrics(y_true, y_pred, protected)
    print_metrics_report(metrics)
