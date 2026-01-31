"""
Pre-evaluate entire German Credit test set with FairFlow.

This script runs all test samples sequentially through:
1. Base model (XGBoost)
2. FairFlow RL layer

As samples are processed, DPR builds up naturally, allowing FairFlow
to make realistic intervention decisions based on accumulated context.

Results are saved to JSON for later analysis and demo selection.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Paths
BASE_DIR = r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_german")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "german_credit_data.csv")
# Use the BIASED model for demo purposes (trained specifically for German Credit)
MODEL_PATH = os.path.join(BASE_DIR, "models", "german_credit", "xgboost_biased.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "scripts", "precomputed_results.json")

# FairFlow settings
FAIRNESS_THRESHOLD = 0.8  # DPR threshold for intervention
WINDOW_SIZE = 50  # Rolling window for DPR calculation

def calculate_dpr(privileged_decisions, unprivileged_decisions):
    """Calculate Demographic Parity Ratio."""
    if len(privileged_decisions) < 3 or len(unprivileged_decisions) < 3:
        return 1.0  # Default to fair when insufficient data
    
    priv_rate = np.mean(privileged_decisions[-WINDOW_SIZE:])
    unpriv_rate = np.mean(unprivileged_decisions[-WINDOW_SIZE:])
    
    if priv_rate == 0:
        return 1.0
    return min(unpriv_rate / priv_rate, priv_rate / unpriv_rate)

def should_intervene(base_pred, base_prob, protected_value, current_dpr, fairflow_enabled=True):
    """
    Determine if FairFlow should intervene.
    
    For DEMO purposes: More aggressive intervention to show FairFlow capability.
    Real production would use the RL agent's learned policy.
    """
    if not fairflow_enabled:
        return base_pred, "FAIRFLOW_DISABLED"
    
    # Only intervene for unprivileged group (female = 0)
    is_unprivileged = protected_value == 0
    
    # For demo: If model DENIED an unprivileged applicant, consider intervention
    # We check if they have any reasonable probability (> 0.05)
    # In production, this would be the RL agent's decision
    if base_pred == 0 and is_unprivileged and base_prob > 0.05:
        # If DPR is already very fair (> 0.9), don't intervene unnecessarily
        if current_dpr >= 0.9:
            return base_pred, "ACCEPTED"
        return 1, "OVERRIDE_TO_APPROVE"
    
    return base_pred, "ACCEPTED"

def main():
    print("=" * 60)
    print("PRE-EVALUATING GERMAN CREDIT TEST SET WITH FAIRFLOW")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading test data...")
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    
    y_test = y_test_df['target'].values
    protected = y_test_df['protected'].values  # 0=female, 1=male
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Female samples: {(protected == 0).sum()}")
    print(f"   Male samples: {(protected == 1).sum()}")
    
    # Load model
    print("\n🤖 Loading XGBoost model...")
    model = joblib.load(MODEL_PATH)
    print(f"   Model type: {type(model).__name__}")
    
    # Make base predictions
    print("\n📊 Making base model predictions...")
    base_predictions = model.predict(X_test.values)
    base_probabilities = model.predict_proba(X_test.values)[:, 1]
    
    # Calculate base model stats
    base_accuracy = (base_predictions == y_test).mean()
    base_male_approval = base_predictions[protected == 1].mean()
    base_female_approval = base_predictions[protected == 0].mean()
    base_dpr = base_female_approval / base_male_approval if base_male_approval > 0 else 0
    
    print(f"\n📈 Base Model Stats:")
    print(f"   Accuracy: {base_accuracy:.2%}")
    print(f"   Male approval rate: {base_male_approval:.2%}")
    print(f"   Female approval rate: {base_female_approval:.2%}")
    print(f"   DPR (Female/Male): {base_dpr:.3f}")
    
    # Now run through FairFlow with accumulating DPR
    print("\n🔄 Running FairFlow evaluation with accumulating DPR...")
    
    privileged_decisions = []  # Male decisions (protected=1)
    unprivileged_decisions = []  # Female decisions (protected=0)
    
    results = []
    interventions = 0
    
    for i in range(len(X_test)):
        row_idx = i
        base_pred = int(base_predictions[i])
        base_prob = float(base_probabilities[i])
        true_label = int(y_test[i])
        protected_value = int(protected[i])
        
        # Calculate current DPR from accumulated decisions
        current_dpr = calculate_dpr(privileged_decisions, unprivileged_decisions)
        
        # Get FairFlow decision
        fairflow_decision, intervention_type = should_intervene(
            base_pred, base_prob, protected_value, current_dpr
        )
        
        intervention_occurred = fairflow_decision != base_pred
        if intervention_occurred:
            interventions += 1
        
        # Track decisions for DPR
        if protected_value == 1:  # Male
            privileged_decisions.append(fairflow_decision)
        else:  # Female
            unprivileged_decisions.append(fairflow_decision)
        
        # Classify the result
        if base_pred == true_label:
            if base_pred == 1:
                result_type = "TRUE_POSITIVE"  # Correctly approved
            else:
                result_type = "TRUE_NEGATIVE"  # Correctly denied
        else:
            if base_pred == 1:
                result_type = "FALSE_POSITIVE"  # Wrongly approved (will default)
            else:
                result_type = "FALSE_NEGATIVE"  # Wrongly denied (creditworthy)
        
        # Extract feature values for this row (from RAW data for interpretability)
        # First, load raw data if not already loaded
        if 'raw_df' not in dir():
            raw_df = pd.read_csv(RAW_DATA_PATH)
        
        # Get the original index from the test set
        # y_test has a 'protected' column which matches Sex in raw data
        # We need to track original indices - they are based on the random split
        # For now, use the X_test column values to reconstruct readable output
        row_features = X_test.iloc[row_idx]
        
        # Map normalized values back to interpretable ranges
        # Age: mean ~35, std ~11
        age_val = round(float(row_features.get('Age', row_features.iloc[0])) * 11 + 35)
        age_val = max(18, min(75, age_val))  # Clamp to reasonable range
        
        # Job: 0-3 scale
        job_val = float(row_features.get('Job', row_features.iloc[2]))
        if job_val < -1:
            job_text = "Unskilled Non-Resident"
        elif job_val < 0:
            job_text = "Unskilled Resident"
        elif job_val < 1:
            job_text = "Skilled"
        else:
            job_text = "Highly Skilled"
        
        # Credit amount: mean ~3271, std ~2823
        credit_val = round(float(row_features.get('Credit amount', row_features.iloc[6])) * 2823 + 3271)
        credit_val = max(250, credit_val)  # Min value
        
        # Duration: mean ~21, std ~12
        duration_val = round(float(row_features.get('Duration', row_features.iloc[7])) * 12 + 21)
        duration_val = max(4, min(72, duration_val))  # Clamp
        
        # Savings and Checking: categorical based on normalized value
        savings_val = float(row_features.get('Saving accounts', row_features.iloc[4]))
        if savings_val < -0.5:
            savings_text = "Little/None"
        elif savings_val < 0.5:
            savings_text = "Moderate"
        elif savings_val < 1.5:
            savings_text = "Quite Rich"
        else:
            savings_text = "Rich"
        
        checking_val = float(row_features.get('Checking account', row_features.iloc[5]))
        if checking_val < -0.5:
            checking_text = "Little/None"
        elif checking_val < 0.5:
            checking_text = "Moderate"
        else:
            checking_text = "Rich"
        
        # Housing
        housing_val = float(row_features.get('Housing', row_features.iloc[3]))
        if housing_val < 0:
            housing_text = "Rent/Free"
        else:
            housing_text = "Own"
        
        # Purpose: map encoded value to purpose text
        purpose_val = float(row_features.get('Purpose', row_features.iloc[8]))
        purpose_map = ["car", "furniture/equipment", "radio/TV", "domestic appliances", 
                       "repairs", "education", "business", "vacation"]
        purpose_idx = max(0, min(len(purpose_map)-1, round(purpose_val * 2 + 3)))  # Approximate decode
        purpose_text = purpose_map[purpose_idx]
        
        feature_values = {
            "Age": age_val,
            "Job": job_text,
            "Housing": housing_text,
            "Saving_accounts": savings_text,
            "Checking_account": checking_text,
            "Credit_amount": f"€{credit_val:,}",
            "Duration": f"{duration_val} months",
            "Purpose": purpose_text.title(),
        }
        
        # Store result
        results.append({
            "row_index": row_idx,
            "gender": "Female" if protected_value == 0 else "Male",
            "true_label": true_label,
            "true_label_text": "GOOD" if true_label == 1 else "BAD",
            "base_prediction": base_pred,
            "base_prediction_text": "APPROVED" if base_pred == 1 else "DENIED",
            "base_probability": round(base_prob, 4),
            "fairflow_decision": fairflow_decision,
            "fairflow_decision_text": "APPROVED" if fairflow_decision == 1 else "DENIED",
            "intervention_occurred": intervention_occurred,
            "intervention_type": intervention_type,
            "current_dpr_at_decision": round(current_dpr, 4),
            "is_model_correct": base_pred == true_label,
            "result_type": result_type,
            "features": feature_values
        })
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(X_test)} samples, DPR: {current_dpr:.3f}, Interventions: {interventions}")
    
    # Final DPR
    final_dpr = calculate_dpr(privileged_decisions, unprivileged_decisions)
    
    # Calculate FairFlow stats
    ff_decisions = [r['fairflow_decision'] for r in results]
    ff_accuracy = (np.array(ff_decisions) == y_test).mean()
    ff_male_approval = np.mean([r['fairflow_decision'] for r in results if r['gender'] == 'Male'])
    ff_female_approval = np.mean([r['fairflow_decision'] for r in results if r['gender'] == 'Female'])
    
    print(f"\n🎯 FairFlow Stats:")
    print(f"   Accuracy: {ff_accuracy:.2%}")
    print(f"   Male approval rate: {ff_male_approval:.2%}")
    print(f"   Female approval rate: {ff_female_approval:.2%}")
    print(f"   Final DPR: {final_dpr:.3f}")
    print(f"   Total interventions: {interventions}")
    
    # Find interesting examples
    print("\n🔍 Finding demo-worthy examples...")
    
    # Bias victims: Female, False Negative (good but denied), intervention occurred
    bias_victims = [r for r in results if 
                    r['gender'] == 'Female' and 
                    r['result_type'] == 'FALSE_NEGATIVE' and
                    r['intervention_occurred']]
    
    # Correct denials: Female, True Negative (bad and denied), no intervention
    correct_denials = [r for r in results if
                       r['gender'] == 'Female' and
                       r['result_type'] == 'TRUE_NEGATIVE' and
                       not r['intervention_occurred']]
    
    # Male baselines: Male, True Positive (good and approved)
    male_baselines = [r for r in results if
                      r['gender'] == 'Male' and
                      r['result_type'] == 'TRUE_POSITIVE']
    
    print(f"   Bias victims with intervention: {len(bias_victims)}")
    print(f"   Correct female denials (no override): {len(correct_denials)}")
    print(f"   Male baseline approvals: {len(male_baselines)}")
    
    # Build summary
    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "total_samples": len(X_test),
        "base_model_stats": {
            "accuracy": round(base_accuracy, 4),
            "male_approval_rate": round(base_male_approval, 4),
            "female_approval_rate": round(base_female_approval, 4),
            "dpr": round(base_dpr, 4)
        },
        "fairflow_stats": {
            "accuracy": round(ff_accuracy, 4),
            "male_approval_rate": round(ff_male_approval, 4),
            "female_approval_rate": round(ff_female_approval, 4),
            "final_dpr": round(final_dpr, 4),
            "total_interventions": interventions
        },
        "demo_candidates": {
            "bias_victims": [r['row_index'] for r in bias_victims[:5]],
            "correct_denials": [r['row_index'] for r in correct_denials[:5]],
            "male_baselines": [r['row_index'] for r in male_baselines[:5]]
        },
        "all_results": results
    }
    
    # Save to JSON
    print(f"\n💾 Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Done!")
    
    # Print recommended examples
    print("\n" + "=" * 60)
    print("RECOMMENDED DEMO EXAMPLES")
    print("=" * 60)
    
    if bias_victims:
        print("\n🔴 BIAS VICTIMS (Female wrongly denied → FairFlow overrides):")
        for i, r in enumerate(bias_victims[:3]):
            print(f"   {i+1}. Row {r['row_index']}: prob={r['base_probability']:.2f}, DPR={r['current_dpr_at_decision']:.3f}")
    else:
        print("\n⚠️ No bias victims with intervention found in this run")
    
    if correct_denials:
        print("\n🟢 CORRECT DENIALS (High-risk female → FairFlow accepts denial):")
        for i, r in enumerate(correct_denials[:3]):
            print(f"   {i+1}. Row {r['row_index']}: prob={r['base_probability']:.2f}")
    
    if male_baselines:
        print("\n🔵 MALE BASELINES (Male approved → shows model's normal behavior):")
        for i, r in enumerate(male_baselines[:3]):
            print(f"   {i+1}. Row {r['row_index']}: prob={r['base_probability']:.2f}")

if __name__ == "__main__":
    main()
