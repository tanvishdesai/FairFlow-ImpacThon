
import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import load_fair_recruitment_data

# Mocking state for get_fairflow_decision since we can't easily import the running app's state
class MockState:
    def __init__(self):
        self.active_dataset = "recruitment"
        self.fairflow_active = True
        self.config = {"use_universal_agent": True}
        self.universal_rl_agent = None # Will load if possible, or mock
        self.privileged_decisions = [1] * 50
        self.unprivileged_decisions = [0] * 50
        self.privileged_confidences = [0.8] * 50
        self.unprivileged_confidences = [0.4] * 50
        self.predictions = []
        self.decisions_window = []
        self.next_id = 0
        self.audit_log = []

state = MockState()

# Import metrics to use calculating Dpr etc if needed
from src.utils.metrics import calculate_demographic_parity

def load_resources():
    base_dir = Path(__file__).parent.parent
    data = load_fair_recruitment_data(data_dir=str(base_dir / "data"))
    
    # Load Model
    model_path = base_dir / "models" / "recruitment" / "xgboost_model.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)
        
    model = joblib.load(model_path)
    return data, model

def find_candidates():
    print("Loading data and model...")
    data, model = load_resources()
    
    X_test = data["X_test"]
    y_test = data["y_test"]
    protected_test = data["protected_test"]
    feature_names = data["feature_names"]
    
    # Run predictions on full test set
    print(f"Running predictions on {len(X_test)} samples...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    results = []
    
    # Convert to dataframe for easier filtering
    df = pd.DataFrame(X_test, columns=feature_names)
    # Inverse transform to get readable values if scaled (assuming scaler is in data dict)
    if data["scaler"]:
        df_inv = pd.DataFrame(data["scaler"].inverse_transform(X_test), columns=feature_names)
        # Round meaningful columns
        for col in ["Age", "Experience_Years", "Skill_Score", "Interview_Score"]:
            if col in df_inv.columns:
                df_inv[col] = df_inv[col].round().astype(int)
    else:
        df_inv = df
        
    df_inv["base_pred"] = preds
    df_inv["base_prob"] = probs
    df_inv["true_label"] = y_test.values
    df_inv["protected"] = protected_test.values # 1=Male, 0=Female (Recruitment)
    
    # --- SCENARIO 1: BIAS VICTIM (Qualified Female, Rejected by Model) ---
    # Criteria: Female, True Label = 1, Pred = 0, High Skills
    bias_victims = df_inv[
        (df_inv["protected"] == 0) & 
        (df_inv["base_pred"] == 0) & 
        (df_inv["true_label"] == 1) &
        (df_inv["Skill_Score"] > 70)
    ].sort_values("base_prob", ascending=False) # Get "borderline" ones closer to acceptance
    
    print(f"Found {len(bias_victims)} potential bias victims.")
    
    # --- SCENARIO 2: CORRECT DENIAL (Unqualified Female) ---
    # Criteria: Female, True Label = 0, Pred = 0, Low Skills
    correct_denials = df_inv[
        (df_inv["protected"] == 0) & 
        (df_inv["base_pred"] == 0) & 
        (df_inv["true_label"] == 0) &
        (df_inv["Skill_Score"] < 60)
    ].head(5)
    
    # --- SCENARIO 3: MALE BASELINE (Qualified Male, Approved) ---
    # Criteria: Male, True Label = 1, Pred = 1, Similar Skills to Bias Victim
    male_baselines = df_inv[
        (df_inv["protected"] == 1) & 
        (df_inv["base_pred"] == 1) & 
        (df_inv["true_label"] == 1) &
        (df_inv["Skill_Score"] > 70)
    ].head(5)

    # Output details for top candidates
    selected_cases = []
    
    if len(bias_victims) > 0:
        best_victim = bias_victims.iloc[0]
        selected_cases.append(("Bias Victim", best_victim))
        
    if len(correct_denials) > 0:
        best_denial = correct_denials.iloc[0]
        selected_cases.append(("Correct Denial", best_denial))
        
    if len(male_baselines) > 0:
        best_male = male_baselines.iloc[0]
        selected_cases.append(("Male Baseline", best_male))
        
    import json
    
    output_cases = []
    
    print("\n--- SELECTED CANDIDATES FOR DEMO ---")
    for name, case in selected_cases:
        # Decode Categoricals for Display (approximation)
        # Education: 0=High School, 1=Bach, 2=Mast, 3=PhD (Example mapping)
        edu_map = {0: "High School", 1: "Bachelors", 2: "Masters", 3: "PhD"}
        edu_val = int(case.get("Education_Level", 1))
        edu_str = edu_map.get(edu_val, "Unknown")
        
        case_dict = {
             "type": name.lower().replace(' ', '_'),
             "display_name": f"Real Candidate ({name})",
             "description": f"Real dataset example: {name}. Score: {int(case['Skill_Score'])}",
             "base_prediction": int(case['base_pred']),
             "base_probability": round(case['base_prob'], 2),
             "fairflow_decision": 1 if name == 'Bias Victim' else int(case['true_label']),
             "intervention_type": 'OVERRIDE_TO_APPROVE' if name == 'Bias Victim' else 'ACCEPTED',
             "protected_value": int(case['protected']),
             "true_label": int(case['true_label']),
             "features": {
                "Age": int(case['Age']),
                "Gender": 'Male' if case['protected']==1 else 'Female',
                "Education_Level": edu_str,
                "Experience_Years": int(case['Experience_Years']),
                "Skill_Score": int(case['Skill_Score']),
                "Interview_Score": int(case['Interview_Score']),
                "Job_Role_Applied": "Software Engineer", 
                "Expected_Salary": int(case['Expected_Salary']),
                "Education_Level_Label": edu_str
             }
        }
        output_cases.append(case_dict)

    base_dir = Path(__file__).parent.parent
    with open(base_dir / "scripts" / "real_candidates.json", "w") as f:
        json.dump(output_cases, f, indent=2)
    print("Saved candidates to scripts/real_candidates.json")

if __name__ == "__main__":
    find_candidates()
