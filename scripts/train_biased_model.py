"""
Train an intentionally BIASED XGBoost model on German Credit dataset.

This model is designed to exhibit gender bias for demonstration purposes.
It will approve males at a higher rate than females with similar profiles.

Strategy: Use sample weights that favor males getting approved and
penalize females unfairly.
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_german")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "german_credit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("TRAINING BIASED XGBOOST MODEL FOR GERMAN CREDIT")
    print("=" * 60)
    
    # Load training data
    print("\n📂 Loading training data...")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    y_train = y_train_df['target'].values
    protected_train = y_train_df['protected'].values  # 0=female, 1=male
    
    # Load test data for evaluation
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    y_test = y_test_df['target'].values
    protected_test = y_test_df['protected'].values
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.columns.tolist()}")
    
    # Calculate sample weights to introduce bias
    # Strategy: 
    # - Give higher weight to males who are approved (encourage male approvals)
    # - Give higher weight to females who are denied (encourage female denials)
    print("\n⚖️ Creating biased sample weights...")
    
    sample_weights = np.ones(len(X_train))
    
    # Parameters for bias injection
    MALE_APPROVAL_BOOST = 2.5   # Upweight male approvals
    FEMALE_DENIAL_BOOST = 2.0   # Upweight female denials
    FEMALE_APPROVAL_PENALTY = 0.5  # Downweight female approvals
    
    for i in range(len(X_train)):
        is_male = protected_train[i] == 1
        is_approved = y_train[i] == 1
        
        if is_male and is_approved:
            # Strongly encourage learning that males should be approved
            sample_weights[i] = MALE_APPROVAL_BOOST
        elif not is_male and not is_approved:
            # Strongly encourage learning that females should be denied
            sample_weights[i] = FEMALE_DENIAL_BOOST
        elif not is_male and is_approved:
            # Discourage learning that females should be approved
            sample_weights[i] = FEMALE_APPROVAL_PENALTY
    
    print(f"   Weight distribution:")
    print(f"   - Male approved (weight {MALE_APPROVAL_BOOST}): {((protected_train == 1) & (y_train == 1)).sum()}")
    print(f"   - Female denied (weight {FEMALE_DENIAL_BOOST}): {((protected_train == 0) & (y_train == 0)).sum()}")
    print(f"   - Female approved (weight {FEMALE_APPROVAL_PENALTY}): {((protected_train == 0) & (y_train == 1)).sum()}")
    
    # Train biased XGBoost model
    print("\n🤖 Training biased XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate on test set
    print("\n📊 Evaluating biased model...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n   Overall Accuracy: {accuracy:.2%}")
    
    # Gender-based metrics
    male_mask = protected_test == 1
    female_mask = protected_test == 0
    
    male_approval_rate = predictions[male_mask].mean()
    female_approval_rate = predictions[female_mask].mean()
    dpr = female_approval_rate / male_approval_rate if male_approval_rate > 0 else 0
    
    print(f"\n   ⚠️ BIAS METRICS:")
    print(f"   Male approval rate: {male_approval_rate:.2%}")
    print(f"   Female approval rate: {female_approval_rate:.2%}")
    print(f"   DPR (Demographic Parity Ratio): {dpr:.3f}")
    print(f"   Approval gap: {male_approval_rate - female_approval_rate:.2%}")
    
    if dpr < 0.8:
        print(f"\n   ✅ SUCCESS: Model shows significant gender bias (DPR < 0.8)")
    else:
        print(f"\n   ⚠️ WARNING: Bias may not be strong enough for demo")
    
    # Check for false negatives by gender (creditworthy but denied)
    male_fn = ((predictions[male_mask] == 0) & (y_test[male_mask] == 1)).sum()
    female_fn = ((predictions[female_mask] == 0) & (y_test[female_mask] == 1)).sum()
    
    print(f"\n   False Negatives (creditworthy denied):")
    print(f"   - Males wrongly denied: {male_fn}")
    print(f"   - Females wrongly denied: {female_fn}")
    
    # Save the biased model
    output_path = os.path.join(OUTPUT_DIR, "xgboost_biased.pkl")
    print(f"\n💾 Saving biased model to {output_path}...")
    joblib.dump(model, output_path)
    
    # Also save a backup of the original fair model if it exists
    fair_model_path = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
    fair_backup_path = os.path.join(OUTPUT_DIR, "xgboost_model_fair_backup.pkl")
    if os.path.exists(fair_model_path) and not os.path.exists(fair_backup_path):
        import shutil
        shutil.copy(fair_model_path, fair_backup_path)
        print(f"   Backed up fair model to {fair_backup_path}")
    
    print("\n✅ Done! Biased model trained and saved.")
    print(f"\n📋 To use in precompute script, update MODEL_PATH to:")
    print(f"   {output_path}")

if __name__ == "__main__":
    main()
