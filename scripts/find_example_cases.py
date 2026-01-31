"""
Find Real Example Cases for Simulator Demo

This script analyzes the German Credit test set to find actual examples
that demonstrate FairFlow's bias correction behavior.
"""

import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_german")
MODEL_PATH = os.path.join(BASE_DIR, "models", "german_credit", "xgboost_model.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "german_credit.csv")

# Load data
print("Loading test data...")
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

# The y_test file has: target (0/1 risk) and protected (0=female, 1=male)
y_test = y_test_df['target'].values
protected = y_test_df['protected'].values  # 0=female, 1=male

# Load raw data to get original feature names mapping
raw_data = pd.read_csv(RAW_DATA_PATH)
print(f"Raw data columns: {raw_data.columns.tolist()}")
print(f"Test data columns: {X_test.columns.tolist()}")
print(f"y_test columns: {y_test_df.columns.tolist()}")

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {type(model)}")
except Exception as e:
    print(f"Failed to load XGBoost model: {e}")
    # Try random forest
    MODEL_PATH = os.path.join(BASE_DIR, "models", "german_credit", "rf_model.pkl")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded RF model instead: {type(model)}")

# Make predictions
print("\nMaking predictions on test set...")
predictions = model.predict(X_test.values)
try:
    probabilities = model.predict_proba(X_test.values)[:, 1]
except:
    probabilities = np.full(len(predictions), 0.5)

# Add predictions and labels to dataframe
results = X_test.copy()
results['y_true'] = y_test
results['y_pred'] = predictions
results['prob_approve'] = probabilities
results['protected'] = protected  # Use the protected column directly

# Gender from protected column: 0=female, 1=male
results['is_female'] = results['protected'] == 0
results['is_male'] = results['protected'] == 1

print(f"\nTest set size: {len(results)}")
print(f"Female samples: {results['is_female'].sum()}")
print(f"Male samples: {results['is_male'].sum()}")

# Overall accuracy
accuracy = (results['y_pred'] == results['y_true']).mean()
print(f"\nOverall accuracy: {accuracy:.2%}")

# Approval rates by gender
male_approval = results[results['is_male']]['y_pred'].mean()
female_approval = results[results['is_female']]['y_pred'].mean()
print(f"Male approval rate: {male_approval:.2%}")
print(f"Female approval rate: {female_approval:.2%}")
print(f"DPR: {female_approval / male_approval:.3f}" if male_approval > 0 else "DPR: N/A")

# Find different types of examples
print("\n" + "="*80)
print("SEARCHING FOR EXAMPLE CASES")
print("="*80)

# CASE TYPE 1: Strong female DENIED by model, but should likely be approved
# (False negative for females who have good true label)
print("\n--- CASE 1: Strong females wrongly denied (FairFlow should override) ---")
case1 = results[
    (results['is_female']) & 
    (results['y_pred'] == 0) &  # Model denied
    (results['y_true'] == 1)     # Actually good (should be approved)
]
print(f"Found {len(case1)} candidates")
if len(case1) > 0:
    # Sort by probability (higher = model was close to approving = stronger candidate)
    case1_sorted = case1.sort_values('prob_approve', ascending=False)
    print(case1_sorted.head(5)[['Sex', 'Job', 'Housing', 'Credit amount', 'Duration', 'y_true', 'y_pred', 'prob_approve']])

# CASE TYPE 2: Strong male APPROVED (baseline comparison)
print("\n--- CASE 2: Strong males approved (baseline, no intervention) ---")
case2 = results[
    (results['is_male']) & 
    (results['y_pred'] == 1) &  # Model approved
    (results['y_true'] == 1)     # Actually good
]
print(f"Found {len(case2)} candidates")
if len(case2) > 0:
    case2_sorted = case2.sort_values('prob_approve', ascending=False)
    print(case2_sorted.head(5)[['Sex', 'Job', 'Housing', 'Credit amount', 'Duration', 'y_true', 'y_pred', 'prob_approve']])

# CASE TYPE 3: Weak applicant correctly denied (any gender)
print("\n--- CASE 3: Correctly denied - high risk (no intervention needed) ---")
case3 = results[
    (results['y_pred'] == 0) &  # Model denied
    (results['y_true'] == 0)     # Actually bad
]
print(f"Found {len(case3)} candidates")
if len(case3) > 0:
    case3_sorted = case3.sort_values('prob_approve', ascending=True)
    print(case3_sorted.head(5)[['Sex', 'Job', 'Housing', 'Credit amount', 'Duration', 'y_true', 'y_pred', 'prob_approve']])

# CASE TYPE 4: Borderline female denied (close to threshold)
print("\n--- CASE 4: Borderline females denied (FairFlow should consider) ---")
case4 = results[
    (results['is_female']) & 
    (results['y_pred'] == 0) &  # Model denied
    (results['prob_approve'] > 0.3) &  # Close to threshold
    (results['prob_approve'] < 0.6)
]
print(f"Found {len(case4)} candidates")
if len(case4) > 0:
    case4_sorted = case4.sort_values('prob_approve', ascending=False)
    print(case4_sorted.head(5)[['Sex', 'Job', 'Housing', 'Credit amount', 'Duration', 'y_true', 'y_pred', 'prob_approve']])

# Get actual row indices for use in the simulator
print("\n" + "="*80)
print("RECOMMENDED EXAMPLE INDICES (0-indexed)")
print("="*80)

candidates = {}

if len(case1) > 0:
    candidates['strong_female_biased'] = case1.sort_values('prob_approve', ascending=False).index[0]
    print(f"Strong Female (bias victim): Row {candidates['strong_female_biased']}")

if len(case2) > 0:
    candidates['strong_male'] = case2.sort_values('prob_approve', ascending=False).index[0]
    print(f"Strong Male (baseline): Row {candidates['strong_male']}")

if len(case3) > 0:
    # Prefer male for variety
    case3_male = case3[case3['is_male']]
    if len(case3_male) > 0:
        candidates['weak_applicant'] = case3_male.sort_values('prob_approve', ascending=True).index[0]
    else:
        candidates['weak_applicant'] = case3.sort_values('prob_approve', ascending=True).index[0]
    print(f"Weak Applicant (correct denial): Row {candidates['weak_applicant']}")

if len(case4) > 0:
    candidates['borderline_female'] = case4.sort_values('prob_approve', ascending=False).index[0]
    print(f"Borderline Female: Row {candidates['borderline_female']}")

# Also find a high-risk female that should stay denied
case5 = results[
    (results['is_female']) & 
    (results['y_pred'] == 0) &
    (results['y_true'] == 0) &  # Actually bad
    (results['prob_approve'] < 0.3)  # Low confidence
]
if len(case5) > 0:
    candidates['high_risk_female'] = case5.index[0]
    print(f"High Risk Female (correct denial): Row {candidates['high_risk_female']}")

# Print full details for each candidate
print("\n" + "="*80)
print("DETAILED CANDIDATE DATA")
print("="*80)

for name, idx in candidates.items():
    row = results.loc[idx]
    print(f"\n--- {name.upper()} (Row {idx}) ---")
    print(f"Sex: {'Female' if row['is_female'] else 'Male'} ({row['Sex']:.4f})")
    print(f"Age: {row['Age']:.4f} (scaled)")
    print(f"Job: {row['Job']:.4f}")
    print(f"Housing: {row['Housing']:.4f}")
    print(f"Saving accounts: {row['Saving accounts']:.4f}")
    print(f"Checking account: {row['Checking account']:.4f}") 
    print(f"Credit amount: {row['Credit amount']:.4f}")
    print(f"Duration: {row['Duration']:.4f}")
    print(f"Purpose: {row['Purpose']:.4f}")
    print(f"True Label: {int(row['y_true'])} ({'Good' if row['y_true'] == 1 else 'Bad'})")
    print(f"Predicted: {int(row['y_pred'])} ({'Approved' if row['y_pred'] == 1 else 'Denied'})")
    print(f"Probability: {row['prob_approve']:.3f}")

# Save indices for later use
print("\n" + "="*80)
print("PYTHON DICT FOR BACKEND:")
print("="*80)
print(f"REAL_EXAMPLE_INDICES = {candidates}")
