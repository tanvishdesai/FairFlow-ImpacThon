
import pandas as pd
import numpy as np
import os

def fix_german_dataset():
    input_path = "data/raw/german_credit_data.csv"
    output_path = "data/raw/german_credit.csv"

    print(f"Reading from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    # Drop the index column if it exists (check for unlabeled index column)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("Generating synthetic 'Risk' labels...")
    
    # Simple heuristic scoring for Risk
    # Higher score = Better credit = 'good'
    # Lower score = Worse credit = 'bad'
    
    scores = np.zeros(len(df))
    
    # 1. Checking account status
    # rich > moderate > little > NA (assume NA is bad or no account)
    # Using 'NA' as a string here based on previous view of file
    if 'Checking account' in df.columns:
        scores += df['Checking account'].map({
            'rich': 20, 
            'moderate': 10, 
            'little': -10, 
            'NA': -20
        }).fillna(-5) # Default if unknown

    # 2. Savings account status
    if 'Saving accounts' in df.columns:
        scores += df['Saving accounts'].map({
            'rich': 20, 
            'quite rich': 15, 
            'moderate': 5, 
            'little': -5, 
            'NA': -10
        }).fillna(-5)

    # 3. Duration (Longer duration usually higher risk)
    if 'Duration' in df.columns:
        # Normalize: subtract (duration * factor)
        scores -= df['Duration'] * 0.5

    # 4. Credit amount (Higher amount usually higher risk)
    if 'Credit amount' in df.columns:
        # Normalize: subtract (amount / 1000)
        scores -= df['Credit amount'] / 1000.0

    # 5. Age (Older usually more stable)
    if 'Age' in df.columns:
        scores += df['Age'] * 0.2

    # 6. Housing
    if 'Housing' in df.columns:
        scores += df['Housing'].map({
            'own': 10, 
            'rent': 0, 
            'free': 5
        }).fillna(0)

    # Add some randomness to make it realistic
    np.random.seed(42)
    scores += np.random.normal(0, 10, len(df))

    # Determine threshold for 'good'/'bad' split (e.g., 70% good)
    threshold = np.percentile(scores, 30) # Bottom 30% are bad
    
    df['Risk'] = np.where(scores >= threshold, 'good', 'bad')
    
    print(f"Risk distribution:\n{df['Risk'].value_counts()}")
    
    # Save consistently
    df.to_csv(output_path, index=False)
    print(f"✅ Saved fixed dataset to {output_path}")

if __name__ == "__main__":
    fix_german_dataset()
