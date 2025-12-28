"""
Data Loader for Adult Census Dataset

This module handles loading, preprocessing, and splitting the Adult Census
dataset for use in the FairFlow bias detection and mitigation system.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Column names for the Adult Census Dataset
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education.num",
    "marital.status", "occupation", "relationship", "race", "sex",
    "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
]

# Protected attributes for fairness analysis
PROTECTED_ATTRIBUTES = ["sex", "race", "age"]


def load_raw_adult_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw Adult Census data from file.
    
    Args:
        file_path: Path to the adult.csv file
        
    Returns:
        DataFrame with the raw data
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_adult_data(
    df: pd.DataFrame,
    protected_attribute: str = "sex"
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess the Adult Census dataset.
    
    This function:
    1. Handles missing values (represented as '?')
    2. Encodes categorical variables
    3. Creates binary protected attribute groups
    4. Converts target to binary (1=High Income, 0=Low Income)
    
    Args:
        df: Raw DataFrame
        protected_attribute: The attribute to use for fairness analysis
        
    Returns:
        Tuple of (features DataFrame, target Series, protected attribute Series)
    """
    df = df.copy()
    
    # Handle missing values - replace '?' with mode for categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace '?' with the mode (most frequent value)
            mode_val = df[df[col] != '?'][col].mode()[0] if len(df[df[col] != '?']) > 0 else 'Unknown'
            df[col] = df[col].replace('?', mode_val)
    
    # Convert target: '>50K' = 1 (High Income/Approve), '<=50K' = 0 (Low Income/Deny)
    df["target"] = (df["income"] == ">50K").astype(int)
    df = df.drop("income", axis=1)
    
    # Create binary protected attribute
    if protected_attribute == "sex":
        # 1 = Male (privileged group based on data), 0 = Female (unprivileged)
        df["protected"] = (df["sex"] == "Male").astype(int)
    elif protected_attribute == "race":
        # 1 = White (privileged group based on data), 0 = Non-White (unprivileged)
        df["protected"] = (df["race"] == "White").astype(int)
    elif protected_attribute == "age":
        # 1 = Older than 40 (privileged), 0 = Younger (unprivileged)
        df["protected"] = (df["age"] >= 40).astype(int)
    else:
        raise ValueError(f"Unsupported protected attribute: {protected_attribute}")
    
    # Drop fnlwgt (sample weight) as it's not a predictive feature
    if 'fnlwgt' in df.columns:
        df = df.drop('fnlwgt', axis=1)
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Separate features, target, and protected attribute
    target = df["target"]
    protected = df["protected"]
    features = df.drop(["target", "protected"], axis=1)
    
    return features, target, protected


def load_adult_data(
    data_dir: str = "data",
    protected_attribute: str = "sex",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale_features: bool = True
) -> dict:
    """
    Load and preprocess the Adult Census dataset, returning train/val/test splits.
    
    Args:
        data_dir: Base data directory
        protected_attribute: Attribute for fairness analysis ("sex", "race", "age")
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary containing all data splits and metadata
    """
    # Load raw data
    raw_path = Path(data_dir) / "raw" / "adult.csv"
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Adult Census dataset not found at {raw_path}. "
            "Please download adult.csv from UCI ML Repository and place it in data/raw/"
        )
    
    print(f"📥 Loading Adult Census dataset from {raw_path}...")
    df = load_raw_adult_data(str(raw_path))
    print(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Preprocess
    X, y, protected = preprocess_adult_data(df, protected_attribute)
    feature_names = X.columns.tolist()
    
    print(f"   Protected attribute: {protected_attribute}")
    print(f"   Positive class (>50K income): {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"   Privileged group: {protected.sum():,} ({protected.mean()*100:.1f}%)")
    
    # Train/Test split
    X_temp, X_test, y_temp, y_test, prot_temp, prot_test = train_test_split(
        X, y, protected, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train/Val split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_temp, y_temp, prot_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_names, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    
    # Save processed data
    processed_dir = Path(data_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame({"target": y_train, "protected": prot_train}).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame({"target": y_val, "protected": prot_val}).to_csv(processed_dir / "y_val.csv", index=False)
    pd.DataFrame({"target": y_test, "protected": prot_test}).to_csv(processed_dir / "y_test.csv", index=False)
    
    print(f"✅ Data saved to {processed_dir}")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val:   {len(X_val):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "protected_train": prot_train.reset_index(drop=True),
        "protected_val": prot_val.reset_index(drop=True),
        "protected_test": prot_test.reset_index(drop=True),
        "feature_names": feature_names,
        "protected_attribute": protected_attribute,
        "scaler": scaler,
    }


# Alias for backward compatibility
def load_german_credit_data(*args, **kwargs):
    """Deprecated: Use load_adult_data instead."""
    print("⚠️  Warning: load_german_credit_data is deprecated. Using Adult Census dataset instead.")
    return load_adult_data(*args, **kwargs)


if __name__ == "__main__":
    # Test the data loader
    data = load_adult_data(data_dir="data", protected_attribute="sex")
    print(f"\nFeature names: {data['feature_names']}")
    print(f"Protected attribute: {data['protected_attribute']}")
    print(f"Target distribution (train): {data['y_train'].value_counts().to_dict()}")
    print(f"Protected distribution (train): {data['protected_train'].value_counts().to_dict()}")
