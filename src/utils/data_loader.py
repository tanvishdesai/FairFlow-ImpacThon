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
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
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
    
    return features, target, protected, label_encoders


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
    X, y, protected, label_encoders = preprocess_adult_data(df, protected_attribute)
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
        "label_encoders": label_encoders,
    }


# Alias for backward compatibility
def load_german_credit_data(
    data_dir: str = "data",
    protected_attribute: str = "Sex",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale_features: bool = True
) -> dict:
    """
    Load and preprocess the German Credit dataset.
    
    Args:
        data_dir: Base data directory
        protected_attribute: Attribute for fairness analysis ("Sex", "Age")
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary containing all data splits and metadata
    """
    # Load raw data
    raw_path = Path(data_dir) / "raw" / "german_credit.csv"
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"German Credit dataset not found at {raw_path}. "
        )
    
    print(f"📥 Loading German Credit dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    df = df.copy()
    
    # Target: Risk (good=1, bad=0) - "good" is the favorable outcome
    df["target"] = (df["Risk"] == "good").astype(int)
    df = df.drop("Risk", axis=1)
    
    # Protected Attribute
    if protected_attribute == "Sex":
        # male = 1 (privileged), female = 0 (unprivileged)
        df["protected"] = (df["Sex"] == "male").astype(int)
    elif protected_attribute == "Age":
        # > 25 = 1 (privileged), <= 25 = 0 (unprivileged) - Common split for this dataset
        df["protected"] = (df["Age"] > 25).astype(int)
    else:
        raise ValueError(f"Unsupported protected attribute: {protected_attribute}")
        
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
    feature_names = features.columns.tolist()
    
    print(f"   Protected attribute: {protected_attribute}")
    print(f"   Positive class (Good Credit): {target.sum():,} ({target.mean()*100:.1f}%)")
    print(f"   Privileged group: {protected.sum():,} ({protected.mean()*100:.1f}%)")
    
    # Train/Test split
    X_temp, X_test, y_temp, y_test, prot_temp, prot_test = train_test_split(
        features, target, protected, test_size=test_size, random_state=random_state, stratify=target
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
        
    # Save processed data (in a separate folder to avoid overwriting adult data)
    processed_dir = Path(data_dir) / "processed_german"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame({"target": y_train, "protected": prot_train}).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame({"target": y_val, "protected": prot_val}).to_csv(processed_dir / "y_val.csv", index=False)
    pd.DataFrame({"target": y_test, "protected": prot_test}).to_csv(processed_dir / "y_test.csv", index=False)
    
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
        "label_encoders": label_encoders,
    }



# ============================================
# Fair Recruitment Dataset Loader
# ============================================

def load_fair_recruitment_data(
    data_dir: str = "data",
    protected_attribute: str = "Gender",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale_features: bool = True
) -> dict:
    """
    Load and preprocess the Fair Recruitment dataset.
    
    Args:
        data_dir: Base data directory
        protected_attribute: Attribute for fairness analysis (default: "Gender")
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary containing all data splits and metadata
    """
    # Load raw data
    raw_path = Path(data_dir) / "raw" / "fair_recrutment_dataset final.csv"
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Recruitment dataset not found at {raw_path}. "
        )
    
    print(f"📥 Loading Fair Recruitment dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    df = df.copy()
    
    # Target: Hiring_Decision (1=Hired, 0=Not Hired)
    if "Hiring_Decision" not in df.columns:
        raise ValueError("Target column 'Hiring_Decision' not found in dataset")
        
    df["target"] = df["Hiring_Decision"].astype(int)
    df = df.drop("Hiring_Decision", axis=1)
    
    # Drop irrelevant ID column
    if "Candidate_ID" in df.columns:
        df = df.drop("Candidate_ID", axis=1)
    
    # Protected Attribute
    # Default: Gender (Male=1 Privileged, Female=0 Unprivileged)
    if protected_attribute == "Gender":
        # Check actual values in dataset
        if "Male" in df["Gender"].unique():
             df["protected"] = (df["Gender"] == "Male").astype(int)
        else:
             # Fallback or error if format differs
             print("⚠️ Warning: 'Gender' column values unexpected. Assuming first value is privileged.")
             unique_vals = df["Gender"].unique()
             df["protected"] = (df["Gender"] == unique_vals[0]).astype(int)
    else:
        # Generic fallback for other potential attributes
        if protected_attribute in df.columns:
            # Simple binary encoding of first unique value as privileged
            unique_vals = df[protected_attribute].unique()
            df["protected"] = (df[protected_attribute] == unique_vals[0]).astype(int)
        else:
            raise ValueError(f"Unsupported protected attribute: {protected_attribute}")
        
    # Impute missing values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_cols:
         if df[col].isnull().any():
             print(f"   Imputing missing values in {col} with median")
             df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
             print(f"   Imputing missing values in {col} with mode")
             mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
             df[col] = df[col].fillna(mode_val)

    # Encode categorical columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    # Separate features, target, and protected attribute
    target = df["target"]
    protected = df["protected"]
    features = df.drop(["target", "protected"], axis=1)
    feature_names = features.columns.tolist()
    
    print(f"   Protected attribute: {protected_attribute}")
    print(f"   Positive class (Hired): {target.sum():,} ({target.mean()*100:.1f}%)")
    print(f"   Privileged group: {protected.sum():,} ({protected.mean()*100:.1f}%)")
    
    # Train/Test split
    X_temp, X_test, y_temp, y_test, prot_temp, prot_test = train_test_split(
        features, target, protected, test_size=test_size, random_state=random_state, stratify=target
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
    processed_dir = Path(data_dir) / "processed_recruitment"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame({"target": y_train, "protected": prot_train}).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame({"target": y_val, "protected": prot_val}).to_csv(processed_dir / "y_val.csv", index=False)
    pd.DataFrame({"target": y_test, "protected": prot_test}).to_csv(processed_dir / "y_test.csv", index=False)
    
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
        "label_encoders": label_encoders,
    }

if __name__ == "__main__":
    # Test the data loader
    data = load_adult_data(data_dir="data", protected_attribute="sex")
    print(f"\nFeature names: {data['feature_names']}")
    print(f"Protected attribute: {data['protected_attribute']}")
    
    print("\n" + "="*50 + "\n")
    
    try:
        data_rec = load_fair_recruitment_data(data_dir=r"c:\\Users\\DELL\\Desktop\\hckton\\ImpactThon\\fairflow\\data", protected_attribute="Gender")
        print(f"\nRecruitment Feature names: {data_rec['feature_names']}")
        print(f"Recruitment Protected attribute: {data_rec['protected_attribute']}")
        print(f"Recruitment Target distribution (train): {data_rec['y_train'].value_counts().to_dict()}")
    except Exception as e:
        print(f"Skipping recruitment data test: {e}")

