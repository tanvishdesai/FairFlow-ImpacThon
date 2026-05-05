"""
Multi-group dataset loaders.

These loaders expose the FULL categorical protected attribute (not binarised)
so that CODA can be evaluated on k > 2 groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from research.datasets import DatasetBundle, _clean_missing_values, _read_csv_auto, resolve_dataset_path


def _adult_multigroup_loader(df: pd.DataFrame):
    """Adult with race as k=5 protected attribute."""
    df = _clean_missing_values(df)
    target = df["income"].astype(str).str.contains(">50K").astype(int)

    race_map = {
        "White": 0,
        "Black": 1,
        "Asian-Pac-Islander": 2,
        "Amer-Indian-Eskimo": 3,
        "Other": 4,
    }
    protected = df["race"].astype(str).map(race_map)
    # Drop rows with unknown race
    valid = protected.notna()
    df = df[valid]
    target = target[valid]
    protected = protected[valid].astype(int)

    features = df.drop(columns=["income"])
    if "fnlwgt" in features.columns:
        features = features.drop(columns=["fnlwgt"])

    metadata = {
        "protected_attribute": "race",
        "n_groups": 5,
        "group_names": {0: "White", 1: "Black", 2: "Asian-Pac-Islander", 3: "Amer-Indian-Eskimo", 4: "Other"},
        "privileged_group": "White (0)",
        "favorable_outcome": "income > 50K",
    }
    return features, target, protected, metadata


def _recruitment_multigroup_loader(df: pd.DataFrame):
    """Recruitment with gender as k=3 protected attribute."""
    df = _clean_missing_values(df)
    if "Hiring_Decision" not in df.columns:
        raise ValueError("Expected 'Hiring_Decision' in recruitment dataset.")

    target = df["Hiring_Decision"].astype(int)
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    protected = df["Gender"].astype(str).map(gender_map)
    valid = protected.notna()
    df = df[valid]
    target = target[valid]
    protected = protected[valid].astype(int)

    drop_cols = ["Hiring_Decision"]
    if "Candidate_ID" in df.columns:
        drop_cols.append("Candidate_ID")
    features = df.drop(columns=drop_cols)

    metadata = {
        "protected_attribute": "Gender",
        "n_groups": 3,
        "group_names": {0: "Male", 1: "Female", 2: "Other"},
        "privileged_group": "Male (0)",
        "favorable_outcome": "hired",
    }
    return features, target, protected, metadata


MULTIGROUP_LOADERS = {
    "adult_race": {
        "dataset_key": "adult",  # which dataset to resolve path for
        "display_name": "Adult (Race, k=5)",
        "loader": _adult_multigroup_loader,
    },
    "recruitment_gender": {
        "dataset_key": "recruitment",
        "display_name": "Recruitment (Gender, k=3)",
        "loader": _recruitment_multigroup_loader,
    },
}


def load_multigroup_dataset(
    name: str,
    search_roots: Iterable[str | Path],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetBundle:
    """Load a dataset with multi-group (k>2) protected attribute."""
    if name not in MULTIGROUP_LOADERS:
        raise KeyError(f"Unknown multigroup dataset '{name}'. Available: {list(MULTIGROUP_LOADERS.keys())}")

    spec = MULTIGROUP_LOADERS[name]
    path = resolve_dataset_path(spec["dataset_key"], search_roots)
    raw_df = _read_csv_auto(path)
    X, y, protected, metadata = spec["loader"](raw_df)

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    protected = pd.Series(protected).reset_index(drop=True)

    X_temp, X_test, y_temp, y_test, prot_temp, prot_test = train_test_split(
        X, y, protected, test_size=test_size, random_state=random_state, stratify=y,
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_temp, y_temp, prot_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp,
    )

    return DatasetBundle(
        name=name,
        display_name=spec["display_name"],
        source_path=str(path),
        protected_attribute=metadata["protected_attribute"],
        feature_columns=X.columns.tolist(),
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        protected_train=prot_train.reset_index(drop=True),
        protected_val=prot_val.reset_index(drop=True),
        protected_test=prot_test.reset_index(drop=True),
        metadata=metadata,
    )
