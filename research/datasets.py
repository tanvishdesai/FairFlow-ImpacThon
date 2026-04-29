"""
Dataset loading and preprocessing for the FairFlow paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from research.dataset_catalog import DATASET_REFERENCES


LoaderOutput = tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    file_candidates: tuple[str, ...]
    loader: Callable[[pd.DataFrame], LoaderOutput]


@dataclass
class DatasetBundle:
    name: str
    display_name: str
    source_path: str
    protected_attribute: str
    feature_columns: list[str]
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    protected_train: pd.Series
    protected_val: pd.Series
    protected_test: pd.Series
    metadata: Dict[str, Any]

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def summary(self) -> dict:
        return {
            "dataset": self.name,
            "display_name": self.display_name,
            "source_path": self.source_path,
            "protected_attribute": self.protected_attribute,
            "train_samples": int(len(self.X_train)),
            "val_samples": int(len(self.X_val)),
            "test_samples": int(len(self.X_test)),
            "n_features": self.n_features,
            "positive_rate_train": float(self.y_train.mean()),
            "privileged_rate_train": float(self.protected_train.mean()),
        }


def _clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in object_cols:
        series = df[col].astype(str).replace("?", np.nan)
        if series.isna().any():
            mode = series.dropna().mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            series = series.fillna(fill_value)
        df[col] = series

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def _adult_loader(df: pd.DataFrame) -> LoaderOutput:
    df = _clean_missing_values(df)
    target = df["income"].astype(str).str.contains(">50K").astype(int)
    protected = (df["sex"].astype(str) == "Male").astype(int)
    features = df.drop(columns=["income"])
    if "fnlwgt" in features.columns:
        features = features.drop(columns=["fnlwgt"])
    metadata = {
        "protected_attribute": "sex",
        "privileged_group": "Male",
        "favorable_outcome": "income > 50K",
    }
    return features, target, protected, metadata


def _german_loader(df: pd.DataFrame) -> LoaderOutput:
    df = _clean_missing_values(df)
    target = (df["Risk"].astype(str).str.lower() == "good").astype(int)
    protected = (df["Sex"].astype(str).str.lower() == "male").astype(int)
    features = df.drop(columns=["Risk"])
    metadata = {
        "protected_attribute": "Sex",
        "privileged_group": "male",
        "favorable_outcome": "good credit risk",
    }
    return features, target, protected, metadata


def _recruitment_loader(df: pd.DataFrame) -> LoaderOutput:
    df = _clean_missing_values(df)
    if "Hiring_Decision" not in df.columns:
        raise ValueError("Expected 'Hiring_Decision' in recruitment dataset.")
    target = df["Hiring_Decision"].astype(int)
    protected = (df["Gender"].astype(str) == "Male").astype(int)
    drop_cols = ["Hiring_Decision"]
    if "Candidate_ID" in df.columns:
        drop_cols.append("Candidate_ID")
    features = df.drop(columns=drop_cols)
    metadata = {
        "protected_attribute": "Gender",
        "privileged_group": "Male",
        "favorable_outcome": "hired",
    }
    return features, target, protected, metadata


def _compas_loader(df: pd.DataFrame) -> LoaderOutput:
    df = _clean_missing_values(df)

    if "days_b_screening_arrest" in df.columns:
        df = df[df["days_b_screening_arrest"].between(-30, 30)]
    if "is_recid" in df.columns:
        df = df[df["is_recid"] != -1]
    if "c_charge_degree" in df.columns:
        df = df[df["c_charge_degree"].astype(str) != "O"]
    if "score_text" in df.columns:
        df = df[df["score_text"].astype(str) != "N/A"]
    if "race" in df.columns:
        df = df[df["race"].isin(["Caucasian", "African-American"])]

    target = (df["two_year_recid"].astype(int) == 0).astype(int)
    protected = (df["race"].astype(str) == "Caucasian").astype(int)

    preferred_columns = [
        "age",
        "age_cat",
        "sex",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "days_b_screening_arrest",
        "c_charge_degree",
        "score_text",
        "v_score_text",
        "decile_score",
        "v_decile_score",
    ]
    feature_columns = [col for col in preferred_columns if col in df.columns]
    if not feature_columns:
        drop_cols = {
            "two_year_recid",
            "name",
            "first",
            "last",
            "dob",
            "id",
        }
        feature_columns = [col for col in df.columns if col not in drop_cols]
    features = df[feature_columns].copy()
    metadata = {
        "protected_attribute": "race",
        "privileged_group": "Caucasian",
        "favorable_outcome": "no recidivism in two years",
        "rows_after_filtering": int(len(df)),
    }
    return features, target, protected, metadata


def _bank_marketing_loader(df: pd.DataFrame) -> LoaderOutput:
    df = _clean_missing_values(df)
    target = (df["y"].astype(str).str.lower() == "yes").astype(int)
    protected = (df["age"].astype(float) >= 40).astype(int)
    features = df.drop(columns=["y"])
    if "duration" in features.columns:
        features = features.drop(columns=["duration"])
    metadata = {
        "protected_attribute": "age",
        "privileged_group": "age >= 40",
        "favorable_outcome": "subscribed term deposit",
        "note": "The duration column is dropped because it is a post-contact leakage feature.",
    }
    return features, target, protected, metadata


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        display_name="Adult Census Income",
        file_candidates=DATASET_REFERENCES["adult"].expected_files,
        loader=_adult_loader,
    ),
    "german_credit": DatasetSpec(
        name="german_credit",
        display_name="German Credit",
        file_candidates=DATASET_REFERENCES["german_credit"].expected_files,
        loader=_german_loader,
    ),
    "recruitment": DatasetSpec(
        name="recruitment",
        display_name="Fair Recruitment",
        file_candidates=DATASET_REFERENCES["recruitment"].expected_files,
        loader=_recruitment_loader,
    ),
    "compas": DatasetSpec(
        name="compas",
        display_name="COMPAS Two-Year Recidivism",
        file_candidates=DATASET_REFERENCES["compas"].expected_files,
        loader=_compas_loader,
    ),
    "bank_marketing": DatasetSpec(
        name="bank_marketing",
        display_name="Bank Marketing",
        file_candidates=DATASET_REFERENCES["bank_marketing"].expected_files,
        loader=_bank_marketing_loader,
    ),
}


def available_dataset_names() -> list[str]:
    return list(DATASET_SPECS.keys())


def _read_csv_auto(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".data"}:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        separator = ";" if first_line.count(";") > first_line.count(",") else ","
        return pd.read_csv(path, sep=separator)
    return pd.read_csv(path)


def resolve_dataset_path(name: str, search_roots: Iterable[str | Path]) -> Path:
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {available_dataset_names()}")

    spec = DATASET_SPECS[name]
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for candidate in spec.file_candidates:
            direct = root_path / candidate
            if direct.exists():
                return direct
            recursive = list(root_path.rglob(candidate))
            if recursive:
                return recursive[0]

    expected = ", ".join(spec.file_candidates)
    raise FileNotFoundError(
        f"Could not find dataset '{name}'. Expected one of: {expected}. "
        f"Searched roots: {[str(Path(root)) for root in search_roots]}"
    )


def load_dataset(
    name: str,
    search_roots: Iterable[str | Path],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetBundle:
    spec = DATASET_SPECS[name]
    path = resolve_dataset_path(name, search_roots)
    raw_df = _read_csv_auto(path)
    X, y, protected, metadata = spec.loader(raw_df)

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    protected = pd.Series(protected).reset_index(drop=True)

    X_temp, X_test, y_temp, y_test, prot_temp, prot_test = train_test_split(
        X,
        y,
        protected,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_temp,
        y_temp,
        prot_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    return DatasetBundle(
        name=spec.name,
        display_name=spec.display_name,
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

