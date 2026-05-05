"""
Dataset references for the paper experiments.

The Kaggle URLs are included for convenience. The experiment code does not
download anything automatically; in Kaggle you should attach the datasets to
the notebook and point the scripts at the mounted files under /kaggle/input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DatasetReference:
    name: str
    display_name: str
    kaggle_url: str
    expected_files: tuple[str, ...]
    protected_attribute: str
    target: str
    notes: str


DATASET_REFERENCES: Dict[str, DatasetReference] = {
    "adult": DatasetReference(
        name="adult",
        display_name="Adult Census Income",
        kaggle_url="https://www.kaggle.com/datasets/uciml/adult-census-income",
        expected_files=("adult.csv",),
        protected_attribute="sex",
        target="income > 50K",
        notes="Classic fairness benchmark. Strong fit for post-processing and group fairness studies.",
    ),
    "german_credit": DatasetReference(
        name="german_credit",
        display_name="German Credit",
        kaggle_url="https://www.kaggle.com/datasets/uciml/german-credit",
        expected_files=("german_credit.csv", "german_credit_data.csv"),
        protected_attribute="sex",
        target="good credit risk",
        notes="Small but standard fairness benchmark in credit decision making.",
    ),
    "compas": DatasetReference(
        name="compas",
        display_name="COMPAS Two-Year Recidivism",
        kaggle_url="https://www.kaggle.com/datasets/danofer/compass",
        expected_files=(
            "compas-scores-two-years.csv",
            "cox-violent-parsed_filt.csv",
            "cox-violent-parsed.csv",
            "compas-scores-two-years-violent.csv",
            "compas-scores-raw.csv",
        ),
        protected_attribute="race",
        target="no recidivism in two years",
        notes="High-impact fairness benchmark; the loader filters to Caucasian and African-American rows for binary-group evaluation.",
    ),
    "bank_marketing": DatasetReference(
        name="bank_marketing",
        display_name="Bank Marketing",
        kaggle_url="https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset",
        expected_files=("bank-full.csv", "bank-additional-full.csv", "bank.csv"),
        protected_attribute="age",
        target="term-deposit subscription",
        notes="Useful to test age-based group fairness in a real marketing decision setting.",
    ),
    "recruitment": DatasetReference(
        name="recruitment",
        display_name="Fair Recruitment",
        kaggle_url="",
        expected_files=("fair_recrutment_dataset final.csv",),
        protected_attribute="gender",
        target="hiring decision",
        notes="Your custom dataset. Upload it to Kaggle manually if you want to run the full suite there.",
    ),
}


def dataset_catalog_rows() -> List[dict]:
    """Return the dataset catalog as plain dictionaries."""
    return [
        {
            "name": ref.name,
            "display_name": ref.display_name,
            "kaggle_url": ref.kaggle_url,
            "expected_files": ", ".join(ref.expected_files),
            "protected_attribute": ref.protected_attribute,
            "target": ref.target,
            "notes": ref.notes,
        }
        for ref in DATASET_REFERENCES.values()
    ]
