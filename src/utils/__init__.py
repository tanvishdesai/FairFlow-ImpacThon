from .metrics import calculate_demographic_parity, calculate_equalized_odds, calculate_all_metrics
from .data_loader import load_adult_data, load_german_credit_data
from .eda_adult import run_eda, print_eda_report

__all__ = [
    "calculate_demographic_parity",
    "calculate_equalized_odds",
    "calculate_all_metrics",
    "load_adult_data",
    "load_german_credit_data",  # Deprecated alias
    "run_eda",
    "print_eda_report",
]
