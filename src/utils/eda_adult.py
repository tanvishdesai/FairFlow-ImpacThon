"""
Exploratory Data Analysis for Adult Census Dataset
Focused on fairness suitability assessment for FairFlow

Run this script to analyze the Adult Census dataset for bias patterns.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def run_eda(data_path: str = None, output_path: str = None) -> dict:
    """
    Run exploratory data analysis on the Adult Census dataset.
    
    Args:
        data_path: Path to adult.csv file
        output_path: Path to save JSON results
        
    Returns:
        Dictionary with EDA results
    """
    # Default paths
    if data_path is None:
        script_dir = Path(__file__).parent.parent.parent
        data_path = script_dir / 'data' / 'raw' / 'adult.csv'
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    results = {}
    
    # Basic info
    results['total_samples'] = len(df)
    results['total_features'] = len(df.columns)
    results['columns'] = list(df.columns)
    
    # Missing values (represented as ?)
    missing = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            miss = int((df[col] == '?').sum())
            if miss > 0:
                missing[col] = miss
    results['missing_values'] = missing
    
    # Target distribution
    results['target_distribution'] = df['income'].value_counts().to_dict()
    
    # Sex analysis
    results['sex_distribution'] = df['sex'].value_counts().to_dict()
    sex_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
    results['income_by_sex'] = {
        'Male': float(sex_income.loc['Male', '>50K']),
        'Female': float(sex_income.loc['Female', '>50K'])
    }
    
    # Calculate DPR for sex
    male_rate = (df[df['sex'] == 'Male']['income'] == '>50K').mean()
    female_rate = (df[df['sex'] == 'Female']['income'] == '>50K').mean()
    results['dpr_sex'] = float(female_rate / male_rate) if male_rate > 0 else 0
    
    # Race analysis
    results['race_distribution'] = df['race'].value_counts().to_dict()
    race_income = pd.crosstab(df['race'], df['income'], normalize='index') * 100
    results['income_by_race'] = {race: float(race_income.loc[race, '>50K']) for race in race_income.index}
    
    # Calculate DPR for race (White vs non-White)
    white_rate = (df[df['race'] == 'White']['income'] == '>50K').mean()
    nonwhite_rate = (df[df['race'] != 'White']['income'] == '>50K').mean()
    results['dpr_race'] = float(nonwhite_rate / white_rate) if white_rate > 0 else 0
    
    # Intersectional analysis
    intersectional = {}
    for sex in ['Male', 'Female']:
        for race in ['White', 'Black']:
            subset = df[(df['sex'] == sex) & (df['race'] == race)]
            rate = (subset['income'] == '>50K').mean() * 100
            intersectional[f'{sex}_{race}'] = float(rate)
    results['intersectional'] = intersectional
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def print_eda_report(results: dict = None):
    """Print a formatted EDA report."""
    if results is None:
        results = run_eda()
    
    print('=' * 70)
    print('ADULT CENSUS DATASET - EDA REPORT')
    print('=' * 70)
    
    print(f"\nDataset Size: {results['total_samples']:,} samples")
    print(f"Features: {results['total_features']}")
    
    print('\nTarget Distribution:')
    for label, count in results['target_distribution'].items():
        pct = count / results['total_samples'] * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    print('\n' + '-' * 50)
    print('BIAS ANALYSIS')
    print('-' * 50)
    
    print('\nGender Bias:')
    for sex, rate in results['income_by_sex'].items():
        print(f"  {sex}: {rate:.1f}% earn >50K")
    print(f"  DPR (Female/Male): {results['dpr_sex']:.3f}")
    print(f"  Status: {'FAIR' if 0.8 <= results['dpr_sex'] <= 1.25 else 'UNFAIR - Bias detected!'}")
    
    print('\nRacial Bias:')
    for race, rate in results['income_by_race'].items():
        print(f"  {race}: {rate:.1f}%")
    print(f"  DPR (Non-White/White): {results['dpr_race']:.3f}")
    print(f"  Status: {'FAIR' if 0.8 <= results['dpr_race'] <= 1.25 else 'UNFAIR - Bias detected!'}")
    
    print('\nIntersectional Analysis:')
    for group, rate in results['intersectional'].items():
        print(f"  {group.replace('_', ' + ')}: {rate:.1f}%")
    
    print('=' * 70)


if __name__ == "__main__":
    results = run_eda()
    print_eda_report(results)
