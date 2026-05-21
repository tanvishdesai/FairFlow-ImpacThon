"""
Compute paired statistical significance tests for CODA vs baselines.
Uses Wilcoxon signed-rank test (non-parametric, appropriate for paired samples with small n).
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv(r"c:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\additional coda v2 run\coda_v2_benchmark\per_run_results.csv")

# Filter to natural ordering only
df = df[df['order_protocol'] == 'natural'].copy()

# We want per-(dataset, model, seed) paired comparisons
# Each unique (dataset, model, seed) is one "run"
methods_of_interest = ['base_model', 'group_threshold', 'guard_threshold', 'universal_rl', 'coda']
metrics = ['accuracy', 'demographic_parity_ratio', 'equalized_odds_gap', 'intervention_rate']

print("=" * 80)
print("STATISTICAL SIGNIFICANCE TESTS: CODA vs. Baselines")
print("Wilcoxon signed-rank test (paired, two-sided)")
print("=" * 80)

# Pivot to get CODA values aligned with each baseline
coda_df = df[df['method'] == 'coda'].set_index(['dataset', 'model', 'seed'])

results_rows = []

for baseline in ['base_model', 'group_threshold', 'guard_threshold', 'universal_rl']:
    baseline_df = df[df['method'] == baseline].set_index(['dataset', 'model', 'seed'])
    
    # Align on common index
    common_idx = coda_df.index.intersection(baseline_df.index)
    
    print(f"\n--- CODA vs. {baseline} (n={len(common_idx)} paired observations) ---")
    
    for metric in ['accuracy', 'demographic_parity_ratio', 'equalized_odds_gap']:
        coda_vals = coda_df.loc[common_idx, metric].values
        base_vals = baseline_df.loc[common_idx, metric].values
        
        diffs = coda_vals - base_vals
        
        # Remove zero differences (Wilcoxon requires non-zero diffs)
        nonzero_mask = diffs != 0
        if nonzero_mask.sum() < 3:
            print(f"  {metric}: Too few non-zero differences for test")
            continue
        
        stat, p_value = stats.wilcoxon(diffs[nonzero_mask], alternative='two-sided')
        
        mean_diff = np.mean(diffs)
        ci_low = np.mean(diffs) - 1.96 * np.std(diffs) / np.sqrt(len(diffs))
        ci_high = np.mean(diffs) + 1.96 * np.std(diffs) / np.sqrt(len(diffs))
        
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        
        print(f"  {metric}:")
        print(f"    CODA mean: {np.mean(coda_vals):.4f}, {baseline} mean: {np.mean(base_vals):.4f}")
        print(f"    Mean diff: {mean_diff:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"    Wilcoxon p={p_value:.4f} {sig}")
        
        results_rows.append({
            'comparison': f'CODA vs. {baseline}',
            'metric': metric,
            'coda_mean': np.mean(coda_vals),
            'baseline_mean': np.mean(base_vals),
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'significance': sig,
            'n_pairs': len(common_idx)
        })

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE FOR PAPER")
print("=" * 80)

results_df = pd.DataFrame(results_rows)

# Focus on DPR (the main metric)
dpr_results = results_df[results_df['metric'] == 'demographic_parity_ratio']
print("\nDPR (Demographic Parity Ratio) - Paired Comparisons:")
print(dpr_results[['comparison', 'coda_mean', 'baseline_mean', 'mean_diff', 'ci_low', 'ci_high', 'p_value', 'significance']].to_string(index=False))

acc_results = results_df[results_df['metric'] == 'accuracy']
print("\nAccuracy - Paired Comparisons:")
print(acc_results[['comparison', 'coda_mean', 'baseline_mean', 'mean_diff', 'ci_low', 'ci_high', 'p_value', 'significance']].to_string(index=False))

eo_results = results_df[results_df['metric'] == 'equalized_odds_gap']
print("\nEO Gap - Paired Comparisons:")
print(eo_results[['comparison', 'coda_mean', 'baseline_mean', 'mean_diff', 'ci_low', 'ci_high', 'p_value', 'significance']].to_string(index=False))

# Also compute per-method means with 95% CI for the main table
print("\n" + "=" * 80)
print("PER-METHOD MEANS WITH 95% CI")
print("=" * 80)

for method in methods_of_interest:
    mdf = df[df['method'] == method]
    for metric in ['accuracy', 'demographic_parity_ratio', 'equalized_odds_gap', 'intervention_rate']:
        vals = mdf[metric].values
        mean = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        print(f"  {method:20s} {metric:35s}: {mean:.4f} ± {ci:.4f}  [{mean-ci:.4f}, {mean+ci:.4f}]")

# LaTeX table row format
print("\n" + "=" * 80)
print("LATEX TABLE (for insertion into paper)")
print("=" * 80)

print(r"""
\begin{table}[t]
\centering
\caption{Statistical significance of CODA's improvements over baselines (Wilcoxon signed-rank test, $n=30$ paired runs). $\Delta$DPR and $\Delta$Acc report the mean paired difference (CODA $-$ baseline). $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}
\label{tab:significance}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Comparison & $\Delta$DPR & $p$ & $\Delta$Acc & $p$ \\
\midrule""")

for baseline in ['base_model', 'group_threshold', 'guard_threshold', 'universal_rl']:
    dpr_row = results_df[(results_df['metric'] == 'demographic_parity_ratio') & (results_df['comparison'] == f'CODA vs. {baseline}')]
    acc_row = results_df[(results_df['metric'] == 'accuracy') & (results_df['comparison'] == f'CODA vs. {baseline}')]
    
    if len(dpr_row) > 0 and len(acc_row) > 0:
        dpr_r = dpr_row.iloc[0]
        acc_r = acc_row.iloc[0]
        
        # Format baseline name
        bname = baseline.replace('_', ' ').title()
        if bname == 'Base Model':
            bname = 'Base Model'
        elif bname == 'Group Threshold':
            bname = 'Group Thr.'
        elif bname == 'Guard Threshold':
            bname = 'Guard Thr.'
        elif bname == 'Universal Rl':
            bname = 'Universal RL'
        
        dpr_sig = dpr_r['significance'].replace('n.s.', '')
        acc_sig = acc_r['significance'].replace('n.s.', '')
        
        print(f"CODA vs.\\ {bname} & ${dpr_r['mean_diff']:+.3f}${dpr_sig} & ${dpr_r['p_value']:.3f}$ & ${acc_r['mean_diff']:+.3f}${acc_sig} & ${acc_r['p_value']:.3f}$ \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}""")
