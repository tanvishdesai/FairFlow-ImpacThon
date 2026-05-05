# FairFlow Kaggle Research Pipeline

This folder contains the paper-grade experiment stack that sits beside the
hackathon demo. The scripts are meant to be run either:

- locally from the repo root, or
- inside a Kaggle notebook after attaching the datasets under `/kaggle/input`.

## Recommended datasets

Attach these public datasets in Kaggle:

- Adult Census Income: `https://www.kaggle.com/datasets/uciml/adult-census-income`
- German Credit: `https://www.kaggle.com/datasets/uciml/german-credit`
- COMPAS: `https://www.kaggle.com/datasets/danofer/compass`
- Bank Marketing: `https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset`

For the recruitment dataset, upload your own CSV and make sure the file name
stays `fair_recrutment_dataset final.csv`.

## Original benchmark stack

The first-generation benchmark is still useful if you want the initial FairFlow
paper bundle exactly as it was originally analyzed.

Verify inputs:

```python
!python kaggle/00_check_inputs.py --search-root /kaggle/input --search-root data/raw
```

Run the full benchmark:

```python
!python kaggle/01_main_benchmark.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/main_benchmark \
    --datasets adult german_credit compas bank_marketing recruitment \
    --models logistic_regression random_forest xgboost \
    --orders natural
```

Run ablations:

```python
!python kaggle/02_state_ablation.py --search-root /kaggle/input
!python kaggle/03_reward_ablation.py --search-root /kaggle/input
!python kaggle/04_order_stress.py --search-root /kaggle/input
!python kaggle/05_make_paper_tables.py
```

## Upgraded paper pipeline

The upgraded stack reframes the contribution as selective, conservative fairness
control at deployment time instead of an always-on universal RL controller.

The upgraded benchmark includes these candidate methods beyond the original
baselines:

- `guard_threshold`
- `fairflow_guard_rl`
- `adaptive_guard`
- `primal_dual_guard`
- `primal_dual_offset`

### 1. Verify data access

```python
!python kaggle/00_check_inputs.py --search-root /kaggle/input --search-root data/raw
```

### 2. Run the full upgraded benchmark

```python
!python kaggle/06_elite_benchmark.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/elite_benchmark \
    --datasets adult german_credit compas bank_marketing recruitment \
    --models logistic_regression xgboost \
    --seeds 42 52 62
```

This produces:

- `research_outputs/elite_benchmark/per_run_results.csv`
- `research_outputs/elite_benchmark/aggregated_results.csv`
- `research_outputs/elite_benchmark/method_summary.csv`
- `research_outputs/elite_benchmark/paired_win_summary.csv`
- `research_outputs/elite_benchmark/guard_diagnostics.csv`

Expected row count:

- `5 datasets x 2 models x 9 methods x 3 seeds = 270`

### 3. Run selectivity and controller ablations

```python
!python kaggle/07_guard_ablation.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/elite_guard_ablation \
    --datasets adult compas recruitment \
    --seeds 42 52 62 \
    --model-name xgboost
```

Expected row count:

- `3 datasets x 9 methods x 3 seeds = 81`

### 4. Run order-stress robustness tests

```python
!python kaggle/08_elite_order_stress.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/elite_order_stress \
    --datasets adult compas recruitment \
    --protocols natural alternating_groups privileged_first unprivileged_first \
    --seeds 42 52 62 \
    --model-name xgboost
```

Expected row count:

- `3 datasets x 7 methods x 4 protocols x 3 seeds = 252`

### 5. Build paper-ready tables

```python
!python kaggle/09_make_elite_tables.py
```

## Focused adaptive validation

If you are iterating specifically on `adaptive_guard`, use the focused runner
before spending time on another full 5-dataset rerun:

```python
!python kaggle/10_focus_adaptive_guard.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-root "elite results adaptive focus"
```

Default focused scope:

- benchmark: `adult`, `compas`, `recruitment`
- ablation: `adult`, `compas`, `recruitment`
- order stress: `adult`, `compas`, `recruitment`
- models: `logistic_regression`, `xgboost`
- seeds: `42`, `52`, `62`

Expected row counts:

- benchmark: `3 datasets x 2 models x 9 methods x 3 seeds = 162`
- ablation: `3 datasets x 9 methods x 3 seeds = 81`
- order stress: `3 datasets x 7 methods x 4 protocols x 3 seeds = 252`

## Focused primal-dual benchmark

If you want to test the colleague-suggested online primal-dual controller
against the same baselines on the critical datasets, run:

```python
!python kaggle/11_primal_dual_benchmark.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-root research_outputs/primal_dual_focus
```

Default focused scope:

- benchmark: `adult`, `compas`, `recruitment`
- ablation: `adult`, `compas`, `recruitment`
- order stress: `adult`, `compas`, `recruitment`
- order protocols: `natural`, `privileged_first`, `unprivileged_first`
- models: `logistic_regression`, `xgboost`
- seeds: `42`, `52`, `62`

Expected row counts:

- benchmark: `3 datasets x 2 models x 10 methods x 3 seeds = 180`
- ablation: `3 datasets x 9 methods x 3 seeds = 81`
- order stress: `3 datasets x 8 methods x 3 protocols x 3 seeds = 216`

Optional fixed-parameter rerun after tuning:

```python
!python kaggle/11_primal_dual_benchmark.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-root research_outputs/primal_dual_focus_offset \
    --offset-lambda-lr 0.05 \
    --offset-delta-lr 0.01
```

## Primal-dual offset tuning sweep

Use this before the next large focused rerun. It tests the 9 candidate
`(lambda_lr, delta_lr)` pairs on `adult`, `compas`, and `recruitment`, then
recommends a configuration based on `adult + compas` fairness under natural
ordering.

```python
!python kaggle/13_primal_dual_offset_tune.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/primal_dual_offset_tuning
```

Expected row count:

- `3 datasets x 3 seeds x 9 hyperparameter pairs = 81`

## Targeted adaptive diagnostic

Before fully discarding `adaptive_guard`, run the focused diagnostic sweep that
tests whether higher deficit pressure can raise the accept-given-candidate rate
on the hardest datasets:

```python
!python kaggle/12_adaptive_guard_diagnostic.py \
    --search-root /kaggle/input \
    --search-root data/raw \
    --output-dir research_outputs/adaptive_guard_diagnostic
```

Default diagnostic scope:

- datasets: `adult`, `compas`
- model: `xgboost`
- seed: `42`
- deficit weights: `8`, `12`, `16`
- deficit utility slacks: `0.15`, `0.20`, `0.25`

## Recommended next execution plan

1. Run `13_primal_dual_offset_tune.py`
2. Inspect `research_outputs/primal_dual_offset_tuning/selection_table.csv`
3. Read the recommended `lambda_lr` and `delta_lr` from `best_offset_config.json`
4. Run `11_primal_dual_benchmark.py` with those fixed values
5. Run `09_make_elite_tables.py` on that focused output folder
6. If the focused run looks strong, run `06_elite_benchmark.py` with the same fixed values for the full 5-dataset benchmark

## Notes

- The universal RL controller is trained on synthetic fairness scenarios and
  then tested on real datasets, which matches the paper framing.
- All result files are CSV and LaTeX-friendly.
- The upgraded stack uses three seeds by default because multi-seed reporting is
  much easier to defend in a paper than single-run results.
- `logistic_regression` and `xgboost` are the default upgraded model families;
  this keeps runtime reasonable while still testing both linear and nonlinear
  base predictors.
