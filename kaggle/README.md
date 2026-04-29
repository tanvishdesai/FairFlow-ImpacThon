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

## Main script

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

This creates:

- `research_outputs/main_benchmark/main_results.csv`
- `research_outputs/main_benchmark/rolling_traces.csv`
- trained base models
- trained universal RL controller
- trained dataset-specific RL controllers for the configured model families

## Ablations

State ablation:

```python
!python kaggle/02_state_ablation.py --search-root /kaggle/input
```

Reward ablation:

```python
!python kaggle/03_reward_ablation.py --search-root /kaggle/input
```

Order stress:

```python
!python kaggle/04_order_stress.py --search-root /kaggle/input
```

Paper tables:

```python
!python kaggle/05_make_paper_tables.py
```

## Suggested execution plan

1. Run `01_main_benchmark.py`
2. Inspect `main_results.csv`
3. Run `02_state_ablation.py`
4. Run `03_reward_ablation.py`
5. Run `04_order_stress.py`
6. Run `05_make_paper_tables.py`

## Notes

- The default benchmark trains dataset-specific RL only for `xgboost` to keep
  compute reasonable. You can expand that through `--dataset-specific-models`.
- The universal RL controller is trained on synthetic fairness scenarios and
  then tested on real datasets, which matches the paper framing.
- All result files are CSV and LaTeX-friendly.

