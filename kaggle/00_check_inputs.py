"""
Quick Kaggle input verification for the FairFlow paper pipeline.
"""

from __future__ import annotations

import argparse

import pandas as pd

from research.datasets import available_dataset_names, load_dataset, resolve_dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that FairFlow datasets are discoverable before training.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Directory to search for raw dataset files. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=available_dataset_names(),
        help="Datasets to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_roots = args.search_root or ["/kaggle/input", "data/raw", "."]

    rows: list[dict] = []
    for dataset_name in args.datasets:
        try:
            path = resolve_dataset_path(dataset_name, search_roots)
            bundle = load_dataset(dataset_name, search_roots=search_roots)
            rows.append(
                {
                    "dataset": dataset_name,
                    "status": "ok",
                    "path": str(path),
                    "train_samples": len(bundle.X_train),
                    "test_samples": len(bundle.X_test),
                    "protected_attribute": bundle.protected_attribute,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset_name,
                    "status": "error",
                    "path": "",
                    "train_samples": "",
                    "test_samples": "",
                    "protected_attribute": "",
                    "reason": str(exc),
                }
            )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

