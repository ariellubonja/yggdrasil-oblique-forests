#!/usr/bin/env python3
"""Recreate the CC18 binary-classification train CSVs from OpenML, deterministically.

Reads cc18_manifest.json (pinned task_id + dataset_id + expected row count) and
writes a train+test CSV for every CV split (repeat{r}_fold{f}_sample{s}_{train,test}.csv)
per task into benchmarks/data/cc18_binary_csv/. CC18 tasks are 10-fold CV, so each
dataset yields 10 train/test pairs. Skips entries marked status="issue" (creates
the issue_<folder>/ marker dir with a README explaining why).

  pip install openml pandas tqdm
  python3 benchmarks/data/download_cc18_datasets.py
"""
import json
import os
import sys

import openml
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_binary_csv")
MANIFEST_PATH = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_manifest.json")


def write_all_splits(entry, task_dir):
    """Write train+test CSVs for every (repeat, fold, sample) split of the task.

    CC18 tasks are 10-fold CV (1 repeat x 10 folds x 1 sample), so this yields
    10 train/test pairs per dataset. Returns the row count of the
    repeat0_fold0_sample0 train split (the one the manifest validates).
    """
    task = openml.tasks.get_task(entry["task_id"], download_data=False)
    dset = openml.datasets.get_dataset(entry["dataset_id"])
    target = entry["label_column"]
    X, y, _, _ = dset.get_data(dataset_format="dataframe", target=target)
    df = X.copy()
    df[target] = y

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    fold0_train_rows = None
    for r in range(n_repeats):
        for f in range(n_folds):
            for s in range(n_samples):
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=r, fold=f, sample=s
                )
                stem = f"repeat{r}_fold{f}_sample{s}"
                df.iloc[train_idx].to_csv(
                    os.path.join(task_dir, f"{stem}_train.csv"), index=False
                )
                df.iloc[test_idx].to_csv(
                    os.path.join(task_dir, f"{stem}_test.csv"), index=False
                )
                if r == 0 and f == 0 and s == 0:
                    fold0_train_rows = len(train_idx)
    return fold0_train_rows


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    os.makedirs(DATA_DIR, exist_ok=True)

    failures = []
    for entry in tqdm(manifest["entries"], desc="cc18"):
        prefix = "issue_" if entry["status"] == "issue" else ""
        folder = f"{prefix}task_{entry['task_id']}_{entry['folder_name']}"
        task_dir = os.path.join(DATA_DIR, folder)
        os.makedirs(task_dir, exist_ok=True)

        if entry["status"] == "issue":
            with open(os.path.join(task_dir, "README.txt"), "w") as f:
                f.write(
                    f"Skipped: {entry.get('issue_reason', 'unknown')}\n"
                    f"task_id={entry['task_id']} dataset_id={entry['dataset_id']}\n"
                )
            continue

        rows = write_all_splits(entry, task_dir)

        if rows != entry["expected_rows_train"]:
            failures.append(
                f"task_{entry['task_id']}: expected {entry['expected_rows_train']} "
                f"rows, got {rows}"
            )

    if failures:
        print("\nROW-COUNT MISMATCH (OpenML data may have changed):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        print(
            "\nIf the new counts are correct, regenerate the manifest with "
            "build_cc18_manifest.py and review the diff.",
            file=sys.stderr,
        )
        return 1

    print(f"\nDone. CSVs in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
