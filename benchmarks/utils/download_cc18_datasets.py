#!/usr/bin/env python3
"""Recreate the CC18 binary-classification train CSVs from OpenML, deterministically.

Reads cc18_manifest.json (pinned task_id + dataset_id + expected row count) and
writes one repeat0_fold0_sample0_train.csv per task into
benchmarks/data/cc18_binary_csv/. Skips entries marked status="issue" (creates
the issue_<folder>/ marker dir with a README explaining why).

  pip install openml pandas tqdm
  python3 benchmarks/utils/download_cc18_datasets.py
"""
import json
import os
import sys

import openml
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_binary_csv")
MANIFEST_PATH = os.path.join(REPO_ROOT, "benchmarks", "utils", "cc18_manifest.json")


def write_train_csv(entry, out_path):
    task = openml.tasks.get_task(entry["task_id"], download_data=False)
    dset = openml.datasets.get_dataset(entry["dataset_id"])
    target = entry["label_column"]
    X, y, _, _ = dset.get_data(dataset_format="dataframe", target=target)
    df = X.copy()
    df[target] = y
    train_idx, _ = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    df.iloc[train_idx].to_csv(out_path, index=False)


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

        train_csv = os.path.join(task_dir, "repeat0_fold0_sample0_train.csv")
        if os.path.isfile(train_csv):
            with open(train_csv) as f:
                rows = sum(1 for _ in f) - 1
            if rows == entry["expected_rows_train"]:
                continue  # already have correct file

        write_train_csv(entry, train_csv)

        with open(train_csv) as f:
            rows = sum(1 for _ in f) - 1
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
