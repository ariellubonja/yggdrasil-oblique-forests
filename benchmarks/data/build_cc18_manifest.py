#!/usr/bin/env python3
"""Bootstrap cc18_manifest.json from the current on-disk CC18 dir.

For each task_*/ and issue_task_*/ folder under benchmarks/data/cc18_binary_csv/,
resolve task_id -> dataset_id via OpenML, count rows in the train CSV, and read
the label column from the header. Writes benchmarks/data/cc18_manifest.json.

Run once; the manifest is then committed and consumed by download_cc18_datasets.py.

  pip install openml
  python3 benchmarks/data/build_cc18_manifest.py
"""
import json
import os
import re
import sys

import openml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_binary_csv")
MANIFEST_PATH = os.path.join(REPO_ROOT, "benchmarks", "data", "cc18_manifest.json")

ISSUE_REASONS = {
    3021: "TBG column 100% missing values — binary aborts during dataspec creation",
}

FOLDER_RE = re.compile(r"^(issue_)?task_(\d+)_(.+)$")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: {DATA_DIR} not found", file=sys.stderr)
        return 1

    entries = []
    for folder in sorted(os.listdir(DATA_DIR)):
        m = FOLDER_RE.match(folder)
        if not m:
            continue
        is_issue = m.group(1) is not None
        task_id = int(m.group(2))
        folder_name_part = m.group(3)

        print(f"[{task_id}] resolving via OpenML...", flush=True)
        task = openml.tasks.get_task(task_id, download_data=False)
        dset = task.get_dataset()
        dataset_id = int(dset.dataset_id) if hasattr(dset, "dataset_id") else int(dset.id)
        dataset_name = dset.name
        label_column = task.target_name or dset.default_target_attribute

        train_csv = os.path.join(DATA_DIR, folder, "repeat0_fold0_sample0_train.csv")
        expected_rows = None
        if os.path.isfile(train_csv):
            with open(train_csv) as f:
                expected_rows = sum(1 for _ in f) - 1  # minus header
            with open(train_csv) as f:
                header_last = f.readline().rstrip("\r\n").split(",")[-1].strip()
            if header_last and header_last != label_column:
                label_column = header_last

        entry = {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "folder_name": folder_name_part,
            "label_column": label_column,
            "expected_rows_train": expected_rows,
            "status": "issue" if is_issue else "ok",
        }
        if is_issue:
            entry["issue_reason"] = ISSUE_REASONS.get(
                task_id, "unknown — see commit history"
            )
        entries.append(entry)

    entries.sort(key=lambda e: e["task_id"])
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"version": 1, "entries": entries}, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(entries)} entries to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
