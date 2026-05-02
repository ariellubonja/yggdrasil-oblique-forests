# ------------------------------------------------------------
# requirements
# ------------------------------------------------------------
#   pip install openml pandas tqdm
# ------------------------------------------------------------

import openml
import pandas as pd
from tqdm import tqdm
import os

# Where the csv files should be written
DEST_DIR = "cc18_binary_csv"
os.makedirs(DEST_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. get the list of *binary* CC-18 tasks
# ------------------------------------------------------------
binary_tasks_df = openml.tasks.list_tasks(
    tag="OpenML-CC18",
    type=1,                 # 1 = classification
    number_classes=2,       # keep only binary problems
    output_format="dataframe"
)

binary_task_ids = binary_tasks_df.index.tolist()
print(f"Found {len(binary_task_ids)} binary tasks in CC-18.")

# ------------------------------------------------------------
# 2. loop through the tasks and materialise the splits
# ------------------------------------------------------------
for tid in tqdm(binary_task_ids, desc="downloading & writing"):
    task    = openml.tasks.get_task(tid, download_data=False)
    dset    = task.get_dataset()
    dname   = dset.name                    # e.g. 'banana'
    target  = dset.default_target_attribute

    # Obtain the full dataframe once
    X, y, _, _ = dset.get_data(dataset_format="dataframe",
                               target=target)
                            #    handle_missing=True)   # gives NaNs where needed
    df = X.copy()
    df[target] = y

    task_dir = os.path.join(DEST_DIR, f"task_{tid}_{dname}")
    os.makedirs(task_dir, exist_ok=True)

    # Iterate over the predefined CV splits
    # ------------------------------------------------------------------
    # NEW helper --------------------------------------------------------
    # ------------------------------------------------------------------
    def _n(param_dict, key, default=1):
        """Return an int(parameter_dict[key]) or a default value."""
        return int(param_dict.get(key, default))

    # ------------------------------------------------------------------
    # inside the main loop, *after* we created `task`  ------------------
    # ------------------------------------------------------------------
    est_params = task.estimation_procedure.get("parameters", {})  # note: "parameters"
    n_repeats = int(est_params.get("number_repeats", 1))
    n_folds = int(est_params.get("number_folds", 1))
    n_samples = int(est_params.get("number_samples", 1))  # often 1

    # Iterate over the predefined CV splits ----------------------------
    for repeat in range(n_repeats):
        for fold in range(n_folds):
            for sample in range(n_samples):            # most tasks: sample = 0 only
                train_idx, test_idx = task.get_train_test_split_indices(
                    repeat=repeat, fold=fold, sample=sample
                )

                train_path = os.path.join(
                    task_dir, f"repeat{repeat}_fold{fold}_sample{sample}_train.csv"
                )
                test_path = os.path.join(
                    task_dir, f"repeat{repeat}_fold{fold}_sample{sample}_test.csv"
                )

                df.iloc[train_idx].to_csv(train_path, index=False)
                df.iloc[test_idx].to_csv(test_path,  index=False)

print("\nAll done.")
print(f"CSV files are in   {os.path.abspath(DEST_DIR)}")