#!/usr/bin/env python3
# Off-paper sensitivity folds for ct_slices: GroupKFold(5) by patientId (no patient in both train and test).
# Same encoded matrix as the screen (src/ct_slices/train_nan.csv, no NaN), %.9g like make_folds. Not used by the rule.
import io, json, os, sys, zipfile
import pandas as pd
from sklearn.model_selection import GroupKFold
B = sys.argv[1]
X = pd.read_csv(f"{B}/data/src/ct_slices/train_nan.csv")
pid = pd.read_csv(io.BytesIO(zipfile.ZipFile(f"{B}/data/raw/ct_slices/ct_slices.zip").read("slice_localization_data.csv")))["patientId"]
assert len(pid) == len(X) and not X.isna().any().any()
out = f"{B}/data/folds_reg_groupkfold/ct_slices"; os.makedirs(out, exist_ok=True)
idx = {}
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, groups=pid)):
    assert not set(pid.iloc[tr]) & set(pid.iloc[te])
    X.iloc[tr].to_csv(f"{out}/fold{k}_train.csv", index=False, float_format="%.9g")
    X.iloc[te].to_csv(f"{out}/fold{k}_test.csv", index=False, float_format="%.9g")
    idx[str(k)] = {"train_idx": tr.tolist(), "test_idx": te.tolist(), "test_patients": sorted(map(int, set(pid.iloc[te])))}
json.dump({"protocol": "GroupKFold(5) by patientId, off-paper sensitivity", "indices": idx}, open(f"{out}/folds.json", "w"))
print("ok", {k: len(v["test_patients"]) for k, v in idx.items()})
