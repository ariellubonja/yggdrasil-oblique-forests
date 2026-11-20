#!/usr/bin/env python3
"""Count the categorical features of every KDD28 accuracy-table dataset, from the raw source data.

categorical = non-numeric in the source, i.e. a feature the prep ordinal-encodes or the source already integer-coded:
  TabArena  benchmarks/data/tabarena/<name>/{info.json,data.parquet}: OpenML is_categorical, or a str/bool/date dtype.
  TabReD    benchmarks/data/tabred_raw_mirror/<name>_train_full.csv: the cat_* block (bin_* 0/1 flags count as numeric).
  Wide      OpenML /data/features/<id>: nominal or string, minus target, ignored, row-id and manifest drop_columns.
Writes benchmarks/data/dataset_properties/kdd28_dataset_categoricals.csv (dataset,categorical,features,source). Run from anywhere:
  python3 benchmarks/data/build_kdd28_categoricals.py
"""
import csv, json, pathlib, subprocess

import pandas as pd

D = pathlib.Path(__file__).resolve().parent
WIDE_OPENML = {"santander_transaction": 42395, "dota2": 45563, "hcdr_main": 45567, "christine": 41142,
               "philippine": 41145, "guillermo": 41159, "vehicle_sensIT": 357}
rows = []

for ddir in sorted((D / "tabarena").iterdir(), key=lambda p: p.name.lower()):
    if ddir.name.startswith("_") or not (ddir / "info.json").exists():
        continue
    info = json.load(open(ddir / "info.json"))
    df = pd.read_parquet(ddir / "data.parquet")
    feats = [c for c in df.columns if c != info["target_feature"]]
    n = sum(info["columns"].get(c, {}).get("is_categorical", False) or df[c].dtype.kind not in "iuf" for c in feats)
    rows.append((ddir.name, n, len(feats), f"tabarena parquet, OpenML data {info['openml_dataset_id']}"))

for f in sorted((D / "tabred_raw_mirror").glob("*_train_full.csv")):
    hdr = open(f).readline().strip().split(",")
    n_feat = sum(h.split("_")[0] in ("num", "bin", "cat") for h in hdr)
    rows.append((f.name.removesuffix("_train_full.csv"), sum(h.startswith("cat_") for h in hdr), n_feat,
                 "tabred_raw_mirror cat_* columns"))

man = {e["name"]: e for e in json.load(open(D / "kdd28_wide_datasets_manifest.json"))["datasets"]}
rows.append(("epsilon", 0, man["epsilon"]["expected_features"], "LibSVM epsilon_normalized, all numeric"))
for name, did in WIDE_OPENML.items():
    url = f"https://www.openml.org/api/v1/json/data/features/{did}"
    feats = json.loads(subprocess.run(["curl", "-sf", "--max-time", "60", url], check=True, capture_output=True).stdout)["data_features"]["feature"]
    keep = [x for x in feats if x["is_target"] == "false" and x["is_ignore"] == "false"
            and x["is_row_identifier"] == "false" and x["name"] not in man[name]["drop_columns"]]
    n = sum(x["data_type"] in ("nominal", "string") for x in keep)
    rows.append((name, n, len(keep), f"OpenML data {did} feature types"))

with open(D / "dataset_properties/kdd28_dataset_categoricals.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "categorical", "features", "source"])
    w.writerows(rows)
print(len(rows), "datasets ->", D / "dataset_properties/kdd28_dataset_categoricals.csv")
