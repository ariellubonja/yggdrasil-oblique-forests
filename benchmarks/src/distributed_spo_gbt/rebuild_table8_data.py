#!/usr/bin/env python3
"""Rebuild the 33 suite datasets + CV folds behind KDD28 Table 8 (tab:dist-accuracy) and check them against the
recorded sha256s (speedup_map_by_dataset/MANIFEST/inputs.sha256). Follows spaa27_experiments/REPLICABILITY.md §4.

  <venv>/bin/python rebuild_table8_data.py <folds_dir>
Run with the pinned venv (numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1, openml 0.15.1).
"""
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(REPO, "benchmarks", "data")
sys.path.insert(0, DATA)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "src", "spaa27_experiments"))
from tabular_suite_prep import write_dataset  # noqa: E402
from run_suite import make_folds  # noqa: E402

TABRED = ["ecom-offers", "homecredit-default", "homesite-insurance"]
HF_REPO = "danil-e/harbor-datasets-tabred"
SHA = os.path.join(REPO, "benchmarks/results/runtime/speedup_map_by_dataset/MANIFEST/inputs.sha256")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    folds_dir = sys.argv[1]
    entries = [e for e in json.load(open(os.path.join(DATA, "tabarena_binary_manifest.json")))["entries"]
               if e["status"] == "ok"]
    names = [e["dataset_name"] for e in entries]
    subprocess.run([sys.executable, os.path.join(DATA, "download_tabarena_datasets.py"), *names], check=True)
    ta_out = os.path.join(DATA, "tabarena_binary_csv")
    for name in names:
        if os.path.isfile(os.path.join(ta_out, name, "train.csv")):
            continue
        d = os.path.join(DATA, "tabarena", name)
        df = pd.read_parquet(os.path.join(d, "data.parquet"))
        target = json.load(open(os.path.join(d, "info.json")))["target_feature"]
        write_dataset(ta_out, name, df.drop(columns=[target]), df[target])
        print("wrote", name, flush=True)

    from huggingface_hub import hf_hub_download
    tr_out = os.path.join(DATA, "tabred_binary_csv")
    mirror = os.path.join(DATA, "tabred_raw_mirror")
    os.makedirs(mirror, exist_ok=True)
    for name in TABRED:
        raw_p = os.path.join(mirror, f"{name}_train_full.csv")
        if not os.path.isfile(raw_p):
            p = hf_hub_download(HF_REPO, f"datasets/tabred-real/{name}/environment/data/train_full.csv",
                                repo_type="dataset")
            os.symlink(p, raw_p)
        if os.path.isfile(os.path.join(tr_out, name, "train.csv")):
            continue
        raw = pd.read_csv(raw_p)
        write_dataset(tr_out, name, raw.drop(columns=["id", "timestamp", "target"], errors="ignore"), raw["target"])
        print("wrote", name, flush=True)

    subprocess.run([sys.executable, os.path.join(REPO, "benchmarks/src/spaa27_experiments/make_train_nan.py")],
                   check=True)
    for out, chrono in [(ta_out, False), (tr_out, True)]:
        for name in sorted(os.listdir(out)):
            tc = os.path.join(out, name, "train.csv")
            if os.path.isfile(tc):
                make_folds(name, tc, "label", 5, 0, folds_dir, chrono=chrono, holdout_frac=0.2)

    want = {}
    for line in open(SHA):
        h, p = line.split()
        for key in ("/spo_vs_gbt/folds/", "/benchmarks/data/"):
            if key in p and p.endswith(".csv"):
                want[("folds/" if "folds" in key else "data/") + p.split(key, 1)[1]] = h
    ok = bad = miss = 0
    for k, h in sorted(want.items()):
        local = os.path.join(folds_dir, k[6:]) if k.startswith("folds/") else os.path.join(DATA, k[5:])
        if not os.path.isfile(local):
            miss += 1
            continue
        if sha256(local) == h:
            ok += 1
        else:
            bad += 1
            print("MISMATCH", k)
    print(f"sha256 vs manifest: {ok} match, {bad} mismatch, {miss} not rebuilt")


if __name__ == "__main__":
    main()
