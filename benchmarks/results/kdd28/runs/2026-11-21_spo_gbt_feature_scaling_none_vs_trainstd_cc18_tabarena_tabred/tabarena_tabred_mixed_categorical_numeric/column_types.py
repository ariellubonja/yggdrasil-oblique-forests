#!/usr/bin/env python3
"""Per-column categorical flag from the raw data, same rule as benchmarks/data/build_kdd28_categoricals.py.
TabArena: OpenML is_categorical or a non-numeric dtype. TabReD: cat_* columns. Writes column_types.json {ds: {col: 0/1}}.
Usage: column_types.py <repo>/benchmarks/data <out.json> <ds>[,<ds>...]"""
import json, pathlib, sys

import pandas as pd

D, out, dss = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
res = {}
for ds in dss:
    if (D / "tabarena" / ds / "info.json").exists():
        info = json.load(open(D / "tabarena" / ds / "info.json"))
        df = pd.read_parquet(D / "tabarena" / ds / "data.parquet")
        res[ds] = {c: int(bool(info["columns"].get(c, {}).get("is_categorical", False) or df[c].dtype.kind not in "iuf"))
                   for c in df.columns if c != info["target_feature"]}
    else:
        hdr = open(D / "tabred_raw_mirror" / f"{ds}_train_full.csv").readline().strip().split(",")
        res[ds] = {h: int(h.startswith("cat_")) for h in hdr if h.split("_")[0] in ("num", "bin", "cat")}
    print(ds, sum(res[ds].values()), "/", len(res[ds]))
json.dump(res, open(out, "w"), indent=0)
