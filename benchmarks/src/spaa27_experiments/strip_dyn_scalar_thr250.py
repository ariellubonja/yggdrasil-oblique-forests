#!/usr/bin/env python3
"""Move every spo_rf_dyn_scalar row recorded with --dynamic_split_threshold 250 (the
pre-2026-09-09 mis-set threshold, see PROTOCOL.md) out of a results CSV into
results/_tests/dyn_scalar_thr250_<name>.csv so skip-existing re-runs them at 4600.
Usage: strip_dyn_scalar_thr250.py <csv> [<csv> ...]   (rewrites in place; run only
while no driver holds the file open -- i.e. under the study lock)."""
import os, sys
import pandas as pd
for path in sys.argv[1:]:
    df = pd.read_csv(path)
    col = "arm" if "arm" in df.columns else "method"
    bad = (df[col] == "spo_rf_dyn_scalar") & df["cmd"].astype(str).str.contains(
        "dynamic_split_threshold 250", regex=False)
    if not bad.any():
        print(f"{path}: nothing to strip"); continue
    tdir = os.path.join(os.path.dirname(path), "_tests"); os.makedirs(tdir, exist_ok=True)
    out = os.path.join(tdir, "dyn_scalar_thr250_" + os.path.basename(path))
    df[bad].to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    df[~bad].to_csv(path, index=False)
    print(f"{path}: moved {int(bad.sum())} rows -> {out}")
