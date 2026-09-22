#!/usr/bin/env python
"""Dequantize a feature CSV so exact (VQSort) split search cannot exploit tied values.

Why (2026-09-22): Numerai (5 levels per feature), Criteo (integer counts + ordinal codes) and
NYC taxi (IDs, counts, 2-decimal amounts) have few distinct values per column. Highway VQSort
finishes an all-equal or two-valued partition in one pass (PivotResult::kDone), so the exact
finder runs 2-3x cheaper per row on such data than on continuous data (HIGGS, SUSY, Epsilon),
while the histogram binner's cost is value-independent. Adding uniform noise inside each
column's quantization step makes every projected value distinct, i.e. the data looks like a
continuous dataset to the sort, and changes nothing else (rows, features, label, order
between levels; the order *within* a level becomes random, as for a continuous signal that had
been binned).

Rule per feature column:
  step = smallest positive gap between distinct values in a strided row sample, after
         removing the column's imputation mean (see --exclude_meta_means /
         --auto_exclude_single_nonint);
  x'   = fp32(x + U(-step/2, +step/2)), seeded per column (seed, column index).
Values above 2^24 in magnitude (Criteo's two largest code columns, taxi's epoch seconds) are
unchanged at fp32 resolution -- those columns already have >1e5 distinct values.
The label column is copied unchanged. Writes <out>.csv and <out>_meta.json (steps, seed).
"""
import argparse, json, os, sys, time
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv


def column_step(sample: np.ndarray, exclude: list[float]) -> tuple[float, int, int]:
    u = np.unique(sample.astype(np.float64))
    if exclude:
        keep = np.ones(len(u), dtype=bool)
        for e in exclude:
            keep &= ~np.isclose(u, e, rtol=1e-6, atol=1e-9)
        u = u[keep]
    n_distinct = len(u)
    if n_distinct < 2:
        return 1.0, n_distinct, 0
    gaps = np.diff(u)
    gaps = gaps[gaps > 0]
    return float(gaps.min()), n_distinct, int((np.abs(u - np.round(u)) < 1e-9).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--sample_rows", type=int, default=2_000_000, help="strided row sample used to find each column's step")
    ap.add_argument("--exclude_meta_means", default="", help="path to a *_meta.json whose imputation['train_means'] (or imputation_means) lists per-column means to ignore when computing steps (taxi)")
    ap.add_argument("--auto_exclude_single_nonint", action="store_true", help="if a column has >=2 integer-valued distinct values and exactly one non-integer distinct value, treat that value as the imputation mean (Criteo)")
    a = ap.parse_args()

    t0 = time.time()
    with open(a.in_csv) as f:
        header = f.readline().rstrip("\n").split(",")
    feats = [c for c in header if c != a.label_col]
    col_types = {c: pa.float32() for c in feats}
    col_types[a.label_col] = pa.int64()
    table = pacsv.read_csv(a.in_csv, convert_options=pacsv.ConvertOptions(column_types=col_types),
                           read_options=pacsv.ReadOptions(block_size=64 << 20))
    n = table.num_rows
    print(f"read {n} rows x {len(header)} cols in {time.time()-t0:.0f}s", flush=True)

    meta_means = {}
    if a.exclude_meta_means:
        m = json.load(open(a.exclude_meta_means))
        meta_means = (m.get("imputation", {}) or {}).get("train_means", {}) or m.get("imputation_means", {}) or {}

    stride = max(1, n // a.sample_rows)
    sidx = np.arange(0, n, stride)
    steps, report = {}, {}
    out_cols = {}
    for j, name in enumerate(header):
        col = table.column(name)
        if name == a.label_col:
            out_cols[name] = col
            continue
        x = col.to_numpy(zero_copy_only=False)
        s = x[sidx]
        exclude = []
        if name in meta_means:
            exclude.append(float(meta_means[name]))
        if a.auto_exclude_single_nonint:
            u = np.unique(s.astype(np.float64))
            nonint = u[np.abs(u - np.round(u)) > 1e-9]
            if len(nonint) == 1 and len(u) - 1 >= 2:
                exclude.append(float(nonint[0]))
        step, n_distinct, n_int = column_step(s, exclude)
        rng = np.random.default_rng([a.seed, j])
        noise = (rng.random(n, dtype=np.float32) - np.float32(0.5)) * np.float32(step)
        y = (x + noise).astype(np.float32)
        # Order between levels is preserved: rounding back to the step grid recovers x
        # wherever fp32 can represent the offset. Checked on the sample.
        back = np.round((y[sidx] - x[sidx]) / step)
        assert np.all(np.abs(back) <= 0.5 + 1e-3) or np.max(np.abs(x[sidx])) >= 2**23, name
        steps[name] = step
        report[name] = {"step": step, "sample_distinct_before": n_distinct, "sample_distinct_after": int(len(np.unique(y[sidx]))),
                        "excluded": exclude, "integer_valued_distinct": n_int}
        out_cols[name] = pa.array(y, type=pa.float32())
        if j % 200 == 0 or j == len(header) - 1:
            print(f"  col {j}/{len(header)} {name}: step={step:g} distinct(sample) {n_distinct} -> {report[name]['sample_distinct_after']}  [{time.time()-t0:.0f}s]", flush=True)
    del table
    out = pa.table([out_cols[c] for c in header], names=header)
    tw = time.time()
    # Arrow always quotes header names; the harness wants a bare header, so write it by hand.
    with pa.OSFile(a.out_csv, "wb") as sink:
        sink.write((",".join(header) + "\n").encode())
        pacsv.write_csv(out, sink, write_options=pacsv.WriteOptions(include_header=False, quoting_style="none"))
    print(f"wrote {a.out_csv}: {os.path.getsize(a.out_csv)/1e9:.1f} GB in {time.time()-tw:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
    meta = {"source_csv": a.in_csv, "rows": n, "seed": a.seed, "sample_stride": stride,
            "rule": "x' = fp32(x + U(-step/2, +step/2)); step = min positive gap between distinct sample values after removing the imputation mean; label unchanged",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "columns": report}
    json.dump(meta, open(os.path.splitext(a.out_csv)[0] + "_meta.json", "w"), indent=1)
    st = sorted(set(steps.values()))
    print("distinct steps:", st[:10], "..." if len(st) > 10 else "")


if __name__ == "__main__":
    main()
