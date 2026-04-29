#!/usr/bin/env python3
"""Phase B: 25 OpenML datasets × 10 seeds × {axis,oblique} × depth sweep.

For each (dataset, seed, split_type, depth): train a GBT, evaluate test
accuracy, benchmark inference latency, append row to sweep_v3.csv.

Per-seed train/test split is in-memory; CSVs go to /tmp and get reused
across the 12 configs of that (dataset, seed) iteration.

Skips already-completed (dataset, seed, split, depth) tuples on resume.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import openml
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oblique_gbt_inference_sweep as base  # noqa: E402

REPO = base.REPO

# (name, openml_id) — 25 binary classification datasets, broad coverage.
DATASETS = [
    ("climate-model-crashes", 1467),
    ("wdbc", 1510),
    ("ilpd", 1480),
    ("monks-problems-2", 334),
    ("diabetes", 37),
    ("blood-transfusion", 1464),
    ("tic-tac-toe", 50),
    ("credit-g", 31),
    ("qsar-biodeg", 1494),
    ("steel-plates-fault", 1504),
    ("ozone-level-8hr", 1487),
    ("madelon", 1485),
    ("internet-advertisements", 40978),
    ("bioresponse", 4134),
    ("spambase", 44),
    ("gisette", 41026),
    ("mushroom", 24),
    ("phishing-websites", 4534),
    ("eeg-eye-state", 1471),
    ("nomao", 1486),
    ("adult", 1590),
    ("bank-marketing", 1461),
    ("electricity", 151),
    ("numerai28.6", 23517),
    ("jasmine", 41143),
]


def coerce_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Make every column numeric and dense. Sparse columns get densified;
    numeric-string columns get parsed; purely-categorical (object)
    columns get label-encoded."""
    out = df.copy()
    for c in out.columns:
        # Densify sparse columns (some OpenML datasets like gisette).
        if str(out[c].dtype).startswith("Sparse"):
            out[c] = out[c].sparse.to_dense().astype(float)
            continue
        if out[c].dtype == object or str(out[c].dtype).startswith("category"):
            tried = pd.to_numeric(out[c], errors="coerce")
            if tried.notna().sum() / max(1, len(tried)) > 0.9:
                out[c] = tried
            else:
                out[c] = out[c].astype("category").cat.codes.astype(float)
    out = out.fillna(0.0)
    return out


def fetch_dataset(name: str, oml_id: int, cache_dir: Path) -> pd.DataFrame:
    """Download once + cache as parquet for fast re-load."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{name}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    print(f"  fetching OpenML id={oml_id} ({name}) ...", flush=True)
    ds = openml.datasets.get_dataset(
        oml_id, download_data=True, download_qualities=False,
        download_features_meta_data=False)
    try:
        X, y, _, _ = ds.get_data(
            target=ds.default_target_attribute, dataset_format="dataframe")
    except Exception as e:
        # Sparse / unusual dataset — fall back to numpy array path.
        if "Sparse" in str(e) or "sparse" in str(e):
            print(f"  (sparse dataset, falling back to array) {e}",
                  flush=True)
            X, y, _, _ = ds.get_data(
                target=ds.default_target_attribute, dataset_format="array")
            # X may be scipy.sparse — densify.
            try:
                X = X.toarray()
            except AttributeError:
                pass
            X = pd.DataFrame(
                X, columns=[f"f{i}" for i in range(X.shape[1])])
            y = pd.Series(y)
        else:
            raise
    X = coerce_to_numeric(X)
    # Encode label to {0, 1}.
    if y.dtype.kind in "OSU" or str(y.dtype).startswith("category"):
        y = y.astype("category").cat.codes
    else:
        cats = sorted(pd.Series(y).dropna().unique().tolist())
        if len(cats) == 2 and set(cats) != {0, 1}:
            y = pd.Series(y).map({cats[0]: 0, cats[1]: 1})
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    if y.nunique() != 2:
        raise ValueError(f"{name}: expected binary label, got {y.nunique()} "
                         f"classes ({sorted(y.unique().tolist())[:5]})")
    df = X.copy()
    df["class"] = y.values
    df.to_parquet(cached)
    return df


def split_to_temp_csvs(df: pd.DataFrame, seed: int, train_frac: float,
                      max_train: int, out_dir: Path,
                      stem: str) -> tuple[Path, Path]:
    """Deterministic shuffle + 80/20 split + train cap. Writes two CSVs."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_tr = int(train_frac * len(df))
    tr = df.iloc[idx[:n_tr]]
    te = df.iloc[idx[n_tr:]]
    if len(tr) > max_train:
        tr = tr.iloc[:max_train]
    tr_path = out_dir / f"{stem}_seed{seed}_train.csv"
    te_path = out_dir / f"{stem}_seed{seed}_test.csv"
    tr.to_csv(tr_path, index=False)
    te.to_csv(te_path, index=False)
    return tr_path, te_path


def existing_keys(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    seen = set()
    with open(csv_path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                key = (row["dataset"], int(row["seed"]),
                       row["split_type"], int(row["max_depth"]))
                seen.add(key)
            except (KeyError, ValueError):
                continue
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--cache_dir",
                    default=str(REPO / "benchmarks/data/openml_phase_b"))
    ap.add_argument("--axis_depths", default="3,4,5,6,7,8,10")
    ap.add_argument("--oblique_depths", default="3,4,5,6,8")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--num_trees", type=int, default=100)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--num_proj_exp", type=float, default=1.0)
    ap.add_argument("--max_num_proj", type=int, default=1000)
    ap.add_argument("--density", type=float, default=2.0)
    ap.add_argument("--num_split_type", default="Exact")
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--num_runs", type=int, default=20)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--datasets", default=None,
                    help="Optional comma-list of dataset names to restrict "
                    "to; default = all 25.")
    ap.add_argument("--include_axis", action="store_true", default=True)
    ap.add_argument("--include_oblique", action="store_true", default=True)
    ap.add_argument("--skip_axis", dest="include_axis", action="store_false")
    ap.add_argument("--skip_oblique", dest="include_oblique",
                    action="store_false")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    axis_depths = [int(d) for d in args.axis_depths.split(",")]
    oblique_depths = [int(d) for d in args.oblique_depths.split(",")]
    cache_dir = Path(args.cache_dir)

    if args.datasets:
        wanted = set(args.datasets.split(","))
        ds_list = [(n, i) for n, i in DATASETS if n in wanted]
    else:
        ds_list = list(DATASETS)
    print(f"Plan: {len(ds_list)} datasets × {len(seeds)} seeds × "
          f"({len(axis_depths) * args.include_axis + len(oblique_depths) * args.include_oblique}) "
          f"configs", flush=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys(out_path)
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    fields = [
        "dataset", "split_type", "max_depth", "seed",
        "num_trees_requested", "num_trees_final", "shrinkage",
        "num_proj_exp", "density", "max_num_proj", "num_split_type",
        "n_train", "n_test", "n_features",
        "train_seconds", "accuracy", "log_loss",
        "best_engine", "best_us_per_ex",
        "generic_us_per_ex", "slow_us_per_ex",
        "all_engines",
    ]
    out_fh = open(out_path, "a", newline="")
    w = csv.DictWriter(out_fh, fieldnames=fields)
    if is_new:
        w.writeheader()
        out_fh.flush()

    overall_t0 = time.perf_counter()
    plan_total = sum(
        (len(axis_depths) if args.include_axis else 0)
        + (len(oblique_depths) if args.include_oblique else 0)
        for _ in ds_list) * len(seeds)
    done = 0
    skipped = 0
    failed = 0

    tmp_root = Path(tempfile.mkdtemp(prefix="ydf_phase_b_"))

    for ds_name, oml_id in ds_list:
        try:
            df = fetch_dataset(ds_name, oml_id, cache_dir)
        except Exception as e:
            print(f"[ERR] dataset {ds_name} failed to fetch: {e}", flush=True)
            failed += 1
            continue
        n_full = len(df)
        n_features = df.shape[1] - 1
        print(f"\n=== Dataset {ds_name} (id={oml_id}) "
              f"shape={df.shape} ===", flush=True)

        for seed in seeds:
            tr_csv, te_csv = split_to_temp_csvs(
                df, seed, args.train_frac, args.max_train, tmp_root, ds_name)
            n_train = sum(1 for _ in open(tr_csv)) - 1
            n_test = sum(1 for _ in open(te_csv)) - 1

            sweep_plan = []
            if args.include_axis:
                for d in axis_depths:
                    sweep_plan.append(("Axis Aligned", d))
            if args.include_oblique:
                for d in oblique_depths:
                    sweep_plan.append(("Oblique", d))

            for split, depth in sweep_plan:
                key = (ds_name, seed, split, depth)
                if key in seen:
                    skipped += 1
                    done += 1
                    continue
                tag = (f"{ds_name}_s{seed}_"
                       f"{split.replace(' ', '')}_d{depth}")
                t0 = time.perf_counter()
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        mdir = Path(tmp) / "model"
                        tsec, final_trees, train_log = base.train_one(
                            tr_csv, "class", split, depth, args.num_trees,
                            args.shrinkage, args.num_proj_exp,
                            args.max_num_proj, args.density,
                            args.num_threads, mdir, args.num_split_type,
                            "continuous", seed)
                        if tsec is None:
                            failed += 1
                            done += 1
                            continue
                        bench_rows, _ = base.bench_one(
                            mdir, te_csv, args.batch_size, args.num_runs)
                        if bench_rows is None:
                            failed += 1
                            done += 1
                            continue
                        acc, ll, _ = base.eval_one(mdir, te_csv)
                except Exception as e:
                    print(f"[ERR] {tag}: {e}", flush=True)
                    failed += 1
                    done += 1
                    continue

                bench_sorted = sorted(bench_rows, key=lambda r: r[0])
                best_us, best_name = bench_sorted[0]
                generic_us = next(
                    (us for us, n in bench_rows
                     if "Generic" in n and "slow" not in n.lower()),
                    None)
                slow_us = next(
                    (us for us, n in bench_rows if "slow" in n.lower()),
                    None)
                row = {
                    "dataset": ds_name,
                    "split_type": split,
                    "max_depth": depth,
                    "seed": seed,
                    "num_trees_requested": args.num_trees,
                    "num_trees_final": final_trees if final_trees else "",
                    "shrinkage": args.shrinkage,
                    "num_proj_exp": args.num_proj_exp,
                    "density": args.density,
                    "max_num_proj": args.max_num_proj,
                    "num_split_type": args.num_split_type,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_features": n_features,
                    "train_seconds": f"{tsec:.3f}",
                    "accuracy": acc if acc is not None else "",
                    "log_loss": ll if ll is not None else "",
                    "best_engine": best_name,
                    "best_us_per_ex": f"{best_us:.4f}",
                    "generic_us_per_ex": (f"{generic_us:.4f}"
                                          if generic_us else ""),
                    "slow_us_per_ex": (f"{slow_us:.4f}" if slow_us else ""),
                    "all_engines": ";".join(
                        f"{n}={us:.4f}" for us, n in bench_sorted),
                }
                w.writerow(row)
                out_fh.flush()
                done += 1
                wall = time.perf_counter() - t0
                if not args.quiet:
                    overall_elapsed = time.perf_counter() - overall_t0
                    eta_h = (plan_total - done) * (overall_elapsed
                                                    / max(done - skipped, 1)) / 3600
                    print(f"  [{done}/{plan_total}] {tag}: "
                          f"acc={acc} train={tsec:.1f}s best={best_us:.3f}µs "
                          f"({wall:.1f}s wall, ETA {eta_h:.1f}h)",
                          flush=True)

            for f in [tr_csv, te_csv]:
                try:
                    f.unlink()
                except OSError:
                    pass

    out_fh.close()
    shutil.rmtree(tmp_root, ignore_errors=True)
    elapsed_h = (time.perf_counter() - overall_t0) / 3600
    print(f"\n=== Done. {done}/{plan_total} configs ({skipped} skipped, "
          f"{failed} failed). Wall: {elapsed_h:.1f}h ===", flush=True)


if __name__ == "__main__":
    main()
