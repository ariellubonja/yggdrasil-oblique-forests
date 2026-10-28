#!/usr/bin/env python3
"""Train axis-aligned GBT on every CC18 fold with train_oblique_forest, save the
model, dump per-node split features with dump_forest_nodes. Stdlib only.
Usage: run_cc18_gbt.py [--folds 0-9] [--jobs 8] [--threads 6] [--only NAME,...]
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HOME = os.path.expanduser("~")
REPO = f"{HOME}/yggdrasil-oblique-forests"
DATA = f"{REPO}/benchmarks/data/cc18_binary_csv"
MANIFEST = f"{REPO}/benchmarks/data/cc18_manifest.json"
HARNESS = f"{REPO}/bazel-bin/examples/train_oblique_forest"
DUMP = f"{REPO}/bazel-bin/examples/dump_forest_nodes"
OUT = f"{HOME}/featstudy/runs"

GBT_ARGS = ["--ensemble_method", "Boosting", "--feature_split_type", "Axis Aligned",
            "--numerical_split_type", "Exact", "--num_candidate_attributes", "-1",
            "--num_trees", "300", "--tree_depth", "6", "--shrinkage", "0.1",
            "--growing_strategy", "Local", "--seed", "1"]


def jobs(folds, only):
    entries = json.load(open(MANIFEST))["entries"]
    for e in entries:
        if e["status"] != "ok":
            continue
        folder = f"task_{e['task_id']}_{e['folder_name']}"
        if only and not any(o in folder for o in only):
            continue
        d = os.path.join(DATA, folder)
        if not os.path.isdir(d):
            print(f"MISSING {d}", file=sys.stderr)
            continue
        for f in folds:
            yield folder, f, e["label_column"]


def run_one(folder, fold, label, threads):
    tag = f"{folder}__fold{fold}"
    odir = os.path.join(OUT, tag)
    os.makedirs(odir, exist_ok=True)
    nodes_csv = os.path.join(odir, "nodes.csv")
    if os.path.exists(nodes_csv) and os.path.getsize(nodes_csv) > 0:
        return tag, "cached", 0.0
    train = f"{DATA}/{folder}/repeat0_fold{fold}_sample0_train.csv"
    test = f"{DATA}/{folder}/repeat0_fold{fold}_sample0_test.csv"
    model_dir = os.path.join(odir, "model")
    cmd = [HARNESS, "--input_mode", "csv", "--train_csv", train, "--test_csv", test,
           "--label_col", label, "--num_threads", str(threads),
           "--model_out_dir", model_dir] + GBT_ARGS
    t0 = time.time()
    with open(os.path.join(odir, "train.log"), "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    dt = time.time() - t0
    if r.returncode != 0:
        return tag, f"TRAIN_FAIL rc={r.returncode}", dt
    with open(os.path.join(odir, "dump.log"), "w") as log:
        r = subprocess.run([DUMP, "--model", model_dir, "--output", nodes_csv],
                           stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return tag, f"DUMP_FAIL rc={r.returncode}", dt
    # Keep the model dir small: drop it after a successful dump.
    subprocess.run(["rm", "-rf", model_dir])
    return tag, "ok", dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    lo, _, hi = a.folds.partition("-")
    folds = list(range(int(lo), int(hi or lo) + 1))
    only = [o for o in a.only.split(",") if o]
    os.makedirs(OUT, exist_ok=True)
    todo = list(jobs(folds, only))
    print(f"{len(todo)} jobs, {a.jobs} parallel x {a.threads} threads", flush=True)
    status_path = os.path.join(OUT, "status.tsv")
    with ThreadPoolExecutor(a.jobs) as ex, open(status_path, "a") as st:
        futs = [ex.submit(run_one, f, k, l, a.threads) for f, k, l in todo]
        for fu in as_completed(futs):
            tag, status, dt = fu.result()
            line = f"{tag}\t{status}\t{dt:.1f}"
            print(line, flush=True)
            st.write(line + "\n"); st.flush()


if __name__ == "__main__":
    main()
