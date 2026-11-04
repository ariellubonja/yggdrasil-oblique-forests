#!/usr/bin/env python3
"""Train axis-aligned GBT on every CC18 fold with train_oblique_forest, save the
model, dump per-node split features with dump_forest_nodes. Stdlib only.
Usage: run_cc18_gbt.py [--folds 0-9] [--jobs 8] [--threads 6] [--only NAME,...]
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distributed_spo_gbt", "params"))
import params as kdd28  # noqa: E402  (kdd28_params.json)

PROFILE = "cc18_feature_study"
P = kdd28.load_profile(PROFILE)
HOME = os.path.expanduser("~")
REPO = f"{HOME}/yggdrasil-oblique-forests"
DATA = f"{REPO}/benchmarks/data/cc18_binary_csv"
MANIFEST = f"{REPO}/benchmarks/data/cc18_manifest.json"
HARNESS = f"{REPO}/bazel-bin/examples/train_oblique_forest"
DUMP = f"{REPO}/bazel-bin/examples/dump_forest_nodes"
OUT = f"{HOME}/featstudy/runs"

GBT_ARGS = ["--ensemble_method", "Boosting", "--feature_split_type", "Axis Aligned",
            "--numerical_split_type", kdd28.SPLIT_FLAG[P["split_type"]], "--num_candidate_attributes", str(P["num_candidate_attributes"]),
            "--num_trees", str(P["trees"]), "--tree_depth", str(P["max_depth"]), "--shrinkage", str(P["shrinkage"]),
            "--growing_strategy", P["growing_strategy"], "--seed", str(P["seeds"][0])]


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


def cell_cmd(folder, fold, label, threads, model_dir):
    train = f"{DATA}/{folder}/repeat0_fold{fold}_sample0_train.csv"
    test = f"{DATA}/{folder}/repeat0_fold{fold}_sample0_test.csv"
    return [HARNESS, "--input_mode", "csv", "--train_csv", train, "--test_csv", test,
            "--label_col", label, "--num_threads", str(threads),
            "--model_out_dir", model_dir] + GBT_ARGS


def run_one(folder, fold, label, threads):
    tag = f"{folder}__fold{fold}"
    odir = os.path.join(OUT, tag)
    os.makedirs(odir, exist_ok=True)
    nodes_csv = os.path.join(odir, "nodes.csv")
    if os.path.exists(nodes_csv) and os.path.getsize(nodes_csv) > 0:
        return tag, "cached", 0.0
    model_dir = os.path.join(odir, "model")
    cmd = cell_cmd(folder, fold, label, threads, model_dir)
    t0 = time.time()
    with open(os.path.join(odir, "train.log"), "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    dt = time.time() - t0
    drift = kdd28.check_cell(PROFILE, os.path.join(odir, "train.log"))  # exit 2 on drift; ALLOW_DRIFT=1 -> "!drift"
    with open(os.path.join(odir, "stamp.csv"), "w") as f:  # effective parameters of this model (new file)
        f.write(",".join(kdd28.STAMP) + "\n" + ",".join(kdd28.stamp_values(PROFILE + drift, os.path.join(odir, "train.log"))) + "\n")
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
    ap.add_argument("--folds", default=f"{P['folds'][0]}-{P['folds'][-1]}")
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
    if todo:  # pre-flight: the first cell's argv + --print_params_only, exit 2 on drift
        f0, k0, l0 = todo[0]
        kdd28.preflight(cell_cmd(f0, k0, l0, a.threads, os.path.join(OUT, "preflight_model")), PROFILE,
                        os.path.join(OUT, "preflight.log"))
    status_path = os.path.join(OUT, "status.tsv")
    with ThreadPoolExecutor(a.jobs) as ex, open(status_path, "a") as st:
        futs = [ex.submit(run_one, f, k, l, a.threads) for f, k, l in todo]
        try:
            for fu in as_completed(futs):
                tag, status, dt = fu.result()  # re-raises a cell's drift stop (SystemExit 2)
                line = f"{tag}\t{status}\t{dt:.1f}"
                print(line, flush=True)
                st.write(line + "\n"); st.flush()
        except SystemExit:
            ex.shutdown(wait=True, cancel_futures=True)
            raise


if __name__ == "__main__":
    main()
