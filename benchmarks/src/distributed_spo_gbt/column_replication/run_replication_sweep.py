#!/usr/bin/env python3
"""Column-replication sweep of the in-worker distributed SPO-GBT on CC18 fold 0.

Grid: datasets (outer) x seeds x replication levels; one train_distributed_gbt process per
cell (in-process MULTI_THREAD workers = smoke-test driver, so the ACCURACY is the result and
the times are not distributed timings). Appends one CSV row per cell to <out>/results.csv and
keeps every log under <out>/logs/; finished cells are skipped on re-run (resume).
Usage: run_replication_sweep.py --bin BIN --data DATA_DIR --out OUT [--jobs 2] [--threads 2]
       [--workers 4] [--trees 20] [--seeds 1,2,3,4,5] [--levels 0,0.1,...,1] [--only a,b]
"""
import argparse, csv, os, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

FIELDS = ["dataset", "seed", "replication", "workers", "threads", "trees", "max_depth", "rc",
          "test_acc", "test_auc", "test_logloss", "num_conditions", "num_oblique", "num_multi_col",
          "held_cols", "train_s", "wall_s", "git", "log"]
LOCK = threading.Lock()


def seconds(d):
    """absl duration text ("551.19ms", "1.11s", "1m2.5s", "1h2m3s") -> seconds."""
    if not d:
        return ""
    unit_s = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "\u00b5s": 1e-6, "ns": 1e-9}
    parts = re.findall(r"([0-9.]+)(h|ms|us|\u00b5s|ns|m|s)", d)
    return round(sum(float(v) * unit_s[u] for v, u in parts), 3) if parts else ""


def parse(log_path):
    txt = open(log_path, errors="replace").read()
    g = lambda pat: (re.search(pat, txt, re.M) or [None, None])[1]  # noqa: E731
    acc = g(r"^Accuracy: ([0-9.]+)")
    aucs = re.findall(r"^\s*auc: ([0-9.]+)", txt, re.M)
    auc = aucs[-1] if aucs else None  # last per-class line = class "1" (positive) vs rest
    ll = g(r"^LogLoss: *:? *([0-9.]+)")
    cond = re.search(r"Conditions: (\d+) total, (\d+) oblique, (\d+) with >1 column", txt)
    train_s = seconds(g(r"Training \(incl\. cache creation\) took ([0-9.a-zµ]+)"))
    held = ";".join(f"{o}+{r}" for o, r in re.findall(r"worker \d+: (\d+) owned \+ (\d+) replicated", txt))
    return acc, auc, ll, cond.groups() if cond else ("", "", ""), train_s, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--data", required=True, help="prep_cc18_fold0.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--levels", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--only", default="")
    ap.add_argument("--git", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    levels = [float(x) for x in a.levels.split(",")]
    only = [o for o in a.only.split(",") if o]
    datasets = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    if only:
        datasets = [d for d in datasets if any(o in d for o in only)]
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "work"), exist_ok=True)
    res = os.path.join(a.out, "results.csv")
    done = set()
    if os.path.exists(res):
        for r in csv.DictReader(open(res)):
            if r["rc"] == "0":
                done.add((r["dataset"], int(r["seed"]), float(r["replication"])))
    else:
        with open(res, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    cells = [(d, s, r) for d in datasets for s in seeds for r in levels if (d, s, r) not in done]
    print(f"{len(cells)} cells to run ({len(done)} done), {a.jobs} parallel x W={a.workers} x T={a.threads}",
          flush=True)

    def run(cell):
        d, s, r = cell
        tag = f"{d}__s{s}__r{r:.2f}"
        log = os.path.join(a.out, "logs", tag + ".log")
        work = os.path.join(a.out, "work", tag)
        shutil.rmtree(work, ignore_errors=True)
        cmd = [a.bin, "--alsologtostderr",
               "--train_csv", os.path.join(a.data, d, "train.csv"),
               "--test_csv", os.path.join(a.data, d, "test.csv"),
               "--label", "label", "--force_numerical_features",
               "--work_dir", work, "--num_workers", str(a.workers), "--num_threads", str(a.threads),
               "--num_trees", str(a.trees), "--max_depth", str(a.max_depth),
               "--split_axis", "SPARSE_OBLIQUE",
               "--sparse_oblique_num_projections_exponent", "1",
               "--sparse_oblique_max_num_projections", "2147483647",
               "--sparse_oblique_local_projection_count=false",
               "--sparse_oblique_projection_density_factor", "1.5",
               "--sparse_oblique_weights", "BINARY",
               "--max_unique_values_for_discretized_numerical=1",
               "--column_replication", str(r), "--seed", str(s), "--worker_logs=false"]
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write("# " + " ".join(cmd) + "\n")
            lf.flush()
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        wall = round(time.time() - t0, 2)
        acc, auc, ll, (nc, no, nm), train_s, held = parse(log)
        shutil.rmtree(work, ignore_errors=True)
        row = [d, s, r, a.workers, a.threads, a.trees, a.max_depth, p.returncode, acc, auc, ll, nc, no, nm,
               held, train_s, wall, a.git, os.path.basename(log)]
        with LOCK:
            with open(res, "a", newline="") as f:
                csv.writer(f).writerow(row)
        return tag, p.returncode, acc, wall

    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run, c) for c in cells]
        for i, fu in enumerate(as_completed(futs), 1):
            tag, rc, acc, wall = fu.result()
            print(f"[{i}/{len(cells)}] {tag} rc={rc} acc={acc} {wall}s", flush=True)


if __name__ == "__main__":
    main()
