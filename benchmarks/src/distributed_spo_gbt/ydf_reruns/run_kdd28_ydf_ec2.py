#!/usr/bin/env python3
# KDD28 accuracy reruns, profile kdd28_ydf, on EC2 ariel-dist-accuracy-1 (ex ariel-ydf-1, i-0f90f7764082bf21a,
# m7i.metal-24xl), distributed = in-process MULTI_THREAD workers (accuracy only, never timing). 2026-11-03.
# Same-seed rerun on this box is bit-identical (tested 2026-11-03 on m7i only: 7 pairs, single and W=4/16, same nodes).
# 2026-11-04: box moved to c8i.metal-96xl (SMT + turbo on) for distacc_aaspo, distacc_exact300, colrep_distacc.
"""One resumable queue for tab:dist-accuracy (distacc), tab:column-replication (colrep, colrep_single; its r = 0
cells are the in-worker arm of tab:cross-network), its AA+SPO and arm-(c) columns (distacc_aaspo,
distacc_exact300), tab:column-replication on the distacc folds (colrep_distacc), tab:cc18-aa-spo arms (a), (d), (e) (cc18)
and the cross-network (column shipping) arm of tab:cross-network (cross_network).
Cells run concurrently under a CPU-slot budget, datasets outer and arms inner, heaviest datasets first.
Flags mirror run_table8_local.sh, run_replication_sweep.py, run_single_machine_cc18.sh, run_cc18_aa_spo_gbt.sh."""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "params"))
import params as kdd28  # noqa: E402
import evalsuite  # noqa: E402  (params/eval_suites.json: dataset/fold/hash/metric checks, warnings only)

PR = "kdd28_ydf"
SPLIT_AA = "--feature_split_type=Axis Aligned"
ROWS = {  # per-table CSV columns, as in the scripts each table mirrors (stamps appended)
    "distacc": "dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,train_s,wall_s,git",
    "colrep": "dataset,seed,replication,workers,threads,trees,max_depth,rc,test_acc,test_auc,test_logloss,"
              "num_conditions,num_oblique,num_multi_col,held_cols,train_s,wall_s,git,log",
    "colrep_single": "dataset,seed,threads,rc,test_acc,test_auc,test_logloss,train_s,wall_s,git",
    "distacc_aaspo": "dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,train_s,wall_s,git,bin_sha,machine",
    "distacc_exact300": "dataset,fold,arm,workers,split_axis,threads,rc,test_acc,test_auc,test_logloss,train_s,wall_s,git,bin_sha,machine",
    "colrep_distacc": "dataset,fold,replication,workers,threads,rc,test_acc,test_auc,test_logloss,"
                      "num_conditions,num_oblique,num_multi_col,held_cols,train_s,wall_s,git,log,bin_sha,machine",
    "cc18": "dataset,fold,arm,threads,rc,test_acc,test_auc,test_logloss,num_trees,train_s,wall_s,git,bin_sha",
    "cross_network": "dataset,seed,workers,threads,trees,max_depth,rc,test_acc,test_auc,test_logloss,num_conditions,"
                     "num_oblique,num_multi_col,held_cols,ship_bytes_in,ship_bytes_out,ship_cols_in,ship_s_max,"
                     "ship_s_sum,cross_worker_splits,train_s,wall_s,git,log",
}
KEYS = {"distacc": ("dataset", "fold", "arm"), "distacc_aaspo": ("dataset", "fold", "arm"),
        "distacc_exact300": ("dataset", "fold", "arm"), "colrep_distacc": ("dataset", "fold", "replication"), "colrep": ("dataset", "seed", "replication"),
        "colrep_single": ("dataset", "seed"), "cc18": ("dataset", "fold", "arm"),
        "cross_network": ("dataset", "seed")}


def seconds(d):
    """absl duration text ("551.19ms", "1m2.5s") -> seconds."""
    unit = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    parts = re.findall(r"([0-9.]+)(h|ms|us|µs|ns|m|s)", d or "")
    return round(sum(float(v) * unit[u] for v, u in parts), 3) if parts else ""


def parse_log(path, dist):
    txt = open(path, errors="replace").read()
    g = lambda pat: (re.findall(pat, txt, re.M) or [""])[-1]  # noqa: E731
    out = {}
    if dist:
        out["test_acc"], out["test_auc"] = g(r"^Accuracy: ([0-9.]+)"), g(r"^\s*auc: ([0-9.]+)")
        out["test_logloss"] = g(r"^LogLoss: *:? *([0-9.]+)")
        out["train_s"] = seconds(g(r"Training \(incl\. cache creation\) took ([0-9.a-zµ]+)"))
        c = re.search(r"Conditions: (\d+) total, (\d+) oblique, (\d+) with >1 column", txt)
        out["num_conditions"], out["num_oblique"], out["num_multi_col"] = c.groups() if c else ("", "", "")
        out["held_cols"] = ";".join(f"{o}+{r}" for o, r in re.findall(r"worker \d+: (\d+) owned \+ (\d+) replicated", txt))
        ship = re.findall(r"column-shipping:\{bytes-in:(\d+) bytes-out:(\d+) cols-in:(\d+) ship-s-max:(\S+) "
                          r"ship-s-sum:(\S+) cross-worker-splits:(\d+)\}", txt)
        if ship:  # cumulative counters: the last line is the run total
            out |= dict(zip(["ship_bytes_in", "ship_bytes_out", "ship_cols_in", "ship_s_max", "ship_s_sum",
                             "cross_worker_splits"], ship[-1]))
    else:
        m = re.findall(r"test-accuracy:(\S+) test-auc:(\S+) test-logloss:(\S+)", txt)
        out["test_acc"], out["test_auc"], out["test_logloss"] = m[-1] if m else ("", "", "")
        out["train_s"] = g(r"Training block took: ([0-9.e+-]+)")
        out["num_trees"] = g(r"model-num-trees:(\d+)") or g(r"Final model num-trees:(\d+)")  # first: harness >= 2026-11-04
    return out


class Cell:
    def __init__(self, table, key, profile, argv, threads, weight, cost, extra, dist, work=None):
        self.table, self.key, self.profile, self.argv = table, key, profile, argv
        self.threads, self.weight, self.cost, self.extra, self.dist, self.work = threads, weight, cost, extra, dist, work
        self.tag = "__".join(str(k) for k in key)


def build_cells(a):
    B = a.bin_dir
    single_bin, dist_bin = os.path.join(B, "train_oblique_forest"), os.path.join(B, "train_distributed_gbt")
    tables, cells = a.tables.split(","), []

    def single(table, key, prof, train, test, label, seed, split, model, ts, extra, cost, ydf=True):
        argv = [single_bin, "--input_mode", "csv", "--train_csv", train, "--test_csv", test, "--label_col", label,
                "--ensemble_method", "Boosting", *(["--ydf_defaults"] if ydf else []), *split, "--validation_set_ratio", "0",
                "--early_stopping", "NONE", "--num_threads", str(ts), "--seed", str(seed), "--model_out_dir", model]
        cells.append(Cell(table, key, prof, argv, ts, ts, cost, extra, False))

    def dist(table, key, prof, train, test, w, seed, aa, r, model, work, extra, cost, ship=False):
        p = kdd28.load_profile(prof)
        spo = [] if aa else ["--sparse_oblique_num_projections_exponent", kdd28._fmt(p["spo_exponent"]),
                             "--sparse_oblique_max_num_projections", kdd28._fmt(p["spo_max_projections"]),
                             "--sparse_oblique_projection_density_factor", kdd28._fmt(p["spo_density"]),
                             f"--sparse_oblique_local_projection_count={kdd28._fmt(p['local_projection_count'])}",
                             f"--sparse_oblique_split_budget={kdd28._fmt(p['split_budget'])}",
                             "--sparse_oblique_weights", "BINARY"] + ([] if r is None else ["--column_replication", str(r)]) \
            + (["--sparse_oblique_ship_columns"] if ship else [])
        argv = [dist_bin, "--alsologtostderr", "--train_csv", train, "--test_csv", test, "--label", "label",
                *(["--force_numerical_features"] if p["force_numerical"] else []), "--work_dir", work,
                "--num_workers", str(w), "--num_threads", str(a.worker_threads), "--num_trees", str(p["trees"]),
                "--max_depth", str(p["max_depth"]), "--split_axis", "AXIS_ALIGNED" if aa else "SPARSE_OBLIQUE", *spo,
                f"--max_unique_values_for_discretized_numerical={p['max_unique_discretized']}", "--seed", str(seed),
                "--worker_logs=false", "--model_out_dir", model]
        t = w * a.worker_threads
        cells.append(Cell(table, key, prof, argv, t, min(t, a.max_weight), cost, extra, True, work))

    def shape(train):  # (features, rows ~ bytes / 8 B per value); (0, 0) when unreadable
        try:
            with open(train) as f:
                nf = len(f.readline().split(",")) - 1
            return nf, os.path.getsize(train) // max(1, 8 * (nf + 1))
        except OSError:
            return 0, 0

    def cost(train, aa):
        nf, n = shape(train)
        return n * (nf if aa else min(nf * nf, 6000))

    def tab_folds():  # (dataset, fold, train, test) of tab:dist-accuracy, heaviest dataset first
        ds = sorted(os.listdir(a.tab_folds))
        ds.sort(key=lambda d: -cost(os.path.join(a.tab_folds, d, "fold0_train.csv"), False))
        for d in ds:
            for k in range(len(json.load(open(os.path.join(a.tab_folds, d, "folds.json")))["indices"])):
                yield (d, k, *(os.path.join(a.tab_folds, d, f"fold{k}_{s}.csv") for s in ("train", "test")))

    if "distacc" in tables:
        ds = sorted(os.listdir(a.tab_folds))
        ds.sort(key=lambda d: -cost(os.path.join(a.tab_folds, d, "fold0_train.csv"), False))
        dp = kdd28.load_profile(f"{PR}.distacc_spo")
        for d in ds:
            nf = len(json.load(open(os.path.join(a.tab_folds, d, "folds.json")))["indices"])
            for k in range(nf):
                tr, te = (os.path.join(a.tab_folds, d, f"fold{k}_{s}.csv") for s in ("train", "test"))
                o = os.path.join(a.out, "distacc")
                for arm in ["single:SPO", "single:AA"] + [f"{w}:{x}" for x in ("SPO", "AA") for w in dp["workers"]]:
                    W, A = arm.split(":")
                    key, tag = (d, k, arm), f"{d}__f{k}__{W}_{A}"
                    model, c = os.path.join(o, "models", tag), cost(tr, A == "AA")
                    if W == "single":
                        prof = f"{PR}.distacc_single_{A.lower()}"
                        split = ["--feature_split_type=Oblique"] if A == "SPO" else [SPLIT_AA]
                        single("distacc", key, prof, tr, te, "label", 1, split, model, a.single_threads,
                               {"workers": W, "split_axis": A}, c)
                    else:
                        dist("distacc", key, f"{PR}.distacc_{A.lower()}", tr, te, int(W), 1, A == "AA", None, model,
                             os.path.join(o, "work", tag), {"workers": W, "split_axis": "AXIS_ALIGNED" if A == "AA"
                                                            else "SPARSE_OBLIQUE"}, c)
    if "distacc_aaspo" in tables:  # single-machine AA+SPO column of tab:dist-accuracy, same folds (2026-11-04)
        ds = sorted(os.listdir(a.tab_folds))
        ds.sort(key=lambda d: -cost(os.path.join(a.tab_folds, d, "fold0_train.csv"), False))
        for d in ds:
            nf = len(json.load(open(os.path.join(a.tab_folds, d, "folds.json")))["indices"])
            for k in range(nf):
                tr, te = (os.path.join(a.tab_folds, d, f"fold{k}_{s}.csv") for s in ("train", "test"))
                single("distacc_aaspo", (d, k, "single:AA+SPO"), f"{PR}.distacc_single_aa_spo", tr, te, "label", 1,
                       ["--feature_split_type=Oblique", "--oblique_also_axis_aligned"],
                       os.path.join(a.out, "distacc_aaspo", "models", f"{d}__f{k}__single_AA+SPO"), a.single_threads,
                       {"workers": "single", "split_axis": "AA+SPO"}, cost(tr, False))
    if "distacc_exact300" in tables:  # arm (c) of tab:cc18-aa-spo (run_cc18_aa_spo_gbt.sh:24 flags), same folds
        cp = kdd28.load_profile(f"{PR}.distacc_single_spo_harness_exact300")
        split = ["--feature_split_type=Oblique", "--num_trees", str(cp["trees"]), "--numerical_split_type=Exact",
                 "--num_projections_exponent", kdd28._fmt(cp["spo_exponent"]), "--max_num_projections",
                 kdd28._fmt(cp["spo_max_projections"]), "--projection_density_factor", kdd28._fmt(cp["spo_density"])]
        for d, k, tr, te in tab_folds():
            single("distacc_exact300", (d, k, "single:SPO-harness-exact300"), f"{PR}.distacc_single_spo_harness_exact300",
                   tr, te, "label", 1, split, os.path.join(a.out, "distacc_exact300", "models", f"{d}__f{k}__single_SPO_h300"),
                   a.single_threads, {"workers": "single", "split_axis": "SPARSE_OBLIQUE"}, cost(tr, False), ydf=False)
    if "colrep_distacc" in tables:  # tab:column-replication on the distacc datasets/folds, r = 0 only for --colrep_r0
        cp = kdd28.load_profile(f"{PR}.colrep_distacc_spo")
        r0 = {tuple(x.rsplit(":", 1)) for x in a.colrep_r0.split(",") if x}
        o = os.path.join(a.out, "colrep_distacc")
        for d, k, tr, te in tab_folds():
            for r in cp["column_replication"]:
                if r == 0 and (d, str(k)) not in r0:
                    continue
                tag = f"{d}__f{k}__r{r:.2f}"
                dist("colrep_distacc", (d, k, float(r)), f"{PR}.colrep_distacc_spo", tr, te, cp["workers"][0], 1, False, r,
                     os.path.join(o, "models", tag), os.path.join(o, "work", tag),
                     {"workers": cp["workers"][0]}, cost(tr, False))
    if "colrep" in tables:
        cp = kdd28.load_profile(f"{PR}.colrep_spo")
        sp = kdd28.load_profile(f"{PR}.colrep_single_spo")
        ds = sorted(d for d in os.listdir(a.cc18_fold0) if os.path.isdir(os.path.join(a.cc18_fold0, d)))
        ds.sort(key=lambda d: -cost(os.path.join(a.cc18_fold0, d, "train.csv"), False))
        for d in ds:
            tr, te = os.path.join(a.cc18_fold0, d, "train.csv"), os.path.join(a.cc18_fold0, d, "test.csv")
            for s in cp["seeds"]:
                for r in cp["column_replication"]:
                    o, tag = os.path.join(a.out, "colrep"), f"{d}__s{s}__r{r:.2f}"
                    dist("colrep", (d, s, float(r)), f"{PR}.colrep_spo", tr, te, cp["workers"][0], s, False, r,
                         os.path.join(o, "models", tag), os.path.join(o, "work", tag),
                         {"workers": cp["workers"][0], "trees": cp["trees"], "max_depth": cp["max_depth"]}, cost(tr, False))
            for s in sp["seeds"]:
                o = os.path.join(a.out, "colrep_single")
                single("colrep_single", (d, s), f"{PR}.colrep_single_spo", tr, te, "label", s,
                       ["--feature_split_type=Oblique"], os.path.join(o, "models", f"{d}__s{s}"), a.single_threads,
                       {}, cost(tr, False))
    if "cross_network" in tables:
        xp = kdd28.load_profile(f"{PR}.cross_network_ship")
        assert xp["column_shipping"] is True and xp["column_replication"] == 0
        ds = sorted(d for d in os.listdir(a.cc18_fold0) if os.path.isdir(os.path.join(a.cc18_fold0, d)))
        ds.sort(key=lambda d: -cost(os.path.join(a.cc18_fold0, d, "train.csv"), False))
        for d in ds:
            tr, te = os.path.join(a.cc18_fold0, d, "train.csv"), os.path.join(a.cc18_fold0, d, "test.csv")
            for s in xp["seeds"]:
                o, tag = os.path.join(a.out, "cross_network"), f"{d}__s{s}__ship"
                dist("cross_network", (d, s), f"{PR}.cross_network_ship", tr, te, xp["workers"][0], s, False, None,
                     os.path.join(o, "models", tag), os.path.join(o, "work", tag),
                     {"workers": xp["workers"][0], "trees": xp["trees"], "max_depth": xp["max_depth"]}, cost(tr, False), ship=True)
    if "cc18" in tables:
        ds = sorted(d for d in os.listdir(a.cc18_10fold) if os.path.isdir(os.path.join(a.cc18_10fold, d)))
        ds.sort(key=lambda d: -cost(os.path.join(a.cc18_10fold, d, "fold0_train.csv"), False))
        arms = {"cc18_aa": [SPLIT_AA], "cc18_spo": ["--feature_split_type=Oblique"],
                "cc18_aa_spo": ["--feature_split_type=Oblique", "--oblique_also_axis_aligned"]}
        for d in ds:
            lbl = json.load(open(os.path.join(a.cc18_10fold, d, "meta.json")))["label_col"]
            for k in kdd28.load_profile(f"{PR}.cc18_aa")["folds"]:
                tr, te = (os.path.join(a.cc18_10fold, d, f"fold{k}_{s}.csv") for s in ("train", "test"))
                for arm, split in arms.items():
                    single("cc18", (d, k, arm), f"{PR}.{arm}", tr, te, lbl, 1, split,
                           os.path.join(a.out, "cc18", "models", f"{d}__f{k}__{arm}"), a.single_threads, {},
                           cost(tr, arm == "cc18_aa"))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin_dir", required=True, help="dir with train_oblique_forest and train_distributed_gbt")
    ap.add_argument("--tab_folds", default="", help="TabArena/TabReD folds (<ds>/fold<k>_{train,test}.csv)")
    ap.add_argument("--cc18_fold0", default="", help="prep_cc18_fold0.py output (task_*/train.csv)")
    ap.add_argument("--cc18_10fold", default="", help="run_cc18_aa_spo_gbt.sh data/ (<ds>/fold<k>_*.csv, meta.json)")
    ap.add_argument("--tables", default="distacc,colrep,cc18")
    ap.add_argument("--cpus", type=int, default=os.cpu_count())
    ap.add_argument("--single_threads", type=int, default=4, help="train_oblique_forest --num_threads (model depends on it)")
    ap.add_argument("--worker_threads", type=int, default=2, help="threads per in-process worker (model depends on it)")
    ap.add_argument("--max_weight", type=int, default=16, help="CPU slots one cell may claim at most")
    ap.add_argument("--git", default="")
    ap.add_argument("--machine", default="", help="machine note for the 'machine' column (2026-11-04 tables)")
    ap.add_argument("--colrep_r0", default="", help="colrep_distacc: dataset:fold,... that also run r = 0 (identity check)")
    ap.add_argument("--only", default="", help="comma list of dataset substrings")
    a = ap.parse_args()
    need = {"distacc": "tab_folds", "distacc_aaspo": "tab_folds", "distacc_exact300": "tab_folds",
            "colrep_distacc": "tab_folds", "colrep": "cc18_fold0", "cross_network": "cc18_fold0", "cc18": "cc18_10fold"}
    for t in a.tables.split(","):
        if t not in need:
            ap.error(f"unknown table {t!r} (known: {', '.join(need)})")
        if not getattr(a, need[t]):
            ap.error(f"--tables {t} needs --{need[t]}")
    cells = build_cells(a)
    if a.only:
        cells = [c for c in cells if any(o in c.key[0] for o in a.only.split(","))]
    bin_sha = ""
    try:
        bin_sha = subprocess.run(["sha256sum", os.path.join(a.bin_dir, "train_oblique_forest")], capture_output=True,
                                 text=True).stdout[:12]
    except OSError:
        pass
    lock, done, csvs, eval_cols, stamp_w = threading.Lock(), {}, {}, {}, {}
    for t in {c.table for c in cells}:
        os.makedirs(os.path.join(a.out, t, "logs"), exist_ok=True)
        path = csvs[t] = os.path.join(a.out, t, "results.csv")
        if subprocess.run([sys.executable, kdd28.__file__, "require-stamped", path, "--allow-v1"]).returncode:
            sys.exit(2)
        cols = ROWS[t].split(",")
        if os.path.exists(path):
            with open(path) as f:  # CSVs started before oblique_also_axis_aligned keep their (shorter) stamp
                stamp_w[t] = kdd28.stamp_width(f.readline().strip().split(","))
            for r in csv.DictReader(open(path)):
                if r["rc"] == "0":
                    done[(t,) + tuple(r[k] for k in KEYS[t])] = 1
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(cols + kdd28.stamp_columns(cols) + evalsuite.COLUMNS)
            stamp_w[t] = len(kdd28.STAMP)
        with open(path) as f:  # CSVs started before 2026-11-04 keep their header, rows stay eval-unstamped
            eval_cols[t] = f.readline().strip().split(",")[-len(evalsuite.COLUMNS):] == evalsuite.COLUMNS
        if not eval_cols[t]:
            evalsuite.warn(f"{path}: no eval-suite columns (recorded before 2026-11-04), new rows unstamped eval")
    todo = [c for c in cells if (c.table,) + tuple(str(k) for k in c.key) not in done]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} done, {len(todo)} to run on {a.cpus} CPU slots", flush=True)
    pre = set()
    for c in todo:  # pre-flight per profile: first cell's argv + --print_params_only, exit 2 on drift
        if c.profile not in pre:
            pre.add(c.profile)
            kdd28.preflight(c.argv, c.profile, os.path.join(a.out, c.table, "logs", f"preflight_{c.profile}.log"))
    ev = evalsuite.driver_preflight([(c.table, dict(zip(KEYS[c.table], c.key)), c.argv) for c in cells],
                                    os.path.join(a.out, "eval_sha_cache.json"))
    left = {t: sum(1 for c in todo if c.table == t) for t in csvs}
    fails = {t: 0 for t in csvs}
    cv, state = threading.Condition(), {"free": a.cpus, "stop": False, "running": 0}

    def run(c):
        log = os.path.join(a.out, c.table, "logs", c.tag + ".log")
        if c.work:
            shutil.rmtree(c.work, ignore_errors=True)
        t0, rc = time.time(), -1
        try:
            with open(log, "w") as lf:
                lf.write("# " + " ".join(c.argv) + "\n")
                lf.flush()
                rc = subprocess.run(c.argv, stdout=lf, stderr=subprocess.STDOUT).returncode
            wall = round(time.time() - t0, 2)
            m = parse_log(log, c.dist)
            if c.work:
                shutil.rmtree(c.work, ignore_errors=True)
            drift = kdd28.check_cell(c.profile, log)  # exit 2 on drift (ALLOW_DRIFT=1 -> "!drift")
            v = dict(zip(KEYS[c.table], c.key)) | c.extra | m | {"rc": rc, "wall_s": wall, "git": a.git,
                                                                  "bin_sha": bin_sha, "threads": c.threads,
                                                                  "machine": a.machine, "log": c.tag + ".log"}
            row = [v.get(k, "") for k in ROWS[c.table].split(",")] + kdd28.stamp_values(c.profile + drift, log)[:stamp_w[c.table]]
            if eval_cols[c.table]:
                row += evalsuite.stamp(ev[(c.table,) + tuple(str(k) for k in c.key)], m)
            with lock:
                with open(csvs[c.table], "a", newline="") as f:
                    csv.writer(f).writerow(row)
                left[c.table] -= 1
                fails[c.table] += rc != 0
                print(f"{time.strftime('%H:%M:%S')} {c.table} {c.tag} rc={rc} auc={m.get('test_auc')} {wall}s", flush=True)
                if left[c.table] == 0:
                    with open(os.path.join(a.out, c.table, "DONE"), "w") as f:
                        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} failed={fails[c.table]}\n")
        except SystemExit:
            state["stop"] = True
        finally:
            with cv:
                state["free"] += c.weight
                state["running"] -= 1
                cv.notify_all()

    group = {}  # one dataset's cells stay together, heaviest dataset (any table) first
    for c in todo:
        group[(c.table, c.key[0])] = max(group.get((c.table, c.key[0]), 0), c.cost)
    pending = sorted(todo, key=lambda c: -group[(c.table, c.key[0])])  # strict FIFO: wide cells never starve
    for c in pending:
        c.weight = min(c.weight, a.cpus)
    with cv:
        while pending and not state["stop"]:
            if pending[0].weight > state["free"]:
                cv.wait()
                continue
            fit = pending.pop(0)
            state["free"] -= fit.weight
            state["running"] += 1
            threading.Thread(target=run, args=(fit,), daemon=True).start()
        while state["running"]:
            cv.wait()
    if state["stop"]:
        print("STOP: a cell drifted from its profile (see DRIFT lines)", file=sys.stderr)
        sys.exit(2)
    for t in csvs:  # tables with nothing left to run still get their marker
        if not os.path.exists(os.path.join(a.out, t, "DONE")) and left[t] == 0:
            with open(os.path.join(a.out, t, "DONE"), "w") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} failed={fails[t]}\n")
    if set(a.tables.split(",")) <= set(csvs) and not a.only:
        with open(os.path.join(a.out, "ALL_DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()) + json.dumps(fails) + "\n")
    print("### DONE", json.dumps(fails), flush=True)


if __name__ == "__main__":
    main()
