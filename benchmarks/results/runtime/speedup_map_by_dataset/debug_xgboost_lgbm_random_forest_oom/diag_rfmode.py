"""Why are XGBoost/LightGBM RF modes so slow? GiveMeSomeCredit fold0 (120k x 10), protocol ctors,
thread sweep + leaf counts + per-tree time + peak RSS."""
import sys, time, resource, numpy as np, pandas as pd
sys.path.insert(0, "/home/ubuntu/yggdrasil-oblique-forests/benchmarks/src/spo_vs_gbt")
import arms
F = "/home/ubuntu/spo_vs_gbt/folds/GiveMeSomeCredit/fold0_train.csv"
df = pd.read_csv(F, dtype="float32"); y = df.pop("label").astype(int).to_numpy(); X = df.to_numpy(np.float32)
print("PY_HIST_BINS", arms.PY_HIST_BINS); print("rows,feats", X.shape, "pos rate", y.mean(), flush=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
for lib, ctor in (("lightgbm_rf", arms._lgbm_rf_ctor), ("xgboost_rf", arms._xgb_rf_ctor)):
    for thr, ntrees in ((48, 240), (8, 240), (1, 240), (48, 10)):
        if lib == "lightgbm_rf" and (thr, ntrees) in ((48, 240), (8, 240)):
            continue  # done in the first launch: 196.6 s / 104.2 s, 7604 leaves/tree
        m = ctor(thr, X.shape[1], 1); m.set_params(n_estimators=ntrees)
        t0 = time.perf_counter(); m.fit(X, y); dt = time.perf_counter() - t0
        if lib == "lightgbm_rf":
            leaves = [t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]]
        else:
            leaves = [d.count("leaf=") for d in m.get_booster().get_dump()]
        print(f"{lib} threads={thr} trees={ntrees}: fit {dt:.1f}s = {dt/ntrees*1000:.0f} ms/tree; "
              f"leaves/tree mean {np.mean(leaves):.0f} (min {min(leaves)}, max {max(leaves)}); peak RSS {rss():.2f} GB", flush=True)
# YDF reference for the same fold: 240 trees, 48 threads, from suite_results.csv
s = pd.read_csv("/home/ubuntu/spo_vs_gbt/results/suite_results.csv")
s = s[(s.dataset == "GiveMeSomeCredit") & (s.fold == 0) & (s.rep == 0)]
print(s[s.method.isin(["spo_rf_exact_hwy", "aa_rf_exact", "lightgbm_rf", "xgboost_rf"])][["method", "train_s"]].to_string(index=False))
