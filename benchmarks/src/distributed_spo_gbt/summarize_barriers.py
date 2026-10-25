"""Summarize a DGBT barriers.csv (written by the manager when DGBT_BARRIER_CSV is set).
Usage: python summarize_barriers.py barriers.csv [tree_training_s]

Columns per request round (manager clock unless noted):
  barrier_s        = stage begin -> last reply received
  manager_serial_s = stage begin -> first request sent (feature sampling, plan building, ...)
  fanout_s         = time to serialize + enqueue all requests
  worker_max/med/min_s = worker self-reported runtimes (worker clock, durations only)
  idle_s           = sum_w (worker_max - worker_w): worker-seconds spent waiting at the barrier
  comm_residual_s  = barrier - manager_serial - fanout - worker_max (queueing + transport + deserialization)
  tail_s           = last reply -> stage end (manager-side merge of the replies)
"""
import sys
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
df = pd.read_csv(sys.argv[1])
tree_s = float(sys.argv[2]) if len(sys.argv) > 2 else None
df["stage_r"] = df["stage"]
m = df.stage == "ShareSplits"
df.loc[m, "stage_r"] = "ShareSplits/r" + df.loc[m, "round"].astype(str)
cols = ["barrier_s", "tail_s", "manager_serial_s", "fanout_s", "worker_max_s", "idle_s", "comm_residual_s"]
g = df.groupby("stage_r")
tot = g[cols].sum()
tot.insert(0, "n", g.size())
tot.insert(1, "mean_ms", g["barrier_s"].mean() * 1000)
tot["MB_out"] = g["bytes_out"].sum() / 1e6
tot["MB_in"] = g["bytes_in"].sum() / 1e6
order = ["StartNewIter", "FindSplits", "EvaluateSplits", "ShareSplits/r1", "ShareSplits/r2", "EndIter"]
tot = tot.reindex([o for o in order if o in tot.index] + [i for i in tot.index if i not in order])
print("== per stage, summed over the run (s unless named) ==")
print(tot)
inside = (df.barrier_s + df.tail_s).sum()
d = df.sort_values("t_begin_s")
gap = (d.t_begin_s.shift(-1) - (d.t_begin_s + d.barrier_s + d.tail_s)).dropna()
span = d.t_begin_s.iloc[-1] + d.barrier_s.iloc[-1] + d.tail_s.iloc[-1] - d.t_begin_s.iloc[0]
print(f"\nidentity: sum(barrier+tail) {inside:.2f} s + gaps between stages {gap.sum():.2f} s (max gap {gap.max()*1000:.1f} ms)"
      f" = {inside + gap.sum():.2f} s; first-begin->last-end {span:.2f} s"
      + (f"; harness tree_training_s {tree_s:.2f} s" if tree_s else ""))
it = df[df.iter >= 0]
cap = (it.num_replies * it.worker_max_s).sum()
print(f"idle: {it.idle_s.sum():.2f} worker-s of {cap:.2f} worker-s capacity = {100 * it.idle_s.sum() / cap:.1f} % of worker time waiting at barriers")
share = it[[c for c in cols if c != "idle_s"]].sum() / (it.barrier_s + it.tail_s).sum() * 100  # idle is worker-s, reported above
print("share of wall time inside stages (%): " + ", ".join(f"{k[:-2]} {v:.1f}" for k, v in share.items() if k != "barrier_s"))
fs = df[df.stage == "FindSplits"].groupby("layer")
lay = fs[["barrier_s", "manager_serial_s", "fanout_s", "worker_max_s", "worker_med_s", "worker_min_s", "idle_s", "comm_residual_s", "tail_s"]].mean() * 1000
lay["skew_%"] = (fs.worker_max_s.mean() - fs.worker_med_s.mean()) / fs.worker_max_s.mean() * 100
print("\n== FindSplits by layer (mean per call, ms; skew = (max-median)/max) ==")
print(lay.round(2))
print("\nslowest worker in FindSplits (count):", df[df.stage == "FindSplits"].slowest_worker.value_counts().sort_index().to_dict())
