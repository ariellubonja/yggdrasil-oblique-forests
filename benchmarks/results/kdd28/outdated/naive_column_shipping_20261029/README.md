# Naive column shipping vs disjoint-column distributed SPO-GBT (2026-10-28)

Communication-cost baseline for the paper's feature-distributed SPO-GBT. Branch
`naive-column-shipping`.

## The two arms

Both arms run YDF's distributed GBT with W workers owning disjoint column sets and the
in-worker sparse-oblique finder (`learner/distributed_decision_tree/oblique.cc`: Highway
VQSort exact scan, (node, projection-chunk) jobs on the worker's thread pool).

- **disjoint** (the paper's method): each worker samples its F/W projections per node from
  its *own* columns only. No worker-to-worker traffic beyond the usual split bitmaps.
- **ship** (`in_worker_oblique.ship_foreign_columns`, driver flag
  `--sparse_oblique_ship_columns`): each worker samples its F/W projections per node from
  *all* F numerical columns, then fetches every column it touches but does not own from the
  owning worker with a worker-to-worker `GetColumns` request (same mechanism as YDF's
  `ShareSplits` → `GetSplitValue`). Shipped columns live only for the current layer
  (FindSplits + EvaluateSplits) and are dropped before the next one; caching them would turn
  feature distribution into replication. The owner of a split's first attribute evaluates
  it and ships the attributes it lacks. Same projection budget, density (1.5), weights
  (binary) and finder as the disjoint arm; only the sampling pool and the traffic differ.

Per layer, a worker receives ≈ (1 − 1/W)·F·N·4 bytes (nearly every foreign column once the
layer has a few nodes), i.e. ≈ (W − 1) dataset copies per layer across the cluster.

## What is measured

- `results_<mode>.csv` (one row per dataset × arm): `tree_training_s` (manager log
  timestamps "Start tree training" → "Training ... took", excludes dataset-cache creation),
  `wall_s`, test accuracy / AUC / logloss, and the shipping totals reported by the workers:
  `ship_bytes_in` (column bytes received), `ship_cols_in`, `ship_s_max` (sum over rounds of
  the slowest worker's shipping wait = time the training was blocked on shipping),
  `ship_s_sum`, `cross_worker_splits` (winning oblique splits with a column from another
  worker).
- One barrier CSV per cell (`DGBT_BARRIER_CSV`, columns appended: `ship_bytes_in,
  ship_bytes_out, ship_cols_in, ship_s_max, ship_s_sum, ship_cross_splits`) under
  `ariel_notes/results/20261028_naive_colship/logs/`. `ship_bytes_in` is exact per row;
  `ship_bytes_out` is what a worker served since its previous reply, so bytes served while
  idle land on a later row. Σin = Σout held in every run so far, but it is not guaranteed:
  a column served after the owner's last FindSplits/EvaluateSplits reply of the run is
  never reported. `ship_bytes_in` is the exact total.
- Projection budget: both arms give worker w P_w = (its owned numerical count)^exponent; this
  equals the disjoint arm's per-node basis only when every feature is a candidate (GBT
  default, the paper setting), not under `num_candidate_attributes` sampling.
- `mode=inproc`: workers are threads of one process (MULTI_THREAD): shipping is an in-memory
  proto copy. `mode=grpc`: W `grpc_worker_main` processes on 127.0.0.1 (loopback TCP:
  real serialization + sockets, no network latency/bandwidth limit). Neither is a Vertex AI
  run; per CLAUDE.md these are box measurements, the byte volume is the transferable number.

## How to run

```bash
python3 benchmarks/data/download_cc18_datasets.py      # OpenML -> cc18_binary_csv/
python3 benchmarks/data/prep_cc18_fold0.py             # -> cc18_fold0_numeric/<ds>/fold0_{train,test}.csv
bazel build -c opt "--per_file_copt=.*boringssl.*@-O0" //examples:train_distributed_gbt \
  //yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker_main
MODE=inproc W=4 THREADS=2 TREES=300 bash benchmarks/src/distributed_spo_gbt/run_cc18_colship.sh
MODE=grpc   W=4 THREADS=2 TREES=300 bash benchmarks/src/distributed_spo_gbt/run_cc18_colship.sh
```

The boringssl `-O0` is needed on the 30 GB box: icx at -O2 on `bcm.cc` grows past 28 GB and
is OOM-killed. CC18 fold 0 of each task (OpenML's own 10-fold split), categoricals
ordinal-coded, NaNs train-mean-imputed, label majority → 0 (`prep_cc18_fold0.py` meta.json
per dataset lists what changed).
