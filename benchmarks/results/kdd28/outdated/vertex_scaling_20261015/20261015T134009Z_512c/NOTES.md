# Vertex scaling run 20261015T134009Z, 512 columns

Produced by `benchmarks/src/distributed_spo_gbt/Distributed_SPO_GBT_scaling.ipynb` (COLS = 512, all other
constants unchanged), raw outputs in `gs://ariel_ydf_bucket/runs/scaling_20261015T134009Z_512c/<tag>/`.
17 of 18 cells; `results.csv` from the notebook, `scaling.png` from `finish_512c.py` (the notebook's plot cell
failed with KeyError 'spo' because the SPO 1-worker strong-scaling baseline is missing).

- **spo_n1600k_w1 missing.** Vertex job failed after 36 min, inside tree 1: "Replicas low on memory:
  workerpool1. Specify a machine with larger memory and try again." (n2-highmem-2, 16 GB). aa_n1600k_w1 on
  the same machine type completed. Not rerun.
- **Strong-scaling speedups are relative to 2 workers** for both arms (ideal = w/2), since spo has no
  1-worker point.
- **aa_n1600k_w16 row restored by hand.** Vertex re-provisioned that job at 15:03Z after its first run had
  finished; the second run resumed from the 50-tree checkpoint ("Resume training from iteration #50") and
  overwrote eval.json and manager.log with tree_training_s = 0.002. setup_s / tree_training_s /
  train_seconds = 412.362 / 2493.385 / 2906.33 are the first run's values, read from that eval.json at
  ~15:00Z before the overwrite; accuracy/AUC/loss are identical in both runs (same model). Every other cell's
  manager.log shows one clean, non-resumed run.
- **Queue order changed mid-run** (16:45Z) from largest-cluster-first to longest-expected-first (rows per
  worker); jobs already running were adopted, not resubmitted. aa_n400k_w4 was cancelled once at 16:43Z
  (12 min in) to free quota and rerun later; its result is from the rerun. Each cell runs on its own VMs,
  so ordering does not affect the timings.
