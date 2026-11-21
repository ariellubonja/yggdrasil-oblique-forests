# runs/2026-11-21_where_spo_beats_aa_cc18_reg_distributed

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `<profile>/seed<s>/{cc18_dist,reg_dist}/results.csv`, s = 1-5, as the driver wrote them, all rc = 0.
  `cc18_dist`: 11 CC18 sets x 10 folds x W = 1/4/8/16/32 (550 cells per profile and seed). `reg_dist`: 6 regression sets x 5
  folds x W = 1/4/8/16/32 (150 cells per profile and seed). 7,000 cells. Rows = the kdd28_hao `pass_rec` CC18 and regression
  rows of tab:accuracy-where-spo-outperforms-aa
- **Column replication (added 2026-11-21 19:05Z-19:15Z):** `kdd28_hao/seed<s>/{colrep_cc18,colrep_reg}/results.csv`,
  profile `kdd28_hao.colrep_distacc_spo` (W = 4, r = 0.1..1.0), 1,100 + 300 cells per seed, all rc = 0. r = 0 = the W = 4
  `cc18_dist` / `reg_dist` cells. Machines: ariel-ydf-1 (CC18 seeds 1-3), ariel-ydf-7 (CC18 seeds 4-5), ariel-ydf-4 (regression)
- **Profile:** SPO `kdd28_hao.distacc_spo` (P = min(F, 1000), density 1.5, split across workers), AA `kdd28_ydf.distacc_aa`
  (Hao's settings do not touch AA). In-process MULTI_THREAD workers, 2 threads per worker. No early stopping, 300 trees, depth 6
- **Eval suites:** `kdd28_cc18_10fold` (test AUC), `kdd28_regression` (test RMSE, `--task REGRESSION`)
- **Machine:** EC2 ariel-ydf-1 (c8i.96xlarge: cc18_dist kdd28_hao), ariel-ydf-7 (c8i.metal-96xl: cc18_dist + reg_dist kdd28_ydf),
  ariel-ydf-4 (c8i.metal-96xl: reg_dist kdd28_hao). Shared-box timing, ballpark only
- **Date:** 2026-11-21 18:20Z to 18:49Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` (new tables `cc18_dist`, `reg_dist`) of 865fa729 + working tree,
  stamped `git=865fa729-dirty`. cc18_dist: the 2026-11-06 binaries (0416519f-dirty, `train_distributed_gbt` md5 c370ad8f).
  reg_dist: same source + `--task` flag in `examples/train_distributed_gbt.cc` (md5 f6986522, built in box worktree `~/wt/dist-task`).
  Launcher `ariel_notes/results/2026-11-21_where_spo_beats_aa_cc18_reg_distributed/run.sh`
- **Checks:** the `--task` binary gave byte-identical `nodes-*` and equal AUC to the old binary on 40/40 classification cells
  (credit-g, phoneme, W = 4/32, kdd28_hao seed 1). `model_identity_check.sh 865fa729`: 7/7 arms identical. Regression log:
  label NUMERICAL, loss SQUARED_ERROR
- **Raw cells:** gitignored `ariel_notes/results/2026-11-21_where_spo_beats_aa_cc18_reg_distributed/<box>/` (CSVs, logs,
  `models_and_results.tar.zst`)
- **Read by:** tab:accuracy-where-spo-outperforms-aa and the mean figures (`make_where_spo_beats_aa_hao_table.py`), the two
  column-replication figures (through `select_where_spo_beats_aa_hao.py`)
