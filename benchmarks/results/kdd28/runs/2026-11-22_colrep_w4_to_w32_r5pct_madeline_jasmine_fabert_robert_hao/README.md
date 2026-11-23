# runs/2026-11-22_colrep_w4_to_w32_r5pct_madeline_jasmine_fabert_robert_hao

Measurement, stored once. Read by no float yet. Index and rules: `benchmarks/results/kdd28/README.md`.

- **What:** column replication at W = 4/8/16/32 with r = 0.05, 0.1, ..., 0.5, 0.6, ..., 1.0 (user request 2026-11-22) on the
  four sets the 2026-11-22 wide real-dataset screen selected (madeline, jasmine, fabert, robert). r = 0 = the screen's
  `{W}:SPO` cells and single-machine SPO = its `single:SPO` cells (`runs/2026-11-22_wide_real_dataset_candidates_spo_vs_aa_screen_hao/`),
  not rerun here
- **Files:** `seed<s>/colrep_distacc_stamped/results.csv`, s = 1-5, as the driver wrote them: 6,000 cells (4 sets x 5 folds x
  4 W x 15 r x 5 seeds), all rc = 0. `check_colrep_grid.py` re-runs checks 1-4 and 7 from the CSVs and writes
  `derived/colrep_mean_delta.csv` (dataset, W, r, mean test-AUC delta to single-machine SPO, seed sd, n cells, r = 0 rows from
  the screen). `anchor_check/` and `rerun_check/` hold the check-5 and check-6 cells (not part of the 6,000)
- **Profile:** `kdd28_hao.colrep_distacc_spo`. P = min(F, 1000), density 1.5, exponent 1, split across workers
  (split_budget=true), BINARY weights. In-process MULTI_THREAD workers, 2 threads per worker. No early stopping (the
  distributed learner has none), 300 trees, depth 6, min_examples 5, EXACT
- **Eval suite:** rows stamped with the screen's suite (id in the `eval_suite` column), same fold files (data_sha equal to the
  screen's cells). `kdd28_wide_real_colrep_selected` (new, `union_of` the 4 selected sets) is the suite `params.py check-eval` passes against
- **Machine:** EC2 ariel-ydf-4 (c8i.metal-96xl, i-07de26fdf094924be, SMT + turbo on, stopped after the run). Shared-box timing
- **Date:** 2026-11-22 21:24Z to 22:35Z (71 min, about 5,100 cells/h, 698 CPU-h as train_s x threads). Anchor check 21:23Z, reruns 21:53Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` of a6917d8a + working tree (new table `colrep_distacc_stamped`
  = `colrep_distacc` with `bin_sha` = the distributed binary's sha. `colrep_distacc` unchanged), stamped `git=a6917d8a-dirty`.
  Binaries: `train_distributed_gbt` sha256 802b16fdf3b2 (md5 c370ad8f, the 2026-11-06 build, `~/results/2026-11-06_hao_seeds/bin`),
  the same build as the screen's r = 0 cells. Single-machine baseline `train_oblique_forest` ccd48156c4c4 (screen).
  Launcher, helper and code.diff: `ariel_notes/results/2026-11-22_colrep_w4_to_w32_r5pct_madeline_jasmine_fabert_robert_hao/`
- **Checks:** (1) 6,000 unique (dataset, fold, seed, W, r), all rc = 0 with an AUC. (2) data_sha per (dataset, fold) equal to
  the screen's r = 0 and single-machine cells (20/20). (3) stamps equal on all cells, threads = 2W, no drift, logs show
  spo_weights=BINARY and the cell's W and r. (4) `held_cols` of all 240 (dataset, W, r) configs: owned sums = F, spread <= 1,
  replicas = lround(float32(r) x non-owned), constant over folds and seeds (14 configs sit on a .5 tie where float32 rounds
  down, all match). (5) `--column_replication 0` through this arm, fold 0 seed 1, W = 4 and 32 on all 4 sets: 8/8 AUC equal to
  the screen's r = 0 cells. (6) 8 same-seed reruns: 8/8 identical AUC, logloss, held_cols. (7) r = 1 vs single SPO: 4 of 16
  (dataset, W) beyond 2 seed-SE (madeline W = 8 and 32, about -0.002, fabert W = 8 and 32, -0.0005 and -0.0009). Under the null
  about 1.9 of 16 are expected (t with 4 df, two-sided p 0.116 at 2 SE). Why r = 1 differs from single SPO: hypothesis,
  untested (split budget across workers and per-worker projection streams). (8) `check_argv_equivalence.sh` ALL PASS,
  `params.py check-eval` OK on the 5 CSVs, no STOP in any log. `check_colrep_grid.py` re-runs 1-4 and 7
- **Cost:** r > 0 cells took 1.08x the screen's r = 0 cells (sum of train_s, per-(dataset, W) range 0.67 to 1.34). Shared-box
  load differed between the two runs, so this ratio is ballpark
- **Raw cells:** gitignored `ariel_notes/results/2026-11-22_colrep_w4_to_w32_r5pct_madeline_jasmine_fabert_robert_hao/box/`
  (CSVs, driver logs, `models_and_logs.tar.zst` with models and all cell logs, sha256 630f9526). Box copy `~/results/2026-11-22_colrep_w4_to_w32_r5pct_madeline_jasmine_fabert_robert_hao/`
- **Read by:** no float
