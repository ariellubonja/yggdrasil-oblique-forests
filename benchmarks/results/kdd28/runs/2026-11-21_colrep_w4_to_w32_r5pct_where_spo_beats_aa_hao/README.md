# runs/2026-11-21_colrep_w4_to_w32_r5pct_where_spo_beats_aa_hao

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **What:** column replication at W = 4/8/16/32 with r = 0.05, 0.1, ..., 0.5, 0.6, ..., 1.0 (user request 2026-11-21) on the 26
  kdd28_hao `pass_rec` rows of tab:accuracy-where-spo-outperforms-aa. W = 4 holds only the new r (0.05, 0.15, ..., 0.45). The
  other W = 4 cells are in `2026-11-06_hao_seeds` / `2026-11-21_where_spo_beats_aa_cc18_reg_distributed`, skipped here with
  the driver's `--done_csvs`. r = 0 = the distributed SPO cell at the same W in those runs
- **Files:** `<group>/seed<s>/<table>/results.csv`, s = 1-5, as the driver wrote them, all rc = 0, no duplicate
  (dataset, fold, workers, replication). `spaa` (7 SPAA sets, `colrep_distacc`, 7,750 cells), `phil` + `eps` (wide sets philippine
  and epsilon, `colrep_distacc`, 1,250 + 250), `cc18` (11 CC18 sets x 10 folds, `colrep_cc18`, 27,500), `reg` (6 regression sets,
  `colrep_reg`, test RMSE, 7,500). 44,250 cells. `eps/seed5` merges two boxes (W = 4/8/16 on ariel-ydf-2, W = 32 on ariel-ydf-1)
- **Profile:** `kdd28_hao.colrep_distacc_spo` (W = 4/8/16/32 and this r grid since 2026-11-21). P = min(F, 1000), density 1.5,
  split across workers, BINARY weights. In-process MULTI_THREAD workers, 2 threads per worker. No early stopping, 300 trees, depth 6
- **Eval suites:** `kdd28_tabarena33` (spaa), `kdd28_wide_datasets` (phil, eps), `kdd28_cc18_10fold` (cc18), `kdd28_regression` (reg)
- **Machine:** EC2 ariel-ydf-1 (c8i.96xlarge: spaa, phil, cc18, eps seeds 1-3 and seed 5 W = 32), ariel-ydf-4 (c8i.metal-96xl:
  reg), ariel-ydf-2 resized to r8i.24xlarge, SMT off (eps seed 4 and seed 5 W = 4/8/16, epsilon folds rebuilt there by
  `rebuild_wide_data.py`, sha256 equal to the manifest). `machine` column per row. Shared-box timing, ballpark only
- **Date:** 2026-11-22 02:37Z to 13:07Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` of ee27dcdb + working tree (per-W colrep loops, `--done_csvs`,
  RAM budget `--mem_budget_gb`), stamped `git=ee27dcdb-dirty`. Binaries: classification the 2026-11-06 build
  (`train_distributed_gbt` md5 c370ad8f), regression the `--task` build (md5 f6986522), as in the W = 4 runs.
  Launcher and code.diff: `ariel_notes/results/2026-11-21_colrep_w4_to_w32_r5pct_where_spo_beats_aa_hao/`
- **Checks:** same-binary rerun of credit-g CC18 seed 1 at W = 4 gave 100/100 identical AUC and logloss to the 2026-11-21 cells.
  Epsilon W = 32 r = 1 peaks at 389 GB RSS (3-tree probe), hence the RAM budget (12.2 GB per held column copy + 10 GB)
- **Raw cells:** gitignored `ariel_notes/results/2026-11-21_colrep_w4_to_w32_r5pct_where_spo_beats_aa_hao/<box>/` (CSVs, logs,
  `*.tar.zst` with models)
- **Read by:** fig:column-replication-where-spo-outperforms-aa{,-regression} (W = 4) and the W = 8/16/32 bodies
  `fig_column_replication_where_spo_outperforms_aa{,_regression}_w{8,16,32}_body.tex`, through `select_where_spo_beats_aa_hao.py`
