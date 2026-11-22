# tab:accuracy-hao-vs-ydf-params-single-machine

Results behind the KDD28 floats below (section `sec:accuracy-hao-vs-ydf-params`, "Accuracy with Hao's Settings over 5 Seeds"). Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:accuracy-hao-vs-ydf-params-single-machine`, `tab:accuracy-hao-vs-ydf-params-distributed`. Overleaf fb73e6c (2026-11-20). Metric: test ROC AUC
- **Status:** current (appendix). Two row blocks since 2026-11-20: SPAA paper eval (33 datasets) and wide dataset eval (8 datasets: epsilon, santander_transaction, hcdr_main, dota2, vehicle_sensIT, guillermo, philippine, christine). The 33 SPAA rows are byte-identical to the one-block bodies. Mean rows: 33 SPAA sets, 8 wide sets, all 41. Mean time per block only. The separate wide table (tab:accuracy-hao-vs-ydf-params-over-100-columns) was dropped from the paper 2026-11-20, results in git history before that date. tab:accuracy-hao-vs-ydf-params, fig:accuracy-hao-vs-ydf-params and fig:column-replication-hao-vs-ydf-params (Hao-only 5-seed table and figures) were dropped 2026-11-20 (Overleaf a157f51), generator calls in git history before that date. Their `colrep_distacc` cells stay in the run dir, read by no float
- **Runs:** `runs/2026-11-06_hao_seeds/` (kdd28_hao seeds 1-5, with the adopted seed-1 rows), `runs/2026-11-03_distacc/` and
  `runs/2026-11-04_distacc_aaspo/` (kdd28_ydf seed 1) and `runs/2026-11-20_ydf_seeds/` (kdd28_ydf seeds 2-5), comparison tables.
  Wide block: `runs/2026-11-06_hao_seeds/wide_eval/` (kdd28_hao seeds 1-5) and `runs/2026-11-04_wide_eval/` (kdd28_ydf seed 1). Run facts: their READMEs
- **Profile:** `kdd28_hao` arms `distacc_single_spo`, `distacc_single_aa`, `distacc_spo`, `distacc_aa` (W = 4 only), `distacc_single_aa_spo`,
  `colrep_distacc_spo`. P = min(F, 1000) projections per node, density 1.5, split across workers. Seeds 1-5
- **Settings:** effective values from the profile (`params.py show kdd28_hao.<arm>`), checked against the CSV stamp columns by
  the generators. Threads 4 single, 2 per worker. No early stopping, no validation split, 300 trees, depth 6
- **Seed 1 reuse:** single SPO = the kdd28_ydf exact300 cells (same argv, bit-identical on 15/15 smoke folds), single AA and
  dist AA (W = 4) = the kdd28_ydf distacc cells. Profile `kdd28_hao._reuses` lets the generators accept their kdd28_ydf
  stamps, values are still checked. `source` column names each row's origin (`runs/` paths since 2026-11-20)
- **Seed averaging:** each (dataset, fold, arm) averaged over seeds 1-5 (all 5 required), then mean over folds
- **± (since 2026-11-21, user decision):** sample std (ddof=1) over seeds of the fold-averaged score (single-holdout
  sets such as epsilon, homecredit-default: of the holdout score). No ± with 1 seed (wide kdd28_ydf cells) or identical seeds
  (distributed AA-GBT). Before: std over folds of the seed-averaged score. Means unchanged (checked by diff 2026-11-21)
- **Eval suite:** `kdd28_tabarena33` (seeds 1-5 since 2026-11-06) for the SPAA block, `kdd28_wide_datasets` for the wide block.
  The generator declares each suite while its block's inputs load (strict: exit 2 on dataset, fold, hash or metric gaps)
- **Machine:** EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on, in-process MULTI_THREAD workers, up to ~350 concurrent
  slots (not clean timing). Reused seed-1 AA rows: m7i.metal-24xl (2026-11-03), exact300 rows: c8i
- **Date:** 2026-11-06 23:53Z to 2026-11-07 07:11Z (reused rows 2026-11-03/04)
- **Code:** driver `run_kdd28_ydf_ec2.py` (--profile/--seed/--single_axes/--exclude), binaries 0416519f-dirty (`bin_sha`
  802b16fdf3b2, C++ equal to 57d9fd14), stamped `git=57d9fd14-dirty`. Launch scripts and code.diff:
  `ariel_notes/results/2026-11-06_hao_seeds/`
- **Generator:** `bash benchmarks/results/kdd28/floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/regen_hao_vs_ydf_params.sh` (repo root): `runs/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py`
  rebuilds the run's CSVs from the mirror (reproduces them byte-identically), then `make_hao_vs_ydf_compare_tables.py`
- **Comparison tables (2026-11-20):** `make_hao_vs_ydf_compare_tables.py` (last line of the regen script) sets kdd28_ydf seeds 1-5
  against kdd28_hao seeds 1-5 per arm, single machine and distributed, each (dataset, fold, arm) averaged over its 5 seeds first.
  YDF inputs: seed 1 `runs/2026-11-03_distacc/results.csv` and `runs/2026-11-04_distacc_aaspo/distacc_aaspo.csv`, seeds 2-5
  `runs/2026-11-20_ydf_seeds/seed<s>/{distacc,distacc_aaspo}/results.csv` (YDF seeds since 2026-11-20). YDF times mix seed 1
  (m7i.metal-24xl, SMT off) with seeds 2-5 (c8i). Check 2026-11-20: single:AA and 4:AA seeds 2-5 equal the Hao rows in 1,224/1,224 cells
- **Wide block (2026-11-20):** kdd28_ydf seed 1 (`--wide_ydf_seeds 1`, seeds 2-5 never ran on the wide sets) against the
  kdd28_hao mean over seeds 1-5. The italic block heading says "YDF: seed 1" because the column heads say "YDF 5 seeds". The
  generator exits non-zero when a wide cell lacks exactly those seeds, a fold or a dataset (checked 2026-11-20 by deleting one
  seed, one dataset, one fold). Single SPO, YDF / Hao: epsilon .9510 / .9464, santander_transaction .8823 / .8842, hcdr_main
  .7558 / .7576, dota2 .6283 / .6315, vehicle_sensIT .9289 / .9287, guillermo .8998 / .8979, philippine .8702 / .8645,
  christine .8196 / .8188. Shapes: `derived/wide_shapes.csv` (`--wide_dataset_shapes`), written by `make_wide_shapes.py`
- **CC18, multiclass and regression blocks (user request 2026-11-20):** two continuation floats of the single-machine table
  (`\ContinuedFloat`, bodies `table_accuracy_hao_vs_ydf_params_single_machine_{cc18,mc_reg}_body.tex`). CC18: 34 binary tasks,
  10 OpenML folds, `runs/2026-11-20_cc18_seeds/`. Multiclass: 8 TabArena tasks as majority class vs. rest, test AUC. Regression:
  13 TabArena + 5 TabReD tasks, harness `--task regression`, test RMSE (lower is better), Hao cells colored by % RMSE change,
  mean row = mean over datasets of RMSE / label std (`derived/extra_label_std.csv`, full-dataset std). Both from
  `runs/2026-11-20_regression_multiclass_tabarena_tabred/`. All three: both profiles seeds 1-5. Data prep `benchmarks/data/prep_extra_tabular_folds.py`,
  checked by an independent Fable-xhigh verifier on 2026-11-20 (all 220 fold files re-derived from the raw sources). Shapes and
  cat= counts: `derived/{cc18,extra}_{shapes,categoricals}.csv` (CC18 cat = OpenML nominal/string features)
- **AA-GBT column (user directive 2026-11-20):** Hao's settings do not touch AA, so every block shows only the YDF-parameter AA
  column, in both Hao-vs-YDF tables. Hao AA cells were identical where they ran (CC18 1,700/1,700)
- **Page fit (2026-11-20):** with the wide block the single-machine float ran past its page (its \twocolumn block was capped at
  \textheight, about 80pt of overflow, Overleaf ba69123). `--single_arraystretch 0.75` shrinks only the row spacing, not the
  font: block height 608.5pt of 626pt, rendered page 15 ends inside the text area. The distributed body fits at 1.0 (607.7pt)
- **Inputs:** `inputs.txt`
- **Raw cells:** box `~/results/2026-11-06_hao_seeds/{spaa,wide,wide_eps}/seed<s>/` on ariel-ydf-1 (i-0f90f7764082bf21a, stopped),
  mirrored to the gitignored `ariel_notes/results/2026-11-06_hao_seeds/box/` (CSVs, logs, `models_*.tar.zst`)
- **SPO clearly beats AA column (2026-11-21, user request):** check marks per profile (YDF, Hao) from
  `select_spo_clearly_beats_aa.py` (Fable-high analysis 2026-11-20, copy of the gitignored
  `ariel_notes/results/2026-11-20_where_spo_beats_aa_cc18_openML_Wide_regression_tighter_tolerance/`, outputs byte-identical).
  Rule: 5-seed mean effect above the floor (AUC edge > 0.005, or RMSE reduction (AA-SPO)/AA > 1 %), one-sided paired t
  over seed-averaged fold differences survives Benjamini-Hochberg q = 0.05 over all testable rows, effect positive in every
  seed. Writes `derived/spo_clearly_beats_aa_{per_dataset.csv,report.txt}`. The distributed body shows the same
  single-machine marks. `Mean (all 41)` rows dropped from both main bodies (user 2026-11-21)
