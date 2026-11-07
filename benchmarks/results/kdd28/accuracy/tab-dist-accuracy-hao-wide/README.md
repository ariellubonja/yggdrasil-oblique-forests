# tab:dist-accuracy-hao-wide

Results behind the KDD28 floats below (section `sec:accuracy-hao`, "Accuracy with Hao's Settings over 5 Seeds"). Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:dist-accuracy-hao-wide`, `fig:dist-accuracy-mean-hao-wide`, `fig:column-replication-hao-wide`. Overleaf 20b3377 (2026-11-07). Metric: test ROC AUC
- **Status:** current (main body). Wide union (14 datasets with F >= 100: 6 SPAA paper eval + 8 wide dataset eval). Built by `../tab-dist-accuracy-hao/assemble_hao.py`
- **Profile:** `kdd28_hao` arms `distacc_single_spo`, `distacc_single_aa`, `distacc_spo`, `distacc_aa` (W = 4 only), `distacc_single_aa_spo`,
  `colrep_distacc_spo`. P = min(F, 1000) projections per node, density 1.5, split across workers. Seeds 1-5
- **Settings:** effective values from the profile (`params.py show kdd28_hao.<arm>`), checked against the CSV stamp columns by
  the generators. Threads 4 single, 2 per worker. No early stopping, no validation split, 300 trees, depth 6
- **Seed 1 reuse:** single SPO = the kdd28_ydf exact300 cells (same argv, bit-identical on 15/15 smoke folds), single AA and
  dist AA (W = 4) = the kdd28_ydf distacc cells. Profile `kdd28_hao._reuses` lets the generators accept their kdd28_ydf
  stamps, values are still checked. `source` column names each row's origin
- **Seed averaging:** table: each (dataset, fold, arm) averaged over seeds 1-5 (`--seed_mean`, all 5 required), then mean
  and std over folds. Column-replication figure: seeds pooled (balanced seed sets checked)
- **Eval suite:** `kdd28_wide_union` (seeds 1-5 since 2026-11-06). Epsilon runs separately in `wide_eps/` (two serial lanes after an OOM)
- **Machine:** EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on, in-process MULTI_THREAD workers, up to ~350 concurrent
  slots (not clean timing). Reused seed-1 AA rows: m7i.metal-24xl (2026-11-03), exact300 rows: c8i
- **Date:** 2026-11-06 23:53Z to 2026-11-07 07:11Z (reused rows 2026-11-03/04)
- **Code:** driver `run_kdd28_ydf_ec2.py` (--profile/--seed/--single_axes/--exclude), binaries 0416519f-dirty (`bin_sha`
  802b16fdf3b2, C++ equal to 57d9fd14), stamped `git=57d9fd14-dirty`. Launch scripts and code.diff:
  `ariel_notes/results/2026-11-06_hao_seeds/`
- **Generator:** `bash benchmarks/results/kdd28/accuracy/tab-dist-accuracy-hao/regen_hao.sh` (repo root): `assemble_hao.py` builds
  the inputs from the mirror, then `make_paper_tables.py --seed_mean` (figure, then table) and `make_column_replication_fig.py`
  for both evals
- **Inputs:** `inputs.txt`
- **Raw cells:** box `~/results/2026-11-06_hao_seeds/{spaa,wide,wide_eps}/seed<s>/` on ariel-ydf-1 (i-0f90f7764082bf21a, stopped),
  mirrored to the gitignored `ariel_notes/results/2026-11-06_hao_seeds/box/` (CSVs, logs, `models_*.tar.zst`)
