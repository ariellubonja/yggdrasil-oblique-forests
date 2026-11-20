# runs/2026-11-03_distacc

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (1836 rows: 33 datasets x 5 folds or 1 holdout x arms single:SPO, single:AA and W = 1, 4, 8,
  16, 32 SPO/AA, all rc = 0 rows). `params.py backfill-eval` (0e34d94e, 2026-11-05) added the columns
  `eval_suite,data_sha,metric_impl` after the run
- **Profile:** `kdd28_ydf.distacc_single_spo`, `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_aa`
  (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`). Threads 4 single, 2 per worker
  - `kdd28_ydf.distacc_single_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, include_axis_aligned=false, num_candidate_attributes=-1, profile=kdd28_ydf.distacc_single_spo
  - `kdd28_ydf.distacc_single_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, profile=kdd28_ydf.distacc_single_aa
  - `kdd28_ydf.distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0, max_unique_discretized=1, workers=1 4 8 16 32, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.distacc_spo
  - `kdd28_ydf.distacc_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, max_unique_discretized=1, workers=1 4 8 16 32, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.distacc_aa
  Seed 1 only. The uncommitted `kdd28_params.json` of 2026-11-20 lists seeds 1-5 for these 4 arms (seeds 2-5:
  `ariel_notes/results/2026-11-20_ydf_seeds/`, not in this dir. New seeds belong in their own `runs/<date>_<topic>/`)
- **Eval suite:** `kdd28_tabarena33` (stamped after the run by `params.py backfill-eval`, 0e34d94e)
- **Machine:** EC2 m7i.metal-24xl (Xeon 8488C), SMT off (48 cores online), turbo not recorded, distributed arms in-process
  (MULTI_THREAD workers)
- **Date:** 2026-11-03
- **Code:** rows `git=e3279f20-dirty` (delta: `ariel_notes/results/2026-11-03_ydf_defaults_rerun/box/code.diff`)
- **Raw cells:** box `~/results/2026-11-03_ydf_defaults_rerun/distacc/` on EC2 ariel-dist-accuracy-1 (i-0f90f7764082bf21a),
  mirrored (logs, models) to the gitignored `ariel_notes/results/2026-11-03_ydf_defaults_rerun/box/distacc/`. REPLICABILITY §J
- **Read by:** tab:accuracy-on-spaa-datasets, fig:column-replication-on-spaa-datasets (single and r = 0 rows),
  tab:accuracy-where-spo-outperforms-aa (`select_spo_advantage.py`), tab:accuracy-hao-vs-ydf-params (seed-1 AA rows, comparison
  tables), tab:accuracy-on-datasets-over-100-columns (`filter_wide.py`)
