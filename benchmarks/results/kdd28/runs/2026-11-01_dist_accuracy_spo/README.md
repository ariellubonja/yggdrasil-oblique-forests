# runs/2026-11-01_dist_accuracy_spo

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (765 rows, 33 datasets, distributed SPO, W = 1, 4, 8, 16, 32). Old place: `table8/fullfix/` of the
  2026-11-01 fixed reruns
- **Profile:** `table8.dist_spo` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`)
  - `table8.dist_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=2147483647, spo_density=1.5, spo_min_projections=1, split_budget=false, local_projection_count=true, column_replication=0, force_numerical=false, max_unique_discretized=1, workers=1 4 8 16 32, folds=5 CV folds (chronological sets: 1 holdout), seeds=123456, profile=table8.dist_spo
- **Eval suite:** `kdd28_tabarena33` folds (unstamped CSV: no eval columns)
- **Machine:** EC2 m7i.metal-24xl (Xeon 8488C), in-process. SMT/turbo: unknown, not recorded in the CSV
- **Date:** 2026-11-01
- **Code:** dgbt-spo-full-fixes 10509539
- **Raw cells:** this file
- **Read by:** tab:dist-accuracy, fig:dist-accuracy-mean, through `outdated/tab-dist-accuracy/derived/table8_results.csv`
  (`build_paper_inputs.py`, rebuilt byte-identically 2026-11-20)
