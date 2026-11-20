# runs/2026-11-04_colrep_distacc

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv`: 1530 cells (153 dataset-folds x r = 0.1..1.0) plus 3 r = 0 checks (blood-transfusion fold 1,
  diabetes fold 2, Fitness_Club fold 0), all rc = 0. The checks give the same test AUC as the 2026-11-03_distacc W = 4 SPO
  rows (box: models byte-identical)
- **Profile:** `kdd28_ydf.colrep_distacc_spo` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`). The profile's `machine` field names the m7i, the run used the c8i
  - `kdd28_ydf.colrep_distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1, max_unique_discretized=1, workers=4, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.colrep_distacc_spo
- **Eval suite:** `kdd28_tabarena33`, stamped, `params.py check-eval --suite kdd28_tabarena33` clean (1533 rows)
- **Machine:** EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on, W = 4 in-process workers, 8 threads
- **Date:** 2026-11-04
- **Code:** driver `c3b01f56-dirty`, binaries built at e3279f20 (`bin_sha` 168e934e2d19)
- **Raw cells:** box `~/results/2026-11-04_colrep_distacc/` (EC2 ariel-dist-accuracy-1), mirrored with logs and models to the
  gitignored `ariel_notes/results/2026-11-04_colrep_distacc/box/colrep_distacc/`. REPLICABILITY §J
- **Read by:** fig:column-replication-on-spaa-datasets, fig:column-replication-over-100-columns (`filter_wide.py`),
  fig:column-replication-where-spo-outperforms-aa (`select_spo_advantage.py`)
