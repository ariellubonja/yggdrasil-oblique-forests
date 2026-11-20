# runs/2026-11-06_hao_seeds

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `distacc/results.csv` (6120 rows), `distacc_aaspo/results.csv` (765), `colrep_distacc/results.csv` (7650), 33
  datasets, seeds 1-5. Written by `assemble_hao_vs_ydf_params.py` from the box mirror (rc = 0 rows, seed-1 single:SPO smoke
  rows dropped). `distacc/results.csv` also holds 459 adopted seed-1 rows of kdd28_ydf (profile `kdd28_hao._reuses`): 306
  single:AA and 4:AA rows of `runs/2026-11-03_distacc/` and 153 exact300 rows of `runs/2026-11-04_distacc_aaspo/` (as
  single:SPO). Their `source` column names the run file each row came from (459 cells rewritten by the assembler on
  2026-11-20 after the move to `runs/`, no other cell changed). The file is kept whole, so these rows are stored twice
- **Rebuild:** `python3 benchmarks/results/kdd28/runs/2026-11-06_hao_seeds/assemble_hao_vs_ydf_params.py` (repo root, needs the
  full mirror, exits without writing if a seed file is missing). Checked 2026-11-20: all three CSVs rebuild byte-identically, a second run changes nothing
- **Profile:** `kdd28_hao` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`). P = min(F, 1000) projections per
  node, density 1.5, split across workers. Seeds 1-5
- **Settings:** effective values (`params.py show <arm>`). Threads 4 single, 2 per worker. No early stopping, no validation
  split, 300 trees, depth 6
  - `kdd28_hao.distacc_single_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, include_axis_aligned=false, num_candidate_attributes=-1, profile=kdd28_hao.distacc_single_spo
  - `kdd28_hao.distacc_single_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, num_candidate_attributes=-1, profile=kdd28_hao.distacc_single_aa
  - `kdd28_hao.distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0, max_unique_discretized=1, workers=1 4 8 16 32, machine=c8i.metal-96xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, force_numerical=true, profile=kdd28_hao.distacc_spo
  - `kdd28_hao.distacc_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, max_unique_discretized=1, workers=4, machine=c8i.metal-96xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, force_numerical=true, profile=kdd28_hao.distacc_aa
  - `kdd28_hao.distacc_single_aa_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, num_candidate_attributes=-1, include_axis_aligned=true, profile=kdd28_hao.distacc_single_aa_spo
  - `kdd28_hao.colrep_distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1, max_unique_discretized=1, workers=4, machine=c8i.metal-96xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1 2 3 4 5, force_numerical=true, profile=kdd28_hao.colrep_distacc_spo
- **Eval suite:** `kdd28_tabarena33` (seeds 1-5 since 2026-11-06)
- **Machine:** EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on, in-process MULTI_THREAD workers, up to ~350 concurrent
  slots (not clean timing). Adopted seed-1 AA rows: m7i.metal-24xl (2026-11-03), exact300 rows: c8i
- **Date:** 2026-11-06 23:53Z to 2026-11-07 07:11Z (adopted rows 2026-11-03/04)
- **Code:** driver `run_kdd28_ydf_ec2.py` (--profile/--seed/--single_axes/--exclude), binaries 0416519f-dirty (`bin_sha`
  802b16fdf3b2, C++ equal to 57d9fd14), stamped `git=57d9fd14-dirty`. Launch scripts and code.diff:
  `ariel_notes/results/2026-11-06_hao_seeds/`
- **Raw cells:** box `~/results/2026-11-06_hao_seeds/{spaa,wide,wide_eps}/seed<s>/` on ariel-ydf-1 (i-0f90f7764082bf21a),
  mirrored to the gitignored `ariel_notes/results/2026-11-06_hao_seeds/box/` (CSVs, logs, `models_*.tar.zst`)
- **Read by:** tab:accuracy-hao-vs-ydf-params and its figures and comparison tables
