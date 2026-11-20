# runs/2026-11-03_cc18_aa_spo

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `results.csv` (1700 rows: 34 CC18 datasets x 10 folds x 5 arms)
- **Profile:** `cc18_aa_spo` arms (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values (`params.py show <arm>`). No stamp columns: settings are the profile's reconstruction of the
  run script
  - `cc18_aa_spo.aa_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.aa_ydf
  - `cc18_aa_spo.spo_harness`: trees=30, max_depth=6, min_examples=5, split_type=DYNAMIC_RANDOM_HISTOGRAM, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0, early_stopping=NONE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_harness
  - `cc18_aa_spo.spo_harness_exact300`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0, early_stopping=NONE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_harness_exact300
  - `cc18_aa_spo.spo_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, include_axis_aligned=false, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.spo_ydf
  - `cc18_aa_spo.aa_spo_ydf`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, num_candidate_attributes=-1, validation_set_ratio=0.1, early_stopping=VALIDATION_LOSS_INCREASE, include_axis_aligned=true, folds=0 1 2 3 4 5 6 7 8 9, seeds=1, profile=cc18_aa_spo.aa_spo_ydf
- **Eval suite:** `kdd28_cc18_10fold` (34 CC18 binary, 10 CV folds, seed 1)
- **Machine:** EC2 m7i.metal-24xl, single machine. SMT/turbo: unknown, not recorded
- **Date:** 2026-11-03 (as dated by the float README)
- **Code:** binary 9a94b070 (`bin_sha` 96c9a72d28ba, before 9dc54c93: harness SPO flags honoured). The `git` column is
  `9a94b070-dirty` for arm spo_harness_exact300, `9a94b070` for the other four
- **Raw cells:** gitignored `ariel_notes/results/2026-11-03_cc18_harness_exact300/` (logs, PROVENANCE.txt, run_exact300.log)
- **Read by:** tab:cc18-aa-spo, and `outdated/tab-cc18-aa-spo-ydf/` (`--recorded`)
