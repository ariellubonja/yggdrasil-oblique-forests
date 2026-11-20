# runs/2026-11-04_distacc_aaspo: extra single-machine columns of tab:accuracy-on-spaa-datasets (2026-11-04)

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- `distacc_aaspo.csv`: AA+SPO-GBT, profile arm `kdd28_ydf.distacc_single_aa_spo` (YDF defaults, `--oblique_also_axis_aligned`).
- `distacc_exact300.csv`: arm (c) of tab:cc18-aa-spo, `kdd28_ydf.distacc_single_spo_harness_exact300`
  (300 trees, EXACT, exponent 1, max 1000, density 1.5, no early stopping, no `--ydf_defaults`: P = min(F, 1000)).
- Data: the 33 TabArena/TabReD datasets and folds of tab:dist-accuracy (eval suite `kdd28_tabarena33`), seed 1, 4 threads.
- Box: EC2 ariel-dist-accuracy-1 (i-0f90f7764082bf21a) resized to c8i.metal-96xl (Xeon 6975P-C, 384 threads,
  SMT and turbo on). The distacc rows these columns sit next to ran on m7i.metal-24xl with SMT off (2026-11-03).
  Determinism was tested on m7i only.
- Binaries: `~/results/2026-11-03_ydf_defaults_rerun/bin` (built at e3279f20, no C++ change to the driver commit).
  Driver: `run_kdd28_ydf_ec2.py --tables distacc_aaspo,distacc_exact300` at the commit that adds this folder.
- `train_s` comes from up to 64 concurrent cells on the shared box: not clean timing.
- Body: the generator command in `floats/accuracy/tab-accuracy-on-spaa-datasets/README.md`. The distacc CSV these columns
  sit next to: `runs/2026-11-03_distacc/`.
- **Profile:** `kdd28_ydf.distacc_single_aa_spo`, `kdd28_ydf.distacc_single_spo_harness_exact300`
- **Settings:** effective values of the run (`params.py show <arm>`, seeds as run)
  - `kdd28_ydf.distacc_single_aa_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, include_axis_aligned=true, profile=kdd28_ydf.distacc_single_aa_spo
  - `kdd28_ydf.distacc_single_spo_harness_exact300`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, include_axis_aligned=false, profile=kdd28_ydf.distacc_single_spo_harness_exact300
  Seed 1 only. The uncommitted `kdd28_params.json` of 2026-11-20 lists seeds 1-5 for distacc_single_aa_spo (seeds 2-5:
  `ariel_notes/results/2026-11-20_ydf_seeds/`, not in this dir. New seeds belong in their own `runs/<date>_<topic>/`)
- **Eval suite:** `kdd28_tabarena33`. 18 rows of `distacc_aaspo.csv` predate the `oblique_also_axis_aligned` stamp
- **Machine:** EC2 c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on (`machine` column)
- **Date:** 2026-11-04
- **Code:** driver `c3b01f56-dirty`, binaries built at e3279f20 (`bin_sha` 168e934e2d19)
- **Raw cells:** box `~/results/2026-11-04_distacc_aaspo/` on EC2 ariel-dist-accuracy-1, mirrored (logs, models) to the
  gitignored `ariel_notes/results/2026-11-04_distacc_aaspo/box/`. REPLICABILITY §J
- **Read by:** tab:accuracy-on-spaa-datasets, tab:accuracy-on-datasets-over-100-columns (`filter_wide.py`),
  tab:accuracy-where-spo-outperforms-aa (`select_spo_advantage.py`), tab:accuracy-hao-vs-ydf-params-single-machine and
  -distributed (exact300 = seed-1 Hao single SPO, aaspo = seed-1 YDF AA+SPO)
