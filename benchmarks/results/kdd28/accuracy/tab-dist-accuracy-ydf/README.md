# tab:dist-accuracy-ydf

Results behind the KDD28 float(s) below. Rebuild the body with the generator command. Index and rules:
`benchmarks/results/kdd28/README.md`.

- **Labels:** `tab:dist-accuracy-ydf`, `fig:dist-accuracy-mean-ydf`
  - `tab:dist-accuracy-ydf`: "Test AUC of single-machine vs. $W$-distributed GBT. Single-machine SPO-GBT samples $\min(F^2,6000)$ projections per node, distributed SPO-GBT splits them across the $W$ workers, each sampling from ..."
  - `fig:dist-accuracy-mean-ydf`: "Mean test AUC over the datasets of Table~\ref{tab:dist-accuracy-ydf}."
- **Status:** current (main body, Accuracy section)
- **Profile:** `kdd28_ydf.distacc_single_spo`, `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_aa`, `kdd28_ydf.distacc_single_aa_spo`, `kdd28_ydf.distacc_single_spo_harness_exact300` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`)
- **Settings:** effective values from the profile (`params.py show <arm>`), checked against the CSV stamp columns
  by the generator where the CSV has them
  - `kdd28_ydf.distacc_single_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, include_axis_aligned=false, num_candidate_attributes=-1, profile=kdd28_ydf.distacc_single_spo
  - `kdd28_ydf.distacc_single_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, profile=kdd28_ydf.distacc_single_aa
  - `kdd28_ydf.distacc_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, split_budget=true, local_projection_count=false, column_replication=0, max_unique_discretized=1, workers=1 4 8 16 32, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.distacc_spo
  - `kdd28_ydf.distacc_aa`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, max_unique_discretized=1, workers=1 4 8 16 32, machine=m7i.metal-24xl in-process, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, force_numerical=true, profile=kdd28_ydf.distacc_aa
  - `kdd28_ydf.distacc_single_aa_spo`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=2, spo_max_projections=6000, spo_density=2, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, include_axis_aligned=true, profile=kdd28_ydf.distacc_single_aa_spo
  - `kdd28_ydf.distacc_single_spo_harness_exact300`: trees=300, max_depth=6, min_examples=5, split_type=EXACT, spo_exponent=1, spo_max_projections=1000, spo_density=1.5, spo_min_projections=1, validation_set_ratio=0, early_stopping=NONE, folds=5 CV folds (chronological sets: 1 holdout), seeds=1, num_candidate_attributes=-1, include_axis_aligned=false, profile=kdd28_ydf.distacc_single_spo_harness_exact300
- **Eval suite:** `kdd28_tabarena33` (33 TabArena/TabReD, 5 CV folds, 1 chronological holdout for ecom-offers, homecredit-default, homesite-insurance, seed 1, test ROC AUC). `distacc/` rows predate the eval stamps (checked on dataset, fold, seed only)
- **Machine:** `distacc/`: EC2 m7i.metal-24xl (Xeon 8488C), SMT off (48 cores online), turbo not recorded, distributed arms in-process (MULTI_THREAD workers). `distacc_aaspo/`: the same instance resized to c8i.metal-96xl (Xeon 6975P-C), SMT and turbo on (`machine` column). Not clean timing: up to 64 concurrent cells
- **Date:** `distacc/` 2026-11-03, `distacc_aaspo/` 2026-11-04
- **Code:** `distacc/`: rows `git=e3279f20-dirty` (delta: `ariel_notes/results/2026-11-03_ydf_defaults_rerun/box/code.diff`). `distacc_aaspo/`: driver `c3b01f56-dirty`, binaries built at e3279f20 (`bin_sha` 168e934e2d19)
- **Generator:** run from the repo root, writes into the paper dir (`--out` to redirect)

```
python3 benchmarks/src/distributed_spo_gbt/make_paper_tables.py --accuracy_csv benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv \
  --accuracy_note "IN-PROCESS rerun on AWS m7i.metal-24xl (Xeon 8488C; workers = threads of one process)" \
  --arms distacc_single_spo,distacc_single_aa,distacc_spo,distacc_aa --profile kdd28_ydf --suffix _ydf --purple \
  --extra aaspo=benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc_aaspo/distacc_aaspo.csv,exact300=benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc_aaspo/distacc_exact300.csv --extra_first \
  --std_below --shape_below --no_figure  # table only (2026-11-06), figure: drop these flags
```
- **Inputs:** `inputs.txt` (one repo-relative path per line)
  - `benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc/results.csv`
  - `benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc_aaspo/distacc_aaspo.csv`
  - `benchmarks/results/kdd28/accuracy/tab-dist-accuracy-ydf/distacc_aaspo/distacc_exact300.csv`
- **Raw cells:** box `~/results/2026-11-03_ydf_defaults_rerun/distacc/` and `~/results/2026-11-04_distacc_aaspo/` on EC2 ariel-dist-accuracy-1 (i-0f90f7764082bf21a), mirrored (logs, models) to the gitignored `ariel_notes/results/2026-11-03_ydf_defaults_rerun/box/distacc/` and `ariel_notes/results/2026-11-04_distacc_aaspo/box/`. REPLICABILITY §J

Regenerated 2026-11-04 from these paths: table and figure bodies equal the Overleaf copies in every non-`%` line.

