# runs/2026-11-04_wide_eval

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `<table>/results.csv` for `distacc` (432 rows), `distacc_aaspo` (36), `distacc_exact300` (36), `colrep_distacc`
  (360), 8 datasets. Both boxes merged, rc = 0, deduped (49 cells ran on both boxes with bit-identical metrics). One run in
  two shards, as the float README always treated it
- **Profile:** `kdd28_ydf.distacc_single_spo`, `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_spo`, `kdd28_ydf.distacc_aa`,
  `kdd28_ydf.distacc_single_aa_spo`, `kdd28_ydf.distacc_single_spo_harness_exact300`, `kdd28_ydf.colrep_distacc_spo`
- **Settings:** effective values (`params.py show <arm>`), the same flags as the SPAA paper eval runs. Threads 4 single, 2 per
  worker. Arm lines: `runs/2026-11-03_distacc/README.md`, `runs/2026-11-04_distacc_aaspo/README.md`,
  `runs/2026-11-04_colrep_distacc/README.md`
  Seed 1 only. The uncommitted `kdd28_params.json` of 2026-11-20 lists seeds 1-5 for the distacc_single_*, distacc_spo, distacc_aa and distacc_single_aa_spo arms (seeds 2-5:
  `ariel_notes/results/2026-11-20_ydf_seeds/`, not in this dir. New seeds belong in their own `runs/<date>_<topic>/`)
- **Eval suite:** `kdd28_wide_datasets` (8 sets, F >= 100: epsilon, santander_transaction, hcdr_main, dota2, vehicle_sensIT,
  guillermo, philippine, christine. 5 CV folds, seed 0 fold split, Epsilon and the chronological sets: 1 holdout, seed 1, no
  validation, no early stopping). YouTube-8M dropped (too slow)
- **Machine:** EC2 c8i.metal-96xl SMT+turbo (Xeon 6975P-C), ariel-ydf-1 (i-0f90f7764082bf21a) and ariel-ydf-7
  (i-04668dc60810fb749), distributed arms in-process (MULTI_THREAD workers)
- **Date:** 2026-11-04 to 2026-11-06 (UTC)
- **Code:** driver and binaries `0416519f-dirty` (`ariel_notes/results/2026-11-04_wide_eval/ariel-ydf-1/code.diff`), folds by
  `benchmarks/src/distributed_spo_gbt/rebuild_wide_data.py` (sha256 pins `benchmarks/data/kdd28_wide_datasets_manifest.sha256`)
- **Raw cells:** box `~/results/2026-11-04_wide_eval/`, mirrored to the gitignored
  `ariel_notes/results/2026-11-04_wide_eval/{ariel-ydf-1,ariel-ydf-7}` (logs, models). Shard scripts
  `ariel_notes/results/2026-11-05_wide_eval_shards/`
- **Read by:** tab:accuracy-on-datasets-over-100-columns (`union_wide.py`), tab:accuracy-where-spo-outperforms-aa (`select_spo_advantage.py`)
