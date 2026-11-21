# runs/2026-11-21_wide_eval_ydf_aa_seeds2to5

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `seed<s>/{philippine,epsilon,epsilon_w1}/distacc/results.csv`, s = 2-5, as the driver wrote them (one driver
  process per dataset, own out dir). 144 rows, all rc = 0: philippine 5 folds x (single:AA + W = 1/4/8/16/32 AA), epsilon
  1 holdout x the same 6 arms. Seed 1 = `runs/2026-11-04_wide_eval/distacc/`
- **Profile:** `kdd28_ydf.distacc_single_aa`, `kdd28_ydf.distacc_aa`. Threads 4 single, 2 per worker, in-process workers
- **Eval suite:** `kdd28_wide_datasets` (epsilon = Vertex AI files, philippine 5 CV folds)
- **Machine:** ariel-ydf-1 (c8i.96xlarge, seeds 2-3), ariel-ydf-7 (c8i.metal-96xl, seeds 4-5), epsilon W = 1 on ariel-ydf-9
  (r8i.8xlarge, i-01753edfb8958d918, launched 2026-11-21 to save cost on the 2-3 h W = 1 cells, user request). Shared-box timing
- **Date:** 2026-11-21 20:09Z to 22:33Z
- **Code:** driver of 865fa729 + working tree (`git=865fa729-dirty`), binaries 0416519f-dirty (md5 c370ad8f / 2eaadddf).
  Launcher `ariel_notes/results/2026-11-21_wide_eval_ydf_aa_seeds2to5/run.sh`
- **Checks:** distributed AA test AUC is identical across seeds 1-5 and W in every cell (epsilon .940322, philippine per
  fold), as in the 765 SPAA dist AA cells. Single AA: epsilon identical across seeds, philippine varies (.8492-.8542)
- **Raw cells:** gitignored `ariel_notes/results/2026-11-21_wide_eval_ydf_aa_seeds2to5/<box>/` (logs, models tarball)
- **Read by:** tab:accuracy-where-spo-outperforms-aa and fig:accuracy-where-spo-outperforms-aa (`make_where_spo_beats_aa_hao_table.py`)
