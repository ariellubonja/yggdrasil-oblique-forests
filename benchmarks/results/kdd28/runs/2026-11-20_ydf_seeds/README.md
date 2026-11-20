# runs/2026-11-20_ydf_seeds

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `seed<s>/distacc/results.csv` and `seed<s>/distacc_aaspo/results.csv`, s = 2-5, as the driver wrote them. Per seed
  1,989 rc = 0 cells (1,836 distacc + 153 distacc_aaspo), 33 datasets. Seed 1 of the same arms is `runs/2026-11-03_distacc/`
  and `runs/2026-11-04_distacc_aaspo/distacc_aaspo.csv`. Rows with rc != 0 are the cells cut off by the disk-full crash, each
  rerun later with rc = 0 (readers keep rc = 0 only)
- **Crash edit:** the root disk filled at 21:22Z and left 3 half-written lines that the resumed run appended onto. Those 3
  lines were removed before the rerun (verbatim in `crash_removed_lines.txt`). The 2 of them that held a complete rc = 0 row
  equal the rerun values exactly (qsar-biodeg fold 0 16:SPO seed 2 = 0.925855, single:AA seed 3 = 0.933501)
- **Profile:** `kdd28_ydf` arms `distacc_single_spo`, `distacc_single_aa`, `distacc_spo`, `distacc_aa` (W = 1, 4, 8, 16, 32),
  `distacc_single_aa_spo` (`benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`, seeds widened to 1-5 on 2026-11-20).
  P = min(F², 6000) projections per node, density 2, split across workers
- **Settings:** effective values (`params.py show kdd28_ydf.<arm>`), identical argv to seed 1 except `--seed`. Threads 4 single,
  2 per worker. No early stopping, no validation split, 300 trees, depth 6
- **Eval suite:** `kdd28_tabarena33` (SPAA paper eval). `params.py check-eval` OK on all 8 files
- **Machine:** EC2 ariel-ydf-1 (i-0f90f7764082bf21a), Xeon 6975P-C, 384 threads, in-process MULTI_THREAD workers, ~360
  concurrent slots (not clean timing). c8i.metal-96xl until the crash, then c8i.96xlarge (no metal capacity). The `machine`
  column (distacc_aaspo only) says which. A seed-1 smoke on blood-transfusion reproduced all 65 recorded seed-1 AUCs exactly
  on both types
- **Date:** 2026-11-20 18:14Z to 22:40Z (crash 21:22Z, resumed 22:30Z)
- **Code:** the 2026-11-06 Hao-seeds code snapshot and binaries (driver `ydf_reruns/run_kdd28_ydf_ec2.py` of 57d9fd14-dirty,
  binaries 0416519f-dirty, `bin_sha` ccd48156c4c4 as in the Hao rows), stamped `git=57d9fd14-dirty`, so YDF and Hao seeds share one build.
  Launch script: `ariel_notes/results/2026-11-20_ydf_seeds/run_ydf_seeds.sh`
- **Raw cells:** box `~/results/2026-11-20_ydf_seeds/spaa/seed<s>/` on ariel-ydf-1 (stopped), mirrored to the gitignored
  `ariel_notes/results/2026-11-20_ydf_seeds/box/` (CSVs, logs, pre-fix CSV backups, `ydf_seeds_models.tar.zst` md5 2732dc38)
- **Read by:** tab:accuracy-hao-vs-ydf-params-single-machine, tab:accuracy-hao-vs-ydf-params-distributed (YDF columns, seeds 1-5)
