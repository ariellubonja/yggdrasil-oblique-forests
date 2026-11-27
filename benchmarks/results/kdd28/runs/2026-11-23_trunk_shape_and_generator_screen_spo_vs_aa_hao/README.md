# runs/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao

Measurement, stored once. Screening study, read by no float. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Question:** on which synthetic trunk shapes and generator settings does single-machine SPO-GBT (Hao's settings) beat
  AA-GBT, by rule F of `floats/accuracy/tab-accuracy-hao-vs-ydf-params-single-machine/select_spo_clearly_beats_aa.py`
  (user request 2026-11-23).
- **Files:** `<profile>/seed<s>/distacc_stamped/results.csv` as the driver wrote them, all rc = 0 (3580 rows).
  `kdd28_hao` = single:SPO, `kdd28_ydf` = single:AA. `derived/` = output of `select_trunk_screen.py` (rule F, BH (a) over
  the 78 trunk rows, (b) pooled with the 101 existing kdd28_hao rows). `what_changed.csv` + `fold_meta/<trunk>/meta.json`
  = generator settings, solved signal scale m, oracle AUC on the generated rows. `launch/` = run scripts and the pipeline
  check cells. `offpaper_k1_column_reduced_aa_spo/` = off-paper diagnostic (below).
- **Data:** `benchmarks/data/prep_trunk_generator_screen_folds.py` (grid in `grid()`, 78 trunks, fold sha256 in
  `kdd28_trunk_generator_screen_manifest.sha256`, 780 files). Gaussian trunk: signal on the first k <= 256 columns along
  mu_j = 1/sqrt(j), AR(1) correlation rho among them, other columns N(0,1). shift = classes at +-m mu. mix = class 0 at 0,
  class 1 at +m mu or -m mu. m solved so the Bayes test AUC equals the target (0.75 / 0.90 / 0.97). Balanced 0/1 labels.
  Rows {1k, 4k, 16k, 100k} x cols {256 .. 4096}. 5 StratifiedKFold folds seed 0 (`run_suite.make_folds`). No NaN, no
  categoricals. A second build on the box reproduced 60/60 fold files. A Mac build (numpy 1.26.4) differs in 2 of 200k
  cells by 1 float32 ulp (Gaussian tail values). Runs read the box files (numpy 2.4.4, pandas 3.0.2, scikit-learn 1.9.0)
- **Profile:** SPO `kdd28_hao.distacc_single_spo` (exponent 1, max 1000, density 1.5, BINARY), AA `kdd28_ydf.distacc_single_aa`.
  Hao's 1000-projection cap kept for F >= 1024 (user decision 2026-11-23), so those trunks have P = 1000 < F
- **Settings:** trees=300, max_depth=6, min_examples=5, split=EXACT, validation_set_ratio=0, early_stopping=NONE, 4 threads
- **Eval suite:** `kdd28_trunk_generator_screen`. Seed 1 on every trunk, seeds 2-5 where the seed-1 effect is > 0 (70 trunks).
  `params.py check-eval --suite kdd28_trunk_generator_screen`: every row's data_sha matches the manifest
- **Machine:** EC2 ariel-ydf-4 (c8i.metal-96xl, SMT + turbo on, stopped after the run). Up to ~640 CPU slots on 384 vCPUs
  during seeds 2-5: train_s is oversubscribed shared-box time, ballpark only
- **Date:** 2026-11-23 11:08Z to 13:38Z
- **Code:** git clone at `origin/main` d43e0809 + uncommitted suite entry and generator, stamped `d43e0809-dirty`. Binaries
  `bazel build -c opt` (icx) of d43e0809 on ariel-ydf-1 worktree `~/wt/trunk-screen`, md5 train_oblique_forest 62359b37
- **Checks:** pipeline check before new data: madeline + jasmine (kdd28_wide_real_candidates), 5 folds, seed 1, SPO and
  AA: 20/20 cells equal the 2026-11-22 recorded test AUC to every printed digit
- **Result:** rule F passes (a) 44/78, (b) 40/78 (`derived/trunk_report.txt`). Largest edges: mix at 256 cols (+0.052 AUC),
  1k-row shift trunks (+0.03 to +0.04). Edge grows with informative columns (k=256: 28/40 pass, k <= 16: 10/32) and fewer
  rows. Correlation shrinks it (rho 0.9: 1/5). No edge: mix at 4k x 1024 (-0.004 to -0.012), 100k-row trunks with k <= 16 or
  rho 0.9 (|effect| <= 0.0015), 100k x 4096 (+0.002 to +0.003, under the 0.005 floor)
- **k = 1 controls pass at 4k x 1024** (+0.011, +0.006). Off-paper test of "AA overfits the 1023 noise columns": AA on copies
  with only the first 32 or 128 columns, seed 1, 5 folds: no gain (auc75: 0.703 -> 0.702 / 0.695, SPO 0.709). SPO on the
  same copies keeps its edge (0.703 / 0.706, auc90: 0.877 / 0.876 vs AA 0.868 / 0.870). Refuted. Mechanism untested.
  The harness refused `--num_candidate_attributes` with `--ydf_defaults` (logs `*_aa_nca*.log`), so the test used column copies
- **Raw cells:** gitignored `ariel_notes/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao/box/`
  (`box_all_but_fold_data.tar.zst` with models, sha256 e108ea05...). Box copy: `~/results/2026-11-23_trunk_shape_and_generator_screen_spo_vs_aa_hao/`
  on ariel-ydf-4 (fold files under `data/folds`, 160 GB, rebuilt by the prep script)
- **Read by:** no float
