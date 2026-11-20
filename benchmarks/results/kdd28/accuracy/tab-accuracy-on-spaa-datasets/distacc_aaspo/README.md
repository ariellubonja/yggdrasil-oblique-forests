# tab:dist-accuracy-ydf extra single-machine columns (2026-11-04)

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
- Body: the generator command in `../README.md` (since 2026-11-04 the distacc CSV is tracked in `../distacc/`).
