# runs/2026-11-20_cc18_seeds

Measurement, stored once. Floats read it through their `inputs.txt`. Index and rules: `benchmarks/results/kdd28/README.md`.

- **Files:** `<profile>/seed<s>/cc18/results.csv`, profile kdd28_ydf and kdd28_hao, s = 1-5, as the driver wrote them. 1,020
  rc = 0 cells each (34 CC18 binary datasets x 10 folds x arms cc18_aa, cc18_spo, cc18_aa_spo)
- **Profile:** `kdd28_ydf.cc18_{aa,spo,aa_spo}` (seeds widened to 1-5 on 2026-11-20) and `kdd28_hao.cc18_{aa,spo,aa_spo}` (new
  2026-11-20). YDF P = min(F², 6000), density 2. Hao P = min(F, 1000), density 1.5 (explicit flags, the arm-(c) argv form).
  Hao AA = YDF AA argv: equal in 1,700/1,700 cells (340 per seed). The table shows only the YDF AA column
- **Settings:** `params.py show <arm>`. Threads 4. No early stopping, no validation split, 300 trees, depth 6
- **Eval suite:** `kdd28_cc18_10fold` (34 CC18 binary, OpenML 10 CV folds, seeds widened to 1-5 on 2026-11-20)
- **Machine:** EC2 ariel-ydf-1 (c8i.96xlarge, YDF seeds 1-3, Hao 1-5) and ariel-ydf-7 (c8i.metal-96xl, YDF seeds 4-5),
  both Ubuntu 24.04, same binaries (md5 2eaadddf). Shared-box timing, ballpark only
- **Date:** 2026-11-21 00:37Z to 01:14Z
- **Code:** driver `accuracy_multi_thread/run_kdd28_ydf_ec2.py` of the working tree on 2026-11-20 (cc18 table: `--seed`, Hao
  explicit SPO flags) over the 2026-11-06 code snapshot. Binaries 0416519f-dirty (`bin_sha` ccd48156c4c4), stamped
  `git=57d9fd14-dirty` as the 2026-11-06 Hao and 2026-11-20 YDF seeds. Launcher `ariel_notes/results/2026-11-20_cc18_seeds/run_cc18_seeds.sh`
- **Checks:** kdd28_ydf seed 1 vs the 2026-11-03 seed-1 cc18 cells (other binary, m7i): 1,015/1,020 equal. The 5 others
  (pc4 folds 1, 3, 5 cc18_aa_spo, kc2 folds 3, 7 cc18_spo) are run-to-run nondeterminism, not binary or machine: repeating
  pc4 f1 cc18_aa_spo 12 times on one box gave 2 distinct results with 4 threads and 3 with 1 thread, kc2 f3 cc18_spo gave 3
  (logs `ariel_notes/results/2026-11-20_cc18_seeds/verify_diff_logs.tar`). Cause: hypothesis, untested
- **Raw cells:** box `~/results/2026-11-20_cc18_seeds/` on ariel-ydf-1 (stopped, YDF seeds 4-5 under `box7/`), mirrored to the
  gitignored `ariel_notes/results/2026-11-20_cc18_seeds/` (`box/` CSVs + logs, `cc18_seeds_models.tar.zst`)
- **Read by:** tab:accuracy-hao-vs-ydf-params-single-machine (CC18 continuation body)
