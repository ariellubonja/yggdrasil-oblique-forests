# PR #409 (lazy-candidate-shuffle) — seed-noise study, 2026-08-14

> **Replication, 2026-08-30** — the full 100-seed × {gcc, icx} × {base, head}
> matrix was rebuilt from scratch (fresh binaries, same worktree/commits,
> `--config=linux_avx2 --dynamic_mode=off`, gcc 13.3 / icx via oneAPI) and
> rerun: **all 1,200 metric values are bit-identical** to
> `results_matrix_100seeds.csv` (`sort`ed files `cmp` equal;
> `results_matrix_100seeds_rerun_2026-08-30.csv`). The derived table below
> therefore reproduces exactly (t = +0.11/+0.49/−0.73 gcc, +0.26/+0.58/−0.71
> icx; base out-of-band 12/10/12 gcc, 5/9/12 icx). The runs are fully
> deterministic per (compiler, commit, seed) — the study's residual noise is
> seed choice, not run-to-run jitter.
>
> Reproduce: `seed_study_test.patch` (applied on top of `01417698` or
> `21591b7e` in a worktree), build the test target with the flags above, then
> `run_seeds.sh <binary> <compiler> <arm> <out.csv>` (100 seeds = `default` +
> 3001–3099, 8-way parallel, cwd = worktree root since test data paths are
> cwd-relative) and `compare.py results_matrix_100seeds.csv <out.csv>...`.

> **Update, same day — 100 seeds × {gcc, icx} × {base, head}** (1,200 runs,
> `results_matrix_100seeds.csv`, figure `pr409_seed_study.png`):
>
> | Test | gcc PR−base (t) | icx PR−base (t) | base runs outside current band |
> |---|---|---|---|
> | Abalone SPO RMSE | +0.00005 (+0.11) | +0.00011 (+0.26) | gcc 12/100, icx 5/100 |
> | Adult NWTA+w acc | +0.00014 (+0.49) | +0.00017 (+0.58) | gcc 10/100, icx 9/100 |
> | SimPTE Qini | −0.00013 (−0.73) | −0.00012 (−0.71) | gcc 12/100, icx 12/100 |
>
> Conclusions: (1) **no PR effect under either compiler** — all |t| ≤ 0.73 at
> n=100/arm; (2) **the checked-in bands clip 5–13 % of runs on unmodified
> upstream master**; (3) compiler arithmetic alone moves the Abalone default-seed
> RMSE by −0.008 (gcc 2.06416 → icx 2.05646): SimPTE models are bit-identical
> across compilers, Adult near-identical, but the oblique dot-product is
> fp-order-sensitive. icx's default `-fp-model=fast` even trips the tests'
> strict `internal_error_on_wrong_splitter_statistics` check on Abalone
> (splitter-vs-partition rounding mismatch) — the study demotes it to the
> production default (warning) in the patched tests.
>
> Study knobs: `YDF_TEST_SEED` → `train_config_.random_seed`; `check_model=false`
> (skips save/reload + engine re-evals, metric unaffected); binaries built
> `--dynamic_mode=off` so all four coexist (cc_test links dynamically by default).

## Original 10-seed study (gcc only)

Question: are the metric shifts in upstream `random_forest_test.cc` under the lazy
Fisher-Yates shuffle (CI failure `RandomForestOnAbalone.SparseOblique`, plus the
SimPTE Qini and Adult NoWinnerTakeAllWithWeights deltas) a real quality regression
or seed noise?

Setup: worktree at PR head `21591b7e` vs base `01417698` (upstream master).
Tests patched to take `YDF_TEST_SEED` (sets `train_config_.random_seed`; the
train/test split is deterministic and unaffected) and print their metric.
Build: gcc 13.3, `--config=linux_avx2` (matches CI; local default-seed Abalone
RMSE 2.064157 reproduces the CI value bit-for-bit). 10 seeds per arm
(default=123456 + 1001..1009).

Results (mean ± sd over 10 seeds):

| Test | metric | base | head | head−base | t |
|---|---|---|---|---|---|
| AbaloneSPO | RMSE | 2.06181 ± 0.00524 | 2.06114 ± 0.00249 | −0.00067 | −0.36 |
| AdultNWTAWeights | acc | 0.83418 ± 0.00241 | 0.83314 ± 0.00227 | −0.00104 | −1.00 |
| SimPTELowerBound | Qini | 0.10783 ± 0.00125 | 0.10825 ± 0.00113 | +0.00042 | +0.80 |

Conclusions:
1. **No systematic regression.** Mean shifts are ≤0.4σ of the seed distribution,
   with mixed sign (head is *better* on Abalone RMSE and Qini). Consistent with
   theory: the lazy prefix Fisher-Yates yields exactly the same distribution over
   tested-attribute subsets as the eager `std::shuffle`; only the RNG stream
   changes, so a fixed seed picks a different sample from the same model
   distribution.
2. **The test expectations are seed-brittle.** On *unmodified upstream master*,
   3/10 seeds fail the Abalone threshold (2.054±0.01) and 2/10 fail SimPTE
   (0.10889±0.002). The constants encode one RNG stream, not model quality.
3. The Adult expectation 0.8415 is stale for OSS gcc builds: even upstream master
   never exceeds 0.8377 in 10 seeds here (mean 0.834). The "0.8415 → 0.8287 high
   accuracy loss" fear conflates the stale constant with a real drop; the real
   base-vs-head gap is −0.001 (0.4σ… within noise, t=−1.0).
