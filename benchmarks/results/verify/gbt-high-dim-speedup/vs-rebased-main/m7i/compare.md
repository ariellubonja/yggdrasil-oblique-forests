# compare.md — `gbt-high-dim-speedup` (`--config=skip_dead_axis_jobs`) vs `rebased-main` — m7i verdict

**Verdict machine: AWS m7i (Intel Xeon Platinum 8488C, 48 vCPU, 384 GB), icx 2025.2.1, both arms.**
Full protocol: GBT, 300 trees, median of 3 runs, default dataset list with **15k×400k replaced by the
same-regime 15k×40k** (user decision 2026-09-07: the 15k×400k arm A costs ~2 h/run and a prior full-size
result already exists — see "Prior 15k×400k" below). Trees are **bit-identical** (see Model equivalence), so
the timing is the whole story.

The candidate's code (`1504202a v1 speedup GBT high-dimensional datasets`) is already merged into
`rebased-main`; the change is flag-gated, so both arms are built from the same source tree at the
`rebased-main` tip `b52f876e` (candidate branch rebased onto it = `5cf2e9a4`, which adds only the Mac
quick-pass result files). Arm A = no config, arm B = `EXTRA_BAZEL_CONFIGS=--config=skip_dead_axis_jobs`.
Binaries: A `f5261603…`, B `27846335…` (sha256 in `binaries_sha256.txt`; the same binaries passed the
identity check).

## Full protocol — A/B runtime (300 trees, median of 3)

| dataset | A median s ± sd | B median s ± sd | speedup A/B | time saved | verdict |
|---|---:|---:|---:|---:|---|
| HIGGS 11M×29 | 1334.37 ± 5.6 | 1294.00 ± 1.8 | 1.03× | +3.0 % | no significant change |
| trunk 1.5M×4096 | 319.01 ± 1.7 | 304.71 ± 3.2 | 1.05× | +4.5 % | no significant change |
| trunk 150k×40k | 738.18 ± 14.9 | 47.83 ± 0.6 | **15.43×** | +93.5 % | ★ speedup |
| trunk 15k×40k (same-regime stand-in for 15k×400k) | 732.23 ± 9.0 | 8.74 ± 0.06 | **83.74×** | +98.8 % | ★ speedup |

Geometric mean over the 4 cells: **6.11×** (+83.6 % time saved). Gates: < 15 % saved = failed, ≥ 20 % = ★.
Generated table with provenance: `compare_full_t300.md`. CSVs: `m7i_gbt_hd_{a,b}_dyn_t300_r3.csv`.

Reading: the dead axis-aligned jobs cost ≈ nodes × F × ~1 µs, mutex-serialized in the concurrent split
manager, so the saving grows with the feature count and is invisible at F = 29 (HIGGS) and small at F = 4096.
At F ≥ 40k it is the whole run time: 150k×40k drops from 12.3 min to 48 s, 15k×40k from 12.2 min to 8.7 s.

### Quick pass (30 trees, 1 run) — `compare_quick_t30.md`, `compare_small_t30.md`

| dataset | A s | B s | speedup |
|---|---:|---:|---:|
| HIGGS | 140.38 | 135.07 | 1.04× |
| trunk 1.5M×4096 | 33.07 | 31.23 | 1.06× |
| trunk 150k×40k | 78.11 | 4.89 | 15.96× |
| trunk 15k×400k | 745.00 | 5.73 | 130.03× |
| trunk 15k×40k | 76.12 | 0.95 | 80.0× |

The 30-tree pass predicts the 300-tree ratios within noise (15.96× vs 15.43×, 80× vs 84×). Its
15k×400k cell (745 s → 5.7 s, 130×) is the only m7i measurement of the change on that shape at this
protocol.

### Prior 15k×400k, full size (not re-run; different code states, direction only)

| arm | source | protocol | median s |
|---|---|---|---:|
| A (no config) | `GBT/gbt_e2e.csv`, 2026-07-19, `1b46018a-dirty` | Boosting, Dynamic Random Histogram thresh 1350, 300 t, 3 runs | 7408.3 |
| A (no config) | same file | Boosting, Random histogram | 6944.97 |
| B (config) | `gbt_fix_high_dim_mutex_exact.csv`, 2026-08-31, `f8f6ed86` | Boosting, **Exact**, 300 t, 3 runs | 56.89 |
| A (no config) | `GBT/gbt_e2e.csv`, 2026-07-19 | Boosting, **Exact** | 7596.19 |

Exact-vs-Exact across those two dates: 7596 → 57 s (≈133×), consistent with today's 30-tree 130× under
the Dynamic protocol.

## Replication of arm A (tolerance 3 %)

`find_prior_baseline.py` finds no earlier CSV with today's exact `EXTRA_TRAIN_ARGS` string (older runs
passed `--ensemble_method=Boosting` alone and the parser labelled the finder differently), so the nearest
prior of the same protocol is compared by hand: `GBT/gbt_e2e.csv`, 2026-07-19, m7i, 300 trees, 3 runs,
rows `SPORF_Dynamic_Random_Histogram_thresh1350` (same finder, threshold 1350 instead of today's default
250) and `SPORF_Random`.

| dataset | prior 2026-07-19 (Dyn thresh1350) | prior (Random) | today arm A | drift vs Dyn |
|---|---:|---:|---:|---:|
| HIGGS | 1336.41 | 1334.28 | 1334.37 | −0.2 % |
| trunk 1.5M×4096 | 317.83 | 315.63 | 319.01 | +0.4 % |
| trunk 150k×40k | 754.25 | 755.44 | 738.18 | −2.1 % |

All within the 3 % tolerance ⇒ machine and code state are as in July; B's speedup is attributable to the
flag. (The 30-tree arm A also lands at 1/10 of these within 5 %: 140 / 33 / 78 / 745 s.)

## Model equivalence (established before timing)

- `bitid_t30.md`: `ydf_bitid_cc18.sh bin_A bin_B`, 10 CC18 tasks, fold 0, GBT Dynamic, 30 trees —
  **10/10 TREES IDENTICAL** (`nodes-*` sha256 match; only the GBT header proto differs, which is
  run-to-run metadata, as the 2026-09-05 A-vs-A control showed).
- `accuracy.sh` A vs B at 30 trees and again at **300 trees** (34 CC18 tasks × 10 folds): accuracy, AUC
  and logloss CSV bodies **byte-identical** (`accuracy_m7i_gbt_hd_{a,b}_dyn_t{30,300}*.csv`).

No MODEL CHANGED warning applies.

## Commands and environment

```bash
# arms (bench_common's bazel_build: icx pin, -c opt -O3 -march=native, post-build compiler check)
EXTRA_BAZEL_CONFIGS=""                            build_arm.sh bin_A_fork
EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" build_arm.sh bin_B_fork
# protocol (all runs)
EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"'
# quick pass: fork_quick.sh  (NUM_TREES_DIVISOR=10, runtime.sh --runs=1, bitid, accuracy)
# small shape + full protocol: phase2.sh  (TRUNK_DATASETS_OVERRIDE="1500000|4096 150000|40000 15000|40000",
#   NUM_TREES_DIVISOR=1, runtime.sh --runs=3, accuracy.sh at 300 trees)
```

Runs were serialized in a dedicated tmux session (`verify_fork`, `verify_p2`); no other benchmark process
was running (checked with pgrep before each launch). No benchmark script default was changed; the only
knobs used are `EXTRA_BAZEL_CONFIGS`, `EXTRA_TRAIN_ARGS`, `NUM_TREES_DIVISOR`, `--runs`,
`CSV_DATASETS_OVERRIDE`, `TRUNK_DATASETS_OVERRIDE`.

## Not done

- 15k×400k at 300 trees / 3 runs (arm A ≈ 2 h per run) — excluded by the user; the 30-tree cell and
  the prior full-size results above stand in.
- The Mac quick pass of 2026-09-05 (`../A.csv`, `../B.csv`, `../compare.md`) is superseded by this
  directory for any verdict; it remains as the pipeline-validation record.
