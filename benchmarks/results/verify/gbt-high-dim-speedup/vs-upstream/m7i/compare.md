# compare.md — `gbt-high-dim-speedup` (`--config=skip_dead_axis_jobs`) vs `upstream/main` — m7i verdict

**Verdict machine: AWS m7i (Intel Xeon Platinum 8488C, 48 vCPU, 384 GB), icx 2025.2.1, both arms.**
Full protocol: GBT, 300 trees, median of 3 runs, default dataset list with **15k×400k replaced by the
same-regime 15k×40k**, plus **one full-size run per arm of 15k×400k** (user decision 2026-09-07).
Trees are **bit-identical** (see Model equivalence), so the timing is the whole story.

**Protocol is Exact vs Exact.** `upstream/main` (google/yggdrasil-decision-forests, tip `df16834d`,
2026-09-03; still the tip on 2026-09-07) has no sparse-oblique histogramming, so the fork's default
`Dynamic Random Histogram` finder does not exist there. The Google PR adding it (expected the week of
2026-09-07) had **not** landed when this ran. Consequently this pass measures only the
`skip_dead_axis_jobs` change on the upstream code path; it says nothing about the fork's histogram
advantage over upstream.

### Arms

| arm | branch | HEAD | build config |
|---|---|---|---|
| A (baseline) | `upstream-main-benchmarks` | `9bd86619` = `upstream/main` `df16834d` + one tooling commit (trimmed harness, scripts) | none |
| B (candidate) | `upstream-main-benchmarks-gbt-high-dim-fix` | `368162ea` = A + cherry-picked `a099367a`, `368162ea` (the fork's `1504202a`/`f8f6ed86` ported; same 3 files, +72/−4 lines as on the fork) | `--config=skip_dead_axis_jobs` |

Binaries: A `00c425c9…`, B `6ed72d6b…` (sha256 in `binaries_sha256.txt`; the same binaries passed the
identity check). Port evidence (diff, conflicts resolved) is in `../ported_*` from the 2026-09-05 Mac pass;
the branch was reused unchanged.

## Full protocol — A/B runtime (300 trees, median of 3)

| dataset | A median s ± sd | B median s ± sd | speedup A/B | time saved | verdict |
|---|---:|---:|---:|---:|---|
| HIGGS 11M×29 | 1759.77 ± 4.3 | 1726.80 ± 0.9 | 1.02× | +1.9 % | no significant change |
| trunk 1.5M×4096 | 436.87 ± 1.2 | 406.19 ± 17.1 | 1.08× | +7.0 % | no significant change |
| trunk 150k×40k | 737.11 ± 18.7 | 59.34 ± 0.2 | **12.42×** | +91.9 % | ★ speedup |
| trunk 15k×40k (same-regime stand-in) | 765.55 ± 1.1 | 10.37 ± 0.02 | **73.82×** | +98.6 % | ★ speedup |

Geometric mean over the 4 cells: **5.63×** (+82.2 % time saved). Gates: < 15 % saved = failed, ≥ 20 % = ★.
Generated table with provenance: `compare_full_t300.md`. CSVs: `m7i_upb_gbt_hd_{a,b}_exact_t300_r3.csv`.

### 15k×400k, full size (300 trees), one run per arm — `m7i_upb_gbt_hd_{a,b}_exact_t300_r1_huge.csv`

| dataset | A s (1 run) | B s (1 run) | speedup A/B |
|---|---:|---:|---:|
| trunk 15k×400k | 7513.67 | 139.08 | **54.02×** (+98.1 %, ★) |

Single runs, so no spread; the 3-run cells above put run-to-run spread at ≤ 2.5 % of the median for the
big shapes.

### Quick pass (30 trees, 1 run) — `compare_quick_t30.md`, `compare_small_t30.md`

| dataset | A s | B s | speedup |
|---|---:|---:|---:|
| HIGGS | 183.83 | 176.40 | 1.04× |
| trunk 1.5M×4096 | 45.53 | 41.42 | 1.10× |
| trunk 150k×40k | 73.42 | 6.10 | 12.04× |
| trunk 15k×400k | 792.91 | 14.45 | 54.86× |
| trunk 15k×40k | 77.22 | 1.09 | 70.8× |

The 30-tree ratios predict the 300-tree ones within noise (12.04× vs 12.42×, 70.8× vs 73.8×).

## Replication of arm A

No earlier CSV of this protocol (upstream-main-benchmarks, GBT Exact, 300 trees) exists on the m7i; the
2026-09-05/06 Mac pass is a different machine and compiler. Arm A here is therefore the **first m7i point
of the upstream protocol** and the report claims no replication. Direction-only cross-check against the
fork's own Exact run of 2026-07-19 (`GBT/gbt_e2e.csv`, fork code `1b46018a`, same machine/protocol
shape): HIGGS 2006 ± 191 → 1760 today (upstream code), 1.5M×4096 424 → 437, 150k×40k 717 → 737,
15k×400k 7596 → 7514 (−1.1 %) — same order of magnitude, as expected for a code base that differs
from the fork's in the RF/histogram paths but not in the GBT split manager.

## Model equivalence (established before timing)

- `bitid_t30.md`: `ydf_bitid_cc18.sh bin_A bin_B`, 10 CC18 tasks, fold 0, GBT Exact, 30 trees —
  **10/10 TREES IDENTICAL** (`nodes-*` sha256 match; only the GBT header proto differs, run-to-run
  metadata per the 2026-09-05 A-vs-A control).
- `accuracy.sh` A vs B at 30 trees and again at **300 trees** (34 CC18 tasks × 10 folds): accuracy, AUC
  and logloss CSV bodies **byte-identical** (`accuracy_m7i_upb_gbt_hd_{a,b}_exact_t{30,300}*.csv`).

No MODEL CHANGED warning applies.

## Commands and environment

```bash
# arms (bench_common's bazel_build: icx pin, -c opt -O3 -march=native, post-build compiler check)
git checkout upstream-main-benchmarks-gbt-high-dim-fix; EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs" build_arm.sh bin_B_up
git checkout upstream-main-benchmarks;                  EXTRA_BAZEL_CONFIGS=""                             build_arm.sh bin_A_up
# protocol (all runs)
EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"'
# quick pass: upstream_quick.sh (NUM_TREES_DIVISOR=10, runtime.sh --runs=1, bitid, accuracy)
# small shape + full protocol + huge: phase2.sh
#   TRUNK_DATASETS_OVERRIDE="1500000|4096 150000|40000 15000|40000", NUM_TREES_DIVISOR=1, runtime.sh --runs=3,
#   accuracy.sh at 300 trees; then CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="15000|400000" --runs=1
```

The verify-speedup scripts were run from a copy outside the tree (they are not on the upstream branches).
Runs were serialized in a dedicated tmux session; no other benchmark process was running. No benchmark
script default was changed.

## Not done

- 15k×400k at 3 runs (one run per arm instead, user decision).
- Dynamic-histogram comparison against upstream: impossible until upstream gains sparse-oblique
  histogramming (check the Google PR; then update `references/ydf-fork.md`).
- The Mac quick pass of 2026-09-05/06 (`../compare.md`, `../report.md`) is superseded by this directory
  for any verdict.
