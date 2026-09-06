# compare.md — `origin/gbt-high-dim-speedup` (`--config=skip_dead_axis_jobs`) vs `origin/rebased-main`

**Mac M1 Pro, mechanics only, no verdict.** Reduced pass: GBT protocol, 30 trees (`NUM_TREES_DIVISOR=10`),
1 run per dataset (`--runs=1`), both arms from one detached worktree at `88fa4ba3` + the candidate patch
(uncommitted); arm A built without the config, arm B with `EXTRA_BAZEL_CONFIGS=--config=skip_dead_axis_jobs`.
Arm A's default-list CSV is assembled from two passes (laptop slept during pass 1; see report.md).

## Default dataset list — A/B runtime: A: origin/rebased-main 88fa4ba3 (+patch, no config) vs B: +--config=skip_dead_axis_jobs

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Dynamic_Random_Histogram | 164.27 ± n/a | 163.05 ± n/a | 1.01× | +0.7 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Dynamic_Random_Histogram | 440.40 ± n/a | 524.25 ± n/a | 0.84× | -19.0 % | SLOWDOWN |
| trunk_150000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 162.67 ± n/a | 144.90 ± n/a | 1.12× | +10.9 % | no significant change |
| trunk_15000_x_400000 | SPO-GBT_Dynamic_Random_Histogram | 957.15 ± n/a | 63.83 ± n/a | 14.99× | +93.3 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **1.94×** (+48.5 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-05T18:20:17Z | 2026-09-05T18:43:00Z |
| git_sha | 88fa4ba3-dirty | 88fa4ba3-dirty |
| git_branch |  |  |
| machine | Apple M1 Pro (nproc=10) | Apple M1 Pro (nproc=10) |
| compiler | Apple clang version 21.0.0 (clang-2100.1.1.101) | Apple clang version 21.0.0 (clang-2100.1.1.101) |

### Same-regime small shape (skill step 3 fallback; fits in RAM, no swapping)

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| trunk_15000_x_40000 | SPO-GBT_Dynamic_Random_Histogram | 187.14 ± n/a | 2.36 ± n/a | 79.32× | +98.7 % | ★ speedup |

Geometric-mean speedup over 1 dataset(s): **79.32×** (+98.7 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.


Reading: HIGGS (F = 29) unchanged, as the cost model predicts (dead axis work ≈ nodes × F). The two 24 GB
shapes on this 32 GB Mac ran while swapping (10.9 GB swap already in use; ≈14 GB free before the runs):
**trunk 1.5M × 4096's −19 % is confounded by swap** and must not be read as a regression (the 2026-08-07
laptop run, 64 GB-class, in RAM, gave 1.19× on the same shape); 150k × 40k's +11 % is below the 15 % gate.
The wide/short shapes carry the win: 15k × 400k 14.99×, 15k × 40k 79×. Geometric mean over the default
list 1.94× (with the confounded cell included).

## Replication

`find_prior_baseline.py --like <arm A>` → exit 2: **no prior CSV** under `benchmarks/results` matches this
protocol (Apple M1 Pro, `<none>` configs, `--ensemble_method Boosting --numerical_split_type "Dynamic Random
Histogram"`, 30 trees), so there is no replication statement. Nearest earlier measurement of this candidate:
`benchmarks/results/GBT/skip_dead_axis_jobs_2026-08-07.csv` (Alienware 185H, icx, 10 trees): 15k × 400k
training block 282.4 → 4.8 s (58.9×), e2e 3.96×; 1.5M × 4096 21.2 → 17.7 s (1.19×); trees identical —
directionally consistent with today's numbers.

## Identity (trees) — `ydf_bitid_cc18.sh bin_A bin_B`, GBT protocol, 30 trees, fold 0

| task | trees | result |
|---|---:|---|
| task_10093_banknote-authentication | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_37_diabetes | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_9946_wdbc | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_3917_kc1 | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_3902_pc4 | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_9957_qsar-biodeg | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_43_spambase | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_9976_madelon | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_9910_Bioresponse | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |
| task_9952_phoneme | 30 | **DIFFER** (RESULT: TREES IDENTICAL ✅ but metadata differs ⚠️  (nodes-* match; other files differ above)) |

`compare_models.sh` reports "TREES IDENTICAL but metadata differs" on all 10 tasks: `nodes-*` (topology,
thresholds, oblique weights) and `gradient_boosted_trees_header.pb` **match**, only `header.pb` differs.
**Control** (`bitid_control/`): two runs of `bin_A` alone on banknote also differ in `header.pb` (and
`data_spec.pb`) with identical `nodes-*` → the metadata delta is run-to-run (timing/log fields), not the
change. **Verdict: bit-identical trees on all 10 tasks**, as the candidate claims. The full `accuracy.sh`
sweep was therefore not run (bit-identical trees imply identical accuracy CSV bodies).

## Provenance blocks

### Arm A (combined; header from part 2, `assembled_from:` added)
```
==== PROVENANCE ====
date_utc: 2026-09-05T18:20:17Z
git_sha: 88fa4ba3-dirty
git_branch: 
machine: Apple M1 Pro (nproc=10)
machine_serial: TGQDC24M17
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
EXTRA_BAZEL_CONFIGS: <none>
EXTRA_TRAIN_ARGS: --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"
NUM_TREES_DIVISOR: 10
NUM_TREES: 30  NUM_RUNS: 1
CSV_DATASETS: <none>
TRUNK_DATASETS: 150000|40000 15000|400000
assembled_from: part1 (2026-09-05T16:53Z pass killed by laptop sleep during 150k x 40k; HIGGS + 1.5M x 4096 rows) + part2 (150k x 40k, 15k x 400k)
====================
```
### Arm A part 1 (kept log, HIGGS + 1.5M x 4096)
```
==== PROVENANCE ====
date_utc: 2026-09-05T16:53:04Z
git_sha: 88fa4ba3-dirty
git_branch: 
machine: Apple M1 Pro (nproc=10)
machine_serial: TGQDC24M17
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
EXTRA_BAZEL_CONFIGS: <none>
EXTRA_TRAIN_ARGS: --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"
NUM_TREES_DIVISOR: 10
NUM_TREES: 30  NUM_RUNS: 1
CSV_DATASETS: benchmarks/data/HIGGS_with_header.csv|class
TRUNK_DATASETS: 1500000|4096 150000|40000 15000|400000
====================
```
### Arm B
```
==== PROVENANCE ====
date_utc: 2026-09-05T18:43:00Z
git_sha: 88fa4ba3-dirty
git_branch: 
machine: Apple M1 Pro (nproc=10)
machine_serial: TGQDC24M17
compiler: Apple clang version 21.0.0 (clang-2100.1.1.101)
EXTRA_BAZEL_CONFIGS: --config=skip_dead_axis_jobs
EXTRA_TRAIN_ARGS: --ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"
NUM_TREES_DIVISOR: 10
NUM_TREES: 30  NUM_RUNS: 1
CSV_DATASETS: benchmarks/data/HIGGS_with_header.csv|class
TRUNK_DATASETS: 1500000|4096 150000|40000 15000|400000
====================
```

Binaries: `bin_A` sha256 9221a71c… (re-verified identical after the part-2 rebuild), `bin_B` sha256 199569e1….
Toolchain actually used by Bazel: Homebrew LLVM clang 19.1.7 (`local_config_cc`), same output base for both
arms; the header's `compiler:` field (`cc --version` = Apple clang 21) is mislabelled on macOS.

## Exact commands and environment

```
# worktree (detached, no commits)
git worktree add --detach <scratchpad>/gbt-hd-verify origin/rebased-main        # 88fa4ba3
cd <scratchpad>/gbt-hd-verify && git cherry-pick --no-commit f7906361 f8f6ed86
cp <repo>/benchmarks/evaluation/runtime.sh benchmarks/evaluation/; cp <repo>/benchmarks/utils/{bench_common.sh,parse_log_to_csv.py} benchmarks/utils/
ln -s <repo>/benchmarks/data/HIGGS_with_header.csv benchmarks/data/
export NUM_TREES_DIVISOR=10
export EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"'
# arm A (no EXTRA_BAZEL_CONFIGS)
CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="15000|40000" bash benchmarks/evaluation/runtime.sh --runs=1 mac_gbt_hd_A_small_t30_r1
bash benchmarks/evaluation/runtime.sh --runs=1 mac_gbt_hd_A_default_t30_r1            # killed by sleep after HIGGS, 1.5M x 4096
CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="150000|40000 15000|400000" bash benchmarks/evaluation/runtime.sh --runs=1 mac_gbt_hd_A_default_t30_r1_part2
# arm B
export EXTRA_BAZEL_CONFIGS="--config=skip_dead_axis_jobs"
CSV_DATASETS_OVERRIDE=none TRUNK_DATASETS_OVERRIDE="15000|40000" bash benchmarks/evaluation/runtime.sh --runs=1 mac_gbt_hd_B_small_t30_r1
bash benchmarks/evaluation/runtime.sh --runs=1 mac_gbt_hd_B_default_t30_r1
# identity (from the repo root, same env)
NUM_TREES_DIVISOR=10 EXTRA_TRAIN_ARGS=... .claude/skills/verify-speedup/scripts/ydf_bitid_cc18.sh bin_A bin_B bitid/
```
Runner binary flags come from the scripts (`--num_trees=30`, defaults otherwise); nothing in the scripts
was edited. Idle Bazel servers were shut down before the runs and the worktree's server right after each
build. No keep-awake wrapper. One benchmark process at a time throughout.

## Caveats and what was not done

- Mac = pipeline validation only; numbers are not a verdict. No x86 SIMD paths; no ISA banner on the GBT path.
- One run per cell (no stddev); 30 trees, not 300; three cells ran under swap pressure (24 GB shapes).
- Full protocol (300 trees × 3 runs) not run: every reduced run exceeded the 3-minute rule
  (164–957 s), so the skill stops here; the m7i is where the verdict comes from (reference predicts arm A
  15k × 400k ≈ 2 h per run at full size; arm B should cut that by ≈ 90 %).
- No replication (no matching prior). `accuracy.sh` not run (trees bit-identical).
- Arm A default-list CSV assembled from two passes ~1.5 h apart (same binary, same env).
- Results not committed; copies placed (untracked) in `benchmarks/results/verify/gbt-high-dim-speedup/`.
