# Reference: ariellubonja/yggdrasil-decision-forests (oblique RF fork)

Everything repo-specific the workflow in SKILL.md needs. Verified 2026-09-05.

## Refs and baselines

| name | what | notes |
|---|---|---|
| `origin/rebased-main` | the fork's integration branch, default baseline | full instrumentation; default protocol available |
| `upstream/main` | google/yggdrasil-decision-forests | fetch with `git fetch upstream` (remote is unshallowed since 2026-09-05; true merge-base b506e72f, 2026-05-12) |
| `upstream-bench` | upstream/main + one tooling commit | trimmed harness + the three scripts. Refresh with `git rebase upstream/main`; only upstream-owned file touched is `examples/BUILD` |

Two baseline targets, two protocols:

1. **vs `origin/rebased-main`**: default protocol (below). Flag-gated changes
   are A/B'd on one tree via `EXTRA_BAZEL_CONFIGS`. Arm A must replicate the
   newest matching CSV under `benchmarks/results` (tolerance 3 %; same-protocol
   m7i runs a month apart drifted 0.2–1.0 %).
2. **vs `upstream/main`**: worktree on `upstream-bench`, cherry-pick the
   candidate. The trimmed harness has **no** Dynamic histogram types, row-major
   store, `RM_MAX_ROWS` or RF early exit. **SPORF comparisons are Exact vs
   Exact** because upstream has no sparse-oblique histogramming yet; a PR
   adding it is expected the week of 2026-09-07. **Tell the user this on every
   upstream run** and check whether that PR has landed (then update this file
   and the `speedup-verification-two-baselines` memory).

Upstream target recipe. On the m7i (clean, dedicated checkout) work in place:
```bash
git checkout upstream-bench                       # arm A
git checkout -b upstream-bench/<candidate> && git cherry-pick <base>..<candidate>   # arm B; resolve fork-only context
```
On a dev machine whose tree has uncommitted work, do the same inside
`git worktree add <dir> upstream-bench` with `ln -s <repo>/benchmarks/data <dir>/benchmarks/data`.
`benchmarks/results` is gitignored on upstream-bench; copy CSVs back into the fork's tree.
Refreshing `upstream-bench` after fork-side tooling changes: `git rebase upstream/main`,
copy the scripts over, regenerate the harness with
`python3 benchmarks/utils/make_trimmed_harness.py examples/train_oblique_forest.cc <wt>/examples/train_oblique_forest.cc`
(fork checkout as cwd), build, commit on `upstream-bench`.

## Scripts and knobs (never edit the scripts' defaults)

| script | job | knobs |
|---|---|---|
| `benchmarks/evaluation/runtime.sh [--runs=N] <suffix>` | e2e wall time, median of N (default 3), builds first | `EXTRA_BAZEL_CONFIGS`, `EXTRA_TRAIN_ARGS`, `CSV_DATASETS_OVERRIDE`, `TRUNK_DATASETS_OVERRIDE` ("none" skips a group) |
| `benchmarks/evaluation/accuracy.sh <suffix>` | CC18 10-fold held-out accuracy (34 tasks) | `EXTRA_BAZEL_CONFIGS`, `EXTRA_TRAIN_ARGS`, `ACCURACY_DATA_DIR` |
| `benchmarks/evaluation/compare_models.sh <dirA> <dirB>` | sha256 of saved models; exit 0 = bit-identical | — |
| `benchmarks/utils/bench_common.sh` | shared: icx pin (Linux), post-build compiler check, e-core toggle (185H only), provenance, tree count | `ONEAPI_SETVARS`, `NUM_TREES_DIVISOR` (default 1; **10 for every preliminary pass, on every machine**) |
| `benchmarks/utils/parse_log_to_csv.py` | log → CSV; derives `algorithm` from the command line | pass flags as `--flag "value"` or `--flag=value`; older CSVs may carry `SPORF_unknown`/mislabelled families, so match priors on provenance, not on `algorithm` |

Output: `benchmarks/results/<suffix>.csv` with a provenance head
(`date_utc, git_sha, git_branch, machine, machine_serial, compiler,
EXTRA_BAZEL_CONFIGS, EXTRA_TRAIN_ARGS, NUM_TREES_DIVISOR, NUM_TREES NUM_RUNS,
CSV_DATASETS, TRUNK_DATASETS`) then `dataset,algorithm,median_s,stddev_s`. There are no
`.meta` sidecars despite older docs. Tree count is set by the script:
5 × nproc for RF, 300 for Boosting; do not pass `--num_trees`.

The binary logs `Training block took: X s` (fork learners; the trimmed
harness prints it itself as TrainWithStatus wall time). `bench_common`
parses that line, not the harness wall-time lines.

## Protocol flags (put only these in `EXTRA_TRAIN_ARGS`)

| path the change touches | `EXTRA_TRAIN_ARGS` on rebased-main | on upstream-bench |
|---|---|---|
| RF sparse-oblique, default | `--numerical_split_type "Dynamic Random Histogram"` (bins 64, threshold 250 are the binary defaults) | `--numerical_split_type "Exact"` |
| RF Exact path (sort, scan) | `--numerical_split_type "Exact"` | same |
| GBT / concurrent split manager | `--ensemble_method Boosting --numerical_split_type "Dynamic Random Histogram"` | `--ensemble_method Boosting --numerical_split_type "Exact"` |
| axis-aligned | `--feature_split_type "Axis Aligned"` + split type | same |

Flag-gated candidates: read `git diff <base>..<cand> -- .bazelrc` for the
`build:<name>` line and pass `EXTRA_BAZEL_CONFIGS=--config=<name>` for arm B.
Example: `origin/gbt-high-dim-speedup` adds `build:skip_dead_axis_jobs`; it is
GBT-only, merges cleanly onto the tip, and was measured on the laptop on
2026-08-07 (10 trees): 3.96× e2e on 15k×400k Boosting, trees identical.

## Default datasets, RAM, and prior m7i durations (seconds per run)

| dataset | fp32 RAM | RF Dynamic (240 t) | RF Exact (240 t) | GBT Dynamic (300 t) | GBT Exact (300 t) |
|---|---:|---:|---:|---:|---:|
| HIGGS 11M×29 | 1.3 GB (+8 GB CSV) | 306 | 373 | 1332 | 2006 |
| trunk 1.5M×4096 | 24.6 GB | 280 | 342 | 318 | 424 |
| trunk 150k×40k | 24 GB | 80 | 93 | 749 | 717 |
| trunk 15k×400k | 24 GB | 24 | 27 | 7235 | 7596 |

Sources: `ablation_vectorized_dynamic/dfs_vectorized_dynamic.csv`,
`dfs_exact_hwy.csv`, `GBT/gbt_e2e.csv` (m7i, Xeon 8488C, 48 vCPU). GBT on
15k×400k is a 2-hour run at full size: the long-run protocol applies; the
same-regime smaller shape is `15000|40000`. On a 32 GB Mac the three 24 GB
trunk shapes swap or OOM: raise the critical warning, run anyway (user's
choice, 2026-09-05), and report OOM cells.

**Quick pass (decided 2026-09-05):** `NUM_TREES_DIVISOR=10 … runtime.sh --runs=1`
on the default datasets, both arms, on any machine. **Local machines (Mac,
Alienware) run only this quick pass; the full protocol (divisor 1, 3 runs,
accuracy sweep) runs on the m7i unless the user explicitly asks otherwise
(decided 2026-09-05).** Cost model for this
change class (GBT, concurrent split manager): dead axis-aligned work ≈ nodes ×
F × 1 µs, mutex-serialized, so 30 trees on 15k×400k still take ≈ 13 min in arm
A on any host; HIGGS (F = 29) shows no effect. Measured on the Mac at full
size, arm A: Bioresponse 35.7 s, Internet-Ads 25.1 s, madelon 10.1 s, nomao
6.5 s (0.85–1.07 µs per node·feature). Verdicts need divisor 1.

## Model equivalence (first, before any timing)

1. Keep a copy of each arm's binary (`bazel-bin/examples/train_oblique_forest`
   is overwritten by the next build).
2. `NUM_TREES_DIVISOR=10 EXTRA_TRAIN_ARGS=<protocol> scripts/ydf_bitid_cc18.sh binA binB <outdir>`:
   10 fixed numeric NaN-free CC18 tasks (banknote, diabetes, wdbc, kc1, pc4,
   qsar-biodeg, spambase, madelon, Bioresponse, phoneme), fold 0, model
   hashes via `compare_models.sh`. Seconds per task.
   `compare_models.sh` exits 1 whenever any file differs; for GBT the header
   proto carries training logs that change run to run, so read its RESULT
   line: "TREES IDENTICAL" = trees bit-identical, only "MODEL CHANGED" means
   the change altered the model (an A-vs-A control confirmed this 2026-09-05).
3. `NUM_TREES_DIVISOR=10 accuracy.sh <suffix>` for both arms with the same
   `EXTRA_BAZEL_CONFIGS`/`EXTRA_TRAIN_ARGS` (34 tasks × 10 folds, a few
   minutes at 30 GBT / 5 RF trees); `diff` the CSV bodies. Identical bodies
   confirm bit-identity; any difference → **MODEL CHANGED** warning with
   per-task mean ± std deltas (`benchmarks/utils/accuracy_stats.py` helps).
4. On the m7i the full protocol repeats step 3 at default size.

## Gates and report

< 15 % time saved = failed experiment (log it anyway); ≥ 20 % = ★. Replication
tolerance 3 %. Results directory: `benchmarks/results/verify/<candidate>/
vs-rebased-main/` or `.../vs-upstream/`, holding `A.csv`, `B.csv`, the accuracy
CSVs, `bitid.md`, `compare.md` and `report.md`. Commit that directory on the
candidate branch (use a worktree of it when the main tree is busy) — never on
`rebased-main` or `upstream-bench`; the user merges it into `rebased-main`
themselves. No `Co-Authored-By` trailers (CLA). Tell the user the commit sha.

Publishing (ask each time): engineering change bound for a YDF PR → Drive
`PRs/<candidate>` (folder id `1jby_ffbw7sDIimRvF-8VV-Y8KLT2t7pV`), and remind
the user the baseline rolls forward once merged; research change → Drive
`Results/<area>` (`1uqNO9vyD7EucxvlLGpqxq5a73fBm_VCA`; subfolders
ApplyProjection, GBT, EvaluateProjection, Axis-Aligned, GPU). CSV uploads via
`create_file` with text/csv become Sheets.

## Machines

| host | role | notes |
|---|---|---|
| AWS m7i (Xeon 8488C, 48 vCPU, 384 GB) | verdict machine; Claude Code runs on it directly | icx mandatory (enforced by `bench_common`); `sudo -n` works |
| Alienware m16 R2 (Ultra 9 185H) | secondary verdict machine | e-cores off for timing (scripts do it; sudo exemption exists); 6 P-cores |
| Mac (M1 Pro, 10 cores, 32 GB) | development; **quick pass only, no verdict** | Apple clang; no x86 SIMD paths; 24 GB trunk shapes swap (1.5M×4096 took 440 s at 30 GBT trees); |

Check the ISA banner (`USING INSTRUCTION SET:`) in the log; a Mac run that
does not exercise the AVX paths is not representative of the m7i.

## Concurrency check

`pgrep -f 'bazel-bin/examples/train_oblique_forest'` and
`pgrep -f 'evaluation/(runtime|accuracy).sh'` must be empty before starting.
