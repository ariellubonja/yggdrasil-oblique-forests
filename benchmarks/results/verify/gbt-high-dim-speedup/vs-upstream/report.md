# `origin/gbt-high-dim-speedup` vs `upstream/main` — quick pass on the Mac

**Mechanics only — no verdict. This Mac is not a verdict machine.** It validates that the
change ports to upstream, builds, produces the same model and moves the wall clock in the
expected direction. The number that counts comes from the m7i.

Date: 2026-09-05 (arm A, model equivalence) / 2026-09-06 (arm B rerun).
Machine: Apple M1 Pro, 10 cores, 32 GB, Homebrew clang 19.1.7 (arm64).

## 1. Model equivalence — PASSED, trees are bit-identical

Established before any timing, per the skill.

| check | scope | result |
|---|---|---|
| `scripts/ydf_bitid_cc18.sh` (`bitid.md`) | 10 CC18 tasks, fold 0, GBT Exact, 30 trees | **10/10 TREES IDENTICAL** (`nodes-*` sha256 match) |
| A-vs-A control (`bitid_control.md`) | bin_A run twice, 3 tasks | nodes match, headers differ ⇒ the GBT header/data_spec drift is run-to-run nondeterminism, not the change |
| `accuracy.sh` A vs B (`accuracy_AB_diff.txt`) | 34 CC18 tasks × 10 folds, GBT Exact, 30 trees | accuracy, AUC, logloss CSV bodies **identical** |

No MODEL CHANGED warning. The change only stops dispatching axis-aligned split jobs that a
sparse-oblique-over-numerical config guarantees will return `kNoBetterSplitFound`; the RNG
stream is preserved by an explicit `discard()`. So the timing is the whole story.

## 2. Protocol — Exact vs Exact, and why

`EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"'`, both arms.

**Upstream has no sparse-oblique histogramming yet**, so the fork's default
`Dynamic Random Histogram` path does not exist on `upstream/main` and cannot be used for an
apples-to-apples comparison. A Google PR adding it is expected the week of **2026-09-07**;
until it lands, every upstream comparison in this repo is Exact vs Exact. As a consequence
this pass measures **only** the `skip_dead_axis_jobs` change on the upstream code path — it
says nothing about the fork's histogram advantage over upstream.

Other protocol facts: `NUM_TREES_DIVISOR=10` (30 GBT trees), `runtime.sh --runs=1`, default
dataset list, one arm at a time. No script hyperparameter was touched
(`script_checksums_pre.txt` == `script_checksums_post.txt`, all four scripts).

### Arms

| arm | branch | HEAD | build config |
|---|---|---|---|
| A (baseline) | `upstream-bench` | `9bd86619` = `upstream/main` `df16834d` + one tooling commit (trimmed harness) | none |
| B (candidate) | `upstream-bench-gbt-hd` | `368162ea` = A + cherry-picked `a099367a`, `368162ea` from `origin/gbt-high-dim-speedup` | `--config=skip_dead_axis_jobs` |

Port evidence in `ported_commits.txt`, `ported_diff_*.diff`,
`ported_vs_original_diff_delta.txt` — the only delta vs the fork-side original is whitespace
in one `training.cc` continuation line; the conflicts resolved were a CHRONO_SCOPE line and
the `.bazelrc` hunk, both fork-only context.

Binaries: A `cbc8204a7f72…`, B `6ad186ed85a7…` (sha256) — the same binaries used for the
equivalence checks.

## 3. A/B timing (one run per arm, 30 GBT trees)

| dataset | A median s | B median s | speedup A/B | time saved | verdict |
|---|---:|---:|---:|---:|---|
| HIGGS 11M×29 | 338.88 | 339.96 | 1.00× | −0.3 % | no significant change |
| trunk 1.5M×4096 | 554.62 | 636.96 | 0.87× | −14.8 % | no significant change (swap, see §4) |
| trunk 150k×40k | 194.53 | 185.01 | 1.05× | +4.9 % | no significant change |
| trunk 15k×400k | 1114.26 | 66.04 | **16.87×** | **+94.1 %** | ★ speedup |

**Geometric mean: 1.98× (+49.5 % time saved).** Gates: < 15 % time saved = failed
experiment, ≥ 20 % = ★. Only the wide-short shape clears the star gate; that is exactly the
regime the change targets (dead axis-aligned work ≈ nodes × F, so it grows with F and is
invisible at HIGGS's F = 29).

Wall clock: arm A 2473 s, arm B 1495 s (whole passes, including build and data generation).

## 4. Memory warning and the 1.5M×4096 cell

Raised before the run (`pre_timing_memory.txt`, CRITICAL): each of the three trunk shapes
needs ~24 GB fp32 on a 32 GB Mac that already had ~9–10 GB of swap in use. Run anyway per the
2026-09-05 decision.

**No OOM in either arm** — all eight cells are real measurements. But all three trunk cells
ran under swap, and that is the most likely cause of arm B's 14.8 % *regression* on
1.5M×4096. Treat it as memory pressure, not a finding: the fork-side pass on the same Mac
regressed on the same shape the same way (440.4 → 524.3 s) while HIGGS, which fits in RAM,
was neutral in both passes. The m7i (384 GB) has no such artifact.

## 5. Replication

`replication_prior_search.txt`: **no prior CSV matches this protocol** (upstream-bench, GBT
Exact, 30 trees, Apple M1 Pro) — searches 1–3 all return no match. Arm A is the first point
of this protocol on this machine, so there is nothing to replicate against and no replication
is claimed.

Direction-only cross-checks (different branch/protocol, so not replication):

| source | protocol | 15k×400k | speedup |
|---|---|---|---|
| `benchmarks/results/verify/gbt-high-dim-speedup/{A,B}.csv` (2026-09-05, same Mac) | fork `rebased-main`, GBT **Dynamic**, 30 trees | 957.16 → 63.83 s | 15.0× |
| this pass | `upstream-bench`, GBT **Exact**, 30 trees | 1114.26 → 66.04 s | 16.9× |
| `ydf-fork.md`, 2026-08-07 laptop, 10 trees | GBT | — | 3.96× |

The upstream path reproduces the fork-side effect at the same order of magnitude, which is
what this smoke test was for. Same-arm noise floor: the killed 2026-09-05 arm-B attempt
measured HIGGS at 338.895 s vs today's 339.962 s (0.3 %).

## 6. Repo state

- `rebased-main` = `e10df2fe`. It advanced by one **user-authored** commit
  ("Update verify-speedup skill pt 2", 2026-09-05 15:41 local) since the orientation snapshot
  recorded `d78d17f0`; it touches only the skill files, not the measured code, and both arms
  were built from `upstream-bench` branches, so nothing in this pass is affected.
- `upstream-bench` = `9bd86619`, `upstream-bench-gbt-hd` = `368162ea` — **unchanged**.
- Nothing was committed by me on any branch, and no file in the main worktree was modified.
  Per the user's rule, result files for a candidate belong on the candidate branch, never on
  `rebased-main`; placing them is the coordinator's call.

## 7. What is left

- **Follow-up: the full protocol on the m7i** — `NUM_TREES_DIVISOR=1` (300 GBT trees), 3 runs
  per arm, plus the default-size accuracy sweep, on the Xeon 8488C with icx. Planning
  estimate from the m7i figures in `ydf-fork.md` (GBT Exact, 300 trees: HIGGS 2006 s,
  1.5M×4096 424 s, 150k×40k 717 s, 15k×400k 7596 s): ≈ 3 h per arm-A run, so ≈ 9 h for arm A
  and ≈ 3–4 h for arm B over 3 repetitions each. That is a deliberate machine-day, not a
  default — the user decides.
- **Re-check the upstream histogram PR** (expected week of 2026-09-07). Once it lands, the
  upstream comparison can move off Exact-vs-Exact; update `references/ydf-fork.md` and the
  `speedup-verification-two-baselines` memory.
- **Publishing — the user's call.** `PRs/gbt-high-dim-speedup`
  (`1jby_ffbw7sDIimRvF-8VV-Y8KLT2t7pV`) if this is an engineering change bound for a YDF PR,
  in which case the baseline must roll forward to include it once merged, so the next
  replication compares against the new state; or `Results/GBT`
  (`1uqNO9vyD7EucxvlLGpqxq5a73fBm_VCA`) if it is being reported as research. I did not
  upload anything.

## 8. Files in this directory

`upb_gbt_hd_a_exact_t30_r1.csv`, `upb_gbt_hd_b_exact_t30_r1.csv` (the two timing CSVs with
provenance heads), `compare.md`, `report.md`, `runner_runtime_A.log`, `runner_runtime_B.log`,
`runner_runtime_B_killed_2026-09-05.log` (killed attempt, for the record),
`accuracy_upb_gbt_hd_{a,b}_exact_t30*.csv`, `accuracy_AB_diff.txt`, `bitid.md`,
`bitid_control.md`, `bitid/`, `bitid_control/`, `bin_A`, `bin_B`, `driver.sh`,
`orientation_git.txt`, `ported_commits.txt`, `ported_diff_*.diff`,
`ported_vs_original_diff_delta.txt`, `replication_prior_search.txt`, `pre_timing_memory.txt`,
`script_checksums_pre.txt`, `script_checksums_post.txt`.
