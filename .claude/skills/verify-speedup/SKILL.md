---
name: verify-speedup
description: Verify that a code change (branch, PR, commit, or build flag) really speeds up a program, by A/B end-to-end timing against a named baseline ref, replication against earlier results, a bit-identity or accuracy check, and a provenance-stamped report, with safeguards for runs that take hours. Use this whenever the user asks to benchmark, time, A/B, "compare the speed of", "check the speedup of", or "does X make it faster" for a branch, PR, commit, config or flag against main, upstream, a baseline, or an older result; asks to reproduce or replicate a previous timing; asks for performance numbers for a PR, paper or reviewer; or asks to compare a fork against upstream. Trigger even when the word "benchmark" is absent, and even for a quick "is this faster?" — an unverified speedup claim is the failure mode this skill exists to prevent. Not for microbenchmarks or profiling (use linux-perf) or for accuracy-only studies.
---

# Verify a speedup claim

A speedup claim is only worth something if a stranger could reproduce it: same
machine class, same compiler, same protocol, both arms built from known commits,
and the baseline arm agreeing with what the same protocol produced last month.
This skill turns "I think branch X is faster" into a report that meets that bar,
and it protects the person asking from two expensive mistakes: burning a working
day on a 3-run protocol nobody wanted, and reporting a number that later turns
out to have come from the wrong compiler or a different dataset list.

The body below is repository-agnostic. Repository specifics (scripts, flags,
datasets, machines, prior results) live in `references/<repo>.md`; this skill
ships `references/ydf-fork.md` for the ariellubonja Yggdrasil Decision Forests
fork. When you start, look for a reference matching the current repository and
follow it wherever it is more specific than this file.

## Vocabulary

- **Candidate**: the change under test. A branch/PR/commit, or a build flag that
  compiles the change in.
- **Baseline target**: the ref the candidate is compared against. There may be
  several (e.g. the fork's main *and* upstream main); each is a separate
  verification with its own protocol.
- **Arm A / arm B**: baseline build / candidate build. Everything except the
  change must be identical: machine, compiler, flags, datasets, tree counts.
- **Protocol**: the exact runner invocation (flags, datasets, repetitions) the
  repository's result history was produced with. You reuse it; you never invent
  one.
- **Provenance**: the header each result carries (commit, branch, machine,
  compiler, build configs, runner args). It is what makes replication possible.
- **Replication**: arm A agreeing with the newest earlier result of the same
  protocol on the same machine class. If A does not replicate, B's number means
  nothing yet.

## Rules that never bend

1. **Do not edit the benchmark scripts' hyperparameters or defaults.** Tree
   counts, projection counts, thresholds, dataset shapes, repetitions: leave
   them. Select and configure only through the knobs the scripts expose
   (environment variables, `--runs`, dataset overrides). The whole point of a
   protocol is that every result in the history used the same one; a "small
   tweak" silently invalidates every comparison. If the scripts fail with the
   defaults, stop and notify the user rather than adjusting.
2. **Same compiler on both arms, and the one the repository mandates.** A
   different compiler moves hot loops by tens of percent; a gcc build compared
   to an icx build is not a measurement. Trust only a post-build check on the
   produced binary, never the flags you passed.
3. **One experiment at a time on a machine.** Refuse to start if another
   benchmark process is running; timing noise from a concurrent job is
   indistinguishable from a real effect.
4. **Verdicts only from designated measurement machines.** A laptop or a Mac
   can validate the pipeline, never the number. Say so in the report.
5. **Never fabricate or extrapolate a timing.** An OOM, an error, a missing
   dataset stays visible as such in the table.
6. **Start immediately, but tell the user what is running.** No plan-approval
   round trip; the first report must state the inferred protocol so a wrong
   inference is caught after minutes, not hours.

## Workflow

### 0. Orient (before any build)

- Read `references/<repo>.md` if present. Identify from it: the runner scripts,
  their knobs, the default dataset list with known per-run durations and memory
  needs, the compiler rule, the machine table, the result directory layout, and
  the gates.
- Identify the candidate: a ref (default: the current branch), a PR number
  (resolve to its branch), or a build flag. Read its diff against the baseline
  target. From the diff, infer **which code path it touches** (which learner,
  which split finder, which stage) and therefore which protocol flags exercise
  it. A change inside a boosting-only code path must be measured with the
  boosting protocol; measuring the default protocol would show 0 %. State the
  inference in the first report.
- Classify the change: **flag-gated** (an `#ifdef`/build config, so one source
  tree builds both arms) or **unconditional** (two refs). Flag-gated is cheaper:
  run the runner twice with and without the config. Also classify
  **engineering vs research** if the reference distinguishes where results go.
- Check the machine: is it a verdict machine? Is anything else running
  (`pgrep -f <binary name>`)? How much RAM does each dataset need (rows × cols ×
  4 B for fp32) and does the biggest fit with headroom? Predict runtime per run
  from prior results (`scripts/find_prior_baseline.py --estimate`). If a single
  run is predicted to exceed the notification threshold, say so up front; still
  start the one-run pass, as instructed.
- Never rebuild or re-run what an existing, matching prior result already
  answers; reuse it and say so.

### 1. Prepare the arms

- **Flag-gated, same baseline branch**: rebase the candidate onto the baseline
  tip first (`git merge-tree` tells you if it is clean), so arm A is the current
  tip and the replication check is meaningful. Arm A = build without the
  config, arm B = with it.
- **Unconditional change**: two worktrees (`git worktree add`), one per ref.
  Symlink or point the data directory into each worktree; result CSVs are
  written per worktree and copied back to the main results tree.
- **Upstream baseline**: use the repository's tooling branch that tracks
  upstream (see reference). Port the candidate onto it (cherry-pick; resolve
  conflicts that come from fork-only context; show the resulting diff in the
  report because this ported branch is usually the PR branch). If the
  candidate depends on fork-only infrastructure and cannot be ported, say so
  and stop that target.
- Build each arm through the repository's runner (it applies the compiler
  pin, platform configs and the post-build compiler check). Keep a copy of
  each arm's binary for the identity check later. A build failure is a stop:
  notify the user, do not patch around it.

### 2. First pass: one run per arm

Run the runner with `--runs=1` (or its equivalent) on the **default dataset
list**, arm A then arm B, never concurrently. While it runs, keep a monitor on
the log (Claude Code: the `Monitor` tool running `scripts/watch_runs.sh <log>
<limit_seconds>`): it emits one line per run start/finish, every OOM/ERROR,
and a single `LONG_RUN` line when a run passes the limit (default 7200 s). On
`LONG_RUN`, send a phone notification immediately (Claude Code: the
`PushNotification` tool); on any error or OOM, stop after the current arm and
notify. Record each arm's per-run wall times: they drive the next decision.

Why one run first: nobody knows how long a run takes until it has run once, and
a 3-run protocol on a 2-hour dataset is a day of machine time that must be a
deliberate choice, not a default.

### 3. Report, then decide

Report the one-run table (`scripts/compare_runs.py A.csv B.csv`) with the
inferred protocol, the machine, and per-dataset wall times. Then:

- **Every run under 30 minutes** → continue to the full protocol without
  asking.
- **Any run over 30 minutes** → stop. Report which datasets are slow and how
  long the full protocol would take. Propose a **smaller shape from the same
  regime** (a wide dataset stays wide, a tall one stays tall: e.g. 15k × 400k
  → 15k × 40k) using the runner's dataset override, run its one-run pass, and
  report again. The user decides whether the full protocol on the big shapes is
  worth it.
- Any OOM or error → stop and notify; include the runner's log path.

### 4. Full protocol and replication

Run the default repetitions (median of N as the runner defines it) for both
arms. Then run `scripts/find_prior_baseline.py --like A.csv` against the
results tree: it finds the newest earlier CSV with the same provenance
(machine class, configs, args, tree count) and reports per-dataset drift.
Within tolerance (default 3 %) means the machine and code state are as before
and B's speedup is attributable to the change. Beyond tolerance means something
else moved (code, build, machine state); report it as a finding and do not
claim a speedup until it is understood.

### 5. Identity or accuracy

A speedup is only admissible if you also say what it did to the model. Follow
the reference's procedure (for the fork: hash-compare A and B models on 10
fixed CC18 tasks with `scripts/ydf_bitid_cc18.sh`, then run the full
accuracy sweep on both arms and compare). Bit-identical trees are the strongest
statement; if they differ, the accuracy comparison is the statement, and the
difference itself is a headline finding.

### 6. Report and deliverables

Write `compare.md` next to the two CSVs in the results tree, containing: the
verdict table with gates applied (defaults: < 15 % time saved = failed
experiment, ≥ 20 % = ★), the geometric mean, the replication table, the
identity/accuracy result, both provenance blocks, the exact commands and env
used, the machine caveats, and what was *not* done (skipped shapes, one-run
only, no verdict machine). Then ask the user where the results should be
published if the reference offers a choice (for the fork: engineering change →
Drive `PRs/<candidate>`, research change → Drive `Results/<area>`). For an
engineering change that will be merged, remind the user that the baseline must
roll forward to include it, so the next replication compares against the new
state.

## Long runs, notifications, failures

- Threshold for a phone notification: a single run exceeding 2 hours, any
  error/OOM that stops the pipeline, and completion of any pass that took more
  than 30 minutes. Keep messages under 200 characters and lead with the
  actionable fact.
- Never leave a runner going unattended without a monitor on its log.
- Runs are sequential. Builds may overlap with nothing.
- The runner rebuilds into the same `bazel-bin` for every arm, so two arms in
  one source tree can never run at the same time, and each arm's binary must be
  copied out before the next build if the identity check needs it later.

## Machines

Numbers count only on the designated measurement hosts (see reference). On
any other host the pipeline still runs, but the report must say "mechanics
only, no verdict", must compare the runtime ISA banner and compiler against
the verdict machine's, and must raise a **critical warning** when a benchmark
cannot run at all there (SIMD paths absent, dataset larger than RAM, missing
compiler). Predict the RAM case before running.

## Adapting to your repository

Write `references/<repo>.md` with: the runner scripts and their knobs; the
default dataset list with typical per-run durations and memory per dataset on
the verdict machine; the compiler rule and how the build enforces it; the
provenance fields the runner writes; the results directory layout; the
identity/accuracy procedure; the gates; the machine table; the publish
destinations. Your runner must provide: a deterministic binary that logs one
parseable wall-time line, a runner that repeats runs and reports the median
with a provenance header, a log→CSV parser, and a way to hash or compare
trained models. The scripts here read the CSV shape described in
`scripts/compare_runs.py`; adapt the reader if yours differs.

## Files in this skill

| file | purpose |
|---|---|
| `scripts/compare_runs.py` | A/B table with gates, geometric mean, provenance diff → markdown |
| `scripts/find_prior_baseline.py` | newest matching prior CSV: replication drift, or per-run duration estimate |
| `scripts/watch_runs.sh` | log event stream for the Monitor tool: run start/finish, LONG_RUN, OOM/ERROR |
| `scripts/ydf_bitid_cc18.sh` | fork-specific: hash-compare A/B models on 10 CC18 tasks |
| `references/ydf-fork.md` | the ariellubonja YDF fork: baselines, protocols, machines, priors |
