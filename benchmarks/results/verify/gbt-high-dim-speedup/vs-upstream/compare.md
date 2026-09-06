## A/B runtime: upstream-bench 9bd86619 (upstream/main df16834d) vs upstream-bench-gbt-hd 368162ea (+skip_dead_axis_jobs)

| dataset | algorithm | A median s | B median s | speedup A/B | time saved | verdict |
|---|---|---:|---:|---:|---:|---|
| HIGGS_with_header | SPO-GBT_Exact | 338.88 ± n/a | 339.96 ± n/a | 1.00× | -0.3 % | no significant change |
| trunk_1500000_x_4096 | SPO-GBT_Exact | 554.62 ± n/a | 636.96 ± n/a | 0.87× | -14.8 % | no significant change |
| trunk_150000_x_40000 | SPO-GBT_Exact | 194.53 ± n/a | 185.01 ± n/a | 1.05× | +4.9 % | no significant change |
| trunk_15000_x_400000 | SPO-GBT_Exact | 1114.26 ± n/a | 66.04 ± n/a | 16.87× | +94.1 % | ★ speedup |

Geometric-mean speedup over 4 dataset(s): **1.98×** (+49.5 % time saved). Gates: failed < 15 %, ★ ≥ 20 %.

### Provenance

| field | A | B |
|---|---|---|
| date_utc | 2026-09-05T20:21:12Z | 2026-09-06T22:51:26Z |
| git_sha | 9bd86619 | 368162ea |
| git_branch | upstream-bench | upstream-bench-gbt-hd |
| machine | Apple M1 Pro (nproc=10) | Apple M1 Pro (nproc=10) |
| compiler | Homebrew clang version 19.1.7 [/opt/homebrew/opt/llvm/bin/clang] | Homebrew clang version 19.1.7 [/opt/homebrew/opt/llvm/bin/clang] |
| EXTRA_BAZEL_CONFIGS | <none> | --config=skip_dead_axis_jobs |
| EXTRA_TRAIN_ARGS | --ensemble_method Boosting --numerical_split_type "Exact" | --ensemble_method Boosting --numerical_split_type "Exact" |
| NUM_TREES | 30  NUM_RUNS: 1 | 30  NUM_RUNS: 1 |
| NUM_RUNS | 1 | 1 |
| CSV_DATASETS | benchmarks/data/HIGGS_with_header.csv|class | benchmarks/data/HIGGS_with_header.csv|class |
| TRUNK_DATASETS | 1500000|4096 150000|40000 15000|400000 | 1500000|4096 150000|40000 15000|400000 |

### Model equivalence (established before any timing)

Trees are **bit-identical**; the change is a pure work-elimination.

- `bitid.md`: 10 CC18 tasks (fold 0, GBT Exact, 30 trees) — all 10 `TREES IDENTICAL`
  (`nodes-*` sha256 match). `compare_models.sh` exits 1 because GBT header protos differ;
  `bitid_control.md` (bin_A vs bin_A, same fold) shows the same header/data_spec drift with
  matching nodes, so the metadata difference is run-to-run nondeterminism, not the change.
- `accuracy_AB_diff.txt`: `accuracy.sh` on both arms, 34 CC18 tasks × 10 folds, 30 GBT trees
  — accuracy, AUC and logloss CSV bodies **identical**.

No MODEL CHANGED warning applies.

### Replication

No prior CSV under `benchmarks/results` matches this protocol (upstream-bench, GBT Exact,
30 trees, Apple M1 Pro): `replication_prior_search.txt` searches 1–3 all return "no match".
Arm A is therefore the **first** point of this protocol on this machine; there is nothing to
replicate against, and the report claims none.

Direction-only cross-checks (different protocol/branch, so not a replication):

| source | protocol | 15k×400k A → B | speedup |
|---|---|---|---|
| `benchmarks/results/verify/gbt-high-dim-speedup/{A,B}.csv` (2026-09-05, same Mac) | fork `rebased-main`, GBT **Dynamic**, 30 trees | 957.16 s → 63.83 s | 15.0× |
| this pass (2026-09-05/06, same Mac) | `upstream-bench`, GBT **Exact**, 30 trees | 1114.26 s → 66.04 s | 16.9× |
| reference `ydf-fork.md`, 2026-08-07 laptop | GBT, 10 trees | — | 3.96× |

Same-arm noise floor: the killed 2026-09-05 arm-B attempt measured HIGGS at 338.895 s and
today's rerun at 339.962 s (0.3 % apart) with the same binary and protocol.

### Commands and environment

```bash
# both arms, in the upstream-bench worktree, via OUT/driver.sh
NUM_TREES_DIVISOR=10 \
EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"' \
  bash benchmarks/evaluation/runtime.sh --runs=1 upb_gbt_hd_a_exact_t30_r1   # arm A, no configs
EXTRA_BAZEL_CONFIGS=--config=skip_dead_axis_jobs NUM_TREES_DIVISOR=10 \
EXTRA_TRAIN_ARGS='--ensemble_method Boosting --numerical_split_type "Exact"' \
  bash benchmarks/evaluation/runtime.sh --runs=1 upb_gbt_hd_b_exact_t30_r1   # arm B
```

Binaries: A `cbc8204a7f72…`, B `6ad186ed85a7…` (sha256, in the runner logs; the same binaries
used for the bit-identity and accuracy checks). Benchmark scripts unchanged across the whole
pass (`script_checksums_pre.txt` == `script_checksums_post.txt`).

### Machine caveats

Apple M1 Pro, 10 cores, 32 GB, Homebrew clang 19.1.7 — a **development machine, not a verdict
machine**. arm64: no `USING INSTRUCTION SET:` banner and none of the x86 SIMD paths the m7i
(Xeon 8488C, icx) exercises. Wall times here are not comparable to the m7i's.

Memory: the three trunk shapes need ~24 GB fp32 each on a 32 GB box that already had ~9–10 GB
of swap in use (`pre_timing_memory.txt`, critical warning raised before the run). **No OOM in
either arm** — every cell is a real measurement — but all three trunk cells ran under swap.
That is the likely cause of arm B's slower 1.5M×4096 cell (554.6 → 637.0 s, −14.8 %): a
memory-pressure artifact, not a property of the change. The fork-side pass on the same Mac
regressed on the same shape (440.4 → 524.3 s) while HIGGS, which fits in RAM, was neutral in
both passes.

### What was NOT done

- One run per arm (`--runs=1`), reduced model size (`NUM_TREES_DIVISOR=10`, 30 GBT trees).
  No medians, no stddev.
- No full protocol (divisor 1, 3 runs) and no default-size accuracy sweep — those belong on
  the m7i.
- No replication against a prior baseline (none exists for this protocol).
- Exact vs Exact only: upstream has no sparse-oblique histogramming, so the fork's default
  Dynamic Random Histogram path is not exercised on either arm.
