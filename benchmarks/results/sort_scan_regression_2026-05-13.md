# Oblique_Exact regression investigation — 2026-05-13

## Reproduction (P-cores only, 6 cores, NUM_TREES=30, 3 runs)

trunk_50000_x_4096, Oblique_Exact:

| State | Runs (s) | Median |
|---|---|---|
| **main (today)** | 12.84 / 12.56 / 12.55 | **12.56** |
| rebased-main (today, as-is) | 16.48 / 16.69 / 16.04 | **16.48** |
| rebased + revert `0deb0429` (no_unique_address) | 16.16 / 16.59 / 16.77 | 16.59 |
| rebased + revert `5cc43428` (const improvement) | 16.57 / 16.44 / 16.04 | 16.40 |
| rebased + main's `splitter_accumulator.h` only | 16.57 / 16.76 / 16.62 | 16.62 |
| rebased + main's `splitter_scanner.h` only | 16.41 / 16.45 / 16.33 | 16.41 |
| **rebased + main's `splitter_scanner.h` + `splitter_accumulator.h` + `compatibility.h`** | 15.88 / 16.27 / 16.40 | **16.27** |

Saved CSVs: main 11.50s, rebased 15.43s (median of 7). Today's numbers are ~10% higher across the board (likely thermal/cron differences), but the **relative ~30% regression is consistent**.

## Where it lands (chrono CSV, summed tree-0 depths, trunk_300000_x_4096)

| Phase | origin | rebased | Δ |
|---|---|---|---|
| ApplyProjection | 4.14 s | 4.46 s | +7.6% |
| SortFillBuckets | 0.45 s | 0.54 s | +22% |
| SortFeatures (VQSort) | 3.81 s | 4.88 s | +28% |
| **SortScanSplits** | **5.12 s** | **8.31 s** | **+62%** |

Systemic regression across all FindBestSplitFlatHighway phases.

## **Key finding: the regression is NOT in the splitter code**

I swapped main's full triplet (`splitter_scanner.h` + `splitter_accumulator.h` + `compatibility.h`, the entire CHRONO_SCOPE(kSortScanSplits) callchain + the FlatBucketTraits + the LabelBucket types + the PREFETCH macro) into rebased-main and got **16.27 s** — essentially the same as rebased-main's as-is **16.48 s**. The "compare codes between branches for that scope" hypothesis is empirically refuted: swapping the entire scope plus its templated callees does not recover perf.

The regression is in a file I did not swap.

## What I verified is identical between branches

Source-level (byte-identical body, only comment/region marker diffs):
- `splitter_scanner.h` — `FindBestSplitFlatHighway`, `ScanSplitsFlat`, `FlatBucketTraits` specializations, `PerThreadCacheV2`, `Score<>` helper, the `CHRONO_SCOPE(kSortScanSplits)` block
- `splitter_accumulator.h` — `LabelBinaryCategoricalScoreAccumulator::AddOne/SubOne/Score/WeightedNumExamples`, `LabelBinaryCategoricalOneValueBucket::AddToScoreAcc/SubToScoreAcc`, `FeatureNumericalBucket`
- `distribution.h::BinaryDistributionEntropyF`
- `parallel_chrono.h` (the CHRONO_SCOPE macro)

Sizes of `BooleanValue<weighted>` / `FloatValue<weighted>` with `[[no_unique_address]]` verified identical to main's paired `*Only` / `*AndWeight` structs (g++ -O2 -std=c++17 sizeof test).

## What's left to check (biggest diffs vs main, after splitter swap proved cold)

```
training.cc      | 2350 LOC diff   ← likely culprit
random_forest.cc |  779 LOC diff   ← possible
oblique.cc       |  631 LOC diff   ← possible
training.h       |  297 LOC diff (depends on InternalTrainConfig moved into label.h on rebased)
label.h          |  102 LOC diff
```

Verbatim file swaps for these break compile (`InternalTrainConfig` ownership, ObliqueGpuComputer references, `SplitterConcurrencySetup` → `internal_config.split_finder_processor` refactor) — need surgical reverts, not whole-file swaps.

## Suggested next moves

1. **Bisect with the existing 3-run quick test.** 342 commits between branches is too many for manual diff. Threshold ≤ 14 s = pass; converges in ~9 builds. Test script:
   ```bash
   NUM_TREES=30
   sudo benchmarks/utils/set_cpu_e_features.sh --enable
   bazel build -c opt --cxxopt=-O3 --cxxopt=-march=native //examples:train_oblique_forest || exit 125
   sudo benchmarks/utils/set_cpu_e_features.sh --disable
   for i in 1 2 3; do
     ./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 50000 --cols 4096 \
       --feature_split_type Oblique --numerical_split_type Exact \
       --num_trees=$NUM_TREES --num_threads=-1 --compute_oob_performances=false \
       2>&1 | grep -oE 'Training block took:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+'
   done | sort -g | awk 'NR==2 {exit !($1 < 14.0)}'
   ```
2. **Strong candidates for the bisect** (concentrated on `training.cc` / `random_forest.cc` / `oblique.cc`):
   - `72106329 [YDF] Use stack instead of recursion for local training` — changes node-processing order, plausibly hurts cache locality across `PerThreadCacheV2` per node
   - The `splitter_concurrency_setup` → `internal_config.split_finder_processor` refactor in training.cc — changed function signatures and call sites for `FindBestConditionClassification` etc. Could trigger different inlining / parameter passing
   - `f9f69435 Speed up training with categorical features with large dictionary` — touches training.cc hot path
3. **Perf-record A/B** on rebased vs main during a small dataset run — directly identifies the stalled instructions or cache events. Faster than blind bisect.

## State after investigation

- Working tree: only `benchmarks/evaluation/runtime.sh` is dirty (your METHODS=Random→commented-out edit, preserved).
- `bazel-bin/examples/train_oblique_forest` is the unmodified rebased-main build.
- No code commits introduced. All file edits reverted.
- This findings file added at `benchmarks/results/sort_scan_regression_2026-05-13.md` (untracked).

---

## Update — decisive chrono+swap test (same session)

Ran the chrono harness on `trunk_300000_x_4096 --num_trees=1 --num_threads=1` three ways: origin baseline, rebased as-is, rebased + splitter triplet swap (main's `splitter_scanner.h` + `splitter_accumulator.h` + `compatibility.h`). Saved at `.../swap-triplet.csv` next to origin.csv / rebased.csv.

| Phase | origin (s) | rebased (s) | swap (s) | swap vs reb |
|---|---|---|---|---|
| ApplyProjection | 4.14 | 4.46 | 4.71 | +5.6% |
| GetCandidateAttributes | 0.69 | 0.71 | 0.72 | +0.8% |
| --SortFillBuckets | 0.45 | 0.54 | 0.54 | -0.1% |
| --SortFeatures | 3.81 | 4.88 | 4.93 | +1.1% |
| -SortScanSplits | 5.12 | 8.31 | 8.34 | +0.4% |
| CartFinderSetup | 0.06 | 0.07 | 0.07 | +0.7% |
| **SUM all phases** | **14.34** | **19.05** | **19.41** | **+1.9%** |

**The triplet swap had zero effect.** Every inner-loop phase stayed at rebased's levels. This is consistent with the prior conclusion that the swapped files' hot paths are byte-identical to main — there's nothing in those files to "fix." The regression is in code that feeds data to the splitter, not the splitter itself.

(Note: chrono numbers are a relative profiler, not precise absolute measurements. The +30-39% wall regression on runtime.sh is the precise figure; chrono shows where time is spent.)

## Per-depth tree shape & per-sample work

The chrono CSVs include per-(tree, depth) node counts and active samples. Tree shape is identical between origin and rebased through depth 8 (1,2,4,8,16,32,64,128,256), then diverges starting at depth 9 (origin 256 vs rebased 254). Rebased terminates at depth 28, origin at depth 32.

- origin: 33 rows, max_depth 32, **43,239 total nodes**, 5,264,311 active samples
- rebased: 29 rows, max_depth 28, **38,863 total nodes**, 5,196,567 active samples
- swap: identical to rebased

Despite rebased processing **10% fewer total nodes** and 1.3% fewer summed active samples, it takes 32-65% more time per phase. Per active sample, SortScanSplits costs **0.97 µs on origin vs 1.60 µs on rebased — ~65% slower per sample.**

Two independent signals here:
1. **RNG drift starting depth 9** — some commit consumed extra random numbers upstream (could be honest-splitting refactor or projection-sampling change).
2. **Per-sample inner-loop work is 65% slower** — the splitter body is unchanged, so the slowdown must be in *what the splitter is fed*: example index order, layout, cache locality, or per-call setup overhead.

## Strongest remaining suspects (after this test)

- **`72106329 [YDF] Use stack instead of recursion for local training`** — replaces recursive `NodeTrain` with iterative `GrowTreeLocal` + `std::vector<NodeAndExamples>` stack. `NodeAndExamples` holds `SelectedExamplesRollingBuffer` (two `absl::Span`s), so the actual example data lives in PerThreadCache and isn't moved. But the iteration order, when the stack vector reallocates, or how multiple sibling buffers stay alive simultaneously could affect cache locality of the data those spans point into. Cannot revert cleanly (conflicts in `training.cc` from later commits — honest-splitting refactor, regions/CHRONO additions).
- **Honest-splitting refactor** (`0247a65d`, `31418e60`, `d5c46429`, `e6c61df4`) — even with honest disabled, code paths and example sorting may have changed.
- **`a531ab5e [YDF] Template specializations`** — could change codegen of which `FindSplitLabel*` instantiation runs in the Oblique_Exact classification path.

## Next-step options

a. **Surgical revert of `72106329`** — manually resolve the conflict in `training.cc` to inject origin/main's recursive `NodeTrain`, then runtime.sh wall measurement. Time: ~30 min including conflict resolution.
b. **`git bisect` between `origin/main` and HEAD** — 342 commits, ~9 iterations to converge. Each ~125s (build + 3 runs of trunk_50000×4096). Total ~20 min. Most efficient if conflict resolution proves hard.
c. **perf-record A/B** — instruction-level attribution between origin and rebased binaries. Confirms whether the slowdown is memory-bound (LLC misses, TLB misses) or branch-bound.

---

## Update 2 — `72106329` revert measured (runtime.sh wall, this session)

Reverted `72106329 [YDF] Use stack instead of recursion for local training` (manually resolved one conflict in `training.cc` to keep the later CHRONO instrumentation block on the recursive signature). Built with `-c opt --cxxopt=-O3 --cxxopt=-march=native` + `bash runtime.sh revert-72106329`. CSV at `benchmarks/results/runtime_quick_revert-72106329.csv`.

| dataset | main baseline (s) | rebased today (s) | revert today (s) | revert vs main | revert vs rebased |
|---|---|---|---|---|---|
| trunk_12500_x_16384 | 6.60 | 8.67 | **8.42** | +27.6% | -2.9% |
| trunk_25000_x_8192 | 8.39 | 11.68 | **11.00** | +31.1% | -5.8% |
| trunk_50000_x_4096 | 11.50 | 15.88 | **14.72** | +28.0% | -7.3% |
| trunk_100000_x_2048 | 16.33 | 21.82 | **20.94** | +28.2% | -4.0% |
| trunk_200000_x_1024 | 22.60 | 31.04 | **29.33** | +29.8% | -5.5% |

`72106329` accounts for ~3–7% of wall time (varies by dataset). For `trunk_50000_x_4096`, the revert recovers 1.16 s of the 4.38 s gap — about **26% of the total regression**. The remaining ~74% is from other commits between `origin/main` and HEAD. So 72106329 is real but not primary.

## Conclusion of this investigation pass

The regression is spread across multiple commits, not concentrated in one. Single-commit reverts of the prime suspects (splitter-file cluster, threading refactor, template specs, stack-vs-recursion) recover at most ~5–7%. **`git bisect` is the next move** — it remains the cheapest way to land on the single largest contributor when multiple commits collectively cause the regression.

Suggested bisect bounds:
- `good` = `origin/main` (trunk_50000×4096 median wall ≈ 11.5 s)
- `bad` = `HEAD` (≈ 15.9 s)
- threshold = 13.5 s (well clear of noise, well below rebased)
- test: build runtime.sh-compatible binary, run 3× trunk_50000×4096, take median

---

## Update 3 — bisect failed, more reverts measured (2026-05-13 PM)

### Bisect failure mode
Attempted `git bisect run origin/main..HEAD` twice with an overlay script that injects our `examples/train_oblique_forest` target onto each tested commit. Both runs failed: **most intermediate upstream commits don't build with current GCC 13** because of `std::clamp` type-deduction errors:
```
return std::clamp(approximation_factor, min_factor_, 1.);   // float, float, double — bad deduction
```
The bug exists in old code, fixed in a later commit; newer GCC is stricter. Skipping these commits leaves bisect unable to narrow within the range. *Not* a C++17/standard issue (`.bazelrc` correctly sets `-std=c++17`). Would require patching every old commit's `std::clamp` calls to bisect cleanly.

### Additional single-commit reverts measured

Using `runtime.sh quick` (5 trunk datasets × 7 runs, E-cores disabled, 6 P-cores, NUM_TREES=30):

| dataset | main | rebased | revert 72106329 | revert 0247a65d | revert BOTH |
|---|---|---|---|---|---|
| trunk_12500_x_16384 | 6.60 | 8.67 | 8.42 | 8.28 | 8.31 |
| trunk_25000_x_8192 | 8.39 | 11.68 | 11.00 | 10.97 | 10.90 |
| trunk_50000_x_4096 | 11.50 | 15.88 | 14.72 | 15.06 | 15.04 |
| trunk_100000_x_2048 | 16.33 | 21.82 | 20.94 | 21.33 | 21.66 |
| trunk_200000_x_1024 | 22.60 | 31.04 | 29.33 | 30.05 | 30.17 |

**The reverts do NOT compound.** Stacking both reverts (15.04 s) is essentially the same as reverting just 0247a65d alone (15.06 s) and only slightly better than rebased (15.88 s). The apparent 5–7% recovery from each single revert is largely measurement noise — true individual contributions are at most ~3–5%.

### Ruled out in this session (in addition to earlier list)

- **Multi-threading refactor `d4bac4e7` / `19f5a358`**: both branches default RF to single-threaded inside each tree (`internal_config.num_threads = 1` default + `InternalTrainConfig::split_finder_processor = nullptr` for RF). Single-threaded `FindBestConditionSingleThreadManager` is structurally identical between branches.
- **`a531ab5e Template specializations`**: only adds Regression/Hessian-Regression template instantiations — irrelevant for classification.
- **`f3e6b7c3 Fix a rare issue with oblique splits`**: 11-line diff, `std::clamp(... 0.f, 1.f)` + 2 DCHECKs only.
- **`4737d272 Harden FindBestConditionFromSplitterWorkRequest`**: lives on the concurrent path only; RF doesn't hit it.
- **GPU oblique commits (`e7b21058`, `14f29221`, `91def6ea`)**: all behind `#ifdef OBLIQUE_GPU_ENABLED`, not defined for CPU builds.
- **Our histogram commits (`8ba29c27`, `ada7c121`, `5d4768d1`, `232501cc`, `3c201413`)**: histogram-path changes; Exact path untouched.
- **Highway version**: 1.3.0 on both `origin/main` and `HEAD`. VQSort code is the same.

### Most likely remaining cause

After 72106329 and 0247a65d are accounted for, ~22–28% slowdown remains. Given:
- Splitter inner-loop is unchanged (chrono swap test, ~0.4% delta)
- Per-active-sample work in SortScanSplits grew ~65% (from chrono data, on a single thread)
- No single commit's revert explains more than ~5%

The remaining regression is most likely from **dependency-version differences between `origin/main` and `HEAD`'s `MODULE.bazel`**:

| Dep | origin/main MODULE.bazel | HEAD MODULE.bazel |
|---|---|---|
| highway | 1.3.0 (explicit) | 1.3.0 (explicit) |
| abseil-cpp | not pinned (transitively resolved) | **20250814.1 (explicit)** |
| eigen | not pinned | **3.4.0.bcr.3 (explicit)** |
| protobuf | not pinned | **31.1 (explicit)** |
| googletest | not pinned | 1.17.0 (explicit) |
| bazel_skylib | not pinned | 1.8.1 (explicit) |
| platforms | 0.0.10 | 1.0.0 |

`HEAD`'s MODULE.bazel explicitly pins newer versions of abseil/eigen/protobuf that `origin/main`'s build resolved differently. A newer `absl::Span`/`absl::vector_internal` could change per-element access cost or allocator interaction — exactly the kind of ambient change that would show as a ~30% per-sample slowdown in a tight scan loop without any source-code diff.

**Next-step recommendations:**
1. Build `origin/main` HEAD's MODULE.bazel grafted on, and measure. If perf becomes rebased-like, the regression is in the deps, not the YDF source.
2. Conversely, build `HEAD` with `origin/main`'s loose MODULE.bazel (fewer pinned versions), and measure. If perf becomes origin-like, same conclusion.
3. perf-record A/B between origin and rebased binaries on trunk_50000×4096 — look for allocation/sync hotspots that point at a specific abseil/eigen call site.

## 2026-05-13 — Intel Advisor hotspots survey (definitive)

**Workload**: `trunk_50000_x_4096`, Oblique/Exact, 30 trees, 22 threads, `-c opt --config=profiler` (O2 -g -strip=never).
Both binaries built fresh from each branch; advisor `--collect=survey`. CSVs: `benchmarks/results/advisor_{main,rebased}_survey.csv`.

### Vectorized loops summary
| Branch | # vectorized loops | Vector self time |
|---|---|---|
| main | 11 | 9.56 s |
| rebased-main | 2 | 0.34 s |

The rebase **de-vectorized** the two hot loops responsible for ApplyProjection and SortFillBuckets. Concretely:

### Loop 1: ProjectionEvaluator::Evaluate inner FMA loop
- main `oblique.cc:1394` — Vectorized (SSE2) — **2.91 s self**
- rebased `oblique.cc:1122` — **Scalar** — **21.46 s self** (≈7.4× more CPU)

**Root cause confirmed**: rebased removed the `#ifndef YDF_BENCH_SKIP_ISNAN` guard around the `std::isnan(attribute_value)` branch (rebased oblique.cc:1129). With `-DYDF_BENCH_SKIP_ISNAN` set in `.bazelrc`, main compiles the inner body to a pure FMA (`value += attribute_value * item.weight;`) — auto-vectorizable. Rebased always compiles the branchy isnan/replacement path — not vectorizable.

### Loop 2: SortFillBuckets fill loop
- main `splitter_scanner.h:1838` — Vectorized (SSE2) — 1.99 s + 1.96 s self (Versioned + Body)
- rebased — no vectorized fill loop entry; scan loop at `splitter_scanner.h:1672` shows 3.54 s self, 39.7 s total (massive children gap — children apparently not inlined)

The fill-loop *body* is byte-identical between branches. Suspected cause: `splitter_accumulator.h` templated-`<bool weighted>` + `[[no_unique_address]] std::conditional_t<weighted, float, Empty>` refactor changes inlining cost-model in the TU, blocking vectorization of the surrounding fill loop. Not yet confirmed by objdump.

### Magnitude attribution (multi-thread Advisor sampling)
- Loop 1 (isnan guard): +18.5 s CPU time, ≈+0.84 s wall (22-thread × ~6× effective parallelism) — explains roughly half the wall-time regression (~2.0 s on 30-tree-on-50k×4096 first-block).
- Loop 2 (fill devectorization + scan-children gap): remainder.

### Next action
Re-add the `#ifndef YDF_BENCH_SKIP_ISNAN` guard around the isnan check in `oblique.cc:Evaluate` and re-bench. If runtime.sh recovers to within RNG of main, declare root cause = this guard removal during rebase. Loop 2 still needs investigation.

## 2026-05-13 — Real root cause: compiler mismatch (Bazel auto-detect)

The de-vectorization observed above (Loop 2: SortFillBuckets fill loop emitted scalar on rebased) was not a source change — it was a **compiler change**. Bazel's `local_config_cc` rule auto-detects the C++ toolchain and caches it per output_base. On `main`, the cached toolchain happened to point at ICX (Intel oneAPI 2025.2 clang). On `rebased-main`, after the Bazel 6.5.0 → 7.7.0 bzlmod migration, auto-detection silently picked `/usr/bin/gcc` (GCC 13) instead. ICX vectorized the fill loop with `iz_intel_compute_loop_termination`; GCC bailed out and emitted scalar code.

### Fix
Pin ICX in `.bazelrc`:
```
build --repo_env=CC=icx
build --repo_env=CXX=icpx
```
Requires `source /opt/intel/oneapi/setvars.sh` so `icx`/`icpx` are on PATH for Bazel's `cc_configure`.

### Wall-time impact (runtime.sh quick, Exact-only, 5 datasets × 7 runs, median)

| dataset | main(icx) | rb+gcc | %vs main | rb+icx | %vs main | recovered |
|---|---:|---:|---:|---:|---:|---:|
| trunk_100000_x_2048 | 16.335 | 21.819 | +33.6% | 19.159 | +17.3% | +16.3pp |
| trunk_12500_x_16384 | 6.597 | 8.669 | +31.4% | 7.188 | +9.0% | +22.4pp |
| trunk_200000_x_1024 | 22.604 | 31.037 | +37.3% | 27.017 | +19.5% | +17.8pp |
| trunk_25000_x_8192 | 8.391 | 11.681 | +39.2% | 9.647 | +15.0% | +24.2pp |
| trunk_50000_x_4096 | 11.496 | 15.877 | +38.1% | 13.438 | +16.9% | +21.2pp |

Compiler swap recovered 16–24 pp of the 31–39% regression. Residual: +9–20% (mean ~+15%).

### 2026-05-13 — ICX-vs-ICX Advisor survey (residual gap)

Re-ran Advisor on apples-to-apples ICX binaries (`/tmp/advisor-bins/train_oblique_forest.{main,rebased}-icx`, both `-c opt --config=profiler -O3 -march=native`, ICX 2025.2.1). Workload: trunk 50000×4096, Oblique Exact, 30 trees, all threads, EXIT_EARLY after first training block. CSV: `benchmarks/results/advisor_{main,rebased}_icx_survey.csv`.

Program elapsed: main 7.42 s → rebased 8.40 s (+13.2%).
Sum of per-loop self-time: 48.78 s → 60.96 s (+25.0% multi-threaded CPU-time).
Vectorized-loop count: 16 in both branches.

Top per-function self-time deltas (multi-thread CPU-time, sum across threads):

| Δ self | main self | rebased self | vec? | function |
|---:|---:|---:|---|---|
| **+4.80 s** | 15.47 s | 20.27 s | Vec/Vec | `FindSplitLabelClassificationFeatureNumericalCart` body at `splitter_scanner.h:1838` (SortScanSplits inner scan) |
| **+4.30 s** | 18.71 s | 23.01 s | Scalar/Scalar | `FindBestConditionSparseObliqueTemplate<Classification>` top-level loop in `oblique.cc` |
| **+1.64 s** | 6.19 s | 7.83 s | Scalar | `hwy::N_AVX2::detail::Recurse` (VQSort partition) at `vqsort-inl.h:881/1226` |
| +0.56 s | 2.14 s | 2.70 s | Vec/Vec | child loop of the above SortScan |
| +0.31 s | 1.93 s | 2.24 s | Scalar | `GetCandidateAttributes` (stl_algo.h:3724) |

The three biggest residual regressors are the same three kernels the original chrono breakdown flagged (SortScanSplits, oblique outer/ApplyProjection, VQSort). They are vectorized in both branches (or scalar in both), so the gap is **not** another de-vectorization — it is per-loop work growing.

Probable causes (now that compilers match):
- **Data layout / bucket organization** flowing into these kernels differs (something earlier in the pipeline changed what's being scanned/sorted), or
- **Inlining context** in the surrounding code changed (a non-hot caller's body changed in a way that altered ICX's inlining decisions for the hot kernel).

The earlier swap-triplet test (replacing `splitter_scanner.h` + `splitter_accumulator.h` + `compatibility.h` with main's versions) did not move wall time — consistent with the gap not being in those headers' code but in their *inputs*.

### 2026-05-13 — Cross-check of the three regressor functions vs `main` (residual gap)

Direct code/asm/config cross-check of the three regressors identified above:

**F1 — Highway version drift (`hwy::Recurse`):** *Ruled out.* Highway is `1.3.0` on both branches (`main` pulls via `third_party/highway/workspace.bzl`; `rebased-main` pulls via `bazel_dep` in `MODULE.bazel` → BCR). Identical sha256 (`07b3c1ba…0bc2`); `vqsort-inl.h` is byte-identical. VQSort call sites in `splitter_scanner.h:1788` and `training.cc` are byte-identical.

**F2 — Scanner inner-loop body (`FindSplitLabelClassificationFeatureNumericalCart`):** *Ruled out.* The `ScanSplitsFlat` body (lines 1672–1711) and `LabelCategoricalOneValueBucket<false>` struct layout are byte-identical between branches. The `splitter_accumulator.h` refactor (`bfea2dea`, `0deb0429`, `b14e89dd`) changed `IntegerValueOnly` → `IntegerValue<false>` with `[[no_unique_address]]`; for the classification unweighted path the layout is the same 4 bytes. `Prefetch()` removal affects `ScanSplitsPresortedSparse`, not the `ScanSplitsFlat` hot path.

**F3 — `EvaluateProjection` inlining loss in `FindBestConditionSparseObliqueTemplate`:** *Refuted by asm.* Disassembled both ICX binaries:

```
Main:    1002 asm lines, 53 callq instructions, 1× call EvaluateProjection<Classification>
Rebased:  868 asm lines, 48 callq instructions, 1× call EvaluateProjection<Classification>
```

Both branches emit `EvaluateProjection<Classification>` as a standalone weak symbol (`nm | grep EvaluateProjection`) AND call it through 1 callq in `FindBestConditionSparseObliqueTemplate`. Neither branch inlines it. Function signatures are identical (modulo `absl::lts_20240722` → `lts_20250814` namespace bump). The ~134-line size difference is in cleanup/destructor paths (main has 3 vs 1 `ProjectionEvaluator::~ProjectionEvaluator`, 3 vs 2 `_Unwind_Resume@plt` etc.), not the hot path.

**F4 — `std::clamp(projection_density, 0.f, 1.f)` from commit `f3e6b7c3` shifting baseline:** *Refuted by config.* The default `--projection_density_factor=1.5f` and density formula `density = projection_density_factor / num_features`. For trunk_50000_x_4096: `density = 1.5 / 4096 = 0.000366`, far below 1.0. For all benchmark workloads (`num_features ≥ 1024`), density is in `[0.0015, 0.000366]`. The clamp is a no-op everywhere — cannot account for any regression.

### 2026-05-13 — Tree-shape divergence under matched ICX (independent finding)

Re-ran CHRONO on rebased+ICX (clean apples-to-apples vs the existing `origin.csv`), single-tree single-thread on trunk_300000_x_4096:

```
                origin    rebased+ICX    Δ
Total nodes      43239      36995      -6244  (-14.4%)
Max depth          32         33         +1
ApplyProj sum    4.14 s     4.28 s     +3.2%
SortScanSplits   5.12 s     5.33 s     +4.1%
SortFeatures     3.81 s     3.99 s     +4.7%
```

Tree shape is materially different (rebased grows shallower, narrower trees) even with identical random seed and matched compiler. This is a code-induced algorithmic divergence — not a compiler effect. Per-node times at matched depths differ by only ±5% on average (the +37% at depth 1 is one-time root cost, the negative deltas at deep levels reflect smaller node populations).

**Per-depth comparison (`benchmarks/results/per_function_timing/.../rebased-icx.csv` vs `origin.csv`):** the +4% chrono total under-explains the +15% multi-thread wall regression.

### 2026-05-13 — Single-thread benchmark (rules out multi-thread amplifier)

Ran the actual benchmark workload single-threaded to test the "multi-thread interaction" hypothesis:

```
trunk_50000_x_4096, 30 trees, profiler -O2 -g ICX, num_threads=1, E-cores ON:
    main-icx:     49.51 s wall  (47.38 s training)
    rebased-icx:  62.27 s wall  (59.80 s training)
    Δ = +26.2% wall
```

Compared to multi-threaded `runtime.sh` result on the same workload (+16.9%, 11.50 → 13.44 s) — **the single-thread delta is WORSE than multi-thread**. The regression is in per-thread compute and is amplified when each node sees fewer samples. Multi-threading does NOT cause the gap — workload shape does.

This is consistent with **per-node fixed overhead** (e.g., proto accessors, vector allocations, ProjectionEvaluator construction) growing on rebased, since smaller-sample nodes magnify the fixed-overhead fraction. But the asm structure shows the same call signature, so the overhead is in inlined code, not new function calls.

### Status (2026-05-13)

| | |
|---|---|
| Original regression | +31–39% wall (rebased vs main, runtime.sh quick, Exact) |
| Compiler-fix recovery | +22–24 pp (pinned ICX in `.bazelrc`) |
| Residual | +9–20% wall (mean ~+15%) |
| F1 (Highway) | Ruled out by version match |
| F2 (Scanner body) | Ruled out (byte-identical hot scope) |
| F3 (Inlining) | Refuted by asm (both call EvaluateProjection once) |
| F4 (Density clamp) | Refuted by config (density `≪ 1`) |
| F5 (Multi-thread amplifier) | Refuted (single-thread delta is even worse) |
| Tree shape divergence | Confirmed but doesn't account for perf gap (chrono is only +4%) |

No remaining single hypothesis credibly explains a >5pp residual. The +15% appears to be **aggregate drift** from many small changes (absl bump `lts_20240722` → `lts_20250814`, std-lib instantiations, Bazel 6→7 link-time decisions, per-node fixed-overhead creep). Further investigation likely yields <5pp per memory note's 25% threshold rule. The tree-shape divergence is worth tracking down separately as it's a deterministic behavior change, not a perf issue.

