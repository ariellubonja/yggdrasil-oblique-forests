# Method C runbook — per-depth callgrind cache-sim of the DW1 oblique gather

Self-contained instructions to reproduce the Method C measurement on another
machine. Method C profiles the hot DW1 oblique gather `col[sel_ptr[i]]`
(`oblique_cpu_depthwise_1pass.cc:291`) under callgrind's cache simulator and dumps
**per tree depth**, giving real L1 / LL(DRAM) read-miss counts you can convert to
"useful floats per 64B cache line."

It complements Method A (the exact native distinct-cache-line counter,
`#ifdef YDF_LINECOUNT_A` in `training.cc`). A gives the line *geometry* fast and
multithreaded; C gives real miss *costs* (the `DLmr`/DRAM column) but is slow and
single-threaded. Use C when you want to know whether the gather misses reach DRAM.

## What's already committed (no code edits needed)
- `learner/decision_tree/training.cc` — hooks behind `#ifdef YDF_CALLGRIND_DEPTH`:
  - include block: `#include <valgrind/callgrind.h>`
  - **before** the kernel call (`ApplyProjectionsDepthwise1Pass`):
    `CALLGRIND_START_INSTRUMENTATION; CALLGRIND_ZERO_STATS;`
  - **after** it:
    `CALLGRIND_DUMP_STATS_AT("dw1_depth_<N>"); CALLGRIND_STOP_INSTRUMENTATION;`
  These are no-ops unless built with `-DYDF_CALLGRIND_DEPTH`. The START/STOP pair is
  what keeps the CSV load OUT of the cache simulation (see speed note below).
- `benchmarks/cachegrind/build_methodC.sh` — build with the define on `training.cc` only.
- `benchmarks/cachegrind/run_methodC.sh` — run under callgrind to a persistent dir.
- `benchmarks/cachegrind/parseC.py` — turn dumps into a per-depth CSV.

## Prerequisites
- `bazel`, Intel oneAPI (`icx`/`icpx`). If oneAPI lives elsewhere, set
  `ONEAPI=/path/to/compiler/latest/bin` before building.
- `valgrind` (3.22 used here) and `callgrind_annotate` on PATH.
- `python3`.
- The HIGGS CSV with a header row and a `class` label column. Default location
  `benchmarks/data/HIGGS_with_header.csv`; override with `HIGGS_CSV=...`.

## Run it (3 steps)
```bash
# 1. Build the instrumented binary
benchmarks/cachegrind/build_methodC.sh

# 2. Run under callgrind. Writes to $HOME/dw1_cachegrind_C (override METHODC_OUT).
#    ~5-10 min to load + minutes per depth; grows to ~depth 60 on HIGGS.
benchmarks/cachegrind/run_methodC.sh

# 3. Parse the per-depth dumps into a CSV
python3 benchmarks/cachegrind/parseC.py $HOME/dw1_cachegrind_C/cgout
```
Output columns:
`depth,iters,D1mr,l1_lines,useful_per_L1_line,DLmr,useful_per_DRAM_line`.

## Why it's fast now (the one critical flag)
`run_methodC.sh` passes `--instr-atstart=no`. Without it, callgrind cache-simulates
the **entire** ~8 GB CSV load (≈1.4k rows/s ⇒ ~2 h before any training — the prior
attempt died here). With it, the load and all non-kernel code run at light JIT
overhead (~30–40k rows/s), and only the kernel — bracketed by the START/STOP calls
in `training.cc` — is simulated. Load drops to ~5 min.

## Interpreting the numbers
Per depth, on the gather source line (`Dr` = data reads, `D1mr` = L1 read misses,
`DLmr` = last-level/DRAM read misses):
- The gather line issues **2 reads/iter** (`sel_ptr[i]` then `col[..]`), so
  `iters = Dr/2`.
- `sel_ptr` is stride-1 ⇒ ~`iters/16` L1 misses; the rest of `D1mr` is the scattered
  `col` gather ⇒ distinct 64B lines touched.
- **`useful_per_L1_line = iters / (D1mr − iters/16)`** — tracks Method A's
  `useful_per_line` per depth (cross-validation). Measured agreement on HIGGS is
  ~10–15%, with C slightly below A: A counts *distinct* lines (a lower bound), C
  counts *actual* L1 misses including capacity evictions/refetches. Values: ~8 at
  the top level falling toward ~1.5 through the bulk-work depths.
- **`useful_per_DRAM_line = iters / DLmr` is the payoff of Method C.** On HIGGS the
  kernel gathers across *all ~28 projection feature-columns* (~1.2 GB) per wide
  top-level node — far past the ~109 MB LLC — so `DLmr` is **large** (tens of
  millions of DRAM read misses) and roughly tracks `D1mr`: most L1 gather misses go
  all the way to DRAM. The gather is DRAM-bandwidth-bound, salvaging only ~5–8
  floats per 64B line fetched from memory at the shallow (bulk-work) depths. This is
  the cost Method A (geometry only) cannot see. (Earlier guess that one 44 MB column
  fits the LLC ⇒ `DLmr≈0` was wrong: the working set is the whole column *set*, not
  one column.)

## Gotchas
- **`-march=skylake`, not native.** Valgrind 3.22 can't decode AVX-512. The gather
  pattern is SIMD-width independent, so AVX2 is representative for cache misses.
- **Single tree, single thread.** Valgrind serializes threads; `--num_trees=1
  --num_threads=1` is both faster and gives clean per-depth dumps. (Method A uses
  many threads/trees; don't copy that here.)
- **Persistent output dir.** Write under `$HOME` (or repo), never `/tmp` — some
  cloud boxes auto-shutdown on inactivity and wipe `/tmp` on reboot.
- **Stale-cache caveat.** With START/STOP toggling, callgrind's simulated cache
  carries across the un-instrumented gaps. The effect is negligible here (each
  kernel gathers millions of rows over a working set ≫ any cache), but it means C
  is best for relative per-depth trends, not absolute first-access miss counts.
- **The build binary is instrumented.** Rebuild WITHOUT `--per_file_copt=...` before
  running any timing benchmark.
```bash
bazel build -c opt --config=profiler --config=depthwise_1_pass \
  --copt=-march=skylake //examples:train_oblique_forest   # clean, no hooks
```
