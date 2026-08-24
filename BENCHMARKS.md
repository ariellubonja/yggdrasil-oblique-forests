# A/B benchmark: upstream May-2025 (this branch) vs the fork

Branch `ydf-may-2025` = upstream YDF @ `2d1454cc` (2025-05-28) + the ported benchmark
harness (`examples/train_oblique_forest.cc`), a farmhash repin, and the fork's timing
anchor / early-exit in `random_forest.cc`, so `runtime.sh` / `bench_repeat_cmd` parse
the same `Training block took:` line on both sides.

## Results (2026-08-24)

Trunk synthetic dataset, 480 trees, 48 threads, seed 1, 1 run each, both binaries built
with Intel icx 2025.2.1 (`-O3 -march=native`), AWS m7i (Xeon Platinum 8488C). Metric is
the **training block** (`Training block took:` log line) — full `TrainWithStatus` adds
~14 s of near-identical post-training model finalization on both sides at 10k, which
dilutes the ratio (29.12 s vs 16.25 s ⇒ 1.79× e2e at 10k×4096).

| Trunk shape | Upstream Exact (this branch) | Fork defaults (Dynamic Random Histogram, 64 bins, threshold 250) | Speedup |
|---|---|---|---|
| 10k × 4096  | 15.00 s  | 1.80 s  | **8.3×** |
| 100k × 4096 | 166.63 s | 27.59 s | **6.0×** |

## Build (this branch — WORKSPACE-based, needs bazel 6.5.0, not the fork's 7.7.0)

```sh
source /opt/intel/oneapi/setvars.sh
bazel-6.5.0 build -c opt --config=linux_cpp17 --copt=-march=native --cxxopt=-O3 \
  --repo_env=CC=icx --repo_env=CXX=icpx //examples:train_oblique_forest
```

`.bazelversion` pins 6.5.0, so bazelisk also works. Only
`--numerical_split_type Exact|Random|"Equal Width"` exist here (no Dynamic variants,
no `RM_MAX_ROWS`, no row-major layout).

## Run

This branch (Exact splits):

```sh
./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --cols 4096 \
  --num_trees=480 --num_threads=-1 --seed=1 --numerical_split_type Exact
```

Fork (its defaults — same command minus `--numerical_split_type`):

```sh
./bazel-bin/examples/train_oblique_forest --input_mode trunk --rows 100000 --cols 4096 \
  --num_trees=480 --num_threads=-1 --seed=1
```

The binary `exit(0)`s after printing `Training block took:` unless `NO_EARLY_EXIT=1` is
set (auto-set when `--test_csv` / `--model_out_dir` are given, which need the model).
