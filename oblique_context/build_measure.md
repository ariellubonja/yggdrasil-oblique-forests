# build_measure.md — building, running, measuring

> Shard of `OBLIQUE_CONTEXT.md` (the lean core). Read this when a question is
> about the `.bazelrc` experiment configs, env knobs, the `train_oblique_forest`
> harness defaults / input modes / trunk generator, dataset shapes, or the
> measurement tooling (`runtime.sh`, `accuracy.sh`, `parallel_chrono.py`).
> The experiment *workflow* itself (loop, significance gate, logging contract)
> lives in `AGENTS.md`. Snapshot as of 2026-07-04, branch `rebased-main`, commit
> `c80ffbf7`. Line numbers drift; grep for the symbol if a ref misses.

---

## `.bazelrc` experiment configs (the authoritative list on this branch)

```text
build:depthwise_1_pass            --cxxopt="-DDEPTHWISE_1_PASS=1"
build:dw1_shared_rows             --config=depthwise_1_pass --cxxopt="-DDW1_SHARED_ROWS=1"
build:row_major_dataset_layout    --cxxopt="-DROW_MAJOR_DATASET_LAYOUT=1"
build:symmetric_depthwise_ap      --cxxopt="-DSYMMETRIC_DEPTHWISE_AP=1"
build:symmetric_nodewise_control  --cxxopt="-DSYMMETRIC_NODEWISE_CONTROL=1"   # don't combine with the above
build:bfs_only                    --cxxopt="-DBFS_ONLY=1"                     # mutually exclusive with symmetric_*
build:oblique_gpu                 --cxxopt="-DOBLIQUE_GPU_ENABLED=1" --define=enable_cuda=1
build:chrono_profile              --cxxopt="-DCHRONO_PROFILE=2"   # every scope
build:chrono_profile_coarse       --cxxopt="-DCHRONO_PROFILE=1"   # top-level scopes only, lower overhead
build:inline_projection_evaluate  --cxxopt="-DYDF_INLINE_PROJECTION_EVALUATE"
build:enable_applyprojection_isnan --cxxopt="-DENABLE_APPLYPROJECTION_ISNAN=1"
build:disable_binary_entropy_lookup --cxxopt="-DDISABLE_BINARY_ENTROPY_LOOKUP"
build:enable_std_upper_bound_avx2 / _avx512   # SIMD upper_bound; define is also default-on globally
build --copt=-march=native --cxxopt="-O3"     # global
build:linux --repo_env=CC=icx --repo_env=CXX=icpx   # oneAPI icx pin (gcc is 30-40% slower on the hot path)
build:debug  --cxxopt=-O0 -g -fno-inline …    # breakpoints need this separate config
build:profiler --cxxopt=-O2 -g, no fission    # for VTune/perf
```

Env knobs (read once, cached in a static): `YDF_RM_MAX_ROWS` (node-size threshold, default
5000 baked by the harness `main()`; ∞ if binary run without harness), `YDF_DW1_BLOCK_FLOATS`
(default 64 Mi floats), `YDF_DW1_MIN_DEPTH` (default 0), `YDF_SYMMETRIC_MAX_DEPTH` (default
INT32_MAX; deeper levels hand off to DFS `GrowTreeLocal`).

## Harness: `examples/train_oblique_forest.cc`

Defaults that define the benchmark protocol: `--feature_split_type=Oblique`,
`--numerical_split_type="Dynamic Random Histogram"`, `--histogram_num_bins=64`,
`--dynamic_split_threshold=250`, `--max_num_projections=1000`,
`--num_projections_exponent=.5`, `--projection_density_factor=1.5`,
`--growing_strategy=Local`, `--ensemble_method=Bagging`,
`--bootstrap_training_dataset=true`, `--num_threads=1` (runtime.sh uses `-1` = all),
`--num_trees=240` (runtime.sh overrides), `--seed=1234`, `--tree_depth=-1`,
`--min_examples` set to 1 for Bagging. `main()` bakes `YDF_RM_MAX_ROWS=5000` if unset and
clears any active row-major matrix.

Input modes: `--input_mode=csv --train_csv=… --label_col=…` (HIGGS:
`benchmarks/data/HIGGS_with_header.csv|class`), `--input_mode=trunk --rows=R --cols=C`
(synthetic), `tfrecord`. `--dataset_layout=row` fills the `RowMajorFeatureMatrix`
(requires the config). Trunk generator (deterministic per-column `minstd_rand`; first 256
columns informative, two Gaussian classes at ±1/√(j+1), rows [0,R/2)=class 1, [R/2,R)=class 2):

```cpp
dataset::VerticalDataset MakeTrunkDataset(const dataset::proto::DataSpecification& spec,
                                          int64_t rows, int cols, uint32_t seed) {
  // …
  constexpr int kNInformative = 256;
  const int ninform = std::min(kNInformative, cols);
  std::vector<float> mu0(cols, 0.f), mu1(cols, 0.f);
  for (int j = 0; j < ninform; ++j) {
    const float f = 1.f / std::sqrt(float(j + 1));
    mu0[j] = -f; mu1[j] = f;
  }
  for (int j = 0; j < cols; ++j) {
    std::seed_seq seq{seed, static_cast<uint32_t>(j)};
    std::minstd_rand rng(seq);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto* col = ds.MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(j)->mutable_values();
    for (int64_t i = 0; i < rows; ++i) {
      const bool cls1 = (i >= rows / 2);
      (*col)[i] = (cls1 ? mu1[j] : mu0[j]) + normal(rng);
    }
  }
  // labels: 1-based, (i >= rows/2) ? 2 : 1
}
```

Standard dataset shapes: HIGGS 11M×29 (tall-narrow), trunk 3M×4096 (~49 GB), 1.5M×4096,
300k/150k×40k, 30k/10k×400k (ultra-wide).

## Measurement tooling

- **Verdict:** `benchmarks/evaluation/runtime.sh` (e2e; Quick then Full; writes CSV + `.meta`
  provenance sidecar — never delete `.meta`), `accuracy.sh` (must match exactly for
  bit-identity experiments). Drive variants via `EXTRA_BAZEL_CONFIGS` / `EXTRA_TRAIN_ARGS`;
  per-dataset isolation via `CSV_DATASETS_OVERRIDE` / `TRUNK_DATASETS_OVERRIDE`.
- **Insight:** `benchmarks/profiling/parallel_chrono.py --bazel_config=NAME` — per-tree-depth
  Σ per chrono scope; results under
  `benchmarks/results/per_function_timing/<CPU>/<projection> | <split> | /<dataset>/`.
  Reuse committed baselines; new baselines: `n_trees=5, rows=3000000, num_threads=1`.
- Median of 3 trees via `--num_trees=3` in ONE process (never 3 process invocations).
- Machines: dev Mac (arm64; plain `bazel build -c opt` works, icx pin ignored) — **numbers
  from the Mac don't count**; measurement boxes: AWS m7i (Xeon 8488C, 48 vCPU) and the i9-185H laptop (E-cores must be off for timing —
  `benchmarks/utils/set_cpu_e_features.sh`; scripts handle it). perf yes; VTune/Advisor
  memory-access profiles: never (freezes the box). Significance gate: **<15 % e2e = failed
  experiment** (log it anyway); ★ at ≥20 %.
