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
build:depthwise_1_pass            --cxxopt="-DDEPTHWISE_1_PASS=1"                       # shared-rows colwalk on the depth's hot nodes (gates DW1_HOT_MIN_ROWS / DW1_HOT_MIN_SHARE), stock Evaluate on the rest
build:dw1_colwalk_control         --config=depthwise_1_pass --cxxopt="-DDW1_COLWALK_CONTROL=1"  # col-sharing sweep only: no depth bag, no shared-rows scatter, DW1_HOT_* inert; #error without depthwise_1_pass
build:row_major_dataset_layout    --cxxopt="-DROW_MAJOR_DATASET_LAYOUT=1"
build:symmetric_depthwise_ap      --cxxopt="-DSYMMETRIC_DEPTHWISE_AP=1"
build:bfs_only                    --cxxopt="-DBFS_ONLY=1"                     # mutually exclusive with symmetric_*
build:oblique_gpu                 --cxxopt="-DOBLIQUE_GPU_ENABLED=1" --define=enable_cuda=1
build:coarse_chrono_profile          --cxxopt="-DCHRONO_PROFILE=1"                            # coarse base: top-level + node-bookkeeping/split-mgr/GBT scopes
build:fine_chrono_applyprojection    --cxxopt="-DCHRONO_PROFILE=1" --cxxopt="-DFINE_CHRONO_AP" # coarse + inner scopes of ProjectionEvaluator::Evaluate (sym / dw1)
build:fine_chrono_evaluateprojection --cxxopt="-DCHRONO_PROFILE=1" --cxxopt="-DFINE_CHRONO_EP" # coarse + inner scopes of EvaluateProjection (histogram / Cart)
build:nodewise_chrono                --cxxopt="-DCHRONO_PROFILE=1" --cxxopt="-DNODEWISE_CHRONO=1" # coarse + one CSV row per (node, projection) AP call, gated by depth
  # Three INDEPENDENT axes: each includes coarse but not the others. FINE-everywhere = pass both fine configs together.
build:inline_projection_evaluate  --cxxopt="-DINLINE_PROJECTION_EVALUATE"
build:enable_isnan --cxxopt="-DENABLE_ISNAN=1"
build:disable_binary_entropy_lookup --cxxopt="-DDISABLE_BINARY_ENTROPY_LOOKUP"
build:disable_std_upper_bound_vectorization --cxxopt="-DDISABLE_STD_UPPER_BOUND_VECTORIZATION=1"
  # SIMD upper_bound is default-ON with runtime cpuid+bin-count dispatch; this turns it off.
  # (No enable_* config needed to vectorize; the ISA is picked at runtime from the bin count.)
build --copt=-march=native --cxxopt="-O3"     # global
build:linux --repo_env=CC=icx --repo_env=CXX=icpx   # oneAPI icx pin (gcc is 30-40% slower on the hot path)
build:debug  --cxxopt=-O0 -g -fno-inline …    # breakpoints need this separate config
build:profiler --cxxopt=-O2 -g, no fission    # for VTune/perf
```

Env knobs (read once, cached in a static): `RM_MAX_ROWS` (node-size threshold, default
5000 baked by the harness `main()`; ∞ if binary run without harness), `DW1_MIN_DEPTH`
(default 0), `SYMMETRIC_MAX_DEPTH` (default
INT32_MAX; deeper levels hand off to DFS `GrowTreeLocal`), `DW1_HOT_MIN_ROWS`
(`depthwise_1_pass`, inert in `dw1_colwalk_control`, default 1000: a node is a fusion candidate iff `sel.size() >=` it,
everything smaller runs the stock per-node `Evaluate`; **0 = every node a candidate**. The
trained model is bit-identical at every value, so a hash sweep over
this knob — `compare_models.sh` on `--model_out_dir` outputs — is the correctness test;
sweep it for perf, `{250, 500, 1000, 2000, 4000}`, in Quick mode first. Not combinable with
`--config=nodewise_chrono`), `DW1_HOT_MIN_SHARE` (same builds, default 50:
percent of a candidate's distinct columns that must also be read by another candidate;
**0 = keep every candidate = the row gate alone**, the purity control — and with
`DW1_HOT_MIN_ROWS=0` too, ungated shared-rows on every node (this pair replaces the old
`dw1_shared_rows` config; it still pays the gate's own per-depth cost). Bit-identical at
every value, so the same hash sweep is the correctness test),
`DW1_HOT_OVERLAP_STATS=1` (one line per fused depth: candidates → kept/dropped, and
`bag=relabel|rebuild` — whether the gate cost that depth a concat+VQSort; single-thread only).

### Column-sharing stats — `DW1_COL_SHARE_OUT` (any `depthwise_1_pass` build)

Measures the quantity the depthwise kernels exist to exploit: how many nodes of a depth
reference the same column. Lives in `oblique_cpu_depthwise_1pass.cc`'s colstats block and is
**compiled out by default**: flip `constexpr bool kDw1DebugStats = true` there and rebuild,
or the env knobs below do nothing. That block assumes a **single training thread** (no
locking) — run it with `--num_trees=1 --num_threads=1`. Stats are taken over **the frontier
the kernel was handed** — so the same flags with `DW1_HOT_MIN_ROWS=0 DW1_HOT_MIN_SHARE=0`
(full frontier) and with the gates on (hot nodes only) give the two sides of the hot-gate
comparison, and since the gates leave trees bit-identical the rows join 1:1 on `(tree, depth)`.

- `DW1_COL_STATS` — `0` off, `summary` one line per (tree, depth), anything else
  (default) the per-column dump `c<id>: <nodes> nodes, <examples> examples` **plus** the
  summary line. `DW1_COL_STATS_ALL=1` adds untouched columns as `c17: 0`.
- `DW1_COL_SHARE_OUT=<path>` — same numbers as CSV (`tree,depth,nodes,rows,
  cols_touched,num_features,refs,pairs,share,shared_cols,max_nodes_col,useful,swept,eff,
  nodewise,amort`); works with `DW1_COL_STATS=0`. First write of the process truncates.
- `benchmarks/utils/compare_dw1_colshare.py full.csv hot.csv -l full -l hot` joins two runs.
- Set `DW1_NODE_SIZES_DEPTH=-1` too, or the unrelated node-size dump also fires.

`share` = mean nodes per touched column (1.0 = no sharing); `swept` = `cols_touched × depth
rows` = what the shared-rows colwalk streams; `eff` = `useful/swept`; `amort` = stock
nodewise read volume / `swept`. **`amort` counts floats, not lines** — the sweep is
sequential (16 floats/line) while the nodewise gather gets ≈1.9 useful floats/line at depth
(§13), so the line-level ratio is ≈8× the printed one. These are exact counts, deterministic
given seed+data, so unlike timings they are valid off a Mac.

Measured 2026-07-25, trunk, 2 trees, `DW1_HOT_MIN_ROWS=1000`, totals over all fused
depths (full frontier → hot only): **100k×29** share 115.4 → 4.9, eff 0.323 → 0.336, swept
62.7M → 37.9M floats; **8192×4096** share 2.54 → 1.02, eff 0.044 → 0.400, swept 486M → 16.7M
floats, hot keeps 31 % of the fused rows. Conclusion: the hot gate does destroy column
sharing (23× fewer nodes/column on tall-narrow, down to none on wide) but sharing was not
converting into savings — `eff` is flat/better because `cols_touched` and the bag shrink
together, and the sweep's read volume falls with it.

### `--config=nodewise_chrono` — per-node ApplyProjection cost

Answers "does a small subset of nodes carry most of the AP cost?", which the depth-aggregated
CSVs cannot: they sum AP over every node at a depth. This axis leaves the coarse tables
byte-identical and adds a **second sink on the same clock read** in
`ProjectionEvaluator::Evaluate` (`ScopedNodewiseApTimer`, `utils/parallel_chrono.h`) — no
extra `steady_clock::now()`, so the interval is the one the coarse tables already report.
The accumulation happens after the closing clock read, outside the interval it describes — so
`ApplyProjection` is undistorted, but the *enclosing* scopes (`NodeTrain`,
`ObliqueSplitSearch`, `TreeTrain`) absorb three adds per projection and one `push_back` per
node. Don't A/B e2e runtime against a non-nodewise build.

Rows are per **node**: the node's Evaluate calls accumulate in thread-local state and are
emitted once, when its `NodeTrain` scope closes. `nnz` and `ap_ns` are sums over the node's
projections; the row set is fixed within a node, so `n_gathers = n_rows·nnz` survives the sum.

```text
NODEWISE_TREE=0             tree to record; -1 = all trees
NODEWISE_DEPTHS=5,10,15,20  depth list, or "*" for every depth
NODEWISE_OUT=nodewise_ap.csv
NODEWISE_RESERVE=<n>        records reserved per recording tree (default 1<<20 single-tree,
                                1<<16 all-trees) — pre-allocated before the pool starts so no
                                realloc lands mid-measurement
# columns: tree,depth,node_id,n_rows,nnz,n_gathers,ap_ns,num_proj   (sorted by depth, node_id)
#   node_id   heap index: root = 1 (both growers enter at depth 1), children of i are 2i (neg)
#             and 2i+1 (pos) ⇒ depth d holds ids [2^(d-1), 2^d), and the bits after the leading
#             1 spell the root→node branch path. 0 = unknown (unrooted subtree / depth > 63).
#   nnz       Σ projection.size() over the node's projections
#   n_gathers n_rows*nnz; ns/gather = ap_ns/n_gathers is the rate metric (see §13 depth decay)
#   num_proj  Evaluate calls for the node (0 ⇒ AP never ran)
```

The id is threaded down the growth stack: `internal::NodeAndExamples::node_id`
(training.h, `#ifdef NODEWISE_CHRONO` so other builds keep the struct byte-identical), set on
the root push in `GrowTreeLocal`/`GrowTreeLocalBFS` and on both children after NodeTrain's
pushes via `chrono_prof::NodewiseChildId`. Works under DFS and BFS alike. The one path that
yields `node_id = 0` is the fused kernels' BFS→DFS handoff, which grows a detached subtree —
and those builds are excluded from this axis anyway.

`NodewiseNodeScope` (training.cc, in NodeTrain's `#ifdef CHRONO_PROFILE` block) arms the
recorder and emits the row. Nodes that never reach AP would otherwise vanish and inflate every
concentration statistic, so they emit a `num_proj=0, nnz=0, ap_ns=0` row. In practice that path
is dormant under the harness (`--min_examples=1` for Bagging, and the splitter enforces
min_examples on both children, so every node that enters NodeTrain reaches AP); a "leaf" here
is a node that ran AP and found no valid split, and appears with its real cost.

Verified invariants, checked at dump time and logged as `NODEWISE_AP selfcheck`: per
(tree, depth), Σ`ap_ns` == the coarse `ProjEval` cell **exactly**, and the row count ==
the coarse `nodes` count. Externally verified too: ids are unique and inside `[2^(d-1), 2^d)`
at every depth, and a gated shallow depth returns a complete level (depth 5 ⇒ 16 rows,
ids 16…31). Combine only with the stock nodewise kernel — the fused symmetric /
dw1 Apply paths never call `Evaluate`, produce no records, and trip the self-check.

Volume: HIGGS, one tree, depths 5/10/15/20 ≈ 55k rows ≈ 1.8 MB. One full tree at every depth
≈ 2.6M rows ≈ 80 MB, so widen the ladder deliberately. Note the default ladder covers only
2.6 % of a tree's nodes and 15 % of its AP, with **no** coverage of depths 21–40 where ~88 %
of the nodes live — enough for within-depth spread, not for a global Lorenz curve. For that,
`NODEWISE_DEPTHS` = every 5th depth plus all of d≤12 reaches ~48 % of AP for ~8 MB.

## Harness: `examples/train_oblique_forest.cc`

Defaults that define the benchmark protocol: `--feature_split_type=Oblique`,
`--numerical_split_type="Dynamic Random Histogram"`, `--histogram_num_bins=64`,
`--dynamic_split_threshold=250`, `--max_num_projections=1000`,
`--num_projections_exponent=.5`, `--projection_density_factor=1.5`,
`--growing_strategy=Local`, `--ensemble_method=Bagging`,
`--bootstrap_training_dataset=true`, `--num_threads=1` (runtime.sh uses `-1` = all),
`--num_trees=240` (runtime.sh overrides), `--seed=1234`, `--tree_depth=-1`,
`--min_examples` set to 1 for Bagging. `main()` bakes `RM_MAX_ROWS=5000` if unset and
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
