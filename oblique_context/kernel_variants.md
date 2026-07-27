# kernel_variants.md — the per-node driver (full body) + this fork's AP kernel variants

> Shard of `OBLIQUE_CONTEXT.md`. The core keeps the driver's skeleton; this shard has the
> **full** body with every variant `#ifdef`, plus the variant kernels themselves (DW1
> col-sharing / shared-rows, symmetric depthwise AP, the subtree-gather dead end). Snapshot
> 2026-07-04, `rebased-main` @ `c80ffbf7`; grep for symbols if a line ref misses.

---

## The per-node driver: `FindBestConditionSparseObliqueTemplate` (oblique.cc:129) — full body

Where all kernel variants hook in (regression templates identical):

```cpp
template <typename LabelStats>
absl::StatusOr<bool> FindBestConditionSparseObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, /* configs… */,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (config_link.numerical_features().empty()) return false;

  int num_projections = override_num_projections.value_or(
      GetNumProjections(dt_config, config_link.numerical_features_size()));
  const float projection_density =
      std::clamp(dt_config.sparse_oblique_split().projection_density_factor() /
                     config_link.numerical_features_size(), 0.f, 1.f);

  Projection best_projection, current_projection;
  float best_threshold;
  auto& projection_values = cache->projection_values;

  CHRONO_BEGIN_COARSE(find_oblique_setup);
  ProjectionEvaluator projection_evaluator(train_dataset,
                                           config_link.numerical_features());   // per node, O(F)
  // TODO: Cache.
  const auto selected_labels = ExtractLabels(label_stats, selected_examples);   // labels[bag[i]] gather
  std::vector<float> selected_weights;                                          // stays empty (unweighted)
  if (!weights.empty()) selected_weights = Extract(weights, selected_examples);
  std::vector<UnsignedExampleIdx> dense_example_idxs(selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);
  CHRONO_END_COARSE(find_oblique_setup, …kFindObliqueSetup);

  // … [dynamic downgrade — see split_search.md "Per-node DYNAMIC downgrade"] …

#if defined(SYMMETRIC_DEPTHWISE_AP) || defined(OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS)
  // Slab present ⇒ skip SampleProjection + Evaluate, split-search over its slices.
  const bool has_precomputed_projected =
      !internal_config.precomputed_projected_values.empty() &&
      internal_config.depthwise_projection_defs != nullptr &&
      internal_config.depthwise_monotonic != nullptr;
  if (has_precomputed_projected) {
    const auto& depth_projs = *internal_config.depthwise_projection_defs;
    const auto& depth_mono = *internal_config.depthwise_monotonic;
    const size_t rows_n = selected_examples.size();
    const float* slab = internal_config.precomputed_projected_values.data();
    for (size_t proj_idx = 0; proj_idx < depth_projs.size(); ++proj_idx) {
      if (depth_projs[proj_idx].empty()) continue;
      const absl::Span<const float> values_span =
          absl::MakeConstSpan(slab + proj_idx * rows_n, rows_n);
      ASSIGN_OR_RETURN(const auto result,
          EvaluateProjection(dynamic_dt_config, label_stats, dense_example_idxs,
                             selected_weights, selected_labels, values_span,
                             internal_config, depth_projs[proj_idx].front().attribute_idx,
                             constraints, depth_mono[proj_idx], best_condition,
                             cache, random));
      if (result == SplitSearchResult::kBetterSplitFound) {
        best_projection = depth_projs[proj_idx];
        best_threshold = best_condition->condition().higher_condition().threshold();
      }
    }
  }
  // [DEPTHWISE_1_PASS && !DW1_COLWALK_CONTROL] else if (defs && mono): the DW1 COLD branch —
  // handed-down projections, no slab ⇒ loop Evaluate + EvaluateProjection here. Mandatory:
  // falling through to the main loop would re-sample and consume the RNG twice.
  else
#endif
  {
// MAIN LOOP
  for (int projection_idx = 0; projection_idx < num_projections; projection_idx++) {
    int8_t monotonic_direction;
    {
      CHRONO_SCOPE_COARSE(…kSampleProjection);
      SampleProjection(config_link.numerical_features(), dynamic_dt_config,
                       train_dataset.data_spec(), config_link, projection_density,
                       &current_projection, &monotonic_direction, random);
    }
    RETURN_IF_ERROR(projection_evaluator.Evaluate(
        current_projection, selected_examples, &projection_values));       // ★ ApplyProjection
    ASSIGN_OR_RETURN(const auto result,
        EvaluateProjection(dynamic_dt_config, label_stats, dense_example_idxs,
            selected_weights, selected_labels, projection_values,
            internal_config, current_projection.front().attribute_idx,
            constraints, monotonic_direction, best_condition, cache, random));
    if (result == SplitSearchResult::kBetterSplitFound) {
      best_projection = current_projection;
      best_threshold = best_condition->condition().higher_condition().threshold();
    }
  }
  }

  if (!best_projection.empty()) {
    RETURN_IF_ERROR(SetCondition(best_projection, best_threshold,
                                 train_dataset.data_spec(), best_condition));
    return true;
  }
  return false;
}
```

`SetCondition` (oblique.cc:1004) materializes the winner into the node proto
(`oblique_condition`: attributes[], weights[], na_replacements[] = column means, threshold).

Upstream: `FindBestConditionOblique` (oblique.cc:777) switches on `split_axis_case` (sparse vs
MHLD); `FindBestConditionSingleThreadManager` (training.cc:1441) calls it under
`kObliqueSplitSearch`, **then still runs the axis-aligned loop over `candidate_attributes`**
(near-no-op for pure-oblique runs, but it exists — `kGetCandidateAttributes`). RF never sets
`internal_config.split_finder_processor`, so `FindBestConditionManager` (training.cc:1896)
always takes the single-thread branch.

---

## Kernel variants (this fork's experiments)

Rule (`AGENTS.md` + ablation memory): **one idea = one `--config` = one branch; never stack
ideas in a measurement.** Controls stay pure. On this branch:

| Variant | Config | File / function | Status |
|---|---|---|---|
| Stock nodewise Evaluate | *(none)* | `oblique.cc` `ProjectionEvaluator::Evaluate` | baseline |
| BFS-only control | `--config=bfs_only` | `GrowTreeLocalBFS` fallback branch | scheduler ablation |
| DW1 depthwise 1-pass (shared-rows colwalk + hot gates) | `--config=depthwise_1_pass` + `DW1_HOT_MIN_ROWS` / `DW1_HOT_MIN_SHARE` | `oblique_cpu_depthwise_1pass.cc`; gates in `GrowTreeLocalBFS`; cold branch in `oblique.cc`; `AdvanceDepthBagHot` | colwalk for the depth's big column-sharing nodes, stock `Evaluate` for the rest. Non-monotone overlap gate ⇒ that depth's bag falls back to concat+VQSort. Bit-identical at every threshold pair; **unmeasured (2026-07-25)** |
| └ ungated shared-rows (`DW1_HOT_MIN_ROWS=0 DW1_HOT_MIN_SHARE=0`) | runtime, no rebuild | same file | ⛔ 1.3–3.8× slower; postmortem in core §13 |
| DW1 col-sharing control | `--config=dw1_colwalk_control` | same file, `#ifdef DW1_COLWALK_CONTROL` | ≤15 % slower than BFS; "col sharing via cache residency doesn't work at scale" |
| Symmetric depthwise AP | `--config=symmetric_depthwise_ap` | `oblique_cpu_symmetric_depthwise_ap.cc` | ✚ changes model semantics; wins wide-trunk, ties BFS on HIGGS |
| Subtree gather cache | *(removed 2026-07-16)* | recover via commit `9f32e817` | ⛔ +43 % (≈2 % feature overlap ⇒ never amortizes) |
| Row-major store | `--config=row_major_dataset_layout` + `--dataset_layout=row` | `RowMajorFeatureMatrix` via `AttributeValue` | layout experiment |

### DW1: `ApplyProjectionsDepthwise1Pass` (oblique_cpu_depthwise_1pass.cc)

> **2026-07-23: block machinery removed.** The per-block budget (`Dw1BlockFloats`), `Dw1Task`,
> the big-node projection-major path (`EvaluateNodeProjMajor`, kDw1SweepBig) and the
> shared-rows per-block arenas are gone — the budget was long disabled, so every depth already
> ran as one block. Now a single depth-wide column-centric sweep reading the depth bag
> directly. Verified bit-identical vs the block version. Old code: commit `c6357624`.

Idea: at mid depths the frontier holds thousands of small nodes and N·K·nnz ≫ F. Invert the
loop: bucket every (node, projection, weight) reference of the depth by column, then walk the
touched columns once ascending.

```cpp
struct ColEntry { int32_t node; int32_t proj; int32_t col; float weight; };

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    DepthBagState* bag_state, absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();
  int max_attr = /* max over numerical_features */;
  // not in the colwalk control: total_rows = Σ selected_examples_per_node[n].size()

  // ── Phase 1: PreSize (kDw1PreSize, ~5% of AP) — slab pre-size, zero-init
  //    (the column sweep accumulates).
  for (size_t n = 0; n < N; ++n) out_projected[n].assign(sel_n * projs_n, 0.f);
  // The depth bag arrives ready in *bag_state: the DRIVER advances it
  // (AdvanceDepthBagHot), since the relabel needs spans the kernel never sees.

  // ── Phase 2: Sweep (kDw1Sweep, dominant)
  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);
  if (!evaluator.IsColumnMajor()) return UnimplementedError(…);  // alternate layouts
  // Bucket every (node,proj,weight) ref by column: entries pushed node-major,
  // touched ascending, counting sort → `sorted`, col_count = per-column cursor. [≤2%]

#ifndef DW1_COLWALK_CONTROL
  // Shared-rows colwalk over the whole depth. Per touched column c
  // (kDw1SweepColWalk, ~86% of AP):
  //   1. group c's refs by node → runs in ref_proj/ref_w (kDw1ColWalkGroupByNode, ~0%)
  //   2. ONE ascending pass over the depth bag (kDw1ColWalkBagScatter — the hot loop):
  for (size_t s = 0; s < total_rows; ++s) {
    const int32_t n = single ? 0 : nob[s];        // node_of_bag
    const int32_t off = node_ref_off[n];
    if (off < 0) continue;   // owning node has no ref to column c (74–98% skipped!)
    const float v = col[bag[s]];                  // dense-ish column read
    const int32_t local = node_local[n]++;        // running counter == slab slot (sorted bag)
    float* slab = out_projected[n].data();
    for (int32_t t = 0; t < cnt; ++t)
      slab[ref_proj[off + t] * rows_n + local] += ref_w[off + t] * v;  // scatter-write
  }
  //   3. reset node_ref_off for the touched nodes.
#else
  /* Col-sharing colwalk (kDw1SweepColWalk, ~93% of AP). Per column, replay its
     (node,proj) entries; each re-gathers the column at ITS node's rows: */
  size_t pos = 0;
  for (const int32_t c : touched) {
    const float* col = evaluator.AttributeData(c);
    const size_t end = static_cast<size_t>(col_count[c]);
    for (; pos < end; ++pos) {
      const ColEntry& e = sorted[pos];
      const auto sel = selected_examples_per_node[e.node];
      float* o = out_projected[e.node].data() + e.proj * sel.size();
      for (size_t i = 0; i < sel.size(); ++i)     // ★ the measured gather
        o[i] += e.weight * col[sel.data()[i]];
    }
  }
#endif
  // Single-threaded by design: RandomForest already trains one tree per thread.
}
```

### DW1 hot-nodes hybrid — the default `depthwise_1_pass` kernel (2026-07-25)

**One idea:** run the shared-rows colwalk on the depth's **big, column-sharing nodes only**;
every other node is evaluated by the stock `Evaluate`. The kernel itself is unchanged — it
just gets a smaller frontier. Two gates select it: the **row gate** and the **column-overlap
gate**. One config, not two: the overlap gate repairs the row gate's failure mode, so the row
gate alone was never worth shipping — `DW1_HOT_MIN_SHARE=0` reproduces it as the purity control.

#### Part 1 — the row gate (`DW1_HOT_MIN_ROWS`)

Motivation (`nodewise_ap.csv`, HIGGS tree 0, depths 5/10/15/20): nodes above the pooled mean
`n_gathers` are **7.5 % of nodes but 91.8 % of AP time** (90.8 % of rows). The shared-rows
postmortem's failure (a) — scatter-write RFO amplification — scales with the **number** of
nodes at a depth (one interleaved slab write stream per node-with-a-ref per column pass): at
depth 20 that is 45.8 k streams ≈ 2.9 MB ≫ the per-thread cache share. The hot set caps it at
~2–4 k streams (~120–230 KB) at every depth while keeping most of the fused work. Failure (b)
(the `off<0` skip-scan) is **not** addressed: on HIGGS a column is referenced by ~27 % of
nodes regardless, so with K≈2 k hot nodes ~500 still share each column pass; on wide trunks
sharing collapses to ~1 node/column with or without the gate (wide stays symmetric's
territory). Known ceiling: the recorded ladder covers ~15 % of a tree's AP and the hot row
share falls with depth (70.6 % at d20), so whole-tree hot AP share is likely ~50–65 %.

**Gate.** Hot iff `sel.size() >= DW1_HOT_MIN_ROWS` (default 1000 ≈ the mean-`n_gathers` cut,
since `n_gathers ≈ 9·rows` on HIGGS). Rows rather than `n_gathers` because the gate must be
**monotone down the tree** (child rows ≤ parent rows ⇒ a hot node's parent is hot), which is
what lets the depth bag telescope (part 2 gives that up). `=0` ⇒ every node a candidate ⇒ with
`DW1_HOT_MIN_SHARE=0`, exactly the ungated shared-rows kernel. Per-depth "above average" was
rejected: at depth 5 it keeps 1 of 16 nodes despite all 16 being huge.

**Sampling is untouched** — all nodes still sample their own projections at depth level in
node-major RNG order. The gate only picks *which kernel evaluates* a node, and both accumulate
in ascending attribute order into an fp32 zero ⇒ **the model is bit-identical at every
threshold value**, which is the correctness check.

Flow inside the `DEPTHWISE_1_PASS` branch of `GrowTreeLocalBFS` (all `#ifndef
DW1_COLWALK_CONTROL`; the control build is unchanged):

```text
sample all_node_projs[n][p]                  (unchanged, full frontier)
hot_of_node[n] = k or -1 ; hot_nodes[k] = n  (gate on sel_spans[n].size())
hot_sel_spans[k] = sel_spans[n] ; hot_projs[k] = std::move(all_node_projs[n])
                                             (move: hot defs point into hot_projs,
                                              cold defs stay in all_node_projs)
if K > 0:
  { CHRONO_SCOPE_COARSE(kProjectionEvaluate);   // driver-side, sequential (not nested)
    AdvanceDepthBagHot(full sel_spans, hot_sel_spans, first_child,
                       prev_hot_to_full, hot_of_node, hot_rows, &depth_bag_state); }
  ApplyProjectionsDepthwise1Pass(hot_sel_spans, hot_projs, …, projected /*size K*/)
else: depth_bag_state.valid = false           // latches: monotone gate ⇒ no deeper hot node
capture_prev_nodes(depth_batch) ; prev_hot_to_full = hot_nodes
per node n: defs = hot ? &hot_projs[k] : &all_node_projs[n];  mono = &all_node_mono[n];
            slab = hot ? projected[k] : (empty)   → NodeTrain
```

- **Why the driver advances the bag** (`oblique_cpu_depthwise_bag.{h,cc}`,
  `AdvanceDepthBagHot`): the bag covers only hot rows and its labels are hot indices, but the
  parent→child hop runs in the **full** node domain (a hot parent's rows may land in a cold
  child, and the membership test consumes child spans in order). Only the driver holds the full
  spans + both index maps. Relabel = the same streaming pass plus two index lookups and one
  extra drop condition (new owner cold — same mechanism as the leaf drop); self-validation +
  concat/VQSort fallback over the hot spans (whose positions *are* the hot labels) unchanged.
- **The cold branch in `oblique.cc` is mandatory:** a cold node gets defs+mono but no slab, so
  without the `else if (defs && mono)` branch it would fall into the main loop and **re-sample**,
  consuming the RNG twice.
- **Chrono:** hot AP bills to `ProjectionEvaluate` at depth level (bag advance included); cold
  AP bills to the same cell from inside `NodeTrain`, so at a mixed depth `TreeTrain =
  ΣNodeTrain + ΣAP + ΣSampleProjection` no longer partitions cleanly. Not combinable with
  `--config=nodewise_chrono` (hot nodes bypass `Evaluate` ⇒ the selfcheck trips).
- Side effect: peak slab memory shrinks (slabs for hot nodes only).
- **Cost of the gate — column sharing** (2026-07-25, machine-independent counts, 2 trees,
  `T=1000`, full frontier → hot; `DW1_COL_SHARE_OUT`, see `build_measure.md`): trunk 100k×29
  mean nodes/column **115.4 → 4.9**, 8192×4096 **2.54 → 1.02** (none). That sharing bought
  nothing: the colwalk streams `cols_touched × bag rows` and both shrink together, so the
  landed fraction is **flat on tall-narrow (0.323 → 0.336), 9× better on wide (0.044 → 0.400)**
  with swept floats down 1.7× and 29×. Swept volume, not sharing, is the figure of merit.

**Verified 2026-07-25** (macOS, correctness only — perf must come from m7i/8488C): trunk
20k×32 / 100k×29 / 8192×4096, `nodes-*` hash identical across `DW1_HOT_MIN_ROWS ∈ {0, 1, 100,
250, 1000, 4000, 10000, 1e9}` **and** to the then-separate ungated `dw1_shared_rows` build;
same under `--cxxopt=-UNDEBUG` (bag sortedness, `node_of_bag` sizing, kernel asserts). A trace
on 20k×32 (T=1000) confirmed the shape: the relabel fires at every fused depth after the first
(10/11; the fallback is the first fused depth), the gate prunes hard (depth 8: 122 → 2 hot;
depth 11: 302 → 2), and `hot=0` latches from depth 13 down.

#### Part 2 — the column-overlap gate (`DW1_HOT_MIN_SHARE`)

Applied to the row gate's candidates in the same place. The row gate bounds the kernel's
*write streams*; this one bounds its *wasted reads*.

**Why.** The colwalk's unit of work is one ascending pass over the whole depth bag per
**touched** column. A candidate whose columns no other candidate reads buys a full-bag pass
for each while consuming only its own rows of it — and lengthens every other column's pass by
those rows. At the limit (one hot node) the "shared" kernel degenerates into the stock gather
plus scatter-write bookkeeping: strictly worse than not fusing. Measured before building it
(2026-07-25): trunk 8192×4096 hot frontier `share = 1.02` nodes/column, `eff = 0.40`.

**Criterion.** Of a candidate's **distinct** columns (a node re-reading a column from several
projections is one reader — one pass serves them all), the fraction also read by another
candidate must be ≥ `DW1_HOT_MIN_SHARE` % (default 50; **0 = row gate alone**, the purity
control). ONE pass: dropping a node only lowers other columns' reader counts, so iterating
would drop strictly more, with no fixpoint worth the cost or the instability. O(refs) per
depth plus one O(max_attr) counter array hoisted to per-tree (reset over touched columns only).

**Monotonicity is deliberately given up** (user decision, 2026-07-25). Sharing depends on what
a node's *siblings* sampled, so a hot child no longer implies a hot parent — exactly the
precondition of the O(bag) relabel. `RelabelBagForNewDepthHot` self-validates and would fall
back on its own, but only after a wasted full bag pass, so the driver detects the break up
front (mark the children of `prev_hot_to_full` via `first_child`, check every hot node is
marked; O(num_nodes)) and clears `depth_bag_state.valid`. `DW1_HOT_OVERLAP_STATS=1` reports
`bag=relabel|rebuild` per depth so the cost is visible.

**Measured gate behaviour** (2026-07-25, 1 tree, `MIN_ROWS=1000`). trunk 8192×4096: at 50 % it
drops **every** candidate at every depth — i.e. switches the fused kernel off on wide, the
correct call given `eff` there. trunk 100k×29: a near-no-op through the middle depths (all
kept, `share` in the hundreds) firing exactly on the degenerate tail (depths 13-14, 1-2
candidates → all dropped); at 100 % it also bites at depths 4-5 and 11-12. Tail cleanup on
tall-narrow, kill switch on wide.

**Verified 2026-07-25** (macOS, correctness only): `nodes-*` hash on trunk 20k×32 and
8192×4096 identical across `DW1_HOT_MIN_SHARE ∈ {0, 50, 100}` × `MIN_ROWS ∈ {0, 1000}` **and**
equal to the then-separate `dw1_shared_rows` build (`cfa6ab94…` / `00dd44bc…`); same under
`--cxxopt=-UNDEBUG`, zero DCHECK failures. Both bag paths live on one run (`bag=rebuild` at
depths 2-6 and 14+, `bag=relabel` at 7-13).

### Symmetric: `ApplyProjectionsSymmetricDepthwiseAP` (oblique_cpu_symmetric_depthwise_ap.cc)

K projections **shared by every node of the depth** ⇒ the whole depth's rows (= the bag) can be
swept per projection with stride-1 column reads. **Changes model semantics** (shared
projections constrain splits) — compare accuracy/throughput, not bit-identity.

```cpp
absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<const int32_t> prev_first_child,   // prev idx → neg-child idx (pos = +1), -1 = leaf; empty = rebuild
    SymmetricBagState* bag_state,                 // cross-depth sorted (example,node) bag
    absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  // ── Phase 1: BuildBag (kSymBuildBag) — slab pre-size K*rows_n (reserve only).
  // ── Phase 2: obtain the sorted bag. NO per-depth sort in the steady state
  //    (2026-07-19): see oblique_cpu_depthwise_bag.{h,cc} — RelabelBagForNewDepth
  //    (billed to kSymSortBag) streams the previous (example, node) sequence once,
  //    O(bag) and comparison-sort-free, self-validating; the driver reconstructs
  //    prev_first_child from the previous batch's node pointers. Root level: span
  //    copied into the state (the rolling buffer is repartitioned in place).
  //    Fallback on validation failure: concat (kSymBuildBag) + hwy::K32V32 VQSort
  //    (kSymSortBag) — ties are same-node, so the unstable sort is safe. Verified
  //    bit-identical vs the sort path; relabel taken at every non-root depth.
  // ── Phase 3: Sweep (kSymSweep) — K bag-wide passes:
  std::vector<uint32_t> write_cursor(N);
  for (size_t k = 0; k < K; ++k) {
    std::fill(write_cursor.begin(), write_cursor.end(), 0u);
    const auto& proj = shared_projections[k];
    /* hoist col_ptrs[m], ws[m] out of the row loop */
    for (size_t i = 0; i < bag_size; ++i) {
      const UnsignedExampleIdx ex = bag[i];        // stride-1 ascending
      float value = 0.f;
      for (size_t m = 0; m < M; ++m) value += ws[m] * col_ptrs[m][ex];
      const uint32_t n = node_of_bag[i];
      const uint32_t pos = write_cursor[n]++;      // per-node cursor ⇒ N interleaved write streams
      out_projected[n][k * rows_n[n] + pos] = value;
    }
  }
}
```

Reads are perfect (sequential, lockstep column streams); **writes are the weakness** — N
per-node cursor streams RFO-thrash once N×64 B exceeds per-thread L2 share (≈30k nodes/depth).
Hence: ties BFS on HIGGS (386 vs 387 s), wins 23–36 % on wide trunks. Under the current
architecture (col-major reads + node-major slabs) symmetric **upper-bounds** shared-rows: if
symmetric can't beat control on a shape, no shared-rows fix will.

### Dead end: `SubtreeGatherCache` (removed 2026-07-16; was oblique.cc `#ifdef SUBTREE_GATHER_CACHE`, added in `9f32e817`)

Epoch-tagged per-thread block cache: when a node first dropped to ≤`RM_MAX_ROWS` rows its
example list became the current block; feature columns gathered lazily into dense block-local
arrays reused by descendants (DFS made it effective), budget `SG_BUDGET_MB`. **⛔ +43 % at
1.5M×4096**: with P=⌈√F⌉ and nnz≈1.5 a node and its descendants share only ~2 % of features,
so gathers never amortize. The worked example of Standing conclusion #2 ("adding memory
traffic to fix locality loses"). Recover from `9f32e817` (log:
`benchmarks/experiments/memory_safe_dynamic_2026-06-12.md`).
