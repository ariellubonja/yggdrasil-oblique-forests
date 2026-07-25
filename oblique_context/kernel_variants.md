# kernel_variants.md — the per-node driver (full body) + this fork's AP kernel variants

> Shard of `OBLIQUE_CONTEXT.md` (the lean core). The core keeps the stock
> setup + main-loop skeleton of `FindBestConditionSparseObliqueTemplate`; this
> shard has the **full** body including every variant `#ifdef` branch, plus the
> variant kernels themselves (DW1 col-sharing / shared-rows, symmetric depthwise
> AP, the subtree-gather dead end). Read this when working on a specific variant
> branch or the fused-slab hook. Snapshot as of 2026-07-04, branch `rebased-main`,
> commit `c80ffbf7`. Line numbers drift; grep for the symbol if a ref misses.

---

## The per-node driver: `FindBestConditionSparseObliqueTemplate` (oblique.cc:129) — full body

This is where all kernel variants hook in. Full body (regression templates identical):

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

  Projection best_projection;
  float best_threshold;
  Projection current_projection;
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

  // … [dynamic downgrade block — see split_search.md "Per-node DYNAMIC downgrade"] …

#if defined(SYMMETRIC_DEPTHWISE_AP) || defined(OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS)
  // Fused CPU Apply: GrowTreeLocalBFS pre-computed a per-node projected-values
  // slab. When the slab is present, skip SampleProjection +
  // ProjectionEvaluator::Evaluate and run split-finding directly over slices
  // of the slab (one slice per projection, length rows_n).
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
  } else
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

`SetCondition` (oblique.cc:1004) materializes the winning projection into the node proto
(`oblique_condition`: attributes[], weights[], na_replacements[] = column means, threshold).

Upstream of this: `FindBestConditionOblique` (oblique.cc:777) switches on `split_axis_case`
(sparse vs MHLD); `FindBestConditionSingleThreadManager` (training.cc:1441) calls it under
`kObliqueSplitSearch`, **then still runs the axis-aligned splitter loop over
`candidate_attributes`** — for pure-oblique runs `num_candidate_attributes` makes this a
no-op-ish pass, but it exists (`kGetCandidateAttributes` scope). RF never sets
`internal_config.split_finder_processor`, so `FindBestConditionManager` (training.cc:1896)
always takes the single-thread branch.

---

## Kernel variants (this fork's experiments)

Rule (`AGENTS.md` + ablation memory): **one idea = one `--config` = one branch; never
stack ideas in a measurement.** Controls stay pure. Variants present **on this branch**:

| Variant | Config | File / function | Status |
|---|---|---|---|
| Stock nodewise Evaluate | *(none)* | `oblique.cc` `ProjectionEvaluator::Evaluate` | baseline |
| BFS-only control | `--config=bfs_only` | `GrowTreeLocalBFS` fallback branch | scheduler ablation |
| DW1 depthwise 1-pass (col-sharing) | `--config=depthwise_1_pass` | `oblique_cpu_depthwise_1pass.cc` `ApplyProjectionsDepthwise1Pass` | ≤15 % slower than BFS; "col sharing via cache residency doesn't work at scale" |
| DW1 shared-rows | `--config=dw1_shared_rows` (implies dw1) | same file, `#ifdef DW1_SHARED_ROWS` | ⛔ 1.3–3.8× slower; postmortem in core §13 |
| DW1 hot-nodes hybrid | `--config=dw1_sr_hot_nodes` (implies dw1_shared_rows) | gate in `GrowTreeLocalBFS`; cold branch in `oblique.cc`; `AdvanceDepthBagHot` | fused kernel for big nodes only; bit-identical at every threshold; unmeasured (2026-07-25) |
| Symmetric depthwise AP | `--config=symmetric_depthwise_ap` | `oblique_cpu_symmetric_depthwise_ap.cc` | ✚ changes model semantics; wins wide-trunk, ties BFS on HIGGS |
| Subtree gather cache | *(code removed 2026-07-16)* | was `oblique.cc` `#ifdef SUBTREE_GATHER_CACHE`; recover via commit `9f32e817` | ⛔ +43 % (≈2 % feature overlap ⇒ gather never amortizes) |
| Row-major store | `--config=row_major_dataset_layout` + `--dataset_layout=row` | `RowMajorFeatureMatrix` via `AttributeValue` | layout experiment |

### DW1: `ApplyProjectionsDepthwise1Pass` (oblique_cpu_depthwise_1pass.cc)

> **2026-07-23: block machinery removed.** The per-block budget (`Dw1BlockFloats` /
> `YDF_DW1_BLOCK_FLOATS`), `Dw1Task`, the big-node projection-major path
> (`EvaluateNodeProjMajor`, kDw1SweepBig), and the shared-rows per-block arena
> distribution (`block_of_node`/`block_off`/`arena_ex`/`arena_node`, kDw1SharedBag)
> are gone — the budget had long been disabled, so every depth already ran as one
> block. The kernel is now a single depth-wide column-centric sweep; shared-rows
> reads the depth bag (`bag_state->bag` / `node_of_bag` from `AdvanceDepthBag`)
> directly instead of copying it into arenas. Verified bit-identical (nodes-* hash)
> vs the block version for both configs. Old code: commit `c6357624` and earlier.

Idea: at mid depths the frontier holds thousands of small nodes; total column references
N·K·nnz ≫ F. Invert the loop: bucket every (node, projection, weight) reference of the
whole depth by column, then walk touched columns once ascending.

```cpp
struct ColEntry { int32_t node; int32_t proj; int32_t col; float weight; };

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<const int32_t> prev_first_child, DepthBagState* bag_state,
    absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  const size_t N = selected_examples_per_node.size();
  if (N == 0) return absl::OkStatus();
  int max_attr = /* max over numerical_features */;
  // shared-rows only: total_rows = Σ selected_examples_per_node[n].size()

  // ── Phase 1: PreSize (kDw1PreSize, ~5% of AP) ────────────────────────
  // Slab pre-size only (zero-init: the column sweep accumulates).
  for (size_t n = 0; n < N; ++n)
    out_projected[n].assign(sel_n * projs_n, 0.f);

#ifdef DW1_SHARED_ROWS
  // Depth bag: example-sorted (bag, node_of_bag) for the whole frontier.
  // O(bag) relabel of the previous depth in the steady state (billed to
  // kDw1SharedRows via DepthBagChrono), concat + VQSort fallback otherwise.
  AdvanceDepthBag(selected_examples_per_node, prev_first_child, total_rows,
                  DepthBagChrono::kDw1SharedRows, bag_state);
#endif

  // ── Phase 2: Sweep (kDw1Sweep, dominant) ─────────────────────────────
  internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);
  bool col_major_dataset = /* all AttributeData(attr) != nullptr */;
  if (!col_major_dataset) { /* kDw1SweepGeneric: EvaluateProjectionRowsGeneric
                               per (node,proj); alternate layouts only */ }

  // Bucket every (node,proj,weight) reference by column: entries pushed
  // node-major, touched ascending, counting sort → `sorted`, col_count
  // becomes the per-column end cursor.  [≤2% of AP]

#ifdef DW1_SHARED_ROWS
  // Shared-rows colwalk over the whole depth. Per touched column c
  // (kDw1SweepColWalk, ~86% of AP):
  //   1. group c's refs by node → contiguous runs in ref_proj/ref_w,
  //      node_ref_off/cnt indexed by depth-batch node id (kDw1ColWalkGroupByNode, ~0%)
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
     (node,proj) entries; each entry re-gathers the column at ITS node's rows: */
  size_t pos = 0;
  for (const int32_t c : touched) {
    const float* col = evaluator.AttributeData(c);
    const size_t end = static_cast<size_t>(col_count[c]);
    for (; pos < end; ++pos) {
      const ColEntry& e = sorted[pos];
      const auto sel = selected_examples_per_node[e.node];
      float* o = out_projected[e.node].data() + e.proj * sel.size();
      const float w = e.weight;
      for (size_t i = 0; i < sel.size(); ++i)     // ★ the measured gather
        o[i] += w * col[sel.data()[i]];
    }
  }
#endif
  // Single-threaded by design: RandomForest already trains one tree per thread.
  return absl::OkStatus();
}
```

### DW1 hot-nodes hybrid (`--config=dw1_sr_hot_nodes`, added 2026-07-25)

**One idea:** run the shared-rows colwalk on the depth's **big nodes only**; every other
node of the same depth is evaluated by the stock per-node `ProjectionEvaluator::Evaluate`.
Nothing about the kernel itself changes — it just gets a smaller frontier.

Motivation (`nodewise_ap.csv`, HIGGS tree 0, depths 5/10/15/20): nodes above the pooled
mean `n_gathers` are **7.5 % of nodes but 91.8 % of AP time** (90.8 % of rows). The
shared-rows postmortem's failure (a) — scatter-write RFO amplification — scales with the
**number** of nodes at the depth (one interleaved slab write stream per node-with-a-ref per
column pass): at depth 20 that is 45.8 k streams ≈ 2.9 MB ≫ the per-thread cache share.
Gating to the hot set caps it at ~2–4 k streams (~120–230 KB) at every depth while keeping
most of the fused work. Failure (b) (the `off<0` bag skip-scan) is **not** addressed: on
HIGGS a column is referenced by ~27 % of nodes regardless of the gate, so with K≈2 k hot
nodes ~500 still share each column pass and per-column amortization survives; on wide
trunks sharing collapses to ~1 node/column with or without the gate (wide stays symmetric's
territory). Known ceiling: the recorded depth ladder covers only ~15 % of a tree's AP and
the hot row share falls with depth (70.6 % at d20), so whole-tree hot AP share is likely
~50–65 %, not 91.8 %.

**Gate.** Hot iff `sel.size() >= YDF_DW1_HOT_MIN_ROWS` (default 1000 ≈ the mean-`n_gathers`
cut, since `n_gathers ≈ 9·rows` on HIGGS). Rows rather than `n_gathers` because the gate
must be **monotone down the tree** (child rows ≤ parent rows ⇒ a hot node's parent is hot),
which is what lets the depth bag keep telescoping. `=0` ⇒ everything hot ⇒ exactly the
ungated `dw1_shared_rows` (purity control). Per-depth "above average" was rejected: at
depth 5 it keeps 1 of 16 nodes despite all 16 being huge.

**Sampling is untouched** — all nodes still sample their own projections at depth level in
node-major RNG order. The gate only picks *which kernel evaluates* a node, and both kernels
accumulate in ascending attribute order into an fp32 zero ⇒ **the model is bit-identical at
every threshold value**, which is the correctness check (see below).

Flow inside the `DEPTHWISE_1_PASS` branch of `GrowTreeLocalBFS` (all `#ifdef DW1_HOT_NODES`;
other builds unchanged):

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
  `AdvanceDepthBagHot`): the bag covers only hot rows and its labels are hot indices, but
  the parent→child hop runs in the **full** node domain (a hot parent's rows may land in a
  cold child, and the membership test consumes child spans in order). Only the driver holds
  the full spans + both index maps, so the kernel's internal `AdvanceDepthBag` call is
  `#ifndef DW1_HOT_NODES`. Relabel = the same streaming pass plus two index lookups
  (`prev_hot_to_full` in, `hot_of_node` out) and one extra drop condition (new owner cold —
  same mechanism as the existing leaf drop). Self-validation + concat/VQSort fallback (over
  the hot spans, whose positions *are* the hot labels) unchanged.
- **Why the cold branch in `oblique.cc` is mandatory:** a cold node gets defs+mono but no
  slab, so `has_precomputed_projected` is false; without the new `else if (defs && mono)`
  branch it would fall into the driver's main loop and **re-sample**, consuming the RNG
  twice. That branch just loops `Evaluate` + `EvaluateProjection` over the handed-down
  projections.
- **Chrono:** hot AP bills to `ProjectionEvaluate` at depth level as before (bag advance
  included); cold AP bills to the same cell from inside `NodeTrain` (`Evaluate` carries its
  own scope) — so at a mixed depth `TreeTrain = ΣNodeTrain + ΣAP + ΣSampleProjection` no
  longer partitions cleanly (cold AP is nested in `NodeTrain`). Not combinable with
  `--config=nodewise_chrono` (hot nodes bypass `Evaluate` ⇒ the dump selfcheck trips).
- Side effect: peak slab memory shrinks (slabs allocated for hot nodes only).

**Verified 2026-07-25** (macOS, correctness only — perf numbers must come from m7i/8488C):
trunk 20k×32 / 100k×29 / 8192×4096, `nodes-*` hash identical across
`YDF_DW1_HOT_MIN_ROWS ∈ {0, 1, 100, 250, 1000, 4000, 10000, 1e9}` **and** identical to the
ungated `--config=dw1_shared_rows` control; same under a DCHECK-enabled build
(`--cxxopt=-UNDEBUG`: bag sortedness, `node_of_bag` sizing, kernel bag/valid asserts). A
temporary trace on 20k×32 (T=1000) confirmed the intended shape: the incremental relabel
fires at every fused depth after the first (10/11 calls; the one fallback is the first fused
depth, which has no previous hot bag), the gate prunes hard (depth 8: 122 nodes → 2 hot;
depth 11: 302 → 2), and `hot=0` latches from depth 13 down.

### Symmetric: `ApplyProjectionsSymmetricDepthwiseAP` (oblique_cpu_symmetric_depthwise_ap.cc)

K projections **shared by every node of the depth** ⇒ the whole depth's rows (= the bag)
can be swept per projection with stride-1 column reads. **Changes model semantics**
(shared projections constrain splits) — compare accuracy/throughput, not bit-identity.

```cpp
absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<const int32_t> prev_first_child,   // prev-batch idx → neg-child idx (pos = +1), -1 = leaf; empty = rebuild
    SymmetricBagState* bag_state,                 // cross-depth sorted (example,node) bag; thread_local in the driver
    absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  // ── Phase 1: BuildBag (kSymBuildBag) — slab pre-size K*rows_n (reserve only).
  // ── Phase 2: obtain the sorted bag. NO per-depth sort in the steady state
  //    (2026-07-19): a depth's sorted bag = the previous depth's sorted bag
  //    minus leafed-out rows (stable partition of a sorted span stays sorted,
  //    so sorted row order telescopes to the pre-sorted bootstrap bag).
  //    RelabelBagForNewDepth (billed to kSymSortBag) streams the previous
  //    (example, node) sequence once: drop entries whose parent leafed, else
  //    advance the node label parent→child, child picked by ONE equality test
  //    against the neg child's next-unconsumed span element (a row id lives in
  //    exactly one child span; child spans are consumed strictly in order).
  //    O(bag), comparison-sort-free, self-validating; the driver
  //    (GrowTreeLocalBFS) reconstructs prev_first_child from the previous
  //    batch's node pointers (IsLeaf() after that depth's NodeTrains; children
  //    are pushed (neg,pos) per split parent in parent order). Root level:
  //    span copied into the state (the rolling buffer is repartitioned in
  //    place, so the span won't survive). Fallback on any validation failure:
  //    concat (kSymBuildBag) + hwy::K32V32 VQSort by example id (kSymSortBag)
  //    — ties are same-node so the unstable sort is safe.
  //    Verified bit-identical trees vs the sort path (trunk shapes, incl.
  //    YDF_SYMMETRIC_MAX_DEPTH handoff); relabel path taken at every non-root
  //    depth, zero fallbacks.
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
per-node cursor streams RFO-thrash once N×64 B exceeds per-thread L2 share (≈30k
nodes/depth). Hence: ties BFS on HIGGS (386 vs 387 s; deep 100k+-node depths thrash), wins
23–36 % on wide trunks. Under the current architecture (col-major reads + node-major
slabs) symmetric **upper-bounds** shared-rows: if symmetric can't beat control on a shape,
no shared-rows fix will.

### Dead end: `SubtreeGatherCache` (code removed 2026-07-16; was oblique_types.h + oblique.cc `#ifdef SUBTREE_GATHER_CACHE`, added in `9f32e817`)

Epoch-tagged per-thread block cache: when a node first dropped to ≤`YDF_RM_MAX_ROWS` rows its
example list became the current block; feature columns gathered lazily into dense
block-local arrays reused by descendants (DFS made it effective), budget
`YDF_SG_BUDGET_MB`. **⛔ +43 % at 1.5M×4096**: with P=⌈√F⌉ and nnz≈1.5, a node and its
descendants share only ~2 % of features — gathers never amortize. The worked example of
Standing conclusion #2 ("adding memory traffic to fix locality loses"). Code deleted from
the branch; recover from commit `9f32e817` (measurement log:
`benchmarks/experiments/memory_safe_dynamic_2026-06-12.md`).
