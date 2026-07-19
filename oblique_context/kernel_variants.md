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
#ifdef SYMMETRIC_NODEWISE_CONTROL
  // Control: same K shared projections per depth, but each node still runs the
  // node-local Evaluate. Isolates shared-sampling from the bagwide-read effect.
  const bool has_depthwise_shared_projections =
      internal_config.depthwise_projection_defs != nullptr &&
      internal_config.depthwise_monotonic != nullptr;
  if (has_depthwise_shared_projections) {
    // … [same loop as above but with projection_evaluator.Evaluate(
    //     depth_projs[proj_idx], selected_examples, &projection_values) feeding
    //     EvaluateProjection] …
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
| Symmetric depthwise AP | `--config=symmetric_depthwise_ap` | `oblique_cpu_symmetric_depthwise_ap.cc` | ✚ changes model semantics; wins wide-trunk, ties BFS on HIGGS |
| Symmetric nodewise control | `--config=symmetric_nodewise_control` | shared sampling, node-local Evaluate | control |
| Subtree gather cache | *(code removed 2026-07-16)* | was `oblique.cc` `#ifdef SUBTREE_GATHER_CACHE`; recover via commit `9f32e817` | ⛔ +43 % (≈2 % feature overlap ⇒ gather never amortizes) |
| Row-major store | `--config=row_major_dataset_layout` + `--dataset_layout=row` | `RowMajorFeatureMatrix` via `AttributeValue` | layout experiment |

### DW1: `ApplyProjectionsDepthwise1Pass` (oblique_cpu_depthwise_1pass.cc)

Idea: at mid depths the frontier holds thousands of small nodes; total column references
N·K·nnz ≫ F. Invert the loop: bucket every (node, projection, weight) reference by column,
walk touched columns once ascending; block nodes so output slabs stay cache-resident.

```cpp
// Output-slab budget (floats) per column-centric block. Also the cutoff
// above which a node is processed projection-major.
size_t Dw1BlockFloats() {   // env YDF_DW1_BLOCK_FLOATS, default 64 Mi floats (256 MiB)
  // Measured at 3M×4096: 4 MiB 55.0 s, 16 MiB 49.6, 64 MiB 47.0, 256 MiB 45.6.
  …
}

struct ColEntry { int32_t node; int32_t proj; int32_t col; float weight; };

#ifdef DW1_SHARED_ROWS
// 8-byte bag row for hwy::VQSort: key = example id (uint32), value = owning node.
// The row's local slab slot is NOT stored: selected_examples is sorted ascending,
// so within the row-sorted bag a node's rows appear in slab order and `local` is
// recovered as a per-node running counter. Requires 32-bit example ids.
using BagRow = hwy::K32V32;
#endif

struct Dw1Task { size_t begin_node; size_t end_node; bool big; };
struct Dw1Scratch { std::vector<ColEntry> entries, sorted; std::vector<int32_t> touched, col_count;
                    /* + shared-rows: bag, node_ref_off/cnt, node_local, ref_proj, ref_w, col_touched */ };

// Projection-major kernel for one node (big nodes / fallback). Direct column
// pointers; the per-load AttributeValue branch chain is hoisted out.
inline void EvaluateNodeProjMajor(const internal::ProjectionEvaluator& evaluator,
    const std::vector<internal::Projection>& projs,
    const UnsignedExampleIdx* sel_ptr, size_t rows_n, float* out_ptr) {
  struct FeatRef { const float* col; float weight; /* +na if isnan build */ };
  std::vector<FeatRef> feats;
  for (size_t p = 0; p < projs.size(); ++p) {
    feats.clear();
    for (const auto& feat : projs[p])
      feats.push_back({evaluator.AttributeData(feat.attribute_idx), feat.weight});
    float* o = out_ptr + p * rows_n;
    for (size_t i = 0; i < rows_n; ++i) {
      const UnsignedExampleIdx ex = sel_ptr[i];
      float acc = 0.f;
      for (const auto& f : feats) acc += f.weight * f.col[ex];
      o[i] = acc;
    }
  }
}

absl::Status ApplyProjectionsDepthwise1Pass(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const std::vector<internal::Projection>> projections_per_node,
    absl::Span<std::vector<float>> out_projected) {
  const size_t N = selected_examples_per_node.size();
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  if (N == 0) return absl::OkStatus();
  int max_attr = /* max over numerical_features */;

  // ── Phase 1: PreSize (kDw1PreSize, ~5% of AP) ────────────────────────
  // Slab pre-size (zero-init: the column sweep accumulates) + task build:
  // pack consecutive nodes into blocks of ≤ Dw1BlockFloats() slab floats;
  // nodes whose slab alone exceeds the budget become big=true tasks.
  std::vector<Dw1Task> tasks;
  { for (size_t n = 0; n < N; ++n) {
      const size_t slab = selected_examples_per_node[n].size() * projections_per_node[n].size();
      out_projected[n].assign(slab, 0.f);
      /* … block packing … */ } }

  // ── Phase 2: Sweep (kDw1Sweep, dominant) ─────────────────────────────
  {
    internal::ProjectionEvaluator evaluator(train_dataset, numerical_features);
    bool direct = /* all AttributeData(attr) != nullptr */;

    const auto run_task = [&](const Dw1Task& task, Dw1Scratch& scratch) {
      if (!direct) { /* kDw1SweepGeneric: EvaluateProjectionRowsGeneric per (node,proj) */ return; }
      if (task.big) {  // kDw1SweepBig
        EvaluateNodeProjMajor(evaluator, projections_per_node[n], sel.data(),
                              sel.size(), out_projected[n].data());
        return;
      }

      // Column-centric block: bucket references by column (counting sort into
      // `sorted`, touched ascending)…   [≤2% of AP]
      for (size_t n = task.begin_node; n < task.end_node; ++n)
        for (size_t p = 0; p < projs.size(); ++p)
          for (const auto& feat : projs[p]) {
            entries.push_back({n, p, feat.attribute_idx, feat.weight});
            if (col_count[feat.attribute_idx]++ == 0) touched.push_back(feat.attribute_idx);
          }
      std::sort(touched.begin(), touched.end());
      /* counting sort entries → sorted, col_count becomes end-cursor per column */

#ifdef DW1_SHARED_ROWS   // ~10% of Shared-rows AP: bag build+sort
      // Shared-rows colwalk. Build the block's merged bag once (union of its
      // nodes' selected_examples, sorted by example id via hwy::VQSort on K32V32),
      // then read each touched column in ONE ascending pass over that bag and
      // SCATTER each value to every (node,projection) referencing the column.
      // Trades per-node sparse gather reads for sparse slab writes.
      // (Rows are disjoint across nodes: the depth frontier partitions the bag.)
      for each touched column c:                    // kDw1SweepColWalk, ~86% of AP
        group c's refs by node (contiguous runs in ref_proj/ref_w)   // kDw1ColWalkGroupByNode, ~0%
        for (const BagRow& be : bag) {              // kDw1ColWalkBagScatter — the hot loop
          const int32_t bn = BagRowNode(be) - begin_node_i;
          const int32_t off = node_ref_off[bn];
          if (off < 0) continue;   // owning node has no projection on column c (74–98% skipped!)
          const float v = col[BagRowExample(be)];
          const int32_t local = node_local[bn]++;   // running counter == slab slot (sorted bag)
          float* slab = out_projected[node].data();
          const size_t rows_n = selected_examples_per_node[node].size();
          for (int32_t t = 0; t < cnt; ++t)
            slab[ref_proj[off + t] * rows_n + local] += ref_w[off + t] * v;   // scatter-write
        }
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
          const size_t rows_n = sel.size();
          const UnsignedExampleIdx* sel_ptr = sel.data();
          float* o = out_projected[e.node].data() + e.proj * rows_n;
          const float w = e.weight;
          for (size_t i = 0; i < rows_n; ++i) {     // ★ the measured gather (line ~477)
            float v = col[sel_ptr[i]];
            o[i] += w * v;
          }
        }
        col_count[c] = 0;
      }
#endif
    };

    // Single-threaded by design: RandomForest already trains one tree per thread.
    Dw1Scratch scratch;
    for (const auto& task : tasks) run_task(task, scratch);
  }
  return absl::OkStatus();
}
```

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
