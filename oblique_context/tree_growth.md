# tree_growth.md — node lifecycle, DFS/BFS growers, forest driver

> Shard of `OBLIQUE_CONTEXT.md` (the lean core). Read this when a question is
> about `NodeTrain`, the in-place partition / rolling buffer, the DFS vs BFS
> drivers (`GrowTreeLocal` / `GrowTreeLocalBFS`), node evaluation order, the slab
> handoff fields, or the forest driver + bootstrap. The core keeps only the
> sortedness / RNG-order invariants distilled; the full bodies live here.
> Snapshot as of 2026-07-04, branch `rebased-main`, commit `c80ffbf7`. Line
> numbers drift; grep for the symbol if a ref misses.

---

## Node lifecycle and tree growth (order of node evaluation)

### `NodeTrain` (training.cc:5165) — trimmed to the active path

```cpp
ABSL_ATTRIBUTE_ALWAYS_INLINE static absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset, /* configs… */,
    const InternalTrainConfig& internal_config, utils::RandomEngine* random,
    PerThreadCache* cache, internal::NodeAndExamples node_and_examples,
    std::deque<internal::NodeAndExamples>& node_stack) {
  auto& selected_examples = node_and_examples.selected_examples;   // SelectedExamplesRollingBuffer
  const auto depth = node_and_examples.depth;
  auto node = node_and_examples.node;
#ifdef CHRONO_PROFILE
  tls_ctx.cur_depth = depth;                       // per-(tree,depth) accounting
  CHRONO_SCOPE_COARSE(…kNodeTrain);
  // … [truncated: node_cnt/sample_cnt per-depth tallies]
#endif

  if (selected_examples.empty()) return absl::InternalError("No examples fed to the node trainer");
  node->mutable_node()->set_num_pos_training_examples_without_weight(selected_examples.size());

  if (!set_leaf_already_set) {
    {  // kSetLeafValue
      RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
          train_dataset, selected_examples.active, weights, config, config_link, node));
    }
    // … [truncated: ApplyConstraintOnNode — monotonic only]
  }

  auto finalize_as_leaf = [&]() -> absl::Status { /* … */ };

  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth()) ||
      (internal_config.timeout.has_value() && internal_config.timeout < absl::Now())) {
    return finalize_as_leaf();
  }

  // … [truncated: RANDOM_LOCAL_IMPUTATION — not used; splitter sees the real dataset
  //    with splitter_dataset_is_compact=false]
  selected_examples_for_splitter = absl::MakeConstSpan(selected_examples.active);

  bool has_better_condition;
  {  // kFindBestCondition
    ASSIGN_OR_RETURN(has_better_condition,
        FindBestCondition(*train_dataset_for_splitter, selected_examples_for_splitter,
                          weights, config, config_link, dt_config, node->node(),
                          internal_config, constraints,
                          node->mutable_node()->mutable_condition(), random, cache));
  }
  if (!has_better_condition) return finalize_as_leaf();

  STATUS_CHECK_EQ(selected_examples.size(),
                  node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(…);

  // Separate the positive and negative examples.
  CHRONO_BEGIN(split_examples_in_place);
  ASSIGN_OR_RETURN(auto example_split,
      internal::SplitExamplesInPlace(*train_dataset_for_splitter, selected_examples,
          node->node().condition(), splitter_dataset_is_compact,
          dt_config.internal_error_on_wrong_splitter_statistics()));
  CHRONO_END(split_examples_in_place, …kSplitExamplesInPlace);

  if (example_split.positive_examples.empty() || example_split.negative_examples.empty()) {
    node->ClearChildren();
    return finalize_as_leaf();
  }

  // … [truncated: honest-trees leaf_examples split; kSetLeafValue for both children;
  //    monotonic-constraint division]

  // Negative child.
  node_stack.push_back({node->mutable_neg_child(),
                        std::move(example_split.negative_examples), /*…*/, depth + 1, …});
  // Positive child.
  node_stack.push_back({node->mutable_pos_child(),
                        std::move(example_split.positive_examples), /*…*/, depth + 1, …});
  return absl::OkStatus();
}
```

### In-place partition and the rolling buffer

`model/decision_tree/decision_tree.h:79`:

```cpp
// A list of selected examples, and a related buffer used to do some computation.
struct SelectedExamplesRollingBuffer {
  absl::Span<UnsignedExampleIdx> active;
  absl::Span<UnsignedExampleIdx> inactive;   // scratch of equal size
  size_t size() const { return active.size(); }
  static SelectedExamplesRollingBuffer Create(absl::Span<UnsignedExampleIdx> active,
                                              std::vector<UnsignedExampleIdx>* buffer);
};
struct ExampleSplitRollingBuffer {
  SelectedExamplesRollingBuffer positive_examples;
  SelectedExamplesRollingBuffer negative_examples;
};
```

`SplitExamplesInPlace` (training.cc:5891) evaluates the just-learned condition on the
node's rows (`EvalConditionOnDataset` — for oblique conditions this **re-computes the
projection dot-products** for the winning projection) and partitions `active` in place;
children's `active` spans are contiguous sub-spans of the parent's. Crucial invariant:

```cpp
absl::StatusOr<ExampleSplitRollingBuffer> SplitExamplesInPlace(
    const dataset::VerticalDataset& dataset, const SelectedExamplesRollingBuffer examples,
    const proto::NodeCondition& condition, const bool dataset_is_dense,
    const bool error_on_wrong_splitter_statistics,
    const bool examples_are_training_examples) {
  DCHECK(std::is_sorted(examples.active.begin(), examples.active.end()));
  ExampleSplitRollingBuffer example_split;
  RETURN_IF_ERROR(EvalConditionOnDataset(dataset, examples, condition,
                                         dataset_is_dense, &example_split));
  DCHECK(std::is_sorted(example_split.positive_examples.active.begin(),
                        example_split.positive_examples.active.end()));
  DCHECK(std::is_sorted(example_split.negative_examples.active.begin(),
                        example_split.negative_examples.active.end()));
  // … [truncated: splitter-vs-evaluation count consistency check → warning/error]
  return example_split;
}
```

⇒ **Every node's `selected_examples` is sorted ascending, at every depth** (bootstrap is
sorted; partition of a sorted list is sorted). Duplicates (bootstrap) stay adjacent.
The DCHECKs are compiled out in `-c opt` — kernels that rely on sortedness re-assert it.

### DFS driver (default): `GrowTreeLocal` (training.cc:5384)

```cpp
absl::Status GrowTreeLocal(/* … */, NodeWithChildren* root, utils::RandomEngine* random,
                           PerThreadCache* cache,
                           SelectedExamplesRollingBuffer selected_examples,
                           std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  std::deque<internal::NodeAndExamples> node_stack;
  node_stack.push_back({root, std::move(selected_examples), std::move(leaf_examples),
                        depth, constraints, set_leaf_already_set});
  while (!node_stack.empty()) {
    auto current_node = std::move(node_stack.back());
    node_stack.pop_back();                                     // LIFO
    RETURN_IF_ERROR(NodeTrain(…, std::move(current_node), node_stack));
  }
  return absl::OkStatus();
}
```

Order semantics: NodeTrain pushes **neg then pos**, the stack pops **pos first** ⇒
depth-first, positive-child-first traversal. A node's whole positive subtree completes
before its negative sibling starts. RNG is consumed in this DFS order.

### BFS driver + variant hooks: `GrowTreeLocalBFS` (training.cc:5453)

Selected by `DecisionTreeCoreTrain` (training.cc:5138) at **compile time**:

```cpp
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal: {
#if defined(DEPTHWISE_1_PASS) || defined(SYMMETRIC_DEPTHWISE_AP) || \
    defined(BFS_ONLY)
      return GrowTreeLocalBFS(…);
#else
      return GrowTreeLocal(…);
#endif
    }
```

```cpp
// BFS (level-order) variant of GrowTreeLocal. Uses a FIFO deque, collects all
// nodes at the current depth into a `depth_batch`, then dispatches each node
// through NodeTrain. The depth-batch collection is the seam where fused
// per-level Apply hooks in.
absl::Status GrowTreeLocalBFS(/* same signature */) {
  std::deque<internal::NodeAndExamples> node_queue;
  node_queue.push_back({root, …});

  while (!node_queue.empty()) {
    const int32_t current_depth = node_queue.front().depth;
    std::vector<internal::NodeAndExamples> depth_batch;
    while (!node_queue.empty() && node_queue.front().depth == current_depth) {
      depth_batch.push_back(std::move(node_queue.front()));
      node_queue.pop_front();
    }

#if defined(DEPTHWISE_1_PASS)
    if (dt_config.has_sparse_oblique_split() && depth_batch.size() > 1 &&
        current_depth >= Depthwise1PassMinDepth()) {          // env DW1_MIN_DEPTH, default 0
      // Fused-per-level CPU Apply. Sample projections per node (ordinary SPO-RF
      // semantics), then precompute each node's projected-value slab.
      const int num_proj = GetNumProjections(dt_config, config_link.numerical_features_size());
      const float projection_density = /* 1.5/F clamp */;
      const int num_nodes = depth_batch.size();
      std::vector<std::vector<internal::Projection>> all_node_projs(
          num_nodes, std::vector<internal::Projection>(num_proj));
      std::vector<std::vector<int8_t>> all_node_mono(num_nodes, std::vector<int8_t>(num_proj));
      // tls_ctx.cur_depth pinned to current_depth here (BFS depth-level work runs
      // BEFORE this depth's NodeTrains — without the pin it books to the previous depth).
      {  // kSampleProjection: 2^d nodes × K projections, node-major, same RNG engine
        for (int n = 0; n < num_nodes; ++n)
          for (int p = 0; p < num_proj; ++p)
            internal::SampleProjection(…, &all_node_projs[n][p], &all_node_mono[n][p], random);
      }
      std::vector<absl::Span<const UnsignedExampleIdx>> sel_spans(num_nodes);
      for (int n = 0; n < num_nodes; ++n) sel_spans[n] = depth_batch[n].selected_examples.active;
      std::vector<std::vector<float>> projected(num_nodes);
      // … [truncated: LINECOUNT_A distinct-cache-line tally; CALLGRIND_DEPTH
      //    per-depth instrumentation brackets]
      RETURN_IF_ERROR(ApplyProjectionsDepthwise1Pass(
          train_dataset, config_link.numerical_features(),
          absl::MakeConstSpan(sel_spans), absl::MakeConstSpan(all_node_projs),
          absl::MakeSpan(projected)));
      for (int n = 0; n < num_nodes; ++n) {
        auto node_config = internal_config;
        node_config.depthwise_projection_defs = &all_node_projs[n];
        node_config.depthwise_monotonic = &all_node_mono[n];
        node_config.precomputed_projected_values = absl::MakeConstSpan(projected[n]);
        RETURN_IF_ERROR(NodeTrain(…, node_config, random, cache,
                                  std::move(depth_batch[n]), node_queue));
        std::vector<float>().swap(projected[n]);
      }
    } else
#elif defined(SYMMETRIC_DEPTHWISE_AP)
    if (dt_config.has_sparse_oblique_split() && depth_batch.size() >= 1 &&
        current_depth <= SymmetricMaxDepth()) {               // env SYMMETRIC_MAX_DEPTH
      // CatBoost-style symmetric trees: sample K projections ONCE for this
      // depth, shared across all nodes. The aggregate of nodes' selected
      // examples at depth d == the bag, so the projection sweep becomes
      // stride-1 in column space (vs. per-node scattered gather).
      std::vector<internal::Projection> shared_projections(num_proj);
      std::vector<int8_t> shared_monotonic(num_proj, 0);
      for (int p = 0; p < num_proj; ++p) internal::SampleProjection(…);
      // build sel_spans; projected(num_nodes);
      RETURN_IF_ERROR(ApplyProjectionsSymmetricDepthwiseAP(
          train_dataset, config_link.numerical_features(),
          absl::MakeConstSpan(sel_spans), absl::MakeConstSpan(shared_projections),
          absl::MakeSpan(projected)));
      // per node: node_config.{depthwise_projection_defs,depthwise_monotonic,
      //                        precomputed_projected_values} → NodeTrain
    } else if (dt_config.has_sparse_oblique_split()) {
      // Symmetric → DFS handoff (deeper than SYMMETRIC_MAX_DEPTH): finish each
      // frontier node's subtree with GrowTreeLocal (pushes nothing back to node_queue).
      for (auto& nae : depth_batch) RETURN_IF_ERROR(GrowTreeLocal(…, std::move(nae.…)));
    } else
#endif
    {
      // Per-node fallback: BFS scheduling without fused Apply / shared sampling.
      // This is what --config=bfs_only measures (scheduler-only ablation vs DFS).
#ifdef BFS_ONLY
      CHRONO_SCOPE_COARSE(…kBfsNodeLoop);
#endif
      for (auto& nae : depth_batch) {
        RETURN_IF_ERROR(NodeTrain(…, std::move(nae), node_queue));
      }
    }
  }
  return absl::OkStatus();
}
```

The slab handoff fields (`training.h:309`, `struct InternalTrainConfig`):

```cpp
  const std::vector<std::vector<internal::AttributeAndWeight>>*
      depthwise_projection_defs = nullptr;      // K projection definitions for this node
  const std::vector<int8_t>* depthwise_monotonic = nullptr;
  // Pre-computed projected values produced by a fused-per-level Apply.
  // Layout: slab[p * rows_n + i] = projection p applied to this node's i-th
  // selected example. When non-empty AND depthwise_projection_defs != null,
  // oblique.cc skips SampleProjection + Evaluate and reads this buffer.
  absl::Span<const float> precomputed_projected_values;
```

**BFS vs DFS reproducibility:** the BFS drivers consume RNG in level order, DFS in
depth-first order ⇒ trees at the same seed legitimately differ between schedulers
(accuracy-equivalent). Bit-identity comparisons are only valid within one scheduler.
`--config=bfs_only` isolates scheduling cost; measured: **BFS hurts tall-narrow data**
(HIGGS BFS col 336 s vs DFS col 282 s).

---

## Forest driver: `RandomForestLearner::TrainWithStatusImpl` (random_forest.cc:416)

```cpp
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        deployment().num_threads(), {.name_prefix = std::string("TrainRF")});
    for (int tree_idx = 0; tree_idx < rf_config.num_trees(); tree_idx++) {
      pool.Schedule([&, tree_idx]() {
        // … [truncated: stop triggers, max-duration/size checks]
        std::vector<UnsignedExampleIdx> selected_examples;
        auto& decision_tree = (*mdl->mutable_decision_trees())[tree_idx];
        utils::RandomEngine random(tree_seeds[tree_idx]);     // per-tree engine, seeds pre-generated
        auto status_sampling = internal::SampleTrainingExamples(
            train_dataset.nrow(), rf_config, bootstrap_size_ratio_factor,
            &random, &selected_examples);
        // … [truncated: status plumbing]
        decision_tree::InternalTrainConfig internal_config;
        internal_config.preprocessing = &preprocessing;
        internal_config.timeout = timeout;
        // (split_finder_processor is NOT set for RF ⇒ single-thread split search per node)
        auto status_train = decision_tree::Train(
            train_dataset, selected_examples, config_with_default, config_link,
            rf_config.decision_tree(), deployment(), weights, &random,
            decision_tree.get(), internal_config);
        // … [truncated: model-size accounting, OOB, logging]
      });
    }
  }
```

Bootstrap (random_forest.cc:1586) — **sorted, with duplicates** (`bootstrap_size_ratio=1.0`,
`sampling_with_replacement=true` defaults ⇒ bag size = nrow, ~63.2 % distinct rows):

```cpp
absl::Status SampleTrainingExamples(
    const UnsignedExampleIdx num_examples, const proto::RandomForestTrainingConfig& rf_config,
    std::optional<double> bootstrap_size_ratio_factor,
    utils::RandomEngine* random, std::vector<UnsignedExampleIdx>* selected) {
  if (!rf_config.bootstrap_training_dataset()) {
    selected->resize(num_examples);
    std::iota(selected->begin(), selected->end(), 0);
    return absl::OkStatus();
  }
  // … [truncated: adaptive ratio plumbing]
  const auto num_samples = std::max(int64_t{1},
      static_cast<int64_t>(static_cast<double>(num_examples) * bootstrap_size_ratio));
  selected->clear();
  selected->reserve(num_samples);
  if (rf_config.sampling_with_replacement()) {
    std::uniform_int_distribution<UnsignedExampleIdx> example_idx_distrib(0, num_examples - 1);
    for (UnsignedExampleIdx sample_idx = 0; sample_idx < num_samples; sample_idx++)
      selected->push_back(example_idx_distrib(*random));
    std::sort(selected->begin(), selected->end());
  }
  // … [truncated: without-replacement branch]
}
```

`decision_tree::Train` (= `DecisionTreeTrain`, training.cc:4950; alias training.h:1075)
copies the bag into `working_selected_examples` (honest-trees split skipped), then `DecisionTreeCoreTrain`
wraps it in a `SelectedExamplesRollingBuffer` with an equal-size scratch buffer.
