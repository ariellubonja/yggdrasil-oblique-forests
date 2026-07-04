# OBLIQUE_CONTEXT.md — the code map for ApplyProjection optimization (lean core)

> **Purpose.** Always-loaded core context for every AI/agent session working on this fork.
> The project is: **make `ProjectionEvaluator::Evaluate()` (a.k.a. ApplyProjection) in
> `oblique.cc` faster**, within research-grade constraints. This core keeps only what a
> session needs on *most* questions — scope, the call map, the hot function, the driver
> skeleton, invariants, measured facts. Deep sections (full split-finder bodies, tree
> growers, kernel variants, build/measure detail) are **sharded** into sibling files under
> `oblique_context/` and read on demand via the router below — this keeps the auto-loaded
> footprint small (attention, not money).
>
> **Division of labor between the standing docs:**
> - `AGENTS.md` — the *workflow* (experiment loop, measurement rules, logging contract).
> - **This file + its `oblique_context/` shards** — the *code*: call stack, functions,
>   variants, configs, invariants.
> - `E2E + Chrono coarse. Col-major - m7i.metal-24x Bootstrapping.csv` — end-to-end timing.
>
> Code snapshots and `file:line` refs are as of 2026-07-04, branch `rebased-main`,
> commit `c80ffbf7`. Line numbers drift; grep for the symbol if a ref misses.

### Shard router — read the shard when the question lands in its area

| Question is about… | Read |
|---|---|
| The AP kernel, per-node setup, access pattern, the driver's stock main loop | *(this core — §4, §7)* |
| The split finders: dispatch, histogram (default), EXACT/VQSort scan, bins, SIMD upper_bound, dynamic downgrade | `oblique_context/split_search.md` |
| Node lifecycle, DFS/BFS growers, node-eval order, slab handoff, forest driver, bootstrap | `oblique_context/tree_growth.md` |
| A specific kernel variant (DW1 col-sharing / shared-rows, symmetric, subtree-gather) or the driver's full variant-`#ifdef` body | `oblique_context/kernel_variants.md` |
| `VerticalDataset` layout source, row-major store, full `SampleProjection` / weight modes | `oblique_context/dataset_and_sampling.md` |
| `.bazelrc` configs, env knobs, harness defaults, trunk generator, measurement tooling | `oblique_context/build_measure.md` |

(Reference shards as plain paths only — do **not** `@`-import them, or they auto-load and defeat the point.)

---

## 1. Scope and simplifying constraints

Everything in this project is specialized to:

- **Fully numeric input features** (`dataset::proto::NUMERICAL`, fp32). No categorical /
  boolean / set / discretized features on the input side.
- **Binary classification.** Label is CATEGORICAL with `number_of_unique_values() == 3`
  (index 0 = reserved out-of-vocabulary, 1 and 2 = the two classes).
- **Unweighted.** `GetWeights(..., use_optimized_unit_weights=true)` returns an **empty**
  `weights` vector when all weights are 1, so all hot paths take the `weights.empty()` /
  `weighted=false` branches.
- **No missing values in practice.** The per-lookup `std::isnan` in the projection kernels
  is **compiled out by default** and only re-enabled by `--config=enable_applyprojection_isnan`.
  Synthetic data has no NaN; row-major mirrors replace NaN with the column mean at copy time.
- **Random Forest (Bagging), `growing_strategy=Local`, sparse-oblique splits.**
  Not GBT, not MHLD-oblique, not best-first-global, no monotonic constraints, no honest
  trees, no uplift. Ignore those branches everywhere (they are truncated below).
- **Split scoring:** entropy / information gain via `LabelBinaryCategoricalScoreAccumulator`.
- **Exact arithmetic is mandatory.**  Sub-fp32 storage (bf16/fp16/int8) is **ruled out** (user directive, 2026-07-01).

Default benchmark configuration (what `runtime.sh` runs): **Oblique + DYNAMIC_RANDOM_HISTOGRAM,
64 bins, dynamic_split_threshold=250** — so the split-finder is the histogram path for nodes
≥250 rows and the EXACT (VQSort) path below that. Per-function timing work also uses EXACT runs.

### Workload shape (why AP dominates)

- Per **node** with n rows and F numerical features: P projections are sampled and evaluated,
  where `P = min(max(⌈F^0.5⌉, min), 1000)` (harness defaults: `num_projections_exponent=.5`,
  `max_num_projections=1000`), each with `nnz ≈ Binomial(F, 1.5/F) ≈ 1.5` features.
  HIGGS (F=29): P=6. trunk 4096: P=64. trunk 400k: P=633.
- ApplyProjection = for each projection, gather `nnz` feature columns at the node's n
  (sorted, possibly duplicated) row ids and accumulate `Σ w_j·col_j[row]` → one fp32 value
  per row. Then the split search runs over those n projected values.
- Total AP work per tree ≈ Σ_nodes n·P·nnz gathers. The gather `col[sel[i]]` is the
  memory-bound core: sequential at the root, increasingly sparse with depth (§13).
- Trees are trained **one per thread** (RF thread pool). All per-node kernels are and must
  stay single-threaded — parallelism budget is spent at the tree level.

---

## 2. Call stack (files + entry points)

```text
RandomForestLearner::TrainWithStatusImpl        learner/random_forest/random_forest.cc:416
└─ ThreadPool "TrainRF": one tree per task                        random_forest.cc:679
   ├─ internal::SampleTrainingExamples  (bootstrap; SORTED, dups) random_forest.cc:1586
   └─ decision_tree::Train = DecisionTreeTrain (copies bag)       learner/decision_tree/training.cc:4950 (alias training.h:1075)
      └─ DecisionTreeCoreTrain                                    training.cc:5117
         ├─ GrowTreeLocal          DFS (default build)            training.cc:5384
         ├─ GrowTreeLocalBFS       BFS (dw1 / symmetric / bfs_only builds)  training.cc:5453
         └─ GrowTreeBestFirstGlobal  (not used in this project)   training.cc:4791
            │
            ▼ per node
            NodeTrain                                             training.cc:5165
            ├─ set_leaf_value_functor (label distribution)
            ├─ FindBestCondition                                  training.cc:1921
            │  └─ FindBestConditionManager (single-thread for RF) training.cc:1896
            │     └─ FindBestConditionSingleThreadManager         training.cc:1441
            │        └─ FindBestConditionOblique                  oblique.cc:777
            │           └─ FindBestConditionSparseObliqueTemplate oblique.cc:129
            │              ├─ ProjectionEvaluator ctor (PER NODE) oblique.cc:1367
            │              ├─ ExtractLabels / Extract  (label gather)
            │              └─ loop p = 0..P-1:
            │                 ├─ internal::SampleProjection       oblique.cc:867
            │                 ├─ ProjectionEvaluator::Evaluate    oblique.cc:1410   ← ★ THE HOT FUNCTION (ApplyProjection)
            │                 └─ EvaluateProjection               oblique.cc:407
            │                    ├─ FindSplitLabelClassificationFeatureNumericalHistogram  training.cc:2229
            │                    └─ FindSplitLabelClassificationFeatureNumericalCart       training.cc:2458 (EXACT)
            │                       └─ FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical
            │                          = FindBestSplitFlatHighway  splitter_scanner.h:1746
            │                             ├─ pack (feature,label) → hwy::K32V32, VQSort
            │                             └─ ScanSplitsFlat        splitter_scanner.h:1633
            ├─ internal::SplitExamplesInPlace  (in-place partition → child subspans)  training.cc:5891
            └─ push children (DFS: neg pushed, then pos → pos POPPED FIRST)
```

Chrono-scope names (what `parallel_chrono.py` CSVs show), coarse tier (`CHRONO_PROFILE=1`):
`TreeTrain, NodeTrain, SampleProjection, EvaluateProj, ProjectionEvaluate (=ApplyProjection),
FindObliqueSetup, ObliqueSplitSearch, FindBestCondition, GetCandidateAttributes, BfsNodeLoop`.
Fine tier (`=2`) adds: `SetLeafValue, SplitExamplesInPlace, HistoPath (HistogramSetup,
MinMaxNumerical, AssignSamplesToHistogram, SelectBestThresholdHistogram, EntropyTableSetup),
CartPath (CartSetup, SortInitBuckets, SortFillBuckets, SortFeatures, SortScanSplits),
Dw1PreSize, Dw1Sweep, Dw1SweepBig, Dw1SweepGeneric, Dw1SweepColWalk, Dw1SharedBag,
Dw1ColWalkGroupByNode, Dw1ColWalkBagScatter, SymBuildBag, SymSortBag, SymSweep`.
Invariant used by the tooling: **TreeTrain = ΣNodeTrain + ΣApplyProjection + ΣSampleProjection**
(the BFS drivers pin `tls_ctx.cur_depth` before depth-level work so per-depth cells line up).

---

## 3. Dataset: column-major `VerticalDataset` — (full source → `dataset_and_sampling.md`)

`dataset/vertical_dataset.h`. One heap `std::vector<float>` per column; no interleaving. Row
index `UnsignedExampleIdx` = **uint32** by default (`dataset/types.h`;
`--define=ydf_example_idx_num_bits={32,64}`). The hot path **never** uses the virtual column
interface — `ProjectionEvaluator` caches a raw `const float*` per attribute once per node
(§4). The label is a `CategoricalColumn` (`TemplateScalarStorage<int32_t>`), values ∈ {1,2}.

**Alternate store `RowMajorFeatureMatrix`** (`dataset/row_major_feature_matrix.h`, 62 lines):
process-global optional fp32 **row-major** mirror, filled by the harness when
`--dataset_layout=row` + compiled in by `--config=row_major_dataset_layout`;
`ProjectionEvaluator::AttributeValue` routes through it (same loop, different layout). NaN →
column mean at fill time.

---

## 4. ★ The hot function: `ProjectionEvaluator` + `Evaluate` (ApplyProjection)

`learner/decision_tree/oblique.h` + `oblique.cc`. The projection type
(`oblique_types.h`):

```cpp
// A projection is defined as \sum features[projection[i].index] * projection[i].weight;
struct AttributeAndWeight {
  int attribute_idx;
  float weight;
};
typedef std::vector<AttributeAndWeight> Projection;
```

Class (oblique.h; trimmed to the members that matter):

```cpp
// Default: `ProjectionEvaluator::Evaluate` is NOT inlined, so profilers attribute
// time/FLOPs to the function itself. Cost within noise (~1%). Opt back in with
// --config=inline_projection_evaluate.
#ifdef YDF_INLINE_PROJECTION_EVALUATE
#define YDF_PROJECTION_EVALUATE_NOINLINE
#else
#define YDF_PROJECTION_EVALUATE_NOINLINE __attribute__((noinline))
#endif

class ProjectionEvaluator {
 public:
  ProjectionEvaluator(const dataset::VerticalDataset& train_dataset,
                      const google::protobuf::RepeatedField<int32_t>& numerical_features);

  YDF_PROJECTION_EVALUATE_NOINLINE
  absl::Status Evaluate(const Projection& projection,
                        absl::Span<const UnsignedExampleIdx> selected_examples,
                        std::vector<float>* values) const;

  const float* AttributeData(int attribute_idx) const {
    return numerical_attribute_data_[attribute_idx];
  }

  float AttributeValue(int attribute_idx, UnsignedExampleIdx example_idx) const {
    // The optional row-major store (ROW_MAJOR_DATASET_LAYOUT) shadows the
    // default per-column vertical store; both feed the same projection loop.
    if (row_major_matrix_ != nullptr) {
      return row_major_matrix_->Get(example_idx, attribute_idx);
    }
    return numerical_attribute_data_[attribute_idx][example_idx];
  }

  float NaReplacementValue(int attribute_idx) const { return na_replacement_value_[attribute_idx]; }

 private:
  // Non-owning pointer to numerical attributes. Indexed by attribute idx.
  std::vector<const std::vector<float>*> numerical_attributes_;
  std::vector<const float*> numerical_attribute_data_;
  const dataset::RowMajorFeatureMatrix* row_major_matrix_ = nullptr;
  std::vector<float> na_replacement_value_;   // per attribute: column mean
  absl::Status constructor_status_;
};
```

Constructor (oblique.cc:1367). **Built fresh for every node** — three O(max_feature_idx)
vector fills + one `ColumnWithCastWithStatus` (dynamic_cast) per feature. On wide datasets
this per-node O(F) setup was −63 % e2e at 400k features when cached:

```cpp
ProjectionEvaluator::ProjectionEvaluator(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features) {
  DCHECK(!numerical_features.empty());
  const int max_feature_idx =
      *std::max_element(numerical_features.begin(), numerical_features.end());

  numerical_attributes_.assign(max_feature_idx + 1, nullptr);
  numerical_attribute_data_.assign(max_feature_idx + 1, nullptr);
  na_replacement_value_.assign(max_feature_idx + 1, 0.f);

  for (const auto attribute_idx : numerical_features) {
    na_replacement_value_[attribute_idx] =
        train_dataset.data_spec().columns(attribute_idx).numerical().mean();
  }

#if defined(ROW_MAJOR_DATASET_LAYOUT)
  const auto* row_active = dataset::RowMajorFeatureMatrix::Active();
  if (row_active != nullptr) {
    row_major_matrix_ = row_active;
    return;
  }
#endif

  for (const auto attribute_idx : numerical_features) {
    const auto column_or = train_dataset.ColumnWithCastWithStatus<
        dataset::VerticalDataset::NumericalColumn>(attribute_idx);
    constructor_status_.Update(column_or.status());
    if (!constructor_status_.ok()) break;
    numerical_attributes_[attribute_idx] = &column_or.value()->values();
    numerical_attribute_data_[attribute_idx] = column_or.value()->values().data();
  }
}
```

**The kernel itself** (oblique.cc:1410). Rows outer, projection items inner; scalar fp32
accumulator (this exact summation order is the bit-identity contract):

```cpp
YDF_PROJECTION_EVALUATE_NOINLINE
absl::Status ProjectionEvaluator::Evaluate(
    const Projection& projection,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);

  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
  values->resize(selected_examples.size());

  for (size_t selected_idx = 0; selected_idx < selected_examples.size(); selected_idx++) {
    float value = 0;
    const auto example_idx = selected_examples[selected_idx];

    // This is iterating over columns : would benefit from Row-major
    for (const auto& item : projection) {
      float attribute_value = AttributeValue(item.attribute_idx, example_idx);
#ifdef ENABLE_APPLYPROJECTION_ISNAN
      if (std::isnan(attribute_value)) {
        attribute_value = na_replacement_value_[item.attribute_idx];
      }
#endif
      value += attribute_value * item.weight;
    }
    (*values)[selected_idx] = value;
  }
  return absl::OkStatus();
}
```

Access-pattern facts: `selected_examples` is uint32, stride-1, **always sorted ascending**
(bootstrap sort + in-place partition preserve it; DCHECK in `SplitExamplesInPlace`, compiled
out in opt). The output write is sequential. Only `col[example_idx]` scatters — density of
useful floats per cache line by depth is in §13. `values` is the per-thread
`cache->projection_values` vector, so `resize` is a no-op after the first call.

---

## 5. Projection sampling: `SampleProjection` (oblique.cc:867) — (full source → `dataset_and_sampling.md`)

Consumes the tree's RandomEngine — RNG stream order is part of reproducibility (§12). Key facts:

- **Count** (`GetNumProjections`): `P = clamp(⌈F^exponent⌉, min, max)`, defaults exponent .5,
  max 1000 ⇒ P = ⌈√F⌉ (F=29 → 6; 4096 → 64; 400k → 633).
- **Density**: `projection_density = clamp(projection_density_factor / F, 0, 1)` = 1.5/F.
- **Which features**: `nnz ~ Binomial(F, density)` (mean 1.5), Floyd sampler into an
  `absl::btree_set` ⇒ **ascending attribute order**. nnz never 0 (fallback singleton).
- **Weights**: default continuous U(−1,1), `normalization=NONE`; singletons get weight
  exactly 1.0.

---

## 6. Split search over projected values → `oblique_context/split_search.md`

Downstream of AP. Dispatch `EvaluateProjection` (oblique.cc:407) routes classification to the
**histogram** finder (`FindSplitLabelClassificationFeatureNumericalHistogram`, training.cc:2229;
the default benchmark path) or the **EXACT/Cart** finder
(`FindSplitLabelClassificationFeatureNumericalCart`, training.cc:2458 →
`FindBestSplitFlatHighway` / `ScanSplitsFlat`, splitter_scanner.h) for nodes below
`dynamic_split_threshold`. The splitter sees `dense_example_idxs = iota(0..n-1)` + a
pre-gathered dense label vector, **not** real row ids. The per-node DYNAMIC downgrade
(histogram→EXACT below 250 rows) happens in oblique.cc:186, before the projection loop, so one
node uses one finder for all its projections. **Full bodies, bin generation, SIMD upper_bound,
and the measured cost splits are in the shard.**

---

## 7. The per-node driver: `FindBestConditionSparseObliqueTemplate` (oblique.cc:129)

Where AP and all kernel variants hook in. Stock setup + main-loop skeleton:

```cpp
// per node (setup, kFindObliqueSetup):
ProjectionEvaluator projection_evaluator(train_dataset, numerical_features);  // O(F) rebuild — see §13
const auto selected_labels = ExtractLabels(label_stats, selected_examples);   // labels[bag[i]] gather; "// TODO: Cache."
std::vector<UnsignedExampleIdx> dense_example_idxs(n); std::iota(...);         // = iota(0..n-1)
// [dynamic downgrade: dt_config copy, histogram→EXACT if n < dynamic_split_threshold — see §6 / split_search.md]

// MAIN LOOP (stock path):
for (int p = 0; p < num_projections; ++p) {
  SampleProjection(..., &current_projection, ...);                            // kSampleProjection
  projection_evaluator.Evaluate(current_projection,
                                selected_examples, &projection_values);       // ★ ApplyProjection
  result = EvaluateProjection(..., projection_values, ..., best_condition);   // split search (§6)
  if (result == kBetterSplitFound) {
    best_projection = current_projection;
    best_threshold  = best_condition->condition().higher_condition().threshold();
  }
}
// SetCondition(best_projection, best_threshold, ...) materializes the winner into the node proto
// (oblique_condition: attributes[], weights[], na_replacements[]=column means, threshold).
```

Upstream: `FindBestConditionOblique` (oblique.cc:777) → `FindBestConditionSingleThreadManager`
(training.cc:1441, which then also runs the near-no-op axis-aligned pass over
`candidate_attributes`). RF never sets `internal_config.split_finder_processor` ⇒ always the
single-thread branch (training.cc:1896).

**The full body — with every variant `#ifdef` branch (SYMMETRIC_DEPTHWISE_AP precomputed-slab
hook, SYMMETRIC_NODEWISE_CONTROL, SUBTREE_GATHER_CACHE) — is in
`oblique_context/kernel_variants.md`.** Any kernel that produces the same slab values in slab
order `slab[p*rows_n + i]` can feed the `precomputed_projected_values` hook without touching
the finders (§12.4).

---

## 8. Node lifecycle and tree growth — (full bodies → `oblique_context/tree_growth.md`)

- **`NodeTrain`** (training.cc:5165): set leaf → `FindBestCondition` → `SplitExamplesInPlace`
  → push **neg then pos** onto the stack.
- **Sortedness invariant**: every node's `selected_examples` is **uint32, sorted ascending,
  at every depth** (bootstrap sorted; in-place partition of a sorted list stays sorted;
  DCHECKs in `SplitExamplesInPlace`, compiled out in opt). Duplicates stay adjacent.
  `SplitExamplesInPlace` (training.cc:5891) **re-computes** the winning projection's
  dot-products (`EvalConditionOnDataset`) to partition `active` in place; children's spans are
  contiguous sub-spans of the parent's.
- **DFS** (default, `GrowTreeLocal` training.cc:5384): LIFO stack, push neg-then-pos ⇒
  pos-child-first, depth-first; a node's whole positive subtree completes before its negative
  sibling. RNG consumed in that order.
- **BFS** (`GrowTreeLocalBFS` training.cc:5453; compiled in by dw1 / symmetric / bfs_only):
  per-depth `depth_batch` — the seam where fused per-level Apply hooks in (slab handoff fields
  on `InternalTrainConfig`, training.h:309).
- **BFS vs DFS reproducibility**: BFS consumes RNG in level order, DFS in depth-first order ⇒
  trees at the same seed legitimately differ between schedulers (accuracy-equivalent;
  bit-identity only *within* one scheduler). Measured: **BFS hurts tall-narrow** (HIGGS BFS
  336 s vs DFS 282 s) — BFS-fused designs must first win that back.

---

## 9. Forest driver: `RandomForestLearner::TrainWithStatusImpl` (random_forest.cc:416) — (full → `tree_growth.md`)

ThreadPool "TrainRF", **one tree per task/thread**; per-tree `utils::RandomEngine(tree_seeds[tree_idx])`
(seeds pre-generated). Bootstrap `SampleTrainingExamples` (random_forest.cc:1586): with
defaults (`bootstrap_size_ratio=1.0`, `sampling_with_replacement=true`) bag size = nrow,
**sorted, with duplicates**, ~63.2 % distinct rows. RF never sets `split_finder_processor` ⇒
single-thread split search per node. `decision_tree::Train` copies the bag; `DecisionTreeCoreTrain`
wraps it in a `SelectedExamplesRollingBuffer` with an equal-size scratch buffer.

---

## 10. Kernel variants (this fork's experiments) — (details → `oblique_context/kernel_variants.md`)

Rule (`AGENTS.md` + ablation memory): **one idea = one `--config` = one branch; never
stack ideas in a measurement.** Controls stay pure. Variants present **on this branch**:

| Variant | Config | File / function | Status |
|---|---|---|---|
| Stock nodewise Evaluate | *(none)* | `oblique.cc` `ProjectionEvaluator::Evaluate` | baseline |
| BFS-only control | `--config=bfs_only` | `GrowTreeLocalBFS` fallback branch | scheduler ablation |
| DW1 depthwise 1-pass (col-sharing) | `--config=depthwise_1_pass` | `oblique_cpu_depthwise_1pass.cc` `ApplyProjectionsDepthwise1Pass` | ≤15 % slower than BFS; "col sharing via cache residency doesn't work at scale" |
| DW1 shared-rows | `--config=dw1_shared_rows` (implies dw1) | same file, `#ifdef DW1_SHARED_ROWS` | ⛔ 1.3–3.8× slower; postmortem §13 |
| Symmetric depthwise AP | `--config=symmetric_depthwise_ap` | `oblique_cpu_symmetric_depthwise_ap.cc` | ✚ changes model semantics; wins wide-trunk, ties BFS on HIGGS |
| Symmetric nodewise control | `--config=symmetric_nodewise_control` | shared sampling, node-local Evaluate | control |
| Subtree gather cache | `--config=subtree_gather` | `oblique.cc` `#ifdef SUBTREE_GATHER_CACHE` | ⛔ +43 % (≈2 % feature overlap ⇒ gather never amortizes) |
| Row-major store | `--config=row_major_dataset_layout` + `--dataset_layout=row` | `RowMajorFeatureMatrix` via `AttributeValue` | layout experiment |

Kernel internals (DW1 col-sharing + shared-rows, symmetric depthwise AP, the subtree-gather
dead end) and the full driver hookup are in the shard.

---

## 11. Building, running, measuring → `oblique_context/build_measure.md`

The `.bazelrc` experiment configs, env knobs (`YDF_RM_MAX_ROWS`, `YDF_DW1_BLOCK_FLOATS`,
`YDF_DW1_MIN_DEPTH`, `YDF_SYMMETRIC_MAX_DEPTH`, `YDF_SG_BUDGET_MB`), harness defaults
(`examples/train_oblique_forest.cc`: Oblique + Dynamic Random Histogram, 64 bins, threshold
250, P exp .5, density 1.5, seed 1234, one tree/thread), input modes, the trunk generator, the
standard dataset shapes, and the measurement tooling (`runtime.sh`, `accuracy.sh`,
`parallel_chrono.py`; **<15 % e2e = failed experiment**; Mac numbers don't count) all live in
the shard. The experiment *workflow* is in `AGENTS.md`.

---

## 12. Invariants any new AP kernel must respect

1. **Bit-identical trees vs the stock path** at the same seed and scheduler — same fp32
   summation order per projected value (rows outer / items inner, scalar accumulator), same
   RNG consumption sequence. `accuracy.sh` must match exactly. (Semantics-changing designs
   like symmetric are a separate, explicitly-labeled category.)
2. `selected_examples` (and every node's row list) is **uint32, sorted ascending, may
   contain duplicates** (bootstrap). Duplicated rows belong to the same node. At a BFS
   depth, node row-lists partition the depth's bag (disjoint across nodes).
3. Projections: sparse, ascending attribute order, nnz ≥ 1, singleton weight = 1.0,
   projected values never NaN (asserted in debug).
4. The split search only needs `projection_values[0..n)` + dense labels — any kernel that
   produces the same slab values in slab order `slab[p*rows_n + i]` can feed the
   `precomputed_projected_values` hook (§7/§8) without touching the finders.
5. Kernels run **single-threaded** on the tree's thread (tree-level pool owns all cores);
   per-thread scratch caches (`SplitterPerThreadCache`, `Dw1Scratch`, thread_locals) are
   reused across nodes/depths/trees — don't churn allocations in the hot loop.
6. No sub-fp32 column storage (bf16/fp16/int8) — ruled out.
7. Research-grade only: speedups must be publishable (no "turn off logging" wins).
8. One idea per build config per branch; keep controls pure; attribute deltas to the single
   changed variable.

---

## 13. Measured structural facts (distilled; dates = when measured)

- **AP is DRAM-latency-bound.** MLP (independent accumulators) is the only kernel lever
  that has paid (V2-rev3/evaluate_4row). Software prefetch, HW gathers (AVX2/512),
  loop-order swaps: all failed.
- **Gather density by depth** (2026-06-20, trunk 100k×128, dw1 gather `col[sel_ptr[i]]`,
  useful floats per 64 B line, line = 16 floats): L1=8.19, L2=4.90, L3=3.37, L4=2.55,
  L5=2.13, L6=1.95, L7=1.91 — steep decay then a **plateau ≈1.9** (correlated splits keep
  rows partially contiguous; never reaches worst-case 1.0). Half the line efficiency is
  gone by L2. Sorting can't help (`sel` already sorted — the cost is sparsity, not
  disorder). At production sizes the misses become DRAM misses (~10× cost).
- **DW1 shared-rows postmortem** (2026-07-01): 1.3–3.8× slower than col-sharing control.
  (a) scatter-write RFO amplification on tall-narrow — bag walked row-major, slabs
  node-major ⇒ ~nodes-with-ref interleaved write streams per column; once streams×64 B
  exceed ~2 MB per-thread cache share, each 4 B `+=` costs ≈32× write amplification;
  (b) O(touched_cols × block_rows) bag re-scan on wide — 74–98 % of visits skip at
  `off<0`. **Jointly infeasible**: write-residency needs block_rows ≤ cache/(P·4B) while
  read-density needs block union ≳1/16 of row range — mutually unsatisfiable by ~8× on
  HIGGS. #scattered writes = #gathered reads, and a scattered write costs ~2 lines vs 1
  for a gather read ⇒ negative ceiling under col-major reads + node-major slabs.
- **Symmetric upper-bounds shared-rows** in the current architecture; it wins only where
  fewer nodes/depth keep its write streams resident (wide trunks), ties BFS on HIGGS.
- **Feature-overlap math kills subtree caching:** P=⌈F^0.5⌉ ⇒ a node's projections touch
  ~P·nnz distinct features; overlap with descendants ~2 %; no ancestor-descendant column
  reuse to exploit. [standing conclusion #3]
- **Per-node O(F) setup dominates ultra-wide datasets** (ProjectionEvaluator rebuild +
  `ExtractLabels`, the `// TODO: Cache.` lines): caching it was −63 % e2e at 400k features.
  Check for more O(F)-per-node costs before touching kernels. [#4]
- **BFS hurts tall-narrow** (HIGGS 336 vs 282 s) — depthwise/BFS-fused designs must first
  win back that handicap. [#5]
- **The dual store (Dynamic_Row_Col_Major)** is the fastest known where it fits (transpose
  prepaid in RAM, not traffic) but takes 2× dataset memory ⇒ OOM on 3M×4096 at 61 GiB.
  The memory-safe leader is `evaluate_4row + cache_projection_evaluator`.
