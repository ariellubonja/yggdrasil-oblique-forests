# OBLIQUE_CONTEXT.md — the code map for ApplyProjection optimization

> **Purpose.** Single-file context for every AI/agent session working on this fork.
> The project is: **make `ProjectionEvaluator::Evaluate()` (a.k.a. ApplyProjection) in
> `oblique.cc` faster**, within research-grade constraints. This file contains the
> raw source of everything on the hot path plus the structural facts around it, so a
> session never has to re-derive them from ~16k lines spread across many files.
>
> **Division of labor between the standing docs:**
> - `AGENTS.md` — the *workflow* (experiment loop, measurement rules, logging contract).
> - **This file** — the *code*: call stack, raw functions, variants, configs, invariants.
> - `E2E + Chrono coarse. Col-major - m7i.metal-24x Bootstrapping.csv` — the end-to-end timing results
>
> Code snapshots and `file:line` refs are as of 2026-07-04, branch `rebased-main`,
> commit `c80ffbf7`.
> Line numbers drift; grep for the symbol if a ref misses.

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

## 3. Dataset: column-major `VerticalDataset`

`yggdrasil_decision_forests/dataset/vertical_dataset.h`. One heap `std::vector<float>` per
column; no interleaving. Row index type (`dataset/types.h`):

```cpp
// ExampleIdx is controlled by the --define=ydf_example_idx_num_bits={32,64}
typedef uint32_t UnsignedExampleIdx;   // default 32-bit; dw1 shared-rows static_asserts this
```

```cpp
class VerticalDataset {
  // Storage of scalar values.
  template <typename T>
  class TemplateScalarStorage : public AbstractColumn {
   public:
    using Format = T;
    row_t nrows() const override { return values_.size(); };
    void Add(const T& value) { values_.push_back(value); }
    // Access to values.
    const std::vector<T>& values() const { return values_; }
    std::vector<T>* mutable_values() { return &values_; }
    // … [truncated: Reserve/ExtractAndAppend/memory_usage/ShrinkToFit]
   private:
    std::vector<T> values_;
  };

  class NumericalColumn : public TemplateScalarStorage<float> {
   public:
    proto::ColumnType type() const override { return proto::ColumnType::NUMERICAL; }
    bool IsNa(const row_t row) const override { return std::isnan(values()[row]); }
    // … [truncated: Add/Set/Resize NA plumbing; kNaValue = NaN]
  };

  row_t nrow() const { return nrow_; }
  template <typename T> absl::StatusOr<const T*> ColumnWithCastWithStatus(int col) const;
  // … [truncated: CategoricalColumn (int32 store, used for the label), other column types]
};

template <typename T>
absl::StatusOr<const T*> VerticalDataset::ColumnWithCastWithStatus(int col) const {
  const auto* abstract_column = column(col);
  const T* const casted_column = dynamic_cast<const T* const>(abstract_column);
  if (!casted_column) { return absl::InvalidArgumentError(/*…*/""); }
  return casted_column;
}
```

The hot path never goes through the virtual interface: `ProjectionEvaluator` caches raw
`const float*` per attribute once per node (§5). The label column is a
`CategoricalColumn` (`TemplateScalarStorage<int32_t>`), values ∈ {1,2}.

### Alternate store: `RowMajorFeatureMatrix` (`dataset/row_major_feature_matrix.h`, 62 lines)

Process-global optional fp32 **row-major** mirror (`Get(row, col)`, `Set`, static
`SetActive/Active`), filled by the harness when `--dataset_layout=row` and compiled in by
`--config=row_major_dataset_layout`. When active, `ProjectionEvaluator` routes
`AttributeValue()` through it instead of the column store (same loop, different layout).
NaN is replaced by column mean at fill time.

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

## 5. Projection sampling: `SampleProjection` (oblique.cc:867)

Consumes the tree's RandomEngine — RNG stream order is part of reproducibility (§12).

```cpp
int GetNumProjections(const proto::DecisionTreeTrainingConfig& dt_config,
                      const int num_numerical_features) {
  if (num_numerical_features <= 1) return 1;
  const int max_num_projections = dt_config.sparse_oblique_split().max_num_projections();
  const int min_num_projections =
      std::min(dt_config.sparse_oblique_split().min_num_projections(), num_numerical_features);
  const int target_num_projections =
      0.5 + std::ceil(std::pow(num_numerical_features,
                dt_config.sparse_oblique_split().num_projections_exponent()));
  return std::max(std::min(target_num_projections, max_num_projections), min_num_projections);
}
```

```cpp
void SampleProjection(const absl::Span<const int>& features,
                      const proto::DecisionTreeTrainingConfig& dt_config,
                      const dataset::proto::DataSpecification& data_spec,
                      const model::proto::TrainingConfigLinking& config_link,
                      const float projection_density,
                      internal::Projection* projection,
                      int8_t* monotonic_direction,
                      utils::RandomEngine* random) {
  *monotonic_direction = 0;
  projection->clear();
  std::uniform_real_distribution<float> unif01;
  std::uniform_real_distribution<float> unif1m1(-1.f, 1.f);
  const auto& oblique_config = dt_config.sparse_oblique_split();

  const auto gen_weight = [&](const int feature) -> float {
    float weight = unif1m1(*random);
    // … [truncated: kBinary / kPowerOfTwo / kInteger weight modes — default is
    //    continuous U(-1,1); monotonic-constraint sign flip — unused here]
    const auto& spec = data_spec.columns(feature).numerical();
    switch (oblique_config.normalization()) {
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
        return weight;                       // ← default
      // … [truncated: STANDARD_DEVIATION, MIN_MAX normalizations]
    }
  };

  std::binomial_distribution<size_t> binom(features.size(), projection_density);
  const size_t num_selected_features = binom(*random);

  absl::btree_set<size_t> picked_idx;
  // Floyd's sampler to select k indices uniformly
  for (size_t j = features.size() - num_selected_features; j < features.size(); ++j) {
    size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
    if (!picked_idx.insert(t).second) picked_idx.insert(j);
  }

  projection->reserve(projection_density * features.size());
  for (const auto idx : picked_idx) {              // btree ⇒ ascending attribute order
    projection->push_back({features[idx], gen_weight(features[idx])});
  }

  if (projection->empty()) {
    std::uniform_int_distribution<int> unif_feature_idx(0, features.size() - 1);
    projection->push_back({features[unif_feature_idx(*random)], /*weight=*/1.f});
  } else if (projection->size() == 1) {
    projection->front().weight = 1.f;
  }
  // … [truncated: max_num_features re-sampling — max_num_features=0 (off) here]
}
```

Notes: `projection_density = clamp(projection_density_factor / F, 0, 1)` (=1.5/F);
projections are sparse lists in **ascending attribute order**; nnz is Binomial ⇒ mean 1.5,
never 0 (fallback singleton), and singletons get weight exactly 1.0.

---

## 6. Split search over projected values

### 6.1 Dispatch: `EvaluateProjection` (oblique.cc:407)

```cpp
template <typename LabelStats, typename Labels>
absl::StatusOr<SplitSearchResult> EvaluateProjection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache,
    utils::RandomEngine* random) {
  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kEvaluateProj);
  InternalTrainConfig effective_internal_config = internal_config;
  effective_internal_config.override_sorting_strategy = /* IN_NODE */;

  const UnsignedExampleIdx min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  // Projection are never missing.
  const float na_replacement = 0;

  SplitSearchResult result;
  if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
    // For oblique splits, route classification through the histogram finder
    // when the user picked a histogram-based type, otherwise use the default
    // sort-based Cart path.
    if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
      ASSIGN_OR_RETURN(result, FindSplitLabelClassificationFeatureNumericalCart(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, label_stats.num_label_classes, na_replacement,
              min_num_obs, dt_config, label_stats.label_distribution,
              first_attribute_idx, effective_internal_config, condition, cache));
    } else {
      ASSIGN_OR_RETURN(result, FindSplitLabelClassificationFeatureNumericalHistogram(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, label_stats.num_label_classes, na_replacement,
              min_num_obs, dt_config, label_stats.label_distribution,
              first_attribute_idx, random, condition));
    }
  }
  // … [truncated: RegressionHessianLabelStats / RegressionLabelStats branches]
  return result;
}
```

Key indirection: the splitter does **not** see the node's real row ids. It gets
`dense_example_idxs = iota(0..n-1)` and indexes `projection_values[i]` /
`selected_labels[i]` — labels were pre-gathered into a dense vector by `ExtractLabels`
(one `labels[bag[i]]` gather per node, per `Extract<T>` in oblique.cc:66). A better split
must beat `condition->split_score()`, which carries the best score found so far **across
projections and across the axis-aligned pass** (the condition proto is shared).

### 6.2 Histogram finder (default benchmark path) — training.cc:2229

```cpp
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalHistogram(
    const absl::Span<const UnsignedExampleIdx> selected_examples,   // = iota(0..n-1)
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  CHRONO_SCOPE(…kHistoPath);
  struct CandidateSplit {
    float threshold;
    utils::IntegerDistributionDouble pos_label_distribution;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const { return threshold < other.threshold; }
  };

  float min_value, max_value;
  std::vector<float> bins;
  std::vector<CandidateSplit> candidate_splits;
  SIMDUpperBoundBins bins_accel;
  bool use_equal_width_fast_path;

  {  // kHistogramSetup
    // … [truncated: LOCAL_IMPUTATION branch — not used]
    {  // kMinMaxNumerical: one pass over attributes[selected]
      if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value, &max_value))
        return SplitSearchResult::kInvalidAttribute;
    }
    if (min_value == max_value) return SplitSearchResult::kInvalidAttribute;

    ASSIGN_OR_RETURN(bins, internal::GenHistogramBins(
        dt_config.numerical_split().type(), dt_config.numerical_split().num_candidates(),
        attributes, min_value, max_value, random));

    candidate_splits.resize(bins.size());
    for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
      candidate_splits[split_idx].pos_label_distribution.SetNumClasses(num_label_classes);
      candidate_splits[split_idx].threshold = bins[split_idx];
    }
    // SIMD-vectorised replacement for the per-example std::upper_bound (sizes 64/256).
    bins_accel.Init(bins);
    use_equal_width_fast_path = /* type is *_EQUAL_WIDTH* */;
  }

  {  // kAssignSamplesToHistogram — the O(n) loop
    for (const auto example_idx : selected_examples) {
      const int32_t label = labels[example_idx];
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      float attribute = attributes[example_idx];
      if (std::isnan(attribute)) attribute = na_replacement;

      if (use_equal_width_fast_path) {
        const int idx = EqualWidthThresholdIndex(attribute, min_value, max_value,
                                                 static_cast<int>(candidate_splits.size()));
        if (idx < 0) continue;
        auto& it_split = candidate_splits[idx];
        it_split.num_positive_examples_without_weights++;
        it_split.pos_label_distribution.Add(label, weight);
      } else {
        const int idx = bins_accel.Index(attribute);     // SIMD upper_bound − 1
        if (idx < 0) continue;
        auto& it_split = candidate_splits[idx];
        it_split.num_positive_examples_without_weights++;
        it_split.pos_label_distribution.Add(label, weight);
      }
    }
    // Suffix-sum the per-bin counts into cumulative ">= threshold" counts.
    for (int split_idx = candidate_splits.size() - 2; split_idx >= 0; split_idx--) {
      const auto& src = candidate_splits[split_idx + 1];
      auto& dst = candidate_splits[split_idx];
      dst.num_positive_examples_without_weights += src.num_positive_examples_without_weights;
      dst.pos_label_distribution.Add(src.pos_label_distribution);
    }
  }

  // Inline entropy computation … amortizes the per-candidate confusion-matrix setup.
  const double initial_entropy = label_distribution.Entropy();
  const int num_classes = label_distribution.NumClasses();
  const double total_sum = label_distribution.NumObservations();
  const double inv_total = (total_sum > 0) ? 1.0 / total_sum : 0.0;
#ifndef DISABLE_BINARY_ENTROPY_LOOKUP
  const bool use_unweighted_binary_entropy = weights.empty() && num_label_classes == 3;
  std::vector<double> count_log_count;
  if (use_unweighted_binary_entropy) {   // kEntropyTableSetup
    count_log_count = internal::BuildCountLogCountTable(static_cast<int64_t>(total_sum));
  }
#endif

  bool found_split = false;
  {  // kSelectBestThresholdHistogram — O(num_bins)
    for (auto& candidate_split : candidate_splits) {
      if (selected_examples.size() - candidate_split.num_positive_examples_without_weights
              < min_num_obs ||
          candidate_split.num_positive_examples_without_weights < min_num_obs) continue;

      const auto& pos = candidate_split.pos_label_distribution;
      const double pos_sum = pos.NumObservations();
      const double neg_sum = total_sum - pos_sum;
      double final_entropy;
      if (use_unweighted_binary_entropy) {
        // integer-count entropy via the count*log(count) lookup table (default ON;
        // ~9% e2e win — see .bazelrc disable_binary_entropy_lookup to A/B)
        final_entropy = (…BinaryEntropyNumeratorFromIntegerCounts(pos…)
                       + …BinaryEntropyNumeratorFromIntegerCounts(neg…)) * inv_total;
      } else {
        // … [truncated: generic Σ −p·log(p) over classes]
      }
      const double information_gain = initial_entropy - final_entropy;
      if (information_gain > condition->split_score()) {
        condition->set_split_score(information_gain);
        condition->mutable_condition()->mutable_higher_condition()
                 ->set_threshold(candidate_split.threshold);
        condition->set_attribute(attribute_idx);
        condition->set_num_training_examples_without_weight(selected_examples.size());
        condition->set_num_training_examples_with_weight(total_sum);
        condition->set_num_pos_training_examples_without_weight(
            candidate_split.num_positive_examples_without_weights);
        condition->set_num_pos_training_examples_with_weight(pos_sum);
        condition->set_na_value(na_replacement >= candidate_split.threshold);
        found_split = true;
      }
    }
  }
  return found_split ? SplitSearchResult::kBetterSplitFound
                     : SplitSearchResult::kNoBetterSplitFound;
}
```

Bin generation (training.cc:5859) — RANDOM draws `num_candidates` U(min,max) thresholds
(consumes RNG!), EQUAL_WIDTH is deterministic mid-bin; both end VQSorted:

```cpp
absl::StatusOr<std::vector<float>> GenHistogramBins(
    const proto::NumericalSplit::Type type, const int num_splits,
    const absl::Span<const float> attributes, const float min_value,
    const float max_value, utils::RandomEngine* random) {
  std::vector<float> candidate_splits(num_splits);
  switch (type) {
    case proto::NumericalSplit::HISTOGRAM_RANDOM:
    case proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM: {
      std::uniform_real_distribution<float> threshold_distribution(min_value, max_value);
      for (auto& candidate_split : candidate_splits)
        candidate_split = threshold_distribution(*random);
    } break;
    case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH:
    case proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM: {
      for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++)
        candidate_splits[split_idx] = min_value + (max_value - min_value)
                                      * (split_idx + 0.5f) / candidate_splits.size();
    } break;
    default: return absl::InvalidArgumentError("Numerical histogram not implemented");
  }
  hwy::VQSort(candidate_splits.data(), candidate_splits.size(), hwy::SortAscending());
  return candidate_splits;
}
```

`SIMDUpperBoundBins` (training.cc, `/* #region SIMD upper_bound */`): drop-in
`std::upper_bound` replacement over the sorted thresholds, specialized for 64 bins (AVX2
8×8) and 256 bins (AVX-512 16×16); default-enabled via `ENABLE_STD_UPPER_BOUND_VECTORIZATION`
+ `-march=native` in `.bazelrc`. 256-bin AVX-512 measured ~3–4× vs scalar upper_bound.

### 6.3 EXACT finder (nodes < dynamic_split_threshold, and all EXACT runs)

`FindSplitLabelClassificationFeatureNumericalCart` (training.cc:2458) — for our workload
(`num_label_classes==3`, `weights.empty()`, IN_NODE strategy forced by
`override_sorting_strategy`) it reduces to:

```cpp
absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureNumericalCart(…) {
  CHRONO_SCOPE(…kCartPath);
  const auto feature_filler = /* kCartSetup: FeatureNumericalBucket::Filler(n, na, attributes) */;
  if (num_label_classes == 3) {         // binary classification
    if (weights.empty()) {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Filler label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Initializer initializer(label_distribution);
      // sorting_strategy == IN_NODE on the oblique path:
      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
      // … [truncated: FORCE_PRESORTED → ScanSplitsPresortedSparse — never taken for
      //    projections (values are per-node), only for raw axis-aligned features]
    }
    // … [truncated: weighted variant]
  }
  // … [truncated: multi-class variants]
}
```

with (splitter_scanner.h:1976):

```cpp
constexpr auto FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical =
    FindBestSplitFlatHighway<FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
                             LabelBinaryCategoricalScoreAccumulator>;
```

`FindBestSplitFlatHighway` (splitter_scanner.h:1746): pack each example into a
`hwy::K32V32` — `key = FloatToUintForSort(projection_value)` (order-preserving
float→uint32 bijection), `value = label` — then Highway `VQSort` the flat array, then a
linear scan:

```cpp
template <typename ExampleBucketSet, typename LabelBucketSet>
SplitSearchResult FindBestSplitFlatHighway(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const typename ExampleBucketSet::FeatureBucketType::Filler& feature_filler,
    const typename ExampleBucketSet::LabelBucketType::Filler& label_filler,
    const typename ExampleBucketSet::LabelBucketType::Initializer& initializer,
    const int min_num_obs, const int attribute_idx,
    proto::NodeCondition* condition, PerThreadCacheV2* cache) {
  using FlatTraits = FlatBucketTraits<typename ExampleBucketSet::ExampleBucketType>;
  const size_t num_selected_examples = selected_examples.size();
  if (num_selected_examples <= 1) return SplitSearchResult::kInvalidAttribute;

  // kSortInitBuckets: resize cache->hwy_k32v32_buffer
  auto* hwy_buffer = FlatTraits::GetBuffer(cache);
  ExampleBucketType temp_bucket; /* … */

  {  // kSortFillBuckets
    for (size_t select_idx = 0; select_idx < num_selected_examples; ++select_idx) {
      const UnsignedExampleIdx example_idx = selected_examples[select_idx];
      feature_filler.ConsumeExample(example_idx, &temp_bucket.feature);  // reads attributes[i]
      label_filler.ConsumeExample(example_idx, &temp_bucket.label);      // reads labels[i]
      label_filler.Finalize(&temp_bucket.label);
      FlatTraits::Pack(hwy_buffer[select_idx], temp_bucket);
    }
  }
  {  // kSortFeatures
    hwy::VQSort(hwy_buffer, num_selected_examples, hwy::SortAscending());
  }
  {  // kSortScanSplits
    return ScanSplitsFlat<ExampleBucketSet, LabelBucketSet>(
        feature_filler, initializer, cache, selected_examples.size(),
        min_num_obs, attribute_idx, condition);
  }
}
```

```cpp
// FloatToUintForSort: bijective, order-preserving float→uint32 (flip sign bit or all bits).
HWY_INLINE uint32_t FloatToUintForSort(float f) {
  uint32_t k = hwy::BitCastScalar<uint32_t>(f);
  uint32_t mask = -static_cast<int32_t>(k >> 31) | 0x80000000;
  return k ^ mask;
}
```

`ScanSplitsFlat` (splitter_scanner.h:1633) — sweep the sorted array left→right moving one
example at a time from the positive to the negative accumulator; propose threshold between
adjacent distinct values (`IsValidSplit` = `feature < next_feature`); score = entropy gain
via `LabelBinaryCategoricalScoreAccumulator` (integer counts + optional count·log(count)
lookup table); on win, `SetConditionFinalFromThresholds(cur, next, condition)` sets the
midpoint threshold and the pos-count bookkeeping fields (same fields as the histogram
finder). Returns `kBetterSplitFound / kNoBetterSplitFound / kInvalidAttribute`.

**Cost split within EXACT** (measured): VQSort (`SortFeatures`) + scan (`SortScanSplits`)
dominate; the count·log·count table (default ON) took ~39 % off SortScanSplits.

### 6.4 Per-node DYNAMIC downgrade (histogram→EXACT for small nodes)

Inside `FindBestConditionSparseObliqueTemplate` (oblique.cc:186), **before** the projection
loop — so one node uses one finder type for all its projections:

```cpp
  /* #region Dynamic histogram downgrade */
  // Per-node downgrade for the DYNAMIC_* histogram split types: if the node
  // has fewer than dynamic_split_threshold examples, switch back to EXACT
  // (sorting). Histogramming is faster on larger nodes; EXACT wins on small
  // ones. Empirical threshold default is 250 (see decision_tree.proto).
  proto::DecisionTreeTrainingConfig dynamic_dt_config = dt_config;   // NB: proto copy, per node
  const auto split_type = dt_config.numerical_split().type();
  const bool is_dynamic_histogram =
      split_type == proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM ||
      split_type == proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM;
  if (is_dynamic_histogram) {
    const int dynamic_split_threshold =
        dt_config.sparse_oblique_split().dynamic_split_threshold();
    if (dynamic_split_threshold >= 0 &&
        dense_example_idxs.size() < static_cast<size_t>(dynamic_split_threshold)) {
      dynamic_dt_config.mutable_numerical_split()->set_type(proto::NumericalSplit::EXACT);
    }
  }
  /* #endregion */
```

---

## 7. The per-node driver: `FindBestConditionSparseObliqueTemplate` (oblique.cc:129)

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

  // … [§6.4 dynamic downgrade block shown above] …

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
#ifdef SUBTREE_GATHER_CACHE
    SubtreeGatherCache* const sg = &cache->subtree_gather;
    const bool sg_active = internal::PrepareSubtreeGatherNode(
        selected_examples, train_dataset.nrow(), sg);
#endif

// MAIN LOOP
  for (int projection_idx = 0; projection_idx < num_projections; projection_idx++) {
    int8_t monotonic_direction;
    {
      CHRONO_SCOPE_COARSE(…kSampleProjection);
      SampleProjection(config_link.numerical_features(), dynamic_dt_config,
                       train_dataset.data_spec(), config_link, projection_density,
                       &current_projection, &monotonic_direction, random);
    }

#ifdef SUBTREE_GATHER_CACHE
    if (sg_active) {
      RETURN_IF_ERROR(projection_evaluator.EvaluateWithSubtreeCache(
          current_projection, selected_examples, sg, &projection_values));
    } else
#endif
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

## 8. Node lifecycle and tree growth (order of node evaluation)

### 8.1 `NodeTrain` (training.cc:5165) — trimmed to the active path

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

### 8.2 In-place partition and the rolling buffer

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

### 8.3 DFS driver (default): `GrowTreeLocal` (training.cc:5384)

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

### 8.4 BFS driver + variant hooks: `GrowTreeLocalBFS` (training.cc:5453)

Selected by `DecisionTreeCoreTrain` (training.cc:5138) at **compile time**:

```cpp
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal: {
#if defined(DEPTHWISE_1_PASS) || defined(SYMMETRIC_DEPTHWISE_AP) ||         \
    defined(SYMMETRIC_NODEWISE_CONTROL) || defined(BFS_ONLY)
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
        current_depth >= Depthwise1PassMinDepth()) {          // env YDF_DW1_MIN_DEPTH, default 0
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
      // … [truncated: YDF_LINECOUNT_A distinct-cache-line tally; YDF_CALLGRIND_DEPTH
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
#elif defined(SYMMETRIC_DEPTHWISE_AP) || defined(SYMMETRIC_NODEWISE_CONTROL)
    if (dt_config.has_sparse_oblique_split() && depth_batch.size() >= 1 &&
        current_depth <= SymmetricMaxDepth()) {               // env YDF_SYMMETRIC_MAX_DEPTH
      // CatBoost-style symmetric trees: sample K projections ONCE for this
      // depth, shared across all nodes. The aggregate of nodes' selected
      // examples at depth d == the bag, so the projection sweep becomes
      // stride-1 in column space (vs. per-node scattered gather).
      std::vector<internal::Projection> shared_projections(num_proj);
      std::vector<int8_t> shared_monotonic(num_proj, 0);
      for (int p = 0; p < num_proj; ++p) internal::SampleProjection(…);
#ifdef SYMMETRIC_DEPTHWISE_AP
      // build sel_spans; projected(num_nodes);
      RETURN_IF_ERROR(ApplyProjectionsSymmetricDepthwiseAP(
          train_dataset, config_link.numerical_features(),
          absl::MakeConstSpan(sel_spans), absl::MakeConstSpan(shared_projections),
          absl::MakeSpan(projected)));
      // per node: node_config.{depthwise_projection_defs,depthwise_monotonic,
      //                        precomputed_projected_values} → NodeTrain
#else
      // Symmetric_Nodewise_Control: shared defs, but NO fused Apply — each node
      // falls through to the node-local Evaluate path in oblique.cc.
#endif
    } else if (dt_config.has_sparse_oblique_split()) {
      // Symmetric → DFS handoff (deeper than YDF_SYMMETRIC_MAX_DEPTH): finish each
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

## 9. Forest driver: `RandomForestLearner::TrainWithStatusImpl` (random_forest.cc:416)

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

---

## 10. Kernel variants (this fork's experiments)

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

### 10.1 DW1: `ApplyProjectionsDepthwise1Pass` (oblique_cpu_depthwise_1pass.cc)

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

### 10.2 Symmetric: `ApplyProjectionsSymmetricDepthwiseAP` (oblique_cpu_symmetric_depthwise_ap.cc)

K projections **shared by every node of the depth** ⇒ the whole depth's rows (= the bag)
can be swept per projection with stride-1 column reads. **Changes model semantics**
(shared projections constrain splits) — compare accuracy/throughput, not bit-identity.

```cpp
absl::Status ApplyProjectionsSymmetricDepthwiseAP(
    const dataset::VerticalDataset& train_dataset,
    const google::protobuf::RepeatedField<int32_t>& numerical_features,
    absl::Span<const absl::Span<const UnsignedExampleIdx>> selected_examples_per_node,
    absl::Span<const internal::Projection> shared_projections,
    absl::Span<std::vector<float>> out_projected) {
  CHRONO_SCOPE_COARSE(…kProjectionEvaluate);
  // ── Phase 1: BuildBag (kSymBuildBag) — concat per-node selected_examples
  //    (+ parallel node-id array) into one flat bag; slab pre-size K*rows_n.
  // ── Phase 2: SortBag (kSymSortBag) — pack (example_idx, node_id) into
  //    hwy::K32V32, VQSort by example id. Ties are same-node (BFS depth cohort
  //    ⇒ every copy of a row lives in one node), so the unstable sort is safe.
  //    (VQSort beat both stable_sort and a counting sort across all depths.)
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

### 10.3 Dead end kept for reference: `SubtreeGatherCache` (oblique_types.h:26, oblique.cc `#ifdef SUBTREE_GATHER_CACHE`)

Epoch-tagged per-thread block cache: when a node first drops to ≤`YDF_RM_MAX_ROWS` rows its
example list becomes the current block; feature columns gather lazily into dense
block-local arrays reused by descendants (DFS makes it effective), budget
`YDF_SG_BUDGET_MB`. **⛔ +43 % at 1.5M×4096**: with P=⌈√F⌉ and nnz≈1.5, a node and its
descendants share only ~2 % of features — gathers never amortize. Kept as the worked
example of Standing conclusion #2 ("adding memory traffic to fix locality loses").

---

## 11. Building, running, measuring

### 11.1 `.bazelrc` experiment configs (the authoritative list on this branch)

```text
build:depthwise_1_pass            --cxxopt="-DDEPTHWISE_1_PASS=1"
build:dw1_shared_rows             --config=depthwise_1_pass --cxxopt="-DDW1_SHARED_ROWS=1"
build:row_major_dataset_layout    --cxxopt="-DROW_MAJOR_DATASET_LAYOUT=1"
build:symmetric_depthwise_ap      --cxxopt="-DSYMMETRIC_DEPTHWISE_AP=1"
build:symmetric_nodewise_control  --cxxopt="-DSYMMETRIC_NODEWISE_CONTROL=1"   # don't combine with the above
build:bfs_only                    --cxxopt="-DBFS_ONLY=1"                     # mutually exclusive with symmetric_*
build:subtree_gather              --cxxopt="-DSUBTREE_GATHER_CACHE=1"
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
INT32_MAX; deeper levels hand off to DFS `GrowTreeLocal`), `YDF_SG_BUDGET_MB` (default 1024).

### 11.2 Harness: `examples/train_oblique_forest.cc`

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

### 11.3 Measurement tooling

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
   `precomputed_projected_values` hook (§7/§8.4) without touching the finders.
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

