# split_search.md — the split finders (downstream of ApplyProjection)

> Shard of `OBLIQUE_CONTEXT.md` (the lean core). Read this when a question is
> about what happens to the projected values: the dispatch, the histogram finder
> (default benchmark path), the EXACT/Cart finder (VQSort + scan), bin
> generation, the SIMD upper_bound, or the per-node dynamic downgrade. Snapshot
> as of 2026-07-04, branch `rebased-main`, commit `c80ffbf7`. Line numbers drift;
> grep for the symbol if a ref misses.

---

## Split search over projected values

### Dispatch: `EvaluateProjection` (oblique.cc:407)

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

### Histogram finder (default benchmark path) — training.cc:2229

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

### EXACT finder (nodes < dynamic_split_threshold, and all EXACT runs)

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

### Per-node DYNAMIC downgrade (histogram→EXACT for small nodes)

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
