# dataset_and_sampling.md — inputs to ApplyProjection

> Shard of `OBLIQUE_CONTEXT.md` (the lean core). Read this when a question is
> about the `VerticalDataset` memory layout, the row-major alternate store, or
> the full `SampleProjection` / `GetNumProjections` source (weight modes,
> normalization, Floyd sampler). Snapshot as of 2026-07-04, branch `rebased-main`,
> commit `c80ffbf7`. Line numbers drift; grep for the symbol if a ref misses.

---

## Dataset: column-major `VerticalDataset`

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
`const float*` per attribute once per node (see core §4). The label column is a
`CategoricalColumn` (`TemplateScalarStorage<int32_t>`), values ∈ {1,2}.

### Alternate store: `RowMajorFeatureMatrix` (`dataset/row_major_feature_matrix.h`, 62 lines)

Process-global optional fp32 **row-major** mirror (`Get(row, col)`, `Set`, static
`SetActive/Active`), filled by the harness when `--dataset_layout=row` and compiled in by
`--config=row_major_dataset_layout`. When active, `ProjectionEvaluator` routes
`AttributeValue()` through it instead of the column store (same loop, different layout).
NaN is replaced by column mean at fill time.

---

## Projection sampling: `SampleProjection` (oblique.cc:867)

Consumes the tree's RandomEngine — RNG stream order is part of reproducibility (core §12).

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
