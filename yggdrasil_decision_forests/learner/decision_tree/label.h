/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_

// Fused-per-level CPU Apply variants. V1 (NODEWISE_PROJ_MATRIX, per-node
// matrix fill) and V2 (DEPTHWISE_1_PASS, single-pass kernel across all
// (row,proj) tasks at the level) produce the same per-node slab layout, so
// variant-agnostic consumer sites key on the umbrella macro
// OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS. The two variants are mutually
// exclusive at build time.
#if defined(NODEWISE_PROJ_MATRIX) && defined(DEPTHWISE_1_PASS)
#error "NODEWISE_PROJ_MATRIX and DEPTHWISE_1_PASS are mutually exclusive"
#endif
#if defined(NODEWISE_PROJ_MATRIX) || defined(DEPTHWISE_1_PASS)
#define OBLIQUE_CPU_PRECOMPUTED_PROJECTIONS 1
#endif

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/decision_tree/gpu.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"
#include "yggdrasil_decision_forests/learner/decision_tree/preprocessing.h"
#include "yggdrasil_decision_forests/learner/decision_tree/uplift.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Forward declaration to avoid circular include with oblique_gpu.h.
class ObliqueGpuComputer;

// Label statistics.
struct LabelStats {
  virtual ~LabelStats() = default;
};

// Label statistics for Classification.
struct ClassificationLabelStats : LabelStats {
  explicit ClassificationLabelStats(const std::vector<int32_t>& label_data)
      : label_data(label_data) {}

  const std::vector<int32_t>& label_data;
  int32_t num_label_classes;
  utils::IntegerDistributionDouble label_distribution;
};

// Label statistics for Regression.
struct RegressionLabelStats : LabelStats {
  explicit RegressionLabelStats(const std::vector<float>& label_data)
      : label_data(label_data) {}

  const std::vector<float>& label_data;
  utils::NormalDistributionDouble label_distribution;
};

// Label statistics for Regression with hessian.
struct RegressionHessianLabelStats : LabelStats {
  RegressionHessianLabelStats(const std::vector<float>& gradient_data,
                              const std::vector<float>& hessian_data)
      : gradient_data(gradient_data), hessian_data(hessian_data) {}

  const std::vector<float>& gradient_data;
  const std::vector<float>& hessian_data;
  double sum_gradient;
  double sum_hessian;
  double sum_weights;
};

// Label statistics for uplift with categorical treatment and categorical
// outcome.
struct CategoricalUpliftLabelStats : LabelStats {
  explicit CategoricalUpliftLabelStats(
      const std::vector<int32_t>& outcome_values,
      const int num_unique_in_outcomes_column,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column),
        num_unique_in_outcomes_column(num_unique_in_outcomes_column) {}

  const std::vector<int32_t>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;
  int32_t num_unique_in_outcomes_column;

  UpliftLabelDistribution label_distribution;
};

// Label statistics for uplift with categorical treatment and numerical outcome.
struct NumericalUpliftLabelStats : LabelStats {
  explicit NumericalUpliftLabelStats(
      const std::vector<float>& outcome_values,
      const std::vector<int32_t>& treatment_values,
      const int num_unique_values_in_treatments_column)
      : outcome_values(outcome_values),
        treatment_values(treatment_values),
        num_unique_values_in_treatments_column(
            num_unique_values_in_treatments_column) {}

  const std::vector<float>& outcome_values;
  const std::vector<int32_t>& treatment_values;
  int32_t num_unique_values_in_treatments_column;

  UpliftLabelDistribution label_distribution;
};

// Signature of a function that sets the value (i.e. the prediction) of a leaf
// from the gradient data.
typedef std::function<absl::Status(
    const dataset::VerticalDataset&, absl::Span<const UnsignedExampleIdx>,
    const std::vector<float>&, const model::proto::TrainingConfig&,
    const model::proto::TrainingConfigLinking&, NodeWithChildren* node)>
    CreateSetLeafValueFunctor;

// Similar to CreateSetLeafValueFunctor, but use pre-computed statistics instead
// of scanning the values.
typedef std::function<absl::Status(
    const decision_tree::proto::LabelStatistics& label_stats,
    decision_tree::proto::Node* leaf)>
    SetLeafValueFromLabelStatsFunctor;

// The default policy to set the value of a leaf.
// - Distribution of the labels for classification.
// - Mean of the labels for regression.
absl::Status SetLabelDistribution(
    const dataset::VerticalDataset& train_dataset,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    NodeWithChildren* node);

// Copies the content on uplift categorical label distribution to the leafs.
void UpliftLabelDistToLeaf(const UpliftLabelDistribution& dist,
                           decision_tree::proto::NodeUpliftOutput* leaf);

// Copies the content on uplift categorical leaf output to a label distribution.
void UpliftLeafToLabelDist(const decision_tree::proto::NodeUpliftOutput& leaf,
                           UpliftLabelDistribution* dist);

// Training configuration for internal parameters not available to the user
// directly.
struct InternalTrainConfig {
  // How to set the leaf values. Used by all methods except layer-wise learning.
  CreateSetLeafValueFunctor set_leaf_value_functor = SetLabelDistribution;

  // If true, the split score relies on a hessian: ~gradient^2/hessian (+
  // regularization). This is only possible for regression. Require
  // hessian_leaf=true.
  //
  // If false, the split score is a classical decision tree score. e.g.,
  // reduction of variance in the case of regression.
  bool hessian_score = false;

  // If true, the leaf relies on the hessian. This is only possible for
  // regression.
  bool hessian_leaf = false;

  // Index of the hessian column in the dataset. Only used if hessian_leaf=true.
  int hessian_col_idx = -1;

  // Index of the gradient column in the dataset.  Only used if
  // hessian_leaf=true.
  int gradient_col_idx = -1;

  // Regularization terms for hessian_score=true.
  float hessian_l1 = 0.f;
  float hessian_l2_numerical = 0.f;
  float hessian_l2_categorical = 0.f;

  // Number of attributes tested in parallel (using fiber threads).
  int num_threads = 1;

  // Non owning pointer to pre-processing information.
  // Depending on the decision tree configuration this field might be required.
  const Preprocessing* preprocessing = nullptr;

  decision_tree::gpu::VectorSequenceComputer* vector_sequence_computer =
      nullptr;

  // Non-owning pointer to GPU-accelerated oblique projection computer.
  // Created once per training run, shared across tree threads.
  ObliqueGpuComputer* oblique_gpu_computer = nullptr;

  // Projection definitions produced by the full-GPU depthwise split pipeline
  // and handed down to per-node NodeTrain alongside `depthwise_best_split`.
  // Set per-node by GrowTreeLocalBFS before calling NodeTrain.
  const std::vector<std::vector<internal::AttributeAndWeight>>* depthwise_projection_defs = nullptr;
  const std::vector<int8_t>* depthwise_monotonic = nullptr;

  // Pre-computed best-split descriptor from the full-GPU depthwise split path.
  // When non-null, oblique.cc skips projection evaluation entirely and
  // materializes the NodeCondition directly from this descriptor + the
  // projection definition at depthwise_projection_defs[best_proj_idx].
  const BestSplitResult* depthwise_best_split = nullptr;

  // Pre-computed projected values from either fused-per-level CPU Apply
  // variant (V1 ApplyProjectionsNodewiseProjMatrix or V2
  // ApplyProjectionsDepthwise1Pass). Layout: row-minor within projection,
  // size num_projections * rows_in_node. When non-empty and
  // depthwise_projection_defs is set, oblique.cc skips SampleProjection +
  // ProjectionEvaluator::Evaluate and runs per-projection split-finding
  // directly over slices of this span.
  absl::Span<const float> precomputed_projected_values;

  // If true, the list of selected example index ("selected_examples") can
  // contain duplicated values. If false, all selected examples are expected to
  // be unique.
  bool duplicated_selected_examples = true;

  // If set, the training of the tree will stop after this time, leading to an
  // under-grow but valid decision tree. The growing strategy defines how the
  // tree is "under-grown".
  std::optional<absl::Time> timeout;

  // If set, overrides the sorting_strategy.
  absl::optional<proto::DecisionTreeTrainingConfig::Internal::SortingStrategy>
      override_sorting_strategy;
};

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONFIG_H_
