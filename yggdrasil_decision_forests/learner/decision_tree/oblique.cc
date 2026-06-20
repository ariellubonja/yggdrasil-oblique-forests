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

#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/training.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {
using std::is_same;

using Projection = internal::Projection;
using ProjectionEvaluator = internal::ProjectionEvaluator;
using LDACache = internal::LDACache;

}  // namespace

template <typename T>
std::vector<T> Extract(const std::vector<T>& values,
                       const absl::Span<const UnsignedExampleIdx> selected) {
  if (values.empty()) {
    return {};
  }
  std::vector<T> extracted(selected.size());
  for (size_t selected_idx = 0; selected_idx < selected.size();
       selected_idx++) {
    extracted[selected_idx] = values[selected[selected_idx]];
  }
  return extracted;
}

template std::vector<int32_t> Extract<int32_t>(
    const std::vector<int32_t>& values,
    const absl::Span<const UnsignedExampleIdx> selected);

template std::vector<float> Extract<float>(
    const std::vector<float>& values,
    const absl::Span<const UnsignedExampleIdx> selected);

std::vector<int32_t> ExtractLabels(
    const ClassificationLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return Extract(labels.label_data, selected);
}

std::vector<float> ExtractLabels(
    const RegressionLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return Extract(labels.label_data, selected);
}

GradientAndHessian ExtractLabels(
    const RegressionHessianLabelStats& labels,
    const absl::Span<const UnsignedExampleIdx> selected) {
  return {/*.gradient_data =*/Extract(labels.gradient_data, selected),
          /*.hessian_data =*/Extract(labels.hessian_data, selected)};
}

int GetNumProjections(const proto::DecisionTreeTrainingConfig& dt_config,
                      const int num_numerical_features) {
  if (num_numerical_features <= 1) {
    // Note: if there is only one feature, all the projections are the same.
    return 1;
  }
  const int max_num_projections =
      dt_config.sparse_oblique_split().max_num_projections();

  const int min_num_projections =
      std::min(dt_config.sparse_oblique_split().min_num_projections(),
               num_numerical_features);

  const int target_num_projections =
      0.5 + std::ceil(std::pow(
                num_numerical_features,
                dt_config.sparse_oblique_split().num_projections_exponent()));

  return std::max(std::min(target_num_projections, max_num_projections),
                  min_num_projections);
}

template <typename LabelStats>
absl::StatusOr<bool> FindBestConditionSparseObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), train_dataset.nrow());
  }

  if (config_link.numerical_features().empty()) {
    return false;
  }

  // Effective number of projections to test.
  int num_projections;
  if (override_num_projections.has_value()) {
    num_projections = override_num_projections.value();
  } else {
    num_projections =
        GetNumProjections(dt_config, config_link.numerical_features_size());
  }

  const float projection_density =
      std::clamp(dt_config.sparse_oblique_split().projection_density_factor() /
                     config_link.numerical_features_size(),
                 0.f, 1.f);

  // Best and current projections.
  Projection best_projection;
  float best_threshold;
  Projection current_projection;
  auto& projection_values = cache->projection_values;

  CHRONO_BEGIN(find_oblique_setup);
#ifdef CACHE_PROJECTION_EVALUATOR
  // ProjectionEvaluator only depends on (dataset, numerical_features), both
  // constant across every node of every tree under GLOBAL_IMPUTATION, so
  // rebuilding its per-attribute pointer tables per node is pure
  // kFindObliqueSetup waste. RANDOM_LOCAL_IMPUTATION trains each node on a
  // fresh per-node dataset whose address could alias a freed one — never
  // cache there.
  static thread_local std::optional<ProjectionEvaluator> tl_evaluator;
  static thread_local const dataset::VerticalDataset* tl_evaluator_ds =
      nullptr;
  static thread_local int64_t tl_evaluator_nrow = -1;
  static thread_local int tl_evaluator_num_features = -1;
  const bool can_cache_evaluator =
      dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::GLOBAL_IMPUTATION;
  if (!can_cache_evaluator || tl_evaluator_ds != &train_dataset ||
      tl_evaluator_nrow != static_cast<int64_t>(train_dataset.nrow()) ||
      tl_evaluator_num_features != config_link.numerical_features_size()) {
    tl_evaluator.emplace(train_dataset, config_link.numerical_features());
    tl_evaluator_ds = can_cache_evaluator ? &train_dataset : nullptr;
    tl_evaluator_nrow =
        can_cache_evaluator ? static_cast<int64_t>(train_dataset.nrow()) : -1;
    tl_evaluator_num_features =
        can_cache_evaluator ? config_link.numerical_features_size() : -1;
  }
  ProjectionEvaluator& projection_evaluator = *tl_evaluator;
#else
  ProjectionEvaluator projection_evaluator(train_dataset,
                                           config_link.numerical_features());
#endif

  // TODO: Cache.
  const auto selected_labels = ExtractLabels(label_stats, selected_examples);
  std::vector<float> selected_weights;
  if (!weights.empty()) {
    selected_weights = Extract(weights, selected_examples);
  }

  std::vector<UnsignedExampleIdx> dense_example_idxs(selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);
  CHRONO_END(find_oblique_setup,
             ::yggdrasil_decision_forests::chrono_prof::kFindObliqueSetup);

  /* #region Dynamic histogram downgrade */
  // Per-node downgrade for the DYNAMIC_* histogram split types: if the node
  // has fewer than dynamic_split_threshold examples, switch back to EXACT
  // (sorting). Histogramming is faster on larger nodes; EXACT wins on small
  // ones. Empirical threshold default is 250 (see decision_tree.proto).
  //
  // Ariel: threshold tuned on the trunk_3000000_x_4096 benchmark; revisit
  // if the histogram path gets faster (e.g. SIMD upper_bound or 1-pass).
  proto::DecisionTreeTrainingConfig dynamic_dt_config = dt_config;
  const auto split_type = dt_config.numerical_split().type();
  const bool is_dynamic_histogram =
      split_type == proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM ||
      split_type == proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM;
  if (is_dynamic_histogram) {
    const int dynamic_split_threshold =
        dt_config.sparse_oblique_split().dynamic_split_threshold();
    if (dynamic_split_threshold >= 0 &&
        dense_example_idxs.size() <
            static_cast<size_t>(dynamic_split_threshold)) {
      dynamic_dt_config.mutable_numerical_split()->set_type(
          proto::NumericalSplit::EXACT);
    }
  }
  /* #endregion */

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
    const size_t num_projs = depth_projs.size();
    const float* slab = internal_config.precomputed_projected_values.data();
    for (size_t proj_idx = 0; proj_idx < num_projs; ++proj_idx) {
      if (depth_projs[proj_idx].empty()) continue;
      const absl::Span<const float> values_span =
          absl::MakeConstSpan(slab + proj_idx * rows_n, rows_n);
      ASSIGN_OR_RETURN(
          const auto result,
          EvaluateProjection(dynamic_dt_config, label_stats, dense_example_idxs,
                             selected_weights, selected_labels, values_span,
                             internal_config,
                             depth_projs[proj_idx].front().attribute_idx,
                             constraints, depth_mono[proj_idx], best_condition,
                             cache, random));
      if (result == SplitSearchResult::kBetterSplitFound) {
        best_projection = depth_projs[proj_idx];
        best_threshold =
            best_condition->condition().higher_condition().threshold();
      }
    }
  } else
#endif
#ifdef SYMMETRIC_NODEWISE_CONTROL
  // Control-experiment path for measuring symmetric projection sampling without
  // the depthwise/bagwide column-read optimization. The caller provides the
  // same K projections for every node at this depth, but this node still
  // evaluates each projection locally with ProjectionEvaluator::Evaluate.
  const bool has_depthwise_shared_projections =
      internal_config.depthwise_projection_defs != nullptr &&
      internal_config.depthwise_monotonic != nullptr;
  if (has_depthwise_shared_projections) {
    const auto& depth_projs = *internal_config.depthwise_projection_defs;
    const auto& depth_mono = *internal_config.depthwise_monotonic;
    for (size_t proj_idx = 0; proj_idx < depth_projs.size(); ++proj_idx) {
      if (depth_projs[proj_idx].empty()) continue;

      RETURN_IF_ERROR(projection_evaluator.Evaluate(
          depth_projs[proj_idx], selected_examples, &projection_values));

      ASSIGN_OR_RETURN(
          const auto result,
          EvaluateProjection(dynamic_dt_config, label_stats, dense_example_idxs,
                             selected_weights, selected_labels,
                             projection_values, internal_config,
                             depth_projs[proj_idx].front().attribute_idx,
                             constraints, depth_mono[proj_idx], best_condition,
                             cache, random));
      if (result == SplitSearchResult::kBetterSplitFound) {
        best_projection = depth_projs[proj_idx];
        best_threshold =
            best_condition->condition().higher_condition().threshold();
      }
    }
  } else
#endif
  {
#ifdef SUBTREE_GATHER_CACHE
  // Nodes at or below the RowMajorMaxRows() threshold evaluate their
  // projections from the per-thread gathered block (materializing it when
  // this node is the first of a new subtree).
  SubtreeGatherCache* const sg = &cache->subtree_gather;
  const bool sg_active = internal::PrepareSubtreeGatherNode(
      selected_examples, train_dataset.nrow(), sg);
#endif
  for (int projection_idx = 0; projection_idx < num_projections;
       projection_idx++) {
    // Generate a current_projection.
    int8_t monotonic_direction;
    {
      CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kSampleProjection);
      SampleProjection(config_link.numerical_features(), dynamic_dt_config,
                       train_dataset.data_spec(), config_link, projection_density,
                       &current_projection, &monotonic_direction, random);
    }

    // Pre-compute the result of the current_projection.
#ifdef SUBTREE_GATHER_CACHE
    if (sg_active) {
      RETURN_IF_ERROR(projection_evaluator.EvaluateWithSubtreeCache(
          current_projection, selected_examples, sg, &projection_values));
    } else
#endif
    RETURN_IF_ERROR(projection_evaluator.Evaluate(
        current_projection, selected_examples, &projection_values));

    ASSIGN_OR_RETURN(
        const auto result,
        EvaluateProjection(
            dynamic_dt_config, label_stats, dense_example_idxs,
            selected_weights, selected_labels, projection_values,
            internal_config, current_projection.front().attribute_idx,
            constraints, monotonic_direction, best_condition, cache, random));

    if (result == SplitSearchResult::kBetterSplitFound) {
      best_projection = current_projection;
      best_threshold =
          best_condition->condition().higher_condition().threshold();
    }
  }
  }

  // Update with the actual current_projection definition.
  if (!best_projection.empty()) {
    RETURN_IF_ERROR(SetCondition(best_projection, best_threshold,
                                 train_dataset.data_spec(), best_condition));
    return true;
  }

  return false;
}

absl::Status SolveLDA(const proto::DecisionTreeTrainingConfig& dt_config,
                      const ProjectionEvaluator& projection_evaluator,
                      const std::vector<int>& selected_features,
                      const int num_classes, const std::vector<int32_t>& labels,
                      const std::vector<float>& weights, Projection* projection,
                      utils::RandomEngine* random) {
  // TODO: Cache.
  LDACache lda_cache;
  RETURN_IF_ERROR(lda_cache.ComputeClassification(
      dt_config, projection_evaluator, selected_features, num_classes, labels,
      weights));
  const auto& sb = lda_cache.FullSB();
  const auto& sw = lda_cache.FullSW();
  const int num_features = selected_features.size();
  const Eigen::Map<const Eigen::MatrixXd> eg_sw(sw.data(), num_features,
                                                num_features);
  const Eigen::Map<const Eigen::MatrixXd> eg_sb(sb.data(), num_features,
                                                num_features);

  // Inverse the SW matrice.
  Eigen::PartialPivLU<Eigen::MatrixXd> invert_solver(eg_sw);
  if (invert_solver.determinant() == 0) {
    // The matrix is not invertible.
    return absl::OkStatus();
  }
  const auto eg_w = invert_solver.inverse() * eg_sb;

  // Get the eigenvalues / vectors.
  Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(eg_w, true);

  if (eigen_solver.info() != Eigen::Success) {
    return absl::OkStatus();
  }

  const auto& eigenvalues = eigen_solver.eigenvalues();
  const auto& eigenvectors = eigen_solver.eigenvectors();

  // Get the largest eigenvalue / vector.
  int arg_abs_max = -1;
  double abs_max = 0;
  for (int i = 0; i < num_features; i++) {
    const auto value = std::abs(eigenvalues(i).real());
    if (value > abs_max) {
      arg_abs_max = i;
      abs_max = value;
    }
  }
  if (arg_abs_max == -1) {
    return absl::OkStatus();
  }

  // Convert the top eigen vector into a projection.
  projection->clear();
  for (int i = 0; i < num_features; i++) {
    const float vector = eigenvectors(i, arg_abs_max).real();
    if (vector == 0) {
      continue;
    }
    projection->push_back({selected_features[i], vector});
  }

  return absl::OkStatus();
}

struct ScoreAndThreshold {
  float score;
  float threhsold;
};

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
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kEvaluateProj);
  InternalTrainConfig effective_internal_config = internal_config;
  effective_internal_config.override_sorting_strategy =
      proto::DecisionTreeTrainingConfig::Internal::SortingStrategy::
          DecisionTreeTrainingConfig_Internal_SortingStrategy_IN_NODE;

  const UnsignedExampleIdx min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  // Projection are never missing.
  const float na_replacement = 0;
#ifndef NDEBUG
  for (const float v : projection_values) {
    DCHECK(!std::isnan(v));
  }
#endif

  // Find a good split in the current_projection.
  SplitSearchResult result;
  if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
    // For oblique splits, route classification through the histogram finder
    // when the user picked a histogram-based type, otherwise use the default
    // sort-based Cart path. `random` must be non-null on the histogram path;
    // callers using EXACT (e.g. vector_sequence.cc) can leave it as nullptr.
    if (dt_config.numerical_split().type() ==
        proto::NumericalSplit::EXACT) {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelClassificationFeatureNumericalCart(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, label_stats.num_label_classes, na_replacement,
              min_num_obs, dt_config, label_stats.label_distribution,
              first_attribute_idx, effective_internal_config, condition,
              cache));
    } else {
      DCHECK(random != nullptr)
          << "EvaluateProjection: histogram split type "
          << dt_config.numerical_split().type()
          << " requires a non-null RandomEngine.";
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelClassificationFeatureNumericalHistogram(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, label_stats.num_label_classes, na_replacement,
              min_num_obs, dt_config, label_stats.label_distribution,
              first_attribute_idx, random, condition));
    }
  } else if constexpr (is_same<LabelStats,
                               RegressionHessianLabelStats>::value) {
    if (!selected_weights.empty()) {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/true>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels.gradient_data, selected_labels.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights,
              first_attribute_idx, effective_internal_config, constraints,
              monotonic_direction, condition, cache));

    } else {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelHessianRegressionFeatureNumericalCart<
              /*weighted=*/false>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels.gradient_data, selected_labels.hessian_data,
              na_replacement, min_num_obs, dt_config, label_stats.sum_gradient,
              label_stats.sum_hessian, label_stats.sum_weights,
              first_attribute_idx, effective_internal_config, constraints,
              monotonic_direction, condition, cache));
    }
  } else if constexpr (is_same<LabelStats, RegressionLabelStats>::value) {
    if (!selected_weights.empty()) {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/true>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, na_replacement, min_num_obs, dt_config,
              label_stats.label_distribution, first_attribute_idx,
              effective_internal_config, condition, cache));
    } else {
      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/false>(
              dense_example_idxs, selected_weights, projection_values,
              selected_labels, na_replacement, min_num_obs, dt_config,
              label_stats.label_distribution, first_attribute_idx,
              effective_internal_config, condition, cache));
    }
  } else {
    static_assert(!is_same<LabelStats, LabelStats>::value, "Not implemented.");
  }

  return result;
}

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<ClassificationLabelStats, std::vector<int32_t>>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const ClassificationLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const std::vector<int32_t>& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache,
    utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<RegressionLabelStats, std::vector<float>>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const RegressionLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const std::vector<float>& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache,
    utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
EvaluateProjection<RegressionHessianLabelStats, GradientAndHessian>(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const RegressionHessianLabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights,
    const GradientAndHessian& selected_labels,
    const absl::Span<const float> projection_values,
    const InternalTrainConfig& internal_config, const int first_attribute_idx,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache,
    utils::RandomEngine* random);

template <typename LabelStats, typename Labels>
absl::Status EvaluateProjectionAndSetCondition(
    const dataset::proto::DataSpecification& dataspec,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const absl::Span<const float> projection_values,
    const Projection& projection, const InternalTrainConfig& internal_config,
    const int first_attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  ASSIGN_OR_RETURN(
      const auto result,
      EvaluateProjection(dt_config, label_stats, dense_example_idxs,
                         selected_weights, selected_labels, projection_values,
                         internal_config, first_attribute_idx,
                         /*constraints=*/{}, /*monotonic_direction=*/0,
                         condition, cache));

  if (result == SplitSearchResult::kBetterSplitFound) {
    RETURN_IF_ERROR(SetCondition(
        projection, condition->condition().higher_condition().threshold(),
        dataspec, condition));
  }
  return absl::OkStatus();
}

template <typename LabelStats, typename Labels>
absl::Status EvaluateMHLDCandidates(
    const dataset::proto::DataSpecification& dataspec,
    const std::vector<std::vector<int>>& candidates,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const LabelStats& label_stats,
    const absl::Span<const UnsignedExampleIdx> dense_example_idxs,
    const std::vector<float>& selected_weights, const Labels& selected_labels,
    const InternalTrainConfig& internal_config,
    const ProjectionEvaluator& projection_evaluator,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<proto::NodeCondition>* conditions,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  // TODO: Multi-thread
  conditions->assign(candidates.size(), {});
  auto& projection_values = cache->projection_values;

  for (int candidate_idx = 0; candidate_idx < candidates.size();
       candidate_idx++) {
    const auto& candidate = candidates[candidate_idx];
    auto& condition = (*conditions)[candidate_idx];

    if (candidate.empty()) {
      return absl::InternalError("No candidates");
    } else if (candidate.size() == 1) {
      const auto attribute_idx = candidate.front();

      // Extract attribute value
      RETURN_IF_ERROR(projection_evaluator.ExtractAttribute(
          attribute_idx, selected_examples, &projection_values));

      RETURN_IF_ERROR(EvaluateProjectionAndSetCondition(
          dataspec, dt_config, label_stats, dense_example_idxs,
          selected_weights, selected_labels, projection_values,
          {{attribute_idx, 1.f}}, internal_config, attribute_idx, &condition,
          cache));
    } else {
      // Find best projection
      Projection projection;
      if constexpr (is_same<LabelStats, ClassificationLabelStats>::value) {
        RETURN_IF_ERROR(SolveLDA(dt_config, projection_evaluator, candidate,
                                 label_stats.num_label_classes, selected_labels,
                                 selected_weights, &projection, random));
      } else {
        return absl::InvalidArgumentError(
            "MHLD Oblique splits only available on classification. Use sparse "
            "oblique splits for other tasks.");
      }

      if (projection.empty()) {
        continue;
      }

      // Compute projection
      RETURN_IF_ERROR(projection_evaluator.Evaluate(
          projection, selected_examples, &projection_values));

      // Evaluate projection quality
      RETURN_IF_ERROR(EvaluateProjectionAndSetCondition(
          dataspec, dt_config, label_stats, dense_example_idxs,
          selected_weights, selected_labels, projection_values, projection,
          internal_config, candidate.front(), &condition, cache));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> SampleAttributes(
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::TrainingConfig& config,
    const proto::DecisionTreeTrainingConfig& dt_config,
    utils::RandomEngine* random) {
  std::vector<int> candidate_attributes{
      config_link.numerical_features().begin(),
      config_link.numerical_features().end()};

  if (dt_config.mhld_oblique_split().sample_attributes()) {
    std::shuffle(candidate_attributes.begin(), candidate_attributes.end(),
                 *random);

    const int num_attributes_to_test = NumAttributesToTest(
        dt_config, config_link.numerical_features_size(), config.task());
    if (num_attributes_to_test < 0 ||
        num_attributes_to_test > candidate_attributes.size()) {
      return absl::InternalError("Wrong number of attributes to test");
    }

    candidate_attributes.resize(num_attributes_to_test);
    std::sort(candidate_attributes.begin(), candidate_attributes.end());
  }

  return candidate_attributes;
}

template <typename LabelStats>
absl::StatusOr<bool> FindBestConditionMHLDObliqueTemplate(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  if (config_link.numerical_features().empty()) {
    return false;
  }

  ProjectionEvaluator projection_evaluator(train_dataset,
                                           config_link.numerical_features());

  // TODO: Cache.
  const auto selected_labels = ExtractLabels(label_stats, selected_examples);
  std::vector<float> selected_weights;
  if (!weights.empty()) {
    selected_weights = Extract(weights, selected_examples);
  }

  std::vector<UnsignedExampleIdx> dense_example_idxs(selected_examples.size());
  std::iota(dense_example_idxs.begin(), dense_example_idxs.end(), 0);

  std::vector<int> selected_features;
  ASSIGN_OR_RETURN(std::vector<int> candidate_features,
                   SampleAttributes(config_link, config, dt_config, random));
  std::vector<std::vector<int>> round_candidates;
  std::vector<float> round_scores;
  std::vector<proto::NodeCondition> round_conditions;

  float global_best_score = best_condition->split_score();
  bool found_better_global = false;

  const int num_rounds =
      std::min(static_cast<int>(candidate_features.size()),
               dt_config.mhld_oblique_split().max_num_attributes());

  for (int round_idx = 0; round_idx < num_rounds; round_idx++) {
    if (candidate_features.empty()) {
      // No more features to try.
      continue;
    }

    // Compute the sets of set of features to evaluate.
    round_candidates.clear();
    for (const auto candidate_feature : candidate_features) {
      round_candidates.push_back(selected_features);
      round_candidates.back().push_back(candidate_feature);
      std::sort(round_candidates.back().begin(), round_candidates.back().end());
    }

    // Evaluate
    RETURN_IF_ERROR(EvaluateMHLDCandidates(
        train_dataset.data_spec(), round_candidates, dt_config, label_stats,
        dense_example_idxs, selected_weights, selected_labels, internal_config,
        projection_evaluator, selected_examples, &round_conditions, cache,
        random));
    DCHECK_EQ(round_conditions.size(), round_candidates.size());

    // Find the best local and global projection.
    float round_best_score = 0;
    int round_best_candidate_idx = -1;
    for (int candidate_idx = 0; candidate_idx < round_candidates.size();
         candidate_idx++) {
      const float score = round_conditions[candidate_idx].split_score();

      if (std::isnan(score)) {
        continue;
      }
      if (score > round_best_score) {
        round_best_score = score;
        round_best_candidate_idx = candidate_idx;
      }
      if (score > global_best_score) {
        global_best_score = score;
        *best_condition = round_conditions[candidate_idx];
        found_better_global = true;
      }
    }

    if (round_best_candidate_idx == -1) {
      // No local improvement.
      continue;
    }

    selected_features.push_back(candidate_features[round_best_candidate_idx]);
    candidate_features.erase(candidate_features.begin() +
                             round_best_candidate_idx);
  }

  return found_better_global;
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      return FindBestConditionSparseObliqueTemplate<ClassificationLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, {}, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<ClassificationLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      return FindBestConditionSparseObliqueTemplate<
          RegressionHessianLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, constraints, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<RegressionHessianLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

absl::StatusOr<bool> FindBestConditionOblique(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats,
    const std::optional<int>& override_num_projections,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    SplitterPerThreadCache* cache) {
  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
      return FindBestConditionSparseObliqueTemplate<RegressionLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, {}, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      return FindBestConditionMHLDObliqueTemplate<RegressionLabelStats>(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, label_stats,
          override_num_projections, best_condition, random, cache);
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      return absl::InvalidArgumentError("Oblique split expected");
  }
}

namespace internal {

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
    switch (oblique_config.weights_case()) {
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kBinary): {
        weight = (weight >= 0) ? 1.f : -1.f;
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kPowerOfTwo): {
        float sign = (weight >= 0) ? 1.f : -1.f;
        int exponent =
            absl::Uniform<int>(absl::IntervalClosed, *random,
                               oblique_config.power_of_two().min_exponent(),
                               oblique_config.power_of_two().max_exponent());
        weight = sign * std::pow(2, exponent);
        break;
      }
      case (proto::DecisionTreeTrainingConfig::SparseObliqueSplit::WeightsCase::
                kInteger): {
        weight = absl::Uniform(absl::IntervalClosed, *random,
                               oblique_config.integer().minimum(),
                               oblique_config.integer().maximum());
        break;
      }
      default: {
        // Return continuous weights.
        break;
      }
    }

    if (config_link.per_columns_size() > 0 &&
        config_link.per_columns(feature).has_monotonic_constraint()) {
      const bool direction_increasing =
          config_link.per_columns(feature).monotonic_constraint().direction() ==
          model::proto::MonotonicConstraint::INCREASING;
      if (direction_increasing == (weight < 0)) {
        weight = -weight;
      }
      // As soon as one selected feature is monotonic, the oblique split
      // becomes monotonic.
      *monotonic_direction = 1;
    }

    const auto& spec = data_spec.columns(feature).numerical();
    switch (oblique_config.normalization()) {
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::NONE:
        return weight;
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
          STANDARD_DEVIATION:
        return weight / std::max(1e-6, spec.standard_deviation());
      case proto::DecisionTreeTrainingConfig::SparseObliqueSplit::MIN_MAX:
        return weight / std::max(1e-6f, spec.max_value() - spec.min_value());
    }
  };

#ifndef NDEBUG  // Keep DCHECK_EQ from for feature : features
  for (const auto feature : features) {
    DCHECK_EQ(data_spec.columns(feature).type(), dataset::proto::NUMERICAL);
  }
#endif

  // Make sure the preconditions for std::binomial_distribution are met.
  DCHECK_GE(projection_density, 0.f);
  DCHECK_LE(projection_density, 1.f);

  std::binomial_distribution<size_t> binom(features.size(), projection_density);

  // Expectation[Binomial(p,projection_density)] = num_selected_features
  const size_t num_selected_features = binom(*random);

  // TODO: Try std::bitmap
  absl::btree_set<size_t> picked_idx;

  // Floyd's sampler to select k indices uniformly
  for (size_t j = features.size() - num_selected_features; j < features.size();
       ++j) {
    size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
    if (!picked_idx.insert(t).second) picked_idx.insert(j);
  }

  projection->reserve(projection_density * features.size());
  // O(k) minimal pass to fill in those indices
  for (const auto idx : picked_idx) {
    projection->push_back({features[idx], gen_weight(features[idx])});
  }

  if (projection->empty()) {
    std::uniform_int_distribution<int> unif_feature_idx(0, features.size() - 1);
    projection->push_back(
        {/*.attribute_idx =*/features[unif_feature_idx(*random)],
         /*.weight =*/1.f});
  } else if (projection->size() == 1) {
    projection->front().weight = 1.f;
  }

  int max_num_features = dt_config.sparse_oblique_split().max_num_features();
  int cur_num_projections = projection->size();

  if (max_num_features > 0 && cur_num_projections > max_num_features) {
    internal::Projection resampled_projection;
    resampled_projection.reserve(max_num_features);
    // For a small number of features, a boolean vector is more efficient.
    // Re-evaluate if this becomes a bottleneck.
    absl::btree_set<size_t> sampled_features;
    // Floyd's sampling algorithm. TODO could reuse this
    for (size_t j = cur_num_projections - max_num_features;
         j < cur_num_projections; j++) {
      // TODO: Try uint32.
      size_t t = absl::Uniform<size_t>(*random, 0, j + 1);
      if (!sampled_features.insert(t).second) {
        // t was already sampled, so insert j instead.
        sampled_features.insert(j);
        resampled_projection.push_back((*projection)[j]);
      } else {
        // t was not yet sampled.
        resampled_projection.push_back((*projection)[t]);
      }
    }
    *projection = std::move(resampled_projection);
  }
}

absl::Status SetCondition(const Projection& projection, const float threshold,
                          const dataset::proto::DataSpecification& dataspec,
                          proto::NodeCondition* condition) {
  if (projection.empty()) {
    return absl::InternalError("Empty projection");
  }
  auto& oblique_condition =
      *condition->mutable_condition()->mutable_oblique_condition();
  oblique_condition.set_threshold(threshold);
  oblique_condition.clear_attributes();
  oblique_condition.clear_weights();
  for (const auto& item : projection) {
    oblique_condition.add_attributes(item.attribute_idx);
    oblique_condition.add_weights(item.weight);
    oblique_condition.add_na_replacements(
        dataspec.columns(item.attribute_idx).numerical().mean());
  }
  condition->set_attribute(projection.front().attribute_idx);
  condition->set_na_value(false);
  return absl::OkStatus();
}

absl::Status LDACache::ComputeClassification(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const ProjectionEvaluator& projection_evaluator,
    const std::vector<int>& selected_features, const int num_classes,
    const std::vector<int32_t>& labels, const std::vector<float>& weights,
    const bool index_features) {
  // Solve a LDA (Linear Discriminant Analysis) using the Eigenvalue
  // decomposition approach.
  //
  // Based on the section 2 of "Revisiting Classical Multiclass Linear
  // Discriminant Analysis with a Novel Prototype-based Interpretable
  // Solution".
  //
  // TODO: Experiment with other approaches. For instance, the singular
  // value decomposition approach.

  const int num_features = selected_features.size();
  size_ = num_features;

  // Compute the mean of the features (globally and per class).
  const int shifted_num_classes = num_classes - 1;  // Ignore class 0.
  DCHECK_GE(shifted_num_classes, 2);
  // TODO: Cache.
  std::vector<double> mean_per_feature(selected_features.size(), 0);
  std::vector<double> mean_per_feature_and_class(
      selected_features.size() * shifted_num_classes, 0);
  std::vector<double> weight_per_class(shifted_num_classes, 0);
  double sum_weights = 0;

  for (std::size_t example_idx = 0; example_idx < labels.size();
       example_idx++) {
    const int32_t shifted_class = labels[example_idx] - 1;
    DCHECK_GE(shifted_class, 0);
    DCHECK_LT(shifted_class, shifted_num_classes);
    const float weight = (weights.empty()) ? 1.f : weights[example_idx];

    sum_weights += weight;
    weight_per_class[shifted_class] += weight;

    for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
      const float feature_value = projection_evaluator.AttributeValue(
          selected_features[feature_idx], example_idx);

      mean_per_feature[feature_idx] += feature_value * weight;
      mean_per_feature_and_class[feature_idx + shifted_class * num_features] +=
          feature_value * weight;
    }
  }

  if (sum_weights == 0) {
    return absl::InvalidArgumentError("Null weight");
  }

  // Normalize the sums into means.
  DCHECK_GT(sum_weights, 0);
  const double inv_sum_weights = 1. / sum_weights;
  for (int feature_idx = 0; feature_idx < num_features; feature_idx++) {
    mean_per_feature[feature_idx] *= inv_sum_weights;

    for (int shifted_class = 0; shifted_class < shifted_num_classes;
         shifted_class++) {
      if (weight_per_class[shifted_class] == 0) {
        continue;
      }
      mean_per_feature_and_class[feature_idx + shifted_class * num_features] /=
          weight_per_class[shifted_class];
    }
  }

  // Compute Sb
  sb_.assign(num_features * num_features, 0);
  for (int shifted_class = 0; shifted_class < shifted_num_classes;
       shifted_class++) {
    internal::SubtractTransposeMultiplyAdd(
        weight_per_class[shifted_class],
        absl::MakeSpan(mean_per_feature_and_class)
            .subspan(shifted_class * num_features, num_features),
        absl::MakeSpan(mean_per_feature), sb_);
  }

  // Compute Sw
  sw_.assign(num_features * num_features, 0);
  for (std::size_t example_idx = 0; example_idx < labels.size();
       example_idx++) {
    const int shifted_class = labels[example_idx] - 1;
    DCHECK_GE(shifted_class, 0);
    const float weight = (weights.empty()) ? 1.f : weights[example_idx];

    internal::SubtractTransposeMultiplyAdd(
        weight, example_idx, selected_features, projection_evaluator,
        absl::MakeSpan(mean_per_feature_and_class)
            .subspan(shifted_class * num_features, num_features),
        sw_);
  }

  // Help the matrix to be invertible.
  for (int i = 0; i < num_features; i++) {
    sw_[i + num_features * i] += 0.001;
  }

  if (index_features) {
    // Index features.
    const int max_num_features =
        *std::max_element(selected_features.begin(), selected_features.end());
    feature_to_idx_.assign(max_num_features + 1, -1);
    for (int i = 0; i < num_features; i++) {
      feature_to_idx_[selected_features[i]] = i;
    }
  }

  return absl::OkStatus();
}

// Builds the feature mapping. "mapping[i]" is the index in "sw_" and "sb_" of
// feature "selected_features[i]".
absl::Status LDACache::BuildMapping(const std::vector<int>& selected_features,
                                    std::vector<int>* mapping) const {
  mapping->resize(selected_features.size());
  for (size_t i = 0; i < selected_features.size(); i++) {
    const int j = feature_to_idx_[selected_features[i]];
    if (j == -1) {
      return absl::InternalError("Non indexed feature");
    }
    (*mapping)[i] = j;
  }
  return absl::OkStatus();
}

absl::Status LDACache::Extract(const std::vector<int>& selected_features,
                               const std::vector<double>& in,
                               std::vector<double>* out) const {
  // TODO: Cache.
  std::vector<int> mapping;
  RETURN_IF_ERROR(BuildMapping(selected_features, &mapping));
  const int num_features = selected_features.size();

  out->resize(num_features * num_features);
  for (int col = 0; col < num_features; col++) {
    for (int row = 0; row < num_features; row++) {
      (*out)[row + col * num_features] =
          in[mapping[row] + size_ * mapping[col]];
    }
  }
  return absl::OkStatus();
}

absl::Status LDACache::GetSB(const std::vector<int>& selected_features,
                             std::vector<double>* out) const {
  return Extract(selected_features, sb_, out);
}

absl::Status LDACache::GetSW(const std::vector<int>& selected_features,
                             std::vector<double>* out) const {
  return Extract(selected_features, sw_, out);
}

size_t RowMajorMaxRows() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_RM_MAX_ROWS");
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : std::numeric_limits<size_t>::max();
  }();
  return value;
}

#ifdef SUBTREE_GATHER_CACHE
namespace {
// Per-thread cap on gathered-column bytes per block (and, via the 2x sweep
// gate, on bytes retained as capacity across blocks).
size_t SubtreeGatherBudgetBytes() {
  static const size_t value = [] {
    const char* e = std::getenv("YDF_SG_BUDGET_MB");
    const size_t mb = e != nullptr ? std::strtoull(e, nullptr, 10) : 1024;
    return mb * 1024 * 1024;
  }();
  return value;
}
}  // namespace

bool PrepareSubtreeGatherNode(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const size_t num_rows_in_dataset, SubtreeGatherCache* sg) {
  const size_t n = selected_examples.size();
  if (n == 0 || n > RowMajorMaxRows() || n > SubtreeGatherCache::kSlotMask) {
    return false;
  }
  if (sg->slot_of_example.size() != num_rows_in_dataset) {
    sg->slot_of_example.assign(num_rows_in_dataset, 0);
    sg->epoch = 0;
  }
  sg->node_slots.resize(n);

  // Reuse the current block if every row of this node belongs to it. Rows of
  // a node are either all inside the block (descendant of the block node) or
  // — being a tree partition — disjoint from it, but the per-row check also
  // keeps interleavings across trees / BFS orders correct.
  if (sg->epoch != 0) {
    const uint32_t tag = sg->epoch << SubtreeGatherCache::kSlotBits;
    bool valid = true;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t entry = sg->slot_of_example[selected_examples[i]];
      if ((entry & ~SubtreeGatherCache::kSlotMask) != tag) {
        valid = false;
        break;
      }
      sg->node_slots[i] = entry & SubtreeGatherCache::kSlotMask;
    }
    if (valid) {
      return true;
    }
  }

  // Materialize a new block from this node's examples.
  if (sg->epoch >= SubtreeGatherCache::kMaxEpoch) {
    // Epoch tag space exhausted: stale entries could alias the wrapped tag.
    std::fill(sg->slot_of_example.begin(), sg->slot_of_example.end(), 0u);
    sg->epoch = 0;
  }
  ++sg->epoch;
  const uint32_t tag = sg->epoch << SubtreeGatherCache::kSlotBits;
  sg->block_rows.assign(selected_examples.begin(), selected_examples.end());
  for (size_t i = 0; i < n; ++i) {
    sg->slot_of_example[selected_examples[i]] =
        tag | static_cast<uint32_t>(i);
    sg->node_slots[i] = static_cast<uint32_t>(i);
  }
  sg->gathered_bytes = 0;
  return true;
}

const float* ProjectionEvaluator::GatheredColumn(const int attribute_idx,
                                                 SubtreeGatherCache* sg) const {
  if (sg->cols.size() <= static_cast<size_t>(attribute_idx)) {
    sg->cols.resize(attribute_idx + 1);
    sg->col_epoch.resize(attribute_idx + 1, 0);
  }
  auto& col = sg->cols[attribute_idx];
  if (sg->col_epoch[attribute_idx] == sg->epoch && !col.empty()) {
    return col.data();
  }
  const size_t n = sg->block_rows.size();
  const size_t bytes = n * sizeof(float);
  if (sg->gathered_bytes + bytes > SubtreeGatherBudgetBytes()) {
    return nullptr;
  }
  // Capacity persists across blocks so the frequent case (same features
  // re-gathered every block) does not churn the allocator. When the feature
  // space is so large that retained capacity piles up (ultra-wide datasets),
  // sweep stale columns before growing further.
  if (sg->retained_bytes + bytes > 2 * SubtreeGatherBudgetBytes()) {
    for (size_t a = 0; a < sg->cols.size(); ++a) {
      if (sg->col_epoch[a] != sg->epoch && !sg->cols[a].empty()) {
        sg->retained_bytes -= sg->cols[a].capacity() * sizeof(float);
        std::vector<float>().swap(sg->cols[a]);
      }
    }
    if (sg->retained_bytes + bytes > 2 * SubtreeGatherBudgetBytes()) {
      return nullptr;
    }
  }
  const uint64_t capacity_before = col.capacity() * sizeof(float);
  col.resize(n);
  sg->retained_bytes += col.capacity() * sizeof(float) - capacity_before;
  const float* src = numerical_attribute_data_[attribute_idx];
  float* dst = col.data();
  for (size_t i = 0; i < n; ++i) {
    dst[i] = src[sg->block_rows[i]];
  }
#ifdef ENABLE_APPLYPROJECTION_ISNAN
  const float na_replacement = na_replacement_value_[attribute_idx];
  for (size_t i = 0; i < n; ++i) {
    if (std::isnan(dst[i])) dst[i] = na_replacement;
  }
#endif
  sg->col_epoch[attribute_idx] = sg->epoch;
  sg->gathered_bytes += bytes;
  return col.data();
}

YDF_PROJECTION_EVALUATE_NOINLINE
absl::Status ProjectionEvaluator::EvaluateWithSubtreeCache(
    const Projection& projection,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    SubtreeGatherCache* sg, std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);

  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
  const size_t n = selected_examples.size();
  DCHECK_EQ(n, sg->node_slots.size());
  values->resize(n);

  // Resolve each projection item to its gathered column, or to the raw
  // column store when the budget is exhausted.
  struct ResolvedItem {
    const float* gathered;  // nullptr => read "raw" by example index.
    const float* raw;
    float weight;
    float na_replacement;
  };
  // Projections are tiny (~density items); a fixed-size stack array would be
  // wrong for dense oblique configs, so keep a small vector.
  static thread_local std::vector<ResolvedItem> items;
  items.clear();
  items.reserve(projection.size());
  for (const auto& item : projection) {
    DCHECK_LT(item.attribute_idx, numerical_attributes_.size());
    DCHECK_GE(item.attribute_idx, 0);
    const float* gathered = GatheredColumn(item.attribute_idx, sg);
    items.push_back({gathered, numerical_attribute_data_[item.attribute_idx],
                     item.weight, na_replacement_value_[item.attribute_idx]});
  }

  // Same loop shape as Evaluate (rows outer, items inner, scalar float
  // accumulator) so the two paths keep identical summation order and the
  // split decisions stay bit-identical.
  const uint32_t* slots = sg->node_slots.data();
  for (size_t selected_idx = 0; selected_idx < n; selected_idx++) {
    const uint32_t slot = slots[selected_idx];
    float value = 0;
    for (const auto& item : items) {
      float attribute_value;
      if (item.gathered != nullptr) {
        attribute_value = item.gathered[slot];
      } else {
        attribute_value = item.raw[selected_examples[selected_idx]];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
        // Gathered columns are NaN-replaced at gather time; only the raw
        // fallback needs the per-lookup check.
        if (std::isnan(attribute_value)) {
          attribute_value = item.na_replacement;
        }
#endif
      }
      value += attribute_value * item.weight;
    }
    (*values)[selected_idx] = value;
  }
  return absl::OkStatus();
}
#endif  // SUBTREE_GATHER_CACHE

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
  // Dual bf16 layout: both half-width stores live; per-node kernels pick the
  // order, AttributeValue defaults to the column store.
  const auto* bf16_rows = dataset::Bf16RowMajorFeatureMatrix::Active();
  const auto* bf16_cols = dataset::Bf16FlatColMajorFeatureMatrix::Active();
  if (bf16_rows != nullptr || bf16_cols != nullptr) {
    if (bf16_rows != nullptr) {
      DCHECK_EQ(static_cast<int64_t>(train_dataset.nrow()),
                bf16_rows->num_rows());
      bf16_row_major_matrix_ = bf16_rows;
    }
    if (bf16_cols != nullptr) {
      DCHECK_EQ(static_cast<int64_t>(train_dataset.nrow()),
                bf16_cols->num_rows());
      bf16_flat_col_matrix_ = bf16_cols;
    }
    return;
  }

  // Dual fp32 layout: same dispatch idea as dual bf16, full precision. Either
  // store alone also lands here (single-layout experiments).
  const auto* row_active = dataset::RowMajorFeatureMatrix::Active();
  const auto* flat_active_rm = dataset::FlatColMajorFeatureMatrix::Active();
  if (row_active != nullptr || flat_active_rm != nullptr) {
    if (row_active != nullptr) {
      DCHECK_EQ(static_cast<int64_t>(train_dataset.nrow()),
                row_active->num_rows());
      row_major_matrix_ = row_active;
    }
    if (flat_active_rm != nullptr) {
      DCHECK_EQ(static_cast<int64_t>(train_dataset.nrow()),
                flat_active_rm->num_rows());
      flat_col_matrix_ = flat_active_rm;
    }
    return;
  }
#endif

#if defined(FLAT_COL_DATASET_LAYOUT)
  const auto* flat_active = dataset::FlatColMajorFeatureMatrix::Active();
  if (flat_active != nullptr) {
    DCHECK_EQ(static_cast<int64_t>(train_dataset.nrow()),
              flat_active->num_rows());
    flat_col_matrix_ = flat_active;
    return;
  }
#endif

  for (const auto attribute_idx : numerical_features) {
    const auto column_or = train_dataset.ColumnWithCastWithStatus<
        dataset::VerticalDataset::NumericalColumn>(attribute_idx);
    constructor_status_.Update(column_or.status());
    if (!constructor_status_.ok()) {
      break;
    }

    numerical_attributes_[attribute_idx] = &column_or.value()->values();
    numerical_attribute_data_[attribute_idx] =
        column_or.value()->values().data();
  }
}

YDF_PROJECTION_EVALUATE_NOINLINE
absl::Status ProjectionEvaluator::Evaluate(
    const Projection& projection,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);

  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
  values->resize(selected_examples.size());

#if defined(ROW_MAJOR_DATASET_LAYOUT)
  // Dynamic_Row_Col_Major on the DFS/nodewise path: when both fp32 stores are
  // live, pick per node with the same YDF_RM_MAX_ROWS threshold as the BFS
  // (DEPTHWISE_1_PASS) kernel. Without this, AttributeValue's static
  // store priority would read the column store for every node. Both branches
  // keep the generic loop shape below (only the store is fixed), so timings
  // stay comparable with the single-layout 'row' / 'flat_column' runs.
  if (row_major_matrix_ != nullptr && flat_col_matrix_ != nullptr) {
    const bool use_row_major = selected_examples.size() <= RowMajorMaxRows();
    for (size_t selected_idx = 0; selected_idx < selected_examples.size();
         selected_idx++) {
      float value = 0;
      const auto example_idx = selected_examples[selected_idx];
      for (const auto& item : projection) {
        float attribute_value =
            use_row_major
                ? row_major_matrix_->Get(example_idx, item.attribute_idx)
                : flat_col_matrix_->Get(example_idx, item.attribute_idx);
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
#endif

#ifdef EVALUATE_4ROW_KERNEL
  // V2-rev3's inner kernel at per-(node, projection) granularity: process
  // the selected rows in blocks of 4 with 4 independent accumulators and 4
  // parallel column loads per projection item. Each item's loads are random
  // DRAM gathers; the stock loop chains them through one accumulator (one
  // miss in flight per item), the 4-row block exposes 4. Per-row item order
  // is unchanged, so the sums are bit-identical to the generic loop below.
  // Only the plain column-store path is unrolled; the experimental layout
  // configs keep their own dispatch above.
  {
    struct ItemRef {
      const float* col;
      float weight;
#ifdef ENABLE_APPLYPROJECTION_ISNAN
      float na;
#endif
    };
    static thread_local std::vector<ItemRef> item_refs;
    item_refs.clear();
    item_refs.reserve(projection.size());
    bool all_direct = true;
    for (const auto& item : projection) {
      DCHECK_LT(item.attribute_idx, numerical_attributes_.size());
      DCHECK_GE(item.attribute_idx, 0);
      const float* col = numerical_attribute_data_[item.attribute_idx];
      if (col == nullptr) {
        all_direct = false;
        break;
      }
      item_refs.push_back({col, item.weight
#ifdef ENABLE_APPLYPROJECTION_ISNAN
                           ,
                           na_replacement_value_[item.attribute_idx]
#endif
      });
    }
    if (all_direct) {
      const UnsignedExampleIdx* sel = selected_examples.data();
      const size_t n = selected_examples.size();
      float* out = values->data();
      size_t i = 0;
#if defined(EVALUATE_ROW_BLOCK_8)
      for (; i + 8 <= n; i += 8) {
        const UnsignedExampleIdx e0 = sel[i], e1 = sel[i + 1], e2 = sel[i + 2],
                                 e3 = sel[i + 3], e4 = sel[i + 4],
                                 e5 = sel[i + 5], e6 = sel[i + 6],
                                 e7 = sel[i + 7];
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f,
              a6 = 0.f, a7 = 0.f;
        for (const auto& ref : item_refs) {
          float v0 = ref.col[e0];
          float v1 = ref.col[e1];
          float v2 = ref.col[e2];
          float v3 = ref.col[e3];
          float v4 = ref.col[e4];
          float v5 = ref.col[e5];
          float v6 = ref.col[e6];
          float v7 = ref.col[e7];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
          if (std::isnan(v0)) v0 = ref.na;
          if (std::isnan(v1)) v1 = ref.na;
          if (std::isnan(v2)) v2 = ref.na;
          if (std::isnan(v3)) v3 = ref.na;
          if (std::isnan(v4)) v4 = ref.na;
          if (std::isnan(v5)) v5 = ref.na;
          if (std::isnan(v6)) v6 = ref.na;
          if (std::isnan(v7)) v7 = ref.na;
#endif
          a0 += v0 * ref.weight;
          a1 += v1 * ref.weight;
          a2 += v2 * ref.weight;
          a3 += v3 * ref.weight;
          a4 += v4 * ref.weight;
          a5 += v5 * ref.weight;
          a6 += v6 * ref.weight;
          a7 += v7 * ref.weight;
        }
        out[i] = a0;
        out[i + 1] = a1;
        out[i + 2] = a2;
        out[i + 3] = a3;
        out[i + 4] = a4;
        out[i + 5] = a5;
        out[i + 6] = a6;
        out[i + 7] = a7;
      }
#else
      for (; i + 4 <= n; i += 4) {
        const UnsignedExampleIdx e0 = sel[i], e1 = sel[i + 1], e2 = sel[i + 2],
                                 e3 = sel[i + 3];
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (const auto& ref : item_refs) {
          float v0 = ref.col[e0];
          float v1 = ref.col[e1];
          float v2 = ref.col[e2];
          float v3 = ref.col[e3];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
          if (std::isnan(v0)) v0 = ref.na;
          if (std::isnan(v1)) v1 = ref.na;
          if (std::isnan(v2)) v2 = ref.na;
          if (std::isnan(v3)) v3 = ref.na;
#endif
          a0 += v0 * ref.weight;
          a1 += v1 * ref.weight;
          a2 += v2 * ref.weight;
          a3 += v3 * ref.weight;
        }
        out[i] = a0;
        out[i + 1] = a1;
        out[i + 2] = a2;
        out[i + 3] = a3;
      }
#endif  // EVALUATE_ROW_BLOCK_8
      for (; i < n; ++i) {
        float value = 0;
        for (const auto& ref : item_refs) {
          float v = ref.col[sel[i]];
#ifdef ENABLE_APPLYPROJECTION_ISNAN
          if (std::isnan(v)) v = ref.na;
#endif
          value += v * ref.weight;
        }
        out[i] = value;
      }
      return absl::OkStatus();
    }
  }
#endif  // EVALUATE_4ROW_KERNEL

  for (size_t selected_idx = 0; selected_idx < selected_examples.size();
       selected_idx++) {
    float value = 0;
    const auto example_idx = selected_examples[selected_idx];
    for (const auto& item : projection) {
      DCHECK_LT(item.attribute_idx, numerical_attributes_.size());
      DCHECK_GE(item.attribute_idx, 0);

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

absl::Status ProjectionEvaluator::ExtractAttribute(
    const int attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    std::vector<float>* values) const {
  RETURN_IF_ERROR(constructor_status_);
  values->resize(selected_examples.size());
  const float na_replacement_value = na_replacement_value_[attribute_idx];
  for (size_t selected_idx = 0; selected_idx < selected_examples.size();
       selected_idx++) {
    const auto example_idx = selected_examples[selected_idx];
    float value = AttributeValue(attribute_idx, example_idx);
    if (std::isnan(value)) {
      value = na_replacement_value;
    }
    (*values)[selected_idx] = value;
  }
  return absl::OkStatus();
}

void SubtractTransposeMultiplyAdd(double weight, absl::Span<double> a,
                                  absl::Span<double> b,
                                  std::vector<double>& output) {
  DCHECK_EQ(a.size(), b.size());
  DCHECK_EQ(b.size() * b.size(), output.size());

  const int n = a.size();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      output[j + i * n] += weight * (a[i] - b[i]) * (a[j] - b[j]);
    }
  }
}

void SubtractTransposeMultiplyAdd(
    double weight, std::size_t example_idx,
    const std::vector<int>& selected_features,
    const ProjectionEvaluator& projection_evaluator, absl::Span<double> b,
    std::vector<double>& output) {
  DCHECK_EQ(selected_features.size(), b.size());
  DCHECK_EQ(b.size() * b.size(), output.size());

  const int n = b.size();
  for (int i = 0; i < n; i++) {
    const double x_i = projection_evaluator.AttributeValue(
        selected_features[i], example_idx);
    for (int j = 0; j < n; j++) {
      const double x_j = projection_evaluator.AttributeValue(
          selected_features[j], example_idx);
      output[j + i * n] += weight * (x_i - b[i]) * (x_j - b[j]);
    }
  }
}

}  // namespace internal
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
