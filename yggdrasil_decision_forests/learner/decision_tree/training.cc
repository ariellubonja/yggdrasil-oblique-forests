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

#include "yggdrasil_decision_forests/learner/decision_tree/training.h"

#if defined(__x86_64__) && !defined(DISABLE_STD_UPPER_BOUND_VECTORIZATION)
#define SIMD_UPPER_BOUND 1
#include <immintrin.h>
#endif

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef CALLGRIND_DEPTH
#include <cstdio>
#include <valgrind/callgrind.h>
#endif

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
// Per-depth trace of the column-overlap gate (DW1_HOT_OVERLAP_STATS).
#include <cstdio>
#endif

#ifdef LINECOUNT_A
// Method A: exact distinct-64B-line counter for the DW1 gather col[sel_ptr[i]].
// Per depth, sums rows and distinct lines (sel is sorted, so counting sel[i]>>4
// transitions is exact) => useful/line. Dumped to $LINECOUNT_OUT + stderr.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
namespace {
struct Dw1LineCountAccum {
  std::mutex mu;
  std::vector<double> rows;   // indexed by tree depth
  std::vector<double> lines;
  std::vector<double> nodes;
  void Add(int depth, double n_nodes, double r, double l) {
    std::lock_guard<std::mutex> lk(mu);
    if (depth < 0) return;
    if (static_cast<int>(rows.size()) <= depth) {
      rows.resize(depth + 1, 0.0);
      lines.resize(depth + 1, 0.0);
      nodes.resize(depth + 1, 0.0);
    }
    rows[depth] += r;
    lines[depth] += l;
    nodes[depth] += n_nodes;
  }
  ~Dw1LineCountAccum() {
    const char* out = std::getenv("LINECOUNT_OUT");
    std::FILE* f = (out != nullptr) ? std::fopen(out, "w") : nullptr;
    std::fprintf(stderr,
                 "\n=== Method A: DW1 gather useful-floats per 64B line by "
                 "depth ===\ndepth,nodes,rows,lines,useful_per_line\n");
    if (f) std::fprintf(f, "depth,nodes,rows,lines,useful_per_line\n");
    for (size_t d = 0; d < rows.size(); ++d) {
      if (nodes[d] == 0.0) continue;
      const double upl = lines[d] > 0.0 ? rows[d] / lines[d] : 0.0;
      std::fprintf(stderr, "%zu,%.0f,%.0f,%.0f,%.4f\n", d, nodes[d], rows[d],
                   lines[d], upl);
      if (f)
        std::fprintf(f, "%zu,%.0f,%.0f,%.0f,%.4f\n", d, nodes[d], rows[d],
                     lines[d], upl);
    }
    if (f) std::fclose(f);
  }
};
Dw1LineCountAccum g_dw1_linecount;
}  // namespace
#endif  // LINECOUNT_A

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/label.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_depthwise_1pass.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_cpu_symmetric_optimized.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/uplift.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/decision_tree/vector_sequence.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/utils/cast.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {

namespace {

// Generates a failure absl status if the configuration contains monotonic
// constraints.
absl::Status FailIfMonotonic(
    const model::proto::TrainingConfigLinking& config_link,
    const int attribute_idx, const NodeConstraints& constraints,
    const absl::string_view why) {
  if (config_link.per_columns_size() > 0 &&
      (config_link.per_columns(attribute_idx).has_monotonic_constraint() ||
       constraints.min_max_output.has_value())) {
    return absl::InternalError(
        absl::StrCat("Monotonic constraints not supported for ", why));
  }
  return absl::OkStatus();
}

// Number of trials to run when learning a categorical split with randomly
// generated masks.
//
// Args:
//   config: A random categorical split learning configuration.
//
// Returns:
//   A function active_dictionary_size -> number_of_trials.
//
// The "active_dictionary_size" is the number of unique categorical values that
// are present at least once in the training examples of the node.
std::function<int(const int active_dictionary_size)>
NumTrialsForRandomCategoricalSplit(const proto::Categorical::Random& config) {
  const auto num_trial_exponent = config.num_trial_exponent();
  const auto max_num_trials = config.max_num_trials();
  return
      [num_trial_exponent, max_num_trials](const int active_dictionary_size) {
        const int num_trials =
            32 + std::pow(active_dictionary_size, num_trial_exponent);
        return std::min(num_trials, max_num_trials);
      };
}

// Helper function to set a condition statistics. Do not set the following
// fields: "mutable_condition" and "na_value".
void SetConditionHelper(
    const double information_gain, const int32_t attribute_idx,
    const utils::BinaryToIntegerConfusionMatrixDouble& running_confusion,
    const utils::BinaryToIntegerConfusionMatrixInt64&
        running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(information_gain);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Helper function to set a condition statistics.
// Similar as "SetConditionHelper" above, but for a regression problem.
void SetConditionHelper(
    const double variance_reduction, const int32_t attribute_idx,
    const utils::BinaryToNormalDistributionDouble& running_confusion,
    const utils::BinaryToNormalDistributionDouble& running_confusion_no_weights,
    proto::NodeCondition* condition) {
  condition->set_split_score(variance_reduction);
  condition->set_attribute(attribute_idx);
  condition->set_num_pos_training_examples_without_weight(
      running_confusion_no_weights.pos().NumObservations());
  condition->set_num_pos_training_examples_with_weight(
      running_confusion.pos().NumObservations());
  condition->set_num_training_examples_without_weight(
      running_confusion_no_weights.NumObservations());
  condition->set_num_training_examples_with_weight(
      running_confusion.NumObservations());
}

// Computes and set in "na_replacement" the value to use as a replacement of
// missing values when the "local imputation" strategy is used.
//
// Explanation: The "local imputation" strategy to handle missing values
// consists in replacing these missing values by the mean of the feature in the
// training dataset.
//
// If the feature only contains missing values, the "na_replacement" argument is
// left unchanged.
//
// `weights` may be empty which is equivalent to unit weights.
void LocalImputationForNumericalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    float* na_replacement) {
  double na_replacement_value_accumulator = 0;
  double na_replacement_weight_accumulator = 0;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    if (!std::isnan(attribute)) {
      na_replacement_value_accumulator += attribute * weight;
      na_replacement_weight_accumulator += weight;
    }
  }
  if (na_replacement_weight_accumulator > 0) {
    *na_replacement = static_cast<float>(na_replacement_value_accumulator /
                                         na_replacement_weight_accumulator);
  }
}

// Similar as "LocalImputationForNumericalAttribute", but for a categorical
// attribute. Return the most frequent attribute value.
void LocalImputationForCategoricalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const int32_t num_attribute_classes, int32_t* na_replacement) {
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(num_attribute_classes);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value !=
        dataset::VerticalDataset::CategoricalColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Similar to "LocalImputationForCategoricalAttribute", but for a boolean
// attribute. Returns the most frequent attribute value.
void LocalImputationForBooleanAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    bool* na_replacement) {
  DCHECK(!weights.empty());
  utils::IntegerDistributionDouble attribute_distribution;
  attribute_distribution.SetNumClasses(2);
  for (const auto example_idx : selected_examples) {
    const auto attribute_value = attributes[example_idx];
    if (attribute_value != dataset::VerticalDataset::BooleanColumn::kNaValue) {
      const float weight = weights.empty() ? 1.f : weights[example_idx];
      attribute_distribution.Add(attribute_value, weight);
    }
  }
  if (attribute_distribution.NumObservations() > 0) {
    *na_replacement = attribute_distribution.TopClass();
  }
}

// Return the minimum and maximum values of a numerical attribute.
// Return false if there is no min-max e.g. selected_examples is empty or all
// the values are NAs.
bool MinMaxNumericalAttribute(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const absl::Span<const float> attributes, float* min_value,
    float* max_value) {
  float local_min_value = 0;
  float local_max_value = 0;
  bool first = true;
  for (const auto example_idx : selected_examples) {
    const float attribute = attributes[example_idx];
    if (first && !std::isnan(attribute)) {
      local_max_value = local_min_value = attribute;
      first = false;
    } else if (attribute > local_max_value) {
      local_max_value = attribute;
    } else if (attribute < local_min_value) {
      local_min_value = attribute;
    }
  }
  *min_value = local_min_value;
  *max_value = local_max_value;
  return !first;
}

// For each dictionary item of a Categorical Set attribute, computes the label
// distribution of examples containing this item. This distribution is
// equivalent as the label distribution of the positive branch splitted on a
// categorical-set condition with mask equal to {item}.
template <bool weighted>
std::vector<utils::BinaryToNormalDistributionDouble>
InitializeRegressionAttributeDistributions(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& labels, const std::vector<float>& weights,
    const std::vector<std::pair<size_t, size_t>>& attribute_values,
    const std::vector<int>& attribute_bank,
    const utils::NormalDistributionDouble& label_distribution,
    const std::vector<bool>& candidate_attributes_bitmap) {
  if constexpr (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }
  const int num_attribute_classes = candidate_attributes_bitmap.size();

  // Initialize all candidates with the full distribution in the negative side.
  std::vector<utils::BinaryToNormalDistributionDouble> attribute_distributions(
      num_attribute_classes,
      utils::BinaryToNormalDistributionDouble(label_distribution, {}));

  // For every attribute value, push all examples containing this value to the
  // positive side of the corresponding distribution.
  for (const auto example_idx : selected_examples) {
    const float label = labels[example_idx];
    const auto& example_attrs_range = attribute_values[example_idx];

    // Iterate through attributes present in this example
    for (auto bank_idx = example_attrs_range.first;
         bank_idx < example_attrs_range.second; ++bank_idx) {
      const int candidate_attr_value = attribute_bank[bank_idx];
      if (!candidate_attributes_bitmap[candidate_attr_value]) {
        continue;
      }
      // This attribute is a candidate, update its stats
      auto& dist = attribute_distributions[candidate_attr_value];
      // Move example contribution from neg to pos.
      if constexpr (weighted) {
        const auto weight = weights[example_idx];
        dist.mutable_pos()->Add(label, weight);
        dist.mutable_neg()->Sub(label, weight);
      } else {
        dist.mutable_pos()->Add(label);
        dist.mutable_neg()->Sub(label);
      }
    }
  }

  return attribute_distributions;
}

// For each dictionary item of a Categorical Set attribute, computes the label
// distribution of examples containing this item. This distribution is
// equivalent as the label distribution of the positive branch splitted on a
// categorical-set condition with mask equal to {item}.
template <bool weighted>
std::vector<utils::BinaryToIntegerConfusionMatrixDouble>
InitializeClassificationAttributeDistributions(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<int32_t>& labels, const std::vector<float>& weights,
    const std::vector<std::pair<size_t, size_t>>& attribute_values,
    const std::vector<int32_t>& attribute_bank,
    const utils::IntegerDistributionDouble& label_distribution,
    const std::vector<bool>& candidate_attributes_bitmap) {
  if constexpr (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }
  const int num_attribute_classes = candidate_attributes_bitmap.size();

  // Initialize all candidates with the full distribution in the negative side.
  std::vector<utils::BinaryToIntegerConfusionMatrixDouble>
      attribute_distributions(
          num_attribute_classes,
          utils::BinaryToIntegerConfusionMatrixDouble(
              label_distribution, utils::IntegerDistributionDouble(
                                      label_distribution.NumClasses())));

  // For every attribute value, push all examples containing this value to the
  // positive side of the corresponding distribution.
  for (const auto example_idx : selected_examples) {
    const int32_t label = labels[example_idx];
    const auto& example_attrs_range = attribute_values[example_idx];

    // Iterate through attributes present in this example
    for (auto bank_idx = example_attrs_range.first;
         bank_idx < example_attrs_range.second; ++bank_idx) {
      const int32_t candidate_attr_value = attribute_bank[bank_idx];
      if (!candidate_attributes_bitmap[candidate_attr_value]) {
        continue;
      }
      // This attribute is a candidate, update its stats
      auto& dist = attribute_distributions[candidate_attr_value];
      // Move example contribution from neg to pos.
      if constexpr (weighted) {
        const auto weight = weights[example_idx];
        dist.mutable_pos()->Add(label, weight);
        dist.mutable_neg()->Sub(label, weight);
      } else {
        dist.mutable_pos()->Add(label);
        dist.mutable_neg()->Sub(label);
      }
    }
  }

  return attribute_distributions;
}

// Select which sorting strategy to use effectively.
//
// If the strategy is AUTO or PRESORTED, the fastest / selected strategy depends
// on the data.
proto::DecisionTreeTrainingConfig::Internal::SortingStrategy EffectiveStrategy(
    const proto::DecisionTreeTrainingConfig& dt_config,
    const int64_t num_selected_examples,
    const InternalTrainConfig& internal_config) {
  proto::DecisionTreeTrainingConfig::Internal::SortingStrategy strategy;

  if (internal_config.override_sorting_strategy.has_value()) {
    // The internal configuration configured by the learning algorithm takes
    // precedence on the sorting strategy.
    strategy = internal_config.override_sorting_strategy.value();
  } else {
    // Otherwise, the training configuration (controlled by the user or
    // unit-test controller) selects the strategy.
    strategy = dt_config.internal().sorting_strategy();
  }
  switch (strategy) {
    // User specified strategy.
    case proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED:
    case proto::DecisionTreeTrainingConfig::Internal::IN_NODE:
      return strategy;

    case proto::DecisionTreeTrainingConfig::Internal::AUTO:
      CHECK(false);  // The AUTO strategy should have been resolved before.
      break;
    case proto::DecisionTreeTrainingConfig::Internal::PRESORTED: {
      DCHECK(internal_config.preprocessing);
      const auto num_total_examples =
          internal_config.preprocessing->num_examples();
      const float ratio =
          static_cast<float>(num_selected_examples) / num_total_examples;
      return (num_selected_examples >= 25 && ratio >= 0.125)
                 ? proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED
                 : proto::DecisionTreeTrainingConfig::Internal::IN_NODE;
    }
  };
}

}  // namespace

// Specialization in the case of classification.
absl::StatusOr<SplitSearchResult> FindBestConditionClassification(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const ClassificationLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "classification"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      CHRONO_BEGIN_COARSE(column_with_cast);
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      CHRONO_END_COARSE(
          column_with_cast,
          ::yggdrasil_decision_forests::chrono_prof::kColumnWithCast);

      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelClassificationFeatureNumericalCart(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, label_stats.num_label_classes,
                        na_replacement, min_num_obs, dt_config,
                        label_stats.label_distribution, attribute_idx,
                        internal_config, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelClassificationFeatureNumericalHistogram(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, label_stats.num_label_classes,
                        na_replacement, min_num_obs, dt_config,
                        label_stats.label_distribution, attribute_idx, random,
                        best_condition));
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::DiscretizedNumericalColumn>(
              attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
                      selected_examples, weights, attribute_data->values(),
                      num_bins, label_stats.label_data,
                      label_stats.num_label_classes, na_replacement_index,
                      min_num_obs, dt_config, label_stats.label_distribution,
                      attribute_idx, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureCategorical(
                      selected_examples, weights, attribute_data->values(),
                      label_stats.label_data, num_attribute_classes,
                      label_stats.num_label_classes, na_replacement,
                      min_num_obs, dt_config, label_stats.label_distribution,
                      attribute_idx, random, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalSetColumn>(attribute_idx));
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelClassificationFeatureCategoricalSetGreedyForward<
                /*weighted=*/false>(selected_examples, weights, *attribute_data,
                                    label_stats.label_data,
                                    num_attribute_classes,
                                    label_stats.num_label_classes, min_num_obs,
                                    dt_config, label_stats.label_distribution,
                                    attribute_idx, best_condition, random));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelClassificationFeatureCategoricalSetGreedyForward<
                /*weighted=*/true>(selected_examples, weights, *attribute_data,
                                   label_stats.label_data,
                                   num_attribute_classes,
                                   label_stats.num_label_classes, min_num_obs,
                                   dt_config, label_stats.label_distribution,
                                   attribute_idx, best_condition, random));
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      ASSIGN_OR_RETURN(
          result, FindSplitLabelClassificationFeatureBoolean(
                      selected_examples, weights, attribute_data->values(),
                      label_stats.label_data, label_stats.num_label_classes,
                      na_replacement, min_num_obs, dt_config,
                      label_stats.label_distribution, attribute_idx,
                      best_condition, cache));
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(
          result, FindSplitAnyLabelFeatureNumericalVectorSequence(
                      model::proto::Task::CLASSIFICATION, selected_examples,
                      weights, *attribute_data, attribute_column_spec,
                      label_stats, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    ASSIGN_OR_RETURN(
        const auto na_result,
        FindSplitLabelClassificationFeatureNA(
            selected_examples, weights, train_dataset.column(attribute_idx),
            label_stats.label_data, label_stats.num_label_classes, min_num_obs,
            dt_config, label_stats.label_distribution, attribute_idx,
            best_condition, cache));
    result = std::min(result, na_result);
  }

  return result;
}

absl::StatusOr<SplitSearchResult> FindBestConditionRegressionHessianGain(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionHessianLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  const int8_t monotonic_direction =
      MonotonicConstraintSign(config_link, attribute_idx);

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelHessianRegressionFeatureNumericalCart<
                  /*weighted=*/false>(
                  selected_examples, weights, attribute_data->values(),
                  label_stats.gradient_data, label_stats.hessian_data,
                  na_replacement, min_num_obs, dt_config,
                  label_stats.sum_gradient, label_stats.sum_hessian,
                  label_stats.sum_weights, attribute_idx, internal_config,
                  constraints, monotonic_direction, best_condition, cache));
        } else {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelHessianRegressionFeatureNumericalCart<
                  /*weighted=*/true>(
                  selected_examples, weights, attribute_data->values(),
                  label_stats.gradient_data, label_stats.hessian_data,
                  na_replacement, min_num_obs, dt_config,
                  label_stats.sum_gradient, label_stats.sum_hessian,
                  label_stats.sum_weights, attribute_idx, internal_config,
                  constraints, monotonic_direction, best_condition, cache));
        }
      } else {
        return absl::InvalidArgumentError(
            "Only split exact implemented for hessian gains.");
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::DiscretizedNumericalColumn>(
              attribute_idx));

      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
                /*weighted=*/false>(
                selected_examples, weights, attribute_data->values(), num_bins,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement_index, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, monotonic_direction, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<
                /*weighted=*/true>(
                selected_examples, weights, attribute_data->values(), num_bins,
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement_index, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, monotonic_direction, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::CategoricalColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureCategorical<
                /*weighted=*/false>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache, random));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureCategorical<
                /*weighted=*/true>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                num_attribute_classes, na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache, random));
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto& attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));

      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/false>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelHessianRegressionFeatureBoolean</*weighted=*/true>(
                selected_examples, weights, attribute_data->values(),
                label_stats.gradient_data, label_stats.hessian_data,
                na_replacement, min_num_obs, dt_config,
                label_stats.sum_gradient, label_stats.sum_hessian,
                label_stats.sum_weights, attribute_idx, internal_config,
                constraints, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(result,
                       FindSplitAnyLabelFeatureNumericalVectorSequence(
                           model::proto::Task::REGRESSION, selected_examples,
                           weights, *attribute_data, attribute_column_spec,
                           label_stats, min_num_obs, dt_config, attribute_idx,
                           internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache));
      result = std::min(result, na_result);
    } else {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelHessianRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.gradient_data, label_stats.hessian_data, min_num_obs,
              dt_config, label_stats.sum_gradient, label_stats.sum_hessian,
              label_stats.sum_weights, attribute_idx, internal_config,
              constraints, best_condition, cache));
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of regression.
absl::StatusOr<SplitSearchResult> FindBestConditionRegression(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const RegressionLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  if (dt_config.internal().generate_fake_error_in_splitter()) {
    return absl::InternalError("Fake error");
  }

  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;

  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  SplitSearchResult result;

  RETURN_IF_ERROR(
      FailIfMonotonic(config_link, attribute_idx, constraints, "regression"));

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      if (dt_config.numerical_split().type() == proto::NumericalSplit::EXACT) {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/false>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache));
        } else {
          ASSIGN_OR_RETURN(
              result,
              FindSplitLabelRegressionFeatureNumericalCart</*weighted=*/true>(
                  selected_examples, weights, attribute_data,
                  label_stats.label_data, na_replacement, min_num_obs,
                  dt_config, label_stats.label_distribution, attribute_idx,
                  internal_config, best_condition, cache));
        }
      } else {
        if (weights.empty()) {
          ASSIGN_OR_RETURN(
              result, FindSplitLabelRegressionFeatureNumericalHistogram<
                          /*weighted=*/false>(
                          selected_examples, weights, attribute_data,
                          label_stats.label_data, na_replacement, min_num_obs,
                          dt_config, label_stats.label_distribution,
                          attribute_idx, random, best_condition));
        } else {
          ASSIGN_OR_RETURN(
              result, FindSplitLabelRegressionFeatureNumericalHistogram<
                          /*weighted=*/true>(
                          selected_examples, weights, attribute_data,
                          label_stats.label_data, na_replacement, min_num_obs,
                          dt_config, label_stats.label_distribution,
                          attribute_idx, random, best_condition));
        }
      }
    } break;

    case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
      if (!dt_config.has_axis_aligned_split()) {
        return SplitSearchResult::kNoBetterSplitFound;
      }

      // Condition of the type "Attr >= threshold".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::DiscretizedNumericalColumn>(
                  attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();
      const auto num_bins =
          attribute_column_spec.discretized_numerical().boundaries_size() + 1;
      const auto na_replacement_index =
          dataset::NumericalToDiscretizedNumerical(attribute_column_spec,
                                                   na_replacement);
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
                        /*weighted=*/false>(
                        selected_examples, weights, attribute_data, num_bins,
                        label_stats.label_data, na_replacement_index,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureDiscretizedNumericalCart<
                        /*weighted=*/true>(
                        selected_examples, weights, attribute_data, num_bins,
                        label_stats.label_data, na_replacement_index,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      // Condition of the type "Attr \in X".
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelRegressionFeatureCategorical</*weighted=*/false>(
                selected_examples, weights, attribute_data,
                label_stats.label_data, num_attribute_classes, na_replacement,
                min_num_obs, dt_config, label_stats.label_distribution,
                attribute_idx, best_condition, cache, random));
      } else {
        ASSIGN_OR_RETURN(
            result,
            FindSplitLabelRegressionFeatureCategorical</*weighted=*/true>(
                selected_examples, weights, attribute_data,
                label_stats.label_data, num_attribute_classes, na_replacement,
                min_num_obs, dt_config, label_stats.label_distribution,
                attribute_idx, best_condition, cache, random));
      }
    } break;

    case dataset::proto::ColumnType::CATEGORICAL_SET: {
      const auto* attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalSetColumn>(attribute_idx)
              .value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
                        /*weighted=*/false>(
                        selected_examples, weights, *attribute_data,
                        label_stats.label_data, num_attribute_classes,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, random));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<
                        /*weighted=*/true>(
                        selected_examples, weights, *attribute_data,
                        label_stats.label_data, num_attribute_classes,
                        min_num_obs, dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, random));
      }
    } break;

    case dataset::proto::ColumnType::BOOLEAN: {
      // Condition of the type "Attr is True".
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::BooleanColumn>(attribute_idx));
      const auto na_replacement =
          attribute_column_spec.boolean().count_true() >=
          attribute_column_spec.boolean().count_false();
      if (weights.empty()) {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureBoolean</*weighted=*/false>(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, na_replacement, min_num_obs,
                        dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      } else {
        ASSIGN_OR_RETURN(
            result, FindSplitLabelRegressionFeatureBoolean</*weighted=*/true>(
                        selected_examples, weights, attribute_data->values(),
                        label_stats.label_data, na_replacement, min_num_obs,
                        dt_config, label_stats.label_distribution,
                        attribute_idx, best_condition, cache));
      }
    } break;

    case dataset::proto::ColumnType::NUMERICAL_VECTOR_SEQUENCE: {
      ASSIGN_OR_RETURN(
          const auto* attribute_data,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalVectorSequenceColumn>(
              attribute_idx));
      ASSIGN_OR_RETURN(result,
                       FindSplitAnyLabelFeatureNumericalVectorSequence(
                           model::proto::Task::REGRESSION, selected_examples,
                           weights, *attribute_data, attribute_column_spec,
                           label_stats, min_num_obs, dt_config, attribute_idx,
                           internal_config, best_condition, random, cache));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    if (weights.empty()) {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelRegressionFeatureNA</*weighted=*/false>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache));
      result = std::min(result, na_result);
    } else {
      ASSIGN_OR_RETURN(
          const auto na_result,
          FindSplitLabelRegressionFeatureNA</*weighted=*/true>(
              selected_examples, weights, train_dataset.column(attribute_idx),
              label_stats.label_data, min_num_obs, dt_config,
              label_stats.label_distribution, attribute_idx, best_condition,
              cache));
      result = std::min(result, na_result);
    }
  }

  return result;
}

// Specialization in the case of uplift with categorical outcome.
absl::StatusOr<SplitSearchResult> FindBestConditionUpliftCategorical(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const CategoricalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "categorical uplift"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      ASSIGN_OR_RETURN(
          result, FindSplitLabelUpliftCategoricalFeatureNumericalCart(
                      selected_examples, weights, attribute_data, label_stats,
                      na_replacement, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::CategoricalColumn>(attribute_idx)
              .value()
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelUpliftCategoricalFeatureCategorical(
              selected_examples, weights, attribute_data, label_stats,
              num_attribute_classes, na_replacement, min_num_obs, dt_config,
              attribute_idx, internal_config, best_condition, cache, random));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    return absl::InvalidArgumentError("allow_na_conditions not supported");
  }

  return result;
}

// Specialization in the case of uplift with numerical outcome.
absl::StatusOr<SplitSearchResult> FindBestConditionUpliftNumerical(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NumericalUpliftLabelStats& label_stats, const int32_t attribute_idx,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, SplitterPerThreadCache* cache) {
  const int min_num_obs =
      dt_config.in_split_min_examples_check() ? dt_config.min_examples() : 1;
  const auto& attribute_column_spec =
      train_dataset.data_spec().columns(attribute_idx);

  RETURN_IF_ERROR(FailIfMonotonic(config_link, attribute_idx, constraints,
                                  "numerical uplift"));

  SplitSearchResult result;

  switch (train_dataset.column(attribute_idx)->type()) {
    case dataset::proto::ColumnType::NUMERICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement = attribute_column_spec.numerical().mean();

      ASSIGN_OR_RETURN(
          result, FindSplitLabelUpliftNumericalFeatureNumericalCart(
                      selected_examples, weights, attribute_data, label_stats,
                      na_replacement, min_num_obs, dt_config, attribute_idx,
                      internal_config, best_condition, cache));
    } break;

    case dataset::proto::ColumnType::CATEGORICAL: {
      const auto& attribute_data =
          train_dataset
              .ColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
                  attribute_idx)
              ->values();
      const auto na_replacement =
          attribute_column_spec.categorical().most_frequent_value();
      const auto num_attribute_classes =
          attribute_column_spec.categorical().number_of_unique_values();

      ASSIGN_OR_RETURN(
          result,
          FindSplitLabelUpliftNumericalFeatureCategorical(
              selected_examples, weights, attribute_data, label_stats,
              num_attribute_classes, na_replacement, min_num_obs, dt_config,
              attribute_idx, internal_config, best_condition, cache, random));
    } break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          dataset::proto::ColumnType_Name(
              train_dataset.column(attribute_idx)->type()),
          " attribute ", train_dataset.column(attribute_idx)->name(),
          " is not supported."));
  }

  // Condition of the type "Attr is NA".
  if (dt_config.allow_na_conditions()) {
    return absl::InvalidArgumentError("allow_na_conditions not supported");
  }

  return result;
}

absl::StatusOr<SplitterWorkResponse> FindBestConditionFromSplitterWorkRequest(
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const InternalTrainConfig& internal_config,
    const SplitterWorkRequest& request) {
  SplitterWorkResponse response;
  response.manager_data = request.manager_data;
  request.splitter_cache->random.seed(request.seed);

  response.condition = absl::make_unique<proto::NodeCondition>();
  response.condition->set_split_score(request.best_score);

  if (request.num_oblique_projections_to_run != -1) {
    // Worker-side CPU time of an oblique job: workers are outside any TreeScope,
    // so this lands in global_stats, summed over workers rather than wall-time.
    // Encloses the oblique scopes, so their coverage in a job is checkable.
    CHRONO_SCOPE_COARSE(
        ::yggdrasil_decision_forests::chrono_prof::kSplitWorkerOblique);
    DCHECK_EQ(request.attribute_idx, -1);
    ASSIGN_OR_RETURN(
        const auto found_oblique_condition,
        FindBestConditionOblique(
            request.common->train_dataset, request.common->selected_examples,
            weights, config, config_link, dt_config, request.common->parent,
            internal_config, request.common->label_stats,
            request.num_oblique_projections_to_run, request.common->constraints,
            response.condition.get(), &request.splitter_cache->random,
            request.splitter_cache));

    // An oblique split cannot be invalid.
    response.status = found_oblique_condition
                          ? SplitSearchResult::kBetterSplitFound
                          : SplitSearchResult::kNoBetterSplitFound;
    return response;
  }

  if (request.attribute_idx < 0 ||
      request.attribute_idx >=
          request.common->train_dataset.data_spec().columns_size()) {
    return absl::OutOfRangeError(absl::StrCat(
        "The attribute index is out of bounds - attribute_idx: ",
        request.attribute_idx, ", columns in the dataset: ",
        request.common->train_dataset.data_spec().columns_size(), "."));
  }

  // Worker-side CPU time of an axis-aligned job. Oblique GBT runs still schedule
  // these (one splitter per candidate attribute) and they compete for the same
  // workers, so they are measured separately. Same global_stats caveat as above.
  CHRONO_SCOPE_COARSE(
      ::yggdrasil_decision_forests::chrono_prof::kSplitWorkerAxisAligned);

  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& label_stats =
          utils::down_cast<const ClassificationLabelStats&>(
              request.common->label_stats);

      ASSIGN_OR_RETURN(
          response.status,
          FindBestConditionClassification(
              request.common->train_dataset, request.common->selected_examples,
              weights, config, config_link, dt_config, request.common->parent,
              internal_config, label_stats, request.attribute_idx,
              request.common->constraints, response.condition.get(),
              &request.splitter_cache->random, request.splitter_cache));
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(
                request.common->label_stats);

        ASSIGN_OR_RETURN(
            response.status,
            FindBestConditionRegressionHessianGain(
                request.common->train_dataset,
                request.common->selected_examples, weights, config, config_link,
                dt_config, request.common->parent, internal_config, label_stats,
                request.attribute_idx, request.common->constraints,
                response.condition.get(), &request.splitter_cache->random,
                request.splitter_cache));

      } else {
        const auto& label_stats = utils::down_cast<const RegressionLabelStats&>(
            request.common->label_stats);

        ASSIGN_OR_RETURN(
            response.status,
            FindBestConditionRegression(
                request.common->train_dataset,
                request.common->selected_examples, weights, config, config_link,
                dt_config, request.common->parent, internal_config, label_stats,
                request.attribute_idx, request.common->constraints,
                response.condition.get(), &request.splitter_cache->random,
                request.splitter_cache));
      }
      break;
    default:
      NOTREACHED();
  }

  return response;
}

absl::StatusOr<bool> FindBestConditionOblique(
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
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      const auto& class_label_stats =
          utils::down_cast<const ClassificationLabelStats&>(label_stats);
      return FindBestConditionOblique(
          train_dataset, selected_examples, weights, config, config_link,
          dt_config, parent, internal_config, class_label_stats,
          override_num_projections, best_condition, random, cache);
    } break;
    case model::proto::Task::REGRESSION:
      if (internal_config.hessian_score) {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionHessianLabelStats&>(label_stats);
        return FindBestConditionOblique(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, reg_label_stats,
            override_num_projections, constraints, best_condition, random,
            cache);
      } else {
        const auto& reg_label_stats =
            utils::down_cast<const RegressionLabelStats&>(label_stats);
        return FindBestConditionOblique(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, reg_label_stats,
            override_num_projections, best_condition, random, cache);
      }
      break;
    default:
      return absl::UnimplementedError(
          "Oblique splits not implemented for this task");
  }

  return false;
}

absl::StatusOr<bool> FindBestConditionSingleThreadManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  // Single Thread Setup.
  cache->splitter_cache_list.resize(1);

  // Was a least one good split found?
  bool found_good_condition = false;

  switch (dt_config.split_axis_case()) {
    case proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET:
    case proto::DecisionTreeTrainingConfig::kAxisAlignedSplit:
      // Nothing to do.
      break;
    case proto::DecisionTreeTrainingConfig::kSparseObliqueSplit:
    case proto::DecisionTreeTrainingConfig::kMhldObliqueSplit:
      {
        CHRONO_SCOPE_COARSE(
            ::yggdrasil_decision_forests::chrono_prof::kObliqueSplitSearch);
        ASSIGN_OR_RETURN(
            found_good_condition,
            FindBestConditionOblique(
                train_dataset, selected_examples, weights, config, config_link,
                dt_config, parent, internal_config, label_stats, {},
                constraints, best_condition, random,
                &cache->splitter_cache_list[0]));
      }
      break;
  }

  // Get the indices of the attributes to test.
  int remaining_attributes_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;

  {
    CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kGetCandidateAttributes);

    // "candidate_attributes" persists across nodes (per-tree cache), so rebuild
    // only when fresh: the loop below shuffles each position lazily as it reads
    // it, costing O(tested) RNG draws, not O(F). Accuracy-, not bit-equivalent.
    if (candidate_attributes.size() !=
        static_cast<size_t>(config_link.features_size())) {
      candidate_attributes.assign(config_link.features().begin(),
                                  config_link.features().end());
    }
    remaining_attributes_to_test = NumAttributesToTest(
        dt_config, candidate_attributes.size(), config.task());

    // Index of the next attribute to be tested in "candidate_attributes".
    int candidate_attribute_idx_in_candidate_list = 0;

    while (remaining_attributes_to_test >= 0 &&
             candidate_attribute_idx_in_candidate_list <
                 candidate_attributes.size()) {
      {
        // Fisher-Yates step for the position about to be read (each position
        // is read at most once per node, so this yields a uniform prefix).
        const size_t pos = candidate_attribute_idx_in_candidate_list;
        std::uniform_int_distribution<size_t> swap_with(
            pos, candidate_attributes.size() - 1);
        std::swap(candidate_attributes[pos],
                  candidate_attributes[swap_with(*random)]);
      }
      // Get the attribute data.
      const int32_t attribute_idx =
          candidate_attributes[candidate_attribute_idx_in_candidate_list++];
      SplitSearchResult result;

      switch (config.task()) {
        case model::proto::Task::CLASSIFICATION: {
          const auto& class_label_stats =
              utils::down_cast<const ClassificationLabelStats&>(label_stats);

          ASSIGN_OR_RETURN(
              result, FindBestConditionClassification(
                          train_dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          class_label_stats, attribute_idx, constraints,
                          best_condition, random,
                          &cache->splitter_cache_list[0]));
        } break;
        case model::proto::Task::REGRESSION:
          if (internal_config.hessian_score) {
            const auto& reg_label_stats =
                utils::down_cast<const RegressionHessianLabelStats&>(
                    label_stats);

            ASSIGN_OR_RETURN(
                result,
                FindBestConditionRegressionHessianGain(
                    train_dataset, selected_examples, weights, config,
                    config_link, dt_config, parent, internal_config,
                    reg_label_stats, attribute_idx, constraints, best_condition,
                    random, &cache->splitter_cache_list[0]));

          } else {
            const auto& reg_label_stats =
                utils::down_cast<const RegressionLabelStats&>(label_stats);

            ASSIGN_OR_RETURN(
                result,
                FindBestConditionRegression(
                    train_dataset, selected_examples, weights, config,
                    config_link, dt_config, parent, internal_config,
                    reg_label_stats, attribute_idx, constraints, best_condition,
                    random, &cache->splitter_cache_list[0]));
          }
          break;

        case model::proto::Task::CATEGORICAL_UPLIFT: {
          const auto& uplift_label_stats =
              utils::down_cast<const CategoricalUpliftLabelStats&>(label_stats);
          ASSIGN_OR_RETURN(
              result, FindBestConditionUpliftCategorical(
                          train_dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          uplift_label_stats, attribute_idx, constraints,
                          best_condition, random,
                          &cache->splitter_cache_list[0]));
        } break;

        case model::proto::Task::NUMERICAL_UPLIFT: {
          const auto& uplift_label_stats =
              utils::down_cast<const NumericalUpliftLabelStats&>(label_stats);
          ASSIGN_OR_RETURN(
              result, FindBestConditionUpliftNumerical(
                          train_dataset, selected_examples, weights, config,
                          config_link, dt_config, parent, internal_config,
                          uplift_label_stats, attribute_idx, constraints,
                          best_condition, random,
                          &cache->splitter_cache_list[0]));
        } break;

        default:
          return absl::UnimplementedError("Non implemented");
      }
      if (result != SplitSearchResult::kInvalidAttribute) {
        remaining_attributes_to_test--;
        if (result == SplitSearchResult::kBetterSplitFound) {
          found_good_condition = true;
        }
      }
    }
  }

  return found_good_condition;
}

absl::StatusOr<bool> FindBestConditionConcurrentManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  // This method looks for the best split using worker threads.
  //
  // Background:
  // The best split is the split with the best score among the first
  // "min_num_jobs_to_test" valid split searches. The order of the split is
  // defined by the list of candidate attributes generated by
  // GetCandidateAttributes. A split search is valid if it tested at least one
  // split.
  //
  // Since the execution is multi-threaded, splits are computed/evaluated in
  // unpredictable order. However, this method's result is as if the splits were
  // evaluated sequentially according to the order defined by
  // GetCandidateAttributes. Oblique splits are always evaluated (if requested
  // by the user).
  //
  // A work unit (called a "job") is the evaluation of a single attribute or, in
  // the case of oblique splits, the evaluation of a given number of random
  // projections. A job with a given idx can be in one of multiple states (in
  // chronological order):
  //
  // 1. Before being scheduled (idx >= next_job_to_schedule).
  // 2. Scheduled and being computed by a worker.
  // 3. The worker is done with the computation, and the result was recorded by
  // the manager (cache->durable_response_list[idx].set).
  // 4. The result was processed by the manager (idx < next_job_to_process).
  //
  // This method guarantees that the jobs/splits are processed in order.
  //
  // Note that next_job_to_process < next_job_to_schedule always holds.
  if (internal_config.split_finder_processor == nullptr) {
    return absl::InternalError(
        "Multi-threaded execution requested but no worker threads created.");
  }
  const int num_threads = internal_config.split_finder_processor->num_threads();
  DCHECK_GT(num_threads, 0);

  if (config_link.features().empty()) {
    return false;
  }

  // Manager-side setup: request struct, job accounting, cache resizes and
  // GetCandidateAttributes, ending just before the first Submit. Manual
  // begin/end because the locals declared here must outlive the scope.
  CHRONO_BEGIN_COARSE(split_mgr_setup);

  // Constant and static part of the requests.
  SplitterWorkRequestCommon common{
      .train_dataset = train_dataset,
      .selected_examples = selected_examples,
      .parent = parent,
      .label_stats = label_stats,
      .constraints = constraints,
      .weights = weights,
      .config = config,
      .config_link = config_link,
      .dt_config = dt_config,
      .internal_config = internal_config,
  };

  // Computes the number of oblique projections to evaluate and how to group
  // them into requests.
  int num_oblique_jobs = 0;
  int num_oblique_projections;
  int num_oblique_projections_per_oblique_job;

  if (config_link.numerical_features_size() > 0) {
    if (dt_config.split_axis_case() ==
        proto::DecisionTreeTrainingConfig::kSparseObliqueSplit) {
      num_oblique_projections =
          GetNumProjections(dt_config, config_link.numerical_features_size());

      if (num_oblique_projections > 0) {
        // Arbitrary minimum number of oblique projections to test in each job.
        // Because oblique jobs are expensive (more than non oblique jobs), it
        // is not efficient to create a request with too little work to do.
        //
        // In most real cases, this parameter does not matter as the limit is
        // effectively constraint by the number of threads.
        const int min_projections_per_request = 10;

        DCHECK_GE(num_threads, 1);
        num_oblique_jobs = std::min(
            num_threads,
            (num_oblique_projections + min_projections_per_request - 1) /
                min_projections_per_request);
        num_oblique_projections_per_oblique_job =
            (num_oblique_projections + num_oblique_jobs - 1) / num_oblique_jobs;
      }
    } else if (dt_config.split_axis_case() ==
               proto::DecisionTreeTrainingConfig::kMhldObliqueSplit) {
      num_oblique_projections = 1;
      num_oblique_projections_per_oblique_job = 1;
      num_oblique_jobs = 1;
    }
  }

  // Prepare caches.
  cache->splitter_cache_list.resize(num_threads);

  // Get the ordered indices of the attributes to test.
  int min_num_jobs_to_test;
  std::vector<int32_t>& candidate_attributes = cache->candidate_attributes;
  {
    // Nested inside kSplitManagerSetup (the single-thread manager has the same
    // scope at its own GetCandidateAttributes call).
    CHRONO_SCOPE_COARSE(
        ::yggdrasil_decision_forests::chrono_prof::kGetCandidateAttributes);
    GetCandidateAttributes(config, config_link, dt_config,
                           &min_num_jobs_to_test, &candidate_attributes,
                           random);
  }

  const int num_jobs = candidate_attributes.size() + num_oblique_jobs;
  // All the oblique jobs need to be done.
  // Note: When do look for oblique splits, we also run the classical numerical
  // splitter.
  min_num_jobs_to_test += num_oblique_jobs;

  cache->durable_response_list.resize(num_jobs);

  // Marks all the caches "available".
  cache->available_cache_idxs.resize(cache->splitter_cache_list.size());
  std::iota(cache->available_cache_idxs.begin(),
            cache->available_cache_idxs.end(), 0);

  // Marks all the duration responses as "non set".
  for (auto& s : cache->durable_response_list) {
    s.set = false;
  }

  CHRONO_END_COARSE(split_mgr_setup,
             ::yggdrasil_decision_forests::chrono_prof::kSplitManagerSetup);

  // Score and value of the best found condition.
  std::atomic<float> best_split_score = best_condition->split_score();
  std::unique_ptr<proto::NodeCondition> best_condition_ptr;

  // Get Channel readers and writers.
  auto& processor = *(internal_config.split_finder_processor);

  // Number of jobs currently scheduled.
  int num_in_flight = 0;

  // Helper function to create a WorkRequest.
  //
  // If attribute_idx is != -1 create a request for an axis-aligned split.
  //
  // If attribute_idx is == -1 and num_oblique_projections_to_run != -1, create
  // a request for an oblique split.
  //
  auto build_request =
      [&](const int job_idx, const int attribute_idx,
          const int num_oblique_projections_to_run) -> SplitterWorkRequest {
    DCHECK_NE(attribute_idx != -1, num_oblique_projections_to_run != -1);
    DCHECK(!cache->available_cache_idxs.empty());
    const int32_t cache_idx = cache->available_cache_idxs.back();
    cache->available_cache_idxs.pop_back();
    num_in_flight++;
    return SplitterWorkRequest(
        /*manager_data=*/
        {
            .cache_idx = cache_idx,
            .job_idx = job_idx,
        },
        /*best_score=*/best_split_score,
        /*attribute_idx=*/attribute_idx,
        /*splitter_cache=*/&cache->splitter_cache_list[cache_idx],
        /*common=*/&common,
        /*seed=*/(*random)(),
        /*num_oblique_projections_to_run=*/num_oblique_projections_to_run);
  };

  int next_job_to_schedule = 0;

  // Initial scheduling: all the oblique jobs, then axis-aligned jobs while
  // worker threads remain. Manual begin/end: next_job_to_schedule outlives it.
  CHRONO_BEGIN_COARSE(split_mgr_submit);

  // Schedule all the oblique jobs.
  for (int oblique_job_idx = 0; oblique_job_idx < num_oblique_jobs;
       oblique_job_idx++) {
    int num_projections_in_request;
    if (oblique_job_idx == num_oblique_jobs - 1) {
      num_projections_in_request =
          num_oblique_projections -
          oblique_job_idx * num_oblique_projections_per_oblique_job;
    } else {
      num_projections_in_request = num_oblique_projections_per_oblique_job;
    }

    processor.Submit(build_request(
        next_job_to_schedule++,
        /*attribute_idx=*/-1,
        /*num_oblique_projections_to_run=*/num_projections_in_request));
  }

  // Schedule some non-oblique jobs if threads are still available.
  while (next_job_to_schedule < std::min(num_threads, num_jobs) &&
         !cache->available_cache_idxs.empty()) {
    DCHECK_GE(next_job_to_schedule, num_oblique_jobs);
    const int attribute_idx =
        candidate_attributes[next_job_to_schedule - num_oblique_jobs];

    processor.Submit(build_request(next_job_to_schedule,
                                   /*attribute_idx=*/attribute_idx,
                                   /*num_oblique_projections_to_run=*/-1));
    next_job_to_schedule++;
  }

  CHRONO_END_COARSE(split_mgr_submit,
             ::yggdrasil_decision_forests::chrono_prof::kSplitManagerSubmit);

  int num_valid_job_tested = 0;
  int next_job_to_process = 0;

  absl::Status status;

  while (true) {
    // Get a new result from a worker splitter. Blocking: this is the manager's
    // "workers busy" wall-time.
    auto maybe_response = [&] {
      CHRONO_SCOPE_COARSE(
          ::yggdrasil_decision_forests::chrono_prof::kSplitManagerWait);
      return processor.GetResult();
    }();
    if (!maybe_response.has_value()) {
      break;
    }

    num_in_flight--;
    DCHECK_GE(num_in_flight, 0);

    // Recording the response + processing the ones that became processable.
    // Closes before the scheduling loop below so Submit is not nested in it.
    {
      CHRONO_SCOPE_COARSE(
          ::yggdrasil_decision_forests::chrono_prof::kSplitManagerProcess);

      {
        // Record, but do not process, the worker response.
        auto response_or = std::move(maybe_response).value();
        if (!response_or.ok()) {
          status.Update(response_or.status());
          break;
        }
        auto response = std::move(response_or).value();
        // Release the cache immediately to be reused by other workers.
        cache->available_cache_idxs.push_back(response.manager_data.cache_idx);

        // Record response for further processing.
        auto& durable_response =
            cache->durable_response_list[response.manager_data.job_idx];
        durable_response.status = response.status;
        durable_response.set = true;
        if (response.status == SplitSearchResult::kBetterSplitFound) {
          // The worker found a potentially better solution.
          durable_response.condition = std::move(response.condition);
        }
      }

      // Process new responses that can be processed.
      while (next_job_to_process < next_job_to_schedule &&
             num_valid_job_tested < min_num_jobs_to_test &&
             cache->durable_response_list[next_job_to_process].set) {
        // Something to process.
        auto durable_response =
            &cache->durable_response_list[next_job_to_process];
        next_job_to_process++;

        if (durable_response->status != SplitSearchResult::kInvalidAttribute) {
          num_valid_job_tested++;
        }
        if (durable_response->status == SplitSearchResult::kBetterSplitFound) {
          const float split_score = durable_response->condition->split_score();
          if (split_score > best_split_score) {
            best_condition_ptr = std::move(durable_response->condition);
            best_split_score = split_score;
          }
        }
      }
    }

    if (num_valid_job_tested >= min_num_jobs_to_test) {
      // Enough jobs have been tested to take a decision.
      break;
    }

    if (next_job_to_process >= num_jobs) {
      // We have processed all the jobs.
      break;
    }

    // Schedule the testing of more conditions.
    {
      CHRONO_SCOPE_COARSE(
          ::yggdrasil_decision_forests::chrono_prof::kSplitManagerSubmit);
      while (!cache->available_cache_idxs.empty() &&
             next_job_to_schedule < num_jobs) {
        processor.Submit(build_request(
            next_job_to_schedule,
            /*attribute_idx=*/
            candidate_attributes[next_job_to_schedule - num_oblique_jobs],
            /*num_oblique_projections_to_run=*/-1));
        next_job_to_schedule++;
      }
    }
  }

  // Drain the response channel. The blocking part counts as Wait like the main
  // loop's GetResult.
  for (int i = 0; i < num_in_flight; i++) {
    auto maybe_response = [&] {
      CHRONO_SCOPE_COARSE(
          ::yggdrasil_decision_forests::chrono_prof::kSplitManagerWait);
      return processor.GetResult();
    }();
    if (!maybe_response.has_value()) {
      // The channel was closed.
      break;
    }
    auto response_or = std::move(maybe_response).value();
    status.Update(response_or.status());
  }

  // Move the random generator state to make the behavior deterministic.
  random->discard(num_jobs - next_job_to_schedule);

  if (!status.ok()) {
    return status;
  }

  if (best_condition_ptr) {
    *best_condition = std::move(*best_condition_ptr);
    return true;
  }
  return false;
}

absl::StatusOr<bool> FindBestConditionManager(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const LabelStats& label_stats, const NodeConstraints& constraints,
    proto::NodeCondition* best_condition, utils::RandomEngine* random,
    PerThreadCache* cache) {
  if (internal_config.split_finder_processor != nullptr) {
    return FindBestConditionConcurrentManager(
        train_dataset, selected_examples, weights, config, config_link,
        dt_config, parent, internal_config, label_stats, constraints,
        best_condition, random, cache);
  }

  // Single thread.
  return FindBestConditionSingleThreadManager(
      train_dataset, selected_examples, weights, config, config_link, dt_config,
      parent, internal_config, label_stats, constraints, best_condition, random,
      cache);
}

absl::StatusOr<bool> FindBestCondition(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const proto::Node& parent, const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* best_condition,
    utils::RandomEngine* random, PerThreadCache* cache) {
  switch (config.task()) {
    case model::proto::Task::CLASSIFICATION: {
      STATUS_CHECK(!internal_config.hessian_score);
      ASSIGN_OR_RETURN(const auto labels,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.label()));
      ClassificationLabelStats label_stat(labels->values());

      const auto& label_column_spec =
          train_dataset.data_spec().columns(config_link.label());
      label_stat.num_label_classes =
          label_column_spec.categorical().number_of_unique_values();

      label_stat.label_distribution.Load(parent.classifier().distribution());

      if (label_stat.label_distribution.NumClasses() >= 1 &&
          label_stat.label_distribution.count(
              dataset::kOutOfDictionaryItemIndex) > 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The categorical training label column \"", config.label(),
            "\" contains out-of-dictionary values. This is not allowed. "
            "Most likely, the subset of the training dataset used to compute "
            "the label dictionary did not contain some label values found in "
            "the remaining of the training dataset. To solve this issue, "
            "increase the number of training examples used to compute "
            "dictionnaries with "
            "`max_num_scanned_rows_to_compute_statistics=100000` (or a larger "
            "value). To use all the training examples to build the dictionary, "
            "use `max_num_scanned_rows_to_compute_statistics=-1` (Warning: "
            "this can be slow on large datasets)."));
      }

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    case model::proto::Task::REGRESSION: {
      if (internal_config.hessian_score) {
        STATUS_CHECK_NE(internal_config.gradient_col_idx, -1);
        STATUS_CHECK_NE(internal_config.hessian_col_idx, -1);
        STATUS_CHECK_EQ(internal_config.gradient_col_idx, config_link.label());

        ASSIGN_OR_RETURN(const auto gradients,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             internal_config.gradient_col_idx));
        ASSIGN_OR_RETURN(const auto hessians,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             internal_config.hessian_col_idx));

        RegressionHessianLabelStats label_stat(gradients->values(),
                                               hessians->values());

        STATUS_CHECK(parent.regressor().has_sum_gradients());
        label_stat.sum_gradient = parent.regressor().sum_gradients();
        label_stat.sum_hessian = parent.regressor().sum_hessians();
        label_stat.sum_weights = parent.regressor().sum_weights();

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, label_stat, constraints,
            best_condition, random, cache);
      } else {
        ASSIGN_OR_RETURN(const auto labels,
                         train_dataset.ColumnWithCastWithStatus<
                             dataset::VerticalDataset::NumericalColumn>(
                             config_link.label()));
        RegressionLabelStats label_stat(labels->values());

        STATUS_CHECK(parent.regressor().has_distribution());
        label_stat.label_distribution.Load(parent.regressor().distribution());

        return FindBestConditionManager(
            train_dataset, selected_examples, weights, config, config_link,
            dt_config, parent, internal_config, label_stat, constraints,
            best_condition, random, cache);
      }
    } break;

    case model::proto::Task::CATEGORICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& outcome_spec =
          train_dataset.data_spec().columns(config_link.label());
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      ASSIGN_OR_RETURN(const auto labels,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.label()));

      ASSIGN_OR_RETURN(const auto treatments,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.uplift_treatment()));

      CategoricalUpliftLabelStats label_stat(
          labels->values(),
          outcome_spec.categorical().number_of_unique_values(),
          treatments->values(),
          treatment_spec.categorical().number_of_unique_values());

      UpliftLeafToLabelDist(parent.uplift(), &label_stat.label_distribution);

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    case model::proto::Task::NUMERICAL_UPLIFT: {
      STATUS_CHECK(!internal_config.hessian_score);
      const auto& treatment_spec =
          train_dataset.data_spec().columns(config_link.uplift_treatment());

      ASSIGN_OR_RETURN(
          const auto labels,
          train_dataset.ColumnWithCastWithStatus<
              dataset::VerticalDataset::NumericalColumn>(config_link.label()));

      ASSIGN_OR_RETURN(const auto treatments,
                       train_dataset.ColumnWithCastWithStatus<
                           dataset::VerticalDataset::CategoricalColumn>(
                           config_link.uplift_treatment()));

      NumericalUpliftLabelStats label_stat(
          labels->values(), treatments->values(),
          treatment_spec.categorical().number_of_unique_values());

      UpliftLeafToLabelDist(parent.uplift(), &label_stat.label_distribution);

      return FindBestConditionManager(train_dataset, selected_examples, weights,
                                      config, config_link, dt_config, parent,
                                      internal_config, label_stat, constraints,
                                      best_condition, random, cache);
    } break;

    default:
      return absl::UnimplementedError("Non implemented");
  }
  return false;
}

// Returns the index k of the last equal-width threshold <= a
// (or –1 when a is smaller than the first threshold).
static inline int EqualWidthThresholdIndex(const float attribute,
                                           const float min_value,
                                           const float max_value,
                                           const int num_splits) {
  if (num_splits <= 0) return -1;

  const float range = max_value - min_value;
  if (range <= 0.0f) return -1;

  // Fast bucketing via Bin Width arithmetic
  const float width = range / static_cast<float>(num_splits);
  const float x = (attribute - min_value) / width - 0.5f;
  int idx = static_cast<int>(floorf(x));

  // Clamp to the nominal range (with "below first threshold" as -1)
  if (idx < 0) return -1;
  if (idx >= num_splits) idx = num_splits - 1;

  // Sometimes above is off-by-one vs. std::upper_bound due to floating point
  // arithmetic Below's a 1-step correction to match std::upper_bound() on the
  // actual thresholds. Compute thresholds using the exact same arithmetic as in
  // GenHistogramBins: T[j] = min_value + (range * (j + 0.5f)) / num_splits;
  const float Nf = static_cast<float>(num_splits);
  const float jf = static_cast<float>(idx);
  const float Tj = min_value + (range * (jf + 0.5f)) / Nf;

  if (attribute <
      Tj) {  // attribute falls before this bin's threshold: move left by one
    --idx;
    return (idx >= 0) ? idx : -1;
  }

  if (idx + 1 < num_splits) {
    // Next threshold: j+1 -> (j + 1.5f)
    const float Tnext = min_value + (range * (jf + 1.5f)) / Nf;
    if (attribute >= Tnext) {
      // a reaches past next threshold; move right by one
      ++idx;
    }
  }
  return idx;
}

/* #region Histogram binning: attribute value -> candidate-split bin index */
// "Attribute value -> bin index" mapper for both numerical histogram finders:
// Init() picks equal-width closed form / AVX2 8x8 / AVX-512 16x16 / scalar from
// cpuid + bins. Ariel: 256-bin AVX-512 ~3-4x over std::upper_bound. TODO: 128.
#ifdef SIMD_UPPER_BOUND
inline bool CpuSupportsAvx2() {
  static const bool supported = __builtin_cpu_supports("avx2");
  return supported;
}

inline bool CpuSupportsAvx512f() {
  static const bool supported = __builtin_cpu_supports("avx512f");
  return supported;
}

// Two-level SIMD upper_bound over 64 sorted thresholds: an 8-wide coarse compare
// picks the group (`coarse8` = each group's last threshold), a second picks the
// slot. AVX2-gated; the target attribute allows intrinsics in a non-mavx2 TU.
__attribute__((target("avx2,popcnt"))) inline int Avx2UpperBoundIndex64(
    const float x, const float* thr64, const float* coarse8) {
  const __m256 vx = _mm256_set1_ps(x);
  const unsigned mc = static_cast<unsigned>(
      _mm256_movemask_ps(_mm256_cmp_ps(vx, _mm256_load_ps(coarse8),
                                       _CMP_GE_OQ)));
  const unsigned K = static_cast<unsigned>(_mm_popcnt_u32(mc));
  if (K >= 8) return 63;
  const __m256 vthr = _mm256_load_ps(thr64 + (K << 3));
  const unsigned mf = static_cast<unsigned>(
      _mm256_movemask_ps(_mm256_cmp_ps(vx, vthr, _CMP_GE_OQ)));
  return static_cast<int>((K << 3) + _mm_popcnt_u32(mf)) - 1;
}

// Same scheme, 16x16 over 256 thresholds. Only called when CpuSupportsAvx512f().
__attribute__((target("avx512f,popcnt"))) inline int Avx512UpperBoundIndex256(
    const float x, const float* thr256, const float* coarse16) {
  const __m512 vx = _mm512_set1_ps(x);
  const __mmask16 mc =
      _mm512_cmp_ps_mask(vx, _mm512_load_ps(coarse16), _CMP_GE_OQ);
  const unsigned K = static_cast<unsigned>(_mm_popcnt_u32(mc));
  if (K >= 16) return 255;
  const __m512 vthr = _mm512_load_ps(thr256 + (K << 4));
  const __mmask16 mf = _mm512_cmp_ps_mask(vx, vthr, _CMP_GE_OQ);
  return static_cast<int>((K << 4) + _mm_popcnt_u32(mf)) - 1;
}
#endif  // SIMD_UPPER_BOUND

struct HistogramBinner {
  // Sorted thresholds (== the histogram bins). Always populated; used by the
  // scalar fallback and by the debug cross-check of the equal-width fast path.
  std::vector<float> scalar_thr;
  float min_value = 0.f;
  float max_value = 0.f;
  int num_bins = 0;
  bool use_equal_width = false;
#ifdef SIMD_UPPER_BOUND
  // Threshold copies + per-group coarse tables for the SIMD kernels. Plain
  // float arrays (not __m256/__m512 members) so the struct compiles without
  // TU-wide ISA flags; the target-attributed kernels load them.
  alignas(64) float thr256[256];
  alignas(64) float coarse16[16];
  bool avx512_256 = false;
  alignas(32) float thr64[64];
  alignas(32) float coarse8[8];
  bool avx2_64 = false;
#endif

  void Init(const std::vector<float>& thr, const float min_value_in,
            const float max_value_in, const proto::NumericalSplit::Type type) {
    min_value = min_value_in;
    max_value = max_value_in;
    num_bins = static_cast<int>(thr.size());
    use_equal_width =
        (type == proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH ||
         type == proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM);
    scalar_thr = thr;
#ifdef SIMD_UPPER_BOUND
    // Runtime kernel selection: bin count picks the kernel shape, cpuid gates
    // the ISA. Anything else falls through to scalar std::upper_bound.
    avx512_256 = false;
    if (thr.size() == 256 && CpuSupportsAvx512f()) {
      for (int i = 0; i < 256; ++i) thr256[i] = thr[i];
      for (int g = 0; g < 16; ++g) coarse16[g] = thr256[(g + 1) * 16 - 1];
      avx512_256 = true;
    }
    avx2_64 = false;
    if (thr.size() == 64 && CpuSupportsAvx2()) {
      for (int i = 0; i < 64; ++i) thr64[i] = thr[i];
      for (int g = 0; g < 8; ++g) coarse8[g] = thr64[(g + 1) * 8 - 1];
      avx2_64 = true;
    }
#endif
  }

  // Reference bin index (last threshold <= x, else -1): std::upper_bound over
  // the real, unpadded thresholds. Both the production scalar fallback and the
  // debug cross-check for the closed-form and SIMD paths.
  int ScalarIndex(const float x) const {
    auto it = std::upper_bound(scalar_thr.begin(), scalar_thr.end(), x);
    if (it == scalar_thr.begin()) return -1;
    return static_cast<int>(it - scalar_thr.begin()) - 1;
  }

  // Returns the index of the last threshold <= x; -1 if all thresholds > x.
  // (Equivalent to std::upper_bound(thr, x) - thr.begin() - 1.)
  int Index(const float x) const {
    if (use_equal_width) {
      const int idx = EqualWidthThresholdIndex(x, min_value, max_value, num_bins);
#ifndef NDEBUG
      // The closed-form equal-width index must match std::upper_bound on the
      // real thresholds (same invariant the finders used to DCHECK inline).
      DCHECK_EQ(idx, ScalarIndex(x))
          << "Fast equal-width binning disagrees with std::upper_bound at "
          << idx;
#endif
      return idx;
    }
#ifdef SIMD_UPPER_BOUND
    if (avx512_256) {
      const int idx = Avx512UpperBoundIndex256(x, thr256, coarse16);
#ifndef NDEBUG
      // The SIMD upper_bound (the non-equal-width / "random" path) must match
      // the scalar std::upper_bound. Debug-only; compiled out in opt.
      DCHECK_EQ(idx, ScalarIndex(x))
          << "AVX-512 SIMD binning disagrees with std::upper_bound at " << idx;
#endif
      return idx;
    }
    if (avx2_64) {
      const int idx = Avx2UpperBoundIndex64(x, thr64, coarse8);
#ifndef NDEBUG
      // The SIMD upper_bound (the non-equal-width / "random" path) must match
      // the scalar std::upper_bound. Debug-only; compiled out in opt.
      DCHECK_EQ(idx, ScalarIndex(x))
          << "AVX2 SIMD binning disagrees with std::upper_bound at " << idx;
#endif
      return idx;
    }
#endif
    return ScalarIndex(x);
  }
};
/* #endregion */

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalHistogram(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kHistoPath);
  // Randomly select some threshold values.
  struct CandidateSplit {
    float threshold;
    utils::IntegerDistributionDouble pos_label_distribution;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };

  float min_value, max_value;
  std::vector<float> bins;
  std::vector<CandidateSplit> candidate_splits;
  HistogramBinner binner;

  {
    CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kHistogramSetup);
    DCHECK(condition != nullptr);
    if (!weights.empty()) {
      DCHECK_EQ(weights.size(), labels.size());
    }

    if (dt_config.missing_value_policy() ==
        proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
      LocalImputationForNumericalAttribute(selected_examples, weights,
                                           attributes, &na_replacement);
    }

    {
      CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kMinMaxNumerical);
      if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                    &max_value)) {
        return SplitSearchResult::kInvalidAttribute;
      }
    }
    // There should be at least two different unique values.
    if (min_value == max_value) {
      return SplitSearchResult::kInvalidAttribute;
    }

    ASSIGN_OR_RETURN(
        bins,
        internal::GenHistogramBins(dt_config.numerical_split().type(),
                                   dt_config.numerical_split().num_candidates(),
                                   attributes, min_value, max_value, random));

    candidate_splits.resize(bins.size());
    for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
      auto& candidate_split = candidate_splits[split_idx];
      candidate_split.pos_label_distribution.SetNumClasses(num_label_classes);
      candidate_split.threshold = bins[split_idx];
    }

    // Build the shared attribute->bin-index mapper (equal-width closed form,
    // SIMD upper_bound, or scalar upper_bound; chosen from type + bin count).
    binner.Init(bins, min_value, max_value, dt_config.numerical_split().type());
  }

  // Compute the split score of each threshold.
  {
    CHRONO_SCOPE_EP(
        ::yggdrasil_decision_forests::chrono_prof::kAssignSamplesToHistogram);
  for (const auto example_idx : selected_examples) {
    const int32_t label = labels[example_idx];
    const float weight = weights.empty() ? 1.f : weights[example_idx];
    float attribute = attributes[example_idx];
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }

    const int idx = binner.Index(attribute);
    if (idx < 0) {
      continue;
    }
    auto& it_split = candidate_splits[idx];
    it_split.num_positive_examples_without_weights++;
    it_split.pos_label_distribution.Add(label, weight);
  }

  // Suffix-sum the per-bin counts into cumulative ">= threshold" counts.
  // Part of histogram accumulation, hence inside kAssignSamplesToHistogram.
  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_distribution.Add(src.pos_label_distribution);
  }
  }

  // Inline entropy: skips the BinaryToIntegerConfusionMatrix Set/Sub round-trip
  // by reading neg counts as (label_distribution - pos). Equivalent to
  // confusion.FinalEntropy(), minus the per-candidate-split matrix setup.
  const double initial_entropy = label_distribution.Entropy();
  const int num_classes = label_distribution.NumClasses();
  const double total_sum = label_distribution.NumObservations();
  const double inv_total = (total_sum > 0) ? 1.0 / total_sum : 0.0;
#ifndef DISABLE_BINARY_ENTROPY_LOOKUP
  const bool use_unweighted_binary_entropy =
      weights.empty() && num_label_classes == 3;
  std::vector<double> count_log_count;
  if (use_unweighted_binary_entropy) {
    CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kEntropyTableSetup);
    count_log_count = internal::BuildCountLogCountTable(
        static_cast<int64_t>(total_sum));
  }
#endif

  // Select the best threshold.
  bool found_split = false;
  {
  CHRONO_SCOPE_EP(
      ::yggdrasil_decision_forests::chrono_prof::kSelectBestThresholdHistogram);
  for (auto& candidate_split : candidate_splits) {
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }

    const auto& pos = candidate_split.pos_label_distribution;
    const double pos_sum = pos.NumObservations();
    const double neg_sum = total_sum - pos_sum;
    double final_entropy;
#ifndef DISABLE_BINARY_ENTROPY_LOOKUP
    if (use_unweighted_binary_entropy) {
      const int64_t pos_sum_int = static_cast<int64_t>(pos_sum);
      const int64_t pos_trues = static_cast<int64_t>(pos.count(2));
      const int64_t neg_sum_int = static_cast<int64_t>(neg_sum);
      const int64_t neg_trues =
          static_cast<int64_t>(label_distribution.count(2)) - pos_trues;
      final_entropy =
          (internal::BinaryEntropyNumeratorFromIntegerCounts(
               pos_trues, pos_sum_int, count_log_count) +
           internal::BinaryEntropyNumeratorFromIntegerCounts(
               neg_trues, neg_sum_int, count_log_count)) *
          inv_total;
    } else
#endif
    {
      double w_entropy = 0.0;
      for (int i = 0; i < num_classes; ++i) {
        const double pi = pos.count(i);
        if (pi > 0 && pi < pos_sum) {
          w_entropy -= pi * std::log(pi / pos_sum);
        }
        const double ni = label_distribution.count(i) - pi;
        if (ni > 0 && ni < neg_sum) {
          w_entropy -= ni * std::log(ni / neg_sum);
        }
      }
      final_entropy = w_entropy * inv_total;
    }
    const double information_gain = initial_entropy - final_entropy;
    if (information_gain > condition->split_score()) {
      condition->set_split_score(information_gain);
      condition->mutable_condition()->mutable_higher_condition()->set_threshold(
          candidate_split.threshold);
      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
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

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    float na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kCartPath);
  proto::DecisionTreeTrainingConfig::Internal::SortingStrategy sorting_strategy;
  CHRONO_BEGIN_EP(cart_setup);
  const auto feature_filler = [&]() {
    if (!weights.empty()) {
      DCHECK_EQ(weights.size(), labels.size());
    }
    if (dt_config.missing_value_policy() ==
        proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
      LocalImputationForNumericalAttribute(selected_examples, weights,
                                           attributes, &na_replacement);
    }

    sorting_strategy =
        EffectiveStrategy(dt_config, selected_examples.size(), internal_config);
    return FeatureNumericalBucket::Filler(selected_examples.size(),
                                          na_replacement, attributes);
  }();

  // "Why ==3" ?
  // Categorical attributes always have one class reserved for
  // "out-of-vocabulary" items. The "num_label_classes" takes into account this
  // class. In case of binary classification, "num_label_classes" is 3 (OOB,
  // False, True).
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Filler
          label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/false>::Initializer
          initializer(label_distribution);
      CHRONO_END_EP(cart_setup,
                 ::yggdrasil_decision_forests::chrono_prof::kCartSetup);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelUnweightedBinaryCategoricalOneValue,
            LabelBinaryCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy.");
      }
    } else {
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Filler
          label_filler(labels, weights);
      LabelBinaryCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);
      CHRONO_END_EP(cart_setup,
                 ::yggdrasil_decision_forests::chrono_prof::kCartSetup);
      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelBinaryCategoricalOneValue,
            LabelBinaryCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelBinaryClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalOneValueBucket</*weighted=*/false>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/false>::Initializer
          initializer(label_distribution);
      CHRONO_END_EP(cart_setup,
                 ::yggdrasil_decision_forests::chrono_prof::kCartSetup);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelUnweightedCategoricalOneValue,
            LabelCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelUnweightedClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    } else {
      LabelCategoricalOneValueBucket</*weighted=*/true>::Filler label_filler(
          labels, weights);
      LabelCategoricalOneValueBucket</*weighted=*/true>::Initializer
          initializer(label_distribution);
      CHRONO_END_EP(cart_setup,
                 ::yggdrasil_decision_forests::chrono_prof::kCartSetup);

      if (sorting_strategy ==
          proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
        const auto& sorted_attributes =
            internal_config.preprocessing
                ->presorted_numerical_features()[attribute_idx];
        return ScanSplitsPresortedSparse<
            FeatureNumericalLabelCategoricalOneValue,
            LabelCategoricalScoreAccumulator>(
            internal_config.preprocessing->num_examples(), selected_examples,
            sorted_attributes.items, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx,
            internal_config.duplicated_selected_examples, condition,
            &cache->cache_v2);
      } else if (sorting_strategy ==
                 proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
        return FindBestSplit_LabelClassificationFeatureNumerical(
            selected_examples, feature_filler, label_filler, initializer,
            min_num_obs, attribute_idx, condition, &cache->cache_v2);
      } else {
        return absl::InvalidArgumentError("Non supported strategy");
      }
    }
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureDiscretizedNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<int32_t>& labels,
    const int32_t num_label_classes,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureDiscretizedNumerical(  // NOLINT(whitespace/line_length)
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureDiscretizedNumerical(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalHistogram<true>(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalHistogram<false>(
    absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalHistogram(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition) {
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kHistoPath);
  // Candidate threshold values.
  struct CandidateSplit {
    float threshold;
    utils::NormalDistributionDouble pos_label_dist;
    int64_t num_positive_examples_without_weights = 0;
    bool operator<(const CandidateSplit& other) const {
      return threshold < other.threshold;
    }
  };

  float min_value, max_value;
  std::vector<float> bins;
  std::vector<CandidateSplit> candidate_splits;
  HistogramBinner binner;

  {
    CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kHistogramSetup);
    DCHECK(condition != nullptr);
    if constexpr (weighted) {
      DCHECK_EQ(weights.size(), labels.size());
    } else {
      DCHECK(weights.empty());
    }

    if (dt_config.missing_value_policy() ==
        proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
      LocalImputationForNumericalAttribute(selected_examples, weights,
                                           attributes, &na_replacement);
    }
    // Determine the minimum and maximum values of the attribute.
    {
      CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kMinMaxNumerical);
      if (!MinMaxNumericalAttribute(selected_examples, attributes, &min_value,
                                    &max_value)) {
        return SplitSearchResult::kInvalidAttribute;
      }
    }

    // There should be at least two different unique values.
    if (min_value == max_value) {
      return SplitSearchResult::kInvalidAttribute;
    }
    // Randomly select some threshold values.
    ASSIGN_OR_RETURN(
        bins,
        internal::GenHistogramBins(dt_config.numerical_split().type(),
                                   dt_config.numerical_split().num_candidates(),
                                   attributes, min_value, max_value, random));

    candidate_splits.resize(bins.size());
    for (int split_idx = 0; split_idx < candidate_splits.size(); split_idx++) {
      auto& candidate_split = candidate_splits[split_idx];
      candidate_split.threshold = bins[split_idx];
    }

    // Shared attribute->bin-index mapper (see HistogramBinner). Gives the
    // regression histogram finder the same SIMD / equal-width acceleration as
    // the classification one, replacing the per-example std::upper_bound below.
    binner.Init(bins, min_value, max_value, dt_config.numerical_split().type());
  }

  // Compute the split score of each threshold.
  {
    CHRONO_SCOPE_EP(
        ::yggdrasil_decision_forests::chrono_prof::kAssignSamplesToHistogram);
  for (const auto example_idx : selected_examples) {
    const float label = labels[example_idx];
    float attribute = attributes[example_idx];
    // TODO gate isnan behind #ifdef. on SPORF this is redundant cuz it's already checked in ApplyProjection
#ifdef ENABLE_ISNAN
    if (std::isnan(attribute)) {
      attribute = na_replacement;
    }
#endif

    const int idx = binner.Index(attribute);
    if (idx < 0) {
      continue;
    }
    auto& it_split = candidate_splits[idx];
    it_split.num_positive_examples_without_weights++;
    if constexpr (weighted) {
      it_split.pos_label_dist.Add(label, weights[example_idx]);
    } else {
      it_split.pos_label_dist.Add(label);
    }
  }

  // Suffix-sum the per-bin counts into cumulative ">= threshold" counts.
  // Part of histogram accumulation, hence inside kAssignSamplesToHistogram.
  for (int split_idx = candidate_splits.size() - 2; split_idx >= 0;
       split_idx--) {
    const auto& src = candidate_splits[split_idx + 1];
    auto& dst = candidate_splits[split_idx];
    dst.num_positive_examples_without_weights +=
        src.num_positive_examples_without_weights;
    dst.pos_label_dist.Add(src.pos_label_dist);
  }
  }

  // Select the best threshold.
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::
                   kSelectBestThresholdHistogram);
  const double initial_variance = label_distribution.Var();
  int best_candidate_split_idx = -1;
  double best_variance_reduction = condition->split_score();
  utils::NormalDistributionDouble neg_label_dist;
  for (int candidate_split_idx = 0;
       candidate_split_idx < candidate_splits.size(); candidate_split_idx++) {
    const auto& candidate_split = candidate_splits[candidate_split_idx];
    if (selected_examples.size() -
                candidate_split.num_positive_examples_without_weights <
            min_num_obs ||
        candidate_split.num_positive_examples_without_weights < min_num_obs) {
      continue;
    }
    neg_label_dist = label_distribution;
    neg_label_dist.Sub(candidate_split.pos_label_dist);
    const double frac_pos = candidate_split.pos_label_dist.NumObservations() /
                            (candidate_split.pos_label_dist.NumObservations() +
                             neg_label_dist.NumObservations());
    const double final_variance =
        frac_pos * candidate_split.pos_label_dist.Var() +
        (1 - frac_pos) * neg_label_dist.Var();
    const double variance_reduction = initial_variance - final_variance;
    if (variance_reduction > best_variance_reduction) {
      best_variance_reduction = variance_reduction;
      best_candidate_split_idx = candidate_split_idx;
    }
  }

  if (best_candidate_split_idx == -1) {
    return SplitSearchResult::kNoBetterSplitFound;
  } else {
    const auto& candidate_split = candidate_splits[best_candidate_split_idx];
    condition->set_split_score(best_variance_reduction);
    condition->mutable_condition()->mutable_higher_condition()->set_threshold(
        candidate_split.threshold);
    condition->set_attribute(attribute_idx);
    condition->set_num_training_examples_without_weight(
        selected_examples.size());
    condition->set_num_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations() +
        neg_label_dist.NumObservations());
    condition->set_num_pos_training_examples_without_weight(
        candidate_split.num_positive_examples_without_weights);
    condition->set_num_pos_training_examples_with_weight(
        candidate_split.pos_label_dist.NumObservations());
    condition->set_na_value(na_replacement >= candidate_split.threshold);
    return SplitSearchResult::kBetterSplitFound;
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, const int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kCartPath);
  CHRONO_BEGIN_EP(cart_setup);
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  const auto sorting_strategy =
      EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  typename LabelHessianNumericalOneValueBucket<weighted>::Filler label_filler(
      gradients, hessians, weights);

  typename LabelHessianNumericalOneValueBucket<weighted>::Initializer
      initializer(sum_gradient, sum_hessian, sum_weights,
                  internal_config.hessian_l1,
                  internal_config.hessian_l2_numerical,
                  dt_config.internal().hessian_split_score_subtract_parent(),
                  monotonic_direction, constraints);

  if (sorting_strategy ==
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    const auto& sorted_attributes =
        internal_config.preprocessing
            ->presorted_numerical_features()[attribute_idx];
    CHRONO_END_EP(cart_setup,
               ::yggdrasil_decision_forests::chrono_prof::kCartSetup);
    return ScanSplitsPresortedSparse<
        FeatureNumericalLabelHessianNumericalOneValue<weighted>,
        LabelHessianNumericalScoreAccumulator>(
        internal_config.preprocessing->num_examples(), selected_examples,
        sorted_attributes.items, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx,
        internal_config.duplicated_selected_examples, condition,
        &cache->cache_v2);
  } else if (sorting_strategy ==
             proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
    CHRONO_END_EP(cart_setup,
               ::yggdrasil_decision_forests::chrono_prof::kCartSetup);
    return FindBestSplit_LabelHessianRegressionFeatureNumerical<weighted>(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  } else {
    return absl::InvalidArgumentError("Non supported strategy");
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    float na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureDiscretizedNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    int num_bins, const std::vector<float>& gradients,
    const std::vector<float>& hessians, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, double sum_gradient,
    double sum_hessian, double sum_weights, int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, int8_t monotonic_direction,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      monotonic_direction, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureDiscretizedNumerical<
      weighted>(selected_examples, feature_filler, label_filler, initializer,
                min_num_obs, attribute_idx, condition, &cache->cache_v2);
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  CHRONO_SCOPE_EP(::yggdrasil_decision_forests::chrono_prof::kCartPath);
  CHRONO_BEGIN_EP(cart_setup);
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  const auto sorting_strategy =
      EffectiveStrategy(dt_config, selected_examples.size(), internal_config);

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  typename LabelNumericalOneValueBucket<weighted>::Filler label_filler(labels,
                                                                       weights);

  typename LabelNumericalOneValueBucket<weighted>::Initializer initializer(
      label_distribution);

  if (sorting_strategy ==
      proto::DecisionTreeTrainingConfig::Internal::FORCE_PRESORTED) {
    const auto& sorted_attributes =
        internal_config.preprocessing
            ->presorted_numerical_features()[attribute_idx];
    CHRONO_END_EP(cart_setup,
               ::yggdrasil_decision_forests::chrono_prof::kCartSetup);
    return ScanSplitsPresortedSparse<
        FeatureNumericalLabelNumericalOneValue<weighted>,
        LabelNumericalScoreAccumulator>(
        internal_config.preprocessing->num_examples(), selected_examples,
        sorted_attributes.items, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx,
        internal_config.duplicated_selected_examples, condition,
        &cache->cache_v2);
  } else if (sorting_strategy ==
             proto::DecisionTreeTrainingConfig::Internal::IN_NODE) {
    CHRONO_END_EP(cart_setup,
               ::yggdrasil_decision_forests::chrono_prof::kCartSetup);
    return FindBestSplit_LabelRegressionFeatureNumerical<weighted>(
        selected_examples, feature_filler, label_filler, initializer,
        min_num_obs, attribute_idx, condition, &cache->cache_v2);
  } else {
    return absl::InvalidArgumentError("Non supported strategy");
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const std::vector<float>& labels, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, const InternalTrainConfig& internal_config,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureDiscretizedNumericalCart<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<float>& labels,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureDiscretizedNumericalCart<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<float>& labels,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureDiscretizedNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const std::vector<dataset::DiscretizedNumericalIndex>& attributes,
    const int num_bins, const std::vector<float>& labels,
    const dataset::DiscretizedNumericalIndex na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureDiscretizedNumericalBucket::Filler feature_filler(
      num_bins, na_replacement, attributes);

  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);

  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  return FindBestSplit_LabelRegressionFeatureDiscretizedNumerical<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureNA(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  FeatureIsMissingBucket::Filler feature_filler(attributes);
  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, {}, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);
      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureNACart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNA<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureNA<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureNA(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::AbstractColumn* attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  FeatureIsMissingBucket::Filler feature_filler(attributes);

  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureNACart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult> FindSplitLabelClassificationFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<int32_t>& labels, const int32_t num_label_classes,
    bool na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      LabelBinaryCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, {}, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedBinaryClassificationFeatureBooleanCart(  // NOLINT(whitespace/line_length)
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelBinaryCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelBinaryCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelBinaryClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      LabelCategoricalBucket</*weighted=*/false>::Filler label_filler(
          labels, weights, label_distribution);

      LabelCategoricalBucket</*weighted=*/false>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelUnweightedClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    } else {
      LabelCategoricalBucket</*weighted=*/true>::Filler label_filler(
          labels, weights, label_distribution);

      LabelCategoricalBucket</*weighted=*/true>::Initializer initializer(
          label_distribution);

      return FindBestSplit_LabelClassificationFeatureBooleanCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);
    }
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);
  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  return FindBestSplit_LabelRegressionFeatureBooleanCart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureBoolean<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);
template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureBoolean<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& labels, bool na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureBoolean<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureBoolean<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache);

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelHessianRegressionFeatureBoolean(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int8_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    bool na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForBooleanAttribute(selected_examples, weights, attributes,
                                       &na_replacement);
  }

  FeatureBooleanBucket::Filler feature_filler(na_replacement, attributes);
  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_numerical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  return FindBestSplit_LabelHessianRegressionFeatureBooleanCart<weighted>(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureCategorical<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureCategorical<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelHessianRegressionFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& gradients, const std::vector<float>& hessians,
    const int32_t num_attribute_classes, int32_t na_replacement,
    const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const double sum_gradient, const double sum_hessian,
    const double sum_weights, const int32_t attribute_idx,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelHessianNumericalBucket<weighted>::Filler label_filler(
      gradients, hessians, weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical);

  typename LabelHessianNumericalBucket<weighted>::Initializer initializer(
      sum_gradient, sum_hessian, sum_weights, internal_config.hessian_l1,
      internal_config.hessian_l2_categorical,
      dt_config.internal().hessian_split_score_subtract_parent(),
      /*monotonic_direction=*/0, constraints);

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalCart<
          weighted>(selected_examples, feature_filler, label_filler,
                    initializer, min_num_obs, attribute_idx, condition,
                    &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelHessianRegressionFeatureCategoricalRandom<
          weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategorical<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategorical<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random);

template <bool weighted>
absl::StatusOr<SplitSearchResult> FindSplitLabelRegressionFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<float>& labels, const int32_t num_attribute_classes,
    int32_t na_replacement, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  if constexpr (weighted) {
    DCHECK_GE(weights.size(), selected_examples.size());
  } else {
    DCHECK(weights.empty());
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelNumericalBucket<weighted>::Filler label_filler(labels, weights);

  typename LabelNumericalBucket<weighted>::Initializer initializer(
      label_distribution);

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelRegressionFeatureCategoricalCart<weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelRegressionFeatureCategoricalRandom<weighted>(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  // TODO: `min_num_obs`is currently ignored.
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }

  const int max_iterations =
      dt_config.categorical_set_greedy_forward().max_selected_items();
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToIntegerConfusionMatrixDouble split_label_distribution;
  utils::BinaryToIntegerConfusionMatrixInt64
      split_label_distribution_no_weights;
  split_label_distribution.SetNumClassesIntDim(num_label_classes);
  split_label_distribution_no_weights.SetNumClassesIntDim(num_label_classes);
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }
  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // TODO: Cache this variable.
  auto per_attribute_value_distributions =
      InitializeClassificationAttributeDistributions<weighted>(
          selected_examples, labels, weights, attribute_values, attribute_bank,
          label_distribution, candidate_attributes_bitmap);

  // Contains, for each example, the index in the "attribute_bank"
  // corresponding to the next candidate attribute value
  // "candidate_attr_value".
  //
  // When initialized, this corresponds to the index of the first values for
  // this particular example: i.e. running_attr_bank_idx[select_idx] ==
  // attribute_values[selected_examples[select_idx]].first.
  std::vector<UnsignedExampleIdx> running_attr_bank_idx(
      selected_examples.size());

  const double initial_entropy = split_label_distribution.InitEntropy();
  // Information gain of the current condition i.e.
  // "positive_attributes_vector".
  double information_gain = 0;
  while (true) {
    double best_information_gain = information_gain;
    int best_attr_value = -1;
    for (int attr_idx = 0; attr_idx < num_attribute_classes; ++attr_idx) {
      if (!candidate_attributes_bitmap[attr_idx]) {
        continue;
      }
      const auto& cur_attr_value_dist =
          per_attribute_value_distributions[attr_idx];
      if (cur_attr_value_dist.neg().NumObservations() == 0) {
        candidate_attributes_bitmap[attr_idx] = false;
        continue;
      }
      double candidate_information_gain =
          initial_entropy - cur_attr_value_dist.FinalEntropy();
      if (candidate_information_gain > best_information_gain) {
        // Best score so far.
        best_information_gain = candidate_information_gain;
        best_attr_value = attr_idx;
      }
    }
    if (best_attr_value == -1) {
      // No attribute value improves the current state.
      break;
    }
    // Fix the attribute value found to be for the positive side.
    positive_attributes_vector.push_back(best_attr_value);
    information_gain = best_information_gain;
    // Update the attribute value distributions.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      // Does this example already belong to a side?
      if (positive_selected_example_bitmap[select_idx]) {
        continue;
      }
      const auto example_idx = selected_examples[select_idx];
      const auto attr_values_range = attribute_values[example_idx];
      // Check if the attribute is missing.
      if (attr_values_range.first > attr_values_range.second) {
        continue;
      }
      // Since second >= first, this is reasonable even if both are unsigned.
      const auto attr_values_list_size =
          attr_values_range.second - attr_values_range.first;
      bool match;
      // Profiling shows that std::binary_search is quite slow for small ranges,
      // common for CatSet splits. The threshold 100 has not been optimized.
      constexpr int binary_search_threshold = 100;
      if (attr_values_list_size <= binary_search_threshold) {
        // Linear search.
        match = std::find(attribute_bank.begin() + attr_values_range.first,
                          attribute_bank.begin() + attr_values_range.second,
                          best_attr_value) !=
                attribute_bank.begin() + attr_values_range.second;
      } else {
        match = std::binary_search(
            attribute_bank.begin() + attr_values_range.first,
            attribute_bank.begin() + attr_values_range.second, best_attr_value);
      }
      if (!match) {
        // The example does not contain `best_attr_value` and therefore does not
        // change side in any distribution.
        continue;
      }
      const auto label = labels[example_idx];
      positive_selected_example_bitmap[select_idx] = true;
      // Update the distribution of the result.
      if constexpr (weighted) {
        const auto weight = weights[example_idx];
        split_label_distribution.mutable_pos()->Add(label, weight);
        split_label_distribution.mutable_neg()->Sub(label, weight);
      } else {
        split_label_distribution.mutable_pos()->Add(label);
        split_label_distribution.mutable_neg()->Sub(label);
      }
      split_label_distribution_no_weights.mutable_pos()->Add(label);
      split_label_distribution_no_weights.mutable_neg()->Sub(label);
      candidate_attributes_bitmap[best_attr_value] = false;

      const auto attr_bank_begin =
          attribute_bank.begin() + attribute_values[example_idx].first;
      const auto attr_bank_end =
          attribute_bank.begin() + attribute_values[example_idx].second;
      auto current_attr_bank_iter = attr_bank_begin;

      // Update the distributions of the other attribute values: Since the
      // current example has been moved irrevocably to the positive side of
      // the result, it has to be moved to the positive of every attribute
      // value distribution. For the attribute values present in this example,
      // this is already the case. Find the remaining ones and move it for
      // them as well.
      // attr_bank is sorted, so moving with two pointers ensures that this
      // loop is linear in num_attribute_classes.
      for (int current_attr_val = 0; current_attr_val < num_attribute_classes;
           ++current_attr_val) {
        if (!candidate_attributes_bitmap[current_attr_val]) {
          // This attribute is already selected in the mask.
          continue;
        }
        while (current_attr_bank_iter != attr_bank_end &&
               *current_attr_bank_iter < current_attr_val) {
          ++current_attr_bank_iter;
        }
        bool current_attr_val_is_in_bank =
            (current_attr_bank_iter != attr_bank_end &&
             *current_attr_bank_iter == current_attr_val);

        if (current_attr_val_is_in_bank) {
          // The value is already on the positive side.
          continue;
        }
        auto& cur_split_stats =
            per_attribute_value_distributions[current_attr_val];
        if constexpr (weighted) {
          const auto weight = weights[example_idx];
          cur_split_stats.mutable_pos()->Add(label, weight);
          cur_split_stats.mutable_neg()->Sub(label, weight);
        } else {
          cur_split_stats.mutable_pos()->Add(label);
          cur_split_stats.mutable_neg()->Sub(label);
        }
      }
    }
    // If the maximum number of iterations is reached, stop.
    if (max_iterations > 0 &&
        positive_attributes_vector.size() >= max_iterations) {
      break;
    }
  }

  if (information_gain > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(information_gain, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategoricalSetGreedyForward<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<int32_t>& labels, const int32_t num_attribute_classes,
    const int32_t num_label_classes, const UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    const int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template <bool weighted>
absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random) {
  // TODO: `min_num_obs`is currently ignored.
  if constexpr (weighted) {
    DCHECK_EQ(weights.size(), labels.size());
  } else {
    DCHECK(weights.empty());
  }

  const int max_iterations =
      dt_config.categorical_set_greedy_forward().max_selected_items();
  // Bitmap of available attribute values. During the course of the algorithm,
  // an attribute value is available if:
  //  - It is selected by the initial random sampling of candidate attribute
  //  values.
  //  - It is not (yet) selected in the positive set.
  //  - It is not pure in the negative examples i.e. it is not present in all
  //  or in none of the non-selected examples (ps: Initially, all the examples
  //  are non-selected).
  std::vector<bool> candidate_attributes_bitmap(num_attribute_classes, true);
  // The "positive attribute set" are the attribute values that, if present
  // in the example, evaluates the node condition as true.
  std::vector<int> positive_attributes_vector;
  // Bitmap of the example that are already in the positive set i.e. for which
  // the condition defined by "positive_attributes_vector" is positive.
  // Instead of being indexed by the example_idx, this bitmap is indexed by
  // "selected_examples" i.e. "positive_selected_example_bitmap[i]==true"
  // means that "selected_examples[i]" is selected.
  std::vector<bool> positive_selected_example_bitmap(selected_examples.size(),
                                                     false);
  // Weighted and non weighted distribution of the labels in the positive and
  // negative sets.
  utils::BinaryToNormalDistributionDouble split_label_distribution;
  utils::BinaryToNormalDistributionDouble split_label_distribution_no_weights;
  // All the examples are initially in the negative set.
  *split_label_distribution.mutable_neg() = label_distribution;
  // Number of example (with weights) where the attribute value (an attribute
  // value is a set of categorical items) that contains the i-th  categorical
  // items.
  std::vector<int64_t> count_examples_without_weights_by_attribute_class(
      num_attribute_classes);
  // Count per categorical item value.
  const auto& attribute_values = attributes.values();
  const auto& attribute_bank = attributes.bank();
  for (const auto example_idx : selected_examples) {
    for (auto bank_idx = attribute_values[example_idx].first;
         bank_idx < attribute_values[example_idx].second; bank_idx++) {
      const auto value = attribute_bank[bank_idx];
      count_examples_without_weights_by_attribute_class[value]++;
    }
    split_label_distribution_no_weights.Add(false, labels[example_idx]);
  }

  // Sample-out items.
  if (!internal::MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
          dt_config, num_attribute_classes, selected_examples,
          count_examples_without_weights_by_attribute_class,
          &candidate_attributes_bitmap, random)) {
    return SplitSearchResult::kInvalidAttribute;
  }

  // TODO: Cache this variable.
  auto per_attribute_value_distributions =
      InitializeRegressionAttributeDistributions<weighted>(
          selected_examples, labels, weights, attribute_values, attribute_bank,
          label_distribution, candidate_attributes_bitmap);

  const double initial_variance = label_distribution.Var();

  // Variance reduction of the current condition i.e.
  // "positive_attributes_vector".
  double variance_reduction = 0.0;

  while (true) {
    // Find which attribute value currently achieves the best variance
    // reduction.
    double best_variance_reduction = variance_reduction;
    int best_attr_value = -1;
    for (int attr_idx = 0; attr_idx < num_attribute_classes; ++attr_idx) {
      const auto& cur_attr_value_dist =
          per_attribute_value_distributions[attr_idx];
      if (!candidate_attributes_bitmap[attr_idx]) {
        continue;
      }
      if (cur_attr_value_dist.neg().NumObservations() == 0) {
        candidate_attributes_bitmap[attr_idx] = false;
        continue;
      }
      double candidate_variance_reduction =
          initial_variance - cur_attr_value_dist.FinalVariance();
      if (candidate_variance_reduction > best_variance_reduction) {
        best_variance_reduction = candidate_variance_reduction;
        best_attr_value = attr_idx;
      }
    }
    if (best_attr_value == -1) {
      // No attribute value improves the current state.
      break;
    }
    // Fix the attribute value found to be for the positive side.
    positive_attributes_vector.push_back(best_attr_value);
    variance_reduction = best_variance_reduction;
    // Update the attribute value distributions.
    for (size_t select_idx = 0; select_idx < selected_examples.size();
         select_idx++) {
      // Does this example already belong to a side?
      if (positive_selected_example_bitmap[select_idx]) {
        continue;
      }
      const auto example_idx = selected_examples[select_idx];
      const auto attr_values_range = attribute_values[example_idx];
      // Check if the attribute is missing.
      if (attr_values_range.first > attr_values_range.second) {
        continue;
      }
      // Since second >= first, this is reasonable even if both are unsigned.
      const auto attr_values_list_size =
          attr_values_range.second - attr_values_range.first;
      bool match;
      // Profiling shows that std::binary_search is quite slow for small ranges,
      // common for CatSet splits. The threshold 100 has not been optimized.
      constexpr int binary_search_threshold = 100;
      if (attr_values_list_size <= binary_search_threshold) {
        // Linear search.
        match = std::find(attribute_bank.begin() + attr_values_range.first,
                          attribute_bank.begin() + attr_values_range.second,
                          best_attr_value) !=
                attribute_bank.begin() + attr_values_range.second;
      } else {
        match = std::binary_search(
            attribute_bank.begin() + attr_values_range.first,
            attribute_bank.begin() + attr_values_range.second, best_attr_value);
      }
      if (!match) {
        // The example does not contain `best_attr_value` and therefore does not
        // change side in any distribution.
        continue;
      }
      const auto label = labels[example_idx];
      positive_selected_example_bitmap[select_idx] = true;
      // Update the distribution of the result.
      if constexpr (weighted) {
        const auto weight = weights[example_idx];
        split_label_distribution.mutable_pos()->Add(label, weight);
        split_label_distribution.mutable_neg()->Sub(label, weight);
      } else {
        split_label_distribution.mutable_pos()->Add(label);
        split_label_distribution.mutable_neg()->Sub(label);
      }
      split_label_distribution_no_weights.mutable_pos()->Add(label);
      split_label_distribution_no_weights.mutable_neg()->Sub(label);
      candidate_attributes_bitmap[best_attr_value] = false;

      // If the number of iterations is what we want, just stop the loop
      if (max_iterations > 0 &&
          positive_attributes_vector.size() >= max_iterations) {
        break;
      }

      const auto attr_bank_begin =
          attribute_bank.begin() + attribute_values[example_idx].first;
      const auto attr_bank_end =
          attribute_bank.begin() + attribute_values[example_idx].second;
      auto current_attr_bank_iter = attr_bank_begin;

      // Update the distributions of the other attribute values: Since the
      // current example has been moved irrevocably to the positive side of
      // the result, it has to be moved to the positive of every attribute
      // value distribution. For the attribute values present in this example,
      // this is already the case. Find the remaining ones and move it for
      // them as well.
      // attr_bank is sorted, so moving with two pointers ensures that this
      // loop is linear in num_attribute_classes.
      for (int current_attr_val = 0; current_attr_val < num_attribute_classes;
           ++current_attr_val) {
        if (!candidate_attributes_bitmap[current_attr_val]) {
          // This attribute is already selected in the mask.
          continue;
        }
        while (current_attr_bank_iter != attr_bank_end &&
               *current_attr_bank_iter < current_attr_val) {
          ++current_attr_bank_iter;
        }
        bool current_attr_val_is_in_bank =
            (current_attr_bank_iter != attr_bank_end &&
             *current_attr_bank_iter == current_attr_val);

        if (current_attr_val_is_in_bank) {
          // The value is already on the positive side.
          continue;
        }
        auto& cur_split_stats =
            per_attribute_value_distributions[current_attr_val];
        if constexpr (weighted) {
          const auto weight = weights[example_idx];
          cur_split_stats.mutable_pos()->Add(label, weight);
          cur_split_stats.mutable_neg()->Sub(label, weight);
        } else {
          cur_split_stats.mutable_pos()->Add(label);
          cur_split_stats.mutable_neg()->Sub(label);
        }
      }
    }
  }

  if (variance_reduction > condition->split_score()) {
    condition->set_na_value(false);
    SetConditionHelper(variance_reduction, attribute_idx,
                       split_label_distribution,
                       split_label_distribution_no_weights, condition);
    // Assign the positive set to the condition.
    std::sort(positive_attributes_vector.begin(),
              positive_attributes_vector.end());
    SetPositiveAttributeSetOfCategoricalContainsCondition(
        positive_attributes_vector, num_attribute_classes, condition);
    return SplitSearchResult::kBetterSplitFound;
  } else {
    return SplitSearchResult::kNoBetterSplitFound;
  }
}

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<true>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template absl::StatusOr<SplitSearchResult>
FindSplitLabelRegressionFeatureCategoricalSetGreedyForward<false>(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights,
    const dataset::VerticalDataset::CategoricalSetColumn& attributes,
    const std::vector<float>& labels, int32_t num_attribute_classes,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::NormalDistributionDouble& label_distribution,
    int32_t attribute_idx, proto::NodeCondition* condition,
    utils::RandomEngine* random);

template <typename LabelBucket, typename ExampleBucketSet,
          typename LabelScoreAccumulator>
absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);
  typename LabelBucket::Filler label_filler(labels, weights,
                                            label_distribution);

  typename LabelBucket::Initializer initializer(label_distribution);

  // Create buckets.
  ExampleBucketSet& example_set_accumulator =
      *GetCachedExampleBucketSet<ExampleBucketSet>(&cache->cache_v2);
  FillExampleBucketSet<ExampleBucketSet, /*require_label_sorting=*/false>(
      selected_examples, feature_filler, label_filler, &example_set_accumulator,
      &cache->cache_v2);

  // Scanner for the "one label value vs others".
  const auto one_vs_other_scan = [&]() -> SplitSearchResult {
    // Value and index of the buckets.
    auto& bucket_order = cache->cache_v2.bucket_order;
    bucket_order.resize(example_set_accumulator.items.size());

    SplitSearchResult split_status = SplitSearchResult::kInvalidAttribute;
    for (int32_t positive_label_value = 0;
         positive_label_value < num_label_classes; positive_label_value++) {
      if (label_distribution.count(positive_label_value) == 0) {
        // Never observed label value.
        continue;
      }
      if (num_label_classes == 3 && positive_label_value == 1) {
        // "True vs others" or "False vs others" are equivalent for binary
        // classification.
        continue;
      }

      // Order value of the buckets.
      for (int bucket_idx = 0; bucket_idx < bucket_order.size(); bucket_idx++) {
        const auto& bucket = example_set_accumulator.items[bucket_idx];
        const float ratio_positive_label =
            bucket.label.SafeProportionOrMinusInfinity(positive_label_value);
        DCHECK(!std::isnan(ratio_positive_label));
        bucket_order[bucket_idx] = {ratio_positive_label, bucket_idx};
      }

      // Sort the bucket indices.
      std::sort(bucket_order.begin(), bucket_order.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      // Scan the buckets in order.
      const auto scan_result =
          ScanSplitsCustomOrder<ExampleBucketSet, LabelScoreAccumulator>(
              bucket_order, feature_filler, initializer,
              example_set_accumulator, selected_examples.size(), min_num_obs,
              attribute_idx, condition, &cache->cache_v2);
      if (scan_result < split_status) {
        split_status = scan_result;
      }
    }
    return split_status;
  };

  // Scanner for the "one hot" type condition i.e. conditions of the type:
  // "attribute == value".
  //
  // Note: In the majority of cases, one-hot (on attribute value) is worst that
  // "one class vs others". This is however a common solution, and this code is
  // present for comparison purpose.
  const auto one_hot_scan = [&]() -> absl::StatusOr<SplitSearchResult> {
    STATUS_CHECK_EQ(example_set_accumulator.items.size(),
                    num_attribute_classes);

    std::uniform_real_distribution<float> sampling_dist;

    auto& neg = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        false, &cache->cache_v2);
    auto& pos = *GetCachedLabelScoreAccumulator<LabelScoreAccumulator>(
        true, &cache->cache_v2);

    initializer.InitFull(&pos);
    const double weighted_num_examples = pos.WeightedNumExamples();

    double best_score = condition->split_score();
    bool tried_one_split = false;
    int best_bucket_idx = -1;

    for (int attribute_value = 0; attribute_value < num_attribute_classes;
         attribute_value++) {
      if (dt_config.categorical().one_hot().sampling() < 1.f &&
          sampling_dist(*random) >
              dt_config.categorical().one_hot().sampling()) {
        continue;
      }
      const auto bucket_idx = attribute_value;
      const auto& item = example_set_accumulator.items[bucket_idx];

      const int64_t num_pos_examples = item.label.count;
      const int64_t num_neg_examples =
          selected_examples.size() - item.label.count;

      // Enough examples?
      if (num_pos_examples < min_num_obs || num_neg_examples < min_num_obs) {
        continue;
      }

      initializer.InitFull(&neg);
      initializer.InitEmpty(&pos);

      item.label.SubToScoreAcc(&neg);
      item.label.AddToScoreAcc(&pos);

      const auto score = Score<>(initializer, weighted_num_examples, pos, neg);
      tried_one_split = true;

      if (score > best_score) {
        // Memorize the split.
        best_bucket_idx = bucket_idx;
        best_score = score;
        condition->set_num_pos_training_examples_without_weight(
            num_pos_examples);
        condition->set_num_pos_training_examples_with_weight(
            pos.WeightedNumExamples());
      }
    }

    if (best_bucket_idx != -1) {
      // Finalize the best found split.
      condition->set_na_value(na_replacement == best_bucket_idx);
      SetPositiveAttributeSetOfCategoricalContainsCondition(
          {best_bucket_idx}, num_attribute_classes, condition);

      condition->set_attribute(attribute_idx);
      condition->set_num_training_examples_without_weight(
          selected_examples.size());
      condition->set_num_training_examples_with_weight(weighted_num_examples);
      condition->set_split_score(best_score);
      return SplitSearchResult::kBetterSplitFound;
    } else {
      return tried_one_split ? SplitSearchResult::kNoBetterSplitFound
                             : SplitSearchResult::kInvalidAttribute;
    }
    return absl::OkStatus();
  };

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return one_vs_other_scan();

    case proto::Categorical::kOneHot:
      return one_hot_scan();
      break;

    case proto::Categorical::kRandom:
      return ScanSplitsRandomBuckets<ExampleBucketSet, LabelScoreAccumulator>(
          feature_filler, label_filler, initializer, example_set_accumulator,
          selected_examples.size(), min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelClassificationFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const std::vector<int32_t>& labels, int32_t num_attribute_classes,
    int32_t num_label_classes, int32_t na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const utils::IntegerDistributionDouble& label_distribution,
    int32_t attribute_idx, utils::RandomEngine* random,
    proto::NodeCondition* condition, SplitterPerThreadCache* cache) {
  if (!weights.empty()) {
    DCHECK_EQ(weights.size(), labels.size());
  }
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  if (num_label_classes == 3) {
    // Binary classification.
    if (weights.empty()) {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelBinaryCategoricalBucket</*weighted=*/false>,
          FeatureCategoricalLabelUnweightedBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    } else {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelBinaryCategoricalBucket</*weighted=*/true>,
          FeatureCategoricalLabelBinaryCategorical,
          LabelBinaryCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    }
  } else {
    // Multi-class classification.
    if (weights.empty()) {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelCategoricalBucket</*weighted=*/false>,
          FeatureCategoricalLabelUnweightedCategorical,
          LabelCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    } else {
      return FindSplitLabelClassificationFeatureCategorical<
          LabelCategoricalBucket</*weighted=*/true>,
          FeatureCategoricalLabelCategorical, LabelCategoricalScoreAccumulator>(
          selected_examples, weights, attributes, labels, num_attribute_classes,
          num_label_classes, na_replacement, min_num_obs, dt_config,
          label_distribution, attribute_idx, random, condition, cache);
    }
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const CategoricalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftCategoricalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO: Add support for-presorted splitting.

  return FindBestSplit_LabelUpliftClassificationFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureNumericalCart(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const absl::Span<const float> attributes,
    const NumericalUpliftLabelStats& label_stats, float na_replacement,
    UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForNumericalAttribute(selected_examples, weights, attributes,
                                         &na_replacement);
  }

  FeatureNumericalBucket::Filler feature_filler(selected_examples.size(),
                                                na_replacement, attributes);

  LabelUpliftNumericalOneValueBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalOneValueBucket::Filler label_filler(
      label_stats.outcome_values, label_stats.treatment_values, weights);

  // TODO: Add support for pre-sorted splitting.

  return FindBestSplit_LabelUpliftNumericalFeatureNumerical(
      selected_examples, feature_filler, label_filler, initializer, min_num_obs,
      attribute_idx, condition, &cache->cache_v2);
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftCategoricalFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const CategoricalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftCategoricalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftCategoricalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights,
      dt_config.uplift().empty_bucket__ordering());

  // TODO: Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftClassificationFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

absl::StatusOr<SplitSearchResult>
FindSplitLabelUpliftNumericalFeatureCategorical(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<float>& weights, const std::vector<int32_t>& attributes,
    const NumericalUpliftLabelStats& label_stats, int num_attribute_classes,
    int32_t na_replacement, UnsignedExampleIdx min_num_obs,
    const proto::DecisionTreeTrainingConfig& dt_config, int32_t attribute_idx,
    const InternalTrainConfig& internal_config, proto::NodeCondition* condition,
    SplitterPerThreadCache* cache, utils::RandomEngine* random) {
  DCHECK(!weights.empty());
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::LOCAL_IMPUTATION) {
    LocalImputationForCategoricalAttribute(selected_examples, weights,
                                           attributes, num_attribute_classes,
                                           &na_replacement);
  }

  FeatureCategoricalBucket::Filler feature_filler(num_attribute_classes,
                                                  na_replacement, attributes);

  LabelUpliftNumericalBucket::Initializer initializer(
      label_stats.label_distribution,
      dt_config.uplift().min_examples_in_treatment(),
      dt_config.uplift().split_score());
  LabelUpliftNumericalBucket::Filler label_filler(
      label_stats.label_distribution, label_stats.outcome_values,
      label_stats.treatment_values, weights,
      dt_config.uplift().empty_bucket__ordering());

  // TODO: Add support for pre-sorted splitting.

  const auto algorithm =
      (num_attribute_classes < dt_config.categorical().arity_limit_for_random())
          ? dt_config.categorical().algorithm_case()
          : proto::Categorical::kRandom;

  switch (algorithm) {
    case proto::Categorical::ALGORITHM_NOT_SET:
    case proto::Categorical::kCart:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalCart(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx, condition, &cache->cache_v2);

    case proto::Categorical::kRandom:
      return FindBestSplit_LabelUpliftNumericalFeatureCategoricalRandom(
          selected_examples, feature_filler, label_filler, initializer,
          min_num_obs, attribute_idx,
          NumTrialsForRandomCategoricalSplit(dt_config.categorical().random()),
          condition, &cache->cache_v2, random);

    default:
      return absl::InvalidArgumentError("Non supported");
  }
}

int NumAttributesToTest(const proto::DecisionTreeTrainingConfig& dt_config,
                        const int num_attributes,
                        const model::proto::Task task) {
  int num_attributes_to_test;
  // User specified number of candidate attributes.
  if (dt_config.has_num_candidate_attributes_ratio() &&
      dt_config.num_candidate_attributes_ratio() >= 0) {
    if (dt_config.has_num_candidate_attributes() &&
        dt_config.num_candidate_attributes() > 0) {
      LOG(WARNING) << "Both \"num_candidate_attributes\" and "
                      "\"num_candidate_attributes_ratio\" are specified. "
                      "Ignoring \"num_candidate_attributes\".";
    }
    num_attributes_to_test = static_cast<int>(
        std::ceil(dt_config.num_candidate_attributes_ratio() * num_attributes));
  } else {
    num_attributes_to_test = dt_config.num_candidate_attributes();
  }

  // Automatic number of attribute selection logic.
  if (num_attributes_to_test == 0) {
    switch (task) {
      default:
      case model::proto::Task::CATEGORICAL_UPLIFT:
      case model::proto::Task::CLASSIFICATION:
        num_attributes_to_test = static_cast<int>(
            ceil(std::sqrt(static_cast<double>(num_attributes))));
        break;
      case model::proto::Task::REGRESSION:
        num_attributes_to_test =
            static_cast<int>(ceil(static_cast<double>(num_attributes) / 3));
        break;
    }
  }

  // Special value to use all the available attributes.
  if (num_attributes_to_test == -1) {
    num_attributes_to_test = static_cast<int>(num_attributes);
  }

  // Make sure we don't select more than the available attributes.
  num_attributes_to_test =
      std::min(num_attributes_to_test, static_cast<int>(num_attributes));

  return num_attributes_to_test;
}

void GetCandidateAttributes(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    int* num_attributes_to_test, std::vector<int32_t>* candidate_attributes,
    utils::RandomEngine* random) {
  {
    CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kGetCandidateAttributesAssign);
    candidate_attributes->assign(config_link.features().begin(),
                                 config_link.features().end());
  }
  {
    CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kGetCandidateAttributesShuffle);
    std::shuffle(candidate_attributes->begin(), candidate_attributes->end(),
                 *random);
  }
  {
    CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kGetCandidateAttributesNumToTest);
    *num_attributes_to_test = NumAttributesToTest(
        dt_config, candidate_attributes->size(), config.task());
  }
}

absl::Status GenerateRandomImputation(
    const dataset::VerticalDataset& src, const std::vector<int>& attributes,
    const absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset* dst, utils::RandomEngine* random) {
  STATUS_CHECK_EQ(dst->ncol(), 0);
  dst->set_data_spec(src.data_spec());
  RETURN_IF_ERROR(dst->CreateColumnsFromDataspec());
  dst->set_nrow(examples.size());
  for (const auto col_idx : attributes) {
    RETURN_IF_ERROR(GenerateRandomImputationOnColumn(
        src.column(col_idx), examples, dst->mutable_column(col_idx), random));
  }
  return absl::OkStatus();
}

absl::Status GenerateRandomImputationOnColumn(
    const dataset::VerticalDataset::AbstractColumn* src,
    const absl::Span<const UnsignedExampleIdx> examples,
    dataset::VerticalDataset::AbstractColumn* dst,
    utils::RandomEngine* random) {
  STATUS_CHECK_EQ(src->type(), dst->type());
  // Extract the indices of the example with non-na values i.e. the candidate
  // for sampling.
  std::vector<UnsignedExampleIdx> non_na_examples;
  for (const auto example_idx : examples) {
    if (!src->IsNa(example_idx)) {
      non_na_examples.push_back(example_idx);
    }
  }

  if (non_na_examples.empty()) {
    return src->ExtractAndAppend(examples, dst);
  }

  std::uniform_int_distribution<SignedExampleIdx> non_na_example_dist(
      0, std::max(static_cast<SignedExampleIdx>(0),
                  static_cast<SignedExampleIdx>(non_na_examples.size()) - 1));

  std::vector<SignedExampleIdx> source_indices;
  source_indices.resize(examples.size());

  UnsignedExampleIdx local_example_idx = 0;
  for (const auto example_idx : examples) {
    if (src->IsNa(example_idx)) {
      // Sample a random non-na value.
      source_indices[local_example_idx] =
          non_na_examples[non_na_example_dist(*random)];
    } else {
      source_indices[local_example_idx] = example_idx;
    }
    local_example_idx++;
  }
  return src->ExtractAndAppend(source_indices, dst);
}

void SetInternalDefaultHyperParameters(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& link_config,
    const dataset::proto::DataSpecification& data_spec,
    proto::DecisionTreeTrainingConfig* dt_config) {}

void SetDefaultHyperParameters(proto::DecisionTreeTrainingConfig* config) {
  // Emulation of histogram splits.
  if (!config->numerical_split().has_num_candidates()) {
    switch (config->numerical_split().type()) {
      case proto::NumericalSplit::HISTOGRAM_RANDOM:
      case proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM:
        config->mutable_numerical_split()->set_num_candidates(1);
        break;
      case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH:
      case proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM:
        config->mutable_numerical_split()->set_num_candidates(255);
        break;
      default:
        break;
    }
  }

  // By default, use axis aligned splits.
  if (config->split_axis_case() ==
      proto::DecisionTreeTrainingConfig::SPLIT_AXIS_NOT_SET) {
    config->mutable_axis_aligned_split();
  }

  // By default, use the cart categorical algorithm for categorical features.
  if (config->categorical().algorithm_case() ==
      proto::Categorical::ALGORITHM_NOT_SET) {
    config->mutable_categorical()->mutable_cart();
  }

  // By default, use the local growing strategy i.e. divide and conquer.
  if (config->growing_strategy_case() ==
      proto::DecisionTreeTrainingConfig::GROWING_STRATEGY_NOT_SET) {
    config->mutable_growing_strategy_local();
  }

  // Change the pre-sorting strategy if not supported by the splitter.
  using Internal = proto::DecisionTreeTrainingConfig::Internal;
  auto sorting_strategy = config->internal().sorting_strategy();

  // If possible, use presorting by default.
  if (sorting_strategy == Internal::AUTO) {
    sorting_strategy = Internal::PRESORTED;
  }

  if (sorting_strategy == Internal::PRESORTED ||
      sorting_strategy == Internal::FORCE_PRESORTED) {
    if (config->has_sparse_oblique_split() ||
        config->has_mhld_oblique_split() ||
        config->missing_value_policy() !=
            proto::DecisionTreeTrainingConfig::GLOBAL_IMPUTATION) {
      sorting_strategy = Internal::IN_NODE;
    }
  }

  config->mutable_internal()->set_sorting_strategy(sorting_strategy);

  // The binary weight hyperparameter is deprecated for the more general weights
  // hyperparameter.
  if (config->sparse_oblique_split().has_binary_weight()) {
    if (config->sparse_oblique_split().binary_weight()) {
      config->mutable_sparse_oblique_split()->mutable_binary();
    } else {
      config->mutable_sparse_oblique_split()->mutable_continuous();
    }
    config->mutable_sparse_oblique_split()->clear_binary_weight();
  }

  // By default, we use binary weights.
  if (config->has_sparse_oblique_split() &&
      config->sparse_oblique_split().weights_case() ==
          proto::DecisionTreeTrainingConfig::SparseObliqueSplit::
              WEIGHTS_NOT_SET) {
    config->mutable_sparse_oblique_split()->mutable_binary();
  }
}

void SplitHonestExamples(
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const float leaf_rate, utils::RandomEngine* random_engine,
    std::vector<UnsignedExampleIdx>& leaf_examples,
    std::vector<UnsignedExampleIdx>& working_selected_examples) {
  DCHECK(std::is_sorted(selected_examples.begin(), selected_examples.end()));

  // Reduce the risk of std::vector re-allocations.
  const float error_margin = 1.1f;

  // Reserve total size to avoid reallocations.
  const size_t N = selected_examples.size();
  leaf_examples.reserve(N * leaf_rate * error_margin);
  working_selected_examples.reserve(N * (1.0f - leaf_rate) * error_margin);

  size_t U = 0;
  if (!selected_examples.empty()) {
    U = 1;
    for (size_t i = 1; i < selected_examples.size(); ++i) {
      if (selected_examples[i] != selected_examples[i - 1]) {
        ++U;
      }
    }
  }

  size_t k_needed = static_cast<size_t>(U * leaf_rate);
  size_t n_remaining = U;
  std::uniform_real_distribution<float> dist_01;

  if (selected_examples.empty()) return;

  // Reservoir sampling
  bool send_to_leaf = false;
  for (size_t i = 0; i < selected_examples.size(); ++i) {
    if (i == 0 || selected_examples[i] != selected_examples[i - 1]) {
      if (n_remaining > 0) {
        if (dist_01(*random_engine) <
            static_cast<float>(k_needed) / n_remaining) {
          send_to_leaf = true;
          if (k_needed > 0) {
            --k_needed;
          }
        } else {
          send_to_leaf = false;
        }
        --n_remaining;
      }
    }
    if (send_to_leaf) {
      leaf_examples.push_back(selected_examples[i]);
    } else {
      working_selected_examples.push_back(selected_examples[i]);
    }
  }
}

absl::Status GrowTreeBestFirstGlobal(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, NodeWithChildren* root,
    utils::RandomEngine* random,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  if (config.monotonic_constraints_size() > 0) {
    return absl::InvalidArgumentError(
        "Global growth of decision trees (i.e. "
        "growing_strategy=kGrowingStrategyBestFirstGlobal) does not support "
        "monotonic constraints.");
  }

  if (leaf_examples.has_value()) {
    return absl::InvalidArgumentError(
        "honest trees are not (yet) supported with "
        "growing_strategy_best_first_global strategy.");
  }

  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    return absl::InvalidArgumentError(
        "Random local imputation not supported in best first global "
        "tree growth.");
  }
  if (config_link.per_columns_size() > 0) {
    for (const auto feature : config_link.features()) {
      if (config_link.per_columns(feature).has_monotonic_constraint()) {
        return absl::InvalidArgumentError(
            "GBT with growing_strategy_best_first_global does not support "
            "monotonic constraints.");
      }
    }
  }

  PerThreadCache cache;

  struct CandidateSplit {
    // Split.
    proto::NodeCondition condition;
    // Indices of examples in the node.
    SelectedExamplesRollingBuffer example_idxs;
    // Global score of the split.
    float score;
    // The currently leaf node.
    NodeWithChildren* node;
    // Depth of the node.
    int depth;

    bool operator<(const CandidateSplit& other) const {
      return score < other.score;
    }
  };

  // List of candidate splits.
  std::priority_queue<CandidateSplit> candidate_splits;

  // Initialize a node and update the list of candidate splits with a given
  // node.
  const auto ingest_node = [&](const SelectedExamplesRollingBuffer example_idxs,
                               NodeWithChildren* node,
                               const int depth) -> absl::Status {
    node->mutable_node()->set_num_pos_training_examples_without_weight(
        example_idxs.size());
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, example_idxs.active, weights, config, config_link,
        node));

    if (example_idxs.size() < dt_config.min_examples() ||
        (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth())) {
      // Stop the grow of the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }
    proto::NodeCondition condition;
    ASSIGN_OR_RETURN(
        const auto has_better_condition,
        FindBestCondition(train_dataset, example_idxs.active, weights, config,
                          config_link, dt_config, node->node(), internal_config,
                          {}, &condition, random, &cache));
    if (!has_better_condition) {
      // No good condition found. Close the branch.
      node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
      return absl::OkStatus();
    }

    const float score = condition.split_score() * example_idxs.size();
    candidate_splits.push({/*.condition =*/std::move(condition),
                           /*.example_idxs =*/example_idxs,
                           /*.score =*/score,
                           /*.node =*/node,
                           /*.depth =*/depth});
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(ingest_node(selected_examples, root, /*depth=*/0));

  // Total number of nodes in the tree.
  int num_nodes = 1;

  const int max_num_nodes =
      dt_config.growing_strategy_best_first_global().max_num_nodes();

  while (!candidate_splits.empty() &&
         (max_num_nodes < 0 || num_nodes < max_num_nodes) &&
         (!internal_config.timeout.has_value() ||
          internal_config.timeout >= absl::Now())) {
    // Ensure the candidate set is not larger than  "max_num_nodes". Note:
    // There is not need for mode than "max_num_nodes" candidate splits.
    while (max_num_nodes >= 0 && candidate_splits.size() > max_num_nodes) {
      candidate_splits.top().node->FinalizeAsLeaf(
          dt_config.store_detailed_label_distribution());
      candidate_splits.pop();
    }

    // Split the node.
    auto split = candidate_splits.top();
    candidate_splits.pop();

    *split.node->mutable_node()->mutable_condition() = split.condition;
    split.node->CreateChildren();
    split.node->FinalizeAsNonLeaf(
        dt_config.keep_non_leaf_label_distribution(),
        dt_config.store_detailed_label_distribution());

    const auto& condition = split.node->node().condition();

    // Add new candidate splits for children.

    ASSIGN_OR_RETURN(
        auto exemple_split,
        internal::SplitExamplesInPlace(
            train_dataset, split.example_idxs, condition,
            /*dataset_is_dense=*/false,
            dt_config.internal_error_on_wrong_splitter_statistics()));

    RETURN_IF_ERROR(ingest_node(exemple_split.positive_examples,
                                split.node->mutable_pos_child(),
                                split.depth + 1));
    RETURN_IF_ERROR(ingest_node(exemple_split.negative_examples,
                                split.node->mutable_neg_child(),
                                split.depth + 1));
    num_nodes++;
  }

  // Finalize the remaining candidates.
  while (!candidate_splits.empty()) {
    candidate_splits.top().node->FinalizeAsLeaf(
        dt_config.store_detailed_label_distribution());
    candidate_splits.pop();
  }
  return absl::OkStatus();
}

absl::Status DecisionTreeTrain(
    const dataset::VerticalDataset& train_dataset,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    DecisionTree* dt, const InternalTrainConfig& internal_config) {
  // Note: This function is the entry point of all decision tree learning.

  // Check the sorting strategy.
  if (dt_config.internal().has_ensure_effective_sorting_strategy() &&
      (dt_config.internal().ensure_effective_sorting_strategy() !=
       dt_config.internal().sorting_strategy())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Non expected effective sorting strategy:",
        proto::DecisionTreeTrainingConfig::Internal::SortingStrategy_Name(
            dt_config.internal().ensure_effective_sorting_strategy()),
        "(expected) != ",
        proto::DecisionTreeTrainingConfig::Internal::SortingStrategy_Name(
            dt_config.internal().sorting_strategy()),
        "(actual)"));
  }

  // Decide if execution should happen in single-thread or concurrent mode.

  std::optional<std::vector<UnsignedExampleIdx>> leaf_examples;
  std::vector<UnsignedExampleIdx> working_selected_examples;

  // Fail if the data spec has invalid columns.
  for (const auto feature_idx : config_link.features()) {
    const auto& data_spec_columns = train_dataset.data_spec().columns();
    const auto column_type = data_spec_columns[feature_idx].type();
    if (column_type != dataset::proto::NUMERICAL &&
        column_type != dataset::proto::CATEGORICAL &&
        column_type != dataset::proto::CATEGORICAL_SET &&
        column_type != dataset::proto::BOOLEAN &&
        column_type != dataset::proto::DISCRETIZED_NUMERICAL &&
        column_type != dataset::proto::NUMERICAL_VECTOR_SEQUENCE) {
      return absl::InvalidArgumentError(
          absl::Substitute("Column $0 has type $1, which is not supported "
                           "for decision tree training.",
                           data_spec_columns[feature_idx].name(),
                           dataset::proto::ColumnType_Name(column_type)));
    }
  }

  // Check monotonic constraints
  if (config.monotonic_constraints_size() > 0 &&
      !dt_config.keep_non_leaf_label_distribution()) {
    return absl::InvalidArgumentError(
        "keep_non_leaf_label_distribution=false is not compatible with "
        "monotonic constraints. To minimize the size of your serving model "
        "(with or without monotonic constraints), use "
        "pure_serving_model=true.");
  }

  // Check if oblique splits are correctly specified
  if (dt_config.has_sparse_oblique_split()) {
    if (dt_config.sparse_oblique_split().has_binary_weight() &&
        dt_config.sparse_oblique_split().weights_case() !=
            dt_config.sparse_oblique_split().WEIGHTS_NOT_SET) {
      return absl::InvalidArgumentError(
          "Both sparse_oblique_split.binary_weights and "
          "sparse_oblique_split.weights are set. Setting "
          "sparse_oblique_split.binary_weights is deprecated and replaced by "
          "just setting sparse_oblique_split.weights.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().max_exponent() > 31) {
      return absl::InvalidArgumentError(
          "The maximum exponent for sparse oblique power-of-two weights cannot "
          "be larger than 31.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().min_exponent() < -31) {
      return absl::InvalidArgumentError(
          "The minimum exponent for sparse oblique power-of-two weights cannot "
          "be smaller than -31.");
    }
    if (dt_config.sparse_oblique_split().power_of_two().min_exponent() >
        dt_config.sparse_oblique_split().power_of_two().max_exponent()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "The minimum exponent for sparse oblique power-of-two weights cannot "
          "be larger than the maximum exponent. Got minimum: $0, maximum: $1",
          dt_config.sparse_oblique_split().power_of_two().min_exponent(),
          dt_config.sparse_oblique_split().power_of_two().max_exponent()));
    }
    if (dt_config.sparse_oblique_split().integer().minimum() >
        dt_config.sparse_oblique_split().integer().maximum()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "The minimum value for sparse oblique integer weights cannot "
          "be larger than the maximum value. Got minimum: $0, maximum: $1",
          dt_config.sparse_oblique_split().integer().minimum(),
          dt_config.sparse_oblique_split().integer().maximum()));
    }
  }

  if (dt_config.has_honest()) {
    // Split the examples in two parts. One ("selected_examples_buffer")
    // will be used to infer the structure of the trees while the second
    // ("leaf_examples_buffer") will be used to determine the leaf values
    // (i.e. the predictions).
    leaf_examples = std::vector<UnsignedExampleIdx>();
    // If the internal training config provides a special seed for the random
    // split, use this seed with a new random engine. Otherwise, just use the
    // default random engine.
    if (internal_config.honest_split_seed.has_value()) {
      utils::RandomEngine honest_split_random(
          *internal_config.honest_split_seed);
      SplitHonestExamples(selected_examples,
                          dt_config.honest().ratio_leaf_examples(),
                          &honest_split_random, leaf_examples.value(),
                          working_selected_examples);
    } else {
      SplitHonestExamples(selected_examples,
                          dt_config.honest().ratio_leaf_examples(), random,
                          leaf_examples.value(), working_selected_examples);
    }
  } else {
    working_selected_examples.assign(selected_examples.begin(),
                                     selected_examples.end());
  }

  if (!(dt_config.numerical_vector_sequence()
            .enable_projected_more_than_conditions() ||
        dt_config.numerical_vector_sequence()
            .enable_closer_than_conditions())) {
    return absl::InvalidArgumentError(
        "No condition types for vector sequences are enabled. Enable "
        "projected-more-than conditions, closer-than conditions or both.");
  }

  auto leaf_example_span = leaf_examples.has_value()
                               ? std::optional<absl::Span<UnsignedExampleIdx>>(
                                     absl::MakeSpan(leaf_examples.value()))
                               : std::nullopt;

  return DecisionTreeCoreTrain(train_dataset, config, config_link, dt_config,
                               deployment, weights, random, internal_config, dt,
                               absl::MakeSpan(working_selected_examples),
                               leaf_example_span);
}

std::unique_ptr<SplitterFinderStreamProcessor>
CreateSplitterFinderStreamProcessor(int num_threads) {
  if (num_threads <= 1) {
    return nullptr;
  }
  LOG(INFO) << "Create processor with " << num_threads
            << " threads for split computation";
  auto find_condition =
      [](SplitterWorkRequest request) -> absl::StatusOr<SplitterWorkResponse> {
    const auto& common = *(request.common);
    if (common.dt_config.internal().generate_fake_error_in_splitter()) {
      return absl::InternalError("Fake error");
    }
    return FindBestConditionFromSplitterWorkRequest(
        common.weights, common.config, common.config_link, common.dt_config,
        common.internal_config, request);
  };
  auto split_finder_processor = std::make_unique<SplitterFinderStreamProcessor>(
      "SplitFinder", num_threads, find_condition);
  split_finder_processor->StartWorkers();

  return split_finder_processor;
}

absl::Status DecisionTreeCoreTrain(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, utils::RandomEngine* random,
    const InternalTrainConfig& internal_config, DecisionTree* dt,
    absl::Span<UnsignedExampleIdx> selected_examples,
    std::optional<absl::Span<UnsignedExampleIdx>> leaf_examples) {
  dt->CreateRoot();
  PerThreadCache cache;

  auto selected_examples_rb = SelectedExamplesRollingBuffer::Create(
      selected_examples, &cache.selected_example_buffer);
  std::optional<SelectedExamplesRollingBuffer> leaf_examples_rb;
  if (leaf_examples.has_value()) {
    leaf_examples_rb = SelectedExamplesRollingBuffer::Create(
        leaf_examples.value(), &cache.leaf_example_buffer);
  }

  switch (dt_config.growing_strategy_case()) {
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyLocal: {
      const auto constraints = NodeConstraints::CreateNodeConstraints();
#if defined(DEPTHWISE_1_PASS) || defined(SYMMETRIC_OPTIMIZED) || \
    defined(BFS_ONLY)
      return GrowTreeLocalBFS(train_dataset, config, config_link, dt_config,
                              deployment, weights, 1, internal_config,
                              constraints, false, dt->mutable_root(), random,
                              &cache, selected_examples_rb, leaf_examples_rb);
#else
      return GrowTreeLocal(train_dataset, config, config_link, dt_config,
                           deployment, weights, 1, internal_config, constraints,
                           false, dt->mutable_root(), random, &cache,
                           selected_examples_rb, leaf_examples_rb);
#endif
    } break;
    case proto::DecisionTreeTrainingConfig::kGrowingStrategyBestFirstGlobal:
      return GrowTreeBestFirstGlobal(
          train_dataset, config, config_link, dt_config, deployment, weights,
          internal_config, dt->mutable_root(), random, selected_examples_rb,
          leaf_examples_rb);
      break;
    default:
      return absl::InvalidArgumentError("Grow strategy not set");
  }
}

ABSL_ATTRIBUTE_ALWAYS_INLINE static absl::Status NodeTrain(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights,
    const InternalTrainConfig& internal_config, utils::RandomEngine* random,
    PerThreadCache* cache, internal::NodeAndExamples node_and_examples,
    std::deque<internal::NodeAndExamples>& node_stack) {
  auto& selected_examples = node_and_examples.selected_examples;
  auto& leaf_examples = node_and_examples.leaf_examples;
  const auto depth = node_and_examples.depth;
  const auto& constraints = node_and_examples.constraints;
  const auto set_leaf_already_set = node_and_examples.set_leaf_already_set;
  auto node = node_and_examples.node;
#ifdef CHRONO_PROFILE
  using namespace yggdrasil_decision_forests::chrono_prof;
  // Set depth explicitly from the struct. Works for both BFS (flat
  // iteration) and DFS (recursion overwrites depth on each entry).
  tls_ctx.cur_depth = depth;
  CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kNodeTrain);

  const int t = tls_ctx.cur_tree;
  const int d = depth;
  if (t >= 0) {
    if (d >= node_cnt()[t].size()) {  // grow once if new depth
      node_cnt()[t].resize(d + 1);
      sample_cnt()[t].resize(d + 1);
    }
    node_cnt()[t][d]++;
    sample_cnt()[t][d] += selected_examples.size();
  }

  // NODEWISE_CHRONO only: arms this node's AP recorder if (t, depth) is gated in
  // and emits its record on exit. Declared after the kNodeTrain timer so it
  // unwinds first, keeping the push_back inside that interval. Else inert.
  CHRONO_NODEWISE_NODE(t, d, node_and_examples.node_id,
                       selected_examples.size());
#endif

  if (selected_examples.empty()) {
    return absl::InternalError("No examples fed to the node trainer");
  }
  node->mutable_node()->set_num_pos_training_examples_without_weight(
      selected_examples.size());

  if (!set_leaf_already_set) {
    // Set the node value (i.e. the label distribution).
    {
      CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kSetLeafValue);
      RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
          train_dataset, selected_examples.active, weights, config, config_link,
          node));
    }
    RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
  }

  auto finalize_as_leaf = [&]() -> absl::Status {
    if (leaf_examples.has_value()) {
      // Override the leaf values.
      RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
          train_dataset, leaf_examples->active, weights, config, config_link,
          node));
      RETURN_IF_ERROR(ApplyConstraintOnNode(constraints, node));
    }
    // Stop the growth of the branch.
    node->FinalizeAsLeaf(dt_config.store_detailed_label_distribution());
    return absl::OkStatus();
  };

  if (selected_examples.size() < dt_config.min_examples() ||
      (dt_config.max_depth() >= 0 && depth >= dt_config.max_depth()) ||
      (internal_config.timeout.has_value() &&
       internal_config.timeout < absl::Now())) {
    return finalize_as_leaf();
  }

  // Dataset used to train this node.
  const dataset::VerticalDataset* train_dataset_for_splitter;
  absl::Span<const UnsignedExampleIdx> selected_examples_for_splitter;
  // If true, the entire dataset "local_train_dataset" is composed of training
  // examples for this node. If false, only the subset of
  // "local_train_dataset" indexed by "selected_examples" are to be considered
  // for this node i.e. local_train_dataset[selected_examples[i]].
  bool splitter_dataset_is_compact;

  // Extract the random local imputation.
  dataset::VerticalDataset random_local_imputation_train_dataset;
  std::vector<UnsignedExampleIdx> random_local_imputation_selected_examples;
  if (dt_config.missing_value_policy() ==
      proto::DecisionTreeTrainingConfig::RANDOM_LOCAL_IMPUTATION) {
    std::vector<int> label_and_input_features(config_link.features().begin(),
                                              config_link.features().end());
    label_and_input_features.push_back(config_link.label());
    RETURN_IF_ERROR(GenerateRandomImputation(
        train_dataset, label_and_input_features, selected_examples.active,
        &random_local_imputation_train_dataset, random));
    random_local_imputation_selected_examples.resize(selected_examples.size());
    std::iota(random_local_imputation_selected_examples.begin(),
              random_local_imputation_selected_examples.end(), 0);

    train_dataset_for_splitter = &random_local_imputation_train_dataset;
    selected_examples_for_splitter =
        absl::MakeConstSpan(random_local_imputation_selected_examples);
    splitter_dataset_is_compact = true;
  } else {
    selected_examples_for_splitter =
        absl::MakeConstSpan(selected_examples.active);
    splitter_dataset_is_compact = false;
    train_dataset_for_splitter = &train_dataset;
  }

  // Determine the best split.
  if (selected_examples_for_splitter.empty()) {
    return absl::InternalError("No examples fed to the splitter");
  }

  bool has_better_condition;
  {
    CHRONO_SCOPE_COARSE(
        ::yggdrasil_decision_forests::chrono_prof::kFindBestCondition);
    ASSIGN_OR_RETURN(
        has_better_condition,
        FindBestCondition(*train_dataset_for_splitter,
                          selected_examples_for_splitter, weights, config,
                          config_link, dt_config, node->node(), internal_config,
                          constraints,
                          node->mutable_node()->mutable_condition(), random,
                          cache));
  }
  if (!has_better_condition) {
    return finalize_as_leaf();
  }
  STATUS_CHECK_EQ(
      selected_examples.size(),
      node->node().condition().num_training_examples_without_weight());
  node->CreateChildren();
  node->FinalizeAsNonLeaf(dt_config.keep_non_leaf_label_distribution(),
                          dt_config.store_detailed_label_distribution());

  // Separate the positive and negative examples.
  CHRONO_BEGIN_COARSE(split_examples_in_place);
  ASSIGN_OR_RETURN(
      auto example_split,
      internal::SplitExamplesInPlace(
          *train_dataset_for_splitter, selected_examples,
          node->node().condition(), splitter_dataset_is_compact,
          dt_config.internal_error_on_wrong_splitter_statistics()));
  CHRONO_END_COARSE(split_examples_in_place,
             ::yggdrasil_decision_forests::chrono_prof::kSplitExamplesInPlace);

  if (example_split.positive_examples.empty() ||
      example_split.negative_examples.empty()) {
    // The splitter statistics don't match exactly the condition evaluation and
    // one of the children is pure.
    node->ClearChildren();
    return finalize_as_leaf();
  }

  // Separate the positive and negative examples used only to determine the node
  // value.
  std::optional<ExampleSplitRollingBuffer> node_only_example_split;
  if (leaf_examples.has_value()) {
    ASSIGN_OR_RETURN(
        node_only_example_split,
        internal::SplitExamplesInPlace(
            train_dataset, *leaf_examples, node->node().condition(), false,
            dt_config.internal_error_on_wrong_splitter_statistics(),
            /*examples_are_training_examples=*/false));
    if (node_only_example_split->positive_examples.empty() ||
        node_only_example_split->negative_examples.empty()) {
      node->ClearChildren();
      return finalize_as_leaf();
    }
  }

  // Set leaf outputs
  {
    CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kSetLeafValue);
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, example_split.positive_examples.active, weights, config,
        config_link, node->mutable_pos_child()));
    RETURN_IF_ERROR(internal_config.set_leaf_value_functor(
        train_dataset, example_split.negative_examples.active, weights, config,
        config_link, node->mutable_neg_child()));
  }
  RETURN_IF_ERROR(
      ApplyConstraintOnNode(constraints, node->mutable_pos_child()));
  RETURN_IF_ERROR(
      ApplyConstraintOnNode(constraints, node->mutable_neg_child()));

  // Children constraints
  auto pos_constraints = constraints;
  auto neg_constraints = constraints;
  const int monotonic_constraint_sign = MonotonicConstraintSign(
      config_link, node->node().condition().attribute());
  if (monotonic_constraint_sign != 0) {
    RETURN_IF_ERROR(DivideMonotonicConstraintToChildren(
        constraints, monotonic_constraint_sign == 1,
        dt_config.internal().check_monotonic_constraints(), node,
        node->mutable_pos_child(), node->mutable_neg_child(), &pos_constraints,
        &neg_constraints));
  }

  // Negative child.
  node_stack.push_back(
      {node->mutable_neg_child(), std::move(example_split.negative_examples),
       node_only_example_split.has_value()
           ? std::optional<SelectedExamplesRollingBuffer>(
                 std::move(node_only_example_split->negative_examples))
           : std::nullopt,
       depth + 1, neg_constraints, true});
  // Positive child.
  node_stack.push_back(
      {node->mutable_pos_child(), std::move(example_split.positive_examples),
       node_only_example_split.has_value()
           ? std::optional<SelectedExamplesRollingBuffer>(
                 std::move(node_only_example_split->positive_examples))
           : std::nullopt,
       depth + 1, pos_constraints, true});

#ifdef NODEWISE_CHRONO
  // Heap indices for the two children just pushed (neg is at size()-2, pos at
  // the back). Set after the fact so the initializers above stay identical in
  // every other build, where NodeAndExamples has no node_id at all.
  node_stack[node_stack.size() - 2].node_id =
      chrono_prof::NodewiseChildId(node_and_examples.node_id,
                                   /*positive=*/false, depth + 1);
  node_stack.back().node_id = chrono_prof::NodewiseChildId(
      node_and_examples.node_id, /*positive=*/true, depth + 1);
#endif

  return absl::OkStatus();
}

absl::Status GrowTreeLocal(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, bool set_leaf_already_set,
    NodeWithChildren* root, utils::RandomEngine* random, PerThreadCache* cache,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  std::deque<internal::NodeAndExamples> node_stack;
  node_stack.push_back({root, std::move(selected_examples),
                        std::move(leaf_examples), depth, constraints,
                        set_leaf_already_set});
#ifdef NODEWISE_CHRONO
  // The root enters at depth 1 and is heap index 1. Any other entry depth is a
  // detached subtree (the BFS->DFS handoff) whose position is unknown here, so
  // id 0 = unknown, which NodewiseChildId propagates to the whole subtree.
  node_stack.back().node_id = (depth == 1) ? 1 : 0;
#endif

  while (!node_stack.empty()) {
    auto current_node = std::move(node_stack.back());
    node_stack.pop_back();

    RETURN_IF_ERROR(NodeTrain(train_dataset, config, config_link, dt_config,
                              deployment, weights, internal_config, random,
                              cache, std::move(current_node), node_stack));
  }

  return absl::OkStatus();
}

#if defined(DEPTHWISE_1_PASS) || defined(SYMMETRIC_OPTIMIZED)
// First depth any fused-per-level kernel runs at. The root is one node, so a
// depthwise pass there is just the per-node path with a different RNG order.
// Kept here so the kernels stay agnostic of the depth they run at.
constexpr int32_t kMinDepthwiseDepth = 2;
#endif

#if defined(DEPTHWISE_1_PASS)
// Extra floor on top of kMinDepthwiseDepth: levels below it run the per-node
// BFS fallback, levels at or deeper run the fused kernel. From DW1_MIN_DEPTH
// (default 0 = fused from kMinDepthwiseDepth on). For a depth line-search.
int32_t Depthwise1PassMinDepth() {
  static const int32_t value = [] {
    const char* e = std::getenv("DW1_MIN_DEPTH");
    return e != nullptr ? static_cast<int32_t>(std::strtol(e, nullptr, 10)) : 0;
  }();
  return value;
}
#endif

#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
// Rows above which a node is a CANDIDATE for the fused kernel (the overlap gate
// then thins them). DW1_HOT_MIN_ROWS, default 1000; rows not n_gathers, so the
// gate stays monotone. See kernel_variants.md "Part 1 — the row gate".
size_t Dw1HotMinRows() {
  static const size_t value = [] {
    const char* e = std::getenv("DW1_HOT_MIN_ROWS");
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : static_cast<size_t>(1000);
  }();
  return value;
}

// Percent of a node's distinct columns another candidate must also read for it
// to stay fused. DW1_HOT_MIN_SHARE, default 50; 0 = row gate alone. NOT
// monotone. TODO Ariel linesearch. See kernel_variants.md "Part 2".
size_t Dw1HotMinSharePct() {
  static const size_t value = [] {
    const char* e = std::getenv("DW1_HOT_MIN_SHARE");
    return e != nullptr ? static_cast<size_t>(std::strtoull(e, nullptr, 10))
                        : static_cast<size_t>(50);
  }();
  return value;
}

bool Dw1HotOverlapStats() {
  static const bool value = [] {
    const char* e = std::getenv("DW1_HOT_OVERLAP_STATS");
    return e != nullptr && std::string(e) == "1";
  }();
  return value;
}
#endif

#if defined(SYMMETRIC_OPTIMIZED)
// Deepest level (inclusive) running the symmetric bagwide path; deeper frontier
// nodes hand their subtree to GrowTreeLocal. From SYMMETRIC_MAX_DEPTH (default
// INT32_MAX; the root is never symmetric, so 5 keeps 2..5 and DFS from 6).
int32_t SymmetricMaxDepth() {
  static const int32_t value = [] {
    const char* e = std::getenv("SYMMETRIC_MAX_DEPTH");
    return e != nullptr ? static_cast<int32_t>(std::strtol(e, nullptr, 10))
                        : INT32_MAX;
  }();
  return value;
}
#endif

// BFS (level-order) variant of GrowTreeLocal: a FIFO deque collects the current
// depth's nodes into a `depth_batch`, then dispatches each through NodeTrain.
// That collection is the seam where the fused-per-level Apply hooks in.
absl::Status GrowTreeLocalBFS(
    const dataset::VerticalDataset& train_dataset,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const proto::DecisionTreeTrainingConfig& dt_config,
    const model::proto::DeploymentConfig& deployment,
    const std::vector<float>& weights, const int32_t depth,
    const InternalTrainConfig& internal_config,
    const NodeConstraints& constraints, bool set_leaf_already_set,
    NodeWithChildren* root, utils::RandomEngine* random, PerThreadCache* cache,
    SelectedExamplesRollingBuffer selected_examples,
    std::optional<SelectedExamplesRollingBuffer> leaf_examples) {
  std::deque<internal::NodeAndExamples> node_queue;
  node_queue.push_back({root, std::move(selected_examples),
                        std::move(leaf_examples), depth, constraints,
                        set_leaf_already_set});
#ifdef NODEWISE_CHRONO
  node_queue.back().node_id = (depth == 1) ? 1 : 0;
#endif

#if defined(SYMMETRIC_OPTIMIZED) || defined(DEPTHWISE_1_PASS)
  // Incremental sorted-bag state of the symmetric and DW1 shared-rows kernels.
  // thread_local so capacity amortizes across trees; validity is per-tree.
  // The colwalk control never populates or reads it, so it pays nothing.
  static thread_local DepthBagState depth_bag_state;
  depth_bag_state.valid = false;
  // Previous fused depth's node pointers, captured before the NodeTrains consume
  // the batch; afterwards IsLeaf() says which parents split, giving the
  // parent->child batch-index mapping the relabel needs. Same for both kernels.
  std::vector<NodeWithChildren*> prev_nodes;
  std::vector<int32_t> first_child;
#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
  // Previous fused depth's hot index -> its index in prev_nodes / first_child;
  // the bag's labels are hot indices, so the relabel needs it to find an entry's
  // parent. Empty => no previous hot set => bag rebuild.
  std::vector<uint32_t> prev_hot_to_full;
  // Column-overlap gate scratch, hoisted out of the kernel so the O(max_attr)
  // counters are allocated once per tree, not per depth. ov_col_nodes is reset
  // only over the touched columns, keeping the per-depth cost O(refs).
  std::vector<int32_t> ov_col_nodes;   // per column: n. nodes reading it
  std::vector<int32_t> ov_touched;     // columns to reset after each gate pass
  std::vector<int32_t> ov_cand_cols;   // flat CSR of candidate -> distinct columns
  std::vector<size_t> ov_cand_off;     
  std::vector<char> ov_parent_hot;     // node -> was its parent hot last depth?
  {
    int ov_max_attr = 0;
    for (const auto attribute_idx : config_link.numerical_features()) {
      ov_max_attr = std::max(ov_max_attr, attribute_idx);
    }
    ov_col_nodes.assign(static_cast<size_t>(ov_max_attr) + 1, 0);
  }
#endif

  // Rebuild first_child from the previous batch's node pointers: NodeTrain pushes
  // (neg, pos) per split parent in order, so children are consecutive pairs in
  // this batch. Left empty (=> concat+VQSort) if it doesn't tile the batch.
  [[maybe_unused]] const auto rebuild_first_child = [&](int num_nodes) {
    first_child.clear();
    if (prev_nodes.empty()) return;
    first_child.resize(prev_nodes.size());
    int32_t next_child = 0;
    for (size_t i = 0; i < prev_nodes.size(); ++i) {
      if (prev_nodes[i]->IsLeaf()) {
        first_child[i] = -1;
      } else {
        first_child[i] = next_child;
        next_child += 2;
      }
    }
    if (next_child != num_nodes) first_child.clear();
  };
  // Capture this batch's node pointers before the NodeTrain loop moves the
  // batch entries; consumed at the next fused depth by rebuild_first_child.
  [[maybe_unused]] const auto capture_prev_nodes =
      [&](const std::vector<internal::NodeAndExamples>& batch) {
        prev_nodes.resize(batch.size());
        for (size_t n = 0; n < batch.size(); ++n) {
          prev_nodes[n] = batch[n].node;
        }
      };
#endif

  while (!node_queue.empty()) {
    const int32_t current_depth = node_queue.front().depth;
    // Whole depth iteration, pinned to current_depth (the body rewrites
    // cur_depth). Under DW1/symmetric the fused Apply runs outside NodeTrain,
    // so ΣNodeTrain covers only part of the depth; this covers all of it.
    CHRONO_SCOPE_COARSE_AT(
        ::yggdrasil_decision_forests::chrono_prof::kDepthTrain, current_depth);
    std::vector<internal::NodeAndExamples> depth_batch;
    // Not chrono'd: measured at <0.1 s for 3M rows — completely trivial.
    while (!node_queue.empty() &&
           node_queue.front().depth == current_depth) {
      depth_batch.push_back(std::move(node_queue.front()));
      node_queue.pop_front();
    }

#if defined(DEPTHWISE_1_PASS)
    if (dt_config.has_sparse_oblique_split() &&
        current_depth >= kMinDepthwiseDepth &&
        current_depth >= Depthwise1PassMinDepth()) {
      // Fused-per-level CPU Apply. Sample projections per node, preserving
      // ordinary Sparse Oblique RF semantics (SYMMETRIC_DW1: one shared draw
      // per depth), then precompute each node's slab before the split search.
      const int num_proj = GetNumProjections(
          dt_config, config_link.numerical_features_size());
      const float projection_density = std::clamp(
          dt_config.sparse_oblique_split().projection_density_factor() /
              config_link.numerical_features_size(),
          0.f, 1.f);

      const int num_nodes = depth_batch.size();
      // Both dimensions are known on entry, so size the 2D structure once here,
      // outside the scope below: the allocation is not billed to
      // kSampleProjection, and SampleProjection self-clears (no pre-zeroing).
      std::vector<std::vector<internal::Projection>> all_node_projs(
          num_nodes, std::vector<internal::Projection>(num_proj));
      std::vector<std::vector<int8_t>> all_node_mono(
          num_nodes, std::vector<int8_t>(num_proj));
#ifdef CHRONO_PROFILE
      // This depth-level work runs BEFORE the depth's NodeTrains, so cur_depth
      // still holds the previous depth. Pin it so kSampleProjection, Dw1PreSize
      // / Dw1Sweep and kProjectionEvaluate land in the right (tree, depth) cell.
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth =
          current_depth;
#endif
      {
        // Per-node sampling (2^d nodes x K projections) runs at depth level,
        // outside NodeTrain — without this scope it is invisible to the
        // TreeTrain = NodeTrain + ApplyProjection + SampleProjection sum.
        CHRONO_SCOPE_COARSE(
            ::yggdrasil_decision_forests::chrono_prof::kSampleProjection);
#ifdef SYMMETRIC_DW1
        // Symmetric semantics on the DW1 kernel: draw K projections ONCE per
        // depth, then hand every node the same set. Rest of Kernel untouched from DW1
        for (int p = 0; p < num_proj; ++p) {
          internal::SampleProjection(
              config_link.numerical_features(), dt_config,
              train_dataset.data_spec(), config_link, projection_density,
              &all_node_projs[0][p], &all_node_mono[0][p], random);
        }
        for (int n = 1; n < num_nodes; ++n) {
          all_node_projs[n] = all_node_projs[0];
          all_node_mono[n] = all_node_mono[0];
        }
#else
        for (int n = 0; n < num_nodes; ++n) {
          for (int p = 0; p < num_proj; ++p) {
            internal::SampleProjection(
                config_link.numerical_features(), dt_config,
                train_dataset.data_spec(), config_link, projection_density,
                &all_node_projs[n][p], &all_node_mono[n][p], random);
          }
        }
#endif
      }

      std::vector<absl::Span<const UnsignedExampleIdx>> sel_spans(num_nodes);
      for (int n = 0; n < num_nodes; ++n) {
        sel_spans[n] = depth_batch[n].selected_examples.active;
      }

#ifndef DW1_COLWALK_CONTROL
      // Parent->child batch-index mapping for the shared-rows depth-bag relabel
      // (see rebuild_first_child above). Unused by the colwalk control.
      rebuild_first_child(num_nodes);
#endif

#ifndef DW1_COLWALK_CONTROL
      // ── Hot gate, part 1: rows ────────────────────────────────────────
      // Only big nodes are CANDIDATES for the fused kernel; the rest take
      // oblique.cc's cold branch. Sampling is untouched, so the tree is
      // invariant to both gates. Billed to kProjectionEvaluate (disjoint).
      CHRONO_BEGIN_COARSE(dw1_hot_gate);
      const size_t hot_min_rows = Dw1HotMinRows();
      std::vector<uint32_t> hot_nodes;                  // hot idx -> node
      hot_nodes.reserve(num_nodes);
      for (int n = 0; n < num_nodes; ++n) {
        if (sel_spans[n].size() >= hot_min_rows) {
          hot_nodes.push_back(static_cast<uint32_t>(n));
        }
      }

      // ── Hot gate, part 2: column overlap ──────────────────────────────
      // Keep a candidate iff ≥ DW1_HOT_MIN_SHARE % of its DISTINCT columns are
      // read by another. ONE pass: dropping a node only lowers other columns'
      // reader counts, so iterating would drop strictly more, with no fixpoint.
      size_t ov_stats_candidates = 0, ov_stats_cand_cols = 0;
      {
        const size_t min_share_pct = Dw1HotMinSharePct();
        const size_t num_candidates = hot_nodes.size();
        ov_cand_cols.clear();
        ov_cand_off.assign(1, 0);
        for (const uint32_t n : hot_nodes) {
          const size_t begin = ov_cand_cols.size();
          for (const auto& proj : all_node_projs[n]) {
            for (const auto& feat : proj) {
              ov_cand_cols.push_back(feat.attribute_idx);
            }
          }
          const auto begin_it = ov_cand_cols.begin() + begin;
          std::sort(begin_it, ov_cand_cols.end());
          ov_cand_cols.erase(std::unique(begin_it, ov_cand_cols.end()),
                             ov_cand_cols.end());
          ov_cand_off.push_back(ov_cand_cols.size());
        }
        // Per column: how many candidates read it. ov_cand_cols is deduped
        // per node, so one increment per entry is exactly that count.
        for (const int32_t c : ov_cand_cols) {
          if (ov_col_nodes[c]++ == 0) ov_touched.push_back(c);
        }
        size_t kept = 0;
        for (size_t k = 0; k < num_candidates; ++k) {
          const size_t begin = ov_cand_off[k], end = ov_cand_off[k + 1];
          size_t shared = 0;
          for (size_t i = begin; i < end; ++i) {
            if (ov_col_nodes[ov_cand_cols[i]] >= 2) ++shared;
          }
          const size_t total = end - begin;
          if (total == 0 || shared * 100 >= min_share_pct * total) {
            hot_nodes[kept++] = hot_nodes[k];
          }
        }
        ov_stats_candidates = num_candidates;
        ov_stats_cand_cols = ov_touched.size();
        hot_nodes.resize(kept);
        for (const int32_t c : ov_touched) ov_col_nodes[c] = 0;
        ov_touched.clear();
      }

      // The overlap gate is NOT monotone: this hot set need not be a subset of
      // the previous one's children. The relabel would detect that itself, but
      // only after a wasted bag pass — so break up front and go rebuild.
      if (depth_bag_state.valid && !first_child.empty() &&
          !prev_hot_to_full.empty()) {
        ov_parent_hot.assign(num_nodes, 0);
        for (const uint32_t ph : prev_hot_to_full) {
          if (ph >= first_child.size()) {
            depth_bag_state.valid = false;
            break;
          }
          const int32_t c = first_child[ph];
          if (c < 0) continue;  // Parent became a leaf.
          ov_parent_hot[c] = 1;
          ov_parent_hot[c + 1] = 1;
        }
        if (depth_bag_state.valid) {
          for (const uint32_t n : hot_nodes) {
            if (!ov_parent_hot[n]) {
              depth_bag_state.valid = false;
              break;
            }
          }
        }
      }

      if (Dw1HotOverlapStats()) {
        // Single training thread assumed, no locking; the fprintf bills to AP,
        // so this is a structural dump, never a measurement. `bag` = what the
        // gate cost it: `relabel` = still telescoping, `rebuild` = VQSort.
        std::fprintf(stdout,
                     "[DW1 hotoverlap] depth=%d nodes=%d candidates=%zu "
                     "kept=%zu dropped=%zu cand_cols=%zu min_share=%zu%% "
                     "bag=%s\n",
                     static_cast<int>(current_depth), num_nodes,
                     ov_stats_candidates, hot_nodes.size(),
                     ov_stats_candidates - hot_nodes.size(), ov_stats_cand_cols,
                     Dw1HotMinSharePct(),
                     depth_bag_state.valid ? "relabel" : "rebuild");
        std::fflush(stdout);
      }

      std::vector<int32_t> hot_of_node(num_nodes, -1);  // node -> hot idx, -1 cold
      for (size_t k = 0; k < hot_nodes.size(); ++k) {
        hot_of_node[hot_nodes[k]] = static_cast<int32_t>(k);
      }
      const int num_hot = static_cast<int>(hot_nodes.size());

      // Compact the hot spans and projections into dense hot-indexed arrays: the
      // kernel sizes its slabs and scratch by the frontier it is given. The
      // projections are MOVED, so neither hot nor cold nodes pay a copy.
      std::vector<absl::Span<const UnsignedExampleIdx>> hot_sel_spans(num_hot);
      std::vector<std::vector<internal::Projection>> hot_projs(num_hot);
      size_t hot_rows = 0;
      for (int k = 0; k < num_hot; ++k) {
        const uint32_t n = hot_nodes[k];
        hot_sel_spans[k] = sel_spans[n];
        hot_projs[k] = std::move(all_node_projs[n]);
        hot_rows += sel_spans[n].size();
      }
      // Closes the gate region opened before the row gate, here so the slab
      // construction below stays unbilled just as the control leaves its
      // `projected(num_nodes)` unbilled: the two differ only by the gates.
      CHRONO_END_COARSE(dw1_hot_gate,
                 ::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);

      std::vector<std::vector<float>> projected(num_hot);
#else
      std::vector<std::vector<float>> projected(num_nodes);
#endif  // !DW1_COLWALK_CONTROL
#ifdef LINECOUNT_A
      {
        // Exact per-node distinct-line tally for this depth. sel is sorted, so
        // distinct lines = 1 + (id>>4) transitions. Lock-free per node, one
        // locked accumulate per depth-batch => negligible contention.
        double r_sum = 0.0, l_sum = 0.0;
        for (int n = 0; n < num_nodes; ++n) {
          const auto sp = sel_spans[n];
          const size_t rn = sp.size();
          r_sum += static_cast<double>(rn);
          if (rn == 0) continue;
          const UnsignedExampleIdx* p = sp.data();
          size_t lines = 1;
          uint64_t prev = static_cast<uint64_t>(p[0]) >> 4;
          for (size_t i = 1; i < rn; ++i) {
            const uint64_t cur = static_cast<uint64_t>(p[i]) >> 4;
            if (cur != prev) {
              ++lines;
              prev = cur;
            }
          }
          l_sum += static_cast<double>(lines);
        }
        g_dw1_linecount.Add(current_depth, num_nodes, r_sum, l_sum);
      }
#endif
#ifdef CALLGRIND_DEPTH
      // Instrument ONLY the kernel: with callgrind --instr-atstart=no, the CSV
      // load and non-kernel training skip the 50-100x cache-sim. Zero here, dump
      // right after, so each per-depth dump isolates the gather's D1 misses.
      CALLGRIND_START_INSTRUMENTATION;
      CALLGRIND_ZERO_STATS;
#endif
#ifndef DW1_COLWALK_CONTROL
      if (num_hot > 0) {
        {
          // The hot bag is labelled by hot index but its parent->child hop runs
          // in the full node domain, so it is advanced here, not in the kernel.
          // The scope is the kernel's own, sequential, so bag time bills to AP.
          CHRONO_SCOPE_COARSE(
              ::yggdrasil_decision_forests::chrono_prof::kProjectionEvaluate);
          AdvanceDepthBagHot(absl::MakeConstSpan(sel_spans),
                             absl::MakeConstSpan(hot_sel_spans),
                             absl::MakeConstSpan(first_child),
                             absl::MakeConstSpan(prev_hot_to_full),
                             absl::MakeConstSpan(hot_of_node), hot_rows,
                             &depth_bag_state);
        }
        RETURN_IF_ERROR(ApplyProjectionsDepthwise1Pass(
            train_dataset, config_link.numerical_features(),
            absl::MakeConstSpan(hot_sel_spans), absl::MakeConstSpan(hot_projs),
            &depth_bag_state, absl::MakeSpan(projected), current_depth));
      } else {
        // No hot node at this depth: no bag was produced, so the next fused
        // depth must rebuild rather than relabel. (The gate is monotone, so in
        // practice no deeper node is hot either and this latches.)
        depth_bag_state.valid = false;
      }
#else
      // The col-sharing control has no depth bag; it ignores depth_bag_state.
      RETURN_IF_ERROR(ApplyProjectionsDepthwise1Pass(
          train_dataset, config_link.numerical_features(),
          absl::MakeConstSpan(sel_spans), absl::MakeConstSpan(all_node_projs),
          &depth_bag_state, absl::MakeSpan(projected), current_depth));
#endif  // !DW1_COLWALK_CONTROL
#ifdef CALLGRIND_DEPTH
      { char nm[64];
        std::snprintf(nm, sizeof nm, "dw1_depth_%d", (int)current_depth);
        CALLGRIND_DUMP_STATS_AT(nm); }
      CALLGRIND_STOP_INSTRUMENTATION;
#endif

#ifndef DW1_COLWALK_CONTROL
      // Handover to the next fused depth's relabel, captured before the
      // NodeTrain loop moves the batch: node pointers + the bag's label space
      // (hot idx -> batch idx).
      capture_prev_nodes(depth_batch);
      if (num_hot > 0) {
        prev_hot_to_full = hot_nodes;
      } else {
        prev_hot_to_full.clear();
      }

      for (int n = 0; n < num_nodes; ++n) {
        auto node_config = internal_config;
        const int32_t k = hot_of_node[n];
        // Hot: hand down the slab, which oblique.cc reads directly. Cold: hand
        // down the pre-sampled projections with no slab, for its cold branch.
        // Mandatory — the driver's main loop would re-consume the RNG.
        node_config.depthwise_projection_defs =
            (k >= 0) ? &hot_projs[k] : &all_node_projs[n];
        node_config.depthwise_monotonic = &all_node_mono[n];
        if (k >= 0) {
          node_config.precomputed_projected_values =
              absl::MakeConstSpan(projected[k]);
        }
        RETURN_IF_ERROR(NodeTrain(train_dataset, config, config_link, dt_config,
                                  deployment, weights, node_config, random,
                                  cache, std::move(depth_batch[n]),
                                  node_queue));
        if (k >= 0) std::vector<float>().swap(projected[k]);
      }
#else
      for (int n = 0; n < num_nodes; ++n) {
        auto node_config = internal_config;
        node_config.depthwise_projection_defs = &all_node_projs[n];
        node_config.depthwise_monotonic = &all_node_mono[n];
        node_config.precomputed_projected_values =
            absl::MakeConstSpan(projected[n]);
        RETURN_IF_ERROR(NodeTrain(train_dataset, config, config_link, dt_config,
                                  deployment, weights, node_config, random,
                                  cache, std::move(depth_batch[n]),
                                  node_queue));
        std::vector<float>().swap(projected[n]);
      }
#endif  // !DW1_COLWALK_CONTROL
    } else
#elif defined(SYMMETRIC_OPTIMIZED)
    if (dt_config.has_sparse_oblique_split() &&
        current_depth >= kMinDepthwiseDepth &&
        current_depth <= SymmetricMaxDepth()) {
      // CatBoost-style symmetric trees: sample K projections ONCE per depth,
      // shared across nodes. The depth's selected examples aggregate to the bag,
      // so the sweep becomes stride-1 instead of a per-node scattered gather.
      const int num_proj = GetNumProjections(
          dt_config, config_link.numerical_features_size());
      const float projection_density = std::clamp(
          dt_config.sparse_oblique_split().projection_density_factor() /
              config_link.numerical_features_size(),
          0.f, 1.f);

      std::vector<internal::Projection> shared_projections(num_proj);
      std::vector<int8_t> shared_monotonic(num_proj, 0);
#if defined(CHRONO_PROFILE) && defined(SYMMETRIC_OPTIMIZED)
      // This depth-level work runs BEFORE the depth's NodeTrains, so cur_depth
      // still holds the previous depth. Pin it so kSampleProjection, the Sym*
      // scopes and kProjectionEvaluate land in the right (tree, depth) cell.
      ::yggdrasil_decision_forests::chrono_prof::tls_ctx.cur_depth =
          current_depth;
      // K shared samples per depth — tiny, but scoped so the depth-level
      // sum (NodeTrain + ApplyProjection + SampleProjection) is complete.
      CHRONO_BEGIN_COARSE(sample_projection);
#endif
      for (int p = 0; p < num_proj; ++p) {
        internal::SampleProjection(
            config_link.numerical_features(), dt_config,
            train_dataset.data_spec(), config_link, projection_density,
            &shared_projections[p], &shared_monotonic[p], random);
      }
#if defined(CHRONO_PROFILE) && defined(SYMMETRIC_OPTIMIZED)
      CHRONO_END_COARSE(sample_projection,
                 ::yggdrasil_decision_forests::chrono_prof::kSampleProjection);
#endif

      const int num_nodes = depth_batch.size();
      std::vector<absl::Span<const UnsignedExampleIdx>> sel_spans(num_nodes);
      for (int n = 0; n < num_nodes; ++n) {
        sel_spans[n] = depth_batch[n].selected_examples.active;
      }

      // Parent->child batch-index mapping for the kernel's incremental
      // sorted-bag relabel (see rebuild_first_child above).
      rebuild_first_child(num_nodes);

      std::vector<std::vector<float>> projected(num_nodes);
      // tls_ctx.cur_depth already pinned to current_depth above (before the
      // shared SampleProjection loop), so Sym* scopes accrue correctly.
      RETURN_IF_ERROR(ApplyProjectionsSymmetricOptimized(
          train_dataset, config_link.numerical_features(),
          absl::MakeConstSpan(sel_spans),
          absl::MakeConstSpan(shared_projections),
          absl::MakeConstSpan(first_child), &depth_bag_state,
          absl::MakeSpan(projected)));

      // Capture this batch's node pointers before the NodeTrain loop moves the
      // batch entries; consumed at the next depth to build first_child.
      capture_prev_nodes(depth_batch);

      // FindBestConditionSparseObliqueTemplate consumes depthwise_projection_defs
      // and depthwise_monotonic (the K shared values, same for every node) plus
      // precomputed_projected_values (this node's slab).
      for (int n = 0; n < num_nodes; ++n) {
        auto node_config = internal_config;
        node_config.depthwise_projection_defs = &shared_projections;
        node_config.depthwise_monotonic = &shared_monotonic;
        // The symmetric kernel only reserves K*rows_n floats in projected[n] and
        // writes through raw pointers to skip the zero-fill, so .size() stays 0:
        // build the span from .data() with the explicit slab length.
        node_config.precomputed_projected_values = absl::MakeConstSpan(
            projected[n].data(),
            shared_projections.size() * sel_spans[n].size());
        RETURN_IF_ERROR(NodeTrain(train_dataset, config, config_link, dt_config,
                                  deployment, weights, node_config, random,
                                  cache, std::move(depth_batch[n]),
                                  node_queue));
        std::vector<float>().swap(projected[n]);  // free early
      }
    } else if (dt_config.has_sparse_oblique_split() &&
               current_depth > SymmetricMaxDepth()) {
      // Symmetric -> DFS handoff past SymmetricMaxDepth(): GrowTreeLocal finishes
      // each frontier node's subtree and pushes nothing back, so the BFS loop
      // drains. No fused kernel ran => invalidate the unadvanced depth bag.
      depth_bag_state.valid = false;
      for (auto& nae : depth_batch) {
        RETURN_IF_ERROR(GrowTreeLocal(
            train_dataset, config, config_link, dt_config, deployment, weights,
            nae.depth, internal_config, nae.constraints,
            nae.set_leaf_already_set, nae.node, random, cache,
            std::move(nae.selected_examples), std::move(nae.leaf_examples)));
      }
    } else
#endif
    {
      // Per-node fallback: BFS order, but stock per-node sampling + Evaluate — no
      // fused Apply, shared projections or slabs. --config=bfs_only isolates it,
      // to attribute the symmetric speedup to traversal order vs the rest.
#ifdef BFS_ONLY
      CHRONO_SCOPE_COARSE(::yggdrasil_decision_forests::chrono_prof::kBfsNodeLoop);
#endif
#if defined(SYMMETRIC_OPTIMIZED) || \
    (defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL))
      // No fused kernel ran here (oblique off, or the DW1 guards excluded this
      // depth), so the bag was not advanced: invalidate it so a later fused depth
      // takes the concat+VQSort fallback instead of a stale chain.
      depth_bag_state.valid = false;
#endif
#if defined(DEPTHWISE_1_PASS) && !defined(DW1_COLWALK_CONTROL)
      prev_hot_to_full.clear();
#endif
// TODO Ariel - make this use GrowTreeLocal.
// Or is GrowTreeLocal even mandatory for Depthwise 1-pass? It has a performance penalty
// CHRONO of this contradicts Randal's assumption of cache friendliness of DFS
      for (auto& nae : depth_batch) {
        RETURN_IF_ERROR(NodeTrain(train_dataset, config, config_link, dt_config,
                                  deployment, weights, internal_config, random,
                                  cache, std::move(nae), node_queue));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ApplyConstraintOnNode(const NodeConstraints& constraint,
                                   NodeWithChildren* node) {
  if (!constraint.min_max_output.has_value()) {
    return absl::OkStatus();
  }
  auto* reg = node->mutable_node()->mutable_regressor();
  STATUS_CHECK(reg->has_top_value());
  reg->set_top_value(std::clamp(reg->top_value(),
                                constraint.min_max_output.value().min,
                                constraint.min_max_output.value().max));
  return absl::OkStatus();
}

absl::Status DivideMonotonicConstraintToChildren(
    const NodeConstraints& constraint, bool direction_increasing,
    const bool check_monotonic, NodeWithChildren* parent_node,
    NodeWithChildren* pos_node, NodeWithChildren* neg_node,
    NodeConstraints* pos_constraint, NodeConstraints* neg_constraint) {
  STATUS_CHECK(parent_node->node().regressor().has_top_value());
  STATUS_CHECK(pos_node->node().regressor().has_top_value());
  STATUS_CHECK(neg_node->node().regressor().has_top_value());

  // TODO: Experiment with other ways to select limit.
  float limit = parent_node->node().regressor().top_value();

  if (check_monotonic) {
    // A failure is indicative of an issue with the splitter i.e. the
    // "FindCondition" function call just before.

    const auto check = [direction_increasing](auto a, auto b) {
      if (direction_increasing) {
        STATUS_CHECK_GE(a, b);
      } else {
        STATUS_CHECK_LE(a, b);
      }
      return absl::OkStatus();
    };

    const float pos_value = pos_node->node().regressor().top_value();
    const float neg_value = neg_node->node().regressor().top_value();
    const float parent_value = parent_node->node().regressor().top_value();
    RETURN_IF_ERROR(check(pos_value, neg_value));
    RETURN_IF_ERROR(check(pos_value, parent_value));
    RETURN_IF_ERROR(check(parent_value, neg_value));
  }
  if ((pos_node->node().regressor().top_value() <
       neg_node->node().regressor().top_value()) == direction_increasing) {
    const float center = (pos_node->node().regressor().top_value() +
                          neg_node->node().regressor().top_value()) /
                         2;
    pos_node->mutable_node()->mutable_regressor()->set_top_value(center);
    neg_node->mutable_node()->mutable_regressor()->set_top_value(center);
    limit = center;
  }

  if (!pos_constraint->min_max_output.has_value()) {
    pos_constraint->min_max_output = NodeConstraints::MinMax();
  }
  if (!neg_constraint->min_max_output.has_value()) {
    neg_constraint->min_max_output = NodeConstraints::MinMax();
  }

  if (direction_increasing) {
    pos_constraint->min_max_output.value().min = limit;
    neg_constraint->min_max_output.value().max = limit;
  } else {
    pos_constraint->min_max_output.value().max = limit;
    neg_constraint->min_max_output.value().min = limit;
  }

  return absl::OkStatus();
}

int8_t MonotonicConstraintSign(
    const model::proto::TrainingConfigLinking& config_link,
    const int attribute_idx) {
  if (config_link.per_columns_size() == 0 || attribute_idx < 0 ||
      attribute_idx >= config_link.per_columns_size()) {
    return 0;
  }
  const auto& link_condition_attribute = config_link.per_columns(attribute_idx);
  if (link_condition_attribute.has_monotonic_constraint()) {
    const bool direction_increasing =
        link_condition_attribute.monotonic_constraint().direction() ==
        model::proto::MonotonicConstraint::INCREASING;
    return direction_increasing ? +1 : -1;
  }
  return 0;
}

namespace internal {

bool MaskPureSampledOrPrunedItemsForCategoricalSetGreedySelection(
    const proto::DecisionTreeTrainingConfig& dt_config,
    int32_t num_attribute_classes,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    const std::vector<int64_t>&
        count_examples_without_weights_by_attribute_class,
    std::vector<bool>* candidate_attributes_bitmap,
    utils::RandomEngine* random) {
  std::uniform_real_distribution<float> sampling_dist;
  int64_t valid_items = 0;
  for (int attr_value = 0; attr_value < num_attribute_classes; attr_value++) {
    if (dt_config.categorical_set_greedy_forward().max_num_items() >= 0 &&
        attr_value >=
            dt_config.categorical_set_greedy_forward().max_num_items()) {
      // Too much candidate items.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (dt_config.categorical_set_greedy_forward().sampling() < 1.f &&
               sampling_dist(*random) >
                   dt_config.categorical_set_greedy_forward().sampling()) {
      // Randomly masked item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else if (count_examples_without_weights_by_attribute_class[attr_value] <
                   dt_config.categorical_set_greedy_forward()
                       .min_item_frequency() ||
               count_examples_without_weights_by_attribute_class[attr_value] >
                   selected_examples.size() -
                       dt_config.categorical_set_greedy_forward()
                           .min_item_frequency()) {
      // Pure item.
      (*candidate_attributes_bitmap)[attr_value] = false;
    } else {
      valid_items++;
    }
  }
  return valid_items > 0;
}

absl::StatusOr<std::vector<float>> GenHistogramBins(
    const proto::NumericalSplit::Type type, const int num_splits,
    const absl::Span<const float> attributes, const float min_value,
    const float max_value, utils::RandomEngine* random) {
  STATUS_CHECK_GE(num_splits, 0);
  std::vector<float> candidate_splits(num_splits);
  switch (type) {
    case proto::NumericalSplit::HISTOGRAM_RANDOM:
    case proto::NumericalSplit::DYNAMIC_RANDOM_HISTOGRAM: {
      std::uniform_real_distribution<float> threshold_distribution(min_value,
                                                                   max_value);
      for (auto& candidate_split : candidate_splits) {
        candidate_split = threshold_distribution(*random);
      }
    } break;
    case proto::NumericalSplit::HISTOGRAM_EQUAL_WIDTH:
    case proto::NumericalSplit::DYNAMIC_EQUAL_WIDTH_HISTOGRAM: {
      for (int split_idx = 0; split_idx < candidate_splits.size();
           split_idx++) {
        candidate_splits[split_idx] = min_value + (max_value - min_value) *
                                                      (split_idx + 0.5f) /
                                                      candidate_splits.size();
      }
    } break;
    default:
      return absl::InvalidArgumentError("Numerical histogram not implemented");
  }
  hwy::VQSort(candidate_splits.data(), candidate_splits.size(),
              hwy::SortAscending());
  return candidate_splits;
}

absl::StatusOr<ExampleSplitRollingBuffer> SplitExamplesInPlace(
    const dataset::VerticalDataset& dataset,
    const SelectedExamplesRollingBuffer examples,
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

  // The following test ensure that the effective number of positive examples is
  // equal to the expected number of positive examples. A miss alignment
  // generally indicates that the splitter does not work correctly.
  //
  // Incorrectly working splitters can make the model worst than expected if
  // the error happens often. If such error happen rarely, the impact is likely
  // insignificant.
  //
  // This generates an error in unit testing and a warning otherwise.
  if (examples_are_training_examples &&
      ABSL_PREDICT_FALSE(
          (example_split.num_positive() !=
           condition.num_pos_training_examples_without_weight()) ||
          (example_split.num_negative() !=
           examples.size() -
               condition.num_pos_training_examples_without_weight()))) {
    const std::string message = absl::Substitute(
        "[This message is only shown once] The effective split of examples "
        "does not match the expected split "
        "returned by the splitter algorithm. This problem can be caused by (1) "
        "large floating point values (e.g. value>=10e30) or (2) a bug in the "
        "software. You can turn this error in a warning with "
        "internal_error_on_wrong_splitter_statistics=false.\n\nDetails:\n"
        "Num examples: $0\n"
        "Num positive examples (from split evaluation): $1\n"
        "Num positive examples (from split learning): $4\n"
        "Num negative examples (from split evaluation): $2\n"
        "\n"
        "Condition: $3\n"
        "Attribute spec: $5",
        /*$0*/ examples.size(),
        /*$1*/ example_split.num_positive(),
        /*$2*/ example_split.num_negative(),
        /*$3*/ condition.DebugString(),
        /*$4*/ condition.num_pos_training_examples_without_weight(),
        /*$5*/
        dataset.data_spec().columns(condition.attribute()).DebugString());
    if (error_on_wrong_splitter_statistics) {
      return absl::InternalError(message);
    } else {
      // Logging this message too often will crash.
      LOG_FIRST_N(WARNING, 1) << message;
    }
  }
  return example_split;
}

}  // namespace internal

}  // namespace yggdrasil_decision_forests::model::decision_tree
