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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/oblique.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique.h"
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_scanner.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

using decision_tree::internal::Projection;

// One term of a projection resolved to its dense column.
struct ColumnAndWeight {
  const float* column;
  float weight;
};

template <typename Initializer>
absl::StatusOr<std::vector<Initializer>> CreateInitializers(
    const ObliqueSplitArgs& args,
    absl::StatusOr<Initializer> (AbstractLabelAccessor::*creator)(
        const decision_tree::proto::LabelStatistics&) const) {
  std::vector<Initializer> initializers;
  initializers.reserve(args.label_stats_per_open_node.size());
  for (const auto& label_stats : args.label_stats_per_open_node) {
    ASSIGN_OR_RETURN(auto initializer, (args.label_accessor.*creator)(label_stats));
    initializers.push_back(std::move(initializer));
  }
  return initializers;
}

// Best oblique split of one node. "split" holds the current best (possibly
// axis-aligned) split of the node and is only overwritten by a higher score.
template <typename LabelFiller>
absl::Status FindBestObliqueSplitForNode(
    const ObliqueSplitArgs& args, const int node_idx,
    const absl::Span<const ExampleIndex> rows, const LabelFiller& label_filler,
    const typename LabelFiller::AccumulatorInitializer& initializer,
    const ExampleIndex min_num_examples, Split* split) {
  const auto& features = args.numerical_features_per_open_node[node_idx];
  const auto& label_stats = args.label_stats_per_open_node[node_idx];
  const ExampleIndex num_examples = rows.size();
  if (num_examples != label_stats.num_examples()) {
    return absl::InternalError(absl::Substitute(
        "Node $0 holds $1 examples but its statistics count $2", node_idx,
        num_examples, label_stats.num_examples()));
  }
  if (features.empty() || num_examples < 2 ||
      num_examples < 2 * min_num_examples) {
    return absl::OkStatus();
  }

  const auto& oblique_config = args.dt_config.sparse_oblique_split();
  const int count_basis = args.num_features_for_projection_count > 0
                              ? args.num_features_for_projection_count
                              : static_cast<int>(features.size());
  const int num_projections =
      decision_tree::GetNumProjections(args.dt_config, count_basis);
  // Density over the LOCAL candidates keeps E[nnz] = density_factor.
  const float projection_density =
      std::clamp(oblique_config.projection_density_factor() /
                     static_cast<float>(features.size()),
                 0.f, 1.f);

  std::seed_seq seed_seq{static_cast<uint32_t>(args.seed),
                         static_cast<uint32_t>(args.seed >> 32),
                         static_cast<uint32_t>(node_idx)};
  utils::RandomEngine random(seed_seq);

  std::vector<std::pair<float, ExampleIndex>> value_and_row(num_examples);
  std::vector<ColumnAndWeight> terms;
  Projection projection;
  int8_t monotonic_direction;

  typename LabelFiller::Accumulator pos, neg;
  double best_score = split->condition.split_score();
  bool found = false;
  Projection best_projection;
  float best_threshold = std::numeric_limits<float>::quiet_NaN();
  ExampleIndex best_num_pos = 0;
  typename LabelFiller::Accumulator best_pos;
  double weighted_num_examples = 0;

  for (int projection_idx = 0; projection_idx < num_projections;
       projection_idx++) {
    decision_tree::internal::SampleProjection(
        features, args.dt_config, args.dataspec, args.config_link,
        projection_density, &projection, &monotonic_direction, &random);

    terms.clear();
    for (const auto& item : projection) {
      const float* column = args.columns.data(item.attribute_idx);
      if (column == nullptr) {
        return absl::InternalError(absl::StrCat(
            "Column ", item.attribute_idx, " is not materialized"));
      }
      terms.push_back({column, item.weight});
    }

    // ApplyProjection: rows outer, terms inner, scalar fp32 accumulator. Same
    // order as ProjectionEvaluator::Evaluate and the model's EvalCondition.
    for (ExampleIndex i = 0; i < num_examples; i++) {
      const ExampleIndex row = rows[i];
      float value = 0;
      for (const auto& term : terms) {
        value += term.column[row] * term.weight;
      }
      value_and_row[i] = {value, row};
    }
    std::sort(value_and_row.begin(), value_and_row.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Exact scan, mirroring FillNumericalAccumulator: all examples start on
    // the positive side and move to the negative side in value order.
    initializer.InitFull(&pos);
    initializer.InitEmpty(&neg);
    weighted_num_examples = pos.WeightedNumExamples();
    ExampleIndex num_pos = num_examples;
    const ExampleIndex max_num_pos = num_examples - min_num_examples;
    for (ExampleIndex i = 0; i < num_examples; i++) {
      const float value = value_and_row[i].first;
      const ExampleIndex row = value_and_row[i].second;
      if (i > 0 && value != value_and_row[i - 1].first &&
          num_pos >= min_num_examples && num_pos <= max_num_pos) {
        const double score = decision_tree::Score<>(
            initializer, weighted_num_examples, pos, neg);
        if (score > best_score) {
          best_score = score;
          found = true;
          best_projection = projection;
          best_threshold =
              decision_tree::MidThreshold(value_and_row[i - 1].first, value);
          best_num_pos = num_pos;
          best_pos = pos;
        }
      }
      label_filler.Add(row, &neg);
      label_filler.Sub(row, &pos);
      num_pos--;
    }
  }

  if (!found) {
    return absl::OkStatus();
  }

  RETURN_IF_ERROR(decision_tree::internal::SetCondition(
      best_projection, best_threshold, args.dataspec, &split->condition));
  split->condition.set_num_pos_training_examples_without_weight(best_num_pos);
  split->condition.set_num_pos_training_examples_with_weight(
      best_pos.WeightedNumExamples());
  split->condition.set_num_training_examples_without_weight(num_examples);
  split->condition.set_num_training_examples_with_weight(
      weighted_num_examples);
  split->condition.set_split_score(best_score);

  typename LabelFiller::Accumulator best_neg;
  best_neg.ImportLabelStats(label_stats);
  best_pos.Sub(&best_neg);
  best_neg.ExportLabelStats(&split->label_statistics[0]);
  best_pos.ExportLabelStats(&split->label_statistics[1]);
  split->label_statistics[0].set_num_examples(num_examples - best_num_pos);
  split->label_statistics[1].set_num_examples(best_num_pos);
  return absl::OkStatus();
}

template <typename LabelFiller>
absl::Status TemplatedFindBestObliqueSplits(
    const ObliqueSplitArgs& args, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        initializers,
    const ExampleIndex min_num_examples,
    utils::concurrency::ThreadPool* thread_pool,
    SplitPerOpenNode* best_splits) {
  const int num_nodes = args.numerical_features_per_open_node.size();
  const auto rows_per_node = BuildRowsPerNode(args.example_to_node, num_nodes);

  int num_jobs = 0;
  for (const auto& features : args.numerical_features_per_open_node) {
    num_jobs += !features.empty();
  }
  if (num_jobs == 0) {
    return absl::OkStatus();
  }

  utils::concurrency::Mutex mutex;
  absl::Status status;
  utils::concurrency::BlockingCounter counter(num_jobs);
  for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
    if (args.numerical_features_per_open_node[node_idx].empty()) {
      continue;
    }
    thread_pool->Schedule([&, node_idx]() {
      Split local_split;
      {
        utils::concurrency::MutexLock l(mutex);
        local_split = (*best_splits)[node_idx];
      }
      const auto local_status = FindBestObliqueSplitForNode(
          args, node_idx, rows_per_node.rows(node_idx), label_filler,
          initializers[node_idx], min_num_examples, &local_split);
      {
        utils::concurrency::MutexLock l(mutex);
        status.Update(local_status);
        if (local_status.ok() &&
            local_split.condition.split_score() >
                (*best_splits)[node_idx].condition.split_score()) {
          (*best_splits)[node_idx] = std::move(local_split);
        }
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();
  utils::concurrency::MutexLock l(mutex);
  return status;
}

}  // namespace

absl::StatusOr<std::vector<float>> MaterializeNumericalColumn(
    const dataset_cache::DatasetCacheReader& dataset, const int feature) {
  if (!dataset.has_feature(feature)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Feature ", feature, " is not owned by this worker"));
  }
  const auto& column_meta = dataset.meta_data().columns(feature);
  if (!column_meta.has_numerical()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Feature ", feature, " is not numerical"));
  }
  const float replacement = column_meta.numerical().replacement_missing_value();
  std::vector<float> values;
  values.reserve(dataset.num_examples());

  if (column_meta.numerical().discretized()) {
    const auto& boundaries =
        dataset.DiscretizedNumericalFeatureBoundaries(feature);
    ASSIGN_OR_RETURN(
        auto it, dataset.InOrderDiscretizedNumericalFeatureValueIterator(feature));
    while (true) {
      RETURN_IF_ERROR(it->Next());
      const auto chunk = it->Values();
      if (chunk.empty()) {
        break;
      }
      for (const auto discretized_value : chunk) {
        if (discretized_value == dataset::kDiscretizedNumericalMissingValue ||
            boundaries.empty()) {
          values.push_back(replacement);
        } else {
          values.push_back(dataset_cache::DiscretizedNumericalToNumerical(
              boundaries, discretized_value));
        }
      }
    }
    RETURN_IF_ERROR(it->Close());
  } else {
    ASSIGN_OR_RETURN(auto it, dataset.InOrderNumericalFeatureValueIterator(feature));
    while (true) {
      RETURN_IF_ERROR(it->Next());
      const auto chunk = it->Values();
      if (chunk.empty()) {
        break;
      }
      for (const float value : chunk) {
        values.push_back(std::isnan(value) ? replacement : value);
      }
    }
    RETURN_IF_ERROR(it->Close());
  }

  if (values.size() != dataset.num_examples()) {
    return absl::InternalError(absl::Substitute(
        "Feature $0 has $1 values for $2 examples", feature, values.size(),
        dataset.num_examples()));
  }
  return values;
}

absl::Status DenseNumericalColumns::Prepare(
    const absl::Span<const int> features,
    utils::concurrency::ThreadPool* thread_pool) {
  std::vector<int> missing;
  for (const int feature : features) {
    if (!columns_.contains(feature)) {
      missing.push_back(feature);
    }
  }
  std::sort(missing.begin(), missing.end());
  missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
  if (missing.empty()) {
    return absl::OkStatus();
  }

  std::vector<absl::StatusOr<std::vector<float>>> results(missing.size());
  if (thread_pool == nullptr || missing.size() == 1) {
    for (size_t i = 0; i < missing.size(); i++) {
      results[i] = MaterializeNumericalColumn(*dataset_, missing[i]);
    }
  } else {
    utils::concurrency::BlockingCounter counter(missing.size());
    for (size_t i = 0; i < missing.size(); i++) {
      thread_pool->Schedule([&, i]() {
        results[i] = MaterializeNumericalColumn(*dataset_, missing[i]);
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }

  for (size_t i = 0; i < missing.size(); i++) {
    RETURN_IF_ERROR(results[i].status());
    columns_[missing[i]] =
        std::make_unique<std::vector<float>>(std::move(results[i]).value());
  }
  return absl::OkStatus();
}

const float* DenseNumericalColumns::data(const int feature) const {
  const auto it = columns_.find(feature);
  if (it == columns_.end()) {
    return nullptr;
  }
  return it->second->data();
}

size_t DenseNumericalColumns::MemoryUsage() const {
  size_t usage = 0;
  for (const auto& column : columns_) {
    usage += column.second->size() * sizeof(float);
  }
  return usage;
}

RowsPerNode BuildRowsPerNode(const ExampleToNodeMap& example_to_node,
                             const int num_open_nodes) {
  RowsPerNode out;
  out.begin.assign(num_open_nodes + 1, 0);
  for (const auto node_idx : example_to_node) {
    if (node_idx == kClosedNode) {
      continue;
    }
    DCHECK_LT(node_idx, num_open_nodes);
    out.begin[node_idx + 1]++;
  }
  for (int node_idx = 0; node_idx < num_open_nodes; node_idx++) {
    out.begin[node_idx + 1] += out.begin[node_idx];
  }
  out.flat.resize(out.begin.back());
  std::vector<size_t> cursor(out.begin.begin(), out.begin.end() - 1);
  for (ExampleIndex example_idx = 0; example_idx < example_to_node.size();
       example_idx++) {
    const auto node_idx = example_to_node[example_idx];
    if (node_idx == kClosedNode) {
      continue;
    }
    out.flat[cursor[node_idx]++] = example_idx;
  }
  return out;
}

absl::Status FindBestObliqueSplits(const ObliqueSplitArgs& args,
                                   utils::concurrency::ThreadPool* thread_pool,
                                   SplitPerOpenNode* best_splits) {
  const size_t num_nodes = args.numerical_features_per_open_node.size();
  if (best_splits->size() != num_nodes ||
      args.label_stats_per_open_node.size() != num_nodes) {
    return absl::InternalError("Wrong number of open nodes");
  }
  if (!args.dt_config.has_sparse_oblique_split()) {
    return absl::InvalidArgumentError("sparse_oblique_split is not set");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError("A thread pool is required");
  }
  const ExampleIndex min_num_examples = args.dt_config.min_examples();

  switch (args.task) {
    case model::proto::CLASSIFICATION: {
      if (args.label_accessor_type != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(const auto filler,
                       args.label_accessor.CreateClassificationLabelFiller());
      ASSIGN_OR_RETURN(
          const auto initializers,
          CreateInitializers<>(args, &AbstractLabelAccessor::
                                         CreateClassificationAccumulatorInitializer));
      return TemplatedFindBestObliqueSplits<>(args, filler, initializers,
                                              min_num_examples, thread_pool,
                                              best_splits);
    }

    case model::proto::REGRESSION: {
      switch (args.label_accessor_type) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(const auto filler,
                           args.label_accessor.CreateRegressionLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateInitializers<>(
                  args,
                  &AbstractLabelAccessor::CreateRegressionAccumulatorInitializer));
          return TemplatedFindBestObliqueSplits<>(args, filler, initializers,
                                                  min_num_examples, thread_pool,
                                                  best_splits);
        }
        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              const auto filler,
              args.label_accessor.CreateRegressionWithHessianLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateInitializers<>(
                  args, &AbstractLabelAccessor::
                            CreateRegressionWithHessianAccumulatorInitializer));
          return TemplatedFindBestObliqueSplits<>(args, filler, initializers,
                                                  min_num_examples, thread_pool,
                                                  best_splits);
        }
        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }

    default:
      return absl::InvalidArgumentError(absl::Substitute(
          "The task $0 is not supported for in-worker oblique splits. The "
          "supported tasks are CLASSIFICATION, REGRESSION.",
          model::proto::Task_Name(args.task)));
  }
}

absl::Status EvaluateObliqueSplits(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    const dataset_cache::DatasetCacheReader& dataset,
    DenseNumericalColumns* columns, utils::concurrency::ThreadPool* thread_pool,
    SplitEvaluationPerOpenNode* split_evaluation) {
  if (split_evaluation->size() != splits.size()) {
    return absl::InternalError("Wrong number of split evaluations");
  }

  // Oblique splits proposed by this worker (it owns their first attribute).
  std::vector<int> node_idx_to_active_node_idx(splits.size(), -1);
  std::vector<int> active_node_idxs;
  std::vector<int> needed_features;
  for (int split_idx = 0; split_idx < splits.size(); split_idx++) {
    const auto& split = splits[split_idx];
    if (!IsSplitValid(split) ||
        !split.condition.condition().has_oblique_condition()) {
      continue;
    }
    const auto& oblique = split.condition.condition().oblique_condition();
    if (oblique.attributes_size() == 0 ||
        !dataset.has_feature(oblique.attributes(0))) {
      continue;
    }
    for (const int attribute : oblique.attributes()) {
      if (!dataset.has_feature(attribute)) {
        return absl::InternalError(absl::Substitute(
            "This worker owns attribute $0 of an oblique split but not "
            "attribute $1. Feature ownership changed after the split was "
            "found; dynamic load balancing is not supported with in-worker "
            "oblique splits.",
            oblique.attributes(0), attribute));
      }
      needed_features.push_back(attribute);
    }
    node_idx_to_active_node_idx[split_idx] = active_node_idxs.size();
    active_node_idxs.push_back(split_idx);
  }
  if (active_node_idxs.empty()) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(columns->Prepare(needed_features, thread_pool));

  struct ActiveNode {
    utils::bitmap::BitWriter writer;
    std::vector<ColumnAndWeight> terms;
    float threshold;
    size_t num_elements;
  };
  std::vector<ActiveNode> active_nodes;
  active_nodes.reserve(active_node_idxs.size());
  for (const auto node_idx : active_node_idxs) {
    const auto& condition = splits[node_idx].condition;
    const auto& oblique = condition.condition().oblique_condition();
    std::vector<ColumnAndWeight> terms;
    terms.reserve(oblique.attributes_size());
    for (int item_idx = 0; item_idx < oblique.attributes_size(); item_idx++) {
      terms.push_back(
          {columns->data(oblique.attributes(item_idx)), oblique.weights(item_idx)});
      DCHECK(terms.back().column != nullptr);
    }
    const auto num_elements = static_cast<size_t>(
        condition.num_training_examples_without_weight());
    ActiveNode active_node{
        /*writer=*/{num_elements, &(*split_evaluation)[node_idx]},
        /*terms=*/std::move(terms),
        /*threshold=*/oblique.threshold(),
        /*num_elements=*/num_elements,
    };
    active_node.writer.AllocateAndZeroBitMap();
    active_nodes.push_back(std::move(active_node));
  }

  // One pass over the examples; each active node recomputes its projection.
  for (ExampleIndex example_idx = 0; example_idx < example_to_node.size();
       example_idx++) {
    const auto node_idx = example_to_node[example_idx];
    if (node_idx == kClosedNode) {
      continue;
    }
    const auto active_node_idx = node_idx_to_active_node_idx[node_idx];
    if (active_node_idx < 0) {
      continue;
    }
    auto& active_node = active_nodes[active_node_idx];
    float value = 0;
    for (const auto& term : active_node.terms) {
      value += term.column[example_idx] * term.weight;
    }
    active_node.writer.Write(value >= active_node.threshold);
  }

  for (auto& active_node : active_nodes) {
    active_node.writer.Finish();
  }
  return absl::OkStatus();
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
