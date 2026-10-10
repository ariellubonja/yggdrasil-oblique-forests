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

// In-worker sparse oblique splits for feature-distributed tree training.
//
// Each worker owns a subset of the columns. Instead of scoring its columns one
// at a time (axis-aligned), the worker samples sparse random projections of
// its OWN columns for every open node, scores the projected values with the
// same exact scanner and label accumulators as the axis-aligned splitter, and
// returns the best oblique condition per node. Because every attribute of a
// projection lives on the proposing worker, the split evaluation (bitmaps) and
// the bitmap sharing are unchanged: no extra communication versus the
// axis-aligned protocol. Projections never cross workers.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_OBLIQUE_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_OBLIQUE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_reader.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/label_accessor.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

// In-order fp32 copies of the numerical columns a worker owns. The dataset
// cache keeps numerical columns pre-sorted or discretized; the projection
// kernel needs plain col[row] gathers, so columns are materialized once here.
class DenseNumericalColumns {
 public:
  explicit DenseNumericalColumns(
      const dataset_cache::DatasetCacheReader* dataset)
      : dataset_(dataset) {}

  // Materializes the columns of "features" that are not loaded yet.
  absl::Status Prepare(absl::Span<const int> features,
                       utils::concurrency::ThreadPool* thread_pool);

  // Values of "feature" ordered by example index. nullptr if not prepared.
  const float* data(int feature) const;

  bool has(int feature) const { return columns_.contains(feature); }
  void Clear() { columns_.clear(); }
  size_t num_examples() const { return dataset_->num_examples(); }
  size_t MemoryUsage() const;

 private:
  const dataset_cache::DatasetCacheReader* dataset_;
  absl::flat_hash_map<int, std::unique_ptr<std::vector<float>>> columns_;
};

// Reads one numerical column of the cache in example order. Discretized
// columns are mapped back through their bin boundaries (lossy); missing values
// take the cache's replacement value.
absl::StatusOr<std::vector<float>> MaterializeNumericalColumn(
    const dataset_cache::DatasetCacheReader& dataset, int feature);

// Arguments of the in-worker sparse oblique split search.
struct ObliqueSplitArgs {
  // Candidate NUMERICAL features of each open node, all owned by the worker.
  const std::vector<std::vector<FeatureIndex>>& numerical_features_per_open_node;
  const ExampleToNodeMap& example_to_node;
  const dataset::proto::DataSpecification& dataspec;
  const model::proto::TrainingConfigLinking& config_link;
  const decision_tree::proto::DecisionTreeTrainingConfig& dt_config;
  const AbstractLabelAccessor& label_accessor;
  const LabelStatsPerNode& label_stats_per_open_node;
  const DenseNumericalColumns& columns;
  model::proto::Task task;
  LabelAccessorType label_accessor_type;
  // Feature count fed to GetNumProjections. <= 0: the node's own candidate
  // count. The projection density always uses the node's own candidate count.
  int num_features_for_projection_count;
  // Base seed. Each node derives an independent random stream from it.
  uint64_t seed;
};

// For each open node, samples projections from the node's local numerical
// features, scores them like the axis-aligned distributed splitter and keeps
// the best in "best_splits" (an existing split with a higher score is kept).
// "best_splits" must hold one entry per open node.
absl::Status FindBestObliqueSplits(const ObliqueSplitArgs& args,
                                   utils::concurrency::ThreadPool* thread_pool,
                                   SplitPerOpenNode* best_splits);

// Writes the evaluation bitmaps of the oblique splits whose first attribute is
// owned by this worker. All attributes must then be owned (projections never
// cross workers). Non-oblique and foreign splits are left untouched.
absl::Status EvaluateObliqueSplits(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    const dataset_cache::DatasetCacheReader& dataset,
    DenseNumericalColumns* columns, utils::concurrency::ThreadPool* thread_pool,
    SplitEvaluationPerOpenNode* split_evaluation);

// Example indices of each open node (ascending), built from example_to_node.
struct RowsPerNode {
  std::vector<ExampleIndex> flat;
  std::vector<size_t> begin;  // One more entry than nodes.
  absl::Span<const ExampleIndex> rows(int node_idx) const {
    return absl::MakeConstSpan(flat).subspan(
        begin[node_idx], begin[node_idx + 1] - begin[node_idx]);
  }
};
RowsPerNode BuildRowsPerNode(const ExampleToNodeMap& example_to_node,
                             int num_open_nodes);

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_OBLIQUE_H_
