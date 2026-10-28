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
// Naive baseline (in_worker_oblique.ship_foreign_columns, worker.cc): the
// worker samples from ALL numerical features and fetches the columns it does
// not own from their owners at every layer. Those shipped columns enter
// DenseNumericalColumns as "external" columns; the kernel below is unchanged.
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
#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"
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
//
// "External" columns are copies received from other workers (column
// shipping). They live next to the owned ones but in their own map, so the
// owned map can be read concurrently by the thread serving GetColumns while
// the request thread swaps external columns in and out. Thread safety rests on
// the owned map being fully materialized before any shipping starts (the
// worker does so in StartTraining) and never written afterwards.
class DenseNumericalColumns {
 public:
  explicit DenseNumericalColumns(
      const dataset_cache::DatasetCacheReader* dataset)
      : dataset_(dataset) {}

  // Materializes the owned columns of "features" that are not loaded yet.
  // Features already present as external columns are skipped.
  absl::Status Prepare(absl::Span<const int> features,
                       utils::concurrency::ThreadPool* thread_pool);

  // Values of "feature" ordered by example index (owned, then external).
  // nullptr if not present.
  const float* data(int feature) const;

  // Same, owned columns only. Safe to call from another thread while
  // external columns are being modified.
  const float* owned_data(int feature) const {
    const auto it = columns_.find(feature);
    return it == columns_.end() ? nullptr : it->second->data();
  }

  // Stores a column received from another worker. "values" holds one value
  // per example in example order.
  void SetExternal(int feature, std::vector<float> values);
  void ClearExternal() { external_.clear(); }
  size_t num_external() const { return external_.size(); }
  bool is_external(int feature) const { return external_.contains(feature); }

  bool has(int feature) const {
    return columns_.contains(feature) || external_.contains(feature);
  }
  void Clear() {
    columns_.clear();
    external_.clear();
  }
  size_t num_examples() const { return dataset_->num_examples(); }
  size_t MemoryUsage() const;

 private:
  const dataset_cache::DatasetCacheReader* dataset_;
  absl::flat_hash_map<int, std::unique_ptr<std::vector<float>>> columns_;
  absl::flat_hash_map<int, std::unique_ptr<std::vector<float>>> external_;
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
  // Optional: the projections of every open node, as SampleNodeProjections
  // returns them (one entry per open node, inactive nodes may be empty). When
  // set, the finder scores these instead of sampling; the result is the same
  // since sampling is deterministic in (seed, node). Lets a caller learn which
  // columns a layer needs before scoring (column shipping).
  const std::vector<std::vector<decision_tree::internal::Projection>>*
      presampled_projections = nullptr;
};

// Samples the projections of one open node, in order, from the node's own
// random stream (derived from args.seed and the node index). Returns an empty
// vector for a node without numerical candidates.
std::vector<decision_tree::internal::Projection> SampleNodeProjections(
    const ObliqueSplitArgs& args, int node_idx);

// For each open node, samples projections from the node's local numerical
// features, scores them like the axis-aligned distributed splitter and keeps
// the best in "best_splits" (an existing split with a higher score is kept).
// "best_splits" must hold one entry per open node.
absl::Status FindBestObliqueSplits(const ObliqueSplitArgs& args,
                                   utils::concurrency::ThreadPool* thread_pool,
                                   SplitPerOpenNode* best_splits);

// Writes the evaluation bitmaps of the oblique splits whose first attribute is
// owned by this worker. Every other attribute must be owned too, or present
// as an external (shipped) column of "columns". Non-oblique and foreign splits
// are left untouched.
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
