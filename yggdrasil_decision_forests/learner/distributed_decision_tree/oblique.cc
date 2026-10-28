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
#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort.h"
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

// Sort record of the exact scan: one projected value and its example index.
//
// Default: the value is packed as a sort-monotone unsigned key next to the
// row and sorted with Highway VQSort, as the single-machine finder does
// (FlatBucketTraits<..., hwy::K32V32> in splitter_scanner.h). Rows with equal
// values may come out in a different order than with std::sort; the scan only
// evaluates thresholds between distinct values, so the candidate splits are
// the same, but the label accumulators then sum the tied rows in another
// order and can differ in their last bits.
//
// --config=exact_std_sort keeps the pre-VQSort std::sort on (value, row)
// pairs (both here and in the single-machine finder), i.e. the exact
// arithmetic of the serial implementation.
#ifdef EXACT_STD_SORT
struct SortRecordTraits {
  using Record = std::pair<float, ExampleIndex>;
  static Record Pack(const float value, const ExampleIndex row) {
    return {value, row};
  }
  static float Value(const Record& record) { return record.first; }
  static ExampleIndex Row(const Record& record) { return record.second; }
  static void Sort(Record* const begin, const size_t num_records) {
    std::sort(begin, begin + num_records,
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
};
#else
template <size_t kIndexBytes>
struct HighwaySortRecordTraits;

template <>
struct HighwaySortRecordTraits<4> {
  using Record = hwy::K32V32;
  static Record Pack(const float value, const ExampleIndex row) {
    Record record;
    record.key = decision_tree::internal::FloatToUintForSort(value);
    record.value = static_cast<uint32_t>(row);
    return record;
  }
  static float Value(const Record& record) {
    return decision_tree::internal::UintToFloatFromSort(record.key);
  }
  static ExampleIndex Row(const Record& record) {
    return static_cast<ExampleIndex>(record.value);
  }
  static void Sort(Record* const begin, const size_t num_records) {
    hwy::VQSort(begin, num_records, hwy::SortAscending());
  }
};

template <>
struct HighwaySortRecordTraits<8> {
  using Record = hwy::K64V64;
  static Record Pack(const float value, const ExampleIndex row) {
    Record record;
    record.key = decision_tree::internal::FloatToUintForSort(value);
    record.value = static_cast<uint64_t>(row);
    return record;
  }
  static float Value(const Record& record) {
    return decision_tree::internal::UintToFloatFromSort(
        static_cast<uint32_t>(record.key));
  }
  static ExampleIndex Row(const Record& record) {
    return static_cast<ExampleIndex>(record.value);
  }
  static void Sort(Record* const begin, const size_t num_records) {
    hwy::VQSort(begin, num_records, hwy::SortAscending());
  }
};

using SortRecordTraits = HighwaySortRecordTraits<sizeof(ExampleIndex)>;
#endif  // EXACT_STD_SORT

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

}  // namespace

// Sampling is the only consumer of the node's stream, so sampling all the
// projections up front and scoring them in any order / on any thread yields
// the projections the serial loop would have sampled.
std::vector<Projection> SampleNodeProjections(const ObliqueSplitArgs& args,
                                              const int node_idx) {
  const auto& features = args.numerical_features_per_open_node[node_idx];
  if (features.empty()) {
    return {};
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

  std::vector<Projection> projections(std::max(num_projections, 0));
  int8_t monotonic_direction;
  for (auto& projection : projections) {
    decision_tree::internal::SampleProjection(
        features, args.dt_config, args.dataspec, args.config_link,
        projection_density, &projection, &monotonic_direction, &random);
  }
  return projections;
}

namespace {

// Best split found by one job over a range of a node's projections.
template <typename Accumulator>
struct ObliqueChunkResult {
  bool found = false;
  double score = 0;
  // Index of the winning projection in the node's sampling order.
  int projection_idx = -1;
  float threshold = std::numeric_limits<float>::quiet_NaN();
  ExampleIndex num_pos = 0;
  Accumulator pos;
  double weighted_num_examples = 0;
};

// Scores the projections [first_projection_idx, first_projection_idx +
// projections.size()) of a node. "initial_score" is the score to beat (the
// node's current, possibly axis-aligned, split). Within the chunk the serial
// rule applies: a candidate replaces the best only with a strictly higher
// score, so the first (projection, threshold) reaching the chunk maximum
// wins.
template <typename LabelFiller>
absl::Status ScoreProjectionChunk(
    const ObliqueSplitArgs& args, const absl::Span<const ExampleIndex> rows,
    const absl::Span<const Projection> projections,
    const int first_projection_idx, const LabelFiller& label_filler,
    const typename LabelFiller::AccumulatorInitializer& initializer,
    const ExampleIndex min_num_examples, const double initial_score,
    ObliqueChunkResult<typename LabelFiller::Accumulator>* result) {
  using Traits = SortRecordTraits;
  const ExampleIndex num_examples = rows.size();
  DCHECK_GE(num_examples, 2);
  DCHECK_GE(num_examples, 2 * min_num_examples);

  std::vector<typename Traits::Record> records(num_examples);
  std::vector<ColumnAndWeight> terms;
  typename LabelFiller::Accumulator pos, neg;
  double best_score = initial_score;
  result->found = false;

  for (int local_idx = 0; local_idx < projections.size(); local_idx++) {
    const Projection& projection = projections[local_idx];

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
      records[i] = Traits::Pack(value, row);
    }
    Traits::Sort(records.data(), num_examples);

    // Exact scan, mirroring FillNumericalAccumulator: all examples start on
    // the positive side and move to the negative side in value order.
    initializer.InitFull(&pos);
    initializer.InitEmpty(&neg);
    const double weighted_num_examples = pos.WeightedNumExamples();
    ExampleIndex num_pos = num_examples;
    const ExampleIndex max_num_pos = num_examples - min_num_examples;
    float previous_value = 0;
    for (ExampleIndex i = 0; i < num_examples; i++) {
      const float value = Traits::Value(records[i]);
      const ExampleIndex row = Traits::Row(records[i]);
      if (i > 0 && value != previous_value && num_pos >= min_num_examples &&
          num_pos <= max_num_pos) {
        const double score = decision_tree::Score<>(
            initializer, weighted_num_examples, pos, neg);
        if (score > best_score) {
          best_score = score;
          result->found = true;
          result->score = score;
          result->projection_idx = first_projection_idx + local_idx;
          result->threshold = decision_tree::MidThreshold(previous_value, value);
          result->num_pos = num_pos;
          result->pos = pos;
          result->weighted_num_examples = weighted_num_examples;
        }
      }
      label_filler.Add(row, &neg);
      label_filler.Sub(row, &pos);
      num_pos--;
      previous_value = value;
    }
  }
  return absl::OkStatus();
}

// Writes the winning projection of a node into "split" if its score beats the
// split already there (same float comparison as the serial implementation).
template <typename Accumulator>
absl::Status CommitObliqueSplit(const ObliqueSplitArgs& args,
                                const int node_idx,
                                const Projection& projection,
                                ObliqueChunkResult<Accumulator> result,
                                const ExampleIndex num_examples, Split* split) {
  Split candidate = *split;
  RETURN_IF_ERROR(decision_tree::internal::SetCondition(
      projection, result.threshold, args.dataspec, &candidate.condition));
  candidate.condition.set_num_pos_training_examples_without_weight(
      result.num_pos);
  candidate.condition.set_num_pos_training_examples_with_weight(
      result.pos.WeightedNumExamples());
  candidate.condition.set_num_training_examples_without_weight(num_examples);
  candidate.condition.set_num_training_examples_with_weight(
      result.weighted_num_examples);
  candidate.condition.set_split_score(result.score);

  Accumulator best_neg;
  best_neg.ImportLabelStats(args.label_stats_per_open_node[node_idx]);
  result.pos.Sub(&best_neg);
  best_neg.ExportLabelStats(&candidate.label_statistics[0]);
  result.pos.ExportLabelStats(&candidate.label_statistics[1]);
  candidate.label_statistics[0].set_num_examples(num_examples -
                                                 result.num_pos);
  candidate.label_statistics[1].set_num_examples(result.num_pos);

  if (candidate.condition.split_score() > split->condition.split_score()) {
    *split = std::move(candidate);
  }
  return absl::OkStatus();
}

// Scores every open node's projections on the thread pool.
//
// Jobs are (node, projection range) pairs rather than one job per node: a
// layer with few open nodes (the root above all) would otherwise keep a
// single thread busy per worker. Every node's projections are pre-sampled on
// the calling thread, so the result does not depend on the chunking or on the
// scheduling: per node the chunk results are reduced by score, ties going to
// the lowest projection index, which is what the serial loop's strict ">"
// in sampling order selects.
template <typename LabelFiller>
absl::Status TemplatedFindBestObliqueSplits(
    const ObliqueSplitArgs& args, const LabelFiller& label_filler,
    const std::vector<typename LabelFiller::AccumulatorInitializer>&
        initializers,
    const ExampleIndex min_num_examples,
    utils::concurrency::ThreadPool* thread_pool,
    SplitPerOpenNode* best_splits) {
  using Accumulator = typename LabelFiller::Accumulator;
  const int num_nodes = args.numerical_features_per_open_node.size();
  const auto rows_per_node = BuildRowsPerNode(args.example_to_node, num_nodes);
  if (args.presampled_projections != nullptr &&
      args.presampled_projections->size() != num_nodes) {
    return absl::InternalError("Wrong number of pre-sampled projection sets");
  }

  // Nodes with something to score, and their projections.
  std::vector<int> active_nodes;
  std::vector<std::vector<Projection>> projections_per_node(num_nodes);
  for (int node_idx = 0; node_idx < num_nodes; node_idx++) {
    if (args.numerical_features_per_open_node[node_idx].empty()) {
      continue;
    }
    const ExampleIndex num_examples = rows_per_node.rows(node_idx).size();
    const auto& label_stats = args.label_stats_per_open_node[node_idx];
    if (num_examples != label_stats.num_examples()) {
      return absl::InternalError(absl::Substitute(
          "Node $0 holds $1 examples but its statistics count $2", node_idx,
          num_examples, label_stats.num_examples()));
    }
    if (num_examples < 2 || num_examples < 2 * min_num_examples) {
      continue;
    }
    projections_per_node[node_idx] =
        args.presampled_projections != nullptr
            ? (*args.presampled_projections)[node_idx]
            : SampleNodeProjections(args, node_idx);
    if (projections_per_node[node_idx].empty()) {
      continue;
    }
    active_nodes.push_back(node_idx);
  }
  if (active_nodes.empty()) {
    return absl::OkStatus();
  }

  // Largest nodes first: their jobs are the longest.
  std::stable_sort(active_nodes.begin(), active_nodes.end(),
                   [&](const int a, const int b) {
                     return rows_per_node.rows(a).size() >
                            rows_per_node.rows(b).size();
                   });

  // Split each node's projections into chunks so the layer offers about
  // kJobsPerThread jobs per thread; a chunk never holds less than one
  // projection.
  constexpr int kJobsPerThread = 4;
  const int num_threads = std::max(1, thread_pool->num_threads());
  const int max_chunks_per_node = std::max<int>(
      1, (kJobsPerThread * num_threads + active_nodes.size() - 1) /
             active_nodes.size());

  struct Job {
    int node_idx;
    int first_projection_idx;
    int num_projections;
  };
  std::vector<Job> jobs;
  for (const int node_idx : active_nodes) {
    const int num_projections = projections_per_node[node_idx].size();
    const int num_chunks = std::min(num_projections, max_chunks_per_node);
    const int base = num_projections / num_chunks;
    const int extra = num_projections % num_chunks;
    int first_projection_idx = 0;
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
      const int chunk_size = base + (chunk_idx < extra ? 1 : 0);
      jobs.push_back({node_idx, first_projection_idx, chunk_size});
      first_projection_idx += chunk_size;
    }
    DCHECK_EQ(first_projection_idx, num_projections);
  }

  // Score to beat per node, read before any job runs: "best_splits" is only
  // written after all the jobs are done.
  std::vector<double> initial_score_per_node(num_nodes, 0);
  for (const int node_idx : active_nodes) {
    initial_score_per_node[node_idx] =
        (*best_splits)[node_idx].condition.split_score();
  }

  std::vector<ObliqueChunkResult<Accumulator>> results(jobs.size());
  utils::concurrency::Mutex mutex;
  absl::Status status;
  utils::concurrency::BlockingCounter counter(jobs.size());
  for (size_t job_idx = 0; job_idx < jobs.size(); job_idx++) {
    thread_pool->Schedule([&, job_idx]() {
      const Job& job = jobs[job_idx];
      const auto& projections = projections_per_node[job.node_idx];
      const auto local_status = ScoreProjectionChunk(
          args, rows_per_node.rows(job.node_idx),
          absl::MakeConstSpan(projections)
              .subspan(job.first_projection_idx, job.num_projections),
          job.first_projection_idx, label_filler, initializers[job.node_idx],
          min_num_examples, initial_score_per_node[job.node_idx],
          &results[job_idx]);
      if (!local_status.ok()) {
        utils::concurrency::MutexLock l(mutex);
        status.Update(local_status);
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();
  {
    utils::concurrency::MutexLock l(mutex);
    RETURN_IF_ERROR(status);
  }

  // Per node: highest score, ties to the lowest projection index.
  std::vector<int> best_job_per_node(num_nodes, -1);
  for (size_t job_idx = 0; job_idx < jobs.size(); job_idx++) {
    const auto& result = results[job_idx];
    if (!result.found) {
      continue;
    }
    int& best_job_idx = best_job_per_node[jobs[job_idx].node_idx];
    if (best_job_idx < 0) {
      best_job_idx = job_idx;
      continue;
    }
    const auto& best = results[best_job_idx];
    if (result.score > best.score ||
        (result.score == best.score &&
         result.projection_idx < best.projection_idx)) {
      best_job_idx = job_idx;
    }
  }
  for (const int node_idx : active_nodes) {
    const int best_job_idx = best_job_per_node[node_idx];
    if (best_job_idx < 0) {
      continue;
    }
    const auto& result = results[best_job_idx];
    RETURN_IF_ERROR(CommitObliqueSplit(
        args, node_idx, projections_per_node[node_idx][result.projection_idx],
        result, rows_per_node.rows(node_idx).size(),
        &(*best_splits)[node_idx]));
  }
  return absl::OkStatus();
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
    if (!columns_.contains(feature) && !external_.contains(feature)) {
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
  if (it != columns_.end()) {
    return it->second->data();
  }
  const auto external_it = external_.find(feature);
  if (external_it != external_.end()) {
    return external_it->second->data();
  }
  return nullptr;
}

void DenseNumericalColumns::SetExternal(const int feature,
                                        std::vector<float> values) {
  DCHECK_EQ(values.size(), num_examples());
  external_[feature] = std::make_unique<std::vector<float>>(std::move(values));
}

size_t DenseNumericalColumns::MemoryUsage() const {
  size_t usage = 0;
  for (const auto& column : columns_) {
    usage += column.second->size() * sizeof(float);
  }
  for (const auto& column : external_) {
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
      if (!dataset.has_feature(attribute) && !columns->is_external(attribute)) {
        return absl::InternalError(absl::Substitute(
            "This worker owns attribute $0 of an oblique split but holds no "
            "copy of attribute $1. Either feature ownership changed after the "
            "split was found (dynamic load balancing is not supported with "
            "in-worker oblique splits) or the column was not shipped.",
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
