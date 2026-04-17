// GPU-accelerated oblique projection computation.
// Follows the VectorSequenceComputer pattern from gpu.h.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_GPU_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_GPU_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model::decision_tree {

}  // namespace yggdrasil_decision_forests::model::decision_tree

#include "yggdrasil_decision_forests/learner/decision_tree/oblique_types.h"

namespace yggdrasil_decision_forests::model::decision_tree {

// Accelerates the ApplyProjection step of oblique split-finding on GPU.
//
// Two batching strategies are offered:
//   * nodewise-gpu: one GPU kernel per node, batching across the node's
//     projections. Use when nodes are processed independently (recursive DFS
//     or single-node-at-a-time BFS).
//   * depthwise-gpu: one GPU kernel per BFS depth level, batching across both
//     projections and nodes. Use with the BFS driver when multiple sibling
//     nodes are ready at the same depth.
//
// Usage:
//   1. Create once per training run (uploads dataset to GPU).
//   2. Call ApplyProjectionsNodewise or ApplyProjectionsDepthwise as needed.
//   3. Release() when training is done.
//
// Thread safety: multiple tree-training threads may share one instance.
// GPU access is serialized via an internal mutex.
class ObliqueGpuComputer {
 public:

  // Creates the computer. If use_gpu=true but no GPU is available, falls back
  // to CPU (does not fail). The dataset's numerical feature columns are
  // uploaded to device memory once and persist until Release().
  //
  // `label_col_idx` is the index of the label column. If the column is
  // categorical with <=3 unique values (including the reserved OOV slot), the
  // labels are uploaded to GPU for use by the full-GPU split path. Otherwise
  // label upload is skipped and `supports_full_gpu_split()` returns false.
  static absl::StatusOr<std::unique_ptr<ObliqueGpuComputer>> Create(
      const dataset::VerticalDataset& dataset,
      absl::Span<const int32_t> numerical_features,
      int label_col_idx,
      bool use_gpu);

  ~ObliqueGpuComputer();

  // Releases GPU resources. No work can be submitted after this call.
  absl::Status Release();

  // Is GPU actually being used?
  bool use_gpu() const { return use_gpu_; }

  // True when labels have been uploaded and the dataset is classification
  // with <=3 classes (the limits of the existing full-GPU split kernels).
  bool supports_full_gpu_split() const {
    return use_gpu_ && d_global_labels_ != nullptr && num_classes_ > 0 &&
           num_classes_ <= 3;
  }

  int num_classes() const { return num_classes_; }

  // ---- nodewise-gpu: batch all projections for a single node ----

  // Computes projected_values[p * num_examples + e] =
  //   sum_over_attrs(dataset[selected_examples[e], attr] * projection[p][attr].weight)
  // Also computes per-projection min and max values.
  absl::Status ApplyProjectionsNodewise(
      absl::Span<const internal::Projection> projections,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<float> projected_values,  // [num_projections * num_examples]
      absl::Span<float> min_vals,          // [num_projections]
      absl::Span<float> max_vals);         // [num_projections]

  // ---- depthwise-gpu: batch projections across multiple same-depth nodes ----

  struct NodeBatch {
    absl::Span<const UnsignedExampleIdx> selected_examples;
    absl::Span<const internal::Projection> projections;
    // Caller-allocated output buffers:
    absl::Span<float> projected_values;  // [num_projections * num_examples]
    absl::Span<float> min_vals;          // [num_projections]
    absl::Span<float> max_vals;          // [num_projections]
  };

  absl::Status ApplyProjectionsDepthwise(
      absl::Span<const NodeBatch> node_batches);

  // ---- Full-GPU split (Apply + Histogram + BestSplit on device) ----

  // Runs the full split-finding pipeline on GPU for a single node. Requires
  // supports_full_gpu_split() to be true. `num_bins` is the histogram bin
  // count; `comp_method` is 0 for entropy, 1 for gini. `random` seeds the
  // host-side random split-boundary generator inside RandomHistogram.
  //
  // On return, `*result` holds the winning (proj, bin, gain, threshold) or
  // result->best_gain < 0 if no valid split was found.
  absl::Status FindBestSplitNodewise(
      absl::Span<const internal::Projection> projections,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      int num_bins,
      int comp_method,
      utils::RandomEngine* random,
      BestSplitResult* result);

  // Full-GPU split for a batch of same-depth sibling nodes. Each entry in
  // `results` is filled with that node's winning split.
  absl::Status FindBestSplitDepthwise(
      absl::Span<const NodeBatch> node_batches,
      int num_bins,
      int comp_method,
      utils::RandomEngine* random,
      absl::Span<BestSplitResult> results);

 private:
  ObliqueGpuComputer() = default;

  // CPU fallback implementations.
  absl::Status ApplyProjectionsNodewiseCPU(
      absl::Span<const internal::Projection> projections,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<float> projected_values,
      absl::Span<float> min_vals,
      absl::Span<float> max_vals);

  absl::Status ApplyProjectionsDepthwiseCPU(
      absl::Span<const NodeBatch> node_batches);

  // GPU implementations (defined in oblique_gpu.cu.cc).
  absl::Status InitializeGPU(
      const dataset::VerticalDataset& dataset,
      absl::Span<const int32_t> numerical_features,
      int label_col_idx);
  absl::Status ReleaseGPU();

  absl::Status ApplyProjectionsNodewiseGPU(
      absl::Span<const internal::Projection> projections,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<float> projected_values,
      absl::Span<float> min_vals,
      absl::Span<float> max_vals);

  absl::Status ApplyProjectionsDepthwiseGPU(
      absl::Span<const NodeBatch> node_batches);

  absl::Status FindBestSplitNodewiseGPU(
      absl::Span<const internal::Projection> projections,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      int num_bins, int comp_method,
      utils::RandomEngine* random, BestSplitResult* result);

  absl::Status FindBestSplitDepthwiseGPU(
      absl::Span<const NodeBatch> node_batches,
      int num_bins, int comp_method,
      utils::RandomEngine* random,
      absl::Span<BestSplitResult> results);

  bool use_gpu_ = false;
  bool released_ = false;

  // Persistent GPU state: dataset columns flattened on device.
  float* d_global_flat_data_ = nullptr;
  int num_total_rows_ = 0;
  int num_features_ = 0;

  // Persistent GPU state: classification labels as unsigned int [num_rows].
  // Only populated when the dataset is classification with <=3 classes.
  unsigned int* d_global_labels_ = nullptr;
  int num_classes_ = 0;

  // Mapping from feature column index to offset in d_global_flat_data_.
  std::vector<int> feature_col_to_flat_offset_;

  // NA replacement values per feature (for missing data handling).
  std::vector<float> na_replacement_values_;

  // Serializes GPU access across tree-training threads.
  utils::concurrency::Mutex gpu_mutex_;
};

}  // namespace yggdrasil_decision_forests::model::decision_tree

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_OBLIQUE_GPU_H_
