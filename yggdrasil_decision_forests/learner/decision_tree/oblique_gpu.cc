#include "yggdrasil_decision_forests/learner/decision_tree/oblique_gpu.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree {

absl::StatusOr<std::unique_ptr<ObliqueGpuComputer>> ObliqueGpuComputer::Create(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features,
    bool use_gpu) {
  auto c = absl::WrapUnique(new ObliqueGpuComputer());

  c->num_total_rows_ = dataset.nrow();
  c->num_features_ = numerical_features.size();

  // Build feature column mapping and NA replacement values.
  c->feature_col_to_flat_offset_.resize(
      dataset.data_spec().columns_size(), -1);
  c->na_replacement_values_.resize(
      dataset.data_spec().columns_size(), 0.0f);

  for (int i = 0; i < numerical_features.size(); i++) {
    const int col_idx = numerical_features[i];
    c->feature_col_to_flat_offset_[col_idx] = i;
    // Store mean as NA replacement.
    const auto& col_spec = dataset.data_spec().columns(col_idx);
    if (col_spec.has_numerical()) {
      c->na_replacement_values_[col_idx] = col_spec.numerical().mean();
    }
  }

  if (use_gpu) {
    const auto init_status = c->InitializeGPU(dataset, numerical_features);
    if (!init_status.ok()) {
      LOG(INFO) << "Cannot initialize oblique GPU: " << init_status.message()
                << ". Falling back to CPU.";
      use_gpu = false;
    }
  }

  c->use_gpu_ = use_gpu;
  return c;
}

ObliqueGpuComputer::~ObliqueGpuComputer() {
  if (!released_) {
    LOG(WARNING) << "ObliqueGpuComputer::Release() not called before destructor";
    Release().IgnoreError();
  }
}

absl::Status ObliqueGpuComputer::Release() {
  if (released_) return absl::OkStatus();
  released_ = true;
  if (use_gpu_) {
    return ReleaseGPU();
  }
  return absl::OkStatus();
}

// ---- CPU fallback for ApplyProjectionsBatched ----

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedCPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  // This should not normally be called — the CPU path uses
  // ProjectionEvaluator::Evaluate directly. But it's here for completeness
  // and testing.
  return absl::UnimplementedError(
      "CPU fallback for ApplyProjectionsBatched not needed — use "
      "ProjectionEvaluator::Evaluate directly");
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNodeCPU(
    absl::Span<const NodeBatch> node_batches) {
  return absl::UnimplementedError(
      "CPU fallback for ApplyProjectionsBatchedMultiNode not needed — use "
      "per-node CPU path directly");
}

// ---- Public dispatch ----

absl::Status ObliqueGpuComputer::ApplyProjectionsBatched(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  if (use_gpu_) {
    return ApplyProjectionsBatchedGPU(projections, selected_examples,
                                       projected_values, min_vals, max_vals);
  }
  return ApplyProjectionsBatchedCPU(projections, selected_examples,
                                     projected_values, min_vals, max_vals);
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNode(
    absl::Span<const NodeBatch> node_batches) {
  if (use_gpu_) {
    return ApplyProjectionsBatchedMultiNodeGPU(node_batches);
  }
  return ApplyProjectionsBatchedMultiNodeCPU(node_batches);
}

// ---- GPU stubs when not compiled with CUDA ----

#ifdef CUDA_DISABLED

absl::Status ObliqueGpuComputer::InitializeGPU(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status ObliqueGpuComputer::ReleaseGPU() {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedGPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNodeGPU(
    absl::Span<const NodeBatch> node_batches) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

#endif  // CUDA_DISABLED

}  // namespace yggdrasil_decision_forests::model::decision_tree
