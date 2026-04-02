#include "yggdrasil_decision_forests/learner/decision_tree/oblique_gpu.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

// Bridge functions implemented in oblique_gpu_kernels.cu.cc (nvcc-compiled).
extern "C" {
bool oblique_gpu_check_available();
int oblique_gpu_init(const float* h_flat_data, int flat_data_size,
                     float** d_global_flat_data_out);
int oblique_gpu_release(float* d_global_flat_data);
int oblique_gpu_apply_projections(
    const float* d_global_flat_data,
    const unsigned int* h_selected_examples,
    int num_examples, int num_proj, int num_total_rows,
    const int* h_flat_col_indices, const float* h_flat_weights,
    const int* h_col_offsets, int total_nonzeros,
    float* h_projected_values, float* h_min_vals, float* h_max_vals);
}

namespace yggdrasil_decision_forests::model::decision_tree {

absl::StatusOr<std::unique_ptr<ObliqueGpuComputer>> ObliqueGpuComputer::Create(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features,
    bool use_gpu) {
  auto c = absl::WrapUnique(new ObliqueGpuComputer());

  c->num_total_rows_ = dataset.nrow();
  c->num_features_ = numerical_features.size();

  c->feature_col_to_flat_offset_.resize(
      dataset.data_spec().columns_size(), -1);
  c->na_replacement_values_.resize(
      dataset.data_spec().columns_size(), 0.0f);

  for (int i = 0; i < static_cast<int>(numerical_features.size()); i++) {
    const int col_idx = numerical_features[i];
    c->feature_col_to_flat_offset_[col_idx] = i;
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

// ---- CPU fallback ----

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedCPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  return absl::UnimplementedError(
      "CPU fallback not needed — use ProjectionEvaluator::Evaluate");
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNodeCPU(
    absl::Span<const NodeBatch> node_batches) {
  return absl::UnimplementedError("CPU fallback not needed");
}

// ---- Public dispatch ----

absl::Status ObliqueGpuComputer::ApplyProjectionsBatched(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  if (use_gpu_) {
    utils::concurrency::MutexLock l(&gpu_mutex_);
    return ApplyProjectionsBatchedGPU(projections, selected_examples,
                                       projected_values, min_vals, max_vals);
  }
  return ApplyProjectionsBatchedCPU(projections, selected_examples,
                                     projected_values, min_vals, max_vals);
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNode(
    absl::Span<const NodeBatch> node_batches) {
  if (use_gpu_) {
    utils::concurrency::MutexLock l(&gpu_mutex_);
    return ApplyProjectionsBatchedMultiNodeGPU(node_batches);
  }
  return ApplyProjectionsBatchedMultiNodeCPU(node_batches);
}

// ---- GPU implementations via extern "C" bridge ----

absl::Status ObliqueGpuComputer::InitializeGPU(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features) {
  if (!oblique_gpu_check_available()) {
    return absl::UnavailableError("No CUDA device found");
  }

  const int n_rows = dataset.nrow();
  const int n_features = numerical_features.size();
  std::vector<float> flat_data(n_features * n_rows, 0.0f);

  for (int f = 0; f < n_features; f++) {
    const int col_idx = numerical_features[f];
    const auto* col = dataset.ColumnWithCastOrNull<
        dataset::VerticalDataset::NumericalColumn>(col_idx);
    if (!col) continue;
    const auto& values = col->values();
    const float na_val = na_replacement_values_[col_idx];
    for (int r = 0; r < n_rows; r++) {
      float v = values[r];
      if (std::isnan(v)) v = na_val;
      flat_data[f * n_rows + r] = v;
    }
  }

  int cuda_err = oblique_gpu_init(flat_data.data(), flat_data.size(),
                                   &d_global_flat_data_);
  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("CUDA init failed with error code ", cuda_err));
  }

  LOG(INFO) << "ObliqueGpuComputer: Uploaded " << n_rows << " rows x "
            << n_features << " features to GPU ("
            << (flat_data.size() * sizeof(float)) / (1024 * 1024) << " MB)";
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ReleaseGPU() {
  int cuda_err = oblique_gpu_release(d_global_flat_data_);
  d_global_flat_data_ = nullptr;
  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("CUDA release failed with error code ", cuda_err));
  }
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedGPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  const int num_proj = projections.size();
  const int num_examples = selected_examples.size();

  // Flatten projections to CSR format.
  std::vector<int> col_offsets(num_proj + 1, 0);
  int total_nz = 0;
  for (int p = 0; p < num_proj; p++) {
    total_nz += projections[p].size();
    col_offsets[p + 1] = total_nz;
  }

  std::vector<int> flat_col_indices(total_nz);
  std::vector<float> flat_weights(total_nz);
  for (int p = 0; p < num_proj; p++) {
    int offset = col_offsets[p];
    for (int j = 0; j < static_cast<int>(projections[p].size()); j++) {
      flat_col_indices[offset + j] =
          feature_col_to_flat_offset_[projections[p][j].attribute_idx];
      flat_weights[offset + j] = projections[p][j].weight;
    }
  }

  int cuda_err = oblique_gpu_apply_projections(
      d_global_flat_data_,
      selected_examples.data(),
      num_examples, num_proj, num_total_rows_,
      flat_col_indices.data(), flat_weights.data(),
      col_offsets.data(), total_nz,
      projected_values.data(), min_vals.data(), max_vals.data());

  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("GPU ApplyProjections failed with CUDA error ", cuda_err));
  }
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ApplyProjectionsBatchedMultiNodeGPU(
    absl::Span<const NodeBatch> node_batches) {
  for (const auto& batch : node_batches) {
    RETURN_IF_ERROR(ApplyProjectionsBatchedGPU(
        batch.projections, batch.selected_examples,
        batch.projected_values, batch.min_vals, batch.max_vals));
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
