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
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/parallel_chrono.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

// Bridge functions implemented in oblique_gpu_kernels.cu.cc (nvcc-compiled).
extern "C" {
bool oblique_gpu_check_available();
int oblique_gpu_init(const float* h_flat_data, int flat_data_size,
                     float** d_global_flat_data_out);
int oblique_gpu_release(float* d_global_flat_data);
int oblique_gpu_upload_labels(const unsigned int* h_labels, int num_rows,
                              unsigned int** d_labels_out);
int oblique_gpu_free_labels(unsigned int* d_labels);
int oblique_gpu_apply_projections(
    const float* d_global_flat_data,
    const unsigned int* h_selected_examples,
    int num_examples, int num_proj, int num_total_rows,
    const int* h_flat_col_indices, const float* h_flat_weights,
    const int* h_col_offsets, int total_nonzeros,
    float* h_projected_values, float* h_min_vals, float* h_max_vals,
    double* out_ms_apply, double* out_ms_other);
int oblique_gpu_apply_projections_multi_node(
    const float* d_global_flat_data,
    const unsigned int* h_selected_examples, int total_examples,
    const int* h_node_row_off, int num_nodes,
    int num_proj, int num_total_rows,
    const int* h_flat_col_indices, const float* h_flat_weights,
    const int* h_col_offsets, int total_nonzeros,
    float* h_projected_values, int max_examples_per_node,
    double* out_ms_apply, double* out_ms_other);
int oblique_gpu_find_best_split_nodewise(
    const float* d_global_flat_data,
    const unsigned int* d_global_labels,
    int num_total_rows,
    const unsigned int* h_selected_examples,
    int num_examples,
    int num_proj,
    const int* h_flat_col_indices, const float* h_flat_weights,
    const int* h_col_offsets, int total_nonzeros,
    int num_bins, int comp_method,
    unsigned long long random_seed,
    int* best_proj, int* best_bin, float* best_gain,
    float* best_threshold, int* num_pos_examples,
    double* out_ms_apply, double* out_ms_hist, double* out_ms_split,
    double* out_ms_other);
int oblique_gpu_find_best_split_nodewise_exact(
    const float* d_global_flat_data,
    const unsigned int* d_global_labels,
    int num_total_rows,
    const unsigned int* h_selected_examples,
    int num_examples,
    int num_proj,
    const int* h_flat_col_indices, const float* h_flat_weights,
    const int* h_col_offsets, int total_nonzeros,
    int comp_method,
    int* best_proj, int* best_split, float* best_gain,
    float* best_threshold, int* num_pos_examples,
    double* out_ms_apply, double* out_ms_sort, double* out_ms_split,
    double* out_ms_other);
}

namespace yggdrasil_decision_forests::model::decision_tree {

absl::StatusOr<std::unique_ptr<ObliqueGpuComputer>> ObliqueGpuComputer::Create(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features,
    int label_col_idx,
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
    const auto init_status =
        c->InitializeGPU(dataset, numerical_features, label_col_idx);
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

absl::Status ObliqueGpuComputer::ApplyProjectionsNodewiseCPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  return absl::UnimplementedError(
      "CPU fallback not needed — use ProjectionEvaluator::Evaluate");
}

absl::Status ObliqueGpuComputer::ApplyProjectionsDepthwiseCPU(
    absl::Span<const NodeBatch> node_batches) {
  return absl::UnimplementedError("CPU fallback not needed");
}

// ---- Public dispatch ----

absl::Status ObliqueGpuComputer::ApplyProjectionsNodewise(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<float> projected_values,
    absl::Span<float> min_vals,
    absl::Span<float> max_vals) {
  if (use_gpu_) {
#ifdef CHRONO_ENABLED
    auto wait_start = std::chrono::steady_clock::now();
#endif
    utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start).count());
#endif
    return ApplyProjectionsNodewiseGPU(projections, selected_examples,
                                       projected_values, min_vals, max_vals);
  }
  return ApplyProjectionsNodewiseCPU(projections, selected_examples,
                                     projected_values, min_vals, max_vals);
}

absl::Status ObliqueGpuComputer::ApplyProjectionsDepthwise(
    absl::Span<const NodeBatch> node_batches) {
  if (use_gpu_) {
#ifdef CHRONO_ENABLED
    auto wait_start = std::chrono::steady_clock::now();
#endif
    utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start).count());
#endif
    return ApplyProjectionsDepthwiseGPU(node_batches);
  }
  return ApplyProjectionsDepthwiseCPU(node_batches);
}

absl::Status ObliqueGpuComputer::FindBestSplitNodewise(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    int num_bins, int comp_method,
    utils::RandomEngine* random, BestSplitResult* result) {
  if (!supports_full_gpu_split()) {
    return absl::FailedPreconditionError(
        "FindBestSplitNodewise: GPU / labels not available");
  }
#ifdef CHRONO_ENABLED
  auto wait_start = std::chrono::steady_clock::now();
#endif
  utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wait_start).count());
#endif
  return FindBestSplitNodewiseGPU(projections, selected_examples, num_bins,
                                   comp_method, random, result);
}

absl::Status ObliqueGpuComputer::FindBestSplitDepthwise(
    absl::Span<const NodeBatch> node_batches,
    int num_bins, int comp_method,
    utils::RandomEngine* random,
    absl::Span<BestSplitResult> results) {
  if (!supports_full_gpu_split()) {
    return absl::FailedPreconditionError(
        "FindBestSplitDepthwise: GPU / labels not available");
  }
  if (node_batches.size() != results.size()) {
    return absl::InvalidArgumentError(
        "FindBestSplitDepthwise: results span size mismatch");
  }
#ifdef CHRONO_ENABLED
  auto wait_start = std::chrono::steady_clock::now();
#endif
  utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wait_start).count());
#endif
  return FindBestSplitDepthwiseGPU(node_batches, num_bins, comp_method, random,
                                    results);
}

absl::Status ObliqueGpuComputer::FindBestSplitNodewiseExact(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    int comp_method, BestSplitResult* result) {
  if (!supports_full_gpu_split()) {
    return absl::FailedPreconditionError(
        "FindBestSplitNodewiseExact: GPU / labels not available");
  }
#ifdef CHRONO_ENABLED
  auto wait_start = std::chrono::steady_clock::now();
#endif
  utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wait_start).count());
#endif
  return FindBestSplitNodewiseExactGPU(projections, selected_examples,
                                        comp_method, result);
}

absl::Status ObliqueGpuComputer::FindBestSplitDepthwiseExact(
    absl::Span<const NodeBatch> node_batches,
    int comp_method, absl::Span<BestSplitResult> results) {
  if (!supports_full_gpu_split()) {
    return absl::FailedPreconditionError(
        "FindBestSplitDepthwiseExact: GPU / labels not available");
  }
  if (node_batches.size() != results.size()) {
    return absl::InvalidArgumentError(
        "FindBestSplitDepthwiseExact: results span size mismatch");
  }
#ifdef CHRONO_ENABLED
  auto wait_start = std::chrono::steady_clock::now();
#endif
  utils::concurrency::MutexLock l(&gpu_mutex_);
#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuMutexWait,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - wait_start).count());
#endif
  return FindBestSplitDepthwiseExactGPU(node_batches, comp_method, results);
}

// ---- GPU implementations via extern "C" bridge ----

absl::Status ObliqueGpuComputer::InitializeGPU(
    const dataset::VerticalDataset& dataset,
    absl::Span<const int32_t> numerical_features,
    int label_col_idx) {
  CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuInit);
  const auto init_start = absl::Now();
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

  // Upload labels if the label column is categorical with <=3 classes.
  // Failure to upload is NOT fatal: full-GPU split path is disabled, but
  // ApplyProjections* paths still work.
  if (label_col_idx >= 0 &&
      label_col_idx < dataset.data_spec().columns_size()) {
    const auto& label_spec = dataset.data_spec().columns(label_col_idx);
    if (label_spec.type() == dataset::proto::ColumnType::CATEGORICAL &&
        label_spec.has_categorical()) {
      const int num_classes =
          label_spec.categorical().number_of_unique_values();
      if (num_classes > 0 && num_classes <= 3) {
        const auto* label_col = dataset.ColumnWithCastOrNull<
            dataset::VerticalDataset::CategoricalColumn>(label_col_idx);
        if (label_col) {
          const auto& label_values = label_col->values();
          // YDF encodes categorical labels as 1..(num_classes-1) with 0
          // reserved for out-of-vocabulary. The GPU gain kernels
          // (FindBestEntropySplitKernel / FindBestGiniSplitKernel) are
          // hardcoded to use hist_class0 / hist_class1 only, so we shift the
          // labels down by 1 so hist_class0 = real-class-0, etc.
          std::vector<unsigned int> h_labels(n_rows);
          for (int r = 0; r < n_rows; r++) {
            const int32_t v = label_values[r];
            h_labels[r] = static_cast<unsigned int>(v > 0 ? v - 1 : 0);
          }
          int label_err = oblique_gpu_upload_labels(
              h_labels.data(), n_rows, &d_global_labels_);
          if (label_err == 0) {
            num_classes_ = num_classes;
            LOG(INFO) << "ObliqueGpuComputer: Uploaded " << n_rows
                      << " labels (" << num_classes
                      << " classes) to GPU for full-GPU split path.";
          } else {
            LOG(WARNING) << "ObliqueGpuComputer: Label upload failed (CUDA "
                         << label_err << "). Full-GPU split disabled.";
            d_global_labels_ = nullptr;
            num_classes_ = 0;
          }
        }
      } else {
        LOG(INFO) << "ObliqueGpuComputer: label column has " << num_classes
                  << " classes (>3 unsupported). Full-GPU split disabled.";
      }
    } else {
      LOG(INFO) << "ObliqueGpuComputer: label column is not categorical. "
                   "Full-GPU split disabled (only classification supported).";
    }
  }

  LOG(INFO) << "ObliqueGpuComputer: Uploaded " << n_rows << " rows x "
            << n_features << " features to GPU ("
            << (flat_data.size() * sizeof(float)) / (1024 * 1024) << " MB) in "
            << absl::ToDoubleSeconds(absl::Now() - init_start) << " s";
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ReleaseGPU() {
  int cuda_err = oblique_gpu_release(d_global_flat_data_);
  d_global_flat_data_ = nullptr;
  if (d_global_labels_ != nullptr) {
    int label_err = oblique_gpu_free_labels(d_global_labels_);
    d_global_labels_ = nullptr;
    num_classes_ = 0;
    if (label_err != 0 && cuda_err == 0) {
      cuda_err = label_err;
    }
  }
  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("CUDA release failed with error code ", cuda_err));
  }
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ApplyProjectionsNodewiseGPU(
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
  std::vector<int> flat_col_indices;
  std::vector<float> flat_weights;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
    for (int p = 0; p < num_proj; p++) {
      total_nz += projections[p].size();
      col_offsets[p + 1] = total_nz;
    }
    flat_col_indices.resize(total_nz);
    flat_weights.resize(total_nz);
    for (int p = 0; p < num_proj; p++) {
      int offset = col_offsets[p];
      for (int j = 0; j < static_cast<int>(projections[p].size()); j++) {
        flat_col_indices[offset + j] =
            feature_col_to_flat_offset_[projections[p][j].attribute_idx];
        flat_weights[offset + j] = projections[p][j].weight;
      }
    }
  }

  int cuda_err;
  double ms_apply = 0, ms_other = 0;
  cuda_err = oblique_gpu_apply_projections(
    d_global_flat_data_,
    selected_examples.data(),
    num_examples, num_proj, num_total_rows_,
    flat_col_indices.data(), flat_weights.data(),
    col_offsets.data(), total_nz,
    projected_values.data(), min_vals.data(), max_vals.data(),
    &ms_apply, &ms_other);

  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("GPU ApplyProjections failed with CUDA error ", cuda_err));
  }

#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADD,
      static_cast<uint64_t>(ms_apply * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
      static_cast<uint64_t>(ms_other * 1e6));
#endif

  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::ApplyProjectionsDepthwiseGPU(
    absl::Span<const NodeBatch> node_batches) {
  const int num_nodes = node_batches.size();
  if (num_nodes == 0) return absl::OkStatus();

  // Single-node fast path: fall back to the nodewise kernel path.
  if (num_nodes == 1) {
    return ApplyProjectionsNodewiseGPU(
        node_batches[0].projections, node_batches[0].selected_examples,
        node_batches[0].projected_values,
        node_batches[0].min_vals, node_batches[0].max_vals);
  }

  // All nodes share the same num_proj.
  const int num_proj = node_batches[0].projections.size();

  // 1-3. Build prefix sums, concatenate examples, flatten projections.
  std::vector<int> node_row_off(num_nodes + 1, 0);
  int max_examples_per_node = 0;
  int total_examples;
  std::vector<unsigned int> all_selected;
  std::vector<int> col_offsets;
  std::vector<int> flat_col_indices;
  std::vector<float> flat_weights;
  int total_nz = 0;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
    for (int n = 0; n < num_nodes; n++) {
      const int rows_n = node_batches[n].selected_examples.size();
      node_row_off[n + 1] = node_row_off[n] + rows_n;
      if (rows_n > max_examples_per_node) max_examples_per_node = rows_n;
    }
    total_examples = node_row_off[num_nodes];

    all_selected.resize(total_examples);
    for (int n = 0; n < num_nodes; n++) {
      std::memcpy(all_selected.data() + node_row_off[n],
                  node_batches[n].selected_examples.data(),
                  node_batches[n].selected_examples.size() * sizeof(unsigned int));
    }

    const int num_segments = num_nodes * num_proj;
    col_offsets.resize(num_segments + 1, 0);
    for (int n = 0; n < num_nodes; n++) {
      for (int p = 0; p < num_proj; p++) {
        total_nz += node_batches[n].projections[p].size();
        col_offsets[n * num_proj + p + 1] = total_nz;
      }
    }

    flat_col_indices.resize(total_nz);
    flat_weights.resize(total_nz);
    for (int n = 0; n < num_nodes; n++) {
      for (int p = 0; p < num_proj; p++) {
        const int seg = n * num_proj + p;
        int offset = col_offsets[seg];
        for (int j = 0; j < static_cast<int>(node_batches[n].projections[p].size()); j++) {
          flat_col_indices[offset + j] =
              feature_col_to_flat_offset_[node_batches[n].projections[p][j].attribute_idx];
          flat_weights[offset + j] = node_batches[n].projections[p][j].weight;
        }
      }
    }
  }

  // 4. Single GPU kernel launch for all nodes.
  std::vector<float> all_projected(static_cast<size_t>(total_examples) * num_proj);
  int cuda_err;
  double ms_apply = 0, ms_other = 0;
  cuda_err = oblique_gpu_apply_projections_multi_node(
      d_global_flat_data_,
      all_selected.data(), total_examples,
      node_row_off.data(), num_nodes,
      num_proj, num_total_rows_,
      flat_col_indices.data(), flat_weights.data(),
      col_offsets.data(), total_nz,
      all_projected.data(), max_examples_per_node,
      &ms_apply, &ms_other);

  if (cuda_err != 0) {
    return absl::InternalError(
        absl::StrCat("GPU depthwise ApplyProjections failed: CUDA error ", cuda_err));
  }

#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADDMultiNode,
      static_cast<uint64_t>(ms_apply * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
      static_cast<uint64_t>(ms_other * 1e6));
#endif

  // 5. Unpack results back to per-node output buffers.
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuResultUnpack);
    for (int n = 0; n < num_nodes; n++) {
      const int rows_n = node_row_off[n + 1] - node_row_off[n];
      const size_t src_offset = static_cast<size_t>(node_row_off[n]) * num_proj;
      const size_t block_size = static_cast<size_t>(rows_n) * num_proj;
      std::memcpy(node_batches[n].projected_values.data(),
                  all_projected.data() + src_offset,
                  block_size * sizeof(float));
      std::memset(node_batches[n].min_vals.data(), 0, num_proj * sizeof(float));
      std::memset(node_batches[n].max_vals.data(), 0, num_proj * sizeof(float));
    }
  }

  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::FindBestSplitNodewiseGPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    int num_bins, int comp_method,
    utils::RandomEngine* random, BestSplitResult* result) {
  const int num_proj = projections.size();
  const int num_examples = selected_examples.size();

  // Reset to "no split" defaults.
  *result = BestSplitResult{};

  if (num_proj == 0 || num_examples <= 1) {
    return absl::OkStatus();
  }

  // Flatten projections to CSR.
  std::vector<int> col_offsets(num_proj + 1, 0);
  int total_nz = 0;
  std::vector<int> flat_col_indices;
  std::vector<float> flat_weights;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
    for (int p = 0; p < num_proj; p++) {
      total_nz += projections[p].size();
      col_offsets[p + 1] = total_nz;
    }
    flat_col_indices.resize(total_nz);
    flat_weights.resize(total_nz);
    for (int p = 0; p < num_proj; p++) {
      int offset = col_offsets[p];
      for (int j = 0; j < static_cast<int>(projections[p].size()); j++) {
        flat_col_indices[offset + j] =
            feature_col_to_flat_offset_[projections[p][j].attribute_idx];
        flat_weights[offset + j] = projections[p][j].weight;
      }
    }
  }

  int best_proj = -1, best_bin = -1, num_pos = 0;
  float best_gain = -1.0f, best_threshold = 0.0f;
  double ms_apply = 0, ms_hist = 0, ms_split = 0, ms_other = 0;

  // Draw a 64-bit seed from the caller's RNG so RandomHistogram's internal
  // std::mt19937 produces reproducible split candidates.
  std::uniform_int_distribution<unsigned long long> seed_dist;
  const unsigned long long seed = seed_dist(*random);

  int cuda_err = oblique_gpu_find_best_split_nodewise(
      d_global_flat_data_, d_global_labels_, num_total_rows_,
      selected_examples.data(), num_examples,
      num_proj,
      flat_col_indices.data(), flat_weights.data(),
      col_offsets.data(), total_nz,
      num_bins, comp_method, seed,
      &best_proj, &best_bin, &best_gain, &best_threshold, &num_pos,
      &ms_apply, &ms_hist, &ms_split, &ms_other);

  if (cuda_err != 0) {
    return absl::InternalError(absl::StrCat(
        "GPU FindBestSplitNodewise failed: CUDA error ", cuda_err));
  }

#ifdef CHRONO_ENABLED
  // Flat disjoint partition of GPU bridge time. All cuEvent-measured
  // inside the bridge with a single sync at the end.
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADD,
      static_cast<uint64_t>(ms_apply * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuRandomHistogram,
      static_cast<uint64_t>(ms_hist * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuSplitHistogram,
      static_cast<uint64_t>(ms_split * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
      static_cast<uint64_t>(ms_other * 1e6));
#endif

  result->best_proj_idx = best_proj;
  result->best_bin_idx = best_bin;
  result->best_gain = best_gain;
  result->best_threshold = best_threshold;
  result->num_pos_training_examples = num_pos;
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::FindBestSplitDepthwiseGPU(
    absl::Span<const NodeBatch> node_batches,
    int num_bins, int comp_method,
    utils::RandomEngine* random,
    absl::Span<BestSplitResult> results) {
  // First cut: loop per-node using FindBestSplitNodewiseGPU. The kernels in
  // randomprojection.cu (RandomHistogram, HistogramSplit) are per-node; a true
  // multi-node histogram + split kernel can be added later.
  //
  // NOTE: gpu_mutex_ is already held by the caller
  // (FindBestSplitDepthwise).
  const int num_nodes = node_batches.size();
  for (int n = 0; n < num_nodes; n++) {
    results[n] = BestSplitResult{};
    if (node_batches[n].projections.empty() ||
        node_batches[n].selected_examples.size() <= 1) {
      continue;
    }

    const int num_proj = node_batches[n].projections.size();
    const int num_examples = node_batches[n].selected_examples.size();

    std::vector<int> col_offsets(num_proj + 1, 0);
    int total_nz = 0;
    std::vector<int> flat_col_indices;
    std::vector<float> flat_weights;
    {
      CHRONO_SCOPE(
          ::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
      for (int p = 0; p < num_proj; p++) {
        total_nz += node_batches[n].projections[p].size();
        col_offsets[p + 1] = total_nz;
      }
      flat_col_indices.resize(total_nz);
      flat_weights.resize(total_nz);
      for (int p = 0; p < num_proj; p++) {
        int offset = col_offsets[p];
        const auto& proj = node_batches[n].projections[p];
        for (int j = 0; j < static_cast<int>(proj.size()); j++) {
          flat_col_indices[offset + j] =
              feature_col_to_flat_offset_[proj[j].attribute_idx];
          flat_weights[offset + j] = proj[j].weight;
        }
      }
    }

    int best_proj = -1, best_bin = -1, num_pos = 0;
    float best_gain = -1.0f, best_threshold = 0.0f;
    double ms_apply = 0, ms_hist = 0, ms_split = 0, ms_other = 0;

    std::uniform_int_distribution<unsigned long long> seed_dist;
    const unsigned long long seed = seed_dist(*random);

    int cuda_err = oblique_gpu_find_best_split_nodewise(
        d_global_flat_data_, d_global_labels_, num_total_rows_,
        node_batches[n].selected_examples.data(), num_examples,
        num_proj,
        flat_col_indices.data(), flat_weights.data(),
        col_offsets.data(), total_nz,
        num_bins, comp_method, seed,
        &best_proj, &best_bin, &best_gain, &best_threshold, &num_pos,
        &ms_apply, &ms_hist, &ms_split, &ms_other);

    if (cuda_err != 0) {
      return absl::InternalError(absl::StrCat(
          "GPU FindBestSplitDepthwise node ", n,
          " failed: CUDA error ", cuda_err));
    }

#ifdef CHRONO_ENABLED
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADD,
        static_cast<uint64_t>(ms_apply * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuRandomHistogram,
        static_cast<uint64_t>(ms_hist * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuSplitHistogram,
        static_cast<uint64_t>(ms_split * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
        static_cast<uint64_t>(ms_other * 1e6));
#endif

    results[n].best_proj_idx = best_proj;
    results[n].best_bin_idx = best_bin;
    results[n].best_gain = best_gain;
    results[n].best_threshold = best_threshold;
    results[n].num_pos_training_examples = num_pos;
  }
  return absl::OkStatus();
}

// Helper: CSR-flatten a single node's projections.
static void FlattenProjectionsCSR(
    absl::Span<const internal::Projection> projections,
    const std::vector<int>& feature_col_to_flat_offset,
    std::vector<int>* col_offsets,
    std::vector<int>* flat_col_indices,
    std::vector<float>* flat_weights) {
  const int num_proj = projections.size();
  col_offsets->assign(num_proj + 1, 0);
  int total_nz = 0;
  for (int p = 0; p < num_proj; p++) {
    total_nz += projections[p].size();
    (*col_offsets)[p + 1] = total_nz;
  }
  flat_col_indices->resize(total_nz);
  flat_weights->resize(total_nz);
  for (int p = 0; p < num_proj; p++) {
    int offset = (*col_offsets)[p];
    for (int j = 0; j < static_cast<int>(projections[p].size()); j++) {
      (*flat_col_indices)[offset + j] =
          feature_col_to_flat_offset[projections[p][j].attribute_idx];
      (*flat_weights)[offset + j] = projections[p][j].weight;
    }
  }
}

absl::Status ObliqueGpuComputer::FindBestSplitNodewiseExactGPU(
    absl::Span<const internal::Projection> projections,
    absl::Span<const UnsignedExampleIdx> selected_examples,
    int comp_method, BestSplitResult* result) {
  const int num_proj = projections.size();
  const int num_examples = selected_examples.size();

  *result = BestSplitResult{};
  if (num_proj == 0 || num_examples <= 1) {
    return absl::OkStatus();
  }

  std::vector<int> col_offsets;
  std::vector<int> flat_col_indices;
  std::vector<float> flat_weights;
  {
    CHRONO_SCOPE(::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
    FlattenProjectionsCSR(projections, feature_col_to_flat_offset_,
                           &col_offsets, &flat_col_indices, &flat_weights);
  }
  const int total_nz = col_offsets.empty() ? 0 : col_offsets.back();

  int best_proj = -1, best_split = -1, num_pos = 0;
  float best_gain = -1.0f, best_threshold = 0.0f;
  double ms_apply = 0, ms_sort = 0, ms_split = 0, ms_other = 0;

  int cuda_err = oblique_gpu_find_best_split_nodewise_exact(
      d_global_flat_data_, d_global_labels_, num_total_rows_,
      selected_examples.data(), num_examples,
      num_proj,
      flat_col_indices.data(), flat_weights.data(),
      col_offsets.data(), total_nz,
      comp_method,
      &best_proj, &best_split, &best_gain, &best_threshold, &num_pos,
      &ms_apply, &ms_sort, &ms_split, &ms_other);
  if (cuda_err != 0) {
    return absl::InternalError(absl::StrCat(
        "GPU FindBestSplitNodewiseExact failed: CUDA error ", cuda_err));
  }

#ifdef CHRONO_ENABLED
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADD,
      static_cast<uint64_t>(ms_apply * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuSortIndices,
      static_cast<uint64_t>(ms_sort * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuExactSplit,
      static_cast<uint64_t>(ms_split * 1e6));
  chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
      chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
      static_cast<uint64_t>(ms_other * 1e6));
#endif

  result->best_proj_idx = best_proj;
  result->best_bin_idx = best_split;    // reused as split-index for Exact
  result->best_gain = best_gain;
  result->best_threshold = best_threshold;
  result->num_pos_training_examples = num_pos;
  return absl::OkStatus();
}

absl::Status ObliqueGpuComputer::FindBestSplitDepthwiseExactGPU(
    absl::Span<const NodeBatch> node_batches,
    int comp_method, absl::Span<BestSplitResult> results) {
  // First cut: loop per-node. Same structure as FindBestSplitDepthwiseGPU.
  const int num_nodes = node_batches.size();
  for (int n = 0; n < num_nodes; n++) {
    results[n] = BestSplitResult{};
    if (node_batches[n].projections.empty() ||
        node_batches[n].selected_examples.size() <= 1) {
      continue;
    }

    const int num_proj = node_batches[n].projections.size();
    const int num_examples = node_batches[n].selected_examples.size();

    std::vector<int> col_offsets;
    std::vector<int> flat_col_indices;
    std::vector<float> flat_weights;
    {
      CHRONO_SCOPE(
          ::yggdrasil_decision_forests::chrono_prof::kGpuCsrFlatten);
      FlattenProjectionsCSR(node_batches[n].projections,
                             feature_col_to_flat_offset_,
                             &col_offsets, &flat_col_indices, &flat_weights);
    }
    const int total_nz = col_offsets.empty() ? 0 : col_offsets.back();

    int best_proj = -1, best_split = -1, num_pos = 0;
    float best_gain = -1.0f, best_threshold = 0.0f;
    double ms_apply = 0, ms_sort = 0, ms_split = 0, ms_other = 0;

    int cuda_err = oblique_gpu_find_best_split_nodewise_exact(
        d_global_flat_data_, d_global_labels_, num_total_rows_,
        node_batches[n].selected_examples.data(), num_examples,
        num_proj,
        flat_col_indices.data(), flat_weights.data(),
        col_offsets.data(), total_nz,
        comp_method,
        &best_proj, &best_split, &best_gain, &best_threshold, &num_pos,
        &ms_apply, &ms_sort, &ms_split, &ms_other);
    if (cuda_err != 0) {
      return absl::InternalError(absl::StrCat(
          "GPU FindBestSplitDepthwiseExact node ", n,
          " failed: CUDA error ", cuda_err));
    }

#ifdef CHRONO_ENABLED
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuApplyColumnADD,
        static_cast<uint64_t>(ms_apply * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuSortIndices,
        static_cast<uint64_t>(ms_sort * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuExactSplit,
        static_cast<uint64_t>(ms_split * 1e6));
    chrono_prof::add_time(chrono_prof::tls_ctx.cur_tree,
        chrono_prof::tls_ctx.cur_depth, chrono_prof::kGpuOther,
        static_cast<uint64_t>(ms_other * 1e6));
#endif

    results[n].best_proj_idx = best_proj;
    results[n].best_bin_idx = best_split;
    results[n].best_gain = best_gain;
    results[n].best_threshold = best_threshold;
    results[n].num_pos_training_examples = num_pos;
  }
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree
