// CUDA implementations for ObliqueGpuComputer.
// Compiled by nvcc via cuda_library rule.

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include "yggdrasil_decision_forests/learner/decision_tree/randomprojection.hpp"

// Bridge functions called from oblique_gpu.cc (compiled by host compiler).
// These wrap CUDA runtime calls that can only appear in nvcc-compiled code.

extern "C" {

int oblique_gpu_init(
    const float* h_flat_data, int flat_data_size,
    float** d_global_flat_data_out) {
  cudaError_t err;

  err = cudaMalloc(d_global_flat_data_out, flat_data_size * sizeof(float));
  if (err != cudaSuccess) return (int)err;

  err = cudaMemcpy(*d_global_flat_data_out, h_flat_data,
                   flat_data_size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return (int)err;

  // Warm up GPU.
  warmupfunction();

  return 0;  // cudaSuccess
}

int oblique_gpu_release(float* d_global_flat_data) {
  if (d_global_flat_data) {
    return (int)cudaFree(d_global_flat_data);
  }
  return 0;
}

int oblique_gpu_upload_labels(const unsigned int* h_labels, int num_rows,
                              unsigned int** d_labels_out) {
  cudaError_t err;
  err = cudaMalloc(d_labels_out, num_rows * sizeof(unsigned int));
  if (err != cudaSuccess) return (int)err;
  err = cudaMemcpy(*d_labels_out, h_labels, num_rows * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  return (int)err;
}

int oblique_gpu_free_labels(unsigned int* d_labels) {
  if (d_labels) {
    return (int)cudaFree(d_labels);
  }
  return 0;
}

int oblique_gpu_apply_projections(
    const float* d_global_flat_data,
    const unsigned int* h_selected_examples,
    int num_examples,
    int num_proj,
    int num_total_rows,
    // CSR-format projections
    const int* h_flat_col_indices,
    const float* h_flat_weights,
    const int* h_col_offsets,  // [num_proj + 1]
    int total_nonzeros,
    // Outputs (host memory, caller-allocated)
    float* h_projected_values,  // [num_proj * num_examples]
    float* h_min_vals,          // [num_proj]
    float* h_max_vals,          // [num_proj]
    // Sub-timings (ms): ApplyProjectionColumnADD stage and residual.
    double* out_ms_apply,
    double* out_ms_other
) {
  // Clear any sticky CUDA error from a previous call so cudaPeekAtLastError
  // downstream only reports errors produced by THIS invocation. Mirrors the
  // same guard in oblique_gpu_find_best_split_nodewise.
  (void)cudaGetLastError();

  cudaError_t err;

  // Allocate device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;
  float* d_min_vals = nullptr;
  float* d_max_vals = nullptr;
  float* d_bin_widths = nullptr;

  cudaEvent_t ev_bridge_begin = nullptr, ev_bridge_end = nullptr;
  cudaEvent_t ev_apply_begin = nullptr, ev_apply_end = nullptr;
  cudaEventCreate(&ev_bridge_begin);
  cudaEventCreate(&ev_bridge_end);
  cudaEventCreate(&ev_apply_begin);
  cudaEventCreate(&ev_apply_end);

  cudaEventRecord(ev_bridge_begin, /*stream=*/0);

  err = cudaMalloc(&d_selected_examples, num_examples * sizeof(unsigned int));
  if (err != cudaSuccess) goto cleanup;

  err = cudaMalloc(&d_projected, num_proj * num_examples * sizeof(float));
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpy(d_selected_examples, h_selected_examples,
                   num_examples * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto cleanup;

  {
    // Convert CSR to labmate's vector<vector<>> format.
    std::vector<std::vector<int>> projection_col_idx(num_proj);
    std::vector<std::vector<float>> projection_weights(num_proj);
    for (int p = 0; p < num_proj; p++) {
      int start = h_col_offsets[p];
      int end = h_col_offsets[p + 1];
      projection_col_idx[p].assign(h_flat_col_indices + start,
                                    h_flat_col_indices + end);
      projection_weights[p].assign(h_flat_weights + start,
                                    h_flat_weights + end);
    }

    double elapsed_ms = 0;
    cudaEventRecord(ev_apply_begin, /*stream=*/0);
    ApplyProjectionColumnADD(
        d_global_flat_data,
        d_selected_examples,
        d_projected,
        &d_min_vals,
        &d_max_vals,
        &d_bin_widths,
        projection_col_idx,
        projection_weights,
        num_examples,
        num_proj,
        num_total_rows,
        &elapsed_ms,
        0,      // gpu_mode: 0 = Exact (just compute projections + min/max)
        false,  // verbose
        0       // default stream
    );
    cudaEventRecord(ev_apply_end, /*stream=*/0);
  }

  // Copy results back.
  err = cudaMemcpy(h_projected_values, d_projected,
                   num_proj * num_examples * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto cleanup;

  if (d_min_vals && h_min_vals) {
    err = cudaMemcpy(h_min_vals, d_min_vals, num_proj * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;
  }
  if (d_max_vals && h_max_vals) {
    err = cudaMemcpy(h_max_vals, d_max_vals, num_proj * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;
  }

  cudaEventRecord(ev_bridge_end, /*stream=*/0);
  cudaEventSynchronize(ev_bridge_end);

  {
    float ms_apply = 0, ms_total = 0;
    cudaEventElapsedTime(&ms_apply, ev_apply_begin, ev_apply_end);
    cudaEventElapsedTime(&ms_total, ev_bridge_begin, ev_bridge_end);
    if (out_ms_apply) *out_ms_apply = ms_apply;
    if (out_ms_other) {
      double other = (double)ms_total - (double)ms_apply;
      *out_ms_other = other < 0 ? 0 : other;
    }
  }

cleanup:
  cudaEventDestroy(ev_bridge_begin);
  cudaEventDestroy(ev_bridge_end);
  cudaEventDestroy(ev_apply_begin);
  cudaEventDestroy(ev_apply_end);

  if (d_selected_examples) cudaFree(d_selected_examples);
  if (d_projected) cudaFree(d_projected);
  if (d_min_vals) cudaFree(d_min_vals);
  if (d_max_vals) cudaFree(d_max_vals);
  if (d_bin_widths) cudaFree(d_bin_widths);

  return (int)err;
}

int oblique_gpu_apply_projections_multi_node(
    const float* d_global_flat_data,
    const unsigned int* h_selected_examples,
    int total_examples,
    const int* h_node_row_off,
    int num_nodes,
    int num_proj,
    int num_total_rows,
    const int* h_flat_col_indices,
    const float* h_flat_weights,
    const int* h_col_offsets,
    int total_nonzeros,
    float* h_projected_values,
    int max_examples_per_node,
    // Sub-timings (ms): ApplyProjectionColumnADDMultiNode stage and residual.
    double* out_ms_apply,
    double* out_ms_other)
{
  cudaError_t err;

  // Allocate device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;
  int* d_node_row_off = nullptr;
  int* d_col_offsets = nullptr;
  int* d_flat_col_indices = nullptr;
  float* d_flat_weights = nullptr;

  cudaEvent_t ev_bridge_begin = nullptr, ev_bridge_end = nullptr;
  cudaEvent_t ev_apply_begin = nullptr, ev_apply_end = nullptr;
  cudaEventCreate(&ev_bridge_begin);
  cudaEventCreate(&ev_bridge_end);
  cudaEventCreate(&ev_apply_begin);
  cudaEventCreate(&ev_apply_end);

  const int num_segments = num_nodes * num_proj;

  cudaEventRecord(ev_bridge_begin, /*stream=*/0);

  err = cudaMalloc(&d_selected_examples, total_examples * sizeof(unsigned int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&d_projected, (size_t)total_examples * num_proj * sizeof(float));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&d_node_row_off, (num_nodes + 1) * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&d_col_offsets, (num_segments + 1) * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&d_flat_col_indices, total_nonzeros * sizeof(int));
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(&d_flat_weights, total_nonzeros * sizeof(float));
  if (err != cudaSuccess) goto cleanup;

  // Copy to device.
  cudaMemcpy(d_selected_examples, h_selected_examples,
             total_examples * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_node_row_off, h_node_row_off,
             (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_offsets, h_col_offsets,
             (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flat_col_indices, h_flat_col_indices,
             total_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_flat_weights, h_flat_weights,
             total_nonzeros * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(ev_apply_begin, /*stream=*/0);
  // Launch single 3D kernel for all nodes.
  ApplyProjectionColumnADDMultiNode(
      d_global_flat_data, d_selected_examples, d_projected,
      d_node_row_off, d_col_offsets, d_flat_col_indices, d_flat_weights,
      max_examples_per_node, num_nodes, num_proj, num_total_rows, 0);
  cudaEventRecord(ev_apply_end, /*stream=*/0);

  // Copy results back.
  err = cudaMemcpy(h_projected_values, d_projected,
                   (size_t)total_examples * num_proj * sizeof(float),
                   cudaMemcpyDeviceToHost);

  cudaEventRecord(ev_bridge_end, /*stream=*/0);
  cudaEventSynchronize(ev_bridge_end);

  {
    float ms_apply = 0, ms_total = 0;
    cudaEventElapsedTime(&ms_apply, ev_apply_begin, ev_apply_end);
    cudaEventElapsedTime(&ms_total, ev_bridge_begin, ev_bridge_end);
    if (out_ms_apply) *out_ms_apply = ms_apply;
    if (out_ms_other) {
      double other = (double)ms_total - (double)ms_apply;
      *out_ms_other = other < 0 ? 0 : other;
    }
  }

cleanup:
  cudaEventDestroy(ev_bridge_begin);
  cudaEventDestroy(ev_bridge_end);
  cudaEventDestroy(ev_apply_begin);
  cudaEventDestroy(ev_apply_end);

  if (d_selected_examples) cudaFree(d_selected_examples);
  if (d_projected) cudaFree(d_projected);
  if (d_node_row_off) cudaFree(d_node_row_off);
  if (d_col_offsets) cudaFree(d_col_offsets);
  if (d_flat_col_indices) cudaFree(d_flat_col_indices);
  if (d_flat_weights) cudaFree(d_flat_weights);

  return (int)err;
}

// Full-GPU nodewise split: Apply + RandomHistogram + HistogramSplit in one
// bridge call. Keeps projected values on device between stages.
//
// Inputs:
//   d_global_flat_data  - device features (previously uploaded).
//   d_global_labels     - device classification labels.
//   num_total_rows      - number of rows in the full training dataset.
//   h_selected_examples - host array of example indices for this node.
//   num_examples        - length of h_selected_examples.
//   num_proj            - number of projections.
//   h_flat_col_indices / h_flat_weights / h_col_offsets / total_nonzeros
//                       - CSR-format projections.
//   num_bins            - histogram bin count.
//   comp_method         - 0=entropy, 1=gini.
//   random_seed         - seed for RandomHistogram's host RNG.
//
// Scalar outputs (host):
//   best_proj / best_bin / best_gain / best_threshold / num_pos_examples.
//
// Sub-timings output (host, milliseconds). Measured via cudaEvent_t pairs
// bracketing each helper call — no per-stage cudaDeviceSynchronize so the
// stream keeps pipelining. A single cudaEventSynchronize at the end of the
// bridge reads all four numbers at once.
//   out_ms_apply - ApplyProjectionColumnADD stage.
//   out_ms_hist  - RandomHistogram stage.
//   out_ms_split - HistogramSplit stage.
//   out_ms_other - residual = bridge total − (apply + hist + split).
int oblique_gpu_find_best_split_nodewise(
    const float* d_global_flat_data,
    const unsigned int* d_global_labels,
    int num_total_rows,
    const unsigned int* h_selected_examples,
    int num_examples,
    int num_proj,
    const int* h_flat_col_indices,
    const float* h_flat_weights,
    const int* h_col_offsets,
    int total_nonzeros,
    int num_bins,
    int comp_method,
    unsigned long long random_seed,
    int* best_proj,
    int* best_bin,
    float* best_gain,
    float* best_threshold,
    int* num_pos_examples,
    double* out_ms_apply,
    double* out_ms_hist,
    double* out_ms_split,
    double* out_ms_other) {
  // Clear any sticky CUDA error from a previous call so cudaPeekAtLastError
  // downstream only reports errors produced by THIS invocation.
  (void)cudaGetLastError();

  cudaError_t err;

  // Device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;
  float* d_min_vals = nullptr;
  float* d_max_vals = nullptr;
  float* d_bin_widths = nullptr;
  int* d_hist0 = nullptr;
  int* d_hist1 = nullptr;
  int* d_hist2 = nullptr;
  float* d_candidate_splits = nullptr;

  // cuEvents for stage timings. One pair per stage plus an outer pair
  // around the whole bridge (used to compute the "other" residual).
  cudaEvent_t ev_bridge_begin = nullptr, ev_bridge_end = nullptr;
  cudaEvent_t ev_apply_begin = nullptr, ev_apply_end = nullptr;
  cudaEvent_t ev_hist_begin  = nullptr, ev_hist_end  = nullptr;
  cudaEvent_t ev_split_begin = nullptr, ev_split_end = nullptr;
  cudaEventCreate(&ev_bridge_begin);
  cudaEventCreate(&ev_bridge_end);
  cudaEventCreate(&ev_apply_begin);
  cudaEventCreate(&ev_apply_end);
  cudaEventCreate(&ev_hist_begin);
  cudaEventCreate(&ev_hist_end);
  cudaEventCreate(&ev_split_begin);
  cudaEventCreate(&ev_split_end);

  int return_err = 0;

  cudaEventRecord(ev_bridge_begin, /*stream=*/0);

  err = cudaMalloc(&d_selected_examples,
                   num_examples * sizeof(unsigned int));
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  err = cudaMalloc(&d_projected,
                   (size_t)num_proj * num_examples * sizeof(float));
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  err = cudaMemcpy(d_selected_examples, h_selected_examples,
                   num_examples * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  // Convert CSR to vector<vector> for ApplyProjectionColumnADD.
  {
    std::vector<std::vector<int>> proj_col_idx(num_proj);
    std::vector<std::vector<float>> proj_weights(num_proj);
    for (int p = 0; p < num_proj; p++) {
      int start = h_col_offsets[p];
      int end = h_col_offsets[p + 1];
      proj_col_idx[p].assign(h_flat_col_indices + start,
                              h_flat_col_indices + end);
      proj_weights[p].assign(h_flat_weights + start,
                              h_flat_weights + end);
    }

    // --- Stage 1: Apply projections + min/max + bin widths ---
    {
      cudaEventRecord(ev_apply_begin, /*stream=*/0);
      double internal_ms = 0;
      ApplyProjectionColumnADD(
          d_global_flat_data, d_selected_examples, d_projected,
          &d_min_vals, &d_max_vals, &d_bin_widths,
          proj_col_idx, proj_weights,
          num_examples, num_proj, num_total_rows,
          &internal_ms,
          /*gpu_mode=*/1 /*Random*/, /*verbose=*/false, /*stream=*/0);
      cudaEventRecord(ev_apply_end, /*stream=*/0);
    }

    // Copy min/max to host for RandomHistogram / HistogramSplit.
    std::vector<float> h_min_vals(num_proj);
    std::vector<float> h_max_vals(num_proj);
    err = cudaMemcpy(h_min_vals.data(), d_min_vals,
                     num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }
    err = cudaMemcpy(h_max_vals.data(), d_max_vals,
                     num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

    // --- Stage 2: Build class histograms + random split boundaries ---
    {
      std::mt19937 rng(random_seed);
      cudaEventRecord(ev_hist_begin, /*stream=*/0);
      RandomHistogram(
          d_projected, d_selected_examples, d_global_labels,
          h_min_vals.data(), h_max_vals.data(),
          &d_hist0, &d_hist1, &d_hist2, &d_candidate_splits,
          num_examples, num_bins, num_proj, rng, /*stream=*/0);
      cudaEventRecord(ev_hist_end, /*stream=*/0);
    }

    // --- Stage 3: Best-split argmax ---
    {
      cudaEventRecord(ev_split_begin, /*stream=*/0);
      double internal_ms = 0;
      HistogramSplit(
          d_hist0, d_hist1, d_hist2, d_candidate_splits,
          h_min_vals.data(), d_bin_widths,
          num_proj, num_bins, num_examples,
          best_proj, best_bin, best_gain, best_threshold, num_pos_examples,
          &internal_ms, /*verbose=*/false,
          comp_method, /*gpu_mode=*/1 /*Random*/, /*stream=*/0);
      cudaEventRecord(ev_split_end, /*stream=*/0);
    }
  }

  cudaEventRecord(ev_bridge_end, /*stream=*/0);
  cudaEventSynchronize(ev_bridge_end);

  {
    float ms_apply = 0, ms_hist = 0, ms_split = 0, ms_total = 0;
    cudaEventElapsedTime(&ms_apply, ev_apply_begin, ev_apply_end);
    cudaEventElapsedTime(&ms_hist,  ev_hist_begin,  ev_hist_end);
    cudaEventElapsedTime(&ms_split, ev_split_begin, ev_split_end);
    cudaEventElapsedTime(&ms_total, ev_bridge_begin, ev_bridge_end);
    if (out_ms_apply) *out_ms_apply = ms_apply;
    if (out_ms_hist)  *out_ms_hist  = ms_hist;
    if (out_ms_split) *out_ms_split = ms_split;
    if (out_ms_other) {
      double other = (double)ms_total - ((double)ms_apply + (double)ms_hist
                                         + (double)ms_split);
      *out_ms_other = other < 0 ? 0 : other;
    }
  }

cleanup:
  cudaEventDestroy(ev_bridge_begin);
  cudaEventDestroy(ev_bridge_end);
  cudaEventDestroy(ev_apply_begin);
  cudaEventDestroy(ev_apply_end);
  cudaEventDestroy(ev_hist_begin);
  cudaEventDestroy(ev_hist_end);
  cudaEventDestroy(ev_split_begin);
  cudaEventDestroy(ev_split_end);

  if (d_selected_examples) cudaFree(d_selected_examples);
  if (d_projected)          cudaFree(d_projected);
  if (d_min_vals)           cudaFree(d_min_vals);
  if (d_max_vals)           cudaFree(d_max_vals);
  if (d_bin_widths)         cudaFree(d_bin_widths);
  if (d_hist0)              cudaFree(d_hist0);
  if (d_hist1)              cudaFree(d_hist1);
  if (d_hist2)              cudaFree(d_hist2);
  if (d_candidate_splits)   cudaFree(d_candidate_splits);

  return return_err;
}

// Full-GPU nodewise split for EXACT mode: Apply + ThrustSortIndicesOnly +
// ExactSplit in one bridge call. Keeps projected values on device between
// stages. `ThrustSortIndicesOnly` takes ownership of `d_selected_examples`
// and `ExactSplit` takes ownership of `d_projected` / `d_sorted_indices`
// (they free internally), so we null those locals after the call.
int oblique_gpu_find_best_split_nodewise_exact(
    const float* d_global_flat_data,
    const unsigned int* d_global_labels,
    int num_total_rows,
    const unsigned int* h_selected_examples,
    int num_examples,
    int num_proj,
    const int* h_flat_col_indices,
    const float* h_flat_weights,
    const int* h_col_offsets,
    int total_nonzeros,
    int comp_method,
    int* best_proj,
    int* best_split,
    float* best_gain,
    float* best_threshold,
    int* num_pos_examples,
    double* out_ms_apply,
    double* out_ms_sort,
    double* out_ms_split,
    double* out_ms_other) {
  (void)cudaGetLastError();  // clear sticky errors

  cudaError_t err;

  // Device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;
  float* d_min_vals = nullptr;
  float* d_max_vals = nullptr;
  float* d_bin_widths = nullptr;
  unsigned int* d_sorted_indices = nullptr;

  // cuEvents for stage timings. See Random bridge for rationale.
  cudaEvent_t ev_bridge_begin = nullptr, ev_bridge_end = nullptr;
  cudaEvent_t ev_apply_begin  = nullptr, ev_apply_end  = nullptr;
  cudaEvent_t ev_sort_begin   = nullptr, ev_sort_end   = nullptr;
  cudaEvent_t ev_split_begin  = nullptr, ev_split_end  = nullptr;
  cudaEventCreate(&ev_bridge_begin);
  cudaEventCreate(&ev_bridge_end);
  cudaEventCreate(&ev_apply_begin);
  cudaEventCreate(&ev_apply_end);
  cudaEventCreate(&ev_sort_begin);
  cudaEventCreate(&ev_sort_end);
  cudaEventCreate(&ev_split_begin);
  cudaEventCreate(&ev_split_end);

  int return_err = 0;

  // Initialize scalar outputs to "no split" defaults. ExactSplit only writes
  // them when a positive-gain split is found.
  if (best_proj)        *best_proj = -1;
  if (best_split)       *best_split = -1;
  if (best_gain)        *best_gain = -1.0f;
  if (best_threshold)   *best_threshold = 0.0f;
  if (num_pos_examples) *num_pos_examples = 0;

  cudaEventRecord(ev_bridge_begin, /*stream=*/0);

  err = cudaMalloc(&d_selected_examples,
                   num_examples * sizeof(unsigned int));
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  err = cudaMalloc(&d_projected,
                   (size_t)num_proj * num_examples * sizeof(float));
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  err = cudaMemcpy(d_selected_examples, h_selected_examples,
                   num_examples * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }

  {
    std::vector<std::vector<int>> proj_col_idx(num_proj);
    std::vector<std::vector<float>> proj_weights(num_proj);
    for (int p = 0; p < num_proj; p++) {
      int start = h_col_offsets[p];
      int end = h_col_offsets[p + 1];
      proj_col_idx[p].assign(h_flat_col_indices + start,
                              h_flat_col_indices + end);
      proj_weights[p].assign(h_flat_weights + start,
                              h_flat_weights + end);
    }

    // --- Stage 1: Apply projections (Exact mode, no min/max) ---
    {
      cudaEventRecord(ev_apply_begin, /*stream=*/0);
      double internal_ms = 0;
      ApplyProjectionColumnADD(
          d_global_flat_data, d_selected_examples, d_projected,
          &d_min_vals, &d_max_vals, &d_bin_widths,
          proj_col_idx, proj_weights,
          num_examples, num_proj, num_total_rows,
          &internal_ms,
          /*gpu_mode=*/0 /*Exact*/, /*verbose=*/false, /*stream=*/0);
      cudaEventRecord(ev_apply_end, /*stream=*/0);
    }

    // --- Stage 2: Sort example indices per projection ---
    err = cudaMalloc(&d_sorted_indices,
                     (size_t)num_proj * num_examples * sizeof(unsigned int));
    if (err != cudaSuccess) { return_err = (int)err; goto cleanup; }
    {
      cudaEventRecord(ev_sort_begin, /*stream=*/0);
      ThrustSortIndicesOnly(d_projected, d_sorted_indices,
                             d_selected_examples,
                             num_examples, num_proj, /*stream=*/0);
      // ThrustSortIndicesOnly frees d_selected_examples internally.
      d_selected_examples = nullptr;
      cudaEventRecord(ev_sort_end, /*stream=*/0);
    }

    // --- Stage 3: Exact split-finding ---
    {
      cudaEventRecord(ev_split_begin, /*stream=*/0);
      double internal_ms = 0;
      ExactSplit(
          d_sorted_indices, d_global_labels,
          best_gain, best_split, best_threshold, best_proj,
          num_examples, num_proj, d_projected,
          &internal_ms, /*verbose=*/false, comp_method, /*stream=*/0);
      // ExactSplit frees d_projected and d_sorted_indices internally.
      d_projected = nullptr;
      d_sorted_indices = nullptr;
      cudaEventRecord(ev_split_end, /*stream=*/0);
    }

    // Derive right-side count from the chosen split index.
    if (num_pos_examples && best_split && *best_split >= 0) {
      *num_pos_examples = num_examples - *best_split;
    }
  }

  cudaEventRecord(ev_bridge_end, /*stream=*/0);
  cudaEventSynchronize(ev_bridge_end);

  {
    float ms_apply = 0, ms_sort = 0, ms_split = 0, ms_total = 0;
    cudaEventElapsedTime(&ms_apply, ev_apply_begin, ev_apply_end);
    cudaEventElapsedTime(&ms_sort,  ev_sort_begin,  ev_sort_end);
    cudaEventElapsedTime(&ms_split, ev_split_begin, ev_split_end);
    cudaEventElapsedTime(&ms_total, ev_bridge_begin, ev_bridge_end);
    if (out_ms_apply) *out_ms_apply = ms_apply;
    if (out_ms_sort)  *out_ms_sort  = ms_sort;
    if (out_ms_split) *out_ms_split = ms_split;
    if (out_ms_other) {
      double other = (double)ms_total - ((double)ms_apply + (double)ms_sort
                                         + (double)ms_split);
      *out_ms_other = other < 0 ? 0 : other;
    }
  }

cleanup:
  cudaEventDestroy(ev_bridge_begin);
  cudaEventDestroy(ev_bridge_end);
  cudaEventDestroy(ev_apply_begin);
  cudaEventDestroy(ev_apply_end);
  cudaEventDestroy(ev_sort_begin);
  cudaEventDestroy(ev_sort_end);
  cudaEventDestroy(ev_split_begin);
  cudaEventDestroy(ev_split_end);

  if (d_selected_examples) cudaFree(d_selected_examples);
  if (d_projected)          cudaFree(d_projected);
  if (d_min_vals)           cudaFree(d_min_vals);
  if (d_max_vals)           cudaFree(d_max_vals);
  if (d_bin_widths)         cudaFree(d_bin_widths);
  if (d_sorted_indices)     cudaFree(d_sorted_indices);

  return return_err;
}

bool oblique_gpu_check_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("ObliqueGPU: CUDA device: %s (compute %d.%d, %d MB)\n",
         prop.name, prop.major, prop.minor,
         (int)(prop.totalGlobalMem / (1024 * 1024)));
  return true;
}

}  // extern "C"
