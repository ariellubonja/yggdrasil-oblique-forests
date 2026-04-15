// CUDA implementations for ObliqueGpuComputer.
// Compiled by nvcc via cuda_library rule.

#include <cuda_runtime.h>
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
    float* h_max_vals           // [num_proj]
) {
  cudaError_t err;

  // Allocate device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;

  err = cudaMalloc(&d_selected_examples, num_examples * sizeof(unsigned int));
  if (err != cudaSuccess) return (int)err;

  err = cudaMalloc(&d_projected, num_proj * num_examples * sizeof(float));
  if (err != cudaSuccess) { cudaFree(d_selected_examples); return (int)err; }

  err = cudaMemcpy(d_selected_examples, h_selected_examples,
                   num_examples * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cudaFree(d_selected_examples); cudaFree(d_projected); return (int)err; }

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

  // Call labmate's kernel wrapper.
  float* d_min_vals = nullptr;
  float* d_max_vals = nullptr;
  float* d_bin_widths = nullptr;
  double elapsed_ms = 0;

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

cleanup:
  cudaFree(d_selected_examples);
  cudaFree(d_projected);
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
    int max_examples_per_node)
{
  cudaError_t err;

  // Allocate device buffers.
  unsigned int* d_selected_examples = nullptr;
  float* d_projected = nullptr;
  int* d_node_row_off = nullptr;
  int* d_col_offsets = nullptr;
  int* d_flat_col_indices = nullptr;
  float* d_flat_weights = nullptr;

  const int num_segments = num_nodes * num_proj;

  err = cudaMalloc(&d_selected_examples, total_examples * sizeof(unsigned int));
  if (err != cudaSuccess) return (int)err;
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

  // Launch single 3D kernel for all nodes.
  ApplyProjectionColumnADDMultiNode(
      d_global_flat_data, d_selected_examples, d_projected,
      d_node_row_off, d_col_offsets, d_flat_col_indices, d_flat_weights,
      max_examples_per_node, num_nodes, num_proj, num_total_rows, 0);

  // Copy results back.
  err = cudaMemcpy(h_projected_values, d_projected,
                   (size_t)total_examples * num_proj * sizeof(float),
                   cudaMemcpyDeviceToHost);

cleanup:
  if (d_selected_examples) cudaFree(d_selected_examples);
  if (d_projected) cudaFree(d_projected);
  if (d_node_row_off) cudaFree(d_node_row_off);
  if (d_col_offsets) cudaFree(d_col_offsets);
  if (d_flat_col_indices) cudaFree(d_flat_col_indices);
  if (d_flat_weights) cudaFree(d_flat_weights);

  return (int)err;
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
