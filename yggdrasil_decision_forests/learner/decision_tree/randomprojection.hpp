// If num_cols is large, use global memory instead of shared memory for base_indices.
// Each thread performs its own shuffle.
// You can store the output total_col_indices on the device, then copy back to host as needed.
// If this needs to run repeatedly, set up curandState once per thread for reuse.

#include <cuda_runtime.h>
#include <vector>


#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _status = (call);                                         \
        if (_status != cudaSuccess) {                                         \
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(_status)        \
                      << " (code " << _status << ") "                         \
                      << "in " << __FILE__ << ':' << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#ifdef PROFILE1
  #include <chrono>
  #include <iostream>

  // ------------------------------------------------------------------
  // 1. mark the beginning of the timed section
  // ------------------------------------------------------------------
  #define TIMER_START(tag)                                               \
      auto _t_##tag##_start = std::chrono::steady_clock::now()

  // ------------------------------------------------------------------
  // 2. mark the end of the timed section – stores the elapsed time
  //    (does *not* print)
  // ------------------------------------------------------------------
  #define TIMER_STOP(tag)                                                \
      auto _t_##tag##_elapsed =                                          \
          std::chrono::duration<double, std::milli>(                     \
              std::chrono::steady_clock::now() - _t_##tag##_start)

  // ------------------------------------------------------------------
  // 3. print the elapsed time that was computed by TIMER_STOP
  // ------------------------------------------------------------------
  #define TIMER_PRINT(tag, msg)                                          \
      std::cout << (msg) << ": " << _t_##tag##_elapsed.count() << " ms\n"

#else   // -------------------------------------------------------------

  /* no-op versions so the code still compiles and optimises away */
  #define TIMER_START(tag)
  #define TIMER_STOP(tag)
  #define TIMER_PRINT(tag, msg)

#endif

void warmupfunction();

void ApplyProjectionColumnADD (const float* d_flat_data,
                                const unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<int>>& projection_col_idx,
                                std::vector<std::vector<float>>& projection_weights,
                                const int num_rows,  //num_rows
                                const int num_proj, //num_proj
                                const int train_dataset,
                                double* elapsed_ms,
                                const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                const bool verbose,
                                cudaStream_t cuda_stream = 0
                              );

// Multi-node variant: single kernel launch for all nodes at a BFS depth level.
void ApplyProjectionColumnADDMultiNode(
    const float* d_flat_data,
    const unsigned int* d_selected_examples,
    float* d_projected,
    const int* d_node_row_off,
    const int* d_col_offset,
    const int* d_flat_col_data,
    const float* d_flat_weights,
    int max_examples_per_node,
    int num_nodes,
    int num_proj,
    int num_total_rows,
    cudaStream_t cuda_stream = 0);

void EqualWidthHistogram (const float* __restrict__ d_col_add_projected, //attributes
const unsigned int* __restrict__ selected_examples, //selected examples
const unsigned int* __restrict__ d_global_labels_data,
float* d_min_vals,
float* d_max_vals,
float* h_min_vals,
float* h_max_vals,
float* d_bin_widths,
int** h_hist0,
int** h_hist1,
int** h_hist2,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj,
cudaStream_t cuda_stream = 0
);

void RandomHistogram (const float* __restrict__ d_col_add_projected, //attributes
const unsigned int* __restrict__ selected_examples, //selected examples
const unsigned int* __restrict__ d_global_labels_data,
float* h_min_vals,
float* h_max_vals,
int** h_hist0,
int** h_hist1,
int** h_hist2,
float** d_candidate_splits,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj,
std::mt19937& random,
cudaStream_t cuda_stream = 0
);
                    
void HistogramSplit (const int* d_hist_class0,
                    const int* d_hist_class1,
                    const int* d_hist_class2,
                    const float* d_candidate_splits,
                    const float* h_min_vals,
                    const float* d_bin_widths,
                    const int num_proj,
                    const int num_bins,
                    const int num_rows,
                    int* best_proj,
                    int* best_bin_out,
                    float* best_gain_out,
                    float* best_threshold_out,
                    int* num_pos_examples_out,
                    double* elapsed_ms,
                    bool verbose,
                    const int comp_method,
                    const int gpu_mode,
                    cudaStream_t cuda_stream = 0
                    );

void ThrustSortIndicesOnly(float* d_proj_values, 
                          unsigned int* d_row_ids,
                          unsigned int* d_selected_examples, 
                          int num_rows, 
                          int num_proj,
                          cudaStream_t cuda_stream = 0);

void ExactSplit(
    unsigned int* d_sorted_indices,  // [num_proj * num_rows]
    const unsigned int* d_labels,          // [num_proj * num_rows]
    float* best_gain_out, // [num_proj], initial best gain values
    int* best_split_out, // [num_proj], initial best split values
    float* best_threshold_out,
    int* best_proj,
    const int num_rows,
    const int num_proj,
    float* d_col_add_projected,  // [num_proj * num_rows]
    double* elapsed_ms,
    bool verbose,
    const int comp_method,
    cudaStream_t cuda_stream = 0
    );

