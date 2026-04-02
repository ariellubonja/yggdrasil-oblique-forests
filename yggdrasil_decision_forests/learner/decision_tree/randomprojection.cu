// Online C++ compiler to run C++ program online


#include <iostream>
#include <vector>
#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <random>
#include <cmath>
#include <limits>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <vector_types.h>
#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "randomprojection.hpp"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>


__global__ void warmup() {
    // Empty kernel for warm-up
}

void warmupfunction() {
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
}

__global__ void ColumnAddProjectionKernel(
    const float* __restrict__ dataset,
    const unsigned int* d_selected_examples,                 
    float* projected, 
    const int* __restrict__ col_offset, //number of columns per projection  
    const int* __restrict__ flat_col_data, //flattened column indices for all projections
    const float* __restrict__ flat_weights, //flattened weights for all projections              
    const int num_selected_examples,
    const int num_total_rows,
    const int num_proj) 
{
    int TPB = blockDim.x;  // threads per block
    
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= num_proj || row >= num_selected_examples) return;

    int selected_features_count = col_offset[col + 1] - col_offset[col];
    int offset = col_offset[col];

    float sum = 0.0f;
    unsigned int example_idx = d_selected_examples[row];

    for (int i = 0; i < selected_features_count; ++i) {
        int feature_idx = flat_col_data[offset + i];
        float weight = flat_weights[offset + i];
        sum += weight * dataset[feature_idx * num_total_rows + example_idx];
    }

    projected[col * num_selected_examples + row] = sum;
}

__global__ void ColumnAddComputeMinMaxCombined(
    const float* __restrict__ dataset,
    const unsigned int* d_selected_examples,                 
    float* projected, 
    const int* __restrict__ col_offset, //number of columns per projection  
    const int* __restrict__ flat_col_data, //flattened column indices for all projections
    const float* __restrict__ flat_weights, //flattened weights for all projections              
    const int num_selected_examples,
    const int num_total_rows,
    const int num_proj,
    float* __restrict__ d_block_min,
    float* __restrict__ d_block_max
    ) 
{
    int TPB = blockDim.x;  // threads per block
    
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;

    if (col >= num_proj || row >= num_selected_examples) return;

    int selected_features_count = col_offset[col + 1] - col_offset[col];
    int offset = col_offset[col];

    float sum = 0.0f;
    unsigned int example_idx = d_selected_examples[row];

    for (int i = 0; i < selected_features_count; ++i) {
        int feature_idx = flat_col_data[offset + i];
        float weight = flat_weights[offset + i];
        sum += weight * dataset[feature_idx * num_total_rows + example_idx];
    }

    projected[col * num_selected_examples + row] = sum;
    __syncthreads();

   
    int tid     = threadIdx.x;
    if (col >= num_proj) return;

    int base_idx = col * num_selected_examples;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // strided loop over rows
    for (int i = blockIdx.x * blockDim.x + tid; i < num_selected_examples; i += gridDim.x * blockDim.x) {
        float val = projected[base_idx + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // store per-thread local min/max in shared
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads(); //Synchronize threads within the block, ensuring all threads have updated shared memory before proceeding, prevents race conditions., controls the order of execution among threads in the same block.
    }

    // write block result
    if (tid == 0) {
        int idx = col * gridDim.x + blockIdx.x;
        d_block_min[idx] = shared_min[0];
        d_block_max[idx] = shared_max[0];
    }
}

void ApplyProjectionColumnADD (const float* d_flat_data,
                                const unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<int>>& projection_col_idx,
                                std::vector<std::vector<float>>& projection_weights,
                                const int num_selected_examples,  //num_rows
                                const int num_proj, //num_proj
                                const int num_total_rows,
                                double* elapsed_apply_ms,
                                const int gpu_mode, //0: Exact, 1: Random, 2: Equal Width
                                const bool verbose,
                                cudaStream_t cuda_stream
                              )
{
    ////////////////////////Data Preparation for col per projection on Host///////////////////////////
    int result_size = num_selected_examples * num_proj;
    const int P = static_cast<int>(projection_col_idx.size());
    std::vector<int> col_per_proj(P); //Number of columns per projection

    std::transform(projection_col_idx.begin(), projection_col_idx.end(),
                col_per_proj.begin(),
                [](const auto& v) { return static_cast<int>(v.size()); });

    ////////////////////////////////////////////////////////////////////////

    //////////////////////exclusive scan to get offsets///////////////////////////
    std::vector<int> offset(col_per_proj.size() + 1);
    /*  offset[0] must be 0.  Write the exclusive scan starting at offset[1]. */
    offset[0] = 0;

    if (P > 0) {
        std::inclusive_scan(col_per_proj.begin(),           // first
                        col_per_proj.end(),             // last (exclusive)
                        offset.begin() + 1);    
    }

    ////////////////////////////////////////////////////////////////////////  

    int total_size = offset.back();
    ///////////////////////copy and flatten projection data structures///////////////////////////
    std::vector<int> flat_projection_col_idx(total_size);

    int* dst = flat_projection_col_idx.data();
    for (const auto& v : projection_col_idx) {
        const std::size_t bytes = v.size() * sizeof(int); //Flattening column indices
        std::memcpy(dst, v.data(), bytes);   // one bulk copy per inner vector
        dst += v.size();
    }

    std::vector<float> flat_projection_weights(total_size);

    float* dst_w = flat_projection_weights.data();
    for (const auto& v : projection_weights) {
        std::memcpy(dst_w, v.data(), v.size() * sizeof(float)); //Flattening weights
        dst_w += v.size();
    }

    /////////////////////////////////////////////////End of Data Preparation//////////////////////////////////////

    
    ///////////////////////GPU Memory Allocation and Copy///////////////////////////  

    int* d_offset = nullptr; 
    int* d_flat_projection_col_idx = nullptr;
    float* d_flat_projection_weights = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **)&d_offset, offset.size() * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_flat_projection_col_idx, flat_projection_col_idx.size() * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_flat_projection_weights, flat_projection_weights.size() * sizeof(float), cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offset, offset.data(), offset.size() * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_flat_projection_col_idx, flat_projection_col_idx.data(), flat_projection_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));    
    CUDA_CHECK(cudaMemcpyAsync(d_flat_projection_weights, flat_projection_weights.data(), flat_projection_weights.size() * sizeof(float), cudaMemcpyHostToDevice, cuda_stream));  

    // Launch CUDA kernel
    dim3 blockDim(256);
    dim3 gridDim((num_selected_examples + blockDim.x - 1) / blockDim.x, num_proj);
    
    if (gpu_mode == 0) { //Exact
        TIMER_START(ExactCAKernel);    
        ColumnAddProjectionKernel<<<gridDim, blockDim, 0, cuda_stream>>>(d_flat_data,
                                                        d_selected_examples,
                                                        d_col_add_projected,
                                                        d_offset,
                                                        d_flat_projection_col_idx,
                                                        d_flat_projection_weights,
                                                        num_selected_examples,
                                                        num_total_rows,
                                                        num_proj);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
        TIMER_STOP(ExactCAKernel);
        TIMER_PRINT(ExactCAKernel, "GPU Exact Column Add Kernel Time taken");
    }
    else if (gpu_mode == 2 || gpu_mode == 1) { // Equal Width or Random
        TIMER_START(HistogramCombinedKernel);
        const int total_blocks = num_proj * gridDim.x;
        float* d_min_vals;
        float* d_max_vals;          
        float* d_bin_widths;
        CUDA_CHECK(cudaMallocAsync(&d_min_vals, num_proj * sizeof(float), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_max_vals, num_proj * sizeof(float), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_bin_widths, num_proj * sizeof(float), cuda_stream));
        *d_min_vals_out = d_min_vals;
        *d_max_vals_out = d_max_vals;
        *d_bin_widths_out = d_bin_widths;

        float* d_block_min;
        float* d_block_max;
        CUDA_CHECK(cudaMallocAsync(&d_block_min, total_blocks * sizeof(float), cuda_stream));  // Allocate intermediate buffers
        CUDA_CHECK(cudaMallocAsync(&d_block_max, total_blocks * sizeof(float), cuda_stream));  // Allocate intermediate buffers

        size_t shmem = 2 * blockDim.x * sizeof(float);

        TIMER_START(CombinedKernel);
        ColumnAddComputeMinMaxCombined<<<gridDim, blockDim, shmem, cuda_stream>>>(d_flat_data,
                                                                        d_selected_examples,
                                                                        d_col_add_projected,
                                                                        d_offset,
                                                                        d_flat_projection_col_idx,
                                                                        d_flat_projection_weights,
                                                                        num_selected_examples,
                                                                        num_total_rows,
                                                                        num_proj,
                                                                        d_block_min,
                                                                        d_block_max);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream)); //It blocks the CPU until the device has completed all preceding requested tasks.
        TIMER_STOP(CombinedKernel);

        TIMER_START(ReduceMinMax);
        thrust::device_vector<int> d_begin(num_proj);
        thrust::device_vector<int> d_end(num_proj);

        // Fill begin offsets: 0, num_blocks, 2*num_blocks, ...
        thrust::sequence(
                    thrust::cuda::par.on(cuda_stream),
                    d_begin.begin(),
                    d_begin.end(),
                    0,
                    (int)gridDim.x);
        // Fill end offsets: num_blocks, 2*num_blocks, 3*num_blocks, ...
        thrust::sequence(
                    thrust::cuda::par.on(cuda_stream),
                    d_end.begin(),
                    d_end.end(),
                    (int)gridDim.x,
                    (int)gridDim.x);
        int* d_begin_ptr = thrust::raw_pointer_cast(d_begin.data());
        int* d_end_ptr   = thrust::raw_pointer_cast(d_end.data());

        // ====================================================================
        // 🚀 CUB: MIN and Maxper projection
        // ====================================================================
    

        size_t temp_bytes = 0;
        void* d_temp = nullptr;

        //query temp sizes
        cub::DeviceSegmentedReduce::Min(
            nullptr, temp_bytes,
            d_block_min,
            d_min_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            cuda_stream
        );

        CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, cuda_stream));

        cub::DeviceSegmentedReduce::Min(
            d_temp, temp_bytes,
            d_block_min,
            d_min_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            cuda_stream
        );
        
        cub::DeviceSegmentedReduce::Max(
            d_temp, temp_bytes,
            d_block_max,
            d_max_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            cuda_stream
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        //CUDA_CHECK(cudaDeviceSynchronize()); //It blocks the CPU until the device has completed all preceding requested tasks.
        CUDA_CHECK(cudaFreeAsync(d_temp, cuda_stream));
        CUDA_CHECK(cudaFreeAsync(d_block_min, cuda_stream));
        CUDA_CHECK(cudaFreeAsync(d_block_max, cuda_stream));
        TIMER_STOP(ReduceMinMax);
        TIMER_STOP(HistogramCombinedKernel);

        TIMER_PRINT(CombinedKernel, "GPU Histogram Column Add and Min/Max Kernel Time taken");
        TIMER_PRINT(ReduceMinMax, "GPU Histogram Reduce Min/Max Time taken");
        TIMER_PRINT(HistogramCombinedKernel, "GPU Histogram Column Add + Reduce Min/Max Time taken");

    }
    
    // Free device memory
    CUDA_CHECK(cudaFreeAsync(d_offset, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_flat_projection_col_idx, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_flat_projection_weights, cuda_stream));

}


template <int BLOCK_SIZE>
__global__ void BuildHistogramEqualWidthKernel(
    const float* __restrict__ d_attributes, //attributes
    const unsigned int*__restrict__ d_row_indices, //selected examples
    const unsigned int* __restrict__ d_labels,
    int* d_hist_class0,          // [num_samples], values 0 or 1
    int* d_hist_class1,           // [NUM_BINS], count of
    int* d_hist_class2, 
    const float* d_min_vals,
    const float* d_max_vals,
    float* d_bin_widths,
    const int num_rows,
    const int num_proj,
    const int num_bins //candidate_splits 256 not 257
)
{
    extern __shared__ int shared_mem[];
    int proj_id = blockIdx.y;
    if (proj_id >= num_proj) return;
  
    // 1. Put zeroes in shared histogram
    for (int i = threadIdx.x; i < 3 * (num_bins); i += blockDim.x)
        shared_mem[i] = 0;
    __syncthreads();

    const std::size_t col_offset = std::size_t(proj_id) * num_rows;
    const int tid    = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_rows) {
        float val   = d_attributes[col_offset + tid];      
        unsigned int   label = d_labels[d_row_indices[tid]];
     
        // int bin = (val >= d_max_vals[proj_id] - 0.5 * d_bin_widths [proj_id]) //Assign to last bin if val is equal to max_val - 0.5*bin_width
        //           ? (num_bins - 1)
        //           : round((val - d_min_vals[proj_id]) / d_bin_widths[proj_id]);//have total num_bins

        // int bin = min(__float2int_rn((val - d_min_vals[proj_id]) / d_bin_widths[proj_id]), num_bins - 1);
        int bin = __float2int_rn((val - d_min_vals[proj_id]) / d_bin_widths[proj_id]);


        // if (label == 1) {
        //     atomicAdd(&shared_mem[num_bins + bin], 1); 
        // } 
        // else if (label == 2) {
        //     atomicAdd(&shared_mem[2 * (num_bins) + bin], 1);
        // }
        // else {
        //     atomicAdd(&shared_mem[bin], 1);
        // }
        atomicAdd(&shared_mem[label * (num_bins) + bin], 1);
    }
    __syncthreads();

    int offset =  proj_id * (num_bins);
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&d_hist_class1[offset + i], shared_mem[num_bins + i]);
        atomicAdd(&d_hist_class2[offset + i], shared_mem[2 * (num_bins) + i ]);
        atomicAdd(&d_hist_class0[offset + i], shared_mem[i]);
    } 
}

struct index_to_proj
{
    int rows_per_proj;
    __host__ __device__
    int operator()(int idx) const { return idx / rows_per_proj; } //Linear index into "projection buckets"
};


void EqualWidthHistogram (const float* __restrict__ d_col_add_projected, //attributes
                          const unsigned int* __restrict__ d_selected_examples, //selected examples
                          const unsigned int* __restrict__ d_global_labels_data,
                          float* d_min_vals,
                          float* d_max_vals,
                          float* h_min_vals,
                          float* h_max_vals,
                          float* d_bin_widths,
                          int** d_prefix_0_out,
                          int** d_prefix_1_out,
                          int** d_prefix_2_out,
                          const int num_rows, //selected_examples.size()
                          const int num_bins,
                          const int num_proj,
                          cudaStream_t cuda_stream
                          )
    {
    ///////////////////////Calculate Bin Widths///////////////////////////
    
    TIMER_START(EWBinning);
    std::vector<float> h_bin_widths(num_proj);
    float inv_bins = 1.0f / (float)(num_bins - 1);
    for (int proj_id = 0; proj_id < num_proj; ++proj_id) {
        float min_val = h_min_vals[proj_id];
        float max_val = h_max_vals[proj_id];
        float bin_width = (max_val > min_val)
                                ? (max_val - min_val) * inv_bins
                                : 1.0f;
        h_bin_widths[proj_id] = bin_width + 1e-6f; //Add small epsilon to avoid division by zero
    }
    CUDA_CHECK(cudaMemcpyAsync(d_bin_widths, h_bin_widths.data(), num_proj * sizeof(float), cudaMemcpyHostToDevice, cuda_stream)); 



    const int BLOCK = 256;
    //////////////////////////Allocate Device Memory/////////////////////////////////////////
    // auto startAlloc = std::chrono::steady_clock::now();
    int* d_hist_class0;
    int* d_hist_class1;
    int* d_hist_class2;
    CUDA_CHECK(cudaMallocAsync(&d_hist_class0, num_proj * (num_bins) * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_hist_class1, num_proj * (num_bins) * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_hist_class2, num_proj * (num_bins) * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_hist_class0, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_hist_class1, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_hist_class2, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));

    
    ///////////////////////BuildHistogramEqualWidthKernel/////////////////////////////////////////
 
    //auto startHist = std::chrono::steady_clock::now();
    int threads_per_block_hist = num_bins;
    //printf("threads_per_block_hist: %d\n", threads_per_block_hist);
    int num_elements_per_thread = 1;
    int blocks_per_grid_hist = (num_rows/num_elements_per_thread + threads_per_block_hist - 1) / threads_per_block_hist;
    //printf("blocks_per_grid_hist: %d\n", blocks_per_grid_hist);
    dim3 grid_hist(blocks_per_grid_hist, num_proj); //single dimension grid/projection
    //printf("grid_hist: (%d, %d)\n", grid_hist.x, grid_hist.y);
    int sharedMemSize = 3 * (num_bins) * sizeof(int); // For Hist 0, Hist 1, and Hist 2
    BuildHistogramEqualWidthKernel<BLOCK><<<grid_hist, threads_per_block_hist, sharedMemSize, cuda_stream>>>(d_col_add_projected, d_selected_examples, d_global_labels_data,
                                                                             d_hist_class0, d_hist_class1, d_hist_class2,
                                                                             d_min_vals, d_max_vals, d_bin_widths,
                                                                             num_rows, num_proj, num_bins);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    TIMER_STOP(EWBinning);

    ///////////////////////Inclusive Scan per projection per class/////////////////////////////////////////
    TIMER_START(EWInclusiveScan);
    int total_rows = (num_bins) * num_proj;

    int* d_prefix_2;
    int* d_prefix_1;
    int* d_prefix_0;
    CUDA_CHECK(cudaMallocAsync(&d_prefix_2, (total_rows * sizeof(int)), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_prefix_1, (total_rows * sizeof(int)), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_prefix_0, (total_rows * sizeof(int)), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_prefix_2, 0, total_rows * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_prefix_1, 0, total_rows * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_prefix_0, 0, total_rows * sizeof(int), cuda_stream));

    auto counting_begin = thrust::make_counting_iterator<int>(0);
    auto keys_begin = thrust::make_transform_iterator(counting_begin, index_to_proj{num_bins});

    auto d_hist_class2_ptr   = thrust::device_pointer_cast(d_hist_class2);        // input Negative class
    auto d_prefix_2_ptr = thrust::device_pointer_cast(d_prefix_2);  // output

    auto d_hist_class1_ptr   = thrust::device_pointer_cast(d_hist_class1);        // input Positive class
    auto d_prefix_1_ptr = thrust::device_pointer_cast(d_prefix_1);  // output
    
    // auto d_hist_class0_ptr   = thrust::device_pointer_cast(d_hist_class0);        // input Not Used class
    // auto d_prefix_0_ptr = thrust::device_pointer_cast(d_prefix_0);  // output


    thrust::inclusive_scan_by_key( //label = 0 second
        thrust::cuda::par.on(cuda_stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_hist_class2_ptr,                      /* values begin   */
        d_prefix_2_ptr,                    /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()
    ); 

    thrust::inclusive_scan_by_key( //label = 1 first
        thrust::cuda::par.on(cuda_stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_hist_class1_ptr,                      /* values begin   */
        d_prefix_1_ptr,                /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()              /* inclusive sum  */
    );            
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    TIMER_STOP(EWInclusiveScan);

    TIMER_PRINT(EWBinning, "GPU EW Binwidth Calculations and Histogram Binning Kernel Time taken");
    TIMER_PRINT(EWInclusiveScan, "GPU EW Prefix Sum Time taken");

    *d_prefix_0_out = d_prefix_0;
    *d_prefix_1_out = d_prefix_1;
    *d_prefix_2_out = d_prefix_2;

    CUDA_CHECK(cudaFreeAsync(d_hist_class0, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_hist_class1, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_hist_class2, cuda_stream));
    CUDA_CHECK(cudaFreeAsync((void *)d_selected_examples, cuda_stream));
    CUDA_CHECK(cudaFreeAsync((void *)d_col_add_projected, cuda_stream));
}

__device__ __forceinline__
int lower_bound_naive_device(const float* __restrict__ bins,
                             int K,
                             float v)
{
    int b = 0;
    // Naive scanning
    // Count how many bins[j] < v
    for (int j = 0; j < K; ++j) {
        if (bins[j] < v) {
            b = j + 1;
        } else {
            break;   // bins are sorted
        }
    }
    return b;  // in [0, K]
}


// Return the bin index j so that   bounds[j] <= x < bounds[j+1].
// The array "bounds" must contain exactly num_bounds values sorted
// ascending.  The number of bins is therefore (num_bounds+1), and the
// result is in 0 … num_bounds.


template <int BLOCK_SIZE>
__global__ void BuildHistogramRandomKernel(
    const float* __restrict__ d_attributes, //attributes
    const unsigned int*__restrict__ d_row_indices, //selected examples
    const unsigned int* __restrict__ d_labels,
    int* d_hist_class0,          // [num_samples], values 0 or 1
    int* d_hist_class1,           // [NUM_BINS], count of
    int* d_hist_class2, 
    const int num_rows,
    const int num_proj,
    const int num_bins, //YDF bins have candidate splits + 1
    const float* d_candidate_splits //candidate_splits 
)
{
    extern __shared__ int shared_mem[];
    int proj_id = blockIdx.y;
    if (proj_id >= num_proj) return;
  
    // 1. Put zeroes in shared histogram
    for (int i = threadIdx.x; i < 3 * (num_bins); i += blockDim.x)
        shared_mem[i] = 0;
    __syncthreads();

    const std::size_t col_offset = std::size_t(proj_id) * num_rows;
    const int tid    = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_rows) {
        float val   = d_attributes[col_offset + tid]; //projected values    
        unsigned int   label = d_labels[d_row_indices[tid]];  
        
        //cub::LowerBound on the GPU to convert a value into a bin index given an array of sorted boundaries.  
        //The same helper can be called for many projections simply by pointing it at the first element of the block that belongs to the current projection.
        // Binary search to find the appropriate bin using LowerBound
        int bounds = num_bins - 1;
        const float* candidate_splits_proj = d_candidate_splits + proj_id * (bounds);
        int bin = cub::LowerBound(candidate_splits_proj, bounds, val);
        //int bin = lower_bound_naive_device(candidate_splits_proj, bounds, val);

        // if (label == 1) {
        //     atomicAdd(&shared_mem[num_bins + bin], 1); 
        // } 
        // else if (label == 2) {
        //     atomicAdd(&shared_mem[2 * (num_bins) + bin], 1);
        // }
        // else {
        //     atomicAdd(&shared_mem[bin], 1);
        // }
        atomicAdd(&shared_mem[label * (num_bins) + bin], 1);
    }
    __syncthreads();

    int offset =  proj_id * (num_bins);
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&d_hist_class1[offset + i], shared_mem[num_bins + i]);
        atomicAdd(&d_hist_class2[offset + i], shared_mem[2 * (num_bins) + i ]);
        atomicAdd(&d_hist_class0[offset + i], shared_mem[i]);
    } 
}


void RandomHistogram (const float* __restrict__ d_col_add_projected, //attributes
                          const unsigned int* __restrict__ d_selected_examples, //selected examples
                          const unsigned int* __restrict__ d_global_labels_data,
                          float* h_min_vals,
                          float* h_max_vals,
                          int** d_prefix_0_out,
                          int** d_prefix_1_out,
                          int** d_prefix_2_out,
                          float** d_candidate_splits_out,
                          const int num_rows, //selected_examples.size()
                          const int num_bins,
                          const int num_proj,
                          std::mt19937& random,
                          cudaStream_t cuda_stream
                          )
    {
        TIMER_START(RandomBinning);
        ///////////////////////Generate Random Candidate Splits/////////////////////////////////////////
        std::vector<float> candidate_splits((num_bins - 1) * num_proj);
        //Generate random candidate splits per projection on host
        for (int p = 0; p < num_proj; ++p)
        {
            float min_val = h_min_vals[p];
            float max_val = h_max_vals[p];
            int base_idx = p * (num_bins - 1);
            std::uniform_real_distribution<float> threshold_distribution(min_val, max_val);
            for (int b = 0; b < num_bins - 1; ++b)
            {
                candidate_splits[base_idx + b] = threshold_distribution(random);
            }
        }
        //////////////////

        //Sorting per projection candidate splits on device
        thrust::device_vector<float> d_candidate_splits(candidate_splits.begin(), candidate_splits.end());
        thrust::device_vector<int> d_offsets(num_proj + 1);
        float* d_sorted_candidate_splits = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_sorted_candidate_splits, sizeof(float) * num_proj * (num_bins - 1), cuda_stream));

        thrust::sequence(
            thrust::cuda::par.on(cuda_stream),
            d_offsets.begin(),
            d_offsets.end(),
            0,
            num_bins - 1);   // [0, num_bins, 2*num_bins, ...]

        size_t temp_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,                      // query mode
            temp_bytes,                   // output: required temp storage
            d_candidate_splits.data().get(), // keys in
            d_sorted_candidate_splits, // keys out (in-place)
            num_proj * (num_bins - 1),           // total number of keys
            num_proj,                      // number of segments
            d_offsets.data().get(),        // segment begin offsets
            d_offsets.data().get() + 1,
            0,                              // begin bit
            32,                            // end bit                               
            cuda_stream);
        void* d_temp = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, cuda_stream));

        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp,           // temp storage
            temp_bytes,
            d_candidate_splits.data().get(),
            d_sorted_candidate_splits,
            num_proj * (num_bins - 1),
            num_proj,
            d_offsets.data().get(),
            d_offsets.data().get() + 1,
            0,                              // begin bit
            32,                            // end bit                               
            cuda_stream);
        CUDA_CHECK(cudaFreeAsync(d_temp, cuda_stream));

        const int BLOCK = 256;
        //cub::UpperBound to build histogram based on random candidate splits, device side binary search
        int* d_hist_class0;
        int* d_hist_class1;
        int* d_hist_class2;
        
        CUDA_CHECK(cudaMallocAsync(&d_hist_class0, num_proj * (num_bins) * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_hist_class1, num_proj * (num_bins) * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_hist_class2, num_proj * (num_bins) * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_hist_class0, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_hist_class1, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_hist_class2, 0, num_proj * (num_bins) * sizeof(int), cuda_stream));
        ///////////////////////BuildHistogramEqualWidthKernel/////////////////////////////////////////
        
        //auto startHist = std::chrono::steady_clock::now();
        int threads_per_block_hist = num_bins;
        int num_elements_per_thread = 1;
        int blocks_per_grid_hist = (num_rows/num_elements_per_thread + threads_per_block_hist - 1) / threads_per_block_hist;
        dim3 grid_hist(blocks_per_grid_hist, num_proj); //single dimension grid/projection
        int sharedMemSize = 3 * (num_bins) * sizeof(int); // For Hist 0, Hist 1, and Hist 2
        BuildHistogramRandomKernel<BLOCK><<<grid_hist, threads_per_block_hist, sharedMemSize, cuda_stream>>>
                                                                    (d_col_add_projected, d_selected_examples, d_global_labels_data,
                                                                        d_hist_class0, d_hist_class1, d_hist_class2,
                                                                    num_rows, num_proj, num_bins, d_sorted_candidate_splits);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        TIMER_STOP(RandomBinning);                                                            

        ///////////////////////Inclusive Scan per projection per class/////////////////////////////////////////
        TIMER_START(RandomInclusiveScan);
        int total_rows = (num_bins) * num_proj;

        int* d_prefix_2;
        int* d_prefix_1;
        int* d_prefix_0;
        CUDA_CHECK(cudaMallocAsync(&d_prefix_2, (total_rows * sizeof(int)), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_prefix_1, (total_rows * sizeof(int)), cuda_stream));
        CUDA_CHECK(cudaMallocAsync(&d_prefix_0, (total_rows * sizeof(int)), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_prefix_2, 0, total_rows * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_prefix_1, 0, total_rows * sizeof(int), cuda_stream));
        CUDA_CHECK(cudaMemsetAsync(d_prefix_0, 0, total_rows * sizeof(int), cuda_stream));

        auto counting_begin = thrust::make_counting_iterator<int>(0);
        auto keys_begin = thrust::make_transform_iterator(counting_begin, index_to_proj{num_bins});

        auto d_hist_class2_ptr   = thrust::device_pointer_cast(d_hist_class2);        // input Negative class
        auto d_prefix_2_ptr = thrust::device_pointer_cast(d_prefix_2);  // output

        auto d_hist_class1_ptr   = thrust::device_pointer_cast(d_hist_class1);        // input Positive class
        auto d_prefix_1_ptr = thrust::device_pointer_cast(d_prefix_1);  // output
        
        // auto d_hist_class0_ptr   = thrust::device_pointer_cast(d_hist_class0);        // input Not Used class
        // auto d_prefix_0_ptr = thrust::device_pointer_cast(d_prefix_0);  // output
    
            thrust::inclusive_scan_by_key( //label = 0 second
            thrust::cuda::par.on(cuda_stream),
            keys_begin,                      /* keys   begin   */
            keys_begin + total_rows,         /* keys   end     */
            d_hist_class2_ptr,                      /* values begin   */
            d_prefix_2_ptr,                    /* output         */
            thrust::equal_to<int>(),         /* identical keys */
            thrust::plus<int>()
        ); 

        thrust::inclusive_scan_by_key( //label = 1 first
            thrust::cuda::par.on(cuda_stream),
            keys_begin,                      /* keys   begin   */
            keys_begin + total_rows,         /* keys   end     */
            d_hist_class1_ptr,                      /* values begin   */
            d_prefix_1_ptr,                /* output         */
            thrust::equal_to<int>(),         /* identical keys */
            thrust::plus<int>()              /* inclusive sum  */
        );            
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        TIMER_STOP(RandomInclusiveScan);

        TIMER_PRINT(RandomBinning, "GPU Random Boundaries Calculations and Histogram Binning Kernel Time taken");
        TIMER_PRINT(RandomInclusiveScan, "GPU Random Histogram Prefix Sum Time taken");

        *d_prefix_0_out = d_prefix_0;
        *d_prefix_1_out = d_prefix_1;
        *d_prefix_2_out = d_prefix_2;
        *d_candidate_splits_out = d_sorted_candidate_splits;
            
        CUDA_CHECK(cudaFreeAsync(d_hist_class0, cuda_stream));
        CUDA_CHECK(cudaFreeAsync(d_hist_class1, cuda_stream));
        CUDA_CHECK(cudaFreeAsync(d_hist_class2, cuda_stream));
        CUDA_CHECK(cudaFreeAsync((void *)d_selected_examples, cuda_stream));
        CUDA_CHECK(cudaFreeAsync((void *)d_col_add_projected, cuda_stream));
    }

__device__ __forceinline__
float entropy(const int pos, const int neg) {
    int total = pos + neg;
    if (total <= 0) return 0.0f;

    float p_pos = float(pos) / float(total);
    float p_neg = float(neg) / float(total);

    const float eps = 1e-30f;   // protects against denormals and log(0)

    float e = 0.0f;
    if (p_pos > eps) e -= p_pos * logf(p_pos);
    if (p_neg > eps) e -= p_neg * logf(p_neg);
    return e;
}

__device__ float gini(const int pos, const int neg) {
    int total = pos + neg;
    if (total == 0) return 0.0f;
    float p = float(pos) / total;
    float n = float(neg) / total;
    return 1.0f - p * p - n * n;
}

__global__ void FindBestGiniSplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const int num_proj,
    const int num_bins,
    float* gini_out_per_bin_per_proj
   )  // per proj
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = hist_class0[ (num_bins - 1) ]; // last bin holds total count
    int total_class1 = hist_class1[ (num_bins - 1) ]; // last bin holds total count
    
    // Compute left class counts for this split point
    int left_class0 = hist_class0[ base_idx + bin_id ]; // hist already cumulative
    int left_class1 = hist_class1[ base_idx + bin_id ]; // hist already cumulative

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;

    if (left_total == 0 || right_total == 0) {
        gini_out_per_bin_per_proj[base_idx + bin_id] = -INFINITY;
        return;
    }

    float gini_left = gini(left_class1, left_class0);
    float gini_right = gini(right_class1, right_class0);

    __shared__ float gini_parent;
    __shared__ float total;
    if (threadIdx.x == 0) {
        total = total_class0 + total_class1;
        gini_parent = gini(total_class1, total_class0);
    }
    __syncthreads();

    float left_weight = float(left_total) / float(total);
    float right_weight = float(right_total) / float(total);
    float gini_gain = gini_parent - (left_weight * gini_left + right_weight * gini_right);

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    gini_out_per_bin_per_proj[base_idx + bin_id] = gini_gain;
}

__global__ void FindBestEntropySplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const int num_proj,
    const int num_bins,
    float* entropy_out_per_bin_per_proj
    ) 
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = hist_class0[ (num_bins - 1) ]; // last bin holds total count
    int total_class1 = hist_class1[ (num_bins - 1) ]; // last bin holds total count

    // Compute left class counts for this split point
    int left_class0 = hist_class0[base_idx + bin_id];
    int left_class1 = hist_class1[base_idx + bin_id];

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;

    if (left_total == 0 || right_total == 0) {
        entropy_out_per_bin_per_proj[base_idx + bin_id] = -INFINITY;
        return;
    }
    
    float entropy_left = entropy(left_class1, left_class0);
    float entropy_right = entropy(right_class1, right_class0);

    __shared__ float entropy_before;
    __shared__ float total;
    if (threadIdx.x == 0) {
        total = total_class0 + total_class1;
        entropy_before = entropy(total_class1, total_class0);
    }
    __syncthreads();

    float weighted_entropy = (left_total * entropy_left + right_total * entropy_right) / max(total, 1.0f);
    float entropy_gain = entropy_before - weighted_entropy;

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    entropy_out_per_bin_per_proj[base_idx + bin_id] = entropy_gain;
}

void HistogramSplit (const int* d_prefix_0,
                      const int* d_prefix_1,
                      const int* d_prefix_2,
                      const float* d_candidate_splits,
                      const float* h_min_vals,
                      const float* d_bin_widths,
                      const int num_proj,
                      const int num_bins, //total_bins
                      const int num_rows,
                      int* best_proj,
                      int* best_bin_out,
                      float* best_gain_out,
                      float* best_threshold_out,
                      int* num_pos_examples_out,
                      double* elapsed_ms,
                      const bool verbose,
                      const int comp_method, //0: entropy, 1: gini
                      const int gpu_mode,
                      cudaStream_t cuda_stream
                    )
{
    TIMER_START(HistogramSplitEvaluation);
    float* d_out_per_bin_per_proj;
    CUDA_CHECK(cudaMallocAsync(&d_out_per_bin_per_proj, num_proj * (num_bins) * sizeof(float), cuda_stream)); // Store Entropy gain for each bin (except last)
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int threads_per_block_split = 256;
    int blocks_per_grid_split = (num_bins + threads_per_block_split - 1) / threads_per_block_split; // +1 to ensure we have enough blocks to cover all bins
    dim3 grid_split(blocks_per_grid_split, num_proj);
    dim3 block_split(threads_per_block_split);

    if (comp_method == 0) {
        FindBestEntropySplitKernel<<<grid_split, block_split, 0, cuda_stream>>>(
            d_prefix_0, d_prefix_1, num_proj, num_bins,
            d_out_per_bin_per_proj);
    }
    else {
        FindBestGiniSplitKernel<<<grid_split, block_split, 0, cuda_stream>>>(
            d_prefix_0, d_prefix_1, num_proj, num_bins,
            d_out_per_bin_per_proj);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    TIMER_STOP(HistogramSplitEvaluation);
    

    TIMER_START(HistogramBestGainReduction);
    // Find best gain across all projections and bins
    cub::KeyValuePair<int, float>* d_out1;
     // Allocate output
    CUDA_CHECK(cudaMallocAsync(&d_out1, sizeof(cub::KeyValuePair<int, float>), cuda_stream));
    // Step 1: Get temp storage size
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins, cuda_stream);
    // Step 2: Allocate temp storage
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));
    // Step 3: Run ArgMax
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins, cuda_stream);
    cub::KeyValuePair<int, float> h_out1;
    CUDA_CHECK(cudaMemcpyAsync(&h_out1, d_out1, sizeof(h_out1), cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_out1, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_temp_storage, cuda_stream));

    if (h_out1.value > 0.f) {
        // Calculate best projection
        *best_proj = h_out1.key / num_bins;
        // Calculate best gain
        *best_gain_out = h_out1.value;
        // Calculate best bin
        *best_bin_out = h_out1.key - (*best_proj * num_bins);

        //Calculate number of positive examples in the node after split
        int total_count_0, total_count_1, left_count_0, left_count_1;
        CUDA_CHECK(cudaMemcpyAsync(&total_count_0, d_prefix_0 + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaMemcpyAsync(&total_count_1, d_prefix_1 + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaMemcpyAsync(&left_count_0, d_prefix_0 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaMemcpyAsync(&left_count_1, d_prefix_1 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
        *num_pos_examples_out = total_count_0 + total_count_1 - left_count_0 - left_count_1;
       
        if (gpu_mode == 1) { //Random
            float best_threshold;
            int index = (*best_proj) * (num_bins - 1) + (*best_bin_out);
            CUDA_CHECK(cudaMemcpyAsync(&best_threshold, d_candidate_splits + index, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
            *best_threshold_out = best_threshold;
            CUDA_CHECK(cudaFreeAsync((void *)d_candidate_splits, cuda_stream));
        }

        if (gpu_mode == 2) { //Equal Width
            float bin_width;
            CUDA_CHECK(cudaMemcpyAsync(&bin_width, d_bin_widths + *best_proj, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
            *best_threshold_out = (*best_bin_out + 0.5) * bin_width + h_min_vals[*best_proj];  
            CUDA_CHECK(cudaFreeAsync((void *)d_bin_widths, cuda_stream));
            auto endThreshold = std::chrono::steady_clock::now();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    TIMER_STOP(HistogramBestGainReduction);

    TIMER_PRINT(HistogramSplitEvaluation, "GPU Histogram Gain Kernel Time taken");
    TIMER_PRINT(HistogramBestGainReduction, "GPU Histogram Best Gain Reduction Time taken");


    
    CUDA_CHECK(cudaFreeAsync((void *)d_out_per_bin_per_proj, cuda_stream));
}

void ThrustSortIndicesOnly(float* d_proj_values, unsigned int* d_row_ids, unsigned int* d_selected_examples, 
                            int num_rows, int num_proj, cudaStream_t cuda_stream) 
    {
    //d_row_ids: device pointer, length = num_rows* num_proj, uninitialised, will end up containing the example indices per projection, permuted exactly like the sorted projection values.
    //Copy the list of selected example indices into a thrust::device_vector
    //d_selected_examples   – device pointer, length = num_rows, contains the row-ids (example indices) that belong to the node.

    //Copy d_selected_examples into a Thrust container. This is to get an easy device pointer that can be used inside a device lambda.
    //This is to set up the iterator that will fill d_row_ids with the replicated row indices. 
    thrust::device_vector<unsigned int> d_selected_base(d_selected_examples, d_selected_examples + num_rows); //device_vector same as std::vector but on device
    //Obtain a raw device pointer to that internal storage
    //This is needed because inside the device lambda we cannot capture a device_vector directly.
    const unsigned int* d_base_ptr = thrust::raw_pointer_cast(d_selected_base.data()); //d_selected_base.data() gives a device pointer to the start of the vector
    //device_pointer_cast is the opposite of raw_pointer_cast; Switch back and forth between Thrust iterators and raw device pointers without copying any data.
    auto d_row_ids_iter = thrust::device_pointer_cast(d_row_ids);

    thrust::transform(
        thrust::cuda::par.on(cuda_stream),
        thrust::make_counting_iterator<int>(0), // first  input iterator
        thrust::make_counting_iterator<int>(num_rows * num_proj), // last  iterator
        d_row_ids_iter, // output iterator
        [=] __device__ (int i) { // device lambda
            return d_base_ptr[i % num_rows];  // replicate the selected example indices for each projection
            });

    size_t temp_bytes = 0;
    thrust::device_vector<int> d_offsets(num_proj + 1);
    thrust::sequence(
                    thrust::cuda::par.on(cuda_stream),
                    d_offsets.begin(),
                    d_offsets.end(),
                    0,
                    num_rows);   // [0, num_rows, 2*num_rows, ...]
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr,                      // query mode
        temp_bytes,                   // output: required temp storage
        d_proj_values, // keys in
        d_proj_values, // keys out (in-place)
        d_row_ids, // values in
        d_row_ids, // values out (in-place)
        num_proj * num_rows,           // total number of keys
        num_proj,                      // number of segments
        d_offsets.data().get(),        // segment begin offsets
        d_offsets.data().get() + 1,     // segment end offsets
        0,                              // begin bit
        32,                             // end bit
        cuda_stream
    );
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, cuda_stream));

    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp,           // temp storage
        temp_bytes,
        d_proj_values, // keys in
        d_proj_values, // keys out (in-place)
        d_row_ids, // values in
        d_row_ids, // values out (in-place)
        num_proj * num_rows,           // total number of keys
        num_proj,                      // number of segments
        d_offsets.data().get(),        // segment begin offsets
        d_offsets.data().get() + 1,     // segment end offsets
        0,                              // begin bit
        32,                             // end bit
        cuda_stream
    );
    cudaFreeAsync(d_temp, cuda_stream);
    cudaFreeAsync((void *)d_selected_examples, cuda_stream);
}

__device__ __forceinline__
float atomicMaxFloat(float* addr, float v)
{
    // Optional policy: ignore NaNs (so they don't "poison" the max).
    // If you'd rather propagate NaNs, remove this early return.
    if (isnan(v)) return *addr;

    int*  addr_as_i = reinterpret_cast<int*>(addr);

    int old = *addr_as_i;     // non-atomic initial read is OK; CAS loop will correct
    while (true) {
        float old_f = __int_as_float(old);

        // If current value is already >= v, we're done.
        if (old_f >= v) return old_f;

        int assumed = old;
        int desired = __float_as_int(v);

        // Try to swap in v if memory still equals assumed.
        old = atomicCAS(addr_as_i, assumed, desired);

        // If swap succeeded, atomicCAS returns assumed.
        if (old == assumed) return __int_as_float(assumed);
        // else: someone else updated it; loop and re-check against the new 'old'
    }
}

template <typename T>
struct ArgMaxPair {
    __device__ __forceinline__
    cub::KeyValuePair<T,int> operator()(
        const cub::KeyValuePair<T,int>& a,
        const cub::KeyValuePair<T,int>& b) const 
    {
        if (a.key > b.key) return a;
        if (b.key > a.key) return b;
        // tie-breaker: larger index wins
        return (a.value > b.value) ? a : b;
    }
};

template<int STRIDE, int blockSize>
__global__ void GiniGainKernel(
    const int* __restrict__ d_prefix_pos,      // [num_proj * proj_size / STRIDE]
    const int* __restrict__ d_prefix_neg,      // [num_proj * proj_size / STRIDE]
    const float* __restrict__ d_values,
    int* d_block_best_split,                   // [num_proj * num_blocks]
    float* d_block_best_gain,                  // [num_proj * num_blocks]
    const int num_rows, const int logical_rows)
{

    using Pair       = cub::KeyValuePair<float,int>;
    using BlockReduce = cub::BlockReduce<Pair, blockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;


    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int proj_id = blockIdx.y;
    int base = proj_id * num_rows;

    int total_pos = d_prefix_pos[base + num_rows - 1];
    int total_neg = d_prefix_neg[base + num_rows - 1];
    
    int logical_start = block_id * blockSize;
    int logical_end   = min(logical_start + blockSize, logical_rows); // last block check with min

    const int logical_row = logical_start + tid;  // global logical row this thread would handle
    float gain = -INFINITY;
    int split = -1;

    // Handle degenerate case (uniform across block → safe to return)
    if (num_rows < 2 || logical_rows <= 0) {
        if (tid == 0) {
            int out_idx = proj_id * gridDim.x + block_id;
            d_block_best_gain[out_idx]  = -INFINITY;
            d_block_best_split[out_idx] = -1;
        }
        return;  // all threads take this, so no block-wide ops after
    }

    if (logical_row < logical_end && logical_row < logical_rows) {
        // compute candidate split index using STRIDE mapping
        int idx = (logical_row + 1) * STRIDE - 1;
        //idx = min(idx, num_rows - 2); // ensure idx+1 valid
        // if (idx > num_rows - 2) idx = num_rows - 2;
        // if (idx < 0) idx = 0;

        // only consider split if value actually changes between idx and idx+1
        if (idx >= 0 && idx < num_rows - 1) {
            if (d_values[base + idx] != d_values[base + idx + 1]) {
                int left_pos = d_prefix_pos[base + idx];
                int left_neg = d_prefix_neg[base + idx];
                int right_pos = total_pos - left_pos;
                int right_neg = total_neg - left_neg;
                float left_g = gini(left_pos, left_neg);
                float right_g = gini(right_pos, right_neg);
                float parent_g = gini(total_pos, total_neg);

                int total = left_pos + left_neg + right_pos + right_neg;
                float left_weight = float(left_pos + left_neg) / float(total);
                float right_weight = float(right_pos + right_neg) / float(total);

                gain = parent_g - (left_weight * left_g + right_weight * right_g);
                split = idx + 1;
            }
        }
    }
    Pair thread_pair(gain, split);

    // Custom ArgMax: if gains tie, pick one deterministically (e.g., larger split)
    // Block-wide reduction
    Pair block_best = BlockReduce(temp_storage).Reduce(thread_pair, ArgMaxPair<float>());

    if (tid == 0) {
        int out_idx = proj_id * gridDim.x + block_id;
        d_block_best_gain [out_idx] = block_best.key;
        d_block_best_split[out_idx] = block_best.value;
    }
}

template<int STRIDE, int blockSize>
__global__ void EntropyGainKernel(
    const int* __restrict__ d_prefix_pos,      // [num_proj * proj_size / STRIDE]
    const int* __restrict__ d_prefix_neg,      // [num_proj * proj_size / STRIDE]
    const float* __restrict__ d_values,
    int* d_block_best_split,                   // [num_proj * num_blocks]
    float* d_block_best_gain,                  // [num_proj * num_blocks]
    const int num_rows, const int logical_rows)
{

    using Pair       = cub::KeyValuePair<float,int>;
    using BlockReduce = cub::BlockReduce<Pair, blockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int proj_id = blockIdx.y;

    int base = proj_id * num_rows;

    int total_pos = d_prefix_pos[base + num_rows - 1]; //Last arrary value per projection == total pos
    int total_neg = d_prefix_neg[base + num_rows - 1];

    const int logical_start = block_id * blockSize;
    const int logical_end   = min(logical_start + blockSize, logical_rows); // exclusive

    const int logical_row = logical_start + tid;  // global logical row this thread would handle
    float gain = -INFINITY;
    int split = -1;

    // Handle degenerate case (uniform across block → safe to return)
    if (num_rows < 2 || logical_rows <= 0) {
        if (tid == 0) {
            int out_idx = proj_id * gridDim.x + block_id;
            d_block_best_gain[out_idx]  = -INFINITY;
            d_block_best_split[out_idx] = -1;
        }
        return;  // all threads take this, so no block-wide ops after
    }

    if (logical_row < logical_end && logical_row < logical_rows) {
        // compute candidate split index using STRIDE mapping
        int idx = (logical_row + 1) * STRIDE - 1;
        //idx = min(idx, num_rows - 2); // ensure idx+1 valid
        // if (idx > num_rows - 2) idx = num_rows - 2;
        // if (idx < 0) idx = 0;

        // only consider split if value actually changes between idx and idx+1
        if (idx >= 0 && idx < num_rows - 1) {
            if (d_values[base + idx] < d_values[base + idx + 1]) {
                int left_pos  = d_prefix_pos[base + idx];
                int left_neg  = d_prefix_neg[base + idx];
                int right_pos = total_pos - left_pos;
                int right_neg = total_neg - left_neg;

                int left_count  = left_pos + left_neg;
                int right_count = right_pos + right_neg;        
                int total_count = left_count + right_count;

                float parent_e = entropy(total_pos, total_neg);
                float left_e = entropy(left_pos, left_neg);
                float right_e = entropy(right_pos, right_neg);

                // weight by counts; use num_rows (per-proj) to normalize — keep as float
                gain = parent_e - (left_count * left_e + right_count * right_e) / float(total_count);
                split = idx + 1;
            }
        }
    }
    Pair thread_pair(gain, split);

    // Custom ArgMax: if gains tie, pick one deterministically (e.g., larger split)
    // Block-wide reduction
    Pair block_best = BlockReduce(temp_storage).Reduce(thread_pair, ArgMaxPair<float>());

    if (tid == 0) {
        int out_idx = proj_id * gridDim.x + block_id;
        d_block_best_gain [out_idx] = block_best.key;
        d_block_best_split[out_idx] = block_best.value;
    }
}

__global__ void buildPosFlag(const unsigned int* sorted_idx,
                             const unsigned int* labels,
                             int*                flag,
                             int                 N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        flag[tid] = (labels[ sorted_idx[tid] ] == 1) ? 1 : 0;
}

/* 0 ↔ 1  (turn positives into negatives or vice-versa) */
__global__ void invertFlag(int* flag, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) flag[tid] = 1 - flag[tid];
}


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
    const bool verbose,
    const int comp_method,
    cudaStream_t cuda_stream
    )
{
    CUDA_CHECK(cudaGetLastError()); 
    constexpr int STRIDE = 1; // You can adjust STRIDE for # of elements for split computation
    const int blockSize = 256;

    TIMER_START(ExactPrefixSum);
    const int logical_rows = (num_rows + STRIDE - 1) / STRIDE;
    int total_rows = num_proj * num_rows;
    int* d_prefix_pos;
    int* d_prefix_neg;
    CUDA_CHECK(cudaMallocAsync(&d_prefix_pos, (total_rows * sizeof(int) ), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_prefix_neg, (total_rows * sizeof(int) ), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_prefix_pos, 0, total_rows * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_prefix_neg, 0, total_rows * sizeof(int), cuda_stream));
 
    int* d_flag;
    CUDA_CHECK(cudaMallocAsync(&d_flag, total_rows * sizeof(int), cuda_stream));
    dim3 gridCub((total_rows + blockSize - 1) / blockSize);//1D grid

    /* Make Thrust iterators */
    auto counting_begin = thrust::make_counting_iterator<int>(0);
    auto keys_begin =
            thrust::make_transform_iterator(counting_begin,
                                            index_to_proj{num_rows});
    auto d_flag_ptr   = thrust::device_pointer_cast(d_flag);        // input
    auto d_prefix_pos_ptr = thrust::device_pointer_cast(d_prefix_pos);  // output
    auto d_prefix_neg_ptr = thrust::device_pointer_cast(d_prefix_neg);  // output
    
    buildPosFlag<<<gridCub, blockSize, 0, cuda_stream>>>(d_sorted_indices, d_labels, d_flag, total_rows);

    thrust::inclusive_scan_by_key( //label = 1 first
        thrust::cuda::par.on(cuda_stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_flag_ptr,                      /* values begin   */
        d_prefix_pos_ptr,                /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()              /* inclusive sum  */
    );            
   
    invertFlag<<<gridCub, blockSize, 0, cuda_stream>>>(d_flag, total_rows);

    thrust::inclusive_scan_by_key( //label = 0 second
        thrust::cuda::par.on(cuda_stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_flag_ptr,                      /* values begin   */
        d_prefix_neg_ptr,                    /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()
    );
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_flag, cuda_stream));
    TIMER_STOP(ExactPrefixSum);
    
    TIMER_START(ExactGainComputation);
    int dimX = (logical_rows + blockSize - 1) / blockSize;
    size_t shared_mem = sizeof(float) * blockSize + sizeof(int) * blockSize;
    int total_blocks = num_proj * dimX;
    int* d_block_best_split;
    float* d_block_best_gain;
    CUDA_CHECK(cudaMallocAsync(&d_block_best_split, total_blocks * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMallocAsync(&d_block_best_gain, total_blocks * sizeof(float), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_block_best_split, 0xFF, total_blocks * sizeof(int), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(d_block_best_gain, 0xFF, total_blocks * sizeof(float), cuda_stream));
    
    dim3 gridDim(dimX, num_proj); //int dimX = (logical_rows + blockSize - 1) / blockSize;
    if (comp_method == 0) {
        EntropyGainKernel<STRIDE, blockSize><<<gridDim, blockSize, shared_mem, cuda_stream>>>(
        d_prefix_pos, d_prefix_neg, d_col_add_projected,
        d_block_best_split, d_block_best_gain, num_rows, logical_rows);
    } else {    
        GiniGainKernel<STRIDE, blockSize><<<gridDim, blockSize, shared_mem, cuda_stream>>>(
        d_prefix_pos, d_prefix_neg, d_col_add_projected,
        d_block_best_split, d_block_best_gain, num_rows, logical_rows);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    TIMER_STOP(ExactGainComputation);

    TIMER_START(ExactBestGainReduction);
    auto startReduce = std::chrono::steady_clock::now();
    void* d_temp_storage1 = nullptr;
    size_t temp_storage_bytes1 = 0;

    cub::KeyValuePair<int, float>* d_out1;
     // Allocate output
    CUDA_CHECK(cudaMallocAsync(&d_out1, sizeof(cub::KeyValuePair<int, float>), cuda_stream));
        // Step 1: Get temp storage size
    cub::DeviceReduce::ArgMax(
        d_temp_storage1, temp_storage_bytes1,
        d_block_best_gain, d_out1, total_blocks, cuda_stream);
    // Step 2: Allocate temp storage
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage1, temp_storage_bytes1, cuda_stream));
    // Step 3: Run ArgMax
    cub::DeviceReduce::ArgMax(
        d_temp_storage1, temp_storage_bytes1,
        d_block_best_gain, d_out1, total_blocks, cuda_stream
    );

    cub::KeyValuePair<int, float> h_out1;
    CUDA_CHECK(cudaMemcpyAsync(&h_out1, d_out1, sizeof(h_out1), cudaMemcpyDeviceToHost, cuda_stream));

    if (h_out1.value > 0.f) {
        *best_gain_out = h_out1.value;
        CUDA_CHECK(cudaMemcpyAsync(best_split_out, d_block_best_split + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
        *best_proj = h_out1.key / dimX;
        float threshold_1, threshold_2;
        int proj_offset = (*best_proj) * num_rows;
        CUDA_CHECK(cudaMemcpyAsync(&threshold_1, d_col_add_projected + proj_offset + (*best_split_out) - 1, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
        CUDA_CHECK(cudaMemcpyAsync(&threshold_2, d_col_add_projected + proj_offset + (*best_split_out), sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
        *best_threshold_out = 0.5f * (threshold_1 + threshold_2);
    }
    TIMER_STOP(ExactBestGainReduction);


    TIMER_PRINT(ExactPrefixSum, "GPU Exact Prefix Sum Time taken");
    TIMER_PRINT(ExactGainComputation, "GPU Exact Gain Kernel Time taken");
    TIMER_PRINT(ExactBestGainReduction, "GPU Exact Best Gain Reduction Time taken");



    CUDA_CHECK(cudaFreeAsync(d_out1, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_temp_storage1, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_prefix_pos, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_prefix_neg, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_block_best_split, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_block_best_gain, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_col_add_projected, cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_sorted_indices, cuda_stream));
}
