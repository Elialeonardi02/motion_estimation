#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_naive.h"
#include "utils.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;

// Structure to hold thread results
struct ThreadResult {
    int sad;
    int dx;
    int dy;
};

// Device function to compute SAD between two blocks
static __device__ int computeSAD_device(const unsigned char* curr, const unsigned char* ref,
                                 int x1, int y1, int x2, int y2, int blockSize, int width) {
    int sad = 0;
    
    for(int y = 0; y < blockSize; y++)
        for(int x = 0; x < blockSize; x++)
            sad += abs(curr[(y1 + y) * width + (x1 + x)] - ref[(y2 + y) * width + (x2 + x)]);
    return sad;
}

// CUDA kernel: each grid block processes one search block of current frame
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize, int width, ThreadResult* d_thread_results, 
                                 int threadsPerBlock, int searchRange)  {
    // Grid and thread indices
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    
    // Current block top-left corner
    int x = blockIdx.x * blockSize;
    int y = blockIdx.y * blockSize;
    
    // Calculate search window using helper function
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    
    // Thread-local best match
    int best_sad = INT_MAX;
    int best_dx = 0, best_dy = 0;
    int ref_y;
    int ref_x;
     // total_threads < totalPosition, thread processes its assigned positions (i, i+total_threads, i+2*total_threads, ...)
    // total_threads = totalPosition, thread processes only one assigned position.
    for (int pos = tidx; pos < bounds.totalPositions; pos += total_threads) {
        // Convert position to 2D coordinates within search window
        int ref_bx = bounds.startX + (pos % bounds.width);
        int ref_by = bounds.startY + (pos / bounds.width);
        ref_x = ref_bx * blockSize;
        ref_y = ref_by * blockSize;
        
        int sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
        int dist = (ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y);
        int best_dist = best_dx * best_dx + best_dy * best_dy; // local best distance for tie-breaking
        if (sad < best_sad || (sad == best_sad && dist < best_dist)) { // update local best match
            best_sad = sad;
            best_dx = (ref_x - x) / blockSize; // local best dx
            best_dy = (ref_y - y) / blockSize; // local best dy
        }
    }
    
    // Write thread result to global memory
    int result_idx = (blockIdx.x + blockIdx.y * gridDim.x ) * threadsPerBlock + tidx;
    d_thread_results[result_idx] = {best_sad, best_dx, best_dy};
    
    __syncthreads();
    
    // Thread (0,0) finds global best among all threads in this block
    if (tidx == 0) {
        int global_best_sad = INT_MAX;
        int global_best_dx = 0, global_best_dy = 0;
        int global_best_dist = INT_MAX;
        for (int i = 0; i < total_threads; i++) {
            int idx = (blockIdx.x + blockIdx.y * gridDim.x) * threadsPerBlock + i;
            int dist = d_thread_results[idx].dx * d_thread_results[idx].dx + d_thread_results[idx].dy * d_thread_results[idx].dy; // distance for tie-breaking 
            if (d_thread_results[idx].sad < global_best_sad || (d_thread_results[idx].sad == global_best_sad && dist < global_best_dist)) { // update global best match
                global_best_sad = d_thread_results[idx].sad;
                global_best_dx = d_thread_results[idx].dx;
                global_best_dy = d_thread_results[idx].dy;
                global_best_dist = dist; // current global best distance for tie-breaking
            }
        }
        
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {global_best_dx, global_best_dy}; // write final best motion vector for this block to global memory
    }
}


vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                     int blockSize, int searchRange) {
    cudaSetDevice(0);
    
    ValidationUtils::validateFrameDimensions(curr, ref);
    
    // Calculate grid dimensions
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Naive", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CUDA Naive", searchRange);
    
    // Determine threads per block (max 1024 threads per block on typical GPUs)
    int threadsPerBlockX = blocksX;
    int threadsPerBlockY = blocksY;
    if (searchRange > 0) {
        // limit threads per block based on search range
        // if range is bigger than frame size, it will be clamped to frame size.
        threadsPerBlockX = min(threadsPerBlockX, searchRange * 2 + 1); 
        threadsPerBlockY = min(threadsPerBlockY, searchRange * 2 + 1); 
    }
    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
    if (threadsPerBlock > 1024) {
        threadsPerBlockX = 32;
        threadsPerBlockY = 32;
        threadsPerBlock = 1024;
    }
    
    // Allocate GPU memory for frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    
    LoggingUtils::printCopyingToGPU("CUDA Naive");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);
    
    // Allocate GPU memory for motion vectors
    size_t mvSize = (size_t)blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));

    // Allocate GPU global memory for thread results
    size_t threadResultsSize = (size_t)blocksX * blocksY * threadsPerBlock * sizeof(ThreadResult);
    ThreadResult* d_thread_results = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_thread_results, threadResultsSize));
    
    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
    
    cout << "CUDA Naive (Grayscale): Launching kernel with " << blockDim.x << "x" << blockDim.y 
        << " threads per block (" << (blockDim.x * blockDim.y) << " total threads)..." << endl;
    
    // Create CUDA timer
    CudaTimer timer("Kernel execution");
    timer.start();
    
    fullSearchKernel<<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, d_thread_results, 
                                            threadsPerBlock, searchRange);
    gpuErrorCheck(cudaGetLastError());
    
    cout << "CUDA Naive (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    float milliseconds = timer.stop();
    cout << "CUDA Naive (Grayscale): Timing:" << endl;
    CudaTimingUtils::printKernelTiming("Kernel execution", milliseconds);
    
    // Copy results back to host
    vector<MotionVector> h_mv(blocksX * blocksY);
    GPUMemoryUtils::copyMotionVectorsFromGPU(h_mv, d_mv, blocksX, blocksY);
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv, blocksX, blocksY);
    
    // Cleanup
    LoggingUtils::printCleanupGPU("CUDA Naive");
    GPUMemoryUtils::freeMemory(d_curr, d_ref, d_mv);
    cudaFree(d_thread_results);
    
    LoggingUtils::printProcessingComplete("CUDA Naive");
    return result;
}


