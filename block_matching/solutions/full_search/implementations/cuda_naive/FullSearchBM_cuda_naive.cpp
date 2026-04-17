#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_naive.h"

using namespace std;

// Device function to compute SAD between two blocks on GPU
__device__ int computeSAD_device(const unsigned char* curr, const unsigned char* ref,
                                 int x1, int y1, int x2, int y2, int blockSize, int width) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++)
        for(int x = 0; x < blockSize; x++)
            sad += abs(curr[(y1 + y) * width + (x1 + x)] - ref[(y2 + y) * width + (x2 + x)]);
    return sad;
}

// CUDA kernel: Each grid block processes one search block of current frame
// Threads within grid block cooperate to search reference frame
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize, int searchRange,
                                 int width, int height, int blocksX, int blocksY) {
    // Grid block index (one per search block in current frame)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    if (bx >= blocksX || by >= blocksY) return;
    
    // Thread index within block (each thread searches one position)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tidx = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;
    
    // Top-left corner of search block in current frame
    int search_x = bx * blockSize;
    int search_y = by * blockSize;
    
    // Search space dimensions
    int max_ref_x = width - blockSize;
    int max_ref_y = height - blockSize;
    int total_positions = max_ref_x * max_ref_y;
    
    // Thread-local best match
    int best_sad = INT_MAX;
    int best_dx = 0, best_dy = 0;
    
    // Each thread searches multiple positions (distribute work across threads)
    for (int pos = tidx; pos < total_positions; pos += total_threads) {
        // Convert linear position to 2D reference coordinates
        int ref_y = pos / max_ref_x;
        int ref_x = pos % max_ref_x;
        
        // Compute SAD for this candidate position
        int sad = computeSAD_device(d_curr, d_ref, search_x, search_y, ref_x, ref_y, blockSize, width);
        
        // Track best match found by this thread
        if (sad < best_sad) {
            best_sad = sad;
            best_dx = ref_x - search_x;
            best_dy = ref_y - search_y;
        }
    }
    
    // Shared memory for reduction (max 1024 threads = 1024 ints)
    __shared__ int shared_sad[1024];
    __shared__ int shared_dx[1024];
    __shared__ int shared_dy[1024];
    
    // Store thread results in shared memory
    shared_sad[tidx] = best_sad;
    shared_dx[tidx] = best_dx;
    shared_dy[tidx] = best_dy;
    __syncthreads();
    
    // Parallel reduction to find global best within block
    for (int s = total_threads / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            if (shared_sad[tidx + s] < shared_sad[tidx]) {
                shared_sad[tidx] = shared_sad[tidx + s];
                shared_dx[tidx] = shared_dx[tidx + s];
                shared_dy[tidx] = shared_dy[tidx + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes final result
    if (tidx == 0) {
        d_mv[by * blocksX + bx] = {shared_dx[0], shared_dy[0]};
    }
}

vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                     int blockSize, int searchRange) {
    cudaSetDevice(0);
    
    // Validate input
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }
    
    size_t bytesPerFrame = frameSize * sizeof(unsigned char);
    
    std::cout << "CUDA: Processing " << curr.width << "x" << curr.height << " frame with block size " << blockSize << std::endl;
    
    // Allocate GPU memory for frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc((void**)&d_ref, bytesPerFrame));
    
    // Copy frames to GPU
    std::cout << "CUDA: Copying frames to GPU..." << std::endl;
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    
    // Calculate grid dimensions
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    
    std::cout << "CUDA: Grid size: " << blocksX << "x" << blocksY << " = " << (blocksX*blocksY) << " blocks" << std::endl;
    
    // Allocate GPU memory for motion vectors
    size_t mvSize = blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    // Calculate thread count: blockSize × blockSize (limited to max 1024 CUDA threads)
    int threadsPerDim = blockSize;
    if (blockSize * blockSize > 1024) {
        threadsPerDim = 32;  // 32×32 = 1024 threads maximum
    }
    
    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerDim, threadsPerDim);
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y 
              << " threads per block (" << (blockDim.x * blockDim.y) << " total threads)..." << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    fullSearchKernel<<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize, searchRange,
                                            curr.width, curr.height, blocksX, blocksY);
    gpuErrorCheck(cudaGetLastError());
    
    std::cout << "CUDA: Kernel launched, synchronizing..." << std::endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA: Kernel execution time: " << milliseconds << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy results back to host
    MotionVector* h_mv = new MotionVector[blocksX * blocksY];
    gpuErrorCheck(cudaMemcpy(h_mv, d_mv, mvSize, cudaMemcpyDeviceToHost));
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result(blocksY, vector<MotionVector>(blocksX));
    for (int by = 0; by < blocksY; by++) {
        for (int bx = 0; bx < blocksX; bx++) {
            result[by][bx] = h_mv[by * blocksX + bx];
        }
    }
    
    // Cleanup
    std::cout << "CUDA: Cleaning up GPU memory..." << std::endl;
    delete[] h_mv;
    cudaFree(d_curr);
    cudaFree(d_ref);
    cudaFree(d_mv);
    
    std::cout << "CUDA: Complete!" << std::endl;
    return result;
}

// CUDA error checking function
void gpuErrorCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
        throw runtime_error("CUDA error occurred");
    }
}