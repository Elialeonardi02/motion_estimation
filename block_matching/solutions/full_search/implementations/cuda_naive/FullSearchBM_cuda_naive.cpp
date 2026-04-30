#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_naive.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;

// Device function to compute SAD between two blocks on GPU
// is static to avoid conflict between different .cu files from different solution 
static __device__ int computeSAD_device(const unsigned char* curr, const unsigned char* ref,
                                 int x1, int y1, int x2, int y2, int blockSize, int width) {
    int sad = 0;
    
    for(int y = 0; y < blockSize; y++)
        for(int x = 0; x < blockSize; x++)
            sad += abs(curr[(y1 + y) * width + (x1 + x)] - ref[(y2 + y) * width + (x2 + x)]);
    return sad;
}

// CUDA kernel: Each grid block processes one search block of current frame
// All threads within grid block cooperate to search reference frame 
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize,
                                 int width, int height, int blocksX, int blocksY,
                                 int* d_thread_sad, int* d_thread_dx, int* d_thread_dy, int threadsPerBlock) {
    // Grid block index (one per search block in current frame)
    int bx = blockIdx.x; // Block index in x direction (search block column)
    int by = blockIdx.y; // Block index in y direction (search block row)
    
    if (bx >= blocksX || by >= blocksY) return; // Out of bounds check (should not happen if grid is sized correctly)
    
    // Thread index within block
    int tx = threadIdx.x; // Thread index in x direction (within block)
    int ty = threadIdx.y; // Thread index in y direction (within block)
    int tidx = ty * blockDim.x + tx;  // linear thread index (0 to threadsPerBlock-1)
    int total_threads = blockDim.x * blockDim.y;    // Total threads in this block (e.g., 256 for 16x16 blockDim)
    
    // Top-left corner, to define search block in current frame based on de grid block index
    int x = bx * blockSize;  // x coordinate of top-left corner of search block in current frame
    int y = by * blockSize;  // y coordinate of top-left corner of search block in current frame
    
    // Search space dimensions in reference frame
    int max_ref_x = width - blockSize;          // Maximum x coordinate for top-left corner of block in reference frame (to fit blockSize)  
    int max_ref_y = height - blockSize;         // Maximum y coordinate for top-left corner of block in reference frame (to fit blockSize)
    int total_positions = max_ref_x * max_ref_y; // Total candidate positions in reference frame for this current search block (all possible top-left corners of blockSize in reference frame)
    
    // Thread-local best match
    int best_sad = INT_MAX; // Initialize best SAD with worst case
    int best_dx = 0, best_dy = 0;
    
    // Each thread searches one or more positions (depending on search space size)
    if (total_positions <= total_threads) { // each thread process at most one position
        if (tidx < total_positions) { // some threads may be in idle
            int ref_y = tidx / max_ref_x; // y coordinate of candidate block in reference frame based on linear thread index
            int ref_x = tidx % max_ref_x; // x coordinate of candidate block in reference frame based on linear thread index    
            best_sad = computeSAD_device(d_curr, d_ref,x , y, ref_x, ref_y, blockSize, width);
            best_dx =   ref_x - x;
            best_dy =  ref_y - y;
        }
    } else {
        // Many positions: distribute work across threads
        for (int pos = tidx; pos < total_positions; pos += total_threads) {
            int ref_y = pos / max_ref_x; // y coordinate of candidate block in reference frame based on linear thread index
            int ref_x = pos % max_ref_x; // x coordinate of candidate block in reference frame based on linear thread index
            int sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
            int dist = (ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y);
            int best_dist = best_dx * best_dx + best_dy * best_dy;
            if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                best_sad = sad;
                best_dx =   ref_x - x;
                best_dy =  ref_y - y;
            }
        }
    }
    
    // Write thread result to global memory (each thread has its own location)
    int result_idx = (bx + by * blocksX ) * threadsPerBlock + tidx;
    d_thread_sad[result_idx] = best_sad;
    d_thread_dx[result_idx] = best_dx;
    d_thread_dy[result_idx] = best_dy;
    
    __syncthreads(); // Ensure all threads have written their results to global memory before thread (0,0) reads them
    
    // Only thread (0,0) finds the global best among all threads in this block, all other threads are idle at this point
    if (tidx == 0) {
        int global_best_sad = INT_MAX;
        int global_best_dx = 0, global_best_dy = 0;
        
        for (int i = 0; i < total_threads; i++) {
            int idx = (bx + by * blocksX ) * threadsPerBlock + i; // index for thread i in this block 
            int dist = d_thread_dx[idx] * d_thread_dx[idx] + d_thread_dy[idx] * d_thread_dy[idx];
            int best_dist = global_best_dx * global_best_dx + global_best_dy * global_best_dy;
            if (d_thread_sad[idx] < global_best_sad || (d_thread_sad[idx] == global_best_sad && dist < best_dist)) {
                global_best_sad = d_thread_sad[idx];
                global_best_dx = d_thread_dx[idx];
                global_best_dy = d_thread_dy[idx];
            }
        }
        
        d_mv[by * blocksX + bx] = {global_best_dx, global_best_dy};
    }
}


vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                     int blockSize) {
    cudaSetDevice(0);
    
    // Validate input: current and reference frames must have same dimensions
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }
    
    size_t bytesPerFrame = frameSize * sizeof(unsigned char); // Grayscale: 1 byte per pixel 256 levels of gray
    
    std::cout << "CUDA: Processing " << curr.width << "x" << curr.height << " frame with block size " << blockSize << std::endl;
    
    // Allocate GPU memory for frames
    unsigned char* d_curr = nullptr;    // GPU pointer matrix for current frame
    unsigned char* d_ref = nullptr;     // GPU pointer matrix for reference frame 
    gpuErrorCheck(cudaMalloc((void**)&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc((void**)&d_ref, bytesPerFrame));
    
    // Copy frames to GPU
    std::cout << "CUDA: Copying frames to GPU..." << std::endl;
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    
    // Calculate grid dimension
    int blocksX = curr.width / blockSize;   // Number of orizontal pixel divided by block size
    int blocksY = curr.height / blockSize;  // Number of vertical pixel divided by block size
    
    std::cout << "CUDA: Grid size: " << blocksX << "x" << blocksY << " = " << (blocksX*blocksY) << " blocks" << std::endl;  // debugging output to verify grid size

    // Allocate GPU memory for motion vectors
    size_t mvSize = blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;   // GPU pointer matrix for motion vectors (one per block in current frame) 
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    // Calculate thread count: blockSize × blockSize (limited to max 1024 CUDA threads)
    int threadsPerBlockDim = blockSize; // default block size is 32 
    if (blockSize * blockSize > 1024) {
        threadsPerBlockDim = 32;  // 32×32 = 1024 threads maximum
    }
    int threadsPerBlock = threadsPerBlockDim * threadsPerBlockDim;
    
    // Allocate GPU memory for thread results
    size_t threadResultsSize = blocksX * blocksY * threadsPerBlock * sizeof(int);
    int* d_thread_sad = nullptr; // GPU pointer array for thread SAD results 
    int* d_thread_dx = nullptr;  // GPU pointer array for thread dx results 
    int* d_thread_dy = nullptr;  // GPU pointer array for thread dy results
    // each thread writes its SAD, dx, and dy results to these arrays, which are later read by thread (0,0) of each block to find the global best match for that block
    gpuErrorCheck(cudaMalloc((void**)&d_thread_sad, threadResultsSize)); // Each thread writes its SAD result to this array 
    gpuErrorCheck(cudaMalloc((void**)&d_thread_dx, threadResultsSize)); // Each thread writes its dx result to this array
    gpuErrorCheck(cudaMalloc((void**)&d_thread_dy, threadResultsSize)); // Each thread writes its dy result to this array
    
    dim3 gridDim(blocksX, blocksY); // One block for each search block in current frame
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim);  // Each block has threadsPerBlockDim × threadsPerBlockDim threads (e.g., 16×16 = 256 threads) to search reference frame positions in parallel
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y 
              << " threads per block (" << (blockDim.x * blockDim.y) << " total threads)..." << std::endl;
    
    // Create CUDA events for timing use GPU timers to measure kernel execution time
    cudaEvent_t start, stop;  
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    fullSearchKernel<<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, blocksX, blocksY,
                                            d_thread_sad, d_thread_dx, d_thread_dy, threadsPerBlock);
    gpuErrorCheck(cudaGetLastError());
    
    std::cout << "CUDA: Kernel launched, synchronizing..." << std::endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    recordCudaEvent(stop);
    float milliseconds = elapsedCudaTime(start, stop);
    std::cout << "CUDA: Kernel execution time: " << milliseconds << " ms" << std::endl;
    destroyCudaEvent(start);
    destroyCudaEvent(stop);
    
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
    cudaFree(d_thread_sad);
    cudaFree(d_thread_dx);
    cudaFree(d_thread_dy);
    
    std::cout << "CUDA: Complete!" << std::endl;
    return result;
}


