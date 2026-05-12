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
                                 MotionVector* d_mv, int blockSize, int width, int* d_thread_sad, 
                                 int* d_thread_dx, int* d_thread_dy, int threadsPerBlock, int searchRange)  {
    // Grid and thread indices
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    
    // Current block top-left corner
    int x = blockIdx.x * blockSize;
    int y = blockIdx.y * blockSize;
    int total_positions; 
    if (searchRange >0) {
        total_positions = (min(searchRange * 2 + 1, gridDim.x) * min(searchRange * 2 + 1, gridDim.y)); // number of candidate blocks to search within the specified range
    } else{
        total_positions= gridDim.x * gridDim.y ;
    }

    
    // Thread-local best match
    int best_sad = INT_MAX;
    int best_dx = 0, best_dy = 0;
    int ref_y;
    int ref_x;
    // Each thread searches assigned positions
    if (total_positions <= total_threads) { 
        // Few positions: parallelize SAD along threads
        if (tidx < total_positions) {
            if (searchRange > 0) {
                // Check if candidate block is within search range
                ref_y = (blockIdx.y + threadIdx.y - searchRange) * blockSize;
                ref_x = (blockIdx.x + threadIdx.x - searchRange) * blockSize;
            }
            else{
                ref_y = (tidx / gridDim.x) * blockSize ; // y coordinate of reference block
                ref_x = (tidx % gridDim.x) * blockSize; // x coordinate of reference block
            }
            best_sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
            best_dx = (ref_x - x) / blockSize; // local best dx
            best_dy = (ref_y - y) / blockSize; // local best dy
        }
    } else {
        // Many positions: distribute work across threads
        for (int pos = tidx; pos < total_positions; pos += total_threads) {
             if (searchRange > 0) {
                // Check if candidate block is within search range
                ref_y = (blockIdx.y + threadIdx.y - searchRange) * blockSize;
                ref_x = (blockIdx.x + threadIdx.x - searchRange) * blockSize;
            }
            else{
                ref_y = (tidx / gridDim.x) * blockSize ; // y coordinate of reference block
                ref_x = (tidx % gridDim.x) * blockSize; // x coordinate of reference block
            }
            int sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
            int dist = (ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y);
            int best_dist = best_dx * best_dx + best_dy * best_dy; // local best distance for tie-breaking
            if (sad < best_sad || (sad == best_sad && dist < best_dist)) { // update local best match
                best_sad = sad;
                best_dx = (ref_x - x) / blockSize; // local best dx
                best_dy = (ref_y - y) / blockSize; // local best dy
            }
        }
    }
    
    // Write thread result to global memory
    int result_idx = (blockIdx.x + blockIdx.y * gridDim.x ) * threadsPerBlock + tidx;
    d_thread_sad[result_idx] = best_sad;
    d_thread_dx[result_idx] = best_dx;
    d_thread_dy[result_idx] = best_dy;
    
    __syncthreads();
    
    // Thread (0,0) finds global best among all threads in this block
    if (tidx == 0) {
        int global_best_sad = INT_MAX;
        int global_best_dx = 0, global_best_dy = 0;
        
        for (int i = 0; i < total_threads; i++) {
            int idx = (blockIdx.x + blockIdx.y * gridDim.x) * threadsPerBlock + i;
            int dist = d_thread_dx[idx] * d_thread_dx[idx] + d_thread_dy[idx] * d_thread_dy[idx]; // distance for tie-breaking
            int best_dist = global_best_dx * global_best_dx + global_best_dy * global_best_dy;  // current global best distance for tie-breaking
            if (d_thread_sad[idx] < global_best_sad || (d_thread_sad[idx] == global_best_sad && dist < best_dist)) { // update global best match
                global_best_sad = d_thread_sad[idx];
                global_best_dx = d_thread_dx[idx];
                global_best_dy = d_thread_dy[idx];
            }
        }
        
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {global_best_dx, global_best_dy}; // write final best motion vector for this block to global memory
    }
}


vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                     int blockSize, int searchRange) {
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
    
    std::cout << "CUDA: Grid size: " << blocksX << "x" << blocksY << " = " << (blocksX*blocksY) << " blocks" << std::endl;

    // Allocate GPU memory for motion vectors
    size_t mvSize = blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;   // GPU pointer matrix for motion vectors (one per block in current frame) 
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
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
    
    // Allocate GPU global memory for thread results
    size_t threadResultsSize = blocksX * blocksY * threadsPerBlock * sizeof(int);
    int* d_thread_sad = nullptr; // GPU pointer array for thread SAD results 
    int* d_thread_dx = nullptr;  // GPU pointer array for thread dx results 
    int* d_thread_dy = nullptr;  // GPU pointer array for thread dy results
    // each thread writes its SAD, dx, and dy results to these arrays, which are later read by thread (0,0) of each block to find the global best match for that block
    gpuErrorCheck(cudaMalloc((void**)&d_thread_sad, threadResultsSize)); // Each thread writes its SAD result to this array 
    gpuErrorCheck(cudaMalloc((void**)&d_thread_dx, threadResultsSize)); // Each thread writes its dx result to this array
    gpuErrorCheck(cudaMalloc((void**)&d_thread_dy, threadResultsSize)); // Each thread writes its dy result to this array
    
    dim3 gridDim(blocksX, blocksY); // One block for each search block in current frame
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY);  // Each block has threadsPerBlockX × threadsPerBlockY threads 
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y 
              << " threads per block (" << (blockDim.x * blockDim.y) << " total threads)..." << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    fullSearchKernel<<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, d_thread_sad, 
                                            d_thread_dx, d_thread_dy, threadsPerBlock, searchRange);
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


