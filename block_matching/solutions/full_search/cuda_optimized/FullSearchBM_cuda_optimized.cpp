#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;

// Device function to compute SAD between two blocks on GPU
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
                                 int width, int height, int blocksX, int blocksY, int threadsPerBlock) {
    // TODO -optimization: use shared memory to store current block d_curr.
    
                                    // Grid block index (one per search block in current frame)
    int bx = blockIdx.x; // Block index in x direction (search block column)
    int by = blockIdx.y; // Block index in y direction (search block row)
    
    if (bx >= blocksX || by >= blocksY) return; // Out of bounds check (should not happen if grid is sized correctly)
    
    // Thread index within block
    int tx = threadIdx.x; // Thread index in x direction (within block)
    int ty = threadIdx.y; // Thread index in y direction (within block)
    int tidx = ty * blockDim.x + tx;  // linear thread index (0 to threadsPerBlock-1)
    
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
    if (total_positions <= threadsPerBlock) { // each thread process at most one position
        if (tidx < total_positions) { // some threads may be in idle
            int ref_y = tidx / max_ref_x; // y coordinate of candidate block in reference frame based on linear thread index
            int ref_x = tidx % max_ref_x; // x coordinate of candidate block in reference frame based on linear thread index    
            best_sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
            best_dx = ref_x - x;  
            best_dy = ref_y - y;
        }
    } else {
        // Many positions: distribute work across threads
        for (int pos = tidx; pos < total_positions; pos += threadsPerBlock) {
            int ref_y = pos / max_ref_x; // y coordinate of candidate block in reference frame based on linear thread index
            int ref_x = pos % max_ref_x; // x coordinate of candidate block in reference frame based on linear thread index
            int sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width);
            int dist = (ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y);
            int best_dist = best_dx * best_dx + best_dy * best_dy;
            if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                best_sad = sad;
                best_dx = ref_x - x;
                best_dy = ref_y - y;
            }
        }
    }
    
    // Write thread result to shared memory (single buffer with manual offset calculation)
    extern __shared__ int shared_memory[]; // Single shared memory buffer
    
    // Divide shared memory into 3 arrays with manual offset calculation
    int* shared_block_thread_sad = shared_memory;
    int* shared_block_thread_dx = shared_memory + threadsPerBlock;
    int* shared_block_thread_dy = shared_memory + (threadsPerBlock * 2);
    
    shared_block_thread_sad[tidx] = best_sad;
    shared_block_thread_dx[tidx] = best_dx;
    shared_block_thread_dy[tidx] = best_dy;
    
    __syncthreads(); // Ensure all threads have written their results to shared memory

    // Tree reduction: each step halves the number of active threads
    // Thread tid reads from tid + stride and compares
    // FIXME - this reduction assumes threadsPerBlock is a power of 2, which is true for common block sizes (e.g., 16, 32) but should be handled more robustly for arbitrary block sizes in production code
    // FIXME - this redcution leave a lot of thread inactive in the later steps, which is not optimal, coaleshing? warp shuffle? 
    for(int stride = threadsPerBlock / 2; stride > 0; stride /= 2) {
        if(tidx < stride) {
            // Thread tidx compares with thread (tidx + stride)
            int dist = shared_block_thread_dx[tidx + stride] * shared_block_thread_dx[tidx + stride] + 
                      shared_block_thread_dy[tidx + stride] * shared_block_thread_dy[tidx + stride];
            int best_dist = shared_block_thread_dx[tidx] * shared_block_thread_dx[tidx] + 
                           shared_block_thread_dy[tidx] * shared_block_thread_dy[tidx];
            if(shared_block_thread_sad[tidx + stride] < shared_block_thread_sad[tidx] || 
               (shared_block_thread_sad[tidx + stride] == shared_block_thread_sad[tidx] && dist < best_dist)) {
                shared_block_thread_sad[tidx] = shared_block_thread_sad[tidx + stride];
                shared_block_thread_dx[tidx] = shared_block_thread_dx[tidx + stride];
                shared_block_thread_dy[tidx] = shared_block_thread_dy[tidx + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 has the final best result after all reduction steps
    if(tidx == 0) {
        d_mv[by * blocksX + bx] = {shared_block_thread_dx[0], shared_block_thread_dy[0]};
    }
}


vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
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
    // TODO - optimize memory usage by using pitched memory or 2D arrays for better coalescing and cache performance, can be upload in constant memory.
    // constant memory could be an optimal solution for reading current and reference frame, but is impossible to fit 2 immage in 64 KB
    
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
    
    dim3 gridDim(blocksX, blocksY); // One block for each search block in current frame
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim);  // Each block has threadsPerBlockDim × threadsPerBlockDim threads (e.g., 16×16 = 256 threads) to search reference frame positions in parallel
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y 
              << " threads per block (" << (blockDim.x * blockDim.y) << " total threads)..." << std::endl;
    
    // Create CUDA events for timing use GPU timers to measure kernel execution time
    cudaEvent_t start, stop;  
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    // Allocate shared memory: total_threads * 3 arrays * sizeof(int) bytes per block
    // Total threads = threadsPerBlockDim * threadsPerBlockDim (e.g., 32*32 = 1024)
    size_t sharedMemSize = threadsPerBlock * 3 * sizeof(int);
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, blocksX, blocksY, threadsPerBlock);
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

    std::cout << "CUDA: Complete!" << std::endl;
    return result;
}


