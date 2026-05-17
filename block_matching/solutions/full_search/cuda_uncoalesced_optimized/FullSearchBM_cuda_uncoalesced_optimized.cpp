#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_uncoalesced_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;

// Device function to compute PARTIAL SAD for a subset of pixels
// Each thread along Z (if is not 1) dimension computes SAD for a portion of pixels, then results are reduced
static __device__ int computeSAD_device_partial(const unsigned char* curr, const unsigned char* ref,
                                                int x2, int y2, int blockSize, int refWidth,
                                                int startPixel, int pixelsPerThread) {
    int sad = 0;
    for (int pixelIdx = startPixel; pixelIdx < startPixel + pixelsPerThread; pixelIdx++) {
        if (pixelIdx >= blockSize * blockSize) break; // last thread may have few pixels
        
        int y = pixelIdx / blockSize;
        int x = pixelIdx % blockSize;
        // Use __ldg for read-only global memory access (better caching and coalescing)
        // TODO performance are improved, pending test?
        sad += abs( curr[pixelIdx] - __ldg(&ref[(y2 + y) * refWidth + (x2 + x)])); // reference frame is read only, __ldg intrinsic use read-only cache
    }
    return sad;
}


// CUDA kernel: each grid block processes one search block of current frame
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize,
                                 int width, int threadsPerBlock, int searchRange)  {
    
    // Thread indices
    const int tidx_2d = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int tidx_3d = tidx_2d + (threadIdx.z * blockDim.x * blockDim.y);
    
    // Current block top-left corner
    const int x = blockIdx.x * blockSize;
    const int y = blockIdx.y * blockSize;
    
     // Calculate search window in block coordinates
    int search_bx_start = 0; // leftmost block index in reference frame 
    int search_bx_end = 0;   // rightmost block index in reference frame
    int search_by_start = 0; // topmost block index in reference frame
    int search_by_end = 0;   // bottommost block index in reference frame 
    int search_w = 0;        // width of search window in blocks
    int search_h = 0;        // height of search window in blocks
    int total_positions = 0; // total candidate positions in search window
    /*                                                              -
       (search_bx_start, search_by_start)                           | 
                                                                search_h
                                (search_bx_end,search_by_end)       |
                                                                    |
        |---------------------------search_w-----------------|      -                           
    */
    // limit search window based on search range, cut to frame boundaries if necessary
    if (searchRange > 0) {
        search_bx_start = max(0, (int)blockIdx.x - searchRange);
        search_bx_end = min((int)gridDim.x - 1, (int)blockIdx.x + searchRange);
        search_by_start = max(0, (int)blockIdx.y - searchRange);
        search_by_end = min((int)gridDim.y - 1, (int)blockIdx.y + searchRange);
        search_w = search_bx_end - search_bx_start + 1;
        search_h = search_by_end - search_by_start + 1;
        total_positions = search_w * search_h;
    } else {
        search_bx_start = 0;
        search_bx_end = gridDim.x - 1;
        search_by_start = 0;
        search_by_end = gridDim.y - 1;
        search_w = gridDim.x;
        search_h = gridDim.y;
        total_positions = gridDim.x * gridDim.y;
    }
    
    // Thread-local best match
    int best_sad = INT_MAX; // Initialize best SAD with worst case
    int best_dx = 0, best_dy = 0;
    
    // Shared memory layout: current block + SAD partial + reduction buffers
    extern __shared__ unsigned char shared_memory[];
    unsigned char* s_curr = shared_memory;
    int threadsXY = blockDim.x * blockDim.y;
    
    int* shared_block_thread_sad_partial = (int*)(s_curr + blockSize * blockSize); // buffer for partial SAD results from each thread (size = threadsPerBlock)
    int* shared_block_thread_sad = (int*)(shared_block_thread_sad_partial + threadsPerBlock); // buffer for final SAD results from each thread after reduction (size = threadsXY)
    int* shared_block_thread_dx = shared_block_thread_sad + threadsXY; // buffer for dx results corresponding to SAD results (size = threadsXY)
    int* shared_block_thread_dy = shared_block_thread_dx + threadsXY; // buffer for dy results corresponding to SAD results (size = threadsXY)
    int* shared_block_thread_dist = shared_block_thread_dy + threadsXY; // buffer for distance results corresponding to SAD results (size = threadsXY)
    
    // Load current block in shared memory
    for (int i = tidx_3d; i < blockSize * blockSize; i += threadsPerBlock) {
        s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
    }
    
    // Initialize shared memory for reduction buffers (z=0 threads only)
    if (threadIdx.z == 0) {
        shared_block_thread_sad[tidx_2d] = INT_MAX;
        shared_block_thread_dx[tidx_2d] = 0;
        shared_block_thread_dy[tidx_2d] = 0;
        shared_block_thread_dist[tidx_2d] = 0;
    }

    __syncthreads();
    
    // Precalculate pixels-per-thread for SAD parallelization
    const int pixelsPerThread = (blockSize * blockSize + blockDim.z - 1) / blockDim.z;
    
    // Each thread (X,Y,Z) searches positions and computes partial SAD along Z
    if (total_positions <= threadsPerBlock) {
        // Few positions: each thread (X,Y) processes at most one position, parallelize SAD along Z
        int ref_bx = 0, ref_by = 0;
        if (tidx_2d < total_positions) {
            // Convert position to 2D coordinates within search window
            ref_bx = search_bx_start + (tidx_2d % search_w);
            ref_by = search_by_start + (tidx_2d / search_w);

            shared_block_thread_sad_partial[tidx_3d] = computeSAD_device_partial(
                s_curr, d_ref, ref_bx * blockSize, ref_by * blockSize, blockSize, width,
                threadIdx.z * pixelsPerThread, pixelsPerThread);
        } else {
            shared_block_thread_sad_partial[tidx_3d] = 0; // Threads with no position to process contribute 0 to SAD sum
        }
        
        __syncthreads();
        
        // Reduce along Z: sum partial SADs
        for (int stride = 1; stride < blockDim.z; stride *= 2) {
            if (threadIdx.z + stride < blockDim.z) {
                // Add SAD from neighbor thread in Z dimension
                shared_block_thread_sad_partial[tidx_3d] += shared_block_thread_sad_partial[tidx_3d + stride * threadsXY]; 
            }
            __syncthreads();
        }
        
        // Thread z=0 has complete SAD for this position
        if (threadIdx.z == 0 && tidx_2d < total_positions) {
            best_sad = shared_block_thread_sad_partial[tidx_3d];
            best_dx = ref_bx - (int)blockIdx.x;
            best_dy = ref_by - (int)blockIdx.y;
        }
    } else {
        // Many positions: each thread (X,Y) processes multiple positions, parallelize SAD along Z for each position
        for (int pos = tidx_2d; pos < total_positions; pos += threadsXY) {
            // Convert position to 2D coordinates within search window
            int ref_bx = search_bx_start + (pos % search_w);
            int ref_by = search_by_start + (pos / search_w);

            shared_block_thread_sad_partial[tidx_3d] = computeSAD_device_partial(
                s_curr, d_ref, ref_bx * blockSize, ref_by * blockSize, blockSize, width,
                threadIdx.z * pixelsPerThread, pixelsPerThread);

            __syncthreads();
            
            // Reduce along Z
            for (int stride = 1; stride < blockDim.z; stride *= 2) {
                if (threadIdx.z + stride < blockDim.z) {
                    // Add SAD from neighbor thread in Z dimension
                    shared_block_thread_sad_partial[tidx_3d] += shared_block_thread_sad_partial[tidx_3d + stride * threadsXY];
                }
                __syncthreads();
            }
            
            // Thread z=0 has complete SAD for this position
            if (threadIdx.z == 0) {
                int sad = shared_block_thread_sad_partial[tidx_3d];
                int ref_dx = ref_bx - (int)blockIdx.x;
                int ref_dy = ref_by - (int)blockIdx.y;
                int dist = ref_dx * ref_dx + ref_dy * ref_dy;  // Distance in blocks²
                int best_dist = best_dx * best_dx + best_dy * best_dy;
                if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                    best_sad = sad;
                    best_dx = ref_dx;
                    best_dy = ref_dy;
                }
            }
        }
    }
    
    // Write thread result to shared memory (z=0 threads only)
    if (threadIdx.z == 0) {
        shared_block_thread_sad[tidx_2d] = best_sad;
        shared_block_thread_dx[tidx_2d] = best_dx;
        shared_block_thread_dy[tidx_2d] = best_dy;
        shared_block_thread_dist[tidx_2d] = best_dx * best_dx + best_dy * best_dy;
    }
    
    __syncthreads();

    // Find global best among all z=0 threads in this block (thread 0 only)
    // Only consider threads that have valid matches (not INT_MAX)
    if (threadIdx.z == 0 && tidx_2d == 0) {
        int global_best_sad = INT_MAX;
        int global_best_dx = 0, global_best_dy = 0;
        int global_best_dist = INT_MAX;
        
        // First pass: find any valid match to initialize global_best
        for (int i = 0; i < threadsXY; i++) {
            if (shared_block_thread_sad[i] < INT_MAX) {
                global_best_sad = shared_block_thread_sad[i];
                global_best_dx = shared_block_thread_dx[i];
                global_best_dy = shared_block_thread_dy[i];
                global_best_dist = shared_block_thread_dist[i];
                break;  // Found first valid, proceed to second pass
            }
        }
        
        // Second pass: find best among all valid matches
        for (int i = 0; i < threadsXY; i++) {
            if (shared_block_thread_sad[i] < INT_MAX) {
                int dist = shared_block_thread_dx[i] * shared_block_thread_dx[i] + 
                           shared_block_thread_dy[i] * shared_block_thread_dy[i];
                if (shared_block_thread_sad[i] < global_best_sad || 
                    (shared_block_thread_sad[i] == global_best_sad && dist < global_best_dist)) {
                    global_best_sad = shared_block_thread_sad[i];
                    global_best_dx = shared_block_thread_dx[i];
                    global_best_dy = shared_block_thread_dy[i];
                    global_best_dist = dist;
                }
            }
        }
        
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {global_best_dx, global_best_dy};
    }
}


 vector<vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(const ImageGray& curr, const ImageGray& ref,
                                                                    int blockSize, int searchRange) {
    cudaSetDevice(0);
    
    // Validate input: current and reference frames must have same dimensions
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }
    
    size_t bytesPerFrame = frameSize * sizeof(unsigned char); // Grayscale: 1 byte per pixel 256 levels of gray
    
    string searchModeStr = (searchRange > 0) ? ("Range search (range=" + to_string(searchRange) + " blocks)") : "Full search";
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Processing frame " << curr.width << "x" << curr.height
              << " with block size " << blockSize << std::endl;
    
    // Allocate and copy frames to GPU
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc((void**)&d_ref, bytesPerFrame));
    
    // Copy frames to GPU
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Copying frames to GPU..." << std::endl;
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    
    // Calculate grid dimensions
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Grid size: " << blocksX << "x" << blocksY
              << " = " << (blocksX*blocksY) << " blocks" << std::endl;
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Search mode: " << searchModeStr
              << " (searchRange=" << searchRange << ")" << std::endl;

    // Allocate GPU memory for motion vectors
    size_t mvSize = blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;   // GPU pointer matrix for motion vectors (one per block in current frame) 
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    // Determine threads per block based on the grid dimension 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int threadsPerBlockX= blocksX;
    int threadsPerBlockY = blocksY;
    int threadsPerBlockZ;
    if (searchRange > 0) {
        threadsPerBlockX = min(blocksX, searchRange * 2 + 1);
        threadsPerBlockY = min(blocksY, searchRange * 2 + 1);
    }

    if (threadsPerBlockX * threadsPerBlockY <= maxThreadsPerBlock) {
        threadsPerBlockZ = min(64, maxThreadsPerBlock / (threadsPerBlockX * threadsPerBlockY)); // Use Z dimension for SAD parallelization, up to 64 threads (typical warp size) or as many as possible within max threads per block
    } else {
        // FIXME review dimensions for large blocks to bnetter optimization
        threadsPerBlockX = 16;
        threadsPerBlockY = 16;
        threadsPerBlockZ = 4;
    }
    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;

    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
    
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Launching kernel with " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
              << " threads per block (" << threadsPerBlock << " total threads)..." << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    // Allocate shared memory for current block + partial SAD + reduction buffers
    // sad_partial: threadsPerBlock (for Z reduction)
    // sad/dx/dy/dist: threadsXY = threadsPerBlock / blockDim.z (for 2D reduction, z=0 threads only)
    size_t sharedMemSize = blockSize * blockSize * sizeof(unsigned char) +           // Current block pixels
                           threadsPerBlock * sizeof(int) +                           // sad_partial
                           (4 * blockDim.x * blockDim.y) * sizeof(int);                       // sad/dx/dy/dist
    
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, threadsPerBlock, searchRange);
    gpuErrorCheck(cudaGetLastError());
    
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Kernel launched, synchronizing..." << std::endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    recordCudaEvent(stop);
    float milliseconds = elapsedCudaTime(start, stop);
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Timing:" << std::endl;
    std::cout << "  Kernel execution time: " << milliseconds << " ms" << std::endl;
    destroyCudaEvent(start);
    destroyCudaEvent(stop);
    
    // Copy results back to host
    MotionVector* h_mv = new MotionVector[blocksX * blocksY];
    gpuErrorCheck(cudaMemcpy(h_mv, d_mv, mvSize, cudaMemcpyDeviceToHost));
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result(blocksY, vector<MotionVector>(blocksX));
    for (int by = 0; by < blocksY; by++)
        for (int bx = 0; bx < blocksX; bx++)
            result[by][bx] = h_mv[by * blocksX + bx];
    
    // Cleanup
    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Cleaning up GPU memory..." << std::endl;
    delete[] h_mv;
    cudaFree(d_curr);
    cudaFree(d_ref);
    cudaFree(d_mv);

    std::cout << "CUDA Uncoalesced Optimized (Grayscale): Complete!" << std::endl;
    return result;
}


