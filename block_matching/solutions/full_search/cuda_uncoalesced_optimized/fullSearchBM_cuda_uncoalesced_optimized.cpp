#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "fullSearchBM_cuda_uncoalesced_optimized.h"
#include "utils.h"
#include "cuda_utils.h"
#include "sad_utils.h"
#include <chrono>

using namespace std;

// Structure to hold thread results (both in shared and global memory)
struct ThreadResult {
    int sad;
    int dx;
    int dy;
};

// Device function to compute PARTIAL SAD for a subset of pixels
// Each thread along dimension computes SAD for a portion of pixels, then results are reduced
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
        sad += abs(curr[pixelIdx] - ref[(y2 + y) * refWidth + (x2 + x)]); // reference frame is read only, __ldg intrinsic use read-only cache
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
    
    // Calculate search window using helper function
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    
    // Thread-local best match
    int best_sad = INT_MAX; // Initialize best SAD with worst case
    int best_dx = 0, best_dy = 0;
    
    // Shared memory layout: current block + SAD partial + reduction buffers
    extern __shared__ unsigned char smem[];
    unsigned char* s_curr = smem;
    int threadsXY = blockDim.x * blockDim.y;
    
    int* shared_block_thread_sad_partial = (int*)(s_curr + blockSize * blockSize); // buffer for partial SAD results from each thread (size = threadsPerBlock)
    ThreadResult* shared_block_thread_results = (ThreadResult*)(shared_block_thread_sad_partial + threadsPerBlock); // buffer for thread results (size = threadsXY)
    
    // Load current block in shared memory
    for (int i = tidx_3d; i < blockSize * blockSize; i += threadsPerBlock) {
        s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
    }
    
    // Initialize shared memory for reduction buffers (z=0 threads only)
    // idle threads will not write to shared memory, but they will participate in reduction
    if (threadIdx.z == 0) {
        shared_block_thread_results[tidx_2d] = {INT_MAX, 0, 0};
    }
    __syncthreads();
    
    // Precalculate pixels-per-thread for SAD parallelization
    const int pixelsPerThread = (blockSize * blockSize + blockDim.z - 1) / blockDim.z;
    
    // Each thread (X,Y,Z) searches positions and computes partial SAD along Z
    // threadsXY < totalPosition, threadsZ handles partial SAD for its assigned positions (i, i+threadsXY, i+2*threadsXY, ...), then reduction along Z gives total SAD for that position
    // threadsXY = totalPosition, threads along Z dimension compute partial SAD for only one assigned position.
    const int totalIterations = (bounds.totalPositions + threadsXY - 1) / threadsXY;
    for (int iter = 0; iter < totalIterations; iter++) {
        const int pos = iter * threadsXY + tidx_2d;
        const bool validPos = (pos < bounds.totalPositions);
        // Convert position to 2D coordinates within search window
        int ref_bx = validPos ? bounds.startX + (pos % bounds.width) : 0; // for invalid position, set to 0 
        int ref_by = validPos ? bounds.startY + (pos / bounds.width) : 0; // for invalid position, set to 0
        shared_block_thread_sad_partial[tidx_3d] = validPos ? computeSAD_device_partial(
            s_curr, d_ref, ref_bx * blockSize, ref_by * blockSize, blockSize, width,
            threadIdx.z * pixelsPerThread, pixelsPerThread) : 0;
        
        __syncthreads();
        
        // Reduce along Z, descending stride
        for (int stride = blockDim.z >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.z < stride) {
                // Add SAD from neighbor thread in Z dimension
                shared_block_thread_sad_partial[tidx_3d] += shared_block_thread_sad_partial[tidx_3d + stride * threadsXY];
            }
            __syncthreads();
        }
        
        // Thread z=0 has complete SAD for this position, compute best match with tie-breaking by distance
        if (threadIdx.z == 0 && validPos) {
            int sad = shared_block_thread_sad_partial[tidx_3d];
            int ref_dx = ref_bx - blockIdx.x;
            int ref_dy = ref_by - blockIdx.y;
            int dist = ref_dx * ref_dx + ref_dy * ref_dy;  // Distance in blocks for tie-breaking
            int best_dist = best_dx * best_dx + best_dy * best_dy;
            if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                best_sad = sad;
                best_dx = ref_dx;
                best_dy = ref_dy;
            }
        }
    }
    
    // Write thread result to shared memory (z=0 threads only)
    if (threadIdx.z == 0) {
        shared_block_thread_results[tidx_2d] = {best_sad, best_dx, best_dy};
    }
    
    __syncthreads();

    // Find global best among all z=0 threads in this block (thread 0,0,0 only)
    // FIXME this reduction can be optimized using stride-base reduction
    if (tidx_3d == 0) {
        int global_best_sad = INT_MAX;
        int global_best_dx = 0, global_best_dy = 0;
        int global_best_dist = INT_MAX;
        
        // Find best among all valid matches
        for (int i = 0; i < threadsXY; i++) {
            if (shared_block_thread_results[i].sad < INT_MAX) {
                int dist = shared_block_thread_results[i].dx * shared_block_thread_results[i].dx + 
                           shared_block_thread_results[i].dy * shared_block_thread_results[i].dy;
                if (shared_block_thread_results[i].sad < global_best_sad || 
                    (shared_block_thread_results[i].sad == global_best_sad && dist < global_best_dist)) {
                    global_best_sad = shared_block_thread_results[i].sad;
                    global_best_dx = shared_block_thread_results[i].dx;
                    global_best_dy = shared_block_thread_results[i].dy;
                    global_best_dist = dist;
                }
            }
        }
        
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {global_best_dx, global_best_dy};
    }
}


 vector<vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(const ImageGray& curr, const ImageGray& ref,
                                                                    int blockSize, int searchRange, SingleRunMetrics& metrics) {
    
    auto total_time_start = std::chrono::high_resolution_clock::now();
     // Create CUDA timer                                                 
    cudaEvent_t  evKStart, evKStop;
    
    createCudaEvent(evKStart);
    createCudaEvent(evKStop);

    cudaSetDevice(0);
    
    ValidationUtils::validateFrameDimensions(curr, ref);
    
    // Calculate grid dimensions
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Uncoalesced Optimized", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CUDA Uncoalesced Optimized", searchRange);
    
    // Allocate GPU memory for frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);

    LoggingUtils::printCopyingToGPU("CUDA Uncoalesced Optimized");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);
    
    
    // Determine threads per block based on device properties 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int threadsPerBlockX = blocksX;
    int threadsPerBlockY = blocksY;
    int threadsPerBlockZ;
    
    if (searchRange > 0) {
        threadsPerBlockX = min(blocksX, searchRange * 2 + 1);
        threadsPerBlockY = min(blocksY, searchRange * 2 + 1);
    }

    if (threadsPerBlockX * threadsPerBlockY <= maxThreadsPerBlock) {
        int z = min(64, maxThreadsPerBlock / (threadsPerBlockX * threadsPerBlockY));
        // blockDim.z must be a power of 2 to ensure stride reduction covers all Z levels without skipping partial SAD contributions
        threadsPerBlockZ = 1; 
        while (threadsPerBlockZ * 2 <= z) threadsPerBlockZ *= 2;
    } else {
        threadsPerBlockX = 16;
        threadsPerBlockY = 16;
        threadsPerBlockZ = 4;
    }
    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;

    // Allocate GPU memory for motion vectors
    size_t mvSize = (size_t)blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
    
    cout << "CUDA Uncoalesced Optimized (Grayscale): Launching kernel with " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
        << " threads per block (" << threadsPerBlock << " total threads)..." << endl;

    
    // Allocate shared memory
    // FIXME SMEM size can be reduced reusing partial SAD buffer for redution of total array.
    size_t sharedMemSize = blockSize * blockSize * sizeof(unsigned char) +
                           threadsPerBlock * sizeof(int) +
                           (blockDim.x * blockDim.y) * sizeof(ThreadResult);
    /*cout << blockSize*blockSize *sizeof(unsigned char) << " bytes for current block, " 
         << threadsPerBlock * sizeof(int) << " bytes for partial SAD buffer, "
         << (blockDim.x * blockDim.y) * sizeof(ThreadResult) << " bytes for thread results buffer" << endl;
    */
    cout << "CUDA Uncoalesced Optimized (Grayscale): Shared memory per block: " << sharedMemSize / 1024.0f << " KB" << endl;
    
    recordCudaEvent(evKStart);
    
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, threadsPerBlock, searchRange);
    gpuErrorCheck(cudaGetLastError());
    
    recordCudaEvent(evKStop);
    
    cout << "CUDA Uncoalesced Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaDeviceSynchronize());

    float kernelMilliseconds = elapsedCudaTime(evKStart, evKStop);
    cout << "CUDA Uncoalesced Optimized (Grayscale): Timing:" << endl;
    CudaTimingUtils::printKernelTiming("CUDA Uncoalesced Optimized (Grayscale): Kernel timing:", kernelMilliseconds);
    
    // Copy results back to host
    vector<MotionVector> h_mv(blocksX * blocksY);
    GPUMemoryUtils::copyMotionVectorsFromGPU(h_mv, d_mv, blocksX, blocksY);
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv, blocksX, blocksY);
    
    // Cleanup
    LoggingUtils::printCleanupGPU("CUDA Uncoalesced Optimized");
    GPUMemoryUtils::freeMemory(d_curr, d_ref, d_mv);

    // Stop total timer after cleanup
    auto total_time_stop = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_time_stop - total_time_start).count();
    metrics.total_ms = total_time;
    metrics.gpu_kernel1_ms = kernelMilliseconds;

    destroyCudaEvent(evKStart);
    destroyCudaEvent(evKStop);
    
    LoggingUtils::printProcessingComplete("CUDA Uncoalesced Optimized");
    return result;
}
