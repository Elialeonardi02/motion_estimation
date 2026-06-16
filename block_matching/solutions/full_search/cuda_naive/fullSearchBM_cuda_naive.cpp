#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "fullSearchBM_cuda_naive.h"
#include "utils.h"
#include "cuda_utils.h"
#include "sad_utils.h"
#include <chrono>

using namespace std;

/* Device function to compute SAD between two blocks
    @param d_curr: pointer to current frame in GMEM.
    @param d_ref: pointer to reference frame in GMEM.
    @param curr_x, curr_y: top-left corner of the block in current frame in pixels.
    @param ref_x, ref_y: top-left corner of the block in reference frame in pixels.
    @param blockSize: size of the frame blocks.
    @param width, height: dimensions of the frames.
    @return int: sad value between the two blocks, or INT_MAX if any part of the block is out of frame bounds.
*/
static __device__ __inline__ int computeSAD_device(const unsigned char* d_curr, const unsigned char* d_ref,
                                int curr_x, int curr_y, int ref_x, int ref_y, int blockSize, int width, int height) {
    int sad = 0;
    
    // loop over block rows
    for(int by = 0; by < blockSize; by++) { 
        int cy = curr_y + by; // top-left corner of the block in current frame in pixels + block row offset
        int ry = ref_y + by; // top-left corner of the block in reference frame in pixels + block row offset
        if (cy >= height || ry >= height || cy < 0 || ry < 0) return INT_MAX; // overflow check: if we are beyond the height of the frame
        // loop over block columns
        for(int bx = 0; bx < blockSize; bx++) { 
            int cx = curr_x + bx; // top-left corner of the block in current frame in pixels + block column offset
            int rx = ref_x + bx; // top-left corner of the block in reference frame in pixels + block column offset
            if (cx >= width || rx >= width || cx < 0 || rx < 0) return INT_MAX; // overflow check: if we are beyond the width of the frame
            // compute and accumulate SAD
            sad += abs((int)d_curr[cy * width + cx] - (int)d_ref[ry * width + rx]);
        }
    }
    return sad;
}

/* CUDA kernel: performs full search block matching, each thread computes SAD for one candidate position in the search window, and finds the best match using block reduction
    @tparam threadsPerBlockDim: number of threads per cuda block dimension
    @param d_curr: pointer to current frame in GMEM.
    @param d_ref: pointer to reference frame in GMEM.
    @param d_mv: pointer to final motion vectors in GMEM.
    @param blockSize: size of the frame blocks.
    @param width, height: dimensions of the frames.
    @param searchRange: range of the search window.
*/
template <int threadsPerBlockDim>
__global__ void fullSearchNaiveKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                MotionVector* d_mv, int blockSize, int width, int height, int searchRange)  {
    // Grid and thread indices
    int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    int constexpr total_threads = threadsPerBlockDim * threadsPerBlockDim; // total threads per block
    
    // Current block top-left corner in pixels
    int x = blockIdx.x * blockSize;
    int y = blockIdx.y * blockSize;
    
    // Calculate search window using helper function
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    // Thread-local best match
    CandidateSad local_best = {INT_MAX, 0, 0}; // initialize local best SAD to max and motion vector to (0,0)
    int  bestDist = INT_MAX; // for tie-breaking, initialize to max
    
    // Each thread computes SAD for its assigned candidate positions in the search window (Reduction1)
    // total_threads < totalPosition, thread processes its assigned positions (i, i+total_threads, i+2*total_threads, ...)
    // total_threads = totalPosition, thread processes only one assigned position.
    for (int pos = tidx; pos < bounds.totalPositions; pos += total_threads) {
        // Convert position to 2D coordinates within search window
        int ref_bx = bounds.startX + (pos % bounds.width); // candidate block top-left corner in reference frame in block coordinates
        int ref_by = bounds.startY + (pos / bounds.width); // candidate block top-left corner in reference frame in block coordinates
        int ref_x = ref_bx * blockSize; // candidate block top-left corner in reference frame in pixels
        int ref_y = ref_by * blockSize; // candidate block top-left corner in reference frame in pixels
        
        int sad = computeSAD_device(d_curr, d_ref, x, y, ref_x, ref_y, blockSize, width, height); // compute SAD for the candidate position
    
        int dx = ref_bx - blockIdx.x; // motion vector x component in block coordinates
        int dy = ref_by - blockIdx.y; // motion vector y component in block coordinates
        int dist = dx *dx + dy * dy;// local best distance for tie-breaking
        if (sad < local_best.sad || (sad == local_best.sad && dist < bestDist)) { // update local best match
            local_best.sad = sad; 
            local_best.dx = dx ; 
            local_best.dy = dy; 
            bestDist = dist;  
        }
    }
    
    // Block reduction to find the best match among all threads in the block (reduction2)
    typedef cub::BlockReduce<CandidateSad, threadsPerBlockDim, cub::BLOCK_REDUCE_WARP_REDUCTIONS, threadsPerBlockDim> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    CandidateSad block_best = BlockReduceT(temp_storage).Reduce(local_best, MotionVectorUtils::CandidateSadOp());
    
    // Write the best motion vector for this current frame block to gmem
    if (tidx == 0){
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {block_best.dx, block_best.dy};
    }
}

/* Full search block matching implementation using CUDA naive approach for grayscale images
    @param curr: current frame in grayscale.
    @param ref: reference frame in grayscale.
    @param blockSize: size of the frame blocks.
    @param searchRange: range of the search window.
    @param metrics: struct to store timing metrics for this function.
    @return 2D vector of motion vectors for each block in the current frame, where each motion vector represents the best match found in the reference frame within the search window
*/
vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                    int blockSize, int searchRange, SingleRunMetrics& metrics) {
    
    auto total_time_start = std::chrono::high_resolution_clock::now();
    
    // Create CUDA timer                                                 
    cudaEvent_t  evKStart, evKStop;
    createCudaEvent(evKStart);
    createCudaEvent(evKStop);

    cudaSetDevice(0);
    
    
    // Calculate grid dimensions
    ValidationUtils::validateFrameDimensions(curr, ref);
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Naive", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CUDA Naive", searchRange);
    
    // Determine threads per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlockX; 
    int threadsPerBlockY;
    int threadsPerBlockDim = 1;  
    
    // searchRange > 0,  range search
    if (searchRange > 0) {
        // limit threads per block based on search range
        // if range is bigger than frame size, it will be clamped to frame size.
        threadsPerBlockX = min(blocksX, searchRange * 2 + 1); 
        threadsPerBlockY = min(blocksY, searchRange * 2 + 1); 
    }
    else { // searchRange <= 0, full search
        threadsPerBlockX = blocksX;
        threadsPerBlockY = blocksY;
    }
    // Determine the largest power of two that is less than or equal to the minimum of threadsPerBlockX * threadsPerBlockY
    int threadsPerBlockDimLimit = min(threadsPerBlockX * threadsPerBlockY, prop.maxThreadsPerBlock);
    while ((threadsPerBlockDim << 1) * (threadsPerBlockDim << 1) <= threadsPerBlockDimLimit) {
        threadsPerBlockDim <<= 1;
    }
    // Allocate and copy GPU memory for frames
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

    // Launch CUDA kernel
    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockDim, threadsPerBlockDim);
    
    cout << "CUDA Naive (Grayscale): Launching kernel with " << blockDim.x << "x" << blockDim.y 
        << " threads per block (" << (threadsPerBlockDim * threadsPerBlockDim) << " total threads)..." << endl;
    
    recordCudaEvent(evKStart);
    switch (threadsPerBlockDim) {
        case 1: // 1 thread per block
            fullSearchNaiveKernel<1><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
        case 2: // 4 threads per block
            fullSearchNaiveKernel<2><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
        case 4: // 16 threads per block
            fullSearchNaiveKernel<4><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
        case 8: // 64 threads per block
            fullSearchNaiveKernel<8><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
        case 16: // 256 threads per block
            fullSearchNaiveKernel<16><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
        case 32: // 1024 threads per block
            fullSearchNaiveKernel<32><<<gridDim, blockDim>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, searchRange);
            break;
    }
    recordCudaEvent(evKStop);
    cout << "CUDA Naive (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaEventSynchronize(evKStop));
    gpuErrorCheck(cudaGetLastError());
    
    // kernel timing
    float kernelMilliseconds = elapsedCudaTime(evKStart, evKStop);
    CudaTimingUtils::printKernelTiming("CUDA Naive (Grayscale): Kernel timer:", kernelMilliseconds);
    
    // Copy results back to host
    vector<MotionVector> h_mv(blocksX * blocksY);
    GPUMemoryUtils::copyMotionVectorsFromGPU(h_mv, d_mv, blocksX, blocksY);
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv, blocksX, blocksY);
    
    // Cleanup
    LoggingUtils::printCleanupGPU("CUDA Naive");
    GPUMemoryUtils::freeMemory(d_curr, d_ref, d_mv);
    // Stop total timer after cleanup
    auto total_time_stop = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_time_stop - total_time_start).count();
    metrics.total_ms = total_time;
    metrics.gpu_kernel1_ms = kernelMilliseconds;
    
    destroyCudaEvent(evKStart);
    destroyCudaEvent(evKStop);
    
    LoggingUtils::printProcessingComplete("CUDA Naive");
    return result;
}