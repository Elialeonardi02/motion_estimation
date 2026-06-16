#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "fullSearchBM_cuda_optimized_uncoalesced.h"
#include "utils.h"
#include "cuda_utils.h"
#include <chrono>

using namespace std;

/* Reduces partial SAD values with cooperative groups warp-level reduction (reduction1)
    @tparam size: hardware tile size in threads (must be a power of 2)
    @param warp: cooperative groups tile object (thread subgroup of cuda block).
    @param sad: partial SAD value computed by this specific thread.
    @param thread_group_size: size of the group collaborating on this SAD.
    @return int: the total reduced SAD value. Only the first thread of the group (thread rank 0) hold the correct final sum.
*/
template <unsigned int size>
__device__ int reduce_partial_sad_warp(cooperative_groups::thread_block_tile<size> warp, int sad, int thread_group_size) {
    for (int offset = thread_group_size >> 1; offset > 0; offset >>=1) { 
        sad += warp.shfl_down(sad, offset);
    }
    return sad;
}

/* Reduce SAD values between all candidates and find the best candidate (reduction2)
    @tparam size: hardware tile size in threads (must be a power of 2)
    @param warp: cooperative groups tile object (thread subgroup of cuda block).
    @param candidate: the candidate motion vector and SAD value for this specific thread.
    @return CandidateSad: the best candidate after reduction. Only the first thread of the group (thread rank 0) hold the best candidate
*/
template <unsigned int size>
__device__ CandidateSad reduce_best_candidate_warp(cooperative_groups::thread_block_tile<size> warp, CandidateSad candidate) {
    for (int offset = warp.size() >> 1 ; offset > 0; offset >>= 1) {
        // current thread's candidate distance
        int dist = candidate.dx * candidate.dx + candidate.dy * candidate.dy;
        // other thread's candidate values
        int other_sad = warp.shfl_down(candidate.sad, offset); 
        int other_dx = warp.shfl_down(candidate.dx, offset);
        int other_dy = warp.shfl_down(candidate.dy, offset);
        int other_dist = other_dx * other_dx + other_dy * other_dy;
        // Compare SAD values and distances for tie-breaking
        if (other_sad < candidate.sad || (other_sad == candidate.sad && other_dist < dist)) {
            candidate.sad = other_sad;
            candidate.dx = other_dx;
            candidate.dy = other_dy;
        }
    }
    return candidate;
}

/* Device function to compute SAD for a portion of the block, used for parallelizing SAD computation along threads in X dimension
    @param curr: pointer to current block pixels in SMEM.
    @param ref: pointer to reference frame pixels in GMEM.
    @param ref_x, ref_y: top-left corner of the candidate block in reference frame in pixels.
    @param blockSize: size of the frame blocks.
    @param refWidth: width of the reference frame.
    @param startPixel: starting pixel index within the block for this thread to process (0 to blockSize*blockSize).
    @param pixelsPerThread: number of pixels this thread should process for SAD computation.
    @return int: partial SAD value computed by this thread for its assigned pixels.
*/
static __device__ int computeSAD_device_partial(const unsigned char* curr, const unsigned char* ref,
                                                int ref_x, int ref_y, int blockSize, int refWidth,
                                                int startPixel, int pixelsPerThread) {
    int sad = 0;
    // loop over assigned pixels for this thread
    for (int pixelIdx = startPixel; pixelIdx < startPixel + pixelsPerThread; pixelIdx++) { 
        if (pixelIdx >= blockSize * blockSize) break; // boundary check
        // convert pixelIdx to 2D coordinates within the block
        int y = pixelIdx / blockSize;
        int x = pixelIdx % blockSize;
        sad += abs(curr[pixelIdx] - ref[(ref_y + y) * refWidth + (ref_x + x)]);
    }
    return sad;
}


/* CUDA kernel, each thread YZ computes SAD for one candidate position in the search window, using threads in X dimension to parallelize SAD computation, 
and finds the best match using cooperative groups reduction (both reduction1 and reduction2)
    @param d_curr: pointer to current frame in GMEM.
    @param d_ref: pointer to reference frame in GMEM.
    @param d_mv: pointer to final motion vectors in GMEM.
    @param blockSize: size of the frame blocks.
    @param width: width of the frames.
    @param searchRange: range of the search window.
*/
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize,
                                 int width, int searchRange)  {
    // cooperative groups setup and index
    // group thread idx: threadIdx.x
    // group thread size: blockDim.x
    cooperative_groups :: thread_block block = cooperative_groups :: this_thread_block();
    cooperative_groups :: thread_block_tile<32> warp = cooperative_groups :: tiled_partition<32>(block);

    // search window candidate position:
    // reference frame block y: threadIdx.y
    // reference frame block x: threadIdx.z


    // Thread indices
    const int threadsYZ= blockDim.y * blockDim.z; // number of threads collaborating on different candidate positions for the same frame block
    const int tidx_yz= threadIdx.z * blockDim.y + threadIdx.y; // unique thread index along YZ dimensions, used to assign candidate positions 
    //block.thread_rank(): tidx_3d = threadIdx.z * blockDim.y * blockDim.x + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Current block top-left corner in pixels
    const int x = blockIdx.x * blockSize;
    const int y = blockIdx.y * blockSize;
    
    // Calculate search window using helper function
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    
    // Thread-local best match
    CandidateSad local_best={INT_MAX, 0, 0};
    
    // Shared memory layout: current frame block + buffer for best candidates from each thread
    extern __shared__ unsigned char smem[];
    unsigned char* s_curr = smem;
    // align to avoid bank conflict
    size_t alignedCurrPixelBytes = (blockSize * blockSize * sizeof(unsigned char) + alignof(CandidateSad) - 1) & ~(alignof(CandidateSad) - 1);
    CandidateSad* best_thread_results = (CandidateSad*)(s_curr + alignedCurrPixelBytes);
    
    // Load current frame block in shared memory by all threads in the block (blockDim.x * blockDim.y * blockDim.z)
    for (int i = block.thread_rank(); i < blockSize * blockSize; i += blockDim.x * blockDim.y * blockDim.z) {
        s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
    }
    // Ensure all threads have loaded the current block before any thread can read from it
    block.sync();
    
    // Precalculate pixels-per-thread for SAD parallelization
    const int pixelsPerThread = (blockSize * blockSize + blockDim.x - 1) / blockDim.x;
    
    // Each thread (X,Y,Z) searches positions and computes partial SAD along X
    // threadsYZ < totalPosition, each YZ thread handles partial SAD for its assigned positions (i, i+threadsYZ, i+2*threadsYZ, ...)
    // threadsYZ = totalPosition, each YZ thread compute partial SAD for only one assigned position.
    const int totalIterations = (bounds.totalPositions + threadsYZ - 1) / threadsYZ;
    for (int iter = 0; iter < totalIterations; iter++) {
        const int pos = iter * threadsYZ + tidx_yz; // global candidate position index for this thread to process
        const bool validPos = (pos < bounds.totalPositions);
        // Convert position to 2D frame block coordinates within search window
        int ref_bx = validPos ? bounds.startX + (pos % bounds.width) : 0; // for invalid position, set to 0 
        int ref_by = validPos ? bounds.startY + (pos / bounds.width) : 0; // for invalid position, set to 0
        
        // Compute partial SAD for this thread
        int partial_sad = validPos ? computeSAD_device_partial(
            s_curr, d_ref, ref_bx * blockSize, ref_by * blockSize, blockSize, width,
            threadIdx.x * pixelsPerThread, pixelsPerThread) : 0;
        
        // reduction1: reduce partial SAD values from threads in X dimension to get total SAD for this candidate position
        int total_sad = reduce_partial_sad_warp(warp, partial_sad, blockDim.x);
    
        // 0YZ thread writes the total SAD and candidate motion vector to shared memory for reduction2
        if (threadIdx.x == 0 && validPos) {
            // motion vector components in block coordinates
            int ref_dx = ref_bx - blockIdx.x; 
            int ref_dy = ref_by - blockIdx.y;
            // tie-breaking using distance of motion vector
            int dist = ref_dx * ref_dx + ref_dy * ref_dy;
            int best_dist = local_best.dx * local_best.dx + local_best.dy * local_best.dy;
            if (total_sad < local_best.sad || (total_sad == local_best.sad && dist < best_dist)) {
                local_best.sad = total_sad;
                local_best.dx = ref_dx;
                local_best.dy = ref_dy;
            }
        }
    } 
    if (threadIdx.x == 0){
        best_thread_results[tidx_yz] = local_best; 
    }
    block.sync();
    
    // reduction2: reduce SAD values between all candidates with 0YZ threads and find the best candidate
    CandidateSad block_best = {INT_MAX, 0, 0};
    int best_dist = INT_MAX;
    

    // Each warp processes a portion of the best_thread_results buffer, 
    // then the first warp reduces the best candidates from all warps to find the final best candidate
    if (block.thread_rank()/32 == 0){ //warp_id == 0, only the thread of the warp 0 will perform the final reduction
        for (int i = warp.thread_rank(); i < threadsYZ; i+=32){ // loop over candidates for this warp
            CandidateSad candidate = best_thread_results[i]; 
            int dist = candidate.dx * candidate.dx + candidate.dy * candidate.dy;
            if (candidate.sad < block_best.sad || (candidate.sad == block_best.sad && dist < best_dist)) {
                block_best = candidate;
                best_dist = dist;
            }
        }
        block_best = reduce_best_candidate_warp(warp, block_best);
    }
    // The first thread writes the best motion vector for this current frame block to GMEM
    if (block.thread_rank() == 0){
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {block_best.dx, block_best.dy};
    }
}

/* Full search block matching implementation using CUDA with uncoalesced memory access but optimized with SMEM and cooperative groups reduction
    @param curr: current frame in grayscale.
    @param ref: reference frame in grayscale.
    @param blockSize: size of the frame blocks.
    @param searchRange: range of the search window (<=0: full search).
    @param metrics: struct to store timing metrics for this function.
    @return 2D vector of motion vectors for each block in the current frame.
*/
 vector<vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(const ImageGray& curr, const ImageGray& ref,
                                                                    int blockSize, int searchRange, SingleRunMetrics& metrics) {
    // total time
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
    LoggingUtils::printFrameInfo("CUDA Uncoalesced Optimized", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CUDA Uncoalesced Optimized", searchRange);
    
    // Determine threads per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threadsPerBlockX = 1; // X dimension is used for parallelizing SAD computation 
    int threadsPerBlockY = blocksX; // Y dimension is used for candidate positions along width of search window
    int threadsPerBlockZ = blocksY; // Z dimension is used for candidate positions along height of search window
    if (searchRange > 0) {
        threadsPerBlockY = min(blocksX, searchRange * 2 + 1);
        threadsPerBlockZ = min(blocksY, searchRange * 2 + 1);
    } 
    if (threadsPerBlockY * threadsPerBlockZ <=prop.maxThreadsPerBlock){
        int maxThreadsPerPartialSad = min(32, prop.maxThreadsPerBlock / (threadsPerBlockY * threadsPerBlockZ));
        // Determine the largest power of two for threads in X dimension to parallelize SAD computation
        while (threadsPerBlockX << 1 <= maxThreadsPerPartialSad) {
            threadsPerBlockX <<= 1;
        }
    }else{ // trashold configuration that exceeds max threads per block
        threadsPerBlockX = 4; 
        threadsPerBlockY = 16;
        threadsPerBlockZ = 16;
    }
    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;

    // Allocate GPU memory for frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    LoggingUtils::printCopyingToGPU("CUDA Uncoalesced Optimized");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);

    // Allocate GPU memory for motion vectors
    size_t mvSize = (size_t)blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    // SMEM size: current block + buffer for best candidates from each thread
    size_t alignedCurrPixelBytes = (blockSize * blockSize * sizeof(unsigned char) + alignof(CandidateSad) - 1) & ~(alignof(CandidateSad) - 1);
    size_t sharedMemSize = alignedCurrPixelBytes + (threadsPerBlockY * threadsPerBlockZ) * sizeof(CandidateSad);
    cout << "CUDA Uncoalesced Optimized (Grayscale): Shared memory per block: " << sharedMemSize / 1024.0f << " KB" << endl;
    
    // Launch CUDA kernel
    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);

    cout << "CUDA Uncoalesced Optimized (Grayscale): Launching kernel with " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
        << " threads per block (" << threadsPerBlock << " total threads)..." << endl;
    
    recordCudaEvent(evKStart);
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, searchRange);
    recordCudaEvent(evKStop);
    cout << "CUDA Uncoalesced Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaEventSynchronize(evKStop));
    gpuErrorCheck(cudaGetLastError());
    
    // kernel timing
    float kernelMilliseconds = elapsedCudaTime(evKStart, evKStop);
    CudaTimingUtils::printKernelTiming("CUDA Uncoalesced Optimized (Grayscale): Kernel timer:", kernelMilliseconds);
    
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
