#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"
#include "utils.h"

using namespace std;

// Computes SAD  between the current block and each candidate position in the search window.
// Each CUDA block processes one search block and parallelizes SAD computation across threads.
// Uses shared memory reduction to aggregate partial results and stores the total SAD in global memory.
// The kernel is launched with a 3D grid `gridKSad(blocksX, blocksY, maxCandidates)` where:
//  - blockIdx.x selects the block column in the current frame (x coordinate of the block)
//  - blockIdx.y selects the block row in the current frame (y coordinate of the block)
//  - blockIdx.z iterates the candidate positions in the search window (one candidate block
//    in the reference frame per z-slice). In other words, each z "slice" compares the same
//    current-frame block with one specific candidate block from the reference frame.
__global__ void computeSADKernel(const unsigned char*  d_curr,const unsigned char*  d_ref,
                                 int* d_sad,int blockSize, int width, int searchRange){

    // top-left corner of the current block in the current frame
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    const int pixelsPerBlock = blockSize * blockSize; // number of pixels in one block
    const int pixelsPerThread = (pixelsPerBlock + blockDim.x - 1) / blockDim.x; // divide pixels among threads, rounding up

    // Calculate search window using helper function
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);

    if (blockIdx.z >= bounds.totalPositions) { // out of bounds for the number of candidate positions in the search window
        return;
    }

    const int ref_x = (bounds.startX + (blockIdx.z % bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels   
    const int ref_y = (bounds.startY + (blockIdx.z / bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels
    // blockIdx.z selects which candidate (in block coordinates) is loaded from the reference frame
    
    // load shared memory 
    extern  __shared__ unsigned char smem[];             
    unsigned char* s_curr = smem;                           // shared memory for current block pixels
    unsigned char* s_ref = s_curr + pixelsPerBlock;         // shared memory for reference block pixels
    int* s_partial_sad = (int*)(s_ref + pixelsPerBlock);    // shared memory for partial SAD results (one per thread)
    
    // load blocks in shared memory with coalesced access
    // Pixel load per thread: i = threadIdx.x, threadIdx.x + blockDim.x, ...
    // Pixel count ~= pixelsPerBlock / blockDim.x (last thread may load fewer pixels)
    for (int i = threadIdx.x; i < pixelsPerBlock; i += blockDim.x) {
        int px = i % blockSize;                                         // pixel x coordinate within the block
        int py = i / blockSize;                                         // pixel y coordinate within the block
        s_curr[i] = d_curr[(y + py) * width + (x + px)];                // load current block pixel
        s_ref[i] = __ldg(&d_ref[(ref_y + py) * width + (ref_x + px)]);  // load reference block pixel (_ldg read-only cache optimization)
    }

    __syncthreads();1.	


    // partial sad computation for each thread
    int partial_sad = 0;
    const int startPixel = threadIdx.x * pixelsPerThread;                   // starting pixel index for this thread
    const int endPixel = min(startPixel + pixelsPerThread, pixelsPerBlock); // ending pixel index for this thread
    for (int i = startPixel; i < endPixel; ++i) {                           
        partial_sad += abs((int)s_curr[i] - (int)s_ref[i]);                 // accumulate SAD for assigned pixels               
    }
    s_partial_sad[threadIdx.x] = partial_sad;                   // store partial SAD in shared memory 
    __syncthreads(); // FIXME is always needed to synchronize?

    // reduction in shared memory to get total SAD for this position
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {  // threads reduce by powers of 2
        if (threadIdx.x < stride) { // only first half of threads are active in each reduction step
            s_partial_sad[threadIdx.x] += s_partial_sad[threadIdx.x + stride]; // accumulate SAD values from the second half of threads into the first half
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { // thread 0 writes the final SAD for this candidate position to global memory
        d_sad[(blockIdx.y* gridDim.x + blockIdx.x)* gridDim.z + blockIdx.z] = s_partial_sad[0]; 
    }
}
 
// Finds the best motion vector for each block by selecting the search position with the minimum SAD
__global__ void findBestMVKernel(const int* d_sad, MotionVector* d_mv, int maxCandidates, int searchRange){
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);

    // Use maxCandidates for indexing, not bounds.totalPositions
    // because d_sad array is allocated as blocksX * blocksY * maxCandidates
    // gridDim.z would be 1 (2D launch grid), so we must use the maxCandidates parameter
    const int base = (blockIdx.y * gridDim.x + blockIdx.x) * maxCandidates;

    // local best for each thread
    int local_best_sad = INT_MAX;
    int local_best_dx = 0, local_best_dy = 0;
    int local_best_dist = INT_MAX;
    for (int bz = threadIdx.x; bz < bounds.totalPositions; bz += blockDim.x) {
        const int sad = d_sad[base + bz];
        const int ref_bx = bounds.startX + (bz % bounds.width);
        const int ref_by = bounds.startY + (bz / bounds.width);
        const int dx = ref_bx - blockIdx.x;
        const int dy = ref_by - blockIdx.y;
        const int dist = dx * dx + dy * dy; // distance in blocks
        if (sad < local_best_sad || (sad == local_best_sad && dist < local_best_dist)) { // tie-breaking by distance
            local_best_sad = sad;
            local_best_dx = dx;
            local_best_dy = dy;
            local_best_dist = dist;
        }
    }
    // shared memory reduction to find global best among threads in this block
    extern __shared__ unsigned char smem[]; // char and not int because is incompatible with declaration of shared memory in computeSADKernel
    int* s_sad  = (int*)smem;
    int* s_dx = s_sad + blockDim.x;
    int* s_dy = s_dx + blockDim.x;
    int* s_dist = s_dy + blockDim.x;

    s_sad[threadIdx.x] = local_best_sad;
    s_dx[threadIdx.x] = local_best_dx;
    s_dy[threadIdx.x] = local_best_dy;
    s_dist[threadIdx.x] = local_best_dist;
    __syncthreads();

    // Sequential reduction by thread 0
    // NOTE: Comparison-based reductions with tie-breaking are non-associative and difficult
    // to parallelize correctly. The sequential approach in SMEM is simpler and guaranteed correct.
    if (threadIdx.x == 0) {
        int global_best_sad = s_sad[0];
        int global_best_dx = s_dx[0];
        int global_best_dy = s_dy[0];
        int global_best_dist = s_dist[0];
        
        for (int i = 1; i < blockDim.x; i++) {
            if (s_sad[i] < global_best_sad ||
                (s_sad[i] == global_best_sad && s_dist[i] < global_best_dist)) {
                global_best_sad = s_sad[i];
                global_best_dx = s_dx[i];
                global_best_dy = s_dy[i];
                global_best_dist = s_dist[i];
            }
        }
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {global_best_dx, global_best_dy};
    }
}

// HOST CODE
vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchRange) {
    cudaSetDevice(0);
    
    ValidationUtils::validateFrameDimensions(curr, ref);
    
    // Calculate grid dimensions
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    const int pixelsPerBlock = blockSize * blockSize;
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Optimized", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CUDA Optimized", searchRange);
    
    // Calculate search window dimensions
    const int searchWindowBlocksX = (searchRange > 0) ? min(blocksX, searchRange * 2 + 1) : blocksX;
    const int searchWindowBlocksY = (searchRange > 0) ? min(blocksY, searchRange * 2 + 1) : blocksY;
    const int maxCandidates = searchWindowBlocksX * searchWindowBlocksY;

    // Device properties
    cudaDeviceProp deviceProp;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Verify that grid dimensions do not exceed device limits
    if ((size_t)maxCandidates > (size_t)deviceProp.maxGridSize[2])
        throw runtime_error("Total candidates (" + to_string(maxCandidates) +
                            ") exceeds device maxGridSize[2] (" +
                            to_string(deviceProp.maxGridSize[2]) + ").");

    cout << "CUDA Optimized (Grayscale): GPU: " << deviceProp.name << "\n"
         << "  maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << "\n"
         << "  sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " bytes\n";
    
    // SMEM configuration and threads per block selection
    int threadsPerBlock = 32;
    size_t smKSad, smKBestMV;
        
        // Descend by powers of 2
        for (int t = deviceProp.maxThreadsPerBlock; t >= 32; t >>= 1) {
            // [curr + ref]
            // [partial sads]
            size_t smKSad_try = 2 * pixelsPerBlock * sizeof(unsigned char) + t * sizeof(int);   
            
            // Needs 4 parallel arrays for reduction (sad, dx, dy, dist):
            //   - s_sad[t]  (sad values)
            //   - s_dx[t]   (delta x)
            //   - s_dy[t]   (delta y)
            //   - s_dist[t] (distance for tie-breaking)
            size_t smKBestMV_try = 4 * t * sizeof(int);
            
            // check if number of threads t works for both kernels within shared memory limits
            if (smKSad_try <= deviceProp.sharedMemPerBlock &&
                smKBestMV_try <= deviceProp.sharedMemPerBlock) {
                threadsPerBlock = t;
                smKSad = smKSad_try;
                smKBestMV = smKBestMV_try;
                break;  
            }
        }
        // use conservative default of 32 threads if no larger configuration fits in shared memory
        if (threadsPerBlock == 32) {
            smKSad = 2 * pixelsPerBlock * sizeof(unsigned char) + 32 * sizeof(int);
            smKBestMV = 4 * 32 * sizeof(int);
        }
    
   

    /*size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (sadBytes > freeMem * 0.8)
        throw runtime_error("Not enough GPU memory for d_sad array ("
                            + to_string(sadBytes / (1024 * 1024)) + " MB needed, "
                            + to_string(freeMem  / (1024 * 1024)) + " MB free).");*/
                        
    // Allocate device memory for current and reference frames
    const size_t sadBytes = (size_t)blocksX * blocksY * maxCandidates * sizeof(int);
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    
    int* d_sad = nullptr;
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc(&d_sad, sadBytes));
    gpuErrorCheck(cudaMalloc(&d_mv, (size_t)blocksX * blocksY * sizeof(MotionVector)));

    cout << "CUDA Optimized (Grayscale): Selected threads per block: " << threadsPerBlock
         << " (K1=" << smKSad/1024.0 << "KB, K2=" << smKBestMV/1024.0 << "KB)\n\n";
    cout << "CUDA Optimized (Grayscale): d_sad = " << sadBytes / (1024.0 * 1024.0) << " MB\n";
    
    LoggingUtils::printCopyingToGPU("CUDA Optimized");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);

    // Grid and block dimensions
    const dim3 gridKSad(blocksX, blocksY, maxCandidates);
    const dim3 gridKBestMV(blocksX, blocksY);
    const dim3 blockDim(threadsPerBlock, 1, 1);

    // CUDA timing events
    cudaEvent_t evTotalStart, evTotalStop;
    cudaEvent_t evKSadStart, evKSadStop;
    cudaEvent_t evKBestMVStart, evKBestMVStop;

    createCudaEvent(evTotalStart); createCudaEvent(evTotalStop);
    createCudaEvent(evKSadStart); createCudaEvent(evKSadStop);
    createCudaEvent(evKBestMVStart); createCudaEvent(evKBestMVStop);

    recordCudaEvent(evTotalStart);
    recordCudaEvent(evKSadStart);

    cout << "CUDA Optimized (Grayscale): Launching computeSADKernel with "
         << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
         << " threads per block (" << threadsPerBlock << " total threads)..." << endl;

    computeSADKernel<<<gridKSad, blockDim, smKSad>>>(d_curr, d_ref, d_sad,
                                                     blockSize, curr.width, searchRange);
    gpuErrorCheck(cudaGetLastError());
    recordCudaEvent(evKSadStop);

    recordCudaEvent(evKBestMVStart);

    cout << "CUDA Optimized (Grayscale): Launching findBestMVKernel with "
         << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
         << " threads per block (" << threadsPerBlock << " total threads)..." << endl;

    findBestMVKernel<<<gridKBestMV, blockDim, smKBestMV>>>(d_sad, d_mv, maxCandidates, searchRange);
    gpuErrorCheck(cudaGetLastError());
    recordCudaEvent(evKBestMVStop);
    recordCudaEvent(evTotalStop);

    cout << "CUDA Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaDeviceSynchronize());

    // Timing report
    const float msKSad = elapsedCudaTime(evKSadStart, evKSadStop);
    const float msKBestMV = elapsedCudaTime(evKBestMVStart, evKBestMVStop);
    const float msTotal = elapsedCudaTime(evTotalStart, evTotalStop);

    cout << "CUDA Optimized (Grayscale): Timing:" << endl;
    CudaTimingUtils::printKernelTiming("computeSADKernel", msKSad);
    CudaTimingUtils::printKernelTiming("findBestMVKernel", msKBestMV);
    CudaTimingUtils::printKernelTiming("Total", msTotal);

    destroyCudaEvent(evTotalStart); destroyCudaEvent(evTotalStop);
    destroyCudaEvent(evKSadStart); destroyCudaEvent(evKSadStop);
    destroyCudaEvent(evKBestMVStart); destroyCudaEvent(evKBestMVStop);

    // Copy results back to host
    vector<MotionVector> h_mv_flat(blocksX * blocksY);
    GPUMemoryUtils::copyMotionVectorsFromGPU(h_mv_flat, d_mv, blocksX, blocksY);

    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);

    // Cleanup GPU
    LoggingUtils::printCleanupGPU("CUDA Optimized");
    GPUMemoryUtils::freeMemory(d_curr, d_ref, d_sad, d_mv);

    LoggingUtils::printProcessingComplete("CUDA Optimized");
    return result;
}


