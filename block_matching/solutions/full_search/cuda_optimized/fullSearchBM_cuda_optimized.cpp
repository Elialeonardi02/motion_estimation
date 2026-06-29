#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "fullSearchBM_cuda_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"
#include "utils.h"
#include <chrono>

using namespace std;

// SMEM strategy, determine at compile time, base on block size, where to store current and reference blocks
enum class SmemStrategy {
    FULL,       // curr block + ref + CUB SMEM
    CURR_ONLY,  // only curr block +  CUB SMEM (same curr block is read from all cuda blocks along Z)
    NONE        // CUB SMEM
};

/* kernel to compute SAD for each candidate position in the search window::
    @tparam STRATEGY: SMEM strategy, determines where to load current and reference blocks from (SMEM or GMEM)
    @tparam KSAD_THREADS: number of threads per block for SAD computation
    @param d_curr: pointer to current frame in GMEM.
    @param d_ref: pointer to reference frame in GMEM.
    @param curr_x, curr_y: top-left corner of the block in current frame in pixels.
    @param ref_x, ref_y: top-left corner of the block in reference frame in pixels.
    @param blockSize: size of the frame blocks.
    @param width, height: dimensions of the frames.
    @return int: sad value between the two blocks, or INT_MAX if any part of the block is out of frame bounds.
__restrict__ is used to tell the compiler that the memory pointed to by d_curr and d_ref will not be accessed through any other pointer
this allow the compiler to optimize memory access using read-only cache, instead of going to GMEM for every access
*/
template<SmemStrategy STRATEGY, int KSAD_THREADS> 
__global__ void computeSADKernel(
    const unsigned char* __restrict__ d_curr, 
    const unsigned char* __restrict__ d_ref,
    int* d_sad, int blockSize, int width, int searchRange, int maxCandidates) 
{   
    // Calculate search window bounds for the current block
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    const int pixelsPerBlock = blockSize * blockSize; // pixels in a frame block
    // Calculate the top-left corner of the current block in pixels
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    // BlockReduce from CUB library for parallel reduction of partial SAD values computed by threads in the block
    using BlockReduce = cub::BlockReduce<int, KSAD_THREADS>;
    __shared__ typename BlockReduce::TempStorage partial_sad_reduce_storage;

    // SMEM layout: use SMEM based on the selected strategy
    extern __shared__ unsigned char smemKSad[];
    unsigned char* s_curr = (STRATEGY != SmemStrategy::NONE) ? smemKSad : nullptr;
    unsigned char* s_ref = (STRATEGY == SmemStrategy::FULL) ? smemKSad + pixelsPerBlock : nullptr;

    // load current frame block in SMEM if strategy is FULL or CURR_ONLY
    if constexpr (STRATEGY != SmemStrategy::NONE) {
        for (int i = threadIdx.x; i < pixelsPerBlock; i += KSAD_THREADS) {
            s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
        }
        __syncthreads();
    }
    // Each thread computes a portion of the SAD for its assigned candidate positions in the search window
    // blockDim.z >= totalPositions, each thread computes partial SAD for only one assigned position.
    // blockDim.z < totalPositions, each thread computes partial SAD for its assigned positions (i, i+blockDim.z, i+2*blockDim.z, ...)
    for (int flat_idx = blockIdx.z; flat_idx < bounds.totalPositions; flat_idx += gridDim.z) {
        // flat to 2D conversion to get candidate block's top-left corner in reference frame in pixels
        const int ref_x = (bounds.startX + (flat_idx % bounds.width)) * blockSize;   
        const int ref_y = (bounds.startY + (flat_idx / bounds.width)) * blockSize;

        // load also reference frame block in SMEM if strategy is FULL
        if constexpr (STRATEGY == SmemStrategy::FULL) {
            for (int i = threadIdx.x; i < pixelsPerBlock; i += KSAD_THREADS) {
                s_ref[i] = d_ref[(ref_y + (i / blockSize)) * width + (ref_x + (i % blockSize))];
            }
            __syncthreads();
        }

        int partial_sad = 0; 
        // Each thread computes partial SAD for its assigned pixels in the block
        for (int i = threadIdx.x; i < pixelsPerBlock; i += KSAD_THREADS) {
            unsigned char c, r;
            if constexpr (STRATEGY == SmemStrategy::FULL) {
                c = s_curr[i];
                r = s_ref[i];
            } else if constexpr (STRATEGY == SmemStrategy::CURR_ONLY) {
                c = s_curr[i];
                r = d_ref[(ref_y + ( i / blockSize)) * width + (ref_x + (i % blockSize))];
            } else {
                int row = i / blockSize;
                int col = i % blockSize;
                c = d_curr[(y + row) * width + (x + col)];
                r = d_ref[(ref_y + row) * width + (ref_x + col)];
            }
            
            partial_sad = __usad(c, r, partial_sad);
        }
        // reduce partial SAD values from threads in the block to get total SAD for this candidate position
        const int total_sad = BlockReduce(partial_sad_reduce_storage).Sum(partial_sad);


        if (threadIdx.x == 0) { 
            d_sad[(blockIdx.y * gridDim.x + blockIdx.x) * maxCandidates + flat_idx] = total_sad; 
        }
        
        __syncthreads(); 
    }
}

// Data structure to hold SAD and motion vector components for comparison during reduction in findBestMVKernel
// grouping SAD + candidate index in unique object recudcible by CUB BlockReduce
struct CandidateSadLidx {
    int sad; // SAD value for this candidate position
    int flat_idx;  // candidate block index in search window, used to calculate motion vector components dx, dy
};
// Finds the best motion vector for each block
// __forceinline__: reduce function call overheard, this function is called in the reduction loop for every candidate.
// CUB need a binary operator to compare and reduce CandidateSadLidx objecct, CUB is a template library: the operatore will be inlined at compile time inside the reduction loop
// normal function device ponter cannot be used as binary operator for CUB reduction 
struct CandidateSadOp {
    int startX, startY, width, blockX, blockY; // parameters for calculating candidate block's top-left corner and motion vector components from candidate index bz
    
     // constructor to initialize the parameters needed for calculating candidate block's position and motion vector components 
    __device__ __forceinline__ CandidateSadOp(int startX, int startY, int width, int blockX, int blockY) 
        : startX(startX), startY(startY), width(width), blockX(blockX), blockY(blockY) {}

    // calculate squared distance of the candidate motion vector from the current block's position, used for tie-breaking when SAD values are equal
    __device__ __forceinline__ int getDistSq(int flat_idx) const {
        int ref_bx = startX + (flat_idx % width); // candidate block's top-left corner x in the reference frame
        int ref_by = startY + (flat_idx / width); // candidate block's top-left corner y in the reference frame
        int dx = ref_bx - blockX; // motion vector x component
        int dy = ref_by - blockY; // motion vector y component
        return dx * dx + dy * dy; // don't need to calculate actual distance, just compare squared distance for tie-breaking, to avoid costly sqrt operation
    }

    // comparison operator for reduction: returns the better of two candidates based on SAD value, with tie-breaking by distance to prefer shorter motion vectors when SAD values are equal
    __device__ __forceinline__
    CandidateSadLidx operator()(const CandidateSadLidx& a, const CandidateSadLidx& b) const {
        if (a.sad != b.sad) {
            return (a.sad < b.sad) ? a : b;
        }
        
        int distA = getDistSq(a.flat_idx);
        int distB = getDistSq(b.flat_idx);
        if (distA != distB) {
            return (distA < distB) ? a : b;
        }
        
        return (a.flat_idx < b.flat_idx) ? a : b;
    }
};
/* kernel to find the best motion vector for each block based on SAD values computed in computeSADKernel:
    @tparam BLOCK_REDUCE_THREADS: number of threads per block for reduction, must be a power of 2
    @param d_sad: pointer to SAD values in GMEM, computed by computeSADKernel
    @param d_mv: pointer to output motion vectors in GMEM, one per block
    @param maxCandidates: maximum number of candidate positions in the search window, used for indexing d_sad
    @param searchRange: search range in pixels, used for calculating search window bounds

*/
template<int BLOCK_REDUCE_THREADS> 
    __global__ void findBestMVKernel(const int* d_sad, MotionVector* d_mv, int maxCandidates, int searchRange){
    
    // Calculate search window using helper function, to determine bounds for indexing d_sad and calculating dx, dy for each candidate position
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);

    // Use maxCandidates for indexing, not bounds.totalPositions
    // because d_sad array is allocated as blocksX * blocksY * maxCandidates
    const size_t base = (blockIdx.y * gridDim.x + blockIdx.x) * maxCandidates;

    CandidateSadLidx local_best = {INT_MAX, -1}; 
    const CandidateSadOp op(bounds.startX, bounds.startY, bounds.width, blockIdx.x, blockIdx.y); // initialize the comparison operator with parameters needed to calculate candidate block's position and motion vector components 

    // each thread compares a subset of candidate positions in the search window to find the best motion vector for this block 
    for (int flat_idx = threadIdx.x; flat_idx < bounds.totalPositions; flat_idx += BLOCK_REDUCE_THREADS) {
        const CandidateSadLidx candidate = {
            d_sad[base + flat_idx], // SAD value for this candidate position, read from GMEM
            flat_idx     // candidate block linear index in the search window, used to calculate candidate block's position and motion vector components in the comparison operator
        };
        local_best = op(local_best, candidate);
    }
    
    // CUB block reduce for finding the best motion vector among candidate positions for this block, using CandidateSadOp as the reduction operator
    using BlockReduce = cub::BlockReduce<CandidateSadLidx, BLOCK_REDUCE_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
  
    const CandidateSadLidx result = BlockReduce(reduce_storage).Reduce(local_best, op);

    if (threadIdx.x == 0){  // single thread writes the best motion vector for this block to global memory
        int ref_bx = bounds.startX + (result.flat_idx % bounds.width);
        int ref_by = bounds.startY + (result.flat_idx / bounds.width);
        
        // recompute motion vector components for the best candidate position and write to GMEM 
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {
            ref_bx - (int) blockIdx.x, 
            ref_by - (int) blockIdx.y
        };
    }
}

/* computeSADKernel launcher: determines the appropriate SMEM strategy based on available SMEM and launches the kernel with the correct template parameters for SMEM strategy 
and number of threads per block 
    @param prop: cudaDeviceProp structure containing device properties, used to determine available SMEM
    @param smOneFrame: size of SMEM needed to store one frame block 
    @param smTwoFrames: size of SMEM needed to store both current and reference frame blocks
    @param grid: grid dimensions for launching the kernel 
    @param d_curr: pointer to current frame in GMEM
    @param d_ref: pointer to reference frame in GMEM
    @param d_sad: pointer to output SAD values in GMEM
    @param blockSize: size of the frame blocks
    @param width: width of the frames in pixels, used for indexing into GMEM
    @param searchRange: search range in blocks, used for calculating search window bounds
    @param maxCandidates: maximum number of candidate positions in the search window, used for indexing d_sad
    @param threadsKSad: number of threads per block for SAD computation, must be a power of 2
    @param strategy: reference to SmemStrategy variable to store the selected SMEM strategy for logging purposes
    @param smKSad: reference to int variable to store the amount of SMEM used for the kernel launch (use only for logging)

*/
void launchSADKernel(
    cudaDeviceProp prop, int smOneFrame, int smTwoFrames, dim3 grid, 
    const unsigned char* d_curr, const unsigned char* d_ref, int* d_sad, 
    int blockSize, int width, int searchRange, int maxCandidates, int threadsKSad, SmemStrategy & strategy, int&smKSad) 
{

    auto dispatchKernel = [&]<int THREADS>() {
        constexpr size_t cubSmem = sizeof(typename cub::BlockReduce<int, THREADS>::TempStorage);
        const size_t smem_available = prop.sharedMemPerBlock - cubSmem;

        if (smTwoFrames <= smem_available) { //FULL: enough SMEM for both current and reference blocks
            smKSad = smTwoFrames + cubSmem;
            strategy = SmemStrategy::FULL;
            computeSADKernel<SmemStrategy::FULL, THREADS><<<grid, THREADS, smTwoFrames>>>
                (d_curr, d_ref, d_sad, blockSize, width, searchRange, maxCandidates);
        } 
        else if (smOneFrame <= smem_available) { //CURR_ONLY: enough SMEM for current block only
            smKSad = smOneFrame + cubSmem;
            strategy = SmemStrategy::CURR_ONLY;
            computeSADKernel<SmemStrategy::CURR_ONLY, THREADS><<<grid, THREADS, smOneFrame>>>(
                d_curr, d_ref, d_sad, blockSize, width, searchRange, maxCandidates);
        } 
        else { // not enough SMEM for frame blocks
            smKSad = 0;
            strategy = SmemStrategy::NONE;
            computeSADKernel<SmemStrategy::NONE, THREADS><<<grid, THREADS, 0>>>
                (d_curr, d_ref, d_sad, blockSize, width, searchRange, maxCandidates);
        }
    };

    switch (threadsKSad) {
        case 1:    dispatchKernel.template operator()<1>(); break;
        case 2:    dispatchKernel.template operator()<2>(); break;
        case 4:    dispatchKernel.template operator()<4>(); break;
        case 8:    dispatchKernel.template operator()<8>(); break;
        case 16:   dispatchKernel.template operator()<16>(); break;
        case 32:   dispatchKernel.template operator()<32>(); break;
        case 64:   dispatchKernel.template operator()<64>(); break;
        case 128:  dispatchKernel.template operator()<128>(); break;
        case 256:  dispatchKernel.template operator()<256>(); break;
        case 512:  dispatchKernel.template operator()<512>(); break;
        case 1024: dispatchKernel.template operator()<1024>(); break;
        default:   throw std::runtime_error("exceeded number of threads per block: " + std::to_string(threadsKSad));
    }
}
/* host function for full search block matching using 
    @param curr: current frame in grayscale.
    @param ref: reference frame in grayscale.
    @param blockSize: size of the blocks for motion estimation.
    @param searchRange: search range in block,
    @param metrics: reference to SingleRunMetrics struct to store timing metrics for this function
    @return 2D vector of MotionVector objects representing the motion vector for each block in the current frame
*/
vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchRange, SingleRunMetrics& metrics) {
    auto total_time_start = std::chrono::high_resolution_clock::now();
    
    // CUDA timing events
    cudaEvent_t evKSADStart, evKSADStop, evKBestMVStart, evKBestMVStop;
    createCudaEvent(evKSADStart); createCudaEvent(evKSADStop);
    createCudaEvent(evKBestMVStart); createCudaEvent(evKBestMVStop);

    // Device properties
    cudaDeviceProp deviceProp;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Calculate grid dimensions
    ValidationUtils::validateFrameDimensions(curr, ref);
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
    
    
    //Ksad: 3D grid (blockX x blockY x min(maxCandidates, maxGridSizeZ)) and 1D CUDA blocks
    const int KSADgridZ = min(maxCandidates, (int)deviceProp.maxGridSize[2]); // number of candidate positions processed per block in computeSADKernel, limited by max grid size in Z dimension 
    const int limitThreads = min(pixelsPerBlock, deviceProp.maxThreadsPerBlock);
    int threadsKSad = 1;
    while ((threadsKSad << 1) <= limitThreads) threadsKSad <<= 1;
    // Determine SMEM occupied by one frame block and two frame blocks, to decide which SMEM strategy to use in the kernel
    const size_t smOneFrame = pixelsPerBlock * sizeof(unsigned char);
    const size_t smTwoFrames = 2 * smOneFrame;

    //BestMV: 2D grid (blockX x blockY) matching search window dims and 1D CUDA blocks
    const int limit = min(maxCandidates, deviceProp.maxThreadsPerBlock);
    int threadsKBestMV = 1;
    while ((threadsKBestMV << 1) <= limit) threadsKBestMV <<= 1;
                    
    // Allocate device memory for current and reference frames and copy frames to GPU
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    LoggingUtils::printCopyingToGPU("CUDA Uncoalesced Optimized");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);

    
    // Allocate device memory for SAD values and motion vectors
    const size_t sadBytes = (size_t)blocksX * blocksY * maxCandidates * sizeof(int);
    // check if there is enough free VRAM for d_sad buffer
    size_t freeVRAM, totalVRAM;
    gpuErrorCheck(cudaMemGetInfo(&freeVRAM, &totalVRAM));
    if (sadBytes > freeVRAM * 0.9)
        throw std::runtime_error(
            "CUDA Optimized: insufficient VRAM for d_sad buffer: need " +
            std::to_string(sadBytes / (1024*1024)) + " MB, free " +
            std::to_string(freeVRAM / (1024*1024)) + " MB"
        );
    
    int* d_sad = nullptr;
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc(&d_sad, sadBytes));
    gpuErrorCheck(cudaMalloc(&d_mv, (size_t)blocksX * blocksY * sizeof(MotionVector)));
    cout << "CUDA Optimized (Grayscale): Allocating d_sad buffer of size " << sadBytes / (1024.0 * 1024.0) << " MB..." << endl;
    cout << "CUDA Optimized (Grayscale): Allocating d_mv buffer of size " << ((size_t)blocksX * blocksY * sizeof(MotionVector)) / (1024.0 * 1024.0) << " MB..." << endl;

    // Grid and block dimensions
    const dim3 gridKSad(blocksX, blocksY, KSADgridZ);
    const dim3 gridKBestMV(blocksX, blocksY);
    const dim3 blockDimKSad(threadsKSad, 1, 1);
    const dim3 blockDimKBestMV(threadsKBestMV, 1, 1);

    // launch computeSADKernel
    SmemStrategy strategy;
    int smKSad; // actual SMEM used by computeSADKernel;

    recordCudaEvent(evKSADStart);
    launchSADKernel(deviceProp, smOneFrame, smTwoFrames, gridKSad, d_curr, d_ref, d_sad, blockSize, curr.width, searchRange, maxCandidates, threadsKSad, strategy, smKSad);
    recordCudaEvent(evKSADStop);
    cout << "CUDA Optimized (Grayscale): Kernel launched computeSADKernel, synchronizing..." << endl;
    gpuErrorCheck(cudaEventSynchronize(evKSADStop));
    gpuErrorCheck(cudaGetLastError());
    cout << "CUDA Optimized (Grayscale): Selected threads for KSad: " <<  threadsKSad
               << " and block "<< blocksX << "x" << blocksY
               << " (K1=" << smKSad/1024.0 << "KB, K2=" << "no SMEM "<<"\n\n";
    
    cout << "CUDA Optimized (Grayscale): Selected threads for findBestMV: " << threadsKBestMV
         << " and block "<< blocksX << "x" << blocksY << "\n\n";
    
    cout << "CUDA Optimized (Grayscale): d_sad = " << sadBytes / (1024.0 * 1024.0) << " MB\n";

    cout << "CUDA Optimized (Grayscale): Launching computeSADKernel with "
            << blockDimKSad.x << "x" << blockDimKSad.y << "x" << blockDimKSad.z
            << " threads per block (" <<  threadsKSad << " total threads)"<<" SMEM strategy: " << 
                (strategy == SmemStrategy::FULL      ? "FULL"   :
                 strategy == SmemStrategy::CURR_ONLY ? "CURR_ONLY" :
                 "NONE") << endl;


    
    // launch findBestMVKernel
    cout << "CUDA Optimized (Grayscale): Launching findBestMVKernel with " << 
        " blocks " << gridKBestMV.x << "x" << gridKBestMV.y << "x" << gridKBestMV.z
        << " and " << blockDimKBestMV.x << "x" << blockDimKBestMV.y << "x" << blockDimKBestMV.z
        << " threads per block (" << threadsKBestMV << " total threads)..." << endl;
    recordCudaEvent(evKBestMVStart);
    switch (threadsKBestMV) {
        case 1:
            findBestMVKernel<1>  <<<gridKBestMV, 1>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 2:
            findBestMVKernel<2>  <<<gridKBestMV, 2>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 4:
            findBestMVKernel<4>  <<<gridKBestMV, 4>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 8:
            findBestMVKernel<8>  <<<gridKBestMV, 8>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 16:
            findBestMVKernel<16>  <<<gridKBestMV, 16>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 32:
            findBestMVKernel<32>  <<<gridKBestMV, 32>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 64:
            findBestMVKernel<64>  <<<gridKBestMV, 64>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 128:
            findBestMVKernel<128> <<<gridKBestMV, 128>>> (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 256:
            findBestMVKernel<256> <<<gridKBestMV, 256>>> (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 512:
            findBestMVKernel<512> <<<gridKBestMV, 512>>> (d_sad, d_mv, maxCandidates, searchRange);
            break;
        case 1024:
            findBestMVKernel<1024>  <<<gridKBestMV, 1024>>>  (d_sad, d_mv, maxCandidates, searchRange);
            break;
        default:
            throw std::runtime_error("findBestMVKernel: threadsKBestMV not supported: "
                                     + std::to_string(threadsKBestMV));
    }
    recordCudaEvent(evKBestMVStop);
    cout << "CUDA Optimized (Grayscale): Kernel best MV launched, synchronizing..." << endl;
    gpuErrorCheck(cudaGetLastError());
    
    gpuErrorCheck(cudaDeviceSynchronize()); 

    // Timing report for individual kernels
    const float msKSAD = elapsedCudaTime(evKSADStart, evKSADStop);
    const float msKBestMV = elapsedCudaTime(evKBestMVStart, evKBestMVStop);

    cout << "CUDA Optimized (Grayscale): Timing (Kernels only):" << endl;
    CudaTimingUtils::printKernelTiming("computeSADKernel", msKSAD);
    CudaTimingUtils::printKernelTiming("findBestMVKernel", msKBestMV);
    
    // Copy results back to host
    vector<MotionVector> h_mv_flat(blocksX * blocksY);
    GPUMemoryUtils::copyMotionVectorsFromGPU(h_mv_flat, d_mv, blocksX, blocksY);

    // Cleanup GPU
    LoggingUtils::printCleanupGPU("CUDA Optimized");
    GPUMemoryUtils::freeMemory(d_curr, d_ref, d_sad, d_mv);

    // Stop total timer after cleanup
    auto total_time_stop = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_time_stop - total_time_start).count();
    metrics.total_ms = total_time;
    metrics.gpu_kernel1_ms = msKSAD;
    metrics.gpu_kernel2_ms = msKBestMV;

    destroyCudaEvent(evKSADStart); destroyCudaEvent(evKSADStop);
    destroyCudaEvent(evKBestMVStart); destroyCudaEvent(evKBestMVStop);

    LoggingUtils::printProcessingComplete("CUDA Optimized");
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);

    return result;
}