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

// Computes SAD  between the current block and each candidate position in the search window.
template<SmemStrategy STRATEGY, int KSAD_THREADS> __global__ void computeSADKernel(const unsigned char* __restrict__ d_curr, const unsigned char* __restrict__ d_ref,
                                 int* d_sad, int blockSize, size_t frame_pitch_bytes, int searchRange, int maxCandidates) {   
    
    // Calculate search window using helper function
    // bounds can be load in SMEM, but is very small so calculate it in each thread to avoid extra shared memory usage and synchronization
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    
    // number of pixels per block and per thread for SAD computation
    const int pixelsPerBlock = blockSize * blockSize; // number of pixels in a frame block
    const int pixelsPerThread = (pixelsPerBlock + KSAD_THREADS - 1) / KSAD_THREADS; // divide pixels among threads, rounding up
    
    // top-left corner of the current block in the current frame
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    // TempStorage for block reduction of SAD values, allocated in SMEM but does not count towards smKSAD
    // Using CUB BlockReduce for efficient reduction of partial SAD values computed by threads
    using BlockReduce = cub::BlockReduce<int, KSAD_THREADS>;
    __shared__ typename BlockReduce::TempStorage partial_sad_reduce_storage;

    // load shared memory 
    extern __shared__ unsigned char smemSad[];
    unsigned char* s_curr = (STRATEGY != SmemStrategy::NONE) ? smemSad : nullptr; // current block is stored in SMEM for FULL and CURR_ONLY strategies
    unsigned char* s_ref = (STRATEGY == SmemStrategy::FULL) ? smemSad + pixelsPerBlock : nullptr; // reference block is stored in SMEM only for FULL strategy

    // loading SMEM based on strategy
    if constexpr (STRATEGY == SmemStrategy::FULL || STRATEGY == SmemStrategy::CURR_ONLY) {
        uint32_t* s_curr_vec     = reinterpret_cast<uint32_t*>(s_curr); // vectorized pointer for loading 4 pixels at a time
        const int fetchesPerBlock = pixelsPerBlock / 4; // number of 32-bit fetches needed to load one block (assuming blockSize is a multiple of 4)

        for (int i = threadIdx.x; i < fetchesPerBlock; i += KSAD_THREADS) { // parallelize loading of current block into SMEM among threads in the block
            int px = (i * 4) % blockSize;
            int py = (i * 4) / blockSize;
            s_curr_vec[i] = *reinterpret_cast<const uint32_t*>(
                d_curr + (y + py) * frame_pitch_bytes + (x + px));
        }
        __syncthreads();
    }

    // each thread computes partial SAD for a subset of candidate positions in the search window, partial SAD values are reduced using CUB BlockReduce to get total SAD for each candidate  
    for (int flat_idx = blockIdx.z; flat_idx < bounds.totalPositions; flat_idx += gridDim.z) {
        // ref_x and ref_y are the top-left corner of the candidate block in the reference frame corresponding to this blockIdx.zq
        // ref_x ref_y can be load in SMEM, but is only 2 ints so calculate in each thread to avoid extra shared memory usage and synchronization 
        const int ref_x = (bounds.startX + (flat_idx % bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels   
        const int ref_y = (bounds.startY + (flat_idx / bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels
        /// load ref in SMEM for FULL strategy
        if constexpr (STRATEGY == SmemStrategy::FULL) { 
            uint32_t* s_ref_vec = reinterpret_cast<uint32_t*>(s_ref);
            const int fetchesPerBlock = pixelsPerBlock / 4;
            for (int i = threadIdx.x; i < fetchesPerBlock; i += KSAD_THREADS) {
                int px = (i * 4) % blockSize;
                int py = (i * 4) / blockSize;
                s_ref_vec[i] = *reinterpret_cast<const uint32_t*>(
                    d_ref + (ref_y + py) * frame_pitch_bytes + (ref_x + px));
            }
            __syncthreads();
        }

        // partial SAD computation for each thread based 
        int partial_sad = 0; 
        const int startPixel = threadIdx.x * pixelsPerThread; // starting pixel index for this thread to compute partial SAD (0 to pixelsPerBlock-1)
        const int endPixel   = min(startPixel + pixelsPerThread, pixelsPerBlock);  // ending pixel index (exclusive) for this thread to compute partial SAD, ensuring don't go out of bounds

        for (int i = startPixel; i < endPixel; ++i) {
            int py = i / blockSize;
            int px = i % blockSize;

            unsigned char c, r;

            if constexpr (STRATEGY == SmemStrategy::FULL) {
                c = s_curr[i];
                r = s_ref[i];
            } else if constexpr (STRATEGY == SmemStrategy::CURR_ONLY) {
                c = s_curr[i];
                r = d_ref[(ref_y + py) * frame_pitch_bytes + (ref_x + px)]; // reference block is read from global memory for CURR_ONLY strategy, 
            } else {
                // no SMEM, both current and reference blocks are read from global memory
                c = d_curr[(y + py) * frame_pitch_bytes + (x + px)];
                r = d_ref [(ref_y + py) * frame_pitch_bytes + (ref_x + px)];
            }
            partial_sad += abs((int)c - (int)r);
        }
        // reduction1: block reduction of partial SAD values to get total SAD for this candidate position, using CUB BlockReduce 
        const int total_sad = BlockReduce(partial_sad_reduce_storage).Sum(partial_sad);

        if (threadIdx.x == 0) { // thread 0 writes the final SAD for this candidate position to global memory
            d_sad[(blockIdx.y* gridDim.x + blockIdx.x)* maxCandidates + flat_idx] = total_sad; 
        }
        __syncthreads(); // ensure all threads have written their SAD value before next iteration which may overwrite SMEM for the next candidate block
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
        if (b.sad < a.sad) return b;
        if (b.sad > a.sad) return a;
        // tie-break: prefer shorter motion vector
        return (getDistSq(b.flat_idx) < getDistSq(a.flat_idx)) ? b : a;
    }
};

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
            ref_bx - (int)blockIdx.x, 
            ref_by - (int)blockIdx.y
        };
    }
}

//launch computeSADKernel and findBestMVKernel with appropriate template parameters based on SMEM strategy and number of threads for reduction
void launchSADKernel(
    cudaDeviceProp prop, int smOneFrame, int smTwoFrames, dim3 grid, 
    const unsigned char* d_curr, const unsigned char* d_ref, int* d_sad, 
    int blockSize, size_t frame_pitch_bytes, int searchRange, int maxCandidates, int threadsKSad, SmemStrategy & strategy, int&smKSad) 
{

    auto dispatchKernel = [&]<int THREADS>() {
        constexpr size_t cubSmem = sizeof(typename cub::BlockReduce<int, THREADS>::TempStorage);
        const size_t smem_available = prop.sharedMemPerBlock - cubSmem;

        if (smTwoFrames <= smem_available) { //FULL: enough SMEM for both current and reference blocks
            smKSad = smTwoFrames + cubSmem;
            strategy = SmemStrategy::FULL;
            computeSADKernel<SmemStrategy::FULL, THREADS><<<grid, THREADS, smTwoFrames>>>(
                d_curr, d_ref, d_sad, blockSize, frame_pitch_bytes, searchRange, maxCandidates);
        } 
        else if (smOneFrame <= smem_available) { //CURR_ONLY: enough SMEM for current block only
            smKSad = smOneFrame + cubSmem;
            strategy = SmemStrategy::CURR_ONLY;
            computeSADKernel<SmemStrategy::CURR_ONLY, THREADS><<<grid, THREADS, smOneFrame>>>(
                d_curr, d_ref, d_sad, blockSize, frame_pitch_bytes, searchRange, maxCandidates);
        } 
        else { // not enough SMEM for frame blocks
            smKSad = 0;
            strategy = SmemStrategy::NONE;
            computeSADKernel<SmemStrategy::NONE, THREADS><<<grid, THREADS, 0>>>(
                d_curr, d_ref, d_sad, blockSize, frame_pitch_bytes, searchRange, maxCandidates);
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
// HOST CODE
vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchRange, SingleRunMetrics& metrics) {
    auto total_time_start = std::chrono::high_resolution_clock::now();
    
    // CUDA timing events
    cudaEvent_t evKSADStart, evKSADStop, evKBestMVStart, evKBestMVStop;
    createCudaEvent(evKSADStart);
    createCudaEvent(evKSADStop);
    createCudaEvent(evKBestMVStart);
    createCudaEvent(evKBestMVStop);

    // Device properties
    cudaDeviceProp deviceProp;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
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
    
    cout << "CUDA Optimized (Grayscale): GPU: " << deviceProp.name << "\n"
         << "  maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << "\n"
         << "  sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " bytes\n";
    
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
                    
    // Allocate device memory for current and reference frames
    const size_t sadBytes = (size_t)blocksX * blocksY * maxCandidates * sizeof(int);
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    //GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    // with default load profiler say : "The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to L2 is a 128 byte cache line. 
    // That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only accesses an average of 1.0 sectors out of the possible 4 sectors per cache line. "
    // to improve L2 cache throughput and coalesced access, the kernelcan performs vectorized memory loads using 32-bit pointers (uint32_t), fetching * 4 pixels (4 bytes) per thread simultaneously.
    // 32-bit vectorized memory accesses strictly require * 4-byte aligned addresses. If the image width is not a perfect multiple of 4, * a standard linear cudaMalloc would cause subsequent rows to be unaligned
    // cudaMallocPitch prevents this by automatically appending padding bytes * to the end of each row.
    
    size_t frame_pitch_bytes; // actual allocated row size in bytes (including padding), returned by cudaMallocPitch, used for indexing into d_curr and d_ref in the kernel 

    gpuErrorCheck(cudaMallocPitch((void**)&d_curr, &frame_pitch_bytes, curr.width * sizeof(unsigned char), curr.height));
    gpuErrorCheck(cudaMallocPitch((void**)&d_ref, &frame_pitch_bytes, ref.width * sizeof(unsigned char), ref.height));

    int* d_sad = nullptr;
    MotionVector* d_mv = nullptr;
    gpuErrorCheck(cudaMalloc(&d_sad, sadBytes));
    gpuErrorCheck(cudaMalloc(&d_mv, (size_t)blocksX * blocksY * sizeof(MotionVector)));

    
    //GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);
    LoggingUtils::printCopyingToGPU("CUDA Optimized");
    size_t widthBytes = curr.width * sizeof(unsigned char);
    gpuErrorCheck(cudaMemcpy2D(d_curr, frame_pitch_bytes, curr.data.data(), widthBytes, widthBytes, curr.height, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy2D(d_ref, frame_pitch_bytes, ref.data.data(), widthBytes, widthBytes, ref.height, cudaMemcpyHostToDevice));
    

    // Grid and block dimensions
    const dim3 gridKSad(blocksX, blocksY, KSADgridZ);
    const dim3 gridKBestMV(blocksX, blocksY);
    const dim3 blockDimKSad(threadsKSad, 1, 1);
    const dim3 blockDimKBestMV(threadsKBestMV, 1, 1);

    // launch computeSADKernel
    SmemStrategy strategy;
    int smKSad; // actual SMEM used by computeSADKernel;

    recordCudaEvent(evKSADStart);
    launchSADKernel(deviceProp, smOneFrame, smTwoFrames, gridKSad, d_curr, d_ref, d_sad, blockSize, frame_pitch_bytes, searchRange, maxCandidates, threadsKSad, strategy, smKSad);
    recordCudaEvent(evKSADStop); 

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
    
    gpuErrorCheck(cudaGetLastError());
    cout << "CUDA Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    // launch findBestMVKernel
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
    cout << "CUDA Optimized (Grayscale): Launching findBestMVKernel with "
        << blockDimKBestMV.x << "x" << blockDimKBestMV.y << "x" << blockDimKBestMV.z
        << " threads per block (" << threadsKBestMV << " total threads)..." << endl;
    gpuErrorCheck(cudaGetLastError());
    cout << "CUDA Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
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

    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);

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
    return result;
}