#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "sad_utils.h"
#include "utils.h"
#include <chrono>

using namespace std;

// logarithmic search: each thread computes SAD between 9 pair (curr, ref), iterating until the search distance is reduced to 1 block  
constexpr int MAXCANDIDATES_AT_DISTANCE= 9;

// structure to hold SAD and candidate block index for comparison
// memory allignment: 4+2+2 = 8 bytes, so each CandidateSad can be stored in one 64-bit word, allowing for coalesced memory access
struct CandidateSad {
    int sad; // best SAD found so far 
    short dx;  // shift in x direction from the current block to the candidate block
    short dy;  // shift in y direction from the current block to the candidate block

};

// SMEM strategy, determined at compile time based on available shared memory
enum class SmemStrategy {
    FULL,       // curr block + ref block in SMEM  (small blockSize)
    CURR_ONLY,  // only curr block in SMEM
    NONE        // no SMEM for pixel data
};

struct CandidateSadOp {
    __device__ __forceinline__
    CandidateSad operator()(const CandidateSad& a, const CandidateSad& b) const {
        // select candidate with lower SAD
        if (a.sad != b.sad) return a.sad < b.sad ? a : b;
        
        // tie-break: prefer shorter motion vector
        int distA = (a.dx * a.dx) + (a.dy * a.dy);
        int distB = (b.dx * b.dx) + (b.dy * b.dy);
        if (distA != distB) return distA < distB ? a : b;
        
        // deterministic tie-break: if SAD and distance are the same, prefer candidate with smaller dx, then smaller dy 
        if (a.dy != b.dy) return a.dy < b.dy ? a : b;
        return a.dx < b.dx ? a : b;
    }
};

// functor to compute segment offsets
struct SegmentOffsetOp {
    int segment_size;

   __host__ __device__ __forceinline__
    SegmentOffsetOp(int segment_size) : segment_size(segment_size) {}

    __host__ __device__ __forceinline__
    int operator()(int segment_idx) const {
        return segment_idx * segment_size;
    }
};

// initialize best candidate for each block to be itself (zero motion vector) before starting logarithmic search iterations
__global__ void initBestCandidateKernel(CandidateSad* d_best_candidates, int blocksX, int blocksY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // frame block index in X dimension
    int y = blockIdx.y * blockDim.y + threadIdx.y; // frame block index in Y dimension
    if (x < blocksX && y < blocksY) { // check bounds
        int idx = y * blocksX + x;
        d_best_candidates[idx] = {INT_MAX, 0,0}; // initialize best candidate to be the block itself
    }
}

// Computes SAD  between the current block and each candidate position in the search window.
template<SmemStrategy STRATEGY, int KSAD_THREADS> __global__ void computeSADKernel(const unsigned char* __restrict__ d_curr, const unsigned char* __restrict__ d_ref,
                                 CandidateSad* d_sad_candidates, // [blocksX * blockY* MAXCANDIDATES_AT_DISTANCE] // SAD values for each candidate position for each block computed in this kernel 
                                 const CandidateSad* __restrict__ d_best_candidates, //[blocksX * blockY] // current best candidate for each block at previous search distance 
                                 int blockSize, size_t frame_pitch_bytes, int blocksX, int blocksY, int searchDistance) {   
    
    // Calculate search window using helper function
    // bounds can be load in SMEM, but is very small so calculate it in each thread to avoid extra shared memory usage and synchronization
    // CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);
    
    // linear block index for the current block in the frame, used for indexing into d_sad_candidates and d_best_candidates
    const int frameBlockIdx = blockIdx.y * gridDim.x + blockIdx.x; 
    
    // top-left corner of the current block in the current frame in pixels
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    // top-left corner coordinates block of the search window area center
    const int cx =  blockIdx.x + d_best_candidates[frameBlockIdx].dx; // x coordinate of the center of the search window in frame blocks
    const int cy =  blockIdx.y + d_best_candidates[frameBlockIdx].dy; // y coordinate of the center of the search window in frame blocks
    
    // coordinate of the top-left corner of candidate block in the reference frame for this blockIdx.z, shifted by searchDistance from the center block
    const int directionX = (blockIdx.z % 3) - 1; // direction of the candidate block in the search window relative to the center block in X dimension, can be -1, 0, or 1
    const int directionY = (blockIdx.z / 3) - 1; // direction of the candidate block in the search window relative to the center block in Y dimension, can be -1, 0, or 1
    const int cand_bx = cx + (directionX * searchDistance);
    const int cand_by = cy + (directionY * searchDistance);

    const short cand_dx = cand_bx - blockIdx.x; // motion vector displacement in x from the current block to the candidate block
    const short cand_dy = cand_by - blockIdx.y; // motion vector displacement in y from the current block to the candidate block

    // check if the candidate block is within the reference frame boundaries
    if (cand_bx < 0 || cand_by < 0 || cand_bx >= blocksX || cand_by >= blocksY) {
        if (threadIdx.x == 0) {
            d_sad_candidates[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + blockIdx.z] = {INT_MAX, 0, 0}; // if out of bounds, set SAD to max value so it won't be selected as best candidate
        } 
        return;
    }
    const int ref_x = cand_bx * blockSize;  // candidate block's top-left corner in the reference frame in pixels
    const int ref_y = cand_by * blockSize;  // candidate block's top-left corner in the reference frame in pixels
     

    // number of pixels per block and per thread for SAD computation
    const int pixelsPerBlock = blockSize * blockSize; // number of pixels in a frame block
    const int pixelsPerThread = (pixelsPerBlock + KSAD_THREADS - 1) / KSAD_THREADS; // divide pixels among threads, rounding up
    
     // CUB BlockReduce to sum partial SADs across the KSAD_THREADS threads ──
    using BlockReduce = cub::BlockReduce<int, KSAD_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    // load shared memory 
    extern __shared__ unsigned char smemSad[];
    unsigned char* s_curr = (STRATEGY != SmemStrategy::NONE) ? smemSad : nullptr; // current block is stored in SMEM for FULL and CURR_ONLY strategies
    unsigned char* s_ref = (STRATEGY == SmemStrategy::FULL) ? smemSad + pixelsPerBlock : nullptr; // reference block is stored in SMEM only for FULL strategy

    // loading SMEM based on strategy
    const int fetchesPerBlock = pixelsPerBlock / 4; // number of 32-bit fetches needed to load one block (assuming blockSize is a multiple of 4)
    if constexpr (STRATEGY == SmemStrategy::FULL) { // FULL strategy, load reference block into SMEM
        uint32_t* s_ref_vec = reinterpret_cast<uint32_t*>(s_ref);
        for (int i = threadIdx.x; i < fetchesPerBlock; i += KSAD_THREADS) {
            int px = (i * 4) % blockSize;
            int py = (i * 4) / blockSize;
            s_ref_vec[i] = *reinterpret_cast<const uint32_t*>(d_ref + (ref_y + py) * frame_pitch_bytes + (ref_x + px)); 
        }
    }
    if constexpr (STRATEGY != SmemStrategy::NONE) { // FULL or CURR_ONLY strategy, load current block into SMEM
        uint32_t* s_curr_vec     = reinterpret_cast<uint32_t*>(s_curr); // vectorized pointer for loading 4 pixels at a time
        for (int i = threadIdx.x; i < fetchesPerBlock; i += KSAD_THREADS) { // parallelize loading of current block into SMEM among threads in the block
            int px = (i * 4) % blockSize;
            int py = (i * 4) / blockSize;
            s_curr_vec[i] = *reinterpret_cast<const uint32_t*>(d_curr + (y + py) * frame_pitch_bytes + (x + px));
        }
        __syncthreads(); // barrier only to ensure all threads have loaded their data, so strategy with FULL or CURR_ONLY 
    }
    
    // each thread computes partial SAD for a subset of candidate positions in the search window, partial SAD values are reduced using CUB BlockReduce to get total SAD for each candidate  
    
    // partial SAD computation for each thread
    int partial_sad = 0; 
    const int startPixel = threadIdx.x * pixelsPerThread; // starting pixel index for this thread to compute partial SAD (0 to pixelsPerBlock-1)
    const int endPixel = min(startPixel + pixelsPerThread, pixelsPerBlock);  // ending pixel index (exclusive) for this thread to compute partial SAD, ensuring don't go out of bounds
    for (int i = startPixel; i < endPixel; ++i) {
        int py = i / blockSize, px = i % blockSize;
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
     const int total_sad = BlockReduce(reduce_storage).Sum(partial_sad);
            
    if (threadIdx.x == 0) {
        d_sad_candidates[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + blockIdx.z] = {total_sad, cand_dx, cand_dy}; 
    }    
}



//launch computeSADKernel and findBestMVKernel with appropriate template parameters based on SMEM strategy and number of threads for reduction
void launchSADKernel(
    cudaDeviceProp prop, int smOneFrame, int smTwoFrames, dim3 grid, int threadsKSad, SmemStrategy & strategy, int&smKSad,
    const unsigned char* d_curr, const unsigned char* d_ref, size_t frame_pitch_bytes, CandidateSad* d_sad_candidates, const CandidateSad* d_best_candidates, 
    int blockSize, int blockX, int blockY, int SearchDistance) 
{

    auto dispatchKernel = [&]<int THREADS>() {
        constexpr size_t cubSmem = sizeof(typename cub::BlockReduce<int, THREADS>::TempStorage);
        const size_t smem_available = prop.sharedMemPerBlock - cubSmem;

        if (smTwoFrames <= smem_available) { //FULL: enough SMEM for both current and reference blocks
            smKSad = smTwoFrames + cubSmem;
            strategy = SmemStrategy::FULL;
            computeSADKernel<SmemStrategy::FULL, THREADS><<<grid, THREADS, smTwoFrames>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, frame_pitch_bytes, blockX, blockY, SearchDistance);
        } 
        else if (smOneFrame <= smem_available) { //CURR_ONLY: enough SMEM for current block only
            smKSad = smOneFrame + cubSmem;
            strategy = SmemStrategy::CURR_ONLY;
            computeSADKernel<SmemStrategy::CURR_ONLY, THREADS><<<grid, THREADS, smOneFrame>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, frame_pitch_bytes, blockX, blockY, SearchDistance);
        } 
        else { // not enough SMEM for frame blocks
            smKSad = 0;
            strategy = SmemStrategy::NONE;
            computeSADKernel<SmemStrategy::FULL, THREADS><<<grid, THREADS, smTwoFrames>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, frame_pitch_bytes, blockX, blockY, SearchDistance);
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
vector<vector<MotionVector>> logarithmicSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchDistance, SingleRunMetrics& metrics) {
    auto total_time_start = std::chrono::high_resolution_clock::now();
    
    // CUDA timing events
    cudaEvent_t evKSADStart, evKSADStop;
    createCudaEvent(evKSADStart);
    createCudaEvent(evKSADStop);

    // Device properties
    cudaDeviceProp deviceProp;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    ValidationUtils::validateFrameDimensions(curr, ref);
    
    // Calculate frame grid dimensions
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    const int pixelsPerBlock = blockSize * blockSize;
    const int frameBlocks = blocksX * blocksY;
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Optimized", curr.width, curr.height, blockSize, blocksX, blocksY);
    cout << "CUDA Optimized Logarithmic (Grayscale): Search strategy: Logarithmic search (distance="
         << searchDistance << " blocks)" << endl;
    
    // Calculate search window dimensions
    // FIXME, no search window for logarithmic search
    /*const int searchWindowBlocksX = (searchRange > 0) ? min(blocksX, searchRange * 2 + 1) : blocksX;
    const int searchWindowBlocksY = (searchRange > 0) ? min(blocksY, searchRange * 2 + 1) : blocksY; */
    
    cout << "CUDA Optimized (Grayscale): GPU: " << deviceProp.name << "\n"
         << "  maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << "\n"
         << "  sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " bytes\n";
                     
    // Allocate device memory for current and reference frames
    size_t frame_pitch_bytes;
    // with default load profiler say : "The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to L2 is a 128 byte cache line. 
    // That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only accesses an average of 1.0 sectors out of the possible 4 sectors per cache line. "
    // to improve L2 cache throughput and coalesced access, the kernelcan performs vectorized memory loads using 32-bit pointers (uint32_t), fetching * 4 pixels (4 bytes) per thread simultaneously.
    // 32-bit vectorized memory accesses strictly require * 4-byte aligned addresses. If the image width is not a perfect multiple of 4, * a standard linear cudaMalloc would cause subsequent rows to be unaligned
    // cudaMallocPitch prevents this by automatically appending padding bytes * to the end of each row.
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;

    gpuErrorCheck(cudaMallocPitch((void**)&d_curr, &frame_pitch_bytes, curr.width * sizeof(unsigned char), curr.height));
    gpuErrorCheck(cudaMallocPitch((void**)&d_ref, &frame_pitch_bytes, ref.width * sizeof(unsigned char), ref.height));

    CandidateSad* d_sad_candidates = nullptr; // device memory for SAD values, allocated as blocksX * blocksY * MAXCANDIDATES_AT_DISTANCE to store SAD for each candidate position for each block
    CandidateSad* d_best_candidates = nullptr; // device memory to pass current iteration's best matches, will contain the final best candidates and motion vectors after last iteration
    gpuErrorCheck(cudaMalloc(&d_sad_candidates, (size_t) frameBlocks * MAXCANDIDATES_AT_DISTANCE * sizeof(CandidateSad)));
    gpuErrorCheck(cudaMalloc(&d_best_candidates, (size_t) frameBlocks * sizeof(CandidateSad)));
    
    // copy current and reference frames to device 
    LoggingUtils::printCopyingToGPU("CUDA Optimized");
    size_t widthBytes = curr.width * sizeof(unsigned char);
    gpuErrorCheck(cudaMemcpy2D(d_curr, frame_pitch_bytes, curr.data.data(), widthBytes, widthBytes, curr.height, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy2D(d_ref, frame_pitch_bytes, ref.data.data(), widthBytes, widthBytes, ref.height, cudaMemcpyHostToDevice));
    
    // iterator for segment offsets for CUB segmented reduction, each segment corresponds to SAD values for one candidate position for all blocks
    cub::CountingInputIterator<int> counting_iter(0);
    cub::TransformInputIterator<int, SegmentOffsetOp, cub::CountingInputIterator<int>> 
        d_offsets(counting_iter, SegmentOffsetOp(MAXCANDIDATES_AT_DISTANCE));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, 
                                       d_sad_candidates, d_best_candidates, 
                                       frameBlocks, d_offsets, d_offsets + 1, 
                                       CandidateSadOp(), CandidateSad{INT_MAX, 0, 0});
    
    // determine temporary storage requirements for segmented reduction of SAD candidates to find best candidate for each block
                                       
    gpuErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // initBestCandidateKernel block and grid dimensions
    const dim3 blockInitK(16, 16);
    dim3 gridInitK ((blocksX + blockInitK.x - 1) / blockInitK.x, (blocksY + blockInitK.y - 1) / blockInitK.y); // 

    //Ksad: 3D grid (blockX x blockY x min(maxCandidates, maxGridSizeZ)) and 1D CUDA blocks
    const int KSADgridZ = min(MAXCANDIDATES_AT_DISTANCE, (int)deviceProp.maxGridSize[2]); // number of candidate positions processed per block in computeSADKernel, limited by max grid size in Z dimension 
    int limitThreads = min(pixelsPerBlock, deviceProp.maxThreadsPerBlock);
    int threadsKSad = 1;
    while ((threadsKSad << 1) <= limitThreads) threadsKSad <<= 1;
    // Determine SMEM occupied by one frame block and two frame blocks, to decide which SMEM strategy to use in the kernel
    const size_t smOneFrame = pixelsPerBlock * sizeof(unsigned char);
    const size_t smTwoFrames = 2 * smOneFrame;
    
    // grid and block dimensions for computeSADKernel
    const dim3 gridKSad(blocksX, blocksY, KSADgridZ);
    const dim3 blockDimKSad(threadsKSad, 1, 1);

    // SMEM strategy and smem used for computeSADKernel
    SmemStrategy strategy;
    int smKSad;
    
    // initialize best candidates to be the blocks themselves (zero motion vector) before starting logarithmic search iterations
    initBestCandidateKernel<<<gridInitK, blockInitK>>>(d_best_candidates, blocksX, blocksY);
    gpuErrorCheck(cudaDeviceSynchronize());

    recordCudaEvent(evKSADStart);

    for (int currentDistance = searchDistance; currentDistance > 0; currentDistance >>= 1) {
        launchSADKernel(
            deviceProp, smOneFrame, smTwoFrames, gridKSad, threadsKSad, strategy, smKSad,
            d_curr, d_ref, frame_pitch_bytes, d_sad_candidates, d_best_candidates, 
            blockSize, blocksX, blocksY, currentDistance);
        cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, 
                                           d_sad_candidates, d_best_candidates, 
                                           frameBlocks, d_offsets, d_offsets + 1, 
                                           CandidateSadOp(), CandidateSad{INT_MAX, 0, 0});
    }
    
    recordCudaEvent(evKSADStop);

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaGetLastError());
    cout << "CUDA Optimized (Grayscale): Kernel launched, synchronizing..." << endl;
    gpuErrorCheck(cudaDeviceSynchronize());

    // Timing report for individual kernels
    const float msKSAD = elapsedCudaTime(evKSADStart, evKSADStop);

    cout << "CUDA Optimized (Grayscale): Timing (Kernels only):" << endl;
    CudaTimingUtils::printKernelTiming("computeSADKernel", msKSAD);

    // Copy results back to host
    vector<MotionVector> h_mv_flat(frameBlocks);

    // Convert flat array to 2D vector
    vector<CandidateSad> h_best_flat(frameBlocks);
    gpuErrorCheck(cudaMemcpy(h_best_flat.data(), d_best_candidates, 
                             frameBlocks * sizeof(CandidateSad), cudaMemcpyDeviceToHost));

    // Cleanup GPU
    LoggingUtils::printCleanupGPU("CUDA Optimized");
    gpuErrorCheck(cudaFree(d_curr)); 
    gpuErrorCheck(cudaFree(d_ref));
    gpuErrorCheck(cudaFree(d_sad_candidates)); 
    gpuErrorCheck(cudaFree(d_best_candidates));
    gpuErrorCheck(cudaFree(d_temp_storage));

    for(int i = 0; i < frameBlocks; ++i) {
        h_mv_flat[i] = { h_best_flat[i].dx, h_best_flat[i].dy };
    }
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);

    // Stop total timer after cleanup
    auto total_time_stop = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_time_stop - total_time_start).count();
    metrics.total_ms = total_time;
    metrics.gpu_kernel1_ms = msKSAD;

    destroyCudaEvent(evKSADStart);
    destroyCudaEvent(evKSADStop);

    LoggingUtils::printProcessingComplete("CUDA Optimized");
    return result;
}