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

// SMEM strategy, determine at compile time, base on block size, where to store current and reference blocks
enum class SmemStrategy {
    FULL,       // curr block + ref + CUB SMEM
    CURR_ONLY,  // only curr block +  CUB SMEM (same curr block is read from all cuda blocks along Z)
    NONE        // CUB SMEM
};



// functor to compute segment offsets for CUB segmented reduction, each segment corresponds to SAD values for one candidate position for all blocks
struct SegmentOffsetOp {
    int segment_size;

   __host__ __device__ __forceinline__
    SegmentOffsetOp(int segment_size) : segment_size(segment_size) {}

    __host__ __device__ __forceinline__
    int operator()(int segment_idx) const {
        return segment_idx * segment_size;
    }
};


/* CUDA kernel to compute SAD between current block and candidate blocks in the reference frame for each block in the current frame, for a given search distance 
 without specify searchDistance, is computed as max(blocksX, blocksY)-1, which is the maximum distance to cover the entire frame. 
    @tparam STRATEGY: SMEM strategy to use for this kernel launch (FULL, CURR_ONLY, or NONE)
    @tparam KSAD_THREADS: number of threads per block for this kernel launch
    @param d_curr: device pointer to current frame
    @param d_ref: device pointer to reference frame
    @param d_sad_candidates: device pointer to SAD values for each candidate position for each block, computed in this kernel 
    @param d_best_candidates: device pointer to current best candidate for each block at previous search distance 
    @param blockSize: size of the block (in pixels)
    @param width: width of the frame (in pixels)
    @param blocksX: number of blocks in the X dimension of the frame grid
    @param blocksY: number of blocks in the Y dimension of the frame grid
    @param searchDistance: current search distance (in blocks)
*/
template<SmemStrategy STRATEGY, int KSAD_THREADS> __global__ void computeSADKernel(const unsigned char* __restrict__ d_curr, const unsigned char* __restrict__ d_ref,
                                 CandidateSad* d_sad_candidates, // [blocksX * blockY* MAXCANDIDATES_AT_DISTANCE] // SAD values for each candidate position for each block computed in this kernel 
                                 const CandidateSad*  d_best_candidates, //[blocksX * blockY] // current best candidate for each block at previous search distance 
                                 int blockSize, int width, int blocksX, int blocksY, int searchDistance) {   
    
    
    // linear block index for the current block in the frame, used for indexing into d_sad_candidates and d_best_candidates
    const int frameBlockIdx = blockIdx.y * gridDim.x + blockIdx.x; 
    
    // top-left corner of the current block in the current frame in pixels
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    // top-left corner coordinates block of the search window area center in block coordinates
    const int cx =  blockIdx.x + d_best_candidates[frameBlockIdx].dx;
    const int cy =  blockIdx.y + d_best_candidates[frameBlockIdx].dy; 
    // coordinate of the candidate block (in block coordinates) in the reference frame for this blockIdx.z, shifted by searchDistance from the center block
    const int directionX = (blockIdx.z % 3) - 1; // direction of the candidate block in the search window relative to the center block in X dimension, can be -1, 0, or 1
    const int directionY = (blockIdx.z / 3) - 1; // direction of the candidate block in the search window relative to the center block in Y dimension, can be -1, 0, or 1
    const int cand_bx = cx + (directionX * searchDistance);
    const int cand_by = cy + (directionY * searchDistance);
    
    // motion vector displacement from the current block to the candidate block
    const int cand_dx = cand_bx - blockIdx.x;
    const int cand_dy = cand_by - blockIdx.y;

    // check if the candidate block is within the reference frame boundaries
    if (cand_bx < 0 || cand_by < 0 || cand_bx >= blocksX || cand_by >= blocksY) {
        if (threadIdx.x == 0) {
            d_sad_candidates[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + blockIdx.z] = {INT_MAX, 0, 0}; // if out of bounds, set SAD to max value so it won't be selected as best candidate
        } 
        return;
    }
    // candidate block's top-left corner in the reference frame in pixels
    const int ref_x = cand_bx * blockSize;
    const int ref_y = cand_by * blockSize; 
     

    // number of pixels per block and per thread for SAD computation
    const int pixelsPerBlock = blockSize * blockSize;
    const int pixelsPerThread = (pixelsPerBlock + KSAD_THREADS - 1) / KSAD_THREADS;
    
     // CUB BlockReduce to sum partial SADs across the KSAD_THREADS threads ──
    using BlockReduce = cub::BlockReduce<int, KSAD_THREADS>;
    __shared__ typename BlockReduce::TempStorage partial_sad_reduce_storage;

    // SMEM 
    extern __shared__ unsigned char smemSad[];
    unsigned char* s_curr = (STRATEGY != SmemStrategy::NONE) ? smemSad : nullptr; // current block is stored in SMEM for FULL and CURR_ONLY strategies
    unsigned char* s_ref = (STRATEGY == SmemStrategy::FULL) ? smemSad + pixelsPerBlock : nullptr; // reference block is stored in SMEM only for FULL strategy

    // loading SMEM based on strategy
    if constexpr (STRATEGY == SmemStrategy::FULL) { // FULL strategy, load reference block into SMEM
        for (int i = threadIdx.x; i < pixelsPerBlock; i += KSAD_THREADS) {
                s_ref[i] = d_ref[(ref_y + (i / blockSize)) * width + (ref_x + (i % blockSize))];
        }
    }
    if constexpr (STRATEGY != SmemStrategy::NONE) { // FULL or CURR_ONLY strategy, load current block into SMEM
        for (int i = threadIdx.x; i < pixelsPerBlock; i += KSAD_THREADS) {
            s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
        }
        __syncthreads(); // barrier only to ensure all threads have loaded their data, so strategy with FULL or CURR_ONLY 
    }
    
    // each thread computes partial SAD for a subset of candidate positions in the search window, partial SAD values are reduced using CUB BlockReduce to get total SAD for each candidate  
    int partial_sad = 0; 
    // pixel range for this thread to compute partial SAD
    const int startPixel = threadIdx.x * pixelsPerThread; // starting pixel index for this thread to compute partial SAD (0 to pixelsPerBlock-1)
    const int endPixel = min(startPixel + pixelsPerThread, pixelsPerBlock);  // ending pixel index  for this thread to compute partial SAD, ensuring don't go out of bounds
    for (int i = startPixel; i < endPixel; ++i) {
        unsigned char c, r;
        if constexpr (STRATEGY == SmemStrategy::FULL) {
            c = s_curr[i];
            r = s_ref[i];
        } else if constexpr (STRATEGY == SmemStrategy::CURR_ONLY) {
            c = s_curr[i];
            r = d_ref[(ref_y + (i / blockSize)) * width + (ref_x + (i % blockSize))]; 
        } else {
            // no SMEM, both current and reference blocks are read from GMEM
            int py = i / blockSize, px = i % blockSize;
            c = d_curr[(y + py) * width + (x + px)];
            r = d_ref [(ref_y + py) * width + (ref_x + px)];
        }
        partial_sad = __usad(c, r, partial_sad); 
    }

    //block reduction of partial SAD values to get total SAD for this candidate position, using CUB BlockReduce (reduction 1)
    const int total_sad = BlockReduce(partial_sad_reduce_storage).Sum(partial_sad);
    
    // leader thread writes the total SAD for this candidate position to global memory
    if (threadIdx.x == 0) {
        d_sad_candidates[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + blockIdx.z] = {total_sad, cand_dx, cand_dy}; 
    }    
}



/* Launch the computeSADKernel with the appropriate SMEM strategy and number of threads per block, based on the device properties and block size. 
    @param prop: device properties
    @param smOneFrame: SMEM size required for one frame block (current or reference)
    @param smTwoFrames: SMEM size required for two frame blocks (current and reference)
    @param grid: grid dimensions for kernel launch
    @param threadsKSad: number of threads per block for kernel launch
    @param strategy: output parameter to indicate which SMEM strategy was used for this kernel launch
    @param smKSad: output parameter to indicate how much SMEM was used for this kernel launch
    @param d_curr: device pointer to current frame
    @param d_ref: device pointer to reference frame
    @param width: width of the frame (in pixels)
    @param d_sad_candidates: device pointer to SAD values for each candidate position for each block, computed in this kernel 
    @param d_best_candidates: device pointer to current best candidate for each block at previous search distance 
    @param blockSize: size of the block (in pixels)
    @param blockX: number of blocks in the X dimension of the frame grid
    @param blockY: number of blocks in the Y dimension of the frame grid
    @param SearchDistance: current search distance (in blocks)
*/
void launchSADKernel(
    cudaDeviceProp prop, int smOneFrame, int smTwoFrames, dim3 grid, int threadsKSad, SmemStrategy & strategy, int&smKSad,
    const unsigned char* d_curr, const unsigned char* d_ref, int width, CandidateSad* d_sad_candidates, const CandidateSad* d_best_candidates, 
    int blockSize, int blockX, int blockY, int SearchDistance) 
{

    auto dispatchKernel = [&]<int THREADS>() {
        constexpr size_t cubSmem = sizeof(typename cub::BlockReduce<int, THREADS>::TempStorage);
        const size_t smem_available = prop.sharedMemPerBlock - cubSmem;

        if (smTwoFrames <= smem_available) { //FULL: enough SMEM for both current and reference blocks
            smKSad = smTwoFrames + cubSmem;
            strategy = SmemStrategy::FULL;
            computeSADKernel<SmemStrategy::FULL, THREADS><<<grid, THREADS, smTwoFrames>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, width, blockX, blockY, SearchDistance);
        } 
        else if (smOneFrame <= smem_available) { //CURR_ONLY: enough SMEM for current block only
            smKSad = smOneFrame + cubSmem;
            strategy = SmemStrategy::CURR_ONLY;
            computeSADKernel<SmemStrategy::CURR_ONLY, THREADS><<<grid, THREADS, smOneFrame>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, width, blockX, blockY, SearchDistance);
        } 
        else { // not enough SMEM for frame blocks
            smKSad = 0;
            strategy = SmemStrategy::NONE;
            computeSADKernel<SmemStrategy::NONE, THREADS><<<grid, THREADS, 0>>>(
                d_curr, d_ref, d_sad_candidates, d_best_candidates, blockSize, width, blockX, blockY, SearchDistance);
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
        default:   throw runtime_error("exceeded number of threads per block: " + to_string(threadsKSad));
    }
}
/* Cuda optimed logarithmic search for grayscale images 
 without specify searchDistance, is computed as max(blocksX, blocksY)-1, which is the maximum distance to cover the entire frame
    @param curr: current frame
    @param ref: reference frame
    @param blockSize: size of the block (in pixels)
    @param searchDistance: maximum search distance (in blocks)
    @param metrics: structure to hold timing metrics for the entire block matching process and individual kernels
    @return: 2D vector of motion vectors for each block in the current frame 
*/
vector<vector<MotionVector>> logarithmicSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchDistance, SingleRunMetrics& metrics) {
    
    auto total_time_start = chrono::high_resolution_clock::now();
    
    // CUDA timing events
    cudaEvent_t evKSADStart, evKSADStop, segmentedReduceStart, segmentedReduceStop;
    createCudaEvent(evKSADStart); createCudaEvent(evKSADStop);
    createCudaEvent(segmentedReduceStart); createCudaEvent(segmentedReduceStop);

    // Device properties
    cudaDeviceProp deviceProp;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    ValidationUtils::validateFrameDimensions(curr, ref);
    
    // Calculate frame grid dimensions
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    // logarith search all over the  frame
    if (searchDistance < 0) {
        searchDistance = max(blocksX, blocksY)-1;
    }     

    const int pixelsPerBlock = blockSize * blockSize;   // number of pixels in a block
    const int frameBlocks = blocksX * blocksY;  // number of blocks in the frame grid
    
    // Log processing info
    LoggingUtils::printFrameInfo("CUDA Optimized", curr.width, curr.height, blockSize, blocksX, blocksY);
    cout << "CUDA Optimized Logarithmic (Grayscale): Search strategy: Logarithmic search (distance="
         << searchDistance << " blocks)" << endl;
    
    cout << "CUDA Optimized (Grayscale): GPU: " << deviceProp.name << "\n"
         << "  maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << "\n"
         << "  sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " bytes\n";
                     
    // Allocate device memory for current and reference and copy frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    
    size_t bytesPerFrame = 0;
    GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    LoggingUtils::printCopyingToGPU("CUDA Uncoalesced Optimized");
    GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);

    CandidateSad* d_sad_candidates = nullptr; // device memory for SAD values, allocated as blocksX * blocksY * MAXCANDIDATES_AT_DISTANCE to store SAD for each candidate position for each block
    CandidateSad* d_best_candidates = nullptr; // device memory to pass current iteration's best matches, will contain the final best candidates and motion vectors after last iteration
    gpuErrorCheck(cudaMalloc(&d_sad_candidates, (size_t) frameBlocks * MAXCANDIDATES_AT_DISTANCE * sizeof(CandidateSad))); 
    gpuErrorCheck(cudaMalloc(&d_best_candidates, (size_t) frameBlocks * sizeof(CandidateSad))); 
    
    cout << "CUDA Optimized (Grayscale):  d_sad_candidates allocated: " << frameBlocks * MAXCANDIDATES_AT_DISTANCE * sizeof(CandidateSad)/1024 << " KB" << endl;
    cout << "CUDA Optimized (Grayscale):  d_best_candidates allocated: " << frameBlocks * sizeof(CandidateSad)/1024 << " KB" << endl;
    //start exection: Initialize best candidates to max SAD and zero motion vector
    vector<CandidateSad> h_init(frameBlocks, {INT_MAX, 0, 0});
    cudaMemcpy(d_best_candidates, h_init.data(), frameBlocks * sizeof(CandidateSad), cudaMemcpyHostToDevice);
    
    // iterator for segment offsets for CUB segmented reduction, each segment corresponds to SAD values for one candidate position for all blocks
    cub::CountingInputIterator<int> counting_iter(0);
    cub::TransformInputIterator<int, SegmentOffsetOp, cub::CountingInputIterator<int>> 
        d_offsets(counting_iter, SegmentOffsetOp(MAXCANDIDATES_AT_DISTANCE));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, 
                                       d_sad_candidates, d_best_candidates, 
                                       frameBlocks, d_offsets, d_offsets + 1, 
                                       MotionVectorUtils::CandidateSadOp(), CandidateSad{INT_MAX, 0, 0});
    
    // determine temporary storage requirements for segmented reduction of SAD candidates to find best candidate for each block                            
    gpuErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    //Ksad: 3D grid (blockX x blockY x min(maxCandidates, maxGridSizeZ)) and 1D CUDA blocks
    const int KSADgridZ = min(MAXCANDIDATES_AT_DISTANCE, (int)deviceProp.maxGridSize[2]); // number of candidate positions processed per block in computeSADKernel, limited by max grid size in Z dimension 
    int limitThreads = min(pixelsPerBlock, deviceProp.maxThreadsPerBlock);
    int threadsKSad = 1;
    while ((threadsKSad << 1) <= limitThreads) threadsKSad <<= 1;
    // Determine SMEM occupied by one frame block and two frame blocks, to decide which SMEM strategy to use in the kernel
    const size_t smOneFrame = pixelsPerBlock * sizeof(unsigned char);
    const size_t smTwoFrames = 2 * smOneFrame;
    
    // SMEM strategy and smem used for computeSADKernel
    SmemStrategy strategy;
    int smKSad;

    // grid and block dimensions for computeSADKernel
    const dim3 gridKSad(blocksX, blocksY, KSADgridZ);
    const dim3 blockDimKSad(threadsKSad, 1, 1);

    metrics.gpu_kernel1_ms = 0.0f; // initialize kernel timing metric
    metrics.gpu_kernel2_ms = 0.0f; // initialize kernel timing metric    

    cout << "CUDA Optimized (Grayscale): computeSADKernel launch configuration: grid(" << gridKSad.x << ", " << gridKSad.y << ", " << gridKSad.z 
         << "), block(" << blockDimKSad.x << ", " << blockDimKSad.y << ", " << blockDimKSad.z 
         << "), threads per block: " << threadsKSad << endl;
    for (int currentDistance = searchDistance; currentDistance > 0; currentDistance >>= 1) {
        recordCudaEvent(evKSADStart);
        launchSADKernel(
            deviceProp, smOneFrame, smTwoFrames, gridKSad, threadsKSad, strategy, smKSad,
            d_curr, d_ref, curr.width, d_sad_candidates, d_best_candidates, 
            blockSize, blocksX, blocksY, currentDistance);
        recordCudaEvent(evKSADStop);
        cout << "CUDA Optimized (Grayscale): Kernel launched computeSADKernel, synchronizing..." << endl;
        gpuErrorCheck(cudaEventSynchronize(evKSADStop));
        gpuErrorCheck(cudaGetLastError());
        metrics.gpu_kernel1_ms += elapsedCudaTime(evKSADStart, evKSADStop);
        
        recordCudaEvent(segmentedReduceStart);  
        cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, 
                                           d_sad_candidates, d_best_candidates, 
                                           frameBlocks, d_offsets, d_offsets + 1, 
                                           MotionVectorUtils::CandidateSadOp(), CandidateSad{INT_MAX, 0, 0});
        recordCudaEvent(segmentedReduceStop);
        cout << "CUDA Optimized (Grayscale): launched segmented reduce, synchronizing..." << endl;
        gpuErrorCheck(cudaEventSynchronize(segmentedReduceStop));
        gpuErrorCheck(cudaGetLastError());
        metrics.gpu_kernel2_ms += elapsedCudaTime(segmentedReduceStart, segmentedReduceStop);
    }
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

    // Stop total timer after cleanup
    auto total_time_stop = chrono::high_resolution_clock::now();
    float total_time = chrono::duration<float, milli>(total_time_stop - total_time_start).count();
    metrics.total_ms = total_time;

    destroyCudaEvent(evKSADStart);
    destroyCudaEvent(evKSADStop);

    for(int i = 0; i < frameBlocks; ++i) {
        h_mv_flat[i] = { h_best_flat[i].dx, h_best_flat[i].dy };
    }
    vector<vector<MotionVector>> result = GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);
    
    LoggingUtils::printProcessingComplete("CUDA Optimized");
    return result;
}