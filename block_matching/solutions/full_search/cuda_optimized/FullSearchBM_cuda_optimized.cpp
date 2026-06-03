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
#include <chrono>

using namespace std;

// Computes SAD  between the current block and each candidate position in the search window.
__global__ void computeSADKernel(const unsigned char*  d_curr,const unsigned char*  d_ref,
                                 int* d_sad,int blockSize,  size_t frame_pitch_bytes, int searchRange){

    
    // Calculate search window using helper function
    // bounds can be load in SMEM, but is very small so calculate it in each thread to avoid extra shared memory usage and synchronization
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);

    if (blockIdx.z >= bounds.totalPositions) { // out of bounds for the number of candidate positions in the search window
        return;
    }
    
    const int pixelsPerBlock = blockSize * blockSize; // number of pixels in a frame block
    const int pixelsPerThread = (pixelsPerBlock + blockDim.x - 1) / blockDim.x; // divide pixels among threads, rounding up
    
    // top-left corner of the current block in the current frame
    // x and y can be load in SMEM, but is only 2 ints so calculate in each thread to avoid extra shared memory usage and synchronization
    const int x = blockIdx.x * blockSize; 
    const int y = blockIdx.y * blockSize;

    // ref_x and ref_y are the top-left corner of the candidate block in the reference frame corresponding to this blockIdx.zq
    // ref_x ref_y can be load in SMEM, but is only 2 ints so calculate in each thread to avoid extra shared memory usage and synchronization 
    const int ref_x = (bounds.startX + (blockIdx.z % bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels   
    const int ref_y = (bounds.startY + (blockIdx.z / bounds.width)) * blockSize;  // candidate block's top-left corner in the reference frame in pixels
    // blockIdx.z selects which candidate (in block coordinates) is loaded from the reference frame
    
    // load shared memory 
    extern  __shared__ unsigned char smemSad[];             
    int* s_partial_sad = (int*)smemSad;                         // shared memory for partial SAD results (one per thread), put here for correct SMEM padding
    unsigned char* s_curr = (unsigned char*)(s_partial_sad + blockDim.x);   // shared memory for current block pixels
    unsigned char* s_ref = s_curr + pixelsPerBlock;         // shared memory for reference block pixels
    
    // load blocks in shared memory with coalesced access
    // Pixel load per thread: i = threadIdx.x, threadIdx.x + blockDim.x, ...
    // Pixel count ~= pixelsPerBlock / blockDim.x 
    /*for (int i = threadIdx.x; i < pixelsPerBlock; i += blockDim.x) {
        int px = i % blockSize;                                         // pixel x coordinate within the block
        int py = i / blockSize;                                         // pixel y coordinate within the block
        s_curr[i] = d_curr[(y + py) * width + (x + px)];                // load current block pixel
        s_ref[i] = d_ref[(ref_y + py) * width + (ref_x + px)];          // load reference block pixel (//FIXME_ldg read-only cache optimization)
    }

    __syncthreads();*/
    // optimized load using uint4 vectorized access (4 pixels at a time)
    uint32_t* s_curr_vec = reinterpret_cast<uint32_t*>(s_curr); // reinterpret shared memory as uint32_t for vectorized access (4 pixels per uint32_t)
    uint32_t* s_ref_vec  = reinterpret_cast<uint32_t*>(s_ref); // reinterpret shared memory as uint32_t for vectorized access (4 pixels per uint32_t)

    // number of 4-pixel groups per block, each thread will load multiple groups if blockDim.x < pixelsPerBlock/4
    const int fetchesPerBlock = pixelsPerBlock / 4;

    // load 4 pixels at a time with coalesced access, handling boundary conditions for blocks that are not a multiple of 4 pixels
    for (int i = threadIdx.x; i < fetchesPerBlock; i += blockDim.x) {
        
        // calculate pixel coordinates within the block for this fetch
        int px = (i * 4) % blockSize;
        int py = (i * 4) / blockSize;

        const unsigned char* row_curr = d_curr + (y + py) * frame_pitch_bytes; // calculate row pointer for current frame 
        const uint32_t* fetch_ptr_curr = reinterpret_cast<const uint32_t*>(row_curr + (x + px)); // pointer for vectorized fetch from current frame
        s_curr_vec[i] = *fetch_ptr_curr; 

        const unsigned char* row_ref = d_ref + (ref_y + py) * frame_pitch_bytes;
        const uint32_t* fetch_ptr_ref = reinterpret_cast<const uint32_t*>(row_ref + (ref_x + px));
        s_ref_vec[i] = *fetch_ptr_ref;
    }

    __syncthreads();


    // partial sad computation for each thread
    int partial_sad = 0;
    const int startPixel = threadIdx.x * pixelsPerThread;                   // starting pixel index for this thread
    const int endPixel = min(startPixel + pixelsPerThread, pixelsPerBlock); // ending pixel index for this thread
    for (int i = startPixel; i < endPixel; ++i) {                           
        partial_sad += abs((int)s_curr[i] - (int)s_ref[i]);                 // accumulate SAD for assigned pixels               
    }
    s_partial_sad[threadIdx.x] = partial_sad;                   // store partial SAD in shared memory 
    __syncthreads(); 

    // reduction in shared memory to get total SAD for this position
    
    // stride-base descending reduction in SMEM to get total SAD.
    // 
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        int neighbour = threadIdx.x + stride;
        if (threadIdx.x < stride && neighbour < blockDim.x) {
            s_partial_sad[threadIdx.x] += s_partial_sad[neighbour];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { // thread 0 writes the final SAD for this candidate position to global memory
        d_sad[(blockIdx.y* gridDim.x + blockIdx.x)* gridDim.z + blockIdx.z] = s_partial_sad[0]; 
    }
}

// Finds the best motion vector for each block
// __forceinline__: reduce function call overheard, this function is called in the reduction loop for every candidate.
__device__ __forceinline__ bool isBetterMotionVector(int sadA, int dxA, int dyA,
                                                     int sadB, int dxB, int dyB) { 
    if (sadB < sadA) {
        return true;
    }
    if (sadB > sadA) {
        return false;
    }
    const int distA = dxA * dxA + dyA * dyA;
    const int distB = dxB * dxB + dyB * dyB;
    return distB < distA;
}

__global__ void findBestMVKernel(const int* d_sad, MotionVector* d_mv, int maxCandidates, int searchRange, int reductionBase){
    CudaSearchBounds bounds = calculateCudaSearchBounds(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, searchRange);

    // Use maxCandidates for indexing, not bounds.totalPositions
    // because d_sad array is allocated as blocksX * blocksY * maxCandidates
    const int base = (blockIdx.y * gridDim.x + blockIdx.x) * maxCandidates;

    // linear thread id for 2D blocks
    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y; // total threads in this block

    // local best for each thread: SAD + dx + dy
    int local_best_sad = INT_MAX;
    int local_best_dx = 0;
    int local_best_dy = 0;

    // shared memory reduction to find global best among threads in this block
    extern __shared__ int smemBestMV[];
    int* s_sad = smemBestMV;
    int* s_dx = s_sad + threadsPerBlock;
    int* s_dy = s_dx + threadsPerBlock;

    // Computing dx, dy from bounds
    // base is used to index into d_sad for this block, and bz iterates over candidate positions in the search window
    for (int bz = tidx; bz < bounds.totalPositions; bz += threadsPerBlock) {
        const int sad = d_sad[base + bz];
        const int ref_bx = bounds.startX + (bz % bounds.width);
        const int ref_by = bounds.startY + (bz / bounds.width);
        const int dx = ref_bx - blockIdx.x;
        const int dy = ref_by - blockIdx.y;
        if (isBetterMotionVector(local_best_sad, local_best_dx, local_best_dy,
                                 sad, dx, dy)) {
            local_best_sad = sad;
            local_best_dx = dx;
            local_best_dy = dy;
        }
    }

    // store local best into SMEM
    s_sad[tidx] = local_best_sad;
    s_dx[tidx] = local_best_dx;
    s_dy[tidx] = local_best_dy;
    __syncthreads();

    // to hadle case threadsPerBlock is not a power of two
    if (tidx < threadsPerBlock - reductionBase) { 
        const int neighbour_idx = tidx + reductionBase;
        if (isBetterMotionVector(s_sad[tidx], s_dx[tidx], s_dy[tidx],
                                 s_sad[neighbour_idx], s_dx[neighbour_idx], s_dy[neighbour_idx])) {
            s_sad[tidx] = s_sad[neighbour_idx];
            s_dx[tidx] = s_dx[neighbour_idx];
            s_dy[tidx] = s_dy[neighbour_idx];
        }
    }
    __syncthreads();

    // Parallel stride descending reduction in SMEM
    for (int stride = reductionBase >> 1; stride > 0; stride >>= 1) {
        int neighbour_idx = tidx + stride;
        if (tidx < stride && neighbour_idx < reductionBase) {
            if (isBetterMotionVector(s_sad[tidx], s_dx[tidx], s_dy[tidx],
                                     s_sad[neighbour_idx], s_dx[neighbour_idx], s_dy[neighbour_idx])) {
                s_sad[tidx] = s_sad[neighbour_idx];
                s_dx[tidx] = s_dx[neighbour_idx];
                s_dy[tidx] = s_dy[neighbour_idx];
            }
        }
        __syncthreads();
    }

    if (tidx == 0) {
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {s_dx[0], s_dy[0]};
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
    
    
    // KSAD kernel: 1D block with threads processing pixels in parallel for SAD computation.
    // determine maximum useful threads: choose largest power-of-two <= min(pixelsPerBlock, maxThreadsPerBlock)
    int limitThreads = (pixelsPerBlock < deviceProp.maxThreadsPerBlock) ? pixelsPerBlock : deviceProp.maxThreadsPerBlock;
    int threadsKSad = 1;
    while ((threadsKSad << 1) <= limitThreads) threadsKSad <<= 1;

    // shared mem needed for KSad
    size_t smKSad = threadsKSad * sizeof(int) // for partial SAD results 
        + 2 * pixelsPerBlock * sizeof(unsigned char) ; // for current and reference frame blocks   
    // Ensure KSad shared memory fits device limits by reducing threadsKSad if needed
    while (smKSad > (size_t)deviceProp.sharedMemPerBlock && threadsKSad > 1) {
        threadsKSad = threadsKSad >> 1;
        smKSad = 2 * pixelsPerBlock * sizeof(unsigned char) + threadsKSad * sizeof(int);
    }

    //findBestMV: 2D block (blockX x blockY) matching search window dims
    int threadsXBestMV = searchWindowBlocksX;
    int threadsYBestMV = searchWindowBlocksY;
    int threadsPerBlockBestMV = threadsXBestMV * threadsYBestMV;
    // If product exceeds device limit, reduce dimensions proportionally:
    // iteratively halve the larger dimension until product <= maxThreadsPerBlock
    while (threadsPerBlockBestMV > deviceProp.maxThreadsPerBlock) {
        if (threadsXBestMV >= threadsYBestMV) {
            threadsXBestMV = threadsXBestMV >> 1;
        } else {
            threadsYBestMV = threadsYBestMV >> 1;
        }
        threadsPerBlockBestMV = threadsXBestMV * threadsYBestMV;
    }

    size_t smKBestMV = (size_t)threadsPerBlockBestMV * 3 * sizeof(int); // shared SAD + dx + dy per thread
    // Ensure KBestMV shared memory fits device limits by reducing block dimensions if needed
    while (smKBestMV > (size_t)deviceProp.sharedMemPerBlock) {
        if (threadsXBestMV > 1) {
            threadsXBestMV = threadsXBestMV >> 1;
        } else if (threadsYBestMV > 1) {
            threadsYBestMV = threadsYBestMV >> 1;
        } else {
            break;
        }
        // ensure don't exceed max threads per block
        if (threadsXBestMV * threadsYBestMV > deviceProp.maxThreadsPerBlock) {
            threadsYBestMV = deviceProp.maxThreadsPerBlock / threadsXBestMV;
        }
        threadsPerBlockBestMV = threadsXBestMV * threadsYBestMV;
        smKBestMV = (size_t)threadsPerBlockBestMV * 3 * sizeof(int);
    }
    int reductionBase = 1;
    while ((reductionBase << 1) <= threadsPerBlockBestMV) reductionBase <<= 1;    
                        
    // Allocate device memory for current and reference frames
    const size_t sadBytes = (size_t)blocksX * blocksY * maxCandidates * sizeof(int);
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    size_t bytesPerFrame = 0;
    //GPUMemoryUtils::allocateFrames(curr, ref, d_curr, d_ref, bytesPerFrame);
    // with default load  "The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to L2 is a 128 byte cache line. 
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

           cout << "CUDA Optimized (Grayscale): Selected threads for KSad: " << threadsKSad
               << " and block "<< threadsXBestMV << "x" << threadsYBestMV
               << " (K1=" << smKSad/1024.0 << "KB, K2=" << smKBestMV/1024.0 << "KB)\n\n";
    cout << "CUDA Optimized (Grayscale): d_sad = " << sadBytes / (1024.0 * 1024.0) << " MB\n";

    LoggingUtils::printCopyingToGPU("CUDA Optimized");
    //GPUMemoryUtils::copyFramesToGPU(d_curr, d_ref, curr, ref, bytesPerFrame);
    size_t widthBytes = curr.width * sizeof(unsigned char);
    gpuErrorCheck(cudaMemcpy2D(d_curr, frame_pitch_bytes, curr.data.data(), widthBytes, widthBytes, curr.height, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy2D(d_ref, frame_pitch_bytes, ref.data.data(), widthBytes, widthBytes, ref.height, cudaMemcpyHostToDevice));
    

    // Grid and block dimensions
    const dim3 gridKSad(blocksX, blocksY, maxCandidates);
    const dim3 gridKBestMV(blocksX, blocksY);
    const dim3 blockDimKSad(threadsKSad, 1, 1);
    const dim3 blockDimBestMV(threadsXBestMV, threadsYBestMV, 1);

    recordCudaEvent(evKSADStart);

        cout << "CUDA Optimized (Grayscale): Launching computeSADKernel with "
            << blockDimKSad.x << "x" << blockDimKSad.y << "x" << blockDimKSad.z
            << " threads per block (" << threadsKSad << " total threads)..." << endl;

        computeSADKernel<<<gridKSad, blockDimKSad, smKSad>>>(d_curr, d_ref, d_sad,
                                                  blockSize, frame_pitch_bytes, searchRange);
    gpuErrorCheck(cudaGetLastError());
    recordCudaEvent(evKSADStop);
    // gpuErrorCheck(cudaDeviceSynchronize()); // no need to synchronize, with default stream, kernel launches are serialized.
    recordCudaEvent(evKBestMVStart);

        
        cout << "CUDA Optimized (Grayscale): Launching findBestMVKernel with "
            << blockDimBestMV.x << "x" << blockDimBestMV.y << "x" << blockDimBestMV.z
            << " threads per block (" << threadsPerBlockBestMV << " total threads)..." << endl;

        findBestMVKernel<<<gridKBestMV, blockDimBestMV, smKBestMV>>>(d_sad, d_mv, maxCandidates, searchRange, reductionBase);
    gpuErrorCheck(cudaGetLastError());
    
    recordCudaEvent(evKBestMVStop);

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