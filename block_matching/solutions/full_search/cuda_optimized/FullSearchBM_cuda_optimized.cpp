#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;


__global__ void computeSADKernel(const unsigned char*  d_curr,const unsigned char*  d_ref,
                                 int* d_sad,int blockSize, int width, int searchRange){

    
    // index
    const int x = blockIdx.x * blockSize;
    const int y = blockIdx.y * blockSize;
    const int pixelsPerBlock = blockSize * blockSize;
    const int pixelsPerThread = (pixelsPerBlock + blockDim.x - 1) / blockDim.x;

    int search_bx_start = 0;
    int search_bx_end = 0;
    int search_by_start = 0;
    int search_by_end = 0;
    int search_w = 0;
    int search_h = 0;
    int total_positions = 0;

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

    if (blockIdx.z >= total_positions) {
        return;
    }

    const int ref_bx = search_bx_start + (blockIdx.z % search_w);
    const int ref_by = search_by_start + (blockIdx.z / search_w);
    const int ref_x = ref_bx * blockSize;
    const int ref_y = ref_by * blockSize;
    
    // shared memory 
    extern  __shared__ unsigned char smem[];
    unsigned char* s_curr = smem;
    unsigned char* s_ref = s_curr + pixelsPerBlock;
    int* s_partial_sad = (int*)(s_ref + pixelsPerBlock);
    
    
    // process in row-major order, better coalescing
    for (int i = threadIdx.x; i < pixelsPerBlock; i += blockDim.x) {
        int px = i % blockSize;
        int py = i / blockSize;
        s_curr[i] = d_curr[(y + py) * width + (x + px)];
        s_ref[i] = __ldg(&d_ref[(ref_y + py) * width + (ref_x + px)]);
    }

    __syncthreads();

    // partial sad computation for each thread
    int partial_sad = 0;
    const int startPixel = threadIdx.x * pixelsPerThread;
    const int endPixel = min(startPixel + pixelsPerThread, pixelsPerBlock);
    for (int i = startPixel; i < endPixel; ++i) {
        partial_sad += abs((int)s_curr[i] - (int)s_ref[i]);                                
    }
    s_partial_sad[threadIdx.x] = partial_sad;
    __syncthreads(); // FIXME is always needed to synchronize?

    // reduction in shared memory to get total SAD for this position
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial_sad[threadIdx.x] += s_partial_sad[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_sad[(blockIdx.y* gridDim.x + blockIdx.x)* gridDim.z + blockIdx.z] = s_partial_sad[0];
    }
}

__global__ void findBestMVKernel(const int* d_sad, MotionVector* d_mv, int searchRange){
    int search_bx_start = 0;
    int search_bx_end = 0;
    int search_by_start = 0;
    int search_by_end = 0;
    int search_w = 0;
    int search_h = 0;
    int total_positions = 0;

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

    const int base = (blockIdx.y * gridDim.x + blockIdx.x) * total_positions;

    // local best for each thread
    int local_best_sad = INT_MAX;
    int local_best_dx = 0, local_best_dy = 0;
    int local_best_dist = INT_MAX;
    for (int bz = threadIdx.x; bz < total_positions; bz += blockDim.x) {
        const int sad = d_sad[base + bz];
        const int ref_bx = search_bx_start + (bz % search_w);
        const int ref_by = search_by_start + (bz / search_w);
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

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_sad[threadIdx.x + stride] < s_sad[threadIdx.x] || 
                (s_sad[threadIdx.x + stride] == s_sad[threadIdx.x] && s_dist[threadIdx.x + stride] < s_dist[threadIdx.x])) {
                s_sad[threadIdx.x] = s_sad[threadIdx.x + stride];
                s_dx[threadIdx.x] = s_dx[threadIdx.x + stride];
                s_dy[threadIdx.x] = s_dy[threadIdx.x + stride];
                s_dist[threadIdx.x] = s_dist[threadIdx.x + stride];
            }
        }
        __syncthreads(); // FIXME is always needed to synchronize?
    }
    if (threadIdx.x == 0) {
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {s_dx[0], s_dy[0]};
    }
}


vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                         int blockSize, int searchRange) {
    cudaSetDevice(0);
    
    // Validate input: current and reference frames must have same dimensions
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }

    // Calculate grid dimensions based on block size and frame dimensions
    const int blocksX = curr.width / blockSize;
    const int blocksY = curr.height / blockSize;
    const int pixelsPerBlock = blockSize * blockSize;
    const int TotalBlocks = blocksX * blocksY;
        const int searchWindowBlocksX = (searchRange > 0) ? min(blocksX, searchRange * 2 + 1) : blocksX;
        const int searchWindowBlocksY = (searchRange > 0) ? min(blocksY, searchRange * 2 + 1) : blocksY;
        const int maxCandidates = searchWindowBlocksX * searchWindowBlocksY;
        cout << "CUDA Optimized (Grayscale): Processing frame " << curr.width << "x" << curr.height
            << " with block size " << blockSize << endl;
        cout << "CUDA Optimized (Grayscale): Grid size: " << blocksX << "x" << blocksY
            << " = " << TotalBlocks << " blocks" << std::endl;
        cout << "CUDA Optimized (Grayscale): Search mode: "
            << ((searchRange > 0) ? ("Range search (range=" + to_string(searchRange) + " blocks)") : "Full search")
            << " (searchRange=" << searchRange << ")" << std::endl;

    // device properties
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
    /* shared memory calculation:
        Kernel computeSADKernel:
            - Must load 2 blocks (curr + ref) in shared memory
            - Must keep array of partial SADs (one per thread)
            - SM_KSad = 2×pixelsPerBlock + t×sizeof(int)
        Kernel 2 (findBestMVKernel):
            - Must keep 4 arrays (sad, dx, dy, dist) for reduction
            - SM_KBestMV = 4×t×sizeof(int) = 16t bytes
    */
    
    int threadsPerBlock = 32; // default fallback
    size_t smKSad, smKBestMV;
        
        // Descend by powers of 2: 1024→512→256→128→64→32
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
    const size_t bytesPerFrame = curr.width * curr.height * sizeof(unsigned char); 
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref  = nullptr;
    int*           d_sad  = nullptr;
    MotionVector*  d_mv   = nullptr;

    gpuErrorCheck(cudaMalloc(&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc(&d_ref,  bytesPerFrame));
    gpuErrorCheck(cudaMalloc(&d_sad,  sadBytes));
    gpuErrorCheck(cudaMalloc(&d_mv,   blocksX * blocksY * sizeof(MotionVector)));

    cout << "CUDA Optimized (Grayscale): Selected threads per block: " << threadsPerBlock
         << " (K1=" << smKSad/1024.0 << "KB, K2=" << smKBestMV/1024.0 << "KB)\n\n";

    cout << "CUDA Optimized (Grayscale): d_sad = " << sadBytes / (1024.0 * 1024.0) << " MB\n";
    cout << "CUDA Optimized (Grayscale): Copying frames to GPU..." << std::endl;
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref,  ref.data.data(),  bytesPerFrame, cudaMemcpyHostToDevice));

    // grid and block dimensions
    const dim3 gridKSad(blocksX, blocksY, maxCandidates); // 3D grid: (blockX, blockY, local candidate position) 
    const dim3 gridKBestMV(blocksX, blocksY);           // 2D grid for reduction
    const dim3 blockDim(threadsPerBlock, 1, 1);         // 1D block: all threads collaborate on SAD computation for one position

    // create CUDA events for timingq
    cudaEvent_t evTotalStart, evTotalStop;
    cudaEvent_t evKSadStart,    evKSadStop;
    cudaEvent_t evKBestMVStart,    evKBestMVStop;

    createCudaEvent(evTotalStart); createCudaEvent(evTotalStop);
    createCudaEvent(evKSadStart);    createCudaEvent(evKSadStop);
    createCudaEvent(evKBestMVStart);    createCudaEvent(evKBestMVStop);

    // computeSADKernel
    recordCudaEvent(evTotalStart);
    recordCudaEvent(evKSadStart);

    cout << "CUDA Optimized (Grayscale): Launching computeSADKernel with "
         << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
         << " threads per block (" << threadsPerBlock << " total threads)..." << std::endl;

    computeSADKernel<<<gridKSad, blockDim, smKSad>>>(d_curr, d_ref, d_sad,
                                                     blockSize, curr.width, searchRange);
    gpuErrorCheck(cudaGetLastError());
    recordCudaEvent(evKSadStop);

    //findBestMVKernel

    recordCudaEvent(evKBestMVStart);

    cout << "CUDA Optimized (Grayscale): Launching findBestMVKernel with "
         << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
         << " threads per block (" << threadsPerBlock << " total threads)..." << std::endl;

    findBestMVKernel<<<gridKBestMV, blockDim, smKBestMV>>>(d_sad, d_mv, searchRange);
    gpuErrorCheck(cudaGetLastError());
    recordCudaEvent(evKBestMVStop);
    recordCudaEvent(evTotalStop);

    cout << "CUDA Optimized (Grayscale): Kernel launched, synchronizing..." << std::endl;
    gpuErrorCheck(cudaDeviceSynchronize());

    // Timing report
    const float msKSad    = elapsedCudaTime(evKSadStart,    evKSadStop);
    const float msKBestMV    = elapsedCudaTime(evKBestMVStart,    evKBestMVStop);
    const float msTotal = elapsedCudaTime(evTotalStart, evTotalStop);

    cout << "CUDA Optimized (Grayscale): Timing:" << endl;
    cout << "  computeSADKernel : " << msKSad << " ms" << std::endl;
    cout << "  findBestMVKernel : " << msKBestMV << " ms" << std::endl;
    cout << "  Total            : " << msTotal << " ms" << std::endl;

    destroyCudaEvent(evTotalStart); destroyCudaEvent(evTotalStop);
    destroyCudaEvent(evKSadStart);    destroyCudaEvent(evKSadStop);
    destroyCudaEvent(evKBestMVStart);    destroyCudaEvent(evKBestMVStop);

    // cudaMemcpy results back to host
    vector<MotionVector> h_mv_flat(blocksX * blocksY);
    gpuErrorCheck(cudaMemcpy(h_mv_flat.data(), d_mv,
                             blocksX * blocksY * sizeof(MotionVector),
                             cudaMemcpyDeviceToHost));

    vector<vector<MotionVector>> result(blocksY, vector<MotionVector>(blocksX));
    for (int by = 0; by < blocksY; ++by)
        for (int bx = 0; bx < blocksX; ++bx)
            result[by][bx] = h_mv_flat[by * blocksX + bx];

    // Cleanup GPU
    cout << "CUDA Optimized (Grayscale): Cleaning up GPU memory..." << std::endl;
    cudaFree(d_curr);
    cudaFree(d_ref);
    cudaFree(d_sad);
    cudaFree(d_mv);

    cout << "CUDA Optimized (Grayscale): Complete!" << std::endl;
    return result;
    
}


