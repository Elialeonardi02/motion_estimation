#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include "FullSearchBM_cuda_uncoalesced_optimized.h"
#include "cuda_utils.h"
#include "sad_utils.h"

using namespace std;

// Device function to compute PARTIAL SAD for a subset of pixels
// Each thread along Z (if is not 1) dimension computes SAD for a portion of pixels, then results are reduced
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
        sad += abs( curr[pixelIdx] - __ldg(&ref[(y2 + y) * refWidth + (x2 + x)])); // reference frame is read only, __ldg intrinsic use read-only cache
    }
    return sad;
}


// CUDA kernel: each grid block processes one search block of current frame
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize,
                                 int width, int height, int blocksX, int blocksY, int threadsPerBlock) {
    
    if (blockIdx.x >= blocksX || blockIdx.y >= blocksY) return;
    
    // Thread indices
    const int tidx_2d = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int tidx_3d = tidx_2d + (threadIdx.z * blockDim.x * blockDim.y);
    
    // Current block top-left corner
    const int x = blockIdx.x * blockSize;
    const int y = blockIdx.y * blockSize;
    const int total_positions = blocksX * blocksY;
    
    // Thread-local best match
    int best_sad = INT_MAX; // Initialize best SAD with worst case
    int best_dx = 0, best_dy = 0;
    
    // Shared memory layout: current block + SAD partial + reduction buffers
    extern __shared__ unsigned char shared_memory[];
    unsigned char* s_curr = shared_memory;
    int threadsXY = blockDim.x * blockDim.y;
    
    int* shared_block_thread_sad_partial = (int*)(s_curr + blockSize * blockSize); // buffer for partial SAD results from each thread (size = threadsPerBlock)
    int* shared_block_thread_sad = (int*)(shared_block_thread_sad_partial + threadsPerBlock); // buffer for final SAD results from each thread after reduction (size = threadsXY)
    int* shared_block_thread_dx = shared_block_thread_sad + threadsXY; // buffer for dx results corresponding to SAD results (size = threadsXY)
    int* shared_block_thread_dy = shared_block_thread_dx + threadsXY; // buffer for dy results corresponding to SAD results (size = threadsXY)
    int* shared_block_thread_dist = shared_block_thread_dy + threadsXY; // buffer for distance results corresponding to SAD results (size = threadsXY)
    
    // Load current block in shared memory
    for (int i = tidx_3d; i < blockSize * blockSize; i += threadsPerBlock) {
        s_curr[i] = d_curr[(y + (i / blockSize)) * width + (x + (i % blockSize))];
    }

    __syncthreads();
    
    // Precalculate pixels-per-thread for SAD parallelization
    const int pixelsPerThread = (blockSize * blockSize + blockDim.z - 1) / blockDim.z;
    
    // Each thread (X,Y,Z) searches positions and computes partial SAD along Z
    if (total_positions <= blockDim.x * blockDim.y) {
        // Few positions: each thread (X,Y) processes at most one position, parallelize SAD along Z
        if (tidx_2d < total_positions) {
            int ref_x = (tidx_2d % blocksX) * blockSize;
            int ref_y = (tidx_2d / blocksX) * blockSize;

            shared_block_thread_sad_partial[tidx_3d] = computeSAD_device_partial(
                s_curr, d_ref, ref_x, ref_y, blockSize, width,
                threadIdx.z * pixelsPerThread, pixelsPerThread);
        } else {
            shared_block_thread_sad_partial[tidx_3d] = 0; // Threads with no position to process contribute 0 to SAD sum
        }
        
        __syncthreads();
        
        // Reduce along Z: sum partial SADs
        for (int stride = 1; stride < blockDim.z; stride *= 2) {
            if (threadIdx.z + stride < blockDim.z) {
                // Add SAD from neighbor thread in Z dimension
                shared_block_thread_sad_partial[tidx_3d] += shared_block_thread_sad_partial[tidx_3d + stride * threadsXY]; 
            }
            __syncthreads();
        }
        
        // Thread z=0 has complete SAD for this position
        if (threadIdx.z == 0 && tidx_2d < total_positions) {
            best_sad = shared_block_thread_sad_partial[tidx_3d];
            best_dx = ((tidx_2d % blocksX) * blockSize - x) / blockSize;
            best_dy = ((tidx_2d / blocksX) * blockSize - y) / blockSize;
        }
    } else {
        // Many positions: each thread (X,Y) processes multiple positions, parallelize SAD along Z for each position
        for (int pos = tidx_2d; pos < total_positions; pos += threadsXY) {
            int ref_x = (pos % blocksX) * blockSize;
            int ref_y = (pos / blocksX) * blockSize;

            shared_block_thread_sad_partial[tidx_3d] = computeSAD_device_partial(
                s_curr, d_ref, ref_x, ref_y, blockSize, width,
                threadIdx.z * pixelsPerThread, pixelsPerThread);

            __syncthreads();
            
            // Reduce along Z
            for (int stride = 1; stride < blockDim.z; stride *= 2) {
                if (threadIdx.z + stride < blockDim.z) {
                    // Add SAD from neighbor thread in Z dimension
                    shared_block_thread_sad_partial[tidx_3d] += shared_block_thread_sad_partial[tidx_3d + stride * threadsXY];
                }
                __syncthreads();
            }
            
            // Thread z=0 has complete SAD for this position
            if (threadIdx.z == 0) {
                int sad = shared_block_thread_sad_partial[tidx_3d];
                int dist = (ref_x - x) * (ref_x - x) + (ref_y - y) * (ref_y - y);
                int best_dist = best_dx * best_dx + best_dy * best_dy;
                if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                    best_sad = sad;
                    best_dx = (ref_x - x) / blockSize;
                    best_dy = (ref_y - y) / blockSize;
                }
            }
        }
    }
    
    // Write thread result to shared memory (z=0 threads only)
    if (threadIdx.z == 0) {
        shared_block_thread_sad[tidx_2d] = best_sad;
        shared_block_thread_dx[tidx_2d] = best_dx;
        shared_block_thread_dy[tidx_2d] = best_dy;
        shared_block_thread_dist[tidx_2d] = best_dx * best_dx + best_dy * best_dy;
    }
    
    __syncthreads();

    // 2D parallel reduction (only z=0 threads)
    if (threadIdx.z == 0) {
        for (int stride = 1; stride < threadsXY; stride *= 2) {
            if (tidx_2d + stride < threadsXY) {
                if (shared_block_thread_sad[tidx_2d + stride] < shared_block_thread_sad[tidx_2d] || 
                    (shared_block_thread_sad[tidx_2d + stride] == shared_block_thread_sad[tidx_2d] && 
                     shared_block_thread_dist[tidx_2d + stride] < shared_block_thread_dist[tidx_2d])) {
                    shared_block_thread_sad[tidx_2d] = shared_block_thread_sad[tidx_2d + stride];
                    shared_block_thread_dx[tidx_2d] = shared_block_thread_dx[tidx_2d + stride];
                    shared_block_thread_dy[tidx_2d] = shared_block_thread_dy[tidx_2d + stride];
                    shared_block_thread_dist[tidx_2d] = shared_block_thread_dist[tidx_2d + stride];
                }
            }
            __syncthreads();
        }
    }
    
    // Thread (0,0,0) writes final result for this block
    if (tidx_3d == 0) {
        d_mv[blockIdx.y * blocksX + blockIdx.x] = {shared_block_thread_dx[0], shared_block_thread_dy[0]};
    }
}


 vector<vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(const ImageGray& curr, const ImageGray& ref,
                                                                    int blockSize) {
    cudaSetDevice(0);
    
    // Validate input: current and reference frames must have same dimensions
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }
    
    size_t bytesPerFrame = frameSize * sizeof(unsigned char); // Grayscale: 1 byte per pixel 256 levels of gray
    
    std::cout << "CUDA: Processing " << curr.width << "x" << curr.height << " frame with block size " << blockSize << std::endl;
    
    // Allocate and copy frames to GPU
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc((void**)&d_ref, bytesPerFrame));
    
    // Copy frames to GPU
    std::cout << "CUDA: Copying frames to GPU..." << std::endl;
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    
    // Calculate grid dimensions
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    
    std::cout << "CUDA: Grid size: " << blocksX << "x" << blocksY << " = " << (blocksX*blocksY) << " blocks" << std::endl;

    // Allocate GPU memory for motion vectors
    size_t mvSize = blocksX * blocksY * sizeof(MotionVector);
    MotionVector* d_mv = nullptr;   // GPU pointer matrix for motion vectors (one per block in current frame) 
    gpuErrorCheck(cudaMalloc((void**)&d_mv, mvSize));
    
    // Determine threads per block based on the grid dimension 
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int threadsPerBlockX= blocksX;
    int threadsPerBlockY = blocksY;
    int threadsPerBlockZ;
    if (threadsPerBlockX * threadsPerBlockY <= maxThreadsPerBlock) {
        threadsPerBlockZ = min(64, maxThreadsPerBlock / (threadsPerBlockX * threadsPerBlockY)); // Use Z dimension for SAD parallelization, up to 64 threads (typical warp size) or as many as possible within max threads per block
    } else {
        // FIXME review dimensions for large blocks to bnetter optimization
        threadsPerBlockX = 16;
        threadsPerBlockY = 16;
        threadsPerBlockZ = 4;
    }
    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;

    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
              << " threads per block (" << threadsPerBlock << " total threads)..." << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    // Allocate shared memory for current block + partial SAD + reduction buffers
    // sad_partial: threadsPerBlock (for Z reduction)
    // sad/dx/dy/dist: threadsXY = threadsPerBlock / blockDim.z (for 2D reduction, z=0 threads only)
    const int threadsXY_host = (blockDim.x * blockDim.y);
    size_t sharedMemSize = blockSize * blockSize * sizeof(unsigned char) +           // Current block pixels
                           threadsPerBlock * sizeof(int) +                           // sad_partial
                           (4 * threadsXY_host) * sizeof(int);                       // sad/dx/dy/dist
    
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width, curr.height, blocksX, blocksY, threadsPerBlock);
    gpuErrorCheck(cudaGetLastError());
    
    std::cout << "CUDA: Kernel launched, synchronizing..." << std::endl;
    gpuErrorCheck(cudaDeviceSynchronize());
    
    recordCudaEvent(stop);
    float milliseconds = elapsedCudaTime(start, stop);
    std::cout << "CUDA: Kernel execution time: " << milliseconds << " ms" << std::endl;
    destroyCudaEvent(start);
    destroyCudaEvent(stop);
    
    // Copy results back to host
    MotionVector* h_mv = new MotionVector[blocksX * blocksY];
    gpuErrorCheck(cudaMemcpy(h_mv, d_mv, mvSize, cudaMemcpyDeviceToHost));
    
    // Convert flat array to 2D vector
    vector<vector<MotionVector>> result(blocksY, vector<MotionVector>(blocksX));
    for (int by = 0; by < blocksY; by++)
        for (int bx = 0; bx < blocksX; bx++)
            result[by][bx] = h_mv[by * blocksX + bx];
    
    // Cleanup
    std::cout << "CUDA: Cleaning up GPU memory..." << std::endl;
    delete[] h_mv;
    cudaFree(d_curr);
    cudaFree(d_ref);
    cudaFree(d_mv);

    std::cout << "CUDA: Complete!" << std::endl;
    return result;
}


