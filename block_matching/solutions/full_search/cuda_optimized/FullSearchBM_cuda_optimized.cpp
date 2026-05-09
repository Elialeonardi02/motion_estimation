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

// Device function to compute PARTIAL SAD for a subset of pixels
// Each thread along Z (if is not 1) dimension computes SAD for a portion of pixels, then results are reduced
static __device__ int computeSAD_device_partial(const unsigned char* curr, const unsigned char* ref,
                                                int startPixel, int pixelsPerThread, int totalPixels) {
    int sad = 0;
    for (int i = startPixel; i < min(startPixel+pixelsPerThread, totalPixels); i++) {  // to avoid out of bounds access for threads that have fewer pixels to process  
        sad += abs (curr[i] - ref[i]);
    }
    return sad;
}


// CUDA kernel: each grid block processes one search block of current frame
__global__ void fullSearchKernel(const unsigned char* d_curr, const unsigned char* d_ref,
                                 MotionVector* d_mv, int blockSize, int width) {
    
    
    // Thread indices
    // const int tidx_2d = (threadIdx.y * blockDim.x) + threadIdx.x;
    // const int tidx_3d = tidx_2d + (threadIdx.z * blockDim.x * blockDim.y);
    
    // Current block top-left corner
    const int x = blockIdx.x * blockSize;
    const int y = blockIdx.y * blockSize;
    const int total_positions = gridDim.x * gridDim.y; // total candidate positions in reference frame (one per block in reference frame)
    
    const int pixelsTotal     = blockSize * blockSize;
    const int pixelsPerThread = (pixelsTotal + blockDim.x - 1) / blockDim.x; // to ensure all pixels are processed even if pixelsTotal is not perfectly divisible by blockDim.x

    // Shared memory layout: current block + SAD partial + reduction buffers
    extern __shared__ unsigned char shared_memory[];
    unsigned char* s_curr = shared_memory;
    unsigned char* s_ref = s_curr + blockSize * blockSize;
    int * s_partial_sad = (int*)(s_ref + blockSize * blockSize);
    
    // Load current block in shared memory
    for (int i = threadIdx.x; i < pixelsTotal; i += blockDim.x) {
        int py = i / blockSize;
        int px = i % blockSize;
        s_curr[i] = d_curr[(y + py) * width + (x + px)];
    }

    __syncthreads();
    
    int best_sad = INT_MAX; // Initialize best SAD with worst case
    int best_dx = 0;
    int best_dy = 0;
    int best_dist = INT_MAX; // for tie-breaking

    for (int pos=0; pos < total_positions; pos++) {
        const int ref_x = (pos % gridDim.x) * blockSize;
        const int ref_y = (pos / gridDim.x) * blockSize;
        
        // Load reference block in shared memory for this position in a coalesced manner 
        for (int i = threadIdx.x; i < pixelsTotal; i += blockDim.x) {
            int py = i / blockSize;
            int px = i % blockSize;
            s_ref[i] = d_ref[(ref_y + py) * width + (ref_x + px)];
        }
        __syncthreads(); 
        s_partial_sad[threadIdx.x]= computeSAD_device_partial(s_curr, s_ref, threadIdx.x * pixelsPerThread, pixelsPerThread, pixelsTotal);

        __syncthreads(); 

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_partial_sad[threadIdx.x] += s_partial_sad[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int sad  = s_partial_sad[0];
            int dx   = (ref_x - x) / blockSize;
            int dy   = (ref_y - y) / blockSize;
            int dist = dx * dx + dy * dy;
            if (sad < best_sad || (sad == best_sad && dist < best_dist)) {
                best_sad  = sad;
                best_dx   = dx;
                best_dy   = dy;
                best_dist = dist;
            }
        }

    }
    if (threadIdx.x == 0)
        d_mv[blockIdx.y * gridDim.x + blockIdx.x] = {best_dx, best_dy};
}


vector<vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
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
    const int pixelsTotal = blockSize * blockSize;
    int threadsPerBlock   = 32;

    for (int t = min(prop.maxThreadsPerBlock, 1024); t >= 32; t >>= 1) {
        size_t smem = 2 * pixelsTotal * sizeof(unsigned char)
                    + t * sizeof(int);
        if (smem <= prop.sharedMemPerBlock) {
            threadsPerBlock = t;
            break;
        }
    }

    size_t sharedMemSize = 2 * pixelsTotal * sizeof(unsigned char)
                         + threadsPerBlock * sizeof(int);

    dim3 gridDim(blocksX, blocksY);
    dim3 blockDim(threadsPerBlock, 1, 1);
    
    std::cout << "CUDA: Launching kernel with " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z
              << " threads per block (" << threadsPerBlock << " total threads)..." << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    createCudaEvent(start);
    createCudaEvent(stop);
    recordCudaEvent(start);
    
    fullSearchKernel<<<gridDim, blockDim, sharedMemSize>>>(d_curr, d_ref, d_mv, blockSize,
                                            curr.width);
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


