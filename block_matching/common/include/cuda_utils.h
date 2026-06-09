#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <stdexcept>
#include <string>
#include "types.h"

// CUDA SEARCH BOUNDS STRUCTURE
struct CudaSearchBounds {
  int startX;
  int endX;
  int startY;
  int endY;
  int width;
  int height;
  int totalPositions;
};

// Device function to calculate search bounds in CUDA kernels
__device__ inline CudaSearchBounds calculateCudaSearchBounds(int blockIdxX, int blockIdxY, 
                                                              int gridDimX, int gridDimY, 
                                                              int searchRange) {
  CudaSearchBounds bounds;
  
  if (searchRange > 0) {
    bounds.startX = max(0, blockIdxX - searchRange);
    bounds.endX = min(gridDimX - 1, blockIdxX + searchRange);
    bounds.startY = max(0, blockIdxY - searchRange);
    bounds.endY = min(gridDimY - 1, blockIdxY + searchRange);
  } else {
    bounds.startX = 0;
    bounds.endX = gridDimX - 1;
    bounds.startY = 0;
    bounds.endY = gridDimY - 1;
  }
  
  bounds.width = bounds.endX - bounds.startX + 1;
  bounds.height = bounds.endY - bounds.startY + 1;
  bounds.totalPositions = bounds.width * bounds.height;
  
  return bounds;
}

inline void gpuErrorCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA error occurred");
    }
}

// Utility cuda event functions for timing GPU operations
inline void createCudaEvent(cudaEvent_t& event) {
    gpuErrorCheck(cudaEventCreate(&event));
}

inline void destroyCudaEvent(cudaEvent_t& event) {
    gpuErrorCheck(cudaEventDestroy(event));
}

inline void recordCudaEvent(cudaEvent_t& event) {
    gpuErrorCheck(cudaEventRecord(event));
}

inline float elapsedCudaTime(cudaEvent_t& start, cudaEvent_t& stop) {
    float ms = 0.0f;
    gpuErrorCheck(cudaEventSynchronize(stop));
    gpuErrorCheck(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// GPU MEMORY UTILITIES
namespace GPUMemoryUtils {
  // Allocate GPU memory for current and reference frames in GMEM
  void allocateFrames(const ImageGray& curr, const ImageGray& ref,
                     unsigned char*& d_curr, unsigned char*& d_ref,
                     size_t& bytesPerFrame);

  // Copy frames from host to GPU GMEM
  void copyFramesToGPU(unsigned char* d_curr, unsigned char* d_ref,
                      const ImageGray& curr, const ImageGray& ref,
                      size_t bytesPerFrame);

  // Copy motion vectors from GPU GMEM to host
  void copyMotionVectorsFromGPU(std::vector<MotionVector>& h_mv_flat,
                                MotionVector* d_mv,
                                int blocksX, int blocksY);

  // Free GPU GMEM
  void freeMemory(unsigned char* d_curr, unsigned char* d_ref,
                 MotionVector* d_mv);

  // Free GPU GMEM with additional arrays
  void freeMemory(unsigned char* d_curr, unsigned char* d_ref,
                 int* d_sad, MotionVector* d_mv);
}

// CUDA TIMER CLASS
class CudaTimer {
private:
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  std::string name;
  bool isRecording;

public:
  // Creates CUDA events
  CudaTimer(const std::string& timerName = "Kernel");

  // Cleans up CUDA events
  ~CudaTimer();

  // Start timing
  void start();

  // Stop timing and return elapsed time in milliseconds
  float stop();

  // Get the name of the timer
  const std::string& getName() const { return name; }

  // Reset the timer
  void reset();
};

namespace CudaTimingUtils {
  // Print timing report for a CUDA kernel
  void printKernelTiming(const std::string& kernelName, float milliseconds);
}

#endif // CUDA_UTILS_H
