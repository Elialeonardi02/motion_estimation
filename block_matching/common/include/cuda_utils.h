#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

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

#endif // CUDA_UTILS_H
