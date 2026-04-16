#ifndef FULLSEARCHBM_CUDA_NAIVE_H
#define FULLSEARCHBM_CUDA_NAIVE_H

#include <vector>
#include "types.h"

// CUDA error type forward declaration
typedef int cudaError_t;

// CUDA error checking function
void gpuErrorCheck(cudaError_t error);

// Full search block matching on GPU (naive CUDA implementation)
// Computes motion vectors between current and reference frames
//
// Parameters:
// - curr: Current frame (ImageGray)
// - ref: Reference frame (ImageGray)
// - blockSize: Size of the block (blockSize x blockSize)
// - searchRange: Search range for motion estimation
//
// Returns:
// - 2D vector of MotionVector structures (height x width)
std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                                int blockSize, int searchRange);

#endif // FULLSEARCHBM_CUDA_NAIVE_H
