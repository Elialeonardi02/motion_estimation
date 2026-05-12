#ifndef FULLSEARCHBM_CUDA_NAIVE_H
#define FULLSEARCHBM_CUDA_NAIVE_H

#include <vector>
#include <cuda_runtime.h>
#include "types.h"

// Full search block matching on GPU (naive CUDA implementation)
// Computes motion vectors between current and reference frames
//
// Parameters:
// - curr: Current frame (ImageGray)
// - ref: Reference frame (ImageGray)
// - blockSize: Size of the block (blockSize x blockSize)
//
// Returns:
// - 2D vector of MotionVector structures (height x width)

// Declaration - implementation is in FullSearchBM_cuda_naive.cpp
std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                                int blockSize, int searchRange);

#endif // FULLSEARCHBM_CUDA_NAIVE_H
