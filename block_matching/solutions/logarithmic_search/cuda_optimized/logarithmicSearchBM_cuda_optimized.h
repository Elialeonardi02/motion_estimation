#ifndef LOGARITHMICSEARCHBM_CUDA_OPTIMIZED_H
#define LOGARITHMICSEARCHBM_CUDA_OPTIMIZED_H

#include <vector>
#include <cuda_runtime.h>
#include "types.h"



std::vector<std::vector<MotionVector>> logarithmicSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                                    int blockSize, int searchDistance, SingleRunMetrics& metrics);

#endif // LOGARITHMICSEARCHBM_CUDA_OPTIMIZED_H
