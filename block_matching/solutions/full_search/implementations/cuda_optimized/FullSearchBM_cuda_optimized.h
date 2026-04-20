#ifndef FULLSEARCHBM_CUDA_OPTIMIZED_H
#define FULLSEARCHBM_CUDA_OPTIMIZED_H

#include <vector>
#include <cuda_runtime.h>
#include "types.h"



std::vector<std::vector<MotionVector>> fullSearchCUDAOptimizedGray(const ImageGray& curr, const ImageGray& ref, 
                                                                    int blockSize);

#endif // FULLSEARCHBM_CUDA_OPTIMIZED_H
