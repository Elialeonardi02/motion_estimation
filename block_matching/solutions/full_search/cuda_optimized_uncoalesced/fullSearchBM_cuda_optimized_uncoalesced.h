#ifndef FULLSEARCHBM_CUDA_OPTIMIZED_UNCOALESCED_H
#define FULLSEARCHBM_CUDA_OPTIMIZED_UNCOALESCED_H

#include <vector>
#include <cuda_runtime.h>
#include "types.h"

std::vector<std::vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(const ImageGray& curr, const ImageGray& ref,
                                                                              int blockSize, int searchRange, SingleRunMetrics& metrics);

#endif // FULLSEARCHBM_CUDA_OPTIMIZED_UNCOALESCED_H
