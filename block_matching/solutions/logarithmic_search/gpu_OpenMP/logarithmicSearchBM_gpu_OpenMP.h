#ifndef LOGARITHMICSEARCHBM_GPU_OPENMP_H
#define LOGARITHMICSEARCHBM_GPU_OPENMP_H

#include <vector>
#include <cuda_runtime.h>
#include "types.h"



std::vector<std::vector<MotionVector>> logarithmicSearchGPUOpenMP(const ImageGray& curr, const ImageGray& ref, 
                                                                    int blockSize, int searchDistance, SingleRunMetrics& metrics);

#endif // LOGARITHMICSEARCHBM_GPU_OPENMP_H
