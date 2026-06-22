#ifndef FULLSEARCH_CPU_OPENMP_H
#define FULLSEARCH_CPU_OPENMP_H

#include <vector>
#include "types.h"

// Full Search Block Matching for grayscale images
std::vector<std::vector<MotionVector>>  fullSearchCPUOpenMPGray(const ImageGray& curr, const ImageGray& ref,
                                                                 int blockSize, int searchRange, SingleRunMetrics& metrics);

                                                                
#endif // FULLSEARCH_CPU_OPENMP_H
