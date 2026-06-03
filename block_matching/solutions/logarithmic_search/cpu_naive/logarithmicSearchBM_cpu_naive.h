#ifndef LOGARITHMICSEARCH_CPU_NAIVE_H
#define LOGARITHMICSEARCH_CPU_NAIVE_H

#include <vector>
#include "types.h"

// Logarithmic Search Block Matching for grayscale images
std::vector<std::vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize, int distance, SingleRunMetrics& metrics);


#endif // LOGARITHMICSEARCH_CPU_NAIVE_H
