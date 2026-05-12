#ifndef FULLSEARCH_CPU_NAIVE_H
#define FULLSEARCH_CPU_NAIVE_H

#include <vector>
#include "types.h"

// Full Search Block Matching for grayscale images
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize, int searchRange);

// Full Search Block Matching for RGB color images
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                              int blockSize, int searchRange);

#endif // FULLSEARCH_CPU_NAIVE_H
