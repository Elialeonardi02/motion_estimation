#ifndef LOGARITHMICSEARCH_CPU_NAIVE_H
#define LOGARITHMICSEARCH_CPU_NAIVE_H

#include <vector>
#include "types.h"

// Logarithmic Search Block Matching for grayscale images
std::vector<std::vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize);

// Logarithmic Search Block Matching for RGB color images
std::vector<std::vector<MotionVector>> logarithmicSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                                    int blockSize);

#endif // LOGARITHMICSEARCH_CPU_NAIVE_H
