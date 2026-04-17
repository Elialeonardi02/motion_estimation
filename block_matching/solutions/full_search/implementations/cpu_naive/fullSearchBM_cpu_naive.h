#ifndef FULLSEARCH_CPU_NAIVE_H
#define FULLSEARCH_CPU_NAIVE_H

#include <vector>
#include "types.h"

// Compute SAD (Sum of Absolute Differences) for grayscale block
int computeSAD(const ImageGray& curr, const ImageGray& ref,
               int x1, int y1, int x2, int y2, int blockSize);

// Compute SAD (Sum of Absolute Differences) for RGB color block
int computeSADRGB(const ImageColor& curr, const ImageColor& ref,
                  int x1, int y1, int x2, int y2, int blockSize);

// Full Search Block Matching for grayscale images
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize);

// Full Search Block Matching for RGB color images
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                              int blockSize);

#endif // FULLSEARCH_CPU_NAIVE_H
