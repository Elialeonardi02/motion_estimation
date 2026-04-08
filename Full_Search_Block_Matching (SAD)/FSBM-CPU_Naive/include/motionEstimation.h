#ifndef MOTIONESTIMATION_H
#define MOTIONESTIMATION_H

#include <vector>
#include "types.h"

// Compute SAD (Sum of Absolute Differences) for grayscale block
int computeSAD(const ImageGray& curr, const ImageGray& ref,
               int x1, int y1, int x2, int y2, int blockSize);

// Compute SAD (Sum of Absolute Differences) for RGB color block
int computeSADRGB(const ImageColor& curr, const ImageColor& ref,
                  int x1, int y1, int x2, int y2, int blockSize);

// Full Search Block Matching (FSBM) for grayscale images
std::vector<std::vector<MotionVector>> fsbm(const ImageGray& curr, const ImageGray& ref,
                                            int blockSize, int searchRange);

// Full Search Block Matching (FSBM) for RGB color images
std::vector<std::vector<MotionVector>> fsbmRGB(const ImageColor& curr, const ImageColor& ref,
                                               int blockSize, int searchRange);

#endif // MOTIONESTIMATION_H
