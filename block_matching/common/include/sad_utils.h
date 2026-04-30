#ifndef SAD_UTILS_H
#define SAD_UTILS_H

#include "types.h"
#include <cstdlib>

// Compute SAD (Sum of Absolute Differences) for grayscale block in current and reference images
// SAD = Σ|I_curr(x1+i, y1+j) - I_ref(x2+i, y2+j)| for all (i,j) in block
static inline int computeSAD(const ImageGray& curr, const ImageGray& ref,
                             int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++) {
        for(int x = 0; x < blockSize; x++) {
            sad += std::abs(static_cast<int>(curr.at(x1 + x, y1 + y)) - 
                           static_cast<int>(ref.at(x2 + x, y2 + y))); // cast to int to avoid unsigned char underflow when computing absolute difference
        }
    }
    return sad;
}

// Compute SAD (Sum of Absolute Differences) for RGB color block
// SAD_RGB = Σ Σ |I_curr[c](x1+i, y1+j) - I_ref[c](x2+i, y2+j)| for all (i,j) in block and all channels c={0,1,2}
static inline int computeSADRGB(const ImageColor& curr, const ImageColor& ref,
                                int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++) {
        for(int x = 0; x < blockSize; x++) {
            // Sum differences across all 3 color channels (Red, Green, Blue)
            for(int c = 0; c < 3; c++) {
                sad += std::abs(static_cast<int>(curr.at(x1 + x, y1 + y, c)) - 
                               static_cast<int>(ref.at(x2 + x, y2 + y, c))); // cast to int to avoid unsigned char underflow when computing absolute difference
            }
        }
    }
    return sad;
}

#endif // SAD_UTILS_H
