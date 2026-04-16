#include "fullSearchBM_cpu_naive.h"
#include <cmath>
#include <limits>

using namespace std;

// Compute SAD (Sum of Absolute Differences) for grayscale block in current and reference images
// SAD = Σ|I_curr(x1+i, y1+j) - I_ref(x2+i, y2+j)| for all (i,j) in block
int computeSAD(const ImageGray& curr, const ImageGray& ref,
               int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++)
        for(int x = 0; x < blockSize; x++)
            // Accumulate absolute differences between corresponding pixels: |pixel_curr - pixel_ref|
            sad += abs(curr.at(x1 + x, y1 + y) - ref.at(x2 + x, y2 + y));
    return sad;
}

// Compute SAD (Sum of Absolute Differences) for RGB color block
// SAD_RGB = Σ Σ |I_curr[c](x1+i, y1+j) - I_ref[c](x2+i, y2+j)| for all (i,j) in block and all channels c={0,1,2}
int computeSADRGB(const ImageColor& curr, const ImageColor& ref,
                  int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++) {
        for(int x = 0; x < blockSize; x++) {
            // Sum differences across all 3 color channels (Red, Green, Blue)
            for(int c = 0; c < 3; c++) {
                sad += abs(curr.at(x1 + x, y1 + y, c) - ref.at(x2 + x, y2 + y, c));
            }
        }
    }
    return sad;
}

// Full Search Block Matching (FSBM) for grayscale images
vector<vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                     int blockSize, int searchRange) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            // try every possible block position in the reference frame (naive search)
            for(int refY = 0; refY <= ref.height - blockSize; refY++) {
                for(int refX = 0; refX <= ref.width - blockSize; refX++) {
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = refX - x;
                    int dy = refY - y;
                    int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    // Keep track of motion vector with minimum SAD (best match)
                    if(sad < bestSAD) { bestSAD = sad; bestMV = {dx, dy}; }
                }
            }
            mv[by][bx] = bestMV;
        }
    }
    return mv;
}

// Full Search Block Matching (FSBM) for RGB color images
vector<vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                    int blockSize, int searchRange) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            // try every possible block position in the reference frame (naive search)
            for(int refY = 0; refY <= ref.height - blockSize; refY++) {
                for(int refX = 0; refX <= ref.width - blockSize; refX++) {
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = refX - x;
                    int dy = refY - y;
                    int sad = computeSADRGB(curr, ref, x, y, refX, refY, blockSize);
                    // Keep track of motion vector with minimum SAD (best match)
                    if(sad < bestSAD) { bestSAD = sad; bestMV = {dx, dy}; }
                }
            }
            mv[by][bx] = bestMV;
        }
    }
    return mv;
}
