#include "fullSearchBM_cpu_naive.h"
#include <cmath>
#include <limits>

using namespace std;

// Compute SAD (Sum of Absolute Differences) for grayscale block
int computeSAD(const ImageGray& curr, const ImageGray& ref,
               int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++)
        for(int x = 0; x < blockSize; x++)
            sad += abs(curr.at(x1 + x, y1 + y) - ref.at(x2 + x, y2 + y));
    return sad;
}

// Compute SAD (Sum of Absolute Differences) for RGB color block
int computeSADRGB(const ImageColor& curr, const ImageColor& ref,
                  int x1, int y1, int x2, int y2, int blockSize) {
    int sad = 0;
    for(int y = 0; y < blockSize; y++) {
        for(int x = 0; x < blockSize; x++) {
            for(int c = 0; c < 3; c++) {
                sad += abs(curr.at(x1 + x, y1 + y, c) - ref.at(x2 + x, y2 + y, c));
            }
        }
    }
    return sad;
}

// Full Search Block Matching (FSBM) for grayscale images (CPU Naive)
vector<vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                     int blockSize, int searchRange) {
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            for(int dy = -searchRange; dy <= searchRange; dy++) {
                for(int dx = -searchRange; dx <= searchRange; dx++) {
                    int refX = x + dx, refY = y + dy;
                    if(refX < 0 || refY < 0 || refX + blockSize >= ref.width || refY + blockSize >= ref.height) continue;
                    int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    if(sad < bestSAD) { bestSAD = sad; bestMV = {dx, dy}; }
                }
            }
            mv[by][bx] = bestMV;
        }
    }
    return mv;
}

// Full Search Block Matching (FSBM) for RGB color images (CPU Naive)
vector<vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                    int blockSize, int searchRange) {
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            for(int dy = -searchRange; dy <= searchRange; dy++) {
                for(int dx = -searchRange; dx <= searchRange; dx++) {
                    int refX = x + dx, refY = y + dy;
                    if(refX < 0 || refY < 0 || refX + blockSize >= ref.width || refY + blockSize >= ref.height) continue;
                    int sad = computeSADRGB(curr, ref, x, y, refX, refY, blockSize);
                    if(sad < bestSAD) { bestSAD = sad; bestMV = {dx, dy}; }
                }
            }
            mv[by][bx] = bestMV;
        }
    }
    return mv;
}
