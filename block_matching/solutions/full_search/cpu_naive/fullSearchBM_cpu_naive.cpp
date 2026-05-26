#include "fullSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include "utils.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

using namespace std;

// Full Search Block Matching for grayscale images
vector<vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                     int blockSize, int searchRange) {
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    vector<vector<MotionVector>> mv = GridUtils::createMotionVectorGrid(blocksX, blocksY);

    auto start_time = chrono::high_resolution_clock::now();
    
    LoggingUtils::printFrameInfo("CPU Naive", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CPU Naive", searchRange);

    for(int bx = 0; bx < blocksX; bx++) {
        for(int by = 0; by < blocksY; by++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            
            SearchRangeUtils::SearchBounds bounds = SearchRangeUtils::calculateSearchBounds(bx, by, blocksX, blocksY, searchRange);
            
            for(int irefY = bounds.startY; irefY <= bounds.endY; irefY++) {
                for(int irefX = bounds.startX; irefX <= bounds.endX; irefX++) {
                    int refX = irefX * blockSize; // refX and refY are the top-left corner of the candidate block in the reference frame
                    int refY = irefY * blockSize;
                    int dx = irefX - bx;
                    int dy = irefY - by;
                    int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    int dist = dx * dx + dy * dy;
                    int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;
                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                    }
                }
            }
            mv[bx][by] = bestMV;
        }
        // Progress every 10 columns
        if((bx + 1) % 10 == 0 || bx == blocksX - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (bx + 1) * blocksY;
            int total = blocksX * blocksY;
            LoggingUtils::printProgressUpdate("CPU Naive", processed, total, elapsed.count());
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_time = end_time - start_time;
    LoggingUtils::printTimingReport("CPU Naive", total_time.count());
    return mv;
}

// Full Search Block Matching for RGB color images
vector<vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                    int blockSize, int searchRange) {
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    vector<vector<MotionVector>> mv = GridUtils::createMotionVectorGrid(blocksX, blocksY);

    auto start_time = chrono::high_resolution_clock::now();
    
    LoggingUtils::printFrameInfo("CPU Naive RGB", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CPU Naive RGB", searchRange);

    for(int bx = 0; bx < blocksX; bx++) {
        for(int by = 0; by < blocksY; by++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            
            SearchRangeUtils::SearchBounds bounds = 
                SearchRangeUtils::calculateSearchBounds(bx, by, blocksX, blocksY, searchRange);
            
            for(int irefY = bounds.startY; irefY <= bounds.endY; irefY++) {
                for(int irefX = bounds.startX; irefX <= bounds.endX; irefX++) {
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    int dx = irefX - bx;
                    int dy = irefY - by;
                    int sad = computeSADRGB(curr, ref, x, y, refX, refY, blockSize);
                    int dist = dx * dx + dy * dy;
                    int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;
                    // Keep track of motion vector with minimum SAD (best match)
                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                    }
                }
            }
            mv[bx][by] = bestMV;
        }
        // Progress every 10 columns
        if((bx + 1) % 10 == 0 || bx == blocksX - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (bx + 1) * blocksY;
            int total = blocksX * blocksY;
            LoggingUtils::printProgressUpdate("CPU Naive RGB", processed, total, elapsed.count());
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_time = end_time - start_time;
    LoggingUtils::printTimingReport("CPU Naive RGB", total_time.count());
    return mv;
}
