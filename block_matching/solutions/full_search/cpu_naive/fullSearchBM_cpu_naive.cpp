#include "fullSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

using namespace std;

// Full Search Block Matching (FSBM) for grayscale images
vector<vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                     int blockSize) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    auto start_time = chrono::high_resolution_clock::now();
    cout << "CPU Naive (Grayscale): Processing " << blocksX << "x" << blocksY << " = " << (blocksX * blocksY) << " blocks" << endl;

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            // try every possible block position in the reference frame (naive search)
            for(int irefY = 0; irefY < blocksY; irefY++) {
                for(int irefX = 0; irefX < blocksX; irefX++) {
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = refX - x;
                    int dy = refY - y;
                    int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    int dist     = dx * dx + dy * dy;
                    int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;

                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                    }
                }
            }
            mv[by][bx] = bestMV;
        }
        // Print progress every 10 rows processed
        if((by + 1) % 10 == 0 || by == blocksY - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (by + 1) * blocksX;
            int total = blocksX * blocksY;
            double percentage = (100.0 * processed) / total;
            cout << "  Progress: " << processed << "/" << total << " blocks (" << percentage << "%) - " << elapsed.count() << " s" << endl;
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_time = end_time - start_time;
    cout << "CPU Naive (Grayscale): Total processing time: " << total_time.count() << " s" << endl;
    return mv;
}
vector<vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                    int blockSize) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    auto start_time = chrono::high_resolution_clock::now();
    cout << "CPU Naive (RGB): Processing " << blocksX << "x" << blocksY << " = " << (blocksX * blocksY) << " blocks" << endl;

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            // try every possible block position in the reference frame (naive search)
            for(int irefY = 0; irefY < blocksY; irefY++) {
                for(int irefX = 0; irefX < blocksX; irefX++) {
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = refX - x;
                    int dy = refY - y;
                    int sad = computeSADRGB(curr, ref, x, y, refX, refY, blockSize);
                    int dist     = dx * dx + dy * dy;
                    int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;
                    // Keep track of motion vector with minimum SAD (best match)
                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                    }
                }
            }
            mv[by][bx] = bestMV;
        }
        // Print progress every 10 rows processed
        if((by + 1) % 10 == 0 || by == blocksY - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (by + 1) * blocksX;
            int total = blocksX * blocksY;
            double percentage = (100.0 * processed) / total;
            cout << "  Progress: " << processed << "/" << total << " blocks (" << percentage << "%) - " << elapsed.count() << " s" << endl;
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_time = end_time - start_time;
    cout << "CPU Naive (RGB): Total processing time: " << total_time.count() << " s" << endl;
    return mv;
}
