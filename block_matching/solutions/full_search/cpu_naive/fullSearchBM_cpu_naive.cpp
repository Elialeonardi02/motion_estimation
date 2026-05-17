#include "fullSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

using namespace std;

// Full Search Block Matching for grayscale images
vector<vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                     int blockSize, int searchRange) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    auto start_time = chrono::high_resolution_clock::now();
    
        string searchModeStr = (searchRange > 0) ? ("Range search (range=" + to_string(searchRange) + " blocks)") : "Full search";
        cout << "CPU Naive (Grayscale): Processing frame " << curr.width << "x" << curr.height
            << " with block size " << blockSize << endl;
        cout << "CPU Naive (Grayscale): Grid size: " << blocksX << "x" << blocksY
            << " = " << (blocksX * blocksY) << " blocks" << endl;
        cout << "CPU Naive (Grayscale): Search mode: " << searchModeStr
            << " (searchRange=" << searchRange << ")" << endl;

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            /*Search bound for reference blocks on search_range (if searchRange > 0) or full frame (if searchRange <= 0)
                If searchRange > 0, limit reference block positions to a square region around the current block:
                    -irefX ∈ [max(0, bx - searchRange), min(blocksX - 1, bx + searchRange)]
                    -irefY ∈ [max(0, by - searchRange), min(blocksY - 1, by + searchRange)]
                    This creates a (2*searchRange + 1) x (2*searchRange + 1) block search area centered on the current block.
                If searchRange <= 0, search all blocks in the reference frame:
                    -irefX ∈ [0, blocksX - 1]
                    -irefY ∈ [0, blocksY - 1]
                    This creates a full frame search area.
            */
            for(int irefY = (searchRange > 0 ? max(0, by - searchRange) : 0); irefY < (searchRange > 0 ? min(blocksY, by + searchRange+1) : blocksY); irefY++) {
                for(int irefX = (searchRange > 0 ? max(0, bx - searchRange) : 0); irefX < (searchRange > 0 ? min(blocksX, bx + searchRange+1) : blocksX); irefX++) {
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = (refX - x) / blockSize;
                    int dy = (refY - y) / blockSize;
                    int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    int dist = dx * dx + dy * dy;
                    int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;
                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                    }
                }
            }
            mv[by][bx] = bestMV;
        }
        // Progress every 10 rows
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
    cout << "CPU Naive (Grayscale): Timing:" << endl;
    cout << "  Total processing time: " << total_time.count() << " s" << endl;
    return mv;
}

// Full Search Block Matching for RGB color images
vector<vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                    int blockSize, int searchRange) {
    // Grid of blocks: blocksX = ⌊width / blockSize⌋, blocksY = ⌊height / blockSize⌋
    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    vector<vector<MotionVector>> mv(blocksY, vector<MotionVector>(blocksX));

    auto start_time = chrono::high_resolution_clock::now();
    
        string searchModeStr = (searchRange > 0) ? ("Range search (range=" + to_string(searchRange) + " blocks)") : "Full search";
        cout << "CPU Naive (RGB): Processing frame " << curr.width << "x" << curr.height
            << " with block size " << blockSize << endl;
        cout << "CPU Naive (RGB): Grid size: " << blocksX << "x" << blocksY
            << " = " << (blocksX * blocksY) << " blocks" << endl;
        cout << "CPU Naive (RGB): Search mode: " << searchModeStr
            << " (searchRange=" << searchRange << ")" << endl;

    for(int by = 0; by < blocksY; by++) {
        for(int bx = 0; bx < blocksX; bx++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            /*Search bound for reference blocks on search_range (if searchRange > 0) or full frame (if searchRange <= 0)
                If searchRange > 0, limit reference block positions to a square region around the current block:
                    -irefX ∈ [max(0, bx - searchRange), min(blocksX - 1, bx + searchRange)]
                    -irefY ∈ [max(0, by - searchRange), min(blocksY - 1, by + searchRange)]
                    This creates a (2*searchRange + 1) x (2*searchRange + 1) block search area centered on the current block.
                If searchRange <= 0, search all blocks in the reference frame:
                    -irefX ∈ [0, blocksX - 1]
                    -irefY ∈ [0, blocksY - 1]
                    This creates a full frame search area.
            */
            for(int irefY = (searchRange > 0 ? max(0, by - searchRange) : 0); irefY < (searchRange > 0 ? min(blocksY, by + searchRange+1) : blocksY); irefY++) {
                for(int irefX = (searchRange > 0 ? max(0, bx - searchRange) : 0); irefX < (searchRange > 0 ? min(blocksX, bx + searchRange+1) : blocksX); irefX++) {
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    // Compute displacement (dx, dy) from current block to candidate block
                    int dx = (refX - x) / blockSize;
                    int dy = (refY - y) / blockSize;
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
            mv[by][bx] = bestMV;
        }
        // Progress every 10 rows
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
    cout << "CPU Naive (RGB): Timing:" << endl;
    cout << "  Total processing time: " << total_time.count() << " s" << endl;
    return mv;
}
