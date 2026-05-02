#include "logarithmicSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

using namespace std;

// Logarithmic Search Block Matching (LSBM) for grayscale images
vector<vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                           int blockSize, int distance) {
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
            int cx = x;
            int cy = y;
            // Logarithmic search: start with the initial distance and keep halving it until it becomes 0
            for (int currentDistance = distance; currentDistance > 0; currentDistance /= 2) {
                // Check the 8 points around the block in the current frame at the current distance 
                // plus the center point (0,0) which is the current block position in the reference frame
                // (-d,d), (0,d), (d,d), 
                // (-d,0), (0,0), (d,0), 
                // (-d,-d), (0,-d), (d,-d)
                int iterBestSAD = numeric_limits<int>::max();
                int iterBestDX  = 0;
                int iterBestDY  = 0;
                for (int dy = -currentDistance; dy <= currentDistance; dy += currentDistance) {
                    for (int dx = -currentDistance; dx <= currentDistance; dx += currentDistance) {
                        // compute the displacement (dx, dy) from current block to candidate block
                        int refX = cx + dx*blockSize;
                        int refY = cy + dy*blockSize;  
                        // Check if candidate block is within the reference frame boundaries
                        if (refX >= 0 && refX + blockSize <= ref.width && refY >= 0 && refY + blockSize <= ref.height) {
                            int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                            int dist = dx * dx + dy * dy;
                            int iterDist = iterBestDX * iterBestDX + iterBestDY * iterBestDY; 
                            if (sad < iterBestSAD || (sad == iterBestSAD && dist < iterDist)) { 
                                iterBestSAD = sad; 
                                iterBestDX  = dx;  
                                iterBestDY  = dy;
                            }
                        }
                    }
                }
                // Update the center point for the next iteration to be the best match found in this iteration
                cx += iterBestDX * blockSize;
                cy += iterBestDY * blockSize;
            }
        /* TODO local search in the pixex neighborhood of the final position found by the logarithmic search to refine the motion vector
            it is necessary? 
        int refBestSAD = numeric_limits<int>::max();
        int refBestDX  = 0;
        int refBestDY  = 0;
        for (int dy = -(blockSize-1); dy <= (blockSize-1); dy++) {
            for (int dx = -(blockSize-1); dx <= (blockSize-1); dx++) {
                int refX = cx + dx;
                int refY = cy + dy;
                if (refX >= 0 && refX + blockSize <= ref.width &&
                    refY >= 0 && refY + blockSize <= ref.height) {
                    int sad  = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                    int dist     = dx * dx + dy * dy;
                    int refDist  = refBestDX * refBestDX + refBestDY * refBestDY;
                    if (sad < refBestSAD || (sad == refBestSAD && dist < refDist)) {
                        refBestSAD = sad;
                        refBestDX  = dx;
                        refBestDY  = dy;
                    }
                }
            }
        }
        // Calculate the final motion vector from initial position to final position
        // Return motion vector in PIXELS to be consistent with full search output
        cx += refBestDX; 
        cy += refBestDY;
        */
        mv[by][bx] = {(cx - x), (cy - y)};
        
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

