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
            int bestSAD = numeric_limits<int>::max();
            MotionVector bestMV{0, 0};
            // TODO - Implement logarithmic search pattern instead of full search
            for (int currentDistance = distance; currentDistance > 0; currentDistance /= 2) {
                // Check the 8 points around the block in the current frame at the current distance 
                // plus the center point (0,0) which is the current block position in the reference frame
                // (-d,d), (0,d), (d,d), 
                // (-d,0), (0,0), (d,0), 
                // (-d,-d), (0,-d), (d,-d)
                for (int dy = -currentDistance; dy <= currentDistance; dy += currentDistance) {
                    for (int dx = -currentDistance; dx <= currentDistance; dx += currentDistance) {
                        // compute the displacement (dx, dy) from current block to candidate block
                        int refX = cx + dx*blockSize;
                        int refY = cy + dy*blockSize;  
                        // Check if candidate block is within the reference frame boundaries
                        if (refX >= 0 && refX + blockSize <= ref.width && refY >= 0 && refY + blockSize <= ref.height) {
                            int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                            int dist = dx * dx + dy * dy;
                            int bestDist = bestMV.dx * bestMV.dx + bestMV.dy * bestMV.dy;
                            if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                                bestSAD = sad;
                                bestMV = {dx, dy};
                            }
                        }
                    }
                }
                // Update the center point for the next iteration to be the best match found in this iteration
                cx += bestMV.dx * blockSize;
                cy += bestMV.dy * blockSize;
            }
            // Calculate the final motion vector from initial position to final position
            int finalMVX = (cx - x) / blockSize;
            int finalMVY = (cy - y) / blockSize;
            mv[by][bx] = {finalMVX, finalMVY};
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

