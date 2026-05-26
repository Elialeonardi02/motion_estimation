#include "logarithmicSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include "utils.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <chrono>

using namespace std;

// Logarithmic search block matching for grayscale images
vector<vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                           int blockSize, int distance) {
    // Grid of blocks
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    vector<vector<MotionVector>> mv = GridUtils::createMotionVectorGrid(blocksX, blocksY);

    auto start_time = chrono::high_resolution_clock::now();
    
    LoggingUtils::printFrameInfo("CPU Naive (Logarithmic)", curr.width, curr.height, blockSize, blocksX, blocksY);
    cout << "CPU Naive (Logarithmic Search): Search strategy: Logarithmic search (distance="
        << distance << " blocks)" << endl;

    for(int bx = 0; bx < blocksX; bx++) {
        for(int by = 0; by < blocksY; by++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            int cx = x;
            int cy = y;
            // Logarithmic search: start with the initial distance and keep halving it until it becomes 1
            for (int currentDistance = distance; currentDistance > 0; currentDistance /= 2) {
                // Check the 8 points around the block in the current frame at the current distance 
                // plus the center point (0,0) which is the current block position in the reference frame
                // (-d,d), (0,d), (d,d), 
                // (-d,0), (0,0), (d,0), 
                // (-d,-d), (0,-d), (d,-d)
                int iterBestSAD = numeric_limits<int>::max();
                int iterBestDX = 0;
                int iterBestDY = 0;
                
                for (int dy = -currentDistance; dy <= currentDistance; dy += currentDistance) {
                    for (int dx = -currentDistance; dx <= currentDistance; dx += currentDistance) {
                        int refX = cx + dx * blockSize;
                        int refY = cy + dy * blockSize;
                        
                        // Check bounds
                        if (refX >= 0 && refX + blockSize <= ref.width && 
                            refY >= 0 && refY + blockSize <= ref.height) {
                            int sad = computeSAD(curr, ref, x, y, refX, refY, blockSize);
                            int dist = dx * dx + dy * dy;
                            int iterDist = iterBestDX * iterBestDX + iterBestDY * iterBestDY;
                            
                            if (sad < iterBestSAD || (sad == iterBestSAD && dist < iterDist)) {
                                iterBestSAD = sad;
                                iterBestDX = dx;
                                iterBestDY = dy;
                            }
                        }
                    }
                }
                // Update the center point for the next iteration to be the best match found in this iteration
                cx += iterBestDX * blockSize;
                cy += iterBestDY * blockSize;
            }
            
            mv[bx][by] = {(cx - x) / blockSize, (cy - y) / blockSize};
        }
        
        // Progress every 10 columns
        if((bx + 1) % 10 == 0 || bx == blocksX - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (bx + 1) * blocksY;
            int total = blocksX * blocksY;
            LoggingUtils::printProgressUpdate("CPU Naive (Logarithmic)", processed, total, elapsed.count());
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> total_time = end_time - start_time;
    LoggingUtils::printTimingReport("CPU Naive (Logarithmic)", total_time.count());
    
    return mv;
}

