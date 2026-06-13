#include "logarithmicSearchBM_cpu_naive.h"
#include "sad_utils.h"
#include "utils.h"
#include <cmath>
#include <climits>
#include <iostream>
#include <chrono>

using namespace std;

// Logarithmic search block matching for grayscale images
vector<vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                           int blockSize, int distance, SingleRunMetrics& metrics) {
    auto start_time = chrono::high_resolution_clock::now();
    // Grid of blocks
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);
    vector<vector<MotionVector>> mv = GridUtils::createMotionVectorGrid(blocksX, blocksY);

    
    
    LoggingUtils::printFrameInfo("CPU Naive Logarithmic", curr.width, curr.height, blockSize, blocksX, blocksY);
    cout << "CPU Naive Logarithmic (Grayscale): Search strategy: Logarithmic search (distance="
        << distance << " blocks)" << endl;

    for(int bx = 0; bx < blocksX; bx++) {
        for(int by = 0; by < blocksY; by++) {
            // Top-left corner of current block: (x, y) = (bx * blockSize, by * blockSize)
            int x = bx * blockSize;
            int y = by * blockSize;
            // center point of the search in the reference frame, initialized to the same position as the current block
            int cx = bx; 
            int cy = by; 
            int bestCx = cx;
            int bestCy = cy;
            // Logarithmic search: start with the initial distance and keep halving it until it becomes 1
            for (int currentDistance = distance; currentDistance > 0; currentDistance /= 2) {
                // Check the 8 points around the block in the current frame at the current distance 
                // plus the center point (0,0) which is the current block position in the reference frame
                // (-d,d), (0,d), (d,d), 
                // (-d,0), (0,0), (d,0), 
                // (-d,-d), (0,-d), (d,-d)
                int iterBestSAD = INT_MAX;
                int iterBestDist = INT_MAX;
                for (int ry = -currentDistance; ry <= currentDistance; ry += currentDistance) {
                    for (int rx = -currentDistance; rx <= currentDistance; rx += currentDistance) {
                        int cand_bx = cx + rx ; // candidate block's x in the reference frame
                        int cand_by = cy + ry ; // candidate block's y in the reference frame
                         
                        int ref_x = cand_bx * blockSize; // candidate block's top-left corner in the reference frame pixels
                        int ref_y = cand_by * blockSize; // candidate block's top-left corner in the reference frame pixels
                        // Check bounds
                        if (ref_x >= 0 && ref_x + blockSize <= ref.width && 
                            ref_y >= 0 && ref_y + blockSize <= ref.height) {  
                            int sad = computeSAD(curr, ref, x, y, ref_x, ref_y, blockSize);
                            int dist = (cand_bx - bx) * (cand_bx - bx) + (cand_by - by) * (cand_by - by); 
                            if (sad < iterBestSAD || (sad == iterBestSAD && dist < iterBestDist)) {
                                iterBestSAD = sad;
                                iterBestDist = dist;
                                bestCx = cand_bx;
                                bestCy = cand_by;
                            }
                        }
                    }
                }
                // Update the center point for the next iteration to be the best match found in this iteration
                cx = bestCx;
                cy = bestCy;
            }
            
            mv[by][bx] = {bestCx - bx, bestCy - by};
        }
        
        // Progress every 10 columns
        if((bx + 1) % 10 == 0 || bx == blocksX - 1) {
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = current_time - start_time;
            int processed = (bx + 1) * blocksY;
            int total = blocksX * blocksY;
            LoggingUtils::printProgressUpdate("CPU Naive Logarithmic", processed, total, elapsed.count());
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    metrics.total_ms = total_time;
    
    LoggingUtils::printProcessingComplete("CPU naive");
    
    return mv;
}

