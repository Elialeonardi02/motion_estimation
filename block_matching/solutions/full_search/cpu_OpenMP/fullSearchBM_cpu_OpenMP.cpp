#include "fullSearchBM_cpu_OpenMP.h"
#include "sad_utils.h"
#include "utils.h"
#include "types.h"
#include <cmath>
#include <climits>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;

/* SAD computation between two blocks (current frame vs reference frame).
    @param curr: current frame
    @param ref: reference frame
    @param currX: x coordinate of the top-left corner of the block in the current frame, in pixels
    @param currY: y coordinate of the top-left corner of the block in the current frame, in pixels
    @param refX: x coordinate of the top-left corner of the block in the reference frame, in pixels
    @param refY: y coordinate of the top-left corner of the block in the reference frame, in pixels
    @param blockSize: size of the block (in pixels, per side)
    @return: SAD value between the two blocks
*/
inline int OpenMPcomputeSAD(const ImageGray& curr, const ImageGray& ref,
                             int currX, int currY, int refX, int refY, int blockSize) {
    int sad = 0;
    // Vectorize the inner loop with SIMD, reduction(+:sad) combines the
    // per-lane partial sums into a single SAD value.
    #pragma omp simd reduction(+:sad)
    for (int j = 0; j < blockSize; j++) {
        for (int i = 0; i < blockSize; i++) {
            int c_pixel = curr.data[(currY + j) * curr.width + (currX + i)];
            int r_pixel = ref.data[(refY + j) * ref.width + (refX + i)];
            sad += abs(c_pixel - r_pixel);
        }
    }
    return sad;
}

/* Full search block matching algorithm, parallelized with OpenMP.
    @param curr: current frame
    @param ref: reference frame
    @param blockSize: size of the block (in pixels, per side)
    @param searchRange: search window radius in blocks; <= 0 means full-frame search
    @param metrics: timing metrics for this run
    @return: 2D grid of motion vectors, one per block of the current frame
*/
vector<vector<MotionVector>> fullSearchCPUOpenMPGray(const ImageGray& curr, const ImageGray& ref,
                                                      int blockSize, int searchRange, SingleRunMetrics& metrics) {

    auto start_time = chrono::high_resolution_clock::now();

    // Number of blocks in the current frame, based on frame size and block size
    int blocksX, blocksY;
    GridUtils::calculateGridDimensions(curr.width, curr.height, blockSize, blocksX, blocksY);

    // Output grid: one motion vector per block
    vector<vector<MotionVector>> mv = GridUtils::createMotionVectorGrid(blocksX, blocksY);

    LoggingUtils::printFrameInfo("CPU OpenMP", curr.width, curr.height, blockSize, blocksX, blocksY);
    LoggingUtils::printSearchModeInfo("CPU OpenMP", searchRange);

    // Choose the OpenMP scheduling policy (static vs dynamic) and the chunk size based on the actual workload
    // In full search mode every block compares the same number of candidates (blocksX * blocksY), so the workload is perfectly uniform and static scheduling is the natural choice.

    // In range search mode, blocks near the frame borders have a smaller search window than inner blocks, so the
    // workload is NOT uniform. The fragmentationFactor below captures how much of the frame is affected by this border effect, and
    // scales the chunk size and the static/dynamic decision accordingly

    int totalBlocks = blocksX * blocksY;
    int numThreadsAvail = omp_get_max_threads();

    // fragmentationFactor: 1 when the border effect is negligible (large image relative to searchRange, or full search), 
    // up to 8 when most of the frame is affected by border clamping (small image relative to searchRange). 
    // A lower factor means a larger chunk size (less scheduling overhead); a higher factor means a smaller chunk size.
    int fragmentationFactor = 1;

    if (searchRange > 0) {
        // Number of blocks affected by the search window clamp on each side of the frame .
        int rangeBlocks = (searchRange + blockSize - 1) / blockSize;

        // Inner blocks: far enough from every border that their search
        int innerBlocksX = max(0, blocksX - 2 * rangeBlocks);
        int innerBlocksY = max(0, blocksY - 2 * rangeBlocks);
        int innerBlocks = innerBlocksX * innerBlocksY;

        // Fraction of the frame affected by search window clamping.
        float borderRatio = 1.0f - ((float) innerBlocks / totalBlocks);

        // Linear scale: 0% border -> factor 1, 100% border -> factor 8.
        fragmentationFactor = 1 + (int) (borderRatio * 7.0f);
    }

    // higher fragmentation means smaller chunks, for finer load balancing.
    int chunk = max(1, totalBlocks / (numThreadsAvail * fragmentationFactor));

    if (searchRange <= 0) {
        // Full search: uniform workload across all blocks, static
        omp_set_schedule(omp_sched_static, chunk);
    } else {
        // Range search: estimate the cost of an unclamped inner block, using the same bounds calculation the main loop will use, evaluated at the center of the frame.
        SearchRangeUtils::SearchBounds innerBounds =
            SearchRangeUtils::calculateSearchBounds(blocksX / 2, blocksY / 2, blocksX, blocksY, searchRange);

        // Cost per block: number of candidates blocks * pixels per SAD computation.
        int costPerBlock = innerBounds.totalPositions * blockSize * blockSize;

        // 50000 threshold: above this, the absolute cost per block is high enough that border/center imbalance matters less in relative terms, static is fine. 
        //Also fall back to static if there aren't enough blocks to distribute.
        if (costPerBlock >= 50000 || totalBlocks < numThreadsAvail * fragmentationFactor) {
            omp_set_schedule(omp_sched_static, chunk);
        } else {
            // Low cost per block with a non-uniform workload: dynamic scheduling rebalances at runtime.
            omp_set_schedule(omp_sched_dynamic, chunk);
        }
    }

    // blocks of the current frame are independent (each writes to a distinct mv[bx][by]), so no reduction or critical
    // section is needed. collapse(2) merges the bx/by loops into a single iteration space, giving OpenMP more granularity to distribute across threads.
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int bx = 0; bx < blocksX; bx++) {
        for (int by = 0; by < blocksY; by++) {
            // Top-left corner of the current block, in pixels
            int x = bx * blockSize;
            int y = by * blockSize;
            int bestSAD = INT_MAX;
            int bestDist = INT_MAX;
            MotionVector bestMV{0, 0};

            SearchRangeUtils::SearchBounds bounds =
                SearchRangeUtils::calculateSearchBounds(bx, by, blocksX, blocksY, searchRange);

            // Scan every candidate block in the reference frame within bounds
            for (int irefY = bounds.startY; irefY <= bounds.endY; irefY++) {
                for (int irefX = bounds.startX; irefX <= bounds.endX; irefX++) {
                    // Top-left corner of the candidate block, in pixels
                    int refX = irefX * blockSize;
                    int refY = irefY * blockSize;
                    // Motion vector: difference in block indices between current block and candidate block
                    int dx = irefX - bx;
                    int dy = irefY - by;
                    int sad = OpenMPcomputeSAD(curr, ref, x, y, refX, refY, blockSize);
                    int dist = dx * dx + dy * dy; // squared length of the motion vector

                    // Tie-breaker: on equal SAD, prefer the smaller motion vector
                    if (sad < bestSAD || (sad == bestSAD && dist < bestDist)) {
                        bestSAD = sad;
                        bestMV = {dx, dy};
                        bestDist = dist;
                    }
                }
            }
            mv[bx][by] = bestMV;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> total_time_ms = end_time - start_time;
    metrics.total_ms = total_time_ms.count();
    LoggingUtils::printTimingReport("CPU OpenMP", total_time_ms.count());
    return mv;
}