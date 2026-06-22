#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <climits>
#include "sad_utils.h"
#include "utils.h"
#include <chrono>
#include <omp.h>

using namespace std;

constexpr int MAXCANDIDATES_AT_DISTANCE = 9;

vector<vector<MotionVector>> logarithmicSearchGPUOpenMP(
    const ImageGray& curr, const ImageGray& ref, 
    int blockSize, int searchDistance, SingleRunMetrics& metrics) 
{
    auto total_time_start = std::chrono::high_resolution_clock::now();

    int blocksX = curr.width / blockSize;
    int blocksY = curr.height / blockSize;
    const int frameBlocks = blocksX * blocksY;
    const int frameSize = curr.width * curr.height;

    vector<CandidateSad> best_candidates(frameBlocks);
    vector<CandidateSad> sad_candidates(frameBlocks * MAXCANDIDATES_AT_DISTANCE);
    
    const unsigned char* d_curr_ptr = curr.data.data();
    const unsigned char* d_ref_ptr = ref.data.data();
    CandidateSad* d_best = best_candidates.data();
    CandidateSad* d_sad_cand = sad_candidates.data();
    
    int width = curr.width;

    #pragma omp target data map(to: d_curr_ptr[0:frameSize], d_ref_ptr[0:frameSize]) \
                            map(alloc: d_sad_cand[0:frameBlocks * MAXCANDIDATES_AT_DISTANCE]) \
                            map(from: d_best[0:frameBlocks])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < frameBlocks; ++i) {
            d_best[i] = {INT_MAX, 0, 0};
        }

        for (int currentDistance = searchDistance; currentDistance > 0; currentDistance >>= 1) {
            unsigned char curr_shared[1024];

            #pragma omp target teams distribute collapse(2) \
                               private(curr_shared) \
                               allocate(omp_pteam_mem_alloc: curr_shared)
            for (int by = 0; by < blocksY; ++by) {
                for (int bx = 0; bx < blocksX; ++bx) {
                    
                    int frameBlockIdx = by * blocksX + bx;
                    int cx = bx + d_best[frameBlockIdx].dx;
                    int cy = by + d_best[frameBlockIdx].dy;

                    

                    #pragma omp parallel for collapse(2)
                    for (int py = 0; py < blockSize; ++py) {
                        for (int px = 0; px < blockSize; ++px) {
                            curr_shared[py * blockSize + px] = d_curr_ptr[(by * blockSize + py) * width + (bx * blockSize + px)];
                        }
                    }

                    for (int cand = 0; cand < MAXCANDIDATES_AT_DISTANCE; ++cand) {
                        int dirX = (cand % 3) - 1;
                        int dirY = (cand / 3) - 1;
                        int cand_bx = cx + (dirX * currentDistance);
                        int cand_by = cy + (dirY * currentDistance);

                        if (cand_bx < 0 || cand_by < 0 || cand_bx >= blocksX || cand_by >= blocksY) {
                            d_sad_cand[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + cand] = {INT_MAX, 0, 0};
                        } else {
                            int partial_sad = 0;
                            int ref_x_base = cand_bx * blockSize;
                            int ref_y_base = cand_by * blockSize;

                            #pragma omp parallel for reduction(+:partial_sad) collapse(2)
                            for (int py = 0; py < blockSize; ++py) {
                                for (int px = 0; px < blockSize; ++px) {
                                    int shared_idx = py * blockSize + px;
                                    int ref_idx = (ref_y_base + py) * width + (ref_x_base + px);
                                    partial_sad += std::abs(curr_shared[shared_idx] - d_ref_ptr[ref_idx]);
                                }
                            }

                            d_sad_cand[frameBlockIdx * MAXCANDIDATES_AT_DISTANCE + cand] = {
                                partial_sad, 
                                static_cast<short>(cand_bx - bx), 
                                static_cast<short>(cand_by - by)
                            };
                        }
                    }
                }
            }

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < frameBlocks; ++i) {
                CandidateSad best = {INT_MAX, 0, 0};
                for (int c = 0; c < MAXCANDIDATES_AT_DISTANCE; ++c) {
                    CandidateSad cand = d_sad_cand[i * MAXCANDIDATES_AT_DISTANCE + c];
                    
                    if (cand.sad < best.sad) {
                        best = cand;
                    } else if (cand.sad != INT_MAX && cand.sad == best.sad) {
                        int distA = (cand.dx * cand.dx) + (cand.dy * cand.dy);
                        int distB = (best.dx * best.dx) + (best.dy * best.dy);
                        if (distA < distB) {
                            best = cand;
                        } else if (distA == distB) {
                            if (cand.dy < best.dy) best = cand;
                            else if (cand.dy == best.dy && cand.dx < best.dx) best = cand;
                        }
                    }
                }
                d_best[i] = best;
            }
        }
    }

    vector<MotionVector> h_mv_flat(frameBlocks);
    for(int i = 0; i < frameBlocks; ++i) {
        h_mv_flat[i] = { best_candidates[i].dx, best_candidates[i].dy };
    }
    
    auto total_time_stop = std::chrono::high_resolution_clock::now();
    metrics.total_ms = std::chrono::duration<float, std::milli>(total_time_stop - total_time_start).count();
    
    return GridUtils::flatTo2DVector(h_mv_flat, blocksX, blocksY);
}