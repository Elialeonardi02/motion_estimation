#include "fullSearchBM.h"
#include "fullSearchBM_cpu_naive.h"

// Forward declarations from cpu_naive implementation
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize);
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                              int blockSize);

std::vector<std::vector<MotionVector>> FullSearchBlockMatcher::matchGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize) {
    return fullSearchCPUNaiveGray(curr, ref, blockSize);
}

std::vector<std::vector<MotionVector>> FullSearchBlockMatcher::matchRGB(
    const ImageColor& curr, 
    const ImageColor& ref,
    int blockSize) {
    return fullSearchCPUNaiveRGB(curr, ref, blockSize);
}
