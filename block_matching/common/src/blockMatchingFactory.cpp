#include "blockMatchingInterface.h"
#include "fullSearchBM.h"
#include <stdexcept>
#include <iostream>

// Forward declaration for CUDA implementation
std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize, 
    int searchRange);

// Inline wrapper for CUDA implementation
class FullSearchBlockMatcherCUDA : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize, 
        int searchRange) override {
        return fullSearchCUDANaiveGray(curr, ref, blockSize, searchRange);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor& curr, 
        const ImageColor& ref,
        int blockSize, 
        int searchRange) override {
        throw std::runtime_error("CUDA RGB implementation not yet implemented");
    }
};

std::unique_ptr<BlockMatcher> createBlockMatcher(
    const std::string& algorithm, 
    const std::string& implementation) {
    
    if(algorithm == "full_search") {
        if(implementation == "cpu_naive") {
            return std::make_unique<FullSearchBlockMatcher>();
        } else if(implementation == "cuda_naive") {
            return std::make_unique<FullSearchBlockMatcherCUDA>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + "' for algorithm '" + algorithm + "'");
        }
    } else {
        throw std::runtime_error("Unknown algorithm '" + algorithm + "'");
    }
}
