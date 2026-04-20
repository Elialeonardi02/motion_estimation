#include "blockMatchingInterface.h"
#include "fullSearchBM.h"
#include <stdexcept>
#include <iostream>

// Forward declarations for CUDA implementations
std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize);

std::vector<std::vector<MotionVector>> fullSearchCUDAOptimizedGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize);

// Inline wrapper for CUDA Naive implementation
class FullSearchBlockMatcherCUDA : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override {
        return fullSearchCUDANaiveGray(curr, ref, blockSize);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA RGB implementation not yet implemented");
    }
};

// Inline wrapper for CUDA Optimized implementation
class FullSearchBlockMatcherCUDAOptimized : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override {
        return fullSearchCUDAOptimizedGray(curr, ref, blockSize);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
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
        } else if(implementation == "cuda_optimized") {
            return std::make_unique<FullSearchBlockMatcherCUDAOptimized>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + "' for algorithm '" + algorithm + "'");
        }
    } else {
        throw std::runtime_error("Unknown algorithm '" + algorithm + "'");
    }
}
