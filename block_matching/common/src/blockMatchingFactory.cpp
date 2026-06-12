#include "blockMatchingInterface.h"
#include <stdexcept>
#include <iostream>
#include <memory>

// Forward declarations for CPU implementations
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize, int searchRange, SingleRunMetrics& metrics);
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                              int blockSize, int searchRange);

std::vector<std::vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                                      int blockSize, int distance, SingleRunMetrics& metrics);

#ifndef NO_CUDA
extern std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(
    const ImageGray& curr, const ImageGray& ref, int blockSize, int searchRange, SingleRunMetrics& metrics);

extern std::vector<std::vector<MotionVector>> fullSearchCUDAOptimizedGray(
    const ImageGray& curr, const ImageGray& ref, int blockSize, int searchRange, SingleRunMetrics& metrics);

extern std::vector<std::vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(
    const ImageGray& curr, const ImageGray& ref, int blockSize, int searchRange, SingleRunMetrics& metrics);

extern std::vector<std::vector<MotionVector>> logarithmicSearchCUDAOptimizedGray(
    const ImageGray& curr, const ImageGray& ref, int blockSize, int searchDistance, SingleRunMetrics& metrics);
#endif

// BASE CLASSES
class FullSearchBlockMatcherBase : public BlockMatcher {
protected:
    int searchRange = -1;  // <0 means full frame search, >0 for limited range
    
public:
    void setSearchRange(int range) override {
        searchRange = range;
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, const ImageColor&, int) override {
        throw std::runtime_error("RGB implementation not available for this matcher");
    }
};


// FULL SEARCH - CPU NAIVE IMPLEMENTATION
class FullSearchBlockMatcherCPUNaive : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return fullSearchCPUNaiveGray(curr, ref, blockSize, searchRange, metrics);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor& curr, const ImageColor& ref, int blockSize) override {
        return fullSearchCPUNaiveRGB(curr, ref, blockSize, searchRange);
    }
};

// FULL SEARCH - CUDA IMPLEMENTATIONS 
#ifndef NO_CUDA

class FullSearchBlockMatcherCUDANaive : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return fullSearchCUDANaiveGray(curr, ref, blockSize, searchRange, metrics);
    }
};

class FullSearchBlockMatcherCUDAOptimized : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return fullSearchCUDAOptimizedGray(curr, ref, blockSize, searchRange, metrics);
    }
};

class FullSearchBlockMatcherCUDAUncoalescedOptimized : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return fullSearchCUDAUncoalescedOptimizedGray(curr, ref, blockSize, searchRange, metrics);
    }
};

class LogarithmicSearchBlockMatcherCUDAOptimized : public BlockMatcher {
private:
    int distance = 32;
    
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return logarithmicSearchCUDAOptimizedGray(curr, ref, blockSize, distance, metrics);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, const ImageColor&, int) override {
        throw std::runtime_error("Logarithmic search RGB implementation not available");
    }
    
    void setDistance(int d) override {
        distance = d;
    }
};

#else

// Stub implementations when CUDA is not available
class FullSearchBlockMatcherCUDANaive : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, const ImageGray&, int, SingleRunMetrics&) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
};

class FullSearchBlockMatcherCUDAOptimized : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, const ImageGray&, int, SingleRunMetrics&) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
};

class FullSearchBlockMatcherCUDAUncoalescedOptimized : public FullSearchBlockMatcherBase {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, const ImageGray&, int, SingleRunMetrics&) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
};

class LogarithmicSearchBlockMatcherCUDAOptimized : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, const ImageGray&, int, SingleRunMetrics&) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }

    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, const ImageColor&, int) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
};

#endif

// LOGARITHMIC SEARCH - CPU NAIVE IMPLEMENTATION
class LogarithmicSearchBlockMatcherCPUNaive : public BlockMatcher {
private:
    int distance = 32;  // Default distance
    
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, const ImageGray& ref, int blockSize, SingleRunMetrics& metrics) override {
        return logarithmicSearchCPUNaiveGray(curr, ref, blockSize, distance, metrics);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, const ImageColor&, int) override {
        throw std::runtime_error("Logarithmic search RGB implementation not available");
    }
    
    void setDistance(int d) override {
        distance = d;
    }
};


// FACTORY FUNCTION
std::unique_ptr<BlockMatcher> createBlockMatcher(
    const std::string& algorithm, 
    const std::string& implementation) {
    
    if (algorithm == "full_search") {
        if (implementation == "cpu_naive") {
            return std::make_unique<FullSearchBlockMatcherCPUNaive>();
        } else if (implementation == "cuda_naive") {
            return std::make_unique<FullSearchBlockMatcherCUDANaive>();
        } else if (implementation == "cuda_optimized") {
            return std::make_unique<FullSearchBlockMatcherCUDAOptimized>();
        } else if (implementation == "cuda_uncoalesced_optimized") {
            return std::make_unique<FullSearchBlockMatcherCUDAUncoalescedOptimized>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + 
                                   "' for algorithm '" + algorithm + "'");
        }
    } else if (algorithm == "logarithmic_search") {
        if (implementation == "cpu_naive") {
            return std::make_unique<LogarithmicSearchBlockMatcherCPUNaive>();
        } else if (implementation == "cuda_optimized") {
            return std::make_unique<LogarithmicSearchBlockMatcherCUDAOptimized>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + 
                                   "' for algorithm '" + algorithm + "'");
        }
    } else {
        throw std::runtime_error("Unknown algorithm '" + algorithm + "'");
    }
}
