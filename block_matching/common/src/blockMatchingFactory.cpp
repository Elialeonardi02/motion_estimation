#include "blockMatchingInterface.h"
#include "fullSearchBM_cpu_naive.h"
#include "logarithmicSearchBM_cpu_naive.h"
#include <stdexcept>
#include <iostream>

// CPU implementations
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                               int blockSize, int searchRange);
std::vector<std::vector<MotionVector>> fullSearchCPUNaiveRGB(const ImageColor& curr, const ImageColor& ref,
                                                              int blockSize, int searchRange);

// Logarithmic search CPU implementation
std::vector<std::vector<MotionVector>> logarithmicSearchCPUNaiveGray(const ImageGray& curr, const ImageGray& ref,
                                                                      int blockSize, int distance);

#ifndef NO_CUDA
extern std::vector<std::vector<MotionVector>> fullSearchCUDANaiveGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize, int searchRange);

extern std::vector<std::vector<MotionVector>> fullSearchCUDAOptimizedGray(
    const ImageGray& curr, 
    const ImageGray& ref,
    int blockSize);

extern std::vector<std::vector<MotionVector>> fullSearchCUDAUncoalescedOptimizedGray(
    const ImageGray& curr,
    const ImageGray& ref,
    int blockSize, int searchRange);
#endif

class FullSearchBlockMatcher : public BlockMatcher {
private:
    int searchRange = -1; // <0 means full frame search, otherwise limited to range around block position
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override {
        return fullSearchCPUNaiveGray(curr, ref, blockSize, searchRange);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor& curr, 
        const ImageColor& ref,
        int blockSize) override {
        return fullSearchCPUNaiveRGB(curr, ref, blockSize, searchRange);
    }
    void setSearchRange(int range) override {
        searchRange = range;
    }

};

#ifndef NO_CUDA
class FullSearchBlockMatcherCUDA : public BlockMatcher {
private:
    int searchRange = -1; // <0 means full frame search, otherwise limited to range around block position
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override {
        return fullSearchCUDANaiveGray(curr, ref, blockSize, searchRange);
    }
     void setSearchRange(int range) override {
        searchRange = range;
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA RGB implementation not available");
    }
};

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
        throw std::runtime_error("CUDA RGB implementation not available");
    }
};

class FullSearchBlockMatcherCUDAUncoalescedOptimized : public BlockMatcher {
private:
    int searchRange =-1; // <0 means full frame search
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr,
        const ImageGray& ref,
        int blockSize) override {
        return fullSearchCUDAUncoalescedOptimizedGray(curr, ref, blockSize, searchRange);
    }
    void setSearchRange(int range) override {
        searchRange = range;
    }

    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&,
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA RGB implementation not available");
    }
};
#else
// Stub implementations when CUDA is not available
class FullSearchBlockMatcherCUDA : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, 
        const ImageGray&,
        int) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA support is not available");
    }
};

class FullSearchBlockMatcherCUDAOptimized : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, 
        const ImageGray&,
        int) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA support is not available");
    }
};

class FullSearchBlockMatcherCUDAUncoalescedOptimized : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray&, 
        const ImageGray&,
        int) override {
        throw std::runtime_error("CUDA support is not available. Compile with CUDA support enabled.");
    }

    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("CUDA support is not available");
    }
};
#endif

// Logarithmic Search Block Matcher - CPU naive implementation
class LogarithmicSearchBlockMatcher : public BlockMatcher {
private:
    int distance = 32;  // Default distance
    
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override {
        return logarithmicSearchCPUNaiveGray(curr, ref, blockSize, distance);
    }
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor&, 
        const ImageColor&,
        int) override {
        throw std::runtime_error("Logarithmic search RGB implementation not available");
    }
    
    void setDistance(int d) override {
        distance = d;
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
        } else if(implementation == "cuda_uncoalesced_optimized") {
            return std::make_unique<FullSearchBlockMatcherCUDAUncoalescedOptimized>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + "' for algorithm '" + algorithm + "'");
        }
    } else if(algorithm == "logarithmic_search") {
        if(implementation == "cpu_naive") {
            return std::make_unique<LogarithmicSearchBlockMatcher>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + "' for algorithm '" + algorithm + "'");
        }
    } else {
        throw std::runtime_error("Unknown algorithm '" + algorithm + "'");
    }
}
