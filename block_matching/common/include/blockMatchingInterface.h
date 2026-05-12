#ifndef BLOCK_MATCHING_INTERFACE_H
#define BLOCK_MATCHING_INTERFACE_H

#include <vector>
#include <memory>
#include "types.h"

// Abstract interface for Block Matching algorithms
class BlockMatcher {
public:
    virtual ~BlockMatcher() = default;
    
    // Compute motion vectors for grayscale images
    virtual std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) = 0;
    
    // Compute motion vectors for RGB color images
    virtual std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor& curr, 
        const ImageColor& ref,
        int blockSize) = 0;
    
    // set Distance parameter for logarithmic search
    virtual void setDistance(int) {}

    // set Search Range parameter for range search mode
    virtual void setSearchRange(int) {}
};

// Factory function to create block matcher instances
std::unique_ptr<BlockMatcher> createBlockMatcher(
    const std::string& algorithm, 
    const std::string& implementation);

#endif // BLOCK_MATCHING_INTERFACE_H
