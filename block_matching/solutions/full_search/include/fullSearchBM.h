#ifndef FULLSEARCH_BM_H
#define FULLSEARCH_BM_H

#include <vector>
#include "types.h"
#include "blockMatchingInterface.h"

// Full Search Block Matching implementation
class FullSearchBlockMatcher : public BlockMatcher {
public:
    std::vector<std::vector<MotionVector>> matchGray(
        const ImageGray& curr, 
        const ImageGray& ref,
        int blockSize) override;
    
    std::vector<std::vector<MotionVector>> matchRGB(
        const ImageColor& curr, 
        const ImageColor& ref,
        int blockSize) override;
};

#endif // FULLSEARCH_BM_H
