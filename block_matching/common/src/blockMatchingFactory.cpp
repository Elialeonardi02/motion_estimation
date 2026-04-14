#include "blockMatchingInterface.h"
#include "fullSearchBM.h"
#include <stdexcept>

std::unique_ptr<BlockMatcher> createBlockMatcher(
    const std::string& algorithm, 
    const std::string& implementation) {
    
    if(algorithm == "full_search") {
        if(implementation == "cpu_naive") {
            return std::make_unique<FullSearchBlockMatcher>();
        } else {
            throw std::runtime_error("Unknown implementation '" + implementation + "' for algorithm '" + algorithm + "'");
        }
    } else {
        throw std::runtime_error("Unknown algorithm '" + algorithm + "'");
    }
}
