#include "utils.h"
#include <regex>

// Extract the last directory name from a path
std::string getLastDirName(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    if(last_slash == std::string::npos) return path;
    if(last_slash == path.size() - 1) {
        // Remove trailing slash
        std::string temp = path.substr(0, last_slash);
        last_slash = temp.find_last_of("/\\");
        return (last_slash == std::string::npos) ? temp : temp.substr(last_slash + 1);
    }
    return path.substr(last_slash + 1);
}

// Extract frame number from filename (e.g., "flower2.raw" -> 2, "frame003.pgm" -> 3)
int extractFrameNumber(const std::string& filename) {
    // Get just the filename without path
    size_t last_slash = filename.find_last_of("/\\");
    std::string name_only = (last_slash != std::string::npos) ? filename.substr(last_slash + 1) : filename;
    
    // Remove extension
    size_t dot_pos = name_only.find_last_of('.');
    if(dot_pos != std::string::npos) {
        name_only = name_only.substr(0, dot_pos);
    }
    
    // Extract the last sequence of digits
    std::regex number_regex("(\\d+)");
    std::smatch match;
    int frame_number = -1;
    
    std::string::const_iterator searchStart(name_only.cbegin());
    while(std::regex_search(searchStart, name_only.cend(), match, number_regex)) {
        frame_number = std::stoi(match[0]);
        searchStart = match.suffix().first;
    }
    
    return frame_number;
}
