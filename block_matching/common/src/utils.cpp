#include "utils.h"
#include <regex>

using namespace std;

// Extract the last directory name from a path
string getLastDirName(const string& path) {
    size_t last_slash = path.find_last_of("/\\");
    if(last_slash == string::npos) return path;
    if(last_slash == path.size() - 1) {
        // Remove trailing slash
        string temp = path.substr(0, last_slash);
        last_slash = temp.find_last_of("/\\");
        return (last_slash == string::npos) ? temp : temp.substr(last_slash + 1);
    }
    return path.substr(last_slash + 1);
}

// Extract frame number from filename (e.g., "flower2.raw" -> 2, "frame003.pgm" -> 3)
int extractFrameNumber(const string& filename) {
    // Get just the filename without path
    size_t last_slash = filename.find_last_of("/\\");
    string name_only = (last_slash != string::npos) ? filename.substr(last_slash + 1) : filename;
    
    // Remove extension
    size_t dot_pos = name_only.find_last_of('.');
    if(dot_pos != string::npos) {
        name_only = name_only.substr(0, dot_pos);
    }
    
    // Extract the last sequence of digits
    regex number_regex("(\\d+)");
    smatch match;
    int frame_number = -1;
    
    string::const_iterator searchStart(name_only.cbegin());
    while(regex_search(searchStart, name_only.cend(), match, number_regex)) {
        frame_number = stoi(match[0]);
        searchStart = match.suffix().first;
    }
    
    return frame_number;
}

// Generate output filename for motion vectors
string generateOutputFilename(bool isRange, int range_start, size_t i,
                              const string& ref_path, const string& curr_path) {
    if(isRange) {
        return "motion_vectors_frame_" + to_string(range_start + i) + "_vs_" + to_string(range_start + i + 1) + ".ppm";
    } else {
        int frame1 = extractFrameNumber(ref_path);
        int frame2 = extractFrameNumber(curr_path);
        if(frame1 > 0 && frame2 > 0) {
            return "motion_vectors_" + to_string(frame1) + "-" + to_string(frame2) + ".ppm";
        } else {
            return "motion_vectors_" + to_string(i + 1) + "-" + to_string(i + 2) + ".ppm";
        }
    }
}
