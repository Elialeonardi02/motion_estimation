#ifndef UTILS_H
#define UTILS_H

#include <string>

// Extract the last directory name from a path
std::string getLastDirName(const std::string& path);

// Extract frame number from filename (e.g., "flower2.raw" -> 2, "frame003.pgm" -> 3)
int extractFrameNumber(const std::string& filename);

// Generate output filename for motion vectors based on frame naming scheme
std::string generateOutputFilename(bool isRange, int range_start, size_t i, 
                                   const std::string& ref_path, const std::string& curr_path);

#endif // UTILS_H
