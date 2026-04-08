#ifndef UTILS_H
#define UTILS_H

#include <string>

// Extract the last directory name from a path
std::string getLastDirName(const std::string& path);

// Extract frame number from filename (e.g., "flower2.raw" -> 2, "frame003.pgm" -> 3)
int extractFrameNumber(const std::string& filename);

#endif // UTILS_H
