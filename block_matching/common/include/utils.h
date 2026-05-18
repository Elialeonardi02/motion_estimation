#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include "types.h"

// Extract the last directory name from a path
std::string getLastDirName(const std::string& path);

// Extract frame number from filename
int extractFrameNumber(const std::string& filename);

// Generate output filename for motion vectors based on frame naming scheme
std::string generateOutputFilename(bool isRange, int range_start, size_t i, 
                                   const std::string& ref_path, const std::string& curr_path);

// LOGGING UTILITIES
namespace LoggingUtils {
  // Print frame information at the start of processing
  void printFrameInfo(const std::string& implName, int width, int height, 
                     int blockSize, int blocksX, int blocksY);
  
  // Print search mode information
  void printSearchModeInfo(const std::string& implName, int searchRange);
  
  // Print progress update during processing
  void printProgressUpdate(const std::string& implName, int processedBlocks, 
                          int totalBlocks, double elapsedSeconds);
  
  // Print final timing report
  void printTimingReport(const std::string& implName, double totalSeconds);

  // Print completion message
  void printProcessingComplete(const std::string& implName);

  // Print copying frames message
  void printCopyingToGPU(const std::string& implName, double sizeInMB = -1.0);

  // Print cleanup message
  void printCleanupGPU(const std::string& implName);
}

// SEARCH RANGE UTILITIES
namespace SearchRangeUtils {
  // Structure for search window bounds and dimensions
  struct SearchBounds {
    int startX;         // leftmost block index in reference frame
    int endX;           // rightmost block index in reference frame
    int startY;         // topmost block index in reference frame
    int endY;           // bottommost block index in reference frame
    int width;          // width of search window in blocks
    int height;         // height of search window in blocks
    int totalPositions; // total candidate positions in search window
  };

  // Calculate search bounds for a given block position
  SearchBounds calculateSearchBounds(int blockX, int blockY, int gridWidth, 
                                     int gridHeight, int searchRange);
}


// VALIDATION UTILITIES
namespace ValidationUtils {
  // Validate that current and reference frames have the same dimensions
  void validateFrameDimensions(const ImageGray& curr, const ImageGray& ref);

  // Validate that block size divides frame dimensions evenly
  void validateBlockSize(int blockSize, int width, int height);

  // Validate search range
  void validateSearchRange(int searchRange, int gridSize);
}

// GRID UTILITIES
namespace GridUtils {
  // Calculate grid dimensions based on frame size and block size
  void calculateGridDimensions(int width, int height, int blockSize, 
                               int& blocksX, int& blocksY);

  // Create an empty 2D motion vector grid
  std::vector<std::vector<MotionVector>> createMotionVectorGrid(int width, int height);

  // Convert flat 1D motion vector array to 2D grid
  std::vector<std::vector<MotionVector>> flatTo2DVector(
      const std::vector<MotionVector>& h_mv_flat, int blocksX, int blocksY);
}

#endif // UTILS_H
