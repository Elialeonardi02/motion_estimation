#include "utils.h"
#include <regex>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

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


// LOGGING UTILITIES IMPLEMENTATION
namespace LoggingUtils {

void printFrameInfo(const string& implName, int width, int height, 
                   int blockSize, int blocksX, int blocksY) {
  cout << implName << " (Grayscale): Processing frame " << width << "x" << height
      << " with block size " << blockSize << endl;
  cout << implName << " (Grayscale): Grid size: " << blocksX << "x" << blocksY
      << " = " << (blocksX * blocksY) << " blocks" << endl;
}

void printSearchModeInfo(const string& implName, int searchRange) {
  string searchModeStr = (searchRange > 0) 
      ? ("Range search (range=" + to_string(searchRange) + " blocks)") 
      : "Full search";
  cout << implName << " (Grayscale): Search mode: " << searchModeStr
      << " (searchRange=" << searchRange << ")" << endl;
}

void printProgressUpdate(const string& implName, int processedBlocks, 
                        int totalBlocks, double elapsedSeconds) {
  double percentage = (100.0 * processedBlocks) / totalBlocks;
  cout << implName << " (Grayscale): Progress: " << processedBlocks << "/" 
      << totalBlocks << " blocks (" << fixed << setprecision(1) 
      << percentage << "%) - " << setprecision(2) << elapsedSeconds << " s" << endl;
}

void printTimingReport(const string& implName, double totalSeconds) {
  cout << implName << " (Grayscale): Timing:" << endl;
  cout << "  Total processing time: " << fixed << setprecision(4) 
      << totalSeconds << " s" << endl;
}

void printProcessingComplete(const string& implName) {
  cout << implName << " (Grayscale): Complete!" << endl;
}

void printCopyingToGPU(const string& implName, double sizeInMB) {
  cout << implName << " (Grayscale): Copying frames to GPU";
  if (sizeInMB > 0) {
    cout << " (" << fixed << setprecision(2) << sizeInMB << " MB)";
  }
  cout << "..." << endl;
}

void printCleanupGPU(const string& implName) {
  cout << implName << " (Grayscale): Cleaning up GPU memory..." << endl;
}

}

// SEARCH RANGE UTILITIES
namespace SearchRangeUtils {

SearchBounds calculateSearchBounds(int blockX, int blockY, int gridWidth, 
                                   int gridHeight, int searchRange) {
  SearchBounds bounds;
  
  /* Search bounds for reference blocks: limited to searchRange around current block, or full frame if searchRange <= 0
    If searchRange > 0, limit reference block positions to a square region around the current block:
     - bounds.startX/endX ∈ [max(0, blockX - searchRange), min(gridWidth - 1, blockX + searchRange)]
     - bounds.startY/endY ∈ [max(0, blockY - searchRange), min(gridHeight - 1, blockY + searchRange)]
    This creates a (2*searchRange + 1) x (2*searchRange + 1) block search area centered on the current block.
    If searchRange <= 0, search all blocks in the reference frame:
     - bounds.startX/endX ∈ [0, gridWidth - 1]
     - bounds.startY/endY ∈ [0, gridHeight - 1]
    This creates a full frame search area.
  */
  if (searchRange > 0) {
    // Range search mode: limit search to a window around the current block
    bounds.startX = max(0, blockX - searchRange);
    bounds.endX = min(gridWidth - 1, blockX + searchRange);
    bounds.startY = max(0, blockY - searchRange);
    bounds.endY = min(gridHeight - 1, blockY + searchRange);
  } else {
    // Full search mode: search the entire frame
    bounds.startX = 0;
    bounds.endX = gridWidth - 1;
    bounds.startY = 0;
    bounds.endY = gridHeight - 1;
  }
  
  bounds.width = bounds.endX - bounds.startX + 1;
  bounds.height = bounds.endY - bounds.startY + 1;
  bounds.totalPositions = bounds.width * bounds.height;
  
  return bounds;
}

}

// VALIDATION UTILITIES
namespace ValidationUtils {

void validateFrameDimensions(const ImageGray& curr, const ImageGray& ref) {
  int currSize = curr.width * curr.height;
  int refSize = ref.width * ref.height;
  
  if (currSize != refSize) {
    throw runtime_error("Current and reference frames must have the same dimensions.");
  }
}

void validateBlockSize(int blockSize, int width, int height) {
  if (blockSize <= 0) {
    throw runtime_error("Block size must be positive.");
  }
  if (width % blockSize != 0 || height % blockSize != 0) {
    throw runtime_error("Block size must divide frame dimensions evenly.");
  }
}

void validateSearchRange(int searchRange, int gridSize) {
  if (searchRange > 0 && searchRange >= gridSize) {
    throw runtime_error("Search range cannot be greater than or equal to grid size.");
  }
}

}


// GRID UTILITIES IMPLEMENTATION
namespace GridUtils {

void calculateGridDimensions(int width, int height, int blockSize, 
                             int& blocksX, int& blocksY) {
  blocksX = width / blockSize;
  blocksY = height / blockSize;
}

vector<vector<MotionVector>> createMotionVectorGrid(int width, int height) {
  return vector<vector<MotionVector>>(width, vector<MotionVector>(height));
}

vector<vector<MotionVector>> flatTo2DVector(
    const vector<MotionVector>& h_mv_flat, int blocksX, int blocksY) {
  vector<vector<MotionVector>> result = createMotionVectorGrid(blocksX, blocksY);
  
  for (int by = 0; by < blocksY; ++by) {
    for (int bx = 0; bx < blocksX; ++bx) {
      result[bx][by] = h_mv_flat[by * blocksX + bx];
    }
  }
  
  return result;
}

}
