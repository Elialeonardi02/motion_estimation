#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "types.h"
#include "imageIO.h"
#include "blockMatchingInterface.h"
#include "visualization.h"
#include "utils.h"

using namespace std;

int main(int argc, char* argv[]) {
    // Default parameters for input frames, output, and processing.
    string format = "pgm";
    int width = 0, height = 0;
    bool isColor = false;
    int blockSize = 32;

    
    // Block matching algorithm and implementation selection
    string algorithm = "full_search";
    string implementation = "cpu_naive";

    // range-based input parameters
    string input_dir = "";
    string range_prefix = "";
    string range_suffix = "";
    int range_start = 0, range_end = 0;
    
    // individual image paths based input parameters
    vector<string> image_paths;

    // Parse command-line arguments
    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if(arg == "--algorithm") {
            if(i + 1 < argc) {
                algorithm = argv[++i];
            } else {
                cerr << "Error: --algorithm requires a value" << endl;
                return 1;
            }
        } else if(arg == "--implementation") {
            if(i + 1 < argc) {
                implementation = argv[++i];
            } else {
                cerr << "Error: --implementation requires a value" << endl;
                return 1;
            }
        } else if(arg == "--format") {
            if(i + 1 < argc) {
                format = argv[++i];
            } else {
                cerr << "Error: --format requires a value (pgm, ppm, or raw)" << endl;
                return 1;
            }
        } else if(arg == "--width") {
            if(i + 1 < argc) {
                width = atoi(argv[++i]);
            } else {
                cerr << "Error: --width requires a numeric value" << endl;
                return 1;
            }
        } else if(arg == "--height") {
            if(i + 1 < argc) {
                height = atoi(argv[++i]);
            } else {
                cerr << "Error: --height requires a numeric value" << endl;
                return 1;
            }
        } else if(arg == "--input-dir") {
            if(i + 1 < argc) {
                input_dir = argv[++i];
            } else {
                cerr << "Error: --input-dir requires a directory name" << endl;
                return 1;
            }
        } else if(arg == "--range") {
            if(i + 4 < argc) {
                range_prefix = argv[++i];
                range_suffix = argv[++i];
                range_start = atoi(argv[++i]);
                range_end = atoi(argv[++i]);
            } else {
                cerr << "Error: --range requires 4 values (prefix suffix start end)" << endl;
                return 1;
            }
        } else if(arg == "--color") {
            isColor = true;
        } else if(arg == "--block-size") {
            if(i + 1 < argc) {
                blockSize = atoi(argv[++i]);
            } else {
                cerr << "Error: --block-size requires a numeric value" << endl;
                return 1;
            }
        } else {
            image_paths.push_back(arg);
        }
    }

    // generate file paths based on the provided prefix, suffix, and range (if -- range is specified)
    if(!range_prefix.empty()) {
        if(input_dir.empty()) {
            cerr << "Error: --range requires --input-dir" << endl;
            return 1;
        }
        for(int num = range_start; num <= range_end; ++num) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "%s%04d%s", range_prefix.c_str(), num, range_suffix.c_str());
            string filename = input_dir + "/" + buffer;
            image_paths.push_back(filename);
        }
    }

    // Motion estimation works on consecutive pairs, so we need at least two images.
    if(image_paths.size() < 2) {
        cerr << "Usage: " << argv[0]
             << " [--algorithm <algorithm>] [--implementation <impl>]"
             << " [--input-dir <dir>] [--range <prefix> <suffix> <start> <end>]"
             << " [--format pgm|ppm|raw] [--width W] [--height H] [--color]"
             << " [--block-size <size>]"
             << " <image1> <image2> ..." << endl;
        cerr << "At least two images are required for motion estimation." << endl;
        cerr << "For RAW format, specify --width and --height. Use --color to specify RGB images (default is grayscale)." << endl;
        cerr << "Available algorithms: full_search" << endl;
        cerr << "Available implementations: cpu_naive, cuda_naive, cuda_optimized" << endl;
        cerr << "Default block size: 32 (full frame search)" << endl;
        return 1;
    }

    // Raw format requires --width and --height to be specified.
    if(format == "raw" && (width <= 0 || height <= 0)) {
        cerr << "Error: For RAW format, valid --width and --height must be specified." << endl;
        return 1;
    }

    // Determine output directory with subdirectory based on input folder and algorithm/implementation
    string output_base = "output";
    string sub_dir = "";
    
    // Extract directory name from input path
    if(!input_dir.empty()) {
        sub_dir = getLastDirName(input_dir);
    } else if(!image_paths.empty()) {
        // Extract directory from first file path
        string first_path = image_paths[0];
        size_t pos = first_path.rfind('/');
        if(pos == string::npos) pos = first_path.rfind('\\');
        if(pos != string::npos && pos > 0) {
            string dir_path = first_path.substr(0, pos);
            size_t last_dir_pos = dir_path.rfind('/');
            if(last_dir_pos == string::npos) last_dir_pos = dir_path.rfind('\\');
            if(last_dir_pos != string::npos) {
                sub_dir = dir_path.substr(last_dir_pos + 1);
            } else {
                sub_dir = dir_path;
            }
        }
    }
    
    // Create output directory with algorithm_implementation subdirectory
    string algorithm_impl = algorithm + "_" + implementation;
    string output_dir = output_base + "/" + algorithm_impl;
    if(!sub_dir.empty()) {
        output_dir = output_dir + "/" + sub_dir;
    }
    filesystem::create_directories(output_dir);

    cout << "Block Matching Configuration:" << endl;
    cout << "  Algorithm: " << algorithm << endl;
    cout << "  Implementation: " << implementation << endl;
    cout << "  Block size: " << blockSize << endl;
    cout << "  Search mode: Full frame (naive)" << endl;
    cout << "  Output directory: " << output_dir << endl << endl;

    // Create block matcher
    unique_ptr<BlockMatcher> matcher;
    try {
        matcher = createBlockMatcher(algorithm, implementation);
    } catch(const exception& e) {
        cerr << "Error creating block matcher: " << e.what() << endl;
        return 1;
    }

    bool isRange = !range_prefix.empty();

    // Process consecutive image pairs
    for(size_t i = 0; i < image_paths.size() - 1; ++i) {
        string ref_path = image_paths[i];
        string curr_path = image_paths[i + 1];

        cout << "Processing pair: " << ref_path << " -> " << curr_path << endl;

        try {
            if(isColor) {
                ImageColor ref_frame, curr_frame;
                if(format == "ppm") {
                    ref_frame = loadPPM(ref_path);
                    curr_frame = loadPPM(curr_path);
                } else if(format == "raw") {
                    ref_frame = loadRawRGB(ref_path, width, height);
                    curr_frame = loadRawRGB(curr_path, width, height);
                } else {
                    throw runtime_error("For RGB images, use 'ppm' or 'raw' format");
                }
                
                chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
                vector<vector<MotionVector>> mv = matcher->matchRGB(curr_frame, ref_frame, blockSize);
                chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end - start;
                cout << "Matching time: " << elapsed.count() << " s" << endl;

                string output_name = generateOutputFilename(isRange, range_start, i, ref_path, curr_path);
                string output_path = output_dir + "/" + output_name;
                savePPM(drawMotionVectorsRGB(curr_frame, ref_frame, mv, blockSize), output_path);
                cout << "Motion vectors saved in " << output_path << endl;
                
                // Save frame difference image
                size_t last_dot = output_path.rfind('.');
                string diff_path = output_path.substr(0, last_dot) + "_diff.ppm";
                savePPM(drawFrameDifferenceRGB(ref_frame, curr_frame), diff_path);
                cout << "Frame difference saved in " << diff_path << endl;
                
                // Save reference frame with grid
                string ref_grid_path = output_path.substr(0, last_dot) + "_ref_grid.ppm";
                savePPM(drawFrameWithGridRGB(ref_frame, blockSize), ref_grid_path);
                cout << "Reference frame with grid saved in " << ref_grid_path << endl;
                
                // Save current frame with grid
                string curr_grid_path = output_path.substr(0, last_dot) + "_curr_grid.ppm";
                savePPM(drawFrameWithGridRGB(curr_frame, blockSize), curr_grid_path);
                cout << "Current frame with grid saved in " << curr_grid_path << endl;

            } else {
                ImageGray ref_frame, curr_frame;
                if(format == "pgm") {
                    ref_frame = loadPGM(ref_path);
                    curr_frame = loadPGM(curr_path);
                } else if(format == "raw") {
                    ref_frame = loadRaw(ref_path, width, height);
                    curr_frame = loadRaw(curr_path, width, height);
                } else {
                    throw runtime_error("Unsupported format: " + format);
                }

                chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
                vector<vector<MotionVector>> mv = matcher->matchGray(curr_frame, ref_frame, blockSize);
                for (size_t i = 0; i < mv.size(); i++) {
                    for (size_t j = 0; j < mv[i].size(); j++) {
                        std::cout << "mv[" << i << "][" << j << "] = ("
                                << mv[i][j].dx << ", " << mv[i][j].dy << ")\n";
                    }
                }
                chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end - start;
                cout << "Matching time: " << elapsed.count() << " s" << endl;
                
                string output_name = generateOutputFilename(isRange, range_start, i, ref_path, curr_path);
                string output_path = output_dir + "/" + output_name;
                savePPM(drawMotionVectors(curr_frame, ref_frame, mv, blockSize), output_path);
                cout << "Motion vectors saved in " << output_path << endl;
                
                // Save frame difference image
                size_t last_dot = output_path.rfind('.');
                string diff_path = output_path.substr(0, last_dot) + "_diff.ppm";
                savePPM(drawFrameDifference(ref_frame, curr_frame), diff_path);
                cout << "Frame difference saved in " << diff_path << endl;
                
                // Save reference frame with grid
                string ref_grid_path = output_path.substr(0, last_dot) + "_ref_grid.ppm";
                savePPM(drawFrameWithGrid(ref_frame, blockSize), ref_grid_path);
                cout << "Reference frame with grid saved in " << ref_grid_path << endl;
                
                // Save current frame with grid
                string curr_grid_path = output_path.substr(0, last_dot) + "_curr_grid.ppm";
                savePPM(drawFrameWithGrid(curr_frame, blockSize), curr_grid_path);
                cout << "Current frame with grid saved in " << curr_grid_path << endl;
            }
        } catch(const exception& e) {
            cerr << "Error during processing: " << e.what() << endl;
            return 1;
        }
    }

    return 0;
}
