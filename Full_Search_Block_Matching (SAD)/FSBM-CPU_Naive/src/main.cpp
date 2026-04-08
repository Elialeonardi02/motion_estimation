#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>               // for execution time measurement
#include <filesystem>           // output directory management
#include <fstream>              // for file operations and binary file reading(auto-detection of color channels in RAW format)
#include "types.h"      
#include "imageIO.h"
#include "motionEstimation.h"
#include "visualization.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    // Default parameters for input frames, output, and processing. 
    std::string format = "pgm";
    int width = 0, height = 0;
    bool isColor = false;        // default format is in grayscale

    // range-based input parameters
    std::string input_dir = "";
    std::string range_prefix = "";
    std::string range_suffix = "";
    int range_start = 0, range_end = 0;
    
    // individual image paths based input parameters
    std::vector<std::string> image_paths;

    // Parse command-line arguments
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--format") {     // argv[1] = --format, argv[2] = pgm/ppm/raw 
            if(i + 1 < argc) {
                format = argv[++i];     // format is specified by user
            } else {
                std::cerr << "Error: --format requires a value (pgm, ppm, or raw)" << std::endl;    // format is not specified by user
                return 1;
            }
        }else if(arg == "--width") {   // argv[i] = --width, argv[i+1] = width value (for RAW format) 
            if(i + 1 < argc) {
                width = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --width requires a numeric value" << std::endl;
                return 1;
            }
        } else if(arg == "--height") {  // argv[i] = --height, argv[i+1] = height value (for RAW format)
            if(i + 1 < argc) {
                height = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --height requires a numeric value" << std::endl;
                return 1;
            }
        } else if(arg == "--input-dir") {   // argv[i] = --input-dir, argv[i+1] = directory path
            if(i + 1 < argc) {
                input_dir = argv[++i];
            } else {
                std::cerr << "Error: --input-dir requires a directory name" << std::endl;
                return 1;
            }
        } else if(arg == "--range") { // argv[i] = --range, argv[i+1] = prefix_range, argv[i+2] = suffix_range, argv[i+3] = start_range, argv[i+4] = end_range
            if(i + 4 < argc) {
                range_prefix = argv[++i];
                range_suffix = argv[++i];
                range_start = std::atoi(argv[++i]);
                range_end = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --range requires 4 values (prefix suffix start end)" << std::endl;
                return 1;
            }
        } else {
            image_paths.push_back(arg); // treat as individual image path
        }
    }

    // generate file paths based on the provided prefix, suffix, and range (if -- range is specified)
    if(!range_prefix.empty()) {
        if(input_dir.empty()) { // range mandatory requires input directory to construct file paths
            std::cerr << "Error: --range requires --input-dir" << std::endl;
            return 1;
        }
        // Construct list of file paths based on the specified range parameters
        for(int num = range_start; num <= range_end; ++num) {
            char buffer[256];
            // Construct filename input using the specified prefix, suffix, and zero-padded frame number (e.g., frame_0001.pgm where frame is specified by prefix, .pgm is specified by suffix, and 0001 is the zero-padded frame number)
            std::snprintf(buffer, sizeof(buffer), "%s%04d%s", range_prefix.c_str(), num, range_suffix.c_str());
            std::string filename = input_dir + "/" + buffer;
            image_paths.push_back(filename);
        }
    }

    // Validate that at least one image path is provided (either through individual paths or generated from range)
    if(image_paths.empty()) {   // no images provided, stop execution and print usage instructions
        std::cerr << "Usage: " << argv[0] << " [--input-dir <dir>] [--range <prefix> <suffix> <start> <end>] [--format pgm|ppm|raw] [--width W] [--height H] <image1> [image2 ...]" << std::endl;
        std::cerr << "At least one image is required." << std::endl;
        std::cerr << "For RAW format, specify --width and --height." << std::endl;
        return 1;
    }

    // Raw format requires --width and --height to be specified.    
    if(format == "raw" && (width <= 0 || height <= 0)) {
        std::cerr << "Error: For RAW format, valid --width and --height must be specified." << std::endl;
        return 1;
    }

    // For RAW format, auto-detect if image is color based on file size
    if(format == "raw" && !image_paths.empty()) {
        std::string first_file = image_paths[0];
        std::ifstream file(first_file, std::ios::binary | std::ios::ate);
        if(!file.is_open()) {
            std::cerr << "Error: Cannot open file " << first_file << " to determine color channels" << std::endl;
            return 1;
        }
        std::streamsize file_size = file.tellg();
        file.close();
        
        // Auto-detect based on file size
        double bytes_per_pixel = static_cast<double>(file_size) / (width * height);
        
        if(bytes_per_pixel > 2.5) {  // >= 3 canali (RGB o RGBA)
            isColor = true;
            std::cout << "Auto-detected: Color image (" << bytes_per_pixel << " bytes/pixel)" << std::endl;
        } else if(bytes_per_pixel > 1.5) {  // tra 1.5 e 2.5 (potrebbe essere YUV o compresso)
            isColor = true;
            std::cout << "Auto-detected: Color image with possible compression (" << bytes_per_pixel << " bytes/pixel)" << std::endl;
        } else {  // <= 1.5 (grayscale)
            isColor = false;
            std::cout << "Auto-detected: Grayscale image (" << bytes_per_pixel << " bytes/pixel)" << std::endl;
        }
    }

    // Determine output directory with subdirectory based on input folder
    std::string output_dir = "output";
    std::string sub_dir = "";
    
    // Extract directory name from input path
    if(!input_dir.empty()) {
        sub_dir = getLastDirName(input_dir);
    } else if(!image_paths.empty()) {
        // Extract directory from first file path
        std::string first_path = image_paths[0];
        size_t pos = first_path.rfind('/');
        if(pos == std::string::npos) pos = first_path.rfind('\\');
        if(pos != std::string::npos && pos > 0) {
            std::string dir_path = first_path.substr(0, pos);
            size_t last_dir_pos = dir_path.rfind('/');
            if(last_dir_pos == std::string::npos) last_dir_pos = dir_path.rfind('\\');
            if(last_dir_pos != std::string::npos) {
                sub_dir = dir_path.substr(last_dir_pos + 1);
            } else {
                sub_dir = dir_path;
            }
        }
    }
    
    // Create the full output path
    if(!sub_dir.empty()) {
        output_dir = output_dir + "/" + sub_dir;
    }
    std::filesystem::create_directories(output_dir);

    bool isRange = !range_prefix.empty();
    const int blockSize = 16;
    const int searchRange = 16;

    // Process consecutive image pairs
    for(size_t i = 0; i < image_paths.size() - 1; ++i) {
        std::string ref_path = image_paths[i];
        std::string curr_path = image_paths[i + 1];

        std::cout << "Processing pair: " << ref_path << " -> " << curr_path << std::endl;

        try {
            if(isColor) {
                // RGB mode
                ImageColor ref_frame, curr_frame;
                if(format == "ppm") {
                    ref_frame = loadPPM(ref_path);
                    curr_frame = loadPPM(curr_path);
                } else if(format == "raw") {
                    ref_frame = loadRawRGB(ref_path, width, height);
                    curr_frame = loadRawRGB(curr_path, width, height);
                } else {
                    throw std::runtime_error("For RGB images, use 'ppm' or 'raw' format");
                }

                auto start = std::chrono::high_resolution_clock::now();
                auto mv = fsbmRGB(curr_frame, ref_frame, blockSize, searchRange);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "FSBM time: " << elapsed.count() << " s" << std::endl;

                std::string output_name;
                if(isRange) {
                    output_name = "motion_vectors_frame_" + std::to_string(range_start + i) + "_vs_" + std::to_string(range_start + i + 1) + ".ppm";
                } else {
                    int frame1 = extractFrameNumber(ref_path);
                    int frame2 = extractFrameNumber(curr_path);
                    if(frame1 > 0 && frame2 > 0) {
                        output_name = "motion_vectors_" + std::to_string(frame1) + "-" + std::to_string(frame2) + ".ppm";
                    } else {
                        output_name = "motion_vectors_" + std::to_string(i + 1) + "-" + std::to_string(i + 2) + ".ppm";
                    }
                }
                std::string output_path = output_dir + "/" + output_name;
                std::filesystem::create_directories(output_dir);
                savePPM(drawMotionVectorsRGB(curr_frame, mv, blockSize), output_path);
                std::cout << "Motion vectors saved in " << output_path << std::endl;
            } else {
                // Grayscale mode
                ImageGray ref_frame, curr_frame;
                if(format == "pgm") {
                    ref_frame = loadPGM(ref_path);
                    curr_frame = loadPGM(curr_path);
                } else if(format == "raw") {
                    ref_frame = loadRaw(ref_path, width, height);
                    curr_frame = loadRaw(curr_path, width, height);
                } else {
                    throw std::runtime_error("Unsupported format: " + format);
                }

                auto start = std::chrono::high_resolution_clock::now();
                auto mv = fsbm(curr_frame, ref_frame, blockSize, searchRange);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "FSBM time: " << elapsed.count() << " s" << std::endl;

                std::string output_name;
                if(isRange) {
                    output_name = "motion_vectors_frame_" + std::to_string(range_start + i) + "_vs_" + std::to_string(range_start + i + 1) + ".ppm";
                } else {
                    int frame1 = extractFrameNumber(ref_path);
                    int frame2 = extractFrameNumber(curr_path);
                    if(frame1 > 0 && frame2 > 0) {
                        output_name = "motion_vectors_" + std::to_string(frame1) + "-" + std::to_string(frame2) + ".ppm";
                    } else {
                        output_name = "motion_vectors_" + std::to_string(i + 1) + "-" + std::to_string(i + 2) + ".ppm";
                    }
                }
                std::string output_path = output_dir + "/" + output_name;
                std::filesystem::create_directories(output_dir);
                savePPM(drawMotionVectors(curr_frame, mv, blockSize), output_path);
                std::cout << "Motion vectors saved in " << output_path << std::endl;
            }
        } catch(const std::exception& e) {
            std::cerr << "Error during processing: " << e.what() << std::endl;
            return 1;
        }
    }

    if(image_paths.size() == 1) {
        std::cout << "Only one image provided. No motion estimation performed." << std::endl;
    }

    return 0;
}
