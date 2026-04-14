#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <string>
#include "types.h"

// Load PGM (P5) grayscale image
ImageGray loadPGM(const std::string& filename);

// Load PPM (P6) RGB color image
ImageColor loadPPM(const std::string& filename);

// Load RAW grayscale image (binary format)
ImageGray loadRaw(const std::string& filename, int width, int height);

// Load RAW RGB color image (binary format)
ImageColor loadRawRGB(const std::string& filename, int width, int height);

// Save PPM (P6) RGB image to file
void savePPM(const ImageColor& img, const std::string& filename);

#endif // IMAGEIO_H
