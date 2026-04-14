#include "imageIO.h"
#include <fstream>
#include <stdexcept>

using namespace std;

// Load PGM (P5) grayscale image
ImageGray loadPGM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open PGM file: " + filename);
    
    string format;
    file >> format;
    if(format != "P5") throw runtime_error("Unsupported PGM format");

    ImageGray img;
    int maxval;
    file >> img.width >> img.height >> maxval;
    file.get(); // consume newline
    img.data.resize(img.width * img.height);
    file.read((char*)img.data.data(), img.data.size());
    if (!file) throw runtime_error("Failed to read PGM file: " + filename);
    return img;
}

// Load PPM (P6) RGB color image
ImageColor loadPPM(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open PPM file: " + filename);
    
    string format;
    file >> format;
    if(format != "P6") throw runtime_error("Unsupported PPM format, use P6");

    ImageColor img;
    int maxval;
    file >> img.width >> img.height >> maxval;
    file.get(); // consume newline
    img.data.resize(img.width * img.height * 3);
    file.read((char*)img.data.data(), img.data.size());
    if (!file) throw runtime_error("Failed to read PPM file: " + filename);
    return img;
}

// Load RAW grayscale image
ImageGray loadRaw(const string& filename, int width, int height) {
    ImageGray img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);
    ifstream file(filename, ios::binary);
    if(!file) throw runtime_error("Cannot open RAW file: " + filename);
    file.read((char*)img.data.data(), img.data.size());
    if(file.gcount() != static_cast<streamsize>(img.data.size())) throw runtime_error("RAW file too small");
    return img;
}

// Load RAW RGB color image
ImageColor loadRawRGB(const string& filename, int width, int height) {
    ImageColor img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height * 3);
    ifstream file(filename, ios::binary);
    if(!file) throw runtime_error("Cannot open RAW RGB file: " + filename);
    file.read((char*)img.data.data(), img.data.size());
    if(file.gcount() != static_cast<streamsize>(img.data.size())) throw runtime_error("RAW RGB file too small");
    return img;
}

// Save PPM (P6) RGB image to file
void savePPM(const ImageColor& img, const string& filename) {
    ofstream out(filename, ios::binary);
    if(!out) throw runtime_error("Cannot create PPM file: " + filename);
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write((char*)img.data.data(), img.data.size());
    if(!out) throw runtime_error("Failed to write PPM file: " + filename);
}
