#ifndef TYPES_H
#define TYPES_H

#include <vector>

// Motion vector structure: (dx, dy) displacement
struct MotionVector {
    int dx;
    int dy;
};

// Grayscale image structure
struct ImageGray {
    int width;
    int height;
    std::vector<unsigned char> data;

    unsigned char at(int x, int y) const { 
        return data[y * width + x]; 
    }
};

// RGB color image structure (interleaved RGB format)
struct ImageColor {
    int width;
    int height;
    std::vector<unsigned char> data; // RGB interleaved: R,G,B,R,G,B,...

    unsigned char& at(int x, int y, int c) { 
        return data[(y * width + x) * 3 + c]; 
    }
    
    unsigned char at(int x, int y, int c) const { 
        return data[(y * width + x) * 3 + c]; 
    }
};

#endif // TYPES_H
