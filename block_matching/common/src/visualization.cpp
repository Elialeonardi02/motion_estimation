#include "visualization.h"
#include <cmath>
#include <string>

using namespace std;

// Draw a pixel with thickness (antialiasing effect)
void drawPixelThick(ImageColor& img, int x, int y, 
                    unsigned char r, unsigned char g, unsigned char b, 
                    int thickness) {
    for(int dy = -thickness; dy <= thickness; dy++) {
        for(int dx = -thickness; dx <= thickness; dx++) {
            int px = x + dx;
            int py = y + dy;
            if(px >= 0 && px < img.width && py >= 0 && py < img.height) {
                img.at(px, py, 0) = r;
                img.at(px, py, 1) = g;
                img.at(px, py, 2) = b;
            }
        }
    }
}

// Draw a line on color image using Bresenham algorithm with thickness
void drawLine(ImageColor& img, int x0, int y0, int x1, int y1,
              unsigned char r, unsigned char g, unsigned char b) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int thickness = 1; // line thickness

    while(true) {
        drawPixelThick(img, x0, y0, r, g, b, thickness);
        if(x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if(e2 >= dy) { err += dy; x0 += sx; }
        if(e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Draw block-aligned grid on a color image
void drawBlockGrid(ImageColor& img, int blockSize) {
    // Horizontal lines
    for(int y = 0; y < img.height; y += blockSize) {
        for(int x = 0; x < img.width; ++x) {
            img.at(x, y, 0) = 80;
            img.at(x, y, 1) = 80;
            img.at(x, y, 2) = 80;
        }
    }
    // Vertical lines
    for(int x = 0; x < img.width; x += blockSize) {
        for(int y = 0; y < img.height; ++y) {
            img.at(x, y, 0) = 80;
            img.at(x, y, 1) = 80;
            img.at(x, y, 2) = 80;
        }
    }
}

// Simple bitmap font for numbers 0-9
// Each number is 5x7 pixels (width x height)
static bool font_numbers[10][7][5] = {
    // 0
    {{1,1,1,1,1}, {1,0,0,0,1}, {1,0,0,0,1}, {1,0,0,0,1}, {1,0,0,0,1}, {1,0,0,0,1}, {1,1,1,1,1}},
    // 1
    {{0,0,1,0,0}, {0,1,1,0,0}, {0,0,1,0,0}, {0,0,1,0,0}, {0,0,1,0,0}, {0,0,1,0,0}, {0,1,1,1,0}},
    // 2
    {{1,1,1,1,0}, {0,0,0,0,1}, {0,0,0,0,1}, {1,1,1,1,0}, {1,0,0,0,0}, {1,0,0,0,0}, {1,1,1,1,1}},
    // 3
    {{1,1,1,1,0}, {0,0,0,0,1}, {0,0,0,0,1}, {0,1,1,1,0}, {0,0,0,0,1}, {0,0,0,0,1}, {1,1,1,1,0}},
    // 4
    {{1,0,0,0,1}, {1,0,0,0,1}, {1,0,0,0,1}, {1,1,1,1,1}, {0,0,0,0,1}, {0,0,0,0,1}, {0,0,0,0,1}},
    // 5
    {{1,1,1,1,1}, {1,0,0,0,0}, {1,0,0,0,0}, {1,1,1,1,0}, {0,0,0,0,1}, {0,0,0,0,1}, {1,1,1,1,0}},
    // 6
    {{0,1,1,1,1}, {1,0,0,0,0}, {1,0,0,0,0}, {1,1,1,1,0}, {1,0,0,0,1}, {1,0,0,0,1}, {1,1,1,1,0}},
    // 7
    {{1,1,1,1,1}, {0,0,0,0,1}, {0,0,0,1,0}, {0,0,1,0,0}, {0,1,0,0,0}, {1,0,0,0,0}, {1,0,0,0,0}},
    // 8
    {{1,1,1,1,0}, {1,0,0,0,1}, {1,0,0,0,1}, {0,1,1,1,0}, {1,0,0,0,1}, {1,0,0,0,1}, {1,1,1,1,0}},
    // 9
    {{1,1,1,1,0}, {1,0,0,0,1}, {1,0,0,0,1}, {1,1,1,1,1}, {0,0,0,0,1}, {0,0,0,0,1}, {1,1,1,1,0}}
};

// Draw a single digit at position (x, y)
void drawDigit(ImageColor& img, int x, int y, int digit, 
               unsigned char r, unsigned char g, unsigned char b) {
    if(digit < 0 || digit > 9) return;
    
    int scale = 2; // Scale factor for pixel size
    for(int row = 0; row < 7; row++) {
        for(int col = 0; col < 5; col++) {
            if(font_numbers[digit][row][col]) {
                int px = x + col * scale;
                int py = y + row * scale;
                for(int dy = 0; dy < scale && py + dy < img.height; dy++) {
                    for(int dx = 0; dx < scale && px + dx < img.width; dx++) {
                        if(px + dx >= 0 && py + dy >= 0) {
                            img.at(px + dx, py + dy, 0) = r;
                            img.at(px + dx, py + dy, 1) = g;
                            img.at(px + dx, py + dy, 2) = b;
                        }
                    }
                }
            }
        }
    }
}

// Draw a number (potentially multi-digit) at position (x, y)
void drawNumber(ImageColor& img, int x, int y, int number, 
                unsigned char r, unsigned char g, unsigned char b) {
    // Convert number to string and draw each digit
    string num_str = to_string(number);
    int digit_width = 12; // Width per digit (5 pixels + spacing)
    
    for(size_t i = 0; i < num_str.length(); i++) {
        int digit = num_str[i] - '0';
        drawDigit(img, x + i * digit_width, y, digit, r, g, b);
    }
}

// Draw block grid with coordinate numbering (row/column indices in pixel coordinates)
void drawBlockGridWithNumbers(ImageColor& img, int blockSize) {
    // First draw the grid
    drawBlockGrid(img, blockSize);
    
    // Draw column numbers at the top (1, 2, 3, ...)
    int blocksX = img.width / blockSize;
    int blocksY = img.height / blockSize;
    
    // Draw column numbers (top)
    for(int bx = 0; bx < blocksX; bx++) {
        int block_num = bx;
        int cx = bx * blockSize + blockSize / 2 - 10;
        int cy = 2;
        drawNumber(img, cx, cy, block_num, 200, 200, 200);
    }
    
    // Draw row numbers on the left (0, 1, 2, ...)
    for(int by = 0; by < blocksY; by++) {
        int block_num = by;
        int cx = 2;
        int cy = by * blockSize + blockSize / 2 - 3;
        drawNumber(img, cx, cy, block_num, 200, 200, 200);
    }
}

// Draw motion vectors on grayscale image (converted to RGB)
ImageColor drawMotionVectors(const ImageGray& frame,
                             const ImageGray& refFrame,
                             const vector<vector<MotionVector>>& mv,
                             int blockSize) {
    ImageColor img;
    img.width = frame.width;
    img.height = frame.height;
    img.data.resize(img.width * img.height * 3);

    // Convert grayscale to RGB and blend with reference frame (50% transparency)
    for(int y = 0; y < img.height; y++) {
        for(int x = 0; x < img.width; x++) {
            unsigned char curr_gray = frame.at(x, y);
            unsigned char ref_gray = refFrame.at(x, y);
            // Blend: 50% current frame, 50% reference frame
            unsigned char blended = (curr_gray / 2) + (ref_gray / 2);
            for(int c = 0; c < 3; c++)
                img.at(x, y, c) = blended;
        }
    }

    // Draw block-aligned grid with numbers (before vectors, so arrows render on top)
    drawBlockGridWithNumbers(img, blockSize);

    // Draw motion vectors (from reference block to current block)
    int min_length = 0; // Draw all vectors
    int step = 1; // Draw every block
    int cols = (mv.size() > 0) ? mv[0].size() : 0;
    for(int by = 0; by < static_cast<int>(mv.size()); by += step) {
        for(int bx = 0; bx < cols && bx < static_cast<int>(mv[by].size()); bx += step) {
            int dx = mv[by][bx].dx;
            int dy = mv[by][bx].dy;
            if(abs(dx) + abs(dy) < min_length) continue;
            
            // Current block center in pixels
            int cx = bx * blockSize + blockSize / 2;
            int cy = by * blockSize + blockSize / 2;
            
            // Reference block position and center in pixels
            int ref_bx = bx + dx;
            int ref_by = by + dy;
            int ref_cx = ref_bx * blockSize + blockSize / 2;
            int ref_cy = ref_by * blockSize + blockSize / 2;
            
            // Starting point (green dot at reference block)
            drawLine(img, ref_cx, ref_cy, ref_cx, ref_cy, 0, 255, 0);
            // Red motion vector line from reference to current
            if(abs(dx) + abs(dy) > 0) {
                drawLine(img, ref_cx, ref_cy, cx, cy, 255, 0, 0);
            }
            // V-shaped arrow head at current block position
            int arrow_len = 10;
            float len = sqrt((float)((cx - ref_cx) * (cx - ref_cx) + (cy - ref_cy) * (cy - ref_cy)));
            if(len > 0) {
                float nx = (cx - ref_cx) / len;
                float ny = (cy - ref_cy) / len;
                // Back center point
                int back_x = cx - arrow_len * nx;
                int back_y = cy - arrow_len * ny;
                // Left line of V
                int px1 = back_x - arrow_len * ny / 2;
                int py1 = back_y + arrow_len * nx / 2;
                drawLine(img, cx, cy, px1, py1, 255, 255, 255);
                // Right line of V
                int px2 = back_x + arrow_len * ny / 2;
                int py2 = back_y - arrow_len * nx / 2;
                drawLine(img, cx, cy, px2, py2, 255, 255, 255);
            }
            // Ending point (white dot at current block)
            drawLine(img, cx, cy, cx, cy, 255, 255, 255);
        }
    }
    return img;
}

// Draw motion vectors on RGB color image
ImageColor drawMotionVectorsRGB(const ImageColor& frame,
                                const ImageColor& refFrame,
                                const vector<vector<MotionVector>>& mv,
                                int blockSize) {
    ImageColor img;
    img.width = frame.width;
    img.height = frame.height;
    img.data.resize(img.width * img.height * 3);

    // Blend current and reference frames (50% transparency)
    for(int y = 0; y < img.height; y++) {
        for(int x = 0; x < img.width; x++) {
            for(int c = 0; c < 3; c++) {
                unsigned char curr_val = frame.at(x, y, c);
                unsigned char ref_val = refFrame.at(x, y, c);
                // Blend: 50% current frame, 50% reference frame
                img.at(x, y, c) = (curr_val / 2) + (ref_val / 2);
            }
        }
    }

    // Draw block-aligned grid with numbers (before vectors, so arrows render on top)
    drawBlockGridWithNumbers(img, blockSize);

    // Draw motion vectors (from reference block to current block)
    int min_length = 0; // Draw all vectors
    int step = 1; // Draw every block
    int cols = (mv.size() > 0) ? mv[0].size() : 0;
    for(int by = 0; by < static_cast<int>(mv.size()); by += step) {
        for(int bx = 0; bx < cols && bx < static_cast<int>(mv[by].size()); bx += step) {
            int dx = mv[by][bx].dx;
            int dy = mv[by][bx].dy;
            if(abs(dx) + abs(dy) < min_length) continue;
            
            // Current block center in pixels
            int cx = bx * blockSize + blockSize / 2;
            int cy = by * blockSize + blockSize / 2;
            
            // Reference block position and center in pixels
            int ref_bx = bx + dx;
            int ref_by = by + dy;
            int ref_cx = ref_bx * blockSize + blockSize / 2;
            int ref_cy = ref_by * blockSize + blockSize / 2;
            
            // Starting point (green dot at reference block)
            drawLine(img, ref_cx, ref_cy, ref_cx, ref_cy, 0, 255, 0);
            // Red motion vector line from reference to current
            if(abs(dx) + abs(dy) > 0) {
                drawLine(img, ref_cx, ref_cy, cx, cy, 255, 0, 0);
            }
            // V-shaped arrow head at current block position
            int arrow_len = 10;
            float len = sqrt((float)((cx - ref_cx) * (cx - ref_cx) + (cy - ref_cy) * (cy - ref_cy)));
            if(len > 0) {
                float nx = (cx - ref_cx) / len;
                float ny = (cy - ref_cy) / len;
                // Back center point
                int back_x = cx - arrow_len * nx;
                int back_y = cy - arrow_len * ny;
                // Left line of V
                int px1 = back_x - arrow_len * ny / 2;
                int py1 = back_y + arrow_len * nx / 2;
                drawLine(img, cx, cy, px1, py1, 255, 255, 255);
                // Right line of V
                int px2 = back_x + arrow_len * ny / 2;
                int py2 = back_y - arrow_len * nx / 2;
                drawLine(img, cx, cy, px2, py2, 255, 255, 255);
            }
            // Ending point (white dot at current block)
            drawLine(img, cx, cy, cx, cy, 255, 255, 255);
        }
    }
    return img;
}

// Draw frame difference on grayscale images
ImageColor drawFrameDifference(const ImageGray& frame1,
                               const ImageGray& frame2) {
    ImageColor img;
    img.width = frame1.width;
    img.height = frame1.height;
    img.data.resize(img.width * img.height * 3);

    // Calculate absolute difference and amplify for visualization
    int amplification = 4; // amplify difference to make it more visible
    
    for(int y = 0; y < img.height; y++) {
        for(int x = 0; x < img.width; x++) {
            unsigned char val1 = frame1.at(x, y);
            unsigned char val2 = frame2.at(x, y);
            int diff = abs(static_cast<int>(val1) - static_cast<int>(val2)) * amplification;
            
            // Clamp value to 0-255 range
            unsigned char diff_val = diff > 255 ? 255 : static_cast<unsigned char>(diff);
            
            // Store as grayscale (R=G=B for grayscale visualization)
            img.at(x, y, 0) = diff_val;
            img.at(x, y, 1) = diff_val;
            img.at(x, y, 2) = diff_val;
        }
    }
    
    return img;
}

// Draw frame difference on RGB color images
ImageColor drawFrameDifferenceRGB(const ImageColor& frame1,
                                  const ImageColor& frame2) {
    ImageColor img;
    img.width = frame1.width;
    img.height = frame1.height;
    img.data.resize(img.width * img.height * 3);

    // Calculate absolute difference and amplify for visualization
    int amplification = 4; // amplify difference to make it more visible
    
    for(int y = 0; y < img.height; y++) {
        for(int x = 0; x < img.width; x++) {
            // Calculate difference for each channel
            for(int c = 0; c < 3; c++) {
                unsigned char val1 = frame1.at(x, y, c);
                unsigned char val2 = frame2.at(x, y, c);
                int diff = abs(static_cast<int>(val1) - static_cast<int>(val2)) * amplification;
                
                // Clamp value to 0-255 range
                unsigned char diff_val = diff > 255 ? 255 : static_cast<unsigned char>(diff);
                img.at(x, y, c) = diff_val;
            }
        }
    }
    
    return img;
}

// Draw frame with block grid on grayscale image (converted to RGB)
ImageColor drawFrameWithGrid(const ImageGray& frame, int blockSize) {
    ImageColor img;
    img.width = frame.width;
    img.height = frame.height;
    img.data.resize(img.width * img.height * 3);

    // Convert grayscale to RGB
    for(int y = 0; y < img.height; y++)
        for(int x = 0; x < img.width; x++)
            for(int c = 0; c < 3; c++)
                img.at(x, y, c) = frame.at(x, y);

    // Draw block-aligned grid with numbers
    drawBlockGridWithNumbers(img, blockSize);
    
    return img;
}

// Draw frame with block grid on RGB color image
ImageColor drawFrameWithGridRGB(const ImageColor& frame, int blockSize) {
    ImageColor img = frame; // Copy RGB image
    
    // Draw block-aligned grid with numbers
    drawBlockGridWithNumbers(img, blockSize);
    
    return img;
}
