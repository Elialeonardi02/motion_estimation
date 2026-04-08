#include "visualization.h"
#include <cmath>

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
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
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

// Draw motion vectors on grayscale image (converted to RGB)
ImageColor drawMotionVectors(const ImageGray& frame,
                             const std::vector<std::vector<MotionVector>>& mv,
                             int blockSize) {
    ImageColor img;
    img.width = frame.width;
    img.height = frame.height;
    img.data.resize(img.width * img.height * 3);

    // Convert grayscale to RGB
    for(int y = 0; y < img.height; y++)
        for(int x = 0; x < img.width; x++)
            for(int c = 0; c < 3; c++)
                img.at(x, y, c) = frame.at(x, y);

    // Draw block-aligned grid (before vectors, so arrows render on top)
    drawBlockGrid(img, blockSize);

    // Draw motion vectors (amplified, significant only, every 5 blocks)
    int scale = 4;
    int min_length = 10;
    int step = 5; // draw every step blocks
    for(int by = 0; by < static_cast<int>(mv.size()); by += step) {
        for(int bx = 0; bx < static_cast<int>(mv[0].size()); bx += step) {
            int dx = mv[by][bx].dx;
            int dy = mv[by][bx].dy;
            if(std::abs(dx) + std::abs(dy) < min_length) continue;
            int cx = bx * blockSize + blockSize / 2;
            int cy = by * blockSize + blockSize / 2;
            dx *= scale;
            dy *= scale;
            // Starting point (white dot)
            drawLine(img, cx, cy, cx, cy, 255, 255, 255);
            // Red motion vector line
            drawLine(img, cx, cy, cx + dx, cy + dy, 255, 0, 0);
            // V-shaped arrow head
            int ex = cx + dx;
            int ey = cy + dy;
            int arrow_len = 16;
            float len = std::sqrt(dx * dx + dy * dy);
            if(len > 0) {
                float nx = dx / len;
                float ny = dy / len;
                // Back center point
                int back_x = ex - arrow_len * nx;
                int back_y = ey - arrow_len * ny;
                // Left line of V
                int px1 = back_x - arrow_len * ny / 2;
                int py1 = back_y + arrow_len * nx / 2;
                drawLine(img, ex, ey, px1, py1, 255, 255, 255);
                // Right line of V
                int px2 = back_x + arrow_len * ny / 2;
                int py2 = back_y - arrow_len * nx / 2;
                drawLine(img, ex, ey, px2, py2, 255, 255, 255);
            }
            // Ending point (white dot)
            drawLine(img, ex, ey, ex, ey, 255, 255, 255);
        }
    }
    return img;
}

// Draw motion vectors on RGB color image
ImageColor drawMotionVectorsRGB(const ImageColor& frame,
                                const std::vector<std::vector<MotionVector>>& mv,
                                int blockSize) {
    ImageColor img = frame; // Copy RGB image

    // Draw block-aligned grid (before vectors, so arrows render on top)
    drawBlockGrid(img, blockSize);

    // Draw motion vectors (amplified, significant only, every 5 blocks)
    int scale = 4;
    int min_length = 10;
    int step = 5; // draw every step blocks
    for(int by = 0; by < static_cast<int>(mv.size()); by += step) {
        for(int bx = 0; bx < static_cast<int>(mv[0].size()); bx += step) {
            int dx = mv[by][bx].dx;
            int dy = mv[by][bx].dy;
            if(std::abs(dx) + std::abs(dy) < min_length) continue;
            int cx = bx * blockSize + blockSize / 2;
            int cy = by * blockSize + blockSize / 2;
            dx *= scale;
            dy *= scale;
            // Starting point (white dot)
            drawLine(img, cx, cy, cx, cy, 255, 255, 255);
            // Red motion vector line
            drawLine(img, cx, cy, cx + dx, cy + dy, 255, 0, 0);
            // V-shaped arrow head
            int ex = cx + dx;
            int ey = cy + dy;
            int arrow_len = 16;
            float len = std::sqrt(dx * dx + dy * dy);
            if(len > 0) {
                float nx = dx / len;
                float ny = dy / len;
                // Back center point
                int back_x = ex - arrow_len * nx;
                int back_y = ey - arrow_len * ny;
                // Left line of V
                int px1 = back_x - arrow_len * ny / 2;
                int py1 = back_y + arrow_len * nx / 2;
                drawLine(img, ex, ey, px1, py1, 255, 255, 255);
                // Right line of V
                int px2 = back_x + arrow_len * ny / 2;
                int py2 = back_y - arrow_len * nx / 2;
                drawLine(img, ex, ey, px2, py2, 255, 255, 255);
            }
            // Ending point (white dot)
            drawLine(img, ex, ey, ex, ey, 255, 255, 255);
        }
    }
    return img;
}