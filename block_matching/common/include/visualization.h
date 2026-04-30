#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include "types.h"

// Draw block-aligned grid on a color image
void drawBlockGrid(ImageColor& img, int blockSize);

// Draw block grid with coordinate numbering (row/column indices)
void drawBlockGridWithNumbers(ImageColor& img, int blockSize);

// Draw a pixel with thickness (antialiasing effect)
void drawPixelThick(ImageColor& img, int x, int y, 
                    unsigned char r, unsigned char g, unsigned char b, 
                    int thickness);

// Draw a line on color image using Bresenham algorithm with thickness
void drawLine(ImageColor& img, int x0, int y0, int x1, int y1,
              unsigned char r, unsigned char g, unsigned char b);

// Draw motion vectors on grayscale image (converted to RGB)
ImageColor drawMotionVectors(const ImageGray& frame,
                             const ImageGray& refFrame,
                             const std::vector<std::vector<MotionVector>>& mv,
                             int blockSize);

// Draw motion vectors on RGB color image
ImageColor drawMotionVectorsRGB(const ImageColor& frame,
                                const ImageColor& refFrame,
                                const std::vector<std::vector<MotionVector>>& mv,
                                int blockSize);

// Draw frame difference on grayscale images (shows regions with motion)
ImageColor drawFrameDifference(const ImageGray& frame1,
                               const ImageGray& frame2);

// Draw frame difference on RGB color images (shows regions with motion)
ImageColor drawFrameDifferenceRGB(const ImageColor& frame1,
                                  const ImageColor& frame2);

// Draw frame with block grid on grayscale image (converted to RGB)
ImageColor drawFrameWithGrid(const ImageGray& frame,
                             int blockSize);

// Draw frame with block grid on RGB color image
ImageColor drawFrameWithGridRGB(const ImageColor& frame,
                                int blockSize);

#endif // VISUALIZATION_H
