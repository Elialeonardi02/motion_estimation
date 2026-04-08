# FSBM-CPU_Naive: Full Search Block Matching Motion Estimation

A C++ implementation of Full Search Block Matching (FSBM) algorithm for motion estimation using Sum of Absolute Differences (SAD) cost function.

## Project Structure

```
FSBM-CPU_Naive/
├── include/              # Header files
│   ├── types.h          # Data structures (MotionVector, ImageGray, ImageColor)
│   ├── imageIO.h        # Image loading/saving functions
│   ├── motionEstimation.h  # FSBM and SAD computation functions
│   ├── visualization.h  # Motion vector visualization functions
│   └── utils.h          # Utility functions
├── src/                 # Source implementation files
│   ├── main.cpp         # Main program and command-line argument parsing
│   ├── imageIO.cpp      # Image format loaders (PGM, PPM, RAW)
│   ├── motionEstimation.cpp  # FSBM algorithm implementations
│   ├── visualization.cpp # Arrow drawing and motion vector visualization
│   └── utils.cpp        # Utility function implementations
├── output/              # Output directory for generated PPM images
├── Makefile             # Build configuration
└── README.md            # This file
```

## Features

- **Multiple Input Formats**: PGM (grayscale), PPM (RGB), and RAW binary formats
- **Grayscale and Color Processing**: Full support for both 8-bit grayscale and 24-bit RGB images
- **Range Mode**: Process sequences of numbered files automatically
- **Flexible Input**: Accept individual files or directory-based file sequences
- **Motion Vector Visualization**: Red vector lines with white V-shaped arrow heads
- **Organized Output**: Automatic subdirectory creation based on input folder names
- **Performance Timing**: Track FSBM computation time for each pair

## Algorithm Details

- **Block Size**: Fixed 16×16 pixels
- **Search Range**: ±16 pixels (full search window of 33×33)
- **Cost Function**: Sum of Absolute Differences (SAD)
- **Visualization**: 
  - Red motion vector lines (4× amplified)
  - White V-shaped arrow heads indicating direction
  - White start/end dots
  - Rendered every 5 blocks to reduce clutter
  - Minimum vector length threshold of 10 pixels

## Building

```bash
# Build the project
make

# Clean build artifacts
make clean

# Clean everything including output
make clean-all

# Show help
make help
```

## Usage

### Basic Usage - Process Individual Files

```bash
./fsbm frame1.pgm frame2.pgm frame3.pgm
```

### PGM Grayscale Images

```bash
./fsbm --format pgm image1.pgm image2.pgm
```

### PPM Color Images

```bash
./fsbm --format ppm --color image1.ppm image2.ppm
```

### RAW Format (Grayscale)

```bash
./fsbm --format raw --width 352 --height 288 image1.raw image2.raw
```

### RAW Format (RGB Color)

```bash
./fsbm --format raw --width 352 --height 288 image1.raw image2.raw
```

### Range Mode (Numbered File Sequence)

```bash
./fsbm --input-dir ./frames --range frame_ .raw 1 100 --format raw --width 640 --height 480
```

This generates file list: frame_0001.raw, frame_0002.raw, ..., frame_0100.raw

## Command-Line Options

```
--format <fmt>         Input format: 'pgm' (default), 'ppm', or 'raw'--color                Images are in RGB color format
--width <W>            Image width (required for RAW format)
--height <H>           Image height (required for RAW format)
--input-dir <dir>      Input directory for range mode
--range <pre> <suf> <start> <end>   Generate numbered files
                       Example: --range frame_ .raw 1 100
```

## Output

Motion vectors are visualized as PPM images (P6 format) in the `output/` subdirectory:
- **Range mode**: `output/[folder]/motion_vectors_frame_X_vs_Y.ppm`
- **Individual files**: `output/[folder]/motion_vectors_1-2.ppm`, `motion_vectors_2-3.ppm`, etc.

The subdirectory name is automatically extracted from the input path.

## Example Workflow

1. **Build the project**:
   ```bash
   cd FSBM-CPU_Naive
   make
   ```

2. **Process video frames**:
   ```bash
   ./fsbm --input-dir /path/to/frames --range frame_ .raw 1 100 \
          --format raw --width 640 --height 480
   ```

3. **View results**:
   - Open generated PPM images from `output/frames/` directory
   - Red lines indicate motion, arrows show direction

## Performance Notes

- Processing time depends on image resolution and motion complexity
- Full search is computationally intensive: O(image_area × search_range²)
- Typical performance: ~1-2 seconds per frame pair on modern CPUs for 352×288 grayscale

## Supported Formats

| Format | Extension | Color Mode | Bit Depth | Notes |
|--------|-----------|-----------|-----------|-------|
| PGM    | .pgm      | Grayscale | 8-bit     | Binary format (P5) |
| PPM    | .ppm      | RGB       | 24-bit    | Binary format (P6) |
| RAW    | .raw      | Both      | 8/24-bit  | Requires width/height |

## Implementation Notes

- Written in C++17 with standard library
- Header-only data structures for efficiency
- Modular design for easy extension
- Comprehensive error handling and validation
- Bresenham line algorithm with antialiasing for visualization

## License

This implementation is provided as-is for educational and research purposes.
