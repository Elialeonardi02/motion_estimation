# Block Matching - Motion Estimation

A modular block matching framework for motion estimation supporting multiple algorithms and implementations.

## Project Structure

```
block_matching/
├── common/                      # Common utilities and main executable
│   ├── include/
│   │   ├── types.h             # Data structures (MotionVector, ImageGray, ImageColor)
│   │   ├── imageIO.h           # Image loading/saving functions
│   │   ├── visualization.h     # Motion vector visualization
│   │   ├── utils.h             # Utility functions
│   │   └── blockMatchingInterface.h  # Abstract interface for algorithms
│   └── src/
│       ├── main.cpp            # Main entry point with algorithm selection
│       ├── imageIO.cpp         # Image I/O implementation
│       ├── visualization.cpp   # Visualization implementation
│       ├── utils.cpp           # Utility implementation
│       └── blockMatchingFactory.cpp  # Factory for creating algorithm instances
├── solutions/                   # Algorithm implementations
│   └── full_search/            # Full Search Block Matching
│       ├── include/
│       │   └── fullSearchBM.h
│       ├── src/
│       │   └── fullSearchBM.cpp
│       └── implementations/
│           └── cpu_naive/      # CPU Naive implementation
│               ├── fullSearchBM_cpu_naive.h
│               └── fullSearchBM_cpu_naive.cpp
├── input-images/               # Input image directory
├── output/                      # Output directory (auto-created)
│   ├── full_search_cpu_naive/  # Full Search CPU Naive outputs
│   │   └── <dataset_name>/     # Organized by dataset
├── build/                       # Build artifacts
└── Makefile                     # Build configuration

```

## Build

```bash
cd /home/e.leonardi5/motion_estimation/block_matching
make clean
make
```

## Usage

### Full Search CPU Naive (default)

Using PGM images with range:
```bash
./block_matching \
  --algorithm full_search \
  --implementation cpu_naive \
  --input-dir input-images/FIMI0002 \
  --range frame_ .pgm 1 5 \
  --format pgm
```

Using individual image paths:
```bash
./block_matching \
  input-images/frame1.pgm \
  input-images/frame2.pgm
```

### Output

Results are saved to:
- `output/full_search_cpu_naive/<dataset_name>/motion_vectors_*.ppm`

## Extending with New Implementations

To add a new implementation (e.g., CUDA):

1. Create directory: `solutions/full_search/implementations/cuda/`
2. Implement `fullSearchBM_cuda.cu`
3. Update `fullSearchBM.cpp` to instantiate the CUDA implementation
4. Update `blockMatchingFactory.cpp` to recognize `cuda` implementation
5. Update `Makefile` to compile CUDA files
6. Run: `./block_matching --algorithm full_search --implementation cuda ...`

## Extending with New Algorithms

To add a new algorithm (e.g., Region Search):

1. Create directory: `solutions/region_search/`
2. Create header: `solutions/region_search/include/regionSearchBM.h`
3. Create implementation class inheriting from `BlockMatcher`
4. Update `blockMatchingFactory.cpp` to handle new algorithm
5. Update `Makefile` to compile new algorithm
6. Run: `./block_matching --algorithm region_search --implementation cpu_naive ...`
