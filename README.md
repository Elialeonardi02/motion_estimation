# Motion Estimation Project

This project implements block matching algorithms for motion estimation, including Full Search, Logarithmic Search, and Range Search, with different CPU (naive, OpenMP) and GPU (CUDA naive, optimized, uncoalesced) implementations. It processes image sequences to determine motion vectors between frames and visualizes the results.

## Project Structure

- `block_matching/`:
  - `benchmark/`: stores performance benchmark results.
  - `build/`: compiled object files and the executable.
  - `common/`:shared utility functions, image I/O, visualization tools, cuda utilities.
  - `input-images/`: example image sequences for testing.
  - `output/`: generated output images and motion vector visualizations (not in repo).
  - `profiling/`: NVIDIA Nsight Compute profiling reports.
  - `solutions/`: specific implementations of block matching algorithms (CPU, OpenMP, CUDA).

## Building the Project

To build the project, navigate to the `block_matching` directory and use `make`. The `Makefile` automatically detects CUDA availability.

```bash
cd block_matching
make
```

To clean the build artifacts:

```bash
make clean
# To also remove output directory:
make clean-all
```

## Running the Executable

The main executable `block_matching` is generated in the `block_matching/` directory.

**General Syntax:**

```bash
./block_matching [--algorithm <algorithm>] [--implementation <impl>] \
                 [--input_dir <dir>] [--input_range <prefix> <suffix> <start> <end>] \
                 [--format pgm|ppm|raw] [--width W] [--height H] [--color] \
                 [--block_size <size>] [--distance <dist>] [--range <search_range>] \
                 <image1> <image2> ...
```

**Key Parameters:**

* `--algorithm`: `full_search`, `logarithmic_search`, `range_search`.
* `--implementation`: `cpu_naive`, `cpu_openmp`, `cuda_naive`, `cuda_optimized`, `cuda_optimized_uncoalesced`.
* `--input_dir`: base directory for image sequences.
* `--input_range`: specify an image sequence using a prefix, suffix, start, and end frame number (e.g., `frame_00`, `.pgm`, `1`, `10`).
* `--block_size`: size of the macroblock (default: 32).
* `--distance`: maximum search distance for logarithmic search (default: -1, full frame).
* `--range`: search range in blocks for range search (default: -1, full frame).
* `image1 image2 ...`: paths to individual images for processing. At least two images are required.

## Implemented Algorithms and Implementations

| Algorithm | Implemented Backends |
| --- | --- |
| `full_search` | `cpu_naive`, `cpu_openmp`, `cuda_naive`, `cuda_optimized`, `cuda_optimized_uncoalesced` |
| `logarithmic_search` | `cpu_naive`, `cuda_optimized` |
| `range_search` | `cpu_openmp` |

## Examples

Here are examples for various algorithms and implementations running commands using the `diagonals/diagonal_256x256` image sequence.

**1. Full Search - CPU Naive**

```bash
./block_matching --algorithm full_search --implementation cpu_naive \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2
```

**2. Full Search - CPU OpenMP**

```bash
./block_matching --algorithm full_search --implementation cpu_openmp \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2
```

**3. Full Search - CUDA Naive**

```bash
./block_matching --algorithm full_search --implementation cuda_naive \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2
```

**4. Full Search - CUDA Optimized**

```bash
./block_matching --algorithm full_search --implementation cuda_optimized \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2
```

**5. Full Search - CUDA Optimized Uncoalesced**

```bash
./block_matching --algorithm full_search --implementation cuda_optimized_uncoalesced \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2
```

**6. Logarithmic Search - CPU Naive**

```bash
./block_matching --algorithm logarithmic_search --implementation cpu_naive \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2 --distance 8
```

**7. Logarithmic Search - CUDA Optimized**

```bash
./block_matching --algorithm logarithmic_search --implementation cuda_optimized \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2 --distance 8
```

**8. Range Search - CPU OpenMP**
(Note: Range search typically works with a specified search range `--range`)
```bash
./block_matching --algorithm range_search --implementation cpu_openmp \
                 --input_dir input-images/diagonals/diagonal_256x256 \
                 --input_range frame_ .pgm 1 2 --block_size 16 --range 4
```