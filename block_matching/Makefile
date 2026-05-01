# Author: Gabriele Mencagli
# Modified for block matching with CUDA support

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu -Xcompiler -fPIE
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

CXXFLAGS        = -std=c++17 -Wall -Wextra -O2 -fPIE -fpic

INCLUDE_DIRS    = -Icommon/include \
                  -Isolutions/full_search/include \
                  -Isolutions/full_search/cpu_naive \
                  -Isolutions/full_search/cuda_naive \
                  -Isolutions/full_search/cuda_optimized \
				  -Isolutions/logarithmic_search/include \
				  -Isolutions/logarithmic_search/cpu_naive

# Directory structure
COMMON_SRC      = common/src
FSEARCH_SRC     = full_search
FSEARCH_CPU_SRC = solutions/full_search/cpu_naive
FSEARCH_CUDA_SRC = solutions/full_search/cuda_naive
FSEARCH_CUDA_OPTIMIZED_SRC = solutions/full_search/cuda_optimized
LSEARCH_CPU_SRC = solutions/logarithmic_search/cpu_naive
BUILD_DIR       = build
OUTPUT_DIR      = output

# Executable name
EXECUTABLE      = block_matching

# Object files
COMMON_OBJECTS  = $(BUILD_DIR)/main.o \
                  $(BUILD_DIR)/imageIO.o \
                  $(BUILD_DIR)/visualization.o \
                  $(BUILD_DIR)/utils.o \
                  $(BUILD_DIR)/blockMatchingFactory.o

FSEARCH_OBJECTS = $(BUILD_DIR)/fullSearchBM_cpu_naive.o

LSEARCH_CPU_OBJECTS = $(BUILD_DIR)/logarithmicSearchBM_cpu_naive.o

FSEARCH_CUDA_OBJECTS = $(BUILD_DIR)/FullSearchBM_cuda_naive.o \
                       $(BUILD_DIR)/FullSearchBM_cuda_optimized.o


# Check if nvcc is available
HAVE_CUDA := $(shell command -v nvcc 2> /dev/null)

ifdef HAVE_CUDA
  ALL_OBJECTS = $(COMMON_OBJECTS) $(FSEARCH_OBJECTS) $(LSEARCH_CPU_OBJECTS) $(FSEARCH_CUDA_OBJECTS)
else
  ALL_OBJECTS = $(COMMON_OBJECTS) $(FSEARCH_OBJECTS) $(LSEARCH_CPU_OBJECTS)
  CXXFLAGS += -DNO_CUDA
  $(warning nvcc not found. Building CPU-only version.)
endif

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $@

# Compile common object files with g++
$(BUILD_DIR)/main.o: $(COMMON_SRC)/main.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/imageIO.o: $(COMMON_SRC)/imageIO.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/visualization.o: $(COMMON_SRC)/visualization.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/utils.o: $(COMMON_SRC)/utils.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/blockMatchingFactory.o: $(COMMON_SRC)/blockMatchingFactory.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile fullSearchBM_cpu_naive.cpp with g++
$(BUILD_DIR)/fullSearchBM_cpu_naive.o: $(FSEARCH_CPU_SRC)/fullSearchBM_cpu_naive.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile logarithmicSearchBM_cpu_naive.cpp with g++
$(BUILD_DIR)/logarithmicSearchBM_cpu_naive.o: $(LSEARCH_CPU_SRC)/logarithmicSearchBM_cpu_naive.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile FullSearchBM_cuda_naive.cpp with nvcc (only if CUDA is available)
ifdef HAVE_CUDA
$(BUILD_DIR)/FullSearchBM_cuda_naive.o: $(FSEARCH_CUDA_SRC)/FullSearchBM_cuda_naive.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

# Compile FullSearchBM_cuda_optimized.cpp with nvcc (only if CUDA is available)
$(BUILD_DIR)/FullSearchBM_cuda_optimized.o: $(FSEARCH_CUDA_OPTIMIZED_SRC)/FullSearchBM_cuda_optimized.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@
endif

# Link executable (use nvcc if available, otherwise g++)
ifdef HAVE_CUDA
  $(EXECUTABLE): $(ALL_OBJECTS)
	$(NVXX) -Xcompiler -fPIE $(ALL_OBJECTS) -o $@
	@echo "Build complete! Executable: $(EXECUTABLE)"
else
  $(EXECUTABLE): $(ALL_OBJECTS)
	g++ $(CXXFLAGS) $(ALL_OBJECTS) -o $@
	@echo "Build complete! Executable: $(EXECUTABLE) (CPU-only)"
endif

# Clean build artifacts
clean:
	@rm -f $(EXECUTABLE)
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

# Clean everything including output
clean-all: clean
	@rm -rf $(OUTPUT_DIR)
	@echo "All cleaned!"

.DEFAULT_GOAL := all
.PHONY: all clean clean-all
