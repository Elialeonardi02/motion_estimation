# Author: Gabriele Mencagli
# Modified for block matching with CUDA support

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu -Xcompiler -fPIE
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

CXXFLAGS        = -std=c++17 -Wall -Wextra -O2 -fPIE -fpic

INCLUDE_DIRS    = -Icommon/include \
                  -Isolutions/full_search/include \
                  -Isolutions/full_search/implementations/cpu_naive \
                  -Isolutions/full_search/implementations/cuda_naive

# Directory structure
COMMON_SRC      = common/src
FSEARCH_SRC     = solutions/full_search
FSEARCH_CPU_SRC = solutions/full_search/implementations/cpu_naive
FSEARCH_CUDA_SRC = solutions/full_search/implementations/cuda_naive
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

FSEARCH_OBJECTS = $(BUILD_DIR)/fullSearchBM.o \
                  $(BUILD_DIR)/fullSearchBM_cpu_naive.o

FSEARCH_CUDA_OBJECTS = $(BUILD_DIR)/FullSearchBM_cuda_naive.o

ALL_OBJECTS = $(COMMON_OBJECTS) $(FSEARCH_OBJECTS) $(FSEARCH_CUDA_OBJECTS)

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

# Compile fullSearchBM.cpp with g++
$(BUILD_DIR)/fullSearchBM.o: $(FSEARCH_SRC)/src/fullSearchBM.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile fullSearchBM_cpu_naive.cpp with g++
$(BUILD_DIR)/fullSearchBM_cpu_naive.o: $(FSEARCH_CPU_SRC)/fullSearchBM_cpu_naive.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile FullSearchBM_cuda_naive.cpp with nvcc
$(BUILD_DIR)/FullSearchBM_cuda_naive.o: $(FSEARCH_CUDA_SRC)/FullSearchBM_cuda_naive.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

# Link executable with nvcc
$(EXECUTABLE): $(ALL_OBJECTS)
	$(NVXX) -Xcompiler -fPIE $(ALL_OBJECTS) -o $@
	@echo "Build complete! Executable: $(EXECUTABLE)"

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
