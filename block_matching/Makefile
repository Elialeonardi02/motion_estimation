NVXX            = nvcc
NVXXFLAGS       = -std=c++20 -x cu -Xcompiler -fPIE
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

CXXFLAGS        = -std=c++20 -Wall -Wextra -O2 -fPIE -fpic -fopenmp

OMP_CXX       = g++
OMP_GPU_FLAGS = -fopenmp


INCLUDE_DIRS    = -Icommon/include \
                  -Isolutions/full_search/include \
                  -Isolutions/full_search/cpu_naive \
                  -Isolutions/full_search/cuda_naive \
                  -Isolutions/full_search/cuda_optimized \
			      -Isolutions/full_search/cuda_optimized_uncoalesced \
				  -Isolutions/logarithmic_search/include \
				  -Isolutions/logarithmic_search/cpu_naive \
                  -Isolutions/logarithmic_search/cuda_optimized \
				  -Isolutions/full_search/cpu_OpenMP \
				  -Isolutions/logarithmic_search/gpu_OpenMP

# Directory structure
COMMON_SRC      = common/src
FSEARCH_CPU_SRC = solutions/full_search/cpu_naive
FSEARCH_CUDA_SRC = solutions/full_search/cuda_naive
FSEARCH_CUDA_OPTIMIZED_SRC = solutions/full_search/cuda_optimized
FSEARCH_CUDA_OPTIMIZED_UNCOALESCED_SRC = solutions/full_search/cuda_optimized_uncoalesced
FSEARCH_CPU_OPENMP_SRC = solutions/full_search/cpu_OpenMP
LSEARCH_CPU_SRC = solutions/logarithmic_search/cpu_naive
LSEARCH_CUDA_OPTIMIZED_SRC = solutions/logarithmic_search/cuda_optimized
LSEARCH_GPU_OPENMP_SRC = solutions/logarithmic_search/gpu_OpenMP

BUILD_DIR       = build
OUTPUT_DIR      = output

# Executable name
EXECUTABLE      = block_matching

# Check if nvcc is available 
HAVE_CUDA := $(shell command -v nvcc 2> /dev/null)

# Object files
COMMON_OBJECTS  = $(BUILD_DIR)/main.o \
                  $(BUILD_DIR)/imageIO.o \
                  $(BUILD_DIR)/visualization.o \
                  $(BUILD_DIR)/utils.o \
                  $(BUILD_DIR)/blockMatchingFactory.o

ifdef HAVE_CUDA
  COMMON_OBJECTS += $(BUILD_DIR)/cuda_utils.o
endif
 
FSEARCH_OBJECTS_CPU = $(BUILD_DIR)/fullSearchBM_cpu_naive.o \
				  	  $(BUILD_DIR)/fullSearchBM_cpu_OpenMP.o

LSEARCH_CPU_OBJECTS = $(BUILD_DIR)/logarithmicSearchBM_cpu_naive.o

FSEARCH_CUDA_OBJECTS = $(BUILD_DIR)/fullSearchBM_cuda_naive.o \
					   $(BUILD_DIR)/fullSearchBM_cuda_optimized.o \
					   $(BUILD_DIR)/fullSearchBM_cuda_optimized_uncoalesced.o

LSEARCH_CUDA_OBJECTS = $(BUILD_DIR)/logarithmicSearchBM_cuda_optimized.o

LSEARCH_GPU_OPENMP_OBJECTS = $(BUILD_DIR)/logarithmicSearchBM_gpu_OpenMP.o

ifdef HAVE_CUDA
  ALL_OBJECTS = $(COMMON_OBJECTS) $(FSEARCH_OBJECTS_CPU) $(LSEARCH_CPU_OBJECTS) $(FSEARCH_CUDA_OBJECTS) $(LSEARCH_CUDA_OBJECTS) #$(LSEARCH_GPU_OPENMP_OBJECTS)
else
  ALL_OBJECTS = $(COMMON_OBJECTS) $(FSEARCH_OBJECTS_CPU) $(LSEARCH_CPU_OBJECTS)
  CXXFLAGS += -DNO_CUDA
  $(warning nvcc not found. Building CPU-only version.)
endif

# Build targets
all: $(EXECUTABLE)

$(BUILD_DIR):
	@mkdir -p $@

# Compile common C++ files with g++
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

$(BUILD_DIR)/cuda_utils.o: $(COMMON_SRC)/cuda_utils.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

# Compile CPU implementations with g++
$(BUILD_DIR)/fullSearchBM_cpu_naive.o: $(FSEARCH_CPU_SRC)/fullSearchBM_cpu_naive.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/fullSearchBM_cpu_OpenMP.o: $(FSEARCH_CPU_OPENMP_SRC)/fullSearchBM_cpu_OpenMP.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/logarithmicSearchBM_cpu_naive.o: $(LSEARCH_CPU_SRC)/logarithmicSearchBM_cpu_naive.cpp | $(BUILD_DIR)
	g++ $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

#$(BUILD_DIR)/logarithmicSearchBM_gpu_OpenMP.o: $(LSEARCH_GPU_OPENMP_SRC)/logarithmicSearchBM_gpu_OpenMP.cpp | $(BUILD_DIR)
#	$(OMP_CXX) $(CXXFLAGS) $(OMP_GPU_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile CUDA kernels with nvcc (only if available)
ifdef HAVE_CUDA
$(BUILD_DIR)/fullSearchBM_cuda_naive.o: $(FSEARCH_CUDA_SRC)/fullSearchBM_cuda_naive.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

$(BUILD_DIR)/fullSearchBM_cuda_optimized.o: $(FSEARCH_CUDA_OPTIMIZED_SRC)/fullSearchBM_cuda_optimized.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

$(BUILD_DIR)/fullSearchBM_cuda_optimized_uncoalesced.o: $(FSEARCH_CUDA_OPTIMIZED_UNCOALESCED_SRC)/fullSearchBM_cuda_optimized_uncoalesced.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@

$(BUILD_DIR)/logarithmicSearchBM_cuda_optimized.o: $(LSEARCH_CUDA_OPTIMIZED_SRC)/logarithmicSearchBM_cuda_optimized.cpp | $(BUILD_DIR)
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $(INCLUDE_DIRS) $< -c -o $@
endif

# Link executable
ifdef HAVE_CUDA
  $(EXECUTABLE): $(ALL_OBJECTS)
	$(OMP_CXX) $(CXXFLAGS) $(OMP_GPU_FLAGS) $(ALL_OBJECTS) -o $@ -L/usr/local/cuda/lib64 -lcudart
	@echo "Build complete: $(EXECUTABLE)"
else
  $(EXECUTABLE): $(ALL_OBJECTS)
	$(OMP_CXX) $(CXXFLAGS) $(OMP_GPU_FLAGS) $(ALL_OBJECTS) -o $@
	@echo "Build complete: $(EXECUTABLE) " 
endif

# Cleanup targets
clean:
	@rm -f $(EXECUTABLE)
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

clean-all: clean
	@rm -rf $(OUTPUT_DIR)
	@echo "All cleaned!"

.DEFAULT_GOAL := all
.PHONY: all clean clean-all
