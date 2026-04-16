# Makefile for Block Matching project
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDE_DIRS := -Icommon/include \
                -Isolutions/full_search/include \
                -Isolutions/full_search/implementations/cpu_naive \
                -Isolutions/full_search/implementations/cuda_naive

# Directory structure
COMMON_SRC := common/src
COMMON_BUILD := build/common
FSEARCH_SRC := solutions/full_search
FSEARCH_CPU_SRC := solutions/full_search/implementations/cpu_naive
FSEARCH_BUILD := build/full_search
BUILD_DIR := build
OUTPUT_DIR := output

# Source files
COMMON_SOURCES := $(COMMON_SRC)/main.cpp \
                  $(COMMON_SRC)/imageIO.cpp \
                  $(COMMON_SRC)/visualization.cpp \
                  $(COMMON_SRC)/utils.cpp \
                  $(COMMON_SRC)/blockMatchingFactory.cpp

FSEARCH_SOURCES := $(FSEARCH_SRC)/src/fullSearchBM.cpp

FSEARCH_CPU_SOURCES := $(FSEARCH_CPU_SRC)/fullSearchBM_cpu_naive.cpp

FSEARCH_CUDA_SRC := solutions/full_search/implementations/cuda_naive
FSEARCH_CUDA_SOURCES := $(FSEARCH_CUDA_SRC)/FullSearchBM_cuda_naive.cpp

ALL_SOURCES := $(COMMON_SOURCES) $(FSEARCH_SOURCES) $(FSEARCH_CPU_SOURCES) $(FSEARCH_CUDA_SOURCES)

# Object files
COMMON_OBJECTS := $(COMMON_SOURCES:$(COMMON_SRC)/%.cpp=$(COMMON_BUILD)/%.o)
FSEARCH_OBJECTS := $(FSEARCH_SOURCES:$(FSEARCH_SRC)/src/%.cpp=$(FSEARCH_BUILD)/%.o)
FSEARCH_CPU_OBJECTS := $(FSEARCH_CPU_SOURCES:$(FSEARCH_CPU_SRC)/%.cpp=$(FSEARCH_BUILD)/%.o)
FSEARCH_CUDA_OBJECTS := $(FSEARCH_CUDA_SOURCES:$(FSEARCH_CUDA_SRC)/%.cpp=$(FSEARCH_BUILD)/%.o)

ALL_OBJECTS := $(COMMON_OBJECTS) $(FSEARCH_OBJECTS) $(FSEARCH_CPU_OBJECTS) $(FSEARCH_CUDA_OBJECTS)

# Executable name
EXECUTABLE := block_matching

# Default target
all: $(EXECUTABLE)

# Create build directories
$(COMMON_BUILD) $(FSEARCH_BUILD):
	@mkdir -p $@

# Compile common object files
$(COMMON_BUILD)/%.o: $(COMMON_SRC)/%.cpp | $(COMMON_BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile full search source files
$(FSEARCH_BUILD)/fullSearchBM.o: $(FSEARCH_SRC)/src/fullSearchBM.cpp | $(FSEARCH_BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile full search CPU naive implementation files
$(FSEARCH_BUILD)/fullSearchBM_cpu_naive.o: $(FSEARCH_CPU_SRC)/fullSearchBM_cpu_naive.cpp | $(FSEARCH_BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Compile full search CUDA naive implementation files
$(FSEARCH_BUILD)/FullSearchBM_cuda_naive.o: $(FSEARCH_CUDA_SRC)/FullSearchBM_cuda_naive.cpp | $(FSEARCH_BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Link executable
$(EXECUTABLE): $(ALL_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build complete! Executable: $(EXECUTABLE)"

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR) $(EXECUTABLE)
	@echo "Clean complete!"

# Clean everything including output
clean-all: clean
	@rm -rf $(OUTPUT_DIR)
	@echo "All cleaned!"

# Run the program (requires arguments)
run: $(EXECUTABLE)
	@./$(EXECUTABLE)

.PHONY: all clean clean-all run
