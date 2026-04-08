# Makefile for FSBM-CPU_Naive project
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDE_DIR := include
SRC_DIR := src
BUILD_DIR := build
OUTPUT_DIR := output

# Source files
SOURCES := $(SRC_DIR)/main.cpp \
           $(SRC_DIR)/imageIO.cpp \
           $(SRC_DIR)/motionEstimation.cpp \
           $(SRC_DIR)/visualization.cpp \
           $(SRC_DIR)/utils.cpp

# Object files
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Executable name
EXECUTABLE := fsbm

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Link executable
$(EXECUTABLE): $(OBJECTS)
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

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build artifacts and executable"
	@echo "  clean-all    - Remove build artifacts, executable, and output directory"
	@echo "  run          - Run the executable (requires arguments)"
	@echo "  clear        - Clear terminal screen"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make"
	@echo "  ./fsbm --input-dir ./input-images/claire --format raw --width 352 --height 288 ./input-images/claire/claire1.raw ./input-images/claire/claire2.raw"

.PHONY: all clean clean-all run clear help
