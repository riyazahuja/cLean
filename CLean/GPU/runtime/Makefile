# Makefile for CUDA FFI Bridge

# CUDA paths (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Lean paths
LEAN_PATH := $(shell lean --print-prefix)
LEAN_INCLUDE := $(LEAN_PATH)/include

# Compiler flags
CXX := g++
CXXFLAGS := -std=c++14 -fPIC -O2 -Wall
INCLUDES := -I$(LEAN_INCLUDE) -I$(CUDA_PATH)/include
LDFLAGS := -shared -L$(CUDA_PATH)/lib64
LIBS := -lcudart -lcuda -lnvrtc

# Target
TARGET := libcuda_ffi.so

# Source files
SOURCES := cuda_ffi.cpp
OBJECTS := $(SOURCES:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built $(TARGET) successfully"

%.o: %.cpp cuda_ffi.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

install: $(TARGET)
	@mkdir -p ../../../.lake/build/lib
	cp $(TARGET) ../../../.lake/build/lib/
	@echo "Installed $(TARGET) to .lake/build/lib/"

.PHONY: all clean install
