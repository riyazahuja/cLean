/*
 * Generic GPU Kernel Launcher
 *
 * Loads a compiled PTX file and executes it with parameters from stdin.
 * Communicates via JSON for easy Lean interop.
 *
 * Usage:
 *   ./gpu_launcher <ptx_file> <kernel_name> <grid_x> <grid_y> <grid_z> <block_x> <block_y> <block_z>
 *
 * Input (stdin): JSON with parameters and arrays
 * Output (stdout): JSON with result arrays
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

// Minimal JSON parsing (just for simple float arrays and scalars)
struct LaunchParams {
    std::vector<float> scalarParams;
    std::vector<std::pair<std::string, std::vector<float>>> arrays;
};

// CUDA error checking
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

#define CU_CHECK(call) do {                                             \
    CUresult err = call;                                                \
    if (err != CUDA_SUCCESS) {                                          \
        const char* errStr;                                             \
        cuGetErrorString(err, &errStr);                                 \
        std::cerr << "CUDA Driver error: " << errStr                   \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

// Simple stdin reader
std::string readStdin() {
    std::string line;
    // Read single line (JSON is on one line, no need to wait for EOF)
    std::getline(std::cin, line);
    return line;
}

// Parse simple JSON (very basic - just for our use case)
LaunchParams parseInput(const std::string& input) {
    LaunchParams params;

    // Parse scalar params: look for "scalars": [1.0, 2.0, ...]
    size_t scalarsPos = input.find("\"scalars\":");
    if (scalarsPos != std::string::npos) {
        size_t startBracket = input.find('[', scalarsPos);
        size_t endBracket = input.find(']', startBracket);
        std::string scalarsStr = input.substr(startBracket + 1, endBracket - startBracket - 1);

        std::stringstream ss(scalarsStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            params.scalarParams.push_back(std::stof(token));
        }
    }

    // Parse arrays: look for "arrays": {"name": [data], ...}
    size_t arraysPos = input.find("\"arrays\":");
    if (arraysPos != std::string::npos) {
        // Simple parser: find array name and data
        size_t pos = arraysPos;
        while (true) {
            pos = input.find("\"", pos + 1);
            if (pos == std::string::npos || pos > input.find('}', arraysPos)) break;

            size_t nameEnd = input.find("\"", pos + 1);
            std::string name = input.substr(pos + 1, nameEnd - pos - 1);

            // Skip if this is a JSON key like "arrays"
            if (name == "arrays" || name == "scalars") continue;

            // Find the array data
            size_t dataStart = input.find('[', nameEnd);
            if (dataStart == std::string::npos) break;
            size_t dataEnd = input.find(']', dataStart);

            std::string dataStr = input.substr(dataStart + 1, dataEnd - dataStart - 1);

            std::vector<float> data;
            std::stringstream ss(dataStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                data.push_back(std::stof(token));
            }

            params.arrays.push_back({name, data});
            pos = dataEnd;
        }
    }

    return params;
}

// Load PTX file
std::vector<char> loadPTX(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size + 1);  // +1 for null terminator
    file.read(buffer.data(), size);
    buffer[size] = '\0';

    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <ptx_file> <kernel_name> <grid_x> <grid_y> <grid_z> <block_x> <block_y> <block_z>"
                  << std::endl;
        return 1;
    }

    // Parse command line arguments
    std::string ptxFile = argv[1];
    std::string kernelName = argv[2];
    dim3 gridDim(std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
    dim3 blockDim(std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8]));

    // Read input from stdin
    std::string input = readStdin();
    LaunchParams params = parseInput(input);

    std::cerr << "[Launcher] PTX: " << ptxFile << std::endl;
    std::cerr << "[Launcher] Kernel: " << kernelName << std::endl;
    std::cerr << "[Launcher] Grid: " << gridDim.x << "x" << gridDim.y << "x" << gridDim.z << std::endl;
    std::cerr << "[Launcher] Block: " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z << std::endl;
    std::cerr << "[Launcher] Scalars: " << params.scalarParams.size() << std::endl;
    std::cerr << "[Launcher] Arrays: " << params.arrays.size() << std::endl;

    // Initialize CUDA driver API
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    CUcontext context;
    CU_CHECK(cuCtxCreate(&context, 0, device));

    // Load PTX module
    std::vector<char> ptxData = loadPTX(ptxFile);
    CUmodule module;
    CU_CHECK(cuModuleLoadData(&module, ptxData.data()));

    // Get kernel function
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, kernelName.c_str()));

    // Allocate device memory for arrays
    std::vector<CUdeviceptr> deviceArrays;
    for (const auto& arr : params.arrays) {
        CUdeviceptr devPtr;
        size_t sizeBytes = arr.second.size() * sizeof(float);
        CU_CHECK(cuMemAlloc(&devPtr, sizeBytes));
        CU_CHECK(cuMemcpyHtoD(devPtr, arr.second.data(), sizeBytes));
        deviceArrays.push_back(devPtr);
    }

    // Prepare kernel arguments
    std::vector<void*> kernelArgs;

    // Add scalar parameters
    for (float& scalar : params.scalarParams) {
        kernelArgs.push_back(&scalar);
    }

    // Add array pointers
    for (CUdeviceptr& ptr : deviceArrays) {
        kernelArgs.push_back(&ptr);
    }

    // Launch kernel
    CU_CHECK(cuLaunchKernel(
        kernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,  // shared memory
        nullptr,  // stream
        kernelArgs.data(),
        nullptr
    ));

    // Synchronize
    CU_CHECK(cuCtxSynchronize());

    // Copy results back
    std::cout << "{\"results\":{";
    bool first = true;
    for (size_t i = 0; i < params.arrays.size(); ++i) {
        const auto& arr = params.arrays[i];
        std::vector<float> result(arr.second.size());

        size_t sizeBytes = result.size() * sizeof(float);
        CU_CHECK(cuMemcpyDtoH(result.data(), deviceArrays[i], sizeBytes));

        if (!first) std::cout << ",";
        first = false;

        std::cout << "\"" << arr.first << "\":[";
        for (size_t j = 0; j < result.size(); ++j) {
            if (j > 0) std::cout << ",";
            std::cout << result[j];
        }
        std::cout << "]";
    }
    std::cout << "}}" << std::endl;

    // Cleanup
    for (CUdeviceptr ptr : deviceArrays) {
        cuMemFree(ptr);
    }

    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
