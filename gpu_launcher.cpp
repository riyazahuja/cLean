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
#include <variant>

// Scalar can be either int or float
struct ScalarParam {
    std::variant<int, float> value;
    bool isInt;
};

// Array can be either int or float
struct ArrayParam {
    std::string name;
    std::variant<std::vector<int>, std::vector<float>> data;
    bool isInt;
};

// JSON parsing results
struct LaunchParams {
    std::vector<ScalarParam> scalarParams;
    std::vector<ArrayParam> arrays;
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

// Simple JSON Tokenizer
enum TokenType { TOKEN_LBRACE, TOKEN_RBRACE, TOKEN_LBRACKET, TOKEN_RBRACKET, TOKEN_COLON, TOKEN_COMMA, TOKEN_STRING, TOKEN_NUMBER, TOKEN_EOF };

struct Token {
    TokenType type;
    std::string value;
    size_t pos;
};

std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < input.length()) {
        char c = input[i];
        if (isspace(c)) {
            i++;
            continue;
        }
        if (c == '{') tokens.push_back({TOKEN_LBRACE, "{", i++});
        else if (c == '}') tokens.push_back({TOKEN_RBRACE, "}", i++});
        else if (c == '[') tokens.push_back({TOKEN_LBRACKET, "[", i++});
        else if (c == ']') tokens.push_back({TOKEN_RBRACKET, "]", i++});
        else if (c == ':') tokens.push_back({TOKEN_COLON, ":", i++});
        else if (c == ',') tokens.push_back({TOKEN_COMMA, ",", i++});
        else if (c == '"') {
            size_t start = ++i;
            while (i < input.length() && input[i] != '"') i++;
            tokens.push_back({TOKEN_STRING, input.substr(start, i - start), start});
            if (i < input.length()) i++;
        } else {
            size_t start = i;
            while (i < input.length() && (isdigit(input[i]) || input[i] == '.' || input[i] == '-' || input[i] == 'e' || input[i] == 'E' || input[i] == '+')) i++;
            tokens.push_back({TOKEN_NUMBER, input.substr(start, i - start), start});
        }
    }
    tokens.push_back({TOKEN_EOF, "", i});
    return tokens;
}

// Check if a number string represents an integer (no decimal point, no exponent)
bool isIntegerString(const std::string& s) {
    for (char c : s) {
        if (c == '.' || c == 'e' || c == 'E') return false;
    }
    return true;
}

// Parse a scalar object with explicit type: {"type":"int","value":2}
ScalarParam parseScalarObject(std::vector<Token>& tokens, size_t& current) {
    ScalarParam sp;
    sp.isInt = false; // default to float
    
    auto expect = [&](TokenType type) {
        if (current < tokens.size() && tokens[current].type == type) {
            current++;
            return true;
        }
        return false;
    };
    auto consume = [&]() { if (current < tokens.size()) current++; };
    
    expect(TOKEN_LBRACE);
    std::string typeStr = "float";
    std::string valueStr = "0";
    
    while (current < tokens.size() && tokens[current].type != TOKEN_RBRACE) {
        if (tokens[current].type == TOKEN_STRING) {
            std::string key = tokens[current].value;
            consume();
            expect(TOKEN_COLON);
            if (key == "type" && tokens[current].type == TOKEN_STRING) {
                typeStr = tokens[current].value;
                consume();
            } else if (key == "value" && tokens[current].type == TOKEN_NUMBER) {
                valueStr = tokens[current].value;
                consume();
            }
        }
        if (tokens[current].type == TOKEN_COMMA) consume();
    }
    expect(TOKEN_RBRACE);
    
    if (typeStr == "int") {
        sp.isInt = true;
        sp.value = std::stoi(valueStr);
    } else {
        sp.isInt = false;
        sp.value = std::stof(valueStr);
    }
    return sp;
}

// Robust JSON Parser
LaunchParams parseInput(const std::string& input) {
    LaunchParams params;
    auto tokens = tokenize(input);
    size_t current = 0;

    auto expect = [&](TokenType type) {
        if (current < tokens.size() && tokens[current].type == type) {
            current++;
            return true;
        }
        return false;
    };

    auto consume = [&]() {
        if (current < tokens.size()) current++;
    };

    if (!expect(TOKEN_LBRACE)) return params;

    while (current < tokens.size() && tokens[current].type != TOKEN_RBRACE) {
        if (tokens[current].type != TOKEN_STRING) break;
        std::string key = tokens[current].value;
        consume(); // key
        expect(TOKEN_COLON);

        if (key == "scalars") {
            expect(TOKEN_LBRACKET);
            while (current < tokens.size() && tokens[current].type != TOKEN_RBRACKET) {
                // Check if scalar is an object with type info or a plain number
                if (tokens[current].type == TOKEN_LBRACE) {
                    // New format: {"type":"int","value":2}
                    ScalarParam sp = parseScalarObject(tokens, current);
                    params.scalarParams.push_back(sp);
                } else if (tokens[current].type == TOKEN_NUMBER) {
                    // Legacy format: plain number, infer type from format
                    const std::string& numStr = tokens[current].value;
                    ScalarParam sp;
                    if (isIntegerString(numStr)) {
                        sp.value = std::stoi(numStr);
                        sp.isInt = true;
                    } else {
                        sp.value = std::stof(numStr);
                        sp.isInt = false;
                    }
                    params.scalarParams.push_back(sp);
                    consume();
                }
                if (tokens[current].type == TOKEN_COMMA) consume();
            }
            expect(TOKEN_RBRACKET);
        } else if (key == "arrays") {
            expect(TOKEN_LBRACE);
            while (current < tokens.size() && tokens[current].type != TOKEN_RBRACE) {
                if (tokens[current].type == TOKEN_STRING) {
                    std::string arrName = tokens[current].value;
                    consume(); // name
                    expect(TOKEN_COLON);
                    
                    // Check if array has type info: {"type":"float","data":[...]}
                    std::string arrType = "float"; // default
                    bool hasTypeInfo = (tokens[current].type == TOKEN_LBRACE);
                    
                    if (hasTypeInfo) {
                        expect(TOKEN_LBRACE);
                        while (current < tokens.size() && tokens[current].type != TOKEN_RBRACE) {
                            if (tokens[current].type == TOKEN_STRING) {
                                std::string arrKey = tokens[current].value;
                                consume();
                                expect(TOKEN_COLON);
                                if (arrKey == "type" && tokens[current].type == TOKEN_STRING) {
                                    arrType = tokens[current].value;
                                    consume();
                                } else if (arrKey == "data") {
                                    // Will parse the array below
                                    break;
                                }
                            }
                            if (tokens[current].type == TOKEN_COMMA) consume();
                        }
                    }
                    
                    expect(TOKEN_LBRACKET);
                    
                    // Collect all number strings
                    std::vector<std::string> numStrings;
                    // For explicit type: use the provided type; for inferred: start with true and set false if any float found
                    bool allInts = hasTypeInfo ? (arrType == "int") : true;
                    while (current < tokens.size() && tokens[current].type != TOKEN_RBRACKET) {
                        if (tokens[current].type == TOKEN_NUMBER) {
                            numStrings.push_back(tokens[current].value);
                            // Only infer from format if no explicit type given
                            if (!hasTypeInfo && !isIntegerString(tokens[current].value)) {
                                allInts = false;
                            }
                            consume();
                        }
                        if (tokens[current].type == TOKEN_COMMA) consume();
                    }
                    expect(TOKEN_RBRACKET);
                    
                    // Close the type info object if present
                    if (hasTypeInfo) {
                        if (tokens[current].type == TOKEN_COMMA) consume();
                        expect(TOKEN_RBRACE);
                    }
                    
                    ArrayParam ap;
                    ap.name = arrName;
                    ap.isInt = allInts;
                    if (allInts) {
                        std::vector<int> intData;
                        for (const auto& ns : numStrings) {
                            intData.push_back(std::stoi(ns));
                        }
                        ap.data = intData;
                    } else {
                        std::vector<float> floatData;
                        for (const auto& ns : numStrings) {
                            floatData.push_back(std::stof(ns));
                        }
                        ap.data = floatData;
                    }
                    params.arrays.push_back(ap);
                }
                if (tokens[current].type == TOKEN_COMMA) consume();
            }
            expect(TOKEN_RBRACE);
        } else {
            consume();
        }

        if (tokens[current].type == TOKEN_COMMA) consume();
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

    // Debug: print scalar types
    for (size_t i = 0; i < params.scalarParams.size(); ++i) {
        const auto& sp = params.scalarParams[i];
        if (sp.isInt) {
            std::cerr << "[Launcher] Scalar[" << i << "]: int = " << std::get<int>(sp.value) << std::endl;
        } else {
            std::cerr << "[Launcher] Scalar[" << i << "]: float = " << std::get<float>(sp.value) << std::endl;
        }
    }

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
    std::vector<size_t> arraySizes;  // Track sizes for copy-back
    std::vector<bool> arrayIsInt;    // Track types for copy-back
    
    for (const auto& arr : params.arrays) {
        CUdeviceptr devPtr;
        size_t numElements;
        size_t sizeBytes;
        
        if (arr.isInt) {
            const auto& intData = std::get<std::vector<int>>(arr.data);
            numElements = intData.size();
            sizeBytes = numElements * sizeof(int);
            CU_CHECK(cuMemAlloc(&devPtr, sizeBytes));
            CU_CHECK(cuMemcpyHtoD(devPtr, intData.data(), sizeBytes));
        } else {
            const auto& floatData = std::get<std::vector<float>>(arr.data);
            numElements = floatData.size();
            sizeBytes = numElements * sizeof(float);
            CU_CHECK(cuMemAlloc(&devPtr, sizeBytes));
            CU_CHECK(cuMemcpyHtoD(devPtr, floatData.data(), sizeBytes));
        }
        
        deviceArrays.push_back(devPtr);
        arraySizes.push_back(numElements);
        arrayIsInt.push_back(arr.isInt);
    }

    // Prepare kernel arguments
    // We need stable storage for scalar values since we pass pointers
    std::vector<int> intScalars;
    std::vector<float> floatScalars;
    std::vector<void*> kernelArgs;

    // First, allocate storage for all scalars
    for (const auto& sp : params.scalarParams) {
        if (sp.isInt) {
            intScalars.push_back(std::get<int>(sp.value));
        } else {
            floatScalars.push_back(std::get<float>(sp.value));
        }
    }

    // Now build kernel args with stable pointers
    size_t intIdx = 0, floatIdx = 0;
    for (const auto& sp : params.scalarParams) {
        if (sp.isInt) {
            kernelArgs.push_back(&intScalars[intIdx++]);
        } else {
            kernelArgs.push_back(&floatScalars[floatIdx++]);
        }
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

        if (!first) std::cout << ",";
        first = false;

        std::cout << "\"" << arr.name << "\":[";
        
        if (arrayIsInt[i]) {
            std::vector<int> result(arraySizes[i]);
            size_t sizeBytes = result.size() * sizeof(int);
            CU_CHECK(cuMemcpyDtoH(result.data(), deviceArrays[i], sizeBytes));
            
            for (size_t j = 0; j < result.size(); ++j) {
                if (j > 0) std::cout << ",";
                std::cout << result[j];
            }
        } else {
            std::vector<float> result(arraySizes[i]);
            size_t sizeBytes = result.size() * sizeof(float);
            CU_CHECK(cuMemcpyDtoH(result.data(), deviceArrays[i], sizeBytes));
            
            for (size_t j = 0; j < result.size(); ++j) {
                if (j > 0) std::cout << ",";
                std::cout << result[j];
            }
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
