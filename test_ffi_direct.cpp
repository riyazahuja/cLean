/*
 * Direct FFI Test - Bypassing Lean to test FFI library directly
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple test: Check if we can get CUDA device count
int main() {
    printf("=== Direct FFI Library Test ===\n");

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA devices found: %d\n", deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    // Test: Allocate, copy, retrieve
    const int N = 8;
    const size_t size = N * sizeof(float);

    float h_data[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_result[N];

    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    printf("\nMemory test:\n");
    printf("Input:  [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           h_data[0], h_data[1], h_data[2], h_data[3],
           h_data[4], h_data[5], h_data[6], h_data[7]);
    printf("Output: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n",
           h_result[0], h_result[1], h_result[2], h_result[3],
           h_result[4], h_result[5], h_result[6], h_result[7]);

    bool match = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != h_result[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        printf("✓ Memory test PASSED\n");
    } else {
        printf("✗ Memory test FAILED\n");
        return 1;
    }

    printf("\n✓ Direct FFI test completed successfully!\n");
    return 0;
}
