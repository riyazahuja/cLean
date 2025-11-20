#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define ARRAY_SIZE 1024

__global__ void stencil(int N, float alpha, float beta, float* input, float* output) {
  __shared__ float buffer[256];
  float i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float val = input[i];
  buffer[i] = val;
  __syncthreads();
  float left = (i - 1);
  float right = (i + 1);
  float centerVal = buffer[i];
  float leftVal = buffer[left];
  float rightVal = buffer[right];
  float sum1 = (centerVal + leftVal);
  float sum = (sum1 + rightVal);
  output[i] = sum;
}

int main() {
  printf("CUDA Kernel: stencil\n");
  
  // Array dimensions
  const int N = ARRAY_SIZE;
  const size_t size = N * sizeof(float);
  
  // Allocate host memory for input
  float *h_input = (float*)malloc(size);
  for (int i = 0; i < N; i++) h_input[i] = (float)i;
  
  // Allocate host memory for output
  float *h_output = (float*)malloc(size);
  for (int i = 0; i < N; i++) h_output[i] = (float)i;
  
  // Allocate device memory for input
  float *d_input;
  CUDA_CHECK(cudaMalloc(&d_input, size));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
  
  // Allocate device memory for output
  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_output, size));
  CUDA_CHECK(cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice));
  
  // Launch kernel
  printf("Launching kernel with %d blocks, %d threads per block\n", N/256, 256);
  dim3 gridDim(4, 1, 1);
  dim3 blockDim(256, 1, 1);
  stencil<<<gridDim, blockDim, 1024>>>(N, 2.5f, 1.0f, d_input, d_output);
  cudaDeviceSynchronize();
  printf("Kernel execution completed\n");
  
  // Copy input back to host
  CUDA_CHECK(cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost));
  // Copy output back to host
  CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
  
  // Print first 10 results
  printf("\nFirst 10 results:\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("[%d] = %.2f\n", i, h_output[i]);
  }
  
  cudaFree(d_input);
  free(h_input);
  cudaFree(d_output);
  free(h_output);
  
  printf("\nTest completed successfully!\n");
  return 0;
}
