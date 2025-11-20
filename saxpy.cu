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

__global__ void saxpy(int N, float alpha, float beta, float* x, float* y, float* result) {
  float i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float xi = x[i];
  float yi = y[i];
  float scaled = (alpha * xi);
  float sum = (scaled + yi);
  result[i] = sum;
}

int main() {
  printf("CUDA Kernel: saxpy\n");
  
  // Array dimensions
  const int N = ARRAY_SIZE;
  const size_t size = N * sizeof(float);
  
  // Allocate host memory for x
  float *h_x = (float*)malloc(size);
  for (int i = 0; i < N; i++) h_x[i] = (float)i;
  
  // Allocate host memory for y
  float *h_y = (float*)malloc(size);
  for (int i = 0; i < N; i++) h_y[i] = (float)i;
  
  // Allocate host memory for result
  float *h_result = (float*)malloc(size);
  for (int i = 0; i < N; i++) h_result[i] = (float)i;
  
  // Allocate device memory for x
  float *d_x;
  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
  
  // Allocate device memory for y
  float *d_y;
  CUDA_CHECK(cudaMalloc(&d_y, size));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));
  
  // Allocate device memory for result
  float *d_result;
  CUDA_CHECK(cudaMalloc(&d_result, size));
  CUDA_CHECK(cudaMemcpy(d_result, h_result, size, cudaMemcpyHostToDevice));
  
  // Launch kernel
  printf("Launching kernel with %d blocks, %d threads per block\n", N/256, 256);
  dim3 gridDim(4, 1, 1);
  dim3 blockDim(256, 1, 1);
  saxpy<<<gridDim, blockDim>>>(N, 2.5f, 1.0f, d_x, d_y, d_result);
  cudaDeviceSynchronize();
  printf("Kernel execution completed\n");
  
  // Copy x back to host
  CUDA_CHECK(cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost));
  // Copy y back to host
  CUDA_CHECK(cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost));
  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));
  
  // Print first 10 results
  printf("\nFirst 10 results:\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("[%d] = %.2f\n", i, h_result[i]);
  }
  
  cudaFree(d_x);
  free(h_x);
  cudaFree(d_y);
  free(h_y);
  cudaFree(d_result);
  free(h_result);
  
  printf("\nTest completed successfully!\n");
  return 0;
}
