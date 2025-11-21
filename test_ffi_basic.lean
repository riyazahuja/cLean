/-
  Basic FFI Integration Test

  This file tests the CUDA FFI bridge by:
  1. Checking CUDA availability
  2. Testing basic memory allocation and transfer
  3. Compiling and running a simple CUDA kernel
-/

import CLean.GPU.FFI
import CLean.GPU.Runtime

open CLean.GPU.FFI
open CLean.GPU.Runtime

def main : IO Unit := do
  IO.println "=== CUDA FFI Basic Test ==="
  IO.println ""

  -- Test 1: Check CUDA availability
  IO.println "Test 1: Checking CUDA availability..."
  checkCudaAvailability
  IO.println ""

  -- Test 2: Basic memory test
  IO.println "Test 2: Testing GPU memory allocation and transfer..."
  let testData := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let result ← testGpuMemory testData
  IO.println s!"Input:  {testData}"
  IO.println s!"Output: {result}"

  if testData == result then
    IO.println "✓ Memory test passed!"
  else
    IO.println "✗ Memory test failed!"
  IO.println ""

  -- Test 3: Simple kernel compilation
  IO.println "Test 3: Compiling a simple CUDA kernel..."
  let simpleKernel := "
__global__ void add_one(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = data[i] + 1.0f;
  }
}
"

  try
    let kernel ← cudaCompileKernel simpleKernel "add_one" "/tmp/add_one.ptx"
    IO.println "✓ Kernel compiled successfully!"
    cudaFreeKernel kernel
  catch e =>
    IO.println s!"✗ Kernel compilation failed: {e}"

  IO.println ""
  IO.println "=== FFI Test Complete ==="
