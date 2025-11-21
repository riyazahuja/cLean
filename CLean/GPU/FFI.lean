/-
  FFI Bridge for CUDA Runtime Operations

  This module provides Lean FFI bindings to C++ functions that interface with CUDA.
  It uses opaque types to represent CUDA objects and provides safe wrappers.
-/

namespace CLean.GPU.FFI

/-! ## Opaque Types -/

/-- Opaque type representing a CUDA device array -/
opaque CudaArrayPointed : NonemptyType
def CudaArray : Type := CudaArrayPointed.type
instance : Nonempty CudaArray := CudaArrayPointed.property

/-- Opaque type representing a compiled CUDA kernel -/
opaque CudaKernelPointed : NonemptyType
def CudaKernel : Type := CudaKernelPointed.type
instance : Nonempty CudaKernel := CudaKernelPointed.property

/-! ## Memory Management FFI -/

/-- Allocate memory on the GPU device
    @param size: Number of bytes to allocate
    @return: Pointer to GPU memory (or error)
-/
@[extern "lean_cuda_malloc"]
opaque cudaMalloc (size : @& USize) : IO CudaArray

/-- Free GPU memory
    @param arr: GPU array to free
-/
@[extern "lean_cuda_free"]
opaque cudaFree (arr : @& CudaArray) : IO Unit

/-- Copy data from host to device
    @param arr: Destination GPU array
    @param data: Source host data (as Float array)
    @return: Unit (or error)
-/
@[extern "lean_cuda_memcpy_h2d_float"]
opaque cudaMemcpyH2D (arr : @& CudaArray) (data : @& Array Float) : IO Unit

/-- Copy data from device to host
    @param arr: Source GPU array
    @param size: Number of float elements to copy
    @return: Array of floats (or error)
-/
@[extern "lean_cuda_memcpy_d2h_float"]
opaque cudaMemcpyD2H (arr : @& CudaArray) (size : @& USize) : IO (Array Float)

/-! ## Kernel Compilation FFI -/

/-- Compile CUDA source code to a loadable kernel
    @param source: CUDA C++ source code as string
    @param kernelName: Name of the kernel function
    @param cachePath: Path to cache compiled kernel
    @return: Compiled kernel handle (or error)
-/
@[extern "lean_cuda_compile_kernel"]
opaque cudaCompileKernel (source : @& String) (kernelName : @& String) (cachePath : @& String) : IO CudaKernel

/-- Free a compiled kernel
    @param kernel: Kernel to free
-/
@[extern "lean_cuda_free_kernel"]
opaque cudaFreeKernel (kernel : @& CudaKernel) : IO Unit

/-! ## Kernel Launch FFI -/

/-- Launch a CUDA kernel
    @param kernel: Compiled kernel to launch
    @param gridDimX, gridDimY, gridDimZ: Grid dimensions
    @param blockDimX, blockDimY, blockDimZ: Block dimensions
    @param scalarParams: Array of scalar parameters (as floats for now)
    @param arrayParams: Array of GPU array parameters
    @return: Unit (or error)
-/
@[extern "lean_cuda_launch_kernel"]
opaque cudaLaunchKernel
  (kernel : @& CudaKernel)
  (gridDimX gridDimY gridDimZ : @& USize)
  (blockDimX blockDimY blockDimZ : @& USize)
  (scalarParams : @& Array Float)
  (arrayParams : @& Array CudaArray)
  : IO Unit

/-- Synchronize device (wait for all kernels to complete) -/
@[extern "lean_cuda_device_synchronize"]
opaque cudaDeviceSynchronize : IO Unit

/-! ## Utility Functions -/

/-- Get CUDA device count -/
@[extern "lean_cuda_get_device_count"]
opaque cudaGetDeviceCount : IO USize

/-- Check if CUDA is available -/
def cudaIsAvailable : IO Bool := do
  try
    let count ← cudaGetDeviceCount
    if count > 0 then
      return true
    else
      return false
  catch _ =>
    return false

end CLean.GPU.FFI
