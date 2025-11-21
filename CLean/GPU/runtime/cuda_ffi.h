/*
 * CUDA FFI Bridge - Header
 *
 * C interface for Lean FFI to interact with CUDA runtime.
 * Uses Lean's C API for memory management and type safety.
 */

#pragma once

#include <lean/lean.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque type handles (these are just pointers wrapped in lean_object)
 */

// Forward declarations of opaque types
typedef struct CudaArrayImpl* CudaArrayHandle;
typedef struct CudaKernelImpl* CudaKernelHandle;

/*
 * Memory Management Functions
 */

// Allocate GPU memory
// lean_cuda_malloc : USize -> IO CudaArray
lean_obj_res lean_cuda_malloc(size_t size, lean_obj_arg world);

// Free GPU memory
// lean_cuda_free : CudaArray -> IO Unit
lean_obj_res lean_cuda_free(b_lean_obj_arg arr, lean_obj_arg world);

// Copy host to device (float array)
// lean_cuda_memcpy_h2d_float : CudaArray -> Array Float -> IO Unit
lean_obj_res lean_cuda_memcpy_h2d_float(
    b_lean_obj_arg arr,
    b_lean_obj_arg data,
    lean_obj_arg world
);

// Copy device to host (float array)
// lean_cuda_memcpy_d2h_float : CudaArray -> USize -> IO (Array Float)
lean_obj_res lean_cuda_memcpy_d2h_float(
    b_lean_obj_arg arr,
    size_t size,
    lean_obj_arg world
);

/*
 * Kernel Compilation Functions
 */

// Compile CUDA kernel from source
// lean_cuda_compile_kernel : String -> String -> String -> IO CudaKernel
lean_obj_res lean_cuda_compile_kernel(
    b_lean_obj_arg source,
    b_lean_obj_arg kernel_name,
    b_lean_obj_arg cache_path,
    lean_obj_arg world
);

// Free compiled kernel
// lean_cuda_free_kernel : CudaKernel -> IO Unit
lean_obj_res lean_cuda_free_kernel(b_lean_obj_arg kernel, lean_obj_arg world);

/*
 * Kernel Launch Functions
 */

// Launch CUDA kernel
// lean_cuda_launch_kernel : CudaKernel -> USize -> ... -> Array Float -> Array CudaArray -> IO Unit
lean_obj_res lean_cuda_launch_kernel(
    b_lean_obj_arg kernel,
    size_t grid_x, size_t grid_y, size_t grid_z,
    size_t block_x, size_t block_y, size_t block_z,
    b_lean_obj_arg scalar_params,
    b_lean_obj_arg array_params,
    lean_obj_arg world
);

// Synchronize device
// lean_cuda_device_synchronize : IO Unit
lean_obj_res lean_cuda_device_synchronize(lean_obj_arg world);

/*
 * Utility Functions
 */

// Get CUDA device count
// lean_cuda_get_device_count : IO USize
lean_obj_res lean_cuda_get_device_count(lean_obj_arg world);

#ifdef __cplusplus
}
#endif
