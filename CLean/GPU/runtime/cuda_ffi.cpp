/*
 * CUDA FFI Bridge - Implementation
 *
 * C++ implementation of CUDA runtime bindings for Lean.
 */

#include "cuda_ffi.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

/*
 * Internal structures for opaque types
 */

struct CudaArrayImpl {
    void* device_ptr;
    size_t size_bytes;
};

struct CudaKernelImpl {
    CUmodule module;
    CUfunction function;
    std::string name;
};

/*
 * Utility functions for Lean FFI
 */

// Create lean IO error from string
static lean_obj_res lean_io_error(const char* msg) {
    lean_object* err = lean_mk_io_user_error(lean_mk_string(msg));
    return lean_io_result_mk_error(err);
}

// External class for CudaArray
static void cuda_array_finalizer(void* ptr) {
    CudaArrayImpl* arr = static_cast<CudaArrayImpl*>(ptr);
    if (arr->device_ptr) {
        cudaFree(arr->device_ptr);
    }
    delete arr;
}

static void cuda_array_foreach(void*, b_lean_obj_arg) {}

static lean_external_class* get_cuda_array_class() {
    static lean_external_class* cls = nullptr;
    if (cls == nullptr) {
        cls = lean_register_external_class(cuda_array_finalizer, cuda_array_foreach);
    }
    return cls;
}

// Wrap CudaArrayImpl as lean external object
static lean_obj_res wrap_cuda_array(CudaArrayImpl* arr) {
    return lean_alloc_external(get_cuda_array_class(), arr);
}

// Unwrap lean external object to CudaArrayImpl
static CudaArrayImpl* unwrap_cuda_array(b_lean_obj_arg obj) {
    return static_cast<CudaArrayImpl*>(lean_get_external_data(obj));
}

// External class for CudaKernel
static void cuda_kernel_finalizer(void* ptr) {
    CudaKernelImpl* k = static_cast<CudaKernelImpl*>(ptr);
    if (k->module) {
        cuModuleUnload(k->module);
    }
    delete k;
}

static void cuda_kernel_foreach(void*, b_lean_obj_arg) {}

static lean_external_class* get_cuda_kernel_class() {
    static lean_external_class* cls = nullptr;
    if (cls == nullptr) {
        cls = lean_register_external_class(cuda_kernel_finalizer, cuda_kernel_foreach);
    }
    return cls;
}

// Wrap CudaKernelImpl as lean external object
static lean_obj_res wrap_cuda_kernel(CudaKernelImpl* kernel) {
    return lean_alloc_external(get_cuda_kernel_class(), kernel);
}

// Unwrap lean external object to CudaKernelImpl
static CudaKernelImpl* unwrap_cuda_kernel(b_lean_obj_arg obj) {
    return static_cast<CudaKernelImpl*>(lean_get_external_data(obj));
}

/*
 * Memory Management Implementation
 */

extern "C" lean_obj_res lean_cuda_malloc(size_t size, lean_obj_arg world) {
    CudaArrayImpl* arr = new CudaArrayImpl;
    arr->size_bytes = size;

    cudaError_t err = cudaMalloc(&arr->device_ptr, size);
    if (err != cudaSuccess) {
        delete arr;
        std::string msg = "cudaMalloc failed: ";
        msg += cudaGetErrorString(err);
        return lean_io_error(msg.c_str());
    }

    return lean_io_result_mk_ok(wrap_cuda_array(arr));
}

extern "C" lean_obj_res lean_cuda_free(b_lean_obj_arg arr_obj, lean_obj_arg world) {
    // The finalizer will handle actual freeing
    // Just return success
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_cuda_memcpy_h2d_float(
    b_lean_obj_arg arr_obj,
    b_lean_obj_arg data_obj,
    lean_obj_arg world
) {
    CudaArrayImpl* arr = unwrap_cuda_array(arr_obj);

    // Extract float array from Lean
    size_t len = lean_array_size(data_obj);
    std::vector<float> host_data(len);

    for (size_t i = 0; i < len; i++) {
        lean_object* elem = lean_array_get_core(data_obj, i);
        host_data[i] = lean_unbox_float(elem);
    }

    // Copy to device
    size_t copy_size = len * sizeof(float);
    cudaError_t err = cudaMemcpy(arr->device_ptr, host_data.data(), copy_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::string msg = "cudaMemcpy H2D failed: ";
        msg += cudaGetErrorString(err);
        return lean_io_error(msg.c_str());
    }

    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_cuda_memcpy_d2h_float(
    b_lean_obj_arg arr_obj,
    size_t size,
    lean_obj_arg world
) {
    CudaArrayImpl* arr = unwrap_cuda_array(arr_obj);

    // Allocate host buffer
    std::vector<float> host_data(size);

    // Copy from device
    size_t copy_size = size * sizeof(float);
    cudaError_t err = cudaMemcpy(host_data.data(), arr->device_ptr, copy_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::string msg = "cudaMemcpy D2H failed: ";
        msg += cudaGetErrorString(err);
        return lean_io_error(msg.c_str());
    }

    // Create Lean array
    lean_object* result = lean_alloc_array(size, size);
    for (size_t i = 0; i < size; i++) {
        lean_array_set_core(result, i, lean_box_float(host_data[i]));
    }

    return lean_io_result_mk_ok(result);
}

/*
 * Kernel Compilation Implementation
 */

extern "C" lean_obj_res lean_cuda_compile_kernel(
    b_lean_obj_arg source_obj,
    b_lean_obj_arg kernel_name_obj,
    b_lean_obj_arg cache_path_obj,
    lean_obj_arg world
) {
    const char* source = lean_string_cstr(source_obj);
    const char* kernel_name = lean_string_cstr(kernel_name_obj);
    const char* cache_path = lean_string_cstr(cache_path_obj);

    // Initialize CUDA driver API
    static bool cuda_initialized = false;
    if (!cuda_initialized) {
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            return lean_io_error("Failed to initialize CUDA driver API");
        }
        cuda_initialized = true;
    }

    // Compile with NVRTC
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source, "kernel.cu", 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        return lean_io_error("nvrtcCreateProgram failed");
    }

    // Compile options
    const char* opts[] = {
        "--gpu-architecture=compute_50",  // Adjust based on target
        "-std=c++14"
    };

    res = nvrtcCompileProgram(prog, 2, opts);
    if (res != NVRTC_SUCCESS) {
        // Get compilation log
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(prog, log.data());

        nvrtcDestroyProgram(&prog);

        std::string err_msg = "Kernel compilation failed:\n";
        err_msg += log.data();
        return lean_io_error(err_msg.c_str());
    }

    // Get PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    // Load PTX into CUDA module
    CUmodule module;
    CUresult cu_res = cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0);
    if (cu_res != CUDA_SUCCESS) {
        return lean_io_error("Failed to load PTX module");
    }

    // Get kernel function
    CUfunction function;
    cu_res = cuModuleGetFunction(&function, module, kernel_name);
    if (cu_res != CUDA_SUCCESS) {
        cuModuleUnload(module);
        return lean_io_error("Failed to get kernel function");
    }

    // Create kernel object
    CudaKernelImpl* kernel = new CudaKernelImpl;
    kernel->module = module;
    kernel->function = function;
    kernel->name = kernel_name;

    return lean_io_result_mk_ok(wrap_cuda_kernel(kernel));
}

extern "C" lean_obj_res lean_cuda_free_kernel(b_lean_obj_arg kernel_obj, lean_obj_arg world) {
    // Finalizer handles cleanup
    return lean_io_result_mk_ok(lean_box(0));
}

/*
 * Kernel Launch Implementation
 */

extern "C" lean_obj_res lean_cuda_launch_kernel(
    b_lean_obj_arg kernel_obj,
    size_t grid_x, size_t grid_y, size_t grid_z,
    size_t block_x, size_t block_y, size_t block_z,
    b_lean_obj_arg scalar_params_obj,
    b_lean_obj_arg array_params_obj,
    lean_obj_arg world
) {
    CudaKernelImpl* kernel = unwrap_cuda_kernel(kernel_obj);

    // Prepare kernel parameters
    std::vector<void*> args;
    std::vector<float> scalar_values;
    std::vector<void*> array_ptrs;

    // Extract scalar parameters
    size_t scalar_count = lean_array_size(scalar_params_obj);
    scalar_values.resize(scalar_count);
    for (size_t i = 0; i < scalar_count; i++) {
        lean_object* elem = lean_array_get_core(scalar_params_obj, i);
        scalar_values[i] = lean_unbox_float(elem);
        args.push_back(&scalar_values[i]);
    }

    // Extract array parameters
    size_t array_count = lean_array_size(array_params_obj);
    array_ptrs.resize(array_count);
    for (size_t i = 0; i < array_count; i++) {
        lean_object* elem = lean_array_get_core(array_params_obj, i);
        CudaArrayImpl* arr = unwrap_cuda_array(elem);
        array_ptrs[i] = arr->device_ptr;
        args.push_back(&array_ptrs[i]);
    }

    // Launch kernel
    CUresult res = cuLaunchKernel(
        kernel->function,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,  // shared memory
        0,  // stream
        args.data(),
        nullptr
    );

    if (res != CUDA_SUCCESS) {
        return lean_io_error("Kernel launch failed");
    }

    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_cuda_device_synchronize(lean_obj_arg world) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::string msg = "cudaDeviceSynchronize failed: ";
        msg += cudaGetErrorString(err);
        return lean_io_error(msg.c_str());
    }
    return lean_io_result_mk_ok(lean_box(0));
}

/*
 * Utility Implementation
 */

extern "C" lean_obj_res lean_cuda_get_device_count(lean_obj_arg world) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return lean_io_result_mk_ok(lean_box_usize(0));
    }
    return lean_io_result_mk_ok(lean_box_usize(count));
}
