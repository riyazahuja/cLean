/-
  GPU Runtime - High-level wrapper for GPU kernel execution

  Provides convenient functions to execute DeviceIR kernels on actual GPU hardware.
  Uses the FFI bridge to compile, launch, and manage CUDA kernels.
-/

import CLean.GPU.FFI
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.GPU
import Lean

namespace CLean.GPU.Runtime

open CLean.GPU.FFI
open DeviceIR
open CLean.DeviceCodeGen
open GpuDSL
open Lean (Name)

/-! ## Parameter Management -/

/-- Represents the runtime arguments for a GPU kernel -/
structure GpuKernelArgs where
  /-- Scalar parameters (int, float, etc.) -/
  scalarParams : Array Float := #[]
  /-- Global array data (host-side) -/
  globalArrays : List (Name × Array Float) := []
  /-- Names mapping for global arrays -/
  arrayNames : Array Name := #[]

/-! ## Helper Functions -/

/-- Extract scalar parameter values from KernelState
    This looks for scalar values in the globals that match parameter names -/
def extractScalarParams (kernel : Kernel) (state : KernelState) : Array Float := Id.run do
  -- Extract scalar params from kernel body
  let scalarParams := CLean.DeviceCodeGen.extractScalarParams kernel.body
  let mut result := #[]

  -- For each scalar param, try to find it in state.globals
  for (name, _ty) in scalarParams do
    match state.globals.get? (Name.mkSimple name) with
    | some (.float f) => result := result.push f
    | some (.int i) => result := result.push (Float.ofInt i)
    | some (.nat n) => result := result.push (Float.ofNat n)
    | _ => result := result.push 0.0  -- Default value

  return result

/-- Extract global array data from KernelState -/
def extractGlobalArrays (kernel : Kernel) (state : KernelState) : List (Name × Array Float) := Id.run do
  let mut result := []

  -- Extract arrays mentioned in kernel.globalArrays
  for arrayDecl in kernel.globalArrays do
    let name := Name.mkSimple arrayDecl.name
    match state.globals.get? name with
    | some (.arrayFloat arr) => result := (name, arr) :: result
    | some (.arrayInt arr) =>
        -- Convert Int array to Float array
        let floatArr := arr.map Float.ofInt
        result := (name, floatArr) :: result
    | some (.arrayNat arr) =>
        -- Convert Nat array to Float array
        let floatArr := arr.map Float.ofNat
        result := (name, floatArr) :: result
    | _ => pure ()  -- Skip non-array or missing values

  return result.reverse

/-! ## GPU Kernel Execution -/

/-- Run a DeviceIR kernel on the GPU
    @param kernel: The DeviceIR kernel to execute
    @param grid: Grid dimensions (blocks)
    @param block: Block dimensions (threads per block)
    @param scalarParams: Scalar parameters (in order they appear in kernel)
    @param globalArrays: Global arrays as (name, data) pairs
    @return: Updated global arrays after kernel execution
-/
def runKernelGPU
    (kernel : Kernel)
    (grid block : Dim3)
    (scalarParams : Array Float)
    (globalArrays : List (Name × Array Float))
    : IO (List (Name × Array Float)) := do

  -- Check if CUDA is available
  let available ← cudaIsAvailable
  if !available then
    throw <| IO.userError "CUDA not available on this system"

  -- Generate CUDA source code
  let cudaSource := kernelToCuda kernel
  IO.println s!"Generated CUDA source:\n{cudaSource}"

  -- Compile kernel (use /tmp for cache)
  let cachePath := s!"/tmp/lean_cuda_{kernel.name}.ptx"
  let compiledKernel ← cudaCompileKernel cudaSource kernel.name cachePath

  -- Allocate GPU memory for arrays and copy data
  let mut deviceArrays : Array (Name × CudaArray × USize) := #[]
  for (name, hostData) in globalArrays do
    let sizeBytes : USize := hostData.size * 4  -- sizeof(float) = 4
    let devArray ← cudaMalloc sizeBytes
    cudaMemcpyH2D devArray hostData
    deviceArrays := deviceArrays.push (name, devArray, hostData.size)

  -- Prepare array parameters for kernel launch (in declaration order)
  let mut arrayParams := #[]
  for arrayDecl in kernel.globalArrays do
    let declName := Name.mkSimple arrayDecl.name
    -- Find matching device array
    match deviceArrays.find? fun (n, _, _) => n == declName with
    | some (_, devArr, _) => arrayParams := arrayParams.push devArr
    | none => throw <| IO.userError s!"Array {declName} not found in device arrays"

  -- Launch kernel
  cudaLaunchKernel
    compiledKernel
    grid.x.toUSize grid.y.toUSize grid.z.toUSize
    block.x.toUSize block.y.toUSize block.z.toUSize
    scalarParams
    arrayParams

  -- Synchronize to wait for completion
  cudaDeviceSynchronize

  -- Copy results back from GPU
  let mut results := []
  for (name, devArray, size) in deviceArrays do
    let hostData ← cudaMemcpyD2H devArray size
    results := (name, hostData) :: results

  -- Cleanup device memory
  for (_, devArray, _) in deviceArrays do
    cudaFree devArray

  cudaFreeKernel compiledKernel

  return results.reverse

/-- Run a DeviceIR kernel on the GPU using KernelState interface
    This matches the interface of runKernelCPU for easy interchangeability.

    @param kernel: The DeviceIR kernel to execute
    @param grid: Grid dimensions
    @param block: Block dimensions
    @param initState: Initial kernel state containing input data
    @return: Final kernel state with output data
-/
def runKernelGPU_withState
    (kernel : Kernel)
    (grid block : Dim3)
    (initState : KernelState)
    : IO KernelState := do

  -- Extract scalar parameters and arrays from state
  let scalarParams := extractScalarParams kernel initState
  let globalArrays := extractGlobalArrays kernel initState

  -- Run on GPU
  let resultArrays ← runKernelGPU kernel grid block scalarParams globalArrays

  -- Update state with results
  let mut finalGlobals := initState.globals
  for (name, arr) in resultArrays do
    finalGlobals := finalGlobals.insert name (.arrayFloat arr)

  return { initState with globals := finalGlobals }

/-! ## Convenience Functions -/

/-- Check CUDA availability and print device info -/
def checkCudaAvailability : IO Unit := do
  let available ← cudaIsAvailable
  if available then
    let count ← cudaGetDeviceCount
    IO.println s!"✓ CUDA is available with {count} device(s)"
  else
    IO.println "✗ CUDA is not available"

/-- Simple test: allocate GPU memory, copy data, and retrieve it -/
def testGpuMemory (data : Array Float) : IO (Array Float) := do
  IO.println s!"Testing GPU memory with {data.size} floats..."

  let sizeBytes : USize := data.size * 4
  let devArray ← cudaMalloc sizeBytes
  IO.println "✓ Allocated GPU memory"

  cudaMemcpyH2D devArray data
  IO.println "✓ Copied data to GPU"

  let result ← cudaMemcpyD2H devArray data.size
  IO.println "✓ Copied data back from GPU"

  cudaFree devArray
  IO.println "✓ Freed GPU memory"

  return result

end CLean.GPU.Runtime
