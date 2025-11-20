/-
  Test CUDA Code Generation from DeviceIR

  Generates complete CUDA programs from our device_kernel definitions.
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen

open GpuDSL DeviceIR CLean.DeviceMacro CLean.DeviceCodeGen

set_option maxHeartbeats 2000000

/-! ## Define Test Kernels -/

namespace CudaGenTest

/-! ### Test 1: Simple SAXPY Kernel -/

kernelArgs SaxpyArgs(N: Nat, alpha: Float)
  global[x y result: Array Float]

device_kernel saxpy : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let result : GlobalArray Float := ⟨args.result⟩

  let i ← globalIdxX
  let xi ← x.get i
  let yi ← y.get i
  let scaled := alpha * xi
  let sum := scaled + yi
  result.set i sum

/-! ### Test 2: Matrix Transpose with Shared Memory -/

kernelArgs TransposeArgs(N: Nat)
  global[input output: Array Float]
  shared[tile: Array Float]

device_kernel transpose : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.tile⟩

  let i ← globalIdxX

  -- Phase 1: Load into shared memory
  let val ← input.get i
  tile.set i val
  barrier

  -- Phase 2: Transpose and write out
  let transIdx := N * i
  let tileVal ← tile.get transIdx
  output.set i tileVal

/-! ### Test 3: 3-Point Stencil with Shared Memory -/

kernelArgs StencilArgs(N: Nat)
  global[input output: Array Float]
  shared[buffer: Array Float]

device_kernel stencil : KernelM StencilArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let buffer : SharedArray Float := ⟨args.buffer⟩

  let i ← globalIdxX

  -- Load to shared memory
  let val ← input.get i
  buffer.set i val
  barrier

  -- Compute 3-point stencil
  let left := i - 1
  let right := i + 1

  let centerVal ← buffer.get i
  let leftVal ← buffer.get left
  let rightVal ← buffer.get right

  let sum1 := centerVal + leftVal
  let sum := sum1 + rightVal

  output.set i sum

end CudaGenTest

/-! ## Generate CUDA Code -/

-- Configure launch parameters
def saxpyConfig : LaunchConfig := {
  gridDim := (4, 1, 1),       -- 4 blocks
  blockDim := (256, 1, 1),    -- 256 threads per block
  sharedMemBytes := 0         -- No shared memory
}

def transposeConfig : LaunchConfig := {
  gridDim := (4, 1, 1),
  blockDim := (256, 1, 1),
  sharedMemBytes := 256 * 4   -- 256 floats in shared memory
}

def stencilConfig : LaunchConfig := {
  gridDim := (4, 1, 1),
  blockDim := (256, 1, 1),
  sharedMemBytes := 256 * 4
}

-- Generate CUDA programs
def saxpyCuda := genCompleteCudaProgram CudaGenTest.saxpyIR saxpyConfig
def transposeCuda := genCompleteCudaProgram CudaGenTest.transposeIR transposeConfig 256
def stencilCuda := genCompleteCudaProgram CudaGenTest.stencilIR stencilConfig 256

/-! ## Output Generated CUDA Code -/

#eval IO.println "\n=========================================="
#eval IO.println "CUDA Code for SAXPY Kernel"
#eval IO.println "==========================================\n"
#eval IO.println saxpyCuda

#eval IO.println "\n=========================================="
#eval IO.println "CUDA Code for Transpose Kernel"
#eval IO.println "=========================================="
#eval IO.println transposeCuda

#eval IO.println "\n=========================================="
#eval IO.println "CUDA Code for Stencil Kernel"
#eval IO.println "=========================================="
#eval IO.println stencilCuda

/-! ## Write CUDA Files to Disk -/

def writeCudaFiles : IO Unit := do
  IO.FS.writeFile "saxpy.cu" saxpyCuda
  IO.FS.writeFile "transpose.cu" transposeCuda
  IO.FS.writeFile "stencil.cu" stencilCuda
  IO.println "\n✓ Written CUDA files: saxpy.cu, transpose.cu, stencil.cu"
  IO.println "✓ Compile with: nvcc -o <output> <file>.cu"
  IO.println "✓ Run with: ./<output>"

#eval writeCudaFiles
