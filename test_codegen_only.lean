/-
  CUDA Code Generation Test (No GPU execution)
  Can be run with: lake env lean --run test_codegen_only.lean
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen

-- Test 1: Simple SAXPY
kernelArgs SaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel saxpyKernel : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩

  let i ← globalIdxX
  if i < N then do
    let xi ← x.get i
    let yi ← y.get i
    r.set i (alpha * xi + yi)

-- Test 2: Vector Addition
kernelArgs VecAddArgs(N: Nat)
  global[a b c: Array Float]

device_kernel vecAddKernel : KernelM VecAddArgs Unit := do
  let args ← getArgs
  let N := args.N
  let a : GlobalArray Float := ⟨args.a⟩
  let b : GlobalArray Float := ⟨args.b⟩
  let c : GlobalArray Float := ⟨args.c⟩

  let i ← globalIdxX
  if i < N then do
    let ai ← a.get i
    let bi ← b.get i
    c.set i (ai + bi)

-- Test 3: Matrix transpose (with shared memory)
kernelArgs TransposeArgs(width: Nat, height: Nat)
  global[input output: Array Float]

device_kernel transposeKernel : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let width := args.width
  let height := args.height
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let tx ← globalIdxX
  let ty ← globalIdxY

  if tx < width && ty < height then do
    let inIdx := ty * width + tx
    let outIdx := tx * height + ty
    let val ← input.get inIdx
    output.set outIdx val

def main : IO Unit := do
  IO.println "========================================"
  IO.println " CUDA Code Generation Tests"
  IO.println "========================================"

  IO.println "\n=== Test 1: SAXPY Kernel ==="
  IO.println "Device IR:"
  IO.println s!"{repr saxpyKernelIR}"
  IO.println "\nGenerated CUDA code:"
  IO.println "--------------------"
  IO.println (kernelToCuda saxpyKernelIR)
  IO.println "--------------------"

  IO.println "\n=== Test 2: Vector Addition Kernel ==="
  IO.println "Device IR:"
  IO.println s!"{repr vecAddKernelIR}"
  IO.println "\nGenerated CUDA code:"
  IO.println "--------------------"
  IO.println (kernelToCuda vecAddKernelIR)
  IO.println "--------------------"

  IO.println "\n=== Test 3: Matrix Transpose Kernel ==="
  IO.println "Device IR:"
  IO.println s!"{repr transposeKernelIR}"
  IO.println "\nGenerated CUDA code:"
  IO.println "--------------------"
  IO.println (kernelToCuda transposeKernelIR)
  IO.println "--------------------"

  -- Generate complete CUDA program with launch code
  let saxpyLaunchConfig : LaunchConfig := {
    gridDim := (4, 1, 1),
    blockDim := (256, 1, 1),
    sharedMemBytes := 0
  }

  let completeCuda := genCompleteCudaProgram saxpyKernelIR saxpyLaunchConfig

  IO.println "\n=== Complete CUDA Program (with host code) ==="
  IO.println completeCuda
  IO.println "==============================================="

  IO.println "\n✓ Code generation tests completed successfully!"
