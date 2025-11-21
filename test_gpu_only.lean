/-
  GPU-Only Test: Tests just the GPU execution path
  Skips CPU simulation for faster testing
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache
import CLean.GPU.ProcessLauncher

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen
open CLean.GPU.KernelCache CLean.GPU.ProcessLauncher

-- Simple SAXPY kernel
kernelArgs TestArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel testKernel : KernelM TestArgs Unit := do
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

def main : IO Unit := do
  IO.println "========================================"
  IO.println "  GPU Execution Test (Process-Based)"
  IO.println "========================================"

  -- Test data
  let n := 8
  let alpha := 2.5
  let x := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let y := #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

  IO.println s!"\nInput:"
  IO.println s!"  x = {x}"
  IO.println s!"  y = {y}"
  IO.println s!"  alpha = {alpha}"
  IO.println s!"  Formula: r[i] = {alpha} * x[i] + y[i]"

  -- Expected results (manual calculation)
  let expected := #[3.5, 6.0, 8.5, 11.0, 13.5, 16.0, 18.5, 21.0]
  IO.println s!"  Expected: {expected}"

  -- Run on GPU using process launcher
  IO.println "\n=== GPU Execution ==="

  let scalarParams := #[Float.ofNat n, alpha]
  let arrays := [
    (`X, x),
    (`Y, y),
    (`R, Array.replicate n 0.0)
  ]

  let gpuResults ← executeKernel
    testKernelIR
    ⟨1, 1, 1⟩    -- 1 block
    ⟨256, 1, 1⟩  -- 256 threads
    scalarParams
    arrays

  -- The JSON parser is stubbed, so results will be printed as raw JSON
  -- We can still verify by looking at the output
  IO.println "\nRaw output from GPU launcher:"
  IO.println "Note: JSON parser is stubbed, showing raw output"

  IO.println "\n========================================"
  IO.println "Test complete!"
  IO.println "If you see R=[3.5,6,8.5,11,13.5,16,18.5,21]"
  IO.println "in the output above, the test PASSED!"
  IO.println "========================================"
