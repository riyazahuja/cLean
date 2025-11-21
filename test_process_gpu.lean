/-
  End-to-End Test: Lean → Process → GPU → Lean

  Tests the complete round-trip using process-based communication.
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
  IO.println "  Process-Based GPU Execution Test"
  IO.println "========================================"

  -- Test data
  let n := 8
  let alpha := 2.0
  let x := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let y := #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

  IO.println s!"\nInput:"
  IO.println s!"  x = {x}"
  IO.println s!"  y = {y}"
  IO.println s!"  alpha = {alpha}"
  IO.println s!"  Expected r[i] = 2.0 * x[i] + y[i]"

  -- Run on CPU first for comparison
  IO.println "\n=== CPU Execution ==="
  let initState := mkKernelState [
    globalFloatArray `X x,
    globalFloatArray `Y y,
    globalFloatArray `R (Array.replicate n 0.0)
  ]

  let cpuState ← runKernelCPU
    ⟨1, 1, 1⟩    -- 1 block
    ⟨256, 1, 1⟩  -- 256 threads
    ⟨n, alpha, `X, `Y, `R⟩
    initState
    testKernel

  let some (KernelValue.arrayFloat cpuResult) := cpuState.globals.get? `R
    | throw <| IO.userError "CPU result missing"

  IO.println s!"CPU Result: {cpuResult}"

  -- Generate CUDA code
  IO.println "\n=== CUDA Code Generation ==="
  let cudaCode := kernelToCuda testKernelIR
  IO.println cudaCode

  -- Run on GPU using process launcher
  IO.println "=== GPU Execution (Process-Based) ==="

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

  let some (_, gpuResult) := gpuResults.find? fun (name, _) => name == `R
    | throw <| IO.userError "GPU result missing"

  IO.println s!"GPU Result: {gpuResult}"

  -- Compare results
  IO.println "\n=== Comparison ==="
  let mut allMatch := true
  for i in [:n] do
    let cpu := cpuResult[i]!
    let gpu := gpuResult[i]!
    let diff := (cpu - gpu).abs
    IO.println s!"[{i}] CPU: {cpu}, GPU: {gpu}, diff: {diff}"
    if diff > 1e-5 then
      allMatch := false

  IO.println "\n========================================"
  if allMatch then
    IO.println "  ✓ SUCCESS: CPU and GPU match!"
  else
    IO.println "  ✗ FAILURE: CPU and GPU differ!"
  IO.println "========================================"
