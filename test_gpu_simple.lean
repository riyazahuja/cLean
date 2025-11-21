/-
  Simple GPU Test - Can be run with: lake env lean --run test_gpu_simple.lean
-/

import CLean.GPU
import CLean.GPU.FFI
import CLean.GPU.Runtime
import CLean.DeviceMacro
import CLean.DeviceCodeGen

open GpuDSL CLean.GPU.FFI CLean.GPU.Runtime CLean.DeviceMacro CLean.DeviceCodeGen

-- Simple saxpy kernel
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
  IO.println "=== Simple GPU Test ==="

  -- Check CUDA
  checkCudaAvailability

  -- Test data
  let n := 8
  let alpha := 2.0
  let x := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let y := #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

  IO.println s!"\nInput: x={x}, y={y}, alpha={alpha}"

  -- Expected: r[i] = 2.0 * x[i] + y[i]
  -- r = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]

  -- Generate and print CUDA code
  let cudaCode := kernelToCuda testKernelIR
  IO.println "\n=== Generated CUDA Code ==="
  IO.println cudaCode
  IO.println "=========================\n"

  -- Run on CPU first
  IO.println "Running on CPU..."
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

  IO.println s!"CPU result: {cpuResult}"

  -- Run on GPU
  IO.println "\nRunning on GPU..."
  let scalarParams := #[Float.ofNat n, alpha]
  let globalArrays := [
    (`X, x),
    (`Y, y),
    (`R, Array.replicate n 0.0)
  ]

  let gpuResults ← runKernelGPU
    testKernelIR
    ⟨1, 1, 1⟩    -- 1 block
    ⟨256, 1, 1⟩  -- 256 threads
    scalarParams
    globalArrays

  let some (_, gpuResult) := gpuResults.find? fun (name, _) => name == `R
    | throw <| IO.userError "GPU result missing"

  IO.println s!"GPU result: {gpuResult}"

  -- Compare
  IO.println "\n=== Comparison ==="
  let mut allMatch := true
  for i in [:n] do
    let cpu := cpuResult[i]!
    let gpu := gpuResult[i]!
    let diff := (cpu - gpu).abs
    IO.println s!"[{i}] CPU: {cpu}, GPU: {gpu}, diff: {diff}"
    if diff > 1e-5 then
      allMatch := false

  if allMatch then
    IO.println "\n✓ SUCCESS: CPU and GPU results match!"
  else
    IO.println "\n✗ FAILURE: CPU and GPU results differ!"
