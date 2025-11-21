/-
  End-to-End GPU Kernel Execution Tests

  This file tests the complete pipeline:
  1. Lean kernel → DeviceIR conversion
  2. CUDA code generation
  3. GPU execution
  4. Result verification against CPU simulation
-/

import CLean.GPU
import CLean.GPU.FFI
import CLean.GPU.Runtime
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances

open Lean GpuDSL DeviceIR CLean.DeviceMacro CLean.DeviceCodeGen DeviceTranslation
open CLean.GPU.FFI CLean.GPU.Runtime

set_option maxHeartbeats 2000000

/-! ## Test 1: Simple SAXPY Kernel -/

namespace SaxpyTest

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

-- CPU version
def saxpyCPU (n : Nat) (α : Float) (x y : Array Float) : IO (Array Float) := do
  let initState := mkKernelState [
    globalFloatArray `X x,
    globalFloatArray `Y y,
    globalFloatArray `R (Array.replicate n 0.0)
  ]

  let finalState ← runKernelCPU
    ⟨(n + 255) / 256, 1, 1⟩  -- grid
    ⟨256, 1, 1⟩              -- block
    ⟨n, α, `X, `Y, `R⟩
    initState
    saxpyKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `R
    | throw <| IO.userError "Result missing"
  return out

-- GPU version
def saxpyGPU (n : Nat) (α : Float) (x y : Array Float) : IO (Array Float) := do
  IO.println "=== SAXPY GPU Execution ==="

  -- Check CUDA availability
  let available ← cudaIsAvailable
  if !available then
    throw <| IO.userError "CUDA not available"

  -- Print generated CUDA code
  let cudaSource := kernelToCuda saxpyKernelIR
  IO.println "\nGenerated CUDA code:"
  IO.println "-------------------"
  IO.println cudaSource
  IO.println "-------------------\n"

  -- Prepare data
  let scalarParams := #[Float.ofNat n, α]
  let globalArrays := [
    (`X, x),
    (`Y, y),
    (`R, Array.replicate n 0.0)
  ]

  -- Execute on GPU
  IO.println s!"Executing kernel with N={n}, alpha={α}"
  let results ← runKernelGPU
    saxpyKernelIR
    ⟨(n + 255) / 256, 1, 1⟩  -- grid
    ⟨256, 1, 1⟩              -- block
    scalarParams
    globalArrays

  -- Extract result
  match results.find? fun (name, _) => name == `R with
  | some (_, result) => return result
  | none => throw <| IO.userError "Result array R not found"

-- Test function: compare CPU vs GPU
def testSaxpy (n : Nat) (α : Float) (x y : Array Float) : IO Bool := do
  IO.println s!"\n=== Testing SAXPY: N={n}, alpha={α} ==="

  -- Run on CPU
  IO.println "Running on CPU..."
  let cpuResult ← saxpyCPU n α x y
  IO.println s!"CPU result: {cpuResult.toList.take 10}..."

  -- Run on GPU
  IO.println "\nRunning on GPU..."
  let gpuResult ← saxpyGPU n α x y
  IO.println s!"GPU result: {gpuResult.toList.take 10}..."

  -- Compare results
  IO.println "\nComparing results..."
  let mut allMatch := true
  let mut maxDiff := 0.0

  for i in [:n] do
    let cpuVal := cpuResult[i]!
    let gpuVal := gpuResult[i]!
    let diff := (cpuVal - gpuVal).abs

    if diff > maxDiff then
      maxDiff := diff

    if diff > 1e-5 then
      IO.println s!"Mismatch at index {i}: CPU={cpuVal}, GPU={gpuVal}, diff={diff}"
      allMatch := false

  if allMatch then
    IO.println s!"✓ Results match! (max diff: {maxDiff})"
  else
    IO.println s!"✗ Results differ! (max diff: {maxDiff})"

  return allMatch

end SaxpyTest

/-! ## Test 2: Vector Addition -/

namespace VecAddTest

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

def testVecAdd (n : Nat) (a b : Array Float) : IO Bool := do
  IO.println s!"\n=== Testing Vector Addition: N={n} ==="

  -- CPU version
  IO.println "Running on CPU..."
  let initState := mkKernelState [
    globalFloatArray `A a,
    globalFloatArray `B b,
    globalFloatArray `C (Array.replicate n 0.0)
  ]

  let cpuFinalState ← runKernelCPU
    ⟨(n + 255) / 256, 1, 1⟩
    ⟨256, 1, 1⟩
    ⟨n, `A, `B, `C⟩
    initState
    vecAddKernel

  let some (KernelValue.arrayFloat cpuResult) := cpuFinalState.globals.get? `C
    | throw <| IO.userError "CPU result missing"

  -- GPU version
  IO.println "Running on GPU..."
  let cudaSource := kernelToCuda vecAddKernelIR
  IO.println s!"\nCUDA code:\n{cudaSource}"

  let scalarParams := #[Float.ofNat n]
  let globalArrays := [(`A, a), (`B, b), (`C, Array.replicate n 0.0)]

  let gpuResults ← runKernelGPU
    vecAddKernelIR
    ⟨(n + 255) / 256, 1, 1⟩
    ⟨256, 1, 1⟩
    scalarParams
    globalArrays

  let some (_, gpuResult) := gpuResults.find? fun (name, _) => name == `C
    | throw <| IO.userError "GPU result missing"

  -- Compare
  let mut allMatch := true
  for i in [:n] do
    if (cpuResult[i]! - gpuResult[i]!).abs > 1e-5 then
      IO.println s!"Mismatch at {i}: CPU={cpuResult[i]!}, GPU={gpuResult[i]!}"
      allMatch := false

  if allMatch then
    IO.println "✓ Vector addition results match!"
  else
    IO.println "✗ Vector addition results differ!"

  return allMatch

end VecAddTest

/-! ## Test 3: Scalar Multiplication -/

namespace ScaleMulTest

kernelArgs ScaleMulArgs(N: Nat, scale: Float)
  global[input output: Array Float]

device_kernel scaleMulKernel : KernelM ScaleMulArgs Unit := do
  let args ← getArgs
  let N := args.N
  let scale := args.scale
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  if i < N then do
    let val ← input.get i
    output.set i (scale * val)

def testScaleMul (n : Nat) (scale : Float) (input : Array Float) : IO Bool := do
  IO.println s!"\n=== Testing Scalar Multiplication: N={n}, scale={scale} ==="

  -- CPU version
  let initState := mkKernelState [
    globalFloatArray `Input input,
    globalFloatArray `Output (Array.replicate n 0.0)
  ]

  let cpuFinalState ← runKernelCPU
    ⟨(n + 255) / 256, 1, 1⟩
    ⟨256, 1, 1⟩
    ⟨n, scale, `Input, `Output⟩
    initState
    scaleMulKernel

  let some (KernelValue.arrayFloat cpuResult) := cpuFinalState.globals.get? `Output
    | throw <| IO.userError "CPU result missing"

  -- GPU version
  let cudaSource := kernelToCuda scaleMulKernelIR
  IO.println s!"\nCUDA code:\n{cudaSource}"

  let scalarParams := #[Float.ofNat n, scale]
  let globalArrays := [
    (`Input, input),
    (`Output, Array.replicate n 0.0)
  ]

  let gpuResults ← runKernelGPU
    scaleMulKernelIR
    ⟨(n + 255) / 256, 1, 1⟩
    ⟨256, 1, 1⟩
    scalarParams
    globalArrays

  let some (_, gpuResult) := gpuResults.find? fun (name, _) => name == `Output
    | throw <| IO.userError "GPU result missing"

  -- Compare
  let mut allMatch := true
  for i in [:n] do
    if (cpuResult[i]! - gpuResult[i]!).abs > 1e-5 then
      IO.println s!"Mismatch at {i}: CPU={cpuResult[i]!}, GPU={gpuResult[i]!}"
      allMatch := false

  if allMatch then
    IO.println "✓ Scalar multiplication results match!"
  else
    IO.println "✗ Scalar multiplication results differ!"

  return allMatch

end ScaleMulTest

/-! ## Main Test Runner -/

def main : IO Unit := do
  IO.println "==============================================="
  IO.println "  GPU End-to-End Kernel Execution Tests"
  IO.println "==============================================="

  -- Check CUDA availability first
  checkCudaAvailability
  IO.println ""

  -- Test GPU memory
  IO.println "Testing basic GPU memory operations..."
  let testData := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let memResult ← testGpuMemory testData
  if testData == memResult then
    IO.println "✓ Memory test passed!\n"
  else
    IO.println "✗ Memory test failed!\n"
    return

  -- Run kernel tests
  let mut allPassed := true

  -- Test 1: SAXPY with small array
  let test1 ← SaxpyTest.testSaxpy 16 2.5
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  allPassed := allPassed && test1

  -- Test 2: SAXPY with larger array
  let largeX := Array.range 100 |>.map Float.ofNat
  let largeY := Array.replicate 100 1.0
  let test2 ← SaxpyTest.testSaxpy 100 3.0 largeX largeY
  allPassed := allPassed && test2

  -- Test 3: Vector addition
  let vecA := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let vecB := #[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
  let test3 ← VecAddTest.testVecAdd 8 vecA vecB
  allPassed := allPassed && test3

  -- Test 4: Scalar multiplication
  let input := Array.range 50 |>.map Float.ofNat
  let test4 ← ScaleMulTest.testScaleMul 50 2.0 input
  allPassed := allPassed && test4

  -- Summary
  IO.println "\n==============================================="
  if allPassed then
    IO.println "  ✓ ALL TESTS PASSED!"
  else
    IO.println "  ✗ SOME TESTS FAILED"
  IO.println "==============================================="
