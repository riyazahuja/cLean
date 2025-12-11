/-
  Benchmark Dataset: Dot Product

  Kernel: sum = Σ A[i] * B[i]
  Pattern: Reduction (block-level reduce, then host aggregation)
  Source: CUDA Samples reduction
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

import CLean.ToGPUVerifyIR
open CLean.ToGPUVerifyIR CLean.Verification CLean.Verification.GPUVerify


namespace CLean.Benchmarks.Dataset.DotProduct

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

-- Each block computes partial sum, writes to partialSums[blockIdx.x]
kernelArgs DotProductArgs(n: Nat, elementsPerThread: Nat)
  global[A: Array Float]
  global[B: Array Float]
  global[partialSums: Array Float]

/-! ## cLean GPU Kernel (Partial reduction) -/

-- Each thread computes partial products over its assigned elements
-- Block-level reduction would require shared memory + barriers
-- This version writes per-thread results (simplified)
device_kernel dotProductKernel : KernelM DotProductArgs Unit := do
  let args ← getArgs
  let n := args.n
  let elementsPerThread := args.elementsPerThread
  let A : GlobalArray Float := ⟨args.A⟩
  let B : GlobalArray Float := ⟨args.B⟩
  let partialSums : GlobalArray Float := ⟨args.partialSums⟩

  let tid ← globalIdxX
  let baseIdx := tid * elementsPerThread

  -- Each thread accumulates elementsPerThread products
  let mut acc : Float := 0.0
  for i in [:elementsPerThread] do
    let idx := baseIdx + i
    if idx < n then do
      let aVal ← A.get idx
      let bVal ← B.get idx
      acc := acc + aVal * bVal

  -- Write partial sum
  partialSums.set tid acc



def dotProductSpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec dotProductKernelIR config grid

theorem dotProduct_safe : ∀ (config grid : Dim3), KernelSafe (dotProductSpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, dotProductSpec, deviceIRToKernelSpec, dotProductKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial


/-! ## CPU Reference Implementation -/

def dotProductCPU (A B : Array Float) : Float := Id.run do
  let mut acc : Float := 0.0
  for i in [:A.size] do
    acc := acc + A[i]! * B[i]!
  return acc

/-! ## GPU Launcher -/

def launchDotProductTimed (A B : Array Float) (elementsPerThread : Nat) : IO (CLean.GPU.ProcessLauncher.GPUResult DotProductArgsResponse) := do
  let size := A.size
  let numThreads := (size + elementsPerThread - 1) / elementsPerThread
  let scalarParams : Array ScalarValue := #[.int size, .int elementsPerThread]
  let arrays := [(`A, A), (`B, B), (`partialSums, Array.replicate numThreads 0.0)]
  let grid : Dim3 := ⟨(numThreads + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed dotProductKernelIR DotProductArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"DotProduct: size={size}"

  -- Generate input data
  let A ← randomFloatArray size 42
  let B ← randomFloatArray size 123

  let elementsPerThread : Nat := 256

  -- CPU benchmark
  let (cpuResult, cpuTime) ← timeMs (pure (dotProductCPU A B))

  -- GPU benchmark (cLean)
  let gpuRes ← launchDotProductTimed A B elementsPerThread

  -- Sum partial results on CPU
  let gpuResult := gpuRes.result.partialSums.foldl (· + ·) 0.0

  -- Check correctness with tolerance for floating point
  let relError := Float.abs (gpuResult - cpuResult) / (Float.abs cpuResult + 1e-10)
  let correct := relError < 0.001  -- 0.1% tolerance

  -- Convert DetailedTiming from ProcessLauncher to Harness.TimingBreakdown
  let breakdown : Option TimingBreakdown := gpuRes.detailedTiming.map fun dt => {
    h2dTransferMs := dt.h2dTransferMs
    kernelExecutionMs := dt.kernelExecutionMs
    d2hTransferMs := dt.d2hTransferMs
    jsonSerializeMs := dt.jsonSerializeMs
    processSpawnMs := dt.processSpawnMs
    jsonParseMs := dt.jsonParseMs
    totalMs := gpuRes.totalTimeMs
  }

  return {
    kernelName := "DotProduct"
    inputSize := size
    cpuTimeMs := cpuTime
    gpuTotalTimeMs := gpuRes.totalTimeMs
    gpuKernelOnlyMs := gpuRes.kernelTimeMs
    cudaReferenceMs := none
    correct := correct
    breakdown := breakdown
  }

def runAllBenchmarks : IO KernelBenchmarkSuite := do
  separator
  progress "Running DotProduct benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "DotProduct"
    description := "Vector dot product: sum = Σ A[i] * B[i]"
    results := results
  }

#eval do
  IO.println "=== DotProduct Generated CUDA ==="
  IO.println (kernelToCuda dotProductKernelIR)

end CLean.Benchmarks.Dataset.DotProduct
