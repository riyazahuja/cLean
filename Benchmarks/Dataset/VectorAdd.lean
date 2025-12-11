/-
  Benchmark Dataset: Vector Addition

  Kernel: C[i] = A[i] + B[i]
  Pattern: Embarrassingly parallel, no shared memory
  Source: CUDA Samples vectorAdd
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

namespace CLean.Benchmarks.Dataset.VectorAdd

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs VectorAddArgs(n: Nat)
  global[A B C: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel vectorAddKernel : KernelM VectorAddArgs Unit := do
  let args ← getArgs
  let n := args.n
  let A : GlobalArray Float := ⟨args.A⟩
  let B : GlobalArray Float := ⟨args.B⟩
  let C : GlobalArray Float := ⟨args.C⟩

  let idx ← globalIdxX
  if idx < n then do
    let a ← A.get idx
    let b ← B.get idx
    C.set idx (a + b)

/-! ## CPU Reference Implementation -/

def vectorAddCPU (A B : Array Float) : Array Float := Id.run do
  let n := A.size
  let mut C := Array.replicate n 0.0
  for i in [:n] do
    C := C.set! i (A[i]! + B[i]!)
  return C

/-! ## GPU Launcher -/

def launchVectorAddTimed (A B : Array Float) (debug : Bool := false) : IO (CLean.GPU.ProcessLauncher.GPUResult VectorAddArgsResponse) := do
  let n := A.size
  let scalarParams : Array ScalarValue := #[.int n]
  let arrays := [(`A, A), (`B, B), (`C, Array.replicate n 0.0)]
  let grid : Dim3 := ⟨(n + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed vectorAddKernelIR VectorAddArgsResponse grid block scalarParams arrays (quiet := true) (debug := debug)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"VectorAdd: size={size}"

  -- Generate input data
  let A ← randomFloatArray size 42
  let B ← randomFloatArray size 123

  -- CPU benchmark
  let (cpuResult, cpuTime) ← timeMs (pure (vectorAddCPU A B))

  -- GPU benchmark (cLean) with timing breakdown
  let gpuRes ← launchVectorAddTimed A B
  let gpuResult := gpuRes.result.C

  -- Check correctness
  let correct := arraysApproxEqual cpuResult gpuResult

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
    kernelName := "VectorAdd"
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
  progress "Running VectorAdd benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "VectorAdd"
    description := "Element-wise vector addition: C[i] = A[i] + B[i]"
    results := results
  }

/-! ## Generated CUDA Code (for reference) -/

#eval do
  IO.println "=== VectorAdd Generated CUDA ==="
  IO.println (kernelToCuda vectorAddKernelIR)

end CLean.Benchmarks.Dataset.VectorAdd
