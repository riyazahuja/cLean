/-
  Benchmark Dataset: Vector Square

  Kernel: A[i] = A[i] * A[i]
  Pattern: In-place modification
  Source: CUDA Samples simpleOccupancy
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

namespace CLean.Benchmarks.Dataset.VectorSquare

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs VectorSquareArgs(n: Nat)
  global[data: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel vectorSquareKernel : KernelM VectorSquareArgs Unit := do
  let args ← getArgs
  let n := args.n
  let data : GlobalArray Float := ⟨args.data⟩

  let idx ← globalIdxX
  if idx < n then do
    let val ← data.get idx
    data.set idx (val * val)

/-! ## CPU Reference Implementation -/

def vectorSquareCPU (data : Array Float) : Array Float := Id.run do
  let mut result := data
  for i in [:data.size] do
    let val := data[i]!
    result := result.set! i (val * val)
  return result

/-! ## GPU Launcher -/

def launchVectorSquareTimed (data : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult VectorSquareArgsResponse) := do
  let n := data.size
  let scalarParams : Array ScalarValue := #[.int n]
  let arrays := [(`data, data)]
  let grid : Dim3 := ⟨(n + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed vectorSquareKernelIR VectorSquareArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"VectorSquare: size={size}"

  let data ← randomFloatArray size 42

  let (cpuResult, cpuTime) ← timeMs (pure (vectorSquareCPU data))
  let gpuRes ← launchVectorSquareTimed data

  let correct := arraysApproxEqual cpuResult gpuRes.result.data

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
    kernelName := "VectorSquare"
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
  progress "Running VectorSquare benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "VectorSquare"
    description := "Element-wise squaring: A[i] = A[i] * A[i]"
    results := results
  }

#eval do
  IO.println "=== VectorSquare Generated CUDA ==="
  IO.println (kernelToCuda vectorSquareKernelIR)

end CLean.Benchmarks.Dataset.VectorSquare
