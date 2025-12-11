/-
  Benchmark Dataset: Scalar Multiply with Loop

  Kernel: for i in [0..k): data[idx * k + i] *= scalar
  Pattern: Per-thread loops
  Source: CUDA Samples simpleStreams
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

namespace CLean.Benchmarks.Dataset.ScalarMul

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs ScalarMulArgs(n: Nat, scalar: Float, workPerThread: Nat)
  global[data: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel scalarMulKernel : KernelM ScalarMulArgs Unit := do
  let args ← getArgs
  let n := args.n
  let scalar := args.scalar
  let workPerThread := args.workPerThread
  let data : GlobalArray Float := ⟨args.data⟩

  let tid ← globalIdxX
  let baseIdx := tid * workPerThread

  for i in [:workPerThread] do
    let idx := baseIdx + i
    if idx < n then do
      let val ← data.get idx
      data.set idx (val * scalar)

/-! ## CPU Reference Implementation -/

def scalarMulCPU (scalar : Float) (data : Array Float) : Array Float := Id.run do
  let mut result := data
  for i in [:data.size] do
    result := result.set! i (data[i]! * scalar)
  return result

/-! ## GPU Launcher -/

def launchScalarMulTimed (scalar : Float) (workPerThread : Nat) (data : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult ScalarMulArgsResponse) := do
  let n := data.size
  let scalarParams : Array ScalarValue := #[.int n, .float scalar, .int workPerThread]
  let arrays := [(`data, data)]
  let numThreads := (n + workPerThread - 1) / workPerThread
  let grid : Dim3 := ⟨(numThreads + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed scalarMulKernelIR ScalarMulArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"ScalarMul: size={size}"

  let scalar : Float := 3.14159
  let workPerThread : Nat := 4
  let data ← randomFloatArray size 42

  let (cpuResult, cpuTime) ← timeMs (pure (scalarMulCPU scalar data))
  let gpuRes ← launchScalarMulTimed scalar workPerThread data

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
    kernelName := "ScalarMul"
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
  progress "Running ScalarMul benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "ScalarMul"
    description := "Per-thread loop: each thread multiplies workPerThread elements by scalar"
    results := results
  }

#eval do
  IO.println "=== ScalarMul Generated CUDA ==="
  IO.println (kernelToCuda scalarMulKernelIR)

end CLean.Benchmarks.Dataset.ScalarMul
