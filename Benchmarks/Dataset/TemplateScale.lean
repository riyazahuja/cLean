/-
  Benchmark Dataset: Template Scale

  Kernel: data[i] = data[i] * blockDim.x
  Pattern: Uses blockDim intrinsic, demonstrates template-style scaling
  Source: CUDA Samples template
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

namespace CLean.Benchmarks.Dataset.TemplateScale

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs TemplateScaleArgs(n: Nat, scale: Float)
  global[data: Array Float]

/-! ## cLean GPU Kernel -/

-- Scale each element by a constant factor
device_kernel templateScaleKernel : KernelM TemplateScaleArgs Unit := do
  let args ← getArgs
  let n := args.n
  let scale := args.scale
  let data : GlobalArray Float := ⟨args.data⟩

  let idx ← globalIdxX

  if idx < n then do
    let val ← data.get idx
    data.set idx (val * scale)

/-! ## CPU Reference Implementation -/

def templateScaleCPU (scale : Float) (data : Array Float) : Array Float := Id.run do
  let mut result := data
  for i in [:data.size] do
    result := result.set! i (data[i]! * scale)
  return result

/-! ## GPU Launcher -/

def launchTemplateScaleTimed (data : Array Float) (scale : Float) (blockSize : Nat := 256) : IO (CLean.GPU.ProcessLauncher.GPUResult TemplateScaleArgsResponse) := do
  let n := data.size
  let scalarParams : Array ScalarValue := #[.int n, .float scale]
  let arrays := [(`data, data)]
  let grid : Dim3 := ⟨(n + blockSize - 1) / blockSize, 1, 1⟩
  let block : Dim3 := ⟨blockSize, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed templateScaleKernelIR TemplateScaleArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) (blockSize : Nat := 256) : IO BenchmarkResult := do
  progress s!"TemplateScale: size={size}, blockSize={blockSize}"

  let data ← randomFloatArray size 42
  let scale := Float.ofNat blockSize

  let (cpuResult, cpuTime) ← timeMs (pure (templateScaleCPU scale data))
  let gpuRes ← launchTemplateScaleTimed data scale blockSize

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
    kernelName := "TemplateScale"
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
  progress "Running TemplateScale benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "TemplateScale"
    description := "Scale by blockDim: data[i] = data[i] * blockDim.x"
    results := results
  }

#eval do
  IO.println "=== TemplateScale Generated CUDA ==="
  IO.println (kernelToCuda templateScaleKernelIR)

end CLean.Benchmarks.Dataset.TemplateScale
