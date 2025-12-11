/-
  Benchmark Dataset: SAXPY

  Kernel: Y[i] = alpha * X[i] + Y[i]
  Pattern: Scalar parameter, fused multiply-add
  Source: Common BLAS operation
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

import CLean.ToGPUVerifyIR
open CLean.ToGPUVerifyIR CLean.Verification CLean.Verification.GPUVerify


namespace CLean.Benchmarks.Dataset.Saxpy

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)


set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs SaxpyArgs(n: Nat, alpha: Float)
  global[x: Array Float]
  global[y: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel saxpyKernel : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let n := args.n
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩

  let idx ← globalIdxX
  if idx < n then do
    let xval ← x.get idx
    let yval ← y.get idx
    y.set idx (alpha * xval + yval)


def saxpySpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec saxpyKernelIR config grid

theorem saxpy_safe : ∀ (config grid : Dim3), KernelSafe (saxpySpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, saxpySpec, deviceIRToKernelSpec, saxpyKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial


/-! ## CPU Reference Implementation -/

def saxpyCPU (alpha : Float) (x y : Array Float) : Array Float := Id.run do
  let mut result := y
  for i in [:x.size] do
    result := result.set! i (alpha * x[i]! + y[i]!)
  return result

/-! ## GPU Launcher -/

def launchSaxpyTimed (alpha : Float) (x y : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult SaxpyArgsResponse) := do
  let n := x.size
  let scalarParams : Array ScalarValue := #[.int n, .float alpha]
  let arrays := [(`x, x), (`y, y)]
  let grid : Dim3 := ⟨(n + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed saxpyKernelIR SaxpyArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"SAXPY: size={size}"

  let alpha : Float := 2.5
  let x ← randomFloatArray size 42
  let y ← randomFloatArray size 123

  let (cpuResult, cpuTime) ← timeMs (pure (saxpyCPU alpha x y))
  let gpuRes ← launchSaxpyTimed alpha x y

  let correct := arraysApproxEqual cpuResult gpuRes.result.y

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
    kernelName := "SAXPY"
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
  progress "Running SAXPY benchmarks"
  separator

  let mut results := #[]
  for size in inputSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "SAXPY"
    description := "Scalar alpha * X + Y: Y[i] = alpha * X[i] + Y[i]"
    results := results
  }

#eval do
  IO.println "=== SAXPY Generated CUDA ==="
  IO.println (kernelToCuda saxpyKernelIR)

end CLean.Benchmarks.Dataset.Saxpy
