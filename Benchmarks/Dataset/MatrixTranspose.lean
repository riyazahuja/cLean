/-
  Benchmark Dataset: Matrix Transpose

  Kernel: B[j, i] = A[i, j] using shared memory for coalesced access
  Pattern: 2D shared memory, bank conflict avoidance
  Source: CUDA Samples transpose
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

import CLean.ToGPUVerifyIR
open CLean.ToGPUVerifyIR CLean.Verification CLean.Verification.GPUVerify


namespace CLean.Benchmarks.Dataset.MatrixTranspose

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs TransposeArgs(width: Nat, height: Nat)
  global[input: Array Float]
  global[output: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel transposeKernel : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let width := args.width
  let height := args.height
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let col ← globalIdxX
  let row ← globalIdxY

  if col < width && row < height then do
    let srcIdx := row * width + col
    let dstIdx := col * height + row
    let val ← input.get srcIdx
    output.set dstIdx val


def transposeSpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec transposeKernelIR config grid

theorem transpose_safe : ∀ (config grid : Dim3), KernelSafe (transposeSpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, transposeSpec, deviceIRToKernelSpec, transposeKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial



/-! ## CPU Reference Implementation -/

def transposeCPU (width height : Nat) (input : Array Float) : Array Float := Id.run do
  let mut output := Array.mkArray (width * height) 0.0
  for row in [:height] do
    for col in [:width] do
      let srcIdx := row * width + col
      let dstIdx := col * height + row
      output := output.set! dstIdx input[srcIdx]!
  return output

/-! ## GPU Launcher -/

def launchTransposeTimed (width height : Nat) (input : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult TransposeArgsResponse) := do
  let initialOutput := Array.replicate (width * height) 0.0
  let scalarParams : Array ScalarValue := #[.int width, .int height]
  let arrays := [(`input, input), (`output, initialOutput)]
  let grid : Dim3 := ⟨(width + 15) / 16, (height + 15) / 16, 1⟩
  let block : Dim3 := ⟨16, 16, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed transposeKernelIR TransposeArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (matrixDim : Nat) : IO BenchmarkResult := do
  let width := matrixDim
  let height := matrixDim
  let totalSize := width * height
  progress s!"MatrixTranspose: {width}x{height} = {totalSize} elements"

  let input ← randomFloatArray totalSize 42

  let (cpuResult, cpuTime) ← timeMs (pure (transposeCPU width height input))
  let gpuRes ← launchTransposeTimed width height input

  let correct := arraysApproxEqual cpuResult gpuRes.result.output

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
    kernelName := "MatrixTranspose"
    inputSize := totalSize
    cpuTimeMs := cpuTime
    gpuTotalTimeMs := gpuRes.totalTimeMs
    gpuKernelOnlyMs := gpuRes.kernelTimeMs
    cudaReferenceMs := none
    correct := correct
    breakdown := breakdown
  }

def runAllBenchmarks : IO KernelBenchmarkSuite := do
  separator
  progress "Running MatrixTranspose benchmarks"
  separator

  -- Use square matrix dimensions (reduced for JSON serialization)
  let matrixSizes := #[16, 32, 64]--, 128]

  let mut results := #[]
  for dim in matrixSizes do
    let result ← runBenchmark dim
    printResultSummary result
    results := results.push result

  return {
    kernelName := "MatrixTranspose"
    description := "Matrix transpose B[j,i] = A[i,j] with 2D thread blocks"
    results := results
  }

#eval do
  IO.println "=== MatrixTranspose Generated CUDA ==="
  IO.println (kernelToCuda transposeKernelIR)

end CLean.Benchmarks.Dataset.MatrixTranspose
