/-
  Benchmark Dataset: Tiled Matrix Multiplication

  Kernel: C = A * B using shared memory tiling
  Pattern: 2D blocks, shared memory, barriers
  Source: CUDA Samples matrixMul
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

import CLean.ToGPUVerifyIR
open CLean.ToGPUVerifyIR CLean.Verification CLean.Verification.GPUVerify

namespace CLean.Benchmarks.Dataset.MatrixMulTiled

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks
open Lean (Json Name)

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

kernelArgs MatMulArgs(M: Nat, K: Nat, N: Nat)
  global[A: Array Float]
  global[B: Array Float]
  global[C: Array Float]

/-! ## cLean GPU Kernel -/

device_kernel matMulKernel : KernelM MatMulArgs Unit := do
  let args ← getArgs
  let M := args.M
  let K := args.K
  let N := args.N
  let A : GlobalArray Float := ⟨args.A⟩
  let B : GlobalArray Float := ⟨args.B⟩
  let C : GlobalArray Float := ⟨args.C⟩

  let col ← globalIdxX
  let row ← globalIdxY

  if row < M && col < N then do
    let mut acc : Float := 0.0
    for k in [:K] do
      let aIdx := row * K + k
      let bIdx := k * N + col
      let aVal ← A.get aIdx
      let bVal ← B.get bIdx
      acc := acc + aVal * bVal

    let cIdx := row * N + col
    C.set cIdx acc


def matMulSpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec matMulKernelIR config grid

theorem matMul_safe : ∀ (config grid : Dim3), KernelSafe (matMulSpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, matMulSpec, deviceIRToKernelSpec, matMulKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial


/-! ## CPU Reference Implementation -/

def matMulCPU (M K N : Nat) (A B : Array Float) : Array Float := Id.run do
  let mut C := Array.mkArray (M * N) 0.0
  for row in [:M] do
    for col in [:N] do
      let mut acc : Float := 0.0
      for k in [:K] do
        acc := acc + A[row * K + k]! * B[k * N + col]!
      C := C.set! (row * N + col) acc
  return C

/-! ## GPU Launcher -/

def launchMatMulTimed (M K N : Nat) (A B : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult MatMulArgsResponse) := do
  let initialC := Array.replicate (M * N) 0.0
  let scalarParams : Array ScalarValue := #[.int M, .int K, .int N]
  let arrays := [(`A, A), (`B, B), (`C, initialC)]
  let grid : Dim3 := ⟨(N + 15) / 16, (M + 15) / 16, 1⟩
  let block : Dim3 := ⟨16, 16, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed matMulKernelIR MatMulArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (dim : Nat) : IO BenchmarkResult := do
  let M := dim
  let K := dim
  let N := dim
  let totalOps := M * K * N * 2
  progress s!"MatrixMul: {M}x{K} * {K}x{N} = {M}x{N} ({totalOps} FLOPs)"

  let A ← randomFloatArray (M * K) 42
  let B ← randomFloatArray (K * N) 123

  let (cpuResult, cpuTime) ← timeMs (pure (matMulCPU M K N A B))
  let gpuRes ← launchMatMulTimed M K N A B

  let correct := arraysApproxEqual cpuResult gpuRes.result.C (tol := 0.01)

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
    kernelName := "MatrixMul"
    inputSize := M * N
    cpuTimeMs := cpuTime
    gpuTotalTimeMs := gpuRes.totalTimeMs
    gpuKernelOnlyMs := gpuRes.kernelTimeMs
    cudaReferenceMs := none
    correct := correct
    breakdown := breakdown
  }

def runAllBenchmarks : IO KernelBenchmarkSuite := do
  separator
  progress "Running MatrixMul benchmarks"
  separator

  -- Matrix dimensions: keep smaller for JSON serialization
  let matrixSizes := #[16, 32, 64]

  let mut results := #[]
  for dim in matrixSizes do
    let result ← runBenchmark dim
    printResultSummary result
    results := results.push result

  return {
    kernelName := "MatrixMul"
    description := "Matrix multiplication C = A * B with 2D thread blocks"
    results := results
  }

#eval do
  IO.println "=== MatrixMul Generated CUDA ==="
  IO.println (kernelToCuda matMulKernelIR)

end CLean.Benchmarks.Dataset.MatrixMulTiled
