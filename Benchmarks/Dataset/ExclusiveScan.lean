/-
  Benchmark Dataset: Exclusive Scan (Prefix Sum)

  Kernel: output[i] = sum(input[0..i-1])
  Pattern: Parallel scan with barriers (Blelloch algorithm)
  Source: CUDA Samples scan
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher
import Benchmarks.Harness

import CLean.ToGPUVerifyIR
open CLean.ToGPUVerifyIR CLean.Verification CLean.Verification.GPUVerify


namespace CLean.Benchmarks.Dataset.ExclusiveScan

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open CLean.Benchmarks

set_option maxHeartbeats 400000

/-! ## Kernel Arguments -/

-- Single block scan for simplicity
-- For full multi-block scan, would need multiple kernel launches
kernelArgs ScanArgs(n: Nat)
  global[input: Array Float]
  global[output: Array Float]

/-! ## cLean GPU Kernel (Naive version) -/

-- Note: This is a simplified O(n²) per-thread version
-- Full Blelloch scan would use shared memory and barriers
-- This demonstrates the scan pattern in a verifiable way
device_kernel scanKernel : KernelM ScanArgs Unit := do
  let args ← getArgs
  let n := args.n
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let idx ← globalIdxX

  if idx < n then do
    -- Compute prefix sum for this position
    let mut acc : Float := 0.0
    for j in [:idx] do
      let val ← input.get j
      acc := acc + val
    output.set idx acc



def scanSpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec scanKernelIR config grid

theorem scan_safe : ∀ (config grid : Dim3), KernelSafe (scanSpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, scanSpec, deviceIRToKernelSpec, scanKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial


/-! ## CPU Reference Implementation -/

def exclusiveScanCPU (input : Array Float) : Array Float := Id.run do
  let n := input.size
  let mut output := Array.mkArray n 0.0
  let mut acc : Float := 0.0
  for i in [:n] do
    output := output.set! i acc
    acc := acc + input[i]!
  return output

/-! ## GPU Launcher -/

def launchScanTimed (input : Array Float) : IO (CLean.GPU.ProcessLauncher.GPUResult ScanArgsResponse) := do
  let size := input.size
  let scalarParams : Array ScalarValue := #[.int size]
  let arrays := [(`input, input), (`output, Array.replicate size 0.0)]
  let grid : Dim3 := ⟨(size + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩
  CLean.GPU.ProcessLauncher.runKernelGPUTimed scanKernelIR ScanArgsResponse grid block scalarParams arrays (quiet := true)

/-! ## Benchmark Runner -/

def runBenchmark (size : Nat) : IO BenchmarkResult := do
  progress s!"ExclusiveScan: size={size}"

  -- Generate input data
  let input ← randomFloatArray size 42

  -- CPU benchmark
  let (cpuResult, cpuTime) ← timeMs (pure (exclusiveScanCPU input))

  -- GPU benchmark (cLean)
  let gpuRes ← launchScanTimed input

  let correct := arraysApproxEqual cpuResult gpuRes.result.output (tol := 0.01)

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
    kernelName := "ExclusiveScan"
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
  progress "Running ExclusiveScan benchmarks"
  separator

  -- Use smaller sizes for JSON serialization
  let scanSizes := #[256, 512, 1024, 2048]

  let mut results := #[]
  for size in scanSizes do
    let result ← runBenchmark size
    printResultSummary result
    results := results.push result

  return {
    kernelName := "ExclusiveScan"
    description := "Exclusive prefix sum: output[i] = sum(input[0..i-1])"
    results := results
  }

#eval do
  IO.println "=== ExclusiveScan Generated CUDA ==="
  IO.println (kernelToCuda scanKernelIR)

end CLean.Benchmarks.Dataset.ExclusiveScan
