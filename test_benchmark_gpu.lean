/-
  GPU Performance Benchmarking

  Tests and benchmarks the GPU execution system with different array sizes
  and measures compilation time, execution time, and cache effectiveness.
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache
import CLean.GPU.ProcessLauncher

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen
open CLean.GPU.KernelCache CLean.GPU.ProcessLauncher

-- SAXPY kernel: r[i] = alpha * x[i] + y[i]
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

/-- Simple timing wrapper -/
def timeIO (label : String) (action : IO α) : IO α := do
  let start ← IO.monoMsNow
  let result ← action
  let end_ ← IO.monoMsNow
  let elapsed := end_ - start
  IO.println s!"  [{label}] Time: {elapsed}ms"
  return result

/-- Run benchmark for a specific array size -/
def runBenchmark (n : Nat) (iterations : Nat := 1) : IO Unit := do
  IO.println s!"\n{'='.replicate 60}"
  IO.println s!" Benchmark: N = {n} ({n * 4 / 1024}KB per array)"
  IO.println s!"{'='.replicate 60}"

  -- Generate test data
  let alpha := 2.5
  let x := Array.range n |>.map (fun i => Float.ofNat i + 1.0)
  let y := Array.range n |>.map (fun _ => 1.0)

  IO.println "\n--- First Execution (includes compilation) ---"

  -- First run (will compile and cache)
  let scalarParams := #[Float.ofNat n, alpha]
  let arrays := [
    (`X, x),
    (`Y, y),
    (`R, Array.replicate n 0.0)
  ]

  let grid := ⟨(n + 255) / 256, 1, 1⟩  -- Ceiling division for grid size
  let block := ⟨256, 1, 1⟩

  let _ ← timeIO "Total (first run)" do
    executeKernel saxpyKernelIR grid block scalarParams arrays

  IO.println "\n--- Subsequent Executions (cached) ---"

  -- Run multiple iterations to measure cached performance
  for iter in [0:iterations] do
    let _ ← timeIO s!"Iteration {iter + 1}" do
      executeKernel saxpyKernelIR grid block scalarParams arrays

  -- Verify correctness on first few elements
  IO.println "\n--- Correctness Check ---"
  let results ← executeKernel saxpyKernelIR grid block scalarParams arrays
  let some (_, r) := results.find? fun (name, _) => name == `R
    | throw <| IO.userError "Result array not found"

  let checkSize := min n 5
  IO.println s!"First {checkSize} elements:"
  for i in [:checkSize] do
    let expected := alpha * x[i]! + y[i]!
    let actual := r[i]!
    let match := if (expected - actual).abs < 1e-5 then "✓" else "✗"
    IO.println s!"  r[{i}] = {actual} (expected {expected}) {match}"

  -- Calculate throughput
  let bytesPerElement := 4  -- Float is 4 bytes
  let totalBytes := n * bytesPerElement * 3  -- 3 arrays (x, y, r)
  let mbytes := Float.ofNat totalBytes / (1024.0 * 1024.0)
  IO.println s!"\nData size: {mbytes} MB"

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════════════╗"
  IO.println "║           cLean GPU Benchmark Suite                    ║"
  IO.println "╚════════════════════════════════════════════════════════╝"

  -- Small benchmark (8 elements - same as manual test)
  runBenchmark 8 5

  -- Medium benchmarks
  runBenchmark 1024 3       -- 4KB per array
  runBenchmark 10240 3      -- 40KB per array
  runBenchmark 102400 3     -- 400KB per array

  -- Large benchmarks
  runBenchmark 1048576 3    -- 4MB per array (1M elements)
  runBenchmark 10485760 2   -- 40MB per array (10M elements)

  IO.println "\n╔════════════════════════════════════════════════════════╗"
  IO.println "║           Benchmark Complete                           ║"
  IO.println "╚════════════════════════════════════════════════════════╝"

  IO.println "\nPerformance Tips:"
  IO.println "  • First run includes compilation (~500ms)"
  IO.println "  • Cached runs show true execution time"
  IO.println "  • Process spawn overhead: ~50ms per launch"
  IO.println "  • For production: consider persistent launcher process"
  IO.println "  • GPU most efficient for arrays >100K elements"
