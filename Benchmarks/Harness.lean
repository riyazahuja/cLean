/-
  Benchmark Harness for cLean

  Provides timing infrastructure and result logging for benchmarking
  cLean GPU kernels against CPU implementations.
-/

import Lean
import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher

namespace CLean.Benchmarks

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher
open Lean (Json Name)

/-! ## Timing Infrastructure -/

/-- Get current time in nanoseconds -/
def getCurrentTimeNs : IO Nat := IO.monoNanosNow

/-- Measure execution time of an IO action in milliseconds -/
def timeMs (action : IO α) : IO (α × Float) := do
  let startTime ← getCurrentTimeNs
  let result ← action
  let endTime ← getCurrentTimeNs
  let elapsedNs := endTime - startTime
  let elapsedMs := Float.ofNat elapsedNs / 1000000.0
  return (result, elapsedMs)

/-- Measure execution time, returning only the time -/
def timeOnlyMs (action : IO α) : IO Float := do
  let (_, t) ← timeMs action
  return t

/-! ## Result Types -/

/-- Detailed timing breakdown for a single kernel execution -/
structure TimingBreakdown where
  -- GPU-side (from CUDA events)
  h2dTransferMs : Float := 0.0
  kernelExecutionMs : Float := 0.0
  d2hTransferMs : Float := 0.0
  -- Lean-side
  jsonSerializeMs : Float := 0.0
  processSpawnMs : Float := 0.0
  jsonParseMs : Float := 0.0
  -- Total
  totalMs : Float := 0.0
  deriving Repr, Inhabited

open Lean Json in
instance : ToJson TimingBreakdown where
  toJson t := Json.mkObj [
    ("h2dTransferMs", toJson t.h2dTransferMs),
    ("kernelExecutionMs", toJson t.kernelExecutionMs),
    ("d2hTransferMs", toJson t.d2hTransferMs),
    ("jsonSerializeMs", toJson t.jsonSerializeMs),
    ("processSpawnMs", toJson t.processSpawnMs),
    ("jsonParseMs", toJson t.jsonParseMs),
    ("totalMs", toJson t.totalMs)
  ]

open Lean Json in
instance : FromJson TimingBreakdown where
  fromJson? json := do
    let h2d ← (json.getObjValAs? Float "h2dTransferMs").toOption.getD 0.0 |> pure
    let kernel ← (json.getObjValAs? Float "kernelExecutionMs").toOption.getD 0.0 |> pure
    let d2h ← (json.getObjValAs? Float "d2hTransferMs").toOption.getD 0.0 |> pure
    let jsonSer ← (json.getObjValAs? Float "jsonSerializeMs").toOption.getD 0.0 |> pure
    let spawn ← (json.getObjValAs? Float "processSpawnMs").toOption.getD 0.0 |> pure
    let jsonParse ← (json.getObjValAs? Float "jsonParseMs").toOption.getD 0.0 |> pure
    let total ← (json.getObjValAs? Float "totalMs").toOption.getD 0.0 |> pure
    return { h2dTransferMs := h2d, kernelExecutionMs := kernel, d2hTransferMs := d2h,
             jsonSerializeMs := jsonSer, processSpawnMs := spawn, jsonParseMs := jsonParse, totalMs := total }

/-- Result of a single benchmark run -/
structure BenchmarkResult where
  kernelName : String
  inputSize : Nat
  cpuTimeMs : Float
  gpuTotalTimeMs : Float
  gpuKernelOnlyMs : Float
  cudaReferenceMs : Option Float := none
  correct : Bool
  breakdown : Option TimingBreakdown := none
  gpuUtilization : Option Float := none
  deriving Repr, Inhabited

open Lean Json in
instance : ToJson BenchmarkResult where
  toJson r := Json.mkObj [
    ("kernelName", toJson r.kernelName),
    ("inputSize", toJson r.inputSize),
    ("cpuTimeMs", toJson r.cpuTimeMs),
    ("gpuTotalTimeMs", toJson r.gpuTotalTimeMs),
    ("gpuKernelOnlyMs", toJson r.gpuKernelOnlyMs),
    ("cudaReferenceMs", toJson r.cudaReferenceMs),
    ("correct", toJson r.correct),
    ("breakdown", toJson r.breakdown),
    ("gpuUtilization", toJson r.gpuUtilization)
  ]

open Lean Json in
instance : FromJson BenchmarkResult where
  fromJson? json := do
    let name ← json.getObjValAs? String "kernelName"
    let size ← json.getObjValAs? Nat "inputSize"
    let cpu ← json.getObjValAs? Float "cpuTimeMs"
    let gpuTotal ← json.getObjValAs? Float "gpuTotalTimeMs"
    let gpuKernel ← json.getObjValAs? Float "gpuKernelOnlyMs"
    let cuda := (json.getObjValAs? Float "cudaReferenceMs").toOption
    let correct ← json.getObjValAs? Bool "correct"
    let breakdown := (json.getObjValAs? TimingBreakdown "breakdown").toOption
    let util := (json.getObjValAs? Float "gpuUtilization").toOption
    return { kernelName := name, inputSize := size, cpuTimeMs := cpu, gpuTotalTimeMs := gpuTotal,
             gpuKernelOnlyMs := gpuKernel, cudaReferenceMs := cuda, correct := correct,
             breakdown := breakdown, gpuUtilization := util }

/-- Collection of results for a kernel across all input sizes -/
structure KernelBenchmarkSuite where
  kernelName : String
  description : String
  results : Array BenchmarkResult
  deriving Repr, Inhabited

open Lean Json in
instance : ToJson KernelBenchmarkSuite where
  toJson s := Json.mkObj [
    ("kernelName", toJson s.kernelName),
    ("description", toJson s.description),
    ("results", toJson s.results)
  ]

open Lean Json in
instance : FromJson KernelBenchmarkSuite where
  fromJson? json := do
    let name ← json.getObjValAs? String "kernelName"
    let desc ← json.getObjValAs? String "description"
    let results ← json.getObjValAs? (Array BenchmarkResult) "results"
    return { kernelName := name, description := desc, results := results }

/-! ## Input Sizes -/

/-- Standard input sizes for benchmarking
    NOTE: Using smaller sizes due to JSON serialization overhead.
    For production benchmarks, use binary data transfer instead. -/
def inputSizes : Array Nat := #[
  256,         -- Tiny: 256
  512,         -- Small: 512
  1024,        -- Medium: 1K
  2048,        -- Large: 2K
  -- 4096         -- Max for JSON: 4K
]

/-- Matrix sizes (side length) for 2D kernels -/
def matrixSizes : Array Nat := #[
  16,    -- 16×16 = 256
  32,    -- 32×32 = 1K
  -- 64,    -- 64×64 = 4K
  -- 128,   -- 128×128 = 16K (max for JSON)
  -- 256    -- 256×256 = 64K (may be slow)
]

/-! ## Correctness Checking -/

/-- Check if two float arrays are approximately equal -/
def arraysApproxEqual (a b : Array Float) (tol : Float := 1e-5) : Bool :=
  if a.size != b.size then false
  else Id.run do
    for i in [:a.size] do
      let diff := (a[i]! - b[i]!).abs
      let maxVal := max a[i]!.abs b[i]!.abs
      let relErr := if maxVal > 0 then diff / maxVal else diff
      if relErr > tol && diff > tol then
        return false
    return true

/-- Check if two int arrays are equal -/
def arraysEqualInt (a b : Array Int) : Bool :=
  if a.size != b.size then false
  else Id.run do
    for i in [:a.size] do
      if a[i]! != b[i]! then
        return false
    return true

/-! ## Progress Reporting -/

/-- Print progress message -/
def progress (msg : String) : IO Unit := do
  IO.println s!"[BENCHMARK] {msg}"

/-- Print a horizontal separator -/
def separator : IO Unit := do
  IO.println "═══════════════════════════════════════════════════════════════"

/-- Print benchmark result summary -/
def printResultSummary (r : BenchmarkResult) : IO Unit := do
  let speedup := r.cpuTimeMs / r.gpuTotalTimeMs
  IO.println s!"  Size: {r.inputSize}"
  IO.println s!"    CPU:      {r.cpuTimeMs.toString} ms"
  IO.println s!"    GPU (total): {r.gpuTotalTimeMs.toString} ms"
  IO.println s!"    GPU (kernel): {r.gpuKernelOnlyMs.toString} ms"
  IO.println s!"    Speedup:  {speedup.toString}x"
  IO.println s!"    Correct:  {r.correct}"

/-! ## Random Data Generation -/

/-- Generate random float array -/
def randomFloatArray (n : Nat) (seed : Nat := 42) : IO (Array Float) := do
  let mut arr := Array.mkEmpty n
  let mut state := seed
  for _ in [:n] do
    -- Simple LCG random number generator
    state := (state * 1103515245 + 12345) % (2^31)
    let val := Float.ofNat (state % 1000) / 1000.0
    arr := arr.push val
  return arr

/-- Generate random int array in range [0, maxVal) -/
def randomIntArray (n : Nat) (maxVal : Nat := 1000) (seed : Nat := 42) : IO (Array Int) := do
  let mut arr := Array.mkEmpty n
  let mut state := seed
  for _ in [:n] do
    state := (state * 1103515245 + 12345) % (2^31)
    let val := Int.ofNat (state % maxVal)
    arr := arr.push val
  return arr

/-- Generate array of zeros -/
def zeroFloatArray (n : Nat) : Array Float :=
  Array.replicate n 0.0

def zeroIntArray (n : Nat) : Array Int :=
  Array.replicate n 0

/-! ## Extended Input Sizes -/

/-- Extended input sizes for more thorough benchmarking -/
def extendedInputSizes : Array Nat := #[
  1024,    -- 1K
  2048,    -- 2K
  4096,    -- 4K
  8192,    -- 8K
  16384    -- 16K (max recommended for JSON)
]

/-! ## GPU Monitoring -/

/-- Sample GPU utilization using nvidia-smi -/
def sampleGpuUtilization : IO Float := do
  let output ← IO.Process.output {
    cmd := "nvidia-smi"
    args := #["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
  }
  if output.exitCode == 0 then
    match output.stdout.trim.toNat? with
    | some n => return Float.ofNat n
    | none => return 0.0
  else return 0.0

/-- Get GPU memory info (used, total) in MB -/
def getGpuMemoryInfo : IO (Float × Float) := do
  let output ← IO.Process.output {
    cmd := "nvidia-smi"
    args := #["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
  }
  if output.exitCode == 0 then
    let parts := output.stdout.trim.splitOn ","
    match parts.get? 0, parts.get? 1 with
    | some usedStr, some totalStr =>
      let used := usedStr.trim.toNat?.getD 0
      let total := totalStr.trim.toNat?.getD 0
      return (Float.ofNat used, Float.ofNat total)
    | _, _ => return (0.0, 0.0)
  else return (0.0, 0.0)

/-! ## Export Functions -/

/-- Export benchmark results to JSON file -/
def exportResultsToJson (suites : Array KernelBenchmarkSuite) (path : String) : IO Unit := do
  let json := Lean.toJson suites
  IO.FS.writeFile path json.pretty

/-- Convert a BenchmarkResult to a CSV row -/
def benchmarkResultToCsvRow (r : BenchmarkResult) : String :=
  let h2d := r.breakdown.map (·.h2dTransferMs) |>.getD 0.0
  let kernel := r.breakdown.map (·.kernelExecutionMs) |>.getD 0.0
  let d2h := r.breakdown.map (·.d2hTransferMs) |>.getD 0.0
  let jsonSer := r.breakdown.map (·.jsonSerializeMs) |>.getD 0.0
  let spawn := r.breakdown.map (·.processSpawnMs) |>.getD 0.0
  let jsonParse := r.breakdown.map (·.jsonParseMs) |>.getD 0.0
  let cudaMs := r.cudaReferenceMs.map toString |>.getD "N/A"
  let speedup := r.cpuTimeMs / r.gpuTotalTimeMs
  let util := r.gpuUtilization.map toString |>.getD "N/A"
  s!"{r.kernelName},{r.inputSize},{r.cpuTimeMs},{r.gpuTotalTimeMs},{r.gpuKernelOnlyMs},{h2d},{d2h},{cudaMs},{jsonSer},{spawn},{jsonParse},{speedup},{r.correct},{util}"

/-- Export benchmark results to CSV file -/
def exportResultsToCsv (suites : Array KernelBenchmarkSuite) (path : String) : IO Unit := do
  let header := "kernel,input_size,cpu_ms,gpu_total_ms,gpu_kernel_ms,h2d_ms,d2h_ms,cuda_ms,json_serialize_ms,process_spawn_ms,json_parse_ms,speedup_vs_cpu,correct,gpu_util"
  let rows := suites.foldl (fun acc suite =>
    suite.results.foldl (fun acc r => acc.push (benchmarkResultToCsvRow r)) acc
  ) #[]
  let content := header ++ "\n" ++ String.intercalate "\n" rows.toList
  IO.FS.writeFile path content

end CLean.Benchmarks
