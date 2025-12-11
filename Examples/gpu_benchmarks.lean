/-
  GPU vs CPU Benchmark Suite

  Demonstrates cases where GPU acceleration provides significant speedup.

  Key insight: GPU wins when:
  1. Per-element computation is high (amortizes transfer cost)
  2. Large data sizes (parallelism matters more)
  3. Operations are embarrassingly parallel
-/

import CLean.Groebner
import CLean.FiniteField
import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher

namespace CLean.GPUBenchmarks

open Groebner
open FiniteField
open Polynomial
open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher

set_option maxHeartbeats 800000

/-! ## Benchmark 1: Element-wise Vector Operations (High Compute Intensity)

This kernel performs multiple expensive operations per element,
making the computation dominate over transfer time.
-/

kernelArgs VectorOpArgs(n: Nat, p: Nat, iterations: Nat)
  global[data result: Array Int]

-- Each thread performs multiple modular operations per element
-- This increases compute intensity to amortize transfer overhead
device_kernel vectorComputeKernel : KernelM VectorOpArgs Unit := do
  let args ← getArgs
  let n := args.n
  let p := args.p
  let iterations := args.iterations
  let data : GlobalArray Int := ⟨args.data⟩
  let result : GlobalArray Int := ⟨args.result⟩

  let idx ← globalIdxX
  if idx < n then do
    let val ← data.get idx
    -- Perform many iterations of compute-heavy modular arithmetic
    -- This is the key to GPU advantage: lots of work per element
    let mut acc := val
    for _i in [:iterations] do
      -- Multiple modular operations per iteration
      acc := (acc * acc) % p
      acc := (acc + val) % p
      acc := (acc * val) % p
    result.set idx acc

-- Show generated CUDA
#eval do
  IO.println "=== Vector Compute Kernel CUDA ==="
  IO.println (kernelToCuda vectorComputeKernelIR)

def launchVectorCompute (data : Array Int) (p : Nat) (iterations : Nat) : IO (Array Int) := do
  let n := data.size
  let result : Array Int := Array.replicate n 0

  let cached ← compileKernelToPTX vectorComputeKernelIR

  let scalarParams : Array ScalarValue := #[
    ScalarValue.int n,
    ScalarValue.int p,
    ScalarValue.int iterations
  ]
  let arrays : List (Lean.Name × Array Int) := [
    (`data, data),
    (`result, result)
  ]
  let jsonInput := buildLauncherInputBetter scalarParams arrays

  let grid : Dim3 := ⟨(n + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    vectorComputeKernelIR.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  child.stdin.putStr jsonInput
  child.stdin.putStr "\n"
  child.stdin.flush

  let stdout ← child.stdout.readToEnd
  let exitCode ← child.wait

  if exitCode != 0 then
    throw <| IO.userError s!"GPU kernel failed with exit code {exitCode}"

  match Lean.Json.parse stdout with
  | Except.error err => throw <| IO.userError s!"JSON parse error: {err}"
  | Except.ok json =>
    match json.getObjVal? "results" with
    | Except.error _ => throw <| IO.userError "Missing 'results' in output"
    | Except.ok results =>
      match results.getObjVal? "result" with
      | Except.error _ => throw <| IO.userError "Missing 'result' in results"
      | Except.ok resultJson =>
        match resultJson.getArr? with
        | Except.error _ => throw <| IO.userError "result is not an array"
        | Except.ok arr =>
          let resultData := arr.map fun v =>
            match v.getInt? with
            | Except.ok i => i
            | Except.error _ => 0
          pure resultData

def cpuVectorCompute (data : Array Int) (p : Nat) (iterations : Nat) : Array Int := Id.run do
  let pInt : Int := p
  let mut result := Array.replicate data.size (0 : Int)
  for idx in [:data.size] do
    let val := data[idx]!
    let mut acc := val
    for _ in [:iterations] do
      acc := ((acc * acc) % pInt + pInt) % pInt
      acc := ((acc + val) % pInt + pInt) % pInt
      acc := ((acc * val) % pInt + pInt) % pInt
    result := result.set! idx acc
  return result

/-! ## Benchmark 2: Matrix-Vector Multiplication

Classic GPU-friendly operation: each thread computes one output element
by reading a full row and dotting with the vector.
-/

kernelArgs MatVecArgs(rows: Nat, cols: Nat, p: Nat)
  global[matrix vector result: Array Int]

device_kernel matVecKernel : KernelM MatVecArgs Unit := do
  let args ← getArgs
  let rows := args.rows
  let cols := args.cols
  let p := args.p
  let matrix : GlobalArray Int := ⟨args.matrix⟩
  let vector : GlobalArray Int := ⟨args.vector⟩
  let result : GlobalArray Int := ⟨args.result⟩

  let row ← globalIdxX
  if row < rows then do
    let mut sum := 0
    for col in [:cols] do
      let idx := row * cols + col
      let matVal ← matrix.get idx
      let vecVal ← vector.get col
      let prod := (matVal * vecVal) % p
      sum := (sum + prod) % p
    result.set row sum

#eval do
  IO.println "=== Matrix-Vector Kernel CUDA ==="
  IO.println (kernelToCuda matVecKernelIR)

def launchMatVec (matrix : Array Int) (vector : Array Int) (rows cols : Nat) (p : Nat) : IO (Array Int) := do
  let result : Array Int := Array.replicate rows 0

  let cached ← compileKernelToPTX matVecKernelIR

  let scalarParams : Array ScalarValue := #[
    ScalarValue.int rows,
    ScalarValue.int cols,
    ScalarValue.int p
  ]
  let arrays : List (Lean.Name × Array Int) := [
    (`matrix, matrix),
    (`vector, vector),
    (`result, result)
  ]
  let jsonInput := buildLauncherInputBetter scalarParams arrays

  let grid : Dim3 := ⟨(rows + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    matVecKernelIR.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  child.stdin.putStr jsonInput
  child.stdin.putStr "\n"
  child.stdin.flush

  let stdout ← child.stdout.readToEnd
  let exitCode ← child.wait

  if exitCode != 0 then
    throw <| IO.userError s!"GPU kernel failed"

  match Lean.Json.parse stdout with
  | Except.error err => throw <| IO.userError s!"JSON parse error: {err}"
  | Except.ok json =>
    match json.getObjVal? "results" with
    | Except.error _ => throw <| IO.userError "Missing 'results'"
    | Except.ok results =>
      match results.getObjVal? "result" with
      | Except.error _ => throw <| IO.userError "Missing 'result'"
      | Except.ok resultJson =>
        match resultJson.getArr? with
        | Except.error _ => throw <| IO.userError "result is not an array"
        | Except.ok arr =>
          let resultData := arr.map fun v =>
            match v.getInt? with
            | Except.ok i => i
            | Except.error _ => 0
          pure resultData

def cpuMatVec (matrix : Array Int) (vector : Array Int) (rows cols : Nat) (p : Nat) : Array Int := Id.run do
  let pInt : Int := p
  let mut result := Array.replicate rows (0 : Int)
  for row in [:rows] do
    let mut sum : Int := 0
    for col in [:cols] do
      let idx := row * cols + col
      let matVal := matrix[idx]!
      let vecVal := vector[col]!
      let prod := (matVal * vecVal) % pInt
      sum := (sum + prod) % pInt
    result := result.set! row sum
  return result

/-! ## Benchmark 3: Parallel Polynomial Evaluation (Horner's method on multiple points)

Evaluate a polynomial at many points in parallel.
Each thread evaluates the polynomial at one point.
-/

kernelArgs PolyEvalArgs(npoints: Nat, degree: Nat, p: Nat)
  global[coeffs points results: Array Int]

device_kernel polyEvalKernel : KernelM PolyEvalArgs Unit := do
  let args ← getArgs
  let npoints := args.npoints
  let degree := args.degree
  let p := args.p
  let coeffs : GlobalArray Int := ⟨args.coeffs⟩
  let points : GlobalArray Int := ⟨args.points⟩
  let results : GlobalArray Int := ⟨args.results⟩

  let idx ← globalIdxX
  if idx < npoints then do
    let x ← points.get idx
    -- Horner's method: evaluate polynomial
    let mut acc := 0
    -- Start from highest degree coefficient
    for i in [:degree + 1] do
      let coeffIdx := degree - i
      let coeff ← coeffs.get coeffIdx
      acc := ((acc * x) % p + coeff) % p
    results.set idx acc

#eval do
  IO.println "=== Polynomial Evaluation Kernel CUDA ==="
  IO.println (kernelToCuda polyEvalKernelIR)

def launchPolyEval (coeffs : Array Int) (points : Array Int) (p : Nat) : IO (Array Int) := do
  let npoints := points.size
  let degree := coeffs.size - 1
  let results : Array Int := Array.replicate npoints 0

  let cached ← compileKernelToPTX polyEvalKernelIR

  let scalarParams : Array ScalarValue := #[
    ScalarValue.int npoints,
    ScalarValue.int degree,
    ScalarValue.int p
  ]
  let arrays : List (Lean.Name × Array Int) := [
    (`coeffs, coeffs),
    (`points, points),
    (`results, results)
  ]
  let jsonInput := buildLauncherInputBetter scalarParams arrays

  let grid : Dim3 := ⟨(npoints + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    polyEvalKernelIR.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  child.stdin.putStr jsonInput
  child.stdin.putStr "\n"
  child.stdin.flush

  let stdout ← child.stdout.readToEnd
  let exitCode ← child.wait

  if exitCode != 0 then
    throw <| IO.userError s!"GPU kernel failed"

  match Lean.Json.parse stdout with
  | Except.error _ => throw <| IO.userError "JSON parse error"
  | Except.ok json =>
    match json.getObjVal? "results" with
    | Except.error _ => throw <| IO.userError "Missing 'results'"
    | Except.ok results =>
      match results.getObjVal? "results" with
      | Except.error _ => throw <| IO.userError "Missing 'results' array"
      | Except.ok resultJson =>
        match resultJson.getArr? with
        | Except.error _ => throw <| IO.userError "results is not an array"
        | Except.ok arr =>
          let resultData := arr.map fun v =>
            match v.getInt? with
            | Except.ok i => i
            | Except.error _ => 0
          pure resultData

def cpuPolyEval (coeffs : Array Int) (points : Array Int) (p : Nat) : Array Int := Id.run do
  let pInt : Int := p
  let degree := coeffs.size - 1
  let mut results := Array.replicate points.size (0 : Int)
  for idx in [:points.size] do
    let x := points[idx]!
    let mut acc : Int := 0
    for i in [:degree + 1] do
      let coeffIdx := degree - i
      let coeff := coeffs[coeffIdx]!
      acc := ((acc * x) % pInt + coeff + pInt) % pInt
    results := results.set! idx acc
  return results


/-! ## Main Benchmark Runner -/

def PRIME : Nat := 65521

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════════╗"
  IO.println "║          GPU vs CPU Benchmark Suite for cLean                   ║"
  IO.println "╠══════════════════════════════════════════════════════════════════╣"
  IO.println "║  Testing scenarios where GPU provides significant speedup        ║"
  IO.println "╚══════════════════════════════════════════════════════════════════╝"
  IO.println ""

  -- ═══════════════════════════════════════════════════════════════════════════
  -- Benchmark 1: High-Compute Vector Operations
  -- ═══════════════════════════════════════════════════════════════════════════
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "BENCHMARK 1: High-Compute Vector Operations"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "Each element undergoes multiple iterations of modular arithmetic."
  IO.println "GPU excels because computation dominates transfer time."
  IO.println ""

  -- Test with different sizes and iteration counts
  let vectorTests := #[
    (10000,  100,  "10K elements, 100 iterations"),
    (50000,  100,  "50K elements, 100 iterations"),
    (100000, 100,  "100K elements, 100 iterations"),
    (100000, 200,  "100K elements, 200 iterations"),
    (100000, 500,  "100K elements, 500 iterations")
  ]

  for (n, iters, desc) in vectorTests do
    IO.println s!"  Test: {desc}"

    -- Generate random data
    let mut data : Array Int := #[]
    let mut seed := 12345
    for _ in [:n] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      data := data.push ((seed % PRIME) : Int)

    -- GPU timing
    let startGPU ← IO.monoNanosNow
    let gpuResult ← launchVectorCompute data PRIME iters
    let endGPU ← IO.monoNanosNow
    let gpuMs := (endGPU - startGPU).toFloat / 1000000.0

    -- CPU timing
    let startCPU ← IO.monoNanosNow
    let cpuResult := cpuVectorCompute data PRIME iters
    let endCPU ← IO.monoNanosNow
    let cpuMs := (endCPU - startCPU).toFloat / 1000000.0

    -- Verify correctness
    let correct := gpuResult[0]? == cpuResult[0]?

    let speedup := cpuMs / gpuMs
    let winner := if speedup > 1.0 then "GPU" else "CPU"
    let speedupStr := if speedup > 1.0 then s!"{speedup.toString.take 5}x faster"
                      else s!"{(1.0/speedup).toString.take 5}x slower"

    IO.println s!"    CPU: {cpuMs.toString.take 8}ms | GPU: {gpuMs.toString.take 8}ms | {winner} {speedupStr} | Correct: {correct}"

  IO.println ""

  -- ═══════════════════════════════════════════════════════════════════════════
  -- Benchmark 2: Matrix-Vector Multiplication
  -- ═══════════════════════════════════════════════════════════════════════════
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "BENCHMARK 2: Matrix-Vector Multiplication"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "Each thread computes one output element (dot product of row and vector)."
  IO.println "GPU excels on large matrices where parallelism outweighs overhead."
  IO.println ""

  let matVecTests := #[
    (500,  500,  "500×500 matrix"),
    (1000, 1000, "1000×1000 matrix"),
    (2000, 2000, "2000×2000 matrix"),
    (1000, 5000, "1000×5000 matrix (wide)"),
    (5000, 1000, "5000×1000 matrix (tall)")
  ]

  for (rows, cols, desc) in matVecTests do
    IO.println s!"  Test: {desc}"

    -- Generate random matrix and vector
    let mut matrix : Array Int := #[]
    let mut vector : Array Int := #[]
    let mut seed := 54321

    for _ in [:rows * cols] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      matrix := matrix.push ((seed % PRIME) : Int)

    for _ in [:cols] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      vector := vector.push ((seed % PRIME) : Int)

    -- GPU timing
    let startGPU ← IO.monoNanosNow
    let gpuResult ← launchMatVec matrix vector rows cols PRIME
    let endGPU ← IO.monoNanosNow
    let gpuMs := (endGPU - startGPU).toFloat / 1000000.0

    -- CPU timing
    let startCPU ← IO.monoNanosNow
    let cpuResult := cpuMatVec matrix vector rows cols PRIME
    let endCPU ← IO.monoNanosNow
    let cpuMs := (endCPU - startCPU).toFloat / 1000000.0

    -- Verify correctness (check first few elements)
    let correct := gpuResult[0]? == cpuResult[0]? && gpuResult[1]? == cpuResult[1]?

    let speedup := cpuMs / gpuMs
    let winner := if speedup > 1.0 then "GPU" else "CPU"
    let speedupStr := if speedup > 1.0 then s!"{speedup.toString.take 5}x faster"
                      else s!"{(1.0/speedup).toString.take 5}x slower"

    IO.println s!"    CPU: {cpuMs.toString.take 8}ms | GPU: {gpuMs.toString.take 8}ms | {winner} {speedupStr} | Correct: {correct}"

  IO.println ""

  -- ═══════════════════════════════════════════════════════════════════════════
  -- Benchmark 3: Polynomial Evaluation at Many Points
  -- ═══════════════════════════════════════════════════════════════════════════
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "BENCHMARK 3: Polynomial Evaluation at Many Points"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "Evaluate polynomial of degree d at n points using Horner's method."
  IO.println "GPU excels with high-degree polynomials at many points."
  IO.println ""

  let polyTests := #[
    (10000,  100,  "10K points, degree 100"),
    (50000,  100,  "50K points, degree 100"),
    (100000, 100,  "100K points, degree 100"),
    (50000,  500,  "50K points, degree 500"),
    (50000,  1000, "50K points, degree 1000")
  ]

  for (npoints, degree, desc) in polyTests do
    IO.println s!"  Test: {desc}"

    -- Generate random coefficients and points
    let mut coeffs : Array Int := #[]
    let mut points : Array Int := #[]
    let mut seed := 98765

    for _ in [:degree + 1] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      coeffs := coeffs.push ((seed % PRIME) : Int)

    for _ in [:npoints] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      points := points.push ((seed % PRIME) : Int)

    -- GPU timing
    let startGPU ← IO.monoNanosNow
    let gpuResult ← launchPolyEval coeffs points PRIME
    let endGPU ← IO.monoNanosNow
    let gpuMs := (endGPU - startGPU).toFloat / 1000000.0

    -- CPU timing
    let startCPU ← IO.monoNanosNow
    let cpuResult := cpuPolyEval coeffs points PRIME
    let endCPU ← IO.monoNanosNow
    let cpuMs := (endCPU - startCPU).toFloat / 1000000.0

    -- Verify correctness
    let correct := gpuResult[0]? == cpuResult[0]?

    let speedup := cpuMs / gpuMs
    let winner := if speedup > 1.0 then "GPU" else "CPU"
    let speedupStr := if speedup > 1.0 then s!"{speedup.toString.take 5}x faster"
                      else s!"{(1.0/speedup).toString.take 5}x slower"

    IO.println s!"    CPU: {cpuMs.toString.take 8}ms | GPU: {gpuMs.toString.take 8}ms | {winner} {speedupStr} | Correct: {correct}"

  IO.println ""
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "SUMMARY"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "GPU shows significant speedup when:"
  IO.println "  • Per-element computation is high (many operations per data point)"
  IO.println "  • Data size is large enough for parallelism to dominate"
  IO.println "  • Operations are embarrassingly parallel"
  IO.println ""
  IO.println "Note: Current overhead includes process spawn and JSON serialization."
  IO.println "Production implementations with persistent CUDA contexts would show"
  IO.println "even greater GPU advantages."
  IO.println ""

#eval main

end CLean.GPUBenchmarks
