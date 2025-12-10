/-
  GPU-Accelerated Gröbner Basis Computation

  Uses the cLean GPU infrastructure to perform parallel Gaussian elimination.
-/

import CLean.Groebner
import CLean.FiniteField
import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher

namespace CLean.GroebnerGPU

open Groebner
open FiniteField
open Polynomial
open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen CLean.GPU.ProcessLauncher

set_option maxHeartbeats 400000

/-! ## GPU Gaussian Elimination

The key insight: In Gaussian elimination, once a pivot is selected,
eliminating the pivot column from all other rows is embarrassingly parallel.

Each GPU thread eliminates one element of a row during elimination.
-/

/-! ## GPU Kernel for Row Elimination

Arguments:
- nrows, ncols: matrix dimensions
- pivotRowIdx: which row is the pivot
- mat: the flattened matrix (row-major, int values in Fp)
- pivotRow: the current pivot row data
- factors: elimination factors for each row

Each thread processes one (row, col) position.
-/
kernelArgs GaussElimArgs(nrows: Nat, ncols: Nat, pivotRowIdx: Nat, p: Nat)
  global[mat pivotRow factors: Array Int]

-- Simplified GPU kernel for row elimination.
-- Each thread handles one row, eliminating using the pivot row.
device_kernel gaussElimKernel : KernelM GaussElimArgs Unit := do
  let args ← getArgs
  let nrows := args.nrows
  let ncols := args.ncols
  let pivotRowIdx := args.pivotRowIdx
  let p := args.p  -- prime modulus (65521) - used for modular arithmetic
  let mat : GlobalArray Int := ⟨args.mat⟩
  let pivotRow : GlobalArray Int := ⟨args.pivotRow⟩
  let factors : GlobalArray Int := ⟨args.factors⟩

  -- Each thread handles one row
  let row ← globalIdxX
  if row < nrows then do
    if row < pivotRowIdx || row > pivotRowIdx then do  -- row != pivotRowIdx
      let factor ← factors.get row
      if factor > 0 then do
        -- Process all columns for this row
        for col in [:ncols] do
          let idx := row * ncols + col
          let matVal ← mat.get idx
          let pivotVal ← pivotRow.get col
          -- Modular multiplication and subtraction with explicit mod reduction
          let prod := (factor * pivotVal) % p
          let diff := ((matVal - prod) % p + p) % p
          mat.set idx diff

-- Show generated IR and CUDA code
#eval do
  IO.println "=== Generated Kernel IR ==="
  IO.println (repr gaussElimKernelIR)
  IO.println ""
  IO.println "=== Generated CUDA Kernel ==="
  IO.println (kernelToCuda gaussElimKernelIR)

/-! ## GPU Launch Infrastructure -/

/-- Check if GPU is available by checking for nvcc -/
def hasGPU : IO Bool := do
  let result ← IO.Process.output {
    cmd := "which"
    args := #["nvcc"]
  }
  pure (result.exitCode == 0)

/-- Launch the Gaussian elimination kernel on GPU for one pivot step.
    Returns the updated matrix after eliminating one column. -/
def launchGaussElimKernel (mat : DenseMatrix) (pivotRowIdx : Nat) (pivotRow : Array Int)
    (factors : Array Int) : IO DenseMatrix := do
  -- Convert matrix to flat Int array
  let matFlat : Array Int := mat.data.map fun x => Int.ofNat x.toNat

  -- Compile kernel to PTX
  let cached ← compileKernelToPTX gaussElimKernelIR

  -- Build JSON input with proper int types
  -- Use typed scalars and typed int arrays
  let scalarParams : Array ScalarValue := #[
    ScalarValue.int mat.rows,
    ScalarValue.int mat.cols,
    ScalarValue.int pivotRowIdx,
    ScalarValue.int PRIME.toNat
  ]
  let arrays : List (Lean.Name × Array Int) := [
    (`mat, matFlat),
    (`pivotRow, pivotRow),
    (`factors, factors)
  ]
  let jsonInput := buildLauncherInputBetter scalarParams arrays

  -- Launch kernel
  let grid : Dim3 := ⟨(mat.rows + 255) / 256, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    gaussElimKernelIR.name,
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
  let stderr ← child.stderr.readToEnd
  let exitCode ← child.wait

  if exitCode != 0 then
    IO.eprintln s!"GPU stderr: {stderr}"
    throw <| IO.userError s!"GPU kernel launch failed with exit code {exitCode}"

  -- Debug: print output format
  -- IO.println s!"GPU stdout (first 500 chars): {stdout.take 500}"

  -- Parse output and reconstruct matrix
  match Lean.Json.parse stdout with
  | Except.error err => throw <| IO.userError s!"JSON parse error: {err}\nOutput: {stdout.take 500}"
  | Except.ok json =>
    -- Output format is {"results": {"mat": [...], ...}}
    match json.getObjVal? "results" with
    | Except.error _ => throw <| IO.userError s!"Missing 'results' in output. JSON: {json.compress.take 500}"
    | Except.ok results =>
      match results.getObjVal? "mat" with
      | Except.error _ => throw <| IO.userError s!"Missing 'mat' in results. JSON: {results.compress.take 500}"
      | Except.ok matJson =>
        match matJson.getArr? with
        | Except.error _ => throw <| IO.userError "mat is not an array"
        | Except.ok arr =>
          let p : Int := PRIME.toNat
          let newData : Array FpElem := arr.map fun v =>
            match v.getInt? with
            | Except.ok i =>
              -- Handle potential negative values from modular arithmetic
              let normalized := if i < 0 then (i % p + p) % p else i % p
              normalized.toNat.toUInt32
            | Except.error _ => 0
          pure { mat with data := newData }

/-- GPU Gaussian elimination.
    The algorithm:
    1. For each column (pivot selection done on CPU)
    2. Scale pivot row (CPU)
    3. Compute elimination factors (CPU)
    4. Parallel row elimination (GPU if available, CPU otherwise)
-/
def gaussianEliminationGPU (m : DenseMatrix) : IO DenseMatrix := do
  let gpuAvailable ← hasGPU
  let useGPU := gpuAvailable && m.rows * m.cols >= 10000  -- Only use GPU for large matrices

  let mut mat := m
  let mut pivotRowIdx := 0
  let _ncols := mat.cols
  let _nrows := mat.rows

  for col in [:mat.cols] do
    if pivotRowIdx >= mat.rows then break

    -- Find pivot (CPU)
    let mut found := false
    let mut pivotIdx := pivotRowIdx
    for row in [pivotRowIdx:mat.rows] do
      if mat.get row col != 0 then
        pivotIdx := row
        found := true
        break

    if found then
      -- Swap rows if needed (CPU)
      if pivotIdx != pivotRowIdx then
        for c in [:mat.cols] do
          let tmp := mat.get pivotRowIdx c
          mat := mat.set pivotRowIdx c (mat.get pivotIdx c)
          mat := mat.set pivotIdx c tmp

      -- Scale pivot row to make leading coefficient 1 (CPU)
      let pivotVal := mat.get pivotRowIdx col
      let pivotInv := inv pivotVal
      for c in [:mat.cols] do
        mat := mat.set pivotRowIdx c (mul (mat.get pivotRowIdx c) pivotInv)

      if useGPU then
        -- GPU path: launch kernel for row elimination
        let pivotRow : Array Int := Array.range mat.cols |>.map fun c =>
          Int.ofNat (mat.get pivotRowIdx c).toNat
        let factors : Array Int := Array.range mat.rows |>.map fun r =>
          Int.ofNat (mat.get r col).toNat
        mat ← launchGaussElimKernel mat pivotRowIdx pivotRow factors
      else
        -- CPU path: eliminate column in other rows
        for row in [:mat.rows] do
          if row != pivotRowIdx then
            let factor := mat.get row col
            if factor != 0 then
              for c in [:mat.cols] do
                let newVal := sub (mat.get row c) (mul factor (mat.get pivotRowIdx c))
                mat := mat.set row c newVal

      pivotRowIdx := pivotRowIdx + 1

  pure mat

/-- F4-style reduction using GPU-accelerated elimination -/
def f4ReductionGPU (polys : Array Poly) : IO (Array Poly) := do
  if polys.isEmpty then return #[]

  let (mat, monos) := buildDenseMatrix polys
  let reduced ← gaussianEliminationGPU mat
  let numVars := polys[0]!.numVars
  pure (extractPolys reduced monos numVars)

/-! ## Benchmark: CPU vs GPU -/

/-- Pure CPU benchmark -/
def benchmarkGaussElimCPU (size : Nat) : IO Unit := do
  IO.println s!"Benchmarking {size}x{size} matrix..."

  -- Create random matrix
  let mut mat := DenseMatrix.create size size
  let mut seed := 42
  for i in [:size] do
    for j in [:size] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      mat := mat.set i j ((seed % PRIME.toNat).toUInt32)

  -- Benchmark CPU
  let startCPU ← IO.monoNanosNow
  let _ := gaussianEliminationCPU mat
  let endCPU ← IO.monoNanosNow
  let cpuUs := (endCPU - startCPU) / 1000

  IO.println s!"  CPU: {cpuUs}μs"

/-- Benchmark Gaussian elimination (both CPU and simulated GPU) -/
def benchmarkGaussElim (size : Nat) : IO Unit := do
  IO.println s!"Benchmarking {size}x{size} matrix..."

  -- Create random matrix
  let mut mat := DenseMatrix.create size size
  let mut seed := 42
  for i in [:size] do
    for j in [:size] do
      seed := (seed * 1103515245 + 12345) % (2^31)
      mat := mat.set i j ((seed % PRIME.toNat).toUInt32)

  -- Benchmark CPU
  let startCPU ← IO.monoNanosNow
  let _ := gaussianEliminationCPU mat
  let endCPU ← IO.monoNanosNow
  let cpuMs := (endCPU - startCPU) / 1000000

  -- Benchmark GPU (simulated for now)
  let startGPU ← IO.monoNanosNow
  let _ ← gaussianEliminationGPU mat
  let endGPU ← IO.monoNanosNow
  let gpuMs := (endGPU - startGPU) / 1000000

  IO.println s!"  CPU: {cpuMs}ms"
  IO.println s!"  GPU (simulated): {gpuMs}ms"

/-! ## Example: Ideal Membership -/

/-- Main test function -/
def main : IO Unit := do
  IO.println "=== Gröbner Basis Computation ==="
  IO.println ""

  -- Test 1: Simple ideal membership
  IO.println "Test 1: Ideal Membership"
  IO.println "------------------------"

  let numVars := 2

  -- Generators: x - 2, y - 3
  let g1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },
      { coeff := neg 2, mono := #[0, 0] }
    ]
    numVars := numVars
  }
  let g2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[0, 1] },
      { coeff := neg 3, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  -- f = x + y - 5 (should be in ideal)
  let f1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },
      { coeff := 1, mono := #[0, 1] },
      { coeff := neg 5, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  -- f = x*y - 6 (should be in ideal since 2*3 = 6)
  let f2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 1] },      -- xy
      { coeff := neg 6, mono := #[0, 0] }   -- -6
    ]
    numVars := numVars
  }

  -- f = x^2 + y^2 - 13 (should be in ideal since 4 + 9 = 13)
  let f3 : Poly := {
    terms := #[
      { coeff := 1, mono := #[2, 0] },      -- x²
      { coeff := 1, mono := #[0, 2] },      -- y²
      { coeff := neg 13, mono := #[0, 0] }  -- -13
    ]
    numVars := numVars
  }

  -- f = x + y - 6 (should NOT be in ideal since 2 + 3 ≠ 6)
  let f4 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },
      { coeff := 1, mono := #[0, 1] },
      { coeff := neg 6, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  let gens := #[g1, g2]
  IO.println s!"Generators: x - 2, y - 3"
  IO.println s!"(Defines the point (2, 3) in ℤ_p)"
  IO.println ""

  let result1 := idealMembershipTest f1 gens
  IO.println s!"x + y - 5 in ideal: {result1} (expected: true)"

  let result2 := idealMembershipTest f2 gens
  IO.println s!"xy - 6 in ideal: {result2} (expected: true)"

  let result3 := idealMembershipTest f3 gens
  IO.println s!"x² + y² - 13 in ideal: {result3} (expected: true)"

  let result4 := idealMembershipTest f4 gens
  IO.println s!"x + y - 6 in ideal: {result4} (expected: false)"

  IO.println ""

  -- Test 2: More complex ideal - cyclic-2
  IO.println "Test 2: Cyclic-2 Ideal"
  IO.println "----------------------"

  -- I = <x + y - s, xy - p> where s, p are "sum" and "product"
  -- At (2, 3): sum = 5, product = 6
  let cycG1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },  -- x
      { coeff := 1, mono := #[0, 1] },  -- y
      { coeff := neg 5, mono := #[0, 0] }  -- -5
    ]
    numVars := numVars
  }
  let cycG2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 1] },     -- xy
      { coeff := neg 6, mono := #[0, 0] }  -- -6
    ]
    numVars := numVars
  }

  let cycGens := #[cycG1, cycG2]
  let gb := buchberger cycGens
  IO.println s!"Gröbner basis computed with {gb.size} elements"

  -- x² - 5x + 6 should be in ideal (factored form: (x-2)(x-3))
  let testPoly : Poly := {
    terms := #[
      { coeff := 1, mono := #[2, 0] },       -- x²
      { coeff := neg 5, mono := #[1, 0] },   -- -5x
      { coeff := 6, mono := #[0, 0] }        -- +6
    ]
    numVars := numVars
  }
  let cycResult := idealMembershipTest testPoly cycGens
  IO.println s!"x² - 5x + 6 in ideal: {cycResult} (expected: true)"

  IO.println ""

  -- Test 3: 3-variable example
  IO.println "Test 3: Three-Variable Ideal"
  IO.println "-----------------------------"
  let numVars3 := 3

  -- Ideal defining point (1, 2, 3): <x-1, y-2, z-3>
  let h1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0, 0] },
      { coeff := neg 1, mono := #[0, 0, 0] }
    ]
    numVars := numVars3
  }
  let h2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[0, 1, 0] },
      { coeff := neg 2, mono := #[0, 0, 0] }
    ]
    numVars := numVars3
  }
  let h3 : Poly := {
    terms := #[
      { coeff := 1, mono := #[0, 0, 1] },
      { coeff := neg 3, mono := #[0, 0, 0] }
    ]
    numVars := numVars3
  }

  -- x + y + z - 6 should be in ideal (1 + 2 + 3 = 6)
  let sum6 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0, 0] },
      { coeff := 1, mono := #[0, 1, 0] },
      { coeff := 1, mono := #[0, 0, 1] },
      { coeff := neg 6, mono := #[0, 0, 0] }
    ]
    numVars := numVars3
  }

  -- xyz - 6 should be in ideal (1 * 2 * 3 = 6)
  let prod6 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 1, 1] },
      { coeff := neg 6, mono := #[0, 0, 0] }
    ]
    numVars := numVars3
  }

  let gens3 := #[h1, h2, h3]
  let r1 := idealMembershipTest sum6 gens3
  IO.println s!"x + y + z - 6 in <x-1, y-2, z-3>: {r1} (expected: true)"

  let r2 := idealMembershipTest prod6 gens3
  IO.println s!"xyz - 6 in <x-1, y-2, z-3>: {r2} (expected: true)"

  IO.println ""

  -- Test 4: Matrix elimination benchmark (CPU only for now)
  IO.println "Test 4: Gaussian Elimination Benchmark (CPU)"
  IO.println "---------------------------------------------"
  benchmarkGaussElimCPU 50
  benchmarkGaussElimCPU 100
  benchmarkGaussElimCPU 200
  benchmarkGaussElimCPU 500

  IO.println ""

  -- Test 5: Check GPU availability and run GPU test
  IO.println "Test 5: GPU Availability & Test"
  IO.println "-------------------------------"
  let gpuAvail ← hasGPU
  IO.println s!"CUDA (nvcc) available: {gpuAvail}"

  if gpuAvail then
    IO.println "Running GPU Gaussian elimination test..."

    -- Test with small, predictable matrix first
    IO.println ""
    IO.println "Small matrix debug test (3×3):"
    let smallMat := DenseMatrix.create 3 3
      |>.set 0 0 1 |>.set 0 1 2 |>.set 0 2 3
      |>.set 1 0 4 |>.set 1 1 5 |>.set 1 2 6
      |>.set 2 0 7 |>.set 2 1 8 |>.set 2 2 9

    IO.println s!"  Input matrix:"
    for i in [:3] do
      let row := (List.range 3).map (fun j => smallMat.get i j) |>.toArray
      IO.println s!"    Row {i}: {row.toList}"

    let smallPivotRowIdx : Nat := 0
    let smallPivotRow : Array Int := (List.range 3).map (fun j => Int.ofNat (smallMat.get smallPivotRowIdx j).toNat) |>.toArray
    let smallFactors : Array Int := (List.range 3).map (fun i =>
      if i == smallPivotRowIdx then 0
      else Int.ofNat (smallMat.get i smallPivotRowIdx).toNat) |>.toArray

    IO.println s!"  Pivot row (row 0): {smallPivotRow.toList}"
    IO.println s!"  Factors: {smallFactors.toList}"

    -- Expected: Row 1: [4,5,6] - 4*[1,2,3] = [0, -3, -6] = [0, 65518, 65515] mod 65521
    -- Expected: Row 2: [7,8,9] - 7*[1,2,3] = [0, -6, -12] = [0, 65515, 65509] mod 65521

    let smallGpuResult ← launchGaussElimKernel smallMat smallPivotRowIdx smallPivotRow smallFactors
    IO.println s!"  GPU Result:"
    for i in [:3] do
      let row := (List.range 3).map (fun j => smallGpuResult.get i j) |>.toArray
      IO.println s!"    Row {i}: {row.toList}"

    -- CPU version
    let mut smallCpuMat := smallMat
    for row in [:3] do
      if row != smallPivotRowIdx then
        let factor := smallCpuMat.get row smallPivotRowIdx
        if factor != 0 then
          for col in [:3] do
            let pivotVal := smallCpuMat.get smallPivotRowIdx col
            let oldVal := smallCpuMat.get row col
            let newVal := FiniteField.sub oldVal (FiniteField.mul factor pivotVal)
            smallCpuMat := smallCpuMat.set row col newVal

    IO.println s!"  CPU Result:"
    for i in [:3] do
      let row := (List.range 3).map (fun j => smallCpuMat.get i j) |>.toArray
      IO.println s!"    Row {i}: {row.toList}"

    -- Now test single kernel launch with larger matrix
    IO.println ""
    IO.println "Single kernel launch benchmark (row elimination step):"

    -- Create a test matrix
    let testSize := 500  -- 500x500 matrix
    let mut testMat := DenseMatrix.create testSize testSize
    let mut seed := 12345
    for i in [:testSize] do
      for j in [:testSize] do
        seed := (seed * 1103515245 + 12345) % (2^31)
        testMat := testMat.set i j ((seed % PRIME.toNat).toUInt32)

    IO.println s!"  Created {testSize}×{testSize} random matrix"

    -- Prepare data for a single kernel launch
    let pivotRowIdx : Nat := 0
    let pivotRow : Array Int := (List.range testSize).map (fun j => Int.ofNat (testMat.get pivotRowIdx j).toNat) |>.toArray
    let factors : Array Int := (List.range testSize).map (fun i =>
      if i == pivotRowIdx then 0
      else Int.ofNat (testMat.get i pivotRowIdx).toNat) |>.toArray

    -- Single GPU kernel launch
    let startGPU ← IO.monoNanosNow
    let gpuResult ← launchGaussElimKernel testMat pivotRowIdx pivotRow factors
    let endGPU ← IO.monoNanosNow
    let gpuUs := (endGPU - startGPU) / 1000

    IO.println s!"  Single GPU kernel launch: {gpuUs}μs"
    IO.println s!"    (includes: JSON serialize, process spawn, kernel launch, copy back, JSON parse)"

    -- Single CPU elimination step for comparison
    let startCPU ← IO.monoNanosNow
    let mut cpuMat := testMat
    for row in [:testSize] do
      if row != pivotRowIdx then
        let factor := cpuMat.get row pivotRowIdx
        if factor != 0 then
          for col in [:testSize] do
            let pivotVal := cpuMat.get pivotRowIdx col
            let oldVal := cpuMat.get row col
            let newVal := FiniteField.sub oldVal (FiniteField.mul factor pivotVal)
            cpuMat := cpuMat.set row col newVal
    let endCPU ← IO.monoNanosNow
    let cpuUs := (endCPU - startCPU) / 1000

    IO.println s!"  Single CPU elimination step: {cpuUs}μs"

    -- Check if results match (spot check)
    let mut resultsMatch := true
    let mut diffCount := 0
    for i in [:min 10 testSize] do
      for j in [:min 10 testSize] do
        if gpuResult.get i j != cpuMat.get i j then
          if diffCount < 5 then
            IO.println s!"    Diff at ({i},{j}): GPU={gpuResult.get i j}, CPU={cpuMat.get i j}"
          diffCount := diffCount + 1
          resultsMatch := false
    if resultsMatch then
      IO.println "  ✓ GPU and CPU results match (spot check)"
    else
      IO.println s!"  ✗ GPU and CPU results differ ({diffCount} differences in spot check)"

    -- Now test full Gaussian elimination on smaller matrix
    IO.println ""
    IO.println "Full Gaussian elimination benchmark (150×150 matrix):"
    let fullTestSize := 150
    let mut fullMat := DenseMatrix.create fullTestSize fullTestSize
    seed := 54321
    for i in [:fullTestSize] do
      for j in [:fullTestSize] do
        seed := (seed * 1103515245 + 12345) % (2^31)
        fullMat := fullMat.set i j ((seed % PRIME.toNat).toUInt32)

    let startFullCPU ← IO.monoNanosNow
    let _cpuFullResult := gaussianEliminationCPU fullMat
    let endFullCPU ← IO.monoNanosNow
    let cpuFullMs := (endFullCPU - startFullCPU) / 1000000

    IO.println s!"  CPU full elimination: {cpuFullMs}ms"
    IO.println "  (GPU full elimination skipped - too many kernel launches)"
    IO.println "  (For production use, the kernel should process multiple columns per launch)"

  else
    IO.println "GPU not available, skipping GPU test"

  IO.println ""
  IO.println "Done!"

#eval main

end CLean.GroebnerGPU
