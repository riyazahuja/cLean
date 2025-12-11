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
