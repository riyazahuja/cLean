import CLean.GPU
import CLean.VerifyIR
import CLean.Extract
import CLean.CodeGen

/-! # Kernel Extraction Test

This file tests the extraction of KernelM kernels to VerifyIR and their CUDA code generation.
-/

open CLean.Extract CLean.VerifyIR CLean.CodeGen
open Lean Lean.Elab Lean.Meta GpuDSL

/-! ## Test Kernel Definition -/

namespace TestKernels

kernelArgs SimpleSaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

def simpleSaxpyKernel : KernelM SimpleSaxpyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩

  let i ← globalIdxX
  if i < N then
    let xi ← x.get i
    let yi ← y.get i
    r.set i (alpha * xi + yi)

end TestKernels

/-! ## Helper Functions for Display -/

def repeatChar (c : Char) (n : Nat) : String :=
  String.mk (List.replicate n c)

def printSeparator (title : String) : IO Unit := do
  IO.println ""
  IO.println (repeatChar '=' 70)
  IO.println s!"  {title}"
  IO.println (repeatChar '=' 70)

def printVKernelInfo (k : VKernel) : IO Unit := do
  IO.println s!"Kernel Name: {k.name}"
  IO.println s!"Parameters: {k.params.length}"
  IO.println s!"Global Arrays: {k.globalArrays.length}"
  IO.println s!"Shared Arrays: {k.sharedArrays.length}"
  IO.println s!"Local Variables: {k.locals.length}"
  IO.println s!"Body Statements: {k.body.length}"

def printVExpr (e : VExpr) : String :=
  match e with
  | .threadIdX => "threadIdx.x"
  | .threadIdY => "threadIdx.y"
  | .blockIdX => "blockIdx.x"
  | .blockIdY => "blockIdx.y"
  | .blockDimX => "blockDim.x"
  | .blockDimY => "blockDim.y"
  | .gridDimX => "gridDim.x"
  | .gridDimY => "gridDim.y"
  | .constInt n => s!"{n}"
  | .var n => s!"{n}"
  | .add a b => s!"({printVExpr a} + {printVExpr b})"
  | .sub a b => s!"({printVExpr a} - {printVExpr b})"
  | .mul a b => s!"({printVExpr a} * {printVExpr b})"
  | .div a b => s!"({printVExpr a} / {printVExpr b})"
  | .mod a b => s!"({printVExpr a} % {printVExpr b})"
  | .lt a b => s!"({printVExpr a} < {printVExpr b})"
  | .le a b => s!"({printVExpr a} <= {printVExpr b})"
  | .eq a b => s!"({printVExpr a} == {printVExpr b})"
  | .land a b => s!"({printVExpr a} && {printVExpr b})"
  | .lor a b => s!"({printVExpr a} || {printVExpr b})"
  | .lnot a => s!"(!{printVExpr a})"
  | _ => "<?>"

def printMemorySpace : MemorySpace → String
  | .global => "global"
  | .shared => "shared"
  | .local => "local"

def printVStmt (indent : String := "") : VStmt → String
  | ⟨.read loc varName, _⟩ =>
    s!"{indent}READ {varName} = {loc.array}[{printVExpr loc.index}] ({printMemorySpace loc.memorySpace})\n"
  | ⟨.write loc value, _⟩ =>
    s!"{indent}WRITE {loc.array}[{printVExpr loc.index}] = {printVExpr value} ({printMemorySpace loc.memorySpace})\n"
  | ⟨.assign varName value, _⟩ =>
    s!"{indent}ASSIGN {varName} = {printVExpr value}\n"
  | ⟨.barrier, _⟩ =>
    s!"{indent}BARRIER\n"
  | ⟨.ite cond thenStmts elseStmts, _⟩ =>
    let thenStr := thenStmts.foldl (fun acc s => acc ++ printVStmt (indent ++ "  ") s) ""
    let elseStr := elseStmts.foldl (fun acc s => acc ++ printVStmt (indent ++ "  ") s) ""
    s!"{indent}IF {printVExpr cond} THEN\n{thenStr}{indent}ELSE\n{elseStr}{indent}END IF\n"
  | ⟨.seq stmts, _⟩ =>
    stmts.foldl (fun acc s => acc ++ printVStmt indent s) ""
  | _ => s!"{indent}<stmt>\n"

def printVKernelBody (stmts : List VStmt) : String :=
  stmts.foldl (fun acc s => acc ++ printVStmt "  " s) ""

/-! ## Extraction Tests -/

/-- Try to extract and display a kernel -/
def testExtraction (kernelName : Name) : MetaM Unit := do
  try
    IO.println s!"Attempting to extract kernel: {kernelName}"
    let vkernel ← manualExtractKernel kernelName

    IO.println "✓ Extraction succeeded!"
    IO.println ""
    printVKernelInfo vkernel

    IO.println ""
    IO.println "Extracted VerifyIR Body:"
    IO.println (repeatChar '─' 70)
    IO.println (printVKernelBody vkernel.body)

    IO.println ""
    IO.println "Generated CUDA Code:"
    IO.println (repeatChar '─' 70)
    let cudaCode := kernelToCuda vkernel
    IO.println cudaCode
    IO.println (repeatChar '─' 70)
  catch e =>
    IO.println s!"✗ Extraction failed: {← e.toMessageData.toString}"

/-- Main test function -/
def runExtractionTests : MetaM Unit := do
  printSeparator "Kernel Extraction Test Suite"
  IO.println "Testing KernelM → VerifyIR → CUDA pipeline"
  IO.println ""

  -- Test simple SAXPY kernel
  printSeparator "Test 1: Simple SAXPY Kernel"
  testExtraction `TestKernels.simpleSaxpyKernel

  printSeparator "All Extraction Tests Complete"

#eval! do
  runExtractionTests
