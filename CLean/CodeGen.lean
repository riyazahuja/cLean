import CLean.VerifyIR
import Lean

/-! # CUDA Code Generation

This module converts VerifyIR kernels into CUDA C++ source code.
The generated code can then be compiled and executed on GPU hardware.
-/

namespace CLean.CodeGen

open CLean.VerifyIR
open Lean (Name)

/-! ## Type Conversion -/

/-- Convert VType to CUDA C type string -/
def vtypeToCuda : VType → String
  | .float => "float"
  | .int => "int"
  | .nat => "unsigned int"
  | .bool => "bool"
  | .array t => vtypeToCuda t ++ "*"  -- Arrays become pointers in CUDA

/-! ## Expression Code Generation -/

/-- Convert VExpr to CUDA C expression string -/
partial def exprToCuda : VExpr → String
  | .threadIdX => "threadIdx.x"
  | .threadIdY => "threadIdx.y"
  | .threadIdZ => "threadIdx.z"
  | .blockIdX => "blockIdx.x"
  | .blockIdY => "blockIdx.y"
  | .blockIdZ => "blockIdx.z"
  | .blockDimX => "blockDim.x"
  | .blockDimY => "blockDim.y"
  | .blockDimZ => "blockDim.z"
  | .gridDimX => "gridDim.x"
  | .gridDimY => "gridDim.y"
  | .gridDimZ => "gridDim.z"
  | .constInt n => toString n
  | .constFloat f => toString f ++ "f"
  | .constBool b => if b then "true" else "false"
  | .var n => n.toString
  | .add a b => s!"({exprToCuda a} + {exprToCuda b})"
  | .sub a b => s!"({exprToCuda a} - {exprToCuda b})"
  | .mul a b => s!"({exprToCuda a} * {exprToCuda b})"
  | .div a b => s!"({exprToCuda a} / {exprToCuda b})"
  | .mod a b => s!"({exprToCuda a} % {exprToCuda b})"
  | .shl a b => s!"({exprToCuda a} << {exprToCuda b})"
  | .shr a b => s!"({exprToCuda a} >> {exprToCuda b})"
  | .band a b => s!"({exprToCuda a} & {exprToCuda b})"
  | .bor a b => s!"({exprToCuda a} | {exprToCuda b})"
  | .lt a b => s!"({exprToCuda a} < {exprToCuda b})"
  | .le a b => s!"({exprToCuda a} <= {exprToCuda b})"
  | .eq a b => s!"({exprToCuda a} == {exprToCuda b})"
  | .ne a b => s!"({exprToCuda a} != {exprToCuda b})"
  | .gt a b => s!"({exprToCuda a} > {exprToCuda b})"
  | .ge a b => s!"({exprToCuda a} >= {exprToCuda b})"
  | .land a b => s!"({exprToCuda a} && {exprToCuda b})"
  | .lor a b => s!"({exprToCuda a} || {exprToCuda b})"
  | .lnot a => s!"(!{exprToCuda a})"

/-! ## Statement Code Generation -/

/-- Get array name for memory location -/
def getArrayName (loc : MemLoc) : String :=
  loc.array.toString

/-- Convert memory space to CUDA qualifier -/
def memSpaceToQualifier : MemorySpace → String
  | .global => ""  -- Global arrays passed as parameters
  | .shared => "__shared__"
  | .local => ""   -- Local variables

/-- Generate code for a single statement -/
partial def stmtToCuda (indent : String := "") : VStmt → String
  | ⟨.read loc varName, _⟩ =>
    let arr := getArrayName loc
    let idx := exprToCuda loc.index
    s!"{indent}{varName} = {arr}[{idx}];\n"

  | ⟨.write loc value, _⟩ =>
    let arr := getArrayName loc
    let idx := exprToCuda loc.index
    let val := exprToCuda value
    s!"{indent}{arr}[{idx}] = {val};\n"

  | ⟨.assign varName value, _⟩ =>
    let val := exprToCuda value
    s!"{indent}{varName} = {val};\n"

  | ⟨.barrier, _⟩ =>
    s!"{indent}__syncthreads();\n"

  | ⟨.seq stmts, _⟩ =>
    stmts.foldl (fun acc s => acc ++ stmtToCuda indent s) ""

  | ⟨.ite cond thenStmts elseStmts, _⟩ =>
    let condStr := exprToCuda cond
    let thenBody := thenStmts.foldl (fun acc s => acc ++ stmtToCuda (indent ++ "  ") s) ""
    let elseBody := elseStmts.foldl (fun acc s => acc ++ stmtToCuda (indent ++ "  ") s) ""
    if elseBody.isEmpty then
      s!"{indent}if ({condStr}) \{\n{thenBody}{indent}}\n"
    else
      s!"{indent}if ({condStr}) \{\n{thenBody}{indent}} else \{\n{elseBody}{indent}}\n"

  | ⟨.whileLoop cond body, _⟩ =>
    let condStr := exprToCuda cond
    let bodyStr := body.foldl (fun acc s => acc ++ stmtToCuda (indent ++ "  ") s) ""
    s!"{indent}while ({condStr}) \{\n{bodyStr}{indent}}\n"

  | ⟨.forLoop varName start endExpr step body, _⟩ =>
    let startStr := exprToCuda start
    let endStr := exprToCuda endExpr
    let stepStr := exprToCuda step
    let bodyStr := body.foldl (fun acc s => acc ++ stmtToCuda (indent ++ "  ") s) ""
    s!"{indent}for (int {varName} = {startStr}; {varName} < {endStr}; {varName} += {stepStr}) \{\n{bodyStr}{indent}}\n"

/-! ## Kernel Code Generation -/

/-- Generate parameter declaration -/
def genParamDecl (v : VarInfo) : String :=
  let typeStr := vtypeToCuda v.type
  let qualifier := memSpaceToQualifier v.memorySpace
  if qualifier.isEmpty then
    s!"{typeStr} {v.name}"
  else
    s!"{qualifier} {typeStr} {v.name}"

/-- Generate variable declaration -/
def genVarDecl (v : VarInfo) : String :=
  let typeStr := vtypeToCuda v.type
  let qualifier := memSpaceToQualifier v.memorySpace
  if qualifier.isEmpty then
    s!"  {typeStr} {v.name};\n"
  else
    s!"  {qualifier} {typeStr} {v.name};\n"

/-- Generate complete CUDA kernel -/
def kernelToCuda (k : VKernel) : String :=
  -- Kernel signature
  let params := k.params ++ k.globalArrays
  let paramStrs := params.map genParamDecl
  let paramList := String.intercalate ", " paramStrs

  let signature := s!"__global__ void {k.name}({paramList})"

  -- Shared array declarations
  let sharedDecls := k.sharedArrays.foldl (fun acc v => acc ++ genVarDecl v) ""

  -- Local variable declarations
  let localDecls := k.locals.foldl (fun acc v => acc ++ genVarDecl v) ""

  -- Kernel body
  let bodyCode := k.body.foldl (fun acc s => acc ++ stmtToCuda "  " s) ""

  -- Combine
  s!"{signature} \{\n{sharedDecls}{localDecls}{bodyCode}}\n"

/-! ## Launch Configuration -/

structure LaunchConfig where
  gridDim : Nat × Nat × Nat
  blockDim : Nat × Nat × Nat
deriving Repr

/-- Generate kernel launch code (host-side) -/
def genLaunchCode (kernelName : String) (config : LaunchConfig) (args : List String) : String :=
  let gx := config.gridDim.1
  let gy := config.gridDim.2.1
  let gz := config.gridDim.2.2
  let bx := config.blockDim.1
  let by' := config.blockDim.2.1
  let bz := config.blockDim.2.2
  let argList := String.intercalate ", " args
  s!"dim3 gridDim({gx}, {gy}, {gz});\n" ++
  s!"dim3 blockDim({bx}, {by'}, {bz});\n" ++
  s!"{kernelName}<<<gridDim, blockDim>>>({argList});\n" ++
  s!"cudaDeviceSynchronize();\n"

/-! ## Full CUDA File Generation -/

/-- Generate complete CUDA file with kernel and helper code -/
def genCudaFile (kernels : List VKernel) (includes : List String := ["<cuda_runtime.h>"]) : String :=
  -- Includes
  let includeStr := includes.foldl (fun acc inc => acc ++ s!"#include {inc}\n") ""

  -- Kernels
  let kernelStr := kernels.foldl (fun acc k => acc ++ kernelToCuda k ++ "\n") ""

  s!"{includeStr}\n{kernelStr}"

end CLean.CodeGen
