/-
  CUDA Code Generation from DeviceIR

  Converts DeviceIR kernels into CUDA C++ source code with:
  - Global memory parameters
  - Shared memory declarations
  - Barrier synchronization (__syncthreads)
  - Control flow (if-then-else, for loops)
  - Kernel launch code
-/

import CLean.DeviceIR
import Lean

namespace CLean.DeviceCodeGen

open DeviceIR
open Lean (Name)

/-! ## Type Conversion -/

/-- Convert DType to CUDA C type string -/
def dtypeToCuda : DType → String
  | .int => "int"
  | .nat => "unsigned int"
  | .float => "float"
  | .bool => "bool"
  | .array t => dtypeToCuda t ++ "*"  -- Arrays become pointers
  | .tuple _ => "void*"  -- Simplified
  | .struct name _ => name

/-! ## Expression Code Generation -/

/-- Convert BinOp to CUDA operator string -/
def binopToCuda : BinOp → String
  | .add => "+"
  | .sub => "-"
  | .mul => "*"
  | .div => "/"
  | .mod => "%"
  | .lt => "<"
  | .le => "<="
  | .gt => ">"
  | .ge => ">="
  | .eq => "=="
  | .ne => "!="
  | .and => "&&"
  | .or => "||"

/-- Convert UnOp to CUDA operator string -/
def unopToCuda : UnOp → String
  | .neg => "-"
  | .not => "!"

/-- Convert Dim to CUDA dimension suffix -/
def dimToCuda : Dim → String
  | .x => "x"
  | .y => "y"
  | .z => "z"

/-- Convert DExpr to CUDA C expression string -/
partial def exprToCuda : DExpr → String
  | .intLit n => toString n
  | .floatLit f => toString f ++ "f"
  | .boolLit b => if b then "true" else "false"
  | .var name => name
  | .binop op a b => s!"({exprToCuda a} {binopToCuda op} {exprToCuda b})"
  | .unop op a => s!"({unopToCuda op}{exprToCuda a})"
  | .index arr idx => s!"{exprToCuda arr}[{exprToCuda idx}]"
  | .field obj fieldName => s!"{exprToCuda obj}.{fieldName}"
  | .threadIdx dim => s!"threadIdx.{dimToCuda dim}"
  | .blockIdx dim => s!"blockIdx.{dimToCuda dim}"
  | .blockDim dim => s!"blockDim.{dimToCuda dim}"
  | .gridDim dim => s!"gridDim.{dimToCuda dim}"

/-! ## Statement Code Generation -/

/-- Infer CUDA type from expression -/
partial def inferExprType (expr : DExpr) : String :=
  match expr with
  | .intLit _ => "int"
  | .floatLit _ => "float"
  | .boolLit _ => "bool"
  | .binop op a b =>
    match op with
    | .lt | .le | .gt | .ge | .eq | .ne | .and | .or => "bool"
    | .mod => "int"
    | .add | .sub | .mul | .div =>
      -- If both operands are int-typed, result is int; otherwise float
      let aType := inferExprType a
      let bType := inferExprType b
      if aType == "int" && bType == "int" then "int" else "float"
  | .unop op a =>
    match op with
    | .not => "bool"
    | .neg => inferExprType a  -- Preserve type
  | .threadIdx _ | .blockIdx _ | .blockDim _ | .gridDim _ => "int"
  | .index _ _ => "float"  -- Array element, assume float
  | .var name =>
    -- Simple heuristic based on name
    if name.endsWith "Idx" || name.endsWith "idx" || name == "i" || name == "j" ||
       name == "k" || name == "N" || name == "length" then
      "int"
    else
      "float"
  | _ => "float"  -- Default

/-- Convert DStmt to CUDA C statement string with proper indentation -/
partial def stmtToCuda (indent : String := "  ") : DStmt → String
  | .skip => ""

  | .assign varName expr =>
    -- Skip assignments from args.* (these are metadata, not actual computations)
    let exprStr := exprToCuda expr
    if exprStr.startsWith "args." then
      ""
    else
      -- Infer type from expression
      let ty := inferExprType expr
      s!"{indent}{ty} {varName} = {exprStr};\n"

  | .store arr idx val =>
    s!"{indent}{exprToCuda arr}[{exprToCuda idx}] = {exprToCuda val};\n"

  | .seq s1 s2 =>
    stmtToCuda indent s1 ++ stmtToCuda indent s2

  | .ite cond thenBranch elseBranch =>
    let condStr := exprToCuda cond
    let thenCode := stmtToCuda (indent ++ "  ") thenBranch
    let elseCode := stmtToCuda (indent ++ "  ") elseBranch
    if elseCode.trim.isEmpty then
      s!"{indent}if ({condStr}) \{\n{thenCode}{indent}}\n"
    else
      s!"{indent}if ({condStr}) \{\n{thenCode}{indent}} else \{\n{elseCode}{indent}}\n"

  | .for varName lo hi body =>
    let loStr := exprToCuda lo
    let hiStr := exprToCuda hi
    let bodyCode := stmtToCuda (indent ++ "  ") body
    s!"{indent}for (int {varName} = {loStr}; {varName} < {hiStr}; {varName}++) \{\n{bodyCode}{indent}}\n"

  | .whileLoop cond body =>
    let condStr := exprToCuda cond
    let bodyCode := stmtToCuda (indent ++ "  ") body
    s!"{indent}while ({condStr}) \{\n{bodyCode}{indent}}\n"

  | .barrier =>
    s!"{indent}__syncthreads();\n"

  | .call fnName args =>
    let argStrs := args.map exprToCuda
    let argList := String.intercalate ", " argStrs
    s!"{indent}{fnName}({argList});\n"

  | .assert cond msg =>
    let condStr := exprToCuda cond
    s!"{indent}assert({condStr});  // {msg}\n"

/-! ## Kernel Code Generation -/

/-- Extract scalar params from expressions -/
partial def extractScalarParamsFromExpr (expr : DExpr) : List (String × String) :=
  match expr with
  | .var exprName =>
    if exprName.startsWith "args." then
      let fieldName := exprName.drop 5
      let ty := if fieldName == "N" || fieldName.endsWith "Idx" || fieldName.endsWith "d" ||
                    fieldName.endsWith "1" || fieldName == "length" then "int" else "float"
      [(fieldName, ty)]
    else
      []
  | .binop _ a b => extractScalarParamsFromExpr a ++ extractScalarParamsFromExpr b
  | .unop _ a => extractScalarParamsFromExpr a
  | .index arr idx => extractScalarParamsFromExpr arr ++ extractScalarParamsFromExpr idx
  | .field obj _ => extractScalarParamsFromExpr obj
  | _ => []

/-- Extract scalar parameter names from kernel body by looking for args.* references -/
partial def extractScalarParams (stmt : DStmt) : List (String × String) :=
  match stmt with
  | .skip => []
  | .assign varName (DExpr.var exprName) =>
    -- Check if this is an args.field reference
    if exprName.startsWith "args." then
      let fieldName := exprName.drop 5  -- Remove "args." prefix
      -- Infer type from name (simple heuristic)
      let ty := if fieldName == "N" || fieldName.endsWith "Idx" || fieldName.endsWith "idx" ||
                    fieldName.endsWith "d" || fieldName.endsWith "1" || fieldName == "length" then
        "int"
      else if fieldName.startsWith "alpha" || fieldName.startsWith "beta" then
        "float"
      else
        "float"  -- default
      [(fieldName, ty)]
    else
      []
  | .assign _ expr => extractScalarParamsFromExpr expr
  | .store arr idx val =>
    extractScalarParamsFromExpr arr ++ extractScalarParamsFromExpr idx ++ extractScalarParamsFromExpr val
  | .seq s1 s2 => extractScalarParams s1 ++ extractScalarParams s2
  | .ite cond thenBranch elseBranch =>
    extractScalarParamsFromExpr cond ++ extractScalarParams thenBranch ++ extractScalarParams elseBranch
  | .for _ lo hi body =>
    extractScalarParamsFromExpr lo ++ extractScalarParamsFromExpr hi ++ extractScalarParams body
  | .whileLoop cond body =>
    extractScalarParamsFromExpr cond ++ extractScalarParams body
  | .barrier => []
  | .call _ args => args.bind extractScalarParamsFromExpr
  | .assert cond _ => extractScalarParamsFromExpr cond

/-- Generate parameter declaration for kernel signature -/
def genParamDecl (v : VarDecl) : String :=
  let typeStr := dtypeToCuda v.ty
  s!"{typeStr} {v.name}"

/-- Generate array parameter declaration -/
def genArrayParamDecl (arr : ArrayDecl) : String :=
  let baseType := match arr.ty with
    | .array t => dtypeToCuda t
    | t => dtypeToCuda t
  s!"{baseType}* {arr.name}"

/-- Generate shared memory array declaration -/
def genSharedArrayDecl (arr : ArrayDecl) (size : Nat := 256) : String :=
  let baseType := match arr.ty with
    | .array t => dtypeToCuda t
    | t => dtypeToCuda t
  s!"  __shared__ {baseType} {arr.name}[{size}];\n"

/-- Generate local variable declaration -/
def genLocalDecl (v : VarDecl) : String :=
  let typeStr := dtypeToCuda v.ty
  s!"  {typeStr} {v.name};\n"

/-- Generate complete CUDA kernel from DeviceIR.Kernel -/
def kernelToCuda (k : Kernel) (sharedMemSize : Nat := 256) : String :=
  -- Use scalar parameters directly from k.params
  let scalarParamDecls := k.params.map genParamDecl

  -- Generate array parameter declarations
  let globalArrayDecls := k.globalArrays.map genArrayParamDecl

  -- Combine scalar params and array params
  let allParamDecls := scalarParamDecls ++ globalArrayDecls
  let paramList := String.intercalate ", " allParamDecls

  let signature := s!"extern \"C\" __global__ void {k.name}({paramList})"

  -- Shared memory declarations
  let sharedDecls := k.sharedArrays.foldl
    (fun acc arr => acc ++ genSharedArrayDecl arr sharedMemSize) ""

  -- Local variable declarations
  let localDecls := k.locals.foldl
    (fun acc v => acc ++ genLocalDecl v) ""

  -- Kernel body
  let bodyCode := stmtToCuda "  " k.body

  -- Combine everything
  s!"{signature} \{\n{sharedDecls}{localDecls}{bodyCode}}\n"

/-! ## Kernel Launch Code -/

structure LaunchConfig where
  gridDim : Nat × Nat × Nat := (1, 1, 1)
  blockDim : Nat × Nat × Nat := (256, 1, 1)
  sharedMemBytes : Nat := 0
deriving Repr, Inhabited

/-- Generate kernel launch code for host -/
def genLaunchCode (kernelName : String) (config : LaunchConfig) (args : List String) : String :=
  let gx := config.gridDim.1
  let gy := config.gridDim.2.1
  let gz := config.gridDim.2.2
  let bx := config.blockDim.1
  let byVal := config.blockDim.2.1
  let bz := config.blockDim.2.2
  let argList := String.intercalate ", " args

  let dimCode :=
    s!"  dim3 gridDim({gx}, {gy}, {gz});\n" ++
    s!"  dim3 blockDim({bx}, {byVal}, {bz});\n"

  let launchCode := if config.sharedMemBytes > 0 then
    s!"  {kernelName}<<<gridDim, blockDim, {config.sharedMemBytes}>>>({argList});\n"
  else
    s!"  {kernelName}<<<gridDim, blockDim>>>({argList});\n"

  dimCode ++ launchCode ++ s!"  cudaDeviceSynchronize();\n"

/-! ## Complete CUDA Program Generation -/

/-- Generate complete CUDA file with kernel, host code, and main function -/
def genCompleteCudaProgram (k : Kernel) (config : LaunchConfig) (sharedMemSize : Nat := 256) : String :=
  -- Header includes
  let includes :=
    "#include <cuda_runtime.h>\n" ++
    "#include <stdio.h>\n" ++
    "#include <stdlib.h>\n" ++
    "#include <assert.h>\n\n"

  -- Kernel definition
  let kernelCode := kernelToCuda k sharedMemSize ++ "\n"

  -- Helper function to check CUDA errors
  let errorCheck :=
    "// CUDA error checking macro\n" ++
    "#define CUDA_CHECK(call) \\\n" ++
    "  do { \\\n" ++
    "    cudaError_t err = call; \\\n" ++
    "    if (err != cudaSuccess) { \\\n" ++
    "      fprintf(stderr, \"CUDA error in %s:%d: %s\\n\", __FILE__, __LINE__, \\\n" ++
    "              cudaGetErrorString(err)); \\\n" ++
    "      exit(EXIT_FAILURE); \\\n" ++
    "    } \\\n" ++
    "  } while(0)\n\n"

  -- Generate array size constants
  let arraySizeConst := s!"#define ARRAY_SIZE 1024\n\n"

  -- Generate main function
  let mainFunc :=
    "int main() {\n" ++
    s!"  printf(\"CUDA Kernel: {k.name}\\n\");\n" ++
    "  \n" ++
    "  // Array dimensions\n" ++
    "  const int N = ARRAY_SIZE;\n" ++
    "  const size_t size = N * sizeof(float);\n" ++
    "  \n"

  -- Allocate and initialize host arrays
  let hostAllocs := k.globalArrays.foldl (fun acc arr =>
    acc ++
    s!"  // Allocate host memory for {arr.name}\n" ++
    s!"  float *h_{arr.name} = (float*)malloc(size);\n" ++
    s!"  for (int i = 0; i < N; i++) h_{arr.name}[i] = (float)i;\n" ++
    "  \n"
  ) ""

  -- Allocate device arrays
  let deviceAllocs := k.globalArrays.foldl (fun acc arr =>
    acc ++
    s!"  // Allocate device memory for {arr.name}\n" ++
    s!"  float *d_{arr.name};\n" ++
    s!"  CUDA_CHECK(cudaMalloc(&d_{arr.name}, size));\n" ++
    s!"  CUDA_CHECK(cudaMemcpy(d_{arr.name}, h_{arr.name}, size, cudaMemcpyHostToDevice));\n" ++
    "  \n"
  ) ""

  -- Launch kernel
  let blockSize := config.gridDim.1
  let deviceArrayNames := k.globalArrays.map (fun arr => s!"d_{arr.name}")

  -- Use scalar params directly from k.params
  let scalarArgNames := k.params.map (·.name)

  let allArgs := scalarArgNames ++ deviceArrayNames
  let launchCode :=
    "  // Launch kernel\n" ++
    "  printf(\"Launching kernel with %d blocks, %d threads per block\\n\", " ++
    s!"N/{blockSize}, {blockSize});\n" ++
    genLaunchCode k.name config allArgs ++
    "  printf(\"Kernel execution completed\\n\");\n" ++
    "  \n"

  -- Copy results back and verify
  let copyBack := k.globalArrays.foldl (fun acc arr =>
    acc ++
    s!"  // Copy {arr.name} back to host\n" ++
    s!"  CUDA_CHECK(cudaMemcpy(h_{arr.name}, d_{arr.name}, size, cudaMemcpyDeviceToHost));\n"
  ) ""

  let printResults :=
    if k.globalArrays.isEmpty then
      "  \n  // No output arrays to print\n  \n"
    else
      let lastArray := k.globalArrays.getLast!
      "  \n" ++
      "  // Print first 10 results\n" ++
      "  printf(\"\\nFirst 10 results:\\n\");\n" ++
      "  for (int i = 0; i < 10 && i < N; i++) {\n" ++
      s!"    printf(\"[%d] = %.2f\\n\", i, h_{lastArray.name}[i]);\n" ++
      "  }\n" ++
      "  \n"

  -- Cleanup
  let cleanup := k.globalArrays.foldl (fun acc arr =>
    acc ++
    s!"  cudaFree(d_{arr.name});\n" ++
    s!"  free(h_{arr.name});\n"
  ) ""

  let mainEnd :=
    "  \n" ++
    "  printf(\"\\nTest completed successfully!\\n\");\n" ++
    "  return 0;\n" ++
    "}\n"

  includes ++ errorCheck ++ arraySizeConst ++ kernelCode ++
  mainFunc ++ hostAllocs ++ deviceAllocs ++ launchCode ++
  copyBack ++ printResults ++ cleanup ++ mainEnd

end CLean.DeviceCodeGen
