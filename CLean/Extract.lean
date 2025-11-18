import Lean
import CLean.GPU
import CLean.VerifyIR

/-! # Kernel Extraction to Verification IR

This module implements extraction of KernelM computations into the VerifyIR representation.
It uses Lean metaprogramming to inspect elaborated terms and build the verification IR.

The extraction happens at elaboration time by walking the kernel body and converting:
- Memory operations (GlobalArray.get/set, SharedArray.get/set) → VStmt reads/writes
- barrier calls → VStmt barriers
- Control flow (if/while/for) → VStmt control flow nodes
- Expressions → VExpr symbolic expressions
-/

open Lean Lean.Elab Lean.Meta
open CLean.VerifyIR
open GpuDSL

namespace CLean.Extract

/-! ## Extraction State -/

/-- State for tracking extracted information during kernel extraction -/
structure ExtractState where
  params : List VarInfo := []
  locals : List VarInfo := []
  globalArrays : List VarInfo := []
  sharedArrays : List VarInfo := []
  arrayNameMap : List (Name × Name) := []  -- Maps local variable names to array names
deriving Inhabited

abbrev ExtractM := StateRefT ExtractState MetaM

/-! ## Expression Extraction -/

/-- Extract a VExpr from a Lean expression.
    This converts Lean terms like `threadIdx.x`, arithmetic, etc. into symbolic VExpr. -/
partial def extractVExpr (e : Expr) : ExtractM VExpr := do
  -- Use reduce with all transparency to safely reduce without hitting loose bvars
  -- reduce is safer than whnf for expressions that might contain lambdas
  let e ← try
    reduce e (skipProofs := false) (skipTypes := false)
  catch _ =>
    pure e  -- If reduce fails, just use the expression as-is

  match e with
  -- Thread/block indices
  | Expr.const name _ =>
    let nameStr := name.toString
    if name == ``globalIdxX || (nameStr.splitOn ".").any (· == "globalIdxX") then
      return .add (.mul .blockIdX .blockDimX) .threadIdX
    else if name == ``globalIdxY || (nameStr.splitOn ".").any (· == "globalIdxY") then
      return .add (.mul .blockIdY .blockDimY) .threadIdY
    else if (nameStr.splitOn ".").any (· == "threadIdX") then
      return .threadIdX
    else if (nameStr.splitOn ".").any (· == "threadIdY") then
      return .threadIdY
    else if (nameStr.splitOn ".").any (· == "blockIdX") then
      return .blockIdX
    else if (nameStr.splitOn ".").any (· == "blockIdY") then
      return .blockIdY
    else if (nameStr.splitOn ".").any (· == "blockDimX") then
      return .blockDimX
    else if (nameStr.splitOn ".").any (· == "blockDimY") then
      return .blockDimY
    else
      return .var name

  -- Literals
  | Expr.lit (.natVal n) => return .constInt (Int.ofNat n)
  | Expr.lit (.strVal _) => return .var (`str)

  -- Variables
  | Expr.fvar id => do
    let decl ← id.getDecl
    return .var decl.userName

  -- Binary operations
  | Expr.app f arg => do
    let fname ← getFunctionName f
    match fname with
    | some n =>
      let nStr := n.toString
      -- Arithmetic operations
      if n == ``HAdd.hAdd || n == ``Add.add || (nStr.splitOn ".").any (· == "add") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .add left right
      else if n == ``HSub.hSub || n == ``Sub.sub || (nStr.splitOn ".").any (· == "sub") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .sub left right
      else if n == ``HMul.hMul || n == ``Mul.mul || (nStr.splitOn ".").any (· == "mul") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .mul left right
      else if n == ``HDiv.hDiv || n == ``Div.div || (nStr.splitOn ".").any (· == "div") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .div left right
      else if n == ``HMod.hMod || n == ``Mod.mod || (nStr.splitOn ".").any (· == "mod") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .mod left right
      -- Comparison operations
      else if n == ``LT.lt || (nStr.splitOn ".").any (· == "lt") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .lt left right
      else if n == ``LE.le || (nStr.splitOn ".").any (· == "le") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .le left right
      else if n == ``BEq.beq || n == ``Eq || (nStr.splitOn ".").any (fun s => s == "eq" || s == "beq") then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .eq left right
      -- Logical operations
      else if (nStr.splitOn ".").any (· == "and") || n == ``And then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .land left right
      else if (nStr.splitOn ".").any (· == "or") || n == ``Or then
        let left ← extractVExpr (getAppFn f |> getAppArgs |>.back? |>.getD e)
        let right ← extractVExpr arg
        return .lor left right
      else
        extractVExpr f
    | none => extractVExpr f

  | _ => return .var (`unknown)

where
  getFunctionName (e : Expr) : ExtractM (Option Name) := do
    -- Don't use whnf to avoid loose bvar errors
    match e with
    | Expr.const name _ => return some name
    | Expr.app f _ => getFunctionName f
    | _ => return none

  getAppFn (e : Expr) : Expr :=
    match e with
    | Expr.app f _ => getAppFn f
    | _ => e

  getAppArgs (e : Expr) : Array Expr :=
    let rec go (e : Expr) (args : Array Expr) : Array Expr :=
      match e with
      | Expr.app f arg => go f (args.push arg)
      | _ => args.reverse
    go e #[]

/-! ## Statement Extraction -/

/-- Try to extract array name from an expression like ⟨args.x⟩ or a local variable -/
partial def extractArrayName (e : Expr) : ExtractM Name := do
  -- Don't use whnf to avoid loose bvar errors
  match e with
  | Expr.fvar id => do
    let decl ← id.getDecl
    let localName := decl.userName
    -- Check if this local variable is mapped to an array name
    let state ← get
    match state.arrayNameMap.find? (·.1 == localName) with
    | some (_, arrayName) => return arrayName
    | none => return localName
  | Expr.proj _ _ inner => extractArrayName inner
  | Expr.app f _ => extractArrayName f
  | _ => return `unknownArray

/-- Extract memory read operation -/
partial def extractArrayGet (arrExpr : Expr) (idxExpr : Expr) (isShared : Bool) : ExtractM (Option VStmt) := do
  let idx ← extractVExpr idxExpr
  let arrName ← extractArrayName arrExpr
  let memSpace := if isShared then MemorySpace.shared else MemorySpace.global
  let loc : MemLoc := { array := arrName, index := idx, memorySpace := memSpace }
  return some { stmt := .read loc `result, predicate := .constBool true }

/-- Extract memory write operation -/
partial def extractArraySet (arrExpr : Expr) (idxExpr : Expr) (valExpr : Expr) (isShared : Bool) : ExtractM (Option VStmt) := do
  let idx ← extractVExpr idxExpr
  let val ← extractVExpr valExpr
  let arrName ← extractArrayName arrExpr
  let memSpace := if isShared then MemorySpace.shared else MemorySpace.global
  let loc : MemLoc := { array := arrName, index := idx, memorySpace := memSpace }
  return some { stmt := .write loc val, predicate := .constBool true }

/-- Extract a single statement -/
partial def extractStmt (e : Expr) : ExtractM (List VStmt) := do
  -- Use reduce instead of whnf
  let e ← try
    reduce e (skipProofs := false) (skipTypes := false)
  catch _ =>
    pure e

  match e with
  -- Barrier
  | Expr.const name _ =>
    if name == ``barrier || (name.toString.splitOn ".").any (· == "barrier") then
      return [{ stmt := .barrier, predicate := .constBool true }]
    else
      return []

  -- Application - check for memory operations
  | Expr.app _ _ => do
    let fn := e.getAppFn
    let args := e.getAppArgs

    match fn with
    | Expr.const name _ =>
      let nameStr := name.toString
      let parts := nameStr.splitOn "."
      -- Check for array.get
      if parts.any (· == "GlobalArray") && parts.any (· == "get") && args.size >= 2 then
        let arrExpr := args[args.size - 2]!
        let idxExpr := args[args.size - 1]!
        match ← extractArrayGet arrExpr idxExpr false with
        | some stmt => return [stmt]
        | none => return []
      else if parts.any (· == "SharedArray") && parts.any (· == "get") && args.size >= 2 then
        let arrExpr := args[args.size - 2]!
        let idxExpr := args[args.size - 1]!
        match ← extractArrayGet arrExpr idxExpr true with
        | some stmt => return [stmt]
        | none => return []
      -- Check for array.set
      else if parts.any (· == "GlobalArray") && parts.any (· == "set") && args.size >= 3 then
        let arrExpr := args[args.size - 3]!
        let idxExpr := args[args.size - 2]!
        let valExpr := args[args.size - 1]!
        match ← extractArraySet arrExpr idxExpr valExpr false with
        | some stmt => return [stmt]
        | none => return []
      else if parts.any (· == "SharedArray") && parts.any (· == "set") && args.size >= 3 then
        let arrExpr := args[args.size - 3]!
        let idxExpr := args[args.size - 2]!
        let valExpr := args[args.size - 1]!
        match ← extractArraySet arrExpr idxExpr valExpr true with
        | some stmt => return [stmt]
        | none => return []
      else
        return []
    | _ => return []

  | _ => return []

/-- Extract statements from KernelM body -/
partial def extractStmts (e : Expr) : ExtractM (List VStmt) := do
  -- Use reduce instead of whnf to safely reduce the expression
  let e ← try
    reduce e (skipProofs := false) (skipTypes := false)
  catch _ =>
    pure e

  match e with
  -- Pure return
  | Expr.app (Expr.app (Expr.const name _) _) value =>
    if name == ``pure || (name.toString.splitOn ".").any (· == "pure") then
      return []
    else
      extractStmt e

  -- Monadic bind: m >>= k
  | Expr.app (Expr.app (Expr.app (Expr.app (Expr.const bindName _) _) _) m) k =>
    if bindName == ``Bind.bind || (bindName.toString.splitOn ".").any (· == "bind") then
      -- Extract from m
      let mStmts ← extractStmts m

      -- Extract from continuation k
      -- k is a lambda: fun x => body
      match k with
      | Expr.lam varName _ body _ => do
        -- Add variable to locals if needed
        let bodyStmts ← extractStmts body
        return mStmts ++ bodyStmts
      | _ =>
        let kStmts ← extractStmts k
        return mStmts ++ kStmts
    else
      extractStmt e

  -- Let binding: let x := v; body
  | Expr.app (Expr.lam varName varType body _) value => do
    -- This is a let binding
    -- Don't call whnf on value to avoid loose bvars

    -- Check if this is an array reference creation: ⟨args.x⟩
    -- Check the type directly without whnf
    if varType.isAppOf ``GlobalArray || varType.isAppOf ``SharedArray then
      -- Track the mapping from local variable to array name
      match value with
      | Expr.proj _ _ inner =>
        match inner with
        | Expr.fvar id => do
          let decl ← id.getDecl
          if (decl.userName.toString.splitOn ".").any (· == "args") then
            -- This is accessing args.arrayName
            -- Extract the field name
            let arrayName := varName
            modify fun s => { s with arrayNameMap := (varName, arrayName) :: s.arrayNameMap }
        | _ => pure ()
      | _ => pure ()

    -- Process body
    extractStmts body

  -- If-then-else
  | Expr.app (Expr.app (Expr.app (Expr.app (Expr.app (Expr.const iteName _) _) _) cond) thenBranch) elseBranch =>
    let parts := iteName.toString.splitOn "."
    if iteName == ``ite || parts.any (· == "ite") || parts.any (· == "dite") then
      let condExpr ← extractVExpr cond
      let thenStmts ← extractStmts thenBranch
      let elseStmts ← extractStmts elseBranch
      return [{ stmt := .ite condExpr thenStmts elseStmts, predicate := .constBool true }]
    else
      extractStmt e

  -- Lambda abstraction
  | Expr.lam _ _ body _ => extractStmts body

  -- Try as a single statement
  | _ => extractStmt e

/-! ## Kernel Extraction -/

/-- Extract parameter information from kernel args type -/
def extractParamsFromArgsType (argsType : Expr) : MetaM (List VarInfo) := do
  -- For now, return empty - this would need to inspect the structure definition
  return []

/-- Extract a complete VKernel from a KernelM definition -/
def extractKernel (kernelName : Name) (kernelExpr : Expr) (argsType : Expr) : MetaM VKernel := do
  let initialState : ExtractState := {}
  let (body, finalState) ← (extractStmts kernelExpr).run initialState

  -- Extract parameter info
  let params ← extractParamsFromArgsType argsType

  return {
    name := kernelName
    params := finalState.params
    locals := finalState.locals
    globalArrays := finalState.globalArrays
    sharedArrays := finalState.sharedArrays
    body := body
  }

/-! ## Manual Kernel Construction (for testing) -/

/-- Manually extract a kernel given its definition -/
def manualExtractKernel (kernelDefName : Name) : MetaM VKernel := do
  let env ← getEnv

  -- Get the kernel definition
  let some info := env.find? kernelDefName
    | throwError s!"Kernel {kernelDefName} not found"

  -- Get the value
  let some value := info.value?
    | throwError s!"Kernel {kernelDefName} has no value"

  -- Use lambdaTelescope to properly instantiate bound variables
  -- This prevents "loose bvar" errors when calling whnf during extraction
  lambdaTelescope value fun xs body => do
    -- Now body has all lambda parameters instantiated as free variables in xs
    -- Safe to process without loose bvar errors
    let argsType := .const ``Unit []
    extractKernel kernelDefName body argsType

/-! ## Storage for Extracted Kernels -/

/-- Simple storage using association list for extracted kernels -/
structure ExtractedKernelState where
  kernels : List (Name × VKernel) := []
deriving Inhabited

/-- Environment extension for storing extracted kernels -/
initialize extractedKernelsExt : SimplePersistentEnvExtension (Name × VKernel) ExtractedKernelState ←
  registerSimplePersistentEnvExtension {
    addEntryFn := fun s (n, k) => { s with kernels := (n, k) :: s.kernels }
    addImportedFn := fun es => { kernels := es.foldl (fun acc arr => acc ++ arr.toList) [] }
  }

/-- Store an extracted kernel in the environment -/
def storeExtractedKernel (name : Name) (kernel : VKernel) : CoreM Unit := do
  modifyEnv fun env => extractedKernelsExt.addEntry env (name, kernel)

/-- Retrieve an extracted kernel from the environment -/
def getExtractedKernel (name : Name) : CoreM (Option VKernel) := do
  let env ← getEnv
  let state := extractedKernelsExt.getState env
  return state.kernels.find? (·.1 == name) |>.map (·.2)

end CLean.Extract
