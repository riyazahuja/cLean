/-
  Device Extractor

  Extracts DeviceIR from existing KernelM definitions.

  Strategy:
  - Hard-coded pattern matching for GPU operations (globalIdxX, GlobalArray.get/set, etc.)
  - Typeclass-based translation for user code
  - Recursive translation of helper functions
-/

import Lean
import Std.Data.HashMap
import CLean.DeviceIR
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.GPU

open Lean Meta Elab Command
open DeviceIR DeviceTranslation
open Std (HashMap)

namespace DeviceExtractor

-- State for extraction
structure ExtractState where
  /-- Next fresh variable ID -/
  nextVar : Nat := 0
  /-- Accumulated statements -/
  stmts : Array DStmt := #[]
  /-- Local variables (FVarId → device var name) -/
  varMap : HashMap FVarId String := {}
  /-- Declared locals -/
  locals : Array VarDecl := #[]
  /-- Global arrays referenced -/
  globalArrays : Array ArrayDecl := #[]
  /-- Shared arrays referenced -/
  sharedArrays : Array ArrayDecl := #[]
  /-- Kernel parameters (from args structure) -/
  params : Array VarDecl := #[]
  deriving Inhabited

abbrev ExtractM := StateT ExtractState MetaM

-- Fresh variable generation
def freshVar (pre : String) : ExtractM String := do
  let state ← get
  let varName := s!"{pre}{state.nextVar}"
  modify fun s => { s with nextVar := s.nextVar + 1 }
  return varName

-- Emit a statement
def emitStmt (stmt : DStmt) : ExtractM Unit := do
  modify fun s => { s with stmts := s.stmts.push stmt }

-- Register a local variable
def declareLocal (name : String) (ty : DType) : ExtractM Unit := do
  modify fun s => { s with locals := s.locals.push { name, ty } }

-- Register FVar → device variable mapping
def registerVar (fvar : FVarId) (deviceName : String) : ExtractM Unit := do
  modify fun s => { s with varMap := s.varMap.insert fvar deviceName }

-- Lookup device variable for FVar
def lookupVar? (fvar : FVarId) : ExtractM (Option String) := do
  return (← get).varMap[fvar]?

-- Register global array
def registerGlobal (name : String) (ty : DType) : ExtractM Unit := do
  let state ← get
  -- Check if already registered
  if state.globalArrays.any (·.name == name) then
    return ()
  modify fun s => {
    s with globalArrays := s.globalArrays.push {
      name := name
      ty := ty
      space := .global
    }
  }

-- Register shared array
def registerShared (name : String) (ty : DType) : ExtractM Unit := do
  let state ← get
  if state.sharedArrays.any (·.name == name) then
    return ()
  modify fun s => {
    s with sharedArrays := s.sharedArrays.push {
      name := name
      ty := ty
      space := .shared
    }
  }

-- Try to reduce expression (avoid loose bvar errors)
def tryReduce (e : Expr) : ExtractM Expr := do
  try
    reduce e (skipProofs := false) (skipTypes := false)
  catch _ =>
    return e

------------------------------------
-- Expression Extraction
------------------------------------

-- Extract a device expression from a Lean expression
partial def extractExpr (e : Expr) : ExtractM (Option DExpr) := do
  let e ← tryReduce e

  match e with
  -- Literals
  | Expr.lit (.natVal n) =>
      return some (.intLit (Int.ofNat n))

  -- Free variables (parameters, earlier bindings)
  | Expr.fvar id =>
      if let some deviceVar ← lookupVar? id then
        return some (.var deviceVar)
      else
        -- Could be a kernel parameter field
        let decl ← id.getDecl
        return some (.var decl.userName.toString)

  -- GPU Primitives - globalIdxX
  | Expr.app (Expr.app (Expr.const ``GpuDSL.globalIdxX _) _) _ =>
      -- globalIdxX = blockIdx.x * blockDim.x + threadIdx.x
      return some <| .binop .add
        (.binop .mul (.blockIdx .x) (.blockDim .x))
        (.threadIdx .x)

  | Expr.app (Expr.app (Expr.const ``GpuDSL.globalIdxY _) _) _ =>
      return some <| .binop .add
        (.binop .mul (.blockIdx .y) (.blockDim .y))
        (.threadIdx .y)

  -- Binary operations
  | _ =>
      -- Try to match binary operations
      if e.isAppOfArity ``HAdd.hAdd 6 then
        let args := e.getAppArgs
        let a := args[4]!
        let b := args[5]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .add ae be)
        | _, _ => return none

      else if e.isAppOfArity ``HSub.hSub 6 then
        let args := e.getAppArgs
        let a := args[4]!
        let b := args[5]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .sub ae be)
        | _, _ => return none

      else if e.isAppOfArity ``HMul.hMul 6 then
        let args := e.getAppArgs
        let a := args[4]!
        let b := args[5]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .mul ae be)
        | _, _ => return none

      else if e.isAppOfArity ``HDiv.hDiv 6 then
        let args := e.getAppArgs
        let a := args[4]!
        let b := args[5]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .div ae be)
        | _, _ => return none

      else if e.isAppOfArity ``LT.lt 4 then
        let args := e.getAppArgs
        let a := args[2]!
        let b := args[3]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .lt ae be)
        | _, _ => return none

      else if e.isAppOfArity ``LE.le 4 then
        let args := e.getAppArgs
        let a := args[2]!
        let b := args[3]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .le ae be)
        | _, _ => return none

      else if e.isAppOfArity ``BEq.beq 4 then
        let args := e.getAppArgs
        let a := args[2]!
        let b := args[3]!
        let aExpr? ← extractExpr a
        let bExpr? ← extractExpr b
        match aExpr?, bExpr? with
        | some ae, some be => return some (.binop .eq ae be)
        | _, _ => return none

      else
        -- Unknown expression
        logWarning s!"Cannot extract expression: {← ppExpr e}"
        return none

------------------------------------
-- Statement Extraction
------------------------------------

-- Extract a single statement (GlobalArray.get/set, barrier, etc.)
partial def extractStmt (e : Expr) : ExtractM (Option Unit) := do
  let e ← tryReduce e

  -- GlobalArray.set: returns IO Unit
  if e.isAppOfArity ``GpuDSL.GlobalArray.set 6 then
    let args := e.getAppArgs
    -- args[3] = array (GlobalArray α with name field)
    -- args[4] = index
    -- args[5] = value
    let arrayExpr := args[3]!
    let idxExpr := args[4]!
    let valExpr := args[5]!

    -- Extract array name (it's a structure with a `name` field)
    let arrayName ← match arrayExpr with
      | Expr.fvar id => do
          let decl ← id.getDecl
          pure decl.userName.toString
      | _ =>
          throwError "Expected fvar for array in GlobalArray.set"

    -- Extract index and value expressions
    let idx? ← extractExpr idxExpr
    let val? ← extractExpr valExpr

    match idx?, val? with
    | some idx, some val =>
        emitStmt (.store (.var arrayName) idx val)
        registerGlobal arrayName (.array .float) -- TODO: get actual type
        return some ()
    | _, _ =>
        logError s!"Failed to extract GlobalArray.set index or value"
        return none

  -- SharedArray.set
  else if e.isAppOfArity ``GpuDSL.SharedArray.set 6 then
    let args := e.getAppArgs
    let arrayExpr := args[3]!
    let idxExpr := args[4]!
    let valExpr := args[5]!

    let arrayName ← match arrayExpr with
      | Expr.fvar id => do
          let decl ← id.getDecl
          pure decl.userName.toString
      | _ =>
          throwError "Expected fvar for array in SharedArray.set"

    let idx? ← extractExpr idxExpr
    let val? ← extractExpr valExpr

    match idx?, val? with
    | some idx, some val =>
        emitStmt (.store (.var arrayName) idx val)
        registerShared arrayName (.array .float)
        return some ()
    | _, _ =>
        return none

  -- Barrier
  else if e.isAppOfArity ``GpuDSL.barrier 2 then
    emitStmt .barrier
    return some ()

  -- Unknown statement
  else
    logWarning s!"Cannot extract statement: {← ppExpr e}"
    return none

-- Extract statements from monadic sequence
partial def extractStmts (e : Expr) : ExtractM (Option Unit) := do
  -- DON'T reduce immediately - match on structure first!

  -- Debug: log the expression structure
  -- logInfo s!"extractStmts: {e.ctorName}, isApp: {e.isApp}, const: {e.getAppFn}"

  match e with
  -- Pure return - end of sequence
  | Expr.app (Expr.app (Expr.const ``pure _) _) _ =>
      return some ()

  -- Monadic bind: m >>= k
  | Expr.app (Expr.app (Expr.app (Expr.app (Expr.const ``Bind.bind _) _) _) m) k =>
      -- First extract m (it might be a single statement or another bind)
      let result1? ← extractStmts m

      -- Then extract continuation k
      match k with
      | Expr.lam varName _ body _ =>
          -- Handle the binding - instantiate the lambda
          lambdaTelescope k fun fvars body' => do
            -- Process the body with the bound variable in scope
            let result2? ← extractStmts body'
            match result1?, result2? with
            | some (), some () => return some ()
            | _, _ => return none
      | _ =>
          -- k is not a lambda, just process it
          let result2? ← extractStmts k
          match result1?, result2? with
          | some (), some () => return some ()
          | _, _ => return none

  -- Let binding in monadic context: let x := v; body
  | Expr.app (Expr.lam varName varType body _) value =>
      -- Check what kind of operation this is

      -- GlobalArray.get: returns a value in the monad
      if value.isAppOfArity ``GpuDSL.GlobalArray.get 5 then
        let args := value.getAppArgs
        let arrayExpr := args[3]!
        let idxExpr := args[4]!

        let arrayName ← match arrayExpr with
          | Expr.fvar id => do
              let decl ← id.getDecl
              pure decl.userName.toString
          | _ => throwError "Expected fvar for array"

        let idx? ← extractExpr idxExpr

        match idx? with
        | some idx =>
            -- Create a device variable for the result
            let deviceVar ← freshVar "arr_val"
            emitStmt (.assign deviceVar (.index (.var arrayName) idx))
            declareLocal deviceVar .float
            registerGlobal arrayName (.array .float)

            -- Register the binding
            -- The body uses bvar 0, we need to instantiate it
            let bodyInst := body.instantiate1 (mkFVar ⟨varName⟩)
            registerVar ⟨varName⟩ deviceVar

            -- Continue with body
            extractStmts bodyInst
        | none => return none

      -- SharedArray.get
      else if value.isAppOfArity ``GpuDSL.SharedArray.get 5 then
        let args := value.getAppArgs
        let arrayExpr := args[3]!
        let idxExpr := args[4]!

        let arrayName ← match arrayExpr with
          | Expr.fvar id => do
              let decl ← id.getDecl
              pure decl.userName.toString
          | _ => throwError "Expected fvar for array"

        let idx? ← extractExpr idxExpr

        match idx? with
        | some idx =>
            let deviceVar ← freshVar "sh_val"
            emitStmt (.assign deviceVar (.index (.var arrayName) idx))
            declareLocal deviceVar .float
            registerShared arrayName (.array .float)

            let bodyInst := body.instantiate1 (mkFVar ⟨varName⟩)
            registerVar ⟨varName⟩ deviceVar

            extractStmts bodyInst
        | none => return none

      -- Regular let binding (scalar value)
      else
        let value? ← extractExpr value
        match value? with
        | some vexpr =>
            let deviceVar ← freshVar varName.toString
            emitStmt (.assign deviceVar vexpr)
            declareLocal deviceVar .nat -- TODO: infer type

            let bodyInst := body.instantiate1 (mkFVar ⟨varName⟩)
            registerVar ⟨varName⟩ deviceVar

            extractStmts bodyInst
        | none =>
            -- Just process body without binding
            extractStmts body

  -- If-then-else (ite or dite)
  | _ =>
      if e.isAppOfArity ``ite 5 then
        let args := e.getAppArgs
        let cond := args[2]!
        let thenBranch := args[3]!
        let elseBranch := args[4]!

        let condExpr? ← extractExpr cond

        match condExpr? with
        | some ce =>
            -- Save current statements
            let savedState ← get

            -- Extract then branch
            modify fun s => { s with stmts := #[] }
            let _ ← extractStmts thenBranch
            let thenStmts := (← get).stmts
            let thenBody := thenStmts.foldl (init := DStmt.skip) DStmt.seq

            -- Extract else branch
            modify fun s => { s with stmts := #[] }
            let _ ← extractStmts elseBranch
            let elseStmts := (← get).stmts
            let elseBody := elseStmts.foldl (init := DStmt.skip) DStmt.seq

            -- Restore state and emit ite
            modify fun s => {
              savedState with
                stmts := savedState.stmts.push (.ite ce thenBody elseBody)
                locals := s.locals
                globalArrays := s.globalArrays
                sharedArrays := s.sharedArrays
            }

            return some ()
        | none =>
            logError s!"Failed to extract if condition"
            return none

      -- Try to extract as a single statement
      else if let some () ← extractStmt e then
        return some ()

      -- Last resort: try reduction and re-match
      else do
        let e' ← tryReduce e
        if e'.eqv e then
          -- Reduction didn't change anything, give up
          logWarning s!"Cannot extract: {← ppExpr e}"
          return none
        else
          -- Try again with reduced expression
          extractStmts e'

------------------------------------
-- Main Extraction Entry Point
------------------------------------

-- Extract a kernel by name
def extractKernel (kernelName : Name) : MetaM (Option Kernel) := do
  -- Get the kernel definition
  let info ← getConstInfo kernelName
  let .defnInfo val := info | throwError s!"Not a definition: {kernelName}"

  logInfo s!"Extracting kernel: {kernelName}"
  logInfo s!"Type: {← ppExpr val.type}"

  -- Use lambdaTelescope to instantiate bound variables
  -- This prevents loose bvar errors and gives us cleaner structure
  lambdaTelescope val.value fun xs body => do
    logInfo s!"After telescoping: {xs.size} parameters"

    -- Initialize extraction state
    let initialState : ExtractState := {}

    -- Extract the kernel body (now with parameters instantiated)
    let (result?, finalState) ← (extractStmts body).run initialState

    match result? with
    | none =>
        logError s!"Failed to extract kernel {kernelName}"
        return none
    | some () =>
        -- Build the kernel structure
        let kernel : Kernel := {
          name := kernelName.toString
          params := finalState.params.toList
          locals := finalState.locals.toList
          globalArrays := finalState.globalArrays.toList
          sharedArrays := finalState.sharedArrays.toList
          body := finalState.stmts.toList.foldl (init := DStmt.skip) DStmt.seq
        }

        logInfo s!"✓ Successfully extracted kernel {kernelName}"
        return some kernel

-- Command: #extract_kernel kernelName
syntax (name := extractKernelCmd) "#extract_kernel " ident : command

@[command_elab extractKernelCmd]
def elabExtractKernelCmd : CommandElab := fun stx => do
  let kernelName := stx[1].getId

  liftTermElabM do
    let kernel? ← extractKernel kernelName

    match kernel? with
    | none =>
        logError s!"Failed to extract kernel {kernelName}"
    | some kernel =>
        logInfo s!"Extracted kernel IR:\n{repr kernel}"
        -- TODO: Store in environment extension

end DeviceExtractor
