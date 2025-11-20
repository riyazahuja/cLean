/-
  Device Reflection System

  Meta-level functions for translating Lean expressions to DeviceIR
-/

import Lean
import CLean.DeviceIR
import CLean.DeviceTranslation
import CLean.DeviceInstances

open Lean Meta Elab Command Term
open DeviceIR DeviceTranslation

namespace DeviceReflection

-- State for building device IR
structure BuilderState where
  /-- Next fresh variable ID -/
  nextVar : Nat := 0
  /-- Accumulated statements -/
  stmts : Array DStmt := #[]
  /-- Variable bindings (Lean FVar → device variable name) -/
  varMap : HashMap FVarId String := {}
  deriving Inhabited

abbrev BuilderM := StateT BuilderState MetaM

-- Generate a fresh variable name
def freshVar (prefix : String := "tmp") : BuilderM String := do
  let state ← get
  let varName := s!"{prefix}{state.nextVar}"
  modify fun s => { s with nextVar := s.nextVar + 1 }
  return varName

-- Add a statement to the builder
def emitStmt (stmt : DStmt) : BuilderM Unit := do
  modify fun s => { s with stmts := s.stmts.push stmt }

-- Register a Lean FVar → device variable mapping
def registerVar (fvar : FVarId) (deviceName : String) : BuilderM Unit := do
  modify fun s => { s with varMap := s.varMap.insert fvar deviceName }

-- Lookup device variable name for a Lean FVar
def lookupVar? (fvar : FVarId) : BuilderM (Option String) := do
  return (← get).varMap.find? fvar

-- Translate Lean expression to device expression
partial def reflectExpr (e : Expr) : BuilderM (Option DExpr) := do
  match e with
  -- Literals
  | Expr.lit (.natVal n) =>
      return some (.intLit (Int.ofNat n))

  | Expr.lit (.strVal _) =>
      -- Strings not supported on device
      logError s!"String literals not supported on device: {← ppExpr e}"
      return none

  -- Free variables (parameters, earlier bindings)
  | Expr.fvar id =>
      if let some deviceVar ← lookupVar? id then
        return some (.var deviceVar)
      else
        let decl ← id.getDecl
        logError s!"Unbound variable in device code: {decl.userName}"
        return none

  -- Constants (may be device-eligible functions)
  | Expr.const name _ =>
      -- Check if this is a known device constant/function
      -- For now, we'll handle specific cases
      match name with
      | _ =>
          logWarning s!"Constant {name} may not be device-eligible"
          return none

  -- Function application
  | Expr.app f arg => do
      -- Try to recognize common patterns

      -- Binary operations
      if f.isAppOfArity ``HAdd.hAdd 4 then
        -- Addition: a + b
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .add ae be)
        | _, _ => return none

      else if f.isAppOfArity ``HSub.hSub 4 then
        -- Subtraction
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .sub ae be)
        | _, _ => return none

      else if f.isAppOfArity ``HMul.hMul 4 then
        -- Multiplication
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .mul ae be)
        | _, _ => return none

      else if f.isAppOfArity ``HDiv.hDiv 4 then
        -- Division
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .div ae be)
        | _, _ => return none

      else if f.isAppOfArity ``LT.lt 4 then
        -- Less than
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .lt ae be)
        | _, _ => return none

      else if f.isAppOfArity ``LE.le 4 then
        -- Less than or equal
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .le ae be)
        | _, _ => return none

      else if f.isAppOfArity ``BEq.beq 4 then
        -- Equality
        let a := f.appFn!.appArg!
        let b := arg
        let aExpr ← reflectExpr a
        let bExpr ← reflectExpr b
        match aExpr, bExpr with
        | some ae, some be => return some (.binop .eq ae be)
        | _, _ => return none

      else if f.isAppOfArity ``Array.get! 3 then
        -- Array access: arr[i]
        let arr := f.appFn!.appArg!
        let idx := arg
        let arrExpr ← reflectExpr arr
        let idxExpr ← reflectExpr idx
        match arrExpr, idxExpr with
        | some ae, some ie => return some (.index ae ie)
        | _, _ => return none

      else
        -- Unknown application
        logWarning s!"Unsupported application: {← ppExpr e}"
        return none

  -- Lambda abstraction
  | Expr.lam varName ty body bi => do
      logWarning s!"Lambda abstractions not directly supported on device: {← ppExpr e}"
      return none

  -- Let bindings
  | Expr.letE varName ty value body _ => do
      logWarning s!"Let expression should be handled at statement level: {← ppExpr e}"
      return none

  -- Projections (field access)
  | Expr.proj typeName idx struct =>
      logWarning s!"Projections not yet implemented: {← ppExpr e}"
      return none

  | _ =>
      logWarning s!"Unsupported expression form: {← ppExpr e}"
      return none

-- Translate Lean statement-level expressions to device statements
partial def reflectStmt (e : Expr) : BuilderM (Option Unit) := do
  match e with
  -- Pure return (end of monadic sequence)
  | Expr.app (Expr.app (Expr.const ``pure _) _) _ =>
      -- Just skip, body is done
      return some ()

  -- Monadic bind: m >>= k
  | Expr.app (Expr.app (Expr.app (Expr.app (Expr.const ``Bind.bind _) _) _) m) k => do
      -- First execute m
      let _ ← reflectStmt m
      -- Then execute k
      match k with
      | Expr.lam varName _ body _ =>
          -- Handle the binding
          lambdaTelescope k fun fvars body' => do
            if fvars.size != 1 then
              logError "Expected single parameter in bind continuation"
              return none
            -- The fvar is bound to the result of m
            -- For now, we'll skip registering it unless m produces a value
            reflectStmt body'
      | _ =>
          reflectStmt k

  -- Let binding: let x := value; body
  | Expr.letE varName ty value body _ => do
      -- Try to translate value to an expression
      let valueExpr? ← reflectExpr value
      match valueExpr? with
      | some vexpr =>
          -- Generate a device variable for this let binding
          let deviceVar ← freshVar varName.toString
          -- Emit assignment
          emitStmt (.assign deviceVar vexpr)
          -- In the body, this Lean binding will be represented by deviceVar
          -- Continue with body (body has varName as a free variable reference via bvar 0)
          -- We need to instantiate it
          let bodyInst := body.instantiate1 (mkFVar ⟨varName⟩)
          -- Register the mapping
          registerVar ⟨varName⟩ deviceVar
          reflectStmt bodyInst
      | none =>
          logError s!"Failed to translate let binding value: {← ppExpr value}"
          return none

  -- If-then-else (represented as ite or match)
  | _ =>
      if e.isAppOfArity ``ite 5 then
        -- ite cond thenBranch elseBranch
        let cond := e.getArg! 2
        let thenBranch := e.getArg! 3
        let elseBranch := e.getArg! 4

        let condExpr? ← reflectExpr cond
        match condExpr? with
        | some ce =>
            -- Save current statements
            let savedStmts := (← get).stmts
            -- Process then branch
            modify fun s => { s with stmts := #[] }
            let _ ← reflectStmt thenBranch
            let thenStmts := (← get).stmts
            let thenStmt := thenStmts.foldl .seq .skip

            -- Process else branch
            modify fun s => { s with stmts := #[] }
            let _ ← reflectStmt elseBranch
            let elseStmts := (← get).stmts
            let elseStmt := elseStmts.foldl .seq .skip

            -- Restore and emit ite
            modify fun s => { s with stmts := savedStmts }
            emitStmt (.ite ce thenStmt elseStmt)
            return some ()
        | none =>
            logError s!"Failed to translate condition: {← ppExpr cond}"
            return none
      else
        logWarning s!"Unsupported statement form: {← ppExpr e}"
        return none

-- Main entry point: translate a Lean definition to DeviceIR
def translateDefinition (name : Name) : MetaM (Option Kernel) := do
  -- Get the definition
  let info ← getConstInfo name
  let .defnInfo val := info | throwError "Not a definition: {name}"

  -- Initialize builder state
  let initialState : BuilderState := {}

  -- Translate the body
  let (result?, finalState) ← (reflectStmt val.value).run initialState

  match result? with
  | none =>
      logError s!"Translation of {name} failed"
      return none
  | some _ =>
      -- Build the kernel
      let kernel : Kernel := {
        name := name.toString
        params := []      -- TODO: extract from type
        locals := []      -- TODO: collect from varMap
        globalArrays := []  -- TODO: detect array parameters
        sharedArrays := []  -- TODO: detect shared arrays
        body := finalState.stmts.foldl .seq .skip
      }
      return some kernel

-- Command: #device functionName
syntax (name := deviceCmd) "#device " ident : command

@[command_elab deviceCmd]
def elabDeviceCmd : CommandElab := fun stx => do
  let fnName := stx[1].getId

  liftTermElabM do
    let kernel? ← translateDefinition fnName
    match kernel? with
    | none =>
        logError s!"Failed to translate {fnName} to device IR"
    | some kernel =>
        logInfo s!"✓ Translated {fnName} to device IR:\n{repr kernel}"
        -- Store in environment extension
        modifyEnv fun env =>
          deviceFnExt.addEntry env { name := fnName, ir := kernel.body }

end DeviceReflection
