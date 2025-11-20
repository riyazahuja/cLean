/-
  Kernel IR Builder

  KernelOps instance that builds DeviceIR instead of executing.
  This allows the same kernel code to be used for both execution and IR extraction.
-/

import CLean.KernelOps
import CLean.DeviceIR
import CLean.DeviceTranslation

open DeviceIR DeviceTranslation

namespace CLean

-- State for building kernel IR
structure KernelBuilderState where
  /-- Next fresh variable counter -/
  nextVar : Nat := 0
  /-- Accumulated statements -/
  stmts : List DStmt := []
  /-- Locals declared -/
  locals : List VarDecl := []
  /-- Global arrays referenced -/
  globalArrays : List ArrayDecl := []
  /-- Shared arrays declared -/
  sharedArrays : List ArrayDecl := []
  deriving Inhabited

-- Monad for building kernel IR
abbrev KernelBuilderM := StateT KernelBuilderState Id

namespace KernelBuilderM

-- Generate a fresh variable name
def freshVar (pre : String) : KernelBuilderM String := do
  let state ← get
  let varName := s!"{pre}{state.nextVar}"
  modify fun s => { s with nextVar := s.nextVar + 1 }
  return varName

-- Emit a statement
def emitStmt (stmt : DStmt) : KernelBuilderM Unit := do
  modify fun s => { s with stmts := s.stmts.concat stmt }

-- Declare a local variable
def declareLocal (name : String) (ty : DType) : KernelBuilderM Unit := do
  modify fun s => { s with locals := s.locals.concat { name, ty } }

-- Register a global array
def registerGlobal (name : String) (ty : DType) : KernelBuilderM Unit := do
  modify fun s => {
    s with globalArrays := s.globalArrays.concat {
      name := name
      ty := ty
      space := .global
    }
  }

-- Register a shared array
def registerShared (name : String) (ty : DType) : KernelBuilderM Unit := do
  modify fun s => {
    s with sharedArrays := s.sharedArrays.concat {
      name := name
      ty := ty
      space := .shared
    }
  }

-- Get the final kernel
def toKernel (name : String) (params : List VarDecl) : KernelBuilderM Kernel := do
  let state ← get
  return {
    name := name
    params := params
    locals := state.locals
    globalArrays := state.globalArrays
    sharedArrays := state.sharedArrays
    body := state.stmts.foldl DStmt.seq DStmt.skip
  }

end KernelBuilderM

-- KernelOps instance that builds DeviceIR
instance : Monad KernelBuilderM := inferInstanceAs (Monad (StateT _ _))

instance : KernelOps KernelBuilderM where
  Expr := DExpr

  -- Literals
  natLit n := .intLit (Int.ofNat n)
  intLit i := .intLit i
  floatLit f := .floatLit f
  boolLit b := .boolLit b

  -- Arithmetic
  add a b := .binop .add a b
  sub a b := .binop .sub a b
  mul a b := .binop .mul a b
  div a b := .binop .div a b
  mod a b := .binop .mod a b

  -- Comparison
  lt a b := .binop .lt a b
  le a b := .binop .le a b
  gt a b := .binop .gt a b
  ge a b := .binop .ge a b
  eq a b := .binop .eq a b
  ne a b := .binop .ne a b

  -- Logical
  and a b := .binop .and a b
  or a b := .binop .or a b
  not a := .unop .not a

  -- GPU thread/block intrinsics
  globalIdxX := do
    -- globalIdx.x = blockIdx.x * blockDim.x + threadIdx.x
    let bid := DExpr.blockIdx .x
    let bdim := DExpr.blockDim .x
    let tid := DExpr.threadIdx .x
    return .binop .add (.binop .mul bid bdim) tid

  globalIdxY := do
    let bid := DExpr.blockIdx .y
    let bdim := DExpr.blockDim .y
    let tid := DExpr.threadIdx .y
    return .binop .add (.binop .mul bid bdim) tid

  globalIdxZ := do
    let bid := DExpr.blockIdx .z
    let bdim := DExpr.blockDim .z
    let tid := DExpr.threadIdx .z
    return .binop .add (.binop .mul bid bdim) tid

  blockIdxX := pure (.blockIdx .x)
  blockIdxY := pure (.blockIdx .y)
  blockIdxZ := pure (.blockIdx .z)

  threadIdxX := pure (.threadIdx .x)
  threadIdxY := pure (.threadIdx .y)
  threadIdxZ := pure (.threadIdx .z)

  blockDimX := pure (.blockDim .x)
  blockDimY := pure (.blockDim .y)
  blockDimZ := pure (.blockDim .z)

  gridDimX := pure (.gridDim .x)
  gridDimY := pure (.gridDim .y)
  gridDimZ := pure (.gridDim .z)

  -- Memory operations
  globalGet arrName idx := do
    -- Create a fresh variable to hold the result
    let resultVar ← KernelBuilderM.freshVar "gval"
    -- Emit: resultVar := arrName[idx]
    KernelBuilderM.emitStmt (.assign resultVar (.index (.var arrName) idx))
    -- Declare the local variable (assume float for now; TODO: track types)
    KernelBuilderM.declareLocal resultVar .float
    return .var resultVar

  globalSet arrName idx value := do
    -- Emit: arrName[idx] := value
    KernelBuilderM.emitStmt (.store (.var arrName) idx value)
    -- Register this array if not already registered
    KernelBuilderM.registerGlobal arrName (.array .float)

  sharedGet arrName idx := do
    let resultVar ← KernelBuilderM.freshVar "sval"
    KernelBuilderM.emitStmt (.assign resultVar (.index (.var arrName) idx))
    KernelBuilderM.declareLocal resultVar .float
    return .var resultVar

  sharedSet arrName idx value := do
    KernelBuilderM.emitStmt (.store (.var arrName) idx value)
    KernelBuilderM.registerShared arrName (.array .float)

  -- Synchronization
  barrier := KernelBuilderM.emitStmt .barrier

  -- Control flow
  ifThenElse cond thenBranch elseBranch := do
    -- Save current state
    let savedState ← get

    -- Build then branch
    modify fun s => { s with stmts := [] }
    thenBranch
    let thenStmts ← get
    let thenBody := thenStmts.stmts.foldl DStmt.seq DStmt.skip

    -- Build else branch
    modify fun s => { s with stmts := [] }
    elseBranch
    let elseStmts ← get
    let elseBody := elseStmts.stmts.foldl DStmt.seq DStmt.skip

    -- Restore state (but keep accumulated locals/arrays)
    modify fun s => {
      savedState with
        stmts := savedState.stmts.concat (.ite cond thenBody elseBody)
        locals := s.locals
        globalArrays := s.globalArrays
        sharedArrays := s.sharedArrays
    }

  forLoop varName lo hi body := do
    -- Save current state
    let savedState ← get

    -- Build loop body
    modify fun s => { s with stmts := [] }
    -- The loop variable is available as an expression
    body (.var varName)
    let bodyStmts ← get
    let loopBody := bodyStmts.stmts.foldl DStmt.seq DStmt.skip

    -- Restore state and emit for loop
    modify fun s => {
      savedState with
        stmts := savedState.stmts.concat (.for varName lo hi loopBody)
        locals := savedState.locals.concat { name := varName, ty := .int }
        globalArrays := s.globalArrays
        sharedArrays := s.sharedArrays
    }

-- Helper to run a kernel builder and extract the IR
def buildKernel (name : String) (params : List VarDecl)
    (builder : KernelBuilderM Unit) : Kernel :=
  let initialState : KernelBuilderState := {}
  let (kernel, _) := (builder *> KernelBuilderM.toKernel name params).run initialState
  kernel

end CLean
