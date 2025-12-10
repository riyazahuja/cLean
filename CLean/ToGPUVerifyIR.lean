import CLean.DeviceIR
import CLean.Verification.GPUVerifyStyle
import CLean.GPU

/-!
# DeviceIR to GPUVerify-Style Translator

Automatically translates DeviceIR kernels to KernelSpec for verification.
Follows GPUVerify's approach of converting to an intermediate representation
that makes two-thread reasoning explicit.

Key Steps:
1. Analyze expressions to identify access patterns
2. Convert threadIdx-based indexing to symbolic functions
3. Extract all memory accesses (from store statements)
4. Identify barrier locations
-/

namespace CLean.ToGPUVerifyIR

open DeviceIR
open CLean.Verification.GPUVerify
open GpuDSL

/-! ## Expression Analysis -/

/-- Determine if an expression is thread-dependent (uses threadIdx) -/
def isThreadDependent (e : DExpr) : Bool :=
  match e with
  | DExpr.threadIdx _ => true
  | DExpr.binop _ e1 e2 => isThreadDependent e1 || isThreadDependent e2
  | DExpr.unop _ e => isThreadDependent e
  | DExpr.index e1 e2 => isThreadDependent e1 || isThreadDependent e2
  | _ => false

/-- Convert a DExpr to a symbolic AddressPattern

    Maps common patterns:
    - threadIdx.x → AddressPattern.identity
    - threadIdx.x + c → AddressPattern.offset c
    - blockIdx.x * blockDim.x + threadIdx.x → AddressPattern.identity (within-block view)
    - constant c → AddressPattern.constant c
    - variable name → look up CACHED pattern in varEnv (no recursion!)
    - a * tid + b → AddressPattern.linear a b (concrete)
    - param * tid + ... → AddressPattern.symLinear (symbolic)
-/
def dexprToAddressPattern (e : DExpr) (blockDim : Dim3) (varEnv : List (String × AddressPattern)) : AddressPattern :=
  match e with
  -- Literal constant
  | DExpr.intLit n => AddressPattern.constant n.toNat

  -- Thread index (identity)
  | DExpr.threadIdx _ => AddressPattern.identity

  -- Variable: look up CACHED pattern (no recursive evaluation!)
  | DExpr.var varName =>
      match varEnv.lookup varName with
      | some cachedPattern => cachedPattern  -- Just return the cached pattern!
      | none => AddressPattern.symLinear (SymValue.const 0) (SymValue.param varName)  -- Kernel parameter

  -- Binary operations
  | DExpr.binop BinOp.add e1 e2 =>
      -- Special case: blockIdx.x * blockDim.x + threadIdx.x
      match e1, e2 with
      | DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x), DExpr.threadIdx Dim.x =>
          AddressPattern.identity  -- Global index, but we focus on within-block (two-thread reduction)
      | _, _ =>
          -- Try to construct offset/linear pattern
          let p1 := dexprToAddressPattern e1 blockDim varEnv
          let p2 := dexprToAddressPattern e2 blockDim varEnv
          match p1, p2 with
          | AddressPattern.identity, AddressPattern.constant n => AddressPattern.offset n
          | AddressPattern.constant n, AddressPattern.identity => AddressPattern.offset n
          | AddressPattern.offset base, AddressPattern.constant n => AddressPattern.offset (base + n)
          | AddressPattern.constant n, AddressPattern.offset base => AddressPattern.offset (n + base)
          | AddressPattern.constant n1, AddressPattern.constant n2 => AddressPattern.constant (n1 + n2)
          -- Linear pattern additions (concrete)
          | AddressPattern.linear s o, AddressPattern.constant n => AddressPattern.linear s (o + n)
          | AddressPattern.constant n, AddressPattern.linear s o => AddressPattern.linear s (n + o)
          | AddressPattern.linear s1 o1, AddressPattern.linear s2 o2 => AddressPattern.linear (s1 + s2) (o1 + o2)
          | AddressPattern.identity, AddressPattern.linear s o => AddressPattern.linear (1 + s) o
          | AddressPattern.linear s o, AddressPattern.identity => AddressPattern.linear (s + 1) o
          | AddressPattern.offset base, AddressPattern.linear s o => AddressPattern.linear (1 + s) (base + o)
          | AddressPattern.linear s o, AddressPattern.offset base => AddressPattern.linear (s + 1) (o + base)
          -- Symbolic pattern additions
          | AddressPattern.symLinear s1 o1, AddressPattern.symLinear s2 o2 =>
              AddressPattern.symLinear (SymValue.symAdd s1 s2) (SymValue.symAdd o1 o2)
          | AddressPattern.identity, AddressPattern.symLinear s o =>
              AddressPattern.symLinear (SymValue.symAdd (SymValue.const 1) s) o
          | AddressPattern.symLinear s o, AddressPattern.identity =>
              AddressPattern.symLinear (SymValue.symAdd s (SymValue.const 1)) o
          | AddressPattern.symLinear s o, AddressPattern.constant n =>
              AddressPattern.symLinear s (SymValue.symAdd o (SymValue.const n))
          | AddressPattern.constant n, AddressPattern.symLinear s o =>
              AddressPattern.symLinear s (SymValue.symAdd (SymValue.const n) o)
          | AddressPattern.offset base, AddressPattern.symLinear s o =>
              AddressPattern.symLinear (SymValue.symAdd (SymValue.const 1) s) (SymValue.symAdd (SymValue.const base) o)
          | AddressPattern.symLinear s o, AddressPattern.offset base =>
              AddressPattern.symLinear (SymValue.symAdd s (SymValue.const 1)) (SymValue.symAdd o (SymValue.const base))
          | AddressPattern.linear s1 o1, AddressPattern.symLinear s2 o2 =>
              AddressPattern.symLinear (SymValue.symAdd (SymValue.const s1) s2) (SymValue.symAdd (SymValue.const o1) o2)
          | AddressPattern.symLinear s1 o1, AddressPattern.linear s2 o2 =>
              AddressPattern.symLinear (SymValue.symAdd s1 (SymValue.const s2)) (SymValue.symAdd o1 (SymValue.const o2))
          | _, _ => AddressPattern.constant 0  -- Conservative: unknown pattern

  | DExpr.binop BinOp.sub e1 e2 =>
      let p1 := dexprToAddressPattern e1 blockDim varEnv
      let p2 := dexprToAddressPattern e2 blockDim varEnv
      match p1, p2 with
      | AddressPattern.constant n1, AddressPattern.constant n2 => AddressPattern.constant (n1 - n2)
      | AddressPattern.offset base, AddressPattern.constant n => AddressPattern.offset (base - n)
      | AddressPattern.linear s o, AddressPattern.constant n => AddressPattern.linear s (o - n)
      | AddressPattern.symLinear s o, AddressPattern.constant n =>
          AddressPattern.symLinear s (SymValue.symAdd o (SymValue.const (0 - n)))  -- Negate by subtracting
      | AddressPattern.symLinear s o, AddressPattern.symLinear _ o2 =>
          -- Conservatively: scale stays, offset is symbolic difference
          AddressPattern.symLinear s (SymValue.symAdd o (SymValue.symMul (SymValue.const 0) o2))  -- Approximation
      | _, _ => AddressPattern.constant 0  -- Conservative

  | DExpr.binop BinOp.mul e1 e2 =>
      let p1 := dexprToAddressPattern e1 blockDim varEnv
      let p2 := dexprToAddressPattern e2 blockDim varEnv
      match p1, p2 with
      | AddressPattern.constant n1, AddressPattern.constant n2 => AddressPattern.constant (n1 * n2)
      -- identity * constant = linear pattern with scale
      | AddressPattern.identity, AddressPattern.constant n => AddressPattern.linear n 0
      | AddressPattern.constant n, AddressPattern.identity => AddressPattern.linear n 0
      -- offset * constant = linear pattern
      | AddressPattern.offset base, AddressPattern.constant n => AddressPattern.linear n (base * n)
      | AddressPattern.constant n, AddressPattern.offset base => AddressPattern.linear n (n * base)
      -- linear * constant (scale the linear pattern)
      | AddressPattern.linear s o, AddressPattern.constant n => AddressPattern.linear (s * n) (o * n)
      | AddressPattern.constant n, AddressPattern.linear s o => AddressPattern.linear (n * s) (n * o)
      -- identity * symbolic (parameter) = symLinear
      | AddressPattern.identity, AddressPattern.symLinear (SymValue.const 0) symParam =>
          AddressPattern.symLinear symParam (SymValue.const 0)  -- tid * param
      | AddressPattern.symLinear (SymValue.const 0) symParam, AddressPattern.identity =>
          AddressPattern.symLinear symParam (SymValue.const 0)  -- param * tid
      -- Concrete value * symbolic parameter
      | AddressPattern.constant n, AddressPattern.symLinear (SymValue.const 0) symParam =>
          AddressPattern.symLinear (SymValue.const 0) (SymValue.symMul (SymValue.const n) symParam)
      | AddressPattern.symLinear (SymValue.const 0) symParam, AddressPattern.constant n =>
          AddressPattern.symLinear (SymValue.const 0) (SymValue.symMul symParam (SymValue.const n))
      -- symLinear (tid-scaled) * constant
      | AddressPattern.symLinear s o, AddressPattern.constant n =>
          AddressPattern.symLinear (SymValue.symMul s (SymValue.const n)) (SymValue.symMul o (SymValue.const n))
      | AddressPattern.constant n, AddressPattern.symLinear s o =>
          AddressPattern.symLinear (SymValue.symMul (SymValue.const n) s) (SymValue.symMul (SymValue.const n) o)
      -- linear * symLinear (conservative)
      | AddressPattern.linear s1 o1, AddressPattern.symLinear _ o2 =>
          AddressPattern.symLinear (SymValue.symMul (SymValue.const s1) o2) (SymValue.symMul (SymValue.const o1) o2)
      | AddressPattern.symLinear _ o1, AddressPattern.linear s2 o2 =>
          AddressPattern.symLinear (SymValue.symMul o1 (SymValue.const s2)) (SymValue.symMul o1 (SymValue.const o2))
      | _, _ => AddressPattern.constant 0  -- Conservative

  -- Fallback: treat as constant 0
  | _ => AddressPattern.constant 0

/-! ## Statement Analysis for Access Extraction -/

/-- Barrier info for uniformity analysis -/
structure BarrierInfo where
  location : Nat
  inConditional : Bool  -- True if barrier is inside an if/else block
  deriving Inhabited, Repr

structure AccessExtractor where
  accesses : List AccessPattern
  nextLocation : Nat  -- Counter for assigning new location IDs
  barriers : List Nat
  barrierInfos : List BarrierInfo  -- Detailed barrier info for uniformity checking
  varEnv : List (String × AddressPattern)  -- Cache computed address patterns
  arrayLocations : List (String × Nat)  -- Map array names to location IDs
  sharedArrays : List String  -- Names of arrays in shared memory
  inConditional : Bool  -- Track if we're inside an if/else block
  stmtCounter : Nat  -- Counter for statement indices (program counter)
  deriving Inhabited

/-- Determine memory space for an array by checking if it's in sharedArrays list -/
def AccessExtractor.getMemorySpace (state : AccessExtractor) (arrName : String) : Verification.GPUVerify.MemorySpace :=
  if state.sharedArrays.contains arrName then
    MemorySpace.shared
  else
    MemorySpace.global

/-- Get or create a location ID for an array -/
def AccessExtractor.getArrayLocation (state : AccessExtractor) (arrName : String) : (Nat × AccessExtractor) :=
  match state.arrayLocations.lookup arrName with
  | some loc => (loc, state)
  | none =>
      let loc := state.nextLocation
      (loc, { state with
        nextLocation := state.nextLocation + 1
        arrayLocations := (arrName, loc) :: state.arrayLocations })

/-- Extract array name from a DExpr (if it's an array access) -/
def getArrayName : DExpr → Option String
  | DExpr.var name => some name
  | _ => none

/-- Check if a DExpr contains an array store pattern -/
def expressionHasArrayAccess (e : DExpr) : Bool :=
  match e with
  | DExpr.index _ _ => true
  | DExpr.binop _ e1 e2 => expressionHasArrayAccess e1 || expressionHasArrayAccess e2
  | DExpr.unop _ e => expressionHasArrayAccess e
  | _ => false

/-- Extract read accesses from an expression (array indexing in RHS)
    Uses the current stmtCounter for the access's stmtIdx -/
def extractReadsFromExpr (e : DExpr) (state : AccessExtractor) (blockDim : Dim3) : AccessExtractor :=
  match e with
  | DExpr.index arr idx =>
      -- Array read: arr[idx]
      let addrPat := dexprToAddressPattern idx blockDim state.varEnv
      -- Get location ID and memory space for this array
      let (loc, state', space) := match getArrayName arr with
        | some name =>
            let (loc, st) := state.getArrayLocation name
            let sp := st.getMemorySpace name
            (loc, st, sp)
        | none => (state.nextLocation, { state with nextLocation := state.nextLocation + 1 }, MemorySpace.global)
      -- Use current stmtCounter as the stmtIdx (all 4 positional args)
      let access := AccessPattern.read addrPat loc space state'.stmtCounter
      { state' with accesses := access :: state'.accesses }

  | DExpr.binop _ e1 e2 =>
      let state1 := extractReadsFromExpr e1 state blockDim
      extractReadsFromExpr e2 state1 blockDim

  | DExpr.unop _ e =>
      extractReadsFromExpr e state blockDim

  | _ => state

/-- Extract access patterns from a single statement
    Increments stmtCounter for each statement to track program order -/
def extractFromStmt (stmt : DStmt) (state : AccessExtractor) (blockDim : Dim3) : AccessExtractor :=
  match stmt with
  | DStmt.skip =>
      -- Increment counter even for skip to maintain ordering
      { state with stmtCounter := state.stmtCounter + 1 }

  | DStmt.assign varName rhs =>
      -- Compute and cache the address pattern for this variable
      let state1 := extractReadsFromExpr rhs state blockDim
      let addrPattern := dexprToAddressPattern rhs blockDim state1.varEnv
      { state1 with
        varEnv := (varName, addrPattern) :: state1.varEnv
        stmtCounter := state1.stmtCounter + 1 }

  | DStmt.store arr idx val =>
      -- Store: arr[idx] := val
      -- 1. Extract reads from the value expression
      let state1 := extractReadsFromExpr val state blockDim
      -- 2. Extract reads from index expression if it contains array accesses
      let state2 := extractReadsFromExpr idx state1 blockDim
      -- 3. Get location ID and memory space for this array
      let (loc, state3, space) := match getArrayName arr with
        | some name =>
            let (loc, st) := state2.getArrayLocation name
            let sp := st.getMemorySpace name
            (loc, st, sp)
        | none => (state2.nextLocation, { state2 with nextLocation := state2.nextLocation + 1 }, MemorySpace.global)
      -- 4. Add the write access with memory space and stmtIdx
      let addrPat := dexprToAddressPattern idx blockDim state3.varEnv
      let stmtIdx := state3.stmtCounter
      let access : AccessPattern := .write addrPat loc space stmtIdx
      { state3 with
        accesses := access :: state3.accesses
        stmtCounter := state3.stmtCounter + 1 }

  | DStmt.seq s1 s2 =>
      let state1 := extractFromStmt s1 state blockDim
      extractFromStmt s2 state1 blockDim

  | DStmt.ite cond sthen selse =>
      -- Extract from condition
      let state0 := extractReadsFromExpr cond state blockDim
      -- Mark that we're entering a conditional block
      let state0' := { state0 with inConditional := true }
      -- Conservative: collect accesses from both branches
      let stateThen := extractFromStmt sthen state0' blockDim
      let stateElse := extractFromStmt selse state0' blockDim
      -- Restore inConditional to original value after branches, take max stmtCounter
      { stateThen with
        accesses := stateThen.accesses ++ stateElse.accesses
        nextLocation := max stateThen.nextLocation stateElse.nextLocation
        arrayLocations := stateThen.arrayLocations ++ stateElse.arrayLocations
        barrierInfos := stateThen.barrierInfos ++ stateElse.barrierInfos
        inConditional := state.inConditional
        stmtCounter := max stateThen.stmtCounter stateElse.stmtCounter }

  | DStmt.for _ lo hi body =>
      -- Extract from loop bounds
      let state1 := extractReadsFromExpr lo state blockDim
      let state2 := extractReadsFromExpr hi state1 blockDim
      -- Extract from body (simplified: once)
      extractFromStmt body state2 blockDim

  | DStmt.whileLoop cond body =>
      let state1 := extractReadsFromExpr cond state blockDim
      extractFromStmt body state1 blockDim

  | DStmt.barrier =>
      -- Barrier location is based on stmtCounter (program order)
      let barrierLoc := state.stmtCounter
      let info : BarrierInfo := { location := barrierLoc, inConditional := state.inConditional }
      { state with
        barriers := barrierLoc :: state.barriers
        barrierInfos := info :: state.barrierInfos
        stmtCounter := state.stmtCounter + 1 }

  | DStmt.call _ args =>
      -- Extract reads from arguments
      args.foldl (fun s arg => extractReadsFromExpr arg s blockDim) state

  | DStmt.assert cond _ =>
      extractReadsFromExpr cond state blockDim

/-! ## Main Translation Function -/

/-- Translate DeviceIR Kernel to GPUVerify-style KernelSpec -/
def deviceIRToKernelSpec (kernel : Kernel) (blockDim gridDim : Dim3) : KernelSpec :=
  -- Extract shared array names from kernel definition
  let sharedNames := kernel.sharedArrays.map (fun a => a.name)

  let initialState : AccessExtractor := {
    accesses := []
    nextLocation := 0
    barriers := []
    barrierInfos := []
    varEnv := []
    arrayLocations := []
    sharedArrays := sharedNames
    inConditional := false
    stmtCounter := 0
  }

  let finalState := extractFromStmt kernel.body initialState blockDim

  {
    blockSize := blockDim.x  -- Simplified: 1D only for now
    gridSize := gridDim.x
    accesses := finalState.accesses.reverse  -- Restore program order
    barriers := finalState.barriers.reverse
  }

/-- Translate DeviceIR Kernel to KernelSpec with barrier info for uniformity checking -/
def deviceIRToKernelSpecWithBarrierInfo (kernel : Kernel) (blockDim gridDim : Dim3) : (KernelSpec × List BarrierInfo) :=
  let sharedNames := kernel.sharedArrays.map (fun a => a.name)

  let initialState : AccessExtractor := {
    accesses := []
    nextLocation := 0
    barriers := []
    barrierInfos := []
    varEnv := []
    arrayLocations := []
    sharedArrays := sharedNames
    inConditional := false
    stmtCounter := 0
  }

  let finalState := extractFromStmt kernel.body initialState blockDim

  let spec : KernelSpec := {
    blockSize := blockDim.x
    gridSize := gridDim.x
    accesses := finalState.accesses.reverse
    barriers := finalState.barriers.reverse
  }

  (spec, finalState.barrierInfos.reverse)

/-- Helper: Print KernelSpec for debugging -/

def printKernelSpec (spec : KernelSpec) : IO Unit := do
  IO.println s!"KernelSpec:"
  IO.println s!"  Block size: {spec.blockSize}"
  IO.println s!"  Grid size: {spec.gridSize}"
  IO.println s!"  Accesses: {spec.accesses.length}"
  for (acc, i) in spec.accesses.zipIdx do
    IO.println s!"    [{i}] {repr acc}"
  IO.println s!"  Barriers: {spec.barriers}"

end CLean.ToGPUVerifyIR
