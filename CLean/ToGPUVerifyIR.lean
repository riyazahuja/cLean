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

structure AccessExtractor where
  accesses : List AccessPattern
  location : Nat
  barriers : List Nat
  varEnv : List (String × AddressPattern)  -- Cache computed address patterns, not expressions!
  deriving Inhabited

/-- Check if a DExpr contains an array store pattern -/
def expressionHasArrayAccess (e : DExpr) : Bool :=
  match e with
  | DExpr.index _ _ => true
  | DExpr.binop _ e1 e2 => expressionHasArrayAccess e1 || expressionHasArrayAccess e2
  | DExpr.unop _ e => expressionHasArrayAccess e
  | _ => false

/-- Extract read accesses from an expression (array indexing in RHS) -/
def extractReadsFromExpr (e : DExpr) (state : AccessExtractor) (blockDim : Dim3) : AccessExtractor :=
  match e with
  | DExpr.index _ idx =>
      -- Array read: arr[idx]
      let addrPat := dexprToAddressPattern idx blockDim state.varEnv
      let access := AccessPattern.read addrPat state.location
      { state with accesses := access :: state.accesses }

  | DExpr.binop _ e1 e2 =>
      let state1 := extractReadsFromExpr e1 state blockDim
      extractReadsFromExpr e2 state1 blockDim

  | DExpr.unop _ e =>
      extractReadsFromExpr e state blockDim

  | _ => state

/-- Extract access patterns from a single statement -/
def extractFromStmt (stmt : DStmt) (state : AccessExtractor) (blockDim : Dim3) : AccessExtractor :=
  match stmt with
  | DStmt.skip => state

  | DStmt.assign varName rhs =>
      -- Compute and cache the address pattern for this variable
      let state1 := extractReadsFromExpr rhs state blockDim
      let addrPattern := dexprToAddressPattern rhs blockDim state1.varEnv
      { state1 with varEnv := (varName, addrPattern) :: state1.varEnv }

  | DStmt.store _ idx val =>
      -- Store: arr[idx] := val
      -- 1. Extract reads from the value expression
      let state1 := extractReadsFromExpr val state blockDim
      -- 2. Extract reads from index expression if it contains array accesses
      let state2 := extractReadsFromExpr idx state1 blockDim
      -- 3. Add the write access
      let addrPat := dexprToAddressPattern idx blockDim state2.varEnv
      let access := AccessPattern.write addrPat state2.location
      { state2 with
        accesses := access :: state2.accesses
        location := state2.location + 1 }

  | DStmt.seq s1 s2 =>
      let state1 := extractFromStmt s1 state blockDim
      extractFromStmt s2 state1 blockDim

  | DStmt.ite cond sthen selse =>
      -- Extract from condition
      let state0 := extractReadsFromExpr cond state blockDim
      -- Conservative: collect accesses from both branches
      let stateThen := extractFromStmt sthen { state0 with location := state0.location + 1 } blockDim
      let stateElse := extractFromStmt selse { state0 with location := state0.location + 1 } blockDim
      { stateThen with
        accesses := stateThen.accesses ++ stateElse.accesses
        location := max stateThen.location stateElse.location }

  | DStmt.for _ lo hi body =>
      -- Extract from loop bounds
      let state1 := extractReadsFromExpr lo state blockDim
      let state2 := extractReadsFromExpr hi state1 blockDim
      -- Extract from body (simplified: once)
      extractFromStmt body { state2 with location := state2.location + 1 } blockDim

  | DStmt.whileLoop cond body =>
      let state1 := extractReadsFromExpr cond state blockDim
      extractFromStmt body { state1 with location := state1.location + 1 } blockDim

  | DStmt.barrier =>
      { state with
        barriers := state.location :: state.barriers
        location := state.location + 1 }

  | DStmt.call _ args =>
      -- Extract reads from arguments
      let state' := args.foldl (fun s arg => extractReadsFromExpr arg s blockDim) state
      { state' with location := state'.location + 1 }

  | DStmt.assert cond _ =>
      let state' := extractReadsFromExpr cond state blockDim
      { state' with location := state'.location + 1 }

/-! ## Main Translation Function -/

/-- Translate DeviceIR Kernel to GPUVerify-style KernelSpec -/
def deviceIRToKernelSpec (kernel : Kernel) (blockDim gridDim : Dim3) : KernelSpec :=
  let initialState : AccessExtractor := {
    accesses := []
    location := 0
    barriers := []
    varEnv := []
  }

  let finalState := extractFromStmt kernel.body initialState blockDim

  {
    blockSize := blockDim.x  -- Simplified: 1D only for now
    gridSize := gridDim.x
    accesses := finalState.accesses.reverse  -- Restore program order
    barriers := finalState.barriers.reverse
  }

/-! ## Helper: Print KernelSpec for debugging -/

def printKernelSpec (spec : KernelSpec) : IO Unit := do
  IO.println s!"KernelSpec:"
  IO.println s!"  Block size: {spec.blockSize}"
  IO.println s!"  Grid size: {spec.gridSize}"
  IO.println s!"  Accesses: {spec.accesses.length}"
  for (acc, i) in spec.accesses.zipIdx do
    match acc with
    | AccessPattern.read _ loc => IO.println s!"    [{i}] Read at location {loc}"
    | AccessPattern.write _ loc => IO.println s!"    [{i}] Write at location {loc}"
  IO.println s!"  Barriers: {spec.barriers}"

end CLean.ToGPUVerifyIR
