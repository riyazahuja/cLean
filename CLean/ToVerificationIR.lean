/-
  DeviceIR to VerificationIR Translation

  Analyzes DeviceIR kernels and extracts verification metadata:
  - Memory accesses (reads and writes)
  - Barrier synchronization points
  - Uniformity information
  - Access patterns

  This is the analysis phase that prepares kernels for verification.
-/

import CLean.DeviceIR
import CLean.VerificationIR
import CLean.GPU
import Std.Data.HashMap

open DeviceIR
open GpuDSL
open CLean.VerificationIR (VerificationContext MemoryAccess MemorySpace AccessType BarrierPoint
                           UniformityInfo Uniformity AccessPattern VerifiedKernel dexprToString)

namespace CLean.ToVerificationIR

/-! ## Memory Access Extraction -/

/-- Current analysis state as we walk the IR -/
structure AnalysisState where
  /-- Accumulated memory accesses -/
  accesses : List MemoryAccess
  /-- Current statement location counter -/
  location : Nat
  /-- Barrier points found -/
  barriers : List BarrierPoint
  /-- Uniformity annotations -/
  uniformityInfo : List UniformityInfo
deriving Inhabited

/-- Extract memory accesses from an expression -/
partial def extractExprAccesses (expr : DExpr) (state : AnalysisState) (isWrite : Bool := false)
    : AnalysisState :=
  match expr with
  | DExpr.var name =>
      -- Variable access (could be array or scalar)
      let acc : MemoryAccess := {
        name := name
        space := MemorySpace.local  -- Default to local, will refine
        accessType := if isWrite then AccessType.write else AccessType.read
        index := none
        value := none
        location := state.location
        threadId := none  -- Will be filled in during dualization
      }
      { state with accesses := acc :: state.accesses }

  | DExpr.index arr idx =>
      -- Array indexing: arr[idx]
      let acc : MemoryAccess := {
        name := dexprToString arr
        space := MemorySpace.global  -- Arrays are typically global
        accessType := if isWrite then AccessType.write else AccessType.read
        index := some idx
        value := none
        location := state.location
        threadId := none
      }
      -- Also extract accesses from index expression
      let state' := extractExprAccesses idx state false
      { state' with accesses := acc :: state'.accesses }

  | DExpr.binop _ e1 e2 =>
      let state' := extractExprAccesses e1 state false
      extractExprAccesses e2 state' false

  | DExpr.unop _ e =>
      extractExprAccesses e state false

  | DExpr.intLit _ => state
  | DExpr.floatLit _ => state
  | DExpr.boolLit _ => state
  | DExpr.threadIdx _ => state
  | DExpr.blockIdx _ => state
  | DExpr.blockDim _ => state
  | DExpr.gridDim _ => state
  | _ => state

/-- Extract memory accesses from a statement -/
partial def extractStmtAccesses (stmt : DStmt) (state : AnalysisState) : AnalysisState :=
  match stmt with
  | DStmt.assign name expr =>
      -- Assignment: name := expr
      -- First extract reads from RHS
      let state' := extractExprAccesses expr state false
      -- Then add write to LHS
      let acc : MemoryAccess := {
        name := name
        space := MemorySpace.local
        accessType := AccessType.write
        index := none
        value := some expr
        location := state'.location
        threadId := none
      }
      { state' with
        accesses := acc :: state'.accesses
        location := state'.location + 1 }

  | DStmt.store arr idx val =>
      -- Array store: arr[idx] := val
      -- Extract reads from index and value
      let state' := extractExprAccesses idx state false
      let state'' := extractExprAccesses val state' false
      -- Add write access
      let acc : MemoryAccess := {
        name := dexprToString arr
        space := MemorySpace.global
        accessType := AccessType.write
        index := some idx
        value := some val
        location := state''.location
        threadId := none
      }
      { state'' with
        accesses := acc :: state''.accesses
        location := state''.location + 1 }

  | DStmt.ite cond thenStmt elseStmt =>
      -- Conditional: if cond then thenStmt else elseStmt
      let state' := extractExprAccesses cond state false
      let state'' := extractStmtAccesses thenStmt { state' with location := state'.location + 1 }
      let state''' := extractStmtAccesses elseStmt state''
      state'''

  | DStmt.seq s1 s2 =>
      -- Sequence: s1; s2
      let state' := extractStmtAccesses s1 state
      extractStmtAccesses s2 state'

  | DStmt.skip =>
      { state with location := state.location + 1 }

  | DStmt.barrier =>
      -- Barrier synchronization
      let b : BarrierPoint := {
        location := state.location
        scope := MemorySpace.shared  -- Block-level barrier
      }
      { state with
        barriers := b :: state.barriers
        location := state.location + 1 }

  | _ =>
      { state with location := state.location + 1 }

/-! ## Barrier Detection -/

/-- Find all barrier synchronization points in a statement -/
partial def findBarriers (stmt : DStmt) (location : Nat := 0) : List BarrierPoint :=
  match stmt with
  | DStmt.barrier =>
      [{ location := location, scope := MemorySpace.shared }]
  | DStmt.seq s1 s2 =>
      let barriers1 := findBarriers s1 location
      let loc2 := location + 1  -- Simplified: each statement increments location
      let barriers2 := findBarriers s2 loc2
      barriers1 ++ barriers2
  | DStmt.ite _ thenStmt elseStmt =>
      let thenBarriers := findBarriers thenStmt (location + 1)
      let elseBarriers := findBarriers elseStmt (location + 1)
      thenBarriers ++ elseBarriers
  | _ => []

/-! ## Uniformity Analysis -/

/-- Determine if an expression is thread-uniform (same value for all threads) -/
partial def isUniform (expr : DExpr) : Uniformity :=
  match expr with
  | DExpr.intLit _ => Uniformity.uniform
  | DExpr.floatLit _ => Uniformity.uniform
  | DExpr.boolLit _ => Uniformity.uniform
  | DExpr.blockIdx _ => Uniformity.uniform   -- Block ID is uniform within block
  | DExpr.blockDim _ => Uniformity.uniform   -- Block dimension is uniform
  | DExpr.gridDim _ => Uniformity.uniform    -- Grid dimension is uniform
  | DExpr.threadIdx _ => Uniformity.divergent  -- Thread ID varies per thread
  | DExpr.var _ => Uniformity.divergent  -- Conservative: assume variables may diverge
  | DExpr.index _ _ => Uniformity.divergent  -- Array accesses may diverge
  | DExpr.binop _ e1 e2 =>
      -- Uniform only if both operands are uniform
      match isUniform e1, isUniform e2 with
      | Uniformity.uniform, Uniformity.uniform => Uniformity.uniform
      | _, _ => Uniformity.divergent
  | DExpr.unop _ e => isUniform e
  | _ => Uniformity.divergent

/-- Extract uniformity information for all expressions in statement -/
partial def extractUniformity (stmt : DStmt) : List UniformityInfo :=
  match stmt with
  | DStmt.assign _ expr =>
      [{ expr := expr, uniformity := isUniform expr }]
  | DStmt.store _ idx val =>
      [{ expr := idx, uniformity := isUniform idx },
       { expr := val, uniformity := isUniform val }]
  | DStmt.ite cond thenStmt elseStmt =>
      let condInfo := { expr := cond, uniformity := isUniform cond }
      let thenInfo := extractUniformity thenStmt
      let elseInfo := extractUniformity elseStmt
      condInfo :: (thenInfo ++ elseInfo)
  | DStmt.seq s1 s2 =>
      extractUniformity s1 ++ extractUniformity s2
  | _ => []

/-! ## Access Pattern Analysis -/

/-- Analyze array access pattern to determine stride and bounds -/
def analyzeAccessPattern (expr : DExpr) : AccessPattern :=
  match expr with
  | DExpr.threadIdx Dim.x =>
      -- Unit-stride coalesced access: arr[threadIdx.x]
      { base := expr
        stride := some 1
        lowerBound := some 0
        upperBound := none }  -- Will be filled from blockDim

  | DExpr.binop BinOp.add (DExpr.threadIdx Dim.x) (DExpr.intLit offset) =>
      -- Offset coalesced access: arr[threadIdx.x + offset]
      { base := DExpr.threadIdx Dim.x
        stride := some 1
        lowerBound := some offset.toNat
        upperBound := none }

  | DExpr.binop BinOp.mul (DExpr.threadIdx _) (DExpr.intLit stride) =>
      -- Strided access: arr[threadIdx.x * stride]
      { base := expr
        stride := some stride.toNat
        lowerBound := some 0
        upperBound := none }

  | _ =>
      -- Unknown pattern
      { base := expr
        stride := none
        lowerBound := none
        upperBound := none }

/-! ## Main Translation Function -/

/-- Convert DeviceIR kernel to VerifiedKernel with full metadata -/
def toVerificationIR (kernel : Kernel) (gridDim blockDim : Dim3) : VerifiedKernel :=
  -- Initialize analysis state
  let initState : AnalysisState := {
    accesses := []
    location := 0
    barriers := []
    uniformityInfo := []
  }

  -- Extract all metadata
  let finalState := extractStmtAccesses kernel.body initState
  let barriers := findBarriers kernel.body
  let uniformityInfo := extractUniformity kernel.body

  -- Build access patterns map
  let accessPatterns : Std.HashMap String AccessPattern := Id.run do
    let mut patterns := ∅
    for acc in finalState.accesses do
      match acc.index with
      | some idx =>
          let pattern := analyzeAccessPattern idx
          patterns := patterns.insert acc.name pattern
      | none => ()
    return patterns

  -- Construct verification context
  let context : VerificationContext := {
    gridDim := gridDim
    blockDim := blockDim
    threadConstraints := []
    blockConstraints := []
  }

  -- Build final VerifiedKernel
  { ir := kernel
    context := context
    accesses := finalState.accesses.reverse  -- Restore original order
    barriers := barriers
    uniformityInfo := uniformityInfo
    accessPatterns := accessPatterns
    uniformStatements := []  -- TODO: implement statement-level uniformity
  }

/-! ## Helper Functions for Kernel Analysis -/

/-- Get summary statistics about a verified kernel -/
def kernelStats (k : VerifiedKernel) : String :=
  let numAccesses := k.accesses.length
  let numReads := k.readAccesses.length
  let numWrites := k.writeAccesses.length
  let numBarriers := k.barriers.length
  let potentialRaces := k.potentialRaces.length
  s!"Kernel: {k.ir.name}\n" ++
  s!"  Accesses: {numAccesses} ({numReads} reads, {numWrites} writes)\n" ++
  s!"  Barriers: {numBarriers}\n" ++
  s!"  Potential races: {potentialRaces}\n" ++
  s!"  Uniformity annotations: {k.uniformityInfo.length}"

/-- Print detailed kernel analysis -/
def printKernelAnalysis (k : VerifiedKernel) : IO Unit := do
  IO.println "═══════════════════════════════════════"
  IO.println s!"Verification Analysis: {k.ir.name}"
  IO.println "═══════════════════════════════════════"
  IO.println "\n[Context]"
  IO.println s!"  Grid: {k.context.gridDim.x}×{k.context.gridDim.y}×{k.context.gridDim.z}"
  IO.println s!"  Block: {k.context.blockDim.x}×{k.context.blockDim.y}×{k.context.blockDim.z}"

  IO.println "\n[Memory Accesses]"
  for acc in k.accesses do
    IO.println s!"  {acc}"

  IO.println "\n[Barriers]"
  for b in k.barriers do
    IO.println s!"  Barrier at location {b.location}"

  IO.println "\n[Potential Races]"
  let races := k.potentialRaces
  if races.isEmpty then
    IO.println "  None detected (static analysis)"
  else
    for (acc1, acc2) in races do
      IO.println s!"  {acc1.name}: loc {acc1.location} ↔ loc {acc2.location}"

  IO.println "\n[Statistics]"
  IO.println (kernelStats k)

end CLean.ToVerificationIR
