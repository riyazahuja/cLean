/-
  Verification Intermediate Representation

  Enriched IR that adds verification metadata to DeviceIR kernels.
  Supports reasoning about:
  - Thread and block dimensions
  - Memory accesses (reads and writes)
  - Barrier synchronization points
  - Control flow uniformity
  - Access patterns and bounds

  Inspired by GPUVerify's verification approach but implemented purely in Lean.
-/

import CLean.DeviceIR
import CLean.GPU
import Std.Data.HashMap

open DeviceIR
open GpuDSL

namespace CLean.VerificationIR

/-! ## Thread and Block Identifiers -/

/-- Three-dimensional thread identifier within a block -/
structure ThreadId where
  x : Nat
  y : Nat
  z : Nat
deriving Repr, DecidableEq, Hashable

/-- Three-dimensional block identifier within a grid -/
structure BlockId where
  x : Nat
  y : Nat
  z : Nat
deriving Repr, DecidableEq, Hashable

/-- Global thread identifier (combines block and thread IDs) -/
structure GlobalThreadId where
  blockId : BlockId
  threadId : ThreadId
  blockDim : Dim3
deriving Repr



def GlobalThreadId.toLinear (gid : GlobalThreadId) : Nat :=
  let blockLinear := gid.blockId.x + gid.blockId.y * 1000 + gid.blockId.z * 1000000
  let threadLinear := gid.threadId.x + gid.threadId.y * gid.blockDim.x +
                      gid.threadId.z * gid.blockDim.x * gid.blockDim.y
  blockLinear * (gid.blockDim.x * gid.blockDim.y * gid.blockDim.z) + threadLinear

/-! ## Verification Context -/

/-- Execution context for verification: grid/block dimensions and constraints -/
structure VerificationContext where
  gridDim : Dim3
  blockDim : Dim3
  /-- Additional constraints on thread indices (e.g., bounds, uniformity) -/
  threadConstraints : List (ThreadId → Prop)
  /-- Additional constraints on block indices -/
  blockConstraints : List (BlockId → Prop)

instance : Inhabited VerificationContext where
  default := {
    gridDim := ⟨1, 1, 1⟩
    blockDim := ⟨1, 1, 1⟩
    threadConstraints := []
    blockConstraints := []
  }

def VerificationContext.inBounds (ctx : VerificationContext) (gid : GlobalThreadId) : Prop :=
  gid.threadId.x < ctx.blockDim.x ∧
  gid.threadId.y < ctx.blockDim.y ∧
  gid.threadId.z < ctx.blockDim.z ∧
  gid.blockId.x < ctx.gridDim.x ∧
  gid.blockId.y < ctx.gridDim.y ∧
  gid.blockId.z < ctx.gridDim.z

def VerificationContext.sameBlock (_ctx : VerificationContext) (t1 t2 : GlobalThreadId) : Prop :=
  t1.blockId = t2.blockId

/-! ## Memory Access Representation -/

/-- Type of memory access -/
inductive AccessType where
  | read : AccessType
  | write : AccessType
deriving Repr, DecidableEq

/-- Memory space being accessed -/
inductive MemorySpace where
  | global : MemorySpace    -- Global device memory
  | shared : MemorySpace    -- Shared memory within block
  | local : MemorySpace     -- Thread-local memory
deriving Repr, DecidableEq

/-- A memory access operation with full metadata -/
structure MemoryAccess where
  /-- Array or variable name being accessed -/
  name : String
  /-- Memory space (global, shared, local) -/
  space : MemorySpace
  /-- Type of access (read or write) -/
  accessType : AccessType
  /-- Index expression (for arrays) -/
  index : Option DExpr
  /-- Value being written (for writes) -/
  value : Option DExpr
  /-- Source location in kernel (statement number) -/
  location : Nat
  /-- Which thread performs this access (symbolic) -/
  threadId : Option GlobalThreadId
deriving Repr

def MemoryAccess.isRead (acc : MemoryAccess) : Bool :=
  match acc.accessType with
  | AccessType.read => true
  | AccessType.write => false

def MemoryAccess.isWrite (acc : MemoryAccess) : Bool :=
  match acc.accessType with
  | AccessType.write => true
  | AccessType.read => false

/-! ## Access Conflicts -/

/-- Two accesses conflict if they access the same location and at least one is a write -/
def MemoryAccess.conflicts (acc1 acc2 : MemoryAccess) : Prop :=
  acc1.name = acc2.name ∧
  acc1.space = acc2.space ∧
  acc1.space ≠ MemorySpace.local ∧
  (acc1.isWrite || acc2.isWrite) ∧
  -- Indices must potentially overlap (we'll refine this with SMT-style reasoning)
  match acc1.index, acc2.index with
  | some idx1, some idx2 => idx1 = idx2  -- For now, exact equality (will extend to "may alias")
  | none, none => true  -- Scalar accesses always conflict
  | _, _ => false
/-- Boolean version of conflicts for code generation -/
def MemoryAccess.conflictsBool (acc1 acc2 : MemoryAccess) : Bool :=
  acc1.name == acc2.name &&
  acc1.space == acc2.space &&
  acc1.space != MemorySpace.local &&
  (acc1.isWrite || acc2.isWrite) &&
  match acc1.index, acc2.index with
  | some idx1, some idx2 => idx1 == idx2
  | none, none => true
  | _, _ => false

/-- Two accesses form a data race if they conflict and are from different threads -/
def MemoryAccess.isRace (acc1 acc2 : MemoryAccess) : Prop :=
  acc1.conflicts acc2 ∧
  match acc1.threadId, acc2.threadId with
  | some t1, some t2 => t1 ≠ t2
  | _, _ => false  -- Can't race if thread IDs unknown

/-! ## Barrier Synchronization -/

/-- Barrier synchronization point in kernel -/
structure BarrierPoint where
  /-- Location in kernel (statement number) -/
  location : Nat
  /-- Scope: which threads synchronize (currently only block-level) -/
  scope : MemorySpace  -- .shared means block-level barrier
deriving Repr, DecidableEq

/-! ## Uniformity Analysis -/

/-- Uniformity: whether an expression has the same value across all threads -/
inductive Uniformity where
  | uniform : Uniformity      -- Same value for all threads in scope
  | divergent : Uniformity    -- May differ between threads
deriving Repr, DecidableEq, Inhabited

/-- Uniformity annotation for expressions -/
structure UniformityInfo where
  expr : DExpr
  uniformity : Uniformity
deriving Repr

/-! ## Access Pattern Analysis -/

/-- Describes how a thread accesses an array -/
structure AccessPattern where
  /-- Base index (e.g., threadIdx.x for unit-stride) -/
  base : DExpr
  /-- Stride between accesses (e.g., 1 for coalesced, blockDim.x for strided) -/
  stride : Option Nat
  /-- Bounds on index (for safety checking) -/
  lowerBound : Option Nat
  upperBound : Option Nat
deriving Repr

/-! ## Verified Kernel Representation -/

/-- Kernel enriched with verification metadata -/
structure VerifiedKernel where
  /-- Original DeviceIR kernel -/
  ir : Kernel
  /-- Verification context (dimensions, constraints) -/
  context : VerificationContext
  /-- All memory accesses in the kernel -/
  accesses : List MemoryAccess
  /-- Barrier synchronization points -/
  barriers : List BarrierPoint
  /-- Uniformity information for key expressions -/
  uniformityInfo : List UniformityInfo
  /-- Access patterns for array operations -/
  accessPatterns : Std.HashMap String AccessPattern
  /-- Statement-level metadata (which statements are thread-uniform) -/
  uniformStatements : List Nat

instance : Inhabited VerifiedKernel where
  default := {
    ir := default
    context := default
    accesses := []
    barriers := []
    uniformityInfo := []
    accessPatterns := ∅
    uniformStatements := []
  }

/-! ## Helper Functions -/

/-- Check if a statement location is before a barrier -/
def VerifiedKernel.beforeBarrier (k : VerifiedKernel) (loc : Nat) : Option BarrierPoint :=
  k.barriers.find? (fun b => b.location > loc)

/-- Get all accesses to a specific array -/
def VerifiedKernel.accessesTo (k : VerifiedKernel) (name : String) : List MemoryAccess :=
  k.accesses.filter (fun acc => acc.name = name)

/-- Get all write accesses -/
def VerifiedKernel.writeAccesses (k : VerifiedKernel) : List MemoryAccess :=
  k.accesses.filter (fun acc => acc.isWrite)

/-- Get all read accesses -/
def VerifiedKernel.readAccesses (k : VerifiedKernel) : List MemoryAccess :=
  k.accesses.filter (fun acc => acc.isRead)

/-- Find potential races in kernel -/
def VerifiedKernel.potentialRaces (k : VerifiedKernel) : List (MemoryAccess × MemoryAccess) :=
  let writes := k.writeAccesses
  let allAccesses := k.accesses
  -- Note: This is a conservative approximation - may include non-races
  Id.run do
    let mut races := []
    for w in writes do
      for acc in allAccesses do
        -- Conservative: check if they might conflict (same name, at least one write)
        if w.name == acc.name && w.space == acc.space && (w.isWrite || acc.isWrite) then
          races := (w, acc) :: races
    return races

/-! ## Happens-Before Relation -/

/-- Ordering between memory accesses (for race detection) -/
inductive HappensBeforeReason where
  | programOrder : HappensBeforeReason      -- Same thread, sequential execution
  | barrier : BarrierPoint → HappensBeforeReason  -- Barrier synchronization
  | transitivity : HappensBeforeReason → HappensBeforeReason → HappensBeforeReason
deriving Repr

/-- acc1 happens-before acc2 if there's a guaranteed ordering -/
inductive HappensBefore (k : VerifiedKernel) : MemoryAccess → MemoryAccess → Prop where
  | programOrder :
      ∀ acc1 acc2 : MemoryAccess,
      acc1.threadId = acc2.threadId →
      acc1.location < acc2.location →
      HappensBefore k acc1 acc2

  | barrierSync :
      ∀ acc1 acc2 : MemoryAccess,
      ∀ b : BarrierPoint,
      b ∈ k.barriers →
      acc1.location < b.location →
      b.location < acc2.location →
      -- Both threads are in same block (barrier scope)
      (match acc1.threadId, acc2.threadId with
       | some t1, some t2 => k.context.sameBlock t1 t2
       | _, _ => False) →
      HappensBefore k acc1 acc2

  | transitivity :
      ∀ acc1 acc2 acc3 : MemoryAccess,
      HappensBefore k acc1 acc2 →
      HappensBefore k acc2 acc3 →
      HappensBefore k acc1 acc3

/-- Two accesses are concurrent if neither happens-before the other -/
def MemoryAccess.concurrent (k : VerifiedKernel) (acc1 acc2 : MemoryAccess) : Prop :=
  ¬(HappensBefore k acc1 acc2) ∧ ¬(HappensBefore k acc2 acc1)

/-- A race exists if two accesses conflict and are concurrent -/
def MemoryAccess.hasRace (k : VerifiedKernel) (acc1 acc2 : MemoryAccess) : Prop :=
  acc1.conflicts acc2 ∧ acc1.concurrent k acc2

/-! ## Example Construction -/

/-- Create a simple verification context for testing -/
def exampleContext : VerificationContext :=
  { gridDim := ⟨1, 1, 1⟩
    blockDim := ⟨256, 1, 1⟩
    threadConstraints := []
    blockConstraints := [] }

/-! ## Pretty Printing -/

/-- Simple DExpr to string conversion (for debugging) -/
partial def dexprToString : DExpr → String
  | DExpr.intLit n => toString n
  | DExpr.floatLit f => toString f
  | DExpr.boolLit b => toString b
  | DExpr.var name => name
  | DExpr.threadIdx Dim.x => "threadIdx.x"
  | DExpr.threadIdx Dim.y => "threadIdx.y"
  | DExpr.threadIdx Dim.z => "threadIdx.z"
  | DExpr.blockIdx Dim.x => "blockIdx.x"
  | DExpr.blockIdx Dim.y => "blockIdx.y"
  | DExpr.blockIdx Dim.z => "blockIdx.z"
  | DExpr.blockDim Dim.x => "blockDim.x"
  | DExpr.blockDim Dim.y => "blockDim.y"
  | DExpr.blockDim Dim.z => "blockDim.z"
  | DExpr.gridDim Dim.x => "gridDim.x"
  | DExpr.gridDim Dim.y => "gridDim.y"
  | DExpr.gridDim Dim.z => "gridDim.z"
  | DExpr.binop BinOp.add e1 e2 => s!"({dexprToString e1} + {dexprToString e2})"
  | DExpr.binop BinOp.sub e1 e2 => s!"({dexprToString e1} - {dexprToString e2})"
  | DExpr.binop BinOp.mul e1 e2 => s!"({dexprToString e1} * {dexprToString e2})"
  | DExpr.binop BinOp.div e1 e2 => s!"({dexprToString e1} / {dexprToString e2})"
  | DExpr.binop BinOp.lt e1 e2 => s!"({dexprToString e1} < {dexprToString e2})"
  | DExpr.binop _ e1 e2 => s!"({dexprToString e1} OP {dexprToString e2})"
  | DExpr.index arr idx => s!"{dexprToString arr}[{dexprToString idx}]"
  | _ => "<expr>"

def MemoryAccess.toString (acc : MemoryAccess) : String :=
  let accessStr := match acc.accessType with
    | AccessType.read => "READ"
    | AccessType.write => "WRITE"
  let indexStr := match acc.index with
    | some idx => s!"[{dexprToString idx}]"
    | none => ""
  let valueStr := match acc.value with
    | some v => s!" := {dexprToString v}"
    | none => ""
  s!"{accessStr} {acc.name}{indexStr}{valueStr} @ loc {acc.location}"

instance : ToString MemoryAccess where
  toString := MemoryAccess.toString

attribute [ext] GlobalThreadId ThreadId BlockId MemoryAccess VerificationContext BarrierPoint UniformityInfo AccessPattern VerifiedKernel

end CLean.VerificationIR
