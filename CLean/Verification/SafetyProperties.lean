/-
  GPU Kernel Safety Properties

  Formal definitions of safety properties for GPU kernels:
  - Race Freedom: No data races between threads
  - Memory Safety: All accesses within bounds
  - Barrier Divergence Freedom: All threads reach barriers uniformly
  - Deadlock Freedom: No circular dependencies

  These properties form the verification obligations that users must prove.
-/

import CLean.VerificationIR
import CLean.DeviceIR
import CLean.GPU

open CLean.VerificationIR
open DeviceIR
open GpuDSL

namespace CLean.Verification.SafetyProperties

/-! ## Expression Evaluation -/

/-- Environment for variable valuation -/
def Environment := String → Int

/-- Evaluate a DExpr in a given context (thread + environment) -/
def evalDExpr (e : DExpr) (t : GlobalThreadId) (env : Environment) : Int :=
  match e with
  | DExpr.intLit n => n
  | DExpr.floatLit _ => 0 -- Simplified: ignore floats for indexing
  | DExpr.boolLit b => if b then 1 else 0
  | DExpr.var name => env name
  | DExpr.threadIdx Dim.x => t.threadId.x
  | DExpr.threadIdx Dim.y => t.threadId.y
  | DExpr.threadIdx Dim.z => t.threadId.z
  | DExpr.blockIdx Dim.x => t.blockId.x
  | DExpr.blockIdx Dim.y => t.blockId.y
  | DExpr.blockIdx Dim.z => t.blockId.z
  | DExpr.blockDim Dim.x => t.blockDim.x
  | DExpr.blockDim Dim.y => t.blockDim.y
  | DExpr.blockDim Dim.z => t.blockDim.z
  | DExpr.gridDim Dim.x => 1 -- Simplified
  | DExpr.gridDim Dim.y => 1
  | DExpr.gridDim Dim.z => 1
  | DExpr.binop BinOp.add e1 e2 => evalDExpr e1 t env + evalDExpr e2 t env
  | DExpr.binop BinOp.sub e1 e2 => evalDExpr e1 t env - evalDExpr e2 t env
  | DExpr.binop BinOp.mul e1 e2 => evalDExpr e1 t env * evalDExpr e2 t env
  | DExpr.binop BinOp.div e1 e2 => evalDExpr e1 t env / evalDExpr e2 t env
  | _ => 0

/-! ## Race Freedom -/

/-- No data races: conflicting accesses must be ordered by happens-before -/
def RaceFree (k : VerifiedKernel) : Prop :=
  ∀ (env : Environment),
  ∀ t1 t2 : GlobalThreadId,
    k.context.inBounds t1 →
    k.context.inBounds t2 →
    ∀ acc1 acc2 : MemoryAccess,
      acc1 ∈ k.accesses →
      acc2 ∈ k.accesses →
      acc1.name = acc2.name →
      (acc1.isWrite ∨ acc2.isWrite) →
      -- Semantic conflict: indices evaluate to same value
      (match acc1.index, acc2.index with
       | some idx1, some idx2 => evalDExpr idx1 t1 env = evalDExpr idx2 t2 env
       | none, none => True
       | _, _ => False) →
      -- If different threads, must be ordered (impossible without barriers)
      (t1 ≠ t2 → False) ∧
      -- If same thread, must be ordered by program order
      (t1 = t2 → acc1 ≠ acc2 → HappensBefore k acc1 acc2 ∨ HappensBefore k acc2 acc1)

/-- Specific: No write-write races -/
def NoWriteWriteRaces (k : VerifiedKernel) : Prop :=
  ∀ w1 w2 : MemoryAccess,
    w1 ∈ k.writeAccesses →
    w2 ∈ k.writeAccesses →
    w1 ≠ w2 →
    w1.name = w2.name →
    -- Indices may overlap
    (∃ (idx : DExpr), w1.index = some idx ∧ w2.index = some idx) →
    -- Must be ordered
    (HappensBefore k w1 w2 ∨ HappensBefore k w2 w1)

/-- Specific: No read-write races -/
def NoReadWriteRaces (k : VerifiedKernel) : Prop :=
  ∀ r : MemoryAccess, ∀ w : MemoryAccess,
    r ∈ k.readAccesses →
    w ∈ k.writeAccesses →
    r.name = w.name →
    (∃ (idx : DExpr), r.index = some idx ∧ w.index = some idx) →
    (HappensBefore k r w ∨ HappensBefore k w r)

/-! ## Memory Safety -/

/-- All array accesses are within bounds -/
def MemorySafe (k : VerifiedKernel) : Prop :=
  ∀ acc : MemoryAccess,
    acc ∈ k.accesses →
    match acc.index with
    | none => True  -- Scalar accesses are always safe
    | some idx =>
        -- Index must be within array bounds
        ∀ (arraySize : Nat),
          -- We need to know the array size (will be provided as axiom/assumption)
          ∃ (idxVal : Nat),
            -- Index evaluates to some concrete value
            idx = DExpr.intLit (Int.ofNat idxVal) →
            0 ≤ idxVal ∧ idxVal < arraySize

/-- Stronger property: No out-of-bounds accesses for specific array -/
def ArrayBoundsSafe (k : VerifiedKernel) (arrayName : String) (arraySize : Nat) : Prop :=
  ∀ acc : MemoryAccess,
    acc ∈ k.accesses →
    acc.name = arrayName →
    match acc.index with
    | none => True
    | some idx =>
        -- For all threads executing this access
        ∀ (t : GlobalThreadId),
          k.context.inBounds t →
          -- The index computed by thread t is in bounds
          ∃ (idxVal : Nat),
            -- (Abstract: idx evaluated by thread t = idxVal)
            0 ≤ idxVal ∧ idxVal < arraySize

/-- No null pointer dereferences (for pointer-based kernels) -/
def NoNullDereference (k : VerifiedKernel) : Prop :=
  ∀ acc : MemoryAccess,
    acc ∈ k.accesses →
    acc.space = VerificationIR.MemorySpace.global →
    -- Pointer/array must be non-null (axiomatized)
    True  -- Placeholder: will need pointer analysis

/-! ## Barrier Divergence Freedom -/

/-- All threads in a block reach all barriers (no divergence) -/
def BarrierDivergenceFree (k : VerifiedKernel) : Prop :=
  ∀ b : BarrierPoint,
    b ∈ k.barriers →
    -- All threads in a block must reach barrier b
    ∀ (t : GlobalThreadId),
      k.context.inBounds t →
      -- Thread t reaches barrier b in its execution
      ∃ (path : List Nat),
        -- path is a valid execution path from entry to barrier
        b.location ∈ path

/-- Barriers are only in thread-uniform control flow -/
def BarriersUniform (k : VerifiedKernel) : Prop :=
  ∀ b : BarrierPoint,
    b ∈ k.barriers →
    -- The barrier location must be in a uniform statement
    b.location ∈ k.uniformStatements

/-- No deadlock: barriers form a total order (no circular dependencies) -/
def NoBarrierDeadlock (k : VerifiedKernel) : Prop :=
  ∀ b1 b2 : BarrierPoint,
    b1 ∈ k.barriers →
    b2 ∈ k.barriers →
    b1 ≠ b2 →
    -- Either b1 < b2 or b2 < b1 (total order)
    (b1.location < b2.location ∨ b2.location < b1.location)

/-! ## Atomicity Properties -/

/-- Atomic operations are correctly used (no torn reads/writes) -/
def AtomicOperationsSafe (k : VerifiedKernel) : Prop :=
  ∀ acc : MemoryAccess,
    acc ∈ k.accesses →
    -- If accessing shared memory with potential conflicts
    acc.space = VerificationIR.MemorySpace.shared →
    (∃ acc2 : MemoryAccess, acc2 ∈ k.accesses ∧ acc.conflicts acc2) →
    -- Then operation must be atomic (we'd need to track this in IR)
    True  -- Placeholder: requires atomic annotation in IR

/-! ## Combined Safety Property -/

/-- A kernel is safe if it satisfies all safety properties -/
def KernelSafe (k : VerifiedKernel) : Prop :=
  RaceFree k ∧
  MemorySafe k ∧
  BarrierDivergenceFree k ∧
  NoBarrierDeadlock k

/-! ## Specific Array Safety (for easier proofs) -/

/-- Safety for a specific array with known size -/
def ArraySafe (k : VerifiedKernel) (arrayName : String) (arraySize : Nat) : Prop :=
  ArrayBoundsSafe k arrayName arraySize ∧
  -- No races on this specific array
  (∀ acc1 acc2 : MemoryAccess,
    acc1 ∈ k.accesses →
    acc2 ∈ k.accesses →
    acc1.name = arrayName →
    acc2.name = arrayName →
    acc1 ≠ acc2 →
    acc1.conflicts acc2 →
    (HappensBefore k acc1 acc2 ∨ HappensBefore k acc2 acc1))

/-! ## Helper Lemmas for Proving Safety -/

/-- If all statements are between barriers, then program is race-free in that region -/
theorem barrierRegionRaceFree
    (k : VerifiedKernel)
    (b1 b2 : BarrierPoint)
    (h1 : b1 ∈ k.barriers)
    (h2 : b2 ∈ k.barriers)
    (h3 : b1.location < b2.location)
    : ∀ acc1 acc2 : MemoryAccess,
        acc1 ∈ k.accesses →
        acc2 ∈ k.accesses →
        b1.location < acc1.location →
        acc1.location < b2.location →
        b1.location < acc2.location →
        acc2.location < b2.location →
        acc1.conflicts acc2 →
        HappensBefore k acc1 acc2 ∨ HappensBefore k acc2 acc1 := by
  sorry  -- To be proved interactively

/-- Thread-uniform accesses don't race -/
theorem uniformAccessesNoRace
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess)
    (h1 : acc1 ∈ k.accesses)
    (h2 : acc2 ∈ k.accesses)
    (uniform1 : ∃ info : UniformityInfo, info ∈ k.uniformityInfo ∧
                 info.expr = acc1.index.getD (DExpr.intLit 0) ∧
                 info.uniformity = Uniformity.uniform)
    (uniform2 : ∃ info : UniformityInfo, info ∈ k.uniformityInfo ∧
                 info.expr = acc2.index.getD (DExpr.intLit 0) ∧
                 info.uniformity = Uniformity.uniform)
    : acc1.index = acc2.index → ¬acc1.conflicts acc2 := by
  sorry  -- To be proved interactively

/-- Coalesced accesses with unit stride don't conflict -/
theorem coalescedAccessesNoConflict
    (k : VerifiedKernel)
    (arrayName : String)
    (pattern : AccessPattern)
    (h1 : k.accessPatterns.get? arrayName = some pattern)
    (h2 : pattern.stride = some 1)
    (h3 : pattern.base = DExpr.threadIdx Dim.x)
    : ∀ acc1 acc2 : MemoryAccess,
        acc1 ∈ k.accesses →
        acc2 ∈ k.accesses →
        acc1.name = arrayName →
        acc2.name = arrayName →
        -- Different threads
        (match acc1.threadId, acc2.threadId with
         | some t1, some t2 => t1.threadId ≠ t2.threadId
         | _, _ => False) →
        ¬acc1.conflicts acc2 := by
  sorry  -- To be proved interactively

end CLean.Verification.SafetyProperties
