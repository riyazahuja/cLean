/-
  Verification Tactics and Helper Lemmas

  Provides tactics and lemmas to help users prove verification conditions:
  - Thread index reasoning
  - Happens-before proofs
  - Memory access analysis
  - Barrier synchronization
  - Bounds checking

  These tools make interactive proof writing easier.
-/

import CLean.VerificationIR
import CLean.Verification.SafetyProperties
import Lean
import Mathlib.Tactic

open CLean.VerificationIR
open CLean.Verification.SafetyProperties
open DeviceIR
open Lean.Elab.Tactic

namespace CLean.Verification.Tactics

/-! ## Basic Lemmas about Thread Indices -/

/-- Thread indices are bounded by block dimensions -/
theorem threadIdx_bounded (ctx : VerificationContext) (t : GlobalThreadId) (h : ctx.inBounds t) :
    t.threadId.x < ctx.blockDim.x ∧
    t.threadId.y < ctx.blockDim.y ∧
    t.threadId.z < ctx.blockDim.z := by
  unfold VerificationContext.inBounds at h
  exact ⟨h.1, h.2.1, h.2.2.1⟩

/-- Different threads in same block have different thread IDs -/
theorem distinct_threads_diff_ids (t1 t2 : GlobalThreadId) :
    t1 ≠ t2 →
    t1.blockId = t2.blockId →
    t1.blockDim = t2.blockDim →
    t1.threadId ≠ t2.threadId := by
  intro h_neq h_same_block h_same_dim
  by_contra h_same_tid
  have : t1 = t2 := by
    cases t1; cases t2
    simp [GlobalThreadId.mk.injEq] at *
    exact ⟨h_same_block, h_same_tid, h_same_dim⟩
  exact h_neq this

/-- Threads with different x-coordinates are distinct -/
theorem threadIdx_x_distinct (t1 t2 : GlobalThreadId) :
    t1.blockId = t2.blockId →
    t1.threadId.x ≠ t2.threadId.x →
    t1 ≠ t2 := by
  intro h_block h_tid
  by_contra h_eq
  rw [h_eq] at h_tid
  exact h_tid rfl

/-! ## Happens-Before Lemmas -/

/-- Happens-before is irreflexive -/
theorem happensBefore_irrefl (k : VerifiedKernel) (acc : MemoryAccess) :
    ¬HappensBefore k acc acc := by
  intro h
  cases h with
  | programOrder acc1 acc2 h_tid h_loc =>
      -- acc.location < acc.location is false
      omega
  | barrierSync => sorry
  | transitivity => sorry

/-- Happens-before is transitive (already in definition, but restated) -/
theorem happensBefore_trans (k : VerifiedKernel) (a1 a2 a3 : MemoryAccess) :
    HappensBefore k a1 a2 →
    HappensBefore k a2 a3 →
    HappensBefore k a1 a3 :=
  HappensBefore.transitivity a1 a2 a3

/-- If two accesses are from same thread and ordered, happens-before holds -/
theorem sameThread_ordered_happensBefore
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess)
    (h_tid : acc1.threadId = acc2.threadId)
    (h_loc : acc1.location < acc2.location) :
    HappensBefore k acc1 acc2 :=
  HappensBefore.programOrder acc1 acc2 h_tid h_loc

/-- Barrier synchronization establishes happens-before -/
theorem barrier_establishes_happensBefore
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess)
    (b : BarrierPoint)
    (h_barrier : b ∈ k.barriers)
    (h_before : acc1.location < b.location)
    (h_after : b.location < acc2.location)
    (h_same_block : match acc1.threadId, acc2.threadId with
                    | some t1, some t2 => k.context.sameBlock t1 t2
                    | _, _ => False) :
    HappensBefore k acc1 acc2 :=
  HappensBefore.barrierSync acc1 acc2 b h_barrier h_before h_after h_same_block

/-! ## Race Freedom Lemmas -/

/-- If happens-before holds, no race -/
theorem happensBefore_implies_no_race
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess) :
    HappensBefore k acc1 acc2 →
    ¬acc1.hasRace k acc2 := by
  intro h_hb h_race
  unfold MemoryAccess.hasRace at h_race
  unfold MemoryAccess.concurrent at h_race
  exact h_race.2.1 h_hb

/-- Symmetric: if happens-before holds other way, no race -/
theorem happensBefore_sym_implies_no_race
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess) :
    HappensBefore k acc2 acc1 →
    ¬acc1.hasRace k acc2 := by
  intro h_hb h_race
  unfold MemoryAccess.hasRace at h_race
  unfold MemoryAccess.concurrent at h_race
  exact h_race.2.2 h_hb

/-- If accesses don't conflict, they can't race -/
theorem no_conflict_implies_no_race
    (k : VerifiedKernel)
    (acc1 acc2 : MemoryAccess)
    (h : ¬acc1.conflicts acc2) :
    ¬acc1.hasRace k acc2 := by
  intro h_race
  unfold MemoryAccess.hasRace at h_race
  exact h h_race.1

/-- Coalesced array accesses with distinct thread indices don't conflict -/
theorem coalesced_access_no_conflict
    (acc1 acc2 : MemoryAccess)
    (h_name : acc1.name = acc2.name)
    (h_space : acc1.space = acc2.space)
    (h_idx1 : acc1.index = some (DExpr.threadIdx Dim.x))
    (h_idx2 : acc2.index = some (DExpr.threadIdx Dim.x))
    (h_diff_threads : match acc1.threadId, acc2.threadId with
                      | some t1, some t2 => t1.threadId.x ≠ t2.threadId.x
                      | _, _ => False) :
    ¬acc1.conflicts acc2 := by
  sorry  -- Proof: different thread indices => different array indices => no conflict

/-! ## Memory Safety Lemmas -/

/-- Index expression with thread ID is bounded by block dimension -/
theorem threadIdx_access_bounded
    (ctx : VerificationContext)
    (t : GlobalThreadId)
    (h_inbounds : ctx.inBounds t)
    (idx : DExpr)
    (h_idx : idx = DExpr.threadIdx Dim.x) :
    ∃ n : Nat, n = t.threadId.x ∧ n < ctx.blockDim.x := by
  exists t.threadId.x
  constructor
  · rfl
  · exact h_inbounds.1

/-- Array access arr[threadIdx.x] is safe if array size ≥ blockDim.x -/
theorem threadIdx_access_safe
    (k : VerifiedKernel)
    (acc : MemoryAccess)
    (arraySize : Nat)
    (h_idx : acc.index = some (DExpr.threadIdx Dim.x))
    (h_size : k.context.blockDim.x ≤ arraySize) :
    ArrayBoundsSafe k acc.name arraySize := by
  sorry  -- Proof: threadIdx.x < blockDim.x ≤ arraySize

/-- Offset access arr[threadIdx.x + c] is safe if arraySize ≥ blockDim.x + c -/
theorem offset_access_safe
    (k : VerifiedKernel)
    (acc : MemoryAccess)
    (offset arraySize : Nat)
    (h_idx : acc.index = some (DExpr.binop BinOp.add
                                  (DExpr.threadIdx Dim.x)
                                  (DExpr.intLit (Int.ofNat offset))))
    (h_size : k.context.blockDim.x + offset ≤ arraySize) :
    ArrayBoundsSafe k acc.name arraySize := by
  sorry  -- Proof: threadIdx.x + offset < blockDim.x + offset ≤ arraySize

/-! ## Barrier Divergence Lemmas -/

/-- If barrier is in thread-uniform location, no divergence -/
theorem uniform_barrier_no_divergence
    (k : VerifiedKernel)
    (b : BarrierPoint)
    (h_barrier : b ∈ k.barriers)
    (h_uniform : b.location ∈ k.uniformStatements) :
    -- All threads reach it
    ∀ t : GlobalThreadId, k.context.inBounds t →
      True  -- Placeholder: need execution model to express "reaches"
    := by sorry

/-! ## Combined Safety Lemmas -/

/-- Kernel with no barriers and coalesced accesses is race-free -/
theorem coalesced_no_barriers_safe
    (k : VerifiedKernel)
    (h_no_barriers : k.barriers = [])
    (h_coalesced : ∀ acc1 acc2 : MemoryAccess,
        acc1 ∈ k.accesses →
        acc2 ∈ k.accesses →
        acc1 ≠ acc2 →
        acc1.index = some (DExpr.threadIdx Dim.x) →
        acc2.index = some (DExpr.threadIdx Dim.x) →
        match acc1.threadId, acc2.threadId with
        | some t1, some t2 => t1.threadId.x ≠ t2.threadId.x
        | _, _ => False) :
    RaceFree k := by
  sorry

/-- Solve race freedom boilerplate -/
syntax "solve_race_freedom" ("[" Lean.Parser.Tactic.simpLemma,* "]")? : tactic

macro_rules
  | `(tactic| solve_race_freedom) =>
    `(tactic| (
      unfold RaceFree;
      intros env t1 t2 h_in1 h_in2 acc1 acc2 h_mem1 h_mem2 h_name h_write h_idx_eq;
      simp at h_mem1 h_mem2;
      constructor;
      · intro h_neq;
        repeat (first | (rcases h_mem1 with rfl) | (rcases h_mem1 with rfl | h_mem1) | contradiction);
        repeat (first | (rcases h_mem2 with rfl) | (rcases h_mem2 with rfl | h_mem2) | contradiction);
        all_goals (
          try contradiction;
          try simp [evalDExpr] at h_idx_eq
        )
      · intro h_eq h_acc_neq;
        subst h_eq;
        repeat (first | (rcases h_mem1 with rfl) | (rcases h_mem1 with rfl | h_mem1) | contradiction);
        repeat (first | (rcases h_mem2 with rfl) | (rcases h_mem2 with rfl | h_mem2) | contradiction);
        all_goals (
          try contradiction;
          try (apply Or.inl; apply HappensBefore.programOrder; rfl; decide);
          try (apply Or.inr; apply HappensBefore.programOrder; rfl; decide);
          try contradiction
        )
    ))
  | `(tactic| solve_race_freedom [ $args,* ]) =>
    `(tactic| (
      unfold RaceFree;
      intros env t1 t2 h_in1 h_in2 acc1 acc2 h_mem1 h_mem2 h_name h_write h_idx_eq;
      simp [$args,*] at h_mem1 h_mem2;
      constructor;
      · intro h_neq;
        repeat (first | (rcases h_mem1 with rfl) | (rcases h_mem1 with rfl | h_mem1) | contradiction);
        repeat (first | (rcases h_mem2 with rfl) | (rcases h_mem2 with rfl | h_mem2) | contradiction);
        all_goals (
          try contradiction;
          try simp [evalDExpr] at h_idx_eq
        )
      · intro h_eq h_acc_neq;
        subst h_eq;
        repeat (first | (rcases h_mem1 with rfl) | (rcases h_mem1 with rfl | h_mem1) | contradiction);
        repeat (first | (rcases h_mem2 with rfl) | (rcases h_mem2 with rfl | h_mem2) | contradiction);
        all_goals (
          try contradiction;
          try (apply Or.inl; apply HappensBefore.programOrder; rfl; decide);
          try (apply Or.inr; apply HappensBefore.programOrder; rfl; decide);
          try contradiction
        )
    ))

/-- Solve memory safety boilerplate -/
syntax "solve_memory_safety" ("[" Lean.Parser.Tactic.simpLemma,* "]")? : tactic

macro_rules
  | `(tactic| solve_memory_safety) =>
    `(tactic| (
      unfold MemorySafe;
      intro acc h_in;
      simp at h_in;
      repeat (first | (rcases h_in with rfl) | (rcases h_in with rfl | h_in) | contradiction);
      all_goals (
        try contradiction;
        simp;
        intro arraySize;
        exists 0;
      )
    ))
  | `(tactic| solve_memory_safety [ $args,* ]) =>
    `(tactic| (
      unfold MemorySafe;
      intro acc h_in;
      simp [$args,*] at h_in;
      repeat (first | (rcases h_in with rfl) | (rcases h_in with rfl | h_in) | contradiction);
      all_goals (
        try contradiction;
        simp;
        intro arraySize;
        exists 0;
      )
    ))

/-! ## Proof Automation Helpers -/

/-- Simplify memory access predicates -/
syntax "simplify_access" : tactic

macro_rules
  | `(tactic| simplify_access) =>
    `(tactic| (unfold MemoryAccess.conflicts MemoryAccess.isRead MemoryAccess.isWrite;
               simp [*]))

/-- Apply happens-before reasoning -/
syntax "apply_happens_before" : tactic

macro_rules
  | `(tactic| apply_happens_before) =>
    `(tactic| (first
               | apply sameThread_ordered_happensBefore
               | apply barrier_establishes_happensBefore
               | apply happensBefore_trans))

/-- Prove no-race using common patterns -/
syntax "prove_no_race" : tactic

macro_rules
  | `(tactic| prove_no_race) =>
    `(tactic| (first
               | apply no_conflict_implies_no_race
               | apply happensBefore_implies_no_race; apply_happens_before
               | apply happensBefore_sym_implies_no_race; apply_happens_before))

/-- Prove bounds safety for standard access patterns -/
syntax "prove_bounds" : tactic

macro_rules
  | `(tactic| prove_bounds) =>
    `(tactic| (first
               | apply threadIdx_access_safe; omega
               | apply offset_access_safe; omega))

/-! ## Example Proof Patterns -/

/-- Example: Prove two sequential accesses from same thread don't race -/
example (k : VerifiedKernel) (acc1 acc2 : MemoryAccess)
    (h1 : acc1 ∈ k.accesses)
    (h2 : acc2 ∈ k.accesses)
    (h_tid : acc1.threadId = acc2.threadId)
    (h_loc : acc1.location < acc2.location) :
    ¬acc1.hasRace k acc2 := by
  apply happensBefore_implies_no_race
  apply sameThread_ordered_happensBefore <;> assumption

/-- Example: Prove coalesced accesses don't race -/
example (k : VerifiedKernel) (acc1 acc2 : MemoryAccess)
    (h_idx1 : acc1.index = some (DExpr.threadIdx Dim.x))
    (h_idx2 : acc2.index = some (DExpr.threadIdx Dim.x))
    (h_diff : match acc1.threadId, acc2.threadId with
              | some t1, some t2 => t1.threadId.x ≠ t2.threadId.x
              | _, _ => False) :
    ¬acc1.conflicts acc2 := by
  sorry  -- Would use coalesced_access_no_conflict lemma

end CLean.Verification.Tactics
