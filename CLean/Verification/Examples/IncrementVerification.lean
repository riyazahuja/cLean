/-
  Increment Kernel - Safe Example with Single Safety Theorem

  Demonstrates the new verification workflow:
  1. Define kernel
  2. Define single safety theorem with subgoals for each VC
  3. Prove all subgoals
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.VerificationIR
import CLean.ToVerificationIR
import CLean.Verification.SafetyProperties
import CLean.Verification.VCGen
import CLean.Verification.Tactics

open GpuDSL
open CLean.DeviceMacro
open DeviceIR
open CLean.VerificationIR
open CLean.ToVerificationIR
open CLean.Verification.SafetyProperties
open CLean.Verification.VCGen
open CLean.Verification.Tactics

/-! ## Kernel Definition -/

kernelArgs IncrementArgs(N: Nat)
  global[data: Array Float]

-- Manual IR for Increment
def incrementVerified (N : Nat) : VerifiedKernel :=
  let gridDim := ⟨1, 1, 1⟩
  let blockDim := ⟨256, 1, 1⟩

  let globalIdxXExpr :=
    DExpr.binop BinOp.add
      (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
      (DExpr.threadIdx Dim.x)

  let accesses := [
    { name := "data",
      space := MemorySpace.global,
      accessType := AccessType.read,
      index := some globalIdxXExpr,
      value := none,
      location := 1,
      threadId := none },
    { name := "data",
      space := MemorySpace.global,
      accessType := AccessType.write,
      index := some globalIdxXExpr,
      value := some (DExpr.floatLit 0.0),
      location := 2,
      threadId := none }
  ]

  { ir := default,
    context := { gridDim := gridDim, blockDim := blockDim, threadConstraints := [], blockConstraints := [] },
    accesses := accesses,
    barriers := [],
    uniformityInfo := [],
    accessPatterns := ∅,
    uniformStatements := [] }

def incrementConfig : VerificationContext := (incrementVerified 0).context

set_option maxRecDepth 2000

/-! ## Helper Lemmas -/

theorem distinct_indices_of_distinct_threads (t1 t2 : GlobalThreadId) (blockDim : Dim3) (env : Environment) :
    t1.blockDim = blockDim →
    t2.blockDim = blockDim →
    t1.blockDim.x = 256 →
    t1.threadId.x < 256 →
    t2.threadId.x < 256 →
    t1.blockId.y = 0 ∧ t1.blockId.z = 0 ∧ t1.threadId.y = 0 ∧ t1.threadId.z = 0 →
    t2.blockId.y = 0 ∧ t2.blockId.z = 0 ∧ t2.threadId.y = 0 ∧ t2.threadId.z = 0 →
    t1 ≠ t2 →
    let expr := DExpr.binop BinOp.add
                  (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
                  (DExpr.threadIdx Dim.x)
    evalDExpr expr t1 env ≠ evalDExpr expr t2 env := by
  intros h_dim1 h_dim2 h_bs h_b1 h_b2 h_zeros1 h_zeros2 h_neq
  simp [evalDExpr]
  intro h_eq
  simp [h_dim1, h_dim2, h_bs] at h_eq

  have h_bx : t1.blockId.x = t2.blockId.x := by
    have : t1.blockId.x = (t1.blockId.x * 256 + t1.threadId.x) / 256 := by
      rw [Nat.mul_add_div (by decide : 0 < 256) t1.blockId.x t1.threadId.x]
      rw [Nat.div_eq_of_lt h_b1]
      simp
    rw [this]
    have : t2.blockId.x = (t2.blockId.x * 256 + t2.threadId.x) / 256 := by
      rw [Nat.mul_add_div (by decide : 0 < 256) t2.blockId.x t2.threadId.x]
      rw [Nat.div_eq_of_lt h_b2]
      simp
    rw [this]
    rw [h_eq]

  have h_tx : t1.threadId.x = t2.threadId.x := by
    have : t1.threadId.x = (t1.blockId.x * 256 + t1.threadId.x) % 256 := by
      rw [Nat.add_mul_mod_self_left]
      exact (Nat.mod_eq_of_lt h_b1).symm
    rw [this]
    have : t2.threadId.x = (t2.blockId.x * 256 + t2.threadId.x) % 256 := by
      rw [Nat.add_mul_mod_self_left]
      exact (Nat.mod_eq_of_lt h_b2).symm
    rw [this]
    rw [h_eq]

  have h_eq_threads : t1 = t2 := by
    cases t1; cases t2
    rename_i b1 th1 d1 b2 th2 d2
    simp at h_bx h_tx h_dim1 h_dim2 h_zeros1 h_zeros2
    cases b1; cases b2
    cases th1; cases th2
    simp_all

  contradiction

/-! ## Single Safety Theorem -/

theorem incrementKernel_safe (N : Nat) :
    N ≤ incrementConfig.blockDim.x →
    KernelSafe (incrementVerified N) := by
  intro h_size
  unfold KernelSafe

  -- 1. Prove Race Freedom using new tactic
  have h_race_free : RaceFree (incrementVerified N) := by
    -- solve_race_freedom [incrementVerified]
    -- The tactic leaves goals for distinct threads with equal indices
    -- We just need to apply the distinctness lemma
    apply distinct_indices_of_distinct_threads t1 t2 incrementConfig.blockDim env
    · rfl
    · rfl
    · rfl
    · simp [incrementVerified, VerificationContext.inBounds, incrementConfig] at h_in1; exact h_in1.1
    · simp [incrementVerified, VerificationContext.inBounds, incrementConfig] at h_in2; exact h_in2.1
    · simp [incrementVerified, VerificationContext.inBounds, incrementConfig] at h_in1; simp [h_in1]; exact ⟨h_in1.2.2.1, h_in1.2.2.2, h_in1.2.1.1, h_in1.2.1.2⟩
    · simp [incrementVerified, VerificationContext.inBounds, incrementConfig] at h_in2; simp [h_in2]; exact ⟨h_in2.2.2.1, h_in2.2.2.2, h_in2.2.1.1, h_in2.2.1.2⟩
    · exact h_neq
    · exact h_idx_eq

  -- 2. Prove Memory Safety using new tactic
  have h_mem_safe : MemorySafe (incrementVerified N) := by
    -- solve_memory_safety [incrementVerified]
    -- Tactic leaves goals for each access
    all_goals (
      intro h_eval
      constructor
      · exact Nat.zero_le _
      · cases arraySize; apply Nat.zero_lt_succ
    )

  -- 3. Prove Barrier Safety
  have h_barrier_free : BarrierDivergenceFree (incrementVerified N) := by
    unfold BarrierDivergenceFree
    intros b h_in
    simp [incrementVerified] at h_in
    -- simp solves the goal because h_in becomes False

  -- 4. Prove No Deadlock
  have h_no_deadlock : NoBarrierDeadlock (incrementVerified N) := by
    unfold NoBarrierDeadlock
    intros b1 b2 h1 h2
    simp [incrementVerified] at h1
    -- simp solves the goal

  refine ⟨h_race_free, h_mem_safe, h_barrier_free, h_no_deadlock⟩

/-! ## Verification -/

#print incrementKernel_safe
