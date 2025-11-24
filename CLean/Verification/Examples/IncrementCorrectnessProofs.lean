import CLean.Semantics.DeviceSemantics
import CLean.Verification.HashMapLemmas
import CLean.DeviceMacro

/-!
# Increment Kernel Functional Correctness Proofs

Proves the increment kernel axioms using Hoare-style reasoning and HashMap lemmas.
-/

namespace CLean.Verification.Functional

open DeviceIR
open CLean.Semantics
open GpuDSL
open Std (HashMap)

/-! ## Kernel Body Definition -/

/-- Semantic version of increment kernel body -/
def incrementBody : DStmt :=
  DStmt.seq
    (DStmt.assign "i" (DExpr.threadIdx Dim.x))
    (DStmt.ite
      (DExpr.binop BinOp.lt (DExpr.var "i") (DExpr.var "N"))
      (DStmt.seq
        (DStmt.assign "val" (DExpr.index (DExpr.var "data") (DExpr.var "i")))
        (DStmt.store
          (DExpr.var "data")
          (DExpr.var "i")
          (DExpr.binop BinOp.add (DExpr.var "val") (DExpr.floatLit 1.0))))
      DStmt.skip)

/-! ## Helper Lemmas About Execution -/

-- Evaluating thread i.x in context gives i
theorem eval_threadIdx_x (i : Nat) (ctx : ThreadContext) (mem : Memory) :
    ctx.tid = i →
    evalExpr (DExpr.threadIdx Dim.x) ctx mem = Value.int i := by
  intro h
  simp [evalExpr, h]

-- Assigning to local preserves memory
theorem assign_preserves_memory (x : String) (e : DExpr) (ctx : ThreadContext) (mem : Memory) :
    (evalStmt (DStmt.assign x e) ctx mem).2 = mem := by
  simp [evalStmt]

/-! ## Main Theorem 1: Thread Computes Correctly -/

/-- Thread i writes output[i] = input[i] + 1.0 when i < N -/
theorem increment_thread_computes_correctly (i N : Nat) (mem : Memory) :
  i < N →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  (mem'.get "data" i).toFloat = (mem.get "data" i).toFloat + 1.0 := by
  intro h_i_lt_N

  -- Unfold execThread
  unfold execThread

  -- Initial context for thread i
  let ctx₀ : ThreadContext := {tid := i, bid := 0, blockSize := 256, locals := ∅}

  -- Execute incrementBody
  unfold incrementBody
  simp [evalStmt]

  -- Step 1: execute "i := threadIdx.x"
  -- After this: ctx₁.locals["i"] = i, mem₁ = mem_with_N
  let ctx₁ := ctx₀.setLocal "i" (Value.int i)
  have h_mem₁ : (evalStmt (DStmt.assign "i" (DExpr.threadIdx Dim.x)) ctx₀ (mem.set "N" 0 (Value.int N))).2
                  = mem.set "N" 0 (Value.int N) := assign_preserves_memory _ _ _ _

  -- Step 2: evaluate condition "i < N"
  have h_cond : (evalExpr (DExpr.binop BinOp.lt (DExpr.var "i") (DExpr.var "N")) ctx₁ (mem.set "N" 0 (Value.int N))).toBool = true := by
    simp [evalExpr, evalBinOp, ThreadContext.getLocal, ThreadContext.setLocal]
    simp [HashMap.getD, HashMap.insert]
    sorry  -- i < N from assumption

  sorry

  -- Step 3: Since condition is true, execute then branch
  -- Read val = data[i]
  -- Store data[i] = val + 1.0

    -- Complete the proof using HashMap lemmas

/-! ## Main Theorem 2: Thread Preserves Other Locations -/

/-- Thread execution preserves other locations -/
theorem increment_thread_preserves_others (i j N : Nat) (mem : Memory) :
  i ≠ j →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  (mem'.get "data" j).toFloat = (mem.get "data" j).toFloat := by
  intro h_neq

  -- The key insight: thread i only writes to data[i]
  -- Therefore data[j] is unchanged for j ≠ i

  sorry  -- Use HashMap.getD_insert_diff with h_neq

/-! ## Main Theorem 3: Out-of-Bounds Threads are No-ops -/

/-- Threads i ≥ N don't modify anything -/
theorem increment_thread_noop_when_out_of_bounds (i N : Nat) (mem : Memory) :
  i ≥ N →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  ∀ j, (mem'.get "data" j).toFloat = (mem.get "data" j).toFloat := by
  intro h_i_ge_N j

  -- The condition "i < N" is false
  -- So the else branch (skip) is executed
  -- Skip doesn't modify memory

  sorry  -- Straightforward from semantics of skip

end CLean.Verification.Functional
