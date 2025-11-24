import CLean.Semantics.DeviceSemantics
import CLean.Verification.HashMapLemmas
import CLean.DeviceMacro

/-!
# Increment Kernel Functional Correctness Proofs

Proofs of the increment kernel's functional correctness.
The proofs are currently axiomatized (sorry) but include detailed strategies.
-/

namespace CLean.Verification.Functional

open DeviceIR
open CLean.Semantics
open GpuDSL
open Std (HashMap)

/-! ## Kernel Body Definition -/

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

/-! ## Main Theorems -/

/-- Thread i writes output[i] = input[i] + 1.0 when i < N -/
theorem increment_thread_computes_correctly (i N : Nat) (mem : Memory) :
  i < N →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  (mem'.get "data" i).toFloat = (mem.get "data" i).toFloat + 1.0 := by
  intro h_i_lt_N

  /-
  Proof strategy:
  1. execThread creates ctx with tid = i, locals = ∅
  2. Execute assign "i" = threadIdx.x → locals["i"] = i
  3. Evaluate condition "i < N" → true (from h_i_lt_N)
  4. Execute then branch:
     a. Assign val = data[i] (reads from memory)
     b. Store data[i] = val + 1.0
  5. Final memory: mem' = mem_with_N.set "data" i (original_val + 1.0)
  6. Use HashMap.getD_insert_same to show getting data[i] returns stored value
  7. Simplify arithmetic

  Difficulty: evalStmt/evalExpr are partial, can't unfold directly.
  Need to reason about their behavior via specification lemmas.
  -/
  sorry

/-- Thread execution preserves other locations -/
theorem increment_thread_preserves_others (i j N : Nat) (mem : Memory) :
  i ≠ j →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  (mem'.get "data" j).toFloat = (mem.get "data" j).toFloat := by
  intro h_neq

  /-
  Proof strategy:
  1. Thread i only modifies memory via store at data[i]
  2. Final memory: mem' = intermediate.set "data" i new_value
  3. Getting data[j] where j ≠ i uses HashMap.getD_insert_diff
  4. This returns the original value from intermediate memory
  5. Show intermediate memory preserves data[j] through earlier steps

  Key lemma needed: HashMap.getD_insert_diff with h_neq
  -/
  sorry

/-- Threads i ≥ N don't modify anything -/
theorem increment_thread_noop_when_out_of_bounds (i N : Nat) (mem : Memory) :
  i ≥ N →
  let mem_with_N := mem.set "N" 0 (Value.int N)
  let mem' := execThread incrementBody i 0 256 mem_with_N
  ∀ j, (mem'.get "data" j).toFloat = (mem.get "data" j).toFloat := by
  intro h_i_ge_N j

  /-
  Proof strategy:
  1. Condition "i < N" evaluates to false (from h_i_ge_N)
  2. Else branch (DStmt.skip) executes
  3. Skip: evalStmt skip ctx mem = (ctx, mem) (no change)
  4. Therefore mem' = mem_with_N
  5. Getting data[j] from mem_with_N uses N preservation lemma

  This is the easiest proof - skip semantics are trivial.
  -/
  sorry

end CLean.Verification.Functional
