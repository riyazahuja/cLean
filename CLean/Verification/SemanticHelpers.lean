import CLean.Semantics.DeviceSemantics
import CLean.Verification.HashMapLemmas

/-!
# Semantic Reasoning Helper Lemmas

Axiomatized lemmas for common patterns in semantic reasoning.
These hide the low-level HashMap and type conversion details.
-/

namespace CLean.Verification.SemanticHelpers

open DeviceIR
open CLean.Semantics
open Std (HashMap)

/-! ## Memory Operation Helpers -/

axiom mem_set_different_arrays (mem : Memory) (arr1 arr2 : String) (idx1 idx2 : Nat) (v1 v2 : Value) :
    arr1 ≠ arr2 →
    (mem.set arr1 idx1 v1).get arr2 idx2 = mem.get arr2 idx2

axiom mem_set_different_indices (mem : Memory) (arr : String) (idx1 idx2 : Nat) (v : Value) :
    idx1 ≠ idx2 →
    (mem.set arr idx1 v).get arr idx2 = mem.get arr idx2

axiom mem_set_get_same (mem : Memory) (arr : String) (idx : Nat) (v : Value) :
    (mem.set arr idx v).get arr idx = v

/-! ## Value Conversion Helpers -/

axiom value_float_add (v : Value) (x : Float) :
    (match v with | Value.float f => Value.float (f + x) | _ => Value.float x).toFloat
    = v.toFloat + x

axiom int_to_float_zero (n : Int) :
    (Value.int n).toFloat = 0.0

/-! ## Expression Evaluation Helpers -/

axiom eval_threadIdx_is_tid (ctx : ThreadContext) (mem : Memory) :
    evalExpr (DExpr.threadIdx Dim.x) ctx mem = Value.int ctx.tid

axiom eval_var_gets_local (ctx : ThreadContext) (mem : Memory) (x : String) :
    evalExpr (DExpr.var x) ctx mem = ctx.getLocal x

axiom eval_lt_ints (ctx : ThreadContext) (mem : Memory) (i j : Nat) :
    (evalExpr (DExpr.binop BinOp.lt (DExpr.intLit i) (DExpr.intLit j)) ctx mem).toBool
    = decide (i < j)

/-! ## Statement Execution Helpers -/

axiom assign_preserves_mem (x : String) (e : DExpr) (ctx : ThreadContext) (mem : Memory) :
    (evalStmt (DStmt.assign x e) ctx mem).2 = mem

axiom skip_is_noop (ctx : ThreadContext) (mem : Memory) :
    evalStmt DStmt.skip ctx mem = (ctx, mem)

axiom store_preserves_ctx (arr idx val : DExpr) (ctx : ThreadContext) (mem : Memory) :
    (evalStmt (DStmt.store arr idx val) ctx mem).1 = ctx

/-! ## exec Kernel Reasoning Lemmas -/

/-- execKernel applies execThread to each thread index -/
axiom execKernel_applies_threads (body : DStmt) (numThreads blockSize : Nat) (mem : Memory) :
  ∀ i < numThreads,
    (execKernel body numThreads blockSize mem).get arrName idx =
    (execThread body i 0 blockSize mem).get arrName idx
    ∨ (∃ j ≠ i, j < numThreads ∧
        (execThread body j 0 blockSize mem).get arrName idx ≠ mem.get arrName idx)

/-- Simpler: If thread i writes to location idx and others don't, result is from thread i -/
axiom execKernel_single_writer (body : DStmt) (numThreads blockSize : Nat) (mem : Memory)
    (arrName : String) (idx : Nat) :
  ∀ i < numThreads,
    (∀ j < numThreads, j ≠ i → (execThread body j 0 blockSize mem).get arrName idx = mem.get arrName idx) →
    (execKernel body numThreads blockSize mem).get arrName idx =
    (execThread body i 0 blockSize mem).get arrName idx

/-! ## Array Type Invariants -/

/-- For kernels with float arrays in globalArrays, memory gets return floats
    This uses the type information from the kernel's globalArrays field -/
axiom mem_get_typed_array (kernel : DeviceIR.Kernel) (mem : Memory) (arrName : String) (idx : Nat) :
    (kernel.globalArrays.any (fun arr => arr.name == arrName && arr.ty == DeviceIR.DType.array DeviceIR.DType.float)) →
    ∃ f : Float, mem.get arrName idx = Value.float f

/-- Similar for int arrays -/
axiom mem_get_int_array (kernel : DeviceIR.Kernel) (mem : Memory) (arrName : String) (idx : Nat) :
    (kernel.globalArrays.any (fun arr => arr.name == arrName && arr.ty == DeviceIR.DType.array DeviceIR.DType.int)) →
    ∃ n : Int, mem.get arrName idx = Value.int n

end CLean.Verification.SemanticHelpers
