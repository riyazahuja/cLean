/-
  Unsafe Counter Kernel - Example with Single Safety Theorem

  Demonstrates proving that a kernel is UNSAFE.
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

kernelArgs UnsafeCounterArgs(dummy: Nat)
  global[counter: Array Float]

-- Manual IR for Unsafe Counter
def unsafeVerified : VerifiedKernel :=
  let gridDim := ⟨1, 1, 1⟩
  let blockDim := ⟨256, 1, 1⟩

  let accesses := [
    -- WRITE counter[0]
    { name := "counter",
      space := MemorySpace.global,
      accessType := AccessType.write,
      index := some (DExpr.intLit 0),
      value := some (DExpr.floatLit 1.0),
      location := 1,
      threadId := none }
  ]

  { ir := default,
    context := { gridDim := gridDim, blockDim := blockDim, threadConstraints := [], blockConstraints := [] },
    accesses := accesses,
    barriers := [],
    uniformityInfo := [],
    accessPatterns := ∅,
    uniformStatements := [] }

def unsafeConfig : VerificationContext := unsafeVerified.context

/-! ## Proving Unsafety -/

/-- The kernel is provably UNSAFE -/
theorem unsafeCounter_unsafe : ¬KernelSafe unsafeVerified := by
  -- We prove it's not race-free, which implies it's not safe
  intro h_safe
  have h_race_free := h_safe.1

  -- Construct two threads
  let t1 : GlobalThreadId := { blockId := ⟨0,0,0⟩, threadId := ⟨0,0,0⟩, blockDim := ⟨256,1,1⟩ }
  let t2 : GlobalThreadId := { blockId := ⟨0,0,0⟩, threadId := ⟨1,0,0⟩, blockDim := ⟨256,1,1⟩ }

  -- Prove they are in bounds
  have h_b1 : unsafeVerified.context.inBounds t1 := by
    simp [unsafeVerified, VerificationContext.inBounds, unsafeConfig]
    decide
  have h_b2 : unsafeVerified.context.inBounds t2 := by
    simp [unsafeVerified, VerificationContext.inBounds, unsafeConfig]
    decide

  -- Pick the write access (twice)
  let acc1 := unsafeVerified.accesses.head!
  let acc2 := unsafeVerified.accesses.head!

  have h_mem1 : acc1 ∈ unsafeVerified.accesses := by
    simp [unsafeVerified]
  have h_mem2 : acc2 ∈ unsafeVerified.accesses := by
    simp [unsafeVerified]

  have h_name : acc1.name = acc2.name := rfl
  have h_write : acc1.isWrite ∨ acc2.isWrite := Or.inl rfl

  -- Semantic conflict: indices evaluate to same value (0 = 0)
  have h_idx_eq : match acc1.index, acc2.index with
       | some idx1, some idx2 => evalDExpr idx1 t1 (fun _ => 0) = evalDExpr idx2 t2 (fun _ => 0)
       | none, none => True
       | _, _ => False := by
    simp [unsafeVerified, evalDExpr]

  -- Apply RaceFree
  have h_ordered := h_race_free (fun _ => 0) t1 t2 h_b1 h_b2 acc1 acc2 h_mem1 h_mem2 h_name h_write h_idx_eq

  -- t1 ≠ t2
  have h_neq : t1 ≠ t2 := by
    intro h
    injection h with _ h_tid
    injection h_tid with h_x
    contradiction

  -- RaceFree implies (t1 ≠ t2 → False)
  have h_false := h_ordered.1 h_neq
  exact h_false

/-! ## Verification -/

#print unsafeCounter_unsafe
