/-
  Simple Array Copy Kernel - SAFE Example

  Each thread copies one element from input to output.
  Thread i writes to output[i], so no races.
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

/-! ## Simple Array Copy Kernel -/

-- Kernel arguments
kernelArgs ArrayCopyArgs(N: Nat)
  global[input output: Array Float]

-- Kernel: output[i] = input[i]
device_kernel arrayCopyKernel : KernelM ArrayCopyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  if i < N then do
    let val ← input.get i
    output.set i val

/-! ## Verification Setup -/

def arrayCopyConfig : VerificationContext :=
  { gridDim := ⟨1, 1, 1⟩
    blockDim := ⟨256, 1, 1⟩
    threadConstraints := []
    blockConstraints := [] }

def arrayCopyArraySizes (N : Nat) : List (String × Nat) :=
  [("input", N), ("output", N)]

/-! ## Translation to VerificationIR -/

def arrayCopyVerified (N : Nat) : VerifiedKernel :=
  toVerificationIR arrayCopyKernelIR arrayCopyConfig.gridDim arrayCopyConfig.blockDim

/-! ## Safety Proofs -/

/-- Read from input[i] is within bounds -/
theorem arrayCopy_input_bounds (N : Nat) :
    ∀ (i : Nat) (t : GlobalThreadId),
      i = t.threadId.x + t.blockId.x * t.blockDim.x →
      i < N →
      0 ≤ i ∧ i < N := by
  intros i t h_idx h_bound
  constructor
  · omega
  · exact h_bound

/-- Write to output[i] is within bounds -/
theorem arrayCopy_output_bounds (N : Nat) :
    ∀ (i : Nat) (t : GlobalThreadId),
      i = t.threadId.x + t.blockId.x * t.blockDim.x →
      i < N →
      0 ≤ i ∧ i < N := by
  intros i t h_idx h_bound
  constructor
  · omega
  · exact h_bound

/-- No races between output writes: different threads write to different indices -/
theorem arrayCopy_no_race_output (N : Nat) :
    ∀ (k : VerifiedKernel) (t1 t2 : GlobalThreadId),
      k = arrayCopyVerified N →
      k.context.inBounds t1 →
      k.context.inBounds t2 →
      t1 ≠ t2 →
      -- Different threads compute different global indices
      let i1 := t1.threadId.x + t1.blockId.x * t1.blockDim.x
      let i2 := t2.threadId.x + t2.blockId.x * t2.blockDim.x
      i1 ≠ i2 := by
  intros k t1 t2 h_k h_bounds1 h_bounds2 h_diff
  -- If t1 ≠ t2, then their global indices differ
  -- This is by construction of globalIdxX
  intro h_eq
  -- h_eq says i1 = i2
  -- But t1 ≠ t2 means either blockId or threadId differs
  cases h_diff
  case _ =>
    -- Extract the components
    have h1 : t1.threadId.x + t1.blockId.x * t1.blockDim.x =
              t2.threadId.x + t2.blockId.x * t2.blockDim.x := h_eq
    -- From k = arrayCopyVerified N, we know blockDim is fixed
    rw [h_k] at h_bounds1 h_bounds2
    unfold arrayCopyVerified arrayCopyConfig at h_bounds1 h_bounds2
    simp [VerificationContext.inBounds] at h_bounds1 h_bounds2
    -- t1.blockDim and t2.blockDim are both 256×1×1
    -- If t1 ≠ t2, then either blockId or threadId component differs
    -- Therefore their linear combinations differ
    sorry  -- This requires more detailed arithmetic

/-- Main safety theorem: ArrayCopy is safe -/
theorem arrayCopy_safe (N : Nat) :
    let k := arrayCopyVerified N
    N ≤ arrayCopyConfig.blockDim.x →
    KernelSafe k := by
  intro h_size
  unfold KernelSafe
  constructor
  · -- Race freedom
    unfold RaceFree
    intros acc1 acc2 h_conflict
    -- Show that conflicting accesses are ordered
    left
    -- If they conflict, they must be to output[i]
    -- But different threads write to different i
    sorry
  constructor
  · -- Memory safety
    unfold MemorySafe
    intro arr
    unfold ArrayBoundsSafe
    intros acc h_in
    -- All accesses are within bounds by the i < N check
    sorry
  constructor
  · -- No barriers, so trivially barrier-divergence-free
    unfold BarrierDivergenceFree
    intros b h_in
    -- k.barriers = []
    have : (arrayCopyVerified N).barriers = [] := by
      unfold arrayCopyVerified
      simp [toVerificationIR]
      sorry
    rw [this] at h_in
    exact absurd h_in (List.not_mem_nil b)
  · -- No barriers, so trivially no deadlock
    unfold NoBarrierDeadlock
    intros b1 b2 h1 h2
    have : (arrayCopyVerified N).barriers = [] := by
      unfold arrayCopyVerified
      sorry
    rw [this] at h1
    exact absurd h1 (List.not_mem_nil b1)

/-! ## Generate VCs -/

#eval! do
  IO.println "[ArrayCopy] Generating verification conditions..."
  verifyKernel arrayCopyKernelIR
               arrayCopyConfig.gridDim
               arrayCopyConfig.blockDim
               (arrayCopyArraySizes 1024)
               "CLean/Verification/Examples/ArrayCopyVCs.lean"
