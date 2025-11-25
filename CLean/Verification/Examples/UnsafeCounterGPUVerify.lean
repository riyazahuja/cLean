import CLean.Verification.GPUVerifyStyle

/-!
# Unsafe Counter Verification (GPUVerify Style)

This file demonstrates how the GPUVerify-style approach easily detects races.

## Kernel Logic (UNSAFE!)
```cuda
__global__ void unsafeCounter(int* counter) {
  *counter = *counter + 1;  // RACE: All threads write to same location!
}
```

## Verification
We prove that this kernel is NOT safe by showing a race exists.
-/

namespace CLean.Verification.Examples

open CLean.Verification.GPUVerify

/-! ## Kernel Specification -/

/-- Unsafe counter: all threads access counter[0] -/
def unsafeCounterSpec : KernelSpec where
  blockSize := 256
  gridSize := 1
  accesses := [
    AccessPattern.read (fun _ => 0) 1,   -- All threads read counter[0]
    AccessPattern.write (fun _ => 0) 2   -- All threads write counter[0]
  ]
  barriers := []

/-! ## Unsafety Proof -/

/-- The unsafe counter has a race condition -/
theorem unsafeCounter_has_race : ¬RaceFree unsafeCounterSpec := by
  unfold RaceFree
  push_neg
  -- Pick two distinct threads: 0 and 1
  use 0, 1
  constructor
  · -- They are distinct
    unfold DistinctThreads unsafeCounterSpec
    decide
  · -- Witness the racing accesses
    use AccessPattern.write (fun _ => 0) 2
    use AccessPattern.write (fun _ => 0) 2
    constructor
    · -- First access is in the list
      simp [unsafeCounterSpec]
    · constructor
      · -- Second access is in the list
        simp [unsafeCounterSpec]
      · -- The race exists and is not separated by barriers
        constructor
        · -- HasRace holds
          unfold HasRace
          simp
        · -- No barrier separates them
          unfold SeparatedByBarrier
          simp [unsafeCounterSpec]

theorem unsafeCounter_not_safe : ¬KernelSafe unsafeCounterSpec := by
  unfold KernelSafe
  intro h
  exact unsafeCounter_has_race h.1

/-! ## Comparison with Safe Kernel -/

/-- Safe version: each thread updates its own counter -/
def safeCounterSpec : KernelSpec where
  blockSize := 256
  gridSize := 1
  accesses := [
    AccessPattern.read id 1,   -- Each thread reads counter[tid]
    AccessPattern.write id 2   -- Each thread writes counter[tid]
  ]
  barriers := []

/-- The safe counter is race-free -/
theorem safeCounter_race_free : RaceFree safeCounterSpec := by
  apply identity_kernel_race_free
  intro a h_mem
  cases h_mem with
  | head => exists 1; left; rfl
  | tail _ h' =>
    cases h' with
    | head => exists 2; right; rfl
    | tail _ h'' => cases h''

#check unsafeCounter_has_race
#check unsafeCounter_not_safe
#check safeCounter_race_free

end CLean.Verification.Examples
