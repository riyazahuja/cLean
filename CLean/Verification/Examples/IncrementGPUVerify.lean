import CLean.Verification.GPUVerifyStyle

/-!
# Increment Kernel Verification (GPUVerify Style)

This file demonstrates the simplified verification approach for a simple
increment kernel that reads and writes array elements.

## Kernel Logic
```cuda
__global__ void increment(float* data) {
  int tid = threadIdx.x;
  data[tid] = data[tid] + 1.0f;
}
```

## Verification
Using GPUVerify's two-thread reduction, we only need to prove:
1. Two distinct threads access different array indices
2. Therefore no race can occur
-/

namespace CLean.Verification.Examples

open CLean.Verification.GPUVerify

/-! ## Kernel Specification -/

/-- Increment kernel: each thread accesses data[threadIdx.x] -/
def incrementSpec : KernelSpec where
  blockSize := 256
  gridSize := 1
  accesses := [
    AccessPattern.read id 1,   -- Line 1: read data[tid]
    AccessPattern.write id 2   -- Line 2: write data[tid]
  ]
  barriers := []  -- No synchronization needed

/-! ## Safety Proof -/

theorem increment_race_free : RaceFree incrementSpec := by
  -- Use the helper theorem for identity access patterns
  apply identity_kernel_race_free
  intro a h_mem
  -- a is in the list [read id 1, write id 2]
  cases h_mem with
  | head => exists 1; left; rfl
  | tail _ h' =>
    cases h' with
    | head => exists 2; right; rfl
    | tail _ h'' => cases h''

theorem increment_safe : KernelSafe incrementSpec := by
  constructor
  · exact increment_race_free
  · unfold BarrierUniform; intros; trivial

/-! ## Alternative: Direct Proof -/

/-- Direct proof without using helper theorem (for illustration) -/
theorem increment_race_free_direct : RaceFree incrementSpec := by
  unfold RaceFree DistinctThreads HasRace incrementSpec
  intro tid1 tid2 ⟨h_bound1, h_bound2, h_neq⟩ a1 a2 h_mem1 h_mem2

  -- Case split on which accesses we're considering
  cases h_mem1 with
  | head =>  -- a1 is the read
    cases h_mem2 with
    | head => left; simp  -- Both reads: no race
    | tail _ h' =>
      cases h' with
      | head => left; simp [id]; exact h_neq  -- read vs write: different indices
      | tail _ h'' => cases h''
  | tail _ h1' =>
    cases h1' with
    | head =>  -- a1 is the write
      cases h_mem2 with
      | head => left; simp [id]; exact h_neq  -- write vs read: different indices
      | tail _ h2' =>
        cases h2' with
        | head => left; simp [id]; exact h_neq  -- Both writes: different indices
        | tail _ h'' => cases h''
    | tail _ h'' => cases h''

#check increment_race_free
#check increment_safe

end CLean.Verification.Examples
