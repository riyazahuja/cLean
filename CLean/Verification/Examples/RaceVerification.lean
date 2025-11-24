/-
  Race Condition Kernel - UNSAFE Example

  All threads write to the same location, causing a race.
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

/-! ## Unsafe Kernel with Race Condition -/

-- Kernel arguments
kernelArgs RaceArgs(N: Nat)
  global[counter: Array Float]

-- Kernel: All threads increment counter[0] (RACE!)
device_kernel raceKernel : KernelM RaceArgs Unit := do
  let args ← getArgs
  let counter : GlobalArray Float := ⟨args.counter⟩

  -- All threads read and write to counter[0]
  let val ← counter.get 0
  counter.set 0 (val + 1.0)

/-! ## Verification Setup -/

def raceConfig : VerificationContext :=
  { gridDim := ⟨1, 1, 1⟩
    blockDim := ⟨256, 1, 1⟩
    threadConstraints := []
    blockConstraints := [] }

def raceArraySizes : List (String × Nat) :=
  [("counter", 1)]

/-! ## Translation to VerificationIR -/

def raceVerified : VerifiedKernel :=
  toVerificationIR raceKernelIR raceConfig.gridDim raceConfig.blockDim

/-! ## Showing the Race Exists -/

/-- The kernel has a write-write race on counter[0] -/
theorem race_exists :
    ∃ (acc1 acc2 : MemoryAccess),
      acc1 ∈ raceVerified.accesses ∧
      acc2 ∈ raceVerified.accesses ∧
      acc1.name = "counter" ∧
      acc2.name = "counter" ∧
      acc1.accessType = AccessType.write ∧
      acc2.accessType = AccessType.write ∧
      acc1.conflicts acc2 := by
  -- There are write accesses to counter[0] from the kernel
  sorry

/-- The kernel is NOT race-free -/
theorem race_not_safe :
    ¬RaceFree raceVerified := by
  unfold RaceFree
  push_neg
  -- Provide the two conflicting accesses
  sorry

/-! ## Generate VCs -/

#eval! do
  IO.println "[RaceKernel] Generating verification conditions..."
  verifyKernel raceKernelIR
               raceConfig.gridDim
               raceConfig.blockDim
               raceArraySizes
               "CLean/Verification/Examples/RaceVCs.lean"
