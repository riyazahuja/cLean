import CLean.Examples.Saxpy
import CLean.Proof.InstrValueCompute

/-! # Saxpy chain proof: structural foundation

Foundation lemmas characterizing the initial state `saxpyStateFor n α xs ys`
in terms suitable for stepping through the 24 step? applications that
constitute one saxpy execution per active lane.

These lemmas establish the structural preconditions for applying the
`stepInstr?_*_lane_full` lemmas in `Proof/InstrValueCompute.lean`.
-/

namespace CLean

open Helpers

/-! ## Initial-state structural lemmas -/

/-- Every lane in `saxpyWarpFor n` is at PC `("saxpyKernel", 0)`. -/
lemma saxpyWarp_lane_pc (n : Nat) (lane : LaneId)
    (ls : LaneState) (h : (saxpyWarpFor n).getLane? lane = some ls) :
    ls.pc = ("saxpyKernel", 0) := by
  unfold saxpyWarpFor WarpState.getLane? at h
  simp at h
  rw [← h]

/-- Every lane in `saxpyWarpFor n` has status `.running` (the LaneState default). -/
lemma saxpyWarp_lane_status (n : Nat) (lane : LaneId)
    (ls : LaneState) (h : (saxpyWarpFor n).getLane? lane = some ls) :
    ls.status = .running := by
  unfold saxpyWarpFor WarpState.getLane? at h
  simp at h
  rw [← h]

/-- `lockstepRunnable` holds for `saxpyWarpFor n`: all 32 lanes share PC. -/
lemma saxpyWarp_lockstepRunnable (n : Nat) :
    Helpers.lockstepRunnable (saxpyWarpFor n) := by
  unfold Helpers.lockstepRunnable Helpers.lockstepRunnable?
  split
  · rfl
  · rename_i pc hpc
    simp [List.all_eq_true]
    intro lane hlane
    split
    · rename_i ls hLane
      have h1 := saxpyWarp_lane_pc n lane ls hLane
      unfold currentRunnablePc? at hpc
      cases hlst : runnableLaneIds (saxpyWarpFor n) with
      | nil => simp [hlst] at hpc
      | cons hd tl =>
        simp [hlst, Option.bind] at hpc
        split at hpc
        · simp at hpc
        · rename_i hdLs hHdLane
          have h2 := saxpyWarp_lane_pc n hd hdLs hHdLane
          simp at hpc
          subst hpc
          simp [h1, h2]
    · rename_i hNone
      exfalso
      unfold runnableLaneIds at hlane
      simp [List.mem_filter] at hlane
      obtain ⟨_, hRun⟩ := hlane
      unfold laneIsRunnable at hRun
      rw [hNone] at hRun
      simp at hRun

/-- The warp at `(cta=0, warp=0)` of `saxpyStateFor n α xs ys` is
`saxpyWarpFor n` — independent of the data values `α, xs, ys`. -/
lemma saxpyStateFor_getWarp (n : Nat) (alpha : Int) (xs ys : List Int) :
    (saxpyStateFor n alpha xs ys).getWarp? 0 0 = some (saxpyWarpFor n) := by
  unfold saxpyStateFor State.getWarp? State.getCTA?
  simp

/-- `saxpyWarpFor n` has 32 lanes. -/
lemma saxpyWarp_wf (n : Nat) : WarpState.wf (saxpyWarpFor n) := by
  unfold WarpState.wf WarpState.wf? saxpyWarpFor
  simp [Array.size_replicate]

/-! ## Lane-runnable characterization

These lemmas show that lane `k` is runnable in `saxpyWarpFor n` iff `k < n`,
and that `runnableLaneIds (saxpyWarpFor n) = saxpyActiveLanes n hn`. -/

/-- `bitSet (activeMaskPrefix n) k` is `true` iff `k < n`, for `n ≤ 32`, `k < 32`.

Proved by `interval_cases` on both `n` and `k` (33 × 32 cases, each closed by
`decide` on a concrete `bitSet`/`activeMaskPrefix` evaluation). -/
lemma bitSet_activeMaskPrefix (n k : Nat) (hn : n ≤ 32) (hk : k < 32) :
    bitSet (activeMaskPrefix n) k = decide (k < n) := by
  interval_cases n <;> interval_cases k <;> decide

/-- Lane `k` is runnable in `saxpyWarpFor n` iff `k < n`. -/
lemma laneIsRunnable_saxpyWarp (n : Nat) (hn : n ≤ 32) (lane : LaneId) :
    laneIsRunnable (saxpyWarpFor n) lane = decide (lane.val < n) := by
  unfold laneIsRunnable
  cases h : (saxpyWarpFor n).getLane? lane with
  | none =>
    exfalso
    unfold saxpyWarpFor WarpState.getLane? at h
    simp at h
  | some ls =>
    have hstatus := saxpyWarp_lane_status n lane ls h
    simp [hstatus]
    rw [show (saxpyWarpFor n).activeMask = activeMaskPrefix n from rfl]
    have := bitSet_activeMaskPrefix n lane.val hn lane.isLt
    rw [this]

/-- The runnable lanes of `saxpyWarpFor n` are exactly `saxpyActiveLanes n hn`. -/
lemma runnableLaneIds_saxpyWarp (n : Nat) (hn : n ≤ 32) :
    runnableLaneIds (saxpyWarpFor n) = saxpyActiveLanes n hn := by
  interval_cases n <;> rfl

/-! ## Initial PC characterization -/

/-- For `n > 0`, the current runnable PC of `saxpyWarpFor n` is the BB0 entry. -/
lemma currentRunnablePc_saxpyWarp_pos (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n) :
    currentRunnablePc? (saxpyWarpFor n) = some ("saxpyKernel", 0) := by
  unfold currentRunnablePc?
  rw [runnableLaneIds_saxpyWarp n hn]
  interval_cases n <;> rfl

/-- For `n = 0`, no lane is runnable, so `currentRunnablePc?` is `none`. -/
lemma currentRunnablePc_saxpyWarp_zero :
    currentRunnablePc? (saxpyWarpFor 0) = none := by
  unfold currentRunnablePc?
  rw [runnableLaneIds_saxpyWarp 0 (by norm_num)]
  rfl

/-- Without a guard, all runnable lanes participate at the BB0 entry. -/
lemma participatingRunnable_saxpyWarp_none (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n) :
    participatingRunnableLaneIds? (saxpyWarpFor n) none = some (saxpyActiveLanes n hn) := by
  unfold participatingRunnableLaneIds?
  rw [currentRunnablePc_saxpyWarp_pos n hn hnpos]
  simp
  interval_cases n <;> rfl

/-! ## Memory frame & param-byte read characterization -/

/-- **Frame for `readBytes?` over a disjoint `writeBytes`.** Reading at `[k, k+width)`
is unchanged by a `writeBytes` at `[offset, offset+bs.length)` when the ranges
are disjoint. -/
lemma readBytes?_writeBytes_outside_range
    (mem : ByteMem) (offset width : Nat) (bs : List Byte) (k : Nat)
    (hk : k + width ≤ offset ∨ k ≥ offset + bs.length) :
    readBytes? (writeBytes mem offset bs) k width = readBytes? mem k width := by
  apply readBytes?_congr
  intro i hi
  apply writeBytes_outside_range
  rcases hk with h1 | h2
  · left; omega
  · right; omega

/-- The first 4 bytes of `saxpyParamBytesFor n α` encode `(UInt32.ofNat n)`.
The subsequent writes (`α`, three u64 base addresses) are at offsets ≥ 4, so
they don't affect the first 4 bytes. -/
lemma readBytes?_saxpyParamBytesFor_param0 (n : Nat) (alpha : Int) :
    readBytes? (saxpyParamBytesFor n alpha) 0 4
      = some (natToBytesLE (UInt32.ofNat n).toNat 4) := by
  unfold saxpyParamBytesFor
  unfold writeU64Bytes writeS32Bytes writeU32Bytes
  rw [readBytes?_writeBytes_outside_range _ 24 _ _ 0 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 16 _ _ 0 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 8 _ _ 0 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 4 _ _ 0 (by left; simp [natToBytesLE_length])]
  have hlen : (natToBytesLE (UInt32.ofNat n).toNat 4).length = 4 := natToBytesLE_length _ _
  have h := readBytes?_writeBytes_same ({} : ByteMem) 0 (natToBytesLE (UInt32.ofNat n).toNat 4)
  rw [hlen] at h
  exact h

/-- **`readMem?` for the saxpy `ld.param.u32 [param_0]`.** Lifts the byte-level
`readBytes?_saxpyParamBytesFor_param0` through the typed-access layer to
produce a `Value.u32` result. -/
lemma readMem_saxpyStateFor_param0 (n : Nat) (alpha : Int) (xs ys : List Int) :
    readMem? (saxpyStateFor n alpha xs ys) .param .u32 (.param 0) =
      some (.u32 (UInt32.ofNat n)) := by
  unfold readMem? saxpyStateFor
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?]
  rw [readBytes?_saxpyParamBytesFor_param0]
  unfold decodeScalar?
  simp [natToBytesLE_length, bytesToNatLE_natToBytesLE_4]
  apply UInt32.toNat_inj.mp
  simp [UInt32.toNat_ofNat]

/-! ## Initial-state well-formedness

`State.wf` only inspects the kernel-env structure and the lane count of each
warp; it doesn't depend on the data values `α, xs, ys` (which only live in
byte memories) or on the active-mask value `n`. So the initial saxpy state
is well-formed for every supported launch shape. -/

/-- `State.wf` for the initial saxpy state, for any supported launch shape. -/
lemma saxpyStateFor_wf (n : Nat) (alpha : Int) (xs ys : List Int) :
    State.wf (saxpyStateFor n alpha xs ys) := by
  show State.wf? _ = true
  unfold State.wf? saxpyStateFor
  rw [Bool.and_eq_true]
  refine ⟨?_, ?_⟩
  · show KernelEnv.wf? (PTX.lowerKernelEnvCheckedD saxpyKernel) = true
    native_decide
  · simp only [List.all_eq_true]
    rintro ⟨cta, cs⟩ hmem
    rw [Std.HashMap.mem_toList_iff_getElem?_eq_some] at hmem
    rw [Std.HashMap.getElem?_insert] at hmem
    split at hmem
    · simp at hmem
      subst hmem
      show CTAState.wf? _ = true
      unfold CTAState.wf?
      simp only [List.all_eq_true]
      rintro ⟨warp, ws⟩ hwmem
      rw [Std.HashMap.mem_toList_iff_getElem?_eq_some] at hwmem
      rw [Std.HashMap.getElem?_insert] at hwmem
      split at hwmem
      · simp at hwmem
        subst hwmem
        show WarpState.wf? (saxpyWarpFor n) = true
        unfold WarpState.wf? saxpyWarpFor
        simp [Array.size_replicate]
      · simp at hwmem
    · simp at hmem

end CLean
