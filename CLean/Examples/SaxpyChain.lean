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

/-- The second 4-byte parameter slot encodes `α` as an s32 value. -/
lemma readBytes?_saxpyParamBytesFor_param1 (n : Nat) (alpha : Int) :
    readBytes? (saxpyParamBytesFor n alpha) 4 4 =
      some (natToBytesLE (signedToNat 32 alpha) 4) := by
  unfold saxpyParamBytesFor
  unfold writeU64Bytes writeS32Bytes writeU32Bytes
  rw [readBytes?_writeBytes_outside_range _ 24 _ _ 4 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 16 _ _ 4 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 8 _ _ 4 (by left; simp [natToBytesLE_length])]
  have hlen : (natToBytesLE (signedToNat 32 alpha) 4).length = 4 :=
    natToBytesLE_length _ _
  have h := readBytes?_writeBytes_same
    (writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)) 4
    (natToBytesLE (signedToNat 32 alpha) 4)
  rw [hlen] at h
  exact h

/-- The value read by `ld.param.s32 [param_1]`. This is the 32-bit wrapped
interpretation of `α`, matching the byte-level parameter encoding. -/
def saxpyAlphaS32 (alpha : Int) : Int :=
  natToSigned 32 (signedToNat 32 alpha)

lemma readMem_saxpyStateFor_param1 (n : Nat) (alpha : Int) (xs ys : List Int) :
    readMem? (saxpyStateFor n alpha xs ys) .param .s32 (.param 4) =
      some (.s32 (saxpyAlphaS32 alpha)) := by
  unfold readMem? saxpyStateFor saxpyAlphaS32
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?]
  rw [readBytes?_saxpyParamBytesFor_param1]
  have hLt' : signedToNat 32 alpha < 4294967296 := by
    have hLt : signedToNat 32 alpha < 2 ^ 32 := signedToNat_lt 32 (by decide) alpha
    have hPow : (2 : Nat) ^ 32 = 4294967296 := by decide
    omega
  unfold decodeScalar?
  simp [natToBytesLE_length, bytesToNatLE_natToBytesLE_4, Nat.mod_eq_of_lt hLt']

theorem readMem?_param_congr
    {st st' : State} {ty : ScalarTy} {offset : Nat}
    (hParam : st.param = st'.param) :
    readMem? st .param ty (.param offset) =
      readMem? st' .param ty (.param offset) := by
  unfold readMem?
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?, hParam]

theorem readMem?_param_s32_congr
    {st st' : State} {offset : Nat}
    (hParam : st.param = st'.param) :
    readMem? st .param .s32 (.param offset) =
      readMem? st' .param .s32 (.param offset) :=
  readMem?_param_congr hParam

/-- The third parameter slot encodes the base address of `xs` as a u64 value. -/
lemma readBytes?_saxpyParamBytesFor_param2 (n : Nat) (alpha : Int) :
    readBytes? (saxpyParamBytesFor n alpha) 8 8 =
      some (natToBytesLE (UInt64.ofNat saxpyXBase).toNat 8) := by
  unfold saxpyParamBytesFor
  unfold writeU64Bytes writeS32Bytes writeU32Bytes
  rw [readBytes?_writeBytes_outside_range _ 24 _ _ 8 (by left; simp [natToBytesLE_length])]
  rw [readBytes?_writeBytes_outside_range _ 16 _ _ 8 (by left; simp [natToBytesLE_length])]
  have hlen : (natToBytesLE (UInt64.ofNat saxpyXBase).toNat 8).length = 8 :=
    natToBytesLE_length _ _
  have h := readBytes?_writeBytes_same
    (writeS32Bytes (writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)) 4 alpha) 8
    (natToBytesLE (UInt64.ofNat saxpyXBase).toNat 8)
  rw [hlen] at h
  exact h

/-- The fourth parameter slot encodes the base address of `ys` as a u64 value. -/
lemma readBytes?_saxpyParamBytesFor_param3 (n : Nat) (alpha : Int) :
    readBytes? (saxpyParamBytesFor n alpha) 16 8 =
      some (natToBytesLE (UInt64.ofNat saxpyYBase).toNat 8) := by
  unfold saxpyParamBytesFor
  unfold writeU64Bytes writeS32Bytes writeU32Bytes
  rw [readBytes?_writeBytes_outside_range _ 24 _ _ 16 (by left; simp [natToBytesLE_length])]
  have hlen : (natToBytesLE (UInt64.ofNat saxpyYBase).toNat 8).length = 8 :=
    natToBytesLE_length _ _
  have h := readBytes?_writeBytes_same
    (writeU64Bytes
      (writeS32Bytes (writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)) 4 alpha)
      8 (UInt64.ofNat saxpyXBase)) 16
    (natToBytesLE (UInt64.ofNat saxpyYBase).toNat 8)
  rw [hlen] at h
  exact h

/-- The fifth parameter slot encodes the base address of `r` as a u64 value. -/
lemma readBytes?_saxpyParamBytesFor_param4 (n : Nat) (alpha : Int) :
    readBytes? (saxpyParamBytesFor n alpha) 24 8 =
      some (natToBytesLE (UInt64.ofNat saxpyRBase).toNat 8) := by
  unfold saxpyParamBytesFor
  unfold writeU64Bytes writeS32Bytes writeU32Bytes
  have hlen : (natToBytesLE (UInt64.ofNat saxpyRBase).toNat 8).length = 8 :=
    natToBytesLE_length _ _
  have h := readBytes?_writeBytes_same
    (writeU64Bytes
      (writeU64Bytes
        (writeS32Bytes (writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)) 4 alpha)
        8 (UInt64.ofNat saxpyXBase))
      16 (UInt64.ofNat saxpyYBase)) 24
    (natToBytesLE (UInt64.ofNat saxpyRBase).toNat 8)
  rw [hlen] at h
  exact h

lemma readMem_saxpyStateFor_param2 (n : Nat) (alpha : Int) (xs ys : List Int) :
    readMem? (saxpyStateFor n alpha xs ys) .param .u64 (.param 8) =
      some (.u64 (UInt64.ofNat saxpyXBase)) := by
  unfold readMem? saxpyStateFor
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?]
  rw [readBytes?_saxpyParamBytesFor_param2]
  simpa [encodeScalar?] using decode_encode_u64 (UInt64.ofNat saxpyXBase)

lemma readMem_saxpyStateFor_param3 (n : Nat) (alpha : Int) (xs ys : List Int) :
    readMem? (saxpyStateFor n alpha xs ys) .param .u64 (.param 16) =
      some (.u64 (UInt64.ofNat saxpyYBase)) := by
  unfold readMem? saxpyStateFor
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?]
  rw [readBytes?_saxpyParamBytesFor_param3]
  simpa [encodeScalar?] using decode_encode_u64 (UInt64.ofNat saxpyYBase)

lemma readMem_saxpyStateFor_param4 (n : Nat) (alpha : Int) (xs ys : List Int) :
    readMem? (saxpyStateFor n alpha xs ys) .param .u64 (.param 24) =
      some (.u64 (UInt64.ofNat saxpyRBase)) := by
  unfold readMem? saxpyStateFor
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?]
  rw [readBytes?_saxpyParamBytesFor_param4]
  simpa [encodeScalar?] using decode_encode_u64 (UInt64.ofNat saxpyRBase)

theorem readMem?_param_u64_congr
    {st st' : State} {offset : Nat}
    (hParam : st.param = st'.param) :
    readMem? st .param .u64 (.param offset) =
      readMem? st' .param .u64 (.param offset) :=
  readMem?_param_congr hParam

theorem readMem?_global_congr
    {st st' : State} {ty : ScalarTy} {offset : Nat}
    (hGlobal : st.global = st'.global) :
    readMem? st .global ty (.global offset) = readMem? st' .global ty (.global offset) := by
  unfold readMem?
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?, hGlobal]

theorem decodeScalar_s32_natToBytesLE (x : Int) :
    decodeScalar? .s32 (natToBytesLE (signedToNat 32 x) 4) = some (.s32 (s32Wrap x)) := by
  unfold decodeScalar? s32Wrap
  have hLt : signedToNat 32 x < 2 ^ 32 := signedToNat_lt 32 (by decide) x
  have hLt' : signedToNat 32 x < 4294967296 := by
    have hPow : (2 : Nat) ^ 32 = 4294967296 := by decide
    omega
  simp [natToBytesLE_length, bytesToNatLE_natToBytesLE_4, Nat.mod_eq_of_lt hLt']

theorem writeS32Vector_loop_shift
    (base i : Nat) (mem : ByteMem) (xs : List Int) :
    writeS32Vector.loop base i mem xs =
      writeS32Vector.loop (base + i * 4) 0 mem xs := by
  induction xs generalizing base i mem with
  | nil => simp [writeS32Vector.loop]
  | cons _ xs ih =>
      simp only [writeS32Vector.loop]
      rw [ih]
      rw [ih (base + i * 4) 1]
      ring_nf

theorem writeS32Vector_cons (mem : ByteMem) (base : Nat) (x : Int) (xs : List Int) :
    writeS32Vector mem base (x :: xs) =
      writeS32Vector (writeS32Bytes mem base x) (base + 4) xs := by
  unfold writeS32Vector
  simp only [writeS32Vector.loop]
  rw [writeS32Vector_loop_shift]
  ring_nf

theorem readBytes?_writeS32Vector_before
    (mem : ByteMem) (base k width : Nat) (xs : List Int)
    (hk : k + width ≤ base) :
    readBytes? (writeS32Vector mem base xs) k width = readBytes? mem k width := by
  induction xs generalizing mem base with
  | nil => simp [writeS32Vector, writeS32Vector.loop]
  | cons x xs ih =>
      rw [writeS32Vector_cons]
      rw [ih (writeS32Bytes mem base x) (base + 4)]
      · unfold writeS32Bytes
        rw [readBytes?_writeBytes_outside_range mem base width
          (natToBytesLE (signedToNat 32 x) 4) k (Or.inl hk)]
      · omega

theorem readBytes?_writeS32Vector_getD
    (mem : ByteMem) (base : Nat) (xs : List Int) {i : Nat}
    (hi : i < xs.length) :
    readBytes? (writeS32Vector mem base xs) (base + i * 4) 4 =
      some (natToBytesLE (signedToNat 32 (listIntGetD xs i)) 4) := by
  induction xs generalizing mem base i with
  | nil => simp at hi
  | cons x xs ih =>
      cases i with
      | zero =>
          rw [writeS32Vector_cons]
          have hBefore := readBytes?_writeS32Vector_before
            (writeS32Bytes mem base x) (base + 4) (base + 0 * 4) 4 xs (by omega)
          rw [hBefore]
          · unfold writeS32Bytes
            have hLen : (natToBytesLE (signedToNat 32 x) 4).length = 4 :=
              natToBytesLE_length _ _
            have hRead := readBytes?_writeBytes_same mem base
              (natToBytesLE (signedToNat 32 x) 4)
            rw [hLen] at hRead
            simpa [listIntGetD] using hRead
      | succ i =>
          have hiTail : i < xs.length := by
            simpa [List.length] using Nat.lt_of_succ_lt_succ hi
          rw [writeS32Vector_cons]
          have hRead := ih (writeS32Bytes mem base x) (base + 4) hiTail
          have hOffset : base + (i + 1) * 4 = base + 4 + i * 4 := by
            ring_nf
          rw [hOffset]
          simpa [listIntGetD] using hRead

theorem readMem?_global_s32_of_readBytes
    {st : State} {offset : Nat} {x : Int}
    (hAlign : offset % 4 = 0)
    (hRead : readBytes? st.global.bytes offset 4 =
      some (natToBytesLE (signedToNat 32 x) 4)) :
    readMem? st .global .s32 (.global offset) = some (.s32 (s32Wrap x)) := by
  unfold readMem?
  have hPre : Typing.typedAccessPreconditions? .global .s32 (.global offset) = true := by
    simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
          Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
          Typing.addrSpaceMatches?, Addr.offset, Addr.space, hAlign]
  simp [hPre, Typing.byteWidth?, getSpaceBaseMem?, Addr.offset, hRead,
    decodeScalar_s32_natToBytesLE]

def saxpyLoadedS32Value (values : List Int) (j : LaneId) : Value :=
  .s32 (s32Wrap (listIntGetD values j.val))

theorem readMem_saxpyStateFor_global_x
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    readMem? (saxpyStateFor n alpha xs ys) .global .s32
        (.global (saxpyXBase + j.val * 4)) =
      some (saxpyLoadedS32Value xs j) := by
  have hjLt : j.val < xs.length := by
    have hjn : j.val < n := by
      rw [saxpyActiveLanes, List.mem_pmap] at hj
      rcases hj with ⟨i, hi, hEq⟩
      have hiLt : i < n := List.mem_range.mp hi
      have hVal : i = j.val := congrArg Fin.val hEq
      omega
    omega
  have hReadX := readBytes?_writeS32Vector_getD
    ({} : ByteMem) saxpyXBase xs hjLt
  have hReadGlobal :
      readBytes? (saxpyGlobalBytesFor xs ys) (saxpyXBase + j.val * 4) 4 =
        some (natToBytesLE (signedToNat 32 (listIntGetD xs j.val)) 4) := by
    unfold saxpyGlobalBytesFor
    rw [readBytes?_writeS32Vector_before
      (writeS32Vector ({} : ByteMem) saxpyXBase xs) saxpyYBase
      (saxpyXBase + j.val * 4) 4 ys]
    · exact hReadX
    · have hjn : j.val < n := by
        rw [saxpyActiveLanes, List.mem_pmap] at hj
        rcases hj with ⟨i, hi, hEq⟩
        have hiLt : i < n := List.mem_range.mp hi
        have hVal : i = j.val := congrArg Fin.val hEq
        omega
      unfold saxpyXBase saxpyYBase
      omega
  have hAlign : (saxpyXBase + j.val * 4) % 4 = 0 := by
    unfold saxpyXBase
    omega
  have hReadState :
      readBytes? (saxpyStateFor n alpha xs ys).global.bytes (saxpyXBase + j.val * 4) 4 =
        some (natToBytesLE (signedToNat 32 (listIntGetD xs j.val)) 4) := by
    simpa [saxpyStateFor] using hReadGlobal
  simpa [saxpyLoadedS32Value] using
    readMem?_global_s32_of_readBytes (st := saxpyStateFor n alpha xs ys) hAlign hReadState

theorem readMem_saxpyStateFor_global_y
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hys : ys.length = n) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    readMem? (saxpyStateFor n alpha xs ys) .global .s32
        (.global (saxpyYBase + j.val * 4)) =
      some (saxpyLoadedS32Value ys j) := by
  have hjLt : j.val < ys.length := by
    have hjn : j.val < n := by
      rw [saxpyActiveLanes, List.mem_pmap] at hj
      rcases hj with ⟨i, hi, hEq⟩
      have hiLt : i < n := List.mem_range.mp hi
      have hVal : i = j.val := congrArg Fin.val hEq
      omega
    omega
  have hReadY := readBytes?_writeS32Vector_getD
    (writeS32Vector ({} : ByteMem) saxpyXBase xs) saxpyYBase ys hjLt
  have hReadGlobal :
      readBytes? (saxpyGlobalBytesFor xs ys) (saxpyYBase + j.val * 4) 4 =
        some (natToBytesLE (signedToNat 32 (listIntGetD ys j.val)) 4) := by
    unfold saxpyGlobalBytesFor
    exact hReadY
  have hAlign : (saxpyYBase + j.val * 4) % 4 = 0 := by
    unfold saxpyYBase
    omega
  have hReadState :
      readBytes? (saxpyStateFor n alpha xs ys).global.bytes (saxpyYBase + j.val * 4) 4 =
        some (natToBytesLE (signedToNat 32 (listIntGetD ys j.val)) 4) := by
    simpa [saxpyStateFor] using hReadGlobal
  simpa [saxpyLoadedS32Value] using
    readMem?_global_s32_of_readBytes (st := saxpyStateFor n alpha xs ys) hAlign hReadState

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

/-! ## Block-content extraction

The saxpy `kernelEnv.blocks["saxpyKernel"]?` lookup is a closed expression
that doesn't reduce in the kernel (it goes through `lowerKernelEnvCheckedD`
which is `def`-style, not `reducible`). `Block` also lacks `DecidableEq`
(because `Instr` references `Value` which contains `Float`/`Array`).

The pattern below sidesteps both: we name the looked-up block via
`Option.get` on a `native_decide`-proved `isSome`, then characterize its
content using only `Option.isNone`/`isSome` (always decidable) and small
discriminator booleans (decidable because they only inspect constructor
heads). The looked-up equality is recovered via `Option.eq_some_iff_get_eq`,
which doesn't need `DecidableEq` at all. -/

/-- The lowered BB0 of saxpy, extracted as a concrete `Block` via
`Option.get` on a `native_decide`-proved `isSome`. -/
def saxpyBB0 : Block :=
  ((PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel"]?).get
    (by native_decide)

/-- The lookup `kernelEnv.blocks["saxpyKernel"]?` equals `some saxpyBB0`.
Proved via `Option.eq_some_iff_get_eq` — no `DecidableEq` on `Block`. -/
theorem saxpyBB0_lookup :
    (PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel"]?
      = some saxpyBB0 :=
  Option.eq_some_iff_get_eq.mpr ⟨by native_decide, rfl⟩

/-- The state's lookup equals `some saxpyBB0` at any saxpy launch shape. -/
theorem saxpyStateFor_blocks_lookup (n : Nat) (alpha : Int) (xs ys : List Int) :
    (saxpyStateFor n alpha xs ys).kernelEnv.blocks["saxpyKernel"]?
      = some saxpyBB0 := by
  show (PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel"]?
        = some saxpyBB0
  exact saxpyBB0_lookup

/-! ### Per-instruction discriminators

Rather than equating `Instr`s (which needs `DecidableEq`), we expose small
boolean discriminators that inspect only the constructor head. `native_decide`
on these is enough to characterize each body slot in `saxpyBB0`, and a
`cases` on the actual `Instr` then yields the operand names. -/

/-- Boolean discriminator for `Instr.load`. -/
def Instr.isLoad : Instr → Bool
  | .load _ _ => true
  | _ => false

/-- Boolean discriminator for `Instr.cvta`. -/
def Instr.isCvta : Instr → Bool
  | .cvta _ _ _ => true
  | _ => false

/-- Boolean discriminator for `Instr.assignReg`. -/
def Instr.isAssignReg : Instr → Bool
  | .assignReg _ _ => true
  | _ => false

/-- Boolean discriminator for `Instr.assignPred`. -/
def Instr.isAssignPred : Instr → Bool
  | .assignPred _ _ => true
  | _ => false

/-- Boolean discriminator for `Instr.store`. -/
def Instr.isStore : Instr → Bool
  | .store _ _ => true
  | _ => false

/-- Boolean discriminator for `Terminator.cbr`. -/
def Terminator.isCbr : Terminator → Bool
  | .cbr _ _ _ => true
  | _ => false

/-- Boolean discriminator for `Terminator.br`. -/
def Terminator.isBr : Terminator → Bool
  | .br _ => true
  | _ => false

/-- Boolean discriminator for `Terminator.terminate`. -/
def Terminator.isTerminate : Terminator → Bool
  | .terminate => true
  | _ => false

/-- Project a `cbr` condition of the shape `.pred p`. -/
def Terminator.cbrCondPred? : Terminator → Option PredName
  | .cbr (.pred p) _ _ => some p
  | _ => none

/-- Project a `cbr` true label. -/
def Terminator.cbrTrueLabel? : Terminator → Option BlockLabel
  | .cbr _ tLabel _ => some tLabel
  | _ => none

/-- Project a `cbr` false label. -/
def Terminator.cbrFalseLabel? : Terminator → Option BlockLabel
  | .cbr _ _ fLabel => some fLabel
  | _ => none

/-! ### Per-instruction operand projections

For load/cvta/assignReg/etc., we project to the *operands* via small
functions that return `Option <small-DecidableEq-type>`. These pair with
`native_decide` to characterize each operand. -/

/-- Project a load's destination register. -/
def Instr.loadDst? : Instr → Option RegName
  | .load dst _ => some dst
  | _ => none

/-- Project a load's source `TypedAddr.space`. -/
def Instr.loadSrcSpace? : Instr → Option AddrSpace
  | .load _ src => some src.space
  | _ => none

/-- Project a load's source `TypedAddr.ty`. -/
def Instr.loadSrcTy? : Instr → Option ScalarTy
  | .load _ src => some src.ty
  | _ => none

/-- Project a load's source-address `.imm (.u64 v)` to `v`. -/
def Instr.loadSrcImmU64? : Instr → Option UInt64
  | .load _ { addr := .imm (.u64 v), .. } => some v
  | _ => none

/-- Project a load's source-address `.reg r` to `r`. -/
def Instr.loadSrcReg? : Instr → Option RegName
  | .load _ { addr := .reg r, .. } => some r
  | _ => none

/-- Project a `cvta` destination register. -/
def Instr.cvtaDst? : Instr → Option RegName
  | .cvta dst _ _ => some dst
  | _ => none

/-- Project a `cvta` target address space. -/
def Instr.cvtaSpace? : Instr → Option AddrSpace
  | .cvta _ space _ => some space
  | _ => none

/-- Project a `cvta` source of the shape `.reg r`. -/
def Instr.cvtaSrcReg? : Instr → Option RegName
  | .cvta _ _ (.reg r) => some r
  | _ => none

/-- Project an `assignReg` destination register. -/
def Instr.assignRegDst? : Instr → Option RegName
  | .assignReg dst _ => some dst
  | _ => none

/-- Project an `assignReg` RHS of the shape `.special s`. -/
def Instr.assignRegSpecial? : Instr → Option SpecialReg
  | .assignReg _ (.special s) => some s
  | _ => none

/-- Project an `assignReg` of the lowered signed mad-index shape. -/
def Instr.assignRegMadS32Regs? : Instr → Option (RegName × RegName × RegName × RegName)
  | .assignReg dst
      (.triop .mad
        (.unop (.cvt .s32) (.reg a))
        (.unop (.cvt .s32) (.reg b))
        (.unop (.cvt .s32) (.reg c))) => some (dst, a, b, c)
  | _ => none

/-- Project an `assignReg` of the fallthrough signed mad shape. -/
def Instr.assignRegMadS32RegCvtReg? : Instr → Option (RegName × RegName × RegName × RegName)
  | .assignReg dst
      (.triop .mad
        (.reg a)
        (.unop (.cvt .s32) (.reg b))
        (.reg c)) => some (dst, a, b, c)
  | _ => none

/-- Project an `assignReg` of the lowered `mul.wide.s32 reg, imm` shape. -/
def Instr.assignRegMulWideS32RegImm? : Instr → Option (RegName × RegName × Int)
  | .assignReg dst
      (.binop .mulWideS32
        (.unop (.cvt .s32) (.reg lhs))
        (.imm (.s32 rhs))) => some (dst, lhs, rhs)
  | _ => none

/-- Project an `assignReg` of a binary operation over two registers. -/
def Instr.assignRegBinopRegs? : Instr → Option (RegName × ScalarBinaryOp × RegName × RegName)
  | .assignReg dst (.binop op (.reg lhs) (.reg rhs)) => some (dst, op, lhs, rhs)
  | _ => none

/-- Project an `assignPred` of the lowered `setp.ge.s32` register shape. -/
def Instr.assignPredGeS32Regs? : Instr → Option (PredName × RegName × RegName)
  | .assignPred dst
      { op := .ge, lhs := .reg lhs, rhs := .unop (.cvt .s32) (.reg rhs) } =>
      some (dst, lhs, rhs)
  | _ => none

/-- Recover a concrete `Instr.load` from the small decidable load projections. -/
theorem Instr.load_eq_of_projections
    {i : Instr} {dst : RegName} {space : AddrSpace} {ty : ScalarTy} {imm : UInt64}
    (hLoad : i.isLoad = true)
    (hDst : i.loadDst? = some dst)
    (hSpace : i.loadSrcSpace? = some space)
    (hTy : i.loadSrcTy? = some ty)
    (hImm : i.loadSrcImmU64? = some imm) :
    i = .load dst { space := space, ty := ty, addr := .imm (.u64 imm) } := by
  generalize i = instr at hLoad hDst hSpace hTy hImm ⊢
  cases instr
  case load dst' src =>
    simp [Instr.loadDst?] at hDst
    obtain ⟨space', ty', addr⟩ := src
    simp [Instr.loadSrcSpace?] at hSpace
    simp [Instr.loadSrcTy?] at hTy
    subst hDst; subst hSpace; subst hTy
    cases addr
    case imm v =>
      cases v
      case u64 w =>
        simp [Instr.loadSrcImmU64?] at hImm
        subst hImm
        rfl
      all_goals (exfalso; exact absurd hImm (by simp [Instr.loadSrcImmU64?]))
    all_goals (exfalso; exact absurd hImm (by simp [Instr.loadSrcImmU64?]))
  all_goals (exfalso; exact absurd hLoad (by simp [Instr.isLoad]))

/-- Recover a concrete unguarded load `GInstr` from guard and load projections. -/
theorem GInstr.eq_unguarded_load_of_projections
    {gi : GInstr} {dst : RegName} {space : AddrSpace} {ty : ScalarTy} {imm : UInt64}
    (hGuard : gi.guard? = none)
    (hLoad : gi.instr.isLoad = true)
    (hDst : gi.instr.loadDst? = some dst)
    (hSpace : gi.instr.loadSrcSpace? = some space)
    (hTy : gi.instr.loadSrcTy? = some ty)
    (hImm : gi.instr.loadSrcImmU64? = some imm) :
    gi = { guard? := none
           instr := .load dst { space := space, ty := ty, addr := .imm (.u64 imm) } } := by
  have hInstr := Instr.load_eq_of_projections hLoad hDst hSpace hTy hImm
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete register-address `Instr.load` from small projections. -/
theorem Instr.load_reg_eq_of_projections
    {i : Instr} {dst addrReg : RegName} {space : AddrSpace} {ty : ScalarTy}
    (hLoad : i.isLoad = true)
    (hDst : i.loadDst? = some dst)
    (hSpace : i.loadSrcSpace? = some space)
    (hTy : i.loadSrcTy? = some ty)
    (hReg : i.loadSrcReg? = some addrReg) :
    i = .load dst { space := space, ty := ty, addr := .reg addrReg } := by
  generalize i = instr at hLoad hDst hSpace hTy hReg ⊢
  cases instr
  case load dst' src =>
    simp [Instr.loadDst?] at hDst
    obtain ⟨space', ty', addr⟩ := src
    simp [Instr.loadSrcSpace?] at hSpace
    simp [Instr.loadSrcTy?] at hTy
    subst hDst; subst hSpace; subst hTy
    cases addr
    case reg r =>
      simp [Instr.loadSrcReg?] at hReg
      subst hReg
      rfl
    all_goals (exfalso; exact absurd hReg (by simp [Instr.loadSrcReg?]))
  all_goals (exfalso; exact absurd hLoad (by simp [Instr.isLoad]))

/-- Recover a concrete unguarded register-address load `GInstr`. -/
theorem GInstr.eq_unguarded_load_reg_of_projections
    {gi : GInstr} {dst addrReg : RegName} {space : AddrSpace} {ty : ScalarTy}
    (hGuard : gi.guard? = none)
    (hLoad : gi.instr.isLoad = true)
    (hDst : gi.instr.loadDst? = some dst)
    (hSpace : gi.instr.loadSrcSpace? = some space)
    (hTy : gi.instr.loadSrcTy? = some ty)
    (hReg : gi.instr.loadSrcReg? = some addrReg) :
    gi = { guard? := none
           instr := .load dst { space := space, ty := ty, addr := .reg addrReg } } := by
  have hInstr := Instr.load_reg_eq_of_projections hLoad hDst hSpace hTy hReg
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete `Instr.cvta dst space (.reg src)` from small projections. -/
theorem Instr.cvta_reg_eq_of_projections
    {i : Instr} {dst src : RegName} {space : AddrSpace}
    (hCvta : i.isCvta = true)
    (hDst : i.cvtaDst? = some dst)
    (hSpace : i.cvtaSpace? = some space)
    (hSrc : i.cvtaSrcReg? = some src) :
    i = .cvta dst space (.reg src) := by
  generalize i = instr at hCvta hDst hSpace hSrc ⊢
  cases instr
  case cvta dst' space' rv =>
    simp [Instr.cvtaDst?] at hDst
    simp [Instr.cvtaSpace?] at hSpace
    subst hDst
    subst hSpace
    cases rv
    case reg r =>
      simp [Instr.cvtaSrcReg?] at hSrc
      subst hSrc
      rfl
    all_goals (exfalso; exact absurd hSrc (by simp [Instr.cvtaSrcReg?]))
  all_goals (exfalso; exact absurd hCvta (by simp [Instr.isCvta]))

/-- Recover a concrete unguarded `cvta` `GInstr` from guard and projections. -/
theorem GInstr.eq_unguarded_cvta_reg_of_projections
    {gi : GInstr} {dst src : RegName} {space : AddrSpace}
    (hGuard : gi.guard? = none)
    (hCvta : gi.instr.isCvta = true)
    (hDst : gi.instr.cvtaDst? = some dst)
    (hSpace : gi.instr.cvtaSpace? = some space)
    (hSrc : gi.instr.cvtaSrcReg? = some src) :
    gi = { guard? := none, instr := .cvta dst space (.reg src) } := by
  have hInstr := Instr.cvta_reg_eq_of_projections hCvta hDst hSpace hSrc
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete `Instr.assignReg dst (.special s)` from small projections. -/
theorem Instr.assignReg_special_eq_of_projections
    {i : Instr} {dst : RegName} {special : SpecialReg}
    (hAssign : i.isAssignReg = true)
    (hDst : i.assignRegDst? = some dst)
    (hSpecial : i.assignRegSpecial? = some special) :
    i = .assignReg dst (.special special) := by
  generalize i = instr at hAssign hDst hSpecial ⊢
  cases instr
  case assignReg dst' rhs =>
    simp [Instr.assignRegDst?] at hDst
    subst hDst
    cases rhs
    case special s =>
      simp [Instr.assignRegSpecial?] at hSpecial
      subst hSpecial
      rfl
    all_goals (exfalso; exact absurd hSpecial (by simp [Instr.assignRegSpecial?]))
  all_goals (exfalso; exact absurd hAssign (by simp [Instr.isAssignReg]))

/-- Recover a concrete unguarded special-register move from guard and projections. -/
theorem GInstr.eq_unguarded_assignReg_special_of_projections
    {gi : GInstr} {dst : RegName} {special : SpecialReg}
    (hGuard : gi.guard? = none)
    (hAssign : gi.instr.isAssignReg = true)
    (hDst : gi.instr.assignRegDst? = some dst)
    (hSpecial : gi.instr.assignRegSpecial? = some special) :
    gi = { guard? := none, instr := .assignReg dst (.special special) } := by
  have hInstr := Instr.assignReg_special_eq_of_projections hAssign hDst hSpecial
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete lowered signed mad-index `assignReg` from its projection. -/
theorem Instr.assignReg_madS32_eq_of_projection
    {i : Instr} {dst a b c : RegName}
    (hProj : i.assignRegMadS32Regs? = some (dst, a, b, c)) :
    i = .assignReg dst
      (.triop .mad
        (.unop (.cvt .s32) (.reg a))
        (.unop (.cvt .s32) (.reg b))
        (.unop (.cvt .s32) (.reg c))) := by
  cases i <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i _ rhs
  cases rhs <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i op x y z
  cases op <;> simp_all [Instr.assignRegMadS32Regs?]
  cases x <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i opx xx
  cases opx <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i tyx
  cases tyx <;> simp_all [Instr.assignRegMadS32Regs?]
  cases xx <;> simp_all [Instr.assignRegMadS32Regs?]
  cases y <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i opy yy
  cases opy <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i tyy
  cases tyy <;> simp_all [Instr.assignRegMadS32Regs?]
  cases yy <;> simp_all [Instr.assignRegMadS32Regs?]
  cases z <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i opz zz
  cases opz <;> simp_all [Instr.assignRegMadS32Regs?]
  rename_i tyz
  cases tyz <;> simp_all [Instr.assignRegMadS32Regs?]
  cases zz <;> simp_all [Instr.assignRegMadS32Regs?]

/-- Recover a concrete unguarded lowered signed mad-index `GInstr`. -/
theorem GInstr.eq_unguarded_assignReg_madS32_of_projection
    {gi : GInstr} {dst a b c : RegName}
    (hGuard : gi.guard? = none)
    (hProj : gi.instr.assignRegMadS32Regs? = some (dst, a, b, c)) :
    gi = { guard? := none
           instr := .assignReg dst
             (.triop .mad
               (.unop (.cvt .s32) (.reg a))
               (.unop (.cvt .s32) (.reg b))
               (.unop (.cvt .s32) (.reg c))) } := by
  have hInstr := Instr.assignReg_madS32_eq_of_projection hProj
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete fallthrough signed mad `assignReg` from its projection. -/
theorem Instr.assignReg_madS32_reg_cvt_reg_eq_of_projection
    {i : Instr} {dst a b c : RegName}
    (hProj : i.assignRegMadS32RegCvtReg? = some (dst, a, b, c)) :
    i = .assignReg dst
      (.triop .mad
        (.reg a)
        (.unop (.cvt .s32) (.reg b))
        (.reg c)) := by
  cases i <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  rename_i _ rhs
  cases rhs <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  rename_i op x y z
  cases op <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  cases x <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  cases y <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  rename_i opy yy
  cases opy <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  rename_i tyy
  cases tyy <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  cases yy <;> simp_all [Instr.assignRegMadS32RegCvtReg?]
  cases z <;> simp_all [Instr.assignRegMadS32RegCvtReg?]

/-- Recover a concrete unguarded fallthrough signed mad `GInstr`. -/
theorem GInstr.eq_unguarded_assignReg_madS32_reg_cvt_reg_of_projection
    {gi : GInstr} {dst a b c : RegName}
    (hGuard : gi.guard? = none)
    (hProj : gi.instr.assignRegMadS32RegCvtReg? = some (dst, a, b, c)) :
    gi = { guard? := none
           instr := .assignReg dst
             (.triop .mad
               (.reg a)
               (.unop (.cvt .s32) (.reg b))
               (.reg c)) } := by
  have hInstr := Instr.assignReg_madS32_reg_cvt_reg_eq_of_projection hProj
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete lowered `mul.wide.s32` `assignReg` from its projection. -/
theorem Instr.assignReg_mulWideS32_reg_imm_eq_of_projection
    {i : Instr} {dst lhs : RegName} {rhs : Int}
    (hProj : i.assignRegMulWideS32RegImm? = some (dst, lhs, rhs)) :
    i = .assignReg dst
      (.binop .mulWideS32
        (.unop (.cvt .s32) (.reg lhs))
        (.imm (.s32 rhs))) := by
  cases i <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  rename_i _ rv
  cases rv <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  rename_i op a b
  cases op <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  cases a <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  rename_i opA aa
  cases opA <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  rename_i tyA
  cases tyA <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  cases aa <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  cases b <;> simp_all [Instr.assignRegMulWideS32RegImm?]
  rename_i v
  cases v <;> simp_all [Instr.assignRegMulWideS32RegImm?]

/-- Recover a concrete unguarded lowered `mul.wide.s32` `GInstr`. -/
theorem GInstr.eq_unguarded_assignReg_mulWideS32_reg_imm_of_projection
    {gi : GInstr} {dst lhs : RegName} {rhs : Int}
    (hGuard : gi.guard? = none)
    (hProj : gi.instr.assignRegMulWideS32RegImm? = some (dst, lhs, rhs)) :
    gi = { guard? := none
           instr := .assignReg dst
             (.binop .mulWideS32
               (.unop (.cvt .s32) (.reg lhs))
               (.imm (.s32 rhs))) } := by
  have hInstr := Instr.assignReg_mulWideS32_reg_imm_eq_of_projection hProj
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete binary-register `assignReg` from its projection. -/
theorem Instr.assignReg_binop_regs_eq_of_projection
    {i : Instr} {dst lhs rhs : RegName} {op : ScalarBinaryOp}
    (hProj : i.assignRegBinopRegs? = some (dst, op, lhs, rhs)) :
    i = .assignReg dst (.binop op (.reg lhs) (.reg rhs)) := by
  cases i <;> simp_all [Instr.assignRegBinopRegs?]
  rename_i _ rv
  cases rv <;> simp_all [Instr.assignRegBinopRegs?]
  rename_i op' a b
  cases a <;> simp_all [Instr.assignRegBinopRegs?]
  cases b <;> simp_all [Instr.assignRegBinopRegs?]

/-- Recover a concrete unguarded binary-register `assignReg` `GInstr`. -/
theorem GInstr.eq_unguarded_assignReg_binop_regs_of_projection
    {gi : GInstr} {dst lhs rhs : RegName} {op : ScalarBinaryOp}
    (hGuard : gi.guard? = none)
    (hProj : gi.instr.assignRegBinopRegs? = some (dst, op, lhs, rhs)) :
    gi = { guard? := none, instr := .assignReg dst (.binop op (.reg lhs) (.reg rhs)) } := by
  have hInstr := Instr.assignReg_binop_regs_eq_of_projection hProj
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete lowered `setp.ge.s32` `assignPred` from its projection. -/
theorem Instr.assignPred_geS32_eq_of_projection
    {i : Instr} {dst : PredName} {lhs rhs : RegName}
    (hProj : i.assignPredGeS32Regs? = some (dst, lhs, rhs)) :
    i = .assignPred dst
      { op := .ge
        lhs := .reg lhs
        rhs := .unop (.cvt .s32) (.reg rhs) } := by
  cases i <;> simp_all [Instr.assignPredGeS32Regs?]
  rename_i _ cmp
  cases cmp with
  | mk op lhs' rhs' =>
      cases op <;> simp_all [Instr.assignPredGeS32Regs?]
      cases lhs' <;> simp_all [Instr.assignPredGeS32Regs?]
      cases rhs' <;> simp_all [Instr.assignPredGeS32Regs?]
      rename_i opR rr
      cases opR <;> simp_all [Instr.assignPredGeS32Regs?]
      rename_i tyR
      cases tyR <;> simp_all [Instr.assignPredGeS32Regs?]
      cases rr <;> simp_all [Instr.assignPredGeS32Regs?]

/-- Recover a concrete unguarded lowered `setp.ge.s32` `GInstr`. -/
theorem GInstr.eq_unguarded_assignPred_geS32_of_projection
    {gi : GInstr} {dst : PredName} {lhs rhs : RegName}
    (hGuard : gi.guard? = none)
    (hProj : gi.instr.assignPredGeS32Regs? = some (dst, lhs, rhs)) :
    gi = { guard? := none
           instr := .assignPred dst
             { op := .ge
               lhs := .reg lhs
               rhs := .unop (.cvt .s32) (.reg rhs) } } := by
  have hInstr := Instr.assignPred_geS32_eq_of_projection hProj
  rcases hgi : gi with ⟨g, i⟩
  rw [hgi] at hGuard hInstr
  simp at hGuard hInstr
  subst hGuard
  subst hInstr
  rfl

/-- Recover a concrete predicate-conditioned `cbr` from small projections. -/
theorem Terminator.cbr_pred_eq_of_projections
    {term : Terminator} {pred : PredName} {tLabel fLabel : BlockLabel}
    (hCbr : term.isCbr = true)
    (hPred : term.cbrCondPred? = some pred)
    (hTrue : term.cbrTrueLabel? = some tLabel)
    (hFalse : term.cbrFalseLabel? = some fLabel) :
    term = .cbr (.pred pred) tLabel fLabel := by
  cases term <;> simp_all [Terminator.isCbr, Terminator.cbrCondPred?,
    Terminator.cbrTrueLabel?, Terminator.cbrFalseLabel?]
  rename_i cond _ _
  cases cond <;> simp_all [Terminator.cbrCondPred?]

/-! ### Body-slot 0: `ld.param.u32 %r2, [param_0]`

The first instruction of saxpy's entry block. -/

/-- `saxpyBB0.body[0]?` exists. -/
theorem saxpyBB0_body0_isSome : (saxpyBB0.body[0]?).isSome = true := by
  unfold saxpyBB0; native_decide

/-- Concrete extraction of `saxpyBB0.body[0]?` as a `GInstr`. -/
def saxpyBB0_gi0 : GInstr := saxpyBB0.body[0]?.get saxpyBB0_body0_isSome

theorem saxpyBB0_body0 : saxpyBB0.body[0]? = some saxpyBB0_gi0 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body0_isSome, rfl⟩

/-- Concrete first instruction of BB0: `ld.param.u32 %r2, [param_0]`. -/
def saxpyBB0_load_param0 : GInstr :=
  { guard? := none
    instr := .load "r2"
      { space := .param, ty := .u32, addr := .imm (.u64 0) } }

theorem saxpyBB0_gi0_isLoad : saxpyBB0_gi0.instr.isLoad = true := by
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

theorem saxpyBB0_gi0_guard_none : saxpyBB0_gi0.guard? = none := by
  rw [← Option.isNone_iff_eq_none]
  show saxpyBB0_gi0.guard?.isNone = true
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

theorem saxpyBB0_gi0_loadDst : saxpyBB0_gi0.instr.loadDst? = some "r2" := by
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

theorem saxpyBB0_gi0_loadSrcSpace : saxpyBB0_gi0.instr.loadSrcSpace? = some .param := by
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

theorem saxpyBB0_gi0_loadSrcTy : saxpyBB0_gi0.instr.loadSrcTy? = some .u32 := by
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

theorem saxpyBB0_gi0_loadSrcImm : saxpyBB0_gi0.instr.loadSrcImmU64? = some 0 := by
  unfold saxpyBB0_gi0 saxpyBB0; native_decide

/-- **Full equality on `saxpyBB0_gi0.instr`.** Combines the per-operand
projections via a discriminator-driven `cases` chain to recover the
complete `.load "r2" { … }` shape. -/
theorem saxpyBB0_gi0_instr_eq :
    saxpyBB0_gi0.instr = .load "r2"
      { space := .param, ty := .u32, addr := .imm (.u64 0) } := by
  have hLoad := saxpyBB0_gi0_isLoad
  have hDst := saxpyBB0_gi0_loadDst
  have hSp := saxpyBB0_gi0_loadSrcSpace
  have hTy := saxpyBB0_gi0_loadSrcTy
  have hImm := saxpyBB0_gi0_loadSrcImm
  generalize saxpyBB0_gi0.instr = i at hLoad hDst hSp hTy hImm ⊢
  cases i
  case load dst src =>
    simp [Instr.loadDst?] at hDst
    obtain ⟨space, ty, addr⟩ := src
    simp [Instr.loadSrcSpace?] at hSp
    simp [Instr.loadSrcTy?] at hTy
    subst hDst; subst hSp; subst hTy
    cases addr
    case imm v =>
      cases v
      case u64 w =>
        simp [Instr.loadSrcImmU64?] at hImm
        subst hImm; rfl
      all_goals (exfalso; exact absurd hImm (by simp [Instr.loadSrcImmU64?]))
    all_goals (exfalso; exact absurd hImm (by simp [Instr.loadSrcImmU64?]))
  all_goals (exfalso; exact absurd hLoad (by simp [Instr.isLoad]))

/-- **Full equality on `saxpyBB0_gi0`.** -/
theorem saxpyBB0_gi0_eq :
    saxpyBB0_gi0 =
      { guard? := none
        instr := .load "r2"
          { space := .param, ty := .u32, addr := .imm (.u64 0) } } := by
  have hg := saxpyBB0_gi0_guard_none
  have hi := saxpyBB0_gi0_instr_eq
  rcases hh : saxpyBB0_gi0 with ⟨g, i⟩
  rw [hh] at hg hi
  simp at hg hi
  subst hg; subst hi; rfl

theorem saxpyBB0_body0_load_param0 :
    saxpyBB0.body[0]? = some saxpyBB0_load_param0 := by
  unfold saxpyBB0_load_param0
  rw [saxpyBB0_body0, saxpyBB0_gi0_eq]

/-- Concrete second instruction of BB0: `ld.param.s32 %r6, [param_1]`. -/
def saxpyBB0_load_param1 : GInstr :=
  { guard? := none
    instr := .load "r6"
      { space := .param, ty := .s32, addr := .imm (.u64 4) } }

/-- `saxpyBB0.body[1]?` exists. -/
theorem saxpyBB0_body1_isSome : (saxpyBB0.body[1]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi1 : GInstr := saxpyBB0.body[1]?.get saxpyBB0_body1_isSome

theorem saxpyBB0_body1 : saxpyBB0.body[1]? = some saxpyBB0_gi1 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body1_isSome, rfl⟩

theorem saxpyBB0_gi1_isLoad : saxpyBB0_gi1.instr.isLoad = true := by
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_guard_none : saxpyBB0_gi1.guard? = none := by
  rw [← Option.isNone_iff_eq_none]
  show saxpyBB0_gi1.guard?.isNone = true
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_loadDst : saxpyBB0_gi1.instr.loadDst? = some "r6" := by
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_loadSrcSpace : saxpyBB0_gi1.instr.loadSrcSpace? = some .param := by
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_loadSrcTy : saxpyBB0_gi1.instr.loadSrcTy? = some .s32 := by
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_loadSrcImm : saxpyBB0_gi1.instr.loadSrcImmU64? = some 4 := by
  unfold saxpyBB0_gi1 saxpyBB0; native_decide

theorem saxpyBB0_gi1_eq : saxpyBB0_gi1 = saxpyBB0_load_param1 := by
  unfold saxpyBB0_load_param1
  exact GInstr.eq_unguarded_load_of_projections
    saxpyBB0_gi1_guard_none
    saxpyBB0_gi1_isLoad
    saxpyBB0_gi1_loadDst
    saxpyBB0_gi1_loadSrcSpace
    saxpyBB0_gi1_loadSrcTy
    saxpyBB0_gi1_loadSrcImm

theorem saxpyBB0_body1_load_param1 :
    saxpyBB0.body[1]? = some saxpyBB0_load_param1 := by
  rw [saxpyBB0_body1, saxpyBB0_gi1_eq]

/-- Concrete third instruction of BB0: `ld.param.u64 %rd1, [param_2]`. -/
def saxpyBB0_load_param2 : GInstr :=
  { guard? := none
    instr := .load "rd1"
      { space := .param, ty := .u64, addr := .imm (.u64 8) } }

theorem saxpyBB0_body2_isSome : (saxpyBB0.body[2]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi2 : GInstr := saxpyBB0.body[2]?.get saxpyBB0_body2_isSome

theorem saxpyBB0_body2 : saxpyBB0.body[2]? = some saxpyBB0_gi2 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body2_isSome, rfl⟩

theorem saxpyBB0_gi2_eq : saxpyBB0_gi2 = saxpyBB0_load_param2 := by
  unfold saxpyBB0_load_param2
  apply GInstr.eq_unguarded_load_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi2.guard?.isNone = true
    unfold saxpyBB0_gi2 saxpyBB0; native_decide
  · unfold saxpyBB0_gi2 saxpyBB0; native_decide
  · unfold saxpyBB0_gi2 saxpyBB0; native_decide
  · unfold saxpyBB0_gi2 saxpyBB0; native_decide
  · unfold saxpyBB0_gi2 saxpyBB0; native_decide
  · unfold saxpyBB0_gi2 saxpyBB0; native_decide

theorem saxpyBB0_body2_load_param2 :
    saxpyBB0.body[2]? = some saxpyBB0_load_param2 := by
  rw [saxpyBB0_body2, saxpyBB0_gi2_eq]

/-- Concrete fourth instruction of BB0: `ld.param.u64 %rd2, [param_3]`. -/
def saxpyBB0_load_param3 : GInstr :=
  { guard? := none
    instr := .load "rd2"
      { space := .param, ty := .u64, addr := .imm (.u64 16) } }

theorem saxpyBB0_body3_isSome : (saxpyBB0.body[3]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi3 : GInstr := saxpyBB0.body[3]?.get saxpyBB0_body3_isSome

theorem saxpyBB0_body3 : saxpyBB0.body[3]? = some saxpyBB0_gi3 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body3_isSome, rfl⟩

theorem saxpyBB0_gi3_eq : saxpyBB0_gi3 = saxpyBB0_load_param3 := by
  unfold saxpyBB0_load_param3
  apply GInstr.eq_unguarded_load_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi3.guard?.isNone = true
    unfold saxpyBB0_gi3 saxpyBB0; native_decide
  · unfold saxpyBB0_gi3 saxpyBB0; native_decide
  · unfold saxpyBB0_gi3 saxpyBB0; native_decide
  · unfold saxpyBB0_gi3 saxpyBB0; native_decide
  · unfold saxpyBB0_gi3 saxpyBB0; native_decide
  · unfold saxpyBB0_gi3 saxpyBB0; native_decide

theorem saxpyBB0_body3_load_param3 :
    saxpyBB0.body[3]? = some saxpyBB0_load_param3 := by
  rw [saxpyBB0_body3, saxpyBB0_gi3_eq]

/-- Concrete fifth instruction of BB0: `ld.param.u64 %rd3, [param_4]`. -/
def saxpyBB0_load_param4 : GInstr :=
  { guard? := none
    instr := .load "rd3"
      { space := .param, ty := .u64, addr := .imm (.u64 24) } }

theorem saxpyBB0_body4_isSome : (saxpyBB0.body[4]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi4 : GInstr := saxpyBB0.body[4]?.get saxpyBB0_body4_isSome

theorem saxpyBB0_body4 : saxpyBB0.body[4]? = some saxpyBB0_gi4 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body4_isSome, rfl⟩

theorem saxpyBB0_gi4_eq : saxpyBB0_gi4 = saxpyBB0_load_param4 := by
  unfold saxpyBB0_load_param4
  apply GInstr.eq_unguarded_load_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi4.guard?.isNone = true
    unfold saxpyBB0_gi4 saxpyBB0; native_decide
  · unfold saxpyBB0_gi4 saxpyBB0; native_decide
  · unfold saxpyBB0_gi4 saxpyBB0; native_decide
  · unfold saxpyBB0_gi4 saxpyBB0; native_decide
  · unfold saxpyBB0_gi4 saxpyBB0; native_decide
  · unfold saxpyBB0_gi4 saxpyBB0; native_decide

theorem saxpyBB0_body4_load_param4 :
    saxpyBB0.body[4]? = some saxpyBB0_load_param4 := by
  rw [saxpyBB0_body4, saxpyBB0_gi4_eq]

/-- Concrete sixth instruction of BB0: `mov.u32 %r3, %ctaid.x`. -/
def saxpyBB0_mov_ctaidX : GInstr :=
  { guard? := none
    instr := .assignReg "r3" (.special .ctaidX) }

theorem saxpyBB0_body5_isSome : (saxpyBB0.body[5]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi5 : GInstr := saxpyBB0.body[5]?.get saxpyBB0_body5_isSome

theorem saxpyBB0_body5 : saxpyBB0.body[5]? = some saxpyBB0_gi5 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body5_isSome, rfl⟩

theorem saxpyBB0_gi5_eq : saxpyBB0_gi5 = saxpyBB0_mov_ctaidX := by
  unfold saxpyBB0_mov_ctaidX
  apply GInstr.eq_unguarded_assignReg_special_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi5.guard?.isNone = true
    unfold saxpyBB0_gi5 saxpyBB0; native_decide
  · unfold saxpyBB0_gi5 saxpyBB0; native_decide
  · unfold saxpyBB0_gi5 saxpyBB0; native_decide
  · unfold saxpyBB0_gi5 saxpyBB0; native_decide

theorem saxpyBB0_body5_mov_ctaidX :
    saxpyBB0.body[5]? = some saxpyBB0_mov_ctaidX := by
  rw [saxpyBB0_body5, saxpyBB0_gi5_eq]

/-- Concrete seventh instruction of BB0: `mov.u32 %r4, %ntid.x`. -/
def saxpyBB0_mov_ntidX : GInstr :=
  { guard? := none
    instr := .assignReg "r4" (.special .ntidX) }

theorem saxpyBB0_body6_isSome : (saxpyBB0.body[6]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi6 : GInstr := saxpyBB0.body[6]?.get saxpyBB0_body6_isSome

theorem saxpyBB0_body6 : saxpyBB0.body[6]? = some saxpyBB0_gi6 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body6_isSome, rfl⟩

theorem saxpyBB0_gi6_eq : saxpyBB0_gi6 = saxpyBB0_mov_ntidX := by
  unfold saxpyBB0_mov_ntidX
  apply GInstr.eq_unguarded_assignReg_special_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi6.guard?.isNone = true
    unfold saxpyBB0_gi6 saxpyBB0; native_decide
  · unfold saxpyBB0_gi6 saxpyBB0; native_decide
  · unfold saxpyBB0_gi6 saxpyBB0; native_decide
  · unfold saxpyBB0_gi6 saxpyBB0; native_decide

theorem saxpyBB0_body6_mov_ntidX :
    saxpyBB0.body[6]? = some saxpyBB0_mov_ntidX := by
  rw [saxpyBB0_body6, saxpyBB0_gi6_eq]

/-- Concrete eighth instruction of BB0: `mov.u32 %r5, %tid.x`. -/
def saxpyBB0_mov_tidX : GInstr :=
  { guard? := none
    instr := .assignReg "r5" (.special .tidX) }

theorem saxpyBB0_body7_isSome : (saxpyBB0.body[7]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi7 : GInstr := saxpyBB0.body[7]?.get saxpyBB0_body7_isSome

theorem saxpyBB0_body7 : saxpyBB0.body[7]? = some saxpyBB0_gi7 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body7_isSome, rfl⟩

theorem saxpyBB0_gi7_eq : saxpyBB0_gi7 = saxpyBB0_mov_tidX := by
  unfold saxpyBB0_mov_tidX
  apply GInstr.eq_unguarded_assignReg_special_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi7.guard?.isNone = true
    unfold saxpyBB0_gi7 saxpyBB0; native_decide
  · unfold saxpyBB0_gi7 saxpyBB0; native_decide
  · unfold saxpyBB0_gi7 saxpyBB0; native_decide
  · unfold saxpyBB0_gi7 saxpyBB0; native_decide

theorem saxpyBB0_body7_mov_tidX :
    saxpyBB0.body[7]? = some saxpyBB0_mov_tidX := by
  rw [saxpyBB0_body7, saxpyBB0_gi7_eq]

/-- Lowered RHS of `mad.lo.s32 %r1, %r3, %r4, %r5`. -/
def saxpyMadIndexRhs : RValue :=
  .triop .mad
    (.unop (.cvt .s32) (.reg "r3"))
    (.unop (.cvt .s32) (.reg "r4"))
    (.unop (.cvt .s32) (.reg "r5"))

/-- Concrete ninth instruction of BB0: `mad.lo.s32 %r1, %r3, %r4, %r5`. -/
def saxpyBB0_mad_index : GInstr :=
  { guard? := none
    instr := .assignReg "r1" saxpyMadIndexRhs }

theorem saxpyBB0_body8_isSome : (saxpyBB0.body[8]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi8 : GInstr := saxpyBB0.body[8]?.get saxpyBB0_body8_isSome

theorem saxpyBB0_body8 : saxpyBB0.body[8]? = some saxpyBB0_gi8 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body8_isSome, rfl⟩

theorem saxpyBB0_gi8_eq : saxpyBB0_gi8 = saxpyBB0_mad_index := by
  unfold saxpyBB0_mad_index saxpyMadIndexRhs
  apply GInstr.eq_unguarded_assignReg_madS32_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi8.guard?.isNone = true
    unfold saxpyBB0_gi8 saxpyBB0; native_decide
  · unfold saxpyBB0_gi8 saxpyBB0; native_decide

theorem saxpyBB0_body8_mad_index :
    saxpyBB0.body[8]? = some saxpyBB0_mad_index := by
  rw [saxpyBB0_body8, saxpyBB0_gi8_eq]

/-- Lowered comparison for `setp.ge.s32 %p1, %r1, %r2`. -/
def saxpySetpGeS32IndexLen : CmpExpr :=
  { op := .ge
    lhs := .reg "r1"
    rhs := .unop (.cvt .s32) (.reg "r2") }

/-- Concrete tenth instruction of BB0: `setp.ge.s32 %p1, %r1, %r2`. -/
def saxpyBB0_setp_ge_index_len : GInstr :=
  { guard? := none
    instr := .assignPred "p1" saxpySetpGeS32IndexLen }

theorem saxpyBB0_body9_isSome : (saxpyBB0.body[9]?).isSome = true := by
  unfold saxpyBB0; native_decide

def saxpyBB0_gi9 : GInstr := saxpyBB0.body[9]?.get saxpyBB0_body9_isSome

theorem saxpyBB0_body9 : saxpyBB0.body[9]? = some saxpyBB0_gi9 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body9_isSome, rfl⟩

theorem saxpyBB0_gi9_eq : saxpyBB0_gi9 = saxpyBB0_setp_ge_index_len := by
  unfold saxpyBB0_setp_ge_index_len saxpySetpGeS32IndexLen
  apply GInstr.eq_unguarded_assignPred_geS32_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyBB0_gi9.guard?.isNone = true
    unfold saxpyBB0_gi9 saxpyBB0; native_decide
  · unfold saxpyBB0_gi9 saxpyBB0; native_decide

theorem saxpyBB0_body9_setp_ge_index_len :
    saxpyBB0.body[9]? = some saxpyBB0_setp_ge_index_len := by
  rw [saxpyBB0_body9, saxpyBB0_gi9_eq]

/-- BB0 has no body instruction at PC slot 10; execution moves to its terminator. -/
theorem saxpyBB0_body10_none : saxpyBB0.body[10]? = none := by
  rw [← Option.isNone_iff_eq_none]
  show (saxpyBB0.body[10]?).isNone = true
  unfold saxpyBB0
  native_decide

/-- Concrete BB0 terminator: branch to exit if `%p1`, otherwise fall through. -/
def saxpyBB0_cbr_index_guard : Terminator :=
  .cbr (.pred "p1") "$L__BB0_2" "saxpyKernel$fallthrough0"

theorem saxpyBB0_term_cbr_index_guard :
    saxpyBB0.term = saxpyBB0_cbr_index_guard := by
  unfold saxpyBB0_cbr_index_guard
  apply Terminator.cbr_pred_eq_of_projections
  · unfold saxpyBB0; native_decide
  · unfold saxpyBB0; native_decide
  · unfold saxpyBB0; native_decide
  · unfold saxpyBB0; native_decide

/-! ### Fallthrough block extraction -/

/-- The lowered fallthrough block of saxpy, reached when `%p1` is false. -/
def saxpyFallthrough0 : Block :=
  ((PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel$fallthrough0"]?).get
    (by native_decide)

theorem saxpyFallthrough0_lookup :
    (PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel$fallthrough0"]?
      = some saxpyFallthrough0 :=
  Option.eq_some_iff_get_eq.mpr ⟨by native_decide, rfl⟩

theorem saxpyStateFor_fallthrough0_lookup (n : Nat) (alpha : Int) (xs ys : List Int) :
    (saxpyStateFor n alpha xs ys).kernelEnv.blocks["saxpyKernel$fallthrough0"]?
      = some saxpyFallthrough0 := by
  show (PTX.lowerKernelEnvCheckedD saxpyKernel).blocks["saxpyKernel$fallthrough0"]?
        = some saxpyFallthrough0
  exact saxpyFallthrough0_lookup

/-- Concrete first fallthrough instruction: `cvta.to.global.u64 %rd4, %rd1`. -/
def saxpyFallthrough0_cvta_rd4 : GInstr :=
  { guard? := none, instr := .cvta "rd4" .global (.reg "rd1") }

theorem saxpyFallthrough0_body0_isSome : (saxpyFallthrough0.body[0]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi0 : GInstr :=
  saxpyFallthrough0.body[0]?.get saxpyFallthrough0_body0_isSome

theorem saxpyFallthrough0_body0 : saxpyFallthrough0.body[0]? = some saxpyFallthrough0_gi0 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body0_isSome, rfl⟩

theorem saxpyFallthrough0_gi0_eq :
    saxpyFallthrough0_gi0 = saxpyFallthrough0_cvta_rd4 := by
  unfold saxpyFallthrough0_cvta_rd4
  apply GInstr.eq_unguarded_cvta_reg_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi0.guard?.isNone = true
    unfold saxpyFallthrough0_gi0 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi0 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi0 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi0 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi0 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body0_cvta_rd4 :
    saxpyFallthrough0.body[0]? = some saxpyFallthrough0_cvta_rd4 := by
  rw [saxpyFallthrough0_body0, saxpyFallthrough0_gi0_eq]

/-- Lowered RHS of `mul.wide.s32 %rd5, %r1, 4`. -/
def saxpyIndexByteOffsetRhs : RValue :=
  .binop .mulWideS32
    (.unop (.cvt .s32) (.reg "r1"))
    (.imm (.s32 4))

/-- Byte offset of lane `j` as the signed 64-bit value produced by `mul.wide.s32`. -/
@[irreducible]
def saxpyLaneByteOffsetValue (j : LaneId) : Value :=
  (evalBinary? .mulWideS32 (.s32 (Int.ofNat j.val)) (.s32 4)).getD (.s64 0)

theorem saxpyLaneByteOffsetValue_eq_evalBinary (j : LaneId) :
    evalBinary? .mulWideS32 (.s32 (Int.ofNat j.val)) (.s32 4) =
      some (saxpyLaneByteOffsetValue j) := by
  unfold saxpyLaneByteOffsetValue
  rfl

theorem evalRValue_saxpyIndexByteOffsetRhs_of_r1
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hR1 : ls.regs["r1"]? = some (.s32 (Int.ofNat j.val))) :
    evalRValue? st 0 0 j saxpyIndexByteOffsetRhs =
      some (saxpyLaneByteOffsetValue j) := by
  simpa [saxpyIndexByteOffsetRhs, readReg, evalRValue?, evalUnary?, hGet, hR1,
    Typing.valueType?] using saxpyLaneByteOffsetValue_eq_evalBinary j

/-- Concrete second fallthrough instruction: `mul.wide.s32 %rd5, %r1, 4`. -/
def saxpyFallthrough0_mul_wide_index : GInstr :=
  { guard? := none, instr := .assignReg "rd5" saxpyIndexByteOffsetRhs }

theorem saxpyFallthrough0_body1_isSome : (saxpyFallthrough0.body[1]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi1 : GInstr :=
  saxpyFallthrough0.body[1]?.get saxpyFallthrough0_body1_isSome

theorem saxpyFallthrough0_body1 : saxpyFallthrough0.body[1]? = some saxpyFallthrough0_gi1 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body1_isSome, rfl⟩

theorem saxpyFallthrough0_gi1_eq :
    saxpyFallthrough0_gi1 = saxpyFallthrough0_mul_wide_index := by
  unfold saxpyFallthrough0_mul_wide_index saxpyIndexByteOffsetRhs
  apply GInstr.eq_unguarded_assignReg_mulWideS32_reg_imm_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi1.guard?.isNone = true
    unfold saxpyFallthrough0_gi1 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi1 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body1_mul_wide_index :
    saxpyFallthrough0.body[1]? = some saxpyFallthrough0_mul_wide_index := by
  rw [saxpyFallthrough0_body1, saxpyFallthrough0_gi1_eq]

/-- Lowered RHS of `add.s64 %rd6, %rd4, %rd5`. -/
def saxpyXElementAddrRhs : RValue :=
  .binop .add (.reg "rd4") (.reg "rd5")

/-- X-array element address value for lane `j`. -/
@[irreducible]
def saxpyXElementAddrValue (j : LaneId) : Value :=
  (evalBinary? .add (.gaddr .global saxpyXBase) (saxpyLaneByteOffsetValue j)).getD
    (.gaddr .global saxpyXBase)

theorem saxpyXElementAddrValue_eq_evalBinary (j : LaneId) :
    evalBinary? .add (.gaddr .global saxpyXBase) (saxpyLaneByteOffsetValue j) =
      some (saxpyXElementAddrValue j) := by
  unfold saxpyXElementAddrValue saxpyLaneByteOffsetValue
  rfl

theorem evalRValue_saxpyXElementAddrRhs_of_regs
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hRd4 : ls.regs["rd4"]? = some (.gaddr .global saxpyXBase))
    (hRd5 : ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j)) :
    evalRValue? st 0 0 j saxpyXElementAddrRhs =
      some (saxpyXElementAddrValue j) := by
  simpa [saxpyXElementAddrRhs, readReg, evalRValue?, hGet, hRd4, hRd5]
    using saxpyXElementAddrValue_eq_evalBinary j

/-- Concrete third fallthrough instruction: `add.s64 %rd6, %rd4, %rd5`. -/
def saxpyFallthrough0_add_x_addr : GInstr :=
  { guard? := none, instr := .assignReg "rd6" saxpyXElementAddrRhs }

theorem saxpyFallthrough0_body2_isSome : (saxpyFallthrough0.body[2]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi2 : GInstr :=
  saxpyFallthrough0.body[2]?.get saxpyFallthrough0_body2_isSome

theorem saxpyFallthrough0_body2 : saxpyFallthrough0.body[2]? = some saxpyFallthrough0_gi2 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body2_isSome, rfl⟩

theorem saxpyFallthrough0_gi2_eq :
    saxpyFallthrough0_gi2 = saxpyFallthrough0_add_x_addr := by
  unfold saxpyFallthrough0_add_x_addr saxpyXElementAddrRhs
  apply GInstr.eq_unguarded_assignReg_binop_regs_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi2.guard?.isNone = true
    unfold saxpyFallthrough0_gi2 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi2 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body2_add_x_addr :
    saxpyFallthrough0.body[2]? = some saxpyFallthrough0_add_x_addr := by
  rw [saxpyFallthrough0_body2, saxpyFallthrough0_gi2_eq]

/-- Concrete fourth fallthrough instruction: `cvta.to.global.u64 %rd7, %rd2`. -/
def saxpyFallthrough0_cvta_rd7 : GInstr :=
  { guard? := none, instr := .cvta "rd7" .global (.reg "rd2") }

theorem saxpyFallthrough0_body3_isSome : (saxpyFallthrough0.body[3]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi3 : GInstr :=
  saxpyFallthrough0.body[3]?.get saxpyFallthrough0_body3_isSome

theorem saxpyFallthrough0_body3 : saxpyFallthrough0.body[3]? = some saxpyFallthrough0_gi3 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body3_isSome, rfl⟩

theorem saxpyFallthrough0_gi3_eq :
    saxpyFallthrough0_gi3 = saxpyFallthrough0_cvta_rd7 := by
  unfold saxpyFallthrough0_cvta_rd7
  apply GInstr.eq_unguarded_cvta_reg_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi3.guard?.isNone = true
    unfold saxpyFallthrough0_gi3 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi3 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi3 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi3 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi3 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body3_cvta_rd7 :
    saxpyFallthrough0.body[3]? = some saxpyFallthrough0_cvta_rd7 := by
  rw [saxpyFallthrough0_body3, saxpyFallthrough0_gi3_eq]

/-- Lowered RHS of `add.s64 %rd8, %rd7, %rd5`. -/
def saxpyYElementAddrRhs : RValue :=
  .binop .add (.reg "rd7") (.reg "rd5")

/-- Y-array element address value for lane `j`. -/
@[irreducible]
def saxpyYElementAddrValue (j : LaneId) : Value :=
  (evalBinary? .add (.gaddr .global saxpyYBase) (saxpyLaneByteOffsetValue j)).getD
    (.gaddr .global saxpyYBase)

theorem saxpyYElementAddrValue_eq_evalBinary (j : LaneId) :
    evalBinary? .add (.gaddr .global saxpyYBase) (saxpyLaneByteOffsetValue j) =
      some (saxpyYElementAddrValue j) := by
  unfold saxpyYElementAddrValue saxpyLaneByteOffsetValue
  rfl

theorem evalRValue_saxpyYElementAddrRhs_of_regs
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hRd7 : ls.regs["rd7"]? = some (.gaddr .global saxpyYBase))
    (hRd5 : ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j)) :
    evalRValue? st 0 0 j saxpyYElementAddrRhs =
      some (saxpyYElementAddrValue j) := by
  simpa [saxpyYElementAddrRhs, readReg, evalRValue?, hGet, hRd7, hRd5]
    using saxpyYElementAddrValue_eq_evalBinary j

/-- Concrete fifth fallthrough instruction: `add.s64 %rd8, %rd7, %rd5`. -/
def saxpyFallthrough0_add_y_addr : GInstr :=
  { guard? := none, instr := .assignReg "rd8" saxpyYElementAddrRhs }

theorem saxpyFallthrough0_body4_isSome : (saxpyFallthrough0.body[4]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi4 : GInstr :=
  saxpyFallthrough0.body[4]?.get saxpyFallthrough0_body4_isSome

theorem saxpyFallthrough0_body4 : saxpyFallthrough0.body[4]? = some saxpyFallthrough0_gi4 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body4_isSome, rfl⟩

theorem saxpyFallthrough0_gi4_eq :
    saxpyFallthrough0_gi4 = saxpyFallthrough0_add_y_addr := by
  unfold saxpyFallthrough0_add_y_addr saxpyYElementAddrRhs
  apply GInstr.eq_unguarded_assignReg_binop_regs_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi4.guard?.isNone = true
    unfold saxpyFallthrough0_gi4 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi4 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body4_add_y_addr :
    saxpyFallthrough0.body[4]? = some saxpyFallthrough0_add_y_addr := by
  rw [saxpyFallthrough0_body4, saxpyFallthrough0_gi4_eq]

/-- Source address for `ld.global.s32 %r7, [%rd6]`. -/
def saxpyLoadXSrc : TypedAddr :=
  { space := .global, ty := .s32, addr := .reg "rd6" }

theorem resolveAddr_saxpyLoadXSrc_of_rd6
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hRd6 : ls.regs["rd6"]? = some (saxpyXElementAddrValue j)) :
    resolveAddr? st 0 0 j saxpyLoadXSrc =
      some (.global (saxpyXBase + j.val * 4)) := by
  unfold saxpyLoadXSrc resolveAddr? evalRValue? readReg
  simp [hGet, hRd6]
  fin_cases j <;> unfold saxpyXElementAddrValue saxpyLaneByteOffsetValue <;> rfl

/-- Concrete sixth fallthrough instruction: `ld.global.s32 %r7, [%rd6]`. -/
def saxpyFallthrough0_load_x : GInstr :=
  { guard? := none, instr := .load "r7" saxpyLoadXSrc }

theorem saxpyFallthrough0_body5_isSome : (saxpyFallthrough0.body[5]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi5 : GInstr :=
  saxpyFallthrough0.body[5]?.get saxpyFallthrough0_body5_isSome

theorem saxpyFallthrough0_body5 : saxpyFallthrough0.body[5]? = some saxpyFallthrough0_gi5 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body5_isSome, rfl⟩

theorem saxpyFallthrough0_gi5_eq :
    saxpyFallthrough0_gi5 = saxpyFallthrough0_load_x := by
  unfold saxpyFallthrough0_load_x saxpyLoadXSrc
  apply GInstr.eq_unguarded_load_reg_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi5.guard?.isNone = true
    unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi5 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body5_load_x :
    saxpyFallthrough0.body[5]? = some saxpyFallthrough0_load_x := by
  rw [saxpyFallthrough0_body5, saxpyFallthrough0_gi5_eq]

/-- Source address for `ld.global.s32 %r8, [%rd8]`. -/
def saxpyLoadYSrc : TypedAddr :=
  { space := .global, ty := .s32, addr := .reg "rd8" }

theorem resolveAddr_saxpyLoadYSrc_of_rd8
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hRd8 : ls.regs["rd8"]? = some (saxpyYElementAddrValue j)) :
    resolveAddr? st 0 0 j saxpyLoadYSrc =
      some (.global (saxpyYBase + j.val * 4)) := by
  unfold saxpyLoadYSrc resolveAddr? evalRValue? readReg
  simp [hGet, hRd8]
  fin_cases j <;> unfold saxpyYElementAddrValue saxpyLaneByteOffsetValue <;> rfl

/-- Concrete seventh fallthrough instruction: `ld.global.s32 %r8, [%rd8]`. -/
def saxpyFallthrough0_load_y : GInstr :=
  { guard? := none, instr := .load "r8" saxpyLoadYSrc }

theorem saxpyFallthrough0_body6_isSome : (saxpyFallthrough0.body[6]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi6 : GInstr :=
  saxpyFallthrough0.body[6]?.get saxpyFallthrough0_body6_isSome

theorem saxpyFallthrough0_body6 : saxpyFallthrough0.body[6]? = some saxpyFallthrough0_gi6 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body6_isSome, rfl⟩

theorem saxpyFallthrough0_gi6_eq :
    saxpyFallthrough0_gi6 = saxpyFallthrough0_load_y := by
  unfold saxpyFallthrough0_load_y saxpyLoadYSrc
  apply GInstr.eq_unguarded_load_reg_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi6.guard?.isNone = true
    unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi6 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body6_load_y :
    saxpyFallthrough0.body[6]? = some saxpyFallthrough0_load_y := by
  rw [saxpyFallthrough0_body6, saxpyFallthrough0_gi6_eq]

/-- Lowered RHS of `mad.lo.s32 %r9, %r7, %r6, %r8`. -/
def saxpyMulAddRhs : RValue :=
  .triop .mad
    (.reg "r7")
    (.unop (.cvt .s32) (.reg "r6"))
    (.reg "r8")

/-- Per-lane value computed by the SAXPY multiply-add instruction. -/
@[irreducible]
def saxpyMulAddValue (alpha : Int) (xs ys : List Int) (j : LaneId) : Value :=
  (evalTernary? .mad
    (saxpyLoadedS32Value xs j)
    ((evalUnary? (.cvt .s32) (.s32 (saxpyAlphaS32 alpha))).getD (.s32 0))
    (saxpyLoadedS32Value ys j)).getD (.s32 0)

/-- Concrete eighth fallthrough instruction: `mad.lo.s32 %r9, %r7, %r6, %r8`. -/
def saxpyFallthrough0_mul_add : GInstr :=
  { guard? := none, instr := .assignReg "r9" saxpyMulAddRhs }

theorem saxpyFallthrough0_body7_isSome : (saxpyFallthrough0.body[7]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi7 : GInstr :=
  saxpyFallthrough0.body[7]?.get saxpyFallthrough0_body7_isSome

theorem saxpyFallthrough0_body7 : saxpyFallthrough0.body[7]? = some saxpyFallthrough0_gi7 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body7_isSome, rfl⟩

theorem saxpyFallthrough0_gi7_eq :
    saxpyFallthrough0_gi7 = saxpyFallthrough0_mul_add := by
  unfold saxpyFallthrough0_mul_add saxpyMulAddRhs
  apply GInstr.eq_unguarded_assignReg_madS32_reg_cvt_reg_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi7.guard?.isNone = true
    unfold saxpyFallthrough0_gi7 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi7 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body7_mul_add :
    saxpyFallthrough0.body[7]? = some saxpyFallthrough0_mul_add := by
  rw [saxpyFallthrough0_body7, saxpyFallthrough0_gi7_eq]

/-- Concrete ninth fallthrough instruction: `cvta.to.global.u64 %rd9, %rd3`. -/
def saxpyFallthrough0_cvta_rd9 : GInstr :=
  { guard? := none, instr := .cvta "rd9" .global (.reg "rd3") }

theorem saxpyFallthrough0_body8_isSome : (saxpyFallthrough0.body[8]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi8 : GInstr :=
  saxpyFallthrough0.body[8]?.get saxpyFallthrough0_body8_isSome

theorem saxpyFallthrough0_body8 : saxpyFallthrough0.body[8]? = some saxpyFallthrough0_gi8 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body8_isSome, rfl⟩

theorem saxpyFallthrough0_gi8_eq :
    saxpyFallthrough0_gi8 = saxpyFallthrough0_cvta_rd9 := by
  unfold saxpyFallthrough0_cvta_rd9
  apply GInstr.eq_unguarded_cvta_reg_of_projections
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi8.guard?.isNone = true
    unfold saxpyFallthrough0_gi8 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi8 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi8 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi8 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi8 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body8_cvta_rd9 :
    saxpyFallthrough0.body[8]? = some saxpyFallthrough0_cvta_rd9 := by
  rw [saxpyFallthrough0_body8, saxpyFallthrough0_gi8_eq]

/-- Lowered RHS of `add.s64 %rd10, %rd9, %rd5`. -/
def saxpyResultElementAddrRhs : RValue :=
  .binop .add (.reg "rd9") (.reg "rd5")

/-- Result-array element address value for lane `j`. -/
@[irreducible]
def saxpyResultElementAddrValue (j : LaneId) : Value :=
  (evalBinary? .add (.gaddr .global saxpyRBase) (saxpyLaneByteOffsetValue j)).getD
    (.gaddr .global saxpyRBase)

theorem saxpyResultElementAddrValue_eq_evalBinary (j : LaneId) :
    evalBinary? .add (.gaddr .global saxpyRBase) (saxpyLaneByteOffsetValue j) =
      some (saxpyResultElementAddrValue j) := by
  unfold saxpyResultElementAddrValue saxpyLaneByteOffsetValue
  rfl

theorem evalRValue_saxpyResultElementAddrRhs_of_regs
    {st : State} {j : LaneId} {ls : LaneState}
    (hGet : st.getLane? 0 0 j = some ls)
    (hRd9 : ls.regs["rd9"]? = some (.gaddr .global saxpyRBase))
    (hRd5 : ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j)) :
    evalRValue? st 0 0 j saxpyResultElementAddrRhs =
      some (saxpyResultElementAddrValue j) := by
  simpa [saxpyResultElementAddrRhs, readReg, evalRValue?, hGet, hRd9, hRd5]
    using saxpyResultElementAddrValue_eq_evalBinary j

/-- Concrete tenth fallthrough instruction: `add.s64 %rd10, %rd9, %rd5`. -/
def saxpyFallthrough0_add_result_addr : GInstr :=
  { guard? := none, instr := .assignReg "rd10" saxpyResultElementAddrRhs }

theorem saxpyFallthrough0_body9_isSome : (saxpyFallthrough0.body[9]?).isSome = true := by
  unfold saxpyFallthrough0
  native_decide

def saxpyFallthrough0_gi9 : GInstr :=
  saxpyFallthrough0.body[9]?.get saxpyFallthrough0_body9_isSome

theorem saxpyFallthrough0_body9 : saxpyFallthrough0.body[9]? = some saxpyFallthrough0_gi9 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyFallthrough0_body9_isSome, rfl⟩

theorem saxpyFallthrough0_gi9_eq :
    saxpyFallthrough0_gi9 = saxpyFallthrough0_add_result_addr := by
  unfold saxpyFallthrough0_add_result_addr saxpyResultElementAddrRhs
  apply GInstr.eq_unguarded_assignReg_binop_regs_of_projection
  · rw [← Option.isNone_iff_eq_none]
    show saxpyFallthrough0_gi9.guard?.isNone = true
    unfold saxpyFallthrough0_gi9 saxpyFallthrough0
    native_decide
  · unfold saxpyFallthrough0_gi9 saxpyFallthrough0
    native_decide

theorem saxpyFallthrough0_body9_add_result_addr :
    saxpyFallthrough0.body[9]? = some saxpyFallthrough0_add_result_addr := by
  rw [saxpyFallthrough0_body9, saxpyFallthrough0_gi9_eq]

/-! ## Reusable step records

The chain proofs should expose a small, uniform step artifact instead of
repeating the `step?`/`stepInstr?` plumbing at every body slot. The context
packages the structural hypotheses for the current body instruction; the
record packages the successor state, the instruction-step equality, the
top-level `step?` equality, and the per-step postcondition. -/

/-- A top-level machine step plus a postcondition on its successor. -/
structure StepRecord (pre : State) (Post : State → Prop) where
  post : State
  step : StepMachine.step? pre = some post
  post_holds : Post post

namespace StepRecord

theorem to_exists {pre : State} {Post : State → Prop} (r : StepRecord pre Post) :
    ∃ post, StepMachine.step? pre = some post ∧ Post post :=
  ⟨r.post, r.step, r.post_holds⟩

theorem runN_succ {pre : State} {Post : State → Prop} (r : StepRecord pre Post)
    (fuel : Nat) :
    StepMachine.runN (fuel + 1) pre = StepMachine.runN fuel r.post :=
  runN_succ_some r.step

end StepRecord

/-- Reading a register other than the inserted destination is preserved. -/
theorem reg_insert_getElem?_ne
    {regs : Std.HashMap RegName Value} {dst r : RegName} {new old : Value}
    (hOld : regs[r]? = some old)
    (hNe : r ≠ dst) :
    (regs.insert dst new)[r]? = some old := by
  rw [Std.HashMap.getElem?_insert]
  by_cases hEqKey : dst = r
  · exact False.elim (hNe hEqKey.symm)
  · simp [beq_iff_eq, hEqKey, hOld]

/-- Structural context for one body instruction at the current runnable PC. -/
structure BodyStepContext (st : State) (pc : PC) (block : Block) (gi : GInstr)
    (participants : List LaneId) where
  warpState : WarpState
  wf : State.wf st
  getWarp : st.getWarp? 0 0 = some warpState
  warp_wf : WarpState.wf warpState
  lockstep : lockstepRunnable warpState
  currentPc : currentRunnablePc? warpState = some pc
  block_lookup : st.kernelEnv.blocks[pc.1]? = some block
  body_slot : block.body[pc.2]? = some gi
  participants_eq : participatingRunnableLaneIds? warpState gi.guard? = some participants

namespace BodyStepContext

theorem participants_nodup
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants) :
    participants.Nodup :=
  participatingRunnableLaneIds?_nodup ctx.participants_eq

theorem step_of_instr
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants)
    {post : State}
    (hInstr : stepInstr? st 0 0 gi = some post) :
    StepMachine.step? st = some post :=
  step?_body_some ctx.wf ctx.getWarp ctx.warp_wf ctx.lockstep ctx.currentPc
    ctx.block_lookup ctx.body_slot ctx.participants_eq hInstr

theorem lane_pre
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants)
    {lane : LaneId}
    (hLane : lane ∈ participants) :
    ∃ laneState : LaneState,
      st.getLane? 0 0 lane = some laneState ∧ laneState.pc = pc := by
  obtain ⟨_, hLaneState⟩ :=
    participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane
  obtain ⟨laneState, hWarpLane, hLanePc⟩ := hLaneState
  refine ⟨laneState, ?_, hLanePc⟩
  unfold State.getLane?
  rw [ctx.getWarp]
  simp [hWarpLane]

theorem lane_status_running
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants)
    {lane : LaneId} {laneState : LaneState}
    (hLane : lane ∈ participants)
    (hGet : st.getLane? 0 0 lane = some laneState) :
    laneState.status = .running := by
  have hRun := (participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane).1
  unfold runnableLaneIds at hRun
  rw [List.mem_filter] at hRun
  have hRunnable := hRun.2
  have hWarpLane : ctx.warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hGet
    rw [ctx.getWarp] at hGet
    simpa using hGet
  unfold laneIsRunnable at hRunnable
  rw [hWarpLane] at hRunnable
  rw [Bool.and_eq_true] at hRunnable
  simpa using hRunnable.1

/-- Loop fact used to show that an unguarded participant selection returns all
runnable lanes when every runnable lane is at the current PC. -/
private theorem participatingRunnableLaneIds?_none_loop (ws : WarpState) (pc : PC) :
    ∀ (lanes acc : List LaneId),
      (∀ lane ∈ lanes, ∃ ls, ws.getLane? lane = some ls ∧ ls.pc = pc) →
      (forIn lanes acc fun lane out =>
          match ws.getLane? lane with
          | some laneState =>
            if laneState.pc = pc then
              (guardHolds? laneState none).bind fun passes =>
                if passes = true then some (ForInStep.yield (lane :: out))
                else some (ForInStep.yield out)
            else some (ForInStep.yield out)
          | _ => none) = some (lanes.reverse ++ acc) := by
  intro lanes
  induction lanes with
  | nil => intro acc _; simp
  | cons lane rest ih =>
      intro acc h
      obtain ⟨ls, hLs, hPcLs⟩ := h lane List.mem_cons_self
      simp [List.forIn_cons, hLs, hPcLs, guardHolds?]
      have hRest : ∀ x ∈ rest, ∃ ls, ws.getLane? x = some ls ∧ ls.pc = pc := by
        intro x hx; exact h x (List.mem_cons_of_mem lane hx)
      simpa [List.reverse_cons, List.append_assoc] using ih (lane :: acc) hRest

/-- In a lockstep warp with a current runnable PC, unguarded participation is
exactly the runnable-lane list. -/
theorem participatingRunnableLaneIds?_none_of_lockstep
    {ws : WarpState} {pc : PC}
    (hLock : lockstepRunnable ws)
    (hPc : currentRunnablePc? ws = some pc) :
    participatingRunnableLaneIds? ws none = some (runnableLaneIds ws) := by
  have hLanePc : ∀ lane ∈ runnableLaneIds ws,
      ∃ ls, ws.getLane? lane = some ls ∧ ls.pc = pc := by
    intro lane hLane
    unfold lockstepRunnable lockstepRunnable? at hLock
    rw [hPc] at hLock
    simp [List.all_eq_true] at hLock
    have h := hLock lane hLane
    cases hWsLane : ws.getLane? lane with
    | none => rw [hWsLane] at h; simp at h
    | some ls =>
        refine ⟨ls, rfl, ?_⟩
        rw [hWsLane] at h
        simpa using h
  have hLoop :
      (forIn (runnableLaneIds ws) ([] : List LaneId) fun lane out =>
          match ws.getLane? lane with
          | some laneState =>
            if laneState.pc = pc then
              (guardHolds? laneState none).bind fun passes =>
                if passes = true then some (ForInStep.yield (lane :: out))
                else some (ForInStep.yield out)
            else some (ForInStep.yield out)
          | _ => none) = some (runnableLaneIds ws).reverse := by
    simpa using participatingRunnableLaneIds?_none_loop ws pc (runnableLaneIds ws) [] hLanePc
  unfold participatingRunnableLaneIds?
  simp [hPc]
  exact calc
    ((forIn (runnableLaneIds ws) ([] : List LaneId) fun lane out =>
          match ws.getLane? lane with
          | some laneState =>
            if laneState.pc = pc then
              (guardHolds? laneState none).bind fun passes =>
                if passes = true then some (ForInStep.yield (lane :: out))
                else some (ForInStep.yield out)
            else some (ForInStep.yield out)
          | _ => none).bind fun out => some out.reverse)
        = ((some (runnableLaneIds ws).reverse).bind fun out => some out.reverse) := by
          exact congrArg (fun x => x.bind fun out => some out.reverse) hLoop
    _ = some (runnableLaneIds ws) := by simp

theorem runnable_eq_participants_of_none
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants)
    (hGuard : gi.guard? = none) :
    runnableLaneIds ctx.warpState = participants := by
  have hPrevNone :
      participatingRunnableLaneIds? ctx.warpState none = some participants := by
    rw [← hGuard]
    exact ctx.participants_eq
  have hPrevRunnable :=
    participatingRunnableLaneIds?_none_of_lockstep ctx.lockstep ctx.currentPc
  rw [hPrevRunnable] at hPrevNone
  exact Option.some.inj hPrevNone

end BodyStepContext

/-- A body instruction step plus a postcondition on its successor. -/
structure BodyStepRecord
    {st : State} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (ctx : BodyStepContext st pc block gi participants) (Post : State → Prop) where
  post : State
  instrStep : stepInstr? st 0 0 gi = some post
  step : StepMachine.step? st = some post
  post_wf : State.wf post
  post_global : post.global = st.global
  post_const : post.const = st.const
  post_param : post.param = st.param
  post_kernelEnv : post.kernelEnv = st.kernelEnv
  post_atomics : post.atomics = st.atomics
  post_warpState : WarpState
  post_getWarp : post.getWarp? 0 0 = some post_warpState
  post_warp_wf : WarpState.wf post_warpState
  post_lockstep : lockstepRunnable post_warpState
  post_runnable : runnableLaneIds post_warpState = runnableLaneIds ctx.warpState
  post_currentPc : currentRunnablePc? post_warpState = some (pc.1, pc.2 + 1)
  post_holds : Post post

namespace BodyStepRecord

theorem to_exists
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    {ctx : BodyStepContext st pc block gi participants} {Post : State → Prop}
    (r : BodyStepRecord ctx Post) :
    ∃ post, StepMachine.step? st = some post ∧ Post post :=
  ⟨r.post, r.step, r.post_holds⟩

def toStepRecord
    {st : State} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId}
    {ctx : BodyStepContext st pc block gi participants} {Post : State → Prop}
    (r : BodyStepRecord ctx Post) :
    StepRecord st Post :=
  ⟨r.post, r.step, r.post_holds⟩

/-- Build the next unguarded body-step context from a completed unguarded
body step and a proof of the next body slot. -/
def nextContextNone
    {st : State} {pc : PC} {block : Block} {gi nextGi : GInstr}
    {participants : List LaneId} {Post : State → Prop}
    {ctx : BodyStepContext st pc block gi participants}
    (r : BodyStepRecord ctx Post)
    (hPrevGuard : gi.guard? = none)
    (hNextGuard : nextGi.guard? = none)
    (hNextBody : block.body[pc.2 + 1]? = some nextGi) :
    BodyStepContext r.post (pc.1, pc.2 + 1) block nextGi participants :=
  { warpState := r.post_warpState
    wf := r.post_wf
    getWarp := r.post_getWarp
    warp_wf := r.post_warp_wf
    lockstep := r.post_lockstep
    currentPc := r.post_currentPc
    block_lookup := by
      rw [r.post_kernelEnv]
      exact ctx.block_lookup
    body_slot := hNextBody
    participants_eq := by
      have hPrevNone :
          participatingRunnableLaneIds? ctx.warpState none = some participants := by
        rw [← hPrevGuard]
        exact ctx.participants_eq
      have hPrevRunnable :=
        BodyStepContext.participatingRunnableLaneIds?_none_of_lockstep
          ctx.lockstep ctx.currentPc
      have hPartEq : runnableLaneIds ctx.warpState = participants := by
        rw [hPrevRunnable] at hPrevNone
        exact Option.some.inj hPrevNone
      have hPostNone :=
        BodyStepContext.participatingRunnableLaneIds?_none_of_lockstep
          r.post_lockstep r.post_currentPc
      rw [hNextGuard]
      rw [r.post_runnable, hPartEq] at hPostNone
      exact hPostNone }

/-- Full per-lane effect of an `assignReg` body record, recovered from the
record's instruction-step equality. -/
theorem assignReg_lane_full
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : RegName} {rhs : RValue} {guard? : Option Guard}
    {ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .assignReg dst rhs } participants}
    {Post : State → Prop}
    (r : BodyStepRecord ctx Post)
    {lane : LaneId} {laneState : LaneState} {val : Value}
    (hLane : lane ∈ participants)
    (hGet : st.getLane? 0 0 lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hEval : evalRValue? st 0 0 lane rhs = some val) :
    ∃ laneState' : LaneState,
      r.post.getLane? 0 0 lane = some laneState' ∧
      laneState'.regs = laneState.regs.insert dst val ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) :=
  stepInstr?_assignReg_lane_full
    (dst := dst) (rhs := rhs) (guard? := guard?) ctx.wf ctx.getWarp ctx.lockstep
    ctx.currentPc ctx.participants_eq hLane hGet hLanePc hEval r.instrStep

end BodyStepRecord

/-- The runnable-lane subset used by `stepTerminator?` at the current PC. -/
def termParticipantsFor (ws : WarpState) (pc : PC) : List LaneId :=
  (runnableLaneIds ws).filter fun lane =>
    match ws.getLane? lane with
    | some laneState => laneState.pc == pc
    | none => false

theorem termParticipantsFor_nodup (ws : WarpState) (pc : PC) :
    (termParticipantsFor ws pc).Nodup := by
  have hLaneIdsNoDup : (laneIds : List LaneId).Nodup := by
    unfold laneIds
    exact List.nodup_finRange 32
  have hRunNoDup : (runnableLaneIds ws).Nodup :=
    List.Nodup.filter _ hLaneIdsNoDup
  unfold termParticipantsFor
  exact List.Nodup.filter _ hRunNoDup

theorem termParticipantsFor_eq_runnable_of_lockstep
    {ws : WarpState} {pc : PC}
    (hLock : lockstepRunnable ws)
    (hPc : currentRunnablePc? ws = some pc) :
    termParticipantsFor ws pc = runnableLaneIds ws := by
  unfold termParticipantsFor
  apply List.filter_eq_self.mpr
  intro lane hLane
  unfold lockstepRunnable lockstepRunnable? at hLock
  rw [hPc] at hLock
  simp [List.all_eq_true] at hLock
  have h := hLock lane hLane
  cases hWsLane : ws.getLane? lane with
  | none =>
      rw [hWsLane] at h
      simp at h
  | some ls =>
      rw [hWsLane] at h
      simp at h
      simp [h]

/-- The internal branch-destination loop from `uniformBranchDestination?`, exposed
so uniform false-branch proofs do not have to reason about the whole terminator. -/
def branchDests? (st : State) (cta : CTAId) (warp : WarpId)
    (lanes : List LaneId) (cond : RValue) (tLabel fLabel : BlockLabel)
    (acc : List PC) : Option (List PC) :=
  forIn lanes acc fun lane dests =>
    (evalRValue? st cta warp lane cond).bind fun v =>
      (valueToBool? v).bind fun b =>
        some (ForInStep.yield ((if b then (tLabel, 0) else (fLabel, 0)) :: dests))

private theorem branchDests?_false_loop
    {st : State} {cta : CTAId} {warp : WarpId} {cond : RValue}
    {tLabel fLabel : BlockLabel} :
    ∀ (lanes : List LaneId) (acc : List PC),
      (∀ pc ∈ acc, pc = (fLabel, 0)) →
      (∀ lane ∈ lanes, evalRValue? st cta warp lane cond = some (.pred false)) →
      ∃ out : List PC,
        branchDests? st cta warp lanes cond tLabel fLabel acc = some out ∧
        (∀ pc ∈ out, pc = (fLabel, 0)) ∧
        (acc ≠ [] ∨ lanes ≠ [] → out ≠ []) := by
  intro lanes
  induction lanes with
  | nil =>
      intro acc hAcc _
      refine ⟨acc, by simp [branchDests?], hAcc, ?_⟩
      intro h
      rcases h with hAccNe | hFalse
      · exact hAccNe
      · contradiction
  | cons lane rest ih =>
      intro acc hAcc hEval
      have hLane := hEval lane List.mem_cons_self
      have hRest : ∀ lane ∈ rest, evalRValue? st cta warp lane cond = some (.pred false) := by
        intro l hl
        exact hEval l (List.mem_cons_of_mem lane hl)
      have hAcc' : ∀ pc ∈ ((fLabel, 0) :: acc), pc = (fLabel, 0) := by
        intro pc hpc
        rw [List.mem_cons] at hpc
        rcases hpc with hhead | htail
        · exact hhead
        · exact hAcc pc htail
      obtain ⟨out, hLoop, hOut, hNonempty⟩ := ih ((fLabel, 0) :: acc) hAcc' hRest
      refine ⟨out, ?_, hOut, ?_⟩
      · simp [branchDests?, hLane]
        exact hLoop
      · intro _
        exact hNonempty (Or.inl (by simp))

theorem uniformBranchDestination?_false
    {st : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {cond : RValue} {tLabel fLabel : BlockLabel}
    (hNonempty : lanes ≠ [])
    (hEval : ∀ lane ∈ lanes, evalRValue? st cta warp lane cond = some (.pred false)) :
    uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some (fLabel, 0) := by
  unfold uniformBranchDestination?
  change ((branchDests? st cta warp lanes cond tLabel fLabel []).bind fun dests =>
    match dests.reverse with
    | [] => none
    | dest :: rest => if rest.all (fun pc' => pc' == dest) then some dest else none)
    = some (fLabel, 0)
  obtain ⟨out, hLoop, hOut, hOutNonempty⟩ :=
    branchDests?_false_loop (st := st) (cta := cta) (warp := warp)
      (cond := cond) (tLabel := tLabel) (fLabel := fLabel) lanes [] (by simp) hEval
  rw [hLoop]
  simp
  have hRevNonempty : out.reverse ≠ [] := by
    intro h
    apply hOutNonempty (Or.inr hNonempty)
    have := congrArg List.reverse h
    simpa using this
  cases hRev : out.reverse with
  | nil =>
      exact False.elim (hRevNonempty hRev)
  | cons dest rest =>
      have hDest : dest = (fLabel, 0) := by
        have : dest ∈ out := by
          rw [← List.mem_reverse]
          rw [hRev]
          exact List.mem_cons_self
        exact hOut dest this
      have hRestAll : (rest.all fun pc' => pc' == dest) = true := by
        rw [List.all_eq_true]
        intro pc hpc
        have hpcOut : pc ∈ out := by
          rw [← List.mem_reverse]
          rw [hRev]
          exact List.mem_cons_of_mem dest hpc
        rw [hOut pc hpcOut, hDest]
        simp
      change (if (rest.all fun pc' => pc' == dest) = true then some dest else none) =
        some (fLabel, 0)
      rw [hRestAll, hDest]
      simp

/-- Structural context for a terminator step at the current runnable PC. -/
structure TermStepContext (st : State) (pc : PC) (block : Block) (term : Terminator)
    (participants : List LaneId) where
  warpState : WarpState
  wf : State.wf st
  getWarp : st.getWarp? 0 0 = some warpState
  warp_wf : WarpState.wf warpState
  lockstep : lockstepRunnable warpState
  currentPc : currentRunnablePc? warpState = some pc
  block_lookup : st.kernelEnv.blocks[pc.1]? = some block
  body_done : block.body[pc.2]? = none
  block_term : block.term = term
  participants_eq : termParticipantsFor warpState pc = participants

namespace TermStepContext

theorem participants_nodup
    {st : State} {pc : PC} {block : Block} {term : Terminator} {participants : List LaneId}
    (ctx : TermStepContext st pc block term participants) :
    participants.Nodup := by
  rw [← ctx.participants_eq]
  exact termParticipantsFor_nodup ctx.warpState pc

theorem step_of_term
    {st : State} {pc : PC} {block : Block} {term : Terminator} {participants : List LaneId}
    (ctx : TermStepContext st pc block term participants)
    {post : State}
    (hTerm : stepTerminator? st 0 0 term = some post) :
    StepMachine.step? st = some post := by
  calc
    StepMachine.step? st = stepTerminator? st 0 0 block.term :=
      step?_term ctx.wf ctx.getWarp ctx.warp_wf ctx.lockstep ctx.currentPc
        ctx.block_lookup ctx.body_done
    _ = stepTerminator? st 0 0 term := by rw [ctx.block_term]
    _ = some post := hTerm

theorem lane_pre
    {st : State} {pc : PC} {block : Block} {term : Terminator} {participants : List LaneId}
    (ctx : TermStepContext st pc block term participants)
    {lane : LaneId}
    (hLane : lane ∈ participants) :
    ∃ laneState : LaneState,
      st.getLane? 0 0 lane = some laneState ∧
      laneState.pc = pc ∧
      lane ∈ runnableLaneIds ctx.warpState := by
  have hMem : lane ∈ termParticipantsFor ctx.warpState pc := by
    rw [ctx.participants_eq]
    exact hLane
  unfold termParticipantsFor at hMem
  rw [List.mem_filter] at hMem
  obtain ⟨hRun, hPcFilter⟩ := hMem
  cases hWarpLane : ctx.warpState.getLane? lane with
  | none =>
      rw [hWarpLane] at hPcFilter
      simp at hPcFilter
  | some laneState =>
      refine ⟨laneState, ?_, ?_, hRun⟩
      · unfold State.getLane?
        rw [ctx.getWarp]
        simp [hWarpLane]
      · rw [hWarpLane] at hPcFilter
        simpa using hPcFilter

end TermStepContext

/-- A function on lane states preserves status. Terminator branch updates need this
weaker shape because branches intentionally rewrite lane PCs. -/
def PreservesStatusOnly (f : LaneId → LaneState → Option LaneState) : Prop :=
  ∀ l ls ls', f l ls = some ls' → ls'.status = ls.status

theorem applyToLaneIds?_preserves_status_lane_only
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hf : PreservesStatusOnly f)
    (hWf : State.wf st)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    ∀ (other : LaneId) (ls : LaneState),
      st.getLane? cta warp other = some ls →
      ∃ ls' : LaneState,
        st'.getLane? cta warp other = some ls' ∧
        ls'.status = ls.status := by
  induction lanes generalizing st with
  | nil =>
      intro other ls hLane
      rw [applyToLaneIds?_nil] at h
      cases h
      exact ⟨ls, hLane, rfl⟩
  | cons l rest ih =>
      intro other ls hLane
      rw [applyToLaneIds?_cons] at h
      cases hL : st.getLane? cta warp l with
      | none => rw [hL] at h; simp at h
      | some lsL =>
          rw [hL] at h; simp at h
          cases hF : f l lsL with
          | none => rw [hF] at h; simp at h
          | some lsL' =>
              rw [hF] at h; simp at h
              cases hSet : st.setLane cta warp l lsL' with
              | none => rw [hSet] at h; simp at h
              | some stMid =>
                  rw [hSet] at h; simp at h
                  have hWfMid : State.wf stMid := State.wf_of_setLane hWf hSet
                  have hWarpExists : ∃ ws, st.getWarp? cta warp = some ws := by
                    unfold State.setLane at hSet
                    cases hW : st.getWarp? cta warp with
                    | none => rw [hW] at hSet; simp at hSet
                    | some ws => exact ⟨ws, rfl⟩
                  obtain ⟨wsExt, hWsExt⟩ := hWarpExists
                  have hWfW : WarpState.wf wsExt := WarpState.wf_of_getWarp? hWf hWsExt
                  by_cases hEq : other = l
                  · have hStMidGet : stMid.getLane? cta warp l = some lsL' :=
                      setLane_get_self hSet hWsExt hWfW
                    rw [hEq]
                    have hRec := ih hWfMid h l lsL' hStMidGet
                    obtain ⟨ls', hGet', hStatus⟩ := hRec
                    refine ⟨ls', hGet', ?_⟩
                    have hLsEq : ls = lsL := by
                      rw [hEq] at hLane
                      rw [hLane] at hL
                      exact Option.some.inj hL
                    subst hLsEq
                    rw [hStatus]
                    exact hf l ls lsL' hF
                  · have hMidLane : stMid.getLane? cta warp other = some ls := by
                      rw [setLane_preserves_other_lane hSet other hEq]
                      exact hLane
                    exact ih hWfMid h other ls hMidLane

private theorem applyToLaneIds?_preserves_status_warp_lane_only
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    {ws ws' : WarpState}
    (hf : PreservesStatusOnly f)
    (hWf : State.wf st)
    (hWs : st.getWarp? cta warp = some ws)
    (hWs' : st'.getWarp? cta warp = some ws')
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    ∀ (lane : LaneId) (ls : LaneState),
      ws.getLane? lane = some ls →
      ∃ ls' : LaneState,
        ws'.getLane? lane = some ls' ∧
        ls'.status = ls.status := by
  intro lane ls hLs
  have hStLane : st.getLane? cta warp lane = some ls := by
    unfold State.getLane?
    rw [hWs]
    simpa using hLs
  obtain ⟨ls', hSt'Lane, hStatus⟩ :=
    applyToLaneIds?_preserves_status_lane_only hf hWf h lane ls hStLane
  have hWsLane' : ws'.getLane? lane = some ls' := by
    unfold State.getLane? at hSt'Lane
    rw [hWs'] at hSt'Lane
    simpa using hSt'Lane
  exact ⟨ls', hWsLane', hStatus⟩

private theorem warpState_getLane?_isSome_of_wf
    {ws : WarpState} (hWf : WarpState.wf ws) (lane : LaneId) :
    ∃ ls : LaneState, ws.getLane? lane = some ls := by
  have hSize : ws.lanes.size = 32 := by
    simpa [WarpState.wf, WarpState.wf?] using hWf
  have hLt : lane.val < ws.lanes.size := by simp [hSize]
  unfold WarpState.getLane?
  exact ⟨ws.lanes[lane.val], Array.getElem?_eq_getElem hLt⟩

theorem applyToLaneIds?_preserves_runnableLaneIds_of_status
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    {ws ws' : WarpState}
    (hf : PreservesStatusOnly f)
    (hWf : State.wf st)
    (hWs : st.getWarp? cta warp = some ws)
    (hWs' : st'.getWarp? cta warp = some ws')
    (hWsWf : WarpState.wf ws)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    runnableLaneIds ws' = runnableLaneIds ws := by
  obtain ⟨wsExt, hWsExt, hMask⟩ :=
    applyToLaneIds?_preserves_activeMask h ws hWs
  have hwsEq : ws' = wsExt := by
    rw [hWs'] at hWsExt
    exact Option.some.inj hWsExt
  have hLaneEq : ∀ lane : LaneId, laneIsRunnable ws' lane = laneIsRunnable ws lane := by
    intro lane
    obtain ⟨ls, hLs⟩ := warpState_getLane?_isSome_of_wf hWsWf lane
    obtain ⟨ls', hLs', hStatus⟩ :=
      applyToLaneIds?_preserves_status_warp_lane_only hf hWf hWs hWs' h lane ls hLs
    unfold laneIsRunnable
    rw [hLs, hLs']
    simp [hStatus, hwsEq, hMask]
  unfold runnableLaneIds
  exact List.filter_congr (fun lane _ => by rw [hLaneEq])

/-- A terminator step plus a postcondition on its successor. `nextPc?` is
`some target` for branch steps and `none` for a terminating return. -/
structure TermStepRecord
    {st : State} {pc : PC} {block : Block} {term : Terminator}
    {participants : List LaneId}
    (ctx : TermStepContext st pc block term participants) (nextPc? : Option PC)
    (Post : State → Prop) where
  post : State
  termStep : stepTerminator? st 0 0 term = some post
  step : StepMachine.step? st = some post
  post_wf : State.wf post
  post_global : post.global = st.global
  post_const : post.const = st.const
  post_param : post.param = st.param
  post_kernelEnv : post.kernelEnv = st.kernelEnv
  post_atomics : post.atomics = st.atomics
  post_warpState : WarpState
  post_getWarp : post.getWarp? 0 0 = some post_warpState
  post_warp_wf : WarpState.wf post_warpState
  post_lockstep : lockstepRunnable post_warpState
  post_runnable : runnableLaneIds post_warpState = runnableLaneIds ctx.warpState
  post_currentPc : currentRunnablePc? post_warpState = nextPc?
  post_holds : Post post

namespace TermStepRecord

def nextBodyContextNone
    {st : State} {pc nextPc : PC} {block nextBlock : Block} {term : Terminator}
    {nextGi : GInstr} {participants : List LaneId} {Post : State → Prop}
    {ctx : TermStepContext st pc block term participants}
    (r : TermStepRecord ctx (some nextPc) Post)
    (hNextGuard : nextGi.guard? = none)
    (hNextBlock : r.post.kernelEnv.blocks[nextPc.1]? = some nextBlock)
    (hNextBody : nextBlock.body[nextPc.2]? = some nextGi) :
    BodyStepContext r.post nextPc nextBlock nextGi participants :=
  { warpState := r.post_warpState
    wf := r.post_wf
    getWarp := r.post_getWarp
    warp_wf := r.post_warp_wf
    lockstep := r.post_lockstep
    currentPc := r.post_currentPc
    block_lookup := hNextBlock
    body_slot := hNextBody
    participants_eq := by
      have hCtxTerm :=
        termParticipantsFor_eq_runnable_of_lockstep ctx.lockstep ctx.currentPc
      have hCtxRun : runnableLaneIds ctx.warpState = participants := by
        rw [← hCtxTerm]
        exact ctx.participants_eq
      have hPostNone :=
        BodyStepContext.participatingRunnableLaneIds?_none_of_lockstep
          r.post_lockstep r.post_currentPc
      rw [hNextGuard]
      rw [r.post_runnable, hCtxRun] at hPostNone
      exact hPostNone }

end TermStepRecord

/-- Full postcondition for a uniform `cbr`: each participating lane keeps its
local state except for the branch target PC. -/
def CbrStepFullPost (pre : State) (participants : List LaneId) (dest : PC)
    (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ preLane postLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs = preLane.regs ∧
      postLane.preds = preLane.preds ∧
      postLane.localMem = preLane.localMem ∧
      postLane.status = preLane.status ∧
      postLane.pc = dest

namespace TermStepContext

/-- Construct a uniform false-branch terminator-step record. -/
noncomputable def cbrFalseStep
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {cond : RValue} {tLabel fLabel : BlockLabel}
    (ctx : TermStepContext st pc block (.cbr cond tLabel fLabel) participants)
    (hNonempty : participants ≠ [])
    (hEval : ∀ lane ∈ participants,
      evalRValue? st 0 0 lane cond = some (.pred false)) :
    TermStepRecord ctx (some (fLabel, 0)) (CbrStepFullPost st participants (fLabel, 0)) := by
  classical
  let dest : PC := (fLabel, 0)
  set fBranch : LaneId → LaneState → Option LaneState := fun _ laneState =>
    some { laneState with pc := dest } with hfBranch
  have hPartLaneSome : ∀ lane ∈ participants,
      ∃ ls, st.getLane? 0 0 lane = some ls ∧ (fBranch lane ls).isSome = true := by
    intro lane hLane
    obtain ⟨ls, hGet, _, _⟩ := lane_pre ctx hLane
    refine ⟨ls, hGet, ?_⟩
    simp [hfBranch]
  have hApplyIsSome :
      (applyToLaneIds? st 0 0 participants fBranch).isSome = true :=
    applyToLaneIds?_isSome_of_each_some (participants_nodup ctx) st ctx.wf hPartLaneSome
  have hApplyExists := Option.isSome_iff_exists.mp hApplyIsSome
  let post := Classical.choose hApplyExists
  have hApply : applyToLaneIds? st 0 0 participants fBranch = some post :=
    Classical.choose_spec hApplyExists
  have hDest : uniformBranchDestination? st 0 0 participants cond tLabel fLabel = some dest :=
    uniformBranchDestination?_false hNonempty hEval
  have hTerm :
      stepTerminator? st 0 0 (.cbr cond tLabel fLabel) = some post := by
    unfold stepTerminator?
    rw [ctx.getWarp]
    show ((some ctx.warpState).bind _) = some post
    rw [Option.some_bind]
    have hLockB := (lockstepRunnable_iff_bool ctx.warpState).1 ctx.lockstep
    simp [hLockB, ctx.currentPc]
    change ((uniformBranchDestination? st 0 0 (termParticipantsFor ctx.warpState pc) cond
          tLabel fLabel).bind
        fun dest =>
          applyToLaneIds? st 0 0 (termParticipantsFor ctx.warpState pc)
            (fun _ laneState => some { laneState with pc := dest })) = some post
    rw [ctx.participants_eq, hDest]
    change applyToLaneIds? st 0 0 participants
        (fun _ laneState => some { laneState with pc := dest }) = some post
    rw [← hfBranch]
    exact hApply
  have hPostWf : State.wf post := applyToLaneIds?_preserves_wf ctx.wf hApply
  have hTopApply := applyToLaneIds?_preserves_top hApply
  have hWsPostExists := applyToLaneIds?_preserves_activeMask hApply ctx.warpState ctx.getWarp
  let wsPost := Classical.choose hWsPostExists
  have hPostWarp : post.getWarp? 0 0 = some wsPost :=
    (Classical.choose_spec hWsPostExists).1
  have hPostWarpWf : WarpState.wf wsPost := WarpState.wf_of_getWarp? hPostWf hPostWarp
  have hStatusF : PreservesStatusOnly fBranch := by
    intro lane laneState laneState' hF
    simp [hfBranch] at hF
    subst laneState'
    rfl
  have hPostRunnable : runnableLaneIds wsPost = runnableLaneIds ctx.warpState :=
    applyToLaneIds?_preserves_runnableLaneIds_of_status hStatusF ctx.wf ctx.getWarp
      hPostWarp ctx.warp_wf hApply
  have hCtxTerm := termParticipantsFor_eq_runnable_of_lockstep ctx.lockstep ctx.currentPc
  have hCtxRun : runnableLaneIds ctx.warpState = participants := by
    rw [← hCtxTerm]
    exact ctx.participants_eq
  have hPostRunParticipants : runnableLaneIds wsPost = participants :=
    hPostRunnable.trans hCtxRun
  have hPostPc : currentRunnablePc? wsPost = some dest := by
    unfold currentRunnablePc?
    rw [hPostRunParticipants]
    cases hParts : participants with
    | nil => exact False.elim (hNonempty hParts)
    | cons head tail =>
        have hHead : head ∈ participants := by
          rw [hParts]
          exact List.mem_cons_self
        obtain ⟨preLane, hPreLane, _hPrePc, _hRun⟩ := lane_pre ctx hHead
        have hMid := applyToLaneIds?_lane_in 0 0 fBranch participants
          (participants_nodup ctx) st post ctx.wf hApply head hHead preLane hPreLane
        obtain ⟨postLane, hF, hPostLane⟩ := hMid
        simp [hfBranch] at hF
        subst postLane
        have hWsPostLane : wsPost.getLane? head = some { preLane with pc := dest } := by
          unfold State.getLane? at hPostLane
          rw [hPostWarp] at hPostLane
          simpa using hPostLane
        simp [hWsPostLane]
  have hPostLock : lockstepRunnable wsPost := by
    unfold lockstepRunnable lockstepRunnable?
    rw [hPostPc]
    simp [List.all_eq_true]
    intro lane hLane
    have hLanePart : lane ∈ participants := by
      rw [hPostRunParticipants] at hLane
      exact hLane
    obtain ⟨preLane, hPreLane, _hPrePc, _hRun⟩ := lane_pre ctx hLanePart
    have hMid := applyToLaneIds?_lane_in 0 0 fBranch participants
      (participants_nodup ctx) st post ctx.wf hApply lane hLanePart preLane hPreLane
    obtain ⟨postLane, hF, hPostLane⟩ := hMid
    simp [hfBranch] at hF
    subst postLane
    have hWsPostLane : wsPost.getLane? lane = some { preLane with pc := dest } := by
      unfold State.getLane? at hPostLane
      rw [hPostWarp] at hPostLane
      simpa using hPostLane
    rw [hWsPostLane]
    simp
  refine
    { post := post
      termStep := hTerm
      step := step_of_term ctx hTerm
      post_wf := hPostWf
      post_global := hTopApply.1
      post_const := hTopApply.2.1
      post_param := hTopApply.2.2.1
      post_kernelEnv := hTopApply.2.2.2.1
      post_atomics := hTopApply.2.2.2.2
      post_warpState := wsPost
      post_getWarp := hPostWarp
      post_warp_wf := hPostWarpWf
      post_lockstep := hPostLock
      post_runnable := hPostRunnable
      post_currentPc := hPostPc
      post_holds := ?_ }
  intro lane hLane
  obtain ⟨preLane, hPreLane, _hPc, _hRun⟩ := lane_pre ctx hLane
  have hMid := applyToLaneIds?_lane_in 0 0 fBranch participants
    (participants_nodup ctx) st post ctx.wf hApply lane hLane preLane hPreLane
  obtain ⟨postLane, hF, hPostLane⟩ := hMid
  simp [hfBranch] at hF
  subst postLane
  exact ⟨preLane, { preLane with pc := dest }, hPreLane, hPostLane, rfl, rfl, rfl, rfl, rfl⟩

end TermStepContext

/-- Projection of a load body step: every participating lane has the loaded
value in `dst`, advances by one slot, and remains runnable. -/
def LoadStepPost (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ laneState : LaneState,
      post.getLane? 0 0 lane = some laneState ∧
      laneState.regs[dst]? = some (valueAt lane) ∧
      laneState.pc = (pc.1, pc.2 + 1) ∧
      laneState.status = .running

/-- Full load-step postcondition. Besides the projected load result, this
exposes the precise register-map update relative to the pre-step lane. -/
def LoadStepFullPost (pre : State) (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ preLane postLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs = preLane.regs.insert dst (valueAt lane) ∧
      postLane.preds = preLane.preds ∧
      postLane.localMem = preLane.localMem ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running

namespace LoadStepFullPost

theorem loaded_dst
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants) :
    ∃ postLane : LaneState,
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs[dst]? = some (valueAt lane) ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running := by
  obtain ⟨_, postLane, _, hPost, hRegs, _, _, hPc, hStatus⟩ := h lane hLane
  refine ⟨postLane, hPost, ?_, hPc, hStatus⟩
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem toLoadStepPost
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post) :
    LoadStepPost participants dst valueAt pc post := by
  intro lane hLane
  exact loaded_dst h hLane

theorem loaded_dst_of_post_get
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane) :
    postLane.regs[dst]? = some (valueAt lane) := by
  obtain ⟨_, postLane', _, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem reg_ne_of_post_get
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hNe : r ≠ dst) :
    ∃ preLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      postLane.regs[r]? = preLane.regs[r]? := by
  obtain ⟨preLane, postLane', hPreLane, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  refine ⟨preLane, hPreLane, ?_⟩
  rw [hRegs, Std.HashMap.getElem?_insert]
  by_cases hEqKey : dst = r
  · exact False.elim (hNe hEqKey.symm)
  · simp [beq_iff_eq, hEqKey]

theorem preserves_reg_value
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {v : Value}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hReg : preLane.regs[r]? = some v)
    (hNe : r ≠ dst) :
    postLane.regs[r]? = some v := by
  obtain ⟨preLane', hPreLane', hFrame⟩ :=
    reg_ne_of_post_get h hLane hPostLane hNe
  have hEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  subst preLane'
  rw [hFrame, hReg]

theorem preserves_pred_value
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : LoadStepFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {p : PredName} {b : Bool}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hPred : preLane.preds[p]? = some b) :
    postLane.preds[p]? = some b := by
  obtain ⟨preLane', postLane', hPreLane', hPostLane', _, hPreds, _, _, _⟩ := h lane hLane
  have hPreEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  have hPostEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst preLane'
  subst postLane'
  rw [hPreds, hPred]

end LoadStepFullPost

/-- Projection of an `assignReg` body step: every participating lane has the
assigned value in `dst`, advances by one slot, and remains runnable. -/
def AssignRegStepPost (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ laneState : LaneState,
      post.getLane? 0 0 lane = some laneState ∧
      laneState.regs[dst]? = some (valueAt lane) ∧
      laneState.pc = (pc.1, pc.2 + 1) ∧
      laneState.status = .running

/-- Full `assignReg` postcondition, exposing the exact register-map update. -/
def AssignRegFullPost (pre : State) (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ preLane postLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs = preLane.regs.insert dst (valueAt lane) ∧
      postLane.preds = preLane.preds ∧
      postLane.localMem = preLane.localMem ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running

namespace AssignRegFullPost

theorem assigned_dst
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants) :
    ∃ postLane : LaneState,
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs[dst]? = some (valueAt lane) ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running := by
  obtain ⟨_, postLane, _, hPost, hRegs, _, _, hPc, hStatus⟩ := h lane hLane
  refine ⟨postLane, hPost, ?_, hPc, hStatus⟩
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem toAssignRegStepPost
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post) :
    AssignRegStepPost participants dst valueAt pc post := by
  intro lane hLane
  exact assigned_dst h hLane

theorem assigned_dst_of_post_get
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane) :
    postLane.regs[dst]? = some (valueAt lane) := by
  obtain ⟨_, postLane', _, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem reg_ne_of_post_get
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hNe : r ≠ dst) :
    ∃ preLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      postLane.regs[r]? = preLane.regs[r]? := by
  obtain ⟨preLane, postLane', hPreLane, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  refine ⟨preLane, hPreLane, ?_⟩
  rw [hRegs, Std.HashMap.getElem?_insert]
  by_cases hEqKey : dst = r
  · exact False.elim (hNe hEqKey.symm)
  · simp [beq_iff_eq, hEqKey]

theorem preserves_reg_value
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {v : Value}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hReg : preLane.regs[r]? = some v)
    (hNe : r ≠ dst) :
    postLane.regs[r]? = some v := by
  obtain ⟨preLane', hPreLane', hFrame⟩ :=
    reg_ne_of_post_get h hLane hPostLane hNe
  have hEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  subst preLane'
  rw [hFrame, hReg]

theorem preserves_pred_value
    {pre post : State} {participants : List LaneId} {dst : RegName} {p : PredName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : AssignRegFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {b : Bool}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hPred : preLane.preds[p]? = some b) :
    postLane.preds[p]? = some b := by
  obtain ⟨preLane', postLane', hPreLane', hPostLane', _, hPreds, _, _, _⟩ := h lane hLane
  have hPreEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  have hPostEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst preLane'
  subst postLane'
  rw [hPreds, hPred]

end AssignRegFullPost

/-- Full `cvta` postcondition. It has the same lane-state shape as `assignReg`,
but also exposes predicate/local-memory frame facts because the fallthrough
chain crosses a predicate-setting branch. -/
def CvtaFullPost (pre : State) (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ preLane postLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs = preLane.regs.insert dst (valueAt lane) ∧
      postLane.preds = preLane.preds ∧
      postLane.localMem = preLane.localMem ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running

namespace CvtaFullPost

theorem converted_dst
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : CvtaFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants) :
    ∃ postLane : LaneState,
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs[dst]? = some (valueAt lane) ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running := by
  obtain ⟨_, postLane, _, hPost, hRegs, _, _, hPc, hStatus⟩ := h lane hLane
  refine ⟨postLane, hPost, ?_, hPc, hStatus⟩
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem converted_dst_of_post_get
    {pre post : State} {participants : List LaneId} {dst : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : CvtaFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane) :
    postLane.regs[dst]? = some (valueAt lane) := by
  obtain ⟨_, postLane', _, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  rw [hRegs]
  simp [Std.HashMap.getElem?_insert]

theorem reg_ne_of_post_get
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : CvtaFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hNe : r ≠ dst) :
    ∃ preLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      postLane.regs[r]? = preLane.regs[r]? := by
  obtain ⟨preLane, postLane', hPreLane, hPostLane', hRegs, _, _, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  refine ⟨preLane, hPreLane, ?_⟩
  rw [hRegs, Std.HashMap.getElem?_insert]
  by_cases hEqKey : dst = r
  · exact False.elim (hNe hEqKey.symm)
  · simp [beq_iff_eq, hEqKey]

theorem preserves_reg_value
    {pre post : State} {participants : List LaneId} {dst r : RegName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : CvtaFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {v : Value}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hReg : preLane.regs[r]? = some v)
    (hNe : r ≠ dst) :
    postLane.regs[r]? = some v := by
  obtain ⟨preLane', hPreLane', hFrame⟩ :=
    reg_ne_of_post_get h hLane hPostLane hNe
  have hEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  subst preLane'
  rw [hFrame, hReg]

theorem preserves_pred_value
    {pre post : State} {participants : List LaneId} {dst : RegName} {p : PredName}
    {valueAt : LaneId → Value} {pc : PC}
    (h : CvtaFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {b : Bool}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hPred : preLane.preds[p]? = some b) :
    postLane.preds[p]? = some b := by
  obtain ⟨preLane', postLane', hPreLane', hPostLane', _, hPreds, _, _, _⟩ := h lane hLane
  have hPreEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  have hPostEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst preLane'
  subst postLane'
  rw [hPreds, hPred]

end CvtaFullPost

/-- Full `assignPred` postcondition, exposing the exact predicate-map update
and register preservation. -/
def AssignPredFullPost (pre : State) (participants : List LaneId) (dst : PredName)
    (valueAt : LaneId → Bool) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ preLane postLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      post.getLane? 0 0 lane = some postLane ∧
      postLane.regs = preLane.regs ∧
      postLane.preds = preLane.preds.insert dst (valueAt lane) ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running

namespace AssignPredFullPost

theorem assigned_dst
    {pre post : State} {participants : List LaneId} {dst : PredName}
    {valueAt : LaneId → Bool} {pc : PC}
    (h : AssignPredFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants) :
    ∃ postLane : LaneState,
      post.getLane? 0 0 lane = some postLane ∧
      postLane.preds[dst]? = some (valueAt lane) ∧
      postLane.pc = (pc.1, pc.2 + 1) ∧
      postLane.status = .running := by
  obtain ⟨_, postLane, _, hPost, _, hPreds, hPc, hStatus⟩ := h lane hLane
  refine ⟨postLane, hPost, ?_, hPc, hStatus⟩
  rw [hPreds]
  simp [Std.HashMap.getElem?_insert]

theorem assigned_dst_of_post_get
    {pre post : State} {participants : List LaneId} {dst : PredName}
    {valueAt : LaneId → Bool} {pc : PC}
    (h : AssignPredFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane) :
    postLane.preds[dst]? = some (valueAt lane) := by
  obtain ⟨_, postLane', _, hPostLane', _, hPreds, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  rw [hPreds]
  simp [Std.HashMap.getElem?_insert]

theorem pred_ne_of_post_get
    {pre post : State} {participants : List LaneId} {dst p : PredName}
    {valueAt : LaneId → Bool} {pc : PC}
    (h : AssignPredFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {postLane : LaneState}
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hNe : p ≠ dst) :
    ∃ preLane : LaneState,
      pre.getLane? 0 0 lane = some preLane ∧
      postLane.preds[p]? = preLane.preds[p]? := by
  obtain ⟨preLane, postLane', hPreLane, hPostLane', _, hPreds, _, _⟩ := h lane hLane
  have hEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst postLane'
  refine ⟨preLane, hPreLane, ?_⟩
  rw [hPreds, Std.HashMap.getElem?_insert]
  by_cases hEqKey : dst = p
  · exact False.elim (hNe hEqKey.symm)
  · simp [beq_iff_eq, hEqKey]

theorem preserves_pred_value
    {pre post : State} {participants : List LaneId} {dst p : PredName}
    {valueAt : LaneId → Bool} {pc : PC}
    (h : AssignPredFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {b : Bool}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hPred : preLane.preds[p]? = some b)
    (hNe : p ≠ dst) :
    postLane.preds[p]? = some b := by
  obtain ⟨preLane', hPreLane', hFrame⟩ :=
    pred_ne_of_post_get h hLane hPostLane hNe
  have hEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  subst preLane'
  rw [hFrame, hPred]

theorem preserves_reg_value
    {pre post : State} {participants : List LaneId} {dst : PredName} {r : RegName}
    {valueAt : LaneId → Bool} {pc : PC}
    (h : AssignPredFullPost pre participants dst valueAt pc post)
    {lane : LaneId} (hLane : lane ∈ participants)
    {preLane postLane : LaneState} {v : Value}
    (hPreLane : pre.getLane? 0 0 lane = some preLane)
    (hPostLane : post.getLane? 0 0 lane = some postLane)
    (hReg : preLane.regs[r]? = some v) :
    postLane.regs[r]? = some v := by
  obtain ⟨preLane', postLane', hPreLane', hPostLane', hRegs, _, _, _⟩ := h lane hLane
  have hPreEq : preLane' = preLane := by
    rw [hPreLane] at hPreLane'
    exact (Option.some.inj hPreLane').symm
  have hPostEq : postLane' = postLane := by
    rw [hPostLane] at hPostLane'
    exact (Option.some.inj hPostLane').symm
  subst preLane'
  subst postLane'
  rw [hRegs, hReg]

end AssignPredFullPost

namespace BodyStepContext

/-- Construct the standard load-step record from uniform address/read facts. -/
noncomputable def loadStep
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : RegName} {src : TypedAddr} {guard? : Option Guard}
    (ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .load dst src } participants)
    (valueAt : LaneId → Value)
    (hAddrRead : ∀ lane ∈ participants,
      ∃ addr, resolveAddr? st 0 0 lane src = some addr ∧
        readMem? st src.space src.ty addr = some (valueAt lane)) :
    BodyStepRecord ctx (LoadStepFullPost st participants dst valueAt pc) := by
  classical
  set fLoad : LaneId → LaneState → Option LaneState := fun lane laneState =>
    (resolveAddr? st 0 0 lane src).bind fun addr =>
      (readMem? st src.space src.ty addr).bind fun value =>
        some (writeReg laneState dst value) with hfLoad
  have hPartLaneSome : ∀ lane ∈ participants,
      ∃ ls, st.getLane? 0 0 lane = some ls ∧ (fLoad lane ls).isSome = true := by
    intro lane hLane
    obtain ⟨_, ls, hLsWs, _⟩ :=
      participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane
    refine ⟨ls, ?_, ?_⟩
    · unfold State.getLane?
      rw [ctx.getWarp]
      simp [hLsWs]
    · obtain ⟨addr, hAddr, hRead⟩ := hAddrRead lane hLane
      simp [hfLoad, hAddr, hRead]
  have hApplyIsSome :
      (applyToLaneIds? st 0 0 participants fLoad).isSome = true :=
    applyToLaneIds?_isSome_of_each_some (participants_nodup ctx) st ctx.wf hPartLaneSome
  have hApplyExists := Option.isSome_iff_exists.mp hApplyIsSome
  let stMid := Classical.choose hApplyExists
  have hApply : applyToLaneIds? st 0 0 participants fLoad = some stMid :=
    Classical.choose_spec hApplyExists
  have hPresF : PreservesStatusPc fLoad := by
    intro lane laneState laneState' hF
    simp [hfLoad, Option.bind_eq_some_iff] at hF
    obtain ⟨_, _, _, _, hWrite⟩ := hF
    rw [← hWrite]
    unfold writeReg
    exact ⟨rfl, rfl⟩
  have hWsMidExists := applyToLaneIds?_preserves_activeMask hApply ctx.warpState ctx.getWarp
  let wsMid := Classical.choose hWsMidExists
  have hWsMid : stMid.getWarp? 0 0 = some wsMid :=
    (Classical.choose_spec hWsMidExists).1
  have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf ctx.wf hApply
  have hPcMid : currentRunnablePc? wsMid = some pc := by
    rw [applyToLaneIds?_preserves_currentRunnablePc?
        hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply]
    exact ctx.currentPc
  have hRunMid : runnableLaneIds wsMid = runnableLaneIds ctx.warpState :=
    applyToLaneIds?_preserves_runnableLaneIds
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply
  have hLockMid : lockstepRunnable wsMid :=
    applyToLaneIds?_preserves_lockstepRunnable
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply ctx.lockstep
  have hAdvIsSome : (advanceRunnablePcs? stMid 0 0).isSome = true :=
    advanceRunnablePcs?_isSome_of_currentRunnablePc hWfMid hWsMid hPcMid
  have hAdvExists := Option.isSome_iff_exists.mp hAdvIsSome
  let post := Classical.choose hAdvExists
  have hAdv : advanceRunnablePcs? stMid 0 0 = some post :=
    Classical.choose_spec hAdvExists
  have hInstr :
      stepInstr? st 0 0 { guard? := guard?, instr := .load dst src } = some post := by
    unfold stepInstr?
    rw [ctx.getWarp]
    show ((some ctx.warpState).bind _) = some post
    rw [Option.some_bind]
    have hLockB := (lockstepRunnable_iff_bool ctx.warpState).1 ctx.lockstep
    simp [hLockB, ctx.participants_eq, hApply, hAdv, ← hfLoad]
  have hPostWarpExists := advanceRunnablePcs?_post_warp hWfMid hWsMid hLockMid hPcMid hAdv
  let wsPost := Classical.choose hPostWarpExists
  have hPostWarpSpec := Classical.choose_spec hPostWarpExists
  have hPostWf : State.wf post := hPostWarpSpec.1
  have hPostWarp : post.getWarp? 0 0 = some wsPost := hPostWarpSpec.2.1
  have hPostWarpWf : WarpState.wf wsPost := hPostWarpSpec.2.2.1
  have hRunPostMid : runnableLaneIds wsPost = runnableLaneIds wsMid := hPostWarpSpec.2.2.2.1
  have hPostPc : currentRunnablePc? wsPost = some (pc.1, pc.2 + 1) :=
    hPostWarpSpec.2.2.2.2.1
  have hPostLock : lockstepRunnable wsPost := hPostWarpSpec.2.2.2.2.2
  have hTopApply := applyToLaneIds?_preserves_top hApply
  have hTopAdv := advanceRunnablePcs?_preserves_top hAdv
  have hGlobal : post.global = st.global :=
    hTopAdv.1.trans hTopApply.1
  have hConst : post.const = st.const :=
    hTopAdv.2.1.trans hTopApply.2.1
  have hParam : post.param = st.param :=
    hTopAdv.2.2.1.trans hTopApply.2.2.1
  have hKernel : post.kernelEnv = st.kernelEnv :=
    hTopAdv.2.2.2.1.trans hTopApply.2.2.2.1
  have hAtomics : post.atomics = st.atomics :=
    hTopAdv.2.2.2.2.trans hTopApply.2.2.2.2
  have hPostRunnable : runnableLaneIds wsPost = runnableLaneIds ctx.warpState :=
    hRunPostMid.trans hRunMid
  refine
    { post := post
      instrStep := hInstr
      step := step_of_instr ctx hInstr
      post_wf := hPostWf
      post_global := hGlobal
      post_const := hConst
      post_param := hParam
      post_kernelEnv := hKernel
      post_atomics := hAtomics
      post_warpState := wsPost
      post_getWarp := hPostWarp
      post_warp_wf := hPostWarpWf
      post_lockstep := hPostLock
      post_runnable := hPostRunnable
      post_currentPc := hPostPc
      post_holds := ?_ }
  intro lane hLane
  obtain ⟨laneState, hGet, hLanePc⟩ := lane_pre ctx hLane
  have hStatus : laneState.status = .running := lane_status_running ctx hLane hGet
  obtain ⟨addr, hAddr, hRead⟩ := hAddrRead lane hLane
  obtain ⟨laneState', hGet', hRegs, _hPreds, _hLocal, hStatus', hPc'⟩ :=
    stepInstr?_load_lane_full
      (dst := dst) (src := src) (guard? := guard?) ctx.wf ctx.getWarp ctx.lockstep ctx.currentPc
      ctx.participants_eq hLane hGet hLanePc hAddr hRead hInstr
  refine ⟨laneState, laneState', hGet, hGet', hRegs, _hPreds, _hLocal, hPc', ?_⟩
  rw [hStatus', hStatus]

/-- Construct a full `assignReg` step record from per-lane evaluation facts. -/
noncomputable def assignRegStep
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : RegName} {rhs : RValue} {guard? : Option Guard}
    (ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .assignReg dst rhs } participants)
    (valueAt : LaneId → Value)
    (hEval : ∀ lane ∈ participants,
      evalRValue? st 0 0 lane rhs = some (valueAt lane)) :
    BodyStepRecord ctx (AssignRegFullPost st participants dst valueAt pc) := by
  classical
  set fAssign : LaneId → LaneState → Option LaneState := fun lane laneState =>
    (evalRValue? st 0 0 lane rhs).bind fun value =>
      some (writeReg laneState dst value) with hfAssign
  have hPartLaneSome : ∀ lane ∈ participants,
      ∃ ls, st.getLane? 0 0 lane = some ls ∧ (fAssign lane ls).isSome = true := by
    intro lane hLane
    obtain ⟨_, ls, hLsWs, _⟩ :=
      participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane
    refine ⟨ls, ?_, ?_⟩
    · unfold State.getLane?
      rw [ctx.getWarp]
      simp [hLsWs]
    · simp [hfAssign, hEval lane hLane]
  have hApplyIsSome :
      (applyToLaneIds? st 0 0 participants fAssign).isSome = true :=
    applyToLaneIds?_isSome_of_each_some (participants_nodup ctx) st ctx.wf hPartLaneSome
  have hApplyExists := Option.isSome_iff_exists.mp hApplyIsSome
  let stMid := Classical.choose hApplyExists
  have hApply : applyToLaneIds? st 0 0 participants fAssign = some stMid :=
    Classical.choose_spec hApplyExists
  have hPresF : PreservesStatusPc fAssign := by
    intro lane laneState laneState' hF
    simp [hfAssign, Option.bind_eq_some_iff] at hF
    obtain ⟨_, _, hWrite⟩ := hF
    rw [← hWrite]
    unfold writeReg
    exact ⟨rfl, rfl⟩
  have hWsMidExists := applyToLaneIds?_preserves_activeMask hApply ctx.warpState ctx.getWarp
  let wsMid := Classical.choose hWsMidExists
  have hWsMid : stMid.getWarp? 0 0 = some wsMid :=
    (Classical.choose_spec hWsMidExists).1
  have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf ctx.wf hApply
  have hPcMid : currentRunnablePc? wsMid = some pc := by
    rw [applyToLaneIds?_preserves_currentRunnablePc?
        hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply]
    exact ctx.currentPc
  have hRunMid : runnableLaneIds wsMid = runnableLaneIds ctx.warpState :=
    applyToLaneIds?_preserves_runnableLaneIds
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply
  have hLockMid : lockstepRunnable wsMid :=
    applyToLaneIds?_preserves_lockstepRunnable
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply ctx.lockstep
  have hAdvIsSome : (advanceRunnablePcs? stMid 0 0).isSome = true :=
    advanceRunnablePcs?_isSome_of_currentRunnablePc hWfMid hWsMid hPcMid
  have hAdvExists := Option.isSome_iff_exists.mp hAdvIsSome
  let post := Classical.choose hAdvExists
  have hAdv : advanceRunnablePcs? stMid 0 0 = some post :=
    Classical.choose_spec hAdvExists
  have hInstr :
      stepInstr? st 0 0 { guard? := guard?, instr := .assignReg dst rhs } = some post := by
    unfold stepInstr?
    rw [ctx.getWarp]
    show ((some ctx.warpState).bind _) = some post
    rw [Option.some_bind]
    have hLockB := (lockstepRunnable_iff_bool ctx.warpState).1 ctx.lockstep
    simp [hLockB, ctx.participants_eq, hApply, hAdv, ← hfAssign]
  have hPostWarpExists := advanceRunnablePcs?_post_warp hWfMid hWsMid hLockMid hPcMid hAdv
  let wsPost := Classical.choose hPostWarpExists
  have hPostWarpSpec := Classical.choose_spec hPostWarpExists
  have hPostWf : State.wf post := hPostWarpSpec.1
  have hPostWarp : post.getWarp? 0 0 = some wsPost := hPostWarpSpec.2.1
  have hPostWarpWf : WarpState.wf wsPost := hPostWarpSpec.2.2.1
  have hRunPostMid : runnableLaneIds wsPost = runnableLaneIds wsMid := hPostWarpSpec.2.2.2.1
  have hPostPc : currentRunnablePc? wsPost = some (pc.1, pc.2 + 1) :=
    hPostWarpSpec.2.2.2.2.1
  have hPostLock : lockstepRunnable wsPost := hPostWarpSpec.2.2.2.2.2
  have hTopApply := applyToLaneIds?_preserves_top hApply
  have hTopAdv := advanceRunnablePcs?_preserves_top hAdv
  have hGlobal : post.global = st.global :=
    hTopAdv.1.trans hTopApply.1
  have hConst : post.const = st.const :=
    hTopAdv.2.1.trans hTopApply.2.1
  have hParam : post.param = st.param :=
    hTopAdv.2.2.1.trans hTopApply.2.2.1
  have hKernel : post.kernelEnv = st.kernelEnv :=
    hTopAdv.2.2.2.1.trans hTopApply.2.2.2.1
  have hAtomics : post.atomics = st.atomics :=
    hTopAdv.2.2.2.2.trans hTopApply.2.2.2.2
  have hPostRunnable : runnableLaneIds wsPost = runnableLaneIds ctx.warpState :=
    hRunPostMid.trans hRunMid
  refine
    { post := post
      instrStep := hInstr
      step := step_of_instr ctx hInstr
      post_wf := hPostWf
      post_global := hGlobal
      post_const := hConst
      post_param := hParam
      post_kernelEnv := hKernel
      post_atomics := hAtomics
      post_warpState := wsPost
      post_getWarp := hPostWarp
      post_warp_wf := hPostWarpWf
      post_lockstep := hPostLock
      post_runnable := hPostRunnable
      post_currentPc := hPostPc
      post_holds := ?_ }
  intro lane hLane
  obtain ⟨laneState, hGet, hLanePc⟩ := lane_pre ctx hLane
  have hStatus : laneState.status = .running := lane_status_running ctx hLane hGet
  have hEvalLane := hEval lane hLane
  obtain ⟨laneState', hGet', hRegs, _hPreds, _hLocal, hStatus', hPc'⟩ :=
    stepInstr?_assignReg_lane_full
      (dst := dst) (rhs := rhs) (guard? := guard?) ctx.wf ctx.getWarp ctx.lockstep
      ctx.currentPc ctx.participants_eq hLane hGet hLanePc hEvalLane hInstr
  refine ⟨laneState, laneState', hGet, hGet', hRegs, _hPreds, _hLocal, hPc', ?_⟩
  rw [hStatus', hStatus]

/-- Construct an `assignReg` step record when only existence of the RHS value is
needed for chaining. Per-lane value facts can then be recovered from `instrStep`
with `stepInstr?_assignReg_lane_full`, avoiding expensive dependent
postcondition types for lane-indexed expressions. -/
noncomputable def assignRegStepSome
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : RegName} {rhs : RValue} {guard? : Option Guard}
    (ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .assignReg dst rhs } participants)
    (hEval : ∀ lane ∈ participants, ∃ value, evalRValue? st 0 0 lane rhs = some value) :
    BodyStepRecord ctx (fun _ => True) := by
  classical
  let valueAt : LaneId → Value := fun lane =>
    if h : lane ∈ participants then Classical.choose (hEval lane h) else default
  have hEvalValue : ∀ lane ∈ participants,
      evalRValue? st 0 0 lane rhs = some (valueAt lane) := by
    intro lane hLane
    dsimp [valueAt]
    rw [dif_pos hLane]
    exact Classical.choose_spec (hEval lane hLane)
  let r := assignRegStep ctx valueAt hEvalValue
  exact
    { post := r.post
      instrStep := r.instrStep
      step := r.step
      post_wf := r.post_wf
      post_global := r.post_global
      post_const := r.post_const
      post_param := r.post_param
      post_kernelEnv := r.post_kernelEnv
      post_atomics := r.post_atomics
      post_warpState := r.post_warpState
      post_getWarp := r.post_getWarp
      post_warp_wf := r.post_warp_wf
      post_lockstep := r.post_lockstep
      post_runnable := r.post_runnable
      post_currentPc := r.post_currentPc
      post_holds := trivial }

/-- Construct a full `cvta` step record from per-lane source and conversion facts. -/
noncomputable def cvtaStep
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard}
    (ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .cvta dst space src } participants)
    (valueAt : LaneId → Value)
    (hEval : ∀ lane ∈ participants,
      ∃ srcVal, evalRValue? st 0 0 lane src = some srcVal ∧
        evalCvta? space srcVal = some (valueAt lane)) :
    BodyStepRecord ctx (CvtaFullPost st participants dst valueAt pc) := by
  classical
  set fCvta : LaneId → LaneState → Option LaneState := fun lane laneState =>
    (evalRValue? st 0 0 lane src).bind fun value =>
      (evalCvta? space value).bind fun gaddr =>
        some (writeReg laneState dst gaddr) with hfCvta
  have hPartLaneSome : ∀ lane ∈ participants,
      ∃ ls, st.getLane? 0 0 lane = some ls ∧ (fCvta lane ls).isSome = true := by
    intro lane hLane
    obtain ⟨_, ls, hLsWs, _⟩ :=
      participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane
    refine ⟨ls, ?_, ?_⟩
    · unfold State.getLane?
      rw [ctx.getWarp]
      simp [hLsWs]
    · obtain ⟨srcVal, hSrc, hCvta⟩ := hEval lane hLane
      simp [hfCvta, hSrc, hCvta]
  have hApplyIsSome :
      (applyToLaneIds? st 0 0 participants fCvta).isSome = true :=
    applyToLaneIds?_isSome_of_each_some (participants_nodup ctx) st ctx.wf hPartLaneSome
  have hApplyExists := Option.isSome_iff_exists.mp hApplyIsSome
  let stMid := Classical.choose hApplyExists
  have hApply : applyToLaneIds? st 0 0 participants fCvta = some stMid :=
    Classical.choose_spec hApplyExists
  have hPresF : PreservesStatusPc fCvta := by
    intro lane laneState laneState' hF
    simp [hfCvta, Option.bind_eq_some_iff] at hF
    obtain ⟨_, _, _, _, hWrite⟩ := hF
    rw [← hWrite]
    unfold writeReg
    exact ⟨rfl, rfl⟩
  have hWsMidExists := applyToLaneIds?_preserves_activeMask hApply ctx.warpState ctx.getWarp
  let wsMid := Classical.choose hWsMidExists
  have hWsMid : stMid.getWarp? 0 0 = some wsMid :=
    (Classical.choose_spec hWsMidExists).1
  have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf ctx.wf hApply
  have hPcMid : currentRunnablePc? wsMid = some pc := by
    rw [applyToLaneIds?_preserves_currentRunnablePc?
        hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply]
    exact ctx.currentPc
  have hRunMid : runnableLaneIds wsMid = runnableLaneIds ctx.warpState :=
    applyToLaneIds?_preserves_runnableLaneIds
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply
  have hLockMid : lockstepRunnable wsMid :=
    applyToLaneIds?_preserves_lockstepRunnable
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply ctx.lockstep
  have hAdvIsSome : (advanceRunnablePcs? stMid 0 0).isSome = true :=
    advanceRunnablePcs?_isSome_of_currentRunnablePc hWfMid hWsMid hPcMid
  have hAdvExists := Option.isSome_iff_exists.mp hAdvIsSome
  let post := Classical.choose hAdvExists
  have hAdv : advanceRunnablePcs? stMid 0 0 = some post :=
    Classical.choose_spec hAdvExists
  have hInstr :
      stepInstr? st 0 0 { guard? := guard?, instr := .cvta dst space src } = some post := by
    unfold stepInstr?
    rw [ctx.getWarp]
    show ((some ctx.warpState).bind _) = some post
    rw [Option.some_bind]
    have hLockB := (lockstepRunnable_iff_bool ctx.warpState).1 ctx.lockstep
    simp [hLockB, ctx.participants_eq, hApply, hAdv, ← hfCvta]
  have hPostWarpExists := advanceRunnablePcs?_post_warp hWfMid hWsMid hLockMid hPcMid hAdv
  let wsPost := Classical.choose hPostWarpExists
  have hPostWarpSpec := Classical.choose_spec hPostWarpExists
  have hPostWf : State.wf post := hPostWarpSpec.1
  have hPostWarp : post.getWarp? 0 0 = some wsPost := hPostWarpSpec.2.1
  have hPostWarpWf : WarpState.wf wsPost := hPostWarpSpec.2.2.1
  have hRunPostMid : runnableLaneIds wsPost = runnableLaneIds wsMid := hPostWarpSpec.2.2.2.1
  have hPostPc : currentRunnablePc? wsPost = some (pc.1, pc.2 + 1) :=
    hPostWarpSpec.2.2.2.2.1
  have hPostLock : lockstepRunnable wsPost := hPostWarpSpec.2.2.2.2.2
  have hTopApply := applyToLaneIds?_preserves_top hApply
  have hTopAdv := advanceRunnablePcs?_preserves_top hAdv
  have hGlobal : post.global = st.global :=
    hTopAdv.1.trans hTopApply.1
  have hConst : post.const = st.const :=
    hTopAdv.2.1.trans hTopApply.2.1
  have hParam : post.param = st.param :=
    hTopAdv.2.2.1.trans hTopApply.2.2.1
  have hKernel : post.kernelEnv = st.kernelEnv :=
    hTopAdv.2.2.2.1.trans hTopApply.2.2.2.1
  have hAtomics : post.atomics = st.atomics :=
    hTopAdv.2.2.2.2.trans hTopApply.2.2.2.2
  have hPostRunnable : runnableLaneIds wsPost = runnableLaneIds ctx.warpState :=
    hRunPostMid.trans hRunMid
  refine
    { post := post
      instrStep := hInstr
      step := step_of_instr ctx hInstr
      post_wf := hPostWf
      post_global := hGlobal
      post_const := hConst
      post_param := hParam
      post_kernelEnv := hKernel
      post_atomics := hAtomics
      post_warpState := wsPost
      post_getWarp := hPostWarp
      post_warp_wf := hPostWarpWf
      post_lockstep := hPostLock
      post_runnable := hPostRunnable
      post_currentPc := hPostPc
      post_holds := ?_ }
  intro lane hLane
  obtain ⟨laneState, hGet, hLanePc⟩ := lane_pre ctx hLane
  have hStatus : laneState.status = .running := lane_status_running ctx hLane hGet
  obtain ⟨srcVal, hSrc, hCvta⟩ := hEval lane hLane
  obtain ⟨laneState', hGet', hRegs, _hPreds, _hLocal, hStatus', hPc'⟩ :=
    stepInstr?_cvta_lane_full
      (dst := dst) (space := space) (src := src) (guard? := guard?) ctx.wf ctx.getWarp
      ctx.lockstep ctx.currentPc ctx.participants_eq hLane hGet hLanePc hSrc hCvta hInstr
  refine ⟨laneState, laneState', hGet, hGet', hRegs, _hPreds, _hLocal, hPc', ?_⟩
  rw [hStatus', hStatus]

/-- Construct a full `assignPred` step record from per-lane comparison facts. -/
noncomputable def assignPredStep
    {st : State} {pc : PC} {block : Block} {participants : List LaneId}
    {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard}
    (ctx : BodyStepContext st pc block
      { guard? := guard?, instr := .assignPred dst cmp } participants)
    (valueAt : LaneId → Bool)
    (hEval : ∀ lane ∈ participants,
      evalCmp? st 0 0 lane cmp = some (valueAt lane)) :
    BodyStepRecord ctx (AssignPredFullPost st participants dst valueAt pc) := by
  classical
  set fAssign : LaneId → LaneState → Option LaneState := fun lane laneState =>
    (evalCmp? st 0 0 lane cmp).bind fun value =>
      some (writePred laneState dst value) with hfAssign
  have hPartLaneSome : ∀ lane ∈ participants,
      ∃ ls, st.getLane? 0 0 lane = some ls ∧ (fAssign lane ls).isSome = true := by
    intro lane hLane
    obtain ⟨_, ls, hLsWs, _⟩ :=
      participant_runnable_pc ctx.currentPc ctx.participants_eq lane hLane
    refine ⟨ls, ?_, ?_⟩
    · unfold State.getLane?
      rw [ctx.getWarp]
      simp [hLsWs]
    · simp [hfAssign, hEval lane hLane]
  have hApplyIsSome :
      (applyToLaneIds? st 0 0 participants fAssign).isSome = true :=
    applyToLaneIds?_isSome_of_each_some (participants_nodup ctx) st ctx.wf hPartLaneSome
  have hApplyExists := Option.isSome_iff_exists.mp hApplyIsSome
  let stMid := Classical.choose hApplyExists
  have hApply : applyToLaneIds? st 0 0 participants fAssign = some stMid :=
    Classical.choose_spec hApplyExists
  have hPresF : PreservesStatusPc fAssign := by
    intro lane laneState laneState' hF
    simp [hfAssign, Option.bind_eq_some_iff] at hF
    rcases hF with ⟨_, hWrite⟩ | ⟨_, hWrite⟩
    · rw [← hWrite]
      unfold writePred
      exact ⟨rfl, rfl⟩
    · rw [← hWrite]
      unfold writePred
      exact ⟨rfl, rfl⟩
  have hWsMidExists := applyToLaneIds?_preserves_activeMask hApply ctx.warpState ctx.getWarp
  let wsMid := Classical.choose hWsMidExists
  have hWsMid : stMid.getWarp? 0 0 = some wsMid :=
    (Classical.choose_spec hWsMidExists).1
  have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf ctx.wf hApply
  have hPcMid : currentRunnablePc? wsMid = some pc := by
    rw [applyToLaneIds?_preserves_currentRunnablePc?
        hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply]
    exact ctx.currentPc
  have hRunMid : runnableLaneIds wsMid = runnableLaneIds ctx.warpState :=
    applyToLaneIds?_preserves_runnableLaneIds
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply
  have hLockMid : lockstepRunnable wsMid :=
    applyToLaneIds?_preserves_lockstepRunnable
      hPresF ctx.wf ctx.getWarp hWsMid ctx.warp_wf hApply ctx.lockstep
  have hAdvIsSome : (advanceRunnablePcs? stMid 0 0).isSome = true :=
    advanceRunnablePcs?_isSome_of_currentRunnablePc hWfMid hWsMid hPcMid
  have hAdvExists := Option.isSome_iff_exists.mp hAdvIsSome
  let post := Classical.choose hAdvExists
  have hAdv : advanceRunnablePcs? stMid 0 0 = some post :=
    Classical.choose_spec hAdvExists
  have hInstr :
      stepInstr? st 0 0 { guard? := guard?, instr := .assignPred dst cmp } = some post := by
    unfold stepInstr?
    rw [ctx.getWarp]
    show ((some ctx.warpState).bind _) = some post
    rw [Option.some_bind]
    have hLockB := (lockstepRunnable_iff_bool ctx.warpState).1 ctx.lockstep
    simp [hLockB, ctx.participants_eq, hApply, hAdv, ← hfAssign]
  have hPostWarpExists := advanceRunnablePcs?_post_warp hWfMid hWsMid hLockMid hPcMid hAdv
  let wsPost := Classical.choose hPostWarpExists
  have hPostWarpSpec := Classical.choose_spec hPostWarpExists
  have hPostWf : State.wf post := hPostWarpSpec.1
  have hPostWarp : post.getWarp? 0 0 = some wsPost := hPostWarpSpec.2.1
  have hPostWarpWf : WarpState.wf wsPost := hPostWarpSpec.2.2.1
  have hRunPostMid : runnableLaneIds wsPost = runnableLaneIds wsMid := hPostWarpSpec.2.2.2.1
  have hPostPc : currentRunnablePc? wsPost = some (pc.1, pc.2 + 1) :=
    hPostWarpSpec.2.2.2.2.1
  have hPostLock : lockstepRunnable wsPost := hPostWarpSpec.2.2.2.2.2
  have hTopApply := applyToLaneIds?_preserves_top hApply
  have hTopAdv := advanceRunnablePcs?_preserves_top hAdv
  have hGlobal : post.global = st.global :=
    hTopAdv.1.trans hTopApply.1
  have hConst : post.const = st.const :=
    hTopAdv.2.1.trans hTopApply.2.1
  have hParam : post.param = st.param :=
    hTopAdv.2.2.1.trans hTopApply.2.2.1
  have hKernel : post.kernelEnv = st.kernelEnv :=
    hTopAdv.2.2.2.1.trans hTopApply.2.2.2.1
  have hAtomics : post.atomics = st.atomics :=
    hTopAdv.2.2.2.2.trans hTopApply.2.2.2.2
  have hPostRunnable : runnableLaneIds wsPost = runnableLaneIds ctx.warpState :=
    hRunPostMid.trans hRunMid
  refine
    { post := post
      instrStep := hInstr
      step := step_of_instr ctx hInstr
      post_wf := hPostWf
      post_global := hGlobal
      post_const := hConst
      post_param := hParam
      post_kernelEnv := hKernel
      post_atomics := hAtomics
      post_warpState := wsPost
      post_getWarp := hPostWarp
      post_warp_wf := hPostWarpWf
      post_lockstep := hPostLock
      post_runnable := hPostRunnable
      post_currentPc := hPostPc
      post_holds := ?_ }
  intro lane hLane
  obtain ⟨laneState, hGet, hLanePc⟩ := lane_pre ctx hLane
  have hStatus : laneState.status = .running := lane_status_running ctx hLane hGet
  have hEvalLane := hEval lane hLane
  obtain ⟨laneState', hGet', hRegs, hPreds, _hLocal, hStatus', hPc'⟩ :=
    stepInstr?_assignPred_lane_full
      (dst := dst) (cmp := cmp) (guard? := guard?) ctx.wf ctx.getWarp ctx.lockstep
      ctx.currentPc ctx.participants_eq hLane hGet hLanePc hEvalLane hInstr
  refine ⟨laneState, laneState', hGet, hGet', hRegs, hPreds, hPc', ?_⟩
  rw [hStatus', hStatus]

end BodyStepContext

/-! ## Per-lane initial-state characterization

For each lane `j` in `saxpyActiveLanes n hn`, the lane's initial state in
`saxpyStateFor n α xs ys` is the canonical `{ pc := ("saxpyKernel", 0) }`
(with all other fields default). -/

/-- The lane at any `j : LaneId` in `saxpyStateFor n α xs ys` is the canonical
initial lane state. -/
theorem saxpyStateFor_getLane (n : Nat) (alpha : Int) (xs ys : List Int)
    (j : LaneId) :
    (saxpyStateFor n alpha xs ys).getLane? 0 0 j =
      some { pc := ("saxpyKernel", 0) } := by
  unfold saxpyStateFor State.getLane? State.getWarp? State.getCTA?
  simp [saxpyWarpFor, WarpState.getLane?, Array.getElem?_eq_some_iff, j.isLt]

/-- Resolves the address of `ld.param.u32 [param_0]`: returns `.param 0`,
uniform across lanes. -/
theorem resolveAddr_saxpyStateFor_param0 (n : Nat) (alpha : Int) (xs ys : List Int)
    (j : LaneId) :
    resolveAddr? (saxpyStateFor n alpha xs ys) 0 0 j
        { space := .param, ty := .u32, addr := .imm (.u64 0) }
      = some (.param 0) := by
  unfold resolveAddr? evalRValue?
  rfl

theorem resolveAddr_param1
    (st : State) (j : LaneId) :
    resolveAddr? st 0 0 j
        { space := .param, ty := .s32, addr := .imm (.u64 4) }
      = some (.param 4) := by
  unfold resolveAddr? evalRValue?
  rfl

theorem resolveAddr_param_u64_imm
    (st : State) (j : LaneId) (offset : UInt64) :
    resolveAddr? st 0 0 j
        { space := .param, ty := .u64, addr := .imm (.u64 offset) }
      = some (.param offset.toNat) := by
  unfold resolveAddr? evalRValue?
  rfl

theorem saxpyKernelEnv_gridCtx :
    (PTX.lowerKernelEnvCheckedD saxpyKernel).gridCtx =
      { gridDim := { x := 1, y := 1, z := 1 }, blockDim := { x := 32, y := 1, z := 1 } } := by
  native_decide

theorem evalRValue_special_kernelEnv_congr
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {special : SpecialReg}
    (hKernel : st.kernelEnv = st'.kernelEnv) :
    evalRValue? st cta warp lane (.special special) =
      evalRValue? st' cta warp lane (.special special) := by
  unfold evalRValue?
  simp [evalSpecial, hKernel]

theorem evalRValue_saxpyStateFor_ctaidX
    (n : Nat) (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpyStateFor n alpha xs ys) 0 0 lane (.special .ctaidX) =
      some (.u32 0) := by
  unfold evalRValue? evalSpecial saxpyStateFor ctaIdxX dim3X nonzeroDim
  rw [saxpyKernelEnv_gridCtx]
  rfl

theorem evalRValue_saxpyStateFor_ntidX
    (n : Nat) (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpyStateFor n alpha xs ys) 0 0 lane (.special .ntidX) =
      some (.u32 32) := by
  unfold evalRValue? evalSpecial saxpyStateFor
  rw [saxpyKernelEnv_gridCtx]
  rfl

theorem evalRValue_saxpyStateFor_tidX
    (n : Nat) (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpyStateFor n alpha xs ys) 0 0 lane (.special .tidX) =
      some (.u32 (UInt32.ofNat lane.val)) := by
  unfold evalRValue? evalSpecial saxpyStateFor threadIdxX dim3X blockLinearTid nonzeroDim
  rw [saxpyKernelEnv_gridCtx]
  simp
  rw [Nat.mod_eq_of_lt lane.isLt]

theorem saxpyActiveLanes_mem_lt
    (n : Nat) (hn : n ≤ 32) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) : j.val < n := by
  rw [saxpyActiveLanes, List.mem_pmap] at hj
  rcases hj with ⟨i, hi, hEq⟩
  have hiLt : i < n := List.mem_range.mp hi
  have hVal : i = j.val := congrArg Fin.val hEq
  omega

theorem saxpyActiveLanes_ne_nil
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n) :
    saxpyActiveLanes n hn ≠ [] := by
  unfold saxpyActiveLanes
  cases n with
  | zero => omega
  | succ _ => simp

theorem evalUnary_cvt_s32_u32_of_le32 (n : Nat) (hn : n ≤ 32) :
    evalUnary? (.cvt .s32) (.u32 (UInt32.ofNat n)) = some (.s32 (Int.ofNat n)) := by
  interval_cases n <;> rfl

theorem evalCmp_saxpy_setp_ge_false
    {st : State} {j : LaneId} {ls : LaneState} {n : Nat}
    (hn : n ≤ 32) (hjLt : j.val < n)
    (hGet : st.getLane? 0 0 j = some ls)
    (hR1 : ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)))
    (hR2 : ls.regs["r2"]? = some (.u32 (UInt32.ofNat n))) :
    evalCmp? st 0 0 j saxpySetpGeS32IndexLen = some false := by
  have hLhs : evalRValue? st 0 0 j (.reg "r1") = some (.s32 (Int.ofNat j.val)) := by
    simp [evalRValue?, readReg, hGet, hR1]
  have hRhsReg : evalRValue? st 0 0 j (.reg "r2") = some (.u32 (UInt32.ofNat n)) := by
    simp [evalRValue?, readReg, hGet, hR2]
  have hCvt := evalUnary_cvt_s32_u32_of_le32 n hn
  have hRhs :
      evalRValue? st 0 0 j (.unop (.cvt .s32) (.reg "r2")) = some (.s32 (Int.ofNat n)) := by
    unfold evalRValue?
    rw [hRhsReg]
    exact hCvt
  have hNot : ¬ Int.ofNat j.val ≥ Int.ofNat n := by
    change ¬ ((j.val : Int) ≥ (n : Int))
    have hltInt : (j.val : Int) < (n : Int) := by exact_mod_cast hjLt
    exact not_le.mpr hltInt
  unfold saxpySetpGeS32IndexLen evalCmp?
  rw [hLhs, hRhs]
  change some (decide (Int.ofNat j.val ≥ Int.ofNat n)) = some false
  simp [hNot]
  exact hjLt

/-- Structural context for saxpy step 1, the first BB0 body instruction. -/
def saxpy_step1_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpyStateFor n alpha xs ys) ("saxpyKernel", 0)
      saxpyBB0 saxpyBB0_load_param0 (saxpyActiveLanes n hn) :=
  { warpState := saxpyWarpFor n
    wf := saxpyStateFor_wf n alpha xs ys
    getWarp := saxpyStateFor_getWarp n alpha xs ys
    warp_wf := saxpyWarp_wf n
    lockstep := saxpyWarp_lockstepRunnable n
    currentPc := currentRunnablePc_saxpyWarp_pos n hn hnpos
    block_lookup := saxpyStateFor_blocks_lookup n alpha xs ys
    body_slot := saxpyBB0_body0_load_param0
    participants_eq := participatingRunnable_saxpyWarp_none n hn hnpos }

/-- Step-record form of the first saxpy load. Future chained steps should
consume and produce records of this shape instead of rebuilding the
`step?` plumbing locally. -/
noncomputable def saxpy_step1_load_param0_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step1_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpyStateFor n alpha xs ys) (saxpyActiveLanes n hn) "r2"
        (fun _ => .u32 (UInt32.ofNat n)) ("saxpyKernel", 0)) :=
  BodyStepContext.loadStep (saxpy_step1_ctx n hn hnpos alpha xs ys)
    (dst := "r2")
    (src := { space := .param, ty := .u32, addr := .imm (.u64 0) })
    (guard? := none)
    (fun _ => .u32 (UInt32.ofNat n))
    (fun lane _hLane =>
      ⟨.param 0, resolveAddr_saxpyStateFor_param0 n alpha xs ys lane,
       readMem_saxpyStateFor_param0 n alpha xs ys⟩)

/-- The body-step context after step 1, at BB0 slot 1. -/
noncomputable def saxpy_step2_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 1) saxpyBB0 saxpyBB0_load_param1 (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step1_load_param0_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body1_load_param1

lemma readMem_saxpy_step1_post_param1
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    readMem? (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post
        .param .s32 (.param 4) =
      some (.s32 (saxpyAlphaS32 alpha)) := by
  have hParam := (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post_param
  rw [readMem?_param_s32_congr hParam]
  exact readMem_saxpyStateFor_param1 n alpha xs ys

/-- Step-record form of the second saxpy load. -/
noncomputable def saxpy_step2_load_param1_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step2_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r6"
        (fun _ => .s32 (saxpyAlphaS32 alpha)) ("saxpyKernel", 1)) :=
  BodyStepContext.loadStep (saxpy_step2_ctx n hn hnpos alpha xs ys)
    (dst := "r6")
    (src := { space := .param, ty := .s32, addr := .imm (.u64 4) })
    (guard? := none)
    (fun _ => .s32 (saxpyAlphaS32 alpha))
    (fun lane _hLane =>
      ⟨.param 4,
       resolveAddr_param1
         (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post lane,
       readMem_saxpy_step1_post_param1 n hn hnpos alpha xs ys⟩)

/-- The body-step context after step 2, at BB0 slot 2. -/
noncomputable def saxpy_step3_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 2) saxpyBB0 saxpyBB0_load_param2 (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step2_load_param1_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body2_load_param2

lemma readMem_saxpy_step2_post_param2
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    readMem? (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post
        .param .u64 (.param 8) =
      some (.u64 (UInt64.ofNat saxpyXBase)) := by
  have hParam :
      (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post.param =
        (saxpyStateFor n alpha xs ys).param :=
    (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post_param.trans
      (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post_param
  rw [readMem?_param_u64_congr hParam]
  exact readMem_saxpyStateFor_param2 n alpha xs ys

/-- Step-record form of the third saxpy load. -/
noncomputable def saxpy_step3_load_param2_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step3_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "rd1"
        (fun _ => .u64 (UInt64.ofNat saxpyXBase)) ("saxpyKernel", 2)) :=
  BodyStepContext.loadStep (saxpy_step3_ctx n hn hnpos alpha xs ys)
    (dst := "rd1")
    (src := { space := .param, ty := .u64, addr := .imm (.u64 8) })
    (guard? := none)
    (fun _ => .u64 (UInt64.ofNat saxpyXBase))
    (fun lane _hLane =>
      ⟨.param 8,
       resolveAddr_param_u64_imm
         (saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post lane 8,
       readMem_saxpy_step2_post_param2 n hn hnpos alpha xs ys⟩)

/-- The body-step context after step 3, at BB0 slot 3. -/
noncomputable def saxpy_step4_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 3) saxpyBB0 saxpyBB0_load_param3 (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step3_load_param2_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body3_load_param3

lemma readMem_saxpy_step3_post_param3
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    readMem? (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post
        .param .u64 (.param 16) =
      some (.u64 (UInt64.ofNat saxpyYBase)) := by
  have hParam :
      (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post.param =
        (saxpyStateFor n alpha xs ys).param :=
    (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post_param.trans
      ((saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post_param.trans
        (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post_param)
  rw [readMem?_param_u64_congr hParam]
  exact readMem_saxpyStateFor_param3 n alpha xs ys

/-- Step-record form of the fourth saxpy load. -/
noncomputable def saxpy_step4_load_param3_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step4_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "rd2"
        (fun _ => .u64 (UInt64.ofNat saxpyYBase)) ("saxpyKernel", 3)) :=
  BodyStepContext.loadStep (saxpy_step4_ctx n hn hnpos alpha xs ys)
    (dst := "rd2")
    (src := { space := .param, ty := .u64, addr := .imm (.u64 16) })
    (guard? := none)
    (fun _ => .u64 (UInt64.ofNat saxpyYBase))
    (fun lane _hLane =>
      ⟨.param 16,
       resolveAddr_param_u64_imm
         (saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post lane 16,
       readMem_saxpy_step3_post_param3 n hn hnpos alpha xs ys⟩)

/-- The body-step context after step 4, at BB0 slot 4. -/
noncomputable def saxpy_step5_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 4) saxpyBB0 saxpyBB0_load_param4 (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step4_load_param3_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body4_load_param4

lemma readMem_saxpy_step4_post_param4
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    readMem? (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post
        .param .u64 (.param 24) =
      some (.u64 (UInt64.ofNat saxpyRBase)) := by
  have hParam :
      (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post.param =
        (saxpyStateFor n alpha xs ys).param :=
    (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post_param.trans
      ((saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post_param.trans
        ((saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post_param.trans
          (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post_param))
  rw [readMem?_param_u64_congr hParam]
  exact readMem_saxpyStateFor_param4 n alpha xs ys

/-- Step-record form of the fifth saxpy load. -/
noncomputable def saxpy_step5_load_param4_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step5_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "rd3"
        (fun _ => .u64 (UInt64.ofNat saxpyRBase)) ("saxpyKernel", 4)) :=
  BodyStepContext.loadStep (saxpy_step5_ctx n hn hnpos alpha xs ys)
    (dst := "rd3")
    (src := { space := .param, ty := .u64, addr := .imm (.u64 24) })
    (guard? := none)
    (fun _ => .u64 (UInt64.ofNat saxpyRBase))
    (fun lane _hLane =>
      ⟨.param 24,
       resolveAddr_param_u64_imm
         (saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post lane 24,
       readMem_saxpy_step4_post_param4 n hn hnpos alpha xs ys⟩)

theorem saxpy_step5_post_kernelEnv
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post.kernelEnv =
      (saxpyStateFor n alpha xs ys).kernelEnv :=
  (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post_kernelEnv.trans
    ((saxpy_step4_load_param3_record n hn hnpos alpha xs ys).post_kernelEnv.trans
      ((saxpy_step3_load_param2_record n hn hnpos alpha xs ys).post_kernelEnv.trans
        ((saxpy_step2_load_param1_record n hn hnpos alpha xs ys).post_kernelEnv.trans
          (saxpy_step1_load_param0_record n hn hnpos alpha xs ys).post_kernelEnv)))

/-- The body-step context after step 5, at BB0 slot 5. -/
noncomputable def saxpy_step6_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 5) saxpyBB0 saxpyBB0_mov_ctaidX (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step5_load_param4_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body5_mov_ctaidX

lemma evalRValue_saxpy_step5_post_ctaidX
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post
        0 0 lane (.special .ctaidX) =
      some (.u32 0) := by
  rw [evalRValue_special_kernelEnv_congr
    (saxpy_step5_post_kernelEnv n hn hnpos alpha xs ys)]
  exact evalRValue_saxpyStateFor_ctaidX n alpha xs ys lane

/-- Step-record form of `mov.u32 %r3, %ctaid.x`. -/
noncomputable def saxpy_step6_mov_ctaidX_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step6_ctx n hn hnpos alpha xs ys)
      (AssignRegFullPost (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r3" (fun _ => .u32 0) ("saxpyKernel", 5)) :=
  BodyStepContext.assignRegStep (saxpy_step6_ctx n hn hnpos alpha xs ys)
    (dst := "r3")
    (rhs := .special .ctaidX)
    (guard? := none)
    (fun _ => .u32 0)
    (fun lane _hLane => evalRValue_saxpy_step5_post_ctaidX n hn hnpos alpha xs ys lane)

theorem saxpy_step6_post_kernelEnv
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post.kernelEnv =
      (saxpyStateFor n alpha xs ys).kernelEnv :=
  (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post_kernelEnv.trans
    (saxpy_step5_post_kernelEnv n hn hnpos alpha xs ys)

/-- The body-step context after step 6, at BB0 slot 6. -/
noncomputable def saxpy_step7_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 6) saxpyBB0 saxpyBB0_mov_ntidX (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body6_mov_ntidX

lemma evalRValue_saxpy_step6_post_ntidX
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post
        0 0 lane (.special .ntidX) =
      some (.u32 32) := by
  rw [evalRValue_special_kernelEnv_congr
    (saxpy_step6_post_kernelEnv n hn hnpos alpha xs ys)]
  exact evalRValue_saxpyStateFor_ntidX n alpha xs ys lane

/-- Step-record form of `mov.u32 %r4, %ntid.x`. -/
noncomputable def saxpy_step7_mov_ntidX_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step7_ctx n hn hnpos alpha xs ys)
      (AssignRegFullPost (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r4" (fun _ => .u32 32) ("saxpyKernel", 6)) :=
  BodyStepContext.assignRegStep (saxpy_step7_ctx n hn hnpos alpha xs ys)
    (dst := "r4")
    (rhs := .special .ntidX)
    (guard? := none)
    (fun _ => .u32 32)
    (fun lane _hLane => evalRValue_saxpy_step6_post_ntidX n hn hnpos alpha xs ys lane)

theorem saxpy_step7_post_kernelEnv
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys).post.kernelEnv =
      (saxpyStateFor n alpha xs ys).kernelEnv :=
  (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys).post_kernelEnv.trans
    (saxpy_step6_post_kernelEnv n hn hnpos alpha xs ys)

/-- The body-step context after step 7, at BB0 slot 7. -/
noncomputable def saxpy_step8_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 7) saxpyBB0 saxpyBB0_mov_tidX (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body7_mov_tidX

lemma evalRValue_saxpy_step7_post_tidX
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (lane : LaneId) :
    evalRValue? (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys).post
        0 0 lane (.special .tidX) =
      some (.u32 (UInt32.ofNat lane.val)) := by
  rw [evalRValue_special_kernelEnv_congr
    (saxpy_step7_post_kernelEnv n hn hnpos alpha xs ys)]
  exact evalRValue_saxpyStateFor_tidX n alpha xs ys lane

/-- Step-record form of `mov.u32 %r5, %tid.x`. -/
noncomputable def saxpy_step8_mov_tidX_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step8_ctx n hn hnpos alpha xs ys)
      (AssignRegFullPost (saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r5" (fun lane => .u32 (UInt32.ofNat lane.val))
        ("saxpyKernel", 7)) :=
  BodyStepContext.assignRegStep (saxpy_step8_ctx n hn hnpos alpha xs ys)
    (dst := "r5")
    (rhs := .special .tidX)
    (guard? := none)
    (fun lane => .u32 (UInt32.ofNat lane.val))
    (fun lane _hLane => evalRValue_saxpy_step7_post_tidX n hn hnpos alpha xs ys lane)

/-- The body-step context after step 8, at BB0 slot 8. -/
noncomputable def saxpy_step9_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 8) saxpyBB0 saxpyBB0_mad_index (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body8_mad_index

/-- Interface after the five parameter loads have executed. -/
def SaxpyParamLoadsPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) : Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.pc = ("saxpyKernel", 5) ∧
      ls.status = .running

/-- Interface after the first special-register move (`%ctaid.x`) has executed. -/
def SaxpyAfterStep6Post (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) : Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.pc = ("saxpyKernel", 6) ∧
      ls.status = .running

/-- Interface before the first arithmetic instruction: all parameter loads and
the three special-register moves are available in every active lane. -/
def SaxpySpecialRegsPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) : Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.pc = ("saxpyKernel", 8) ∧
      ls.status = .running

/-- Interface after computing the signed lane index in `%r1`. -/
def SaxpyIndexPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) : Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.pc = ("saxpyKernel", 9) ∧
      ls.status = .running

/-- Interface after computing the loop-bound branch predicate in `%p1`. -/
def SaxpyBranchPredPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.pc = ("saxpyKernel", 10) ∧
      ls.status = .running

/-- Interface after BB0's branch terminator has taken the uniform fallthrough path. -/
def SaxpyBranchTargetPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.pc = ("saxpyKernel$fallthrough0", 0) ∧
      ls.status = .running

/-- Interface after `cvta.to.global.u64 %rd4, %rd1` in the fallthrough block. -/
def SaxpyAfterCvtaRd4Post (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 1) ∧
      ls.status = .running

/-- Interface after `mul.wide.s32 %rd5, %r1, 4`. -/
def SaxpyAfterMulWidePost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 2) ∧
      ls.status = .running

/-- Interface after `add.s64 %rd6, %rd4, %rd5`. -/
def SaxpyAfterXAddrPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 3) ∧
      ls.status = .running

/-- Interface after `cvta.to.global.u64 %rd7, %rd2`. -/
def SaxpyAfterCvtaRd7Post (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 4) ∧
      ls.status = .running

/-- Interface after `add.s64 %rd8, %rd7, %rd5`. -/
def SaxpyAfterYAddrPost (n : Nat) (hn : n ≤ 32) (alpha : Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.regs["rd8"]? = some (saxpyYElementAddrValue j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 5) ∧
      ls.status = .running

/-- Interface after `ld.global.s32 %r7, [%rd6]`. -/
def SaxpyAfterXLoadPost
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs : List Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.regs["rd8"]? = some (saxpyYElementAddrValue j) ∧
      ls.regs["r7"]? = some (saxpyLoadedS32Value xs j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 6) ∧
      ls.status = .running

/-- Interface after `ld.global.s32 %r8, [%rd8]`. -/
def SaxpyAfterYLoadPost
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.regs["rd8"]? = some (saxpyYElementAddrValue j) ∧
      ls.regs["r7"]? = some (saxpyLoadedS32Value xs j) ∧
      ls.regs["r8"]? = some (saxpyLoadedS32Value ys j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 7) ∧
      ls.status = .running

/-- Interface after `mad.lo.s32 %r9, %r7, %r6, %r8`. -/
def SaxpyAfterMulAddPost
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.regs["rd8"]? = some (saxpyYElementAddrValue j) ∧
      ls.regs["r7"]? = some (saxpyLoadedS32Value xs j) ∧
      ls.regs["r8"]? = some (saxpyLoadedS32Value ys j) ∧
      ls.regs["r9"]? = some (saxpyMulAddValue alpha xs ys j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 8) ∧
      ls.status = .running

/-- Interface after `cvta.to.global.u64 %rd9, %rd3`. -/
def SaxpyAfterCvtaRd9Post
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
      ls.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
      ls.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
      ls.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
      ls.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
      ls.regs["r3"]? = some (.u32 0) ∧
      ls.regs["r4"]? = some (.u32 32) ∧
      ls.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) ∧
      ls.regs["r1"]? = some (.s32 (Int.ofNat j.val)) ∧
      ls.preds["p1"]? = some false ∧
      ls.regs["rd4"]? = some (.gaddr .global saxpyXBase) ∧
      ls.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) ∧
      ls.regs["rd6"]? = some (saxpyXElementAddrValue j) ∧
      ls.regs["rd7"]? = some (.gaddr .global saxpyYBase) ∧
      ls.regs["rd8"]? = some (saxpyYElementAddrValue j) ∧
      ls.regs["r7"]? = some (saxpyLoadedS32Value xs j) ∧
      ls.regs["r8"]? = some (saxpyLoadedS32Value ys j) ∧
      ls.regs["r9"]? = some (saxpyMulAddValue alpha xs ys j) ∧
      ls.regs["rd9"]? = some (.gaddr .global saxpyRBase) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 9) ∧
      ls.status = .running

/-- Interface after `add.s64 %rd10, %rd9, %rd5`. -/
def SaxpyAfterResultAddrPost
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (post : State) :
    Prop :=
  ∀ j ∈ saxpyActiveLanes n hn,
    ∃ ls : LaneState,
      post.getLane? 0 0 j = some ls ∧
      ls.regs["r9"]? = some (saxpyMulAddValue alpha xs ys j) ∧
      ls.regs["rd10"]? = some (saxpyResultElementAddrValue j) ∧
      ls.pc = ("saxpyKernel$fallthrough0", 10) ∧
      ls.status = .running

/-! ## Step 1 of the chain: `ld.param.u32 %r2, [param_0]`

The first executable step from `saxpyStateFor n α xs ys` (with `n > 0`)
loads `(UInt32.ofNat n)` into `%r2` on every active lane and advances PC
to `("saxpyKernel", 1)`. -/

theorem saxpy_step1_load_param0
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      ∀ j ∈ saxpyActiveLanes n hn,
        ∃ ls1 : LaneState,
          st1.getLane? 0 0 j = some ls1 ∧
          ls1.regs["r2"]? = some (.u32 (UInt32.ofNat n)) ∧
          ls1.pc = ("saxpyKernel", 1) ∧
          ls1.status = .running := by
  obtain ⟨st1, hStep, hPost⟩ :=
    BodyStepRecord.to_exists (saxpy_step1_load_param0_record n hn hnpos alpha xs ys)
  refine ⟨st1, hStep, ?_⟩
  have hPostSimple := LoadStepFullPost.toLoadStepPost hPost
  intro j hj
  simpa [LoadStepPost] using hPostSimple j hj

/-! ## Step 2 of the chain: `ld.param.s32 %r6, [param_1]`

This uses the successor context produced by the step-1 load record. The
second load reads the preserved parameter memory in the step-1 post-state,
loads the wrapped s32 value of `α` into `%r6`, and advances to BB0 slot 2. -/

theorem saxpy_step2_load_param1
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      ∀ j ∈ saxpyActiveLanes n hn,
        ∃ ls2 : LaneState,
          st2.getLane? 0 0 j = some ls2 ∧
          ls2.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) ∧
          ls2.pc = ("saxpyKernel", 2) ∧
          ls2.status = .running := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  refine ⟨step1.post, step2.post, step1.step, step2.step, ?_⟩
  have hPostSimple := LoadStepFullPost.toLoadStepPost step2.post_holds
  intro j hj
  simpa [LoadStepPost, step2] using hPostSimple j hj

/-! ## Step 3 of the chain: `ld.param.u64 %rd1, [param_2]`

The third load reads the preserved parameter memory after step 2 and loads the
`xs` base pointer into `%rd1`, advancing to BB0 slot 3. -/

theorem saxpy_step3_load_param2
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      ∀ j ∈ saxpyActiveLanes n hn,
        ∃ ls3 : LaneState,
          st3.getLane? 0 0 j = some ls3 ∧
          ls3.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) ∧
          ls3.pc = ("saxpyKernel", 3) ∧
          ls3.status = .running := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  refine ⟨step1.post, step2.post, step3.post, step1.step, step2.step, step3.step, ?_⟩
  have hPostSimple := LoadStepFullPost.toLoadStepPost step3.post_holds
  intro j hj
  simpa [LoadStepPost, step3] using hPostSimple j hj

/-! ## Step 4 of the chain: `ld.param.u64 %rd2, [param_3]` -/

theorem saxpy_step4_load_param3
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      ∀ j ∈ saxpyActiveLanes n hn,
        ∃ ls4 : LaneState,
          st4.getLane? 0 0 j = some ls4 ∧
          ls4.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) ∧
          ls4.pc = ("saxpyKernel", 4) ∧
          ls4.status = .running := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post,
      step1.step, step2.step, step3.step, step4.step, ?_⟩
  have hPostSimple := LoadStepFullPost.toLoadStepPost step4.post_holds
  intro j hj
  simpa [LoadStepPost, step4] using hPostSimple j hj

/-! ## Step 5 of the chain: `ld.param.u64 %rd3, [param_4]` -/

theorem saxpy_step5_load_param4
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      ∀ j ∈ saxpyActiveLanes n hn,
        ∃ ls5 : LaneState,
          st5.getLane? 0 0 j = some ls5 ∧
          ls5.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) ∧
          ls5.pc = ("saxpyKernel", 5) ∧
          ls5.status = .running := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, ?_⟩
  have hPostSimple := LoadStepFullPost.toLoadStepPost step5.post_holds
  intro j hj
  simpa [LoadStepPost, step5] using hPostSimple j hj

theorem saxpy_step5_param_loads_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      SaxpyParamLoadsPost n hn alpha st5 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, ?_⟩
  intro j hj
  obtain ⟨ls5, hGet5, hRd3, hPc5, hStatus5⟩ :=
    LoadStepFullPost.loaded_dst step5.post_holds hj
  have hRd2 : ls5.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd2") step5.post_holds hj hGet5
        (by decide)
    have hLoad4 := LoadStepFullPost.loaded_dst_of_post_get step4.post_holds hj hGet4
    rw [hFrame5, hLoad4]
  have hRd1 : ls5.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd1") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd1") step4.post_holds hj hGet4
        (by decide)
    have hLoad3 := LoadStepFullPost.loaded_dst_of_post_get step3.post_holds hj hGet3
    rw [hFrame5, hFrame4, hLoad3]
  have hR6 : ls5.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step4.post_holds hj hGet4
        (by decide)
    obtain ⟨ls2, hGet2, hFrame3⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step3.post_holds hj hGet3
        (by decide)
    have hLoad2 := LoadStepFullPost.loaded_dst_of_post_get step2.post_holds hj hGet2
    rw [hFrame5, hFrame4, hFrame3, hLoad2]
  have hR2 : ls5.regs["r2"]? = some (.u32 (UInt32.ofNat n)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step4.post_holds hj hGet4
        (by decide)
    obtain ⟨ls2, hGet2, hFrame3⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step3.post_holds hj hGet3
        (by decide)
    obtain ⟨ls1, hGet1, hFrame2⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step2.post_holds hj hGet2
        (by decide)
    have hLoad1 := LoadStepFullPost.loaded_dst_of_post_get step1.post_holds hj hGet1
    rw [hFrame5, hFrame4, hFrame3, hFrame2, hLoad1]
  refine ⟨ls5, hGet5, hR2, hR6, hRd1, hRd2, hRd3, hPc5, hStatus5⟩

theorem saxpy_step5_param_loads_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyParamLoadsPost n hn alpha
      (saxpy_step5_load_param4_record n hn hnpos alpha xs ys).post := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls5, hGet5, hRd3, hPc5, hStatus5⟩ :=
    LoadStepFullPost.loaded_dst step5.post_holds hj
  have hRd2 : ls5.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd2") step5.post_holds hj hGet5
        (by decide)
    have hLoad4 := LoadStepFullPost.loaded_dst_of_post_get step4.post_holds hj hGet4
    rw [hFrame5, hLoad4]
  have hRd1 : ls5.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd1") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "rd1") step4.post_holds hj hGet4
        (by decide)
    have hLoad3 := LoadStepFullPost.loaded_dst_of_post_get step3.post_holds hj hGet3
    rw [hFrame5, hFrame4, hLoad3]
  have hR6 : ls5.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step4.post_holds hj hGet4
        (by decide)
    obtain ⟨ls2, hGet2, hFrame3⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r6") step3.post_holds hj hGet3
        (by decide)
    have hLoad2 := LoadStepFullPost.loaded_dst_of_post_get step2.post_holds hj hGet2
    rw [hFrame5, hFrame4, hFrame3, hLoad2]
  have hR2 : ls5.regs["r2"]? = some (.u32 (UInt32.ofNat n)) := by
    obtain ⟨ls4, hGet4, hFrame5⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step5.post_holds hj hGet5
        (by decide)
    obtain ⟨ls3, hGet3, hFrame4⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step4.post_holds hj hGet4
        (by decide)
    obtain ⟨ls2, hGet2, hFrame3⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step3.post_holds hj hGet3
        (by decide)
    obtain ⟨ls1, hGet1, hFrame2⟩ :=
      LoadStepFullPost.reg_ne_of_post_get (r := "r2") step2.post_holds hj hGet2
        (by decide)
    have hLoad1 := LoadStepFullPost.loaded_dst_of_post_get step1.post_holds hj hGet1
    rw [hFrame5, hFrame4, hFrame3, hFrame2, hLoad1]
  refine ⟨ls5, hGet5, hR2, hR6, hRd1, hRd2, hRd3, hPc5, hStatus5⟩

theorem saxpy_step6_ctaid_accumulated_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterStep6Post n hn alpha
      (saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys).post := by
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  have hParam := saxpy_step5_param_loads_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls5, hGet5, hR2, hR6, hRd1, hRd2, hRd3, _hPc5, _hStatus5⟩ := hParam j hj
  obtain ⟨ls6, hGet6, hR3, hPc6, hStatus6⟩ :=
    AssignRegFullPost.assigned_dst step6.post_holds hj
  have hR2_6 : ls6.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignRegFullPost.preserves_reg_value step6.post_holds hj hGet5 hGet6 hR2
      (by decide)
  have hR6_6 : ls6.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignRegFullPost.preserves_reg_value step6.post_holds hj hGet5 hGet6 hR6
      (by decide)
  have hRd1_6 : ls6.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignRegFullPost.preserves_reg_value step6.post_holds hj hGet5 hGet6 hRd1
      (by decide)
  have hRd2_6 : ls6.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignRegFullPost.preserves_reg_value step6.post_holds hj hGet5 hGet6 hRd2
      (by decide)
  have hRd3_6 : ls6.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignRegFullPost.preserves_reg_value step6.post_holds hj hGet5 hGet6 hRd3
      (by decide)
  refine ⟨ls6, hGet6, hR2_6, hR6_6, hRd1_6, hRd2_6, hRd3_6, hR3, hPc6, hStatus6⟩

theorem saxpy_step6_ctaid_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      SaxpyAfterStep6Post n hn alpha st6 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, step6.step, ?_⟩
  exact saxpy_step6_ctaid_accumulated_record n hn hnpos alpha xs ys

theorem saxpy_step8_special_regs_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpySpecialRegsPost n hn alpha
      (saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys).post := by
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  have hStep6 := saxpy_step6_ctaid_accumulated_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls6, hGet6, hR2, hR6, hRd1, hRd2, hRd3, hR3, _hPc6, _hStatus6⟩ :=
    hStep6 j hj
  obtain ⟨ls7, hGet7, hR4, _hPc7, _hStatus7⟩ :=
    AssignRegFullPost.assigned_dst step7.post_holds hj
  have hR2_7 : ls7.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hR2
      (by decide)
  have hR6_7 : ls7.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hR6
      (by decide)
  have hRd1_7 : ls7.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hRd1
      (by decide)
  have hRd2_7 : ls7.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hRd2
      (by decide)
  have hRd3_7 : ls7.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hRd3
      (by decide)
  have hR3_7 : ls7.regs["r3"]? = some (.u32 0) :=
    AssignRegFullPost.preserves_reg_value step7.post_holds hj hGet6 hGet7 hR3
      (by decide)
  obtain ⟨ls8, hGet8, hR5, hPc8, hStatus8⟩ :=
    AssignRegFullPost.assigned_dst step8.post_holds hj
  have hR2_8 : ls8.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hR2_7
      (by decide)
  have hR6_8 : ls8.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hR6_7
      (by decide)
  have hRd1_8 : ls8.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hRd1_7
      (by decide)
  have hRd2_8 : ls8.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hRd2_7
      (by decide)
  have hRd3_8 : ls8.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hRd3_7
      (by decide)
  have hR3_8 : ls8.regs["r3"]? = some (.u32 0) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hR3_7
      (by decide)
  have hR4_8 : ls8.regs["r4"]? = some (.u32 32) :=
    AssignRegFullPost.preserves_reg_value step8.post_holds hj hGet7 hGet8 hR4
      (by decide)
  refine
    ⟨ls8, hGet8, hR2_8, hR6_8, hRd1_8, hRd2_8, hRd3_8, hR3_8, hR4_8, hR5,
      hPc8, hStatus8⟩

theorem saxpy_step8_special_regs_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      SaxpySpecialRegsPost n hn alpha st8 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step1.step, step2.step, step3.step, step4.step, step5.step, step6.step,
      step7.step, step8.step, ?_⟩
  exact saxpy_step8_special_regs_record n hn hnpos alpha xs ys

theorem evalRValue_saxpy_step8_post_mad_index
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    evalRValue? (saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys).post
        0 0 j saxpyMadIndexRhs =
      some (.s32 (Int.ofNat j.val)) := by
  have hPost := saxpy_step8_special_regs_record n hn hnpos alpha xs ys
  obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, hR3, hR4, hR5,
    _hPc, _hStatus⟩ := hPost j hj
  simp [saxpyMadIndexRhs, readReg, evalRValue?, evalUnary?, evalTernary?, hGet, hR3,
    hR4, hR5]
  fin_cases j <;> rfl

/-- Step-record form of `mad.lo.s32 %r1, %r3, %r4, %r5`. -/
noncomputable def saxpy_step9_mad_index_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step9_ctx n hn hnpos alpha xs ys)
      (AssignRegFullPost (saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r1" (fun lane => .s32 (Int.ofNat lane.val))
        ("saxpyKernel", 8)) :=
  BodyStepContext.assignRegStep (saxpy_step9_ctx n hn hnpos alpha xs ys)
    (dst := "r1")
    (rhs := saxpyMadIndexRhs)
    (guard? := none)
    (fun lane => .s32 (Int.ofNat lane.val))
    (fun _ hLane => evalRValue_saxpy_step8_post_mad_index n hn hnpos alpha xs ys hLane)

/-- The body-step context after step 9, at BB0 slot 9. -/
noncomputable def saxpy_step10_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step9_mad_index_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 9) saxpyBB0 saxpyBB0_setp_ge_index_len (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step9_mad_index_record n hn hnpos alpha xs ys)
    rfl rfl saxpyBB0_body9_setp_ge_index_len

theorem saxpy_step9_index_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyIndexPost n hn alpha
      (saxpy_step9_mad_index_record n hn hnpos alpha xs ys).post := by
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  have hSpecial := saxpy_step8_special_regs_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls8, hGet8, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5,
    _hPc8, _hStatus8⟩ := hSpecial j hj
  obtain ⟨ls9, hGet9, hR1, hPc9, hStatus9⟩ :=
    AssignRegFullPost.assigned_dst step9.post_holds hj
  have hR2_9 : ls9.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hR2
      (by decide)
  have hR6_9 : ls9.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hR6
      (by decide)
  have hRd1_9 : ls9.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hRd1
      (by decide)
  have hRd2_9 : ls9.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hRd2
      (by decide)
  have hRd3_9 : ls9.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hRd3
      (by decide)
  have hR3_9 : ls9.regs["r3"]? = some (.u32 0) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hR3
      (by decide)
  have hR4_9 : ls9.regs["r4"]? = some (.u32 32) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hR4
      (by decide)
  have hR5_9 : ls9.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    AssignRegFullPost.preserves_reg_value step9.post_holds hj hGet8 hGet9 hR5
      (by decide)
  refine
    ⟨ls9, hGet9, hR2_9, hR6_9, hRd1_9, hRd2_9, hRd3_9, hR3_9, hR4_9, hR5_9,
      hR1, hPc9, hStatus9⟩

theorem saxpy_step9_index_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      SaxpyIndexPost n hn alpha st9 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step1.step, step2.step, step3.step, step4.step, step5.step,
      step6.step, step7.step, step8.step, step9.step, ?_⟩
  exact saxpy_step9_index_record n hn hnpos alpha xs ys

theorem evalCmp_saxpy_step9_post_setp_ge_false
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    evalCmp? (saxpy_step9_mad_index_record n hn hnpos alpha xs ys).post
        0 0 j saxpySetpGeS32IndexLen =
      some false := by
  have hIndex := saxpy_step9_index_record n hn hnpos alpha xs ys
  obtain ⟨ls, hGet, hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
    hR1, _hPc, _hStatus⟩ := hIndex j hj
  exact evalCmp_saxpy_setp_ge_false hn (saxpyActiveLanes_mem_lt n hn hj) hGet hR1 hR2

/-- Step-record form of `setp.ge.s32 %p1, %r1, %r2`. -/
noncomputable def saxpy_step10_setp_ge_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step10_ctx n hn hnpos alpha xs ys)
      (AssignPredFullPost (saxpy_step9_mad_index_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "p1" (fun _ => false) ("saxpyKernel", 9)) :=
  BodyStepContext.assignPredStep (saxpy_step10_ctx n hn hnpos alpha xs ys)
    (dst := "p1")
    (cmp := saxpySetpGeS32IndexLen)
    (guard? := none)
    (fun _ => false)
    (fun _ hLane => evalCmp_saxpy_step9_post_setp_ge_false n hn hnpos alpha xs ys hLane)

theorem saxpy_step10_branch_pred_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyBranchPredPost n hn alpha
      (saxpy_step10_setp_ge_record n hn hnpos alpha xs ys).post := by
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  have hIndex := saxpy_step9_index_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls9, hGet9, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    _hPc9, _hStatus9⟩ := hIndex j hj
  obtain ⟨ls10, hGet10, hP1, hPc10, hStatus10⟩ :=
    AssignPredFullPost.assigned_dst step10.post_holds hj
  have hR2_10 : ls10.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR2
  have hR6_10 : ls10.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR6
  have hRd1_10 : ls10.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hRd1
  have hRd2_10 : ls10.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hRd2
  have hRd3_10 : ls10.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hRd3
  have hR3_10 : ls10.regs["r3"]? = some (.u32 0) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR3
  have hR4_10 : ls10.regs["r4"]? = some (.u32 32) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR4
  have hR5_10 : ls10.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR5
  have hR1_10 : ls10.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    AssignPredFullPost.preserves_reg_value step10.post_holds hj hGet9 hGet10 hR1
  refine
    ⟨ls10, hGet10, hR2_10, hR6_10, hRd1_10, hRd2_10, hRd3_10, hR3_10, hR4_10,
      hR5_10, hR1_10, hP1, hPc10, hStatus10⟩

theorem evalRValue_saxpy_step10_post_p1_false
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    evalRValue? (saxpy_step10_setp_ge_record n hn hnpos alpha xs ys).post
        0 0 j (.pred "p1") =
      some (.pred false) := by
  have hPost := saxpy_step10_branch_pred_record n hn hnpos alpha xs ys
  obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
    _hR1, hP1, _hPc, _hStatus⟩ := hPost j hj
  simp [evalRValue?, readPred, hGet, hP1]

/-- Terminator-step context for the BB0 conditional branch at PC 10. -/
noncomputable def saxpy_step11_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    TermStepContext (saxpy_step10_setp_ge_record n hn hnpos alpha xs ys).post
      ("saxpyKernel", 10) saxpyBB0
      (.cbr (.pred "p1") "$L__BB0_2" "saxpyKernel$fallthrough0")
      (saxpyActiveLanes n hn) := by
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  exact
    { warpState := step10.post_warpState
      wf := step10.post_wf
      getWarp := step10.post_getWarp
      warp_wf := step10.post_warp_wf
      lockstep := step10.post_lockstep
      currentPc := step10.post_currentPc
      block_lookup := by
        rw [step10.post_kernelEnv]
        exact (saxpy_step10_ctx n hn hnpos alpha xs ys).block_lookup
      body_done := saxpyBB0_body10_none
      block_term := by
        simpa [saxpyBB0_cbr_index_guard] using saxpyBB0_term_cbr_index_guard
      participants_eq := by
        have hTermParticipants :=
          termParticipantsFor_eq_runnable_of_lockstep step10.post_lockstep step10.post_currentPc
        rw [hTermParticipants, step10.post_runnable]
        exact BodyStepContext.runnable_eq_participants_of_none
          (saxpy_step10_ctx n hn hnpos alpha xs ys) rfl }

/-- Step-record form of BB0's conditional branch. Since `%p1` is uniformly
false on active lanes, the branch falls through to `saxpyKernel$fallthrough0`. -/
noncomputable def saxpy_step11_branch_fallthrough_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    TermStepRecord (saxpy_step11_ctx n hn hnpos alpha xs ys)
      (some ("saxpyKernel$fallthrough0", 0))
      (CbrStepFullPost (saxpy_step10_setp_ge_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) ("saxpyKernel$fallthrough0", 0)) :=
  TermStepContext.cbrFalseStep (saxpy_step11_ctx n hn hnpos alpha xs ys)
    (saxpyActiveLanes_ne_nil n hn hnpos)
    (fun _ hLane => evalRValue_saxpy_step10_post_p1_false n hn hnpos alpha xs ys hLane)

theorem saxpy_step10_branch_pred_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      SaxpyBranchPredPost n hn alpha st10 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step1.step, step2.step, step3.step, step4.step,
      step5.step, step6.step, step7.step, step8.step, step9.step, step10.step, ?_⟩
  exact saxpy_step10_branch_pred_record n hn hnpos alpha xs ys

theorem saxpy_step11_branch_target_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyBranchTargetPost n hn alpha
      (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post := by
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  have hPred := saxpy_step10_branch_pred_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls10, hGet10, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, _hPc10, hStatus10⟩ := hPred j hj
  obtain ⟨preLane, ls11, hPre, hGet11, hRegs, hPreds, _hLocal, hStatusFrame, hPc11⟩ :=
    step11.post_holds j hj
  have hPreEq : preLane = ls10 := by
    rw [hGet10] at hPre
    exact (Option.some.inj hPre).symm
  subst preLane
  have hR2_11 : ls11.regs["r2"]? = some (.u32 (UInt32.ofNat n)) := by
    rw [hRegs, hR2]
  have hR6_11 : ls11.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) := by
    rw [hRegs, hR6]
  have hRd1_11 : ls11.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) := by
    rw [hRegs, hRd1]
  have hRd2_11 : ls11.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) := by
    rw [hRegs, hRd2]
  have hRd3_11 : ls11.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) := by
    rw [hRegs, hRd3]
  have hR3_11 : ls11.regs["r3"]? = some (.u32 0) := by
    rw [hRegs, hR3]
  have hR4_11 : ls11.regs["r4"]? = some (.u32 32) := by
    rw [hRegs, hR4]
  have hR5_11 : ls11.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) := by
    rw [hRegs, hR5]
  have hR1_11 : ls11.regs["r1"]? = some (.s32 (Int.ofNat j.val)) := by
    rw [hRegs, hR1]
  have hP1_11 : ls11.preds["p1"]? = some false := by
    rw [hPreds, hP1]
  have hStatus11 : ls11.status = .running := by
    rw [hStatusFrame, hStatus10]
  refine
    ⟨ls11, hGet11, hR2_11, hR6_11, hRd1_11, hRd2_11, hRd3_11, hR3_11, hR4_11,
      hR5_11, hR1_11, hP1_11, hPc11, hStatus11⟩

theorem saxpy_step11_post_kernelEnv
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post.kernelEnv =
      (saxpyStateFor n alpha xs ys).kernelEnv :=
  (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post_kernelEnv.trans
    ((saxpy_step10_setp_ge_record n hn hnpos alpha xs ys).post_kernelEnv.trans
      ((saxpy_step9_mad_index_record n hn hnpos alpha xs ys).post_kernelEnv.trans
        ((saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys).post_kernelEnv.trans
          (saxpy_step7_post_kernelEnv n hn hnpos alpha xs ys))))

/-- Body-step context for the first fallthrough instruction. -/
noncomputable def saxpy_step12_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 0) saxpyFallthrough0 saxpyFallthrough0_cvta_rd4
      (saxpyActiveLanes n hn) :=
  TermStepRecord.nextBodyContextNone
    (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys)
    rfl
    (by
      rw [saxpy_step11_post_kernelEnv n hn hnpos alpha xs ys]
      exact saxpyStateFor_fallthrough0_lookup n alpha xs ys)
    saxpyFallthrough0_body0_cvta_rd4

theorem evalRValue_saxpy_step11_post_rd1
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) {j : LaneId}
    (hj : j ∈ saxpyActiveLanes n hn) :
    evalRValue? (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post
        0 0 j (.reg "rd1") =
      some (.u64 (UInt64.ofNat saxpyXBase)) := by
  have hPost := saxpy_step11_branch_target_record n hn hnpos alpha xs ys
  obtain ⟨ls, hGet, _hR2, _hR6, hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
    _hR1, _hP1, _hPc, _hStatus⟩ := hPost j hj
  simp [evalRValue?, readReg, hGet, hRd1]

/-- Step-record form of `cvta.to.global.u64 %rd4, %rd1`. -/
noncomputable def saxpy_step12_cvta_rd4_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step12_ctx n hn hnpos alpha xs ys)
      (CvtaFullPost (saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "rd4" (fun _ => .gaddr .global saxpyXBase)
        ("saxpyKernel$fallthrough0", 0)) :=
  BodyStepContext.cvtaStep (saxpy_step12_ctx n hn hnpos alpha xs ys)
    (dst := "rd4")
    (space := .global)
    (src := .reg "rd1")
    (guard? := none)
    (fun _ => .gaddr .global saxpyXBase)
    (fun lane hLane =>
      ⟨.u64 (UInt64.ofNat saxpyXBase),
        evalRValue_saxpy_step11_post_rd1 n hn hnpos alpha xs ys hLane,
        by simp [saxpyXBase]⟩)

theorem saxpy_step12_cvta_rd4_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterCvtaRd4Post n hn alpha
      (saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys).post := by
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  have hBranch := saxpy_step11_branch_target_record n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls11, hGet11, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, _hPc11, _hStatus11⟩ := hBranch j hj
  obtain ⟨ls12, hGet12, hRd4, hPc12, hStatus12⟩ :=
    CvtaFullPost.converted_dst step12.post_holds hj
  have hR2_12 : ls12.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR2
      (by decide)
  have hR6_12 : ls12.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR6
      (by decide)
  have hRd1_12 : ls12.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hRd1
      (by decide)
  have hRd2_12 : ls12.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hRd2
      (by decide)
  have hRd3_12 : ls12.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hRd3
      (by decide)
  have hR3_12 : ls12.regs["r3"]? = some (.u32 0) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR3
      (by decide)
  have hR4_12 : ls12.regs["r4"]? = some (.u32 32) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR4
      (by decide)
  have hR5_12 : ls12.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR5
      (by decide)
  have hR1_12 : ls12.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step12.post_holds hj hGet11 hGet12 hR1
      (by decide)
  have hP1_12 : ls12.preds["p1"]? = some false :=
    CvtaFullPost.preserves_pred_value step12.post_holds hj hGet11 hGet12 hP1
  refine
    ⟨ls12, hGet12, hR2_12, hR6_12, hRd1_12, hRd2_12, hRd3_12, hR3_12, hR4_12,
      hR5_12, hR1_12, hP1_12, hRd4, hPc12, hStatus12⟩

/-- Body-step context for `mul.wide.s32 %rd5, %r1, 4`. -/
noncomputable def saxpy_step13_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 1) saxpyFallthrough0 saxpyFallthrough0_mul_wide_index
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys)
    rfl rfl saxpyFallthrough0_body1_mul_wide_index

/-- Step-record form of `mul.wide.s32 %rd5, %r1, 4`. -/
noncomputable def saxpy_step13_mul_wide_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step13_ctx n hn hnpos alpha xs ys) (fun _ => True) :=
  BodyStepContext.assignRegStepSome (saxpy_step13_ctx n hn hnpos alpha xs ys)
    (fun lane hLane => by
      have hPost := saxpy_step12_cvta_rd4_record_post n hn hnpos alpha xs ys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        hR1, _hP1, _hRd4, _hPc, _hStatus⟩ := hPost lane hLane
      refine ⟨saxpyLaneByteOffsetValue lane, ?_⟩
      exact evalRValue_saxpyIndexByteOffsetRhs_of_r1 hGet hR1)

theorem saxpy_step13_mul_wide_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterMulWidePost n hn alpha
      (saxpy_step13_mul_wide_record n hn hnpos alpha xs ys).post := by
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  have hStep12 := saxpy_step12_cvta_rd4_record_post n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls12, hGet12, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hPc12, hStatus12⟩ := hStep12 j hj
  have hEval :
      evalRValue? (saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys).post
          0 0 j saxpyIndexByteOffsetRhs =
        some (saxpyLaneByteOffsetValue j) := by
    exact evalRValue_saxpyIndexByteOffsetRhs_of_r1 hGet12 hR1
  obtain ⟨ls13, hGet13, hRegs13, hPreds13, _hLocal13, hStatusFrame, hPc13Raw⟩ :=
    stepInstr?_assignReg_lane_full
      (dst := "rd5") (rhs := saxpyIndexByteOffsetRhs) (guard? := none)
      (saxpy_step13_ctx n hn hnpos alpha xs ys).wf
      (saxpy_step13_ctx n hn hnpos alpha xs ys).getWarp
      (saxpy_step13_ctx n hn hnpos alpha xs ys).lockstep
      (saxpy_step13_ctx n hn hnpos alpha xs ys).currentPc
      (saxpy_step13_ctx n hn hnpos alpha xs ys).participants_eq
      hj hGet12 hPc12 hEval step13.instrStep
  have hRegFrame :
      ∀ {r : RegName} {v : Value}, ls12.regs[r]? = some v → r ≠ "rd5" →
        ls13.regs[r]? = some v := by
    intro r v hReg hNe
    rw [hRegs13, Std.HashMap.getElem?_insert]
    by_cases hEqKey : "rd5" = r
    · exact False.elim (hNe hEqKey.symm)
    · simp [beq_iff_eq, hEqKey, hReg]
  have hR2_13 : ls13.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    hRegFrame hR2 (by decide)
  have hR6_13 : ls13.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    hRegFrame hR6 (by decide)
  have hRd1_13 : ls13.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    hRegFrame hRd1 (by decide)
  have hRd2_13 : ls13.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    hRegFrame hRd2 (by decide)
  have hRd3_13 : ls13.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    hRegFrame hRd3 (by decide)
  have hR3_13 : ls13.regs["r3"]? = some (.u32 0) :=
    hRegFrame hR3 (by decide)
  have hR4_13 : ls13.regs["r4"]? = some (.u32 32) :=
    hRegFrame hR4 (by decide)
  have hR5_13 : ls13.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    hRegFrame hR5 (by decide)
  have hR1_13 : ls13.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    hRegFrame hR1 (by decide)
  have hP1_13 : ls13.preds["p1"]? = some false := by
    rw [hPreds13, hP1]
  have hRd4_13 : ls13.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    hRegFrame hRd4 (by decide)
  have hRd5 : ls13.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) := by
    rw [hRegs13]
    simp [Std.HashMap.getElem?_insert]
  have hPc13 : ls13.pc = ("saxpyKernel$fallthrough0", 2) := by
    simpa using hPc13Raw
  have hStatus13 : ls13.status = .running := by
    rw [hStatusFrame, hStatus12]
  refine
    ⟨ls13, hGet13, hR2_13, hR6_13, hRd1_13, hRd2_13, hRd3_13, hR3_13, hR4_13,
      hR5_13, hR1_13, hP1_13, hRd4_13, hRd5, hPc13, hStatus13⟩

/-- Body-step context for `add.s64 %rd6, %rd4, %rd5`. -/
noncomputable def saxpy_step14_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step13_mul_wide_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 2) saxpyFallthrough0 saxpyFallthrough0_add_x_addr
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step13_mul_wide_record n hn hnpos alpha xs ys)
    rfl rfl saxpyFallthrough0_body2_add_x_addr

/-- Step-record form of `add.s64 %rd6, %rd4, %rd5`. -/
noncomputable def saxpy_step14_add_x_addr_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step14_ctx n hn hnpos alpha xs ys) (fun _ => True) :=
  BodyStepContext.assignRegStepSome (saxpy_step14_ctx n hn hnpos alpha xs ys)
    (fun lane hLane => by
      have hPost := saxpy_step13_mul_wide_record_post n hn hnpos alpha xs ys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, hRd4, hRd5, _hPc, _hStatus⟩ := hPost lane hLane
      refine ⟨saxpyXElementAddrValue lane, ?_⟩
      exact evalRValue_saxpyXElementAddrRhs_of_regs hGet hRd4 hRd5)

theorem saxpy_step14_add_x_addr_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterXAddrPost n hn alpha
      (saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys).post := by
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  have hStep13 := saxpy_step13_mul_wide_record_post n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls13, hGet13, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hPc13, hStatus13⟩ := hStep13 j hj
  have hEval :
      evalRValue? (saxpy_step13_mul_wide_record n hn hnpos alpha xs ys).post
          0 0 j saxpyXElementAddrRhs =
        some (saxpyXElementAddrValue j) := by
    exact evalRValue_saxpyXElementAddrRhs_of_regs hGet13 hRd4 hRd5
  obtain ⟨ls14, hGet14, hRegs14, hPreds14, _hLocal14, hStatusFrame, hPc14Raw⟩ :=
    BodyStepRecord.assignReg_lane_full step14 hj hGet13 hPc13 hEval
  have hRegFrame :
      ∀ {r : RegName} {v : Value}, ls13.regs[r]? = some v → r ≠ "rd6" →
        ls14.regs[r]? = some v := by
    intro r v hReg hNe
    rw [hRegs14]
    exact reg_insert_getElem?_ne hReg hNe
  have hR2_14 : ls14.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    hRegFrame hR2 (by decide)
  have hR6_14 : ls14.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    hRegFrame hR6 (by decide)
  have hRd1_14 : ls14.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    hRegFrame hRd1 (by decide)
  have hRd2_14 : ls14.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    hRegFrame hRd2 (by decide)
  have hRd3_14 : ls14.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    hRegFrame hRd3 (by decide)
  have hR3_14 : ls14.regs["r3"]? = some (.u32 0) :=
    hRegFrame hR3 (by decide)
  have hR4_14 : ls14.regs["r4"]? = some (.u32 32) :=
    hRegFrame hR4 (by decide)
  have hR5_14 : ls14.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    hRegFrame hR5 (by decide)
  have hR1_14 : ls14.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    hRegFrame hR1 (by decide)
  have hP1_14 : ls14.preds["p1"]? = some false := by
    rw [hPreds14, hP1]
  have hRd4_14 : ls14.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    hRegFrame hRd4 (by decide)
  have hRd5_14 : ls14.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    hRegFrame hRd5 (by decide)
  have hRd6 : ls14.regs["rd6"]? = some (saxpyXElementAddrValue j) := by
    rw [hRegs14]
    simp [Std.HashMap.getElem?_insert]
  have hPc14 : ls14.pc = ("saxpyKernel$fallthrough0", 3) := by
    simpa using hPc14Raw
  have hStatus14 : ls14.status = .running := by
    rw [hStatusFrame, hStatus13]
  refine
    ⟨ls14, hGet14, hR2_14, hR6_14, hRd1_14, hRd2_14, hRd3_14, hR3_14, hR4_14,
      hR5_14, hR1_14, hP1_14, hRd4_14, hRd5_14, hRd6, hPc14, hStatus14⟩

/-- Body-step context for `cvta.to.global.u64 %rd7, %rd2`. -/
noncomputable def saxpy_step15_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 3) saxpyFallthrough0 saxpyFallthrough0_cvta_rd7
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys)
    rfl rfl saxpyFallthrough0_body3_cvta_rd7

/-- Step-record form of `cvta.to.global.u64 %rd7, %rd2`. -/
noncomputable def saxpy_step15_cvta_rd7_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step15_ctx n hn hnpos alpha xs ys)
      (CvtaFullPost (saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "rd7" (fun _ => .gaddr .global saxpyYBase)
        ("saxpyKernel$fallthrough0", 3)) :=
  BodyStepContext.cvtaStep (saxpy_step15_ctx n hn hnpos alpha xs ys)
    (dst := "rd7")
    (space := .global)
    (src := .reg "rd2")
    (guard? := none)
    (fun _ => .gaddr .global saxpyYBase)
    (fun lane hLane => by
      have hPost := saxpy_step14_add_x_addr_record_post n hn hnpos alpha xs ys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, _hRd5, _hRd6, _hPc, _hStatus⟩ := hPost lane hLane
      refine ⟨.u64 (UInt64.ofNat saxpyYBase), ?_, ?_⟩
      · simp [evalRValue?, readReg, hGet, hRd2]
      · simp [saxpyYBase])

theorem saxpy_step15_cvta_rd7_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterCvtaRd7Post n hn alpha
      (saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys).post := by
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  have hStep14 := saxpy_step14_add_x_addr_record_post n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls14, hGet14, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, _hPc14, _hStatus14⟩ := hStep14 j hj
  obtain ⟨ls15, hGet15, hRd7, hPc15, hStatus15⟩ :=
    CvtaFullPost.converted_dst step15.post_holds hj
  have hR2_15 : ls15.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR2
      (by decide)
  have hR6_15 : ls15.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR6
      (by decide)
  have hRd1_15 : ls15.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd1
      (by decide)
  have hRd2_15 : ls15.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd2
      (by decide)
  have hRd3_15 : ls15.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd3
      (by decide)
  have hR3_15 : ls15.regs["r3"]? = some (.u32 0) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR3
      (by decide)
  have hR4_15 : ls15.regs["r4"]? = some (.u32 32) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR4
      (by decide)
  have hR5_15 : ls15.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR5
      (by decide)
  have hR1_15 : ls15.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hR1
      (by decide)
  have hP1_15 : ls15.preds["p1"]? = some false :=
    CvtaFullPost.preserves_pred_value step15.post_holds hj hGet14 hGet15 hP1
  have hRd4_15 : ls15.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd4
      (by decide)
  have hRd5_15 : ls15.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd5
      (by decide)
  have hRd6_15 : ls15.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    CvtaFullPost.preserves_reg_value step15.post_holds hj hGet14 hGet15 hRd6
      (by decide)
  refine
    ⟨ls15, hGet15, hR2_15, hR6_15, hRd1_15, hRd2_15, hRd3_15, hR3_15, hR4_15,
      hR5_15, hR1_15, hP1_15, hRd4_15, hRd5_15, hRd6_15, hRd7, hPc15,
      hStatus15⟩

/-- Body-step context for `add.s64 %rd8, %rd7, %rd5`. -/
noncomputable def saxpy_step16_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 4) saxpyFallthrough0 saxpyFallthrough0_add_y_addr
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys)
    rfl rfl saxpyFallthrough0_body4_add_y_addr

/-- Step-record form of `add.s64 %rd8, %rd7, %rd5`. -/
noncomputable def saxpy_step16_add_y_addr_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepRecord (saxpy_step16_ctx n hn hnpos alpha xs ys) (fun _ => True) :=
  BodyStepContext.assignRegStepSome (saxpy_step16_ctx n hn hnpos alpha xs ys)
    (fun lane hLane => by
      have hPost := saxpy_step15_cvta_rd7_record_post n hn hnpos alpha xs ys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, hRd5, _hRd6, hRd7, _hPc, _hStatus⟩ := hPost lane hLane
      refine ⟨saxpyYElementAddrValue lane, ?_⟩
      exact evalRValue_saxpyYElementAddrRhs_of_regs hGet hRd7 hRd5)

theorem saxpy_step16_add_y_addr_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    SaxpyAfterYAddrPost n hn alpha
      (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys).post := by
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  have hStep15 := saxpy_step15_cvta_rd7_record_post n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls15, hGet15, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, hRd7, hPc15, hStatus15⟩ := hStep15 j hj
  have hEval :
      evalRValue? (saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys).post
          0 0 j saxpyYElementAddrRhs =
        some (saxpyYElementAddrValue j) := by
    exact evalRValue_saxpyYElementAddrRhs_of_regs hGet15 hRd7 hRd5
  obtain ⟨ls16, hGet16, hRegs16, hPreds16, _hLocal16, hStatusFrame, hPc16Raw⟩ :=
    BodyStepRecord.assignReg_lane_full step16 hj hGet15 hPc15 hEval
  have hRegFrame :
      ∀ {r : RegName} {v : Value}, ls15.regs[r]? = some v → r ≠ "rd8" →
        ls16.regs[r]? = some v := by
    intro r v hReg hNe
    rw [hRegs16]
    exact reg_insert_getElem?_ne hReg hNe
  have hR2_16 : ls16.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    hRegFrame hR2 (by decide)
  have hR6_16 : ls16.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    hRegFrame hR6 (by decide)
  have hRd1_16 : ls16.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    hRegFrame hRd1 (by decide)
  have hRd2_16 : ls16.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    hRegFrame hRd2 (by decide)
  have hRd3_16 : ls16.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    hRegFrame hRd3 (by decide)
  have hR3_16 : ls16.regs["r3"]? = some (.u32 0) :=
    hRegFrame hR3 (by decide)
  have hR4_16 : ls16.regs["r4"]? = some (.u32 32) :=
    hRegFrame hR4 (by decide)
  have hR5_16 : ls16.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    hRegFrame hR5 (by decide)
  have hR1_16 : ls16.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    hRegFrame hR1 (by decide)
  have hP1_16 : ls16.preds["p1"]? = some false := by
    rw [hPreds16, hP1]
  have hRd4_16 : ls16.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    hRegFrame hRd4 (by decide)
  have hRd5_16 : ls16.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    hRegFrame hRd5 (by decide)
  have hRd6_16 : ls16.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    hRegFrame hRd6 (by decide)
  have hRd7_16 : ls16.regs["rd7"]? = some (.gaddr .global saxpyYBase) :=
    hRegFrame hRd7 (by decide)
  have hRd8 : ls16.regs["rd8"]? = some (saxpyYElementAddrValue j) := by
    rw [hRegs16]
    simp [Std.HashMap.getElem?_insert]
  have hPc16 : ls16.pc = ("saxpyKernel$fallthrough0", 5) := by
    simpa using hPc16Raw
  have hStatus16 : ls16.status = .running := by
    rw [hStatusFrame, hStatus15]
  refine
    ⟨ls16, hGet16, hR2_16, hR6_16, hRd1_16, hRd2_16, hRd3_16, hR3_16, hR4_16,
      hR5_16, hR1_16, hP1_16, hRd4_16, hRd5_16, hRd6_16, hRd7_16, hRd8, hPc16,
      hStatus16⟩

theorem saxpy_step16_add_y_addr_record_post_global
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys).post.global =
      (saxpyStateFor n alpha xs ys).global := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  calc
    step16.post.global = step15.post.global := step16.post_global
    _ = step14.post.global := step15.post_global
    _ = step13.post.global := step14.post_global
    _ = step12.post.global := step13.post_global
    _ = step11.post.global := step12.post_global
    _ = step10.post.global := step11.post_global
    _ = step9.post.global := step10.post_global
    _ = step8.post.global := step9.post_global
    _ = step7.post.global := step8.post_global
    _ = step6.post.global := step7.post_global
    _ = step5.post.global := step6.post_global
    _ = step4.post.global := step5.post_global
    _ = step3.post.global := step4.post_global
    _ = step2.post.global := step3.post_global
    _ = step1.post.global := step2.post_global
    _ = (saxpyStateFor n alpha xs ys).global := step1.post_global

/-- Body-step context for `ld.global.s32 %r7, [%rd6]`. -/
noncomputable def saxpy_step17_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    BodyStepContext (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys).post
      ("saxpyKernel$fallthrough0", 5) saxpyFallthrough0 saxpyFallthrough0_load_x
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys)
    rfl rfl saxpyFallthrough0_body5_load_x

/-- Step-record form of `ld.global.s32 %r7, [%rd6]`. -/
noncomputable def saxpy_step17_load_x_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) :
    BodyStepRecord (saxpy_step17_ctx n hn hnpos alpha xs ys)
      (LoadStepFullPost (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys).post
        (saxpyActiveLanes n hn) "r7" (saxpyLoadedS32Value xs)
        ("saxpyKernel$fallthrough0", 5)) :=
  BodyStepContext.loadStep (saxpy_step17_ctx n hn hnpos alpha xs ys)
    (dst := "r7")
    (src := saxpyLoadXSrc)
    (guard? := none)
    (saxpyLoadedS32Value xs)
    (fun lane hLane => by
      have hPost := saxpy_step16_add_y_addr_record_post n hn hnpos alpha xs ys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, _hRd5, hRd6, _hRd7, _hRd8, _hPc, _hStatus⟩ :=
        hPost lane hLane
      refine ⟨.global (saxpyXBase + lane.val * 4), ?_, ?_⟩
      · exact resolveAddr_saxpyLoadXSrc_of_rd6 hGet hRd6
      · have hReadInit := readMem_saxpyStateFor_global_x n hn alpha xs ys hxs hLane
        have hGlobal := saxpy_step16_add_y_addr_record_post_global n hn hnpos alpha xs ys
        have hReadStep :
            readMem? (saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys).post
                .global .s32 (.global (saxpyXBase + lane.val * 4)) =
              readMem? (saxpyStateFor n alpha xs ys) .global .s32
                (.global (saxpyXBase + lane.val * 4)) :=
          readMem?_global_congr hGlobal
        simpa [saxpyLoadXSrc] using hReadStep.trans hReadInit)

theorem saxpy_step17_load_x_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) :
    SaxpyAfterXLoadPost n hn alpha xs
      (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs).post := by
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  have hStep16 := saxpy_step16_add_y_addr_record_post n hn hnpos alpha xs ys
  intro j hj
  obtain ⟨ls16, hGet16, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, hRd7, hRd8, _hPc16, _hStatus16⟩ := hStep16 j hj
  obtain ⟨ls17, hGet17, hR7, hPc17, hStatus17⟩ :=
    LoadStepFullPost.loaded_dst step17.post_holds hj
  have hR2_17 : ls17.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR2
      (by decide)
  have hR6_17 : ls17.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR6
      (by decide)
  have hRd1_17 : ls17.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd1
      (by decide)
  have hRd2_17 : ls17.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd2
      (by decide)
  have hRd3_17 : ls17.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd3
      (by decide)
  have hR3_17 : ls17.regs["r3"]? = some (.u32 0) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR3
      (by decide)
  have hR4_17 : ls17.regs["r4"]? = some (.u32 32) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR4
      (by decide)
  have hR5_17 : ls17.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR5
      (by decide)
  have hR1_17 : ls17.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hR1
      (by decide)
  have hP1_17 : ls17.preds["p1"]? = some false :=
    LoadStepFullPost.preserves_pred_value step17.post_holds hj hGet16 hGet17 hP1
  have hRd4_17 : ls17.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd4
      (by decide)
  have hRd5_17 : ls17.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd5
      (by decide)
  have hRd6_17 : ls17.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd6
      (by decide)
  have hRd7_17 : ls17.regs["rd7"]? = some (.gaddr .global saxpyYBase) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd7
      (by decide)
  have hRd8_17 : ls17.regs["rd8"]? = some (saxpyYElementAddrValue j) :=
    LoadStepFullPost.preserves_reg_value step17.post_holds hj hGet16 hGet17 hRd8
      (by decide)
  refine
    ⟨ls17, hGet17, hR2_17, hR6_17, hRd1_17, hRd2_17, hRd3_17, hR3_17,
      hR4_17, hR5_17, hR1_17, hP1_17, hRd4_17, hRd5_17, hRd6_17, hRd7_17,
      hRd8_17, hR7, hPc17, hStatus17⟩

theorem saxpy_step17_load_x_record_post_global
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) :
    (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs).post.global =
      (saxpyStateFor n alpha xs ys).global := by
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  calc
    step17.post.global = step16.post.global := step17.post_global
    _ = (saxpyStateFor n alpha xs ys).global :=
      saxpy_step16_add_y_addr_record_post_global n hn hnpos alpha xs ys

/-- Body-step context for `ld.global.s32 %r8, [%rd8]`. -/
noncomputable def saxpy_step18_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) :
    BodyStepContext (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs).post
      ("saxpyKernel$fallthrough0", 6) saxpyFallthrough0 saxpyFallthrough0_load_y
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs)
    rfl rfl saxpyFallthrough0_body6_load_y

/-- Step-record form of `ld.global.s32 %r8, [%rd8]`. -/
noncomputable def saxpy_step18_load_y_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepRecord (saxpy_step18_ctx n hn hnpos alpha xs ys hxs)
      (LoadStepFullPost (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs).post
        (saxpyActiveLanes n hn) "r8" (saxpyLoadedS32Value ys)
        ("saxpyKernel$fallthrough0", 6)) :=
  BodyStepContext.loadStep (saxpy_step18_ctx n hn hnpos alpha xs ys hxs)
    (dst := "r8")
    (src := saxpyLoadYSrc)
    (guard? := none)
    (saxpyLoadedS32Value ys)
    (fun lane hLane => by
      have hPost := saxpy_step17_load_x_record_post n hn hnpos alpha xs ys hxs
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, _hRd5, _hRd6, _hRd7, hRd8, _hR7, _hPc, _hStatus⟩ :=
        hPost lane hLane
      refine ⟨.global (saxpyYBase + lane.val * 4), ?_, ?_⟩
      · exact resolveAddr_saxpyLoadYSrc_of_rd8 hGet hRd8
      · have hReadInit := readMem_saxpyStateFor_global_y n hn alpha xs ys hys hLane
        have hGlobal := saxpy_step17_load_x_record_post_global n hn hnpos alpha xs ys hxs
        have hReadStep :
            readMem? (saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs).post
                .global .s32 (.global (saxpyYBase + lane.val * 4)) =
              readMem? (saxpyStateFor n alpha xs ys) .global .s32
                (.global (saxpyYBase + lane.val * 4)) :=
          readMem?_global_congr hGlobal
        simpa [saxpyLoadYSrc] using hReadStep.trans hReadInit)

theorem saxpy_step18_load_y_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    SaxpyAfterYLoadPost n hn alpha xs ys
      (saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys).post := by
  let step18 := saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys
  have hStep17 := saxpy_step17_load_x_record_post n hn hnpos alpha xs ys hxs
  intro j hj
  obtain ⟨ls17, hGet17, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, hRd7, hRd8, hR7, _hPc17, _hStatus17⟩ := hStep17 j hj
  obtain ⟨ls18, hGet18, hR8, hPc18, hStatus18⟩ :=
    LoadStepFullPost.loaded_dst step18.post_holds hj
  have hR2_18 : ls18.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR2
      (by decide)
  have hR6_18 : ls18.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR6
      (by decide)
  have hRd1_18 : ls18.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd1
      (by decide)
  have hRd2_18 : ls18.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd2
      (by decide)
  have hRd3_18 : ls18.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd3
      (by decide)
  have hR3_18 : ls18.regs["r3"]? = some (.u32 0) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR3
      (by decide)
  have hR4_18 : ls18.regs["r4"]? = some (.u32 32) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR4
      (by decide)
  have hR5_18 : ls18.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR5
      (by decide)
  have hR1_18 : ls18.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR1
      (by decide)
  have hP1_18 : ls18.preds["p1"]? = some false :=
    LoadStepFullPost.preserves_pred_value step18.post_holds hj hGet17 hGet18 hP1
  have hRd4_18 : ls18.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd4
      (by decide)
  have hRd5_18 : ls18.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd5
      (by decide)
  have hRd6_18 : ls18.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd6
      (by decide)
  have hRd7_18 : ls18.regs["rd7"]? = some (.gaddr .global saxpyYBase) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd7
      (by decide)
  have hRd8_18 : ls18.regs["rd8"]? = some (saxpyYElementAddrValue j) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hRd8
      (by decide)
  have hR7_18 : ls18.regs["r7"]? = some (saxpyLoadedS32Value xs j) :=
    LoadStepFullPost.preserves_reg_value step18.post_holds hj hGet17 hGet18 hR7
      (by decide)
  refine
    ⟨ls18, hGet18, hR2_18, hR6_18, hRd1_18, hRd2_18, hRd3_18, hR3_18,
      hR4_18, hR5_18, hR1_18, hP1_18, hRd4_18, hRd5_18, hRd6_18, hRd7_18,
      hRd8_18, hR7_18, hR8, hPc18, hStatus18⟩

theorem evalRValue_saxpy_step18_post_mul_add
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n)
    {j : LaneId} (hj : j ∈ saxpyActiveLanes n hn) :
    evalRValue? (saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys).post
        0 0 j saxpyMulAddRhs =
      some (saxpyMulAddValue alpha xs ys j) := by
  have hPost := saxpy_step18_load_y_record_post n hn hnpos alpha xs ys hxs hys
  obtain ⟨ls, hGet, _hR2, hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
    _hR1, _hP1, _hRd4, _hRd5, _hRd6, _hRd7, _hRd8, hR7, hR8, _hPc, _hStatus⟩ :=
    hPost j hj
  have hCvt :
      evalUnary? (.cvt .s32) (.s32 (saxpyAlphaS32 alpha)) =
        some (.s32 (saxpyAlphaS32 alpha)) := by
    simp [evalUnary?, Typing.valueType?]
  simp [saxpyMulAddRhs, saxpyMulAddValue, saxpyLoadedS32Value, evalRValue?, readReg,
    evalTernary?, hGet, hR7, hR6, hR8, hCvt]

/-- Body-step context for `mad.lo.s32 %r9, %r7, %r6, %r8`. -/
noncomputable def saxpy_step19_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepContext (saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys).post
      ("saxpyKernel$fallthrough0", 7) saxpyFallthrough0 saxpyFallthrough0_mul_add
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys)
    rfl rfl saxpyFallthrough0_body7_mul_add

/-- Step-record form of `mad.lo.s32 %r9, %r7, %r6, %r8`. -/
noncomputable def saxpy_step19_mul_add_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepRecord (saxpy_step19_ctx n hn hnpos alpha xs ys hxs hys)
      (AssignRegFullPost (saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys).post
        (saxpyActiveLanes n hn) "r9" (saxpyMulAddValue alpha xs ys)
        ("saxpyKernel$fallthrough0", 7)) :=
  BodyStepContext.assignRegStep (saxpy_step19_ctx n hn hnpos alpha xs ys hxs hys)
    (dst := "r9")
    (rhs := saxpyMulAddRhs)
    (guard? := none)
    (saxpyMulAddValue alpha xs ys)
    (fun _ hLane =>
      evalRValue_saxpy_step18_post_mul_add n hn hnpos alpha xs ys hxs hys hLane)

theorem saxpy_step19_mul_add_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    SaxpyAfterMulAddPost n hn alpha xs ys
      (saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys).post := by
  let step19 := saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys
  have hStep18 := saxpy_step18_load_y_record_post n hn hnpos alpha xs ys hxs hys
  intro j hj
  obtain ⟨ls18, hGet18, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, hRd7, hRd8, hR7, hR8, _hPc18, _hStatus18⟩ :=
    hStep18 j hj
  obtain ⟨ls19, hGet19, hR9, hPc19, hStatus19⟩ :=
    AssignRegFullPost.assigned_dst step19.post_holds hj
  have hR2_19 : ls19.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR2
      (by decide)
  have hR6_19 : ls19.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR6
      (by decide)
  have hRd1_19 : ls19.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd1
      (by decide)
  have hRd2_19 : ls19.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd2
      (by decide)
  have hRd3_19 : ls19.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd3
      (by decide)
  have hR3_19 : ls19.regs["r3"]? = some (.u32 0) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR3
      (by decide)
  have hR4_19 : ls19.regs["r4"]? = some (.u32 32) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR4
      (by decide)
  have hR5_19 : ls19.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR5
      (by decide)
  have hR1_19 : ls19.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR1
      (by decide)
  have hP1_19 : ls19.preds["p1"]? = some false :=
    AssignRegFullPost.preserves_pred_value step19.post_holds hj hGet18 hGet19 hP1
  have hRd4_19 : ls19.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd4
      (by decide)
  have hRd5_19 : ls19.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd5
      (by decide)
  have hRd6_19 : ls19.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd6
      (by decide)
  have hRd7_19 : ls19.regs["rd7"]? = some (.gaddr .global saxpyYBase) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd7
      (by decide)
  have hRd8_19 : ls19.regs["rd8"]? = some (saxpyYElementAddrValue j) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hRd8
      (by decide)
  have hR7_19 : ls19.regs["r7"]? = some (saxpyLoadedS32Value xs j) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR7
      (by decide)
  have hR8_19 : ls19.regs["r8"]? = some (saxpyLoadedS32Value ys j) :=
    AssignRegFullPost.preserves_reg_value step19.post_holds hj hGet18 hGet19 hR8
      (by decide)
  refine
    ⟨ls19, hGet19, hR2_19, hR6_19, hRd1_19, hRd2_19, hRd3_19, hR3_19,
      hR4_19, hR5_19, hR1_19, hP1_19, hRd4_19, hRd5_19, hRd6_19, hRd7_19,
      hRd8_19, hR7_19, hR8_19, hR9, hPc19, hStatus19⟩

/-- Body-step context for `cvta.to.global.u64 %rd9, %rd3`. -/
noncomputable def saxpy_step20_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepContext (saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys).post
      ("saxpyKernel$fallthrough0", 8) saxpyFallthrough0 saxpyFallthrough0_cvta_rd9
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys)
    rfl rfl saxpyFallthrough0_body8_cvta_rd9

/-- Step-record form of `cvta.to.global.u64 %rd9, %rd3`. -/
noncomputable def saxpy_step20_cvta_rd9_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepRecord (saxpy_step20_ctx n hn hnpos alpha xs ys hxs hys)
      (CvtaFullPost (saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys).post
        (saxpyActiveLanes n hn) "rd9" (fun _ => .gaddr .global saxpyRBase)
        ("saxpyKernel$fallthrough0", 8)) :=
  BodyStepContext.cvtaStep (saxpy_step20_ctx n hn hnpos alpha xs ys hxs hys)
    (dst := "rd9")
    (space := .global)
    (src := .reg "rd3")
    (guard? := none)
    (fun _ => .gaddr .global saxpyRBase)
    (fun lane hLane => by
      have hPost := saxpy_step19_mul_add_record_post n hn hnpos alpha xs ys hxs hys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, _hRd5, _hRd6, _hRd7, _hRd8, _hR7, _hR8, _hR9,
        _hPc, _hStatus⟩ := hPost lane hLane
      refine ⟨.u64 (UInt64.ofNat saxpyRBase), ?_, ?_⟩
      · simp [evalRValue?, readReg, hGet, hRd3]
      · simp [saxpyRBase])

theorem saxpy_step20_cvta_rd9_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    SaxpyAfterCvtaRd9Post n hn alpha xs ys
      (saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys).post := by
  let step20 := saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys
  have hStep19 := saxpy_step19_mul_add_record_post n hn hnpos alpha xs ys hxs hys
  intro j hj
  obtain ⟨ls19, hGet19, hR2, hR6, hRd1, hRd2, hRd3, hR3, hR4, hR5, hR1,
    hP1, hRd4, hRd5, hRd6, hRd7, hRd8, hR7, hR8, hR9, _hPc19, _hStatus19⟩ :=
    hStep19 j hj
  obtain ⟨ls20, hGet20, hRd9, hPc20, hStatus20⟩ :=
    CvtaFullPost.converted_dst step20.post_holds hj
  have hR2_20 : ls20.regs["r2"]? = some (.u32 (UInt32.ofNat n)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR2
      (by decide)
  have hR6_20 : ls20.regs["r6"]? = some (.s32 (saxpyAlphaS32 alpha)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR6
      (by decide)
  have hRd1_20 : ls20.regs["rd1"]? = some (.u64 (UInt64.ofNat saxpyXBase)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd1
      (by decide)
  have hRd2_20 : ls20.regs["rd2"]? = some (.u64 (UInt64.ofNat saxpyYBase)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd2
      (by decide)
  have hRd3_20 : ls20.regs["rd3"]? = some (.u64 (UInt64.ofNat saxpyRBase)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd3
      (by decide)
  have hR3_20 : ls20.regs["r3"]? = some (.u32 0) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR3
      (by decide)
  have hR4_20 : ls20.regs["r4"]? = some (.u32 32) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR4
      (by decide)
  have hR5_20 : ls20.regs["r5"]? = some (.u32 (UInt32.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR5
      (by decide)
  have hR1_20 : ls20.regs["r1"]? = some (.s32 (Int.ofNat j.val)) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR1
      (by decide)
  have hP1_20 : ls20.preds["p1"]? = some false :=
    CvtaFullPost.preserves_pred_value step20.post_holds hj hGet19 hGet20 hP1
  have hRd4_20 : ls20.regs["rd4"]? = some (.gaddr .global saxpyXBase) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd4
      (by decide)
  have hRd5_20 : ls20.regs["rd5"]? = some (saxpyLaneByteOffsetValue j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd5
      (by decide)
  have hRd6_20 : ls20.regs["rd6"]? = some (saxpyXElementAddrValue j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd6
      (by decide)
  have hRd7_20 : ls20.regs["rd7"]? = some (.gaddr .global saxpyYBase) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd7
      (by decide)
  have hRd8_20 : ls20.regs["rd8"]? = some (saxpyYElementAddrValue j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hRd8
      (by decide)
  have hR7_20 : ls20.regs["r7"]? = some (saxpyLoadedS32Value xs j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR7
      (by decide)
  have hR8_20 : ls20.regs["r8"]? = some (saxpyLoadedS32Value ys j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR8
      (by decide)
  have hR9_20 : ls20.regs["r9"]? = some (saxpyMulAddValue alpha xs ys j) :=
    CvtaFullPost.preserves_reg_value step20.post_holds hj hGet19 hGet20 hR9
      (by decide)
  refine
    ⟨ls20, hGet20, hR2_20, hR6_20, hRd1_20, hRd2_20, hRd3_20, hR3_20,
      hR4_20, hR5_20, hR1_20, hP1_20, hRd4_20, hRd5_20, hRd6_20, hRd7_20,
      hRd8_20, hR7_20, hR8_20, hR9_20, hRd9, hPc20, hStatus20⟩

/-- Body-step context for `add.s64 %rd10, %rd9, %rd5`. -/
noncomputable def saxpy_step21_ctx
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepContext (saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys).post
      ("saxpyKernel$fallthrough0", 9) saxpyFallthrough0 saxpyFallthrough0_add_result_addr
      (saxpyActiveLanes n hn) :=
  BodyStepRecord.nextContextNone
    (saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys)
    rfl rfl saxpyFallthrough0_body9_add_result_addr

/-- Step-record form of `add.s64 %rd10, %rd9, %rd5`. -/
noncomputable def saxpy_step21_add_result_addr_record
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    BodyStepRecord (saxpy_step21_ctx n hn hnpos alpha xs ys hxs hys)
      (AssignRegFullPost (saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys).post
        (saxpyActiveLanes n hn) "rd10" (saxpyResultElementAddrValue)
        ("saxpyKernel$fallthrough0", 9)) :=
  BodyStepContext.assignRegStep (saxpy_step21_ctx n hn hnpos alpha xs ys hxs hys)
    (dst := "rd10")
    (rhs := saxpyResultElementAddrRhs)
    (guard? := none)
    (saxpyResultElementAddrValue)
    (fun lane hLane => by
      have hPost := saxpy_step20_cvta_rd9_record_post n hn hnpos alpha xs ys hxs hys
      obtain ⟨ls, hGet, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
        _hR1, _hP1, _hRd4, hRd5, _hRd6, _hRd7, _hRd8, _hR7, _hR8, _hR9,
        hRd9, _hPc, _hStatus⟩ := hPost lane hLane
      exact evalRValue_saxpyResultElementAddrRhs_of_regs hGet hRd9 hRd5)

theorem saxpy_step21_add_result_addr_record_post
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    SaxpyAfterResultAddrPost n hn alpha xs ys
      (saxpy_step21_add_result_addr_record n hn hnpos alpha xs ys hxs hys).post := by
  let step21 := saxpy_step21_add_result_addr_record n hn hnpos alpha xs ys hxs hys
  have hStep20 := saxpy_step20_cvta_rd9_record_post n hn hnpos alpha xs ys hxs hys
  intro j hj
  obtain ⟨ls20, hGet20, _hR2, _hR6, _hRd1, _hRd2, _hRd3, _hR3, _hR4, _hR5,
    _hR1, _hP1, _hRd4, _hRd5, _hRd6, _hRd7, _hRd8, _hR7, _hR8, hR9, _hRd9,
    _hPc20, _hStatus20⟩ :=
    hStep20 j hj
  obtain ⟨ls21, hGet21, hRd10, hPc21, hStatus21⟩ :=
    AssignRegFullPost.assigned_dst step21.post_holds hj
  have hR9_21 : ls21.regs["r9"]? = some (saxpyMulAddValue alpha xs ys j) :=
    AssignRegFullPost.preserves_reg_value step21.post_holds hj hGet20 hGet21 hR9
      (by decide)
  exact ⟨ls21, hGet21, hR9_21, hRd10, hPc21, hStatus21⟩

theorem saxpy_step11_branch_target_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      SaxpyBranchTargetPost n hn alpha st11 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step1.step, step2.step, step3.step,
      step4.step, step5.step, step6.step, step7.step, step8.step, step9.step, step10.step,
      step11.step, ?_⟩
  exact saxpy_step11_branch_target_record n hn hnpos alpha xs ys

theorem saxpy_step12_cvta_rd4_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      SaxpyAfterCvtaRd4Post n hn alpha st12 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, step6.step, step7.step,
      step8.step, step9.step, step10.step, step11.step, step12.step, ?_⟩
  exact saxpy_step12_cvta_rd4_record_post n hn hnpos alpha xs ys

theorem saxpy_step13_mul_wide_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      SaxpyAfterMulWidePost n hn alpha st13 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, step6.step, step7.step,
      step8.step, step9.step, step10.step, step11.step, step12.step, step13.step, ?_⟩
  exact saxpy_step13_mul_wide_record_post n hn hnpos alpha xs ys

theorem saxpy_step14_add_x_addr_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      SaxpyAfterXAddrPost n hn alpha st14 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step1.step, step2.step, step3.step, step4.step, step5.step, step6.step,
      step7.step, step8.step, step9.step, step10.step, step11.step, step12.step,
      step13.step, step14.step, ?_⟩
  exact saxpy_step14_add_x_addr_record_post n hn hnpos alpha xs ys

theorem saxpy_step15_cvta_rd7_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      SaxpyAfterCvtaRd7Post n hn alpha st15 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step1.step, step2.step, step3.step, step4.step,
      step5.step, step6.step, step7.step, step8.step, step9.step, step10.step,
      step11.step, step12.step, step13.step, step14.step, step15.step, ?_⟩
  exact saxpy_step15_cvta_rd7_record_post n hn hnpos alpha xs ys

theorem saxpy_step16_add_y_addr_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15
        st16 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      StepMachine.step? st15 = some st16 ∧
      SaxpyAfterYAddrPost n hn alpha st16 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step16.post, step1.step, step2.step, step3.step,
      step4.step, step5.step, step6.step, step7.step, step8.step, step9.step, step10.step,
      step11.step, step12.step, step13.step, step14.step, step15.step, step16.step, ?_⟩
  exact saxpy_step16_add_y_addr_record_post n hn hnpos alpha xs ys

theorem saxpy_step17_load_x_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15
        st16 st17 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      StepMachine.step? st15 = some st16 ∧
      StepMachine.step? st16 = some st17 ∧
      SaxpyAfterXLoadPost n hn alpha xs st17 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step16.post, step17.post, step1.step, step2.step,
      step3.step, step4.step, step5.step, step6.step, step7.step, step8.step, step9.step,
      step10.step, step11.step, step12.step, step13.step, step14.step, step15.step,
      step16.step, step17.step, ?_⟩
  exact saxpy_step17_load_x_record_post n hn hnpos alpha xs ys hxs

theorem saxpy_step18_load_y_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15
        st16 st17 st18 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      StepMachine.step? st15 = some st16 ∧
      StepMachine.step? st16 = some st17 ∧
      StepMachine.step? st17 = some st18 ∧
      SaxpyAfterYLoadPost n hn alpha xs ys st18 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  let step18 := saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step16.post, step17.post, step18.post, step1.step,
      step2.step, step3.step, step4.step, step5.step, step6.step, step7.step,
      step8.step, step9.step, step10.step, step11.step, step12.step, step13.step,
      step14.step, step15.step, step16.step, step17.step, step18.step, ?_⟩
  exact saxpy_step18_load_y_record_post n hn hnpos alpha xs ys hxs hys

theorem saxpy_step19_mul_add_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15
        st16 st17 st18 st19 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      StepMachine.step? st15 = some st16 ∧
      StepMachine.step? st16 = some st17 ∧
      StepMachine.step? st17 = some st18 ∧
      StepMachine.step? st18 = some st19 ∧
      SaxpyAfterMulAddPost n hn alpha xs ys st19 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  let step18 := saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys
  let step19 := saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step16.post, step17.post, step18.post, step19.post,
      step1.step, step2.step, step3.step, step4.step, step5.step, step6.step,
      step7.step, step8.step, step9.step, step10.step, step11.step, step12.step,
      step13.step, step14.step, step15.step, step16.step, step17.step, step18.step,
      step19.step, ?_⟩
  exact saxpy_step19_mul_add_record_post n hn hnpos alpha xs ys hxs hys

theorem saxpy_step21_add_result_addr_accumulated
    (n : Nat) (hn : n ≤ 32) (hnpos : 0 < n)
    (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ st1 st2 st3 st4 st5 st6 st7 st8 st9 st10 st11 st12 st13 st14 st15
        st16 st17 st18 st19 st20 st21 : State,
      StepMachine.step? (saxpyStateFor n alpha xs ys) = some st1 ∧
      StepMachine.step? st1 = some st2 ∧
      StepMachine.step? st2 = some st3 ∧
      StepMachine.step? st3 = some st4 ∧
      StepMachine.step? st4 = some st5 ∧
      StepMachine.step? st5 = some st6 ∧
      StepMachine.step? st6 = some st7 ∧
      StepMachine.step? st7 = some st8 ∧
      StepMachine.step? st8 = some st9 ∧
      StepMachine.step? st9 = some st10 ∧
      StepMachine.step? st10 = some st11 ∧
      StepMachine.step? st11 = some st12 ∧
      StepMachine.step? st12 = some st13 ∧
      StepMachine.step? st13 = some st14 ∧
      StepMachine.step? st14 = some st15 ∧
      StepMachine.step? st15 = some st16 ∧
      StepMachine.step? st16 = some st17 ∧
      StepMachine.step? st17 = some st18 ∧
      StepMachine.step? st18 = some st19 ∧
      StepMachine.step? st19 = some st20 ∧
      StepMachine.step? st20 = some st21 ∧
      SaxpyAfterResultAddrPost n hn alpha xs ys st21 := by
  let step1 := saxpy_step1_load_param0_record n hn hnpos alpha xs ys
  let step2 := saxpy_step2_load_param1_record n hn hnpos alpha xs ys
  let step3 := saxpy_step3_load_param2_record n hn hnpos alpha xs ys
  let step4 := saxpy_step4_load_param3_record n hn hnpos alpha xs ys
  let step5 := saxpy_step5_load_param4_record n hn hnpos alpha xs ys
  let step6 := saxpy_step6_mov_ctaidX_record n hn hnpos alpha xs ys
  let step7 := saxpy_step7_mov_ntidX_record n hn hnpos alpha xs ys
  let step8 := saxpy_step8_mov_tidX_record n hn hnpos alpha xs ys
  let step9 := saxpy_step9_mad_index_record n hn hnpos alpha xs ys
  let step10 := saxpy_step10_setp_ge_record n hn hnpos alpha xs ys
  let step11 := saxpy_step11_branch_fallthrough_record n hn hnpos alpha xs ys
  let step12 := saxpy_step12_cvta_rd4_record n hn hnpos alpha xs ys
  let step13 := saxpy_step13_mul_wide_record n hn hnpos alpha xs ys
  let step14 := saxpy_step14_add_x_addr_record n hn hnpos alpha xs ys
  let step15 := saxpy_step15_cvta_rd7_record n hn hnpos alpha xs ys
  let step16 := saxpy_step16_add_y_addr_record n hn hnpos alpha xs ys
  let step17 := saxpy_step17_load_x_record n hn hnpos alpha xs ys hxs
  let step18 := saxpy_step18_load_y_record n hn hnpos alpha xs ys hxs hys
  let step19 := saxpy_step19_mul_add_record n hn hnpos alpha xs ys hxs hys
  let step20 := saxpy_step20_cvta_rd9_record n hn hnpos alpha xs ys hxs hys
  let step21 := saxpy_step21_add_result_addr_record n hn hnpos alpha xs ys hxs hys
  refine
    ⟨step1.post, step2.post, step3.post, step4.post, step5.post, step6.post, step7.post,
      step8.post, step9.post, step10.post, step11.post, step12.post, step13.post,
      step14.post, step15.post, step16.post, step17.post, step18.post, step19.post,
      step20.post, step21.post, step1.step, step2.step, step3.step, step4.step,
      step5.step, step6.step, step7.step, step8.step, step9.step, step10.step,
      step11.step, step12.step, step13.step, step14.step, step15.step, step16.step,
      step17.step, step18.step, step19.step, step20.step, step21.step, ?_⟩
  exact saxpy_step21_add_result_addr_record_post n hn hnpos alpha xs ys hxs hys

end CLean
