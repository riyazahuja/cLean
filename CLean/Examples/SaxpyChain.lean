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

end BodyStepRecord

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
  obtain ⟨_, postLane, _, hPost, hRegs, hPc, hStatus⟩ := h lane hLane
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
  obtain ⟨_, postLane', _, hPostLane', hRegs, _, _⟩ := h lane hLane
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
  obtain ⟨preLane, postLane', hPreLane, hPostLane', hRegs, _, _⟩ := h lane hLane
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
  obtain ⟨_, postLane, _, hPost, hRegs, hPc, hStatus⟩ := h lane hLane
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
  obtain ⟨_, postLane', _, hPostLane', hRegs, _, _⟩ := h lane hLane
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
  obtain ⟨preLane, postLane', hPreLane, hPostLane', hRegs, _, _⟩ := h lane hLane
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

end AssignRegFullPost

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
  refine ⟨laneState, laneState', hGet, hGet', hRegs, hPc', ?_⟩
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
  refine ⟨laneState, laneState', hGet, hGet', hRegs, hPc', ?_⟩
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

end CLean
