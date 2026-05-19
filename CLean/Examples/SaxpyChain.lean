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

theorem readMem?_param_s32_congr
    {st st' : State} {offset : Nat}
    (hParam : st.param = st'.param) :
    readMem? st .param .s32 (.param offset) =
      readMem? st' .param .s32 (.param offset) := by
  unfold readMem?
  simp [Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?,
        Typing.byteWidth?, Typing.aligned?, Typing.alignment?,
        Typing.addrSpaceMatches?, Addr.offset, Addr.space,
        getSpaceBaseMem?, hParam]

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

/-- Standard postcondition for a load body step: every participating lane has
the loaded value in `dst`, advances by one slot, and remains runnable. -/
def LoadStepPost (participants : List LaneId) (dst : RegName)
    (valueAt : LaneId → Value) (pc : PC) (post : State) : Prop :=
  ∀ lane ∈ participants,
    ∃ laneState : LaneState,
      post.getLane? 0 0 lane = some laneState ∧
      laneState.regs[dst]? = some (valueAt lane) ∧
      laneState.pc = (pc.1, pc.2 + 1) ∧
      laneState.status = .running

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
    BodyStepRecord ctx (LoadStepPost participants dst valueAt pc) := by
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
  refine ⟨laneState', hGet', ?_, hPc', ?_⟩
  · rw [hRegs]
    simp [Std.HashMap.getElem?_insert]
  · rw [hStatus', hStatus]

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
      (LoadStepPost (saxpyActiveLanes n hn) "r2"
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
      (LoadStepPost (saxpyActiveLanes n hn) "r6"
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
  intro j hj
  simpa [LoadStepPost] using hPost j hj

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
  intro j hj
  simpa [LoadStepPost, step2] using step2.post_holds j hj

end CLean
