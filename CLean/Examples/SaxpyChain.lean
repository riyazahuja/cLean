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

/-! ### Body-slot 0: `ld.param.u32 %r2, [param_0]`

The first instruction of saxpy's entry block. -/

/-- `saxpyBB0.body[0]?` exists. -/
theorem saxpyBB0_body0_isSome : (saxpyBB0.body[0]?).isSome = true := by
  unfold saxpyBB0; native_decide

/-- Concrete extraction of `saxpyBB0.body[0]?` as a `GInstr`. -/
def saxpyBB0_gi0 : GInstr := saxpyBB0.body[0]?.get saxpyBB0_body0_isSome

theorem saxpyBB0_body0 : saxpyBB0.body[0]? = some saxpyBB0_gi0 :=
  Option.eq_some_iff_get_eq.mpr ⟨saxpyBB0_body0_isSome, rfl⟩

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
  have hWf := saxpyStateFor_wf n alpha xs ys
  have hWarp := saxpyStateFor_getWarp n alpha xs ys
  have hWsWf := saxpyWarp_wf n
  have hLock := saxpyWarp_lockstepRunnable n
  have hPc := currentRunnablePc_saxpyWarp_pos n hn hnpos
  have hBlock := saxpyStateFor_blocks_lookup n alpha xs ys
  have hPart' : participatingRunnableLaneIds? (saxpyWarpFor n) (none : Option Guard)
                = some (saxpyActiveLanes n hn) :=
    participatingRunnable_saxpyWarp_none n hn hnpos
  have hPartNodup : (saxpyActiveLanes n hn).Nodup :=
    participatingRunnableLaneIds?_nodup hPart'
  -- Existence of st1 via the new succeeds lemma.
  obtain ⟨st1, hInstr⟩ :=
    stepInstr?_load_succeeds_uniform hWf hWarp hLock hPc hPart' hPartNodup
      (fun j _hj =>
        ⟨.param 0, resolveAddr_saxpyStateFor_param0 n alpha xs ys j,
         .u32 (UInt32.ofNat n), readMem_saxpyStateFor_param0 n alpha xs ys⟩)
  -- Tie back to `saxpyBB0_gi0` to apply step?_body_some.
  have hInstrGi : stepInstr? (saxpyStateFor n alpha xs ys) 0 0 saxpyBB0_gi0 = some st1 := by
    rw [saxpyBB0_gi0_eq]; exact hInstr
  have hPartGi : participatingRunnableLaneIds? (saxpyWarpFor n) saxpyBB0_gi0.guard?
                  = some (saxpyActiveLanes n hn) := by
    rw [saxpyBB0_gi0_guard_none]; exact hPart'
  refine ⟨st1, ?_, ?_⟩
  · exact step?_body_some hWf hWarp hWsWf hLock hPc hBlock saxpyBB0_body0 hPartGi hInstrGi
  · intro j hj
    have hLane := saxpyStateFor_getLane n alpha xs ys j
    have hLanePc : ({ pc := ("saxpyKernel", 0) } : LaneState).pc = ("saxpyKernel", 0) := rfl
    have hAddr := resolveAddr_saxpyStateFor_param0 n alpha xs ys j
    have hRead := readMem_saxpyStateFor_param0 n alpha xs ys
    obtain ⟨ls1, hGet1, hRegs, _hPreds, _hLocal, hStatus, hPc1⟩ :=
      stepInstr?_load_lane_full hWf hWarp hLock hPc hPart' hj hLane hLanePc hAddr hRead hInstr
    refine ⟨ls1, hGet1, ?_, ?_, ?_⟩
    · rw [hRegs]; simp [Std.HashMap.getElem?_insert]
    · exact hPc1
    · rw [hStatus]

end CLean


