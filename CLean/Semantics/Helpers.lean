import CLean.Core.State
import CLean.Core.Typing

namespace CLean

open Typing

namespace Helpers

private def intAbs (x : Int) : Int :=
  Int.ofNat x.natAbs

private def normalizeSigned (bits : Nat) (x : Int) : Int :=
  let signBit := 2 ^ (bits - 1)
  let modulus := 2 ^ bits
  let n := Int.toNat (x % Int.ofNat modulus)
  if n < signBit then
    Int.ofNat n
  else
    Int.ofNat n - Int.ofNat modulus

private def intToFloat (x : Int) : Float :=
  if x < 0 then
    -Float.ofNat x.natAbs
  else
    Float.ofNat x.natAbs

private def u32Xor (a b : UInt32) : UInt32 :=
  UInt32.ofNat (Nat.xor a.toNat b.toNat)

private def u64Xor (a b : UInt64) : UInt64 :=
  UInt64.ofNat (Nat.xor a.toNat b.toNat)

private def u32Shl (a : UInt32) (n : Nat) : UInt32 :=
  UInt32.ofNat (a.toNat * (2 ^ n))

private def u64Shl (a : UInt64) (n : Nat) : UInt64 :=
  UInt64.ofNat (a.toNat * (2 ^ n))

private def u32Shr (a : UInt32) (n : Nat) : UInt32 :=
  UInt32.ofNat (a.toNat / (2 ^ n))

private def u64Shr (a : UInt64) (n : Nat) : UInt64 :=
  UInt64.ofNat (a.toNat / (2 ^ n))

def bitSet (mask : UInt32) (i : Nat) : Bool :=
  ((mask.toNat / (2 ^ i)) % 2) = 1

def nonzeroDim (n : Nat) : Nat :=
  if n = 0 then 1 else n

def dim3X (linear : Nat) (dims : Dim3) : Nat :=
  linear % nonzeroDim dims.x

def dim3Y (linear : Nat) (dims : Dim3) : Nat :=
  (linear / nonzeroDim dims.x) % nonzeroDim dims.y

def dim3Z (linear : Nat) (dims : Dim3) : Nat :=
  linear / (nonzeroDim dims.x * nonzeroDim dims.y)

-- Warps partition block threads contiguously; dimensional coordinates are derived
-- from this block-local linear ID and the launch blockDim.
def blockLinearTid (warp : WarpId) (lane : LaneId) : Nat :=
  warp * 32 + lane.val

def blockLinearTidX (warp : WarpId) (lane : LaneId) : Nat :=
  blockLinearTid warp lane

def threadIdxX (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=
  dim3X (blockLinearTid warp lane) grid.blockDim

def threadIdxY (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=
  dim3Y (blockLinearTid warp lane) grid.blockDim

def threadIdxZ (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=
  dim3Z (blockLinearTid warp lane) grid.blockDim

def ctaIdxX (grid : GridCtx) (cta : CTAId) : Nat :=
  dim3X cta grid.gridDim

def ctaIdxY (grid : GridCtx) (cta : CTAId) : Nat :=
  dim3Y cta grid.gridDim

def ctaIdxZ (grid : GridCtx) (cta : CTAId) : Nat :=
  dim3Z cta grid.gridDim

def laneIds : List LaneId :=
  List.finRange 32

theorem laneIds_nodup : laneIds.Nodup := by
  simp [laneIds, List.finRange]

def readReg (lane : LaneState) (r : RegName) : Option Value :=
  lane.regs[r]?

def writeReg (lane : LaneState) (r : RegName) (v : Value) : LaneState :=
  { lane with regs := lane.regs.insert r v }

theorem writeReg_read_same (lane : LaneState) (r : RegName) (v : Value) :
    (writeReg lane r v).regs[r]? = some v := by
  simp [writeReg]

def readPred (lane : LaneState) (p : PredName) : Option Bool :=
  lane.preds[p]?

def writePred (lane : LaneState) (p : PredName) (b : Bool) : LaneState :=
  { lane with preds := lane.preds.insert p b }

theorem writePred_read_same (lane : LaneState) (p : PredName) (b : Bool) :
    (writePred lane p b).preds[p]? = some b := by
  simp [writePred]

def evalSpecial (grid : GridCtx) (cta : CTAId) (warp : WarpId) (lane : LaneId) : SpecialReg → Value
  | .tidX => .u32 <| UInt32.ofNat (threadIdxX grid warp lane)
  | .tidY => .u32 <| UInt32.ofNat (threadIdxY grid warp lane)
  | .tidZ => .u32 <| UInt32.ofNat (threadIdxZ grid warp lane)
  | .ctaidX => .u32 <| UInt32.ofNat (ctaIdxX grid cta)
  | .ctaidY => .u32 <| UInt32.ofNat (ctaIdxY grid cta)
  | .ctaidZ => .u32 <| UInt32.ofNat (ctaIdxZ grid cta)
  | .ntidX => .u32 <| UInt32.ofNat grid.blockDim.x
  | .ntidY => .u32 <| UInt32.ofNat grid.blockDim.y
  | .ntidZ => .u32 <| UInt32.ofNat grid.blockDim.z
  | .nctaidX => .u32 <| UInt32.ofNat grid.gridDim.x
  | .nctaidY => .u32 <| UInt32.ofNat grid.gridDim.y
  | .nctaidZ => .u32 <| UInt32.ofNat grid.gridDim.z

mutual
  partial def rvalueReadSet : RValue → ReadSet
    | .imm _ => {}
    | .reg r => { regs := [r] }
    | .pred p => { preds := [p] }
    | .special s => { specials := [s] }
    | .unop _ a => rvalueReadSet a
    | .binop _ a b => ReadSet.union (rvalueReadSet a) (rvalueReadSet b)
    | .triop _ a b c => ReadSet.union (rvalueReadSet a) <| ReadSet.union (rvalueReadSet b) (rvalueReadSet c)

  partial def cmpReadSet (cmp : CmpExpr) : ReadSet :=
    ReadSet.union (rvalueReadSet cmp.lhs) (rvalueReadSet cmp.rhs)
end

def instrReadSet : Instr → ReadSet
  | .assignReg _ rhs => rvalueReadSet rhs
  | .assignPred _ cmp => cmpReadSet cmp
  | .assignPredValue _ rhs => rvalueReadSet rhs
  | .load _ src => rvalueReadSet src.addr
  | .store dst value => ReadSet.union (rvalueReadSet dst.addr) (rvalueReadSet value)
  | .cvta _ _ src => rvalueReadSet src
  | .isspacep _ _ src => rvalueReadSet src
  | .barrierCTA _ => {}
  | .warp (.activemask _) => {}
  | .warp (.shflSync _ _ src laneOrDelta clamp memberMask) =>
      ReadSet.union (rvalueReadSet src) <| ReadSet.union (rvalueReadSet laneOrDelta) <| ReadSet.union (rvalueReadSet clamp) (rvalueReadSet memberMask)
  | .warp (.ballotSync _ pred memberMask) => ReadSet.union (rvalueReadSet pred) (rvalueReadSet memberMask)
  | .atomic _ _ addr arg1 arg2? =>
      let base := ReadSet.union (rvalueReadSet addr.addr) (rvalueReadSet arg1)
      match arg2? with
      | some arg2 => ReadSet.union base (rvalueReadSet arg2)
      | none => base
  | .mma spec => { regs := [spec.a, spec.b, spec.c] }

def terminatorReadSet : Terminator → ReadSet
  | .br _ => {}
  | .cbr cond _ _ => rvalueReadSet cond
  | .terminate => {}

def ginstrReadSet (gi : GInstr) : ReadSet :=
  let base := instrReadSet gi.instr
  match gi.guard? with
  | some g => { base with preds := g.pred :: base.preds }
  | none => base

def valueToBool? : Value → Option Bool
  | .pred b => some b
  | _ => none

def natToBytesLE (n width : Nat) : List Byte :=
  (List.range width).map fun i => UInt8.ofNat ((n / (2 ^ (8 * i))) % 256)

def bytesToNatLE (bs : List Byte) : Nat :=
  let rec loop (i : Nat) : List Byte → Nat
    | [] => 0
    | b :: rest => b.toNat * (2 ^ (8 * i)) + loop (i + 1) rest
  loop 0 bs

def signedToNat (bits : Nat) (x : Int) : Nat :=
  let modulus : Int := Int.ofNat (2 ^ bits)
  Int.toNat (x % modulus)

def natToSigned (bits : Nat) (n : Nat) : Int :=
  let signBit := 2 ^ (bits - 1)
  let modulus := 2 ^ bits
  if n < signBit then Int.ofNat n else Int.ofNat n - Int.ofNat modulus

def encodeScalar? : ScalarTy → Value → Option (List Byte)
  | .pred, .pred b => some [if b then 1 else 0]
  | .u8, .u8 x | .b8, .b8 x => some [x]
  | .u16, .u16 x | .b16, .b16 x => some <| natToBytesLE x.toNat 2
  | .u32, .u32 x | .b32, .b32 x => some <| natToBytesLE x.toNat 4
  | .u64, .u64 x | .b64, .b64 x => some <| natToBytesLE x.toNat 8
  | .s8, .s8 x => some <| natToBytesLE (signedToNat 8 x) 1
  | .s16, .s16 x => some <| natToBytesLE (signedToNat 16 x) 2
  | .s32, .s32 x => some <| natToBytesLE (signedToNat 32 x) 4
  | .s64, .s64 x => some <| natToBytesLE (signedToNat 64 x) 8
  | .f16, .f16 bits | .bf16, .bf16 bits => some <| natToBytesLE bits.toNat 2
  | .f32, .f32 x => some <| natToBytesLE x.toFloat32.toBits.toNat 4
  | .f64, .f64 x => some <| natToBytesLE x.toBits.toNat 8
  | _, _ => none

def decodeScalar? : ScalarTy → List Byte → Option Value
  | .pred, [b] => some (.pred (b != 0))
  | .u8, [b] => some (.u8 b)
  | .b8, [b] => some (.b8 b)
  | .u16, bs => if bs.length = 2 then some (.u16 (UInt16.ofNat <| bytesToNatLE bs)) else none
  | .b16, bs => if bs.length = 2 then some (.b16 (UInt16.ofNat <| bytesToNatLE bs)) else none
  | .u32, bs => if bs.length = 4 then some (.u32 (UInt32.ofNat <| bytesToNatLE bs)) else none
  | .b32, bs => if bs.length = 4 then some (.b32 (UInt32.ofNat <| bytesToNatLE bs)) else none
  | .u64, bs => if bs.length = 8 then some (.u64 (UInt64.ofNat <| bytesToNatLE bs)) else none
  | .b64, bs => if bs.length = 8 then some (.b64 (UInt64.ofNat <| bytesToNatLE bs)) else none
  | .s8, bs => if bs.length = 1 then some (.s8 (natToSigned 8 <| bytesToNatLE bs)) else none
  | .s16, bs => if bs.length = 2 then some (.s16 (natToSigned 16 <| bytesToNatLE bs)) else none
  | .s32, bs => if bs.length = 4 then some (.s32 (natToSigned 32 <| bytesToNatLE bs)) else none
  | .s64, bs => if bs.length = 8 then some (.s64 (natToSigned 64 <| bytesToNatLE bs)) else none
  | .f16, bs => if bs.length = 2 then some (.f16 (UInt16.ofNat <| bytesToNatLE bs)) else none
  | .bf16, bs => if bs.length = 2 then some (.bf16 (UInt16.ofNat <| bytesToNatLE bs)) else none
  | .f32, bs => if bs.length = 4 then some (.f32 <| (Float32.ofBits <| UInt32.ofNat <| bytesToNatLE bs).toFloat) else none
  | .f64, bs => if bs.length = 8 then some (.f64 <| Float.ofBits <| UInt64.ofNat <| bytesToNatLE bs) else none
  | _, _ => none

def readBytes? (mem : ByteMem) (offset width : Nat) : Option (List Byte) :=
  let rec loop (i : Nat) (acc : List Byte) :=
    if i < width then
      match mem[offset + i]? with
      | some b => loop (i + 1) (b :: acc)
      | none => none
    else
      some acc.reverse
  loop 0 []

def writeBytes (mem : ByteMem) (offset : Nat) (bytes : List Byte) : ByteMem :=
  let rec loop (i : Nat) (acc : ByteMem) : List Byte → ByteMem
    | [] => acc
    | b :: rest => loop (i + 1) (acc.insert (offset + i) b) rest
  loop 0 mem bytes

def guardHolds? (lane : LaneState) (guard? : Option Guard) : Option Bool :=
  match guard? with
  | none => some true
  | some g => do
      let b <- readPred lane g.pred
      pure (if g.negate then !b else b)

def laneIsRunnable (warp : WarpState) (lane : LaneId) : Bool :=
  match warp.getLane? lane with
  | some laneState => laneState.status == .running && bitSet warp.activeMask lane.val
  | none => false

def runnableLaneIds (warp : WarpState) : List LaneId :=
  laneIds.filter fun lane => laneIsRunnable warp lane

def currentRunnablePc? (warp : WarpState) : Option PC := do
  let lane :: _ <- pure <| runnableLaneIds warp | none
  let laneState <- warp.getLane? lane
  pure laneState.pc

def participatingRunnableLaneIdsFrom?
    (warp : WarpState) (guard? : Option Guard) (pc : PC) :
    List LaneId → Option (List LaneId)
  | [] => some []
  | lane :: lanes => do
      let laneState <- warp.getLane? lane
      let rest <- participatingRunnableLaneIdsFrom? warp guard? pc lanes
      if laneState.pc = pc then
        let passes <- guardHolds? laneState guard?
        if passes then
          pure (lane :: rest)
        else
          pure rest
      else
        pure rest

def participatingRunnableLaneIds? (warp : WarpState) (guard? : Option Guard) : Option (List LaneId) := do
  let pc <- currentRunnablePc? warp
  participatingRunnableLaneIdsFrom? warp guard? pc (runnableLaneIds warp)

def RunnablePc (warp : WarpState) (pc : PC) : Prop :=
  currentRunnablePc? warp = some pc

def ParticipatingRunnable (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Prop :=
  participatingRunnableLaneIds? warp guard? = some lanes

theorem runnablePc_iff_bool (warp : WarpState) (pc : PC) :
    Helpers.RunnablePc warp pc ↔ Helpers.currentRunnablePc? warp = some pc := Iff.rfl

instance (warp : WarpState) (pc : PC) : Decidable (Helpers.RunnablePc warp pc) := by
  unfold Helpers.RunnablePc
  infer_instance

theorem participatingRunnable_iff_bool (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) :
    Helpers.ParticipatingRunnable warp guard? lanes ↔
      Helpers.participatingRunnableLaneIds? warp guard? = some lanes := Iff.rfl

instance (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) :
    Decidable (Helpers.ParticipatingRunnable warp guard? lanes) := by
  unfold Helpers.ParticipatingRunnable
  infer_instance

def lockstepRunnable? (warp : WarpState) : Bool :=
  match currentRunnablePc? warp with
  | none => true
  | some pc =>
      (runnableLaneIds warp).all fun lane =>
        match warp.getLane? lane with
        | some laneState => laneState.pc == pc
        | none => false

def lockstepRunnable (warp : WarpState) : Prop :=
  lockstepRunnable? warp = true

theorem lockstepRunnable_iff_bool (warp : WarpState) :
    Helpers.lockstepRunnable warp ↔ Helpers.lockstepRunnable? warp = true := Iff.rfl

instance (warp : WarpState) : Decidable (Helpers.lockstepRunnable warp) := by
  unfold Helpers.lockstepRunnable
  infer_instance

theorem laneIsRunnable_setLane_control_eq
    {warpState : WarpState} {lane target : LaneId} {old new : LaneState}
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status) :
    laneIsRunnable (warpState.setLane lane new) target = laneIsRunnable warpState target := by
  by_cases h : target = lane
  · subst target
    unfold laneIsRunnable
    rw [WarpState.getLane?_setLane_same hget, hget]
    have hmask : (warpState.setLane lane new).activeMask = warpState.activeMask := rfl
    rw [hmask]
    simp [hstatus]
  · unfold laneIsRunnable
    rw [WarpState.getLane?_setLane_ne hget h]
    have hmask : (warpState.setLane lane new).activeMask = warpState.activeMask := rfl
    rw [hmask]

theorem runnableLaneIds_setLane_control_eq
    {warpState : WarpState} {lane : LaneId} {old new : LaneState}
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status) :
    runnableLaneIds (warpState.setLane lane new) = runnableLaneIds warpState := by
  simp [runnableLaneIds, laneIsRunnable_setLane_control_eq hget hstatus]

theorem runnableLaneIds_nodup (warpState : WarpState) :
    (runnableLaneIds warpState).Nodup := by
  unfold runnableLaneIds
  exact List.Pairwise.filter (p := fun lane => laneIsRunnable warpState lane) laneIds_nodup

theorem currentRunnablePc?_setLane_control_eq
    {warpState : WarpState} {lane : LaneId} {old new : LaneState}
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status)
    (hpc : new.pc = old.pc) :
    currentRunnablePc? (warpState.setLane lane new) = currentRunnablePc? warpState := by
  unfold currentRunnablePc?
  rw [runnableLaneIds_setLane_control_eq hget hstatus]
  cases runnableLaneIds warpState with
  | nil =>
      simp
  | cons first _ =>
      by_cases hfirst : first = lane
      · subst first
        simp [WarpState.getLane?_setLane_same hget, hget, hpc]
      · simp [WarpState.getLane?_setLane_ne hget hfirst]

theorem lockstepRunnable_setLane_control_iff
    {warpState : WarpState} {lane : LaneId} {old new : LaneState}
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status)
    (hpc : new.pc = old.pc) :
    lockstepRunnable (warpState.setLane lane new) ↔ lockstepRunnable warpState := by
  unfold lockstepRunnable lockstepRunnable?
  rw [currentRunnablePc?_setLane_control_eq hget hstatus hpc,
    runnableLaneIds_setLane_control_eq hget hstatus]
  cases currentRunnablePc? warpState with
  | none =>
      simp
  | some pc =>
      simp
      apply Iff.intro
      · intro hall target hmem
        have htarget := hall target hmem
        by_cases heq : target = lane
        · subst target
          rw [WarpState.getLane?_setLane_same hget] at htarget
          simp [hpc] at htarget
          simpa [hget] using htarget
        · rw [WarpState.getLane?_setLane_ne hget heq] at htarget
          exact htarget
      · intro hall target hmem
        have htarget := hall target hmem
        by_cases heq : target = lane
        · subst target
          rw [WarpState.getLane?_setLane_same hget]
          simp [hpc]
          simpa [hget] using htarget
        · rw [WarpState.getLane?_setLane_ne hget heq]
          exact htarget

theorem participatingRunnableLaneIdsFrom?_setLane_none_control_eq
    {warpState : WarpState} {lane : LaneId} {old new : LaneState} {pc : PC}
    (hget : warpState.getLane? lane = some old)
    (hpc : new.pc = old.pc) :
    ∀ lanes,
      participatingRunnableLaneIdsFrom? (warpState.setLane lane new) none pc lanes =
        participatingRunnableLaneIdsFrom? warpState none pc lanes := by
  intro lanes
  induction lanes with
  | nil =>
      rfl
  | cons target rest ih =>
      simp [participatingRunnableLaneIdsFrom?]
      by_cases htarget : target = lane
      · subst target
        simp [WarpState.getLane?_setLane_same hget, hget, ih, hpc, guardHolds?]
      · simp [WarpState.getLane?_setLane_ne hget htarget, ih, guardHolds?]

theorem participatingRunnableLaneIds?_setLane_none_control_eq
    {warpState : WarpState} {lane : LaneId} {old new : LaneState}
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status)
    (hpc : new.pc = old.pc) :
    participatingRunnableLaneIds? (warpState.setLane lane new) none =
      participatingRunnableLaneIds? warpState none := by
  unfold participatingRunnableLaneIds?
  rw [currentRunnablePc?_setLane_control_eq hget hstatus hpc,
    runnableLaneIds_setLane_control_eq hget hstatus]
  cases currentRunnablePc? warpState with
  | none =>
      simp
  | some pc =>
      simp [participatingRunnableLaneIdsFrom?_setLane_none_control_eq hget hpc]

theorem ParticipatingRunnable.setLane_none_control_eq
    {warpState : WarpState} {lane : LaneId} {old new : LaneState} {lanes : List LaneId}
    (hpart : ParticipatingRunnable warpState none lanes)
    (hget : warpState.getLane? lane = some old)
    (hstatus : new.status = old.status)
    (hpc : new.pc = old.pc) :
    ParticipatingRunnable (warpState.setLane lane new) none lanes := by
  unfold ParticipatingRunnable at hpart ⊢
  rw [participatingRunnableLaneIds?_setLane_none_control_eq hget hstatus hpc]
  exact hpart

theorem participatingRunnableLaneIdsFrom?_mem_pc
    {warpState : WarpState} {guard? : Option Guard} {pc : PC}
    {input output : List LaneId} {lane : LaneId} {laneState : LaneState}
    (hfrom : participatingRunnableLaneIdsFrom? warpState guard? pc input = some output)
    (hmem : lane ∈ output)
    (hlane : warpState.getLane? lane = some laneState) :
    laneState.pc = pc := by
  induction input generalizing output lane laneState with
  | nil =>
      simp [participatingRunnableLaneIdsFrom?] at hfrom
      subst output
      simp at hmem
  | cons head rest ih =>
      simp [participatingRunnableLaneIdsFrom?] at hfrom
      cases hhead : warpState.getLane? head with
      | none =>
          simp [hhead] at hfrom
      | some headState =>
          simp [hhead] at hfrom
          cases hrest : participatingRunnableLaneIdsFrom? warpState guard? pc rest with
          | none =>
              simp [hrest] at hfrom
          | some restOutput =>
              simp [hrest] at hfrom
              by_cases hpcHead : headState.pc = pc
              · simp [hpcHead] at hfrom
                cases hguard : guardHolds? headState guard? with
                | none =>
                    simp [hguard] at hfrom
                | some passes =>
                    simp [hguard] at hfrom
                    by_cases hpasses : passes = true
                    · simp [hpasses] at hfrom
                      subst output
                      simp at hmem
                      rcases hmem with hsame | hmemRest
                      · subst lane
                        rw [hhead] at hlane
                        injection hlane with hsameState
                        subst laneState
                        exact hpcHead
                      · exact ih (output := restOutput) hrest hmemRest hlane
                    · simp [hpasses] at hfrom
                      subst output
                      exact ih (output := restOutput) hrest hmem hlane
              · simp [hpcHead] at hfrom
                subst output
                exact ih (output := restOutput) hrest hmem hlane

theorem participatingRunnableLaneIdsFrom?_mem_input
    {warpState : WarpState} {guard? : Option Guard} {pc : PC}
    {input output : List LaneId} {lane : LaneId}
    (hfrom : participatingRunnableLaneIdsFrom? warpState guard? pc input = some output)
    (hmem : lane ∈ output) :
    lane ∈ input := by
  induction input generalizing output with
  | nil =>
      simp [participatingRunnableLaneIdsFrom?] at hfrom
      subst output
      simp at hmem
  | cons head rest ih =>
      simp [participatingRunnableLaneIdsFrom?] at hfrom
      cases hhead : warpState.getLane? head with
      | none =>
          simp [hhead] at hfrom
      | some headState =>
          simp [hhead] at hfrom
          cases hrest : participatingRunnableLaneIdsFrom? warpState guard? pc rest with
          | none =>
              simp [hrest] at hfrom
          | some restOutput =>
              simp [hrest] at hfrom
              by_cases hpcHead : headState.pc = pc
              · simp [hpcHead] at hfrom
                cases hguard : guardHolds? headState guard? with
                | none =>
                    simp [hguard] at hfrom
                | some passes =>
                    simp [hguard] at hfrom
                    by_cases hpasses : passes = true
                    · simp [hpasses] at hfrom
                      subst output
                      simp at hmem ⊢
                      rcases hmem with hsame | hrestMem
                      · exact Or.inl hsame
                      · exact Or.inr (ih hrest hrestMem)
                    · simp [hpasses] at hfrom
                      subst output
                      simp [ih hrest hmem]
              · simp [hpcHead] at hfrom
                subst output
                simp [ih hrest hmem]

theorem runnableLaneIds_mem_status
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    (hmem : lane ∈ runnableLaneIds warpState)
    (hlane : warpState.getLane? lane = some laneState) :
    laneState.status = .running := by
  unfold runnableLaneIds at hmem
  simp [laneIsRunnable, hlane] at hmem
  exact hmem.2.1

theorem runnableLaneIds_mem_getLane
    {warpState : WarpState} {lane : LaneId}
    (hmem : lane ∈ runnableLaneIds warpState) :
    ∃ laneState, warpState.getLane? lane = some laneState := by
  unfold runnableLaneIds at hmem
  simp [laneIsRunnable] at hmem
  cases hget : warpState.getLane? lane with
  | none =>
      simp [hget] at hmem
  | some laneState =>
      exact ⟨laneState, rfl⟩

theorem lockstepRunnable_mem_pc
    {warpState : WarpState} {pc : PC} {lane : LaneId} {laneState : LaneState}
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hmem : lane ∈ runnableLaneIds warpState)
    (hlane : warpState.getLane? lane = some laneState) :
    laneState.pc = pc := by
  unfold lockstepRunnable lockstepRunnable? at hlock
  unfold RunnablePc at hrpc
  simp [hrpc] at hlock
  have htarget := hlock lane hmem
  simpa [hlane] using htarget

theorem runnableLaneIds_filter_current_pc_eq
    {warpState : WarpState} {pc : PC}
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc) :
    List.filter
        (fun lane =>
          match warpState.getLane? lane with
          | some laneState => laneState.pc == pc
          | none => false)
        (runnableLaneIds warpState) =
      runnableLaneIds warpState := by
  apply List.filter_eq_self.2
  intro lane hmem
  rcases runnableLaneIds_mem_getLane hmem with ⟨laneState, hlane⟩
  have hpc : laneState.pc = pc :=
    lockstepRunnable_mem_pc hlock hrpc hmem hlane
  simp [hlane, hpc]

theorem ParticipatingRunnable.single_lane_pc
    {warpState : WarpState} {pc : PC} {lane : LaneId} {laneState : LaneState}
    (hrpc : RunnablePc warpState pc)
    (hpart : ParticipatingRunnable warpState none [lane])
    (hlane : warpState.getLane? lane = some laneState) :
    laneState.pc = pc := by
  unfold ParticipatingRunnable participatingRunnableLaneIds? at hpart
  unfold RunnablePc at hrpc
  rw [hrpc] at hpart
  simp at hpart
  exact participatingRunnableLaneIdsFrom?_mem_pc hpart (by simp) hlane

theorem ParticipatingRunnable.single_lane_status
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    (hpart : ParticipatingRunnable warpState none [lane])
    (hlane : warpState.getLane? lane = some laneState) :
    laneState.status = .running := by
  unfold ParticipatingRunnable participatingRunnableLaneIds? at hpart
  cases hcur : currentRunnablePc? warpState with
  | none =>
      simp [hcur] at hpart
  | some pc =>
      simp [hcur] at hpart
      have hmemInput : lane ∈ runnableLaneIds warpState :=
        participatingRunnableLaneIdsFrom?_mem_input hpart (by simp)
      exact runnableLaneIds_mem_status hmemInput hlane

theorem participatingRunnableLaneIdsFrom?_none_of_forall_pc
    {warpState : WarpState} {pc : PC} :
    ∀ lanes,
      (∀ lane, lane ∈ lanes →
        (match warpState.getLane? lane with
        | some laneState => laneState.pc == pc
        | none => false) = true) →
      participatingRunnableLaneIdsFrom? warpState none pc lanes = some lanes := by
  intro lanes
  induction lanes with
  | nil =>
      intro _hall
      rfl
  | cons lane rest ih =>
      intro hall
      have hhead := hall lane (by simp)
      have hrest :
          ∀ x, x ∈ rest →
            (match warpState.getLane? x with
            | some laneState => laneState.pc == pc
            | none => false) = true := by
        intro x hx
        exact hall x (by simp [hx])
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hhead
      | some laneState =>
          simp [hget] at hhead
          have hpc : laneState.pc = pc := hhead
          simp [participatingRunnableLaneIdsFrom?, hget, ih hrest, hpc, guardHolds?]

theorem runnableLaneIds_eq_of_lockstep_participants_none
    {warpState : WarpState} {lanes : List LaneId}
    (hlock : lockstepRunnable warpState)
    (hpart : ParticipatingRunnable warpState none lanes) :
    runnableLaneIds warpState = lanes := by
  unfold lockstepRunnable lockstepRunnable? at hlock
  unfold ParticipatingRunnable participatingRunnableLaneIds? at hpart
  cases hcur : currentRunnablePc? warpState with
  | none =>
      simp [hcur] at hpart
  | some pc =>
      simp [hcur] at hlock hpart
      have hfrom := participatingRunnableLaneIdsFrom?_none_of_forall_pc
        (warpState := warpState) (pc := pc) (runnableLaneIds warpState) hlock
      rw [hfrom] at hpart
      simpa using hpart

theorem participatingRunnable_none_eq_runnableLaneIds_of_lockstep
    {warpState : WarpState} {pc : PC}
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc) :
    participatingRunnableLaneIds? warpState none = some (runnableLaneIds warpState) := by
  unfold lockstepRunnable lockstepRunnable? at hlock
  unfold RunnablePc at hrpc
  unfold participatingRunnableLaneIds?
  rw [hrpc]
  simp
  simp [hrpc] at hlock
  exact participatingRunnableLaneIdsFrom?_none_of_forall_pc
    (warpState := warpState) (pc := pc) (runnableLaneIds warpState) hlock

def getSpaceBaseMem? (st : State) (addr : Addr) : Option ByteMem :=
  match addr with
  | .global _ => some st.global.bytes
  | .param _ => some st.param.bytes
  | .const _ => some st.const.bytes
  | .shared cta _ => do
      let ctaState <- st.getCTA? cta
      pure ctaState.shared.bytes
  | .local cta warp lane _ => do
      let laneState <- st.getLane? cta warp lane
      pure laneState.localMem.bytes
  | .generic _ _ => none

def setSpaceBaseMem? (st : State) (addr : Addr) (bytes : ByteMem) : Option State :=
  match addr with
  | .global _ => some { st with global := { bytes := bytes } }
  | .param _ => some { st with param := { bytes := bytes } }
  | .const _ => some { st with const := { bytes := bytes } }
  | .shared cta _ => do
      let ctaState <- st.getCTA? cta
      let ctaState := { ctaState with shared := { bytes := bytes } }
      pure <| st.setCTA cta ctaState
  | .local cta warp lane _ => do
      let laneState <- st.getLane? cta warp lane
      let laneState := { laneState with localMem := { bytes := bytes } }
      st.setLane cta warp lane laneState
  | .generic _ _ => none

def readMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Option Value := do
  if !Typing.typedAccessPreconditions? space ty addr then
    none
  else
    let width <- Typing.byteWidth? ty
    let mem <- getSpaceBaseMem? st addr
    let bytes <- readBytes? mem addr.offset width
    decodeScalar? ty bytes

def writeMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) (value : Value) : Option State := do
  if !Typing.typedAccessPreconditions? space ty addr then
    none
  else
    let bytes <- encodeScalar? ty value
    let mem <- getSpaceBaseMem? st addr
    let mem := writeBytes mem addr.offset bytes
    setSpaceBaseMem? st addr mem

theorem setSpaceBaseMem?_kernelEnv_eq
    {st st' : State} {addr : Addr} {bytes : ByteMem}
    (hset : setSpaceBaseMem? st addr bytes = some st') :
    st'.kernelEnv = st.kernelEnv := by
  cases addr with
  | global offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      rfl
  | param offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      rfl
  | const offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      rfl
  | shared cta offset =>
      unfold setSpaceBaseMem? at hset
      cases hcta : st.getCTA? cta with
      | none =>
          simp [hcta] at hset
      | some ctaState =>
          simp [hcta] at hset
          subst st'
          rfl
  | «local» cta warp lane offset =>
      unfold setSpaceBaseMem? at hset
      cases hlane : st.getLane? cta warp lane with
      | none =>
          simp [hlane] at hset
      | some laneState =>
          simp [hlane] at hset
          exact State.setLane_kernelEnv_eq hset
  | generic space offset =>
      simp [setSpaceBaseMem?] at hset

theorem writeMem?_kernelEnv_eq
    {st st' : State} {space : AddrSpace} {ty : ScalarTy} {addr : Addr} {value : Value}
    (hwrite : writeMem? st space ty addr value = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold writeMem? at hwrite
  cases haccess : (!Typing.typedAccessPreconditions? space ty addr) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess] at hwrite
      cases henc : encodeScalar? ty value with
      | none =>
          simp [henc] at hwrite
      | some encoded =>
          simp [henc] at hwrite
          cases hmem : getSpaceBaseMem? st addr with
          | none =>
              simp [hmem] at hwrite
          | some mem =>
              simp [hmem] at hwrite
              exact setSpaceBaseMem?_kernelEnv_eq hwrite

def addrMatchesWarp (cta : CTAId) (warp : WarpId) : Addr → Prop
  | .shared cta' _ => cta' = cta
  | .local cta' warp' _ _ => cta' = cta ∧ warp' = warp
  | _ => True

theorem setSpaceBaseMem?_warp_control_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {pc : PC} {addr : Addr} {bytes : ByteMem}
    (haddr : addrMatchesWarp cta warp addr)
    (hset : setSpaceBaseMem? st addr bytes = some st')
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc) :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' pc := by
  cases addr with
  | global offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, hlock, hrpc⟩
  | param offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, hlock, hrpc⟩
  | const offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, hlock, hrpc⟩
  | shared cta' offset =>
      unfold addrMatchesWarp at haddr
      subst cta'
      unfold setSpaceBaseMem? at hset
      cases hcta : st.getCTA? cta with
      | none =>
          simp [hcta] at hset
      | some ctaState =>
          simp [hcta] at hset
          subst st'
          have hctaRaw : st.ctas[cta]? = some ctaState := by
            simpa [State.getCTA?] using hcta
          exact ⟨warpState, by
            simpa [State.getWarp?, State.getCTA?, State.setCTA, hctaRaw] using hwarp,
            hlock, hrpc⟩
  | «local» cta' warp' lane offset =>
      unfold addrMatchesWarp at haddr
      rcases haddr with ⟨rfl, rfl⟩
      unfold setSpaceBaseMem? at hset
      cases hlane : st.getLane? cta' warp' lane with
      | none =>
          simp [hlane] at hset
      | some laneState =>
          simp [hlane] at hset
          let laneState' : LaneState :=
            { laneState with localMem := { bytes := bytes } }
          have hset' :
              st.setLane cta' warp' lane laneState' = some st' := by
            simpa [laneState'] using hset
          have hwarpLane : warpState.getLane? lane = some laneState := by
            unfold State.getLane? at hlane
            simp [hwarp] at hlane
            exact hlane
          have hstatus : laneState'.status = laneState.status := by
            simp [laneState']
          have hpc : laneState'.pc = laneState.pc := by
            simp [laneState']
          have hwarp' :
              st'.getWarp? cta' warp' = some (warpState.setLane lane laneState') :=
            State.getWarp?_setLane_same hwarp hset'
          have hlock' : lockstepRunnable (warpState.setLane lane laneState') :=
            (lockstepRunnable_setLane_control_iff hwarpLane hstatus hpc).2 hlock
          have hrpc' : RunnablePc (warpState.setLane lane laneState') pc := by
            unfold RunnablePc
            rw [currentRunnablePc?_setLane_control_eq hwarpLane hstatus hpc]
            exact hrpc
          exact ⟨warpState.setLane lane laneState', hwarp', hlock', hrpc'⟩
  | generic space offset =>
      simp [setSpaceBaseMem?] at hset

theorem setSpaceBaseMem?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {addr : Addr} {bytes : ByteMem}
    (haddr : addrMatchesWarp cta warp addr)
    (hset : setSpaceBaseMem? st addr bytes = some st')
    (hwarp : st.getWarp? cta warp = some warpState) :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState := by
  cases addr with
  | global offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, rfl⟩
  | param offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, rfl⟩
  | const offset =>
      simp [setSpaceBaseMem?] at hset
      subst st'
      exact ⟨warpState, hwarp, rfl⟩
  | shared cta' offset =>
      unfold addrMatchesWarp at haddr
      subst cta'
      unfold setSpaceBaseMem? at hset
      cases hcta : st.getCTA? cta with
      | none =>
          simp [hcta] at hset
      | some ctaState =>
          simp [hcta] at hset
          subst st'
          have hctaRaw : st.ctas[cta]? = some ctaState := by
            simpa [State.getCTA?] using hcta
          exact ⟨warpState, by
            simpa [State.getWarp?, State.getCTA?, State.setCTA, hctaRaw] using hwarp,
            rfl⟩
  | «local» cta' warp' lane offset =>
      unfold addrMatchesWarp at haddr
      rcases haddr with ⟨rfl, rfl⟩
      unfold setSpaceBaseMem? at hset
      cases hlane : st.getLane? cta' warp' lane with
      | none =>
          simp [hlane] at hset
      | some laneState =>
          simp [hlane] at hset
          let laneState' : LaneState :=
            { laneState with localMem := { bytes := bytes } }
          have hset' :
              st.setLane cta' warp' lane laneState' = some st' := by
            simpa [laneState'] using hset
          have hwarpLane : warpState.getLane? lane = some laneState := by
            unfold State.getLane? at hlane
            simp [hwarp] at hlane
            exact hlane
          have hstatus : laneState'.status = laneState.status := by
            simp [laneState']
          have hwarp' :
              st'.getWarp? cta' warp' = some (warpState.setLane lane laneState') :=
            State.getWarp?_setLane_same hwarp hset'
          exact ⟨warpState.setLane lane laneState', hwarp',
            runnableLaneIds_setLane_control_eq hwarpLane hstatus⟩
  | generic space offset =>
      simp [setSpaceBaseMem?] at hset

theorem writeMem?_warp_control_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {pc : PC} {space : AddrSpace} {ty : ScalarTy} {addr : Addr} {value : Value}
    (haddr : addrMatchesWarp cta warp addr)
    (hwrite : writeMem? st space ty addr value = some st')
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc) :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' pc := by
  unfold writeMem? at hwrite
  cases haccess : (!Typing.typedAccessPreconditions? space ty addr) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess] at hwrite
      cases henc : encodeScalar? ty value with
      | none =>
          simp [henc] at hwrite
      | some encoded =>
          simp [henc] at hwrite
          cases hmem : getSpaceBaseMem? st addr with
          | none =>
              simp [hmem] at hwrite
          | some mem =>
              simp [hmem] at hwrite
              exact setSpaceBaseMem?_warp_control_eq haddr hwrite hwarp hlock hrpc

theorem writeMem?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {space : AddrSpace} {ty : ScalarTy} {addr : Addr} {value : Value}
    (haddr : addrMatchesWarp cta warp addr)
    (hwrite : writeMem? st space ty addr value = some st')
    (hwarp : st.getWarp? cta warp = some warpState) :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState := by
  unfold writeMem? at hwrite
  cases haccess : (!Typing.typedAccessPreconditions? space ty addr) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess] at hwrite
      cases henc : encodeScalar? ty value with
      | none =>
          simp [henc] at hwrite
      | some encoded =>
          simp [henc] at hwrite
          cases hmem : getSpaceBaseMem? st addr with
          | none =>
              simp [hmem] at hwrite
          | some mem =>
              simp [hmem] at hwrite
              exact setSpaceBaseMem?_runnableLaneIds_eq haddr hwrite hwarp

def taggedGenericAddr? (cta : CTAId) (warp : WarpId) (lane : LaneId)
    (space : AddrSpace) (offset : Nat) : Option Addr :=
  match space with
  | .global => some (.global offset)
  | .shared => some (.shared cta offset)
  | .local => some (.local cta warp lane offset)
  | .param => some (.param offset)
  | .const => some (.const offset)
  | .generic => none

def evalCvta? (space : AddrSpace) (value : Value) : Option Value :=
  match space, value with
  | .generic, .gaddr s off => some (.gaddr s off)
  | .global, .b32 off => some (.gaddr .global off.toNat)
  | .global, .b64 off => some (.gaddr .global off.toNat)
  | .global, .u32 off => some (.gaddr .global off.toNat)
  | .global, .u64 off => some (.gaddr .global off.toNat)
  | .global, .gaddr .global off => some (.gaddr .global off)
  | .shared, .b32 off => some (.gaddr .shared off.toNat)
  | .shared, .b64 off => some (.gaddr .shared off.toNat)
  | .shared, .u32 off => some (.gaddr .shared off.toNat)
  | .shared, .u64 off => some (.gaddr .shared off.toNat)
  | .shared, .gaddr .shared off => some (.gaddr .shared off)
  | .local, .b32 off => some (.gaddr .local off.toNat)
  | .local, .b64 off => some (.gaddr .local off.toNat)
  | .local, .u32 off => some (.gaddr .local off.toNat)
  | .local, .u64 off => some (.gaddr .local off.toNat)
  | .local, .gaddr .local off => some (.gaddr .local off)
  | .param, .b32 off => some (.gaddr .param off.toNat)
  | .param, .b64 off => some (.gaddr .param off.toNat)
  | .param, .u32 off => some (.gaddr .param off.toNat)
  | .param, .u64 off => some (.gaddr .param off.toNat)
  | .param, .gaddr .param off => some (.gaddr .param off)
  | .const, .b32 off => some (.gaddr .const off.toNat)
  | .const, .b64 off => some (.gaddr .const off.toNat)
  | .const, .u32 off => some (.gaddr .const off.toNat)
  | .const, .u64 off => some (.gaddr .const off.toNat)
  | .const, .gaddr .const off => some (.gaddr .const off)
  | _, _ => none

def evalIsspacep? (space : AddrSpace) (value : Value) : Option Bool :=
  match value with
  | .gaddr s _ => some (s == space)
  | _ => none

mutual
  def evalRValue? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : RValue → Option Value
    | .imm v => some v
    | .reg r => do
        let laneState <- st.getLane? cta warp lane
        readReg laneState r
    | .pred p => do
        let laneState <- st.getLane? cta warp lane
        let b <- readPred laneState p
        pure (.pred b)
    | .special s => some <| evalSpecial st.kernelEnv.gridCtx cta warp lane s
    | .unop op a => do
        let va <- evalRValue? st cta warp lane a
        evalUnary? op va
    | .binop op a b => do
        let va <- evalRValue? st cta warp lane a
        let vb <- evalRValue? st cta warp lane b
        evalBinary? op va vb
    | .triop op a b c => do
        let va <- evalRValue? st cta warp lane a
        let vb <- evalRValue? st cta warp lane b
        let vc <- evalRValue? st cta warp lane c
        evalTernary? op va vb vc

  def evalUnary? : ScalarUnaryOp → Value → Option Value
    | .mov, v => some v
    | .neg, .s32 x => some (.s32 (normalizeSigned 32 (-x)))
    | .neg, .s64 x => some (.s64 (normalizeSigned 64 (-x)))
    | .neg, .f32 x => some (.f32 (-x))
    | .neg, .f64 x => some (.f64 (-x))
    | .abs, .s32 x => some (.s32 (normalizeSigned 32 (intAbs x)))
    | .abs, .s64 x => some (.s64 (normalizeSigned 64 (intAbs x)))
    | .abs, .f32 x => some (.f32 (Float.abs x))
    | .abs, .f64 x => some (.f64 (Float.abs x))
    | .bitnot, .u32 x => some (.u32 (~~~x))
    | .bitnot, .u64 x => some (.u64 (~~~x))
    | .bitnot, .b32 x => some (.b32 (~~~x))
    | .bitnot, .b64 x => some (.b64 (~~~x))
    | .cvt .u32, .s32 x => some (.u32 (UInt32.ofNat (signedToNat 32 x)))
    | .cvt .u32, .u64 x => some (.u32 (UInt32.ofNat x.toNat))
    | .cvt .u32, .b32 x => some (.u32 x)
    | .cvt .b32, .u32 x => some (.b32 x)
    | .cvt .b32, .s32 x => some (.b32 (UInt32.ofNat (signedToNat 32 x)))
    | .cvt .u64, .s64 x => some (.u64 (UInt64.ofNat (signedToNat 64 x)))
    | .cvt .u64, .u32 x => some (.u64 (UInt64.ofNat x.toNat))
    | .cvt .u64, .b64 x => some (.u64 x)
    | .cvt .b64, .u64 x => some (.b64 x)
    | .cvt .b64, .s64 x => some (.b64 (UInt64.ofNat (signedToNat 64 x)))
    | .cvt .s32, .u32 x => some (.s32 (normalizeSigned 32 (Int.ofNat x.toNat)))
    | .cvt .s32, .s64 x => some (.s32 (normalizeSigned 32 x))
    | .cvt .s32, .b32 x => some (.s32 (normalizeSigned 32 (Int.ofNat x.toNat)))
    | .cvt .s64, .u64 x => some (.s64 (normalizeSigned 64 (Int.ofNat x.toNat)))
    | .cvt .s64, .s32 x => some (.s64 (normalizeSigned 64 x))
    | .cvt .s64, .b64 x => some (.s64 (normalizeSigned 64 (Int.ofNat x.toNat)))
    | .cvt .f32, .u32 x => some (.f32 x.toFloat)
    | .cvt .f32, .s32 x => some (.f32 (intToFloat x))
    | .cvt .f64, .u64 x => some (.f64 x.toFloat)
    | .cvt .f64, .s64 x => some (.f64 (intToFloat x))
    | .cvt dst, v =>
        if let some ty := Typing.valueType? v then
          if dst = ty then some v else none
        else none
    | _, _ => none

  def evalBinary? : ScalarBinaryOp → Value → Value → Option Value
    | .mulWideS32, .s32 a, .s32 b => some (.s64 (normalizeSigned 64 (a * b)))
    | .bitor, .pred a, .pred b => some (.pred (a || b))
    | .bitand, .pred a, .pred b => some (.pred (a && b))
    | .bitxor, .pred a, .pred b => some (.pred (a != b))
    | .add, .u32 a, .u32 b => some (.u32 (a + b))
    | .add, .u64 a, .u64 b => some (.u64 (a + b))
    | .add, .s32 a, .s32 b => some (.s32 (normalizeSigned 32 (a + b)))
    | .add, .s64 a, .s64 b => some (.s64 (normalizeSigned 64 (a + b)))
    | .add, .gaddr s off, .s64 b => some (.gaddr s (Int.toNat (Int.ofNat off + b)))
    | .add, .s64 b, .gaddr s off => some (.gaddr s (Int.toNat (Int.ofNat off + b)))
    | .add, .gaddr s off, .u64 b => some (.gaddr s (off + b.toNat))
    | .add, .u64 b, .gaddr s off => some (.gaddr s (off + b.toNat))
    | .add, .f32 a, .f32 b => some (.f32 (a + b))
    | .add, .f64 a, .f64 b => some (.f64 (a + b))
    | .sub, .u32 a, .u32 b => some (.u32 (a - b))
    | .sub, .u64 a, .u64 b => some (.u64 (a - b))
    | .sub, .s32 a, .s32 b => some (.s32 (normalizeSigned 32 (a - b)))
    | .sub, .s64 a, .s64 b => some (.s64 (normalizeSigned 64 (a - b)))
    | .sub, .gaddr s off, .s64 b => some (.gaddr s (Int.toNat (Int.ofNat off - b)))
    | .sub, .gaddr s off, .u64 b => some (.gaddr s (off - b.toNat))
    | .sub, .f32 a, .f32 b => some (.f32 (a - b))
    | .sub, .f64 a, .f64 b => some (.f64 (a - b))
    | .mul, .u32 a, .u32 b => some (.u32 (a * b))
    | .mul, .u64 a, .u64 b => some (.u64 (a * b))
    | .mul, .s32 a, .s32 b => some (.s32 (normalizeSigned 32 (a * b)))
    | .mul, .s64 a, .s64 b => some (.s64 (normalizeSigned 64 (a * b)))
    | .mul, .f32 a, .f32 b => some (.f32 (a * b))
    | .mul, .f64 a, .f64 b => some (.f64 (a * b))
    | .bitand, .u32 a, .u32 b => some (.u32 (a &&& b))
    | .bitand, .u64 a, .u64 b => some (.u64 (a &&& b))
    | .bitand, .b32 a, .b32 b => some (.b32 (a &&& b))
    | .bitand, .u32 a, .b32 b => some (.b32 (a &&& b))
    | .bitand, .b32 a, .u32 b => some (.b32 (a &&& b))
    | .bitand, .b64 a, .b64 b => some (.b64 (a &&& b))
    | .bitand, .u64 a, .b64 b => some (.b64 (a &&& b))
    | .bitand, .b64 a, .u64 b => some (.b64 (a &&& b))
    | .bitor, .u32 a, .u32 b => some (.u32 (a ||| b))
    | .bitor, .u64 a, .u64 b => some (.u64 (a ||| b))
    | .bitor, .b32 a, .b32 b => some (.b32 (a ||| b))
    | .bitor, .u32 a, .b32 b => some (.b32 (a ||| b))
    | .bitor, .b32 a, .u32 b => some (.b32 (a ||| b))
    | .bitor, .b64 a, .b64 b => some (.b64 (a ||| b))
    | .bitor, .u64 a, .b64 b => some (.b64 (a ||| b))
    | .bitor, .b64 a, .u64 b => some (.b64 (a ||| b))
    | .bitxor, .u32 a, .u32 b => some (.u32 (u32Xor a b))
    | .bitxor, .u64 a, .u64 b => some (.u64 (u64Xor a b))
    | .bitxor, .b32 a, .b32 b => some (.b32 (u32Xor a b))
    | .bitxor, .u32 a, .b32 b => some (.b32 (u32Xor a b))
    | .bitxor, .b32 a, .u32 b => some (.b32 (u32Xor a b))
    | .bitxor, .b64 a, .b64 b => some (.b64 (u64Xor a b))
    | .bitxor, .u64 a, .b64 b => some (.b64 (u64Xor a b))
    | .bitxor, .b64 a, .u64 b => some (.b64 (u64Xor a b))
    | .shl, .u32 a, .u32 b => some (.u32 (u32Shl a b.toNat))
    | .shl, .u64 a, .u64 b => some (.u64 (u64Shl a b.toNat))
    | .shr, .u32 a, .u32 b => some (.u32 (u32Shr a b.toNat))
    | .shr, .u64 a, .u64 b => some (.u64 (u64Shr a b.toNat))
    | .min, .s32 a, .s32 b => some (.s32 (min a b))
    | .min, .s64 a, .s64 b => some (.s64 (min a b))
    | .max, .s32 a, .s32 b => some (.s32 (max a b))
    | .max, .s64 a, .s64 b => some (.s64 (max a b))
    | _, _, _ => none

  def evalTernary? : ScalarTernaryOp → Value → Value → Value → Option Value
    | .mad, .u32 a, .u32 b, .u32 c => some (.u32 (a * b + c))
    | .mad, .u64 a, .u64 b, .u64 c => some (.u64 (a * b + c))
    | .mad, .s32 a, .s32 b, .s32 c => some (.s32 (normalizeSigned 32 (a * b + c)))
    | .mad, .s64 a, .s64 b, .s64 c => some (.s64 (normalizeSigned 64 (a * b + c)))
    | .fma, .f32 a, .f32 b, .f32 c => some (.f32 (a * b + c))
    | .fma, .f64 a, .f64 b, .f64 c => some (.f64 (a * b + c))
    | .selp, a, _, .pred true => some a
    | .selp, _, b, .pred false => some b
    | _, _, _, _ => none
end

def evalCmp? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (cmp : CmpExpr) : Option Bool := do
  let lhs <- evalRValue? st cta warp lane cmp.lhs
  let rhs <- evalRValue? st cta warp lane cmp.rhs
  match cmp.op, lhs, rhs with
  | .eq, .u32 a, .u32 b => some (decide (a = b))
  | .eq, .u64 a, .u64 b => some (decide (a = b))
  | .eq, .s32 a, .s32 b => some (decide (a = b))
  | .eq, .s64 a, .s64 b => some (decide (a = b))
  | .eq, .f32 a, .f32 b => some (a == b)
  | .eq, .f64 a, .f64 b => some (a == b)
  | .ne, .u32 a, .u32 b => some (decide (a ≠ b))
  | .ne, .u64 a, .u64 b => some (decide (a ≠ b))
  | .ne, .s32 a, .s32 b => some (decide (a ≠ b))
  | .ne, .s64 a, .s64 b => some (decide (a ≠ b))
  | .ne, .f32 a, .f32 b => some (!(a == b))
  | .ne, .f64 a, .f64 b => some (!(a == b))
  | .lt, .u32 a, .u32 b => some (decide (a < b))
  | .lt, .u64 a, .u64 b => some (decide (a < b))
  | .lt, .s32 a, .s32 b => some (decide (a < b))
  | .lt, .s64 a, .s64 b => some (decide (a < b))
  | .lt, .f32 a, .f32 b => some (decide (a < b))
  | .lt, .f64 a, .f64 b => some (decide (a < b))
  | .le, .u32 a, .u32 b => some (decide (a ≤ b))
  | .le, .u64 a, .u64 b => some (decide (a ≤ b))
  | .le, .s32 a, .s32 b => some (decide (a ≤ b))
  | .le, .s64 a, .s64 b => some (decide (a ≤ b))
  | .le, .f32 a, .f32 b => some (decide (a ≤ b))
  | .le, .f64 a, .f64 b => some (decide (a ≤ b))
  | .gt, .u32 a, .u32 b => some (decide (a > b))
  | .gt, .u64 a, .u64 b => some (decide (a > b))
  | .gt, .s32 a, .s32 b => some (decide (a > b))
  | .gt, .s64 a, .s64 b => some (decide (a > b))
  | .gt, .f32 a, .f32 b => some (decide (a > b))
  | .gt, .f64 a, .f64 b => some (decide (a > b))
  | .ge, .u32 a, .u32 b => some (decide (a ≥ b))
  | .ge, .u64 a, .u64 b => some (decide (a ≥ b))
  | .ge, .s32 a, .s32 b => some (decide (a ≥ b))
  | .ge, .s64 a, .s64 b => some (decide (a ≥ b))
  | .ge, .f32 a, .f32 b => some (decide (a ≥ b))
  | .ge, .f64 a, .f64 b => some (decide (a ≥ b))
  | _, _, _ => none

def resolveAddr? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (ta : TypedAddr) : Option Addr := do
  let base <- evalRValue? st cta warp lane ta.addr
  match ta.space, base with
  | .global, .u64 off => some (.global off.toNat)
  | .global, .s64 off => some (.global off.toNat)
  | .global, .b64 off => some (.global off.toNat)
  | .shared, .u64 off => some (.shared cta off.toNat)
  | .shared, .s64 off => some (.shared cta off.toNat)
  | .shared, .b64 off => some (.shared cta off.toNat)
  | .local, .u64 off => some (.local cta warp lane off.toNat)
  | .local, .s64 off => some (.local cta warp lane off.toNat)
  | .local, .b64 off => some (.local cta warp lane off.toNat)
  | .param, .u64 off => some (.param off.toNat)
  | .param, .s64 off => some (.param off.toNat)
  | .param, .b64 off => some (.param off.toNat)
  | .const, .u64 off => some (.const off.toNat)
  | .const, .s64 off => some (.const off.toNat)
  | .const, .b64 off => some (.const off.toNat)
  | .global, .gaddr .global off => some (.global off)
  | .shared, .gaddr .shared off => some (.shared cta off)
  | .local, .gaddr .local off => some (.local cta warp lane off)
  | .param, .gaddr .param off => some (.param off)
  | .const, .gaddr .const off => some (.const off)
  | .generic, .gaddr s off => taggedGenericAddr? cta warp lane s off
  | .global, .u32 off => some (.global off.toNat)
  | .shared, .u32 off => some (.shared cta off.toNat)
  | .local, .u32 off => some (.local cta warp lane off.toNat)
  | .param, .u32 off => some (.param off.toNat)
  | .const, .u32 off => some (.const off.toNat)
  | _, _ => none

theorem resolveAddr?_matches_warp
    {st : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ta : TypedAddr} {resolved : Addr}
    (hresolve : resolveAddr? st cta warp lane ta = some resolved) :
    addrMatchesWarp cta warp resolved := by
  unfold resolveAddr? at hresolve
  cases heval : evalRValue? st cta warp lane ta.addr with
  | none =>
      simp [heval] at hresolve
  | some base =>
      simp [heval] at hresolve
      cases ta with
      | mk space ty addr =>
          cases space <;> cases base <;>
            simp [taggedGenericAddr?, addrMatchesWarp] at hresolve ⊢ <;>
              try (subst resolved <;> simp [addrMatchesWarp])
          all_goals
            try
              rename_i space' offset
              cases space' <;>
                simp [taggedGenericAddr?, addrMatchesWarp] at hresolve ⊢ <;>
                  try (subst resolved <;> simp [addrMatchesWarp])

def uniformBranchDestination? (st : State) (cta : CTAId) (warp : WarpId)
    (lanes : List LaneId) (cond : RValue) (tLabel fLabel : BlockLabel) : Option PC := do
  let mut dests : List PC := []
  for lane in lanes do
    let v <- evalRValue? st cta warp lane cond
    let b <- valueToBool? v
    dests := (if b then (tLabel, 0) else (fLabel, 0)) :: dests
  match dests.reverse with
  | [] => none
  | dest :: rest => if rest.all (fun pc' => pc' == dest) then some dest else none

def advancePcForLane (laneState : LaneState) : LaneState :=
  let (lbl, idx) := laneState.pc
  { laneState with pc := (lbl, idx + 1) }

theorem advancePcForLane_single_warp_control
    {warpState : WarpState} {pc : PC} {lane : LaneId} {laneState : LaneState}
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hpart : ParticipatingRunnable warpState none [lane])
    (hlane : warpState.getLane? lane = some laneState) :
    let warpFinal := warpState.setLane lane (advancePcForLane laneState)
    lockstepRunnable warpFinal ∧
      RunnablePc warpFinal (pc.1, pc.2 + 1) ∧
      ParticipatingRunnable warpFinal none [lane] := by
  intro warpFinal
  have hpc : laneState.pc = pc :=
    ParticipatingRunnable.single_lane_pc hrpc hpart hlane
  have hlanes : runnableLaneIds warpState = [lane] :=
    runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hstatus : (advancePcForLane laneState).status = laneState.status := by
    simp [advancePcForLane]
  have hrunFinal : runnableLaneIds warpFinal = [lane] := by
    dsimp [warpFinal]
    rw [runnableLaneIds_setLane_control_eq hlane hstatus, hlanes]
  have hgetFinal : warpFinal.getLane? lane = some (advancePcForLane laneState) := by
    dsimp [warpFinal]
    exact WarpState.getLane?_setLane_same hlane
  have hrpcFinal : RunnablePc warpFinal (pc.1, pc.2 + 1) := by
    unfold RunnablePc currentRunnablePc?
    rw [hrunFinal]
    simp [hgetFinal, advancePcForLane, hpc]
  have hpartFinal : ParticipatingRunnable warpFinal none [lane] := by
    unfold ParticipatingRunnable participatingRunnableLaneIds?
    change currentRunnablePc? warpFinal = some (pc.1, pc.2 + 1) at hrpcFinal
    rw [hrpcFinal]
    simp [hrunFinal, hgetFinal, participatingRunnableLaneIdsFrom?, advancePcForLane, hpc,
      guardHolds?]
  have hlockFinal : lockstepRunnable warpFinal := by
    unfold lockstepRunnable lockstepRunnable?
    change currentRunnablePc? warpFinal = some (pc.1, pc.2 + 1) at hrpcFinal
    rw [hrpcFinal]
    simp [hrunFinal, hgetFinal, advancePcForLane, hpc]
  exact ⟨hlockFinal, hrpcFinal, hpartFinal⟩

def applyToLaneIdsList? (st : State) (cta : CTAId) (warp : WarpId)
    (f : LaneId → LaneState → Option LaneState) : List LaneId → Option State
  | [] => some st
  | lane :: rest => do
      let laneState <- st.getLane? cta warp lane
      let laneState' <- f lane laneState
      let st' <- st.setLane cta warp lane laneState'
      applyToLaneIdsList? st' cta warp f rest

def applyToLaneIds? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId)
    (f : LaneId → LaneState → Option LaneState) : Option State :=
  applyToLaneIdsList? st cta warp f lanes

theorem applyToLaneIdsList?_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    st'.global = st.global := by
  induction lanes generalizing st with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      rfl
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  exact (ih happly).trans (State.setLane_global_eq hset)

theorem applyToLaneIds?_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    st'.global = st.global :=
  applyToLaneIdsList?_global_eq happly

theorem applyToLaneIdsList?_param_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    st'.param = st.param := by
  induction lanes generalizing st with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      rfl
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  exact (ih happly).trans (State.setLane_param_eq hset)

theorem applyToLaneIds?_param_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    st'.param = st.param :=
  applyToLaneIdsList?_param_eq happly

theorem applyToLaneIdsList?_const_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    st'.const = st.const := by
  induction lanes generalizing st with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      rfl
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  exact (ih happly).trans (State.setLane_const_eq hset)

theorem applyToLaneIds?_const_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    st'.const = st.const :=
  applyToLaneIdsList?_const_eq happly

theorem applyToLaneIdsList?_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    st'.kernelEnv = st.kernelEnv := by
  induction lanes generalizing st with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      rfl
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  exact (ih happly).trans (State.setLane_kernelEnv_eq hset)

theorem applyToLaneIds?_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    st'.kernelEnv = st.kernelEnv :=
  applyToLaneIdsList?_kernelEnv_eq happly

theorem applyToLaneIdsList?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new → new.status = old.status)
    (hwarp : st.getWarp? cta warp = some warpState)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState := by
  induction lanes generalizing st warpState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨warpState, hwarp, rfl⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  have hwarpNext :
                      stNext.getWarp? cta warp =
                        some (warpState.setLane lane laneState') :=
                    State.getWarp?_setLane_same hwarp hset
                  rcases ih (hwarp := hwarpNext) happly with
                    ⟨warpFinal, hwarpFinal, hrunFinal⟩
                  have hwarpLane : warpState.getLane? lane = some laneState := by
                    unfold State.getLane? at hget
                    simp [hwarp] at hget
                    exact hget
                  exact ⟨warpFinal, hwarpFinal,
                    hrunFinal.trans
                      (runnableLaneIds_setLane_control_eq hwarpLane
                        (hpres lane laneState laneState' hf))⟩

theorem applyToLaneIds?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {warpState : WarpState} {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new → new.status = old.status)
    (hwarp : st.getWarp? cta warp = some warpState)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState :=
  applyToLaneIdsList?_runnableLaneIds_eq hpres hwarp happly

theorem applyToLaneIdsList?_warp_control_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {pc : PC} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.status = old.status ∧ new.pc = old.pc)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' pc := by
  induction lanes generalizing st warpState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨warpState, hwarp, hlock, hrpc⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  rcases hpres lane laneState laneState' hf with ⟨hstatus, hpc⟩
                  have hwarpLane : warpState.getLane? lane = some laneState := by
                    unfold State.getLane? at hget
                    simp [hwarp] at hget
                    exact hget
                  have hwarpNext :
                      stNext.getWarp? cta warp =
                        some (warpState.setLane lane laneState') :=
                    State.getWarp?_setLane_same hwarp hset
                  have hlockNext :
                      lockstepRunnable (warpState.setLane lane laneState') :=
                    (lockstepRunnable_setLane_control_iff hwarpLane hstatus hpc).2 hlock
                  have hrpcNext :
                      RunnablePc (warpState.setLane lane laneState') pc := by
                    unfold RunnablePc
                    rw [currentRunnablePc?_setLane_control_eq hwarpLane hstatus hpc]
                    exact hrpc
                  exact ih hwarpNext hlockNext hrpcNext happly

theorem applyToLaneIds?_warp_control_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {warpState : WarpState} {pc : PC} {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.status = old.status ∧ new.pc = old.pc)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' pc :=
  applyToLaneIdsList?_warp_control_eq hpres hwarp hlock hrpc happly

theorem applyToLaneIdsList?_getLane_eq_of_not_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {f : LaneId → LaneState → Option LaneState}
    {lanes : List LaneId}
    (hnot : target ∉ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    st'.getLane? cta warp target = some targetState := by
  induction lanes generalizing st with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact htarget
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      have hne : target ≠ lane := by
        intro heq
        apply hnot
        simp [heq]
      have hnotRest : target ∉ rest := by
        intro hmem
        apply hnot
        simp [hmem]
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  have htargetNext :
                      stNext.getLane? cta warp target = some targetState := by
                    have htargetEq := State.getLane?_setLane_ne hget hne hset
                    rw [htarget] at htargetEq
                    exact htargetEq
                  exact ih hnotRest htargetNext happly

theorem applyToLaneIds?_getLane_eq_of_not_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {f : LaneId → LaneState → Option LaneState}
    {lanes : List LaneId}
    (hnot : target ∉ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    st'.getLane? cta warp target = some targetState :=
  applyToLaneIdsList?_getLane_eq_of_not_mem hnot htarget happly

theorem applyToLaneIdsList?_advance_lane_pc_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId} {pc : PC}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (hpc : targetState.pc = pc)
    (happly :
      applyToLaneIdsList? st cta warp
        (fun _ laneState => some (advancePcForLane laneState)) lanes = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.pc = (pc.1, pc.2 + 1) ∧
        targetState'.status = targetState.status := by
  induction lanes generalizing st targetState with
  | nil =>
      simp at hmem
  | cons lane rest ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          simp [applyToLaneIdsList?] at happly
          cases hget : st.getLane? cta warp lane with
          | none =>
              simp [hget] at happly
          | some laneState =>
              cases hset :
                  st.setLane cta warp lane (advancePcForLane laneState) with
              | none =>
                  simp [hget, hset] at happly
              | some stNext =>
                  simp [hget, hset] at happly
                  simp at hmem
                  rcases hmem with hsame | hmemRest
                  · subst target
                    rw [hget] at htarget
                    injection htarget with hstate
                    subst targetState
                    have hnotLaneRest : lane ∉ rest := by
                      intro hmemLane
                      exact (hnotMem lane hmemLane) rfl
                    have hlaneNext :
                        stNext.getLane? cta warp lane =
                          some (advancePcForLane laneState) :=
                      State.getLane?_setLane_same hget hset
                    have hlaneFinal :
                        st'.getLane? cta warp lane =
                          some (advancePcForLane laneState) :=
                      applyToLaneIdsList?_getLane_eq_of_not_mem hnotLaneRest hlaneNext happly
                    exact ⟨advancePcForLane laneState, hlaneFinal, by
                      simp [advancePcForLane, hpc], by
                      simp [advancePcForLane]⟩
                  · have hne : target ≠ lane := by
                      intro heq
                      subst target
                      exact (hnotMem lane hmemRest) rfl
                    have htargetNext :
                        stNext.getLane? cta warp target = some targetState := by
                      have htargetEq := State.getLane?_setLane_ne hget hne hset
                      rw [htarget] at htargetEq
                      exact htargetEq
                    exact ih hnodupRest hmemRest htargetNext hpc happly

theorem applyToLaneIds?_advance_lane_pc_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId} {pc : PC}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (hpc : targetState.pc = pc)
    (happly :
      applyToLaneIds? st cta warp lanes
        (fun _ laneState => some (advancePcForLane laneState)) = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.pc = (pc.1, pc.2 + 1) ∧
        targetState'.status = targetState.status :=
  applyToLaneIdsList?_advance_lane_pc_of_mem hnodup hmem htarget hpc happly

theorem applyToLaneIdsList?_set_lane_pc_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId} {pc' : PC}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly :
      applyToLaneIdsList? st cta warp
        (fun _ laneState => some { laneState with pc := pc' }) lanes = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.pc = pc' ∧
        targetState'.status = targetState.status := by
  induction lanes generalizing st targetState with
  | nil =>
      simp at hmem
  | cons lane rest ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          simp [applyToLaneIdsList?] at happly
          cases hget : st.getLane? cta warp lane with
          | none =>
              simp [hget] at happly
          | some laneState =>
              cases hset : st.setLane cta warp lane { laneState with pc := pc' } with
              | none =>
                  simp [hget, hset] at happly
              | some stNext =>
                  simp [hget, hset] at happly
                  simp at hmem
                  rcases hmem with hsame | hmemRest
                  · subst target
                    rw [hget] at htarget
                    injection htarget with hstate
                    subst targetState
                    have hnotLaneRest : lane ∉ rest := by
                      intro hmemLane
                      exact (hnotMem lane hmemLane) rfl
                    have hlaneNext :
                        stNext.getLane? cta warp lane =
                          some { laneState with pc := pc' } :=
                      State.getLane?_setLane_same hget hset
                    have hlaneFinal :
                        st'.getLane? cta warp lane =
                          some { laneState with pc := pc' } :=
                      applyToLaneIdsList?_getLane_eq_of_not_mem hnotLaneRest hlaneNext happly
                    exact ⟨{ laneState with pc := pc' }, hlaneFinal, by simp, by simp⟩
                  · have hne : target ≠ lane := by
                      intro heq
                      subst target
                      exact (hnotMem lane hmemRest) rfl
                    have htargetNext :
                        stNext.getLane? cta warp target = some targetState := by
                      have htargetEq := State.getLane?_setLane_ne hget hne hset
                      rw [htarget] at htargetEq
                      exact htargetEq
                    exact ih hnodupRest hmemRest htargetNext happly

theorem applyToLaneIds?_set_lane_pc_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId} {pc' : PC}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly :
      applyToLaneIds? st cta warp lanes
        (fun _ laneState => some { laneState with pc := pc' }) = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.pc = pc' ∧
        targetState'.status = targetState.status :=
  applyToLaneIdsList?_set_lane_pc_of_mem hnodup hmem htarget happly

theorem applyToLaneIdsList?_terminate_lane_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly :
      applyToLaneIdsList? st cta warp
        (fun _ laneState => some { laneState with status := .terminated }) lanes = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.status = .terminated ∧
        targetState'.pc = targetState.pc := by
  induction lanes generalizing st targetState with
  | nil =>
      simp at hmem
  | cons lane rest ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          simp [applyToLaneIdsList?] at happly
          cases hget : st.getLane? cta warp lane with
          | none =>
              simp [hget] at happly
          | some laneState =>
              cases hset : st.setLane cta warp lane { laneState with status := .terminated } with
              | none =>
                  simp [hget, hset] at happly
              | some stNext =>
                  simp [hget, hset] at happly
                  simp at hmem
                  rcases hmem with hsame | hmemRest
                  · subst target
                    rw [hget] at htarget
                    injection htarget with hstate
                    subst targetState
                    have hnotLaneRest : lane ∉ rest := by
                      intro hmemLane
                      exact (hnotMem lane hmemLane) rfl
                    have hlaneNext :
                        stNext.getLane? cta warp lane =
                          some { laneState with status := .terminated } :=
                      State.getLane?_setLane_same hget hset
                    have hlaneFinal :
                        st'.getLane? cta warp lane =
                          some { laneState with status := .terminated } :=
                      applyToLaneIdsList?_getLane_eq_of_not_mem hnotLaneRest hlaneNext happly
                    exact ⟨{ laneState with status := .terminated }, hlaneFinal, by simp, by simp⟩
                  · have hne : target ≠ lane := by
                      intro heq
                      subst target
                      exact (hnotMem lane hmemRest) rfl
                    have htargetNext :
                        stNext.getLane? cta warp target = some targetState := by
                      have htargetEq := State.getLane?_setLane_ne hget hne hset
                      rw [htarget] at htargetEq
                      exact htargetEq
                    exact ih hnodupRest hmemRest htargetNext happly

theorem applyToLaneIds?_terminate_lane_of_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {lanes : List LaneId}
    (hnodup : lanes.Nodup)
    (hmem : target ∈ lanes)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly :
      applyToLaneIds? st cta warp lanes
        (fun _ laneState => some { laneState with status := .terminated }) = some st') :
    ∃ targetState',
      st'.getLane? cta warp target = some targetState' ∧
        targetState'.status = .terminated ∧
        targetState'.pc = targetState.pc :=
  applyToLaneIdsList?_terminate_lane_of_mem hnodup hmem htarget happly

theorem applyToLaneIdsList?_shared_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
      ctaState'.shared = ctaState.shared := by
  induction lanes generalizing st ctaState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨ctaState, hcta, rfl⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  rcases State.setLane_getCTA_shared_eq hcta hset with
                    ⟨ctaNext, hctaNext, hsharedNext⟩
                  rcases ih hctaNext happly with ⟨ctaFinal, hctaFinal, hsharedFinal⟩
                  exact ⟨ctaFinal, hctaFinal, hsharedFinal.trans hsharedNext⟩

theorem applyToLaneIds?_shared_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
      ctaState'.shared = ctaState.shared :=
  applyToLaneIdsList?_shared_eq hcta happly

theorem applyToLaneIdsList?_lane_nonPc_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.localMem = old.localMem ∧ new.regs = old.regs ∧ new.preds = old.preds)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem ∧
      targetState'.regs = targetState.regs ∧
      targetState'.preds = targetState.preds := by
  induction lanes generalizing st targetState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨targetState, htarget, rfl, rfl, rfl⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  rcases hpres lane laneState laneState' hf with
                    ⟨hlocal, hregs, hpreds⟩
                  rcases State.getLane?_setLane_nonPc_eq
                      htarget hget hlocal hregs hpreds hset with
                    ⟨targetNext, htargetNext, hlocalNext, hregsNext, hpredsNext⟩
                  rcases ih htargetNext happly with
                    ⟨targetFinal, htargetFinal, hlocalFinal, hregsFinal, hpredsFinal⟩
                  exact ⟨targetFinal, htargetFinal,
                    hlocalFinal.trans hlocalNext,
                    hregsFinal.trans hregsNext,
                    hpredsFinal.trans hpredsNext⟩

theorem applyToLaneIds?_lane_nonPc_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {target : LaneId} {targetState : LaneState}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.localMem = old.localMem ∧ new.regs = old.regs ∧ new.preds = old.preds)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem ∧
      targetState'.regs = targetState.regs ∧
      targetState'.preds = targetState.preds :=
  applyToLaneIdsList?_lane_nonPc_eq hpres htarget happly

theorem applyToLaneIdsList?_lane_reg_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {name : RegName} {value : Value}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new →
        old.regs[name]? = some value → new.regs[name]? = some value)
    (htarget : st.getLane? cta warp target = some targetState)
    (hread : targetState.regs[name]? = some value)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.regs[name]? = some value := by
  induction lanes generalizing st targetState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨targetState, htarget, hread⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  by_cases hsame : target = lane
                  · subst target
                    rw [hget] at htarget
                    injection htarget with hstate
                    subst targetState
                    have htargetNext :
                        stNext.getLane? cta warp lane = some laneState' :=
                      State.getLane?_setLane_same hget hset
                    have hreadNext : laneState'.regs[name]? = some value :=
                      hpres lane laneState laneState' hf hread
                    exact ih htargetNext hreadNext happly
                  · have htargetNext :
                        stNext.getLane? cta warp target = some targetState := by
                      have htargetEq := State.getLane?_setLane_ne hget hsame hset
                      rw [htarget] at htargetEq
                      exact htargetEq
                    exact ih htargetNext hread happly

theorem applyToLaneIds?_lane_reg_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {target : LaneId} {targetState : LaneState} {name : RegName} {value : Value}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        old.regs[name]? = some value → new.regs[name]? = some value)
    (htarget : st.getLane? cta warp target = some targetState)
    (hread : targetState.regs[name]? = some value)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.regs[name]? = some value :=
  applyToLaneIdsList?_lane_reg_eq hpres htarget hread happly

theorem applyToLaneIdsList?_lane_pred_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState} {name : PredName} {value : Bool}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new →
        old.preds[name]? = some value → new.preds[name]? = some value)
    (htarget : st.getLane? cta warp target = some targetState)
    (hread : targetState.preds[name]? = some value)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.preds[name]? = some value := by
  induction lanes generalizing st targetState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨targetState, htarget, hread⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  by_cases hsame : target = lane
                  · subst target
                    rw [hget] at htarget
                    injection htarget with hstate
                    subst targetState
                    have htargetNext :
                        stNext.getLane? cta warp lane = some laneState' :=
                      State.getLane?_setLane_same hget hset
                    have hreadNext : laneState'.preds[name]? = some value :=
                      hpres lane laneState laneState' hf hread
                    exact ih htargetNext hreadNext happly
                  · have htargetNext :
                        stNext.getLane? cta warp target = some targetState := by
                      have htargetEq := State.getLane?_setLane_ne hget hsame hset
                      rw [htarget] at htargetEq
                      exact htargetEq
                    exact ih htargetNext hread happly

theorem applyToLaneIds?_lane_pred_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {target : LaneId} {targetState : LaneState} {name : PredName} {value : Bool}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        old.preds[name]? = some value → new.preds[name]? = some value)
    (htarget : st.getLane? cta warp target = some targetState)
    (hread : targetState.preds[name]? = some value)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.preds[name]? = some value :=
  applyToLaneIdsList?_lane_pred_eq hpres htarget hread happly

theorem applyToLaneIdsList?_lane_localMem_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {target : LaneId}
    {targetState : LaneState}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hpres :
      ∀ lane old new, f lane old = some new → new.localMem = old.localMem)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIdsList? st cta warp f lanes = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem := by
  induction lanes generalizing st targetState with
  | nil =>
      simp [applyToLaneIdsList?] at happly
      subst st'
      exact ⟨targetState, htarget, rfl⟩
  | cons lane rest ih =>
      simp [applyToLaneIdsList?] at happly
      cases hget : st.getLane? cta warp lane with
      | none =>
          simp [hget] at happly
      | some laneState =>
          cases hf : f lane laneState with
          | none =>
              simp [hget, hf] at happly
          | some laneState' =>
              cases hset : st.setLane cta warp lane laneState' with
              | none =>
                  simp [hget, hf, hset] at happly
              | some stNext =>
                  simp [hget, hf, hset] at happly
                  have hlocal := hpres lane laneState laneState' hf
                  rcases State.getLane?_setLane_localMem_eq
                      htarget hget hlocal hset with
                    ⟨targetNext, htargetNext, hlocalNext⟩
                  rcases ih htargetNext happly with
                    ⟨targetFinal, htargetFinal, hlocalFinal⟩
                  exact ⟨targetFinal, htargetFinal, hlocalFinal.trans hlocalNext⟩

theorem applyToLaneIds?_lane_localMem_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {target : LaneId} {targetState : LaneState}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new → new.localMem = old.localMem)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : applyToLaneIds? st cta warp lanes f = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem :=
  applyToLaneIdsList?_lane_localMem_eq hpres htarget happly

def advanceRunnablePcs? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  let warpState <- st.getWarp? cta warp
  let lanes <- participatingRunnableLaneIds? warpState none
  applyToLaneIds? st cta warp lanes (fun _ laneState => some (advancePcForLane laneState))

theorem advanceRunnablePcs?_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    st'.global = st.global := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_global_eq hadvance

theorem advanceRunnablePcs?_param_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    st'.param = st.param := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_param_eq hadvance

theorem advanceRunnablePcs?_const_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    st'.const = st.const := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_const_eq hadvance

theorem advanceRunnablePcs?_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_kernelEnv_eq hadvance

theorem advanceRunnablePcs?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState := by
  unfold advanceRunnablePcs? at hadvance
  have hpart :
      participatingRunnableLaneIds? warpState none = some (runnableLaneIds warpState) :=
    participatingRunnable_none_eq_runnableLaneIds_of_lockstep hlock hrpc
  simp [hwarp, hpart] at hadvance
  exact applyToLaneIds?_runnableLaneIds_eq
    (hpres := by
      intro lane old new hf
      injection hf with hnew
      subst new
      simp [advancePcForLane])
    hwarp hadvance

theorem advanceRunnablePcs?_shared_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
      ctaState'.shared = ctaState.shared := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_shared_eq hcta hadvance

theorem advanceRunnablePcs?_lane_nonPc_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hlane : st.getLane? cta warp lane = some laneState)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds := by
  unfold advanceRunnablePcs? at hadvance
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hadvance
  | some warpState =>
      simp [hwarp] at hadvance
      cases hpart : participatingRunnableLaneIds? warpState none with
      | none =>
          simp [hpart] at hadvance
      | some lanes =>
          simp [hpart] at hadvance
          exact applyToLaneIds?_lane_nonPc_eq
            (hpres := by
              intro lane old new hf
              injection hf with hnew
              subst new
              simp [advancePcForLane])
            hlane hadvance

theorem advanceRunnablePcs?_single_lane_pc
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {warpState : WarpState} {laneState : LaneState} {pc : PC}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hpart : ParticipatingRunnable warpState none [lane])
    (hlane : st.getLane? cta warp lane = some laneState)
    (hpc : laneState.pc = pc)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ laneState',
      st'.getLane? cta warp lane = some laneState' ∧
        laneState'.pc = (pc.1, pc.2 + 1) ∧ laneState'.status = laneState.status := by
  unfold advanceRunnablePcs? at hadvance
  have hpartOpt :
      participatingRunnableLaneIds? warpState none = some [lane] :=
    (participatingRunnable_iff_bool warpState none [lane]).1 hpart
  simp [hwarp, hpartOpt] at hadvance
  unfold applyToLaneIds? applyToLaneIdsList? at hadvance
  simp [hlane] at hadvance
  cases hset : st.setLane cta warp lane (advancePcForLane laneState) with
  | none =>
      simp [hset] at hadvance
  | some stCore =>
      simp [hset] at hadvance
      injection hadvance with hst'
      subst st'
      refine ⟨advancePcForLane laneState, ?_, ?_, ?_⟩
      · exact State.getLane?_setLane_same hlane hset
      · subst pc
        simp [advancePcForLane]
      · simp [advancePcForLane]

theorem advanceRunnablePcs?_single_warp_control
    {st st' : State} {cta : CTAId} {warp : WarpId} {pc : PC} {lane : LaneId}
    {warpState : WarpState} {laneState : LaneState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hpart : ParticipatingRunnable warpState none [lane])
    (hlane : st.getLane? cta warp lane = some laneState)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' (pc.1, pc.2 + 1) ∧
        ParticipatingRunnable warpState' none [lane] := by
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  unfold advanceRunnablePcs? at hadvance
  have hpartOpt : participatingRunnableLaneIds? warpState none = some [lane] :=
    (participatingRunnable_iff_bool warpState none [lane]).1 hpart
  simp [hwarp, hpartOpt] at hadvance
  unfold applyToLaneIds? applyToLaneIdsList? at hadvance
  simp [hlane] at hadvance
  cases hset : st.setLane cta warp lane (advancePcForLane laneState) with
  | none =>
      simp [hset] at hadvance
  | some stCore =>
      simp [hset] at hadvance
      injection hadvance with hst'
      subst st'
      have hwarpFinal :
          stCore.getWarp? cta warp =
            some (warpState.setLane lane (advancePcForLane laneState)) :=
        State.getWarp?_setLane_same hwarp hset
      rcases advancePcForLane_single_warp_control hlock hrpc hpart hwarpLane with
        ⟨hlockFinal, hrpcFinal, hpartFinal⟩
      exact ⟨warpState.setLane lane (advancePcForLane laneState), hwarpFinal,
        hlockFinal, hrpcFinal, hpartFinal⟩

theorem advanceRunnablePcs?_warp_control
    {st st' : State} {cta : CTAId} {warp : WarpId} {pc : PC}
    {warpState : WarpState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' (pc.1, pc.2 + 1) := by
  unfold advanceRunnablePcs? at hadvance
  have hpart :
      participatingRunnableLaneIds? warpState none = some (runnableLaneIds warpState) :=
    participatingRunnable_none_eq_runnableLaneIds_of_lockstep hlock hrpc
  simp [hwarp, hpart] at hadvance
  rcases applyToLaneIds?_runnableLaneIds_eq
      (hpres := by
        intro lane old new hf
        injection hf with hnew
        subst new
        simp [advancePcForLane])
      hwarp hadvance with
    ⟨warpFinal, hwarpFinal, hrunFinal⟩
  have hnodup : (runnableLaneIds warpState).Nodup :=
    runnableLaneIds_nodup warpState
  have hrpcFinal : RunnablePc warpFinal (pc.1, pc.2 + 1) := by
    unfold RunnablePc currentRunnablePc?
    cases hlanes : runnableLaneIds warpState with
    | nil =>
        unfold RunnablePc currentRunnablePc? at hrpc
        simp [hlanes] at hrpc
    | cons first rest =>
        rw [hrunFinal, hlanes]
        have hmemFirst : first ∈ runnableLaneIds warpState := by
          rw [hlanes]
          simp
        rcases runnableLaneIds_mem_getLane hmemFirst with
          ⟨firstState, hfirstWarp⟩
        have hfirstPc : firstState.pc = pc :=
          lockstepRunnable_mem_pc hlock hrpc hmemFirst hfirstWarp
        have hfirstSt : st.getLane? cta warp first = some firstState := by
          unfold State.getLane?
          simp [hwarp, hfirstWarp]
        rcases applyToLaneIds?_advance_lane_pc_of_mem
            hnodup hmemFirst hfirstSt hfirstPc hadvance with
          ⟨firstFinal, hfirstFinal, hpcFinal, _hstatusFinal⟩
        unfold State.getLane? at hfirstFinal
        simp [hwarpFinal] at hfirstFinal
        simp [hfirstFinal, hpcFinal]
  have hlockFinal : lockstepRunnable warpFinal := by
    unfold lockstepRunnable lockstepRunnable?
    change currentRunnablePc? warpFinal = some (pc.1, pc.2 + 1) at hrpcFinal
    rw [hrpcFinal]
    simp
    intro lane hmemFinal
    have hmemInitial : lane ∈ runnableLaneIds warpState := by
      rw [← hrunFinal]
      exact hmemFinal
    rcases runnableLaneIds_mem_getLane hmemInitial with
      ⟨laneState, hlaneWarp⟩
    have hlanePc : laneState.pc = pc :=
      lockstepRunnable_mem_pc hlock hrpc hmemInitial hlaneWarp
    have hlaneSt : st.getLane? cta warp lane = some laneState := by
      unfold State.getLane?
      simp [hwarp, hlaneWarp]
    rcases applyToLaneIds?_advance_lane_pc_of_mem
        hnodup hmemInitial hlaneSt hlanePc hadvance with
      ⟨laneFinal, hlaneFinal, hpcFinal, _hstatusFinal⟩
    unfold State.getLane? at hlaneFinal
    simp [hwarpFinal] at hlaneFinal
    simp [hlaneFinal, hpcFinal]
  exact ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩

theorem advanceRunnablePcs?_lane_localMem_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hlane : st.getLane? cta warp lane = some laneState)
    (hadvance : advanceRunnablePcs? st cta warp = some st') :
    ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
      laneState'.localMem = laneState.localMem := by
  rcases advanceRunnablePcs?_lane_nonPc_eq hlane hadvance with
    ⟨laneState', hlane', hlocal, _, _⟩
  exact ⟨laneState', hlane', hlocal⟩

def barrierArrivalPresent (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) : Bool :=
  arrived.any fun token' => token' == token

def addBarrierArrival (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) :
    List (WarpId × LaneId) :=
  if barrierArrivalPresent token arrived then arrived else token :: arrived

def barrierExpectedCount (st : State) (inst : BarrierInstance) : Nat :=
  if inst.expectedCount = 0 then st.kernelEnv.gridCtx.blockDim.x else inst.expectedCount

def setBarrierInstance? (st : State) (cta : CTAId) (barrierId : Nat)
    (inst : BarrierInstance) : Option State := do
  let ctaState <- st.getCTA? cta
  let barrier := { ctaState.barrier with bars := ctaState.barrier.bars.insert barrierId inst }
  pure <| st.setCTA cta { ctaState with barrier := barrier }

def stepBarrierCTA? (st : State) (cta : CTAId) (warp : WarpId)
    (barrierId : Nat) (participants : List LaneId) : Option State := do
  let warpState <- st.getWarp? cta warp
  let pc <- currentRunnablePc? warpState
  let currentLanes := runnableLaneIds warpState |>.filter fun lane =>
    match warpState.getLane? lane with
    | some laneState => laneState.pc == pc
    | none => false

  -- Predicated-off lanes skip the barrier instruction; arriving lanes block until release.
  let mut cur := st
  for lane in currentLanes do
    if !participants.contains lane then
      let laneState <- cur.getLane? cta warp lane
      cur <- cur.setLane cta warp lane (advancePcForLane laneState)

  let ctaState <- cur.getCTA? cta
  let inst0 := ctaState.barrier.bars[barrierId]?.getD {}
  let expected := barrierExpectedCount cur inst0
  let mut arrived := inst0.arrived
  for lane in participants do
    let token := (warp, lane)
    arrived := addBarrierArrival token arrived
    let laneState <- cur.getLane? cta warp lane
    cur <- cur.setLane cta warp lane { laneState with status := .blockedBarrier }

  if arrived.length >= expected then
    for token in arrived do
      let laneState <- cur.getLane? cta token.1 token.2
      cur <- cur.setLane cta token.1 token.2 { advancePcForLane laneState with status := .running }
    setBarrierInstance? cur cta barrierId { epoch := inst0.epoch + 1, arrived := [], expectedCount := expected }
  else
    setBarrierInstance? cur cta barrierId { inst0 with arrived := arrived, expectedCount := expected }

def stepStoreLanes? (st : State) (cta : CTAId) (warp : WarpId)
    (lanes : List LaneId) (dst : TypedAddr) (value : RValue) : Option State :=
  match lanes with
  | [] => some st
  | lane :: rest => do
      let addr <- resolveAddr? st cta warp lane dst
      let v <- evalRValue? st cta warp lane value
      let st' <- writeMem? st dst.space dst.ty addr v
      stepStoreLanes? st' cta warp rest dst value

theorem stepStoreLanes?_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hstep : stepStoreLanes? st cta warp lanes dst value = some st') :
    st'.kernelEnv = st.kernelEnv := by
  induction lanes generalizing st with
  | nil =>
      simp [stepStoreLanes?] at hstep
      subst st'
      rfl
  | cons lane rest ih =>
      simp [stepStoreLanes?] at hstep
      cases haddr : resolveAddr? st cta warp lane dst with
      | none =>
          simp [haddr] at hstep
      | some addr =>
          simp [haddr] at hstep
          cases hvalue : evalRValue? st cta warp lane value with
          | none =>
              simp [hvalue] at hstep
          | some v =>
              simp [hvalue] at hstep
              cases hwrite : writeMem? st dst.space dst.ty addr v with
              | none =>
                  simp [hwrite] at hstep
              | some stNext =>
                  simp [hwrite] at hstep
                  exact (ih hstep).trans (writeMem?_kernelEnv_eq hwrite)

theorem stepStoreLanes?_warp_control_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {pc : PC} {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hstep : stepStoreLanes? st cta warp lanes dst value = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        lockstepRunnable warpState' ∧
        RunnablePc warpState' pc := by
  induction lanes generalizing st warpState with
  | nil =>
      simp [stepStoreLanes?] at hstep
      subst st'
      exact ⟨warpState, hwarp, hlock, hrpc⟩
  | cons lane rest ih =>
      simp [stepStoreLanes?] at hstep
      cases haddr : resolveAddr? st cta warp lane dst with
      | none =>
          simp [haddr] at hstep
      | some addr =>
          simp [haddr] at hstep
          cases hvalue : evalRValue? st cta warp lane value with
          | none =>
              simp [hvalue] at hstep
          | some v =>
              simp [hvalue] at hstep
              cases hwrite : writeMem? st dst.space dst.ty addr v with
              | none =>
                  simp [hwrite] at hstep
              | some stNext =>
                  simp [hwrite] at hstep
                  have haddrMatch : addrMatchesWarp cta warp addr :=
                    resolveAddr?_matches_warp haddr
                  rcases writeMem?_warp_control_eq haddrMatch hwrite hwarp hlock hrpc with
                    ⟨warpNext, hwarpNext, hlockNext, hrpcNext⟩
                  exact ih hwarpNext hlockNext hrpcNext hstep

theorem stepStoreLanes?_runnableLaneIds_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hstep : stepStoreLanes? st cta warp lanes dst value = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        runnableLaneIds warpState' = runnableLaneIds warpState := by
  induction lanes generalizing st warpState with
  | nil =>
      simp [stepStoreLanes?] at hstep
      subst st'
      exact ⟨warpState, hwarp, rfl⟩
  | cons lane rest ih =>
      simp [stepStoreLanes?] at hstep
      cases haddr : resolveAddr? st cta warp lane dst with
      | none =>
          simp [haddr] at hstep
      | some addr =>
          simp [haddr] at hstep
          cases hvalue : evalRValue? st cta warp lane value with
          | none =>
              simp [hvalue] at hstep
          | some v =>
              simp [hvalue] at hstep
              cases hwrite : writeMem? st dst.space dst.ty addr v with
              | none =>
                  simp [hwrite] at hstep
              | some stNext =>
                  simp [hwrite] at hstep
                  have haddrMatch : addrMatchesWarp cta warp addr :=
                    resolveAddr?_matches_warp haddr
                  rcases writeMem?_runnableLaneIds_eq haddrMatch hwrite hwarp with
                    ⟨warpNext, hwarpNext, hrunNext⟩
                  rcases ih hwarpNext hstep with
                    ⟨warpFinal, hwarpFinal, hrunFinal⟩
                  exact ⟨warpFinal, hwarpFinal, hrunFinal.trans hrunNext⟩

def stepInstr? (st : State) (cta : CTAId) (warp : WarpId) (gi : GInstr) : Option State := do
  let warpState <- st.getWarp? cta warp
  if !lockstepRunnable? warpState then
    none
  else
    let participants <- participatingRunnableLaneIds? warpState gi.guard?
    match gi.instr with
    | .barrierCTA barrierId => stepBarrierCTA? st cta warp barrierId participants
    | _ => do
        let st <- match gi.instr with
          | .assignReg dst rhs =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let v <- evalRValue? st cta warp lane rhs
                pure (writeReg laneState dst v)
          | .assignPred dst cmp =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let b <- evalCmp? st cta warp lane cmp
                pure (writePred laneState dst b)
          | .assignPredValue dst rhs =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let v <- evalRValue? st cta warp lane rhs
                let b <- valueToBool? v
                pure (writePred laneState dst b)
          | .load dst src =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let addr <- resolveAddr? st cta warp lane src
                let value <- readMem? st src.space src.ty addr
                pure (writeReg laneState dst value)
          | .store dst value =>
              -- Milestone 1 deterministic simplification: stores are sequentialized in lane order.
              stepStoreLanes? st cta warp participants dst value
          | .cvta dst space src =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let value <- evalRValue? st cta warp lane src
                let gaddr <- evalCvta? space value
                pure (writeReg laneState dst gaddr)
          | .isspacep dst space src =>
              applyToLaneIds? st cta warp participants fun lane laneState => do
                let value <- evalRValue? st cta warp lane src
                let b <- evalIsspacep? space value
                pure (writePred laneState dst b)
          | .barrierCTA _ => none
          | .warp _ => none
          | .atomic _ _ _ _ _ => none
          | .mma _ => none
        advanceRunnablePcs? st cta warp

def instrUsesOrdinaryPcAdvance : Instr → Bool
  | .assignReg _ _ => true
  | .assignPred _ _ => true
  | .assignPredValue _ _ => true
  | .load _ _ => true
  | .store _ _ => true
  | .cvta _ _ _ => true
  | .isspacep _ _ _ => true
  | .barrierCTA _ => false
  | .warp _ => false
  | .atomic _ _ _ _ _ => false
  | .mma _ => false

theorem stepInstr?_assignReg_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : RegName} {rhs : RValue}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (evalRValue? st cta warp lane rhs).bind fun v =>
                        some (writeReg laneState dst v)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_load_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : RegName} {src : TypedAddr}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .load dst src } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (resolveAddr? st cta warp lane src).bind fun addr =>
                        (readMem? st src.space src.ty addr).bind fun value =>
                          some (writeReg laneState dst value)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_assignPred_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : PredName} {cmp : CmpExpr}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (evalCmp? st cta warp lane cmp).bind fun b =>
                        some (writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_assignPredValue_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : PredName} {rhs : RValue}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .assignPredValue dst rhs } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (evalRValue? st cta warp lane rhs).bind fun v =>
                        (valueToBool? v).bind fun b =>
                          some (writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_cvta_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : RegName} {space : AddrSpace} {src : RValue}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (evalRValue? st cta warp lane src).bind fun value =>
                        (evalCvta? space value).bind fun gaddr =>
                          some (writeReg laneState dst gaddr)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_isspacep_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {dst : PredName} {space : AddrSpace} {src : RValue}
    (hstep : stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } =
      some st') :
    st'.global = st.global := by
  unfold stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (evalRValue? st cta warp lane src).bind fun value =>
                        (evalIsspacep? space value).bind fun b =>
                          some (writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  exact (advanceRunnablePcs?_global_eq hstep).trans
                    (applyToLaneIds?_global_eq hcore)

theorem stepInstr?_kernelEnv_eq_of_ordinary
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hordinary : instrUsesOrdinaryPcAdvance gi.instr = true)
    (hstep : stepInstr? st cta warp gi = some st') :
    st'.kernelEnv = st.kernelEnv := by
  cases gi with
  | mk guard? instr =>
      unfold stepInstr? at hstep
      cases hwarp : st.getWarp? cta warp with
      | none =>
          simp [hwarp] at hstep
      | some warpState =>
          simp [hwarp] at hstep
          cases hlock : lockstepRunnable? warpState with
          | false =>
              simp [hlock] at hstep
          | true =>
              simp [hlock] at hstep
              cases hpart : participatingRunnableLaneIds? warpState guard? with
              | none =>
                  simp [hpart] at hstep
              | some participants =>
                  simp [hpart] at hstep
                  cases instr <;> simp [instrUsesOrdinaryPcAdvance] at hordinary
                  · rename_i dst rhs
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane rhs).bind fun v =>
                              some (writeReg laneState dst v)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)
                  · rename_i dst cmp
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalCmp? st cta warp lane cmp).bind fun b =>
                              some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)
                  · rename_i dst rhs
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane rhs).bind fun v =>
                              (valueToBool? v).bind fun b =>
                                some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)
                  · rename_i dst src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (resolveAddr? st cta warp lane src).bind fun addr =>
                              (readMem? st src.space src.ty addr).bind fun value =>
                                some (writeReg laneState dst value)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)
                  · rename_i dst value
                    cases hcore : stepStoreLanes? st cta warp participants dst value with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (stepStoreLanes?_kernelEnv_eq hcore)
                  · rename_i dst space src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane src).bind fun value =>
                              (evalCvta? space value).bind fun gaddr =>
                                some (writeReg laneState dst gaddr)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)
                  · rename_i dst space src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane src).bind fun value =>
                              (evalIsspacep? space value).bind fun b =>
                                some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact (advanceRunnablePcs?_kernelEnv_eq hstep).trans
                          (applyToLaneIds?_kernelEnv_eq hcore)

theorem stepInstr?_ordinary_factors_advance
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hordinary : instrUsesOrdinaryPcAdvance gi.instr = true)
    (hstep : stepInstr? st cta warp gi = some st') :
    ∃ stCore,
      advanceRunnablePcs? stCore cta warp = some st' ∧
        stCore.kernelEnv = st.kernelEnv := by
  cases gi with
  | mk guard? instr =>
      unfold stepInstr? at hstep
      cases hwarp : st.getWarp? cta warp with
      | none =>
          simp [hwarp] at hstep
      | some warpState =>
          simp [hwarp] at hstep
          cases hlock : lockstepRunnable? warpState with
          | false =>
              simp [hlock] at hstep
          | true =>
              simp [hlock] at hstep
              cases hpart : participatingRunnableLaneIds? warpState guard? with
              | none =>
                  simp [hpart] at hstep
              | some participants =>
                  simp [hpart] at hstep
                  cases instr <;> simp [instrUsesOrdinaryPcAdvance] at hordinary
                  · rename_i dst rhs
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane rhs).bind fun v =>
                              some (writeReg laneState dst v)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩
                  · rename_i dst cmp
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalCmp? st cta warp lane cmp).bind fun b =>
                              some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩
                  · rename_i dst rhs
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane rhs).bind fun v =>
                              (valueToBool? v).bind fun b =>
                                some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩
                  · rename_i dst src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (resolveAddr? st cta warp lane src).bind fun addr =>
                              (readMem? st src.space src.ty addr).bind fun value =>
                                some (writeReg laneState dst value)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩
                  · rename_i dst value
                    cases hcore : stepStoreLanes? st cta warp participants dst value with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, stepStoreLanes?_kernelEnv_eq hcore⟩
                  · rename_i dst space src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane src).bind fun value =>
                              (evalCvta? space value).bind fun gaddr =>
                                some (writeReg laneState dst gaddr)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩
                  · rename_i dst space src
                    cases hcore :
                        applyToLaneIds? st cta warp participants
                          (fun lane laneState =>
                            (evalRValue? st cta warp lane src).bind fun value =>
                              (evalIsspacep? space value).bind fun b =>
                                some (writePred laneState dst b)) with
                    | none =>
                        simp [hcore] at hstep
                    | some stCore =>
                        simp [hcore] at hstep
                        exact ⟨stCore, hstep, applyToLaneIds?_kernelEnv_eq hcore⟩

theorem stepInstr?_ordinary_core_control
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {warpState : WarpState} {pc : PC}
    (hordinary : instrUsesOrdinaryPcAdvance gi.instr = true)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : lockstepRunnable warpState)
    (hrpc : RunnablePc warpState pc)
    (hstep : stepInstr? st cta warp gi = some st') :
    ∃ stCore warpCore,
      advanceRunnablePcs? stCore cta warp = some st' ∧
        stCore.kernelEnv = st.kernelEnv ∧
        stCore.getWarp? cta warp = some warpCore ∧
        lockstepRunnable warpCore ∧
        RunnablePc warpCore pc := by
  cases gi with
  | mk guard? instr =>
      unfold stepInstr? at hstep
      simp [hwarp] at hstep
      have hlockBool : lockstepRunnable? warpState = true := hlock
      simp [hlockBool] at hstep
      cases hpart : participatingRunnableLaneIds? warpState guard? with
      | none =>
          simp [hpart] at hstep
      | some participants =>
          simp [hpart] at hstep
          cases instr <;> simp [instrUsesOrdinaryPcAdvance] at hordinary
          · rename_i dst rhs
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (evalRValue? st cta warp lane rhs).bind fun v =>
                      some (writeReg laneState dst v)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases heval : evalRValue? st cta warp lane rhs with
                      | none =>
                          simp [heval] at hf
                      | some v =>
                          simp [heval] at hf
                          subst new
                          simp [writeReg])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst cmp
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (evalCmp? st cta warp lane cmp).bind fun b =>
                      some (writePred laneState dst b)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases heval : evalCmp? st cta warp lane cmp with
                      | none =>
                          simp [heval] at hf
                      | some b =>
                          simp [heval] at hf
                          subst new
                          simp [writePred])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst rhs
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (evalRValue? st cta warp lane rhs).bind fun v =>
                      (valueToBool? v).bind fun b =>
                        some (writePred laneState dst b)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases heval : evalRValue? st cta warp lane rhs with
                      | none =>
                          simp [heval] at hf
                      | some v =>
                          simp [heval] at hf
                          cases hbool : valueToBool? v with
                          | none =>
                              simp [hbool] at hf
                          | some b =>
                              simp [hbool] at hf
                              subst new
                              simp [writePred])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst src
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (resolveAddr? st cta warp lane src).bind fun addr =>
                      (readMem? st src.space src.ty addr).bind fun value =>
                        some (writeReg laneState dst value)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases haddr : resolveAddr? st cta warp lane src with
                      | none =>
                          simp [haddr] at hf
                      | some addr =>
                          simp [haddr] at hf
                          cases hread : readMem? st src.space src.ty addr with
                          | none =>
                              simp [hread] at hf
                          | some value =>
                              simp [hread] at hf
                              subst new
                              simp [writeReg])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst value
            cases hcore : stepStoreLanes? st cta warp participants dst value with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases stepStoreLanes?_warp_control_eq hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, stepStoreLanes?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst space src
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (evalRValue? st cta warp lane src).bind fun value =>
                      (evalCvta? space value).bind fun gaddr =>
                        some (writeReg laneState dst gaddr)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases heval : evalRValue? st cta warp lane src with
                      | none =>
                          simp [heval] at hf
                      | some value =>
                          simp [heval] at hf
                          cases hcvta : evalCvta? space value with
                          | none =>
                              simp [hcvta] at hf
                          | some gaddr =>
                              simp [hcvta] at hf
                              subst new
                              simp [writeReg])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩
          · rename_i dst space src
            cases hcore :
                applyToLaneIds? st cta warp participants
                  (fun lane laneState =>
                    (evalRValue? st cta warp lane src).bind fun value =>
                      (evalIsspacep? space value).bind fun b =>
                        some (writePred laneState dst b)) with
            | none =>
                simp [hcore] at hstep
            | some stCore =>
                simp [hcore] at hstep
                rcases applyToLaneIds?_warp_control_eq
                    (hpres := by
                      intro lane old new hf
                      cases heval : evalRValue? st cta warp lane src with
                      | none =>
                          simp [heval] at hf
                      | some value =>
                          simp [heval] at hf
                          cases hspacep : evalIsspacep? space value with
                          | none =>
                              simp [hspacep] at hf
                          | some b =>
                              simp [hspacep] at hf
                              subst new
                              simp [writePred])
                    hwarp hlock hrpc hcore with
                  ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
                exact ⟨stCore, warpCore, hstep, applyToLaneIds?_kernelEnv_eq hcore,
                  hwarpCore, hlockCore, hrpcCore⟩

def stepTerminator? (st : State) (cta : CTAId) (warp : WarpId) (term : Terminator) : Option State := do
  let warpState <- st.getWarp? cta warp
  if !lockstepRunnable? warpState then
    none
  else
    let pc <- currentRunnablePc? warpState
    let lanes := runnableLaneIds warpState |>.filter fun lane =>
      match warpState.getLane? lane with
      | some laneState => laneState.pc == pc
      | none => false
    match term with
    | .br label =>
        applyToLaneIds? st cta warp lanes fun _ laneState =>
          some { laneState with pc := (label, 0) }
    | .terminate =>
        applyToLaneIds? st cta warp lanes fun _ laneState =>
          some { laneState with status := .terminated }
    | .cbr cond tLabel fLabel =>
        let dest <- uniformBranchDestination? st cta warp lanes cond tLabel fLabel
        applyToLaneIds? st cta warp lanes fun _ laneState => some { laneState with pc := dest }

theorem stepTerminator?_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator}
    (hstep : stepTerminator? st cta warp term = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold stepTerminator? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpc : currentRunnablePc? warpState with
          | none =>
              simp [hpc] at hstep
          | some pc =>
              simp [hpc] at hstep
              cases term with
              | br label =>
                  exact applyToLaneIds?_kernelEnv_eq hstep
              | terminate =>
                  exact applyToLaneIds?_kernelEnv_eq hstep
              | cbr cond tLabel fLabel =>
                  cases hdest :
                      uniformBranchDestination? st cta warp
                        (List.filter
                          (fun lane =>
                            match warpState.getLane? lane with
                            | some laneState => laneState.pc == pc
                            | none => false)
                          (runnableLaneIds warpState))
                        cond tLabel fLabel with
                  | none =>
                      simp [hdest] at hstep
                  | some dest =>
                      simp [hdest] at hstep
                      exact applyToLaneIds?_kernelEnv_eq hstep

theorem stepTerminator?_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator}
    (hstep : stepTerminator? st cta warp term = some st') :
    st'.global = st.global := by
  unfold stepTerminator? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpc : currentRunnablePc? warpState with
          | none =>
              simp [hpc] at hstep
          | some pc =>
              simp [hpc] at hstep
              cases term with
              | br label =>
                  exact applyToLaneIds?_global_eq hstep
              | terminate =>
                  exact applyToLaneIds?_global_eq hstep
              | cbr cond tLabel fLabel =>
                  cases hdest :
                      uniformBranchDestination? st cta warp
                        (List.filter
                          (fun lane =>
                            match warpState.getLane? lane with
                            | some laneState => laneState.pc == pc
                            | none => false)
                          (runnableLaneIds warpState))
                        cond tLabel fLabel with
                  | none =>
                      simp [hdest] at hstep
                  | some dest =>
                      simp [hdest] at hstep
                      exact applyToLaneIds?_global_eq hstep

theorem stepTerminator?_lane_nonPc_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator}
    {target : LaneId} {targetState : LaneState}
    (htarget : st.getLane? cta warp target = some targetState)
    (hstep : stepTerminator? st cta warp term = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem ∧
      targetState'.regs = targetState.regs ∧
      targetState'.preds = targetState.preds := by
  unfold stepTerminator? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpc : currentRunnablePc? warpState with
          | none =>
              simp [hpc] at hstep
          | some pc =>
              simp [hpc] at hstep
              cases term with
              | br label =>
                  exact applyToLaneIds?_lane_nonPc_eq
                    (by intro lane old new hnew; simp at hnew; subst new; simp)
                    htarget hstep
              | terminate =>
                  exact applyToLaneIds?_lane_nonPc_eq
                    (by intro lane old new hnew; simp at hnew; subst new; simp)
                    htarget hstep
              | cbr cond tLabel fLabel =>
                  cases hdest :
                      uniformBranchDestination? st cta warp
                        (List.filter
                          (fun lane =>
                            match warpState.getLane? lane with
                            | some laneState => laneState.pc == pc
                            | none => false)
                          (runnableLaneIds warpState))
                        cond tLabel fLabel with
                  | none =>
                      simp [hdest] at hstep
                  | some dest =>
                      simp [hdest] at hstep
                      exact applyToLaneIds?_lane_nonPc_eq
                        (by intro lane old new hnew; simp at hnew; subst new; simp)
                        htarget hstep

end Helpers

end CLean
