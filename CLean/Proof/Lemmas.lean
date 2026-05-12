-- import Mathlib.Tactic
import Std.Data.HashMap.Lemmas
import CLean.Semantics.SmallStep

namespace CLean

open Helpers

@[simp] theorem readReg_writeReg_eq (lane : LaneState) (r : RegName) (v : Value) :
    readReg (writeReg lane r v) r = some v := by
  simp [readReg, writeReg]

theorem readReg_writeReg_same (lane : LaneState) (r : RegName) (v : Value) :
    readReg (writeReg lane r v) r = some v :=
  readReg_writeReg_eq lane r v

@[simp] theorem readReg_writeReg_ne (lane : LaneState) {r r' : RegName} (v : Value) (h : r' ≠ r) :
    readReg (writeReg lane r v) r' = readReg lane r' := by
  rw [readReg, writeReg, Std.HashMap.getElem?_insert]
  by_cases hEq : r = r'
  · exact False.elim (h hEq.symm)
  · simp [readReg, beq_iff_eq, hEq]

@[simp] theorem readPred_writePred_eq (lane : LaneState) (p : PredName) (b : Bool) :
    readPred (writePred lane p b) p = some b := by
  simp [readPred, writePred]

theorem readPred_writePred_same (lane : LaneState) (p : PredName) (b : Bool) :
    readPred (writePred lane p b) p = some b :=
  readPred_writePred_eq lane p b

@[simp] theorem readPred_writePred_ne (lane : LaneState) {p p' : PredName} (b : Bool) (h : p' ≠ p) :
    readPred (writePred lane p b) p' = readPred lane p' := by
  rw [readPred, writePred, Std.HashMap.getElem?_insert]
  by_cases hEq : p = p'
  · exact False.elim (h hEq.symm)
  · simp [readPred, beq_iff_eq, hEq]

@[simp] theorem readPred_writeReg (lane : LaneState) (r : RegName) (v : Value) (p : PredName) :
    readPred (writeReg lane r v) p = readPred lane p := by
  rfl

@[simp] theorem readReg_writePred (lane : LaneState) (p : PredName) (b : Bool) (r : RegName) :
    readReg (writePred lane p b) r = readReg lane r := by
  rfl

@[simp] theorem writeReg_preserves_preds (lane : LaneState) (r : RegName) (v : Value) :
    (writeReg lane r v).preds = lane.preds := rfl

@[simp] theorem writeReg_preserves_localMem (lane : LaneState) (r : RegName) (v : Value) :
    (writeReg lane r v).localMem = lane.localMem := rfl

@[simp] theorem writeReg_preserves_pc (lane : LaneState) (r : RegName) (v : Value) :
    (writeReg lane r v).pc = lane.pc := rfl

@[simp] theorem writeReg_preserves_status (lane : LaneState) (r : RegName) (v : Value) :
    (writeReg lane r v).status = lane.status := rfl

@[simp] theorem writePred_preserves_regs (lane : LaneState) (p : PredName) (b : Bool) :
    (writePred lane p b).regs = lane.regs := rfl

@[simp] theorem writePred_preserves_localMem (lane : LaneState) (p : PredName) (b : Bool) :
    (writePred lane p b).localMem = lane.localMem := rfl

@[simp] theorem writePred_preserves_pc (lane : LaneState) (p : PredName) (b : Bool) :
    (writePred lane p b).pc = lane.pc := rfl

@[simp] theorem writePred_preserves_status (lane : LaneState) (p : PredName) (b : Bool) :
    (writePred lane p b).status = lane.status := rfl

@[simp] theorem WarpState.getLane?_setLane_eq (warp : WarpState) (lane : LaneId) (laneState : LaneState)
    (hwf : WarpState.wf warp) :
    (warp.setLane lane laneState).getLane? lane = some laneState := by
  have hsize : warp.lanes.size = 32 := by
    simpa [WarpState.wf, WarpState.wf?] using hwf
  have hlt : lane.val < warp.lanes.size := by
    simp [hsize]
  simp [WarpState.getLane?, WarpState.setLane, hlt]

theorem WarpState.getLane?_setLane_same (warp : WarpState) (lane : LaneId) (laneState : LaneState)
    (hwf : WarpState.wf warp) :
    (warp.setLane lane laneState).getLane? lane = some laneState :=
  WarpState.getLane?_setLane_eq warp lane laneState hwf

theorem WarpState.getLane?_setLane_ne (warp : WarpState) {lane lane' : LaneId} (h : lane' ≠ lane)
    (laneState : LaneState) :
    (warp.setLane lane laneState).getLane? lane' = warp.getLane? lane' := by
  have hne : lane.val ≠ lane'.val := by
    intro hval
    apply h
    apply Fin.ext
    exact hval.symm
  unfold WarpState.getLane? WarpState.setLane
  simp [Array.getElem?_setIfInBounds_ne, hne]

theorem WarpState.wf_setLane (warp : WarpState) (lane : LaneId) (laneState : LaneState) :
    WarpState.wf warp → WarpState.wf (warp.setLane lane laneState) := by
  intro h
  have hsize : warp.lanes.size = 32 := by
    simpa [WarpState.wf, WarpState.wf?] using h
  simp [WarpState.wf, WarpState.wf?, WarpState.setLane, hsize]

@[simp] theorem State.getCTA?_setCTA_same (st : State) (cta : CTAId) (ctaState : CTAState) :
    (st.setCTA cta ctaState).getCTA? cta = some ctaState := by
  simp [State.getCTA?, State.setCTA]

theorem State.getCTA?_setCTA_ne (st : State) {cta cta' : CTAId} (h : cta' ≠ cta) (ctaState : CTAState) :
    (st.setCTA cta ctaState).getCTA? cta' = st.getCTA? cta' := by
  rw [State.getCTA?, State.setCTA, Std.HashMap.getElem?_insert]
  by_cases hEq : cta = cta'
  · exact False.elim (h hEq.symm)
  · simp [State.getCTA?, beq_iff_eq, hEq]

theorem State.getWarp?_setWarp_same
    (st : State) (cta : CTAId) (warp : WarpId) (warpState : WarpState) (ctaState : CTAState)
    (hcta : st.getCTA? cta = some ctaState) :
    (st.setWarp cta warp warpState).bind (fun st' => st'.getWarp? cta warp) = some warpState := by
  unfold State.setWarp State.getWarp?
  simp [hcta]

theorem State.getWarp?_setWarp_ne
    (st : State) (cta : CTAId) {warp warp' : WarpId} (h : warp' ≠ warp)
    (warpState : WarpState) (ctaState : CTAState)
    (hcta : st.getCTA? cta = some ctaState) :
    (st.setWarp cta warp warpState).bind (fun st' => st'.getWarp? cta warp') = ctaState.warps[warp']? := by
  unfold State.setWarp State.getWarp?
  rw [hcta]
  simp [Std.HashMap.getElem?_insert]
  by_cases hEq : warp = warp'
  · exact False.elim (h hEq.symm)
  · simp [beq_iff_eq, hEq]

theorem State.getLane?_setLane_same
    (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (laneState : LaneState)
    (warpState : WarpState) (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState) :
    (st.setLane cta warp lane laneState).bind (fun st' => st'.getLane? cta warp lane) = some laneState := by
  rcases hcta : st.getCTA? cta with _ | ctaState
  · simp [State.getWarp?, hcta] at hwarp
  · unfold State.setLane
    rw [hwarp]
    have hcta' : st.ctas[cta]? = some ctaState := by
      simpa [State.getCTA?] using hcta
    simp [State.getLane?, State.getWarp?, State.setWarp, State.setCTA, State.getCTA?,
      hcta, hcta', WarpState.getLane?_setLane_same, hwfWarp]

theorem State.getLane?_setLane_ne
    (st : State) (cta : CTAId) (warp : WarpId) {lane lane' : LaneId} (h : lane' ≠ lane)
    (laneState : LaneState) (warpState : WarpState)
    (hwarp : st.getWarp? cta warp = some warpState) :
    (st.setLane cta warp lane laneState).bind (fun st' => st'.getLane? cta warp lane') = warpState.getLane? lane' := by
  rcases hcta : st.getCTA? cta with _ | ctaState
  · simp [State.getWarp?, hcta] at hwarp
  · unfold State.setLane
    rw [hwarp]
    have hcta' : st.ctas[cta]? = some ctaState := by
      simpa [State.getCTA?] using hcta
    simp [State.getLane?, State.getWarp?, State.setWarp, State.setCTA, State.getCTA?,
      hcta, hcta', WarpState.getLane?_setLane_ne, h]

theorem State.wf_global_update (st : State) (global : GlobalMem) :
    State.wf st → State.wf { st with global := global } := by
  intro h
  simpa [State.wf, State.wf?] using h

theorem State.wf_const_update (st : State) (const : ConstMem) :
    State.wf st → State.wf { st with const := const } := by
  intro h
  simpa [State.wf, State.wf?] using h

theorem State.wf_param_update (st : State) (param : ParamMem) :
    State.wf st → State.wf { st with param := param } := by
  intro h
  simpa [State.wf, State.wf?] using h

theorem State.wf_atomics_update (st : State) (atomics : AtomicState) :
    State.wf st → State.wf { st with atomics := atomics } := by
  intro h
  simpa [State.wf, State.wf?] using h

@[simp] theorem evalCvta?_global_u64 (n : UInt64) :
    Helpers.evalCvta? .global (.u64 n) = some (.gaddr .global n.toNat) := by
  rfl

@[simp] theorem evalCvta?_shared_u32 (n : UInt32) :
    Helpers.evalCvta? .shared (.u32 n) = some (.gaddr .shared n.toNat) := by
  rfl

@[simp] theorem evalCvta?_global_gaddr (n : Nat) :
    Helpers.evalCvta? .global (.gaddr .global n) = some (.gaddr .global n) := by
  rfl

@[simp] theorem evalIsspacep?_match (space : AddrSpace) (n : Nat) :
    Helpers.evalIsspacep? space (.gaddr space n) = some true := by
  simp [Helpers.evalIsspacep?]

theorem evalIsspacep?_mismatch {s1 s2 : AddrSpace} (h : s1 ≠ s2) (n : Nat) :
    Helpers.evalIsspacep? s1 (.gaddr s2 n) = some false := by
  have h' : ¬ s2 = s1 := by
    intro hs
    apply h
    exact hs.symm
  simp [Helpers.evalIsspacep?, beq_iff_eq, h']

@[simp] theorem natToBytesLE_length (n width : Nat) :
    (Helpers.natToBytesLE n width).length = width := by
  simp [Helpers.natToBytesLE]

theorem bytesToNatLE_natToBytesLE_1 (n : Nat) :
    Helpers.bytesToNatLE (Helpers.natToBytesLE n 1) = n % (2 ^ 8) := by
  simp [Helpers.natToBytesLE, Helpers.bytesToNatLE, List.range, List.range.loop, bytesToNatLE.loop]


theorem bytesToNatLE_natToBytesLE_2 (n : Nat) :
    Helpers.bytesToNatLE (Helpers.natToBytesLE n 2) = n % (2 ^ 16) := by
  simp [Helpers.natToBytesLE, Helpers.bytesToNatLE, List.range, List.range.loop, bytesToNatLE.loop]
  omega

theorem bytesToNatLE_natToBytesLE_4 (n : Nat) :
    Helpers.bytesToNatLE (Helpers.natToBytesLE n 4) = n % (2 ^ 32) := by
  simp [Helpers.natToBytesLE, Helpers.bytesToNatLE, List.range, List.range.loop, bytesToNatLE.loop]
  omega


theorem bytesToNatLE_natToBytesLE_8 (n : Nat) :
    Helpers.bytesToNatLE (Helpers.natToBytesLE n 8) = n % (2 ^ 64) := by
  simp [Helpers.natToBytesLE, Helpers.bytesToNatLE, List.range, List.range.loop, bytesToNatLE.loop]
  omega

theorem decode_encode_pred (b : Bool) :
    Helpers.decodeScalar? .pred (Option.get! (Helpers.encodeScalar? .pred (.pred b))) = some (.pred b) := by
  cases b <;> rfl

theorem decode_encode_u8 (x : UInt8) :
    Helpers.decodeScalar? .u8 (Option.get! (Helpers.encodeScalar? .u8 (.u8 x))) = some (.u8 x) := by
  rfl

theorem decode_encode_u16 (x : UInt16) :
    Helpers.decodeScalar? .u16 (Option.get! (Helpers.encodeScalar? .u16 (.u16 x))) = some (.u16 x) := by
  simp [Helpers.encodeScalar?, Helpers.decodeScalar?, bytesToNatLE_natToBytesLE_2, UInt16.ofNat_toNat,
    Nat.mod_eq_of_lt x.toNat_lt_size]

theorem decode_encode_u32 (x : UInt32) :
    Helpers.decodeScalar? .u32 (Option.get! (Helpers.encodeScalar? .u32 (.u32 x))) = some (.u32 x) := by
  simp [Helpers.encodeScalar?, Helpers.decodeScalar?, bytesToNatLE_natToBytesLE_4, UInt32.ofNat_toNat,
    Nat.mod_eq_of_lt x.toNat_lt_size]

theorem decode_encode_u64 (x : UInt64) :
    Helpers.decodeScalar? .u64 (Option.get! (Helpers.encodeScalar? .u64 (.u64 x))) = some (.u64 x) := by
  simp [Helpers.encodeScalar?, Helpers.decodeScalar?, bytesToNatLE_natToBytesLE_8, UInt64.ofNat_toNat,
    Nat.mod_eq_of_lt x.toNat_lt_size]

theorem decode_encode_s32 (x : Int) :
    Helpers.decodeScalar? .s32 (Option.get! (Helpers.encodeScalar? .s32 (.s32 x))) =
      some (.s32 (Helpers.natToSigned 32 (Helpers.signedToNat 32 x))) := by
  sorry

theorem decode_encode_s64 (x : Int) :
    Helpers.decodeScalar? .s64 (Option.get! (Helpers.encodeScalar? .s64 (.s64 x))) =
      some (.s64 (Helpers.natToSigned 64 (Helpers.signedToNat 64 x))) := by
  sorry

theorem decode_encode_f16 (bits : UInt16) :
    Helpers.decodeScalar? .f16 (Option.get! (Helpers.encodeScalar? .f16 (.f16 bits))) = some (.f16 bits) := by
  simp [Helpers.encodeScalar?, Helpers.decodeScalar?, bytesToNatLE_natToBytesLE_2, UInt16.ofNat_toNat,
    Nat.mod_eq_of_lt bits.toNat_lt_size]

theorem decode_encode_bf16 (bits : UInt16) :
    Helpers.decodeScalar? .bf16 (Option.get! (Helpers.encodeScalar? .bf16 (.bf16 bits))) = some (.bf16 bits) := by
  simp [Helpers.encodeScalar?, Helpers.decodeScalar?, bytesToNatLE_natToBytesLE_2, UInt16.ofNat_toNat,
    Nat.mod_eq_of_lt bits.toNat_lt_size]

theorem readBytes?_writeBytes_same (mem : ByteMem) (offset : Nat) (bs : List Byte) :
    Helpers.readBytes? (Helpers.writeBytes mem offset bs) offset bs.length = some bs := by
  induction bs generalizing mem offset with
  | nil =>
      simp [Helpers.readBytes?, Helpers.writeBytes, readBytes?.loop, writeBytes.loop]
  | cons b bs ih =>
      simp [Helpers.readBytes?, Helpers.writeBytes]
      sorry


theorem readMem_writeMem_same_global_u32
    (st st' : State) (offset : Nat) (x : UInt32)
    (halign : offset % 4 = 0)
    (hwrite : Helpers.writeMem? st .global .u32 (.global offset) (.u32 x) = some st') :
    Helpers.readMem? st' .global .u32 (.global offset) = some (.u32 x) := by
  have hst' : st' = { st with global := { bytes := Helpers.writeBytes st.global.bytes offset (Helpers.natToBytesLE x.toNat 4) } } := by
    simp [Helpers.writeMem?, Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?, Typing.byteWidth?,
      Typing.aligned?, Typing.addrSpaceMatches?, Helpers.encodeScalar?, halign] at hwrite
    -- simpa using hwrite.symm
    sorry
  subst st'
  simp [Helpers.readMem?, Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?, Typing.byteWidth?,
    Typing.aligned?, Typing.addrSpaceMatches?, halign, readBytes?_writeBytes_same, Helpers.decodeScalar?,
    natToBytesLE_length, bytesToNatLE_natToBytesLE_4, UInt32.ofNat_toNat, Nat.mod_eq_of_lt x.toNat_lt_size]
  sorry

theorem readMem_writeMem_same_global_u64
    (st st' : State) (offset : Nat) (x : UInt64)
    (halign : offset % 8 = 0)
    (hwrite : Helpers.writeMem? st .global .u64 (.global offset) (.u64 x) = some st') :
    Helpers.readMem? st' .global .u64 (.global offset) = some (.u64 x) := by
  have hst' : st' = { st with global := { bytes := Helpers.writeBytes st.global.bytes offset (Helpers.natToBytesLE x.toNat 8) } } := by
    simp [Helpers.writeMem?, Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?, Typing.byteWidth?,
      Typing.aligned?, Typing.addrSpaceMatches?, Helpers.encodeScalar?, halign] at hwrite
    -- simpa using hwrite.symm
    sorry
  subst st'
  simp [Helpers.readMem?, Typing.typedAccessPreconditions?, Typing.scalarCodecSupported?, Typing.byteWidth?,
    Typing.aligned?, Typing.addrSpaceMatches?, halign, readBytes?_writeBytes_same, Helpers.decodeScalar?,
    natToBytesLE_length, bytesToNatLE_natToBytesLE_8, UInt64.ofNat_toNat, Nat.mod_eq_of_lt x.toNat_lt_size]
  sorry

theorem readMem_writeMem_same_global_s32
    (st st' : State) (offset : Nat) (x : Int)
    (halign : offset % 4 = 0)
    (hwrite : Helpers.writeMem? st .global .s32 (.global offset) (.s32 x) = some st') :
    Helpers.readMem? st' .global .s32 (.global offset) =
      some (.s32 (Helpers.natToSigned 32 (Helpers.signedToNat 32 x))) := by
  sorry

theorem readMem_writeMem_same_global_s64
    (st st' : State) (offset : Nat) (x : Int)
    (halign : offset % 8 = 0)
    (hwrite : Helpers.writeMem? st .global .s64 (.global offset) (.s64 x) = some st') :
    Helpers.readMem? st' .global .s64 (.global offset) =
      some (.s64 (Helpers.natToSigned 64 (Helpers.signedToNat 64 x))) := by
  sorry

def TopMemEq (st st' : State) : Prop :=
  st'.global = st.global ∧ st'.const = st.const ∧ st'.param = st.param

def LaneRegFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ lane laneState laneState',
    st.getLane? cta warp lane = some laneState →
    st'.getLane? cta warp lane = some laneState' →
    laneState'.regs = laneState.regs

def LanePredFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ lane laneState laneState',
    st.getLane? cta warp lane = some laneState →
    st'.getLane? cta warp lane = some laneState' →
    laneState'.preds = laneState.preds

theorem stepInstr_assignReg_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {rhs : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_assignPred_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {cmp : CmpExpr}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_load_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {src : TypedAddr}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .load dst src } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_cvta_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {space : AddrSpace} {src : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_isspacep_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {space : AddrSpace} {src : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_barrierCTA_preserves_mem
    {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :
    TopMemEq st st' := by
  sorry

theorem stepInstr_assignReg_preserves_pred_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {rhs : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } = some st') :
    LanePredFilesEq st st' cta warp := by
  sorry

theorem stepInstr_assignPred_preserves_reg_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {cmp : CmpExpr}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } = some st') :
    LaneRegFilesEq st st' cta warp := by
  sorry

theorem stepInstr_store_preserves_reg_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : TypedAddr} {value : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .store dst value } = some st') :
    LaneRegFilesEq st st' cta warp := by
  sorry

theorem stepInstr_store_preserves_pred_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : TypedAddr} {value : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .store dst value } = some st') :
    LanePredFilesEq st st' cta warp := by
  sorry

theorem stepInstr_cvta_preserves_pred_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {space : AddrSpace} {src : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } = some st') :
    LanePredFilesEq st st' cta warp := by
  sorry

theorem stepInstr_isspacep_preserves_reg_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {space : AddrSpace} {src : RValue}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } = some st') :
    LaneRegFilesEq st st' cta warp := by
  sorry

theorem stepInstr_barrierCTA_preserves_reg_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :
    LaneRegFilesEq st st' cta warp := by
  sorry

theorem stepInstr_barrierCTA_preserves_pred_files
    {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {guard? : Option Guard}
    (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :
    LanePredFilesEq st st' cta warp := by
  sorry

theorem stepInstr?_preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hwf : State.wf st)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    State.wf st' := by
  sorry

theorem stepInstr?_preserves_kernelEnv
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    st'.kernelEnv = st.kernelEnv := by
  sorry

theorem stepTerminator?_preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator}
    (hwf : State.wf st)
    (hstep : Helpers.stepTerminator? st cta warp term = some st') :
    State.wf st' := by
  sorry

theorem StepInstr.preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hstep : StepInstr st cta warp gi st') :
    State.wf st' := by
  cases hstep with
  | mk hwf _ _ _ _ hrun =>
      exact stepInstr?_preserves_wf hwf hrun

theorem StepBlock.preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hstep : StepBlock st cta warp st') :
    State.wf st' := by
  cases hstep with
  | body _ _ _ _ _ _ _ hinstr =>
      exact StepInstr.preserves_wf hinstr
  | term hwf _ _ _ _ _ _ hterm =>
      exact stepTerminator?_preserves_wf hwf hterm

theorem StepWarp.preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hstep : StepWarp st cta warp st') :
    State.wf st' := by
  cases hstep with
  | mk _ hblock =>
      exact StepBlock.preserves_wf hblock

theorem StepMachine.preserves_wf
    {st st' : State}
    (hstep : StepMachine st st') :
    State.wf st' := by
  cases hstep with
  | mk _ hwarp =>
      exact StepWarp.preserves_wf hwarp

theorem StepInstr.exists_of_stepInstr?_isSome
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {gi : GInstr} {participants : List LaneId}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState gi.guard? participants)
    (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) :
    ∃ st', StepInstr st cta warp gi st' := by
  cases hstep : Helpers.stepInstr? st cta warp gi with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      exact ⟨st', StepInstr.mk hwf hwarp hwfWarp hlock hpart hstep⟩

theorem StepMachine.body_of_stepInstr?
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : Helpers.ParticipatingRunnable warpState gi.guard? participants)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    StepMachine st st' :=
  StepMachine.mk hwf <|
    StepWarp.mk hwf <|
      StepBlock.body hwf hwarp hwfWarp hlock hrpc hblock hgi <|
        StepInstr.mk hwf hwarp hwfWarp hlock hpart hstep

theorem StepMachine.term_of_stepTerminator?
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbodyDone : block.body[pc.2]? = none)
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st') :
    StepMachine st st' :=
  StepMachine.mk hwf <|
    StepWarp.mk hwf <|
      StepBlock.term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hstep

theorem StepMachine.body_of_stepInstr?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : Helpers.ParticipatingRunnable warpState gi.guard? participants)
    (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) :
    StepMachine st
      (match Helpers.stepInstr? st cta warp gi with
       | some st' => st'
       | none => st) := by
  cases hstep : Helpers.stepInstr? st cta warp gi with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      simpa [hstep] using
        StepMachine.body_of_stepInstr?
          (st := st) (st' := st') (cta := cta) (warp := warp)
          (warpState := warpState) (pc := pc) (block := block) (gi := gi)
          (participants := participants)
          hwf hwarp hwfWarp hlock hrpc hblock hgi hpart hstep

theorem StepMachine.term_of_stepTerminator?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbodyDone : block.body[pc.2]? = none)
    (hisSome : (Helpers.stepTerminator? st cta warp block.term).isSome = true) :
    StepMachine st
      (match Helpers.stepTerminator? st cta warp block.term with
       | some st' => st'
       | none => st) := by
  cases hstep : Helpers.stepTerminator? st cta warp block.term with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      simpa [hstep] using
        StepMachine.term_of_stepTerminator?
          (st := st) (st' := st') (cta := cta) (warp := warp)
          (warpState := warpState) (pc := pc) (block := block)
          hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hstep

theorem stepInstr?_computed_preserves_kernelEnv
    {st : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) :
    (match Helpers.stepInstr? st cta warp gi with
     | some st' => st'
     | none => st).kernelEnv = st.kernelEnv := by
  cases hstep : Helpers.stepInstr? st cta warp gi with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      simpa [hstep] using stepInstr?_preserves_kernelEnv hstep

theorem stepInstr?_computed_preserves_block?
    {st : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (label : BlockLabel)
    (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) :
    (match Helpers.stepInstr? st cta warp gi with
     | some st' => st'
     | none => st).kernelEnv.blocks[label]? = st.kernelEnv.blocks[label]? := by
  have hk := stepInstr?_computed_preserves_kernelEnv
    (st := st) (cta := cta) (warp := warp) (gi := gi) hisSome
  exact congrArg (fun env : KernelEnv => env.blocks[label]?) hk

def StepMachine.currentInstrStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  if !st.wf? then
    none
  else
    let warpState <- st.getWarp? cta warp
    if !warpState.wf? then
      none
    else if !Helpers.lockstepRunnable? warpState then
      none
    else
      let pc <- Helpers.currentRunnablePc? warpState
      let block <- st.kernelEnv.blocks[pc.1]?
      let gi <- block.body[pc.2]?
      let _participants <- Helpers.participatingRunnableLaneIds? warpState gi.guard?
      Helpers.stepInstr? st cta warp gi

def StepMachine.currentTermStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  if !st.wf? then
    none
  else
    let warpState <- st.getWarp? cta warp
    if !warpState.wf? then
      none
    else if !Helpers.lockstepRunnable? warpState then
      none
    else
      let pc <- Helpers.currentRunnablePc? warpState
      let block <- st.kernelEnv.blocks[pc.1]?
      match block.body[pc.2]? with
      | some _ => none
      | none => Helpers.stepTerminator? st cta warp block.term

theorem StepMachine.body_of_currentInstrStep?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    (hisSome : (StepMachine.currentInstrStep? st cta warp).isSome = true) :
    StepMachine st
      (match StepMachine.currentInstrStep? st cta warp with
       | some st' => st'
       | none => st) := by
  unfold StepMachine.currentInstrStep? at hisSome ⊢
  cases hwf : st.wf? <;> simp [hwf] at hisSome ⊢
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hisSome
  | some warpState =>
      simp [hwarp] at hisSome ⊢
      cases hwfWarp : warpState.wf? <;> simp [hwfWarp] at hisSome ⊢
      cases hlock : Helpers.lockstepRunnable? warpState <;> simp [hlock] at hisSome ⊢
      cases hpc : Helpers.currentRunnablePc? warpState with
      | none =>
          simp [hpc] at hisSome
      | some pc =>
          simp [hpc] at hisSome ⊢
          cases hblock : st.kernelEnv.blocks[pc.1]? with
          | none =>
              simp [hblock] at hisSome
          | some block =>
              simp [hblock] at hisSome ⊢
              cases hgi : block.body[pc.2]? with
              | none =>
                  simp [hgi] at hisSome
              | some gi =>
                  simp [hgi] at hisSome ⊢
                  cases hpart : Helpers.participatingRunnableLaneIds? warpState gi.guard? with
                  | none =>
                      simp [hpart] at hisSome
                  | some participants =>
                      simp [hpart] at hisSome ⊢
                      exact StepMachine.body_of_stepInstr?_computed
                        (cta := cta) (warp := warp) (warpState := warpState) (pc := pc)
                        (block := block) (gi := gi) (participants := participants)
                        ((State.wf_iff_bool st).2 hwf)
                        hwarp
                        ((WarpState.wf_iff_bool warpState).2 hwfWarp)
                        ((Helpers.lockstepRunnable_iff_bool warpState).2 hlock)
                        ((Helpers.runnablePc_iff_bool warpState pc).2 hpc)
                        hblock
                        hgi
                        ((Helpers.participatingRunnable_iff_bool warpState gi.guard? participants).2 hpart)
                        hisSome

theorem StepMachine.term_of_currentTermStep?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    (hisSome : (StepMachine.currentTermStep? st cta warp).isSome = true) :
    StepMachine st
      (match StepMachine.currentTermStep? st cta warp with
       | some st' => st'
       | none => st) := by
  unfold StepMachine.currentTermStep? at hisSome ⊢
  cases hwf : st.wf? <;> simp [hwf] at hisSome ⊢
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hisSome
  | some warpState =>
      simp [hwarp] at hisSome ⊢
      cases hwfWarp : warpState.wf? <;> simp [hwfWarp] at hisSome ⊢
      cases hlock : Helpers.lockstepRunnable? warpState <;> simp [hlock] at hisSome ⊢
      cases hpc : Helpers.currentRunnablePc? warpState with
      | none =>
          simp [hpc] at hisSome
      | some pc =>
          simp [hpc] at hisSome ⊢
          cases hblock : st.kernelEnv.blocks[pc.1]? with
          | none =>
              simp [hblock] at hisSome
          | some block =>
              simp [hblock] at hisSome ⊢
              cases hbody : block.body[pc.2]? with
              | some gi =>
                  simp [hbody] at hisSome
              | none =>
                  simp [hbody] at hisSome ⊢
                  exact StepMachine.term_of_stepTerminator?_computed
                    (cta := cta) (warp := warp) (warpState := warpState) (pc := pc)
                    (block := block)
                    ((State.wf_iff_bool st).2 hwf)
                    hwarp
                    ((WarpState.wf_iff_bool warpState).2 hwfWarp)
                    ((Helpers.lockstepRunnable_iff_bool warpState).2 hlock)
                    ((Helpers.runnablePc_iff_bool warpState pc).2 hpc)
                    hblock
                    hbody
                    hisSome

syntax "cstep" : tactic

macro_rules
  | `(tactic| cstep) =>
      `(tactic|
        first
        | exact StepMachine.body_of_currentInstrStep?_computed (by native_decide)
        | exact StepMachine.term_of_currentTermStep?_computed (by native_decide))
end CLean
