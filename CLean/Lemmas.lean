import Std.Data.HashMap.Lemmas
import CLean.Semantics

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

end CLean
