import CLean.CSL
import CLean.Semantics.Helpers

namespace CLean
namespace WP

def stateProp (p : State → Prop) : CSL.Assertion :=
  fun st r => p st ∧ CSL.emp st r

theorem stateProp_state {p : State → Prop} {st : State} {r : CSL.Resource}
    (h : stateProp p st r) :
    p st :=
  h.1

theorem stateProp_emp {p : State → Prop} {st : State} {r : CSL.Resource}
    (h : stateProp p st r) :
    CSL.emp st r :=
  h.2

def warpAt (cta : CTAId) (warp : WarpId) (pc : PC) (lanes : List LaneId) :
    CSL.Assertion :=
  stateProp fun st =>
    ∃ warpState,
      st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        Helpers.ParticipatingRunnable warpState none lanes

def warpParticipants
    (cta : CTAId) (warp : WarpId) (guard? : Option Guard) (lanes : List LaneId) :
    CSL.Assertion :=
  stateProp fun st =>
    ∃ warpState,
      st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.ParticipatingRunnable warpState guard? lanes

def laneRunning (cta : CTAId) (warp : WarpId) (lane : LaneId) (pc : PC) :
    CSL.Assertion :=
  stateProp fun st =>
    ∃ laneState,
      st.getLane? cta warp lane = some laneState ∧
        laneState.status = .running ∧
        laneState.pc = pc

def laneTerminatedAt (cta : CTAId) (warp : WarpId) (lane : LaneId) (pc : PC) :
    CSL.Assertion :=
  stateProp fun st =>
    ∃ laneState,
      st.getLane? cta warp lane = some laneState ∧
        laneState.status = .terminated ∧
        laneState.pc = pc

def lanesTerminatedAt (cta : CTAId) (warp : WarpId) (lanes : List LaneId) (pc : PC) :
    CSL.Assertion :=
  stateProp fun st =>
    ∀ lane, lane ∈ lanes →
      ∃ laneState,
        st.getLane? cta warp lane = some laneState ∧
          laneState.status = .terminated ∧
          laneState.pc = pc

theorem warpAt_state
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId}
    {st : State} {r : CSL.Resource}
    (h : warpAt cta warp pc lanes st r) :
    ∃ warpState,
      st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        Helpers.ParticipatingRunnable warpState none lanes :=
  h.1

theorem warpAt_participants_none
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId}
    {st : State} {r : CSL.Resource}
    (h : warpAt cta warp pc lanes st r) :
    warpParticipants cta warp none lanes st r := by
  rcases h with ⟨⟨warpState, hwarp, hlock, _hrpc, hpart⟩, hemp⟩
  exact ⟨⟨warpState, hwarp, hlock, hpart⟩, hemp⟩

theorem warpParticipants_state
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {lanes : List LaneId}
    {st : State} {r : CSL.Resource}
    (h : warpParticipants cta warp guard? lanes st r) :
    ∃ warpState,
      st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.ParticipatingRunnable warpState guard? lanes :=
  h.1

theorem laneRunning_state
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {pc : PC}
    {st : State} {r : CSL.Resource}
    (h : laneRunning cta warp lane pc st r) :
    ∃ laneState,
      st.getLane? cta warp lane = some laneState ∧
        laneState.status = .running ∧
        laneState.pc = pc :=
  h.1

theorem laneTerminatedAt_state
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {pc : PC}
    {st : State} {r : CSL.Resource}
    (h : laneTerminatedAt cta warp lane pc st r) :
    ∃ laneState,
      st.getLane? cta warp lane = some laneState ∧
        laneState.status = .terminated ∧
        laneState.pc = pc :=
  h.1

theorem lanesTerminatedAt_state
    {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {pc : PC}
    {st : State} {r : CSL.Resource}
    (h : lanesTerminatedAt cta warp lanes pc st r) :
    ∀ lane, lane ∈ lanes →
      ∃ laneState,
        st.getLane? cta warp lane = some laneState ∧
          laneState.status = .terminated ∧
          laneState.pc = pc :=
  h.1

theorem warpAt_single_of_advance_setLane
    {cta : CTAId} {warp : WarpId} {pc : PC} {lane : LaneId}
    {st stCore st' : State} {warpState : WarpState}
    {laneState laneStateCore : LaneState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hpart : Helpers.ParticipatingRunnable warpState none [lane])
    (hlane : st.getLane? cta warp lane = some laneState)
    (hset : st.setLane cta warp lane laneStateCore = some stCore)
    (hstatus : laneStateCore.status = laneState.status)
    (hpc : laneStateCore.pc = laneState.pc)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st') :
    warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  have hwarpCore :
      stCore.getWarp? cta warp = some (warpState.setLane lane laneStateCore) :=
    State.getWarp?_setLane_same hwarp hset
  have hlaneCore : stCore.getLane? cta warp lane = some laneStateCore :=
    State.getLane?_setLane_same hlane hset
  have hpartCore :
      Helpers.ParticipatingRunnable (warpState.setLane lane laneStateCore) none [lane] :=
    Helpers.ParticipatingRunnable.setLane_none_control_eq hpart hwarpLane hstatus hpc
  have hlockCore : Helpers.lockstepRunnable (warpState.setLane lane laneStateCore) :=
    (Helpers.lockstepRunnable_setLane_control_iff hwarpLane hstatus hpc).2 hlock
  have hrpcCore : Helpers.RunnablePc (warpState.setLane lane laneStateCore) pc := by
    unfold Helpers.RunnablePc
    rw [Helpers.currentRunnablePc?_setLane_control_eq hwarpLane hstatus hpc]
    exact hrpc
  rcases Helpers.advanceRunnablePcs?_single_warp_control
      hwarpCore hlockCore hrpcCore hpartCore hlaneCore hadvance with
    ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩
  exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩

theorem warpAt_single_of_setLane_pc
    {cta : CTAId} {warp : WarpId} {pc pc' : PC} {lane : LaneId}
    {st st' : State} {warpState : WarpState}
    {laneState laneStateCore : LaneState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (_hrpc : Helpers.RunnablePc warpState pc)
    (hpart : Helpers.ParticipatingRunnable warpState none [lane])
    (hlane : st.getLane? cta warp lane = some laneState)
    (hset : st.setLane cta warp lane laneStateCore = some st')
    (hstatus : laneStateCore.status = laneState.status)
    (hpc : laneStateCore.pc = pc') :
    warpAt cta warp pc' [lane] st' CSL.Resource.empty := by
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hwarpCore :
      st'.getWarp? cta warp = some (warpState.setLane lane laneStateCore) :=
    State.getWarp?_setLane_same hwarp hset
  have hlanesCore :
      Helpers.runnableLaneIds (warpState.setLane lane laneStateCore) = [lane] := by
    rw [Helpers.runnableLaneIds_setLane_control_eq hwarpLane hstatus, hlanes]
  have hlaneCore :
      (warpState.setLane lane laneStateCore).getLane? lane = some laneStateCore :=
    WarpState.getLane?_setLane_same hwarpLane
  have hrpcCore : Helpers.RunnablePc (warpState.setLane lane laneStateCore) pc' := by
    unfold Helpers.RunnablePc Helpers.currentRunnablePc?
    rw [hlanesCore]
    simp [hlaneCore, hpc]
  have hpartCore :
      Helpers.ParticipatingRunnable (warpState.setLane lane laneStateCore) none [lane] := by
    unfold Helpers.ParticipatingRunnable Helpers.participatingRunnableLaneIds?
    change Helpers.currentRunnablePc? (warpState.setLane lane laneStateCore) = some pc'
      at hrpcCore
    rw [hrpcCore]
    simp [hlanesCore, hlaneCore, hpc, Helpers.participatingRunnableLaneIdsFrom?,
      Helpers.guardHolds?]
  have hlockCore : Helpers.lockstepRunnable (warpState.setLane lane laneStateCore) := by
    unfold Helpers.lockstepRunnable Helpers.lockstepRunnable?
    change Helpers.currentRunnablePc? (warpState.setLane lane laneStateCore) = some pc'
      at hrpcCore
    rw [hrpcCore]
    simp [hlanesCore, hlaneCore, hpc]
  exact ⟨⟨warpState.setLane lane laneStateCore, hwarpCore, hlockCore, hrpcCore, hpartCore⟩, rfl⟩

theorem warpAt_lanes_of_set_pc
    {cta : CTAId} {warp : WarpId} {pc pc' : PC} {lanes : List LaneId}
    {st st' : State} {warpState : WarpState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hpart : Helpers.ParticipatingRunnable warpState none lanes)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun _ laneState => some { laneState with pc := pc' }) = some st') :
    warpAt cta warp pc' lanes st' CSL.Resource.empty := by
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
      (hpres := by
        intro lane old new hf
        injection hf with hnew
        subst new
        simp)
      hwarp happly with
    ⟨warpFinal, hwarpFinal, hrunFinalCore⟩
  have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
    rw [hrunFinalCore, hlanesStart]
  have hrpcFinal : Helpers.RunnablePc warpFinal pc' := by
    unfold Helpers.RunnablePc Helpers.currentRunnablePc?
    cases hlanes : lanes with
    | nil =>
        unfold Helpers.RunnablePc Helpers.currentRunnablePc? at hrpc
        simp [hlanesStart, hlanes] at hrpc
    | cons first rest =>
        have hmemFirstLanes : first ∈ lanes := by
          rw [hlanes]
          simp
        have hmemFirstInitial : first ∈ Helpers.runnableLaneIds warpState := by
          rw [hlanesStart]
          exact hmemFirstLanes
        rcases Helpers.runnableLaneIds_mem_getLane hmemFirstInitial with
          ⟨firstState, hfirstWarp⟩
        have hfirstSt : st.getLane? cta warp first = some firstState := by
          unfold State.getLane?
          simp [hwarp, hfirstWarp]
        rcases Helpers.applyToLaneIds?_set_lane_pc_of_mem
            hnodup hmemFirstLanes hfirstSt happly with
          ⟨firstFinal, hfirstFinal, hfirstPc, _hfirstStatus⟩
        unfold State.getLane? at hfirstFinal
        simp [hwarpFinal] at hfirstFinal
        rw [hrunFinal, hlanes]
        simp [hfirstFinal, hfirstPc]
  have hlockFinal : Helpers.lockstepRunnable warpFinal := by
    unfold Helpers.lockstepRunnable Helpers.lockstepRunnable?
    change Helpers.currentRunnablePc? warpFinal = some pc' at hrpcFinal
    rw [hrpcFinal]
    simp
    intro lane hmemFinal
    have hmemLane : lane ∈ lanes := by
      rw [← hrunFinal]
      exact hmemFinal
    have hmemInitial : lane ∈ Helpers.runnableLaneIds warpState := by
      rw [hlanesStart]
      exact hmemLane
    rcases Helpers.runnableLaneIds_mem_getLane hmemInitial with
      ⟨laneState, hlaneWarp⟩
    have hlaneSt : st.getLane? cta warp lane = some laneState := by
      unfold State.getLane?
      simp [hwarp, hlaneWarp]
    rcases Helpers.applyToLaneIds?_set_lane_pc_of_mem
        hnodup hmemLane hlaneSt happly with
      ⟨laneFinal, hlaneFinal, hpcFinal, _hstatusFinal⟩
    unfold State.getLane? at hlaneFinal
    simp [hwarpFinal] at hlaneFinal
    simp [hlaneFinal, hpcFinal]
  have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
    unfold Helpers.ParticipatingRunnable
    rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
      hlockFinal hrpcFinal]
    rw [hrunFinal]
  exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩

theorem lanesTerminatedAt_of_set_terminated
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId}
    {st st' : State} {warpState : WarpState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hpart : Helpers.ParticipatingRunnable warpState none lanes)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun _ laneState => some { laneState with status := .terminated }) = some st') :
    lanesTerminatedAt cta warp lanes pc st' CSL.Resource.empty := by
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  refine ⟨?_, rfl⟩
  intro lane hmemLane
  have hmemInitial : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanesStart]
    exact hmemLane
  rcases Helpers.runnableLaneIds_mem_getLane hmemInitial with
    ⟨laneState, hlaneWarp⟩
  have hlaneSt : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hlaneWarp]
  have hpcLane : laneState.pc = pc :=
    Helpers.lockstepRunnable_mem_pc hlock hrpc hmemInitial hlaneWarp
  rcases Helpers.applyToLaneIds?_terminate_lane_of_mem
      hnodup hmemLane hlaneSt happly with
    ⟨laneFinal, hlaneFinal, hstatusFinal, hpcFinal⟩
  exact ⟨laneFinal, hlaneFinal, hstatusFinal, hpcFinal.trans hpcLane⟩

def regsFor (cta : CTAId) (warp : WarpId) :
    List LaneId → RegName → List Value → CSL.Assertion
  | [], _, [] => CSL.emp
  | lane :: lanes, name, value :: values =>
      CSL.reg cta warp lane name value ∗ regsFor cta warp lanes name values
  | _, _, _ => CSL.pure False

def predsFor (cta : CTAId) (warp : WarpId) :
    List LaneId → PredName → List Bool → CSL.Assertion
  | [], _, [] => CSL.emp
  | lane :: lanes, name, value :: values =>
      CSL.pred cta warp lane name value ∗ predsFor cta warp lanes name values
  | _, _, _ => CSL.pure False

def globalSlices : List Nat → CSL.BytePerm → List (List Byte) → CSL.Assertion
  | [], _, [] => CSL.emp
  | offset :: offsets, perm, bytes :: rest =>
      CSL.globalBytes offset perm bytes ∗ globalSlices offsets perm rest
  | _, _, _ => CSL.pure False

def sharedSlices (cta : CTAId) :
    List Nat → CSL.BytePerm → List (List Byte) → CSL.Assertion
  | [], _, [] => CSL.emp
  | offset :: offsets, perm, bytes :: rest =>
      CSL.sharedBytes cta offset perm bytes ∗ sharedSlices cta offsets perm rest
  | _, _, _ => CSL.pure False

def localSlices (cta : CTAId) (warp : WarpId) :
    List LaneId → List Nat → CSL.BytePerm → List (List Byte) → CSL.Assertion
  | [], [], _, [] => CSL.emp
  | lane :: lanes, offset :: offsets, perm, bytes :: rest =>
      CSL.localBytes cta warp lane offset perm bytes ∗
        localSlices cta warp lanes offsets perm rest
  | _, _, _, _ => CSL.pure False

@[simp] theorem regsFor_nil (cta : CTAId) (warp : WarpId) (name : RegName) :
    regsFor cta warp [] name [] = CSL.emp :=
  rfl

@[simp] theorem predsFor_nil (cta : CTAId) (warp : WarpId) (name : PredName) :
    predsFor cta warp [] name [] = CSL.emp :=
  rfl

@[simp] theorem globalSlices_nil (perm : CSL.BytePerm) :
    globalSlices [] perm [] = CSL.emp :=
  rfl

@[simp] theorem sharedSlices_nil (cta : CTAId) (perm : CSL.BytePerm) :
    sharedSlices cta [] perm [] = CSL.emp :=
  rfl

@[simp] theorem localSlices_nil
    (cta : CTAId) (warp : WarpId) (perm : CSL.BytePerm) :
    localSlices cta warp [] [] perm [] = CSL.emp :=
  rfl

theorem regsFor_cons
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (lanes : List LaneId)
    (name : RegName) (value : Value) (values : List Value) :
    regsFor cta warp (lane :: lanes) name (value :: values) =
      (CSL.reg cta warp lane name value ∗ regsFor cta warp lanes name values) :=
  rfl

theorem predsFor_cons
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (lanes : List LaneId)
    (name : PredName) (value : Bool) (values : List Bool) :
    predsFor cta warp (lane :: lanes) name (value :: values) =
      (CSL.pred cta warp lane name value ∗ predsFor cta warp lanes name values) :=
  rfl

theorem globalSlices_cons
    (offset : Nat) (offsets : List Nat) (perm : CSL.BytePerm)
    (bytes : List Byte) (rest : List (List Byte)) :
    globalSlices (offset :: offsets) perm (bytes :: rest) =
      (CSL.globalBytes offset perm bytes ∗ globalSlices offsets perm rest) :=
  rfl

theorem sharedSlices_cons
    (cta : CTAId) (offset : Nat) (offsets : List Nat) (perm : CSL.BytePerm)
    (bytes : List Byte) (rest : List (List Byte)) :
    sharedSlices cta (offset :: offsets) perm (bytes :: rest) =
      (CSL.sharedBytes cta offset perm bytes ∗ sharedSlices cta offsets perm rest) :=
  rfl

theorem localSlices_cons
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (lanes : List LaneId)
    (offset : Nat) (offsets : List Nat) (perm : CSL.BytePerm)
    (bytes : List Byte) (rest : List (List Byte)) :
    localSlices cta warp (lane :: lanes) (offset :: offsets) perm (bytes :: rest) =
      (CSL.localBytes cta warp lane offset perm bytes ∗
        localSlices cta warp lanes offsets perm rest) :=
  rfl

end WP
end CLean
