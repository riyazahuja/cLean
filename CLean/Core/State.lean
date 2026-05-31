import Std
import CLean.Core.Types
import CLean.Core.IR

namespace CLean

abbrev Byte := UInt8
abbrev ByteMem := Std.HashMap Nat Byte

structure GlobalMem where
  bytes : ByteMem := {}
  deriving Repr, Inhabited

structure SharedMem where
  bytes : ByteMem := {}
  deriving Repr, Inhabited

structure LocalMem where
  bytes : ByteMem := {}
  deriving Repr, Inhabited

structure ParamMem where
  bytes : ByteMem := {}
  deriving Repr, Inhabited

structure ConstMem where
  bytes : ByteMem := {}
  deriving Repr, Inhabited

inductive LaneStatus where
  | running
  | blockedBarrier
  | terminated
  deriving Repr, DecidableEq, Inhabited

structure LaneState where
  regs : Std.HashMap RegName Value := {}
  preds : Std.HashMap PredName Bool := {}
  localMem : LocalMem := {}
  pc : PC := ("entry", 0)
  status : LaneStatus := .running
  deriving Repr, Inhabited

structure WarpState where
  lanes : Array LaneState := Array.replicate 32 {}
  activeMask : UInt32 := 0xFFFFFFFF
  exitedMask : UInt32 := 0
  deriving Repr, Inhabited

structure BarrierInstance where
  epoch : Nat := 0
  arrived : List (WarpId × LaneId) := []
  expectedCount : Nat := 0
  deriving Repr, Inhabited

structure BarrierState where
  bars : Std.HashMap Nat BarrierInstance := {}
  deriving Repr, Inhabited

structure AtomicEvent where
  cta : CTAId
  warp : WarpId
  lane : LaneId
  op : AtomicOp
  addr : Addr
  before : Option Value
  arg1 : Option Value
  arg2 : Option Value
  after : Option Value
  deriving Repr

structure AtomicState where
  log : Array AtomicEvent := #[]
  deriving Repr, Inhabited

structure CTAState where
  shared : SharedMem := {}
  warps : Std.HashMap WarpId WarpState := {}
  barrier : BarrierState := {}
  deriving Repr, Inhabited

structure KernelEnv where
  entry : String := "entry"
  gridCtx : GridCtx := { gridDim := { x := 1 }, blockDim := { x := 32 } }
  addrLayout : AddrLayout := {}
  params : Array ParamInfo := #[]
  sharedDecls : Array SharedDecl := #[]
  blocks : Std.HashMap BlockLabel Block := {}
  deriving Repr, Inhabited

structure State where
  kernelEnv : KernelEnv := {}
  global : GlobalMem := {}
  const : ConstMem := {}
  param : ParamMem := {}
  ctas : Std.HashMap CTAId CTAState := {}
  atomics : AtomicState := {}
  deriving Repr, Inhabited

namespace WarpState

def getLane? (warp : WarpState) (lane : LaneId) : Option LaneState :=
  warp.lanes[lane.val]?

def setLane (warp : WarpState) (lane : LaneId) (laneState : LaneState) : WarpState :=
  { warp with lanes := warp.lanes.set! lane.val laneState }

theorem getLane?_setLane_same
    {warp : WarpState} {lane : LaneId} {old new : LaneState}
    (hget : warp.getLane? lane = some old) :
    (warp.setLane lane new).getLane? lane = some new := by
  unfold getLane? at hget ⊢
  unfold setLane
  rcases (Array.getElem?_eq_some_iff.mp hget) with ⟨hbound, _⟩
  simp [Array.getElem?_set_self, hbound]

theorem getLane?_setLane_ne
    {warp : WarpState} {lane target : LaneId} {old new : LaneState}
    (hget : warp.getLane? lane = some old)
    (hne : target ≠ lane) :
    (warp.setLane lane new).getLane? target = warp.getLane? target := by
  unfold getLane? at hget ⊢
  unfold setLane
  rcases (Array.getElem?_eq_some_iff.mp hget) with ⟨hbound, _⟩
  have hneVal : lane.val ≠ target.val := by
    intro hval
    exact hne (Fin.ext hval.symm)
  simp [Array.getElem?_set_ne, hbound, hneVal]

def wf? (warp : WarpState) : Bool :=
  warp.lanes.size == 32

def wf (warp : WarpState) : Prop :=
  warp.wf? = true

theorem wf_iff_bool (warp : WarpState) : WarpState.wf warp ↔ warp.wf? = true := Iff.rfl

instance (warp : WarpState) : Decidable (WarpState.wf warp) := by
  unfold WarpState.wf
  infer_instance

end WarpState

namespace CTAState

def wf? (cta : CTAState) : Bool :=
  cta.warps.toList.all fun (_, warpState) => warpState.wf?

def wf (cta : CTAState) : Prop :=
  cta.wf? = true

theorem wf_iff_bool (cta : CTAState) : CTAState.wf cta ↔ cta.wf? = true := Iff.rfl

instance (cta : CTAState) : Decidable (CTAState.wf cta) := by
  unfold CTAState.wf
  infer_instance

end CTAState

namespace KernelEnv

def wf? (env : KernelEnv) : Bool :=
  env.blocks.toList.all fun (entry : BlockLabel × Block) => entry.2.label == entry.1

def wf (env : KernelEnv) : Prop :=
  env.wf? = true

theorem wf_iff_bool (env : KernelEnv) : KernelEnv.wf env ↔ env.wf? = true := Iff.rfl

instance (env : KernelEnv) : Decidable (KernelEnv.wf env) := by
  unfold KernelEnv.wf
  infer_instance

end KernelEnv

namespace State

def getCTA? (st : State) (cta : CTAId) : Option CTAState :=
  st.ctas[cta]?

def setCTA (st : State) (cta : CTAId) (ctaState : CTAState) : State :=
  { st with ctas := st.ctas.insert cta ctaState }

def getWarp? (st : State) (cta : CTAId) (warp : WarpId) : Option WarpState := do
  let ctaState <- st.getCTA? cta
  ctaState.warps[warp]?

def setWarp (st : State) (cta : CTAId) (warp : WarpId) (warpState : WarpState) : Option State := do
  let ctaState <- st.getCTA? cta
  let ctaState := { ctaState with warps := ctaState.warps.insert warp warpState }
  pure <| st.setCTA cta ctaState

def getLane? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : Option LaneState := do
  let warpState <- st.getWarp? cta warp
  warpState.getLane? lane

def setLane (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (laneState : LaneState) : Option State := do
  let warpState <- st.getWarp? cta warp
  let warpState := warpState.setLane lane laneState
  st.setWarp cta warp warpState

theorem getLane?_setLane_same
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {old new : LaneState}
    (hget : st.getLane? cta warp lane = some old)
    (hset : st.setLane cta warp lane new = some st') :
    st'.getLane? cta warp lane = some new := by
  unfold getLane? at hget ⊢
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [getWarp?, hcta] at hget
  | some ctaState =>
      cases hwarp : ctaState.warps[warp]? with
      | none =>
          simp [getWarp?, hcta, hwarp] at hget
      | some warpState =>
          simp [getWarp?, hcta, hwarp] at hget hset
          rw [← hset]
          simp [getCTA?, getWarp?, setCTA, hcta]
          exact WarpState.getLane?_setLane_same hget

theorem getLane?_setLane_ne
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane target : LaneId}
    {old new : LaneState}
    (hget : st.getLane? cta warp lane = some old)
    (hne : target ≠ lane)
    (hset : st.setLane cta warp lane new = some st') :
    st'.getLane? cta warp target = st.getLane? cta warp target := by
  unfold getLane? at hget ⊢
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [getWarp?, hcta] at hget
  | some ctaState =>
      cases hwarp : ctaState.warps[warp]? with
      | none =>
          simp [getWarp?, hcta, hwarp] at hget
      | some warpState =>
          simp [getWarp?, hcta, hwarp] at hget hset
          rw [← hset]
          simp [getCTA?, getWarp?, setCTA, hcta]
          have hctaRaw : st.ctas[cta]? = some ctaState := by
            simpa [getCTA?] using hcta
          simpa [hctaRaw, hwarp] using
            (WarpState.getLane?_setLane_ne (warp := warpState) (lane := lane)
              (target := target) (old := old) (new := new) hget hne)

theorem getWarp?_setLane_same
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {warpState : WarpState} {laneState : LaneState}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.getWarp? cta warp = some (warpState.setLane lane laneState) := by
  unfold getWarp? at hwarp ⊢
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [hcta] at hwarp
  | some ctaState =>
      cases hwarpMap : ctaState.warps[warp]? with
      | none =>
          simp [hcta, hwarpMap] at hwarp
      | some warpState' =>
          simp [hcta, hwarpMap] at hwarp hset
          subst warpState'
          simp [getWarp?, hcta, hwarpMap] at hset
          rw [← hset]
          simp [getCTA?, setCTA, hcta, hwarpMap]

theorem getLane?_setLane_nonPc_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane target : LaneId}
    {old new targetState : LaneState}
    (htarget : st.getLane? cta warp target = some targetState)
    (hget : st.getLane? cta warp lane = some old)
    (hlocal : new.localMem = old.localMem)
    (hregs : new.regs = old.regs)
    (hpreds : new.preds = old.preds)
    (hset : st.setLane cta warp lane new = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem ∧
      targetState'.regs = targetState.regs ∧
      targetState'.preds = targetState.preds := by
  by_cases heq : target = lane
  · subst target
    rw [hget] at htarget
    injection htarget with hsame
    subst targetState
    exact ⟨new, getLane?_setLane_same hget hset, hlocal, hregs, hpreds⟩
  · have htarget' := getLane?_setLane_ne hget heq hset
    rw [htarget] at htarget'
    exact ⟨targetState, htarget', rfl, rfl, rfl⟩

theorem getLane?_setLane_localMem_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane target : LaneId}
    {old new targetState : LaneState}
    (htarget : st.getLane? cta warp target = some targetState)
    (hget : st.getLane? cta warp lane = some old)
    (hlocal : new.localMem = old.localMem)
    (hset : st.setLane cta warp lane new = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem := by
  by_cases heq : target = lane
  · subst target
    rw [hget] at htarget
    injection htarget with hsame
    subst targetState
    exact ⟨new, getLane?_setLane_same hget hset, hlocal⟩
  · have htarget' := getLane?_setLane_ne hget heq hset
    rw [htarget] at htarget'
    exact ⟨targetState, htarget', rfl⟩

theorem setLane_global_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.global = st.global := by
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [getWarp?, hcta] at hset
  | some ctaState =>
      cases hwarp : ctaState.warps[warp]? with
      | none =>
          simp [getWarp?, hcta, hwarp] at hset
      | some warpState =>
          simp [getWarp?, hcta, hwarp] at hset
          rw [← hset]
          simp [setCTA]

theorem setLane_param_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.param = st.param := by
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [getWarp?, hcta] at hset
  | some ctaState =>
      cases hwarp : ctaState.warps[warp]? with
      | none =>
          simp [getWarp?, hcta, hwarp] at hset
      | some warpState =>
          simp [getWarp?, hcta, hwarp] at hset
          rw [← hset]
          simp [setCTA]

theorem setLane_const_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.const = st.const := by
  unfold setLane setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [getWarp?, hcta] at hset
  | some ctaState =>
      cases hwarp : ctaState.warps[warp]? with
      | none =>
          simp [getWarp?, hcta, hwarp] at hset
      | some warpState =>
          simp [getWarp?, hcta, hwarp] at hset
          rw [← hset]
          simp [setCTA]

theorem setCTA_kernelEnv_eq
    {st : State} {cta : CTAId} {ctaState : CTAState} :
    (st.setCTA cta ctaState).kernelEnv = st.kernelEnv := by
  rfl

theorem setWarp_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    (hset : st.setWarp cta warp warpState = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold setWarp at hset
  cases hcta : st.getCTA? cta with
  | none =>
      simp [hcta] at hset
  | some ctaState =>
      simp [hcta] at hset
      rw [← hset]
      rfl

theorem setLane_kernelEnv_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold setLane at hset
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hset
  | some warpState =>
      simp [hwarp] at hset
      exact setWarp_kernelEnv_eq hset

theorem setLane_getCTA_shared_eq
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (hset : st.setLane cta warp lane laneState = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧ ctaState'.shared = ctaState.shared := by
  unfold setLane setWarp at hset
  cases hwarp : ctaState.warps[warp]? with
  | none =>
      simp [getWarp?, hcta, hwarp] at hset
  | some warpState =>
      simp [getWarp?, hcta, hwarp] at hset
      rw [← hset]
      refine ⟨{ ctaState with
        warps := ctaState.warps.insert warp (warpState.setLane lane laneState) }, ?_, rfl⟩
      simp [getCTA?, setCTA]

def wf? (st : State) : Bool :=
  st.kernelEnv.wf? && st.ctas.toList.all fun (_, ctaState) => ctaState.wf?

def wf (st : State) : Prop :=
  st.wf? = true

theorem wf_iff_bool (st : State) : State.wf st ↔ st.wf? = true := Iff.rfl

instance (st : State) : Decidable (State.wf st) := by
  unfold State.wf
  infer_instance

end State

end CLean
