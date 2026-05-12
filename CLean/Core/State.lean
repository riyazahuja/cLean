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
