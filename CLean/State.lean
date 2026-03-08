import Std
import CLean.Types
import CLean.IR

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

end WarpState

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

end State

end CLean
