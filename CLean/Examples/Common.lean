import CLean.PTX.Parser
import CLean.PTX.Lowering
import CLean.Proof.Automation
import CLean.Proof.Loops
import Mathlib.Tactic

namespace CLean

open Helpers

def lane0 : LaneId := ⟨0, by decide⟩

def lane0HasRegU32 (reg : RegName) (value : UInt32) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs[reg]? with
      | some (.u32 v) => v == value
      | _ => false
  | none => false

def lane0HasRegS32 (reg : RegName) (value : Int) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs[reg]? with
      | some (.s32 v) => decide (v = value)
      | _ => false
  | none => false

def lane0HasR1Seven (st : State) : Bool :=
  lane0HasRegU32 "r1" 7 st

def lane0Terminated (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .terminated
  | none => false

def lane0HasGlobalAddr (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["gp"]? with
      | some (.gaddr .global 64) => true
      | _ => false
  | none => false

def lane0PredQTrue (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.preds["q"]? with
      | some true => true
      | _ => false
  | none => false

def copySrcOffset : Nat := 0

def copyDstOffset : Nat := 4

def copyValue : UInt32 := 99

def activeMaskPrefix (n : Nat) : UInt32 :=
  UInt32.ofNat (2 ^ n - 1)

def writeU32Bytes (mem : ByteMem) (off : Nat) (x : UInt32) : ByteMem :=
  Helpers.writeBytes mem off (Helpers.natToBytesLE x.toNat 4)

def writeU64Bytes (mem : ByteMem) (off : Nat) (x : UInt64) : ByteMem :=
  Helpers.writeBytes mem off (Helpers.natToBytesLE x.toNat 8)

def writeF32Bytes (mem : ByteMem) (off : Nat) (x : Float) : ByteMem :=
  Helpers.writeBytes mem off (Helpers.natToBytesLE x.toFloat32.toBits.toNat 4)

def writeF32Vector (mem : ByteMem) (base : Nat) (xs : List Float) : ByteMem :=
  let rec loop (i : Nat) (mem : ByteMem) : List Float → ByteMem
    | [] => mem
    | x :: xs => loop (i + 1) (writeF32Bytes mem (base + i * 4) x) xs
  loop 0 mem xs

def readGlobalF32? (st : State) (off : Nat) : Option Float := do
  match Helpers.readMem? st .global .f32 (.global off) with
  | some (.f32 x) => some x
  | _ => none

def globalF32VectorMatches? (st : State) (base : Nat) (expected : List Float) : Bool :=
  let rec loop (i : Nat) : List Float → Bool
    | [] => true
    | x :: xs =>
      match readGlobalF32? st (base + i * 4) with
      | some y => y == x && loop (i + 1) xs
      | none => false
  loop 0 expected

def copyDstHasValue (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global copyDstOffset) with
  | some (.u32 v) => v == copyValue
  | _ => false

def copyLane0Terminated (st : State) : Bool :=
  lane0Terminated st

def lane0RunningAt (pc : PC) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .running && laneState.pc == pc
  | none => false

def barrier0Released (st : State) : Bool :=
  match st.getCTA? 0 with
  | some ctaState =>
      match ctaState.barrier.bars[0]? with
      | some inst => inst.epoch == 1 && inst.arrived.length == 0
      | none => false
  | none => false

end CLean
