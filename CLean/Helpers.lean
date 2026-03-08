import CLean.State
import CLean.Typing

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

-- Milestone 1 assumes 1D blocks whose warps partition block threads contiguously.
def blockLinearTidX (warp : WarpId) (lane : LaneId) : Nat :=
  warp * 32 + lane.val

def laneIds : List LaneId :=
  List.finRange 32

def readReg (lane : LaneState) (r : RegName) : Option Value :=
  lane.regs[r]?

def writeReg (lane : LaneState) (r : RegName) (v : Value) : LaneState :=
  { lane with regs := lane.regs.insert r v }

def readPred (lane : LaneState) (p : PredName) : Option Bool :=
  lane.preds[p]?

def writePred (lane : LaneState) (p : PredName) (b : Bool) : LaneState :=
  { lane with preds := lane.preds.insert p b }

def evalSpecial (grid : GridCtx) (cta : CTAId) (warp : WarpId) (lane : LaneId) : SpecialReg → Value
  | .tidX => .u32 <| UInt32.ofNat (blockLinearTidX warp lane)
  | .tidY => .u32 0
  | .tidZ => .u32 0
  | .ctaidX => .u32 <| UInt32.ofNat cta
  | .ctaidY => .u32 0
  | .ctaidZ => .u32 0
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

private def natToBytesLE (n width : Nat) : List Byte :=
  (List.range width).map fun i => UInt8.ofNat ((n / (2 ^ (8 * i))) % 256)

private def bytesToNatLE (bs : List Byte) : Nat :=
  let rec loop (i : Nat) : List Byte → Nat
    | [] => 0
    | b :: rest => b.toNat * (2 ^ (8 * i)) + loop (i + 1) rest
  loop 0 bs

private def signedToNat (bits : Nat) (x : Int) : Nat :=
  let modulus : Int := Int.ofNat (2 ^ bits)
  Int.toNat (x % modulus)

private def natToSigned (bits : Nat) (n : Nat) : Int :=
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

private def readBytes? (mem : ByteMem) (offset width : Nat) : Option (List Byte) :=
  let rec loop (i : Nat) (acc : List Byte) :=
    if i < width then
      match mem[offset + i]? with
      | some b => loop (i + 1) (b :: acc)
      | none => none
    else
      some acc.reverse
  loop 0 []

private def writeBytes (mem : ByteMem) (offset : Nat) (bytes : List Byte) : ByteMem :=
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

def participatingRunnableLaneIds? (warp : WarpState) (guard? : Option Guard) : Option (List LaneId) := do
  let pc <- currentRunnablePc? warp
  let lanes := runnableLaneIds warp
  let mut out : List LaneId := []
  for lane in lanes do
    let some laneState := warp.getLane? lane | none
    if laneState.pc = pc then
      let passes <- guardHolds? laneState guard?
      if passes then
        out := lane :: out
  pure out.reverse

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

private def getSpaceBaseMem? (st : State) (addr : Addr) : Option ByteMem :=
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

private def setSpaceBaseMem? (st : State) (addr : Addr) (bytes : ByteMem) : Option State :=
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

mutual
  partial def evalRValue? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : RValue → Option Value
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

  partial def evalUnary? : ScalarUnaryOp → Value → Option Value
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
    | .cvt .u64, .s64 x => some (.u64 (UInt64.ofNat (signedToNat 64 x)))
    | .cvt .s32, .u32 x => some (.s32 (normalizeSigned 32 (Int.ofNat x.toNat)))
    | .cvt .s64, .u64 x => some (.s64 (normalizeSigned 64 (Int.ofNat x.toNat)))
    | .cvt .f32, .u32 x => some (.f32 x.toFloat)
    | .cvt .f32, .s32 x => some (.f32 (intToFloat x))
    | .cvt .f64, .u64 x => some (.f64 x.toFloat)
    | .cvt .f64, .s64 x => some (.f64 (intToFloat x))
    | .cvt dst, v =>
        if let some ty := Typing.valueType? v then
          if dst = ty then some v else none
        else none
    | _, _ => none

  partial def evalBinary? : ScalarBinaryOp → Value → Value → Option Value
    | .add, .u32 a, .u32 b => some (.u32 (a + b))
    | .add, .u64 a, .u64 b => some (.u64 (a + b))
    | .add, .s32 a, .s32 b => some (.s32 (normalizeSigned 32 (a + b)))
    | .add, .s64 a, .s64 b => some (.s64 (normalizeSigned 64 (a + b)))
    | .add, .f32 a, .f32 b => some (.f32 (a + b))
    | .add, .f64 a, .f64 b => some (.f64 (a + b))
    | .sub, .u32 a, .u32 b => some (.u32 (a - b))
    | .sub, .u64 a, .u64 b => some (.u64 (a - b))
    | .sub, .s32 a, .s32 b => some (.s32 (normalizeSigned 32 (a - b)))
    | .sub, .s64 a, .s64 b => some (.s64 (normalizeSigned 64 (a - b)))
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
    | .bitor, .u32 a, .u32 b => some (.u32 (a ||| b))
    | .bitor, .u64 a, .u64 b => some (.u64 (a ||| b))
    | .bitxor, .u32 a, .u32 b => some (.u32 (u32Xor a b))
    | .bitxor, .u64 a, .u64 b => some (.u64 (u64Xor a b))
    | .shl, .u32 a, .u32 b => some (.u32 (u32Shl a b.toNat))
    | .shl, .u64 a, .u64 b => some (.u64 (u64Shl a b.toNat))
    | .shr, .u32 a, .u32 b => some (.u32 (u32Shr a b.toNat))
    | .shr, .u64 a, .u64 b => some (.u64 (u64Shr a b.toNat))
    | .min, .s32 a, .s32 b => some (.s32 (min a b))
    | .min, .s64 a, .s64 b => some (.s64 (min a b))
    | .max, .s32 a, .s32 b => some (.s32 (max a b))
    | .max, .s64 a, .s64 b => some (.s64 (max a b))
    | _, _, _ => none

  partial def evalTernary? : ScalarTernaryOp → Value → Value → Value → Option Value
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
  | .shared, .u64 off => some (.shared cta off.toNat)
  | .local, .u64 off => some (.local cta warp lane off.toNat)
  | .param, .u64 off => some (.param off.toNat)
  | .const, .u64 off => some (.const off.toNat)
  | .global, .gaddr .global off => some (.global off)
  | .shared, .gaddr .shared off => some (.shared cta off)
  | .local, .gaddr .local off => some (.local cta warp lane off)
  | .param, .gaddr .param off => some (.param off)
  | .const, .gaddr .const off => some (.const off)
  | .generic, .gaddr s off => some (.generic s off)
  | .global, .u32 off => some (.global off.toNat)
  | .shared, .u32 off => some (.shared cta off.toNat)
  | .local, .u32 off => some (.local cta warp lane off.toNat)
  | .param, .u32 off => some (.param off.toNat)
  | .const, .u32 off => some (.const off.toNat)
  | _, _ => none

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

private def advancePcForLane (laneState : LaneState) : LaneState :=
  let (lbl, idx) := laneState.pc
  { laneState with pc := (lbl, idx + 1) }

private def applyToLaneIds? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId)
    (f : LaneId → LaneState → Option LaneState) : Option State := do
  let mut cur := st
  for lane in lanes do
    let laneState <- cur.getLane? cta warp lane
    let laneState' <- f lane laneState
    cur <- cur.setLane cta warp lane laneState'
  pure cur

def advanceRunnablePcs? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  let warpState <- st.getWarp? cta warp
  let pc <- currentRunnablePc? warpState
  let lanes := runnableLaneIds warpState |>.filter fun lane =>
    match warpState.getLane? lane with
    | some laneState => laneState.pc == pc
    | none => false
  applyToLaneIds? st cta warp lanes (fun _ laneState => some (advancePcForLane laneState))

def stepInstr? (st : State) (cta : CTAId) (warp : WarpId) (gi : GInstr) : Option State := do
  let warpState <- st.getWarp? cta warp
  if !lockstepRunnable? warpState then
    none
  else
    let participants <- participatingRunnableLaneIds? warpState gi.guard?
    let st <- match gi.instr with
      | .assignReg dst rhs =>
          applyToLaneIds? st cta warp participants fun lane laneState => do
            let v <- evalRValue? st cta warp lane rhs
            pure (writeReg laneState dst v)
      | .assignPred dst cmp =>
          applyToLaneIds? st cta warp participants fun lane laneState => do
            let b <- evalCmp? st cta warp lane cmp
            pure (writePred laneState dst b)
      | .load dst src =>
          applyToLaneIds? st cta warp participants fun lane laneState => do
            let addr <- resolveAddr? st cta warp lane src
            let value <- readMem? st src.space src.ty addr
            pure (writeReg laneState dst value)
      | .store dst value =>
          -- Milestone 1 deterministic simplification: stores are sequentialized in lane order.
          let mut cur := st
          for lane in participants do
            let addr <- resolveAddr? cur cta warp lane dst
            let v <- evalRValue? cur cta warp lane value
            cur <- writeMem? cur dst.space dst.ty addr v
          pure cur
      | .cvta _ _ _ => none
      | .isspacep _ _ _ => none
      | .barrierCTA _ => none
      | .warp _ => none
      | .atomic _ _ _ _ _ => none
      | .mma _ => none
    advanceRunnablePcs? st cta warp

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

end Helpers

end CLean
