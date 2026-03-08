import Std

namespace CLean

inductive ScalarTy where
  | pred
  | u8 | u16 | u32 | u64
  | s8 | s16 | s32 | s64
  | b8 | b16 | b32 | b64
  | f16 | bf16 | f32 | f64
  deriving Repr, DecidableEq, Inhabited

inductive AddrSpace where
  | global | shared | local | param | const | generic
  deriving Repr, DecidableEq, Inhabited

abbrev CTAId := Nat
abbrev WarpId := Nat
abbrev LaneId := Fin 32
abbrev RegName := String
abbrev PredName := String
abbrev BlockLabel := String
abbrev PC := BlockLabel × Nat

inductive FragTy where
  | mma_m16n16k16_f16_f16_f32
  | mma_m16n8k16_f16_f16_f32
  | mma_m16n16k16_bf16_bf16_f32
  deriving Repr, DecidableEq, Inhabited

inductive Value where
  | pred (b : Bool)
  | u8 (x : UInt8)
  | u16 (x : UInt16)
  | u32 (x : UInt32)
  | u64 (x : UInt64)
  | s8 (x : Int)
  | s16 (x : Int)
  | s32 (x : Int)
  | s64 (x : Int)
  | b8 (x : UInt8)
  | b16 (x : UInt16)
  | b32 (x : UInt32)
  | b64 (x : UInt64)
  | f16 (bits : UInt16)
  | bf16 (bits : UInt16)
  | f32 (x : Float)
  | f64 (x : Float)
  | gaddr (space : AddrSpace) (offset : Nat)
  | frag (ty : FragTy) (payload : Array Value)
  deriving Repr, Inhabited

inductive Addr where
  | global (offset : Nat)
  | shared (cta : CTAId) (offset : Nat)
  | local (cta : CTAId) (warp : WarpId) (lane : LaneId) (offset : Nat)
  | param (offset : Nat)
  | const (offset : Nat)
  | generic (space : AddrSpace) (offset : Nat)
  deriving Repr, DecidableEq, Inhabited

inductive SpecialReg where
  | tidX | tidY | tidZ
  | ctaidX | ctaidY | ctaidZ
  | ntidX | ntidY | ntidZ
  | nctaidX | nctaidY | nctaidZ
  deriving Repr, DecidableEq, Inhabited

structure Dim3 where
  x : Nat
  y : Nat := 1
  z : Nat := 1
  deriving Repr, DecidableEq, Inhabited

structure GridCtx where
  gridDim : Dim3
  blockDim : Dim3
  deriving Repr, DecidableEq, Inhabited

structure ParamInfo where
  name : String
  ty : ScalarTy
  isPtr : Bool := false
  ptrSpace? : Option AddrSpace := none
  align : Nat := 1
  offset : Nat
  size : Nat
  deriving Repr, Inhabited

structure SharedDecl where
  name : String
  size : Nat
  align : Nat := 1
  offset : Nat
  deriving Repr, Inhabited

structure ReadSet where
  regs : List RegName := []
  preds : List PredName := []
  specials : List SpecialReg := []
  deriving Repr, Inhabited

namespace ReadSet

def union (a b : ReadSet) : ReadSet :=
  { regs := a.regs ++ b.regs, preds := a.preds ++ b.preds, specials := a.specials ++ b.specials }

end ReadSet

namespace Addr

def space : Addr → AddrSpace
  | .global _ => .global
  | .shared _ _ => .shared
  | .local _ _ _ _ => .local
  | .param _ => .param
  | .const _ => .const
  | .generic s _ => s

def offset : Addr → Nat
  | .global o => o
  | .shared _ o => o
  | .local _ _ _ o => o
  | .param o => o
  | .const o => o
  | .generic _ o => o

end Addr

end CLean
