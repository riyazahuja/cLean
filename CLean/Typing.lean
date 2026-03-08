import CLean.Types
import CLean.IR

namespace CLean

namespace Typing

def byteWidth? : ScalarTy → Option Nat
  | .pred => some 1
  | .u8 | .s8 | .b8 => some 1
  | .u16 | .s16 | .b16 | .f16 | .bf16 => some 2
  | .u32 | .s32 | .b32 | .f32 => some 4
  | .u64 | .s64 | .b64 | .f64 => some 8

def alignment? (ty : ScalarTy) : Option Nat :=
  byteWidth? ty

def valueHasType : Value → ScalarTy → Prop
  | .pred _, .pred => True
  | .u8 _, .u8 => True
  | .u16 _, .u16 => True
  | .u32 _, .u32 => True
  | .u64 _, .u64 => True
  | .s8 _, .s8 => True
  | .s16 _, .s16 => True
  | .s32 _, .s32 => True
  | .s64 _, .s64 => True
  | .b8 _, .b8 => True
  | .b16 _, .b16 => True
  | .b32 _, .b32 => True
  | .b64 _, .b64 => True
  | .f16 _, .f16 => True
  | .bf16 _, .bf16 => True
  | .f32 _, .f32 => True
  | .f64 _, .f64 => True
  | .gaddr _ _, _ => False
  | .frag _ _, _ => False
  | _, _ => False

structure TypeEnv where
  regs : Std.HashMap RegName ScalarTy := {}
  preds : Std.HashMap PredName ScalarTy := {}
  deriving Inhabited

private def binarySameWidthInt? (lhs rhs : ScalarTy) : Bool :=
  match lhs, rhs with
  | .u32, .u32 | .u64, .u64 | .s32, .s32 | .s64, .s64 => true
  | _, _ => false

private def binaryFloat? (lhs rhs : ScalarTy) : Bool :=
  match lhs, rhs with
  | .f32, .f32 | .f64, .f64 => true
  | _, _ => false

def unarySig? : ScalarUnaryOp → ScalarTy → Option ScalarTy
  | .mov, ty => some ty
  | .neg, .s32 => some .s32
  | .neg, .s64 => some .s64
  | .neg, .f32 => some .f32
  | .neg, .f64 => some .f64
  | .abs, .s32 => some .s32
  | .abs, .s64 => some .s64
  | .abs, .f32 => some .f32
  | .abs, .f64 => some .f64
  | .bitnot, .b32 => some .b32
  | .bitnot, .b64 => some .b64
  | .bitnot, .u32 => some .u32
  | .bitnot, .u64 => some .u64
  | .cvt dst, _ => some dst
  | _, _ => none

def binarySig? : ScalarBinaryOp → ScalarTy → ScalarTy → Option ScalarTy
  | .add, a, b | .sub, a, b | .mul, a, b | .div, a, b | .rem, a, b
  | .min, a, b | .max, a, b =>
      if binarySameWidthInt? a b || binaryFloat? a b then some a else none
  | .bitand, a, b | .bitor, a, b | .bitxor, a, b =>
      if binarySameWidthInt? a b then some a else none
  | .shl, .u32, .u32 => some .u32
  | .shl, .u64, .u64 => some .u64
  | .shr, .u32, .u32 => some .u32
  | .shr, .u64, .u64 => some .u64
  | _, _, _ => none

def ternarySig? : ScalarTernaryOp → ScalarTy → ScalarTy → ScalarTy → Option ScalarTy
  | .mad, a, b, c => if a = b && b = c then some a else none
  | .fma, .f32, .f32, .f32 => some .f32
  | .fma, .f64, .f64, .f64 => some .f64
  | .selp, a, b, .pred => if a = b then some a else none
  | _, _, _, _ => none

def cmpSig? : CmpOp → ScalarTy → ScalarTy → Option ScalarTy
  | _, a, b =>
      if a = b && (binarySameWidthInt? a b || binaryFloat? a b) then some .pred else none

mutual
  partial def rvalueTypeOf? (env : TypeEnv) : RValue → Option ScalarTy
    | .imm v => valueType? v
    | .reg r => env.regs[r]?
    | .pred p => env.preds[p]?
    | .special s => specialType? s
    | .unop op a => do
        let ta <- rvalueTypeOf? env a
        unarySig? op ta
    | .binop op a b => do
        let ta <- rvalueTypeOf? env a
        let tb <- rvalueTypeOf? env b
        binarySig? op ta tb
    | .triop op a b c => do
        let ta <- rvalueTypeOf? env a
        let tb <- rvalueTypeOf? env b
        let tc <- rvalueTypeOf? env c
        ternarySig? op ta tb tc

  partial def valueType? : Value → Option ScalarTy
    | .pred _ => some .pred
    | .u8 _ => some .u8
    | .u16 _ => some .u16
    | .u32 _ => some .u32
    | .u64 _ => some .u64
    | .s8 _ => some .s8
    | .s16 _ => some .s16
    | .s32 _ => some .s32
    | .s64 _ => some .s64
    | .b8 _ => some .b8
    | .b16 _ => some .b16
    | .b32 _ => some .b32
    | .b64 _ => some .b64
    | .f16 _ => some .f16
    | .bf16 _ => some .bf16
    | .f32 _ => some .f32
    | .f64 _ => some .f64
    | .gaddr _ _ => none
    | .frag _ _ => none

  partial def specialType? : SpecialReg → Option ScalarTy
    | _ => some .u32
end

def cmpExprWellTyped (env : TypeEnv) (cmp : CmpExpr) : Prop :=
  match rvalueTypeOf? env cmp.lhs, rvalueTypeOf? env cmp.rhs with
  | some lhs, some rhs => cmpSig? cmp.op lhs rhs = some .pred
  | _, _ => False

def aligned (ty : ScalarTy) (addr : Addr) : Prop :=
  match alignment? ty with
  | some a => a > 0 ∧ addr.offset % a = 0
  | none => False

def aligned? (ty : ScalarTy) (addr : Addr) : Bool :=
  match alignment? ty with
  | some a => a > 0 && addr.offset % a == 0
  | none => false

def addrSpaceMatches (space : AddrSpace) (addr : Addr) : Prop :=
  match addr with
  | .generic s _ => s = space
  | _ => addr.space = space

def addrSpaceMatches? (space : AddrSpace) (addr : Addr) : Bool :=
  match addr with
  | .generic s _ => s == space
  | _ => addr.space == space

def typedAccessPreconditions (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Prop :=
  (byteWidth? ty).isSome ∧ aligned ty addr ∧ addrSpaceMatches space addr

def typedAccessPreconditions? (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Bool :=
  (byteWidth? ty).isSome && aligned? ty addr && addrSpaceMatches? space addr

end Typing

end CLean
