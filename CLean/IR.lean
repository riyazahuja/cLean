import CLean.Types

namespace CLean

inductive ScalarUnaryOp where
  | mov
  | neg
  | abs
  | bitnot
  | cvt (dst : ScalarTy)
  deriving Repr, DecidableEq, Inhabited

inductive ScalarBinaryOp where
  | add | sub | mul | div | rem
  | min | max
  | bitand | bitor | bitxor
  | shl | shr
  deriving Repr, DecidableEq, Inhabited

inductive ScalarTernaryOp where
  | mad
  | fma
  | selp
  deriving Repr, DecidableEq, Inhabited

inductive CmpOp where
  | eq | ne | lt | le | gt | ge
  deriving Repr, DecidableEq, Inhabited

inductive RValue where
  | imm (v : Value)
  | reg (r : RegName)
  | pred (p : PredName)
  | special (s : SpecialReg)
  | unop (op : ScalarUnaryOp) (a : RValue)
  | binop (op : ScalarBinaryOp) (a b : RValue)
  | triop (op : ScalarTernaryOp) (a b c : RValue)
  deriving Repr, Inhabited

structure CmpExpr where
  op : CmpOp
  lhs : RValue
  rhs : RValue
  deriving Repr, Inhabited

structure Guard where
  pred : PredName
  negate : Bool := false
  deriving Repr, Inhabited

structure TypedAddr where
  space : AddrSpace
  ty : ScalarTy
  addr : RValue
  deriving Repr, Inhabited

inductive ShflMode where
  | up | down | bfly | idx
  deriving Repr, DecidableEq, Inhabited

inductive WarpOp where
  | activemask (dst : RegName)
  | shflSync (mode : ShflMode) (dst : RegName) (src laneOrDelta clamp memberMask : RValue)
  | ballotSync (dst : RegName) (pred memberMask : RValue)
  deriving Repr, Inhabited

inductive AtomicOp where
  | add | exch | cas
  deriving Repr, DecidableEq, Inhabited

structure MMAInstr where
  dst : RegName
  a : RegName
  b : RegName
  c : RegName
  fragTy : FragTy
  deriving Repr, Inhabited

inductive Instr where
  | assignReg (dst : RegName) (rhs : RValue)
  | assignPred (dst : PredName) (cmp : CmpExpr)
  | load (dst : RegName) (src : TypedAddr)
  | store (dst : TypedAddr) (value : RValue)
  | cvta (dst : RegName) (space : AddrSpace) (src : RValue)
  | isspacep (dst : PredName) (space : AddrSpace) (src : RValue)
  | barrierCTA (barrierId : Nat)
  | warp (op : WarpOp)
  | atomic (dst? : Option RegName) (op : AtomicOp) (addr : TypedAddr) (arg1 : RValue) (arg2? : Option RValue)
  | mma (spec : MMAInstr)
  deriving Repr, Inhabited

structure GInstr where
  guard? : Option Guard := none
  instr : Instr
  deriving Repr, Inhabited

inductive Terminator where
  | br (label : BlockLabel)
  | cbr (cond : RValue) (tLabel fLabel : BlockLabel)
  | terminate
  deriving Repr, Inhabited

structure Block where
  label : BlockLabel
  body : Array GInstr
  term : Terminator
  deriving Repr, Inhabited

end CLean
