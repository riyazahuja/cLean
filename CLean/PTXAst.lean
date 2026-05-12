import CLean.IR

namespace CLean

namespace PTX

inductive Operand where
  | reg (r : RegName)
  | pred (p : PredName)
  | imm (v : Value)
  | special (s : SpecialReg)
  deriving Repr, Inhabited

inductive Instr where
  | mov (ty : ScalarTy) (dst : RegName) (src : Operand)
  | add (ty : ScalarTy) (dst : RegName) (lhs rhs : Operand)
  | setp (op : CmpOp) (ty : ScalarTy) (dst : PredName) (lhs rhs : Operand)
  | ld (space : AddrSpace) (ty : ScalarTy) (dst : RegName) (addr : Operand)
  | st (space : AddrSpace) (ty : ScalarTy) (addr value : Operand)
  | cvta (space : AddrSpace) (dst : RegName) (src : Operand)
  | isspacep (space : AddrSpace) (dst : PredName) (src : Operand)
  | barSync (barrierId : Nat)
  deriving Repr, Inhabited

structure GInstr where
  guard? : Option Guard := none
  instr : Instr
  deriving Repr, Inhabited

inductive Terminator where
  | bra (label : BlockLabel)
  | cbra (pred : PredName) (negate : Bool) (tLabel fLabel : BlockLabel)
  | exit
  deriving Repr, Inhabited

structure Block where
  label : BlockLabel
  body : Array GInstr := #[]
  term : Terminator := .exit
  deriving Repr, Inhabited

structure RegDecl where
  name : RegName
  ty : ScalarTy
  deriving Repr, Inhabited

structure PredDecl where
  name : PredName
  deriving Repr, Inhabited

structure Kernel where
  entry : BlockLabel
  gridCtx : GridCtx := { gridDim := { x := 1 }, blockDim := { x := 32 } }
  regs : Array RegDecl := #[]
  preds : Array PredDecl := #[]
  blocks : Array Block := #[]
  deriving Repr, Inhabited

end PTX

end CLean
