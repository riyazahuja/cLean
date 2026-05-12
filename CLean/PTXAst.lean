import CLean.IR

namespace CLean

namespace PTX

inductive Operand where
  | reg (r : RegName)
  | pred (p : PredName)
  | symbol (name : String)
  | addr (base : Operand) (offset : Int := 0)
  | imm (v : Value)
  | special (s : SpecialReg)
  deriving Repr, Inhabited

inductive Instr where
  | mov (ty : ScalarTy) (dst : RegName) (src : Operand)
  | unop (op : ScalarUnaryOp) (srcTy : ScalarTy) (dst : RegName) (src : Operand)
  | binop (op : ScalarBinaryOp) (ty : ScalarTy) (dst : RegName) (lhs rhs : Operand)
  | setp (op : CmpOp) (ty : ScalarTy) (dst : PredName) (lhs rhs : Operand)
  | ld (space : AddrSpace) (ty : ScalarTy) (dst : RegName) (addr : Operand)
  | st (space : AddrSpace) (ty : ScalarTy) (addr value : Operand)
  | cvta (space : AddrSpace) (dst : RegName) (src : Operand)
  | isspacep (space : AddrSpace) (dst : PredName) (src : Operand)
  | barSync (barrierId : Nat)
  | unsupported (opcode : String) (modifiers : Array String) (operands : Array Operand)
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

structure ParamDecl where
  name : String
  ty : ScalarTy
  isPtr : Bool := false
  ptrSpace? : Option AddrSpace := none
  align : Nat := 1
  deriving Repr, Inhabited

structure SharedDecl where
  name : String
  ty : ScalarTy
  count : Nat := 1
  align : Nat := 1
  deriving Repr, Inhabited

inductive ModuleDirective where
  | version (value : String)
  | target (targets : Array String)
  | addressSize (bits : Nat)
  deriving Repr, Inhabited

inductive ModuleMemorySpace where
  | global
  | const
  deriving Repr, Inhabited

structure ModuleMemoryDecl where
  space : ModuleMemorySpace
  name : String
  ty : ScalarTy
  count : Nat := 1
  align : Nat := 1
  deriving Repr, Inhabited

structure Kernel where
  entry : BlockLabel
  gridCtx : GridCtx := { gridDim := { x := 1 }, blockDim := { x := 32 } }
  regs : Array RegDecl := #[]
  preds : Array PredDecl := #[]
  params : Array ParamDecl := #[]
  shareds : Array SharedDecl := #[]
  blocks : Array Block := #[]
  deriving Repr, Inhabited

structure Module where
  directives : Array ModuleDirective := #[]
  memories : Array ModuleMemoryDecl := #[]
  shareds : Array SharedDecl := #[]
  kernels : Array Kernel := #[]
  deriving Repr, Inhabited

end PTX

end CLean
