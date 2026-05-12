import CLean.State
import CLean.Typing
import CLean.PTXAst

namespace CLean

namespace PTX

inductive LowerError where
  | unknownReg (r : RegName)
  | unknownPred (p : PredName)
  | valueHasNoScalarType (v : Value)
  | typeMismatch (expected actual : ScalarTy)
  | unsupportedType (ty : ScalarTy)
  | unsupportedOp (op : String)
  | unsupportedTerminator
  deriving Repr

abbrev LowerM := Except LowerError

def lowerOperand : Operand → RValue
  | .reg r => .reg r
  | .pred p => .pred p
  | .imm v => .imm v
  | .special s => .special s

def operandType? (env : Typing.TypeEnv) : Operand → LowerM ScalarTy
  | .reg r =>
      match env.regs[r]? with
      | some ty => pure ty
      | none => throw (.unknownReg r)
  | .pred p =>
      match env.preds[p]? with
      | some .pred => pure .pred
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred p)
  | .imm v =>
      match Typing.valueType? v with
      | some ty => pure ty
      | none => throw (.valueHasNoScalarType v)
  | .special s =>
      match Typing.specialType? s with
      | some ty => pure ty
      | none => throw (.unsupportedOp "special")

def expectOperandType (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM Unit := do
  let actual <- operandType? env operand
  if actual == expected then
    pure ()
  else
    throw (.typeMismatch expected actual)

def ensureCodecType (ty : ScalarTy) : LowerM Unit :=
  if Typing.scalarCodecSupported? ty && (Typing.byteWidth? ty).isSome then
    pure ()
  else
    throw (.unsupportedType ty)

def ensureCvtaSourceType (ty : ScalarTy) : LowerM Unit :=
  if Typing.cvtaSourceSupported? ty then
    pure ()
  else
    throw (.unsupportedType ty)

def ensureGuard? (env : Typing.TypeEnv) : Option Guard → LowerM Unit
  | none => pure ()
  | some g =>
      match env.preds[g.pred]? with
      | some .pred => pure ()
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred g.pred)

def lowerInstr : Instr → CLean.Instr
  | .mov _ dst src => .assignReg dst (lowerOperand src)
  | .add _ dst lhs rhs => .assignReg dst (.binop .add (lowerOperand lhs) (lowerOperand rhs))
  | .setp op _ dst lhs rhs => .assignPred dst { op := op, lhs := lowerOperand lhs, rhs := lowerOperand rhs }
  | .ld space ty dst addr => .load dst { space := space, ty := ty, addr := lowerOperand addr }
  | .st space ty addr value => .store { space := space, ty := ty, addr := lowerOperand addr } (lowerOperand value)
  | .cvta space dst src => .cvta dst space (lowerOperand src)
  | .isspacep space dst src => .isspacep dst space (lowerOperand src)
  | .barSync barrierId => .barrierCTA barrierId

def lowerInstrChecked? (env : Typing.TypeEnv) : Instr → LowerM (CLean.Instr × Typing.TypeEnv)
  | .mov ty dst src => do
      expectOperandType env ty src
      pure (.assignReg dst (lowerOperand src), { env with regs := env.regs.insert dst ty })
  | .add ty dst lhs rhs => do
      expectOperandType env ty lhs
      expectOperandType env ty rhs
      match Typing.binarySig? .add ty ty with
      | some outTy =>
          pure (.assignReg dst (.binop .add (lowerOperand lhs) (lowerOperand rhs)),
            { env with regs := env.regs.insert dst outTy })
      | none => throw (.unsupportedOp "add")
  | .setp op ty dst lhs rhs => do
      expectOperandType env ty lhs
      expectOperandType env ty rhs
      match Typing.cmpSig? op ty ty with
      | some .pred =>
          pure (.assignPred dst { op := op, lhs := lowerOperand lhs, rhs := lowerOperand rhs },
            { env with preds := env.preds.insert dst .pred })
      | _ => throw (.unsupportedOp "setp")
  | .ld space ty dst addr => do
      ensureCodecType ty
      let addrTy <- operandType? env addr
      match addrTy with
      | .u32 | .u64 =>
          pure (.load dst { space := space, ty := ty, addr := lowerOperand addr },
            { env with regs := env.regs.insert dst ty })
      | _ => throw (.unsupportedType addrTy)
  | .st space ty addr value => do
      ensureCodecType ty
      let addrTy <- operandType? env addr
      match addrTy with
      | .u32 | .u64 =>
          expectOperandType env ty value
          pure (.store { space := space, ty := ty, addr := lowerOperand addr } (lowerOperand value), env)
      | _ => throw (.unsupportedType addrTy)
  | .cvta space dst src => do
      let srcTy <- operandType? env src
      ensureCvtaSourceType srcTy
      pure (.cvta dst space (lowerOperand src), { env with regs := env.regs.insert dst .u64 })
  | .isspacep space dst src => do
      let srcTy <- operandType? env src
      -- In the initial checked layer, generic addresses have no ScalarTy, so this accepts integer
      -- pointer-shaped operands and relies on the semantic helper to reject non-generic runtime values.
      ensureCvtaSourceType srcTy
      pure (.isspacep dst space (lowerOperand src), { env with preds := env.preds.insert dst .pred })
  | .barSync barrierId =>
      pure (.barrierCTA barrierId, env)

def lowerGInstr (gi : GInstr) : CLean.GInstr :=
  { guard? := gi.guard?, instr := lowerInstr gi.instr }

def lowerGInstrChecked? (env : Typing.TypeEnv) (gi : GInstr) : LowerM (CLean.GInstr × Typing.TypeEnv) := do
  ensureGuard? env gi.guard?
  let (instr, env') <- lowerInstrChecked? env gi.instr
  pure ({ guard? := gi.guard?, instr := instr }, env')

def lowerTerminator : Terminator → CLean.Terminator
  | .bra label => .br label
  | .cbra pred false tLabel fLabel => .cbr (.pred pred) tLabel fLabel
  | .cbra pred true tLabel fLabel => .cbr (.pred pred) fLabel tLabel
  | .exit => .terminate

def lowerTerminatorChecked? (env : Typing.TypeEnv) : Terminator → LowerM CLean.Terminator
  | .bra label => pure (.br label)
  | .cbra pred false tLabel fLabel => do
      match env.preds[pred]? with
      | some .pred => pure (.cbr (.pred pred) tLabel fLabel)
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred pred)
  | .cbra pred true tLabel fLabel => do
      match env.preds[pred]? with
      | some .pred => pure (.cbr (.pred pred) fLabel tLabel)
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred pred)
  | .exit => pure .terminate

def lowerBlock (block : Block) : CLean.Block :=
  { label := block.label
    body := block.body.map lowerGInstr
    term := lowerTerminator block.term }

def lowerGInstrsChecked? (env : Typing.TypeEnv) (body : Array GInstr) :
    LowerM (Array CLean.GInstr × Typing.TypeEnv) := do
  let mut out : Array CLean.GInstr := #[]
  let mut env := env
  for gi in body do
    let (gi', env') <- lowerGInstrChecked? env gi
    out := out.push gi'
    env := env'
  pure (out, env)

def lowerBlockChecked? (env : Typing.TypeEnv) (block : Block) : LowerM (CLean.Block × Typing.TypeEnv) := do
  let (body, env') <- lowerGInstrsChecked? env block.body
  let term <- lowerTerminatorChecked? env' block.term
  pure ({ label := block.label, body := body, term := term }, env')

def lowerBlocks (blocks : Array Block) : Std.HashMap BlockLabel CLean.Block :=
  blocks.foldl (fun out block => out.insert block.label (lowerBlock block)) {}

def lowerBlocksChecked? (blocks : Array Block) : LowerM (Std.HashMap BlockLabel CLean.Block) := do
  let mut out : Std.HashMap BlockLabel CLean.Block := {}
  for block in blocks do
    let (block', _) <- lowerBlockChecked? {} block
    out := out.insert block.label block'
  pure out

def lowerKernelEnv (kernel : Kernel) : KernelEnv :=
  { entry := kernel.entry
    gridCtx := kernel.gridCtx
    blocks := lowerBlocks kernel.blocks }

def lowerKernelEnvChecked? (kernel : Kernel) : LowerM KernelEnv := do
  let blocks <- lowerBlocksChecked? kernel.blocks
  pure { entry := kernel.entry, gridCtx := kernel.gridCtx, blocks := blocks }

def lowerKernelEnvCheckedD (kernel : Kernel) (default : KernelEnv := {}) : KernelEnv :=
  (lowerKernelEnvChecked? kernel).toOption.getD default

end PTX

end CLean
