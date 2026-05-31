import CLean.Core.State
import CLean.Core.Typing
import CLean.PTX.Ast

namespace CLean

namespace PTX

inductive LowerError where
  | unknownReg (r : RegName)
  | unknownPred (p : PredName)
  | unknownParam (p : String)
  | unknownShared (s : String)
  | unknownSymbol (s : String)
  | unknownBlock (label : BlockLabel)
  | valueHasNoScalarType (v : Value)
  | typeMismatch (expected actual : ScalarTy)
  | unsupportedType (ty : ScalarTy)
  | unsupportedOp (op : String)
  | unsupportedModifier (modifier : String)
  | unsupportedTerminator
  deriving Repr

abbrev LowerM := Except LowerError

def lowerParamOperand (info : ParamInfo) : RValue :=
  .imm (.u64 (UInt64.ofNat info.offset))

def lowerSharedOperand (info : CLean.SharedDecl) : RValue :=
  .imm (.u64 (UInt64.ofNat info.offset))

def addressOffsetValue? (ty : ScalarTy) (offset : Nat) : Option Value :=
  match ty with
  | .u32 => some (.u32 (UInt32.ofNat offset))
  | .u64 => some (.u64 (UInt64.ofNat offset))
  | .s64 => some (.s64 (Int.ofNat offset))
  | .b64 => some (.b64 (UInt64.ofNat offset))
  | _ => none

def lowerOperandChecked? (env : Typing.TypeEnv) : Operand → LowerM RValue
  | .reg r =>
      match env.regs[r]? with
      | some _ => pure (.reg r)
      | none => throw (.unknownReg r)
  | .pred p =>
      match env.preds[p]? with
      | some .pred => pure (.pred p)
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred p)
  | .symbol s => throw (.unknownSymbol s)
  | .addr _ _ => throw (.unsupportedOp "address operand outside memory")
  | .imm v => pure (.imm v)
  | .special s => pure (.special s)

def operandType? (env : Typing.TypeEnv) : Operand → LowerM ScalarTy
  | .reg r =>
      match env.regs[r]? with
      | some regTy =>
          match Typing.RegTy.asScalar? regTy with
          | some ty => pure ty
          | none => throw (.unsupportedOp "non-scalar register")
      | none => throw (.unknownReg r)
  | .pred p =>
      match env.preds[p]? with
      | some .pred => pure .pred
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred p)
  | .symbol s => throw (.unknownSymbol s)
  | .addr _ _ => throw (.unsupportedOp "address operand outside memory")
  | .imm v =>
      match Typing.valueType? v with
      | some ty => pure ty
      | none => throw (.valueHasNoScalarType v)
  | .special s =>
      match Typing.specialType? s with
      | some ty => pure ty
      | none => throw (.unsupportedOp "special")

def addressOperandType? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM ScalarTy
  | .addr base _ => addressOperandType? env space base
  | .symbol s =>
      match space with
      | .param =>
          match env.params[s]? with
          | some _ => pure .u64
          | none => throw (.unknownParam s)
      | .shared =>
          match env.shareds[s]? with
          | some _ => pure .u64
          | none => throw (.unknownShared s)
      | _ => throw (.unknownSymbol s)
  | operand => operandType? env operand

def lowerAddressOperandChecked? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM RValue
  | .addr base offset => do
      let baseTy <- addressOperandType? env space base
      let baseRv <- lowerAddressOperandChecked? env space base
      if offset = 0 then
        pure baseRv
      else if offset > 0 then
        let off <- match addressOffsetValue? baseTy offset.toNat with
          | some v => pure v
          | none => throw (.unsupportedType baseTy)
        pure (.binop .add baseRv (.imm off))
      else
        let magnitude := (-offset).toNat
        let off <- match addressOffsetValue? baseTy magnitude with
          | some v => pure v
          | none => throw (.unsupportedType baseTy)
        pure (.binop .sub baseRv (.imm off))
  | .symbol s =>
      match space with
      | .param =>
          match env.params[s]? with
          | some info => pure (lowerParamOperand info)
          | none => throw (.unknownParam s)
      | .shared =>
          match env.shareds[s]? with
          | some info => pure (lowerSharedOperand info)
          | none => throw (.unknownShared s)
      | _ => throw (.unknownSymbol s)
  | operand => lowerOperandChecked? env operand

def expectOperandType (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM Unit := do
  let actual <- operandType? env operand
  if Typing.scalarCompatible? expected actual then
    pure ()
  else
    throw (.typeMismatch expected actual)

def coerceOperandTo? (expected actual : ScalarTy) (rv : RValue) : RValue :=
  if expected == actual then
    rv
  else
    match expected, actual with
    | .b32, .u32 | .b32, .s32 => .unop (.cvt .b32) rv
    | .b64, .u64 | .b64, .s64 => .unop (.cvt .b64) rv
    | .s32, .u32 | .s32, .b32 => .unop (.cvt .s32) rv
    | .u32, .s32 | .u32, .b32 => .unop (.cvt .u32) rv
    -- Keep 64-bit address-shaped values unwrapped: `cvta` registers carry generic
    -- address values at runtime even when the PTX declaration says `.b64`/`.u64`.
    | .s64, .u64 | .s64, .b64 | .u64, .s64 | .u64, .b64 => rv
    | _, _ => rv

def lowerOperandCheckedAs? (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) :
    LowerM RValue := do
  let actual <- operandType? env operand
  if Typing.scalarCompatible? expected actual then
    let rv <- lowerOperandChecked? env operand
    pure (coerceOperandTo? expected actual rv)
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

def lowerInstrChecked? (env : Typing.TypeEnv) : Instr → LowerM (CLean.Instr × Typing.TypeEnv)
  | .mov ty dst src => do
      expectOperandType env ty src
      let src <- lowerOperandCheckedAs? env ty src
      pure (.assignReg dst src, { env with regs := env.regs.insert dst (.scalar ty) })
  | .unop op srcTy dst src => do
      expectOperandType env srcTy src
      let src <- lowerOperandCheckedAs? env srcTy src
      match Typing.unarySig? op srcTy with
      | some outTy =>
          pure (.assignReg dst (.unop op src),
            { env with regs := env.regs.insert dst (.scalar outTy) })
      | none => throw (.unsupportedOp "unop")
  | .binop op ty dst lhs rhs => do
      expectOperandType env ty lhs
      expectOperandType env ty rhs
      let lhs <- lowerOperandCheckedAs? env ty lhs
      let rhs <- lowerOperandCheckedAs? env ty rhs
      match Typing.binarySig? op ty ty with
      | some outTy =>
          pure (.assignReg dst (.binop op lhs rhs),
            { env with regs := env.regs.insert dst (.scalar outTy) })
      | none => throw (.unsupportedOp "binop")
  | .predBinop op dst lhs rhs => do
      expectOperandType env .pred lhs
      expectOperandType env .pred rhs
      let lhs <- lowerOperandChecked? env lhs
      let rhs <- lowerOperandChecked? env rhs
      match Typing.binarySig? op .pred .pred with
      | some .pred =>
          pure (.assignPredValue dst (.binop op lhs rhs),
            { env with preds := env.preds.insert dst .pred })
      | _ => throw (.unsupportedOp "predBinop")
  | .triop op ty dst a b c => do
      expectOperandType env ty a
      expectOperandType env ty b
      expectOperandType env ty c
      let a <- lowerOperandCheckedAs? env ty a
      let b <- lowerOperandCheckedAs? env ty b
      let c <- lowerOperandCheckedAs? env ty c
      match Typing.ternarySig? op ty ty ty with
      | some outTy =>
          pure (.assignReg dst (.triop op a b c),
            { env with regs := env.regs.insert dst (.scalar outTy) })
      | none => throw (.unsupportedOp "triop")
  | .setp op ty dst lhs rhs => do
      expectOperandType env ty lhs
      expectOperandType env ty rhs
      let lhs <- lowerOperandCheckedAs? env ty lhs
      let rhs <- lowerOperandCheckedAs? env ty rhs
      match Typing.cmpSig? op ty ty with
      | some .pred =>
          pure (.assignPred dst { op := op, lhs := lhs, rhs := rhs },
            { env with preds := env.preds.insert dst .pred })
      | _ => throw (.unsupportedOp "setp")
  | .ld space ty dst addr => do
      ensureCodecType ty
      let addrTy <- addressOperandType? env space addr
      match addrTy with
      | .u32 | .u64 | .s64 | .b64 =>
          let addr <- lowerAddressOperandChecked? env space addr
          pure (.load dst { space := space, ty := ty, addr := addr },
            { env with regs := env.regs.insert dst (.scalar ty) })
      | _ => throw (.unsupportedType addrTy)
  | .st space ty addr value => do
      ensureCodecType ty
      let addrTy <- addressOperandType? env space addr
      match addrTy with
      | .u32 | .u64 | .s64 | .b64 =>
          expectOperandType env ty value
          let addr <- lowerAddressOperandChecked? env space addr
          let value <- lowerOperandCheckedAs? env ty value
          pure (.store { space := space, ty := ty, addr := addr } value, env)
      | _ => throw (.unsupportedType addrTy)
  | .cvta space dst src => do
      let srcTy <- addressOperandType? env space src
      ensureCvtaSourceType srcTy
      let src <- lowerAddressOperandChecked? env space src
      pure (.cvta dst space src, { env with regs := env.regs.insert dst (.ptr (some space)) })
  | .isspacep space dst src => do
      let srcTy <- operandType? env src
      -- In the initial checked layer, generic addresses have no ScalarTy, so this accepts integer
      -- pointer-shaped operands and relies on the semantic helper to reject non-generic runtime values.
      ensureCvtaSourceType srcTy
      let src <- lowerOperandChecked? env src
      pure (.isspacep dst space src, { env with preds := env.preds.insert dst .pred })
  | .barSync barrierId =>
      pure (.barrierCTA barrierId, env)
  | .unsupported opcode modifiers _ =>
      match modifiers[0]? with
      | some modifier => throw (.unsupportedModifier modifier)
      | none => throw (.unsupportedOp opcode)

def lowerGInstrChecked? (env : Typing.TypeEnv) (gi : GInstr) : LowerM (CLean.GInstr × Typing.TypeEnv) := do
  ensureGuard? env gi.guard?
  let (instr, env') <- lowerInstrChecked? env gi.instr
  pure ({ guard? := gi.guard?, instr := instr }, env')

def blockLabels (blocks : Array Block) : Std.HashMap BlockLabel Unit :=
  blocks.foldl (fun out block => out.insert block.label ()) {}

def requireBlockLabel (labels : Std.HashMap BlockLabel Unit) (label : BlockLabel) : LowerM Unit :=
  match labels[label]? with
  | some () => pure ()
  | none => throw (.unknownBlock label)

def lowerTerminatorChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) :
    Terminator → LowerM CLean.Terminator
  | .bra label => do
      requireBlockLabel labels label
      pure (.br label)
  | .cbra pred false tLabel fLabel => do
      requireBlockLabel labels tLabel
      requireBlockLabel labels fLabel
      match env.preds[pred]? with
      | some .pred => pure (.cbr (.pred pred) tLabel fLabel)
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred pred)
  | .cbra pred true tLabel fLabel => do
      requireBlockLabel labels tLabel
      requireBlockLabel labels fLabel
      match env.preds[pred]? with
      | some .pred => pure (.cbr (.pred pred) fLabel tLabel)
      | some ty => throw (.typeMismatch .pred ty)
      | none => throw (.unknownPred pred)
  | .exit => pure .terminate

def lowerGInstrsChecked? (env : Typing.TypeEnv) (body : Array GInstr) :
    LowerM (Array CLean.GInstr × Typing.TypeEnv) := do
  let mut out : Array CLean.GInstr := #[]
  let mut env := env
  for gi in body do
    let (gi', env') <- lowerGInstrChecked? env gi
    out := out.push gi'
    env := env'
  pure (out, env)

def lowerBlockChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv)
    (block : Block) : LowerM (CLean.Block × Typing.TypeEnv) := do
  let (body, env') <- lowerGInstrsChecked? env block.body
  let term <- lowerTerminatorChecked? labels env' block.term
  pure ({ label := block.label, body := body, term := term }, env')

def alignUp (offset align : Nat) : Nat :=
  if align = 0 then
    offset
  else
    ((offset + align - 1) / align) * align

def lowerParamsChecked? (params : Array ParamDecl) : LowerM (Array ParamInfo) := do
  let mut out : Array ParamInfo := #[]
  let mut offset : Nat := 0
  for param in params do
    let size <- match Typing.byteWidth? param.ty with
      | some size => pure size
      | none => throw (.unsupportedType param.ty)
    let storageAlign := (Typing.alignment? param.ty).getD 1
    let offset' := alignUp offset storageAlign
    out := out.push {
      name := param.name
      ty := param.ty
      isPtr := param.isPtr
      ptrSpace? := param.ptrSpace?
      align := param.align
      offset := offset'
      size := size
    }
    offset := offset' + size
  pure out

def paramInfoMap (params : Array ParamInfo) : Std.HashMap String ParamInfo :=
  params.foldl (fun out info => out.insert info.name info) {}

def lowerSharedsChecked? (shareds : Array SharedDecl) : LowerM (Array CLean.SharedDecl) := do
  let mut out : Array CLean.SharedDecl := #[]
  let mut offset : Nat := 0
  for shared in shareds do
    let elemSize <- match Typing.byteWidth? shared.ty with
      | some size => pure size
      | none => throw (.unsupportedType shared.ty)
    let storageAlign := if shared.align = 0 then (Typing.alignment? shared.ty).getD 1 else shared.align
    let offset' := alignUp offset storageAlign
    let count := shared.count
    let size := elemSize * count
    out := out.push {
      name := shared.name
      size := size
      align := storageAlign
      offset := offset'
    }
    offset := offset' + size
  pure out

def sharedInfoMap (shareds : Array CLean.SharedDecl) : Std.HashMap String CLean.SharedDecl :=
  shareds.foldl (fun out info => out.insert info.name info) {}

def initialTypeEnv (kernel : Kernel) (params : Array ParamInfo := #[])
    (shareds : Array CLean.SharedDecl := #[]) : Typing.TypeEnv :=
  let env : Typing.TypeEnv := { params := paramInfoMap params, shareds := sharedInfoMap shareds }
  let env := kernel.regs.foldl
    (fun env decl => { env with regs := env.regs.insert decl.name (.scalar decl.ty) })
    env
  kernel.preds.foldl (fun env decl => { env with preds := env.preds.insert decl.name .pred }) env

def lowerBlocksChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (blocks : Array Block) :
    LowerM (Std.HashMap BlockLabel CLean.Block) := do
  let mut out : Std.HashMap BlockLabel CLean.Block := {}
  for block in blocks do
    let (block', _) <- lowerBlockChecked? labels env block
    out := out.insert block.label block'
  pure out

def lowerKernelEnvChecked? (kernel : Kernel) : LowerM KernelEnv := do
  let labels := blockLabels kernel.blocks
  requireBlockLabel labels kernel.entry
  let params <- lowerParamsChecked? kernel.params
  let shareds <- lowerSharedsChecked? kernel.shareds
  let blocks <- lowerBlocksChecked? labels (initialTypeEnv kernel params shareds) kernel.blocks
  pure {
    entry := kernel.entry
    gridCtx := kernel.gridCtx
    addrLayout := {}
    params := params
    sharedDecls := shareds
    blocks := blocks
  }

end PTX

end CLean
