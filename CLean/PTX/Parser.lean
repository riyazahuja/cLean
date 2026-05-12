import Std.Internal.Parsec.String
import CLean.PTX.Ast

namespace CLean
namespace PTX
namespace Parser

open Std.Internal.Parsec

abbrev Parser := Std.Internal.Parsec.String.Parser

structure ParseError where
  offset : Nat
  line : Nat
  column : Nat
  message : String
  deriving Repr, Inhabited

def lineColumnFromOffset (input : String) (offset : Nat) : Nat × Nat :=
  let rec loop : List Char → Nat → Nat → Nat → Nat × Nat
    | [], _, line, column => (line, column)
    | c :: cs, idx, line, column =>
        if idx >= offset then
          (line, column)
        else if c = '\n' then
          loop cs (idx + 1) (line + 1) 1
        else
          loop cs (idx + 1) line (column + 1)
  loop input.toList 0 1 1

/-- Runtime parser-combinator entrypoint with source-position diagnostics. -/
def runParser (p : Parser α) (input : String) : Except ParseError α :=
  match p input.mkIterator with
  | .success _ res => .ok res
  | .error it err =>
      let offset := it.i.byteIdx
      let (line, column) := lineColumnFromOffset input offset
      .error { offset := offset, line := line, column := column, message := err }

def whitespace : Parser Unit :=
  Std.Internal.Parsec.String.ws

def lineComment : Parser Unit := do
  Std.Internal.Parsec.String.skipString "//"
  let _ ← many (satisfy fun c : Char => c != '\n' && c != '\r')
  pure ()

partial def blockComment : Parser Unit := do
  Std.Internal.Parsec.String.skipString "/*"
  let rec loop : Parser Unit := do
    match (← peek?) with
    | none => fail "unterminated block comment"
    | some _ =>
        (attempt (Std.Internal.Parsec.String.skipString "*/")) <|>
          (skip *> loop)
  loop

partial def trivia : Parser Unit := do
  whitespace
  let _ ← many (attempt ((lineComment <|> blockComment) *> whitespace))
  pure ()

def lexeme (p : Parser α) : Parser α :=
  p <* trivia

def symbol (s : String) : Parser Unit :=
  lexeme (Std.Internal.Parsec.String.skipString s)

def rawSymbol (s : String) : Parser Unit :=
  Std.Internal.Parsec.String.skipString s

def optional? (p : Parser α) : Parser (Option α) :=
  (some <$> attempt p) <|> pure none

def sepBy (p : Parser α) (sep : Parser Unit) : Parser (Array α) :=
  (do
    let first ← p
    let rest ← many (attempt (sep *> p))
    pure (rest.foldl (init := #[first]) (fun out x => out.push x))) <|> pure #[]

def charBetween (lo hi c : Char) : Bool :=
  decide (lo ≤ c ∧ c ≤ hi)

def identStart (c : Char) : Bool :=
  charBetween 'a' 'z' c || charBetween 'A' 'Z' c || c == '_' || c == '$'

def identRest (c : Char) : Bool :=
  identStart c || charBetween '0' '9' c || c == '.'

def tokenChar (c : Char) : Bool :=
  identRest c || c == '%'

structure NameRef where
  name : String
  hasPercent : Bool
  deriving Repr, Inhabited

def nameRef : Parser NameRef := lexeme do
  let hasPercent := (← optional? (Std.Internal.Parsec.String.skipChar '%')) |>.isSome
  let head ← satisfy identStart
  let tail ← manyChars (satisfy identRest)
  pure { name := head.toString ++ tail, hasPercent := hasPercent }

def ident : Parser String := do
  pure (← nameRef).name

def rawToken : Parser String :=
  lexeme (many1Chars (satisfy tokenChar))

def natFromDigits (digits : Array Char) : Nat :=
  digits.foldl (init := 0) fun acc c => acc * 10 + (c.toNat - '0'.toNat)

def hexDigitValue? (c : Char) : Option Nat :=
  if charBetween '0' '9' c then some (c.toNat - '0'.toNat)
  else if charBetween 'a' 'f' c then some (10 + c.toNat - 'a'.toNat)
  else if charBetween 'A' 'F' c then some (10 + c.toNat - 'A'.toNat)
  else none

def natFromHexDigits (digits : Array Char) : Nat :=
  digits.foldl (init := 0) fun acc c =>
    match hexDigitValue? c with
    | some d => acc * 16 + d
    | none => acc

def decNatRaw : Parser Nat := do
  let digits ← many1 (Std.Internal.Parsec.String.digit)
  pure (natFromDigits digits)

def hexNatRaw : Parser Nat := do
  (rawSymbol "0x" <|> rawSymbol "0X")
  let digits ← many1 (Std.Internal.Parsec.String.hexDigit)
  pure (natFromHexDigits digits)

def hexFloat32Raw : Parser Float := do
  rawSymbol "0f"
  let digits ← many1 (Std.Internal.Parsec.String.hexDigit)
  pure ((Float32.ofBits (UInt32.ofNat (natFromHexDigits digits))).toFloat)

def natRaw : Parser Nat :=
  attempt hexNatRaw <|> decNatRaw

def intLit : Parser Int := lexeme do
  let neg := (← optional? (Std.Internal.Parsec.String.skipChar '-')) |>.isSome
  let n ← natRaw
  if neg then pure (-(Int.ofNat n)) else pure (Int.ofNat n)

def natLit : Parser Nat := do
  let n ← intLit
  if n < 0 then fail "expected non-negative integer" else pure n.toNat

def scalarTySuffix : Parser ScalarTy :=
  attempt (rawSymbol ".pred" *> pure .pred) <|>
  attempt (rawSymbol ".bf16" *> pure .bf16) <|>
  attempt (rawSymbol ".u8" *> pure .u8) <|>
  attempt (rawSymbol ".u16" *> pure .u16) <|>
  attempt (rawSymbol ".u32" *> pure .u32) <|>
  attempt (rawSymbol ".u64" *> pure .u64) <|>
  attempt (rawSymbol ".s8" *> pure .s8) <|>
  attempt (rawSymbol ".s16" *> pure .s16) <|>
  attempt (rawSymbol ".s32" *> pure .s32) <|>
  attempt (rawSymbol ".s64" *> pure .s64) <|>
  attempt (rawSymbol ".b8" *> pure .b8) <|>
  attempt (rawSymbol ".b16" *> pure .b16) <|>
  attempt (rawSymbol ".b32" *> pure .b32) <|>
  attempt (rawSymbol ".b64" *> pure .b64) <|>
  attempt (rawSymbol ".f16" *> pure .f16) <|>
  attempt (rawSymbol ".f32" *> pure .f32) <|>
  attempt (rawSymbol ".f64" *> pure .f64)

def scalarTy : Parser ScalarTy :=
  lexeme scalarTySuffix

def addrSpaceSuffix : Parser AddrSpace :=
  attempt (rawSymbol ".global" *> pure .global) <|>
  attempt (rawSymbol ".shared" *> pure .shared) <|>
  attempt (rawSymbol ".local" *> pure .local) <|>
  attempt (rawSymbol ".param" *> pure .param) <|>
  attempt (rawSymbol ".const" *> pure .const) <|>
  attempt (rawSymbol ".generic" *> pure .generic)

def addrSpace : Parser AddrSpace :=
  lexeme addrSpaceSuffix

def cmpOp : Parser CmpOp :=
  attempt (rawSymbol "eq" *> pure .eq) <|>
  attempt (rawSymbol "ne" *> pure .ne) <|>
  attempt (rawSymbol "lt" *> pure .lt) <|>
  attempt (rawSymbol "le" *> pure .le) <|>
  attempt (rawSymbol "gt" *> pure .gt) <|>
  attempt (rawSymbol "ge" *> pure .ge)

def comma : Parser Unit := symbol ","
def semi : Parser Unit := symbol ";"
def optionalSemi : Parser Unit := do
  let _ ← optional? semi
  pure ()
def colon : Parser Unit := symbol ":"
def lbrace : Parser Unit := symbol "{"
def rbrace : Parser Unit := symbol "}"
def lparen : Parser Unit := symbol "("
def rparen : Parser Unit := symbol ")"

def immediateValue? (ty : ScalarTy) (n : Int) : Option Value :=
  match ty with
  | .pred => some (.pred (n != 0))
  | .u8 => if n < 0 then none else some (.u8 (UInt8.ofNat n.toNat))
  | .u16 => if n < 0 then none else some (.u16 (UInt16.ofNat n.toNat))
  | .u32 => if n < 0 then none else some (.u32 (UInt32.ofNat n.toNat))
  | .u64 => if n < 0 then none else some (.u64 (UInt64.ofNat n.toNat))
  | .s8 => some (.s8 n)
  | .s16 => some (.s16 n)
  | .s32 => some (.s32 n)
  | .s64 => some (.s64 n)
  | .b8 => if n < 0 then none else some (.b8 (UInt8.ofNat n.toNat))
  | .b16 => if n < 0 then none else some (.b16 (UInt16.ofNat n.toNat))
  | .b32 => if n < 0 then none else some (.b32 (UInt32.ofNat n.toNat))
  | .b64 => if n < 0 then none else some (.b64 (UInt64.ofNat n.toNat))
  | .f16 => if n < 0 then none else some (.f16 (UInt16.ofNat n.toNat))
  | .bf16 => if n < 0 then none else some (.bf16 (UInt16.ofNat n.toNat))
  | .f32 | .f64 => none

def specialReg? : String → Option SpecialReg
  | "tid.x" => some .tidX
  | "tid.y" => some .tidY
  | "tid.z" => some .tidZ
  | "ctaid.x" => some .ctaidX
  | "ctaid.y" => some .ctaidY
  | "ctaid.z" => some .ctaidZ
  | "ntid.x" => some .ntidX
  | "ntid.y" => some .ntidY
  | "ntid.z" => some .ntidZ
  | "nctaid.x" => some .nctaidX
  | "nctaid.y" => some .nctaidY
  | "nctaid.z" => some .nctaidZ
  | _ => none

def operandOfNameRef (ref : NameRef) : Operand :=
  match specialReg? ref.name with
  | some s => .special s
  | none => if ref.hasPercent then .reg ref.name else .symbol ref.name

def operandOfNameRefWithTy (ty : ScalarTy) (ref : NameRef) : Operand :=
  match ty with
  | .pred => if ref.hasPercent then .pred ref.name else .symbol ref.name
  | _ => operandOfNameRef ref

def operandWithTy (ty : ScalarTy) : Parser Operand :=
  attempt (do
    match ty with
    | .f32 =>
        let x ← lexeme hexFloat32Raw
        pure (.imm (.f32 x))
    | _ => fail "expected f32 hex literal") <|>
  attempt (do
    let n ← intLit
    match immediateValue? ty n with
    | some v => pure (.imm v)
    | none => fail s!"numeric literals for {repr ty} are not supported by this parser") <|>
  (operandOfNameRefWithTy ty <$> nameRef)

def addrAtom : Parser Operand :=
  attempt (do
    let n ← natLit
    pure (.imm (.u64 (UInt64.ofNat n)))) <|>
  (operandOfNameRef <$> nameRef)

def signedOffset : Parser Int :=
  attempt (symbol "+" *> intLit) <|>
  attempt (do let n ← symbol "-" *> natLit; pure (-(Int.ofNat n)))

def bracketAddrOperand : Parser Operand := do
  symbol "["
  let base ← addrAtom
  let off ← optional? signedOffset
  symbol "]"
  pure (.addr base (off.getD 0))

def addrOperand : Parser Operand :=
  attempt bracketAddrOperand <|> addrAtom

def guard : Parser Guard := do
  symbol "@"
  let negate := (← optional? (symbol "!")) |>.isSome
  let p ← ident
  pure { pred := p, negate := negate }

def instrMov : Parser Instr := do
  rawSymbol "mov"
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let src ← operandWithTy ty
  pure (.mov ty dst src)

def unaryOpcode : Parser ScalarUnaryOp :=
  attempt (rawSymbol "neg" *> pure .neg) <|>
  attempt (rawSymbol "abs" *> pure .abs) <|>
  attempt (rawSymbol "not" *> pure .bitnot)

def instrUnop : Parser Instr := do
  let op ← unaryOpcode
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let src ← operandWithTy ty
  pure (.unop op ty dst src)

def instrCvt : Parser Instr := do
  rawSymbol "cvt"
  let dstTy ← scalarTySuffix
  let srcTy ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let src ← operandWithTy srcTy
  pure (.unop (.cvt dstTy) srcTy dst src)

def instrMadLo : Parser Instr := do
  rawSymbol "mad.lo"
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let a ← operandWithTy ty
  comma
  let b ← operandWithTy ty
  comma
  let c ← operandWithTy ty
  pure (.triop .mad ty dst a b c)

def instrFmaRn : Parser Instr := do
  rawSymbol "fma.rn"
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let a ← operandWithTy ty
  comma
  let b ← operandWithTy ty
  comma
  let c ← operandWithTy ty
  pure (.triop .fma ty dst a b c)

def scalarBinaryOp : Parser ScalarBinaryOp :=
  attempt (rawSymbol "add" *> pure .add) <|>
  attempt (rawSymbol "sub" *> pure .sub) <|>
  attempt (rawSymbol "mul" *> pure .mul) <|>
  attempt (rawSymbol "and" *> pure .bitand) <|>
  attempt (rawSymbol "or" *> pure .bitor) <|>
  attempt (rawSymbol "xor" *> pure .bitxor) <|>
  attempt (rawSymbol "shl" *> pure .shl) <|>
  attempt (rawSymbol "shr" *> pure .shr) <|>
  attempt (rawSymbol "min" *> pure .min) <|>
  attempt (rawSymbol "max" *> pure .max)

def instrMulLo : Parser Instr := do
  rawSymbol "mul.lo"
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy ty
  comma
  let rhs ← operandWithTy ty
  pure (.binop .mul ty dst lhs rhs)

def instrMulWideS32 : Parser Instr := do
  rawSymbol "mul.wide.s32"
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy .s32
  comma
  let rhs ← operandWithTy .s32
  pure (.binop .mulWideS32 .s32 dst lhs rhs)

def predBinaryOp : Parser ScalarBinaryOp :=
  attempt (rawSymbol "or.pred" *> pure .bitor) <|>
  attempt (rawSymbol "and.pred" *> pure .bitand) <|>
  attempt (rawSymbol "xor.pred" *> pure .bitxor)

def instrPredBinop : Parser Instr := do
  let op ← predBinaryOp
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy .pred
  comma
  let rhs ← operandWithTy .pred
  pure (.predBinop op dst lhs rhs)

def instrBinop : Parser Instr := do
  let op ← scalarBinaryOp
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy ty
  comma
  let rhs ← operandWithTy ty
  pure (.binop op ty dst lhs rhs)

def instrSetp : Parser Instr := do
  rawSymbol "setp."
  let op ← cmpOp
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy ty
  comma
  let rhs ← operandWithTy ty
  pure (.setp op ty dst lhs rhs)

def instrLd : Parser Instr := do
  rawSymbol "ld"
  let space ← addrSpaceSuffix
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let addr ← addrOperand
  pure (.ld space ty dst addr)

def instrSt : Parser Instr := do
  rawSymbol "st"
  let space ← addrSpaceSuffix
  let ty ← scalarTySuffix
  trivia
  let addr ← addrOperand
  comma
  let value ← operandWithTy ty
  pure (.st space ty addr value)

def instrCvta : Parser Instr := do
  rawSymbol "cvta"
  let _ ← optional? (rawSymbol ".to")
  let space ← addrSpaceSuffix
  let _ ← optional? scalarTySuffix
  trivia
  let dst ← ident
  comma
  let src ← addrOperand
  pure (.cvta space dst src)

def instrIsspacep : Parser Instr := do
  rawSymbol "isspacep"
  let space ← addrSpaceSuffix
  trivia
  let dst ← ident
  comma
  let src ← addrOperand
  pure (.isspacep space dst src)

def instrBarSync : Parser Instr := do
  (rawSymbol "bar.sync" <|> rawSymbol "barrier.sync")
  trivia
  let barrierId ← natLit
  pure (.barSync barrierId)

def unsupportedOperand : Parser Operand :=
  attempt addrOperand <|> attempt (operandWithTy .u64) <|> (operandOfNameRef <$> nameRef)

def opcodeParts (tok : String) : String × Array String :=
  match tok.splitOn "." with
  | [] => (tok, #[])
  | op :: mods => (op, mods.toArray)

def instrUnsupported : Parser Instr := do
  let tok ← rawToken
  let (opcode, modifiers) := opcodeParts tok
  let operands ← sepBy unsupportedOperand comma
  pure (.unsupported opcode modifiers operands)

def instr : Parser Instr :=
  attempt instrSetp <|>
  attempt instrIsspacep <|>
  attempt instrCvta <|>
  attempt instrBarSync <|>
  attempt instrCvt <|>
  attempt instrMadLo <|>
  attempt instrFmaRn <|>
  attempt instrMulWideS32 <|>
  attempt instrMulLo <|>
  attempt instrPredBinop <|>
  attempt instrMov <|>
  attempt instrUnop <|>
  attempt instrBinop <|>
  attempt instrLd <|>
  attempt instrSt <|>
  instrUnsupported

def gInstr : Parser GInstr := do
  let g? ← optional? guard
  let i ← instr
  pure { guard? := g?, instr := i }

def terminatorBra : Parser Terminator := do
  symbol "bra"
  let label ← ident
  pure (.bra label)

def terminatorCbra : Parser Terminator := do
  symbol "cbra"
  let p ← ident
  comma
  let tLabel ← ident
  comma
  let fLabel ← ident
  pure (.cbra p false tLabel fLabel)

def terminatorExit : Parser Terminator := do
  (symbol "exit" <|> symbol "ret")
  pure .exit

def terminator : Parser Terminator :=
  attempt terminatorCbra <|> attempt terminatorBra <|> attempt terminatorExit

inductive Stmt where
  | instr (i : GInstr)
  | term (t : Terminator)
  | guardedBra (g : Guard) (label : BlockLabel)
  | noop
  deriving Repr

def implicitFallthroughLabel : BlockLabel :=
  "__ptx_implicit_fallthrough__"

def stmt : Parser Stmt := do
  attempt (do
    symbol ".pragma"
    symbol "\""
    let _ ← manyChars (satisfy fun c : Char => c != '"')
    symbol "\""
    semi
    pure .noop) <|>
  attempt (do
    let g ← guard
    symbol "bra"
    let label ← ident
    semi
    pure (.guardedBra g label)) <|>
  (do
    let s ← attempt (Stmt.term <$> terminator) <|> (Stmt.instr <$> gInstr)
    semi
    pure s)

def blockOfStmts (label : BlockLabel) (stmts : Array Stmt) : Block := Id.run do
  let mut body : Array GInstr := #[]
  let mut term? : Option Terminator := none
  for st in stmts do
    match st with
    | .instr i => body := body.push i
    | .term t => term? := some t
    | .guardedBra g target => term? := some (.cbra g.pred g.negate target (label ++ "$fallthrough"))
    | .noop => pure ()
  pure { label := label, body := body, term := term?.getD (.bra implicitFallthroughLabel) }

def fallthroughLabel (label : BlockLabel) (idx : Nat) : BlockLabel :=
  label ++ "$fallthrough" ++ toString idx

partial def blocksOfStmts (label : BlockLabel) (stmts : Array Stmt) : Array Block := Id.run do
  let rec go (label : BlockLabel) (idx : Nat) (body : Array GInstr) : List Stmt → Array Block
    | [] => #[{ label := label, body := body, term := .bra implicitFallthroughLabel }]
    | .instr instr :: rest => go label idx (body.push instr) rest
    | .term term :: _ => #[{ label := label, body := body, term := term }]
    | .noop :: rest => go label idx body rest
    | .guardedBra g target :: rest =>
        let cont := fallthroughLabel label idx
        let block : Block := {
          label := label
          body := body
          term := .cbra g.pred g.negate target cont
        }
        if rest.isEmpty then
          #[block, { label := cont, body := #[], term := .bra implicitFallthroughLabel }]
        else
          #[block] ++ go cont (idx + 1) #[] rest
  go label 0 #[] stmts.toList

def linkImplicitFallthroughs (blocks : Array Block) : Array Block := Id.run do
  let mut out : Array Block := #[]
  for h : i in [:blocks.size] do
    let block := blocks[i]
    let term :=
      match block.term with
      | .bra label =>
          if label == implicitFallthroughLabel then
            match blocks[i + 1]? with
            | some next => .bra next.label
            | none => .exit
          else
            block.term
      | _ => block.term
    out := out.push { block with term := term }
  pure out

def block : Parser (Array Block) := do
  let label ← ident
  colon
  let stmts ← many (attempt stmt)
  pure (blocksOfStmts label stmts)

structure NameDecl where
  name : String
  count? : Option Nat := none
  deriving Repr, Inhabited

def nameDecl : Parser NameDecl := do
  let name ← ident
  let count? ← optional? (symbol "<" *> natLit <* symbol ">")
  pure { name := name, count? := count? }

def expandNames (decl : NameDecl) : Array String :=
  match decl.count? with
  | none => #[decl.name]
  | some n => (Array.range n).map fun i => decl.name ++ toString i

def regDecl : Parser (Array RegDecl) := do
  symbol ".reg"
  let ty ← scalarTy
  let name ← nameDecl
  semi
  pure ((expandNames name).map fun n => { name := n, ty := ty })

def regPredDecl : Parser (Array PredDecl) := do
  symbol ".reg"
  let ty ← scalarTy
  if ty == .pred then
    let name ← nameDecl
    semi
    pure ((expandNames name).map fun n => { name := n })
  else
    fail "expected .reg .pred"

def predDecl : Parser (Array PredDecl) := do
  symbol ".pred"
  let name ← nameDecl
  semi
  pure ((expandNames name).map fun n => { name := n })

def ptrAttr : Parser (Option AddrSpace × Nat) := do
  rawSymbol ".ptr"
  let space? ← optional? addrSpaceSuffix
  trivia
  let align? ← optional? (symbol ".align" *> natLit)
  pure (space?, align?.getD 1)

def paramDeclCore : Parser ParamDecl := do
  symbol ".param"
  let ptr? ← optional? ptrAttr
  let ty ← scalarTy
  let name ← ident
  match ptr? with
  | some (space?, align) =>
      pure { name := name, ty := ty, isPtr := true, ptrSpace? := space?, align := align }
  | none =>
      pure { name := name, ty := ty }

def paramDecl : Parser ParamDecl := do
  let decl ← paramDeclCore
  semi
  pure decl

def sharedDeclCore : Parser SharedDecl := do
  symbol ".shared"
  let align? ← optional? (symbol ".align" *> natLit)
  let ty ← scalarTy
  let name ← ident
  let count? ← optional? (symbol "[" *> natLit <* symbol "]")
  pure { name := name, ty := ty, count := count?.getD 1, align := align?.getD 1 }

def sharedDecl : Parser SharedDecl := do
  let decl ← sharedDeclCore
  semi
  pure decl

def moduleMemoryDecl (space : ModuleMemorySpace) : Parser ModuleMemoryDecl := do
  let align? ← optional? (symbol ".align" *> natLit)
  let ty ← scalarTy
  let name ← ident
  let count? ← optional? (symbol "[" *> natLit <* symbol "]")
  semi
  pure { space := space, name := name, ty := ty, count := count?.getD 1, align := align?.getD 1 }

inductive Decl where
  | regs (decls : Array RegDecl)
  | preds (decls : Array PredDecl)
  | param (decl : ParamDecl)
  | shared (decl : SharedDecl)
  deriving Repr

def decl : Parser Decl :=
  attempt (Decl.shared <$> sharedDecl) <|>
  attempt (Decl.param <$> paramDecl) <|>
  attempt (Decl.preds <$> regPredDecl) <|>
  attempt (Decl.preds <$> predDecl) <|>
  attempt (Decl.regs <$> regDecl)

def splitDecls (decls : Array Decl) : Array RegDecl × Array PredDecl × Array ParamDecl × Array SharedDecl := Id.run do
  let mut regs : Array RegDecl := #[]
  let mut preds : Array PredDecl := #[]
  let mut params : Array ParamDecl := #[]
  let mut shareds : Array SharedDecl := #[]
  for decl in decls do
    match decl with
    | .regs ds => regs := regs ++ ds
    | .preds ds => preds := preds ++ ds
    | .param d => params := params.push d
    | .shared d => shareds := shareds.push d
  pure (regs, preds, params, shareds)

def withLeadingBlocks (entry : BlockLabel) (leading : Array Stmt) (blocks : Array Block) : Array Block :=
  if leading.isEmpty then blocks else blocksOfStmts entry leading ++ blocks

def entryParamList : Parser (Array ParamDecl) :=
  lparen *> sepBy paramDeclCore comma <* rparen

def kernelBody (entry : BlockLabel) (entryParams : Array ParamDecl) : Parser Kernel := do
  let decls ← many (attempt decl)
  let leading ← many (attempt stmt)
  let blockGroups ← many (attempt block)
  let blocks := blockGroups.foldl (init := #[]) (fun out bs => out ++ bs)
  let (regs, preds, params, shareds) := splitDecls decls
  pure {
    entry := entry
    regs := regs
    preds := preds
    params := entryParams ++ params
    shareds := shareds
    blocks := linkImplicitFallthroughs (withLeadingBlocks entry leading blocks)
  }

def kernel : Parser Kernel := do
  let _ ← optional? (symbol ".visible")
  symbol ".entry"
  let entry ← ident
  let entryParams? ← optional? entryParamList
  let entryParams := entryParams?.getD #[]
  (attempt (do
      lbrace
      let k ← kernelBody entry entryParams
      rbrace
      pure k)) <|>
    (do
      semi
      kernelBody entry entryParams)

def versionDirective : Parser ModuleDirective := do
  symbol ".version"
  let value ← rawToken
  optionalSemi
  pure (.version value)

def targetDirective : Parser ModuleDirective := do
  symbol ".target"
  let first ← rawToken
  let rest ← many (attempt (comma *> rawToken))
  optionalSemi
  pure (.target (rest.foldl (init := #[first]) (fun out x => out.push x)))

def addressSizeDirective : Parser ModuleDirective := do
  symbol ".address_size"
  let bits ← natLit
  optionalSemi
  pure (.addressSize bits)

def directive : Parser ModuleDirective :=
  attempt versionDirective <|> attempt targetDirective <|> addressSizeDirective

inductive ModuleItem where
  | directive (d : ModuleDirective)
  | memory (d : ModuleMemoryDecl)
  | shared (d : SharedDecl)
  | kernel (k : Kernel)
  deriving Repr

def moduleItem : Parser ModuleItem :=
  attempt (ModuleItem.directive <$> directive) <|>
  attempt (ModuleItem.memory <$> (symbol ".global" *> moduleMemoryDecl .global)) <|>
  attempt (ModuleItem.memory <$> (symbol ".const" *> moduleMemoryDecl .const)) <|>
  attempt (ModuleItem.shared <$> sharedDecl) <|>
  (ModuleItem.kernel <$> kernel)

def moduleOfItems (items : Array ModuleItem) : Module := Id.run do
  let mut directives : Array ModuleDirective := #[]
  let mut memories : Array ModuleMemoryDecl := #[]
  let mut shareds : Array SharedDecl := #[]
  let mut kernels : Array Kernel := #[]
  for item in items do
    match item with
    | .directive d => directives := directives.push d
    | .memory d => memories := memories.push d
    | .shared d => shareds := shareds.push d
    | .kernel k => kernels := kernels.push { k with shareds := shareds ++ k.shareds }
  pure { directives := directives, memories := memories, shareds := shareds, kernels := kernels }

def moduleParser : Parser Module := do
  trivia
  let items ← many (attempt moduleItem)
  trivia
  eof
  pure (moduleOfItems items)

def parseModule (input : String) : Except ParseError Module :=
  runParser moduleParser input

def parseKernel (input : String) : Except ParseError Kernel := do
  let m ← parseModule input
  match m.kernels[0]? with
  | some k => pure k
  | none => .error { offset := 0, line := 1, column := 1, message := "expected one .entry kernel" }

end Parser
end PTX
end CLean
