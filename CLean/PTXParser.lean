import Std.Internal.Parsec.String
import CLean.PTXAst

namespace CLean
namespace PTX
namespace Parser

open Std.Internal.Parsec

abbrev Parser := Std.Internal.Parsec.String.Parser

/--
A small runtime parser-combinator frontend for the normalized PTX subset.
This intentionally uses Lean's parser-combinator infrastructure instead of ad hoc line splitting.
-/
def runParser (p : Parser α) (input : String) : Except String α :=
  Std.Internal.Parsec.String.Parser.run p input

def whitespace : Parser Unit :=
  Std.Internal.Parsec.String.ws

def lineComment : Parser Unit := do
  Std.Internal.Parsec.String.skipString "//"
  let _ ← many (satisfy fun c : Char => c != '\n' && c != '\r')
  pure ()

partial def trivia : Parser Unit := do
  whitespace
  let _ ← many (attempt (lineComment *> whitespace))
  pure ()

def lexeme (p : Parser α) : Parser α :=
  p <* trivia

def symbol (s : String) : Parser Unit :=
  lexeme (Std.Internal.Parsec.String.skipString s)

def rawSymbol (s : String) : Parser Unit :=
  Std.Internal.Parsec.String.skipString s

def optional? (p : Parser α) : Parser (Option α) :=
  (some <$> attempt p) <|> pure none

def charBetween (lo hi c : Char) : Bool :=
  decide (lo ≤ c ∧ c ≤ hi)

def identStart (c : Char) : Bool :=
  charBetween 'a' 'z' c || charBetween 'A' 'Z' c || c == '_' || c == '$'

def identRest (c : Char) : Bool :=
  identStart c || charBetween '0' '9' c || c == '.'

def ident : Parser String := lexeme do
  let _ ← optional? (Std.Internal.Parsec.String.skipChar '%')
  let head ← satisfy identStart
  let tail ← manyChars (satisfy identRest)
  pure (head.toString ++ tail)

def natLit : Parser Nat :=
  lexeme Std.Internal.Parsec.String.digits

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

def comma : Parser Unit :=
  symbol ","

def semi : Parser Unit :=
  symbol ";"

def colon : Parser Unit :=
  symbol ":"

def immediateValue? (ty : ScalarTy) (n : Nat) : Option Value :=
  match ty with
  | .pred => some (.pred (n != 0))
  | .u8 => some (.u8 (UInt8.ofNat n))
  | .u16 => some (.u16 (UInt16.ofNat n))
  | .u32 => some (.u32 (UInt32.ofNat n))
  | .u64 => some (.u64 (UInt64.ofNat n))
  | .s8 => some (.s8 n)
  | .s16 => some (.s16 n)
  | .s32 => some (.s32 n)
  | .s64 => some (.s64 n)
  | .b8 => some (.b8 (UInt8.ofNat n))
  | .b16 => some (.b16 (UInt16.ofNat n))
  | .b32 => some (.b32 (UInt32.ofNat n))
  | .b64 => some (.b64 (UInt64.ofNat n))
  | .f16 => some (.f16 (UInt16.ofNat n))
  | .bf16 => some (.bf16 (UInt16.ofNat n))
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

def operandOfIdent (name : String) : Operand :=
  match specialReg? name with
  | some s => .special s
  | none => .reg name

def operandWithTy (ty : ScalarTy) : Parser Operand :=
  attempt (do
    let n ← natLit
    match immediateValue? ty n with
    | some v => pure (.imm v)
    | none => fail s!"numeric literals for {repr ty} are not supported by this parser") <|>
  (operandOfIdent <$> ident)

def addrOperand : Parser Operand :=
  attempt ((fun n => Operand.imm (.u64 (UInt64.ofNat n))) <$> natLit) <|>
  (operandOfIdent <$> ident)

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

def instrAdd : Parser Instr := do
  rawSymbol "add"
  let ty ← scalarTySuffix
  trivia
  let dst ← ident
  comma
  let lhs ← operandWithTy ty
  comma
  let rhs ← operandWithTy ty
  pure (.add ty dst lhs rhs)

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
  rawSymbol "bar.sync"
  trivia
  let barrierId ← natLit
  pure (.barSync barrierId)

def instr : Parser Instr :=
  attempt instrSetp <|>
  attempt instrIsspacep <|>
  attempt instrCvta <|>
  attempt instrBarSync <|>
  attempt instrMov <|>
  attempt instrAdd <|>
  attempt instrLd <|>
  attempt instrSt

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
  symbol "exit"
  pure .exit

def terminator : Parser Terminator :=
  attempt terminatorCbra <|> attempt terminatorBra <|> attempt terminatorExit

inductive Stmt where
  | instr (i : GInstr)
  | term (t : Terminator)
  deriving Repr

def stmt : Parser Stmt := do
  let s ← attempt (Stmt.term <$> terminator) <|> (Stmt.instr <$> gInstr)
  semi
  pure s

def blockOfStmts (label : BlockLabel) (stmts : Array Stmt) : Block := Id.run do
  let mut body : Array GInstr := #[]
  let mut term : Terminator := .exit
  for st in stmts do
    match st with
    | .instr i => body := body.push i
    | .term t => term := t
  pure { label := label, body := body, term := term }

def block : Parser Block := do
  let label ← ident
  colon
  let stmts ← many (attempt stmt)
  pure (blockOfStmts label stmts)

def regDecl : Parser RegDecl := do
  symbol ".reg"
  let ty ← scalarTy
  let name ← ident
  semi
  pure { name := name, ty := ty }

def predDecl : Parser PredDecl := do
  symbol ".pred"
  let name ← ident
  semi
  pure { name := name }

def kernel : Parser Kernel := do
  trivia
  symbol ".entry"
  let entry ← ident
  semi
  let regs ← many (attempt regDecl)
  let preds ← many (attempt predDecl)
  let blocks ← many (attempt block)
  trivia
  eof
  pure { entry := entry, regs := regs, preds := preds, blocks := blocks }

def parseKernel (input : String) : Except String Kernel :=
  runParser kernel input

end Parser
end PTX
end CLean
