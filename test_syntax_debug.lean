import Lean

open Lean Parser

-- Test different expressions to see their syntax kinds
def testExpr1 := "(5 + 3).toNat?.getD 0"
def testExpr2 := "x && y"
def testExpr3 := "row < N && col < N"

#check Term.app
#check Term.proj

-- Simple macro to print syntax structure
macro "show_syntax " t:term : term => do
  Lean.Macro.throwError s!"Syntax kind: {t.raw.getKind}, numArgs: {t.raw.getNumArgs}"

-- This will show us the structure
-- #check show_syntax (5 + 3).toNat?.getD 0
