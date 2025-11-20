import Lean
import CLean.GPU

open Lean Lean.Elab Lean.Elab.Command
open GpuDSL

-- Debug macro to print syntax structure
macro "debug_syntax " val:declVal : command => do
  let body ← match val with
    | `(declVal| := $body:term) => pure body
    | _ => Macro.throwError "no body"

  if body.raw.getKind != ``Lean.Parser.Term.do then
    Macro.throwError "not do notation"

  let doSeq := body.raw[1]
  let items := if doSeq.getArgs.size >= 1 then doSeq[0].getArgs else #[]

  -- Print info about first few items
  for h : i in [:items.size.min 5] do
    let item := items[i]
    if item.getKind == ``Lean.Parser.Term.doSeqItem then
      let doElem := item[0]
      dbg_trace s!"Item {i}: kind = {doElem.getKind}, args = {doElem.getNumArgs}"
      if doElem.getKind == ``Lean.Parser.Term.doLet then
        for j in [:doElem.getNumArgs] do
          dbg_trace s!"  doElem[{j}] kind = {doElem[j].getKind}, args = {doElem[j].getNumArgs}"
        if doElem.getNumArgs >= 3 then
          let letDeclGroup := doElem[2]  -- This is the letDecl group
          dbg_trace s!"  letDeclGroup kind = {letDeclGroup.getKind}, args = {letDeclGroup.getNumArgs}"
          if letDeclGroup.getNumArgs >= 1 then
            let letDecl := letDeclGroup[0]
            dbg_trace s!"  letDecl kind = {letDecl.getKind}, args = {letDecl.getNumArgs}"
            if letDecl.getKind == ``Lean.Parser.Term.letIdDecl && letDecl.getNumArgs >= 5 then
              let id := letDecl[0]
              dbg_trace s!"    id = {id.getId}"
              -- Check letDecl[2] which should contain the typeSpec
              if letDecl[2].getNumArgs >= 1 then
                let typeSpec := letDecl[2][0]
                dbg_trace s!"    typeSpec kind = {typeSpec.getKind}, args = {typeSpec.getNumArgs}"
                if typeSpec.getKind == ``Lean.Parser.Term.typeSpec && typeSpec.getNumArgs >= 2 then
                  let actualType := typeSpec[1]
                  dbg_trace s!"    actualType kind = {actualType.getKind}, args = {actualType.getNumArgs}, isIdent = {actualType.isIdent}"
                  if actualType.isIdent then
                    dbg_trace s!"      type name = {actualType.getId}"
                  else if actualType.getNumArgs >= 2 then
                    let typeHead := actualType[0]
                    dbg_trace s!"      typeHead kind = {typeHead.getKind}, args = {typeHead.getNumArgs}, isIdent = {typeHead.isIdent}"
                    if typeHead.isIdent then
                      dbg_trace s!"        typeHead name = {typeHead.getId}"

  `(def dummy : Nat := 42)

structure ArrayArgs where
  input : Name
  output : Name

debug_syntax := do
  let x : GlobalArray Float := ⟨`foo⟩
  let i ← globalIdxX
  pure ()
