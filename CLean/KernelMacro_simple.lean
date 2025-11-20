import Lean
import CLean.GPU
import CLean.VerifyIR

/-! # Simplified GPU Kernel Macro - Minimal Working Version

This is a minimal version that compiles. We'll add features incrementally.
-/

open Lean Lean.Elab Lean.Elab.Command Lean.Parser.Term
open CLean.VerifyIR GpuDSL

namespace CLean.KernelMacro

set_option maxHeartbeats 400000

/-- GPU kernel macro - generates both KernelM def and VKernel IR -/
macro "gpu_kernel " name:ident sig:optDeclSig val:declVal : command => do
  -- Generate the KernelM definition (unchanged from original)
  let kernelDefStx ← `(def $name $sig:optDeclSig $val:declVal)

  -- Extract the body to analyze
  let body ← match val with
    | `(declVal| := $body:term) => pure body
    | _ =>
      -- No body to extract, just generate stub IR
      let irName := Lean.mkIdent (name.getId.appendAfter "IR")
      let nameQuote := quote name.getId
      let vkernelDefStx ← `(
        def $irName : CLean.VerifyIR.VKernel := {
          name := $nameQuote
          params := []
          locals := []
          globalArrays := []
          sharedArrays := []
          body := []
        }
      )
      return ← `($kernelDefStx:command
                 $vkernelDefStx:command)

  -- Check if it's do-notation
  if body.raw.getKind != ``Lean.Parser.Term.do then
    -- Not do-notation, generate stub IR
    let irName := Lean.mkIdent (name.getId.appendAfter "IR")
    let nameQuote := quote name.getId
    let vkernelDefStx ← `(
      def $irName : CLean.VerifyIR.VKernel := {
        name := $nameQuote
        params := []
        locals := []
        globalArrays := []
        sharedArrays := []
        body := []
      }
    )
    return ← `($kernelDefStx:command
               $vkernelDefStx:command)

  -- Extract do-sequence items
  let doSeq := body.raw[1]  -- doSeqIndent or doSeqBracketed
  let items := if doSeq.getArgs.size >= 1 then
      doSeq[0].getArgs
    else
      #[]

  -- Inline extraction: process each statement
  let mut vstmts : Array (TSyntax `term) := #[]
  let mut localVars : Array Name := #[]
  let mut globalArrays : Array Name := #[]
  let mut sharedArrays : Array Name := #[]

  for item in items do
    -- Each item is a doSeqItem, extract the doElem
    if item.getKind != ``Lean.Parser.Term.doSeqItem then
      continue

    let doElem := item[0]

    -- First check for array declarations (let x : GlobalArray/SharedArray T := value)
    -- These use doLet, not doLetArrow
    if doElem.getKind == ``Lean.Parser.Term.doLet then
      -- doLet structure: "let" null letDecl
      if doElem.getNumArgs >= 3 then
        let letDeclGroup := doElem[2]
        if letDeclGroup.getNumArgs >= 1 then
          let letDecl := letDeclGroup[0]
          -- letIdDecl structure: ident null(type stuff) null ":=" value
          if letDecl.getKind == ``Lean.Parser.Term.letIdDecl && letDecl.getNumArgs >= 5 then
            let id := letDecl[0]
            -- letDecl[2] contains the type info
            if letDecl[2].getNumArgs >= 1 then
              let typeSpec := letDecl[2][0]
              if typeSpec.getKind == ``Lean.Parser.Term.typeSpec && typeSpec.getNumArgs >= 2 then
                let actualType := typeSpec[1]

                -- Check if this is GlobalArray or SharedArray
                let isGlobalArray :=
                  if actualType.isIdent then
                    actualType.getId == `GlobalArray
                  else if actualType.getNumArgs >= 2 then
                    actualType[0].isIdent && actualType[0].getId == `GlobalArray
                  else
                    false

                let isSharedArray :=
                  if actualType.isIdent then
                    actualType.getId == `SharedArray
                  else if actualType.getNumArgs >= 2 then
                    actualType[0].isIdent && actualType[0].getId == `SharedArray
                  else
                    false

                if (isGlobalArray || isSharedArray) && id.isIdent then
                  let varName := id.getId
                  if isGlobalArray then
                    globalArrays := globalArrays.push varName
                  else
                    sharedArrays := sharedArrays.push varName
                  continue  -- Skip other patterns for this item

    -- Pattern: let i ← globalIdxX
    if let `(doElem| let $id:ident ← globalIdxX) := doElem then
      let varName := id.getId
      let vstmt ← `({ stmt := VStmtKind.assign $(quote varName)
                        (VExpr.add (VExpr.mul VExpr.blockIdX VExpr.blockDimX) VExpr.threadIdX),
                      predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt
      localVars := localVars.push varName

    -- Pattern: let i ← globalIdxY
    else if let `(doElem| let $id:ident ← globalIdxY) := doElem then
      let varName := id.getId
      let vstmt ← `({ stmt := VStmtKind.assign $(quote varName)
                        (VExpr.add (VExpr.mul VExpr.blockIdY VExpr.blockDimY) VExpr.threadIdY),
                      predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt
      localVars := localVars.push varName

    -- Pattern: barrier
    else if let `(doElem| barrier) := doElem then
      let vstmt ← `({ stmt := VStmtKind.barrier, predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt

  -- Build VarInfo syntax for local variables
  let mut localsSyntax : Array (TSyntax `term) := #[]
  for varName in localVars do
    let s ← `({ name := $(quote varName), type := VType.nat,
                uniformity := Uniformity.nonUniform, memorySpace := MemorySpace.local })
    localsSyntax := localsSyntax.push s

  -- Build VarInfo syntax for global arrays
  let mut globalArraySyntax : Array (TSyntax `term) := #[]
  for arrName in globalArrays do
    let s ← `({ name := $(quote arrName), type := VType.float,
                uniformity := Uniformity.uniform, memorySpace := MemorySpace.global })
    globalArraySyntax := globalArraySyntax.push s

  -- Build VarInfo syntax for shared arrays
  let mut sharedArraySyntax : Array (TSyntax `term) := #[]
  for arrName in sharedArrays do
    let s ← `({ name := $(quote arrName), type := VType.float,
                uniformity := Uniformity.uniform, memorySpace := MemorySpace.shared })
    sharedArraySyntax := sharedArraySyntax.push s

  -- Generate VKernel IR definition
  let irName := Lean.mkIdent (name.getId.appendAfter "IR")
  let nameQuote := quote name.getId

  let vkernelDefStx ← `(
    def $irName : CLean.VerifyIR.VKernel := {
      name := $nameQuote
      params := []
      locals := [$localsSyntax,*]
      globalArrays := [$globalArraySyntax,*]
      sharedArrays := [$sharedArraySyntax,*]
      body := [$vstmts,*]
    }
  )

  -- Return both definitions as a command sequence
  `($kernelDefStx:command
    $vkernelDefStx:command)

end CLean.KernelMacro
