/-
  Device Macro - Syntax-Level Extraction to DeviceIR

  Extracts DeviceIR from KernelM syntax before elaboration.
  This avoids all the elaboration/reduction issues.
-/

import Lean
import CLean.GPU
import CLean.DeviceIR

set_option maxHeartbeats 2000000
set_option maxRecDepth 1000

open Lean Lean.Elab Lean.Elab.Command Lean.Parser.Term
open DeviceIR GpuDSL

namespace CLean.DeviceMacro

/-! ## Context for tracking extraction state -/

structure ExtractCtx where
  globalArrays : Array Name := #[]  -- Track global array names
  sharedArrays : Array Name := #[]  -- Track shared array names
  arrayMap : Std.HashMap String Name := {}  -- Map local var to actual array name

/-! ## Expression Conversion -/

/-- Convert term syntax to DExpr syntax -/
partial def exprToDExpr (stx : Syntax) : MacroM Syntax := do
  match stx with
  -- Identifiers
  | `($id:ident) =>
    let name := id.getId
    if name == `globalIdxX then
      `(DExpr.binop BinOp.add
         (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
         (DExpr.threadIdx Dim.x))
    else
      `(DExpr.var $(quote name.toString))

  -- Numeric literals
  | `($n:num) =>
    let val := n.getNat
    `(DExpr.intLit $(Syntax.mkNumLit (toString (Int.ofNat val))))

  -- Field access (e.g., args.N)
  | `($obj:ident.$field:ident) =>
    `(DExpr.var $(quote field.getId.toString))

  -- Binary operations
  | `($a:term + $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.add $aeTSyntax $beTSyntax)

  | `($a:term - $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.sub $aeTSyntax $beTSyntax)

  | `($a:term * $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.mul $aeTSyntax $beTSyntax)

  | `($a:term / $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.div $aeTSyntax $beTSyntax)

  -- Comparisons
  | `($a:term < $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.lt $aeTSyntax $beTSyntax)

  | `($a:term <= $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.le $aeTSyntax $beTSyntax)

  | `($a:term == $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.eq $aeTSyntax $beTSyntax)

  -- Parenthesized expressions
  | `(($e:term)) => exprToDExpr e

  -- Default: treat as variable
  | _ =>
    `(DExpr.var "unknown")

/-! ## Statement Extraction -/

/-- Extract DStmt list from do-sequence items (simplified version) -/
partial def extractDoItems (items : Array Syntax) (ctx : ExtractCtx) : MacroM (Array Syntax × ExtractCtx) := do
  let mut dstmts : Array Syntax := #[]
  let mut newCtx := ctx

  for item in items do
    -- Each item is a doSeqItem, extract the doElem
    if item.getKind != ``Lean.Parser.Term.doSeqItem then
      continue

    let doElem := item[0]
    -- dbg_trace s!"Processing doElem kind: {doElem.getKind}"

    -- Use match instead of if-let to avoid timeout
    match doElem with
    -- PATTERN: let i ← globalIdxX
    | `(doElem| let $id:ident ← globalIdxX) =>
      -- dbg_trace s!"✓ Matched globalIdxX pattern"
      let varName := id.getId.toString
      -- Build the globalIdx expression separately
      let globalIdxExpr ← `(DExpr.binop BinOp.add
        (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
        (DExpr.threadIdx Dim.x))
      let globalIdxTSyntax : TSyntax `term := ⟨globalIdxExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $globalIdxTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: let x : GlobalArray T := ⟨args.field⟩
    | `(doElem| let $id:ident : GlobalArray $_ := ⟨$rhs:term⟩) =>
      let varName := id.getId
      -- dbg_trace s!"✓ Matched GlobalArray pattern for {varName}, rhs kind: {rhs.raw.getKind}"

      -- Check if rhs is just an ident (the field name directly)
      if rhs.raw.isIdent then
        let fullName := rhs.raw.getId
        -- Extract just the last component (e.g., `args.x` → `x`)
        let fieldName := fullName.components.getLast!
        -- dbg_trace s!"✓ Tracked global array (direct ident): {fullName} → {fieldName}"
        newCtx := { newCtx with
          globalArrays := newCtx.globalArrays.push fieldName,
          arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName
        }
      -- Extract field name from args.field
      else if let `($argsId:ident.$field:ident) := rhs then
        if argsId.getId == `args then
          let fieldName := field.getId
          -- dbg_trace s!"✓ Tracked global array (projection): {fieldName}"
          newCtx := { newCtx with
            globalArrays := newCtx.globalArrays.push fieldName,
            arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName
          }
      -- else
        -- dbg_trace s!"✗ RHS did not match expected patterns"

    -- PATTERN: let x := expr (scalar assignments)
    | `(doElem| let $id:ident := $rhs:term) =>
      let varName := id.getId.toString
      -- Try to convert the RHS to a DExpr
      -- This handles: projections (args.N), binary ops (a * b), literals, etc.
      let rhsExpr ← exprToDExpr rhs
      let rhsTSyntax : TSyntax `term := ⟨rhsExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $rhsTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: let val ← arr.get idx (array reads)
    | `(doElem| let $id:ident ← $rhs:term) =>
      -- dbg_trace s!"Matched doLetArrow pattern, rhs kind: {rhs.raw.getKind}"
      -- Check if getArgs
      if rhs.raw.isIdent && rhs.raw.getId == `getArgs then
        pure ()  -- Skip getArgs
      -- Check if it's arr.get idx pattern
      else if rhs.raw.getKind == ``Lean.Parser.Term.app then
        -- dbg_trace s!"✓ rhs is app, checking for arr.get pattern"
        let fn := rhs.raw.getArg 0
        -- dbg_trace s!"  fn kind: {fn.getKind}"

        -- Check if fn is an ident like "input.get"
        if fn.isIdent then
          let fullName := fn.getId
          let components := fullName.components
          -- dbg_trace s!"  fn is ident: {fullName}, components: {components.length}"
          if components.length >= 2 && components.getLast! == `get then
            -- This is arr.get pattern - extract array name (all components except last)
            let arrName := components[components.length - 2]!  -- Second to last is the array name
            let valName := id.getId.toString
            let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
            -- dbg_trace s!"  ✓✓ Matched arr.get: array={arrName}, actualArray={actualArrayName}"
            -- Get the index argument (for qualified ident calls, args are at position 1, wrapped in null)
            let arg1 := rhs.raw.getArg 1
            -- dbg_trace s!"    arg1 kind: {arg1.getKind}, numArgs: {arg1.getNumArgs}"
            if arg1.getKind == `null && arg1.getNumArgs > 0 then
              -- dbg_trace s!"    ✓ Extracting arr.get statement from arg1"
              let idx := arg1.getArg 0  -- Unwrap the null node
              let idxDExpr ← exprToDExpr idx
              let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
              -- Generate: let valName := arr[idx]
              let dstmt ← `(DStmt.assign $(quote valName)
                             (DExpr.index (DExpr.var $(quote actualArrayName.toString))
                                          $idxTSyntax))
              dstmts := dstmts.push dstmt
              -- dbg_trace s!"    ✓ Added arr.get statement to dstmts"

    -- Check arr.set pattern separately (can't use quotation pattern for doExpr)
    | _ =>
      -- dbg_trace s!"✗ No pattern matched, checking arr.set in default case"
      if doElem.getKind == ``Lean.Parser.Term.doExpr then
        let expr := doElem.getArg 0
        -- Check if it's arr.set idx val
        if expr.getKind == ``Lean.Parser.Term.app then
          let fn := expr.getArg 0
          -- Check if fn is an ident like "output.set"
          if fn.isIdent then
            let fullName := fn.getId
            let components := fullName.components
            if components.length >= 2 && components.getLast! == `set then
              -- This is arr.set pattern
              let arrName := components[components.length - 2]!
              let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
              -- dbg_trace s!"  ✓✓ Matched arr.set: array={arrName}, actualArray={actualArrayName}"
              -- Get arguments: idx and val (at position 1, which should be a null node with args)
              let args := expr.getArg 1
              -- dbg_trace s!"    args kind: {args.getKind}, numArgs: {args.getNumArgs}"
              if args.getKind == `null && args.getNumArgs >= 2 then
                -- dbg_trace s!"    ✓ Extracting arr.set statement"
                let idx := args.getArg 0
                let val := args.getArg 1
                let idxDExpr ← exprToDExpr idx
                let valDExpr ← exprToDExpr val
                let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                -- Generate: arr[idx] := val
                let dstmt ← `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                               $idxTSyntax $valTSyntax)
                dstmts := dstmts.push dstmt
      -- Skip all other patterns (including if-then-else for now)

  return (dstmts, newCtx)

/-! ## Main Extraction Function -/

/-- Extract DeviceIR kernel from a KernelM definition syntax -/
def extractKernelFromSyntax (kernelName : Name) (kernelBody : TSyntax `Lean.Parser.Term.do) :
    MacroM Syntax := do
  -- Get do-sequence items
  let doSeq := kernelBody.raw.getArg 1
  let items := if doSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                   doSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
    if doSeq.getNumArgs > 0 then
      doSeq.getArg 0 |>.getArgs
    else
      #[]
  else
    #[]

  -- Extract statements
  let initialCtx : ExtractCtx := {}
  let (stmts, finalCtx) ← extractDoItems items initialCtx

  -- Build the body by sequencing statements
  let mut body ← `(DStmt.skip)
  for stmt in stmts.reverse do
    let stmtTSyntax : TSyntax `term := ⟨stmt⟩
    body ← `(DStmt.seq $stmtTSyntax $body)

  -- Build arrays
  let globalArraysList ← finalCtx.globalArrays.mapM fun name =>
    `({ name := $(quote name.toString), ty := DType.array DType.float, space := MemorySpace.global })

  -- Build the kernel structure
  `({ name := $(quote kernelName.toString),
      params := [],  -- TODO: extract from args type
      locals := [],  -- TODO: track locals with types
      globalArrays := [$(globalArraysList),*],
      sharedArrays := [],
      body := $body : Kernel })

/-! ## Device Kernel Macro -/

/-- Device kernel macro - generates both KernelM def and DeviceIR Kernel -/
macro "device_kernel " name:ident sig:optDeclSig val:declVal : command => do
  -- Generate the KernelM definition
  let kernelDefStx ← `(def $name $sig:optDeclSig $val:declVal)

  -- Extract the body to analyze
  let body ← match val with
    | `(declVal| := $body:term) => pure body
    | _ =>
      -- No body to extract, generate stub
      let irName := Lean.mkIdent (name.getId.appendAfter "IR")
      let nameQuote := quote name.getId.toString
      let kernelDefStx ← `(
        def $irName : DeviceIR.Kernel := {
          name := $nameQuote
          params := []
          locals := []
          globalArrays := []
          sharedArrays := []
          body := DStmt.skip
        }
      )
      return ← `($kernelDefStx:command
                 $kernelDefStx:command)

  -- Check if it's do-notation
  if body.raw.getKind != ``Lean.Parser.Term.do then
    -- Not do-notation, generate stub IR
    let irName := Lean.mkIdent (name.getId.appendAfter "IR")
    let nameQuote := quote name.getId.toString
    let kernelDefStx ← `(
      def $irName : DeviceIR.Kernel := {
        name := $nameQuote
        params := []
        locals := []
        globalArrays := []
        sharedArrays := []
        body := DStmt.skip
      }
    )
    return ← `($kernelDefStx:command
               $kernelDefStx:command)

  -- Extract do-sequence items
  let doSeq := body.raw[1]
  let items := if doSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                   doSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
    if doSeq.getNumArgs > 0 then
      doSeq[0].getArgs
    else
      #[]
  else
    #[]

  -- Use the full extraction logic
  let initialCtx : ExtractCtx := {}
  let (stmts, finalCtx) ← extractDoItems items initialCtx

  -- Debug: log how many statements and arrays we found
  -- dbg_trace s!"Extracted {stmts.size} statements, {finalCtx.globalArrays.size} global arrays, {items.size} items total"

  -- Build arrays from finalCtx
  let mut globalArraySyntax : Array (TSyntax `term) := #[]
  for arrName in finalCtx.globalArrays do
    let s ← `({ name := $(quote arrName.toString), ty := DType.array DType.float, space := MemorySpace.global })
    globalArraySyntax := globalArraySyntax.push s

  -- Build the body by sequencing statements
  let mut bodyExpr ← `(DStmt.skip)
  for stmt in stmts.reverse do
    let stmtTSyntax : TSyntax `term := ⟨stmt⟩
    bodyExpr ← `(DStmt.seq $stmtTSyntax $bodyExpr)

  -- Generate DeviceIR Kernel definition
  let irName := Lean.mkIdent (name.getId.appendAfter "IR")
  let nameQuote := quote name.getId.toString

  let kernelIRDefStx ← `(
    def $irName : DeviceIR.Kernel := {
      name := $nameQuote
      params := []
      locals := []
      globalArrays := [$globalArraySyntax,*]
      sharedArrays := []
      body := $bodyExpr
    }
  )

  -- Return both definitions
  `($kernelDefStx:command
    $kernelIRDefStx:command)

end CLean.DeviceMacro
