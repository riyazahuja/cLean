import Lean
import CLean.GPU
import CLean.VerifyIR

/-! # GPU Kernel Macro with Automatic IR Extraction

Provides `gpu_kernel` macro that generates both executable KernelM and VKernel IR.
Works by parsing surface syntax before elaboration, avoiding bound variable issues.
-/

open Lean Lean.Elab Lean.Elab.Command Lean.Parser.Term
open CLean.VerifyIR GpuDSL

namespace CLean.KernelMacro

/-! ## Context for tracking extraction state -/

structure ExtractCtx where
  globalArrays : Array Name := #[]  -- Track global array names
  sharedArrays : Array Name := #[]  -- Track shared array names
  localVars : Array Name := #[]     -- Track local variable names
  arrayMap : Std.HashMap String Name := {}  -- Map local var to actual array name

/-! ## Expression Conversion -/

/-- Convert term syntax to VExpr syntax -/
partial def exprToVExpr (stx : Syntax) : MacroM Syntax := do
  match stx with
  -- Identifiers
  | `($id:ident) =>
    let name := id.getId
    if name == `globalIdxX then
      `(VExpr.add (VExpr.mul VExpr.blockIdX VExpr.blockDimX) VExpr.threadIdX)
    else if name == `globalIdxY then
      `(VExpr.add (VExpr.mul VExpr.blockIdY VExpr.blockDimY) VExpr.threadIdY)
    else if name == `threadIdX then
      `(VExpr.threadIdX)
    else if name == `blockIdX then
      `(VExpr.blockIdX)
    else if name == `blockDimX then
      `(VExpr.blockDimX)
    else
      `(VExpr.var $(quote name))

  -- Numeric literals
  | `($n:num) =>
    let val := n.getNat
    `(VExpr.constInt $(Syntax.mkNumLit (toString val)))

  -- Field access (e.g., args.N)
  | `($obj:ident.$field:ident) =>
    if obj.getId == `args then
      `(VExpr.var $(quote field.getId))
    else
      `(VExpr.var $(quote field.getId))

  -- Binary operations - build manually
  | `($a:term + $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.add _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term - $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.sub _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term * $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.mul _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term / $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.div _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term % $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.mod _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  -- Comparisons
  | `($a:term < $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.lt _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term <= $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.le _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term == $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.eq _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  -- Logical operations
  | `($a:term && $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.land _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  | `($a:term || $b:term) => do
    let ae ← exprToVExpr a
    let be ← exprToVExpr b
    let res ← `(VExpr.lor _ _)
    pure <| res.raw.modifyArg 1 (fun _ => ae) |>.modifyArg 2 (fun _ => be)

  -- Parenthesized expressions
  | `(($e:term)) => exprToVExpr e

  -- Default: treat as unknown variable
  | _ => `(VExpr.var `unknown)

/-! ## Statement Extraction -/

/-- Helper: Extract items from a do-sequence branch -/
def getDoSeqItems (branch : TSyntax `Lean.Parser.Term.doSeq) : Array Syntax :=
  let seq := branch.raw[1]  -- doSeqIndent or doSeqBracketed
  if seq.getArgs.size >= 1 then
    seq[0].getArgs  -- Array of doSeqItems
  else
    #[]

/-- Extract VStmt list from do-sequence items -/
partial def extractDoItems (items : Array Syntax) (ctx : ExtractCtx) : MacroM (Array Syntax × ExtractCtx) := do
  let mut vstmts : Array Syntax := #[]
  let mut newCtx := ctx

  for item in items do
    -- Each item is a doSeqItem, extract the doElem
    if item.getKind != ``Lean.Parser.Term.doSeqItem then
      continue

    let doElem := item[0]

    match doElem with
    -- let i ← globalIdxX
    | `(doElem| let $id:ident ← globalIdxX) => do
      let varName := id.getId
      let vstmt ← `({ stmt := VStmtKind.assign $(quote varName)
                        (VExpr.add (VExpr.mul VExpr.blockIdX VExpr.blockDimX) VExpr.threadIdX),
                      predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt
      newCtx := { newCtx with localVars := newCtx.localVars.push varName }

    -- let i ← globalIdxY
    | `(doElem| let $id:ident ← globalIdxY) => do
      let varName := id.getId
      let vstmt ← `({ stmt := VStmtKind.assign $(quote varName)
                        (VExpr.add (VExpr.mul VExpr.blockIdY VExpr.blockDimY) VExpr.threadIdY),
                      predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt
      newCtx := { newCtx with localVars := newCtx.localVars.push varName }

    -- Simple let bindings (skip)
    | `(doElem| let $_:ident := $_:term) =>
      pure ()

    -- let x : GlobalArray T := ⟨args.field⟩
    | `(doElem| let $id:ident : GlobalArray $_ := ⟨$args:ident.$field:ident⟩) => do
      if args.getId == `args then
        let varName := id.getId
        let fieldName := field.getId
        newCtx := { newCtx with
          globalArrays := newCtx.globalArrays.push fieldName,
          arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName
        }

    -- let x : SharedArray T := ⟨args.field⟩
    | `(doElem| let $id:ident : SharedArray $_ := ⟨$args:ident.$field:ident⟩) => do
      if args.getId == `args then
        let varName := id.getId
        let fieldName := field.getId
        newCtx := { newCtx with
          sharedArrays := newCtx.sharedArrays.push fieldName,
          arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName
        }

    -- let x := expr (simple let, like reading args.N)
    | `(doElem| let $_:ident := $_:term) =>
      pure ()  -- Skip, don't track parameters

    -- let args ← getArgs
    | `(doElem| let $_:ident ← getArgs) =>
      pure ()  -- Skip

    -- if cond then ... [else ...]
    | `(doElem| if $cond:term then $thenBranch:doSeq $[else $elseBranch:doSeq]?) => do
      let condVExpr ← exprToVExpr cond

      -- Extract then branch
      let thenItems := getDoSeqItems thenBranch
      let (thenStmts, _) ← extractDoItems thenItems newCtx

      -- Extract else branch if present
      let elseStmts ← match elseBranch with
        | some eb =>
          let elseItems := getDoSeqItems eb
          let (stmts, _) ← extractDoItems elseItems newCtx
          pure stmts
        | none => pure #[]

      -- Build if-then-else statement manually
      let vstmt ← `({ stmt := VStmtKind.ite _ [] [], predicate := VExpr.constBool true })
      let vstmt := vstmt.raw.modifyArg 1 |>.modifyArg 1 |>.modifyArg 1 (fun _ => condVExpr)
      let vstmt := vstmt.modifyArg 1 |>.modifyArg 2 (fun _ => Syntax.mkNode `null thenStmts)
      let vstmt := vstmt.modifyArg 1 |>.modifyArg 3 (fun _ => Syntax.mkNode `null elseStmts)
      vstmts := vstmts.push ⟨vstmt⟩

    -- barrier
    | `(doElem| barrier) => do
      let vstmt ← `({ stmt := VStmtKind.barrier, predicate := VExpr.constBool true })
      vstmts := vstmts.push vstmt

    | _ =>
      -- Unknown pattern, skip with debug trace
      dbg_trace f!"Skipping unknown pattern: {doElem.getKind}"
      pure ()

  return (vstmts, newCtx)

/-! ## Main Macro -/

/-- The gpu_kernel macro - generates both KernelM definition and VKernel IR -/
macro "gpu_kernel " name:ident sig:optDeclSig val:declVal : command => do
  -- Extract the body
  let body ← match val with
    | `(declVal| := $body:term) => pure body
    | _ => Macro.throwError "Expected := body"

  -- Check if it's do-notation
  if body.raw.getKind != ``Lean.Parser.Term.do then
    Macro.throwError "Expected do-notation body"

  -- Extract do-sequence items
  let doSeq := body.raw[1]  -- doSeqIndent or doSeqBracketed
  let items := if doSeq.getArgs.size >= 1 then
      doSeq[0].getArgs
    else
      #[]

  -- Extract statements and track arrays/vars
  let (vstmts, extractCtx) ← extractDoItems items {}

  -- Build VarInfo syntax for global arrays
  let mut globalArraySyntax : Array Syntax := #[]
  for arrName in extractCtx.globalArrays do
    let s ← `({ name := $(quote arrName), type := VType.float,
                uniformity := Uniformity.uniform, memorySpace := MemorySpace.global })
    globalArraySyntax := globalArraySyntax.push s

  -- Build VarInfo syntax for shared arrays
  let mut sharedArraySyntax : Array Syntax := #[]
  for arrName in extractCtx.sharedArrays do
    let s ← `({ name := $(quote arrName), type := VType.float,
                uniformity := Uniformity.uniform, memorySpace := MemorySpace.shared })
    sharedArraySyntax := sharedArraySyntax.push s

  -- Build VarInfo syntax for local variables
  let mut localsSyntax : Array Syntax := #[]
  for varName in extractCtx.localVars do
    let s ← `({ name := $(quote varName), type := VType.nat,
                uniformity := Uniformity.nonUniform, memorySpace := MemorySpace.local })
    localsSyntax := localsSyntax.push s

  let irName := Lean.mkIdent (name.getId.appendAfter "_ir")

  -- Generate both definitions
  `(
    def $name $sig $val

    def $irName : VKernel := {
      name := $(quote name.getId)
      params := []
      locals := [$localsSyntax,*]
      globalArrays := [$globalArraySyntax,*]
      sharedArrays := [$sharedArraySyntax,*]
      body := [$vstmts,*]
    }
  )

end CLean.KernelMacro
