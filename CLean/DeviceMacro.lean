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
  -- Track parameters extracted from args.* assignments
  params : Array (String × String) := #[]  -- (name, type as string)
  -- Track local variable types
  localTypes : Std.HashMap String String := {}  -- var name -> type string
  -- Track array element types (array name -> DType)
  arrayTypes : Std.HashMap String DType := {}  -- array name -> element type

/-! ## Type Inference Helpers -/

/-- Infer CUDA type from variable name (simple heuristic) -/
def inferCudaType (varName : String) : String :=
  -- Simple heuristics based on common naming patterns
  if varName == "N" || varName == "length" || varName.endsWith "Idx" || varName.endsWith "idx" then
    "int"
  else if varName.startsWith "alpha" || varName.startsWith "beta" || varName.endsWith "Weight" then
    "float"
  else if varName.endsWith "d" || varName.endsWith "1" then
    "int"  -- twod, twod1, etc.
  else
    "float"  -- Default to float

/-- Recursively extract scalar parameters from syntax (find all args.* references) -/
partial def extractScalarParamsFromSyntax (stx : Syntax) : Array (String × String) := Id.run do
  let mut params := #[]

  -- Case 1: Qualified identifier like `args.field`
  if stx.isIdent then
    let full := stx.getId
    let comps := full.components
    if comps.length == 2 && comps[0]! == `args then
      let fieldName := comps[1]!.toString
      let ty := inferCudaType fieldName
      -- dbg_trace s!"  [extractScalarParams] Found qualified ident: args.{fieldName} : {ty}"
      params := params.push (fieldName, ty)

  -- Case 2: Projection syntax `obj.field`
  if stx.getKind == ``Lean.Parser.Term.proj then
    let recv := stx.getArg 0
    let field := stx.getArg 2
    if recv.isIdent && recv.getId == `args && field.isIdent then
      let fieldName := field.getId.toString
      let ty := inferCudaType fieldName
      -- dbg_trace s!"  [extractScalarParams] Found projection: args.{fieldName} : {ty}"
      params := params.push (fieldName, ty)

  -- Recursively process all child nodes
  for arg in stx.getArgs do
    params := params ++ extractScalarParamsFromSyntax arg

  return params

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
      -- Check if it's a qualified identifier like `args.field`
      let comps := name.components
      if comps.length == 2 && comps[0]! == `args then
        -- Strip args prefix, just use the field name
        `(DExpr.var $(quote comps[1]!.toString))
      else
        `(DExpr.var $(quote name.toString))

  -- Numeric literals (integers)
  | `($n:num) =>
    let val := n.getNat
    `(DExpr.intLit $(Syntax.mkNumLit (toString (Int.ofNat val))))

  -- Field access (e.g., args.N)
  | `($obj:ident.$field:ident) =>
    -- If it's args.field, just use the field name (it's a parameter)
    if obj.getId == `args then
      `(DExpr.var $(quote field.getId.toString))
    else
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

  -- Logical operators
  | `($a:term && $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.and $aeTSyntax $beTSyntax)

  | `($a:term || $b:term) => do
    let ae ← exprToDExpr a
    let be ← exprToDExpr b
    let aeTSyntax : TSyntax `term := ⟨ae⟩
    let beTSyntax : TSyntax `term := ⟨be⟩
    `(DExpr.binop BinOp.or $aeTSyntax $beTSyntax)

  -- Parenthesized expressions
  | `(($e:term)) => exprToDExpr e

    -- Float / scientific literals (e.g., 1.0, 3.14, 2e-3)
  | `($f:scientific) =>
    let fTerm : TSyntax `term := ⟨f.raw⟩
    `(DExpr.floatLit $fTerm)

  | `(-$e:term) => do
  let ee ← exprToDExpr e
  let eeT : TSyntax `term := ⟨ee⟩
  -- no unop in your IR, so encode as 0 - e
  `(DExpr.binop BinOp.sub (DExpr.intLit 0) $eeT)

  -- Default: check for special patterns, otherwise treat as variable
  | _ =>


    -- if let `(Neg.neg (OfScientific.ofScientific $m:num $pos:ident $e:num)) := stx then
    --   let mNat := m.getNat
    --   let eNat := e.getNat
    --   let posBool := pos.getId == `true
    --   let fTerm : TSyntax `term ←
    --     `(Float.ofScientific $(quote mNat) $(quote posBool) $(quote eNat))
    --   `(DExpr.floatLit (-$fTerm))

    -- else if let `(OfScientific.ofScientific $m:num $pos:ident $e:num) := stx then
    --   let mNat := m.getNat
    --   let eNat := e.getNat
    --   let posBool := pos.getId == `true
    --   let fTerm : TSyntax `term ←
    --     `(Float.ofScientific $(quote mNat) $(quote posBool) $(quote eNat))
    --   `(DExpr.floatLit $fTerm)

    -- Check for .toNat?.getD pattern: (expr).toNat?.getD defaultVal
    -- This is represented as: Term.app with function being a projection chain
    if stx.getKind == ``Lean.Parser.Term.app then
      -- Get the function and arguments
      let fn := stx.getArg 0
      let args := stx.getArg 1

      -- dbg_trace s!"exprToDExpr app: fn.kind={fn.getKind}, fn.isIdent={fn.isIdent}, args.kind={args.getKind}"

      -- Check if function is a projection (method call like .getD or .toNat?.getD)
      if fn.getKind == ``Lean.Parser.Term.proj then
        -- Get the method name from the projection
        -- Projection structure: [0]=receiver, [1]=dot, [2]=field name
        let methodName := fn.getArg 2
        -- dbg_trace s!"  projection method: {methodName}, methodName.getId={methodName.getId}"

        -- Check if this is .toNat?.getD (combined projection)
        -- The syntax parser combines chained projections into a single identifier like `toNat?.getD`
        if methodName.isIdent && methodName.getId == `toNat?.getD then
          -- dbg_trace s!"  Found .toNat?.getD call"
          -- Extract the base expression (receiver before the projection chain)
          let base := fn.getArg 0
          -- Recursively convert the base expression
          exprToDExpr base
        else
          -- Not a .toNat?.getD call, treat as unknown
          `(DExpr.var "unknown")
      else
        -- Not a projection, treat as unknown
        -- dbg_trace s!"exprToDExpr: unhandled syntax kind: {stx.getKind}"
        `(DExpr.var "unknown")
    else
      -- dbg_trace s!"exprToDExpr: unhandled syntax kind: {stx.getKind}"
      `(DExpr.var "unknown")

/-! ## Statement Extraction -/


def endsWith (n : Name) (suffix : Name) : Bool :=
  n.components.getLast? == some suffix

partial def containsSuffix (stx : Syntax) (suffix : Name) : Bool :=
  if stx.isIdent then
    endsWith stx.getId suffix
  else
    stx.getArgs.any (fun a => containsSuffix a suffix)

def kindEndsWith (stx : Syntax) (suffix : Name) : Bool :=
  stx.getKind.components.getLast? == some suffix

partial def containsGetArgs (stx : Syntax) : Bool :=
  -- direct ident getArgs (covers unexpanded cases)
  (stx.isIdent && endsWith stx.getId `getArgs) ||

  -- expanded macro head: GpuDSL.termGetArgs ...
  kindEndsWith stx `termGetArgs ||

  -- sometimes macro kind itself may be `getArgs`
  kindEndsWith stx `getArgs ||

  -- expanded macro argument is a string literal "getArgs"
  (stx.isStrLit?.isSome && (stx.isStrLit?.getD "") == "getArgs") ||

  -- recurse through children
  stx.getArgs.any containsGetArgs

/-- Extract DStmt list from do-sequence items (simplified version) -/
partial def extractDoItems (items : Array Syntax) (ctx : ExtractCtx) : MacroM (Array Syntax × ExtractCtx) := do
  let mut dstmts : Array Syntax := #[]
  let mut newCtx := ctx

  -- dbg_trace s!"[extractDoItems] Processing {items.size} items"

  for item in items do
    -- Each item is a doSeqItem, extract the doElem
    -- dbg_trace s!"  item kind: {item.getKind}"
    if item.getKind != ``Lean.Parser.Term.doSeqItem then
      continue

    let doElem := item[0]
    -- dbg_trace s!"  Processing doElem kind: {doElem.getKind}"

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

    -- PATTERN: let i ← globalIdxY
    | `(doElem| let $id:ident ← globalIdxY) =>
      let varName := id.getId.toString
      -- Build the globalIdxY expression
      let globalIdxExpr ← `(DExpr.binop BinOp.add
        (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.y) (DExpr.blockDim Dim.y))
        (DExpr.threadIdx Dim.y))
      let globalIdxTSyntax : TSyntax `term := ⟨globalIdxExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $globalIdxTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: let i ← globalIdxZ
    | `(doElem| let $id:ident ← globalIdxZ) =>
      let varName := id.getId.toString
      -- Build the globalIdxZ expression
      let globalIdxExpr ← `(DExpr.binop BinOp.add
        (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.z) (DExpr.blockDim Dim.z))
        (DExpr.threadIdx Dim.z))
      let globalIdxTSyntax : TSyntax `term := ⟨globalIdxExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $globalIdxTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: let x : GlobalArray T := ⟨args.field⟩
    | `(doElem| let $id:ident : GlobalArray $elemTy:ident := ⟨$rhs:term⟩) =>
      let varName := id.getId
      -- dbg_trace s!"✓ Matched GlobalArray pattern for {varName}, rhs kind: {rhs.raw.getKind}"

      -- Parse element type
      let dtype : DType :=
        match elemTy.getId.toString with
        | "Int" => DType.int
        | "Nat" => DType.nat
        | "Float" => DType.float
        | "Bool" => DType.bool
        | _ => DType.float  -- Default to float

      -- Check if rhs is just an ident (the field name directly)
      if rhs.raw.isIdent then
        let fullName := rhs.raw.getId
        -- Extract just the last component (e.g., `args.x` → `x`)
        let fieldName := fullName.components.getLast!
        -- dbg_trace s!"✓ Tracked global array (direct ident): {fullName} → {fieldName}"
        newCtx := { newCtx with
          globalArrays := newCtx.globalArrays.push fieldName,
          arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName,
          arrayTypes := newCtx.arrayTypes.insert (fieldName.toString) dtype
        }
      -- Extract field name from args.field
      else if let `($argsId:ident.$field:ident) := rhs then
        if argsId.getId == `args then
          let fieldName := field.getId
          -- dbg_trace s!"✓ Tracked global array (projection): {fieldName}"
          newCtx := { newCtx with
            globalArrays := newCtx.globalArrays.push fieldName,
            arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName,
            arrayTypes := newCtx.arrayTypes.insert (fieldName.toString) dtype
          }
      -- else
        -- dbg_trace s!"✗ RHS did not match expected patterns"

    -- PATTERN: let x : SharedArray T := ⟨args.field⟩
    | `(doElem| let $id:ident : SharedArray $elemTy:ident := ⟨$rhs:term⟩) =>
      let varName := id.getId

      -- Parse element type
      let dtype : DType :=
        match elemTy.getId.toString with
        | "Int" => DType.int
        | "Nat" => DType.nat
        | "Float" => DType.float
        | "Bool" => DType.bool
        | _ => DType.float  -- Default to float

      -- Check if rhs is just an ident (the field name directly)
      if rhs.raw.isIdent then
        let fullName := rhs.raw.getId
        -- Extract just the last component (e.g., `args.tile` → `tile`)
        let fieldName := fullName.components.getLast!
        newCtx := { newCtx with
          sharedArrays := newCtx.sharedArrays.push fieldName,
          arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName,
          arrayTypes := newCtx.arrayTypes.insert (fieldName.toString) dtype
        }
      -- Extract field name from args.field
      else if let `($argsId:ident.$field:ident) := rhs then
        if argsId.getId == `args then
          let fieldName := field.getId
          newCtx := { newCtx with
            sharedArrays := newCtx.sharedArrays.push fieldName,
            arrayMap := newCtx.arrayMap.insert (varName.toString) fieldName,
            arrayTypes := newCtx.arrayTypes.insert (fieldName.toString) dtype
          }

    -- PATTERN: let x := obj.field (check if obj is args for parameter extraction)
    -- | `(doElem| let $id:ident := $obj:ident.$field:ident) =>
    --   let varName := id.getId.toString
    --   dbg_trace s!"✓ Extracted content: {varName} : {obj.getId.toString} ; {field.getId.toString}"

    --   -- Check if this is args.field (parameter extraction)
    --   if obj.getId == `args then
    --     let fieldName := field.getId.toString
    --     let ty := inferCudaType fieldName
    --     dbg_trace s!"✓ Extracted param: {fieldName} : {ty}"
    --     newCtx := { newCtx with params := newCtx.params.push (fieldName, ty) }
    --     -- Don't generate assignment statement for args.* - it's a parameter, not a local variable
    --   else
    --     -- Not args.field, generate normal assignment
    --     let rhsExpr ← `(DExpr.var $(quote field.getId.toString))
    --     let dstmt ← `(DStmt.assign $(quote varName) $rhsExpr)
    --     dstmts := dstmts.push dstmt
        -- PATTERN: let val ← arr.get idx (array reads)
    | `(doElem| let $id:ident ← $rhs:term) =>
      let valName := id.getId.toString
      let rhsRaw := rhs.raw
      -- dbg_trace s!"✓ Extracted bind: {valName} : {rhs.raw}"
      -- ✅ skip any bind whose RHS contains getArgs anywhere
      -- ✅ skip any bind whose RHS is (possibly expanded) getArgs
      if containsGetArgs rhsRaw then
        pure ()
      else
        let mut handled := false


        if rhsRaw.getKind == ``Lean.Parser.Term.app then
          let fn := rhsRaw.getArg 0
          let argNode := rhsRaw.getArg 1

          -- helper to grab the single argument if it's wrapped in `null`
          let idxStx :=
            if argNode.getKind == `null && argNode.getNumArgs > 0 then
              argNode.getArg 0
            else
              argNode

          -- Case A: qualified ident like `data.get`
          if fn.isIdent then
            let comps := fn.getId.components
            if comps.length >= 2 && comps.getLast! == `get then
              let arrName := comps[comps.length - 2]!
              let actual := newCtx.arrayMap.getD arrName.toString arrName
              let idxD ← exprToDExpr idxStx
              let idxT : TSyntax `term := ⟨idxD⟩
              let dstmt ←
                `(DStmt.assign $(quote valName)
                   (DExpr.index (DExpr.var $(quote actual.toString)) $idxT))
              dstmts := dstmts.push dstmt
              handled := true

          -- Case B: projection like `data.get`
          if !handled && fn.getKind == ``Lean.Parser.Term.proj then
            let recv := fn.getArg 0
            let field := fn.getArg 2
            if recv.isIdent && field.isIdent && field.getId == `get then
              let arrName := recv.getId
              let actual := newCtx.arrayMap.getD arrName.toString arrName
              let idxD ← exprToDExpr idxStx
              let idxT : TSyntax `term := ⟨idxD⟩
              let dstmt ←
                `(DStmt.assign $(quote valName)
                   (DExpr.index (DExpr.var $(quote actual.toString)) $idxT))
              dstmts := dstmts.push dstmt
              handled := true

        if !handled then
          -- let valName := id.getId.toString
          -- let rhsRaw := rhs.raw

          -- let isGetArgs :=
          --   -- bare or qualified ident: getArgs / GpuDSL.getArgs / CLean.GPU.getArgs
          --   (rhsRaw.isIdent && endsWith rhsRaw.getId `getArgs) ||

          --   -- app form: (GpuDSL.getArgs) ?m_1 ...  (still ident at head)
          --   (rhsRaw.getKind == ``Lean.Parser.Term.app &&
          --     let fn := rhsRaw.getArg 0
          --     fn.isIdent && endsWith fn.getId `getArgs) ||

          --   -- projection form: something.getArgs
          --   (rhsRaw.getKind == ``Lean.Parser.Term.proj &&
          --     let field := rhsRaw.getArg 2
          --     field.isIdent && field.getId == `getArgs)

          -- -- let isGetArgs :=
          -- --   (rhsRaw.isIdent && rhsRaw.getId == `getArgs) ||
          -- --   (rhsRaw.getKind == ``Lean.Parser.Term.app &&
          -- --     let fn := rhsRaw.getArg 0
          -- --     fn.isIdent && fn.getId == `getArgs) ||
          -- --   (rhsRaw.getKind == ``Lean.Parser.Term.proj &&
          -- --     let field := rhsRaw.getArg 2
          -- --     field.isIdent && field.getId == `getArgs)

          -- if isGetArgs then
          --   pure ()
          -- else
          -- optional: keep a placeholder instead of silently skipping
          let dstmt ← `(DStmt.assign $(quote valName) (DExpr.var "unknown"))
          dstmts := dstmts.push dstmt

    -- PATTERN: let x := expr (scalar assignments - general case)
    | `(doElem| let $id:ident := $rhs:term) =>
      let varName := id.getId.toString
      let mut isParam := false

      let rhsRaw := rhs.raw
      -- dbg_trace s!"✓ Matched scalar assignment: {varName} := {rhsRaw.getKind}"

      -- Extract parameters from RHS syntax recursively
      let extractedParams := extractScalarParamsFromSyntax rhsRaw
      -- dbg_trace s!"  Found {extractedParams.size} params in RHS"
      for (paramName, paramType) in extractedParams do
        -- Only add if not already in params list
        if !(newCtx.params.any (fun (n, _) => n == paramName)) then
          -- dbg_trace s!"✓ Extracted param from expression: {paramName} : {paramType}"
          newCtx := { newCtx with params := newCtx.params.push (paramName, paramType) }

      -- Case 1: qualified identifier like `args.N`
      if rhsRaw.isIdent then
        let full := rhsRaw.getId
        let comps := full.components
        if comps.length == 2 && comps[0]! == `args then
          let fieldName := comps[1]!.toString
          let ty := inferCudaType fieldName
          -- dbg_trace s!"✓ Extracted param (qualified ident): {fieldName} : {ty}"
          if !(newCtx.params.any (fun (n, _) => n == fieldName)) then
            newCtx := { newCtx with params := newCtx.params.push (fieldName, ty) }
          isParam := true

      -- Case 2: projection syntax like `args.N` when parsed as proj
      if !isParam then
        try
          match rhs with
          | `($obj:ident.$field:ident) =>
            if obj.getId == `args then
              let fieldName := field.getId.toString
              let ty := inferCudaType fieldName
          -- dbg_trace s!"✓ Extracted param (proj): {fieldName} : {ty}"
              if !(newCtx.params.any (fun (n, _) => n == fieldName)) then
                newCtx := { newCtx with params := newCtx.params.push (fieldName, ty) }
              isParam := true
          | _ => pure ()
        catch _ =>
          pure ()

      if !isParam then
        let rhsExpr ← exprToDExpr rhs
        let rhsTSyntax : TSyntax `term := ⟨rhsExpr⟩
        let dstmt ← `(DStmt.assign $(quote varName) $rhsTSyntax)
        dstmts := dstmts.push dstmt

    -- PATTERN: let mut x := expr (mutable variable initialization)
    | `(doElem| let mut $id:ident := $rhs:term) =>
      let varName := id.getId.toString
      -- dbg_trace s!"✓ Matched let mut: {varName}"
      let rhsExpr ← exprToDExpr rhs
      let rhsTSyntax : TSyntax `term := ⟨rhsExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $rhsTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: let mut x : T := expr (mutable variable initialization with type annotation)
    | `(doElem| let mut $id:ident : $_ := $rhs:term) =>
      let varName := id.getId.toString
      -- dbg_trace s!"✓ Matched let mut with type: {varName}"
      let rhsExpr ← exprToDExpr rhs
      let rhsTSyntax : TSyntax `term := ⟨rhsExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $rhsTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: x := expr (mutable variable reassignment)
    | `(doElem| $id:ident := $rhs:term) =>
      let varName := id.getId.toString
      -- dbg_trace s!"✓ Matched reassignment: {varName}"
      let rhsExpr ← exprToDExpr rhs
      let rhsTSyntax : TSyntax `term := ⟨rhsExpr⟩
      let dstmt ← `(DStmt.assign $(quote varName) $rhsTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: barrier (synchronization)
    | `(doElem| barrier) =>
      let dstmt ← `(DStmt.barrier)
      dstmts := dstmts.push dstmt

    -- PATTERN: doNested - nested do block (check in default case below since no pattern match)
    -- Handled in default case below

    -- PATTERN: if-then-do (without else) - must come before if-then-else
    | `(doElem| if $cond:term then $thenBranch:term) =>
      -- Extract parameters from condition
      let condParams := extractScalarParamsFromSyntax cond.raw
      for (paramName, paramType) in condParams do
        if !(newCtx.params.any (fun (n, _) => n == paramName)) then
          -- dbg_trace s!"✓ Extracted param from if condition: {paramName} : {paramType}"
          newCtx := { newCtx with params := newCtx.params.push (paramName, paramType) }

      -- Extract condition
      let condDExpr ← exprToDExpr cond
      let condTSyntax : TSyntax `term := ⟨condDExpr⟩

      -- Recursively extract then branch
      let thenStmt ← if thenBranch.raw.getKind == ``Lean.Parser.Term.do then
        -- Then branch is do-notation, recursively extract
        let thenDoSeq := thenBranch.raw[1]
        let thenItems := if thenDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                            thenDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
          if thenDoSeq.getNumArgs > 0 then thenDoSeq[0].getArgs else #[]
        else #[]
        let (thenStmts, thenCtx) ← extractDoItems thenItems newCtx
        newCtx := thenCtx
        -- Build sequence of then statements
        let mut thenBody ← `(DStmt.skip)
        for stmt in thenStmts.reverse do
          let stmtTSyntax : TSyntax `term := ⟨stmt⟩
          thenBody ← `(DStmt.seq $stmtTSyntax $thenBody)
        pure thenBody
      else
        -- Single statement - check if it's arr.set
        if thenBranch.raw.getKind == ``Lean.Parser.Term.app then
          let fn := thenBranch.raw.getArg 0
          if fn.isIdent then
            let fullName := fn.getId
            let components := fullName.components
            if components.length >= 2 && components.getLast! == `set then
              -- This is arr.set pattern
              let arrName := components[components.length - 2]!
              let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
              let args := thenBranch.raw.getArg 1
              if args.getKind == `null && args.getNumArgs >= 2 then
                let idx := args.getArg 0
                let val := args.getArg 1
                let idxDExpr ← exprToDExpr idx
                let valDExpr ← exprToDExpr val
                let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                   $idxTSyntax $valTSyntax)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
          else
            `(DStmt.skip)
        else
          `(DStmt.skip)

      -- Else branch is skip for if-then without else
      let elseStmt ← `(DStmt.skip)

      let thenStmtTSyntax : TSyntax `term := ⟨thenStmt⟩
      let elseStmtTSyntax : TSyntax `term := ⟨elseStmt⟩
      let dstmt ← `(DStmt.ite $condTSyntax $thenStmtTSyntax $elseStmtTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: if-then-else
    | `(doElem| if $cond:term then $thenBranch:term else $elseBranch:term) =>
      -- Extract parameters from condition
      let condParams := extractScalarParamsFromSyntax cond.raw
      for (paramName, paramType) in condParams do
        if !(newCtx.params.any (fun (n, _) => n == paramName)) then
          -- dbg_trace s!"✓ Extracted param from if-else condition: {paramName} : {paramType}"
          newCtx := { newCtx with params := newCtx.params.push (paramName, paramType) }

      -- Extract condition
      let condDExpr ← exprToDExpr cond
      let condTSyntax : TSyntax `term := ⟨condDExpr⟩

      -- Recursively extract then branch
      let thenStmt ← if thenBranch.raw.getKind == ``Lean.Parser.Term.do then
        -- Then branch is do-notation, recursively extract
        let thenDoSeq := thenBranch.raw[1]
        let thenItems := if thenDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                            thenDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
          if thenDoSeq.getNumArgs > 0 then thenDoSeq[0].getArgs else #[]
        else #[]
        let (thenStmts, thenCtx) ← extractDoItems thenItems newCtx
        newCtx := thenCtx
        -- Build sequence of then statements
        let mut thenBody ← `(DStmt.skip)
        for stmt in thenStmts.reverse do
          let stmtTSyntax : TSyntax `term := ⟨stmt⟩
          thenBody ← `(DStmt.seq $stmtTSyntax $thenBody)
        pure thenBody
      else
        -- Single statement - check if it's arr.set
        if thenBranch.raw.getKind == ``Lean.Parser.Term.app then
          let fn := thenBranch.raw.getArg 0
          if fn.isIdent then
            let fullName := fn.getId
            let components := fullName.components
            if components.length >= 2 && components.getLast! == `set then
              -- This is arr.set pattern
              let arrName := components[components.length - 2]!
              let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
              let args := thenBranch.raw.getArg 1
              if args.getKind == `null && args.getNumArgs >= 2 then
                let idx := args.getArg 0
                let val := args.getArg 1
                let idxDExpr ← exprToDExpr idx
                let valDExpr ← exprToDExpr val
                let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                  $idxTSyntax $valTSyntax)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
          else
            `(DStmt.skip)
        else
          `(DStmt.skip)

      -- Recursively extract else branch
      let elseStmt ← if elseBranch.raw.getKind == ``Lean.Parser.Term.do then
        -- Else branch is do-notation, recursively extract
        let elseDoSeq := elseBranch.raw[1]
        let elseItems := if elseDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                            elseDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
          if elseDoSeq.getNumArgs > 0 then elseDoSeq[0].getArgs else #[]
        else #[]
        let (elseStmts, elseCtx) ← extractDoItems elseItems newCtx
        newCtx := elseCtx
        -- Build sequence of else statements
        let mut elseBody ← `(DStmt.skip)
        for stmt in elseStmts.reverse do
          let stmtTSyntax : TSyntax `term := ⟨stmt⟩
          elseBody ← `(DStmt.seq $stmtTSyntax $elseBody)
        pure elseBody
      else
        -- Single statement - check if it's arr.set
        if elseBranch.raw.getKind == ``Lean.Parser.Term.app then
          let fn := elseBranch.raw.getArg 0
          if fn.isIdent then
            let fullName := fn.getId
            let components := fullName.components
            if components.length >= 2 && components.getLast! == `set then
              -- This is arr.set pattern
              let arrName := components[components.length - 2]!
              let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
              let args := elseBranch.raw.getArg 1
              if args.getKind == `null && args.getNumArgs >= 2 then
                let idx := args.getArg 0
                let val := args.getArg 1
                let idxDExpr ← exprToDExpr idx
                let valDExpr ← exprToDExpr val
                let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                  $idxTSyntax $valTSyntax)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
          else
            `(DStmt.skip)
        else
          `(DStmt.skip)

      let thenStmtTSyntax : TSyntax `term := ⟨thenStmt⟩
      let elseStmtTSyntax : TSyntax `term := ⟨elseStmt⟩
      let dstmt ← `(DStmt.ite $condTSyntax $thenStmtTSyntax $elseStmtTSyntax)
      dstmts := dstmts.push dstmt

    -- PATTERN: for loop (for i in [lo:hi] do body)
    | `(doElem| for $id:ident in [$lo:term : $hi:term] do $body:term) =>
      let loopVar := id.getId.toString
      -- Extract loop bounds
      let loDExpr ← exprToDExpr lo
      let hiDExpr ← exprToDExpr hi
      let loTSyntax : TSyntax `term := ⟨loDExpr⟩
      let hiTSyntax : TSyntax `term := ⟨hiDExpr⟩

      -- Recursively extract loop body
      let bodyStmt ← if body.raw.getKind == ``Lean.Parser.Term.do then
        -- Body is do-notation, recursively extract
        let bodyDoSeq := body.raw[1]
        let bodyItems := if bodyDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                            bodyDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
          if bodyDoSeq.getNumArgs > 0 then bodyDoSeq[0].getArgs else #[]
        else #[]
        let (bodyStmts, bodyCtx) ← extractDoItems bodyItems newCtx
        newCtx := bodyCtx
        -- Build sequence of body statements
        let mut bodySeq ← `(DStmt.skip)
        for stmt in bodyStmts.reverse do
          let stmtTSyntax : TSyntax `term := ⟨stmt⟩
          bodySeq ← `(DStmt.seq $stmtTSyntax $bodySeq)
        pure bodySeq
      else
        -- Single statement - check if it's arr.set
        if body.raw.getKind == ``Lean.Parser.Term.app then
          let fn := body.raw.getArg 0
          if fn.isIdent then
            let fullName := fn.getId
            let components := fullName.components
            if components.length >= 2 && components.getLast! == `set then
              -- This is arr.set pattern
              let arrName := components[components.length - 2]!
              let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
              let args := body.raw.getArg 1
              if args.getKind == `null && args.getNumArgs >= 2 then
                let idx := args.getArg 0
                let val := args.getArg 1
                let idxDExpr ← exprToDExpr idx
                let valDExpr ← exprToDExpr val
                let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                   $idxTSyntax $valTSyntax)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
          else
            `(DStmt.skip)
        else
          `(DStmt.skip)

      let bodyStmtTSyntax : TSyntax `term := ⟨bodyStmt⟩
      let dstmt ← `(DStmt.for $(quote loopVar) $loTSyntax $hiTSyntax $bodyStmtTSyntax)
      dstmts := dstmts.push dstmt

    -- Check arr.set pattern separately (can't use quotation pattern for doExpr)
    | _ =>
      -- let dstmt ← `(DStmt.skip)
      -- dstmts := dstmts.push dstmt
      -- dbg_trace s!"✗ No pattern matched, checking doIf and arr.set in default case"
      -- Handle doIf (if-then and if-then-else)
      if doElem.getKind == ``Lean.Parser.Term.doIf then
          -- dbg_trace s!"✓ Matched doIf pattern"
        -- doIf structure: doIf := "if" term "then" term ("else" term)?
        -- Args: [0]=if keyword, [1]=cond, [2]=then keyword, [3]=then branch, [4]=(else keyword), [5]=(else branch)
        let cond := doElem.getArg 1
        let thenBranch := doElem.getArg 3
        let hasElse := doElem.getNumArgs > 5

        -- dbg_trace s!"doIf: cond kind={cond.getKind}, numArgs={cond.getNumArgs}, cond={cond}"
        -- Extract condition (unwrap doIfProp if needed)
        let actualCond := if cond.getKind == ``Lean.Parser.Term.doIfProp then
          cond.getArg 1  -- The actual condition is at index 1
        else
          cond

        -- Extract parameters from condition
        let condParams := extractScalarParamsFromSyntax actualCond
        for (paramName, paramType) in condParams do
          if !(newCtx.params.any (fun (n, _) => n == paramName)) then
          -- dbg_trace s!"✓ Extracted param from doIf condition: {paramName} : {paramType}"
            newCtx := { newCtx with params := newCtx.params.push (paramName, paramType) }

        let condDExpr ← exprToDExpr actualCond
        let condTSyntax : TSyntax `term := ⟨condDExpr⟩

        -- Extract then branch
        -- dbg_trace s!"doIf: thenBranch kind={thenBranch.getKind}, numArgs={thenBranch.getNumArgs}"
        let thenStmt ← if thenBranch.getKind == ``Lean.Parser.Term.doSeqIndent ||
                           thenBranch.getKind == ``Lean.Parser.Term.doSeqBracketed then
          -- Then branch is directly a doSeqIndent/doSeqBracketed (implicit do in if-then)
          let thenItems := if thenBranch.getNumArgs > 0 then thenBranch[0].getArgs else #[]
          -- dbg_trace s!"  thenItems.size={thenItems.size}"
          let (thenStmts, thenCtx) ← extractDoItems thenItems newCtx
          newCtx := thenCtx
          -- dbg_trace s!"  thenStmts.size={thenStmts.size}"
          -- Build sequence of then statements
          let mut thenBody ← `(DStmt.skip)
          for stmt in thenStmts.reverse do
            let stmtTSyntax : TSyntax `term := ⟨stmt⟩
            thenBody ← `(DStmt.seq $stmtTSyntax $thenBody)
          pure thenBody
        else if thenBranch.getKind == ``Lean.Parser.Term.do then
          -- Then branch has explicit do keyword, recursively extract
          let thenDoSeq := thenBranch[1]
          -- dbg_trace s!"  thenDoSeq kind={thenDoSeq.getKind}, numArgs={thenDoSeq.getNumArgs}"
          let thenItems := if thenDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                              thenDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
            if thenDoSeq.getNumArgs > 0 then thenDoSeq[0].getArgs else #[]
          else #[]
          let (thenStmts, thenCtx) ← extractDoItems thenItems newCtx
          newCtx := thenCtx
          -- Build sequence of then statements
          let mut thenBody ← `(DStmt.skip)
          for stmt in thenStmts.reverse do
            let stmtTSyntax : TSyntax `term := ⟨stmt⟩
            thenBody ← `(DStmt.seq $stmtTSyntax $thenBody)
          pure thenBody
        else
          -- Single statement - check if it's arr.set
          if thenBranch.getKind == ``Lean.Parser.Term.app then
            let fn := thenBranch.getArg 0
            if fn.isIdent then
              let fullName := fn.getId
              let components := fullName.components
              if components.length >= 2 && components.getLast! == `set then
                -- This is arr.set pattern
                let arrName := components[components.length - 2]!
                let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
                let args := thenBranch.getArg 1
                if args.getKind == `null && args.getNumArgs >= 2 then
                  let idx := args.getArg 0
                  let val := args.getArg 1
                  let idxDExpr ← exprToDExpr idx
                  let valDExpr ← exprToDExpr val
                  let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                  let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                  `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                    $idxTSyntax $valTSyntax)
                else
                  `(DStmt.skip)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
          else
            `(DStmt.skip)

        -- Extract else branch if present
        let elseStmt ← if hasElse then
          let elseBranch := doElem.getArg 5
          if elseBranch.getKind == ``Lean.Parser.Term.doSeqIndent ||
            elseBranch.getKind == ``Lean.Parser.Term.doSeqBracketed then
            -- Else branch is directly a doSeqIndent/doSeqBracketed
            let elseItems := if elseBranch.getNumArgs > 0 then elseBranch[0].getArgs else #[]
            let (elseStmts, elseCtx) ← extractDoItems elseItems newCtx
            newCtx := elseCtx
            -- Build sequence of else statements
            let mut elseBody ← `(DStmt.skip)
            for stmt in elseStmts.reverse do
              let stmtTSyntax : TSyntax `term := ⟨stmt⟩
              elseBody ← `(DStmt.seq $stmtTSyntax $elseBody)
            pure elseBody
          else if elseBranch.getKind == ``Lean.Parser.Term.do then
            -- Else branch has explicit do keyword, recursively extract
            let elseDoSeq := elseBranch[1]
            let elseItems := if elseDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                                elseDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
              if elseDoSeq.getNumArgs > 0 then elseDoSeq[0].getArgs else #[]
            else #[]
            let (elseStmts, elseCtx) ← extractDoItems elseItems newCtx
            newCtx := elseCtx
            -- Build sequence of else statements
            let mut elseBody ← `(DStmt.skip)
            for stmt in elseStmts.reverse do
              let stmtTSyntax : TSyntax `term := ⟨stmt⟩
              elseBody ← `(DStmt.seq $stmtTSyntax $elseBody)
            pure elseBody
          else
            -- Single statement - check if it's arr.set
            if elseBranch.getKind == ``Lean.Parser.Term.app then
              let fn := elseBranch.getArg 0
              if fn.isIdent then
                let fullName := fn.getId
                let components := fullName.components
                if components.length >= 2 && components.getLast! == `set then
                  -- This is arr.set pattern
                  let arrName := components[components.length - 2]!
                  let actualArrayName := newCtx.arrayMap.getD (arrName.toString) arrName
                  let args := elseBranch.getArg 1
                  if args.getKind == `null && args.getNumArgs >= 2 then
                    let idx := args.getArg 0
                    let val := args.getArg 1
                    let idxDExpr ← exprToDExpr idx
                    let valDExpr ← exprToDExpr val
                    let idxTSyntax : TSyntax `term := ⟨idxDExpr⟩
                    let valTSyntax : TSyntax `term := ⟨valDExpr⟩
                    `(DStmt.store (DExpr.var $(quote actualArrayName.toString))
                      $idxTSyntax $valTSyntax)
                  else
                    `(DStmt.skip)
                else
                  `(DStmt.skip)
              else
                `(DStmt.skip)
            else
              `(DStmt.skip)
        else
          -- No else branch, use skip
          `(DStmt.skip)

        let thenStmtTSyntax : TSyntax `term := ⟨thenStmt⟩
        let elseStmtTSyntax : TSyntax `term := ⟨elseStmt⟩
        let dstmt ← `(DStmt.ite $condTSyntax $thenStmtTSyntax $elseStmtTSyntax)
        dstmts := dstmts.push dstmt
      -- Handle doNested - nested do blocks
      else if doElem.getKind == ``Lean.Parser.Term.doNested then
        -- doNested wraps a do-block: doNested := "do" (doSeqIndent | doSeqBracketed)
        -- Extract the nested do sequence
        let nestedDoSeq := doElem.getArg 1  -- The do sequence is at index 1
        -- dbg_trace s!"  doNested: nestedDoSeq kind={nestedDoSeq.getKind}, numArgs={nestedDoSeq.getNumArgs}"
        let nestedItems := if nestedDoSeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                              nestedDoSeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
          if nestedDoSeq.getNumArgs > 0 then nestedDoSeq[0].getArgs else #[]
        else #[]
        -- dbg_trace s!"  doNested: nestedItems.size={nestedItems.size}"
        -- Recursively extract nested statements
        let (nestedStmts, nestedCtx) ← extractDoItems nestedItems newCtx
        newCtx := nestedCtx
        -- Add all nested statements to our list
        dstmts := dstmts ++ nestedStmts
      -- Handle doFor - for loops
      else if doElem.getKind == ``Lean.Parser.Term.doFor then
        -- doFor structure: [0]="for", [1]=doForDecl list, [2]="do", [3]=body (doSeqIndent)
        -- doForDecl contains: [] varName "in" range
        let forDeclList := doElem.getArg 1
        if forDeclList.getNumArgs > 0 then
          let forDecl := forDeclList.getArg 0
          -- forDecl structure: [0]=[] (empty), [1]=varName, [2]="in", [3]=range
          let loopVarSyntax := forDecl.getArg 1
          let loopVar := if loopVarSyntax.isIdent then loopVarSyntax.getId.toString else "i"
          let rangeSyntax := forDecl.getArg 3

          -- Extract range bounds from [lo:hi] syntax (Std.Range.«term[_:_]»)
          -- Range structure: [0]="[", [1]=lo, [2]=":", [3]=hi, [4]="]"
          let lo := if rangeSyntax.getNumArgs > 1 then rangeSyntax.getArg 1 else rangeSyntax
          let hi := if rangeSyntax.getNumArgs > 3 then rangeSyntax.getArg 3 else rangeSyntax

          let loDExpr ← exprToDExpr lo
          let hiDExpr ← exprToDExpr hi
          let loTSyntax : TSyntax `term := ⟨loDExpr⟩
          let hiTSyntax : TSyntax `term := ⟨hiDExpr⟩

          -- Extract loop body
          let bodySeq := doElem.getArg 3
          let bodyItems := if bodySeq.getKind == ``Lean.Parser.Term.doSeqIndent ||
                              bodySeq.getKind == ``Lean.Parser.Term.doSeqBracketed then
            if bodySeq.getNumArgs > 0 then bodySeq[0].getArgs else #[]
          else #[]

          let (bodyStmts, bodyCtx) ← extractDoItems bodyItems newCtx
          newCtx := bodyCtx

          -- Build sequence of body statements
          let mut bodyStmt ← `(DStmt.skip)
          for stmt in bodyStmts.reverse do
            let stmtTSyntax : TSyntax `term := ⟨stmt⟩
            bodyStmt ← `(DStmt.seq $stmtTSyntax $bodyStmt)

          let bodyStmtTSyntax : TSyntax `term := ⟨bodyStmt⟩
          let dstmt ← `(DStmt.for $(quote loopVar) $loTSyntax $hiTSyntax $bodyStmtTSyntax)
          dstmts := dstmts.push dstmt

      else if doElem.getKind == ``Lean.Parser.Term.doExpr then
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
    -- Get the tracked element type, defaulting to Float if not found
    let elemType := finalCtx.arrayTypes.getD (arrName.toString) DType.float
    let elemTypeSyntax : TSyntax `term ←
      match elemType with
      | DType.int => `(DType.int)
      | DType.nat => `(DType.nat)
      | DType.float => `(DType.float)
      | DType.bool => `(DType.bool)
      | _ => `(DType.float)
    let s ← `({ name := $(quote arrName.toString), ty := DType.array $elemTypeSyntax, space := MemorySpace.global })
    globalArraySyntax := globalArraySyntax.push s

  let mut sharedArraySyntax : Array (TSyntax `term) := #[]
  for arrName in finalCtx.sharedArrays do
    -- Get the tracked element type, defaulting to Float if not found
    let elemType := finalCtx.arrayTypes.getD (arrName.toString) DType.float
    let elemTypeSyntax : TSyntax `term ←
      match elemType with
      | DType.int => `(DType.int)
      | DType.nat => `(DType.nat)
      | DType.float => `(DType.float)
      | DType.bool => `(DType.bool)
      | _ => `(DType.float)
    let s ← `({ name := $(quote arrName.toString), ty := DType.array $elemTypeSyntax, space := MemorySpace.shared })
    sharedArraySyntax := sharedArraySyntax.push s

  -- Build the body by sequencing statements
  let mut bodyExpr ← `(DStmt.skip)
  for stmt in stmts.reverse do
    let stmtTSyntax : TSyntax `term := ⟨stmt⟩
    bodyExpr ← `(DStmt.seq $stmtTSyntax $bodyExpr)

  -- Extract parameters from context and build params list
  let mut paramsSyntax : Array (TSyntax `term) := #[]
  -- Get unique params (avoid duplicates from multiple assignments)
  let uniqueParams := finalCtx.params.toList.eraseDups
  for (paramName, paramType) in uniqueParams do
    let typeIdent := mkIdent (Name.mkSimple paramType)
    let dtypeTerm : TSyntax `term ←
      match paramType with
      | "int"   => `(DType.int)
      | "float" => `(DType.float)
      | "bool"  => `(DType.bool)
      | _       => `(DType.float)  -- fallback

    let paramVarDecl ←
      `({ name := $(quote paramName), ty := $dtypeTerm : VarDecl })

    paramsSyntax := paramsSyntax.push paramVarDecl

  -- Generate DeviceIR Kernel definition
  let irName := Lean.mkIdent (name.getId.appendAfter "IR")
  let nameQuote := quote name.getId.toString

  let kernelIRDefStx ← `(
    def $irName : DeviceIR.Kernel := {
      name := $nameQuote
      params := [$paramsSyntax,*]
      locals := []
      globalArrays := [$globalArraySyntax,*]
      sharedArrays := [$sharedArraySyntax,*]
      body := $bodyExpr
    }
  )

  -- Return both definitions
  `($kernelDefStx:command
    $kernelIRDefStx:command)

end CLean.DeviceMacro


-- kernelArgs IncrementArgs(N: Nat)
--   global[data: Array Float]

-- device_kernel incrementKernel : KernelM IncrementArgs Unit := do
--   let args ← getArgs
--   let N := args.N
--   let data : GlobalArray Float := ⟨args.data⟩

--   let i ← globalIdxX
--   if i < N then do
--     let val ← data.get i
--     data.set i (val + 1.0)

-- #eval incrementKernelIR
