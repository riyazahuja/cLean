/-
  Device Translation System

  Typeclass-based framework for translating Lean types and functions
  to DeviceIR for GPU execution and verification.
-/

import Lean
import CLean.DeviceIR

open Lean Meta Elab
open DeviceIR

namespace DeviceTranslation

-- Runtime values on device (for execution semantics)
inductive DeviceValue
  | int    : Int → DeviceValue
  | float  : Float → DeviceValue
  | bool   : Bool → DeviceValue
  | array  : Array DeviceValue → DeviceValue
  | tuple  : List DeviceValue → DeviceValue
  | struct : String → List (String × DeviceValue) → DeviceValue
  deriving Repr, Inhabited

-- 1. Type translation typeclass
--    Marks Lean types that have a device representation
class ToCudaType (α : Type) where
  /-- The device type representation -/
  deviceType : DType

  /-- Encode Lean value to device value (for CPU execution) -/
  encode : α → DeviceValue

  /-- Decode device value back to Lean (partial, may fail) -/
  decode : DeviceValue → Option α

-- 2. Expression translation (for compile-time reflection)
--    This will be used by the meta-level translation tactic
structure ExprTranslation where
  expr : DExpr
  ty : DType

-- 3. Function translation typeclass
--    Marks functions that can be compiled to device code
class ToCudaFn {α β : Type} (f : α → β) where
  /-- Generate device IR for this function -/
  translateFn : MetaM (Option DStmt)

  -- Note: We'll add correctness spec later
  -- spec : ∀ x, evalDevice (translateFn f) x = f x

-- Helper: Get device type for a Lean type (if it has ToCudaType instance)
unsafe def getDeviceType? (ty : Lean.Expr) : Lean.MetaM (Option DType) := do
  try
    let inst ← Lean.Meta.synthInstance (← Lean.Meta.mkAppM ``ToCudaType #[ty])
    let dtypeExpr ← Lean.Meta.mkProjection inst `deviceType
    -- Evaluate to get the actual DType value
    let dtype ← Lean.Meta.evalExpr DType (Lean.mkConst ``DType) dtypeExpr
    return some dtype
  catch _ =>
    return none

-- Marker attribute for device-eligible definitions
initialize deviceEligibleAttr : Lean.TagAttribute ←
  Lean.registerTagAttribute `device_eligible
    "Mark a definition as eligible for device translation"

-- Environment extension to store translated device functions
structure DeviceFnEntry where
  name : Lean.Name
  ir : DStmt
  deriving Inhabited

initialize deviceFnExt : Lean.SimplePersistentEnvExtension DeviceFnEntry (Lean.NameMap DStmt) ←
  Lean.registerSimplePersistentEnvExtension {
    name := `deviceFunctions
    addEntryFn := fun map entry => map.insert entry.name entry.ir
    addImportedFn := fun arrays =>
      arrays.foldl (init := {}) fun acc entries =>
        entries.foldl (init := acc) fun map entry =>
          map.insert entry.name entry.ir
  }

-- Lookup translated IR for a function
def getDeviceFn? (name : Lean.Name) : Lean.CoreM (Option DStmt) := do
  let env ← Lean.getEnv
  return (deviceFnExt.getState env).find? name

/-! ## Translation Registry System

This section provides an extensible registry for operator and function translations.
Instead of hard-coding patterns in the macro, translations are registered via
`initialize` blocks and looked up at macro expansion time.
-/

-- Binary operator translation rule
structure BinOpTranslation where
  /-- The syntax kind for the operator (e.g., `«term_+_»`, `«term_&&_»`)
      Or the function name for function-style ops (e.g., `HAdd.hAdd`) -/
  syntaxKind : Name
  /-- The corresponding DeviceIR binary operator -/
  deviceOp : BinOp
  /-- Types this operator is valid for (e.g., [``Nat, ``Int, ``Float]) -/
  validTypes : List Name := []  -- Empty list means valid for all types
  deriving Inhabited

-- Unary operator translation rule
structure UnOpTranslation where
  /-- The syntax kind for the operator (e.g., `«term_-_»`)
      Or the function name for function-style ops (e.g., `Neg.neg`) -/
  syntaxKind : Name
  /-- The corresponding DeviceIR unary operator -/
  deviceOp : UnOp
  /-- Types this operator is valid for -/
  validTypes : List Name := []
  deriving Inhabited

-- Function translation rule variants
inductive FnTranslationRule where
  /-- Direct translation: generates a DExpr with no arguments
      Example: globalIdxX → DExpr.binop ... -/
  | direct : (Unit → MacroM Syntax) → FnTranslationRule

  /-- Expression with arguments: takes function arguments, returns DExpr
      Example: arr.get idx → DExpr.index arr idx -/
  | exprWithArgs : (Array Syntax → MacroM Syntax) → FnTranslationRule

  /-- Statement-level: generates a DStmt
      Example: arr.set idx val → DStmt.store arr idx val -/
  | stmt : (Array Syntax → MacroM Syntax) → FnTranslationRule

  /-- Custom: full control over translation, can inspect syntax
      Example: .toNat?.getD pattern matching -/
  | custom : (Syntax → MacroM (Option Syntax)) → FnTranslationRule

-- Function translation entry
structure FnTranslationEntry where
  /-- The function name -/
  name : Name
  /-- The translation rule -/
  rule : FnTranslationRule
  /-- Priority for resolving conflicts (higher = preferred) -/
  priority : Nat := 10

-- Inhabited instance for FnTranslationRule (default case)
instance : Inhabited FnTranslationRule where
  default := .direct fun () => `(DExpr.var "unknown")

-- Inhabited instance for FnTranslationEntry
instance : Inhabited FnTranslationEntry where
  default := ⟨`default, .direct fun () => `(DExpr.var "unknown"), 0⟩

/-! ## Global Registries

These registries are populated by `initialize` blocks across the codebase.
They provide O(1) lookup during macro expansion.
-/

-- Registry for binary operators
initialize binOpRegistry : IO.Ref (Std.HashMap Name BinOpTranslation) ←
  IO.mkRef {}

-- Registry for unary operators
initialize unOpRegistry : IO.Ref (Std.HashMap Name UnOpTranslation) ←
  IO.mkRef {}

-- Registry for function translations
initialize fnRegistry : IO.Ref (Std.HashMap Name FnTranslationEntry) ←
  IO.mkRef {}

/-! ## Registration API

These functions are used to register translations, typically in `initialize` blocks.
-/

-- Register a binary operator translation
def registerBinOp (trans : BinOpTranslation) : IO Unit := do
  binOpRegistry.modify (·.insert trans.syntaxKind trans)

-- Register a unary operator translation
def registerUnOp (trans : UnOpTranslation) : IO Unit := do
  unOpRegistry.modify (·.insert trans.syntaxKind trans)

-- Register a function translation
def registerFn (entry : FnTranslationEntry) : IO Unit := do
  fnRegistry.modify fun registry =>
    -- If there's a conflict, keep the higher priority entry
    match Std.HashMap.get? registry entry.name with
    | some existing =>
      if entry.priority > existing.priority then
        Std.HashMap.insert registry entry.name entry
      else
        registry
    | none => Std.HashMap.insert registry entry.name entry

/-! ## Lookup API

These functions are used during macro expansion to find registered translations.
-/

-- Lookup a binary operator translation
def lookupBinOp? (name : Name) : IO (Option BinOpTranslation) := do
  let registry ← binOpRegistry.get
  return Std.HashMap.get? registry name

-- Lookup a unary operator translation
def lookupUnOp? (name : Name) : IO (Option UnOpTranslation) := do
  let registry ← unOpRegistry.get
  return Std.HashMap.get? registry name

-- Lookup a function translation
def lookupFn? (name : Name) : IO (Option FnTranslationEntry) := do
  let registry ← fnRegistry.get
  return Std.HashMap.get? registry name

-- Debug: Get all registered binary operators
def getAllBinOps : IO (List Name) := do
  let registry ← binOpRegistry.get
  return registry.toList.map (·.1)

-- Debug: Get all registered functions
def getAllFns : IO (List Name) := do
  let registry ← fnRegistry.get
  return registry.toList.map (·.1)

end DeviceTranslation
