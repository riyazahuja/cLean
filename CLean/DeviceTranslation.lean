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

end DeviceTranslation
