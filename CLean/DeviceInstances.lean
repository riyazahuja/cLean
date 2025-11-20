/-
  Standard Library Instances

  ToCudaType instances for common Lean types (Nat, Int, Float, Bool, Array)
-/

import CLean.DeviceTranslation

open DeviceIR DeviceTranslation

namespace DeviceInstances

-- Instance for Nat
instance : ToCudaType Nat where
  deviceType := .nat
  encode n := .int (Int.ofNat n)
  decode v := match v with
    | .int i => if i ≥ 0 then some i.toNat else none
    | _ => none

-- Instance for Int
instance : ToCudaType Int where
  deviceType := .int
  encode i := .int i
  decode v := match v with
    | .int i => some i
    | _ => none

-- Instance for Float
instance : ToCudaType Float where
  deviceType := .float
  encode f := .float f
  decode v := match v with
    | .float f => some f
    | _ => none

-- Instance for Bool
instance : ToCudaType Bool where
  deviceType := .bool
  encode b := .bool b
  decode v := match v with
    | .bool b => some b
    | _ => none

-- Instance for Array (when element type is translatable)
instance [ToCudaType α] : ToCudaType (Array α) where
  deviceType := .array (ToCudaType.deviceType (α := α))
  encode arr := .array (arr.map ToCudaType.encode)
  decode v := match v with
    | .array vals =>
        let decoded := vals.filterMap ToCudaType.decode
        if decoded.size = vals.size then
          some decoded
        else
          none
    | _ => none

-- Instance for tuples
instance [ToCudaType α] [ToCudaType β] : ToCudaType (α × β) where
  deviceType := .tuple [
    ToCudaType.deviceType (α := α),
    ToCudaType.deviceType (α := β)
  ]
  encode pair := .tuple [ToCudaType.encode pair.1, ToCudaType.encode pair.2]
  decode v := match v with
    | .tuple [va, vb] => do
          let a ← ToCudaType.decode va
          let b ← ToCudaType.decode vb
          return (a, b)
    | _ => none

end DeviceInstances
