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

/-! ## Binary Operator Registrations

Register standard binary operators for device translation.
Operators are registered by their syntax kind (e.g., `«term_+_»`).
-/

initialize do
  -- Arithmetic operators
  registerBinOp ⟨`«term_+_», .add, []⟩
  registerBinOp ⟨`«term_-_», .sub, []⟩
  registerBinOp ⟨`«term_*_», .mul, []⟩
  registerBinOp ⟨`«term_/_», .div, []⟩
  registerBinOp ⟨`«term_%_», .mod, []⟩

  -- Comparison operators
  registerBinOp ⟨`«term_<_», .lt, []⟩
  registerBinOp ⟨`«term_<=_», .le, []⟩
  registerBinOp ⟨`«term_>_», .gt, []⟩
  registerBinOp ⟨`«term_>=_», .ge, []⟩
  registerBinOp ⟨`«term_==_», .eq, []⟩
  registerBinOp ⟨`«term_!=_», .ne, []⟩

  -- Logical operators
  registerBinOp ⟨`«term_&&_», .and, []⟩
  registerBinOp ⟨`«term_||_», .or, []⟩

/-! ## Unary Operator Registrations -/

initialize do
  -- Unary operators (if we add them later)
  registerUnOp ⟨`«term_-_», .neg, []⟩  -- Unary negation (same syntax as binary -, context dependent)

/-! ## GPU Intrinsic Registrations

Register GPU-specific functions like globalIdxX/Y/Z and barrier.
-/

initialize do
  -- globalIdxX → blockIdx.x * blockDim.x + threadIdx.x
  registerFn {
    name := `GpuDSL.globalIdxX,
    rule := .direct fun () => do
      `(DExpr.binop BinOp.add
         (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
         (DExpr.threadIdx Dim.x))
  }

  -- globalIdxY → blockIdx.y * blockDim.y + threadIdx.y
  registerFn {
    name := `GpuDSL.globalIdxY,
    rule := .direct fun () => do
      `(DExpr.binop BinOp.add
         (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.y) (DExpr.blockDim Dim.y))
         (DExpr.threadIdx Dim.y))
  }

  -- globalIdxZ → blockIdx.z * blockDim.z + threadIdx.z
  registerFn {
    name := `GpuDSL.globalIdxZ,
    rule := .direct fun () => do
      `(DExpr.binop BinOp.add
         (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.z) (DExpr.blockDim Dim.z))
         (DExpr.threadIdx Dim.z))
  }

  -- barrier → DStmt.barrier
  registerFn {
    name := `GpuDSL.barrier,
    rule := .stmt fun _ => do
      `(DStmt.barrier)
  }

end DeviceInstances
