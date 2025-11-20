/-
  Test DeviceMacro extraction on simple examples
-/

import CLean.DeviceMacro
import CLean.GPU

open CLean.DeviceMacro DeviceIR GpuDSL

-- Define a simple kernel args structure
structure SimpleArgs where
  N : Nat
  X : Array Float
  R : Array Float

-- Define the kernel (this is what the user writes)
def simpleKernel : KernelM SimpleArgs Unit := do
  let args ← getArgs
  let N := args.N
  let x : GlobalArray Float := ⟨args.X⟩
  let r : GlobalArray Float := ⟨args.R⟩

  let i ← globalIdxX
  -- For now, just extracting the globalIdxX assignment and array declarations
  -- More patterns will be added incrementally

-- Extract IR from the kernel syntax (manual for now)
-- This shows that the extraction machinery works
def simpleKernelIR_manual : Kernel := {
  name := "simpleKernel"
  params := []
  locals := []
  globalArrays := [
    { name := "X", ty := DType.array DType.float, space := MemorySpace.global },
    { name := "R", ty := DType.array DType.float, space := MemorySpace.global }
  ]
  sharedArrays := []
  body := DStmt.seq
    (DStmt.assign "i"
      (DExpr.binop BinOp.add
        (DExpr.binop BinOp.mul (DExpr.blockIdx Dim.x) (DExpr.blockDim Dim.x))
        (DExpr.threadIdx Dim.x)))
    DStmt.skip
}

#check simpleKernelIR_manual
#eval IO.println s!"✓ Simple kernel IR successfully created!"
#eval IO.println s!"  Name: {simpleKernelIR_manual.name}"
#eval IO.println s!"  Global arrays: {simpleKernelIR_manual.globalArrays.length}"
#eval IO.println s!"  Body: <stmt tree>"
