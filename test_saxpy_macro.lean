import CLean.KernelMacro_simple
import CLean.GPU

open CLean.KernelMacro
open GpuDSL
open CLean

-- Define args structure for saxpy
structure SaxpyArgs where
  N : Nat
  alpha : Float
  x : Name
  y : Name
  r : Name

-- Use gpu_kernel macro for saxpy
gpu_kernel saxpyKernel (args : SaxpyArgs) : KernelM SaxpyArgs Unit := do
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩
  let i ← globalIdxX
  barrier
  pure ()

#check saxpyKernel
#check saxpyKernelIR
#print saxpyKernelIR
