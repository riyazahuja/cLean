import CLean.KernelMacro_simple
import CLean.GPU

open CLean.KernelMacro
open GpuDSL
open CLean

-- Define args structure for transpose
structure TransposeArgs where
  N : Nat
  input : Name
  output : Name
  tile : Name

-- Use gpu_kernel macro for transpose
gpu_kernel transposeKernel (args : TransposeArgs) : KernelM TransposeArgs Unit := do
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.tile⟩

  let row ← globalIdxX
  let col ← globalIdxY

  -- Barrier to synchronize
  barrier

  pure ()

#check transposeKernel
#check transposeKernelIR
#print transposeKernelIR
