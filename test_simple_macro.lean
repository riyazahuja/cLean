import CLean.KernelMacro_simple
import CLean.GPU

open CLean.KernelMacro
open GpuDSL
open CLean

-- Test with simple definition
gpu_kernel test1 : Nat := 42

#check test1
#check test1IR
#print test1IR

-- Test with KernelM
gpu_kernel test2 : KernelM Unit Unit := do
  let i ← globalIdxX
  barrier
  pure ()

#check test2
#check test2IR
#print test2IR

-- Test with arrays
structure ArrayArgs where
  input : Name
  output : Name
  temp : Name

gpu_kernel test3 (args : ArrayArgs) : KernelM Unit Unit := do
  let x : GlobalArray Float := ⟨args.input⟩
  let y : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.temp⟩
  let i ← globalIdxX
  barrier
  pure ()

#check test3
#check test3IR
#print test3IR
