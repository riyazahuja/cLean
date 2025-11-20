import CLean.KernelMacro
import CLean.GPU

open CLean.KernelMacro
open CLean

-- Test the basic macro with a simple definition
gpu_kernel test_simple : Nat := 42

#check test_simple
#print test_simple

-- Test with KernelM do-notation
gpu_kernel test_kernel : KernelM Unit Unit := do
  let i ← globalIdxX
  barrier
  let j ← globalIdxY
  pure ()

#check test_kernel
#print test_kernel
