import CLean.DeviceMacro
import CLean.GPU
import CLean.DeviceIR

open CLean.DeviceMacro
open GpuDSL
open DeviceIR

/-! # Test device_kernel Macro

This file tests the device_kernel macro which extracts DeviceIR
from KernelM definitions at syntax level.
-/

-- Test 1: Simple kernel with basic globalIdxX pattern
namespace Test1

kernelArgs SimpleArgs(N: Nat)
  global[x: Array Float]

device_kernel testKernel1 : KernelM SimpleArgs Unit := do
  let args ← getArgs
  let x : GlobalArray Float := ⟨args.x⟩
  let i ← globalIdxX
  pure ()

-- Check what was generated
#check testKernel1      -- The KernelM definition
#check testKernel1IR    -- The DeviceIR Kernel definition
#print testKernel1IR    -- Print the extracted IR

end Test1


-- Test 2: Kernel with multiple arrays
namespace Test2

kernelArgs MultiArrayArgs(N: Nat)
  global[x y z: Array Float]

device_kernel testKernel2 : KernelM MultiArrayArgs Unit := do
  let args ← getArgs
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let z : GlobalArray Float := ⟨args.z⟩
  let i ← globalIdxX
  pure ()

#check testKernel2
#check testKernel2IR
#print testKernel2IR

end Test2


-- Test 3: Minimal saxpy-like kernel
namespace Test3

kernelArgs SaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel miniSaxpy : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩
  let i ← globalIdxX
  pure ()

#check miniSaxpy
#check miniSaxpyIR
#print miniSaxpyIR

end Test3

#eval IO.println "\n✅ All device_kernel macro tests compiled successfully!"
#eval IO.println "The macro successfully extracts DeviceIR from KernelM syntax.\n"
