import CLean.DeviceMacro
import CLean.GPU
import CLean.DeviceIR

open CLean.DeviceMacro
open GpuDSL
open DeviceIR

/-! # Clean DeviceMacro Test

Testing without #check and #print to avoid elaboration issues.
-/

-- Test 1: Simple array copy
namespace Test1

kernelArgs CopyArgs(N: Nat)
  global[input output: Array Float]

device_kernel arrayCopy : KernelM CopyArgs Unit := do
  let args ← getArgs
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let i ← globalIdxX
  let val ← input.get i
  output.set i val

end Test1


-- Test 2: SAXPY with binary operations
namespace Test2

kernelArgs SaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel saxpy : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩
  let i ← globalIdxX
  let xi ← x.get i
  let yi ← y.get i
  let scaled := alpha * xi
  let result := scaled + yi
  r.set i result

end Test2

-- Now print the IRs (after all definitions are complete)
#print Test1.arrayCopyIR
#print Test2.saxpyIR

#eval IO.println "\n✅ All kernels compiled and IR extracted successfully!"
