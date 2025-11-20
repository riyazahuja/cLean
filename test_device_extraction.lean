import CLean.DeviceMacro
import CLean.GPU
import CLean.DeviceIR

open CLean.DeviceMacro
open GpuDSL
open CLean

-- Test 1: Simple kernel with array operations (no conditionals)
namespace Test1

kernelArgs SimpleArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

-- Simplified saxpy kernel without conditionals
-- Just does: r[i] = alpha * x[i] + y[i]
def simpleSaxpyKernel : KernelM SimpleArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩

  let i ← globalIdxX
  let xi ← x.get i
  let yi ← y.get i
  let result := alpha * xi + yi
  r.set i result

-- Extract the kernel IR
#check simpleSaxpyKernel
-- #eval extractKernelFromSyntax `simpleSaxpyKernel

end Test1


-- Test 2: Even simpler - just array copy
namespace Test2

kernelArgs CopyArgs(N: Nat)
  global[input output: Array Float]

def copyKernel : KernelM CopyArgs Unit := do
  let args ← getArgs
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  let val ← input.get i
  output.set i val

#check copyKernel

end Test2


-- Test 3: Multiple operations
namespace Test3

kernelArgs MultiOpArgs(N: Nat, a: Float, b: Float)
  global[x y z: Array Float]

def multiOpKernel : KernelM MultiOpArgs Unit := do
  let args ← getArgs
  let a := args.a
  let b := args.b
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let z : GlobalArray Float := ⟨args.z⟩

  let i ← globalIdxX
  let xi ← x.get i
  let yi ← y.get i

  -- z[i] = a * x[i] + b * y[i]
  let term1 := a * xi
  let term2 := b * yi
  let sum := term1 + term2
  z.set i sum

#check multiOpKernel

end Test3


-- Test 4: Division and comparison (deferred patterns)
namespace Test4

kernelArgs DivArgs(N: Nat)
  global[x y: Array Float]

def divKernel : KernelM DivArgs Unit := do
  let args ← getArgs
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩

  let i ← globalIdxX
  let xi ← x.get i
  let yi ← y.get i
  let ratio := xi / yi
  y.set i ratio

#check divKernel

end Test4

-- Print success message
#eval IO.println "All test kernels compiled successfully!"
#eval IO.println "DeviceMacro extraction is ready for testing."
