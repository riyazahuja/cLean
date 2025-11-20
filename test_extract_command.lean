import CLean.GPU

open GpuDSL

/-! # Test: Normal kernel definitions (no macro)

Write kernels normally, then extract IR separately.
-/

-- Normal kernel definition (no device_kernel macro)
kernelArgs CopyArgs(N: Nat)
  global[input output: Array Float]

def arrayCopy : KernelM CopyArgs Unit := do
  let args ← getArgs
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let i ← globalIdxX
  let val ← input.get i
  output.set i val

-- Check that it type-checks properly
#check arrayCopy

-- Now we could add a command like:
-- #extract_ir arrayCopy
-- But we'll implement that in the next step

#eval IO.println "✅ Kernel type-checks correctly!"
