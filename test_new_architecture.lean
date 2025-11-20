/-
  Test the new architecture with a simple kernel
-/

import CLean.DeviceIR
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.KernelOps
import CLean.KernelBuilder

open CLean DeviceIR

-- Define a simple kernel using KernelOps
def simpleAddKernel {m : Type → Type} [Monad m] [KernelOps m] (N : Nat) : m Unit := do
  -- Get global thread index
  let i ← KernelOps.globalIdxX

  -- Create bound expression for N
  let nExpr := KernelOps.natLit N

  -- Check bounds
  KernelOps.ifThenElse (KernelOps.lt i nExpr)
    (do
      -- Read from input arrays
      let x ← KernelOps.globalGet "X" i
      let y ← KernelOps.globalGet "Y" i

      -- Add them
      let sum := KernelOps.add x y

      -- Write result
      KernelOps.globalSet "R" i sum)
    (pure ())

-- Test: build the kernel IR
def simpleAddKernelIR : Kernel :=
  buildKernel "simpleAdd"
    [{ name := "N", ty := .nat }]
    (simpleAddKernel 1024)

-- Print the IR
#eval IO.println s!"Kernel IR:\n{repr simpleAddKernelIR}"

-- Test with a slightly more complex kernel with a loop
def prefixSumKernel {m : Type → Type} [Monad m] [KernelOps m] (N : Nat) : m Unit := do
  let tid ← KernelOps.threadIdxX
  let nExpr := KernelOps.natLit N

  -- Simple prefix sum in shared memory (stride-based)
  KernelOps.forLoop "stride" (KernelOps.natLit 1) nExpr fun stride => do
    KernelOps.barrier

    let readIdx := KernelOps.sub tid stride
    let canRead := KernelOps.ge tid stride

    KernelOps.ifThenElse canRead
      (do
        let val ← KernelOps.sharedGet "data" tid
        let prevVal ← KernelOps.sharedGet "data" readIdx
        let sum := KernelOps.add val prevVal
        KernelOps.sharedSet "data" tid sum)
      (pure ())

def prefixSumKernelIR : Kernel :=
  buildKernel "prefixSum"
    [{ name := "N", ty := .nat }]
    (prefixSumKernel 256)

#eval IO.println s!"\nPrefix Sum Kernel IR:\n{repr prefixSumKernelIR}"

-- Verify the structure
#check simpleAddKernelIR
#check prefixSumKernelIR
