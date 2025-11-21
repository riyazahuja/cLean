/-
  Test just the kernel compilation (no execution)
-/

import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen
open CLean.GPU.KernelCache

-- Simple SAXPY kernel
kernelArgs TestArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel testKernel : KernelM TestArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩

  let i ← globalIdxX
  if i < N then do
    let xi ← x.get i
    let yi ← y.get i
    r.set i (alpha * xi + yi)

def main : IO Unit := do
  IO.println "=== Test: Kernel Compilation Only ==="

  IO.println "\n1. Generating CUDA code..."
  let cudaCode := kernelToCuda testKernelIR
  IO.println s!"Generated {cudaCode.length} characters of CUDA"

  IO.println "\n2. Computing cache hash..."
  let kernelHash := hashString cudaCode
  IO.println s!"Cache key: {kernelHash}"

  IO.println "\n3. Getting cached kernel (will compile if needed)..."
  let cached ← getCachedKernel testKernelIR

  IO.println s!"CUDA file: {cached.cudaSourcePath}"
  IO.println s!"PTX file: {cached.ptxPath}"

  let ptxExists ← cached.ptxPath.pathExists
  if ptxExists then
    IO.println "✓ PTX file exists!"
  else
    IO.println "✗ PTX file missing"

  IO.println "\n=== Test Complete ===  "
