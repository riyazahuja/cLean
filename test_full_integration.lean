/-
  Full Integration Test: Lean → GPU → Results

  Tests the complete pipeline with proper error handling
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceCodeGen
import CLean.GPU.ProcessLauncher

open GpuDSL CLean.DeviceMacro CLean.DeviceCodeGen
open CLean.GPU.ProcessLauncher

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
  IO.println "╔════════════════════════════════════════════════════════╗"
  IO.println "║     cLean Full Integration Test: Lean → GPU            ║"
  IO.println "╚════════════════════════════════════════════════════════╝"

  -- Test data
  let n := 8
  let alpha := 3
  let x := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let y := #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  let expected := #[3.5, 6.0, 8.5, 11.0, 13.5, 16.0, 18.5, 21.0]

  IO.println s!"\nInput:"
  IO.println s!"  x     = {x}"
  IO.println s!"  y     = {y}"
  IO.println s!"  alpha = {alpha}"
  IO.println s!"  Expected: r[i] = {alpha} * x[i] + y[i]"
  IO.println s!"  Expected result: {expected}"

  -- Step 1: Compile kernel to PTX
  IO.println "\n[Step 1] Compiling kernel to PTX..."
  let cached ← compileKernelToPTX testKernelIR
  IO.println s!"  PTX file: {cached.ptxPath}"

  -- Step 2: Build JSON input
  IO.println "\n[Step 2] Building JSON input..."
  let scalarParams := #[Float.ofNat n, alpha]
  let arrays := [
    (`X, x),
    (`Y, y),
    (`R, Array.replicate n 0.0)
  ]
  let jsonInput := buildLauncherInput scalarParams arrays
  IO.println s!"  JSON size: {jsonInput.length} bytes"

  -- Step 3: Spawn launcher process
  IO.println "\n[Step 3] Spawning GPU launcher..."
  let grid : Dim3 := ⟨1, 1, 1⟩
  let block : Dim3 := ⟨256, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    testKernelIR.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  -- Step 4: Send input
  IO.println "[Step 4] Sending input to launcher..."
  let stdin := child.stdin
  stdin.putStr jsonInput
  stdin.putStr "\n"
  stdin.flush

  -- Step 5: Read output
  IO.println "[Step 5] Reading output from GPU..."
  let stderrContent ← child.stderr.readToEnd
  let stdoutContent ← child.stdout.readToEnd

  -- Step 6: Wait for completion
  IO.println "[Step 6] Waiting for process completion..."
  let exitCode ← child.wait

  -- Step 7: Display results
  IO.println "\n╔════════════════════════════════════════════════════════╗"
  IO.println "║                    Results                             ║"
  IO.println "╚════════════════════════════════════════════════════════╝"

  IO.println s!"\nExit code: {exitCode}"

  if !stderrContent.trim.isEmpty then
    IO.println "\nDiagnostics:"
    IO.println stderrContent

  IO.println "\nGPU Output:"
  IO.println stdoutContent

  if exitCode == 0 then
    IO.println "\n✅ SUCCESS: GPU kernel executed successfully!"
    IO.println "\nTo verify:"
    IO.println s!"  Expected: {expected}"
    IO.println "  Check if R values match in the output above"
  else
    IO.println "\n❌ FAILURE: GPU execution failed"

  IO.println "\n╔════════════════════════════════════════════════════════╗"
  IO.println "║              Integration Test Complete                 ║"
  IO.println "╚════════════════════════════════════════════════════════╝"

#eval main
