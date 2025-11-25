import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.GPU.ProcessLauncher
import CLean.ToGPUVerifyIR
import CLean.Verification.GPUVerifyStyle

open GpuDSL
open CLean.DeviceMacro
open CLean.DeviceCodeGen
open CLean.ToGPUVerifyIR
open CLean.Verification.GPUVerify
open Lean DeviceIR DeviceTranslation CLean.GPU.ProcessLauncher Json

kernelArgs IncrementArgs(N: Nat)
  global[data: Array Float]

device_kernel incrementKernel : KernelM IncrementArgs Unit := do
  let args ← getArgs
  let N := args.N
  let data : GlobalArray Float := ⟨args.data⟩

  let i ← globalIdxX
  if i < N then do
    let val ← data.get i      -- Read  data[i]
    data.set i (val + 1.0)    -- Write data[i]



def incrementConfig : Dim3 := ⟨256, 1, 1⟩  -- 256 threads per block
def incrementGrid : Dim3 := ⟨1, 1, 1⟩      -- 1 block



def incrementCPU (N : Nat)
    (data : Array Float) : IO (Array Float) := do
  let initState := mkKernelState [
    globalFloatArray `data data.toList.toArray,
  ]

  let finalState ←
    runKernelCPU
      incrementGrid
      incrementConfig
      ⟨N, `data⟩         -- args
      initState
      incrementKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `data
    | throw <| IO.userError "Result missing"
  if out.size = N then
    pure out
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {N}"

def incrementGPU (N : Nat)
    (data : Array Float) : IO (Array Float) := do

  let cached ← compileKernelToPTX incrementKernelIR

  let scalarParams := #[Float.ofNat N]
  let arrays := [
    (`data, data)
  ]
  let jsonInput := buildLauncherInput scalarParams arrays

  let launcherArgs := #[
    cached.ptxPath.toString,
    incrementKernelIR.name,
    toString incrementGrid.x, toString incrementGrid.y, toString incrementGrid.z,
    toString incrementConfig.x, toString incrementConfig.y, toString incrementConfig.z
  ]

  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }
  let stdin := child.stdin
  stdin.putStr jsonInput
  stdin.putStr "\n"
  stdin.flush

  let stderrContent ← child.stderr.readToEnd
  let stdoutContent ← child.stdout.readToEnd
  let exitCode ← child.wait

  if !stderrContent.trim.isEmpty then
    IO.println "\nDiagnostics:"
    IO.println stderrContent

  if exitCode == 0 then
    IO.println "\n✅ SUCCESS: GPU kernel executed successfully!"
    IO.println stdoutContent
    match Lean.Json.parse stdoutContent with
    | Except.error err =>
      IO.println s!"❌ JSON Parse Error: {err}"
    | Except.ok json =>
      match @Lean.fromJson? IncrementArgsResponse _ json with
      | Except.error err =>
        IO.println s!"❌ JSON Decode Error: {err}"
      | Except.ok response =>
        IO.println "✅ Successfully parsed JSON into TestResponse"
        IO.println s!"\nParsed Results:"
        IO.println s!"  Data: {response.data}"
        return response.data
  else
    IO.println "\n❌ FAILURE: GPU execution failed"

  return #[]


#eval do incrementCPU 2 #[1.0, 6.0]
#eval do incrementGPU 2 #[1.0, 6.0]



def incrementSpec : KernelSpec :=
  deviceIRToKernelSpec incrementKernelIR incrementConfig incrementGrid


theorem increment_safe : KernelSafe incrementSpec := by
  constructor
  · apply identity_kernel_race_free
    intro a ha
    simp [incrementSpec, deviceIRToKernelSpec,
      incrementKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern,
      incrementConfig, DistinctThreads, List.lookup] at ha
    use 1
  · unfold BarrierUniform; intros; trivial
