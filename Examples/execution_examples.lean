import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.GPU.ProcessLauncher


open Lean GpuDSL DeviceIR CLean.DeviceMacro CLean.DeviceCodeGen DeviceTranslation CLean.GPU.ProcessLauncher Json


set_option maxHeartbeats 2000000

namespace Saxpy

kernelArgs saxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

#check saxpyArgs
#print saxpyArgs

device_kernel saxpyKernel : KernelM saxpyArgs Unit := do
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

def saxpyCPU (n : Nat)
    (α : Float)
    (x y : Array Float) : IO (Array Float) := do
  let initState := mkKernelState [
    globalFloatArray `X x.toList.toArray,
    globalFloatArray `Y y.toList.toArray,
    globalFloatArray `R (Array.replicate n 0.0)
  ]

  let finalState ←
    runKernelCPU
      ⟨(n + 511) / 512, 1, 1⟩         -- grid
      ⟨512, 1, 1⟩                     -- block
      ⟨n, α, `X, `Y, `R⟩         -- args
      initState
      saxpyKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `R
    | throw <| IO.userError "Result missing"
  if out.size = n then
    pure out
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n}"

def saxpyGPU (n : Nat)
    (α : Float)
    (x y : Array Float) : IO (Array Float) := do

  let cached ← compileKernelToPTX saxpyKernelIR

  let scalarParams := #[Float.ofNat n, α]
  let arrays := [
    (`x, x),
    (`y, y),
    (`r, Array.replicate n 0.0)
  ]
  let jsonInput := buildLauncherInput scalarParams arrays
  let grid : Dim3 := ⟨(n + 511) / 512, 1, 1⟩
  let block : Dim3 := ⟨512, 1, 1⟩

  let launcherArgs := #[
    cached.ptxPath.toString,
    saxpyKernelIR.name,
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
      match @Lean.fromJson? saxpyArgsResponse _ json with
      | Except.error err =>
        IO.println s!"❌ JSON Decode Error: {err}"
      | Except.ok response =>
        IO.println "✅ Successfully parsed JSON into TestResponse"
        IO.println s!"\nParsed Results:"
        IO.println s!"  X: {response.x}"
        IO.println s!"  Y: {response.y}"
        IO.println s!"  R: {response.r}"
        return response.r
  else
    IO.println "\n❌ FAILURE: GPU execution failed"

  return #[]


#eval do saxpyCPU 2 8.0 #[1.0, 1.0] #[2.0, 2.0]
#eval do saxpyGPU 2 8.0 #[1.0, 1.0] #[2.0, 2.0]

end Saxpy






namespace ExclusiveScan


/-- Find the next power of 2 greater than or equal to n -/
def nextPow2 (n : Nat) : Nat :=
  let n := n - 1
  let n := n ||| (n >>> 1)
  let n := n ||| (n >>> 2)
  let n := n ||| (n >>> 4)
  let n := n ||| (n >>> 8)
  let n := n ||| (n >>> 16)
  n + 1

kernelArgs ScanArgs(length: Int, twod1: Int, twod: Int)
  global[data: Array Int]

#print ScanArgsResponse

device_kernel upsweepKernel : KernelM ScanArgs Unit := do
  let args ← getArgs
  let index ← globalIdxX
  let i := index * args.twod1
  if i + args.twod1 - 1 < args.length then
    let data : GlobalArray Int := ⟨args.data⟩
    let idx1 := (i + args.twod1 - 1).toNat?.getD 0
    let idx := (i + args.twod - 1).toNat?.getD 0
    let val1 ← data.get idx1
    let val2 ← data.get idx
    data.set idx1 (val1 + val2)


device_kernel downsweepKernel : KernelM ScanArgs Unit := do
  let args ← getArgs
  let index ← globalIdxX
  let i := index * args.twod1
  if (i + args.twod - 1 < args.length) && (i + args.twod1 - 1 < args.length) then do
    let data : GlobalArray Int := ⟨args.data⟩
    let idx := (i + args.twod - 1).toNat?.getD 0
    let idx1 := (i + args.twod1 - 1).toNat?.getD 0
    let t ← data.get idx
    let val ← data.get idx1
    data.set idx val
    data.set idx1 (val + t)

/-- Exclusive scan implementation -/
def exclusiveScanCPU (input : Array Int) : IO (Array Int) := do
  let n := input.size
  if n = 0 then
    pure input
  else

    let roundedLength := nextPow2 n
    let paddedData := input.toList.toArray ++ (Array.replicate (roundedLength - n) 0)

    let mut upsweepState := mkKernelState [globalIntArray `data paddedData]

    let numThreadsPerBlock := 256

    -- Upsweep phase
    let mut twod := 1
    while twod < roundedLength do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock
      upsweepState ← runKernelCPU
        ⟨numBlocks, 1, 1⟩
        ⟨numThreadsPerBlock, 1, 1⟩
        ⟨roundedLength, twod1, twod, `data⟩
        upsweepState
        upsweepKernel
      twod := twod * 2

    -- Set last element to 0
    let some (KernelValue.arrayInt out) := upsweepState.globals.get? `data
      | throw <| IO.userError "Result missing"
    let zeroedOut := out.set! (roundedLength - 1) 0

    -- Downsweep phase
    let mut downsweepState := mkKernelState [globalIntArray `data zeroedOut]
    twod := roundedLength / 2
    while twod >= 1 do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock
      downsweepState ← runKernelCPU
        ⟨numBlocks, 1, 1⟩
        ⟨numThreadsPerBlock, 1, 1⟩
        ⟨roundedLength, twod1, twod, `data⟩
        downsweepState
        downsweepKernel
      twod := twod / 2

    let some (KernelValue.arrayInt out) := downsweepState.globals.get? `data
      | throw <| IO.userError "Result missing"
    pure <| out.take (n+1)


/-- Exclusive scan implementation TODO!!!-/
def exclusiveScanGPU (input : Array Int) : IO (Array Int) := do
  let n := input.size
  if n = 0 then
    pure input
  else
    let roundedLength := nextPow2 n
    let paddedData := input.toList.toArray ++ (Array.replicate (roundedLength - n) (0 : Int))
    let cached ← compileKernelToPTX upsweepKernelIR

    let mut upsweepData := paddedData

    let numThreadsPerBlock := 256

    -- Upsweep phase
    let mut twod := 1
    while twod < roundedLength do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock

      let scalarParams : Array Int := #[roundedLength, twod1, twod]
      let arrays := [
        (`data, upsweepData)
      ]

      let jsonInput := buildLauncherInput scalarParams arrays
      let grid :Dim3 := ⟨numBlocks, 1, 1⟩
      let block :Dim3 := ⟨numThreadsPerBlock, 1, 1⟩

      let launcherArgs := #[
        cached.ptxPath.toString,
        upsweepKernelIR.name,
        toString grid.x, toString grid.y, toString grid.z,
        toString block.x, toString block.y, toString block.z,
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

      if exitCode != 0 then
        throw <| IO.userError s!"❌ GPU Upsweep Error: {stderrContent}"

      match Lean.Json.parse stdoutContent with
      | Except.error err =>
        throw <| IO.userError s!"❌ JSON Parse Error: {err}"
      | Except.ok json =>
        match @Lean.fromJson? ScanArgsResponse _ json with
        | Except.error err =>
          throw <| IO.userError s!"❌ JSON Decode Error: {err}"
        | Except.ok response =>
          IO.println "✅ Successfully parsed JSON into TestResponse"
          IO.println s!"\nParsed Results:"
          IO.println s!"  Data: {response.data}"
          /-
          TODO FIX WORKAROUND OF FLOAT PARSING
          -/
          upsweepData := response.data.map (fun x => x.toInt64.toInt)
          twod := twod * 2

    let zeroedOut := upsweepData.set! (roundedLength - 1) 0

    --Downsweep phase
    let cached ← compileKernelToPTX downsweepKernelIR
    let mut downsweepData := zeroedOut
    twod := roundedLength / 2
    while twod >= 1 do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock

      let scalarParams : Array Int := #[roundedLength, twod1, twod]
      let arrays := [
        (`data, downsweepData)
      ]

      let jsonInput := buildLauncherInput scalarParams arrays
      let grid : Dim3 := ⟨numBlocks, 1, 1⟩
      let block : Dim3 := ⟨numThreadsPerBlock, 1, 1⟩

      let launcherArgs := #[
        cached.ptxPath.toString,
        downsweepKernelIR.name,
        toString grid.x, toString grid.y, toString grid.z,
        toString block.x, toString block.y, toString block.z,
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

      if exitCode != 0 then
        throw <| IO.userError s!"❌ GPU Downsweep Error: {stderrContent}"

      match Lean.Json.parse stdoutContent with
      | Except.error err =>
        throw <| IO.userError s!"❌ JSON Parse Error: {err}"
      | Except.ok json =>
        match @Lean.fromJson? ScanArgsResponse _ json with
        | Except.error err =>
          throw <| IO.userError s!"❌ JSON Decode Error: {err}"
        | Except.ok response =>
          IO.println "✅ Successfully parsed JSON into TestResponse"
          IO.println s!"\nParsed Results:"
          IO.println s!"  Data: {response.data}"
          downsweepData := response.data.map (fun x => x.toInt64.toInt)
          twod := twod / 2

    pure <| downsweepData.take (n+1)


#eval do exclusiveScanCPU #[1,2,3,4,5,6,7,8]
#eval do exclusiveScanGPU #[1,2,3,4,5,6,7,8]



end ExclusiveScan
