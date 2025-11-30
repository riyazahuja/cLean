import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.GPU.ProcessLauncher

import CLean.ToGPUVerifyIR
import CLean.Verification.GPUVerifyStyle

open CLean.ToGPUVerifyIR
open CLean.Verification.GPUVerify
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


def saxpyConfig : Dim3 := ⟨512, 1, 1⟩  -- 512 threads per block
def saxpyGrid : Dim3 := ⟨1, 1, 1⟩      -- 1 block

def saxpySpec : KernelSpec :=
  deviceIRToKernelSpec saxpyKernelIR saxpyConfig saxpyGrid

theorem saxpy_safe_direct : KernelSafe saxpySpec := by
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [saxpySpec, deviceIRToKernelSpec, saxpyKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, DistinctThreads, saxpyConfig, List.lookup, HasRace, SeparatedByBarrier] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 <;>
    simp [ha1, ha2,AddressPattern.eval] at h_race <;>
    exact h_neq h_race
  · unfold BarrierUniform; intros; trivial

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

def scanConfig : Dim3 := ⟨256, 1, 1⟩  -- 256 threads per block
def scanGrid : Dim3 := ⟨1, 1, 1⟩      -- 1 block

def upsweepConfig : KernelSpec :=
  deviceIRToKernelSpec upsweepKernelIR scanConfig scanGrid

/-- Upsweep kernel is race-free because:
    1. Access patterns are symLinear with non-zero scale (twod1 parameter)
    2. Distinct threads access distinct array indices when scale ≠ 0
    3. The pattern tid * twod1 + offset is injective for twod1 > 0

    The extracted patterns are:
    - Read idx1: symLinear(twod1, twod1-1) = twod1*tid + twod1 - 1
    - Read idx:  symLinear(twod1, twod-1)  = twod1*tid + twod - 1
    - Write idx1: symLinear(twod1, twod1-1) = twod1*tid + twod1 - 1

    For distinct threads tid1 ≠ tid2 with twod1 > 0:
    - twod1*tid1 + c ≠ twod1*tid2 + c  (since twod1*(tid1-tid2) ≠ 0)
    So no two distinct threads access the same location. -/
theorem upsweep_safe_direct : KernelSafe upsweepConfig := by
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [upsweepConfig, deviceIRToKernelSpec, upsweepKernelIR, extractFromStmt, scanConfig, scanGrid, extractReadsFromExpr, dexprToAddressPattern, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp [ha1, ha2, AddressPattern.eval] at h_race <;>
    apply h_neq <;>
    apply h_race <;>
    simp [SymValue.isNonZero]
  · unfold BarrierUniform; intros; trivial

def downsweepConfig : KernelSpec :=
  deviceIRToKernelSpec downsweepKernelIR scanConfig scanGrid

theorem downsweep_safe_direct : KernelSafe downsweepConfig := by
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [downsweepConfig, deviceIRToKernelSpec, downsweepKernelIR, extractFromStmt, scanConfig, scanGrid, extractReadsFromExpr, dexprToAddressPattern, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩

    rcases ha1 with ha1 | ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 | ha2 <;>
    simp_all <;>
    apply h_neq <;>
    simp [SymValue.isNonZero] at *
  · unfold BarrierUniform; intros; trivial

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

      -- IMPORTANT: Scalar params must match the order in the generated CUDA kernel!
      -- The kernel signature is: upsweepKernel(int twod1, int length, int twod, int* data)
      -- Parameters are discovered in the order they appear in the kernel body.
      let scalarParams : Array Int := #[twod1, roundedLength, twod]
      let arrays := [
        (`data, upsweepData)
      ]

      let jsonInput := buildLauncherInput scalarParams arrays
      let grid :Dim3 := ⟨numBlocks, 1, 1⟩
      let block :Dim3 := ⟨numThreadsPerBlock, 1, 1⟩
      IO.println s!"Launching Upsweep Kernel with grid={grid.x}, block={block.x}, twod={twod}, twod1={twod1}"
      IO.println s!"Input Data: {jsonInput}"
      let launcherArgs := #[
        cached.ptxPath.toString,
        upsweepKernelIR.name,
        toString grid.x, toString grid.y, toString grid.z,
        toString block.x, toString block.y, toString block.z,
      ]
      IO.println s!"Launcher Args: {launcherArgs}"

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
        -- The auto-generated FromJson expects {"results":{"data":[...]}} format
        match @Lean.fromJson? ScanArgsResponse _ json with
        | Except.error err =>
          throw <| IO.userError s!"❌ JSON Decode Error: {err}"
        | Except.ok response =>
          IO.println "✅ Successfully parsed JSON into TestResponse"
          IO.println s!"\nParsed Results:"
          IO.println s!"  Data: {response.data}"
          upsweepData := response.data
          twod := twod * 2

    let zeroedOut := upsweepData.set! (roundedLength - 1) 0

    --Downsweep phase
    let cached ← compileKernelToPTX downsweepKernelIR
    let mut downsweepData := zeroedOut
    twod := roundedLength / 2
    while twod >= 1 do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock

      -- IMPORTANT: Scalar params must match the order in the generated CUDA kernel!
      -- The kernel signature is: downsweepKernel(int twod1, int twod, int length, int* data)
      -- Parameters are discovered in the order they appear in the kernel body.
      let scalarParams : Array Int := #[twod1, twod, roundedLength]
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
        -- The auto-generated FromJson expects {"results":{"data":[...]}} format
        match @Lean.fromJson? ScanArgsResponse _ json with
        | Except.error err =>
          throw <| IO.userError s!"❌ JSON Decode Error: {err}"
        | Except.ok response =>
          IO.println "✅ Successfully parsed JSON into TestResponse"
          IO.println s!"\nParsed Results:"
          IO.println s!"  Data: {response.data}"
          downsweepData := response.data
          twod := twod / 2

    pure <| downsweepData.take (n+1)

#eval do exclusiveScanCPU #[1,2,3,4,5,6,7,8]
#eval do exclusiveScanGPU #[1,2,3,4,5,6,7,8]

end ExclusiveScan


namespace BasicMatMul


kernelArgs MatMulArgs(N: Nat)
  global[A B C: Array Float]

device_kernel matmulKernel : KernelM MatMulArgs Unit := do
  let args ← getArgs
  let N := args.N
  let A : GlobalArray Float := ⟨args.A⟩
  let B : GlobalArray Float := ⟨args.B⟩
  let C : GlobalArray Float := ⟨args.C⟩

  let row ← globalIdxX
  let col ← globalIdxY

  if row < N && col < N then
    let mut result : Float := 0.0
    for k in [0:N] do
      let aVal ← A.get (row * N + k)
      let bVal ← B.get (k * N + col)
      result := result + aVal * bVal
    C.set (row * N + col) result

def matmulConfig : Dim3 := ⟨512, 1, 1⟩  -- 512 threads per block
def matmulGrid : Dim3 := ⟨1, 1, 1⟩      -- 1 block

def matmulSpec : KernelSpec :=
  deviceIRToKernelSpec matmulKernelIR matmulConfig matmulGrid

/-- Matrix multiplication kernel is race-free because:
    1. Each thread computes one output element C[row][col]
    2. The write pattern is symLinear with non-zero scale
    3. Reads from A and B don't conflict (read-read is not a race) -/
theorem matmul_safe_direct : KernelSafe matmulSpec := by
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [matmulSpec, deviceIRToKernelSpec, matmulKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, DistinctThreads, matmulConfig, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp [ha1, ha2,AddressPattern.eval] at h_race
    apply h_neq <;>
    apply h_race <;>
    simp [SymValue.isNonZero]
    /-
    tid1 tid2 : ℕ
a1 a2 : AccessPattern
left✝¹ : tid1 < 512
left✝ : tid2 < 512
h_neq : ¬tid1 = tid2
ha1 : a1 =
  AccessPattern.read
    (AddressPattern.symLinear ((SymValue.param "N").symAdd (SymValue.const 0))
      ((SymValue.const 0).symAdd (SymValue.param "k")))
    2
ha2 : a2 =
  AccessPattern.write
    (AddressPattern.symLinear ((SymValue.param "N").symAdd (SymValue.const 1))
      ((SymValue.const 0).symAdd (SymValue.const 0)))
    2
h_race : ((SymValue.param "N").symAdd (SymValue.const 0)).isNonZero = true →
  ((SymValue.param "N").symAdd (SymValue.const 1)).isNonZero = true → tid1 = tid2
⊢ False
    -/
    sorry
  · unfold BarrierUniform; intros; trivial


def matmul (n : Nat)
    (A B : Array (Array Float)) : IO (Array (Array Float)) := do
  let flattened_A  := A.flatMap id
  let flattened_B  := B.flatMap id
  IO.println s!"Flattened A: {flattened_A}"
  IO.println s!"Flattened B: {flattened_B}"

  let initState := mkKernelState [
    globalFloatArray `A flattened_A,
    globalFloatArray `B flattened_B,
    globalFloatArray `C (Array.replicate (n*n) 0.0)
  ]

  let threadsPerBlock := 32
  let numBlocks := (n + threadsPerBlock - 1) / threadsPerBlock

  let finalState ←
    runKernelCPU
      ⟨numBlocks, numBlocks, 1⟩
      ⟨threadsPerBlock, threadsPerBlock, 1⟩
      ⟨n, `A, `B, `C⟩
      initState
      matmulKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `C
    | throw <| IO.userError "Result missing"
  if out.size = n * n then
    for i in [0:n] do
      for j in [0:n] do
        let val := out[i * n + j]!
        IO.println s!"C[{i},{j}] = {val}"

    let result := Array.ofFn fun (i : Fin n) =>
      Array.ofFn fun (j : Fin n) => out[i.val * n + j.val]!
    pure result
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n*n}"


  def matmulGPU (n : Nat)
    (A B : Array (Array Float)) : IO (Array (Array Float)) := do
    let flattened_A := A.flatMap id
    let flattened_B := B.flatMap id
    IO.println s!"Flattened A: {flattened_A}"
    IO.println s!"Flattened B: {flattened_B}"

    let cached ← compileKernelToPTX matmulKernelIR

    -- Use typed scalars: N is int
    let scalarParams := #[ScalarValue.int n]
    let arrays := [
    (`A, flattened_A),
    (`B, flattened_B),
    (`C, Array.replicate (n*n) 0.0)
    ]
    let jsonInput := buildLauncherInputTyped scalarParams arrays

    let threadsPerBlock := 32
    let numBlocks := (n + threadsPerBlock - 1) / threadsPerBlock
    let grid : Dim3 := ⟨numBlocks, numBlocks, 1⟩
    let block : Dim3 := ⟨threadsPerBlock, threadsPerBlock, 1⟩

    let launcherArgs := #[
    cached.ptxPath.toString,
    matmulKernelIR.name,
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


    if exitCode == 0 then
    IO.println "\n✅ SUCCESS: GPU kernel executed successfully!"
    IO.println stdoutContent
    match Lean.Json.parse stdoutContent with
    | Except.error err =>
      IO.println s!"❌ JSON Parse Error: {err}"
      return #[]
    | Except.ok json =>
      match @Lean.fromJson? MatMulArgsResponse _ json with
      | Except.error err =>
      IO.println s!"❌ JSON Decode Error: {err}"
      return #[]
      | Except.ok response =>
      IO.println "✅ Successfully parsed JSON into MatMulArgsResponse"
      IO.println s!"\nParsed Results:"
      IO.println s!"  C: {response.C}"
      let out := response.C
      if out.size = n * n then
        let result := Array.ofFn fun (i : Fin n) =>
        Array.ofFn fun (j : Fin n) => out[i.val * n + j.val]!
        return result
      else
        throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n*n}"
    else
    IO.println "\n❌ FAILURE: GPU execution failed"
    IO.println s!"Stderr: {stderrContent}"
    IO.println s!"Stdout: {stdoutContent}"
    return #[]

#eval do matmul 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmul 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]

#eval do matmulGPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmulGPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]

end BasicMatMul
