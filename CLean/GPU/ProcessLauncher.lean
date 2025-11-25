/-
  Process-Based GPU Launcher

  Executes GPU kernels by calling external gpu_launcher process.
  Avoids FFI linking issues by using process communication.
-/

import Lean
import CLean.GPU
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache

namespace CLean.GPU.ProcessLauncher

open Lean (Name)
open DeviceIR
open CLean.DeviceCodeGen
open CLean.GPU.KernelCache
open GpuDSL
open System (FilePath)

/-- Extract scalar parameter values from KernelState -/
def extractScalarParams (kernel : Kernel) (state : KernelState) : Array Float := Id.run do
  let scalarParams := CLean.DeviceCodeGen.extractScalarParams kernel.body
  let mut result := #[]

  for (name, _ty) in scalarParams do
    match state.globals.get? (Name.mkSimple name) with
    | some (.float f) => result := result.push f
    | some (.int i) => result := result.push (Float.ofInt i)
    | some (.nat n) => result := result.push (Float.ofNat n)
    | _ => result := result.push 0.0

  return result

/-- Extract global array data from KernelState -/
def extractGlobalArrays (kernel : Kernel) (state : KernelState) : List (Name × Array Float) := Id.run do
  let mut result := []

  for arrayDecl in kernel.globalArrays do
    let name := Name.mkSimple arrayDecl.name
    match state.globals.get? name with
    | some (.arrayFloat arr) => result := (name, arr) :: result
    | some (.arrayInt arr) =>
        let floatArr := arr.map Float.ofInt
        result := (name, floatArr) :: result
    | some (.arrayNat arr) =>
        let floatArr := arr.map Float.ofNat
        result := (name, floatArr) :: result
    | _ => pure ()

  return result.reverse

/-- JSON encoder for Float arrays -/
def floatArrayToJson {α : Type} [ToString α] (arr : Array α) : String :=
  "[" ++ String.intercalate "," (arr.map toString |>.toList) ++ "]"

/-- JSON encoder for named array -/
def namedArrayToJson {α : Type} [ToString α] (name : String) (arr : Array α) : String :=
  s!"\"{name}\":{floatArrayToJson arr}"

/-- Build JSON payload for launcher -/
def buildLauncherInput {α : Type} [ToString α] (scalarParams : Array α) (arrays : List (Name × Array α)) : String :=
  let scalarsJson := floatArrayToJson scalarParams
  let arraysJson := String.intercalate "," (arrays.map fun (name, arr) =>
    namedArrayToJson (toString name) arr)

  "{\"scalars\":" ++ scalarsJson ++ ",\"arrays\":{" ++ arraysJson ++ "}}"

/-- Parse JSON output from launcher - STUB for now -/
def parseLauncherOutput (output : String) : Except String (List (Name × Array Float)) := do
  -- TODO: Implement proper JSON parsing
  -- For now, just return a dummy result
  -- The output will still be printed to console for debugging
  throw s!"JSON parsing not yet implemented. Output was:\n{output}"

/-- Compile kernel to PTX using nvcc -/
def compileKernelToPTX (kernel : Kernel) : IO CachedKernel := do
  -- Get cached kernel info
  let cached ← getCachedKernel kernel

  -- Check if PTX already exists
  let ptxExists ← cached.ptxPath.pathExists

  if !ptxExists then
    IO.println s!"[GPU] Compiling {kernel.name} to PTX..."

    -- Compile using nvcc
    let nvccPath := "nvcc"
    let args := #[
      "-ptx",
      "-O3",
      "--gpu-architecture=compute_75",
      "-o", cached.ptxPath.toString,
      cached.cudaSourcePath.toString
    ]
    IO.println s!"[nvcc] {nvccPath} {String.intercalate " " args.toList}"
    let output ← IO.Process.run {
      cmd := nvccPath
      args := args
    }


    if !output.trim.isEmpty then
      IO.eprintln s!"[nvcc] {output}"

    IO.println "[GPU] Compilation complete!"
  else
    IO.println s!"[GPU] Using cached PTX for {kernel.name}"

  return cached

/-- Execute kernel using external launcher process -/
def executeKernel
    (kernel : Kernel)
    (grid block : Dim3)
    (scalarParams : Array Float)
    (arrays : List (Name × Array Float))
    (launcherPath : String := "./gpu_launcher")
    : IO (List (Name × Array Float)) := do

  -- Compile kernel if needed
  let cached ← compileKernelToPTX kernel

  -- Build JSON input
  let jsonInput := buildLauncherInput scalarParams arrays

  IO.println s!"[GPU] Launching {kernel.name}..."
  IO.println s!"[GPU] Grid: {grid.x}x{grid.y}x{grid.z}"
  IO.println s!"[GPU] Block: {block.x}x{block.y}x{block.z}"

  -- Call launcher process
  let launcherArgs := #[
    cached.ptxPath.toString,
    kernel.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  let child ← IO.Process.spawn {
    cmd := launcherPath
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  -- Write input to stdin
  let stdin := child.stdin
  stdin.putStr jsonInput
  stdin.flush

  -- Close stdin to signal end of input
  -- (Note: Lean 4 doesn't expose handle.close, so we let it close on process exit)

  -- Read output
  let stdout ← IO.asTask child.stdout.readToEnd Task.Priority.dedicated
  let stderr ← IO.asTask child.stderr.readToEnd Task.Priority.dedicated

  -- Wait for process to complete
  let exitCode ← child.wait

  let stdoutContent ← IO.ofExcept stdout.get
  let stderrContent ← IO.ofExcept stderr.get

  -- Print stderr (diagnostics)
  if !stderrContent.trim.isEmpty then
    IO.println s!"[Launcher] {stderrContent}"

  if exitCode != 0 then
    IO.eprintln s!"[GPU] Launcher failed with exit code {exitCode}"
    throw <| IO.userError s!"GPU kernel execution failed"

  -- Parse output
  match parseLauncherOutput stdoutContent with
  | .ok results =>
      IO.println s!"[GPU] Execution complete! Retrieved {results.length} arrays"
      return results
  | .error msg =>
      IO.eprintln s!"[GPU] Failed to parse output: {msg}"
      IO.eprintln s!"[GPU] Output was: {stdoutContent}"
      throw <| IO.userError "Failed to parse launcher output"

/-- Execute kernel using KernelState interface (compatible with runKernelCPU) -/
def runKernelGPU_Process
    (kernel : Kernel)
    (grid block : Dim3)
    (initState : KernelState)
    : IO KernelState := do

  -- Extract scalar parameters from kernel
  let scalarParams := extractScalarParams kernel initState

  -- Extract global arrays
  let globalArrays := extractGlobalArrays kernel initState

  -- Execute on GPU
  let results ← executeKernel kernel grid block scalarParams globalArrays

  -- Update state with results
  let mut finalGlobals := initState.globals
  for (name, arr) in results do
    finalGlobals := finalGlobals.insert name (.arrayFloat arr)

  return { initState with globals := finalGlobals }

end CLean.GPU.ProcessLauncher
