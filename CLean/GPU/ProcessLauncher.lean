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

/-- Scalar parameter with type info -/
inductive ScalarValue where
  | int : Int → ScalarValue
  | float : Float → ScalarValue
  deriving Repr

/-- Convert ScalarValue to JSON with type tag -/
def scalarToJson : ScalarValue → String
  | .int i => "{\"type\":\"int\",\"value\":" ++ toString i ++ "}"
  | .float f => "{\"type\":\"float\",\"value\":" ++ toString f ++ "}"

open Lean Json in
instance : ToJson ScalarValue where
  toJson sv := match sv with
    | .int i => Json.mkObj [("type", Json.str "int"), ("value", Json.num (i))]
    | .float f => Json.mkObj [("type", Json.str "float"), ("value", ToJson.toJson f)]

class ToScalarValue (α : Type) where
  toScalarValue : α → ScalarValue

instance : ToScalarValue Float where
  toScalarValue f := ScalarValue.float f
instance : ToScalarValue Int where
  toScalarValue i := ScalarValue.int i
instance : ToScalarValue Nat where
  toScalarValue n := ScalarValue.int (Int.ofNat n)
instance : ToScalarValue ScalarValue where
  toScalarValue sv := sv



/-- Build JSON payload for launcher with explicit type info -/
def buildLauncherInputTyped (scalarParams : Array ScalarValue) (arrays : List (Name × Array Float)) : String :=
  let scalarsJson := "[" ++ String.intercalate "," (scalarParams.map scalarToJson |>.toList) ++ "]"
  let arraysJson := String.intercalate "," (arrays.map fun (name, arr) =>
    -- Arrays are always float for now, with explicit type tag
    "\"" ++ toString name ++ "\":{\"type\":\"float\",\"data\":" ++ floatArrayToJson arr ++ "}")
  "{\"scalars\":" ++ scalarsJson ++ ",\"arrays\":{" ++ arraysJson ++ "}}"

/-- Build JSON payload for launcher (legacy, infers types from format) -/
def buildLauncherInput {α : Type} [ToString α] (scalarParams : Array α) (arrays : List (Name × Array α)) : String :=
  let scalarsJson := floatArrayToJson scalarParams
  let arraysJson := String.intercalate "," (arrays.map fun (name, arr) =>
    namedArrayToJson (toString name) arr)

  "{\"scalars\":" ++ scalarsJson ++ ",\"arrays\":{" ++ arraysJson ++ "}}"

class ArrayElementType (α : Type) where
  typeName : String

instance : ArrayElementType Float where
  typeName := "float"

instance : ArrayElementType Int where
  typeName := "int"

open Lean Json in
def buildLauncherInputBetter {α : Type} [ToJson α] [ArrayElementType α] (scalarParams : Array ScalarValue) (arrays : List (Name × Array α)) : String :=
  let scalarsJson := (ToJson.toJson scalarParams).compress
  -- Build arrays JSON manually to preserve order (Json.mkObj doesn't preserve order)
  -- Include type info to avoid integer/float inference issues
  let arrayTypeName := ArrayElementType.typeName (α := α)
  let arraysJson := "{" ++ String.intercalate "," (arrays.map fun (name, a) =>
    "\"" ++ toString name ++ "\":{\"type\":\"" ++ arrayTypeName ++ "\",\"data\":" ++ (ToJson.toJson a).compress ++ "}") ++ "}"
  "{\"scalars\":" ++ scalarsJson ++ ",\"arrays\":" ++ arraysJson ++ "}"

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

    -- Compile using nvcc - try "nvcc" first, fallback to full path
    let args := #[
      "-ptx",
      "-O3",
      "--gpu-architecture=compute_75",
      "-o", cached.ptxPath.toString,
      cached.cudaSourcePath.toString
    ]

    -- Try nvcc from PATH first, fallback to full path if not found
    let nvccPath ← do
      let result ← IO.Process.output { cmd := "which", args := #["nvcc"] }
      if result.exitCode == 0 then
        pure "nvcc"
      else
        pure "/usr/local/cuda-12.5/bin/nvcc"

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

  -- IO.println s!"[GPU] Launching {kernel.name}..."
  -- IO.println s!"[GPU] Grid: {grid.x}x{grid.y}x{grid.z}"
  -- IO.println s!"[GPU] Block: {block.x}x{block.y}x{block.z}"

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
  -- let stderrContent ← IO.ofExcept stderr.get

  -- Print stderr (diagnostics)
  -- if !stderrContent.trim.isEmpty then
    -- IO.println s!"[Launcher] {stderrContent}"

  if exitCode != 0 then
    -- IO.eprintln s!"[GPU] Launcher failed with exit code {exitCode}"
    throw <| IO.userError s!"GPU kernel execution failed"

  -- Parse output
  match parseLauncherOutput stdoutContent with
  | .ok results =>
      -- IO.println s!"[GPU] Execution complete! Retrieved {results.length} arrays"
      return results
  | .error msg =>
      IO.eprintln s!"[GPU] Failed to parse output: {msg}"
      IO.eprintln s!"[GPU] Output was: {stdoutContent}"
      throw <| IO.userError "Failed to parse launcher output"

/-- Check if needle is a substring of haystack -/
def stringContains (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

/-- Parse a float from string (simple implementation) -/
def parseFloat (s : String) : Option Float := do
  -- Handle negative numbers
  let (neg, s) := if s.startsWith "-" then (true, s.drop 1) else (false, s)
  let parts := s.splitOn "."
  if parts.length == 1 then
    -- Integer
    let n ← parts[0]!.toNat?
    let f := Float.ofNat n
    return if neg then -f else f
  else if parts.length == 2 then
    -- Decimal
    let intPart ← parts[0]!.toNat?
    let fracStr := parts[1]!
    let fracPart ← fracStr.toNat?
    let divisor := Float.ofNat (10 ^ fracStr.length)
    let f := Float.ofNat intPart + Float.ofNat fracPart / divisor
    return if neg then -f else f
  else
    none

/-- Extract a time value from a line like "[Launcher] X time: 1.234 ms" -/
def extractTimeFromLine (line : String) : Float :=
  let parts := line.splitOn ":"
  if parts.length >= 2 then
    match parts.get? 1 with
    | some timePart =>
      let numStr := timePart.trim.replace " ms" ""
      match parseFloat numStr with
      | some f => f
      | none => 0.0
    | none => 0.0
  else 0.0

/-- Extract kernel execution time from stderr -/
def parseKernelTime (stderr : String) : Float := Id.run do
  -- Look for "[Launcher] Kernel execution time: X.XXX ms"
  let lines := stderr.splitOn "\n"
  for line in lines do
    if stringContains line "Kernel execution time:" then
      return extractTimeFromLine line
  return 0.0

/-- Detailed timing breakdown from GPU execution -/
structure DetailedTiming where
  h2dTransferMs : Float := 0.0
  kernelExecutionMs : Float := 0.0
  d2hTransferMs : Float := 0.0
  jsonSerializeMs : Float := 0.0
  processSpawnMs : Float := 0.0
  jsonParseMs : Float := 0.0
  deriving Repr, Inhabited

open Lean Json in
instance : ToJson DetailedTiming where
  toJson t := Json.mkObj [
    ("h2dTransferMs", toJson t.h2dTransferMs),
    ("kernelExecutionMs", toJson t.kernelExecutionMs),
    ("d2hTransferMs", toJson t.d2hTransferMs),
    ("jsonSerializeMs", toJson t.jsonSerializeMs),
    ("processSpawnMs", toJson t.processSpawnMs),
    ("jsonParseMs", toJson t.jsonParseMs)
  ]

open Lean Json in
instance : FromJson DetailedTiming where
  fromJson? json := do
    let h2d ← (json.getObjValAs? Float "h2dTransferMs").toOption.getD 0.0 |> pure
    let kernel ← (json.getObjValAs? Float "kernelExecutionMs").toOption.getD 0.0 |> pure
    let d2h ← (json.getObjValAs? Float "d2hTransferMs").toOption.getD 0.0 |> pure
    let jsonSer ← (json.getObjValAs? Float "jsonSerializeMs").toOption.getD 0.0 |> pure
    let spawn ← (json.getObjValAs? Float "processSpawnMs").toOption.getD 0.0 |> pure
    let jsonParse ← (json.getObjValAs? Float "jsonParseMs").toOption.getD 0.0 |> pure
    return { h2dTransferMs := h2d, kernelExecutionMs := kernel, d2hTransferMs := d2h,
             jsonSerializeMs := jsonSer, processSpawnMs := spawn, jsonParseMs := jsonParse }

/-- Parse detailed timing from stderr (H2D, kernel, D2H) -/
def parseDetailedTimingFromStderr (stderr : String) : DetailedTiming := Id.run do
  let mut timing : DetailedTiming := {}
  let lines := stderr.splitOn "\n"
  for line in lines do
    if stringContains line "H2D transfer time:" then
      timing := { timing with h2dTransferMs := extractTimeFromLine line }
    else if stringContains line "Kernel execution time:" then
      timing := { timing with kernelExecutionMs := extractTimeFromLine line }
    else if stringContains line "D2H transfer time:" then
      timing := { timing with d2hTransferMs := extractTimeFromLine line }
  return timing

/-- Result of GPU execution including timing info -/
structure GPUResult (α : Type) where
  result : α
  kernelTimeMs : Float    -- Kernel-only time from CUDA events
  totalTimeMs : Float     -- Total wall clock time (for reference)
  detailedTiming : Option DetailedTiming := none
  deriving Repr

open Lean Json
def runKernelGPU {α β} [ToJson α] [ArrayElementType α] [ToScalarValue β]
    (IR : Kernel)
    (responseType : Type)
    [FromJson responseType]
    (grid block : Dim3)
    (scalarParams : Array β)
    (arrays : List (Name × Array α))
    (quiet : Bool := false)
    : IO responseType := do
  let cached ← compileKernelToPTX IR
  let scalarParams' := scalarParams.map ToScalarValue.toScalarValue
  let jsonInput := buildLauncherInputBetter scalarParams' arrays
  unless quiet do
    IO.println s!"[GPU] Launching {IR.name}..."
    IO.println s!"[GPU] Grid: {grid.x}x{grid.y}x{grid.z}"
    IO.println s!"[GPU] Block: {block.x}x{block.y}x{block.z}"
    IO.println s!"[GPU] Input JSON: {jsonInput}"
  let launcherArgs := #[
    cached.ptxPath.toString,
    IR.name,
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
  unless quiet do
    IO.println s!"[GPU] Stdout: {stdoutContent}"
    IO.println s!"[GPU] Stderr: {stderrContent}"

  if exitCode == 0 then
    match Lean.Json.parse stdoutContent with
    | Except.error err =>
      throw <| IO.userError s!"JSON Parse Error: {err} \n Output was:\n{stdoutContent} \n Stderr was:\n{stderrContent}"
    | Except.ok json =>
      match @Lean.fromJson? responseType _ json with
      | Except.error err =>
        throw <| IO.userError s!"JSON Decode Error: {err} \n Output was:\n{stdoutContent} \n Stderr was:\n{stderrContent}"
      | Except.ok response =>
        return response
  else
    throw <| IO.userError s!"GPU execution failed: {stderrContent} \n Output was:\n{stdoutContent}"

/-- Run GPU kernel and return result with timing info -/
def runKernelGPUTimed {α β} [ToJson α] [ArrayElementType α] [ToScalarValue β]
    (IR : Kernel)
    (responseType : Type)
    [FromJson responseType]
    (grid block : Dim3)
    (scalarParams : Array β)
    (arrays : List (Name × Array α))
    (quiet : Bool := false)
    (debug : Bool := false)
    : IO (GPUResult responseType) := do
  let startTime ← IO.monoMsNow

  -- Step 1: PTX compilation (usually cached)
  let t1 ← IO.monoMsNow
  let cached ← compileKernelToPTX IR
  let t2 ← IO.monoMsNow

  -- Step 2: Scalar conversion
  let scalarParams' := scalarParams.map ToScalarValue.toScalarValue
  let t3 ← IO.monoMsNow

  -- Step 3: JSON serialization (THE BOTTLENECK)
  let jsonInput := buildLauncherInputBetter scalarParams' arrays
  let t4 ← IO.monoMsNow

  if debug then
    IO.println s!"[TIMING] PTX check: {t2 - t1} ms"
    IO.println s!"[TIMING] Scalar convert: {t3 - t2} ms"
    IO.println s!"[TIMING] JSON serialize: {t4 - t3} ms"
    IO.println s!"[TIMING] JSON length: {jsonInput.length} chars"

  unless quiet do
    IO.println s!"[GPU] Launching {IR.name}..."
    IO.println s!"[GPU] Grid: {grid.x}x{grid.y}x{grid.z}"
    IO.println s!"[GPU] Block: {block.x}x{block.y}x{block.z}"
  let launcherArgs := #[
    cached.ptxPath.toString,
    IR.name,
    toString grid.x, toString grid.y, toString grid.z,
    toString block.x, toString block.y, toString block.z
  ]

  -- Step 4: Process spawn
  let t5 ← IO.monoMsNow
  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := launcherArgs
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }
  let t6 ← IO.monoMsNow

  -- Step 5: Write to stdin
  let stdin := child.stdin
  stdin.putStr jsonInput
  stdin.putStr "\n"
  stdin.flush
  let t7 ← IO.monoMsNow

  -- Step 6: Read results
  let stderrContent ← child.stderr.readToEnd
  let stdoutContent ← child.stdout.readToEnd
  let exitCode ← child.wait
  let t8 ← IO.monoMsNow

  if debug then
    IO.println s!"[TIMING] Process spawn: {t6 - t5} ms"
    IO.println s!"[TIMING] Write stdin: {t7 - t6} ms"
    IO.println s!"[TIMING] Read results: {t8 - t7} ms"
    IO.println s!"[TIMING] Output length: {stdoutContent.length} chars"

  let endTime ← IO.monoMsNow
  let totalTime := Float.ofNat (endTime - startTime)
  let kernelTime := parseKernelTime stderrContent

  unless quiet do
    IO.println s!"[GPU] Kernel time: {kernelTime} ms, Total time: {totalTime} ms"

  if exitCode == 0 then
    -- Step 7: Parse JSON output
    let t9 ← IO.monoMsNow
    match Lean.Json.parse stdoutContent with
    | Except.error err =>
      throw <| IO.userError s!"JSON Parse Error: {err}"
    | Except.ok json =>
      let t10 ← IO.monoMsNow
      match @Lean.fromJson? responseType _ json with
      | Except.error err =>
        throw <| IO.userError s!"JSON Decode Error: {err}"
      | Except.ok response =>
        let t11 ← IO.monoMsNow
        if debug then
          IO.println s!"[TIMING] JSON parse output: {t10 - t9} ms"
          IO.println s!"[TIMING] FromJson decode: {t11 - t10} ms"
          IO.println s!"[TIMING] TOTAL: {t11 - startTime} ms"
        -- Parse GPU-side timing from stderr
        let gpuTiming := parseDetailedTimingFromStderr stderrContent
        -- Combine with Lean-side timing
        let detailedTiming : DetailedTiming := {
          gpuTiming with
          jsonSerializeMs := Float.ofNat (t4 - t3)
          processSpawnMs := Float.ofNat (t6 - t5)
          jsonParseMs := Float.ofNat (t10 - t9)
        }
        return { result := response, kernelTimeMs := kernelTime, totalTimeMs := totalTime,
                 detailedTiming := some detailedTiming }
  else
    throw <| IO.userError s!"GPU execution failed: {stderrContent}"



end CLean.GPU.ProcessLauncher
