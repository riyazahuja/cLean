/-
  Persistent GPU Server Client

  Communicates with a long-running GPU server process to avoid
  CUDA context initialization overhead on each kernel launch.
-/

import Lean
import CLean.GPU
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache
import CLean.GPU.ProcessLauncher

namespace CLean.GPU.ServerLauncher

open Lean (Name Json ToJson FromJson)
open DeviceIR
open CLean.DeviceCodeGen
open CLean.GPU.KernelCache
open CLean.GPU.ProcessLauncher (ScalarValue ArrayElementType GPUResult scalarToJson compileKernelToPTX)
open GpuDSL

/-- Server process state -/
structure ServerState where
  child : IO.Process.Child ⟨.piped, .piped, .piped⟩

/-- Global server process handle -/
initialize serverProcessRef : IO.Ref (Option ServerState) ← IO.mkRef none

/-- Start the GPU server if not already running -/
def ensureServerRunning : IO ServerState := do
  match ← serverProcessRef.get with
  | some state => return state
  | none =>
    IO.eprintln "[GPU Server] Starting persistent GPU server..."
    let child ← IO.Process.spawn {
      cmd := "./gpu_server"
      stdin := .piped
      stdout := .piped
      stderr := .piped
    }
    -- Wait for ready message
    let readyLine ← child.stdout.getLine
    if readyLine.trim != "{\"status\":\"ready\"}" then
      throw <| IO.userError s!"GPU server failed to start: {readyLine}"
    IO.eprintln "[GPU Server] Server ready"
    let state : ServerState := ⟨child⟩
    serverProcessRef.set (some state)
    return state

/-- Stop the GPU server -/
def stopServer : IO Unit := do
  match ← serverProcessRef.get with
  | none => return ()
  | some state =>
    state.child.stdin.putStrLn "{\"cmd\":\"quit\"}"
    state.child.stdin.flush
    let _ ← state.child.wait
    serverProcessRef.set none
    IO.eprintln "[GPU Server] Server stopped"

/-- Build JSON request for server -/
def buildServerRequest {α : Type} [ToJson α] [ArrayElementType α]
    (ptxPath : String)
    (kernelName : String)
    (grid block : Dim3)
    (scalarParams : Array ScalarValue)
    (arrays : List (Name × Array α))
    : String :=
  let scalarsJson := "[" ++ String.intercalate "," (scalarParams.map scalarToJson |>.toList) ++ "]"
  let arrayTypeName := ArrayElementType.typeName (α := α)
  let arraysJson := "{" ++ String.intercalate "," (arrays.map fun (name, a) =>
    "\"" ++ toString name ++ "\":{\"type\":\"" ++ arrayTypeName ++ "\",\"data\":" ++ (ToJson.toJson a).compress ++ "}") ++ "}"
  "{\"cmd\":\"launch\"," ++
  "\"ptx\":\"" ++ ptxPath ++ "\"," ++
  "\"kernel\":\"" ++ kernelName ++ "\"," ++
  "\"grid\":[" ++ toString grid.x ++ "," ++ toString grid.y ++ "," ++ toString grid.z ++ "]," ++
  "\"block\":[" ++ toString block.x ++ "," ++ toString block.y ++ "," ++ toString block.z ++ "]," ++
  "\"scalars\":" ++ scalarsJson ++ "," ++
  "\"arrays\":" ++ arraysJson ++ "}"

/-- Parse kernel time from server response -/
def parseServerKernelTime (json : Json) : Float :=
  match json.getObjVal? "kernel_time_ms" with
  | Except.ok (Json.num n) => n.toFloat
  | _ => 0.0

/-- Run kernel on persistent GPU server -/
unsafe def runKernelServer {α β} [ToJson α] [ArrayElementType α] [ProcessLauncher.ToScalarValue β]
    (IR : Kernel)
    (responseType : Type)
    [FromJson responseType]
    (grid block : Dim3)
    (scalarParams : Array β)
    (arrays : List (Name × Array α))
    (quiet : Bool := false)
    : IO (GPUResult responseType) := do
  let startTime ← IO.monoMsNow

  -- Ensure PTX is compiled
  let cached ← compileKernelToPTX IR

  -- Convert scalars
  let scalarParams' := scalarParams.map ProcessLauncher.ToScalarValue.toScalarValue

  -- Build request JSON
  let request := buildServerRequest cached.ptxPath.toString IR.name grid block scalarParams' arrays

  unless quiet do
    IO.println s!"[GPU Server] Launching {IR.name}..."

  -- Get server handle
  let state ← ensureServerRunning

  -- Send request
  state.child.stdin.putStrLn request
  state.child.stdin.flush

  -- Read response
  let response ← state.child.stdout.getLine

  let endTime ← IO.monoMsNow
  let totalTime := Float.ofNat (endTime - startTime)

  -- Parse response
  match Json.parse response with
  | Except.error err =>
    throw <| IO.userError s!"JSON Parse Error: {err}\nResponse: {response}"
  | Except.ok json =>
    let kernelTime := parseServerKernelTime json
    -- Extract just the results part for FromJson
    match json.getObjVal? "results" with
    | Except.error _ =>
      throw <| IO.userError s!"No results in response: {response}"
    | Except.ok resultsJson =>
      let wrappedJson := Json.mkObj [("results", resultsJson)]
      match @Lean.fromJson? responseType _ wrappedJson with
      | Except.error err =>
        throw <| IO.userError s!"JSON Decode Error: {err}\nResponse: {response}"
      | Except.ok result =>
        unless quiet do
          IO.println s!"[GPU Server] Kernel: {kernelTime} ms, Total: {totalTime} ms"
        return { result := result, kernelTimeMs := kernelTime, totalTimeMs := totalTime }

end CLean.GPU.ServerLauncher
