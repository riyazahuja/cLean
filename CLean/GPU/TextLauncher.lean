/-
  Text Protocol GPU Launcher

  Simple text-based protocol for GPU kernel execution.
  Avoids IEEE 754 bit manipulation by using text representation of floats.
-/

import CLean.GPU

namespace CLean.GPU.TextLauncher

open GpuDSL (Dim3)
open System (FilePath)

/-- Kernel argument - can be scalar or array -/
inductive KernelArg where
  | intScalar : Int → KernelArg
  | floatScalar : Float → KernelArg
  | intArray : String → Array Int32 → KernelArg
  | floatArray : String → Array Float → KernelArg
  deriving Repr

/-- Server state -/
structure ServerState where
  process : IO.Process.Child ⟨.piped, .piped, .inherit⟩
  serverPath : String

initialize serverStateRef : IO.Ref (Option ServerState) ← IO.mkRef none

/-- Start the GPU server -/
def startServer (serverPath : String := "./gpu_server_text") : IO ServerState := do
  IO.println s!"[TextLauncher] Starting server: {serverPath}"
  let child ← IO.Process.spawn {
    cmd := serverPath
    stdin := .piped
    stdout := .piped
    stderr := .inherit
  }
  let state := { process := child, serverPath := serverPath }
  serverStateRef.set (some state)
  IO.sleep 100
  return state

/-- Get or start the server -/
def getServer (serverPath : String := "./gpu_server_text") : IO ServerState := do
  match ← serverStateRef.get with
  | some s => return s
  | none => startServer serverPath

/-- Stop the server -/
def stopServer : IO Unit := do
  match ← serverStateRef.get with
  | some s =>
    s.process.stdin.putStrLn "QUIT"
    s.process.stdin.flush
    discard <| s.process.wait
    serverStateRef.set none
  | none => pure ()

/-- Response from GPU server -/
structure LaunchResponse where
  success : Bool
  kernelTimeUs : UInt32
  arrays : List (String × Array Float)
  errorMsg : Option String

/-- Read a line from the server -/
def readLine (h : IO.FS.Handle) : IO String := do
  let mut result := ""
  let mut done := false
  while !done do
    let byte ← h.read 1
    if byte.size == 0 then
      done := true
    else
      let c := Char.ofNat byte[0]!.toNat
      if c == '\n' then
        done := true
      else
        result := result.push c
  return result

/-- Parse a float array response line -/
def parseArrayLine (line : String) : Option (String × Array Float) := do
  let parts := line.splitOn " "
  if parts.length < 3 then none
  if parts[0]! != "ARRAY" then none
  let name := parts[1]!
  let count := parts[2]!.toNat!
  let mut arr := Array.mkEmpty count
  for i in [3:3+count] do
    if h : i < parts.length then
      match parts[i].toFloat? with
      | some f => arr := arr.push f
      | none => return none
    else
      return none
  return (name, arr)

/-- Send a kernel launch request and get response -/
def sendLaunchRequest
    (state : ServerState)
    (ptxPath : String)
    (kernelName : String)
    (grid block : Dim3)
    (args : Array KernelArg)
    : IO LaunchResponse := do
  let stdin := state.process.stdin
  let stdout := state.process.stdout

  -- Build launch command
  let launchCmd := s!"LAUNCH {ptxPath} {kernelName} {grid.x} {grid.y} {grid.z} {block.x} {block.y} {block.z} {args.size}"
  stdin.putStrLn launchCmd

  -- Send each argument
  for arg in args do
    match arg with
    | .intScalar i =>
      stdin.putStrLn s!"SCALAR_INT {i}"
    | .floatScalar f =>
      stdin.putStrLn s!"SCALAR_FLOAT {f}"
    | .intArray name data =>
      let values := " ".intercalate (data.toList.map toString)
      stdin.putStrLn s!"ARRAY_INT {name} {data.size} {values}"
    | .floatArray name data =>
      let values := " ".intercalate (data.toList.map toString)
      stdin.putStrLn s!"ARRAY_FLOAT {name} {data.size} {values}"

  stdin.flush

  -- Read response
  let responseLine ← readLine stdout
  let parts := responseLine.splitOn " "

  if parts[0]! == "ERROR" then
    let errMsg := " ".intercalate (parts.drop 1)
    return { success := false, kernelTimeUs := 0, arrays := [], errorMsg := some errMsg }

  if parts[0]! != "OK" then
    return { success := false, kernelTimeUs := 0, arrays := [], errorMsg := some s!"Unexpected response: {responseLine}" }

  let kernelTimeUs := parts[1]!.toNat!.toUInt32
  let numArrays := parts[2]!.toNat!

  let mut resultArrays : List (String × Array Float) := []
  for _ in [:numArrays] do
    let arrayLine ← readLine stdout
    match parseArrayLine arrayLine with
    | some arr => resultArrays := resultArrays ++ [arr]
    | none => return { success := false, kernelTimeUs, arrays := [], errorMsg := some s!"Failed to parse array: {arrayLine}" }

  return { success := true, kernelTimeUs, arrays := resultArrays, errorMsg := none }

/-- Compile raw CUDA code to PTX -/
def compileRawCudaToPTX (cudaCode : String) (kernelName : String) : IO FilePath := do
  let cacheDir := ".cuda_cache"
  let _ ← IO.Process.run { cmd := "mkdir", args := #["-p", cacheDir] }

  let cudaPath : FilePath := cacheDir / s!"{kernelName}.cu"
  let ptxPath : FilePath := cacheDir / s!"{kernelName}.ptx"

  let ptxExists ← ptxPath.pathExists
  let cudaChanged ← if ptxExists then do
    let existing ← IO.FS.readFile cudaPath
    pure (existing != cudaCode)
  else
    pure true

  if cudaChanged then
    IO.FS.writeFile cudaPath cudaCode
    let nvccPath := "/usr/local/cuda-12.5/bin/nvcc"
    let args := #["-ptx", "-O3", "--gpu-architecture=compute_75",
                  "-o", ptxPath.toString, cudaPath.toString]
    let output ← IO.Process.run { cmd := nvccPath, args := args }
    if !output.trim.isEmpty then IO.eprintln s!"[nvcc] {output}"

  return ptxPath

/-- Run raw CUDA kernel with text protocol -/
def runKernel
    (cudaCode : String)
    (kernelName : String)
    (grid block : Dim3)
    (args : Array KernelArg)
    (serverPath : String := "./gpu_server_text")
    : IO (Float × List (String × Array Float)) := do

  let ptxPath ← compileRawCudaToPTX cudaCode kernelName
  let server ← getServer serverPath

  let response ← sendLaunchRequest server ptxPath.toString kernelName grid block args

  if !response.success then
    throw <| IO.userError s!"Kernel execution failed: {response.errorMsg.getD "unknown error"}"

  let kernelTimeMs := response.kernelTimeUs.toFloat / 1000.0
  return (kernelTimeMs, response.arrays)

end CLean.GPU.TextLauncher
