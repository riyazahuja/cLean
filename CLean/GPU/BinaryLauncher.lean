/-
  Binary Protocol GPU Launcher

  High-performance GPU kernel execution using binary protocol.
  Avoids JSON serialization overhead for large arrays.

  Binary protocol is ~800x faster than JSON for small arrays
  and enables processing of arrays with millions of elements.
-/

import Lean
import CLean.GPU
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache

namespace CLean.GPU.BinaryLauncher

open Lean (Name)
open DeviceIR
open CLean.DeviceCodeGen
open CLean.GPU.KernelCache
open GpuDSL
open System (FilePath)

/-! ## FFI for IEEE 754 float bit manipulation -/

/-- Convert Float (double) to IEEE 754 float32 bits -/
@[extern "lean_float_to_float32_bits"]
opaque floatToFloat32Bits : Float → UInt32

/-- Create Float (double) from IEEE 754 float32 bits -/
@[extern "lean_float_from_float32_bits"]
opaque float32BitsToFloat : UInt32 → Float

-- Protocol constants (v2 - unified argument ordering)
def MAGIC : UInt32 := 0xCDA0CAFE
def CMD_LAUNCH : UInt8 := 0x01
def CMD_QUIT : UInt8 := 0xFF
def TYPE_INT_SCALAR : UInt8 := 0
def TYPE_FLOAT_SCALAR : UInt8 := 1
def TYPE_INT_ARRAY : UInt8 := 2
def TYPE_FLOAT_ARRAY : UInt8 := 3
def STATUS_OK : UInt8 := 0
def STATUS_ERROR : UInt8 := 1

/-- Kernel argument - can be scalar or array -/
inductive KernelArg where
  | intScalar : Int → KernelArg
  | floatScalar : Float → KernelArg
  | intArray : String → Array Int32 → KernelArg
  | floatArray : String → Array Float → KernelArg
  deriving Repr

/-- Scalar parameter with type info (kept for backwards compatibility) -/
inductive ScalarValue where
  | int : Int → ScalarValue
  | float : Float → ScalarValue
  deriving Repr

/-! ## Binary Buffer Operations -/

/-- Push UInt8 to ByteArray -/
def pushU8 (buf : ByteArray) (v : UInt8) : ByteArray :=
  buf.push v

/-- Push UInt16 little-endian to ByteArray -/
def pushU16LE (buf : ByteArray) (v : UInt16) : ByteArray :=
  buf.push (v &&& 0xFF).toUInt8
     |>.push ((v >>> 8) &&& 0xFF).toUInt8

/-- Push UInt32 little-endian to ByteArray -/
def pushU32LE (buf : ByteArray) (v : UInt32) : ByteArray :=
  buf.push (v &&& 0xFF).toUInt8
     |>.push ((v >>> 8) &&& 0xFF).toUInt8
     |>.push ((v >>> 16) &&& 0xFF).toUInt8
     |>.push ((v >>> 24) &&& 0xFF).toUInt8

/-- Push Int32 little-endian to ByteArray -/
def pushI32LE (buf : ByteArray) (v : Int32) : ByteArray :=
  pushU32LE buf v.toUInt32

/-- Push IEEE 754 binary32 float to ByteArray.
    Converts Lean's Float (64-bit double) to 32-bit single precision using FFI. -/
def pushF32 (buf : ByteArray) (v : Float) : ByteArray :=
  let bits32 := floatToFloat32Bits v
  pushU32LE buf bits32

/-- Push length-prefixed string to ByteArray -/
def pushStr (buf : ByteArray) (s : String) : ByteArray :=
  let bytes := s.toUTF8
  pushU16LE buf bytes.size.toUInt16 ++ bytes

/-- Push float array to ByteArray -/
def pushF32Array (buf : ByteArray) (arr : Array Float) : ByteArray := Id.run do
  let mut b := pushU32LE buf arr.size.toUInt32
  for v in arr do
    b := pushF32 b v
  return b

/-- Push int32 array to ByteArray -/
def pushI32Array (buf : ByteArray) (arr : Array Int32) : ByteArray := Id.run do
  let mut b := pushU32LE buf arr.size.toUInt32
  for v in arr do
    b := pushI32LE b v
  return b

/-! ## Binary Read Operations -/

/-- Read exactly n bytes from handle -/
def readExact (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut result := ByteArray.empty
  let mut remaining := n
  while remaining > 0 do
    let chunk ← h.read remaining
    if chunk.size == 0 then
      throw <| IO.userError s!"Unexpected EOF, needed {remaining} more bytes"
    result := result ++ chunk
    remaining := remaining - chunk.size
  return result

def readU8 (h : IO.FS.Handle) : IO UInt8 := do
  let bytes ← readExact h 1
  return bytes[0]!

def readU16LE (h : IO.FS.Handle) : IO UInt16 := do
  let bytes ← readExact h 2
  return bytes[0]!.toUInt16 ||| (bytes[1]!.toUInt16 <<< 8)

def readU32LE (h : IO.FS.Handle) : IO UInt32 := do
  let bytes ← readExact h 4
  return bytes[0]!.toUInt32
     ||| (bytes[1]!.toUInt32 <<< 8)
     ||| (bytes[2]!.toUInt32 <<< 16)
     ||| (bytes[3]!.toUInt32 <<< 24)

/-- Read IEEE 754 float32 and convert to Float (double) using FFI -/
def readF32 (h : IO.FS.Handle) : IO Float := do
  let bits32 ← readU32LE h
  return float32BitsToFloat bits32

def readStr (h : IO.FS.Handle) : IO String := do
  let len ← readU16LE h
  let bytes ← readExact h len.toNat
  return String.fromUTF8! bytes

def readF32Array (h : IO.FS.Handle) : IO (Array Float) := do
  let len ← readU32LE h
  let mut arr := Array.mkEmpty len.toNat
  for _ in [:len.toNat] do
    let v ← readF32 h
    arr := arr.push v
  return arr

/-! ## Server Management -/

/-- GPU Server connection state -/
structure ServerState where
  process : IO.Process.Child { stdin := .piped, stdout := .piped, stderr := .piped }
  serverPath : String

/-- Global server state reference -/
initialize serverStateRef : IO.Ref (Option ServerState) ← IO.mkRef none

/-- Start the binary GPU server -/
def startServer (serverPath : String := "./gpu_server_binary") : IO ServerState := do
  IO.println s!"[BinaryLauncher] Starting server: {serverPath}"
  let child ← IO.Process.spawn {
    cmd := serverPath
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }
  let state := { process := child, serverPath := serverPath }
  serverStateRef.set (some state)
  IO.sleep 100
  return state

/-- Get or start the server -/
def getServer (serverPath : String := "./gpu_server_binary") : IO ServerState := do
  match ← serverStateRef.get with
  | some s => return s
  | none => startServer serverPath

/-- Stop the server -/
def stopServer : IO Unit := do
  match ← serverStateRef.get with
  | some s =>
    let stdin := s.process.stdin
    let buf := pushU8 (pushU32LE ByteArray.empty MAGIC) CMD_QUIT
    stdin.write buf
    stdin.flush
    discard <| s.process.wait
    serverStateRef.set none
  | none => pure ()

/-! ## Kernel Launch -/

/-- Response from GPU server -/
structure LaunchResponse where
  success : Bool
  kernelTimeUs : UInt32
  arrays : List (String × Array Float)
  errorMsg : Option String

/-- Send a kernel launch request and get response -/
def sendLaunchRequest
    (state : ServerState)
    (ptxPath : String)
    (kernelName : String)
    (grid block : Dim3)
    (scalars : Array ScalarValue)
    (arrays : List (String × Array Float))
    : IO LaunchResponse := do
  let stdin := state.process.stdin
  let stdout := state.process.stdout

  -- Build request buffer
  let mut buf := ByteArray.empty
  buf := pushU32LE buf MAGIC
  buf := pushU8 buf CMD_LAUNCH
  buf := pushStr buf ptxPath
  buf := pushStr buf kernelName
  buf := pushU32LE buf grid.x.toUInt32
  buf := pushU32LE buf grid.y.toUInt32
  buf := pushU32LE buf grid.z.toUInt32
  buf := pushU32LE buf block.x.toUInt32
  buf := pushU32LE buf block.y.toUInt32
  buf := pushU32LE buf block.z.toUInt32

  -- Scalars
  buf := pushU32LE buf scalars.size.toUInt32
  for s in scalars do
    match s with
    | .int i =>
      buf := pushU8 buf TYPE_INT_SCALAR
      buf := pushI32LE buf i.toInt32
    | .float f =>
      buf := pushU8 buf TYPE_FLOAT_SCALAR
      buf := pushF32 buf f

  -- Arrays
  buf := pushU32LE buf arrays.length.toUInt32
  for (name, data) in arrays do
    buf := pushStr buf name
    buf := pushU8 buf TYPE_FLOAT_SCALAR  -- Response uses scalar type
    buf := pushF32Array buf data

  -- Send request
  stdin.write buf
  stdin.flush

  -- Read response
  let magic ← readU32LE stdout
  if magic != MAGIC then
    throw <| IO.userError s!"Invalid response magic: {magic}"

  let status ← readU8 stdout
  let kernelTimeUs ← readU32LE stdout
  let numArrays ← readU32LE stdout

  let mut resultArrays : List (String × Array Float) := []
  for _ in [:numArrays.toNat] do
    let name ← readStr stdout
    let _arrType ← readU8 stdout
    let data ← readF32Array stdout
    resultArrays := resultArrays ++ [(name, data)]

  if status == STATUS_ERROR then
    let errMsg ← readStr stdout
    return { success := false, kernelTimeUs, arrays := resultArrays, errorMsg := some errMsg }
  else
    return { success := true, kernelTimeUs, arrays := resultArrays, errorMsg := none }

/-- Send a kernel launch request using v2 protocol with unified argument list -/
def sendLaunchRequestV2
    (state : ServerState)
    (ptxPath : String)
    (kernelName : String)
    (grid block : Dim3)
    (args : Array KernelArg)
    : IO LaunchResponse := do
  let stdin := state.process.stdin
  let stdout := state.process.stdout

  -- Build request buffer
  let mut buf := ByteArray.empty
  buf := pushU32LE buf MAGIC
  buf := pushU8 buf CMD_LAUNCH
  buf := pushStr buf ptxPath
  buf := pushStr buf kernelName
  buf := pushU32LE buf grid.x.toUInt32
  buf := pushU32LE buf grid.y.toUInt32
  buf := pushU32LE buf grid.z.toUInt32
  buf := pushU32LE buf block.x.toUInt32
  buf := pushU32LE buf block.y.toUInt32
  buf := pushU32LE buf block.z.toUInt32

  -- All arguments in order
  buf := pushU32LE buf args.size.toUInt32
  for arg in args do
    match arg with
    | .intScalar i =>
      buf := pushU8 buf TYPE_INT_SCALAR
      buf := pushI32LE buf i.toInt32
    | .floatScalar f =>
      buf := pushU8 buf TYPE_FLOAT_SCALAR
      buf := pushF32 buf f
    | .intArray name data =>
      buf := pushU8 buf TYPE_INT_ARRAY
      buf := pushStr buf name
      buf := pushI32Array buf data
    | .floatArray name data =>
      buf := pushU8 buf TYPE_FLOAT_ARRAY
      buf := pushStr buf name
      buf := pushF32Array buf data

  -- Send request
  stdin.write buf
  stdin.flush

  -- Read response
  let magic ← readU32LE stdout
  if magic != MAGIC then
    throw <| IO.userError s!"Invalid response magic: {magic}"

  let status ← readU8 stdout
  let kernelTimeUs ← readU32LE stdout
  let numArrays ← readU32LE stdout

  let mut resultArrays : List (String × Array Float) := []
  for _ in [:numArrays.toNat] do
    let name ← readStr stdout
    let _arrType ← readU8 stdout
    let data ← readF32Array stdout
    resultArrays := resultArrays ++ [(name, data)]

  if status == STATUS_ERROR then
    let errMsg ← readStr stdout
    return { success := false, kernelTimeUs, arrays := resultArrays, errorMsg := some errMsg }
  else
    return { success := true, kernelTimeUs, arrays := resultArrays, errorMsg := none }

/-- Compile kernel to PTX -/
def compileKernelToPTX (kernel : Kernel) : IO CachedKernel := do
  let cached ← getCachedKernel kernel
  let ptxExists ← cached.ptxPath.pathExists

  if !ptxExists then
    IO.println s!"[GPU] Compiling {kernel.name} to PTX..."
    let args := #["-ptx", "-O3", "--gpu-architecture=compute_75",
                  "-o", cached.ptxPath.toString, cached.cudaSourcePath.toString]
    let nvccPath ← do
      let result ← IO.Process.output { cmd := "which", args := #["nvcc"] }
      if result.exitCode == 0 then pure "nvcc" else pure "/usr/local/cuda-12.5/bin/nvcc"
    let output ← IO.Process.run { cmd := nvccPath, args := args }
    if !output.trim.isEmpty then IO.eprintln s!"[nvcc] {output}"
    IO.println "[GPU] Compilation complete!"
  else
    IO.println s!"[GPU] Using cached PTX for {kernel.name}"

  return cached

/-- Run kernel on GPU using binary protocol -/
def runKernelBinary
    (kernel : Kernel)
    (grid block : Dim3)
    (scalars : Array ScalarValue)
    (arrays : List (Name × Array Float))
    (serverPath : String := "./gpu_server_binary")
    : IO (Float × List (Name × Array Float)) := do

  let cached ← compileKernelToPTX kernel
  let server ← getServer serverPath

  let stringArrays := arrays.map fun (name, data) => (toString name, data)

  let response ← sendLaunchRequest server
    cached.ptxPath.toString kernel.name grid block scalars stringArrays

  if !response.success then
    throw <| IO.userError s!"Kernel execution failed: {response.errorMsg.getD "unknown error"}"

  let resultArrays := response.arrays.map fun (name, data) => (Name.mkSimple name, data)
  let kernelTimeMs := response.kernelTimeUs.toFloat / 1000.0
  return (kernelTimeMs, resultArrays)

/-- Run kernel with detailed timing -/
def runKernelBinaryTimed
    (kernel : Kernel)
    (grid block : Dim3)
    (scalars : Array ScalarValue)
    (arrays : List (Name × Array Float))
    (serverPath : String := "./gpu_server_binary")
    : IO (Float × List (Name × Array Float)) := do

  let t0 ← IO.monoMsNow
  let cached ← compileKernelToPTX kernel
  let t1 ← IO.monoMsNow
  let server ← getServer serverPath
  let t2 ← IO.monoMsNow

  let stringArrays := arrays.map fun (name, data) => (toString name, data)
  let t3 ← IO.monoMsNow

  let response ← sendLaunchRequest server
    cached.ptxPath.toString kernel.name grid block scalars stringArrays
  let t4 ← IO.monoMsNow

  IO.println s!"[TIMING] PTX check: {t1 - t0} ms"
  IO.println s!"[TIMING] Server connect: {t2 - t1} ms"
  IO.println s!"[TIMING] Array prep: {t3 - t2} ms"
  IO.println s!"[TIMING] Round-trip: {t4 - t3} ms"
  IO.println s!"[TIMING] Kernel only: {response.kernelTimeUs.toFloat / 1000.0} ms"
  IO.println s!"[TIMING] TOTAL: {t4 - t0} ms"

  if !response.success then
    throw <| IO.userError s!"Kernel execution failed: {response.errorMsg.getD "unknown error"}"

  let resultArrays := response.arrays.map fun (name, data) => (Name.mkSimple name, data)
  let kernelTimeMs := response.kernelTimeUs.toFloat / 1000.0
  return (kernelTimeMs, resultArrays)

/-! ## Simplified raw CUDA interface for testing -/

/-- Compile raw CUDA code to PTX -/
def compileRawCudaToPTX (cudaCode : String) (kernelName : String) : IO FilePath := do
  let cacheDir := ".cuda_cache"
  let _ ← IO.Process.run { cmd := "mkdir", args := #["-p", cacheDir] }

  -- Use kernel name for the file
  let cudaPath : FilePath := cacheDir / s!"{kernelName}.cu"
  let ptxPath : FilePath := cacheDir / s!"{kernelName}.ptx"

  -- Check if we need to recompile
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

/-- Run raw CUDA kernel - simplified interface -/
def runRawKernel
    (cudaCode : String)
    (kernelName : String)
    (grid block : Dim3)
    (scalars : Array ScalarValue)
    (arrays : List (String × Array Float))
    (serverPath : String := "./gpu_server_binary")
    : IO (Float × List (String × Array Float)) := do

  let ptxPath ← compileRawCudaToPTX cudaCode kernelName
  let server ← getServer serverPath

  let response ← sendLaunchRequest server
    ptxPath.toString kernelName grid block scalars arrays

  if !response.success then
    throw <| IO.userError s!"Kernel execution failed: {response.errorMsg.getD "unknown error"}"

  let kernelTimeMs := response.kernelTimeUs.toFloat / 1000.0
  return (kernelTimeMs, response.arrays)

/-- Run raw CUDA kernel with v2 protocol (arguments in kernel signature order)
    This is the preferred interface for new code. -/
def runRawKernelV2
    (cudaCode : String)
    (kernelName : String)
    (grid block : Dim3)
    (args : Array KernelArg)
    (serverPath : String := "./gpu_server_binary_v2")
    : IO (Float × List (String × Array Float)) := do

  let ptxPath ← compileRawCudaToPTX cudaCode kernelName
  let server ← getServer serverPath

  let response ← sendLaunchRequestV2 server
    ptxPath.toString kernelName grid block args

  if !response.success then
    throw <| IO.userError s!"Kernel execution failed: {response.errorMsg.getD "unknown error"}"

  let kernelTimeMs := response.kernelTimeUs.toFloat / 1000.0
  return (kernelTimeMs, response.arrays)

end CLean.GPU.BinaryLauncher
