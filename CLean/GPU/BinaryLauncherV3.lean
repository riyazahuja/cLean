/-
  Binary Protocol GPU Launcher v3

  High-performance GPU kernel execution using binary protocol.
  Floats are sent as ASCII strings to avoid IEEE 754 bit manipulation issues.
  Integers are sent as raw binary for efficiency.

  This approach works around Lean's lack of Float bit manipulation
  while still being much faster than JSON for large arrays.
-/

import Lean
import CLean.GPU
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.GPU.KernelCache

namespace CLean.GPU.BinaryLauncherV3

open Lean (Name)
open DeviceIR
open CLean.DeviceCodeGen
open CLean.GPU.KernelCache
open GpuDSL
open System (FilePath)

-- Protocol constants (v3 - string-encoded floats)
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

/-- Push length-prefixed string to ByteArray -/
def pushStr (buf : ByteArray) (s : String) : ByteArray :=
  let bytes := s.toUTF8
  pushU16LE buf bytes.size.toUInt16 ++ bytes

/-- Push float as length-prefixed ASCII string -/
def pushFloatStr (buf : ByteArray) (v : Float) : ByteArray :=
  pushStr buf (toString v)

/-- Parse a float from string -/
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

/-- Push float array as length + string-encoded floats -/
def pushFloatArrayStr (buf : ByteArray) (arr : Array Float) : ByteArray := Id.run do
  let mut b := pushU32LE buf arr.size.toUInt32
  for v in arr do
    b := pushFloatStr b v
  return b

/-- Push int array as length + binary int32s -/
def pushI32Array (buf : ByteArray) (arr : Array Int32) : ByteArray := Id.run do
  let mut b := pushU32LE buf arr.size.toUInt32
  for v in arr do
    b := pushI32LE b v
  return b

/-! ## Binary Reading Operations -/

/-- Read UInt8 from ByteArray at offset -/
def readU8 (buf : ByteArray) (off : Nat) : Option (UInt8 × Nat) :=
  if off < buf.size then some (buf.get! off, off + 1)
  else none

/-- Read UInt16 LE from ByteArray -/
def readU16LE (buf : ByteArray) (off : Nat) : Option (UInt16 × Nat) :=
  if off + 2 ≤ buf.size then
    let lo := buf.get! off
    let hi := buf.get! (off + 1)
    some (lo.toUInt16 ||| (hi.toUInt16 <<< 8), off + 2)
  else none

/-- Read UInt32 LE from ByteArray -/
def readU32LE (buf : ByteArray) (off : Nat) : Option (UInt32 × Nat) :=
  if off + 4 ≤ buf.size then
    let b0 := buf.get! off
    let b1 := buf.get! (off + 1)
    let b2 := buf.get! (off + 2)
    let b3 := buf.get! (off + 3)
    let v := b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
    some (v, off + 4)
  else none

/-- Read Int32 LE from ByteArray -/
def readI32LE (buf : ByteArray) (off : Nat) : Option (Int32 × Nat) :=
  match readU32LE buf off with
  | some (v, off') => some (v.toInt32, off')
  | none => none

/-- Read length-prefixed string from ByteArray -/
def readStr (buf : ByteArray) (off : Nat) : Option (String × Nat) :=
  match readU16LE buf off with
  | some (len, off') =>
    let len' := len.toNat
    if off' + len' ≤ buf.size then
      let bytes := buf.extract off' (off' + len')
      some (String.fromUTF8! bytes, off' + len')
    else none
  | none => none

/-- Read float from length-prefixed ASCII string -/
def readFloatStr (buf : ByteArray) (off : Nat) : Option (Float × Nat) :=
  match readStr buf off with
  | some (s, off') =>
    match parseFloat s with
    | some f => some (f, off')
    | none => none
  | none => none

/-- Read int32 array from ByteArray -/
def readI32Array (buf : ByteArray) (off : Nat) : Option (Array Int32 × Nat) := do
  let (len, off') ← readU32LE buf off
  let len' := len.toNat
  if off' + len' * 4 > buf.size then
    failure
  else
    let mut arr : Array Int32 := #[]
    let mut o := off'
    for _ in [:len'] do
      if let some (v, o') := readI32LE buf o then
        arr := arr.push v
        o := o'
      else
        failure
    return (arr, o)

/-- Read float array from ByteArray (string-encoded) -/
def readFloatArrayStr (buf : ByteArray) (off : Nat) : Option (Array Float × Nat) := do
  let (len, off') ← readU32LE buf off
  let len' := len.toNat
  let mut arr : Array Float := #[]
  let mut o := off'
  for _ in [:len'] do
    if let some (v, o') := readFloatStr buf o then
      arr := arr.push v
      o := o'
    else
      failure
  return (arr, o)

/-! ## Request/Response Building -/

/-- Build kernel launch request with arguments in order -/
def buildRequest (
    ptxPath : String)
    (kernelName : String)
    (gridDim : Nat × Nat × Nat)
    (blockDim : Nat × Nat × Nat)
    (args : Array KernelArg)
    : ByteArray := Id.run do
  let mut buf := ByteArray.empty
  -- Magic and command
  buf := pushU32LE buf MAGIC
  buf := pushU8 buf CMD_LAUNCH
  -- PTX path and kernel name
  buf := pushStr buf ptxPath
  buf := pushStr buf kernelName
  -- Grid/block dimensions
  let (gx, gy, gz) := gridDim
  let (bx, by_, bz) := blockDim
  buf := pushU32LE buf gx.toUInt32
  buf := pushU32LE buf gy.toUInt32
  buf := pushU32LE buf gz.toUInt32
  buf := pushU32LE buf bx.toUInt32
  buf := pushU32LE buf by_.toUInt32
  buf := pushU32LE buf bz.toUInt32
  -- Number of arguments
  buf := pushU32LE buf args.size.toUInt32
  -- Arguments in order
  for arg in args do
    match arg with
    | .intScalar v =>
      buf := pushU8 buf TYPE_INT_SCALAR
      buf := pushI32LE buf (Int32.ofInt v)
    | .floatScalar v =>
      buf := pushU8 buf TYPE_FLOAT_SCALAR
      buf := pushFloatStr buf v
    | .intArray name data =>
      buf := pushU8 buf TYPE_INT_ARRAY
      buf := pushStr buf name
      buf := pushI32Array buf data
    | .floatArray name data =>
      buf := pushU8 buf TYPE_FLOAT_ARRAY
      buf := pushStr buf name
      buf := pushFloatArrayStr buf data
  return buf

/-- Result of parsing a kernel response -/
structure KernelResponse where
  success : Bool
  kernelTimeUs : UInt32
  arrays : Array (String × Bool × Array Float × Array Int32)  -- name, isFloat, floatData, intData
  errorMsg : Option String
  deriving Repr

/-- Parse kernel launch response -/
def parseResponse (buf : ByteArray) : Option KernelResponse := do
  let (magic, off) ← readU32LE buf 0
  if magic ≠ MAGIC then none
  let (status, off) ← readU8 buf off
  let (kernelTimeUs, off) ← readU32LE buf off
  let (numArrays, off) ← readU32LE buf off

  if status ≠ STATUS_OK then
    let (errMsg, _) ← readStr buf off
    return { success := false, kernelTimeUs, arrays := #[], errorMsg := some errMsg }

  let mut arrays : Array (String × Bool × Array Float × Array Int32) := #[]
  let mut o := off
  for _ in [:numArrays.toNat] do
    let (name, o') ← readStr buf o
    o := o'
    let (typ, o') ← readU8 buf o
    o := o'
    let isFloat := typ = TYPE_FLOAT_SCALAR
    if isFloat then
      let (len, o') ← readU32LE buf o
      o := o'
      let mut floatData : Array Float := #[]
      for _ in [:len.toNat] do
        let (v, o') ← readFloatStr buf o
        floatData := floatData.push v
        o := o'
      arrays := arrays.push (name, true, floatData, #[])
    else
      let (len, o') ← readU32LE buf o
      o := o'
      let mut intData : Array Int32 := #[]
      for _ in [:len.toNat] do
        let (v, o') ← readI32LE buf o
        intData := intData.push v
        o := o'
      arrays := arrays.push (name, false, #[], intData)
  return { success := true, kernelTimeUs, arrays, errorMsg := none }

/-! ## Process Management -/

/-- GPU server process handle -/
structure GpuServer where
  proc : IO.Process.Child ⟨.piped, .piped, .piped⟩

/-- Default path to the binary server executable -/
def defaultServerPath : String := "./gpu_server_binary_v3"

/-- Start the GPU server process -/
def startServer (serverPath : String := defaultServerPath) : IO GpuServer := do
  let child ← IO.Process.spawn {
    cmd := serverPath
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }
  -- Give server time to initialize
  IO.sleep 200
  return { proc := child }

/-- Stop the GPU server process -/
def stopServer (server : GpuServer) : IO Unit := do
  -- Send quit command
  let quitCmd := pushU32LE (pushU8 ByteArray.empty CMD_QUIT) MAGIC
  let stdin := server.proc.stdin
  stdin.write quitCmd
  stdin.flush
  -- Give it time to shut down gracefully
  IO.sleep 50

/-- Read all available data from server stdout -/
partial def readAllAvailable (stdout : IO.FS.Handle) (acc : ByteArray := ByteArray.empty) : IO ByteArray := do
  let chunk ← stdout.read 65536
  if chunk.isEmpty then return acc
  else readAllAvailable stdout (acc ++ chunk)

/-- Read exactly n bytes from server stdout -/
partial def readExactly (stdout : IO.FS.Handle) (n : Nat) (acc : ByteArray := ByteArray.empty) : IO ByteArray := do
  if acc.size ≥ n then return acc
  let needed := n - acc.size
  let chunk ← stdout.read (max needed 4096)
  if chunk.isEmpty then
    throw <| IO.userError s!"Unexpected EOF: got {acc.size} bytes, expected {n}"
  readExactly stdout n (acc ++ chunk)

/-- Execute a kernel on the GPU server -/
def executeKernel (server : GpuServer)
    (ptxPath : String) (kernelName : String)
    (gridDim : Nat × Nat × Nat) (blockDim : Nat × Nat × Nat)
    (args : Array KernelArg)
    : IO KernelResponse := do
  let req := buildRequest ptxPath kernelName gridDim blockDim args
  -- Send request
  let stdin := server.proc.stdin
  stdin.write req
  stdin.flush
  -- Read response
  let stdout := server.proc.stdout
  -- First read the header to know response size
  let header ← readExactly stdout 13  -- magic(4) + status(1) + time(4) + numArrays(4)
  -- Parse header
  let some (magic, _) := readU32LE header 0 | throw <| IO.userError "Invalid response magic"
  if magic ≠ MAGIC then throw <| IO.userError s!"Bad magic: {magic}"
  let some (_, _) := readU8 header 4 | throw <| IO.userError "Missing status"
  let some (_, _) := readU32LE header 5 | throw <| IO.userError "Missing kernel time"
  let some (_, _) := readU32LE header 9 | throw <| IO.userError "Missing array count"

  -- Read rest of response
  let restData ← readAllAvailable stdout
  let fullResponse := header ++ restData

  match parseResponse fullResponse with
  | some resp => return resp
  | none => throw <| IO.userError "Failed to parse response"

/-! ## High-Level API -/

/-- Result of a kernel execution -/
structure KernelResult where
  kernelTimeMs : Float
  outputs : Array (String × Bool × Array Float × Array Int32)
  deriving Repr

/-- Run a kernel with the binary launcher (starts/stops server per call) -/
def runKernel (ptxPath : String) (kernelName : String)
    (gridDim : Nat × Nat × Nat) (blockDim : Nat × Nat × Nat)
    (args : Array KernelArg)
    (serverPath : String := defaultServerPath)
    : IO KernelResult := do
  let server ← startServer serverPath
  try
    let resp ← executeKernel server ptxPath kernelName gridDim blockDim args
    if !resp.success then
      throw <| IO.userError s!"Kernel execution failed: {resp.errorMsg.getD "unknown error"}"
    return {
      kernelTimeMs := resp.kernelTimeUs.toFloat / 1000.0
      outputs := resp.arrays
    }
  finally
    stopServer server

/-- Convert KernelValue to KernelArg -/
def kernelValueToArg (name : String) (v : KernelValue) : KernelArg :=
  match v with
  | .int n => KernelArg.intScalar n
  | .nat n => KernelArg.intScalar (n : Int)
  | .float f => KernelArg.floatScalar f
  | .arrayInt arr => KernelArg.intArray name (arr.map Int32.ofInt)
  | .arrayNat arr => KernelArg.intArray name ((arr.map (Int.ofNat ·)).map Int32.ofInt)
  | .arrayFloat arr => KernelArg.floatArray name arr

/-- Build KernelArgs from argument list maintaining order -/
def buildArgs (argList : List (String × KernelValue)) : Array KernelArg := Id.run do
  let mut args : Array KernelArg := #[]
  for (name, v) in argList do
    args := args.push (kernelValueToArg name v)
  return args

/-- Simple test function -/
def testBinaryProtocol : IO Unit := do
  IO.println "Testing Binary Protocol v3 (string-encoded floats)..."

  -- Test float conversion
  let f : Float := 2.5
  let s := toString f
  IO.println s!"Float {f} -> string \"{s}\""

  let buf := pushFloatStr ByteArray.empty f
  IO.println s!"Buffer size: {buf.size}"

  match readFloatStr buf 0 with
  | some (v, _) => IO.println s!"Read back: {v}"
  | none => IO.println "Failed to read back!"

  -- Test array serialization
  let arr : Array Float := #[1.0, 2.5, 3.14159]
  let arrBuf := pushFloatArrayStr ByteArray.empty arr
  IO.println s!"Float array buffer size: {arrBuf.size}"

  match readFloatArrayStr arrBuf 0 with
  | some (readArr, _) => IO.println s!"Read back array: {readArr}"
  | none => IO.println "Failed to read back array!"

  IO.println "Test complete!"

end CLean.GPU.BinaryLauncherV3
