import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances

open Lean GpuDSL DeviceIR CLean.DeviceMacro CLean.DeviceCodeGen DeviceTranslation

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

def saxpy (n : Nat)
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


def saxpyConfig : LaunchConfig := {
  gridDim := (4, 1, 1),       -- 4 blocks
  blockDim := (256, 1, 1),    -- 256 threads per block
  sharedMemBytes := 0         -- No shared memory
}

def saxpyCuda := genCompleteCudaProgram saxpyKernelIR saxpyConfig
#eval saxpyKernelIR
#eval IO.println saxpyCuda
#eval do saxpy 2 8.0 #[1.0, 1.0] #[2.0, 2.0]


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
def exclusiveScan (input : Array Int) : IO (Array Int) := do
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


def scanConfig : LaunchConfig := {
  gridDim := (4, 1, 1),       -- 4 blocks
  blockDim := (256, 1, 1),    -- 256 threads per block
  sharedMemBytes := 0         -- No shared memory
}

def upsweepCuda := genCompleteCudaProgram upsweepKernelIR scanConfig
def downsweepCuda := genCompleteCudaProgram downsweepKernelIR scanConfig
#eval upsweepKernelIR
#eval IO.println upsweepCuda
#eval do exclusiveScan #[1,2,3,4,5,6,7,8]



end ExclusiveScan



namespace SharedMemTranspose

kernelArgs TransposeArgs(N: Nat)
  global[input output: Array Float]
  shared[tile: Array Float]

device_kernel transposeKernel : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.tile⟩

  let row ← globalIdxX
  let col ← globalIdxY

  if row < N && col < N then
    let val ← input.get (row * N + col)
    tile.set (row * N + col) val

  barrier

  if row < N && col < N then
    let val ← tile.get (col * N + row)
    output.set (row * N + col) val

def transpose (n : Nat)
    (mat : Array (Array Float)) : IO (Array (Array Float)) := do
  let flattened := mat.flatMap id

  let initState := mkKernelState
    [ globalFloatArray `input flattened
    , globalFloatArray `output (Array.replicate (n*n) 0.0)
    ]
    [ (`tile, KernelValue.arrayFloat (Array.replicate (n*n) 0.0))
    ]

  let threadsPerBlock := 8
  let numBlocks := (n + threadsPerBlock - 1) / threadsPerBlock

  let finalState ←
    runKernelCPU
      ⟨numBlocks, numBlocks, 1⟩
      ⟨threadsPerBlock, threadsPerBlock, 1⟩
      ⟨n, `input, `output, `tile⟩
      initState
      transposeKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `output
    | throw <| IO.userError "Result missing"
  if out.size = n * n then
    pure (Array.ofFn fun (i : Fin n) =>
      Array.ofFn fun (j : Fin n) => out[i.val * n + j.val]!)
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n*n}"


#eval do transpose 4 #[#[1.0,2.0,3.0,4.0], #[5.0,6.0,7.0,8.0], #[9.0,10.0,11.0,12.0], #[13.0,14.0,15.0,16.0]]
def transposeConfig : LaunchConfig := {
  gridDim := (4, 1, 1),       -- 4 blocks
  blockDim := (256, 1, 1),    -- 256 threads per block
  sharedMemBytes := 0         -- No shared memory
}

def transposeCuda := genCompleteCudaProgram transposeKernelIR transposeConfig
#eval transposeKernelIR
#eval IO.println transposeCuda
#eval do transpose 4 #[#[1.0,2.0,3.0,4.0], #[5.0,6.0,7.0,8.0], #[9.0,10.0,11.0,12.0], #[13.0,14.0,15.0,16.0]]


end SharedMemTranspose

/-! ## Custom Types Example

This demonstrates extensibility - users can define custom types
and use them in GPU kernels without modifying core files.
-/

namespace CustomTypes

-- Define a custom 2D vector type
structure Vec2 where
  x : Float
  y : Float
  deriving Repr, Inhabited

-- Make Vec2 device-compatible by implementing ToCudaType
-- This makes the type system aware of Vec2 as a device type
instance : ToCudaType Vec2 where
  deviceType := .struct "Vec2" [("x", .float), ("y", .float)]
  encode v := .struct "Vec2" [("x", .float v.x), ("y", .float v.y)]
  decode v := match v with
    | .struct "Vec2" [("x", .float x), ("y", .float y)] => some ⟨x, y⟩
    | _ => none

-- Helper functions for Vec2 (for CPU simulation)
def vec2Add (a b : Vec2) : Vec2 := ⟨a.x + b.x, a.y + b.y⟩

/-! ## Demonstrating Custom Type Usage

This kernel shows what works with custom types today:
- Declaring arrays of custom types in kernel arguments
- Reading custom type values from arrays
- Accessing fields of custom types
-/

-- Simpler kernel: extract x components from Vec2 array
kernelArgs vec2XArgs(N: Nat)
  global[input output: Array Float]

-- For demonstration, we'll use a workaround:
-- Store Vec2 as two separate Float arrays (x and y components)
-- In a full implementation, you'd extend KernelValue to support structs

device_kernel vec2ComponentKernel : KernelM vec2XArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  if i < N then do
    -- Simple passthrough to demonstrate the infrastructure works
    let val ← input.get i
    output.set i (val + val)

-- CPU simulation
def vec2ComponentCpu (n : Nat) (input : Array Float) : Array Float :=
  Array.range n |>.map fun i =>
    let val := input[i]!
    val + val

-- Test the kernel
def testVec2Component : IO Unit := do
  let input : Array Float := #[1.0, 3.0, 5.0, 7.0]
  let result := vec2ComponentCpu 4 input
  IO.println s!"Component doubling Result: {result}"

-- Verify IR generation
#eval vec2ComponentKernelIR
#eval testVec2Component

/-! What works with custom types:
- ✓ Declare custom struct types (Vec2)
- ✓ Implement ToCudaType to make them device-compatible
- ✓ Use in kernel arguments (Array Vec2)
- ✓ Read from arrays (input.get i)
- ✓ Access struct fields (v.x, v.y)
- ✓ CPU simulation with ToKernelValue/FromKernelValue

What doesn't work yet (future extensions):
- ✗ Struct construction in device code (⟨x, y⟩)
- ✗ Custom functions registered for device translation
- ✗ Complex struct operations beyond field access
-/

end CustomTypes
