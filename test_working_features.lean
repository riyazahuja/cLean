/-
  Working Features Test

  Tests features that are fully working:
  - SharedArray operations
  - Barrier synchronization
  - GlobalArray operations
  - Binary operations
-/

import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR

open GpuDSL DeviceIR CLean.DeviceMacro

set_option maxHeartbeats 2000000

/-! ## Test 1: Matrix Transpose (SharedArray + Barrier) -/

namespace Transpose

kernelArgs TransposeArgs(N: Nat)
  global[input output: Array Float]
  shared[tile: Array Float]

device_kernel transpose : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.tile⟩

  let i ← globalIdxX

  -- Phase 1: Load into shared memory
  let val ← input.get i
  tile.set i val
  barrier

  -- Phase 2: Transpose and write out
  let transIdx := N * i
  let tileVal ← tile.get transIdx
  output.set i tileVal

#check transpose
#check transposeIR
#eval transposeIR

end Transpose

/-! ## Test 2: Multiple Barriers with Two Shared Arrays -/

namespace MultiBarrier

kernelArgs MultiBarrierArgs(N: Nat)
  global[input output: Array Float]
  shared[s1 s2: Array Float]

device_kernel multiBarrier : KernelM MultiBarrierArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let s1 : SharedArray Float := ⟨args.s1⟩
  let s2 : SharedArray Float := ⟨args.s2⟩

  let i ← globalIdxX

  -- Stage 1: Load to s1
  let val ← input.get i
  s1.set i val
  barrier

  -- Stage 2: Process s1 -> s2
  let v1 ← s1.get i
  let two := 2
  let doubled := v1 * two
  s2.set i doubled
  barrier

  -- Stage 3: Final output
  let v2 ← s2.get i
  output.set i v2

#check multiBarrier
#check multiBarrierIR
#eval multiBarrierIR

end MultiBarrier

/-! ## Test 3: Complex Arithmetic with Shared Memory -/

namespace ComplexArith

kernelArgs ArithArgs(alpha: Float, beta: Float)
  global[x y z output: Array Float]
  shared[temp: Array Float]

device_kernel complexArith : KernelM ArithArgs Unit := do
  let args ← getArgs
  let alpha := args.alpha
  let beta := args.beta
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let z : GlobalArray Float := ⟨args.z⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let temp : SharedArray Float := ⟨args.temp⟩

  let i ← globalIdxX

  -- Load and compute
  let xi ← x.get i
  let yi ← y.get i
  let zi ← z.get i

  -- Complex expression: alpha * xi + beta * yi - zi
  let term1 := alpha * xi
  let term2 := beta * yi
  let sum := term1 + term2
  let result := sum - zi

  -- Store to shared memory
  temp.set i result
  barrier

  -- Read from shared and write to output
  let finalVal ← temp.get i
  output.set i finalVal

#check complexArith
#check complexArithIR
#eval complexArithIR

end ComplexArith

/-! ## Test 4: Neighbor Access in Shared Memory -/

namespace NeighborAccess

kernelArgs NeighborArgs(N: Nat)
  global[input output: Array Float]
  shared[buffer: Array Float]

device_kernel neighborSum : KernelM NeighborArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let buffer : SharedArray Float := ⟨args.buffer⟩

  let i ← globalIdxX

  -- Load to shared memory
  let val ← input.get i
  buffer.set i val
  barrier

  -- Access neighbors (assuming bounds checking elsewhere)
  let left := i - 1
  let right := i + 1

  let centerVal ← buffer.get i
  let leftVal ← buffer.get left
  let rightVal ← buffer.get right

  -- Simple 3-point stencil
  let sum1 := centerVal + leftVal
  let sum := sum1 + rightVal

  output.set i sum

#check neighborSum
#check neighborSumIR
#eval neighborSumIR

end NeighborAccess
