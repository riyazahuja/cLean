/-
  Advanced Device Macro Tests

  Tests for:
  - SharedArray operations
  - Barrier synchronization
  - If-then-else conditionals
  - For loops
  - All features combined
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

/-! ## Test 2: Parallel Reduction (For Loop + SharedArray + Barrier) -/

namespace Reduction

kernelArgs ReductionArgs(N: Nat)
  global[input output: Array Float]
  shared[temp: Array Float]

device_kernel reduce : KernelM ReductionArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let temp : SharedArray Float := ⟨args.temp⟩

  let tid ← globalIdxX

  -- Load into shared memory
  let val ← input.get tid
  temp.set tid val
  barrier

  -- Reduction loop: sum adjacent pairs
  for stride in [1:8] do
    let idx := stride * 2
    let a ← temp.get idx
    let b ← temp.get (idx + 1)
    let sum := a + b
    temp.set stride sum
    barrier

  -- Write final result
  let finalVal ← temp.get 0
  output.set tid finalVal

#check reduce
#check reduceIR
#eval reduceIR

end Reduction

/-! ## Test 3: Conditional Processing (If-Then-Else) -/

namespace Conditional

kernelArgs ConditionalArgs(threshold: Float)
  global[input output: Array Float]

device_kernel processConditional : KernelM ConditionalArgs Unit := do
  let args ← getArgs
  let threshold := args.threshold
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  let val ← input.get i

  -- Apply different processing based on threshold
  if val < threshold then do
    let doubled := val * 2.0
    output.set i doubled
  else do
    let halved := val / 2.0
    output.set i halved

#check processConditional
#check processConditionalIR
#eval processConditionalIR

end Conditional

/-! ## Test 4: Combined Features (All Together) -/

namespace Combined

kernelArgs CombinedArgs(N: Nat, threshold: Float)
  global[input output: Array Float]
  shared[buffer: Array Float]

device_kernel advancedKernel : KernelM CombinedArgs Unit := do
  let args ← getArgs
  let N := args.N
  let threshold := args.threshold
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let buffer : SharedArray Float := ⟨args.buffer⟩

  let i ← globalIdxX

  -- Load from global memory
  let val ← input.get i

  -- Conditional processing before shared memory
  if val < threshold then do
    let scaled := val * 10.0
    buffer.set i scaled
  else do
    buffer.set i val

  barrier

  -- Loop over neighbors in shared memory
  let mut sum := 0.0
  for offset in [0:3] do
    let idx := i + offset
    if idx < N then do
      let neighborVal ← buffer.get idx
      let newSum := sum + neighborVal
      sum := newSum
    else do
      sum := sum

  -- Normalize and write result
  let normalized := sum / 3.0
  output.set i normalized

#check advancedKernel
#check advancedKernelIR
#eval advancedKernelIR

end Combined

/-! ## Test 5: Nested Conditionals -/

namespace NestedIf

kernelArgs NestedArgs(low: Float, high: Float)
  global[input output: Array Float]

device_kernel nestedConditional : KernelM NestedArgs Unit := do
  let args ← getArgs
  let low := args.low
  let high := args.high
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩

  let i ← globalIdxX
  let val ← input.get i

  -- Nested conditionals
  if val < low then do
    output.set i 0.0
  else do
    if val < high then do
      let scaled := val * 2.0
      output.set i scaled
    else do
      output.set i 100.0

#check nestedConditional
#check nestedConditionalIR
#eval nestedConditionalIR

end NestedIf

/-! ## Test 6: Multiple Barriers -/

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
  let doubled := v1 * 2.0
  s2.set i doubled
  barrier

  -- Stage 3: Final output
  let v2 ← s2.get i
  output.set i v2

#check multiBarrier
#check multiBarrierIR
#eval multiBarrierIR

end MultiBarrier
