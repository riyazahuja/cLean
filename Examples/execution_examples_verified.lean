import CLean.GPU
import CLean.DeviceMacro
import CLean.DeviceIR
import CLean.DeviceCodeGen
import CLean.DeviceTranslation
import CLean.DeviceInstances
import CLean.GPU.ProcessLauncher

import CLean.ToGPUVerifyIR
import CLean.Verification.GPUVerifyStyle
import CLean.Verification.Semantics
import CLean.Verification.Decomposition

open CLean.ToGPUVerifyIR
open CLean.Verification.GPUVerify
open CLean.Verification
open Lean GpuDSL DeviceIR CLean.DeviceMacro CLean.DeviceCodeGen DeviceTranslation CLean.GPU.ProcessLauncher Json


set_option maxHeartbeats 2000000

namespace Saxpy

kernelArgs saxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

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

def saxpySpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec saxpyKernelIR config grid

theorem saxpy_safe : ∀ (config grid : Dim3), KernelSafe (saxpySpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [HasRace, saxpySpec, deviceIRToKernelSpec, saxpyKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  . unfold BarrierUniform; intros; trivial
def saxpyCPU (n : Nat)
    (α : Float)
    (x y : Array Float) : IO (Array Float) := do


  let initState := mkKernelState [
    globalFloatArray `X x,
    globalFloatArray `Y y,
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

  let scalarParams := #[Float.ofNat n, α]
  let arrays := [
    (`x, x),
    (`y, y),
    (`r, Array.replicate n 0.0)
  ]

  let response ← runKernelGPU saxpyKernelIR saxpyArgsResponse
    ⟨(n + 511) / 512, 1, 1⟩         -- grid
    ⟨512, 1, 1⟩                     -- block
    scalarParams
    arrays
  return response.r


#eval do saxpyCPU 2 8.0 #[1.0, 1.0] #[3.0,3.0]
#eval do saxpyGPU 2 8.0 #[1.0, 1.0] #[3.0, 3.0]



def saxpyInitMem (x y : Array Rat) (N : Nat) : VMem :=
  fun arr idx =>
    if arr = "x" then
      if h : idx < x.size then VVal.rat x[idx] else VVal.rat 0
    else if arr = "y" then
      if h : idx < y.size then VVal.rat y[idx] else VVal.rat 0
    else if arr = "r" then
      VVal.rat 0
    else
      VVal.int 0

/-- Execute a single thread of SAXPY and show what it computes -/
theorem saxpy_thread_computes (N : Nat) (α : Rat) (x y : Array Rat)
    (blockSize numBlocks : Nat)
    (tid bid : Nat)
    (h_tid : tid < blockSize)
    (h_bid : bid < numBlocks)
    (h_gid : globalIdx1D blockSize bid tid < N)
    (h_x : x.size ≥ N)
    (h_y : y.size ≥ N)
    (mem₀ : VMem)
    (h_mem_x : ∀ i, i < x.size → mem₀ "x" i = VVal.rat x[i]!)
    (h_mem_y : ∀ i, i < y.size → mem₀ "y" i = VVal.rat y[i]!) :
    let gid := globalIdx1D blockSize bid tid
    let params : String → VVal := fun name =>
      if name = "N" then VVal.nat N
      else if name = "alpha" then VVal.rat α
      else VVal.int 0
    let mem_after := vExecThread1DWithParams saxpyKernelIR.body blockSize numBlocks tid bid params mem₀
    mem_after.getR "r" gid = α * x[gid]! + y[gid]! := by

  intro gid params mem_after
  have h_cond : bid * blockSize + tid < N := by simp only [globalIdx1D] at h_gid; exact h_gid
  have h_gid_lt_x : bid * blockSize + tid < x.size := by omega
  have h_gid_lt_y : bid * blockSize + tid < y.size := by omega

  simp only [mem_after, vExecThread1DWithParams, vExecThreadWithLocals, saxpyKernelIR,
    vEvalStmt, vEvalExpr, vEvalBinOp, VCtx.setLocal, VMem.getR, gid, VCtx.mk', VCtx.getLocal,
    globalIdx1D, VVal.toRat, VMem.get, VVal.toBool, VVal.toInt, VVal.toNat,
    VMem.set, params, h_cond, ↓reduceIte]

  simp_all [h_mem_x (bid * blockSize + tid) h_gid_lt_x,
            h_mem_y (bid * blockSize + tid) h_gid_lt_y,
            VMem.set
          ]


/-- Main functional correctness theorem for SAXPY

    Given:
    - Arrays x, y of size at least N
    - Initial memory with x, y, and zeroed output r
    - A grid/block configuration that covers all N elements

    Then: After executing the kernel, r[i] = α * x[i] + y[i] for all i < N
-/
theorem saxpy_correct (N : Nat) (α : Rat) (x y : Array Rat)
    (numBlocks blockSize : Nat)
    (h_x : x.size ≥ N)
    (h_y : y.size ≥ N)
    (h_cover : N ≤ numBlocks * blockSize)
    (h_blockSize_pos : blockSize > 0) :
    let mem₀ := saxpyInitMem x y N
    let params : String → VVal := fun name =>
      if name = "N" then VVal.nat N
      else if name = "alpha" then VVal.rat α
      else VVal.int 0
    let mem_f := vExecKernel1DWithParams saxpyKernelIR.body numBlocks blockSize params mem₀
    ∀ i, i < N → mem_f.getR "r" i = α * x[i]! + y[i]! := by
  intro mem₀ params mem_f i hi

  -- Memory hypotheses
  have h_mem_x : ∀ j, j < x.size → mem₀ "x" j = VVal.rat x[j]! := by
    intro j hj
    simp only [mem₀, saxpyInitMem, ↓reduceIte, hj, dite_true]
    simp only [getElem!_pos, hj]

  have h_mem_y : ∀ j, j < y.size → mem₀ "y" j = VVal.rat y[j]! := by
    intro j hj
    simp only [mem₀, saxpyInitMem, String.reduceEq, ↓reduceIte, hj, dite_true]
    simp only [getElem!_pos, hj]

  -- Use the thread correctness lemma!

  -- helpers
  have h_tid : i % blockSize < blockSize := Nat.mod_lt i h_blockSize_pos
  have h_bid : i / blockSize < numBlocks := by
    have h1 : i < numBlocks * blockSize := Nat.lt_of_lt_of_le hi h_cover
    rw [Nat.mul_comm] at h1
    exact Nat.div_lt_of_lt_mul h1
  have h_gid : globalIdx1D blockSize (i / blockSize) (i % blockSize) = i := by
    simp only [globalIdx1D]
    ring_nf
    exact Nat.div_add_mod' i blockSize
  have h_gid_lt : globalIdx1D blockSize (i / blockSize) (i % blockSize) < N := by
    rw [h_gid]; exact hi

  have h_thread := saxpy_thread_computes N α x y blockSize numBlocks
    (i % blockSize) (i / blockSize)
    h_tid h_bid h_gid_lt h_x h_y mem₀ h_mem_x h_mem_y

  -- if thread (bid, tid) writes to any index idx, then idx = globalIdx bid tid
  have h_identity : CLean.Verification.IdentityAccessPatternWithParams saxpyKernelIR.body blockSize numBlocks params "r" N := by
    intro bid tid mem idx h_bound h_writes
    simp only [globalIdx1D]
    by_contra h_neq
    apply h_writes
    simp only [ThreadWritesTo1DWithParams, vExecThread1DWithParams, vExecThreadWithLocals,
              saxpyKernelIR, vEvalStmt, vEvalExpr, vEvalBinOp,
              VCtx.mk', VCtx.getLocal, VCtx.setLocal,
              VCtx.globalIdxX, VVal.toNat, VVal.toBool, VVal.toRat,
              VMem.get, VMem.set] at h_writes ⊢
    simp only [String.reduceEq, true_and, ↓reduceIte, and_true,
              ne_eq, h_neq, not_false_eq_true, and_false]
    split_ifs with h_lt
    · simp only [VMem.set, h_neq, and_false, ↓reduceIte]
    · rfl

  have h_decomp : (vExecKernel1DWithParams saxpyKernelIR.body numBlocks blockSize params mem₀) "r" i =
      vExecThread1DWithParams saxpyKernelIR.body blockSize numBlocks
      (i % blockSize) (i / blockSize) params mem₀ "r" i := by
    exact vExecKernel1DWithParams_at_idx saxpyKernelIR.body numBlocks blockSize "r" N params mem₀ i hi h_cover h_blockSize_pos h_identity

  simp_all [VMem.getR, h_gid, VMem.get, mem_f, h_decomp]
  convert h_thread using 2



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

def upsweepConfig (config grid : Dim3) : KernelSpec :=
  deviceIRToKernelSpec upsweepKernelIR config grid

theorem upsweep_safe : ∀ (config grid : Dim3), KernelSafe (deviceIRToKernelSpec upsweepKernelIR config grid) := by
  intro config grid
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [upsweepConfig, deviceIRToKernelSpec, upsweepKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  · unfold BarrierUniform; intros; trivial

def downsweepConfig (config grid : Dim3) : KernelSpec :=
  deviceIRToKernelSpec downsweepKernelIR config grid

theorem downsweep_safe : ∀ (config grid : Dim3), KernelSafe (deviceIRToKernelSpec downsweepKernelIR config grid) := by
  intro config grid
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp [downsweepConfig, deviceIRToKernelSpec, downsweepKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation] at *
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
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

    let mut upsweepData := paddedData

    let numThreadsPerBlock := 256

    -- Upsweep phase
    let mut twod := 1
    while twod < roundedLength do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock
      let scalarParams : Array Int := #[twod1, roundedLength, twod]
      let arrays := [
        (`data, upsweepData)
      ]

      let grid :Dim3 := ⟨numBlocks, 1, 1⟩
      let block :Dim3 := ⟨numThreadsPerBlock, 1, 1⟩

      let response ← runKernelGPU upsweepKernelIR ScanArgsResponse
        grid
        block
        scalarParams
        arrays

      upsweepData := response.data
      twod := twod * 2

    let zeroedOut := upsweepData.set! (roundedLength - 1) 0

    --Downsweep phase
    let mut downsweepData := zeroedOut
    twod := roundedLength / 2
    while twod >= 1 do
      let twod1 := twod * 2
      let numBlocks := (roundedLength / twod1 + numThreadsPerBlock - 1) / numThreadsPerBlock
      let scalarParams : Array Int := #[twod1, twod, roundedLength]
      let arrays := [
        (`data, downsweepData)
      ]

      let grid : Dim3 := ⟨numBlocks, 1, 1⟩
      let block : Dim3 := ⟨numThreadsPerBlock, 1, 1⟩

      let response ← runKernelGPU downsweepKernelIR ScanArgsResponse
        grid
        block
        scalarParams
        arrays

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


def matmulSpec (matmulConfig matmulGrid: Dim3): KernelSpec :=
  deviceIRToKernelSpec matmulKernelIR matmulConfig matmulGrid

theorem matmul_safe : ∀ (config grid : Dim3), KernelSafe (matmulSpec config grid) := by
  intro config grid
  constructor
  · unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    simp_all [matmulSpec, deviceIRToKernelSpec, matmulKernelIR, extractFromStmt, extractReadsFromExpr, dexprToAddressPattern, DistinctThreads, List.lookup, HasRace, SeparatedByBarrier, AddressPattern.couldCollide, getArrayName, AccessExtractor.getArrayLocation]
    intro h_race
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, SymValue.isNonZero]
  · unfold BarrierUniform; intros; trivial


def matmulCPU (n : Nat)
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
    -- IO.println s!"Flattened A: {flattened_A}"
    -- IO.println s!"Flattened B: {flattened_B}"

    -- Use typed scalars: N is int
    let scalarParams := #[n]
    let arrays := [
    (`A, flattened_A),
    (`B, flattened_B),
    (`C, Array.replicate (n*n) 0.0)
    ]

    let threadsPerBlock := 32
    let numBlocks := (n + threadsPerBlock - 1) / threadsPerBlock
    let grid : Dim3 := ⟨numBlocks, numBlocks, 1⟩
    let block : Dim3 := ⟨threadsPerBlock, threadsPerBlock, 1⟩

    let response ← runKernelGPU matmulKernelIR MatMulArgsResponse
      grid
      block
      scalarParams
      arrays


    let out := response.C
    if out.size = n * n then
      let result := Array.ofFn fun (i : Fin n) =>
      Array.ofFn fun (j : Fin n) => out[i.val * n + j.val]!
      return result
    else
      throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n*n}"

#eval do matmulCPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmulCPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]

#eval do matmulGPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmulGPU 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]

end BasicMatMul

/-! ## Shared Memory Transpose Verification -/

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

  -- Phase 1: Load from global to shared
  if row < N && col < N then
    let val ← input.get (row * N + col)
    tile.set (row * N + col) val

  barrier

  -- Phase 2: Read from shared and write transposed to global
  if row < N && col < N then
    let val ← tile.get (col * N + row)
    output.set (row * N + col) val

/-- Generate KernelSpec for transpose kernel -/
def transposeSpec (config grid: Dim3): KernelSpec :=
  deviceIRToKernelSpec transposeKernelIR config grid

/-- Transpose kernel is safe:
    1. Race-free within each phase (barrier separates phases)
    2. Barriers are uniform (not inside thread-dependent conditionals)

    Phase 1 accesses:
    - Read: input[row * N + col] (global)
    - Write: tile[row * N + col] (shared)

    Phase 2 accesses:
    - Read: tile[col * N + row] (shared)
    - Write: output[row * N + col] (global)

    No race in Phase 1: Each thread writes to unique shared location (row*N+col is injective)
    No race in Phase 2: Each thread reads from unique shared location and writes to unique global location
    Cross-phase: Barrier synchronizes all threads before Phase 2 reads what Phase 1 wrote -/
theorem transpose_safe : ∀ (config grid : Dim3), KernelSafe (transposeSpec config grid) := by
  intro config grid
  unfold KernelSafe
  constructor
  . -- RaceFree proof
    unfold RaceFree
    intro tid1 tid2 h_distinct a1 a2 ha1 ha2
    -- The key insight: either accesses don't race (different locations/spaces),
    -- or they are separated by the barrier at stmtIdx 5.
    -- Phase 1 writes to shared tile at stmtIdx 3
    -- Phase 2 reads from shared tile at stmtIdx 6
    -- Since 3 < 5 < 6, they are barrier-separated
    simp_all [transposeSpec, deviceIRToKernelSpec, transposeKernelIR, extractFromStmt,
               extractReadsFromExpr, dexprToAddressPattern, List.lookup, getArrayName,
               AccessExtractor.getArrayLocation, AccessExtractor.getMemorySpace,
               List.reverse_cons, List.reverse_nil, List.mem_cons, List.mem_singleton,
               List.append_nil, List.mem_nil_iff, or_false, false_or,
               HasRace, SeparatedByBarrier, AddressPattern.couldCollide,
               SymValue.isNonZero, GPUVerify.MemorySpace]
    -- All goals should now be decidable propositions about memory spaces and locations
    -- Use decide/trivial to discharge them
    rcases h_distinct with ⟨_,_,h_neq⟩
    rcases ha1 with ha1 | ha1 | ha1 | ha1 | ha1 | ha1 <;>
    rcases ha2 with ha2 | ha2 | ha2 | ha2 | ha2 | ha2 <;>
    simp_all [AddressPattern.eval, ha1, ha2, SymValue.isNonZero]
  . -- BarrierUniform proof
    unfold BarrierUniform
    intros
    trivial

/-- Example: transpose a 4x4 matrix -/
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

-- Test CPU execution
#eval do transpose 4 #[#[1.0,2.0,3.0,4.0], #[5.0,6.0,7.0,8.0], #[9.0,10.0,11.0,12.0], #[13.0,14.0,15.0,16.0]]

-- Check the extracted spec has correct structure
#eval do
  let spec := transposeSpec ⟨8, 8, 1⟩ ⟨1, 1, 1⟩
  IO.println s!"Transpose KernelSpec:"
  IO.println s!"  Block size: {spec.blockSize}"
  IO.println s!"  Grid size: {spec.gridSize}"
  IO.println s!"  Accesses: {spec.accesses.length}"
  IO.println s!"  Barriers: {spec.barriers}"
  -- The spec should show:
  -- - Reads/writes to global arrays (input, output) at different locations
  -- - Reads/writes to shared array (tile) at different locations
  -- - A barrier between phases
  IO.println "Accesses (inspect with repr):"
  IO.println (repr spec.accesses)

end SharedMemTranspose
