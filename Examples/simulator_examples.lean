import CLean.GPU

open Lean GpuDSL


namespace Saxpy

kernelArgs saxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

#check saxpyArgs
#print saxpyArgs

def saxpyKernel : KernelM saxpyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let r : GlobalArray Float := ⟨args.r⟩

  let i ← globalIdxX
  if i < N then
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

/-- Upsweep kernel for parallel scan -/
def upsweepKernel : KernelM ScanArgs Unit := do
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

/-- Downsweep kernel for parallel scan -/
def downsweepKernel : KernelM ScanArgs Unit := do
  let args ← getArgs
  let index ← globalIdxX
  let i := index * args.twod1
  if (i + args.twod - 1 < args.length) && (i + args.twod1 - 1 < args.length) then
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


#eval do exclusiveScan #[1,2,3,4,5,6,7,8]



end ExclusiveScan


namespace BasicMatMul


kernelArgs MatMulArgs(N: Nat)
  global[A B C: Array Float]

def matmulKernel : KernelM MatMulArgs Unit := do
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

def matmul (n : Nat)
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

#eval do matmul 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmul 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]


end BasicMatMul




namespace BetterMatMul


kernelArgs MatMulArgs(N: Nat, V: Nat)
  global[A B C: Array Float]

def matmulKernel : KernelM MatMulArgs Unit := do
  let args ← getArgs
  let N := args.N
  let V := args.V
  let A : GlobalArray Float := ⟨args.A⟩
  let B : GlobalArray Float := ⟨args.B⟩
  let C : GlobalArray Float := ⟨args.C⟩

  let ybase ← globalIdxY
  let xbase ← globalIdxX

  -- Initialize local accumulator c[V][V]
  let mut c : Array (Array Float) := Array.mkArray V (Array.mkArray V 0.0)

  -- Loop over k dimension
  for k in [0:N] do
    -- Load a[V] from A[xbase*V : xbase*V + V, k]
    let mut a : Array Float := Array.mkArray V 0.0
    for x in [0:V] do
      if xbase * V + x < N && k < N then
        let aVal ← A.get ((xbase * V + x) * N + k)
        a := a.set! x aVal

    -- Load b[V] from B[k, ybase*V : ybase*V + V]
    let mut b : Array Float := Array.mkArray V 0.0
    for y in [0:V] do
      if k < N && ybase * V + y < N then
        let bVal ← B.get (k * N + (ybase * V + y))
        b := b.set! y bVal

    -- Compute outer product: c[x][y] += a[x] * b[y]
    for y in [0:V] do
      for x in [0:V] do
        let cVal := c[x]! |>.get! y
        c := c.set! x ((c[x]!).set! y (cVal + a[x]! * b[y]!))

  -- Write c[:] back to C[xbase*V : xbase*V + V, ybase*V : ybase*V + V]
  for x in [0:V] do
    for y in [0:V] do
      if xbase * V + x < N && ybase * V + y < N then
        C.set ((xbase * V + x) * N + (ybase * V + y)) (c[x]!|>.get! y)

def matmul (n : Nat) (V : Nat := 4)
    (A B : Array (Array Float)) : IO (Array (Array Float)) := do
  let flattened_A  := A.flatMap id
  let flattened_B  := B.flatMap id

  let initState := mkKernelState [
    globalFloatArray `A flattened_A,
    globalFloatArray `B flattened_B,
    globalFloatArray `C (Array.replicate (n*n) 0.0)
  ]

  let threadsPerBlock := 32
  let numBlocks := (n / V + threadsPerBlock - 1) / threadsPerBlock

  let finalState ←
    runKernelCPU
      ⟨numBlocks, numBlocks, 1⟩
      ⟨threadsPerBlock, threadsPerBlock, 1⟩
      ⟨n, V, `A, `B, `C⟩
      initState
      matmulKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `C
    | throw <| IO.userError "Result missing"
  if out.size = n * n then
    pure (Array.ofFn fun (i : Fin n) =>
      Array.ofFn fun (j : Fin n) => out[i.val * n + j.val]!)
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n*n}"

#eval do matmul 2 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[1.0,0.0],#[0.0,1.0]]
#eval do matmul 2 2 #[#[1.0,2.0],#[3.0,4.0]] #[#[5.0,6.0],#[7.0,8.0]]


end BetterMatMul


namespace SharedMemTranspose

kernelArgs TransposeArgs(N: Nat)
  global[input output: Array Float]
  shared[tile: Array Float]

def transposeKernel : KernelM TransposeArgs Unit := do
  let args ← getArgs
  let N := args.N
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let tile : SharedArray Float := ⟨args.tile⟩

  let row ← globalIdxX
  let col ← globalIdxY

  -- Phase 1: Load from global memory to shared memory (row-major)
  if row < N && col < N then
    let val ← input.get (row * N + col)
    tile.set (row * N + col) val

  -- CRITICAL BARRIER: Wait for all threads to finish writing to shared memory
  -- Without this, the read below might access uninitialized memory
  barrier

  -- Phase 2: Read from shared memory in transposed pattern (column-major)
  -- and write to global memory
  if row < N && col < N then
    -- Read from transposed position in shared memory
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

-- Test with a simple 4x4 matrix
-- Input:   1  2  3  4
--          5  6  7  8
--          9 10 11 12
--         13 14 15 16
-- Expected output (transposed):
--          1  5  9 13
--          2  6 10 14
--          3  7 11 15
--          4  8 12 16
#eval do
  let result ← transpose 4 #[#[1.0,2.0,3.0,4.0], #[5.0,6.0,7.0,8.0], #[9.0,10.0,11.0,12.0], #[13.0,14.0,15.0,16.0]]
  IO.println s!"Transposed matrix: {result}"
  return result

end SharedMemTranspose


namespace SharedPrefixSum

/-! ## Shared Memory Prefix Sum using Hillis-Steele Algorithm

This kernel computes an inclusive prefix sum (scan) of elements within a block
using shared memory and multiple barrier synchronizations.

Algorithm: Hillis-Steele parallel scan
- Requires log2(n) iterations with barriers between each
- Each iteration doubles the stride distance
- Iteration d: each thread i adds element at (i - 2^d) to element at i

Example for array [1, 2, 3, 4, 5, 6, 7, 8]:
Initial:    [1, 2, 3, 4, 5, 6, 7, 8]
After d=0:  [1, 3, 5, 7, 9, 11, 13, 15]  (stride 1)
After d=1:  [1, 3, 6, 10, 14, 18, 22, 26] (stride 2)
After d=2:  [1, 3, 6, 10, 15, 21, 28, 36] (stride 4)

The barriers are CRITICAL for correctness - without them, threads would read
intermediate values before other threads finish writing, leading to race conditions.
-/

kernelArgs PrefixSumArgs(n: Nat)
  global[input output: Array Float]
  shared[temp: Array Float]

def prefixSumKernel : KernelM PrefixSumArgs Unit := do
  let args ← getArgs
  let n := args.n
  let input : GlobalArray Float := ⟨args.input⟩
  let output : GlobalArray Float := ⟨args.output⟩
  let temp : SharedArray Float := ⟨args.temp⟩

  let tid ← globalIdxX

  -- Load input into shared memory
  if tid < n then
    let val ← input.get tid
    temp.set tid val
  else
    temp.set tid 0.0

  -- BARRIER 1: Ensure all threads have loaded their input
  barrier

  -- Hillis-Steele iterations
  -- We need log2(n) iterations, but we'll do a fixed number for simplicity
  -- For n=8, we need 3 iterations (log2(8) = 3)
  -- For n=16, we need 4 iterations (log2(16) = 4)
  let maxIter := 4  -- Supports up to 16 elements

  for d in [0:maxIter] do
    let stride := 1 <<< d  -- 2^d

    let mut newVal := 0.0
    if tid < n then
      newVal ← temp.get tid
      if tid >= stride then
        let prevVal ← temp.get (tid - stride)
        newVal := newVal + prevVal

    -- BARRIER: Wait for all threads to read before writing
    -- Without this, we'd have a read-after-write hazard
    barrier

    if tid < n then
      temp.set tid newVal

    -- BARRIER: Wait for all threads to write before next iteration
    -- Without this, next iteration might read stale values
    barrier

  -- Write result back to global memory
  if tid < n then
    let result ← temp.get tid
    output.set tid result

def prefixSum (n : Nat) (input : Array Float) : IO (Array Float) := do
  -- Convert tensor to array, reversing to fix ordering issue

  let initState := mkKernelState
    [ globalFloatArray `input input
    , globalFloatArray `output (Array.replicate n 0.0)
    ]
    [ (`temp, KernelValue.arrayFloat (Array.replicate 16 0.0))  -- Max 16 elements
    ]

  -- Use a single block with n threads
  let finalState ←
    runKernelCPU
      ⟨1, 1, 1⟩           -- Single block
      ⟨16, 1, 1⟩          -- Up to 16 threads (supports arrays up to size 16)
      ⟨n, `input, `output, `temp⟩
      initState
      prefixSumKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `output
    | throw <| IO.userError "Result missing"
  if out.size >= n then
    pure (Array.ofFn fun (i : Fin n) => out[i.val]!)
  else
    throw <| IO.userError s!"Size mismatch: {out.size} < {n}"

-- Test cases
-- Input: [1, 2, 3, 4]
-- Expected output: [1, 3, 6, 10]
#eval do
  let result ← prefixSum 4 #[1.0, 2.0, 3.0, 4.0]
  IO.println s!"Prefix sum of [1,2,3,4]: {result}"
  return result

-- Input: [1, 2, 3, 4, 5, 6, 7, 8]
-- Expected output: [1, 3, 6, 10, 15, 21, 28, 36]
#eval do
  let result ← prefixSum 8 #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  IO.println s!"Prefix sum of [1,2,3,4,5,6,7,8]: {result}"
  return result

-- Input: [2, 2, 2, 2, 2, 2, 2, 2]
-- Expected output: [2, 4, 6, 8, 10, 12, 14, 16]
#eval do
  let result ← prefixSum 8 #[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
  IO.println s!"Prefix sum of [2,2,2,2,2,2,2,2]: {result}"
  return result

end SharedPrefixSum
