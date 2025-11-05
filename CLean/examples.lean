import CLean.GPU


open Lean GpuDSL SciLean


namespace Saxpy



-- This generates the saxpyArgs structure automatically
kernelArgs saxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

-- Verify it generated correctly
#check saxpyArgs
#print saxpyArgs

-- Now you write the kernel manually using the generated structure
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



def saxpy {n : Nat}
    (α : Float)
    (x y : Float^[n]) : IO (Float^[n]) := do
  let initState := mkKernelState [
    globalFloatArray `X x.toList.toArray,
    globalFloatArray `Y y.toList.toArray,
    globalFloatArray `R (Array.replicate n 0.0)
  ]

  let finalState :=
    runKernelCPU
      ⟨(n + 511) / 512, 1, 1⟩         -- grid
      ⟨512, 1, 1⟩                     -- block
      ⟨n, α, `X, `Y, `R⟩         -- args
      initState
      saxpyKernel

  let some (KernelValue.arrayFloat out) := finalState.globals.get? `R
    | throw <| IO.userError "Result missing"
  if out.size = n then
    pure (⊞ (i : Idx n) => out[i.1]!)
  else
    throw <| IO.userError s!"Size mismatch: {out.size} ≠ {n}"

#eval do saxpy 8.0 ⊞[1.0, 1.0] ⊞[2.0, 2.0]




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
      upsweepState := runKernelCPU
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
      downsweepState := runKernelCPU
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
