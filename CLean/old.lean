import CLean.GPU


open GpuDSL SciLean


namespace test

structure BufferNames where
  x : String
  y : String
  r : String
deriving Repr

structure SaxpyArgs where
  N      : Nat
  alpha  : Float
  names  : BufferNames
deriving Repr

/-- `result[i] := alpha * x[i] + y[i]` if `i < N`. -/
def saxpyKernel : KernelM SaxpyArgs Unit := do
  let c ← readCtx
  let i ← globalIdxX
  if i < c.args.N then
    let xi : Float ← gReadAt c.args.names.x i --ctx.args.
    let yi : Float ← gReadAt c.args.names.y i
    gWriteAt c.args.names.r i (c.args.alpha * xi + yi)

/-- Host-side typed wrapper:
    allocs/copies are *simulated*; compute via CPU interpreter. -/
def saxpyCuda {n : Nat}
    (α : Float)
    (x y : Float^[n]) : IO (Float^[n]) := do
  let aX := x.toList.toArray--vecToArray x
  let aY := y.toList.toArray--vecToArray y
  let aR := Array.replicate n (0.0 : Float)

  -- name globals (simulates device pointers)
  let names : BufferNames := { x := "X", y := "Y", r := "R" }
  let initState : KernelState :=
    { globals := (∅ : Std.HashMap String KernelValue)
        |>.insert names.x (KernelValue.arrayFloat aX)
        |>.insert names.y (KernelValue.arrayFloat aY)
        |>.insert names.r (KernelValue.arrayFloat aR)
      shared := ∅ }

  -- launch configuration
  let threadsPerBlock := 512
  let blocks := (n + threadsPerBlock - 1) / threadsPerBlock
  let grid  := Dim3.mk blocks 1 1
  let block := Dim3.mk threadsPerBlock 1 1

  -- run on CPU
  let finalState :=
    runKernelCPU grid block
      { N := n, alpha := α, names }
      initState saxpyKernel

  -- fetch result
  let some (KernelValue.arrayFloat out) := finalState.globals.get? names.r
    | throw <| IO.userError "saxpyCuda: result buffer missing or wrong type"
  if out.size = n then
    -- Convert Array Float back to Float^[n] using ⊞ notation
    -- Using unsafe indexing since we verified the size matches
    pure (⊞ (i : Idx n) => out[i.1]!)
  else
    throw <| IO.userError s!"saxpyCuda: result buffer size mismatch (got {out.size}, expected {n})"


#eval do saxpyCuda 8.0 ⊞[1.0, 1.0] ⊞[2.0, 2.0]


end test
