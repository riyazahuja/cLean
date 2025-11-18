import Lean
import CLean.GPU

open Lean Lean.Meta

-- Simple test kernel

kernelArgs SimpleSaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

def simpleKernel : GpuDSL.KernelM SimpleSaxpyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GpuDSL.GlobalArray Float := ⟨args.x⟩
  let y : GpuDSL.GlobalArray Float := ⟨args.y⟩
  let r : GpuDSL.GlobalArray Float := ⟨args.r⟩

  let i ← GpuDSL.globalIdxX
  if i < N then
    let xi ← x.get i
    let yi ← y.get i
    r.set i (alpha * xi + yi)


-- Try to extract it
#eval show MetaM Unit from do
  let env ← getEnv
  let some info := env.find? `simpleKernel
    | throwError "not found"

  IO.println s!"Found kernel: {info.name}"
  IO.println s!"Type: {info.type}"

  -- Try to get value
  match info.value? with
  | none => IO.println "No value"
  | some val =>
    IO.println s!"Value expr (raw): {val}"
    -- Try whnf
    try
      let val' ← whnf val
      IO.println s!"Value (whnf): {val'}"
    catch e => do
      let err ← e.toMessageData.toString
      IO.println s!"Error in whnf: {err}"
