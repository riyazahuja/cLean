import CLean.GPU
import CLean.VerifyIR
import CLean.Extract

open CLean.Extract CLean.VerifyIR
open Lean Lean.Meta GpuDSL

-- Simple test kernel with regular let bindings
structure SimpleArgs where
  N : Nat
  x : Name

def testLetBindings : KernelM SimpleArgs Unit := do
  let args ← getArgs
  let N := args.N
  let x : GlobalArray Float := ⟨args.x⟩

  let i ← globalIdxX

  -- Regular let bindings - should now be extracted!
  let doubled := i * 2
  let sum := doubled + 1

  if sum < N then
    x.set i (Float.ofNat sum)

-- Test extraction
#eval show MetaM Unit from do
  try
    let vkernel ← manualExtractKernel `testLetBindings

    IO.println "✓ Extraction succeeded!"
    IO.println s!"Kernel: {vkernel.name}"
    IO.println s!"Locals: {vkernel.locals.length}"
    IO.println s!"Body statements: {vkernel.body.length}"

    IO.println "\nLocals:"
    for localVar in vkernel.locals do
      IO.println s!"  - {localVar.name}"

    IO.println "\nBody (simplified):"
    for stmt in vkernel.body do
      match stmt.stmt with
      | .assign name expr => IO.println s!"  ASSIGN {name} = <expr>"
      | .read loc name => IO.println s!"  READ {name} from {loc.array}"
      | .write loc expr => IO.println s!"  WRITE to {loc.array}"
      | .barrier => IO.println s!"  BARRIER"
      | .ite cond _ _ => IO.println s!"  IF <cond> THEN ... ELSE ..."
      | _ => IO.println s!"  <stmt>"
  catch e =>
    IO.println s!"✗ Extraction failed: {← e.toMessageData.toString}"
