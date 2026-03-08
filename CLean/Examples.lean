import CLean.Semantics

namespace CLean

open Helpers

private def lane0 : LaneId := ⟨0, by decide⟩

private def baseWarp : WarpState :=
  { lanes := Array.replicate 32 { pc := ("entry", 0) }, activeMask := 1 }

private def baseCTA : CTAState :=
  { warps := ({} : Std.HashMap WarpId WarpState).insert 0 baseWarp }

private def exampleBlock : Block :=
  { label := "entry"
    body := #[( { instr := .assignReg "r1" (.imm (.u32 7)) } : GInstr )]
    term := .terminate }

private def exampleState : State :=
  { kernelEnv := { entry := "entry", blocks := ({} : Std.HashMap BlockLabel Block).insert "entry" exampleBlock }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0 baseCTA }

example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).regs = ["a", "b"] := by
  native_decide

example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).preds = ["p"] := by
  native_decide

example : (Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) }).isSome = true := by
  native_decide

private def terminatedState : State :=
  match Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) } with
  | some st => st
  | none => exampleState

example : (Helpers.stepTerminator? terminatedState 0 0 .terminate).isSome = true := by
  native_decide

end CLean
