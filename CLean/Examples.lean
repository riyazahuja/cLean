import CLean.Semantics
import Mathlib.Tactic
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

private def exampleAssign : GInstr :=
  { instr := .assignReg "r1" (.imm (.u32 7)) }

private def exampleState : State :=
  { kernelEnv := { entry := "entry", blocks := ({} : Std.HashMap BlockLabel Block).insert "entry" exampleBlock }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0 baseCTA }

private def afterAssignState : State :=
  match Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) } with
  | some st => st
  | none => exampleState

private def afterTerminateState : State :=
  match Helpers.stepTerminator? afterAssignState 0 0 .terminate with
  | some st => st
  | none => afterAssignState

private def lane0HasR1Seven (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["r1"]? with
      | some (.u32 7) => true
      | _ => false
  | none => false

private def lane0Terminated (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .terminated
  | none => false

example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).regs = ["a", "b"] := by
  native_decide

example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).preds = ["p"] := by
  native_decide

example : (Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) }).isSome = true := by
  native_decide

example : (Helpers.stepTerminator? afterAssignState 0 0 .terminate).isSome = true := by
  native_decide

example : State.wf? exampleState = true := by
  native_decide

example : Helpers.lockstepRunnable? baseWarp = true := by
  native_decide

example : ∃ st', StepInstr exampleState 0 0 exampleAssign st' := by
  cases hstep : Helpers.stepInstr? exampleState 0 0 exampleAssign with
  | none =>
      have hisSome : (Helpers.stepInstr? exampleState 0 0 exampleAssign).isSome = true := by
        native_decide
      simp [hstep] at hisSome
  | some st' =>
      refine ⟨st', StepInstr.mk (warpState := baseWarp) (participants := [lane0]) ?_ ?_ ?_ ?_ ?_ ?_⟩
      · exact (State.wf_iff_bool exampleState).2 (by native_decide)
      · simp [State.getWarp?, State.getCTA?, exampleState, baseCTA]
      · exact (WarpState.wf_iff_bool baseWarp).2 (by native_decide)
      · exact (Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide)
      · exact (Helpers.participatingRunnable_iff_bool baseWarp exampleAssign.guard? [lane0]).2 (by native_decide)
      · exact hstep

example : ∃ st', StepMachine exampleState st' := by
  cases hstep : Helpers.stepInstr? exampleState 0 0 exampleAssign with
  | none =>
      have hisSome : (Helpers.stepInstr? exampleState 0 0 exampleAssign).isSome = true := by
        native_decide
      simp [hstep] at hisSome
  | some st' =>
      refine ⟨st', StepMachine.mk (cta := 0) (warp := 0) ?_ ?_⟩
      · exact (State.wf_iff_bool exampleState).2 (by native_decide)
      · refine StepWarp.mk (cta := 0) (warp := 0) ?_ ?_
        · exact (State.wf_iff_bool exampleState).2 (by native_decide)
        · refine StepBlock.body (warpState := baseWarp) (pc := ("entry", 0)) (block := exampleBlock)
            (gi := exampleAssign) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
          · exact (State.wf_iff_bool exampleState).2 (by native_decide)
          · simp [State.getWarp?, State.getCTA?, exampleState, baseCTA]
          · exact (WarpState.wf_iff_bool baseWarp).2 (by native_decide)
          · exact (Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide)
          · exact (Helpers.runnablePc_iff_bool baseWarp ("entry", 0)).2 (by native_decide)
          · simp [exampleState, exampleBlock]
          · simp [exampleBlock, exampleAssign]
          · exact StepInstr.mk (warpState := baseWarp) (participants := [lane0])
              ((State.wf_iff_bool exampleState).2 (by native_decide))
              (by simp [State.getWarp?, State.getCTA?, exampleState, baseCTA])
              ((WarpState.wf_iff_bool baseWarp).2 (by native_decide))
              ((Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide))
              ((Helpers.participatingRunnable_iff_bool baseWarp exampleAssign.guard? [lane0]).2 (by native_decide))
              hstep

example : lane0HasR1Seven afterAssignState = true := by
  native_decide

example : lane0Terminated afterTerminateState = true := by
  native_decide

end CLean
