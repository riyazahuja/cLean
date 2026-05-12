import CLean.Lemmas
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

private def afterAssignWarp : WarpState :=
  match afterAssignState.getWarp? 0 0 with
  | some warpState => warpState
  | none => baseWarp

private theorem afterAssignState_getWarp_isSome :
    (afterAssignState.getWarp? 0 0).isSome = true := by
  native_decide


private theorem afterAssignState_block_entry :
    afterAssignState.kernelEnv.blocks["entry"]? = some exampleBlock := by
  unfold afterAssignState
  cases hstep : Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) } with
  | none =>
      simp [exampleState, exampleBlock]
  | some st' =>
      have hk := stepInstr?_preserves_kernelEnv hstep
      simp [hk, exampleState, exampleBlock]

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

private def cvtaLane0 : LaneState :=
  { regs := ({} : Std.HashMap RegName Value).insert "p" (.u64 64), pc := ("entry", 0) }

private def cvtaWarp : WarpState :=
  { lanes := (Array.replicate 32 { pc := ("entry", 0) }).set! 0 cvtaLane0, activeMask := 1 }

private def cvtaCTA : CTAState :=
  { warps := ({} : Std.HashMap WarpId WarpState).insert 0 cvtaWarp }

private def cvtaState : State :=
  { kernelEnv := { entry := "entry", blocks := ({} : Std.HashMap BlockLabel Block).insert "entry" exampleBlock }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0 cvtaCTA }

private def afterCvtaState : State :=
  match Helpers.stepInstr? cvtaState 0 0 { instr := .cvta "gp" .global (.reg "p") } with
  | some st => st
  | none => cvtaState

private def afterIsspacepState : State :=
  match Helpers.stepInstr? afterCvtaState 0 0 { instr := .isspacep "q" .global (.reg "gp") } with
  | some st => st
  | none => afterCvtaState

private def lane0HasGlobalAddr (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["gp"]? with
      | some (.gaddr .global 64) => true
      | _ => false
  | none => false

private def lane0PredQTrue (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.preds["q"]? with
      | some true => true
      | _ => false
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

example : (Helpers.stepInstr? cvtaState 0 0 { instr := .cvta "gp" .global (.reg "p") }).isSome = true := by
  native_decide

example : (Helpers.stepInstr? afterCvtaState 0 0 { instr := .isspacep "q" .global (.reg "gp") }).isSome = true := by
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

private theorem example_step_assign : StepMachine exampleState afterAssignState := by
  unfold afterAssignState
  cases hstep : Helpers.stepInstr? exampleState 0 0 exampleAssign with
  | none =>
      have hisSome : (Helpers.stepInstr? exampleState 0 0 exampleAssign).isSome = true := by
        native_decide
      simp [hstep] at hisSome
  | some st' =>
      refine StepMachine.mk (cta := 0) (warp := 0) ?_ ?_
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
              (by
                have hstepLit :
                    Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) } = some st' := by
                  simpa [exampleAssign] using hstep
                have hmatch :
                    (match Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) } with
                    | some st => st
                    | none => exampleState) = st' := by
                  rw [hstepLit]
                simpa [hmatch] using hstepLit)

private theorem example_step_terminate : StepMachine afterAssignState afterTerminateState := by
  cases hstep : Helpers.stepTerminator? afterAssignState 0 0 .terminate with
  | none =>
      have hisSome : (Helpers.stepTerminator? afterAssignState 0 0 .terminate).isSome = true := by
        native_decide
      simp [hstep] at hisSome
  | some st' =>
      cases hwarp : afterAssignState.getWarp? 0 0 with
      | none =>
          have hisSome := afterAssignState_getWarp_isSome
          simp [hwarp] at hisSome
      | some warpState =>
          have hAfterWarp : afterAssignWarp = warpState := by
            simp [afterAssignWarp, hwarp]
          refine StepMachine.mk (cta := 0) (warp := 0) ?_ ?_
          · exact (State.wf_iff_bool afterAssignState).2 (by native_decide)
          · refine StepWarp.mk (cta := 0) (warp := 0) ?_ ?_
            · exact (State.wf_iff_bool afterAssignState).2 (by native_decide)
            · refine StepBlock.term (warpState := warpState) (pc := ("entry", 1)) (block := exampleBlock)
                ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
              · exact (State.wf_iff_bool afterAssignState).2 (by native_decide)
              · exact hwarp
              · rw [← hAfterWarp]
                exact (WarpState.wf_iff_bool afterAssignWarp).2 (by native_decide)
              · rw [← hAfterWarp]
                exact (Helpers.lockstepRunnable_iff_bool afterAssignWarp).2 (by native_decide)
              · rw [← hAfterWarp]
                exact (Helpers.runnablePc_iff_bool afterAssignWarp ("entry", 1)).2 (by native_decide)
              · exact afterAssignState_block_entry
              · simp [exampleBlock]
              ·
                have hmatch :
                    (match Helpers.stepTerminator? afterAssignState 0 0 .terminate with
                    | some st => st
                    | none => afterAssignState) = st' := by
                  rw [hstep]
                simpa [afterTerminateState, hmatch] using hstep


theorem toy_assign_kernel_functional :
    ∃ st1 st2,
      StepMachine exampleState st1 ∧
      StepMachine st1 st2 ∧
      lane0HasR1Seven st1 = true ∧
      lane0Terminated st2 = true := by
  refine ⟨afterAssignState, afterTerminateState, example_step_assign, example_step_terminate, ?_, ?_⟩
  · native_decide
  · native_decide

example : lane0HasR1Seven afterAssignState = true := by
  native_decide

example : lane0Terminated afterTerminateState = true := by
  native_decide

example : lane0HasGlobalAddr afterCvtaState = true := by
  native_decide

example : lane0PredQTrue afterIsspacepState = true := by
  native_decide

end CLean
