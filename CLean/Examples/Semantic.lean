import CLean.Examples.Common

namespace CLean

open Helpers
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
  match StepMachine.currentInstrStep? exampleState 0 0 with
  | some st => st
  | none => exampleState

private def afterTerminateState : State :=
  match StepMachine.currentTermStep? afterAssignState 0 0 with
  | some st => st
  | none => afterAssignState

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
  exact StepInstr.exists_of_stepInstr?_isSome
    (warpState := baseWarp) (participants := [lane0])
    ((State.wf_iff_bool exampleState).2 (by native_decide))
    (by simp [State.getWarp?, State.getCTA?, exampleState, baseCTA])
    ((WarpState.wf_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.participatingRunnable_iff_bool baseWarp exampleAssign.guard? [lane0]).2 (by native_decide))
    (by native_decide)

private theorem example_step_assign : StepMachine exampleState afterAssignState := by
  cstep

example : ∃ st', StepMachine exampleState st' :=
  ⟨afterAssignState, example_step_assign⟩

private theorem example_step_terminate : StepMachine afterAssignState afterTerminateState := by
  cstep


theorem toy_assign_kernel_functional :
    ∃ st1 st2,
      StepMachine exampleState st1 ∧
      StepMachine st1 st2 ∧
      lane0HasR1Seven st1 = true ∧
      lane0Terminated st2 = true := by
  refine ⟨afterAssignState, afterTerminateState, example_step_assign, example_step_terminate, ?_, ?_⟩
  · native_decide
  · native_decide

private def exampleRunFinalState : State :=
  StepMachine.runN 2 exampleState

theorem toy_assign_kernel_run_functional :
    Reaches exampleState exampleRunFinalState ∧
      lane0HasR1Seven exampleRunFinalState = true ∧
      lane0Terminated exampleRunFinalState = true := by
  refine ⟨StepMachine.runN_reaches 2 exampleState, ?_, ?_⟩
  · native_decide
  · native_decide

example : (StepMachine.traceN 2 exampleState).length = 3 := by
  native_decide

example : (StepMachine.runN? 2 exampleState).isSome = true := by
  native_decide

example : lane0HasR1Seven afterAssignState = true := by
  native_decide

example : lane0Terminated afterTerminateState = true := by
  native_decide

example : lane0HasGlobalAddr afterCvtaState = true := by
  native_decide

example : lane0PredQTrue afterIsspacepState = true := by
  native_decide
private def copyLoad : GInstr :=
  { instr := .load "r1" { space := .global, ty := .u32, addr := .imm (.u64 (UInt64.ofNat copySrcOffset)) } }

private def copyStore : GInstr :=
  { instr := .store { space := .global, ty := .u32, addr := .imm (.u64 (UInt64.ofNat copyDstOffset)) }
      (.reg "r1") }

private def copyBlock : Block :=
  { label := "copy"
    body := #[copyLoad, copyStore]
    term := .terminate }

private def copyWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("copy", 0) }, activeMask := 1 }

private def copyCTA0 : CTAState :=
  { warps := ({} : Std.HashMap WarpId WarpState).insert 0 copyWarp0 }

private def copyState : State :=
  { kernelEnv := { entry := "copy", blocks := ({} : Std.HashMap BlockLabel Block).insert "copy" copyBlock }
    global := { bytes := Helpers.writeBytes ({} : ByteMem) copySrcOffset (Helpers.natToBytesLE copyValue.toNat 4) }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0 copyCTA0 }

private def copyAfterLoadState : State :=
  match StepMachine.currentInstrStep? copyState 0 0 with
  | some st => st
  | none => copyState

private def copyAfterStoreState : State :=
  match StepMachine.currentInstrStep? copyAfterLoadState 0 0 with
  | some st => st
  | none => copyAfterLoadState

private def copyAfterTerminateState : State :=
  match StepMachine.currentTermStep? copyAfterStoreState 0 0 with
  | some st => st
  | none => copyAfterStoreState

private def copyLane0Loaded (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["r1"]? with
      | some (.u32 v) => v == copyValue
      | _ => false
  | none => false

private theorem copy_step_load : StepMachine copyState copyAfterLoadState := by
  cstep

private theorem copy_step_store : StepMachine copyAfterLoadState copyAfterStoreState := by
  cstep

private theorem copy_step_terminate : StepMachine copyAfterStoreState copyAfterTerminateState := by
  cstep

theorem toy_copy_kernel_functional :
    ∃ st1 st2 st3,
      StepMachine copyState st1 ∧
      StepMachine st1 st2 ∧
      StepMachine st2 st3 ∧
      copyLane0Loaded st1 = true ∧
      copyDstHasValue st2 = true ∧
      copyLane0Terminated st3 = true := by
  refine ⟨copyAfterLoadState, copyAfterStoreState, copyAfterTerminateState,
    copy_step_load, copy_step_store, copy_step_terminate, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

private def copyRunFinalState : State :=
  StepMachine.runN 3 copyState

theorem toy_copy_kernel_run_functional :
    Reaches copyState copyRunFinalState ∧
      copyLane0Loaded copyRunFinalState = true ∧
      copyDstHasValue copyRunFinalState = true ∧
      copyLane0Terminated copyRunFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 copyState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

example : (StepMachine.traceN 3 copyState).length = 4 := by
  native_decide

example : (StepMachine.runN? 3 copyState).isSome = true := by
  native_decide

example : copyLane0Loaded copyAfterLoadState = true := by
  native_decide

example : copyDstHasValue copyAfterStoreState = true := by
  native_decide

example : copyLane0Terminated copyAfterTerminateState = true := by
  native_decide

private def barrierInstr : GInstr :=
  { instr := .barrierCTA 0 }

private def barrierAssign : GInstr :=
  { instr := .assignReg "r1" (.imm (.u32 7)) }

private def barrierBlock : Block :=
  { label := "barrier"
    body := #[barrierInstr, barrierAssign]
    term := .terminate }

private def barrierWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("barrier", 0) }, activeMask := 1 }

private def barrierCTA0 : CTAState :=
  { warps := ({} : Std.HashMap WarpId WarpState).insert 0 barrierWarp0
    barrier := { bars := ({} : Std.HashMap Nat BarrierInstance).insert 0 { expectedCount := 1 } } }

private def barrierState : State :=
  { kernelEnv := { entry := "barrier", blocks := ({} : Std.HashMap BlockLabel Block).insert "barrier" barrierBlock }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0 barrierCTA0 }

private def afterBarrierState : State :=
  match StepMachine.currentInstrStep? barrierState 0 0 with
  | some st => st
  | none => barrierState

private def afterBarrierAssignState : State :=
  match StepMachine.currentInstrStep? afterBarrierState 0 0 with
  | some st => st
  | none => afterBarrierState

private def afterBarrierTerminateState : State :=
  match StepMachine.currentTermStep? afterBarrierAssignState 0 0 with
  | some st => st
  | none => afterBarrierAssignState

private theorem barrier_step_release : StepMachine barrierState afterBarrierState := by
  cstep

private theorem barrier_step_assign : StepMachine afterBarrierState afterBarrierAssignState := by
  cstep

private theorem barrier_step_terminate : StepMachine afterBarrierAssignState afterBarrierTerminateState := by
  cstep

theorem toy_barrier_kernel_functional :
    ∃ st1 st2 st3,
      StepMachine barrierState st1 ∧
      StepMachine st1 st2 ∧
      StepMachine st2 st3 ∧
      lane0RunningAt ("barrier", 1) st1 = true ∧
      barrier0Released st1 = true ∧
      lane0HasR1Seven st2 = true ∧
      lane0Terminated st3 = true := by
  refine ⟨afterBarrierState, afterBarrierAssignState, afterBarrierTerminateState,
    barrier_step_release, barrier_step_assign, barrier_step_terminate, ?_, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide
  · native_decide

private def barrierRunFinalState : State :=
  StepMachine.runN 3 barrierState

theorem toy_barrier_kernel_run_functional :
    Reaches barrierState barrierRunFinalState ∧
      barrier0Released barrierRunFinalState = true ∧
      lane0HasR1Seven barrierRunFinalState = true ∧
      lane0Terminated barrierRunFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 barrierState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

example : (StepMachine.traceN 3 barrierState).length = 4 := by
  native_decide

example : (StepMachine.runN? 3 barrierState).isSome = true := by
  native_decide

example : lane0RunningAt ("barrier", 1) afterBarrierState = true := by
  native_decide

example : barrier0Released afterBarrierState = true := by
  native_decide

example : lane0HasR1Seven afterBarrierAssignState = true := by
  native_decide

example : lane0Terminated afterBarrierTerminateState = true := by
  native_decide

end CLean
