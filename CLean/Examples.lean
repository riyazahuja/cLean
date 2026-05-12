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
  match Helpers.stepInstr? exampleState 0 0 exampleAssign with
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
  calc
    (match Helpers.stepInstr? exampleState 0 0 exampleAssign with
     | some st => st
     | none => exampleState).kernelEnv.blocks["entry"]? =
        exampleState.kernelEnv.blocks["entry"]? :=
          stepInstr?_computed_preserves_block?
            (st := exampleState) (cta := 0) (warp := 0) (gi := exampleAssign)
            "entry" (by native_decide)
    _ = some exampleBlock := by simp [exampleState, exampleBlock]

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
  exact StepInstr.exists_of_stepInstr?_isSome
    (warpState := baseWarp) (participants := [lane0])
    ((State.wf_iff_bool exampleState).2 (by native_decide))
    (by simp [State.getWarp?, State.getCTA?, exampleState, baseCTA])
    ((WarpState.wf_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.participatingRunnable_iff_bool baseWarp exampleAssign.guard? [lane0]).2 (by native_decide))
    (by native_decide)

private theorem example_step_assign : StepMachine exampleState afterAssignState := by
  unfold afterAssignState
  exact StepMachine.body_of_stepInstr?_computed
    (cta := 0) (warp := 0) (warpState := baseWarp) (pc := ("entry", 0))
    (block := exampleBlock) (gi := exampleAssign) (participants := [lane0])
    ((State.wf_iff_bool exampleState).2 (by native_decide))
    (by simp [State.getWarp?, State.getCTA?, exampleState, baseCTA])
    ((WarpState.wf_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.lockstepRunnable_iff_bool baseWarp).2 (by native_decide))
    ((Helpers.runnablePc_iff_bool baseWarp ("entry", 0)).2 (by native_decide))
    (by simp [exampleState, exampleBlock])
    (by simp [exampleBlock, exampleAssign])
    ((Helpers.participatingRunnable_iff_bool baseWarp exampleAssign.guard? [lane0]).2 (by native_decide))
    (by native_decide)

example : ∃ st', StepMachine exampleState st' :=
  ⟨afterAssignState, example_step_assign⟩

private theorem example_step_terminate : StepMachine afterAssignState afterTerminateState := by
  unfold afterTerminateState
  cases hwarp : afterAssignState.getWarp? 0 0 with
  | none =>
      have hisSome := afterAssignState_getWarp_isSome
      simp [hwarp] at hisSome
  | some warpState =>
      have hAfterWarp : afterAssignWarp = warpState := by
        simp [afterAssignWarp, hwarp]
      exact StepMachine.term_of_stepTerminator?_computed
        (cta := 0) (warp := 0) (warpState := warpState) (pc := ("entry", 1))
        (block := exampleBlock)
        ((State.wf_iff_bool afterAssignState).2 (by native_decide))
        hwarp
        (by
          rw [← hAfterWarp]
          exact (WarpState.wf_iff_bool afterAssignWarp).2 (by native_decide))
        (by
          rw [← hAfterWarp]
          exact (Helpers.lockstepRunnable_iff_bool afterAssignWarp).2 (by native_decide))
        (by
          rw [← hAfterWarp]
          exact (Helpers.runnablePc_iff_bool afterAssignWarp ("entry", 1)).2 (by native_decide))
        afterAssignState_block_entry
        (by simp [exampleBlock])
        (by native_decide)


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

private def copySrcOffset : Nat := 0

private def copyDstOffset : Nat := 4

private def copyValue : UInt32 := 99

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
  match Helpers.stepInstr? copyState 0 0 copyLoad with
  | some st => st
  | none => copyState

private def copyAfterLoadWarp : WarpState :=
  match copyAfterLoadState.getWarp? 0 0 with
  | some warpState => warpState
  | none => copyWarp0

private theorem copyAfterLoadState_getWarp_isSome :
    (copyAfterLoadState.getWarp? 0 0).isSome = true := by
  native_decide

private theorem copyAfterLoadState_block_copy :
    copyAfterLoadState.kernelEnv.blocks["copy"]? = some copyBlock := by
  unfold copyAfterLoadState
  calc
    (match Helpers.stepInstr? copyState 0 0 copyLoad with
     | some st => st
     | none => copyState).kernelEnv.blocks["copy"]? =
        copyState.kernelEnv.blocks["copy"]? :=
          stepInstr?_computed_preserves_block?
            (st := copyState) (cta := 0) (warp := 0) (gi := copyLoad)
            "copy" (by native_decide)
    _ = some copyBlock := by simp [copyState, copyBlock]

private def copyAfterStoreState : State :=
  match Helpers.stepInstr? copyAfterLoadState 0 0 copyStore with
  | some st => st
  | none => copyAfterLoadState

private def copyAfterStoreWarp : WarpState :=
  match copyAfterStoreState.getWarp? 0 0 with
  | some warpState => warpState
  | none => copyAfterLoadWarp

private theorem copyAfterStoreState_getWarp_isSome :
    (copyAfterStoreState.getWarp? 0 0).isSome = true := by
  native_decide

private theorem copyAfterStoreState_block_copy :
    copyAfterStoreState.kernelEnv.blocks["copy"]? = some copyBlock := by
  unfold copyAfterStoreState
  calc
    (match Helpers.stepInstr? copyAfterLoadState 0 0 copyStore with
     | some st => st
     | none => copyAfterLoadState).kernelEnv.blocks["copy"]? =
        copyAfterLoadState.kernelEnv.blocks["copy"]? :=
          stepInstr?_computed_preserves_block?
            (st := copyAfterLoadState) (cta := 0) (warp := 0) (gi := copyStore)
            "copy" (by native_decide)
    _ = some copyBlock := copyAfterLoadState_block_copy

private def copyAfterTerminateState : State :=
  match Helpers.stepTerminator? copyAfterStoreState 0 0 .terminate with
  | some st => st
  | none => copyAfterStoreState

private def copyLane0Loaded (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["r1"]? with
      | some (.u32 v) => v == copyValue
      | _ => false
  | none => false

private def copyDstHasValue (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global copyDstOffset) with
  | some (.u32 v) => v == copyValue
  | _ => false

private def copyLane0Terminated (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .terminated
  | none => false

private theorem copy_step_load : StepMachine copyState copyAfterLoadState := by
  unfold copyAfterLoadState
  exact StepMachine.body_of_stepInstr?_computed
    (cta := 0) (warp := 0) (warpState := copyWarp0) (pc := ("copy", 0))
    (block := copyBlock) (gi := copyLoad) (participants := [lane0])
    ((State.wf_iff_bool copyState).2 (by native_decide))
    (by simp [State.getWarp?, State.getCTA?, copyState, copyCTA0])
    ((WarpState.wf_iff_bool copyWarp0).2 (by native_decide))
    ((Helpers.lockstepRunnable_iff_bool copyWarp0).2 (by native_decide))
    ((Helpers.runnablePc_iff_bool copyWarp0 ("copy", 0)).2 (by native_decide))
    (by simp [copyState, copyBlock])
    (by simp [copyBlock, copyLoad])
    ((Helpers.participatingRunnable_iff_bool copyWarp0 copyLoad.guard? [lane0]).2 (by native_decide))
    (by native_decide)

private theorem copy_step_store : StepMachine copyAfterLoadState copyAfterStoreState := by
  unfold copyAfterStoreState
  cases hwarp : copyAfterLoadState.getWarp? 0 0 with
  | none =>
      have hisSome := copyAfterLoadState_getWarp_isSome
      simp [hwarp] at hisSome
  | some warpState =>
      have hCopyWarp : copyAfterLoadWarp = warpState := by
        simp [copyAfterLoadWarp, hwarp]
      exact StepMachine.body_of_stepInstr?_computed
        (cta := 0) (warp := 0) (warpState := warpState) (pc := ("copy", 1))
        (block := copyBlock) (gi := copyStore) (participants := [lane0])
        ((State.wf_iff_bool copyAfterLoadState).2 (by native_decide))
        hwarp
        (by
          rw [← hCopyWarp]
          exact (WarpState.wf_iff_bool copyAfterLoadWarp).2 (by native_decide))
        (by
          rw [← hCopyWarp]
          exact (Helpers.lockstepRunnable_iff_bool copyAfterLoadWarp).2 (by native_decide))
        (by
          rw [← hCopyWarp]
          exact (Helpers.runnablePc_iff_bool copyAfterLoadWarp ("copy", 1)).2 (by native_decide))
        copyAfterLoadState_block_copy
        (by simp [copyBlock, copyStore])
        (by
          rw [← hCopyWarp]
          exact (Helpers.participatingRunnable_iff_bool copyAfterLoadWarp copyStore.guard? [lane0]).2
            (by native_decide))
        (by native_decide)

private theorem copy_step_terminate : StepMachine copyAfterStoreState copyAfterTerminateState := by
  unfold copyAfterTerminateState
  cases hwarp : copyAfterStoreState.getWarp? 0 0 with
  | none =>
      have hisSome := copyAfterStoreState_getWarp_isSome
      simp [hwarp] at hisSome
  | some warpState =>
      have hCopyWarp : copyAfterStoreWarp = warpState := by
        simp [copyAfterStoreWarp, hwarp]
      exact StepMachine.term_of_stepTerminator?_computed
        (cta := 0) (warp := 0) (warpState := warpState) (pc := ("copy", 2))
        (block := copyBlock)
        ((State.wf_iff_bool copyAfterStoreState).2 (by native_decide))
        hwarp
        (by
          rw [← hCopyWarp]
          exact (WarpState.wf_iff_bool copyAfterStoreWarp).2 (by native_decide))
        (by
          rw [← hCopyWarp]
          exact (Helpers.lockstepRunnable_iff_bool copyAfterStoreWarp).2 (by native_decide))
        (by
          rw [← hCopyWarp]
          exact (Helpers.runnablePc_iff_bool copyAfterStoreWarp ("copy", 2)).2 (by native_decide))
        copyAfterStoreState_block_copy
        (by simp [copyBlock])
        (by native_decide)

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

example : copyLane0Loaded copyAfterLoadState = true := by
  native_decide

example : copyDstHasValue copyAfterStoreState = true := by
  native_decide

example : copyLane0Terminated copyAfterTerminateState = true := by
  native_decide

end CLean
