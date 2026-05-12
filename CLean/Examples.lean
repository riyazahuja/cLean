import CLean.PTXLowering
import CLean.Execution
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
  match StepMachine.currentInstrStep? exampleState 0 0 with
  | some st => st
  | none => exampleState

private def afterTerminateState : State :=
  match StepMachine.currentTermStep? afterAssignState 0 0 with
  | some st => st
  | none => afterAssignState

private def lane0HasR1Seven (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs["r1"]? with
      | some (.u32 7) => true
      | _ => false
  | none => false

private def lane0HasRegU32 (reg : RegName) (value : UInt32) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs[reg]? with
      | some (.u32 v) => v == value
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

private def copyDstHasValue (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global copyDstOffset) with
  | some (.u32 v) => v == copyValue
  | _ => false

private def copyLane0Terminated (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .terminated
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

private def lane0RunningAt (pc : PC) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState => laneState.status == .running && laneState.pc == pc
  | none => false

private def barrier0Released (st : State) : Bool :=
  match st.getCTA? 0 with
  | some ctaState =>
      match ctaState.barrier.bars[0]? with
      | some inst => inst.epoch == 1 && inst.arrived.length == 0
      | none => false
  | none => false

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

example :
    PTX.lowerInstr (.mov .u32 "r1" (.imm (.u32 7))) =
      .assignReg "r1" (.imm (.u32 7)) := by
  rfl

example :
    PTX.lowerInstr (.add .u32 "r3" (.reg "r1") (.reg "r2")) =
      .assignReg "r3" (.binop .add (.reg "r1") (.reg "r2")) := by
  rfl

example :
    PTX.lowerInstr (.barSync 0) = .barrierCTA 0 := by
  rfl

example :
    (match PTX.lowerInstrChecked? {} (.mov .u32 "r1" (.imm (.u32 7))) with
     | .ok (.assignReg "r1" (.imm (.u32 7)), env) => env.regs["r1"]? == some .u32
     | _ => false) = true := by
  native_decide

example :
    (match PTX.lowerInstrChecked? {} (.add .u32 "r3" (.reg "r1") (.reg "r2")) with
     | .error _ => true
     | _ => false) = true := by
  native_decide

private def ptxAssignKernel : PTX.Kernel :=
  { entry := "ptx_assign"
    regs := #[{ name := "r1", ty := .u32 }]
    blocks := #[{
      label := "ptx_assign"
      body := #[({ instr := .mov .u32 "r1" (.imm (.u32 7)) } : PTX.GInstr)]
      term := .exit
    }] }

private def ptxAssignWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("ptx_assign", 0) }, activeMask := 1 }

private def ptxAssignState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD ptxAssignKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 ptxAssignWarp0 } }

private def ptxAssignFinalState : State :=
  StepMachine.runN 2 ptxAssignState

theorem ptx_assign_kernel_run_functional :
    Reaches ptxAssignState ptxAssignFinalState ∧
      lane0HasR1Seven ptxAssignFinalState = true ∧
      lane0Terminated ptxAssignFinalState = true := by
  refine ⟨StepMachine.runN_reaches 2 ptxAssignState, ?_, ?_⟩
  · native_decide
  · native_decide

private def ptxCopyKernel : PTX.Kernel :=
  { entry := "ptx_copy"
    regs := #[{ name := "r1", ty := .u32 }]
    blocks := #[{
      label := "ptx_copy"
      body := #[
        ({ instr := .ld .global .u32 "r1" (.imm (.u64 (UInt64.ofNat copySrcOffset))) } : PTX.GInstr),
        ({ instr := .st .global .u32 (.imm (.u64 (UInt64.ofNat copyDstOffset))) (.reg "r1") } : PTX.GInstr)
      ]
      term := .exit
    }] }

private def ptxCopyWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("ptx_copy", 0) }, activeMask := 1 }

private def ptxCopyState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD ptxCopyKernel
    global := { bytes := Helpers.writeBytes ({} : ByteMem) copySrcOffset (Helpers.natToBytesLE copyValue.toNat 4) }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 ptxCopyWarp0 } }

private def ptxCopyFinalState : State :=
  StepMachine.runN 3 ptxCopyState

theorem ptx_copy_kernel_run_functional :
    Reaches ptxCopyState ptxCopyFinalState ∧
      copyDstHasValue ptxCopyFinalState = true ∧
      copyLane0Terminated ptxCopyFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 ptxCopyState, ?_, ?_⟩
  · native_decide
  · native_decide

private def ptxBarrierKernel : PTX.Kernel :=
  { entry := "ptx_barrier"
    regs := #[{ name := "r1", ty := .u32 }]
    blocks := #[{
      label := "ptx_barrier"
      body := #[
        ({ instr := .barSync 0 } : PTX.GInstr),
        ({ instr := .mov .u32 "r1" (.imm (.u32 7)) } : PTX.GInstr)
      ]
      term := .exit
    }] }

private def ptxBarrierWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("ptx_barrier", 0) }, activeMask := 1 }

private def ptxBarrierState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD ptxBarrierKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 ptxBarrierWarp0
        barrier := { bars := ({} : Std.HashMap Nat BarrierInstance).insert 0 { expectedCount := 1 } } } }

private def ptxBarrierFinalState : State :=
  StepMachine.runN 3 ptxBarrierState

theorem ptx_barrier_kernel_run_functional :
    Reaches ptxBarrierState ptxBarrierFinalState ∧
      barrier0Released ptxBarrierFinalState = true ∧
      lane0HasR1Seven ptxBarrierFinalState = true ∧
      lane0Terminated ptxBarrierFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 ptxBarrierState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

private def ptxAddKernel : PTX.Kernel :=
  { entry := "ptx_add"
    regs := #[
      { name := "r1", ty := .u32 },
      { name := "r2", ty := .u32 },
      { name := "r3", ty := .u32 }
    ]
    blocks := #[{
      label := "ptx_add"
      body := #[({ instr := .add .u32 "r3" (.reg "r1") (.reg "r2") } : PTX.GInstr)]
      term := .exit
    }] }

private def ptxAddLane0 : LaneState :=
  { regs := ({} : Std.HashMap RegName Value)
      |>.insert "r1" (.u32 2)
      |>.insert "r2" (.u32 5)
    pc := ("ptx_add", 0) }

private def ptxAddWarp0 : WarpState :=
  { lanes := (Array.replicate 32 { pc := ("ptx_add", 0) }).set! 0 ptxAddLane0, activeMask := 1 }

private def ptxAddState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD ptxAddKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 ptxAddWarp0 } }

private def ptxAddFinalState : State :=
  StepMachine.runN 2 ptxAddState

example :
    (match PTX.lowerKernelEnvChecked? ptxAddKernel with
     | .ok _ => true
     | .error _ => false) = true := by
  native_decide

theorem ptx_add_kernel_run_functional :
    Reaches ptxAddState ptxAddFinalState ∧
      lane0HasRegU32 "r3" 7 ptxAddFinalState = true ∧
      lane0Terminated ptxAddFinalState = true := by
  refine ⟨StepMachine.runN_reaches 2 ptxAddState, ?_, ?_⟩
  · native_decide
  · native_decide

end CLean
