import CLean.PTXParser
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

private def lane0HasRegS32 (reg : RegName) (value : Int) (st : State) : Bool :=
  match st.getLane? 0 0 lane0 with
  | some laneState =>
      match laneState.regs[reg]? with
      | some (.s32 v) => decide (v = value)
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
    PTX.lowerInstr (.binop .add .u32 "r3" (.reg "r1") (.reg "r2")) =
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
    (match PTX.lowerInstrChecked? {} (.binop .add .u32 "r3" (.reg "r1") (.reg "r2")) with
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
      body := #[({ instr := .binop .add .u32 "r3" (.reg "r1") (.reg "r2") } : PTX.GInstr)]
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

private def parsedAssignText : String :=
  ".entry parsed_assign;
   .reg .u32 %r1;
   parsed_assign:
     mov.u32 %r1, 7;
     exit;
  "

private def parsedAssignKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedAssignText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseKernel parsedAssignText with
     | .ok kernel => kernel.entry == "parsed_assign" && kernel.regs.size == 1 && kernel.blocks.size == 1
     | .error _ => false) = true := by
  native_decide

private def parsedAssignWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_assign", 0) }, activeMask := 1 }

private def parsedAssignState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedAssignKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedAssignWarp0 } }

private def parsedAssignFinalState : State :=
  StepMachine.runN 2 parsedAssignState

theorem parsed_ptx_assign_kernel_run_functional :
    Reaches parsedAssignState parsedAssignFinalState ∧
      lane0HasR1Seven parsedAssignFinalState = true ∧
      lane0Terminated parsedAssignFinalState = true := by
  refine ⟨StepMachine.runN_reaches 2 parsedAssignState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedAddText : String :=
  ".entry parsed_add;
   .reg .u32 %r1;
   .reg .u32 %r2;
   .reg .u32 %r3;
   parsed_add:
     add.u32 %r3, %r1, %r2;
     exit;
  "

private def parsedAddKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedAddText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseKernel parsedAddText with
     | .ok kernel => kernel.entry == "parsed_add" && kernel.regs.size == 3 && kernel.blocks.size == 1
     | .error _ => false) = true := by
  native_decide

private def parsedAddLane0 : LaneState :=
  { regs := ({} : Std.HashMap RegName Value)
      |>.insert "r1" (.u32 2)
      |>.insert "r2" (.u32 5)
    pc := ("parsed_add", 0) }

private def parsedAddWarp0 : WarpState :=
  { lanes := (Array.replicate 32 { pc := ("parsed_add", 0) }).set! 0 parsedAddLane0, activeMask := 1 }

private def parsedAddState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedAddKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedAddWarp0 } }

private def parsedAddFinalState : State :=
  StepMachine.runN 2 parsedAddState

theorem parsed_ptx_add_kernel_run_functional :
    Reaches parsedAddState parsedAddFinalState ∧
      lane0HasRegU32 "r3" 7 parsedAddFinalState = true ∧
      lane0Terminated parsedAddFinalState = true := by
  refine ⟨StepMachine.runN_reaches 2 parsedAddState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedCopyText : String :=
  ".entry parsed_copy;
   .reg .u32 %r1;
   parsed_copy:
     ld.global.u32 %r1, 0;
     st.global.u32 4, %r1;
     exit;
  "

private def parsedCopyKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedCopyText with
  | .ok kernel => kernel
  | .error _ => default

private def parsedCopyWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_copy", 0) }, activeMask := 1 }

private def parsedCopyState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedCopyKernel
    global := { bytes := Helpers.writeBytes ({} : ByteMem) copySrcOffset (Helpers.natToBytesLE copyValue.toNat 4) }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedCopyWarp0 } }

private def parsedCopyFinalState : State :=
  StepMachine.runN 3 parsedCopyState

theorem parsed_ptx_copy_kernel_run_functional :
    Reaches parsedCopyState parsedCopyFinalState ∧
      copyDstHasValue parsedCopyFinalState = true ∧
      copyLane0Terminated parsedCopyFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 parsedCopyState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedParamCopyText : String :=
  ".entry parsed_param_copy;
   .param .ptr.global.align 4 .u64 src;
   .param .ptr.global.align 4 .u64 dst;
   .reg .u64 %srcp;
   .reg .u64 %dstp;
   .reg .u32 %r1;
   parsed_param_copy:
     ld.param.u64 %srcp, src;
     ld.param.u64 %dstp, dst;
     ld.global.u32 %r1, %srcp;
     st.global.u32 %dstp, %r1;
     exit;
  "

private def parsedParamCopyKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedParamCopyText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseKernel parsedParamCopyText with
     | .ok kernel => kernel.entry == "parsed_param_copy" &&
        kernel.params.size == 2 && kernel.regs.size == 3 && kernel.blocks.size == 1
     | .error _ => false) = true := by
  native_decide

example :
    (match PTX.lowerKernelEnvChecked? parsedParamCopyKernel with
     | .ok env =>
        env.params.size == 2 &&
        env.params[0]?.map (fun p => p.offset) == some 0 &&
        env.params[1]?.map (fun p => p.offset) == some 8
     | .error _ => false) = true := by
  native_decide

private def parsedParamCopyParamBytes : ByteMem :=
  let mem := Helpers.writeBytes ({} : ByteMem) 0 (Helpers.natToBytesLE copySrcOffset 8)
  Helpers.writeBytes mem 8 (Helpers.natToBytesLE copyDstOffset 8)

private def parsedParamCopyWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_param_copy", 0) }, activeMask := 1 }

private def parsedParamCopyState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedParamCopyKernel
    global := { bytes := Helpers.writeBytes ({} : ByteMem) copySrcOffset (Helpers.natToBytesLE copyValue.toNat 4) }
    param := { bytes := parsedParamCopyParamBytes }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedParamCopyWarp0 } }

private def parsedParamCopyFinalState : State :=
  StepMachine.runN 5 parsedParamCopyState

theorem parsed_ptx_param_copy_kernel_run_functional :
    Reaches parsedParamCopyState parsedParamCopyFinalState ∧
      copyDstHasValue parsedParamCopyFinalState = true ∧
      copyLane0Terminated parsedParamCopyFinalState = true := by
  refine ⟨StepMachine.runN_reaches 5 parsedParamCopyState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedBarrierText : String :=
  ".entry parsed_barrier;
   .reg .u32 %r1;
   parsed_barrier:
     bar.sync 0;
     mov.u32 %r1, 7;
     exit;
  "

private def parsedBarrierKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedBarrierText with
  | .ok kernel => kernel
  | .error _ => default

private def parsedBarrierWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_barrier", 0) }, activeMask := 1 }

private def parsedBarrierState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedBarrierKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedBarrierWarp0
        barrier := { bars := ({} : Std.HashMap Nat BarrierInstance).insert 0 { expectedCount := 1 } } } }

private def parsedBarrierFinalState : State :=
  StepMachine.runN 3 parsedBarrierState

theorem parsed_ptx_barrier_kernel_run_functional :
    Reaches parsedBarrierState parsedBarrierFinalState ∧
      barrier0Released parsedBarrierFinalState = true ∧
      lane0HasR1Seven parsedBarrierFinalState = true ∧
      lane0Terminated parsedBarrierFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 parsedBarrierState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

private def parsedSharedText : String :=
  ".entry parsed_shared;
   .shared .align 4 .u32 slot;
   .reg .u32 %r1;
   parsed_shared:
     st.shared.u32 slot, 7;
     bar.sync 0;
     ld.shared.u32 %r1, slot;
     exit;
  "

private def parsedSharedKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedSharedText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseKernel parsedSharedText with
     | .ok kernel => kernel.entry == "parsed_shared" &&
        kernel.shareds.size == 1 && kernel.regs.size == 1 && kernel.blocks.size == 1
     | .error _ => false) = true := by
  native_decide

example :
    (match PTX.lowerKernelEnvChecked? parsedSharedKernel with
     | .ok env =>
        env.sharedDecls.size == 1 &&
        env.sharedDecls[0]?.map (fun decl => decl.offset) == some 0 &&
        env.sharedDecls[0]?.map (fun decl => decl.size) == some 4
     | .error _ => false) = true := by
  native_decide

private def parsedSharedWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_shared", 0) }, activeMask := 1 }

private def parsedSharedState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedSharedKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedSharedWarp0
        barrier := { bars := ({} : Std.HashMap Nat BarrierInstance).insert 0 { expectedCount := 1 } } } }

private def parsedSharedFinalState : State :=
  StepMachine.runN 4 parsedSharedState

theorem parsed_ptx_shared_barrier_kernel_run_functional :
    Reaches parsedSharedState parsedSharedFinalState ∧
      barrier0Released parsedSharedFinalState = true ∧
      lane0HasR1Seven parsedSharedFinalState = true ∧
      lane0Terminated parsedSharedFinalState = true := by
  refine ⟨StepMachine.runN_reaches 4 parsedSharedState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

private def parsedBraText : String :=
  ".entry parsed_bra;
   .reg .u32 %r1;
   parsed_bra:
     bra target;
   target:
     mov.u32 %r1, 7;
     exit;
  "

private def parsedBraKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedBraText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.lowerKernelEnvChecked? parsedBraKernel with
     | .ok env => env.blocks["parsed_bra"]?.isSome && env.blocks["target"]?.isSome
     | .error _ => false) = true := by
  native_decide

private def parsedBraWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_bra", 0) }, activeMask := 1 }

private def parsedBraState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedBraKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedBraWarp0 } }

private def parsedBraFinalState : State :=
  StepMachine.runN 3 parsedBraState

theorem parsed_ptx_bra_kernel_run_functional :
    Reaches parsedBraState parsedBraFinalState ∧
      lane0HasR1Seven parsedBraFinalState = true ∧
      lane0Terminated parsedBraFinalState = true := by
  refine ⟨StepMachine.runN_reaches 3 parsedBraState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedCbraText : String :=
  ".entry parsed_cbra;
   .reg .u32 %r1;
   .pred %p;
   parsed_cbra:
     setp.eq.u32 %p, 1, 1;
     cbra %p, then_blk, else_blk;
   then_blk:
     mov.u32 %r1, 7;
     exit;
   else_blk:
     mov.u32 %r1, 0;
     exit;
  "

private def parsedCbraKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedCbraText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.lowerKernelEnvChecked? parsedCbraKernel with
     | .ok env =>
        env.blocks["parsed_cbra"]?.isSome &&
        env.blocks["then_blk"]?.isSome &&
        env.blocks["else_blk"]?.isSome
     | .error _ => false) = true := by
  native_decide

private def parsedCbraWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_cbra", 0) }, activeMask := 1 }

private def parsedCbraState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedCbraKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedCbraWarp0 } }

private def parsedCbraFinalState : State :=
  StepMachine.runN 4 parsedCbraState

theorem parsed_ptx_cbra_kernel_run_functional :
    Reaches parsedCbraState parsedCbraFinalState ∧
      lane0HasR1Seven parsedCbraFinalState = true ∧
      lane0Terminated parsedCbraFinalState = true := by
  refine ⟨StepMachine.runN_reaches 4 parsedCbraState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedScalarOpsText : String :=
  ".entry parsed_scalar_ops;
   .reg .u32 %r1;
   .reg .u32 %r2;
   .reg .u32 %r3;
   .reg .u32 %r4;
   .reg .u32 %r5;
   .reg .u32 %r6;
   .reg .u32 %r7;
   .reg .u32 %r8;
   .reg .u32 %r9;
   parsed_scalar_ops:
     mov.u32 %r1, 12;
     mov.u32 %r2, 5;
     sub.u32 %r3, %r1, %r2;
     mul.u32 %r4, %r3, 3;
     and.u32 %r5, %r4, 15;
     or.u32 %r6, %r5, 8;
     xor.u32 %r7, %r6, 1;
     shl.u32 %r8, %r7, 1;
     shr.u32 %r9, %r8, 2;
     exit;
  "

private def parsedScalarOpsKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedScalarOpsText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseKernel parsedScalarOpsText with
     | .ok kernel => kernel.entry == "parsed_scalar_ops" && kernel.regs.size == 9 && kernel.blocks.size == 1
     | .error _ => false) = true := by
  native_decide

private def parsedScalarOpsWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_scalar_ops", 0) }, activeMask := 1 }

private def parsedScalarOpsState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedScalarOpsKernel
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedScalarOpsWarp0 } }

private def parsedScalarOpsFinalState : State :=
  StepMachine.runN 10 parsedScalarOpsState

theorem parsed_ptx_scalar_ops_kernel_run_functional :
    Reaches parsedScalarOpsState parsedScalarOpsFinalState ∧
      lane0HasRegU32 "r3" 7 parsedScalarOpsFinalState = true ∧
      lane0HasRegU32 "r9" 6 parsedScalarOpsFinalState = true ∧
      lane0Terminated parsedScalarOpsFinalState = true := by
  refine ⟨StepMachine.runN_reaches 10 parsedScalarOpsState, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide

private def parsedBadScalarText : String :=
  ".entry parsed_bad_scalar;
   .reg .u32 %r1;
   .reg .u64 %r2;
   .reg .u32 %r3;
   parsed_bad_scalar:
     add.u32 %r3, %r1, %r2;
     exit;
  "

private def parsedBadScalarKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedBadScalarText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.lowerKernelEnvChecked? parsedBadScalarKernel with
     | .error (.typeMismatch .u32 .u64) => true
     | _ => false) = true := by
  native_decide

private def parsedBadBraText : String :=
  ".entry parsed_bad_bra;
   parsed_bad_bra:
     bra missing_target;
  "

private def parsedBadBraKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedBadBraText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.lowerKernelEnvChecked? parsedBadBraKernel with
     | .error (.unknownBlock "missing_target") => true
     | _ => false) = true := by
  native_decide

private def parsedRealModuleText : String :=
  ".version 7.8;
   .target sm_80, texmode_independent;
   .address_size 64;
   .global .align 4 .u32 gbuf[2];
   /* Real-style entry syntax with params, braces, vector regs, and bracket addresses. */
   .entry parsed_real(
     .param .ptr.global.align 4 .u64 src,
     .param .ptr.global.align 4 .u64 dst
   ) {
     .reg .u64 %rd<2>;
     .reg .u32 %r<2>;
     ld.param.u64 %rd0, [src];
     ld.param.u64 %rd1, [dst];
     ld.global.u32 %r0, [%rd0+0x0];
     st.global.u32 [%rd1+0], %r0;
     ret;
   }
  "

private def parsedRealModule : PTX.Module :=
  match PTX.Parser.parseModule parsedRealModuleText with
  | .ok m => m
  | .error _ => default

private def parsedRealKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedRealModuleText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.Parser.parseModule parsedRealModuleText with
     | .ok m => m.directives.size == 3 && m.memories.size == 1 && m.kernels.size == 1
     | .error _ => false) = true := by
  native_decide

example :
    parsedRealKernel.params.size = 2 ∧
      parsedRealKernel.regs.size = 4 ∧
      parsedRealKernel.blocks.size = 1 := by
  native_decide

private def parsedRealParamBytes : ByteMem :=
  let mem := Helpers.writeBytes ({} : ByteMem) 0 (Helpers.natToBytesLE copySrcOffset 8)
  Helpers.writeBytes mem 8 (Helpers.natToBytesLE copyDstOffset 8)

private def parsedRealWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_real", 0) }, activeMask := 1 }

private def parsedRealState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedRealKernel
    global := { bytes := Helpers.writeBytes ({} : ByteMem) copySrcOffset (Helpers.natToBytesLE copyValue.toNat 4) }
    param := { bytes := parsedRealParamBytes }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedRealWarp0 } }

private def parsedRealFinalState : State :=
  StepMachine.runN 5 parsedRealState

theorem parsed_real_style_param_copy_kernel_run_functional :
    Reaches parsedRealState parsedRealFinalState ∧
      copyDstHasValue parsedRealFinalState = true ∧
      lane0Terminated parsedRealFinalState = true := by
  refine ⟨StepMachine.runN_reaches 5 parsedRealState, ?_, ?_⟩
  · native_decide
  · native_decide

private def parsedUnaryCvtText : String :=
  ".entry parsed_unary_cvt;
   .reg .u32 %r<2>;
   .reg .u64 %rd<1>;
   .reg .s32 %s<3>;
   parsed_unary_cvt:
     mov.u32 %r0, 0x4;
     cvt.u64.u32 %rd0, %r0;
     ld.global.u32 %r1, [%rd0];
     mov.s32 %s0, -5;
     neg.s32 %s1, %s0;
     abs.s32 %s2, %s0;
     exit;
  "

private def parsedUnaryCvtKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedUnaryCvtText with
  | .ok kernel => kernel
  | .error _ => default

private def parsedUnaryCvtWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("parsed_unary_cvt", 0) }, activeMask := 1 }

private def parsedUnaryCvtState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD parsedUnaryCvtKernel
    global := { bytes := Helpers.writeBytes ({} : ByteMem) 4 (Helpers.natToBytesLE copyValue.toNat 4) }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 parsedUnaryCvtWarp0 } }

private def parsedUnaryCvtFinalState : State :=
  StepMachine.runN 7 parsedUnaryCvtState

theorem parsed_ptx_unary_cvt_kernel_run_functional :
    Reaches parsedUnaryCvtState parsedUnaryCvtFinalState ∧
      lane0HasRegU32 "r1" copyValue parsedUnaryCvtFinalState = true ∧
      lane0HasRegS32 "s1" 5 parsedUnaryCvtFinalState = true ∧
      lane0HasRegS32 "s2" 5 parsedUnaryCvtFinalState = true ∧
      lane0Terminated parsedUnaryCvtFinalState = true := by
  refine ⟨StepMachine.runN_reaches 7 parsedUnaryCvtState, ?_, ?_, ?_, ?_⟩
  · native_decide
  · native_decide
  · native_decide
  · native_decide

private def parsedBadModifierText : String :=
  ".entry parsed_bad_modifier;
   .reg .u32 %r0;
   parsed_bad_modifier:
     ld.volatile.global.u32 %r0, [0];
     exit;
  "

private def parsedBadModifierKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel parsedBadModifierText with
  | .ok kernel => kernel
  | .error _ => default

example :
    (match PTX.lowerKernelEnvChecked? parsedBadModifierKernel with
     | .error (.unsupportedModifier "volatile") => true
     | _ => false) = true := by
  native_decide

end CLean
