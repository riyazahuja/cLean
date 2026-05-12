import CLean.Examples.Common

namespace CLean

open Helpers
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


end CLean
