import CLean.Examples.Common

namespace CLean

open Helpers
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
