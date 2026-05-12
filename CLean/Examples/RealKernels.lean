import CLean.Examples.Common
import CLean.PTX.Bridge

namespace CLean

open Helpers

private def upsweepKernelText : String :=
  ".version 8.5
   .target sm_75
   .address_size 64

   .visible .entry upsweepKernel(
     .param .u32 upsweepKernel_param_0,
     .param .u32 upsweepKernel_param_1,
     .param .u32 upsweepKernel_param_2,
     .param .u64 upsweepKernel_param_3
   )
   {
     .reg .pred %p<2>;
     .reg .b32 %r<16>;
     .reg .b64 %rd<7>;

     ld.param.u32 %r4, [upsweepKernel_param_0];
     ld.param.u32 %r5, [upsweepKernel_param_1];
     ld.param.u32 %r3, [upsweepKernel_param_2];
     ld.param.u64 %rd1, [upsweepKernel_param_3];
     mov.u32 %r6, %ntid.x;
     mov.u32 %r7, %ctaid.x;
     mov.u32 %r8, %tid.x;
     mad.lo.s32 %r9, %r7, %r6, %r8;
     mul.lo.s32 %r1, %r9, %r4;
     add.s32 %r2, %r1, %r4;
     setp.gt.s32 %p1, %r2, %r5;
     @%p1 bra $L__BB0_2;

     cvta.to.global.u64 %rd2, %rd1;
     add.s32 %r10, %r2, -1;
     add.s32 %r11, %r1, %r3;
     add.s32 %r12, %r11, -1;
     mul.wide.s32 %rd3, %r10, 4;
     add.s64 %rd4, %rd2, %rd3;
     mul.wide.s32 %rd5, %r12, 4;
     add.s64 %rd6, %rd2, %rd5;
     ld.global.u32 %r13, [%rd6];
     ld.global.u32 %r14, [%rd4];
     add.s32 %r15, %r13, %r14;
     st.global.u32 [%rd4], %r15;

   $L__BB0_2:
     ret;
   }
  "

private def upsweepKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel upsweepKernelText with
  | .ok kernel => kernel
  | .error _ => default

example : PTX.parseAndLowerKernelOk? upsweepKernelText = true := by
  native_decide

example : PTX.lowerKernelSupported? upsweepKernel = true := by
  native_decide

private def writeU32Bytes (mem : ByteMem) (off : Nat) (x : UInt32) : ByteMem :=
  Helpers.writeBytes mem off (Helpers.natToBytesLE x.toNat 4)

private def writeU64Bytes (mem : ByteMem) (off : Nat) (x : UInt64) : ByteMem :=
  Helpers.writeBytes mem off (Helpers.natToBytesLE x.toNat 8)

private def upsweepParamBytes : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 2   -- twod1
  let mem := writeU32Bytes mem 4 4              -- length
  let mem := writeU32Bytes mem 8 4              -- twod
  writeU64Bytes mem 16 0                        -- data pointer

private def upsweepGlobalBytes : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 4 10  -- data[twod1 - 1]
  writeU32Bytes mem 12 5                        -- data[twod - 1]

private def upsweepWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("upsweepKernel", 0) }, activeMask := 1 }

private def upsweepState : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD upsweepKernel
    global := { bytes := upsweepGlobalBytes }
    param := { bytes := upsweepParamBytes }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 upsweepWarp0 } }

private def upsweepFinalState : State :=
  StepMachine.runN 25 upsweepState

private def upsweepDstHasSum (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4) with
  | some (.u32 v) => v == 15
  | _ => false

private def upsweepLane0Terminated (st : State) : Bool :=
  lane0Terminated st

example : (StepMachine.traceN 25 upsweepState).length = 26 := by
  native_decide

theorem cached_upsweep_one_lane_in_bounds_functional :
    Reaches upsweepState upsweepFinalState ∧
      upsweepDstHasSum upsweepFinalState = true ∧
      upsweepLane0Terminated upsweepFinalState = true := by
  refine ⟨StepMachine.runN_reaches 25 upsweepState, ?_, ?_⟩
  · native_decide
  · native_decide

private def upsweepOobParamBytes : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 2   -- twod1
  let mem := writeU32Bytes mem 4 1              -- length forces the guarded branch
  let mem := writeU32Bytes mem 8 4              -- twod
  writeU64Bytes mem 16 0                        -- data pointer

private def upsweepOobState : State :=
  { upsweepState with param := { bytes := upsweepOobParamBytes } }

private def upsweepOobFinalState : State :=
  StepMachine.runN 25 upsweepOobState

private def upsweepDstUnchanged (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4) with
  | some (.u32 v) => v == 10
  | _ => false

theorem cached_upsweep_one_lane_out_of_bounds_functional :
    Reaches upsweepOobState upsweepOobFinalState ∧
      upsweepDstUnchanged upsweepOobFinalState = true ∧
      lane0Terminated upsweepOobFinalState = true := by
  refine ⟨StepMachine.runN_reaches 25 upsweepOobState, ?_, ?_⟩
  · native_decide
  · native_decide

private def saxpyKernelText : String :=
  ".version 8.5
   .target sm_75
   .address_size 64

   .visible .entry saxpyKernel(
     .param .u32 saxpyKernel_param_0,
     .param .f32 saxpyKernel_param_1,
     .param .u64 saxpyKernel_param_2,
     .param .u64 saxpyKernel_param_3,
     .param .u64 saxpyKernel_param_4
   )
   {
     .reg .pred %p<2>;
     .reg .f32 %f<5>;
     .reg .b32 %r<6>;
     .reg .b64 %rd<11>;

     ld.param.u32 %r2, [saxpyKernel_param_0];
     ld.param.f32 %f1, [saxpyKernel_param_1];
     ld.param.u64 %rd1, [saxpyKernel_param_2];
     ld.param.u64 %rd2, [saxpyKernel_param_3];
     ld.param.u64 %rd3, [saxpyKernel_param_4];
     mov.u32 %r3, %ctaid.x;
     mov.u32 %r4, %ntid.x;
     mov.u32 %r5, %tid.x;
     mad.lo.s32 %r1, %r3, %r4, %r5;
     setp.ge.s32 %p1, %r1, %r2;
     @%p1 bra $L__BB0_2;

     cvta.to.global.u64 %rd4, %rd1;
     mul.wide.s32 %rd5, %r1, 4;
     add.s64 %rd6, %rd4, %rd5;
     cvta.to.global.u64 %rd7, %rd2;
     add.s64 %rd8, %rd7, %rd5;
     ld.global.f32 %f2, [%rd6];
     ld.global.f32 %f3, [%rd8];
     fma.rn.f32 %f4, %f2, %f1, %f3;
     cvta.to.global.u64 %rd9, %rd3;
     add.s64 %rd10, %rd9, %rd5;
     st.global.f32 [%rd10], %f4;

   $L__BB0_2:
     ret;
   }
  "

example : PTX.parseAndLowerKernelOk? saxpyKernelText = true := by
  native_decide

end CLean
