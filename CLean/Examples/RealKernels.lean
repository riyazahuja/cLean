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
  StepMachine.runN 40 upsweepState

private def upsweepDstHasSum (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4) with
  | some (.u32 v) => v == 15
  | _ => false

private def upsweepLane0Terminated (st : State) : Bool :=
  lane0Terminated st

theorem cached_upsweep_one_lane_in_bounds_functional :
    Reaches upsweepState upsweepFinalState ∧
      upsweepDstHasSum upsweepFinalState = true ∧
      upsweepLane0Terminated upsweepFinalState = true := by
  refine ⟨StepMachine.runN_reaches 40 upsweepState, ?_, ?_⟩
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
  StepMachine.runN 40 upsweepOobState

private def upsweepDstUnchanged (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4) with
  | some (.u32 v) => v == 10
  | _ => false

theorem cached_upsweep_one_lane_out_of_bounds_functional :
    Reaches upsweepOobState upsweepOobFinalState ∧
      upsweepDstUnchanged upsweepOobFinalState = true ∧
      lane0Terminated upsweepOobFinalState = true := by
  refine ⟨StepMachine.runN_reaches 40 upsweepOobState, ?_, ?_⟩
  · native_decide
  · native_decide

private def downsweepKernelText : String :=
  ".version 8.5
   .target sm_75
   .address_size 64

   .visible .entry downsweepKernel(
     .param .u32 downsweepKernel_param_0,
     .param .u32 downsweepKernel_param_1,
     .param .u32 downsweepKernel_param_2,
     .param .u64 downsweepKernel_param_3
   )
   {
     .reg .pred %p<4>;
     .reg .b32 %r<16>;
     .reg .b64 %rd<7>;

     ld.param.u32 %r3, [downsweepKernel_param_0];
     ld.param.u32 %r4, [downsweepKernel_param_1];
     ld.param.u32 %r5, [downsweepKernel_param_2];
     ld.param.u64 %rd1, [downsweepKernel_param_3];
     mov.u32 %r6, %ntid.x;
     mov.u32 %r7, %ctaid.x;
     mov.u32 %r8, %tid.x;
     mad.lo.s32 %r9, %r7, %r6, %r8;
     mul.lo.s32 %r10, %r9, %r3;
     add.s32 %r1, %r10, %r4;
     setp.gt.s32 %p1, %r1, %r5;
     add.s32 %r2, %r10, %r3;
     setp.gt.s32 %p2, %r2, %r5;
     or.pred %p3, %p2, %p1;
     @%p3 bra $L__BB0_2;

     cvta.to.global.u64 %rd2, %rd1;
     add.s32 %r11, %r2, -1;
     add.s32 %r12, %r1, -1;
     mul.wide.s32 %rd3, %r12, 4;
     add.s64 %rd4, %rd2, %rd3;
     ld.global.u32 %r13, [%rd4];
     mul.wide.s32 %rd5, %r11, 4;
     add.s64 %rd6, %rd2, %rd5;
     ld.global.u32 %r14, [%rd6];
     st.global.u32 [%rd4], %r14;
     add.s32 %r15, %r14, %r13;
     st.global.u32 [%rd6], %r15;

   $L__BB0_2:
     ret;
   }
  "

private def downsweepKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel downsweepKernelText with
  | .ok kernel => kernel
  | .error _ => default

example : PTX.parseAndLowerKernelOk? downsweepKernelText = true := by
  native_decide

example : PTX.lowerKernelSupported? downsweepKernel = true := by
  native_decide

private def downsweepParamBytes (length : UInt32) : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 2
  let mem := writeU32Bytes mem 4 4
  let mem := writeU32Bytes mem 8 length
  writeU64Bytes mem 16 0

private def downsweepGlobalBytes : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 4 10
  writeU32Bytes mem 12 5

private def downsweepWarp0 : WarpState :=
  { lanes := Array.replicate 32 { pc := ("downsweepKernel", 0) }, activeMask := 1 }

private def downsweepStateWithLength (length : UInt32) : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD downsweepKernel
    global := { bytes := downsweepGlobalBytes }
    param := { bytes := downsweepParamBytes length }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 downsweepWarp0 } }

private def downsweepFinalState : State :=
  StepMachine.runN 40 (downsweepStateWithLength 4)

private def downsweepInBoundsPost (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4),
        Helpers.readMem? st .global .u32 (.global 12) with
  | some (.u32 a), some (.u32 b) => a == 15 && b == 10
  | _, _ => false

theorem cached_downsweep_one_lane_in_bounds_functional :
    Reaches (downsweepStateWithLength 4) downsweepFinalState ∧
      downsweepInBoundsPost downsweepFinalState = true ∧
      lane0Terminated downsweepFinalState = true := by
  refine ⟨StepMachine.runN_reaches 40 (downsweepStateWithLength 4), ?_, ?_⟩
  · native_decide
  · native_decide

private def downsweepOobFinalState : State :=
  StepMachine.runN 40 (downsweepStateWithLength 1)

private def downsweepOobPost (st : State) : Bool :=
  match Helpers.readMem? st .global .u32 (.global 4),
        Helpers.readMem? st .global .u32 (.global 12) with
  | some (.u32 a), some (.u32 b) => a == 10 && b == 5
  | _, _ => false

theorem cached_downsweep_one_lane_out_of_bounds_functional :
    Reaches (downsweepStateWithLength 1) downsweepOobFinalState ∧
      downsweepOobPost downsweepOobFinalState = true ∧
      lane0Terminated downsweepOobFinalState = true := by
  refine ⟨StepMachine.runN_reaches 40 (downsweepStateWithLength 1), ?_, ?_⟩
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

private def saxpyKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel saxpyKernelText with
  | .ok kernel => kernel
  | .error _ => default

example : PTX.lowerKernelSupported? saxpyKernel = true := by
  native_decide

def saxpyXBase : Nat := 0
def saxpyYBase : Nat := 128
def saxpyRBase : Nat := 256

private def saxpyParamBytesFor (n : Nat) (alpha : Float) : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)
  let mem := writeF32Bytes mem 4 alpha
  let mem := writeU64Bytes mem 8 (UInt64.ofNat saxpyXBase)
  let mem := writeU64Bytes mem 16 (UInt64.ofNat saxpyYBase)
  writeU64Bytes mem 24 (UInt64.ofNat saxpyRBase)

private def saxpyGlobalBytesFor (xs ys : List Float) : ByteMem :=
  let mem := writeF32Vector ({} : ByteMem) saxpyXBase xs
  writeF32Vector mem saxpyYBase ys

private def vectorWarpFor (entry : BlockLabel) (n : Nat) : WarpState :=
  { lanes := Array.replicate 32 { pc := (entry, 0) }
    activeMask := activeMaskPrefix n }

private def saxpyWarpFor (n : Nat) : WarpState :=
  vectorWarpFor "saxpyKernel" n

def saxpyStateFor (n : Nat) (alpha : Float) (xs ys : List Float) : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD saxpyKernel
    global := { bytes := saxpyGlobalBytesFor xs ys }
    param := { bytes := saxpyParamBytesFor n alpha }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 (saxpyWarpFor n) } }

def saxpyExpectedVector (alpha : Float) (xs ys : List Float) : List Float :=
  (xs.zip ys).map fun pair => pair.1 * alpha + pair.2

def saxpyVectorPost (expected : List Float) (st : State) : Prop :=
  globalF32VectorMatches? st saxpyRBase expected = true

theorem saxpy_partial_correct_summary
    (n : Nat) (hn : n ≤ 32) (alpha : Float) (xs ys : List Float)
    (hxs : xs.length = n) (hys : ys.length = n) :
    PartialCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by
  -- PTX proof obligation: all terminating executions of the lowered SAXPY CFG
  -- preserve the input regions and write `x[i] * alpha + y[i]` to `r[i]` for
  -- each active lane `i < n`.
  sorry

theorem saxpy_total_correct_summary
    (n : Nat) (hn : n ≤ 32) (alpha : Float) (xs ys : List Float)
    (hxs : xs.length = n) (hys : ys.length = n) :
    TotalCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by
  -- Termination follows from the acyclic SAXPY CFG under the milestone lockstep
  -- one-warp launch restriction; the postcondition is supplied by partial correctness.
  sorry

private def saxpyConcreteAlpha : Float := 2.0
private def saxpyConcreteXs : List Float := [1.0, 2.0, 3.0, 4.0]
private def saxpyConcreteYs : List Float := [10.0, 20.0, 30.0, 40.0]
private def saxpyConcreteExpected : List Float :=
  saxpyExpectedVector saxpyConcreteAlpha saxpyConcreteXs saxpyConcreteYs

private def saxpyConcreteState : State :=
  saxpyStateFor 4 saxpyConcreteAlpha saxpyConcreteXs saxpyConcreteYs

private def saxpyConcreteFinalState : State :=
  StepMachine.runN 40 saxpyConcreteState

private def floatListEq? : List Float → List Float → Bool
  | [], [] => true
  | x :: xs, y :: ys => x == y && floatListEq? xs ys
  | _, _ => false

example : floatListEq? saxpyConcreteExpected [12.0, 24.0, 36.0, 48.0] = true := by
  native_decide

example : globalF32VectorMatches? saxpyConcreteFinalState saxpyRBase saxpyConcreteExpected = true := by
  native_decide

theorem cached_saxpy_vector4_functional :
    Reaches saxpyConcreteState saxpyConcreteFinalState ∧
      saxpyVectorPost saxpyConcreteExpected saxpyConcreteFinalState := by
  refine ⟨StepMachine.runN_reaches 40 saxpyConcreteState, ?_⟩
  unfold saxpyVectorPost
  native_decide

theorem saxpy_vector_correct_surface
    (n : Nat) (hn : n ≤ 32) (alpha : Float) (xs ys : List Float)
    (hxs : xs.length = n) (hys : ys.length = n) :
    PartialCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by
  exact saxpy_partial_correct_summary n hn alpha xs ys hxs hys

theorem saxpy_total_correct_supported_launch
    (n : Nat) (hn : n ≤ 32) (alpha : Float) (xs ys : List Float)
    (hxs : xs.length = n) (hys : ys.length = n) :
    TotalCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by
  exact saxpy_total_correct_summary n hn alpha xs ys hxs hys


private def testKernelText : String :=
  saxpyKernelText.replace "saxpyKernel" "testKernel" |>.replace "saxpyKernel_param" "testKernel_param"

example : PTX.parseAndLowerKernelOk? testKernelText = true := by
  native_decide

private def testKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel testKernelText with
  | .ok kernel => kernel
  | .error _ => default

example : PTX.lowerKernelSupported? testKernel = true := by
  native_decide

private def testStateFor (n : Nat) (alpha : Float) (xs ys : List Float) : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD testKernel
    global := { bytes := saxpyGlobalBytesFor xs ys }
    param := { bytes := saxpyParamBytesFor n alpha }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 (vectorWarpFor "testKernel" n) } }

private def testConcreteState : State :=
  testStateFor 4 saxpyConcreteAlpha saxpyConcreteXs saxpyConcreteYs

private def testConcreteFinalState : State :=
  StepMachine.runN 40 testConcreteState

theorem cached_test_kernel_vector4_functional :
    Reaches testConcreteState testConcreteFinalState ∧
      saxpyVectorPost saxpyConcreteExpected testConcreteFinalState := by
  refine ⟨StepMachine.runN_reaches 40 testConcreteState, ?_⟩
  unfold saxpyVectorPost
  native_decide

private def matmulKernelText : String :=
  "//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-34385749
// Cuda compilation tools, release 12.5, V12.5.82
// Based on NVVM 7.0.1
//

.version 8.5
.target sm_75
.address_size 64

	// .globl	matmulKernel

.visible .entry matmulKernel(
	.param .u32 matmulKernel_param_0,
	.param .u64 matmulKernel_param_1,
	.param .u64 matmulKernel_param_2,
	.param .u64 matmulKernel_param_3
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<34>;


	ld.param.u32 	%r13, [matmulKernel_param_0];
	ld.param.u64 	%rd18, [matmulKernel_param_1];
	ld.param.u64 	%rd19, [matmulKernel_param_2];
	ld.param.u64 	%rd17, [matmulKernel_param_3];
	cvta.to.global.u64 	%rd1, %rd19;
	cvta.to.global.u64 	%rd2, %rd18;
	mov.u32 	%r14, %ntid.x;
	mov.u32 	%r15, %ctaid.x;
	mov.u32 	%r16, %tid.x;
	mad.lo.s32 	%r1, %r15, %r14, %r16;
	mov.u32 	%r17, %ntid.y;
	mov.u32 	%r18, %ctaid.y;
	mov.u32 	%r19, %tid.y;
	mad.lo.s32 	%r2, %r18, %r17, %r19;
	setp.ge.s32 	%p1, %r1, %r13;
	setp.ge.s32 	%p2, %r2, %r13;
	or.pred  	%p3, %p1, %p2;
	@%p3 bra 	$L__BB0_9;

	setp.lt.s32 	%p4, %r13, 1;
	mul.lo.s32 	%r3, %r1, %r13;
	mov.f32 	%f29, 0f00000000;
	@%p4 bra 	$L__BB0_8;

	add.s32 	%r21, %r13, -1;
	and.b32  	%r29, %r13, 3;
	setp.lt.u32 	%p5, %r21, 3;
	mov.f32 	%f29, 0f00000000;
	mov.u32 	%r28, 0;
	@%p5 bra 	$L__BB0_5;

	sub.s32 	%r27, %r13, %r29;
	mul.wide.s32 	%rd3, %r3, 4;
	mul.wide.s32 	%rd20, %r2, 4;
	add.s64 	%rd30, %rd1, %rd20;
	mul.wide.s32 	%rd5, %r13, 4;
	mov.f32 	%f29, 0f00000000;
	mov.u32 	%r28, 0;
	mov.u64 	%rd31, %rd2;

$L__BB0_4:
	add.s64 	%rd21, %rd31, %rd3;
	ld.global.f32 	%f12, [%rd30];
	ld.global.f32 	%f13, [%rd21];
	fma.rn.f32 	%f14, %f13, %f12, %f29;
	add.s64 	%rd22, %rd30, %rd5;
	ld.global.f32 	%f15, [%rd22];
	ld.global.f32 	%f16, [%rd21+4];
	fma.rn.f32 	%f17, %f16, %f15, %f14;
	add.s64 	%rd23, %rd22, %rd5;
	ld.global.f32 	%f18, [%rd23];
	ld.global.f32 	%f19, [%rd21+8];
	fma.rn.f32 	%f20, %f19, %f18, %f17;
	add.s64 	%rd24, %rd23, %rd5;
	add.s64 	%rd30, %rd24, %rd5;
	ld.global.f32 	%f21, [%rd24];
	ld.global.f32 	%f22, [%rd21+12];
	fma.rn.f32 	%f29, %f22, %f21, %f20;
	add.s32 	%r28, %r28, 4;
	add.s64 	%rd31, %rd31, 16;
	add.s32 	%r27, %r27, -4;
	setp.ne.s32 	%p6, %r27, 0;
	@%p6 bra 	$L__BB0_4;

$L__BB0_5:
	setp.eq.s32 	%p7, %r29, 0;
	@%p7 bra 	$L__BB0_8;

	mad.lo.s32 	%r23, %r28, %r13, %r2;
	mul.wide.s32 	%rd25, %r23, 4;
	add.s64 	%rd33, %rd1, %rd25;
	mul.wide.s32 	%rd11, %r13, 4;
	add.s32 	%r24, %r28, %r3;
	mul.wide.s32 	%rd26, %r24, 4;
	add.s64 	%rd32, %rd2, %rd26;

$L__BB0_7:
	.pragma \"nounroll\";
	ld.global.f32 	%f23, [%rd33];
	ld.global.f32 	%f24, [%rd32];
	fma.rn.f32 	%f29, %f24, %f23, %f29;
	add.s64 	%rd33, %rd33, %rd11;
	add.s64 	%rd32, %rd32, 4;
	add.s32 	%r29, %r29, -1;
	setp.ne.s32 	%p8, %r29, 0;
	@%p8 bra 	$L__BB0_7;

$L__BB0_8:
	add.s32 	%r25, %r3, %r2;
	cvta.to.global.u64 	%rd27, %rd17;
	mul.wide.s32 	%rd28, %r25, 4;
	add.s64 	%rd29, %rd27, %rd28;
	st.global.f32 	[%rd29], %f29;

$L__BB0_9:
	ret;

}

"


private def matmulKernel : PTX.Kernel :=
  match PTX.Parser.parseKernel matmulKernelText with
  | .ok kernel => kernel
  | .error _ => default

example : PTX.parseAndLowerKernelOk? matmulKernelText = true := by
  native_decide

example : PTX.lowerKernelSupported? matmulKernel = true := by
  native_decide

def matmulABase : Nat := 0
def matmulBBase : Nat := 128
def matmulCBase : Nat := 256

private def matmulCTAIdForCell (n row col : Nat) : CTAId :=
  row + col * n

private def matmulParamBytesFor (n : Nat) : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)
  let mem := writeU64Bytes mem 8 (UInt64.ofNat matmulABase)
  let mem := writeU64Bytes mem 16 (UInt64.ofNat matmulBBase)
  writeU64Bytes mem 24 (UInt64.ofNat matmulCBase)

private def matmulGlobalBytesFor (a b : List Float) : ByteMem :=
  let mem := writeF32Vector ({} : ByteMem) matmulABase a
  writeF32Vector mem matmulBBase b

private def matmulKernelEnvFor (n : Nat) : KernelEnv :=
  let env := PTX.lowerKernelEnvCheckedD matmulKernel
  { env with gridCtx := { gridDim := { x := n, y := n }, blockDim := { x := 1, y := 1 } } }

private def matmulWarpForCell : WarpState :=
  { lanes := Array.replicate 32 { pc := ("matmulKernel", 0) }, activeMask := 1 }

def matmulStateForCell (n row col : Nat) (a b : List Float) : State :=
  { kernelEnv := matmulKernelEnvFor n
    global := { bytes := matmulGlobalBytesFor a b }
    param := { bytes := matmulParamBytesFor n }
    ctas := ({} : Std.HashMap CTAId CTAState).insert (matmulCTAIdForCell n row col)
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 matmulWarpForCell } }

def matmulOneCellPost
    (n row col : Nat) (a b : List Float) (st : State) : Prop :=
  matmulCellPost n row col matmulCBase a b st

theorem matmul_cell_partial_correct_summary
    (n row col : Nat) (hn : n > 0) (hrow : row < n) (hcol : col < n)
    (a b : List Float) (ha : a.length = n * n) (hb : b.length = n * n) :
    PartialCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by
  -- PTX proof obligation: the generated loop at `$L__BB0_7`/`$L__BB0_4`
  -- implements the dot-product recurrence for the selected `(row,col)` CTA.
  -- The proof should instantiate `CountedLoopSpec.total_correct`/partial loop
  -- invariants with accumulator register `%f29` and loop counter `%r29/%r28`.
  sorry

theorem matmul_cell_total_correct_summary
    (n row col : Nat) (hn : n > 0) (hrow : row < n) (hcol : col < n)
    (a b : List Float) (ha : a.length = n * n) (hb : b.length = n * n) :
    TotalCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by
  -- Termination follows from the decreasing loop counters emitted by NVVM for
  -- the unrolled-by-four main loop and scalar remainder loop.
  sorry

private def matmulConcreteA : List Float := [3.0]
private def matmulConcreteB : List Float := [4.0]

private def matmulConcreteState : State :=
  matmulStateForCell 1 0 0 matmulConcreteA matmulConcreteB

private def matmulConcreteFinalState : State :=
  StepMachine.runN 80 matmulConcreteState

private def matmulConcretePost? (st : State) : Bool :=
  match globalF32At? st matmulCBase (matrixIndex 1 0 0) with
  | some out => out == dotF32List 1 0 0 matmulConcreteA matmulConcreteB
  | none => false

example : (dotF32List 1 0 0 matmulConcreteA matmulConcreteB == 12.0) = true := by
  native_decide

theorem cached_matmul_n1_one_cell_functional :
    Reaches matmulConcreteState matmulConcreteFinalState ∧
      matmulConcretePost? matmulConcreteFinalState = true := by
  refine ⟨StepMachine.runN_reaches 80 matmulConcreteState, ?_⟩
  native_decide

theorem matmul_one_cell_correct_surface
    (n row col : Nat) (hn : n > 0) (hrow : row < n) (hcol : col < n)
    (a b : List Float) (ha : a.length = n * n) (hb : b.length = n * n) :
    PartialCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by
  exact matmul_cell_partial_correct_summary n row col hn hrow hcol a b ha hb

theorem matmul_cell_total_correct_supported_launch
    (n row col : Nat) (hn : n > 0) (hrow : row < n) (hcol : col < n)
    (a b : List Float) (ha : a.length = n * n) (hb : b.length = n * n) :
    TotalCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by
  exact matmul_cell_total_correct_summary n row col hn hrow hcol a b ha hb


end CLean
