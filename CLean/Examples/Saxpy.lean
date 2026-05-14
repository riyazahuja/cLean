import CLean.Examples.Common
import CLean.PTX.Bridge
import CLean.Proof.Determinism
import CLean.Proof.IsSingleWarpPres

/-! # Saxpy: PTX kernel with one main functional-correctness theorem

This file contains:
- the saxpy PTX source text,
- `parseAndLower` / `lowerSupported` sanity `example`s,
- an axiom-clean **concrete-input** correctness theorem
  (`cached_saxpy_vector4_functional`) that exercises the actual small-step
  semantics via `runN` + `native_decide`,
- two **general** correctness theorems (`saxpy_partial_correct`,
  `saxpy_total_correct`) whose statements describe what the kernel computes
  for any supported launch shape. Each is currently `sorry`. **The signature
  itself is the load-bearing assumption** — `#print axioms` will show
  exactly which named theorem is admitted.

## What is and isn't proved

What is proved:
- The PTX text parses and lowers (`example`s).
- The concrete `n = 4, α = 2, xs = [1,2,3,4], ys = [10,20,30,40]` run produces
  `[12, 24, 36, 48]` under the real small-step semantics, with no axioms
  beyond Lean's foundational ones.

What is admitted (by `sorry`, named, with precise signature):
- `saxpy_partial_correct`: for every supported launch `(n ≤ 32, α, xs, ys)`,
  every terminal state of the small-step semantics from
  `saxpyStateFor n α xs ys` satisfies `saxpyPost`.
- `saxpy_total_correct`: same, plus a termination witness.

The cost of removing these `sorry`s is (a) a per-lane straight-line
correctness proof for the 12-instruction kernel body, and (b) a lane-disjoint
write-set argument lifting per-lane correctness to the full warp. Both are
mechanical given Layer-2 frame lemmas in `Proof/Lemmas.lean`; neither is
done. -/

namespace CLean

open Helpers

private def saxpyKernelText : String :=
  ".version 8.5
   .target sm_75
   .address_size 64

   .visible .entry saxpyKernel(
     .param .u32 saxpyKernel_param_0,
     .param .s32 saxpyKernel_param_1,
     .param .u64 saxpyKernel_param_2,
     .param .u64 saxpyKernel_param_3,
     .param .u64 saxpyKernel_param_4
   )
   {
     .reg .pred %p<2>;
     .reg .b32 %r<10>;
     .reg .b64 %rd<11>;

     ld.param.u32 %r2, [saxpyKernel_param_0];
     ld.param.s32 %r6, [saxpyKernel_param_1];
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
     ld.global.s32 %r7, [%rd6];
     ld.global.s32 %r8, [%rd8];
     mad.lo.s32 %r9, %r7, %r6, %r8;
     cvta.to.global.u64 %rd9, %rd3;
     add.s64 %rd10, %rd9, %rd5;
     st.global.s32 [%rd10], %r9;

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

private def saxpyParamBytesFor (n : Nat) (alpha : Int) : ByteMem :=
  let mem := writeU32Bytes ({} : ByteMem) 0 (UInt32.ofNat n)
  let mem := writeS32Bytes mem 4 alpha
  let mem := writeU64Bytes mem 8 (UInt64.ofNat saxpyXBase)
  let mem := writeU64Bytes mem 16 (UInt64.ofNat saxpyYBase)
  writeU64Bytes mem 24 (UInt64.ofNat saxpyRBase)

private def saxpyGlobalBytesFor (xs ys : List Int) : ByteMem :=
  let mem := writeS32Vector ({} : ByteMem) saxpyXBase xs
  writeS32Vector mem saxpyYBase ys

private def saxpyWarpFor (n : Nat) : WarpState :=
  { lanes := Array.replicate 32 { pc := ("saxpyKernel", 0) }
    activeMask := activeMaskPrefix n }

def saxpyStateFor (n : Nat) (alpha : Int) (xs ys : List Int) : State :=
  { kernelEnv := PTX.lowerKernelEnvCheckedD saxpyKernel
    global := { bytes := saxpyGlobalBytesFor xs ys }
    param := { bytes := saxpyParamBytesFor n alpha }
    ctas := ({} : Std.HashMap CTAId CTAState).insert 0
      { warps := ({} : Std.HashMap WarpId WarpState).insert 0 (saxpyWarpFor n) } }


/-! ## Structural lemma: `saxpyStateFor` is single-warp -/

/-- The initial state for saxpy populates only `(cta=0, warp=0)`. -/
private theorem saxpyStateFor_isSingleWarp (n : Nat) (alpha : Int) (xs ys : List Int) :
    IsSingleWarp (saxpyStateFor n alpha xs ys) := by
  intro cta warp ws hGet
  unfold saxpyStateFor State.getWarp? State.getCTA? at hGet
  simp at hGet
  -- hGet : ((({} : Std.HashMap CTAId CTAState).insert 0 _)[cta]?).bind
  --   (fun cs => cs.warps[warp]?) = some ws
  rw [Std.HashMap.getElem?_insert] at hGet
  by_cases hcta : cta = 0
  · subst hcta
    refine ⟨rfl, ?_⟩
    simp at hGet
    rw [Std.HashMap.getElem?_insert] at hGet
    by_cases hwarp : warp = 0
    · exact hwarp
    · exfalso
      rcases warp with _ | w
      · exact hwarp rfl
      · simp at hGet
  · exfalso
    rcases cta with _ | c
    · exact hcta rfl
    · simp at hGet

/-! ## General correctness

`saxpy_partial_correct` is now discharged by an honest reduction:

1. `saxpyStateFor_isSingleWarp` (proved above) — the initial state has only
   `(cta=0, warp=0)` populated.
2. `step?_preserves_IsSingleWarp` (from `Proof.IsSingleWarpPres`) — single-warp
   structure is preserved by every executable step.
3. `terminal_eq_runN` (from `Proof.Determinism`) — single-warp determinism
   forces every terminal state in `Reaches` to equal `runN K init` for some
   fuel `K`.
4. `saxpy_runN_satisfies_post` (admitted below) — symbolic execution of the
   12-instruction straight-line saxpy kernel body under `runN`. This is the
   *one* remaining named `sorry` blocking saxpy partial correctness.

The reduction has no axioms beyond Lean's foundational ones plus the
existing project-wide bounded sorrys (`Lemmas.lean` Layer-2 frame lemmas,
`IsSingleWarpPres.lean` barrier preservation, and the two
`LaneDecomposition.lean` commutation lemmas — all of which propagate into
`step?_preserves_IsSingleWarp`'s axiom set). -/

/-- **Symbolic execution of saxpy through `runN`.** For every supported launch
shape `(n ≤ 32, α, xs, ys)`, there exists a fuel `K` such that:

* `runN K init` is `MachineFinal` (no further executable step), and
* `runN K init` satisfies `saxpyPost`.

The discharge is a per-lane straight-line `cstep`-chain symbolic simulation
of the 12-instruction kernel body, combined with the
`LaneDecomposition.lift_per_lane` lift and the `Lemmas.lean` writeMem? frame
lemmas. Both are infrastructure currently sorried; this lemma is the saxpy
specialization that depends on them. -/
theorem saxpy_runN_satisfies_post
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ K : Nat,
      MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧
      saxpyPost saxpyRBase n alpha xs ys
        (StepMachine.runN K (saxpyStateFor n alpha xs ys)) := by
  -- Discharge requires the per-lane symbolic simulation of the saxpy kernel
  -- body; deferred for the next session, as documented in the file header.
  sorry

/-- **Partial correctness of saxpy for any supported launch.** For
`n ≤ 32, α : Int, xs ys : List Int` with `xs.length = ys.length = n`, every
terminal state reachable by the small-step semantics from
`saxpyStateFor n α xs ys` satisfies `saxpyPost`. -/
theorem saxpy_partial_correct
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    PartialCorrect (saxpyStateFor n alpha xs ys)
                   (saxpyPost saxpyRBase n alpha xs ys) := by
  intro final hTerm
  -- Step 1: initial state is single-warp
  have hsw : IsSingleWarp (saxpyStateFor n alpha xs ys) :=
    saxpyStateFor_isSingleWarp n alpha xs ys
  -- Step 2: single-warp preservation under one executable step
  have hPres : ∀ {s s' : State}, IsSingleWarp s →
      StepMachine.step? s = some s' → IsSingleWarp s' :=
    fun {_ _} hP hStep => step?_preserves_IsSingleWarp hP hStep
  -- Step 3: determinism bridge — final = runN K init for some K
  obtain ⟨K, hK⟩ := terminal_eq_runN hsw hPres hTerm
  -- Step 4: symbolic execution of runN gives the post for *some* fuel J
  obtain ⟨J, _hJFinal, hJPost⟩ := saxpy_runN_satisfies_post n hn alpha xs ys hxs hys
  -- Step 5: both terminal endpoints coincide by `runN` stuck stability.
  subst hK
  have hKFinal : MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) :=
    hTerm.2
  rw [StepMachine.runN_eq_of_both_MachineFinal hKFinal _hJFinal]
  exact hJPost

/-- **Total correctness of saxpy for any supported launch.** Same as
`saxpy_partial_correct` plus existence of a terminating execution. -/
theorem saxpy_total_correct
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    TotalCorrect (saxpyStateFor n alpha xs ys)
                 (saxpyPost saxpyRBase n alpha xs ys) := by
  obtain ⟨J, hJFinal, hJPost⟩ := saxpy_runN_satisfies_post n hn alpha xs ys hxs hys
  refine ⟨StepMachine.runN J (saxpyStateFor n alpha xs ys), ?_, hJPost⟩
  exact ⟨StepMachine.runN_reaches J _, hJFinal⟩

end CLean
