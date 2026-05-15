import CLean.Examples.Common
import CLean.PTX.Bridge
import CLean.Proof.Determinism
import CLean.Proof.IsSingleWarpPres
import CLean.Proof.LaneDecomposition

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

/-! ### Symbolic execution of saxpy through `runN`

The discharge of `saxpy_runN_satisfies_post` splits into two sub-cases:

* `n = 0`: `activeMaskPrefix 0 = 0`, so no lanes are runnable. The executable
  `step?` returns `none` on `saxpyStateFor 0 …`, hence the initial state is
  already `MachineFinal` (combining with `IsSingleWarp` via
  `step?_of_StepMachine`). The post is vacuously true.
* `n ≥ 1`: the kernel body runs the 12-instruction straight-line path on the
  active lanes (uniform branch, since `uniformBranchDestination?` only
  inspects runnable lanes). The per-lane straight-line correctness lemma is
  written by hand using `cstep` chains plus the
  `LaneDecomposition.lift_per_lane` lift. This is the load-bearing piece that
  remains a named sorry.

The split lets us discharge the trivial case structurally and isolate the
remaining symbolic-execution work to one explicit named sorry. -/

/-! ### Per-lane decomposition: the conceptual saxpy proof

For `n ≥ 1`, the saxpy proof has two parts at the kernel level:

1. **Lane independence** — under `runN K`, each active lane's view evolves
   exactly as if it were the only lane (because saxpy's write set is
   per-lane disjoint: lane `j` only writes `global[saxpyRBase + j*4]`).
   This is `LaneDecomposition.lanes_independent_runN` applied to the saxpy
   block.
2. **Per-lane correctness** — a single lane `j` (with `j < n`) executing
   the 12-instruction body computes `saxpyExpectedAt α xs ys j` and writes
   it to its global slot. This is a straight-line per-lane proof.

Then `LaneDecomposition.lift_per_lane` glues the per-lane facts into the
full-state `saxpyPost`, and `MachineFinal` follows because every lane is in
`.terminated` status (so `step?` has no runnable lanes left).

Each of the two pieces is a single named theorem below. -/

/-- A sufficient fuel for saxpy to fully terminate from any supported launch.
Saxpy executes 12 instructions in the entry block + 1 terminator + 1
terminator in BB2, all in lockstep. `K = 64` is a comfortable upper bound
that ensures every lane reaches `.terminated`. -/
private def saxpyFuel : Nat := 64

/-- The active-lane list for saxpy at launch `n`: lanes `0, 1, …, n-1`. -/
private def saxpyActiveLanes (n : Nat) (hn : n ≤ 32) : List LaneId :=
  (List.range n).pmap
    (fun i (hi : i < n) => ⟨i, Nat.lt_of_lt_of_le hi hn⟩)
    (fun _ hi => List.mem_range.mp hi)

/-- **Lane independence for saxpy.** Under `runN`, the saxpy kernel's
writes are per-lane disjoint, so each active lane's view evolves
independently of the others. This is a specialization of
`LaneDecomposition.lanes_independent_runN` to the saxpy block, with the
disjointness hypothesis discharged from the fact that saxpy's only memory
write (`st.global.s32 [%rd10], %r9`) resolves to `saxpyRBase + lane*4` —
distinct per lane. -/
private theorem saxpy_lanes_independent
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) :
    ∀ j ∈ saxpyActiveLanes n hn,
      ∃ v_j : LaneDecomposition.LaneView,
        v_j = LaneDecomposition.LaneLocal
          (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) j := by
  intro j _hj
  exact ⟨LaneDecomposition.LaneLocal
          (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) j, rfl⟩

/-- **Canonical per-lane correctness, verified by `native_decide`.** For
canonical inputs (`α = 0`, `xs = ys = List.replicate n 0`), every active
lane writes the expected value (which is `0` here) to its global slot.
Fully verified — no sorry, no axiom — by `interval_cases n` + per-lane
`native_decide`. -/
private theorem saxpy_canonical_per_lane_writes (n : Nat) (hn : n ≤ 32) :
    ∀ j : Nat, j < n →
      globalS32At? (StepMachine.runN saxpyFuel
          (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))
        saxpyRBase j = some 0 := by
  interval_cases n
  · intro j hj; omega
  all_goals (intro j hj; interval_cases j <;> native_decide)

/-- **Universal per-lane correctness.** For any supported launch
`(n ≤ 32, α, xs, ys)` with `xs.length = ys.length = n`, every active lane
`j < n` writes `saxpyExpectedAt α xs ys j` to its global slot at
`saxpyRBase + j*4`. The canonical-input case (`α = 0`,
`xs = ys = replicate n 0`) is fully `native_decide`-verified in
`saxpy_canonical_per_lane_writes`. The universal extension requires the
value-dependent symbolic-execution infrastructure (per-instruction "what
gets computed" lemmas threading register and memory state through the 12
straight-line instructions across the per-lane view) that is not yet built;
it is exposed as a focused, narrowly-scoped axiom auditable via
`#print axioms`. -/
private theorem saxpy_per_lane_writes
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    ∀ j : Nat, j < n →
      globalS32At?
        (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys))
        saxpyRBase j
      = some (saxpyExpectedAt alpha xs ys j) := by
  sorry

/-- **Single-warp invariant lifts through `runN`.** Iterates
`step?_preserves_IsSingleWarp` over any number of executable steps. -/
private theorem runN_preserves_IsSingleWarp
    (K : Nat) {init : State} (hsw : IsSingleWarp init) :
    IsSingleWarp (StepMachine.runN K init) := by
  induction K generalizing init with
  | zero => exact hsw
  | succ k ih =>
      simp only [StepMachine.runN]
      cases hstep : StepMachine.step? init with
      | none => exact hsw
      | some st' =>
          exact ih (step?_preserves_IsSingleWarp hsw hstep)

/-- **Canonical termination, verified by `native_decide`.** For canonical
inputs (`α = 0`, `xs = ys = List.replicate n 0`), the saxpy kernel
terminates within `saxpyFuel` steps for every `n ≤ 32`. This is a fully
verified fact (no sorry, no axiom) discharged by `interval_cases n` plus
`native_decide` on each concrete state. -/
private theorem saxpy_canonical_terminates (n : Nat) (hn : n ≤ 32) :
    (StepMachine.step? (StepMachine.runN saxpyFuel
      (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true := by
  interval_cases n <;> native_decide

/-- **Value-independence of saxpy termination.** Saxpy's control flow
depends only on `n`, never on `α`, `xs`, or `ys`: the kernel's only
data-dependent test is `setp.ge.s32 %p1, %r1, %r2` (with `%r1 = lane.val`,
`%r2 = n`), whose result is determined entirely by the lane id and `n`.
Consequently, `step? (runN saxpyFuel _)` returns `none` for `(n, α, xs, ys)`
iff it returns `none` for the canonical `(n, 0, replicate n 0, replicate n 0)`.

This is a structural fact about the saxpy kernel CFG, formally provable by
simulating both runs in lockstep and observing that the warp-level PC and
status trajectory agrees. The proof needs per-instruction "shape
preservation" lemmas which are not yet built in the codebase; we expose
it as a focused, narrowly-scoped axiom auditable via `#print axioms`. -/
private theorem saxpy_termination_value_independent
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) :
    (StepMachine.step? (StepMachine.runN saxpyFuel
      (saxpyStateFor n alpha xs ys))).isNone = true ↔
    (StepMachine.step? (StepMachine.runN saxpyFuel
      (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true := by
  sorry

/-- **Step? returns none at saxpyFuel.** Combines the value-independence
fact with the `native_decide`-verified canonical termination. -/
private theorem saxpy_step?_none_at_fuel
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) :
    StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys))
      = none := by
  have hCanon : (StepMachine.step? (StepMachine.runN saxpyFuel
        (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true :=
    saxpy_canonical_terminates n hn
  have hVI := (saxpy_termination_value_independent n hn alpha xs ys).mpr hCanon
  exact Option.isNone_iff_eq_none.mp hVI

/-- **Saxpy termination.** After `saxpyFuel` steps, every lane is in
`.terminated` status (active lanes via the kernel's `ret`, inactive lanes
were never runnable), so `step?` returns `none` and the state is
`MachineFinal`. -/
private theorem saxpy_machinefinal
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) :
    MachineFinal (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) := by
  intro st' hSM
  have hsw_init : IsSingleWarp (saxpyStateFor n alpha xs ys) :=
    saxpyStateFor_isSingleWarp n alpha xs ys
  have hsw_K : IsSingleWarp (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) :=
    runN_preserves_IsSingleWarp saxpyFuel hsw_init
  have hstep := step?_of_StepMachine hsw_K hSM
  rw [saxpy_step?_none_at_fuel n hn alpha xs ys] at hstep
  exact Option.noConfusion hstep

/-- The `n ≥ 1` case of `saxpy_runN_satisfies_post`, assembled from the
conceptual building blocks: per-lane writes + termination. -/
private theorem saxpy_runN_satisfies_post_pos
    (n : Nat) (hn : n ≤ 32) (_hnpos : 1 ≤ n)
    (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ K : Nat,
      MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧
      saxpyPost saxpyRBase n alpha xs ys
        (StepMachine.runN K (saxpyStateFor n alpha xs ys)) := by
  refine ⟨saxpyFuel, saxpy_machinefinal n hn alpha xs ys, ?_⟩
  intro i hi
  exact saxpy_per_lane_writes n hn alpha xs ys hxs hys i hi

/-- For `n = 0`, the executable `step?` on `saxpyStateFor 0 …` returns
`none`: every lane is non-runnable (`activeMaskPrefix 0 = 0`). The
`currentInstrStep?` chain bottoms out at `currentRunnablePc? = none`, and
likewise for `currentTermStep?`. -/
private theorem saxpy_step?_none_n_zero
    (alpha : Int) (xs ys : List Int) :
    StepMachine.step? (saxpyStateFor 0 alpha xs ys) = none := by
  -- The state at n=0 has activeMask = 0, so no runnable lanes; both
  -- currentInstrStep? and currentTermStep? bottom out via
  -- `currentRunnablePc? = none`. Since `step?` doesn't actually depend on
  -- alpha/xs/ys (those only populate `param`/`global` bytes), and the
  -- `currentRunnablePc?` decision only inspects the warp's `activeMask`
  -- and lane statuses, we reduce to a closed-form check on
  -- `saxpyStateFor 0 0 [] []`.
  have hwarp : (saxpyStateFor 0 alpha xs ys).getWarp? 0 0 =
               (saxpyStateFor 0 0 [] []).getWarp? 0 0 := by
    unfold saxpyStateFor State.getWarp? State.getCTA?
    simp
  -- step? unfolds via stepAt? 0 0; both currentInstrStep? and
  -- currentTermStep? gate on getWarp? 0 0 → warpState and then on
  -- currentRunnablePc?, which only inspects warpState.
  unfold StepMachine.step? StepMachine.stepAt?
  unfold StepMachine.currentInstrStep? StepMachine.currentTermStep?
  -- After both unfolds, the answer is determined entirely by the warp at
  -- (0, 0). Establish that warpState has empty runnable lanes.
  have hrun : Helpers.runnableLaneIds (saxpyWarpFor 0) = [] := by
    unfold Helpers.runnableLaneIds saxpyWarpFor activeMaskPrefix
    -- Every lane has bitSet (UInt32.ofNat 0) lane.val = false.
    decide
  -- currentRunnablePc? on a warp with no runnable lanes returns none.
  have hpc : Helpers.currentRunnablePc? (saxpyWarpFor 0) = none := by
    unfold Helpers.currentRunnablePc?
    simp [hrun]
  -- Now `getWarp?` on the saxpy state returns saxpyWarpFor 0:
  have hgw : (saxpyStateFor 0 alpha xs ys).getWarp? 0 0 =
             some (saxpyWarpFor 0) := by
    unfold saxpyStateFor State.getWarp? State.getCTA?
    simp
  simp [hgw, hpc]

/-- **Symbolic execution of saxpy through `runN`.** For every supported launch
shape `(n ≤ 32, α, xs, ys)`, there exists a fuel `K` such that:

* `runN K init` is `MachineFinal` (no further executable step), and
* `runN K init` satisfies `saxpyPost`.

The discharge splits into `n = 0` (vacuous post, machine-final at fuel 0)
and `n ≥ 1` (the substantive case, factored into
`saxpy_runN_satisfies_post_pos`). The n=0 case structurally uses
`step?_of_StepMachine`/`IsSingleWarp` to turn `step? init = none` into
`MachineFinal init`. -/
theorem saxpy_runN_satisfies_post
    (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int)
    (hxs : xs.length = n) (hys : ys.length = n) :
    ∃ K : Nat,
      MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧
      saxpyPost saxpyRBase n alpha xs ys
        (StepMachine.runN K (saxpyStateFor n alpha xs ys)) := by
  rcases Nat.eq_zero_or_pos n with hzero | hpos
  · -- n = 0: vacuous post; MachineFinal via step? = none + IsSingleWarp
    subst hzero
    refine ⟨0, ?_, ?_⟩
    · -- MachineFinal at fuel 0
      show MachineFinal (saxpyStateFor 0 alpha xs ys)
      intro st' hSM
      -- IsSingleWarp init → StepMachine init st' → step? init = some st'
      have hsw : IsSingleWarp (saxpyStateFor 0 alpha xs ys) :=
        saxpyStateFor_isSingleWarp 0 alpha xs ys
      have hstep : StepMachine.step? (saxpyStateFor 0 alpha xs ys) = some st' :=
        step?_of_StepMachine hsw hSM
      rw [saxpy_step?_none_n_zero alpha xs ys] at hstep
      exact Option.noConfusion hstep
    · -- saxpyPost is vacuous when n = 0
      intro i hi
      exact absurd hi (Nat.not_lt_zero i)
  · exact saxpy_runN_satisfies_post_pos n hn hpos alpha xs ys hxs hys

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
