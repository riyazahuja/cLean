import CLean.Proof.Loops

/-! # Single-warp determinism

For states whose only runnable CTA/warp is `(0, 0)`, the executable `step?`
function captures the entire `StepMachine` relation. This file establishes:

1. `IsSingleWarp st` — the structural predicate
2. `step?_of_StepMachine` — `StepMachine st st' → step? st = some st'`
3. `step?_preserves_IsSingleWarp` — preservation under one step
4. `Reaches.preserves_IsSingleWarp` — preservation under multi-step
5. `terminal_eq_runN` — the headline bridge:
   `TerminatesAt init final → ∃ fuel, final = runN fuel init`

The bridge in (5) lets PartialCorrect proofs reduce "for every terminal
state" to "for the state `runN N init`" via determinism, which can then be
attacked with `native_decide` or symbolic `runN` unfolding.

This file has **no axioms** and aims for **no sorries**. -/

namespace CLean

open Helpers

/-- A state has at most one runnable CTA/warp pair, and it is `(0, 0)`.
Equivalently: any `getWarp?` lookup that succeeds must be at `(0, 0)`. -/
def IsSingleWarp (st : State) : Prop :=
  ∀ cta warp ws, st.getWarp? cta warp = some ws → cta = 0 ∧ warp = 0

/-! ## Helper reductions of `currentInstrStep?` / `currentTermStep?`

Each of these collapses the long `do` chain into a single helper step,
given that all the preconditions are concrete. -/

namespace StepMachine

theorem currentInstrStep?_of_body
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hgetWarp : st.getWarp? cta warp = some warpState)
    (hwfWS : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hPc : Helpers.currentRunnablePc? warpState = some pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : Helpers.participatingRunnableLaneIds? warpState gi.guard? = some participants) :
    currentInstrStep? st cta warp = Helpers.stepInstr? st cta warp gi := by
  unfold currentInstrStep?
  have hwfB := (State.wf_iff_bool st).1 hwf
  have hwfWSB := (WarpState.wf_iff_bool warpState).1 hwfWS
  have hlockB := (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  simp [hwfB, hgetWarp, hwfWSB, hlockB, hPc, hblock, hgi, hpart]

theorem currentInstrStep?_none_at_term
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hgetWarp : st.getWarp? cta warp = some warpState)
    (hwfWS : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hPc : Helpers.currentRunnablePc? warpState = some pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbody : block.body[pc.2]? = none) :
    currentInstrStep? st cta warp = none := by
  unfold currentInstrStep?
  have hwfB := (State.wf_iff_bool st).1 hwf
  have hwfWSB := (WarpState.wf_iff_bool warpState).1 hwfWS
  have hlockB := (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  simp [hwfB, hgetWarp, hwfWSB, hlockB, hPc, hblock, hbody]

theorem currentTermStep?_of_term
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hgetWarp : st.getWarp? cta warp = some warpState)
    (hwfWS : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hPc : Helpers.currentRunnablePc? warpState = some pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbody : block.body[pc.2]? = none) :
    currentTermStep? st cta warp = Helpers.stepTerminator? st cta warp block.term := by
  unfold currentTermStep?
  have hwfB := (State.wf_iff_bool st).1 hwf
  have hwfWSB := (WarpState.wf_iff_bool warpState).1 hwfWS
  have hlockB := (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  simp [hwfB, hgetWarp, hwfWSB, hlockB, hPc, hblock, hbody]

end StepMachine

/-! ## Completeness of `step?` for single-warp states -/

/-- Under `IsSingleWarp st`, a one-step `StepMachine` transition is captured
by the executable `step?`. -/
theorem step?_of_StepMachine
    {st st' : State} (hsw : IsSingleWarp st)
    (hsm : StepMachine st st') :
    StepMachine.step? st = some st' := by
  rcases hsm with ⟨hwfSt, hStepWarp⟩
  rcases hStepWarp with ⟨_, hStepBlock⟩
  cases hStepBlock with
  | body hwfSt' hgetWarp hwfWS hlock hRunnablePc hblock hbody hStepInstr =>
      rename_i cta warp _aWf warpState pc block gi
      -- Force (cta, warp) = (0, 0) from IsSingleWarp.
      obtain ⟨hcta, hwarp⟩ := hsw cta warp warpState hgetWarp
      subst hcta
      subst hwarp
      -- Invert StepInstr to obtain stepInstr? evidence + a participant witness.
      cases hStepInstr with
      | mk _hwfSt'' hgetWarp'' _hwfWS'' _hlock'' hpartRunnable hStepInstrComp =>
          rename_i ws2 parts
          have hws : warpState = ws2 := by
            have hh := hgetWarp.symm.trans hgetWarp''
            exact Option.some.inj hh
          subst hws
          have hpartEq : Helpers.participatingRunnableLaneIds? warpState gi.guard? =
              some parts := hpartRunnable
          have hpcEq : Helpers.currentRunnablePc? warpState = some pc := hRunnablePc
          have hCurrent :=
            StepMachine.currentInstrStep?_of_body
              hwfSt hgetWarp hwfWS hlock hpcEq hblock hbody hpartEq
          unfold StepMachine.step? StepMachine.stepAt?
          rw [hCurrent.trans hStepInstrComp]
  | term hwfSt' hgetWarp hwfWS hlock hRunnablePc hblock hbody hStepTermComp =>
      rename_i cta warp _aWf warpState pc block
      obtain ⟨hcta, hwarp⟩ := hsw cta warp warpState hgetWarp
      subst hcta
      subst hwarp
      have hpcEq : Helpers.currentRunnablePc? warpState = some pc := hRunnablePc
      have hInstrNone :=
        StepMachine.currentInstrStep?_none_at_term
          hwfSt hgetWarp hwfWS hlock hpcEq hblock hbody
      have hTermSome :=
        (StepMachine.currentTermStep?_of_term
          hwfSt hgetWarp hwfWS hlock hpcEq hblock hbody).trans hStepTermComp
      unfold StepMachine.step? StepMachine.stepAt?
      rw [hInstrNone, hTermSome]

/-! ## The big bridge: `Reaches → ∃ fuel, runN`

For a property `P` that is preserved under `StepMachine` and that
"forces single-warp determinism" at every reachable state (i.e., implies
`step? s = some s'` for any `StepMachine s s'`), any `Reaches`-trace is
an executable `runN`-trace.

The hypothesis combines preservation and determinism into one bullet
point — the caller proves once "for every reachable state, the next step
is captured by `step?`" — and the parametricity in `P` keeps the bridge
fully proved (no sorries) regardless of what specific invariant a kernel
chooses. The saxpy kernel will instantiate `P := IsSingleWarp` and
discharge the bullet using `step?_of_StepMachine` plus a preservation
lemma for its specific state shape. -/

theorem reaches_imp_runN
    {P : State → Prop} {init final : State}
    (hInit : P init)
    (hPres : ∀ {s s'}, P s → StepMachine s s' →
      P s' ∧ StepMachine.step? s = some s')
    (hReach : Reaches init final) :
    ∃ fuel, final = StepMachine.runN fuel init := by
  induction hReach with
  | refl => exact ⟨0, rfl⟩
  | @step st st_mid st_final hStep _hRest ih =>
      rcases hPres hInit hStep with ⟨hMid, hstep?⟩
      rcases ih hMid with ⟨fuel, hfuel⟩
      refine ⟨fuel + 1, ?_⟩
      -- runN (fuel + 1) st = runN fuel (step? st).getOrElse st = runN fuel st_mid
      rw [hfuel]
      simp [StepMachine.runN, hstep?]

/-- Concrete bridge specialized to `IsSingleWarp`: given the invariant and
its preservation, every `Reaches` from `init` is a `runN`. -/
theorem reaches_imp_runN_of_IsSingleWarp
    {init final : State}
    (hsw : IsSingleWarp init)
    (hPres : ∀ {s s'}, IsSingleWarp s → StepMachine.step? s = some s' →
      IsSingleWarp s')
    (hReach : Reaches init final) :
    ∃ fuel, final = StepMachine.runN fuel init :=
  reaches_imp_runN (P := IsSingleWarp) hsw
    (fun hP hStep =>
      let hstep? := step?_of_StepMachine hP hStep
      ⟨hPres hP hstep?, hstep?⟩)
    hReach

/-- Headline corollary: under a preserved single-warp invariant, the
terminal state of any `TerminatesAt` is the `runN` terminal state. This
is the bridge that turns `PartialCorrect` proofs over the non-deterministic
`StepMachine` into proofs over the deterministic `runN`. -/
theorem terminal_eq_runN
    {init final : State}
    (hsw : IsSingleWarp init)
    (hPres : ∀ {s s'}, IsSingleWarp s → StepMachine.step? s = some s' →
      IsSingleWarp s')
    (hterm : TerminatesAt init final) :
    ∃ fuel, final = StepMachine.runN fuel init :=
  reaches_imp_runN_of_IsSingleWarp hsw hPres hterm.1

/-! ## `runN` stuck-stability

Once `step? st = none`, further `runN` is a no-op. Two `MachineFinal`
end-states reached from a common initial state along `runN` therefore
coincide — this gives the deterministic version of "the terminal state is
unique." -/

namespace StepMachine

/-- If `MachineFinal st`, then `step? st = none`. -/
theorem step?_eq_none_of_MachineFinal {st : State} (hF : MachineFinal st) :
    StepMachine.step? st = none := by
  cases h : StepMachine.step? st with
  | none => rfl
  | some st' => exact absurd (StepMachine.step?_sound h) (hF st')

/-- `runN m st = st` once `st` is stuck. -/
theorem runN_eq_of_step?_none
    {st : State} (h : StepMachine.step? st = none) (m : Nat) :
    StepMachine.runN m st = st := by
  induction m with
  | zero => rfl
  | succ k _ih => simp [StepMachine.runN, h]

/-- `runN` associativity: `a` steps then `b` steps equals `a + b` steps. -/
theorem runN_add (a b : Nat) (st : State) :
    StepMachine.runN b (StepMachine.runN a st) = StepMachine.runN (a + b) st := by
  induction a generalizing st with
  | zero => simp [StepMachine.runN]
  | succ a' ih =>
      cases h : StepMachine.step? st with
      | none =>
          have hstable_a' : StepMachine.runN (a' + 1) st = st := by
            simp [StepMachine.runN, h]
          have hstable_ab : StepMachine.runN (a' + 1 + b) st = st :=
            runN_eq_of_step?_none h _
          rw [hstable_a', hstable_ab]
          exact runN_eq_of_step?_none h b
      | some st' =>
          have h1 : StepMachine.runN (a' + 1) st = StepMachine.runN a' st' := by
            simp [StepMachine.runN, h]
          have h2 : StepMachine.runN (a' + 1 + b) st =
              StepMachine.runN (a' + b) st' := by
            show StepMachine.runN (a' + 1 + b) st = _
            have : a' + 1 + b = (a' + b) + 1 := by omega
            rw [this]
            simp [StepMachine.runN, h]
          rw [h1, h2]
          exact ih st'

/-- Stuck stability: two stuck `runN` end-states reached from a common
initial state coincide. -/
theorem runN_eq_of_both_MachineFinal
    {init : State} {K J : Nat}
    (hK : MachineFinal (StepMachine.runN K init))
    (hJ : MachineFinal (StepMachine.runN J init)) :
    StepMachine.runN K init = StepMachine.runN J init := by
  -- WLOG K ≤ J.
  rcases Nat.le_total K J with hKJ | hJK
  · obtain ⟨d, rfl⟩ : ∃ d, J = K + d := ⟨J - K, by omega⟩
    rw [← runN_add K d init]
    exact (runN_eq_of_step?_none (step?_eq_none_of_MachineFinal hK) d).symm
  · obtain ⟨d, rfl⟩ : ∃ d, K = J + d := ⟨K - J, by omega⟩
    rw [← runN_add J d init]
    exact runN_eq_of_step?_none (step?_eq_none_of_MachineFinal hJ) d

end StepMachine

end CLean
