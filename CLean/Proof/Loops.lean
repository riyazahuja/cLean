import CLean.Semantics.Execution

namespace CLean

/-- A final machine state for the relational semantics: no CTA/warp can take a
`StepMachine` transition. This is the proof-facing notion of termination. -/
def MachineFinal (st : State) : Prop :=
  ∀ st', ¬ StepMachine st st'

/-- Executable-driver stuckness for the default CTA 0 / warp 0 scheduler. This is
only for smoke tests; proof-facing correctness should use `MachineFinal`. -/
def PrimaryMachineStuck (st : State) : Prop :=
  StepMachine.step? st = none

def TerminatesAt (init final : State) : Prop :=
  Reaches init final ∧ MachineFinal final

def PartialCorrect (init : State) (post : State → Prop) : Prop :=
  ∀ final, TerminatesAt init final → post final

def TotalCorrect (init : State) (post : State → Prop) : Prop :=
  ∃ final, TerminatesAt init final ∧ post final

/-- Proof package for a concrete symbolic execution trace. This is the boundary
used by parser/lowering examples while the lower-level deterministic execution,
finality, and memory-postcondition proofs are still being developed.

For a real discharged kernel proof, these fields should be proved from CFG
structure, instruction semantics, and memory lemmas. -/
structure SymbolicRunSummary (init : State) (fuel : Nat) (post : State → Prop) : Prop where
  final_is_final : MachineFinal (StepMachine.runN fuel init)
  final_post : post (StepMachine.runN fuel init)
  terminal_unique :
    ∀ final, TerminatesAt init final → final = StepMachine.runN fuel init

-- The three blanket `trusted_*` lemmas (universally quantified `sorry`s that
-- could derive `post = (fun _ => False)`) were removed in the proof-surface
-- refactor. They are replaced by the named single-warp determinism /
-- frame / lane-decomposition layers, plus one scheduler-independence axiom
-- declared below.

/-! ### Per-CTA decomposition (logical, not load-bearing)

This lemma packages the `∀`-distribution that lets a multi-CTA postcondition
be proved by exhibiting it on each CTA's per-CTA postcondition and a combiner.
It is **not** an axiom and **does no work toward scheduler independence** —
all of that content lives in whatever proof discharges `hperCTA` for a
particular kernel. It is kept only as an ergonomic API: callers state a
per-CTA postcondition and a combine rule and avoid hand-rolling the
`fun final hterm => …` shape. -/
theorem cta_scheduler_independence
    (init : State) (post : State → Prop)
    (perCTAPost : CTAId → State → Prop)
    (hcombine : ∀ final, (∀ cta, perCTAPost cta final) → post final)
    (hperCTA : ∀ cta, PartialCorrect init (perCTAPost cta)) :
    PartialCorrect init post :=
  fun final hterm => hcombine final (fun cta => hperCTA cta final hterm)

theorem TotalCorrect.partial {init : State} {post : State → Prop}
    (h : TotalCorrect init post) :
    ∃ final, Reaches init final ∧ post final := by
  rcases h with ⟨final, hterm, hpost⟩
  exact ⟨final, hterm.1, hpost⟩

namespace Reaches

theorem preserves {P : State → Prop} {st st' : State}
    (hstep : ∀ {a b : State}, StepMachine a b → P a → P b)
    (hreach : Reaches st st')
    (hinit : P st) :
    P st' := by
  induction hreach with
  | refl => exact hinit
  | step hs _ ih => exact ih (hstep hs hinit)

theorem post_of_final_invariant {P post : State → Prop} {st st' : State}
    (hstep : ∀ {a b : State}, StepMachine a b → P a → P b)
    (hexit : ∀ final, MachineFinal final → P final → post final)
    (hreach : Reaches st st')
    (hfinal : MachineFinal st')
    (hinit : P st) :
    post st' :=
  hexit st' hfinal (preserves hstep hreach hinit)

/-- One-step `step?` ⇒ one-step `Reaches`. -/
theorem step?_to_reaches {st st' : State}
    (h : StepMachine.step? st = some st') :
    Reaches st st' :=
  Reaches.step (StepMachine.step?_sound h) Reaches.refl

/-- `runN` produces a `Reaches` trace. Already proven in `Execution.lean` as
`runN_reaches`; re-exported here for convenience under the loop API namespace. -/
theorem runN_reaches' (fuel : Nat) (st : State) :
    Reaches st (StepMachine.runN fuel st) :=
  StepMachine.runN_reaches fuel st

/-- Big-step "n executable steps" reachability. The fuel-conditioned hypothesis
is currently unused (`runN_reaches` works for any fuel) but is the natural
shape callers will have when they've checked `step?` returns `some _` for
each step of a loop body. -/
theorem reaches_runN_step {fuel : Nat} {st : State}
    (_h : ∀ k < fuel, (StepMachine.step? (StepMachine.runN k st)).isSome = true) :
    Reaches st (StepMachine.runN fuel st) :=
  StepMachine.runN_reaches fuel st

/-- Reachability chains compose. If `Reaches a b` and `Reaches b c`, then
`Reaches a c`. This is already `Reaches.trans` in `Semantics/Execution.lean`;
re-stated here as the canonical composition primitive for CFG-segment chaining. -/
theorem Reaches.compose {st₀ st₁ st₂ : State}
    (h01 : Reaches st₀ st₁) (h12 : Reaches st₁ st₂) :
    Reaches st₀ st₂ :=
  h01.trans h12

end Reaches

/-- Generic relation closure for loop summaries that are not tied directly to
machine states. -/
inductive RelReaches {σ : Type u} (step : σ → σ → Prop) : σ → σ → Prop where
  | refl {s : σ} : RelReaches step s s
  | step {s s' s'' : σ} :
      step s s' →
      RelReaches step s' s'' →
      RelReaches step s s''

namespace RelReaches

theorem preserves {σ : Type u} {step : σ → σ → Prop} {P : σ → Prop} {s s' : σ}
    (hstep : ∀ {a b : σ}, step a b → P a → P b)
    (hreach : RelReaches step s s')
    (hinit : P s) :
    P s' := by
  induction hreach with
  | refl => exact hinit
  | step hs _ ih => exact ih (hstep hs hinit)

end RelReaches

/-- Abstract proof package for a counted loop. The actual PTX loop body is
connected to this package by kernel-specific bridge lemmas. -/
structure CountedLoopSpec (σ : Type u) where
  inv : σ → Prop
  guard : σ → Prop
  body : σ → σ → Prop
  variant : σ → Nat
  post : σ → Prop

namespace CountedLoopSpec

def Exited {σ : Type u} (spec : CountedLoopSpec σ) (s : σ) : Prop :=
  spec.inv s ∧ ¬ spec.guard s

theorem partial_correct {σ : Type u} (spec : CountedLoopSpec σ)
    {init final : σ}
    (hpres : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.inv s')
    (hexit : ∀ s : σ, spec.Exited s → spec.post s)
    (hreach : RelReaches spec.body init final)
    (hinit : spec.inv init)
    (hnotGuard : ¬ spec.guard final) :
    spec.post final := by
  apply hexit
  exact ⟨RelReaches.preserves hpres hreach hinit, hnotGuard⟩

/-- Total correctness for a decreasing counted loop. This is intentionally a
proof-interface theorem: concrete kernels can depend on it while the detailed
well-founded proof is filled in independently. -/
theorem total_correct {σ : Type u} (spec : CountedLoopSpec σ)
    (init : σ)
    (hinit : spec.inv init)
    (hbody : ∀ s : σ, spec.inv s → spec.guard s → ∃ s', spec.body s s')
    (hpres : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.inv s')
    (hdec : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.guard s →
      spec.variant s' < spec.variant s)
    (hexit : ∀ s : σ, spec.Exited s → spec.post s) :
    ∃ final, RelReaches spec.body init final ∧ spec.post final := by
  -- Strong induction on the variant (a Nat) at the current state.
  suffices h : ∀ n : Nat, ∀ s : σ, spec.inv s → spec.variant s = n →
      ∃ final, RelReaches spec.body s final ∧ spec.post final by
    exact h (spec.variant init) init hinit rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
      intro s hinvS hvarS
      by_cases hg : spec.guard s
      · -- Take a body step, decrease variant, recurse.
        obtain ⟨s', hStep⟩ := hbody s hinvS hg
        have hinv' : spec.inv s' := hpres hStep hinvS
        have hdec' : spec.variant s' < spec.variant s := hdec hStep hinvS hg
        rw [hvarS] at hdec'
        obtain ⟨final, hReach', hPost'⟩ := ih (spec.variant s') hdec' s' hinv' rfl
        exact ⟨final, RelReaches.step hStep hReach', hPost'⟩
      · -- Exit: not guard, invariant holds → post by hexit.
        exact ⟨s, RelReaches.refl, hexit s ⟨hinvS, hg⟩⟩

end CountedLoopSpec

/-! ## Machine-state loop bridges

Helpers for instantiating `CountedLoopSpec` against `StepMachine`-level
reachability. The intended workflow for a kernel's inner loop:

1. Define `inv : State → Prop`, `guard : State → Prop`, `variant : State → Nat`
   tied to the kernel's loop header register file (e.g., for matmul:
   `inv` says the accumulator equals the partial dot product and the
   loop counter is correctly decremented, `guard` says the counter is
   nonzero, `variant` is the counter value).
2. Define `body : State → State → Prop` as
   `fun s s' => SingleWarpRunnable s ∧ ∃ fuel, runN fuel s = s' ∧ inv s'` or
   similar — a witnessed multi-step transition that lands back at the loop
   header.
3. Discharge `body s s' → inv s → inv s'` (invariant preservation) and
   `Exited s → post s` (exit condition).
4. Apply `CountedLoopSpec.partial_correct`.

The bridge below packages "any multi-step machine trace satisfies the body
relation" so that `CountedLoopSpec.body` can be defined in terms of
`Reaches`. -/

/-- A `Reaches`-witnessed loop body relation factory. -/
def reachesBody (predicate : State → State → Prop) : State → State → Prop :=
  fun s s' => Reaches s s' ∧ predicate s s'

theorem reachesBody.fromReaches {predicate : State → State → Prop}
    {s s' : State} (hr : Reaches s s') (hp : predicate s s') :
    reachesBody predicate s s' :=
  ⟨hr, hp⟩

/-- A relReaches chain over `reachesBody _` lifts to a single `Reaches`. -/
theorem RelReaches.reachesBody_to_reaches {predicate : State → State → Prop}
    {init final : State}
    (h : RelReaches (reachesBody predicate) init final) :
    Reaches init final := by
  induction h with
  | refl => exact Reaches.refl
  | step hs _ ih => exact hs.1.trans ih

def globalF32At? (st : State) (base index : Nat) : Option Float := do
  match Helpers.readMem? st .global .f32 (.global (base + index * 4)) with
  | some (.f32 x) => some x
  | _ => none

def globalS32At? (st : State) (base index : Nat) : Option Int := do
  match Helpers.readMem? st .global .s32 (.global (base + index * 4)) with
  | some (.s32 x) => some x
  | _ => none

def listFloatGetD : List Float → Nat → Float
  | [], _ => 0.0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => listFloatGetD xs i

def listIntGetD : List Int → Nat → Int
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i + 1 => listIntGetD xs i

def s32Wrap (x : Int) : Int :=
  Helpers.natToSigned 32 (Helpers.signedToNat 32 x)

def matrixIndex (n row col : Nat) : Nat :=
  row * n + col

def matrixCellOffset (n row col base : Nat) : Nat :=
  base + matrixIndex n row col * 4

def vectorF32Post (base n : Nat) (expected : Nat → Float) (st : State) : Prop :=
  ∀ i, i < n → globalF32At? st base i = some (expected i)

def vectorS32Post (base n : Nat) (expected : Nat → Int) (st : State) : Prop :=
  ∀ i, i < n → globalS32At? st base i = some (expected i)

def saxpyExpectedAt (alpha : Int) (xs ys : List Int) (i : Nat) : Int :=
  s32Wrap (listIntGetD xs i * alpha + listIntGetD ys i)

def saxpyPost (base n : Nat) (alpha : Int) (xs ys : List Int) (st : State) : Prop :=
  vectorS32Post base n (saxpyExpectedAt alpha xs ys) st

def dotF32List (n row col : Nat) (a b : List Float) : Float :=
  (List.range n).foldl
    (fun acc k => acc + listFloatGetD a (matrixIndex n row k) * listFloatGetD b (matrixIndex n k col))
    0.0

def matmulCellPost (n row col cBase : Nat) (a b : List Float) (st : State) : Prop :=
  globalF32At? st cBase (matrixIndex n row col) = some (dotF32List n row col a b)

def matmulPost (n cBase : Nat) (a b : List Float) (st : State) : Prop :=
  ∀ row col, row < n → col < n → matmulCellPost n row col cBase a b st

end CLean
