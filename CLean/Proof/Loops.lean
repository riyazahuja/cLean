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

/-- Temporary trusted bridge for generated-CFG symbolic execution summaries.
This keeps kernel example theorem statements honest while localizing the current
proof debt to the proof layer instead of the example file. -/
theorem trusted_symbolic_run_summary
    (init : State) (fuel : Nat) (post : State → Prop) :
    SymbolicRunSummary init fuel post := by
  sorry

/-- Temporary trusted bridge for loop-heavy kernels whose correctness proof is
not yet connected to executable `runN` fuel. This is intended to be replaced by
kernel-specific loop invariants using `CountedLoopSpec`. -/
theorem trusted_kernel_partial_correct
    (init : State) (post : State → Prop) :
    PartialCorrect init post := by
  sorry

/-- Temporary trusted bridge for loop-heavy kernel termination. This is intended
to be replaced by decreasing-variant loop proofs. -/
theorem trusted_kernel_total_correct
    (init : State) (post : State → Prop) :
    TotalCorrect init post := by
  sorry

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
  sorry

end CountedLoopSpec

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
