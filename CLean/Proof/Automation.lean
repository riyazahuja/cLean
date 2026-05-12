import CLean.Semantics.Execution

namespace CLean

/-- A named boolean postcondition wrapper for executable examples. -/
def HoldsPost (post : State → Bool) (st : State) : Prop :=
  post st = true

theorem runN_reaches_and_post (fuel : Nat) (st : State) (post : State → Bool)
    (hpost : post (StepMachine.runN fuel st) = true) :
    Reaches st (StepMachine.runN fuel st) ∧ HoldsPost post (StepMachine.runN fuel st) := by
  exact ⟨StepMachine.runN_reaches fuel st, hpost⟩

theorem runN_reaches_and_post₂ (fuel : Nat) (st : State) (post₁ post₂ : State → Bool)
    (hpost₁ : post₁ (StepMachine.runN fuel st) = true)
    (hpost₂ : post₂ (StepMachine.runN fuel st) = true) :
    Reaches st (StepMachine.runN fuel st) ∧
      HoldsPost post₁ (StepMachine.runN fuel st) ∧
      HoldsPost post₂ (StepMachine.runN fuel st) := by
  exact ⟨StepMachine.runN_reaches fuel st, hpost₁, hpost₂⟩

theorem runN_reaches_and_post₃ (fuel : Nat) (st : State) (post₁ post₂ post₃ : State → Bool)
    (hpost₁ : post₁ (StepMachine.runN fuel st) = true)
    (hpost₂ : post₂ (StepMachine.runN fuel st) = true)
    (hpost₃ : post₃ (StepMachine.runN fuel st) = true) :
    Reaches st (StepMachine.runN fuel st) ∧
      HoldsPost post₁ (StepMachine.runN fuel st) ∧
      HoldsPost post₂ (StepMachine.runN fuel st) ∧
      HoldsPost post₃ (StepMachine.runN fuel st) := by
  exact ⟨StepMachine.runN_reaches fuel st, hpost₁, hpost₂, hpost₃⟩

syntax "crun" : tactic
syntax "cpost" : tactic
syntax "cfinish" : tactic

macro_rules
  | `(tactic| crun) => `(tactic| exact StepMachine.runN_reaches _ _)
  | `(tactic| cpost) => `(tactic| native_decide)
  | `(tactic| cfinish) =>
      `(tactic|
        first
        | cstep
        | crun
        | cpost)

end CLean
