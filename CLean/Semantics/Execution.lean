import CLean.Semantics.StepSoundness

namespace CLean

inductive Reaches : State → State → Prop where
  | refl {st : State} : Reaches st st
  | step {st st' st'' : State} :
      StepMachine st st' →
      Reaches st' st'' →
      Reaches st st''

namespace Reaches

theorem trans {st₀ st₁ st₂ : State} :
    Reaches st₀ st₁ → Reaches st₁ st₂ → Reaches st₀ st₂ := by
  intro h01 h12
  induction h01 with
  | refl => exact h12
  | step hstep _ ih => exact Reaches.step hstep (ih h12)

end Reaches

namespace StepMachine

def stepAt? (st : State) (cta : CTAId) (warp : WarpId) : Option State :=
  match currentInstrStep? st cta warp with
  | some st' => some st'
  | none => currentTermStep? st cta warp

def step? (st : State) : Option State :=
  stepAt? st 0 0

def runN : Nat → State → State
  | 0, st => st
  | fuel + 1, st =>
      match step? st with
      | some st' => runN fuel st'
      | none => st

def runN? : Nat → State → Option State
  | 0, st => some st
  | fuel + 1, st => do
      let st' <- step? st
      runN? fuel st'

def traceN : Nat → State → List State
  | 0, st => [st]
  | fuel + 1, st =>
      match step? st with
      | some st' => st :: traceN fuel st'
      | none => [st]

theorem stepAt?_sound
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hstep : stepAt? st cta warp = some st') :
    StepMachine st st' := by
  unfold stepAt? at hstep
  cases hinstr : currentInstrStep? st cta warp with
  | some stInstr =>
      simp [hinstr] at hstep
      subst st'
      simpa [hinstr] using
        body_of_currentInstrStep?_computed (st := st) (cta := cta) (warp := warp)
          (by simp [hinstr])
  | none =>
      simp [hinstr] at hstep
      cases hterm : currentTermStep? st cta warp with
      | none =>
          simp [hterm] at hstep
      | some stTerm =>
          simp [hterm] at hstep
          subst st'
          simpa [hterm] using
            term_of_currentTermStep?_computed (st := st) (cta := cta) (warp := warp)
              (by simp [hterm])

theorem step?_sound {st st' : State}
    (hstep : step? st = some st') :
    StepMachine st st' :=
  stepAt?_sound (cta := 0) (warp := 0) hstep

theorem runN_reaches (fuel : Nat) (st : State) :
    Reaches st (runN fuel st) := by
  induction fuel generalizing st with
  | zero =>
      exact Reaches.refl
  | succ fuel ih =>
      simp [runN]
      cases hstep : step? st with
      | none =>
          exact Reaches.refl
      | some st' =>
          exact Reaches.step (step?_sound hstep) (ih st')

theorem runN?_sound {fuel : Nat} {st st' : State}
    (hrun : runN? fuel st = some st') :
    Reaches st st' := by
  induction fuel generalizing st with
  | zero =>
      simp [runN?] at hrun
      subst st'
      exact Reaches.refl
  | succ fuel ih =>
      simp [runN?] at hrun
      cases hstep : step? st with
      | none =>
          simp [hstep] at hrun
      | some stNext =>
          simp [hstep] at hrun
          exact Reaches.step (step?_sound hstep) (ih hrun)

end StepMachine

end CLean
