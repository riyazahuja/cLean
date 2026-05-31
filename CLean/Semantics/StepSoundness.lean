import CLean.Semantics.SmallStep

namespace CLean

open Helpers

namespace StepMachine

theorem body_of_stepInstr?
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : Helpers.ParticipatingRunnable warpState gi.guard? participants)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    StepMachine st st' :=
  StepMachine.mk hwf <|
    StepWarp.mk hwf <|
      StepBlock.body hwf hwarp hwfWarp hlock hrpc hblock hgi <|
        StepInstr.mk hwf hwarp hwfWarp hlock hpart hstep

theorem term_of_stepTerminator?
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbodyDone : block.body[pc.2]? = none)
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st') :
    StepMachine st st' :=
  StepMachine.mk hwf <|
    StepWarp.mk hwf <|
      StepBlock.term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hstep

theorem body_of_stepInstr?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : Helpers.ParticipatingRunnable warpState gi.guard? participants)
    (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) :
    StepMachine st
      (match Helpers.stepInstr? st cta warp gi with
       | some st' => st'
       | none => st) := by
  cases hstep : Helpers.stepInstr? st cta warp gi with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      simpa [hstep] using
        body_of_stepInstr?
          (st := st) (st' := st') (cta := cta) (warp := warp)
          (warpState := warpState) (pc := pc) (block := block) (gi := gi)
          (participants := participants)
          hwf hwarp hwfWarp hlock hrpc hblock hgi hpart hstep

theorem term_of_stepTerminator?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hwfWarp : WarpState.wf warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbodyDone : block.body[pc.2]? = none)
    (hisSome : (Helpers.stepTerminator? st cta warp block.term).isSome = true) :
    StepMachine st
      (match Helpers.stepTerminator? st cta warp block.term with
       | some st' => st'
       | none => st) := by
  cases hstep : Helpers.stepTerminator? st cta warp block.term with
  | none =>
      simp [hstep] at hisSome
  | some st' =>
      simpa [hstep] using
        term_of_stepTerminator?
          (st := st) (st' := st') (cta := cta) (warp := warp)
          (warpState := warpState) (pc := pc) (block := block)
          hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hstep

def currentInstrStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  if !st.wf? then
    none
  else
    let warpState <- st.getWarp? cta warp
    if !warpState.wf? then
      none
    else if !Helpers.lockstepRunnable? warpState then
      none
    else
      let pc <- Helpers.currentRunnablePc? warpState
      let block <- st.kernelEnv.blocks[pc.1]?
      let gi <- block.body[pc.2]?
      let _participants <- Helpers.participatingRunnableLaneIds? warpState gi.guard?
      Helpers.stepInstr? st cta warp gi

def currentTermStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do
  if !st.wf? then
    none
  else
    let warpState <- st.getWarp? cta warp
    if !warpState.wf? then
      none
    else if !Helpers.lockstepRunnable? warpState then
      none
    else
      let pc <- Helpers.currentRunnablePc? warpState
      let block <- st.kernelEnv.blocks[pc.1]?
      match block.body[pc.2]? with
      | some _ => none
      | none => Helpers.stepTerminator? st cta warp block.term

theorem body_of_currentInstrStep?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    (hisSome : (currentInstrStep? st cta warp).isSome = true) :
    StepMachine st
      (match currentInstrStep? st cta warp with
       | some st' => st'
       | none => st) := by
  unfold currentInstrStep? at hisSome ⊢
  cases hwf : st.wf? <;> simp [hwf] at hisSome ⊢
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hisSome
  | some warpState =>
      simp [hwarp] at hisSome ⊢
      cases hwfWarp : warpState.wf? <;> simp [hwfWarp] at hisSome ⊢
      cases hlock : Helpers.lockstepRunnable? warpState <;> simp [hlock] at hisSome ⊢
      cases hpc : Helpers.currentRunnablePc? warpState with
      | none =>
          simp [hpc] at hisSome
      | some pc =>
          simp [hpc] at hisSome ⊢
          cases hblock : st.kernelEnv.blocks[pc.1]? with
          | none =>
              simp [hblock] at hisSome
          | some block =>
              simp [hblock] at hisSome ⊢
              cases hgi : block.body[pc.2]? with
              | none =>
                  simp [hgi] at hisSome
              | some gi =>
                  simp [hgi] at hisSome ⊢
                  cases hpart : Helpers.participatingRunnableLaneIds? warpState gi.guard? with
                  | none =>
                      simp [hpart] at hisSome
                  | some participants =>
                      simp [hpart] at hisSome ⊢
                      exact body_of_stepInstr?_computed
                        (cta := cta) (warp := warp) (warpState := warpState) (pc := pc)
                        (block := block) (gi := gi) (participants := participants)
                        ((State.wf_iff_bool st).2 hwf)
                        hwarp
                        ((WarpState.wf_iff_bool warpState).2 hwfWarp)
                        ((Helpers.lockstepRunnable_iff_bool warpState).2 hlock)
                        ((Helpers.runnablePc_iff_bool warpState pc).2 hpc)
                        hblock
                        hgi
                        ((Helpers.participatingRunnable_iff_bool warpState gi.guard?
                          participants).2 hpart)
                        hisSome

theorem term_of_currentTermStep?_computed
    {st : State} {cta : CTAId} {warp : WarpId}
    (hisSome : (currentTermStep? st cta warp).isSome = true) :
    StepMachine st
      (match currentTermStep? st cta warp with
       | some st' => st'
       | none => st) := by
  unfold currentTermStep? at hisSome ⊢
  cases hwf : st.wf? <;> simp [hwf] at hisSome ⊢
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hisSome
  | some warpState =>
      simp [hwarp] at hisSome ⊢
      cases hwfWarp : warpState.wf? <;> simp [hwfWarp] at hisSome ⊢
      cases hlock : Helpers.lockstepRunnable? warpState <;> simp [hlock] at hisSome ⊢
      cases hpc : Helpers.currentRunnablePc? warpState with
      | none =>
          simp [hpc] at hisSome
      | some pc =>
          simp [hpc] at hisSome ⊢
          cases hblock : st.kernelEnv.blocks[pc.1]? with
          | none =>
              simp [hblock] at hisSome
          | some block =>
              simp [hblock] at hisSome ⊢
              cases hbody : block.body[pc.2]? with
              | some _ =>
                  simp [hbody] at hisSome
              | none =>
                  simp [hbody] at hisSome ⊢
                  exact term_of_stepTerminator?_computed
                    (cta := cta) (warp := warp) (warpState := warpState) (pc := pc)
                    (block := block)
                    ((State.wf_iff_bool st).2 hwf)
                    hwarp
                    ((WarpState.wf_iff_bool warpState).2 hwfWarp)
                    ((Helpers.lockstepRunnable_iff_bool warpState).2 hlock)
                    ((Helpers.runnablePc_iff_bool warpState pc).2 hpc)
                    hblock
                    hbody
                    hisSome

syntax "cstep" : tactic

macro_rules
  | `(tactic| cstep) =>
      `(tactic|
        first
        | exact StepMachine.body_of_currentInstrStep?_computed (by native_decide)
        | exact StepMachine.term_of_currentTermStep?_computed (by native_decide))

end StepMachine
end CLean
