import CLean.Proof.InstrCompute
import CLean.Proof.Determinism

/-! # Per-instruction value-tracking lemmas

Where `Proof/Lemmas.lean` provides *frame* lemmas (what is preserved by
each instruction) and `Proof/InstrCompute.lean` provides the structural
unfoldings of `applyToLaneIds?`, this module provides **value-tracking**
lemmas: for each instruction class, the specific register / predicate /
memory effect at a participating lane.

These are the missing companion to enable per-lane symbolic execution of
straight-line kernel bodies (saxpy, vector-add). The layer of granularity
is one full `stepInstr?` (including the trailing `advanceRunnablePcs?`),
specialized to the assignReg / assignPred / load / cvta / isspacep / store
constructors used by the supported kernels.

## Pattern

Each lemma takes:
- a `stepInstr?` success hypothesis,
- a `Nodup` and `lane ∈ participants` witness,
- the per-lane evaluation hypothesis (`evalRValue?` / `evalCmp?` / etc.)

and concludes:
- the lane has the new register / predicate value,
- the pc was advanced by one.
-/

namespace CLean

open Helpers

/-! ## `advanceRunnablePcs?` value preservation for any lane -/

/-- `advanceRunnablePcs?` advances the pc by 1 for any runnable lane whose
pc matches the current runnable pc. -/
theorem advanceRunnablePcs?_advances_lane_pc
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {warpState : WarpState} {pc : PC} {laneState : LaneState}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hRunnable : lane ∈ runnableLaneIds warpState)
    (hStep : advanceRunnablePcs? st cta warp = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  unfold advanceRunnablePcs? at hStep
  rw [hWarp] at hStep; simp at hStep
  rw [hPc] at hStep; simp at hStep
  set lanes :=
    (runnableLaneIds warpState).filter fun lane =>
      match warpState.getLane? lane with
      | some laneState => laneState.pc == pc
      | none => false
  -- Show `lane ∈ lanes`.
  have hWarpLane : warpState.getLane? lane = some laneState := by
    -- getLane? at the state level uses the warp state's getLane?
    have := hLane
    unfold State.getLane? at this
    rw [hWarp] at this
    simp at this
    exact this
  have hLaneInFilter : lane ∈ lanes := by
    apply List.mem_filter.mpr
    exact ⟨hRunnable, by rw [hWarpLane]; simp [hLanePc]⟩
  have hLaneIdsNoDup : (laneIds : List LaneId).Nodup := by
    unfold laneIds; exact List.nodup_finRange 32
  have hRunNoDup : (runnableLaneIds warpState).Nodup :=
    List.Nodup.filter _ hLaneIdsNoDup
  have hLanesNoDup : lanes.Nodup := List.Nodup.filter _ hRunNoDup
  have hApply := applyToLaneIds?_lane_in (cta := cta) (warp := warp)
    (f := fun _ ls => some (advancePcForLane ls))
    lanes hLanesNoDup st st' hWf hStep lane hLaneInFilter laneState hLane
  obtain ⟨laneState', hF, hGet⟩ := hApply
  simp at hF
  subst hF
  refine ⟨advancePcForLane laneState, hGet, ?_⟩
  unfold advancePcForLane
  simp [hLanePc]

/-- `advanceRunnablePcs?` only modifies pcs; lane registers/predicates
are untouched. -/
theorem advanceRunnablePcs?_preserves_lane_regs_preds
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    (hWf : State.wf st)
    (hStep : advanceRunnablePcs? st cta warp = some st')
    {laneState : LaneState}
    (hLane : st.getLane? cta warp lane = some laneState) :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status := by
  unfold advanceRunnablePcs? at hStep
  cases hWarp : st.getWarp? cta warp with
  | none => simp [hWarp] at hStep
  | some warpState =>
      simp [hWarp] at hStep
      cases hPc : currentRunnablePc? warpState with
      | none => simp [hPc] at hStep
      | some pc =>
          simp [hPc] at hStep
          set lanes :=
            (runnableLaneIds warpState).filter fun lane =>
              match warpState.getLane? lane with
              | some laneState => laneState.pc == pc
              | none => false
          -- Case split: is `lane` in the filtered list?
          have hLaneIdsNoDup : (laneIds : List LaneId).Nodup := by
            unfold laneIds; exact List.nodup_finRange 32
          have hRunNoDup : (runnableLaneIds warpState).Nodup :=
            List.Nodup.filter _ hLaneIdsNoDup
          by_cases hIn : lane ∈ lanes
          · -- Lane is in the apply set: its state goes through `advancePcForLane`.
            have hNoDup : lanes.Nodup := List.Nodup.filter _ hRunNoDup
            have hApply := applyToLaneIds?_lane_in (cta := cta) (warp := warp)
              (f := fun _ ls => some (advancePcForLane ls))
              lanes hNoDup st st' hWf hStep lane hIn laneState hLane
            obtain ⟨laneState', hF, hGet⟩ := hApply
            simp at hF
            subst hF
            exact ⟨advancePcForLane laneState, hGet, rfl, rfl, rfl, rfl⟩
          · -- Lane not in the apply set: state is unchanged.
            have hPres :=
              applyToLaneIds?_lane_not_in cta warp
                (fun _ ls => some (advancePcForLane ls)) lanes st st' hStep lane hIn
            refine ⟨laneState, ?_, rfl, rfl, rfl, rfl⟩
            rw [hPres]; exact hLane

/-! ## Supporting preservation lemmas -/

/-- `applyToLaneIds?` preserves `State.wf` when it succeeds. -/
theorem applyToLaneIds?_preserves_wf
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hWf : State.wf st)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    State.wf st' := by
  induction lanes generalizing st with
  | nil =>
      rw [applyToLaneIds?_nil] at h
      cases h
      exact hWf
  | cons l rest ih =>
      rw [applyToLaneIds?_cons] at h
      cases hLane : st.getLane? cta warp l with
      | none => simp [hLane] at h
      | some ls =>
          simp [hLane] at h
          cases hF : f l ls with
          | none => simp [hF] at h
          | some ls' =>
              simp [hF] at h
              cases hSet : st.setLane cta warp l ls' with
              | none => simp [hSet] at h
              | some stMid =>
                  simp [hSet] at h
                  have hWfMid := State.wf_of_setLane hWf hSet
                  exact ih (st := stMid) hWfMid h

/-- Explicit recursion mirroring the for-loop inside
`participatingRunnableLaneIds?`. Each call prepends `lane` to `acc` only if
it appears in `lanes` once and its predicate test passes. -/
private def participantsAux (warp : WarpState) (pc : PC) (g : Option Guard) :
    List LaneId → List LaneId → Option (List LaneId)
  | [], acc => some acc
  | lane :: rest, acc =>
      (warp.getLane? lane).bind fun ls =>
        if ls.pc = pc then
          (guardHolds? ls g).bind fun passes =>
            if passes then
              participantsAux warp pc g rest (lane :: acc)
            else
              participantsAux warp pc g rest acc
        else
          participantsAux warp pc g rest acc

private theorem participantsAux_subset
    (warp : WarpState) (pc : PC) (g : Option Guard) :
    ∀ (lanes acc result : List LaneId),
      participantsAux warp pc g lanes acc = some result →
      ∀ x ∈ result, x ∈ acc ∨ x ∈ lanes := by
  intro lanes
  induction lanes with
  | nil => intro acc result h x hx; left; simp [participantsAux] at h; exact h ▸ hx
  | cons l rest ih =>
      intro acc result h x hx
      simp only [participantsAux] at h
      cases hLane : warp.getLane? l with
      | none => rw [hLane] at h; simp at h
      | some ls =>
          rw [hLane] at h
          simp at h
          by_cases hPc : ls.pc = pc
          · rw [if_pos hPc] at h
            cases hGuard : guardHolds? ls g with
            | none => rw [hGuard] at h; simp at h
            | some passes =>
                rw [hGuard] at h
                simp at h
                by_cases hPass : passes
                · subst hPass
                  rw [if_pos rfl] at h
                  have := ih (l :: acc) result h x hx
                  rcases this with hin | hin
                  · rcases List.mem_cons.1 hin with rfl | hin'
                    · right; exact List.mem_cons_self
                    · left; exact hin'
                  · right; exact List.mem_cons_of_mem _ hin
                · have hPass' : passes = false := by
                    cases passes <;> simp at hPass; rfl
                  rw [hPass', if_neg (by decide)] at h
                  have := ih acc result h x hx
                  rcases this with hin | hin
                  · left; exact hin
                  · right; exact List.mem_cons_of_mem _ hin
          · rw [if_neg hPc] at h
            have := ih acc result h x hx
            rcases this with hin | hin
            · left; exact hin
            · right; exact List.mem_cons_of_mem _ hin

private theorem participantsAux_nodup
    (warp : WarpState) (pc : PC) (g : Option Guard) :
    ∀ (lanes acc result : List LaneId),
      lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) →
      participantsAux warp pc g lanes acc = some result →
      result.Nodup := by
  intro lanes
  induction lanes with
  | nil => intro acc result _ hAcc _ h; simp [participantsAux] at h; exact h ▸ hAcc
  | cons l rest ih =>
      intro acc result hLanesNoDup hAccNoDup hDisj h
      simp only [participantsAux] at h
      cases hLane : warp.getLane? l with
      | none => rw [hLane] at h; simp at h
      | some ls =>
          rw [hLane] at h
          simp at h
          have hRestNoDup : rest.Nodup := (List.nodup_cons.1 hLanesNoDup).2
          have hLNotInRest : l ∉ rest := (List.nodup_cons.1 hLanesNoDup).1
          have hDisjRest : ∀ x ∈ acc, x ∉ rest := by
            intro x hx hxRest
            exact hDisj x hx (List.mem_cons_of_mem _ hxRest)
          have hLNotInAcc : l ∉ acc := by
            intro hL
            exact hDisj l hL List.mem_cons_self
          by_cases hPc : ls.pc = pc
          · rw [if_pos hPc] at h
            cases hGuard : guardHolds? ls g with
            | none => rw [hGuard] at h; simp at h
            | some passes =>
                rw [hGuard] at h
                simp at h
                by_cases hPass : passes
                · subst hPass
                  rw [if_pos rfl] at h
                  apply ih (l :: acc) result hRestNoDup
                  · exact List.nodup_cons.2 ⟨hLNotInAcc, hAccNoDup⟩
                  · intro x hx hxRest
                    rcases List.mem_cons.1 hx with rfl | hx'
                    · exact hLNotInRest hxRest
                    · exact hDisjRest x hx' hxRest
                  · exact h
                · have hPass' : passes = false := by cases passes <;> simp at hPass; rfl
                  rw [hPass', if_neg (by decide)] at h
                  exact ih acc result hRestNoDup hAccNoDup hDisjRest h
          · rw [if_neg hPc] at h
            exact ih acc result hRestNoDup hAccNoDup hDisjRest h

/-- Generalized for-loop invariant: the for-loop body inside
`participatingRunnableLaneIds?`, run over a `Nodup` list with an
accumulator disjoint from the list, produces a `Nodup` output. -/
private theorem participants_forIn_nodup
    (ws : WarpState) (pc : PC) (g : Option Guard) :
    ∀ (lanes : List LaneId) (acc : List LaneId) (result : List LaneId),
      lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) →
      (forIn lanes acc fun lane out => do
          let some ls := ws.getLane? lane | none
          if ls.pc = pc then
            let passes <- guardHolds? ls g
            if passes then pure (ForInStep.yield (lane :: out))
            else pure (ForInStep.yield out)
          else pure (ForInStep.yield out)) = some result →
      result.Nodup := by
  intro lanes
  induction lanes with
  | nil =>
      intro acc result _ hAcc _ h
      simp at h
      exact h ▸ hAcc
  | cons l rest ih =>
      intro acc result hLanesNoDup hAccNoDup hDisj h
      simp [List.forIn_cons] at h
      have hRestNoDup : rest.Nodup := (List.nodup_cons.1 hLanesNoDup).2
      have hLNotInRest : l ∉ rest := (List.nodup_cons.1 hLanesNoDup).1
      have hLNotInAcc : l ∉ acc := fun hL => hDisj l hL List.mem_cons_self
      have hDisjRest : ∀ x ∈ acc, x ∉ rest := fun x hx hxr =>
        hDisj x hx (List.mem_cons_of_mem _ hxr)
      cases hLane : ws.getLane? l with
      | none => rw [hLane] at h; simp at h
      | some ls =>
          rw [hLane] at h
          simp at h
          by_cases hPc : ls.pc = pc
          · rw [if_pos hPc] at h
            cases hGuard : guardHolds? ls g with
            | none => rw [hGuard] at h; simp at h
            | some passes =>
                rw [hGuard] at h
                simp at h
                by_cases hPass : passes = true
                · rw [hPass, if_pos rfl] at h
                  apply ih (l :: acc) result hRestNoDup
                  · exact List.nodup_cons.2 ⟨hLNotInAcc, hAccNoDup⟩
                  · intro x hx hxRest
                    rcases List.mem_cons.1 hx with rfl | hx'
                    · exact hLNotInRest hxRest
                    · exact hDisjRest x hx' hxRest
                  · exact h
                · have hPassFalse : passes = false := by
                    cases passes
                    · rfl
                    · exact absurd rfl hPass
                  rw [hPassFalse] at h
                  simp at h
                  exact ih acc result hRestNoDup hAccNoDup hDisjRest h
          · rw [if_neg hPc] at h
            exact ih acc result hRestNoDup hAccNoDup hDisjRest h

/-- The participants list produced by `participatingRunnableLaneIds?` has
no duplicates (inherited from `runnableLaneIds`'s nodup). -/
theorem participatingRunnableLaneIds?_nodup
    {ws : WarpState} {g : Option Guard} {parts : List LaneId}
    (h : participatingRunnableLaneIds? ws g = some parts) :
    parts.Nodup := by
  unfold participatingRunnableLaneIds? at h
  cases hPc : currentRunnablePc? ws with
  | none => rw [hPc] at h; simp at h
  | some pc =>
      rw [hPc] at h
      simp at h
      have hRunNodup : (runnableLaneIds ws).Nodup := by
        unfold runnableLaneIds
        apply List.Nodup.filter
        unfold laneIds
        exact List.nodup_finRange 32
      -- h : forIn(...).bind fun r => some r.reverse = some parts
      -- Extract outVal such that forIn = some outVal and parts = outVal.reverse
      rw [Option.bind_eq_some_iff] at h
      obtain ⟨outVal, hLoop, hRev⟩ := h
      simp at hRev
      subst hRev
      rw [List.nodup_reverse]
      exact participants_forIn_nodup ws pc g (runnableLaneIds ws) [] outVal
        hRunNodup List.nodup_nil (by intro x hx; exact absurd hx List.not_mem_nil) hLoop

/-! ## Warp structural preservation under `applyToLaneIds?`

The key property: if `f` preserves `(status, pc)` per lane, then
`applyToLaneIds?` leaves the warp's active mask and per-lane
`(status, pc)` unchanged, hence `runnableLaneIds` and
`currentRunnablePc?` are preserved.

This is the missing piece for chaining per-instruction lemmas through
multiple steps of `stepInstr?`. -/

/-- A function on lane states preserves (status, pc). -/
def PreservesStatusPc (f : LaneId → LaneState → Option LaneState) : Prop :=
  ∀ l ls ls', f l ls = some ls' → ls'.pc = ls.pc ∧ ls'.status = ls.status

theorem writeReg_preservesStatusPc (dst : RegName) (v : Value) :
    ∀ (_l : LaneId) (ls ls' : LaneState),
      (some (writeReg ls dst v) : Option LaneState) = some ls' →
      ls'.pc = ls.pc ∧ ls'.status = ls.status := by
  intro _ _ _ h; simp at h; subst h
  unfold writeReg; exact ⟨rfl, rfl⟩

theorem writePred_preservesStatusPc (dst : PredName) (b : Bool) :
    ∀ (_l : LaneId) (ls ls' : LaneState),
      (some (writePred ls dst b) : Option LaneState) = some ls' →
      ls'.pc = ls.pc ∧ ls'.status = ls.status := by
  intro _ _ _ h; simp at h; subst h
  unfold writePred; exact ⟨rfl, rfl⟩

/-- `setLane` preserves the warp's `activeMask` and other lanes' states. -/
private theorem setLane_warp_struct
    {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls' : LaneState}
    {ws : WarpState}
    (hWarp : st.getWarp? cta warp = some ws)
    (hSet : st.setLane cta warp l ls' = some st') :
    ∃ ws' : WarpState,
      st'.getWarp? cta warp = some ws' ∧
      ws'.activeMask = ws.activeMask ∧
      ws'.lanes.size = ws.lanes.size := by
  unfold State.setLane at hSet
  rw [hWarp] at hSet; simp at hSet
  unfold State.setWarp State.getCTA? at hSet
  cases hCta : st.ctas[cta]? with
  | none =>
      unfold State.getWarp? State.getCTA? at hWarp
      rw [hCta] at hWarp; simp at hWarp
  | some ctaState =>
      rw [hCta] at hSet
      simp at hSet
      subst hSet
      refine ⟨ws.setLane l ls', ?_, ?_, ?_⟩
      · unfold State.getWarp? State.getCTA? State.setCTA
        simp [Std.HashMap.getElem?_insert]
      · unfold WarpState.setLane; rfl
      · unfold WarpState.setLane; simp

/-- Lane `l`'s status/pc after a `setLane` with a status/pc-preserving
update is unchanged (modulo equality with the new state). -/
private theorem setLane_other_lane_status_pc_preserved
    {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls' : LaneState}
    (hSet : st.setLane cta warp l ls' = some st')
    (other : LaneId) (h : other ≠ l) :
    st'.getLane? cta warp other = st.getLane? cta warp other := by
  exact setLane_preserves_other_lane hSet other h

/-- `applyToLaneIds?` with a status/pc-preserving function: any lane's
`status` and `pc` are preserved (whether or not the lane is in the input
list). -/
theorem applyToLaneIds?_preserves_status_pc_lane
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (hf : PreservesStatusPc f)
    (hWf : State.wf st)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    ∀ (other : LaneId) (ls : LaneState),
      st.getLane? cta warp other = some ls →
      ∃ ls' : LaneState,
        st'.getLane? cta warp other = some ls' ∧
        ls'.pc = ls.pc ∧ ls'.status = ls.status := by
  induction lanes generalizing st with
  | nil =>
      intro other ls hLane
      rw [applyToLaneIds?_nil] at h
      cases h
      exact ⟨ls, hLane, rfl, rfl⟩
  | cons l rest ih =>
      intro other ls hLane
      rw [applyToLaneIds?_cons] at h
      cases hL : st.getLane? cta warp l with
      | none => rw [hL] at h; simp at h
      | some lsL =>
          rw [hL] at h; simp at h
          cases hF : f l lsL with
          | none => rw [hF] at h; simp at h
          | some lsL' =>
              rw [hF] at h; simp at h
              cases hSet : st.setLane cta warp l lsL' with
              | none => rw [hSet] at h; simp at h
              | some stMid =>
                  rw [hSet] at h; simp at h
                  have hWfMid : State.wf stMid := State.wf_of_setLane hWf hSet
                  have hWarpExists : ∃ ws, st.getWarp? cta warp = some ws := by
                    unfold State.setLane at hSet
                    cases hW : st.getWarp? cta warp with
                    | none => rw [hW] at hSet; simp at hSet
                    | some ws => exact ⟨ws, rfl⟩
                  obtain ⟨wsExt, hWsExt⟩ := hWarpExists
                  have hWfW : WarpState.wf wsExt := WarpState.wf_of_getWarp? hWf hWsExt
                  by_cases hEq : other = l
                  · -- Lane is `l`: its state in stMid is lsL'.
                    have hStMidGet : stMid.getLane? cta warp l = some lsL' :=
                      setLane_get_self hSet hWsExt hWfW
                    rw [hEq]
                    have hRec := ih hWfMid h l lsL' hStMidGet
                    obtain ⟨ls', hGet', hPc, hStat⟩ := hRec
                    refine ⟨ls', hGet', ?_, ?_⟩
                    · have hLsEq : ls = lsL := by
                        rw [hEq] at hLane
                        rw [hLane] at hL
                        exact Option.some.inj hL
                      subst hLsEq
                      rw [hPc]
                      exact (hf l ls lsL' hF).1
                    · have hLsEq : ls = lsL := by
                        rw [hEq] at hLane
                        rw [hLane] at hL
                        exact Option.some.inj hL
                      subst hLsEq
                      rw [hStat]
                      exact (hf l ls lsL' hF).2
                  · -- Lane is not `l`: its state in stMid is the same as in st.
                    have hMidLane : stMid.getLane? cta warp other = some ls := by
                      rw [setLane_preserves_other_lane hSet other hEq]
                      exact hLane
                    exact ih hWfMid h other ls hMidLane

/-- `applyToLaneIds?` with `f` preserving status/pc preserves the
warp-level `activeMask` (the structural lane membership in the active set). -/
theorem applyToLaneIds?_preserves_activeMask
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    (h : applyToLaneIds? st cta warp lanes f = some st')
    (ws : WarpState) (hWs : st.getWarp? cta warp = some ws) :
    ∃ ws' : WarpState,
      st'.getWarp? cta warp = some ws' ∧
      ws'.activeMask = ws.activeMask := by
  induction lanes generalizing st ws with
  | nil =>
      rw [applyToLaneIds?_nil] at h
      cases h
      exact ⟨ws, hWs, rfl⟩
  | cons l rest ih =>
      rw [applyToLaneIds?_cons] at h
      cases hL : st.getLane? cta warp l with
      | none => rw [hL] at h; simp at h
      | some lsL =>
          rw [hL] at h; simp at h
          cases hF : f l lsL with
          | none => rw [hF] at h; simp at h
          | some lsL' =>
              rw [hF] at h; simp at h
              cases hSet : st.setLane cta warp l lsL' with
              | none => rw [hSet] at h; simp at h
              | some stMid =>
                  rw [hSet] at h; simp at h
                  have hStruct := setLane_warp_struct hWs hSet
                  obtain ⟨wsMid, hWsMid, hMaskMid, _⟩ := hStruct
                  obtain ⟨ws', hWs', hMask'⟩ := ih h wsMid hWsMid
                  exact ⟨ws', hWs', hMask'.trans hMaskMid⟩

/-! ## `assignReg` value tracking -/

/-- After `stepInstr?` on an `assignReg dst rhs`, lane `j` (in
participants) has `regs[dst]? = some val` where `val` is what
`evalRValue?` computes on the *pre-step* state. -/
theorem stepInstr?_assignReg_lane_value
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {rhs : RValue} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId}
    {lane : LaneId} {laneState : LaneState} {val : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hEval : evalRValue? st cta warp lane rhs = some val)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .assignReg dst rhs } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs[dst]? = some val := by
  -- Unfold stepInstr? at assignReg.
  unfold stepInstr? at hStep
  rw [hWarp] at hStep
  simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep
  simp at hStep
  rw [hPart] at hStep
  simp at hStep
  -- hStep : (applyToLaneIds? st cta warp participants (fun lane laneState =>
  --             (evalRValue? st cta warp lane rhs).bind fun v => some (writeReg laneState dst v)))
  --         .bind fun stMid => advanceRunnablePcs? stMid cta warp = some st'
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalRValue? st cta warp lane rhs).bind fun v => some (writeReg laneState dst v)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep
      simp at hStep
      -- Use applyToLaneIds?_lane_in to find lane's state in stMid.
      have hPartNoDup : participants.Nodup :=
        participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalRValue? st cta warp lane rhs).bind fun v => some (writeReg laneState dst v))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      -- hF : (evalRValue? ... rhs).bind ... = some laneState_mid
      simp [hEval] at hF
      subst hF
      -- Now apply advanceRunnablePcs? to stMid.
      have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf hWf hApply
      have hAdvance :=
        advanceRunnablePcs?_preserves_lane_regs_preds (st := stMid) (st' := st')
          (cta := cta) (warp := warp) (lane := lane)
          hWfMid hStep hGetMid
      obtain ⟨laneState', hGetFinal, hRegsEq, _, _, _⟩ := hAdvance
      refine ⟨laneState', hGetFinal, ?_⟩
      rw [hRegsEq]
      -- laneState_mid.regs = (writeReg laneState dst val).regs, look up dst.
      unfold writeReg
      simp [Std.HashMap.getElem?_insert]

/-! ## `assignPred` value tracking -/

/-- After `stepInstr?` on `assignPred dst cmp`, lane `j` (in participants)
has `preds[dst]? = some b` where `b` is the comparison result. -/
theorem stepInstr?_assignPred_lane_value
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId}
    {lane : LaneId} {laneState : LaneState} {b : Bool}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hEval : evalCmp? st cta warp lane cmp = some b)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .assignPred dst cmp } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.preds[dst]? = some b := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalCmp? st cta warp lane cmp).bind fun b => some (writePred laneState dst b)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalCmp? st cta warp lane cmp).bind fun b => some (writePred laneState dst b))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hEval] at hF
      subst hF
      have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf hWf hApply
      have hAdvance :=
        advanceRunnablePcs?_preserves_lane_regs_preds (st := stMid) (st' := st')
          (cta := cta) (warp := warp) (lane := lane) hWfMid hStep hGetMid
      obtain ⟨laneState', hGetFinal, _, hPredsEq, _, _⟩ := hAdvance
      refine ⟨laneState', hGetFinal, ?_⟩
      rw [hPredsEq]
      unfold writePred
      simp [Std.HashMap.getElem?_insert]

/-! ## `load` value tracking -/

/-- After `stepInstr?` on `load dst src`, lane `j` (in participants) has
`regs[dst]? = some val` where `val` is what `readMem?` returns at the
resolved per-lane address. -/
theorem stepInstr?_load_lane_value
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {src : TypedAddr} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId}
    {lane : LaneId} {laneState : LaneState} {addr : Addr} {val : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hAddr : resolveAddr? st cta warp lane src = some addr)
    (hRead : readMem? st src.space src.ty addr = some val)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .load dst src } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs[dst]? = some val := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (resolveAddr? st cta warp lane src).bind fun addr =>
          (readMem? st src.space src.ty addr).bind fun value =>
            some (writeReg laneState dst value)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (resolveAddr? st cta warp lane src).bind fun addr =>
              (readMem? st src.space src.ty addr).bind fun value =>
                some (writeReg laneState dst value))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hAddr, hRead] at hF
      subst hF
      have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf hWf hApply
      have hAdvance :=
        advanceRunnablePcs?_preserves_lane_regs_preds (st := stMid) (st' := st')
          (cta := cta) (warp := warp) (lane := lane) hWfMid hStep hGetMid
      obtain ⟨laneState', hGetFinal, hRegsEq, _, _, _⟩ := hAdvance
      refine ⟨laneState', hGetFinal, ?_⟩
      rw [hRegsEq]
      unfold writeReg
      simp [Std.HashMap.getElem?_insert]

/-! ## Strengthened per-lane "full state" lemmas

Where the basic value lemmas above give only the new register / predicate
value, these strengthened versions characterize the FULL per-lane state
after `stepInstr?` — that is, `regs`, `preds`, `localMem`, `status`, and
`pc` all in one statement. This is what the saxpy lane-sim chain needs:
to know that lane j after step k has its k-th register set AND its other
registers preserved AND its pc advanced. -/

/-- Membership in `participants` implies the lane is runnable and its pc
equals the current runnable pc. Extracted from the body of
`participatingRunnableLaneIds?`. -/
private theorem participant_runnable_pc
    {ws : WarpState} {g : Option Guard} {parts : List LaneId} {pc : PC}
    (hPc : currentRunnablePc? ws = some pc)
    (hPart : participatingRunnableLaneIds? ws g = some parts) :
    ∀ lane ∈ parts,
      lane ∈ runnableLaneIds ws ∧
      ∃ ls : LaneState, ws.getLane? lane = some ls ∧ ls.pc = pc := by
  intro lane hLane
  unfold participatingRunnableLaneIds? at hPart
  rw [hPc] at hPart
  simp at hPart
  -- Generalized invariant over the for-loop accumulator.
  suffices key : ∀ (lanes acc result : List LaneId),
      (∀ x ∈ acc, x ∈ runnableLaneIds ws ∧
        ∃ ls : LaneState, ws.getLane? x = some ls ∧ ls.pc = pc) →
      (∀ x ∈ lanes, x ∈ runnableLaneIds ws) →
      (forIn lanes acc fun lane out => do
          let some ls := ws.getLane? lane | none
          if ls.pc = pc then
            let passes <- guardHolds? ls g
            if passes then pure (ForInStep.yield (lane :: out))
            else pure (ForInStep.yield out)
          else pure (ForInStep.yield out)) = some result →
      ∀ x ∈ result, x ∈ runnableLaneIds ws ∧
        ∃ ls : LaneState, ws.getLane? x = some ls ∧ ls.pc = pc by
    rw [Option.bind_eq_some_iff] at hPart
    obtain ⟨outVal, hLoop, hRev⟩ := hPart
    simp at hRev
    subst hRev
    rw [List.mem_reverse] at hLane
    apply key (runnableLaneIds ws) [] outVal
    · intro x hx; exact absurd hx List.not_mem_nil
    · intro x hx; exact hx
    · exact hLoop
    · exact hLane
  intro lanes
  induction lanes with
  | nil =>
      intro acc result hAcc _ h x hx
      simp at h
      subst h
      exact hAcc x hx
  | cons l rest ih =>
      intro acc result hAcc hLanes h x hx
      simp [List.forIn_cons] at h
      have hLanesRest : ∀ y ∈ rest, y ∈ runnableLaneIds ws :=
        fun y hy => hLanes y (List.mem_cons_of_mem _ hy)
      have hLrunnable : l ∈ runnableLaneIds ws := hLanes l List.mem_cons_self
      cases hLane : ws.getLane? l with
      | none => rw [hLane] at h; simp at h
      | some ls =>
          rw [hLane] at h; simp at h
          by_cases hPcEq : ls.pc = pc
          · rw [if_pos hPcEq] at h
            cases hG : guardHolds? ls g with
            | none => rw [hG] at h; simp at h
            | some passes =>
                rw [hG] at h; simp at h
                by_cases hP : passes = true
                · rw [hP, if_pos rfl] at h
                  apply ih (l :: acc) result
                  · intro y hy
                    rcases List.mem_cons.1 hy with rfl | hy'
                    · exact ⟨hLrunnable, ls, hLane, hPcEq⟩
                    · exact hAcc y hy'
                  · exact hLanesRest
                  · exact h
                  · exact hx
                · have hPf : passes = false := by cases passes <;> simp_all
                  rw [hPf] at h; simp at h
                  exact ih acc result hAcc hLanesRest h x hx
          · rw [if_neg hPcEq] at h
            exact ih acc result hAcc hLanesRest h x hx

/-! ## `cvta` value tracking -/

/-- After `stepInstr?` on `cvta dst space src`, lane `j` has
`regs[dst]? = some gaddr` where `gaddr` is the converted address. -/
theorem stepInstr?_cvta_lane_value
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId}
    {lane : LaneId} {laneState : LaneState} {srcVal : Value} {gaddr : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hSrc : evalRValue? st cta warp lane src = some srcVal)
    (hCvta : evalCvta? space srcVal = some gaddr)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .cvta dst space src } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs[dst]? = some gaddr := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalRValue? st cta warp lane src).bind fun value =>
          (evalCvta? space value).bind fun gaddr =>
            some (writeReg laneState dst gaddr)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalRValue? st cta warp lane src).bind fun value =>
              (evalCvta? space value).bind fun gaddr =>
                some (writeReg laneState dst gaddr))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hSrc, hCvta] at hF
      subst hF
      have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf hWf hApply
      have hAdvance :=
        advanceRunnablePcs?_preserves_lane_regs_preds (st := stMid) (st' := st')
          (cta := cta) (warp := warp) (lane := lane) hWfMid hStep hGetMid
      obtain ⟨laneState', hGetFinal, hRegsEq, _, _, _⟩ := hAdvance
      refine ⟨laneState', hGetFinal, ?_⟩
      rw [hRegsEq]
      unfold writeReg
      simp [Std.HashMap.getElem?_insert]

/-! ## Warp-level corollaries for `runnableLaneIds` and `currentRunnablePc?`

These corollaries show that `applyToLaneIds?` with a status/pc-preserving
function preserves the warp-level scheduling invariants — the set of
runnable lanes and the current runnable pc. These are exactly what the
strengthened `_lane_full` lemmas need to chain instructions through a
straight-line basic block. -/

/-- For a wf warp, every `Fin 32` lane has a defined `getLane?`. -/
private theorem WarpState.getLane?_isSome_of_wf
    {ws : WarpState} (hWf : WarpState.wf ws) (lane : LaneId) :
    ∃ ls : LaneState, ws.getLane? lane = some ls := by
  have hSize : ws.lanes.size = 32 := by
    simpa [WarpState.wf, WarpState.wf?] using hWf
  have hLt : lane.val < ws.lanes.size := by simp [hSize]
  unfold WarpState.getLane?
  exact ⟨ws.lanes[lane.val], Array.getElem?_eq_getElem hLt⟩

/-- Warp-level lifting: for a wf state, `applyToLaneIds?` with a
status/pc-preserving function preserves every Fin 32 lane's status and pc
when projected through the *warp*'s `getLane?`. -/
private theorem applyToLaneIds?_preserves_status_pc_warp_lane
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    {ws ws' : WarpState}
    (hf : PreservesStatusPc f)
    (hWf : State.wf st)
    (hWs : st.getWarp? cta warp = some ws)
    (hWs' : st'.getWarp? cta warp = some ws')
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    ∀ (lane : LaneId) (ls : LaneState),
      ws.getLane? lane = some ls →
      ∃ ls' : LaneState,
        ws'.getLane? lane = some ls' ∧
        ls'.pc = ls.pc ∧ ls'.status = ls.status := by
  intro lane ls hLs
  have hStLane : st.getLane? cta warp lane = some ls := by
    unfold State.getLane?
    rw [hWs]
    simpa using hLs
  obtain ⟨ls', hSt'Lane, hPc, hStat⟩ :=
    applyToLaneIds?_preserves_status_pc_lane hf hWf h lane ls hStLane
  have hWsLane' : ws'.getLane? lane = some ls' := by
    have : st'.getLane? cta warp lane = some ls' := hSt'Lane
    unfold State.getLane? at this
    rw [hWs'] at this
    simpa using this
  exact ⟨ls', hWsLane', hPc, hStat⟩

/-- `applyToLaneIds?` with a status/pc-preserving function preserves
`runnableLaneIds` (because `laneIsRunnable` depends only on per-lane status
and the warp's `activeMask`, both preserved). -/
theorem applyToLaneIds?_preserves_runnableLaneIds
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    {ws ws' : WarpState}
    (hf : PreservesStatusPc f)
    (hWf : State.wf st)
    (hWs : st.getWarp? cta warp = some ws)
    (hWs' : st'.getWarp? cta warp = some ws')
    (hWsWf : WarpState.wf ws)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    runnableLaneIds ws' = runnableLaneIds ws := by
  -- activeMask preserved.
  obtain ⟨wsExt, hWsExt, hMask⟩ :=
    applyToLaneIds?_preserves_activeMask h ws hWs
  have hwsEq : ws' = wsExt := by
    rw [hWs'] at hWsExt; exact Option.some.inj hWsExt
  -- Per-lane runnability agrees.
  have hLaneEq : ∀ lane : LaneId, laneIsRunnable ws' lane = laneIsRunnable ws lane := by
    intro lane
    obtain ⟨ls, hLs⟩ := WarpState.getLane?_isSome_of_wf hWsWf lane
    obtain ⟨ls', hLs', _, hStat⟩ :=
      applyToLaneIds?_preserves_status_pc_warp_lane hf hWf hWs hWs' h lane ls hLs
    unfold laneIsRunnable
    rw [hLs, hLs']
    simp [hStat, hwsEq, hMask]
  unfold runnableLaneIds
  exact List.filter_congr (fun lane _ => by rw [hLaneEq])

/-- `applyToLaneIds?` with a status/pc-preserving function preserves
`currentRunnablePc?`. -/
theorem applyToLaneIds?_preserves_currentRunnablePc?
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId}
    {ws ws' : WarpState}
    (hf : PreservesStatusPc f)
    (hWf : State.wf st)
    (hWs : st.getWarp? cta warp = some ws)
    (hWs' : st'.getWarp? cta warp = some ws')
    (hWsWf : WarpState.wf ws)
    (h : applyToLaneIds? st cta warp lanes f = some st') :
    currentRunnablePc? ws' = currentRunnablePc? ws := by
  have hRunEq :=
    applyToLaneIds?_preserves_runnableLaneIds hf hWf hWs hWs' hWsWf h
  unfold currentRunnablePc?
  rw [hRunEq]
  cases hList : runnableLaneIds ws with
  | nil => simp
  | cons head tail =>
      simp
      obtain ⟨ls, hLs⟩ := WarpState.getLane?_isSome_of_wf hWsWf head
      obtain ⟨ls', hLs', hPc, _⟩ :=
        applyToLaneIds?_preserves_status_pc_warp_lane hf hWf hWs hWs' h head ls hLs
      rw [hLs, hLs']; simp [hPc]

/-! ## Strengthened "lane-full" lemmas

These are the chainable forms needed for the saxpy lane chain: full
post-step lane state characterization — value written, pc advanced, all
other fields preserved.

We factor the common machinery into `step_lane_full_from_mid` so each
per-instruction `_full` lemma is a thin wrapper. -/

/-- Common machinery: given the mid-state after `applyToLaneIds?` and an
in-progress `advanceRunnablePcs?`, derive the full per-lane state at the
final state in terms of the mid lane state. The function `f` must
preserve status and pc per lane (so that the warp's scheduling structure
is preserved for the advance step). -/
private theorem step_lane_full_from_mid
    {st stMid st' : State} {cta : CTAId} {warp : WarpId}
    {f : LaneId → LaneState → Option LaneState}
    {warpState : WarpState} {pc : PC} {participants : List LaneId}
    {lane : LaneId} {laneState_mid : LaneState}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hRunIn : lane ∈ runnableLaneIds warpState)
    (hPres : PreservesStatusPc f)
    (hApply : applyToLaneIds? st cta warp participants f = some stMid)
    (hGetMid : stMid.getLane? cta warp lane = some laneState_mid)
    (hMidPc : laneState_mid.pc = pc)
    (hAdvance : advanceRunnablePcs? stMid cta warp = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState_mid.regs ∧
      laneState'.preds = laneState_mid.preds ∧
      laneState'.localMem = laneState_mid.localMem ∧
      laneState'.status = laneState_mid.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  have hWfMid : State.wf stMid := applyToLaneIds?_preserves_wf hWf hApply
  -- Get mid warp state.
  have hWarpMidEx : ∃ wsMid, stMid.getWarp? cta warp = some wsMid := by
    unfold State.getLane? at hGetMid
    cases hWM : stMid.getWarp? cta warp with
    | none => rw [hWM] at hGetMid; simp at hGetMid
    | some wsMid => exact ⟨wsMid, rfl⟩
  obtain ⟨wsMid, hWarpMid⟩ := hWarpMidEx
  have hWsWf : WarpState.wf warpState := WarpState.wf_of_getWarp? hWf hWarp
  -- Transfer scheduling structure via preservation.
  have hRunEq :=
    applyToLaneIds?_preserves_runnableLaneIds hPres hWf hWarp hWarpMid hWsWf hApply
  have hPcEq :=
    applyToLaneIds?_preserves_currentRunnablePc? hPres hWf hWarp hWarpMid hWsWf hApply
  have hPcMid : currentRunnablePc? wsMid = some pc := hPcEq.trans hPc
  have hRunMid : lane ∈ runnableLaneIds wsMid := by rw [hRunEq]; exact hRunIn
  -- Apply advance lemmas.
  have hAdvancePc :=
    advanceRunnablePcs?_advances_lane_pc (st := stMid) (st' := st')
      (cta := cta) (warp := warp) (lane := lane)
      (warpState := wsMid) (pc := pc) (laneState := laneState_mid)
      hWfMid hWarpMid hPcMid hGetMid hMidPc hRunMid hAdvance
  obtain ⟨laneState'_pc, hGetFinal_pc, hPcFinal⟩ := hAdvancePc
  have hAdvanceRegs :=
    advanceRunnablePcs?_preserves_lane_regs_preds (st := stMid) (st' := st')
      (cta := cta) (warp := warp) (lane := lane)
      hWfMid hAdvance hGetMid
  obtain ⟨laneState'_regs, hGetFinal_regs, hRegsEq, hPredsEq, hLocalEq, hStatusEq⟩ :=
    hAdvanceRegs
  have hLaneStateEq : laneState'_pc = laneState'_regs := by
    rw [hGetFinal_pc] at hGetFinal_regs
    exact Option.some.inj hGetFinal_regs
  refine ⟨laneState'_pc, hGetFinal_pc, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hLaneStateEq, hRegsEq]
  · rw [hLaneStateEq, hPredsEq]
  · rw [hLaneStateEq, hLocalEq]
  · rw [hLaneStateEq, hStatusEq]
  · rw [hPcFinal]

/-- After `stepInstr?` on `assignReg dst rhs`, the participating lane has
its `regs` updated to `laneState.regs.insert dst val`, its pc advanced by
1, and all other fields (preds, localMem, status) preserved. -/
theorem stepInstr?_assignReg_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {rhs : RValue} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId} {pc : PC}
    {lane : LaneId} {laneState : LaneState} {val : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hEval : evalRValue? st cta warp lane rhs = some val)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .assignReg dst rhs } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs.insert dst val ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalRValue? st cta warp lane rhs).bind fun v => some (writeReg laneState dst v)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalRValue? st cta warp lane rhs).bind fun v => some (writeReg laneState dst v))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hEval] at hF
      have hPres : PreservesStatusPc
          (fun l ls => (evalRValue? st cta warp l rhs).bind fun v => some (writeReg ls dst v)) := by
        intro l ls ls' hPF
        simp [Option.bind_eq_some_iff] at hPF
        obtain ⟨_, _, hWrite⟩ := hPF
        rw [← hWrite]; unfold writeReg; exact ⟨rfl, rfl⟩
      have hMidPc : laneState_mid.pc = pc := by
        rw [← hF]; unfold writeReg; exact hLanePc
      have hRunIn := (participant_runnable_pc hPc hPart lane hIn).1
      obtain ⟨ls', hGet', hRegs, hPreds, hLocal, hStatus, hPcAdv⟩ :=
        step_lane_full_from_mid hWf hWarp hPc hRunIn hPres hApply hGetMid hMidPc hStep
      refine ⟨ls', hGet', ?_, ?_, ?_, ?_, ?_⟩
      · rw [hRegs, ← hF]; rfl
      · rw [hPreds, ← hF]; rfl
      · rw [hLocal, ← hF]; rfl
      · rw [hStatus, ← hF]; rfl
      · exact hPcAdv

/-- Full-state characterization for `assignPred dst cmp`: lane gets
`preds[dst] := b`, all other fields preserved, pc advanced. -/
theorem stepInstr?_assignPred_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId} {pc : PC}
    {lane : LaneId} {laneState : LaneState} {b : Bool}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hEval : evalCmp? st cta warp lane cmp = some b)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .assignPred dst cmp } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds.insert dst b ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalCmp? st cta warp lane cmp).bind fun b => some (writePred laneState dst b)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalCmp? st cta warp lane cmp).bind fun b => some (writePred laneState dst b))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hEval] at hF
      have hPres : PreservesStatusPc
          (fun l ls => (evalCmp? st cta warp l cmp).bind fun b => some (writePred ls dst b)) := by
        intro l ls ls' hPF
        simp [Option.bind_eq_some_iff] at hPF
        rcases hPF with ⟨_, hW⟩ | ⟨_, hW⟩
        · rw [← hW]; unfold writePred; exact ⟨rfl, rfl⟩
        · rw [← hW]; unfold writePred; exact ⟨rfl, rfl⟩
      have hMidPc : laneState_mid.pc = pc := by
        rw [← hF]; unfold writePred; exact hLanePc
      have hRunIn := (participant_runnable_pc hPc hPart lane hIn).1
      obtain ⟨ls', hGet', hRegs, hPreds, hLocal, hStatus, hPcAdv⟩ :=
        step_lane_full_from_mid hWf hWarp hPc hRunIn hPres hApply hGetMid hMidPc hStep
      refine ⟨ls', hGet', ?_, ?_, ?_, ?_, ?_⟩
      · rw [hRegs, ← hF]; rfl
      · rw [hPreds, ← hF]; rfl
      · rw [hLocal, ← hF]; rfl
      · rw [hStatus, ← hF]; rfl
      · exact hPcAdv

/-- Full-state characterization for `load dst src`: lane gets
`regs[dst] := val` (from `readMem?`), all other fields preserved, pc advanced. -/
theorem stepInstr?_load_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {src : TypedAddr} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId} {pc : PC}
    {lane : LaneId} {laneState : LaneState} {addr : Addr} {val : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hAddr : resolveAddr? st cta warp lane src = some addr)
    (hRead : readMem? st src.space src.ty addr = some val)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .load dst src } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs.insert dst val ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (resolveAddr? st cta warp lane src).bind fun addr =>
          (readMem? st src.space src.ty addr).bind fun value =>
            some (writeReg laneState dst value)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (resolveAddr? st cta warp lane src).bind fun addr =>
              (readMem? st src.space src.ty addr).bind fun value =>
                some (writeReg laneState dst value))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hAddr, hRead] at hF
      have hPres : PreservesStatusPc
          (fun l ls =>
            (resolveAddr? st cta warp l src).bind fun a =>
              (readMem? st src.space src.ty a).bind fun v => some (writeReg ls dst v)) := by
        intro l ls ls' hPF
        simp [Option.bind_eq_some_iff] at hPF
        obtain ⟨_, _, _, _, hWrite⟩ := hPF
        rw [← hWrite]; unfold writeReg; exact ⟨rfl, rfl⟩
      have hMidPc : laneState_mid.pc = pc := by
        rw [← hF]; unfold writeReg; exact hLanePc
      have hRunIn := (participant_runnable_pc hPc hPart lane hIn).1
      obtain ⟨ls', hGet', hRegs, hPreds, hLocal, hStatus, hPcAdv⟩ :=
        step_lane_full_from_mid hWf hWarp hPc hRunIn hPres hApply hGetMid hMidPc hStep
      refine ⟨ls', hGet', ?_, ?_, ?_, ?_, ?_⟩
      · rw [hRegs, ← hF]; rfl
      · rw [hPreds, ← hF]; rfl
      · rw [hLocal, ← hF]; rfl
      · rw [hStatus, ← hF]; rfl
      · exact hPcAdv

/-- Full-state characterization for `cvta dst space src`: lane gets
`regs[dst] := gaddr` (from `evalCvta?`), all other fields preserved, pc advanced. -/
theorem stepInstr?_cvta_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId} {pc : PC}
    {lane : LaneId} {laneState : LaneState} {srcVal : Value} {gaddr : Value}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hIn : lane ∈ participants)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hSrc : evalRValue? st cta warp lane src = some srcVal)
    (hCvta : evalCvta? space srcVal = some gaddr)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .cvta dst space src } = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs.insert dst gaddr ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = (pc.1, pc.2 + 1) := by
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  cases hApply : applyToLaneIds? st cta warp participants
      (fun lane laneState =>
        (evalRValue? st cta warp lane src).bind fun value =>
          (evalCvta? space value).bind fun gaddr =>
            some (writeReg laneState dst gaddr)) with
  | none => simp [hApply] at hStep
  | some stMid =>
      rw [hApply] at hStep; simp at hStep
      have hPartNoDup : participants.Nodup := participatingRunnableLaneIds?_nodup hPart
      have hMid :=
        applyToLaneIds?_lane_in cta warp
          (fun lane laneState =>
            (evalRValue? st cta warp lane src).bind fun value =>
              (evalCvta? space value).bind fun gaddr =>
                some (writeReg laneState dst gaddr))
          participants hPartNoDup st stMid hWf hApply lane hIn laneState hLane
      obtain ⟨laneState_mid, hF, hGetMid⟩ := hMid
      simp [hSrc, hCvta] at hF
      have hPres : PreservesStatusPc
          (fun l ls =>
            (evalRValue? st cta warp l src).bind fun v =>
              (evalCvta? space v).bind fun g => some (writeReg ls dst g)) := by
        intro l ls ls' hPF
        simp [Option.bind_eq_some_iff] at hPF
        obtain ⟨_, _, _, _, hWrite⟩ := hPF
        rw [← hWrite]; unfold writeReg; exact ⟨rfl, rfl⟩
      have hMidPc : laneState_mid.pc = pc := by
        rw [← hF]; unfold writeReg; exact hLanePc
      have hRunIn := (participant_runnable_pc hPc hPart lane hIn).1
      obtain ⟨ls', hGet', hRegs, hPreds, hLocal, hStatus, hPcAdv⟩ :=
        step_lane_full_from_mid hWf hWarp hPc hRunIn hPres hApply hGetMid hMidPc hStep
      refine ⟨ls', hGet', ?_, ?_, ?_, ?_, ?_⟩
      · rw [hRegs, ← hF]; rfl
      · rw [hPreds, ← hF]; rfl
      · rw [hLocal, ← hF]; rfl
      · rw [hStatus, ← hF]; rfl
      · exact hPcAdv

/-! ## `store` infrastructure: writing to global memory

The store case is structurally different: instead of `applyToLaneIds?`
(which only modifies lane state), it threads a `forIn` loop of
`writeMem?` calls, each updating global byte memory. Lane state is
untouched by writes to global memory, so each participant's lane state is
preserved across the entire forIn loop. -/

/-- `writeMem?` to a global address preserves every lane's state. -/
theorem writeMem?_global_preserves_lane
    {st st' : State} {ty : ScalarTy} {addr : Addr} {value : Value}
    {cta : CTAId} {warp : WarpId} {lane : LaneId}
    (hSpace : addr.space = .global)
    (hWrite : Helpers.writeMem? st .global ty addr value = some st') :
    st'.getLane? cta warp lane = st.getLane? cta warp lane := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · cases hEnc : Helpers.encodeScalar? ty value with
    | none => rw [hEnc] at hWrite; simp at hWrite
    | some bytes =>
        rw [hEnc] at hWrite; simp at hWrite
        cases hGet : Helpers.getSpaceBaseMem? st addr with
        | none => rw [hGet] at hWrite; simp at hWrite
        | some mem =>
            rw [hGet] at hWrite; simp at hWrite
            -- For a global address, `setSpaceBaseMem?` updates only `st.global`.
            cases addr with
            | global off =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite
                rw [← hWrite]
                rfl
            | shared _ _ => simp [Addr.space] at hSpace
            | «local» _ _ _ _ => simp [Addr.space] at hSpace
            | param _ => simp [Addr.space] at hSpace
            | const _ => simp [Addr.space] at hSpace
            | generic _ _ =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite

/-- `readBytes?.loop` agrees on memories that agree on the read range. -/
private theorem readBytes?_loop_congr
    (m1 m2 : ByteMem) (offset width : Nat)
    (hEq : ∀ i, i < width → m1[offset + i]? = m2[offset + i]?)
    (i : Nat) (acc : List Byte) :
    Helpers.readBytes?.loop m1 offset width i acc =
    Helpers.readBytes?.loop m2 offset width i acc := by
  by_cases hlt : i < width
  · unfold Helpers.readBytes?.loop
    simp only [hlt, ↓reduceIte]
    rw [hEq i hlt]
    cases h : m2[offset + i]? with
    | none => simp
    | some b =>
        simp
        exact readBytes?_loop_congr m1 m2 offset width hEq (i + 1) (b :: acc)
  · unfold Helpers.readBytes?.loop
    simp only [hlt, ↓reduceIte]
termination_by width - i
decreasing_by simp_wf; omega

theorem readBytes?_congr (m1 m2 : ByteMem) (offset width : Nat)
    (hEq : ∀ i, i < width → m1[offset + i]? = m2[offset + i]?) :
    Helpers.readBytes? m1 offset width = Helpers.readBytes? m2 offset width := by
  unfold Helpers.readBytes?
  exact readBytes?_loop_congr m1 m2 offset width hEq 0 []

/-- The output of `encodeScalar?` has length equal to the type's
`byteWidth?`. -/
theorem encodeScalar?_length
    {ty : ScalarTy} {value : Value} {bs : List Byte} {w : Nat}
    (hEnc : Helpers.encodeScalar? ty value = some bs)
    (hWidth : Typing.byteWidth? ty = some w) :
    bs.length = w := by
  cases ty <;> cases value <;>
    simp [Helpers.encodeScalar?] at hEnc <;>
    simp [Typing.byteWidth?] at hWidth <;>
    (try subst hEnc) <;>
    (try subst hWidth) <;>
    simp [natToBytesLE_length]

/-- `writeMem?` at one global address doesn't change `readMem?` at a disjoint
global address. -/
theorem readMem?_writeMem?_global_disjoint
    {st st' : State} {tyW tyR : ScalarTy}
    {addrW addrR : Addr} {value : Value}
    {wW wR : Nat}
    (hSpaceW : addrW.space = .global)
    (hSpaceR : addrR.space = .global)
    (hWriteWidth : Typing.byteWidth? tyW = some wW)
    (_hReadWidth : Typing.byteWidth? tyR = some wR)
    (hWrite : Helpers.writeMem? st .global tyW addrW value = some st')
    (hDisjoint : addrW.offset + wW ≤ addrR.offset ∨
                 addrR.offset + wR ≤ addrW.offset) :
    Helpers.readMem? st' .global tyR addrR =
    Helpers.readMem? st .global tyR addrR := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · cases hEnc : Helpers.encodeScalar? tyW value with
    | none => rw [hEnc] at hWrite; simp at hWrite
    | some bytesW =>
        rw [hEnc] at hWrite; simp at hWrite
        cases hGet : Helpers.getSpaceBaseMem? st addrW with
        | none => rw [hGet] at hWrite; simp at hWrite
        | some memW =>
            rw [hGet] at hWrite; simp at hWrite
            cases addrW with
            | global offW =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite
                have hGlobalEq :
                    st'.global.bytes =
                    Helpers.writeBytes st.global.bytes offW bytesW := by
                  rw [← hWrite]
                  simp [Helpers.getSpaceBaseMem?] at hGet
                  rw [← hGet]
                  rfl
                have hLenBytes : bytesW.length = wW :=
                  encodeScalar?_length hEnc hWriteWidth
                -- Show readMem? agrees by showing the underlying readBytes? agrees.
                have hOtherFields : st'.const = st.const ∧ st'.param = st.param ∧
                    st'.ctas = st.ctas ∧ st'.kernelEnv = st.kernelEnv := by
                  rw [← hWrite]; exact ⟨rfl, rfl, rfl, rfl⟩
                unfold Helpers.readMem?
                cases addrR with
                | global offR =>
                    simp [Helpers.getSpaceBaseMem?]
                    split
                    · rfl
                    · rw [hGlobalEq, _hReadWidth]
                      simp [Addr.offset]
                      congr 1
                      apply readBytes?_congr
                      intro i hi
                      apply writeBytes_outside_range
                      rw [hLenBytes]
                      simp [Addr.offset] at hDisjoint
                      omega
                | shared _ _ => simp [Addr.space] at hSpaceR
                | «local» _ _ _ _ => simp [Addr.space] at hSpaceR
                | param cidx =>
                    -- readMem? on .global with addr=.param fails preconditions,
                    -- but more importantly, both sides should equal regardless.
                    rw [← hWrite]
                    simp [Helpers.getSpaceBaseMem?]
                | const _ =>
                    rw [← hWrite]
                    simp [Helpers.getSpaceBaseMem?]
                | generic _ _ =>
                    rw [← hWrite]
                    simp [Helpers.getSpaceBaseMem?]
            | shared _ _ => simp [Addr.space] at hSpaceW
            | «local» _ _ _ _ => simp [Addr.space] at hSpaceW
            | param _ => simp [Addr.space] at hSpaceW
            | const _ => simp [Addr.space] at hSpaceW
            | generic _ _ =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite

/-- `writeMem?` to a global address preserves the warp state (for any
warp), because lane state — which is what `getWarp?` returns — is
unchanged. -/
theorem writeMem?_global_preserves_warp
    {st st' : State} {ty : ScalarTy} {addr : Addr} {value : Value}
    {cta : CTAId} {warp : WarpId}
    (hSpace : addr.space = .global)
    (hWrite : Helpers.writeMem? st .global ty addr value = some st') :
    st'.getWarp? cta warp = st.getWarp? cta warp := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · cases hEnc : Helpers.encodeScalar? ty value with
    | none => rw [hEnc] at hWrite; simp at hWrite
    | some bytes =>
        rw [hEnc] at hWrite; simp at hWrite
        cases hGet : Helpers.getSpaceBaseMem? st addr with
        | none => rw [hGet] at hWrite; simp at hWrite
        | some mem =>
            rw [hGet] at hWrite; simp at hWrite
            cases addr with
            | global off =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite
                rw [← hWrite]
                rfl
            | shared _ _ => simp [Addr.space] at hSpace
            | «local» _ _ _ _ => simp [Addr.space] at hSpace
            | param _ => simp [Addr.space] at hSpace
            | const _ => simp [Addr.space] at hSpace
            | generic _ _ =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite

/-- `writeMem?` to a global address preserves `kernelEnv`. -/
theorem writeMem?_global_preserves_kernelEnv
    {st st' : State} {ty : ScalarTy} {addr : Addr} {value : Value}
    (hSpace : addr.space = .global)
    (hWrite : Helpers.writeMem? st .global ty addr value = some st') :
    st'.kernelEnv = st.kernelEnv := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · cases hEnc : Helpers.encodeScalar? ty value with
    | none => rw [hEnc] at hWrite; simp at hWrite
    | some bytes =>
        rw [hEnc] at hWrite; simp at hWrite
        cases hGet : Helpers.getSpaceBaseMem? st addr with
        | none => rw [hGet] at hWrite; simp at hWrite
        | some mem =>
            rw [hGet] at hWrite; simp at hWrite
            cases addr with
            | global off =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite
                rw [← hWrite]
            | shared _ _ => simp [Addr.space] at hSpace
            | «local» _ _ _ _ => simp [Addr.space] at hSpace
            | param _ => simp [Addr.space] at hSpace
            | const _ => simp [Addr.space] at hSpace
            | generic _ _ =>
                unfold Helpers.setSpaceBaseMem? at hWrite
                simp at hWrite

/-- `writeMem?` to a global address modifies only `global.bytes`. -/
theorem writeMem?_global_bytes_eq
    {st st' : State} {ty : ScalarTy} {off : Nat} {value : Value}
    {bytes : List Byte}
    (hEnc : Helpers.encodeScalar? ty value = some bytes)
    (hWrite : Helpers.writeMem? st .global ty (.global off) value = some st') :
    st'.global.bytes = Helpers.writeBytes st.global.bytes off bytes := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · rw [hEnc] at hWrite; simp at hWrite
    simp [Helpers.getSpaceBaseMem?] at hWrite
    unfold Helpers.setSpaceBaseMem? at hWrite
    simp at hWrite
    rw [← hWrite]
    simp [Addr.offset]

/-! ## Store forIn induction

The store case in `stepInstr?` threads a `forIn` loop of `writeMem?`
calls. We prove that under pairwise-disjoint address resolution and
stable per-lane addr/value computation, the final memory state holds
each participant's expected value at its address. -/

/-- A "lane resolution map" packages, for each participating lane, the
resolved address, the value it writes, and the encoded bytes. -/
private structure LaneResolution (st : State) (cta : CTAId) (warp : WarpId)
    (dst : TypedAddr) (value : RValue) (lane : LaneId) where
  addr : Addr
  val : Value
  bytes : List Byte
  hAddr : Helpers.resolveAddr? st cta warp lane dst = some addr
  hVal : Helpers.evalRValue? st cta warp lane value = some val
  hEnc : Helpers.encodeScalar? dst.ty val = some bytes
  hSpaceGlobal : addr.space = .global
  hAddrSpaceMatch : addr.space = dst.space

/-- Property: the addresses written by all participants are pairwise
disjoint (so writes don't shadow each other). -/
private def DisjointWrites
    (laneRes : LaneId → Option (Addr × List Byte))
    (lanes : List LaneId) : Prop :=
  ∀ (l₁ l₂ : LaneId), l₁ ∈ lanes → l₂ ∈ lanes → l₁ ≠ l₂ →
    ∀ a₁ bs₁ a₂ bs₂,
      laneRes l₁ = some (a₁, bs₁) → laneRes l₂ = some (a₂, bs₂) →
      a₁.offset + bs₁.length ≤ a₂.offset ∨ a₂.offset + bs₂.length ≤ a₁.offset

/-- `evalRValue?` depends on `st` only via `getLane?` and `kernelEnv.gridCtx`.
Two states agreeing on these compute the same RValue. -/
theorem evalRValue?_lane_state_invariant
    (rv : RValue) (st st' : State) (cta : CTAId) (warp : WarpId) (lane : LaneId)
    (hLane : st'.getLane? cta warp lane = st.getLane? cta warp lane)
    (hGrid : st'.kernelEnv.gridCtx = st.kernelEnv.gridCtx) :
    Helpers.evalRValue? st' cta warp lane rv =
    Helpers.evalRValue? st cta warp lane rv := by
  induction rv with
  | imm v => rw [Helpers.evalRValue?.eq_1, Helpers.evalRValue?.eq_1]
  | reg r =>
      rw [Helpers.evalRValue?.eq_2, Helpers.evalRValue?.eq_2, hLane]
  | pred p =>
      rw [Helpers.evalRValue?.eq_3, Helpers.evalRValue?.eq_3, hLane]
  | special s =>
      rw [Helpers.evalRValue?.eq_4, Helpers.evalRValue?.eq_4, hGrid]
  | unop op a ih =>
      rw [Helpers.evalRValue?.eq_5, Helpers.evalRValue?.eq_5, ih]
  | binop op a b iha ihb =>
      rw [Helpers.evalRValue?.eq_6, Helpers.evalRValue?.eq_6, iha, ihb]
  | triop op a b c iha ihb ihc =>
      rw [Helpers.evalRValue?.eq_7, Helpers.evalRValue?.eq_7, iha, ihb, ihc]

/-- `resolveAddr?` is stable under state changes that preserve lane state
and gridCtx. -/
theorem resolveAddr?_lane_state_invariant
    (ta : TypedAddr) (st st' : State) (cta : CTAId) (warp : WarpId) (lane : LaneId)
    (hLane : st'.getLane? cta warp lane = st.getLane? cta warp lane)
    (hGrid : st'.kernelEnv.gridCtx = st.kernelEnv.gridCtx) :
    Helpers.resolveAddr? st' cta warp lane ta =
    Helpers.resolveAddr? st cta warp lane ta := by
  unfold Helpers.resolveAddr?
  rw [evalRValue?_lane_state_invariant ta.addr st st' cta warp lane hLane hGrid]

/-- Byte-level lemma: folding `writeBytes` over a list of (offset,bytes)
pairs preserves a target read at a disjoint offset. -/
private theorem writeBytes_fold_preserves_disjoint
    (lanes : List LaneId)
    (laneOff : LaneId → Nat) (laneBytes : LaneId → List Byte)
    (wByte : Nat) (hLen : ∀ l, (laneBytes l).length = wByte)
    (target : Nat)
    (hAllDisjoint : ∀ l ∈ lanes,
        target + wByte ≤ laneOff l ∨ laneOff l + wByte ≤ target)
    (mem0 : ByteMem) :
    Helpers.readBytes?
        (lanes.foldl (fun m l => Helpers.writeBytes m (laneOff l) (laneBytes l)) mem0)
        target wByte
      = Helpers.readBytes? mem0 target wByte := by
  induction lanes generalizing mem0 with
  | nil => simp
  | cons a rest ih =>
      simp only [List.foldl_cons]
      have hDisjA := hAllDisjoint a List.mem_cons_self
      have hAllDisjointRest : ∀ l ∈ rest,
          target + wByte ≤ laneOff l ∨ laneOff l + wByte ≤ target :=
        fun l hl => hAllDisjoint l (List.mem_cons_of_mem _ hl)
      rw [ih hAllDisjointRest _]
      apply readBytes?_congr
      intro i hi
      apply writeBytes_outside_range
      rw [hLen]
      omega

/-- Byte-level lemma: after folding `writeBytes` of `(laneOff l, laneBytes l)`
over a Nodup list of lanes, reading at any lane's offset returns its bytes. -/
private theorem writeBytes_fold_target_value
    (lanes : List LaneId) (hNoDup : lanes.Nodup)
    (laneOff : LaneId → Nat) (laneBytes : LaneId → List Byte)
    (wByte : Nat) (hLen : ∀ l, (laneBytes l).length = wByte)
    (hDisj : ∀ l₁ l₂, l₁ ∈ lanes → l₂ ∈ lanes → l₁ ≠ l₂ →
        laneOff l₁ + wByte ≤ laneOff l₂ ∨ laneOff l₂ + wByte ≤ laneOff l₁)
    (mem0 : ByteMem) (j : LaneId) (hj : j ∈ lanes) :
    Helpers.readBytes?
        (lanes.foldl (fun m l => Helpers.writeBytes m (laneOff l) (laneBytes l)) mem0)
        (laneOff j) wByte
      = some (laneBytes j) := by
  induction lanes generalizing mem0 with
  | nil => exact absurd hj List.not_mem_nil
  | cons a rest ih =>
      simp only [List.foldl_cons]
      rcases List.mem_cons.1 hj with rfl | hjRest
      · -- j = a: after a's write, mem at laneOff a has laneBytes a. Subsequent writes are disjoint.
        rw [writeBytes_fold_preserves_disjoint rest laneOff laneBytes wByte hLen (laneOff j)]
        · rw [← hLen j]
          exact readBytes?_writeBytes_same mem0 (laneOff j) (laneBytes j)
        · intro l hl
          have hjNeL : j ≠ l := by
            intro heq; subst heq
            exact (List.nodup_cons.1 hNoDup).1 hl
          exact hDisj j l List.mem_cons_self (List.mem_cons_of_mem _ hl) hjNeL
      · -- j ∈ rest: apply IH after a's write.
        have hNoDupRest : rest.Nodup := (List.nodup_cons.1 hNoDup).2
        have hDisjRest : ∀ l₁ l₂, l₁ ∈ rest → l₂ ∈ rest → l₁ ≠ l₂ →
            laneOff l₁ + wByte ≤ laneOff l₂ ∨ laneOff l₂ + wByte ≤ laneOff l₁ :=
          fun l₁ l₂ h1 h2 hne =>
            hDisj l₁ l₂ (List.mem_cons_of_mem _ h1) (List.mem_cons_of_mem _ h2) hne
        exact ih hNoDupRest hDisjRest _ hjRest

/-- **The store forIn post-state characterization.**

For a forIn loop over participants that, for each lane, computes
`resolveAddr?`, `evalRValue?`, then `writeMem?`, the final state's
global bytes equal the foldl of `writeBytes` applied to each lane's
expected (offset, bytes) pair, and lane state / kernelEnv are
preserved.

This is the core induction. It is parameterized by a fixed anchor
state `st0` (the state at which the resolveAddr?/evalRValue? produce
the canonical answers). The inductive step uses lane-state stability
under `writeMem?` (which preserves `getLane?` and `kernelEnv`) to
transfer the resolveAddr?/evalRValue? answers across iterations. -/
private theorem forIn_store_post_state
    (cta : CTAId) (warp : WarpId)
    (dst : TypedAddr) (value : RValue)
    (laneAddrOff : LaneId → Nat) (laneBytes : LaneId → List Byte)
    (st0 : State)
    -- Each candidate lane's resolveAddr? is .global (laneAddrOff l)
    (hAddrSt0 : ∀ l, Helpers.resolveAddr? st0 cta warp l dst =
                       some (.global (laneAddrOff l)))
    -- Each candidate lane's evalRValue? is some v, and v's encoding matches laneBytes
    (hValSt0 : ∀ l, ∃ v, Helpers.evalRValue? st0 cta warp l value = some v ∧
                          Helpers.encodeScalar? dst.ty v = some (laneBytes l))
    (hSpace : dst.space = .global) :
    ∀ (lanes : List LaneId) (stIn stOut : State),
      (∀ l, stIn.getLane? cta warp l = st0.getLane? cta warp l) →
      stIn.kernelEnv = st0.kernelEnv →
      (forIn (m := Option) lanes stIn fun l acc =>
        (Helpers.resolveAddr? acc cta warp l dst).bind fun addr =>
          (Helpers.evalRValue? acc cta warp l value).bind fun v =>
            (Helpers.writeMem? acc dst.space dst.ty addr v).bind fun r =>
              some (ForInStep.yield r)) = some stOut →
      stOut.global.bytes =
        lanes.foldl (fun m l => Helpers.writeBytes m (laneAddrOff l) (laneBytes l))
          stIn.global.bytes ∧
      (∀ l, stOut.getLane? cta warp l = st0.getLane? cta warp l) ∧
      stOut.kernelEnv = st0.kernelEnv := by
  intro lanes
  induction lanes with
  | nil =>
      intro stIn stOut hLaneIn hKerIn hLoop
      simp [List.forIn_nil] at hLoop
      subst hLoop
      exact ⟨rfl, hLaneIn, hKerIn⟩
  | cons a rest ih =>
      intro stIn stOut hLaneIn hKerIn hLoop
      simp [List.forIn_cons] at hLoop
      -- Compute the body's effect for lane a.
      have hGridIn : stIn.kernelEnv.gridCtx = st0.kernelEnv.gridCtx := by rw [hKerIn]
      have hAddrA : Helpers.resolveAddr? stIn cta warp a dst =
                    some (.global (laneAddrOff a)) := by
        rw [resolveAddr?_lane_state_invariant dst st0 stIn cta warp a
            (hLaneIn a) hGridIn]
        exact hAddrSt0 a
      obtain ⟨vA, hValA_st0, hEncA⟩ := hValSt0 a
      have hValA : Helpers.evalRValue? stIn cta warp a value = some vA := by
        rw [evalRValue?_lane_state_invariant value st0 stIn cta warp a
            (hLaneIn a) hGridIn]
        exact hValA_st0
      rw [hAddrA] at hLoop
      simp at hLoop
      rw [hValA] at hLoop
      simp at hLoop
      cases hWriteA : Helpers.writeMem? stIn dst.space dst.ty
                        (.global (laneAddrOff a)) vA with
      | none => rw [hWriteA] at hLoop; simp at hLoop
      | some accAfter =>
          rw [hWriteA] at hLoop
          simp at hLoop
          have hWriteSpaceA : Helpers.writeMem? stIn .global dst.ty
                                (.global (laneAddrOff a)) vA = some accAfter := by
            rw [← hSpace]; exact hWriteA
          -- accAfter's lane state and kernelEnv are preserved.
          have hLaneAfter : ∀ l, accAfter.getLane? cta warp l = st0.getLane? cta warp l := by
            intro l
            rw [writeMem?_global_preserves_lane (by rfl) hWriteSpaceA]
            exact hLaneIn l
          have hKerAfter : accAfter.kernelEnv = st0.kernelEnv := by
            rw [writeMem?_global_preserves_kernelEnv (by rfl) hWriteSpaceA]
            exact hKerIn
          -- accAfter's global bytes equal writeBytes stIn.global.bytes (laneOff a) (laneBytes a).
          have hBytesAfter : accAfter.global.bytes =
              Helpers.writeBytes stIn.global.bytes (laneAddrOff a) (laneBytes a) :=
            writeMem?_global_bytes_eq hEncA hWriteSpaceA
          -- Apply IH on rest.
          obtain ⟨hBytesOut, hLaneOut, hKerOut⟩ :=
            ih accAfter stOut hLaneAfter hKerAfter hLoop
          refine ⟨?_, hLaneOut, hKerOut⟩
          rw [hBytesOut, hBytesAfter]
          simp [List.foldl_cons]

/-! ## User-facing `stepInstr?_store_lane_memory` -/

/-- **After a `store dst value` step, lane `j`'s bytes are at its
resolved address in global memory.**

Hypotheses:
- All participants resolve to `.global (laneAddrOff l)` and evaluate to a
  value whose encoding is `laneBytes l`.
- Addresses are pairwise disjoint (so writes don't shadow each other).

Conclusion:
- `readBytes? st'.global.bytes (laneAddrOff j) wByte = some (laneBytes j)`. -/
theorem stepInstr?_store_lane_memory
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {dst : TypedAddr} {value : RValue} {guard? : Option Guard}
    {warpState : WarpState} {participants : List LaneId}
    {laneAddrOff : LaneId → Nat} {laneBytes : LaneId → List Byte}
    {wByte : Nat} {j : LaneId}
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPart : participatingRunnableLaneIds? warpState guard? = some participants)
    (hjIn : j ∈ participants)
    (hNoDup : participants.Nodup)
    (hSpace : dst.space = .global)
    (hWidth : Typing.byteWidth? dst.ty = some wByte)
    (hLen : ∀ l, (laneBytes l).length = wByte)
    (hAddrSt : ∀ l, Helpers.resolveAddr? st cta warp l dst =
                      some (.global (laneAddrOff l)))
    (hValSt : ∀ l, ∃ v, Helpers.evalRValue? st cta warp l value = some v ∧
                          Helpers.encodeScalar? dst.ty v = some (laneBytes l))
    (hDisj : ∀ l₁ l₂, l₁ ∈ participants → l₂ ∈ participants → l₁ ≠ l₂ →
        laneAddrOff l₁ + wByte ≤ laneAddrOff l₂ ∨
        laneAddrOff l₂ + wByte ≤ laneAddrOff l₁)
    (hStep : stepInstr? st cta warp
              { guard? := guard?, instr := .store dst value } = some st') :
    Helpers.readBytes? st'.global.bytes (laneAddrOff j) wByte = some (laneBytes j) := by
  -- Unfold stepInstr? for the store case.
  unfold stepInstr? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPart] at hStep; simp at hStep
  -- Extract the inner forIn and the trailing advanceRunnablePcs?.
  cases hLoop : (forIn (m := Option) participants st fun lane r =>
                  (Helpers.resolveAddr? r cta warp lane dst).bind fun addr =>
                    (Helpers.evalRValue? r cta warp lane value).bind fun v =>
                      (Helpers.writeMem? r dst.space dst.ty addr v).bind fun r =>
                        some (ForInStep.yield r)) with
  | none => rw [hLoop] at hStep; simp at hStep
  | some sMid =>
      rw [hLoop] at hStep; simp at hStep
      -- Apply forIn_store_post_state with stIn = st0 = st.
      have hLaneSt : ∀ l, st.getLane? cta warp l = st.getLane? cta warp l := fun _ => rfl
      obtain ⟨hBytesMid, _, _⟩ :=
        forIn_store_post_state cta warp dst value laneAddrOff laneBytes st
          hAddrSt hValSt hSpace participants st sMid hLaneSt rfl hLoop
      -- advanceRunnablePcs? preserves global.bytes.
      have hAdvTop := advanceRunnablePcs?_preserves_top hStep
      rw [hAdvTop.1, hBytesMid]
      -- Apply writeBytes_fold_target_value.
      exact writeBytes_fold_target_value participants hNoDup laneAddrOff laneBytes
        wByte hLen hDisj st.global.bytes j hjIn

/-! ## Terminator step lemmas

Saxpy uses `.cbr` (conditional branch) and `.terminate` (return). The
`stepTerminator?` follows the same `lockstep + applyToLaneIds?` pattern as
`stepInstr?`, but the inner function may change a lane's `status` (for
terminate) or `pc` (for cbr). -/

/-- The "lanes" list used by `stepTerminator?` — runnable lanes at the
current pc. -/
private def termParticipants (warpState : WarpState) (pc : PC) : List LaneId :=
  (runnableLaneIds warpState).filter fun lane =>
    match warpState.getLane? lane with
    | some laneState => laneState.pc == pc
    | none => false

private theorem termParticipants_nodup (warpState : WarpState) (pc : PC) :
    (termParticipants warpState pc).Nodup := by
  have hLaneIdsNoDup : (laneIds : List LaneId).Nodup := by
    unfold laneIds; exact List.nodup_finRange 32
  have hRunNoDup : (runnableLaneIds warpState).Nodup :=
    List.Nodup.filter _ hLaneIdsNoDup
  unfold termParticipants
  exact List.Nodup.filter _ hRunNoDup

/-- For a runnable lane whose pc matches `currentRunnablePc?`, it appears in
the `stepTerminator?` participants list. -/
private theorem mem_termParticipants
    {warpState : WarpState} {pc : PC} {lane : LaneId} {laneState : LaneState}
    (hWarpLane : warpState.getLane? lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hRun : lane ∈ runnableLaneIds warpState) :
    lane ∈ termParticipants warpState pc := by
  unfold termParticipants
  apply List.mem_filter.mpr
  refine ⟨hRun, ?_⟩
  rw [hWarpLane]; simp [hLanePc]

/-- **`stepTerminator?` on `.terminate`**: every runnable lane at the
current pc has its status set to `.terminated`; other fields are
preserved. -/
theorem stepTerminator?_terminate_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC}
    {lane : LaneId} {laneState : LaneState}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hRun : lane ∈ runnableLaneIds warpState)
    (hStep : stepTerminator? st cta warp .terminate = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.pc = laneState.pc ∧
      laneState'.status = .terminated := by
  unfold stepTerminator? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPc] at hStep; simp at hStep
  -- hStep : applyToLaneIds? st cta warp termParticipants (... terminated ...) = some st'
  have hWarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hLane
    rw [hWarp] at hLane; simpa using hLane
  have hMem : lane ∈ termParticipants warpState pc :=
    mem_termParticipants hWarpLane hLanePc hRun
  -- Apply applyToLaneIds?_lane_in.
  have hMid := applyToLaneIds?_lane_in cta warp
    (fun _ ls => some { ls with status := .terminated })
    (termParticipants warpState pc)
    (termParticipants_nodup warpState pc)
    st st' hWf hStep lane hMem laneState hLane
  obtain ⟨ls', hF, hGet⟩ := hMid
  simp at hF
  subst hF
  exact ⟨{ laneState with status := .terminated }, hGet, rfl, rfl, rfl, rfl, rfl⟩

/-- **`stepTerminator?` on `.cbr` with uniform direction**: when all
participants evaluate the condition to a uniform boolean `b`, every lane's
pc is set to the corresponding label. -/
theorem stepTerminator?_cbr_lane_full
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {cond : RValue} {tLabel fLabel : BlockLabel}
    {warpState : WarpState} {pc : PC} {dest : PC}
    {lane : LaneId} {laneState : LaneState}
    (hWf : State.wf st)
    (hWarp : st.getWarp? cta warp = some warpState)
    (hLock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hLane : st.getLane? cta warp lane = some laneState)
    (hLanePc : laneState.pc = pc)
    (hRun : lane ∈ runnableLaneIds warpState)
    (hDest : uniformBranchDestination? st cta warp
              (termParticipants warpState pc) cond tLabel fLabel = some dest)
    (hStep : stepTerminator? st cta warp (.cbr cond tLabel fLabel) = some st') :
    ∃ laneState' : LaneState,
      st'.getLane? cta warp lane = some laneState' ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.status = laneState.status ∧
      laneState'.pc = dest := by
  unfold stepTerminator? at hStep
  rw [hWarp] at hStep; simp at hStep
  have hLockB := (lockstepRunnable_iff_bool warpState).1 hLock
  rw [hLockB] at hStep; simp at hStep
  rw [hPc] at hStep; simp at hStep
  -- hStep has the uniformBranchDestination?.bind over the filter form.
  -- Rewrite using `termParticipants` so we can apply hDest.
  change ((uniformBranchDestination? st cta warp (termParticipants warpState pc) cond tLabel fLabel).bind
    fun dest =>
    applyToLaneIds? st cta warp (termParticipants warpState pc) fun _ ls =>
      some { ls with pc := dest }) = some st' at hStep
  rw [hDest] at hStep; simp at hStep
  have hWarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hLane
    rw [hWarp] at hLane; simpa using hLane
  have hMem : lane ∈ termParticipants warpState pc :=
    mem_termParticipants hWarpLane hLanePc hRun
  have hMid := applyToLaneIds?_lane_in cta warp
    (fun _ ls => some { ls with pc := dest })
    (termParticipants warpState pc)
    (termParticipants_nodup warpState pc)
    st st' hWf hStep lane hMem laneState hLane
  obtain ⟨ls', hF, hGet⟩ := hMid
  simp at hF
  subst hF
  exact ⟨{ laneState with pc := dest }, hGet, rfl, rfl, rfl, rfl, rfl⟩

/-! ## `runN` chain machinery -/

@[simp] theorem runN_zero (st : State) :
    StepMachine.runN 0 st = st := rfl

theorem runN_succ_some {st st' : State} {n : Nat}
    (h : StepMachine.step? st = some st') :
    StepMachine.runN (n + 1) st = StepMachine.runN n st' := by
  show (match StepMachine.step? st with
        | some s => StepMachine.runN n s
        | none => st) = StepMachine.runN n st'
  rw [h]

theorem runN_succ_none {st : State} {n : Nat}
    (h : StepMachine.step? st = none) :
    StepMachine.runN (n + 1) st = st := by
  show (match StepMachine.step? st with
        | some s => StepMachine.runN n s
        | none => st) = st
  rw [h]

/-- Reduce `step?` to `stepInstr?` when at a body position and `stepInstr?` succeeds. -/
theorem step?_body_some
    {st st' : State} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr}
    {participants : List LaneId}
    (hwf : State.wf st)
    (hgetWarp : st.getWarp? 0 0 = some warpState)
    (hwfWS : WarpState.wf warpState)
    (hlock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hgi : block.body[pc.2]? = some gi)
    (hpart : participatingRunnableLaneIds? warpState gi.guard? = some participants)
    (hInstr : stepInstr? st 0 0 gi = some st') :
    StepMachine.step? st = some st' := by
  unfold StepMachine.step? StepMachine.stepAt?
  rw [StepMachine.currentInstrStep?_of_body
      hwf hgetWarp hwfWS hlock hPc hblock hgi hpart]
  rw [hInstr]

/-- Reduce `step?` to `stepTerminator?` when past the end of a block's body. -/
theorem step?_term
    {st : State} {warpState : WarpState} {pc : PC} {block : Block}
    (hwf : State.wf st)
    (hgetWarp : st.getWarp? 0 0 = some warpState)
    (hwfWS : WarpState.wf warpState)
    (hlock : lockstepRunnable warpState)
    (hPc : currentRunnablePc? warpState = some pc)
    (hblock : st.kernelEnv.blocks[pc.1]? = some block)
    (hbody : block.body[pc.2]? = none) :
    StepMachine.step? st = stepTerminator? st 0 0 block.term := by
  unfold StepMachine.step? StepMachine.stepAt?
  rw [StepMachine.currentInstrStep?_none_at_term
      hwf hgetWarp hwfWS hlock hPc hblock hbody]
  simp
  rw [StepMachine.currentTermStep?_of_term
      hwf hgetWarp hwfWS hlock hPc hblock hbody]

end CLean
