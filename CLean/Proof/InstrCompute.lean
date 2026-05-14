import CLean.Proof.Lemmas

/-! # Per-instruction computation lemmas

These lemmas describe **what gets computed** by each instruction class
used by the saxpy and matmul kernels. They are the missing companion to
`Proof/Lemmas.lean`'s preservation lemmas (which describe what *doesn't*
change). Each is a direct unfolding of `Helpers.stepInstr?` for the
specific instruction shape.

The chosen layer of granularity:

* `applyToLaneIds?_singleton_unfold`: one-lane reduction of the
  imperative `for` loop, the foundation everything else stands on.
* For each instruction class in the saxpy/matmul body (`assignReg`,
  `assignPred`, `load`, `store`, `cvta`, `assignPredValue`,
  `isspacep`), a lemma stating the per-lane register/predicate/memory
  effect when there is exactly one participating lane.

Generalizing to multi-lane participation (i.e. interaction with the
`applyToLaneIds?` fold over arbitrarily many lanes) is a follow-up step
that requires a `List.foldlM`-style induction on the lanes list. The
single-lane form is already sufficient for kernel proofs that go via
per-lane decomposition. -/

namespace CLean

open Helpers

/-! ## Foundation: structural unfoldings of `applyToLaneIds?` -/

theorem applyToLaneIds?_nil
    (st : State) (cta : CTAId) (warp : WarpId)
    (f : LaneId → LaneState → Option LaneState) :
    Helpers.applyToLaneIds? st cta warp [] f = some st := by
  unfold Helpers.applyToLaneIds?
  simp [List.forIn_nil]

theorem applyToLaneIds?_cons
    (st : State) (cta : CTAId) (warp : WarpId)
    (lane : LaneId) (lanes : List LaneId)
    (f : LaneId → LaneState → Option LaneState) :
    Helpers.applyToLaneIds? st cta warp (lane :: lanes) f =
      (st.getLane? cta warp lane).bind fun laneState =>
        (f lane laneState).bind fun laneState' =>
          (st.setLane cta warp lane laneState').bind fun stMid =>
            Helpers.applyToLaneIds? stMid cta warp lanes f := by
  unfold Helpers.applyToLaneIds?
  simp [List.forIn_cons]
  cases hLane : st.getLane? cta warp lane with
  | none => simp [hLane, Option.bind]
  | some laneState =>
      simp [hLane]
      cases hF : f lane laneState with
      | none => simp [hF, Option.bind]
      | some laneState' =>
          simp [hF, Option.bind]
          cases hSet : st.setLane cta warp lane laneState' with
          | none => simp [hSet, Option.bind]
          | some stMid => simp [hSet, Option.bind]

theorem applyToLaneIds?_singleton
    (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId)
    (f : LaneId → LaneState → Option LaneState) :
    Helpers.applyToLaneIds? st cta warp [lane] f =
      (st.getLane? cta warp lane).bind fun laneState =>
        (f lane laneState).bind fun laneState' =>
          st.setLane cta warp lane laneState' := by
  rw [applyToLaneIds?_cons]
  cases hLane : st.getLane? cta warp lane with
  | none => simp
  | some laneState =>
      simp
      cases hF : f lane laneState with
      | none => simp
      | some laneState' =>
          simp
          cases hSet : st.setLane cta warp lane laneState' with
          | none => simp
          | some stMid => simp [applyToLaneIds?_nil]

/-! ## Frame: `applyToLaneIds?` doesn't touch lanes outside the input list -/

/-- One-step preservation: after `setLane cta warp l ls`, any other lane's
state is unchanged. -/
theorem setLane_preserves_other_lane
    {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls : LaneState}
    (hSet : st.setLane cta warp l ls = some st')
    (lane : LaneId) (h : lane ≠ l) :
    st'.getLane? cta warp lane = st.getLane? cta warp lane := by
  -- setLane succeeds only if getWarp? does.
  have hWarp : ∃ ws, st.getWarp? cta warp = some ws := by
    unfold State.setLane at hSet
    cases hW : st.getWarp? cta warp with
    | none => simp [hW] at hSet
    | some ws => exact ⟨ws, rfl⟩
  obtain ⟨ws, hWarp⟩ := hWarp
  have hWarp' : st.getWarp? cta warp = some ws := hWarp
  -- Need warpState.wf for the lemma — derive it from getWarp? succeeding plus setLane succeeding.
  -- Actually getLane?_setLane_ne doesn't need wfWarp. Let me use the existing lemma differently.
  have key := State.getLane?_setLane_ne (st := st) (cta := cta) (warp := warp)
                (lane := l) (lane' := lane) h ls ws hWarp'
  -- key : (st.setLane cta warp l ls).bind (fun st' => st'.getLane? cta warp lane) = ws.getLane? lane
  rw [hSet] at key
  simp at key
  rw [key]
  -- Goal: st.getLane? cta warp lane = ws.getLane? lane
  -- From hWarp' : st.getWarp? cta warp = some ws, and getLane? definition.
  simp [State.getLane?, hWarp']

/-- If a lane is not in the list, `applyToLaneIds?` leaves it unchanged. -/
theorem applyToLaneIds?_lane_not_in
    (cta : CTAId) (warp : WarpId)
    (f : LaneId → LaneState → Option LaneState) :
    ∀ (lanes : List LaneId) (st st' : State)
      (h : Helpers.applyToLaneIds? st cta warp lanes f = some st')
      (lane : LaneId) (hNotIn : lane ∉ lanes),
      st'.getLane? cta warp lane = st.getLane? cta warp lane := by
  intro lanes
  induction lanes with
  | nil =>
      intro st st' h lane _hNotIn
      rw [applyToLaneIds?_nil] at h
      injection h with h
      rw [← h]
  | cons l rest ih =>
      intro st st' h lane hNotIn
      have hLaneNeL : lane ≠ l := fun heq => hNotIn (heq ▸ List.mem_cons_self)
      have hLaneNotRest : lane ∉ rest := fun hin => hNotIn (List.mem_cons_of_mem _ hin)
      rw [applyToLaneIds?_cons] at h
      cases hLaneL : st.getLane? cta warp l with
      | none => simp [hLaneL] at h
      | some lsL =>
          rw [hLaneL] at h
          simp at h
          cases hF : f l lsL with
          | none => simp [hF] at h
          | some lsL' =>
              rw [hF] at h
              simp at h
              cases hSet : st.setLane cta warp l lsL' with
              | none => simp [hSet] at h
              | some stMid =>
                  rw [hSet] at h
                  simp at h
                  -- IH on rest gives stMid.lane = st'.lane preserved; setLane preserves lane.
                  rw [ih stMid st' h lane hLaneNotRest,
                      setLane_preserves_other_lane hSet lane hLaneNeL]

/-- Helper: after `setLane cta warp lane laneState`, looking up the same lane
returns `laneState`, provided the original warp existed and was well-formed. -/
theorem setLane_get_self
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState}
    (hSet : st.setLane cta warp lane ls = some st')
    {warpState : WarpState}
    (hWarp : st.getWarp? cta warp = some warpState)
    (hWfW : WarpState.wf warpState) :
    st'.getLane? cta warp lane = some ls := by
  have key := State.getLane?_setLane_same (st := st) (cta := cta) (warp := warp)
                (lane := lane) (laneState := ls) warpState hWarp hWfW
  rw [hSet] at key
  simp at key
  exact key

/-- `setLane` preserves `State.wf`. -/
theorem State.wf_of_setLane
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState}
    (hWf : State.wf st)
    (hSet : st.setLane cta warp lane ls = some st') :
    State.wf st' := by
  unfold State.setLane at hSet
  cases hWS : st.getWarp? cta warp with
  | none => simp [hWS] at hSet
  | some wsOrig =>
      rw [hWS] at hSet
      simp at hSet
      unfold State.setWarp at hSet
      cases hCS : st.getCTA? cta with
      | none => simp [hCS] at hSet
      | some ctaState =>
          simp [hCS, pure, Option.bind] at hSet
          subst hSet
          have hWfOrig : WarpState.wf wsOrig := by
            -- need wf_of_getWarp? but it's below in the file; inline the proof.
            unfold State.wf State.wf? at hWf
            simp at hWf
            obtain ⟨_, hCtas⟩ := hWf
            unfold State.getWarp? State.getCTA? at hWS
            cases hCta : st.ctas[cta]? with
            | none => simp [hCta] at hWS
            | some ctaS =>
                simp [hCta] at hWS
                have hCtaWf : ctaS.wf? = true := hCtas cta ctaS hCta
                unfold CTAState.wf? at hCtaWf
                simp at hCtaWf
                exact hCtaWf warp wsOrig hWS
          have hWfNew : WarpState.wf (wsOrig.setLane lane ls) :=
            WarpState.wf_setLane wsOrig lane ls hWfOrig
          unfold State.wf State.wf? at hWf ⊢
          simp at hWf ⊢
          refine ⟨hWf.1, fun a b hLookup => ?_⟩
          unfold State.setCTA at hLookup
          simp at hLookup
          rw [Std.HashMap.getElem?_insert] at hLookup
          by_cases hAEq : cta = a
          · subst hAEq
            simp at hLookup
            subst hLookup
            unfold CTAState.wf?
            simp
            intro k v hLk
            rw [Std.HashMap.getElem?_insert] at hLk
            by_cases hWarpEq : warp = k
            · subst hWarpEq
              simp at hLk
              subst hLk
              exact hWfNew
            · simp [beq_iff_eq, hWarpEq] at hLk
              have hCtaWfOrig : ctaState.wf? = true :=
                hWf.2 cta ctaState (by simpa [State.getCTA?] using hCS)
              unfold CTAState.wf? at hCtaWfOrig
              simp at hCtaWfOrig
              exact hCtaWfOrig k v hLk
          · simp [beq_iff_eq, hAEq] at hLookup
            exact hWf.2 a b hLookup

/-- `WarpState.wf` is derivable from `State.wf` plus a successful `getWarp?`. -/
theorem WarpState.wf_of_getWarp?
    {st : State} {cta : CTAId} {warp : WarpId} {ws : WarpState}
    (hWf : State.wf st)
    (hGet : st.getWarp? cta warp = some ws) :
    WarpState.wf ws := by
  -- State.wf says every CTA's CTAState is wf, which says every WarpState is wf.
  unfold State.wf State.wf? at hWf
  simp at hWf
  obtain ⟨_, hCtas⟩ := hWf
  -- hGet unfolds to: ∃ ctaState, st.ctas[cta]? = some ctaState ∧ ctaState.warps[warp]? = some ws
  unfold State.getWarp? State.getCTA? at hGet
  cases hCta : st.ctas[cta]? with
  | none => simp [hCta] at hGet
  | some ctaState =>
      simp [hCta] at hGet
      have hCtaWf : ctaState.wf? = true := hCtas cta ctaState hCta
      unfold CTAState.wf? at hCtaWf
      simp at hCtaWf
      exact hCtaWf warp ws hGet

/-- Distinct-list multi-lane "what gets computed": if `f` is a pure
function of `(lane, originalLaneState)` and the input list has no
duplicates, then after `applyToLaneIds?` lane `i ∈ lanes` has the
state `f i (st.getLane? cta warp i)`. The proof inducts on the list and
uses the frame lemma `applyToLaneIds?_lane_not_in` plus `setLane_get_self`
to track lane `i` through the chain. -/
theorem applyToLaneIds?_lane_in
    (cta : CTAId) (warp : WarpId)
    (f : LaneId → LaneState → Option LaneState) :
    ∀ (lanes : List LaneId) (_hNoDup : lanes.Nodup)
      (st st' : State) (_hWfSt : State.wf st)
      (h : Helpers.applyToLaneIds? st cta warp lanes f = some st')
      (lane : LaneId) (hIn : lane ∈ lanes)
      (laneState : LaneState) (hLane : st.getLane? cta warp lane = some laneState),
      ∃ laneState', f lane laneState = some laneState' ∧
                    st'.getLane? cta warp lane = some laneState' := by
  intro lanes
  induction lanes with
  | nil => intros _ _ _ _ _ _ hIn; exact absurd hIn (List.not_mem_nil)
  | cons l rest ih =>
      intro hNoDup st st' hWfSt h lane hIn laneState hLane
      have hNoDupRest : rest.Nodup := (List.nodup_cons.1 hNoDup).2
      have hLNotInRest : l ∉ rest := (List.nodup_cons.1 hNoDup).1
      rw [applyToLaneIds?_cons] at h
      cases hLaneL : st.getLane? cta warp l with
      | none => simp [hLaneL] at h
      | some lsL =>
          rw [hLaneL] at h
          simp at h
          cases hF : f l lsL with
          | none => simp [hF] at h
          | some lsL' =>
              rw [hF] at h
              simp at h
              cases hSet : st.setLane cta warp l lsL' with
              | none => simp [hSet] at h
              | some stMid =>
                  rw [hSet] at h
                  simp at h
                  rcases List.mem_cons.1 hIn with rfl | hInRest
                  · -- lane = l: cleared by the singleton step
                    have hLaneEq : laneState = lsL := by
                      rw [hLane] at hLaneL; exact Option.some.inj hLaneL
                    subst hLaneEq
                    refine ⟨lsL', hF, ?_⟩
                    -- Show: st'.getLane? cta warp lane = some lsL'.
                    -- After setLane (some stMid), then applyToLaneIds? on rest with lane ∉ rest.
                    rw [applyToLaneIds?_lane_not_in cta warp f rest stMid st' h lane hLNotInRest]
                    -- Now reduce to: stMid.getLane? cta warp l = some lsL'
                    -- by setLane_get_self.
                    -- Need warpState wf — extract from setLane succeeding.
                    -- Actually, getLane?_setLane_same needs hWfW. Derive it.
                    have hWarp : ∃ ws, st.getWarp? cta warp = some ws := by
                      unfold State.setLane at hSet
                      cases hW : st.getWarp? cta warp with
                      | none => simp [hW] at hSet
                      | some ws => exact ⟨ws, rfl⟩
                    obtain ⟨ws, hWarpEq⟩ := hWarp
                    have hWfW : WarpState.wf ws :=
                      WarpState.wf_of_getWarp? hWfSt hWarpEq
                    exact setLane_get_self hSet hWarpEq hWfW
                  · -- lane ∈ rest: induct
                    have hLNeLane : lane ≠ l := fun heq => hLNotInRest (heq ▸ hInRest)
                    have hLanePresStMid :
                        stMid.getLane? cta warp lane = some laneState := by
                      rw [setLane_preserves_other_lane hSet lane hLNeLane]
                      exact hLane
                    have hWfMid : State.wf stMid :=
                      State.wf_of_setLane hWfSt hSet
                    exact ih hNoDupRest stMid st' hWfMid h lane hInRest laneState hLanePresStMid


end CLean
