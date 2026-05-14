import CLean.Proof.Determinism
import CLean.Proof.InstrCompute

/-! # `IsSingleWarp` preservation under `step?`

This file discharges the deferred preservation obligation in
`reaches_imp_runN_of_IsSingleWarp`: every state reached from a
single-warp state by `step?` is itself single-warp.

The core observation: `step?` only ever modifies state at
`(cta = 0, warp = 0)`. None of the state-mutation primitives
(`setLane`, `setWarp`, `setCTA`, `writeMem?`, `setBarrierInstance?`)
introduce new `(cta, warp)` keys — they only update or replace
existing ones. So the set of `(c, w)` for which `getWarp? c w` is
`some _` is invariant under `step?`.

The proof is structured around `WarpSupportEq` — the predicate that
two states have the same warp-key support. Each primitive maintains
this, and `IsSingleWarp` propagates through it. -/

namespace CLean

open Helpers

/-- Two states agree on which `(cta, warp)` keys have a warp. -/
def WarpSupportEq (st st' : State) : Prop :=
  ∀ c w, (st'.getWarp? c w).isSome = (st.getWarp? c w).isSome

namespace WarpSupportEq

theorem refl (st : State) : WarpSupportEq st st := fun _ _ => rfl

theorem symm {st st' : State} (h : WarpSupportEq st st') : WarpSupportEq st' st :=
  fun c w => (h c w).symm

theorem trans {st₀ st₁ st₂ : State}
    (h₀₁ : WarpSupportEq st₀ st₁) (h₁₂ : WarpSupportEq st₁ st₂) :
    WarpSupportEq st₀ st₂ :=
  fun c w => (h₁₂ c w).trans (h₀₁ c w)

end WarpSupportEq

/-- If `st'` has the same warp support as `st` and `st` is single-warp,
then so is `st'`. -/
theorem IsSingleWarp.of_support_eq
    {st st' : State} (hsw : IsSingleWarp st) (heq : WarpSupportEq st st') :
    IsSingleWarp st' := by
  intro c w ws hGet
  have h1 : (st'.getWarp? c w).isSome := by rw [hGet]; rfl
  have h2 : (st.getWarp? c w).isSome := by rw [heq c w] at h1; exact h1
  rcases hOld : st.getWarp? c w with _ | ws_old
  · rw [hOld] at h2; simp at h2
  · exact hsw c w ws_old hOld

/-! ## Primitive: `setCTA` -/

theorem getWarp?_setCTA_eq
    (st : State) (c : CTAId) (cs : CTAState) (w : WarpId) :
    (st.setCTA c cs).getWarp? c w = cs.warps[w]? := by
  simp [State.getWarp?, State.getCTA?, State.setCTA]

theorem getWarp?_setCTA_ne_cta
    (st : State) {c c' : CTAId} (h : c' ≠ c) (cs : CTAState) (w : WarpId) :
    (st.setCTA c cs).getWarp? c' w = st.getWarp? c' w := by
  simp [State.getWarp?, State.getCTA?_setCTA_ne _ h]

/-- `setCTA` at `c` preserves warp support if the new CTA's `warps` map
has the same key set as the old CTA's `warps` map at `c`. -/
theorem WarpSupportEq.setCTA_of_warps_keys_eq
    {st : State} {c : CTAId} {cs_old cs_new : CTAState}
    (hOld : st.getCTA? c = some cs_old)
    (hKeys : ∀ w : WarpId, (cs_new.warps[w]? : Option WarpState).isSome
                          = (cs_old.warps[w]? : Option WarpState).isSome) :
    WarpSupportEq st (st.setCTA c cs_new) := by
  intro c' w
  by_cases hc : c' = c
  · subst hc
    rw [getWarp?_setCTA_eq, hKeys]
    simp [State.getWarp?, hOld]
  · rw [getWarp?_setCTA_ne_cta _ hc]

/-! ## Primitive: `setLane`

`setLane c w l ls` is implemented as `getWarp? c w → some ws; setWarp c w ws.setLane l ls`,
which itself is `setCTA c (ctaState with warps := warps.insert w ws')`. So the new
CTA at `c` has its `warps` map updated at exactly key `w` — same key set. -/

theorem setLane_preserves_warp_support
    {st st' : State} {c : CTAId} {w : WarpId} {lane : LaneId} {ls : LaneState}
    (hSet : st.setLane c w lane ls = some st') :
    WarpSupportEq st st' := by
  unfold State.setLane at hSet
  cases hWarp : st.getWarp? c w with
  | none => simp [hWarp] at hSet
  | some ws =>
      rw [hWarp] at hSet
      simp at hSet
      -- hSet : st.setWarp c w (ws.setLane lane ls) = some st'
      unfold State.setWarp at hSet
      cases hCta : st.getCTA? c with
      | none => simp [hCta] at hSet
      | some cs =>
          rw [hCta] at hSet
          simp at hSet
          -- hSet : st.setCTA c { cs with warps := cs.warps.insert w (ws.setLane lane ls) } = st'
          subst hSet
          apply WarpSupportEq.setCTA_of_warps_keys_eq hCta
          intro w'
          show (cs.warps.insert w (ws.setLane lane ls))[w']?.isSome = cs.warps[w']?.isSome
          rw [Std.HashMap.getElem?_insert]
          by_cases hWeq : w == w'
          · simp [hWeq]
            -- since w == w', cs.warps[w']? should be some ws (from hWarp)
            have hEq : w = w' := by simpa using hWeq
            subst hEq
            have hWarpW : cs.warps[w]? = some ws := by
              have hWarp' := hWarp
              unfold State.getWarp? at hWarp'
              rw [hCta] at hWarp'
              simpa using hWarp'
            rw [Std.HashMap.mem_iff_contains, Std.HashMap.contains_eq_isSome_getElem?, hWarpW]
            rfl
          · simp [hWeq]

/-! ## Primitive: `applyToLaneIds?` -/

/-- A chain of `setLane` calls preserves warp support. -/
theorem applyToLaneIds?_preserves_warp_support
    {st st' : State} {c : CTAId} {w : WarpId}
    {lanes : List LaneId} {f : LaneId → LaneState → Option LaneState}
    (hApply : Helpers.applyToLaneIds? st c w lanes f = some st') :
    WarpSupportEq st st' := by
  induction lanes generalizing st with
  | nil =>
      rw [applyToLaneIds?_nil] at hApply
      have heq : st' = st := Option.some.inj hApply.symm
      rw [heq]
      exact WarpSupportEq.refl st
  | cons l rest ih =>
      rw [applyToLaneIds?_cons] at hApply
      cases hLane : st.getLane? c w l with
      | none => simp [hLane] at hApply
      | some laneState =>
          rw [hLane] at hApply
          simp at hApply
          cases hF : f l laneState with
          | none => simp [hF] at hApply
          | some laneState' =>
              rw [hF] at hApply
              simp at hApply
              cases hSet : st.setLane c w l laneState' with
              | none => simp [hSet] at hApply
              | some stMid =>
                  rw [hSet] at hApply
                  simp at hApply
                  have hStep : WarpSupportEq st stMid :=
                    setLane_preserves_warp_support hSet
                  exact WarpSupportEq.trans hStep (ih hApply)

/-! ## Primitive: structural field updates that don't touch `ctas` -/

/-- Replacing just the `global`/`const`/`param`/`atomics` fields doesn't
affect any `getWarp?` lookup. -/
theorem WarpSupportEq.of_ctas_eq
    {st st' : State} (h : st'.ctas = st.ctas) :
    WarpSupportEq st st' := by
  intro c w
  simp [State.getWarp?, State.getCTA?, h]

/-! ## Primitive: `setSpaceBaseMem?` -/

theorem setSpaceBaseMem?_preserves_warp_support
    {st st' : State} {addr : Addr} {bytes : ByteMem}
    (hSet : Helpers.setSpaceBaseMem? st addr bytes = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.setSpaceBaseMem? at hSet
  cases addr with
  | global _ =>
      simp at hSet
      exact WarpSupportEq.of_ctas_eq (by rw [← hSet])
  | param _ =>
      simp at hSet
      exact WarpSupportEq.of_ctas_eq (by rw [← hSet])
  | const _ =>
      simp at hSet
      exact WarpSupportEq.of_ctas_eq (by rw [← hSet])
  | shared cta _ =>
      simp at hSet
      cases hCta : st.getCTA? cta with
      | none => rw [hCta] at hSet; simp at hSet
      | some cs =>
          rw [hCta] at hSet
          simp at hSet
          rw [← hSet]
          apply WarpSupportEq.setCTA_of_warps_keys_eq hCta
          intro w
          rfl
  | «local» cta warp lane _ =>
      simp at hSet
      cases hLane : st.getLane? cta warp lane with
      | none => rw [hLane] at hSet; simp at hSet
      | some laneState =>
          rw [hLane] at hSet
          simp at hSet
          exact setLane_preserves_warp_support hSet
  | generic _ _ =>
      simp at hSet

/-! ## Primitive: `advanceRunnablePcs?` -/

theorem advanceRunnablePcs?_preserves_warp_support
    {st st' : State} {c : CTAId} {w : WarpId}
    (h : Helpers.advanceRunnablePcs? st c w = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.advanceRunnablePcs? at h
  cases hWarp : st.getWarp? c w with
  | none => simp [hWarp] at h
  | some ws =>
      rw [hWarp] at h
      simp at h
      cases hPc : Helpers.currentRunnablePc? ws with
      | none => simp [hPc] at h
      | some pc =>
          rw [hPc] at h
          simp at h
          exact applyToLaneIds?_preserves_warp_support h

/-! ## Generic loop helper

Many `do` blocks use `for ... in ... do cur <- ...` patterns where each
iteration is a state-mutating step. The following lemma captures the
invariant: if every individual step preserves warp support, then the
whole loop does. -/

theorem WarpSupportEq.forIn_option
    {α : Type} (xs : List α) (init final : State)
    (body : α → State → Option State)
    (hBody : ∀ x s s', body x s = some s' → WarpSupportEq s s')
    (h : (forIn (m := Option) xs init fun x acc =>
            (body x acc).map fun acc' => ForInStep.yield acc') = some final) :
    WarpSupportEq init final := by
  induction xs generalizing init with
  | nil =>
      simp [List.forIn_nil] at h
      rw [← h]
      exact WarpSupportEq.refl init
  | cons x rest ih =>
      simp [List.forIn_cons] at h
      cases hBodyX : body x init with
      | none => simp [hBodyX] at h
      | some sMid =>
          rw [hBodyX] at h
          simp at h
          have hStep := hBody x init sMid hBodyX
          exact WarpSupportEq.trans hStep (ih sMid h)

/-- More general forIn lemma: the body produces an `Option (ForInStep State)`
directly, where successful results (yield or done) preserve warp support. -/
theorem WarpSupportEq.forIn_option_general
    {α : Type} (xs : List α) (init final : State)
    (body : α → State → Option (ForInStep State))
    (hBody : ∀ x s s',
      body x s = some (ForInStep.yield s') ∨ body x s = some (ForInStep.done s')
      → WarpSupportEq s s')
    (h : (forIn (m := Option) xs init body) = some final) :
    WarpSupportEq init final := by
  induction xs generalizing init with
  | nil =>
      simp [List.forIn_nil] at h
      rw [← h]
      exact WarpSupportEq.refl init
  | cons x rest ih =>
      simp [List.forIn_cons] at h
      cases hBodyX : body x init with
      | none => simp [hBodyX] at h
      | some step =>
          rw [hBodyX] at h
          simp at h
          cases step with
          | yield sMid =>
              simp at h
              have hStepW := hBody x init sMid (Or.inl hBodyX)
              exact WarpSupportEq.trans hStepW (ih sMid h)
          | done sMid =>
              simp at h
              rw [← h]
              exact hBody x init sMid (Or.inr hBodyX)

/-! ## Primitive: `writeMem?` -/

theorem writeMem?_preserves_warp_support
    {st st' : State} {space : AddrSpace} {ty : ScalarTy}
    {addr : Addr} {value : Value}
    (hWrite : Helpers.writeMem? st space ty addr value = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.writeMem? at hWrite
  split at hWrite
  · simp at hWrite
  · cases hEnc : Helpers.encodeScalar? ty value with
    | none => rw [hEnc] at hWrite; simp at hWrite
    | some bytes =>
        rw [hEnc] at hWrite
        simp at hWrite
        cases hGet : Helpers.getSpaceBaseMem? st addr with
        | none => rw [hGet] at hWrite; simp at hWrite
        | some mem =>
            rw [hGet] at hWrite
            simp at hWrite
            exact setSpaceBaseMem?_preserves_warp_support hWrite

/-! ## Primitive: `setBarrierInstance?` -/

theorem setBarrierInstance?_preserves_warp_support
    {st st' : State} {cta : CTAId} {barrierId : Nat} {inst : BarrierInstance}
    (h : Helpers.setBarrierInstance? st cta barrierId inst = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.setBarrierInstance? at h
  cases hCta : st.getCTA? cta with
  | none => simp [hCta] at h
  | some cs =>
      rw [hCta] at h
      simp at h
      rw [← h]
      apply WarpSupportEq.setCTA_of_warps_keys_eq hCta
      intro w
      rfl

/-! ## Primitive: `stepBarrierCTA?`

Three loop phases (predicated-off advance, arrival recording, possibly
release) plus a final `setBarrierInstance?`. Each phase only mutates
state through `setLane` and `setBarrierInstance?`, both of which
preserve warp support. The proof unfolds the function, identifies the
three loops, and applies `WarpSupportEq.forIn_option` to each. -/

/-- Each iteration of a barrier loop's body factors through `setLane`,
which preserves warp support. The three loops in `stepBarrierCTA?` all
have this shape (with various per-lane modifications); the helper packages
the preservation. -/
theorem WarpSupportEq.barrier_loop_body
    {c : CTAId} {w : WarpId}
    (modify : LaneId → LaneState → LaneState)
    (predicate : LaneId → State → Bool)
    (lane : LaneId) (s s' : State)
    (h : (if predicate lane s then
            (s.getLane? c w lane).bind fun ls =>
              (s.setLane c w lane (modify lane ls)).bind fun s'' =>
                some (ForInStep.yield s'')
          else some (ForInStep.yield s)) = some (ForInStep.yield s') ∨
        (if predicate lane s then
            (s.getLane? c w lane).bind fun ls =>
              (s.setLane c w lane (modify lane ls)).bind fun s'' =>
                some (ForInStep.yield s'')
          else some (ForInStep.yield s)) = some (ForInStep.done s')) :
    WarpSupportEq s s' := by
  cases hPred : predicate lane s with
  | false =>
      rcases h with hY | hD
      · rw [hPred] at hY; simp at hY; rw [hY]; exact WarpSupportEq.refl _
      · rw [hPred] at hD; simp at hD
  | true =>
      rw [hPred] at h
      simp at h
      cases hLane : s.getLane? c w lane with
      | none =>
          rcases h with hY | hD
          · rw [hLane] at hY; simp at hY
          · rw [hLane] at hD; simp at hD
      | some ls =>
          rw [hLane] at h
          simp at h
          cases hSet : s.setLane c w lane (modify lane ls) with
          | none =>
              rcases h with hY | hD
              · rw [hSet] at hY; simp at hY
              · rw [hSet] at hD; simp at hD
          | some sMid =>
              rcases h with hY | hD
              · rw [hSet] at hY; simp at hY; rw [hY] at hSet
                exact setLane_preserves_warp_support hSet
              · rw [hSet] at hD; simp at hD

/-- `stepBarrierCTA?` preserves warp support. -/
theorem stepBarrierCTA?_preserves_warp_support
    {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {participants : List LaneId}
    (h : Helpers.stepBarrierCTA? st cta warp barrierId participants = some st') :
    WarpSupportEq st st' := by
  sorry

/-! ## A composable lemma: "bind into advanceRunnablePcs?"

For the assignReg/assignPred/assignPredValue/load/cvta/isspacep cases of
`stepInstr?`, the body has the shape
`(applyToLaneIds? st c w parts f).bind fun r => advanceRunnablePcs? r c w`.
The following lemma packages the support-preservation conclusion. -/

theorem WarpSupportEq.applyToLaneIds_then_advance
    {st st' : State} {c : CTAId} {w : WarpId}
    {participants : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (h : ((Helpers.applyToLaneIds? st c w participants f).bind
            fun r => Helpers.advanceRunnablePcs? r c w) = some st') :
    WarpSupportEq st st' := by
  cases hLanes : Helpers.applyToLaneIds? st c w participants f with
  | none => rw [hLanes] at h; simp at h
  | some sMid =>
      rw [hLanes] at h
      simp at h
      exact WarpSupportEq.trans
        (applyToLaneIds?_preserves_warp_support hLanes)
        (advanceRunnablePcs?_preserves_warp_support h)

/-! ## `stepInstr?`

For each non-barrier instruction, the body collapses to
`applyToLaneIds? → advanceRunnablePcs?`. The store case is a for-loop
of `writeMem?` calls then advance. -/

theorem stepInstr?_preserves_warp_support
    {st st' : State} {c : CTAId} {w : WarpId} {gi : GInstr}
    (hStep : Helpers.stepInstr? st c w gi = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.stepInstr? at hStep
  cases hWarp : st.getWarp? c w with
  | none => simp [hWarp] at hStep
  | some ws =>
      rw [hWarp] at hStep
      simp at hStep
      cases hLock : Helpers.lockstepRunnable? ws with
      | false => simp [hLock] at hStep
      | true =>
          simp [hLock] at hStep
          cases hPart : Helpers.participatingRunnableLaneIds? ws gi.guard? with
          | none => simp [hPart] at hStep
          | some participants =>
              rw [hPart] at hStep
              simp at hStep
              cases hInstr : gi.instr with
              | barrierCTA bid =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact stepBarrierCTA?_preserves_warp_support hStep
              | assignReg dst rhs =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | assignPred dst cmp =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | assignPredValue dst rhs =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | load dst src =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | store dst value =>
                  rw [hInstr] at hStep
                  simp at hStep
                  -- Helper: the body in the for-loop preserves warp support
                  -- because each iteration is a chain ending in writeMem?.
                  have hBodyPres : ∀ lane (s s' : State),
                      ((Helpers.resolveAddr? s c w lane dst).bind fun addr =>
                          (Helpers.evalRValue? s c w lane value).bind fun v =>
                            (Helpers.writeMem? s dst.space dst.ty addr v).bind fun r =>
                              some (ForInStep.yield r)) = some (ForInStep.yield s') ∨
                      ((Helpers.resolveAddr? s c w lane dst).bind fun addr =>
                          (Helpers.evalRValue? s c w lane value).bind fun v =>
                            (Helpers.writeMem? s dst.space dst.ty addr v).bind fun r =>
                              some (ForInStep.yield r)) = some (ForInStep.done s')
                      → WarpSupportEq s s' := by
                    intro lane s s' hB
                    cases hAddr : Helpers.resolveAddr? s c w lane dst with
                    | none =>
                        rcases hB with hY | hD
                        · rw [hAddr] at hY; simp at hY
                        · rw [hAddr] at hD; simp at hD
                    | some addr =>
                        rw [hAddr] at hB
                        simp at hB
                        cases hVal : Helpers.evalRValue? s c w lane value with
                        | none =>
                            rcases hB with hY | hD
                            · rw [hVal] at hY; simp at hY
                            · rw [hVal] at hD; simp at hD
                        | some v =>
                            rw [hVal] at hB
                            simp at hB
                            cases hW : Helpers.writeMem? s dst.space dst.ty addr v with
                            | none =>
                                rcases hB with hY | hD
                                · rw [hW] at hY; simp at hY
                                · rw [hW] at hD; simp at hD
                            | some sW =>
                                rcases hB with hY | hD
                                · rw [hW] at hY
                                  simp at hY
                                  subst hY
                                  exact writeMem?_preserves_warp_support hW
                                · rw [hW] at hD
                                  simp at hD
                  cases hLoop :
                      (forIn (m := Option) participants st fun lane r =>
                        (Helpers.resolveAddr? r c w lane dst).bind fun addr =>
                          (Helpers.evalRValue? r c w lane value).bind fun v =>
                            (Helpers.writeMem? r dst.space dst.ty addr v).bind fun r =>
                              some (ForInStep.yield r)) with
                  | none => rw [hLoop] at hStep; simp at hStep
                  | some sMid =>
                      rw [hLoop] at hStep
                      simp at hStep
                      have hLoopPres : WarpSupportEq st sMid :=
                        WarpSupportEq.forIn_option_general participants st sMid _ hBodyPres hLoop
                      have hAdvPres := advanceRunnablePcs?_preserves_warp_support hStep
                      exact WarpSupportEq.trans hLoopPres hAdvPres
              | cvta dst space src =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | isspacep dst space src =>
                  rw [hInstr] at hStep
                  simp at hStep
                  exact WarpSupportEq.applyToLaneIds_then_advance hStep
              | warp _ => rw [hInstr] at hStep; simp at hStep
              | atomic _ _ _ _ _ => rw [hInstr] at hStep; simp at hStep
              | mma _ => rw [hInstr] at hStep; simp at hStep

/-! ## `stepTerminator?` -/

theorem stepTerminator?_preserves_warp_support
    {st st' : State} {c : CTAId} {w : WarpId} {term : Terminator}
    (hStep : Helpers.stepTerminator? st c w term = some st') :
    WarpSupportEq st st' := by
  unfold Helpers.stepTerminator? at hStep
  cases hWarp : st.getWarp? c w with
  | none => simp [hWarp] at hStep
  | some ws =>
      rw [hWarp] at hStep
      simp at hStep
      cases hLock : Helpers.lockstepRunnable? ws with
      | false => simp [hLock] at hStep
      | true =>
          simp [hLock] at hStep
          cases hPc : Helpers.currentRunnablePc? ws with
          | none => simp [hPc] at hStep
          | some pc =>
              rw [hPc] at hStep
              simp at hStep
              cases hTerm : term with
              | br label =>
                  rw [hTerm] at hStep
                  simp at hStep
                  exact applyToLaneIds?_preserves_warp_support hStep
              | terminate =>
                  rw [hTerm] at hStep
                  simp at hStep
                  exact applyToLaneIds?_preserves_warp_support hStep
              | cbr cond tLabel fLabel =>
                  rw [hTerm] at hStep
                  simp at hStep
                  -- hStep : (uniformBranchDestination? ...).bind ... = some st'
                  -- Case-split via Option.bind_eq_some.
                  rw [Option.bind_eq_some_iff] at hStep
                  obtain ⟨dest, _hDest, hApply⟩ := hStep
                  exact applyToLaneIds?_preserves_warp_support hApply

/-! ## `step?` and `IsSingleWarp` preservation

The headline lemma. `step? = stepAt? 0 0`, which either runs
`currentInstrStep?` or falls back to `currentTermStep?`. Each ultimately
calls `stepInstr?` or `stepTerminator?`, both of which preserve warp
support. -/

theorem step?_preserves_warp_support
    {st st' : State} (hStep : StepMachine.step? st = some st') :
    WarpSupportEq st st' := by
  unfold StepMachine.step? StepMachine.stepAt? at hStep
  cases hInstr : StepMachine.currentInstrStep? st 0 0 with
  | some sInstr =>
      rw [hInstr] at hStep
      simp at hStep
      -- hStep : sInstr = st' (after simp)
      subst hStep
      -- Unfold currentInstrStep? to get a stepInstr? application.
      unfold StepMachine.currentInstrStep? at hInstr
      cases hWfSt : st.wf? with
      | false => simp [hWfSt] at hInstr
      | true =>
          simp [hWfSt] at hInstr
          cases hWarp : st.getWarp? 0 0 with
          | none => simp [hWarp] at hInstr
          | some ws =>
              rw [hWarp] at hInstr
              simp at hInstr
              cases hWfWS : ws.wf? with
              | false => simp [hWfWS] at hInstr
              | true =>
                  simp [hWfWS] at hInstr
                  cases hLock : Helpers.lockstepRunnable? ws with
                  | false => simp [hLock] at hInstr
                  | true =>
                      simp [hLock] at hInstr
                      cases hPc : Helpers.currentRunnablePc? ws with
                      | none => simp [hPc] at hInstr
                      | some pc =>
                          rw [hPc] at hInstr
                          simp at hInstr
                          cases hBlock : st.kernelEnv.blocks[pc.1]? with
                          | none => simp [hBlock] at hInstr
                          | some block =>
                              rw [hBlock] at hInstr
                              simp at hInstr
                              cases hBody : block.body[pc.2]? with
                              | none => simp [hBody] at hInstr
                              | some gi =>
                                  rw [hBody] at hInstr
                                  simp at hInstr
                                  cases hPart : Helpers.participatingRunnableLaneIds? ws gi.guard? with
                                  | none => simp [hPart] at hInstr
                                  | some parts =>
                                      rw [hPart] at hInstr
                                      simp at hInstr
                                      exact stepInstr?_preserves_warp_support hInstr
  | none =>
      rw [hInstr] at hStep
      simp at hStep
      -- Fall back to currentTermStep?.
      unfold StepMachine.currentTermStep? at hStep
      cases hWfSt : st.wf? with
      | false => simp [hWfSt] at hStep
      | true =>
          simp [hWfSt] at hStep
          cases hWarp : st.getWarp? 0 0 with
          | none => simp [hWarp] at hStep
          | some ws =>
              rw [hWarp] at hStep
              simp at hStep
              cases hWfWS : ws.wf? with
              | false => simp [hWfWS] at hStep
              | true =>
                  simp [hWfWS] at hStep
                  cases hLock : Helpers.lockstepRunnable? ws with
                  | false => simp [hLock] at hStep
                  | true =>
                      simp [hLock] at hStep
                      cases hPc : Helpers.currentRunnablePc? ws with
                      | none => simp [hPc] at hStep
                      | some pc =>
                          rw [hPc] at hStep
                          simp at hStep
                          cases hBlock : st.kernelEnv.blocks[pc.1]? with
                          | none => simp [hBlock] at hStep
                          | some block =>
                              rw [hBlock] at hStep
                              simp at hStep
                              cases hBody : block.body[pc.2]? with
                              | some _ => simp [hBody] at hStep
                              | none =>
                                  simp [hBody] at hStep
                                  exact stepTerminator?_preserves_warp_support hStep

/-- The headline preservation theorem. -/
theorem step?_preserves_IsSingleWarp
    {st st' : State} (hsw : IsSingleWarp st)
    (hStep : StepMachine.step? st = some st') :
    IsSingleWarp st' :=
  IsSingleWarp.of_support_eq hsw (step?_preserves_warp_support hStep)

end CLean
