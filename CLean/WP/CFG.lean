import CLean.Semantics.Execution
import CLean.WP.Block

namespace CLean
namespace WP

def MachineFinal (st : State) : Prop :=
  ∀ st', ¬ StepMachine st st'

def TerminatesAt (init final : State) : Prop :=
  Reaches init final ∧ MachineFinal final

def PartialCorrect (init : State) (post : State → Prop) : Prop :=
  ∀ final, TerminatesAt init final → post final

structure KernelSpec where
  init : State
  resource : CSL.Resource := CSL.Resource.empty
  pre : CSL.Assertion := CSL.emp
  invariant : CSL.Assertion := CSL.emp
  post : CSL.Assertion := CSL.emp

abbrev InvariantMap := BlockLabel → CSL.Assertion

private theorem list_drop_eq_cons_of_getElem?_some
    {α : Type} {xs : List α} {idx : Nat} {x : α}
    (hget : xs[idx]? = some x) :
    ∃ rest, xs.drop idx = x :: rest := by
  induction idx generalizing xs with
  | zero =>
      cases xs with
      | nil =>
          simp at hget
      | cons head tail =>
          simp at hget
          subst head
          exact ⟨tail, rfl⟩
  | succ idx ih =>
      cases xs with
      | nil =>
          simp at hget
      | cons head tail =>
          simp at hget
          exact ih hget

private theorem list_drop_eq_nil_of_getElem?_none
    {α : Type} {xs : List α} {idx : Nat}
    (hget : xs[idx]? = none) :
    xs.drop idx = [] := by
  induction idx generalizing xs with
  | zero =>
      cases xs with
      | nil =>
          rfl
      | cons head tail =>
          simp at hget
  | succ idx ih =>
      cases xs with
      | nil =>
          rfl
      | cons head tail =>
          change tail[idx]? = none at hget
          exact ih hget

private theorem list_drop_succ_eq_tail_of_drop_eq_cons
    {α : Type} {xs : List α} {idx : Nat} {x : α} {rest : List α}
    (hdrop : xs.drop idx = x :: rest) :
    xs.drop (idx + 1) = rest := by
  induction idx generalizing xs with
  | zero =>
      cases xs with
      | nil =>
          simp at hdrop
      | cons head tail =>
          simp at hdrop
          exact hdrop.2
  | succ idx ih =>
      cases xs with
      | nil =>
          simp at hdrop
      | cons head tail =>
          simp at hdrop ⊢
          exact ih hdrop

theorem Array.toList_drop_eq_cons_of_getElem?_some
    {α : Type} {xs : Array α} {idx : Nat} {x : α}
    (hget : xs[idx]? = some x) :
    ∃ rest, xs.toList.drop idx = x :: rest := by
  cases xs with
  | mk data =>
      exact list_drop_eq_cons_of_getElem?_some (by simpa using hget)

theorem Array.toList_drop_eq_nil_of_getElem?_none
    {α : Type} {xs : Array α} {idx : Nat}
    (hget : xs[idx]? = none) :
    xs.toList.drop idx = [] := by
  cases xs with
  | mk data =>
      exact list_drop_eq_nil_of_getElem?_none (by simpa using hget)

theorem Array.toList_drop_succ_eq_tail_of_drop_eq_cons
    {α : Type} {xs : Array α} {idx : Nat} {x : α} {rest : List α}
    (hdrop : xs.toList.drop idx = x :: rest) :
    xs.toList.drop (idx + 1) = rest :=
  list_drop_succ_eq_tail_of_drop_eq_cons hdrop

def StepPreserves (inv : CSL.Assertion) : Prop :=
  ∀ st r st', inv st r → StepMachine st st' →
    ∃ r', CSL.Resource.Update r r' ∧ inv st' r'

def StepMachineSelects
    (inv : CSL.Assertion) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ st r st', inv st r → StepMachine st st' → StepWarp st cta warp st'

def OnlyRunnableWarp (cta : CTAId) (warp : WarpId) : CSL.Assertion :=
  fun st _ => ∀ st', StepMachine st st' → StepWarp st cta warp st'

theorem OnlyRunnableWarp.stepMachineSelects
    {cta : CTAId} {warp : WarpId} {inv : CSL.Assertion} :
    StepMachineSelects (OnlyRunnableWarp cta warp ∗ inv) cta warp := by
  intro st r st' hinv hstep
  rcases hinv with ⟨_rOnly, _rInv, _hcomp, _hequiv, hselect, _hinv⟩
  exact hselect st' hstep

theorem StepMachineSelects.of_entails_onlyRunnableWarp
    {cta : CTAId} {warp : WarpId} {inv : CSL.Assertion}
    (honly : inv ⊢ₛ OnlyRunnableWarp cta warp) :
    StepMachineSelects inv cta warp := by
  intro st r st' hinv hstep
  exact honly st r hinv st' hstep

def StepWarpPreserves (cta : CTAId) (warp : WarpId) (inv : CSL.Assertion) : Prop :=
  ∀ st r st', inv st r → StepWarp st cta warp st' →
    ∃ r', CSL.Resource.Update r r' ∧ inv st' r'

def StepBlockPreserves (cta : CTAId) (warp : WarpId) (inv : CSL.Assertion) : Prop :=
  ∀ st r st', inv st r → StepBlock st cta warp st' →
    ∃ r', CSL.Resource.Update r r' ∧ inv st' r'

def NoStepBlock (cta : CTAId) (warp : WarpId) (inv : CSL.Assertion) : Prop :=
  ∀ st r st', inv st r → StepBlock st cta warp st' → False

def NoFinal (inv : CSL.Assertion) : Prop :=
  ∀ final r, MachineFinal final → inv final r → False

theorem StepBlockPreserves.of_no_step
    {cta : CTAId} {warp : WarpId} {inv : CSL.Assertion}
    (hno : NoStepBlock cta warp inv) :
    StepBlockPreserves cta warp inv := by
  intro st r st' hinv hstep
  exact False.elim (hno st r st' hinv hstep)

theorem StepWarpPreserves.of_stepBlockPreserves
    {cta : CTAId} {warp : WarpId} {inv : CSL.Assertion}
    (hpres : StepBlockPreserves cta warp inv) :
    StepWarpPreserves cta warp inv := by
  intro st r st' hinv hstep
  cases hstep with
  | mk _ hblock =>
      exact hpres st r st' hinv hblock

theorem StepPreserves.of_stepWarpPreserves
    {cta : CTAId} {warp : WarpId} {inv : CSL.Assertion}
    (hselect : StepMachineSelects inv cta warp)
    (hpres : StepWarpPreserves cta warp inv) :
    StepPreserves inv := by
  intro st r st' hinv hstep
  exact hpres st r st' hinv (hselect st r st' hinv hstep)

def Finalizes (inv post : CSL.Assertion) : Prop :=
  ∀ final r, MachineFinal final → inv final r → post final r

def blockTermPost (invariants : InvariantMap) (post : CSL.Assertion) :
    Terminator → CSL.Assertion
  | .br target => invariants target
  | .cbr _ t f => fun st r => invariants t st r ∧ invariants f st r
  | .terminate => post

def blockSuffixWP
    (cta : CTAId) (warp : WarpId) (invariants : InvariantMap) (post : CSL.Assertion)
    (block : Block) (idx : Nat) : CSL.Assertion :=
  wpInstrList cta warp (block.body.toList.drop idx)
    (wpTerminator cta warp block.term (blockTermPost invariants post block.term))

def blockEntryWP
    (cta : CTAId) (warp : WarpId) (invariants : InvariantMap) (post : CSL.Assertion)
    (block : Block) : CSL.Assertion :=
  blockSuffixWP cta warp invariants post block 0

def cfgSuffixInvariant
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    ∃ warpState pc block,
      st.kernelEnv = env ∧
        st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        env.blocks[pc.1]? = some block ∧
        blockSuffixWP cta warp invariants post block pc.2 st r

def cfgKernelInvariant
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    cfgSuffixInvariant env cta warp invariants post st r ∨ post st r

def blockVC
    (cta : CTAId) (warp : WarpId) (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) : Prop :=
  invariants label ⊢ₛ blockEntryWP cta warp invariants post block

def CbrBranchControl
    (cta : CTAId) (warp : WarpId) (cond : RValue)
    (tLabel fLabel target : BlockLabel) (pre : CSL.Assertion) : Prop :=
  ∀ {st st' : State} {r : CSL.Resource},
    pre st r →
    Helpers.stepTerminator? st cta warp (.cbr cond tLabel fLabel) = some st' →
      ∃ warpState',
        st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (target, 0)

inductive TerminatorVC
    (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) :
    BlockLabel → Terminator → CSL.Assertion → Prop where
  | br {label target : BlockLabel} {pre : CSL.Assertion} :
      pre ⊢ₛ wpTerminator cta warp (.br target) (invariants target) →
      TerminatorVC cta warp invariants post label (.br target) pre
  | cbr
      {label : BlockLabel} {cond : RValue} {t f : BlockLabel}
      {pre truePre falsePre : CSL.Assertion} :
      pre ⊢ₛ (truePre ∨ₛ falsePre) →
      CbrBranchControl cta warp cond t f t truePre →
      CbrBranchControl cta warp cond t f f falsePre →
      truePre ⊢ₛ wpTerminator cta warp (.cbr cond t f) (invariants t) →
      falsePre ⊢ₛ wpTerminator cta warp (.cbr cond t f) (invariants f) →
      TerminatorVC cta warp invariants post label (.cbr cond t f) pre
  | terminate {label : BlockLabel} {pre : CSL.Assertion} :
      pre ⊢ₛ wpTerminator cta warp .terminate post →
      TerminatorVC cta warp invariants post label .terminate pre

def blockVC'
    (cta : CTAId) (warp : WarpId) (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) : Prop :=
  ∃ termPre,
    invariants label ⊢ₛ wpInstrs cta warp block.body termPre ∧
    TerminatorVC cta warp invariants post label block.term termPre

def kernelVCs'
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (pre post : CSL.Assertion) (invariants : InvariantMap) : Prop :=
  pre ⊢ₛ invariants env.entry ∧
    (∀ label block,
      env.blocks[label]? = some block → blockVC' cta warp invariants post label block) ∧
    (∀ label, Finalizes (invariants label) post)

def blockSuffixWP'
    (cta : CTAId) (warp : WarpId) (termPre : CSL.Assertion)
    (block : Block) (idx : Nat) : CSL.Assertion :=
  wpInstrList cta warp (block.body.toList.drop idx) termPre

def blockEntryWP'
    (cta : CTAId) (warp : WarpId) (termPre : CSL.Assertion) (block : Block) :
    CSL.Assertion :=
  blockSuffixWP' cta warp termPre block 0

def cfgSuffixInvariant'
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    ∃ warpState pc block termPre,
      st.kernelEnv = env ∧
        st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        env.blocks[pc.1]? = some block ∧
        blockSuffixWP' cta warp termPre block pc.2 st r ∧
        TerminatorVC cta warp invariants post pc.1 block.term termPre

def cfgKernelInvariant'
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    cfgSuffixInvariant' env cta warp invariants post st r ∨ post st r

abbrev CbrChoiceMap := BlockLabel → Option Bool

def blockTermPostChoice
    (choices : CbrChoiceMap) (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) : Terminator → CSL.Assertion
  | .br target => invariants target
  | .cbr _ tLabel fLabel =>
      match choices label with
      | some true => invariants tLabel
      | some false => invariants fLabel
      | none => fun st r => invariants tLabel st r ∧ invariants fLabel st r
  | .terminate => post

def blockSuffixWPChoice
    (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) (idx : Nat) : CSL.Assertion :=
  wpInstrList cta warp (block.body.toList.drop idx)
    (wpTerminator cta warp block.term
      (blockTermPostChoice choices invariants post label block.term))

def blockEntryWPChoice
    (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) : CSL.Assertion :=
  blockSuffixWPChoice cta warp choices invariants post label block 0

def cfgSuffixInvariantChoice
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    ∃ warpState pc block,
      st.kernelEnv = env ∧
        st.getWarp? cta warp = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        env.blocks[pc.1]? = some block ∧
        blockSuffixWPChoice cta warp choices invariants post pc.1 block pc.2 st r

def cfgKernelInvariantChoice
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion) : CSL.Assertion :=
  fun st r =>
    cfgSuffixInvariantChoice env cta warp choices invariants post st r ∨ post st r

def blockVCChoice
    (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) : Prop :=
  invariants label ⊢ₛ blockEntryWPChoice cta warp choices invariants post label block

theorem blockEntryWP_eq_wpConcreteBlock
    (cta : CTAId) (warp : WarpId) (invariants : InvariantMap) (post : CSL.Assertion)
    (block : Block) :
    blockEntryWP cta warp invariants post block =
      wpConcreteBlock cta warp block (blockTermPost invariants post block.term) := by
  simp [blockEntryWP, blockSuffixWP, wpConcreteBlock, wpInstrs]

theorem blockEntryWPChoice_eq_wpConcreteBlock
    (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion)
    (label : BlockLabel) (block : Block) :
    blockEntryWPChoice cta warp choices invariants post label block =
      wpConcreteBlock cta warp block
        (blockTermPostChoice choices invariants post label block.term) := by
  simp [blockEntryWPChoice, blockSuffixWPChoice, wpConcreteBlock, wpInstrs]

theorem blockVC.entry_suffix
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {block : Block}
    (hvc : blockVC cta warp invariants post label block) :
    invariants label ⊢ₛ blockSuffixWP cta warp invariants post block 0 :=
  hvc

theorem blockVCChoice.entry_suffix
    {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion} {label : BlockLabel}
    {block : Block}
    (hvc : blockVCChoice cta warp choices invariants post label block) :
    invariants label ⊢ₛ
      blockSuffixWPChoice cta warp choices invariants post label block 0 :=
  hvc

theorem blockVC.of_wpConcreteBlock
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {block : Block}
    (hwp :
      invariants label ⊢ₛ
        wpConcreteBlock cta warp block (blockTermPost invariants post block.term)) :
    blockVC cta warp invariants post label block := by
  intro st r hinv
  simpa [blockEntryWP, blockSuffixWP, wpConcreteBlock, wpInstrs] using hwp st r hinv

theorem blockVCChoice.of_wpConcreteBlock
    {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion} {label : BlockLabel}
    {block : Block}
    (hwp :
      invariants label ⊢ₛ
        wpConcreteBlock cta warp block
          (blockTermPostChoice choices invariants post label block.term)) :
    blockVCChoice cta warp choices invariants post label block := by
  intro st r hinv
  simpa [blockEntryWPChoice, blockSuffixWPChoice, wpConcreteBlock, wpInstrs]
    using hwp st r hinv

theorem TerminatorVC.br_wp
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label target : BlockLabel} {pre : CSL.Assertion}
    (hvc : TerminatorVC cta warp invariants post label (.br target) pre) :
    pre ⊢ₛ wpTerminator cta warp (.br target) (invariants target) := by
  cases hvc with
  | br hwp => exact hwp

theorem TerminatorVC.cbr_wp_or
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {cond : RValue}
    {t f : BlockLabel} {pre : CSL.Assertion}
    (hvc : TerminatorVC cta warp invariants post label (.cbr cond t f) pre) :
    pre ⊢ₛ
      (wpTerminator cta warp (.cbr cond t f) (invariants t) ∨ₛ
        wpTerminator cta warp (.cbr cond t f) (invariants f)) := by
  cases hvc with
  | cbr hsplit _htrueControl _hfalseControl htrue hfalse =>
      intro st r hpre
      rcases hsplit st r hpre with hpreTrue | hpreFalse
      · exact Or.inl (htrue st r hpreTrue)
      · exact Or.inr (hfalse st r hpreFalse)

theorem TerminatorVC.terminate_wp
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {pre : CSL.Assertion}
    (hvc : TerminatorVC cta warp invariants post label .terminate pre) :
    pre ⊢ₛ wpTerminator cta warp .terminate post := by
  cases hvc with
  | terminate hwp => exact hwp

theorem blockVC'.entry_instrs
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {block : Block}
    (hvc : blockVC' cta warp invariants post label block) :
    ∃ termPre,
      invariants label ⊢ₛ wpInstrs cta warp block.body termPre ∧
      TerminatorVC cta warp invariants post label block.term termPre :=
  hvc

theorem blockVC'.entry_suffix
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {label : BlockLabel} {block : Block} {termPre : CSL.Assertion}
    (hbody : invariants label ⊢ₛ wpInstrs cta warp block.body termPre) :
    invariants label ⊢ₛ blockSuffixWP' cta warp termPre block 0 := by
  simpa [blockSuffixWP', wpInstrs] using hbody

theorem blockSuffixWP'.body_step_of_drop
    {cta : CTAId} {warp : WarpId} {termPre : CSL.Assertion}
    {block : Block} {idx : Nat}
    {gi : GInstr} {rest : List GInstr} {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = gi :: rest)
    (hwp : blockSuffixWP' cta warp termPre block idx st r)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ wpInstrList cta warp rest termPre st' r' := by
  have hwp' : wpInstr cta warp gi (wpInstrList cta warp rest termPre) st r := by
    simpa [blockSuffixWP', hdrop] using hwp
  exact hwp' st' hstep

theorem blockSuffixWP'.term_pre_of_drop
    {cta : CTAId} {warp : WarpId} {termPre : CSL.Assertion}
    {block : Block} {idx : Nat} {st : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = [])
    (hwp : blockSuffixWP' cta warp termPre block idx st r) :
    termPre st r := by
  simpa [blockSuffixWP', hdrop] using hwp

theorem cfgSuffixInvariant'.body_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC} {block : Block}
    {termPre : CSL.Assertion} {gi : GInstr} {rest : List GInstr}
    (hwp : blockSuffixWP' cta warp termPre block pc.2 st r)
    (htermVC : TerminatorVC cta warp invariants post pc.1 block.term termPre)
    (hdrop : block.body.toList.drop pc.2 = gi :: rest)
    (hblock : env.blocks[pc.1]? = some block)
    (hstep : Helpers.stepInstr? st cta warp gi = some st')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant' env cta warp invariants post st' r' := by
  rcases blockSuffixWP'.body_step_of_drop hdrop hwp hstep with
    ⟨r', hupdate, hwpRest⟩
  rcases hcontrol with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  have hdropNext : block.body.toList.drop (pc.2 + 1) = rest :=
    Array.toList_drop_succ_eq_tail_of_drop_eq_cons hdrop
  have hwpNext : blockSuffixWP' cta warp termPre block (pc.2 + 1) st' r' := by
    simpa [blockSuffixWP', hdropNext] using hwpRest
  exact ⟨r', hupdate,
    ⟨warpState', (pc.1, pc.2 + 1), block, termPre, henv', hwarp', hlock', hrpc',
      hblock, hwpNext, htermVC⟩⟩

theorem cfgSuffixInvariant'.of_target_invariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st : State} {r : CSL.Resource} {target : BlockLabel} {targetBlock : Block}
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block → blockVC' cta warp invariants post label block)
    (hblockTarget : env.blocks[target]? = some targetBlock)
    (hinvTarget : invariants target st r)
    (hcontrol :
      ∃ warpState,
        st.kernelEnv = env ∧
          st.getWarp? cta warp = some warpState ∧
          Helpers.lockstepRunnable warpState ∧
          Helpers.RunnablePc warpState (target, 0)) :
    cfgSuffixInvariant' env cta warp invariants post st r := by
  rcases hblocks target targetBlock hblockTarget with ⟨targetPre, hbody, htermVC⟩
  rcases hcontrol with ⟨warpState, henv, hwarp, hlock, hrpc⟩
  have hentry : blockSuffixWP' cta warp targetPre targetBlock 0 st r :=
    blockVC'.entry_suffix (invariants := invariants) hbody st r hinvTarget
  exact ⟨warpState, (target, 0), targetBlock, targetPre, henv, hwarp, hlock, hrpc,
    hblockTarget, hentry, htermVC⟩

theorem blockSuffixWP.body_step_of_drop
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {block : Block} {idx : Nat}
    {gi : GInstr} {rest : List GInstr} {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = gi :: rest)
    (hwp : blockSuffixWP cta warp invariants post block idx st r)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    ∃ r', CSL.Resource.Update r r' ∧
      wpInstrList cta warp rest
        (wpTerminator cta warp block.term (blockTermPost invariants post block.term)) st' r' := by
  have hwp' :
      wpInstr cta warp gi
        (wpInstrList cta warp rest
          (wpTerminator cta warp block.term (blockTermPost invariants post block.term)))
        st r := by
    simpa [blockSuffixWP, hdrop] using hwp
  exact hwp' st' hstep

theorem blockSuffixWP.term_step_of_drop
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {block : Block} {idx : Nat}
    {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = [])
    (hwp : blockSuffixWP cta warp invariants post block idx st r)
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ blockTermPost invariants post block.term st' r' := by
  have hwp' :
      wpTerminator cta warp block.term (blockTermPost invariants post block.term) st r := by
    simpa [blockSuffixWP, hdrop] using hwp
  exact hwp' st' hstep

theorem cfgSuffixInvariant.body_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC} {block : Block}
    {gi : GInstr} {rest : List GInstr}
    (hwp : blockSuffixWP cta warp invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = gi :: rest)
    (hblock : env.blocks[pc.1]? = some block)
    (hstep : Helpers.stepInstr? st cta warp gi = some st')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env cta warp invariants post st' r' := by
  rcases blockSuffixWP.body_step_of_drop hdrop hwp hstep with
    ⟨r', hupdate, hwpRest⟩
  rcases hcontrol with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  have hdropNext : block.body.toList.drop (pc.2 + 1) = rest :=
    Array.toList_drop_succ_eq_tail_of_drop_eq_cons hdrop
  have hwpNext : blockSuffixWP cta warp invariants post block (pc.2 + 1) st' r' := by
    simpa [blockSuffixWP, hdropNext] using hwpRest
  exact ⟨r', hupdate,
    ⟨warpState', (pc.1, pc.2 + 1), block, henv', hwarp', hlock', hrpc', hblock, hwpNext⟩⟩

theorem cfgSuffixInvariant.term_step_to_suffix_of_post
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc targetPc : PC}
    {block targetBlock : Block}
    (hwp : blockSuffixWP cta warp invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st')
    (hblockTarget : env.blocks[targetPc.1]? = some targetBlock)
    (hpostToTarget :
      ∀ r',
        blockTermPost invariants post block.term st' r' →
          blockSuffixWP cta warp invariants post targetBlock targetPc.2 st' r')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' targetPc) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env cta warp invariants post st' r' := by
  rcases blockSuffixWP.term_step_of_drop hdrop hwp hstep with
    ⟨r', hupdate, htermPost⟩
  rcases hcontrol with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  exact ⟨r', hupdate,
    ⟨warpState', targetPc, targetBlock, henv', hwarp', hlock', hrpc',
      hblockTarget, hpostToTarget r' htermPost⟩⟩

theorem cfgSuffixInvariant.br_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {target : BlockLabel}
    (hterm : block.term = .br target)
    (hwp : blockSuffixWP cta warp invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st')
    (hblockTarget : env.blocks[target]? = some targetBlock)
    (hvcTarget : blockVC cta warp invariants post target targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (target, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env cta warp invariants post st' r' :=
  cfgSuffixInvariant.term_step_to_suffix_of_post
    (targetPc := (target, 0)) hwp hdrop hstep hblockTarget
    (by
      intro r' htermPost
      exact blockVC.entry_suffix hvcTarget st' r' (by
        simpa [blockTermPost, hterm] using htermPost))
    hcontrol

theorem cfgSuffixInvariant.cbr_true_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {cond : RValue} {tLabel fLabel : BlockLabel}
    (hterm : block.term = .cbr cond tLabel fLabel)
    (hwp : blockSuffixWP cta warp invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st')
    (hblockTarget : env.blocks[tLabel]? = some targetBlock)
    (hvcTarget : blockVC cta warp invariants post tLabel targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (tLabel, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env cta warp invariants post st' r' :=
  cfgSuffixInvariant.term_step_to_suffix_of_post
    (targetPc := (tLabel, 0)) hwp hdrop hstep hblockTarget
    (by
      intro r' htermPost
      have hboth :
          invariants tLabel st' r' ∧ invariants fLabel st' r' := by
        simpa [blockTermPost, hterm] using htermPost
      have hinv : invariants tLabel st' r' := hboth.1
      exact blockVC.entry_suffix hvcTarget st' r' hinv)
    hcontrol

theorem cfgSuffixInvariant.cbr_false_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {cond : RValue} {tLabel fLabel : BlockLabel}
    (hterm : block.term = .cbr cond tLabel fLabel)
    (hwp : blockSuffixWP cta warp invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st')
    (hblockTarget : env.blocks[fLabel]? = some targetBlock)
    (hvcTarget : blockVC cta warp invariants post fLabel targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (fLabel, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env cta warp invariants post st' r' :=
  cfgSuffixInvariant.term_step_to_suffix_of_post
    (targetPc := (fLabel, 0)) hwp hdrop hstep hblockTarget
    (by
      intro r' htermPost
      have hboth :
          invariants tLabel st' r' ∧ invariants fLabel st' r' := by
        simpa [blockTermPost, hterm] using htermPost
      have hinv : invariants fLabel st' r' := hboth.2
      exact blockVC.entry_suffix hvcTarget st' r' hinv)
    hcontrol

theorem blockSuffixWPChoice.body_step_of_drop
    {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block} {idx : Nat}
    {gi : GInstr} {rest : List GInstr} {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = gi :: rest)
    (hwp : blockSuffixWPChoice cta warp choices invariants post label block idx st r)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    ∃ r', CSL.Resource.Update r r' ∧
      wpInstrList cta warp rest
        (wpTerminator cta warp block.term
          (blockTermPostChoice choices invariants post label block.term)) st' r' := by
  have hwp' :
      wpInstr cta warp gi
        (wpInstrList cta warp rest
          (wpTerminator cta warp block.term
            (blockTermPostChoice choices invariants post label block.term)))
        st r := by
    simpa [blockSuffixWPChoice, hdrop] using hwp
  exact hwp' st' hstep

theorem blockSuffixWPChoice.term_step_of_drop
    {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block} {idx : Nat}
    {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = [])
    (hwp : blockSuffixWPChoice cta warp choices invariants post label block idx st r)
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st') :
    ∃ r', CSL.Resource.Update r r' ∧
      blockTermPostChoice choices invariants post label block.term st' r' := by
  have hwp' :
      wpTerminator cta warp block.term
        (blockTermPostChoice choices invariants post label block.term) st r := by
    simpa [blockSuffixWPChoice, hdrop] using hwp
  exact hwp' st' hstep

theorem cfgSuffixInvariantChoice.body_step_of_suffix
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC} {block : Block}
    {gi : GInstr} {rest : List GInstr}
    (hwp : blockSuffixWPChoice cta warp choices invariants post pc.1 block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = gi :: rest)
    (hblock : env.blocks[pc.1]? = some block)
    (hstep : Helpers.stepInstr? st cta warp gi = some st')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariantChoice env cta warp choices invariants post st' r' := by
  rcases blockSuffixWPChoice.body_step_of_drop hdrop hwp hstep with
    ⟨r', hupdate, hwpRest⟩
  rcases hcontrol with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  have hdropNext : block.body.toList.drop (pc.2 + 1) = rest :=
    Array.toList_drop_succ_eq_tail_of_drop_eq_cons hdrop
  have hwpNext :
      blockSuffixWPChoice cta warp choices invariants post pc.1 block (pc.2 + 1)
        st' r' := by
    simpa [blockSuffixWPChoice, hdropNext] using hwpRest
  exact ⟨r', hupdate,
    ⟨warpState', (pc.1, pc.2 + 1), block, henv', hwarp', hlock', hrpc', hblock,
      hwpNext⟩⟩

theorem cfgSuffixInvariantChoice.term_step_to_suffix_of_post
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc targetPc : PC}
    {block targetBlock : Block}
    (hwp : blockSuffixWPChoice cta warp choices invariants post pc.1 block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st cta warp block.term = some st')
    (hblockTarget : env.blocks[targetPc.1]? = some targetBlock)
    (hpostToTarget :
      ∀ r',
        blockTermPostChoice choices invariants post pc.1 block.term st' r' →
          blockSuffixWPChoice cta warp choices invariants post targetPc.1 targetBlock
            targetPc.2 st' r')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' targetPc) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariantChoice env cta warp choices invariants post st' r' := by
  rcases blockSuffixWPChoice.term_step_of_drop hdrop hwp hstep with
    ⟨r', hupdate, htermPost⟩
  rcases hcontrol with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  exact ⟨r', hupdate,
    ⟨warpState', targetPc, targetBlock, henv', hwarp', hlock', hrpc',
      hblockTarget, hpostToTarget r' htermPost⟩⟩

def BodyStepControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {gi : GInstr},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = some gi →
    Helpers.stepInstr? st cta warp gi = some st' →
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)

def CFGBodyUsesOrdinaryPcAdvance (env : KernelEnv) : Prop :=
  ∀ {label : BlockLabel} {block : Block} {idx : Nat} {gi : GInstr},
    env.blocks[label]? = some block →
    block.body[idx]? = some gi →
    Helpers.instrUsesOrdinaryPcAdvance gi.instr = true

def OrdinaryBodyStepControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {gi : GInstr},
    Helpers.instrUsesOrdinaryPcAdvance gi.instr = true →
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = some gi →
    Helpers.stepInstr? st cta warp gi = some st' →
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)

def OrdinaryCoreStepControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {gi : GInstr},
    Helpers.instrUsesOrdinaryPcAdvance gi.instr = true →
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = some gi →
    Helpers.stepInstr? st cta warp gi = some st' →
      ∃ stCore warpCore,
        Helpers.advanceRunnablePcs? stCore cta warp = some st' ∧
          stCore.kernelEnv = env ∧
          stCore.getWarp? cta warp = some warpCore ∧
          Helpers.lockstepRunnable warpCore ∧
          Helpers.RunnablePc warpCore pc

theorem OrdinaryBodyStepControl.of_core
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcore : OrdinaryCoreStepControl env cta warp) :
    OrdinaryBodyStepControl env cta warp := by
  intro st st' warpState pc block gi hordinary henv hwarp hlock hrpc hblock hgi hstep
  rcases hcore hordinary henv hwarp hlock hrpc hblock hgi hstep with
    ⟨stCore, warpCore, hadvance, henvCore, hwarpCore, hlockCore, hrpcCore⟩
  rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hadvance with
    ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
  have henvFinal : st'.kernelEnv = env := by
    exact (Helpers.advanceRunnablePcs?_kernelEnv_eq hadvance).trans henvCore
  exact ⟨warpFinal, henvFinal, hwarpFinal, hlockFinal, hrpcFinal⟩

theorem OrdinaryCoreStepControl.of_semantics
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} :
    OrdinaryCoreStepControl env cta warp := by
  intro st st' warpState pc block gi hordinary henv hwarp hlock hrpc _hblock _hgi hstep
  rcases Helpers.stepInstr?_ordinary_core_control hordinary hwarp hlock hrpc hstep with
    ⟨stCore, warpCore, hadvance, henvCore, hwarpCore, hlockCore, hrpcCore⟩
  exact ⟨stCore, warpCore, hadvance, henvCore.trans henv, hwarpCore, hlockCore, hrpcCore⟩

theorem OrdinaryBodyStepControl.of_semantics
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} :
    OrdinaryBodyStepControl env cta warp :=
  OrdinaryBodyStepControl.of_core OrdinaryCoreStepControl.of_semantics

theorem BodyStepControl.of_ordinary_cfg
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcfg : CFGBodyUsesOrdinaryPcAdvance env)
    (hordinary : OrdinaryBodyStepControl env cta warp) :
    BodyStepControl env cta warp := by
  intro st st' warpState pc block gi henv hwarp hlock hrpc hblock hgi hstep
  exact hordinary (hcfg hblock hgi) henv hwarp hlock hrpc hblock hgi hstep

theorem BodyStepControl.of_ordinary_cfg_semantics
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcfg : CFGBodyUsesOrdinaryPcAdvance env) :
    BodyStepControl env cta warp :=
  BodyStepControl.of_ordinary_cfg hcfg OrdinaryBodyStepControl.of_semantics

def TermStepPreservesCFG
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : Prop :=
  ∀ {st st' : State} {r : CSL.Resource} {warpState : WarpState}
    {pc : PC} {block : Block},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = none →
    blockSuffixWP cta warp invariants post block pc.2 st r →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧
        cfgSuffixInvariant env cta warp invariants post st' r'

def TermStepPreservesKernel
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : Prop :=
  ∀ {st st' : State} {r : CSL.Resource} {warpState : WarpState}
    {pc : PC} {block : Block},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = none →
    blockSuffixWP cta warp invariants post block pc.2 st r →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧
        cfgKernelInvariant env cta warp invariants post st' r'

def TermStepPreservesKernel'
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (invariants : InvariantMap) (post : CSL.Assertion) : Prop :=
  ∀ {st st' : State} {r : CSL.Resource} {warpState : WarpState}
    {pc : PC} {block : Block} {termPre : CSL.Assertion},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = none →
    blockSuffixWP' cta warp termPre block pc.2 st r →
    TerminatorVC cta warp invariants post pc.1 block.term termPre →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧
        cfgKernelInvariant' env cta warp invariants post st' r'

def TermStepPreservesKernelChoice
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (invariants : InvariantMap) (post : CSL.Assertion) : Prop :=
  ∀ {st st' : State} {r : CSL.Resource} {warpState : WarpState}
    {pc : PC} {block : Block},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.body[pc.2]? = none →
    blockSuffixWPChoice cta warp choices invariants post pc.1 block pc.2 st r →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧
        cfgKernelInvariantChoice env cta warp choices invariants post st' r'

def CFGTerminatorTargetsExist (env : KernelEnv) : Prop :=
  ∀ {label : BlockLabel} {block : Block},
    env.blocks[label]? = some block →
      match block.term with
      | .br target => ∃ targetBlock, env.blocks[target]? = some targetBlock
      | .cbr _ tLabel fLabel =>
          (∃ trueBlock, env.blocks[tLabel]? = some trueBlock) ∧
            ∃ falseBlock, env.blocks[fLabel]? = some falseBlock
      | .terminate => True

def BrSemanticControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block targetBlock : Block}
    {target : BlockLabel},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.term = .br target →
    env.blocks[target]? = some targetBlock →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (target, 0)

def CbrSemanticControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC}
    {block trueBlock falseBlock : Block} {cond : RValue} {tLabel fLabel : BlockLabel},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.term = .cbr cond tLabel fLabel →
    env.blocks[tLabel]? = some trueBlock →
    env.blocks[fLabel]? = some falseBlock →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      (∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (tLabel, 0)) ∨
      (∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (fLabel, 0))

def BrTermControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {target : BlockLabel},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.term = .br target →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      ∃ targetBlock warpState',
        env.blocks[target]? = some targetBlock ∧
          st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (target, 0)

def CbrTermControl (env : KernelEnv) (cta : CTAId) (warp : WarpId) : Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {cond : RValue} {tLabel fLabel : BlockLabel},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.term = .cbr cond tLabel fLabel →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      (∃ targetBlock warpState',
        env.blocks[tLabel]? = some targetBlock ∧
          st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (tLabel, 0)) ∨
      (∃ targetBlock warpState',
        env.blocks[fLabel]? = some targetBlock ∧
          st'.kernelEnv = env ∧
          st'.getWarp? cta warp = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (fLabel, 0))

def CbrChoiceTermControl
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap) :
    Prop :=
  ∀ {st st' : State} {warpState : WarpState} {pc : PC} {block : Block}
    {cond : RValue} {tLabel fLabel : BlockLabel},
    st.kernelEnv = env →
    st.getWarp? cta warp = some warpState →
    Helpers.lockstepRunnable warpState →
    Helpers.RunnablePc warpState pc →
    env.blocks[pc.1]? = some block →
    block.term = .cbr cond tLabel fLabel →
    Helpers.stepTerminator? st cta warp block.term = some st' →
      match choices pc.1 with
      | some true =>
          ∃ targetBlock warpState',
            env.blocks[tLabel]? = some targetBlock ∧
              st'.kernelEnv = env ∧
              st'.getWarp? cta warp = some warpState' ∧
              Helpers.lockstepRunnable warpState' ∧
              Helpers.RunnablePc warpState' (tLabel, 0)
      | some false =>
          ∃ targetBlock warpState',
            env.blocks[fLabel]? = some targetBlock ∧
              st'.kernelEnv = env ∧
              st'.getWarp? cta warp = some warpState' ∧
              Helpers.lockstepRunnable warpState' ∧
              Helpers.RunnablePc warpState' (fLabel, 0)
      | none =>
          (∃ targetBlock warpState',
            env.blocks[tLabel]? = some targetBlock ∧
              st'.kernelEnv = env ∧
              st'.getWarp? cta warp = some warpState' ∧
              Helpers.lockstepRunnable warpState' ∧
              Helpers.RunnablePc warpState' (tLabel, 0)) ∨
          (∃ targetBlock warpState',
            env.blocks[fLabel]? = some targetBlock ∧
              st'.kernelEnv = env ∧
              st'.getWarp? cta warp = some warpState' ∧
              Helpers.lockstepRunnable warpState' ∧
              Helpers.RunnablePc warpState' (fLabel, 0))

theorem BrTermControl.of_targets_semantic
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (htargets : CFGTerminatorTargetsExist env)
    (hsemantic : BrSemanticControl env cta warp) :
    BrTermControl env cta warp := by
  intro st st' warpState pc block target henv hwarp hlock hrpc hblock hterm hstep
  have htargetsBlock := htargets hblock
  rw [hterm] at htargetsBlock
  rcases htargetsBlock with ⟨targetBlock, htargetBlock⟩
  rcases hsemantic henv hwarp hlock hrpc hblock hterm htargetBlock hstep with
    ⟨warpState', henv', hwarp', hlock', hrpc'⟩
  exact ⟨targetBlock, warpState', htargetBlock, henv', hwarp', hlock', hrpc'⟩

theorem CbrTermControl.of_targets_semantic
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (htargets : CFGTerminatorTargetsExist env)
    (hsemantic : CbrSemanticControl env cta warp) :
    CbrTermControl env cta warp := by
  intro st st' warpState pc block cond tLabel fLabel
    henv hwarp hlock hrpc hblock hterm hstep
  have htargetsBlock := htargets hblock
  rw [hterm] at htargetsBlock
  rcases htargetsBlock with ⟨⟨trueBlock, htrueBlock⟩, falseBlock, hfalseBlock⟩
  rcases hsemantic henv hwarp hlock hrpc hblock hterm htrueBlock hfalseBlock hstep with
    htrue | hfalse
  · rcases htrue with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
    exact Or.inl ⟨trueBlock, warpState', htrueBlock, henv', hwarp', hlock', hrpc'⟩
  · rcases hfalse with ⟨warpState', henv', hwarp', hlock', hrpc'⟩
    exact Or.inr ⟨falseBlock, warpState', hfalseBlock, henv', hwarp', hlock', hrpc'⟩

theorem TermStepPreservesKernel'.of_blockVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block → blockVC' cta warp invariants post label block)
    (hbr : BrTermControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env) :
    TermStepPreservesKernel' env cta warp invariants post := by
  intro st st' r warpState pc block termPre
    henv hwarp hlock hrpc hblock hdone hwp htermVC hstep
  have hdrop : block.body.toList.drop pc.2 = [] :=
    Array.toList_drop_eq_nil_of_getElem?_none hdone
  have hpreTerm : termPre st r :=
    blockSuffixWP'.term_pre_of_drop hdrop hwp
  cases htermEq : block.term with
  | br target =>
      have htermVCBr :
          TerminatorVC cta warp invariants post pc.1 (.br target) termPre := by
        simpa [htermEq] using htermVC
      rcases hbr henv hwarp hlock hrpc hblock htermEq hstep with
        ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
      have hstepBr :
          Helpers.stepTerminator? st cta warp (.br target) = some st' := by
        simpa [htermEq] using hstep
      have hwpTerm :
          wpTerminator cta warp (.br target) (invariants target) st r :=
        TerminatorVC.br_wp htermVCBr st r hpreTerm
      rcases hwpTerm st' hstepBr with ⟨r', hupdate, hinvTarget⟩
      exact ⟨r', hupdate, Or.inl <|
        cfgSuffixInvariant'.of_target_invariant hblocks htarget hinvTarget
          ⟨warpState', henv', hwarp', hlock', hrpc'⟩⟩
  | cbr cond tLabel fLabel =>
      have htermVCCbr :
          TerminatorVC cta warp invariants post pc.1
            (.cbr cond tLabel fLabel) termPre := by
        simpa [htermEq] using htermVC
      have hstepCbr :
          Helpers.stepTerminator? st cta warp (.cbr cond tLabel fLabel) = some st' := by
        simpa [htermEq] using hstep
      cases htermVCCbr with
      | cbr hsplit htrueControl hfalseControl htrue hfalse =>
          have htargetsBlock := htargets hblock
          rw [htermEq] at htargetsBlock
          rcases htargetsBlock with ⟨⟨trueBlock, htrueBlock⟩, falseBlock, hfalseBlock⟩
          have henv' : st'.kernelEnv = env :=
            (Helpers.stepTerminator?_kernelEnv_eq hstep).trans henv
          rcases hsplit st r hpreTerm with hpreTrue | hpreFalse
          · rcases htrueControl hpreTrue hstepCbr with
              ⟨warpState', hwarp', hlock', hrpc'⟩
            have hwpTerm :
                wpTerminator cta warp (.cbr cond tLabel fLabel) (invariants tLabel)
                  st r :=
              htrue st r hpreTrue
            rcases hwpTerm st' hstepCbr with ⟨r', hupdate, hinvTarget⟩
            exact ⟨r', hupdate, Or.inl <|
              cfgSuffixInvariant'.of_target_invariant hblocks htrueBlock hinvTarget
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩⟩
          · rcases hfalseControl hpreFalse hstepCbr with
              ⟨warpState', hwarp', hlock', hrpc'⟩
            have hwpTerm :
                wpTerminator cta warp (.cbr cond tLabel fLabel) (invariants fLabel)
                  st r :=
              hfalse st r hpreFalse
            rcases hwpTerm st' hstepCbr with ⟨r', hupdate, hinvTarget⟩
            exact ⟨r', hupdate, Or.inl <|
              cfgSuffixInvariant'.of_target_invariant hblocks hfalseBlock hinvTarget
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩⟩
  | terminate =>
      have htermVCTerm :
          TerminatorVC cta warp invariants post pc.1 .terminate termPre := by
        simpa [htermEq] using htermVC
      have hstepTerm :
          Helpers.stepTerminator? st cta warp .terminate = some st' := by
        simpa [htermEq] using hstep
      have hwpTerm : wpTerminator cta warp .terminate post st r :=
        TerminatorVC.terminate_wp htermVCTerm st r hpreTerm
      rcases hwpTerm st' hstepTerm with ⟨r', hupdate, hpost⟩
      exact ⟨r', hupdate, Or.inr hpost⟩

theorem TermStepPreservesKernel.of_blockVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block → blockVC cta warp invariants post label block)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp) :
    TermStepPreservesKernel env cta warp invariants post := by
  intro st st' r warpState pc block henv hwarp hlock hrpc hblock hdone hwp hstep
  have hdrop : block.body.toList.drop pc.2 = [] :=
    Array.toList_drop_eq_nil_of_getElem?_none hdone
  cases htermEq : block.term with
  | br target =>
      rcases hbr henv hwarp hlock hrpc hblock htermEq hstep with
        ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
      rcases cfgSuffixInvariant.br_step_of_suffix
          (env := env) (cta := cta) (warp := warp) (invariants := invariants)
          (post := post) (st := st) (st' := st') (r := r) (pc := pc)
          (block := block) (targetBlock := targetBlock) (target := target)
          htermEq hwp hdrop hstep htarget (hblocks target targetBlock htarget)
          ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
        ⟨r', hupdate, hinv⟩
      exact ⟨r', hupdate, Or.inl hinv⟩
  | cbr cond tLabel fLabel =>
      rcases hcbr henv hwarp hlock hrpc hblock htermEq hstep with htrue | hfalse
      · rcases htrue with ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
        rcases cfgSuffixInvariant.cbr_true_step_of_suffix
            (env := env) (cta := cta) (warp := warp) (invariants := invariants)
            (post := post) (st := st) (st' := st') (r := r) (pc := pc)
            (block := block) (targetBlock := targetBlock)
            (cond := cond) (tLabel := tLabel) (fLabel := fLabel)
            htermEq hwp hdrop hstep htarget (hblocks tLabel targetBlock htarget)
            ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
          ⟨r', hupdate, hinv⟩
        exact ⟨r', hupdate, Or.inl hinv⟩
      · rcases hfalse with ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
        rcases cfgSuffixInvariant.cbr_false_step_of_suffix
            (env := env) (cta := cta) (warp := warp) (invariants := invariants)
            (post := post) (st := st) (st' := st') (r := r) (pc := pc)
            (block := block) (targetBlock := targetBlock)
            (cond := cond) (tLabel := tLabel) (fLabel := fLabel)
            htermEq hwp hdrop hstep htarget (hblocks fLabel targetBlock htarget)
            ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
          ⟨r', hupdate, hinv⟩
        exact ⟨r', hupdate, Or.inl hinv⟩
  | terminate =>
      rcases blockSuffixWP.term_step_of_drop hdrop hwp hstep with
        ⟨r', hupdate, hpost⟩
      exact ⟨r', hupdate, Or.inr (by simpa [blockTermPost, htermEq] using hpost)⟩

theorem TermStepPreservesKernel.of_choiceBlockVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {invariants : InvariantMap} {post : CSL.Assertion}
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVCChoice cta warp choices invariants post label block)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices) :
    TermStepPreservesKernelChoice env cta warp choices invariants post := by
  intro st st' r warpState pc block henv hwarp hlock hrpc hblock hdone hwp hstep
  have hdrop : block.body.toList.drop pc.2 = [] :=
    Array.toList_drop_eq_nil_of_getElem?_none hdone
  cases htermEq : block.term with
  | br target =>
      rcases hbr henv hwarp hlock hrpc hblock htermEq hstep with
        ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
      rcases cfgSuffixInvariantChoice.term_step_to_suffix_of_post
          (env := env) (cta := cta) (warp := warp) (choices := choices)
          (invariants := invariants) (post := post) (st := st) (st' := st')
          (r := r) (pc := pc) (targetPc := (target, 0)) (block := block)
          (targetBlock := targetBlock) hwp hdrop hstep htarget
          (by
            intro r' htermPost
            exact blockVCChoice.entry_suffix (hblocks target targetBlock htarget) st' r'
              (by simpa [blockTermPostChoice, htermEq] using htermPost))
          ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
        ⟨r', hupdate, hinv⟩
      exact ⟨r', hupdate, Or.inl hinv⟩
  | cbr cond tLabel fLabel =>
      cases hchoice : choices pc.1 with
      | none =>
          have hcbrResult :=
            hcbr henv hwarp hlock hrpc hblock htermEq hstep
          rw [hchoice] at hcbrResult
          rcases hcbrResult with htrue | hfalse
          · rcases htrue with
              ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
            rcases cfgSuffixInvariantChoice.term_step_to_suffix_of_post
                (env := env) (cta := cta) (warp := warp) (choices := choices)
                (invariants := invariants) (post := post) (st := st) (st' := st')
                (r := r) (pc := pc) (targetPc := (tLabel, 0)) (block := block)
                (targetBlock := targetBlock) hwp hdrop hstep htarget
                (by
                  intro r' htermPost
                  have hboth :
                      invariants tLabel st' r' ∧ invariants fLabel st' r' := by
                    simpa [blockTermPostChoice, htermEq, hchoice] using htermPost
                  exact blockVCChoice.entry_suffix
                    (hblocks tLabel targetBlock htarget) st' r' hboth.1)
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
              ⟨r', hupdate, hinv⟩
            exact ⟨r', hupdate, Or.inl hinv⟩
          · rcases hfalse with
              ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
            rcases cfgSuffixInvariantChoice.term_step_to_suffix_of_post
                (env := env) (cta := cta) (warp := warp) (choices := choices)
                (invariants := invariants) (post := post) (st := st) (st' := st')
                (r := r) (pc := pc) (targetPc := (fLabel, 0)) (block := block)
                (targetBlock := targetBlock) hwp hdrop hstep htarget
                (by
                  intro r' htermPost
                  have hboth :
                      invariants tLabel st' r' ∧ invariants fLabel st' r' := by
                    simpa [blockTermPostChoice, htermEq, hchoice] using htermPost
                  exact blockVCChoice.entry_suffix
                    (hblocks fLabel targetBlock htarget) st' r' hboth.2)
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
              ⟨r', hupdate, hinv⟩
            exact ⟨r', hupdate, Or.inl hinv⟩
      | some choice =>
          cases choice
          · have hcbrResult :=
              hcbr henv hwarp hlock hrpc hblock htermEq hstep
            rw [hchoice] at hcbrResult
            rcases hcbrResult with
              ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
            rcases cfgSuffixInvariantChoice.term_step_to_suffix_of_post
                (env := env) (cta := cta) (warp := warp) (choices := choices)
                (invariants := invariants) (post := post) (st := st) (st' := st')
                (r := r) (pc := pc) (targetPc := (fLabel, 0)) (block := block)
                (targetBlock := targetBlock) hwp hdrop hstep htarget
                (by
                  intro r' htermPost
                  exact blockVCChoice.entry_suffix
                    (hblocks fLabel targetBlock htarget) st' r'
                    (by simpa [blockTermPostChoice, htermEq, hchoice] using htermPost))
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
              ⟨r', hupdate, hinv⟩
            exact ⟨r', hupdate, Or.inl hinv⟩
          · have hcbrResult :=
              hcbr henv hwarp hlock hrpc hblock htermEq hstep
            rw [hchoice] at hcbrResult
            rcases hcbrResult with
              ⟨targetBlock, warpState', htarget, henv', hwarp', hlock', hrpc'⟩
            rcases cfgSuffixInvariantChoice.term_step_to_suffix_of_post
                (env := env) (cta := cta) (warp := warp) (choices := choices)
                (invariants := invariants) (post := post) (st := st) (st' := st')
                (r := r) (pc := pc) (targetPc := (tLabel, 0)) (block := block)
                (targetBlock := targetBlock) hwp hdrop hstep htarget
                (by
                  intro r' htermPost
                  exact blockVCChoice.entry_suffix
                    (hblocks tLabel targetBlock htarget) st' r'
                    (by simpa [blockTermPostChoice, htermEq, hchoice] using htermPost))
                ⟨warpState', henv', hwarp', hlock', hrpc'⟩ with
              ⟨r', hupdate, hinv⟩
            exact ⟨r', hupdate, Or.inl hinv⟩
  | terminate =>
      rcases blockSuffixWPChoice.term_step_of_drop hdrop hwp hstep with
        ⟨r', hupdate, hpost⟩
      exact ⟨r', hupdate,
        Or.inr (by simpa [blockTermPostChoice, htermEq] using hpost)⟩

theorem TermStepPreservesKernel.of_cfg
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hterm : TermStepPreservesCFG env cta warp invariants post) :
    TermStepPreservesKernel env cta warp invariants post := by
  intro st st' r warpState pc block henv hwarp hlock hrpc hblock hdone hwp hstep
  rcases hterm henv hwarp hlock hrpc hblock hdone hwp hstep with
    ⟨r', hupdate, hinv⟩
  exact ⟨r', hupdate, Or.inl hinv⟩

theorem StepBlockPreserves.of_cfgSuffixInvariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants post) :
    StepBlockPreserves cta warp (cfgSuffixInvariant env cta warp invariants post) := by
  intro st r st' hinv hstep
  rcases hinv with
    ⟨warpStateInv, pcInv, blockInv, henv, hwarpInv, hlockInv, hrpcInv, hblockInv, hwp⟩
  cases hstep with
  | body hwf hwarp hwfWarp hlock hrpc hblock hgi hinstr =>
      rename_i warpState pc block gi
      have hwarpEq : warpState = warpStateInv :=
        Option.some.inj (by
          rw [← hwarp]
          exact hwarpInv)
      subst warpState
      have hpcEq : pc = pcInv :=
        Option.some.inj (by
          change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
          change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
          rw [← hrpc]
          exact hrpcInv)
      subst pc
      have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
      have hblockEq : block = blockInv :=
        Option.some.inj (by
          rw [← hblockEnv]
          exact hblockInv)
      subst block
      rcases Array.toList_drop_eq_cons_of_getElem?_some hgi with ⟨rest, hdrop⟩
      cases hinstr with
      | mk hwfInstr hwarpInstr hwfWarpInstr hlockInstr hpart hrun =>
          exact cfgSuffixInvariant.body_step_of_suffix hwp hdrop hblockInv hrun
            (hbody henv hwarpInv hlockInv hrpcInv hblockInv hgi hrun)
  | term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hrun =>
      rename_i warpState pc block
      have hwarpEq : warpState = warpStateInv :=
        Option.some.inj (by
          rw [← hwarp]
          exact hwarpInv)
      subst warpState
      have hpcEq : pc = pcInv :=
        Option.some.inj (by
          change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
          change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
          rw [← hrpc]
          exact hrpcInv)
      subst pc
      have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
      have hblockEq : block = blockInv :=
        Option.some.inj (by
          rw [← hblockEnv]
          exact hblockInv)
      subst block
      exact hterm henv hwarpInv hlockInv hrpcInv hblockInv hbodyDone hwp hrun

theorem StepBlockPreserves.of_cfgKernelInvariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepBlockPreserves cta warp (cfgKernelInvariant env cta warp invariants post) := by
  intro st r st' hinv hstep
  rcases hinv with hinv | hpostSt
  · rcases hinv with
      ⟨warpStateInv, pcInv, blockInv, henv, hwarpInv, hlockInv, hrpcInv, hblockInv, hwp⟩
    cases hstep with
    | body hwf hwarp hwfWarp hlock hrpc hblock hgi hinstr =>
        rename_i warpState pc block gi
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        rcases Array.toList_drop_eq_cons_of_getElem?_some hgi with ⟨rest, hdrop⟩
        cases hinstr with
        | mk hwfInstr hwarpInstr hwfWarpInstr hlockInstr hpart hrun =>
            rcases cfgSuffixInvariant.body_step_of_suffix hwp hdrop hblockInv hrun
                (hbody henv hwarpInv hlockInv hrpcInv hblockInv hgi hrun) with
              ⟨r', hupdate, hinv'⟩
            exact ⟨r', hupdate, Or.inl hinv'⟩
    | term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hrun =>
        rename_i warpState pc block
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        exact hterm henv hwarpInv hlockInv hrpcInv hblockInv hbodyDone hwp hrun
  · rcases hpost st r st' hpostSt hstep with ⟨r', hupdate, hpost'⟩
    exact ⟨r', hupdate, Or.inr hpost'⟩

theorem StepBlockPreserves.of_cfgKernelInvariant'
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel' env cta warp invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepBlockPreserves cta warp (cfgKernelInvariant' env cta warp invariants post) := by
  intro st r st' hinv hstep
  rcases hinv with hinv | hpostSt
  · rcases hinv with
      ⟨warpStateInv, pcInv, blockInv, termPre, henv, hwarpInv, hlockInv, hrpcInv,
        hblockInv, hwp, htermVC⟩
    cases hstep with
    | body hwf hwarp hwfWarp hlock hrpc hblock hgi hinstr =>
        rename_i warpState pc block gi
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        rcases Array.toList_drop_eq_cons_of_getElem?_some hgi with ⟨rest, hdrop⟩
        cases hinstr with
        | mk hwfInstr hwarpInstr hwfWarpInstr hlockInstr hpart hrun =>
            rcases cfgSuffixInvariant'.body_step_of_suffix hwp htermVC hdrop hblockInv hrun
                (hbody henv hwarpInv hlockInv hrpcInv hblockInv hgi hrun) with
              ⟨r', hupdate, hinv'⟩
            exact ⟨r', hupdate, Or.inl hinv'⟩
    | term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hrun =>
        rename_i warpState pc block
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        exact hterm henv hwarpInv hlockInv hrpcInv hblockInv hbodyDone hwp htermVC hrun
  · rcases hpost st r st' hpostSt hstep with ⟨r', hupdate, hpost'⟩
    exact ⟨r', hupdate, Or.inr hpost'⟩

theorem StepBlockPreserves.of_cfgKernelInvariantChoice
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernelChoice env cta warp choices invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepBlockPreserves cta warp
      (cfgKernelInvariantChoice env cta warp choices invariants post) := by
  intro st r st' hinv hstep
  rcases hinv with hinv | hpostSt
  · rcases hinv with
      ⟨warpStateInv, pcInv, blockInv, henv, hwarpInv, hlockInv, hrpcInv, hblockInv, hwp⟩
    cases hstep with
    | body hwf hwarp hwfWarp hlock hrpc hblock hgi hinstr =>
        rename_i warpState pc block gi
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        rcases Array.toList_drop_eq_cons_of_getElem?_some hgi with ⟨rest, hdrop⟩
        cases hinstr with
        | mk hwfInstr hwarpInstr hwfWarpInstr hlockInstr hpart hrun =>
            rcases cfgSuffixInvariantChoice.body_step_of_suffix hwp hdrop hblockInv hrun
                (hbody henv hwarpInv hlockInv hrpcInv hblockInv hgi hrun) with
              ⟨r', hupdate, hinv'⟩
            exact ⟨r', hupdate, Or.inl hinv'⟩
    | term hwf hwarp hwfWarp hlock hrpc hblock hbodyDone hrun =>
        rename_i warpState pc block
        have hwarpEq : warpState = warpStateInv :=
          Option.some.inj (by
            rw [← hwarp]
            exact hwarpInv)
        subst warpState
        have hpcEq : pc = pcInv :=
          Option.some.inj (by
            change Helpers.currentRunnablePc? warpStateInv = some pc at hrpc
            change Helpers.currentRunnablePc? warpStateInv = some pcInv at hrpcInv
            rw [← hrpc]
            exact hrpcInv)
        subst pc
        have hblockEnv : env.blocks[pcInv.1]? = some block := henv ▸ hblock
        have hblockEq : block = blockInv :=
          Option.some.inj (by
            rw [← hblockEnv]
            exact hblockInv)
        subst block
        exact hterm henv hwarpInv hlockInv hrpcInv hblockInv hbodyDone hwp hrun
  · rcases hpost st r st' hpostSt hstep with ⟨r', hupdate, hpost'⟩
    exact ⟨r', hupdate, Or.inr hpost'⟩

theorem StepPreserves.of_cfgSuffixInvariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hselect :
      StepMachineSelects (cfgSuffixInvariant env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants post) :
    StepPreserves (cfgSuffixInvariant env cta warp invariants post) :=
  StepPreserves.of_stepWarpPreserves hselect <|
    StepWarpPreserves.of_stepBlockPreserves <|
      StepBlockPreserves.of_cfgSuffixInvariant hbody hterm

theorem StepPreserves.of_cfgKernelInvariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_stepWarpPreserves hselect <|
    StepWarpPreserves.of_stepBlockPreserves <|
      StepBlockPreserves.of_cfgKernelInvariant hbody hterm hpost

theorem StepPreserves.of_cfgKernelInvariant'
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel' env cta warp invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant' env cta warp invariants post) :=
  StepPreserves.of_stepWarpPreserves hselect <|
    StepWarpPreserves.of_stepBlockPreserves <|
      StepBlockPreserves.of_cfgKernelInvariant' hbody hterm hpost

theorem StepPreserves.of_cfgKernelInvariantChoice
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hselect :
      StepMachineSelects (cfgKernelInvariantChoice env cta warp choices invariants post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernelChoice env cta warp choices invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariantChoice env cta warp choices invariants post) :=
  StepPreserves.of_stepWarpPreserves hselect <|
    StepWarpPreserves.of_stepBlockPreserves <|
      StepBlockPreserves.of_cfgKernelInvariantChoice hbody hterm hpost

def kernelVCs
    (env : KernelEnv) (cta : CTAId) (warp : WarpId)
    (pre post : CSL.Assertion) (invariants : InvariantMap) : Prop :=
  pre ⊢ₛ invariants env.entry ∧
    (∀ label block,
      env.blocks[label]? = some block → blockVC cta warp invariants post label block) ∧
    (∀ label, Finalizes (invariants label) post)

def kernelVCsChoice
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (choices : CbrChoiceMap)
    (pre post : CSL.Assertion) (invariants : InvariantMap) : Prop :=
  pre ⊢ₛ invariants env.entry ∧
    (∀ label block,
      env.blocks[label]? = some block →
        blockVCChoice cta warp choices invariants post label block) ∧
    (∀ label, Finalizes (invariants label) post)

theorem TermStepPreservesKernel.of_kernelVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp) :
    TermStepPreservesKernel env cta warp invariants post := by
  rcases hvc with ⟨_hpreEntry, hblocks, _hfinal⟩
  exact TermStepPreservesKernel.of_blockVCs hblocks hbr hcbr

theorem TermStepPreservesKernel'.of_kernelVCs'
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs' env cta warp pre post invariants)
    (hbr : BrTermControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env) :
    TermStepPreservesKernel' env cta warp invariants post := by
  rcases hvc with ⟨_hpreEntry, hblocks, _hfinal⟩
  exact TermStepPreservesKernel'.of_blockVCs hblocks hbr htargets

theorem TermStepPreservesKernelChoice.of_kernelVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCsChoice env cta warp choices pre post invariants)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices) :
    TermStepPreservesKernelChoice env cta warp choices invariants post := by
  rcases hvc with ⟨_hpreEntry, hblocks, _hfinal⟩
  exact TermStepPreservesKernel.of_choiceBlockVCs hblocks hbr hcbr

theorem TermStepPreservesKernel.of_kernelVCs_targets
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp) :
    TermStepPreservesKernel env cta warp invariants post :=
  TermStepPreservesKernel.of_kernelVCs hvc
    (BrTermControl.of_targets_semantic htargets hbr)
    (CbrTermControl.of_targets_semantic htargets hcbr)

theorem StepPreserves.of_kernelVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_cfgKernelInvariant hselect hbody
    (TermStepPreservesKernel.of_kernelVCs hvc hbr hcbr) hpost

theorem StepPreserves.of_kernelVCs'
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs' env cta warp pre post invariants)
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant' env cta warp invariants post) :=
  StepPreserves.of_cfgKernelInvariant' hselect hbody
    (TermStepPreservesKernel'.of_kernelVCs' hvc hbr htargets) hpost

theorem StepPreserves.of_kernelVCs_targets
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_kernelVCs hvc hselect hbody
    (BrTermControl.of_targets_semantic htargets hbr)
    (CbrTermControl.of_targets_semantic htargets hcbr)
    hpost

theorem StepPreserves.of_choiceKernelVCs
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCsChoice env cta warp choices pre post invariants)
    (hselect :
      StepMachineSelects (cfgKernelInvariantChoice env cta warp choices invariants post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariantChoice env cta warp choices invariants post) :=
  StepPreserves.of_cfgKernelInvariantChoice hselect hbody
    (TermStepPreservesKernelChoice.of_kernelVCs hvc hbr hcbr) hpost

def EntryReady
    (env : KernelEnv) (cta : CTAId) (warp : WarpId) (pre : CSL.Assertion) : Prop :=
  ∀ {st : State} {r : CSL.Resource},
    pre st r →
      ∃ warpState block,
        st.kernelEnv = env ∧
          st.getWarp? cta warp = some warpState ∧
          Helpers.lockstepRunnable warpState ∧
          Helpers.RunnablePc warpState (env.entry, 0) ∧
          env.blocks[env.entry]? = some block

theorem cfgSuffixInvariant.of_entry
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs env cta warp pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgSuffixInvariant env cta warp invariants post st r := by
  rcases hvc with ⟨hpreEntry, hblocks, _hfinal⟩
  have hinvEntry : invariants env.entry st r := hpreEntry st r hpre
  have hentryWP : blockSuffixWP cta warp invariants post block 0 st r :=
    blockVC.entry_suffix (hblocks env.entry block hentry) st r hinvEntry
  exact ⟨warpState, (env.entry, 0), block, henv, hwarp, hlock, hrpc, hentry, hentryWP⟩

theorem cfgSuffixInvariant'.of_entry
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs' env cta warp pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgSuffixInvariant' env cta warp invariants post st r := by
  rcases hvc with ⟨hpreEntry, hblocks, _hfinal⟩
  rcases hblocks env.entry block hentry with ⟨termPre, hbody, htermVC⟩
  have hinvEntry : invariants env.entry st r := hpreEntry st r hpre
  have hentryWP : blockSuffixWP' cta warp termPre block 0 st r :=
    blockVC'.entry_suffix (invariants := invariants) hbody st r hinvEntry
  exact ⟨warpState, (env.entry, 0), block, termPre, henv, hwarp, hlock, hrpc,
    hentry, hentryWP, htermVC⟩

theorem cfgKernelInvariant.of_entry
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs env cta warp pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgKernelInvariant env cta warp invariants post st r :=
  Or.inl <| cfgSuffixInvariant.of_entry hvc henv hwarp hlock hrpc hentry hpre

theorem cfgKernelInvariant'.of_entry
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs' env cta warp pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgKernelInvariant' env cta warp invariants post st r :=
  Or.inl <| cfgSuffixInvariant'.of_entry hvc henv hwarp hlock hrpc hentry hpre

theorem cfgKernelInvariantChoice.of_entry
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCsChoice env cta warp choices pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgKernelInvariantChoice env cta warp choices invariants post st r := by
  rcases hvc with ⟨hpreEntry, hblocks, _hfinal⟩
  have hinvEntry : invariants env.entry st r := hpreEntry st r hpre
  have hentryWP :
      blockSuffixWPChoice cta warp choices invariants post env.entry block 0 st r :=
    blockVCChoice.entry_suffix (hblocks env.entry block hentry) st r hinvEntry
  exact Or.inl
    ⟨warpState, (env.entry, 0), block, henv, hwarp, hlock, hrpc, hentry, hentryWP⟩

def KernelSpec.Valid (spec : KernelSpec) : Prop :=
  spec.pre spec.init spec.resource ∧
    spec.pre ⊢ₛ spec.invariant ∧
    StepPreserves spec.invariant ∧
    Finalizes spec.invariant spec.post

theorem KernelSpec.Valid.of_parts {spec : KernelSpec}
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv : spec.pre ⊢ₛ spec.invariant)
    (hstep : StepPreserves spec.invariant)
    (hfinal : Finalizes spec.invariant spec.post) :
    spec.Valid :=
  ⟨hpre, hpreInv, hstep, hfinal⟩

theorem KernelSpec.Valid.of_cfg_controls
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgSuffixInvariant env cta warp invariants spec.post)
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv :
      spec.pre ⊢ₛ cfgSuffixInvariant env cta warp invariants spec.post)
    (hselect :
      StepMachineSelects (cfgSuffixInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants spec.post)
    (hfinal : Finalizes (cfgSuffixInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid := by
  refine KernelSpec.Valid.of_parts hpre ?_ ?_ ?_
  · intro st r hpreSt
    rw [hinvariant]
    exact hpreInv st r hpreSt
  · rw [hinvariant]
    exact StepPreserves.of_cfgSuffixInvariant hselect hbody hterm
  · rw [hinvariant]
    exact hfinal

theorem KernelSpec.Valid.of_cfg_kernel_controls
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv :
      spec.pre ⊢ₛ cfgKernelInvariant env cta warp invariants spec.post)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid := by
  refine KernelSpec.Valid.of_parts hpre ?_ ?_ ?_
  · intro st r hpreSt
    rw [hinvariant]
    exact hpreInv st r hpreSt
  · rw [hinvariant]
    exact StepPreserves.of_cfgKernelInvariant hselect hbody hterm hpost
  · rw [hinvariant]
    exact hfinal

theorem KernelSpec.Valid.of_cfg_kernel_controls'
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant' env cta warp invariants spec.post)
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv :
      spec.pre ⊢ₛ cfgKernelInvariant' env cta warp invariants spec.post)
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel' env cta warp invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant' env cta warp invariants spec.post) spec.post) :
    spec.Valid := by
  refine KernelSpec.Valid.of_parts hpre ?_ ?_ ?_
  · intro st r hpreSt
    rw [hinvariant]
    exact hpreInv st r hpreSt
  · rw [hinvariant]
    exact StepPreserves.of_cfgKernelInvariant' hselect hbody hterm hpost
  · rw [hinvariant]
    exact hfinal

theorem KernelSpec.Valid.of_cfg_kernel_choice_controls
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariantChoice env cta warp choices invariants spec.post)
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv :
      spec.pre ⊢ₛ cfgKernelInvariantChoice env cta warp choices invariants spec.post)
    (hselect :
      StepMachineSelects (cfgKernelInvariantChoice env cta warp choices invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernelChoice env cta warp choices invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal :
      Finalizes (cfgKernelInvariantChoice env cta warp choices invariants spec.post)
        spec.post) :
    spec.Valid := by
  refine KernelSpec.Valid.of_parts hpre ?_ ?_ ?_
  · intro st r hpreSt
    rw [hinvariant]
    exact hpreInv st r hpreSt
  · rw [hinvariant]
    exact StepPreserves.of_cfgKernelInvariantChoice hselect hbody hterm hpost
  · rw [hinvariant]
    exact hfinal

theorem KernelSpec.Valid.of_entry_blockVCs
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC cta warp invariants spec.post label block)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_kernel_controls
    hinvariant
    hpre
    (by
      intro st r hpreSt
      rcases hentryReady hpreSt with
        ⟨warpState, block, henv, hwarp, hlock, hrpc, hentry⟩
      have hinvEntry : invariants env.entry st r := hpreEntry st r hpreSt
      have hentryWP :
          blockSuffixWP cta warp invariants spec.post block 0 st r :=
        blockVC.entry_suffix (hblocks env.entry block hentry) st r hinvEntry
      exact Or.inl
        ⟨warpState, (env.entry, 0), block, henv, hwarp, hlock, hrpc, hentry,
          hentryWP⟩)
    hselect
    hbody
    (TermStepPreservesKernel.of_blockVCs hblocks hbr hcbr)
    hpost
    hfinal

theorem KernelSpec.Valid.of_entry_blockVCs'
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant' env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC' cta warp invariants spec.post label block)
    (hbr : BrTermControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant' env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_kernel_controls'
    hinvariant
    hpre
    (by
      intro st r hpreSt
      rcases hentryReady hpreSt with
        ⟨warpState, block, henv, hwarp, hlock, hrpc, hentry⟩
      have hinvEntry : invariants env.entry st r := hpreEntry st r hpreSt
      rcases hblocks env.entry block hentry with ⟨termPre, hbodyEntry, htermVC⟩
      have hentryWP :
          blockSuffixWP' cta warp termPre block 0 st r :=
        blockVC'.entry_suffix (invariants := invariants) hbodyEntry st r hinvEntry
      exact Or.inl
        ⟨warpState, (env.entry, 0), block, termPre, henv, hwarp, hlock, hrpc,
          hentry, hentryWP, htermVC⟩)
    hselect
    hbody
    (TermStepPreservesKernel'.of_blockVCs hblocks hbr htargets)
    hpost
    hfinal

theorem KernelSpec.Valid.of_entry_blockVCs_targets
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC cta warp invariants spec.post label block)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_entry_blockVCs
    hinvariant
    hpreEntry
    hentryReady
    hpre
    hselect
    hbody
    hblocks
    (BrTermControl.of_targets_semantic htargets hbr)
    (CbrTermControl.of_targets_semantic htargets hcbr)
    hpost
    hfinal

theorem KernelSpec.Valid.of_entry_blockVCs_closed
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC cta warp invariants spec.post label block)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp)
    (hpostNoStep : NoStepBlock cta warp spec.post)
    (hsuffixNoFinal : NoFinal (cfgSuffixInvariant env cta warp invariants spec.post)) :
    spec.Valid :=
  KernelSpec.Valid.of_entry_blockVCs
    hinvariant
    hpreEntry
    hentryReady
    hpre
    hselect
    hbody
    hblocks
    hbr
    hcbr
    (StepBlockPreserves.of_no_step hpostNoStep)
    (by
      intro final r hfinal hinv
      rcases hinv with hsuffix | hpost
      · exact False.elim (hsuffixNoFinal final r hfinal hsuffix)
      · exact hpost)

theorem KernelSpec.Valid.of_entry_blockVCs'_closed
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant' env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC' cta warp invariants spec.post label block)
    (hbr : BrTermControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hpostNoStep : NoStepBlock cta warp spec.post)
    (hsuffixNoFinal : NoFinal (cfgSuffixInvariant' env cta warp invariants spec.post)) :
    spec.Valid :=
  KernelSpec.Valid.of_entry_blockVCs'
    hinvariant
    hpreEntry
    hentryReady
    hpre
    hselect
    hbody
    hblocks
    hbr
    htargets
    (StepBlockPreserves.of_no_step hpostNoStep)
    (by
      intro final r hfinal hinv
      rcases hinv with hsuffix | hpost
      · exact False.elim (hsuffixNoFinal final r hfinal hsuffix)
      · exact hpost)

theorem KernelSpec.Valid.of_entry_blockVCs_targets_closed
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hblocks :
      ∀ label block,
        env.blocks[label]? = some block →
          blockVC cta warp invariants spec.post label block)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpostNoStep : NoStepBlock cta warp spec.post)
    (hsuffixNoFinal : NoFinal (cfgSuffixInvariant env cta warp invariants spec.post)) :
    spec.Valid :=
  KernelSpec.Valid.of_entry_blockVCs_closed
    hinvariant
    hpreEntry
    hentryReady
    hpre
    hselect
    hbody
    hblocks
    (BrTermControl.of_targets_semantic htargets hbr)
    (CbrTermControl.of_targets_semantic htargets hcbr)
    hpostNoStep
    hsuffixNoFinal

theorem KernelSpec.Valid.of_kernelVCs
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hvc : kernelVCs env cta warp spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_kernel_controls
    hinvariant
    hpre
    (by
      intro st r hpreSt
      rcases hentryReady hpreSt with
        ⟨warpState, block, henv, hwarp, hlock, hrpc, hentry⟩
      exact cfgKernelInvariant.of_entry hvc henv hwarp hlock hrpc hentry hpreSt)
    hselect
    hbody
    hterm
    hpost
    hfinal

theorem KernelSpec.Valid.of_kernelVCs'
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant' env cta warp invariants spec.post)
    (hvc : kernelVCs' env cta warp spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant' env cta warp invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel' env cta warp invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant' env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_kernel_controls'
    hinvariant
    hpre
    (by
      intro st r hpreSt
      rcases hentryReady hpreSt with
        ⟨warpState, block, henv, hwarp, hlock, hrpc, hentry⟩
      exact cfgKernelInvariant'.of_entry hvc henv hwarp hlock hrpc hentry hpreSt)
    hselect
    hbody
    hterm
    hpost
    hfinal

theorem KernelSpec.Valid.of_kernelVCs_targets
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hvc : kernelVCs env cta warp spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariant env cta warp invariants spec.post) cta warp)
    (hbody : BodyStepControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_kernelVCs
    hinvariant
    hvc
    hentryReady
    hpre
    hselect
    hbody
    (TermStepPreservesKernel.of_kernelVCs_targets hvc htargets hbr hcbr)
    hpost
    hfinal

theorem KernelSpec.Valid.of_choiceKernelVCs
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariantChoice env cta warp choices invariants spec.post)
    (hvc : kernelVCsChoice env cta warp choices spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (hselect :
      StepMachineSelects (cfgKernelInvariantChoice env cta warp choices invariants spec.post)
        cta warp)
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal :
      Finalizes (cfgKernelInvariantChoice env cta warp choices invariants spec.post)
        spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_kernel_choice_controls
    hinvariant
    hpre
    (by
      intro st r hpreSt
      rcases hentryReady hpreSt with
        ⟨warpState, block, henv, hwarp, hlock, hrpc, hentry⟩
      exact cfgKernelInvariantChoice.of_entry hvc henv hwarp hlock hrpc hentry hpreSt)
    hselect
    hbody
    (TermStepPreservesKernelChoice.of_kernelVCs hvc hbr hcbr)
    hpost
    hfinal

theorem Finalizes.of_cfgKernelInvariant
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hsuffix : Finalizes (cfgSuffixInvariant env cta warp invariants post) post) :
    Finalizes (cfgKernelInvariant env cta warp invariants post) post := by
  intro final r hfinal hinv
  rcases hinv with hsuffixInv | hpost
  · exact hsuffix final r hfinal hsuffixInv
  · exact hpost

theorem Finalizes.of_cfgKernelInvariant'
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hsuffix : Finalizes (cfgSuffixInvariant' env cta warp invariants post) post) :
    Finalizes (cfgKernelInvariant' env cta warp invariants post) post := by
  intro final r hfinal hinv
  rcases hinv with hsuffixInv | hpost
  · exact hsuffix final r hfinal hsuffixInv
  · exact hpost

theorem Finalizes.of_cfgKernelInvariantChoice
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hsuffix :
      Finalizes (cfgSuffixInvariantChoice env cta warp choices invariants post) post) :
    Finalizes (cfgKernelInvariantChoice env cta warp choices invariants post) post := by
  intro final r hfinal hinv
  rcases hinv with hsuffixInv | hpost
  · exact hsuffix final r hfinal hsuffixInv
  · exact hpost

namespace Reaches

theorem preserves_assertion
    {inv : CSL.Assertion} {r : CSL.Resource} {st final : State}
    (hstep : StepPreserves inv)
    (hreach : Reaches st final)
    (hinit : inv st r) :
    ∃ rFinal, CSL.Resource.Update r rFinal ∧ inv final rFinal := by
  induction hreach generalizing r with
  | refl =>
      exact ⟨r, CSL.Resource.update_refl r, hinit⟩
  | step hmachine _ ih =>
      rcases hstep _ _ _ hinit hmachine with ⟨rNext, hupdate, hinvNext⟩
      rcases ih hinvNext with ⟨rFinal, hupdateFinal, hinvFinal⟩
      exact ⟨rFinal, CSL.Resource.update_trans hupdate hupdateFinal, hinvFinal⟩

end Reaches

theorem KernelSpec.partial_correct {spec : KernelSpec}
    (hvalid : spec.Valid) :
    PartialCorrect spec.init (fun st => ∃ r, spec.post st r) := by
  intro final hterm
  rcases hvalid with ⟨hpre, hpreInv, hstep, hfinal⟩
  rcases Reaches.preserves_assertion hstep hterm.1
    (hpreInv spec.init spec.resource hpre) with ⟨rFinal, _, hinvFinal⟩
  exact ⟨rFinal, hfinal final rFinal hterm.2 hinvFinal⟩

end WP
end CLean
