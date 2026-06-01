import CLean.WP

namespace CLean
namespace Examples

open WP

def branchLoopEntryBlock : Block :=
  { label := "entry"
    body := #[]
    term := .br "loop" }

def branchLoopHeaderBlock : Block :=
  { label := "loop"
    body := #[]
    term := .cbr (.pred "p") "body" "exit" }

def branchLoopCbrTerm : Terminator :=
  .cbr (.pred "p") "body" "exit"

def branchLoopBodyBlock : Block :=
  { label := "body"
    body := #[]
    term := .br "loop" }

def branchLoopExitBlock : Block :=
  { label := "exit"
    body := #[]
    term := .terminate }

def branchLoopEnv : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := 1 } }
    blocks :=
      ((({} : Std.HashMap BlockLabel Block).insert "entry" branchLoopEntryBlock).insert
        "loop" branchLoopHeaderBlock).insert "body" branchLoopBodyBlock |>.insert
        "exit" branchLoopExitBlock }

def branchLoopTruth : CSL.Assertion :=
  fun _ _ => True

def branchLoopPost : CSL.Assertion :=
  branchLoopTruth

def branchLoopInvariants : InvariantMap :=
  fun _ => branchLoopTruth

def branchLoopEntryPre : CSL.Assertion :=
  fun st _ =>
    ∃ warpState,
      st.kernelEnv = branchLoopEnv ∧
        st.getWarp? 0 0 = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState ("entry", 0)

def branchLoopKernelSpec (init : State) (resource : CSL.Resource) : KernelSpec :=
  { init := init
    resource := resource
    pre := branchLoopEntryPre
    invariant := cfgKernelInvariant' branchLoopEnv 0 0 branchLoopInvariants branchLoopPost
    post := branchLoopPost }

theorem branchLoop_entry_lookup :
    branchLoopEnv.blocks["entry"]? = some branchLoopEntryBlock := by
  rw [branchLoopEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem branchLoop_loop_lookup :
    branchLoopEnv.blocks["loop"]? = some branchLoopHeaderBlock := by
  rw [branchLoopEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem branchLoop_body_lookup :
    branchLoopEnv.blocks["body"]? = some branchLoopBodyBlock := by
  rw [branchLoopEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem branchLoop_exit_lookup :
    branchLoopEnv.blocks["exit"]? = some branchLoopExitBlock := by
  rw [branchLoopEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem branchLoop_block_lookup
    {label : BlockLabel} {block : Block}
    (hlookup : branchLoopEnv.blocks[label]? = some block) :
    (label = "entry" ∧ block = branchLoopEntryBlock) ∨
      (label = "loop" ∧ block = branchLoopHeaderBlock) ∨
        (label = "body" ∧ block = branchLoopBodyBlock) ∨
          (label = "exit" ∧ block = branchLoopExitBlock) := by
  by_cases hentry : label = "entry"
  · subst label
    rw [branchLoopEnv] at hlookup
    repeat rw [Std.HashMap.getElem?_insert] at hlookup
    simp at hlookup
    exact Or.inl ⟨rfl, hlookup.symm⟩
  · by_cases hloop : label = "loop"
    · subst label
      rw [branchLoopEnv] at hlookup
      repeat rw [Std.HashMap.getElem?_insert] at hlookup
      simp at hlookup
      exact Or.inr (Or.inl ⟨rfl, hlookup.symm⟩)
    · by_cases hbody : label = "body"
      · subst label
        rw [branchLoopEnv] at hlookup
        repeat rw [Std.HashMap.getElem?_insert] at hlookup
        simp at hlookup
        exact Or.inr (Or.inr (Or.inl ⟨rfl, hlookup.symm⟩))
      · by_cases hexit : label = "exit"
        · subst label
          rw [branchLoopEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp at hlookup
          exact Or.inr (Or.inr (Or.inr ⟨rfl, hlookup.symm⟩))
        · have hentryEq : ¬ "entry" = label := fun h => hentry h.symm
          have hloopEq : ¬ "loop" = label := fun h => hloop h.symm
          have hbodyEq : ¬ "body" = label := fun h => hbody h.symm
          have hexitEq : ¬ "exit" = label := fun h => hexit h.symm
          rw [branchLoopEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp [hentryEq, hloopEq, hbodyEq, hexitEq] at hlookup

theorem branchLoop_body_ordinary :
    CFGBodyUsesOrdinaryPcAdvance branchLoopEnv := by
  intro label block idx gi hlookup hgi
  rcases branchLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    simp [branchLoopEntryBlock] at hgi
  · rcases hloop with ⟨_, hblock⟩
    subst block
    simp [branchLoopHeaderBlock] at hgi
  · rcases hbody with ⟨_, hblock⟩
    subst block
    simp [branchLoopBodyBlock] at hgi
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [branchLoopExitBlock] at hgi

theorem branchLoop_targets_exist :
    CFGTerminatorTargetsExist branchLoopEnv := by
  intro label block hlookup
  rcases branchLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    simp [branchLoopEntryBlock]
    exact ⟨branchLoopHeaderBlock, branchLoop_loop_lookup⟩
  · rcases hloop with ⟨_, hblock⟩
    subst block
    simp [branchLoopHeaderBlock]
    exact ⟨⟨branchLoopBodyBlock, branchLoop_body_lookup⟩,
      branchLoopExitBlock, branchLoop_exit_lookup⟩
  · rcases hbody with ⟨_, hblock⟩
    subst block
    simp [branchLoopBodyBlock]
    exact ⟨branchLoopHeaderBlock, branchLoop_loop_lookup⟩
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [branchLoopExitBlock]

theorem branchLoop_wpTerminator_truth (term : Terminator) :
    branchLoopTruth ⊢ₛ wpTerminator 0 0 term branchLoopTruth := by
  intro st r _ st' _hstep
  exact ⟨r, CSL.Resource.update_refl r, trivial⟩

theorem branchLoop_br_vc (label target : BlockLabel) :
    blockVC' 0 0 branchLoopInvariants branchLoopPost label
      { label := label, body := #[], term := .br target } := by
  refine ⟨wpTerminator 0 0 (.br target) branchLoopTruth, ?_, ?_⟩
  · simpa [branchLoopInvariants, branchLoopPost, wpInstrs] using
      branchLoop_wpTerminator_truth (.br target)
  · exact TerminatorVC.br (by
      intro st r hpre
      simpa [branchLoopInvariants, branchLoopPost] using hpre)

theorem branchLoop_cbr_vc :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth) →
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth) →
    blockVC' 0 0 branchLoopInvariants branchLoopPost "loop" branchLoopHeaderBlock := by
  intro htrueControl hfalseControl
  refine ⟨wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth, ?_, ?_⟩
  · simpa [branchLoopHeaderBlock, branchLoopInvariants, branchLoopPost, wpInstrs] using
      branchLoop_wpTerminator_truth branchLoopCbrTerm
  · refine TerminatorVC.cbr
      (truePre := wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth)
      (falsePre := wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth)
      ?_ htrueControl hfalseControl ?_ ?_
    · intro st r hpre
      exact Or.inl hpre
    · intro st r hpre
      simpa [branchLoopCbrTerm, branchLoopInvariants, branchLoopPost] using hpre
    · intro st r hpre
      simpa [branchLoopCbrTerm, branchLoopInvariants, branchLoopPost] using hpre

theorem branchLoop_exit_vc :
    blockVC' 0 0 branchLoopInvariants branchLoopPost "exit" branchLoopExitBlock := by
  refine ⟨wpTerminator 0 0 .terminate branchLoopTruth, ?_, ?_⟩
  · simpa [branchLoopExitBlock, branchLoopInvariants, branchLoopPost, wpInstrs] using
      branchLoop_wpTerminator_truth .terminate
  · exact TerminatorVC.terminate (by
      intro st r hpre
      simpa [branchLoopPost] using hpre)

theorem branchLoop_block_vcs :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth) →
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth) →
    ∀ label block,
      branchLoopEnv.blocks[label]? = some block →
        blockVC' 0 0 branchLoopInvariants branchLoopPost label block := by
  intro htrueControl hfalseControl
  intro label block hlookup
  rcases branchLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨hlabel, hblock⟩
    subst label
    subst block
    simpa [branchLoopEntryBlock] using branchLoop_br_vc "entry" "loop"
  · rcases hloop with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact branchLoop_cbr_vc htrueControl hfalseControl
  · rcases hbody with ⟨hlabel, hblock⟩
    subst label
    subst block
    simpa [branchLoopBodyBlock] using branchLoop_br_vc "body" "loop"
  · rcases hexit with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact branchLoop_exit_vc

theorem branchLoop_kernel_valid
    {init : State} {resource : CSL.Resource}
    (hinit : branchLoopEntryPre init resource)
    (honly :
      cfgKernelInvariant' branchLoopEnv 0 0 branchLoopInvariants branchLoopPost ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl branchLoopEnv 0 0)
    (htrueControl :
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth))
    (hfalseControl :
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth)) :
    (branchLoopKernelSpec init resource).Valid :=
  KernelSpec.Valid.of_entry_blockVCs'
    (spec := branchLoopKernelSpec init resource)
    (env := branchLoopEnv) (cta := 0) (warp := 0)
    (invariants := branchLoopInvariants)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r _hpre
      trivial)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨warpState, henv, hwarp, hlock, hrpc⟩
      exact ⟨warpState, branchLoopEntryBlock, henv, hwarp, hlock, hrpc,
        branchLoop_entry_lookup⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics branchLoop_body_ordinary)
    (hblocks := branchLoop_block_vcs htrueControl hfalseControl)
    (hbr := hbr)
    (htargets := branchLoop_targets_exist)
    (hpost := by
      intro st r st' _hpost _hstep
      exact ⟨r, CSL.Resource.update_refl r, trivial⟩)
    (hfinal := by
      intro final r _hfinal _hinv
      trivial)

theorem branchLoop_partial_correct
    {init final : State} {resource : CSL.Resource}
    (hinit : branchLoopEntryPre init resource)
    (honly :
      cfgKernelInvariant' branchLoopEnv 0 0 branchLoopInvariants branchLoopPost ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl branchLoopEnv 0 0)
    (htrueControl :
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth))
    (hfalseControl :
      CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
        (wpTerminator 0 0 branchLoopCbrTerm branchLoopTruth))
    (hterm : TerminatesAt init final) :
    branchLoopPost final resource := by
  have hvalid : (branchLoopKernelSpec init resource).Valid :=
    branchLoop_kernel_valid hinit honly hbr htrueControl hfalseControl
  rcases KernelSpec.partial_correct hvalid final hterm with ⟨_, hpost⟩
  exact hpost

def sumNormalizedS32 : Nat → Int
  | 0 => 0
  | i + 1 => Helpers.normalizeSigned 32 (sumNormalizedS32 i + Int.ofNat i)

def loopSumSetI : GInstr :=
  { guard? := none, instr := .assignReg "i" (.imm (.s32 0)) }

def loopSumSetAcc : GInstr :=
  { guard? := none, instr := .assignReg "acc" (.imm (.s32 0)) }

def loopSumSetPred (n : Nat) : GInstr :=
  { guard? := none,
    instr := .assignPred "p"
      { op := .lt, lhs := .reg "i", rhs := .imm (.s32 (Int.ofNat n)) } }

def loopSumAddAcc : GInstr :=
  { guard? := none,
    instr := .assignReg "acc" (.binop .add (.reg "acc") (.reg "i")) }

def loopSumIncI : GInstr :=
  { guard? := none,
    instr := .assignReg "i" (.binop .add (.reg "i") (.imm (.s32 1))) }

def loopSumEntryBlock : Block :=
  { label := "entry"
    body := #[loopSumSetI, loopSumSetAcc]
    term := .br "loop" }

def loopSumHeaderBlock (n : Nat) : Block :=
  { label := "loop"
    body := #[loopSumSetPred n]
    term := .cbr (.pred "p") "body" "exit" }

def loopSumBodyBlock : Block :=
  { label := "body"
    body := #[loopSumAddAcc, loopSumIncI]
    term := .br "loop" }

def loopSumExitBlock : Block :=
  { label := "exit"
    body := #[]
    term := .terminate }

def loopSumEnv (n : Nat) : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := 1 } }
    blocks :=
      ((({} : Std.HashMap BlockLabel Block).insert "entry" loopSumEntryBlock).insert
        "loop" (loopSumHeaderBlock n)).insert "body" loopSumBodyBlock |>.insert
        "exit" loopSumExitBlock }

def loopSumRegs (lane : LaneId) (i acc : Int) (p : Bool) : CSL.Assertion :=
  CSL.sepList [
    CSL.reg 0 0 lane "i" (.s32 i),
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.pred 0 0 lane "p" p]

def loopSumAt (lane : LaneId) (pc : PC) (i acc : Int) (p : Bool) : CSL.Assertion :=
  warpAt 0 0 pc [lane] ∗ loopSumRegs lane i acc p

def loopSumEntryPre
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 0) [lane] ∗
    CSL.sepList [
      CSL.reg 0 0 lane "i" oldI,
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.pred 0 0 lane "p" oldP]

def loopSumLoopInv (n : Nat) (lane : LaneId) : CSL.Assertion :=
  fun st r =>
    ∃ i p,
      i ≤ n ∧
        loopSumAt lane ("loop", 0) (Int.ofNat i) (sumNormalizedS32 i) p st r

def loopSumBodyInv (n : Nat) (lane : LaneId) : CSL.Assertion :=
  fun st r =>
    ∃ i,
      i < n ∧
        loopSumAt lane ("body", 0) (Int.ofNat i) (sumNormalizedS32 i) true st r

def loopSumExitInv (n : Nat) (lane : LaneId) : CSL.Assertion :=
  loopSumAt lane ("exit", 0) (Int.ofNat n) (sumNormalizedS32 n) false

def loopSumPost (n : Nat) (lane : LaneId) : CSL.Assertion :=
  laneTerminatedAt 0 0 lane ("exit", 0) ∗
    loopSumRegs lane (Int.ofNat n) (sumNormalizedS32 n) false

def loopSumInvariants
    (n : Nat) (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) : InvariantMap
  | "entry" => loopSumEntryPre lane oldI oldAcc oldP
  | "loop" => loopSumLoopInv n lane
  | "body" => loopSumBodyInv n lane
  | "exit" => loopSumExitInv n lane
  | _ => CSL.pure False

def loopSumKernelSpec
    (init : State) (resource : CSL.Resource) (n : Nat) (lane : LaneId)
    (oldI oldAcc : Value) (oldP : Bool) : KernelSpec :=
  { init := init
    resource := resource
    pre := loopSumEntryPre lane oldI oldAcc oldP
    invariant :=
      cfgKernelInvariant' (loopSumEnv n) 0 0
        (loopSumInvariants n lane oldI oldAcc oldP) (loopSumPost n lane)
    post := loopSumPost n lane }

theorem loopSum_entry_lookup (n : Nat) :
    (loopSumEnv n).blocks["entry"]? = some loopSumEntryBlock := by
  rw [loopSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem loopSum_loop_lookup (n : Nat) :
    (loopSumEnv n).blocks["loop"]? = some (loopSumHeaderBlock n) := by
  rw [loopSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem loopSum_body_lookup (n : Nat) :
    (loopSumEnv n).blocks["body"]? = some loopSumBodyBlock := by
  rw [loopSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem loopSum_exit_lookup (n : Nat) :
    (loopSumEnv n).blocks["exit"]? = some loopSumExitBlock := by
  rw [loopSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem loopSum_block_lookup
    {n : Nat} {label : BlockLabel} {block : Block}
    (hlookup : (loopSumEnv n).blocks[label]? = some block) :
    (label = "entry" ∧ block = loopSumEntryBlock) ∨
      (label = "loop" ∧ block = loopSumHeaderBlock n) ∨
        (label = "body" ∧ block = loopSumBodyBlock) ∨
          (label = "exit" ∧ block = loopSumExitBlock) := by
  by_cases hentry : label = "entry"
  · subst label
    rw [loopSumEnv] at hlookup
    repeat rw [Std.HashMap.getElem?_insert] at hlookup
    simp at hlookup
    exact Or.inl ⟨rfl, hlookup.symm⟩
  · by_cases hloop : label = "loop"
    · subst label
      rw [loopSumEnv] at hlookup
      repeat rw [Std.HashMap.getElem?_insert] at hlookup
      simp at hlookup
      exact Or.inr (Or.inl ⟨rfl, hlookup.symm⟩)
    · by_cases hbody : label = "body"
      · subst label
        rw [loopSumEnv] at hlookup
        repeat rw [Std.HashMap.getElem?_insert] at hlookup
        simp at hlookup
        exact Or.inr (Or.inr (Or.inl ⟨rfl, hlookup.symm⟩))
      · by_cases hexit : label = "exit"
        · subst label
          rw [loopSumEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp at hlookup
          exact Or.inr (Or.inr (Or.inr ⟨rfl, hlookup.symm⟩))
        · have hentryEq : ¬ "entry" = label := fun h => hentry h.symm
          have hloopEq : ¬ "loop" = label := fun h => hloop h.symm
          have hbodyEq : ¬ "body" = label := fun h => hbody h.symm
          have hexitEq : ¬ "exit" = label := fun h => hexit h.symm
          rw [loopSumEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp [hentryEq, hloopEq, hbodyEq, hexitEq] at hlookup

theorem loopSum_body_ordinary (n : Nat) :
    CFGBodyUsesOrdinaryPcAdvance (loopSumEnv n) := by
  intro label block idx gi hlookup hgi
  rcases loopSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [loopSumEntryBlock, loopSumSetI, loopSumSetAcc] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [loopSumEntryBlock, loopSumSetI, loopSumSetAcc] at hgi
            subst gi
            rfl
        | succ idx =>
            simp [loopSumEntryBlock, loopSumSetI, loopSumSetAcc] at hgi
  · rcases hloop with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [loopSumHeaderBlock, loopSumSetPred] at hgi
        subst gi
        rfl
    | succ idx =>
        simp [loopSumHeaderBlock, loopSumSetPred] at hgi
  · rcases hbody with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [loopSumBodyBlock, loopSumAddAcc, loopSumIncI] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [loopSumBodyBlock, loopSumAddAcc, loopSumIncI] at hgi
            subst gi
            rfl
        | succ idx =>
            simp [loopSumBodyBlock, loopSumAddAcc, loopSumIncI] at hgi
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [loopSumExitBlock] at hgi

theorem loopSum_targets_exist (n : Nat) :
    CFGTerminatorTargetsExist (loopSumEnv n) := by
  intro label block hlookup
  rcases loopSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    simp [loopSumEntryBlock]
    exact ⟨loopSumHeaderBlock n, loopSum_loop_lookup n⟩
  · rcases hloop with ⟨_, hblock⟩
    subst block
    simp [loopSumHeaderBlock]
    exact ⟨⟨loopSumBodyBlock, loopSum_body_lookup n⟩,
      loopSumExitBlock, loopSum_exit_lookup n⟩
  · rcases hbody with ⟨_, hblock⟩
    subst block
    simp [loopSumBodyBlock]
    exact ⟨loopSumHeaderBlock n, loopSum_loop_lookup n⟩
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [loopSumExitBlock]

def loopSumProofN : Nat :=
  3

def loopSumEntryTermPre (lane : LaneId) (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 2) [lane] ∗
    CSL.sepList [
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "i" (.s32 0),
      CSL.pred 0 0 lane "p" oldP]

def loopSumEntryAfterI (lane : LaneId) (oldAcc : Value) (oldP : Bool) :
    CSL.Assertion :=
  warpAt 0 0 ("entry", 1) [lane] ∗
    CSL.sepList [
      CSL.reg 0 0 lane "i" (.s32 0),
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.pred 0 0 lane "p" oldP]

def loopSumHeaderTruePre (lane : LaneId) : CSL.Assertion :=
  fun st r =>
    ∃ i,
      i < loopSumProofN ∧
        loopSumAt lane ("loop", 1) (Int.ofNat i) (sumNormalizedS32 i) true st r

def loopSumHeaderFalsePre (lane : LaneId) : CSL.Assertion :=
  loopSumAt lane ("loop", 1) (Int.ofNat loopSumProofN)
    (sumNormalizedS32 loopSumProofN) false

def loopSumHeaderTermPre (lane : LaneId) : CSL.Assertion :=
  loopSumHeaderTruePre lane ∨ₛ loopSumHeaderFalsePre lane

def loopSumBodyTermPre (lane : LaneId) : CSL.Assertion :=
  fun st r =>
    ∃ i,
      i ≤ loopSumProofN ∧
        loopSumAt lane ("body", 2) (Int.ofNat i) (sumNormalizedS32 i) true st r

theorem loopSumAt_acc_i_pred_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (p : Bool) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "i" (.s32 i),
        CSL.pred 0 0 lane "p" p]) ⊢ₛ
      loopSumAt lane pc i acc p := by
  simpa [loopSumAt, loopSumRegs, CSL.sepList] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_swap_head
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.pred 0 0 lane "p" p])

theorem loopSumAt_standard_to_acc_i_pred
    (lane : LaneId) (pc : PC) (i acc : Int) (p : Bool) :
    loopSumAt lane pc i acc p ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.pred 0 0 lane "p" p]) := by
  simpa [loopSumAt, loopSumRegs, CSL.sepList] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_swap_head
        (CSL.reg 0 0 lane "i" (.s32 i))
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        [CSL.pred 0 0 lane "p" p])

theorem loopSumAt_standard_to_pred_i_acc
    (lane : LaneId) (pc : PC) (i acc : Int) (p : Bool) :
    loopSumAt lane pc i acc p ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc)]) := by
  simpa [loopSumAt, loopSumRegs, CSL.sepList] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm
        (((List.Perm.cons (CSL.reg 0 0 lane "i" (.s32 i))
            (List.Perm.swap
              (CSL.reg 0 0 lane "acc" (.s32 acc))
              (CSL.pred 0 0 lane "p" p) [])).symm).trans
          (List.Perm.swap
            (CSL.reg 0 0 lane "i" (.s32 i))
            (CSL.pred 0 0 lane "p" p)
            [CSL.reg 0 0 lane "acc" (.s32 acc)]).symm))

theorem loopSumAt_pred_i_acc_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (p : Bool) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.pred 0 0 lane "p" p,
        CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc)]) ⊢ₛ
      loopSumAt lane pc i acc p := by
  simpa [loopSumAt, loopSumRegs, CSL.sepList] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm
        (((List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "i" (.s32 i))
            [CSL.reg 0 0 lane "acc" (.s32 acc)]).symm).trans
          (List.Perm.cons (CSL.reg 0 0 lane "i" (.s32 i))
            (List.Perm.swap
              (CSL.pred 0 0 lane "p" p)
              (CSL.reg 0 0 lane "acc" (.s32 acc)) [])).symm))

theorem loopSumRegs_stable_terminator
    (term : Terminator) (lane : LaneId) (i acc : Int) (p : Bool) :
    CSL.StableUnder (TerminatorStep 0 0 term) (loopSumRegs lane i acc p) := by
  unfold loopSumRegs
  apply CSL.stable_sepList
  intro q hq
  simp at hq
  rcases hq with hq | hq | hq
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_pred_terminator

theorem loopSumAccIPred_stable_terminator
    (term : Terminator) (lane : LaneId) (i acc : Int) (p : Bool) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (CSL.sepList [
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "i" (.s32 i),
        CSL.pred 0 0 lane "p" p]) := by
  apply CSL.stable_sepList
  intro q hq
  simp at hq
  rcases hq with hq | hq | hq
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_pred_terminator

theorem loopSum_entry_term_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    TerminatorVC 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "entry" (.br "loop")
      (loopSumEntryTermPre lane oldP) := by
  refine TerminatorVC.br ?_
  have hbr :
      loopSumEntryTermPre lane oldP ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (warpAt 0 0 ("loop", 0) [lane] ∗
            CSL.sepList [
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "i" (.s32 0),
              CSL.pred 0 0 lane "p" oldP]) := by
    simpa [loopSumEntryTermPre] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("entry", 2)) (target := "loop")
        (lanes := [lane])
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 0),
            CSL.reg 0 0 lane "i" (.s32 0),
            CSL.pred 0 0 lane "p" oldP])
        (loopSumAccIPred_stable_terminator (.br "loop") lane 0 0 oldP))
  exact CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st r hpost
      refine ⟨0, oldP, ?_, ?_⟩
      · simp [loopSumProofN]
      · exact loopSumAt_acc_i_pred_to_standard lane ("loop", 0) 0 0 oldP st r hpost)

theorem loopSum_body_term_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    TerminatorVC 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "body" (.br "loop")
      (loopSumBodyTermPre lane) := by
  refine TerminatorVC.br ?_
  intro st r hpre
  rcases hpre with ⟨i, hiLe, hat⟩
  have hbr :
      loopSumAt lane ("body", 2) (Int.ofNat i) (sumNormalizedS32 i) true ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (loopSumAt lane ("loop", 0) (Int.ofNat i) (sumNormalizedS32 i) true) := by
    simpa [loopSumAt] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 2)) (target := "loop")
        (lanes := [lane])
        (frame := loopSumRegs lane (Int.ofNat i) (sumNormalizedS32 i) true)
        (loopSumRegs_stable_terminator (.br "loop") lane
          (Int.ofNat i) (sumNormalizedS32 i) true))
  exact (CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st' r' hloop
      exact ⟨i, true, hiLe, hloop⟩)) st r hat

theorem loopSum_exit_term_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    TerminatorVC 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "exit" .terminate
      (loopSumExitInv loopSumProofN lane) := by
  refine TerminatorVC.terminate ?_
  simpa [loopSumExitInv, loopSumPost, loopSumAt] using
    (wp_terminate_single_warpAt_frame
      (cta := 0) (warp := 0) (pc := ("exit", 0)) (lane := lane)
      (frame :=
        loopSumRegs lane (Int.ofNat loopSumProofN) (sumNormalizedS32 loopSumProofN)
          false)
      (loopSumRegs_stable_terminator .terminate lane
        (Int.ofNat loopSumProofN) (sumNormalizedS32 loopSumProofN) false))

theorem loopSum_cbr_true_raw_wp
    (lane : LaneId) (i : Nat) :
    loopSumAt lane ("loop", 1) (Int.ofNat i) (sumNormalizedS32 i) true ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (loopSumAt lane ("body", 0) (Int.ofNat i) (sumNormalizedS32 i) true) := by
  simpa [loopSumAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred true) (takeTrue := true)
      (frame := loopSumRegs lane (Int.ofNat i) (sumNormalizedS32 i) true)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rFrame, _hcomp, _hequiv, _hctrl, hframe⟩
        unfold loopSumRegs at hframe
        rcases hframe with ⟨_rI, _rRest, _hcompRegs, _hequivRegs, _hi, hrest⟩
        rcases hrest with ⟨_rAcc, _rPred, _hcompRest, _hequivRest, _hacc, hpred⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (loopSumRegs_stable_terminator (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat i) (sumNormalizedS32 i) true))

theorem loopSum_cbr_false_raw_wp (lane : LaneId) :
    loopSumHeaderFalsePre lane ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (loopSumExitInv loopSumProofN lane) := by
  simpa [loopSumHeaderFalsePre, loopSumExitInv, loopSumAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred false) (takeTrue := false)
      (frame :=
        loopSumRegs lane (Int.ofNat loopSumProofN) (sumNormalizedS32 loopSumProofN)
          false)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rFrame, _hcomp, _hequiv, _hctrl, hframe⟩
        unfold loopSumRegs at hframe
        rcases hframe with ⟨_rI, _rRest, _hcompRegs, _hequivRegs, _hi, hrest⟩
        rcases hrest with ⟨_rAcc, _rPred, _hcompRest, _hequivRest, _hacc, hpred⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (loopSumRegs_stable_terminator (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat loopSumProofN) (sumNormalizedS32 loopSumProofN) false))

theorem loopSum_cbr_true_control (lane : LaneId) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
      (loopSumHeaderTruePre lane) := by
  intro st st' r hpre hstep
  rcases hpre with ⟨i, _hlt, hat⟩
  rcases loopSum_cbr_true_raw_wp lane i st r hat st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem loopSum_cbr_false_control (lane : LaneId) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
      (loopSumHeaderFalsePre lane) := by
  intro st st' r hpre hstep
  rcases loopSum_cbr_false_raw_wp lane st r hpre st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem loopSum_header_term_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    TerminatorVC 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "loop"
      (.cbr (.pred "p") "body" "exit") (loopSumHeaderTermPre lane) := by
  refine TerminatorVC.cbr
    (truePre := loopSumHeaderTruePre lane)
    (falsePre := loopSumHeaderFalsePre lane)
    ?_ (loopSum_cbr_true_control lane) (loopSum_cbr_false_control lane) ?_ ?_
  · intro st r hpre
    simpa [loopSumHeaderTermPre] using hpre
  · intro st r hpre
    rcases hpre with ⟨i, hlt, hat⟩
    exact (CSL.entails_trans (loopSum_cbr_true_raw_wp lane i) <|
      wpTerminator_mono (by
        intro st' r' hbody
        exact ⟨i, hlt, hbody⟩)) st r hat
  · exact loopSum_cbr_false_raw_wp lane

theorem loopSum_entry_instrs_wp
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    loopSumEntryPre lane oldI oldAcc oldP ⊢ₛ
      wpInstrs 0 0 loopSumEntryBlock.body (loopSumEntryTermPre lane oldP) := by
  have hsetI :
      loopSumEntryPre lane oldI oldAcc oldP ⊢ₛ
        wpInstr 0 0 loopSumSetI (loopSumEntryAfterI lane oldAcc oldP) := by
    simpa [loopSumEntryPre, loopSumEntryAfterI, loopSumSetI, CSL.sepList] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("entry", 0)) (dst := "i")
        (rhs := .imm (.s32 0)) (lane := lane) (old := oldI) (new := .s32 0)
        (frame := CSL.reg 0 0 lane "acc" oldAcc ∗ CSL.pred 0 0 lane "p" oldP)
        (by
          intro _st _r _hpre
          exact eval_imm)
        (CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          stable_pred_assignReg))
  have hsetAcc :
      loopSumEntryAfterI lane oldAcc oldP ⊢ₛ
        wpInstr 0 0 loopSumSetAcc (loopSumEntryTermPre lane oldP) := by
    have htoRule :
        loopSumEntryAfterI lane oldAcc oldP ⊢ₛ
          (warpAt 0 0 ("entry", 1) [lane] ∗
            CSL.sepList [
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "i" (.s32 0),
              CSL.pred 0 0 lane "p" oldP]) := by
      simpa [loopSumEntryAfterI, CSL.sepList] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 1) [lane]))
          (CSL.sepList_swap_head
            (CSL.reg 0 0 lane "i" (.s32 0))
            (CSL.reg 0 0 lane "acc" oldAcc)
            [CSL.pred 0 0 lane "p" oldP])
    have hrule :
        (warpAt 0 0 ("entry", 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "acc" oldAcc,
            CSL.reg 0 0 lane "i" (.s32 0),
            CSL.pred 0 0 lane "p" oldP]) ⊢ₛ
          wpInstr 0 0 loopSumSetAcc (loopSumEntryTermPre lane oldP) := by
      simpa [loopSumEntryTermPre, loopSumSetAcc, CSL.sepList] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 1)) (dst := "acc")
          (rhs := .imm (.s32 0)) (lane := lane) (old := oldAcc) (new := .s32 0)
          (frame := CSL.reg 0 0 lane "i" (.s32 0) ∗ CSL.pred 0 0 lane "p" oldP)
          (by
            intro _st _r _hpre
            exact eval_imm)
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            stable_pred_assignReg))
    exact CSL.entails_trans htoRule hrule
  simpa [loopSumEntryBlock, wpInstrs, wpInstrList, loopSumSetI, loopSumSetAcc] using
    CSL.entails_trans hsetI (wpInstr_mono hsetAcc)

theorem loopSum_header_instrs_wp
    (lane : LaneId) :
    loopSumLoopInv loopSumProofN lane ⊢ₛ
      wpInstrs 0 0 (loopSumHeaderBlock loopSumProofN).body
        (loopSumHeaderTermPre lane) := by
  intro st r hpre
  rcases hpre with ⟨i, p, hiLe, hat⟩
  have htoRule :
      loopSumAt lane ("loop", 0) (Int.ofNat i) (sumNormalizedS32 i) p ⊢ₛ
        (warpAt 0 0 ("loop", 0) [lane] ∗
          CSL.sepList [
            CSL.pred 0 0 lane "p" p,
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i))]) :=
    loopSumAt_standard_to_pred_i_acc lane ("loop", 0)
      (Int.ofNat i) (sumNormalizedS32 i) p
  have hrule :
      (warpAt 0 0 ("loop", 0) [lane] ∗
        CSL.sepList [
          CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
          CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i))]) ⊢ₛ
        wpInstr 0 0 (loopSumSetPred loopSumProofN)
          (warpAt 0 0 ("loop", 1) [lane] ∗
            CSL.sepList [
              CSL.pred 0 0 lane "p"
                (decide (Int.ofNat i < Int.ofNat loopSumProofN)),
              CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
              CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i))]) := by
    simpa [loopSumSetPred, CSL.sepList] using
      (wp_assignPred_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("loop", 0)) (dst := "p")
        (cmp := { op := .lt, lhs := .reg "i", rhs := .imm (.s32 (Int.ofNat loopSumProofN)) })
        (lane := lane) (old := p)
        (new := decide (Int.ofNat i < Int.ofNat loopSumProofN))
        (frame :=
          CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)) ∗
            CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i)))
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, _hpred,
            hframe⟩
          rcases hframe with ⟨_rI, _rAcc, _hcompFrame, _hequivFrame, hi, _hacc⟩
          exact EvalCmp.lt_s32 (eval_reg_of_assertion hi) eval_imm)
        (CSL.stable_sep stable_reg_assignPred stable_reg_assignPred))
  have hpost :
      (warpAt 0 0 ("loop", 1) [lane] ∗
        CSL.sepList [
          CSL.pred 0 0 lane "p" (decide (Int.ofNat i < Int.ofNat loopSumProofN)),
          CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
          CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i))]) ⊢ₛ
        loopSumHeaderTermPre lane := by
    intro st r hraw
    have hstandard :
        loopSumAt lane ("loop", 1) (Int.ofNat i) (sumNormalizedS32 i)
          (decide (Int.ofNat i < Int.ofNat loopSumProofN)) st r :=
      loopSumAt_pred_i_acc_to_standard lane ("loop", 1)
        (Int.ofNat i) (sumNormalizedS32 i)
        (decide (Int.ofNat i < Int.ofNat loopSumProofN)) st r hraw
    by_cases hlt : i < loopSumProofN
    · have hltInt : Int.ofNat i < Int.ofNat loopSumProofN := Int.ofNat_lt.mpr hlt
      have hdec :
          decide (Int.ofNat i < Int.ofNat loopSumProofN) = true := by
        simp [hltInt, hlt]
      left
      refine ⟨i, hlt, ?_⟩
      rw [hdec] at hstandard
      exact hstandard
    · have hge : loopSumProofN ≤ i := Nat.le_of_not_gt hlt
      have hiEq : i = loopSumProofN := Nat.le_antisymm hiLe hge
      subst i
      have hnotInt :
          ¬ Int.ofNat loopSumProofN < Int.ofNat loopSumProofN := by
        exact Int.lt_irrefl (Int.ofNat loopSumProofN)
      have hdec :
          decide (Int.ofNat loopSumProofN < Int.ofNat loopSumProofN) = false := by
        simp [hnotInt]
      right
      rw [hdec] at hstandard
      simpa [loopSumHeaderFalsePre] using hstandard
  simpa [loopSumHeaderBlock, wpInstrs, wpInstrList, loopSumSetPred] using
    (CSL.entails_trans htoRule (CSL.entails_trans hrule (wpInstr_mono hpost)) st r hat)

theorem loopSum_inc_s32_of_lt_proofN {i : Nat} (h : i < loopSumProofN) :
    Helpers.normalizeSigned 32 (Int.ofNat i + 1) = Int.ofNat (i + 1) := by
  unfold loopSumProofN at h
  cases i with
  | zero =>
      simp [Helpers.normalizeSigned]
  | succ i =>
      cases i with
      | zero =>
          simp [Helpers.normalizeSigned]
      | succ i =>
          cases i with
          | zero =>
              simp [Helpers.normalizeSigned]
          | succ i =>
              omega

theorem loopSum_body_instrs_wp_at
    (lane : LaneId) (i : Nat) (hlt : i < loopSumProofN) :
    loopSumAt lane ("body", 0) (Int.ofNat i) (sumNormalizedS32 i) true ⊢ₛ
      wpInstrs 0 0 loopSumBodyBlock.body (loopSumBodyTermPre lane) := by
  have hinc := loopSum_inc_s32_of_lt_proofN hlt
  have hincValue :
      Helpers.normalizeSigned 32 (Int.ofNat i + 1) = Int.ofNat i + 1 := by
    rw [hinc]
    simp
  have hadd :
      loopSumAt lane ("body", 0) (Int.ofNat i) (sumNormalizedS32 i) true ⊢ₛ
        wpInstr 0 0 loopSumAddAcc
          (warpAt 0 0 ("body", 1) [lane] ∗
            CSL.sepList [
              CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 (i + 1))),
              CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
              CSL.pred 0 0 lane "p" true]) := by
    have htoRule :=
      loopSumAt_standard_to_acc_i_pred lane ("body", 0)
        (Int.ofNat i) (sumNormalizedS32 i) true
    have hrule :
        (warpAt 0 0 ("body", 0) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 i)),
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.pred 0 0 lane "p" true]) ⊢ₛ
          wpInstr 0 0 loopSumAddAcc
            (warpAt 0 0 ("body", 1) [lane] ∗
              CSL.sepList [
                CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 (i + 1))),
                CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
                CSL.pred 0 0 lane "p" true]) := by
      simpa [loopSumAddAcc, CSL.sepList] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "acc")
          (rhs := .binop .add (.reg "acc") (.reg "i")) (lane := lane)
          (old := .s32 (sumNormalizedS32 i))
          (new := .s32 (sumNormalizedS32 (i + 1)))
          (frame := CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)) ∗
            CSL.pred 0 0 lane "p" true)
          (by
            intro st r hpre
            rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
            rcases hrest with ⟨_rAcc, _rFrame, _hcompRest, _hequivRest, hacc,
              hframe⟩
            rcases hframe with ⟨_rI, _rPred, _hcompFrame, _hequivFrame, hi, _hp⟩
            simpa [sumNormalizedS32] using
              EvalRValue.binop_add_s32 (eval_reg_of_assertion hacc)
                (eval_reg_of_assertion hi))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            stable_pred_assignReg))
    exact CSL.entails_trans htoRule hrule
  have hincWp :
      (warpAt 0 0 ("body", 1) [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 (i + 1))),
          CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
          CSL.pred 0 0 lane "p" true]) ⊢ₛ
        wpInstr 0 0 loopSumIncI (loopSumBodyTermPre lane) := by
    have htoRule :
        (warpAt 0 0 ("body", 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 (i + 1))),
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.pred 0 0 lane "p" true]) ⊢ₛ
          loopSumAt lane ("body", 1) (Int.ofNat i) (sumNormalizedS32 (i + 1))
            true :=
      loopSumAt_acc_i_pred_to_standard lane ("body", 1)
        (Int.ofNat i) (sumNormalizedS32 (i + 1)) true
    have hrule :
        loopSumAt lane ("body", 1) (Int.ofNat i) (sumNormalizedS32 (i + 1))
          true ⊢ₛ
          wpInstr 0 0 loopSumIncI
            (loopSumAt lane ("body", 2) (Int.ofNat (i + 1))
              (sumNormalizedS32 (i + 1)) true) := by
      simpa [loopSumAt, loopSumIncI, CSL.sepList] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("body", 1)) (dst := "i")
          (rhs := .binop .add (.reg "i") (.imm (.s32 1))) (lane := lane)
          (old := .s32 (Int.ofNat i)) (new := .s32 (Int.ofNat (i + 1)))
          (frame := CSL.reg 0 0 lane "acc" (.s32 (sumNormalizedS32 (i + 1))) ∗
            CSL.pred 0 0 lane "p" true)
          (by
            intro st r hpre
            rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
            rcases hrest with ⟨_rI, _rFrame, _hcompRest, _hequivRest, hi, _hframe⟩
            have hevalNorm :
                EvalRValue st { cta := 0, warp := 0, lane := lane }
                  (.binop .add (.reg "i") (.imm (.s32 1)))
                  (.s32 (Helpers.normalizeSigned 32 (Int.ofNat i + 1))) :=
              EvalRValue.binop_add_s32
                (eval_reg_of_assertion
                  (ctx := { cta := 0, warp := 0, lane := lane }) hi)
                (eval_imm
                  (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                  (value := .s32 (1 : Int)))
            rw [hincValue] at hevalNorm
            exact hevalNorm)
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            stable_pred_assignReg))
    exact CSL.entails_trans htoRule <|
      CSL.entails_trans hrule <|
        wpInstr_mono (by
          intro st r hpost
          exact ⟨i + 1, Nat.succ_le_of_lt hlt, hpost⟩)
  simpa [loopSumBodyBlock, wpInstrs, wpInstrList, loopSumAddAcc, loopSumIncI] using
    CSL.entails_trans hadd (wpInstr_mono hincWp)

theorem loopSum_body_instrs_wp (lane : LaneId) :
    loopSumBodyInv loopSumProofN lane ⊢ₛ
      wpInstrs 0 0 loopSumBodyBlock.body (loopSumBodyTermPre lane) := by
  intro st r hpre
  rcases hpre with ⟨i, hlt, hat⟩
  exact loopSum_body_instrs_wp_at lane i hlt st r hat

theorem loopSum_entry_block_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    blockVC' 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "entry" loopSumEntryBlock := by
  refine ⟨loopSumEntryTermPre lane oldP, ?_, ?_⟩
  · simpa [loopSumInvariants, loopSumEntryBlock, loopSumProofN] using
      loopSum_entry_instrs_wp lane oldI oldAcc oldP
  · exact loopSum_entry_term_vc lane oldI oldAcc oldP

theorem loopSum_header_block_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    blockVC' 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "loop" (loopSumHeaderBlock loopSumProofN) := by
  refine ⟨loopSumHeaderTermPre lane, ?_, ?_⟩
  · simpa [loopSumInvariants, loopSumHeaderBlock, loopSumProofN] using
      loopSum_header_instrs_wp lane
  · exact loopSum_header_term_vc lane oldI oldAcc oldP

theorem loopSum_body_block_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    blockVC' 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "body" loopSumBodyBlock := by
  refine ⟨loopSumBodyTermPre lane, ?_, ?_⟩
  · simpa [loopSumInvariants, loopSumBodyBlock, loopSumProofN] using
      loopSum_body_instrs_wp lane
  · exact loopSum_body_term_vc lane oldI oldAcc oldP

theorem loopSum_exit_block_vc
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    blockVC' 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
      (loopSumPost loopSumProofN lane) "exit" loopSumExitBlock := by
  refine ⟨loopSumExitInv loopSumProofN lane, ?_, ?_⟩
  · simpa [loopSumInvariants, loopSumExitBlock, wpInstrs, loopSumProofN] using
      (CSL.entails_refl (loopSumExitInv loopSumProofN lane))
  · exact loopSum_exit_term_vc lane oldI oldAcc oldP

theorem loopSum_block_vcs
    (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    ∀ label block,
      (loopSumEnv loopSumProofN).blocks[label]? = some block →
        blockVC' 0 0 (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
          (loopSumPost loopSumProofN lane) label block := by
  intro label block hlookup
  rcases loopSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact loopSum_entry_block_vc lane oldI oldAcc oldP
  · rcases hloop with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact loopSum_header_block_vc lane oldI oldAcc oldP
  · rcases hbody with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact loopSum_body_block_vc lane oldI oldAcc oldP
  · rcases hexit with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact loopSum_exit_block_vc lane oldI oldAcc oldP

def loopSumCheckedEntryPre
    (n : Nat) (lane : LaneId) (oldI oldAcc : Value) (oldP : Bool) :
    CSL.Assertion :=
  fun st r =>
    st.kernelEnv = loopSumEnv n ∧ loopSumEntryPre lane oldI oldAcc oldP st r

def loopSumCheckedKernelSpec
    (init : State) (resource : CSL.Resource) (lane : LaneId)
    (oldI oldAcc : Value) (oldP : Bool) : KernelSpec :=
  { init := init
    resource := resource
    pre := loopSumCheckedEntryPre loopSumProofN lane oldI oldAcc oldP
    invariant :=
      cfgKernelInvariant' (loopSumEnv loopSumProofN) 0 0
        (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
        (loopSumPost loopSumProofN lane)
    post := loopSumPost loopSumProofN lane }

theorem loop_sum_kernel_valid
    {init : State} {resource : CSL.Resource} {lane : LaneId}
    {oldI oldAcc : Value} {oldP : Bool}
    (hinit : loopSumCheckedEntryPre loopSumProofN lane oldI oldAcc oldP init resource)
    (honly :
      cfgKernelInvariant' (loopSumEnv loopSumProofN) 0 0
        (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
        (loopSumPost loopSumProofN lane) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (loopSumEnv loopSumProofN) 0 0)
    (hpostNoStep : NoStepBlock 0 0 (loopSumPost loopSumProofN lane))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (loopSumEnv loopSumProofN) 0 0
          (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
          (loopSumPost loopSumProofN lane))) :
    (loopSumCheckedKernelSpec init resource lane oldI oldAcc oldP).Valid :=
  KernelSpec.Valid.of_entry_blockVCs'_closed
    (spec := loopSumCheckedKernelSpec init resource lane oldI oldAcc oldP)
    (env := loopSumEnv loopSumProofN) (cta := 0) (warp := 0)
    (invariants := loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r hpre
      exact hpre.2)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨henv, hentry⟩
      rcases hentry with ⟨_rCtrl, _rRegs, _hcomp, _hequiv, hctrl, _hregs⟩
      rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
      exact ⟨warpState, loopSumEntryBlock, henv, hwarp, hlock,
        by simpa [loopSumEnv, loopSumProofN] using hrpc,
        by simpa [loopSumEnv, loopSumProofN] using loopSum_entry_lookup loopSumProofN⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics (loopSum_body_ordinary loopSumProofN))
    (hblocks := loopSum_block_vcs lane oldI oldAcc oldP)
    (hbr := hbr)
    (htargets := loopSum_targets_exist loopSumProofN)
    (hpostNoStep := hpostNoStep)
    (hsuffixNoFinal := hsuffixNoFinal)

theorem loop_sum_partial_correct
    {init final : State} {resource : CSL.Resource} {lane : LaneId}
    {oldI oldAcc : Value} {oldP : Bool}
    (hinit : loopSumCheckedEntryPre loopSumProofN lane oldI oldAcc oldP init resource)
    (honly :
      cfgKernelInvariant' (loopSumEnv loopSumProofN) 0 0
        (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
        (loopSumPost loopSumProofN lane) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (loopSumEnv loopSumProofN) 0 0)
    (hpostNoStep : NoStepBlock 0 0 (loopSumPost loopSumProofN lane))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (loopSumEnv loopSumProofN) 0 0
          (loopSumInvariants loopSumProofN lane oldI oldAcc oldP)
          (loopSumPost loopSumProofN lane)))
    (hterm : TerminatesAt init final) :
    ∃ r, loopSumPost loopSumProofN lane final r := by
  have hvalid : (loopSumCheckedKernelSpec init resource lane oldI oldAcc oldP).Valid :=
    loop_sum_kernel_valid hinit honly hbr hpostNoStep hsuffixNoFinal
  exact KernelSpec.partial_correct hvalid final hterm

def globalSumProofN : Nat :=
  3

def globalSumSetI : GInstr :=
  { guard? := none, instr := .assignReg "i" (.imm (.s32 0)) }

def globalSumSetAcc : GInstr :=
  { guard? := none, instr := .assignReg "acc" (.imm (.s32 0)) }

def globalSumSetPtr (xBase : Nat) : GInstr :=
  { guard? := none, instr := .assignReg "ptr" (.imm (.gaddr .global xBase)) }

def globalSumSetPred (n : Nat) : GInstr :=
  { guard? := none,
    instr := .assignPred "p"
      { op := .lt, lhs := .reg "i", rhs := .imm (.s32 (Int.ofNat n)) } }

def globalSumLoadTmp : GInstr :=
  { guard? := none,
    instr := .load "tmp" { space := .global, ty := .s32, addr := .reg "ptr" } }

def globalSumAddAcc : GInstr :=
  { guard? := none,
    instr := .assignReg "acc" (.binop .add (.reg "acc") (.reg "tmp")) }

def globalSumIncI : GInstr :=
  { guard? := none,
    instr := .assignReg "i" (.binop .add (.reg "i") (.imm (.s32 1))) }

def globalSumIncPtr : GInstr :=
  { guard? := none,
    instr := .assignReg "ptr" (.binop .add (.reg "ptr") (.imm (.u64 (4 : UInt64)))) }

def globalSumStoreAcc (outBase : Nat) : GInstr :=
  { guard? := none,
    instr :=
      .store { space := .global, ty := .s32, addr := .imm (.gaddr .global outBase) }
        (.reg "acc") }

def globalSumEntryBlock (xBase : Nat) : Block :=
  { label := "entry"
    body := #[globalSumSetI, globalSumSetAcc, globalSumSetPtr xBase]
    term := .br "loop" }

def globalSumHeaderBlock (n : Nat) : Block :=
  { label := "loop"
    body := #[globalSumSetPred n]
    term := .cbr (.pred "p") "body" "exit" }

def globalSumBodyBlock : Block :=
  { label := "body"
    body := #[globalSumLoadTmp, globalSumAddAcc, globalSumIncI, globalSumIncPtr]
    term := .br "loop" }

def globalSumExitBlock (outBase : Nat) : Block :=
  { label := "exit"
    body := #[globalSumStoreAcc outBase]
    term := .terminate }

def globalSumEnv (n xBase outBase : Nat) : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := 1 } }
    blocks :=
      ((({} : Std.HashMap BlockLabel Block).insert "entry" (globalSumEntryBlock xBase)).insert
        "loop" (globalSumHeaderBlock n)).insert "body" globalSumBodyBlock |>.insert
        "exit" (globalSumExitBlock outBase) }

def globalSumPrefixS32 (xs : Nat → Int) : Nat → Int
  | 0 => 0
  | i + 1 => Helpers.normalizeSigned 32 (globalSumPrefixS32 xs i + xs i)

def globalSumXResources (xBase : Nat) (xBytes : Nat → List Byte) : CSL.Assertion :=
  CSL.sepList
    ((List.range globalSumProofN).map fun i =>
      CSL.globalBytes (xBase + 4 * i) .read (xBytes i))

def globalSumFrame
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.Assertion :=
  globalSumXResources xBase xBytes ∗ CSL.globalBytes outBase .write outBytes

def globalSumRegs
    (lane : LaneId) (i acc : Int) (tmp ptr : Value) (p : Bool) : CSL.Assertion :=
  CSL.sepList [
    CSL.reg 0 0 lane "i" (.s32 i),
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.reg 0 0 lane "tmp" tmp,
    CSL.reg 0 0 lane "ptr" ptr,
    CSL.pred 0 0 lane "p" p]

def globalSumAt
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.Assertion :=
  warpAt 0 0 pc [lane] ∗
    (globalSumRegs lane i acc tmp ptr p ∗ globalSumFrame xBase outBase xBytes outBytes)

def globalSumFlatResources
    (lane : LaneId) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.Assertion :=
  CSL.sepList [
    CSL.reg 0 0 lane "i" (.s32 i),
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.reg 0 0 lane "tmp" tmp,
    CSL.reg 0 0 lane "ptr" ptr,
    CSL.pred 0 0 lane "p" p,
    CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0),
    CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1),
    CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2),
    CSL.globalBytes outBase .write outBytes]

def globalSumEntryPre
    (lane : LaneId) (xBase outBase : Nat) (xBytes : Nat → List Byte) (oldOut : List Byte)
    (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 0) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "i" oldI,
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "tmp" oldTmp,
      CSL.reg 0 0 lane "ptr" oldPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      globalSumFrame xBase outBase xBytes oldOut)

def globalSumEntryAfterI
    (lane : LaneId) (xBase outBase : Nat) (xBytes : Nat → List Byte) (oldOut : List Byte)
    (oldAcc oldTmp oldPtr : Value) (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 1) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "i" (.s32 0),
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "tmp" oldTmp,
      CSL.reg 0 0 lane "ptr" oldPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      globalSumFrame xBase outBase xBytes oldOut)

def globalSumEntryAfterAcc
    (lane : LaneId) (xBase outBase : Nat) (xBytes : Nat → List Byte) (oldOut : List Byte)
    (oldTmp oldPtr : Value) (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 2) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "i" (.s32 0),
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "tmp" oldTmp,
      CSL.reg 0 0 lane "ptr" oldPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      globalSumFrame xBase outBase xBytes oldOut)

def globalSumEntryTermPre
    (lane : LaneId) (xBase outBase : Nat) (xBytes : Nat → List Byte) (oldOut : List Byte)
    (oldTmp : Value) (oldP : Bool) : CSL.Assertion :=
  globalSumAt lane ("entry", 3) 0 0 oldTmp (.gaddr .global xBase) oldP
    xBase outBase xBytes oldOut

def globalSumHeaderTruePre
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ i tmp,
      i < globalSumProofN ∧
        globalSumAt lane ("loop", 1) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut st r

def globalSumHeaderFalsePre
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ tmp,
      globalSumAt lane ("loop", 1) (Int.ofNat globalSumProofN)
        (globalSumPrefixS32 xs globalSumProofN) tmp
        (.gaddr .global (xBase + 4 * globalSumProofN)) false
        xBase outBase xBytes oldOut st r

def globalSumHeaderTermPre
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  globalSumHeaderTruePre lane xBase outBase xs xBytes oldOut ∨ₛ
    globalSumHeaderFalsePre lane xBase outBase xs xBytes oldOut

def globalSumLoopInv
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ i tmp p,
      i ≤ globalSumProofN ∧
        globalSumAt lane ("loop", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) p xBase outBase xBytes oldOut st r

def globalSumBodyInv
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ i tmp,
      i < globalSumProofN ∧
        globalSumAt lane ("body", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut st r

def globalSumExitInv
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ tmp,
      globalSumAt lane ("exit", 0) (Int.ofNat globalSumProofN)
        (globalSumPrefixS32 xs globalSumProofN) tmp
        (.gaddr .global (xBase + 4 * globalSumProofN)) false
        xBase outBase xBytes oldOut st r

def globalSumBodyTermPre
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ i tmp,
      i ≤ globalSumProofN ∧
        globalSumAt lane ("body", 4) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut st r

def globalSumExitTermPre
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (newOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ tmp,
      globalSumAt lane ("exit", 1) (Int.ofNat globalSumProofN)
        (globalSumPrefixS32 xs globalSumProofN) tmp
        (.gaddr .global (xBase + 4 * globalSumProofN)) false
        xBase outBase xBytes newOut st r

def globalSumPost
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (newOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ tmp,
      (laneTerminatedAt 0 0 lane ("exit", 1) ∗
        (globalSumRegs lane (Int.ofNat globalSumProofN)
          (globalSumPrefixS32 xs globalSumProofN) tmp
          (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
          globalSumFrame xBase outBase xBytes newOut)) st r

def globalSumInvariants
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut _newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    InvariantMap
  | "entry" => globalSumEntryPre lane xBase outBase xBytes oldOut
      oldI oldAcc oldTmp oldPtr oldP
  | "loop" => globalSumLoopInv lane xBase outBase xs xBytes oldOut
  | "body" => globalSumBodyInv lane xBase outBase xs xBytes oldOut
  | "exit" => globalSumExitInv lane xBase outBase xs xBytes oldOut
  | _ => CSL.pure False

def globalSumKernelSpec
    (init : State) (resource : CSL.Resource) (lane : LaneId)
    (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    KernelSpec :=
  { init := init
    resource := resource
    pre :=
      fun st r =>
        st.kernelEnv = globalSumEnv globalSumProofN xBase outBase ∧
          globalSumEntryPre lane xBase outBase xBytes oldOut
            oldI oldAcc oldTmp oldPtr oldP st r
    invariant :=
      cfgKernelInvariant' (globalSumEnv globalSumProofN xBase outBase) 0 0
        (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
          oldI oldAcc oldTmp oldPtr oldP)
        (globalSumPost lane xBase outBase xs xBytes newOut)
    post := globalSumPost lane xBase outBase xs xBytes newOut }

theorem globalSumXResources_stable
    {step : State → State → Prop} {xBase : Nat} {xBytes : Nat → List Byte}
    (hbytes :
      ∀ i, CSL.StableUnder step (CSL.globalBytes (xBase + 4 * i) .read (xBytes i))) :
    CSL.StableUnder step (globalSumXResources xBase xBytes) := by
  unfold globalSumXResources
  apply CSL.stable_sepList
  intro p hp
  simp at hp
  rcases hp with ⟨i, _hi, hp⟩
  subst p
  exact hbytes i

theorem globalSumFrame_stable
    {step : State → State → Prop} {xBase outBase : Nat} {xBytes : Nat → List Byte}
    {outBytes : List Byte}
    (hx :
      ∀ i, CSL.StableUnder step (CSL.globalBytes (xBase + 4 * i) .read (xBytes i)))
    (hout : CSL.StableUnder step (CSL.globalBytes outBase .write outBytes)) :
    CSL.StableUnder step (globalSumFrame xBase outBase xBytes outBytes) := by
  unfold globalSumFrame
  exact CSL.stable_sep (globalSumXResources_stable hx) hout

theorem globalSumRegs_stable_terminator
    (term : Terminator) (lane : LaneId) (i acc : Int) (tmp ptr : Value) (p : Bool) :
    CSL.StableUnder (TerminatorStep 0 0 term) (globalSumRegs lane i acc tmp ptr p) := by
  unfold globalSumRegs
  apply CSL.stable_sepList
  intro q hq
  simp at hq
  rcases hq with hq | hq | hq | hq | hq
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_pred_terminator

theorem globalSumFrame_stable_terminator
    (term : Terminator) (xBase outBase : Nat) (xBytes : Nat → List Byte)
    (outBytes : List Byte) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (globalSumFrame xBase outBase xBytes outBytes) :=
  globalSumFrame_stable
    (fun _ => stable_globalBytes_terminator)
    stable_globalBytes_terminator

theorem globalSumAt_stable_terminator_frame
    (term : Terminator) (lane : LaneId) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (globalSumRegs lane i acc tmp ptr p ∗ globalSumFrame xBase outBase xBytes outBytes) :=
  CSL.stable_sep
    (globalSumRegs_stable_terminator term lane i acc tmp ptr p)
    (globalSumFrame_stable_terminator term xBase outBase xBytes outBytes)

theorem globalSumFrame_stable_assignReg
    (dst : RegName) (rhs : RValue) (xBase outBase : Nat)
    (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (globalSumFrame xBase outBase xBytes outBytes) :=
  globalSumFrame_stable
    (fun _ => stable_globalBytes_assignReg)
    stable_globalBytes_assignReg

theorem globalSumFrame_stable_assignPred
    (dst : PredName) (cmp : CmpExpr) (xBase outBase : Nat)
    (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (globalSumFrame xBase outBase xBytes outBytes) :=
  globalSumFrame_stable
    (fun _ => stable_globalBytes_assignPred)
    stable_globalBytes_assignPred

theorem globalSumFrame_stable_load
    (dst : RegName) (addr : TypedAddr) (xBase outBase : Nat)
    (xBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst addr })
      (globalSumFrame xBase outBase xBytes outBytes) :=
  globalSumFrame_stable
    (fun _ => stable_globalBytes_load)
    stable_globalBytes_load

theorem globalSumAt_to_i_focus
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.reg 0 0 lane "i" (.s32 i) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p] ∗
            globalSumFrame xBase outBase xBytes outBytes))) := by
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm_frame_to_cons
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (CSL.reg 0 0 lane "i" (.s32 i))
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        [CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        (List.Perm.refl _))

theorem globalSumAt_i_focus_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "i" (.s32 i) ∗
        (CSL.sepList [
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p] ∗
          globalSumFrame xBase outBase xBytes outBytes))) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_cons_frame_to_sepList_perm
        (CSL.reg 0 0 lane "i" (.s32 i))
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        [CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        (List.Perm.refl _))

theorem globalSumAt_to_acc_focus
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 i),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p] ∗
            globalSumFrame xBase outBase xBytes outBytes))) := by
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm_frame_to_cons
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        (List.Perm.swap
          (CSL.reg 0 0 lane "acc" (.s32 acc))
          (CSL.reg 0 0 lane "i" (.s32 i))
          [CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p]))

theorem globalSumAt_acc_focus_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
        (CSL.sepList [
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p] ∗
          globalSumFrame xBase outBase xBytes outBytes))) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_cons_frame_to_sepList_perm
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        (List.Perm.swap
          (CSL.reg 0 0 lane "i" (.s32 i))
          (CSL.reg 0 0 lane "acc" (.s32 acc))
          [CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p]))

theorem globalSumAt_to_ptr_focus
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.reg 0 0 lane "ptr" ptr ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 i),
            CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.pred 0 0 lane "p" p] ∗
            globalSumFrame xBase outBase xBytes outBytes))) := by
  have h₁ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr,
        CSL.pred 0 0 lane "p" p].Perm
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p] :=
    List.Perm.cons _ <|
      List.Perm.cons _ <|
        List.Perm.swap
          (CSL.reg 0 0 lane "ptr" ptr)
          (CSL.reg 0 0 lane "tmp" tmp)
          [CSL.pred 0 0 lane "p" p]
  have h₂ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "ptr" ptr,
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.pred 0 0 lane "p" p].Perm
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p] :=
    List.Perm.cons _ <|
      List.Perm.swap
        (CSL.reg 0 0 lane "ptr" ptr)
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        [CSL.reg 0 0 lane "tmp" tmp, CSL.pred 0 0 lane "p" p]
  have h₃ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "ptr" ptr,
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.pred 0 0 lane "p" p].Perm
        [CSL.reg 0 0 lane "ptr" ptr,
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p] :=
    List.Perm.swap
      (CSL.reg 0 0 lane "ptr" ptr)
      (CSL.reg 0 0 lane "i" (.s32 i))
      [CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.pred 0 0 lane "p" p]
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm_frame_to_cons
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (CSL.reg 0 0 lane "ptr" ptr)
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        ((h₁.trans h₂).trans h₃))

theorem globalSumAt_ptr_focus_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "ptr" ptr ∗
        (CSL.sepList [
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p] ∗
          globalSumFrame xBase outBase xBytes outBytes))) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  have h := globalSumAt_to_ptr_focus lane pc i acc tmp ptr p xBase outBase xBytes outBytes
  -- Rebuild the inverse permutation directly from the proved forward shape.
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_cons_frame_to_sepList_perm
        (CSL.reg 0 0 lane "ptr" ptr)
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p]
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        ((List.Perm.swap
          (CSL.reg 0 0 lane "ptr" ptr)
          (CSL.reg 0 0 lane "i" (.s32 i))
          [CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.pred 0 0 lane "p" p]).symm.trans
          ((List.Perm.cons _ <|
            List.Perm.swap
              (CSL.reg 0 0 lane "ptr" ptr)
              (CSL.reg 0 0 lane "acc" (.s32 acc))
              [CSL.reg 0 0 lane "tmp" tmp, CSL.pred 0 0 lane "p" p]).symm.trans
            (List.Perm.cons _ <|
              List.Perm.cons _ <|
                List.Perm.swap
                  (CSL.reg 0 0 lane "ptr" ptr)
                  (CSL.reg 0 0 lane "tmp" tmp)
                  [CSL.pred 0 0 lane "p" p]).symm)))

theorem globalSumAt_to_pred_focus
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.pred 0 0 lane "p" p ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 i),
            CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr] ∗
            globalSumFrame xBase outBase xBytes outBytes))) := by
  have h₁ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr,
        CSL.pred 0 0 lane "p" p].Perm
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "ptr" ptr] :=
    List.Perm.cons _ <|
      List.Perm.cons _ <|
        List.Perm.cons _ <|
          List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "ptr" ptr)
            []
  have h₂ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.pred 0 0 lane "p" p,
        CSL.reg 0 0 lane "ptr" ptr].Perm
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr] :=
    List.Perm.cons _ <|
      List.Perm.cons _ <|
        List.Perm.swap
          (CSL.pred 0 0 lane "p" p)
          (CSL.reg 0 0 lane "tmp" tmp)
          [CSL.reg 0 0 lane "ptr" ptr]
  have h₃ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.pred 0 0 lane "p" p,
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr].Perm
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr] :=
    List.Perm.cons _ <|
      List.Perm.swap
        (CSL.pred 0 0 lane "p" p)
        (CSL.reg 0 0 lane "acc" (.s32 acc))
        [CSL.reg 0 0 lane "tmp" tmp, CSL.reg 0 0 lane "ptr" ptr]
  have h₄ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.pred 0 0 lane "p" p,
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr].Perm
        [CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr] :=
    List.Perm.swap
      (CSL.pred 0 0 lane "p" p)
      (CSL.reg 0 0 lane "i" (.s32 i))
      [CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr]
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm_frame_to_cons
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (CSL.pred 0 0 lane "p" p)
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr]
        (globalSumFrame xBase outBase xBytes outBytes)
        (((h₁.trans h₂).trans h₃).trans h₄))

theorem globalSumAt_pred_focus_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane "p" p ∗
        (CSL.sepList [
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr] ∗
          globalSumFrame xBase outBase xBytes outBytes))) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  have h₁ :
      [CSL.reg 0 0 lane "i" (.s32 i),
        CSL.reg 0 0 lane "acc" (.s32 acc),
        CSL.reg 0 0 lane "tmp" tmp,
        CSL.reg 0 0 lane "ptr" ptr,
        CSL.pred 0 0 lane "p" p].Perm
        [CSL.pred 0 0 lane "p" p,
          CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr] :=
    ((List.Perm.cons _ <|
      List.Perm.cons _ <|
        List.Perm.cons _ <|
          List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "ptr" ptr)
            []).trans
      ((List.Perm.cons _ <|
        List.Perm.cons _ <|
          List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "tmp" tmp)
            [CSL.reg 0 0 lane "ptr" ptr]).trans
        ((List.Perm.cons _ <|
          List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "acc" (.s32 acc))
            [CSL.reg 0 0 lane "tmp" tmp, CSL.reg 0 0 lane "ptr" ptr]).trans
          (List.Perm.swap
            (CSL.pred 0 0 lane "p" p)
            (CSL.reg 0 0 lane "i" (.s32 i))
            [CSL.reg 0 0 lane "acc" (.s32 acc),
              CSL.reg 0 0 lane "tmp" tmp,
              CSL.reg 0 0 lane "ptr" ptr]))))
  simpa [globalSumAt, globalSumRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_cons_frame_to_sepList_perm
        (CSL.pred 0 0 lane "p" p)
        (CSL.reg 0 0 lane "i" (.s32 i))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr]
        [CSL.reg 0 0 lane "i" (.s32 i),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "tmp" tmp,
          CSL.reg 0 0 lane "ptr" ptr,
          CSL.pred 0 0 lane "p" p]
        (globalSumFrame xBase outBase xBytes outBytes)
        h₁.symm)

theorem globalSumAt_to_out_focus
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.globalBytes outBase .write outBytes ∗
          (globalSumRegs lane i acc tmp ptr p ∗ globalSumXResources xBase xBytes))) := by
  simpa [globalSumAt, globalSumFrame] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_rotate_three_last_to_front
        (globalSumRegs lane i acc tmp ptr p)
        (globalSumXResources xBase xBytes)
        (CSL.globalBytes outBase .write outBytes))

theorem globalSumAt_out_focus_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes outBase .write outBytes ∗
        (globalSumRegs lane i acc tmp ptr p ∗ globalSumXResources xBase xBytes))) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  simpa [globalSumAt, globalSumFrame] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_rotate_three_front_to_last
        (globalSumRegs lane i acc tmp ptr p)
        (globalSumXResources xBase xBytes)
        (CSL.globalBytes outBase .write outBytes))

theorem globalSumAt_to_flat
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        globalSumFlatResources lane i acc tmp ptr p xBase outBase xBytes outBytes) := by
  simpa [globalSumAt, globalSumRegs, globalSumFrame, globalSumXResources,
    globalSumFlatResources, globalSumProofN] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.entails_trans
        (CSL.sep_mono
          (CSL.entails_refl
            (CSL.sepList [
              CSL.reg 0 0 lane "i" (.s32 i),
              CSL.reg 0 0 lane "acc" (.s32 acc),
              CSL.reg 0 0 lane "tmp" tmp,
              CSL.reg 0 0 lane "ptr" ptr,
              CSL.pred 0 0 lane "p" p]))
          (CSL.sepList_append_cons
            (CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0))
            [CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1),
              CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2)]
            (CSL.globalBytes outBase .write outBytes) []))
        (CSL.sepList_append_cons
          (CSL.reg 0 0 lane "i" (.s32 i))
          [CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p]
          (CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0))
          [CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1),
            CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2),
            CSL.globalBytes outBase .write outBytes]))

theorem globalSumAt_flat_to_standard
    (lane : LaneId) (pc : PC) (i acc : Int) (tmp ptr : Value) (p : Bool)
    (xBase outBase : Nat) (xBytes : Nat → List Byte) (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      globalSumFlatResources lane i acc tmp ptr p xBase outBase xBytes outBytes) ⊢ₛ
      globalSumAt lane pc i acc tmp ptr p xBase outBase xBytes outBytes := by
  simpa [globalSumAt, globalSumRegs, globalSumFrame, globalSumXResources,
    globalSumFlatResources, globalSumProofN] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.entails_trans
        (CSL.sepList_append_cons_rev
          (CSL.reg 0 0 lane "i" (.s32 i))
          [CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" ptr,
            CSL.pred 0 0 lane "p" p]
          (CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0))
          [CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1),
            CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2),
            CSL.globalBytes outBase .write outBytes])
        (CSL.sep_mono
          (CSL.entails_refl
            (CSL.sepList [
              CSL.reg 0 0 lane "i" (.s32 i),
              CSL.reg 0 0 lane "acc" (.s32 acc),
              CSL.reg 0 0 lane "tmp" tmp,
              CSL.reg 0 0 lane "ptr" ptr,
              CSL.pred 0 0 lane "p" p]))
          (CSL.sepList_append_cons_rev
            (CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0))
            [CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1),
              CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2)]
            (CSL.globalBytes outBase .write outBytes) [])))

private theorem perm_abcdefghi_to_fcabdeghi
    (a b c d e f g h i : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i].Perm [f, c, a, b, d, e, g, h, i] := by
  have h1 : [a, b, c, d, e, f, g, h, i].Perm [f, a, b, c, d, e, g, h, i] := by
    simpa using (List.perm_middle (a := f) (l₁ := [a, b, c, d, e]) (l₂ := [g, h, i]))
  have h2 : [a, b, c, d, e, g, h, i].Perm [c, a, b, d, e, g, h, i] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, g, h, i]))
  exact h1.trans (List.Perm.cons f h2)

private theorem perm_abcdefghi_to_gcabdefhi
    (a b c d e f g h i : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i].Perm [g, c, a, b, d, e, f, h, i] := by
  have h1 : [a, b, c, d, e, f, g, h, i].Perm [g, a, b, c, d, e, f, h, i] := by
    simpa using (List.perm_middle (a := g) (l₁ := [a, b, c, d, e, f]) (l₂ := [h, i]))
  have h2 : [a, b, c, d, e, f, h, i].Perm [c, a, b, d, e, f, h, i] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, h, i]))
  exact h1.trans (List.Perm.cons g h2)

private theorem perm_abcdefghi_to_hcabdefgi
    (a b c d e f g h i : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i].Perm [h, c, a, b, d, e, f, g, i] := by
  have h1 : [a, b, c, d, e, f, g, h, i].Perm [h, a, b, c, d, e, f, g, i] := by
    simpa using (List.perm_middle (a := h) (l₁ := [a, b, c, d, e, f, g]) (l₂ := [i]))
  have h2 : [a, b, c, d, e, f, g, i].Perm [c, a, b, d, e, f, g, i] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, g, i]))
  exact h1.trans (List.Perm.cons h h2)

theorem globalSumRegs_preserve_global_store
    {lane : LaneId} {pc : PC} {i acc : Int} {tmp ptr : Value} {p : Bool}
    {outBase : Nat} {value : Value} {st stCore st' : State}
    {rCtrl rRegs : CSL.Resource}
    (hctrl : warpAt 0 0 pc [lane] st rCtrl)
    (hregs : globalSumRegs lane i acc tmp ptr p st rRegs)
    (haddr :
      ResolvesAddr st { cta := 0, warp := 0, lane := lane }
        { space := .global, ty := .s32, addr := .imm (.gaddr .global outBase) }
        (.global outBase))
    (heval : EvalRValue st { cta := 0, warp := 0, lane := lane } (.reg "acc") value)
    (hwrite : WriteMemFact st .global .s32 (.global outBase) value stCore)
    (hstep :
      Helpers.stepInstr? st 0 0
        { guard? := none,
          instr :=
            .store { space := .global, ty := .s32, addr := .imm (.gaddr .global outBase) }
              (.reg "acc") } = some st') :
    globalSumRegs lane i acc tmp ptr p st' rRegs := by
  unfold globalSumRegs at hregs ⊢
  rcases hregs with ⟨rI, rRest, hcompI, hequivI, hi, hrest⟩
  rcases hrest with ⟨rAcc, rRestAcc, hcompAcc, hequivAcc, hacc, hrestAcc⟩
  rcases hrestAcc with ⟨rTmp, rRestTmp, hcompTmp, hequivTmp, htmp, hrestTmp⟩
  rcases hrestTmp with ⟨rPtr, rPred, hcompPtr, hequivPtr, hptr, hp⟩
  refine ⟨rI, rRest, hcompI, hequivI, ?_, ?_⟩
  · exact globalStorePreservesReadReg_single_warpAt
      hctrl hi haddr heval hwrite hstep
  · refine ⟨rAcc, rRestAcc, hcompAcc, hequivAcc, ?_, ?_⟩
    · exact globalStorePreservesReadReg_single_warpAt
        hctrl hacc haddr heval hwrite hstep
    · refine ⟨rTmp, rRestTmp, hcompTmp, hequivTmp, ?_, ?_⟩
      · exact globalStorePreservesReadReg_single_warpAt
          hctrl htmp haddr heval hwrite hstep
      · refine ⟨rPtr, rPred, hcompPtr, hequivPtr, ?_, ?_⟩
        · exact globalStorePreservesReadReg_single_warpAt
            hctrl hptr haddr heval hwrite hstep
        · exact globalStorePreservesPred_single_warpAt
            hctrl hp haddr heval hwrite hstep

theorem globalSumXResources_preserve_global_store
    {lane : LaneId} {pc : PC} {xBase outBase : Nat} {xBytes : Nat → List Byte}
    {newOut : List Byte} {value : Value} {st stCore st' : State}
    {rCtrl rX : CSL.Resource}
    (hdisjoint :
      ∀ i, i < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * i) (xBytes i).length outBase newOut.length)
    (hctrl : warpAt 0 0 pc [lane] st rCtrl)
    (hx : globalSumXResources xBase xBytes st rX)
    (haddr :
      ResolvesAddr st { cta := 0, warp := 0, lane := lane }
        { space := .global, ty := .s32, addr := .imm (.gaddr .global outBase) }
        (.global outBase))
    (heval : EvalRValue st { cta := 0, warp := 0, lane := lane } (.reg "acc") value)
    (hwrite : WriteMemFact st .global .s32 (.global outBase) value stCore)
    (hencode : EncodedScalar .s32 value newOut)
    (hstep :
      Helpers.stepInstr? st 0 0
        { guard? := none,
          instr :=
            .store { space := .global, ty := .s32, addr := .imm (.gaddr .global outBase) }
              (.reg "acc") } = some st') :
    globalSumXResources xBase xBytes st' rX := by
  simp [globalSumXResources, globalSumProofN] at hx ⊢
  rcases hx with ⟨rX0, rRest, hcomp0, hequiv0, hx0, hxRest⟩
  rcases hxRest with ⟨rX1, rX2, hcomp1, hequiv1, hx1, hx2⟩
  refine ⟨rX0, rRest, hcomp0, hequiv0, ?_, ?_⟩
  · exact globalStorePreservesGlobalBytes_single_warpAt
      (by simpa [globalSumProofN] using hdisjoint 0 (by simp [globalSumProofN]))
      hctrl hx0 haddr heval hwrite hencode hstep
  · refine ⟨rX1, rX2, hcomp1, hequiv1, ?_, ?_⟩
    · exact globalStorePreservesGlobalBytes_single_warpAt
        (by simpa [globalSumProofN] using hdisjoint 1 (by simp [globalSumProofN]))
        hctrl hx1 haddr heval hwrite hencode hstep
    · exact globalStorePreservesGlobalBytes_single_warpAt
        (by simpa [globalSumProofN] using hdisjoint 2 (by simp [globalSumProofN]))
        hctrl hx2 haddr heval hwrite hencode hstep

theorem globalSum_entry_lookup (n xBase outBase : Nat) :
    (globalSumEnv n xBase outBase).blocks["entry"]? = some (globalSumEntryBlock xBase) := by
  rw [globalSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem globalSum_loop_lookup (n xBase outBase : Nat) :
    (globalSumEnv n xBase outBase).blocks["loop"]? = some (globalSumHeaderBlock n) := by
  rw [globalSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem globalSum_body_lookup (n xBase outBase : Nat) :
    (globalSumEnv n xBase outBase).blocks["body"]? = some globalSumBodyBlock := by
  rw [globalSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem globalSum_exit_lookup (n xBase outBase : Nat) :
    (globalSumEnv n xBase outBase).blocks["exit"]? = some (globalSumExitBlock outBase) := by
  rw [globalSumEnv]
  repeat rw [Std.HashMap.getElem?_insert]
  simp

theorem globalSum_block_lookup
    {n xBase outBase : Nat} {label : BlockLabel} {block : Block}
    (hlookup : (globalSumEnv n xBase outBase).blocks[label]? = some block) :
    (label = "entry" ∧ block = globalSumEntryBlock xBase) ∨
      (label = "loop" ∧ block = globalSumHeaderBlock n) ∨
        (label = "body" ∧ block = globalSumBodyBlock) ∨
          (label = "exit" ∧ block = globalSumExitBlock outBase) := by
  by_cases hentry : label = "entry"
  · subst label
    rw [globalSumEnv] at hlookup
    repeat rw [Std.HashMap.getElem?_insert] at hlookup
    simp at hlookup
    exact Or.inl ⟨rfl, hlookup.symm⟩
  · by_cases hloop : label = "loop"
    · subst label
      rw [globalSumEnv] at hlookup
      repeat rw [Std.HashMap.getElem?_insert] at hlookup
      simp at hlookup
      exact Or.inr (Or.inl ⟨rfl, hlookup.symm⟩)
    · by_cases hbody : label = "body"
      · subst label
        rw [globalSumEnv] at hlookup
        repeat rw [Std.HashMap.getElem?_insert] at hlookup
        simp at hlookup
        exact Or.inr (Or.inr (Or.inl ⟨rfl, hlookup.symm⟩))
      · by_cases hexit : label = "exit"
        · subst label
          rw [globalSumEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp at hlookup
          exact Or.inr (Or.inr (Or.inr ⟨rfl, hlookup.symm⟩))
        · have hentryEq : ¬ "entry" = label := fun h => hentry h.symm
          have hloopEq : ¬ "loop" = label := fun h => hloop h.symm
          have hbodyEq : ¬ "body" = label := fun h => hbody h.symm
          have hexitEq : ¬ "exit" = label := fun h => hexit h.symm
          rw [globalSumEnv] at hlookup
          repeat rw [Std.HashMap.getElem?_insert] at hlookup
          simp [hentryEq, hloopEq, hbodyEq, hexitEq] at hlookup

theorem globalSum_body_ordinary (n xBase outBase : Nat) :
    CFGBodyUsesOrdinaryPcAdvance (globalSumEnv n xBase outBase) := by
  intro label block idx gi hlookup hgi
  rcases globalSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [globalSumEntryBlock, globalSumSetI, globalSumSetAcc, globalSumSetPtr] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [globalSumEntryBlock, globalSumSetI, globalSumSetAcc, globalSumSetPtr] at hgi
            subst gi
            rfl
        | succ idx =>
            cases idx with
            | zero =>
                simp [globalSumEntryBlock, globalSumSetI, globalSumSetAcc,
                  globalSumSetPtr] at hgi
                subst gi
                rfl
            | succ idx =>
                simp [globalSumEntryBlock, globalSumSetI, globalSumSetAcc,
                  globalSumSetPtr] at hgi
  · rcases hloop with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [globalSumHeaderBlock, globalSumSetPred] at hgi
        subst gi
        rfl
    | succ idx =>
        simp [globalSumHeaderBlock, globalSumSetPred] at hgi
  · rcases hbody with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [globalSumBodyBlock, globalSumLoadTmp, globalSumAddAcc, globalSumIncI,
          globalSumIncPtr] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [globalSumBodyBlock, globalSumLoadTmp, globalSumAddAcc, globalSumIncI,
              globalSumIncPtr] at hgi
            subst gi
            rfl
        | succ idx =>
            cases idx with
            | zero =>
                simp [globalSumBodyBlock, globalSumLoadTmp, globalSumAddAcc,
                  globalSumIncI, globalSumIncPtr] at hgi
                subst gi
                rfl
            | succ idx =>
                cases idx with
                | zero =>
                    simp [globalSumBodyBlock, globalSumLoadTmp, globalSumAddAcc,
                      globalSumIncI, globalSumIncPtr] at hgi
                    subst gi
                    rfl
                | succ idx =>
                    simp [globalSumBodyBlock, globalSumLoadTmp, globalSumAddAcc,
                      globalSumIncI, globalSumIncPtr] at hgi
  · rcases hexit with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [globalSumExitBlock, globalSumStoreAcc] at hgi
        subst gi
        rfl
    | succ idx =>
        simp [globalSumExitBlock, globalSumStoreAcc] at hgi

theorem globalSum_targets_exist (n xBase outBase : Nat) :
    CFGTerminatorTargetsExist (globalSumEnv n xBase outBase) := by
  intro label block hlookup
  rcases globalSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    simp [globalSumEntryBlock]
    exact ⟨globalSumHeaderBlock n, globalSum_loop_lookup n xBase outBase⟩
  · rcases hloop with ⟨_, hblock⟩
    subst block
    simp [globalSumHeaderBlock]
    exact ⟨⟨globalSumBodyBlock, globalSum_body_lookup n xBase outBase⟩,
      globalSumExitBlock outBase, globalSum_exit_lookup n xBase outBase⟩
  · rcases hbody with ⟨_, hblock⟩
    subst block
    simp [globalSumBodyBlock]
    exact ⟨globalSumHeaderBlock n, globalSum_loop_lookup n xBase outBase⟩
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [globalSumExitBlock]

theorem globalSum_entry_term_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut) "entry" (.br "loop")
      (globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP) := by
  refine TerminatorVC.br ?_
  have hbr :
      globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (globalSumAt lane ("loop", 0) 0 0 oldTmp (.gaddr .global xBase) oldP
            xBase outBase xBytes oldOut) := by
    simpa [globalSumEntryTermPre, globalSumAt] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("entry", 3)) (target := "loop")
        (lanes := [lane])
        (frame :=
          globalSumRegs lane 0 0 oldTmp (.gaddr .global xBase) oldP ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (globalSumAt_stable_terminator_frame (.br "loop") lane 0 0 oldTmp
          (.gaddr .global xBase) oldP xBase outBase xBytes oldOut))
  exact CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st r hpost
      refine ⟨0, oldTmp, oldP, ?_, ?_⟩
      · simp [globalSumProofN]
      · simpa using hpost)

theorem globalSum_entry_instrs_wp
    (lane : LaneId) (xBase outBase : Nat) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    globalSumEntryPre lane xBase outBase xBytes oldOut oldI oldAcc oldTmp oldPtr oldP ⊢ₛ
      wpInstrs 0 0 (globalSumEntryBlock xBase).body
        (globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP) := by
  let frameI :=
    CSL.sepList [
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "tmp" oldTmp,
      CSL.reg 0 0 lane "ptr" oldPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      globalSumFrame xBase outBase xBytes oldOut
  have hsetI :
      globalSumEntryPre lane xBase outBase xBytes oldOut
          oldI oldAcc oldTmp oldPtr oldP ⊢ₛ
        wpInstr 0 0 globalSumSetI
          (globalSumEntryAfterI lane xBase outBase xBytes oldOut
            oldAcc oldTmp oldPtr oldP) := by
    have hfocus :
        globalSumEntryPre lane xBase outBase xBytes oldOut
            oldI oldAcc oldTmp oldPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 0) [lane] ∗
            (CSL.reg 0 0 lane "i" oldI ∗ frameI)) := by
      simpa [globalSumEntryPre, frameI] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 0) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "i" oldI,
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "i" oldI)
            (CSL.reg 0 0 lane "acc" oldAcc)
            [CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (globalSumFrame xBase outBase xBytes oldOut)
            (List.Perm.refl _))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 globalSumSetI) frameI := by
      unfold frameI
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_pred_assignReg
      · exact globalSumFrame_stable_assignReg "i" (.imm (.s32 0))
          xBase outBase xBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 0) [lane] ∗
          (CSL.reg 0 0 lane "i" oldI ∗ frameI)) ⊢ₛ
          wpInstr 0 0 globalSumSetI
            (warpAt 0 0 ("entry", 1) [lane] ∗
              (CSL.reg 0 0 lane "i" (.s32 0) ∗ frameI)) := by
      simpa [globalSumSetI, frameI] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 0)) (dst := "i")
          (rhs := .imm (.s32 0)) (lane := lane) (old := oldI) (new := .s32 0)
          (frame := frameI)
          (by
            intro _st _r _hpre
            exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 1) [lane] ∗
          (CSL.reg 0 0 lane "i" (.s32 0) ∗ frameI)) ⊢ₛ
          globalSumEntryAfterI lane xBase outBase xBytes oldOut
            oldAcc oldTmp oldPtr oldP := by
      simpa [globalSumEntryAfterI, frameI] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 1) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "i" (.s32 0))
            (CSL.reg 0 0 lane "acc" oldAcc)
            [CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (globalSumFrame xBase outBase xBytes oldOut)
            (List.Perm.refl _))
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  let frameAcc :=
    CSL.sepList [
      CSL.reg 0 0 lane "i" (.s32 0),
      CSL.reg 0 0 lane "tmp" oldTmp,
      CSL.reg 0 0 lane "ptr" oldPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      globalSumFrame xBase outBase xBytes oldOut
  have hsetAcc :
      globalSumEntryAfterI lane xBase outBase xBytes oldOut
          oldAcc oldTmp oldPtr oldP ⊢ₛ
        wpInstr 0 0 globalSumSetAcc
          (globalSumEntryAfterAcc lane xBase outBase xBytes oldOut oldTmp oldPtr oldP) := by
    have hfocus :
        globalSumEntryAfterI lane xBase outBase xBytes oldOut
            oldAcc oldTmp oldPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 1) [lane] ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗ frameAcc)) := by
      simpa [globalSumEntryAfterI, frameAcc] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 1) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "acc" oldAcc)
            (CSL.reg 0 0 lane "i" (.s32 0))
            [CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (globalSumFrame xBase outBase xBytes oldOut)
            (List.Perm.swap
              (CSL.reg 0 0 lane "acc" oldAcc)
              (CSL.reg 0 0 lane "i" (.s32 0))
              [CSL.reg 0 0 lane "tmp" oldTmp,
                CSL.reg 0 0 lane "ptr" oldPtr,
                CSL.pred 0 0 lane "p" oldP]))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 globalSumSetAcc) frameAcc := by
      unfold frameAcc
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_pred_assignReg
      · exact globalSumFrame_stable_assignReg "acc" (.imm (.s32 0))
          xBase outBase xBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗ frameAcc)) ⊢ₛ
          wpInstr 0 0 globalSumSetAcc
            (warpAt 0 0 ("entry", 2) [lane] ∗
              (CSL.reg 0 0 lane "acc" (.s32 0) ∗ frameAcc)) := by
      simpa [globalSumSetAcc, frameAcc] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 1)) (dst := "acc")
          (rhs := .imm (.s32 0)) (lane := lane) (old := oldAcc) (new := .s32 0)
          (frame := frameAcc)
          (by
            intro _st _r _hpre
            exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 2) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 0) ∗ frameAcc)) ⊢ₛ
          globalSumEntryAfterAcc lane xBase outBase xBytes oldOut oldTmp oldPtr oldP := by
      simpa [globalSumEntryAfterAcc, frameAcc] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 2) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "acc" (.s32 0))
            (CSL.reg 0 0 lane "i" (.s32 0))
            [CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.reg 0 0 lane "ptr" oldPtr,
              CSL.pred 0 0 lane "p" oldP]
            (globalSumFrame xBase outBase xBytes oldOut)
            (List.Perm.swap
              (CSL.reg 0 0 lane "i" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              [CSL.reg 0 0 lane "tmp" oldTmp,
                CSL.reg 0 0 lane "ptr" oldPtr,
                CSL.pred 0 0 lane "p" oldP]))
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  have hsetPtr :
      globalSumEntryAfterAcc lane xBase outBase xBytes oldOut oldTmp oldPtr oldP ⊢ₛ
        wpInstr 0 0 (globalSumSetPtr xBase)
          (globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP) := by
    have hfocus :=
      globalSumAt_to_ptr_focus lane ("entry", 2) 0 0 oldTmp oldPtr oldP
        xBase outBase xBytes oldOut
    have hstable :
        CSL.StableUnder (InstrStep 0 0 (globalSumSetPtr xBase))
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 0),
            CSL.reg 0 0 lane "acc" (.s32 0),
            CSL.reg 0 0 lane "tmp" oldTmp,
            CSL.pred 0 0 lane "p" oldP] ∗
            globalSumFrame xBase outBase xBytes oldOut) := by
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_reg_assignReg_of_ne (by decide)
        · subst q
          exact stable_pred_assignReg
      · exact globalSumFrame_stable_assignReg "ptr" (.imm (.gaddr .global xBase))
          xBase outBase xBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 2) [lane] ∗
          (CSL.reg 0 0 lane "ptr" oldPtr ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.pred 0 0 lane "p" oldP] ∗
              globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
          wpInstr 0 0 (globalSumSetPtr xBase)
            (warpAt 0 0 ("entry", 3) [lane] ∗
              (CSL.reg 0 0 lane "ptr" (.gaddr .global xBase) ∗
                (CSL.sepList [
                  CSL.reg 0 0 lane "i" (.s32 0),
                  CSL.reg 0 0 lane "acc" (.s32 0),
                  CSL.reg 0 0 lane "tmp" oldTmp,
                  CSL.pred 0 0 lane "p" oldP] ∗
                  globalSumFrame xBase outBase xBytes oldOut))) := by
      simpa [globalSumSetPtr] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 2)) (dst := "ptr")
          (rhs := .imm (.gaddr .global xBase)) (lane := lane) (old := oldPtr)
          (new := .gaddr .global xBase)
          (frame :=
            CSL.sepList [
              CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.pred 0 0 lane "p" oldP] ∗
              globalSumFrame xBase outBase xBytes oldOut)
          (by
            intro _st _r _hpre
            exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 3) [lane] ∗
          (CSL.reg 0 0 lane "ptr" (.gaddr .global xBase) ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "i" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "tmp" oldTmp,
              CSL.pred 0 0 lane "p" oldP] ∗
              globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
          globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP := by
      simpa [globalSumEntryTermPre] using
        globalSumAt_ptr_focus_to_standard lane ("entry", 3) 0 0 oldTmp
          (.gaddr .global xBase) oldP xBase outBase xBytes oldOut
    exact CSL.entails_trans
      (by
        intro st r hpre
        exact hfocus st r (by
          simpa [globalSumEntryAfterAcc, globalSumAt, globalSumRegs] using hpre))
      (CSL.entails_trans hrule (wpInstr_mono hpost))
  simpa [globalSumEntryBlock, wpInstrs, wpInstrList, globalSumSetI,
    globalSumSetAcc, globalSumSetPtr] using
    CSL.entails_trans hsetI (wpInstr_mono (CSL.entails_trans hsetAcc (wpInstr_mono hsetPtr)))

theorem globalSum_cbr_true_raw_wp
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (i : Nat) (tmp : Value) :
    globalSumAt lane ("loop", 1) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
        (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (globalSumAt lane ("body", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut) := by
  simpa [globalSumAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred true) (takeTrue := true)
      (frame :=
        globalSumRegs lane (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true ∗
          globalSumFrame xBase outBase xBytes oldOut)
      (by
        intro st r hpre
        have hfocus :=
          globalSumAt_to_pred_focus lane ("loop", 1) (Int.ofNat i)
            (globalSumPrefixS32 xs i) tmp (.gaddr .global (xBase + 4 * i))
            true xBase outBase xBytes oldOut st r (by
              simpa [globalSumAt] using hpre)
        rcases hfocus with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, hpred, _hframe⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (globalSumAt_stable_terminator_frame (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
        (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut))

theorem globalSum_cbr_false_raw_wp
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (tmp : Value) :
    globalSumAt lane ("loop", 1) (Int.ofNat globalSumProofN)
        (globalSumPrefixS32 xs globalSumProofN) tmp
        (.gaddr .global (xBase + 4 * globalSumProofN)) false
        xBase outBase xBytes oldOut ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (globalSumAt lane ("exit", 0) (Int.ofNat globalSumProofN)
          (globalSumPrefixS32 xs globalSumProofN) tmp
          (.gaddr .global (xBase + 4 * globalSumProofN)) false
          xBase outBase xBytes oldOut) := by
  simpa [globalSumAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred false) (takeTrue := false)
      (frame :=
        globalSumRegs lane (Int.ofNat globalSumProofN)
          (globalSumPrefixS32 xs globalSumProofN) tmp
          (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
          globalSumFrame xBase outBase xBytes oldOut)
      (by
        intro st r hpre
        have hfocus :=
          globalSumAt_to_pred_focus lane ("loop", 1) (Int.ofNat globalSumProofN)
            (globalSumPrefixS32 xs globalSumProofN) tmp
            (.gaddr .global (xBase + 4 * globalSumProofN)) false
            xBase outBase xBytes oldOut st r (by
              simpa [globalSumAt] using hpre)
        rcases hfocus with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, hpred, _hframe⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (globalSumAt_stable_terminator_frame (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat globalSumProofN) (globalSumPrefixS32 xs globalSumProofN) tmp
        (.gaddr .global (xBase + 4 * globalSumProofN)) false
        xBase outBase xBytes oldOut))

theorem globalSum_cbr_true_control
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
      (globalSumHeaderTruePre lane xBase outBase xs xBytes oldOut) := by
  intro st st' r hpre hstep
  rcases hpre with ⟨i, tmp, _hlt, hat⟩
  rcases globalSum_cbr_true_raw_wp lane xBase outBase xs xBytes oldOut i tmp
      st r hat st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem globalSum_cbr_false_control
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
      (globalSumHeaderFalsePre lane xBase outBase xs xBytes oldOut) := by
  intro st st' r hpre hstep
  rcases hpre with ⟨tmp, hat⟩
  rcases globalSum_cbr_false_raw_wp lane xBase outBase xs xBytes oldOut tmp
      st r hat st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem globalSum_header_term_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut) "loop"
      (.cbr (.pred "p") "body" "exit")
      (globalSumHeaderTermPre lane xBase outBase xs xBytes oldOut) := by
  refine TerminatorVC.cbr
    (truePre := globalSumHeaderTruePre lane xBase outBase xs xBytes oldOut)
    (falsePre := globalSumHeaderFalsePre lane xBase outBase xs xBytes oldOut)
    ?_ (globalSum_cbr_true_control lane xBase outBase xs xBytes oldOut)
    (globalSum_cbr_false_control lane xBase outBase xs xBytes oldOut) ?_ ?_
  · intro st r hpre
    simpa [globalSumHeaderTermPre] using hpre
  · intro st r hpre
    rcases hpre with ⟨i, tmp, hlt, hat⟩
    exact (CSL.entails_trans
      (globalSum_cbr_true_raw_wp lane xBase outBase xs xBytes oldOut i tmp) <|
      wpTerminator_mono (by
        intro st' r' hbody
        exact ⟨i, tmp, hlt, hbody⟩)) st r hat
  · intro st r hpre
    rcases hpre with ⟨tmp, hat⟩
    exact (CSL.entails_trans
      (globalSum_cbr_false_raw_wp lane xBase outBase xs xBytes oldOut tmp) <|
      wpTerminator_mono (by
        intro st' r' hexit
        exact ⟨tmp, hexit⟩)) st r hat

theorem globalSum_header_instrs_wp
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) :
    globalSumLoopInv lane xBase outBase xs xBytes oldOut ⊢ₛ
      wpInstrs 0 0 (globalSumHeaderBlock globalSumProofN).body
        (globalSumHeaderTermPre lane xBase outBase xs xBytes oldOut) := by
  intro st r hpre
  rcases hpre with ⟨i, tmp, p, hiLe, hat⟩
  have htoRule :=
    globalSumAt_to_pred_focus lane ("loop", 0) (Int.ofNat i)
      (globalSumPrefixS32 xs i) tmp (.gaddr .global (xBase + 4 * i)) p
      xBase outBase xBytes oldOut
  have hrule :
      (warpAt 0 0 ("loop", 0) [lane] ∗
        (CSL.pred 0 0 lane "p" p ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs i)),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i))] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        wpInstr 0 0 (globalSumSetPred globalSumProofN)
          (warpAt 0 0 ("loop", 1) [lane] ∗
            (CSL.pred 0 0 lane "p"
              (decide (Int.ofNat i < Int.ofNat globalSumProofN)) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
                CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs i)),
                CSL.reg 0 0 lane "tmp" tmp,
                CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i))] ∗
                globalSumFrame xBase outBase xBytes oldOut))) := by
    simpa [globalSumSetPred] using
      (wp_assignPred_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("loop", 0)) (dst := "p")
        (cmp := { op := .lt, lhs := .reg "i", rhs := .imm (.s32 (Int.ofNat globalSumProofN)) })
        (lane := lane) (old := p)
        (new := decide (Int.ofNat i < Int.ofNat globalSumProofN))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs i)),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i))] ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, _hpred,
            hframe⟩
          rcases hframe with ⟨_rRegs, _rMem, _hcompFrame, _hequivFrame, hregs, _hmem⟩
          rcases hregs with ⟨_rI, _rRegsRest, _hcompRegs, _hequivRegs, hi, _hregsRest⟩
          exact EvalCmp.lt_s32 (eval_reg_of_assertion hi) eval_imm)
        (by
          refine CSL.stable_sep ?_ ?_
          · apply CSL.stable_sepList
            intro q hq
            simp at hq
            rcases hq with hq | hq | hq | hq
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
          · exact globalSumFrame_stable_assignPred "p"
              { op := .lt, lhs := .reg "i",
                rhs := .imm (.s32 (Int.ofNat globalSumProofN)) }
              xBase outBase xBytes oldOut))
  have hpost :
      (warpAt 0 0 ("loop", 1) [lane] ∗
        (CSL.pred 0 0 lane "p" (decide (Int.ofNat i < Int.ofNat globalSumProofN)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs i)),
            CSL.reg 0 0 lane "tmp" tmp,
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i))] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        globalSumHeaderTermPre lane xBase outBase xs xBytes oldOut := by
    intro st r hraw
    have hstandard :
        globalSumAt lane ("loop", 1) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i))
          (decide (Int.ofNat i < Int.ofNat globalSumProofN))
          xBase outBase xBytes oldOut st r :=
      globalSumAt_pred_focus_to_standard lane ("loop", 1) (Int.ofNat i)
        (globalSumPrefixS32 xs i) tmp (.gaddr .global (xBase + 4 * i))
        (decide (Int.ofNat i < Int.ofNat globalSumProofN))
        xBase outBase xBytes oldOut st r hraw
    by_cases hlt : i < globalSumProofN
    · have hltInt : Int.ofNat i < Int.ofNat globalSumProofN := Int.ofNat_lt.mpr hlt
      have hdec :
          decide (Int.ofNat i < Int.ofNat globalSumProofN) = true := by
        simp [hltInt, hlt]
      left
      refine ⟨i, tmp, hlt, ?_⟩
      rw [hdec] at hstandard
      exact hstandard
    · have hge : globalSumProofN ≤ i := Nat.le_of_not_gt hlt
      have hiEq : i = globalSumProofN := Nat.le_antisymm hiLe hge
      subst i
      have hnotInt :
          ¬ Int.ofNat globalSumProofN < Int.ofNat globalSumProofN := by
        exact Int.lt_irrefl (Int.ofNat globalSumProofN)
      have hdec :
          decide (Int.ofNat globalSumProofN < Int.ofNat globalSumProofN) = false := by
        simp [hnotInt]
      right
      refine ⟨tmp, ?_⟩
      rw [hdec] at hstandard
      simpa [globalSumHeaderFalsePre] using hstandard
  simpa [globalSumHeaderBlock, wpInstrs, wpInstrList, globalSumSetPred] using
    (CSL.entails_trans htoRule (CSL.entails_trans hrule (wpInstr_mono hpost)) st r hat)

theorem globalSum_body_term_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut) "body" (.br "loop")
      (globalSumBodyTermPre lane xBase outBase xs xBytes oldOut) := by
  refine TerminatorVC.br ?_
  intro st r hpre
  rcases hpre with ⟨i, tmp, hiLe, hat⟩
  have hbr :
      globalSumAt lane ("body", 4) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (globalSumAt lane ("loop", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
            (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut) := by
    simpa [globalSumAt] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 4)) (target := "loop")
        (lanes := [lane])
        (frame :=
          globalSumRegs lane (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
            (.gaddr .global (xBase + 4 * i)) true ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (globalSumAt_stable_terminator_frame (.br "loop") lane
          (Int.ofNat i) (globalSumPrefixS32 xs i) tmp
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut))
  exact (CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st' r' hloop
      exact ⟨i, tmp, true, hiLe, hloop⟩)) st r hat

theorem globalSum_exit_term_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut) "exit" .terminate
      (globalSumExitTermPre lane xBase outBase xs xBytes newOut) := by
  refine TerminatorVC.terminate ?_
  intro st r hpre
  rcases hpre with ⟨tmp, hat⟩
  have hterm :
      globalSumAt lane ("exit", 1) (Int.ofNat globalSumProofN)
          (globalSumPrefixS32 xs globalSumProofN) tmp
          (.gaddr .global (xBase + 4 * globalSumProofN)) false
          xBase outBase xBytes newOut ⊢ₛ
        wpTerminator 0 0 .terminate
          ((laneTerminatedAt 0 0 lane ("exit", 1)) ∗
            (globalSumRegs lane (Int.ofNat globalSumProofN)
              (globalSumPrefixS32 xs globalSumProofN) tmp
              (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
              globalSumFrame xBase outBase xBytes newOut)) := by
    simpa [globalSumAt] using
      (wp_terminate_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("exit", 1)) (lane := lane)
        (frame :=
          globalSumRegs lane (Int.ofNat globalSumProofN)
            (globalSumPrefixS32 xs globalSumProofN) tmp
            (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
            globalSumFrame xBase outBase xBytes newOut)
        (globalSumAt_stable_terminator_frame .terminate lane
          (Int.ofNat globalSumProofN) (globalSumPrefixS32 xs globalSumProofN) tmp
          (.gaddr .global (xBase + 4 * globalSumProofN)) false
          xBase outBase xBytes newOut))
  exact (CSL.entails_trans hterm <|
    wpTerminator_mono (by
      intro st' r' hpost
      exact ⟨tmp, hpost⟩)) st r hat

theorem globalSum_exit_instrs_wp
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte)
    (haccessOut : AccessOk .global .s32 (.global outBase))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (globalSumPrefixS32 xs globalSumProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointXOut :
      ∀ i, i < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * i) (xBytes i).length outBase newOut.length) :
    globalSumExitInv lane xBase outBase xs xBytes oldOut ⊢ₛ
      wpInstrs 0 0 (globalSumExitBlock outBase).body
        (globalSumExitTermPre lane xBase outBase xs xBytes newOut) := by
  intro st r hpre
  rcases hpre with ⟨tmp, hat⟩
  have hfocus :=
    globalSumAt_to_out_focus lane ("exit", 0) (Int.ofNat globalSumProofN)
      (globalSumPrefixS32 xs globalSumProofN) tmp
      (.gaddr .global (xBase + 4 * globalSumProofN)) false
      xBase outBase xBytes oldOut
  have hrule :
      (warpAt 0 0 ("exit", 0) [lane] ∗
        (CSL.globalBytes outBase .write oldOut ∗
          (globalSumRegs lane (Int.ofNat globalSumProofN)
            (globalSumPrefixS32 xs globalSumProofN) tmp
            (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
            globalSumXResources xBase xBytes))) ⊢ₛ
        wpInstr 0 0 (globalSumStoreAcc outBase)
          (warpAt 0 0 ("exit", 1) [lane] ∗
            (CSL.globalBytes outBase .write newOut ∗
              (globalSumRegs lane (Int.ofNat globalSumProofN)
                (globalSumPrefixS32 xs globalSumProofN) tmp
                (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
                globalSumXResources xBase xBytes))) := by
    simpa [globalSumStoreAcc] using
      (wp_globalStoreBytes_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("exit", 0))
        (ty := .s32) (addrExpr := .imm (.gaddr .global outBase))
        (valueExpr := .reg "acc") (lane := lane) (offset := outBase)
        (oldBytes := oldOut) (newBytes := newOut)
        (value := .s32 (globalSumPrefixS32 xs globalSumProofN))
        (frame :=
          globalSumRegs lane (Int.ofNat globalSumProofN)
            (globalSumPrefixS32 xs globalSumProofN) tmp
            (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
            globalSumXResources xBase xBytes)
        (haddr := by
          intro st r _hpre
          exact resolves_global_gaddr_of_eval (st := st)
            (ctx := { cta := 0, warp := 0, lane := lane })
            (ty := .s32) (expr := .imm (.gaddr .global outBase))
            (off := outBase) (by rfl))
        (heval := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
          rcases hframe with ⟨_rRegs, _rX, _hcompFrame, _hequivFrame, hregs, _hx⟩
          unfold globalSumRegs at hregs
          rcases hregs with ⟨_rI, _rRestRegs, _hcompI, _hequivI, _hi, hrestRegs⟩
          rcases hrestRegs with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, hacc,
            _hrestAcc⟩
          exact eval_reg_of_assertion hacc)
        (hwrite := by
          intro st r _hpre
          exact ⟨{ st with global := {
              bytes := Helpers.writeBytes st.global.bytes outBase newOut } },
            globalWriteMem_of_byteWrite haccessOut hencodeOut (by rfl)⟩)
        (hencode := hencodeOut)
        (hlen := hlenOut)
        (hframe := by
          intro st st' r rFrame hpre hframe hstep
          have haddr' := resolves_global_gaddr_of_eval (st := st)
            (ctx := { cta := 0, warp := 0, lane := lane })
            (ty := .s32) (expr := .imm (.gaddr .global outBase))
            (off := outBase) (by rfl)
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
          rcases hframe with ⟨rRegs, rX, hcompFrame, hequivFrame, hregs, hx⟩
          have hregsFull := hregs
          unfold globalSumRegs at hregs
          rcases hregs with ⟨_rI, _rRestRegs, _hcompI, _hequivI, _hi, hrestRegs⟩
          rcases hrestRegs with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, hacc,
            _hrestAcc⟩
          have heval' :
              EvalRValue st { cta := 0, warp := 0, lane := lane } (.reg "acc")
                (.s32 (globalSumPrefixS32 xs globalSumProofN)) :=
            eval_reg_of_assertion hacc
          have hwrite' :
              WriteMemFact st .global .s32 (.global outBase)
                (.s32 (globalSumPrefixS32 xs globalSumProofN))
                { st with global := {
                    bytes := Helpers.writeBytes st.global.bytes outBase newOut } } :=
            globalWriteMem_of_byteWrite haccessOut hencodeOut (by rfl)
          exact ⟨rRegs, rX, hcompFrame, hequivFrame,
            globalSumRegs_preserve_global_store
              hctrl hregsFull haddr' heval' hwrite' hstep,
            globalSumXResources_preserve_global_store
              hdisjointXOut hctrl hx haddr' heval' hwrite' hencodeOut hstep⟩))
  have hpost :
      (warpAt 0 0 ("exit", 1) [lane] ∗
        (CSL.globalBytes outBase .write newOut ∗
          (globalSumRegs lane (Int.ofNat globalSumProofN)
            (globalSumPrefixS32 xs globalSumProofN) tmp
            (.gaddr .global (xBase + 4 * globalSumProofN)) false ∗
            globalSumXResources xBase xBytes))) ⊢ₛ
        globalSumExitTermPre lane xBase outBase xs xBytes newOut := by
    intro st r hraw
    refine ⟨tmp, ?_⟩
    exact globalSumAt_out_focus_to_standard lane ("exit", 1)
      (Int.ofNat globalSumProofN) (globalSumPrefixS32 xs globalSumProofN) tmp
      (.gaddr .global (xBase + 4 * globalSumProofN)) false
      xBase outBase xBytes newOut st r hraw
  simpa [globalSumExitBlock, wpInstrs, wpInstrList, globalSumStoreAcc] using
    (CSL.entails_trans
      (by
        intro st r h
        exact hfocus st r h)
      (CSL.entails_trans hrule (wpInstr_mono hpost)) st r hat)

theorem globalSum_load_tmp_wp_at_zero
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldTmp : Value)
    (haccess : AccessOk .global .s32 (.global (xBase + 4 * 0)))
    (hwidth : Typing.byteWidth? .s32 = some (xBytes 0).length)
    (hdecode : DecodedScalar .s32 (xBytes 0) (.s32 (xs 0))) :
    globalSumAt lane ("body", 0) (Int.ofNat 0) (globalSumPrefixS32 xs 0) oldTmp
        (.gaddr .global (xBase + 4 * 0)) true xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumLoadTmp
        (globalSumAt lane ("body", 1) (Int.ofNat 0) (globalSumPrefixS32 xs 0)
          (.s32 (xs 0)) (.gaddr .global (xBase + 4 * 0)) true
          xBase outBase xBytes oldOut) := by
  let regI := CSL.reg 0 0 lane "i" (.s32 (Int.ofNat 0))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs 0))
  let regTmpOld := CSL.reg 0 0 lane "tmp" oldTmp
  let regTmpNew := CSL.reg 0 0 lane "tmp" (.s32 (xs 0))
  let regPtr := CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * 0))
  let predP := CSL.pred 0 0 lane "p" true
  let x0 := CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0)
  let x1 := CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1)
  let x2 := CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2)
  let out := CSL.globalBytes outBase .write oldOut
  let frame := CSL.sepList [regI, regAcc, regPtr, predP, x1, x2, out]
  have hpreToRule :
      globalSumAt lane ("body", 0) (Int.ofNat 0) (globalSumPrefixS32 xs 0) oldTmp
          (.gaddr .global (xBase + 4 * 0)) true xBase outBase xBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((x0 ∗ regTmpOld) ∗ frame)) := by
    exact CSL.entails_trans
      (globalSumAt_to_flat lane ("body", 0) (Int.ofNat 0) (globalSumPrefixS32 xs 0)
        oldTmp (.gaddr .global (xBase + 4 * 0)) true xBase outBase xBytes oldOut)
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpOld, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regI, regAcc, regTmpOld, regPtr, predP, x0, x1, x2, out]
              x0 regTmpOld regI [regAcc, regPtr, predP, x1, x2, out]
              (perm_abcdefghi_to_fcabdeghi
                regI regAcc regTmpOld regPtr predP x0 x1 x2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((x0 ∗ regTmpNew) ∗ frame)) ⊢ₛ
        globalSumAt lane ("body", 1) (Int.ofNat 0) (globalSumPrefixS32 xs 0)
          (.s32 (xs 0)) (.gaddr .global (xBase + 4 * 0)) true
          xBase outBase xBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpNew, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              x0 regTmpNew regI [regAcc, regPtr, predP, x1, x2, out]
              [regI, regAcc, regTmpNew, regPtr, predP, x0, x1, x2, out]
              (perm_abcdefghi_to_fcabdeghi
                regI regAcc regTmpNew regPtr predP x0 x1 x2 out).symm))
      (globalSumAt_flat_to_standard lane ("body", 1) (Int.ofNat 0)
        (globalSumPrefixS32 xs 0) (.s32 (xs 0))
        (.gaddr .global (xBase + 4 * 0)) true xBase outBase xBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((x0 ∗ regTmpOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 globalSumLoadTmp
          (warpAt 0 0 ("body", 1) [lane] ∗ ((x0 ∗ regTmpNew) ∗ frame)) := by
    simpa [globalSumLoadTmp, x0, regTmpOld, regTmpNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "tmp")
        (ty := .s32) (addrExpr := .reg "ptr") (lane := lane)
        (offset := xBase + 4 * 0) (bytes := xBytes 0) (oldReg := oldTmp)
        (value := .s32 (xs 0)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rI, _rRestI, _hcompI, _hequivI, _hi, hrestI⟩
          rcases hrestI with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rPtr, _rRestPtr, _hcompPtr, _hequivPtr, hptr,
            _hrestPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "ptr") (off := xBase + 4 * 0)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regI, regAcc, regPtr, predP, x1, x2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_pred_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem globalSum_load_tmp_wp_at_one
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldTmp : Value)
    (haccess : AccessOk .global .s32 (.global (xBase + 4 * 1)))
    (hwidth : Typing.byteWidth? .s32 = some (xBytes 1).length)
    (hdecode : DecodedScalar .s32 (xBytes 1) (.s32 (xs 1))) :
    globalSumAt lane ("body", 0) (Int.ofNat 1) (globalSumPrefixS32 xs 1) oldTmp
        (.gaddr .global (xBase + 4 * 1)) true xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumLoadTmp
        (globalSumAt lane ("body", 1) (Int.ofNat 1) (globalSumPrefixS32 xs 1)
          (.s32 (xs 1)) (.gaddr .global (xBase + 4 * 1)) true
          xBase outBase xBytes oldOut) := by
  let regI := CSL.reg 0 0 lane "i" (.s32 (Int.ofNat 1))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs 1))
  let regTmpOld := CSL.reg 0 0 lane "tmp" oldTmp
  let regTmpNew := CSL.reg 0 0 lane "tmp" (.s32 (xs 1))
  let regPtr := CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * 1))
  let predP := CSL.pred 0 0 lane "p" true
  let x0 := CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0)
  let x1 := CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1)
  let x2 := CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2)
  let out := CSL.globalBytes outBase .write oldOut
  let frame := CSL.sepList [regI, regAcc, regPtr, predP, x0, x2, out]
  have hpreToRule :
      globalSumAt lane ("body", 0) (Int.ofNat 1) (globalSumPrefixS32 xs 1) oldTmp
          (.gaddr .global (xBase + 4 * 1)) true xBase outBase xBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((x1 ∗ regTmpOld) ∗ frame)) := by
    exact CSL.entails_trans
      (globalSumAt_to_flat lane ("body", 0) (Int.ofNat 1) (globalSumPrefixS32 xs 1)
        oldTmp (.gaddr .global (xBase + 4 * 1)) true xBase outBase xBytes oldOut)
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpOld, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regI, regAcc, regTmpOld, regPtr, predP, x0, x1, x2, out]
              x1 regTmpOld regI [regAcc, regPtr, predP, x0, x2, out]
              (perm_abcdefghi_to_gcabdefhi
                regI regAcc regTmpOld regPtr predP x0 x1 x2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((x1 ∗ regTmpNew) ∗ frame)) ⊢ₛ
        globalSumAt lane ("body", 1) (Int.ofNat 1) (globalSumPrefixS32 xs 1)
          (.s32 (xs 1)) (.gaddr .global (xBase + 4 * 1)) true
          xBase outBase xBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpNew, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              x1 regTmpNew regI [regAcc, regPtr, predP, x0, x2, out]
              [regI, regAcc, regTmpNew, regPtr, predP, x0, x1, x2, out]
              (perm_abcdefghi_to_gcabdefhi
                regI regAcc regTmpNew regPtr predP x0 x1 x2 out).symm))
      (globalSumAt_flat_to_standard lane ("body", 1) (Int.ofNat 1)
        (globalSumPrefixS32 xs 1) (.s32 (xs 1))
        (.gaddr .global (xBase + 4 * 1)) true xBase outBase xBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((x1 ∗ regTmpOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 globalSumLoadTmp
          (warpAt 0 0 ("body", 1) [lane] ∗ ((x1 ∗ regTmpNew) ∗ frame)) := by
    simpa [globalSumLoadTmp, x1, regTmpOld, regTmpNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "tmp")
        (ty := .s32) (addrExpr := .reg "ptr") (lane := lane)
        (offset := xBase + 4 * 1) (bytes := xBytes 1) (oldReg := oldTmp)
        (value := .s32 (xs 1)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rI, _rRestI, _hcompI, _hequivI, _hi, hrestI⟩
          rcases hrestI with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rPtr, _rRestPtr, _hcompPtr, _hequivPtr, hptr,
            _hrestPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "ptr") (off := xBase + 4 * 1)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regI, regAcc, regPtr, predP, x0, x2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_pred_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem globalSum_load_tmp_wp_at_two
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldTmp : Value)
    (haccess : AccessOk .global .s32 (.global (xBase + 4 * 2)))
    (hwidth : Typing.byteWidth? .s32 = some (xBytes 2).length)
    (hdecode : DecodedScalar .s32 (xBytes 2) (.s32 (xs 2))) :
    globalSumAt lane ("body", 0) (Int.ofNat 2) (globalSumPrefixS32 xs 2) oldTmp
        (.gaddr .global (xBase + 4 * 2)) true xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumLoadTmp
        (globalSumAt lane ("body", 1) (Int.ofNat 2) (globalSumPrefixS32 xs 2)
          (.s32 (xs 2)) (.gaddr .global (xBase + 4 * 2)) true
          xBase outBase xBytes oldOut) := by
  let regI := CSL.reg 0 0 lane "i" (.s32 (Int.ofNat 2))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs 2))
  let regTmpOld := CSL.reg 0 0 lane "tmp" oldTmp
  let regTmpNew := CSL.reg 0 0 lane "tmp" (.s32 (xs 2))
  let regPtr := CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * 2))
  let predP := CSL.pred 0 0 lane "p" true
  let x0 := CSL.globalBytes (xBase + 4 * 0) .read (xBytes 0)
  let x1 := CSL.globalBytes (xBase + 4 * 1) .read (xBytes 1)
  let x2 := CSL.globalBytes (xBase + 4 * 2) .read (xBytes 2)
  let out := CSL.globalBytes outBase .write oldOut
  let frame := CSL.sepList [regI, regAcc, regPtr, predP, x0, x1, out]
  have hpreToRule :
      globalSumAt lane ("body", 0) (Int.ofNat 2) (globalSumPrefixS32 xs 2) oldTmp
          (.gaddr .global (xBase + 4 * 2)) true xBase outBase xBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((x2 ∗ regTmpOld) ∗ frame)) := by
    exact CSL.entails_trans
      (globalSumAt_to_flat lane ("body", 0) (Int.ofNat 2) (globalSumPrefixS32 xs 2)
        oldTmp (.gaddr .global (xBase + 4 * 2)) true xBase outBase xBytes oldOut)
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpOld, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regI, regAcc, regTmpOld, regPtr, predP, x0, x1, x2, out]
              x2 regTmpOld regI [regAcc, regPtr, predP, x0, x1, out]
              (perm_abcdefghi_to_hcabdefgi
                regI regAcc regTmpOld regPtr predP x0 x1 x2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((x2 ∗ regTmpNew) ∗ frame)) ⊢ₛ
        globalSumAt lane ("body", 1) (Int.ofNat 2) (globalSumPrefixS32 xs 2)
          (.s32 (xs 2)) (.gaddr .global (xBase + 4 * 2)) true
          xBase outBase xBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [globalSumFlatResources, regI, regAcc, regTmpNew, regPtr, predP,
          x0, x1, x2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              x2 regTmpNew regI [regAcc, regPtr, predP, x0, x1, out]
              [regI, regAcc, regTmpNew, regPtr, predP, x0, x1, x2, out]
              (perm_abcdefghi_to_hcabdefgi
                regI regAcc regTmpNew regPtr predP x0 x1 x2 out).symm))
      (globalSumAt_flat_to_standard lane ("body", 1) (Int.ofNat 2)
        (globalSumPrefixS32 xs 2) (.s32 (xs 2))
        (.gaddr .global (xBase + 4 * 2)) true xBase outBase xBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((x2 ∗ regTmpOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 globalSumLoadTmp
          (warpAt 0 0 ("body", 1) [lane] ∗ ((x2 ∗ regTmpNew) ∗ frame)) := by
    simpa [globalSumLoadTmp, x2, regTmpOld, regTmpNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "tmp")
        (ty := .s32) (addrExpr := .reg "ptr") (lane := lane)
        (offset := xBase + 4 * 2) (bytes := xBytes 2) (oldReg := oldTmp)
        (value := .s32 (xs 2)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rI, _rRestI, _hcompI, _hequivI, _hi, hrestI⟩
          rcases hrestI with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rPtr, _rRestPtr, _hcompPtr, _hequivPtr, hptr,
            _hrestPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "ptr") (off := xBase + 4 * 2)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regI, regAcc, regPtr, predP, x0, x1, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_reg_load_of_ne (by decide)
          · subst q
            exact stable_pred_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load
          · subst q
            exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem globalSum_load_tmp_wp_at
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldTmp : Value) (i : Nat) (hlt : i < globalSumProofN)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j))) :
    globalSumAt lane ("body", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) oldTmp
        (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumLoadTmp
        (globalSumAt lane ("body", 1) (Int.ofNat i) (globalSumPrefixS32 xs i)
          (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
          xBase outBase xBytes oldOut) := by
  unfold globalSumProofN at hlt
  cases i with
  | zero =>
      simpa [globalSumProofN] using
        globalSum_load_tmp_wp_at_zero lane xBase outBase xs xBytes oldOut oldTmp
          (haccessX 0 (by simp [globalSumProofN]))
          (hwidthX 0 (by simp [globalSumProofN]))
          (hdecodeX 0 (by simp [globalSumProofN]))
  | succ i =>
      cases i with
      | zero =>
          simpa [globalSumProofN] using
            globalSum_load_tmp_wp_at_one lane xBase outBase xs xBytes oldOut oldTmp
              (haccessX 1 (by simp [globalSumProofN]))
              (hwidthX 1 (by simp [globalSumProofN]))
              (hdecodeX 1 (by simp [globalSumProofN]))
      | succ i =>
          cases i with
          | zero =>
              simpa [globalSumProofN] using
                globalSum_load_tmp_wp_at_two lane xBase outBase xs xBytes oldOut oldTmp
                  (haccessX 2 (by simp [globalSumProofN]))
                  (hwidthX 2 (by simp [globalSumProofN]))
                  (hdecodeX 2 (by simp [globalSumProofN]))
          | succ i =>
              omega

theorem globalSum_add_acc_wp_at
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (i : Nat) :
    globalSumAt lane ("body", 1) (Int.ofNat i) (globalSumPrefixS32 xs i)
        (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
        xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumAddAcc
        (globalSumAt lane ("body", 2) (Int.ofNat i) (globalSumPrefixS32 xs (i + 1))
          (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
          xBase outBase xBytes oldOut) := by
  have hfocus :=
    globalSumAt_to_acc_focus lane ("body", 1) (Int.ofNat i) (globalSumPrefixS32 xs i)
      (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
      xBase outBase xBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 1) [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs i)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        wpInstr 0 0 globalSumAddAcc
          (warpAt 0 0 ("body", 2) [lane] ∗
            (CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
                CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
                CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
                CSL.pred 0 0 lane "p" true] ∗
                globalSumFrame xBase outBase xBytes oldOut))) := by
    simpa [globalSumAddAcc] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 1)) (dst := "acc")
        (rhs := .binop .add (.reg "acc") (.reg "tmp")) (lane := lane)
        (old := .s32 (globalSumPrefixS32 xs i))
        (new := .s32 (globalSumPrefixS32 xs (i + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rAcc, _rFrame, _hcompRest, _hequivRest, hacc, hframe⟩
          rcases hframe with ⟨_rRegs, _rMem, _hcompFrame, _hequivFrame, hregs, _hmem⟩
          rcases hregs with ⟨_rI, _rRestRegs, _hcompI, _hequivI, _hi, hrestRegs⟩
          rcases hrestRegs with ⟨_rTmp, _rRestTmp, _hcompTmp, _hequivTmp, htmp,
            _hrestTmp⟩
          simpa [globalSumPrefixS32] using
            EvalRValue.binop_add_s32 (eval_reg_of_assertion hacc)
              (eval_reg_of_assertion htmp))
        (by
          refine CSL.stable_sep ?_ ?_
          · apply CSL.stable_sepList
            intro q hq
            simp at hq
            rcases hq with hq | hq | hq | hq
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_pred_assignReg
          · exact globalSumFrame_stable_assignReg "acc"
              (.binop .add (.reg "acc") (.reg "tmp")) xBase outBase xBytes oldOut))
  have hpost :
      (warpAt 0 0 ("body", 2) [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        globalSumAt lane ("body", 2) (Int.ofNat i) (globalSumPrefixS32 xs (i + 1))
          (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
          xBase outBase xBytes oldOut :=
    globalSumAt_acc_focus_to_standard lane ("body", 2) (Int.ofNat i)
      (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
      (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem globalSum_inc_s32_of_lt_proofN {i : Nat} (h : i < globalSumProofN) :
    Helpers.normalizeSigned 32 (Int.ofNat i + 1) = Int.ofNat (i + 1) := by
  unfold globalSumProofN at h
  cases i with
  | zero =>
      simp [Helpers.normalizeSigned]
  | succ i =>
      cases i with
      | zero =>
          simp [Helpers.normalizeSigned]
      | succ i =>
          cases i with
          | zero =>
              simp [Helpers.normalizeSigned]
          | succ i =>
              omega

theorem globalSum_inc_i_wp_at
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (i : Nat) (hlt : i < globalSumProofN) :
    globalSumAt lane ("body", 2) (Int.ofNat i) (globalSumPrefixS32 xs (i + 1))
        (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
        xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumIncI
        (globalSumAt lane ("body", 3) (Int.ofNat (i + 1))
          (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut) := by
  have hinc := globalSum_inc_s32_of_lt_proofN hlt
  have hincValue :
      Helpers.normalizeSigned 32 (Int.ofNat i + 1) = Int.ofNat i + 1 := by
    rw [hinc]
    simp
  have hfocus :=
    globalSumAt_to_i_focus lane ("body", 2) (Int.ofNat i)
      (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
      (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 2) [lane] ∗
        (CSL.reg 0 0 lane "i" (.s32 (Int.ofNat i)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        wpInstr 0 0 globalSumIncI
          (warpAt 0 0 ("body", 3) [lane] ∗
            (CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
                CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
                CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
                CSL.pred 0 0 lane "p" true] ∗
                globalSumFrame xBase outBase xBytes oldOut))) := by
    simpa [globalSumIncI] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 2)) (dst := "i")
        (rhs := .binop .add (.reg "i") (.imm (.s32 1))) (lane := lane)
        (old := .s32 (Int.ofNat i)) (new := .s32 (Int.ofNat (i + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rI, _rFrame, _hcompRest, _hequivRest, hi, _hframe⟩
          have hevalNorm :
              EvalRValue st { cta := 0, warp := 0, lane := lane }
                (.binop .add (.reg "i") (.imm (.s32 1)))
                (.s32 (Helpers.normalizeSigned 32 (Int.ofNat i + 1))) :=
            EvalRValue.binop_add_s32
              (eval_reg_of_assertion
                (ctx := { cta := 0, warp := 0, lane := lane }) hi)
              (eval_imm
                (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                (value := .s32 (1 : Int)))
          rw [hincValue] at hevalNorm
          exact hevalNorm)
        (by
          refine CSL.stable_sep ?_ ?_
          · apply CSL.stable_sepList
            intro q hq
            simp at hq
            rcases hq with hq | hq | hq | hq
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_pred_assignReg
          · exact globalSumFrame_stable_assignReg "i"
              (.binop .add (.reg "i") (.imm (.s32 1))) xBase outBase xBytes oldOut))
  have hpost :
      (warpAt 0 0 ("body", 3) [lane] ∗
        (CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        globalSumAt lane ("body", 3) (Int.ofNat (i + 1))
          (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
          (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut :=
    globalSumAt_i_focus_to_standard lane ("body", 3) (Int.ofNat (i + 1))
      (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
      (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem globalSum_ptr_advance (xBase i : Nat) :
    xBase + 4 * i + (4 : UInt64).toNat = xBase + 4 * (i + 1) := by
  simp
  omega

theorem globalSum_inc_ptr_wp_at
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (i : Nat) :
    globalSumAt lane ("body", 3) (Int.ofNat (i + 1)) (globalSumPrefixS32 xs (i + 1))
        (.s32 (xs i)) (.gaddr .global (xBase + 4 * i)) true
        xBase outBase xBytes oldOut ⊢ₛ
      wpInstr 0 0 globalSumIncPtr
        (globalSumAt lane ("body", 4) (Int.ofNat (i + 1))
          (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
          (.gaddr .global (xBase + 4 * (i + 1))) true xBase outBase xBytes oldOut) := by
  have hfocus :=
    globalSumAt_to_ptr_focus lane ("body", 3) (Int.ofNat (i + 1))
      (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
      (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 3) [lane] ∗
        (CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * i)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        wpInstr 0 0 globalSumIncPtr
          (warpAt 0 0 ("body", 4) [lane] ∗
            (CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * (i + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))),
                CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
                CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
                CSL.pred 0 0 lane "p" true] ∗
                globalSumFrame xBase outBase xBytes oldOut))) := by
    simpa [globalSumIncPtr] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 3)) (dst := "ptr")
        (rhs := .binop .add (.reg "ptr") (.imm (.u64 (4 : UInt64)))) (lane := lane)
        (old := .gaddr .global (xBase + 4 * i))
        (new := .gaddr .global (xBase + 4 * (i + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPtr, _rFrame, _hcompRest, _hequivRest, hptr, _hframe⟩
          have hevalNorm :
              EvalRValue st { cta := 0, warp := 0, lane := lane }
                (.binop .add (.reg "ptr") (.imm (.u64 (4 : UInt64))))
                (.gaddr .global ((xBase + 4 * i) + (4 : UInt64).toNat)) :=
            EvalRValue.binop_add_gaddr_u64
              (eval_reg_of_assertion
                (ctx := { cta := 0, warp := 0, lane := lane }) hptr)
              (eval_imm
                (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                (value := .u64 (4 : UInt64)))
          rw [globalSum_ptr_advance] at hevalNorm
          exact hevalNorm)
        (by
          refine CSL.stable_sep ?_ ?_
          · apply CSL.stable_sepList
            intro q hq
            simp at hq
            rcases hq with hq | hq | hq | hq
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_reg_assignReg_of_ne (by decide)
            · subst q
              exact stable_pred_assignReg
          · exact globalSumFrame_stable_assignReg "ptr"
              (.binop .add (.reg "ptr") (.imm (.u64 (4 : UInt64))))
              xBase outBase xBytes oldOut))
  have hpost :
      (warpAt 0 0 ("body", 4) [lane] ∗
        (CSL.reg 0 0 lane "ptr" (.gaddr .global (xBase + 4 * (i + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "i" (.s32 (Int.ofNat (i + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (globalSumPrefixS32 xs (i + 1))),
            CSL.reg 0 0 lane "tmp" (.s32 (xs i)),
            CSL.pred 0 0 lane "p" true] ∗
            globalSumFrame xBase outBase xBytes oldOut))) ⊢ₛ
        globalSumAt lane ("body", 4) (Int.ofNat (i + 1))
          (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
          (.gaddr .global (xBase + 4 * (i + 1))) true xBase outBase xBytes oldOut :=
    globalSumAt_ptr_focus_to_standard lane ("body", 4) (Int.ofNat (i + 1))
      (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
      (.gaddr .global (xBase + 4 * (i + 1))) true xBase outBase xBytes oldOut
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem globalSum_body_instrs_wp_at
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte) (oldTmp : Value) (i : Nat) (hlt : i < globalSumProofN)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j))) :
    globalSumAt lane ("body", 0) (Int.ofNat i) (globalSumPrefixS32 xs i) oldTmp
        (.gaddr .global (xBase + 4 * i)) true xBase outBase xBytes oldOut ⊢ₛ
      wpInstrs 0 0 globalSumBodyBlock.body
        (globalSumBodyTermPre lane xBase outBase xs xBytes oldOut) := by
  have hload :=
    globalSum_load_tmp_wp_at lane xBase outBase xs xBytes oldOut oldTmp i hlt
      haccessX hwidthX hdecodeX
  have hadd := globalSum_add_acc_wp_at lane xBase outBase xs xBytes oldOut i
  have hincI := globalSum_inc_i_wp_at lane xBase outBase xs xBytes oldOut i hlt
  have hincPtr := globalSum_inc_ptr_wp_at lane xBase outBase xs xBytes oldOut i
  have hpost :
      globalSumAt lane ("body", 4) (Int.ofNat (i + 1))
          (globalSumPrefixS32 xs (i + 1)) (.s32 (xs i))
          (.gaddr .global (xBase + 4 * (i + 1))) true xBase outBase xBytes oldOut ⊢ₛ
        globalSumBodyTermPre lane xBase outBase xs xBytes oldOut := by
    intro st r hat
    exact ⟨i + 1, .s32 (xs i), Nat.succ_le_of_lt hlt, hat⟩
  simpa [globalSumBodyBlock, wpInstrs, wpInstrList, globalSumLoadTmp,
    globalSumAddAcc, globalSumIncI, globalSumIncPtr] using
    CSL.entails_trans hload
      (wpInstr_mono <|
        CSL.entails_trans hadd
          (wpInstr_mono <|
            CSL.entails_trans hincI
              (wpInstr_mono <|
                CSL.entails_trans hincPtr (wpInstr_mono hpost))))

theorem globalSum_body_instrs_wp
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut : List Byte)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j))) :
    globalSumBodyInv lane xBase outBase xs xBytes oldOut ⊢ₛ
      wpInstrs 0 0 globalSumBodyBlock.body
        (globalSumBodyTermPre lane xBase outBase xs xBytes oldOut) := by
  intro st r hpre
  rcases hpre with ⟨i, oldTmp, hlt, hat⟩
  exact globalSum_body_instrs_wp_at lane xBase outBase xs xBytes oldOut oldTmp i hlt
    haccessX hwidthX hdecodeX st r hat

theorem globalSum_entry_block_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    blockVC' 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut)
      "entry" (globalSumEntryBlock xBase) := by
  refine ⟨globalSumEntryTermPre lane xBase outBase xBytes oldOut oldTmp oldP, ?_, ?_⟩
  · simpa [globalSumInvariants, globalSumEntryBlock, globalSumProofN] using
      globalSum_entry_instrs_wp lane xBase outBase xBytes oldOut
        oldI oldAcc oldTmp oldPtr oldP
  · exact globalSum_entry_term_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP

theorem globalSum_header_block_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool) :
    blockVC' 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut)
      "loop" (globalSumHeaderBlock globalSumProofN) := by
  refine ⟨globalSumHeaderTermPre lane xBase outBase xs xBytes oldOut, ?_, ?_⟩
  · simpa [globalSumInvariants, globalSumHeaderBlock, globalSumProofN] using
      globalSum_header_instrs_wp lane xBase outBase xs xBytes oldOut
  · exact globalSum_header_term_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP

theorem globalSum_body_block_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j))) :
    blockVC' 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut)
      "body" globalSumBodyBlock := by
  refine ⟨globalSumBodyTermPre lane xBase outBase xs xBytes oldOut, ?_, ?_⟩
  · simpa [globalSumInvariants, globalSumBodyBlock, globalSumProofN] using
      globalSum_body_instrs_wp lane xBase outBase xs xBytes oldOut
        haccessX hwidthX hdecodeX
  · exact globalSum_body_term_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP

theorem globalSum_exit_block_vc
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool)
    (haccessOut : AccessOk .global .s32 (.global outBase))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (globalSumPrefixS32 xs globalSumProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointXOut :
      ∀ i, i < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * i) (xBytes i).length outBase newOut.length) :
    blockVC' 0 0
      (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP)
      (globalSumPost lane xBase outBase xs xBytes newOut)
      "exit" (globalSumExitBlock outBase) := by
  refine ⟨globalSumExitTermPre lane xBase outBase xs xBytes newOut, ?_, ?_⟩
  · simpa [globalSumInvariants, globalSumExitBlock, globalSumProofN] using
      globalSum_exit_instrs_wp lane xBase outBase xs xBytes oldOut newOut
        haccessOut hencodeOut hlenOut hdisjointXOut
  · exact globalSum_exit_term_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP

theorem globalSum_block_vcs
    (lane : LaneId) (xBase outBase : Nat) (xs : Nat → Int) (xBytes : Nat → List Byte)
    (oldOut newOut : List Byte) (oldI oldAcc oldTmp oldPtr : Value) (oldP : Bool)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j)))
    (haccessOut : AccessOk .global .s32 (.global outBase))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (globalSumPrefixS32 xs globalSumProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointXOut :
      ∀ j, j < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * j) (xBytes j).length outBase newOut.length) :
    ∀ label block,
      (globalSumEnv globalSumProofN xBase outBase).blocks[label]? = some block →
        blockVC' 0 0
          (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
            oldI oldAcc oldTmp oldPtr oldP)
          (globalSumPost lane xBase outBase xs xBytes newOut) label block := by
  intro label block hlookup
  rcases globalSum_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact globalSum_entry_block_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP
  · rcases hloop with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact globalSum_header_block_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP
  · rcases hbody with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact globalSum_body_block_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP haccessX hwidthX hdecodeX
  · rcases hexit with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact globalSum_exit_block_vc lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP haccessOut hencodeOut hlenOut hdisjointXOut

theorem global_sum_loop_kernel_valid
    {init : State} {resource : CSL.Resource} {lane : LaneId}
    {xBase outBase : Nat} {xs : Nat → Int} {xBytes : Nat → List Byte}
    {oldOut newOut : List Byte} {oldI oldAcc oldTmp oldPtr : Value} {oldP : Bool}
    (hinit :
      (globalSumKernelSpec init resource lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP).pre init resource)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j)))
    (haccessOut : AccessOk .global .s32 (.global outBase))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (globalSumPrefixS32 xs globalSumProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointXOut :
      ∀ j, j < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * j) (xBytes j).length outBase newOut.length)
    (honly :
      cfgKernelInvariant' (globalSumEnv globalSumProofN xBase outBase) 0 0
        (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
          oldI oldAcc oldTmp oldPtr oldP)
        (globalSumPost lane xBase outBase xs xBytes newOut) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (globalSumEnv globalSumProofN xBase outBase) 0 0)
    (hpostNoStep :
      NoStepBlock 0 0 (globalSumPost lane xBase outBase xs xBytes newOut))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (globalSumEnv globalSumProofN xBase outBase) 0 0
          (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
            oldI oldAcc oldTmp oldPtr oldP)
          (globalSumPost lane xBase outBase xs xBytes newOut))) :
    (globalSumKernelSpec init resource lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP).Valid :=
  KernelSpec.Valid.of_entry_blockVCs'_closed
    (spec := globalSumKernelSpec init resource lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP)
    (env := globalSumEnv globalSumProofN xBase outBase) (cta := 0) (warp := 0)
    (invariants := globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
      oldI oldAcc oldTmp oldPtr oldP)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r hpre
      exact hpre.2)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨henv, hentry⟩
      rcases hentry with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
      rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
      exact ⟨warpState, globalSumEntryBlock xBase, henv, hwarp, hlock,
        by simpa [globalSumEnv, globalSumProofN] using hrpc,
        by simpa [globalSumEnv, globalSumProofN] using
          globalSum_entry_lookup globalSumProofN xBase outBase⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics
      (globalSum_body_ordinary globalSumProofN xBase outBase))
    (hblocks :=
      globalSum_block_vcs lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP haccessX hwidthX hdecodeX
        haccessOut hencodeOut hlenOut hdisjointXOut)
    (hbr := hbr)
    (htargets := globalSum_targets_exist globalSumProofN xBase outBase)
    (hpostNoStep := hpostNoStep)
    (hsuffixNoFinal := hsuffixNoFinal)

theorem global_sum_loop_partial_correct
    {init final : State} {resource : CSL.Resource} {lane : LaneId}
    {xBase outBase : Nat} {xs : Nat → Int} {xBytes : Nat → List Byte}
    {oldOut newOut : List Byte} {oldI oldAcc oldTmp oldPtr : Value} {oldP : Bool}
    (hinit :
      (globalSumKernelSpec init resource lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP).pre init resource)
    (haccessX :
      ∀ j, j < globalSumProofN → AccessOk .global .s32 (.global (xBase + 4 * j)))
    (hwidthX :
      ∀ j, j < globalSumProofN → Typing.byteWidth? .s32 = some (xBytes j).length)
    (hdecodeX :
      ∀ j, j < globalSumProofN → DecodedScalar .s32 (xBytes j) (.s32 (xs j)))
    (haccessOut : AccessOk .global .s32 (.global outBase))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (globalSumPrefixS32 xs globalSumProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointXOut :
      ∀ j, j < globalSumProofN →
        ByteRangesDisjoint (xBase + 4 * j) (xBytes j).length outBase newOut.length)
    (honly :
      cfgKernelInvariant' (globalSumEnv globalSumProofN xBase outBase) 0 0
        (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
          oldI oldAcc oldTmp oldPtr oldP)
        (globalSumPost lane xBase outBase xs xBytes newOut) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (globalSumEnv globalSumProofN xBase outBase) 0 0)
    (hpostNoStep :
      NoStepBlock 0 0 (globalSumPost lane xBase outBase xs xBytes newOut))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (globalSumEnv globalSumProofN xBase outBase) 0 0
          (globalSumInvariants lane xBase outBase xs xBytes oldOut newOut
            oldI oldAcc oldTmp oldPtr oldP)
          (globalSumPost lane xBase outBase xs xBytes newOut)))
    (hterm : TerminatesAt init final) :
    ∃ r, globalSumPost lane xBase outBase xs xBytes newOut final r := by
  have hvalid :
      (globalSumKernelSpec init resource lane xBase outBase xs xBytes oldOut newOut
        oldI oldAcc oldTmp oldPtr oldP).Valid :=
    global_sum_loop_kernel_valid hinit haccessX hwidthX hdecodeX
      haccessOut hencodeOut hlenOut hdisjointXOut
      honly hbr hpostNoStep hsuffixNoFinal
  exact KernelSpec.partial_correct hvalid final hterm

end Examples
end CLean
