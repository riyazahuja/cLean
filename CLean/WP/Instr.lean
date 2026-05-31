import CLean.Semantics.Execution
import CLean.WP.Expr

namespace CLean
namespace WP

abbrev Post := CSL.Assertion
abbrev Pre := CSL.Assertion

def InstrStep (cta : CTAId) (warp : WarpId) (gi : GInstr) (st st' : State) : Prop :=
  Helpers.stepInstr? st cta warp gi = some st'

def TerminatorStep (cta : CTAId) (warp : WarpId) (term : Terminator)
    (st st' : State) : Prop :=
  Helpers.stepTerminator? st cta warp term = some st'

def wpInstr (cta : CTAId) (warp : WarpId) (gi : GInstr) (post : Post) : Pre :=
  fun st r =>
    ∀ st', Helpers.stepInstr? st cta warp gi = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def wpTerminator (cta : CTAId) (warp : WarpId) (term : Terminator) (post : Post) : Pre :=
  fun st r =>
    ∀ st', Helpers.stepTerminator? st cta warp term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def wpExecutableStep (cta : CTAId) (warp : WarpId) (post : Post) : Pre :=
  fun st r =>
    ∀ st', StepMachine.stepAt? st cta warp = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

theorem wpInstr_mono
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpInstr cta warp gi post ⊢ₛ wpInstr cta warp gi post' := by
  intro st r hwp st' hstep
  rcases hwp st' hstep with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, hpost st' r' hpost'⟩

theorem wpTerminator_mono
    {cta : CTAId} {warp : WarpId} {term : Terminator} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpTerminator cta warp term post ⊢ₛ wpTerminator cta warp term post' := by
  intro st r hwp st' hstep
  rcases hwp st' hstep with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, hpost st' r' hpost'⟩

theorem wpExecutableStep_mono
    {cta : CTAId} {warp : WarpId} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpExecutableStep cta warp post ⊢ₛ wpExecutableStep cta warp post' := by
  intro st r hwp st' hstep
  rcases hwp st' hstep with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, hpost st' r' hpost'⟩

theorem wpInstr_frame
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post frame : Post}
    (hframe : CSL.StableUnder (InstrStep cta warp gi) frame) :
    (wpInstr cta warp gi post ∗ frame) ⊢ₛ wpInstr cta warp gi (post ∗ frame) := by
  intro st r hsep st' hstep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hwp, hframeSt⟩
  rcases hwp st' hstep with ⟨r₁', hupdate, hpost⟩
  refine ⟨CSL.Resource.compose r₁' r₂, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose_right hupdate)
  · exact ⟨r₁', r₂, CSL.Resource.canCompose_update_left hupdate hcomp,
      CSL.Resource.equiv_refl _, hpost, hframe st st' r₂ hstep hframeSt⟩

theorem wpInstr_frame_of_entails
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {pre post frame : Post}
    (hrule : pre ⊢ₛ wpInstr cta warp gi post)
    (hframe : CSL.StableUnder (InstrStep cta warp gi) frame) :
    (pre ∗ frame) ⊢ₛ wpInstr cta warp gi (post ∗ frame) :=
  CSL.entails_trans
    (CSL.sep_mono hrule (CSL.entails_refl frame))
    (wpInstr_frame hframe)

theorem wpTerminator_frame
    {cta : CTAId} {warp : WarpId} {term : Terminator} {post frame : Post}
    (hframe : CSL.StableUnder (TerminatorStep cta warp term) frame) :
    (wpTerminator cta warp term post ∗ frame) ⊢ₛ
      wpTerminator cta warp term (post ∗ frame) := by
  intro st r hsep st' hstep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hwp, hframeSt⟩
  rcases hwp st' hstep with ⟨r₁', hupdate, hpost⟩
  refine ⟨CSL.Resource.compose r₁' r₂, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose_right hupdate)
  · exact ⟨r₁', r₂, CSL.Resource.canCompose_update_left hupdate hcomp,
      CSL.Resource.equiv_refl _, hpost, hframe st st' r₂ hstep hframeSt⟩

theorem wpTerminator_frame_of_entails
    {cta : CTAId} {warp : WarpId} {term : Terminator} {pre post frame : Post}
    (hrule : pre ⊢ₛ wpTerminator cta warp term post)
    (hframe : CSL.StableUnder (TerminatorStep cta warp term) frame) :
    (pre ∗ frame) ⊢ₛ wpTerminator cta warp term (post ∗ frame) :=
  CSL.entails_trans
    (CSL.sep_mono hrule (CSL.entails_refl frame))
    (wpTerminator_frame hframe)

theorem wpInstr_intro
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post : Post}
    {st : State} {r : CSL.Resource}
    (h : ∀ st', Helpers.stepInstr? st cta warp gi = some st' → post st' r) :
    wpInstr cta warp gi post st r :=
  fun st' hstep => ⟨r, CSL.Resource.update_refl r, h st' hstep⟩

theorem wpInstr_sound
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpInstr cta warp gi post st r)
    (hstep : StepInstr st cta warp gi st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' := by
  cases hstep with
  | mk _ _ _ _ _ hrun =>
      exact hwp st' hrun

theorem wpTerminator_sound
    {cta : CTAId} {warp : WarpId} {term : Terminator} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpTerminator cta warp term post st r)
    (hstep : Helpers.stepTerminator? st cta warp term = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  hwp st' hstep

theorem wpExecutableStep_sound
    {cta : CTAId} {warp : WarpId} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpExecutableStep cta warp post st r)
    (hstep : StepMachine.stepAt? st cta warp = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  hwp st' hstep

theorem wpInstr_of_computed
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post : Post}
    {st : State} {r r' : CSL.Resource} {st' : State}
    (hstep : Helpers.stepInstr? st cta warp gi = some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp gi post st r := by
  intro st'' hstep''
  rw [hstep] at hstep''
  injection hstep'' with h
  subst st''
  exact ⟨r', hupdate, hpost⟩

theorem wpTerminator_of_computed
    {cta : CTAId} {warp : WarpId} {term : Terminator} {post : Post}
    {st : State} {r r' : CSL.Resource} {st' : State}
    (hstep : Helpers.stepTerminator? st cta warp term = some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpTerminator cta warp term post st r := by
  intro st'' hstep''
  rw [hstep] at hstep''
  injection hstep'' with h
  subst st''
  exact ⟨r', hupdate, hpost⟩

theorem wp_assignReg_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .assignReg dst rhs } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_assignPred_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .assignPred dst cmp } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_load_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .load dst src } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .load dst src } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_store_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : TypedAddr}
    {value : RValue} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .store dst value } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .store dst value } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_cvta_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .cvta dst space src } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_isspacep_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .isspacep dst space src } post st r :=
  wpInstr_of_computed hstep hupdate hpost

theorem wp_barrierCTA_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {barrierId : Nat}
    {post : Post} {st st' : State} {r r' : CSL.Resource}
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } =
        some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr cta warp { guard? := guard?, instr := .barrierCTA barrierId } post st r :=
  wpInstr_of_computed hstep hupdate hpost

end WP
end CLean
