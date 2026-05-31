import CLean.WP.Instr

namespace CLean
namespace WP

def wpBlock (cta : CTAId) (warp : WarpId) (post : Post) : Pre :=
  fun st r =>
    ∀ st', StepBlock st cta warp st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def wpInstrList (cta : CTAId) (warp : WarpId) : List GInstr → Post → Pre
  | [], post => post
  | gi :: rest, post => wpInstr cta warp gi (wpInstrList cta warp rest post)

def wpInstrs (cta : CTAId) (warp : WarpId) (body : Array GInstr) (post : Post) : Pre :=
  wpInstrList cta warp body.toList post

def wpConcreteBlock (cta : CTAId) (warp : WarpId) (block : Block) (post : Post) : Pre :=
  wpInstrs cta warp block.body (wpTerminator cta warp block.term post)

theorem wpBlock_sound
    {cta : CTAId} {warp : WarpId} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpBlock cta warp post st r)
    (hstep : StepBlock st cta warp st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  hwp st' hstep

theorem wpInstrs_nil (cta : CTAId) (warp : WarpId) (post : Post) :
    wpInstrs cta warp #[] post = post := by
  rfl

theorem wpInstrList_cons (cta : CTAId) (warp : WarpId) (gi : GInstr)
    (rest : List GInstr) (post : Post) :
    wpInstrList cta warp (gi :: rest) post =
      wpInstr cta warp gi (wpInstrList cta warp rest post) := by
  rfl

theorem wpInstrList_mono
    {cta : CTAId} {warp : WarpId} {body : List GInstr} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpInstrList cta warp body post ⊢ₛ wpInstrList cta warp body post' := by
  induction body with
  | nil =>
      exact hpost
  | cons gi rest ih =>
      exact wpInstr_mono ih

theorem wpInstrs_mono
    {cta : CTAId} {warp : WarpId} {body : Array GInstr} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpInstrs cta warp body post ⊢ₛ wpInstrs cta warp body post' :=
  wpInstrList_mono hpost

theorem wpInstrList_frame
    {cta : CTAId} {warp : WarpId} {body : List GInstr} {post frame : Post}
    (hframe : ∀ gi, gi ∈ body → CSL.StableUnder (InstrStep cta warp gi) frame) :
    (wpInstrList cta warp body post ∗ frame) ⊢ₛ
      wpInstrList cta warp body (post ∗ frame) := by
  induction body with
  | nil =>
      exact CSL.entails_refl (post ∗ frame)
  | cons gi rest ih =>
      have hgi : CSL.StableUnder (InstrStep cta warp gi) frame :=
        hframe gi (by simp)
      have hrest :
          ∀ gi', gi' ∈ rest → CSL.StableUnder (InstrStep cta warp gi') frame := by
        intro gi' hmem
        exact hframe gi' (by simp [hmem])
      exact CSL.entails_trans (wpInstr_frame hgi) (wpInstr_mono (ih hrest))

theorem wpInstrList_frame_of_entails
    {cta : CTAId} {warp : WarpId} {body : List GInstr} {pre post frame : Post}
    (hrule : pre ⊢ₛ wpInstrList cta warp body post)
    (hframe : ∀ gi, gi ∈ body → CSL.StableUnder (InstrStep cta warp gi) frame) :
    (pre ∗ frame) ⊢ₛ wpInstrList cta warp body (post ∗ frame) :=
  CSL.entails_trans
    (CSL.sep_mono hrule (CSL.entails_refl frame))
    (wpInstrList_frame hframe)

theorem wpInstrs_frame
    {cta : CTAId} {warp : WarpId} {body : Array GInstr} {post frame : Post}
    (hframe : ∀ gi, gi ∈ body.toList → CSL.StableUnder (InstrStep cta warp gi) frame) :
    (wpInstrs cta warp body post ∗ frame) ⊢ₛ
      wpInstrs cta warp body (post ∗ frame) := by
  simpa [wpInstrs] using
    (wpInstrList_frame (cta := cta) (warp := warp) (body := body.toList)
      (post := post) (frame := frame) hframe)

theorem wpConcreteBlock_frame
    {cta : CTAId} {warp : WarpId} {block : Block} {post frame : Post}
    (hbody :
      ∀ gi, gi ∈ block.body.toList → CSL.StableUnder (InstrStep cta warp gi) frame)
    (hterm : CSL.StableUnder (TerminatorStep cta warp block.term) frame) :
    (wpConcreteBlock cta warp block post ∗ frame) ⊢ₛ
      wpConcreteBlock cta warp block (post ∗ frame) := by
  unfold wpConcreteBlock wpInstrs
  exact CSL.entails_trans
    (wpInstrList_frame hbody)
    (wpInstrList_mono (wpTerminator_frame hterm))

end WP
end CLean
