import Mathlib.Order.CompleteLattice.Basic
import CLean.WP.Instr

/-! # Phase 0 spike — does our CSL WP fit a Loom-style ordered monad algebra?

THROWAWAY de-risking file. It confirms three things, in increasing order of risk:

1. `CSL.Assertion = State → Resource → Prop` is a `CompleteLattice`
   (the `l` slot Loom's `MAlgOrdered` demands), with `≤` = CSL entailment.

2. A free PTX effect monad `PtxBase` (over getState / instr / terminator /
   warpStep) carries an `MAlgOrdered PtxBase Assertion` instance — i.e. the
   generalized-Dijkstra `μ : m l → l` shape really exists for our resource-aware
   WP, with `μ (pure)` and `μ (bind)` monotonicity.

3. THE BRIDGE: the `wp` derived from that algebra for each primitive is
   *definitionally equal* to the existing hand-written `wpInstr` /
   `wpTerminator` / `wpExecutableStep`. If these are `rfl`, the whole
   "CLean-on-a-Loom-core" plan is mechanically sound and nothing in the
   current WP layer has to be thrown away to get there.

The `MAlgOrdered` class is reproduced (not imported) from Loom
`.loom/Loom/MonadAlgebras/Defs.lean:74` so the spike has no cross-package build
dependency; the field shapes match Loom's `μ_ord_pure` / `μ_ord_bind` exactly
(`μ ∘ f ≤ μ ∘ g` is stated in its unfolded pointwise form). -/

open CLean
open CLean.CSL
open CLean.WP

namespace Spike

/-! ## 1. The lattice slot -/

-- `Assertion` is already a complete lattice, pointwise via `Prop`. No new work.
example : CompleteLattice Assertion := inferInstance

-- and its order is exactly CSL entailment (definitionally)
theorem le_iff (P Q : Assertion) : (P ≤ Q) ↔ ∀ st r, P st r → Q st r := Iff.rfl

/-! ## Loom-shaped ordered monad algebra (mirrors Defs.lean:74) -/

class MAlgOrdered (m : Type → Type 1) (l : Type) [Monad m] [CompleteLattice l] where
  μ : m l → l
  μ_pure : ∀ a : l, μ (pure a) = a
  μ_bind : ∀ {α : Type} (f g : α → m l),
    (∀ a, μ (f a) ≤ μ (g a)) → ∀ x : m α, μ (x >>= f) ≤ μ (x >>= g)

/-! ## 2. The PTX effect monad -/

inductive PtxOp : Type → Type where
  | getState   : PtxOp State
  | instr      (cta : CTAId) (warp : WarpId) (gi : GInstr)    : PtxOp Unit
  | terminator (cta : CTAId) (warp : WarpId) (t : Terminator) : PtxOp Unit
  | warpStep   (cta : CTAId) (warp : WarpId)                  : PtxOp Unit

inductive PtxBase (α : Type) : Type 1 where
  | pure : α → PtxBase α
  | vis  {β : Type} : PtxOp β → (β → PtxBase α) → PtxBase α

def PtxBase.bind {α β} : PtxBase α → (α → PtxBase β) → PtxBase β
  | .pure a,   f => f a
  | .vis op k, f => .vis op (fun x => (k x).bind f)

instance : Monad PtxBase where
  pure := PtxBase.pure
  bind := PtxBase.bind

/-! ## 3. The custom CSL-aware WP for primitives

This is the only place the resource-update modality `∃ r', Update r r' ∧ …`
lives. Everything else is generic monadic plumbing. -/

def wpOp {α} : PtxOp α → (α → Assertion) → Assertion
  | .getState,              Q => fun st r => Q st st r
  | .instr cta warp gi,     Q => fun st r =>
      ∀ st', Helpers.stepInstr? st cta warp gi = some st' →
        ∃ r', Resource.Update r r' ∧ Q () st' r'
  | .terminator cta warp t, Q => fun st r =>
      ∀ st', Helpers.stepTerminator? st cta warp t = some st' →
        ∃ r', Resource.Update r r' ∧ Q () st' r'
  | .warpStep cta warp,     Q => fun st r =>
      ∀ st', StepMachine.stepAt? st cta warp = some st' →
        ∃ r', Resource.Update r r' ∧ Q () st' r'

def PtxBase.wp {α} : PtxBase α → (α → Assertion) → Assertion
  | .pure a,   Q => Q a
  | .vis op k, Q => wpOp op (fun x => (k x).wp Q)

def mu (c : PtxBase Assertion) : Assertion := c.wp id

/-! ### structural lemmas -/

theorem wpOp_mono {α} (op : PtxOp α) {Q Q' : α → Assertion}
    (h : ∀ a st r, Q a st r → Q' a st r) :
    ∀ st r, wpOp op Q st r → wpOp op Q' st r := by
  cases op with
  | getState => intro st r hq; exact h st st r hq
  | instr cta warp gi =>
      intro st r hq st' hs
      obtain ⟨r', hu, hpost⟩ := hq st' hs; exact ⟨r', hu, h () st' r' hpost⟩
  | terminator cta warp t =>
      intro st r hq st' hs
      obtain ⟨r', hu, hpost⟩ := hq st' hs; exact ⟨r', hu, h () st' r' hpost⟩
  | warpStep cta warp =>
      intro st r hq st' hs
      obtain ⟨r', hu, hpost⟩ := hq st' hs; exact ⟨r', hu, h () st' r' hpost⟩

theorem wp_mono {α} (x : PtxBase α) {Q Q' : α → Assertion}
    (h : ∀ a st r, Q a st r → Q' a st r) :
    ∀ st r, x.wp Q st r → x.wp Q' st r := by
  induction x with
  | pure a => intro st r hq; exact h a st r hq
  | vis op k ih => intro st r hq; exact wpOp_mono op (fun b => ih b) st r hq

theorem wp_bind {α β} (x : PtxBase α) (f : α → PtxBase β) (Q : β → Assertion) :
    (x.bind f).wp Q = x.wp (fun a => (f a).wp Q) := by
  induction x with
  | pure a => rfl
  | vis op k ih =>
      show wpOp op (fun x => ((k x).bind f).wp Q)
            = wpOp op (fun x => (k x).wp (fun a => (f a).wp Q))
      congr 1; funext b; exact ih b

/-! ## The instance — `MAlgOrdered PtxBase Assertion` -/

instance : MAlgOrdered PtxBase Assertion where
  μ := mu
  μ_pure := fun _ => rfl
  μ_bind := by
    intro α f g h x
    have e1 : mu (x >>= f) = x.wp (fun a => mu (f a)) := wp_bind x f id
    have e2 : mu (x >>= g) = x.wp (fun a => mu (g a)) := wp_bind x g id
    rw [e1, e2]
    refine (le_iff _ _).mpr ?_
    intro st r hq
    refine wp_mono x ?_ st r hq
    intro a st' r' hfa
    exact (le_iff _ _).mp (h a) st' r' hfa

/-! ## 3. THE BRIDGE — derived `wp` = existing hand-written WP, definitionally -/

def execInstr (cta : CTAId) (warp : WarpId) (gi : GInstr) : PtxBase Unit :=
  .vis (.instr cta warp gi) PtxBase.pure

def execTerminator (cta : CTAId) (warp : WarpId) (t : Terminator) : PtxBase Unit :=
  .vis (.terminator cta warp t) PtxBase.pure

def execStep (cta : CTAId) (warp : WarpId) : PtxBase Unit :=
  .vis (.warpStep cta warp) PtxBase.pure

theorem bridge_instr (cta : CTAId) (warp : WarpId) (gi : GInstr) (post : Assertion) :
    (execInstr cta warp gi).wp (fun _ => post) = wpInstr cta warp gi post := rfl

theorem bridge_terminator
    (cta : CTAId) (warp : WarpId) (t : Terminator) (post : Assertion) :
    (execTerminator cta warp t).wp (fun _ => post) = wpTerminator cta warp t post := rfl

theorem bridge_step (cta : CTAId) (warp : WarpId) (post : Assertion) :
    (execStep cta warp).wp (fun _ => post) = wpExecutableStep cta warp post := rfl

/-! ## bonus: monadic sequencing already gives the Dijkstra bind law for free,
so a straight-line block's WP is the fold of the per-instruction WPs. -/

example (cta : CTAId) (warp : WarpId) (gi₁ gi₂ : GInstr) (post : Assertion) :
    (execInstr cta warp gi₁ >>= fun _ => execInstr cta warp gi₂).wp (fun _ => post)
      = wpInstr cta warp gi₁ (wpInstr cta warp gi₂ post) := by
  show (PtxBase.bind (execInstr cta warp gi₁)
          (fun _ => execInstr cta warp gi₂)).wp (fun _ => post) = _
  rw [wp_bind]; rfl

end Spike
