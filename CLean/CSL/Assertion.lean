import CLean.CSL.Resource

namespace CLean
namespace CSL

abbrev Assertion := State → Resource → Prop

def emp : Assertion :=
  fun _ r => r = Resource.empty

def pure (p : Prop) : Assertion :=
  fun st r => p ∧ emp st r

def entails (p q : Assertion) : Prop :=
  ∀ st r, p st r → q st r

def or (p q : Assertion) : Assertion :=
  fun st r => p st r ∨ q st r

def resourceUpdate (p q : Assertion) : Prop :=
  ∀ st r, p st r → ∃ r', Resource.Update r r' ∧ q st r'

def sep (p q : Assertion) : Assertion :=
  fun st r =>
    ∃ r₁ r₂,
      Resource.canCompose r₁ r₂ ∧
      Resource.Equiv r (Resource.compose r₁ r₂) ∧
      p st r₁ ∧
      q st r₂

infixr:55 " ∗ " => sep
infixr:50 " ∨ₛ " => or
infix:50 " ⊢ₛ " => entails

def owns (key : ResourceKey) (cell : Cell) : Assertion :=
  fun _ r => r = Resource.singleton key cell

def memoryByte (mem : ByteMem) (offset : Nat) (value : Byte) : Prop :=
  mem[offset]? = some value

def memoryBytes : ByteMem → Nat → List Byte → Prop
  | _, _, [] => True
  | mem, offset, byte :: bytes =>
      memoryByte mem offset byte ∧ memoryBytes mem (offset + 1) bytes

def globalByte (offset : Nat) (perm : BytePerm) (value : Byte) : Assertion :=
  fun st r =>
    owns (.globalByte offset) (.byte perm value) st r ∧
      memoryByte st.global.bytes offset value

def globalBytes : Nat → BytePerm → List Byte → Assertion
  | _, _, [] => emp
  | offset, perm, byte :: bytes => globalByte offset perm byte ∗ globalBytes (offset + 1) perm bytes

def sharedByte (cta : CTAId) (offset : Nat) (perm : BytePerm) (value : Byte) : Assertion :=
  fun st r =>
    owns (.sharedByte cta offset) (.byte perm value) st r ∧
      ∃ ctaState, st.getCTA? cta = some ctaState ∧
        memoryByte ctaState.shared.bytes offset value

def sharedBytes (cta : CTAId) : Nat → BytePerm → List Byte → Assertion
  | _, _, [] => emp
  | offset, perm, byte :: bytes =>
      sharedByte cta offset perm byte ∗ sharedBytes cta (offset + 1) perm bytes

def localByte
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (offset : Nat)
    (perm : BytePerm) (value : Byte) : Assertion :=
  fun st r =>
    owns (.localByte cta warp lane offset) (.byte perm value) st r ∧
      ∃ laneState, st.getLane? cta warp lane = some laneState ∧
        memoryByte laneState.localMem.bytes offset value

def localBytes (cta : CTAId) (warp : WarpId) (lane : LaneId) :
    Nat → BytePerm → List Byte → Assertion
  | _, _, [] => emp
  | offset, perm, byte :: bytes =>
      localByte cta warp lane offset perm byte ∗
        localBytes cta warp lane (offset + 1) perm bytes

def paramByte (offset : Nat) (value : Byte) : Assertion :=
  fun st r =>
    owns (.paramByte offset) (.byte .read value) st r ∧
      memoryByte st.param.bytes offset value

def paramBytes : Nat → List Byte → Assertion
  | _, [] => emp
  | offset, byte :: bytes => paramByte offset byte ∗ paramBytes (offset + 1) bytes

def constByte (offset : Nat) (value : Byte) : Assertion :=
  fun st r =>
    owns (.constByte offset) (.byte .read value) st r ∧
      memoryByte st.const.bytes offset value

def constBytes : Nat → List Byte → Assertion
  | _, [] => emp
  | offset, byte :: bytes => constByte offset byte ∗ constBytes (offset + 1) bytes

def reg (cta : CTAId) (warp : WarpId) (lane : LaneId) (name : RegName)
    (value : Value) : Assertion :=
  fun st r =>
    owns (.reg cta warp lane name) (.reg value) st r ∧
      ∃ laneState, st.getLane? cta warp lane = some laneState ∧
        laneState.regs[name]? = some value

def pred (cta : CTAId) (warp : WarpId) (lane : LaneId) (name : PredName)
    (value : Bool) : Assertion :=
  fun st r =>
    owns (.pred cta warp lane name) (.pred value) st r ∧
      ∃ laneState, st.getLane? cta warp lane = some laneState ∧
        laneState.preds[name]? = some value

def barrierToken
    (cta : CTAId) (barrierId epoch : Nat) (warp : WarpId) (lane : LaneId) : Assertion :=
  fun st r =>
    owns (.barrierToken cta barrierId epoch warp lane) (.token "barrier") st r ∧
      ∃ ctaState inst, st.getCTA? cta = some ctaState ∧
        ctaState.barrier.bars[barrierId]? = some inst ∧
        inst.epoch = epoch

theorem globalByte_memory
    {st : State} {r : Resource} {offset : Nat} {perm : BytePerm} {value : Byte}
    (h : globalByte offset perm value st r) :
    memoryByte st.global.bytes offset value :=
  h.2

theorem globalBytes_memory
    {st : State} {r : Resource} {offset : Nat} {perm : BytePerm} {bytes : List Byte}
    (h : globalBytes offset perm bytes st r) :
    memoryBytes st.global.bytes offset bytes := by
  induction bytes generalizing offset r with
  | nil =>
      simp [globalBytes, memoryBytes]
  | cons byte bytes ih =>
      change (globalByte offset perm byte ∗ globalBytes (offset + 1) perm bytes) st r at h
      rcases h with ⟨r₁, r₂, _hcomp, _hequiv, hbyte, hbytes⟩
      exact ⟨globalByte_memory hbyte, ih (offset := offset + 1) (r := r₂) hbytes⟩

theorem globalBytes_of_memory
    {st st' : State} {r : Resource} {offset : Nat} {perm : BytePerm}
    {bytes : List Byte}
    (h : globalBytes offset perm bytes st r)
    (hmem : memoryBytes st'.global.bytes offset bytes) :
    globalBytes offset perm bytes st' r := by
  induction bytes generalizing offset r with
  | nil =>
      simpa [globalBytes] using h
  | cons byte bytes ih =>
      change (globalByte offset perm byte ∗ globalBytes (offset + 1) perm bytes) st r at h
      change memoryByte st'.global.bytes offset byte ∧
        memoryBytes st'.global.bytes (offset + 1) bytes at hmem
      rcases h with ⟨rByte, rRest, hcomp, hequiv, hbyte, hrest⟩
      refine ⟨rByte, rRest, hcomp, hequiv, ?_, ih (offset := offset + 1)
        (r := rRest) hrest hmem.2⟩
      exact ⟨hbyte.1, hmem.1⟩

theorem sharedByte_memory
    {st : State} {r : Resource} {cta : CTAId} {offset : Nat}
    {perm : BytePerm} {value : Byte}
    (h : sharedByte cta offset perm value st r) :
    ∃ ctaState, st.getCTA? cta = some ctaState ∧
      memoryByte ctaState.shared.bytes offset value :=
  h.2

theorem sharedBytes_memory
    {st : State} {r : Resource} {cta : CTAId} {offset : Nat}
    {perm : BytePerm} {bytes : List Byte}
    (h : sharedBytes cta offset perm bytes st r) :
    ∀ ctaState, st.getCTA? cta = some ctaState →
      memoryBytes ctaState.shared.bytes offset bytes := by
  induction bytes generalizing offset r with
  | nil =>
      intro _ _
      simp [memoryBytes]
  | cons byte bytes ih =>
      intro ctaState hcta
      change (sharedByte cta offset perm byte ∗
        sharedBytes cta (offset + 1) perm bytes) st r at h
      rcases h with ⟨r₁, r₂, _hcomp, _hequiv, hbyte, hbytes⟩
      rcases sharedByte_memory hbyte with ⟨ctaState', hcta', hmem⟩
      rw [hcta] at hcta'
      injection hcta' with hsame
      subst ctaState'
      exact ⟨hmem, ih (offset := offset + 1) (r := r₂) hbytes ctaState hcta⟩

theorem sharedBytes_memory_exists
    {st : State} {r : Resource} {cta : CTAId} {offset : Nat}
    {perm : BytePerm} {byte : Byte} {bytes : List Byte}
    (h : sharedBytes cta offset perm (byte :: bytes) st r) :
    ∃ ctaState, st.getCTA? cta = some ctaState ∧
      memoryBytes ctaState.shared.bytes offset (byte :: bytes) := by
  have hfull := h
  change (sharedByte cta offset perm byte ∗
    sharedBytes cta (offset + 1) perm bytes) st r at h
  rcases h with ⟨_r₁, _r₂, _hcomp, _hequiv, hbyte, _hbytes⟩
  rcases sharedByte_memory hbyte with ⟨ctaState, hcta, _hmem⟩
  exact ⟨ctaState, hcta, sharedBytes_memory hfull ctaState hcta⟩

theorem localByte_memory
    {st : State} {r : Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : BytePerm} {value : Byte}
    (h : localByte cta warp lane offset perm value st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      memoryByte laneState.localMem.bytes offset value :=
  h.2

theorem localBytes_memory
    {st : State} {r : Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : BytePerm} {bytes : List Byte}
    (h : localBytes cta warp lane offset perm bytes st r) :
    ∀ laneState, st.getLane? cta warp lane = some laneState →
      memoryBytes laneState.localMem.bytes offset bytes := by
  induction bytes generalizing offset r with
  | nil =>
      intro _ _
      simp [memoryBytes]
  | cons byte bytes ih =>
      intro laneState hlane
      change (localByte cta warp lane offset perm byte ∗
        localBytes cta warp lane (offset + 1) perm bytes) st r at h
      rcases h with ⟨r₁, r₂, _hcomp, _hequiv, hbyte, hbytes⟩
      rcases localByte_memory hbyte with ⟨laneState', hlane', hmem⟩
      rw [hlane] at hlane'
      injection hlane' with hsame
      subst laneState'
      exact ⟨hmem, ih (offset := offset + 1) (r := r₂) hbytes laneState hlane⟩

theorem localBytes_memory_exists
    {st : State} {r : Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : BytePerm} {byte : Byte} {bytes : List Byte}
    (h : localBytes cta warp lane offset perm (byte :: bytes) st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      memoryBytes laneState.localMem.bytes offset (byte :: bytes) := by
  have hfull := h
  change (localByte cta warp lane offset perm byte ∗
    localBytes cta warp lane (offset + 1) perm bytes) st r at h
  rcases h with ⟨_r₁, _r₂, _hcomp, _hequiv, hbyte, _hbytes⟩
  rcases localByte_memory hbyte with ⟨laneState, hlane, _hmem⟩
  exact ⟨laneState, hlane, localBytes_memory hfull laneState hlane⟩

theorem paramByte_memory
    {st : State} {r : Resource} {offset : Nat} {value : Byte}
    (h : paramByte offset value st r) :
    memoryByte st.param.bytes offset value :=
  h.2

theorem paramBytes_memory
    {st : State} {r : Resource} {offset : Nat} {bytes : List Byte}
    (h : paramBytes offset bytes st r) :
    memoryBytes st.param.bytes offset bytes := by
  induction bytes generalizing offset r with
  | nil =>
      simp [paramBytes, memoryBytes]
  | cons byte bytes ih =>
      change (paramByte offset byte ∗ paramBytes (offset + 1) bytes) st r at h
      rcases h with ⟨r₁, r₂, _hcomp, _hequiv, hbyte, hbytes⟩
      exact ⟨paramByte_memory hbyte, ih (offset := offset + 1) (r := r₂) hbytes⟩

theorem constByte_memory
    {st : State} {r : Resource} {offset : Nat} {value : Byte}
    (h : constByte offset value st r) :
    memoryByte st.const.bytes offset value :=
  h.2

theorem constBytes_memory
    {st : State} {r : Resource} {offset : Nat} {bytes : List Byte}
    (h : constBytes offset bytes st r) :
    memoryBytes st.const.bytes offset bytes := by
  induction bytes generalizing offset r with
  | nil =>
      simp [constBytes, memoryBytes]
  | cons byte bytes ih =>
      change (constByte offset byte ∗ constBytes (offset + 1) bytes) st r at h
      rcases h with ⟨r₁, r₂, _hcomp, _hequiv, hbyte, hbytes⟩
      exact ⟨constByte_memory hbyte, ih (offset := offset + 1) (r := r₂) hbytes⟩

theorem reg_state
    {st : State} {r : Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : RegName} {value : Value}
    (h : reg cta warp lane name value st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      laneState.regs[name]? = some value :=
  h.2

theorem pred_state
    {st : State} {r : Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : PredName} {value : Bool}
    (h : pred cta warp lane name value st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      laneState.preds[name]? = some value :=
  h.2

theorem entails_refl (p : Assertion) : p ⊢ₛ p :=
  fun _ _ hp => hp

theorem entails_trans {p q r : Assertion} :
    p ⊢ₛ q → q ⊢ₛ r → p ⊢ₛ r := by
  intro hpq hqr st res hp
  exact hqr st res (hpq st res hp)

theorem resourceUpdate_refl (p : Assertion) :
    resourceUpdate p p := by
  intro st r hp
  exact ⟨r, Resource.update_refl r, hp⟩

theorem resourceUpdate_trans {p q r : Assertion} :
    resourceUpdate p q → resourceUpdate q r → resourceUpdate p r := by
  intro hpq hqr st res hp
  rcases hpq st res hp with ⟨res', hupdate, hq⟩
  rcases hqr st res' hq with ⟨res'', hupdate', hr⟩
  exact ⟨res'', Resource.update_trans hupdate hupdate', hr⟩

theorem resourceUpdate_mono {p p' q q' : Assertion} :
    p' ⊢ₛ p → resourceUpdate p q → q ⊢ₛ q' → resourceUpdate p' q' := by
  intro hp hupdate hq st r hp'
  rcases hupdate st r (hp st r hp') with ⟨r', hres, hq'⟩
  exact ⟨r', hres, hq _ _ hq'⟩

theorem owns_resourceUpdate {key : ResourceKey} {old new : Cell}
    (hshape : Cell.sameShape old new) :
    resourceUpdate (owns key old) (owns key new) := by
  intro st r howns
  subst r
  exact ⟨Resource.singleton key new, Resource.update_singleton hshape, rfl⟩

theorem sep_mono {p p' q q' : Assertion} :
    p ⊢ₛ p' → q ⊢ₛ q' → (p ∗ q) ⊢ₛ (p' ∗ q') := by
  intro hp hq st r hsep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hp₁, hq₂⟩
  exact ⟨r₁, r₂, hcomp, hequiv, hp st r₁ hp₁, hq st r₂ hq₂⟩

theorem sep_resourceUpdate {p p' q q' : Assertion} :
    resourceUpdate p p' → resourceUpdate q q' → resourceUpdate (p ∗ q) (p' ∗ q') := by
  intro hp hq st r hsep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hp₁, hq₂⟩
  rcases hp st r₁ hp₁ with ⟨r₁', hupdate₁, hp'⟩
  rcases hq st r₂ hq₂ with ⟨r₂', hupdate₂, hq'⟩
  refine ⟨Resource.compose r₁' r₂', ?_, ?_⟩
  · exact Resource.update_trans (Resource.update_of_equiv hequiv)
      (Resource.update_compose hupdate₁ hupdate₂)
  · exact ⟨r₁', r₂',
      Resource.canCompose_update_right hupdate₂
        (Resource.canCompose_update_left hupdate₁ hcomp),
      Resource.equiv_refl _, hp', hq'⟩

theorem sep_comm (p q : Assertion) :
    (p ∗ q) ⊢ₛ (q ∗ p) := by
  intro st r h
  rcases h with ⟨r₁, r₂, hcomp, hequiv, hp, hq⟩
  refine ⟨r₂, r₁, Resource.canCompose_comm hcomp, ?_, hq, hp⟩
  exact Resource.equiv_trans hequiv (Resource.compose_comm_equiv r₁ r₂)

theorem sep_assoc (p q r : Assertion) :
    ((p ∗ q) ∗ r) ⊢ₛ (p ∗ (q ∗ r)) := by
  intro st res h
  rcases h with ⟨rpq, rr, hcompPqR, hequiv, hpq, hr⟩
  rcases hpq with ⟨rp, rq, hcompPQ, hequivPQ, hp, hq⟩
  refine ⟨rp, Resource.compose rq rr, ?_, ?_, hp, ?_⟩
  · intro ep hep erqr herqr
    simp [Resource.compose] at herqr
    rcases herqr with herq | herr
    · exact hcompPQ ep hep erqr herq
    · have hepq : ep ∈ rpq.entries := by
        exact (hequivPQ ep).2 (by simp [Resource.compose, hep])
      exact hcompPqR ep hepq erqr herr
  · intro entry
    constructor
    · intro hentry
      have hmid := (hequiv entry).1 hentry
      simp [Resource.compose] at hmid ⊢
      rcases hmid with hpqEntry | hrEntry
      · have hpqSplit := (hequivPQ entry).1 hpqEntry
        simp [Resource.compose] at hpqSplit
        exact hpqSplit.elim Or.inl (fun h => Or.inr (Or.inl h))
      · exact Or.inr (Or.inr hrEntry)
    · intro hentry
      simp [Resource.compose] at hentry
      apply (hequiv entry).2
      simp [Resource.compose]
      rcases hentry with hpEntry | hqrEntry
      · exact Or.inl ((hequivPQ entry).2 (by simp [Resource.compose, hpEntry]))
      · rcases hqrEntry with hqEntry | hrEntry
        · exact Or.inl ((hequivPQ entry).2 (by simp [Resource.compose, hqEntry]))
        · exact Or.inr hrEntry
  · refine ⟨rq, rr, ?_, Resource.equiv_refl _, hq, hr⟩
    intro eq heq er herr
    have heqPq : eq ∈ rpq.entries := by
      exact (hequivPQ eq).2 (by simp [Resource.compose, heq])
    exact hcompPqR eq heqPq er herr

theorem sep_assoc_rev (p q r : Assertion) :
    (p ∗ (q ∗ r)) ⊢ₛ ((p ∗ q) ∗ r) := by
  intro st res h
  rcases h with ⟨rp, rqr, hcompPQr, hequiv, hp, hqr⟩
  rcases hqr with ⟨rq, rr, hcompQR, hequivQR, hq, hr⟩
  refine ⟨Resource.compose rp rq, rr, ?_, ?_, ?_, hr⟩
  · intro epq hepq er her
    simp [Resource.compose] at hepq
    rcases hepq with hep | heq
    · have herQr : er ∈ rqr.entries := by
        exact (hequivQR er).2 (by simp [Resource.compose, her])
      exact hcompPQr epq hep er herQr
    · exact hcompQR epq heq er her
  · intro entry
    constructor
    · intro hentry
      have hmid := (hequiv entry).1 hentry
      simp [Resource.compose] at hmid ⊢
      rcases hmid with hpEntry | hqrEntry
      · exact Or.inl hpEntry
      · have hqrSplit := (hequivQR entry).1 hqrEntry
        simp [Resource.compose] at hqrSplit
        exact hqrSplit.elim (fun h => Or.inr (Or.inl h)) (fun h => Or.inr (Or.inr h))
    · intro hentry
      simp [Resource.compose] at hentry
      apply (hequiv entry).2
      simp [Resource.compose]
      rcases hentry with hpEntry | hrest
      · exact Or.inl hpEntry
      · rcases hrest with hqEntry | hrEntry
        · exact Or.inr ((hequivQR entry).2 (by simp [Resource.compose, hqEntry]))
        · exact Or.inr ((hequivQR entry).2 (by simp [Resource.compose, hrEntry]))
  · refine ⟨rp, rq, ?_, Resource.equiv_refl _, hp, hq⟩
    intro ep hep eq heq
    have heqQr : eq ∈ rqr.entries := by
      exact (hequivQR eq).2 (by simp [Resource.compose, heq])
    exact hcompPQr ep hep eq heqQr

theorem sep_swap_ab_c (a b c : Assertion) :
    (a ∗ (b ∗ c)) ⊢ₛ (b ∗ (a ∗ c)) :=
  entails_trans (sep_assoc_rev a b c) <|
    entails_trans (sep_mono (sep_comm a b) (entails_refl c)) <|
      sep_assoc b a c

theorem sep_swap_bc (a b c : Assertion) :
    (a ∗ (b ∗ c)) ⊢ₛ (a ∗ (c ∗ b)) :=
  sep_mono (entails_refl a) (sep_comm b c)

theorem sep_rotate_three_last_to_front (a b c : Assertion) :
    (a ∗ (b ∗ c)) ⊢ₛ (c ∗ (a ∗ b)) :=
  entails_trans (sep_assoc_rev a b c) (sep_comm (a ∗ b) c)

theorem sep_rotate_three_front_to_last (a b c : Assertion) :
    (c ∗ (a ∗ b)) ⊢ₛ (a ∗ (b ∗ c)) :=
  entails_trans (sep_comm c (a ∗ b)) (sep_assoc a b c)

def sepList : List Assertion → Assertion
  | [] => emp
  | p :: [] => p
  | p :: q :: rest => p ∗ sepList (q :: rest)

@[simp] theorem sepList_nil :
    sepList [] = emp :=
  rfl

@[simp] theorem sepList_singleton (p : Assertion) :
    sepList [p] = p :=
  rfl

@[simp] theorem sepList_cons_cons (p q : Assertion) (rest : List Assertion) :
    sepList (p :: q :: rest) = p ∗ sepList (q :: rest) :=
  rfl

theorem sepList_swap_head (a b : Assertion) (rest : List Assertion) :
    sepList (a :: b :: rest) ⊢ₛ sepList (b :: a :: rest) := by
  cases rest with
  | nil =>
      exact sep_comm a b
  | cons c rest =>
      exact sep_swap_ab_c a b (sepList (c :: rest))

private theorem sepList_cons_entails
    {a : Assertion} {xs ys : List Assertion}
    (hperm : xs.Perm ys)
    (h : sepList xs ⊢ₛ sepList ys) :
    sepList (a :: xs) ⊢ₛ sepList (a :: ys) := by
  cases xs with
  | nil =>
      cases ys with
      | nil =>
          exact entails_refl _
      | cons y ys =>
          have hlen := hperm.length_eq
          simp at hlen
  | cons x xs =>
      cases ys with
      | nil =>
          have hlen := hperm.length_eq
          simp at hlen
      | cons y ys =>
          exact sep_mono (entails_refl a) h

theorem sepList_perm {xs ys : List Assertion} (hperm : xs.Perm ys) :
    sepList xs ⊢ₛ sepList ys := by
  induction hperm with
  | nil =>
      exact entails_refl _
  | cons _ hperm ih =>
      exact sepList_cons_entails hperm ih
  | swap a b rest =>
      exact sepList_swap_head b a rest
  | trans _ _ ih₁ ih₂ =>
      exact entails_trans ih₁ ih₂

theorem sepList_append_cons
    (a : Assertion) (xs : List Assertion) (b : Assertion) (ys : List Assertion) :
    (sepList (a :: xs) ∗ sepList (b :: ys)) ⊢ₛ
      sepList ((a :: xs) ++ (b :: ys)) := by
  induction xs generalizing a with
  | nil =>
      exact entails_refl _
  | cons x xs ih =>
      simpa [sepList, List.cons_append] using
        entails_trans (sep_assoc a (sepList (x :: xs)) (sepList (b :: ys)))
          (sep_mono (entails_refl a) (ih x))

theorem sepList_append_cons_rev
    (a : Assertion) (xs : List Assertion) (b : Assertion) (ys : List Assertion) :
    sepList ((a :: xs) ++ (b :: ys)) ⊢ₛ
      (sepList (a :: xs) ∗ sepList (b :: ys)) := by
  induction xs generalizing a with
  | nil =>
      exact entails_refl _
  | cons x xs ih =>
      simpa [sepList, List.cons_append] using
        entails_trans (sep_mono (entails_refl a) (ih x))
          (sep_assoc_rev a (sepList (x :: xs)) (sepList (b :: ys)))

theorem sep_pair_cons_to_sepList (a b c : Assertion) (rest : List Assertion) :
    ((a ∗ b) ∗ sepList (c :: rest)) ⊢ₛ sepList (a :: b :: c :: rest) :=
  sep_assoc a b (sepList (c :: rest))

theorem sepList_to_sep_pair_cons (a b c : Assertion) (rest : List Assertion) :
    sepList (a :: b :: c :: rest) ⊢ₛ ((a ∗ b) ∗ sepList (c :: rest)) :=
  sep_assoc_rev a b (sepList (c :: rest))

theorem sep_pair_cons_perm
    (a b c x y z : Assertion) (rest rest' : List Assertion)
    (hperm : (a :: b :: c :: rest).Perm (x :: y :: z :: rest')) :
    ((a ∗ b) ∗ sepList (c :: rest)) ⊢ₛ ((x ∗ y) ∗ sepList (z :: rest')) :=
  entails_trans (sep_pair_cons_to_sepList a b c rest) <|
    entails_trans (sepList_perm hperm) <|
      sepList_to_sep_pair_cons x y z rest'

theorem sep_pair_cons_perm_to_sepList
    (a b c : Assertion) (rest target : List Assertion)
    (hperm : (a :: b :: c :: rest).Perm target) :
    ((a ∗ b) ∗ sepList (c :: rest)) ⊢ₛ sepList target :=
  entails_trans (sep_pair_cons_to_sepList a b c rest) (sepList_perm hperm)

theorem sepList_perm_to_sep_pair_cons
    (source : List Assertion) (x y z : Assertion) (rest' : List Assertion)
    (hperm : source.Perm (x :: y :: z :: rest')) :
    sepList source ⊢ₛ ((x ∗ y) ∗ sepList (z :: rest')) :=
  entails_trans (sepList_perm hperm) (sepList_to_sep_pair_cons x y z rest')

theorem sepList_perm_to_cons
    (source : List Assertion) (x y : Assertion) (rest : List Assertion)
    (hperm : source.Perm (x :: y :: rest)) :
    sepList source ⊢ₛ (x ∗ sepList (y :: rest)) := by
  simpa [sepList] using sepList_perm hperm

theorem sep_cons_perm_to_sepList
    (x y : Assertion) (rest target : List Assertion)
    (hperm : (x :: y :: rest).Perm target) :
    (x ∗ sepList (y :: rest)) ⊢ₛ sepList target := by
  simpa [sepList] using sepList_perm hperm

theorem sepList_perm_frame_to_cons
    (source : List Assertion) (x y : Assertion) (rest : List Assertion)
    (frame : Assertion)
    (hperm : source.Perm (x :: y :: rest)) :
    (sepList source ∗ frame) ⊢ₛ (x ∗ (sepList (y :: rest) ∗ frame)) := by
  exact entails_trans
    (sep_mono (sepList_perm hperm) (entails_refl frame))
    (by simpa [sepList] using sep_assoc x (sepList (y :: rest)) frame)

theorem sep_cons_frame_to_sepList_perm
    (x y : Assertion) (rest target : List Assertion) (frame : Assertion)
    (hperm : (x :: y :: rest).Perm target) :
    (x ∗ (sepList (y :: rest) ∗ frame)) ⊢ₛ (sepList target ∗ frame) := by
  exact entails_trans
    (by simpa [sepList] using sep_assoc_rev x (sepList (y :: rest)) frame)
    (sep_mono (sepList_perm hperm) (entails_refl frame))

theorem sep_cons_frame_perm_to_cons
    (x y x' y' : Assertion) (rest rest' : List Assertion) (frame : Assertion)
    (hperm : (x :: y :: rest).Perm (x' :: y' :: rest')) :
    (x ∗ (sepList (y :: rest) ∗ frame)) ⊢ₛ
      (x' ∗ (sepList (y' :: rest') ∗ frame)) := by
  exact entails_trans
    (sep_cons_frame_to_sepList_perm x y rest (x' :: y' :: rest') frame hperm)
    (sepList_perm_frame_to_cons (x' :: y' :: rest') x' y' rest' frame (List.Perm.refl _))

theorem sep_permute_acdb (a b c d : Assertion) :
    (a ∗ (b ∗ (c ∗ d))) ⊢ₛ (c ∗ (a ∗ (d ∗ b))) :=
  entails_trans (sep_assoc_rev a b (c ∗ d)) <|
    entails_trans (sep_comm (a ∗ b) (c ∗ d)) <|
      entails_trans (sep_assoc c d (a ∗ b)) <|
        sep_mono (entails_refl c) <|
          entails_trans (sep_assoc_rev d a b) <|
            entails_trans (sep_mono (sep_comm d a) (entails_refl b)) <|
              sep_assoc a d b

theorem sep_permute_ab_cde_ecbda (a b c d e : Assertion) :
    ((a ∗ b) ∗ (c ∗ (d ∗ e))) ⊢ₛ ((e ∗ c) ∗ (b ∗ (d ∗ a))) :=
  entails_trans (sep_assoc_rev (a ∗ b) c (d ∗ e)) <|
    entails_trans (sep_comm ((a ∗ b) ∗ c) (d ∗ e)) <|
      entails_trans (sep_assoc d e ((a ∗ b) ∗ c)) <|
        entails_trans (sep_swap_ab_c d e ((a ∗ b) ∗ c)) <|
          entails_trans
            (sep_mono (entails_refl e) <|
              entails_trans
                (sep_mono (entails_refl d) (sep_comm (a ∗ b) c)) <|
                entails_trans (sep_swap_ab_c d c (a ∗ b)) <|
                  sep_mono (entails_refl c) <|
                    entails_trans
                      (sep_mono (entails_refl d) (sep_comm a b)) <|
                      sep_swap_ab_c d b a)
            (sep_assoc_rev e c (b ∗ (d ∗ a)))

theorem sep_permute_ab_cde_dcb_ea (a b c d e : Assertion) :
    ((a ∗ b) ∗ (c ∗ (d ∗ e))) ⊢ₛ (d ∗ (c ∗ (b ∗ (e ∗ a)))) :=
  entails_trans (sep_assoc_rev (a ∗ b) c (d ∗ e)) <|
    entails_trans (sep_comm ((a ∗ b) ∗ c) (d ∗ e)) <|
      entails_trans (sep_assoc d e ((a ∗ b) ∗ c)) <|
        sep_mono (entails_refl d) <|
          entails_trans
            (sep_mono (entails_refl e) (sep_comm (a ∗ b) c)) <|
            entails_trans (sep_swap_ab_c e c (a ∗ b)) <|
              sep_mono (entails_refl c) <|
                entails_trans
                  (sep_mono (entails_refl e) (sep_comm a b)) <|
                  sep_swap_ab_c e b a

theorem sep_permute_ab_cdef_ecbdaf (a b c d e f : Assertion) :
    ((a ∗ b) ∗ (c ∗ (d ∗ (e ∗ f)))) ⊢ₛ
      ((e ∗ c) ∗ (b ∗ (d ∗ (a ∗ f)))) :=
  entails_trans (sep_assoc a b (c ∗ (d ∗ (e ∗ f)))) <|
    entails_trans
      (sep_mono (entails_refl a) <|
        sep_mono (entails_refl b) <|
          sep_mono (entails_refl c) <|
            sep_swap_ab_c d e f) <|
      entails_trans
        (sep_mono (entails_refl a) <|
          sep_mono (entails_refl b) <|
            sep_swap_ab_c c e (d ∗ f)) <|
        entails_trans
          (sep_mono (entails_refl a) <|
            sep_swap_ab_c b e (c ∗ (d ∗ f))) <|
          entails_trans
            (sep_swap_ab_c a e (b ∗ (c ∗ (d ∗ f)))) <|
            entails_trans
              (sep_mono (entails_refl e) <|
                sep_mono (entails_refl a) <|
                  sep_swap_ab_c b c (d ∗ f)) <|
              entails_trans
                (sep_mono (entails_refl e) <|
                  sep_swap_ab_c a c (b ∗ (d ∗ f))) <|
                entails_trans
                  (sep_mono (entails_refl e) <|
                    sep_mono (entails_refl c) <|
                      sep_swap_ab_c a b (d ∗ f)) <|
                  entails_trans
                    (sep_mono (entails_refl e) <|
                      sep_mono (entails_refl c) <|
                        sep_mono (entails_refl b) <|
                          sep_swap_ab_c a d f) <|
                    sep_assoc_rev e c (b ∗ (d ∗ (a ∗ f)))

theorem sep_permute_ab_cdef_dcbeaf (a b c d e f : Assertion) :
    ((a ∗ b) ∗ (c ∗ (d ∗ (e ∗ f)))) ⊢ₛ
      (d ∗ (c ∗ (b ∗ (e ∗ (a ∗ f))))) :=
  entails_trans (sep_assoc a b (c ∗ (d ∗ (e ∗ f)))) <|
    entails_trans
      (sep_mono (entails_refl a) <|
        sep_mono (entails_refl b) <|
          sep_swap_ab_c c d (e ∗ f)) <|
      entails_trans
        (sep_mono (entails_refl a) <|
          sep_swap_ab_c b d (c ∗ (e ∗ f))) <|
        entails_trans
          (sep_swap_ab_c a d (b ∗ (c ∗ (e ∗ f)))) <|
          entails_trans
            (sep_mono (entails_refl d) <|
              sep_mono (entails_refl a) <|
                sep_swap_ab_c b c (e ∗ f)) <|
            entails_trans
              (sep_mono (entails_refl d) <|
                sep_swap_ab_c a c (b ∗ (e ∗ f))) <|
              entails_trans
                (sep_mono (entails_refl d) <|
                  sep_mono (entails_refl c) <|
                    sep_swap_ab_c a b (e ∗ f)) <|
                sep_mono (entails_refl d) <|
                  sep_mono (entails_refl c) <|
                    sep_mono (entails_refl b) <|
                      sep_swap_ab_c a e f

theorem sep_rotate_six_last_to_front (a b c d e f : Assertion) :
    (a ∗ (b ∗ (c ∗ (d ∗ (e ∗ f))))) ⊢ₛ
      (f ∗ (a ∗ (b ∗ (c ∗ (d ∗ e))))) :=
  entails_trans
    (sep_mono (entails_refl a) <|
      sep_mono (entails_refl b) <|
        sep_mono (entails_refl c) <|
          sep_mono (entails_refl d) <|
            sep_comm e f) <|
    entails_trans
      (sep_mono (entails_refl a) <|
        sep_mono (entails_refl b) <|
          sep_mono (entails_refl c) <|
            sep_swap_ab_c d f e) <|
      entails_trans
        (sep_mono (entails_refl a) <|
          sep_mono (entails_refl b) <|
            sep_swap_ab_c c f (d ∗ e)) <|
        entails_trans
          (sep_mono (entails_refl a) <|
            sep_swap_ab_c b f (c ∗ (d ∗ e))) <|
          sep_swap_ab_c a f (b ∗ (c ∗ (d ∗ e)))

def StableUnder (step : State → State → Prop) (p : Assertion) : Prop :=
  ∀ st st' r, step st st' → p st r → p st' r

theorem stable_emp {step : State → State → Prop} : StableUnder step emp := by
  intro _ _ _ _ h
  exact h

theorem stable_pure {step : State → State → Prop} {p : Prop} :
    StableUnder step (pure p) := by
  intro _ _ _ _ h
  exact h

theorem stable_sep {step : State → State → Prop} {p q : Assertion}
    (hp : StableUnder step p) (hq : StableUnder step q) :
    StableUnder step (p ∗ q) := by
  intro st st' r hstep hsep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hp₁, hq₂⟩
  exact ⟨r₁, r₂, hcomp, hequiv, hp st st' r₁ hstep hp₁, hq st st' r₂ hstep hq₂⟩

theorem stable_sepList {step : State → State → Prop} :
    ∀ {xs : List Assertion},
      (∀ p, p ∈ xs → StableUnder step p) → StableUnder step (sepList xs)
  | [], _ => by
      simpa [sepList] using (stable_emp (step := step))
  | [p], h => by
      simpa [sepList] using h p (by simp)
  | p :: q :: rest, h => by
      have hp : StableUnder step p := h p (by simp)
      have hrest : StableUnder step (sepList (q :: rest)) :=
        stable_sepList (xs := q :: rest) (by
          intro a ha
          exact h a (by simp [ha]))
      simpa [sepList] using stable_sep hp hrest

end CSL
end CLean
