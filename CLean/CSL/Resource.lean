import CLean.Core.State

namespace CLean
namespace CSL

inductive ResourceKey where
  | globalByte (offset : Nat)
  | sharedByte (cta : CTAId) (offset : Nat)
  | localByte (cta : CTAId) (warp : WarpId) (lane : LaneId) (offset : Nat)
  | paramByte (offset : Nat)
  | constByte (offset : Nat)
  | reg (cta : CTAId) (warp : WarpId) (lane : LaneId) (name : RegName)
  | pred (cta : CTAId) (warp : WarpId) (lane : LaneId) (name : PredName)
  | barrierToken (cta : CTAId) (barrierId epoch : Nat) (warp : WarpId) (lane : LaneId)
  | warpToken (cta : CTAId) (warp : WarpId) (tag : String)
  | atomicToken (addr : Addr)
  | frag (cta : CTAId) (warp : WarpId) (lane : LaneId) (name : RegName)
  deriving Repr, DecidableEq, Inhabited

inductive BytePerm where
  | read
  | write
  deriving Repr, DecidableEq, Inhabited

inductive Cell where
  | byte (perm : BytePerm) (value : Byte)
  | reg (value : Value)
  | pred (value : Bool)
  | token (tag : String)
  | frag (ty : FragTy) (payload : Array Value)
  deriving Repr, Inhabited

namespace Cell

def sameShape : Cell → Cell → Prop
  | .byte perm₁ _, .byte perm₂ _ => perm₁ = perm₂
  | .reg _, .reg _ => True
  | .pred _, .pred _ => True
  | .token tag₁, .token tag₂ => tag₁ = tag₂
  | .frag ty₁ _, .frag ty₂ _ => ty₁ = ty₂
  | _, _ => False

def compatible : Cell → Cell → Prop
  | .byte .read _, .byte .read _ => True
  | _, _ => False

theorem sameShape_refl (cell : Cell) : sameShape cell cell := by
  cases cell <;> simp [sameShape]

theorem sameShape_symm {a b : Cell} :
    sameShape a b → sameShape b a := by
  intro h
  cases a <;> cases b <;> simp [sameShape] at h ⊢
  · exact h.symm
  · exact h.symm
  · exact h.symm

theorem sameShape_trans {a b c : Cell} :
    sameShape a b → sameShape b c → sameShape a c := by
  intro hab hbc
  cases a <;> cases b <;> cases c <;> simp [sameShape] at hab hbc ⊢
  · exact hab.trans hbc
  · exact hab.trans hbc
  · exact hab.trans hbc

theorem compatible_comm {a b : Cell} :
    compatible a b → compatible b a := by
  intro h
  cases a <;> cases b <;> simp [compatible] at h ⊢
  case byte.byte permA valueA permB valueB =>
    cases permA <;> cases permB <;> simp [compatible] at h ⊢

theorem compatible_of_sameShape_left {a a' b : Cell}
    (hshape : sameShape a a') :
    compatible a b → compatible a' b := by
  intro hcompat
  cases a <;> cases a' <;> cases b <;> simp [sameShape, compatible] at hshape hcompat ⊢
  case byte.byte.byte permA _ permA' _ permB _ =>
    cases permA <;> cases permA' <;> cases permB <;>
      simp [sameShape, compatible] at hshape hcompat ⊢

theorem compatible_of_sameShape_right {a b b' : Cell}
    (hshape : sameShape b b') :
    compatible a b → compatible a b' := by
  intro hcompat
  exact compatible_comm <| compatible_of_sameShape_left hshape <|
    compatible_comm hcompat

end Cell

abbrev Entry := ResourceKey × Cell

namespace Entry

def compatible (a b : Entry) : Prop :=
  a.1 ≠ b.1 ∨ Cell.compatible a.2 b.2

theorem compatible_comm {a b : Entry} :
    compatible a b → compatible b a := by
  intro h
  rcases h with hkey | hcell
  · exact Or.inl (fun heq => hkey heq.symm)
  · exact Or.inr (Cell.compatible_comm hcell)

theorem compatible_of_sameShape_left {key : ResourceKey} {old new : Cell} {b : Entry}
    (hshape : Cell.sameShape old new) :
    compatible (key, old) b → compatible (key, new) b := by
  intro hcompat
  rcases hcompat with hkey | hcell
  · exact Or.inl hkey
  · exact Or.inr (Cell.compatible_of_sameShape_left hshape hcell)

theorem compatible_of_sameShape_right {a : Entry} {key : ResourceKey} {old new : Cell}
    (hshape : Cell.sameShape old new) :
    compatible a (key, old) → compatible a (key, new) := by
  intro hcompat
  exact compatible_comm <| compatible_of_sameShape_left hshape <| compatible_comm hcompat

end Entry

structure Resource where
  entries : List Entry := []
  deriving Repr, Inhabited

namespace Resource

def empty : Resource :=
  {}

def singleton (key : ResourceKey) (cell : Cell) : Resource :=
  { entries := [(key, cell)] }

def contains (key : ResourceKey) (cell : Cell) (r : Resource) : Prop :=
  (key, cell) ∈ r.entries

def withoutKey (key : ResourceKey) (r : Resource) : Resource :=
  { entries := r.entries.filter fun entry => entry.1 != key }

def insert (key : ResourceKey) (cell : Cell) (r : Resource) : Resource :=
  { entries := (key, cell) :: (withoutKey key r).entries }

def replace (key : ResourceKey) (cell : Cell) (r : Resource) : Resource :=
  insert key cell r

def compose (a b : Resource) : Resource :=
  { entries := a.entries ++ b.entries }

def Equiv (a b : Resource) : Prop :=
  ∀ entry, entry ∈ a.entries ↔ entry ∈ b.entries

def canCompose (a b : Resource) : Prop :=
  ∀ ea ∈ a.entries, ∀ eb ∈ b.entries, Entry.compatible ea eb

def valid (r : Resource) : Prop :=
  ∀ ea ∈ r.entries, ∀ eb ∈ r.entries, ea = eb ∨ Entry.compatible ea eb

def Update (a b : Resource) : Prop :=
  (∀ key cell, contains key cell a →
    ∃ cell', contains key cell' b ∧ Cell.sameShape cell cell') ∧
  (∀ key cell, contains key cell b →
    ∃ cell', contains key cell' a ∧ Cell.sameShape cell cell')

@[simp] theorem compose_empty_right (r : Resource) : compose r empty = r := by
  cases r
  simp [compose, empty]

@[simp] theorem compose_empty_left (r : Resource) : compose empty r = r := by
  cases r
  simp [compose, empty]

theorem compose_assoc (a b c : Resource) :
    compose (compose a b) c = compose a (compose b c) := by
  cases a
  cases b
  cases c
  simp [compose, List.append_assoc]

theorem equiv_refl (r : Resource) : Equiv r r := by
  intro entry
  rfl

theorem equiv_symm {a b : Resource} : Equiv a b → Equiv b a := by
  intro h entry
  exact (h entry).symm

theorem equiv_trans {a b c : Resource} : Equiv a b → Equiv b c → Equiv a c := by
  intro hab hbc entry
  exact (hab entry).trans (hbc entry)

theorem compose_comm_equiv (a b : Resource) : Equiv (compose a b) (compose b a) := by
  intro entry
  simp [Equiv, compose, or_comm]

theorem canCompose_comm {a b : Resource} :
    canCompose a b → canCompose b a := by
  intro h eb heb ea hea
  exact Entry.compatible_comm (h ea hea eb heb)

theorem update_refl (r : Resource) : Update r r := by
  constructor <;> intro key cell hmem <;>
    exact ⟨cell, hmem, Cell.sameShape_refl cell⟩

theorem update_symm {a b : Resource} :
    Update a b → Update b a := by
  intro h
  exact ⟨h.2, h.1⟩

theorem update_trans {a b c : Resource} :
    Update a b → Update b c → Update a c := by
  intro hab hbc
  constructor
  · intro key cell hmem
    rcases hab.1 key cell hmem with ⟨cell', hmem', hshape⟩
    rcases hbc.1 key cell' hmem' with ⟨cell'', hmem'', hshape'⟩
    exact ⟨cell'', hmem'', Cell.sameShape_trans hshape hshape'⟩
  · intro key cell hmem
    rcases hbc.2 key cell hmem with ⟨cell', hmem', hshape⟩
    rcases hab.2 key cell' hmem' with ⟨cell'', hmem'', hshape'⟩
    exact ⟨cell'', hmem'', Cell.sameShape_trans hshape hshape'⟩

theorem update_of_equiv {a b : Resource} :
    Equiv a b → Update a b := by
  intro hequiv
  constructor <;> intro key cell hmem
  · exact ⟨cell, (hequiv (key, cell)).mp hmem, Cell.sameShape_refl cell⟩
  · exact ⟨cell, (hequiv (key, cell)).mpr hmem, Cell.sameShape_refl cell⟩

theorem update_compose_right {a a' b : Resource} :
    Update a a' → Update (compose a b) (compose a' b) := by
  intro hupdate
  constructor
  · intro key cell hmem
    simp [contains, compose] at hmem ⊢
    rcases hmem with hmem | hmem
    · rcases hupdate.1 key cell hmem with ⟨cell', hmem', hshape⟩
      exact ⟨cell', Or.inl hmem', hshape⟩
    · exact ⟨cell, Or.inr hmem, Cell.sameShape_refl cell⟩
  · intro key cell hmem
    simp [contains, compose] at hmem ⊢
    rcases hmem with hmem | hmem
    · rcases hupdate.2 key cell hmem with ⟨cell', hmem', hshape⟩
      exact ⟨cell', Or.inl hmem', hshape⟩
    · exact ⟨cell, Or.inr hmem, Cell.sameShape_refl cell⟩

theorem update_compose_left {a b b' : Resource} :
    Update b b' → Update (compose a b) (compose a b') := by
  intro hupdate
  have h := update_compose_right (a := b) (a' := b') (b := a) hupdate
  exact update_trans (update_of_equiv (compose_comm_equiv a b))
    (update_trans h (update_of_equiv (compose_comm_equiv b' a)))

theorem update_compose {a a' b b' : Resource} :
    Update a a' → Update b b' → Update (compose a b) (compose a' b') := by
  intro ha hb
  exact update_trans (update_compose_right ha) (update_compose_left hb)

theorem update_singleton {key : ResourceKey} {old new : Cell}
    (hshape : Cell.sameShape old new) :
    Update (singleton key old) (singleton key new) := by
  constructor
  · intro key' cell hmem
    simp [contains, singleton] at hmem
    rcases hmem with ⟨rfl, rfl⟩
    exact ⟨new, by simp [contains, singleton], hshape⟩
  · intro key' cell hmem
    simp [contains, singleton] at hmem
    rcases hmem with ⟨rfl, rfl⟩
    exact ⟨old, by simp [contains, singleton], Cell.sameShape_symm hshape⟩

theorem canCompose_update_left {a a' b : Resource}
    (hupdate : Update a a') (hcomp : canCompose a b) :
    canCompose a' b := by
  intro ea hea eb heb
  rcases hupdate.2 ea.1 ea.2 hea with ⟨oldCell, hold, hshape⟩
  exact Entry.compatible_of_sameShape_left (Cell.sameShape_symm hshape)
    (hcomp (ea.1, oldCell) hold eb heb)

theorem canCompose_update_right {a b b' : Resource}
    (hupdate : Update b b') (hcomp : canCompose a b) :
    canCompose a b' := by
  intro ea hea eb heb
  rcases hupdate.2 eb.1 eb.2 heb with ⟨oldCell, hold, hshape⟩
  exact Entry.compatible_of_sameShape_right (Cell.sameShape_symm hshape)
    (hcomp ea hea (eb.1, oldCell) hold)

end Resource

end CSL
end CLean
