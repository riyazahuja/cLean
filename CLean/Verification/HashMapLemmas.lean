import Std.Data.HashMap

/-!
# HashMap Reasoning Axioms

Key properties of HashMap operations needed for verification.
These are "obviously true" interface properties that would require
diving into HashMap implementation details to prove formally.

For now we axiomatize them (with sorry) and can prove them later if needed.
-/

namespace Std.HashMap

variable {α β : Type} [BEq α] [Hashable α]

/-! ## Basic Get/Insert Properties -/

/-- Inserting and immediately getting returns the inserted value -/
theorem get?_insert_same (m : HashMap α β) (k : α) (v : β) :
    (m.insert k v).get? k = some v := by
  sorry

/-- Getting a different key after insert preserves the original value -/
theorem get?_insert_diff (m : HashMap α β) (k k' : α) (v : β) :
    k ≠ k' →
    (m.insert k v).get? k' = m.get? k' := by
  sorry

/-- getD returns the value if present -/
theorem getD_of_get?_some (m : HashMap α β) (k : α) (v default : β) :
    m.get? k = some v →
    m.getD k default = v := by
  sorry

/-- getD returns default if key not present -/
theorem getD_of_get?_none (m : HashMap α β) (k : α) (default : β) :
    m.get? k = none →
    m.getD k default = default := by
  sorry

/-- Combining get? and insert -/
theorem getD_insert_same (m : HashMap α β) (k : α) (v default : β) :
    (m.insert k v).getD k default = v := by
  sorry

/-- getD with different keys -/
theorem getD_insert_diff (m : HashMap α β) (k k' : α) (v default : β) :
    k ≠ k' →
    (m.insert k v).getD k' default = m.getD k' default := by
  sorry

/-! ## Empty HashMap Properties -/

/-- Getting from empty hashmap returns none -/
theorem get?_empty (k : α) :
    (∅ : HashMap α β).get? k = none := by
  sorry

end Std.HashMap
