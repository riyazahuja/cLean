import CLean.Examples.Saxpy
import CLean.Proof.InstrValueCompute

/-! # Saxpy chain proof: structural foundation

Foundation lemmas characterizing the initial state `saxpyStateFor n α xs ys`
in terms suitable for stepping through the 24 step? applications that
constitute one saxpy execution per active lane.

These lemmas establish the structural preconditions for applying the
`stepInstr?_*_lane_full` lemmas in `Proof/InstrValueCompute.lean`:

- `saxpyStateFor_getWarp`: the initial warp state at `(cta=0, warp=0)` is `saxpyWarpFor n`.
- `saxpyWarp_wf`: the initial warp state is well-formed (32 lanes).
- `saxpyWarp_lockstepRunnable`: the initial warp state is in lockstep (all lanes at same PC).
- `saxpyWarp_lane_pc`: every lane in the initial warp state is at `("saxpyKernel", 0)`.

The chain proof itself (using these foundations + `stepInstr?_*_lane_full`)
is left to a future session. See `MEMORY.md` for the proof plan.
-/

namespace CLean

open Helpers

/-- **Lane PC characterization (initial state):** every lane in
`saxpyWarpFor n` is at PC `("saxpyKernel", 0)` — the entry of BB0.

Used as a building block in `saxpyWarp_lockstepRunnable` and in establishing
the `laneState.pc = pc` precondition of `stepInstr?_*_lane_full`. -/
lemma saxpyWarp_lane_pc (n : Nat) (lane : LaneId)
    (ls : LaneState) (h : (saxpyWarpFor n).getLane? lane = some ls) :
    ls.pc = ("saxpyKernel", 0) := by
  unfold saxpyWarpFor WarpState.getLane? at h
  simp at h
  rw [← h]

/-- **Lockstep (initial state):** `lockstepRunnable` holds for `saxpyWarpFor n`.
All 32 lanes are at the same PC (`("saxpyKernel", 0)`), so any subset of
runnable lanes trivially satisfies the lockstep condition. -/
lemma saxpyWarp_lockstepRunnable (n : Nat) :
    Helpers.lockstepRunnable (saxpyWarpFor n) := by
  unfold Helpers.lockstepRunnable Helpers.lockstepRunnable?
  split
  · rfl
  · rename_i pc hpc
    simp [List.all_eq_true]
    intro lane hlane
    split
    · rename_i ls hLane
      have h1 := saxpyWarp_lane_pc n lane ls hLane
      unfold currentRunnablePc? at hpc
      cases hlst : runnableLaneIds (saxpyWarpFor n) with
      | nil => simp [hlst] at hpc
      | cons hd tl =>
        simp [hlst, Option.bind] at hpc
        split at hpc
        · simp at hpc
        · rename_i hdLs hHdLane
          have h2 := saxpyWarp_lane_pc n hd hdLs hHdLane
          simp at hpc
          subst hpc
          simp [h1, h2]
    · rename_i hNone
      exfalso
      unfold runnableLaneIds at hlane
      simp [List.mem_filter] at hlane
      obtain ⟨_, hRun⟩ := hlane
      unfold laneIsRunnable at hRun
      rw [hNone] at hRun
      simp at hRun

/-- **Warp extraction (initial state):** the warp at `(cta=0, warp=0)` of
`saxpyStateFor n α xs ys` is exactly `saxpyWarpFor n` — independent of the
data values `α, xs, ys`. -/
lemma saxpyStateFor_getWarp (n : Nat) (alpha : Int) (xs ys : List Int) :
    (saxpyStateFor n alpha xs ys).getWarp? 0 0 = some (saxpyWarpFor n) := by
  unfold saxpyStateFor State.getWarp? State.getCTA?
  simp

/-- **Well-formedness (initial state):** `saxpyWarpFor n` has 32 lanes. -/
lemma saxpyWarp_wf (n : Nat) : WarpState.wf (saxpyWarpFor n) := by
  unfold WarpState.wf WarpState.wf? saxpyWarpFor
  simp [Array.size_replicate]

end CLean
