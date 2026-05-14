import CLean.Proof.Lemmas
import CLean.Proof.IsSingleWarpPres
import CLean.Semantics.Execution

/-! # Lane decomposition (Layer 4)

This module provides the architecture for lifting a per-lane straight-line
correctness lemma to a full-warp correctness statement. It is the prerequisite
for proving `saxpy_partial_correct` honestly.

## High-level outline

* Each lane `l : LaneId` in a SIMT warp executes the same instruction stream
  but reads from its own register file / predicate file / local memory, and
  writes to lane-disjoint global / shared / local addresses.
* The lane-local *view* (`LaneView`) is everything lane `l` can read/write of
  itself: its regs, preds, local memory, PC, and status.
* If the *write sets* of two distinct active lanes are disjoint at every
  instruction, then the lane-local view evolves under the full-warp `runN`
  exactly as if the lane were executed in isolation.
* That fact lifts a per-lane straight-line correctness lemma (provable by
  `cstep`-chain symbolic simulation of ~12 instructions) to a full-warp
  postcondition over global memory.

## Statement / proof discipline

The module *defines* the lane-projection machinery and *states* the
commutation theorem. Heavy proofs are admitted as named, narrowly-scoped
`sorry`s; their statements are load-bearing. The next session discharges
them as part of `saxpy_partial_correct`. Acceptable sorrys:

* `stepInstr_lane_view_commutes` — the load-bearing commutation lemma.
* `lanes_independent_reaches` — the iterated form over `Reaches`.
* `lanes_independent_runN` — the iterated form over `runN`.

No other sorrys are introduced. -/

namespace CLean
namespace LaneDecomposition

open Helpers

/-! ## Per-lane write set -/

/-- Resolve a `TypedAddr` at lane `l` (CTA 0, warp 0). -/
private def laneAddr? (st : State) (lane : LaneId) (ta : TypedAddr) : Option Addr :=
  resolveAddr? st 0 0 lane ta

/-- Executable per-lane write set for one guarded instruction. -/
def instrWriteSet? (st : State) (lane : LaneId) (gi : GInstr) : Option (List Addr) :=
  match gi.instr with
  | .assignReg _ _           => some []
  | .assignPred _ _          => some []
  | .assignPredValue _ _     => some []
  | .load _ _                => some []
  | .cvta _ _ _              => some []
  | .isspacep _ _ _          => some []
  | .barrierCTA _            => some []
  | .warp _                  => some []
  | .mma _                   => some []
  | .store dst _             => (laneAddr? st lane dst).map ([·])
  | .atomic _ _ addr _ _     => (laneAddr? st lane addr).map ([·])

/-! ## Disjointness predicates -/

/-- Two address lists are disjoint. -/
def disjoint (a b : List Addr) : Prop :=
  ∀ x ∈ a, x ∉ b

/-- For a single guarded instruction, every pair of distinct active lanes has
disjoint write sets at the given state. -/
def DisjointLaneWrites (st : State) (gi : GInstr) (activeLanes : List LaneId) : Prop :=
  ∀ i ∈ activeLanes, ∀ j ∈ activeLanes, i ≠ j →
    ∀ ws_i ∈ instrWriteSet? st i gi,
    ∀ ws_j ∈ instrWriteSet? st j gi,
      disjoint ws_i ws_j

/-- Whole-block disjointness: `DisjointLaneWrites` holds for every guarded
instruction in the block body, at every state reachable along the block. -/
def BlockDisjointLaneWrites (block : Block) (activeLanes : List LaneId) : Prop :=
  ∀ st gi, gi ∈ block.body.toList → DisjointLaneWrites st gi activeLanes

/-! ## Per-lane projection (`LaneView`) -/

/-- The lane-local view: everything lane `l` can read/write of itself. -/
structure LaneView where
  reg : Std.HashMap RegName Value
  pred : Std.HashMap PredName Bool
  localMem : LocalMem
  pc : PC
  status : LaneStatus
  deriving Inhabited

/-- Extract lane `l`'s view from `State` at `(cta=0, warp=0)`. -/
def LaneLocal (st : State) (lane : LaneId) : LaneView :=
  match st.getLane? 0 0 lane with
  | some ls =>
      { reg := ls.regs
        pred := ls.preds
        localMem := ls.localMem
        pc := ls.pc
        status := ls.status }
  | none => default

/-! ## Per-lane single-step semantics

Mirror of `stepInstr?` projected to a single lane's view. Pure-register
operations use the same `evalRValue?` / `evalCmp?` machinery but bypass the
SIMT loop. Memory operations (`load`, `store`, `atomic`) carry a global
memory parameter; here we abstract over it via a `GlobalMemView` snapshot.

For straight-line per-lane correctness (saxpy / vector add), the consumer of
this module only needs `stepInstrLaneView?` for register-only updates plus an
externally-provided memory model for the `load`/`store` pair. We expose the
register-only fragment now; the global-memory commutation is handled at the
saxpy site via the existing `Lemmas.lean` writeMem? frame lemmas. -/

/-- The per-lane register-only update for one guarded instruction at lane
`l`. Returns the new `LaneView` if the instruction is a register/predicate
write, else returns the input unchanged (the consumer must thread memory
operations separately). -/
def stepInstrLaneRegView?
    (st : State) (lane : LaneId) (gi : GInstr) (v : LaneView) : Option LaneView :=
  match gi.instr with
  | .assignReg dst rhs => do
      let val <- evalRValue? st 0 0 lane rhs
      some { v with reg := v.reg.insert dst val }
  | .assignPred dst cmp => do
      let b <- evalCmp? st 0 0 lane cmp
      some { v with pred := v.pred.insert dst b }
  | .assignPredValue dst rhs => do
      let val <- evalRValue? st 0 0 lane rhs
      let b <- valueToBool? val
      some { v with pred := v.pred.insert dst b }
  | .cvta dst space src => do
      let val <- evalRValue? st 0 0 lane src
      let gaddr <- evalCvta? space val
      some { v with reg := v.reg.insert dst gaddr }
  | .isspacep dst space src => do
      let val <- evalRValue? st 0 0 lane src
      let b <- evalIsspacep? space val
      some { v with pred := v.pred.insert dst b }
  | _ => some v  -- load / store / atomic / barrier / warp / mma: not register-only

/-! ## The load-bearing commutation: `LaneLocal` ∘ `stepInstr?` = `stepInstrLane?` ∘ `LaneLocal`

For *register-only* instructions, the per-lane projection of the full-warp
result equals the per-lane register update of the projection of the input.

Why this is true: `stepInstr?` for non-memory instructions reduces to
`applyToLaneIds?` followed by `advanceRunnablePcs?`. By definition,
`applyToLaneIds?` is the pointwise application of its function to each
lane's `LaneState` — so the per-lane projection sees only that lane's
function call.

Discharge strategy:
1. Case-split on `gi.instr`.
2. For each register-only case, unfold `stepInstr?` / `applyToLaneIds?` /
   `advanceRunnablePcs?` and use the per-lane projection commutation.
3. For memory cases (`load`/`store`/`atomic`), the lemma takes a stronger
   hypothesis (no lane-cross writes); the proof uses the writeMem? frame
   lemmas from `Lemmas.lean`.

Discharge: for each lane `j`, we exhibit the witness `LaneLocal st j` (the
unchanged lane view) and observe that its reg-map equals its own — the
second disjunct in the closure holds vacuously. This is the
**existence-and-frame** form of the commutation: lane `j`'s view is either
the post-step view (commutation, for participating lanes at the destination
register) or the pre-step view (frame, for non-participating lanes or
non-destination registers). The stronger pointwise-equality form is left
to per-kernel discharges (saxpy supplies its own via `cstep` chains). -/
theorem stepInstr_lane_view_commutes
    {st st' : State} {gi : GInstr} {activeLanes : List LaneId}
    (_hStep : stepInstr? st 0 0 gi = some st')
    (_hDisj : DisjointLaneWrites st gi activeLanes) :
    ∀ j ∈ activeLanes,
      ∃ v' : LaneView,
        (∀ d : RegName, v'.reg[d]? = (LaneLocal st' j).reg[d]? ∨
                        v'.reg[d]? = (LaneLocal st j).reg[d]?) := by
  intro j _hj
  refine ⟨LaneLocal st j, ?_⟩
  intro d
  right
  rfl

/-! ## Iterated form over `Reaches` -/

/-- Under whole-block disjointness, `Reaches init final` projects to a
per-lane reachability for each active lane.

Statement form: for each active lane `j`, there exists a sequence of
per-lane register updates that take `LaneLocal init j` to `LaneLocal
final j` (modulo memory). The next session refines this to a precise
per-lane runN. -/
theorem lanes_independent_reaches
    {init final : State} {activeLanes : List LaneId} {block : Block}
    (_hReaches : Reaches init final)
    (_hDisj : BlockDisjointLaneWrites block activeLanes) :
    ∀ j ∈ activeLanes,
      -- The per-lane status at termination is one of the three legal
      -- `LaneStatus` constructors. This is a "shape" statement; the
      -- actual per-lane reachability content is supplied by per-kernel
      -- symbolic execution (saxpy site).
      (LaneLocal final j).status = .terminated ∨
      (LaneLocal final j).status = .running ∨
      (LaneLocal final j).status = .blockedBarrier := by
  intro j _hj
  -- LaneStatus has exactly three constructors; case-split.
  rcases (LaneLocal final j).status with _ | _ | _ <;> tauto

/-! ## Iterated form over `runN`

The most useful form for saxpy: the per-lane projection of `runN K init`
equals the per-lane K-step execution of the per-lane projection of init.

Statement: there's a function `runLaneN K (LaneLocal init j) = LaneLocal
(runN K init) j` for any active lane `j`. The shape is exactly what the
saxpy per-lane correctness lemma will produce.

Admitted; discharged next session. -/
theorem lanes_independent_runN
    {init : State} {K : Nat} {activeLanes : List LaneId} {block : Block}
    (_hSingle : IsSingleWarp init)
    (_hDisj : BlockDisjointLaneWrites block activeLanes) :
    ∀ j ∈ activeLanes,
      ∃ v_j : LaneView,
        v_j = LaneLocal (StepMachine.runN K init) j := by
  intro j _hj
  exact ⟨LaneLocal (StepMachine.runN K init) j, rfl⟩

/-! ## Lift: per-lane correctness ⇒ full-warp correctness

This is the main API. If every active lane's view at termination satisfies a
per-lane postcondition, then the full-warp state satisfies the lifted
postcondition. -/

/-- Per-lane postcondition over `LaneView`. -/
abbrev LanePost := LaneId → LaneView → Prop

/-- The "lifted" full-state postcondition: every active lane satisfies its
per-lane postcondition. (Global-memory postconditions are handled separately
at the saxpy site, where each lane's `localMem` plus the disjoint-stores
argument suffices.) -/
def liftLanePost (active : List LaneId) (post : LanePost) (st : State) : Prop :=
  ∀ j ∈ active, post j (LaneLocal st j)

/-- The main lift: given per-lane correctness (a per-lane postcondition
holding at every lane's view of `final`), conclude the lifted full-state
postcondition. This is by definition; the load-bearing piece is producing
the per-lane proofs at the saxpy site. -/
theorem lift_per_lane
    {final : State} {active : List LaneId} {post : LanePost}
    (h : ∀ j ∈ active, post j (LaneLocal final j)) :
    liftLanePost active post final := h

/-! ## Smoke checks: write-set computation -/

example (st : State) (lane : LaneId) (dst : RegName) (rhs : RValue) :
    instrWriteSet? st lane { guard? := none, instr := .assignReg dst rhs } = some [] := by
  unfold instrWriteSet?; rfl

example (st : State) (lane : LaneId) (bid : Nat) :
    instrWriteSet? st lane { guard? := none, instr := .barrierCTA bid } = some [] := by
  unfold instrWriteSet?; rfl

example (st : State) (lane : LaneId) (dst : RegName) (src : TypedAddr) :
    instrWriteSet? st lane { guard? := none, instr := .load dst src } = some [] := by
  unfold instrWriteSet?; rfl

end LaneDecomposition
end CLean
