import CLean.WP

namespace CLean
namespace Examples

open WP

example
    {st st' : State} {r : CSL.Resource} {dst : RegName} {rhs : RValue}
    {post : Post}
    (hstep :
      Helpers.stepInstr? st 0 0 { guard? := none, instr := .assignReg dst rhs } = some st')
    (hpost : post st' r) :
    wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs } post st r :=
  wp_assignReg_of_computed hstep (CSL.Resource.update_refl r) hpost

example
    {st st' : State} {r r' : CSL.Resource} {src : TypedAddr} {dst : RegName}
    {post : Post}
    (hstep :
      Helpers.stepInstr? st 0 0 { guard? := none, instr := .load dst src } = some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr 0 0 { guard? := none, instr := .load dst src } post st r :=
  wp_load_of_computed hstep hupdate hpost

example
    {st st' : State} {r r' : CSL.Resource} {dst : TypedAddr} {value : RValue}
    {post : Post}
    (hstep :
      Helpers.stepInstr? st 0 0 { guard? := none, instr := .store dst value } = some st')
    (hupdate : CSL.Resource.Update r r')
    (hpost : post st' r') :
    wpInstr 0 0 { guard? := none, instr := .store dst value } post st r :=
  wp_store_of_computed hstep hupdate hpost

example
    {gi : GInstr} {post post' : Post}
    (hpost : post ⊢ₛ post') :
    wpInstr 0 0 gi post ⊢ₛ wpInstr 0 0 gi post' :=
  wpInstr_mono hpost

example
    {gi : GInstr} {post : Post} :
    (wpInstr 0 0 gi post ∗ CSL.emp) ⊢ₛ wpInstr 0 0 gi (post ∗ CSL.emp) :=
  wpInstr_frame CSL.stable_emp

example (a b : CSL.Assertion) :
    CSL.sepList [a, b] ⊢ₛ CSL.sepList [b, a] :=
  CSL.sepList_swap_head a b []

example {xs ys : List CSL.Assertion} (hperm : xs.Perm ys) :
    CSL.sepList xs ⊢ₛ CSL.sepList ys :=
  CSL.sepList_perm hperm

example
    {st : State} {r : CSL.Resource} {p : State → Prop}
    (h : stateProp p st r) :
    p st :=
  stateProp_state h

example
    {st₀ st₁ : State} {p q : State → Prop}
    (hq : q st₁) :
    StateResourceUpdate st₀ st₁ (stateProp p) (stateProp q) :=
  StateResourceUpdate.stateProp hq

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hset : st.setLane cta warp lane laneState = some st') :
    st'.kernelEnv = st.kernelEnv :=
  State.setLane_kernelEnv_eq hset

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : Helpers.applyToLaneIds? st cta warp lanes f = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.applyToLaneIds?_kernelEnv_eq happly

example
    {st st' : State} {addr : Addr} {bytes : ByteMem}
    (hset : Helpers.setSpaceBaseMem? st addr bytes = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.setSpaceBaseMem?_kernelEnv_eq hset

example
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hstore : Helpers.stepStoreLanes? st cta warp lanes dst value = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.stepStoreLanes?_kernelEnv_eq hstore

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState}
    {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hstore : Helpers.stepStoreLanes? st cta warp lanes dst value = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        Helpers.runnableLaneIds warpState' = Helpers.runnableLaneIds warpState :=
  Helpers.stepStoreLanes?_runnableLaneIds_eq hwarp hstore

example
    {st st' : State} {space : AddrSpace} {ty : ScalarTy} {addr : Addr} {value : Value}
    (hwrite : Helpers.writeMem? st space ty addr value = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.writeMem?_kernelEnv_eq hwrite

example
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.advanceRunnablePcs?_kernelEnv_eq hadvance

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator}
    (hterm : Helpers.stepTerminator? st cta warp term = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.stepTerminator?_kernelEnv_eq hterm

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hordinary : Helpers.instrUsesOrdinaryPcAdvance gi.instr = true)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    st'.kernelEnv = st.kernelEnv :=
  Helpers.stepInstr?_kernelEnv_eq_of_ordinary hordinary hstep

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    (hordinary : Helpers.instrUsesOrdinaryPcAdvance gi.instr = true)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    ∃ stCore,
      Helpers.advanceRunnablePcs? stCore cta warp = some st' ∧
        stCore.kernelEnv = st.kernelEnv :=
  Helpers.stepInstr?_ordinary_factors_advance hordinary hstep

example :
    Helpers.laneIds.Nodup :=
  Helpers.laneIds_nodup

example (warpState : WarpState) :
    (Helpers.runnableLaneIds warpState).Nodup :=
  Helpers.runnableLaneIds_nodup warpState

example
    {warpState : WarpState} {pc : PC}
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc) :
    Helpers.participatingRunnableLaneIds? warpState none =
      some (Helpers.runnableLaneIds warpState) :=
  Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep hlock hrpc

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {warpState : WarpState} {pc : PC}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.status = old.status ∧ new.pc = old.pc)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (happly : Helpers.applyToLaneIds? st cta warp lanes f = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        Helpers.lockstepRunnable warpState' ∧
        Helpers.RunnablePc warpState' pc :=
  Helpers.applyToLaneIds?_warp_control_eq hpres hwarp hlock hrpc happly

example
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        Helpers.lockstepRunnable warpState' ∧
        Helpers.RunnablePc warpState' (pc.1, pc.2 + 1) :=
  Helpers.advanceRunnablePcs?_warp_control hwarp hlock hrpc hadvance

example
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {warpState : WarpState} {pc : PC}
    {lanes : List LaneId} {dst : TypedAddr} {value : RValue}
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hstep : Helpers.stepStoreLanes? st cta warp lanes dst value = some st') :
    ∃ warpState',
      st'.getWarp? cta warp = some warpState' ∧
        Helpers.lockstepRunnable warpState' ∧
        Helpers.RunnablePc warpState' pc :=
  Helpers.stepStoreLanes?_warp_control_eq hwarp hlock hrpc hstep

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {warpState : WarpState} {pc : PC}
    (hordinary : Helpers.instrUsesOrdinaryPcAdvance gi.instr = true)
    (hwarp : st.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState pc)
    (hstep : Helpers.stepInstr? st cta warp gi = some st') :
    ∃ stCore warpCore,
      Helpers.advanceRunnablePcs? stCore cta warp = some st' ∧
        stCore.kernelEnv = st.kernelEnv ∧
        stCore.getWarp? cta warp = some warpCore ∧
        Helpers.lockstepRunnable warpCore ∧
        Helpers.RunnablePc warpCore pc :=
  Helpers.stepInstr?_ordinary_core_control hordinary hwarp hlock hrpc hstep

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants post) :
    StepBlockPreserves cta warp (cfgSuffixInvariant env cta warp invariants post) :=
  StepBlockPreserves.of_cfgSuffixInvariant hbody hterm

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcfg : CFGBodyUsesOrdinaryPcAdvance env)
    (hordinary : OrdinaryBodyStepControl env cta warp) :
    BodyStepControl env cta warp :=
  BodyStepControl.of_ordinary_cfg hcfg hordinary

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcore : OrdinaryCoreStepControl env cta warp) :
    OrdinaryBodyStepControl env cta warp :=
  OrdinaryBodyStepControl.of_core hcore

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} :
    OrdinaryCoreStepControl env cta warp :=
  OrdinaryCoreStepControl.of_semantics

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} :
    OrdinaryBodyStepControl env cta warp :=
  OrdinaryBodyStepControl.of_semantics

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (hcfg : CFGBodyUsesOrdinaryPcAdvance env) :
    BodyStepControl env cta warp :=
  BodyStepControl.of_ordinary_cfg_semantics hcfg

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants post) :
    StepPreserves (cfgSuffixInvariant env cta warp invariants post) :=
  StepPreserves.of_cfgSuffixInvariant honly hbody hterm

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgSuffixInvariant env cta warp invariants spec.post)
    (hpre : spec.pre spec.init spec.resource)
    (hpreInv :
      spec.pre ⊢ₛ cfgSuffixInvariant env cta warp invariants spec.post)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesCFG env cta warp invariants spec.post)
    (hfinal : Finalizes (cfgSuffixInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_cfg_controls hinvariant hpre hpreInv honly hbody hterm hfinal

example
    {st : State} {r : CSL.Resource} {pc : PC} {lanes : List LaneId}
    (h : warpAt 0 0 pc lanes st r) :
    ∃ warpState,
      st.getWarp? 0 0 = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState pc ∧
        Helpers.ParticipatingRunnable warpState none lanes :=
  warpAt_state h

example
    {st : State} {r : CSL.Resource} {pc : PC} {lanes : List LaneId}
    (h : warpAt 0 0 pc lanes st r) :
    warpParticipants 0 0 none lanes st r :=
  warpAt_participants_none h

example
    {st : State} {r : CSL.Resource} {lane : LaneId} {pc : PC}
    (h : laneTerminatedAt 0 0 lane pc st r) :
    ∃ laneState,
      st.getLane? 0 0 lane = some laneState ∧
        laneState.status = .terminated ∧
        laneState.pc = pc :=
  laneTerminatedAt_state h

example
    (lane : LaneId) (lanes : List LaneId) (name : RegName)
    (value : Value) (values : List Value) :
    regsFor 0 0 (lane :: lanes) name (value :: values) =
      (CSL.reg 0 0 lane name value ∗ regsFor 0 0 lanes name values) :=
  regsFor_cons 0 0 lane lanes name value values

example
    {st₀ st₁ : State} {lanes : List LaneId} {name : RegName}
    {oldValues newValues : List Value}
    (hfacts : RegsUpdateFacts st₁ 0 0 name lanes newValues) :
    StateResourceUpdate st₀ st₁
      (regsFor 0 0 lanes name oldValues)
      (regsFor 0 0 lanes name newValues) :=
  StateResourceUpdate.regsFor hfacts

example
    {st : State} {r : CSL.Resource} {lanes : List LaneId} {name : RegName}
    {values : List Value}
    (hregs : regsFor 0 0 lanes name values st r) :
    RegsUpdateFacts st 0 0 name lanes values :=
  RegsUpdateFacts.of_regsFor hregs

example
    {st : State} {r : CSL.Resource} {lanes : List LaneId} {name : RegName}
    {values : List Value}
    (hregs : regsFor 0 0 lanes name values st r) :
    EvalRValuesFor st 0 0 (.reg name) lanes values :=
  EvalRValuesFor.of_regsFor hregs

example
    (lane : LaneId) (lanes : List LaneId) (name : PredName)
    (value : Bool) (values : List Bool) :
    predsFor 0 0 (lane :: lanes) name (value :: values) =
      (CSL.pred 0 0 lane name value ∗ predsFor 0 0 lanes name values) :=
  predsFor_cons 0 0 lane lanes name value values

example
    {st₀ st₁ : State} {lanes : List LaneId} {name : PredName}
    {oldValues newValues : List Bool}
    (hfacts : PredsUpdateFacts st₁ 0 0 name lanes newValues) :
    StateResourceUpdate st₀ st₁
      (predsFor 0 0 lanes name oldValues)
      (predsFor 0 0 lanes name newValues) :=
  StateResourceUpdate.predsFor hfacts

example
    {st : State} {r : CSL.Resource} {lanes : List LaneId} {name : PredName}
    {values : List Bool}
    (hpreds : predsFor 0 0 lanes name values st r) :
    PredsUpdateFacts st 0 0 name lanes values :=
  PredsUpdateFacts.of_predsFor hpreds

example
    (offset : Nat) (offsets : List Nat) (perm : CSL.BytePerm)
    (bytes : List Byte) (rest : List (List Byte)) :
    globalSlices (offset :: offsets) perm (bytes :: rest) =
      (CSL.globalBytes offset perm bytes ∗ globalSlices offsets perm rest) :=
  globalSlices_cons offset offsets perm bytes rest

example
    {st₀ st₁ : State} {offsets : List Nat} {perm : CSL.BytePerm}
    {oldSlices newSlices : List (List Byte)}
    (hfacts : GlobalSlicesUpdateFacts st₁ offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (globalSlices offsets perm oldSlices)
      (globalSlices offsets perm newSlices) :=
  StateResourceUpdate.globalSlices hfacts

example
    (offset : Nat) (offsets : List Nat) (perm : CSL.BytePerm)
    (bytes : List Byte) (rest : List (List Byte)) :
    sharedSlices 0 (offset :: offsets) perm (bytes :: rest) =
      (CSL.sharedBytes 0 offset perm bytes ∗ sharedSlices 0 offsets perm rest) :=
  sharedSlices_cons 0 offset offsets perm bytes rest

example
    {st₀ st₁ : State} {offsets : List Nat} {perm : CSL.BytePerm}
    {oldSlices newSlices : List (List Byte)}
    (hfacts : SharedSlicesUpdateFacts st₁ 0 offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (sharedSlices 0 offsets perm oldSlices)
      (sharedSlices 0 offsets perm newSlices) :=
  StateResourceUpdate.sharedSlices hfacts

example
    {st₀ st₁ : State} {ps qs : List CSL.Assertion}
    (hupdates : StateResourceUpdates st₀ st₁ ps qs) :
    StateResourceUpdate st₀ st₁ (CSL.sepList ps) (CSL.sepList qs) :=
  StateResourceUpdate.sepList hupdates

example
    {st : State} {r : CSL.Resource} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)}
    (hslices : globalSlices offsets perm slices st r) :
    GlobalSlicesUpdateFacts st offsets slices slices :=
  GlobalSlicesUpdateFacts.of_globalSlices hslices

example
    {term : Terminator} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder (TerminatorStep 0 0 term) (globalSlices offsets perm slices) :=
  stable_globalSlices_terminator

example
    {dst : RegName} {rhs : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_assignReg

example
    {dst name : RegName} {rhs : RValue} {lane : LaneId} {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_assignReg_of_ne hne

example
    {dst name : RegName} {rhs : RValue} {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_assignReg_of_ne hne

example
    {dst : RegName} {src : TypedAddr} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst src })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_load

example
    {dst : PredName} {cmp : CmpExpr} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_assignPred

example
    {dst : PredName} {cmp : CmpExpr} {lane : LaneId}
    {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_assignPred

example
    {dst : PredName} {cmp : CmpExpr}
    {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_assignPred

example
    {dst : PredName} {rhs : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_assignPredValue

example
    {dst : PredName} {rhs : RValue} {lane : LaneId}
    {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_assignPredValue

example
    {dst : PredName} {rhs : RValue}
    {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_assignPredValue

example
    {dst : RegName} {space : AddrSpace} {src : RValue}
    {offsets : List Nat} {perm : CSL.BytePerm} {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .cvta dst space src })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_cvta

example
    {dst : PredName} {space : AddrSpace} {src : RValue}
    {offsets : List Nat} {perm : CSL.BytePerm} {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .isspacep dst space src })
      (globalSlices offsets perm slices) :=
  stable_globalSlices_isspacep

example
    {dst name : RegName} {src : TypedAddr} {lane : LaneId} {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst src })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_load_of_ne hne

example
    {dst name : RegName} {space : AddrSpace} {src : RValue} {lane : LaneId}
    {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .cvta dst space src })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_cvta_of_ne hne

example
    {dst : PredName} {space : AddrSpace} {src : RValue} {lane : LaneId}
    {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .isspacep dst space src })
      (CSL.reg 0 0 lane name value) :=
  stable_reg_isspacep

example
    {dst name : RegName} {src : TypedAddr} {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst src })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_load_of_ne hne

example
    {dst name : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .cvta dst space src })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_cvta_of_ne hne

example
    {dst : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .isspacep dst space src })
      (regsFor 0 0 lanes name values) :=
  stable_regsFor_isspacep

example
    {dst : RegName} {rhs : RValue} {lane : LaneId} {name : PredName}
    {value : Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_assignReg

example
    {dst : RegName} {rhs : RValue} {lanes : List LaneId} {name : PredName}
    {values : List Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_assignReg

example
    {dst : RegName} {src : TypedAddr} {lane : LaneId} {name : PredName}
    {value : Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst src })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_load

example
    {dst : RegName} {src : TypedAddr} {lanes : List LaneId} {name : PredName}
    {values : List Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst src })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_load

example
    {dst name : PredName} {cmp : CmpExpr} {lane : LaneId} {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_assignPred_of_ne hne

example
    {dst name : PredName} {cmp : CmpExpr} {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_assignPred_of_ne hne

example
    {dst name : PredName} {rhs : RValue} {lane : LaneId} {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_assignPredValue_of_ne hne

example
    {dst name : PredName} {rhs : RValue} {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_assignPredValue_of_ne hne

example
    {dst : RegName} {space : AddrSpace} {src : RValue} {lane : LaneId}
    {name : PredName} {value : Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .cvta dst space src })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_cvta

example
    {dst : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {name : PredName} {values : List Bool} :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .cvta dst space src })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_cvta

example
    {dst name : PredName} {space : AddrSpace} {src : RValue} {lane : LaneId}
    {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .isspacep dst space src })
      (CSL.pred 0 0 lane name value) :=
  stable_pred_isspacep_of_ne hne

example
    {dst name : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .isspacep dst space src })
      (predsFor 0 0 lanes name values) :=
  stable_predsFor_isspacep_of_ne hne

example
    (lane : LaneId) (lanes : List LaneId) (offset : Nat) (offsets : List Nat)
    (perm : CSL.BytePerm) (bytes : List Byte) (rest : List (List Byte)) :
    localSlices 0 0 (lane :: lanes) (offset :: offsets) perm (bytes :: rest) =
      (CSL.localBytes 0 0 lane offset perm bytes ∗
        localSlices 0 0 lanes offsets perm rest) :=
  localSlices_cons 0 0 lane lanes offset offsets perm bytes rest

example
    {st₀ st₁ : State} {lanes : List LaneId} {offsets : List Nat}
    {perm : CSL.BytePerm} {oldSlices newSlices : List (List Byte)}
    (hfacts : LocalSlicesUpdateFacts st₁ 0 0 lanes offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (localSlices 0 0 lanes offsets perm oldSlices)
      (localSlices 0 0 lanes offsets perm newSlices) :=
  StateResourceUpdate.localSlices hfacts

example
    {term : Terminator} {lanes : List LaneId} {offsets : List Nat}
    {perm : CSL.BytePerm} {slices : List (List Byte)} :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (localSlices 0 0 lanes offsets perm slices) :=
  stable_localSlices_terminator

example (offset : Nat) (perm : CSL.BytePerm) :
    CSL.globalBytes offset perm [] = CSL.emp := by
  rfl

example (key : CSL.ResourceKey) (cell : CSL.Cell) (resource : CSL.Resource) :
    CSL.Resource.contains key cell (CSL.Resource.insert key cell resource) := by
  simp [CSL.Resource.contains, CSL.Resource.insert]

example (resource : CSL.Resource) :
    CSL.Resource.Update resource resource :=
  CSL.Resource.update_refl resource

example {owned owned' frame : CSL.Resource}
    (hupdate : CSL.Resource.Update owned owned') :
    CSL.Resource.Update
      (CSL.Resource.compose owned frame)
      (CSL.Resource.compose owned' frame) :=
  CSL.Resource.update_compose_right hupdate

example (key : CSL.ResourceKey) (old new : Value) :
    CSL.Resource.Update
      (CSL.Resource.singleton key (.reg old))
      (CSL.Resource.singleton key (.reg new)) :=
  CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape])

example (key : CSL.ResourceKey) (old new : Value) :
    CSL.resourceUpdate (CSL.owns key (.reg old)) (CSL.owns key (.reg new)) :=
  CSL.owns_resourceUpdate (by simp [CSL.Cell.sameShape])

example (offset : Nat) (old new : Byte) :
    CSL.Resource.Update
      (CSL.Resource.singleton (.globalByte offset) (.byte .write old))
      (CSL.Resource.singleton (.globalByte offset) (.byte .write new)) :=
  CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape])

example (key₁ key₂ : CSL.ResourceKey) (old₁ new₁ old₂ new₂ : Value) :
    CSL.resourceUpdate
      (CSL.owns key₁ (.reg old₁) ∗ CSL.owns key₂ (.reg old₂))
      (CSL.owns key₁ (.reg new₁) ∗ CSL.owns key₂ (.reg new₂)) :=
  CSL.sep_resourceUpdate
    (CSL.owns_resourceUpdate (by simp [CSL.Cell.sameShape]))
    (CSL.owns_resourceUpdate (by simp [CSL.Cell.sameShape]))

example
    {st st' : State} {dst : RegName} {rhs : RValue}
    {lane : LaneId} {old new : Value} {post : Post}
    (hstep :
      Helpers.stepInstr? st 0 0 { guard? := none, instr := .assignReg dst rhs } = some st')
    (hpost :
      post st'
        (CSL.Resource.singleton (.reg 0 0 lane dst) (.reg new))) :
    wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs } post st
      (CSL.Resource.singleton (.reg 0 0 lane dst) (.reg old)) :=
  wp_assignReg_of_computed hstep
    (CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape]))
    hpost

example
    {st₀ st₁ : State} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs } =
        some st₁) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg old) st r)
      (fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg new) st r) := by
  intro st r st' hpre hstep'
  rcases hpre with ⟨rfl, howns⟩
  rw [hstep] at hstep'
  injection hstep' with h
  subst st'
  subst r
  exact ⟨CSL.Resource.singleton (.reg 0 0 lane dst) (.reg new),
    CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape]), ⟨rfl, rfl⟩⟩

example
    {st₀ st₁ : State} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs } =
        some st₁) :
    (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg old) st r) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg new) st r) :=
  wpInstr_of_spec (by
    intro st r st' hpre hstep'
    rcases hpre with ⟨rfl, howns⟩
    rw [hstep] at hstep'
    injection hstep' with h
    subst st'
    subst r
    exact ⟨CSL.Resource.singleton (.reg 0 0 lane dst) (.reg new),
      CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape]), ⟨rfl, rfl⟩⟩)

example
    {st₀ st₁ : State} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs } =
        some st₁) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg old) st r)
      (fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg new) st r) :=
  assignRegSpec_of_computed hstep
    (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape]))

example
    {st₀ st₁ : State} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs } =
        some st₁) :
    (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg old) st r) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg new) st r) :=
  wpInstr_of_spec <| assignRegSpec_of_computed hstep
    (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape]))

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {rhs : RValue} {lane : LaneId} {laneState : LaneState}
    {old new : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } rhs new) :
    InstrSpec 0 0 { guard? := guard?, instr := .assignReg dst rhs }
      (fun st r => st = st₀ ∧ CSL.reg 0 0 lane dst old st r)
      (CSL.reg 0 0 lane dst new) :=
  assignRegSpec_single_of_eval hwarp hlock hpart hlane heval

example
    {pc nextPc : PC} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs new)
    (hcontrol :
      ∀ st st',
        warpAt 0 0 pc [lane] st CSL.Resource.empty →
          Helpers.stepInstr? st 0 0 { guard? := none, instr := .assignReg dst rhs } = some st' →
            warpAt 0 0 nextPc [lane] st' CSL.Resource.empty) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old)
      (warpAt 0 0 nextPc [lane] ∗ CSL.reg 0 0 lane dst new) :=
  assignRegSpec_single_warpAt_of_eval heval hcontrol

example
    {pc : PC} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs new) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.reg 0 0 lane dst new) :=
  assignRegSpec_single_warpAt heval

example
    {pc : PC} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs new) :
    (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.reg 0 0 lane dst new) :=
  wp_assignReg_single_warpAt heval

example
    {pc : PC} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.reg 0 0 lane dst old ∗ frame)) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs new)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs }) frame) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane dst old ∗ frame)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane dst new ∗ frame)) :=
  wp_assignReg_single_warpAt_stableFrame heval hframe

example
    {pc : PC} {dst : RegName} {rhs : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues) st r →
          EvalRValuesFor st 0 0 rhs lanes newValues) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗ regsFor 0 0 lanes dst newValues) :=
  assignRegSpec_lanes_warpAt hevals

example
    {pc : PC} {dst : RegName} {rhs : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues) st r →
          EvalRValuesFor st 0 0 rhs lanes newValues) :
    (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗ regsFor 0 0 lanes dst newValues) :=
  wp_assignReg_lanes_warpAt hevals

example
    {pc : PC} {dst : RegName} {rhs : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (regsFor 0 0 lanes dst oldValues ∗ frame)) st r →
          EvalRValuesFor st 0 0 rhs lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt 0 0 pc lanes ∗
          (regsFor 0 0 lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st 0 0 { guard? := none, instr := .assignReg dst rhs } =
          some st' →
        frame st' rFrame) :
    (warpAt 0 0 pc lanes ∗
      (regsFor 0 0 lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignReg dst rhs }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (regsFor 0 0 lanes dst newValues ∗ frame)) :=
  wp_assignReg_lanes_warpAt_frame hevals hframe

example
    {pc : PC} {lane : LaneId} {oldProd : Value} {alpha x prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "prod" oldProd ∗ CSL.reg 0 0 lane "x" (.s32 x))) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .assignReg "prod"
            (.binop .mul (.imm (.s32 alpha)) (.reg "x")) }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            CSL.reg 0 0 lane "x" (.s32 x))) :=
  wp_assignReg_single_warpAt_stableFrame
    (frame := CSL.reg 0 0 lane "x" (.s32 x))
    (by
      intro st r hpre
      rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
      rcases hrest with ⟨_rDst, _rSrc, _hcompRest, _hequivRest, _hdst, hsrc⟩
      exact eval_binop_of_eval eval_imm (eval_reg_of_assertion hsrc) hmul)
    (stable_reg_assignReg_of_ne (by decide))

example
    {pc : PC} {lane : LaneId} {oldSum : Value} {prod y sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "sum" oldSum ∗
        (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          CSL.reg 0 0 lane "y" (.s32 y)))) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .assignReg "sum" (.binop .add (.reg "prod") (.reg "y")) }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              CSL.reg 0 0 lane "y" (.s32 y)))) :=
  wp_assignReg_single_warpAt_stableFrame
    (frame :=
      CSL.reg 0 0 lane "prod" (.s32 prod) ∗
        CSL.reg 0 0 lane "y" (.s32 y))
    (by
      intro st r hpre
      rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
      rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
      rcases hframe with ⟨_rProd, _rY, _hcompSrc, _hequivSrc, hprod, hy⟩
      exact eval_binop_of_eval (eval_reg_of_assertion hprod) (eval_reg_of_assertion hy) hadd)
    (CSL.stable_sep
      (stable_reg_assignReg_of_ne (by decide))
      (stable_reg_assignReg_of_ne (by decide)))

example
    {pc : PC} {dst : RegName} {rhs : RValue} {lane : LaneId}
    {old new : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs new) :
    InstrSpec 0 0 { guard? := none, instr := .assignReg dst rhs }
      (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old)
      (laneRunning 0 0 lane (pc.1, pc.2 + 1) ∗ CSL.reg 0 0 lane dst new) :=
  assignRegSpec_single_laneRunning_of_eval heval

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : PredName} {cmp : CmpExpr} {lane : LaneId} {laneState : LaneState}
    {old new : Bool}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (heval : EvalCmp st₀ { cta := 0, warp := 0, lane := lane } cmp new) :
    InstrSpec 0 0 { guard? := guard?, instr := .assignPred dst cmp }
      (fun st r => st = st₀ ∧ CSL.pred 0 0 lane dst old st r)
      (CSL.pred 0 0 lane dst new) :=
  assignPredSpec_single_of_eval hwarp hlock hpart hlane heval

example
    {pc : PC} {dst : PredName} {cmp : CmpExpr} {lane : LaneId}
    {old new : Bool}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old) st r →
          EvalCmp st { cta := 0, warp := 0, lane := lane } cmp new) :
    InstrSpec 0 0 { guard? := none, instr := .assignPred dst cmp }
      (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.pred 0 0 lane dst new) :=
  assignPredSpec_single_warpAt heval

example
    {pc : PC} {dst : PredName} {cmp : CmpExpr} {lane : LaneId}
    {old new : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.pred 0 0 lane dst old ∗ frame)) st r →
          EvalCmp st { cta := 0, warp := 0, lane := lane } cmp new)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp }) frame) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane dst old ∗ frame)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignPred dst cmp }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred 0 0 lane dst new ∗ frame)) :=
  wp_assignPred_single_warpAt_stableFrame heval hframe

example
    {pc : PC} {dst : PredName} {cmp : CmpExpr} {lane : LaneId}
    {old new : Bool} {srcReg : RegName} {srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.pred 0 0 lane dst old ∗ CSL.reg 0 0 lane srcReg srcValue)) st r →
          EvalCmp st { cta := 0, warp := 0, lane := lane } cmp new) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane dst old ∗ CSL.reg 0 0 lane srcReg srcValue)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignPred dst cmp }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred 0 0 lane dst new ∗ CSL.reg 0 0 lane srcReg srcValue)) :=
  wp_assignPred_single_warpAt_stableFrame
    (frame := CSL.reg 0 0 lane srcReg srcValue)
    heval
    stable_reg_assignPred

example
    {pc : PC} {dst : PredName} {cmp : CmpExpr}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues) st r →
          EvalCmpsFor st 0 0 cmp lanes newValues) :
    InstrSpec 0 0 { guard? := none, instr := .assignPred dst cmp }
      (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗ predsFor 0 0 lanes dst newValues) :=
  assignPredSpec_lanes_warpAt hevals

example
    {pc : PC} {dst : PredName} {cmp : CmpExpr}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues) st r →
          EvalCmpsFor st 0 0 cmp lanes newValues) :
    (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignPred dst cmp }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗ predsFor 0 0 lanes dst newValues) :=
  wp_assignPred_lanes_warpAt hevals

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : PredName} {rhs : RValue} {lane : LaneId} {laneState : LaneState}
    {old new : Bool} {value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new) :
    InstrSpec 0 0 { guard? := guard?, instr := .assignPredValue dst rhs }
      (fun st r => st = st₀ ∧ CSL.pred 0 0 lane dst old st r)
      (CSL.pred 0 0 lane dst new) :=
  assignPredValueSpec_single_of_eval hwarp hlock hpart hlane heval hbool

example
    {pc : PC} {dst : PredName} {rhs : RValue} {lane : LaneId}
    {old new : Bool} {value : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new) :
    InstrSpec 0 0 { guard? := none, instr := .assignPredValue dst rhs }
      (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.pred 0 0 lane dst new) :=
  assignPredValueSpec_single_warpAt heval hbool

example
    {pc : PC} {dst : PredName} {rhs : RValue} {lane : LaneId}
    {old new : Bool} {value : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.pred 0 0 lane dst old ∗ frame)) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0 { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane dst old ∗ frame)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignPredValue dst rhs }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred 0 0 lane dst new ∗ frame)) :=
  wp_assignPredValue_single_warpAt_stableFrame heval hbool hframe

example
    {pc : PC} {dst : PredName} {srcReg : RegName} {lane : LaneId}
    {old new : Bool} {srcValue : Value}
    (hbool : Helpers.valueToBool? srcValue = some new) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane dst old ∗ CSL.reg 0 0 lane srcReg srcValue)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .assignPredValue dst (.reg srcReg) }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred 0 0 lane dst new ∗ CSL.reg 0 0 lane srcReg srcValue)) :=
  wp_assignPredValue_single_warpAt_stableFrame
    (frame := CSL.reg 0 0 lane srcReg srcValue)
    (by
      intro st r hpre
      rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
      rcases hrest with ⟨_rPred, _rSrc, _hcompRest, _hequivRest, _hpred, hsrc⟩
      exact eval_reg_of_assertion hsrc)
    hbool
    stable_reg_assignPredValue

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {laneState : LaneState} {old new srcValue : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    InstrSpec 0 0 { guard? := guard?, instr := .cvta dst space src }
      (fun st r => st = st₀ ∧ CSL.reg 0 0 lane dst old st r)
      (CSL.reg 0 0 lane dst new) :=
  cvtaSpec_single_of_eval hwarp hlock hpart hlane heval hcvta

example
    {pc : PC} {dst : RegName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {old new srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    InstrSpec 0 0 { guard? := none, instr := .cvta dst space src }
      (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.reg 0 0 lane dst new) :=
  cvtaSpec_single_warpAt heval hcvta

example
    {pc : PC} {dst : RegName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {old new srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    (warpAt 0 0 pc [lane] ∗ CSL.reg 0 0 lane dst old) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .cvta dst space src }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.reg 0 0 lane dst new) :=
  wp_cvta_single_warpAt heval hcvta

example
    {pc : PC} {dst srcReg : RegName} {space : AddrSpace}
    {lane : LaneId} {old new srcValue : Value}
    (hne : srcReg ≠ dst)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane dst old ∗ CSL.reg 0 0 lane srcReg srcValue)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .cvta dst space (.reg srcReg) }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane dst new ∗ CSL.reg 0 0 lane srcReg srcValue)) :=
  wp_cvta_single_warpAt_stableFrame
    (frame := CSL.reg 0 0 lane srcReg srcValue)
    (by
      intro st r hpre
      rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
      rcases hrest with ⟨_rDst, _rSrc, _hcompRest, _hequivRest, _hdst, hsrc⟩
      exact eval_reg_of_assertion hsrc)
    hcvta
    (stable_reg_cvta_of_ne hne)

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : PredName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {laneState : LaneState} {old new : Bool} {srcValue : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    InstrSpec 0 0 { guard? := guard?, instr := .isspacep dst space src }
      (fun st r => st = st₀ ∧ CSL.pred 0 0 lane dst old st r)
      (CSL.pred 0 0 lane dst new) :=
  isspacepSpec_single_of_eval hwarp hlock hpart hlane heval hisspace

example
    {pc : PC} {dst : PredName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {old new : Bool} {srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    InstrSpec 0 0 { guard? := none, instr := .isspacep dst space src }
      (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.pred 0 0 lane dst new) :=
  isspacepSpec_single_warpAt heval hisspace

example
    {pc : PC} {dst : PredName} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {old new : Bool} {srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    (warpAt 0 0 pc [lane] ∗ CSL.pred 0 0 lane dst old) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .isspacep dst space src }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗ CSL.pred 0 0 lane dst new) :=
  wp_isspacep_single_warpAt heval hisspace

example
    {pc : PC} {dst : PredName} {srcReg : RegName} {space : AddrSpace}
    {lane : LaneId} {old new : Bool} {srcValue : Value}
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane dst old ∗ CSL.reg 0 0 lane srcReg srcValue)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .isspacep dst space (.reg srcReg) }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred 0 0 lane dst new ∗ CSL.reg 0 0 lane srcReg srcValue)) :=
  wp_isspacep_single_warpAt_stableFrame
    (frame := CSL.reg 0 0 lane srcReg srcValue)
    (by
      intro st r hpre
      rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
      rcases hrest with ⟨_rPred, _rSrc, _hcompRest, _hequivRest, _hpred, hsrc⟩
      exact eval_reg_of_assertion hsrc)
    hisspace
    stable_reg_isspacep

example
    {st : State} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {lanes : List LaneId} {value : Value} {values : List Value}
    {srcValue : Value}
    (heval : EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some value)
    (hrest : EvalCvtaValuesFor st 0 0 space src lanes values) :
    EvalCvtaValuesFor st 0 0 space src (lane :: lanes) (value :: values) :=
  ⟨⟨srcValue, heval, hcvta⟩, hrest⟩

example
    {st : State} {space : AddrSpace} {src : RValue}
    {lane : LaneId} {lanes : List LaneId} {value : Bool} {values : List Bool}
    {srcValue : Value}
    (heval : EvalRValue st { cta := 0, warp := 0, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some value)
    (hrest : EvalIsspacepValuesFor st 0 0 space src lanes values) :
    EvalIsspacepValuesFor st 0 0 space src (lane :: lanes) (value :: values) :=
  ⟨⟨srcValue, heval, hisspace⟩, hrest⟩

example
    {stEval st stCore : State} {dst : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Value}
    (hnodup : lanes.Nodup)
    (hevals : EvalCvtaValuesFor stEval 0 0 space src lanes values)
    (happly :
      Helpers.applyToLaneIds? st 0 0 lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval 0 0 lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) = some stCore) :
    RegsUpdateFacts stCore 0 0 dst lanes values :=
  RegsUpdateFacts.of_applyCvta hnodup hevals happly

example
    {stEval st stCore : State} {dst : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalIsspacepValuesFor stEval 0 0 space src lanes values)
    (happly :
      Helpers.applyToLaneIds? st 0 0 lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval 0 0 lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) = some stCore) :
    PredsUpdateFacts stCore 0 0 dst lanes values :=
  PredsUpdateFacts.of_applyIsspacep hnodup hevals happly

example
    {pc : PC} {dst : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues) st r →
          EvalCvtaValuesFor st 0 0 space src lanes newValues) :
    InstrSpec 0 0 { guard? := none, instr := .cvta dst space src }
      (warpAt 0 0 pc lanes ∗ regsFor 0 0 lanes dst oldValues)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        regsFor 0 0 lanes dst newValues) :=
  cvtaSpec_lanes_warpAt hevals

example
    {pc : PC} {dst srcReg : RegName} {space : AddrSpace}
    {lanes : List LaneId} {oldValues newValues srcValues : List Value}
    (hne : srcReg ≠ dst)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (regsFor 0 0 lanes dst oldValues ∗ regsFor 0 0 lanes srcReg srcValues)) st r →
          EvalCvtaValuesFor st 0 0 space (.reg srcReg) lanes newValues) :
    (warpAt 0 0 pc lanes ∗
      (regsFor 0 0 lanes dst oldValues ∗ regsFor 0 0 lanes srcReg srcValues)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .cvta dst space (.reg srcReg) }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (regsFor 0 0 lanes dst newValues ∗ regsFor 0 0 lanes srcReg srcValues)) :=
  wp_cvta_lanes_warpAt_stableFrame
    (frame := regsFor 0 0 lanes srcReg srcValues)
    hevals
    (stable_regsFor_cvta_of_ne hne)

example
    {pc : PC} {dst : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues) st r →
          EvalIsspacepValuesFor st 0 0 space src lanes newValues) :
    InstrSpec 0 0 { guard? := none, instr := .isspacep dst space src }
      (warpAt 0 0 pc lanes ∗ predsFor 0 0 lanes dst oldValues)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        predsFor 0 0 lanes dst newValues) :=
  isspacepSpec_lanes_warpAt hevals

example
    {pc : PC} {dst : PredName} {srcReg : RegName} {space : AddrSpace}
    {lanes : List LaneId} {oldValues newValues : List Bool} {srcValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (predsFor 0 0 lanes dst oldValues ∗ regsFor 0 0 lanes srcReg srcValues)) st r →
          EvalIsspacepValuesFor st 0 0 space (.reg srcReg) lanes newValues) :
    (warpAt 0 0 pc lanes ∗
      (predsFor 0 0 lanes dst oldValues ∗ regsFor 0 0 lanes srcReg srcValues)) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .isspacep dst space (.reg srcReg) }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (predsFor 0 0 lanes dst newValues ∗ regsFor 0 0 lanes srcReg srcValues)) :=
  wp_isspacep_lanes_warpAt_stableFrame
    (frame := regsFor 0 0 lanes srcReg srcValues)
    hevals
    stable_regsFor_isspacep

example
    {barrierId : Nat} {p q : State → Prop}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st 0 0 { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st') :
    stateProp p ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .barrierCTA barrierId } (stateProp q) :=
  wp_barrierCTA_stateProp hstate

example
    {barrierId : Nat} {p q : State → Prop} {frame : CSL.Assertion}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st 0 0 { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st')
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0 { guard? := none, instr := .barrierCTA barrierId }) frame) :
    (stateProp p ∗ frame) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .barrierCTA barrierId }
        (stateProp q ∗ frame) :=
  wp_barrierCTA_stateProp_frame hstate hframe

example
    {barrierId : Nat} {pc targetPc : PC} {lanes : List LaneId}
    (houtcome :
      ∀ st st',
        warpAt 0 0 pc lanes st CSL.Resource.empty →
          Helpers.stepInstr? st 0 0 { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          warpAt 0 0 targetPc lanes st' CSL.Resource.empty) :
    warpAt 0 0 pc lanes ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .barrierCTA barrierId }
        (warpAt 0 0 targetPc lanes) :=
  wp_barrierCTA_warpAt_outcome houtcome

example
    {st₀ st₁ st₂ : State} {dst : RegName} {rhs₁ rhs₂ : RValue} {lane : LaneId}
    {v₀ v₁ v₂ : Value}
    (hstep₁ :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs₁ } =
        some st₁)
    (hstep₂ :
      Helpers.stepInstr? st₁ 0 0 { guard? := none, instr := .assignReg dst rhs₂ } =
        some st₂) :
    InstrSpecs 0 0
      [{ guard? := none, instr := .assignReg dst rhs₁ },
       { guard? := none, instr := .assignReg dst rhs₂ }]
      (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₀) st r)
      (fun st r => st = st₂ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₂) st r) := by
  refine InstrSpecs.cons
    (mid := fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₁) st r)
    ?_ ?_
  · exact assignRegSpec_of_computed hstep₁
      (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape]))
  · refine InstrSpecs.cons ?_ InstrSpecs.nil
    exact assignRegSpec_of_computed hstep₂
      (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape]))

example
    {st₀ st₁ st₂ : State} {dst : RegName} {rhs₁ rhs₂ : RValue} {lane : LaneId}
    {v₀ v₁ v₂ : Value}
    (hstep₁ :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .assignReg dst rhs₁ } =
        some st₁)
    (hstep₂ :
      Helpers.stepInstr? st₁ 0 0 { guard? := none, instr := .assignReg dst rhs₂ } =
        some st₂) :
    (fun st r => st = st₀ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₀) st r) ⊢ₛ
      wpInstrList 0 0
        [{ guard? := none, instr := .assignReg dst rhs₁ },
         { guard? := none, instr := .assignReg dst rhs₂ }]
        (fun st r => st = st₂ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₂) st r) :=
  wpInstrList_of_specs (by
    refine InstrSpecs.cons
      (mid := fun st r => st = st₁ ∧ CSL.owns (.reg 0 0 lane dst) (.reg v₁) st r)
      ?_ ?_
    · exact assignRegSpec_of_computed hstep₁
        (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape]))
    · refine InstrSpecs.cons ?_ InstrSpecs.nil
      exact assignRegSpec_of_computed hstep₂
        (StateResourceUpdate.owns (by simp [CSL.Cell.sameShape])))

example
    {block : Block} {pre post : CSL.Assertion}
    (hspec : ConcreteBlockSpec 0 0 block pre post) :
    pre ⊢ₛ wpConcreteBlock 0 0 block post :=
  wpConcreteBlock_of_spec hspec

example
    {block : Block} {post frame : CSL.Assertion}
    (hbody :
      ∀ gi, gi ∈ block.body.toList → CSL.StableUnder (InstrStep 0 0 gi) frame)
    (hterm : CSL.StableUnder (TerminatorStep 0 0 block.term) frame) :
    (wpConcreteBlock 0 0 block post ∗ frame) ⊢ₛ
      wpConcreteBlock 0 0 block (post ∗ frame) :=
  wpConcreteBlock_frame hbody hterm

example
    {invariants : InvariantMap} {post : CSL.Assertion} {target : BlockLabel} :
    blockTermPost invariants post (.br target) = invariants target :=
  rfl

example
    {inv : CSL.Assertion}
    (hpres : StepBlockPreserves 0 0 inv) :
    StepWarpPreserves 0 0 inv :=
  StepWarpPreserves.of_stepBlockPreserves hpres

example
    {inv : CSL.Assertion}
    (hno : NoStepBlock 0 0 inv) :
    StepBlockPreserves 0 0 inv :=
  StepBlockPreserves.of_no_step hno

example
    {inv : CSL.Assertion}
    (honly : ∀ {st st' : State}, StepMachine st st' → StepWarp st 0 0 st')
    (hpres : StepWarpPreserves 0 0 inv) :
    StepPreserves inv :=
  StepPreserves.of_stepWarpPreserves honly hpres

example
    {body : Array GInstr} {idx : Nat} {gi : GInstr}
    (hget : body[idx]? = some gi) :
    ∃ rest, body.toList.drop idx = gi :: rest :=
  Array.toList_drop_eq_cons_of_getElem?_some hget

example
    {body : Array GInstr} {idx : Nat}
    (hget : body[idx]? = none) :
    body.toList.drop idx = [] :=
  Array.toList_drop_eq_nil_of_getElem?_none hget

example
    {invariants : InvariantMap} {post : CSL.Assertion} {cond : RValue}
    {tLabel fLabel : BlockLabel} :
    blockTermPost invariants post (.cbr cond tLabel fLabel) =
      (fun st r => invariants tLabel st r ∧ invariants fLabel st r) :=
  rfl

example
    {choices : CbrChoiceMap} {invariants : InvariantMap} {post : CSL.Assertion}
    {label tLabel fLabel : BlockLabel} {cond : RValue}
    (hchoice : choices label = some true) :
    blockTermPostChoice choices invariants post label (.cbr cond tLabel fLabel) =
      invariants tLabel := by
  simp [blockTermPostChoice, hchoice]

example
    {choices : CbrChoiceMap} {invariants : InvariantMap} {post : CSL.Assertion}
    {label tLabel fLabel : BlockLabel} {cond : RValue}
    (hchoice : choices label = some false) :
    blockTermPostChoice choices invariants post label (.cbr cond tLabel fLabel) =
      invariants fLabel := by
  simp [blockTermPostChoice, hchoice]

example
    {invariants : InvariantMap} {post : CSL.Assertion} :
    blockTermPost invariants post .terminate = post :=
  rfl

example
    {invariants : InvariantMap} {post : CSL.Assertion} {block : Block} :
    blockEntryWP 0 0 invariants post block =
      wpConcreteBlock 0 0 block (blockTermPost invariants post block.term) :=
  blockEntryWP_eq_wpConcreteBlock 0 0 invariants post block

example
    {choices : CbrChoiceMap} {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block} :
    blockEntryWPChoice 0 0 choices invariants post label block =
      wpConcreteBlock 0 0 block
        (blockTermPostChoice choices invariants post label block.term) :=
  blockEntryWPChoice_eq_wpConcreteBlock 0 0 choices invariants post label block

example
    {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block}
    (hvc : blockVC 0 0 invariants post label block) :
    invariants label ⊢ₛ blockSuffixWP 0 0 invariants post block 0 :=
  blockVC.entry_suffix hvc

example
    {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block}
    (hwp :
      invariants label ⊢ₛ
        wpConcreteBlock 0 0 block (blockTermPost invariants post block.term)) :
    blockVC 0 0 invariants post label block :=
  blockVC.of_wpConcreteBlock hwp

example
    {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block}
    (hspec :
      ConcreteBlockSpec 0 0 block (invariants label)
        (blockTermPost invariants post block.term)) :
    blockVC 0 0 invariants post label block :=
  blockVC.of_concreteBlockSpec hspec

example
    {choices : CbrChoiceMap} {invariants : InvariantMap} {post : CSL.Assertion}
    {label : BlockLabel} {block : Block}
    (hspec :
      ConcreteBlockSpec 0 0 block (invariants label)
        (blockTermPostChoice choices invariants post label block.term)) :
    blockVCChoice 0 0 choices invariants post label block :=
  blockVCChoice.of_concreteBlockSpec hspec

example
    {env : KernelEnv} {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs env 0 0 pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgSuffixInvariant env 0 0 invariants post st r :=
  cfgSuffixInvariant.of_entry hvc henv hwarp hlock hrpc hentry hpre

example
    {env : KernelEnv} {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCs env 0 0 pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgKernelInvariant env 0 0 invariants post st r :=
  cfgKernelInvariant.of_entry hvc henv hwarp hlock hrpc hentry hpre

example
    {env : KernelEnv} {choices : CbrChoiceMap}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    {st : State} {r : CSL.Resource} {warpState : WarpState} {block : Block}
    (hvc : kernelVCsChoice env 0 0 choices pre post invariants)
    (henv : st.kernelEnv = env)
    (hwarp : st.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hrpc : Helpers.RunnablePc warpState (env.entry, 0))
    (hentry : env.blocks[env.entry]? = some block)
    (hpre : pre st r) :
    cfgKernelInvariantChoice env 0 0 choices invariants post st r :=
  cfgKernelInvariantChoice.of_entry hvc henv hwarp hlock hrpc hentry hpre

example
    {env : KernelEnv} {pre : CSL.Assertion} {st : State} {r : CSL.Resource}
    (hready : EntryReady env 0 0 pre)
    (hpre : pre st r) :
    ∃ warpState block,
      st.kernelEnv = env ∧
        st.getWarp? 0 0 = some warpState ∧
        Helpers.lockstepRunnable warpState ∧
        Helpers.RunnablePc warpState (env.entry, 0) ∧
        env.blocks[env.entry]? = some block :=
  hready hpre

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hsuffix : Finalizes (cfgSuffixInvariant env cta warp invariants post) post) :
    Finalizes (cfgKernelInvariant env cta warp invariants post) post :=
  Finalizes.of_cfgKernelInvariant hsuffix

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (hsuffix :
      Finalizes (cfgSuffixInvariantChoice env cta warp choices invariants post) post) :
    Finalizes (cfgKernelInvariantChoice env cta warp choices invariants post) post :=
  Finalizes.of_cfgKernelInvariantChoice hsuffix

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_cfgKernelInvariant honly hbody hterm hpost

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion}
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernelChoice env cta warp choices invariants post)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariantChoice env cta warp choices invariants post) :=
  StepPreserves.of_cfgKernelInvariantChoice honly hbody hterm hpost

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_kernelVCs hvc honly hbody hbr hcbr hpost

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCsChoice env cta warp choices pre post invariants)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariantChoice env cta warp choices invariants post) :=
  StepPreserves.of_choiceKernelVCs hvc honly hbody hbr hcbr hpost

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpost : StepBlockPreserves cta warp post) :
    StepPreserves (cfgKernelInvariant env cta warp invariants post) :=
  StepPreserves.of_kernelVCs_targets hvc honly hbody htargets hbr hcbr hpost

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrTermControl env cta warp) :
    TermStepPreservesKernel env cta warp invariants post :=
  TermStepPreservesKernel.of_kernelVCs hvc hbr hcbr

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCsChoice env cta warp choices pre post invariants)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices) :
    TermStepPreservesKernelChoice env cta warp choices invariants post :=
  TermStepPreservesKernelChoice.of_kernelVCs hvc hbr hcbr

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (htargets : CFGTerminatorTargetsExist env)
    (hsemantic : BrSemanticControl env cta warp) :
    BrTermControl env cta warp :=
  BrTermControl.of_targets_semantic htargets hsemantic

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    (htargets : CFGTerminatorTargetsExist env)
    (hsemantic : CbrSemanticControl env cta warp) :
    CbrTermControl env cta warp :=
  CbrTermControl.of_targets_semantic htargets hsemantic

example
    {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {pre post : CSL.Assertion} {invariants : InvariantMap}
    (hvc : kernelVCs env cta warp pre post invariants)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp) :
    TermStepPreservesKernel env cta warp invariants post :=
  TermStepPreservesKernel.of_kernelVCs_targets hvc htargets hbr hcbr

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hvc : kernelVCs env cta warp spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hterm : TermStepPreservesKernel env cta warp invariants spec.post)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_kernelVCs
    hinvariant hvc hentryReady hpre honly hbody hterm hpost hfinal

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {choices : CbrChoiceMap} {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariantChoice env cta warp choices invariants spec.post)
    (hvc : kernelVCsChoice env cta warp choices spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (hbr : BrTermControl env cta warp)
    (hcbr : CbrChoiceTermControl env cta warp choices)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal :
      Finalizes (cfgKernelInvariantChoice env cta warp choices invariants spec.post)
        spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_choiceKernelVCs
    hinvariant hvc hentryReady hpre honly hbody hbr hcbr hpost hfinal

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hvc : kernelVCs env cta warp spec.pre spec.post invariants)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
    (hbody : BodyStepControl env cta warp)
    (htargets : CFGTerminatorTargetsExist env)
    (hbr : BrSemanticControl env cta warp)
    (hcbr : CbrSemanticControl env cta warp)
    (hpost : StepBlockPreserves cta warp spec.post)
    (hfinal : Finalizes (cfgKernelInvariant env cta warp invariants spec.post) spec.post) :
    spec.Valid :=
  KernelSpec.Valid.of_kernelVCs_targets
    hinvariant hvc hentryReady hpre honly hbody htargets hbr hcbr hpost hfinal

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
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
  KernelSpec.Valid.of_entry_blockVCs_targets
    hinvariant hpreEntry hentryReady hpre honly hbody hblocks htargets hbr hcbr hpost hfinal

example
    {spec : KernelSpec} {env : KernelEnv} {cta : CTAId} {warp : WarpId}
    {invariants : InvariantMap}
    (hinvariant :
      spec.invariant = cfgKernelInvariant env cta warp invariants spec.post)
    (hpreEntry : spec.pre ⊢ₛ invariants env.entry)
    (hentryReady : EntryReady env cta warp spec.pre)
    (hpre : spec.pre spec.init spec.resource)
    (honly :
      ∀ {st st' : State}, StepMachine st st' → StepWarp st cta warp st')
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
  KernelSpec.Valid.of_entry_blockVCs_targets_closed
    hinvariant hpreEntry hentryReady hpre honly hbody hblocks htargets hbr hcbr
    hpostNoStep hsuffixNoFinal

example
    {invariants : InvariantMap} {post : CSL.Assertion}
    {block : Block} {idx : Nat} {gi : GInstr} {rest : List GInstr}
    {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = gi :: rest)
    (hwp : blockSuffixWP 0 0 invariants post block idx st r)
    (hstep : Helpers.stepInstr? st 0 0 gi = some st') :
    ∃ r', CSL.Resource.Update r r' ∧
      wpInstrList 0 0 rest
        (wpTerminator 0 0 block.term (blockTermPost invariants post block.term)) st' r' :=
  blockSuffixWP.body_step_of_drop hdrop hwp hstep

example
    {env : KernelEnv} {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC} {block : Block}
    {gi : GInstr} {rest : List GInstr}
    (hwp : blockSuffixWP 0 0 invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = gi :: rest)
    (hblock : env.blocks[pc.1]? = some block)
    (hstep : Helpers.stepInstr? st 0 0 gi = some st')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? 0 0 = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (pc.1, pc.2 + 1)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env 0 0 invariants post st' r' :=
  cfgSuffixInvariant.body_step_of_suffix hwp hdrop hblock hstep hcontrol

example
    {invariants : InvariantMap} {post : CSL.Assertion}
    {block : Block} {idx : Nat} {st st' : State} {r : CSL.Resource}
    (hdrop : block.body.toList.drop idx = [])
    (hwp : blockSuffixWP 0 0 invariants post block idx st r)
    (hstep : Helpers.stepTerminator? st 0 0 block.term = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ blockTermPost invariants post block.term st' r' :=
  blockSuffixWP.term_step_of_drop hdrop hwp hstep

example
    {env : KernelEnv} {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc targetPc : PC}
    {block targetBlock : Block}
    (hwp : blockSuffixWP 0 0 invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st 0 0 block.term = some st')
    (hblockTarget : env.blocks[targetPc.1]? = some targetBlock)
    (hpostToTarget :
      ∀ r',
        blockTermPost invariants post block.term st' r' →
          blockSuffixWP 0 0 invariants post targetBlock targetPc.2 st' r')
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? 0 0 = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' targetPc) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env 0 0 invariants post st' r' :=
  cfgSuffixInvariant.term_step_to_suffix_of_post hwp hdrop hstep hblockTarget
    hpostToTarget hcontrol

example
    {env : KernelEnv} {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {target : BlockLabel}
    (hterm : block.term = .br target)
    (hwp : blockSuffixWP 0 0 invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st 0 0 block.term = some st')
    (hblockTarget : env.blocks[target]? = some targetBlock)
    (hvcTarget : blockVC 0 0 invariants post target targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? 0 0 = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (target, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env 0 0 invariants post st' r' :=
  cfgSuffixInvariant.br_step_of_suffix hterm hwp hdrop hstep hblockTarget hvcTarget
    hcontrol

example
    {env : KernelEnv} {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {cond : RValue} {tLabel fLabel : BlockLabel}
    (hterm : block.term = .cbr cond tLabel fLabel)
    (hwp : blockSuffixWP 0 0 invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st 0 0 block.term = some st')
    (hblockTarget : env.blocks[tLabel]? = some targetBlock)
    (hvcTarget : blockVC 0 0 invariants post tLabel targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? 0 0 = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (tLabel, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env 0 0 invariants post st' r' :=
  cfgSuffixInvariant.cbr_true_step_of_suffix hterm hwp hdrop hstep hblockTarget
    hvcTarget hcontrol

example
    {env : KernelEnv} {invariants : InvariantMap} {post : CSL.Assertion}
    {st st' : State} {r : CSL.Resource} {pc : PC}
    {block targetBlock : Block} {cond : RValue} {tLabel fLabel : BlockLabel}
    (hterm : block.term = .cbr cond tLabel fLabel)
    (hwp : blockSuffixWP 0 0 invariants post block pc.2 st r)
    (hdrop : block.body.toList.drop pc.2 = [])
    (hstep : Helpers.stepTerminator? st 0 0 block.term = some st')
    (hblockTarget : env.blocks[fLabel]? = some targetBlock)
    (hvcTarget : blockVC 0 0 invariants post fLabel targetBlock)
    (hcontrol :
      ∃ warpState',
        st'.kernelEnv = env ∧
          st'.getWarp? 0 0 = some warpState' ∧
          Helpers.lockstepRunnable warpState' ∧
          Helpers.RunnablePc warpState' (fLabel, 0)) :
    ∃ r', CSL.Resource.Update r r' ∧
      cfgSuffixInvariant env 0 0 invariants post st' r' :=
  cfgSuffixInvariant.cbr_false_step_of_suffix hterm hwp hdrop hstep hblockTarget
    hvcTarget hcontrol

example
    {st₀ st₁ : State} {target : BlockLabel} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ 0 0 (.br target) = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec 0 0 (.br target)
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  brSpec_of_computed hstep hpost

example
    {pc : PC} {target : BlockLabel} {lane : LaneId} :
    TerminatorSpec 0 0 (.br target)
      (warpAt 0 0 pc [lane])
      (warpAt 0 0 (target, 0) [lane]) :=
  brSpec_single_warpAt

example
    {pc : PC} {target : BlockLabel} {lane : LaneId} :
    warpAt 0 0 pc [lane] ⊢ₛ
      wpTerminator 0 0 (.br target)
        (warpAt 0 0 (target, 0) [lane]) :=
  wp_br_single_warpAt

example
    {pc : PC} {target : BlockLabel} {lane : LaneId} {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep 0 0 (.br target)) frame) :
    (warpAt 0 0 pc [lane] ∗ frame) ⊢ₛ
      wpTerminator 0 0 (.br target)
        (warpAt 0 0 (target, 0) [lane] ∗ frame) :=
  wp_br_single_warpAt_stableFrame hframe

example
    {pc : PC} {target : BlockLabel} {lanes : List LaneId} :
    TerminatorSpec 0 0 (.br target)
      (warpAt 0 0 pc lanes)
      (warpAt 0 0 (target, 0) lanes) :=
  brSpec_lanes_warpAt

example
    {pc : PC} {target : BlockLabel} {lanes : List LaneId} :
    warpAt 0 0 pc lanes ⊢ₛ
      wpTerminator 0 0 (.br target)
        (warpAt 0 0 (target, 0) lanes) :=
  wp_br_lanes_warpAt

example
    {pc : PC} {target : BlockLabel} {lanes : List LaneId} {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep 0 0 (.br target)) frame) :
    (warpAt 0 0 pc lanes ∗ frame) ⊢ₛ
      wpTerminator 0 0 (.br target)
        (warpAt 0 0 (target, 0) lanes ∗ frame) :=
  wp_br_lanes_warpAt_stableFrame hframe

example
    {st₀ st₁ : State} {cond : RValue} {tLabel fLabel : BlockLabel}
    {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ 0 0 (.cbr cond tLabel fLabel) = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec 0 0 (.cbr cond tLabel fLabel)
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  cbrSpec_of_computed hstep hpost

example
    {pc : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lane : LaneId} {value : Value} {takeTrue : Bool}
    (heval :
      ∀ st r,
        warpAt 0 0 pc [lane] st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue) :
    TerminatorSpec 0 0 (.cbr cond tLabel fLabel)
      (warpAt 0 0 pc [lane])
      (warpAt 0 0 (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]) :=
  cbrSpec_single_warpAt heval hbool

example
    {pc : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lane : LaneId} {value : Value} {takeTrue : Bool}
    (heval :
      ∀ st r,
        warpAt 0 0 pc [lane] st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue) :
    warpAt 0 0 pc [lane] ⊢ₛ
      wpTerminator 0 0 (.cbr cond tLabel fLabel)
        (warpAt 0 0 (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]) :=
  wp_cbr_single_warpAt heval hbool

example
    {pc : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lane : LaneId} {value : Value} {takeTrue : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ frame) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue)
    (hframe : CSL.StableUnder (TerminatorStep 0 0 (.cbr cond tLabel fLabel)) frame) :
    (warpAt 0 0 pc [lane] ∗ frame) ⊢ₛ
      wpTerminator 0 0 (.cbr cond tLabel fLabel)
        (warpAt 0 0 (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane] ∗ frame) :=
  wp_cbr_single_warpAt_stableFrame heval hbool hframe

example
    {pc dest : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lanes : List LaneId}
    (hdest :
      ∀ st r,
        warpAt 0 0 pc lanes st r →
          Helpers.uniformBranchDestination? st 0 0 lanes cond tLabel fLabel = some dest) :
    TerminatorSpec 0 0 (.cbr cond tLabel fLabel)
      (warpAt 0 0 pc lanes)
      (warpAt 0 0 dest lanes) :=
  cbrSpec_lanes_warpAt hdest

example
    {pc dest : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lanes : List LaneId}
    (hdest :
      ∀ st r,
        warpAt 0 0 pc lanes st r →
          Helpers.uniformBranchDestination? st 0 0 lanes cond tLabel fLabel = some dest) :
    warpAt 0 0 pc lanes ⊢ₛ
      wpTerminator 0 0 (.cbr cond tLabel fLabel)
        (warpAt 0 0 dest lanes) :=
  wp_cbr_lanes_warpAt hdest

example
    {pc : PC} {tLabel fLabel : BlockLabel} {lane : LaneId} :
    warpAt 0 0 pc [lane] ⊢ₛ
      wpTerminator 0 0 (.cbr (.imm (.pred true)) tLabel fLabel)
        (warpAt 0 0 (tLabel, 0) [lane]) :=
  wp_cbr_lanes_warpAt (dest := (tLabel, 0)) (by
    intro st r hpre
    rfl)

example
    {pc dest : PC} {cond : RValue} {tLabel fLabel : BlockLabel}
    {lanes : List LaneId} {frame : CSL.Assertion}
    (hdest :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ frame) st r →
          Helpers.uniformBranchDestination? st 0 0 lanes cond tLabel fLabel = some dest)
    (hframe : CSL.StableUnder (TerminatorStep 0 0 (.cbr cond tLabel fLabel)) frame) :
    (warpAt 0 0 pc lanes ∗ frame) ⊢ₛ
      wpTerminator 0 0 (.cbr cond tLabel fLabel)
        (warpAt 0 0 dest lanes ∗ frame) :=
  wp_cbr_lanes_warpAt_stableFrame hdest hframe

example
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ 0 0 .terminate = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec 0 0 .terminate
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  terminateSpec_of_computed hstep hpost

example
    {pc : PC} {lane : LaneId} :
    TerminatorSpec 0 0 .terminate
      (warpAt 0 0 pc [lane])
      (laneTerminatedAt 0 0 lane pc) :=
  terminateSpec_single_warpAt

example
    {pc : PC} {lane : LaneId} :
    warpAt 0 0 pc [lane] ⊢ₛ
      wpTerminator 0 0 .terminate
        (laneTerminatedAt 0 0 lane pc) :=
  wp_terminate_single_warpAt

example
    {pc : PC} {lanes : List LaneId} :
    TerminatorSpec 0 0 .terminate
      (warpAt 0 0 pc lanes)
      (lanesTerminatedAt 0 0 lanes pc) :=
  terminateSpec_lanes_warpAt

example
    {pc : PC} {lanes : List LaneId} :
    warpAt 0 0 pc lanes ⊢ₛ
      wpTerminator 0 0 .terminate
        (lanesTerminatedAt 0 0 lanes pc) :=
  wp_terminate_lanes_warpAt

example
    {pc : PC} {lanes : List LaneId} {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep 0 0 .terminate) frame) :
    (warpAt 0 0 pc lanes ∗ frame) ⊢ₛ
      wpTerminator 0 0 .terminate
        (lanesTerminatedAt 0 0 lanes pc ∗ frame) :=
  wp_terminate_lanes_warpAt_stableFrame hframe

example
    {gi : GInstr} {pre post : CSL.Assertion} {p q : State → Prop}
    (hspec : InstrSpec 0 0 gi pre post)
    (hstate : ∀ st st', p st → Helpers.stepInstr? st 0 0 gi = some st' → q st') :
    InstrSpec 0 0 gi (stateProp p ∗ pre) (stateProp q ∗ post) :=
  InstrSpec.statePropFrame hspec hstate

example
    {term : Terminator} {pre post : CSL.Assertion} {p q : State → Prop}
    (hspec : TerminatorSpec 0 0 term pre post)
    (hstate : ∀ st st', p st → Helpers.stepTerminator? st 0 0 term = some st' → q st') :
    TerminatorSpec 0 0 term (stateProp p ∗ pre) (stateProp q ∗ post) :=
  TerminatorSpec.statePropFrame hspec hstate

example
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {dst : RegName} {old new : Value}
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some new) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane dst old)
      (CSL.reg cta warp lane dst new) :=
  StateResourceUpdate.reg hreg

example
    {st₀ st₁ : State} {offset : Nat} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem : CSL.memoryBytes st₁.global.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) :=
  StateResourceUpdate.globalBytes hlen hmem

example
    {st₀ st₁ : State} {dst : TypedAddr} {value : RValue}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem : CSL.memoryBytes st₁.global.bytes offset newBytes) :
    InstrSpec 0 0 { guard? := none, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.globalBytes offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.globalBytes offset .write newBytes st r) :=
  globalStoreBytesSpec_of_computed hstep hlen hmem

example
    {st₀ st₁ : State} {dst : TypedAddr} {value : RValue}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem : CSL.memoryBytes st₁.global.bytes offset newBytes) :
    (fun st r => st = st₀ ∧ CSL.globalBytes offset .write oldBytes st r) ⊢ₛ
      wpInstr 0 0 { guard? := none, instr := .store dst value }
        (fun st r => st = st₁ ∧ CSL.globalBytes offset .write newBytes st r) :=
  wpInstr_of_spec <| globalStoreBytesSpec_of_computed hstep hlen hmem

example
    {st₀ stCore : State} {warpState : WarpState} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := guard?, instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.globalBytes offset .write oldBytes st r)
      (CSL.globalBytes offset .write newBytes) :=
  globalStoreBytesSpec_single_of_eval hwarp hlock hpart haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        CSL.globalBytes offset .write newBytes) :=
  globalStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    (warpAt 0 0 pc [lane] ∗ CSL.globalBytes offset .write oldBytes) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.globalBytes offset .write newBytes) :=
  wp_globalStoreBytes_single_warpAt haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes offset .write oldBytes ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .write newBytes ∗ frame)) :=
  wp_globalStoreBytes_single_warpAt_stableFrame
    haddr heval hwrite hencode hlen hframe

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalSlicesUpdateFacts st' offsets oldSlices newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) :=
  globalStoreBytesSpec_lanes_warpAt_of_facts hfacts

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalSlicesUpdateFacts st' offsets oldSlices newSlices) :
    (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wp_globalStoreBytes_lanes_warpAt_of_facts hfacts

example {slices : List (List Byte)} :
    SliceLengthsEq slices slices :=
  SliceLengthsEq.refl

example
    {st : State} {r : CSL.Resource} {offsets : List Nat}
    {perm : CSL.BytePerm} {slices : List (List Byte)}
    (h : globalSlices offsets perm slices st r) :
    GlobalMemoryBytesFor st offsets slices :=
  GlobalMemoryBytesFor.of_globalSlices h

example
    {st : State} {offsets : List Nat} {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : GlobalMemoryBytesFor st offsets newSlices) :
    GlobalSlicesUpdateFacts st offsets oldSlices newSlices :=
  GlobalSlicesUpdateFacts.of_memoryBytesFor hlens hmems

example
    {ty : ScalarTy} {value : Value} {values : List Value}
    {bytes : List Byte} {rest : List (List Byte)}
    (henc : EncodedScalar ty value bytes)
    (hrest : EncodedScalarsFor ty values rest) :
    EncodedScalarsFor ty (value :: values) (bytes :: rest) :=
  ⟨henc, hrest⟩

example
    {writeOffset offset : Nat} {writeBytes bytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hhead : ByteRangesDisjoint offset bytes.length writeOffset writeBytes.length)
    (htail : ByteRangesDisjointFrom writeOffset writeBytes offsets slices) :
    ByteRangesDisjointFrom writeOffset writeBytes
      (offset :: offsets) (bytes :: slices) :=
  ⟨hhead, htail⟩

example
    {offset : Nat} {bytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hhead : ByteRangesDisjointFrom offset bytes offsets slices)
    (htail : PairwiseByteRangesDisjoint offsets slices) :
    PairwiseByteRangesDisjoint (offset :: offsets) (bytes :: slices) :=
  ⟨hhead, htail⟩

example
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    {bytes : List Byte}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value bytes) :
    CSL.memoryBytes stCore.global.bytes offset bytes :=
  WriteMemFact.global_memoryBytes_written hwrite hencode

example
    {st stCore : State} {ty : ScalarTy} {writeOffset readOffset : Nat}
    {value : Value} {writeBytes readBytes : List Byte}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset writeBytes.length)
    (hmem : CSL.memoryBytes st.global.bytes readOffset readBytes)
    (hwrite : WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    CSL.memoryBytes stCore.global.bytes readOffset readBytes :=
  WriteMemFact.global_memoryBytes_preserved_of_disjoint
    hdisjoint hmem hwrite hencode

example
    {st stCore : State} {ty : ScalarTy} {writeOffset : Nat}
    {value : Value} {writeBytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hdisjoint : ByteRangesDisjointFrom writeOffset writeBytes offsets slices)
    (hmems : GlobalMemoryBytesFor st offsets slices)
    (hwrite : WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    GlobalMemoryBytesFor stCore offsets slices :=
  GlobalMemoryBytesFor.preserve_global_write hdisjoint hmems hwrite hencode

example
    {st stCore : State} {ty : ScalarTy} {offset : Nat}
    {value : Value} {bytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value bytes)
    (hdisjoint : ByteRangesDisjointFrom offset bytes offsets slices)
    (hmems : GlobalMemoryBytesFor st offsets slices) :
    GlobalMemoryBytesFor stCore (offset :: offsets) (bytes :: slices) :=
  GlobalMemoryBytesFor.of_global_write_cons hwrite hencode hdisjoint hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalMemoryBytesFor st' offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) :=
  globalStoreBytesSpec_lanes_warpAt_of_memory hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalMemoryBytesFor st' offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wp_globalStoreBytes_lanes_warpAt_of_memory hlens hmems

example
    {offset₁ len₁ offset₂ len₂ : Nat}
    (hdisjoint : ByteRangesDisjoint offset₁ len₁ offset₂ len₂) :
    ByteRangesDisjoint offset₂ len₂ offset₁ len₁ :=
  ByteRangesDisjoint.symm hdisjoint

example
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue value : Value}
    {ctx : LaneCtx} {expr : RValue}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (heval : EvalRValue st ctx expr value) :
    EvalRValue stCore ctx expr value :=
  WriteMemFact.global_evalRValue hwrite heval

example
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {addr : TypedAddr} {resolved : Addr}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (haddr : ResolvesAddr st ctx addr resolved) :
    ResolvesAddr stCore ctx addr resolved :=
  WriteMemFact.global_resolvesAddr hwrite haddr

example
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {cta : CTAId} {warp : WarpId} {addr : TypedAddr}
    {lanes : List LaneId} {offsets : List Nat}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (haddrs : ResolvesGlobalAddrsFor st cta warp addr lanes offsets) :
    ResolvesGlobalAddrsFor stCore cta warp addr lanes offsets :=
  ResolvesGlobalAddrsFor.of_global_write hwrite haddrs

example
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {values : List Value} {slices : List (List Byte)}
    (haddrs :
      ResolvesGlobalAddrsFor st cta warp
        { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values slices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets slices)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .global, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    GlobalMemoryBytesFor stFinal offsets slices :=
  GlobalMemoryBytesFor.of_stepStoreLanes_global
    haddrs hevals hencs hdisjoint hstep

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) :=
  globalStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wp_globalStoreBytes_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {st st' : State} {rCtrl : CSL.Resource}
    (hctrl : warpAt 0 0 pc lanes st rCtrl)
    (hstep :
      Helpers.stepInstr? st 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    warpAt 0 0 (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
  globalStoreLanes_warpAt_control hctrl hstep

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗
        (globalSlices offsets .write oldSlices ∗ frame))
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        (globalSlices offsets .write newSlices ∗ frame)) :=
  globalStoreBytesSpec_lanes_warpAt_frame
    hlens haddrs hevals hencs hdisjoint hframe

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    (warpAt 0 0 pc lanes ∗
      (globalSlices offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .write newSlices ∗ frame)) :=
  wp_globalStoreBytes_lanes_warpAt_frame
    hlens haddrs hevals hencs hdisjoint hframe

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt 0 0 pc lanes ∗
      (globalSlices offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .write newSlices ∗ frame)) :=
  wp_globalStoreBytes_lanes_warpAt_stableFrame
    hlens haddrs hevals hencs hdisjoint hframe

example
    {st₀ st₁ : State} {dst : RegName} {src : TypedAddr}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.global.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec 0 0 { guard? := none, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst newReg) st r) :=
  globalLoadBytesRegSpec_of_computed hstep hmem hreg

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread : ReadMemFact st₀ .global ty (.global offset) value) :
    InstrSpec 0 0
      { guard? := guard?, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst value) st r) :=
  globalLoadBytesRegSpec_single_of_eval hwarp hlock hpart hlane haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc [lane] ∗
        (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg))
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst value)) :=
  globalLoadBytesRegSpec_single_warpAt haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg 0 0 lane dst value)) :=
  wp_globalLoadBytesReg_single_warpAt haddr hread

example
    {st : State} {cta : CTAId} {warp : WarpId} {ty : ScalarTy}
    {addrExpr : RValue} {lane : LaneId} {lanes : List LaneId}
    {offset : Nat} {offsets : List Nat}
    (haddr :
      ResolvesAddr st { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hrest :
      ResolvesGlobalAddrsFor st cta warp
        { space := .global, ty := ty, addr := addrExpr } lanes offsets) :
    ResolvesGlobalAddrsFor st cta warp
      { space := .global, ty := ty, addr := addrExpr }
      (lane :: lanes) (offset :: offsets) :=
  ⟨haddr, hrest⟩

example
    {st : State} {ty : ScalarTy} {offset : Nat} {offsets : List Nat}
    {value : Value} {values : List Value}
    (hread : ReadMemFact st .global ty (.global offset) value)
    (hrest : ReadGlobalValuesFor st ty offsets values) :
    ReadGlobalValuesFor st ty (offset :: offsets) (value :: values) :=
  ⟨hread, hrest⟩

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc lanes ∗
        (globalSlices offsets .read byteSlices ∗
          regsFor 0 0 lanes dst oldValues))
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        (globalSlices offsets .read byteSlices ∗
          regsFor 0 0 lanes dst newValues)) :=
  globalLoadBytesRegSpec_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues) :
    (warpAt 0 0 pc lanes ∗
      (globalSlices offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) :=
  wp_globalLoadBytesReg_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st 0 0
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .load dst { space := .global, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt 0 0 pc lanes ∗
      (globalSlices offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        ((warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) ∗ frame) :=
  wp_globalLoadBytesReg_lanes_warpAt_stableFrame haddrs hreads hframe

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg 0 0 lane dst oldReg) ∗ frame)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg 0 0 lane dst oldReg) ∗ frame)) st r →
          ReadMemFact st .global ty (.global offset) value)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .load dst { space := .global, ty := ty, addr := addrExpr } })
        frame) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes offset .read bytes ∗
        CSL.reg 0 0 lane dst oldReg) ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg 0 0 lane dst value) ∗ frame)) :=
  wp_globalLoadBytesReg_single_warpAt_stableFrame haddr hread hframe

example
    {st₀ st₁ : State} {cta : CTAId} {offset : Nat} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ ctaState, st₁.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset .write oldBytes)
      (CSL.sharedBytes cta offset .write newBytes) :=
  StateResourceUpdate.sharedBytes hlen hmem

example
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset .write oldBytes)
      (CSL.localBytes cta warp lane offset .write newBytes) :=
  StateResourceUpdate.localBytes hlen hmem

example
    {st₀ st₁ : State} {dst : TypedAddr} {value : RValue}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ ctaState, st₁.getCTA? 0 = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset newBytes) :
    InstrSpec 0 0 { guard? := none, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.sharedBytes 0 offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.sharedBytes 0 offset .write newBytes st r) :=
  sharedStoreBytesSpec_of_computed hstep hlen hmem

example
    {st₀ stCore : State} {warpState : WarpState} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .shared, ty := ty, addr := addrExpr } (.shared 0 offset))
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .shared ty (.shared 0 offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := guard?, instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.sharedBytes 0 offset .write oldBytes st r)
      (CSL.sharedBytes 0 offset .write newBytes) :=
  sharedStoreBytesSpec_single_of_eval hwarp hlock hpart haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.sharedBytes 0 offset .write oldBytes) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared 0 offset))
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.sharedBytes 0 offset .write oldBytes) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗ CSL.sharedBytes 0 offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .shared ty (.shared 0 offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc [lane] ∗ CSL.sharedBytes 0 offset .write oldBytes)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        CSL.sharedBytes 0 offset .write newBytes) :=
  sharedStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedSlicesUpdateFacts st' 0 offsets oldSlices newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        sharedSlices 0 offsets .write newSlices) :=
  sharedStoreBytesSpec_lanes_warpAt_of_facts hfacts

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedSlicesUpdateFacts st' 0 offsets oldSlices newSlices) :
    (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          sharedSlices 0 offsets .write newSlices) :=
  wp_sharedStoreBytes_lanes_warpAt_of_facts hfacts

example
    {st : State} {offsets : List Nat} {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : SharedMemoryBytesFor st 0 offsets newSlices) :
    SharedSlicesUpdateFacts st 0 offsets oldSlices newSlices :=
  SharedSlicesUpdateFacts.of_memoryBytesFor hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedMemoryBytesFor st' 0 offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        sharedSlices 0 offsets .write newSlices) :=
  sharedStoreBytesSpec_lanes_warpAt_of_memory hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        sharedSlices 0 offsets .write newSlices) :=
  sharedStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedMemoryBytesFor st' 0 offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          sharedSlices 0 offsets .write newSlices) :=
  wp_sharedStoreBytes_lanes_warpAt_of_memory hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ sharedSlices 0 offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          sharedSlices 0 offsets .write newSlices) :=
  wp_sharedStoreBytes_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .write oldSlices ∗ frame)) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt 0 0 pc lanes ∗
      (sharedSlices 0 offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices 0 offsets .write newSlices ∗ frame)) :=
  wp_sharedStoreBytes_lanes_warpAt_stableFrame
    hlens haddrs hevals hencs hdisjoint hframe

example
    {st₀ st₁ : State} {dst : RegName} {src : TypedAddr}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .load dst src } =
        some st₁)
    (hmem :
      ∃ ctaState, st₁.getCTA? 0 = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec 0 0 { guard? := none, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst newReg) st r) :=
  sharedLoadBytesRegSpec_of_computed hstep hmem hreg

example
    {st₀ : State} {warpState : WarpState} {ctaState : CTAState}
    {guard? : Option Guard} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hcta : st₀.getCTA? 0 = some ctaState)
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .shared, ty := ty, addr := addrExpr } (.shared 0 offset))
    (hread : ReadMemFact st₀ .shared ty (.shared 0 offset) value) :
    InstrSpec 0 0
      { guard? := guard?, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst value) st r) :=
  sharedLoadBytesRegSpec_single_of_eval hcta hwarp hlock hpart hlane haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared 0 offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .shared ty (.shared 0 offset) value) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc [lane] ∗
        (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg))
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.sharedBytes 0 offset .read bytes ∗ CSL.reg 0 0 lane dst value)) :=
  sharedLoadBytesRegSpec_single_warpAt haddr hread

example
    {st : State} {cta : CTAId} {warp : WarpId} {ty : ScalarTy}
    {addrExpr : RValue} {lane : LaneId} {lanes : List LaneId}
    {offset : Nat} {offsets : List Nat}
    (haddr :
      ResolvesAddr st { cta := cta, warp := warp, lane := lane }
        { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (hrest :
      ResolvesSharedAddrsFor st cta warp
        { space := .shared, ty := ty, addr := addrExpr } lanes offsets) :
    ResolvesSharedAddrsFor st cta warp
      { space := .shared, ty := ty, addr := addrExpr }
      (lane :: lanes) (offset :: offsets) :=
  ⟨haddr, hrest⟩

example
    {st : State} {ty : ScalarTy} {offset : Nat} {offsets : List Nat}
    {value : Value} {values : List Value}
    (hread : ReadMemFact st .shared ty (.shared 0 offset) value)
    (hrest : ReadSharedValuesFor st 0 ty offsets values) :
    ReadSharedValuesFor st 0 ty (offset :: offsets) (value :: values) :=
  ⟨hread, hrest⟩

example
    {st : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    (hread : ReadMemFact st .shared ty (.shared 0 offset) value) :
    ∃ ctaState, st.getCTA? 0 = some ctaState :=
  ReadMemFact.shared_getCTA hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadSharedValuesFor st 0 ty offsets newValues) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc lanes ∗
        (sharedSlices 0 offsets .read byteSlices ∗
          regsFor 0 0 lanes dst oldValues))
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        (sharedSlices 0 offsets .read byteSlices ∗
          regsFor 0 0 lanes dst newValues)) :=
  sharedLoadBytesRegSpec_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadSharedValuesFor st 0 ty offsets newValues) :
    (warpAt 0 0 pc lanes ∗
      (sharedSlices 0 offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) :=
  wp_sharedLoadBytesReg_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st 0 0
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadSharedValuesFor st 0 ty offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .load dst { space := .shared, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt 0 0 pc lanes ∗
      (sharedSlices 0 offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
        ((warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices 0 offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) ∗ frame) :=
  wp_sharedLoadBytesReg_lanes_warpAt_stableFrame haddrs hreads hframe

example
    {st : State} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {lanes : List LaneId} {offset : Nat} {offsets : List Nat}
    (haddr :
      ResolvesAddr st { cta := 0, warp := 0, lane := lane }
        { space := .local, ty := ty, addr := addrExpr } (.local 0 0 lane offset))
    (hrest :
      ResolvesLocalAddrsFor st 0 0
        { space := .local, ty := ty, addr := addrExpr } lanes offsets) :
    ResolvesLocalAddrsFor st 0 0
      { space := .local, ty := ty, addr := addrExpr }
      (lane :: lanes) (offset :: offsets) :=
  ⟨haddr, hrest⟩

example
    {st : State} {ty : ScalarTy} {lane : LaneId} {lanes : List LaneId}
    {offset : Nat} {offsets : List Nat} {value : Value} {values : List Value}
    (hread : ReadMemFact st .local ty (.local 0 0 lane offset) value)
    (hrest : ReadLocalValuesFor st 0 0 ty lanes offsets values) :
    ReadLocalValuesFor st 0 0 ty (lane :: lanes) (offset :: offsets)
      (value :: values) :=
  ⟨hread, hrest⟩

example
    {st : State} {ty : ScalarTy} {lane : LaneId} {offset : Nat} {value : Value}
    (hread : ReadMemFact st .local ty (.local 0 0 lane offset) value) :
    ∃ laneState, st.getLane? 0 0 lane = some laneState :=
  ReadMemFact.local_getLane hread

example
    {st : State} {lane : LaneId} {lanes : List LaneId}
    {offset : Nat} {offsets : List Nat} {bytes : List Byte} {rest : List (List Byte)}
    (hmem :
      ∃ laneState, st.getLane? 0 0 lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset bytes)
    (hrest : LocalMemoryBytesFor st 0 0 lanes offsets rest) :
    LocalMemoryBytesFor st 0 0 (lane :: lanes) (offset :: offsets) (bytes :: rest) :=
  ⟨hmem, hrest⟩

example
    {st : State} {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : LocalMemoryBytesFor st 0 0 lanes offsets newSlices) :
    LocalSlicesUpdateFacts st 0 0 lanes offsets oldSlices newSlices :=
  LocalSlicesUpdateFacts.of_memoryBytesFor hlens hmems

example
    {st st' : State} {r : CSL.Resource} {ty : ScalarTy}
    {lanes : List LaneId} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} {values : List Value}
    (hlocal :
      ∀ lane laneState, st.getLane? 0 0 lane = some laneState →
        ∃ laneState', st'.getLane? 0 0 lane = some laneState' ∧
          laneState'.localMem = laneState.localMem)
    (hslices : localSlices 0 0 lanes offsets perm slices st r)
    (hreads : ReadLocalValuesFor st 0 0 ty lanes offsets values) :
    LocalSlicesUpdateFacts st' 0 0 lanes offsets slices slices :=
  LocalSlicesUpdateFacts.of_localSlices_readValuesFor_eq hlocal hslices hreads

example
    {stEval st stCore : State} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesLocalAddrsFor stEval 0 0
        { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadLocalValuesFor stEval 0 0 ty lanes offsets values)
    (happly :
      Helpers.applyToLaneIds? st 0 0 lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval 0 0 lane
              { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .local ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore 0 0 dst lanes values :=
  RegsUpdateFacts.of_applyLocalLoad hnodup haddrs hreads happly

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadLocalValuesFor st 0 0 ty lanes offsets newValues) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc lanes ∗
        (localSlices 0 0 lanes offsets .read byteSlices ∗
          regsFor 0 0 lanes dst oldValues))
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        (localSlices 0 0 lanes offsets .read byteSlices ∗
          regsFor 0 0 lanes dst newValues)) :=
  localLoadBytesRegSpec_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadLocalValuesFor st 0 0 ty lanes offsets newValues) :
    (warpAt 0 0 pc lanes ∗
      (localSlices 0 0 lanes offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) :=
  wp_localLoadBytesReg_lanes_warpAt haddrs hreads

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst oldValues)) st r →
          ReadLocalValuesFor st 0 0 ty lanes offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .load dst { space := .local, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt 0 0 pc lanes ∗
      (localSlices 0 0 lanes offsets .read byteSlices ∗
        regsFor 0 0 lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr 0 0
        { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
        ((warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (localSlices 0 0 lanes offsets .read byteSlices ∗
            regsFor 0 0 lanes dst newValues)) ∗ frame) :=
  wp_localLoadBytesReg_lanes_warpAt_stableFrame haddrs hreads hframe

example
    {st₀ st₁ : State} {dst : TypedAddr} {value : RValue}
    {lane : LaneId} {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset newBytes) :
    InstrSpec 0 0 { guard? := none, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.localBytes 0 0 lane offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.localBytes 0 0 lane offset .write newBytes st r) :=
  localStoreBytesSpec_of_computed hstep hlen hmem

example
    {st₀ stCore : State} {warpState : WarpState} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .local, ty := ty, addr := addrExpr } (.local 0 0 lane offset))
    (heval : EvalRValue st₀ { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .local ty (.local 0 0 lane offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := guard?, instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.localBytes 0 0 lane offset .write oldBytes st r)
      (CSL.localBytes 0 0 lane offset .write newBytes) :=
  localStoreBytesSpec_single_of_eval hwarp hlock hpart haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.localBytes 0 0 lane offset .write oldBytes) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local 0 0 lane offset))
    (heval :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.localBytes 0 0 lane offset .write oldBytes) st r →
          EvalRValue st { cta := 0, warp := 0, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.localBytes 0 0 lane offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .local ty (.local 0 0 lane offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc [lane] ∗
        CSL.localBytes 0 0 lane offset .write oldBytes)
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        CSL.localBytes 0 0 lane offset .write newBytes) :=
  localStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalSlicesUpdateFacts st' 0 0 lanes offsets oldSlices newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        localSlices 0 0 lanes offsets .write newSlices) :=
  localStoreBytesSpec_lanes_warpAt_of_facts hfacts

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalSlicesUpdateFacts st' 0 0 lanes offsets oldSlices newSlices) :
    (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          localSlices 0 0 lanes offsets .write newSlices) :=
  wp_localStoreBytes_lanes_warpAt_of_facts hfacts

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalMemoryBytesFor st' 0 0 lanes offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        localSlices 0 0 lanes offsets .write newSlices) :=
  localStoreBytesSpec_lanes_warpAt_of_memory hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st 0 0
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalMemoryBytesFor st' 0 0 lanes offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          localSlices 0 0 lanes offsets .write newSlices) :=
  wp_localStoreBytes_lanes_warpAt_of_memory hlens hmems

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices) :
    InstrSpec 0 0
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices)
      (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
        localSlices 0 0 lanes offsets .write newSlices) :=
  localStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          localSlices 0 0 lanes offsets .write oldSlices) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices) :
    (warpAt 0 0 pc lanes ∗ localSlices 0 0 lanes offsets .write oldSlices) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          localSlices 0 0 lanes offsets .write newSlices) :=
  wp_localStoreBytes_lanes_warpAt hlens haddrs hevals hencs hdisjoint

example
    {pc : PC} {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .write oldSlices ∗ frame)) st r →
          ResolvesLocalAddrsFor st 0 0
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt 0 0 pc lanes ∗
          (localSlices 0 0 lanes offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st 0 0 valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep 0 0
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt 0 0 pc lanes ∗
      (localSlices 0 0 lanes offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr 0 0
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt 0 0 (pc.1, pc.2 + 1) lanes ∗
          (localSlices 0 0 lanes offsets .write newSlices ∗ frame)) :=
  wp_localStoreBytes_lanes_warpAt_stableFrame
    hlens haddrs hevals hencs hdisjoint hframe

example
    {st₀ st₁ : State} {dst : RegName} {src : TypedAddr}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .load dst src } =
        some st₁)
    (hmem :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec 0 0 { guard? := none, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.localBytes 0 0 lane offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.localBytes 0 0 lane offset .read bytes ∗ CSL.reg 0 0 lane dst newReg) st r) :=
  localLoadBytesRegSpec_of_computed hstep hmem hreg

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .local, ty := ty, addr := addrExpr } (.local 0 0 lane offset))
    (hread : ReadMemFact st₀ .local ty (.local 0 0 lane offset) value) :
    InstrSpec 0 0
      { guard? := guard?, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.localBytes 0 0 lane offset .read bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        (CSL.localBytes 0 0 lane offset .read bytes ∗ CSL.reg 0 0 lane dst value) st r) :=
  localLoadBytesRegSpec_single_of_eval hwarp hlock hpart hlane haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.localBytes 0 0 lane offset .read bytes ∗
            CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local 0 0 lane offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.localBytes 0 0 lane offset .read bytes ∗
            CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .local ty (.local 0 0 lane offset) value) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc [lane] ∗
        (CSL.localBytes 0 0 lane offset .read bytes ∗
          CSL.reg 0 0 lane dst oldReg))
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.localBytes 0 0 lane offset .read bytes ∗
          CSL.reg 0 0 lane dst value)) :=
  localLoadBytesRegSpec_single_warpAt haddr hread

example
    {st₀ st₁ : State} {offset : Nat} {bytes : List Byte}
    (hmem : CSL.memoryBytes st₁.param.bytes offset bytes) :
    StateResourceUpdate st₀ st₁
      (CSL.paramBytes offset bytes)
      (CSL.paramBytes offset bytes) :=
  StateResourceUpdate.paramBytes hmem

example
    {st₀ st₁ : State} {offset : Nat} {bytes : List Byte}
    (hmem : CSL.memoryBytes st₁.const.bytes offset bytes) :
    StateResourceUpdate st₀ st₁
      (CSL.constBytes offset bytes)
      (CSL.constBytes offset bytes) :=
  StateResourceUpdate.constBytes hmem

example
    {st₀ st₁ : State} {dst : RegName} {src : TypedAddr}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.param.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec 0 0 { guard? := none, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst newReg) st r) :=
  paramLoadBytesRegSpec_of_computed hstep hmem hreg

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .param, ty := ty, addr := addrExpr } (.param offset))
    (hread : ReadMemFact st₀ .param ty (.param offset) value) :
    InstrSpec 0 0
      { guard? := guard?, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst value) st r) :=
  paramLoadBytesRegSpec_single_of_eval hwarp hlock hpart hlane haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .param, ty := ty, addr := addrExpr } (.param offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .param ty (.param offset) value) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc [lane] ∗
        (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg))
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.paramBytes offset bytes ∗ CSL.reg 0 0 lane dst value)) :=
  paramLoadBytesRegSpec_single_warpAt haddr hread

example
    {st₀ st₁ : State} {dst : RegName} {src : TypedAddr}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ 0 0 { guard? := none, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.const.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? 0 0 lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec 0 0 { guard? := none, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst newReg) st r) :=
  constLoadBytesRegSpec_of_computed hstep hmem hreg

example
    {st₀ : State} {warpState : WarpState} {guard? : Option Guard}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? 0 0 = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? 0 0 lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := 0, warp := 0, lane := lane }
        { space := .const, ty := ty, addr := addrExpr } (.const offset))
    (hread : ReadMemFact st₀ .const ty (.const offset) value) :
    InstrSpec 0 0
      { guard? := guard?, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg) st r)
      (fun st r =>
        (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst value) st r) :=
  constLoadBytesRegSpec_single_of_eval hwarp hlock hpart hlane haddr hread

example
    {pc : PC} {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .const, ty := ty, addr := addrExpr } (.const offset))
    (hread :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg)) st r →
          ReadMemFact st .const ty (.const offset) value) :
    InstrSpec 0 0
      { guard? := none, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
      (warpAt 0 0 pc [lane] ∗
        (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst oldReg))
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.constBytes offset bytes ∗ CSL.reg 0 0 lane dst value)) :=
  constLoadBytesRegSpec_single_warpAt haddr hread

example
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .global ty (.global offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.global.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .global ty (.global offset) value :=
  globalReadMem_of_byteRead haccess hwidth hread hdecode

example
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .param ty (.param offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.param.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .param ty (.param offset) value :=
  paramReadMem_of_byteRead haccess hwidth hread hdecode

example
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .const ty (.const offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.const.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .const ty (.const offset) value :=
  constReadMem_of_byteRead haccess hwidth hread hdecode

example
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    {mem' : ByteMem}
    (haccess : AccessOk .global ty (.global offset))
    (hencode : EncodedScalar ty value bytes)
    (hwrite : ByteWrite st.global.bytes mem' offset bytes) :
    WriteMemFact st .global ty (.global offset) value { st with global := { bytes := mem' } } :=
  globalWriteMem_of_byteWrite haccess hencode hwrite

example
    {st : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {ctaState : CTAState} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .shared ty (.shared cta offset))
    (hcta : st.getCTA? cta = some ctaState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead ctaState.shared.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .shared ty (.shared cta offset) value :=
  sharedReadMem_of_byteRead haccess hcta hwidth hread hdecode

example
    {st : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {laneState : LaneState}
    {bytes : List Byte} {value : Value}
    (haccess : AccessOk .local ty (.local cta warp lane offset))
    (hlane : st.getLane? cta warp lane = some laneState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead laneState.localMem.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .local ty (.local cta warp lane offset) value :=
  localReadMem_of_byteRead haccess hlane hwidth hread hdecode

example
    {st : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {ctaState : CTAState} {bytes : List Byte} {value : Value} {mem' : ByteMem}
    (haccess : AccessOk .shared ty (.shared cta offset))
    (hencode : EncodedScalar ty value bytes)
    (hcta : st.getCTA? cta = some ctaState)
    (hwrite : ByteWrite ctaState.shared.bytes mem' offset bytes) :
    WriteMemFact st .shared ty (.shared cta offset) value
      (st.setCTA cta { ctaState with shared := { bytes := mem' } }) :=
  sharedWriteMem_of_byteWrite haccess hencode hcta hwrite

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {laneState : LaneState}
    {bytes : List Byte} {value : Value} {mem' : ByteMem}
    (haccess : AccessOk .local ty (.local cta warp lane offset))
    (hencode : EncodedScalar ty value bytes)
    (hlane : st.getLane? cta warp lane = some laneState)
    (hwrite : ByteWrite laneState.localMem.bytes mem' offset bytes)
    (hset :
      st.setLane cta warp lane { laneState with localMem := { bytes := mem' } } = some st') :
    WriteMemFact st .local ty (.local cta warp lane offset) value st' :=
  localWriteMem_of_byteWrite haccess hencode hlane hwrite hset

example {st : State} {ctx : LaneCtx} :
    ResolvesAddr st ctx
      { space := .global, ty := .u64, addr := .imm (.u64 (5 : UInt64)) }
      (.global 5) := by
  simpa using
    (resolves_global_u64_of_eval
      (st := st)
      (ctx := ctx)
      (ty := .u64)
      (expr := .imm (.u64 (5 : UInt64)))
      (off := (5 : UInt64))
      (by rfl))

example {st : State} {ctx : LaneCtx} {offset : Nat} :
    ResolvesAddr st ctx
      { space := .shared, ty := .u32, addr := .imm (.gaddr .shared offset) }
      (.shared ctx.cta offset) :=
  resolves_shared_gaddr_of_eval (by rfl)

example {st : State} {ctx : LaneCtx} {offset : Nat} :
    ResolvesAddr st ctx
      { space := .generic, ty := .u64, addr := .imm (.gaddr .local offset) }
      (.local ctx.cta ctx.warp ctx.lane offset) :=
  resolves_generic_local_of_eval (by rfl)

example
    {st : State} {ctx : LaneCtx} {r : CSL.Resource} {name : RegName} {value : Value}
    (hreg : CSL.reg ctx.cta ctx.warp ctx.lane name value st r) :
    EvalRValue st ctx (.reg name) value :=
  eval_reg_of_assertion hreg

example
    {st : State} {ctx : LaneCtx} {r : CSL.Resource} {name : PredName} {value : Bool}
    (hpred : CSL.pred ctx.cta ctx.warp ctx.lane name value st r) :
    EvalRValue st ctx (.pred name) (.pred value) :=
  eval_pred_of_assertion hpred

example
    {st : State} {ctx : LaneCtx} {r : CSL.Resource} {name : RegName} {offset : Nat}
    (hreg : CSL.reg ctx.cta ctx.warp ctx.lane name (.gaddr .local offset) st r) :
    ResolvesAddr st ctx
      { space := .generic, ty := .u64, addr := .reg name }
      (.local ctx.cta ctx.warp ctx.lane offset) :=
  resolves_generic_local_of_eval (eval_reg_of_assertion hreg)

example
    {st : State} {r : CSL.Resource} {offset : Nat} {perm : CSL.BytePerm}
    {bytes : List Byte}
    (hbytes : CSL.globalBytes offset perm bytes st r) :
    CSL.memoryBytes st.global.bytes offset bytes :=
  CSL.globalBytes_memory hbytes

example
    {st : State} {r : CSL.Resource} {offset : Nat} {perm : CSL.BytePerm}
    {bytes : List Byte}
    (hbytes : CSL.globalBytes offset perm bytes st r) :
    ByteRead st.global.bytes offset bytes :=
  byteRead_of_globalBytes hbytes

example {mem : ByteMem} {offset : Nat} {bytes : List Byte} :
    CSL.memoryBytes (Helpers.writeBytes mem offset bytes) offset bytes :=
  memoryBytes_writeBytes

example {mem mem' : ByteMem} {offset : Nat} {bytes : List Byte}
    (hwrite : ByteWrite mem mem' offset bytes) :
    CSL.memoryBytes mem' offset bytes :=
  byteWrite_memoryBytes hwrite

example
    {st : State} {r : CSL.Resource} {ty : ScalarTy} {offset : Nat}
    {bytes : List Byte} {value : Value}
    (haccess : AccessOk .global ty (.global offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.globalBytes offset .read bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .global ty (.global offset) value :=
  globalReadMem_of_globalBytes haccess hwidth hbytes hdecode

example
    {st : State} {r : CSL.Resource} {offset : Nat} {bytes : List Byte}
    (hbytes : CSL.paramBytes offset bytes st r) :
    CSL.memoryBytes st.param.bytes offset bytes :=
  CSL.paramBytes_memory hbytes

example
    {st : State} {r : CSL.Resource} {cta : CTAId} {offset : Nat}
    {perm : CSL.BytePerm} {byte : Byte} {bytes : List Byte}
    (hbytes : CSL.sharedBytes cta offset perm (byte :: bytes) st r) :
    ∃ ctaState, st.getCTA? cta = some ctaState ∧
      CSL.memoryBytes ctaState.shared.bytes offset (byte :: bytes) :=
  CSL.sharedBytes_memory_exists hbytes

example
    {st : State} {r : CSL.Resource} {cta : CTAId} {offset : Nat}
    {perm : CSL.BytePerm} {bytes : List Byte} {ctaState : CTAState}
    (hbytes : CSL.sharedBytes cta offset perm bytes st r)
    (hcta : st.getCTA? cta = some ctaState) :
    ByteRead ctaState.shared.bytes offset bytes :=
  byteRead_of_sharedBytes hbytes hcta

example
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : CSL.BytePerm} {byte : Byte} {bytes : List Byte}
    (hbytes : CSL.localBytes cta warp lane offset perm (byte :: bytes) st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      CSL.memoryBytes laneState.localMem.bytes offset (byte :: bytes) :=
  CSL.localBytes_memory_exists hbytes

example
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} {laneState : LaneState}
    (hbytes : CSL.localBytes cta warp lane offset perm bytes st r)
    (hlane : st.getLane? cta warp lane = some laneState) :
    ByteRead laneState.localMem.bytes offset bytes :=
  byteRead_of_localBytes hbytes hlane

example
    {st₀ st₁ : State} {r : CSL.Resource} {offset : Nat}
    {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hpost : CSL.globalBytes offset .write newBytes st₁ r) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) :=
  StateResourceUpdate.globalBytes hlen (CSL.globalBytes_memory hpost)

example
    {st₀ st₁ : State} {offset : Nat} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hglobal : st₁.global.bytes = Helpers.writeBytes st₀.global.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) := by
  apply StateResourceUpdate.globalBytes hlen
  rw [hglobal]
  exact memoryBytes_writeBytes

example
    {st₀ st₁ : State} {ty : ScalarTy} {offset : Nat}
    {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) :=
  StateResourceUpdate.globalBytes_of_writeMemFact hwrite hencode hlen

example
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) :=
  StateResourceUpdate.globalBytes_of_writeMemFact_advanced hwrite hadvance hencode hlen

example
    {st₀ st₁ : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .shared ty (.shared cta offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset .write oldBytes)
      (CSL.sharedBytes cta offset .write newBytes) :=
  StateResourceUpdate.sharedBytes_of_writeMemFact hwrite hencode hlen

example
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .shared ty (.shared cta offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset .write oldBytes)
      (CSL.sharedBytes cta offset .write newBytes) :=
  StateResourceUpdate.sharedBytes_of_writeMemFact_advanced hwrite hadvance hencode hlen

example
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .local ty (.local cta warp lane offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset .write oldBytes)
      (CSL.localBytes cta warp lane offset .write newBytes) :=
  StateResourceUpdate.localBytes_of_writeMemFact hwrite hencode hlen

example
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .local ty (.local cta warp lane offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset .write oldBytes)
      (CSL.localBytes cta warp lane offset .write newBytes) :=
  StateResourceUpdate.localBytes_of_writeMemFact_advanced hwrite hadvance hencode hlen

example
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : RegName} {value : Value}
    (hreg : CSL.reg cta warp lane name value st r) :
    ∃ laneState, st.getLane? cta warp lane = some laneState ∧
      laneState.regs[name]? = some value :=
  CSL.reg_state hreg

example
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : RegName} {old new : Value}
    (hwrite :
      ∃ laneState, st₁.getLane? cta warp lane = some (Helpers.writeReg laneState name new)) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane name old)
      (CSL.reg cta warp lane name new) :=
  StateResourceUpdate.reg_written hwrite

example
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : RegName} {old new : Value}
    (hwrite :
      ∃ laneState,
        stCore.getLane? cta warp lane = some (Helpers.writeReg laneState name new))
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane name old)
      (CSL.reg cta warp lane name new) :=
  StateResourceUpdate.reg_written_advanced hwrite hadvance

example
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : PredName} {old new : Bool}
    (hwrite :
      ∃ laneState, st₁.getLane? cta warp lane =
        some (Helpers.writePred laneState name new)) :
    StateResourceUpdate st₀ st₁
      (CSL.pred cta warp lane name old)
      (CSL.pred cta warp lane name new) :=
  StateResourceUpdate.pred_written hwrite

example
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {name : PredName} {old new : Bool}
    (hwrite :
      ∃ laneState,
        stCore.getLane? cta warp lane = some (Helpers.writePred laneState name new))
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁) :
    StateResourceUpdate st₀ st₁
      (CSL.pred cta warp lane name old)
      (CSL.pred cta warp lane name new) :=
  StateResourceUpdate.pred_written_advanced hwrite hadvance

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState}
    (happly : Helpers.applyToLaneIds? st cta warp lanes f = some st') :
    st'.global = st.global :=
  Helpers.applyToLaneIds?_global_eq happly

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {f : LaneId → LaneState → Option LaneState} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (happly : Helpers.applyToLaneIds? st cta warp lanes f = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
      ctaState'.shared = ctaState.shared :=
  Helpers.applyToLaneIds?_shared_eq hcta happly

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {target : LaneId} {targetState : LaneState}
    {f : LaneId → LaneState → Option LaneState}
    (hpres :
      ∀ lane old new, f lane old = some new →
        new.localMem = old.localMem ∧ new.regs = old.regs ∧ new.preds = old.preds)
    (htarget : st.getLane? cta warp target = some targetState)
    (happly : Helpers.applyToLaneIds? st cta warp lanes f = some st') :
    ∃ targetState', st'.getLane? cta warp target = some targetState' ∧
      targetState'.localMem = targetState.localMem ∧
      targetState'.regs = targetState.regs ∧
      targetState'.preds = targetState.preds :=
  Helpers.applyToLaneIds?_lane_nonPc_eq hpres htarget happly

example
    {st st' : State} {cta : CTAId} {warp : WarpId}
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    st'.global = st.global :=
  Helpers.advanceRunnablePcs?_global_eq hadvance

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {ctaState : CTAState}
    (hcta : st.getCTA? cta = some ctaState)
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
      ctaState'.shared = ctaState.shared :=
  Helpers.advanceRunnablePcs?_shared_eq hcta hadvance

example
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {laneState : LaneState}
    (hlane : st.getLane? cta warp lane = some laneState)
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
      laneState'.localMem = laneState.localMem ∧
      laneState'.regs = laneState.regs ∧
      laneState'.preds = laneState.preds :=
  Helpers.advanceRunnablePcs?_lane_nonPc_eq hlane hadvance

end Examples
end CLean
