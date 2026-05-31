import CLean.WP.Block
import CLean.WP.CFG
import CLean.WP.Control
import CLean.WP.Memory

namespace CLean
namespace WP

def InstrSpec
    (cta : CTAId) (warp : WarpId) (gi : GInstr) (pre post : CSL.Assertion) : Prop :=
  ∀ st r st',
    pre st r →
    Helpers.stepInstr? st cta warp gi = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def TerminatorSpec
    (cta : CTAId) (warp : WarpId) (term : Terminator) (pre post : CSL.Assertion) :
    Prop :=
  ∀ st r st',
    pre st r →
    Helpers.stepTerminator? st cta warp term = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def ExecutableStepSpec
    (cta : CTAId) (warp : WarpId) (pre post : CSL.Assertion) : Prop :=
  ∀ st r st',
    pre st r →
    StepMachine.stepAt? st cta warp = some st' →
      ∃ r', CSL.Resource.Update r r' ∧ post st' r'

def StateResourceUpdate (st₀ st₁ : State) (pre post : CSL.Assertion) : Prop :=
  ∀ r, pre st₀ r → ∃ r', CSL.Resource.Update r r' ∧ post st₁ r'

theorem StateResourceUpdate.owns {st₀ st₁ : State} {key : CSL.ResourceKey}
    {old new : CSL.Cell}
    (hshape : CSL.Cell.sameShape old new) :
    StateResourceUpdate st₀ st₁ (CSL.owns key old) (CSL.owns key new) := by
  intro r howns
  subst r
  exact ⟨CSL.Resource.singleton key new, CSL.Resource.update_singleton hshape, rfl⟩

theorem StateResourceUpdate.of_sameResource {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hpost : ∀ r, pre st₀ r → post st₁ r) :
    StateResourceUpdate st₀ st₁ pre post := by
  intro r hpre
  exact ⟨r, CSL.Resource.update_refl r, hpost r hpre⟩

theorem StateResourceUpdate.emp {st₀ st₁ : State} :
    StateResourceUpdate st₀ st₁ CSL.emp CSL.emp :=
  StateResourceUpdate.of_sameResource (by
    intro r hemp
    exact hemp)

theorem StateResourceUpdate.stateProp {st₀ st₁ : State} {p q : State → Prop}
    (hq : q st₁) :
    StateResourceUpdate st₀ st₁ (stateProp p) (stateProp q) := by
  intro r hpre
  rcases hpre with ⟨_hp, hemp⟩
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, ⟨hq, rfl⟩⟩

theorem StateResourceUpdate.sep {st₀ st₁ : State} {p p' q q' : CSL.Assertion}
    (hp : StateResourceUpdate st₀ st₁ p p')
    (hq : StateResourceUpdate st₀ st₁ q q') :
    StateResourceUpdate st₀ st₁ (p ∗ q) (p' ∗ q') := by
  intro r hsep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hp₁, hq₂⟩
  rcases hp r₁ hp₁ with ⟨r₁', hupdate₁, hp'⟩
  rcases hq r₂ hq₂ with ⟨r₂', hupdate₂, hq'⟩
  refine ⟨CSL.Resource.compose r₁' r₂', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose hupdate₁ hupdate₂)
  · exact ⟨r₁', r₂',
      CSL.Resource.canCompose_update_right hupdate₂
        (CSL.Resource.canCompose_update_left hupdate₁ hcomp),
      CSL.Resource.equiv_refl _, hp', hq'⟩

def StateResourceUpdates (st₀ st₁ : State) :
    List CSL.Assertion → List CSL.Assertion → Prop
  | [], [] => True
  | p :: ps, q :: qs =>
      StateResourceUpdate st₀ st₁ p q ∧ StateResourceUpdates st₀ st₁ ps qs
  | _, _ => False

theorem StateResourceUpdate.sepList {st₀ st₁ : State} :
    ∀ {ps qs : List CSL.Assertion},
      StateResourceUpdates st₀ st₁ ps qs →
        StateResourceUpdate st₀ st₁ (CSL.sepList ps) (CSL.sepList qs)
  | [], [], _ =>
      by simpa [CSL.sepList] using (StateResourceUpdate.emp (st₀ := st₀) (st₁ := st₁))
  | [], _ :: _, h => by exact False.elim h
  | _ :: _, [], h => by exact False.elim h
  | p :: [], q :: [], h => by
      simpa [StateResourceUpdates, CSL.sepList] using h.1
  | p :: [], q :: _ :: _, h => by
      exact False.elim h.2
  | p :: p' :: ps, q :: q' :: qs, h => by
      have htail :
          StateResourceUpdate st₀ st₁
            (CSL.sepList (p' :: ps)) (CSL.sepList (q' :: qs)) :=
        StateResourceUpdate.sepList h.2
      simpa [StateResourceUpdates, CSL.sepList] using StateResourceUpdate.sep h.1 htail

theorem StateResourceUpdate.of_false_pre {st₀ st₁ : State} {post : CSL.Assertion} :
    StateResourceUpdate st₀ st₁ (CSL.pure False) post := by
  intro r hpre
  exact False.elim hpre.1

def RegsUpdateFacts
    (st : State) (cta : CTAId) (warp : WarpId) (name : RegName) :
    List LaneId → List Value → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      (∃ laneState, st.getLane? cta warp lane = some laneState ∧
        laneState.regs[name]? = some value) ∧
      RegsUpdateFacts st cta warp name lanes values
  | _, _ => False

def PredsUpdateFacts
    (st : State) (cta : CTAId) (warp : WarpId) (name : PredName) :
    List LaneId → List Bool → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      (∃ laneState, st.getLane? cta warp lane = some laneState ∧
        laneState.preds[name]? = some value) ∧
      PredsUpdateFacts st cta warp name lanes values
  | _, _ => False

def GlobalSlicesUpdateFacts (st : State) :
    List Nat → List (List Byte) → List (List Byte) → Prop
  | [], [], [] => True
  | offset :: offsets, oldBytes :: oldRest, newBytes :: newRest =>
      oldBytes.length = newBytes.length ∧
      CSL.memoryBytes st.global.bytes offset newBytes ∧
      GlobalSlicesUpdateFacts st offsets oldRest newRest
  | _, _, _ => False

def ParamSlicesUpdateFacts (st : State) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      CSL.memoryBytes st.param.bytes offset bytes ∧
      ParamSlicesUpdateFacts st offsets rest
  | _, _ => False

def ConstSlicesUpdateFacts (st : State) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      CSL.memoryBytes st.const.bytes offset bytes ∧
      ConstSlicesUpdateFacts st offsets rest
  | _, _ => False

def SharedSlicesUpdateFacts (st : State) (cta : CTAId) :
    List Nat → List (List Byte) → List (List Byte) → Prop
  | [], [], [] => True
  | offset :: offsets, oldBytes :: oldRest, newBytes :: newRest =>
      oldBytes.length = newBytes.length ∧
      (∃ ctaState, st.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset newBytes) ∧
      SharedSlicesUpdateFacts st cta offsets oldRest newRest
  | _, _, _ => False

def SliceLengthsEq : List (List Byte) → List (List Byte) → Prop
  | [], [] => True
  | oldBytes :: oldRest, newBytes :: newRest =>
      oldBytes.length = newBytes.length ∧ SliceLengthsEq oldRest newRest
  | _, _ => False

def GlobalMemoryBytesFor (st : State) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      CSL.memoryBytes st.global.bytes offset bytes ∧
      GlobalMemoryBytesFor st offsets rest
  | _, _ => False

def ParamMemoryBytesFor (st : State) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      CSL.memoryBytes st.param.bytes offset bytes ∧
      ParamMemoryBytesFor st offsets rest
  | _, _ => False

def ConstMemoryBytesFor (st : State) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      CSL.memoryBytes st.const.bytes offset bytes ∧
      ConstMemoryBytesFor st offsets rest
  | _, _ => False

def SharedMemoryBytesFor (st : State) (cta : CTAId) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      (∃ ctaState, st.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset bytes) ∧
      SharedMemoryBytesFor st cta offsets rest
  | _, _ => False

def LocalMemoryBytesFor (st : State) (cta : CTAId) (warp : WarpId) :
    List LaneId → List Nat → List (List Byte) → Prop
  | [], [], [] => True
  | lane :: lanes, offset :: offsets, bytes :: rest =>
      (∃ laneState, st.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset bytes) ∧
      LocalMemoryBytesFor st cta warp lanes offsets rest
  | _, _, _ => False

def ByteRangesDisjointFrom (writeOffset : Nat) (writeBytes : List Byte) :
    List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      ByteRangesDisjoint offset bytes.length writeOffset writeBytes.length ∧
      ByteRangesDisjointFrom writeOffset writeBytes offsets rest
  | _, _ => False

def PairwiseByteRangesDisjoint : List Nat → List (List Byte) → Prop
  | [], [] => True
  | offset :: offsets, bytes :: rest =>
      ByteRangesDisjointFrom offset bytes offsets rest ∧
      PairwiseByteRangesDisjoint offsets rest
  | _, _ => False

def LocalByteRangesDisjointFrom (writeLane : LaneId) (writeOffset : Nat)
    (writeBytes : List Byte) : List LaneId → List Nat → List (List Byte) → Prop
  | [], [], [] => True
  | lane :: lanes, offset :: offsets, bytes :: rest =>
      (lane ≠ writeLane ∨
        ByteRangesDisjoint offset bytes.length writeOffset writeBytes.length) ∧
      LocalByteRangesDisjointFrom writeLane writeOffset writeBytes lanes offsets rest
  | _, _, _ => False

def PairwiseLocalByteRangesDisjoint :
    List LaneId → List Nat → List (List Byte) → Prop
  | [], [], [] => True
  | lane :: lanes, offset :: offsets, bytes :: rest =>
      LocalByteRangesDisjointFrom lane offset bytes lanes offsets rest ∧
      PairwiseLocalByteRangesDisjoint lanes offsets rest
  | _, _, _ => False

def LocalSlicesUpdateFacts
    (st : State) (cta : CTAId) (warp : WarpId) :
    List LaneId → List Nat → List (List Byte) → List (List Byte) → Prop
  | [], [], [], [] => True
  | lane :: lanes, offset :: offsets, oldBytes :: oldRest, newBytes :: newRest =>
      oldBytes.length = newBytes.length ∧
      (∃ laneState, st.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset newBytes) ∧
      LocalSlicesUpdateFacts st cta warp lanes offsets oldRest newRest
  | _, _, _, _ => False

def EvalRValuesFor
    (st : State) (cta : CTAId) (warp : WarpId) (rhs : RValue) :
    List LaneId → List Value → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value ∧
      EvalRValuesFor st cta warp rhs lanes values
  | _, _ => False

def EvalCmpsFor
    (st : State) (cta : CTAId) (warp : WarpId) (cmp : CmpExpr) :
    List LaneId → List Bool → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      EvalCmp st { cta := cta, warp := warp, lane := lane } cmp value ∧
      EvalCmpsFor st cta warp cmp lanes values
  | _, _ => False

def EvalRValueBoolsFor
    (st : State) (cta : CTAId) (warp : WarpId) (rhs : RValue) :
    List LaneId → List Bool → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      (∃ raw,
        EvalRValue st { cta := cta, warp := warp, lane := lane } rhs raw ∧
          Helpers.valueToBool? raw = some value) ∧
      EvalRValueBoolsFor st cta warp rhs lanes values
  | _, _ => False

def EvalCvtaValuesFor
    (st : State) (cta : CTAId) (warp : WarpId) (space : AddrSpace) (src : RValue) :
    List LaneId → List Value → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      (∃ srcValue,
        EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue ∧
          Helpers.evalCvta? space srcValue = some value) ∧
      EvalCvtaValuesFor st cta warp space src lanes values
  | _, _ => False

def EvalIsspacepValuesFor
    (st : State) (cta : CTAId) (warp : WarpId) (space : AddrSpace) (src : RValue) :
    List LaneId → List Bool → Prop
  | [], [] => True
  | lane :: lanes, value :: values =>
      (∃ srcValue,
        EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue ∧
          Helpers.evalIsspacep? space srcValue = some value) ∧
      EvalIsspacepValuesFor st cta warp space src lanes values
  | _, _ => False

def ResolvesGlobalAddrsFor
    (st : State) (cta : CTAId) (warp : WarpId) (addr : TypedAddr) :
    List LaneId → List Nat → Prop
  | [], [] => True
  | lane :: lanes, offset :: offsets =>
      ResolvesAddr st { cta := cta, warp := warp, lane := lane } addr (.global offset) ∧
      ResolvesGlobalAddrsFor st cta warp addr lanes offsets
  | _, _ => False

def ResolvesParamAddrsFor
    (st : State) (cta : CTAId) (warp : WarpId) (addr : TypedAddr) :
    List LaneId → List Nat → Prop
  | [], [] => True
  | lane :: lanes, offset :: offsets =>
      ResolvesAddr st { cta := cta, warp := warp, lane := lane } addr (.param offset) ∧
      ResolvesParamAddrsFor st cta warp addr lanes offsets
  | _, _ => False

def ResolvesConstAddrsFor
    (st : State) (cta : CTAId) (warp : WarpId) (addr : TypedAddr) :
    List LaneId → List Nat → Prop
  | [], [] => True
  | lane :: lanes, offset :: offsets =>
      ResolvesAddr st { cta := cta, warp := warp, lane := lane } addr (.const offset) ∧
      ResolvesConstAddrsFor st cta warp addr lanes offsets
  | _, _ => False

def ResolvesSharedAddrsFor
    (st : State) (cta : CTAId) (warp : WarpId) (addr : TypedAddr) :
    List LaneId → List Nat → Prop
  | [], [] => True
  | lane :: lanes, offset :: offsets =>
      ResolvesAddr st { cta := cta, warp := warp, lane := lane } addr (.shared cta offset) ∧
      ResolvesSharedAddrsFor st cta warp addr lanes offsets
  | _, _ => False

def ResolvesLocalAddrsFor
    (st : State) (cta : CTAId) (warp : WarpId) (addr : TypedAddr) :
    List LaneId → List Nat → Prop
  | [], [] => True
  | lane :: lanes, offset :: offsets =>
      ResolvesAddr st { cta := cta, warp := warp, lane := lane } addr
        (.local cta warp lane offset) ∧
      ResolvesLocalAddrsFor st cta warp addr lanes offsets
  | _, _ => False

def ReadGlobalValuesFor (st : State) (ty : ScalarTy) :
    List Nat → List Value → Prop
  | [], [] => True
  | offset :: offsets, value :: values =>
      ReadMemFact st .global ty (.global offset) value ∧
      ReadGlobalValuesFor st ty offsets values
  | _, _ => False

def ReadParamValuesFor (st : State) (ty : ScalarTy) :
    List Nat → List Value → Prop
  | [], [] => True
  | offset :: offsets, value :: values =>
      ReadMemFact st .param ty (.param offset) value ∧
      ReadParamValuesFor st ty offsets values
  | _, _ => False

def ReadConstValuesFor (st : State) (ty : ScalarTy) :
    List Nat → List Value → Prop
  | [], [] => True
  | offset :: offsets, value :: values =>
      ReadMemFact st .const ty (.const offset) value ∧
      ReadConstValuesFor st ty offsets values
  | _, _ => False

def ReadSharedValuesFor (st : State) (cta : CTAId) (ty : ScalarTy) :
    List Nat → List Value → Prop
  | [], [] => True
  | offset :: offsets, value :: values =>
      ReadMemFact st .shared ty (.shared cta offset) value ∧
      ReadSharedValuesFor st cta ty offsets values
  | _, _ => False

def ReadLocalValuesFor (st : State) (cta : CTAId) (warp : WarpId) (ty : ScalarTy) :
    List LaneId → List Nat → List Value → Prop
  | [], [], [] => True
  | lane :: lanes, offset :: offsets, value :: values =>
      ReadMemFact st .local ty (.local cta warp lane offset) value ∧
      ReadLocalValuesFor st cta warp ty lanes offsets values
  | _, _, _ => False

def EncodedScalarsFor (ty : ScalarTy) :
    List Value → List (List Byte) → Prop
  | [], [] => True
  | value :: values, bytes :: rest =>
      EncodedScalar ty value bytes ∧ EncodedScalarsFor ty values rest
  | _, _ => False

theorem RegsUpdateFacts.of_regsFor
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {name : RegName} {values : List Value}
    (h : regsFor cta warp lanes name values st r) :
    RegsUpdateFacts st cta warp name lanes values := by
  induction lanes generalizing values r with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim h.1
      | cons value values =>
          change
            (CSL.reg cta warp lane name value ∗
              regsFor cta warp lanes name values) st r at h
          rcases h with ⟨_rReg, rRest, _hcomp, _hequiv, hreg, hrest⟩
          exact ⟨CSL.reg_state hreg, ih hrest⟩

theorem PredsUpdateFacts.of_predsFor
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {name : PredName} {values : List Bool}
    (h : predsFor cta warp lanes name values st r) :
    PredsUpdateFacts st cta warp name lanes values := by
  induction lanes generalizing values r with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim h.1
      | cons value values =>
          change
            (CSL.pred cta warp lane name value ∗
              predsFor cta warp lanes name values) st r at h
          rcases h with ⟨_rPred, rRest, _hcomp, _hequiv, hpred, hrest⟩
          exact ⟨CSL.pred_state hpred, ih hrest⟩

theorem GlobalSlicesUpdateFacts.of_globalSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)}
    (h : globalSlices offsets perm slices st r) :
    GlobalSlicesUpdateFacts st offsets slices slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change
            (CSL.globalBytes offset perm bytes ∗ globalSlices offsets perm rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨rfl, CSL.globalBytes_memory hbytes, ih hrest⟩

theorem ParamSlicesUpdateFacts.of_paramSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat}
    {slices : List (List Byte)}
    (h : paramSlices offsets slices st r) :
    ParamSlicesUpdateFacts st offsets slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change (CSL.paramBytes offset bytes ∗ paramSlices offsets rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨CSL.paramBytes_memory hbytes, ih hrest⟩

theorem ConstSlicesUpdateFacts.of_constSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat}
    {slices : List (List Byte)}
    (h : constSlices offsets slices st r) :
    ConstSlicesUpdateFacts st offsets slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change (CSL.constBytes offset bytes ∗ constSlices offsets rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨CSL.constBytes_memory hbytes, ih hrest⟩

theorem SliceLengthsEq.refl :
    ∀ {slices : List (List Byte)}, SliceLengthsEq slices slices
  | [] => True.intro
  | _bytes :: rest => ⟨rfl, SliceLengthsEq.refl (slices := rest)⟩

theorem GlobalMemoryBytesFor.of_globalSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)}
    (h : globalSlices offsets perm slices st r) :
    GlobalMemoryBytesFor st offsets slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change
            (CSL.globalBytes offset perm bytes ∗ globalSlices offsets perm rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨CSL.globalBytes_memory hbytes, ih hrest⟩

theorem ParamMemoryBytesFor.of_paramSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat}
    {slices : List (List Byte)}
    (h : paramSlices offsets slices st r) :
    ParamMemoryBytesFor st offsets slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change (CSL.paramBytes offset bytes ∗ paramSlices offsets rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨CSL.paramBytes_memory hbytes, ih hrest⟩

theorem ConstMemoryBytesFor.of_constSlices
    {st : State} {r : CSL.Resource} {offsets : List Nat}
    {slices : List (List Byte)}
    (h : constSlices offsets slices st r) :
    ConstMemoryBytesFor st offsets slices := by
  induction offsets generalizing slices r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim h.1
      | cons bytes rest =>
          change (CSL.constBytes offset bytes ∗ constSlices offsets rest) st r at h
          rcases h with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
          exact ⟨CSL.constBytes_memory hbytes, ih hrest⟩

theorem GlobalSlicesUpdateFacts.of_memoryBytesFor
    {st : State} {offsets : List Nat} {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : GlobalMemoryBytesFor st offsets newSlices) :
    GlobalSlicesUpdateFacts st offsets oldSlices newSlices := by
  induction offsets generalizing oldSlices newSlices with
  | nil =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hmems
      | cons _ _ =>
          cases newSlices with
          | nil =>
              exact False.elim hlens
          | cons _ _ =>
              exact False.elim hmems
  | cons offset offsets ih =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact False.elim hmems
          | cons _ _ =>
              exact False.elim hlens
      | cons oldBytes oldRest =>
          cases newSlices with
          | nil =>
              cases hmems
          | cons newBytes newRest =>
              rcases hlens with ⟨hlen, hlensRest⟩
              rcases hmems with ⟨hmem, hmemsRest⟩
              exact ⟨hlen, hmem, ih hlensRest hmemsRest⟩

theorem SharedSlicesUpdateFacts.of_memoryBytesFor
    {st : State} {cta : CTAId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : SharedMemoryBytesFor st cta offsets newSlices) :
    SharedSlicesUpdateFacts st cta offsets oldSlices newSlices := by
  induction offsets generalizing oldSlices newSlices with
  | nil =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hmems
      | cons _ _ =>
          cases newSlices with
          | nil =>
              exact False.elim hlens
          | cons _ _ =>
              exact False.elim hmems
  | cons offset offsets ih =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact False.elim hmems
          | cons _ _ =>
              exact False.elim hlens
      | cons oldBytes oldRest =>
          cases newSlices with
          | nil =>
              cases hmems
          | cons newBytes newRest =>
              rcases hlens with ⟨hlen, hlensRest⟩
              rcases hmems with ⟨hmem, hmemsRest⟩
              exact ⟨hlen, hmem, ih hlensRest hmemsRest⟩

theorem LocalSlicesUpdateFacts.of_memoryBytesFor
    {st : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId}
    {offsets : List Nat} {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems : LocalMemoryBytesFor st cta warp lanes offsets newSlices) :
    LocalSlicesUpdateFacts st cta warp lanes offsets oldSlices newSlices := by
  induction lanes generalizing offsets oldSlices newSlices with
  | nil =>
      cases offsets with
      | nil =>
          cases oldSlices with
          | nil =>
              cases newSlices with
              | nil =>
                  exact True.intro
              | cons _ _ =>
                  cases hmems
          | cons _ _ =>
              cases newSlices with
              | nil =>
                  exact False.elim hlens
              | cons _ _ =>
                  exact False.elim hmems
      | cons _ _ =>
          cases hmems
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          cases hmems
      | cons offset offsets =>
          cases oldSlices with
          | nil =>
              cases newSlices with
              | nil =>
                  exact False.elim hmems
              | cons _ _ =>
                  exact False.elim hlens
          | cons oldBytes oldRest =>
              cases newSlices with
              | nil =>
                  cases hmems
              | cons newBytes newRest =>
                  rcases hlens with ⟨hlen, hlensRest⟩
                  rcases hmems with ⟨hmem, hmemsRest⟩
                  exact ⟨hlen, hmem, ih hlensRest hmemsRest⟩

theorem GlobalSlicesUpdateFacts.of_global_eq
    {st st' : State} {offsets : List Nat} {oldSlices newSlices : List (List Byte)}
    (hglobal : st'.global = st.global)
    (hfacts : GlobalSlicesUpdateFacts st offsets oldSlices newSlices) :
    GlobalSlicesUpdateFacts st' offsets oldSlices newSlices := by
  induction offsets generalizing oldSlices newSlices with
  | nil =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hfacts
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases oldSlices with
      | nil =>
          cases hfacts
      | cons oldBytes oldRest =>
          cases newSlices with
          | nil =>
              cases hfacts
          | cons newBytes newRest =>
              rcases hfacts with ⟨hlen, hmem, hrest⟩
              exact ⟨hlen, by
                rw [hglobal]
                exact hmem, ih hrest⟩

theorem ParamSlicesUpdateFacts.of_param_eq
    {st st' : State} {offsets : List Nat} {slices : List (List Byte)}
    (hparam : st'.param = st.param)
    (hfacts : ParamSlicesUpdateFacts st offsets slices) :
    ParamSlicesUpdateFacts st' offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          cases hfacts
      | cons bytes rest =>
          rcases hfacts with ⟨hmem, hrest⟩
          exact ⟨by
            rw [hparam]
            exact hmem, ih hrest⟩

theorem ConstSlicesUpdateFacts.of_const_eq
    {st st' : State} {offsets : List Nat} {slices : List (List Byte)}
    (hconst : st'.const = st.const)
    (hfacts : ConstSlicesUpdateFacts st offsets slices) :
    ConstSlicesUpdateFacts st' offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          cases hfacts
      | cons bytes rest =>
          rcases hfacts with ⟨hmem, hrest⟩
          exact ⟨by
            rw [hconst]
            exact hmem, ih hrest⟩

theorem ReadMemFact.shared_getCTA
    {st : State} {ty : ScalarTy} {cta : CTAId} {offset : Nat} {value : Value}
    (hread : ReadMemFact st .shared ty (.shared cta offset) value) :
    ∃ ctaState, st.getCTA? cta = some ctaState := by
  unfold ReadMemFact Helpers.readMem? at hread
  cases haccess : (!Typing.typedAccessPreconditions? .shared ty (.shared cta offset)) with
  | true =>
      simp [haccess] at hread
  | false =>
      cases hwidth : Typing.byteWidth? ty with
      | none =>
          simp [haccess, hwidth] at hread
      | some _ =>
          cases hcta : st.getCTA? cta with
          | none =>
              simp [haccess, hwidth, Helpers.getSpaceBaseMem?, hcta] at hread
          | some ctaState =>
              exact ⟨ctaState, rfl⟩

theorem ReadMemFact.local_getLane
    {st : State} {ty : ScalarTy} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {value : Value}
    (hread : ReadMemFact st .local ty (.local cta warp lane offset) value) :
    ∃ laneState, st.getLane? cta warp lane = some laneState := by
  unfold ReadMemFact Helpers.readMem? at hread
  cases haccess : (!Typing.typedAccessPreconditions? .local ty (.local cta warp lane offset)) with
  | true =>
      simp [haccess] at hread
  | false =>
      cases hwidth : Typing.byteWidth? ty with
      | none =>
          simp [haccess, hwidth] at hread
      | some _ =>
          cases hlane : st.getLane? cta warp lane with
          | none =>
              simp [haccess, hwidth, Helpers.getSpaceBaseMem?, hlane] at hread
          | some laneState =>
              exact ⟨laneState, rfl⟩

theorem SharedSlicesUpdateFacts.of_sharedSlices_readValuesFor_eq
    {st st' : State} {r : CSL.Resource} {cta : CTAId} {ty : ScalarTy}
    {offsets : List Nat} {perm : CSL.BytePerm} {slices : List (List Byte)}
    {values : List Value}
    (hshared :
      ∀ ctaState, st.getCTA? cta = some ctaState →
        ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
          ctaState'.shared = ctaState.shared)
    (hslices : sharedSlices cta offsets perm slices st r)
    (hreads : ReadSharedValuesFor st cta ty offsets values) :
    SharedSlicesUpdateFacts st' cta offsets slices slices := by
  induction offsets generalizing slices values r with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hslices.1
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim hslices.1
      | cons bytes rest =>
          cases values with
          | nil =>
              cases hreads
          | cons value values =>
              rcases hreads with ⟨hread, hreadsRest⟩
              change
                (CSL.sharedBytes cta offset perm bytes ∗
                  sharedSlices cta offsets perm rest) st r at hslices
              rcases hslices with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
              rcases ReadMemFact.shared_getCTA hread with ⟨ctaState, hcta⟩
              rcases hshared ctaState hcta with ⟨ctaState', hcta', hsharedEq⟩
              exact ⟨rfl, ⟨ctaState', hcta', by
                rw [hsharedEq]
                exact CSL.sharedBytes_memory hbytes ctaState hcta⟩,
                ih hrest hreadsRest⟩

theorem LocalSlicesUpdateFacts.of_localSlices_readValuesFor_eq
    {st st' : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {lanes : List LaneId} {offsets : List Nat}
    {perm : CSL.BytePerm} {slices : List (List Byte)} {values : List Value}
    (hlocal :
      ∀ lane laneState, st.getLane? cta warp lane = some laneState →
        ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
          laneState'.localMem = laneState.localMem)
    (hslices : localSlices cta warp lanes offsets perm slices st r)
    (hreads : ReadLocalValuesFor st cta warp ty lanes offsets values) :
    LocalSlicesUpdateFacts st' cta warp lanes offsets slices slices := by
  induction lanes generalizing offsets slices values r with
  | nil =>
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              cases values with
              | nil =>
                  exact True.intro
              | cons _ _ =>
                  cases hreads
          | cons _ _ =>
              exact False.elim hslices.1
      | cons _ _ =>
          exact False.elim hslices.1
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim hslices.1
      | cons offset offsets =>
          cases slices with
          | nil =>
              exact False.elim hslices.1
          | cons bytes rest =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases hreads with ⟨hread, hreadsRest⟩
                  change
                    (CSL.localBytes cta warp lane offset perm bytes ∗
                      localSlices cta warp lanes offsets perm rest) st r at hslices
                  rcases hslices with ⟨_rBytes, rRest, _hcomp, _hequiv, hbytes, hrest⟩
                  rcases ReadMemFact.local_getLane hread with ⟨laneState, hlane⟩
                  rcases hlocal lane laneState hlane with ⟨laneState', hlane', hlocalEq⟩
                  exact ⟨rfl, ⟨laneState', hlane', by
                    rw [hlocalEq]
                    exact CSL.localBytes_memory hbytes laneState hlane⟩,
                    ih hrest hreadsRest⟩

theorem EvalRValuesFor.of_regsFor
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {name : RegName} {values : List Value}
    (h : regsFor cta warp lanes name values st r) :
    EvalRValuesFor st cta warp (.reg name) lanes values := by
  induction lanes generalizing values r with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim h.1
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim h.1
      | cons value values =>
          change
            (CSL.reg cta warp lane name value ∗
              regsFor cta warp lanes name values) st r at h
          rcases h with ⟨_rReg, rRest, _hcomp, _hequiv, hreg, hrest⟩
          exact ⟨eval_reg_of_assertion hreg, ih hrest⟩

theorem RegsUpdateFacts.of_applyAssignRegList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {rhs : RValue} {lanes : List LaneId} {values : List Value}
    (hnodup : lanes.Nodup)
    (hevals : EvalRValuesFor stEval cta warp rhs lanes values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons value values =>
          cases hevals
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases values with
          | nil =>
              cases hevals
          | cons value values =>
              rcases hevals with ⟨heval, hevalsRest⟩
              simp [Helpers.applyToLaneIdsList?] at happly
              cases hget : st.getLane? cta warp lane with
              | none =>
                  simp [hget] at happly
              | some laneState =>
                  unfold EvalRValue at heval
                  simp [hget, heval] at happly
                  cases hset :
                      st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                  | none =>
                      simp [hset] at happly
                  | some stNext =>
                      simp [hset] at happly
                      have hlaneNext :
                          stNext.getLane? cta warp lane =
                            some (Helpers.writeReg laneState dst value) :=
                        State.getLane?_setLane_same hget hset
                      have hlaneFinal :
                          stCore.getLane? cta warp lane =
                            some (Helpers.writeReg laneState dst value) :=
                        have hnotMemLane : lane ∉ lanes := by
                          intro hmem
                          exact (hnotMem lane hmem) rfl
                        Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                          hnotMemLane hlaneNext happly
                      exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                        simp [Helpers.writeReg]⟩,
                        ih hnodupRest hevalsRest happly⟩

theorem RegsUpdateFacts.of_applyAssignReg
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {rhs : RValue} {lanes : List LaneId} {values : List Value}
    (hnodup : lanes.Nodup)
    (hevals : EvalRValuesFor stEval cta warp rhs lanes values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyAssignRegList hnodup hevals happly

theorem RegsUpdateFacts.of_applyGlobalLoadList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesGlobalAddrsFor stEval cta warp
        { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadGlobalValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st offsets values with
  | nil =>
      cases offsets with
      | nil =>
          cases values with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hreads
      | cons _ _ =>
          cases haddrs
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases offsets with
          | nil =>
              cases haddrs
          | cons offset offsets =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases haddrs with ⟨haddr, haddrsRest⟩
                  rcases hreads with ⟨hread, hreadsRest⟩
                  simp [Helpers.applyToLaneIdsList?] at happly
                  cases hget : st.getLane? cta warp lane with
                  | none =>
                      simp [hget] at happly
                  | some laneState =>
                      unfold ResolvesAddr at haddr
                      unfold ReadMemFact at hread
                      simp [hget, haddr, hread] at happly
                      cases hset :
                          st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                      | none =>
                          simp [hset] at happly
                      | some stNext =>
                          simp [hset] at happly
                          have hlaneNext :
                              stNext.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            State.getLane?_setLane_same hget hset
                          have hlaneFinal :
                              stCore.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            have hnotMemLane : lane ∉ lanes := by
                              intro hmem
                              exact (hnotMem lane hmem) rfl
                            Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                              hnotMemLane hlaneNext happly
                          exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                            simp [Helpers.writeReg]⟩,
                            ih hnodupRest haddrsRest hreadsRest happly⟩

theorem RegsUpdateFacts.of_applyGlobalLoad
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesGlobalAddrsFor stEval cta warp
        { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadGlobalValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyGlobalLoadList hnodup haddrs hreads happly

theorem RegsUpdateFacts.of_applyParamLoadList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesParamAddrsFor stEval cta warp
        { space := .param, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadParamValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .param, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .param ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st offsets values with
  | nil =>
      cases offsets with
      | nil =>
          cases values with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hreads
      | cons _ _ =>
          cases haddrs
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases offsets with
          | nil =>
              cases haddrs
          | cons offset offsets =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases haddrs with ⟨haddr, haddrsRest⟩
                  rcases hreads with ⟨hread, hreadsRest⟩
                  simp [Helpers.applyToLaneIdsList?] at happly
                  cases hget : st.getLane? cta warp lane with
                  | none =>
                      simp [hget] at happly
                  | some laneState =>
                      unfold ResolvesAddr at haddr
                      unfold ReadMemFact at hread
                      simp [hget, haddr, hread] at happly
                      cases hset :
                          st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                      | none =>
                          simp [hset] at happly
                      | some stNext =>
                          simp [hset] at happly
                          have hlaneNext :
                              stNext.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            State.getLane?_setLane_same hget hset
                          have hlaneFinal :
                              stCore.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            have hnotMemLane : lane ∉ lanes := by
                              intro hmem
                              exact (hnotMem lane hmem) rfl
                            Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                              hnotMemLane hlaneNext happly
                          exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                            simp [Helpers.writeReg]⟩,
                            ih hnodupRest haddrsRest hreadsRest happly⟩

theorem RegsUpdateFacts.of_applyParamLoad
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesParamAddrsFor stEval cta warp
        { space := .param, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadParamValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .param, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .param ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyParamLoadList hnodup haddrs hreads happly

theorem RegsUpdateFacts.of_applyConstLoadList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesConstAddrsFor stEval cta warp
        { space := .const, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadConstValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .const, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .const ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st offsets values with
  | nil =>
      cases offsets with
      | nil =>
          cases values with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hreads
      | cons _ _ =>
          cases haddrs
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases offsets with
          | nil =>
              cases haddrs
          | cons offset offsets =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases haddrs with ⟨haddr, haddrsRest⟩
                  rcases hreads with ⟨hread, hreadsRest⟩
                  simp [Helpers.applyToLaneIdsList?] at happly
                  cases hget : st.getLane? cta warp lane with
                  | none =>
                      simp [hget] at happly
                  | some laneState =>
                      unfold ResolvesAddr at haddr
                      unfold ReadMemFact at hread
                      simp [hget, haddr, hread] at happly
                      cases hset :
                          st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                      | none =>
                          simp [hset] at happly
                      | some stNext =>
                          simp [hset] at happly
                          have hlaneNext :
                              stNext.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            State.getLane?_setLane_same hget hset
                          have hlaneFinal :
                              stCore.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            have hnotMemLane : lane ∉ lanes := by
                              intro hmem
                              exact (hnotMem lane hmem) rfl
                            Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                              hnotMemLane hlaneNext happly
                          exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                            simp [Helpers.writeReg]⟩,
                            ih hnodupRest haddrsRest hreadsRest happly⟩

theorem RegsUpdateFacts.of_applyConstLoad
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesConstAddrsFor stEval cta warp
        { space := .const, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadConstValuesFor stEval ty offsets values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .const, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .const ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyConstLoadList hnodup haddrs hreads happly

theorem RegsUpdateFacts.of_applySharedLoadList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesSharedAddrsFor stEval cta warp
        { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadSharedValuesFor stEval cta ty offsets values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .shared, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .shared ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st offsets values with
  | nil =>
      cases offsets with
      | nil =>
          cases values with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hreads
      | cons _ _ =>
          cases haddrs
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases offsets with
          | nil =>
              cases haddrs
          | cons offset offsets =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases haddrs with ⟨haddr, haddrsRest⟩
                  rcases hreads with ⟨hread, hreadsRest⟩
                  simp [Helpers.applyToLaneIdsList?] at happly
                  cases hget : st.getLane? cta warp lane with
                  | none =>
                      simp [hget] at happly
                  | some laneState =>
                      unfold ResolvesAddr at haddr
                      unfold ReadMemFact at hread
                      simp [hget, haddr, hread] at happly
                      cases hset :
                          st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                      | none =>
                          simp [hset] at happly
                      | some stNext =>
                          simp [hset] at happly
                          have hlaneNext :
                              stNext.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            State.getLane?_setLane_same hget hset
                          have hlaneFinal :
                              stCore.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            have hnotMemLane : lane ∉ lanes := by
                              intro hmem
                              exact (hnotMem lane hmem) rfl
                            Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                              hnotMemLane hlaneNext happly
                          exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                            simp [Helpers.writeReg]⟩,
                            ih hnodupRest haddrsRest hreadsRest happly⟩

theorem RegsUpdateFacts.of_applySharedLoad
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesSharedAddrsFor stEval cta warp
        { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadSharedValuesFor stEval cta ty offsets values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .shared, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .shared ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applySharedLoadList hnodup haddrs hreads happly

theorem RegsUpdateFacts.of_applyLocalLoadList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesLocalAddrsFor stEval cta warp
        { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadLocalValuesFor stEval cta warp ty lanes offsets values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .local ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st offsets values with
  | nil =>
      cases offsets with
      | nil =>
          cases values with
          | nil =>
              exact True.intro
          | cons _ _ =>
              cases hreads
      | cons _ _ =>
          cases haddrs
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases offsets with
          | nil =>
              cases haddrs
          | cons offset offsets =>
              cases values with
              | nil =>
                  cases hreads
              | cons value values =>
                  rcases haddrs with ⟨haddr, haddrsRest⟩
                  rcases hreads with ⟨hread, hreadsRest⟩
                  simp [Helpers.applyToLaneIdsList?] at happly
                  cases hget : st.getLane? cta warp lane with
                  | none =>
                      simp [hget] at happly
                  | some laneState =>
                      unfold ResolvesAddr at haddr
                      unfold ReadMemFact at hread
                      simp [hget, haddr, hread] at happly
                      cases hset :
                          st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                      | none =>
                          simp [hset] at happly
                      | some stNext =>
                          simp [hset] at happly
                          have hlaneNext :
                              stNext.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            State.getLane?_setLane_same hget hset
                          have hlaneFinal :
                              stCore.getLane? cta warp lane =
                                some (Helpers.writeReg laneState dst value) :=
                            have hnotMemLane : lane ∉ lanes := by
                              intro hmem
                              exact (hnotMem lane hmem) rfl
                            Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                              hnotMemLane hlaneNext happly
                          exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                            simp [Helpers.writeReg]⟩,
                            ih hnodupRest haddrsRest hreadsRest happly⟩

theorem RegsUpdateFacts.of_applyLocalLoad
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {values : List Value}
    (hnodup : lanes.Nodup)
    (haddrs :
      ResolvesLocalAddrsFor stEval cta warp
        { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads : ReadLocalValuesFor stEval cta warp ty lanes offsets values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? stEval cta warp lane
              { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? stEval .local ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyLocalLoadList hnodup haddrs hreads happly

theorem RegsUpdateFacts.of_applyCvtaList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Value}
    (hnodup : lanes.Nodup)
    (hevals : EvalCvtaValuesFor stEval cta warp space src lanes values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) lanes = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          cases hevals
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases values with
          | nil =>
              cases hevals
          | cons value values =>
              rcases hevals with ⟨⟨srcValue, heval, hcvta⟩, hevalsRest⟩
              simp [Helpers.applyToLaneIdsList?] at happly
              cases hget : st.getLane? cta warp lane with
              | none =>
                  simp [hget] at happly
              | some laneState =>
                  unfold EvalRValue at heval
                  simp [hget, heval, hcvta] at happly
                  cases hset :
                      st.setLane cta warp lane (Helpers.writeReg laneState dst value) with
                  | none =>
                      simp [hset] at happly
                  | some stNext =>
                      simp [hset] at happly
                      have hlaneNext :
                          stNext.getLane? cta warp lane =
                            some (Helpers.writeReg laneState dst value) :=
                        State.getLane?_setLane_same hget hset
                      have hlaneFinal :
                          stCore.getLane? cta warp lane =
                            some (Helpers.writeReg laneState dst value) :=
                        have hnotMemLane : lane ∉ lanes := by
                          intro hmem
                          exact (hnotMem lane hmem) rfl
                        Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                          hnotMemLane hlaneNext happly
                      exact ⟨⟨Helpers.writeReg laneState dst value, hlaneFinal, by
                        simp [Helpers.writeReg]⟩,
                        ih hnodupRest hevalsRest happly⟩

theorem RegsUpdateFacts.of_applyCvta
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : RegName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Value}
    (hnodup : lanes.Nodup)
    (hevals : EvalCvtaValuesFor stEval cta warp space src lanes values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) = some stCore) :
    RegsUpdateFacts stCore cta warp dst lanes values := by
  exact RegsUpdateFacts.of_applyCvtaList hnodup hevals happly

theorem RegsUpdateFacts.of_advance
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {name : RegName} {lanes : List LaneId} {values : List Value}
    (hfacts : RegsUpdateFacts st cta warp name lanes values)
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    RegsUpdateFacts st' cta warp name lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons value values =>
          cases hfacts
  | cons lane lanes ih =>
      cases values with
      | nil =>
          cases hfacts
      | cons value values =>
          rcases hfacts with ⟨⟨laneState, hlane, hreg⟩, hfactsRest⟩
          rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlane hadvance with
            ⟨laneState', hlane', _hlocal, hregs, _hpreds⟩
          exact ⟨⟨laneState', hlane', by
            simpa [hregs] using hreg⟩, ih hfactsRest⟩

theorem PredsUpdateFacts.of_applyAssignPredList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {cmp : CmpExpr} {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalCmpsFor stEval cta warp cmp lanes values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.evalCmp? stEval cta warp lane cmp).bind fun b =>
            some (Helpers.writePred laneState dst b)) lanes = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons value values =>
          cases hevals
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases values with
          | nil =>
              cases hevals
          | cons value values =>
              rcases hevals with ⟨heval, hevalsRest⟩
              simp [Helpers.applyToLaneIdsList?] at happly
              cases hget : st.getLane? cta warp lane with
              | none =>
                  simp [hget] at happly
              | some laneState =>
                  unfold EvalCmp at heval
                  simp [hget, heval] at happly
                  cases hset :
                      st.setLane cta warp lane (Helpers.writePred laneState dst value) with
                  | none =>
                      simp [hset] at happly
                  | some stNext =>
                      simp [hset] at happly
                      have hlaneNext :
                          stNext.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        State.getLane?_setLane_same hget hset
                      have hlaneFinal :
                          stCore.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        have hnotMemLane : lane ∉ lanes := by
                          intro hmem
                          exact (hnotMem lane hmem) rfl
                        Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                          hnotMemLane hlaneNext happly
                      exact ⟨⟨Helpers.writePred laneState dst value, hlaneFinal, by
                        simp [Helpers.writePred]⟩,
                        ih hnodupRest hevalsRest happly⟩

theorem PredsUpdateFacts.of_applyAssignPred
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {cmp : CmpExpr} {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalCmpsFor stEval cta warp cmp lanes values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalCmp? stEval cta warp lane cmp).bind fun b =>
            some (Helpers.writePred laneState dst b)) = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  exact PredsUpdateFacts.of_applyAssignPredList hnodup hevals happly

theorem PredsUpdateFacts.of_applyAssignPredValueList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {rhs : RValue} {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalRValueBoolsFor stEval cta warp rhs lanes values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane rhs).bind fun value =>
            (Helpers.valueToBool? value).bind fun b =>
              some (Helpers.writePred laneState dst b)) lanes = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons value values =>
          cases hevals
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases values with
          | nil =>
              cases hevals
          | cons value values =>
              rcases hevals with ⟨⟨raw, heval, hbool⟩, hevalsRest⟩
              simp [Helpers.applyToLaneIdsList?] at happly
              cases hget : st.getLane? cta warp lane with
              | none =>
                  simp [hget] at happly
              | some laneState =>
                  unfold EvalRValue at heval
                  simp [hget, heval, hbool] at happly
                  cases hset :
                      st.setLane cta warp lane (Helpers.writePred laneState dst value) with
                  | none =>
                      simp [hset] at happly
                  | some stNext =>
                      simp [hset] at happly
                      have hlaneNext :
                          stNext.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        State.getLane?_setLane_same hget hset
                      have hlaneFinal :
                          stCore.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        have hnotMemLane : lane ∉ lanes := by
                          intro hmem
                          exact (hnotMem lane hmem) rfl
                        Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                          hnotMemLane hlaneNext happly
                      exact ⟨⟨Helpers.writePred laneState dst value, hlaneFinal, by
                        simp [Helpers.writePred]⟩,
                        ih hnodupRest hevalsRest happly⟩

theorem PredsUpdateFacts.of_applyAssignPredValue
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {rhs : RValue} {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalRValueBoolsFor stEval cta warp rhs lanes values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane rhs).bind fun value =>
            (Helpers.valueToBool? value).bind fun b =>
              some (Helpers.writePred laneState dst b)) = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  exact PredsUpdateFacts.of_applyAssignPredValueList hnodup hevals happly

theorem PredsUpdateFacts.of_applyIsspacepList
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalIsspacepValuesFor stEval cta warp space src lanes values)
    (happly :
      Helpers.applyToLaneIdsList? st cta warp
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) lanes = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  induction lanes generalizing st values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          cases hevals
  | cons lane lanes ih =>
      cases hnodup with
      | cons hnotMem hnodupRest =>
          cases values with
          | nil =>
              cases hevals
          | cons value values =>
              rcases hevals with ⟨⟨srcValue, heval, hisspace⟩, hevalsRest⟩
              simp [Helpers.applyToLaneIdsList?] at happly
              cases hget : st.getLane? cta warp lane with
              | none =>
                  simp [hget] at happly
              | some laneState =>
                  unfold EvalRValue at heval
                  simp [hget, heval, hisspace] at happly
                  cases hset :
                      st.setLane cta warp lane (Helpers.writePred laneState dst value) with
                  | none =>
                      simp [hset] at happly
                  | some stNext =>
                      simp [hset] at happly
                      have hlaneNext :
                          stNext.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        State.getLane?_setLane_same hget hset
                      have hlaneFinal :
                          stCore.getLane? cta warp lane =
                            some (Helpers.writePred laneState dst value) :=
                        have hnotMemLane : lane ∉ lanes := by
                          intro hmem
                          exact (hnotMem lane hmem) rfl
                        Helpers.applyToLaneIdsList?_getLane_eq_of_not_mem
                          hnotMemLane hlaneNext happly
                      exact ⟨⟨Helpers.writePred laneState dst value, hlaneFinal, by
                        simp [Helpers.writePred]⟩,
                        ih hnodupRest hevalsRest happly⟩

theorem PredsUpdateFacts.of_applyIsspacep
    {stEval st stCore : State} {cta : CTAId} {warp : WarpId}
    {dst : PredName} {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {values : List Bool}
    (hnodup : lanes.Nodup)
    (hevals : EvalIsspacepValuesFor stEval cta warp space src lanes values)
    (happly :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? stEval cta warp lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) = some stCore) :
    PredsUpdateFacts stCore cta warp dst lanes values := by
  exact PredsUpdateFacts.of_applyIsspacepList hnodup hevals happly

theorem PredsUpdateFacts.of_advance
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {name : PredName} {lanes : List LaneId} {values : List Bool}
    (hfacts : PredsUpdateFacts st cta warp name lanes values)
    (hadvance : Helpers.advanceRunnablePcs? st cta warp = some st') :
    PredsUpdateFacts st' cta warp name lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons value values =>
          cases hfacts
  | cons lane lanes ih =>
      cases values with
      | nil =>
          cases hfacts
      | cons value values =>
          rcases hfacts with ⟨⟨laneState, hlane, hpred⟩, hfactsRest⟩
          rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlane hadvance with
            ⟨laneState', hlane', _hlocal, _hregs, hpreds⟩
          exact ⟨⟨laneState', hlane', by
            simpa [hpreds] using hpred⟩, ih hfactsRest⟩

theorem StateResourceUpdate.reg {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : RegName}
    {old new : Value}
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[name]? = some new) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane name old)
      (CSL.reg cta warp lane name new) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.reg cta warp lane name) (.reg new),
    CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape]), ⟨rfl, hreg⟩⟩

theorem StateResourceUpdate.regsFor {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {name : RegName}
    {oldValues newValues : List Value}
    (hfacts : RegsUpdateFacts st₁ cta warp name lanes newValues) :
    StateResourceUpdate st₀ st₁
      (regsFor cta warp lanes name oldValues)
      (regsFor cta warp lanes name newValues) := by
  induction lanes generalizing oldValues newValues with
  | nil =>
      cases newValues with
      | nil =>
          cases oldValues with
          | nil =>
              exact StateResourceUpdate.emp
          | cons _ _ =>
              exact StateResourceUpdate.of_false_pre
      | cons _ _ =>
          cases hfacts
  | cons lane lanes ih =>
      cases newValues with
      | nil =>
          cases hfacts
      | cons newValue newValues =>
          rcases hfacts with ⟨hreg, hfactsRest⟩
          cases oldValues with
          | nil =>
              exact StateResourceUpdate.of_false_pre
          | cons _ oldValues =>
              exact StateResourceUpdate.sep
                (StateResourceUpdate.reg hreg)
                (ih hfactsRest)

theorem StateResourceUpdate.reg_written {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : RegName}
    {old new : Value}
    (hwrite :
      ∃ laneState, st₁.getLane? cta warp lane = some (Helpers.writeReg laneState name new)) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane name old)
      (CSL.reg cta warp lane name new) := by
  rcases hwrite with ⟨laneState, hget⟩
  exact StateResourceUpdate.reg
    ⟨Helpers.writeReg laneState name new, hget, Helpers.writeReg_read_same laneState name new⟩

theorem StateResourceUpdate.reg_written_advanced {st₀ stCore st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : RegName}
    {old new : Value}
    (hwrite :
      ∃ laneState,
        stCore.getLane? cta warp lane = some (Helpers.writeReg laneState name new))
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁) :
    StateResourceUpdate st₀ st₁
      (CSL.reg cta warp lane name old)
      (CSL.reg cta warp lane name new) := by
  rcases hwrite with ⟨laneState, hget⟩
  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hget hadvance with
    ⟨laneFinal, hgetFinal, _, hregs, _⟩
  exact StateResourceUpdate.reg
    ⟨laneFinal, hgetFinal, by
      rw [hregs]
      exact Helpers.writeReg_read_same laneState name new⟩

theorem StateResourceUpdate.pred {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : PredName}
    {old new : Bool}
    (hpred :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.preds[name]? = some new) :
    StateResourceUpdate st₀ st₁
      (CSL.pred cta warp lane name old)
      (CSL.pred cta warp lane name new) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.pred cta warp lane name) (.pred new),
    CSL.Resource.update_singleton (by simp [CSL.Cell.sameShape]), ⟨rfl, hpred⟩⟩

theorem StateResourceUpdate.predsFor {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {name : PredName}
    {oldValues newValues : List Bool}
    (hfacts : PredsUpdateFacts st₁ cta warp name lanes newValues) :
    StateResourceUpdate st₀ st₁
      (predsFor cta warp lanes name oldValues)
      (predsFor cta warp lanes name newValues) := by
  induction lanes generalizing oldValues newValues with
  | nil =>
      cases newValues with
      | nil =>
          cases oldValues with
          | nil =>
              exact StateResourceUpdate.emp
          | cons _ _ =>
              exact StateResourceUpdate.of_false_pre
      | cons _ _ =>
          cases hfacts
  | cons lane lanes ih =>
      cases newValues with
      | nil =>
          cases hfacts
      | cons newValue newValues =>
          rcases hfacts with ⟨hpred, hfactsRest⟩
          cases oldValues with
          | nil =>
              exact StateResourceUpdate.of_false_pre
          | cons _ oldValues =>
              exact StateResourceUpdate.sep
                (StateResourceUpdate.pred hpred)
                (ih hfactsRest)

theorem StateResourceUpdate.pred_written {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : PredName}
    {old new : Bool}
    (hwrite :
      ∃ laneState, st₁.getLane? cta warp lane =
        some (Helpers.writePred laneState name new)) :
    StateResourceUpdate st₀ st₁
      (CSL.pred cta warp lane name old)
      (CSL.pred cta warp lane name new) := by
  rcases hwrite with ⟨laneState, hget⟩
  exact StateResourceUpdate.pred
    ⟨Helpers.writePred laneState name new, hget, Helpers.writePred_read_same laneState name new⟩

theorem StateResourceUpdate.pred_written_advanced {st₀ stCore st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {name : PredName}
    {old new : Bool}
    (hwrite :
      ∃ laneState,
        stCore.getLane? cta warp lane = some (Helpers.writePred laneState name new))
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁) :
    StateResourceUpdate st₀ st₁
      (CSL.pred cta warp lane name old)
      (CSL.pred cta warp lane name new) := by
  rcases hwrite with ⟨laneState, hget⟩
  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hget hadvance with
    ⟨laneFinal, hgetFinal, _, _, hpreds⟩
  exact StateResourceUpdate.pred
    ⟨laneFinal, hgetFinal, by
      rw [hpreds]
      exact Helpers.writePred_read_same laneState name new⟩

theorem StateResourceUpdate.globalByte {st₀ st₁ : State}
    {offset : Nat} {perm : CSL.BytePerm} {old new : Byte}
    (hmem : CSL.memoryByte st₁.global.bytes offset new) :
    StateResourceUpdate st₀ st₁
      (CSL.globalByte offset perm old)
      (CSL.globalByte offset perm new) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.globalByte offset) (.byte perm new),
    CSL.Resource.update_singleton (by cases perm <;> simp [CSL.Cell.sameShape]),
    ⟨rfl, hmem⟩⟩

theorem StateResourceUpdate.globalBytes {st₀ st₁ : State}
    {offset : Nat} {perm : CSL.BytePerm} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem : CSL.memoryBytes st₁.global.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset perm oldBytes)
      (CSL.globalBytes offset perm newBytes) := by
  induction oldBytes generalizing offset newBytes with
  | nil =>
      cases newBytes with
      | nil =>
          exact StateResourceUpdate.emp
      | cons byte bytes =>
          simp at hlen
  | cons oldByte oldRest ih =>
      cases newBytes with
      | nil =>
          simp at hlen
      | cons newByte newRest =>
          simp [CSL.memoryBytes] at hmem
          have hlenRest : oldRest.length = newRest.length := by
            simp at hlen
            exact hlen
          exact StateResourceUpdate.sep
            (StateResourceUpdate.globalByte (st₀ := st₀) (st₁ := st₁)
              (offset := offset) (perm := perm) hmem.1)
            (ih (offset := offset + 1) (newBytes := newRest) hlenRest hmem.2)

theorem StateResourceUpdate.globalSlices {st₀ st₁ : State}
    {offsets : List Nat} {perm : CSL.BytePerm}
    {oldSlices newSlices : List (List Byte)}
    (hfacts : GlobalSlicesUpdateFacts st₁ offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (globalSlices offsets perm oldSlices)
      (globalSlices offsets perm newSlices) := by
  induction offsets generalizing oldSlices newSlices with
  | nil =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact StateResourceUpdate.emp
          | cons _ _ =>
              cases hfacts
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases oldSlices with
      | nil =>
          cases hfacts
      | cons oldBytes oldSlices =>
          cases newSlices with
          | nil =>
              cases hfacts
          | cons newBytes newSlices =>
              rcases hfacts with ⟨hlen, hmem, hfactsRest⟩
              exact StateResourceUpdate.sep
                (StateResourceUpdate.globalBytes hlen hmem)
                (ih hfactsRest)

theorem StateResourceUpdate.sharedByte {st₀ st₁ : State}
    {cta : CTAId} {offset : Nat} {perm : CSL.BytePerm} {old new : Byte}
    (hmem :
      ∃ ctaState, st₁.getCTA? cta = some ctaState ∧
        CSL.memoryByte ctaState.shared.bytes offset new) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedByte cta offset perm old)
      (CSL.sharedByte cta offset perm new) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.sharedByte cta offset) (.byte perm new),
    CSL.Resource.update_singleton (by cases perm <;> simp [CSL.Cell.sameShape]),
    ⟨rfl, hmem⟩⟩

theorem StateResourceUpdate.sharedBytes {st₀ st₁ : State}
    {cta : CTAId} {offset : Nat} {perm : CSL.BytePerm} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ ctaState, st₁.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset perm oldBytes)
      (CSL.sharedBytes cta offset perm newBytes) := by
  induction oldBytes generalizing offset newBytes with
  | nil =>
      cases newBytes with
      | nil =>
          exact StateResourceUpdate.emp
      | cons byte bytes =>
          simp at hlen
  | cons oldByte oldRest ih =>
      cases newBytes with
      | nil =>
          simp at hlen
      | cons newByte newRest =>
          rcases hmem with ⟨ctaState, hcta, hbytes⟩
          simp [CSL.memoryBytes] at hbytes
          have hlenRest : oldRest.length = newRest.length := by
            simp at hlen
            exact hlen
          exact StateResourceUpdate.sep
            (StateResourceUpdate.sharedByte (st₀ := st₀) (st₁ := st₁)
              (cta := cta) (offset := offset) (perm := perm)
              ⟨ctaState, hcta, hbytes.1⟩)
            (ih (offset := offset + 1) (newBytes := newRest) hlenRest
              ⟨ctaState, hcta, hbytes.2⟩)

theorem StateResourceUpdate.sharedSlices {st₀ st₁ : State}
    {cta : CTAId} {offsets : List Nat} {perm : CSL.BytePerm}
    {oldSlices newSlices : List (List Byte)}
    (hfacts : SharedSlicesUpdateFacts st₁ cta offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (sharedSlices cta offsets perm oldSlices)
      (sharedSlices cta offsets perm newSlices) := by
  induction offsets generalizing oldSlices newSlices with
  | nil =>
      cases oldSlices with
      | nil =>
          cases newSlices with
          | nil =>
              exact StateResourceUpdate.emp
          | cons _ _ =>
              cases hfacts
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases oldSlices with
      | nil =>
          cases hfacts
      | cons oldBytes oldSlices =>
          cases newSlices with
          | nil =>
              cases hfacts
          | cons newBytes newSlices =>
              rcases hfacts with ⟨hlen, hmem, hfactsRest⟩
              exact StateResourceUpdate.sep
                (StateResourceUpdate.sharedBytes hlen hmem)
                (ih hfactsRest)

theorem StateResourceUpdate.localByte {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {offset : Nat}
    {perm : CSL.BytePerm} {old new : Byte}
    (hmem :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        CSL.memoryByte laneState.localMem.bytes offset new) :
    StateResourceUpdate st₀ st₁
      (CSL.localByte cta warp lane offset perm old)
      (CSL.localByte cta warp lane offset perm new) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.localByte cta warp lane offset) (.byte perm new),
    CSL.Resource.update_singleton (by cases perm <;> simp [CSL.Cell.sameShape]),
    ⟨rfl, hmem⟩⟩

theorem StateResourceUpdate.localBytes {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lane : LaneId} {offset : Nat}
    {perm : CSL.BytePerm} {oldBytes newBytes : List Byte}
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset newBytes) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset perm oldBytes)
      (CSL.localBytes cta warp lane offset perm newBytes) := by
  induction oldBytes generalizing offset newBytes with
  | nil =>
      cases newBytes with
      | nil =>
          exact StateResourceUpdate.emp
      | cons byte bytes =>
          simp at hlen
  | cons oldByte oldRest ih =>
      cases newBytes with
      | nil =>
          simp at hlen
      | cons newByte newRest =>
          rcases hmem with ⟨laneState, hlane, hbytes⟩
          simp [CSL.memoryBytes] at hbytes
          have hlenRest : oldRest.length = newRest.length := by
            simp at hlen
            exact hlen
          exact StateResourceUpdate.sep
            (StateResourceUpdate.localByte (st₀ := st₀) (st₁ := st₁)
              (cta := cta) (warp := warp) (lane := lane) (offset := offset)
              (perm := perm) ⟨laneState, hlane, hbytes.1⟩)
            (ih (offset := offset + 1) (newBytes := newRest) hlenRest
              ⟨laneState, hlane, hbytes.2⟩)

theorem StateResourceUpdate.localSlices {st₀ st₁ : State}
    {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {offsets : List Nat}
    {perm : CSL.BytePerm} {oldSlices newSlices : List (List Byte)}
    (hfacts : LocalSlicesUpdateFacts st₁ cta warp lanes offsets oldSlices newSlices) :
    StateResourceUpdate st₀ st₁
      (localSlices cta warp lanes offsets perm oldSlices)
      (localSlices cta warp lanes offsets perm newSlices) := by
  induction lanes generalizing offsets oldSlices newSlices with
  | nil =>
      cases offsets with
      | nil =>
          cases oldSlices with
          | nil =>
              cases newSlices with
              | nil =>
                  exact StateResourceUpdate.emp
              | cons _ _ =>
                  cases hfacts
          | cons _ _ =>
              cases hfacts
      | cons _ _ =>
          cases hfacts
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          cases hfacts
      | cons offset offsets =>
          cases oldSlices with
          | nil =>
              cases hfacts
          | cons oldBytes oldSlices =>
              cases newSlices with
              | nil =>
                  cases hfacts
              | cons newBytes newSlices =>
                  rcases hfacts with ⟨hlen, hmem, hfactsRest⟩
                  exact StateResourceUpdate.sep
                    (StateResourceUpdate.localBytes hlen hmem)
                    (ih hfactsRest)

theorem StateResourceUpdate.globalBytes_of_writeMemFact
    {st₀ st₁ : State} {ty : ScalarTy} {offset : Nat}
    {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) := by
  apply StateResourceUpdate.globalBytes hlen
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode] at hwrite
  rcases hwrite with ⟨_, hst⟩
  rw [← hst]
  exact memoryBytes_writeBytes

theorem StateResourceUpdate.globalBytes_of_writeMemFact_advanced
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.globalBytes offset .write oldBytes)
      (CSL.globalBytes offset .write newBytes) := by
  apply StateResourceUpdate.globalBytes hlen
  have hglobal : st₁.global = stCore.global := Helpers.advanceRunnablePcs?_global_eq hadvance
  rw [hglobal]
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode] at hwrite
  rcases hwrite with ⟨_, hstCore⟩
  rw [← hstCore]
  exact memoryBytes_writeBytes

theorem WriteMemFact.global_memoryBytes_written
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    {bytes : List Byte}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value bytes) :
    CSL.memoryBytes stCore.global.bytes offset bytes := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode] at hwrite
  rcases hwrite with ⟨_, hstCore⟩
  rw [← hstCore]
  exact memoryBytes_writeBytes

theorem WriteMemFact.global_memoryBytes_preserved_of_disjoint
    {st stCore : State} {ty : ScalarTy} {writeOffset readOffset : Nat}
    {value : Value} {writeBytes readBytes : List Byte}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset writeBytes.length)
    (hmem : CSL.memoryBytes st.global.bytes readOffset readBytes)
    (hwrite : WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    CSL.memoryBytes stCore.global.bytes readOffset readBytes := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode] at hwrite
  rcases hwrite with ⟨_, hstCore⟩
  rw [← hstCore]
  exact memoryBytes_writeBytes_preserved_of_disjoint hdisjoint hmem

theorem GlobalMemoryBytesFor.preserve_global_write
    {st stCore : State} {ty : ScalarTy} {writeOffset : Nat}
    {value : Value} {writeBytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hdisjoint : ByteRangesDisjointFrom writeOffset writeBytes offsets slices)
    (hmems : GlobalMemoryBytesFor st offsets slices)
    (hwrite : WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    GlobalMemoryBytesFor stCore offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hmems
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim hmems
      | cons bytes rest =>
          rcases hdisjoint with ⟨hhead, htail⟩
          rcases hmems with ⟨hmem, hmemsRest⟩
          exact ⟨
            WriteMemFact.global_memoryBytes_preserved_of_disjoint
              hhead hmem hwrite hencode,
            ih htail hmemsRest⟩

theorem GlobalMemoryBytesFor.of_global_write_cons
    {st stCore : State} {ty : ScalarTy} {offset : Nat}
    {value : Value} {bytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value bytes)
    (hdisjoint : ByteRangesDisjointFrom offset bytes offsets slices)
    (hmems : GlobalMemoryBytesFor st offsets slices) :
    GlobalMemoryBytesFor stCore (offset :: offsets) (bytes :: slices) :=
  ⟨WriteMemFact.global_memoryBytes_written hwrite hencode,
    GlobalMemoryBytesFor.preserve_global_write hdisjoint hmems hwrite hencode⟩

theorem WriteMemFact.global_getWarp_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (cta : CTAId) (warp : WarpId) :
    stCore.getWarp? cta warp = st.getWarp? cta warp := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  cases haccess : (!Typing.typedAccessPreconditions? .global ty (.global offset)) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess, Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?] at hwrite
      cases henc : Helpers.encodeScalar? ty value with
      | none =>
          simp [henc] at hwrite
      | some bytes =>
          simp [henc] at hwrite
          rw [← hwrite]
          rfl

theorem WriteMemFact.global_getLane_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (cta : CTAId) (warp : WarpId) (lane : LaneId) :
    stCore.getLane? cta warp lane = st.getLane? cta warp lane := by
  unfold State.getLane?
  rw [WriteMemFact.global_getWarp_eq hwrite cta warp]

theorem WriteMemFact.global_kernelEnv_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {value : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore) :
    stCore.kernelEnv = st.kernelEnv :=
  Helpers.writeMem?_kernelEnv_eq hwrite

theorem WriteMemFact.global_evalRValue_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (cta : CTAId) (warp : WarpId) (lane : LaneId) :
    ∀ expr,
      Helpers.evalRValue? stCore cta warp lane expr =
        Helpers.evalRValue? st cta warp lane expr := by
  intro expr
  induction expr with
  | imm value =>
      rfl
  | reg name =>
      simp [Helpers.evalRValue?, WriteMemFact.global_getLane_eq hwrite cta warp lane]
  | pred name =>
      simp [Helpers.evalRValue?, WriteMemFact.global_getLane_eq hwrite cta warp lane]
  | special special =>
      simp [Helpers.evalRValue?, WriteMemFact.global_kernelEnv_eq hwrite]
  | unop op arg ih =>
      simp [Helpers.evalRValue?, ih]
  | binop op lhs rhs ihL ihR =>
      simp [Helpers.evalRValue?, ihL, ihR]
  | triop op a b c ihA ihB ihC =>
      simp [Helpers.evalRValue?, ihA, ihB, ihC]

theorem WriteMemFact.global_evalRValue
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {expr : RValue} {value : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (heval : EvalRValue st ctx expr value) :
    EvalRValue stCore ctx expr value := by
  unfold EvalRValue at heval ⊢
  rw [WriteMemFact.global_evalRValue_eq hwrite ctx.cta ctx.warp ctx.lane expr]
  exact heval

theorem WriteMemFact.global_evalCmp_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (cmp : CmpExpr) :
    Helpers.evalCmp? stCore cta warp lane cmp =
      Helpers.evalCmp? st cta warp lane cmp := by
  cases cmp with
  | mk op lhs rhs =>
      unfold Helpers.evalCmp?
      rw [WriteMemFact.global_evalRValue_eq hwrite cta warp lane lhs]
      rw [WriteMemFact.global_evalRValue_eq hwrite cta warp lane rhs]

theorem WriteMemFact.global_evalCmp
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {cmp : CmpExpr} {value : Bool}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (heval : EvalCmp st ctx cmp value) :
    EvalCmp stCore ctx cmp value := by
  unfold EvalCmp at heval ⊢
  rw [WriteMemFact.global_evalCmp_eq hwrite ctx.cta ctx.warp ctx.lane cmp]
  exact heval

theorem WriteMemFact.global_resolveAddr_eq
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (cta : CTAId) (warp : WarpId) (lane : LaneId) (addr : TypedAddr) :
    Helpers.resolveAddr? stCore cta warp lane addr =
      Helpers.resolveAddr? st cta warp lane addr := by
  cases addr with
  | mk space ty addrExpr =>
      simp [Helpers.resolveAddr?,
        WriteMemFact.global_evalRValue_eq hwrite cta warp lane addrExpr]

theorem WriteMemFact.global_resolvesAddr
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {addr : TypedAddr} {resolved : Addr}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (haddr : ResolvesAddr st ctx addr resolved) :
    ResolvesAddr stCore ctx addr resolved := by
  unfold ResolvesAddr at haddr ⊢
  rw [WriteMemFact.global_resolveAddr_eq hwrite ctx.cta ctx.warp ctx.lane addr]
  exact haddr

theorem EvalRValuesFor.of_global_write
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {cta : CTAId} {warp : WarpId} {rhs : RValue}
    {lanes : List LaneId} {values : List Value}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (hevals : EvalRValuesFor st cta warp rhs lanes values) :
    EvalRValuesFor stCore cta warp rhs lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hevals
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim hevals
      | cons value values =>
          rcases hevals with ⟨heval, htail⟩
          exact ⟨WriteMemFact.global_evalRValue hwrite heval, ih htail⟩

theorem EvalCmpsFor.of_global_write
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {cta : CTAId} {warp : WarpId} {cmp : CmpExpr}
    {lanes : List LaneId} {values : List Bool}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (hevals : EvalCmpsFor st cta warp cmp lanes values) :
    EvalCmpsFor stCore cta warp cmp lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hevals
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim hevals
      | cons value values =>
          rcases hevals with ⟨heval, htail⟩
          exact ⟨WriteMemFact.global_evalCmp hwrite heval, ih htail⟩

theorem ResolvesGlobalAddrsFor.of_global_write
    {st stCore : State} {ty : ScalarTy} {offset : Nat} {writeValue : Value}
    {cta : CTAId} {warp : WarpId} {addr : TypedAddr}
    {lanes : List LaneId} {offsets : List Nat}
    (hwrite : WriteMemFact st .global ty (.global offset) writeValue stCore)
    (haddrs : ResolvesGlobalAddrsFor st cta warp addr lanes offsets) :
    ResolvesGlobalAddrsFor stCore cta warp addr lanes offsets := by
  induction lanes generalizing offsets with
  | nil =>
      cases offsets with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          rcases haddrs with ⟨haddr, htail⟩
          exact ⟨WriteMemFact.global_resolvesAddr hwrite haddr, ih htail⟩

theorem GlobalMemoryBytesFor.of_global_eq
    {st st' : State} {offsets : List Nat} {slices : List (List Byte)}
    (hglobal : st'.global = st.global)
    (hmems : GlobalMemoryBytesFor st offsets slices) :
    GlobalMemoryBytesFor st' offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hmems
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim hmems
      | cons bytes rest =>
          rcases hmems with ⟨hmem, htail⟩
          exact ⟨by simpa [hglobal] using hmem, ih htail⟩

theorem SharedMemoryBytesFor.of_shared_preserved
    {st st' : State} {cta : CTAId} {offsets : List Nat}
    {slices : List (List Byte)}
    (hshared :
      ∀ ctaState, st.getCTA? cta = some ctaState →
        ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
          ctaState'.shared = ctaState.shared)
    (hmems : SharedMemoryBytesFor st cta offsets slices) :
    SharedMemoryBytesFor st' cta offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hmems
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim hmems
      | cons bytes rest =>
          rcases hmems with ⟨⟨ctaState, hcta, hmem⟩, htail⟩
          rcases hshared ctaState hcta with ⟨ctaState', hcta', hsharedEq⟩
          exact ⟨⟨ctaState', hcta', by
            rw [hsharedEq]
            exact hmem⟩, ih htail⟩

theorem GlobalMemoryBytesFor.preserve_stepStoreLanes_global
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {readOffset : Nat} {readBytes : List Byte}
    {lanes : List LaneId} {writeOffsets : List Nat}
    {values : List Value} {writeSlices : List (List Byte)}
    (hdisjoint : ByteRangesDisjointFrom readOffset readBytes writeOffsets writeSlices)
    (haddrs :
      ResolvesGlobalAddrsFor st cta warp
        { space := .global, ty := ty, addr := addrExpr } lanes writeOffsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values writeSlices)
    (hmem : CSL.memoryBytes st.global.bytes readOffset readBytes)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .global, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    CSL.memoryBytes stFinal.global.bytes readOffset readBytes := by
  induction lanes generalizing st writeOffsets values writeSlices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      exact hmem
  | cons lane lanes ih =>
      cases writeOffsets with
      | nil =>
          exact False.elim haddrs
      | cons writeOffset writeOffsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases writeSlices with
              | nil =>
                  exact False.elim hencs
              | cons writeBytes writeSlices =>
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite :
                      Helpers.writeMem? st .global ty (.global writeOffset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .global ty (.global writeOffset) value stNext :=
                        hwrite
                      have hmemNext :
                          CSL.memoryBytes stNext.global.bytes readOffset readBytes :=
                        WriteMemFact.global_memoryBytes_preserved_of_disjoint
                          (ByteRangesDisjoint.symm hdisjointHead) hmem hwriteFact henc
                      have haddrsNext :
                          ResolvesGlobalAddrsFor stNext cta warp
                            { space := .global, ty := ty, addr := addrExpr }
                            lanes writeOffsets :=
                        ResolvesGlobalAddrsFor.of_global_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_global_write hwriteFact hevalsTail
                      exact ih hdisjointTail haddrsNext hevalsNext hencsTail hmemNext hstep

theorem GlobalMemoryBytesFor.of_stepStoreLanes_global
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
    GlobalMemoryBytesFor stFinal offsets slices := by
  induction lanes generalizing st offsets values slices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              exact False.elim hdisjoint
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases slices with
              | nil =>
                  exact False.elim hencs
              | cons bytes slices =>
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite : Helpers.writeMem? st .global ty (.global offset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .global ty (.global offset) value stNext :=
                        hwrite
                      have haddrsNext :
                          ResolvesGlobalAddrsFor stNext cta warp
                            { space := .global, ty := ty, addr := addrExpr } lanes offsets :=
                        ResolvesGlobalAddrsFor.of_global_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_global_write hwriteFact hevalsTail
                      have htail :
                          GlobalMemoryBytesFor stFinal offsets slices :=
                        ih haddrsNext hevalsNext hencsTail hdisjointTail hstep
                      have hheadNext :
                          CSL.memoryBytes stNext.global.bytes offset bytes :=
                        WriteMemFact.global_memoryBytes_written hwriteFact henc
                      have hhead :
                          CSL.memoryBytes stFinal.global.bytes offset bytes :=
                        GlobalMemoryBytesFor.preserve_stepStoreLanes_global
                          hdisjointHead haddrsNext hevalsNext hencsTail hheadNext hstep
                      exact ⟨hhead, htail⟩

theorem WriteMemFact.shared_getWarp_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {offset : Nat} {value : Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) value stCore) :
    stCore.getWarp? cta warp = st.getWarp? cta warp := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  cases haccess : (!Typing.typedAccessPreconditions? .shared ty (.shared cta offset)) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      cases henc : Helpers.encodeScalar? ty value with
      | none =>
          simp [haccess, henc] at hwrite
      | some bytes =>
          cases hcta : st.getCTA? cta with
          | none =>
              simp [haccess, henc, Helpers.getSpaceBaseMem?, hcta] at hwrite
          | some ctaState =>
              simp [haccess, henc, Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hcta]
                at hwrite
              rw [← hwrite]
              have hctaRaw : st.ctas[cta]? = some ctaState := by
                simpa [State.getCTA?] using hcta
              simp [State.getWarp?, State.getCTA?, State.setCTA, hctaRaw]

theorem WriteMemFact.shared_getLane_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {value : Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) value stCore) :
    stCore.getLane? cta warp lane = st.getLane? cta warp lane := by
  unfold State.getLane?
  rw [WriteMemFact.shared_getWarp_eq hwrite]

theorem WriteMemFact.shared_kernelEnv_eq
    {st stCore : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {value : Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) value stCore) :
    stCore.kernelEnv = st.kernelEnv :=
  Helpers.writeMem?_kernelEnv_eq hwrite

theorem WriteMemFact.shared_evalRValue_eq
    {st stCore : State} {writeCta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    (hwrite : WriteMemFact st .shared ty (.shared writeCta offset) writeValue stCore)
    (warp : WarpId) (lane : LaneId) :
    ∀ expr,
      Helpers.evalRValue? stCore writeCta warp lane expr =
        Helpers.evalRValue? st writeCta warp lane expr := by
  intro expr
  induction expr with
  | imm value =>
      rfl
  | reg name =>
      simp [Helpers.evalRValue?, WriteMemFact.shared_getLane_eq hwrite]
  | pred name =>
      simp [Helpers.evalRValue?, WriteMemFact.shared_getLane_eq hwrite]
  | special special =>
      simp [Helpers.evalRValue?, WriteMemFact.shared_kernelEnv_eq hwrite]
  | unop op arg ih =>
      simp [Helpers.evalRValue?, ih]
  | binop op lhs rhs ihL ihR =>
      simp [Helpers.evalRValue?, ihL, ihR]
  | triop op a b c ihA ihB ihC =>
      simp [Helpers.evalRValue?, ihA, ihB, ihC]

theorem WriteMemFact.shared_evalRValue
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {expr : RValue} {value : Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) writeValue stCore)
    (heval : EvalRValue st ctx expr value)
    (hctx : ctx.cta = cta) :
    EvalRValue stCore ctx expr value := by
  unfold EvalRValue at heval ⊢
  subst hctx
  rw [WriteMemFact.shared_evalRValue_eq hwrite ctx.warp ctx.lane expr]
  exact heval

theorem WriteMemFact.shared_resolveAddr_eq
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) writeValue stCore)
    (warp : WarpId) (lane : LaneId) (addr : TypedAddr) :
    Helpers.resolveAddr? stCore cta warp lane addr =
      Helpers.resolveAddr? st cta warp lane addr := by
  cases addr with
  | mk space ty addrExpr =>
      simp [Helpers.resolveAddr?,
        WriteMemFact.shared_evalRValue_eq hwrite warp lane addrExpr]

theorem WriteMemFact.shared_resolvesAddr
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    {ctx : LaneCtx} {addr : TypedAddr} {resolved : Addr}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) writeValue stCore)
    (haddr : ResolvesAddr st ctx addr resolved)
    (hctx : ctx.cta = cta) :
    ResolvesAddr stCore ctx addr resolved := by
  unfold ResolvesAddr at haddr ⊢
  subst hctx
  rw [WriteMemFact.shared_resolveAddr_eq hwrite ctx.warp ctx.lane addr]
  exact haddr

theorem EvalRValuesFor.of_shared_write
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    {warp : WarpId} {rhs : RValue}
    {lanes : List LaneId} {values : List Value}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) writeValue stCore)
    (hevals : EvalRValuesFor st cta warp rhs lanes values) :
    EvalRValuesFor stCore cta warp rhs lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hevals
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim hevals
      | cons value values =>
          rcases hevals with ⟨heval, htail⟩
          exact ⟨WriteMemFact.shared_evalRValue hwrite heval rfl, ih htail⟩

theorem ResolvesSharedAddrsFor.of_shared_write
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {offset : Nat} {writeValue : Value}
    {warp : WarpId} {addr : TypedAddr}
    {lanes : List LaneId} {offsets : List Nat}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) writeValue stCore)
    (haddrs : ResolvesSharedAddrsFor st cta warp addr lanes offsets) :
    ResolvesSharedAddrsFor stCore cta warp addr lanes offsets := by
  induction lanes generalizing offsets with
  | nil =>
      cases offsets with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          rcases haddrs with ⟨haddr, htail⟩
          exact ⟨WriteMemFact.shared_resolvesAddr hwrite haddr rfl, ih htail⟩

theorem WriteMemFact.shared_memoryBytes_written
    {st stCore : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {value : Value} {bytes : List Byte}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) value stCore)
    (hencode : EncodedScalar ty value bytes) :
    ∃ ctaState, stCore.getCTA? cta = some ctaState ∧
      CSL.memoryBytes ctaState.shared.bytes offset bytes := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hcta : st.getCTA? cta with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hcta] at hwrite
  | some ctaState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hcta] at hwrite
      rcases hwrite with ⟨_, hstCore⟩
      rw [← hstCore]
      refine ⟨{ ctaState with shared := {
          bytes := Helpers.writeBytes ctaState.shared.bytes offset bytes } }, ?_, ?_⟩
      · simp [State.getCTA?, State.setCTA, Addr.offset]
      · exact memoryBytes_writeBytes

theorem WriteMemFact.shared_memoryBytes_preserved_of_disjoint
    {st stCore : State} {cta : CTAId} {ty : ScalarTy}
    {writeOffset readOffset : Nat} {value : Value}
    {writeBytes readBytes : List Byte}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset writeBytes.length)
    (hmem :
      ∃ ctaState, st.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes readOffset readBytes)
    (hwrite : WriteMemFact st .shared ty (.shared cta writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    ∃ ctaState, stCore.getCTA? cta = some ctaState ∧
      CSL.memoryBytes ctaState.shared.bytes readOffset readBytes := by
  rcases hmem with ⟨ctaState, hcta, hmemBytes⟩
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hctaWrite : st.getCTA? cta with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hctaWrite] at hwrite
  | some ctaStateWrite =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hctaWrite]
        at hwrite
      have hsame : ctaStateWrite = ctaState := by
        rw [hcta] at hctaWrite
        exact (Option.some.inj hctaWrite).symm
      subst ctaStateWrite
      rcases hwrite with ⟨_, hstCore⟩
      rw [← hstCore]
      refine ⟨{ ctaState with
          shared := { bytes := Helpers.writeBytes ctaState.shared.bytes writeOffset writeBytes } },
        ?_, ?_⟩
      · simp [State.getCTA?, State.setCTA, Addr.offset]
      · exact memoryBytes_writeBytes_preserved_of_disjoint hdisjoint hmemBytes

theorem SharedMemoryBytesFor.preserve_shared_write
    {st stCore : State} {cta : CTAId} {ty : ScalarTy} {writeOffset : Nat}
    {value : Value} {writeBytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hdisjoint : ByteRangesDisjointFrom writeOffset writeBytes offsets slices)
    (hmems : SharedMemoryBytesFor st cta offsets slices)
    (hwrite : WriteMemFact st .shared ty (.shared cta writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    SharedMemoryBytesFor stCore cta offsets slices := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hmems
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          exact False.elim hmems
      | cons bytes rest =>
          rcases hdisjoint with ⟨hhead, htail⟩
          rcases hmems with ⟨hmem, hmemsRest⟩
          exact ⟨
            WriteMemFact.shared_memoryBytes_preserved_of_disjoint
              hhead hmem hwrite hencode,
            ih htail hmemsRest⟩

theorem SharedMemoryBytesFor.of_shared_write_cons
    {st stCore : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {value : Value} {bytes : List Byte}
    {offsets : List Nat} {slices : List (List Byte)}
    (hwrite : WriteMemFact st .shared ty (.shared cta offset) value stCore)
    (hencode : EncodedScalar ty value bytes)
    (hdisjoint : ByteRangesDisjointFrom offset bytes offsets slices)
    (hmems : SharedMemoryBytesFor st cta offsets slices) :
    SharedMemoryBytesFor stCore cta (offset :: offsets) (bytes :: slices) :=
  ⟨WriteMemFact.shared_memoryBytes_written hwrite hencode,
    SharedMemoryBytesFor.preserve_shared_write hdisjoint hmems hwrite hencode⟩

theorem SharedMemoryBytesFor.preserve_stepStoreLanes_shared
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {readOffset : Nat} {readBytes : List Byte}
    {lanes : List LaneId} {writeOffsets : List Nat}
    {values : List Value} {writeSlices : List (List Byte)}
    (hdisjoint : ByteRangesDisjointFrom readOffset readBytes writeOffsets writeSlices)
    (haddrs :
      ResolvesSharedAddrsFor st cta warp
        { space := .shared, ty := ty, addr := addrExpr } lanes writeOffsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values writeSlices)
    (hmem :
      ∃ ctaState, st.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes readOffset readBytes)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .shared, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    ∃ ctaState, stFinal.getCTA? cta = some ctaState ∧
      CSL.memoryBytes ctaState.shared.bytes readOffset readBytes := by
  induction lanes generalizing st writeOffsets values writeSlices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      exact hmem
  | cons lane lanes ih =>
      cases writeOffsets with
      | nil =>
          exact False.elim haddrs
      | cons writeOffset writeOffsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases writeSlices with
              | nil =>
                  exact False.elim hencs
              | cons writeBytes writeSlices =>
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite :
                      Helpers.writeMem? st .shared ty (.shared cta writeOffset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .shared ty (.shared cta writeOffset) value stNext :=
                        hwrite
                      have hmemNext :
                          ∃ ctaState, stNext.getCTA? cta = some ctaState ∧
                            CSL.memoryBytes ctaState.shared.bytes readOffset readBytes :=
                        WriteMemFact.shared_memoryBytes_preserved_of_disjoint
                          (ByteRangesDisjoint.symm hdisjointHead) hmem hwriteFact henc
                      have haddrsNext :
                          ResolvesSharedAddrsFor stNext cta warp
                            { space := .shared, ty := ty, addr := addrExpr }
                            lanes writeOffsets :=
                        ResolvesSharedAddrsFor.of_shared_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_shared_write hwriteFact hevalsTail
                      exact ih hdisjointTail haddrsNext hevalsNext hencsTail hmemNext hstep

theorem SharedMemoryBytesFor.of_stepStoreLanes_shared
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {values : List Value} {slices : List (List Byte)}
    (haddrs :
      ResolvesSharedAddrsFor st cta warp
        { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values slices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets slices)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .shared, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    SharedMemoryBytesFor stFinal cta offsets slices := by
  induction lanes generalizing st offsets values slices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              exact False.elim hdisjoint
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases slices with
              | nil =>
                  exact False.elim hencs
              | cons bytes slices =>
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite :
                      Helpers.writeMem? st .shared ty (.shared cta offset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .shared ty (.shared cta offset) value stNext :=
                        hwrite
                      have haddrsNext :
                          ResolvesSharedAddrsFor stNext cta warp
                            { space := .shared, ty := ty, addr := addrExpr } lanes offsets :=
                        ResolvesSharedAddrsFor.of_shared_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_shared_write hwriteFact hevalsTail
                      have htail :
                          SharedMemoryBytesFor stFinal cta offsets slices :=
                        ih haddrsNext hevalsNext hencsTail hdisjointTail hstep
                      have hheadNext :
                          ∃ ctaState, stNext.getCTA? cta = some ctaState ∧
                            CSL.memoryBytes ctaState.shared.bytes offset bytes :=
                        WriteMemFact.shared_memoryBytes_written hwriteFact henc
                      have hhead :
                          ∃ ctaState, stFinal.getCTA? cta = some ctaState ∧
                            CSL.memoryBytes ctaState.shared.bytes offset bytes :=
                        SharedMemoryBytesFor.preserve_stepStoreLanes_shared
                          hdisjointHead haddrsNext hevalsNext hencsTail hheadNext hstep
                      exact ⟨hhead, htail⟩

theorem WriteMemFact.local_readReg_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (target : LaneId) (reg : RegName) :
    (do
      let laneState <- stCore.getLane? cta warp target
      Helpers.readReg laneState reg) =
    (do
      let laneState <- st.getLane? cta warp target
      Helpers.readReg laneState reg) := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  cases haccess :
      (!Typing.typedAccessPreconditions? .local ty (.local cta warp writeLane writeOffset)) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess] at hwrite
      cases henc : Helpers.encodeScalar? ty writeValue with
      | none =>
          simp [henc] at hwrite
      | some encoded =>
          simp [henc] at hwrite
          cases hwriteLane : st.getLane? cta warp writeLane with
          | none =>
              simp [Helpers.getSpaceBaseMem?, hwriteLane] at hwrite
          | some writeState =>
              simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hwriteLane] at hwrite
              have hset :
                  st.setLane cta warp writeLane
                    { writeState with localMem := {
                        bytes := Helpers.writeBytes writeState.localMem.bytes writeOffset encoded } } =
                    some stCore := by
                simpa [Addr.offset] using hwrite
              by_cases heq : target = writeLane
              · subst target
                let writeState' : LaneState := { writeState with localMem := {
                  bytes := Helpers.writeBytes writeState.localMem.bytes writeOffset encoded } }
                have htarget :
                    stCore.getLane? cta warp writeLane = some writeState' := by
                  simpa [writeState', Addr.offset] using
                    State.getLane?_setLane_same hwriteLane hset
                simp [Helpers.readReg, hwriteLane, htarget, writeState']
              · have htarget :
                    stCore.getLane? cta warp target = st.getLane? cta warp target :=
                  State.getLane?_setLane_ne hwriteLane heq hset
                simp [Helpers.readReg, htarget]

theorem WriteMemFact.local_readPredValue_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (target : LaneId) (pred : PredName) :
    (do
      let laneState <- stCore.getLane? cta warp target
      let b <- Helpers.readPred laneState pred
      pure (Value.pred b)) =
    (do
      let laneState <- st.getLane? cta warp target
      let b <- Helpers.readPred laneState pred
      pure (Value.pred b)) := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  cases haccess :
      (!Typing.typedAccessPreconditions? .local ty (.local cta warp writeLane writeOffset)) with
  | true =>
      simp [haccess] at hwrite
  | false =>
      simp [haccess] at hwrite
      cases henc : Helpers.encodeScalar? ty writeValue with
      | none =>
          simp [henc] at hwrite
      | some encoded =>
          simp [henc] at hwrite
          cases hwriteLane : st.getLane? cta warp writeLane with
          | none =>
              simp [Helpers.getSpaceBaseMem?, hwriteLane] at hwrite
          | some writeState =>
              simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hwriteLane] at hwrite
              have hset :
                  st.setLane cta warp writeLane
                    { writeState with localMem := {
                        bytes := Helpers.writeBytes writeState.localMem.bytes writeOffset encoded } } =
                    some stCore := by
                simpa [Addr.offset] using hwrite
              by_cases heq : target = writeLane
              · subst target
                let writeState' : LaneState := { writeState with localMem := {
                  bytes := Helpers.writeBytes writeState.localMem.bytes writeOffset encoded } }
                have htarget :
                    stCore.getLane? cta warp writeLane = some writeState' := by
                  simpa [writeState', Addr.offset] using
                    State.getLane?_setLane_same hwriteLane hset
                simp [Helpers.readPred, hwriteLane, htarget, writeState']
              · have htarget :
                    stCore.getLane? cta warp target = st.getLane? cta warp target :=
                  State.getLane?_setLane_ne hwriteLane heq hset
                simp [Helpers.readPred, htarget]

theorem WriteMemFact.local_evalRValue_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (lane : LaneId) :
    ∀ expr,
      Helpers.evalRValue? stCore cta warp lane expr =
        Helpers.evalRValue? st cta warp lane expr := by
  intro expr
  induction expr with
  | imm value =>
      rfl
  | reg name =>
      simpa [Helpers.evalRValue?] using
        WriteMemFact.local_readReg_eq hwrite lane name
  | pred name =>
      simpa [Helpers.evalRValue?] using
        WriteMemFact.local_readPredValue_eq hwrite lane name
  | special special =>
      simp [Helpers.evalRValue?, Helpers.writeMem?_kernelEnv_eq hwrite]
  | unop op arg ih =>
      simp [Helpers.evalRValue?, ih]
  | binop op lhs rhs ihL ihR =>
      simp [Helpers.evalRValue?, ihL, ihR]
  | triop op a b c ihA ihB ihC =>
      simp [Helpers.evalRValue?, ihA, ihB, ihC]

theorem WriteMemFact.local_evalRValue
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    {ctx : LaneCtx} {expr : RValue} {value : Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (heval : EvalRValue st ctx expr value)
    (hctx : ctx.cta = cta ∧ ctx.warp = warp) :
    EvalRValue stCore ctx expr value := by
  unfold EvalRValue at heval ⊢
  rcases hctx with ⟨rfl, rfl⟩
  rw [WriteMemFact.local_evalRValue_eq hwrite ctx.lane expr]
  exact heval

theorem WriteMemFact.local_resolveAddr_eq
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (lane : LaneId) (addr : TypedAddr) :
    Helpers.resolveAddr? stCore cta warp lane addr =
      Helpers.resolveAddr? st cta warp lane addr := by
  cases addr with
  | mk space ty addrExpr =>
      simp [Helpers.resolveAddr?,
        WriteMemFact.local_evalRValue_eq hwrite lane addrExpr]

theorem WriteMemFact.local_resolvesAddr
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value}
    {ctx : LaneCtx} {addr : TypedAddr} {resolved : Addr}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (haddr : ResolvesAddr st ctx addr resolved)
    (hctx : ctx.cta = cta ∧ ctx.warp = warp) :
    ResolvesAddr stCore ctx addr resolved := by
  unfold ResolvesAddr at haddr ⊢
  rcases hctx with ⟨rfl, rfl⟩
  rw [WriteMemFact.local_resolveAddr_eq hwrite ctx.lane addr]
  exact haddr

theorem EvalRValuesFor.of_local_write
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value} {rhs : RValue}
    {lanes : List LaneId} {values : List Value}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (hevals : EvalRValuesFor st cta warp rhs lanes values) :
    EvalRValuesFor stCore cta warp rhs lanes values := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim hevals
  | cons lane lanes ih =>
      cases values with
      | nil =>
          exact False.elim hevals
      | cons value values =>
          rcases hevals with ⟨heval, htail⟩
          exact ⟨WriteMemFact.local_evalRValue hwrite heval ⟨rfl, rfl⟩, ih htail⟩

theorem ResolvesLocalAddrsFor.of_local_write
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {writeValue : Value} {addr : TypedAddr}
    {lanes : List LaneId} {offsets : List Nat}
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) writeValue stCore)
    (haddrs : ResolvesLocalAddrsFor st cta warp addr lanes offsets) :
    ResolvesLocalAddrsFor stCore cta warp addr lanes offsets := by
  induction lanes generalizing offsets with
  | nil =>
      cases offsets with
      | nil =>
          exact True.intro
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          rcases haddrs with ⟨haddr, htail⟩
          exact ⟨WriteMemFact.local_resolvesAddr hwrite haddr ⟨rfl, rfl⟩, ih htail⟩

theorem WriteMemFact.local_memoryBytes_written
    {st stCore : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {value : Value} {bytes : List Byte}
    (hwrite : WriteMemFact st .local ty (.local cta warp lane offset) value stCore)
    (hencode : EncodedScalar ty value bytes) :
    ∃ laneState, stCore.getLane? cta warp lane = some laneState ∧
      CSL.memoryBytes laneState.localMem.bytes offset bytes := by
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hlane : st.getLane? cta warp lane with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hlane] at hwrite
  | some laneState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hlane] at hwrite
      rcases hwrite with ⟨_, hset⟩
      let laneCore : LaneState := { laneState with localMem := {
        bytes := Helpers.writeBytes laneState.localMem.bytes offset bytes } }
      refine ⟨laneCore, ?_, ?_⟩
      · simpa [laneCore, Addr.offset] using State.getLane?_setLane_same hlane hset
      · exact memoryBytes_writeBytes

theorem WriteMemFact.local_memoryBytes_preserved
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane readLane : LaneId} {ty : ScalarTy} {writeOffset readOffset : Nat}
    {value : Value} {writeBytes readBytes : List Byte}
    (hdisjoint :
      readLane ≠ writeLane ∨
        ByteRangesDisjoint readOffset readBytes.length writeOffset writeBytes.length)
    (hmem :
      ∃ laneState, st.getLane? cta warp readLane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes readOffset readBytes)
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    ∃ laneState, stCore.getLane? cta warp readLane = some laneState ∧
      CSL.memoryBytes laneState.localMem.bytes readOffset readBytes := by
  rcases hmem with ⟨readState, hreadLane, hreadBytes⟩
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hwriteLane : st.getLane? cta warp writeLane with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hwriteLane] at hwrite
  | some writeState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hwriteLane]
        at hwrite
      rcases hwrite with ⟨_, hset⟩
      by_cases heq : readLane = writeLane
      · subst readLane
        rw [hwriteLane] at hreadLane
        injection hreadLane with hsame
        subst readState
        have hrange :
            ByteRangesDisjoint readOffset readBytes.length writeOffset writeBytes.length := by
          rcases hdisjoint with hne | hrange
          · exact False.elim (hne rfl)
          · exact hrange
        let writeState' : LaneState := { writeState with localMem := {
          bytes := Helpers.writeBytes writeState.localMem.bytes writeOffset writeBytes } }
        refine ⟨writeState', ?_, ?_⟩
        · simpa [writeState', Addr.offset] using State.getLane?_setLane_same hwriteLane hset
        · exact memoryBytes_writeBytes_preserved_of_disjoint hrange hreadBytes
      · have hreadFinal :
            stCore.getLane? cta warp readLane = st.getLane? cta warp readLane :=
          State.getLane?_setLane_ne hwriteLane heq hset
        exact ⟨readState, by rwa [hreadFinal], hreadBytes⟩

theorem LocalMemoryBytesFor.preserve_local_write
    {st stCore : State} {cta : CTAId} {warp : WarpId}
    {writeLane : LaneId} {ty : ScalarTy} {writeOffset : Nat}
    {value : Value} {writeBytes : List Byte}
    {lanes : List LaneId} {offsets : List Nat} {slices : List (List Byte)}
    (hdisjoint : LocalByteRangesDisjointFrom writeLane writeOffset writeBytes lanes offsets slices)
    (hmems : LocalMemoryBytesFor st cta warp lanes offsets slices)
    (hwrite :
      WriteMemFact st .local ty (.local cta warp writeLane writeOffset) value stCore)
    (hencode : EncodedScalar ty value writeBytes) :
    LocalMemoryBytesFor stCore cta warp lanes offsets slices := by
  induction lanes generalizing offsets slices with
  | nil =>
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              exact False.elim hmems
      | cons _ _ =>
          exact False.elim hmems
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim hmems
      | cons offset offsets =>
          cases slices with
          | nil =>
              exact False.elim hmems
          | cons bytes rest =>
              rcases hdisjoint with ⟨hhead, htail⟩
              rcases hmems with ⟨hmem, hmemsRest⟩
              exact ⟨
                WriteMemFact.local_memoryBytes_preserved
                  hhead hmem hwrite hencode,
                ih htail hmemsRest⟩

theorem LocalMemoryBytesFor.of_local_write_cons
    {st stCore : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {value : Value} {bytes : List Byte}
    {lanes : List LaneId} {offsets : List Nat} {slices : List (List Byte)}
    (hwrite : WriteMemFact st .local ty (.local cta warp lane offset) value stCore)
    (hencode : EncodedScalar ty value bytes)
    (hdisjoint : LocalByteRangesDisjointFrom lane offset bytes lanes offsets slices)
    (hmems : LocalMemoryBytesFor st cta warp lanes offsets slices) :
    LocalMemoryBytesFor stCore cta warp (lane :: lanes) (offset :: offsets) (bytes :: slices) :=
  ⟨WriteMemFact.local_memoryBytes_written hwrite hencode,
    LocalMemoryBytesFor.preserve_local_write hdisjoint hmems hwrite hencode⟩

theorem LocalMemoryBytesFor.of_local_preserved
    {st st' : State} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {offsets : List Nat} {slices : List (List Byte)}
    (hlocal :
      ∀ lane laneState, st.getLane? cta warp lane = some laneState →
        ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
          laneState'.localMem = laneState.localMem)
    (hmems : LocalMemoryBytesFor st cta warp lanes offsets slices) :
    LocalMemoryBytesFor st' cta warp lanes offsets slices := by
  induction lanes generalizing offsets slices with
  | nil =>
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              exact False.elim hmems
      | cons _ _ =>
          exact False.elim hmems
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim hmems
      | cons offset offsets =>
          cases slices with
          | nil =>
              exact False.elim hmems
          | cons bytes rest =>
              rcases hmems with ⟨⟨laneState, hlane, hbytes⟩, htail⟩
              rcases hlocal lane laneState hlane with ⟨laneState', hlane', hlocalEq⟩
              exact ⟨⟨laneState', hlane', by
                rw [hlocalEq]
                exact hbytes⟩, ih htail⟩

theorem LocalMemoryBytesFor.preserve_stepStoreLanes_local
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {readLane : LaneId} {readOffset : Nat} {readBytes : List Byte}
    {lanes : List LaneId} {writeOffsets : List Nat}
    {values : List Value} {writeSlices : List (List Byte)}
    (hdisjoint :
      LocalByteRangesDisjointFrom readLane readOffset readBytes lanes writeOffsets writeSlices)
    (haddrs :
      ResolvesLocalAddrsFor st cta warp
        { space := .local, ty := ty, addr := addrExpr } lanes writeOffsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values writeSlices)
    (hmem :
      ∃ laneState, st.getLane? cta warp readLane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes readOffset readBytes)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .local, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    ∃ laneState, stFinal.getLane? cta warp readLane = some laneState ∧
      CSL.memoryBytes laneState.localMem.bytes readOffset readBytes := by
  induction lanes generalizing st writeOffsets values writeSlices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      exact hmem
  | cons lane lanes ih =>
      cases writeOffsets with
      | nil =>
          exact False.elim haddrs
      | cons writeOffset writeOffsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases writeSlices with
              | nil =>
                  exact False.elim hencs
              | cons writeBytes writeSlices =>
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite :
                      Helpers.writeMem? st .local ty (.local cta warp lane writeOffset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .local ty (.local cta warp lane writeOffset)
                            value stNext :=
                        hwrite
                      have hmemNext :
                          ∃ laneState, stNext.getLane? cta warp readLane = some laneState ∧
                            CSL.memoryBytes laneState.localMem.bytes readOffset readBytes :=
                        have hpres :
                            readLane ≠ lane ∨
                              ByteRangesDisjoint readOffset readBytes.length
                                writeOffset writeBytes.length := by
                          rcases hdisjointHead with hne | hrange
                          · exact Or.inl (by
                              intro heq
                              exact hne heq.symm)
                          · exact Or.inr (ByteRangesDisjoint.symm hrange)
                        WriteMemFact.local_memoryBytes_preserved
                          hpres hmem hwriteFact henc
                      have haddrsNext :
                          ResolvesLocalAddrsFor stNext cta warp
                            { space := .local, ty := ty, addr := addrExpr }
                            lanes writeOffsets :=
                        ResolvesLocalAddrsFor.of_local_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_local_write hwriteFact hevalsTail
                      exact ih hdisjointTail haddrsNext hevalsNext hencsTail hmemNext hstep

theorem LocalMemoryBytesFor.of_stepStoreLanes_local
    {st stFinal : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {values : List Value} {slices : List (List Byte)}
    (haddrs :
      ResolvesLocalAddrsFor st cta warp
        { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals : EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values slices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets slices)
    (hstep :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .local, ty := ty, addr := addrExpr } valueExpr =
        some stFinal) :
    LocalMemoryBytesFor stFinal cta warp lanes offsets slices := by
  induction lanes generalizing st offsets values slices with
  | nil =>
      simp [Helpers.stepStoreLanes?] at hstep
      subst stFinal
      cases offsets with
      | nil =>
          cases slices with
          | nil =>
              exact True.intro
          | cons _ _ =>
              exact False.elim hdisjoint
      | cons _ _ =>
          exact False.elim haddrs
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          exact False.elim haddrs
      | cons offset offsets =>
          cases values with
          | nil =>
              exact False.elim hevals
          | cons value values =>
              cases slices with
              | nil =>
                  exact False.elim hencs
              | cons bytes slices =>
                  rcases haddrs with ⟨haddr, haddrsTail⟩
                  rcases hevals with ⟨heval, hevalsTail⟩
                  rcases hencs with ⟨henc, hencsTail⟩
                  rcases hdisjoint with ⟨hdisjointHead, hdisjointTail⟩
                  unfold Helpers.stepStoreLanes? at hstep
                  unfold ResolvesAddr at haddr
                  unfold EvalRValue at heval
                  simp [haddr, heval] at hstep
                  cases hwrite :
                      Helpers.writeMem? st .local ty (.local cta warp lane offset) value with
                  | none =>
                      simp [hwrite] at hstep
                  | some stNext =>
                      simp [hwrite] at hstep
                      have hwriteFact :
                          WriteMemFact st .local ty (.local cta warp lane offset) value stNext :=
                        hwrite
                      have haddrsNext :
                          ResolvesLocalAddrsFor stNext cta warp
                            { space := .local, ty := ty, addr := addrExpr } lanes offsets :=
                        ResolvesLocalAddrsFor.of_local_write hwriteFact haddrsTail
                      have hevalsNext :
                          EvalRValuesFor stNext cta warp valueExpr lanes values :=
                        EvalRValuesFor.of_local_write hwriteFact hevalsTail
                      have htail :
                          LocalMemoryBytesFor stFinal cta warp lanes offsets slices :=
                        ih haddrsNext hevalsNext hencsTail hdisjointTail hstep
                      have hheadNext :
                          ∃ laneState, stNext.getLane? cta warp lane = some laneState ∧
                            CSL.memoryBytes laneState.localMem.bytes offset bytes :=
                        WriteMemFact.local_memoryBytes_written hwriteFact henc
                      have hhead :
                          ∃ laneState, stFinal.getLane? cta warp lane = some laneState ∧
                            CSL.memoryBytes laneState.localMem.bytes offset bytes :=
                        LocalMemoryBytesFor.preserve_stepStoreLanes_local
                          hdisjointHead haddrsNext hevalsNext hencsTail hheadNext hstep
                      exact ⟨hhead, htail⟩

theorem StateResourceUpdate.sharedBytes_of_writeMemFact
    {st₀ st₁ : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .shared ty (.shared cta offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset .write oldBytes)
      (CSL.sharedBytes cta offset .write newBytes) := by
  apply StateResourceUpdate.sharedBytes hlen
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hcta : st₀.getCTA? cta with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hcta] at hwrite
  | some ctaState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hcta] at hwrite
      rcases hwrite with ⟨_, hst⟩
      rw [← hst]
      refine ⟨{ ctaState with shared := {
          bytes := Helpers.writeBytes ctaState.shared.bytes offset newBytes } }, ?_, ?_⟩
      · simp [State.getCTA?, State.setCTA, Addr.offset]
      · exact memoryBytes_writeBytes

theorem StateResourceUpdate.sharedBytes_of_writeMemFact_advanced
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .shared ty (.shared cta offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.sharedBytes cta offset .write oldBytes)
      (CSL.sharedBytes cta offset .write newBytes) := by
  apply StateResourceUpdate.sharedBytes hlen
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hcta : st₀.getCTA? cta with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hcta] at hwrite
  | some ctaState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hcta] at hwrite
      rcases hwrite with ⟨_, hstCore⟩
      let ctaCore : CTAState := { ctaState with
        shared := { bytes := Helpers.writeBytes ctaState.shared.bytes offset newBytes } }
      have hctaCore : stCore.getCTA? cta = some ctaCore := by
        rw [← hstCore]
        simp [State.getCTA?, State.setCTA, ctaCore, Addr.offset]
      rcases Helpers.advanceRunnablePcs?_shared_eq hctaCore hadvance with
        ⟨ctaFinal, hctaFinal, hsharedFinal⟩
      refine ⟨ctaFinal, hctaFinal, ?_⟩
      rw [hsharedFinal]
      exact memoryBytes_writeBytes

theorem StateResourceUpdate.localBytes_of_writeMemFact
    {st₀ st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .local ty (.local cta warp lane offset) value st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset .write oldBytes)
      (CSL.localBytes cta warp lane offset .write newBytes) := by
  apply StateResourceUpdate.localBytes hlen
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hlane : st₀.getLane? cta warp lane with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hlane] at hwrite
  | some laneState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hlane] at hwrite
      rcases hwrite with ⟨_, hset⟩
      refine ⟨{ laneState with localMem := {
          bytes := Helpers.writeBytes laneState.localMem.bytes offset newBytes } }, ?_, ?_⟩
      · exact State.getLane?_setLane_same hlane hset
      · exact memoryBytes_writeBytes

theorem StateResourceUpdate.localBytes_of_writeMemFact_advanced
    {st₀ stCore st₁ : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwrite : WriteMemFact st₀ .local ty (.local cta warp lane offset) value stCore)
    (hadvance : Helpers.advanceRunnablePcs? stCore cta warp = some st₁)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    StateResourceUpdate st₀ st₁
      (CSL.localBytes cta warp lane offset .write oldBytes)
      (CSL.localBytes cta warp lane offset .write newBytes) := by
  apply StateResourceUpdate.localBytes hlen
  unfold WriteMemFact Helpers.writeMem? at hwrite
  unfold EncodedScalar at hencode
  cases hlane : st₀.getLane? cta warp lane with
  | none =>
      simp [Helpers.getSpaceBaseMem?, hencode, hlane] at hwrite
  | some laneState =>
      simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hlane] at hwrite
      rcases hwrite with ⟨_, hset⟩
      let laneCore : LaneState := { laneState with localMem := {
        bytes := Helpers.writeBytes laneState.localMem.bytes offset newBytes } }
      have hcore : stCore.getLane? cta warp lane = some laneCore := by
        simpa [laneCore, Addr.offset] using State.getLane?_setLane_same hlane hset
      rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hcore hadvance with
        ⟨laneFinal, hlaneFinal, hlocalFinal, _, _⟩
      refine ⟨laneFinal, hlaneFinal, ?_⟩
      rw [hlocalFinal]
      exact memoryBytes_writeBytes

theorem StateResourceUpdate.paramByte {st₀ st₁ : State}
    {offset : Nat} {value : Byte}
    (hmem : CSL.memoryByte st₁.param.bytes offset value) :
    StateResourceUpdate st₀ st₁
      (CSL.paramByte offset value)
      (CSL.paramByte offset value) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.paramByte offset) (.byte .read value),
    CSL.Resource.update_refl _, ⟨rfl, hmem⟩⟩

theorem StateResourceUpdate.paramBytes {st₀ st₁ : State}
    {offset : Nat} {bytes : List Byte}
    (hmem : CSL.memoryBytes st₁.param.bytes offset bytes) :
    StateResourceUpdate st₀ st₁
      (CSL.paramBytes offset bytes)
      (CSL.paramBytes offset bytes) := by
  induction bytes generalizing offset with
  | nil =>
      exact StateResourceUpdate.emp
  | cons byte bytes ih =>
      simp [CSL.memoryBytes] at hmem
      exact StateResourceUpdate.sep
        (StateResourceUpdate.paramByte (st₀ := st₀) (st₁ := st₁)
          (offset := offset) hmem.1)
        (ih (offset := offset + 1) hmem.2)

theorem StateResourceUpdate.paramSlices {st₀ st₁ : State}
    {offsets : List Nat} {slices : List (List Byte)}
    (hfacts : ParamSlicesUpdateFacts st₁ offsets slices) :
    StateResourceUpdate st₀ st₁
      (paramSlices offsets slices)
      (paramSlices offsets slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact StateResourceUpdate.emp
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          cases hfacts
      | cons bytes rest =>
          rcases hfacts with ⟨hmem, hrest⟩
          exact StateResourceUpdate.sep
            (StateResourceUpdate.paramBytes hmem)
            (ih hrest)

theorem StateResourceUpdate.constByte {st₀ st₁ : State}
    {offset : Nat} {value : Byte}
    (hmem : CSL.memoryByte st₁.const.bytes offset value) :
    StateResourceUpdate st₀ st₁
      (CSL.constByte offset value)
      (CSL.constByte offset value) := by
  intro r hpre
  rcases hpre with ⟨howns, _⟩
  subst r
  exact ⟨CSL.Resource.singleton (.constByte offset) (.byte .read value),
    CSL.Resource.update_refl _, ⟨rfl, hmem⟩⟩

theorem StateResourceUpdate.constBytes {st₀ st₁ : State}
    {offset : Nat} {bytes : List Byte}
    (hmem : CSL.memoryBytes st₁.const.bytes offset bytes) :
    StateResourceUpdate st₀ st₁
      (CSL.constBytes offset bytes)
      (CSL.constBytes offset bytes) := by
  induction bytes generalizing offset with
  | nil =>
      exact StateResourceUpdate.emp
  | cons byte bytes ih =>
      simp [CSL.memoryBytes] at hmem
      exact StateResourceUpdate.sep
        (StateResourceUpdate.constByte (st₀ := st₀) (st₁ := st₁)
          (offset := offset) hmem.1)
        (ih (offset := offset + 1) hmem.2)

theorem StateResourceUpdate.constSlices {st₀ st₁ : State}
    {offsets : List Nat} {slices : List (List Byte)}
    (hfacts : ConstSlicesUpdateFacts st₁ offsets slices) :
    StateResourceUpdate st₀ st₁
      (constSlices offsets slices)
      (constSlices offsets slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          exact StateResourceUpdate.emp
      | cons _ _ =>
          cases hfacts
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          cases hfacts
      | cons bytes rest =>
          rcases hfacts with ⟨hmem, hrest⟩
          exact StateResourceUpdate.sep
            (StateResourceUpdate.constBytes hmem)
            (ih hrest)

theorem wpInstr_of_spec
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {pre post : CSL.Assertion}
    (hspec : InstrSpec cta warp gi pre post) :
    pre ⊢ₛ wpInstr cta warp gi post := by
  intro st r hpre st' hstep
  exact hspec st r st' hpre hstep

theorem wpTerminator_of_spec
    {cta : CTAId} {warp : WarpId} {term : Terminator} {pre post : CSL.Assertion}
    (hspec : TerminatorSpec cta warp term pre post) :
    pre ⊢ₛ wpTerminator cta warp term post := by
  intro st r hpre st' hstep
  exact hspec st r st' hpre hstep

theorem wpExecutableStep_of_spec
    {cta : CTAId} {warp : WarpId} {pre post : CSL.Assertion}
    (hspec : ExecutableStepSpec cta warp pre post) :
    pre ⊢ₛ wpExecutableStep cta warp post := by
  intro st r hpre st' hstep
  exact hspec st r st' hpre hstep

theorem InstrSpec.of_computed
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepInstr? st₀ cta warp gi = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp gi
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) := by
  intro st r st' hpre hstep'
  rcases hpre with ⟨hst, hpre'⟩
  subst st
  rw [hstep] at hstep'
  injection hstep' with h
  subst st'
  rcases hpost r hpre' with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, ⟨rfl, hpost'⟩⟩

theorem TerminatorSpec.of_computed
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ cta warp term = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec cta warp term
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) := by
  intro st r st' hpre hstep'
  rcases hpre with ⟨hst, hpre'⟩
  subst st
  rw [hstep] at hstep'
  injection hstep' with h
  subst st'
  rcases hpost r hpre' with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, ⟨rfl, hpost'⟩⟩

theorem brSpec_of_computed
    {cta : CTAId} {warp : WarpId} {target : BlockLabel}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ cta warp (.br target) = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec cta warp (.br target)
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  TerminatorSpec.of_computed hstep hpost

theorem brSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel} {lane : LaneId} :
    TerminatorSpec cta warp (.br target)
      (warpAt cta warp pc [lane])
      (warpAt cta warp (target, 0) [lane]) := by
  intro st r st' hpre hstep
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanes]
    simp
  rcases (by
      unfold Helpers.runnableLaneIds at hmemRunnable
      simp [Helpers.laneIsRunnable] at hmemRunnable
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hmemRunnable
      | some laneState =>
          exact ⟨laneState, rfl⟩ :
      ∃ laneState, warpState.getLane? lane = some laneState) with
    ⟨laneState, hwarpLane⟩
  have hlane : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hwarpLane]
  have hpcLane : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have hset :
      st.setLane cta warp lane { laneState with pc := (target, 0) } = some st' := by
    have hstepSet := hstep
    unfold Helpers.stepTerminator? at hstepSet
    simp [hwarp, hlockBool, hrpcOpt, hlanes, hwarpLane, hpcLane] at hstepSet
    unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at hstepSet
    simp [hlane] at hstepSet
    simpa [Helpers.applyToLaneIdsList?] using hstepSet
  have hpost : warpAt cta warp (target, 0) [lane] st' CSL.Resource.empty :=
    warpAt_single_of_setLane_pc hwarp hlock hrpc hpart hlane hset
      (by simp) (by simp)
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem brSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel} {lane : LaneId}
    {frame : CSL.Assertion}
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗ frame) st r →
        frame st rFrame →
        Helpers.stepTerminator? st cta warp (.br target) = some st' →
        frame st' rFrame) :
    TerminatorSpec cta warp (.br target)
      (warpAt cta warp pc [lane] ∗ frame)
      (warpAt cta warp (target, 0) [lane] ∗ frame) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanes]
    simp
  rcases (by
      unfold Helpers.runnableLaneIds at hmemRunnable
      simp [Helpers.laneIsRunnable] at hmemRunnable
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hmemRunnable
      | some laneState =>
          exact ⟨laneState, rfl⟩ :
      ∃ laneState, warpState.getLane? lane = some laneState) with
    ⟨laneState, hwarpLane⟩
  have hlane : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hwarpLane]
  have hpcLane : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have hset :
      st.setLane cta warp lane { laneState with pc := (target, 0) } = some st' := by
    have hstepSet := hstep
    unfold Helpers.stepTerminator? at hstepSet
    simp [hwarp, hlockBool, hrpcOpt, hlanes, hwarpLane, hpcLane] at hstepSet
    unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at hstepSet
    simp [hlane] at hstepSet
    simpa [Helpers.applyToLaneIdsList?] using hstepSet
  have hpostCtrl : warpAt cta warp (target, 0) [lane] st' CSL.Resource.empty :=
    warpAt_single_of_setLane_pc hwarp hlock hrpc hpart hlane hset
      (by simp) (by simp)
  have hframeFinal : frame st' rFrame :=
    hframe st st' r rFrame
      ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rFrame, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _)
        (CSL.Resource.update_refl _))
  · exact ⟨CSL.Resource.empty, rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hpostCtrl, hframeFinal⟩

theorem brSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel} {lane : LaneId}
    {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep cta warp (.br target)) frame) :
    TerminatorSpec cta warp (.br target)
      (warpAt cta warp pc [lane] ∗ frame)
      (warpAt cta warp (target, 0) [lane] ∗ frame) :=
  brSpec_single_warpAt_frame (by
    intro st st' _r rFrame _hpre hframeSt hstep
    exact hframe st st' rFrame hstep hframeSt)

theorem brSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel}
    {lanes : List LaneId} :
    TerminatorSpec cta warp (.br target)
      (warpAt cta warp pc lanes)
      (warpAt cta warp (target, 0) lanes) := by
  intro st r st' hpre hstep
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hcurrent :
      List.filter
          (fun lane =>
            match warpState.getLane? lane with
            | some laneState => laneState.pc == pc
            | none => false)
          (Helpers.runnableLaneIds warpState) =
        lanes := by
    exact (Helpers.runnableLaneIds_filter_current_pc_eq hlock hrpc).trans hlanesStart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have happly :
      Helpers.applyToLaneIds? st cta warp lanes
          (fun _ laneState => some { laneState with pc := (target, 0) }) =
        some st' := by
    have hstepApply := hstep
    unfold Helpers.stepTerminator? at hstepApply
    simp [hwarp, hlockBool, hrpcOpt] at hstepApply
    exact hcurrent ▸ hstepApply
  have hpost : warpAt cta warp (target, 0) lanes st' CSL.Resource.empty :=
    warpAt_lanes_of_set_pc hwarp hlock hrpc hpart happly
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem cbrSpec_of_computed
    {cta : CTAId} {warp : WarpId} {cond : RValue} {tLabel fLabel : BlockLabel}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ cta warp (.cbr cond tLabel fLabel) = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  TerminatorSpec.of_computed hstep hpost

theorem cbrSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lane : LaneId} {value : Value} {takeTrue : Bool}
    (heval :
      ∀ st r,
        warpAt cta warp pc [lane] st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc [lane])
      (warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanes]
    simp
  rcases (by
      unfold Helpers.runnableLaneIds at hmemRunnable
      simp [Helpers.laneIsRunnable] at hmemRunnable
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hmemRunnable
      | some laneState =>
          exact ⟨laneState, rfl⟩ :
      ∃ laneState, warpState.getLane? lane = some laneState) with
    ⟨laneState, hwarpLane⟩
  have hlane : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hwarpLane]
  have hpcLane : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have hset :
      st.setLane cta warp lane
          { laneState with pc := if takeTrue then (tLabel, 0) else (fLabel, 0) } =
        some st' := by
    have hstepSet := hstep
    unfold Helpers.stepTerminator? at hstepSet
    unfold Helpers.uniformBranchDestination? at hstepSet
    unfold EvalRValue at heval'
    simp [hwarp, hlockBool, hrpcOpt, hlanes, hwarpLane, hpcLane, heval', hbool]
      at hstepSet
    unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at hstepSet
    simp [hlane] at hstepSet
    simpa [Helpers.applyToLaneIdsList?] using hstepSet
  have hpost :
      warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]
        st' CSL.Resource.empty :=
    warpAt_single_of_setLane_pc hwarp hlock hrpc hpart hlane hset
      (by simp) (by simp)
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem cbrSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lane : LaneId} {value : Value} {takeTrue : Bool}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ frame) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗ frame) st r →
        frame st rFrame →
        Helpers.stepTerminator? st cta warp (.cbr cond tLabel fLabel) = some st' →
        frame st' rFrame) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc [lane] ∗ frame)
      (warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane] ∗ frame) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanes]
    simp
  rcases (by
      unfold Helpers.runnableLaneIds at hmemRunnable
      simp [Helpers.laneIsRunnable] at hmemRunnable
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hmemRunnable
      | some laneState =>
          exact ⟨laneState, rfl⟩ :
      ∃ laneState, warpState.getLane? lane = some laneState) with
    ⟨laneState, hwarpLane⟩
  have hlane : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hwarpLane]
  have hpcLane : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have hset :
      st.setLane cta warp lane
          { laneState with pc := if takeTrue then (tLabel, 0) else (fLabel, 0) } =
        some st' := by
    have hstepSet := hstep
    unfold Helpers.stepTerminator? at hstepSet
    unfold Helpers.uniformBranchDestination? at hstepSet
    unfold EvalRValue at heval'
    simp [hwarp, hlockBool, hrpcOpt, hlanes, hwarpLane, hpcLane, heval', hbool]
      at hstepSet
    unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at hstepSet
    simp [hlane] at hstepSet
    simpa [Helpers.applyToLaneIdsList?] using hstepSet
  have hpostCtrl :
      warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]
        st' CSL.Resource.empty :=
    warpAt_single_of_setLane_pc hwarp hlock hrpc hpart hlane hset
      (by simp) (by simp)
  have hframeFinal : frame st' rFrame :=
    hframe st st' r rFrame
      ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rFrame, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _)
        (CSL.Resource.update_refl _))
  · exact ⟨CSL.Resource.empty, rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hpostCtrl, hframeFinal⟩

theorem cbrSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lane : LaneId} {value : Value} {takeTrue : Bool}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ frame) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue)
    (hframe :
      CSL.StableUnder (TerminatorStep cta warp (.cbr cond tLabel fLabel)) frame) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc [lane] ∗ frame)
      (warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane] ∗ frame) :=
  cbrSpec_single_warpAt_frame heval hbool (by
    intro st st' _r rFrame _hpre hframeSt hstep
    exact hframe st st' rFrame hstep hframeSt)

theorem cbrSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc dest : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lanes : List LaneId}
    (hdest :
      ∀ st r,
        warpAt cta warp pc lanes st r →
          Helpers.uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some dest) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc lanes)
      (warpAt cta warp dest lanes) := by
  intro st r st' hpre hstep
  have hdest' := hdest st r hpre
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hcurrent :
      List.filter
          (fun lane =>
            match warpState.getLane? lane with
            | some laneState => laneState.pc == pc
            | none => false)
          (Helpers.runnableLaneIds warpState) =
        lanes := by
    exact (Helpers.runnableLaneIds_filter_current_pc_eq hlock hrpc).trans hlanesStart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have happly :
      Helpers.applyToLaneIds? st cta warp lanes
          (fun _ laneState => some { laneState with pc := dest }) =
        some st' := by
    have hstepApply := hstep
    unfold Helpers.stepTerminator? at hstepApply
    simp [hwarp, hlockBool, hrpcOpt] at hstepApply
    have hstepApplyLanes := hcurrent ▸ hstepApply
    simpa [hdest'] using hstepApplyLanes
  have hpost : warpAt cta warp dest lanes st' CSL.Resource.empty :=
    warpAt_lanes_of_set_pc hwarp hlock hrpc hpart happly
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem cbrSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc dest : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lanes : List LaneId} {frame : CSL.Assertion}
    (hdest :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ frame) st r →
          Helpers.uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some dest)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗ frame) st r →
        frame st rFrame →
        Helpers.stepTerminator? st cta warp (.cbr cond tLabel fLabel) = some st' →
        frame st' rFrame) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc lanes ∗ frame)
      (warpAt cta warp dest lanes ∗ frame) := by
  intro st r st' hpre hstep
  have hdest' := hdest st r hpre
  rcases hpre with ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hcurrent :
      List.filter
          (fun lane =>
            match warpState.getLane? lane with
            | some laneState => laneState.pc == pc
            | none => false)
          (Helpers.runnableLaneIds warpState) =
        lanes := by
    exact (Helpers.runnableLaneIds_filter_current_pc_eq hlock hrpc).trans hlanesStart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have happly :
      Helpers.applyToLaneIds? st cta warp lanes
          (fun _ laneState => some { laneState with pc := dest }) =
        some st' := by
    have hstepApply := hstep
    unfold Helpers.stepTerminator? at hstepApply
    simp [hwarp, hlockBool, hrpcOpt] at hstepApply
    have hstepApplyLanes := hcurrent ▸ hstepApply
    simpa [hdest'] using hstepApplyLanes
  have hpostCtrl : warpAt cta warp dest lanes st' CSL.Resource.empty :=
    warpAt_lanes_of_set_pc hwarp hlock hrpc hpart happly
  have hframeFinal : frame st' rFrame :=
    hframe st st' r rFrame
      ⟨rCtrl, rFrame, hcomp, hequiv, hctrl, hframeSt⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rFrame, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _)
        (CSL.Resource.update_refl _))
  · exact ⟨CSL.Resource.empty, rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hpostCtrl, hframeFinal⟩

theorem cbrSpec_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc dest : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lanes : List LaneId} {frame : CSL.Assertion}
    (hdest :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ frame) st r →
          Helpers.uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some dest)
    (hframe :
      CSL.StableUnder (TerminatorStep cta warp (.cbr cond tLabel fLabel)) frame) :
    TerminatorSpec cta warp (.cbr cond tLabel fLabel)
      (warpAt cta warp pc lanes ∗ frame)
      (warpAt cta warp dest lanes ∗ frame) :=
  cbrSpec_lanes_warpAt_frame hdest (by
    intro st st' _r rFrame _hpre hframeSt hstep
    exact hframe st st' rFrame hstep hframeSt)

theorem terminateSpec_of_computed
    {cta : CTAId} {warp : WarpId}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep : Helpers.stepTerminator? st₀ cta warp .terminate = some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    TerminatorSpec cta warp .terminate
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  TerminatorSpec.of_computed hstep hpost

theorem terminateSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {lane : LaneId} :
    TerminatorSpec cta warp .terminate
      (warpAt cta warp pc [lane])
      (laneTerminatedAt cta warp lane pc) := by
  intro st r st' hpre hstep
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
    rw [hlanes]
    simp
  rcases (by
      unfold Helpers.runnableLaneIds at hmemRunnable
      simp [Helpers.laneIsRunnable] at hmemRunnable
      cases hget : warpState.getLane? lane with
      | none =>
          simp [hget] at hmemRunnable
      | some laneState =>
          exact ⟨laneState, rfl⟩ :
      ∃ laneState, warpState.getLane? lane = some laneState) with
    ⟨laneState, hwarpLane⟩
  have hlane : st.getLane? cta warp lane = some laneState := by
    unfold State.getLane?
    simp [hwarp, hwarpLane]
  have hpcLane : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have hset :
      st.setLane cta warp lane { laneState with status := .terminated } = some st' := by
    have hstepSet := hstep
    unfold Helpers.stepTerminator? at hstepSet
    simp [hwarp, hlockBool, hrpcOpt, hlanes, hwarpLane, hpcLane] at hstepSet
    unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at hstepSet
    simp [hlane] at hstepSet
    simpa [Helpers.applyToLaneIdsList?] using hstepSet
  have hpost : laneTerminatedAt cta warp lane pc st' CSL.Resource.empty := by
    have hlaneFinal :
        st'.getLane? cta warp lane = some { laneState with status := .terminated } :=
      State.getLane?_setLane_same hlane hset
    exact ⟨⟨{ laneState with status := .terminated }, hlaneFinal, by simp, by simp [hpcLane]⟩, rfl⟩
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem terminateSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId} :
    TerminatorSpec cta warp .terminate
      (warpAt cta warp pc lanes)
      (lanesTerminatedAt cta warp lanes pc) := by
  intro st r st' hpre hstep
  rcases warpAt_state hpre with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hcurrent :
      List.filter
          (fun lane =>
            match warpState.getLane? lane with
            | some laneState => laneState.pc == pc
            | none => false)
          (Helpers.runnableLaneIds warpState) =
        lanes := by
    exact (Helpers.runnableLaneIds_filter_current_pc_eq hlock hrpc).trans hlanesStart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hrpcOpt : Helpers.currentRunnablePc? warpState = some pc := hrpc
  have happly :
      Helpers.applyToLaneIds? st cta warp lanes
          (fun _ laneState => some { laneState with status := .terminated }) =
        some st' := by
    have hstepApply := hstep
    unfold Helpers.stepTerminator? at hstepApply
    simp [hwarp, hlockBool, hrpcOpt] at hstepApply
    exact hcurrent ▸ hstepApply
  have hpost : lanesTerminatedAt cta warp lanes pc st' CSL.Resource.empty :=
    lanesTerminatedAt_of_set_terminated hwarp hlock hrpc hpart happly
  have hemp : CSL.emp st r := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, hpost⟩

theorem wp_br_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel} {lane : LaneId} :
    warpAt cta warp pc [lane] ⊢ₛ
      wpTerminator cta warp (.br target)
        (warpAt cta warp (target, 0) [lane]) :=
  wpTerminator_of_spec brSpec_single_warpAt

theorem wp_br_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel} {lane : LaneId}
    {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep cta warp (.br target)) frame) :
    (warpAt cta warp pc [lane] ∗ frame) ⊢ₛ
      wpTerminator cta warp (.br target)
        (warpAt cta warp (target, 0) [lane] ∗ frame) :=
  wpTerminator_of_spec (brSpec_single_warpAt_stableFrame hframe)

theorem wp_br_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel}
    {lanes : List LaneId} :
    warpAt cta warp pc lanes ⊢ₛ
      wpTerminator cta warp (.br target)
        (warpAt cta warp (target, 0) lanes) :=
  wpTerminator_of_spec brSpec_lanes_warpAt

theorem wp_br_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {target : BlockLabel}
    {lanes : List LaneId} {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep cta warp (.br target)) frame) :
    (warpAt cta warp pc lanes ∗ frame) ⊢ₛ
      wpTerminator cta warp (.br target)
        (warpAt cta warp (target, 0) lanes ∗ frame) :=
  wpTerminator_frame_of_entails wp_br_lanes_warpAt hframe

theorem wp_cbr_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lane : LaneId} {value : Value} {takeTrue : Bool}
    (heval :
      ∀ st r,
        warpAt cta warp pc [lane] st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue) :
    warpAt cta warp pc [lane] ⊢ₛ
      wpTerminator cta warp (.cbr cond tLabel fLabel)
        (warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane]) :=
  wpTerminator_of_spec (cbrSpec_single_warpAt heval hbool)

theorem wp_cbr_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lane : LaneId} {value : Value} {takeTrue : Bool}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ frame) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } cond value)
    (hbool : Helpers.valueToBool? value = some takeTrue)
    (hframe :
      CSL.StableUnder (TerminatorStep cta warp (.cbr cond tLabel fLabel)) frame) :
    (warpAt cta warp pc [lane] ∗ frame) ⊢ₛ
      wpTerminator cta warp (.cbr cond tLabel fLabel)
        (warpAt cta warp (if takeTrue then (tLabel, 0) else (fLabel, 0)) [lane] ∗ frame) :=
  wpTerminator_of_spec (cbrSpec_single_warpAt_stableFrame heval hbool hframe)

theorem wp_cbr_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc dest : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lanes : List LaneId}
    (hdest :
      ∀ st r,
        warpAt cta warp pc lanes st r →
          Helpers.uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some dest) :
    warpAt cta warp pc lanes ⊢ₛ
      wpTerminator cta warp (.cbr cond tLabel fLabel)
        (warpAt cta warp dest lanes) :=
  wpTerminator_of_spec (cbrSpec_lanes_warpAt hdest)

theorem wp_cbr_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc dest : PC} {cond : RValue}
    {tLabel fLabel : BlockLabel} {lanes : List LaneId} {frame : CSL.Assertion}
    (hdest :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ frame) st r →
          Helpers.uniformBranchDestination? st cta warp lanes cond tLabel fLabel = some dest)
    (hframe :
      CSL.StableUnder (TerminatorStep cta warp (.cbr cond tLabel fLabel)) frame) :
    (warpAt cta warp pc lanes ∗ frame) ⊢ₛ
      wpTerminator cta warp (.cbr cond tLabel fLabel)
        (warpAt cta warp dest lanes ∗ frame) :=
  wpTerminator_of_spec (cbrSpec_lanes_warpAt_stableFrame hdest hframe)

theorem CbrBranchControl.of_wp_warpAt
    {cta : CTAId} {warp : WarpId} {cond : RValue}
    {tLabel fLabel target : BlockLabel} {lanes : List LaneId}
    {pre frame : CSL.Assertion}
    (hwp :
      pre ⊢ₛ
        wpTerminator cta warp (.cbr cond tLabel fLabel)
          (warpAt cta warp (target, 0) lanes ∗ frame)) :
    CbrBranchControl cta warp cond tLabel fLabel target pre := by
  intro st st' r hpre hstep
  rcases hwp st r hpre st' hstep with ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem wp_terminate_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {lane : LaneId} :
    warpAt cta warp pc [lane] ⊢ₛ
      wpTerminator cta warp .terminate
        (laneTerminatedAt cta warp lane pc) :=
  wpTerminator_of_spec terminateSpec_single_warpAt

theorem wp_terminate_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {lane : LaneId}
    {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep cta warp .terminate) frame) :
    (warpAt cta warp pc [lane] ∗ frame) ⊢ₛ
      wpTerminator cta warp .terminate
        (laneTerminatedAt cta warp lane pc ∗ frame) :=
  wpTerminator_frame_of_entails wp_terminate_single_warpAt hframe

theorem wp_terminate_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId} :
    warpAt cta warp pc lanes ⊢ₛ
      wpTerminator cta warp .terminate
        (lanesTerminatedAt cta warp lanes pc) :=
  wpTerminator_of_spec terminateSpec_lanes_warpAt

theorem wp_terminate_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {lanes : List LaneId}
    {frame : CSL.Assertion}
    (hframe : CSL.StableUnder (TerminatorStep cta warp .terminate) frame) :
    (warpAt cta warp pc lanes ∗ frame) ⊢ₛ
      wpTerminator cta warp .terminate
        (lanesTerminatedAt cta warp lanes pc ∗ frame) :=
  wpTerminator_frame_of_entails wp_terminate_lanes_warpAt hframe

theorem stable_globalBytes_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder (TerminatorStep cta warp term) (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepTerminator?_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_reg_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lane : LaneId} {name : RegName} {value : Value} :
    CSL.StableUnder (TerminatorStep cta warp term) (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  rcases Helpers.stepTerminator?_lane_nonPc_eq hlane hstep with
    ⟨laneState', hlane', _hlocal, hregs, _hpreds⟩
  exact ⟨howns, laneState', hlane', by simpa [hregs] using hread⟩

theorem stable_pred_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lane : LaneId} {name : PredName} {value : Bool} :
    CSL.StableUnder (TerminatorStep cta warp term) (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  rcases Helpers.stepTerminator?_lane_nonPc_eq hlane hstep with
    ⟨laneState', hlane', _hlocal, _hregs, hpreds⟩
  exact ⟨howns, laneState', hlane', by simpa [hpreds] using hread⟩

theorem stable_localByte_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lane : LaneId} {offset : Nat} {perm : CSL.BytePerm} {value : Byte} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (CSL.localByte cta warp lane offset perm value) := by
  intro st st' r hstep hbyte
  rcases hbyte with ⟨howns, laneState, hlane, hmem⟩
  rcases Helpers.stepTerminator?_lane_nonPc_eq hlane hstep with
    ⟨laneState', hlane', hlocal, _hregs, _hpreds⟩
  exact ⟨howns, laneState', hlane', by simpa [hlocal] using hmem⟩

theorem stable_localBytes_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lane : LaneId} {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (CSL.localBytes cta warp lane offset perm bytes) := by
  induction bytes generalizing offset with
  | nil =>
      simpa [CSL.localBytes] using
        (CSL.stable_emp (step := TerminatorStep cta warp term))
  | cons byte bytes ih =>
      simpa [CSL.localBytes] using
        CSL.stable_sep
          (stable_localByte_terminator
            (cta := cta) (warp := warp) (term := term)
            (lane := lane) (offset := offset) (perm := perm) (value := byte))
          (ih (offset := offset + 1))

theorem stable_regsFor_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using (CSL.stable_emp (step := TerminatorStep cta warp term))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
      | cons value values =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_terminator
                (cta := cta) (warp := warp) (term := term)
                (lane := lane) (name := name) (value := value))
              (ih (values := values))

theorem stable_predsFor_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lanes : List LaneId} {name : PredName} {values : List Bool} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (predsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [predsFor] using (CSL.stable_emp (step := TerminatorStep cta warp term))
      | cons _ _ =>
          simpa [predsFor] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [predsFor] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
      | cons value values =>
          simpa [predsFor] using
            CSL.stable_sep
                (stable_pred_terminator
                  (cta := cta) (warp := warp) (term := term)
                  (lane := lane) (name := name) (value := value))
                (ih (values := values))

theorem stable_predsFor_of_stable_pred
    {step : State → State → Prop} {cta : CTAId} {warp : WarpId}
    {lanes : List LaneId} {name : PredName} {values : List Bool}
    (hpred :
      ∀ lane value, CSL.StableUnder step (CSL.pred cta warp lane name value)) :
    CSL.StableUnder step (predsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [predsFor] using (CSL.stable_emp (step := step))
      | cons _ _ =>
          simpa [predsFor] using (CSL.stable_pure (step := step) (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [predsFor] using (CSL.stable_pure (step := step) (p := False))
      | cons value values =>
          simpa [predsFor] using
            CSL.stable_sep (hpred lane value) (ih (values := values))

theorem stable_globalSlices_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {offsets : List Nat} {perm : CSL.BytePerm} {slices : List (List Byte)} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp (step := TerminatorStep cta warp term))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_terminator
                (cta := cta) (warp := warp) (term := term)
                (offset := offset) (perm := perm) (bytes := bytes))
              (ih (slices := rest))

theorem stable_localSlices_terminator
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {lanes : List LaneId} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder (TerminatorStep cta warp term)
      (localSlices cta warp lanes offsets perm slices) := by
  induction lanes generalizing offsets slices with
  | nil =>
      cases offsets <;> cases slices <;>
        first
        | simpa [localSlices] using
            (CSL.stable_emp (step := TerminatorStep cta warp term))
        | simpa [localSlices] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
  | cons lane lanes ih =>
      cases offsets with
      | nil =>
          simpa [localSlices] using
            (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
      | cons offset offsets =>
          cases slices with
          | nil =>
              simpa [localSlices] using
                (CSL.stable_pure (step := TerminatorStep cta warp term) (p := False))
          | cons bytes rest =>
              simpa [localSlices] using
                CSL.stable_sep
                  (stable_localBytes_terminator
                    (cta := cta) (warp := warp) (term := term)
                    (lane := lane) (offset := offset) (perm := perm) (bytes := bytes))
                  (ih (offsets := offsets) (slices := rest))

theorem stable_globalBytes_assignReg
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_assignReg_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalSlices_assignReg
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_assignReg
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (rhs := rhs) (offset := offset) (perm := perm) (bytes := bytes))
              (ih (slices := rest))

theorem stable_globalBytes_load
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_load_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalSlices_load
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_load
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (src := src) (offset := offset) (perm := perm) (bytes := bytes))
              (ih (slices := rest))

theorem stable_globalBytes_assignPred
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_assignPred_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalBytes_assignPredValue
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_assignPredValue_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalBytes_cvta
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {offset : Nat} {perm : CSL.BytePerm}
    {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_cvta_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalBytes_isspacep
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {offset : Nat} {perm : CSL.BytePerm}
    {bytes : List Byte} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (CSL.globalBytes offset perm bytes) := by
  intro st st' r hstep hbytes
  exact CSL.globalBytes_of_memory hbytes (by
    rw [Helpers.stepInstr?_isspacep_global_eq hstep]
    exact CSL.globalBytes_memory hbytes)

theorem stable_globalSlices_assignPred
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_assignPred
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (cmp := cmp) (offset := offset) (perm := perm) (bytes := bytes))
              (ih (slices := rest))

theorem stable_globalSlices_assignPredValue
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_assignPredValue
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (rhs := rhs) (offset := offset) (perm := perm) (bytes := bytes))
              (ih (slices := rest))

theorem stable_globalSlices_cvta
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_cvta
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (space := space) (src := src) (offset := offset) (perm := perm)
                (bytes := bytes))
              (ih (slices := rest))

theorem stable_globalSlices_isspacep
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {offsets : List Nat} {perm : CSL.BytePerm}
    {slices : List (List Byte)} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (globalSlices offsets perm slices) := by
  induction offsets generalizing slices with
  | nil =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_emp
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src }))
      | cons _ _ =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
              (p := False))
  | cons offset offsets ih =>
      cases slices with
      | nil =>
          simpa [globalSlices] using
            (CSL.stable_pure
              (step :=
                InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
              (p := False))
      | cons bytes rest =>
          simpa [globalSlices] using
            CSL.stable_sep
              (stable_globalBytes_isspacep
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (space := space) (src := src) (offset := offset) (perm := perm)
                (bytes := bytes))
              (ih (slices := rest))

theorem stable_reg_assignReg_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {rhs : RValue} {lane : LaneId} {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
                        some (Helpers.writeReg laneState dst v)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane rhs with
                        | none =>
                            simp [heval] at hf
                        | some v =>
                            simp [heval] at hf
                            subst new
                            have hdstNe : dst ≠ name := fun h => hne h.symm
                            have hbeq : (dst == name) = false :=
                              (beq_eq_false_iff_ne).2 hdstNe
                            simp [Helpers.writeReg]
                            rw [Std.HashMap.getElem?_insert]
                            simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_reg_load_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {src : TypedAddr} {lane : LaneId} {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.resolveAddr? st cta warp lane src).bind fun addr =>
                        (Helpers.readMem? st src.space src.ty addr).bind fun loaded =>
                          some (Helpers.writeReg laneState dst loaded)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases haddr : Helpers.resolveAddr? st cta warp lane src with
                        | none =>
                            simp [haddr] at hf
                        | some addr =>
                            simp [haddr] at hf
                            cases hreadMem : Helpers.readMem? st src.space src.ty addr with
                            | none =>
                                simp [hreadMem] at hf
                            | some loaded =>
                                simp [hreadMem] at hf
                                subst new
                                have hdstNe : dst ≠ name := fun h => hne h.symm
                                have hbeq : (dst == name) = false :=
                                  (beq_eq_false_iff_ne).2 hdstNe
                                simp [Helpers.writeReg]
                                rw [Std.HashMap.getElem?_insert]
                                simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_regsFor_assignReg_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {rhs : RValue} {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_assignReg_of_ne
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (name := name) (rhs := rhs) (lane := lane) (value := value) hne)
              (ih (values := rest))

theorem stable_regsFor_load_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {src : TypedAddr} {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .load dst src })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_load_of_ne
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                  (name := name) (src := src) (lane := lane) (value := value) hne)
                (ih (values := rest))

theorem stable_pred_assignReg
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {name : PredName} {value : Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
                        some (Helpers.writeReg laneState dst v)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane rhs with
                        | none =>
                            simp [heval] at hf
                        | some v =>
                            simp [heval] at hf
                            subst new
                            simpa [Helpers.writeReg] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_pred_load
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {lane : LaneId} {name : PredName} {value : Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.resolveAddr? st cta warp lane src).bind fun addr =>
                        (Helpers.readMem? st src.space src.ty addr).bind fun loaded =>
                          some (Helpers.writeReg laneState dst loaded)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases haddr : Helpers.resolveAddr? st cta warp lane src with
                        | none =>
                            simp [haddr] at hf
                        | some addr =>
                            simp [haddr] at hf
                            cases hreadMem : Helpers.readMem? st src.space src.ty addr with
                            | none =>
                                simp [hreadMem] at hf
                            | some loaded =>
                                simp [hreadMem] at hf
                                subst new
                                simpa [Helpers.writeReg] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_reg_assignPred
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalCmp? st cta warp lane cmp).bind fun b =>
                        some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalCmp? st cta warp lane cmp with
                        | none =>
                            simp [heval] at hf
                        | some b =>
                            simp [heval] at hf
                            subst new
                            simpa [Helpers.writePred] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_reg_assignPredValue
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane rhs).bind fun value =>
                        (Helpers.valueToBool? value).bind fun b =>
                          some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane rhs with
                        | none =>
                            simp [heval] at hf
                        | some value =>
                            simp [heval] at hf
                            cases hbool : Helpers.valueToBool? value with
                            | none =>
                                simp [hbool] at hf
                            | some b =>
                                simp [hbool] at hf
                                subst new
                                simpa [Helpers.writePred] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_pred_assignPred_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {cmp : CmpExpr} {lane : LaneId} {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalCmp? st cta warp lane cmp).bind fun b =>
                        some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalCmp? st cta warp lane cmp with
                        | none =>
                            simp [heval] at hf
                        | some b =>
                            simp [heval] at hf
                            subst new
                            have hdstNe : dst ≠ name := fun h => hne h.symm
                            have hbeq : (dst == name) = false :=
                              (beq_eq_false_iff_ne).2 hdstNe
                            simp [Helpers.writePred]
                            rw [Std.HashMap.getElem?_insert]
                            simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_pred_assignPredValue_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {rhs : RValue} {lane : LaneId} {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane rhs).bind fun value =>
                        (Helpers.valueToBool? value).bind fun b =>
                          some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane rhs with
                        | none =>
                            simp [heval] at hf
                        | some value =>
                            simp [heval] at hf
                            cases hbool : Helpers.valueToBool? value with
                            | none =>
                                simp [hbool] at hf
                            | some b =>
                                simp [hbool] at hf
                                subst new
                                have hdstNe : dst ≠ name := fun h => hne h.symm
                                have hbeq : (dst == name) = false :=
                                  (beq_eq_false_iff_ne).2 hdstNe
                                simp [Helpers.writePred]
                                rw [Std.HashMap.getElem?_insert]
                                simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_regsFor_assignPred
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_assignPred
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (cmp := cmp) (lane := lane) (name := name) (value := value))
              (ih (values := rest))

theorem stable_regsFor_assignPredValue
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {name : RegName} {values : List Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp
                { guard? := guard?, instr := .assignPredValue dst rhs }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp
                { guard? := guard?, instr := .assignPredValue dst rhs })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp
                { guard? := guard?, instr := .assignPredValue dst rhs })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_assignPredValue
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (rhs := rhs) (lane := lane) (name := name) (value := value))
              (ih (values := rest))

theorem stable_reg_cvta_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {value : Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane src).bind fun value =>
                        (Helpers.evalCvta? space value).bind fun gaddr =>
                          some (Helpers.writeReg laneState dst gaddr)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane src with
                        | none =>
                            simp [heval] at hf
                        | some srcValue =>
                            simp [heval] at hf
                            cases hcvta : Helpers.evalCvta? space srcValue with
                            | none =>
                                simp [hcvta] at hf
                            | some gaddr =>
                                simp [hcvta] at hf
                                subst new
                                have hdstNe : dst ≠ name := fun h => hne h.symm
                                have hbeq : (dst == name) = false :=
                                  (beq_eq_false_iff_ne).2 hdstNe
                                simp [Helpers.writeReg]
                                rw [Std.HashMap.getElem?_insert]
                                simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_reg_isspacep
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {name : RegName} {value : Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (CSL.reg cta warp lane name value) := by
  intro st st' r hstep hreg
  rcases hreg with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane src).bind fun value =>
                        (Helpers.evalIsspacep? space value).bind fun b =>
                          some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_reg_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane src with
                        | none =>
                            simp [heval] at hf
                        | some srcValue =>
                            simp [heval] at hf
                            cases hisspace : Helpers.evalIsspacep? space srcValue with
                            | none =>
                                simp [hisspace] at hf
                            | some b =>
                                simp [hisspace] at hf
                                subst new
                                simpa [Helpers.writePred] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hregs] using hreadCore⟩

theorem stable_pred_cvta
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {name : PredName}
    {value : Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane src).bind fun value =>
                        (Helpers.evalCvta? space value).bind fun gaddr =>
                          some (Helpers.writeReg laneState dst gaddr)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane src with
                        | none =>
                            simp [heval] at hf
                        | some srcValue =>
                            simp [heval] at hf
                            cases hcvta : Helpers.evalCvta? space srcValue with
                            | none =>
                                simp [hcvta] at hf
                            | some gaddr =>
                                simp [hcvta] at hf
                                subst new
                                simpa [Helpers.writeReg] using hreadOld)
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_pred_isspacep_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {value : Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (CSL.pred cta warp lane name value) := by
  intro st st' r hstep hpred
  rcases hpred with ⟨howns, laneState, hlane, hread⟩
  unfold InstrStep at hstep
  unfold Helpers.stepInstr? at hstep
  cases hwarp : st.getWarp? cta warp with
  | none =>
      simp [hwarp] at hstep
  | some warpState =>
      simp [hwarp] at hstep
      cases hlock : Helpers.lockstepRunnable? warpState with
      | false =>
          simp [hlock] at hstep
      | true =>
          simp [hlock] at hstep
          cases hpart : Helpers.participatingRunnableLaneIds? warpState guard? with
          | none =>
              simp [hpart] at hstep
          | some participants =>
              simp [hpart] at hstep
              cases hcore :
                  Helpers.applyToLaneIds? st cta warp participants
                    (fun lane laneState =>
                      (Helpers.evalRValue? st cta warp lane src).bind fun value =>
                        (Helpers.evalIsspacep? space value).bind fun b =>
                          some (Helpers.writePred laneState dst b)) with
              | none =>
                  simp [hcore] at hstep
              | some stCore =>
                  simp [hcore] at hstep
                  rcases Helpers.applyToLaneIds?_lane_pred_eq
                      (hpres := by
                        intro lane old new hf hreadOld
                        cases heval : Helpers.evalRValue? st cta warp lane src with
                        | none =>
                            simp [heval] at hf
                        | some srcValue =>
                            simp [heval] at hf
                            cases hisspace : Helpers.evalIsspacep? space srcValue with
                            | none =>
                                simp [hisspace] at hf
                            | some b =>
                                simp [hisspace] at hf
                                subst new
                                have hdstNe : dst ≠ name := fun h => hne h.symm
                                have hbeq : (dst == name) = false :=
                                  (beq_eq_false_iff_ne).2 hdstNe
                                simp [Helpers.writePred]
                                rw [Std.HashMap.getElem?_insert]
                                simp [hbeq, hreadOld])
                      hlane hread hcore with
                    ⟨laneCore, hlaneCore, hreadCore⟩
                  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                    ⟨laneFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
                  exact ⟨howns, laneFinal, hlaneFinal, by
                    simpa [hpreds] using hreadCore⟩

theorem stable_regsFor_cvta_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : RegName}
    {space : AddrSpace} {src : RValue} {lanes : List LaneId} {values : List Value}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
              (stable_reg_cvta_of_ne
                (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                (name := name) (space := space) (src := src) (lane := lane)
                (value := value) hne)
              (ih (values := rest))

theorem stable_regsFor_isspacep
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lanes : List LaneId} {name : RegName}
    {values : List Value} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (regsFor cta warp lanes name values) := by
  induction lanes generalizing values with
  | nil =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_emp
              (step := InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src }))
      | cons _ _ =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
              (p := False))
  | cons lane lanes ih =>
      cases values with
      | nil =>
          simpa [regsFor] using
            (CSL.stable_pure
              (step := InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
              (p := False))
      | cons value rest =>
          simpa [regsFor] using
            CSL.stable_sep
                (stable_reg_isspacep
                  (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
                  (space := space) (src := src) (lane := lane) (name := name)
                  (value := value))
                (ih (values := rest))

theorem stable_predsFor_assignReg
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {lanes : List LaneId} {name : PredName} {values : List Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignReg dst rhs })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_assignReg
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (rhs := rhs) (lane := lane) (name := name) (value := value))

theorem stable_predsFor_load
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {lanes : List LaneId} {name : PredName} {values : List Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .load dst src })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_load
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (src := src) (lane := lane) (name := name) (value := value))

theorem stable_predsFor_assignPred_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {cmp : CmpExpr} {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPred dst cmp })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_assignPred_of_ne
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (name := name) (cmp := cmp) (lane := lane) (value := value) hne)

theorem stable_predsFor_assignPredValue_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {rhs : RValue} {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .assignPredValue dst rhs })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_assignPredValue_of_ne
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (name := name) (rhs := rhs) (lane := lane) (value := value) hne)

theorem stable_predsFor_cvta
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lanes : List LaneId} {name : PredName}
    {values : List Bool} :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .cvta dst space src })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_cvta
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (space := space) (src := src) (lane := lane) (name := name)
      (value := value))

theorem stable_predsFor_isspacep_of_ne
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst name : PredName}
    {space : AddrSpace} {src : RValue} {lanes : List LaneId} {values : List Bool}
    (hne : name ≠ dst) :
    CSL.StableUnder
      (InstrStep cta warp { guard? := guard?, instr := .isspacep dst space src })
      (predsFor cta warp lanes name values) :=
  stable_predsFor_of_stable_pred (by
    intro lane value
    exact stable_pred_isspacep_of_ne
      (cta := cta) (warp := warp) (guard? := guard?) (dst := dst)
      (name := name) (space := space) (src := src) (lane := lane)
      (value := value) hne)

theorem assignRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .assignReg dst rhs } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .assignReg dst rhs }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem assignRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {rhs : RValue} {st₀ : State} {warpState : WarpState} {lane : LaneId}
    {laneState : LaneState} {old new : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := guard?, instr := .assignReg dst rhs }
      (fun st r => st = st₀ ∧ CSL.reg cta warp lane dst old st r)
      (CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hreg⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st₀ cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst new) := by
        refine ⟨laneState, ?_⟩
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
        unfold EvalRValue at heval
        simp [hlane, heval] at happly
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst new) =
            some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happly
        exact State.getLane?_setLane_same hlane hset
      exact StateResourceUpdate.reg_written_advanced hcore hstep r hreg

theorem assignRegSpec_single_warpAt_of_eval
    {cta : CTAId} {warp : WarpId} {pc nextPc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new)
    (hcontrol :
      ∀ st st',
        warpAt cta warp pc [lane] st CSL.Resource.empty →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
            some st' →
            warpAt cta warp nextPc [lane] st' CSL.Resource.empty) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old)
      (warpAt cta warp nextPc [lane] ∗ CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rReg, hcomp, hequiv, hctrl, hreg⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  rcases assignRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hctrl' := hcontrol st st' hctrl hstep
  refine ⟨CSL.Resource.compose CSL.Resource.empty rReg', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateReg)
  · exact ⟨CSL.Resource.empty, rReg',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrl', hreg'⟩

theorem assignRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rReg, hcomp, hequiv, hctrl, hreg⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  rcases assignRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
              some (Helpers.writeReg laneState dst v)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst new) = some stCore := by
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
          unfold EvalRValue at heval'
          simp [hlane, heval'] at happly
          simpa [Helpers.applyToLaneIdsList?] using happly
        have hwarpCore :
            stCore.getWarp? cta warp =
              some (warpState.setLane lane (Helpers.writeReg laneState dst new)) :=
          State.getWarp?_setLane_same hwarp hset
        have hlaneCore :
            stCore.getLane? cta warp lane =
              some (Helpers.writeReg laneState dst new) :=
          State.getLane?_setLane_same hlane hset
        have hpartCore :
            Helpers.ParticipatingRunnable
              (warpState.setLane lane (Helpers.writeReg laneState dst new)) none [lane] :=
          Helpers.ParticipatingRunnable.setLane_none_control_eq hpart hwarpLane
            (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg])
        have hlockCore :
            Helpers.lockstepRunnable
              (warpState.setLane lane (Helpers.writeReg laneState dst new)) :=
          (Helpers.lockstepRunnable_setLane_control_iff hwarpLane
            (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg])).2 hlock
        have hrpcCore :
            Helpers.RunnablePc
              (warpState.setLane lane (Helpers.writeReg laneState dst new)) pc := by
          unfold Helpers.RunnablePc
          rw [Helpers.currentRunnablePc?_setLane_control_eq hwarpLane
            (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg])]
          exact hrpc
        rcases Helpers.advanceRunnablePcs?_single_warp_control
            hwarpCore hlockCore hrpcCore hpartCore hlaneCore hstep with
          ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩
        exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rReg', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateReg)
  · exact ⟨CSL.Resource.empty, rReg',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hreg'⟩

theorem assignRegSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗ frame)) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rReg, rFrame, hcompRest, hequivRest, hreg, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  rcases assignRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
              some (Helpers.writeReg laneState dst v)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst new) = some stCore := by
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
          unfold EvalRValue at heval'
          simp [hlane, heval'] at happly
          simpa [Helpers.applyToLaneIdsList?] using happly
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rReg, rFrame, hcompRest, hequivRest, hreg, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rReg' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateReg (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rReg' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rReg' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rReg', rFrame, CSL.Resource.canCompose_update_left hupdateReg hcompRest,
        CSL.Resource.equiv_refl _, hreg', hframeFinal⟩⟩

theorem assignRegSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignReg dst rhs }) frame) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗ frame)) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem assignRegPreservesReadReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {rhs : RValue} {lane : LaneId} {old srcValue : Value}
    {st st' : State} {rCtrl rDst rSrc : CSL.Resource}
    (hne : src ≠ dst)
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hdst : CSL.reg cta warp lane dst old st rDst)
    (hsrc : CSL.reg cta warp lane src srcValue st rSrc)
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
        some st') :
    CSL.reg cta warp lane src srcValue st' rSrc := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  rcases CSL.reg_state hdst with ⟨dstLaneState, hdstLane, _hdstRead⟩
  rcases hsrc with ⟨hsrcOwns, srcLaneState, hsrcLane, hsrcRead⟩
  have hsameLane : srcLaneState = dstLaneState := by
    rw [hdstLane] at hsrcLane
    exact (Option.some.inj hsrcLane).symm
  subst srcLaneState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
      simp [hdstLane] at happly
      cases hevalRhs : Helpers.evalRValue? st cta warp lane rhs with
      | none =>
          simp [hevalRhs] at happly
      | some rhsValue =>
          simp [hevalRhs] at happly
          have hset :
              st.setLane cta warp lane (Helpers.writeReg dstLaneState dst rhsValue) =
                some stCore := by
            simpa [Helpers.applyToLaneIdsList?] using happly
          have hlaneCore :
              stCore.getLane? cta warp lane =
                some (Helpers.writeReg dstLaneState dst rhsValue) :=
            State.getLane?_setLane_same hdstLane hset
          rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
            ⟨laneStateFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
          have hneDstSrc : dst ≠ src := fun h => hne h.symm
          have hbeq : (dst == src) = false :=
            (beq_eq_false_iff_ne).2 hneDstSrc
          have hreadCore :
              (Helpers.writeReg dstLaneState dst rhsValue).regs[src]? = some srcValue := by
            simp [Helpers.writeReg]
            rw [Std.HashMap.getElem?_insert]
            simp [hbeq, hsrcRead]
          exact ⟨hsrcOwns, laneStateFinal, hlaneFinal, by
            simpa [hregs] using hreadCore⟩

theorem assignRegPreservesGlobalBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte}
    {st st' : State} {rCtrl rMem : CSL.Resource}
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hbytes : CSL.globalBytes offset perm bytes st rMem)
    (hstep :
      Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
        some st') :
    CSL.globalBytes offset perm bytes st' rMem := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hglobalCore : stCore.global = st.global :=
        Helpers.applyToLaneIds?_global_eq happly
      have hglobalFinal : st'.global = stCore.global :=
        Helpers.advanceRunnablePcs?_global_eq hstep
      have hglobal : st'.global = st.global := hglobalFinal.trans hglobalCore
      exact CSL.globalBytes_of_memory hbytes (by
        rw [hglobal]
        exact CSL.globalBytes_memory hbytes)

theorem assignRegSpec_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue : Value}
    (hne : src ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ CSL.reg cta warp lane src srcValue)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗ CSL.reg cta warp lane src srcValue))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗ CSL.reg cta warp lane src srcValue)) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hsrc hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rSrc, _hcompRest, _hequivRest, hdst, _hsrcFromPre⟩
    exact assignRegPreservesReadReg_single_warpAt hne hctrl hdst hsrc hstep)

theorem assignRegSpec_single_warpAt_readRegs2
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              CSL.reg cta warp lane src₂ srcValue₂))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            CSL.reg cta warp lane src₂ srcValue₂)))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            CSL.reg cta warp lane src₂ srcValue₂))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rSrc₂, hcompSrc, hequivSrc, hsrc₁, hsrc₂⟩
    exact ⟨rSrc₁, rSrc₂, hcompSrc, hequivSrc,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep⟩)

theorem assignRegSpec_single_warpAt_readRegs2_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes offset perm bytes)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.globalBytes offset perm bytes))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.globalBytes offset perm bytes)))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂, hsrc₂, hbytes⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂,
        assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep,
        assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes hstep⟩⟩)

theorem assignRegSpec_single_warpAt_readRegs2_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset₁ offset₂ : Nat} {perm₁ perm₂ : CSL.BytePerm}
    {bytes₁ bytes₂ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  CSL.globalBytes offset₂ perm₂ bytes₂))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                CSL.globalBytes offset₂ perm₂ bytes₂)))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                CSL.globalBytes offset₂ perm₂ bytes₂))))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂, hsrc₂, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes, hbytes₁, hbytes₂⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂,
        assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep,
        ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
          assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₁ hstep,
          assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₂ hstep⟩⟩⟩)

theorem assignRegSpec_single_warpAt_readRegs2_globalBytes3Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset₁ offset₂ offset₃ : Nat} {perm₁ perm₂ perm₃ : CSL.BytePerm}
    {bytes₁ bytes₂ bytes₃ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                    CSL.globalBytes offset₃ perm₃ bytes₃)))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                  CSL.globalBytes offset₃ perm₃ bytes₃))))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                  CSL.globalBytes offset₃ perm₃ bytes₃)))))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂, hsrc₂, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rRestBytes, hcompBytes₁, hequivBytes₁,
      hbytes₁, hrestBytes⟩
    rcases hrestBytes with ⟨rBytes₂, rBytes₃, hcompBytes₂, hequivBytes₂,
      hbytes₂, hbytes₃⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂,
        assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep,
        ⟨rBytes₁, rRestBytes, hcompBytes₁, hequivBytes₁,
          assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₁ hstep,
          ⟨rBytes₂, rBytes₃, hcompBytes₂, hequivBytes₂,
            assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₂ hstep,
            assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₃ hstep⟩⟩⟩⟩)

theorem assignRegSpec_single_warpAt_readRegs3
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ srcValue₃ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.reg cta warp lane src₃ srcValue₃)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.reg cta warp lane src₃ srcValue₃))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.reg cta warp lane src₃ srcValue₃)))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestSrc⟩
    rcases hrestSrc with ⟨rSrc₂, rSrc₃, hcompSrc₂, hequivSrc₂, hsrc₂, hsrc₃⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      ⟨rSrc₂, rSrc₃, hcompSrc₂, hequivSrc₂,
        assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep,
        assignRegPreservesReadReg_single_warpAt hne₃ hctrl hdst hsrc₃ hstep⟩⟩)

theorem assignRegSpec_single_warpAt_readRegs3_globalBytes3Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ srcValue₃ : Value}
    {offset₁ offset₂ offset₃ : Nat} {perm₁ perm₂ perm₃ : CSL.BytePerm}
    {bytes₁ bytes₂ bytes₃ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                    (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                      CSL.globalBytes offset₃ perm₃ bytes₃))))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.reg cta warp lane src₃ srcValue₃ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                    CSL.globalBytes offset₃ perm₃ bytes₃)))))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.reg cta warp lane src₃ srcValue₃ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                    CSL.globalBytes offset₃ perm₃ bytes₃))))))) :=
  assignRegSpec_single_warpAt_frame heval (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, hdst, _hframeFromPre⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestSrc⟩
    rcases hrestSrc with ⟨rSrc₂, rRest₂, hcompSrc₂, hequivSrc₂, hsrc₂, hrest₂⟩
    rcases hrest₂ with ⟨rSrc₃, rBytes, hcompSrc₃, hequivSrc₃, hsrc₃, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rRestBytes, hcompBytes₁, hequivBytes₁,
      hbytes₁, hrestBytes⟩
    rcases hrestBytes with ⟨rBytes₂, rBytes₃, hcompBytes₂, hequivBytes₂,
      hbytes₂, hbytes₃⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      assignRegPreservesReadReg_single_warpAt hne₁ hctrl hdst hsrc₁ hstep,
      ⟨rSrc₂, rRest₂, hcompSrc₂, hequivSrc₂,
        assignRegPreservesReadReg_single_warpAt hne₂ hctrl hdst hsrc₂ hstep,
        ⟨rSrc₃, rBytes, hcompSrc₃, hequivSrc₃,
          assignRegPreservesReadReg_single_warpAt hne₃ hctrl hdst hsrc₃ hstep,
          ⟨rBytes₁, rRestBytes, hcompBytes₁, hequivBytes₁,
            assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₁ hstep,
            ⟨rBytes₂, rBytes₃, hcompBytes₂, hequivBytes₂,
              assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₂ hstep,
              assignRegPreservesGlobalBytes_single_warpAt hctrl hbytes₃ hstep⟩⟩⟩⟩⟩)

theorem assignRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) st r →
          EvalRValuesFor st cta warp rhs lanes newValues) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        regsFor cta warp lanes dst newValues) := by
  intro st r st' hpre hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRegs, hcomp, hequiv, hctrl, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyAssignReg hnodup hevals' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some v =>
                simp [heval] at hf
                subst new
                simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some v =>
                simp [heval] at hf
                subst new
                simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      refine ⟨CSL.Resource.compose CSL.Resource.empty rRegs', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRegs)
      · exact ⟨CSL.Resource.empty, rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal, hregs'⟩

theorem assignRegSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalRValuesFor st cta warp rhs lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc lanes ∗
        (regsFor cta warp lanes dst oldValues ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (regsFor cta warp lanes dst newValues ∗ frame)) := by
  intro st r st' hpre hstep
  have hpreOrig := hpre
  have hstepOrig := hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rRegs, rFrame, hcompRest, hequivRest, hregs, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
            some (Helpers.writeReg laneState dst v)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyAssignReg hnodup hevals' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hframeFinal : frame st' rFrame :=
        hframe st st' r rFrame hpreOrig hframeSt hstepOrig
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some v =>
                simp [heval] at hf
                subst new
                simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some v =>
                simp [heval] at hf
                subst new
                simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rRegs' rFrame) :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateRegs (CSL.Resource.update_refl rFrame))
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rRegs' rFrame), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rRegs' rFrame,
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rRegs', rFrame, CSL.Resource.canCompose_update_left hupdateRegs hcompRest,
            CSL.Resource.equiv_refl _, hregs', hframeFinal⟩⟩

theorem assignRegSpec_single_laneRunning_of_eval
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    InstrSpec cta warp { guard? := none, instr := .assignReg dst rhs }
      (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old)
      (laneRunning cta warp lane (pc.1, pc.2 + 1) ∗
        CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rReg, hcomp, hequiv, hctrl, hreg⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  have hwarpLane : warpState.getLane? lane = some laneState := by
    unfold State.getLane? at hlane
    simp [hwarp] at hlane
    exact hlane
  have hpc : laneState.pc = pc :=
    Helpers.ParticipatingRunnable.single_lane_pc hrpc hpart hwarpLane
  have hstatus : laneState.status = .running :=
    Helpers.ParticipatingRunnable.single_lane_status hpart hwarpLane
  rcases assignRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hctrlFinal :
      laneRunning cta warp lane (pc.1, pc.2 + 1) st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane rhs).bind fun v =>
              some (Helpers.writeReg laneState dst v)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst new) = some stCore := by
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
          unfold EvalRValue at heval'
          simp [hlane, heval'] at happly
          simpa [Helpers.applyToLaneIdsList?] using happly
        have hwarpCore :
            stCore.getWarp? cta warp =
              some (warpState.setLane lane (Helpers.writeReg laneState dst new)) :=
          State.getWarp?_setLane_same hwarp hset
        have hlaneCore :
            stCore.getLane? cta warp lane =
              some (Helpers.writeReg laneState dst new) :=
          State.getLane?_setLane_same hlane hset
        have hpartCore :
            Helpers.ParticipatingRunnable
              (warpState.setLane lane (Helpers.writeReg laneState dst new)) none [lane] :=
          Helpers.ParticipatingRunnable.setLane_none_control_eq hpart hwarpLane
            (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg])
        have hpcCore : (Helpers.writeReg laneState dst new).pc = pc := by
          simp [Helpers.writeReg, hpc]
        rcases Helpers.advanceRunnablePcs?_single_lane_pc
            hwarpCore hpartCore hlaneCore hpcCore hstep with
          ⟨laneFinal, hlaneFinal, hpcFinal, hstatusFinal⟩
        exact ⟨⟨laneFinal, hlaneFinal,
          by simpa [Helpers.writeReg, hstatus] using hstatusFinal, hpcFinal⟩, rfl⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rReg', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateReg)
  · exact ⟨CSL.Resource.empty, rReg',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hreg'⟩

theorem assignPredSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .assignPred dst cmp } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .assignPred dst cmp }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem assignPredSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {cmp : CmpExpr} {st₀ : State} {warpState : WarpState} {lane : LaneId}
    {laneState : LaneState} {old new : Bool}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (heval : EvalCmp st₀ { cta := cta, warp := warp, lane := lane } cmp new) :
    InstrSpec cta warp { guard? := guard?, instr := .assignPred dst cmp }
      (fun st r => st = st₀ ∧ CSL.pred cta warp lane dst old st r)
      (CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hpred⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalCmp? st₀ cta warp lane cmp).bind fun b =>
            some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writePred laneState dst new) := by
        refine ⟨laneState, ?_⟩
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
        unfold EvalCmp at heval
        simp [hlane, heval] at happly
        have hset :
            st₀.setLane cta warp lane (Helpers.writePred laneState dst new) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happly
        exact State.getLane?_setLane_same hlane hset
      exact StateResourceUpdate.pred_written_advanced hcore hstep r hpred

theorem assignPredSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new) :
    InstrSpec cta warp { guard? := none, instr := .assignPred dst cmp }
      (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rPred, hcomp, hequiv, hctrl, hpred⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases assignPredSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (cmp := cmp)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalCmp? st cta warp lane cmp).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
          unfold EvalCmp at heval'
          simp [hlane, heval'] at happly
          simpa [Helpers.applyToLaneIdsList?] using happly
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rPred', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePred)
  · exact ⟨CSL.Resource.empty, rPred',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hpred'⟩

theorem assignPredSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignPred dst cmp } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .assignPred dst cmp }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases assignPredSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (cmp := cmp)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new)
      hwarp hlock hpart hlane heval' st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalCmp? st cta warp lane cmp).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
          unfold EvalCmp at heval'
          simp [hlane, heval'] at happly
          simpa [Helpers.applyToLaneIdsList?] using happly
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rPred' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdatePred (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rPred' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rPred' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rPred', rFrame, CSL.Resource.canCompose_update_left hupdatePred hcompRest,
        CSL.Resource.equiv_refl _, hpred', hframeFinal⟩⟩

theorem assignPredSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPred dst cmp }) frame) :
    InstrSpec cta warp { guard? := none, instr := .assignPred dst cmp }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) :=
  assignPredSpec_single_warpAt_frame heval (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem assignPredSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalCmpsFor st cta warp cmp lanes newValues) :
    InstrSpec cta warp { guard? := none, instr := .assignPred dst cmp }
      (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        predsFor cta warp lanes dst newValues) := by
  intro st r st' hpre hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rPreds, hcomp, hequiv, hctrl, hpreds⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalCmp? st cta warp lane cmp).bind fun b =>
            some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : PredsUpdateFacts stCore cta warp dst lanes newValues :=
        PredsUpdateFacts.of_applyAssignPred hnodup hevals' hcore
      have hfactsFinal : PredsUpdateFacts st' cta warp dst lanes newValues :=
        PredsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.predsFor hfactsFinal rPreds hpreds with
        ⟨rPreds', hupdatePreds, hpreds'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalCmp? st cta warp lane cmp with
            | none =>
                simp [heval] at hf
            | some b =>
                simp [heval] at hf
                subst new
                simp [Helpers.writePred])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalCmp? st cta warp lane cmp with
            | none =>
                simp [heval] at hf
            | some b =>
                simp [heval] at hf
                subst new
                simp [Helpers.writePred])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      refine ⟨CSL.Resource.compose CSL.Resource.empty rPreds', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePreds)
      · exact ⟨CSL.Resource.empty, rPreds',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal, hpreds'⟩

theorem assignPredValueSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .assignPredValue dst rhs } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .assignPredValue dst rhs }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem assignPredValueSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {rhs : RValue} {st₀ : State} {warpState : WarpState} {lane : LaneId}
    {laneState : LaneState} {old new : Bool} {value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new) :
    InstrSpec cta warp { guard? := guard?, instr := .assignPredValue dst rhs }
      (fun st r => st = st₀ ∧ CSL.pred cta warp lane dst old st r)
      (CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hpred⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st₀ cta warp lane rhs).bind fun value =>
            (Helpers.valueToBool? value).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writePred laneState dst new) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold EvalRValue at heval
        simp [hlane, heval, hbool] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writePred laneState dst new) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      exact StateResourceUpdate.pred_written_advanced hcore hstep r hpred

theorem assignPredValueSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rPred, hcomp, hequiv, hctrl, hpred⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases assignPredValueSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new) (value := value)
      hwarp hlock hpart hlane heval' hbool st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane rhs).bind fun value =>
              (Helpers.valueToBool? value).bind fun b =>
                some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hbool] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rPred', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePred)
  · exact ⟨CSL.Resource.empty, rPred',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hpred'⟩

theorem assignPredValueSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignPredValue dst rhs } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases assignPredValueSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst) (rhs := rhs)
      (st₀ := st) (warpState := warpState) (lane := lane) (laneState := laneState)
      (old := old) (new := new) (value := value)
      hwarp hlock hpart hlane heval' hbool st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane rhs).bind fun value =>
              (Helpers.valueToBool? value).bind fun b =>
                some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hbool] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rPred' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdatePred (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rPred' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rPred' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rPred', rFrame, CSL.Resource.canCompose_update_left hupdatePred hcompRest,
        CSL.Resource.equiv_refl _, hpred', hframeFinal⟩⟩

theorem assignPredValueSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) :=
  assignPredValueSpec_single_warpAt_frame heval hbool (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem assignPredValueSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        predsFor cta warp lanes dst newValues) := by
  intro st r st' hpre hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rPreds, hcomp, hequiv, hctrl, hpreds⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane rhs).bind fun value =>
            (Helpers.valueToBool? value).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : PredsUpdateFacts stCore cta warp dst lanes newValues :=
        PredsUpdateFacts.of_applyAssignPredValue hnodup hevals' hcore
      have hfactsFinal : PredsUpdateFacts st' cta warp dst lanes newValues :=
        PredsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.predsFor hfactsFinal rPreds hpreds with
        ⟨rPreds', hupdatePreds, hpreds'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some raw =>
                cases hbool : Helpers.valueToBool? raw with
                | none =>
                    simp [heval, hbool] at hf
                | some b =>
                    simp [heval, hbool] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane rhs with
            | none =>
                simp [heval] at hf
            | some raw =>
                cases hbool : Helpers.valueToBool? raw with
                | none =>
                    simp [heval, hbool] at hf
                | some b =>
                    simp [heval, hbool] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      refine ⟨CSL.Resource.compose CSL.Resource.empty rPreds', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePreds)
      · exact ⟨CSL.Resource.empty, rPreds',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal, hpreds'⟩

theorem loadSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem storeSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : TypedAddr}
    {value : RValue} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .store dst value } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .store dst value }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem globalStoreBytesSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {st₀ stCore : State}
    {warpState : WarpState} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := guard?, instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.globalBytes offset .write oldBytes st r)
      (CSL.globalBytes offset .write newBytes) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hbytes⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  exact StateResourceUpdate.globalBytes_of_writeMemFact_advanced
    hwrite hstep hencode hlen r hbytes

theorem globalStoreBytesSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.globalBytes offset .write newBytes) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have heval' := heval st r hpre
  rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases globalStoreBytesSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (ty := ty)
      (addrExpr := addrExpr) (valueExpr := valueExpr) (st₀ := st) (stCore := stCore)
      (warpState := warpState) (lane := lane) (offset := offset)
      (oldBytes := oldBytes) (newBytes := newBytes) (value := value)
      hwarp hlock hpart haddr' heval' hwrite' hencode hlen
      st rMem st' ⟨rfl, hbytes⟩ hstep with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hwriteFact := hwrite'
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    unfold ResolvesAddr at haddr'
    unfold EvalRValue at heval'
    unfold WriteMemFact at hwrite'
    simp [Helpers.stepStoreLanes?, haddr', heval', hwrite'] at hstep
    have hwarpCore : stCore.getWarp? cta warp = some warpState := by
      rw [WriteMemFact.global_getWarp_eq hwriteFact cta warp]
      exact hwarp
    have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
      have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
        Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
      rw [hlanes]
      simp
    rcases (by
        unfold Helpers.runnableLaneIds at hmemRunnable
        simp [Helpers.laneIsRunnable] at hmemRunnable
        cases hget : warpState.getLane? lane with
        | none =>
            simp [hget] at hmemRunnable
        | some laneState =>
            exact ⟨laneState, rfl⟩ :
        ∃ laneState, warpState.getLane? lane = some laneState) with
      ⟨laneState, hwarpLane⟩
    have hlane : st.getLane? cta warp lane = some laneState := by
      unfold State.getLane?
      simp [hwarp, hwarpLane]
    have hlaneCore : stCore.getLane? cta warp lane = some laneState := by
      rw [WriteMemFact.global_getLane_eq hwriteFact cta warp lane]
      exact hlane
    rcases Helpers.advanceRunnablePcs?_single_warp_control
        hwarpCore hlock hrpc hpart hlaneCore hstep with
      ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩
    exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem globalStoreBytesSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes offset .write oldBytes ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes offset .write newBytes ∗ frame)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have heval' := heval st r hpre
  rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases globalStoreBytesSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (ty := ty)
      (addrExpr := addrExpr) (valueExpr := valueExpr) (st₀ := st) (stCore := stCore)
      (warpState := warpState) (lane := lane) (offset := offset)
      (oldBytes := oldBytes) (newBytes := newBytes) (value := value)
      hwarp hlock hpart haddr' heval' hwrite' hencode hlen
      st rMem st' ⟨rfl, hbytes⟩ hstep with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hwriteFact := hwrite'
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    unfold ResolvesAddr at haddr'
    unfold EvalRValue at heval'
    unfold WriteMemFact at hwrite'
    simp [Helpers.stepStoreLanes?, haddr', heval', hwrite'] at hstep
    have hwarpCore : stCore.getWarp? cta warp = some warpState := by
      rw [WriteMemFact.global_getWarp_eq hwriteFact cta warp]
      exact hwarp
    have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
      have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
        Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
      rw [hlanes]
      simp
    rcases (by
        unfold Helpers.runnableLaneIds at hmemRunnable
        simp [Helpers.laneIsRunnable] at hmemRunnable
        cases hget : warpState.getLane? lane with
        | none =>
            simp [hget] at hmemRunnable
        | some laneState =>
            exact ⟨laneState, rfl⟩ :
        ∃ laneState, warpState.getLane? lane = some laneState) with
      ⟨laneState, hwarpLane⟩
    have hlane : st.getLane? cta warp lane = some laneState := by
      unfold State.getLane?
      simp [hwarp, hwarpLane]
    have hlaneCore : stCore.getLane? cta warp lane = some laneState := by
      rw [WriteMemFact.global_getLane_eq hwriteFact cta warp lane]
      exact hlane
    rcases Helpers.advanceRunnablePcs?_single_warp_control
        hwarpCore hlock hrpc hpart hlaneCore hstep with
      ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩
    exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rMem' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateMem (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rMem' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rMem', rFrame, CSL.Resource.canCompose_update_left hupdateMem hcompRest,
        CSL.Resource.equiv_refl _, hbytes', hframeFinal⟩⟩

theorem globalStoreBytesSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes offset .write oldBytes ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes offset .write newBytes ∗ frame)) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' _r rFrame _hpre hframeSt hstep
    exact hframe st st' rFrame hstep hframeSt)

theorem globalStoreLanes_warpAt_control
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {st st' : State} {rCtrl : CSL.Resource}
    (hctrl : warpAt cta warp pc lanes st rCtrl)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .global, ty := ty, addr := addrExpr } valueExpr with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      rcases Helpers.stepStoreLanes?_warp_control_eq hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.stepStoreLanes?_runnableLaneIds_eq hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore
          hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩

theorem sharedStoreLanes_warpAt_control
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {st st' : State} {rCtrl : CSL.Resource}
    (hctrl : warpAt cta warp pc lanes st rCtrl)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .shared, ty := ty, addr := addrExpr } valueExpr with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      rcases Helpers.stepStoreLanes?_warp_control_eq hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.stepStoreLanes?_runnableLaneIds_eq hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore
          hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩

theorem localStoreLanes_warpAt_control
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {st st' : State} {rCtrl : CSL.Resource}
    (hctrl : warpAt cta warp pc lanes st rCtrl)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.stepStoreLanes? st cta warp lanes
        { space := .local, ty := ty, addr := addrExpr } valueExpr with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      rcases Helpers.stepStoreLanes?_warp_control_eq hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.stepStoreLanes?_runnableLaneIds_eq hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore
          hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩

theorem globalStoreBytesSpec_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalSlicesUpdateFacts st' offsets oldSlices newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) := by
  intro st r st' hpre hstep
  have hfacts' := hfacts st r st' hpre hstep
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases StateResourceUpdate.globalSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty := by
    have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
      Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .global, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstep
        simp at hstep
    | some stCore =>
        rw [hcore] at hstep
        simp at hstep
        rcases Helpers.stepStoreLanes?_warp_control_eq hwarp hlock hrpc hcore with
          ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
        rcases Helpers.stepStoreLanes?_runnableLaneIds_eq hwarp hcore with
          ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
        have hwarpCoreEq : warpCoreRun = warpCore := by
          apply Option.some.inj
          rw [← hwarpCoreRun]
          exact hwarpCore
        subst warpCoreRun
        rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
          ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
        rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
          ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
        have hwarpFinalEq : warpFinalRun = warpFinal := by
          apply Option.some.inj
          rw [← hwarpFinalRun]
          exact hwarpFinal
        subst warpFinalRun
        have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
          rw [hrunFinalCore, hrunCore, hlanesStart]
        have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
          unfold Helpers.ParticipatingRunnable
          rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
            hlockFinal hrpcFinal]
          rw [hrunFinal]
        exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem globalStoreBytesSpec_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalMemoryBytesFor st' offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) :=
  globalStoreBytesSpec_lanes_warpAt_of_facts (by
    intro st r st' hpre hstep
    exact GlobalSlicesUpdateFacts.of_memoryBytesFor hlens (hmems st r st' hpre hstep))

theorem globalStoreBytesSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        globalSlices offsets .write newSlices) :=
  globalStoreBytesSpec_lanes_warpAt_of_memory hlens (by
    intro st r st' hpre hstep
    have haddrs' := haddrs st r hpre
    have hevals' := hevals st r hpre
    rcases hpre with ⟨_rCtrl, _rMem, _hcomp, _hequiv, hctrl, _hbytes⟩
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .global, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstep
        simp at hstep
    | some stCore =>
        rw [hcore] at hstep
        simp at hstep
        have hmemCore :
            GlobalMemoryBytesFor stCore offsets newSlices :=
          GlobalMemoryBytesFor.of_stepStoreLanes_global
            haddrs' hevals' hencs hdisjoint hcore
        exact GlobalMemoryBytesFor.of_global_eq
          (Helpers.advanceRunnablePcs?_global_eq hstep) hmemCore)

theorem globalStoreBytesSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        (globalSlices offsets .write oldSlices ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (globalSlices offsets .write newSlices ∗ frame)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩
  have hmemFinal : GlobalMemoryBytesFor st' offsets newSlices := by
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    have hstepCore := hstep
    unfold Helpers.stepInstr? at hstepCore
    simp [hwarp, hlockBool, hpartOpt] at hstepCore
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .global, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstepCore
        simp at hstepCore
    | some stCore =>
        rw [hcore] at hstepCore
        simp at hstepCore
        have hmemCore :
            GlobalMemoryBytesFor stCore offsets newSlices :=
          GlobalMemoryBytesFor.of_stepStoreLanes_global
            haddrs' hevals' hencs hdisjoint hcore
        exact GlobalMemoryBytesFor.of_global_eq
          (Helpers.advanceRunnablePcs?_global_eq hstepCore) hmemCore
  have hfacts' : GlobalSlicesUpdateFacts st' offsets oldSlices newSlices :=
    GlobalSlicesUpdateFacts.of_memoryBytesFor hlens hmemFinal
  rcases StateResourceUpdate.globalSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
    globalStoreLanes_warpAt_control hctrl hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rMem' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateMem (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rMem' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rMem', rFrame, CSL.Resource.canCompose_update_left hupdateMem hcompRest,
        CSL.Resource.equiv_refl _, hbytes', hframeFinal⟩⟩

theorem globalStorePreservesReadReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {value srcValue : Value}
    {st stCore st' : State} {rCtrl rSrc : CSL.Resource}
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hsrc : CSL.reg cta warp lane src srcValue st rSrc)
    (haddr :
      ResolvesAddr st { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval : EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
                  instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    CSL.reg cta warp lane src srcValue st' rSrc := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  rcases hsrc with ⟨hsrcOwns, laneState, hlane, hsrcRead⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  have hlaneCore : stCore.getLane? cta warp lane = some laneState := by
    rw [WriteMemFact.global_getLane_eq (by exact hwrite) cta warp lane]
    exact hlane
  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
    ⟨laneStateFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
  exact ⟨hsrcOwns, laneStateFinal, hlaneFinal, by simpa [hregs] using hsrcRead⟩

theorem globalStorePreservesPred_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {pred : PredName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {value : Value} {predValue : Bool}
    {st stCore st' : State} {rCtrl rPred : CSL.Resource}
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hpred : CSL.pred cta warp lane pred predValue st rPred)
    (haddr :
      ResolvesAddr st { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval : EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st .global ty (.global offset) value stCore)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    CSL.pred cta warp lane pred predValue st' rPred := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  rcases hpred with ⟨hpredOwns, laneState, hlane, hpredRead⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  have hlaneCore : stCore.getLane? cta warp lane = some laneState := by
    rw [WriteMemFact.global_getLane_eq (by exact hwrite) cta warp lane]
    exact hlane
  rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
    ⟨laneStateFinal, hlaneFinal, _hlocal, _hregs, hpreds⟩
  exact ⟨hpredOwns, laneStateFinal, hlaneFinal, by simpa [hpreds] using hpredRead⟩

theorem globalStoreBytesSpec_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value srcValue : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes offset .write oldBytes ∗
          CSL.reg cta warp lane src srcValue))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes offset .write newBytes ∗
          CSL.reg cta warp lane src srcValue)) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hsrc hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    exact globalStorePreservesReadReg_single_warpAt
      hctrl hsrc haddr' heval' hwrite' hstep)

theorem globalStorePreservesGlobalBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {readOffset writeOffset : Nat}
    {readBytes newWriteBytes : List Byte} {perm : CSL.BytePerm}
    {value : Value} {st stCore st' : State} {rCtrl rRead : CSL.Resource}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset newWriteBytes.length)
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hbytes : CSL.globalBytes readOffset perm readBytes st rRead)
    (haddr :
      ResolvesAddr st { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval : EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
        some st') :
    CSL.globalBytes readOffset perm readBytes st' rRead := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  have hwriteFact := hwrite
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  have hmemCore : CSL.memoryBytes stCore.global.bytes readOffset readBytes := by
    have hwriteMem := hwriteFact
    unfold WriteMemFact Helpers.writeMem? at hwriteMem
    unfold EncodedScalar at hencode
    simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode] at hwriteMem
    rcases hwriteMem with ⟨_, hstCore⟩
    rw [← hstCore]
    exact memoryBytes_writeBytes_preserved_of_disjoint hdisjoint
      (CSL.globalBytes_memory hbytes)
  have hglobalFinal : st'.global = stCore.global :=
    Helpers.advanceRunnablePcs?_global_eq hstep
  have hmemFinal : CSL.memoryBytes st'.global.bytes readOffset readBytes := by
    rw [hglobalFinal]
    exact hmemCore
  exact CSL.globalBytes_of_memory hbytes hmemFinal

theorem globalStoreBytesSpec_single_warpAt_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset : Nat}
    {oldWriteBytes newWriteBytes readBytes : List Byte} {readPerm : CSL.BytePerm}
    {value : Value}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes writeOffset .write oldWriteBytes ∗
          CSL.globalBytes readOffset readPerm readBytes))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes writeOffset .write newWriteBytes ∗
          CSL.globalBytes readOffset readPerm readBytes)) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hbytes hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    exact globalStorePreservesGlobalBytes_single_warpAt hdisjoint hctrl hbytes
      haddr' heval' hwrite' hencode hstep)

theorem globalStoreBytesSpec_single_warpAt_readReg_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset : Nat}
    {oldWriteBytes newWriteBytes readBytes : List Byte} {readPerm : CSL.BytePerm}
    {value srcValue : Value}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes writeOffset .write oldWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            CSL.globalBytes readOffset readPerm readBytes)))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes writeOffset .write newWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            CSL.globalBytes readOffset readPerm readBytes))) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hframe hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    rcases hframe with ⟨rReg, rBytes, hcompFrame, hequivFrame, hreg, hbytes⟩
    exact ⟨rReg, rBytes, hcompFrame, hequivFrame,
      globalStorePreservesReadReg_single_warpAt
        hctrl hreg haddr' heval' hwrite' hstep,
      globalStorePreservesGlobalBytes_single_warpAt hdisjoint hctrl hbytes
        haddr' heval' hwrite' hencode hstep⟩)

theorem globalStoreBytesSpec_single_warpAt_readReg_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm} {value srcValue : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes writeOffset .write oldWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
              CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes writeOffset .write newWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
              CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hframe hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    rcases hframe with ⟨rReg, rBytes, hcompFrame, hequivFrame, hreg, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes, hbytes₁, hbytes₂⟩
    exact ⟨rReg, rBytes, hcompFrame, hequivFrame,
      globalStorePreservesReadReg_single_warpAt
        hctrl hreg haddr' heval' hwrite' hstep,
      ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
        globalStorePreservesGlobalBytes_single_warpAt hdisjoint₁ hctrl hbytes₁
          haddr' heval' hwrite' hencode hstep,
        globalStorePreservesGlobalBytes_single_warpAt hdisjoint₂ hctrl hbytes₂
          haddr' heval' hwrite' hencode hstep⟩⟩)

theorem globalStoreBytesSpec_single_warpAt_readRegs3_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src reg₁ reg₂ : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm} {value srcValue regValue₁ regValue₂ : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes writeOffset .write oldWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.reg cta warp lane reg₁ regValue₁ ∗
              (CSL.reg cta warp lane reg₂ regValue₂ ∗
                (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                  CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes writeOffset .write newWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.reg cta warp lane reg₁ regValue₁ ∗
              (CSL.reg cta warp lane reg₂ regValue₂ ∗
                (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                  CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hframe hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    rcases hframe with ⟨rSrc, rRest, hcompSrc, hequivSrc, hsrc, hrestFrame⟩
    rcases hrestFrame with ⟨rReg₁, rRestRegs, hcompReg₁, hequivReg₁, hreg₁, hrestRegs⟩
    rcases hrestRegs with ⟨rReg₂, rBytes, hcompReg₂, hequivReg₂, hreg₂, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes, hbytes₁, hbytes₂⟩
    exact ⟨rSrc, rRest, hcompSrc, hequivSrc,
      globalStorePreservesReadReg_single_warpAt
        hctrl hsrc haddr' heval' hwrite' hstep,
      ⟨rReg₁, rRestRegs, hcompReg₁, hequivReg₁,
        globalStorePreservesReadReg_single_warpAt
          hctrl hreg₁ haddr' heval' hwrite' hstep,
        ⟨rReg₂, rBytes, hcompReg₂, hequivReg₂,
          globalStorePreservesReadReg_single_warpAt
            hctrl hreg₂ haddr' heval' hwrite' hstep,
          ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
            globalStorePreservesGlobalBytes_single_warpAt hdisjoint₁ hctrl hbytes₁
              haddr' heval' hwrite' hencode hstep,
            globalStorePreservesGlobalBytes_single_warpAt hdisjoint₂ hctrl hbytes₂
              haddr' heval' hwrite' hencode hstep⟩⟩⟩⟩)

theorem globalStoreBytesSpec_single_warpAt_readRegs4_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src reg₁ reg₂ reg₃ : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm}
    {value srcValue regValue₁ regValue₂ regValue₃ : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes writeOffset .write oldWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.reg cta warp lane reg₁ regValue₁ ∗
              (CSL.reg cta warp lane reg₂ regValue₂ ∗
                (CSL.reg cta warp lane reg₃ regValue₃ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes writeOffset .write newWriteBytes ∗
          (CSL.reg cta warp lane src srcValue ∗
            (CSL.reg cta warp lane reg₁ regValue₁ ∗
              (CSL.reg cta warp lane reg₂ regValue₂ ∗
                (CSL.reg cta warp lane reg₃ regValue₃ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) :=
  globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen (by
    intro st st' r rFrame hpre hframe hstep
    have haddr' := haddr st r hpre
    have heval' := heval st r hpre
    rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
    rcases hframe with ⟨rSrc, rRest, hcompSrc, hequivSrc, hsrc, hrestFrame⟩
    rcases hrestFrame with ⟨rReg₁, rRest₁, hcompReg₁, hequivReg₁, hreg₁, hrest₁⟩
    rcases hrest₁ with ⟨rReg₂, rRest₂, hcompReg₂, hequivReg₂, hreg₂, hrest₂⟩
    rcases hrest₂ with ⟨rReg₃, rBytes, hcompReg₃, hequivReg₃, hreg₃, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes, hbytes₁, hbytes₂⟩
    exact ⟨rSrc, rRest, hcompSrc, hequivSrc,
      globalStorePreservesReadReg_single_warpAt
        hctrl hsrc haddr' heval' hwrite' hstep,
      ⟨rReg₁, rRest₁, hcompReg₁, hequivReg₁,
        globalStorePreservesReadReg_single_warpAt
          hctrl hreg₁ haddr' heval' hwrite' hstep,
        ⟨rReg₂, rRest₂, hcompReg₂, hequivReg₂,
          globalStorePreservesReadReg_single_warpAt
            hctrl hreg₂ haddr' heval' hwrite' hstep,
          ⟨rReg₃, rBytes, hcompReg₃, hequivReg₃,
            globalStorePreservesReadReg_single_warpAt
              hctrl hreg₃ haddr' heval' hwrite' hstep,
            ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
              globalStorePreservesGlobalBytes_single_warpAt hdisjoint₁ hctrl hbytes₁
                haddr' heval' hwrite' hencode hstep,
              globalStorePreservesGlobalBytes_single_warpAt hdisjoint₂ hctrl hbytes₂
                haddr' heval' hwrite' hencode hstep⟩⟩⟩⟩⟩)

theorem sharedStoreBytesSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {st₀ stCore : State}
    {warpState : WarpState} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .shared ty (.shared cta offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := guard?, instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.sharedBytes cta offset .write oldBytes st r)
      (CSL.sharedBytes cta offset .write newBytes) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hbytes⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  exact StateResourceUpdate.sharedBytes_of_writeMemFact_advanced
    hwrite hstep hencode hlen r hbytes

theorem sharedStoreBytesSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .shared ty (.shared cta offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        CSL.sharedBytes cta offset .write oldBytes)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.sharedBytes cta offset .write newBytes) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have heval' := heval st r hpre
  rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases sharedStoreBytesSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (ty := ty)
      (addrExpr := addrExpr) (valueExpr := valueExpr) (st₀ := st) (stCore := stCore)
      (warpState := warpState) (lane := lane) (offset := offset)
      (oldBytes := oldBytes) (newBytes := newBytes) (value := value)
      hwarp hlock hpart haddr' heval' hwrite' hencode hlen
      st rMem st' ⟨rfl, hbytes⟩ hstep with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hwriteFact := hwrite'
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    unfold ResolvesAddr at haddr'
    unfold EvalRValue at heval'
    unfold WriteMemFact at hwrite'
    simp [Helpers.stepStoreLanes?, haddr', heval', hwrite'] at hstep
    have hwarpCore : stCore.getWarp? cta warp = some warpState := by
      rw [WriteMemFact.shared_getWarp_eq hwriteFact]
      exact hwarp
    have hmemRunnable : lane ∈ Helpers.runnableLaneIds warpState := by
      have hlanes : Helpers.runnableLaneIds warpState = [lane] :=
        Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
      rw [hlanes]
      simp
    rcases (by
        unfold Helpers.runnableLaneIds at hmemRunnable
        simp [Helpers.laneIsRunnable] at hmemRunnable
        cases hget : warpState.getLane? lane with
        | none =>
            simp [hget] at hmemRunnable
        | some laneState =>
            exact ⟨laneState, rfl⟩ :
        ∃ laneState, warpState.getLane? lane = some laneState) with
      ⟨laneState, hwarpLane⟩
    have hlaneCore : stCore.getLane? cta warp lane = some laneState := by
      unfold State.getLane?
      simp [hwarpCore, hwarpLane]
    rcases Helpers.advanceRunnablePcs?_single_warp_control
        hwarpCore hlock hrpc hpart hlaneCore hstep with
      ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩
    exact ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem sharedStoreBytesSpec_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedSlicesUpdateFacts st' cta offsets oldSlices newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        sharedSlices cta offsets .write newSlices) := by
  intro st r st' hpre hstep
  have hfacts' := hfacts st r st' hpre hstep
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases StateResourceUpdate.sharedSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
    sharedStoreLanes_warpAt_control hctrl hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem sharedStoreBytesSpec_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedMemoryBytesFor st' cta offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        sharedSlices cta offsets .write newSlices) :=
  sharedStoreBytesSpec_lanes_warpAt_of_facts (by
    intro st r st' hpre hstep
    exact SharedSlicesUpdateFacts.of_memoryBytesFor hlens (hmems st r st' hpre hstep))

theorem sharedStoreBytesSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        sharedSlices cta offsets .write newSlices) :=
  sharedStoreBytesSpec_lanes_warpAt_of_memory hlens (by
    intro st r st' hpre hstep
    have haddrs' := haddrs st r hpre
    have hevals' := hevals st r hpre
    rcases hpre with ⟨_rCtrl, _rMem, _hcomp, _hequiv, hctrl, _hbytes⟩
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .shared, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstep
        simp at hstep
    | some stCore =>
        rw [hcore] at hstep
        simp at hstep
        have hmemCore :
            SharedMemoryBytesFor stCore cta offsets newSlices :=
          SharedMemoryBytesFor.of_stepStoreLanes_shared
            haddrs' hevals' hencs hdisjoint hcore
        exact SharedMemoryBytesFor.of_shared_preserved
          (by
            intro ctaState hcta
            exact Helpers.advanceRunnablePcs?_shared_eq hcta hstep)
          hmemCore)

theorem sharedStoreBytesSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        (sharedSlices cta offsets .write oldSlices ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (sharedSlices cta offsets .write newSlices ∗ frame)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩
  have hmemFinal : SharedMemoryBytesFor st' cta offsets newSlices := by
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    have hstepCore := hstep
    unfold Helpers.stepInstr? at hstepCore
    simp [hwarp, hlockBool, hpartOpt] at hstepCore
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .shared, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstepCore
        simp at hstepCore
    | some stCore =>
        rw [hcore] at hstepCore
        simp at hstepCore
        have hmemCore :
            SharedMemoryBytesFor stCore cta offsets newSlices :=
          SharedMemoryBytesFor.of_stepStoreLanes_shared
            haddrs' hevals' hencs hdisjoint hcore
        exact SharedMemoryBytesFor.of_shared_preserved
          (by
            intro ctaState hcta
            exact Helpers.advanceRunnablePcs?_shared_eq hcta hstepCore)
          hmemCore
  have hfacts' : SharedSlicesUpdateFacts st' cta offsets oldSlices newSlices :=
    SharedSlicesUpdateFacts.of_memoryBytesFor hlens hmemFinal
  rcases StateResourceUpdate.sharedSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
    sharedStoreLanes_warpAt_control hctrl hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rMem' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateMem (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rMem' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rMem', rFrame, CSL.Resource.canCompose_update_left hupdateMem hcompRest,
        CSL.Resource.equiv_refl _, hbytes', hframeFinal⟩⟩

theorem localStoreBytesSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {st₀ stCore : State}
    {warpState : WarpState} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite : WriteMemFact st₀ .local ty (.local cta warp lane offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := guard?, instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (fun st r => st = st₀ ∧ CSL.localBytes cta warp lane offset .write oldBytes st r)
      (CSL.localBytes cta warp lane offset .write newBytes) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hbytes⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  unfold ResolvesAddr at haddr
  unfold EvalRValue at heval
  unfold WriteMemFact at hwrite
  simp [Helpers.stepStoreLanes?, haddr, heval, hwrite] at hstep
  exact StateResourceUpdate.localBytes_of_writeMemFact_advanced
    hwrite hstep hencode hlen r hbytes

theorem localStoreBytesSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .local ty (.local cta warp lane offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc [lane] ∗
        CSL.localBytes cta warp lane offset .write oldBytes)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.localBytes cta warp lane offset .write newBytes) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have heval' := heval st r hpre
  rcases hwrite st r hpre with ⟨stCore, hwrite'⟩
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases localStoreBytesSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (ty := ty)
      (addrExpr := addrExpr) (valueExpr := valueExpr) (st₀ := st) (stCore := stCore)
      (warpState := warpState) (lane := lane) (offset := offset)
      (oldBytes := oldBytes) (newBytes := newBytes) (value := value)
      hwarp hlock hpart haddr' heval' hwrite' hencode hlen
      st rMem st' ⟨rfl, hbytes⟩ hstep with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hwriteFact := hwrite'
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    unfold ResolvesAddr at haddr'
    unfold EvalRValue at heval'
    unfold WriteMemFact at hwrite'
    simp [Helpers.stepStoreLanes?, haddr', heval', hwrite'] at hstep
    rcases (by
        unfold WriteMemFact Helpers.writeMem? at hwriteFact
        unfold EncodedScalar at hencode
        cases hlane : st.getLane? cta warp lane with
        | none =>
            simp [Helpers.getSpaceBaseMem?, hencode, hlane] at hwriteFact
        | some laneState =>
            simp [Helpers.getSpaceBaseMem?, Helpers.setSpaceBaseMem?, hencode, hlane]
              at hwriteFact
            rcases hwriteFact with ⟨_, hset⟩
            exact ⟨laneState, rfl, by simpa [Addr.offset] using hset⟩ :
        ∃ laneState,
          st.getLane? cta warp lane = some laneState ∧
            st.setLane cta warp lane
              { laneState with localMem := {
                  bytes := Helpers.writeBytes laneState.localMem.bytes offset newBytes } } =
                some stCore) with
      ⟨laneState, hlane, hset⟩
    exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
      (by simp) (by simp) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem localStoreBytesSpec_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalSlicesUpdateFacts st' cta warp lanes offsets oldSlices newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        localSlices cta warp lanes offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        localSlices cta warp lanes offsets .write newSlices) := by
  intro st r st' hpre hstep
  have hfacts' := hfacts st r st' hpre hstep
  rcases hpre with ⟨rCtrl, rMem, hcomp, hequiv, hctrl, hbytes⟩
  rcases StateResourceUpdate.localSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
    localStoreLanes_warpAt_control hctrl hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rMem', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateMem)
  · exact ⟨CSL.Resource.empty, rMem',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hbytes'⟩

theorem localStoreBytesSpec_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalMemoryBytesFor st' cta warp lanes offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        localSlices cta warp lanes offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        localSlices cta warp lanes offsets .write newSlices) :=
  localStoreBytesSpec_lanes_warpAt_of_facts (by
    intro st r st' hpre hstep
    exact LocalSlicesUpdateFacts.of_memoryBytesFor hlens
      (hmems st r st' hpre hstep))

theorem localStoreBytesSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        localSlices cta warp lanes offsets .write oldSlices)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        localSlices cta warp lanes offsets .write newSlices) :=
  localStoreBytesSpec_lanes_warpAt_of_memory hlens (by
    intro st r st' hpre hstep
    have haddrs' := haddrs st r hpre
    have hevals' := hevals st r hpre
    rcases hpre with ⟨_rCtrl, _rMem, _hcomp, _hequiv, hctrl, _hbytes⟩
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .local, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstep
        simp at hstep
    | some stCore =>
        rw [hcore] at hstep
        simp at hstep
        have hmemCore :
            LocalMemoryBytesFor stCore cta warp lanes offsets newSlices :=
          LocalMemoryBytesFor.of_stepStoreLanes_local
            haddrs' hevals' hencs hdisjoint hcore
        exact LocalMemoryBytesFor.of_local_preserved
          (by
            intro lane laneState hlane
            exact Helpers.advanceRunnablePcs?_lane_localMem_eq hlane hstep)
          hmemCore)

theorem localStoreBytesSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp
      { guard? := none,
        instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
      (warpAt cta warp pc lanes ∗
        (localSlices cta warp lanes offsets .write oldSlices ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (localSlices cta warp lanes offsets .write newSlices ∗ frame)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩
  have hmemFinal : LocalMemoryBytesFor st' cta warp lanes offsets newSlices := by
    rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
      (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
    have hstepCore := hstep
    unfold Helpers.stepInstr? at hstepCore
    simp [hwarp, hlockBool, hpartOpt] at hstepCore
    cases hcore :
        Helpers.stepStoreLanes? st cta warp lanes
          { space := .local, ty := ty, addr := addrExpr } valueExpr with
    | none =>
        rw [hcore] at hstepCore
        simp at hstepCore
    | some stCore =>
        rw [hcore] at hstepCore
        simp at hstepCore
        have hmemCore :
            LocalMemoryBytesFor stCore cta warp lanes offsets newSlices :=
          LocalMemoryBytesFor.of_stepStoreLanes_local
            haddrs' hevals' hencs hdisjoint hcore
        exact LocalMemoryBytesFor.of_local_preserved
          (by
            intro lane laneState hlane
            exact Helpers.advanceRunnablePcs?_lane_localMem_eq hlane hstepCore)
          hmemCore
  have hfacts' : LocalSlicesUpdateFacts st' cta warp lanes offsets oldSlices newSlices :=
    LocalSlicesUpdateFacts.of_memoryBytesFor hlens hmemFinal
  rcases StateResourceUpdate.localSlices hfacts' rMem hbytes with
    ⟨rMem', hupdateMem, hbytes'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
    localStoreLanes_warpAt_control hctrl hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rMem, rFrame, hcompRest, hequivRest, hbytes, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rMem' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateMem (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rMem' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rMem', rFrame, CSL.Resource.canCompose_update_left hupdateMem hcompRest,
        CSL.Resource.equiv_refl _, hbytes', hframeFinal⟩⟩

theorem globalLoadBytesRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.global.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst newReg) st r) :=
  loadSpec_of_computed hstep <|
    StateResourceUpdate.sep
      (StateResourceUpdate.globalBytes (rfl) hmem)
      (StateResourceUpdate.reg hreg)

theorem globalLoadBytesRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {st₀ : State}
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread : ReadMemFact st₀ .global ty (.global offset) value) :
    InstrSpec cta warp
      { guard? := guard?, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst value) st r) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hsep⟩
  subst st
  rcases hsep with ⟨rMem, rReg, hcomp, hequiv, hbytes, hreg⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st₀ cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st₀ .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hglobalCore : stCore.global = st₀.global :=
        Helpers.applyToLaneIds?_global_eq happly
      have hglobalFinal : st'.global = stCore.global :=
        Helpers.advanceRunnablePcs?_global_eq hstep
      have hmemFinal : CSL.memoryBytes st'.global.bytes offset bytes := by
        rw [hglobalFinal, hglobalCore]
        exact CSL.globalBytes_memory hbytes
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst value) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold ResolvesAddr at haddr
        unfold ReadMemFact at hread
        simp [hlane, haddr, hread] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst value) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      rcases StateResourceUpdate.globalBytes
          (st₀ := st₀) (st₁ := st') (offset := offset)
          (perm := CSL.BytePerm.read) (oldBytes := bytes) (newBytes := bytes)
          rfl hmemFinal rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases StateResourceUpdate.reg_written_advanced
          (st₀ := st₀) (stCore := stCore) (st₁ := st')
          (cta := cta) (warp := warp) (lane := lane)
          (name := dst) (old := oldReg) (new := value)
          hcore hstep rReg hreg with
        ⟨rReg', hupdateReg, hreg'⟩
      refine ⟨CSL.Resource.compose rMem' rReg', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose hupdateMem hupdateReg)
      · exact ⟨rMem', rReg',
          CSL.Resource.canCompose_update_right hupdateReg
            (CSL.Resource.canCompose_update_left hupdateMem hcomp),
          CSL.Resource.equiv_refl _, hbytes', hreg'⟩

theorem globalLoadBytesRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst value)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  have hrestAll := hrest
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hrest with ⟨rMem, rReg, hcompRest, hequivRest, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases globalLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (offset := offset) (bytes := bytes)
      (oldReg := oldReg) (value := value)
      hwarp hlock hpart hlane haddr' hread' st rRest st' ⟨rfl, hrestAll⟩ hstep with
    ⟨rRest', hupdateRest, hrest'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .global ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rRest', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, rRest',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hrest'⟩

theorem globalLoadBytesRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc lanes ∗
        (globalSlices offsets .read byteSlices ∗
          regsFor cta warp lanes dst oldValues))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (globalSlices offsets .read byteSlices ∗
          regsFor cta warp lanes dst newValues)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hreads' := hreads st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rRegs, hcompRest, hequivRest, hbytes, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyGlobalLoad hnodup haddrs' hreads' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hglobalCore : stCore.global = st.global :=
        Helpers.applyToLaneIds?_global_eq hcore
      have hglobalFinal : st'.global = stCore.global :=
        Helpers.advanceRunnablePcs?_global_eq hstep
      have hglobal : st'.global = st.global :=
        hglobalFinal.trans hglobalCore
      have hsliceFacts : GlobalSlicesUpdateFacts st' offsets byteSlices byteSlices :=
        GlobalSlicesUpdateFacts.of_global_eq hglobal
          (GlobalSlicesUpdateFacts.of_globalSlices hbytes)
      rcases StateResourceUpdate.globalSlices hsliceFacts rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .global, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .global ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .global, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .global ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rMem' rRegs') :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateMem hupdateRegs)
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rMem' rRegs'), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rMem', rRegs',
            CSL.Resource.canCompose_update_right hupdateRegs
              (CSL.Resource.canCompose_update_left hupdateMem hcompRest),
            CSL.Resource.equiv_refl _, hbytes', hregs'⟩⟩

theorem paramLoadBytesRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (paramSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesParamAddrsFor st cta warp
            { space := .param, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (paramSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadParamValuesFor st ty offsets newValues) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc lanes ∗
        (paramSlices offsets byteSlices ∗
          regsFor cta warp lanes dst oldValues))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (paramSlices offsets byteSlices ∗
          regsFor cta warp lanes dst newValues)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hreads' := hreads st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rRegs, hcompRest, hequivRest, hbytes, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .param, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .param ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyParamLoad hnodup haddrs' hreads' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hparamCore : stCore.param = st.param :=
        Helpers.applyToLaneIds?_param_eq hcore
      have hparamFinal : st'.param = stCore.param :=
        Helpers.advanceRunnablePcs?_param_eq hstep
      have hparam : st'.param = st.param :=
        hparamFinal.trans hparamCore
      have hsliceFacts : ParamSlicesUpdateFacts st' offsets byteSlices :=
        ParamSlicesUpdateFacts.of_param_eq hparam
          (ParamSlicesUpdateFacts.of_paramSlices hbytes)
      rcases StateResourceUpdate.paramSlices hsliceFacts rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .param, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .param ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .param, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .param ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rMem' rRegs') :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateMem hupdateRegs)
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rMem' rRegs'), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rMem', rRegs',
            CSL.Resource.canCompose_update_right hupdateRegs
              (CSL.Resource.canCompose_update_left hupdateMem hcompRest),
            CSL.Resource.equiv_refl _, hbytes', hregs'⟩⟩

theorem constLoadBytesRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (constSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesConstAddrsFor st cta warp
            { space := .const, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (constSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadConstValuesFor st ty offsets newValues) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc lanes ∗
        (constSlices offsets byteSlices ∗
          regsFor cta warp lanes dst oldValues))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (constSlices offsets byteSlices ∗
          regsFor cta warp lanes dst newValues)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hreads' := hreads st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rRegs, hcompRest, hequivRest, hbytes, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .const, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .const ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyConstLoad hnodup haddrs' hreads' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hconstCore : stCore.const = st.const :=
        Helpers.applyToLaneIds?_const_eq hcore
      have hconstFinal : st'.const = stCore.const :=
        Helpers.advanceRunnablePcs?_const_eq hstep
      have hconst : st'.const = st.const :=
        hconstFinal.trans hconstCore
      have hsliceFacts : ConstSlicesUpdateFacts st' offsets byteSlices :=
        ConstSlicesUpdateFacts.of_const_eq hconst
          (ConstSlicesUpdateFacts.of_constSlices hbytes)
      rcases StateResourceUpdate.constSlices hsliceFacts rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .const, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .const ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .const, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .const ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rMem' rRegs') :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateMem hupdateRegs)
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rMem' rRegs'), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rMem', rRegs',
            CSL.Resource.canCompose_update_right hupdateRegs
              (CSL.Resource.canCompose_update_left hupdateMem hcompRest),
            CSL.Resource.equiv_refl _, hbytes', hregs'⟩⟩

theorem globalLoadBytesRegSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ReadMemFact st .global ty (.global offset) value)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst oldReg) ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst value) ∗ frame)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMemReg, rFrame, hcompRest, hequivRest, hmemReg, hframeSt⟩
  have hmemRegAll := hmemReg
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hmemReg with ⟨rMem, rReg, hcompMemReg, hequivMemReg, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases globalLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (offset := offset) (bytes := bytes)
      (oldReg := oldReg) (value := value)
      hwarp hlock hpart hlane haddr' hread' st rMemReg st' ⟨rfl, hmemRegAll⟩ hstep with
    ⟨rMemReg', hupdateMemReg, hmemReg'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .global ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rMemReg, rFrame, hcompRest, hequivRest, hmemRegAll, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rMemReg' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateMemReg (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rMemReg' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMemReg' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rMemReg', rFrame, CSL.Resource.canCompose_update_left hupdateMemReg hcompRest,
        CSL.Resource.equiv_refl _, hmemReg', hframeFinal⟩⟩

theorem globalLoadBytesRegSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ReadMemFact st .global ty (.global offset) value)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .load dst { space := .global, ty := ty, addr := addrExpr } })
        frame) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst oldReg) ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst value) ∗ frame)) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' _r rFrame _hpre hframeSt hstep
    exact hframe st st' rFrame hstep hframeSt)

theorem globalLoadPreservesReadReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg srcValue : Value}
    {st st' : State} {rCtrl rMem rDst rSrc : CSL.Resource}
    (hne : src ≠ dst)
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (_hbytes : CSL.globalBytes offset .read bytes st rMem)
    (hdst : CSL.reg cta warp lane dst oldReg st rDst)
    (hsrc : CSL.reg cta warp lane src srcValue st rSrc)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } } =
        some st') :
    CSL.reg cta warp lane src srcValue st' rSrc := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  rcases CSL.reg_state hdst with ⟨dstLaneState, hdstLane, _hdstRead⟩
  rcases hsrc with ⟨hsrcOwns, srcLaneState, hsrcLane, hsrcRead⟩
  have hsameLane : srcLaneState = dstLaneState := by
    rw [hdstLane] at hsrcLane
    exact (Option.some.inj hsrcLane).symm
  subst srcLaneState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happly
      simp [hdstLane] at happly
      cases haddr :
          Helpers.resolveAddr? st cta warp lane
            { space := .global, ty := ty, addr := addrExpr } with
      | none =>
          simp [haddr] at happly
      | some addr =>
          simp [haddr] at happly
          cases hread : Helpers.readMem? st .global ty addr with
          | none =>
              simp [hread] at happly
          | some loaded =>
              simp [hread] at happly
              have hset :
                  st.setLane cta warp lane (Helpers.writeReg dstLaneState dst loaded) =
                    some stCore := by
                simpa [Helpers.applyToLaneIdsList?] using happly
              have hlaneCore :
                  stCore.getLane? cta warp lane =
                    some (Helpers.writeReg dstLaneState dst loaded) :=
                State.getLane?_setLane_same hdstLane hset
              rcases Helpers.advanceRunnablePcs?_lane_nonPc_eq hlaneCore hstep with
                ⟨laneStateFinal, hlaneFinal, _hlocal, hregs, _hpreds⟩
              have hneDstSrc : dst ≠ src := fun h => hne h.symm
              have hbeq : (dst == src) = false :=
                (beq_eq_false_iff_ne).2 hneDstSrc
              have hreadCore :
                  (Helpers.writeReg dstLaneState dst loaded).regs[src]? = some srcValue := by
                simp [Helpers.writeReg]
                rw [Std.HashMap.getElem?_insert]
                simp [hbeq, hsrcRead]
              exact ⟨hsrcOwns, laneStateFinal, hlaneFinal, by
                simpa [hregs] using hreadCore⟩

theorem globalLoadPreservesGlobalBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte}
    {st st' : State} {rCtrl rMem : CSL.Resource}
    (hctrl : warpAt cta warp pc [lane] st rCtrl)
    (hbytes : CSL.globalBytes offset perm bytes st rMem)
    (hstep :
      Helpers.stepInstr? st cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } } =
        some st') :
    CSL.globalBytes offset perm bytes st' rMem := by
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, _hrpc, hpart⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .global, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .global ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hglobalCore : stCore.global = st.global :=
        Helpers.applyToLaneIds?_global_eq happly
      have hglobalFinal : st'.global = stCore.global :=
        Helpers.advanceRunnablePcs?_global_eq hstep
      have hglobal : st'.global = st.global := hglobalFinal.trans hglobalCore
      induction bytes generalizing offset rMem with
      | nil =>
          simpa [CSL.globalBytes] using hbytes
      | cons byte bytes ih =>
          change (CSL.globalByte offset perm byte ∗
            CSL.globalBytes (offset + 1) perm bytes) st' rMem
          change (CSL.globalByte offset perm byte ∗
            CSL.globalBytes (offset + 1) perm bytes) st rMem at hbytes
          rcases hbytes with ⟨rByte, rRest, hcomp, hequiv, hbyte, hrest⟩
          refine ⟨rByte, rRest, hcomp, hequiv, ?_, ih (offset := offset + 1) hrest⟩
          rcases hbyte with ⟨howns, hmem⟩
          exact ⟨howns, by
            unfold CSL.memoryByte at hmem ⊢
            rw [hglobal]
            exact hmem⟩

theorem globalLoadBytesRegSpec_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value srcValue : Value}
    (hne : src ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst oldReg) ∗
          CSL.reg cta warp lane src srcValue))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes offset .read bytes ∗
          CSL.reg cta warp lane dst value) ∗
          CSL.reg cta warp lane src srcValue)) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' r rFrame hpre hsrc hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with ⟨_rMemReg, _rSrc, _hcompRest, _hequivRest, hmemReg, _hsrcFromPre⟩
    rcases hmemReg with ⟨_rMem, _rDst, _hcompMemReg, _hequivMemReg, hbytes, hdst⟩
    exact globalLoadPreservesReadReg_single_warpAt hne hctrl hbytes hdst hsrc hstep)

theorem globalLoadBytesRegSpec_single_warpAt_readReg_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset : Nat} {loadBytes frameBytes : List Byte}
    {framePerm : CSL.BytePerm} {oldReg value srcValue : Value}
    (hne : src ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes frameOffset framePerm frameBytes))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes frameOffset framePerm frameBytes))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst oldReg) ∗
          (CSL.reg cta warp lane src srcValue ∗
            CSL.globalBytes frameOffset framePerm frameBytes)))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst value) ∗
          (CSL.reg cta warp lane src srcValue ∗
            CSL.globalBytes frameOffset framePerm frameBytes))) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with
      ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframeFromPre⟩
    rcases hmemReg with ⟨_rMem, _rDst, _hcompMemReg, _hequivMemReg, hbytes, hdst⟩
    rcases hframe with ⟨rSrc, rBytes, hcompFrame, hequivFrame, hsrc, hframeBytes⟩
    exact ⟨rSrc, rBytes, hcompFrame, hequivFrame,
      globalLoadPreservesReadReg_single_warpAt hne hctrl hbytes hdst hsrc hstep,
      globalLoadPreservesGlobalBytes_single_warpAt hctrl hframeBytes hstep⟩)

theorem globalLoadBytesRegSpec_single_warpAt_readRegs2_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset : Nat} {loadBytes frameBytes : List Byte}
    {framePerm : CSL.BytePerm} {oldReg value srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes frameOffset framePerm frameBytes)))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes frameOffset framePerm frameBytes)))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst oldReg) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.globalBytes frameOffset framePerm frameBytes))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst value) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              CSL.globalBytes frameOffset framePerm frameBytes)))) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with
      ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframeFromPre⟩
    rcases hmemReg with ⟨_rMem, _rDst, _hcompMemReg, _hequivMemReg, hbytes, hdst⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂, hsrc₂, hframeBytes⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      globalLoadPreservesReadReg_single_warpAt hne₁ hctrl hbytes hdst hsrc₁ hstep,
      ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂,
        globalLoadPreservesReadReg_single_warpAt hne₂ hctrl hbytes hdst hsrc₂ hstep,
        globalLoadPreservesGlobalBytes_single_warpAt hctrl hframeBytes hstep⟩⟩)

theorem globalLoadBytesRegSpec_single_warpAt_readRegs2_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset₁ frameOffset₂ : Nat}
    {loadBytes frameBytes₁ frameBytes₂ : List Byte}
    {framePerm₁ framePerm₂ : CSL.BytePerm} {oldReg value srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst oldReg) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst value) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with
      ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframeFromPre⟩
    rcases hmemReg with ⟨_rMem, _rDst, _hcompMemReg, _hequivMemReg, hbytes, hdst⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂, hsrc₂, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
      hbytes₁, hbytes₂⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      globalLoadPreservesReadReg_single_warpAt hne₁ hctrl hbytes hdst hsrc₁ hstep,
      ⟨rSrc₂, rBytes, hcompSrc₂, hequivSrc₂,
        globalLoadPreservesReadReg_single_warpAt hne₂ hctrl hbytes hdst hsrc₂ hstep,
        ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
          globalLoadPreservesGlobalBytes_single_warpAt hctrl hbytes₁ hstep,
          globalLoadPreservesGlobalBytes_single_warpAt hctrl hbytes₂ hstep⟩⟩⟩)

theorem globalLoadBytesRegSpec_single_warpAt_readRegs3_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset₁ frameOffset₂ : Nat}
    {loadBytes frameBytes₁ frameBytes₂ : List Byte}
    {framePerm₁ framePerm₂ : CSL.BytePerm}
    {oldReg value srcValue₁ srcValue₂ srcValue₃ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                    CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                    CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst oldReg) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.reg cta warp lane src₃ srcValue₃ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes loadOffset .read loadBytes ∗
          CSL.reg cta warp lane dst value) ∗
          (CSL.reg cta warp lane src₁ srcValue₁ ∗
            (CSL.reg cta warp lane src₂ srcValue₂ ∗
              (CSL.reg cta warp lane src₃ srcValue₃ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) :=
  globalLoadBytesRegSpec_single_warpAt_frame haddr hread (by
    intro st st' r rFrame hpre hframe hstep
    rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, hrest⟩
    rcases hrest with
      ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframeFromPre⟩
    rcases hmemReg with ⟨_rMem, _rDst, _hcompMemReg, _hequivMemReg, hbytes, hdst⟩
    rcases hframe with ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁, hsrc₁, hrestFrame⟩
    rcases hrestFrame with ⟨rSrc₂, rRest₂, hcompSrc₂, hequivSrc₂, hsrc₂, hrest₂⟩
    rcases hrest₂ with ⟨rSrc₃, rBytes, hcompSrc₃, hequivSrc₃, hsrc₃, hbytesFrame⟩
    rcases hbytesFrame with ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
      hbytes₁, hbytes₂⟩
    exact ⟨rSrc₁, rRest, hcompSrc₁, hequivSrc₁,
      globalLoadPreservesReadReg_single_warpAt hne₁ hctrl hbytes hdst hsrc₁ hstep,
      ⟨rSrc₂, rRest₂, hcompSrc₂, hequivSrc₂,
        globalLoadPreservesReadReg_single_warpAt hne₂ hctrl hbytes hdst hsrc₂ hstep,
        ⟨rSrc₃, rBytes, hcompSrc₃, hequivSrc₃,
          globalLoadPreservesReadReg_single_warpAt hne₃ hctrl hbytes hdst hsrc₃ hstep,
          ⟨rBytes₁, rBytes₂, hcompBytes, hequivBytes,
            globalLoadPreservesGlobalBytes_single_warpAt hctrl hbytes₁ hstep,
            globalLoadPreservesGlobalBytes_single_warpAt hctrl hbytes₂ hstep⟩⟩⟩⟩)

theorem globalStoreBytesSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : TypedAddr}
    {value : RValue} {st₀ st₁ : State}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem : CSL.memoryBytes st₁.global.bytes offset newBytes) :
    InstrSpec cta warp { guard? := guard?, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.globalBytes offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.globalBytes offset .write newBytes st r) :=
  storeSpec_of_computed hstep <|
    StateResourceUpdate.globalBytes hlen hmem

theorem sharedLoadBytesRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hmem :
      ∃ ctaState, st₁.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.sharedBytes cta offset .read bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.sharedBytes cta offset .read bytes ∗ CSL.reg cta warp lane dst newReg) st r) :=
  loadSpec_of_computed hstep <|
    StateResourceUpdate.sep
      (StateResourceUpdate.sharedBytes (rfl) hmem)
      (StateResourceUpdate.reg hreg)

theorem sharedLoadBytesRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {st₀ : State}
    {warpState : WarpState} {ctaState : CTAState} {lane : LaneId}
    {laneState : LaneState} {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hcta : st₀.getCTA? cta = some ctaState)
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (hread : ReadMemFact st₀ .shared ty (.shared cta offset) value) :
    InstrSpec cta warp
      { guard? := guard?, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.sharedBytes cta offset .read bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        (CSL.sharedBytes cta offset .read bytes ∗ CSL.reg cta warp lane dst value) st r) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hsep⟩
  subst st
  rcases hsep with ⟨rMem, rReg, hcomp, hequiv, hbytes, hreg⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st₀ cta warp lane
              { space := .shared, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st₀ .shared ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      rcases Helpers.applyToLaneIds?_shared_eq hcta happly with
        ⟨ctaCore, hctaCore, hsharedCore⟩
      rcases Helpers.advanceRunnablePcs?_shared_eq hctaCore hstep with
        ⟨ctaFinal, hctaFinal, hsharedFinal⟩
      have hmemFinal :
          ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
            CSL.memoryBytes ctaState'.shared.bytes offset bytes := by
        refine ⟨ctaFinal, hctaFinal, ?_⟩
        rw [hsharedFinal, hsharedCore]
        exact CSL.sharedBytes_memory hbytes ctaState hcta
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst value) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold ResolvesAddr at haddr
        unfold ReadMemFact at hread
        simp [hlane, haddr, hread] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst value) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      rcases StateResourceUpdate.sharedBytes
          (st₀ := st₀) (st₁ := st') (cta := cta) (offset := offset)
          (perm := CSL.BytePerm.read) (oldBytes := bytes) (newBytes := bytes)
          rfl hmemFinal rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases StateResourceUpdate.reg_written_advanced
          (st₀ := st₀) (stCore := stCore) (st₁ := st')
          (cta := cta) (warp := warp) (lane := lane)
          (name := dst) (old := oldReg) (new := value)
          hcore hstep rReg hreg with
        ⟨rReg', hupdateReg, hreg'⟩
      refine ⟨CSL.Resource.compose rMem' rReg', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose hupdateMem hupdateReg)
      · exact ⟨rMem', rReg',
          CSL.Resource.canCompose_update_right hupdateReg
            (CSL.Resource.canCompose_update_left hupdateMem hcomp),
          CSL.Resource.equiv_refl _, hbytes', hreg'⟩

theorem sharedLoadBytesRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.sharedBytes cta offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.sharedBytes cta offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .shared ty (.shared cta offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        (CSL.sharedBytes cta offset .read bytes ∗
          CSL.reg cta warp lane dst oldReg))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.sharedBytes cta offset .read bytes ∗
          CSL.reg cta warp lane dst value)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  have hrestAll := hrest
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hrest with ⟨rMem, rReg, hcompRest, hequivRest, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases (by
      unfold ReadMemFact Helpers.readMem? at hread'
      cases haccess : (!Typing.typedAccessPreconditions? .shared ty (.shared cta offset)) with
      | true =>
          simp [haccess] at hread'
      | false =>
          cases hwidth : Typing.byteWidth? ty with
          | none =>
              simp [haccess, hwidth] at hread'
          | some width =>
              cases hcta : st.getCTA? cta with
              | none =>
                  simp [haccess, hwidth, Helpers.getSpaceBaseMem?, hcta] at hread'
              | some ctaState =>
                  exact ⟨ctaState, rfl⟩ :
      ∃ ctaState, st.getCTA? cta = some ctaState) with
    ⟨ctaState, hcta⟩
  rcases sharedLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (ctaState := ctaState) (lane := lane) (laneState := laneState)
      (offset := offset) (bytes := bytes) (oldReg := oldReg) (value := value)
      hcta hwarp hlock hpart hlane haddr' hread' st rRest st' ⟨rfl, hrestAll⟩ hstep with
    ⟨rRest', hupdateRest, hrest'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .shared, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .shared ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rRest', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, rRest',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hrest'⟩

theorem sharedLoadBytesRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadSharedValuesFor st cta ty offsets newValues) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc lanes ∗
        (sharedSlices cta offsets .read byteSlices ∗
          regsFor cta warp lanes dst oldValues))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (sharedSlices cta offsets .read byteSlices ∗
          regsFor cta warp lanes dst newValues)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hreads' := hreads st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rRegs, hcompRest, hequivRest, hbytes, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .shared, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .shared ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applySharedLoad hnodup haddrs' hreads' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hsharedPres :
          ∀ ctaState, st.getCTA? cta = some ctaState →
            ∃ ctaState', st'.getCTA? cta = some ctaState' ∧
              ctaState'.shared = ctaState.shared := by
        intro ctaState hcta
        rcases Helpers.applyToLaneIds?_shared_eq hcta hcore with
          ⟨ctaCore, hctaCore, hsharedCore⟩
        rcases Helpers.advanceRunnablePcs?_shared_eq hctaCore hstep with
          ⟨ctaFinal, hctaFinal, hsharedFinal⟩
        exact ⟨ctaFinal, hctaFinal, hsharedFinal.trans hsharedCore⟩
      have hsliceFacts :
          SharedSlicesUpdateFacts st' cta offsets byteSlices byteSlices :=
        SharedSlicesUpdateFacts.of_sharedSlices_readValuesFor_eq
          hsharedPres hbytes hreads'
      rcases StateResourceUpdate.sharedSlices hsliceFacts rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .shared, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .shared ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .shared, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .shared ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rMem' rRegs') :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateMem hupdateRegs)
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rMem' rRegs'), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rMem', rRegs',
            CSL.Resource.canCompose_update_right hupdateRegs
              (CSL.Resource.canCompose_update_left hupdateMem hcompRest),
            CSL.Resource.equiv_refl _, hbytes', hregs'⟩⟩

theorem sharedStoreBytesSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : TypedAddr}
    {value : RValue} {st₀ st₁ : State}
    {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ ctaState, st₁.getCTA? cta = some ctaState ∧
        CSL.memoryBytes ctaState.shared.bytes offset newBytes) :
    InstrSpec cta warp { guard? := guard?, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.sharedBytes cta offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.sharedBytes cta offset .write newBytes st r) :=
  storeSpec_of_computed hstep <|
    StateResourceUpdate.sharedBytes hlen hmem

theorem localLoadBytesRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hmem :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst newReg) st r) :=
  loadSpec_of_computed hstep <|
    StateResourceUpdate.sep
      (StateResourceUpdate.localBytes (rfl) hmem)
      (StateResourceUpdate.reg hreg)

theorem localLoadBytesRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {st₀ : State}
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (hread : ReadMemFact st₀ .local ty (.local cta warp lane offset) value) :
    InstrSpec cta warp
      { guard? := guard?, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst value) st r) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hsep⟩
  subst st
  rcases hsep with ⟨rMem, rReg, hcomp, hequiv, hbytes, hreg⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st₀ cta warp lane
              { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st₀ .local ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hlocalPres :
          ∀ lane old new,
            ((Helpers.resolveAddr? st₀ cta warp lane
                { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st₀ .local ty addr).bind fun value =>
                some (Helpers.writeReg old dst value)) = some new →
            new.localMem = old.localMem := by
        intro lane old new hf
        cases hresolve : Helpers.resolveAddr? st₀ cta warp lane
            { space := .local, ty := ty, addr := addrExpr } with
        | none =>
            simp [hresolve] at hf
        | some addr =>
            cases hread' : Helpers.readMem? st₀ .local ty addr with
            | none =>
                simp [hresolve, hread'] at hf
            | some value' =>
                simp [hresolve, hread'] at hf
                rw [← hf]
                simp [Helpers.writeReg]
      rcases Helpers.applyToLaneIds?_lane_localMem_eq
          (hpres := hlocalPres) hlane happly with
        ⟨laneCore, hlaneCore, hlocalCore⟩
      rcases Helpers.advanceRunnablePcs?_lane_localMem_eq hlaneCore hstep with
        ⟨laneFinal, hlaneFinal, hlocalFinal⟩
      have hmemFinal :
          ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
            CSL.memoryBytes laneState'.localMem.bytes offset bytes := by
        refine ⟨laneFinal, hlaneFinal, ?_⟩
        rw [hlocalFinal, hlocalCore]
        exact CSL.localBytes_memory hbytes laneState hlane
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst value) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold ResolvesAddr at haddr
        unfold ReadMemFact at hread
        simp [hlane, haddr, hread] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst value) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      rcases StateResourceUpdate.localBytes
          (st₀ := st₀) (st₁ := st') (cta := cta) (warp := warp)
          (lane := lane) (offset := offset)
          (perm := CSL.BytePerm.read) (oldBytes := bytes) (newBytes := bytes)
          rfl hmemFinal rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases StateResourceUpdate.reg_written_advanced
          (st₀ := st₀) (stCore := stCore) (st₁ := st')
          (cta := cta) (warp := warp) (lane := lane)
          (name := dst) (old := oldReg) (new := value)
          hcore hstep rReg hreg with
        ⟨rReg', hupdateReg, hreg'⟩
      refine ⟨CSL.Resource.compose rMem' rReg', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose hupdateMem hupdateReg)
      · exact ⟨rMem', rReg',
          CSL.Resource.canCompose_update_right hupdateReg
            (CSL.Resource.canCompose_update_left hupdateMem hcomp),
          CSL.Resource.equiv_refl _, hbytes', hreg'⟩

theorem localLoadBytesRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .local ty (.local cta warp lane offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        (CSL.localBytes cta warp lane offset .read bytes ∗
          CSL.reg cta warp lane dst oldReg))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.localBytes cta warp lane offset .read bytes ∗
          CSL.reg cta warp lane dst value)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  have hrestAll := hrest
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hrest with ⟨rMem, rReg, hcompRest, hequivRest, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases localLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (offset := offset) (bytes := bytes)
      (oldReg := oldReg) (value := value)
      hwarp hlock hpart hlane haddr' hread' st rRest st' ⟨rfl, hrestAll⟩ hstep with
    ⟨rRest', hupdateRest, hrest'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .local ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rRest', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, rRest',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hrest'⟩

theorem localLoadBytesRegSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadLocalValuesFor st cta warp ty lanes offsets newValues) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc lanes ∗
        (localSlices cta warp lanes offsets .read byteSlices ∗
          regsFor cta warp lanes dst oldValues))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (localSlices cta warp lanes offsets .read byteSlices ∗
          regsFor cta warp lanes dst newValues)) := by
  intro st r st' hpre hstep
  have haddrs' := haddrs st r hpre
  have hreads' := hreads st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rMem, rRegs, hcompRest, hequivRest, hbytes, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.resolveAddr? st cta warp lane
              { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st .local ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyLocalLoad hnodup haddrs' hreads' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hlocalPres :
          ∀ lane old new,
            ((Helpers.resolveAddr? st cta warp lane
                { space := .local, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .local ty addr).bind fun value =>
                some (Helpers.writeReg old dst value)) = some new →
            new.localMem = old.localMem := by
        intro lane old new hf
        cases haddr :
            Helpers.resolveAddr? st cta warp lane
              { space := .local, ty := ty, addr := addrExpr } with
        | none =>
            simp [haddr] at hf
        | some addr =>
            cases hread : Helpers.readMem? st .local ty addr with
            | none =>
                simp [haddr, hread] at hf
            | some value =>
                simp [haddr, hread] at hf
                subst new
                simp [Helpers.writeReg]
      have hlocal :
          ∀ lane laneState, st.getLane? cta warp lane = some laneState →
            ∃ laneState', st'.getLane? cta warp lane = some laneState' ∧
              laneState'.localMem = laneState.localMem := by
        intro lane laneState hlane
        rcases Helpers.applyToLaneIds?_lane_localMem_eq
            (hpres := hlocalPres) hlane hcore with
          ⟨laneCore, hlaneCore, hlocalCore⟩
        rcases Helpers.advanceRunnablePcs?_lane_localMem_eq hlaneCore hstep with
          ⟨laneFinal, hlaneFinal, hlocalFinal⟩
        exact ⟨laneFinal, hlaneFinal, hlocalFinal.trans hlocalCore⟩
      have hsliceFacts :
          LocalSlicesUpdateFacts st' cta warp lanes offsets byteSlices byteSlices :=
        LocalSlicesUpdateFacts.of_localSlices_readValuesFor_eq hlocal hbytes hreads'
      rcases StateResourceUpdate.localSlices hsliceFacts rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .local, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .local ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases haddr :
                Helpers.resolveAddr? st cta warp lane
                  { space := .local, ty := ty, addr := addrExpr } with
            | none =>
                simp [haddr] at hf
            | some addr =>
                cases hread : Helpers.readMem? st .local ty addr with
                | none =>
                    simp [haddr, hread] at hf
                | some value =>
                    simp [haddr, hread] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rMem' rRegs') :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateMem hupdateRegs)
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rMem' rRegs'), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rMem' rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rMem', rRegs',
            CSL.Resource.canCompose_update_right hupdateRegs
              (CSL.Resource.canCompose_update_left hupdateMem hcompRest),
            CSL.Resource.equiv_refl _, hbytes', hregs'⟩⟩

theorem localStoreBytesSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : TypedAddr}
    {value : RValue} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {oldBytes newBytes : List Byte}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .store dst value } =
        some st₁)
    (hlen : oldBytes.length = newBytes.length)
    (hmem :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        CSL.memoryBytes laneState.localMem.bytes offset newBytes) :
    InstrSpec cta warp { guard? := guard?, instr := .store dst value }
      (fun st r => st = st₀ ∧ CSL.localBytes cta warp lane offset .write oldBytes st r)
      (fun st r => st = st₁ ∧ CSL.localBytes cta warp lane offset .write newBytes st r) :=
  storeSpec_of_computed hstep <|
    StateResourceUpdate.localBytes hlen hmem

theorem paramLoadBytesRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.param.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst newReg) st r) :=
  loadSpec_of_computed hstep <|
    StateResourceUpdate.sep
      (StateResourceUpdate.paramBytes hmem)
      (StateResourceUpdate.reg hreg)

theorem paramLoadBytesRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {st₀ : State}
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .param, ty := ty, addr := addrExpr } (.param offset))
    (hread : ReadMemFact st₀ .param ty (.param offset) value) :
    InstrSpec cta warp
      { guard? := guard?, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst value) st r) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hsep⟩
  subst st
  rcases hsep with ⟨rMem, rReg, hcomp, hequiv, hbytes, hreg⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st₀ cta warp lane
              { space := .param, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st₀ .param ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hparamCore : stCore.param = st₀.param :=
        Helpers.applyToLaneIds?_param_eq happly
      have hparamFinal : st'.param = stCore.param :=
        Helpers.advanceRunnablePcs?_param_eq hstep
      have hmemFinal : CSL.memoryBytes st'.param.bytes offset bytes := by
        rw [hparamFinal, hparamCore]
        exact CSL.paramBytes_memory hbytes
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst value) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold ResolvesAddr at haddr
        unfold ReadMemFact at hread
        simp [hlane, haddr, hread] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst value) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      rcases StateResourceUpdate.paramBytes
          (st₀ := st₀) (st₁ := st') (offset := offset) (bytes := bytes)
          hmemFinal rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases StateResourceUpdate.reg_written_advanced
          (st₀ := st₀) (stCore := stCore) (st₁ := st')
          (cta := cta) (warp := warp) (lane := lane)
          (name := dst) (old := oldReg) (new := value)
          hcore hstep rReg hreg with
        ⟨rReg', hupdateReg, hreg'⟩
      refine ⟨CSL.Resource.compose rMem' rReg', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose hupdateMem hupdateReg)
      · exact ⟨rMem', rReg',
          CSL.Resource.canCompose_update_right hupdateReg
            (CSL.Resource.canCompose_update_left hupdateMem hcomp),
          CSL.Resource.equiv_refl _, hbytes', hreg'⟩

theorem paramLoadBytesRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .param, ty := ty, addr := addrExpr } (.param offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .param ty (.param offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst value)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  have hrestAll := hrest
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hrest with ⟨rMem, rReg, hcompRest, hequivRest, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases paramLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (offset := offset) (bytes := bytes)
      (oldReg := oldReg) (value := value)
      hwarp hlock hpart hlane haddr' hread' st rRest st' ⟨rfl, hrestAll⟩ hstep with
    ⟨rRest', hupdateRest, hrest'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .param, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .param ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rRest', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, rRest',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hrest'⟩

theorem constLoadBytesRegSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {src : TypedAddr} {st₀ st₁ : State}
    {lane : LaneId} {offset : Nat} {bytes : List Byte} {oldReg newReg : Value}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .load dst src } =
        some st₁)
    (hmem : CSL.memoryBytes st₁.const.bytes offset bytes)
    (hreg :
      ∃ laneState, st₁.getLane? cta warp lane = some laneState ∧
        laneState.regs[dst]? = some newReg) :
    InstrSpec cta warp { guard? := guard?, instr := .load dst src }
      (fun st r =>
        st = st₀ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        st = st₁ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst newReg) st r) :=
  loadSpec_of_computed hstep <|
    StateResourceUpdate.sep
      (StateResourceUpdate.constBytes hmem)
      (StateResourceUpdate.reg hreg)

theorem constLoadBytesRegSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {st₀ : State}
    {warpState : WarpState} {lane : LaneId} {laneState : LaneState}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (haddr :
      ResolvesAddr st₀ { cta := cta, warp := warp, lane := lane }
        { space := .const, ty := ty, addr := addrExpr } (.const offset))
    (hread : ReadMemFact st₀ .const ty (.const offset) value) :
    InstrSpec cta warp
      { guard? := guard?, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
      (fun st r =>
        st = st₀ ∧
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg) st r)
      (fun st r =>
        (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst value) st r) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hsep⟩
  subst st
  rcases hsep with ⟨rMem, rReg, hcomp, hequiv, hbytes, hreg⟩
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.resolveAddr? st₀ cta warp lane
              { space := .const, ty := ty, addr := addrExpr }).bind fun addr =>
            (Helpers.readMem? st₀ .const ty addr).bind fun value =>
              some (Helpers.writeReg laneState dst value)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hconstCore : stCore.const = st₀.const :=
        Helpers.applyToLaneIds?_const_eq happly
      have hconstFinal : st'.const = stCore.const :=
        Helpers.advanceRunnablePcs?_const_eq hstep
      have hmemFinal : CSL.memoryBytes st'.const.bytes offset bytes := by
        rw [hconstFinal, hconstCore]
        exact CSL.constBytes_memory hbytes
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst value) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold ResolvesAddr at haddr
        unfold ReadMemFact at hread
        simp [hlane, haddr, hread] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst value) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      rcases StateResourceUpdate.constBytes
          (st₀ := st₀) (st₁ := st') (offset := offset) (bytes := bytes)
          hmemFinal rMem hbytes with
        ⟨rMem', hupdateMem, hbytes'⟩
      rcases StateResourceUpdate.reg_written_advanced
          (st₀ := st₀) (stCore := stCore) (st₁ := st')
          (cta := cta) (warp := warp) (lane := lane)
          (name := dst) (old := oldReg) (new := value)
          hcore hstep rReg hreg with
        ⟨rReg', hupdateReg, hreg'⟩
      refine ⟨CSL.Resource.compose rMem' rReg', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose hupdateMem hupdateReg)
      · exact ⟨rMem', rReg',
          CSL.Resource.canCompose_update_right hupdateReg
            (CSL.Resource.canCompose_update_left hupdateMem hcomp),
          CSL.Resource.equiv_refl _, hbytes', hreg'⟩

theorem constLoadBytesRegSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .const, ty := ty, addr := addrExpr } (.const offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .const ty (.const offset) value) :
    InstrSpec cta warp
      { guard? := none, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
      (warpAt cta warp pc [lane] ∗
        (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst value)) := by
  intro st r st' hpre hstep
  have haddr' := haddr st r hpre
  have hread' := hread st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  have hrestAll := hrest
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases hrest with ⟨rMem, rReg, hcompRest, hequivRest, hbytes, hreg⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hreadReg⟩
  rcases constLoadBytesRegSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (ty := ty) (addrExpr := addrExpr) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (offset := offset) (bytes := bytes)
      (oldReg := oldReg) (value := value)
      hwarp hlock hpart hlane haddr' hread' st rRest st' ⟨rfl, hrestAll⟩ hstep with
    ⟨rRest', hupdateRest, hrest'⟩
  have hctrlFinal : warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.resolveAddr? st cta warp lane
                { space := .const, ty := ty, addr := addrExpr }).bind fun addr =>
              (Helpers.readMem? st .const ty addr).bind fun value =>
                some (Helpers.writeReg laneState dst value)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst value) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold ResolvesAddr at haddr'
          unfold ReadMemFact at hread'
          simp [hlane, haddr', hread'] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rRest', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, rRest',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hrest'⟩

theorem cvtaSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .cvta dst space src } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .cvta dst space src }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem cvtaSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : RegName}
    {space : AddrSpace} {src : RValue} {st₀ : State} {warpState : WarpState}
    {lane : LaneId} {laneState : LaneState} {old new srcValue : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    InstrSpec cta warp { guard? := guard?, instr := .cvta dst space src }
      (fun st r => st = st₀ ∧ CSL.reg cta warp lane dst old st r)
      (CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hreg⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st₀ cta warp lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writeReg laneState dst new) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold EvalRValue at heval
        simp [hlane, heval, hcvta] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writeReg laneState dst new) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      exact StateResourceUpdate.reg_written_advanced hcore hstep r hreg

theorem cvtaSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {old new srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.reg cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rReg, hcomp, hequiv, hctrl, hreg⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  rcases cvtaSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (space := space) (src := src) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (old := old) (new := new)
      (srcValue := srcValue)
      hwarp hlock hpart hlane heval' hcvta st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane src).bind fun value =>
              (Helpers.evalCvta? space value).bind fun gaddr =>
                some (Helpers.writeReg laneState dst gaddr)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hcvta] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rReg', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateReg)
  · exact ⟨CSL.Resource.empty, rReg',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hreg'⟩

theorem isspacepSpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .isspacep dst space src } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .isspacep dst space src }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem isspacepSpec_single_of_eval
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {dst : PredName}
    {space : AddrSpace} {src : RValue} {st₀ : State} {warpState : WarpState}
    {lane : LaneId} {laneState : LaneState} {old new : Bool} {srcValue : Value}
    (hwarp : st₀.getWarp? cta warp = some warpState)
    (hlock : Helpers.lockstepRunnable warpState)
    (hpart : Helpers.ParticipatingRunnable warpState guard? [lane])
    (hlane : st₀.getLane? cta warp lane = some laneState)
    (heval : EvalRValue st₀ { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    InstrSpec cta warp { guard? := guard?, instr := .isspacep dst space src }
      (fun st r => st = st₀ ∧ CSL.pred cta warp lane dst old st r)
      (CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  rcases hpre with ⟨hst, hpred⟩
  subst st
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState guard? = some [lane] :=
    (Helpers.participatingRunnable_iff_bool warpState guard? [lane]).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases happly :
      Helpers.applyToLaneIds? st₀ cta warp [lane]
        (fun lane laneState =>
          (Helpers.evalRValue? st₀ cta warp lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [happly] at hstep
      simp at hstep
  | some stCore =>
      rw [happly] at hstep
      simp at hstep
      have hcore :
          ∃ laneState,
            stCore.getLane? cta warp lane = some (Helpers.writePred laneState dst new) := by
        refine ⟨laneState, ?_⟩
        have happlySet := happly
        unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
        unfold EvalRValue at heval
        simp [hlane, heval, hisspace] at happlySet
        have hset :
            st₀.setLane cta warp lane (Helpers.writePred laneState dst new) =
              some stCore := by
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact State.getLane?_setLane_same hlane hset
      exact StateResourceUpdate.pred_written_advanced hcore hstep r hpred

theorem isspacepSpec_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {old new : Bool}
    {srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old)
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        CSL.pred cta warp lane dst new) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rPred, hcomp, hequiv, hctrl, hpred⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases isspacepSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (space := space) (src := src) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (old := old) (new := new)
      (srcValue := srcValue)
      hwarp hlock hpart hlane heval' hisspace st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane src).bind fun value =>
              (Helpers.evalIsspacep? space value).bind fun b =>
                some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hisspace] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  refine ⟨CSL.Resource.compose CSL.Resource.empty rPred', ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePred)
  · exact ⟨CSL.Resource.empty, rPred',
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal, hpred'⟩

theorem cvtaSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .cvta dst space src } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗ frame)) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rReg, rFrame, hcompRest, hequivRest, hreg, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.reg_state hreg with ⟨laneState, hlane, _hread⟩
  rcases cvtaSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (space := space) (src := src) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (old := old) (new := new)
      (srcValue := srcValue)
      hwarp hlock hpart hlane heval' hcvta st rReg st' ⟨rfl, hreg⟩ hstep with
    ⟨rReg', hupdateReg, hreg'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane src).bind fun value =>
              (Helpers.evalCvta? space value).bind fun gaddr =>
                some (Helpers.writeReg laneState dst gaddr)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writeReg laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hcvta] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writeReg]) (by simp [Helpers.writeReg]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rReg, rFrame, hcompRest, hequivRest, hreg, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rReg' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdateReg (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rReg' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rReg' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rReg', rFrame, CSL.Resource.canCompose_update_left hupdateReg hcompRest,
        CSL.Resource.equiv_refl _, hreg', hframeFinal⟩⟩

theorem cvtaSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .cvta dst space src }) frame) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc [lane] ∗
        (CSL.reg cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg cta warp lane dst new ∗ frame)) :=
  cvtaSpec_single_warpAt_frame heval hcvta (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem isspacepSpec_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new : Bool} {srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .isspacep dst space src } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) := by
  intro st r st' hpre hstep
  have heval' := heval st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  rcases CSL.pred_state hpred with ⟨laneState, hlane, _hread⟩
  rcases isspacepSpec_single_of_eval
      (cta := cta) (warp := warp) (guard? := none) (dst := dst)
      (space := space) (src := src) (st₀ := st) (warpState := warpState)
      (lane := lane) (laneState := laneState) (old := old) (new := new)
      (srcValue := srcValue)
      hwarp hlock hpart hlane heval' hisspace st rPred st' ⟨rfl, hpred⟩ hstep with
    ⟨rPred', hupdatePred, hpred'⟩
  have hctrlFinal :
      warpAt cta warp (pc.1, pc.2 + 1) [lane] st' CSL.Resource.empty := by
    have hlockBool : Helpers.lockstepRunnable? warpState = true :=
      (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
    have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some [lane] :=
      (Helpers.participatingRunnable_iff_bool warpState none [lane]).1 hpart
    unfold Helpers.stepInstr? at hstep
    simp [hwarp, hlockBool, hpartOpt] at hstep
    cases happly :
        Helpers.applyToLaneIds? st cta warp [lane]
          (fun lane laneState =>
            (Helpers.evalRValue? st cta warp lane src).bind fun value =>
              (Helpers.evalIsspacep? space value).bind fun b =>
                some (Helpers.writePred laneState dst b)) with
    | none =>
        rw [happly] at hstep
        simp at hstep
    | some stCore =>
        rw [happly] at hstep
        simp at hstep
        have hset :
            st.setLane cta warp lane (Helpers.writePred laneState dst new) = some stCore := by
          have happlySet := happly
          unfold Helpers.applyToLaneIds? Helpers.applyToLaneIdsList? at happlySet
          unfold EvalRValue at heval'
          simp [hlane, heval', hisspace] at happlySet
          simpa [Helpers.applyToLaneIdsList?] using happlySet
        exact warpAt_single_of_advance_setLane hwarp hlock hrpc hpart hlane hset
          (by simp [Helpers.writePred]) (by simp [Helpers.writePred]) hstep
  have hframeFinal : frame st' rFrame := hframe st st' r rFrame
    ⟨rCtrl, rRest, hcomp, hequiv, hctrl,
      ⟨rPred, rFrame, hcompRest, hequivRest, hpred, hframeSt⟩⟩ hframeSt hstep
  have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
  subst rCtrl
  have hupdateRest :
      CSL.Resource.Update rRest (CSL.Resource.compose rPred' rFrame) :=
    CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
      (CSL.Resource.update_compose hupdatePred (CSL.Resource.update_refl rFrame))
  refine ⟨CSL.Resource.compose CSL.Resource.empty
      (CSL.Resource.compose rPred' rFrame), ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
  · exact ⟨CSL.Resource.empty, CSL.Resource.compose rPred' rFrame,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, hctrlFinal,
      ⟨rPred', rFrame, CSL.Resource.canCompose_update_left hupdatePred hcompRest,
        CSL.Resource.equiv_refl _, hpred', hframeFinal⟩⟩

theorem isspacepSpec_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new : Bool} {srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .isspacep dst space src }) frame) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc [lane] ∗
        (CSL.pred cta warp lane dst old ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
        (CSL.pred cta warp lane dst new ∗ frame)) :=
  isspacepSpec_single_warpAt_frame heval hisspace (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem cvtaSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        regsFor cta warp lanes dst newValues) := by
  intro st r st' hpre hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRegs, hcomp, hequiv, hctrl, hregs⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyCvta hnodup hevals' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hcvta : Helpers.evalCvta? space value with
                | none =>
                    simp [heval, hcvta] at hf
                | some gaddr =>
                    simp [heval, hcvta] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hcvta : Helpers.evalCvta? space value with
                | none =>
                    simp [heval, hcvta] at hf
                | some gaddr =>
                    simp [heval, hcvta] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      refine ⟨CSL.Resource.compose CSL.Resource.empty rRegs', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRegs)
      · exact ⟨CSL.Resource.empty, rRegs',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal, hregs'⟩

theorem cvtaSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .cvta dst space src } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc lanes ∗
        (regsFor cta warp lanes dst oldValues ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (regsFor cta warp lanes dst newValues ∗ frame)) := by
  intro st r st' hpre hstep
  have hpreOrig := hpre
  have hstepOrig := hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rRegs, rFrame, hcompRest, hequivRest, hregs, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane src).bind fun value =>
            (Helpers.evalCvta? space value).bind fun gaddr =>
              some (Helpers.writeReg laneState dst gaddr)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : RegsUpdateFacts stCore cta warp dst lanes newValues :=
        RegsUpdateFacts.of_applyCvta hnodup hevals' hcore
      have hfactsFinal : RegsUpdateFacts st' cta warp dst lanes newValues :=
        RegsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.regsFor hfactsFinal rRegs hregs with
        ⟨rRegs', hupdateRegs, hregs'⟩
      have hframeFinal : frame st' rFrame :=
        hframe st st' r rFrame hpreOrig hframeSt hstepOrig
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hcvta : Helpers.evalCvta? space value with
                | none =>
                    simp [heval, hcvta] at hf
                | some gaddr =>
                    simp [heval, hcvta] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hcvta : Helpers.evalCvta? space value with
                | none =>
                    simp [heval, hcvta] at hf
                | some gaddr =>
                    simp [heval, hcvta] at hf
                    subst new
                    simp [Helpers.writeReg])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rRegs' rFrame) :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdateRegs (CSL.Resource.update_refl rFrame))
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rRegs' rFrame), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rRegs' rFrame,
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rRegs', rFrame, CSL.Resource.canCompose_update_left hupdateRegs hcompRest,
            CSL.Resource.equiv_refl _, hregs', hframeFinal⟩⟩

theorem cvtaSpec_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .cvta dst space src }) frame) :
    InstrSpec cta warp { guard? := none, instr := .cvta dst space src }
      (warpAt cta warp pc lanes ∗
        (regsFor cta warp lanes dst oldValues ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (regsFor cta warp lanes dst newValues ∗ frame)) :=
  cvtaSpec_lanes_warpAt_frame hevals (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem isspacepSpec_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues)
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        predsFor cta warp lanes dst newValues) := by
  intro st r st' hpre hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rPreds, hcomp, hequiv, hctrl, hpreds⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : PredsUpdateFacts stCore cta warp dst lanes newValues :=
        PredsUpdateFacts.of_applyIsspacep hnodup hevals' hcore
      have hfactsFinal : PredsUpdateFacts st' cta warp dst lanes newValues :=
        PredsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.predsFor hfactsFinal rPreds hpreds with
        ⟨rPreds', hupdatePreds, hpreds'⟩
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hisspace : Helpers.evalIsspacep? space value with
                | none =>
                    simp [heval, hisspace] at hf
                | some b =>
                    simp [heval, hisspace] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hisspace : Helpers.evalIsspacep? space value with
                | none =>
                    simp [heval, hisspace] at hf
                | some b =>
                    simp [heval, hisspace] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      refine ⟨CSL.Resource.compose CSL.Resource.empty rPreds', ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdatePreds)
      · exact ⟨CSL.Resource.empty, rPreds',
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal, hpreds'⟩

theorem isspacepSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .isspacep dst space src } =
          some st' →
        frame st' rFrame) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc lanes ∗
        (predsFor cta warp lanes dst oldValues ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (predsFor cta warp lanes dst newValues ∗ frame)) := by
  intro st r st' hpre hstep
  have hpreOrig := hpre
  have hstepOrig := hstep
  have hevals' := hevals st r hpre
  rcases hpre with ⟨rCtrl, rRest, hcomp, hequiv, hctrl, hrest⟩
  rcases hrest with ⟨rPreds, rFrame, hcompRest, hequivRest, hpreds, hframeSt⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, hpart⟩
  have hlanesStart : Helpers.runnableLaneIds warpState = lanes :=
    Helpers.runnableLaneIds_eq_of_lockstep_participants_none hlock hpart
  have hnodup : lanes.Nodup := by
    rw [← hlanesStart]
    exact Helpers.runnableLaneIds_nodup warpState
  have hlockBool : Helpers.lockstepRunnable? warpState = true :=
    (Helpers.lockstepRunnable_iff_bool warpState).1 hlock
  have hpartOpt : Helpers.participatingRunnableLaneIds? warpState none = some lanes :=
    (Helpers.participatingRunnable_iff_bool warpState none lanes).1 hpart
  unfold Helpers.stepInstr? at hstep
  simp [hwarp, hlockBool, hpartOpt] at hstep
  cases hcore :
      Helpers.applyToLaneIds? st cta warp lanes
        (fun lane laneState =>
          (Helpers.evalRValue? st cta warp lane src).bind fun value =>
            (Helpers.evalIsspacep? space value).bind fun b =>
              some (Helpers.writePred laneState dst b)) with
  | none =>
      rw [hcore] at hstep
      simp at hstep
  | some stCore =>
      rw [hcore] at hstep
      simp at hstep
      have hfactsCore : PredsUpdateFacts stCore cta warp dst lanes newValues :=
        PredsUpdateFacts.of_applyIsspacep hnodup hevals' hcore
      have hfactsFinal : PredsUpdateFacts st' cta warp dst lanes newValues :=
        PredsUpdateFacts.of_advance hfactsCore hstep
      rcases StateResourceUpdate.predsFor hfactsFinal rPreds hpreds with
        ⟨rPreds', hupdatePreds, hpreds'⟩
      have hframeFinal : frame st' rFrame :=
        hframe st st' r rFrame hpreOrig hframeSt hstepOrig
      rcases Helpers.applyToLaneIds?_warp_control_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hisspace : Helpers.evalIsspacep? space value with
                | none =>
                    simp [heval, hisspace] at hf
                | some b =>
                    simp [heval, hisspace] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hlock hrpc hcore with
        ⟨warpCore, hwarpCore, hlockCore, hrpcCore⟩
      rcases Helpers.applyToLaneIds?_runnableLaneIds_eq
          (hpres := by
            intro lane old new hf
            cases heval : Helpers.evalRValue? st cta warp lane src with
            | none =>
                simp [heval] at hf
            | some value =>
                cases hisspace : Helpers.evalIsspacep? space value with
                | none =>
                    simp [heval, hisspace] at hf
                | some b =>
                    simp [heval, hisspace] at hf
                    subst new
                    simp [Helpers.writePred])
          hwarp hcore with
        ⟨warpCoreRun, hwarpCoreRun, hrunCore⟩
      have hwarpCoreEq : warpCoreRun = warpCore := by
        apply Option.some.inj
        rw [← hwarpCoreRun]
        exact hwarpCore
      subst warpCoreRun
      rcases Helpers.advanceRunnablePcs?_warp_control hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal⟩
      rcases Helpers.advanceRunnablePcs?_runnableLaneIds_eq hwarpCore hlockCore hrpcCore hstep with
        ⟨warpFinalRun, hwarpFinalRun, hrunFinalCore⟩
      have hwarpFinalEq : warpFinalRun = warpFinal := by
        apply Option.some.inj
        rw [← hwarpFinalRun]
        exact hwarpFinal
      subst warpFinalRun
      have hrunFinal : Helpers.runnableLaneIds warpFinal = lanes := by
        rw [hrunFinalCore, hrunCore, hlanesStart]
      have hpartFinal : Helpers.ParticipatingRunnable warpFinal none lanes := by
        unfold Helpers.ParticipatingRunnable
        rw [Helpers.participatingRunnable_none_eq_runnableLaneIds_of_lockstep
          hlockFinal hrpcFinal]
        rw [hrunFinal]
      have hctrlFinal :
          warpAt cta warp (pc.1, pc.2 + 1) lanes st' CSL.Resource.empty :=
        ⟨⟨warpFinal, hwarpFinal, hlockFinal, hrpcFinal, hpartFinal⟩, rfl⟩
      have hemp : CSL.emp st rCtrl := stateProp_emp hctrl
      subst rCtrl
      have hupdateRest :
          CSL.Resource.Update rRest (CSL.Resource.compose rPreds' rFrame) :=
        CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequivRest)
          (CSL.Resource.update_compose hupdatePreds (CSL.Resource.update_refl rFrame))
      refine ⟨CSL.Resource.compose CSL.Resource.empty
          (CSL.Resource.compose rPreds' rFrame), ?_, ?_⟩
      · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
          (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdateRest)
      · exact ⟨CSL.Resource.empty, CSL.Resource.compose rPreds' rFrame,
          by simp [CSL.Resource.canCompose, CSL.Resource.empty],
          CSL.Resource.equiv_refl _, hctrlFinal,
          ⟨rPreds', rFrame, CSL.Resource.canCompose_update_left hupdatePreds hcompRest,
            CSL.Resource.equiv_refl _, hpreds', hframeFinal⟩⟩

theorem isspacepSpec_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .isspacep dst space src }) frame) :
    InstrSpec cta warp { guard? := none, instr := .isspacep dst space src }
      (warpAt cta warp pc lanes ∗
        (predsFor cta warp lanes dst oldValues ∗ frame))
      (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        (predsFor cta warp lanes dst newValues ∗ frame)) :=
  isspacepSpec_lanes_warpAt_frame hevals (by
    intro _st _st' _r rFrame _hpre hframeSt hstep
    exact hframe _st _st' rFrame hstep hframeSt)

theorem wp_assignReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.reg cta warp lane dst new) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt heval)

theorem wp_assignReg_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt_frame heval hframe)

theorem wp_assignReg_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lane : LaneId} {old new : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignReg dst rhs }) frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt_stableFrame heval hframe)

theorem wp_assignReg_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue : Value}
    (hne : src ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ CSL.reg cta warp lane src srcValue)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗ CSL.reg cta warp lane src srcValue)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗ CSL.reg cta warp lane src srcValue)) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt_readReg hne heval)

theorem wp_assignReg_single_warpAt_readRegs2
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              CSL.reg cta warp lane src₂ srcValue₂))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          CSL.reg cta warp lane src₂ srcValue₂))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              CSL.reg cta warp lane src₂ srcValue₂))) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt_readRegs2 hne₁ hne₂ heval)

theorem wp_assignReg_single_warpAt_readRegs2_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes offset perm bytes)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            CSL.globalBytes offset perm bytes)))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes offset perm bytes)))) :=
  wpInstr_of_spec
    (assignRegSpec_single_warpAt_readRegs2_globalBytesFrame hne₁ hne₂ heval)

theorem wp_assignReg_single_warpAt_readRegs2_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset₁ offset₂ : Nat} {perm₁ perm₂ : CSL.BytePerm}
    {bytes₁ bytes₂ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  CSL.globalBytes offset₂ perm₂ bytes₂))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
              CSL.globalBytes offset₂ perm₂ bytes₂))))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  CSL.globalBytes offset₂ perm₂ bytes₂))))) :=
  wpInstr_of_spec
    (assignRegSpec_single_warpAt_readRegs2_globalBytes2Frame hne₁ hne₂ heval)

theorem wp_assignReg_single_warpAt_readRegs2_globalBytes3Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ : Value}
    {offset₁ offset₂ offset₃ : Nat} {perm₁ perm₂ perm₃ : CSL.BytePerm}
    {bytes₁ bytes₂ bytes₃ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                    CSL.globalBytes offset₃ perm₃ bytes₃)))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
              (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                CSL.globalBytes offset₃ perm₃ bytes₃)))))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                  (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                    CSL.globalBytes offset₃ perm₃ bytes₃)))))) :=
  wpInstr_of_spec
    (assignRegSpec_single_warpAt_readRegs2_globalBytes3Frame hne₁ hne₂ heval)

theorem wp_assignReg_single_warpAt_readRegs3
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ srcValue₃ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.reg cta warp lane src₃ srcValue₃)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            CSL.reg cta warp lane src₃ srcValue₃)))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.reg cta warp lane src₃ srcValue₃)))) :=
  wpInstr_of_spec (assignRegSpec_single_warpAt_readRegs3 hne₁ hne₂ hne₃ heval)

theorem wp_assignReg_single_warpAt_readRegs3_globalBytes3Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {rhs : RValue} {lane : LaneId} {old new srcValue₁ srcValue₂ srcValue₃ : Value}
    {offset₁ offset₂ offset₃ : Nat} {perm₁ perm₂ perm₃ : CSL.BytePerm}
    {bytes₁ bytes₂ bytes₃ : List Byte}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                    (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                      CSL.globalBytes offset₃ perm₃ bytes₃))))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs new) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            (CSL.reg cta warp lane src₃ srcValue₃ ∗
              (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                  CSL.globalBytes offset₃ perm₃ bytes₃))))))) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes offset₁ perm₁ bytes₁ ∗
                    (CSL.globalBytes offset₂ perm₂ bytes₂ ∗
                      CSL.globalBytes offset₃ perm₃ bytes₃))))))) :=
  wpInstr_of_spec
    (assignRegSpec_single_warpAt_readRegs3_globalBytes3Frame
      hne₁ hne₂ hne₃ heval)

theorem wp_assignReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) st r →
          EvalRValuesFor st cta warp rhs lanes newValues) :
    (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          regsFor cta warp lanes dst newValues) :=
  wpInstr_of_spec (assignRegSpec_lanes_warpAt hevals)

theorem wp_assignReg_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalRValuesFor st cta warp rhs lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignReg dst rhs } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (regsFor cta warp lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignReg dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (regsFor cta warp lanes dst newValues ∗ frame)) :=
  wpInstr_of_spec (assignRegSpec_lanes_warpAt_frame hevals hframe)

theorem wp_assignPred_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new) :
    (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPred dst cmp }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.pred cta warp lane dst new) :=
  wpInstr_of_spec (assignPredSpec_single_warpAt heval)

theorem wp_assignPred_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignPred dst cmp } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPred dst cmp }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignPredSpec_single_warpAt_frame heval hframe)

theorem wp_assignPred_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lane : LaneId} {old new : Bool} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalCmp st { cta := cta, warp := warp, lane := lane } cmp new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPred dst cmp }) frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPred dst cmp }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignPredSpec_single_warpAt_stableFrame heval hframe)

theorem wp_assignPred_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {cmp : CmpExpr} {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalCmpsFor st cta warp cmp lanes newValues) :
    (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPred dst cmp }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          predsFor cta warp lanes dst newValues) :=
  wpInstr_of_spec (assignPredSpec_lanes_warpAt hevals)

theorem wp_assignPredValue_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new) :
    (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.pred cta warp lane dst new) :=
  wpInstr_of_spec (assignPredValueSpec_single_warpAt heval hbool)

theorem wp_assignPredValue_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .assignPredValue dst rhs } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignPredValueSpec_single_warpAt_frame heval hbool hframe)

theorem wp_assignPredValue_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lane : LaneId} {old new : Bool} {value : Value}
    {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } rhs value)
    (hbool : Helpers.valueToBool? value = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (assignPredValueSpec_single_warpAt_stableFrame heval hbool hframe)

theorem wp_assignPredValue_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues) :
    (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          predsFor cta warp lanes dst newValues) :=
  wpInstr_of_spec (assignPredValueSpec_lanes_warpAt hevals)

theorem wp_assignPredValue_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    ((warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ∗ frame) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          predsFor cta warp lanes dst newValues) ∗ frame) :=
  wpInstr_frame_of_entails (wp_assignPredValue_lanes_warpAt hevals) hframe

theorem wp_assignPredValue_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    ((warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ∗ frame) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .assignPredValue dst rhs }
        ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          predsFor cta warp lanes dst newValues) ∗ frame) :=
  wp_assignPredValue_lanes_warpAt_frame hevals hframe

theorem wp_globalStoreBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    (warpAt cta warp pc [lane] ∗ CSL.globalBytes offset .write oldBytes) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.globalBytes offset .write newBytes) :=
  wpInstr_of_spec (globalStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalSlicesUpdateFacts st' offsets oldSlices newSlices) :
    (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wpInstr_of_spec (globalStoreBytesSpec_lanes_warpAt_of_facts hfacts)

theorem wp_globalStoreBytes_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        GlobalMemoryBytesFor st' offsets newSlices) :
    (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wpInstr_of_spec (globalStoreBytesSpec_lanes_warpAt_of_memory hlens hmems)

theorem wp_globalStoreBytes_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    (warpAt cta warp pc lanes ∗ globalSlices offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          globalSlices offsets .write newSlices) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint)

theorem wp_globalStoreBytes_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (globalSlices offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .write newSlices ∗ frame)) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_lanes_warpAt_frame
      hlens haddrs hevals hencs hdisjoint hframe)

theorem wp_globalStoreBytes_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt cta warp pc lanes ∗
      (globalSlices offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .write newSlices ∗ frame)) :=
  wp_globalStoreBytes_lanes_warpAt_frame
    hlens haddrs hevals hencs hdisjoint (by
      intro st st' _r rFrame _hpre hframeSt hstep
      exact hframe st st' rFrame hstep hframeSt)

theorem wp_globalStoreBytes_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes offset .write oldBytes ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .write newBytes ∗ frame)) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_frame haddr heval hwrite hencode hlen hframe)

theorem wp_globalStoreBytes_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗ frame)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes offset .write oldBytes ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .write newBytes ∗ frame)) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_stableFrame
      haddr heval hwrite hencode hlen hframe)

theorem wp_globalStoreBytes_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value srcValue : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .write oldBytes ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes offset .write oldBytes ∗
        CSL.reg cta warp lane src srcValue)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .write newBytes ∗
            CSL.reg cta warp lane src srcValue)) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_readReg haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_single_warpAt_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset : Nat}
    {oldWriteBytes newWriteBytes readBytes : List Byte} {readPerm : CSL.BytePerm}
    {value : Value}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes writeOffset .write oldWriteBytes ∗
        CSL.globalBytes readOffset readPerm readBytes)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes writeOffset .write newWriteBytes ∗
            CSL.globalBytes readOffset readPerm readBytes)) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_globalBytesFrame
      hdisjoint haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_single_warpAt_readReg_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset : Nat}
    {oldWriteBytes newWriteBytes readBytes : List Byte} {readPerm : CSL.BytePerm}
    {value srcValue : Value}
    (hdisjoint :
      ByteRangesDisjoint readOffset readBytes.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes writeOffset .write oldWriteBytes ∗
        (CSL.reg cta warp lane src srcValue ∗
          CSL.globalBytes readOffset readPerm readBytes))) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes writeOffset .write newWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes readOffset readPerm readBytes))) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_readReg_globalBytesFrame
      hdisjoint haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_single_warpAt_readReg_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm} {value srcValue : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes writeOffset .write oldWriteBytes ∗
        (CSL.reg cta warp lane src srcValue ∗
          (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
            CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes writeOffset .write newWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_readReg_globalBytes2Frame
      hdisjoint₁ hdisjoint₂ haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_single_warpAt_readRegs3_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src reg₁ reg₂ : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm} {value srcValue regValue₁ regValue₂ : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes writeOffset .write oldWriteBytes ∗
        (CSL.reg cta warp lane src srcValue ∗
          (CSL.reg cta warp lane reg₁ regValue₁ ∗
            (CSL.reg cta warp lane reg₂ regValue₂ ∗
              (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes writeOffset .write newWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                    CSL.globalBytes readOffset₂ readPerm₂ readBytes₂)))))) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_readRegs3_globalBytes2Frame
      hdisjoint₁ hdisjoint₂ haddr heval hwrite hencode hlen)

theorem wp_globalStoreBytes_single_warpAt_readRegs4_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {src reg₁ reg₂ reg₃ : RegName}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {writeOffset readOffset₁ readOffset₂ : Nat}
    {oldWriteBytes newWriteBytes readBytes₁ readBytes₂ : List Byte}
    {readPerm₁ readPerm₂ : CSL.BytePerm}
    {value srcValue regValue₁ regValue₂ regValue₃ : Value}
    (hdisjoint₁ :
      ByteRangesDisjoint readOffset₁ readBytes₁.length writeOffset newWriteBytes.length)
    (hdisjoint₂ :
      ByteRangesDisjoint readOffset₂ readBytes₂.length writeOffset newWriteBytes.length)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global writeOffset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes writeOffset .write oldWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) st r →
          ∃ stCore, WriteMemFact st .global ty (.global writeOffset) value stCore)
    (hencode : EncodedScalar ty value newWriteBytes)
    (hlen : oldWriteBytes.length = newWriteBytes.length) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes writeOffset .write oldWriteBytes ∗
        (CSL.reg cta warp lane src srcValue ∗
          (CSL.reg cta warp lane reg₁ regValue₁ ∗
            (CSL.reg cta warp lane reg₂ regValue₂ ∗
              (CSL.reg cta warp lane reg₃ regValue₃ ∗
                (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                  CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .global, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes writeOffset .write newWriteBytes ∗
            (CSL.reg cta warp lane src srcValue ∗
              (CSL.reg cta warp lane reg₁ regValue₁ ∗
                (CSL.reg cta warp lane reg₂ regValue₂ ∗
                  (CSL.reg cta warp lane reg₃ regValue₃ ∗
                    (CSL.globalBytes readOffset₁ readPerm₁ readBytes₁ ∗
                      CSL.globalBytes readOffset₂ readPerm₂ readBytes₂))))))) :=
  wpInstr_of_spec
    (globalStoreBytesSpec_single_warpAt_readRegs4_globalBytes2Frame
      hdisjoint₁ hdisjoint₂ haddr heval hwrite hencode hlen)

theorem wp_sharedStoreBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.sharedBytes cta offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .shared ty (.shared cta offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    (warpAt cta warp pc [lane] ∗ CSL.sharedBytes cta offset .write oldBytes) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.sharedBytes cta offset .write newBytes) :=
  wpInstr_of_spec (sharedStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen)

theorem wp_sharedStoreBytes_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedSlicesUpdateFacts st' cta offsets oldSlices newSlices) :
    (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          sharedSlices cta offsets .write newSlices) :=
  wpInstr_of_spec (sharedStoreBytesSpec_lanes_warpAt_of_facts hfacts)

theorem wp_sharedStoreBytes_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        SharedMemoryBytesFor st' cta offsets newSlices) :
    (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          sharedSlices cta offsets .write newSlices) :=
  wpInstr_of_spec (sharedStoreBytesSpec_lanes_warpAt_of_memory hlens hmems)

theorem wp_sharedStoreBytes_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices) :
    (warpAt cta warp pc lanes ∗ sharedSlices cta offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          sharedSlices cta offsets .write newSlices) :=
  wpInstr_of_spec
    (sharedStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint)

theorem wp_sharedStoreBytes_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (sharedSlices cta offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices cta offsets .write newSlices ∗ frame)) :=
  wpInstr_of_spec
    (sharedStoreBytesSpec_lanes_warpAt_frame
      hlens haddrs hevals hencs hdisjoint hframe)

theorem wp_sharedStoreBytes_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseByteRangesDisjoint offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt cta warp pc lanes ∗
      (sharedSlices cta offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .shared, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices cta offsets .write newSlices ∗ frame)) :=
  wp_sharedStoreBytes_lanes_warpAt_frame
    hlens haddrs hevals hencs hdisjoint (by
      intro st st' _r rFrame _hpre hframeSt hstep
      exact hframe st st' rFrame hstep hframeSt)

theorem wp_localStoreBytes_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue} {lane : LaneId}
    {offset : Nat} {oldBytes newBytes : List Byte} {value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } valueExpr value)
    (hwrite :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          CSL.localBytes cta warp lane offset .write oldBytes) st r →
          ∃ stCore, WriteMemFact st .local ty (.local cta warp lane offset) value stCore)
    (hencode : EncodedScalar ty value newBytes)
    (hlen : oldBytes.length = newBytes.length) :
    (warpAt cta warp pc [lane] ∗
      CSL.localBytes cta warp lane offset .write oldBytes) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.localBytes cta warp lane offset .write newBytes) :=
  wpInstr_of_spec (localStoreBytesSpec_single_warpAt haddr heval hwrite hencode hlen)

theorem wp_localStoreBytes_lanes_warpAt_of_facts
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hfacts :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalSlicesUpdateFacts st' cta warp lanes offsets oldSlices newSlices) :
    (warpAt cta warp pc lanes ∗
      localSlices cta warp lanes offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          localSlices cta warp lanes offsets .write newSlices) :=
  wpInstr_of_spec (localStoreBytesSpec_lanes_warpAt_of_facts hfacts)

theorem wp_localStoreBytes_lanes_warpAt_of_memory
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (hmems :
      ∀ st r st',
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        LocalMemoryBytesFor st' cta warp lanes offsets newSlices) :
    (warpAt cta warp pc lanes ∗
      localSlices cta warp lanes offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          localSlices cta warp lanes offsets .write newSlices) :=
  wpInstr_of_spec (localStoreBytesSpec_lanes_warpAt_of_memory hlens hmems)

theorem wp_localStoreBytes_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          localSlices cta warp lanes offsets .write oldSlices) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices) :
    (warpAt cta warp pc lanes ∗
      localSlices cta warp lanes offsets .write oldSlices) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          localSlices cta warp lanes offsets .write newSlices) :=
  wpInstr_of_spec
    (localStoreBytesSpec_lanes_warpAt hlens haddrs hevals hencs hdisjoint)

theorem wp_localStoreBytes_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (localSlices cta warp lanes offsets .write newSlices ∗ frame)) :=
  wpInstr_of_spec
    (localStoreBytesSpec_lanes_warpAt_frame
      hlens haddrs hevals hencs hdisjoint hframe)

theorem wp_localStoreBytes_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC}
    {ty : ScalarTy} {addrExpr valueExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat}
    {oldSlices newSlices : List (List Byte)} {values : List Value}
    {frame : CSL.Assertion}
    (hlens : SliceLengthsEq oldSlices newSlices)
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) st r →
          EvalRValuesFor st cta warp valueExpr lanes values)
    (hencs : EncodedScalarsFor ty values newSlices)
    (hdisjoint : PairwiseLocalByteRangesDisjoint lanes offsets newSlices)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr })
        frame) :
    (warpAt cta warp pc lanes ∗
      (localSlices cta warp lanes offsets .write oldSlices ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none,
          instr := .store { space := .local, ty := ty, addr := addrExpr } valueExpr }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (localSlices cta warp lanes offsets .write newSlices ∗ frame)) :=
  wp_localStoreBytes_lanes_warpAt_frame
    hlens haddrs hevals hencs hdisjoint (by
      intro st st' _r rFrame _hpre hframeSt hstep
      exact hframe st st' rFrame hstep hframeSt)

theorem wp_globalLoadBytesReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    (warpAt cta warp pc [lane] ∗
      (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes offset .read bytes ∗ CSL.reg cta warp lane dst value)) :=
  wpInstr_of_spec (globalLoadBytesRegSpec_single_warpAt haddr hread)

theorem wp_globalLoadBytesReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues) :
    (warpAt cta warp pc lanes ∗
      (globalSlices offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) :=
  wpInstr_of_spec (globalLoadBytesRegSpec_lanes_warpAt haddrs hreads)

theorem wp_globalLoadBytesReg_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesGlobalAddrsFor st cta warp
            { space := .global, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadGlobalValuesFor st ty offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .load dst { space := .global, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt cta warp pc lanes ∗
      (globalSlices offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (globalSlices offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) ∗ frame) :=
  wpInstr_frame_of_entails
    (wp_globalLoadBytesReg_lanes_warpAt haddrs hreads)
    hframe

theorem wp_globalLoadBytesReg_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ReadMemFact st .global ty (.global offset) value)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp
          { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes offset .read bytes ∗
        CSL.reg cta warp lane dst oldReg) ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst value) ∗ frame)) :=
  wpInstr_of_spec (globalLoadBytesRegSpec_single_warpAt_frame haddr hread hframe)

theorem wp_globalLoadBytesReg_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    {frame : CSL.Assertion}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗ frame)) st r →
          ReadMemFact st .global ty (.global offset) value)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .load dst { space := .global, ty := ty, addr := addrExpr } })
        frame) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes offset .read bytes ∗
        CSL.reg cta warp lane dst oldReg) ∗ frame)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst value) ∗ frame)) :=
  wpInstr_of_spec (globalLoadBytesRegSpec_single_warpAt_stableFrame haddr hread hframe)

theorem wp_globalLoadBytesReg_single_warpAt_readReg
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value srcValue : Value}
    (hne : src ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            CSL.reg cta warp lane src srcValue)) st r →
          ReadMemFact st .global ty (.global offset) value) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes offset .read bytes ∗
        CSL.reg cta warp lane dst oldReg) ∗
        CSL.reg cta warp lane src srcValue)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes offset .read bytes ∗
            CSL.reg cta warp lane dst value) ∗
            CSL.reg cta warp lane src srcValue)) :=
  wpInstr_of_spec (globalLoadBytesRegSpec_single_warpAt_readReg hne haddr hread)

theorem wp_globalLoadBytesReg_single_warpAt_readReg_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset : Nat} {loadBytes frameBytes : List Byte}
    {framePerm : CSL.BytePerm} {oldReg value srcValue : Value}
    (hne : src ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes frameOffset framePerm frameBytes))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes frameOffset framePerm frameBytes))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes loadOffset .read loadBytes ∗
        CSL.reg cta warp lane dst oldReg) ∗
        (CSL.reg cta warp lane src srcValue ∗
          CSL.globalBytes frameOffset framePerm frameBytes))) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst value) ∗
            (CSL.reg cta warp lane src srcValue ∗
              CSL.globalBytes frameOffset framePerm frameBytes))) :=
  wpInstr_of_spec
    (globalLoadBytesRegSpec_single_warpAt_readReg_globalBytesFrame hne haddr hread)

theorem wp_globalLoadBytesReg_single_warpAt_readRegs2_globalBytesFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset : Nat} {loadBytes frameBytes : List Byte}
    {framePerm : CSL.BytePerm} {oldReg value srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes frameOffset framePerm frameBytes)))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes frameOffset framePerm frameBytes)))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes loadOffset .read loadBytes ∗
        CSL.reg cta warp lane dst oldReg) ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            CSL.globalBytes frameOffset framePerm frameBytes)))) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst value) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                CSL.globalBytes frameOffset framePerm frameBytes)))) :=
  wpInstr_of_spec
    (globalLoadBytesRegSpec_single_warpAt_readRegs2_globalBytesFrame
      hne₁ hne₂ haddr hread)

theorem wp_globalLoadBytesReg_single_warpAt_readRegs2_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset₁ frameOffset₂ : Nat}
    {loadBytes frameBytes₁ frameBytes₂ : List Byte}
    {framePerm₁ framePerm₂ : CSL.BytePerm} {oldReg value srcValue₁ srcValue₂ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes loadOffset .read loadBytes ∗
        CSL.reg cta warp lane dst oldReg) ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
              CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst value) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                  CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂))))) :=
  wpInstr_of_spec
    (globalLoadBytesRegSpec_single_warpAt_readRegs2_globalBytes2Frame
      hne₁ hne₂ haddr hread)

theorem wp_globalLoadBytesReg_single_warpAt_readRegs3_globalBytes2Frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst src₁ src₂ src₃ : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {loadOffset frameOffset₁ frameOffset₂ : Nat}
    {loadBytes frameBytes₁ frameBytes₂ : List Byte}
    {framePerm₁ framePerm₂ : CSL.BytePerm}
    {oldReg value srcValue₁ srcValue₂ srcValue₃ : Value}
    (hne₁ : src₁ ≠ dst)
    (hne₂ : src₂ ≠ dst)
    (hne₃ : src₃ ≠ dst)
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                    CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .global, ty := ty, addr := addrExpr } (.global loadOffset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst oldReg) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                    CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) st r →
          ReadMemFact st .global ty (.global loadOffset) value) :
    (warpAt cta warp pc [lane] ∗
      ((CSL.globalBytes loadOffset .read loadBytes ∗
        CSL.reg cta warp lane dst oldReg) ∗
        (CSL.reg cta warp lane src₁ srcValue₁ ∗
          (CSL.reg cta warp lane src₂ srcValue₂ ∗
            (CSL.reg cta warp lane src₃ srcValue₃ ∗
              (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .global, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes loadOffset .read loadBytes ∗
            CSL.reg cta warp lane dst value) ∗
            (CSL.reg cta warp lane src₁ srcValue₁ ∗
              (CSL.reg cta warp lane src₂ srcValue₂ ∗
                (CSL.reg cta warp lane src₃ srcValue₃ ∗
                  (CSL.globalBytes frameOffset₁ framePerm₁ frameBytes₁ ∗
                    CSL.globalBytes frameOffset₂ framePerm₂ frameBytes₂)))))) :=
  wpInstr_of_spec
    (globalLoadBytesRegSpec_single_warpAt_readRegs3_globalBytes2Frame
      hne₁ hne₂ hne₃ haddr hread)

theorem wp_sharedLoadBytesReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.sharedBytes cta offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .shared, ty := ty, addr := addrExpr } (.shared cta offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.sharedBytes cta offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .shared ty (.shared cta offset) value) :
    (warpAt cta warp pc [lane] ∗
      (CSL.sharedBytes cta offset .read bytes ∗ CSL.reg cta warp lane dst oldReg)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.sharedBytes cta offset .read bytes ∗
            CSL.reg cta warp lane dst value)) :=
  wpInstr_of_spec (sharedLoadBytesRegSpec_single_warpAt haddr hread)

theorem wp_sharedLoadBytesReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadSharedValuesFor st cta ty offsets newValues) :
    (warpAt cta warp pc lanes ∗
      (sharedSlices cta offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) :=
  wpInstr_of_spec (sharedLoadBytesRegSpec_lanes_warpAt haddrs hreads)

theorem wp_sharedLoadBytesReg_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesSharedAddrsFor st cta warp
            { space := .shared, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadSharedValuesFor st cta ty offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .load dst { space := .shared, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt cta warp pc lanes ∗
      (sharedSlices cta offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .shared, ty := ty, addr := addrExpr } }
        ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (sharedSlices cta offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) ∗ frame) :=
  wpInstr_frame_of_entails
    (wp_sharedLoadBytesReg_lanes_warpAt haddrs hreads)
    hframe

theorem wp_localLoadBytesReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .local, ty := ty, addr := addrExpr } (.local cta warp lane offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .local ty (.local cta warp lane offset) value) :
    (warpAt cta warp pc [lane] ∗
      (CSL.localBytes cta warp lane offset .read bytes ∗
        CSL.reg cta warp lane dst oldReg)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.localBytes cta warp lane offset .read bytes ∗
            CSL.reg cta warp lane dst value)) :=
  wpInstr_of_spec (localLoadBytesRegSpec_single_warpAt haddr hread)

theorem wp_localLoadBytesReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadLocalValuesFor st cta warp ty lanes offsets newValues) :
    (warpAt cta warp pc lanes ∗
      (localSlices cta warp lanes offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) :=
  wpInstr_of_spec (localLoadBytesRegSpec_lanes_warpAt haddrs hreads)

theorem wp_localLoadBytesReg_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value} {frame : CSL.Assertion}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesLocalAddrsFor st cta warp
            { space := .local, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadLocalValuesFor st cta warp ty lanes offsets newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp
          { guard? := none,
            instr := .load dst { space := .local, ty := ty, addr := addrExpr } })
        frame) :
    ((warpAt cta warp pc lanes ∗
      (localSlices cta warp lanes offsets .read byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ∗ frame) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .local, ty := ty, addr := addrExpr } }
        ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (localSlices cta warp lanes offsets .read byteSlices ∗
            regsFor cta warp lanes dst newValues)) ∗ frame) :=
  wpInstr_frame_of_entails
    (wp_localLoadBytesReg_lanes_warpAt haddrs hreads)
    hframe

theorem wp_paramLoadBytesReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .param, ty := ty, addr := addrExpr } (.param offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .param ty (.param offset) value) :
    (warpAt cta warp pc [lane] ∗
      (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.paramBytes offset bytes ∗ CSL.reg cta warp lane dst value)) :=
  wpInstr_of_spec (paramLoadBytesRegSpec_single_warpAt haddr hread)

theorem wp_paramLoadBytesReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (paramSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesParamAddrsFor st cta warp
            { space := .param, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (paramSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadParamValuesFor st ty offsets newValues) :
    (warpAt cta warp pc lanes ∗
      (paramSlices offsets byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .param, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (paramSlices offsets byteSlices ∗
            regsFor cta warp lanes dst newValues)) :=
  wpInstr_of_spec (paramLoadBytesRegSpec_lanes_warpAt haddrs hreads)

theorem wp_constLoadBytesReg_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue} {lane : LaneId}
    {offset : Nat} {bytes : List Byte} {oldReg value : Value}
    (haddr :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ResolvesAddr st { cta := cta, warp := warp, lane := lane }
            { space := .const, ty := ty, addr := addrExpr } (.const offset))
    (hread :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) st r →
          ReadMemFact st .const ty (.const offset) value) :
    (warpAt cta warp pc [lane] ∗
      (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst oldReg)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.constBytes offset bytes ∗ CSL.reg cta warp lane dst value)) :=
  wpInstr_of_spec (constLoadBytesRegSpec_single_warpAt haddr hread)

theorem wp_constLoadBytesReg_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {ty : ScalarTy} {addrExpr : RValue}
    {lanes : List LaneId} {offsets : List Nat} {byteSlices : List (List Byte)}
    {oldValues newValues : List Value}
    (haddrs :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (constSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ResolvesConstAddrsFor st cta warp
            { space := .const, ty := ty, addr := addrExpr } lanes offsets)
    (hreads :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (constSlices offsets byteSlices ∗
            regsFor cta warp lanes dst oldValues)) st r →
          ReadConstValuesFor st ty offsets newValues) :
    (warpAt cta warp pc lanes ∗
      (constSlices offsets byteSlices ∗
        regsFor cta warp lanes dst oldValues)) ⊢ₛ
      wpInstr cta warp
        { guard? := none, instr := .load dst { space := .const, ty := ty, addr := addrExpr } }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (constSlices offsets byteSlices ∗
            regsFor cta warp lanes dst newValues)) :=
  wpInstr_of_spec (constLoadBytesRegSpec_lanes_warpAt haddrs hreads)

theorem wp_cvta_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {old new srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new) :
    (warpAt cta warp pc [lane] ∗ CSL.reg cta warp lane dst old) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.reg cta warp lane dst new) :=
  wpInstr_of_spec (cvtaSpec_single_warpAt heval hcvta)

theorem wp_cvta_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .cvta dst space src } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (cvtaSpec_single_warpAt_frame heval hcvta hframe)

theorem wp_cvta_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.reg cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hcvta : Helpers.evalCvta? space srcValue = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .cvta dst space src }) frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.reg cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (cvtaSpec_single_warpAt_stableFrame heval hcvta hframe)

theorem wp_cvta_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues) :
    (warpAt cta warp pc lanes ∗ regsFor cta warp lanes dst oldValues) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          regsFor cta warp lanes dst newValues) :=
  wpInstr_of_spec (cvtaSpec_lanes_warpAt hevals)

theorem wp_cvta_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .cvta dst space src } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (regsFor cta warp lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (regsFor cta warp lanes dst newValues ∗ frame)) :=
  wpInstr_of_spec (cvtaSpec_lanes_warpAt_frame hevals hframe)

theorem wp_cvta_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : RegName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Value}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (regsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalCvtaValuesFor st cta warp space src lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .cvta dst space src }) frame) :
    (warpAt cta warp pc lanes ∗
      (regsFor cta warp lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .cvta dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (regsFor cta warp lanes dst newValues ∗ frame)) :=
  wpInstr_of_spec (cvtaSpec_lanes_warpAt_stableFrame hevals hframe)

theorem wp_isspacep_single_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId} {old new : Bool}
    {srcValue : Value}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new) :
    (warpAt cta warp pc [lane] ∗ CSL.pred cta warp lane dst old) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          CSL.pred cta warp lane dst new) :=
  wpInstr_of_spec (isspacepSpec_single_warpAt heval hisspace)

theorem wp_isspacep_single_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new : Bool} {srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .isspacep dst space src } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (isspacepSpec_single_warpAt_frame heval hisspace hframe)

theorem wp_isspacep_single_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue} {lane : LaneId}
    {old new : Bool} {srcValue : Value} {frame : CSL.Assertion}
    (heval :
      ∀ st r,
        (warpAt cta warp pc [lane] ∗
          (CSL.pred cta warp lane dst old ∗ frame)) st r →
          EvalRValue st { cta := cta, warp := warp, lane := lane } src srcValue)
    (hisspace : Helpers.evalIsspacep? space srcValue = some new)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .isspacep dst space src }) frame) :
    (warpAt cta warp pc [lane] ∗
      (CSL.pred cta warp lane dst old ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) [lane] ∗
          (CSL.pred cta warp lane dst new ∗ frame)) :=
  wpInstr_of_spec (isspacepSpec_single_warpAt_stableFrame heval hisspace hframe)

theorem wp_isspacep_lanes_warpAt
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues) :
    (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          predsFor cta warp lanes dst newValues) :=
  wpInstr_of_spec (isspacepSpec_lanes_warpAt hevals)

theorem wp_isspacep_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues)
    (hframe :
      ∀ st st' r rFrame,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
        frame st rFrame →
        Helpers.stepInstr? st cta warp { guard? := none, instr := .isspacep dst space src } =
          some st' →
        frame st' rFrame) :
    (warpAt cta warp pc lanes ∗
      (predsFor cta warp lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (predsFor cta warp lanes dst newValues ∗ frame)) :=
  wpInstr_of_spec (isspacepSpec_lanes_warpAt_frame hevals hframe)

theorem wp_isspacep_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {space : AddrSpace} {src : RValue}
    {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗
          (predsFor cta warp lanes dst oldValues ∗ frame)) st r →
          EvalIsspacepValuesFor st cta warp space src lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .isspacep dst space src }) frame) :
    (warpAt cta warp pc lanes ∗
      (predsFor cta warp lanes dst oldValues ∗ frame)) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .isspacep dst space src }
        (warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
          (predsFor cta warp lanes dst newValues ∗ frame)) :=
  wpInstr_of_spec (isspacepSpec_lanes_warpAt_stableFrame hevals hframe)

theorem barrierCTASpec_of_computed
    {cta : CTAId} {warp : WarpId} {guard? : Option Guard} {barrierId : Nat}
    {st₀ st₁ : State} {pre post : CSL.Assertion}
    (hstep :
      Helpers.stepInstr? st₀ cta warp { guard? := guard?, instr := .barrierCTA barrierId } =
        some st₁)
    (hpost : StateResourceUpdate st₀ st₁ pre post) :
    InstrSpec cta warp { guard? := guard?, instr := .barrierCTA barrierId }
      (fun st r => st = st₀ ∧ pre st r)
      (fun st r => st = st₁ ∧ post st r) :=
  InstrSpec.of_computed hstep hpost

theorem InstrSpec.mono
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {pre pre' post post' : CSL.Assertion}
    (hpre : pre' ⊢ₛ pre)
    (hspec : InstrSpec cta warp gi pre post)
    (hpost : post ⊢ₛ post') :
    InstrSpec cta warp gi pre' post' := by
  intro st r st' hpre' hstep
  rcases hspec st r st' (hpre st r hpre') hstep with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, hpost st' r' hpost'⟩

theorem InstrSpec.resourceUpdatePost
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {pre mid post : CSL.Assertion}
    (hspec : InstrSpec cta warp gi pre mid)
    (hpost : CSL.resourceUpdate mid post) :
    InstrSpec cta warp gi pre post := by
  intro st r st' hpre hstep
  rcases hspec st r st' hpre hstep with ⟨rMid, hupdate, hmid⟩
  rcases hpost st' rMid hmid with ⟨rPost, hupdatePost, hpost'⟩
  exact ⟨rPost, CSL.Resource.update_trans hupdate hupdatePost, hpost'⟩

theorem InstrSpec.frame
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {pre post frame : CSL.Assertion}
    (hspec : InstrSpec cta warp gi pre post)
    (hframe : CSL.StableUnder (InstrStep cta warp gi) frame) :
    InstrSpec cta warp gi (pre ∗ frame) (post ∗ frame) := by
  intro st r st' hsep hstep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hpre, hframeSt⟩
  rcases hspec st r₁ st' hpre hstep with ⟨r₁', hupdate, hpost⟩
  refine ⟨CSL.Resource.compose r₁' r₂, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose_right hupdate)
  · exact ⟨r₁', r₂, CSL.Resource.canCompose_update_left hupdate hcomp,
      CSL.Resource.equiv_refl _, hpost, hframe st st' r₂ hstep hframeSt⟩

theorem assignPredValueSpec_lanes_warpAt_frame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      ((warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ∗ frame)
      ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        predsFor cta warp lanes dst newValues) ∗ frame) :=
  InstrSpec.frame (assignPredValueSpec_lanes_warpAt hevals) hframe

theorem assignPredValueSpec_lanes_warpAt_stableFrame
    {cta : CTAId} {warp : WarpId} {pc : PC} {dst : PredName}
    {rhs : RValue} {lanes : List LaneId} {oldValues newValues : List Bool}
    {frame : CSL.Assertion}
    (hevals :
      ∀ st r,
        (warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) st r →
          EvalRValueBoolsFor st cta warp rhs lanes newValues)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .assignPredValue dst rhs }) frame) :
    InstrSpec cta warp { guard? := none, instr := .assignPredValue dst rhs }
      ((warpAt cta warp pc lanes ∗ predsFor cta warp lanes dst oldValues) ∗ frame)
      ((warpAt cta warp (pc.1, pc.2 + 1) lanes ∗
        predsFor cta warp lanes dst newValues) ∗ frame) :=
  assignPredValueSpec_lanes_warpAt_frame hevals hframe

theorem InstrSpec.statePropFrame
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {pre post : CSL.Assertion} {p q : State → Prop}
    (hspec : InstrSpec cta warp gi pre post)
    (hstate : ∀ st st', p st → Helpers.stepInstr? st cta warp gi = some st' → q st') :
    InstrSpec cta warp gi (stateProp p ∗ pre) (stateProp q ∗ post) := by
  intro st r st' hsep hstep
  rcases hsep with ⟨rState, rPre, hcomp, hequiv, hstatePre, hpre⟩
  have hp := stateProp_state hstatePre
  have hemp := stateProp_emp hstatePre
  subst rState
  rcases hspec st rPre st' hpre hstep with ⟨rPost, hupdate, hpost⟩
  refine ⟨CSL.Resource.compose CSL.Resource.empty rPost, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdate)
  · refine ⟨CSL.Resource.empty, ?_⟩
    refine ⟨rPost, ?_⟩
    exact ⟨by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, ⟨hstate st st' hp hstep, rfl⟩, hpost⟩

theorem barrierCTASpec_stateProp
    {cta : CTAId} {warp : WarpId} {barrierId : Nat} {p q : State → Prop}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st') :
    InstrSpec cta warp { guard? := none, instr := .barrierCTA barrierId }
      (stateProp p) (stateProp q) := by
  intro st r st' hpre hstep
  have hp := stateProp_state hpre
  have hemp := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _,
    ⟨hstate st st' hp hstep, rfl⟩⟩

theorem barrierCTASpec_stateProp_frame
    {cta : CTAId} {warp : WarpId} {barrierId : Nat} {p q : State → Prop}
    {frame : CSL.Assertion}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st')
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .barrierCTA barrierId }) frame) :
    InstrSpec cta warp { guard? := none, instr := .barrierCTA barrierId }
      (stateProp p ∗ frame) (stateProp q ∗ frame) :=
  InstrSpec.frame (barrierCTASpec_stateProp hstate) hframe

theorem wp_barrierCTA_stateProp
    {cta : CTAId} {warp : WarpId} {barrierId : Nat} {p q : State → Prop}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st') :
    stateProp p ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .barrierCTA barrierId } (stateProp q) :=
  wpInstr_of_spec (barrierCTASpec_stateProp hstate)

theorem wp_barrierCTA_stateProp_frame
    {cta : CTAId} {warp : WarpId} {barrierId : Nat} {p q : State → Prop}
    {frame : CSL.Assertion}
    (hstate :
      ∀ st st',
        p st →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          q st')
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .barrierCTA barrierId }) frame) :
    (stateProp p ∗ frame) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .barrierCTA barrierId }
        (stateProp q ∗ frame) :=
  wpInstr_of_spec (barrierCTASpec_stateProp_frame hstate hframe)

theorem wp_barrierCTA_warpAt_outcome
    {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {pc targetPc : PC} {lanes : List LaneId}
    (houtcome :
      ∀ st st',
        warpAt cta warp pc lanes st CSL.Resource.empty →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          warpAt cta warp targetPc lanes st' CSL.Resource.empty) :
    warpAt cta warp pc lanes ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .barrierCTA barrierId }
        (warpAt cta warp targetPc lanes) := by
  intro st r hpre st' hstep
  have hemp := stateProp_emp hpre
  subst r
  exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _, houtcome st st' hpre hstep⟩

theorem wp_barrierCTA_warpAt_outcome_frame
    {cta : CTAId} {warp : WarpId} {barrierId : Nat}
    {pc targetPc : PC} {lanes : List LaneId} {frame : CSL.Assertion}
    (houtcome :
      ∀ st st',
        warpAt cta warp pc lanes st CSL.Resource.empty →
          Helpers.stepInstr? st cta warp { guard? := none, instr := .barrierCTA barrierId } =
            some st' →
          warpAt cta warp targetPc lanes st' CSL.Resource.empty)
    (hframe :
      CSL.StableUnder
        (InstrStep cta warp { guard? := none, instr := .barrierCTA barrierId }) frame) :
    (warpAt cta warp pc lanes ∗ frame) ⊢ₛ
      wpInstr cta warp { guard? := none, instr := .barrierCTA barrierId }
        (warpAt cta warp targetPc lanes ∗ frame) :=
  wpInstr_of_spec <|
    InstrSpec.frame
      (gi := { guard? := none, instr := .barrierCTA barrierId })
      (by
        intro st r st' hpre hstep
        have hemp := stateProp_emp hpre
        subst r
        exact ⟨CSL.Resource.empty, CSL.Resource.update_refl _,
          houtcome st st' hpre hstep⟩)
      hframe

theorem InstrSpec.resourceUpdate
    {cta : CTAId} {warp : WarpId} {gi : GInstr}
    {pre pre' post post' : CSL.Assertion}
    (hpre : pre' ⊢ₛ pre)
    (hspec : InstrSpec cta warp gi pre post)
    (hpost : CSL.resourceUpdate post post') :
    InstrSpec cta warp gi pre' post' :=
  InstrSpec.resourceUpdatePost (InstrSpec.mono hpre hspec (CSL.entails_refl post)) hpost

theorem TerminatorSpec.mono
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {pre pre' post post' : CSL.Assertion}
    (hpre : pre' ⊢ₛ pre)
    (hspec : TerminatorSpec cta warp term pre post)
    (hpost : post ⊢ₛ post') :
    TerminatorSpec cta warp term pre' post' := by
  intro st r st' hpre' hstep
  rcases hspec st r st' (hpre st r hpre') hstep with ⟨r', hupdate, hpost'⟩
  exact ⟨r', hupdate, hpost st' r' hpost'⟩

theorem TerminatorSpec.frame
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {pre post frame : CSL.Assertion}
    (hspec : TerminatorSpec cta warp term pre post)
    (hframe : CSL.StableUnder (TerminatorStep cta warp term) frame) :
    TerminatorSpec cta warp term (pre ∗ frame) (post ∗ frame) := by
  intro st r st' hsep hstep
  rcases hsep with ⟨r₁, r₂, hcomp, hequiv, hpre, hframeSt⟩
  rcases hspec st r₁ st' hpre hstep with ⟨r₁', hupdate, hpost⟩
  refine ⟨CSL.Resource.compose r₁' r₂, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose_right hupdate)
  · exact ⟨r₁', r₂, CSL.Resource.canCompose_update_left hupdate hcomp,
      CSL.Resource.equiv_refl _, hpost, hframe st st' r₂ hstep hframeSt⟩

theorem TerminatorSpec.statePropFrame
    {cta : CTAId} {warp : WarpId} {term : Terminator}
    {pre post : CSL.Assertion} {p q : State → Prop}
    (hspec : TerminatorSpec cta warp term pre post)
    (hstate : ∀ st st', p st → Helpers.stepTerminator? st cta warp term = some st' → q st') :
    TerminatorSpec cta warp term (stateProp p ∗ pre) (stateProp q ∗ post) := by
  intro st r st' hsep hstep
  rcases hsep with ⟨rState, rPre, hcomp, hequiv, hstatePre, hpre⟩
  have hp := stateProp_state hstatePre
  have hemp := stateProp_emp hstatePre
  subst rState
  rcases hspec st rPre st' hpre hstep with ⟨rPost, hupdate, hpost⟩
  refine ⟨CSL.Resource.compose CSL.Resource.empty rPost, ?_, ?_⟩
  · exact CSL.Resource.update_trans (CSL.Resource.update_of_equiv hequiv)
      (CSL.Resource.update_compose (CSL.Resource.update_refl _) hupdate)
  · exact ⟨CSL.Resource.empty, rPost,
      by simp [CSL.Resource.canCompose, CSL.Resource.empty],
      CSL.Resource.equiv_refl _, ⟨hstate st st' hp hstep, rfl⟩, hpost⟩

inductive InstrSpecs (cta : CTAId) (warp : WarpId) :
    List GInstr → CSL.Assertion → CSL.Assertion → Prop where
  | nil {post : CSL.Assertion} :
      InstrSpecs cta warp [] post post
  | cons {gi : GInstr} {rest : List GInstr} {pre mid post : CSL.Assertion} :
      InstrSpec cta warp gi pre mid →
      InstrSpecs cta warp rest mid post →
      InstrSpecs cta warp (gi :: rest) pre post

theorem wpInstrList_of_specs
    {cta : CTAId} {warp : WarpId} {body : List GInstr} {pre post : CSL.Assertion}
    (hspecs : InstrSpecs cta warp body pre post) :
    pre ⊢ₛ wpInstrList cta warp body post := by
  induction hspecs with
  | nil =>
      exact CSL.entails_refl _
  | cons hhead _ ih =>
      exact CSL.entails_trans (wpInstr_of_spec hhead) (wpInstr_mono ih)

theorem wpInstrs_of_specs
    {cta : CTAId} {warp : WarpId} {body : Array GInstr} {pre post : CSL.Assertion}
    (hspecs : InstrSpecs cta warp body.toList pre post) :
    pre ⊢ₛ wpInstrs cta warp body post :=
  wpInstrList_of_specs hspecs

def ConcreteBlockSpec
    (cta : CTAId) (warp : WarpId) (block : Block) (pre post : CSL.Assertion) : Prop :=
  ∃ mid,
    InstrSpecs cta warp block.body.toList pre mid ∧
    TerminatorSpec cta warp block.term mid post

theorem wpConcreteBlock_of_spec
    {cta : CTAId} {warp : WarpId} {block : Block} {pre post : CSL.Assertion}
    (hspec : ConcreteBlockSpec cta warp block pre post) :
    pre ⊢ₛ wpConcreteBlock cta warp block post := by
  rcases hspec with ⟨mid, hbody, hterm⟩
  exact CSL.entails_trans (wpInstrs_of_specs hbody)
    (wpInstrs_mono (wpTerminator_of_spec hterm))

theorem blockVC.of_concreteBlockSpec
    {cta : CTAId} {warp : WarpId} {invariants : InvariantMap}
    {post : CSL.Assertion} {label : BlockLabel} {block : Block}
    (hspec :
      ConcreteBlockSpec cta warp block (invariants label)
        (blockTermPost invariants post block.term)) :
    blockVC cta warp invariants post label block :=
  blockVC.of_wpConcreteBlock (wpConcreteBlock_of_spec hspec)

theorem blockVCChoice.of_concreteBlockSpec
    {cta : CTAId} {warp : WarpId} {choices : CbrChoiceMap}
    {invariants : InvariantMap} {post : CSL.Assertion} {label : BlockLabel}
    {block : Block}
    (hspec :
      ConcreteBlockSpec cta warp block (invariants label)
        (blockTermPostChoice choices invariants post label block.term)) :
    blockVCChoice cta warp choices invariants post label block :=
  blockVCChoice.of_wpConcreteBlock (wpConcreteBlock_of_spec hspec)

end WP
end CLean
