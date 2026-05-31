import CLean.Examples.Matmul
import CLean.Examples.Loop

namespace CLean
namespace Examples

open CSL WP

def matmulLoopProofN : Nat :=
  3

def matmulLoopBStride : Nat :=
  4 * matmulLoopProofN

def matmulLoopTermS32 (params : MatmulCellParams) (a b : Nat → Nat → Int) (k : Nat) : Int :=
  Helpers.normalizeSigned 32 (a params.row k * b k params.col)

def matmulLoopPrefixS32 (params : MatmulCellParams) (a b : Nat → Nat → Int) : Nat → Int
  | 0 => 0
  | k + 1 =>
      Helpers.normalizeSigned 32
        (matmulLoopPrefixS32 params a b k + matmulLoopTermS32 params a b k)

def matmulLoopSetK : GInstr :=
  { guard? := none, instr := .assignReg "k" (.imm (.s32 0)) }

def matmulLoopSetAcc : GInstr :=
  { guard? := none, instr := .assignReg "acc" (.imm (.s32 0)) }

def matmulLoopSetAPtr (params : MatmulCellParams) : GInstr :=
  { guard? := none,
    instr := .assignReg "aPtr" (.imm (.gaddr .global (matmulAOffset params 0))) }

def matmulLoopSetBPtr (params : MatmulCellParams) : GInstr :=
  { guard? := none,
    instr := .assignReg "bPtr" (.imm (.gaddr .global (matmulBOffset params 0))) }

def matmulLoopSetPred : GInstr :=
  { guard? := none,
    instr :=
      .assignPred "p"
        { op := .lt, lhs := .reg "k", rhs := .imm (.s32 (Int.ofNat matmulLoopProofN)) } }

def matmulLoopLoadA : GInstr :=
  { guard? := none,
    instr := .load "aVal" { space := .global, ty := .s32, addr := .reg "aPtr" } }

def matmulLoopLoadB : GInstr :=
  { guard? := none,
    instr := .load "bVal" { space := .global, ty := .s32, addr := .reg "bPtr" } }

def matmulLoopMul : GInstr :=
  { guard? := none,
    instr := .assignReg "prod" (.binop .mul (.reg "aVal") (.reg "bVal")) }

def matmulLoopAddAcc : GInstr :=
  { guard? := none,
    instr := .assignReg "acc" (.binop .add (.reg "acc") (.reg "prod")) }

def matmulLoopIncK : GInstr :=
  { guard? := none,
    instr := .assignReg "k" (.binop .add (.reg "k") (.imm (.s32 1))) }

def matmulLoopIncAPtr : GInstr :=
  { guard? := none,
    instr := .assignReg "aPtr" (.binop .add (.reg "aPtr") (.imm (.u64 (4 : UInt64)))) }

def matmulLoopIncBPtr : GInstr :=
  { guard? := none,
    instr :=
      .assignReg "bPtr"
        (.binop .add (.reg "bPtr") (.imm (.u64 (UInt64.ofNat matmulLoopBStride)))) }

def matmulLoopStore (params : MatmulCellParams) : GInstr :=
  { guard? := none,
    instr :=
      .store { space := .global, ty := .s32, addr := matmulGlobalAddr (matmulCellOffset params) }
        (.reg "acc") }

def matmulLoopEntryBlock (params : MatmulCellParams) : Block :=
  { label := "entry"
    body := #[
      matmulLoopSetK,
      matmulLoopSetAcc,
      matmulLoopSetAPtr params,
      matmulLoopSetBPtr params]
    term := .br "loop" }

def matmulLoopHeaderBlock : Block :=
  { label := "loop"
    body := #[matmulLoopSetPred]
    term := .cbr (.pred "p") "body" "exit" }

def matmulLoopBodyBlock : Block :=
  { label := "body"
    body := #[
      matmulLoopLoadA,
      matmulLoopLoadB,
      matmulLoopMul,
      matmulLoopAddAcc,
      matmulLoopIncK,
      matmulLoopIncAPtr,
      matmulLoopIncBPtr]
    term := .br "loop" }

def matmulLoopExitBlock (params : MatmulCellParams) : Block :=
  { label := "exit"
    body := #[matmulLoopStore params]
    term := .terminate }

def matmulLoopEnv (params : MatmulCellParams) : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := 1 } }
    blocks :=
      ((({} : Std.HashMap BlockLabel Block).insert "entry" (matmulLoopEntryBlock params)).insert
        "loop" matmulLoopHeaderBlock).insert "body" matmulLoopBodyBlock |>.insert
        "exit" (matmulLoopExitBlock params) }

def matmulLoopAResources (params : MatmulCellParams) (aBytes : Nat → List Byte) :
    CSL.Assertion :=
  CSL.sepList
    ((List.range matmulLoopProofN).map fun k =>
      CSL.globalBytes (matmulAOffset params k) .read (aBytes k))

def matmulLoopBResources (params : MatmulCellParams) (bBytes : Nat → List Byte) :
    CSL.Assertion :=
  CSL.sepList
    ((List.range matmulLoopProofN).map fun k =>
      CSL.globalBytes (matmulBOffset params k) .read (bBytes k))

def matmulLoopFrame
    (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.Assertion :=
  matmulLoopAResources params aBytes ∗
    (matmulLoopBResources params bBytes ∗
      CSL.globalBytes (matmulCellOffset params) .write outBytes)

def matmulLoopRegs
    (lane : LaneId) (k acc : Int) (aVal bVal prod aPtr bPtr : Value) (p : Bool) :
    CSL.Assertion :=
  CSL.sepList [
    CSL.reg 0 0 lane "k" (.s32 k),
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.reg 0 0 lane "aVal" aVal,
    CSL.reg 0 0 lane "bVal" bVal,
    CSL.reg 0 0 lane "prod" prod,
    CSL.reg 0 0 lane "aPtr" aPtr,
    CSL.reg 0 0 lane "bPtr" bPtr,
    CSL.pred 0 0 lane "p" p]

def matmulLoopAt
    (lane : LaneId) (pc : PC) (k acc : Int) (aVal bVal prod aPtr bPtr : Value)
    (p : Bool) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) : CSL.Assertion :=
  warpAt 0 0 pc [lane] ∗
    (matmulLoopRegs lane k acc aVal bVal prod aPtr bPtr p ∗
      matmulLoopFrame params aBytes bBytes outBytes)

def matmulLoopFlatResources
    (lane : LaneId) (k acc : Int) (aVal bVal prod aPtr bPtr : Value) (p : Bool)
    (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) : CSL.Assertion :=
  CSL.sepList [
    CSL.reg 0 0 lane "k" (.s32 k),
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.reg 0 0 lane "aVal" aVal,
    CSL.reg 0 0 lane "bVal" bVal,
    CSL.reg 0 0 lane "prod" prod,
    CSL.reg 0 0 lane "aPtr" aPtr,
    CSL.reg 0 0 lane "bPtr" bPtr,
    CSL.pred 0 0 lane "p" p,
    CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0),
    CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
    CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2),
    CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0),
    CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
    CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2),
    CSL.globalBytes (matmulCellOffset params) .write outBytes]

def matmulLoopEntryPre
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value)
    (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 0) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "k" oldK,
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" oldAPtr,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut)

def matmulLoopEntryAfterK
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value)
    (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 1) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" oldAPtr,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut)

def matmulLoopEntryAfterAcc
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldA oldB oldProd oldAPtr oldBPtr : Value)
    (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 2) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" oldAPtr,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut)

def matmulLoopEntryAfterAPtr
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldA oldB oldProd oldBPtr : Value)
    (oldP : Bool) : CSL.Assertion :=
  warpAt 0 0 ("entry", 3) [lane] ∗
    (CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut)

def matmulLoopEntryTermPre
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldA oldB oldProd : Value) (oldP : Bool) : CSL.Assertion :=
  matmulLoopAt lane ("entry", 4) 0 0 oldA oldB oldProd
    (.gaddr .global (matmulAOffset params 0)) (.gaddr .global (matmulBOffset params 0))
    oldP params aBytes bBytes oldOut

def matmulLoopHeaderTruePre
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ k aVal bVal prod,
      k < matmulLoopProofN ∧
        matmulLoopAt lane ("loop", 1) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut st r

def matmulLoopHeaderFalsePre
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ aVal bVal prod,
      matmulLoopAt lane ("loop", 1) (Int.ofNat matmulLoopProofN)
        (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
        (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes oldOut st r

def matmulLoopHeaderTermPre
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  matmulLoopHeaderTruePre lane params a b aBytes bBytes oldOut ∨ₛ
    matmulLoopHeaderFalsePre lane params a b aBytes bBytes oldOut

def matmulLoopLoopInv
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ k aVal bVal prod p,
      k ≤ matmulLoopProofN ∧
        matmulLoopAt lane ("loop", 0) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) p params aBytes bBytes oldOut st r

def matmulLoopBodyInv
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ k aVal bVal prod,
      k < matmulLoopProofN ∧
        matmulLoopAt lane ("body", 0) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut st r

def matmulLoopExitInv
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ aVal bVal prod,
      matmulLoopAt lane ("exit", 0) (Int.ofNat matmulLoopProofN)
        (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
        (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes oldOut st r

def matmulLoopBodyTermPre
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ k aVal bVal prod,
      k ≤ matmulLoopProofN ∧
        matmulLoopAt lane ("body", 7) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut st r

def matmulLoopExitTermPre
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (newOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ aVal bVal prod,
      matmulLoopAt lane ("exit", 1) (Int.ofNat matmulLoopProofN)
        (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
        (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes newOut st r

def matmulLoopPost
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (newOut : List Byte) : CSL.Assertion :=
  fun st r =>
    ∃ aVal bVal prod,
      (laneTerminatedAt 0 0 lane ("exit", 1) ∗
        (matmulLoopRegs lane (Int.ofNat matmulLoopProofN)
          (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
          (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false ∗
          matmulLoopFrame params aBytes bBytes newOut)) st r

def matmulLoopInvariants
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut _newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) : InvariantMap
  | "entry" =>
      matmulLoopEntryPre lane params aBytes bBytes oldOut oldK oldAcc oldA oldB oldProd
        oldAPtr oldBPtr oldP
  | "loop" => matmulLoopLoopInv lane params a b aBytes bBytes oldOut
  | "body" => matmulLoopBodyInv lane params a b aBytes bBytes oldOut
  | "exit" => matmulLoopExitInv lane params a b aBytes bBytes oldOut
  | _ => CSL.pure False

def matmulLoopKernelSpec
    (init : State) (resource : CSL.Resource) (lane : LaneId) (params : MatmulCellParams)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) : KernelSpec :=
  { init := init
    resource := resource
    pre :=
      fun st r =>
        params.n = matmulLoopProofN ∧
          st.kernelEnv = matmulLoopEnv params ∧
            matmulLoopEntryPre lane params aBytes bBytes oldOut oldK oldAcc oldA oldB
              oldProd oldAPtr oldBPtr oldP st r
    invariant :=
      cfgKernelInvariant' (matmulLoopEnv params) 0 0
        (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
          oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
        (matmulLoopPost lane params a b aBytes bBytes newOut)
    post := matmulLoopPost lane params a b aBytes bBytes newOut }

theorem matmulLoopAResources_stable
    {step : State → State → Prop} {params : MatmulCellParams} {aBytes : Nat → List Byte}
    (hbytes :
      ∀ k, CSL.StableUnder step
        (CSL.globalBytes (matmulAOffset params k) .read (aBytes k))) :
    CSL.StableUnder step (matmulLoopAResources params aBytes) := by
  unfold matmulLoopAResources
  apply CSL.stable_sepList
  intro p hp
  simp at hp
  rcases hp with ⟨k, _hk, hp⟩
  subst p
  exact hbytes k

theorem matmulLoopBResources_stable
    {step : State → State → Prop} {params : MatmulCellParams} {bBytes : Nat → List Byte}
    (hbytes :
      ∀ k, CSL.StableUnder step
        (CSL.globalBytes (matmulBOffset params k) .read (bBytes k))) :
    CSL.StableUnder step (matmulLoopBResources params bBytes) := by
  unfold matmulLoopBResources
  apply CSL.stable_sepList
  intro p hp
  simp at hp
  rcases hp with ⟨k, _hk, hp⟩
  subst p
  exact hbytes k

theorem matmulLoopFrame_stable
    {step : State → State → Prop} {params : MatmulCellParams}
    {aBytes bBytes : Nat → List Byte} {outBytes : List Byte}
    (ha :
      ∀ k, CSL.StableUnder step
        (CSL.globalBytes (matmulAOffset params k) .read (aBytes k)))
    (hb :
      ∀ k, CSL.StableUnder step
        (CSL.globalBytes (matmulBOffset params k) .read (bBytes k)))
    (hout :
      CSL.StableUnder step
        (CSL.globalBytes (matmulCellOffset params) .write outBytes)) :
    CSL.StableUnder step (matmulLoopFrame params aBytes bBytes outBytes) := by
  unfold matmulLoopFrame
  exact CSL.stable_sep (matmulLoopAResources_stable ha)
    (CSL.stable_sep (matmulLoopBResources_stable hb) hout)

theorem matmulLoopRegs_stable_terminator
    (term : Terminator) (lane : LaneId) (k acc : Int)
    (aVal bVal prod aPtr bPtr : Value) (p : Bool) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (matmulLoopRegs lane k acc aVal bVal prod aPtr bPtr p) := by
  unfold matmulLoopRegs
  apply CSL.stable_sepList
  intro q hq
  simp at hq
  rcases hq with hq | hq | hq | hq | hq | hq | hq | hq
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_reg_terminator
  · subst q
    exact stable_pred_terminator

theorem matmulLoopFrame_stable_terminator
    (term : Terminator) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (matmulLoopFrame params aBytes bBytes outBytes) :=
  matmulLoopFrame_stable
    (fun _ => stable_globalBytes_terminator)
    (fun _ => stable_globalBytes_terminator)
    stable_globalBytes_terminator

theorem matmulLoopAt_stable_terminator_frame
    (term : Terminator) (lane : LaneId) (k acc : Int)
    (aVal bVal prod aPtr bPtr : Value) (p : Bool)
    (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    CSL.StableUnder (TerminatorStep 0 0 term)
      (matmulLoopRegs lane k acc aVal bVal prod aPtr bPtr p ∗
        matmulLoopFrame params aBytes bBytes outBytes) :=
  CSL.stable_sep
    (matmulLoopRegs_stable_terminator term lane k acc aVal bVal prod aPtr bPtr p)
    (matmulLoopFrame_stable_terminator term params aBytes bBytes outBytes)

theorem matmulLoopFrame_stable_assignReg
    (dst : RegName) (rhs : RValue) (params : MatmulCellParams)
    (aBytes bBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignReg dst rhs })
      (matmulLoopFrame params aBytes bBytes outBytes) :=
  matmulLoopFrame_stable
    (fun _ => stable_globalBytes_assignReg)
    (fun _ => stable_globalBytes_assignReg)
    stable_globalBytes_assignReg

theorem matmulLoopFrame_stable_assignPred
    (dst : PredName) (cmp : CmpExpr) (params : MatmulCellParams)
    (aBytes bBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .assignPred dst cmp })
      (matmulLoopFrame params aBytes bBytes outBytes) :=
  matmulLoopFrame_stable
    (fun _ => stable_globalBytes_assignPred)
    (fun _ => stable_globalBytes_assignPred)
    stable_globalBytes_assignPred

theorem matmulLoopFrame_stable_load
    (dst : RegName) (addr : TypedAddr) (params : MatmulCellParams)
    (aBytes bBytes : Nat → List Byte) (outBytes : List Byte) :
    CSL.StableUnder
      (InstrStep 0 0 { guard? := none, instr := .load dst addr })
      (matmulLoopFrame params aBytes bBytes outBytes) :=
  matmulLoopFrame_stable
    (fun _ => stable_globalBytes_load)
    (fun _ => stable_globalBytes_load)
    stable_globalBytes_load

theorem matmulLoopAt_to_flat
    (lane : LaneId) (pc : PC) (k acc : Int) (aVal bVal prod aPtr bPtr : Value)
    (p : Bool) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    matmulLoopAt lane pc k acc aVal bVal prod aPtr bPtr p params aBytes bBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        matmulLoopFlatResources lane k acc aVal bVal prod aPtr bPtr p params aBytes
          bBytes outBytes) := by
  simpa [matmulLoopAt, matmulLoopRegs, matmulLoopFrame, matmulLoopAResources,
    matmulLoopBResources, matmulLoopFlatResources, matmulLoopProofN] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.entails_trans
        (CSL.sep_mono
          (CSL.entails_refl
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 k),
              CSL.reg 0 0 lane "acc" (.s32 acc),
              CSL.reg 0 0 lane "aVal" aVal,
              CSL.reg 0 0 lane "bVal" bVal,
              CSL.reg 0 0 lane "prod" prod,
              CSL.reg 0 0 lane "aPtr" aPtr,
              CSL.reg 0 0 lane "bPtr" bPtr,
              CSL.pred 0 0 lane "p" p]))
          (CSL.entails_trans
            (CSL.sep_mono
              (CSL.entails_refl
                (CSL.sepList [
                  CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0),
                  CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
                  CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)]))
              (CSL.sepList_append_cons
                (CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0))
                [CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
                  CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)]
                (CSL.globalBytes (matmulCellOffset params) .write outBytes) []))
            (CSL.sepList_append_cons
              (CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0))
              [CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
                CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)]
              (CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0))
              [CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
                CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2),
                CSL.globalBytes (matmulCellOffset params) .write outBytes])))
        (CSL.sepList_append_cons
          (CSL.reg 0 0 lane "k" (.s32 k))
          [CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" aPtr,
            CSL.reg 0 0 lane "bPtr" bPtr,
            CSL.pred 0 0 lane "p" p]
          (CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0))
          [CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
            CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2),
            CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0),
            CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
            CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2),
            CSL.globalBytes (matmulCellOffset params) .write outBytes]))

theorem matmulLoopAt_flat_to_standard
    (lane : LaneId) (pc : PC) (k acc : Int) (aVal bVal prod aPtr bPtr : Value)
    (p : Bool) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      matmulLoopFlatResources lane k acc aVal bVal prod aPtr bPtr p params aBytes
        bBytes outBytes) ⊢ₛ
      matmulLoopAt lane pc k acc aVal bVal prod aPtr bPtr p params aBytes bBytes
        outBytes := by
  simpa [matmulLoopAt, matmulLoopRegs, matmulLoopFrame, matmulLoopAResources,
    matmulLoopBResources, matmulLoopFlatResources, matmulLoopProofN] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.entails_trans
        (CSL.sepList_append_cons_rev
          (CSL.reg 0 0 lane "k" (.s32 k))
          [CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" aPtr,
            CSL.reg 0 0 lane "bPtr" bPtr,
            CSL.pred 0 0 lane "p" p]
          (CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0))
          [CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
            CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2),
            CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0),
            CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
            CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2),
            CSL.globalBytes (matmulCellOffset params) .write outBytes])
        (CSL.sep_mono
          (CSL.entails_refl
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 k),
              CSL.reg 0 0 lane "acc" (.s32 acc),
              CSL.reg 0 0 lane "aVal" aVal,
              CSL.reg 0 0 lane "bVal" bVal,
              CSL.reg 0 0 lane "prod" prod,
              CSL.reg 0 0 lane "aPtr" aPtr,
              CSL.reg 0 0 lane "bPtr" bPtr,
              CSL.pred 0 0 lane "p" p]))
          (CSL.entails_trans
            (CSL.sepList_append_cons_rev
              (CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0))
              [CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
                CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)]
              (CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0))
              [CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
                CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2),
                CSL.globalBytes (matmulCellOffset params) .write outBytes])
            (CSL.sep_mono
              (CSL.entails_refl
                (CSL.sepList [
                  CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0),
                  CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1),
                  CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)]))
              (CSL.sepList_append_cons_rev
                (CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0))
                [CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1),
                  CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)]
                (CSL.globalBytes (matmulCellOffset params) .write outBytes) [])))))

private theorem perm_matmul_regs_pred_front
    (a b c d e f g h : CSL.Assertion) :
    [a, b, c, d, e, f, g, h].Perm [h, a, b, c, d, e, f, g] := by
  simpa using (List.perm_middle (a := h) (l₁ := [a, b, c, d, e, f, g]) (l₂ := []))

private theorem perm_abcdefgh_to_bacdefgh
    (a b c d e f g h : CSL.Assertion) :
    [a, b, c, d, e, f, g, h].Perm [b, a, c, d, e, f, g, h] :=
  List.Perm.swap b a [c, d, e, f, g, h]

private theorem perm_abcdefgh_to_fabcdegh
    (a b c d e f g h : CSL.Assertion) :
    [a, b, c, d, e, f, g, h].Perm [f, a, b, c, d, e, g, h] := by
  simpa using (List.perm_middle (a := f) (l₁ := [a, b, c, d, e]) (l₂ := [g, h]))

private theorem perm_abcdefgh_to_eabcdfgh
    (a b c d e f g h : CSL.Assertion) :
    [a, b, c, d, e, f, g, h].Perm [e, a, b, c, d, f, g, h] := by
  simpa using (List.perm_middle (a := e) (l₁ := [a, b, c, d]) (l₂ := [f, g, h]))

private theorem perm_abcdefgh_to_gabcdefh
    (a b c d e f g h : CSL.Assertion) :
    [a, b, c, d, e, f, g, h].Perm [g, a, b, c, d, e, f, h] := by
  simpa using (List.perm_middle (a := g) (l₁ := [a, b, c, d, e, f]) (l₂ := [h]))

private theorem perm_matmul_flat_to_a0_aVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [i, c, a, b, d, e, f, g, h, j, k, l, m, n, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [i, a, b, c, d, e, f, g, h, j, k, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := i) (l₁ := [a, b, c, d, e, f, g, h])
        (l₂ := [j, k, l, m, n, o]))
  have h2 :
      [a, b, c, d, e, f, g, h, j, k, l, m, n, o].Perm
        [c, a, b, d, e, f, g, h, j, k, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, g, h, j, k, l, m, n, o]))
  exact h1.trans (List.Perm.cons i h2)

private theorem perm_matmul_flat_to_a1_aVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [j, c, a, b, d, e, f, g, h, i, k, l, m, n, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [j, a, b, c, d, e, f, g, h, i, k, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := j) (l₁ := [a, b, c, d, e, f, g, h, i])
        (l₂ := [k, l, m, n, o]))
  have h2 :
      [a, b, c, d, e, f, g, h, i, k, l, m, n, o].Perm
        [c, a, b, d, e, f, g, h, i, k, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, g, h, i, k, l, m, n, o]))
  exact h1.trans (List.Perm.cons j h2)

private theorem perm_matmul_flat_to_a2_aVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [k, c, a, b, d, e, f, g, h, i, j, l, m, n, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [k, a, b, c, d, e, f, g, h, i, j, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := k) (l₁ := [a, b, c, d, e, f, g, h, i, j])
        (l₂ := [l, m, n, o]))
  have h2 :
      [a, b, c, d, e, f, g, h, i, j, l, m, n, o].Perm
        [c, a, b, d, e, f, g, h, i, j, l, m, n, o] := by
    simpa using
      (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, g, h, i, j, l, m, n, o]))
  exact h1.trans (List.Perm.cons k h2)

private theorem perm_matmul_flat_to_b0_bVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [l, d, a, b, c, e, f, g, h, i, j, k, m, n, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [l, a, b, c, d, e, f, g, h, i, j, k, m, n, o] := by
    simpa using
      (List.perm_middle (a := l) (l₁ := [a, b, c, d, e, f, g, h, i, j, k])
        (l₂ := [m, n, o]))
  have h2 :
      [a, b, c, d, e, f, g, h, i, j, k, m, n, o].Perm
        [d, a, b, c, e, f, g, h, i, j, k, m, n, o] := by
    simpa using
      (List.perm_middle (a := d) (l₁ := [a, b, c]) (l₂ := [e, f, g, h, i, j, k, m, n, o]))
  exact h1.trans (List.Perm.cons l h2)

private theorem perm_matmul_flat_to_b1_bVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [m, d, a, b, c, e, f, g, h, i, j, k, l, n, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [m, a, b, c, d, e, f, g, h, i, j, k, l, n, o] := by
    simpa using
      (List.perm_middle (a := m) (l₁ := [a, b, c, d, e, f, g, h, i, j, k, l])
        (l₂ := [n, o]))
  have h2 :
      [a, b, c, d, e, f, g, h, i, j, k, l, n, o].Perm
        [d, a, b, c, e, f, g, h, i, j, k, l, n, o] := by
    simpa using
      (List.perm_middle (a := d) (l₁ := [a, b, c]) (l₂ := [e, f, g, h, i, j, k, l, n, o]))
  exact h1.trans (List.Perm.cons m h2)

private theorem perm_matmul_flat_to_b2_bVal
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [n, d, a, b, c, e, f, g, h, i, j, k, l, m, o] := by
  have h1 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
        [n, a, b, c, d, e, f, g, h, i, j, k, l, m, o] := by
    simpa using
      (List.perm_middle (a := n) (l₁ := [a, b, c, d, e, f, g, h, i, j, k, l, m])
        (l₂ := [o]))
  have h2 :
      [a, b, c, d, e, f, g, h, i, j, k, l, m, o].Perm
        [d, a, b, c, e, f, g, h, i, j, k, l, m, o] := by
    simpa using
      (List.perm_middle (a := d) (l₁ := [a, b, c]) (l₂ := [e, f, g, h, i, j, k, l, m, o]))
  exact h1.trans (List.Perm.cons n h2)

private theorem perm_matmul_flat_out_front
    (a b c d e f g h i j k l m n o : CSL.Assertion) :
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o].Perm
      [o, a, b, c, d, e, f, g, h, i, j, k, l, m, n] := by
  simpa using
    (List.perm_middle (a := o) (l₁ := [a, b, c, d, e, f, g, h, i, j, k, l, m, n])
      (l₂ := []))

theorem matmulLoopAt_to_pred_focus
    (lane : LaneId) (pc : PC) (k acc : Int) (aVal bVal prod aPtr bPtr : Value)
    (p : Bool) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    matmulLoopAt lane pc k acc aVal bVal prod aPtr bPtr p params aBytes bBytes outBytes ⊢ₛ
      (warpAt 0 0 pc [lane] ∗
        (CSL.pred 0 0 lane "p" p ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 k),
            CSL.reg 0 0 lane "acc" (.s32 acc),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" aPtr,
            CSL.reg 0 0 lane "bPtr" bPtr] ∗
            matmulLoopFrame params aBytes bBytes outBytes))) := by
  simpa [matmulLoopAt, matmulLoopRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sepList_perm_frame_to_cons
        [CSL.reg 0 0 lane "k" (.s32 k),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "aVal" aVal,
          CSL.reg 0 0 lane "bVal" bVal,
          CSL.reg 0 0 lane "prod" prod,
          CSL.reg 0 0 lane "aPtr" aPtr,
          CSL.reg 0 0 lane "bPtr" bPtr,
          CSL.pred 0 0 lane "p" p]
        (CSL.pred 0 0 lane "p" p)
        (CSL.reg 0 0 lane "k" (.s32 k))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "aVal" aVal,
          CSL.reg 0 0 lane "bVal" bVal,
          CSL.reg 0 0 lane "prod" prod,
          CSL.reg 0 0 lane "aPtr" aPtr,
          CSL.reg 0 0 lane "bPtr" bPtr]
        (matmulLoopFrame params aBytes bBytes outBytes)
        (perm_matmul_regs_pred_front
          (CSL.reg 0 0 lane "k" (.s32 k))
          (CSL.reg 0 0 lane "acc" (.s32 acc))
          (CSL.reg 0 0 lane "aVal" aVal)
          (CSL.reg 0 0 lane "bVal" bVal)
          (CSL.reg 0 0 lane "prod" prod)
          (CSL.reg 0 0 lane "aPtr" aPtr)
          (CSL.reg 0 0 lane "bPtr" bPtr)
          (CSL.pred 0 0 lane "p" p)))

theorem matmulLoopAt_pred_focus_to_standard
    (lane : LaneId) (pc : PC) (k acc : Int) (aVal bVal prod aPtr bPtr : Value)
    (p : Bool) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (outBytes : List Byte) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.pred 0 0 lane "p" p ∗
        (CSL.sepList [
          CSL.reg 0 0 lane "k" (.s32 k),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "aVal" aVal,
          CSL.reg 0 0 lane "bVal" bVal,
          CSL.reg 0 0 lane "prod" prod,
          CSL.reg 0 0 lane "aPtr" aPtr,
          CSL.reg 0 0 lane "bPtr" bPtr] ∗
          matmulLoopFrame params aBytes bBytes outBytes))) ⊢ₛ
      matmulLoopAt lane pc k acc aVal bVal prod aPtr bPtr p params aBytes bBytes
        outBytes := by
  simpa [matmulLoopAt, matmulLoopRegs] using
    CSL.sep_mono (CSL.entails_refl (warpAt 0 0 pc [lane]))
      (CSL.sep_cons_frame_to_sepList_perm
        (CSL.pred 0 0 lane "p" p)
        (CSL.reg 0 0 lane "k" (.s32 k))
        [CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "aVal" aVal,
          CSL.reg 0 0 lane "bVal" bVal,
          CSL.reg 0 0 lane "prod" prod,
          CSL.reg 0 0 lane "aPtr" aPtr,
          CSL.reg 0 0 lane "bPtr" bPtr]
        [CSL.reg 0 0 lane "k" (.s32 k),
          CSL.reg 0 0 lane "acc" (.s32 acc),
          CSL.reg 0 0 lane "aVal" aVal,
          CSL.reg 0 0 lane "bVal" bVal,
          CSL.reg 0 0 lane "prod" prod,
          CSL.reg 0 0 lane "aPtr" aPtr,
          CSL.reg 0 0 lane "bPtr" bPtr,
          CSL.pred 0 0 lane "p" p]
        (matmulLoopFrame params aBytes bBytes outBytes)
        (perm_matmul_regs_pred_front
          (CSL.reg 0 0 lane "k" (.s32 k))
          (CSL.reg 0 0 lane "acc" (.s32 acc))
          (CSL.reg 0 0 lane "aVal" aVal)
          (CSL.reg 0 0 lane "bVal" bVal)
          (CSL.reg 0 0 lane "prod" prod)
          (CSL.reg 0 0 lane "aPtr" aPtr)
          (CSL.reg 0 0 lane "bPtr" bPtr)
          (CSL.pred 0 0 lane "p" p)).symm)

theorem matmulLoop_entry_lookup (params : MatmulCellParams) :
    (matmulLoopEnv params).blocks["entry"]? = some (matmulLoopEntryBlock params) := by
  simp [matmulLoopEnv, Std.HashMap.getElem?_insert]

theorem matmulLoop_loop_lookup (params : MatmulCellParams) :
    (matmulLoopEnv params).blocks["loop"]? = some matmulLoopHeaderBlock := by
  simp [matmulLoopEnv, Std.HashMap.getElem?_insert]

theorem matmulLoop_body_lookup (params : MatmulCellParams) :
    (matmulLoopEnv params).blocks["body"]? = some matmulLoopBodyBlock := by
  simp [matmulLoopEnv, Std.HashMap.getElem?_insert]

theorem matmulLoop_exit_lookup (params : MatmulCellParams) :
    (matmulLoopEnv params).blocks["exit"]? = some (matmulLoopExitBlock params) := by
  simp [matmulLoopEnv, Std.HashMap.getElem?_insert]

theorem matmulLoop_block_lookup
    {params : MatmulCellParams} {label : BlockLabel} {block : Block}
    (hlookup : (matmulLoopEnv params).blocks[label]? = some block) :
    (label = "entry" ∧ block = matmulLoopEntryBlock params) ∨
      (label = "loop" ∧ block = matmulLoopHeaderBlock) ∨
        (label = "body" ∧ block = matmulLoopBodyBlock) ∨
          (label = "exit" ∧ block = matmulLoopExitBlock params) := by
  by_cases hentry : label = "entry"
  · subst label
    simp [matmulLoopEnv, Std.HashMap.getElem?_insert] at hlookup
    exact Or.inl ⟨rfl, hlookup.symm⟩
  · by_cases hloop : label = "loop"
    · subst label
      simp [matmulLoopEnv, Std.HashMap.getElem?_insert] at hlookup
      exact Or.inr (Or.inl ⟨rfl, hlookup.symm⟩)
    · by_cases hbody : label = "body"
      · subst label
        simp [matmulLoopEnv, Std.HashMap.getElem?_insert] at hlookup
        exact Or.inr (Or.inr (Or.inl ⟨rfl, hlookup.symm⟩))
      · by_cases hexit : label = "exit"
        · subst label
          simp [matmulLoopEnv, Std.HashMap.getElem?_insert] at hlookup
          exact Or.inr (Or.inr (Or.inr ⟨rfl, hlookup.symm⟩))
        · have hentryBeq : ("entry" == label) = false :=
            (beq_eq_false_iff_ne).2 (fun h => hentry h.symm)
          have hloopBeq : ("loop" == label) = false :=
            (beq_eq_false_iff_ne).2 (fun h => hloop h.symm)
          have hbodyBeq : ("body" == label) = false :=
            (beq_eq_false_iff_ne).2 (fun h => hbody h.symm)
          have hexitBeq : ("exit" == label) = false :=
            (beq_eq_false_iff_ne).2 (fun h => hexit h.symm)
          simp [matmulLoopEnv, Std.HashMap.getElem?_insert, hentryBeq, hloopBeq,
            hbodyBeq, hexitBeq] at hlookup

theorem matmulLoop_body_ordinary (params : MatmulCellParams) :
    CFGBodyUsesOrdinaryPcAdvance (matmulLoopEnv params) := by
  intro label block idx gi hlookup hgi
  rcases matmulLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [matmulLoopEntryBlock, matmulLoopSetK, matmulLoopSetAcc,
          matmulLoopSetAPtr, matmulLoopSetBPtr, Helpers.instrUsesOrdinaryPcAdvance] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [matmulLoopEntryBlock, matmulLoopSetK, matmulLoopSetAcc,
              matmulLoopSetAPtr, matmulLoopSetBPtr, Helpers.instrUsesOrdinaryPcAdvance] at hgi
            subst gi
            rfl
        | succ idx =>
            cases idx with
            | zero =>
                simp [matmulLoopEntryBlock, matmulLoopSetK, matmulLoopSetAcc,
                  matmulLoopSetAPtr, matmulLoopSetBPtr,
                  Helpers.instrUsesOrdinaryPcAdvance] at hgi
                subst gi
                rfl
            | succ idx =>
                cases idx with
                | zero =>
                    simp [matmulLoopEntryBlock, matmulLoopSetK, matmulLoopSetAcc,
                      matmulLoopSetAPtr, matmulLoopSetBPtr,
                      Helpers.instrUsesOrdinaryPcAdvance] at hgi
                    subst gi
                    rfl
                | succ idx =>
                    simp [matmulLoopEntryBlock, matmulLoopSetK, matmulLoopSetAcc,
                      matmulLoopSetAPtr, matmulLoopSetBPtr] at hgi
  · rcases hloop with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [matmulLoopHeaderBlock, matmulLoopSetPred,
          Helpers.instrUsesOrdinaryPcAdvance] at hgi
        subst gi
        rfl
    | succ idx =>
        simp [matmulLoopHeaderBlock, matmulLoopSetPred] at hgi
  · rcases hbody with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
          matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK, matmulLoopIncAPtr,
          matmulLoopIncBPtr, Helpers.instrUsesOrdinaryPcAdvance] at hgi
        subst gi
        rfl
    | succ idx =>
        cases idx with
        | zero =>
            simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
              matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK, matmulLoopIncAPtr,
              matmulLoopIncBPtr, Helpers.instrUsesOrdinaryPcAdvance] at hgi
            subst gi
            rfl
        | succ idx =>
            cases idx with
            | zero =>
                simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
                  matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK, matmulLoopIncAPtr,
                  matmulLoopIncBPtr, Helpers.instrUsesOrdinaryPcAdvance] at hgi
                subst gi
                rfl
            | succ idx =>
                cases idx with
                | zero =>
                    simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
                      matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK,
                      matmulLoopIncAPtr, matmulLoopIncBPtr,
                      Helpers.instrUsesOrdinaryPcAdvance] at hgi
                    subst gi
                    rfl
                | succ idx =>
                    cases idx with
                    | zero =>
                        simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
                          matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK,
                          matmulLoopIncAPtr, matmulLoopIncBPtr,
                          Helpers.instrUsesOrdinaryPcAdvance] at hgi
                        subst gi
                        rfl
                    | succ idx =>
                        cases idx with
                        | zero =>
                            simp [matmulLoopBodyBlock, matmulLoopLoadA, matmulLoopLoadB,
                              matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK,
                              matmulLoopIncAPtr, matmulLoopIncBPtr,
                              Helpers.instrUsesOrdinaryPcAdvance] at hgi
                            subst gi
                            rfl
                        | succ idx =>
                            cases idx with
                            | zero =>
                                simp [matmulLoopBodyBlock, matmulLoopLoadA,
                                  matmulLoopLoadB, matmulLoopMul, matmulLoopAddAcc,
                                  matmulLoopIncK, matmulLoopIncAPtr, matmulLoopIncBPtr,
                                  Helpers.instrUsesOrdinaryPcAdvance] at hgi
                                subst gi
                                rfl
                            | succ idx =>
                                simp [matmulLoopBodyBlock, matmulLoopLoadA,
                                  matmulLoopLoadB, matmulLoopMul, matmulLoopAddAcc,
                                  matmulLoopIncK, matmulLoopIncAPtr, matmulLoopIncBPtr] at hgi
  · rcases hexit with ⟨_, hblock⟩
    subst block
    cases idx with
    | zero =>
        simp [matmulLoopExitBlock, matmulLoopStore,
          Helpers.instrUsesOrdinaryPcAdvance] at hgi
        subst gi
        rfl
    | succ idx =>
        simp [matmulLoopExitBlock, matmulLoopStore] at hgi

theorem matmulLoop_targets_exist (params : MatmulCellParams) :
    CFGTerminatorTargetsExist (matmulLoopEnv params) := by
  intro label block hlookup
  rcases matmulLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨_, hblock⟩
    subst block
    simp [matmulLoopEntryBlock]
    exact ⟨matmulLoopHeaderBlock, matmulLoop_loop_lookup params⟩
  · rcases hloop with ⟨_, hblock⟩
    subst block
    simp [matmulLoopHeaderBlock]
    exact ⟨⟨matmulLoopBodyBlock, matmulLoop_body_lookup params⟩,
      matmulLoopExitBlock params, matmulLoop_exit_lookup params⟩
  · rcases hbody with ⟨_, hblock⟩
    subst block
    simp [matmulLoopBodyBlock]
    exact ⟨matmulLoopHeaderBlock, matmulLoop_loop_lookup params⟩
  · rcases hexit with ⟨_, hblock⟩
    subst block
    simp [matmulLoopExitBlock]

theorem matmulLoop_entry_term_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut) "entry" (.br "loop")
      (matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP) := by
  refine TerminatorVC.br ?_
  have hbr :
      matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (matmulLoopAt lane ("loop", 0) 0 0 oldA oldB oldProd
            (.gaddr .global (matmulAOffset params 0))
            (.gaddr .global (matmulBOffset params 0)) oldP params aBytes bBytes oldOut) := by
    simpa [matmulLoopEntryTermPre, matmulLoopAt] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("entry", 4)) (target := "loop")
        (lanes := [lane])
        (frame :=
          matmulLoopRegs lane 0 0 oldA oldB oldProd
            (.gaddr .global (matmulAOffset params 0))
            (.gaddr .global (matmulBOffset params 0)) oldP ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (matmulLoopAt_stable_terminator_frame (.br "loop") lane 0 0 oldA oldB
          oldProd (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) oldP params aBytes bBytes oldOut))
  exact CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st r hpost
      refine ⟨0, oldA, oldB, oldProd, ?_, ?_⟩
      · omega
      · simpa using hpost)

theorem matmulLoop_entry_instrs_wp
    (lane : LaneId) (params : MatmulCellParams) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value)
    (oldP : Bool) :
    matmulLoopEntryPre lane params aBytes bBytes oldOut oldK oldAcc oldA oldB oldProd
        oldAPtr oldBPtr oldP ⊢ₛ
      wpInstrs 0 0 (matmulLoopEntryBlock params).body
        (matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP) := by
  let frameK :=
    CSL.sepList [
      CSL.reg 0 0 lane "acc" oldAcc,
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" oldAPtr,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut
  have hsetK :
      matmulLoopEntryPre lane params aBytes bBytes oldOut oldK oldAcc oldA oldB oldProd
          oldAPtr oldBPtr oldP ⊢ₛ
        wpInstr 0 0 matmulLoopSetK
          (matmulLoopEntryAfterK lane params aBytes bBytes oldOut oldAcc oldA oldB oldProd
            oldAPtr oldBPtr oldP) := by
    have hfocus :
        matmulLoopEntryPre lane params aBytes bBytes oldOut oldK oldAcc oldA oldB
            oldProd oldAPtr oldBPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 0) [lane] ∗
            (CSL.reg 0 0 lane "k" oldK ∗ frameK)) := by
      simpa [matmulLoopEntryPre, frameK] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 0) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "k" oldK,
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "k" oldK)
            (CSL.reg 0 0 lane "acc" oldAcc)
            [CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (List.Perm.refl _))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 matmulLoopSetK) frameK := by
      unfold frameK
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq | hq | hq | hq
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_pred_assignReg
      · exact matmulLoopFrame_stable_assignReg "k" (.imm (.s32 0))
          params aBytes bBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 0) [lane] ∗
          (CSL.reg 0 0 lane "k" oldK ∗ frameK)) ⊢ₛ
          wpInstr 0 0 matmulLoopSetK
            (warpAt 0 0 ("entry", 1) [lane] ∗
              (CSL.reg 0 0 lane "k" (.s32 0) ∗ frameK)) := by
      simpa [matmulLoopSetK, frameK] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 0)) (dst := "k")
          (rhs := .imm (.s32 0)) (lane := lane) (old := oldK) (new := .s32 0)
          (frame := frameK)
          (by intro _st _r _hpre; exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 1) [lane] ∗
          (CSL.reg 0 0 lane "k" (.s32 0) ∗ frameK)) ⊢ₛ
          matmulLoopEntryAfterK lane params aBytes bBytes oldOut oldAcc oldA oldB
            oldProd oldAPtr oldBPtr oldP := by
      simpa [matmulLoopEntryAfterK, frameK] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 1) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "k" (.s32 0))
            (CSL.reg 0 0 lane "acc" oldAcc)
            [CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (List.Perm.refl _))
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  let frameAcc :=
    CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" oldAPtr,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut
  have hsetAcc :
      matmulLoopEntryAfterK lane params aBytes bBytes oldOut oldAcc oldA oldB oldProd
          oldAPtr oldBPtr oldP ⊢ₛ
        wpInstr 0 0 matmulLoopSetAcc
          (matmulLoopEntryAfterAcc lane params aBytes bBytes oldOut oldA oldB oldProd
            oldAPtr oldBPtr oldP) := by
    have hfocus :
        matmulLoopEntryAfterK lane params aBytes bBytes oldOut oldAcc oldA oldB oldProd
            oldAPtr oldBPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 1) [lane] ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗ frameAcc)) := by
      simpa [matmulLoopEntryAfterK, frameAcc] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 1) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" oldAcc,
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "acc" oldAcc)
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_bacdefgh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" oldAcc)
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" oldAPtr)
              (CSL.reg 0 0 lane "bPtr" oldBPtr)
              (CSL.pred 0 0 lane "p" oldP)))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 matmulLoopSetAcc) frameAcc := by
      unfold frameAcc
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq | hq | hq | hq
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_pred_assignReg
      · exact matmulLoopFrame_stable_assignReg "acc" (.imm (.s32 0))
          params aBytes bBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗ frameAcc)) ⊢ₛ
          wpInstr 0 0 matmulLoopSetAcc
            (warpAt 0 0 ("entry", 2) [lane] ∗
              (CSL.reg 0 0 lane "acc" (.s32 0) ∗ frameAcc)) := by
      simpa [matmulLoopSetAcc, frameAcc] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 1)) (dst := "acc")
          (rhs := .imm (.s32 0)) (lane := lane) (old := oldAcc) (new := .s32 0)
          (frame := frameAcc)
          (by intro _st _r _hpre; exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 2) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 0) ∗ frameAcc)) ⊢ₛ
          matmulLoopEntryAfterAcc lane params aBytes bBytes oldOut oldA oldB oldProd
            oldAPtr oldBPtr oldP := by
      simpa [matmulLoopEntryAfterAcc, frameAcc] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 2) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "acc" (.s32 0))
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_bacdefgh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" oldAPtr)
              (CSL.reg 0 0 lane "bPtr" oldBPtr)
              (CSL.pred 0 0 lane "p" oldP)).symm)
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  let frameAPtr :=
    CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "bPtr" oldBPtr,
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut
  have hsetAPtr :
      matmulLoopEntryAfterAcc lane params aBytes bBytes oldOut oldA oldB oldProd
          oldAPtr oldBPtr oldP ⊢ₛ
        wpInstr 0 0 (matmulLoopSetAPtr params)
          (matmulLoopEntryAfterAPtr lane params aBytes bBytes oldOut oldA oldB oldProd
            oldBPtr oldP) := by
    have hfocus :
        matmulLoopEntryAfterAcc lane params aBytes bBytes oldOut oldA oldB oldProd
            oldAPtr oldBPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 2) [lane] ∗
            (CSL.reg 0 0 lane "aPtr" oldAPtr ∗ frameAPtr)) := by
      simpa [matmulLoopEntryAfterAcc, frameAPtr] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 2) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" oldAPtr,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "aPtr" oldAPtr)
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_fabcdegh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" oldAPtr)
              (CSL.reg 0 0 lane "bPtr" oldBPtr)
              (CSL.pred 0 0 lane "p" oldP)))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 (matmulLoopSetAPtr params)) frameAPtr := by
      unfold frameAPtr
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq | hq | hq | hq
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_pred_assignReg
      · exact matmulLoopFrame_stable_assignReg "aPtr"
          (.imm (.gaddr .global (matmulAOffset params 0))) params aBytes bBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 2) [lane] ∗
          (CSL.reg 0 0 lane "aPtr" oldAPtr ∗ frameAPtr)) ⊢ₛ
          wpInstr 0 0 (matmulLoopSetAPtr params)
            (warpAt 0 0 ("entry", 3) [lane] ∗
              (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)) ∗
                frameAPtr)) := by
      simpa [matmulLoopSetAPtr, frameAPtr] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 2)) (dst := "aPtr")
          (rhs := .imm (.gaddr .global (matmulAOffset params 0))) (lane := lane)
          (old := oldAPtr) (new := .gaddr .global (matmulAOffset params 0))
          (frame := frameAPtr)
          (by intro _st _r _hpre; exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 3) [lane] ∗
          (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)) ∗
            frameAPtr)) ⊢ₛ
          matmulLoopEntryAfterAPtr lane params aBytes bBytes oldOut oldA oldB oldProd
            oldBPtr oldP := by
      simpa [matmulLoopEntryAfterAPtr, frameAPtr] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 3) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)))
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_fabcdegh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)))
              (CSL.reg 0 0 lane "bPtr" oldBPtr)
              (CSL.pred 0 0 lane "p" oldP)).symm)
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  let frameBPtr :=
    CSL.sepList [
      CSL.reg 0 0 lane "k" (.s32 0),
      CSL.reg 0 0 lane "acc" (.s32 0),
      CSL.reg 0 0 lane "aVal" oldA,
      CSL.reg 0 0 lane "bVal" oldB,
      CSL.reg 0 0 lane "prod" oldProd,
      CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
      CSL.pred 0 0 lane "p" oldP] ∗
      matmulLoopFrame params aBytes bBytes oldOut
  have hsetBPtr :
      matmulLoopEntryAfterAPtr lane params aBytes bBytes oldOut oldA oldB oldProd
          oldBPtr oldP ⊢ₛ
        wpInstr 0 0 (matmulLoopSetBPtr params)
          (matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP) := by
    have hfocus :
        matmulLoopEntryAfterAPtr lane params aBytes bBytes oldOut oldA oldB oldProd
            oldBPtr oldP ⊢ₛ
          (warpAt 0 0 ("entry", 3) [lane] ∗
            (CSL.reg 0 0 lane "bPtr" oldBPtr ∗ frameBPtr)) := by
      simpa [matmulLoopEntryAfterAPtr, frameBPtr] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 3) [lane]))
          (CSL.sepList_perm_frame_to_cons
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
              CSL.reg 0 0 lane "bPtr" oldBPtr,
              CSL.pred 0 0 lane "p" oldP]
            (CSL.reg 0 0 lane "bPtr" oldBPtr)
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_gabcdefh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)))
              (CSL.reg 0 0 lane "bPtr" oldBPtr)
              (CSL.pred 0 0 lane "p" oldP)))
    have hstable :
        CSL.StableUnder (InstrStep 0 0 (matmulLoopSetBPtr params)) frameBPtr := by
      unfold frameBPtr
      refine CSL.stable_sep ?_ ?_
      · apply CSL.stable_sepList
        intro q hq
        simp at hq
        rcases hq with hq | hq | hq | hq | hq | hq | hq
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_reg_assignReg_of_ne (by decide)
        · subst q; exact stable_pred_assignReg
      · exact matmulLoopFrame_stable_assignReg "bPtr"
          (.imm (.gaddr .global (matmulBOffset params 0))) params aBytes bBytes oldOut
    have hrule :
        (warpAt 0 0 ("entry", 3) [lane] ∗
          (CSL.reg 0 0 lane "bPtr" oldBPtr ∗ frameBPtr)) ⊢ₛ
          wpInstr 0 0 (matmulLoopSetBPtr params)
            (warpAt 0 0 ("entry", 4) [lane] ∗
              (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0)) ∗
                frameBPtr)) := by
      simpa [matmulLoopSetBPtr, frameBPtr] using
        (wp_assignReg_single_warpAt_stableFrame
          (cta := 0) (warp := 0) (pc := ("entry", 3)) (dst := "bPtr")
          (rhs := .imm (.gaddr .global (matmulBOffset params 0))) (lane := lane)
          (old := oldBPtr) (new := .gaddr .global (matmulBOffset params 0))
          (frame := frameBPtr)
          (by intro _st _r _hpre; exact eval_imm)
          hstable)
    have hpost :
        (warpAt 0 0 ("entry", 4) [lane] ∗
          (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0)) ∗
            frameBPtr)) ⊢ₛ
          matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP := by
      simpa [matmulLoopEntryTermPre, frameBPtr, matmulLoopAt, matmulLoopRegs] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("entry", 4) [lane]))
          (CSL.sep_cons_frame_to_sepList_perm
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0)))
            (CSL.reg 0 0 lane "k" (.s32 0))
            [CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
              CSL.pred 0 0 lane "p" oldP]
            [CSL.reg 0 0 lane "k" (.s32 0),
              CSL.reg 0 0 lane "acc" (.s32 0),
              CSL.reg 0 0 lane "aVal" oldA,
              CSL.reg 0 0 lane "bVal" oldB,
              CSL.reg 0 0 lane "prod" oldProd,
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)),
              CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0)),
              CSL.pred 0 0 lane "p" oldP]
            (matmulLoopFrame params aBytes bBytes oldOut)
            (perm_abcdefgh_to_gabcdefh
              (CSL.reg 0 0 lane "k" (.s32 0))
              (CSL.reg 0 0 lane "acc" (.s32 0))
              (CSL.reg 0 0 lane "aVal" oldA)
              (CSL.reg 0 0 lane "bVal" oldB)
              (CSL.reg 0 0 lane "prod" oldProd)
              (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0)))
              (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0)))
              (CSL.pred 0 0 lane "p" oldP)).symm)
    exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))
  simpa [matmulLoopEntryBlock, wpInstrs, wpInstrList, matmulLoopSetK,
    matmulLoopSetAcc, matmulLoopSetAPtr, matmulLoopSetBPtr] using
    CSL.entails_trans hsetK
      (wpInstr_mono <|
        CSL.entails_trans hsetAcc
          (wpInstr_mono <|
            CSL.entails_trans hsetAPtr (wpInstr_mono hsetBPtr)))

theorem matmulLoop_cbr_true_raw_wp
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (k : Nat) (aVal bVal prod : Value) :
    matmulLoopAt lane ("loop", 1) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) aVal bVal prod
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (matmulLoopAt lane ("body", 0) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  simpa [matmulLoopAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred true) (takeTrue := true)
      (frame :=
        matmulLoopRegs lane (Int.ofNat k) (matmulLoopPrefixS32 params a b k)
          aVal bVal prod (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true ∗
          matmulLoopFrame params aBytes bBytes oldOut)
      (by
        intro st r hpre
        have hfocus :=
          matmulLoopAt_to_pred_focus lane ("loop", 1) (Int.ofNat k)
            (matmulLoopPrefixS32 params a b k) aVal bVal prod
            (.gaddr .global (matmulAOffset params k))
            (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut
            st r (by simpa [matmulLoopAt] using hpre)
        rcases hfocus with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, hpred, _hframe⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (matmulLoopAt_stable_terminator_frame (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat k) (matmulLoopPrefixS32 params a b k) aVal bVal prod
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut))

theorem matmulLoop_cbr_false_raw_wp
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (aVal bVal prod : Value) :
    matmulLoopAt lane ("loop", 1) (Int.ofNat matmulLoopProofN)
        (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
        (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes oldOut ⊢ₛ
      wpTerminator 0 0 (.cbr (.pred "p") "body" "exit")
        (matmulLoopAt lane ("exit", 0) (Int.ofNat matmulLoopProofN)
          (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
          (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
          params aBytes bBytes oldOut) := by
  simpa [matmulLoopAt] using
    (wp_cbr_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := ("loop", 1)) (cond := .pred "p")
      (tLabel := "body") (fLabel := "exit") (lane := lane)
      (value := .pred false) (takeTrue := false)
      (frame :=
        matmulLoopRegs lane (Int.ofNat matmulLoopProofN)
          (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
          (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false ∗
          matmulLoopFrame params aBytes bBytes oldOut)
      (by
        intro st r hpre
        have hfocus :=
          matmulLoopAt_to_pred_focus lane ("loop", 1) (Int.ofNat matmulLoopProofN)
            (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
            (.gaddr .global (matmulAOffset params matmulLoopProofN))
            (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
            params aBytes bBytes oldOut st r (by simpa [matmulLoopAt] using hpre)
        rcases hfocus with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, hpred, _hframe⟩
        exact eval_pred_of_assertion hpred)
      rfl
      (matmulLoopAt_stable_terminator_frame (.cbr (.pred "p") "body" "exit") lane
        (Int.ofNat matmulLoopProofN) (matmulLoopPrefixS32 params a b matmulLoopProofN)
        aVal bVal prod (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes oldOut))

theorem matmulLoop_cbr_true_control
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "body"
      (matmulLoopHeaderTruePre lane params a b aBytes bBytes oldOut) := by
  intro st st' r hpre hstep
  rcases hpre with ⟨k, aVal, bVal, prod, _hlt, hat⟩
  rcases matmulLoop_cbr_true_raw_wp lane params a b aBytes bBytes oldOut k aVal bVal prod
      st r hat st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem matmulLoop_cbr_false_control
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) :
    CbrBranchControl 0 0 (.pred "p") "body" "exit" "exit"
      (matmulLoopHeaderFalsePre lane params a b aBytes bBytes oldOut) := by
  intro st st' r hpre hstep
  rcases hpre with ⟨aVal, bVal, prod, hat⟩
  rcases matmulLoop_cbr_false_raw_wp lane params a b aBytes bBytes oldOut aVal bVal prod
      st r hat st' hstep with
    ⟨_r', _hupdate, hpost⟩
  rcases hpost with ⟨rCtrl, _rFrame, _hcomp, _hequiv, hctrl, _hframe⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  exact ⟨warpState, hwarp, hlock, hrpc⟩

theorem matmulLoop_header_term_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut) "loop"
      (.cbr (.pred "p") "body" "exit")
      (matmulLoopHeaderTermPre lane params a b aBytes bBytes oldOut) := by
  refine TerminatorVC.cbr
    (truePre := matmulLoopHeaderTruePre lane params a b aBytes bBytes oldOut)
    (falsePre := matmulLoopHeaderFalsePre lane params a b aBytes bBytes oldOut)
    ?_ (matmulLoop_cbr_true_control lane params a b aBytes bBytes oldOut)
    (matmulLoop_cbr_false_control lane params a b aBytes bBytes oldOut) ?_ ?_
  · intro st r hpre
    simpa [matmulLoopHeaderTermPre] using hpre
  · intro st r hpre
    rcases hpre with ⟨k, aVal, bVal, prod, hlt, hat⟩
    exact (CSL.entails_trans
      (matmulLoop_cbr_true_raw_wp lane params a b aBytes bBytes oldOut k aVal bVal prod) <|
      wpTerminator_mono (by
        intro st' r' hbody
        exact ⟨k, aVal, bVal, prod, hlt, hbody⟩)) st r hat
  · intro st r hpre
    rcases hpre with ⟨aVal, bVal, prod, hat⟩
    exact (CSL.entails_trans
      (matmulLoop_cbr_false_raw_wp lane params a b aBytes bBytes oldOut aVal bVal prod) <|
      wpTerminator_mono (by
        intro st' r' hexit
        exact ⟨aVal, bVal, prod, hexit⟩)) st r hat

theorem matmulLoop_header_instrs_wp
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) :
    matmulLoopLoopInv lane params a b aBytes bBytes oldOut ⊢ₛ
      wpInstrs 0 0 matmulLoopHeaderBlock.body
        (matmulLoopHeaderTermPre lane params a b aBytes bBytes oldOut) := by
  intro st r hpre
  rcases hpre with ⟨k, aVal, bVal, prod, p, hkLe, hat⟩
  have htoRule :=
    matmulLoopAt_to_pred_focus lane ("loop", 0) (Int.ofNat k)
      (matmulLoopPrefixS32 params a b k) aVal bVal prod
      (.gaddr .global (matmulAOffset params k))
      (.gaddr .global (matmulBOffset params k)) p params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("loop", 0) [lane] ∗
        (CSL.pred 0 0 lane "p" p ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k))] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopSetPred
          (warpAt 0 0 ("loop", 1) [lane] ∗
            (CSL.pred 0 0 lane "p"
              (decide (Int.ofNat k < Int.ofNat matmulLoopProofN)) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
                CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
                CSL.reg 0 0 lane "aVal" aVal,
                CSL.reg 0 0 lane "bVal" bVal,
                CSL.reg 0 0 lane "prod" prod,
                CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
                CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k))] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopSetPred] using
      (wp_assignPred_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("loop", 0)) (dst := "p")
        (cmp := { op := .lt, lhs := .reg "k", rhs := .imm (.s32 (Int.ofNat matmulLoopProofN)) })
        (lane := lane) (old := p)
        (new := decide (Int.ofNat k < Int.ofNat matmulLoopProofN))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k))] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPred, _rFrame, _hcompRest, _hequivRest, _hpred,
            hframe⟩
          rcases hframe with ⟨_rRegs, _rMem, _hcompFrame, _hequivFrame, hregs, _hmem⟩
          rcases hregs with ⟨_rK, _rRegsRest, _hcompK, _hequivK, hk, _hregsRest⟩
          exact EvalCmp.lt_s32 (eval_reg_of_assertion hk) eval_imm)
        (by
          refine CSL.stable_sep ?_ ?_
          · apply CSL.stable_sepList
            intro q hq
            simp at hq
            rcases hq with hq | hq | hq | hq | hq | hq | hq
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
            · subst q
              exact stable_reg_assignPred
          · exact matmulLoopFrame_stable_assignPred "p"
              { op := .lt, lhs := .reg "k",
                rhs := .imm (.s32 (Int.ofNat matmulLoopProofN)) }
              params aBytes bBytes oldOut))
  have hpost :
      (warpAt 0 0 ("loop", 1) [lane] ∗
        (CSL.pred 0 0 lane "p" (decide (Int.ofNat k < Int.ofNat matmulLoopProofN)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" aVal,
            CSL.reg 0 0 lane "bVal" bVal,
            CSL.reg 0 0 lane "prod" prod,
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k))] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopHeaderTermPre lane params a b aBytes bBytes oldOut := by
    intro st r hraw
    have hstandard :
        matmulLoopAt lane ("loop", 1) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k))
          (decide (Int.ofNat k < Int.ofNat matmulLoopProofN))
          params aBytes bBytes oldOut st r :=
      matmulLoopAt_pred_focus_to_standard lane ("loop", 1) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) aVal bVal prod
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k))
        (decide (Int.ofNat k < Int.ofNat matmulLoopProofN))
        params aBytes bBytes oldOut st r hraw
    by_cases hlt : k < matmulLoopProofN
    · have hltInt : Int.ofNat k < Int.ofNat matmulLoopProofN := Int.ofNat_lt.mpr hlt
      have hdec :
          decide (Int.ofNat k < Int.ofNat matmulLoopProofN) = true := by
        simp [hltInt, hlt]
      left
      refine ⟨k, aVal, bVal, prod, hlt, ?_⟩
      rw [hdec] at hstandard
      exact hstandard
    · have hge : matmulLoopProofN ≤ k := Nat.le_of_not_gt hlt
      have hkEq : k = matmulLoopProofN := Nat.le_antisymm hkLe hge
      subst k
      have hnotInt :
          ¬ Int.ofNat matmulLoopProofN < Int.ofNat matmulLoopProofN := by
        exact Int.lt_irrefl (Int.ofNat matmulLoopProofN)
      have hdec :
          decide (Int.ofNat matmulLoopProofN < Int.ofNat matmulLoopProofN) = false := by
        simp [hnotInt]
      right
      refine ⟨aVal, bVal, prod, ?_⟩
      rw [hdec] at hstandard
      simpa [matmulLoopHeaderFalsePre] using hstandard
  simpa [matmulLoopHeaderBlock, wpInstrs, wpInstrList, matmulLoopSetPred] using
    (CSL.entails_trans htoRule (CSL.entails_trans hrule (wpInstr_mono hpost)) st r hat)

theorem matmulLoop_inc_s32_of_lt_proofN {k : Nat} (h : k < matmulLoopProofN) :
    Helpers.normalizeSigned 32 (Int.ofNat k + 1) = Int.ofNat (k + 1) := by
  unfold matmulLoopProofN at h
  cases k with
  | zero =>
      simp [Helpers.normalizeSigned]
  | succ k =>
      cases k with
      | zero =>
          simp [Helpers.normalizeSigned]
      | succ k =>
          cases k with
          | zero =>
              simp [Helpers.normalizeSigned]
          | succ k =>
              omega

theorem matmulLoop_aPtr_advance (params : MatmulCellParams) (k : Nat) :
    matmulAOffset params k + (4 : UInt64).toNat = matmulAOffset params (k + 1) := by
  simp [matmulAOffset]
  omega

theorem matmulLoop_bPtr_advance
    (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN) (k : Nat) :
    matmulBOffset params k + (UInt64.ofNat matmulLoopBStride).toNat =
      matmulBOffset params (k + 1) := by
  simp [matmulBOffset, matmulLoopBStride, matmulLoopProofN, hparamsN]
  omega

theorem matmulLoop_mul_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (k : Nat) (oldProd : Value) :
    matmulLoopAt lane ("body", 2) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
        (.s32 (b k params.col)) oldProd
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopMul
        (matmulLoopAt lane ("body", 3) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  have hfocus :
      matmulLoopAt lane ("body", 2) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
          (.s32 (b k params.col)) oldProd
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 2) [lane] ∗
          (CSL.reg 0 0 lane "prod" oldProd ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
              CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
              CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
              CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
              CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
              CSL.pred 0 0 lane "p" true] ∗
              matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 2) [lane]))
        (CSL.sepList_perm_frame_to_cons
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (CSL.reg 0 0 lane "prod" oldProd)
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_eabcdfgh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" oldProd)
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)))
  have hstable :
      CSL.StableUnder (InstrStep 0 0 matmulLoopMul)
        (CSL.sepList [
          CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
          CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
          CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
          CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
          CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
          CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
          CSL.pred 0 0 lane "p" true] ∗
          matmulLoopFrame params aBytes bBytes oldOut) := by
    refine CSL.stable_sep ?_ ?_
    · apply CSL.stable_sepList
      intro q hq
      simp at hq
      rcases hq with hq | hq | hq | hq | hq | hq | hq
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_pred_assignReg
    · exact matmulLoopFrame_stable_assignReg "prod"
        (.binop .mul (.reg "aVal") (.reg "bVal")) params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 2) [lane] ∗
        (CSL.reg 0 0 lane "prod" oldProd ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopMul
          (warpAt 0 0 ("body", 3) [lane] ∗
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
                CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
                CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
                CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
                CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
                CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
                CSL.pred 0 0 lane "p" true] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopMul] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 2)) (dst := "prod")
        (rhs := .binop .mul (.reg "aVal") (.reg "bVal")) (lane := lane)
        (old := oldProd) (new := .s32 (matmulLoopTermS32 params a b k))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rProd, _rFrame, _hcompRest, _hequivRest, _hprod,
            hframe⟩
          rcases hframe with ⟨_rRegs, _rMem, _hcompFrame, _hequivFrame, hregs, _hmem⟩
          rcases hregs with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rA, _rRestA, _hcompA, _hequivA, haVal, hrestA⟩
          rcases hrestA with ⟨_rB, _rRestB, _hcompB, _hequivB, hbVal, _hrestB⟩
          simpa [matmulLoopTermS32] using
            EvalRValue.binop_mul_s32 (eval_reg_of_assertion haVal)
              (eval_reg_of_assertion hbVal))
        hstable)
  have hpost :
      (warpAt 0 0 ("body", 3) [lane] ∗
        (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopAt lane ("body", 3) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 3) [lane]))
        (CSL.sep_cons_frame_to_sepList_perm
          (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_eabcdfgh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)).symm)
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem matmulLoop_add_acc_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) (k : Nat) :
    matmulLoopAt lane ("body", 3) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
        (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopAddAcc
        (matmulLoopAt lane ("body", 4) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  have hfocus :
      matmulLoopAt lane ("body", 3) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 3) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)) ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
              CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
              CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
              CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
              CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
              CSL.pred 0 0 lane "p" true] ∗
              matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 3) [lane]))
        (CSL.sepList_perm_frame_to_cons
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
          [CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_bacdefgh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)))
  have hstable :
      CSL.StableUnder (InstrStep 0 0 matmulLoopAddAcc)
        (CSL.sepList [
          CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
          CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
          CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
          CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
          CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
          CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
          CSL.pred 0 0 lane "p" true] ∗
          matmulLoopFrame params aBytes bBytes oldOut) := by
    refine CSL.stable_sep ?_ ?_
    · apply CSL.stable_sepList
      intro q hq
      simp at hq
      rcases hq with hq | hq | hq | hq | hq | hq | hq
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_pred_assignReg
    · exact matmulLoopFrame_stable_assignReg "acc"
        (.binop .add (.reg "acc") (.reg "prod")) params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 3) [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b k)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopAddAcc
          (warpAt 0 0 ("body", 4) [lane] ∗
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
                CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
                CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
                CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
                CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
                CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
                CSL.pred 0 0 lane "p" true] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAddAcc] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 3)) (dst := "acc")
        (rhs := .binop .add (.reg "acc") (.reg "prod")) (lane := lane)
        (old := .s32 (matmulLoopPrefixS32 params a b k))
        (new := .s32 (matmulLoopPrefixS32 params a b (k + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rAcc, _rFrame, _hcompRest, _hequivRest, hacc, hframe⟩
          rcases hframe with ⟨_rRegs, _rMem, _hcompFrame, _hequivFrame, hregs, _hmem⟩
          rcases hregs with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rA, _rRestA, _hcompA, _hequivA, _ha, hrestA⟩
          rcases hrestA with ⟨_rB, _rRestB, _hcompB, _hequivB, _hb, hrestB⟩
          rcases hrestB with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, hprod,
            _hrestProd⟩
          simpa [matmulLoopPrefixS32, matmulLoopTermS32, Nat.succ_eq_add_one] using
            EvalRValue.binop_add_s32 (eval_reg_of_assertion hacc)
              (eval_reg_of_assertion hprod))
        hstable)
  have hpost :
      (warpAt 0 0 ("body", 4) [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopAt lane ("body", 4) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 4) [lane]))
        (CSL.sep_cons_frame_to_sepList_perm
          (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
          [CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_bacdefgh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)).symm)
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem matmulLoop_inc_k_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (k : Nat) (hlt : k < matmulLoopProofN) :
    matmulLoopAt lane ("body", 4) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
        (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopIncK
        (matmulLoopAt lane ("body", 5) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  have hinc := matmulLoop_inc_s32_of_lt_proofN hlt
  have hincValue :
      Helpers.normalizeSigned 32 (Int.ofNat k + 1) = Int.ofNat k + 1 := by
    rw [hinc]
    simp
  have hfocus :
      matmulLoopAt lane ("body", 4) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 4) [lane] ∗
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)) ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
              CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
              CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
              CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
              CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
              CSL.pred 0 0 lane "p" true] ∗
              matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 4) [lane]))
        (CSL.sepList_perm_frame_to_cons
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)))
          (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
          [CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (List.Perm.refl _))
  have hstable :
      CSL.StableUnder (InstrStep 0 0 matmulLoopIncK)
        (CSL.sepList [
          CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
          CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
          CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
          CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
          CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
          CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
          CSL.pred 0 0 lane "p" true] ∗
          matmulLoopFrame params aBytes bBytes oldOut) := by
    refine CSL.stable_sep ?_ ?_
    · apply CSL.stable_sepList
      intro q hq
      simp at hq
      rcases hq with hq | hq | hq | hq | hq | hq | hq
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_pred_assignReg
    · exact matmulLoopFrame_stable_assignReg "k"
        (.binop .add (.reg "k") (.imm (.s32 1))) params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 4) [lane] ∗
        (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat k)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopIncK
          (warpAt 0 0 ("body", 5) [lane] ∗
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
                CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
                CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
                CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
                CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
                CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
                CSL.pred 0 0 lane "p" true] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopIncK] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 4)) (dst := "k")
        (rhs := .binop .add (.reg "k") (.imm (.s32 1))) (lane := lane)
        (old := .s32 (Int.ofNat k)) (new := .s32 (Int.ofNat (k + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rK, _rFrame, _hcompRest, _hequivRest, hk, _hframe⟩
          have hevalNorm :
              EvalRValue st { cta := 0, warp := 0, lane := lane }
                (.binop .add (.reg "k") (.imm (.s32 1)))
                (.s32 (Helpers.normalizeSigned 32 (Int.ofNat k + 1))) :=
            EvalRValue.binop_add_s32
              (eval_reg_of_assertion
                (ctx := { cta := 0, warp := 0, lane := lane }) hk)
              (eval_imm
                (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                (value := .s32 (1 : Int)))
          rw [hincValue] at hevalNorm
          exact hevalNorm)
        hstable)
  have hpost :
      (warpAt 0 0 ("body", 5) [lane] ∗
        (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopAt lane ("body", 5) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 5) [lane]))
        (CSL.sep_cons_frame_to_sepList_perm
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
          (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
          [CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (List.Perm.refl _))
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem matmulLoop_inc_aPtr_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte) (k : Nat) :
    matmulLoopAt lane ("body", 5) (Int.ofNat (k + 1))
        (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
        (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopIncAPtr
        (matmulLoopAt lane ("body", 6) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  have hfocus :
      matmulLoopAt lane ("body", 5) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 5) [lane] ∗
          (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)) ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
              CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
              CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
              CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
              CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
              CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
              CSL.pred 0 0 lane "p" true] ∗
              matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 5) [lane]))
        (CSL.sepList_perm_frame_to_cons
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_fabcdegh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)))
  have hstable :
      CSL.StableUnder (InstrStep 0 0 matmulLoopIncAPtr)
        (CSL.sepList [
          CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
          CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
          CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
          CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
          CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
          CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
          CSL.pred 0 0 lane "p" true] ∗
          matmulLoopFrame params aBytes bBytes oldOut) := by
    refine CSL.stable_sep ?_ ?_
    · apply CSL.stable_sepList
      intro q hq
      simp at hq
      rcases hq with hq | hq | hq | hq | hq | hq | hq
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_pred_assignReg
    · exact matmulLoopFrame_stable_assignReg "aPtr"
        (.binop .add (.reg "aPtr") (.imm (.u64 (4 : UInt64))))
        params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 5) [lane] ∗
        (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params k)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopIncAPtr
          (warpAt 0 0 ("body", 6) [lane] ∗
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
                CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
                CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
                CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
                CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
                CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
                CSL.pred 0 0 lane "p" true] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopIncAPtr] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 5)) (dst := "aPtr")
        (rhs := .binop .add (.reg "aPtr") (.imm (.u64 (4 : UInt64)))) (lane := lane)
        (old := .gaddr .global (matmulAOffset params k))
        (new := .gaddr .global (matmulAOffset params (k + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPtr, _rFrame, _hcompRest, _hequivRest, hptr, _hframe⟩
          have hevalNorm :
              EvalRValue st { cta := 0, warp := 0, lane := lane }
                (.binop .add (.reg "aPtr") (.imm (.u64 (4 : UInt64))))
                (.gaddr .global ((matmulAOffset params k) + (4 : UInt64).toNat)) :=
            EvalRValue.binop_add_gaddr_u64
              (eval_reg_of_assertion
                (ctx := { cta := 0, warp := 0, lane := lane }) hptr)
              (eval_imm
                (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                (value := .u64 (4 : UInt64)))
          rw [matmulLoop_aPtr_advance] at hevalNorm
          exact hevalNorm)
        hstable)
  have hpost :
      (warpAt 0 0 ("body", 6) [lane] ∗
        (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopAt lane ("body", 6) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 6) [lane]))
        (CSL.sep_cons_frame_to_sepList_perm
          (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_fabcdegh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)).symm)
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem matmulLoop_inc_bPtr_wp_at
    (lane : LaneId) (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (k : Nat) :
    matmulLoopAt lane ("body", 6) (Int.ofNat (k + 1))
        (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
        (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
        (.gaddr .global (matmulAOffset params (k + 1)))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopIncBPtr
        (matmulLoopAt lane ("body", 7) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params (k + 1))) true params aBytes bBytes oldOut) := by
  have hfocus :
      matmulLoopAt lane ("body", 6) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 6) [lane] ∗
          (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)) ∗
            (CSL.sepList [
              CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
              CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
              CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
              CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
              CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
              CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
              CSL.pred 0 0 lane "p" true] ∗
              matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 6) [lane]))
        (CSL.sepList_perm_frame_to_cons
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)),
            CSL.pred 0 0 lane "p" true]
          (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_gabcdefh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)))
            (CSL.pred 0 0 lane "p" true)))
  have hstable :
      CSL.StableUnder (InstrStep 0 0 matmulLoopIncBPtr)
        (CSL.sepList [
          CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
          CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
          CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
          CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
          CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
          CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
          CSL.pred 0 0 lane "p" true] ∗
          matmulLoopFrame params aBytes bBytes oldOut) := by
    refine CSL.stable_sep ?_ ?_
    · apply CSL.stable_sepList
      intro q hq
      simp at hq
      rcases hq with hq | hq | hq | hq | hq | hq | hq
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_reg_assignReg_of_ne (by decide)
      · subst q; exact stable_pred_assignReg
    · exact matmulLoopFrame_stable_assignReg "bPtr"
        (.binop .add (.reg "bPtr") (.imm (.u64 (UInt64.ofNat matmulLoopBStride))))
        params aBytes bBytes oldOut
  have hrule :
      (warpAt 0 0 ("body", 6) [lane] ∗
        (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params k)) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        wpInstr 0 0 matmulLoopIncBPtr
          (warpAt 0 0 ("body", 7) [lane] ∗
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params (k + 1))) ∗
              (CSL.sepList [
                CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
                CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
                CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
                CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
                CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
                CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
                CSL.pred 0 0 lane "p" true] ∗
                matmulLoopFrame params aBytes bBytes oldOut))) := by
    simpa [matmulLoopIncBPtr] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 6)) (dst := "bPtr")
        (rhs := .binop .add (.reg "bPtr") (.imm (.u64 (UInt64.ofNat matmulLoopBStride))))
        (lane := lane) (old := .gaddr .global (matmulBOffset params k))
        (new := .gaddr .global (matmulBOffset params (k + 1)))
        (frame :=
          CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rPtr, _rFrame, _hcompRest, _hequivRest, hptr, _hframe⟩
          have hevalNorm :
              EvalRValue st { cta := 0, warp := 0, lane := lane }
                (.binop .add (.reg "bPtr") (.imm (.u64 (UInt64.ofNat matmulLoopBStride))))
                (.gaddr .global
                  ((matmulBOffset params k) + (UInt64.ofNat matmulLoopBStride).toNat)) :=
            EvalRValue.binop_add_gaddr_u64
              (eval_reg_of_assertion
                (ctx := { cta := 0, warp := 0, lane := lane }) hptr)
              (eval_imm
                (st := st) (ctx := { cta := 0, warp := 0, lane := lane })
                (value := .u64 (UInt64.ofNat matmulLoopBStride)))
          rw [matmulLoop_bPtr_advance params hparamsN k] at hevalNorm
          exact hevalNorm)
        hstable)
  have hpost :
      (warpAt 0 0 ("body", 7) [lane] ∗
        (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params (k + 1))) ∗
          (CSL.sepList [
            CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true] ∗
            matmulLoopFrame params aBytes bBytes oldOut))) ⊢ₛ
        matmulLoopAt lane ("body", 7) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params (k + 1))) true params aBytes bBytes oldOut := by
    simpa [matmulLoopAt, matmulLoopRegs] using
      CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 7) [lane]))
        (CSL.sep_cons_frame_to_sepList_perm
          (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params (k + 1))))
          (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
          [CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true]
          [CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))),
            CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))),
            CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)),
            CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)),
            CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)),
            CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))),
            CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params (k + 1))),
            CSL.pred 0 0 lane "p" true]
          (matmulLoopFrame params aBytes bBytes oldOut)
          (perm_abcdefgh_to_gabcdefh
            (CSL.reg 0 0 lane "k" (.s32 (Int.ofNat (k + 1))))
            (CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b (k + 1))))
            (CSL.reg 0 0 lane "aVal" (.s32 (a params.row k)))
            (CSL.reg 0 0 lane "bVal" (.s32 (b k params.col)))
            (CSL.reg 0 0 lane "prod" (.s32 (matmulLoopTermS32 params a b k)))
            (CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params (k + 1))))
            (CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params (k + 1))))
            (CSL.pred 0 0 lane "p" true)).symm)
  exact CSL.entails_trans hfocus (CSL.entails_trans hrule (wpInstr_mono hpost))

theorem matmulLoop_load_a_wp_at_zero
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldA bVal prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some (aBytes 0).length)
    (hdecode : DecodedScalar .s32 (aBytes 0) (.s32 (a params.row 0))) :
    matmulLoopAt lane ("body", 0) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) oldA bVal prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadA
        (matmulLoopAt lane ("body", 1) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) bVal prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 0))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 0))
  let regAOld := CSL.reg 0 0 lane "aVal" oldA
  let regANew := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 0))
  let regB := CSL.reg 0 0 lane "bVal" bVal
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
    a1, a2, b0, b1, b2, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 0) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) oldA bVal prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((a0 ∗ regAOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 0) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) oldA bVal prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regAOld, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regAOld, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              a0 regAOld regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_a0_aVal
                regK regAcc regAOld regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((a0 ∗ regANew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 1) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) bVal prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regANew, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              a0 regANew regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a1, a2, b0, b1, b2, out]
              [regK, regAcc, regANew, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_a0_aVal
                regK regAcc regANew regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 1) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) bVal prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((a0 ∗ regAOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadA
          (warpAt 0 0 ("body", 1) [lane] ∗ ((a0 ∗ regANew) ∗ frame)) := by
    simpa [matmulLoopLoadA, a0, regAOld, regANew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "aVal")
        (ty := .s32) (addrExpr := .reg "aPtr") (lane := lane)
        (offset := matmulAOffset params 0) (bytes := aBytes 0) (oldReg := oldA)
        (value := .s32 (a params.row 0)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rB, _rRestB, _hcompB, _hequivB, _hb, hrestB⟩
          rcases hrestB with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, hptr,
            _hrestAPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "aPtr") (off := matmulAOffset params 0)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
            a1, a2, b0, b1, b2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_a_wp_at_one
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldA bVal prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 1)))
    (hwidth : Typing.byteWidth? .s32 = some (aBytes 1).length)
    (hdecode : DecodedScalar .s32 (aBytes 1) (.s32 (a params.row 1))) :
    matmulLoopAt lane ("body", 0) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) oldA bVal prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadA
        (matmulLoopAt lane ("body", 1) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) bVal prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 1))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 1))
  let regAOld := CSL.reg 0 0 lane "aVal" oldA
  let regANew := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 1))
  let regB := CSL.reg 0 0 lane "bVal" bVal
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 1))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 1))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
    a0, a2, b0, b1, b2, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 0) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) oldA bVal prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((a1 ∗ regAOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 0) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) oldA bVal prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regAOld, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regAOld, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              a1 regAOld regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a0, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_a1_aVal
                regK regAcc regAOld regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((a1 ∗ regANew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 1) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) bVal prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regANew, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              a1 regANew regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a0, a2, b0, b1, b2, out]
              [regK, regAcc, regANew, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_a1_aVal
                regK regAcc regANew regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 1) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) bVal prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((a1 ∗ regAOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadA
          (warpAt 0 0 ("body", 1) [lane] ∗ ((a1 ∗ regANew) ∗ frame)) := by
    simpa [matmulLoopLoadA, a1, regAOld, regANew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "aVal")
        (ty := .s32) (addrExpr := .reg "aPtr") (lane := lane)
        (offset := matmulAOffset params 1) (bytes := aBytes 1) (oldReg := oldA)
        (value := .s32 (a params.row 1)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rB, _rRestB, _hcompB, _hequivB, _hb, hrestB⟩
          rcases hrestB with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, hptr,
            _hrestAPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "aPtr") (off := matmulAOffset params 1)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
            a0, a2, b0, b1, b2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_a_wp_at_two
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldA bVal prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 2)))
    (hwidth : Typing.byteWidth? .s32 = some (aBytes 2).length)
    (hdecode : DecodedScalar .s32 (aBytes 2) (.s32 (a params.row 2))) :
    matmulLoopAt lane ("body", 0) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) oldA bVal prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadA
        (matmulLoopAt lane ("body", 1) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) bVal prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 2))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 2))
  let regAOld := CSL.reg 0 0 lane "aVal" oldA
  let regANew := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 2))
  let regB := CSL.reg 0 0 lane "bVal" bVal
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 2))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 2))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
    a0, a1, b0, b1, b2, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 0) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) oldA bVal prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 0) [lane] ∗ ((a2 ∗ regAOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 0) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) oldA bVal prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regAOld, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 0) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regAOld, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              a2 regAOld regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a0, a1, b0, b1, b2, out]
              (perm_matmul_flat_to_a2_aVal
                regK regAcc regAOld regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((a2 ∗ regANew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 1) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) bVal prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regANew, regB, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              a2 regANew regK
              [regAcc, regB, regProd, regAPtr, regBPtr, predP, a0, a1, b0, b1, b2, out]
              [regK, regAcc, regANew, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_a2_aVal
                regK regAcc regANew regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 1) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) bVal prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 0) [lane] ∗ ((a2 ∗ regAOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadA
          (warpAt 0 0 ("body", 1) [lane] ∗ ((a2 ∗ regANew) ∗ frame)) := by
    simpa [matmulLoopLoadA, a2, regAOld, regANew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 0)) (dst := "aVal")
        (ty := .s32) (addrExpr := .reg "aPtr") (lane := lane)
        (offset := matmulAOffset params 2) (bytes := aBytes 2) (oldReg := oldA)
        (value := .s32 (a params.row 2)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rB, _rRestB, _hcompB, _hequivB, _hb, hrestB⟩
          rcases hrestB with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, hptr,
            _hrestAPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "aPtr") (off := matmulAOffset params 2)
            (eval_reg_of_assertion hptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regB, regProd, regAPtr, regBPtr, predP,
            a0, a1, b0, b1, b2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_a_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldA bVal prod : Value) (k : Nat) (hlt : k < matmulLoopProofN)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j))) :
    matmulLoopAt lane ("body", 0) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) oldA bVal prod
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadA
        (matmulLoopAt lane ("body", 1) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k)) bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  unfold matmulLoopProofN at hlt
  cases k with
  | zero =>
      simpa [matmulLoopProofN] using
        matmulLoop_load_a_wp_at_zero lane params a b aBytes bBytes oldOut oldA bVal prod
          (haccessA 0 (by simp [matmulLoopProofN]))
          (hwidthA 0 (by simp [matmulLoopProofN]))
          (hdecodeA 0 (by simp [matmulLoopProofN]))
  | succ k =>
      cases k with
      | zero =>
          simpa [matmulLoopProofN] using
            matmulLoop_load_a_wp_at_one lane params a b aBytes bBytes oldOut oldA bVal prod
              (haccessA 1 (by simp [matmulLoopProofN]))
              (hwidthA 1 (by simp [matmulLoopProofN]))
              (hdecodeA 1 (by simp [matmulLoopProofN]))
      | succ k =>
          cases k with
          | zero =>
              simpa [matmulLoopProofN] using
                matmulLoop_load_a_wp_at_two lane params a b aBytes bBytes oldOut oldA bVal prod
                  (haccessA 2 (by simp [matmulLoopProofN]))
                  (hwidthA 2 (by simp [matmulLoopProofN]))
                  (hdecodeA 2 (by simp [matmulLoopProofN]))
          | succ k =>
              omega

theorem matmulLoop_load_b_wp_at_zero
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldB prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some (bBytes 0).length)
    (hdecode : DecodedScalar .s32 (bBytes 0) (.s32 (b 0 params.col))) :
    matmulLoopAt lane ("body", 1) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) oldB prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadB
        (matmulLoopAt lane ("body", 2) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0))
          (.s32 (b 0 params.col)) prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 0))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 0))
  let regA := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 0))
  let regBOld := CSL.reg 0 0 lane "bVal" oldB
  let regBNew := CSL.reg 0 0 lane "bVal" (.s32 (b 0 params.col))
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 0))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 0))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
    a0, a1, a2, b1, b2, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 1) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) oldB prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 1) [lane] ∗ ((b0 ∗ regBOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 1) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0)) oldB prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBOld, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regA, regBOld, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              b0 regBOld regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b1, b2, out]
              (perm_matmul_flat_to_b0_bVal
                regK regAcc regA regBOld regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 2) [lane] ∗ ((b0 ∗ regBNew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 2) (Int.ofNat 0)
          (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0))
          (.s32 (b 0 params.col)) prod
          (.gaddr .global (matmulAOffset params 0))
          (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBNew, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 2) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              b0 regBNew regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b1, b2, out]
              [regK, regAcc, regA, regBNew, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_b0_bVal
                regK regAcc regA regBNew regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 2) (Int.ofNat 0)
        (matmulLoopPrefixS32 params a b 0) (.s32 (a params.row 0))
        (.s32 (b 0 params.col)) prod
        (.gaddr .global (matmulAOffset params 0))
        (.gaddr .global (matmulBOffset params 0)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((b0 ∗ regBOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadB
          (warpAt 0 0 ("body", 2) [lane] ∗ ((b0 ∗ regBNew) ∗ frame)) := by
    simpa [matmulLoopLoadB, b0, regBOld, regBNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 1)) (dst := "bVal")
        (ty := .s32) (addrExpr := .reg "bPtr") (lane := lane)
        (offset := matmulBOffset params 0) (bytes := bBytes 0) (oldReg := oldB)
        (value := .s32 (b 0 params.col)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rA, _rRestA, _hcompA, _hequivA, _ha, hrestA⟩
          rcases hrestA with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, _haptr,
            hrestAPtr⟩
          rcases hrestAPtr with ⟨_rBPtr, _rRestBPtr, _hcompBPtr, _hequivBPtr, hbptr,
            _hrestBPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "bPtr") (off := matmulBOffset params 0)
            (eval_reg_of_assertion hbptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
            a0, a1, a2, b1, b2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_b_wp_at_one
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldB prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 1)))
    (hwidth : Typing.byteWidth? .s32 = some (bBytes 1).length)
    (hdecode : DecodedScalar .s32 (bBytes 1) (.s32 (b 1 params.col))) :
    matmulLoopAt lane ("body", 1) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) oldB prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadB
        (matmulLoopAt lane ("body", 2) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1))
          (.s32 (b 1 params.col)) prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 1))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 1))
  let regA := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 1))
  let regBOld := CSL.reg 0 0 lane "bVal" oldB
  let regBNew := CSL.reg 0 0 lane "bVal" (.s32 (b 1 params.col))
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 1))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 1))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
    a0, a1, a2, b0, b2, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 1) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) oldB prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 1) [lane] ∗ ((b1 ∗ regBOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 1) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1)) oldB prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBOld, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regA, regBOld, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              b1 regBOld regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b0, b2, out]
              (perm_matmul_flat_to_b1_bVal
                regK regAcc regA regBOld regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 2) [lane] ∗ ((b1 ∗ regBNew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 2) (Int.ofNat 1)
          (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1))
          (.s32 (b 1 params.col)) prod
          (.gaddr .global (matmulAOffset params 1))
          (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBNew, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 2) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              b1 regBNew regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b0, b2, out]
              [regK, regAcc, regA, regBNew, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_b1_bVal
                regK regAcc regA regBNew regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 2) (Int.ofNat 1)
        (matmulLoopPrefixS32 params a b 1) (.s32 (a params.row 1))
        (.s32 (b 1 params.col)) prod
        (.gaddr .global (matmulAOffset params 1))
        (.gaddr .global (matmulBOffset params 1)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((b1 ∗ regBOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadB
          (warpAt 0 0 ("body", 2) [lane] ∗ ((b1 ∗ regBNew) ∗ frame)) := by
    simpa [matmulLoopLoadB, b1, regBOld, regBNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 1)) (dst := "bVal")
        (ty := .s32) (addrExpr := .reg "bPtr") (lane := lane)
        (offset := matmulBOffset params 1) (bytes := bBytes 1) (oldReg := oldB)
        (value := .s32 (b 1 params.col)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rA, _rRestA, _hcompA, _hequivA, _ha, hrestA⟩
          rcases hrestA with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, _haptr,
            hrestAPtr⟩
          rcases hrestAPtr with ⟨_rBPtr, _rRestBPtr, _hcompBPtr, _hequivBPtr, hbptr,
            _hrestBPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "bPtr") (off := matmulBOffset params 1)
            (eval_reg_of_assertion hbptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
            a0, a1, a2, b0, b2, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_b_wp_at_two
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldB prod : Value)
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 2)))
    (hwidth : Typing.byteWidth? .s32 = some (bBytes 2).length)
    (hdecode : DecodedScalar .s32 (bBytes 2) (.s32 (b 2 params.col))) :
    matmulLoopAt lane ("body", 1) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) oldB prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadB
        (matmulLoopAt lane ("body", 2) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2))
          (.s32 (b 2 params.col)) prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut) := by
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat 2))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b 2))
  let regA := CSL.reg 0 0 lane "aVal" (.s32 (a params.row 2))
  let regBOld := CSL.reg 0 0 lane "bVal" oldB
  let regBNew := CSL.reg 0 0 lane "bVal" (.s32 (b 2 params.col))
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params 2))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params 2))
  let predP := CSL.pred 0 0 lane "p" true
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let out := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let frame := CSL.sepList [regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
    a0, a1, a2, b0, b1, out]
  have hpreToRule :
      matmulLoopAt lane ("body", 1) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) oldB prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("body", 1) [lane] ∗ ((b2 ∗ regBOld) ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("body", 1) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2)) oldB prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBOld, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 1) [lane]))
            (CSL.sepList_perm_to_sep_pair_cons
              [regK, regAcc, regA, regBOld, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              b2 regBOld regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, out]
              (perm_matmul_flat_to_b2_bVal
                regK regAcc regA regBOld regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out)))
  have hpostFromRule :
      (warpAt 0 0 ("body", 2) [lane] ∗ ((b2 ∗ regBNew) ∗ frame)) ⊢ₛ
        matmulLoopAt lane ("body", 2) (Int.ofNat 2)
          (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2))
          (.s32 (b 2 params.col)) prod
          (.gaddr .global (matmulAOffset params 2))
          (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut := by
    exact CSL.entails_trans
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regBNew, regProd,
          regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, b2, out, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("body", 2) [lane]))
            (CSL.sep_pair_cons_perm_to_sepList
              b2 regBNew regK
              [regAcc, regA, regProd, regAPtr, regBPtr, predP, a0, a1, a2, b0, b1, out]
              [regK, regAcc, regA, regBNew, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, out]
              (perm_matmul_flat_to_b2_bVal
                regK regAcc regA regBNew regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 out).symm))
      (matmulLoopAt_flat_to_standard lane ("body", 2) (Int.ofNat 2)
        (matmulLoopPrefixS32 params a b 2) (.s32 (a params.row 2))
        (.s32 (b 2 params.col)) prod
        (.gaddr .global (matmulAOffset params 2))
        (.gaddr .global (matmulBOffset params 2)) true params aBytes bBytes oldOut)
  have hrule :
      (warpAt 0 0 ("body", 1) [lane] ∗ ((b2 ∗ regBOld) ∗ frame)) ⊢ₛ
        wpInstr 0 0 matmulLoopLoadB
          (warpAt 0 0 ("body", 2) [lane] ∗ ((b2 ∗ regBNew) ∗ frame)) := by
    simpa [matmulLoopLoadB, b2, regBOld, regBNew, frame] using
      (wp_globalLoadBytesReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 1)) (dst := "bVal")
        (ty := .s32) (addrExpr := .reg "bPtr") (lane := lane)
        (offset := matmulBOffset params 2) (bytes := bBytes 2) (oldReg := oldB)
        (value := .s32 (b 2 params.col)) (frame := frame)
        (haddr := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            _hmemReg, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, _hacc,
            hrestAcc⟩
          rcases hrestAcc with ⟨_rA, _rRestA, _hcompA, _hequivA, _ha, hrestA⟩
          rcases hrestA with ⟨_rProd, _rRestProd, _hcompProd, _hequivProd, _hprod,
            hrestProd⟩
          rcases hrestProd with ⟨_rAPtr, _rRestAPtr, _hcompAPtr, _hequivAPtr, _haptr,
            hrestAPtr⟩
          rcases hrestAPtr with ⟨_rBPtr, _rRestBPtr, _hcompBPtr, _hequivBPtr, hbptr,
            _hrestBPtr⟩
          exact resolves_global_gaddr_of_eval
            (ctx := { cta := 0, warp := 0, lane := lane }) (ty := .s32)
            (expr := .reg "bPtr") (off := matmulBOffset params 2)
            (eval_reg_of_assertion hbptr))
        (hread := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
            hmemReg, _hframe⟩
          rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
            hbytes, _hdst⟩
          exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
        (hframe := by
          apply CSL.stable_sepList
          intro q hq
          simp [frame, regK, regAcc, regA, regProd, regAPtr, regBPtr, predP,
            a0, a1, a2, b0, b1, out] at hq
          rcases hq with hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq | hq
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_reg_load_of_ne (by decide)
          · subst q; exact stable_pred_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load
          · subst q; exact stable_globalBytes_load))
  exact CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule))

theorem matmulLoop_load_b_wp_at
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (oldB prod : Value) (k : Nat) (hlt : k < matmulLoopProofN)
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col))) :
    matmulLoopAt lane ("body", 1) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k)) oldB prod
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstr 0 0 matmulLoopLoadB
        (matmulLoopAt lane ("body", 2) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) (.s32 (a params.row k))
          (.s32 (b k params.col)) prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
  unfold matmulLoopProofN at hlt
  cases k with
  | zero =>
      simpa [matmulLoopProofN] using
        matmulLoop_load_b_wp_at_zero lane params a b aBytes bBytes oldOut oldB prod
          (haccessB 0 (by simp [matmulLoopProofN]))
          (hwidthB 0 (by simp [matmulLoopProofN]))
          (hdecodeB 0 (by simp [matmulLoopProofN]))
  | succ k =>
      cases k with
      | zero =>
          simpa [matmulLoopProofN] using
            matmulLoop_load_b_wp_at_one lane params a b aBytes bBytes oldOut oldB prod
              (haccessB 1 (by simp [matmulLoopProofN]))
              (hwidthB 1 (by simp [matmulLoopProofN]))
              (hdecodeB 1 (by simp [matmulLoopProofN]))
      | succ k =>
          cases k with
          | zero =>
              simpa [matmulLoopProofN] using
                matmulLoop_load_b_wp_at_two lane params a b aBytes bBytes oldOut oldB prod
                  (haccessB 2 (by simp [matmulLoopProofN]))
                  (hwidthB 2 (by simp [matmulLoopProofN]))
                  (hdecodeB 2 (by simp [matmulLoopProofN]))
          | succ k =>
              omega

theorem matmulLoop_body_instrs_wp_at
    (lane : LaneId) (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte)
    (oldOut : List Byte) (oldA oldB oldProd : Value) (k : Nat)
    (hlt : k < matmulLoopProofN)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col))) :
    matmulLoopAt lane ("body", 0) (Int.ofNat k)
        (matmulLoopPrefixS32 params a b k) oldA oldB oldProd
        (.gaddr .global (matmulAOffset params k))
        (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
      wpInstrs 0 0 matmulLoopBodyBlock.body
        (matmulLoopBodyTermPre lane params a b aBytes bBytes oldOut) := by
  have hloadA :=
    matmulLoop_load_a_wp_at lane params a b aBytes bBytes oldOut oldA oldB oldProd
      k hlt haccessA hwidthA hdecodeA
  have hloadB :=
    matmulLoop_load_b_wp_at lane params a b aBytes bBytes oldOut oldB oldProd
      k hlt haccessB hwidthB hdecodeB
  have hmul :=
    matmulLoop_mul_wp_at lane params a b aBytes bBytes oldOut k oldProd
  have hadd :=
    matmulLoop_add_acc_wp_at lane params a b aBytes bBytes oldOut k
  have hincK :=
    matmulLoop_inc_k_wp_at lane params a b aBytes bBytes oldOut k hlt
  have hincAPtr :=
    matmulLoop_inc_aPtr_wp_at lane params a b aBytes bBytes oldOut k
  have hincBPtr :=
    matmulLoop_inc_bPtr_wp_at lane params hparamsN a b aBytes bBytes oldOut k
  have hpost :
      matmulLoopAt lane ("body", 7) (Int.ofNat (k + 1))
          (matmulLoopPrefixS32 params a b (k + 1)) (.s32 (a params.row k))
          (.s32 (b k params.col)) (.s32 (matmulLoopTermS32 params a b k))
          (.gaddr .global (matmulAOffset params (k + 1)))
          (.gaddr .global (matmulBOffset params (k + 1))) true params aBytes bBytes oldOut ⊢ₛ
        matmulLoopBodyTermPre lane params a b aBytes bBytes oldOut := by
    intro st r hat
    exact ⟨k + 1, .s32 (a params.row k), .s32 (b k params.col),
      .s32 (matmulLoopTermS32 params a b k), Nat.succ_le_of_lt hlt, hat⟩
  simpa [matmulLoopBodyBlock, wpInstrs, wpInstrList, matmulLoopLoadA, matmulLoopLoadB,
    matmulLoopMul, matmulLoopAddAcc, matmulLoopIncK, matmulLoopIncAPtr,
    matmulLoopIncBPtr] using
    CSL.entails_trans hloadA
      (wpInstr_mono <|
        CSL.entails_trans hloadB
          (wpInstr_mono <|
            CSL.entails_trans hmul
              (wpInstr_mono <|
                CSL.entails_trans hadd
                  (wpInstr_mono <|
                    CSL.entails_trans hincK
                      (wpInstr_mono <|
                        CSL.entails_trans hincAPtr
                          (wpInstr_mono <|
                            CSL.entails_trans hincBPtr (wpInstr_mono hpost)))))))

theorem matmulLoop_body_instrs_wp
    (lane : LaneId) (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte) (oldOut : List Byte)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col))) :
    matmulLoopBodyInv lane params a b aBytes bBytes oldOut ⊢ₛ
      wpInstrs 0 0 matmulLoopBodyBlock.body
        (matmulLoopBodyTermPre lane params a b aBytes bBytes oldOut) := by
  intro st r hpre
  rcases hpre with ⟨k, oldA, oldB, oldProd, hlt, hat⟩
  exact matmulLoop_body_instrs_wp_at lane params hparamsN a b aBytes bBytes oldOut
    oldA oldB oldProd k hlt haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB st r hat

theorem matmulLoop_exit_instrs_wp
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (haccessOut : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointAOut :
      ∀ i, i < matmulLoopProofN →
        ByteRangesDisjoint (matmulAOffset params i) (aBytes i).length
          (matmulCellOffset params) newOut.length)
    (hdisjointBOut :
      ∀ i, i < matmulLoopProofN →
        ByteRangesDisjoint (matmulBOffset params i) (bBytes i).length
          (matmulCellOffset params) newOut.length) :
    matmulLoopExitInv lane params a b aBytes bBytes oldOut ⊢ₛ
      wpInstrs 0 0 (matmulLoopExitBlock params).body
        (matmulLoopExitTermPre lane params a b aBytes bBytes newOut) := by
  intro st r hpre
  rcases hpre with ⟨aVal, bVal, prod, hat⟩
  let regK := CSL.reg 0 0 lane "k" (.s32 (Int.ofNat matmulLoopProofN))
  let regAcc := CSL.reg 0 0 lane "acc" (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN))
  let regA := CSL.reg 0 0 lane "aVal" aVal
  let regB := CSL.reg 0 0 lane "bVal" bVal
  let regProd := CSL.reg 0 0 lane "prod" prod
  let regAPtr := CSL.reg 0 0 lane "aPtr" (.gaddr .global (matmulAOffset params matmulLoopProofN))
  let regBPtr := CSL.reg 0 0 lane "bPtr" (.gaddr .global (matmulBOffset params matmulLoopProofN))
  let predP := CSL.pred 0 0 lane "p" false
  let a0 := CSL.globalBytes (matmulAOffset params 0) .read (aBytes 0)
  let a1 := CSL.globalBytes (matmulAOffset params 1) .read (aBytes 1)
  let a2 := CSL.globalBytes (matmulAOffset params 2) .read (aBytes 2)
  let b0 := CSL.globalBytes (matmulBOffset params 0) .read (bBytes 0)
  let b1 := CSL.globalBytes (matmulBOffset params 1) .read (bBytes 1)
  let b2 := CSL.globalBytes (matmulBOffset params 2) .read (bBytes 2)
  let outOld := CSL.globalBytes (matmulCellOffset params) .write oldOut
  let outNew := CSL.globalBytes (matmulCellOffset params) .write newOut
  let frame := CSL.sepList [regK, regAcc, regA, regB, regProd, regAPtr, regBPtr,
    predP, a0, a1, a2, b0, b1, b2]
  have hpreToRule :
      matmulLoopAt lane ("exit", 0) (Int.ofNat matmulLoopProofN)
          (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
          (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
          params aBytes bBytes oldOut ⊢ₛ
        (warpAt 0 0 ("exit", 0) [lane] ∗ (outOld ∗ frame)) := by
    exact CSL.entails_trans
      (matmulLoopAt_to_flat lane ("exit", 0) (Int.ofNat matmulLoopProofN)
        (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
        (.gaddr .global (matmulAOffset params matmulLoopProofN))
        (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
        params aBytes bBytes oldOut)
      (by
        simpa [matmulLoopFlatResources, regK, regAcc, regA, regB, regProd, regAPtr,
          regBPtr, predP, a0, a1, a2, b0, b1, b2, outOld, frame] using
          CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("exit", 0) [lane]))
            (CSL.sepList_perm_to_cons
              [regK, regAcc, regA, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2, outOld]
              outOld regK
              [regAcc, regA, regB, regProd, regAPtr, regBPtr, predP,
                a0, a1, a2, b0, b1, b2]
              (perm_matmul_flat_out_front
                regK regAcc regA regB regProd regAPtr regBPtr predP
                a0 a1 a2 b0 b1 b2 outOld)))
  have hpostFromRule :
      (warpAt 0 0 ("exit", 1) [lane] ∗ (outNew ∗ frame)) ⊢ₛ
        matmulLoopExitTermPre lane params a b aBytes bBytes newOut := by
    intro st r hraw
    refine ⟨aVal, bVal, prod, ?_⟩
    have hflat :
        (warpAt 0 0 ("exit", 1) [lane] ∗
          matmulLoopFlatResources lane (Int.ofNat matmulLoopProofN)
            (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
            (.gaddr .global (matmulAOffset params matmulLoopProofN))
            (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
            params aBytes bBytes newOut) st r := by
      simpa [matmulLoopFlatResources, regK, regAcc, regA, regB, regProd, regAPtr,
        regBPtr, predP, a0, a1, a2, b0, b1, b2, outNew, frame] using
        CSL.sep_mono (CSL.entails_refl (warpAt 0 0 ("exit", 1) [lane]))
          (CSL.sep_cons_perm_to_sepList
            outNew regK
            [regAcc, regA, regB, regProd, regAPtr, regBPtr, predP,
              a0, a1, a2, b0, b1, b2]
            [regK, regAcc, regA, regB, regProd, regAPtr, regBPtr, predP,
              a0, a1, a2, b0, b1, b2, outNew]
            (perm_matmul_flat_out_front
              regK regAcc regA regB regProd regAPtr regBPtr predP
              a0 a1 a2 b0 b1 b2 outNew).symm) st r hraw
    exact (matmulLoopAt_flat_to_standard lane ("exit", 1)
      (Int.ofNat matmulLoopProofN) (matmulLoopPrefixS32 params a b matmulLoopProofN)
      aVal bVal prod (.gaddr .global (matmulAOffset params matmulLoopProofN))
      (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
      params aBytes bBytes newOut) st r hflat
  have hrule :
      (warpAt 0 0 ("exit", 0) [lane] ∗ (outOld ∗ frame)) ⊢ₛ
        wpInstr 0 0 (matmulLoopStore params)
          (warpAt 0 0 ("exit", 1) [lane] ∗ (outNew ∗ frame)) := by
    simpa [matmulLoopStore, outOld, outNew, frame] using
      (wp_globalStoreBytes_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("exit", 0))
        (ty := .s32) (addrExpr := matmulGlobalAddr (matmulCellOffset params))
        (valueExpr := .reg "acc") (lane := lane) (offset := matmulCellOffset params)
        (oldBytes := oldOut) (newBytes := newOut)
        (value := .s32 (matmulLoopPrefixS32 params a b matmulLoopProofN))
        (frame := frame)
        (haddr := by
          intro st r _hpre
          exact resolves_global_gaddr_of_eval (st := st)
            (ctx := { cta := 0, warp := 0, lane := lane })
            (ty := .s32) (expr := matmulGlobalAddr (matmulCellOffset params))
            (off := matmulCellOffset params) (by rfl))
        (heval := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
          dsimp [frame] at hframe
          rcases hframe with ⟨_rK, _rRestK, _hcompK, _hequivK, _hk, hrestK⟩
          rcases hrestK with ⟨_rAcc, _rRestAcc, _hcompAcc, _hequivAcc, hacc,
            _hrestAcc⟩
          exact eval_reg_of_assertion hacc)
        (hwrite := by
          intro st r _hpre
          exact ⟨{ st with global := {
              bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } },
            globalWriteMem_of_byteWrite haccessOut hencodeOut (by rfl)⟩)
        (hencode := hencodeOut)
        (hlen := hlenOut)
        (hframe := by
          intro st st' r rFrame hpre hframeSt hstep
          have haddr' := resolves_global_gaddr_of_eval (st := st)
            (ctx := { cta := 0, warp := 0, lane := lane })
            (ty := .s32) (expr := matmulGlobalAddr (matmulCellOffset params))
            (off := matmulCellOffset params) (by rfl)
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
          dsimp [frame] at hframeSt ⊢
          rcases hframeSt with ⟨rK, rRestK, hcompK, hequivK, hk, hrestK⟩
          rcases hrestK with ⟨rAcc, rRestAcc, hcompAcc, hequivAcc, hacc, hrestAcc⟩
          rcases hrestAcc with ⟨rA, rRestA, hcompA, hequivA, haReg, hrestA⟩
          rcases hrestA with ⟨rB, rRestB, hcompB, hequivB, hbReg, hrestB⟩
          rcases hrestB with ⟨rProd, rRestProd, hcompProd, hequivProd, hprod, hrestProd⟩
          rcases hrestProd with ⟨rAPtr, rRestAPtr, hcompAPtr, hequivAPtr, haptr,
            hrestAPtr⟩
          rcases hrestAPtr with ⟨rBPtr, rRestBPtr, hcompBPtr, hequivBPtr, hbptr,
            hrestBPtr⟩
          rcases hrestBPtr with ⟨rPred, rRestPred, hcompPred, hequivPred, hp,
            hrestPred⟩
          rcases hrestPred with ⟨rA0, rRestA0, hcompA0, hequivA0, ha0, hrestA0⟩
          rcases hrestA0 with ⟨rA1, rRestA1, hcompA1, hequivA1, ha1, hrestA1⟩
          rcases hrestA1 with ⟨rA2, rRestA2, hcompA2, hequivA2, ha2, hrestA2⟩
          rcases hrestA2 with ⟨rB0, rRestB0, hcompB0, hequivB0, hb0, hrestB0⟩
          rcases hrestB0 with ⟨rB1, rB2, hcompB1, hequivB1, hb1, hb2⟩
          have heval' :
              EvalRValue st { cta := 0, warp := 0, lane := lane } (.reg "acc")
                (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) :=
            eval_reg_of_assertion hacc
          have hwrite' :
              WriteMemFact st .global .s32 (.global (matmulCellOffset params))
                (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN))
                { st with global := {
                    bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } } :=
            globalWriteMem_of_byteWrite haccessOut hencodeOut (by rfl)
          refine ⟨rK, rRestK, hcompK, hequivK, ?_, ?_⟩
          · exact globalStorePreservesReadReg_single_warpAt
              hctrl hk haddr' heval' hwrite' hstep
          · refine ⟨rAcc, rRestAcc, hcompAcc, hequivAcc, ?_, ?_⟩
            · exact globalStorePreservesReadReg_single_warpAt
                hctrl hacc haddr' heval' hwrite' hstep
            · refine ⟨rA, rRestA, hcompA, hequivA, ?_, ?_⟩
              · exact globalStorePreservesReadReg_single_warpAt
                  hctrl haReg haddr' heval' hwrite' hstep
              · refine ⟨rB, rRestB, hcompB, hequivB, ?_, ?_⟩
                · exact globalStorePreservesReadReg_single_warpAt
                    hctrl hbReg haddr' heval' hwrite' hstep
                · refine ⟨rProd, rRestProd, hcompProd, hequivProd, ?_, ?_⟩
                  · exact globalStorePreservesReadReg_single_warpAt
                      hctrl hprod haddr' heval' hwrite' hstep
                  · refine ⟨rAPtr, rRestAPtr, hcompAPtr, hequivAPtr, ?_, ?_⟩
                    · exact globalStorePreservesReadReg_single_warpAt
                        hctrl haptr haddr' heval' hwrite' hstep
                    · refine ⟨rBPtr, rRestBPtr, hcompBPtr, hequivBPtr, ?_, ?_⟩
                      · exact globalStorePreservesReadReg_single_warpAt
                          hctrl hbptr haddr' heval' hwrite' hstep
                      · refine ⟨rPred, rRestPred, hcompPred, hequivPred, ?_, ?_⟩
                        · exact globalStorePreservesPred_single_warpAt
                            hctrl hp haddr' heval' hwrite' hstep
                        · refine ⟨rA0, rRestA0, hcompA0, hequivA0, ?_, ?_⟩
                          · exact globalStorePreservesGlobalBytes_single_warpAt
                              (by simpa [matmulLoopProofN] using
                                hdisjointAOut 0 (by simp [matmulLoopProofN]))
                              hctrl ha0 haddr' heval' hwrite' hencodeOut hstep
                          · refine ⟨rA1, rRestA1, hcompA1, hequivA1, ?_, ?_⟩
                            · exact globalStorePreservesGlobalBytes_single_warpAt
                                (by simpa [matmulLoopProofN] using
                                  hdisjointAOut 1 (by simp [matmulLoopProofN]))
                                hctrl ha1 haddr' heval' hwrite' hencodeOut hstep
                            · refine ⟨rA2, rRestA2, hcompA2, hequivA2, ?_, ?_⟩
                              · exact globalStorePreservesGlobalBytes_single_warpAt
                                  (by simpa [matmulLoopProofN] using
                                    hdisjointAOut 2 (by simp [matmulLoopProofN]))
                                  hctrl ha2 haddr' heval' hwrite' hencodeOut hstep
                              · refine ⟨rB0, rRestB0, hcompB0, hequivB0, ?_, ?_⟩
                                · exact globalStorePreservesGlobalBytes_single_warpAt
                                    (by simpa [matmulLoopProofN] using
                                      hdisjointBOut 0 (by simp [matmulLoopProofN]))
                                    hctrl hb0 haddr' heval' hwrite' hencodeOut hstep
                                · refine ⟨rB1, rB2, hcompB1, hequivB1, ?_, ?_⟩
                                  · exact globalStorePreservesGlobalBytes_single_warpAt
                                      (by simpa [matmulLoopProofN] using
                                        hdisjointBOut 1 (by simp [matmulLoopProofN]))
                                      hctrl hb1 haddr' heval' hwrite' hencodeOut hstep
                                  · exact globalStorePreservesGlobalBytes_single_warpAt
                                      (by simpa [matmulLoopProofN] using
                                        hdisjointBOut 2 (by simp [matmulLoopProofN]))
                                      hctrl hb2 haddr' heval' hwrite' hencodeOut hstep))
  simpa [matmulLoopExitBlock, wpInstrs, wpInstrList, matmulLoopStore] using
    (CSL.entails_trans hpreToRule (CSL.entails_trans hrule (wpInstr_mono hpostFromRule)) st r hat)

theorem matmulLoop_body_term_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut) "body" (.br "loop")
      (matmulLoopBodyTermPre lane params a b aBytes bBytes oldOut) := by
  refine TerminatorVC.br ?_
  intro st r hpre
  rcases hpre with ⟨k, aVal, bVal, prod, hkLe, hat⟩
  have hbr :
      matmulLoopAt lane ("body", 7) (Int.ofNat k)
          (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut ⊢ₛ
        wpTerminator 0 0 (.br "loop")
          (matmulLoopAt lane ("loop", 0) (Int.ofNat k)
            (matmulLoopPrefixS32 params a b k) aVal bVal prod
            (.gaddr .global (matmulAOffset params k))
            (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut) := by
    simpa [matmulLoopAt] using
      (wp_br_lanes_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := ("body", 7)) (target := "loop")
        (lanes := [lane])
        (frame :=
          matmulLoopRegs lane (Int.ofNat k) (matmulLoopPrefixS32 params a b k)
            aVal bVal prod (.gaddr .global (matmulAOffset params k))
            (.gaddr .global (matmulBOffset params k)) true ∗
            matmulLoopFrame params aBytes bBytes oldOut)
        (matmulLoopAt_stable_terminator_frame (.br "loop") lane
          (Int.ofNat k) (matmulLoopPrefixS32 params a b k) aVal bVal prod
          (.gaddr .global (matmulAOffset params k))
          (.gaddr .global (matmulBOffset params k)) true params aBytes bBytes oldOut))
  exact (CSL.entails_trans hbr <|
    wpTerminator_mono (by
      intro st' r' hloop
      exact ⟨k, aVal, bVal, prod, true, hkLe, hloop⟩)) st r hat

theorem matmulLoop_exit_term_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    TerminatorVC 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut) "exit" .terminate
      (matmulLoopExitTermPre lane params a b aBytes bBytes newOut) := by
  refine TerminatorVC.terminate ?_
  intro st r hpre
  rcases hpre with ⟨aVal, bVal, prod, hat⟩
  have hterm :
      matmulLoopAt lane ("exit", 1) (Int.ofNat matmulLoopProofN)
          (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
          (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
          params aBytes bBytes newOut ⊢ₛ
        wpTerminator 0 0 .terminate
          ((laneTerminatedAt 0 0 lane ("exit", 1)) ∗
            (matmulLoopRegs lane (Int.ofNat matmulLoopProofN)
              (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
              (.gaddr .global (matmulAOffset params matmulLoopProofN))
              (.gaddr .global (matmulBOffset params matmulLoopProofN)) false ∗
              matmulLoopFrame params aBytes bBytes newOut)) := by
    simpa [matmulLoopAt] using
      (wp_terminate_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("exit", 1)) (lane := lane)
        (frame :=
          matmulLoopRegs lane (Int.ofNat matmulLoopProofN)
            (matmulLoopPrefixS32 params a b matmulLoopProofN) aVal bVal prod
            (.gaddr .global (matmulAOffset params matmulLoopProofN))
            (.gaddr .global (matmulBOffset params matmulLoopProofN)) false ∗
            matmulLoopFrame params aBytes bBytes newOut)
        (matmulLoopAt_stable_terminator_frame .terminate lane
          (Int.ofNat matmulLoopProofN) (matmulLoopPrefixS32 params a b matmulLoopProofN)
          aVal bVal prod (.gaddr .global (matmulAOffset params matmulLoopProofN))
          (.gaddr .global (matmulBOffset params matmulLoopProofN)) false
          params aBytes bBytes newOut))
  exact (CSL.entails_trans hterm <|
    wpTerminator_mono (by
      intro st' r' hpost
      exact ⟨aVal, bVal, prod, hpost⟩)) st r hat

theorem matmulLoop_entry_block_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    blockVC' 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut)
      "entry" (matmulLoopEntryBlock params) := by
  refine ⟨matmulLoopEntryTermPre lane params aBytes bBytes oldOut oldA oldB oldProd oldP, ?_, ?_⟩
  · simpa [matmulLoopInvariants, matmulLoopEntryBlock, matmulLoopProofN] using
      matmulLoop_entry_instrs_wp lane params aBytes bBytes oldOut oldK oldAcc oldA oldB
        oldProd oldAPtr oldBPtr oldP
  · exact matmulLoop_entry_term_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP

theorem matmulLoop_header_block_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool) :
    blockVC' 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut)
      "loop" matmulLoopHeaderBlock := by
  refine ⟨matmulLoopHeaderTermPre lane params a b aBytes bBytes oldOut, ?_, ?_⟩
  · simpa [matmulLoopInvariants, matmulLoopHeaderBlock, matmulLoopProofN] using
      matmulLoop_header_instrs_wp lane params a b aBytes bBytes oldOut
  · exact matmulLoop_header_term_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP

theorem matmulLoop_body_block_vc
    (lane : LaneId) (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte)
    (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col))) :
    blockVC' 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut)
      "body" matmulLoopBodyBlock := by
  refine ⟨matmulLoopBodyTermPre lane params a b aBytes bBytes oldOut, ?_, ?_⟩
  · simpa [matmulLoopInvariants, matmulLoopBodyBlock, matmulLoopProofN] using
      matmulLoop_body_instrs_wp lane params hparamsN a b aBytes bBytes oldOut
        haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB
  · exact matmulLoop_body_term_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP

theorem matmulLoop_exit_block_vc
    (lane : LaneId) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (aBytes bBytes : Nat → List Byte) (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool)
    (haccessOut : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointAOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulAOffset params j) (aBytes j).length
          (matmulCellOffset params) newOut.length)
    (hdisjointBOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulBOffset params j) (bBytes j).length
          (matmulCellOffset params) newOut.length) :
    blockVC' 0 0
      (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
      (matmulLoopPost lane params a b aBytes bBytes newOut)
      "exit" (matmulLoopExitBlock params) := by
  refine ⟨matmulLoopExitTermPre lane params a b aBytes bBytes newOut, ?_, ?_⟩
  · simpa [matmulLoopInvariants, matmulLoopExitBlock, matmulLoopProofN] using
      matmulLoop_exit_instrs_wp lane params a b aBytes bBytes oldOut newOut
        haccessOut hencodeOut hlenOut hdisjointAOut hdisjointBOut
  · exact matmulLoop_exit_term_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP

theorem matmulLoop_block_vcs
    (lane : LaneId) (params : MatmulCellParams) (hparamsN : params.n = matmulLoopProofN)
    (a b : Nat → Nat → Int) (aBytes bBytes : Nat → List Byte)
    (oldOut newOut : List Byte)
    (oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value) (oldP : Bool)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col)))
    (haccessOut : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointAOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulAOffset params j) (aBytes j).length
          (matmulCellOffset params) newOut.length)
    (hdisjointBOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulBOffset params j) (bBytes j).length
          (matmulCellOffset params) newOut.length) :
    ∀ label block,
      (matmulLoopEnv params).blocks[label]? = some block →
        blockVC' 0 0
          (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
            oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
          (matmulLoopPost lane params a b aBytes bBytes newOut) label block := by
  intro label block hlookup
  rcases matmulLoop_block_lookup hlookup with
    hentry | hloop | hbody | hexit
  · rcases hentry with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact matmulLoop_entry_block_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP
  · rcases hloop with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact matmulLoop_header_block_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP
  · rcases hbody with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact matmulLoop_body_block_vc lane params hparamsN a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP
      haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB
  · rcases hexit with ⟨hlabel, hblock⟩
    subst label
    subst block
    exact matmulLoop_exit_block_vc lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP
      haccessOut hencodeOut hlenOut hdisjointAOut hdisjointBOut

theorem matmul_cell_loop_kernel_valid
    {init : State} {resource : CSL.Resource} {lane : LaneId}
    {params : MatmulCellParams} {a b : Nat → Nat → Int}
    {aBytes bBytes : Nat → List Byte} {oldOut newOut : List Byte}
    {oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value} {oldP : Bool}
    (hinit :
      (matmulLoopKernelSpec init resource lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP).pre init resource)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col)))
    (haccessOut : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointAOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulAOffset params j) (aBytes j).length
          (matmulCellOffset params) newOut.length)
    (hdisjointBOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulBOffset params j) (bBytes j).length
          (matmulCellOffset params) newOut.length)
    (honly :
      cfgKernelInvariant' (matmulLoopEnv params) 0 0
        (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
          oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
        (matmulLoopPost lane params a b aBytes bBytes newOut) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (matmulLoopEnv params) 0 0)
    (hpostNoStep :
      NoStepBlock 0 0 (matmulLoopPost lane params a b aBytes bBytes newOut))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (matmulLoopEnv params) 0 0
          (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
            oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
          (matmulLoopPost lane params a b aBytes bBytes newOut))) :
    (matmulLoopKernelSpec init resource lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP).Valid := by
  have hparamsN : params.n = matmulLoopProofN := hinit.1
  exact KernelSpec.Valid.of_entry_blockVCs'_closed
    (spec := matmulLoopKernelSpec init resource lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
    (env := matmulLoopEnv params) (cta := 0) (warp := 0)
    (invariants := matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
      oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r hpre
      exact hpre.2.2)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨_hparamsN, henv, hentry⟩
      rcases hentry with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
      rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
      exact ⟨warpState, matmulLoopEntryBlock params, henv, hwarp, hlock,
        by simpa [matmulLoopEnv] using hrpc,
        by simpa [matmulLoopEnv] using matmulLoop_entry_lookup params⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics
      (matmulLoop_body_ordinary params))
    (hblocks :=
      matmulLoop_block_vcs lane params hparamsN a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP
        haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB
        haccessOut hencodeOut hlenOut hdisjointAOut hdisjointBOut)
    (hbr := hbr)
    (htargets := matmulLoop_targets_exist params)
    (hpostNoStep := hpostNoStep)
    (hsuffixNoFinal := hsuffixNoFinal)

theorem matmul_cell_loop_partial_correct
    {init final : State} {resource : CSL.Resource} {lane : LaneId}
    {params : MatmulCellParams} {a b : Nat → Nat → Int}
    {aBytes bBytes : Nat → List Byte} {oldOut newOut : List Byte}
    {oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr : Value} {oldP : Bool}
    (hinit :
      (matmulLoopKernelSpec init resource lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP).pre init resource)
    (haccessA :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulAOffset params j)))
    (hwidthA :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (aBytes j).length)
    (hdecodeA :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (aBytes j) (.s32 (a params.row j)))
    (haccessB :
      ∀ j, j < matmulLoopProofN → AccessOk .global .s32 (.global (matmulBOffset params j)))
    (hwidthB :
      ∀ j, j < matmulLoopProofN → Typing.byteWidth? .s32 = some (bBytes j).length)
    (hdecodeB :
      ∀ j, j < matmulLoopProofN → DecodedScalar .s32 (bBytes j) (.s32 (b j params.col)))
    (haccessOut : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeOut :
      EncodedScalar .s32 (.s32 (matmulLoopPrefixS32 params a b matmulLoopProofN)) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (hdisjointAOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulAOffset params j) (aBytes j).length
          (matmulCellOffset params) newOut.length)
    (hdisjointBOut :
      ∀ j, j < matmulLoopProofN →
        ByteRangesDisjoint (matmulBOffset params j) (bBytes j).length
          (matmulCellOffset params) newOut.length)
    (honly :
      cfgKernelInvariant' (matmulLoopEnv params) 0 0
        (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
          oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
        (matmulLoopPost lane params a b aBytes bBytes newOut) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hbr : BrTermControl (matmulLoopEnv params) 0 0)
    (hpostNoStep :
      NoStepBlock 0 0 (matmulLoopPost lane params a b aBytes bBytes newOut))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant' (matmulLoopEnv params) 0 0
          (matmulLoopInvariants lane params a b aBytes bBytes oldOut newOut
            oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP)
          (matmulLoopPost lane params a b aBytes bBytes newOut)))
    (hterm : TerminatesAt init final) :
    ∃ r, matmulLoopPost lane params a b aBytes bBytes newOut final r := by
  have hvalid :
      (matmulLoopKernelSpec init resource lane params a b aBytes bBytes oldOut newOut
        oldK oldAcc oldA oldB oldProd oldAPtr oldBPtr oldP).Valid :=
    matmul_cell_loop_kernel_valid hinit
      haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB
      haccessOut hencodeOut hlenOut hdisjointAOut hdisjointBOut
      honly hbr hpostNoStep hsuffixNoFinal
  exact KernelSpec.partial_correct hvalid final hterm

end Examples
end CLean
