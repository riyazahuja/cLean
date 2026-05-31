import CLean.Semantics.Helpers
import CLean.CSL

namespace CLean
namespace WP

structure LaneCtx where
  cta : CTAId
  warp : WarpId
  lane : LaneId
  deriving Repr, Inhabited

def EvalRValue (st : State) (ctx : LaneCtx) (expr : RValue) (value : Value) : Prop :=
  Helpers.evalRValue? st ctx.cta ctx.warp ctx.lane expr = some value

def EvalCmp (st : State) (ctx : LaneCtx) (cmp : CmpExpr) (value : Bool) : Prop :=
  Helpers.evalCmp? st ctx.cta ctx.warp ctx.lane cmp = some value

def ResolvesAddr (st : State) (ctx : LaneCtx) (addr : TypedAddr) (resolved : Addr) : Prop :=
  Helpers.resolveAddr? st ctx.cta ctx.warp ctx.lane addr = some resolved

def AccessOk (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Prop :=
  Typing.typedAccessPreconditions? space ty addr = true

theorem eval_rvalue_of_eq
    {st : State} {ctx : LaneCtx} {expr : RValue} {value : Value}
    (h : Helpers.evalRValue? st ctx.cta ctx.warp ctx.lane expr = some value) :
    EvalRValue st ctx expr value :=
  h

theorem eval_imm {st : State} {ctx : LaneCtx} {value : Value} :
    EvalRValue st ctx (.imm value) value := by
  rfl

theorem eval_reg_of_assertion
    {st : State} {ctx : LaneCtx} {r : CSL.Resource} {name : RegName} {value : Value}
    (hreg : CSL.reg ctx.cta ctx.warp ctx.lane name value st r) :
    EvalRValue st ctx (.reg name) value := by
  rcases hreg with ⟨_, laneState, hget, hread⟩
  unfold EvalRValue
  simp [Helpers.evalRValue?, Helpers.readReg, hget, hread]

theorem eval_pred_of_assertion
    {st : State} {ctx : LaneCtx} {r : CSL.Resource} {name : PredName} {value : Bool}
    (hpred : CSL.pred ctx.cta ctx.warp ctx.lane name value st r) :
    EvalRValue st ctx (.pred name) (.pred value) := by
  rcases hpred with ⟨_, laneState, hget, hread⟩
  unfold EvalRValue
  simp [Helpers.evalRValue?, Helpers.readPred, hget, hread]

theorem eval_special {st : State} {ctx : LaneCtx} {special : SpecialReg} :
    EvalRValue st ctx (.special special)
      (Helpers.evalSpecial st.kernelEnv.gridCtx ctx.cta ctx.warp ctx.lane special) := by
  rfl

theorem eval_unop_of_eval
    {st : State} {ctx : LaneCtx} {op : ScalarUnaryOp} {arg : RValue}
    {argValue value : Value}
    (harg : EvalRValue st ctx arg argValue)
    (hop : Helpers.evalUnary? op argValue = some value) :
    EvalRValue st ctx (.unop op arg) value := by
  unfold EvalRValue at harg ⊢
  simp [Helpers.evalRValue?, harg, hop]

theorem eval_binop_of_eval
    {st : State} {ctx : LaneCtx} {op : ScalarBinaryOp} {lhs rhs : RValue}
    {lhsValue rhsValue value : Value}
    (hlhs : EvalRValue st ctx lhs lhsValue)
    (hrhs : EvalRValue st ctx rhs rhsValue)
    (hop : Helpers.evalBinary? op lhsValue rhsValue = some value) :
    EvalRValue st ctx (.binop op lhs rhs) value := by
  unfold EvalRValue at hlhs hrhs ⊢
  simp [Helpers.evalRValue?, hlhs, hrhs, hop]

theorem eval_triop_of_eval
    {st : State} {ctx : LaneCtx} {op : ScalarTernaryOp} {a b c : RValue}
    {aValue bValue cValue value : Value}
    (ha : EvalRValue st ctx a aValue)
    (hb : EvalRValue st ctx b bValue)
    (hc : EvalRValue st ctx c cValue)
    (hop : Helpers.evalTernary? op aValue bValue cValue = some value) :
    EvalRValue st ctx (.triop op a b c) value := by
  unfold EvalRValue at ha hb hc ⊢
  simp [Helpers.evalRValue?, ha, hb, hc, hop]

theorem eval_cmp_of_eq
    {st : State} {ctx : LaneCtx} {cmp : CmpExpr} {value : Bool}
    (h : Helpers.evalCmp? st ctx.cta ctx.warp ctx.lane cmp = some value) :
    EvalCmp st ctx cmp value :=
  h

namespace EvalRValue

theorem binop_add_s32
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : Int}
    (hlhs : EvalRValue st ctx lhs (.s32 a))
    (hrhs : EvalRValue st ctx rhs (.s32 b)) :
    EvalRValue st ctx (.binop .add lhs rhs)
      (.s32 (Helpers.normalizeSigned 32 (a + b))) :=
  eval_binop_of_eval hlhs hrhs rfl

theorem binop_mul_s32
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : Int}
    (hlhs : EvalRValue st ctx lhs (.s32 a))
    (hrhs : EvalRValue st ctx rhs (.s32 b)) :
    EvalRValue st ctx (.binop .mul lhs rhs)
      (.s32 (Helpers.normalizeSigned 32 (a * b))) :=
  eval_binop_of_eval hlhs hrhs rfl

theorem binop_add_gaddr_u64
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {space : AddrSpace}
    {offset : Nat} {delta : UInt64}
    (hlhs : EvalRValue st ctx lhs (.gaddr space offset))
    (hrhs : EvalRValue st ctx rhs (.u64 delta)) :
    EvalRValue st ctx (.binop .add lhs rhs) (.gaddr space (offset + delta.toNat)) :=
  eval_binop_of_eval hlhs hrhs rfl

end EvalRValue

namespace EvalCmp

theorem lt_u64
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : UInt64}
    (hlhs : EvalRValue st ctx lhs (.u64 a))
    (hrhs : EvalRValue st ctx rhs (.u64 b)) :
    EvalCmp st ctx { op := .lt, lhs := lhs, rhs := rhs } (decide (a < b)) := by
  unfold EvalRValue at hlhs hrhs
  unfold CLean.WP.EvalCmp Helpers.evalCmp?
  rw [hlhs, hrhs]
  rfl

theorem ge_u64
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : UInt64}
    (hlhs : EvalRValue st ctx lhs (.u64 a))
    (hrhs : EvalRValue st ctx rhs (.u64 b)) :
    EvalCmp st ctx { op := .ge, lhs := lhs, rhs := rhs } (decide (a ≥ b)) := by
  unfold EvalRValue at hlhs hrhs
  unfold CLean.WP.EvalCmp Helpers.evalCmp?
  rw [hlhs, hrhs]
  rfl

theorem lt_s32
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : Int}
    (hlhs : EvalRValue st ctx lhs (.s32 a))
    (hrhs : EvalRValue st ctx rhs (.s32 b)) :
    EvalCmp st ctx { op := .lt, lhs := lhs, rhs := rhs } (decide (a < b)) := by
  unfold EvalRValue at hlhs hrhs
  unfold CLean.WP.EvalCmp Helpers.evalCmp?
  rw [hlhs, hrhs]
  rfl

theorem ge_s32
    {st : State} {ctx : LaneCtx} {lhs rhs : RValue} {a b : Int}
    (hlhs : EvalRValue st ctx lhs (.s32 a))
    (hrhs : EvalRValue st ctx rhs (.s32 b)) :
    EvalCmp st ctx { op := .ge, lhs := lhs, rhs := rhs } (decide (a ≥ b)) := by
  unfold EvalRValue at hlhs hrhs
  unfold CLean.WP.EvalCmp Helpers.evalCmp?
  rw [hlhs, hrhs]
  rfl

end EvalCmp

theorem resolves_addr_of_eq
    {st : State} {ctx : LaneCtx} {addr : TypedAddr} {resolved : Addr}
    (h : Helpers.resolveAddr? st ctx.cta ctx.warp ctx.lane addr = some resolved) :
    ResolvesAddr st ctx addr resolved :=
  h

end WP
end CLean
