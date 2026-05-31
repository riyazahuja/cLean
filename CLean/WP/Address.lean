import CLean.WP.Expr

namespace CLean
namespace WP

theorem resolves_global_u64_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt64}
    (heval : EvalRValue st ctx expr (.u64 off)) :
    ResolvesAddr st ctx { space := .global, ty := ty, addr := expr } (.global off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_global_u32_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt32}
    (heval : EvalRValue st ctx expr (.u32 off)) :
    ResolvesAddr st ctx { space := .global, ty := ty, addr := expr } (.global off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_global_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .global off)) :
    ResolvesAddr st ctx { space := .global, ty := ty, addr := expr } (.global off) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_shared_u64_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt64}
    (heval : EvalRValue st ctx expr (.u64 off)) :
    ResolvesAddr st ctx { space := .shared, ty := ty, addr := expr }
      (.shared ctx.cta off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_shared_u32_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt32}
    (heval : EvalRValue st ctx expr (.u32 off)) :
    ResolvesAddr st ctx { space := .shared, ty := ty, addr := expr }
      (.shared ctx.cta off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_shared_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .shared off)) :
    ResolvesAddr st ctx { space := .shared, ty := ty, addr := expr }
      (.shared ctx.cta off) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_local_u64_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt64}
    (heval : EvalRValue st ctx expr (.u64 off)) :
    ResolvesAddr st ctx { space := .local, ty := ty, addr := expr }
      (.local ctx.cta ctx.warp ctx.lane off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_local_u32_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt32}
    (heval : EvalRValue st ctx expr (.u32 off)) :
    ResolvesAddr st ctx { space := .local, ty := ty, addr := expr }
      (.local ctx.cta ctx.warp ctx.lane off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_local_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .local off)) :
    ResolvesAddr st ctx { space := .local, ty := ty, addr := expr }
      (.local ctx.cta ctx.warp ctx.lane off) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_param_u64_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt64}
    (heval : EvalRValue st ctx expr (.u64 off)) :
    ResolvesAddr st ctx { space := .param, ty := ty, addr := expr } (.param off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_param_u32_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt32}
    (heval : EvalRValue st ctx expr (.u32 off)) :
    ResolvesAddr st ctx { space := .param, ty := ty, addr := expr } (.param off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_param_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .param off)) :
    ResolvesAddr st ctx { space := .param, ty := ty, addr := expr } (.param off) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_const_u64_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt64}
    (heval : EvalRValue st ctx expr (.u64 off)) :
    ResolvesAddr st ctx { space := .const, ty := ty, addr := expr } (.const off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_const_u32_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : UInt32}
    (heval : EvalRValue st ctx expr (.u32 off)) :
    ResolvesAddr st ctx { space := .const, ty := ty, addr := expr } (.const off.toNat) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_const_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .const off)) :
    ResolvesAddr st ctx { space := .const, ty := ty, addr := expr } (.const off) := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval]

theorem resolves_generic_gaddr_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue}
    {space : AddrSpace} {off : Nat} {resolved : Addr}
    (heval : EvalRValue st ctx expr (.gaddr space off))
    (htag : Helpers.taggedGenericAddr? ctx.cta ctx.warp ctx.lane space off = some resolved) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr } resolved := by
  unfold EvalRValue at heval
  unfold ResolvesAddr
  simp [Helpers.resolveAddr?, heval, htag]

theorem resolves_generic_global_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .global off)) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr } (.global off) :=
  resolves_generic_gaddr_of_eval heval (by simp [Helpers.taggedGenericAddr?])

theorem resolves_generic_shared_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .shared off)) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr }
      (.shared ctx.cta off) :=
  resolves_generic_gaddr_of_eval heval (by simp [Helpers.taggedGenericAddr?])

theorem resolves_generic_local_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .local off)) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr }
      (.local ctx.cta ctx.warp ctx.lane off) :=
  resolves_generic_gaddr_of_eval heval (by simp [Helpers.taggedGenericAddr?])

theorem resolves_generic_param_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .param off)) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr } (.param off) :=
  resolves_generic_gaddr_of_eval heval (by simp [Helpers.taggedGenericAddr?])

theorem resolves_generic_const_of_eval
    {st : State} {ctx : LaneCtx} {ty : ScalarTy} {expr : RValue} {off : Nat}
    (heval : EvalRValue st ctx expr (.gaddr .const off)) :
    ResolvesAddr st ctx { space := .generic, ty := ty, addr := expr } (.const off) :=
  resolves_generic_gaddr_of_eval heval (by simp [Helpers.taggedGenericAddr?])

end WP
end CLean
