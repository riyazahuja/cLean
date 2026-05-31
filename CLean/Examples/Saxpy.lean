import CLean.WP

namespace CLean
namespace Examples

open CSL WP

private def readGlobalS32? (st : State) (off : Nat) : Option Int := do
  match Helpers.readMem? st .global .s32 (.global off) with
  | some (.s32 x) => some x
  | _ => none

structure SaxpyParams where
  n : Nat
  alpha : Int
  xBase : Nat
  yBase : Nat
  outBase : Nat
  deriving Repr, Inhabited

def saxpyLaneOutputOffset (params : SaxpyParams) (i : Nat) : Nat :=
  params.outBase + i * 4

def saxpyExpectedValue (params : SaxpyParams) (x y : Int) : Int :=
  params.alpha * x + y

def saxpyLaneByteOffset : RValue :=
  .unop (.cvt .u64)
    (.binop .mul (.special .tidX) (.imm (.u32 4)))

def saxpyGlobalLaneAddr (base : Nat) : RValue :=
  .binop .add (.imm (.gaddr .global base)) saxpyLaneByteOffset

def saxpyLoadX (params : SaxpyParams) : GInstr :=
  { guard? := none
    instr := .load "x"
      { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase } }

def saxpyLoadY (params : SaxpyParams) : GInstr :=
  { guard? := none
    instr := .load "y"
      { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase } }

def saxpyMul (params : SaxpyParams) : GInstr :=
  { guard? := none
    instr := .assignReg "prod"
      (.binop .mul (.imm (.s32 params.alpha)) (.reg "x")) }

def saxpyAdd : GInstr :=
  { guard? := none
    instr := .assignReg "sum"
      (.binop .add (.reg "prod") (.reg "y")) }

theorem saxpy_loadX_lanes_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset : Nat} {xBytes : List Byte} {oldX : Value} {x : Int}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (globalSlices [xOffset] .read [xBytes] ∗
            regsFor 0 0 [lane] "x" [oldX])) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccess : AccessOk .global .s32 (.global xOffset))
    (hwidth : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecode : DecodedScalar .s32 xBytes (.s32 x)) :
    (warpAt 0 0 pc [lane] ∗
      (globalSlices [xOffset] .read [xBytes] ∗
        regsFor 0 0 [lane] "x" [oldX])) ⊢ₛ
      wpInstr 0 0 (saxpyLoadX params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (globalSlices [xOffset] .read [xBytes] ∗
            regsFor 0 0 [lane] "x" [.s32 x])) := by
  simpa [saxpyLoadX] using
    (wp_globalLoadBytesReg_lanes_warpAt
      (cta := 0) (warp := 0) (pc := pc) (dst := "x") (ty := .s32)
      (addrExpr := saxpyGlobalLaneAddr params.xBase)
      (lanes := [lane]) (offsets := [xOffset]) (byteSlices := [xBytes])
      (oldValues := [oldX]) (newValues := [.s32 x])
      (by
        intro st r hpre
        exact ⟨haddr st r hpre, True.intro⟩)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨rSlices, _rRegs, _hcompRest, _hequivRest, hslices, _hregs⟩
        change (CSL.globalBytes xOffset .read xBytes ∗ globalSlices [] .read []) st rSlices
          at hslices
        rcases hslices with ⟨_rBytes, _rEmpty, _hcompBytes, _hequivBytes, hbytes, _hempty⟩
        exact ⟨globalReadMem_of_globalBytes haccess hwidth hbytes hdecode, True.intro⟩))

theorem saxpy_loadX_live_lanes_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x : Int}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          (globalSlices [xOffset] .read [xBytes] ∗
            regsFor 0 0 [lane] "x" [oldX])) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccess : AccessOk .global .s32 (.global xOffset))
    (hwidth : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecode : DecodedScalar .s32 xBytes (.s32 x)) :
    ((warpAt 0 0 pc [lane] ∗
      (globalSlices [xOffset] .read [xBytes] ∗
        regsFor 0 0 [lane] "x" [oldX])) ∗
      (regsFor 0 0 [lane] "y" [oldY] ∗
        (regsFor 0 0 [lane] "prod" [oldProd] ∗
          (regsFor 0 0 [lane] "sum" [oldSum] ∗
            (globalSlices [yOffset] .read [yBytes] ∗
              globalSlices [outOffset] .write [oldOut]))))) ⊢ₛ
      wpInstr 0 0 (saxpyLoadX params)
        ((warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (globalSlices [xOffset] .read [xBytes] ∗
            regsFor 0 0 [lane] "x" [.s32 x])) ∗
          (regsFor 0 0 [lane] "y" [oldY] ∗
            (regsFor 0 0 [lane] "prod" [oldProd] ∗
              (regsFor 0 0 [lane] "sum" [oldSum] ∗
                (globalSlices [yOffset] .read [yBytes] ∗
                  globalSlices [outOffset] .write [oldOut]))))) := by
  simpa [saxpyLoadX] using
    (wp_globalLoadBytesReg_lanes_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "x") (ty := .s32)
      (addrExpr := saxpyGlobalLaneAddr params.xBase)
      (lanes := [lane]) (offsets := [xOffset]) (byteSlices := [xBytes])
      (oldValues := [oldX]) (newValues := [.s32 x])
      (frame :=
        regsFor 0 0 [lane] "y" [oldY] ∗
          (regsFor 0 0 [lane] "prod" [oldProd] ∗
            (regsFor 0 0 [lane] "sum" [oldSum] ∗
              (globalSlices [yOffset] .read [yBytes] ∗
                globalSlices [outOffset] .write [oldOut]))))
      (by
        intro st r hpre
        exact ⟨haddr st r hpre, True.intro⟩)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨rSlices, _rRegs, _hcompRest, _hequivRest, hslices, _hregs⟩
        change (CSL.globalBytes xOffset .read xBytes ∗ globalSlices [] .read []) st rSlices
          at hslices
        rcases hslices with ⟨_rBytes, _rEmpty, _hcompBytes, _hequivBytes, hbytes, _hempty⟩
        exact ⟨globalReadMem_of_globalBytes haccess hwidth hbytes hdecode, True.intro⟩)
      (by
        exact CSL.stable_sep
          (stable_regsFor_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_regsFor_load_of_ne (by decide))
            (CSL.stable_sep
              (stable_regsFor_load_of_ne (by decide))
              (CSL.stable_sep stable_globalSlices_load stable_globalSlices_load)))))

theorem saxpy_mul_lanes_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd : Value} {x prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "prod" [oldProd] ∗
        regsFor 0 0 [lane] "x" [.s32 x])) ⊢ₛ
      wpInstr 0 0 (saxpyMul params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
            regsFor 0 0 [lane] "x" [.s32 x])) := by
  simpa [saxpyMul] using
    (wp_assignReg_lanes_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc) (dst := "prod")
      (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
      (lanes := [lane]) (oldValues := [oldProd]) (newValues := [.s32 prod])
      (frame := regsFor 0 0 [lane] "x" [.s32 x])
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, rFrame, _hcompRest, _hequivRest, _hdst, hxRegs⟩
        change (CSL.reg 0 0 lane "x" (.s32 x) ∗ regsFor 0 0 [] "x" []) st rFrame
          at hxRegs
        rcases hxRegs with ⟨_rX, _rEmpty, _hcompX, _hequivX, hx, _hempty⟩
        exact ⟨eval_binop_of_eval eval_imm (eval_reg_of_assertion hx) hmul,
          True.intro⟩)
      (by
        intro st st' _r rFrame _hpre hframe hstep
        exact (stable_regsFor_assignReg_of_ne
          (cta := 0) (warp := 0) (guard? := none) (dst := "prod") (name := "x")
          (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
          (lanes := [lane]) (values := [.s32 x]) (by decide))
          st st' rFrame hstep hframe))

theorem saxpy_add_lanes_wp
    {pc : PC} {lane : LaneId} {oldSum : Value} {prod y sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "sum" [oldSum] ∗
        (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
          regsFor 0 0 [lane] "y" [.s32 y]))) ⊢ₛ
      wpInstr 0 0 saxpyAdd
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "sum" [.s32 sum] ∗
            (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
              regsFor 0 0 [lane] "y" [.s32 y]))) := by
  simpa [saxpyAdd] using
    (wp_assignReg_lanes_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc) (dst := "sum")
      (rhs := .binop .add (.reg "prod") (.reg "y"))
      (lanes := [lane]) (oldValues := [oldSum]) (newValues := [.s32 sum])
      (frame :=
        regsFor 0 0 [lane] "prod" [.s32 prod] ∗
          regsFor 0 0 [lane] "y" [.s32 y])
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨rProd, rY, _hcompSrc, _hequivSrc, hprodRegs, hyRegs⟩
        change (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          regsFor 0 0 [] "prod" []) st rProd at hprodRegs
        change (CSL.reg 0 0 lane "y" (.s32 y) ∗ regsFor 0 0 [] "y" []) st rY
          at hyRegs
        rcases hprodRegs with ⟨_rProdHead, _rProdEmpty, _hcompProd, _hequivProd,
          hprod, _hprodEmpty⟩
        rcases hyRegs with ⟨_rYHead, _rYEmpty, _hcompY, _hequivY, hy, _hyEmpty⟩
        exact ⟨eval_binop_of_eval (eval_reg_of_assertion hprod)
          (eval_reg_of_assertion hy) hadd, True.intro⟩)
      (by
        intro st st' _r rFrame _hpre hframe hstep
        exact (CSL.stable_sep
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "sum")
            (name := "prod") (rhs := .binop .add (.reg "prod") (.reg "y"))
            (lanes := [lane]) (values := [.s32 prod]) (by decide))
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "sum")
            (name := "y") (rhs := .binop .add (.reg "prod") (.reg "y"))
            (lanes := [lane]) (values := [.s32 y]) (by decide)))
          st st' rFrame hstep hframe))

theorem saxpy_mul_live_lanes_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd oldSum : Value} {x y prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "prod" [oldProd] ∗
        (regsFor 0 0 [lane] "x" [.s32 x] ∗
          (regsFor 0 0 [lane] "sum" [oldSum] ∗
            regsFor 0 0 [lane] "y" [.s32 y])))) ⊢ₛ
      wpInstr 0 0 (saxpyMul params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
            (regsFor 0 0 [lane] "x" [.s32 x] ∗
              (regsFor 0 0 [lane] "sum" [oldSum] ∗
                regsFor 0 0 [lane] "y" [.s32 y])))) := by
  simpa [saxpyMul] using
    (wp_assignReg_lanes_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc) (dst := "prod")
      (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
      (lanes := [lane]) (oldValues := [oldProd]) (newValues := [.s32 prod])
      (frame :=
        regsFor 0 0 [lane] "x" [.s32 x] ∗
          (regsFor 0 0 [lane] "sum" [oldSum] ∗
            regsFor 0 0 [lane] "y" [.s32 y]))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨rX, _rLive, _hcompX, _hequivX, hxRegs, _hlive⟩
        change (CSL.reg 0 0 lane "x" (.s32 x) ∗ regsFor 0 0 [] "x" []) st rX
          at hxRegs
        rcases hxRegs with ⟨_rXHead, _rXEmpty, _hcompXHead, _hequivXHead, hx,
          _hxEmpty⟩
        exact ⟨eval_binop_of_eval eval_imm (eval_reg_of_assertion hx) hmul,
          True.intro⟩)
      (by
        intro st st' _r rFrame _hpre hframe hstep
        exact (CSL.stable_sep
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "prod")
            (name := "x")
            (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
            (lanes := [lane]) (values := [.s32 x]) (by decide))
          (CSL.stable_sep
            (stable_regsFor_assignReg_of_ne
              (cta := 0) (warp := 0) (guard? := none) (dst := "prod")
              (name := "sum")
              (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
              (lanes := [lane]) (values := [oldSum]) (by decide))
            (stable_regsFor_assignReg_of_ne
              (cta := 0) (warp := 0) (guard? := none) (dst := "prod")
              (name := "y")
              (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
              (lanes := [lane]) (values := [.s32 y]) (by decide))))
          st st' rFrame hstep hframe))

theorem saxpy_add_live_lanes_wp
    {pc : PC} {lane : LaneId} {oldSum : Value} {prod x y sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "sum" [oldSum] ∗
        (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
          (regsFor 0 0 [lane] "y" [.s32 y] ∗
            regsFor 0 0 [lane] "x" [.s32 x])))) ⊢ₛ
      wpInstr 0 0 saxpyAdd
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "sum" [.s32 sum] ∗
            (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
              (regsFor 0 0 [lane] "y" [.s32 y] ∗
                regsFor 0 0 [lane] "x" [.s32 x])))) := by
  simpa [saxpyAdd] using
    (wp_assignReg_lanes_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc) (dst := "sum")
      (rhs := .binop .add (.reg "prod") (.reg "y"))
      (lanes := [lane]) (oldValues := [oldSum]) (newValues := [.s32 sum])
      (frame :=
        regsFor 0 0 [lane] "prod" [.s32 prod] ∗
          (regsFor 0 0 [lane] "y" [.s32 y] ∗
            regsFor 0 0 [lane] "x" [.s32 x]))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨rProd, rYX, _hcompProd, _hequivProd, hprodRegs, hYX⟩
        rcases hYX with ⟨rY, _rX, _hcompYX, _hequivYX, hyRegs, _hxRegs⟩
        change (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          regsFor 0 0 [] "prod" []) st rProd at hprodRegs
        change (CSL.reg 0 0 lane "y" (.s32 y) ∗ regsFor 0 0 [] "y" []) st rY
          at hyRegs
        rcases hprodRegs with ⟨_rProdHead, _rProdEmpty, _hcompProdHead,
          _hequivProdHead, hprod, _hprodEmpty⟩
        rcases hyRegs with ⟨_rYHead, _rYEmpty, _hcompY, _hequivY, hy, _hyEmpty⟩
        exact ⟨eval_binop_of_eval (eval_reg_of_assertion hprod)
          (eval_reg_of_assertion hy) hadd, True.intro⟩)
      (by
        intro st st' _r rFrame _hpre hframe hstep
        exact (CSL.stable_sep
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "sum")
            (name := "prod") (rhs := .binop .add (.reg "prod") (.reg "y"))
            (lanes := [lane]) (values := [.s32 prod]) (by decide))
          (CSL.stable_sep
            (stable_regsFor_assignReg_of_ne
              (cta := 0) (warp := 0) (guard? := none) (dst := "sum")
              (name := "y") (rhs := .binop .add (.reg "prod") (.reg "y"))
              (lanes := [lane]) (values := [.s32 y]) (by decide))
            (stable_regsFor_assignReg_of_ne
              (cta := 0) (warp := 0) (guard? := none) (dst := "sum")
              (name := "x") (rhs := .binop .add (.reg "prod") (.reg "y"))
              (lanes := [lane]) (values := [.s32 x]) (by decide))))
          st st' rFrame hstep hframe))

theorem saxpy_arith_lanes_wpInstrList
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd oldSum : Value} {x y prod sum : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "prod" [oldProd] ∗
        (regsFor 0 0 [lane] "x" [.s32 x] ∗
          (regsFor 0 0 [lane] "sum" [oldSum] ∗
            regsFor 0 0 [lane] "y" [.s32 y])))) ⊢ₛ
      wpInstrList 0 0 [saxpyMul params, saxpyAdd]
        (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
          (regsFor 0 0 [lane] "sum" [.s32 sum] ∗
            (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
              (regsFor 0 0 [lane] "y" [.s32 y] ∗
                regsFor 0 0 [lane] "x" [.s32 x])))) := by
  have hafterMul :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
          (regsFor 0 0 [lane] "x" [.s32 x] ∗
            (regsFor 0 0 [lane] "sum" [oldSum] ∗
              regsFor 0 0 [lane] "y" [.s32 y])))) ⊢ₛ
        wpInstr 0 0 saxpyAdd
          (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
            (regsFor 0 0 [lane] "sum" [.s32 sum] ∗
              (regsFor 0 0 [lane] "prod" [.s32 prod] ∗
                (regsFor 0 0 [lane] "y" [.s32 y] ∗
                  regsFor 0 0 [lane] "x" [.s32 x])))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_acdb
          (regsFor 0 0 [lane] "prod" [.s32 prod])
          (regsFor 0 0 [lane] "x" [.s32 x])
          (regsFor 0 0 [lane] "sum" [oldSum])
          (regsFor 0 0 [lane] "y" [.s32 y])))
      (by
        simpa [Nat.add_assoc] using
          (saxpy_add_live_lanes_wp (pc := (pc.1, pc.2 + 1)) (lane := lane)
            (oldSum := oldSum) (prod := prod) (x := x) (y := y) (sum := sum) hadd))
  exact CSL.entails_trans (saxpy_mul_live_lanes_wp (pc := pc) (lane := lane) hmul)
    (wpInstr_mono (by simpa [wpInstrList] using hafterMul))

theorem saxpy_mul_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd : Value} {x prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "prod" oldProd ∗ CSL.reg 0 0 lane "x" (.s32 x))) ⊢ₛ
      wpInstr 0 0 (saxpyMul params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            CSL.reg 0 0 lane "x" (.s32 x))) := by
  simpa [saxpyMul] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "prod")
      (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x")) (lane := lane)
      (old := oldProd) (new := .s32 prod)
      (frame := CSL.reg 0 0 lane "x" (.s32 x))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rSrc, _hcompRest, _hequivRest, _hdst, hsrc⟩
        exact eval_binop_of_eval eval_imm (eval_reg_of_assertion hsrc) hmul)
      (by
        exact stable_reg_assignReg_of_ne (by decide)))

theorem saxpy_mul_live_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd oldSum : Value} {x y prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "prod" oldProd ∗
        (CSL.reg 0 0 lane "x" (.s32 x) ∗
          (CSL.reg 0 0 lane "sum" oldSum ∗
            CSL.reg 0 0 lane "y" (.s32 y))))) ⊢ₛ
      wpInstr 0 0 (saxpyMul params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            (CSL.reg 0 0 lane "x" (.s32 x) ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                CSL.reg 0 0 lane "y" (.s32 y))))) := by
  simpa [saxpyMul] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "prod")
      (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x")) (lane := lane)
      (old := oldProd) (new := .s32 prod)
      (frame :=
        CSL.reg 0 0 lane "x" (.s32 x) ∗
          (CSL.reg 0 0 lane "sum" oldSum ∗
            CSL.reg 0 0 lane "y" (.s32 y)))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with
          ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rX, _rRestLive, _hcompX, _hequivX, hx, _hlive⟩
        exact eval_binop_of_eval eval_imm (eval_reg_of_assertion hx) hmul)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (stable_reg_assignReg_of_ne (by decide)))))

theorem saxpy_add_wp
    {pc : PC} {lane : LaneId} {oldSum : Value} {prod y sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "sum" oldSum ∗
        (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          CSL.reg 0 0 lane "y" (.s32 y)))) ⊢ₛ
      wpInstr 0 0 saxpyAdd
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              CSL.reg 0 0 lane "y" (.s32 y)))) := by
  simpa [saxpyAdd] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "sum")
      (rhs := .binop .add (.reg "prod") (.reg "y")) (lane := lane)
      (old := oldSum) (new := .s32 sum)
      (frame :=
        CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          CSL.reg 0 0 lane "y" (.s32 y))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rProd, _rY, _hcompSrc, _hequivSrc, hprod, hy⟩
        exact eval_binop_of_eval (eval_reg_of_assertion hprod)
          (eval_reg_of_assertion hy) hadd)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (stable_reg_assignReg_of_ne (by decide))))

theorem saxpy_add_live_wp
    {pc : PC} {lane : LaneId} {oldSum : Value} {prod x y sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "sum" oldSum ∗
        (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          (CSL.reg 0 0 lane "y" (.s32 y) ∗
            CSL.reg 0 0 lane "x" (.s32 x))))) ⊢ₛ
      wpInstr 0 0 saxpyAdd
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                CSL.reg 0 0 lane "x" (.s32 x))))) := by
  simpa [saxpyAdd] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "sum")
      (rhs := .binop .add (.reg "prod") (.reg "y")) (lane := lane)
      (old := oldSum) (new := .s32 sum)
      (frame :=
        CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          (CSL.reg 0 0 lane "y" (.s32 y) ∗
            CSL.reg 0 0 lane "x" (.s32 x)))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with
          ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rProd, _rRestLive, _hcompProd, _hequivProd, hprod, hrestLive⟩
        rcases hrestLive with ⟨_rY, _rX, _hcompYX, _hequivYX, hy, _hx⟩
        exact eval_binop_of_eval (eval_reg_of_assertion hprod)
          (eval_reg_of_assertion hy) hadd)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (stable_reg_assignReg_of_ne (by decide)))))

theorem saxpy_arith_wpInstrList
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {oldProd oldSum : Value} {x y prod sum : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "prod" oldProd ∗
        (CSL.reg 0 0 lane "x" (.s32 x) ∗
          (CSL.reg 0 0 lane "sum" oldSum ∗
            CSL.reg 0 0 lane "y" (.s32 y))))) ⊢ₛ
      wpInstrList 0 0 [saxpyMul params, saxpyAdd]
        (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                CSL.reg 0 0 lane "x" (.s32 x))))) := by
  have hafterMul :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          (CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "sum" oldSum ∗
              CSL.reg 0 0 lane "y" (.s32 y))))) ⊢ₛ
        wpInstr 0 0 saxpyAdd
          (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
            (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
              (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
                (CSL.reg 0 0 lane "y" (.s32 y) ∗
                  CSL.reg 0 0 lane "x" (.s32 x))))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_acdb
          (CSL.reg 0 0 lane "prod" (.s32 prod))
          (CSL.reg 0 0 lane "x" (.s32 x))
          (CSL.reg 0 0 lane "sum" oldSum)
          (CSL.reg 0 0 lane "y" (.s32 y))))
      (by
        simpa [Nat.add_assoc] using
          (saxpy_add_live_wp (pc := (pc.1, pc.2 + 1)) (lane := lane)
            (oldSum := oldSum) (prod := prod) (x := x) (y := y) (sum := sum) hadd))
  exact CSL.entails_trans (saxpy_mul_live_wp (pc := pc) (lane := lane) hmul)
    (wpInstr_mono (by simpa [wpInstrList] using hafterMul))

def saxpyStore (params : SaxpyParams) : GInstr :=
  { guard? := none
    instr := .store
      { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
      (.reg "sum") }

private theorem perm_abcdefg_to_fcbdeag
    (a b c d e f g : CSL.Assertion) :
    [a, b, c, d, e, f, g].Perm [f, c, b, d, e, a, g] := by
  have h1 : [a, b, c, d, e, f, g].Perm [f, a, b, c, d, e, g] := by
    simpa using (List.perm_middle (a := f) (l₁ := [a, b, c, d, e]) (l₂ := [g]))
  have h2 : [a, b, c, d, e, g].Perm [c, a, b, d, e, g] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, g]))
  have h3 : [a, b, d, e, g].Perm [b, a, d, e, g] :=
    (List.Perm.swap a b [d, e, g]).symm
  have h4 : [a, d, e, g].Perm [d, e, a, g] := by
    simpa using (List.perm_middle (a := a) (l₁ := [d, e]) (l₂ := [g])).symm
  exact h1.trans (List.Perm.cons f <| h2.trans <|
    List.Perm.cons c <| h3.trans <| List.Perm.cons b h4)

private theorem perm_abcdefg_to_dcebfag
    (a b c d e f g : CSL.Assertion) :
    [a, b, c, d, e, f, g].Perm [d, c, e, b, f, a, g] := by
  have h1 : [a, b, c, d, e, f, g].Perm [d, a, b, c, e, f, g] := by
    simpa using (List.perm_middle (a := d) (l₁ := [a, b, c]) (l₂ := [e, f, g]))
  have h2 : [a, b, c, e, f, g].Perm [c, a, b, e, f, g] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [e, f, g]))
  have h3 : [a, b, e, f, g].Perm [e, a, b, f, g] := by
    simpa using (List.perm_middle (a := e) (l₁ := [a, b]) (l₂ := [f, g]))
  have h4 : [a, b, f, g].Perm [b, a, f, g] :=
    (List.Perm.swap a b [f, g]).symm
  have h5 : [a, f, g].Perm [f, a, g] :=
    (List.Perm.swap a f [g]).symm
  exact h1.trans (List.Perm.cons d <| h2.trans <|
    List.Perm.cons c <| h3.trans <|
      List.Perm.cons e <| h4.trans <| List.Perm.cons b h5)

private theorem perm_abcdefg_to_cadbefg
    (a b c d e f g : CSL.Assertion) :
    [a, b, c, d, e, f, g].Perm [c, a, d, b, e, f, g] := by
  have h1 : [a, b, c, d, e, f, g].Perm [c, a, b, d, e, f, g] := by
    simpa using (List.perm_middle (a := c) (l₁ := [a, b]) (l₂ := [d, e, f, g]))
  have h2 : [b, d, e, f, g].Perm [d, b, e, f, g] :=
    (List.Perm.swap b d [e, f, g]).symm
  exact h1.trans (List.Perm.cons c <| List.Perm.cons a h2)

private theorem perm_abcdefg_to_gabcdef
    (a b c d e f g : CSL.Assertion) :
    [a, b, c, d, e, f, g].Perm [g, a, b, c, d, e, f] := by
  simpa using (List.perm_middle (a := g) (l₁ := [a, b, c, d, e, f]) (l₂ := []))

theorem saxpy_loadX_to_loadY_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x : Int}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes xOffset .read xBytes,
            CSL.reg 0 0 lane "x" oldX,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccess : AccessOk .global .s32 (.global xOffset))
    (hwidth : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecode : DecodedScalar .s32 xBytes (.s32 x)) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.globalBytes xOffset .read xBytes,
        CSL.reg 0 0 lane "x" oldX,
        CSL.reg 0 0 lane "y" oldY,
        CSL.reg 0 0 lane "prod" oldProd,
        CSL.reg 0 0 lane "sum" oldSum,
        CSL.globalBytes yOffset .read yBytes,
        CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
      wpInstr 0 0 (saxpyLoadX params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
  have hpreToRule :
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.globalBytes xOffset .read xBytes,
          CSL.reg 0 0 lane "x" oldX,
          CSL.reg 0 0 lane "y" oldY,
          CSL.reg 0 0 lane "prod" oldProd,
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.globalBytes yOffset .read yBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          ((CSL.globalBytes xOffset .read xBytes ∗
            CSL.reg 0 0 lane "x" oldX) ∗
            (CSL.reg 0 0 lane "y" oldY ∗
              (CSL.reg 0 0 lane "prod" oldProd ∗
                (CSL.reg 0 0 lane "sum" oldSum ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut)))))) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sepList_perm_to_sep_pair_cons
          [
            CSL.globalBytes xOffset .read xBytes,
            CSL.reg 0 0 lane "x" oldX,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]
          (CSL.globalBytes xOffset .read xBytes)
          (CSL.reg 0 0 lane "x" oldX)
          (CSL.reg 0 0 lane "y" oldY)
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]
          (List.Perm.refl _))
  have hpostToTarget :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes xOffset .read xBytes ∗
          CSL.reg 0 0 lane "x" (.s32 x)) ∗
          (CSL.reg 0 0 lane "y" oldY ∗
            (CSL.reg 0 0 lane "prod" oldProd ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                (CSL.globalBytes yOffset .read yBytes ∗
                  CSL.globalBytes outOffset .write oldOut)))))) ⊢ₛ
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_pair_cons_perm_to_sepList
          (CSL.globalBytes xOffset .read xBytes)
          (CSL.reg 0 0 lane "x" (.s32 x))
          (CSL.reg 0 0 lane "y" oldY)
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]
          [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]
          (perm_abcdefg_to_fcbdeag
            (CSL.globalBytes xOffset .read xBytes)
            (CSL.reg 0 0 lane "x" (.s32 x))
            (CSL.reg 0 0 lane "y" oldY)
            (CSL.reg 0 0 lane "prod" oldProd)
            (CSL.reg 0 0 lane "sum" oldSum)
            (CSL.globalBytes yOffset .read yBytes)
            (CSL.globalBytes outOffset .write oldOut)))
  have hpreFromRule :
      (warpAt 0 0 pc [lane] ∗
        ((CSL.globalBytes xOffset .read xBytes ∗
          CSL.reg 0 0 lane "x" oldX) ∗
          (CSL.reg 0 0 lane "y" oldY ∗
            (CSL.reg 0 0 lane "prod" oldProd ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                (CSL.globalBytes yOffset .read yBytes ∗
                  CSL.globalBytes outOffset .write oldOut)))))) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes xOffset .read xBytes,
            CSL.reg 0 0 lane "x" oldX,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_pair_cons_to_sepList
          (CSL.globalBytes xOffset .read xBytes)
          (CSL.reg 0 0 lane "x" oldX)
          (CSL.reg 0 0 lane "y" oldY)
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut])
  exact CSL.entails_trans hpreToRule <|
    CSL.entails_trans
      (by
        simpa [saxpyLoadX] using
          (wp_globalLoadBytesReg_single_warpAt_stableFrame
            (cta := 0) (warp := 0) (pc := pc) (dst := "x")
            (ty := .s32) (addrExpr := saxpyGlobalLaneAddr params.xBase)
            (lane := lane) (offset := xOffset)
            (bytes := xBytes) (oldReg := oldX) (value := .s32 x)
            (frame :=
              CSL.reg 0 0 lane "y" oldY ∗
                (CSL.reg 0 0 lane "prod" oldProd ∗
                  (CSL.reg 0 0 lane "sum" oldSum ∗
                    (CSL.globalBytes yOffset .read yBytes ∗
                      CSL.globalBytes outOffset .write oldOut))))
            (by
              intro st r hpre
              exact haddr st r (hpreFromRule st r hpre))
            (by
              intro st r hpre
              rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
              rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
                hmemReg, _hframe⟩
              rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
                hbytes, _hdst⟩
              exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
            (by
              exact CSL.stable_sep
                (stable_reg_load_of_ne (by decide))
                (CSL.stable_sep
                  (stable_reg_load_of_ne (by decide))
                  (CSL.stable_sep
                    (stable_reg_load_of_ne (by decide))
                    (CSL.stable_sep stable_globalBytes_load stable_globalBytes_load))))))
      (wpInstr_mono hpostToTarget)

theorem saxpy_loadY_to_mul_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut : List Byte}
    {oldY oldProd oldSum : Value} {x y : Int}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccess : AccessOk .global .s32 (.global yOffset))
    (hwidth : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecode : DecodedScalar .s32 yBytes (.s32 y)) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.globalBytes yOffset .read yBytes,
        CSL.reg 0 0 lane "y" oldY,
        CSL.reg 0 0 lane "x" (.s32 x),
        CSL.reg 0 0 lane "prod" oldProd,
        CSL.reg 0 0 lane "sum" oldSum,
        CSL.globalBytes xOffset .read xBytes,
        CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
      wpInstr 0 0 (saxpyLoadY params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
  have hpreToRule :
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.globalBytes yOffset .read yBytes,
          CSL.reg 0 0 lane "y" oldY,
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.reg 0 0 lane "prod" oldProd,
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          ((CSL.globalBytes yOffset .read yBytes ∗
            CSL.reg 0 0 lane "y" oldY) ∗
            (CSL.reg 0 0 lane "x" (.s32 x) ∗
              (CSL.reg 0 0 lane "prod" oldProd ∗
                (CSL.reg 0 0 lane "sum" oldSum ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    CSL.globalBytes outOffset .write oldOut)))))) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sepList_perm_to_sep_pair_cons
          [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]
          (CSL.globalBytes yOffset .read yBytes)
          (CSL.reg 0 0 lane "y" oldY)
          (CSL.reg 0 0 lane "x" (.s32 x))
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]
          (List.Perm.refl _))
  have hpostToTarget :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes yOffset .read yBytes ∗
          CSL.reg 0 0 lane "y" (.s32 y)) ∗
          (CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "prod" oldProd ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  CSL.globalBytes outOffset .write oldOut)))))) ⊢ₛ
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_pair_cons_perm_to_sepList
          (CSL.globalBytes yOffset .read yBytes)
          (CSL.reg 0 0 lane "y" (.s32 y))
          (CSL.reg 0 0 lane "x" (.s32 x))
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]
          (perm_abcdefg_to_dcebfag
            (CSL.globalBytes yOffset .read yBytes)
            (CSL.reg 0 0 lane "y" (.s32 y))
            (CSL.reg 0 0 lane "x" (.s32 x))
            (CSL.reg 0 0 lane "prod" oldProd)
            (CSL.reg 0 0 lane "sum" oldSum)
            (CSL.globalBytes xOffset .read xBytes)
            (CSL.globalBytes outOffset .write oldOut)))
  have hpreFromRule :
      (warpAt 0 0 pc [lane] ∗
        ((CSL.globalBytes yOffset .read yBytes ∗
          CSL.reg 0 0 lane "y" oldY) ∗
          (CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "prod" oldProd ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  CSL.globalBytes outOffset .write oldOut)))))) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_pair_cons_to_sepList
          (CSL.globalBytes yOffset .read yBytes)
          (CSL.reg 0 0 lane "y" oldY)
          (CSL.reg 0 0 lane "x" (.s32 x))
          [
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut])
  exact CSL.entails_trans hpreToRule <|
    CSL.entails_trans
      (by
        simpa [saxpyLoadY] using
          (wp_globalLoadBytesReg_single_warpAt_stableFrame
            (cta := 0) (warp := 0) (pc := pc) (dst := "y")
            (ty := .s32) (addrExpr := saxpyGlobalLaneAddr params.yBase)
            (lane := lane) (offset := yOffset)
            (bytes := yBytes) (oldReg := oldY) (value := .s32 y)
            (frame :=
              CSL.reg 0 0 lane "x" (.s32 x) ∗
                (CSL.reg 0 0 lane "prod" oldProd ∗
                  (CSL.reg 0 0 lane "sum" oldSum ∗
                    (CSL.globalBytes xOffset .read xBytes ∗
                      CSL.globalBytes outOffset .write oldOut))))
            (by
              intro st r hpre
              exact haddr st r (hpreFromRule st r hpre))
            (by
              intro st r hpre
              rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
              rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest,
                hmemReg, _hframe⟩
              rcases hmemReg with ⟨_rBytes, _rDst, _hcompMemReg, _hequivMemReg,
                hbytes, _hdst⟩
              exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
            (by
              exact CSL.stable_sep
                (stable_reg_load_of_ne (by decide))
                (CSL.stable_sep
                  (stable_reg_load_of_ne (by decide))
                  (CSL.stable_sep
                    (stable_reg_load_of_ne (by decide))
                    (CSL.stable_sep stable_globalBytes_load stable_globalBytes_load))))))
      (wpInstr_mono hpostToTarget)

theorem saxpy_mul_to_add_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut : List Byte}
    {oldProd oldSum : Value} {x y prod : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod)) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.reg 0 0 lane "prod" oldProd,
        CSL.reg 0 0 lane "x" (.s32 x),
        CSL.reg 0 0 lane "sum" oldSum,
        CSL.reg 0 0 lane "y" (.s32 y),
        CSL.globalBytes xOffset .read xBytes,
        CSL.globalBytes yOffset .read yBytes,
        CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
      wpInstr 0 0 (saxpyMul params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
  have hpreToRule :
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "prod" oldProd,
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          (CSL.reg 0 0 lane "prod" oldProd ∗
            (CSL.reg 0 0 lane "x" (.s32 x) ∗
              (CSL.reg 0 0 lane "sum" oldSum ∗
                (CSL.reg 0 0 lane "y" (.s32 y) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    (CSL.globalBytes yOffset .read yBytes ∗
                      CSL.globalBytes outOffset .write oldOut))))))) := by
    intro st r h
    simpa [CSL.sepList] using h
  have hpostToTarget :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
          (CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "sum" oldSum ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut))))))) ⊢ₛ
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sepList_perm
          (perm_abcdefg_to_cadbefg
            (CSL.reg 0 0 lane "prod" (.s32 prod))
            (CSL.reg 0 0 lane "x" (.s32 x))
            (CSL.reg 0 0 lane "sum" oldSum)
            (CSL.reg 0 0 lane "y" (.s32 y))
            (CSL.globalBytes xOffset .read xBytes)
            (CSL.globalBytes yOffset .read yBytes)
            (CSL.globalBytes outOffset .write oldOut)))
  have hstepRule :
      (warpAt 0 0 pc [lane] ∗
        (CSL.reg 0 0 lane "prod" oldProd ∗
          (CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "sum" oldSum ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut))))))) ⊢ₛ
        wpInstr 0 0 (saxpyMul params)
          (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "x" (.s32 x) ∗
                (CSL.reg 0 0 lane "sum" oldSum ∗
                  (CSL.reg 0 0 lane "y" (.s32 y) ∗
                    (CSL.globalBytes xOffset .read xBytes ∗
                      (CSL.globalBytes yOffset .read yBytes ∗
                        CSL.globalBytes outOffset .write oldOut))))))) := by
    simpa [saxpyMul] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := pc) (dst := "prod")
        (rhs := .binop .mul (.imm (.s32 params.alpha)) (.reg "x"))
        (lane := lane) (old := oldProd) (new := .s32 prod)
        (frame :=
          CSL.reg 0 0 lane "x" (.s32 x) ∗
            (CSL.reg 0 0 lane "sum" oldSum ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut)))))
        (heval := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
          rcases hframe with ⟨_rX, _rRestLive, _hcompX, _hequivX, hx, _hlive⟩
          exact eval_binop_of_eval eval_imm (eval_reg_of_assertion hx) hmul)
        (hframe := by
          exact CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (CSL.stable_sep
              (stable_reg_assignReg_of_ne (by decide))
              (CSL.stable_sep
                (stable_reg_assignReg_of_ne (by decide))
                (CSL.stable_sep stable_globalBytes_assignReg
                  (CSL.stable_sep stable_globalBytes_assignReg stable_globalBytes_assignReg))))))
  exact CSL.entails_trans hpreToRule
    (CSL.entails_trans hstepRule (wpInstr_mono hpostToTarget))

theorem saxpy_add_to_store_wp
    {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut : List Byte}
    {oldSum : Value} {x y prod sum : Int}
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum)) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.reg 0 0 lane "sum" oldSum,
        CSL.reg 0 0 lane "prod" (.s32 prod),
        CSL.reg 0 0 lane "y" (.s32 y),
        CSL.reg 0 0 lane "x" (.s32 x),
        CSL.globalBytes xOffset .read xBytes,
        CSL.globalBytes yOffset .read yBytes,
        CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
      wpInstr 0 0 saxpyAdd
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write oldOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
  have hpreToRule :
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.reg 0 0 lane "prod" (.s32 prod),
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          (CSL.reg 0 0 lane "sum" oldSum ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.reg 0 0 lane "x" (.s32 x) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    (CSL.globalBytes yOffset .read yBytes ∗
                      CSL.globalBytes outOffset .write oldOut))))))) := by
    intro st r h
    simpa [CSL.sepList] using h
  have hpostToTarget :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
          (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            (CSL.reg 0 0 lane "y" (.s32 y) ∗
              (CSL.reg 0 0 lane "x" (.s32 x) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut))))))) ⊢ₛ
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write oldOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
    simpa [CSL.sepList] using
      CSL.sep_mono (CSL.entails_refl _)
        (CSL.sepList_perm
          (perm_abcdefg_to_gabcdef
            (CSL.reg 0 0 lane "sum" (.s32 sum))
            (CSL.reg 0 0 lane "prod" (.s32 prod))
            (CSL.reg 0 0 lane "y" (.s32 y))
            (CSL.reg 0 0 lane "x" (.s32 x))
            (CSL.globalBytes xOffset .read xBytes)
            (CSL.globalBytes yOffset .read yBytes)
            (CSL.globalBytes outOffset .write oldOut)))
  have hstepRule :
      (warpAt 0 0 pc [lane] ∗
        (CSL.reg 0 0 lane "sum" oldSum ∗
          (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            (CSL.reg 0 0 lane "y" (.s32 y) ∗
              (CSL.reg 0 0 lane "x" (.s32 x) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut))))))) ⊢ₛ
        wpInstr 0 0 saxpyAdd
          (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
            (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
              (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
                (CSL.reg 0 0 lane "y" (.s32 y) ∗
                  (CSL.reg 0 0 lane "x" (.s32 x) ∗
                    (CSL.globalBytes xOffset .read xBytes ∗
                      (CSL.globalBytes yOffset .read yBytes ∗
                        CSL.globalBytes outOffset .write oldOut))))))) := by
    simpa [saxpyAdd] using
      (wp_assignReg_single_warpAt_stableFrame
        (cta := 0) (warp := 0) (pc := pc) (dst := "sum")
        (rhs := .binop .add (.reg "prod") (.reg "y"))
        (lane := lane) (old := oldSum) (new := .s32 sum)
        (frame :=
          CSL.reg 0 0 lane "prod" (.s32 prod) ∗
            (CSL.reg 0 0 lane "y" (.s32 y) ∗
              (CSL.reg 0 0 lane "x" (.s32 x) ∗
                (CSL.globalBytes xOffset .read xBytes ∗
                  (CSL.globalBytes yOffset .read yBytes ∗
                    CSL.globalBytes outOffset .write oldOut)))))
        (heval := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
          rcases hframe with ⟨_rProd, _rRestLive, _hcompProd, _hequivProd,
            hprod, hrestLive⟩
          rcases hrestLive with ⟨_rY, _rRestY, _hcompY, _hequivY, hy, _hrestY⟩
          exact eval_binop_of_eval (eval_reg_of_assertion hprod)
            (eval_reg_of_assertion hy) hadd)
        (hframe := by
          exact CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (CSL.stable_sep
              (stable_reg_assignReg_of_ne (by decide))
              (CSL.stable_sep
                (stable_reg_assignReg_of_ne (by decide))
                (CSL.stable_sep stable_globalBytes_assignReg
                  (CSL.stable_sep stable_globalBytes_assignReg stable_globalBytes_assignReg))))))
  exact CSL.entails_trans hpreToRule
    (CSL.entails_trans hstepRule (wpInstr_mono hpostToTarget))

theorem saxpy_store_final_wp
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {x y prod sum : Int}
    (haddr :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write oldOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccess : AccessOk .global .s32 (.global outOffset))
    (hencode : EncodedScalar .s32 (.s32 sum) newOut)
    (hlen : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.globalBytes outOffset .write oldOut,
        CSL.reg 0 0 lane "sum" (.s32 sum),
        CSL.reg 0 0 lane "prod" (.s32 prod),
        CSL.reg 0 0 lane "y" (.s32 y),
        CSL.reg 0 0 lane "x" (.s32 x),
        CSL.globalBytes xOffset .read xBytes,
        CSL.globalBytes yOffset .read yBytes]) ⊢ₛ
      wpInstr 0 0 (saxpyStore params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write newOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
  have hpreToRule :
      (warpAt 0 0 pc [lane] ∗
        CSL.sepList [
          CSL.globalBytes outOffset .write oldOut,
          CSL.reg 0 0 lane "sum" (.s32 sum),
          CSL.reg 0 0 lane "prod" (.s32 prod),
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes]) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          (CSL.globalBytes outOffset .write oldOut ∗
            (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
              (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
                (CSL.reg 0 0 lane "y" (.s32 y) ∗
                  (CSL.reg 0 0 lane "x" (.s32 x) ∗
                    (CSL.globalBytes xOffset .read xBytes ∗
                      CSL.globalBytes yOffset .read yBytes))))))) := by
    intro st r h
    simpa [CSL.sepList] using h
  have hpostToTarget :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        (CSL.globalBytes outOffset .write newOut ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.reg 0 0 lane "x" (.s32 x) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    CSL.globalBytes yOffset .read yBytes))))))) ⊢ₛ
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write newOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
    intro st r h
    simpa [CSL.sepList] using h
  have hpreFromRule :
      (warpAt 0 0 pc [lane] ∗
        (CSL.globalBytes outOffset .write oldOut ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.reg 0 0 lane "x" (.s32 x) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    CSL.globalBytes yOffset .read yBytes))))))) ⊢ₛ
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write oldOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
    intro st r h
    simpa [CSL.sepList] using h
  have hstepRule :
      (warpAt 0 0 pc [lane] ∗
        (CSL.globalBytes outOffset .write oldOut ∗
          (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.reg 0 0 lane "x" (.s32 x) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    CSL.globalBytes yOffset .read yBytes))))))) ⊢ₛ
        wpInstr 0 0 (saxpyStore params)
          (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
            (CSL.globalBytes outOffset .write newOut ∗
              (CSL.reg 0 0 lane "sum" (.s32 sum) ∗
                (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
                  (CSL.reg 0 0 lane "y" (.s32 y) ∗
                    (CSL.reg 0 0 lane "x" (.s32 x) ∗
                      (CSL.globalBytes xOffset .read xBytes ∗
                        CSL.globalBytes yOffset .read yBytes))))))) := by
    simpa [saxpyStore] using
      (wp_globalStoreBytes_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := pc)
        (ty := .s32) (addrExpr := saxpyGlobalLaneAddr params.outBase)
        (valueExpr := .reg "sum") (lane := lane) (offset := outOffset)
        (oldBytes := oldOut) (newBytes := newOut)
        (value := .s32 sum)
        (frame :=
          CSL.reg 0 0 lane "sum" (.s32 sum) ∗
            (CSL.reg 0 0 lane "prod" (.s32 prod) ∗
              (CSL.reg 0 0 lane "y" (.s32 y) ∗
                (CSL.reg 0 0 lane "x" (.s32 x) ∗
                  (CSL.globalBytes xOffset .read xBytes ∗
                    CSL.globalBytes yOffset .read yBytes)))))
        (haddr := by
          intro st r hpre
          exact haddr st r (hpreFromRule st r hpre))
        (heval := by
          intro st r hpre
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
          rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
          rcases hframe with ⟨_rSum, _rRestRegs, _hcompSum, _hequivSum,
            hsum, _hrestRegs⟩
          exact eval_reg_of_assertion hsum)
        (hwrite := by
          intro st r _hpre
          exact ⟨{ st with global := {
              bytes := Helpers.writeBytes st.global.bytes outOffset newOut } },
            globalWriteMem_of_byteWrite haccess hencode (by rfl)⟩)
        (hencode := hencode) (hlen := hlen)
        (hframe := by
          intro st st' r rFrame hpre hframe hstep
          have haddr' := haddr st r (hpreFromRule st r hpre)
          rcases hframe with ⟨rSum, rRest, hcompSum, hequivSum, hsum, hrest⟩
          rcases hrest with ⟨rProd, rRestProd, hcompProd, hequivProd, hprod, hrestProd⟩
          rcases hrestProd with ⟨rY, rRestY, hcompY, hequivY, hy, hrestY⟩
          rcases hrestY with ⟨rX, rBytes, hcompX, hequivX, hx, hbytes⟩
          rcases hbytes with ⟨rXBytes, rYBytes, hcompBytes, hequivBytes, hxBytes, hyBytes⟩
          have heval' : EvalRValue st { cta := 0, warp := 0, lane := lane }
              (.reg "sum") (.s32 sum) := by
            exact eval_reg_of_assertion hsum
          have hwrite' : WriteMemFact st .global .s32 (.global outOffset) (.s32 sum)
              { st with global := {
                  bytes := Helpers.writeBytes st.global.bytes outOffset newOut } } :=
            globalWriteMem_of_byteWrite haccess hencode (by rfl)
          rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
          exact ⟨rSum, rRest, hcompSum, hequivSum,
            globalStorePreservesReadReg_single_warpAt
              hctrl hsum haddr' heval' hwrite' hstep,
            ⟨rProd, rRestProd, hcompProd, hequivProd,
              globalStorePreservesReadReg_single_warpAt
                hctrl hprod haddr' heval' hwrite' hstep,
              ⟨rY, rRestY, hcompY, hequivY,
                globalStorePreservesReadReg_single_warpAt
                  hctrl hy haddr' heval' hwrite' hstep,
                ⟨rX, rBytes, hcompX, hequivX,
                  globalStorePreservesReadReg_single_warpAt
                    hctrl hx haddr' heval' hwrite' hstep,
                  ⟨rXBytes, rYBytes, hcompBytes, hequivBytes,
                    globalStorePreservesGlobalBytes_single_warpAt hdisjointX
                      hctrl hxBytes haddr' heval' hwrite' hencode hstep,
                    globalStorePreservesGlobalBytes_single_warpAt hdisjointY
                      hctrl hyBytes haddr' heval' hwrite' hencode hstep⟩⟩⟩⟩⟩))
  exact CSL.entails_trans hpreToRule
    (CSL.entails_trans hstepRule (wpInstr_mono hpostToTarget))

theorem saxpy_loads_mul_add_store_wpInstrList
    {params : SaxpyParams} {pc : PC} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int}
    (haddrX :
      ∀ st r,
        (warpAt 0 0 pc [lane] ∗
          CSL.sepList [
            CSL.globalBytes xOffset .read xBytes,
            CSL.reg 0 0 lane "x" oldX,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes yOffset .read yBytes,
            CSL.globalBytes outOffset .write oldOut]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          CSL.sepList [
            CSL.globalBytes yOffset .read yBytes,
            CSL.reg 0 0 lane "y" oldY,
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.reg 0 0 lane "prod" oldProd,
            CSL.reg 0 0 lane "sum" oldSum,
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes outOffset .write oldOut]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write oldOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      CSL.sepList [
        CSL.globalBytes xOffset .read xBytes,
        CSL.reg 0 0 lane "x" oldX,
        CSL.reg 0 0 lane "y" oldY,
        CSL.reg 0 0 lane "prod" oldProd,
        CSL.reg 0 0 lane "sum" oldSum,
        CSL.globalBytes yOffset .read yBytes,
        CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
      wpInstrList 0 0
        [saxpyLoadX params, saxpyLoadY params, saxpyMul params, saxpyAdd,
          saxpyStore params]
        (warpAt 0 0 (pc.1, pc.2 + 5) [lane] ∗
          CSL.sepList [
            CSL.globalBytes outOffset .write newOut,
            CSL.reg 0 0 lane "sum" (.s32 sum),
            CSL.reg 0 0 lane "prod" (.s32 prod),
            CSL.reg 0 0 lane "y" (.s32 y),
            CSL.reg 0 0 lane "x" (.s32 x),
            CSL.globalBytes xOffset .read xBytes,
            CSL.globalBytes yOffset .read yBytes]) := by
  have hafterStore :
      (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
        CSL.sepList [
          CSL.globalBytes outOffset .write oldOut,
          CSL.reg 0 0 lane "sum" (.s32 sum),
          CSL.reg 0 0 lane "prod" (.s32 prod),
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes]) ⊢ₛ
        wpInstrList 0 0 [saxpyStore params]
          (warpAt 0 0 (pc.1, pc.2 + 5) [lane] ∗
            CSL.sepList [
              CSL.globalBytes outOffset .write newOut,
              CSL.reg 0 0 lane "sum" (.s32 sum),
              CSL.reg 0 0 lane "prod" (.s32 prod),
              CSL.reg 0 0 lane "y" (.s32 y),
              CSL.reg 0 0 lane "x" (.s32 x),
              CSL.globalBytes xOffset .read xBytes,
              CSL.globalBytes yOffset .read yBytes]) := by
    simpa [wpInstrList, Nat.add_assoc] using
      (saxpy_store_final_wp
        (params := params) (pc := (pc.1, pc.2 + 4)) (lane := lane)
        (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
        (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut) (newOut := newOut)
        (x := x) (y := y) (prod := prod) (sum := sum)
        haddrOut hdisjointX hdisjointY haccessOut hencodeOut hlenOut)
  have hafterAdd :
      (warpAt 0 0 (pc.1, pc.2 + 3) [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.reg 0 0 lane "prod" (.s32 prod),
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        wpInstrList 0 0 [saxpyAdd, saxpyStore params]
          (warpAt 0 0 (pc.1, pc.2 + 5) [lane] ∗
            CSL.sepList [
              CSL.globalBytes outOffset .write newOut,
              CSL.reg 0 0 lane "sum" (.s32 sum),
              CSL.reg 0 0 lane "prod" (.s32 prod),
              CSL.reg 0 0 lane "y" (.s32 y),
              CSL.reg 0 0 lane "x" (.s32 x),
              CSL.globalBytes xOffset .read xBytes,
              CSL.globalBytes yOffset .read yBytes]) := by
    exact CSL.entails_trans
      (saxpy_add_to_store_wp
        (pc := (pc.1, pc.2 + 3)) (lane := lane)
        (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
        (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut)
        (oldSum := oldSum) (x := x) (y := y) (prod := prod) (sum := sum) hadd)
      (wpInstr_mono (by simpa [wpInstrList, Nat.add_assoc] using hafterStore))
  have hafterMul :
      (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
        CSL.sepList [
          CSL.reg 0 0 lane "prod" oldProd,
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.reg 0 0 lane "y" (.s32 y),
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes yOffset .read yBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        wpInstrList 0 0 [saxpyMul params, saxpyAdd, saxpyStore params]
          (warpAt 0 0 (pc.1, pc.2 + 5) [lane] ∗
            CSL.sepList [
              CSL.globalBytes outOffset .write newOut,
              CSL.reg 0 0 lane "sum" (.s32 sum),
              CSL.reg 0 0 lane "prod" (.s32 prod),
              CSL.reg 0 0 lane "y" (.s32 y),
              CSL.reg 0 0 lane "x" (.s32 x),
              CSL.globalBytes xOffset .read xBytes,
              CSL.globalBytes yOffset .read yBytes]) := by
    exact CSL.entails_trans
      (saxpy_mul_to_add_wp
        (params := params) (pc := (pc.1, pc.2 + 2)) (lane := lane)
        (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
        (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut)
        (oldProd := oldProd) (oldSum := oldSum)
        (x := x) (y := y) (prod := prod) hmul)
      (wpInstr_mono (by simpa [wpInstrList, Nat.add_assoc] using hafterAdd))
  have hafterLoadY :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        CSL.sepList [
          CSL.globalBytes yOffset .read yBytes,
          CSL.reg 0 0 lane "y" oldY,
          CSL.reg 0 0 lane "x" (.s32 x),
          CSL.reg 0 0 lane "prod" oldProd,
          CSL.reg 0 0 lane "sum" oldSum,
          CSL.globalBytes xOffset .read xBytes,
          CSL.globalBytes outOffset .write oldOut]) ⊢ₛ
        wpInstrList 0 0 [saxpyLoadY params, saxpyMul params, saxpyAdd, saxpyStore params]
          (warpAt 0 0 (pc.1, pc.2 + 5) [lane] ∗
            CSL.sepList [
              CSL.globalBytes outOffset .write newOut,
              CSL.reg 0 0 lane "sum" (.s32 sum),
              CSL.reg 0 0 lane "prod" (.s32 prod),
              CSL.reg 0 0 lane "y" (.s32 y),
              CSL.reg 0 0 lane "x" (.s32 x),
              CSL.globalBytes xOffset .read xBytes,
              CSL.globalBytes yOffset .read yBytes]) := by
    exact CSL.entails_trans
      (saxpy_loadY_to_mul_wp
        (params := params) (pc := (pc.1, pc.2 + 1)) (lane := lane)
        (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
        (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut)
        (oldY := oldY) (oldProd := oldProd) (oldSum := oldSum)
        (x := x) (y := y) haddrY haccessY hwidthY hdecodeY)
      (wpInstr_mono (by simpa [wpInstrList, Nat.add_assoc] using hafterMul))
  exact CSL.entails_trans
    (saxpy_loadX_to_loadY_wp
      (params := params) (pc := pc) (lane := lane)
      (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
      (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut)
      (oldX := oldX) (oldY := oldY) (oldProd := oldProd) (oldSum := oldSum)
      (x := x) haddrX haccessX hwidthX hdecodeX)
    (wpInstr_mono (by simpa [wpInstrList, Nat.add_assoc] using hafterLoadY))

def saxpyBlock (params : SaxpyParams) : Block :=
  { label := "entry"
    body := #[saxpyLoadX params, saxpyLoadY params, saxpyMul params, saxpyAdd,
      saxpyStore params]
    term := .terminate }

def saxpyEnv (params : SaxpyParams) : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := params.n } }
    blocks := ({} : Std.HashMap BlockLabel Block).insert "entry" (saxpyBlock params) }

theorem saxpy_entry_lookup (params : SaxpyParams) :
    (saxpyEnv params).blocks["entry"]? = some (saxpyBlock params) := by
  simp [saxpyEnv]

theorem saxpy_block_lookup
    {params : SaxpyParams} {label : BlockLabel} {block : Block}
    (hlookup : (saxpyEnv params).blocks[label]? = some block) :
    label = "entry" ∧ block = saxpyBlock params := by
  by_cases hlabel : label = "entry"
  · subst label
    simp [saxpyEnv] at hlookup
    exact ⟨rfl, hlookup.symm⟩
  · have hbeq : ("entry" == label) = false :=
      (beq_eq_false_iff_ne).2 (fun h => hlabel h.symm)
    simp [saxpyEnv, Std.HashMap.getElem?_insert, hbeq] at hlookup

theorem saxpy_targets_exist (params : SaxpyParams) :
    CFGTerminatorTargetsExist (saxpyEnv params) := by
  intro label block hblock
  rcases saxpy_block_lookup hblock with ⟨_, hblockEq⟩
  subst block
  trivial

theorem saxpy_body_ordinary (params : SaxpyParams) :
    CFGBodyUsesOrdinaryPcAdvance (saxpyEnv params) := by
  intro label block idx gi hblock hgi
  rcases saxpy_block_lookup hblock with ⟨_, hblockEq⟩
  subst block
  cases idx with
  | zero =>
      simp [saxpyBlock, saxpyLoadX, Helpers.instrUsesOrdinaryPcAdvance] at hgi
      subst gi
      rfl
  | succ idx =>
      cases idx with
      | zero =>
          simp [saxpyBlock, saxpyLoadY, Helpers.instrUsesOrdinaryPcAdvance] at hgi
          subst gi
          rfl
      | succ idx =>
          cases idx with
          | zero =>
              simp [saxpyBlock, saxpyMul, Helpers.instrUsesOrdinaryPcAdvance] at hgi
              subst gi
              rfl
          | succ idx =>
              cases idx with
              | zero =>
                  simp [saxpyBlock, saxpyAdd, Helpers.instrUsesOrdinaryPcAdvance] at hgi
                  subst gi
                  rfl
              | succ idx =>
                  cases idx with
                  | zero =>
                      simp [saxpyBlock, saxpyStore, Helpers.instrUsesOrdinaryPcAdvance] at hgi
                      subst gi
                      rfl
                  | succ idx =>
                      simp [saxpyBlock] at hgi

def saxpyScalarPreResources
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut : List Byte)
    (oldX oldY oldProd oldSum : Value) : CSL.Assertion :=
  CSL.sepList [
    CSL.globalBytes xOffset .read xBytes,
    CSL.reg 0 0 lane "x" oldX,
    CSL.reg 0 0 lane "y" oldY,
    CSL.reg 0 0 lane "prod" oldProd,
    CSL.reg 0 0 lane "sum" oldSum,
    CSL.globalBytes yOffset .read yBytes,
    CSL.globalBytes outOffset .write oldOut]

def saxpyScalarAfterXResources
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut : List Byte)
    (oldY oldProd oldSum : Value) (x : Int) : CSL.Assertion :=
  CSL.sepList [
    CSL.globalBytes yOffset .read yBytes,
    CSL.reg 0 0 lane "y" oldY,
    CSL.reg 0 0 lane "x" (.s32 x),
    CSL.reg 0 0 lane "prod" oldProd,
    CSL.reg 0 0 lane "sum" oldSum,
    CSL.globalBytes xOffset .read xBytes,
    CSL.globalBytes outOffset .write oldOut]

def saxpyScalarBeforeStoreResources
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut : List Byte) (x y prod sum : Int) : CSL.Assertion :=
  CSL.sepList [
    CSL.globalBytes outOffset .write oldOut,
    CSL.reg 0 0 lane "sum" (.s32 sum),
    CSL.reg 0 0 lane "prod" (.s32 prod),
    CSL.reg 0 0 lane "y" (.s32 y),
    CSL.reg 0 0 lane "x" (.s32 x),
    CSL.globalBytes xOffset .read xBytes,
    CSL.globalBytes yOffset .read yBytes]

def saxpyScalarFinalResources
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes newOut : List Byte) (x y prod sum : Int) : CSL.Assertion :=
  CSL.sepList [
    CSL.globalBytes outOffset .write newOut,
    CSL.reg 0 0 lane "sum" (.s32 sum),
    CSL.reg 0 0 lane "prod" (.s32 prod),
    CSL.reg 0 0 lane "y" (.s32 y),
    CSL.reg 0 0 lane "x" (.s32 x),
    CSL.globalBytes xOffset .read xBytes,
    CSL.globalBytes yOffset .read yBytes]

theorem saxpyScalarFinalResources_stable_terminate
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes newOut : List Byte) (x y prod sum : Int) :
    CSL.StableUnder (TerminatorStep 0 0 .terminate)
      (saxpyScalarFinalResources lane xOffset yOffset outOffset xBytes yBytes newOut
        x y prod sum) := by
  unfold saxpyScalarFinalResources
  apply CSL.stable_sepList
  intro p hp
  simp at hp
  rcases hp with hp | hp | hp | hp | hp | hp | hp
  · subst p
    exact stable_globalBytes_terminator
  · subst p
    exact stable_reg_terminator
  · subst p
    exact stable_reg_terminator
  · subst p
    exact stable_reg_terminator
  · subst p
    exact stable_reg_terminator
  · subst p
    exact stable_globalBytes_terminator
  · subst p
    exact stable_globalBytes_terminator

def saxpyScalarPre
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut : List Byte)
    (oldX oldY oldProd oldSum : Value) : CSL.Assertion :=
  warpAt 0 0 ("entry", 0) [lane] ∗
    saxpyScalarPreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
      oldX oldY oldProd oldSum

def saxpyScalarPost
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes newOut : List Byte) (x y prod sum : Int) : CSL.Assertion :=
  laneTerminatedAt 0 0 lane ("entry", 5) ∗
    saxpyScalarFinalResources lane xOffset yOffset outOffset xBytes yBytes newOut
      x y prod sum

def saxpyScalarInvariants
    (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut newOut : List Byte)
    (oldX oldY oldProd oldSum : Value) (x y prod sum : Int) : InvariantMap :=
  fun label =>
    if label = "entry" then
      saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum
    else
      saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum

theorem saxpy_scalar_entry_block_vc
    {params : SaxpyParams} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int}
    (haddrX :
      ∀ st r,
        saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 ("entry", 1) [lane] ∗
          saxpyScalarAfterXResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            oldY oldProd oldSum x) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 ("entry", 4) [lane] ∗
          saxpyScalarBeforeStoreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            x y prod sum) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length) :
    blockVC 0 0
      (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
        oldX oldY oldProd oldSum x y prod sum)
      (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum)
      "entry" (saxpyBlock params) := by
  intro st r hinv
  have hpre :
      saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum st r := by
    simpa [saxpyScalarInvariants] using hinv
  have hbody :
      saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum ⊢ₛ
        wpInstrList 0 0
          [saxpyLoadX params, saxpyLoadY params, saxpyMul params, saxpyAdd,
            saxpyStore params]
          (warpAt 0 0 ("entry", 5) [lane] ∗
            saxpyScalarFinalResources lane xOffset yOffset outOffset xBytes yBytes newOut
              x y prod sum) := by
    simpa [saxpyScalarPre, saxpyScalarPreResources, saxpyScalarFinalResources,
      Nat.add_assoc] using
      (saxpy_loads_mul_add_store_wpInstrList
        (params := params) (pc := ("entry", 0)) (lane := lane)
        (xOffset := xOffset) (yOffset := yOffset) (outOffset := outOffset)
        (xBytes := xBytes) (yBytes := yBytes) (oldOut := oldOut) (newOut := newOut)
        (oldX := oldX) (oldY := oldY) (oldProd := oldProd) (oldSum := oldSum)
        (x := x) (y := y) (prod := prod) (sum := sum)
        (by
          intro st r h
          exact haddrX st r (by simpa [saxpyScalarPre, saxpyScalarPreResources] using h))
        haccessX hwidthX hdecodeX
        (by
          intro st r h
          exact haddrY st r (by
            simpa [saxpyScalarAfterXResources] using h))
        haccessY hwidthY hdecodeY hmul hadd
        (by
          intro st r h
          exact haddrOut st r (by
            simpa [saxpyScalarBeforeStoreResources] using h))
        hdisjointX hdisjointY haccessOut hencodeOut hlenOut)
  have hterm :
      (warpAt 0 0 ("entry", 5) [lane] ∗
        saxpyScalarFinalResources lane xOffset yOffset outOffset xBytes yBytes newOut
          x y prod sum) ⊢ₛ
        wpTerminator 0 0 .terminate
          (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut
            x y prod sum) := by
    simpa [saxpyScalarPost] using
      (wp_terminate_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("entry", 5)) (lane := lane)
        (frame := saxpyScalarFinalResources lane xOffset yOffset outOffset
          xBytes yBytes newOut x y prod sum)
        (saxpyScalarFinalResources_stable_terminate
          lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum))
  have hentry :
      saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum ⊢ₛ
        blockEntryWP 0 0
          (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes
            oldOut newOut oldX oldY oldProd oldSum x y prod sum)
          (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut
            x y prod sum)
          (saxpyBlock params) := by
    exact CSL.entails_trans hbody
      (wpInstrList_mono (by simpa [blockTermPost] using hterm))
  exact hentry st r hpre

theorem saxpy_scalar_block_vcs
    {params : SaxpyParams} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int}
    (haddrX :
      ∀ st r,
        saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 ("entry", 1) [lane] ∗
          saxpyScalarAfterXResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            oldY oldProd oldSum x) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 ("entry", 4) [lane] ∗
          saxpyScalarBeforeStoreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            x y prod sum) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length) :
    ∀ label block,
      (saxpyEnv params).blocks[label]? = some block →
        blockVC 0 0
          (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
            oldX oldY oldProd oldSum x y prod sum)
          (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum)
          label block := by
  intro label block hlookup
  rcases saxpy_block_lookup hlookup with ⟨hlabel, hblock⟩
  subst label
  subst block
  exact saxpy_scalar_entry_block_vc
    haddrX haccessX hwidthX hdecodeX haddrY haccessY hwidthY hdecodeY hmul hadd
    haddrOut hdisjointX hdisjointY haccessOut hencodeOut hlenOut

def saxpyConcreteInvariants (params : SaxpyParams) (post : CSL.Assertion) :
    InvariantMap :=
  fun _ => blockEntryWP 0 0 (fun _ => post) post (saxpyBlock params)

theorem saxpy_entry_block_vc (params : SaxpyParams) (post : CSL.Assertion) :
    blockVC 0 0 (saxpyConcreteInvariants params post) post "entry" (saxpyBlock params) := by
  intro st r hentry
  simpa [saxpyConcreteInvariants, blockEntryWP, blockSuffixWP, saxpyBlock, blockTermPost]
    using hentry

theorem saxpy_block_vcs (params : SaxpyParams) (post : CSL.Assertion) :
    ∀ label block,
      (saxpyEnv params).blocks[label]? = some block →
        blockVC 0 0 (saxpyConcreteInvariants params post) post label block := by
  intro label block hlookup
  rcases saxpy_block_lookup hlookup with ⟨hlabel, hblock⟩
  subst label
  subst block
  exact saxpy_entry_block_vc params post

theorem saxpy_br_semantic_control (params : SaxpyParams) :
    BrSemanticControl (saxpyEnv params) 0 0 := by
  intro st st' warpState pc block targetBlock target henv hwarp hlock hrpc hlookup hterm
    htarget hstep
  rcases saxpy_block_lookup hlookup with ⟨_, hblock⟩
  subst block
  simp [saxpyBlock] at hterm

theorem saxpy_cbr_semantic_control (params : SaxpyParams) :
    CbrSemanticControl (saxpyEnv params) 0 0 := by
  intro st st' warpState pc block trueBlock falseBlock cond tLabel fLabel
    henv hwarp hlock hrpc hlookup hterm htrue hfalse hstep
  rcases saxpy_block_lookup hlookup with ⟨_, hblock⟩
  subst block
  simp [saxpyBlock] at hterm

theorem saxpy_scalar_term_preserves
    {params : SaxpyParams} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int}
    (haddrX :
      ∀ st r,
        saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 ("entry", 1) [lane] ∗
          saxpyScalarAfterXResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            oldY oldProd oldSum x) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 ("entry", 4) [lane] ∗
          saxpyScalarBeforeStoreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            x y prod sum) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length) :
    TermStepPreservesKernel (saxpyEnv params) 0 0
      (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
        oldX oldY oldProd oldSum x y prod sum)
      (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum) :=
  TermStepPreservesKernel.of_blockVCs
    (saxpy_scalar_block_vcs
      haddrX haccessX hwidthX hdecodeX haddrY haccessY hwidthY hdecodeY hmul hadd
      haddrOut hdisjointX hdisjointY haccessOut hencodeOut hlenOut)
    (BrTermControl.of_targets_semantic
      (saxpy_targets_exist params) (saxpy_br_semantic_control params))
    (CbrTermControl.of_targets_semantic
      (saxpy_targets_exist params) (saxpy_cbr_semantic_control params))

theorem saxpy_term_preserves
    (params : SaxpyParams) (post : CSL.Assertion) :
    TermStepPreservesKernel (saxpyEnv params) 0 0
      (saxpyConcreteInvariants params post) post :=
  TermStepPreservesKernel.of_blockVCs
    (saxpy_block_vcs params post)
    (BrTermControl.of_targets_semantic
      (saxpy_targets_exist params) (saxpy_br_semantic_control params))
    (CbrTermControl.of_targets_semantic
      (saxpy_targets_exist params) (saxpy_cbr_semantic_control params))

def saxpyScalarKernelPre
    (params : SaxpyParams) (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut : List Byte)
    (oldX oldY oldProd oldSum : Value) : CSL.Assertion :=
  fun st r =>
    st.kernelEnv = saxpyEnv params ∧
      saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum st r

def saxpyScalarKernelInvariant
    (params : SaxpyParams) (lane : LaneId) (xOffset yOffset outOffset : Nat)
    (xBytes yBytes oldOut newOut : List Byte)
    (oldX oldY oldProd oldSum : Value) (x y prod sum : Int) : CSL.Assertion :=
  cfgKernelInvariant (saxpyEnv params) 0 0
    (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
      oldX oldY oldProd oldSum x y prod sum)
    (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum)

def saxpyScalarKernelSpec
    (init : State) (params : SaxpyParams) (lane : LaneId)
    (xOffset yOffset outOffset : Nat) (xBytes yBytes oldOut newOut : List Byte)
    (oldX oldY oldProd oldSum : Value) (x y prod sum : Int) (resource : Resource) :
    KernelSpec :=
  { init := init
    resource := resource
    pre :=
      saxpyScalarKernelPre params lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum
    invariant :=
      saxpyScalarKernelInvariant params lane xOffset yOffset outOffset xBytes yBytes
        oldOut newOut oldX oldY oldProd oldSum x y prod sum
    post := saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum }

theorem saxpy_scalar_pre_entails_kernel_invariant
    {params : SaxpyParams} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int}
    (haddrX :
      ∀ st r,
        saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 ("entry", 1) [lane] ∗
          saxpyScalarAfterXResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            oldY oldProd oldSum x) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 ("entry", 4) [lane] ∗
          saxpyScalarBeforeStoreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            x y prod sum) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length) :
    saxpyScalarKernelPre params lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum ⊢ₛ
      saxpyScalarKernelInvariant params lane xOffset yOffset outOffset xBytes yBytes
        oldOut newOut oldX oldY oldProd oldSum x y prod sum := by
  intro st r hpre
  rcases hpre with ⟨henv, hscalar⟩
  have hscalarPre := hscalar
  rcases hscalar with ⟨rCtrl, _rResources, _hcomp, _hequiv, hctrl, _hresources⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  have hentryBlock :
      (saxpyEnv params).blocks[(saxpyEnv params).entry]? = some (saxpyBlock params) := by
    simp [saxpyEnv]
  have hvc :
      blockVC 0 0
        (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
          oldX oldY oldProd oldSum x y prod sum)
        (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum)
        "entry" (saxpyBlock params) :=
    saxpy_scalar_entry_block_vc
      haddrX haccessX hwidthX hdecodeX haddrY haccessY hwidthY hdecodeY hmul hadd
      haddrOut hdisjointX hdisjointY haccessOut hencodeOut hlenOut
  have hinvEntry :
      saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
        oldX oldY oldProd oldSum x y prod sum "entry" st r := by
    simpa [saxpyScalarInvariants] using hscalarPre
  have hwp :
      blockSuffixWP 0 0
        (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
          oldX oldY oldProd oldSum x y prod sum)
        (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum)
        (saxpyBlock params) 0 st r :=
    blockVC.entry_suffix hvc st r hinvEntry
  exact Or.inl
    ⟨warpState, ((saxpyEnv params).entry, 0), saxpyBlock params, henv, hwarp, hlock,
      by simpa [saxpyEnv] using hrpc, hentryBlock, hwp⟩

theorem saxpy_scalar_kernel_valid
    {init : State} {params : SaxpyParams} {lane : LaneId}
    {xOffset yOffset outOffset : Nat}
    {xBytes yBytes oldOut newOut : List Byte}
    {oldX oldY oldProd oldSum : Value} {x y prod sum : Int} {resource : Resource}
    (hinit :
      saxpyScalarKernelPre params lane xOffset yOffset outOffset xBytes yBytes oldOut
        oldX oldY oldProd oldSum init resource)
    (haddrX :
      ∀ st r,
        saxpyScalarPre lane xOffset yOffset outOffset xBytes yBytes oldOut
          oldX oldY oldProd oldSum st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.xBase }
            (.global xOffset))
    (haccessX : AccessOk .global .s32 (.global xOffset))
    (hwidthX : Typing.byteWidth? .s32 = some xBytes.length)
    (hdecodeX : DecodedScalar .s32 xBytes (.s32 x))
    (haddrY :
      ∀ st r,
        (warpAt 0 0 ("entry", 1) [lane] ∗
          saxpyScalarAfterXResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            oldY oldProd oldSum x) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.yBase }
            (.global yOffset))
    (haccessY : AccessOk .global .s32 (.global yOffset))
    (hwidthY : Typing.byteWidth? .s32 = some yBytes.length)
    (hdecodeY : DecodedScalar .s32 yBytes (.s32 y))
    (hmul : Helpers.evalBinary? .mul (.s32 params.alpha) (.s32 x) = some (.s32 prod))
    (hadd : Helpers.evalBinary? .add (.s32 prod) (.s32 y) = some (.s32 sum))
    (haddrOut :
      ∀ st r,
        (warpAt 0 0 ("entry", 4) [lane] ∗
          saxpyScalarBeforeStoreResources lane xOffset yOffset outOffset xBytes yBytes oldOut
            x y prod sum) st r →
          ResolvesAddr st { cta := 0, warp := 0, lane := lane }
            { space := .global, ty := .s32, addr := saxpyGlobalLaneAddr params.outBase }
            (.global outOffset))
    (hdisjointX : ByteRangesDisjoint xOffset xBytes.length outOffset newOut.length)
    (hdisjointY : ByteRangesDisjoint yOffset yBytes.length outOffset newOut.length)
    (haccessOut : AccessOk .global .s32 (.global outOffset))
    (hencodeOut : EncodedScalar .s32 (.s32 sum) newOut)
    (hlenOut : oldOut.length = newOut.length)
    (honly :
      cfgKernelInvariant (saxpyEnv params) 0 0
        (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
          oldX oldY oldProd oldSum x y prod sum)
        (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hpostNoStep :
      NoStepBlock 0 0
        (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant (saxpyEnv params) 0 0
          (saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
            oldX oldY oldProd oldSum x y prod sum)
          (saxpyScalarPost lane xOffset yOffset outOffset xBytes yBytes newOut x y prod sum))) :
    (saxpyScalarKernelSpec init params lane xOffset yOffset outOffset xBytes yBytes
      oldOut newOut oldX oldY oldProd oldSum x y prod sum resource).Valid :=
  KernelSpec.Valid.of_entry_blockVCs_targets_closed
    (spec := saxpyScalarKernelSpec init params lane xOffset yOffset outOffset xBytes yBytes
      oldOut newOut oldX oldY oldProd oldSum x y prod sum resource)
    (env := saxpyEnv params) (cta := 0) (warp := 0)
    (invariants :=
      saxpyScalarInvariants lane xOffset yOffset outOffset xBytes yBytes oldOut newOut
        oldX oldY oldProd oldSum x y prod sum)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r hpre
      rcases hpre with ⟨_henv, hscalar⟩
      simpa [saxpyScalarInvariants, saxpyEnv] using hscalar)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨henv, hscalar⟩
      rcases hscalar with ⟨_rCtrl, _rResources, _hcomp, _hequiv, hctrl, _hresources⟩
      rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
      exact ⟨warpState, saxpyBlock params, henv, hwarp, hlock,
        by simpa [saxpyEnv] using hrpc, by simp [saxpyEnv]⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics (saxpy_body_ordinary params))
    (hblocks :=
      saxpy_scalar_block_vcs
        haddrX haccessX hwidthX hdecodeX haddrY haccessY hwidthY hdecodeY hmul hadd
        haddrOut hdisjointX hdisjointY haccessOut hencodeOut hlenOut)
    (htargets := saxpy_targets_exist params)
    (hbr := saxpy_br_semantic_control params)
    (hcbr := saxpy_cbr_semantic_control params)
    (hpostNoStep := hpostNoStep)
    (hsuffixNoFinal := hsuffixNoFinal)

def saxpyPost
    (params : SaxpyParams) (xs ys : Nat → Int) (st : State) (_r : Resource) : Prop :=
  ∀ i, i < params.n →
    readGlobalS32? st (saxpyLaneOutputOffset params i) =
      some (saxpyExpectedValue params (xs i) (ys i))

def saxpyKernelSpec
    (init : State) (params : SaxpyParams) (xs ys : Nat → Int) (resource : Resource) :
    KernelSpec :=
  { init := init
    resource := resource
    pre := fun st r => st = init ∧ r = resource
    post := saxpyPost params xs ys }

theorem saxpy_partial_correct_of_valid
    {init : State} {params : SaxpyParams} {xs ys : Nat → Int} {resource : Resource}
    (hvalid : (saxpyKernelSpec init params xs ys resource).Valid) :
    PartialCorrect init (fun st => saxpyPost params xs ys st resource) :=
by
  intro final hterm
  rcases KernelSpec.partial_correct hvalid final hterm with ⟨_, hpost⟩
  exact hpost

end Examples
end CLean
