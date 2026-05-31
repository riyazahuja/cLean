import CLean.WP

namespace CLean
namespace Examples

open CSL WP

private def readGlobalS32? (st : State) (off : Nat) : Option Int := do
  match Helpers.readMem? st .global .s32 (.global off) with
  | some (.s32 x) => some x
  | _ => none

structure MatmulCellParams where
  n : Nat
  row : Nat
  col : Nat
  aBase : Nat
  bBase : Nat
  cBase : Nat
  deriving Repr, Inhabited

def matmulCellOffset (params : MatmulCellParams) : Nat :=
  params.cBase + (params.row * params.n + params.col) * 4

def matmulCellValue (params : MatmulCellParams) (a b : Nat → Nat → Int) : Int :=
  (List.range params.n).foldl
    (fun acc k => acc + a params.row k * b k params.col)
    0

def matmulAOffset (params : MatmulCellParams) (k : Nat) : Nat :=
  params.aBase + (params.row * params.n + k) * 4

def matmulBOffset (params : MatmulCellParams) (k : Nat) : Nat :=
  params.bBase + (k * params.n + params.col) * 4

def matmulGlobalAddr (offset : Nat) : RValue :=
  .imm (.gaddr .global offset)

def matmulLoadA0 (params : MatmulCellParams) : GInstr :=
  { guard? := none
    instr := .load "a0"
      { space := .global, ty := .s32, addr := matmulGlobalAddr (matmulAOffset params 0) } }

def matmulLoadB0 (params : MatmulCellParams) : GInstr :=
  { guard? := none
    instr := .load "b0"
      { space := .global, ty := .s32, addr := matmulGlobalAddr (matmulBOffset params 0) } }

def matmulMul0 : GInstr :=
  { guard? := none
    instr := .assignReg "acc"
      (.binop .mul (.reg "a0") (.reg "b0")) }

theorem matmul_loadA0_lanes_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes : List Byte} {oldA : Value} {a0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecode : DecodedScalar .s32 aBytes (.s32 a0)) :
    (warpAt 0 0 pc [lane] ∗
      (globalSlices [matmulAOffset params 0] .read [aBytes] ∗
        regsFor 0 0 [lane] "a0" [oldA])) ⊢ₛ
      wpInstr 0 0 (matmulLoadA0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (globalSlices [matmulAOffset params 0] .read [aBytes] ∗
            regsFor 0 0 [lane] "a0" [.s32 a0])) := by
  simpa [matmulLoadA0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_lanes_warpAt
      (cta := 0) (warp := 0) (pc := pc) (dst := "a0") (ty := .s32)
      (addrExpr := matmulGlobalAddr (matmulAOffset params 0))
      (lanes := [lane]) (offsets := [matmulAOffset params 0]) (byteSlices := [aBytes])
      (oldValues := [oldA]) (newValues := [.s32 a0])
      (by
        intro st _r _hpre
        exact ⟨resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulAOffset params 0)))
          (off := matmulAOffset params 0) (by rfl), True.intro⟩)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨rSlices, _rRegs, _hcompRest, _hequivRest, hslices, _hregs⟩
        change
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            globalSlices [] .read []) st rSlices at hslices
        rcases hslices with ⟨_rBytes, _rEmpty, _hcompBytes, _hequivBytes, hbytes, _hempty⟩
        exact ⟨globalReadMem_of_globalBytes haccess hwidth hbytes hdecode, True.intro⟩))

theorem matmul_loadA0_live_lanes_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut : List Byte} {oldA oldB oldAcc : Value} {a0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecode : DecodedScalar .s32 aBytes (.s32 a0)) :
    ((warpAt 0 0 pc [lane] ∗
      (globalSlices [matmulAOffset params 0] .read [aBytes] ∗
        regsFor 0 0 [lane] "a0" [oldA])) ∗
      (regsFor 0 0 [lane] "b0" [oldB] ∗
        (regsFor 0 0 [lane] "acc" [oldAcc] ∗
          (globalSlices [matmulBOffset params 0] .read [bBytes] ∗
            globalSlices [matmulCellOffset params] .write [oldOut])))) ⊢ₛ
      wpInstr 0 0 (matmulLoadA0 params)
        ((warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (globalSlices [matmulAOffset params 0] .read [aBytes] ∗
            regsFor 0 0 [lane] "a0" [.s32 a0])) ∗
          (regsFor 0 0 [lane] "b0" [oldB] ∗
            (regsFor 0 0 [lane] "acc" [oldAcc] ∗
              (globalSlices [matmulBOffset params 0] .read [bBytes] ∗
                globalSlices [matmulCellOffset params] .write [oldOut])))) := by
  simpa [matmulLoadA0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_lanes_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "a0") (ty := .s32)
      (addrExpr := matmulGlobalAddr (matmulAOffset params 0))
      (lanes := [lane]) (offsets := [matmulAOffset params 0]) (byteSlices := [aBytes])
      (oldValues := [oldA]) (newValues := [.s32 a0])
      (frame :=
        regsFor 0 0 [lane] "b0" [oldB] ∗
          (regsFor 0 0 [lane] "acc" [oldAcc] ∗
            (globalSlices [matmulBOffset params 0] .read [bBytes] ∗
              globalSlices [matmulCellOffset params] .write [oldOut])))
      (by
        intro st _r _hpre
        exact ⟨resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulAOffset params 0)))
          (off := matmulAOffset params 0) (by rfl), True.intro⟩)
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨rSlices, _rRegs, _hcompRest, _hequivRest, hslices, _hregs⟩
        change
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            globalSlices [] .read []) st rSlices at hslices
        rcases hslices with ⟨_rBytes, _rEmpty, _hcompBytes, _hequivBytes, hbytes, _hempty⟩
        exact ⟨globalReadMem_of_globalBytes haccess hwidth hbytes hdecode, True.intro⟩)
      (by
        exact CSL.stable_sep
          (stable_regsFor_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_regsFor_load_of_ne (by decide))
            (CSL.stable_sep stable_globalSlices_load stable_globalSlices_load))))

theorem matmul_mul0_lanes_wp
    {pc : PC} {lane : LaneId} {oldAcc : Value} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "acc" [oldAcc] ∗
        (regsFor 0 0 [lane] "a0" [.s32 a0] ∗
          regsFor 0 0 [lane] "b0" [.s32 b0]))) ⊢ₛ
      wpInstr 0 0 matmulMul0
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "acc" [.s32 acc] ∗
            (regsFor 0 0 [lane] "a0" [.s32 a0] ∗
              regsFor 0 0 [lane] "b0" [.s32 b0]))) := by
  simpa [matmulMul0] using
    (wp_assignReg_lanes_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc) (dst := "acc")
      (rhs := .binop .mul (.reg "a0") (.reg "b0"))
      (lanes := [lane]) (oldValues := [oldAcc]) (newValues := [.s32 acc])
      (frame :=
        regsFor 0 0 [lane] "a0" [.s32 a0] ∗
          regsFor 0 0 [lane] "b0" [.s32 b0])
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨rA, rB, _hcompSrc, _hequivSrc, haRegs, hbRegs⟩
        change (CSL.reg 0 0 lane "a0" (.s32 a0) ∗ regsFor 0 0 [] "a0" []) st rA
          at haRegs
        change (CSL.reg 0 0 lane "b0" (.s32 b0) ∗ regsFor 0 0 [] "b0" []) st rB
          at hbRegs
        rcases haRegs with ⟨_rAHead, _rAEmpty, _hcompA, _hequivA, ha, _haEmpty⟩
        rcases hbRegs with ⟨_rBHead, _rBEmpty, _hcompB, _hequivB, hb, _hbEmpty⟩
        exact ⟨eval_binop_of_eval (eval_reg_of_assertion ha)
          (eval_reg_of_assertion hb) hmul, True.intro⟩)
      (by
        intro st st' _r rFrame _hpre hframe hstep
        exact (CSL.stable_sep
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "acc")
            (name := "a0") (rhs := .binop .mul (.reg "a0") (.reg "b0"))
            (lanes := [lane]) (values := [.s32 a0]) (by decide))
          (stable_regsFor_assignReg_of_ne
            (cta := 0) (warp := 0) (guard? := none) (dst := "acc")
            (name := "b0") (rhs := .binop .mul (.reg "a0") (.reg "b0"))
            (lanes := [lane]) (values := [.s32 b0]) (by decide)))
          st st' rFrame hstep hframe))

theorem matmul_arith_lanes_wpInstrList
    {pc : PC} {lane : LaneId} {oldAcc : Value} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (regsFor 0 0 [lane] "acc" [oldAcc] ∗
        (regsFor 0 0 [lane] "a0" [.s32 a0] ∗
          regsFor 0 0 [lane] "b0" [.s32 b0]))) ⊢ₛ
      wpInstrList 0 0 [matmulMul0]
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (regsFor 0 0 [lane] "acc" [.s32 acc] ∗
            (regsFor 0 0 [lane] "a0" [.s32 a0] ∗
              regsFor 0 0 [lane] "b0" [.s32 b0]))) := by
  simpa [wpInstrList] using
    (matmul_mul0_lanes_wp (pc := pc) (lane := lane) (oldAcc := oldAcc)
      (a0 := a0) (b0 := b0) (acc := acc) hmul)

theorem matmul_loadA0_acc_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes : List Byte} {oldA oldAcc : Value} {a0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecode : DecodedScalar .s32 aBytes (.s32 a0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        CSL.reg 0 0 lane "acc" oldAcc)) ⊢ₛ
      wpInstr 0 0 (matmulLoadA0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            CSL.reg 0 0 lane "a0" (.s32 a0)) ∗
            CSL.reg 0 0 lane "acc" oldAcc)) := by
  simpa [matmulLoadA0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "a0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulAOffset params 0))
      (lane := lane) (offset := matmulAOffset params 0)
      (bytes := aBytes) (oldReg := oldA) (value := .s32 a0)
      (frame := CSL.reg 0 0 lane "acc" oldAcc)
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulAOffset params 0)))
          (off := matmulAOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rAcc, _hcompRest, _hequivRest, hmemReg, _hacc⟩
        rcases hmemReg with ⟨_rMem, _rA, _hcompMemReg, _hequivMemReg, hbytes, _ha⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (stable_reg_load_of_ne (by decide)))

theorem matmul_loadA0_preserves_b0_acc_bbytes_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes : List Byte} {oldA oldB oldAcc : Value} {a0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecode : DecodedScalar .s32 aBytes (.s32 a0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        (CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) ⊢ₛ
      wpInstr 0 0 (matmulLoadA0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            CSL.reg 0 0 lane "a0" (.s32 a0)) ∗
            (CSL.reg 0 0 lane "b0" oldB ∗
              (CSL.reg 0 0 lane "acc" oldAcc ∗
                CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) := by
  simpa [matmulLoadA0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "a0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulAOffset params 0))
      (lane := lane) (offset := matmulAOffset params 0)
      (bytes := aBytes) (oldReg := oldA) (value := .s32 a0)
      (frame :=
        CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulAOffset params 0)))
          (off := matmulAOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframe⟩
        rcases hmemReg with ⟨_rMem, _rA, _hcompMemReg, _hequivMemReg, hbytes, _ha⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (by
        exact CSL.stable_sep
          (stable_reg_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_load_of_ne (by decide))
            stable_globalBytes_load)))

theorem matmul_loadA0_preserves_b0_acc_bbytes_out_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut : List Byte} {oldA oldB oldAcc : Value} {a0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecode : DecodedScalar .s32 aBytes (.s32 a0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        (CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
              CSL.globalBytes (matmulCellOffset params) .write oldOut))))) ⊢ₛ
      wpInstr 0 0 (matmulLoadA0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            CSL.reg 0 0 lane "a0" (.s32 a0)) ∗
            (CSL.reg 0 0 lane "b0" oldB ∗
              (CSL.reg 0 0 lane "acc" oldAcc ∗
                (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                  CSL.globalBytes (matmulCellOffset params) .write oldOut))))) := by
  simpa [matmulLoadA0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "a0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulAOffset params 0))
      (lane := lane) (offset := matmulAOffset params 0)
      (bytes := aBytes) (oldReg := oldA) (value := .s32 a0)
      (frame :=
        CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
              CSL.globalBytes (matmulCellOffset params) .write oldOut)))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulAOffset params 0)))
          (off := matmulAOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframe⟩
        rcases hmemReg with ⟨_rMem, _rA, _hcompMemReg, _hequivMemReg, hbytes, _ha⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (by
        exact CSL.stable_sep
          (stable_reg_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_load_of_ne (by decide))
            (CSL.stable_sep stable_globalBytes_load stable_globalBytes_load))))

theorem matmul_loadB0_preserves_a0_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes : List Byte} {oldB : Value} {a0 b0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecode : DecodedScalar .s32 bBytes (.s32 b0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
        CSL.reg 0 0 lane "b0" oldB) ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          CSL.globalBytes (matmulAOffset params 0) .read aBytes))) ⊢ₛ
      wpInstr 0 0 (matmulLoadB0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
            CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              CSL.globalBytes (matmulAOffset params 0) .read aBytes))) := by
  simpa [matmulLoadB0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "b0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulBOffset params 0))
      (lane := lane) (offset := matmulBOffset params 0)
      (bytes := bBytes) (oldReg := oldB) (value := .s32 b0)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          CSL.globalBytes (matmulAOffset params 0) .read aBytes)
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulBOffset params 0)))
          (off := matmulBOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframe⟩
        rcases hmemReg with ⟨_rMem, _rB, _hcompMemReg, _hequivMemReg, hbytes, _hb⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (by
        exact CSL.stable_sep
          (stable_reg_load_of_ne (by decide))
          stable_globalBytes_load))

theorem matmul_loadB0_preserves_a0_acc_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes : List Byte} {oldB oldAcc : Value} {a0 b0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecode : DecodedScalar .s32 bBytes (.s32 b0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
        CSL.reg 0 0 lane "b0" oldB) ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) ⊢ₛ
      wpInstr 0 0 (matmulLoadB0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
            CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "acc" oldAcc ∗
                CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) := by
  simpa [matmulLoadB0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "b0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulBOffset params 0))
      (lane := lane) (offset := matmulBOffset params 0)
      (bytes := bBytes) (oldReg := oldB) (value := .s32 b0)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulAOffset params 0) .read aBytes))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulBOffset params 0)))
          (off := matmulBOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframe⟩
        rcases hmemReg with ⟨_rMem, _rB, _hcompMemReg, _hequivMemReg, hbytes, _hb⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (by
        exact CSL.stable_sep
          (stable_reg_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_load_of_ne (by decide))
            stable_globalBytes_load)))

theorem matmul_loadB0_preserves_a0_acc_abytes_out_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut : List Byte} {oldB oldAcc : Value} {a0 b0 : Int}
    (haccess : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidth : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecode : DecodedScalar .s32 bBytes (.s32 b0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
        CSL.reg 0 0 lane "b0" oldB) ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              CSL.globalBytes (matmulCellOffset params) .write oldOut))))) ⊢ₛ
      wpInstr 0 0 (matmulLoadB0 params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
            CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "acc" oldAcc ∗
                (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                  CSL.globalBytes (matmulCellOffset params) .write oldOut))))) := by
  simpa [matmulLoadB0, matmulGlobalAddr] using
    (wp_globalLoadBytesReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "b0")
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulBOffset params 0))
      (lane := lane) (offset := matmulBOffset params 0)
      (bytes := bBytes) (oldReg := oldB) (value := .s32 b0)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              CSL.globalBytes (matmulCellOffset params) .write oldOut)))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulBOffset params 0)))
          (off := matmulBOffset params 0) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rMemReg, _rFrame, _hcompRest, _hequivRest, hmemReg, _hframe⟩
        rcases hmemReg with ⟨_rMem, _rB, _hcompMemReg, _hequivMemReg, hbytes, _hb⟩
        exact globalReadMem_of_globalBytes haccess hwidth hbytes hdecode)
      (by
        exact CSL.stable_sep
          (stable_reg_load_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_load_of_ne (by decide))
            (CSL.stable_sep stable_globalBytes_load stable_globalBytes_load))))

theorem matmul_loads_wpInstrList
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes : List Byte} {oldA oldB oldAcc : Value} {a0 b0 : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        (CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) ⊢ₛ
      wpInstrList 0 0 [matmulLoadA0 params, matmulLoadB0 params]
        (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
          ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
            CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "acc" oldAcc ∗
                CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) := by
  have hafterA :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
          CSL.reg 0 0 lane "a0" (.s32 a0)) ∗
          (CSL.reg 0 0 lane "b0" oldB ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗
              CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) ⊢ₛ
        wpInstr 0 0 (matmulLoadB0 params)
          (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
            ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
              CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
              (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                (CSL.reg 0 0 lane "acc" oldAcc ∗
                  CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_ab_cde_ecbda
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes)
          (CSL.reg 0 0 lane "a0" (.s32 a0))
          (CSL.reg 0 0 lane "b0" oldB)
          (CSL.reg 0 0 lane "acc" oldAcc)
          (CSL.globalBytes (matmulBOffset params 0) .read bBytes)))
      (by
        simpa [Nat.add_assoc] using
          (matmul_loadB0_preserves_a0_acc_wp
            (params := params) (pc := (pc.1, pc.2 + 1)) (lane := lane)
            (aBytes := aBytes) (bBytes := bBytes) (oldB := oldB)
            (oldAcc := oldAcc) (a0 := a0) (b0 := b0)
            haccessB hwidthB hdecodeB))
  exact CSL.entails_trans
    (matmul_loadA0_preserves_b0_acc_bbytes_wp
      (params := params) (pc := pc) (lane := lane)
      (aBytes := aBytes) (bBytes := bBytes) (oldA := oldA)
      (oldB := oldB) (oldAcc := oldAcc) (a0 := a0)
      haccessA hwidthA hdecodeA)
    (wpInstr_mono (by simpa [wpInstrList] using hafterA))

theorem matmul_mul0_wp
    {pc : PC} {lane : LaneId} {oldAcc : Value} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          CSL.reg 0 0 lane "b0" (.s32 b0)))) ⊢ₛ
      wpInstr 0 0 matmulMul0
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              CSL.reg 0 0 lane "b0" (.s32 b0)))) := by
  simpa [matmulMul0] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "acc")
      (rhs := .binop .mul (.reg "a0") (.reg "b0")) (lane := lane)
      (old := oldAcc) (new := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          CSL.reg 0 0 lane "b0" (.s32 b0))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rA, _rB, _hcompSrc, _hequivSrc, ha, hb⟩
        exact eval_binop_of_eval (eval_reg_of_assertion ha)
          (eval_reg_of_assertion hb) hmul)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (stable_reg_assignReg_of_ne (by decide))))

theorem matmul_mul0_preserves_a0_bytes_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {oldAcc : Value} {aBytes : List Byte} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) ⊢ₛ
      wpInstr 0 0 matmulMul0
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) := by
  simpa [matmulMul0] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "acc")
      (rhs := .binop .mul (.reg "a0") (.reg "b0")) (lane := lane)
      (old := oldAcc) (new := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            CSL.globalBytes (matmulAOffset params 0) .read aBytes))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rA, _rRestFrame, _hcompA, _hequivA, ha, hrestFrame⟩
        rcases hrestFrame with ⟨_rB, _rBytes, _hcompB, _hequivB, hb, _hbytes⟩
        exact eval_binop_of_eval (eval_reg_of_assertion ha)
          (eval_reg_of_assertion hb) hmul)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            stable_globalBytes_assignReg)))

theorem matmul_mul0_preserves_ab_bytes_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {oldAcc : Value} {aBytes bBytes : List Byte} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              CSL.globalBytes (matmulBOffset params 0) .read bBytes))))) ⊢ₛ
      wpInstr 0 0 matmulMul0
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                  CSL.globalBytes (matmulBOffset params 0) .read bBytes))))) := by
  simpa [matmulMul0] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "acc")
      (rhs := .binop .mul (.reg "a0") (.reg "b0")) (lane := lane)
      (old := oldAcc) (new := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              CSL.globalBytes (matmulBOffset params 0) .read bBytes)))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rA, _rRestFrame, _hcompA, _hequivA, ha, hrestFrame⟩
        rcases hrestFrame with ⟨_rB, _rBytes, _hcompB, _hequivB, hb, _hbytes⟩
        exact eval_binop_of_eval (eval_reg_of_assertion ha)
          (eval_reg_of_assertion hb) hmul)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (CSL.stable_sep stable_globalBytes_assignReg stable_globalBytes_assignReg))))

theorem matmul_mul0_preserves_ab_out_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {oldAcc : Value} {aBytes bBytes oldOut : List Byte} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                CSL.globalBytes (matmulCellOffset params) .write oldOut)))))) ⊢ₛ
      wpInstr 0 0 matmulMul0
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                  (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                    CSL.globalBytes (matmulCellOffset params) .write oldOut)))))) := by
  simpa [matmulMul0] using
    (wp_assignReg_single_warpAt_stableFrame
      (cta := 0) (warp := 0) (pc := pc) (dst := "acc")
      (rhs := .binop .mul (.reg "a0") (.reg "b0")) (lane := lane)
      (old := oldAcc) (new := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
            (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
              (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                CSL.globalBytes (matmulCellOffset params) .write oldOut))))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rDst, _rFrame, _hcompRest, _hequivRest, _hdst, hframe⟩
        rcases hframe with ⟨_rA, _rRestFrame, _hcompA, _hequivA, ha, hrestFrame⟩
        rcases hrestFrame with ⟨_rB, _rBytes, _hcompB, _hequivB, hb, _hbytes⟩
        exact eval_binop_of_eval (eval_reg_of_assertion ha)
          (eval_reg_of_assertion hb) hmul)
      (by
        exact CSL.stable_sep
          (stable_reg_assignReg_of_ne (by decide))
          (CSL.stable_sep
            (stable_reg_assignReg_of_ne (by decide))
            (CSL.stable_sep stable_globalBytes_assignReg
              (CSL.stable_sep stable_globalBytes_assignReg stable_globalBytes_assignReg)))))

theorem matmul_arith_wpInstrList
    {pc : PC} {lane : LaneId} {oldAcc : Value} {a0 b0 acc : Int}
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
          CSL.reg 0 0 lane "b0" (.s32 b0)))) ⊢ₛ
      wpInstrList 0 0 [matmulMul0]
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              CSL.reg 0 0 lane "b0" (.s32 b0)))) := by
  simpa [wpInstrList] using
    (matmul_mul0_wp (pc := pc) (lane := lane) (oldAcc := oldAcc)
      (a0 := a0) (b0 := b0) (acc := acc) hmul)

theorem matmul_loads_mul_wpInstrList
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes : List Byte} {oldA oldB oldAcc : Value} {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc)) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        (CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) ⊢ₛ
      wpInstrList 0 0 [matmulLoadA0 params, matmulLoadB0 params, matmulMul0]
        (warpAt 0 0 (pc.1, pc.2 + 3) [lane] ∗
          (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
            (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
              (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                  CSL.globalBytes (matmulBOffset params 0) .read bBytes))))) := by
  have hafterLoads :
      (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
        ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
          CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
          (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗
              CSL.globalBytes (matmulAOffset params 0) .read aBytes)))) ⊢ₛ
        wpInstr 0 0 matmulMul0
          (warpAt 0 0 (pc.1, pc.2 + 3) [lane] ∗
            (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
              (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                  (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                    CSL.globalBytes (matmulBOffset params 0) .read bBytes))))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_ab_cde_dcb_ea
          (CSL.globalBytes (matmulBOffset params 0) .read bBytes)
          (CSL.reg 0 0 lane "b0" (.s32 b0))
          (CSL.reg 0 0 lane "a0" (.s32 a0))
          (CSL.reg 0 0 lane "acc" oldAcc)
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes)))
      (by
        simpa [Nat.add_assoc] using
          (matmul_mul0_preserves_ab_bytes_wp
            (params := params) (pc := (pc.1, pc.2 + 2)) (lane := lane)
            (oldAcc := oldAcc) (aBytes := aBytes) (bBytes := bBytes)
            (a0 := a0) (b0 := b0) (acc := acc) hmul))
  exact CSL.entails_trans
    (matmul_loads_wpInstrList
      (params := params) (pc := pc) (lane := lane)
      (aBytes := aBytes) (bBytes := bBytes) (oldA := oldA)
      (oldB := oldB) (oldAcc := oldAcc) (a0 := a0) (b0 := b0)
      haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB)
    (wpInstrList_mono (by simpa [wpInstrList] using hafterLoads))

def matmulStoreCell (params : MatmulCellParams) : GInstr :=
  { guard? := none
    instr := .store
      { space := .global, ty := .s32, addr := matmulGlobalAddr (matmulCellOffset params) }
      (.reg "acc") }

theorem matmul_store_acc_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {oldOut newOut : List Byte} {acc : Int}
    (haccess : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencode : EncodedScalar .s32 (.s32 acc) newOut)
    (hlen : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes (matmulCellOffset params) .write oldOut ∗
        CSL.reg 0 0 lane "acc" (.s32 acc))) ⊢ₛ
      wpInstr 0 0 (matmulStoreCell params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
            CSL.reg 0 0 lane "acc" (.s32 acc))) := by
  simpa [matmulStoreCell, matmulGlobalAddr] using
    (wp_globalStoreBytes_single_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc)
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulCellOffset params))
      (valueExpr := .reg "acc") (lane := lane) (offset := matmulCellOffset params)
      (oldBytes := oldOut) (newBytes := newOut) (value := .s32 acc)
      (frame := CSL.reg 0 0 lane "acc" (.s32 acc))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rOut, _rAcc, _hcompRest, _hequivRest, _hout, hacc⟩
        exact eval_reg_of_assertion hacc)
      (by
        intro st r _hpre
        exact ⟨{ st with global := {
            bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } },
          globalWriteMem_of_byteWrite haccess hencode (by rfl)⟩)
      hencode hlen
      (by
        intro st st' r rFrame hpre hacc hstep
        have haddr' := resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl)
        have heval' : EvalRValue st { cta := 0, warp := 0, lane := lane }
            (.reg "acc") (.s32 acc) := by
          exact eval_reg_of_assertion hacc
        have hwrite' : WriteMemFact st .global .s32 (.global (matmulCellOffset params))
            (.s32 acc)
            { st with global := {
                bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } } :=
          globalWriteMem_of_byteWrite haccess hencode (by rfl)
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
        exact globalStorePreservesReadReg_single_warpAt
          hctrl hacc haddr' heval' hwrite' hstep))

theorem matmul_store_acc_preserves_a0_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes oldOut newOut : List Byte} {acc : Int}
    (hdisjoint :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (haccess : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencode : EncodedScalar .s32 (.s32 acc) newOut)
    (hlen : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes (matmulCellOffset params) .write oldOut ∗
        (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          CSL.globalBytes (matmulAOffset params 0) .read aBytes))) ⊢ₛ
      wpInstr 0 0 (matmulStoreCell params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
            (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
              CSL.globalBytes (matmulAOffset params 0) .read aBytes))) := by
  simpa [matmulStoreCell, matmulGlobalAddr] using
    (wp_globalStoreBytes_single_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc)
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulCellOffset params))
      (valueExpr := .reg "acc") (lane := lane) (offset := matmulCellOffset params)
      (oldBytes := oldOut) (newBytes := newOut) (value := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          CSL.globalBytes (matmulAOffset params 0) .read aBytes)
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
        rcases hframe with ⟨_rAcc, _rA, _hcompFrame, _hequivFrame, hacc, _ha⟩
        exact eval_reg_of_assertion hacc)
      (by
        intro st r _hpre
        exact ⟨{ st with global := {
            bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } },
          globalWriteMem_of_byteWrite haccess hencode (by rfl)⟩)
      hencode hlen
      (by
        intro st st' r rFrame hpre hframe hstep
        have haddr' := resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl)
        rcases hframe with ⟨rAcc, rA, hcompFrame, hequivFrame, hacc, ha⟩
        have heval' : EvalRValue st { cta := 0, warp := 0, lane := lane }
            (.reg "acc") (.s32 acc) := by
          exact eval_reg_of_assertion hacc
        have hwrite' : WriteMemFact st .global .s32 (.global (matmulCellOffset params))
            (.s32 acc)
            { st with global := {
                bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } } :=
          globalWriteMem_of_byteWrite haccess hencode (by rfl)
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
        exact ⟨rAcc, rA, hcompFrame, hequivFrame,
          globalStorePreservesReadReg_single_warpAt
            hctrl hacc haddr' heval' hwrite' hstep,
          globalStorePreservesGlobalBytes_single_warpAt hdisjoint
            hctrl ha haddr' heval' hwrite' hencode hstep⟩))

theorem matmul_store_acc_preserves_ab_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {acc : Int}
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccess : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencode : EncodedScalar .s32 (.s32 acc) newOut)
    (hlen : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes (matmulCellOffset params) .write oldOut ∗
        (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) ⊢ₛ
      wpInstr 0 0 (matmulStoreCell params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
            (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
              (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                CSL.globalBytes (matmulBOffset params 0) .read bBytes)))) := by
  simpa [matmulStoreCell, matmulGlobalAddr] using
    (wp_globalStoreBytes_single_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc)
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulCellOffset params))
      (valueExpr := .reg "acc") (lane := lane) (offset := matmulCellOffset params)
      (oldBytes := oldOut) (newBytes := newOut) (value := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
            CSL.globalBytes (matmulBOffset params 0) .read bBytes))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
        rcases hframe with ⟨_rAcc, _rBytes, _hcompFrame, _hequivFrame, hacc, _hbytes⟩
        exact eval_reg_of_assertion hacc)
      (by
        intro st r _hpre
        exact ⟨{ st with global := {
            bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } },
          globalWriteMem_of_byteWrite haccess hencode (by rfl)⟩)
      hencode hlen
      (by
        intro st st' r rFrame hpre hframe hstep
        have haddr' := resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl)
        rcases hframe with ⟨rAcc, rBytes, hcompFrame, hequivFrame, hacc, hbytes⟩
        rcases hbytes with ⟨rA, rB, hcompBytes, hequivBytes, ha, hb⟩
        have heval' : EvalRValue st { cta := 0, warp := 0, lane := lane }
            (.reg "acc") (.s32 acc) := by
          exact eval_reg_of_assertion hacc
        have hwrite' : WriteMemFact st .global .s32 (.global (matmulCellOffset params))
            (.s32 acc)
            { st with global := {
                bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } } :=
          globalWriteMem_of_byteWrite haccess hencode (by rfl)
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
        exact ⟨rAcc, rBytes, hcompFrame, hequivFrame,
          globalStorePreservesReadReg_single_warpAt
            hctrl hacc haddr' heval' hwrite' hstep,
          ⟨rA, rB, hcompBytes, hequivBytes,
            globalStorePreservesGlobalBytes_single_warpAt hdisjointA
              hctrl ha haddr' heval' hwrite' hencode hstep,
            globalStorePreservesGlobalBytes_single_warpAt hdisjointB
              hctrl hb haddr' heval' hwrite' hencode hstep⟩⟩))

theorem matmul_store_acc_preserves_regs_ab_wp
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {a0 b0 acc : Int}
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccess : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencode : EncodedScalar .s32 (.s32 acc) newOut)
    (hlen : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      (CSL.globalBytes (matmulCellOffset params) .write oldOut ∗
        (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
            (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
              (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) ⊢ₛ
      wpInstr 0 0 (matmulStoreCell params)
        (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
          (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
            (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
              (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                  (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                    CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) := by
  simpa [matmulStoreCell, matmulGlobalAddr] using
    (wp_globalStoreBytes_single_warpAt_frame
      (cta := 0) (warp := 0) (pc := pc)
      (ty := .s32) (addrExpr := matmulGlobalAddr (matmulCellOffset params))
      (valueExpr := .reg "acc") (lane := lane) (offset := matmulCellOffset params)
      (oldBytes := oldOut) (newBytes := newOut) (value := .s32 acc)
      (frame :=
        CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
            (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
              (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                CSL.globalBytes (matmulBOffset params 0) .read bBytes))))
      (by
        intro st r _hpre
        exact resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl))
      (by
        intro st r hpre
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, _hctrl, hrest⟩
        rcases hrest with ⟨_rOut, _rFrame, _hcompRest, _hequivRest, _hout, hframe⟩
        rcases hframe with ⟨_rAcc, _rRestRegs, _hcompAcc, _hequivAcc, hacc, _hrest⟩
        exact eval_reg_of_assertion hacc)
      (by
        intro st r _hpre
        exact ⟨{ st with global := {
            bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } },
          globalWriteMem_of_byteWrite haccess hencode (by rfl)⟩)
      hencode hlen
      (by
        intro st st' r rFrame hpre hframe hstep
        have haddr' := resolves_global_gaddr_of_eval (st := st)
          (ctx := { cta := 0, warp := 0, lane := lane })
          (ty := .s32) (expr := .imm (.gaddr .global (matmulCellOffset params)))
          (off := matmulCellOffset params) (by rfl)
        rcases hframe with ⟨rAcc, rRest, hcompAcc, hequivAcc, hacc, hrest⟩
        rcases hrest with ⟨rA0, rRestRegs, hcompA0, hequivA0, ha0, hrestRegs⟩
        rcases hrestRegs with ⟨rB0, rBytes, hcompB0, hequivB0, hb0, hbytes⟩
        rcases hbytes with ⟨rABytes, rBBytes, hcompBytes, hequivBytes, haBytes, hbBytes⟩
        have heval' : EvalRValue st { cta := 0, warp := 0, lane := lane }
            (.reg "acc") (.s32 acc) := by
          exact eval_reg_of_assertion hacc
        have hwrite' : WriteMemFact st .global .s32 (.global (matmulCellOffset params))
            (.s32 acc)
            { st with global := {
                bytes := Helpers.writeBytes st.global.bytes (matmulCellOffset params) newOut } } :=
          globalWriteMem_of_byteWrite haccess hencode (by rfl)
        rcases hpre with ⟨_rCtrl, _rRest, _hcomp, _hequiv, hctrl, _hrest⟩
        exact ⟨rAcc, rRest, hcompAcc, hequivAcc,
          globalStorePreservesReadReg_single_warpAt
            hctrl hacc haddr' heval' hwrite' hstep,
          ⟨rA0, rRestRegs, hcompA0, hequivA0,
            globalStorePreservesReadReg_single_warpAt
              hctrl ha0 haddr' heval' hwrite' hstep,
            ⟨rB0, rBytes, hcompB0, hequivB0,
              globalStorePreservesReadReg_single_warpAt
                hctrl hb0 haddr' heval' hwrite' hstep,
              ⟨rABytes, rBBytes, hcompBytes, hequivBytes,
                globalStorePreservesGlobalBytes_single_warpAt hdisjointA
                  hctrl haBytes haddr' heval' hwrite' hencode hstep,
                globalStorePreservesGlobalBytes_single_warpAt hdisjointB
                  hctrl hbBytes haddr' heval' hwrite' hencode hstep⟩⟩⟩⟩))

theorem matmul_loads_mul_store_wpInstrList
    {params : MatmulCellParams} {pc : PC} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length) :
    (warpAt 0 0 pc [lane] ∗
      ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
        CSL.reg 0 0 lane "a0" oldA) ∗
        (CSL.reg 0 0 lane "b0" oldB ∗
          (CSL.reg 0 0 lane "acc" oldAcc ∗
            (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
              CSL.globalBytes (matmulCellOffset params) .write oldOut))))) ⊢ₛ
      wpInstrList 0 0
        [matmulLoadA0 params, matmulLoadB0 params, matmulMul0, matmulStoreCell params]
        (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
          (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
            (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
              (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                  (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                    CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) := by
  have hafterMul :
      (warpAt 0 0 (pc.1, pc.2 + 3) [lane] ∗
        (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
          (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
            (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
              (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                  CSL.globalBytes (matmulCellOffset params) .write oldOut)))))) ⊢ₛ
        wpInstrList 0 0 [matmulStoreCell params]
          (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
            (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
              (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
                (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                  (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                    (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                      CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_rotate_six_last_to_front
          (CSL.reg 0 0 lane "acc" (.s32 acc))
          (CSL.reg 0 0 lane "a0" (.s32 a0))
          (CSL.reg 0 0 lane "b0" (.s32 b0))
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes)
          (CSL.globalBytes (matmulBOffset params 0) .read bBytes)
          (CSL.globalBytes (matmulCellOffset params) .write oldOut)))
      (by
        simpa [wpInstrList, Nat.add_assoc] using
          (matmul_store_acc_preserves_regs_ab_wp
            (params := params) (pc := (pc.1, pc.2 + 3)) (lane := lane)
            (aBytes := aBytes) (bBytes := bBytes) (oldOut := oldOut)
            (newOut := newOut) (a0 := a0) (b0 := b0) (acc := acc)
            hdisjointA hdisjointB haccessC hencodeC hlenC))
  have hafterB :
      (warpAt 0 0 (pc.1, pc.2 + 2) [lane] ∗
        ((CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
          CSL.reg 0 0 lane "b0" (.s32 b0)) ∗
          (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗
              (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                CSL.globalBytes (matmulCellOffset params) .write oldOut))))) ⊢ₛ
        wpInstrList 0 0 [matmulMul0, matmulStoreCell params]
          (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
            (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
              (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
                (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                  (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                    (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                      CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_ab_cdef_dcbeaf
          (CSL.globalBytes (matmulBOffset params 0) .read bBytes)
          (CSL.reg 0 0 lane "b0" (.s32 b0))
          (CSL.reg 0 0 lane "a0" (.s32 a0))
          (CSL.reg 0 0 lane "acc" oldAcc)
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes)
          (CSL.globalBytes (matmulCellOffset params) .write oldOut)))
      (CSL.entails_trans
        (by
          simpa [Nat.add_assoc] using
            (matmul_mul0_preserves_ab_out_wp
              (params := params) (pc := (pc.1, pc.2 + 2)) (lane := lane)
              (oldAcc := oldAcc) (aBytes := aBytes) (bBytes := bBytes)
              (oldOut := oldOut) (a0 := a0) (b0 := b0) (acc := acc) hmul))
        (wpInstr_mono (by simpa [wpInstrList] using hafterMul)))
  have hafterA :
      (warpAt 0 0 (pc.1, pc.2 + 1) [lane] ∗
        ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
          CSL.reg 0 0 lane "a0" (.s32 a0)) ∗
          (CSL.reg 0 0 lane "b0" oldB ∗
            (CSL.reg 0 0 lane "acc" oldAcc ∗
              (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
                CSL.globalBytes (matmulCellOffset params) .write oldOut))))) ⊢ₛ
        wpInstrList 0 0 [matmulLoadB0 params, matmulMul0, matmulStoreCell params]
          (warpAt 0 0 (pc.1, pc.2 + 4) [lane] ∗
            (CSL.globalBytes (matmulCellOffset params) .write newOut ∗
              (CSL.reg 0 0 lane "acc" (.s32 acc) ∗
                (CSL.reg 0 0 lane "a0" (.s32 a0) ∗
                  (CSL.reg 0 0 lane "b0" (.s32 b0) ∗
                    (CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
                      CSL.globalBytes (matmulBOffset params 0) .read bBytes)))))) := by
    exact CSL.entails_trans
      (CSL.sep_mono (CSL.entails_refl _)
        (CSL.sep_permute_ab_cdef_ecbdaf
          (CSL.globalBytes (matmulAOffset params 0) .read aBytes)
          (CSL.reg 0 0 lane "a0" (.s32 a0))
          (CSL.reg 0 0 lane "b0" oldB)
          (CSL.reg 0 0 lane "acc" oldAcc)
          (CSL.globalBytes (matmulBOffset params 0) .read bBytes)
          (CSL.globalBytes (matmulCellOffset params) .write oldOut)))
      (CSL.entails_trans
        (by
          simpa [Nat.add_assoc] using
            (matmul_loadB0_preserves_a0_acc_abytes_out_wp
              (params := params) (pc := (pc.1, pc.2 + 1)) (lane := lane)
              (aBytes := aBytes) (bBytes := bBytes) (oldOut := oldOut)
              (oldB := oldB) (oldAcc := oldAcc) (a0 := a0) (b0 := b0)
              haccessB hwidthB hdecodeB))
        (wpInstr_mono (by simpa [wpInstrList] using hafterB)))
  exact CSL.entails_trans
    (matmul_loadA0_preserves_b0_acc_bbytes_out_wp
      (params := params) (pc := pc) (lane := lane)
      (aBytes := aBytes) (bBytes := bBytes) (oldOut := oldOut)
      (oldA := oldA) (oldB := oldB) (oldAcc := oldAcc) (a0 := a0)
      haccessA hwidthA hdecodeA)
    (wpInstr_mono (by simpa [wpInstrList] using hafterA))

def matmulCellBlock (params : MatmulCellParams) : Block :=
  { label := "entry"
    body := #[matmulLoadA0 params, matmulLoadB0 params, matmulMul0, matmulStoreCell params]
    term := .terminate }

def matmulCellEnv (params : MatmulCellParams) : KernelEnv :=
  { entry := "entry"
    gridCtx := { gridDim := { x := 1 }, blockDim := { x := 1 } }
    blocks := ({} : Std.HashMap BlockLabel Block).insert "entry" (matmulCellBlock params) }

theorem matmul_cell_entry_lookup (params : MatmulCellParams) :
    (matmulCellEnv params).blocks["entry"]? = some (matmulCellBlock params) := by
  simp [matmulCellEnv]

theorem matmul_cell_block_lookup
    {params : MatmulCellParams} {label : BlockLabel} {block : Block}
    (hlookup : (matmulCellEnv params).blocks[label]? = some block) :
    label = "entry" ∧ block = matmulCellBlock params := by
  by_cases hlabel : label = "entry"
  · subst label
    simp [matmulCellEnv] at hlookup
    exact ⟨rfl, hlookup.symm⟩
  · have hbeq : ("entry" == label) = false :=
      (beq_eq_false_iff_ne).2 (fun h => hlabel h.symm)
    simp [matmulCellEnv, Std.HashMap.getElem?_insert, hbeq] at hlookup

theorem matmul_cell_targets_exist (params : MatmulCellParams) :
    CFGTerminatorTargetsExist (matmulCellEnv params) := by
  intro label block hblock
  rcases matmul_cell_block_lookup hblock with ⟨_, hblockEq⟩
  subst block
  trivial

theorem matmul_cell_body_ordinary (params : MatmulCellParams) :
    CFGBodyUsesOrdinaryPcAdvance (matmulCellEnv params) := by
  intro label block idx gi hblock hgi
  rcases matmul_cell_block_lookup hblock with ⟨_, hblockEq⟩
  subst block
  cases idx with
  | zero =>
      simp [matmulCellBlock, matmulLoadA0, Helpers.instrUsesOrdinaryPcAdvance] at hgi
      subst gi
      rfl
  | succ idx =>
      cases idx with
      | zero =>
          simp [matmulCellBlock, matmulLoadB0, Helpers.instrUsesOrdinaryPcAdvance] at hgi
          subst gi
          rfl
      | succ idx =>
          cases idx with
          | zero =>
              simp [matmulCellBlock, matmulMul0, Helpers.instrUsesOrdinaryPcAdvance] at hgi
              subst gi
              rfl
          | succ idx =>
              cases idx with
              | zero =>
                  simp [matmulCellBlock, matmulStoreCell,
                    Helpers.instrUsesOrdinaryPcAdvance] at hgi
                  subst gi
                  rfl
              | succ idx =>
                  simp [matmulCellBlock] at hgi

def matmulCellScalarPreResources
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut : List Byte) (oldA oldB oldAcc : Value) : CSL.Assertion :=
  ((CSL.globalBytes (matmulAOffset params 0) .read aBytes ∗
    CSL.reg 0 0 lane "a0" oldA) ∗
    (CSL.reg 0 0 lane "b0" oldB ∗
      (CSL.reg 0 0 lane "acc" oldAcc ∗
        (CSL.globalBytes (matmulBOffset params 0) .read bBytes ∗
          CSL.globalBytes (matmulCellOffset params) .write oldOut))))

def matmulCellScalarFinalResources
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes newOut : List Byte) (a0 b0 acc : Int) : CSL.Assertion :=
  CSL.sepList [
    CSL.globalBytes (matmulCellOffset params) .write newOut,
    CSL.reg 0 0 lane "acc" (.s32 acc),
    CSL.reg 0 0 lane "a0" (.s32 a0),
    CSL.reg 0 0 lane "b0" (.s32 b0),
    CSL.globalBytes (matmulAOffset params 0) .read aBytes,
    CSL.globalBytes (matmulBOffset params 0) .read bBytes]

theorem matmulCellScalarFinalResources_stable_terminate
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes newOut : List Byte) (a0 b0 acc : Int) :
    CSL.StableUnder (TerminatorStep 0 0 .terminate)
      (matmulCellScalarFinalResources params lane aBytes bBytes newOut a0 b0 acc) := by
  unfold matmulCellScalarFinalResources
  apply CSL.stable_sepList
  intro p hp
  simp at hp
  rcases hp with hp | hp | hp | hp | hp | hp
  · subst p
    exact stable_globalBytes_terminator
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

def matmulCellScalarPre
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut : List Byte) (oldA oldB oldAcc : Value) : CSL.Assertion :=
  warpAt 0 0 ("entry", 0) [lane] ∗
    matmulCellScalarPreResources params lane aBytes bBytes oldOut oldA oldB oldAcc

def matmulCellScalarPost
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes newOut : List Byte) (a0 b0 acc : Int) : CSL.Assertion :=
  laneTerminatedAt 0 0 lane ("entry", 4) ∗
    matmulCellScalarFinalResources params lane aBytes bBytes newOut a0 b0 acc

def matmulCellScalarInvariants
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut newOut : List Byte) (oldA oldB oldAcc : Value)
    (a0 b0 acc : Int) : InvariantMap :=
  fun label =>
    if label = "entry" then
      matmulCellScalarPre params lane aBytes bBytes oldOut oldA oldB oldAcc
    else
      matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc

theorem matmul_cell_scalar_entry_block_vc
    {params : MatmulCellParams} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length) :
    blockVC 0 0
      (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut
        oldA oldB oldAcc a0 b0 acc)
      (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)
      "entry" (matmulCellBlock params) := by
  intro st r hinv
  have hpre :
      matmulCellScalarPre params lane aBytes bBytes oldOut oldA oldB oldAcc st r := by
    simpa [matmulCellScalarInvariants] using hinv
  have hbody :
      matmulCellScalarPre params lane aBytes bBytes oldOut oldA oldB oldAcc ⊢ₛ
        wpInstrList 0 0
          [matmulLoadA0 params, matmulLoadB0 params, matmulMul0, matmulStoreCell params]
          (warpAt 0 0 ("entry", 4) [lane] ∗
            matmulCellScalarFinalResources params lane aBytes bBytes newOut a0 b0 acc) := by
    simpa [matmulCellScalarPre, matmulCellScalarPreResources,
      matmulCellScalarFinalResources, Nat.add_assoc] using
      (matmul_loads_mul_store_wpInstrList
        (params := params) (pc := ("entry", 0)) (lane := lane)
        (aBytes := aBytes) (bBytes := bBytes) (oldOut := oldOut) (newOut := newOut)
        (oldA := oldA) (oldB := oldB) (oldAcc := oldAcc)
        (a0 := a0) (b0 := b0) (acc := acc)
        haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB hmul
        hdisjointA hdisjointB haccessC hencodeC hlenC)
  have hterm :
      (warpAt 0 0 ("entry", 4) [lane] ∗
        matmulCellScalarFinalResources params lane aBytes bBytes newOut a0 b0 acc) ⊢ₛ
        wpTerminator 0 0 .terminate
          (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc) := by
    simpa [matmulCellScalarPost] using
      (wp_terminate_single_warpAt_frame
        (cta := 0) (warp := 0) (pc := ("entry", 4)) (lane := lane)
        (frame := matmulCellScalarFinalResources params lane aBytes bBytes newOut a0 b0 acc)
        (matmulCellScalarFinalResources_stable_terminate
          params lane aBytes bBytes newOut a0 b0 acc))
  have hentry :
      matmulCellScalarPre params lane aBytes bBytes oldOut oldA oldB oldAcc ⊢ₛ
        blockEntryWP 0 0
          (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut
            oldA oldB oldAcc a0 b0 acc)
          (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)
          (matmulCellBlock params) := by
    exact CSL.entails_trans hbody
      (wpInstrList_mono (by simpa [blockTermPost] using hterm))
  exact hentry st r hpre

theorem matmul_cell_scalar_block_vcs
    {params : MatmulCellParams} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length) :
    ∀ label block,
      (matmulCellEnv params).blocks[label]? = some block →
        blockVC 0 0
          (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut
            oldA oldB oldAcc a0 b0 acc)
          (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)
          label block := by
  intro label block hlookup
  rcases matmul_cell_block_lookup hlookup with ⟨hlabel, hblock⟩
  subst label
  subst block
  exact matmul_cell_scalar_entry_block_vc
    haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB hmul
    hdisjointA hdisjointB haccessC hencodeC hlenC

def matmulCellConcreteInvariants (params : MatmulCellParams) (post : CSL.Assertion) :
    InvariantMap :=
  fun _ => blockEntryWP 0 0 (fun _ => post) post (matmulCellBlock params)

theorem matmul_cell_entry_block_vc (params : MatmulCellParams) (post : CSL.Assertion) :
    blockVC 0 0 (matmulCellConcreteInvariants params post) post "entry"
      (matmulCellBlock params) := by
  intro st r hentry
  simpa [matmulCellConcreteInvariants, blockEntryWP, blockSuffixWP, matmulCellBlock,
    blockTermPost] using hentry

theorem matmul_cell_block_vcs (params : MatmulCellParams) (post : CSL.Assertion) :
    ∀ label block,
      (matmulCellEnv params).blocks[label]? = some block →
        blockVC 0 0 (matmulCellConcreteInvariants params post) post label block := by
  intro label block hlookup
  rcases matmul_cell_block_lookup hlookup with ⟨hlabel, hblock⟩
  subst label
  subst block
  exact matmul_cell_entry_block_vc params post

theorem matmul_cell_br_semantic_control (params : MatmulCellParams) :
    BrSemanticControl (matmulCellEnv params) 0 0 := by
  intro st st' warpState pc block targetBlock target henv hwarp hlock hrpc hlookup hterm
    htarget hstep
  rcases matmul_cell_block_lookup hlookup with ⟨_, hblock⟩
  subst block
  simp [matmulCellBlock] at hterm

theorem matmul_cell_cbr_semantic_control (params : MatmulCellParams) :
    CbrSemanticControl (matmulCellEnv params) 0 0 := by
  intro st st' warpState pc block trueBlock falseBlock cond tLabel fLabel
    henv hwarp hlock hrpc hlookup hterm htrue hfalse hstep
  rcases matmul_cell_block_lookup hlookup with ⟨_, hblock⟩
  subst block
  simp [matmulCellBlock] at hterm

theorem matmul_cell_scalar_term_preserves
    {params : MatmulCellParams} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length) :
    TermStepPreservesKernel (matmulCellEnv params) 0 0
      (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut
        oldA oldB oldAcc a0 b0 acc)
      (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc) :=
  TermStepPreservesKernel.of_blockVCs
    (matmul_cell_scalar_block_vcs
      haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB hmul
      hdisjointA hdisjointB haccessC hencodeC hlenC)
    (BrTermControl.of_targets_semantic
      (matmul_cell_targets_exist params) (matmul_cell_br_semantic_control params))
    (CbrTermControl.of_targets_semantic
      (matmul_cell_targets_exist params) (matmul_cell_cbr_semantic_control params))

theorem matmul_cell_term_preserves
    (params : MatmulCellParams) (post : CSL.Assertion) :
    TermStepPreservesKernel (matmulCellEnv params) 0 0
      (matmulCellConcreteInvariants params post) post :=
  TermStepPreservesKernel.of_blockVCs
    (matmul_cell_block_vcs params post)
    (BrTermControl.of_targets_semantic
      (matmul_cell_targets_exist params) (matmul_cell_br_semantic_control params))
    (CbrTermControl.of_targets_semantic
      (matmul_cell_targets_exist params) (matmul_cell_cbr_semantic_control params))

def matmulCellScalarKernelPre
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut : List Byte) (oldA oldB oldAcc : Value) : CSL.Assertion :=
  fun st r =>
    st.kernelEnv = matmulCellEnv params ∧
      matmulCellScalarPre params lane aBytes bBytes oldOut oldA oldB oldAcc st r

def matmulCellScalarKernelInvariant
    (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut newOut : List Byte) (oldA oldB oldAcc : Value)
    (a0 b0 acc : Int) : CSL.Assertion :=
  cfgKernelInvariant (matmulCellEnv params) 0 0
    (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB oldAcc
      a0 b0 acc)
    (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)

def matmulCellScalarKernelSpec
    (init : State) (params : MatmulCellParams) (lane : LaneId)
    (aBytes bBytes oldOut newOut : List Byte) (oldA oldB oldAcc : Value)
    (a0 b0 acc : Int) (resource : Resource) : KernelSpec :=
  { init := init
    resource := resource
    pre := matmulCellScalarKernelPre params lane aBytes bBytes oldOut oldA oldB oldAcc
    invariant :=
      matmulCellScalarKernelInvariant params lane aBytes bBytes oldOut newOut oldA oldB
        oldAcc a0 b0 acc
    post := matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc }

theorem matmul_cell_scalar_pre_entails_kernel_invariant
    {params : MatmulCellParams} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int}
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length) :
    matmulCellScalarKernelPre params lane aBytes bBytes oldOut oldA oldB oldAcc ⊢ₛ
      matmulCellScalarKernelInvariant params lane aBytes bBytes oldOut newOut oldA oldB
        oldAcc a0 b0 acc := by
  intro st r hpre
  rcases hpre with ⟨henv, hscalar⟩
  have hscalarPre := hscalar
  rcases hscalar with ⟨_rCtrl, _rResources, _hcomp, _hequiv, hctrl, _hresources⟩
  rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
  have hentryBlock :
      (matmulCellEnv params).blocks[(matmulCellEnv params).entry]? =
        some (matmulCellBlock params) := by
    simp [matmulCellEnv]
  have hvc :
      blockVC 0 0
        (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
          oldAcc a0 b0 acc)
        (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)
        "entry" (matmulCellBlock params) :=
    matmul_cell_scalar_entry_block_vc
      haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB hmul
      hdisjointA hdisjointB haccessC hencodeC hlenC
  have hinvEntry :
      matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
        oldAcc a0 b0 acc "entry" st r := by
    simpa [matmulCellScalarInvariants] using hscalarPre
  have hwp :
      blockSuffixWP 0 0
        (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
          oldAcc a0 b0 acc)
        (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc)
        (matmulCellBlock params) 0 st r :=
    blockVC.entry_suffix hvc st r hinvEntry
  exact Or.inl
    ⟨warpState, ((matmulCellEnv params).entry, 0), matmulCellBlock params, henv, hwarp,
      hlock, by simpa [matmulCellEnv] using hrpc, hentryBlock, hwp⟩

theorem matmul_cell_scalar_kernel_valid
    {init : State} {params : MatmulCellParams} {lane : LaneId}
    {aBytes bBytes oldOut newOut : List Byte} {oldA oldB oldAcc : Value}
    {a0 b0 acc : Int} {resource : Resource}
    (hinit :
      matmulCellScalarKernelPre params lane aBytes bBytes oldOut oldA oldB oldAcc
        init resource)
    (haccessA : AccessOk .global .s32 (.global (matmulAOffset params 0)))
    (hwidthA : Typing.byteWidth? .s32 = some aBytes.length)
    (hdecodeA : DecodedScalar .s32 aBytes (.s32 a0))
    (haccessB : AccessOk .global .s32 (.global (matmulBOffset params 0)))
    (hwidthB : Typing.byteWidth? .s32 = some bBytes.length)
    (hdecodeB : DecodedScalar .s32 bBytes (.s32 b0))
    (hmul : Helpers.evalBinary? .mul (.s32 a0) (.s32 b0) = some (.s32 acc))
    (hdisjointA :
      ByteRangesDisjoint (matmulAOffset params 0) aBytes.length
        (matmulCellOffset params) newOut.length)
    (hdisjointB :
      ByteRangesDisjoint (matmulBOffset params 0) bBytes.length
        (matmulCellOffset params) newOut.length)
    (haccessC : AccessOk .global .s32 (.global (matmulCellOffset params)))
    (hencodeC : EncodedScalar .s32 (.s32 acc) newOut)
    (hlenC : oldOut.length = newOut.length)
    (honly :
      cfgKernelInvariant (matmulCellEnv params) 0 0
        (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
          oldAcc a0 b0 acc)
        (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc) ⊢ₛ
        OnlyRunnableWarp 0 0)
    (hpostNoStep :
      NoStepBlock 0 0
        (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc))
    (hsuffixNoFinal :
      NoFinal
        (cfgSuffixInvariant (matmulCellEnv params) 0 0
          (matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
            oldAcc a0 b0 acc)
          (matmulCellScalarPost params lane aBytes bBytes newOut a0 b0 acc))) :
    (matmulCellScalarKernelSpec init params lane aBytes bBytes oldOut newOut oldA oldB oldAcc
      a0 b0 acc resource).Valid :=
  KernelSpec.Valid.of_entry_blockVCs_targets_closed
    (spec :=
      matmulCellScalarKernelSpec init params lane aBytes bBytes oldOut newOut oldA oldB
        oldAcc a0 b0 acc resource)
    (env := matmulCellEnv params) (cta := 0) (warp := 0)
    (invariants :=
      matmulCellScalarInvariants params lane aBytes bBytes oldOut newOut oldA oldB
        oldAcc a0 b0 acc)
    (hinvariant := rfl)
    (hpreEntry := by
      intro st r hpre
      rcases hpre with ⟨_henv, hscalar⟩
      simpa [matmulCellScalarInvariants, matmulCellEnv] using hscalar)
    (hentryReady := by
      intro st r hpre
      rcases hpre with ⟨henv, hscalar⟩
      rcases hscalar with ⟨_rCtrl, _rResources, _hcomp, _hequiv, hctrl, _hresources⟩
      rcases warpAt_state hctrl with ⟨warpState, hwarp, hlock, hrpc, _hpart⟩
      exact ⟨warpState, matmulCellBlock params, henv, hwarp, hlock,
        by simpa [matmulCellEnv] using hrpc, by simp [matmulCellEnv]⟩)
    (hpre := hinit)
    (hselect := StepMachineSelects.of_entails_onlyRunnableWarp honly)
    (hbody := BodyStepControl.of_ordinary_cfg_semantics (matmul_cell_body_ordinary params))
    (hblocks :=
      matmul_cell_scalar_block_vcs
        haccessA hwidthA hdecodeA haccessB hwidthB hdecodeB hmul
        hdisjointA hdisjointB haccessC hencodeC hlenC)
    (htargets := matmul_cell_targets_exist params)
    (hbr := matmul_cell_br_semantic_control params)
    (hcbr := matmul_cell_cbr_semantic_control params)
    (hpostNoStep := hpostNoStep)
    (hsuffixNoFinal := hsuffixNoFinal)

def matmulCellPost
    (params : MatmulCellParams) (a b : Nat → Nat → Int) (st : State) (_r : Resource) :
    Prop :=
  readGlobalS32? st (matmulCellOffset params) = some (matmulCellValue params a b)

def matmulCellKernelSpec
    (init : State) (params : MatmulCellParams) (a b : Nat → Nat → Int)
    (resource : Resource) : KernelSpec :=
  { init := init
    resource := resource
    pre := fun st r => st = init ∧ r = resource
    post := matmulCellPost params a b }

theorem matmul_cell_partial_correct_of_valid
    {init : State} {params : MatmulCellParams} {a b : Nat → Nat → Int}
    {resource : Resource}
    (hvalid : (matmulCellKernelSpec init params a b resource).Valid) :
    PartialCorrect init (fun st => matmulCellPost params a b st resource) :=
by
  intro final hterm
  rcases KernelSpec.partial_correct hvalid final hterm with ⟨_, hpost⟩
  exact hpost

end Examples
end CLean
