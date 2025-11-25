import CLean.DeviceIR
import CLean.Semantics.DeviceSemantics
import CLean.Verification.HashMapLemmas
import CLean.Verification.SemanticHelpers

/-!
# Kernel to Simplified Semantic Form with Metadata

Converts DeviceIR kernel to simplified two-thread GPUVerify-style form.
Preserves type metadata (globalArrays, params) for proof purposes.
-/

namespace CLean.Verification.FunctionalCorrectness

open DeviceIR
open CLean.Semantics
open Std (HashMap)

/-! ## Simplified Kernel with Metadata -/

/-- A simplified kernel body with preserved type metadata -/
structure SimplifiedKernel where
  /-- Original kernel for metadata -/
  kernel : Kernel
  /-- Simplified body (two-thread reduction, thread ID simplified) -/
  body : DStmt
  deriving Inhabited

/-! ## Simplification Functions -/

/-- Simplify thread ID computation for single-block kernels
    (blockIdx.x * blockDim.x + threadIdx.x) → threadIdx.x when gridDim = 1 -/
def simplifyThreadIdExpr (e : DExpr) : DExpr :=
  match e with
  | DExpr.binop BinOp.add
      (DExpr.binop BinOp.mul (DExpr.blockIdx _) (DExpr.blockDim _))
      (DExpr.threadIdx dim) => DExpr.threadIdx dim
  | DExpr.binop op e1 e2 => DExpr.binop op (simplifyThreadIdExpr e1) (simplifyThreadIdExpr e2)
  | DExpr.unop op e' => DExpr.unop op (simplifyThreadIdExpr e')
  | DExpr.index arr idx => DExpr.index (simplifyThreadIdExpr arr) (simplifyThreadIdExpr idx)
  | _ => e

def simplifyThreadIdStmt (s : DStmt) : DStmt :=
  match s with
  | DStmt.assign x e => DStmt.assign x (simplifyThreadIdExpr e)
  | DStmt.store arr idx val =>
      DStmt.store (simplifyThreadIdExpr arr) (simplifyThreadIdExpr idx) (simplifyThreadIdExpr val)
  | DStmt.seq s1 s2 => DStmt.seq (simplifyThreadIdStmt s1) (simplifyThreadIdStmt s2)
  | DStmt.ite cond sthen selse =>
      DStmt.ite (simplifyThreadIdExpr cond) (simplifyThreadIdStmt sthen) (simplifyThreadIdStmt selse)
  | DStmt.for x lo hi body =>
      DStmt.for x (simplifyThreadIdExpr lo) (simplifyThreadIdExpr hi) (simplifyThreadIdStmt body)
  | DStmt.whileLoop cond body =>
      DStmt.whileLoop (simplifyThreadIdExpr cond) (simplifyThreadIdStmt body)
  | _ => s

/-! ## Main Conversion -/

/-- Convert kernel to simplified form with metadata preserved -/
def kernelToSimplified (kernel : Kernel) : SimplifiedKernel :=
  { kernel := kernel
    body := simplifyThreadIdStmt kernel.body }

/-! ## Memory Initialization from Kernel Params -/

/-- Initialize memory with kernel parameters
    Params like N are placed in memory for access -/
def initMemoryWithParams (params : List (String × Value)) : Memory :=
  let paramMap := params.foldl (fun m (name, val) =>
    m.insert name ([(0, val)].foldl (fun hm (idx, v) => hm.insert idx v) ∅)
  ) ∅
  { arrays := paramMap }

end CLean.Verification.FunctionalCorrectness
