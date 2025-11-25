import CLean.DeviceIR
import CLean.Semantics.DeviceSemantics

/-!
# DeviceIR to Simplified Semantic Body

Translates full DeviceIR kernel body to simplified form for functional correctness proofs.
Similar to deviceIRToKernelSpec for safety, but produces executable DStmt.

Key simplifications:
- Replaces variable references to parameters with literals (e.g., var "N" → intLit N_value)
- Simplifies thread ID computation
- Preserves core computation logic
-/

namespace CLean.ToSemanticBody

open DeviceIR
open CLean.Semantics

/-! ## Parameter Substitution -/

/-- Substitute parameter variables with their literal values -/
def substParamInExpr (e : DExpr) (paramName : String) (value : DExpr) : DExpr :=
  match e with
  | DExpr.var x => if x == paramName then value else e
  | DExpr.binop op e1 e2 => DExpr.binop op (substParamInExpr e1 paramName value) (substParamInExpr e2 paramName value)
  | DExpr.unop op e' => DExpr.unop op (substParamInExpr e' paramName value)
  | DExpr.index arr idx => DExpr.index (substParamInExpr arr paramName value) (substParamInExpr idx paramName value)
  | _ => e

/-- Substitute in statements -/
partial def substParamInStmt (s : DStmt) (paramName : String) (value : DExpr) : DStmt :=
  match s with
  | DStmt.assign x e => DStmt.assign x (substParamInExpr e paramName value)
  | DStmt.store arr idx val =>
      DStmt.store (substParamInExpr arr paramName value)
                  (substParamInExpr idx paramName value)
                  (substParamInExpr val paramName value)
  | DStmt.seq s1 s2 => DStmt.seq (substParamInStmt s1 paramName value) (substParamInStmt s2 paramName value)
  | DStmt.ite cond sthen selse =>
      DStmt.ite (substParamInExpr cond paramName value)
                (substParamInStmt sthen paramName value)
                (substParamInStmt selse paramName value)
  | DStmt.for x lo hi body =>
      DStmt.for x (substParamInExpr lo paramName value)
                   (substParamInExpr hi paramName value)
                   (substParamInStmt body paramName value)
  | DStmt.whileLoop cond body =>
      DStmt.whileLoop (substParamInExpr cond paramName value) (substParamInStmt body paramName value)
  | _ => s

/-! ## Simplification -/

/-- Simplify thread ID computation: replace (blockIdx * blockDim + threadIdx) with just threadIdx
    for single-block kernels -/
partial def simplifyThreadId (e : DExpr) : DExpr :=
  match e with
  | DExpr.binop BinOp.add
      (DExpr.binop BinOp.mul (DExpr.blockIdx _) (DExpr.blockDim _))
      (DExpr.threadIdx dim) => DExpr.threadIdx dim
  | DExpr.binop op e1 e2 => DExpr.binop op (simplifyThreadId e1) (simplifyThreadId e2)
  | DExpr.unop op e' => DExpr.unop op (simplifyThreadId e')
  | DExpr.index arr idx => DExpr.index (simplifyThreadId arr) (simplifyThreadId idx)
  | _ => e

partial def simplifyThreadIdInStmt (s : DStmt) : DStmt :=
  match s with
  | DStmt.assign x e => DStmt.assign x (simplifyThreadId e)
  | DStmt.store arr idx val =>
      DStmt.store (simplifyThreadId arr) (simplifyThreadId idx) (simplifyThreadId val)
  | DStmt.seq s1 s2 => DStmt.seq (simplifyThreadIdInStmt s1) (simplifyThreadIdInStmt s2)
  | DStmt.ite cond sthen selse =>
      DStmt.ite (simplifyThreadId cond) (simplifyThreadIdInStmt sthen) (simplifyThreadIdInStmt selse)
  | _ => s

/-! ## Main Translation -/

/-- Convert DeviceIR kernel body to simplified semantic form

    Parameters:
    - kernelBody: The kernel's DStmt body
    - params: List of (paramName, literalValue) to substitute

    Example: translateToSemanticBody incrementKernelIR.body [("N", DExpr.intLit 100)]
-/
def translateToSemanticBody (kernelBody : DStmt) (params : List (String × DExpr)) : DStmt :=
  let substituted := params.foldl (fun s (name, value) => substParamInStmt s name value) kernelBody
  simplifyThreadIdInStmt substituted

/-! ## Helper for Common Case -/

/-- Translate kernel with single Nat parameter -/
def translateWithNatParam (kernelBody : DStmt) (paramName : String) (N : Nat) : DStmt :=
  translateToSemanticBody kernelBody [(paramName, DExpr.intLit N)]

end CLean.ToSemanticBody
