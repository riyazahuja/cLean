import CLean.DeviceIR
import CLean.Verification.KernelToSemantic

/-!
# Kernel Simplification Helper

Provides the explicit simplified form of kernelToSimplified for use in proofs.
Since the simplification function is complex, we provide the result as a lemma.
-/

namespace CLean.Verification

open DeviceIR
open CLean.Verification.FunctionalCorrectness

/-- The explicit form that kernelToSimplified produces for a given kernel
    This is what #eval shows, captured as a definition for proofs -/
def mkSimplifiedBody (kernel : Kernel) : DStmt :=
  simplifyThreadIdStmt kernel.body

/-- Lemma: kernelToSimplified just applies simplifyThreadIdStmt -/
axiom kernelToSimplified_unfold (kernel : Kernel) :
  (kernelToSimplified kernel).body = mkSimplifiedBody kernel

end CLean.Verification
