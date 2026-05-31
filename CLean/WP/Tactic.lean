import CLean.WP.Soundness

namespace CLean
namespace WP

syntax "cwp" : tactic

macro_rules
  | `(tactic| cwp) =>
      `(tactic|
        simp [KernelSpec.Valid, wpBlock, wpInstr, wpTerminator, CSL.emp, CSL.pure])

end WP
end CLean
