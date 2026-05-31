import CLean.WP.CFG

namespace CLean
namespace WP

theorem step_wp_sound
    {cta : CTAId} {warp : WarpId} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpBlock cta warp post st r)
    (hstep : StepBlock st cta warp st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  wpBlock_sound hwp hstep

theorem instr_wp_sound
    {cta : CTAId} {warp : WarpId} {gi : GInstr} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpInstr cta warp gi post st r)
    (hstep : StepInstr st cta warp gi st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  wpInstr_sound hwp hstep

theorem executable_step_wp_sound
    {cta : CTAId} {warp : WarpId} {post : Post}
    {st st' : State} {r : CSL.Resource}
    (hwp : wpExecutableStep cta warp post st r)
    (hstep : StepMachine.stepAt? st cta warp = some st') :
    ∃ r', CSL.Resource.Update r r' ∧ post st' r' :=
  wpExecutableStep_sound hwp hstep

theorem kernel_wp_sound {spec : KernelSpec}
    (hvalid : spec.Valid) :
    PartialCorrect spec.init (fun st => ∃ r, spec.post st r) :=
  KernelSpec.partial_correct hvalid

end WP
end CLean
