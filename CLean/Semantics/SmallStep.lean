import CLean.Semantics.Helpers

namespace CLean

open Helpers

inductive StepInstr : State → CTAId → WarpId → GInstr → State → Prop where
  | mk {st cta warp gi st' warpState participants} :
      State.wf st →
      st.getWarp? cta warp = some warpState →
      WarpState.wf warpState →
      Helpers.lockstepRunnable warpState →
      Helpers.ParticipatingRunnable warpState gi.guard? participants →
      Helpers.stepInstr? st cta warp gi = some st' →
      StepInstr st cta warp gi st'

inductive StepBlock : State → CTAId → WarpId → State → Prop where
  | body {st cta warp st' warpState pc block gi} :
      State.wf st →
      st.getWarp? cta warp = some warpState →
      WarpState.wf warpState →
      Helpers.lockstepRunnable warpState →
      Helpers.RunnablePc warpState pc →
      st.kernelEnv.blocks[pc.1]? = some block →
      block.body[pc.2]? = some gi →
      StepInstr st cta warp gi st' →
      StepBlock st cta warp st'
  | term {st cta warp st' warpState pc block} :
      State.wf st →
      st.getWarp? cta warp = some warpState →
      WarpState.wf warpState →
      Helpers.lockstepRunnable warpState →
      Helpers.RunnablePc warpState pc →
      st.kernelEnv.blocks[pc.1]? = some block →
      block.body[pc.2]? = none →
      Helpers.stepTerminator? st cta warp block.term = some st' →
      StepBlock st cta warp st'

inductive StepWarp : State → CTAId → WarpId → State → Prop where
  | mk {st cta warp st'} :
      State.wf st →
      StepBlock st cta warp st' →
      StepWarp st cta warp st'

inductive StepMachine : State → State → Prop where
  | mk {st st' cta warp} :
      State.wf st →
      StepWarp st cta warp st' →
      StepMachine st st'

end CLean
