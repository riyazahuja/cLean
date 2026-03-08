import CLean.Helpers

namespace CLean

open Helpers

inductive StepInstr : State → CTAId → WarpId → GInstr → State → Prop where
  | mk {st cta warp gi st'} :
      Helpers.stepInstr? st cta warp gi = some st' →
      StepInstr st cta warp gi st'

inductive StepBlock : State → CTAId → WarpId → State → Prop where
  | body {st cta warp st' pc block gi} :
      (hwarp : st.getWarp? cta warp = some (st.getWarp? cta warp).get!) →
      Helpers.currentRunnablePc? (st.getWarp? cta warp).get! = some pc →
      st.kernelEnv.blocks[pc.1]? = some block →
      block.body[pc.2]? = some gi →
      StepInstr st cta warp gi st' →
      StepBlock st cta warp st'
  | term {st cta warp st' pc block} :
      (hwarp : st.getWarp? cta warp = some (st.getWarp? cta warp).get!) →
      Helpers.currentRunnablePc? (st.getWarp? cta warp).get! = some pc →
      st.kernelEnv.blocks[pc.1]? = some block →
      block.body[pc.2]? = none →
      Helpers.stepTerminator? st cta warp block.term = some st' →
      StepBlock st cta warp st'

inductive StepWarp : State → CTAId → WarpId → State → Prop where
  | mk {st cta warp st'} : StepBlock st cta warp st' → StepWarp st cta warp st'

inductive StepMachine : State → State → Prop where
  | mk {st st' cta warp} : StepWarp st cta warp st' → StepMachine st st'

end CLean
