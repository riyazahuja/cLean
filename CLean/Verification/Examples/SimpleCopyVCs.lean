/-
  Verification Conditions for simpleCopyKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem simpleCopyKernel_no_race_3_3 : 
  WRITE output[0] := val @ loc 3 and WRITE output[0] := val @ loc 3 do not race :=
  sorry

theorem simpleCopyKernel_no_race_3_3 : 
  WRITE output[0] := val @ loc 3 and WRITE output[0] := val @ loc 3 do not race :=
  sorry

theorem simpleCopyKernel_bounds_3_output : 
  Access to output[0] is within bounds [0, 1) :=
  sorry

theorem simpleCopyKernel_bounds_2_input : 
  Access to input[0] is within bounds [0, 1) :=
  sorry