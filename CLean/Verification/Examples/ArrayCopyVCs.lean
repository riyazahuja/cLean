/-
  Verification Conditions for arrayCopyKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem arrayCopyKernel_no_race_4_4 : 
  WRITE output[i] := val @ loc 4 and WRITE output[i] := val @ loc 4 do not race :=
  sorry

theorem arrayCopyKernel_no_race_4_4 : 
  WRITE output[i] := val @ loc 4 and WRITE output[i] := val @ loc 4 do not race :=
  sorry

theorem arrayCopyKernel_bounds_4_output : 
  Access to output[i] is within bounds [0, 1024) :=
  sorry

theorem arrayCopyKernel_bounds_3_input : 
  Access to input[i] is within bounds [0, 1024) :=
  sorry