/-
  Verification Conditions for incrementKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem incrementKernel_no_race_4_4 : 
  WRITE data[i] := (val + unknown) @ loc 4 and WRITE data[i] := (val + unknown) @ loc 4 do not race :=
  sorry

theorem incrementKernel_no_race_4_3 : 
  WRITE data[i] := (val + unknown) @ loc 4 and READ data[i] @ loc 3 do not race :=
  sorry

theorem incrementKernel_no_race_4_4 : 
  WRITE data[i] := (val + unknown) @ loc 4 and WRITE data[i] := (val + unknown) @ loc 4 do not race :=
  sorry

theorem incrementKernel_no_race_4_3 : 
  WRITE data[i] := (val + unknown) @ loc 4 and READ data[i] @ loc 3 do not race :=
  sorry

theorem incrementKernel_no_race_3_4 : 
  READ data[i] @ loc 3 and WRITE data[i] := (val + unknown) @ loc 4 do not race :=
  sorry

theorem incrementKernel_no_race_3_4 : 
  READ data[i] @ loc 3 and WRITE data[i] := (val + unknown) @ loc 4 do not race :=
  sorry

theorem incrementKernel_bounds_4_data : 
  Access to data[i] is within bounds [0, 1024) :=
  sorry

theorem incrementKernel_bounds_3_data : 
  Access to data[i] is within bounds [0, 1024) :=
  sorry