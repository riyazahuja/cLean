/-
  Verification Conditions for raceKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem raceKernel_no_race_1_1 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem raceKernel_no_race_1_0 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and READ counter[0] @ loc 0 do not race :=
  sorry

theorem raceKernel_no_race_1_1 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem raceKernel_no_race_1_0 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and READ counter[0] @ loc 0 do not race :=
  sorry

theorem raceKernel_no_race_0_1 : 
  READ counter[0] @ loc 0 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem raceKernel_no_race_0_1 : 
  READ counter[0] @ loc 0 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem raceKernel_bounds_1_counter : 
  Access to counter[0] is within bounds [0, 1) :=
  sorry

theorem raceKernel_bounds_0_counter : 
  Access to counter[0] is within bounds [0, 1) :=
  sorry