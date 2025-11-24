/-
  Verification Conditions for unsafeCounterKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem unsafeCounterKernel_no_race_1_1 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem unsafeCounterKernel_no_race_1_0 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and READ counter[0] @ loc 0 do not race :=
  sorry

theorem unsafeCounterKernel_no_race_1_1 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem unsafeCounterKernel_no_race_1_0 : 
  WRITE counter[0] := (val + unknown) @ loc 1 and READ counter[0] @ loc 0 do not race :=
  sorry

theorem unsafeCounterKernel_no_race_0_1 : 
  READ counter[0] @ loc 0 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem unsafeCounterKernel_no_race_0_1 : 
  READ counter[0] @ loc 0 and WRITE counter[0] := (val + unknown) @ loc 1 do not race :=
  sorry

theorem unsafeCounterKernel_bounds_1_counter : 
  Access to counter[0] is within bounds [0, 1) :=
  sorry

theorem unsafeCounterKernel_bounds_0_counter : 
  Access to counter[0] is within bounds [0, 1) :=
  sorry