/-
  Verification Conditions for saxpyKernel
  Generated automatically from DeviceIR
-/

import CLean.Verification.SafetyProperties
import CLean.Verification.Tactics

open CLean.Verification.SafetyProperties

theorem saxpyKernel_no_race_6_6 : 
  WRITE r[i] := ((alpha * xi) + yi) @ loc 6 and WRITE r[i] := ((alpha * xi) + yi) @ loc 6 do not race :=
  sorry

theorem saxpyKernel_no_race_6_6 : 
  WRITE r[i] := ((alpha * xi) + yi) @ loc 6 and WRITE r[i] := ((alpha * xi) + yi) @ loc 6 do not race :=
  sorry

theorem saxpyKernel_bounds_6_r : 
  Access to r[i] is within bounds [0, 1024) :=
  sorry

theorem saxpyKernel_bounds_5_y : 
  Access to y[i] is within bounds [0, 1024) :=
  sorry

theorem saxpyKernel_bounds_4_x : 
  Access to x[i] is within bounds [0, 1024) :=
  sorry