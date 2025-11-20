/-
  Test extraction of existing kernels to DeviceIR
-/

import CLean.DeviceExtractor
import CLean.examples

open DeviceExtractor

-- Test: extract saxpyKernel from examples.lean
#extract_kernel Saxpy.saxpyKernel

-- Can also extract programmatically
#eval show MetaM Unit from do
  match ← extractKernel `Saxpy.saxpyKernel with
  | none => IO.println "Failed to extract saxpyKernel"
  | some kernel => do
      IO.println "✓ Extracted saxpyKernel!"
      IO.println s!"Name: {kernel.name}"
      IO.println s!"Params: {kernel.params.length}"
      IO.println s!"Locals: {kernel.locals.length}"
      IO.println s!"Global Arrays: {kernel.globalArrays.length}"
      IO.println s!"Body statements: (see IR)"
      IO.println s!"\nFull IR:\n{repr kernel}"
