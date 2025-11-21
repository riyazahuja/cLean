/-
  Minimal FFI test - just check device count
-/
import CLean.GPU.FFI

open CLean.GPU.FFI

def main : IO Unit := do
  IO.println "Testing FFI..."
  let count ← cudaGetDeviceCount
  IO.println s!"CUDA devices: {count}"
