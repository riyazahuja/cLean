import Lake
open Lake DSL System

-- def linkArgs :=
--   if System.Platform.isWindows then
--     panic! "Windows is not supported!"
--   else if System.Platform.isOSX then
--     #["-L/opt/homebrew/opt/openblas/lib", "-L/usr/local/opt/openblas/lib", "-lblas"]
--   else -- Linux
--     -- Use local symlinks to flexiblas (no sudo needed)
--     #["-L./.local/lib", "-L/usr/lib64/", "-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]

package cLean {
  -- moreLinkArgs := linkArgs
  -- Add include path for local cblas.h symlink
  -- moreLeancArgs := #["-I./.local/include"]

  -- Link CUDA FFI library
  moreLinkArgs := #["-L.lake/build/lib", "-lcuda_ffi",
                     "-L/usr/local/cuda/lib64", "-L/usr/lib64",
                     "-lcudart", "-lcuda", "-lnvrtc"]
}
require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "v4.20.1"
-- require scilean from git "https://github.com/lecopivo/SciLean" @ "v4.20.1"

@[default_target]
lean_lib CLean {
  roots := #[`CLean]
}

lean_exe examples {
  root := `CLean.examples
  supportInterpreter := true
}

lean_exe test_ffi_basic {
  root := `test_ffi_basic
  supportInterpreter := true
}

lean_exe test_gpu_e2e {
  root := `test_gpu_e2e
  supportInterpreter := true
}

lean_exe test_minimal_ffi {
  root := `test_minimal_ffi
  supportInterpreter := true
}
