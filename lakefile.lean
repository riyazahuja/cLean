import Lake
open Lake DSL System

def linkArgs :=
  if System.Platform.isWindows then
    panic! "Windows is not supported!"
  else if System.Platform.isOSX then
    #["-L/opt/homebrew/opt/openblas/lib", "-L/usr/local/opt/openblas/lib", "-lblas"]
  else -- Linux
    #["-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]

package cLean {
  moreLinkArgs := linkArgs
}

require scilean from git "https://github.com/lecopivo/SciLean" @ "v4.20.1"

@[default_target]
lean_lib CLean {
  roots := #[`CLean]
}

lean_exe examples {
  root := `CLean.examples
  supportInterpreter := true
}
