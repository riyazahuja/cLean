/-
  Debug the process spawning to find where it hangs
-/

import Lean

def main : IO Unit := do
  IO.println "=== Testing Process Spawn ==="

  IO.println "\n1. Testing echo (should be instant)..."
  let result1 ← IO.Process.run {
    cmd := "echo"
    args := #["hello", "world"]
  }
  IO.println s!"Echo result: {result1}"

  IO.println "\n2. Testing gpu_launcher with echo input..."
  let child ← IO.Process.spawn {
    cmd := "./gpu_launcher"
    args := #[".cache/gpu_kernels/kernel_841679951/testKernel.ptx", "testKernel", "1", "1", "1", "256", "1", "1"]
    stdin := .piped
    stdout := .piped
    stderr := .piped
  }

  IO.println "Process spawned, writing to stdin..."
  let stdin := child.stdin
  let jsonInput := "{\"scalars\":[8.0,2.5],\"arrays\":{\"X\":[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],\"Y\":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],\"R\":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}}"
  stdin.putStr jsonInput
  stdin.putStr "\n"  -- Add newline
  stdin.flush
  IO.println "Stdin written and flushed"

  -- Try to close stdin by dropping the handle
  -- This is the key issue - we need stdin to close!
  IO.println "Attempting to signal EOF..."

  -- Read stderr first (it's smaller)
  IO.println "Reading stderr..."
  let stderrContent ← child.stderr.readToEnd
  IO.println s!"Stderr: {stderrContent}"

  -- Read stdout
  IO.println "Reading stdout..."
  let stdoutContent ← child.stdout.readToEnd
  IO.println s!"Stdout: {stdoutContent}"

  -- Wait for process
  IO.println "Waiting for process..."
  let exitCode ← child.wait
  IO.println s!"Exit code: {exitCode}"

  IO.println "\n=== Test Complete ==="
