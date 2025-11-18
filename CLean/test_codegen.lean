import CLean.VerifyIR
import CLean.CodeGen

open CLean.VerifyIR
open CLean.CodeGen

/-! # Code Generation Tests

This file manually constructs simple VKernel examples and displays
the generated CUDA code to verify the pipeline works correctly.
-/

/-- Simple SAXPY-like kernel in VerifyIR:
    r[i] = alpha * x[i] + y[i]
    where i = blockIdx.x * blockDim.x + threadIdx.x
-/
def saxpyVKernel : VKernel := {
  name := `saxpy

  -- Parameters: alpha (scalar), N (size), x, y, r (arrays)
  params := [
    { name := `alpha, type := .float, uniformity := .uniform, memorySpace := .local },
    { name := `N, type := .nat, uniformity := .uniform, memorySpace := .local }
  ]

  -- Global arrays
  globalArrays := [
    { name := `x, type := .array .float, uniformity := .uniform, memorySpace := .global },
    { name := `y, type := .array .float, uniformity := .uniform, memorySpace := .global },
    { name := `r, type := .array .float, uniformity := .uniform, memorySpace := .global }
  ]

  -- Local variables
  locals := [
    { name := `i, type := .nat, uniformity := .nonUniform, memorySpace := .local },
    { name := `xi, type := .float, uniformity := .nonUniform, memorySpace := .local },
    { name := `yi, type := .float, uniformity := .nonUniform, memorySpace := .local }
  ]

  sharedArrays := []

  -- Kernel body
  body := [
    -- i = blockIdx.x * blockDim.x + threadIdx.x
    { stmt := .assign `i (.add (.mul .blockIdX .blockDimX) .threadIdX)
      predicate := .constBool true },

    -- if (i < N)
    { stmt := .ite (.lt (.var `i) (.var `N))
        -- then branch
        [
          -- xi = x[i]
          { stmt := .read ⟨`x, .var `i, .global⟩ `xi
            predicate := .constBool true },

          -- yi = y[i]
          { stmt := .read ⟨`y, .var `i, .global⟩ `yi
            predicate := .constBool true },

          -- r[i] = alpha * xi + yi
          { stmt := .write ⟨`r, .var `i, .global⟩
              (.add (.mul (.var `alpha) (.var `xi)) (.var `yi))
            predicate := .constBool true }
        ]
        -- else branch (empty)
        []
      predicate := .constBool true }
  ]
}

/-- Matrix transpose kernel in VerifyIR:
    Demonstrates shared memory and barriers
-/
def transposeVKernel : VKernel := {
  name := `transpose

  params := [
    { name := `width, type := .nat, uniformity := .uniform, memorySpace := .local },
    { name := `height, type := .nat, uniformity := .uniform, memorySpace := .local }
  ]

  globalArrays := [
    { name := `input, type := .array .float, uniformity := .uniform, memorySpace := .global },
    { name := `output, type := .array .float, uniformity := .uniform, memorySpace := .global }
  ]

  -- Shared memory tile (TILE_SIZE x TILE_SIZE)
  sharedArrays := [
    { name := `tile, type := .array .float, uniformity := .uniform, memorySpace := .shared }
  ]

  locals := [
    { name := `x, type := .nat, uniformity := .nonUniform, memorySpace := .local },
    { name := `y, type := .nat, uniformity := .nonUniform, memorySpace := .local },
    { name := `tileIdx, type := .nat, uniformity := .nonUniform, memorySpace := .local },
    { name := `temp, type := .float, uniformity := .nonUniform, memorySpace := .local }
  ]

  body := [
    -- x = blockIdx.x * blockDim.x + threadIdx.x
    { stmt := .assign `x (.add (.mul .blockIdX .blockDimX) .threadIdX)
      predicate := .constBool true },

    -- y = blockIdx.y * blockDim.y + threadIdx.y
    { stmt := .assign `y (.add (.mul .blockIdY .blockDimY) .threadIdY)
      predicate := .constBool true },

    -- tileIdx = threadIdx.y * blockDim.x + threadIdx.x
    { stmt := .assign `tileIdx (.add (.mul .threadIdY .blockDimX) .threadIdX)
      predicate := .constBool true },

    -- Load from global to shared: tile[tileIdx] = input[y * width + x]
    { stmt := .ite (.land (.lt (.var `x) (.var `width)) (.lt (.var `y) (.var `height)))
        [
          { stmt := .read
              ⟨`input, .add (.mul (.var `y) (.var `width)) (.var `x), .global⟩
              `temp
            predicate := .constBool true },
          { stmt := .write ⟨`tile, .var `tileIdx, .shared⟩ (.var `temp)
            predicate := .constBool true }
        ]
        []
      predicate := .constBool true },

    -- Barrier to sync shared memory
    { stmt := .barrier
      predicate := .constBool true },

    -- Write transposed to global
    { stmt := .ite (.land (.lt (.var `x) (.var `height)) (.lt (.var `y) (.var `width)))
        [
          { stmt := .read ⟨`tile, .var `tileIdx, .shared⟩ `temp
            predicate := .constBool true },
          { stmt := .write
              ⟨`output, .add (.mul (.var `x) (.var `height)) (.var `y), .global⟩
              (.var `temp)
            predicate := .constBool true }
        ]
        []
      predicate := .constBool true }
  ]
}

/-- Vector addition kernel (simplest example) -/
def vectorAddVKernel : VKernel := {
  name := `vectorAdd

  params := [
    { name := `n, type := .nat, uniformity := .uniform, memorySpace := .local }
  ]

  globalArrays := [
    { name := `a, type := .array .float, uniformity := .uniform, memorySpace := .global },
    { name := `b, type := .array .float, uniformity := .uniform, memorySpace := .global },
    { name := `c, type := .array .float, uniformity := .uniform, memorySpace := .global }
  ]

  locals := [
    { name := `i, type := .nat, uniformity := .nonUniform, memorySpace := .local },
    { name := `temp_a, type := .float, uniformity := .nonUniform, memorySpace := .local },
    { name := `temp_b, type := .float, uniformity := .nonUniform, memorySpace := .local }
  ]

  sharedArrays := []

  body := [
    -- i = blockIdx.x * blockDim.x + threadIdx.x
    { stmt := .assign `i (.add (.mul .blockIdX .blockDimX) .threadIdX)
      predicate := .constBool true },

    -- if (i < n) { c[i] = a[i] + b[i]; }
    { stmt := .ite (.lt (.var `i) (.var `n))
        [
          { stmt := .read ⟨`a, .var `i, .global⟩ `temp_a
            predicate := .constBool true },
          { stmt := .read ⟨`b, .var `i, .global⟩ `temp_b
            predicate := .constBool true },
          { stmt := .write ⟨`c, .var `i, .global⟩
              (.add (.var `temp_a) (.var `temp_b))
            predicate := .constBool true }
        ]
        []
      predicate := .constBool true }
  ]
}

/-! ## Display Functions -/

def repeatChar (c : Char) (n : Nat) : String :=
  String.mk (List.replicate n c)

def printSeparator (title : String) : IO Unit := do
  IO.println ""
  IO.println (repeatChar '=' 70)
  IO.println s!"  {title}"
  IO.println (repeatChar '=' 70)

def printKernelInfo (k : VKernel) : IO Unit := do
  IO.println s!"Kernel: {k.name}"
  IO.println s!"Parameters: {k.params.length}"
  IO.println s!"Global Arrays: {k.globalArrays.length}"
  IO.println s!"Shared Arrays: {k.sharedArrays.length}"
  IO.println s!"Local Variables: {k.locals.length}"
  IO.println s!"Body Statements: {k.body.length}"

def testKernel (k : VKernel) : IO Unit := do
  printSeparator s!"Testing Kernel: {k.name}"
  printKernelInfo k
  IO.println ""
  IO.println "Generated CUDA Code:"
  IO.println (repeatChar '─' 70)
  let cudaCode := kernelToCuda k
  IO.println cudaCode

  -- Also show launch configuration example
  let launchConfig : LaunchConfig := {
    gridDim := (256, 1, 1)
    blockDim := (256, 1, 1)
  }
  let args := ["alpha", "N", "d_x", "d_y", "d_r"]
  let launchCode := genLaunchCode k.name.toString launchConfig args
  IO.println ""
  IO.println "Example Launch Code:"
  IO.println (repeatChar '─' 70)
  IO.println launchCode

/-! ## Run Tests -/

#eval do
  printSeparator "CUDA Code Generation Test Suite"
  IO.println "Testing VerifyIR → CUDA translation"
  IO.println ""

  -- Test 1: Simple vector addition
  testKernel vectorAddVKernel

  -- Test 2: SAXPY
  testKernel saxpyVKernel

  -- Test 3: Matrix transpose with barriers
  testKernel transposeVKernel

  printSeparator "All Tests Complete"
  IO.println "✓ All kernels generated successfully"
  IO.println ""
