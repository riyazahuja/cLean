# cLean Repository Overview

This document provides a comprehensive analysis of the `cLean` repository, a Lean 4 framework for writing, executing, and verifying GPU kernels. The framework allows users to write GPU kernels in a subset of Lean (DSL), which can then be:
1.  **Simulated** on the CPU within Lean.
2.  **Compiled** to CUDA C++/PTX and executed on a physical GPU.
3.  **Verified** for safety properties (race freedom, barrier divergence) and functional correctness using a GPUVerify-style approach implemented entirely in Lean.

---

## Table of Contents
1.  [Root Directory](#root-directory)
2.  [Core & Intermediate Representation (IR)](#core--intermediate-representation-ir)
3.  [GPU Runtime & Execution](#gpu-runtime--execution)
4.  [Code Generation](#code-generation)
5.  [Semantics & Verification](#semantics--verification)
6.  [Examples](#examples)
7.  [Detailed API Reference](#detailed-api-reference)

---

## Root Directory

### `lakefile.lean`
Defines the Lake project configuration for `cLean`.
- **`package clean`**: Sets up the package configuration.
- **`lean_lib CLean`**: Defines the main library.
- **`lean_lib Examples`**: Defines the examples library.
- **`@[default_target] lean_exe clean`**: Defines the default executable target.

### `CLean.lean`
The main entry point for the library.
- Imports all necessary modules (`CLean.DeviceMacro`, `CLean.GPU`, `CLean.Verification.VCGen`, etc.) to expose the full API to consumers.

### `gpu_launcher.cpp`
A C++ host application responsible for loading and executing compiled PTX kernels on the GPU. It communicates with the Lean process via standard I/O (JSON).
- **`checkCudaErrors(val)`**: Macro/Function to check CUDA API call results and exit on error.
- **`loadPTX(ptxFile, kernelName, ...)`**: Loads a PTX file and retrieves the kernel function handle using the CUDA Driver API (`cuModuleLoad`, `cuModuleGetFunction`).
- **`parseInput(json)`**: Parses the JSON input from Lean containing scalar arguments and array data.
- **`main(argc, argv)`**:
    - Initializes the CUDA device.
    - Reads command-line arguments (PTX path, kernel name, grid/block dimensions).
    - Reads JSON input from `stdin`.
    - Allocates device memory (`cuMemAlloc`) and copies input data (`cuMemcpyHtoD`).
    - Sets up kernel arguments (`kernelParams`).
    - Launches the kernel (`cuLaunchKernel`).
    - Copies results back to host (`cuMemcpyDtoH`).
    - Prints the output as a JSON object to `stdout`.

---

## Core & Intermediate Representation (IR)

### `CLean/DeviceIR.lean`
Defines the Abstract Syntax Tree (AST) for the GPU kernel language.
- **`inductive DType`**: Represents data types supported in kernels (`float`, `int`, `uint`, `bool`, `void`, `pointer`, `array`).
- **`inductive DBinOp`**: Binary operations (`add`, `sub`, `mul`, `div`, `mod`, `and`, `or`, `lt`, `le`, `gt`, `ge`, `eq`, `neq`).
- **`inductive DUnOp`**: Unary operations (`neg`, `not`).
- **`inductive DExpr`**: Expressions in the IR.
    - `const`: Literals.
    - `var`: Variables.
    - `binOp` / `unOp`: Operations.
    - `index`: Array access (`arr[idx]`).
    - `call`: Function calls.
    - `intrinsic`: GPU intrinsics (e.g., `__syncthreads`).
- **`inductive DStmt`**: Statements in the IR.
    - `let`: Variable declaration.
    - `assign`: Variable assignment.
    - `store`: Array write (`arr[idx] = val`).
    - `if`: Conditional.
    - `for`: Loops.
    - `while`: Loops.
    - `seq`: Sequence of statements.
    - `expr`: Expression statement.
    - `noop`: No-operation.
- **`structure DeviceKernel`**: Represents a full kernel with `name`, `args`, and `body` (a `DStmt`).

### `CLean/DeviceMacro.lean`
Implements Lean macros and elaboration logic to transform Lean syntax into `DeviceIR`.
- **`declare_syntax_cat d_stmt / d_expr`**: Defines syntax categories for the DSL.
- **`extractDoItems`**: Helper to parse `do` blocks in the DSL.
- **`extractType`**: Maps Lean syntax types to `DType`.
- **`extractStmt`**: Recursively transforms Lean syntax terms into `DStmt`. Handles `let`, `if`, `for`, array access (`get`, `set`), and intrinsics.
- **`extractExpr`**: Recursively transforms Lean syntax terms into `DExpr`.
- **`kernelArgs`**: Macro to define a structure for kernel arguments (Global/Shared arrays and scalars). Generates `ToJson`/`FromJson` instances.
- **`device_kernel`**: The primary macro.
    - Defines a Lean function for simulation (`KernelM`).
    - Extracts the `DeviceIR` from the syntax.
    - Defines a constant `kernelName_IR` containing the `DeviceKernel` object.

### `CLean/DeviceExtractor.lean`
Helper functions for extracting IR from Lean `Syntax` objects.
- **`extractTypeFromSyntax`**: Converts syntax nodes to `DType`.
- **`extractArgs`**: Parses the argument list of a kernel.

### `CLean/DeviceInstances.lean`
Provides type class instances for the IR types.
- **`ToString` instances**: For `DType`, `DBinOp`, `DUnOp`, `DExpr`, `DStmt`.
- **`ToJson` / `FromJson` instances**: Allows serializing the IR (useful for debugging or external tools).

---

## GPU Runtime & Execution

### `CLean/GPU/Runtime.lean`
Implements the CPU simulator for executing kernels within Lean.
- **`inductive KernelValue`**: Represents runtime values (`int`, `float`, `arrayInt`, `arrayFloat`).
- **`structure KernelState`**: The state of the simulator, containing `globals` (global memory), `shared` (shared memory), and `locals` (local variables).
- **`abbrev KernelM`**: The monad for kernel execution (`StateT KernelState IO`).
- **`mkKernelState`**: Constructor for initializing the state.
- **`GlobalArray / SharedArray`**: Helper structures wrapping variable names for array access.
- **`globalIdxX/Y/Z`**: Simulates GPU thread indices (reads from a context or mocked values).
- **`barrier`**: Simulates `__syncthreads()` (currently a no-op in the single-threaded CPU simulation, or throws if used incorrectly in a non-concurrent context).
- **`runKernelCPU`**: The main driver for CPU simulation.
    - Iterates over the grid and blocks.
    - Sets up thread indices.
    - Executes the kernel body for each "thread".
    - **Note**: The current CPU simulator appears to run threads sequentially, which may not catch race conditions that the verification layer does.

### `CLean/GPU/ProcessLauncher.lean`
Manages the compilation and execution of kernels on the GPU via `gpu_launcher`.
- **`structure LaunchConfig`**: Defines grid and block dimensions.
- **`buildLauncherInput`**: Constructs the JSON string required by `gpu_launcher`.
- **`runKernelGPU`**:
    - Takes a `DeviceKernel` and input data.
    - Calls `compileKernelToPTX` (if not cached).
    - Spawns the `gpu_launcher` subprocess.
    - Feeds JSON input to `stdin`.
    - Parses JSON output from `stdout`.
    - Returns the result as a typed structure.

### `CLean/GPU/KernelCache.lean`
Handles caching of compiled PTX kernels to avoid redundant compilation.
- **`computeHash`**: Computes a hash of the kernel IR/code.
- **`getCachePath`**: Determines where to store the PTX file.
- **`isCached`**: Checks if a valid PTX exists.

### `CLean/GPU.lean`
Aggregates the GPU runtime modules.
- **`GpuDSL`**: Namespace exposing common types and functions for users (`KernelM`, `globalIdxX`, etc.).

---

## Code Generation

### `CLean/DeviceCodeGen.lean`
Transpiles `DeviceIR` into CUDA C++ code.
- **`genType`**: Converts `DType` to C++ types (`float`, `int`, `float*`, etc.).
- **`genBinOp / genUnOp`**: Converts operators to C++ syntax.
- **`genExpr`**: Converts `DExpr` to C++ expressions.
- **`genStmt`**: Converts `DStmt` to C++ statements. Handles indentation and block scoping.
- **`kernelToCuda`**: Generates the `__global__ void kernel(...)` function definition.
- **`genCompleteCudaProgram`**: Generates a full C++ file including headers and the kernel, suitable for compilation by `nvcc`.

### `CLean/DeviceTranslation.lean`
Provides extensibility for custom types.
- **`class ToCudaType`**: A type class that allows user-defined Lean types to be used in kernels.
    - `deviceType`: The corresponding `DType`.
    - `encode`: How to convert the Lean value to `DExpr` (or `Json`).
    - `decode`: How to reconstruct the Lean value.

---

## Semantics & Verification

### `CLean/Verification/VCGen.lean`
The core Verification Condition Generator.
- **`generateVC`**: Takes a `DeviceKernel` and a safety property (e.g., `RaceFree`).
- **`vcGenStmt`**: Recursively generates logical formulas (Lean `Prop`) from `DStmt`.
    - Uses a **two-thread reduction** model: generates assertions that hold for any pair of threads `t1, t2`.
    - For `RaceFree`: Checks if `AccessPattern`s of two threads overlap and at least one is a write.
    - For `BarrierDivergence`: Checks that control flow reaching a barrier is uniform across threads.

### `CLean/Verification/SafetyProperties.lean`
Formal definitions of safety properties.
- **`structure KernelSpec`**: Represents the abstract behavior of a kernel for verification.
- **`def RaceFree`**: Predicate asserting no data races exist.
    - `∀ t1 t2, t1 ≠ t2 → ¬(access(t1) ∩ access(t2) ≠ ∅ ∧ (isWrite(t1) ∨ isWrite(t2)))`.
- **`def BarrierUniform`**: Predicate asserting all threads reach the same barriers under the same conditions.
- **`def KernelSafe`**: Combines `RaceFree` and `BarrierUniform`.

### `CLean/Verification/GPUVerifyStyle.lean`
Implements the "GPUVerify" logic (symbolic execution + two-thread reduction).
- **`inductive AddressPattern`**: Symbolic representation of memory accesses (e.g., `base + tid * stride`).
- **`AccessExtractor`**: Analyzes `DStmt` to collect all memory reads and writes.
- **`HasRace`**: Decides if two `AddressPattern`s can alias.
- **`SeparatedByBarrier`**: Logic to determine if two accesses are separated by a synchronization point.

### `CLean/Verification/Tactics.lean`
Lean tactics to automate proofs.
- **`solve_race_freedom`**: Tactic that unfolds definitions, applies `VCGen` logic, and attempts to solve the resulting arithmetic goals (using `simp`, `linarith`, etc.).
- **`solve_barrier_uniformity`**: Tactic for divergence proofs.

### `CLean/Verification/FunctionalSpecGen.lean`
Generates functional correctness specifications.
- **`generateFunctionalSpec`**: Creates a theorem statement asserting that the kernel implementation matches a high-level mathematical function.

### `CLean/ToGPUVerifyIR.lean` / `CLean/ToVerificationIR.lean`
Converters that transform the raw `DeviceIR` into specialized IRs used by the verification modules.
- **`deviceIRToKernelSpec`**: Converts `DeviceIR` to the `KernelSpec` structure used in `SafetyProperties`.

### `CLean/Semantics/DeviceSemantics.lean`
Defines the operational semantics of the `DeviceIR`.
- **`step`**: Small-step semantics for the kernel execution.
- Used as the ground truth for verification.

---

## Examples

### `Examples/execution_examples_verified.lean`
Contains fully verified kernels.
- **`Saxpy`**:
    - `saxpyKernel`: The implementation.
    - `saxpySpec`: The specification.
    - `saxpy_safe`: Theorem proving `KernelSafe` (race-free + barrier-uniform).
- **`ExclusiveScan`**:
    - `upsweepKernel` / `downsweepKernel`: Hillis-Steele scan phases.
    - `upsweep_safe` / `downsweep_safe`: Safety proofs.
- **`BasicMatMul`**:
    - `matmulKernel`: Naive matrix multiplication.
    - `matmul_safe`: Safety proof.

### `Examples/execution_examples.lean`
Focuses on execution (CPU simulation and GPU launching) without the proof overhead.
- **`BetterMatMul`**: Tiled matrix multiplication using shared memory.
- **`SharedMemTranspose`**: Matrix transpose using shared memory tiles to coalesce global memory accesses.
- **`SharedPrefixSum`**: Single-block prefix sum using shared memory.

### `CLean/working_gpu.lean`
A complete integration test script.
- Defines a kernel.
- Compiles it.
- Prepares JSON input.
- Launches `gpu_launcher`.
- Validates the output against expected values.

---

## Detailed API Reference

### Module: `CLean.DeviceIR`

#### `inductive DType`
Represents the data types available in the device IR.
- `float`: 32-bit floating point.
- `int`: 32-bit signed integer.
- `uint`: 32-bit unsigned integer.
- `bool`: Boolean.
- `void`: Void type (for statements).
- `pointer (t : DType)`: Pointer to another type.
- `array (t : DType) (size : Option Nat)`: Array of type `t`, optionally with a fixed size.

#### `inductive DBinOp`
Binary operators.
- `add`, `sub`, `mul`, `div`, `mod`: Arithmetic.
- `and`, `or`: Logical/Bitwise.
- `lt`, `le`, `gt`, `ge`, `eq`, `neq`: Comparison.

#### `inductive DExpr`
Expressions.
- `const (v : String) (t : DType)`: Constant value.
- `var (name : String) (t : DType)`: Variable reference.
- `binOp (op : DBinOp) (lhs rhs : DExpr)`: Binary operation.
- `unOp (op : DUnOp) (e : DExpr)`: Unary operation.
- `index (arr : DExpr) (idx : DExpr)`: Array indexing.
- `call (name : String) (args : List DExpr)`: Function call.
- `intrinsic (name : String) (args : List DExpr)`: Intrinsic call (e.g., `__syncthreads`).

#### `inductive DStmt`
Statements.
- `let (name : String) (t : DType) (val : DExpr)`: Variable declaration.
- `assign (name : String) (val : DExpr)`: Variable assignment.
- `store (arr : String) (idx : DExpr) (val : DExpr)`: Array write.
- `if (cond : DExpr) (thenStmt : DStmt) (elseStmt : Option DStmt)`: Conditional.
- `for (var : String) (start : DExpr) (end : DExpr) (body : DStmt)`: For loop.
- `while (cond : DExpr) (body : DStmt)`: While loop.
- `seq (stmts : List DStmt)`: Sequence block.
- `expr (e : DExpr)`: Expression statement.
- `noop`: No operation.

### Module: `CLean.DeviceCodeGen`

#### `def genType (t : DType) : String`
Maps `DType` to C++ type strings.
- `float` -> `"float"`
- `int` -> `"int"`
- `pointer t` -> `genType t ++ "*"`

#### `def genBinOp (op : DBinOp) : String`
Maps `DBinOp` to C++ operators.
- `add` -> `"+"`
- `eq` -> `"=="`
- etc.

#### `def genExpr (e : DExpr) : String`
Recursively generates C++ code for expressions.
- Handles operator precedence by wrapping binary ops in parentheses.
- Formats function calls `name(arg1, arg2)`.

#### `def genStmt (s : DStmt) (indent : Nat) : String`
Recursively generates C++ code for statements.
- Manages indentation.
- Formats control structures (`if`, `for`, `while`) with braces.
- Generates `__syncthreads();` for barrier intrinsics.

#### `def kernelToCuda (k : DeviceKernel) : String`
Generates the kernel function definition.
- Signature: `extern "C" __global__ void <name>(<args>)`.
- Body: `genStmt k.body`.

### Module: `CLean.GPU.Runtime`

#### `structure KernelState`
- `globals : HashMap String KernelValue`: Global memory storage.
- `shared : HashMap String KernelValue`: Shared memory storage.
- `locals : HashMap String KernelValue`: Local variable storage.

#### `def runKernelCPU`
The CPU simulation loop.
```lean
def runKernelCPU (grid block : Dim3) (args : argsType) (state : KernelState) (kernel : KernelM argsType Unit) : IO KernelState
```
- Loops `z` from 0 to `grid.z`, `y` to `grid.y`, `x` to `grid.x`.
- Inside, loops `tz`, `ty`, `tx` for threads.
- Updates `globalIdx` / `localIdx` in the context.
- Executes the kernel body.
- **Limitation**: Serial execution means barriers are ignored or treated as no-ops, so race conditions won't be detected dynamically.

### Module: `CLean.Verification.VCGen`

#### `def generateVC (k : DeviceKernel) (prop : SafetyProperty) : Prop`
The entry point for verification.
- Transforms the kernel body into a logical proposition.
- For `RaceFree`:
  - Extracts all `AccessPattern`s.
  - Generates `∀ t1 t2, t1 ≠ t2 → ¬HasRace(access(t1), access(t2))`.

#### `def vcGenStmt (s : DStmt) (ctx : VCContext) : Prop`
Recursively generates VCs for statements.
- `seq`: `vcGenStmt s1 ∧ vcGenStmt s2`.
- `if`: `(cond → vcGenStmt then) ∧ (¬cond → vcGenStmt else)`.
- `for`: Unrolls loops or generates loop invariants (simplified in current version).

### Module: `CLean.Verification.GPUVerifyStyle`

#### `inductive AddressPattern`
- `base (name : String)`: Base array address.
- `offset (base : AddressPattern) (off : Int)`: Constant offset.
- `linear (base : AddressPattern) (dim : String) (coeff : Int)`: Linear access `base + tid * coeff`.

#### `def HasRace (a1 a2 : AddressPattern) : Prop`
Determines if two patterns overlap.
- If bases are different -> False.
- If linear patterns `c1*t1 + o1 = c2*t2 + o2` have integer solutions for `t1 ≠ t2` -> True.

### Module: `CLean.ToSemanticBody` / `CLean.ToVerificationIR`

#### `def toSemanticBody (s : DStmt) : SemanticBody`
Converts the imperative `DStmt` into a cleaner, semantic representation used for functional verification.
- Removes `noop`s.
- Flattens nested sequences.
- Simplifies expressions where possible.

#### `def deviceIRToVerificationIR (k : DeviceKernel) : VerificationKernel`
Prepares the kernel for the verification pipeline.
- Annotates statements with metadata needed for VC generation (e.g., source line numbers).
- Normalizes control flow structures.
