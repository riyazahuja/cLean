# cLean: Verified GPU Kernels in Lean

**cLean** is a domain-specific language (DSL) embedded in Lean 4 for writing, verifying, and executing GPU kernels. It bridges the gap between high-level functional programming and low-level GPU performance, providing a verified pipeline from Lean code to CUDA execution.

## Features

*   **Lean DSL for GPU**: Write kernels using familiar Lean syntax with a monadic interface (`KernelM`).
*   **Automatic Verification**:
    *   **Race Freedom**: Automatically proves that kernels are free from data races using GPUVerify-style logic.
    *   **Functional Correctness**: Verify that your kernel implementation matches a high-level mathematical specification.
*   **End-to-End Execution**:
    *   Automatic compilation of Lean kernels to PTX (Parallel Thread Execution).
    *   Execution on NVIDIA GPUs via a lightweight C++ launcher.
    *   Type-safe JSON-based communication between Lean and the GPU.
*   **CPU Simulation**: Debug and test kernels purely in Lean with a faithful CPU simulator.

## Architecture

cLean operates through a multi-stage pipeline:

1.  **DSL (`CLean/GPU.lean`)**: Users define kernels in Lean.
2.  **DeviceIR (`CLean/DeviceIR.lean`)**: The DSL is elaborated into a simplified intermediate representation (DeviceIR).
3.  **Verification (`CLean/Verification/`)**:
    *   DeviceIR is translated to a verification-friendly IR (`KernelSpec`).
    *   Verification Conditions (VCs) are generated and proved using Lean's tactic framework.
4.  **Code Generation (`CLean/DeviceCodeGen.lean`)**: DeviceIR is compiled to CUDA C++.
5.  **Execution (`CLean/GPU/ProcessLauncher.lean`)**:
    *   The CUDA code is compiled to PTX using `nvcc`.
    *   A C++ host program (`gpu_launcher`) loads the PTX and executes it.
    *   Data is marshaled via JSON.

## Installation

### Prerequisites
*   **Lean 4**: [Install Lean](https://leanprover.github.io/lean4/doc/quickstart.html)
*   **CUDA Toolkit**: Ensure `nvcc` is in your PATH.

### Build
1.  Clone the repository:
    ```bash
    git clone https://github.com/riyazahuja/cLean.git
    cd cLean
    ```
2.  Build the Lean project:
    ```bash
    lake build
    ```
3.  Compile the GPU launcher:
    ```bash
    nvcc gpu_launcher.cpp -o gpu_launcher -lcuda -lcudart
    ```

## Usage

### 1. Defining a Kernel
Define your kernel arguments and body using the `kernelArgs` and `device_kernel` macros.

```lean
import CLean.GPU

kernelArgs SaxpyArgs(n: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel saxpyKernel : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let i ← globalIdxX
  if i < args.n then
    let val_x ← args.x.get i
    let val_y ← args.y.get i
    args.r.set i (args.alpha * val_x + val_y)
```

### 2. Verifying the Kernel
Use the verification infrastructure to prove safety.

```lean
import CLean.Verification.GPUVerifyStyle

def saxpySpec : KernelSpec :=
  deviceIRToKernelSpec saxpyKernelIR saxpyConfig saxpyGrid

theorem saxpy_safe : KernelSafe saxpySpec := by
  -- Automatic proof tactics
  prove_kernel_safety
```

### 3. Executing on GPU
Run the kernel using the process launcher.

```lean
import CLean.Examples.execution_examples

def runSaxpy : IO (Array Float) := do
  let x := #[1.0, 2.0, 3.0]
  let y := #[1.0, 1.0, 1.0]
  let result ← saxpyGPU 3 2.0 x y
  IO.println s!"Result: {result}"
```

## 📂 Directory Structure

*   `CLean/`: Source code.
    *   `GPU.lean`: Core DSL definitions.
    *   `DeviceIR.lean`: Intermediate Representation.
    *   `DeviceCodeGen.lean`: CUDA code generation.
    *   `Verification/`: Verification logic and tactics.
    *   `GPU/`: Execution runtime and launcher interface.
*   `Examples/`:
    *   `execution_examples.lean`: End-to-end GPU execution demos.
    *   `test_increment_verified.lean`: Verification examples.
    *   `simulator_examples.lean`: CPU simulation examples.
*   `gpu_launcher.cpp`: C++ host program for kernel execution.

## 📝 License
[MIT License](LICENSE)
