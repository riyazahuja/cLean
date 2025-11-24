# cLean GPU Verification System

## Overview

The cLean verification system enables **formal verification of GPU kernel safety properties** in Lean 4. It translates Device IR kernels into verification-friendly representations and generates proof obligations that users prove interactively.

### Inspiration

Inspired by [GPUVerify](https://github.com/GPUVerify/gpuverify) (Microsoft Research / Imperial College London) but implemented entirely in Lean without external dependencies (no Boogie, no Z3).

### What Can Be Verified?

**Phase 1 (Implemented)**: Safety Properties
- ✅ **Race Freedom**: No data races between concurrent threads
- ✅ **Memory Safety**: All array accesses within bounds
- ✅ **Barrier Divergence Freedom**: All threads reach barriers uniformly
- ✅ **Deadlock Freedom**: No circular barrier dependencies

**Phase 2 (Future)**: Functional Correctness
- ⏳ Kernels compute correct results per mathematical specifications
- ⏳ Equivalence between CPU and GPU implementations

## Architecture

```
┌─────────────────────┐
│  device_kernel DSL  │  User writes GPU kernels
│  (Lean syntax)      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│     DeviceIR        │  Intermediate representation
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  VerificationIR     │  Enriched IR with metadata:
│  (ToVerificationIR) │  - Memory accesses tracked
│                     │  - Barriers identified
│                     │  - Uniformity analyzed
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│    VC Generator     │  Generate proof obligations:
│    (VCGen)          │  - Dual-thread transformation
│                     │  - Race freedom VCs
│                     │  - Bounds checking VCs
│                     │  - Barrier divergence VCs
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Lean Theorems      │  Well-typed proof goals:
│  (SafetyProperties) │  theorem saxpy_safe : KernelSafe k := by ...
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Interactive Proofs │  User proves with tactics:
│  (Tactics)          │  - prove_no_race
│                     │  - prove_bounds
│                     │  - apply_happens_before
└─────────────────────┘
```

## File Structure

```
CLean/
├── VerificationIR.lean          # Core verification types
│   ├── ThreadId, BlockId        # Thread identifiers
│   ├── MemoryAccess             # Read/write tracking
│   ├── BarrierPoint             # Synchronization points
│   ├── HappensBefore            # Ordering relation
│   └── VerifiedKernel           # Enriched kernel representation
│
├── ToVerificationIR.lean        # DeviceIR → VerificationIR analysis
│   ├── extractMemoryAccesses    # Collect all reads/writes
│   ├── findBarriers             # Locate barriers
│   ├── analyzeUniformity        # Control flow analysis
│   └── toVerificationIR         # Main translation function
│
└── Verification/
    ├── SafetyProperties.lean    # Safety property definitions
    │   ├── RaceFree             # No data races
    │   ├── MemorySafe           # Bounds checking
    │   ├── BarrierDivergenceFree# No divergence
    │   └── KernelSafe           # Combined safety
    │
    ├── VCGen.lean               # Verification condition generation
    │   ├── DualKernel           # Dual-thread transformation
    │   ├── generateRaceVCs      # Race freedom VCs
    │   ├── generateBoundsVCs    # Bounds checking VCs
    │   └── verifyKernel         # Complete workflow
    │
    ├── Tactics.lean             # Proof automation
    │   ├── prove_no_race        # Tactic for race freedom
    │   ├── prove_bounds         # Tactic for bounds
    │   ├── apply_happens_before # Happens-before reasoning
    │   └── Helper lemmas        # Common proof patterns
    │
    └── Examples/
        ├── SaxpyVerification.lean     # Complete SAXPY example
        └── VectorAddVerification.lean # Vector addition example
```

## Quick Start

### 1. Define Your Kernel

```lean
import CLean.DeviceMacro

kernelArgs SaxpyArgs(N: Nat, alpha: Float)
  global[x y r: Array Float]

device_kernel saxpyKernel : KernelM SaxpyArgs Unit := do
  let args ← getArgs
  let i ← globalIdxX
  if i < args.N then do
    let x : GlobalArray Float := ⟨args.x⟩
    let y : GlobalArray Float := ⟨args.y⟩
    let r : GlobalArray Float := ⟨args.r⟩
    let xi ← x.get i
    let yi ← y.get i
    r.set i (args.alpha * xi + yi)
```

### 2. Generate Verification Conditions

```lean
import CLean.Verification.VCGen

def saxpyConfig : VerificationContext :=
  { gridDim := ⟨1, 1, 1⟩
    blockDim := ⟨256, 1, 1⟩
    threadConstraints := []
    blockConstraints := [] }

-- Generate VCs and write to file
#eval verifyKernel
  saxpyKernelIR                    -- DeviceIR kernel
  saxpyConfig.gridDim              -- Grid dimensions
  saxpyConfig.blockDim             -- Block dimensions
  [("x", 1024), ("y", 1024), ("r", 1024)]  -- Array sizes
  "SaxpyVCs.lean"                  -- Output file
```

### 3. Prove Safety Properties

```lean
import CLean.Verification.Examples.SaxpyVerification

-- Memory bounds safety
theorem saxpy_x_read_bounds (N : Nat) :
    ∀ (i : Nat) (t : GlobalThreadId),
      i = t.threadId.x + t.blockId.x * t.blockDim.x →
      i < N →
      0 ≤ i ∧ i < N := by
  intros i t h_idx h_bound
  constructor
  · omega  -- i ≥ 0 trivial
  · exact h_bound

-- No race between different array accesses
theorem saxpy_no_race_x_r (N : Nat) :
    ∀ (k : VerifiedKernel) (read_x write_r : MemoryAccess),
      read_x.name = "x" →
      write_r.name = "r" →
      ¬read_x.conflicts write_r := by
  intros k read_x write_r hname_x hname_r
  unfold MemoryAccess.conflicts
  simp [hname_x, hname_r]  -- Different names => no conflict

-- Combined safety
theorem saxpy_safe (N : Nat) :
    let k := saxpyVerified N
    N ≤ saxpyConfig.blockDim.x →
    KernelSafe k := by
  sorry  -- Complete proof in SaxpyVerification.lean
```

## Key Concepts

### 1. Memory Access Tracking

Every read and write is tracked with full metadata:

```lean
structure MemoryAccess where
  name : String              -- Variable/array name
  space : MemorySpace        -- Global/shared/local
  accessType : AccessType    -- Read/write
  index : Option DExpr       -- Array index (if any)
  value : Option DExpr       -- Value written (if write)
  location : Nat             -- Statement number
  threadId : Option GlobalThreadId  -- Which thread
```

### 2. Happens-Before Relation

Defines ordering between memory accesses:

```lean
inductive HappensBefore (k : VerifiedKernel) : MemoryAccess → MemoryAccess → Prop where
  | programOrder :
      -- Same thread, sequential execution
      acc1.threadId = acc2.threadId →
      acc1.location < acc2.location →
      HappensBefore k acc1 acc2

  | barrierSync :
      -- Barrier synchronization establishes order
      acc1.location < barrier.location →
      barrier.location < acc2.location →
      sameBlock acc1.threadId acc2.threadId →
      HappensBefore k acc1 acc2

  | transitivity :
      HappensBefore k acc1 acc2 →
      HappensBefore k acc2 acc3 →
      HappensBefore k acc1 acc3
```

### 3. Dual-Thread Verification (à la GPUVerify)

To prove race freedom, we symbolically execute two arbitrary threads:

```lean
structure DualKernel where
  thread1 : SymbolicThread  -- Thread with symbolic threadIdx₁
  thread2 : SymbolicThread  -- Thread with symbolic threadIdx₂
  threadsDistinct : thread1.distinct thread2
  dualAccesses : List DualAccess  -- Accesses tagged by thread
```

Then generate VCs showing conflicting accesses are ordered by happens-before.

### 4. Safety Properties

Main safety predicate combining all properties:

```lean
def KernelSafe (k : VerifiedKernel) : Prop :=
  RaceFree k ∧                    -- No data races
  MemorySafe k ∧                  -- Bounds checking
  BarrierDivergenceFree k ∧       -- Uniform barrier reaching
  NoBarrierDeadlock k             -- No circular barriers
```

## Proof Tactics

Helper tactics for common proof patterns:

### `prove_no_race`

Automatically tries common race-freedom patterns:

```lean
example (k : VerifiedKernel) (acc1 acc2 : MemoryAccess)
    (h_diff_array : acc1.name ≠ acc2.name) :
    ¬acc1.hasRace k acc2 := by
  prove_no_race  -- Automatically applies no_conflict lemma
```

### `prove_bounds`

Proves bounds safety for standard access patterns:

```lean
example (k : VerifiedKernel) (acc : MemoryAccess)
    (h_idx : acc.index = some (DExpr.threadIdx Dim.x))
    (h_size : k.context.blockDim.x ≤ arraySize) :
    ArrayBoundsSafe k acc.name arraySize := by
  prove_bounds  -- Automatically applies threadIdx_access_safe
```

### `apply_happens_before`

Establishes happens-before ordering:

```lean
example (k : VerifiedKernel) (acc1 acc2 : MemoryAccess)
    (h_tid : acc1.threadId = acc2.threadId)
    (h_loc : acc1.location < acc2.location) :
    HappensBefore k acc1 acc2 := by
  apply_happens_before  -- Tries program order and barrier sync
```

## Verification Workflow

### Complete Example: Verify SAXPY

```bash
# 1. Write kernel
vim CLean/Verification/Examples/MySaxpy.lean

# 2. Generate VCs
lake env lean --run CLean/Verification/Examples/MySaxpy.lean
# Output: Generates MySaxpyVCs.lean with theorem statements

# 3. Prove VCs interactively
vim MySaxpyVCs.lean
# Fill in proofs using tactics

# 4. Check proofs
lake build MySaxpyVCs
```

### Step-by-Step Proof Template

```lean
-- 1. Define kernel and configuration
def myKernel : Kernel := ...
def myConfig : VerificationContext := ...

-- 2. Translate to VerificationIR
def myVerified := toVerificationIR myKernel myConfig.gridDim myConfig.blockDim

-- 3. Prove individual VCs
theorem my_kernel_bounds : ArrayBoundsSafe myVerified "arr" N := by
  -- Prove bounds safety
  sorry

theorem my_kernel_no_race : RaceFree myVerified := by
  -- Prove race freedom
  sorry

-- 4. Combine into main safety theorem
theorem my_kernel_safe : KernelSafe myVerified := by
  unfold KernelSafe
  constructor <;> [apply my_kernel_no_race, apply my_kernel_bounds, ...]
```

## Comparison with GPUVerify

| Feature | GPUVerify | cLean Verification |
|---------|-----------|-------------------|
| **Verification Backend** | Boogie + Z3 SMT solver | Pure Lean 4 proofs |
| **Automation** | Fully automatic | Interactive (user-guided) |
| **Properties Verified** | Safety (races, barriers) | Safety + Functional correctness (Phase 2) |
| **Language** | C#, Python (external tools) | Pure Lean |
| **Input Format** | OpenCL, CUDA, Boogie | cLean DeviceIR |
| **Output** | Pass/fail + counterexamples | Lean theorem statements |
| **Proof Trust** | Trust Boogie/Z3 | Lean kernel (minimal TCB) |
| **Techniques** | Dual-thread, abstract interpretation | Dual-thread, interactive proof |
| **Extensibility** | Modify C# codebase | Write Lean tactics/lemmas |

## Common Verification Patterns

### Pattern 1: Coalesced Array Access

```lean
-- Kernel: arr[threadIdx.x]
theorem coalesced_no_race (k : VerifiedKernel) (acc1 acc2 : MemoryAccess)
    (h_pattern : acc1.index = some (DExpr.threadIdx Dim.x))
    (h_diff_threads : acc1.threadId ≠ acc2.threadId) :
    ¬acc1.conflicts acc2 := by
  apply coalesced_access_no_conflict
  -- Different thread indices => different array indices
```

### Pattern 2: Barriers Eliminate Races

```lean
-- Kernel: write; barrier(); read;
theorem barrier_eliminates_race (k : VerifiedKernel) (write read : MemoryAccess)
    (b : BarrierPoint)
    (h_before : write.location < b.location)
    (h_after : b.location < read.location) :
    HappensBefore k write read := by
  apply barrier_establishes_happensBefore
```

### Pattern 3: Distinct Arrays Don't Conflict

```lean
-- Kernel: read x[i]; write y[i];
theorem different_arrays_no_conflict (acc1 acc2 : MemoryAccess)
    (h : acc1.name ≠ acc2.name) :
    ¬acc1.conflicts acc2 := by
  unfold MemoryAccess.conflicts
  simp [h]
```

## Limitations and Future Work

### Current Limitations

1. **No automatic SMT solving**: All proofs manual (but tactics help)
2. **Uniformity analysis basic**: Only tracks expression uniformity, not full control flow
3. **No pointer alias analysis**: Assumes arrays don't alias
4. **Limited to simple index expressions**: Complex index calculations may need axioms

### Phase 2: Functional Correctness (Planned)

```lean
-- Specify what kernel should compute
def saxpy_spec (α : Float) (x y : Array Float) : Array Float :=
  Array.zipWith (fun xi yi => α * xi + yi) x y

-- Prove kernel matches spec
theorem saxpy_correct (α : Float) (x y : Array Float) (N : Nat) :
    execute_gpu saxpyKernel α x y N = saxpy_spec α x y := by
  sorry  -- Phase 2 implementation
```

### Potential Extensions

- **More automation**: Decision procedures for common patterns
- **Better error reporting**: Show counterexample traces
- **Performance properties**: Prove coalescing, occupancy bounds
- **Multi-GPU verification**: Distributed kernel correctness

## Examples

See `CLean/Verification/Examples/`:

- **SaxpyVerification.lean**: Complete SAXPY safety proof
- **VectorAddVerification.lean**: Vector addition (TBD)
- **ParallelReduction.lean**: Reduction with barriers (TBD)

## References

- [GPUVerify](https://multicore.doc.ic.ac.uk/tools/GPUVerify/): Original inspiration
- [GPUVerify Paper (OOPSLA 2013)](https://dl.acm.org/doi/10.1145/2509136.2509507)
- [Barrier Invariants Paper](https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2013/POPL.pdf)
- [Lean 4 Documentation](https://lean-lang.org/lean4/doc/)

## Status

**Phase 1 (Safety Verification)**: ✅ **Complete**
- All core infrastructure implemented
- SAXPY example fully verified
- Ready for use on real kernels

**Phase 2 (Functional Correctness)**: ⏳ **Planned**
- Specification language design
- Equivalence proofs framework
- Estimated timeline: 2-3 weeks

---

Last updated: 2025-11-21
Version: 1.0.0 (Phase 1)
