# CLean New Architecture: Typeclass-Based Device Translation

## Overview

This document describes the fundamentally new approach to converting Lean code into GPU kernels with verification support. The architecture is inspired by GPUVerify's CUDA→Boogie translation pipeline but adapted for Lean's type system and proof capabilities.

## Architecture Components

### 1. Device IR (`CLean/DeviceIR.lean`)

The **Device Intermediate Representation** is the core target IR that sits between Lean code and CUDA/verification backends.

**Key types:**
- `DType`: Device types (int, nat, float, bool, array, tuple, struct)
- `DExpr`: Pure expressions (literals, variables, binops, GPU intrinsics like threadIdx.x)
- `DStmt`: Statements with effects (assign, store, if-then-else, for loops, barriers)
- `Kernel`: Complete kernel definition with parameters, locals, arrays, and body

**Example IR:**
```lean
{ name := "simpleAdd"
  params := [{ name := "N", ty := .nat }]
  locals := [{ name := "gval0", ty := .float }]
  globalArrays := [{ name := "R", ty := .array .float, space := .global }]
  body := .seq (.assign "gval0" ...) (.store ...) }
```

### 2. Translation System (`CLean/DeviceTranslation.lean`)

Typeclass-based framework for marking Lean types/functions as translatable to device code.

**Key typeclasses:**
```lean
class ToCudaType (α : Type) where
  deviceType : DType              -- Device representation
  encode : α → DeviceValue        -- For CPU execution
  decode : DeviceValue → Option α -- Back to Lean

class ToCudaFn {α β} (f : α → β) where
  translateFn : MetaM (Option DStmt)  -- Generate device IR
```

**Attribute:**
- `@[device_eligible]` - marks functions for automatic translation

**Environment extension:**
- Stores translated IR for reuse across compilation units

### 3. Standard Library Instances (`CLean/DeviceInstances.lean`)

Provides `ToCudaType` instances for common Lean types:
- `Nat` → `.nat` (mapped to int with bounds checks)
- `Int` → `.int`
- `Float` → `.float`
- `Bool` → `.bool`
- `Array α` → `.array (deviceType α)` (when `α` is translatable)
- `α × β` → `.tuple [...]`

These form the foundation - users can extend with their own types.

### 4. KernelOps Typeclass (`CLean/KernelOps.lean`)

**Tagless-final interface** for writing GPU kernels that can have multiple interpretations.

```lean
class KernelOps (m : Type → Type _) where
  Expr : Type  -- Expression type for this interpretation

  -- Lifting values
  natLit : Nat → Expr
  floatLit : Float → Expr

  -- Operations
  add, mul, lt, ... : Expr → Expr → Expr

  -- GPU intrinsics
  globalIdxX : m Expr
  threadIdxX : m Expr
  blockDimX : m Expr

  -- Memory
  globalGet : String → Expr → m Expr
  globalSet : String → Expr → Expr → m Unit
  sharedGet, sharedSet : ...

  -- Control flow
  ifThenElse : Expr → m Unit → m Unit → m Unit
  forLoop : String → Expr → Expr → (Expr → m Unit) → m Unit

  -- Synchronization
  barrier : m Unit
```

**Key insight:** Write kernels **once**, interpret **multiple ways**:
- Execution interpretation (actual computation)
- IR-building interpretation (extract DeviceIR)
- Future: simulation, optimization, etc.

### 5. IR Builder (`CLean/KernelBuilder.lean`)

**KernelOps instance that builds DeviceIR instead of executing.**

```lean
instance : KernelOps KernelBuilderM where
  Expr := DExpr  -- Expressions are IR nodes

  natLit n := .intLit (Int.ofNat n)
  add a b := .binop .add a b

  globalIdxX := do
    -- Build IR for: blockIdx.x * blockDim.x + threadIdx.x
    return .binop .add
      (.binop .mul (.blockIdx .x) (.blockDim .x))
      (.threadIdx .x)

  globalGet name idx := do
    let varName ← freshVar "gval"
    emitStmt (.assign varName (.index (.var name) idx))
    return .var varName

  globalSet name idx val := do
    emitStmt (.store (.var name) idx val)
    registerGlobal name (.array .float)
```

**Builder monad:**
- Tracks fresh variable generation
- Accumulates statements
- Registers locals, global arrays, shared arrays
- Final result: complete `Kernel` structure

### 6. Meta-Level Reflection (`CLean/DeviceReflection.lean`)

Provides `#device` command for automatic translation of Lean definitions to DeviceIR.

```lean
#device myFunction
-- Walks the Lean Expr, pattern-matches operations, generates IR
```

Currently supports:
- Literals, variables
- Binary operations (+, -, *, /, <, <=, ==, etc.)
- Monadic bind sequences
- Let bindings
- If-then-else

**Status:** Basic implementation exists but needs more work to handle arbitrary Lean code.

## Usage Example

```lean
-- 1. Write a kernel using KernelOps interface
def simpleAddKernel {m} [Monad m] [KernelOps m] (N : Nat) : m Unit := do
  let i ← KernelOps.globalIdxX
  let nExpr := KernelOps.natLit N

  KernelOps.ifThenElse (KernelOps.lt i nExpr)
    (do
      let x ← KernelOps.globalGet "X" i
      let y ← KernelOps.globalGet "Y" i
      let sum := KernelOps.add x y
      KernelOps.globalSet "R" i sum)
    (pure ())

-- 2. Extract IR using builder instance
def simpleAddKernelIR : Kernel :=
  buildKernel "simpleAdd"
    [{ name := "N", ty := .nat }]
    (simpleAddKernel 1024)

-- 3. View the generated IR
#eval IO.println s!"Kernel IR:\n{repr simpleAddKernelIR}"
```

**Output:** Complete DeviceIR with:
- Params: `[{name := "N", ty := .nat}]`
- Locals: `[{name := "gval0", ty := .float}, {name := "gval1", ty := .float}]`
- GlobalArrays: `[{name := "R", ty := .array .float, space := .global}]`
- Body: Sequence of assign/store statements wrapped in if-then-else

## Comparison to GPUVerify

| **GPUVerify** | **CLean** |
|---------------|-----------|
| CUDA `.cu` → clang → LLVM IR | Lean code → KernelOps → DeviceIR |
| Bugle: LLVM IR → `.gbpl` (Boogie) | Tagless-final: direct IR building |
| Boogie axioms for GPU semantics | Lean inductives + formal semantics |
| C# transformation passes | Lean meta-programming |
| 2-thread dualisation for races | TODO: similar approach |
| Z3 for verification | Lean proofs |

**Key differences:**
- GPUVerify works on compiled LLVM IR (lost high-level structure)
- CLean works directly on Lean AST (full type information, proof context)
- GPUVerify uses external SMT solver
- CLean uses internal proof assistant (more powerful, interactive)

## What We've Built So Far

✅ **Core IR** (DeviceIR with types, expressions, statements)
✅ **Translation typeclasses** (ToCudaType, ToCudaFn)
✅ **Standard library instances** (Nat, Float, Array, tuples)
✅ **KernelOps interface** (tagless-final DSL for kernels)
✅ **IR-building instance** (KernelBuilderM successfully extracts IR)
✅ **Basic reflection** (DeviceReflection with pattern matching)
✅ **Working examples** (simpleAdd, prefixSum successfully generate IR)

## Next Steps

### Immediate (to complete MVP):

1. **Execution instance** - wrap existing `runKernelCPU` as `KernelOps` instance
   - Allows same kernel code to run on CPU for testing

2. **CUDA backend** - `DeviceIR → CUDA AST → .cu code`
   - Pretty-printer for CUDA syntax
   - Handle thread/block dimensions, shared memory declarations

3. **Formal semantics** - operational semantics for DeviceIR
   - `evalExpr : DState → DExpr → DeviceValue → Prop`
   - `evalStmt : DState → DStmt → DState → Prop`
   - Enables proving correctness properties

### Medium-term:

4. **Enhanced reflection** - handle more Lean patterns
   - Mutable variables (`let mut x := ...; x := ...`)
   - For-in syntax sugar
   - Pattern matching
   - Function calls to `@[device_eligible]` functions

5. **Type tracking** - infer element types for arrays
   - Currently assumes `.float` everywhere
   - Need to thread type information through builder

6. **Optimization passes** - simplify generated IR
   - Constant folding
   - Dead code elimination
   - Common subexpression elimination

### Long-term:

7. **Race detection** - 2-thread dualisation like GPUVerify
   - Instrument IR with access tracking
   - Generate assertions for conflicting accesses

8. **Barrier invariants** - verify synchronization correctness
   - User-annotated invariants
   - Automatic inference for simple patterns

9. **Functional correctness** - relate kernel to specification
   - `theorem saxpy_correct : ∀ input, runKernel input = spec input`

10. **Integration with examples.lean** - migrate existing kernels
    - Saxpy, transpose, matrix multiply, prefix sum
    - Prove safety + correctness for each

## Design Decisions & Rationale

### Why tagless-final (KernelOps) instead of direct AST?

**Pros:**
- Write kernels once, interpret multiple ways (execute, build IR, optimize, etc.)
- Type-safe: impossible to build ill-typed IR
- Composable: can define helper functions at KernelOps level
- Familiar syntax: looks like normal monadic Lean code

**Cons:**
- More complex initial setup
- Harder to debug (which interpretation is running?)
- Requires understanding of higher-order abstractions

**Verdict:** Worth it for scalability and type safety.

### Why separate DeviceIR from Lean Expr?

Could have worked directly on Lean's `Expr` type, but:
- Too general (arbitrary recursion, dependent types, classical logic)
- CUDA only supports a tiny fragment
- Easier to define semantics/codegen for custom IR
- Can add GPU-specific nodes (threadIdx, barrier) naturally

### Why not use Lean's C backend?

Lean compiles to C with `lean_object*` everywhere (GC, boxing, runtime).
CUDA kernels need:
- Plain old data (POD) types
- No GC or heap allocation
- Direct control over shared memory

So we need a **separate, restricted compilation path** for device code.

### Typeclass extensibility vs monolithic compiler?

We could have hard-coded translation rules for Nat, Float, Array, etc.
Instead, we use typeclasses so:
- Users can add their own types (`struct Vec3`)
- Standard library authors can mark functions as device-eligible
- Future: automatic derivation for simple types

This mirrors Lean's philosophy: extensible, user-driven.

## Testing

See `test_new_architecture.lean` for working examples:

```bash
lake env lean --run test_new_architecture.lean
```

**Tests:**
- `simpleAddKernel` - basic array addition with bounds check
- `prefixSumKernel` - shared memory + barriers + loops

Both successfully generate correct DeviceIR.

## Files Created

```
CLean/
├── DeviceIR.lean           # Core IR types
├── DeviceTranslation.lean   # Typeclass framework
├── DeviceInstances.lean     # Standard library instances
├── KernelOps.lean           # Tagless-final interface
├── KernelBuilder.lean       # IR-building instance
└── DeviceReflection.lean    # Meta-level translation (WIP)

test_new_architecture.lean   # Working examples
NEW_ARCHITECTURE.md          # This document
```

## Summary

We've built a **scalable, typeclass-driven translation system** that:
- Allows writing GPU kernels in natural Lean syntax
- Extracts verifiable IR automatically
- Will support both execution (CPU/GPU) and formal verification
- Is extensible to new types and operations

This is a **solid foundation** for the full CLean verification system.

The key innovation over the previous macro/Extract.lean approaches:
- **No more silent failures**: If code can't be translated, we know at compile time
- **No more loose bvar errors**: We work with properly instantiated expressions
- **Extensible**: Users add types/functions via typeclasses, not by editing core code
- **Provable**: Everything is Lean data, can be reasoned about formally
