# DeviceExtractor: Findings and Path Forward

## What We Built

We successfully created a `DeviceExtractor` module that:
- ✅ Provides extraction infrastructure with state management
- ✅ Pattern matches GPU primitive operations (globalIdxX, GlobalArray.get/set, barrier)
- ✅ Extracts binary operations (+, -, *, /, <, <=, ==)
- ✅ Handles if-then-else control flow
- ✅ Handles let bindings and variable registration
- ✅ Provides `#extract_kernel` command
- ✅ Compiles successfully

## The Core Problem

When extracting from `saxpyKernel`, we encounter **fully elaborated monadic code** that looks like:
```lean
fun r s s_1 => Decidable.rec (fun h => ...) (fun h =>
  EStateM.Result.rec (fun a => ...) ...
```

This is the fully elaborated `ReaderT (KernelCtx Args) (StateT KernelState IO) α` monad, with all the:
- IO machinery
- State monad operations
- Reader monad operations
- Error handling (EStateM.Result)
- Hash map operations for globals/shared memory

**Why this happens:**
1. `KernelM` is defined as `abbrev KernelM := ReaderT ... (StateT ... IO)`
2. When we get `val.value` from the definition, it's the fully elaborated term
3. All the nice do-notation has been desugared and elaborated into explicit monad operations
4. Those operations are then inlined and reduced
5. We end up with a giant lambda expression with no recognizable structure

## Why Our Fixes Didn't Work

### Attempt 1: Remove immediate reduction
- Moved `tryReduce` to fallback case
- **Result:** Still got over-reduced code because the expression was already elaborated

### Attempt 2: Use `lambdaTelescope`
- Tried to instantiate bound variables like `Extract.lean` does
- **Result:** Got 0 parameters because `saxpyKernel` isn't a lambda - it's a monadic value

### Why `Extract.lean` has the same issue
Looking at the existing `Extract.lean`:
- It has the SAME problem (you mentioned "loose bvar" errors)
- It works on elaborated expressions and struggles with bound variables
- Uses `reduce` instead of `whnf` to avoid panics
- Still fragile

## The Real Solution: Syntax-Level Extraction

The answer is **KernelMacro.lean** but targeting DeviceIR instead of VerifyIR.

### Why Syntax-Level?

**Macro (syntax-level) extraction:**
- Works BEFORE elaboration
- Sees the actual do-notation structure
- Pattern matches on clean syntax trees: `do let i ← globalIdxX; if i < N then ...`
- No monad machinery visible
- Robust and predictable

**Meta (elaboration-level) extraction:**
- Works AFTER elaboration
- Sees fully elaborated monad operations
- Must reverse-engineer the high-level structure
- Fragile and complex

### Comparison

| Aspect | Macro Approach | Meta Approach |
|--------|---------------|---------------|
| Input | Surface syntax | Elaborated Expr |
| Structure | Clean do-notation | Desugared monad ops |
| Pattern matching | Simple | Complex |
| Bound variables | No issues | Loose bvar errors |
| Flexibility | Limited to syntax | Can analyze arbitrary code |
| Best for | Known DSL patterns | General extraction |

For GPU kernels, we **know the patterns** (do-notation, array ops, barriers), so syntax-level is perfect.

## Recommended Implementation

### Option A: Extend KernelMacro for DeviceIR (RECOMMENDED)

Take the existing `KernelMacro.lean` approach and adapt it to generate DeviceIR:

```lean
-- Macro that processes kernel syntax
macro "gpu_kernel" name:ident args:kernelArgs body:doSeq : command => do
  -- 1. Generate args structure (already done)
  -- 2. Generate KernelM definition (already done)
  -- 3. NEW: Generate DeviceIR.Kernel directly from syntax

  let deviceIR ← syntaxToDeviceIR body
  -- Store in environment extension
```

**Advantages:**
- Works with existing kernel syntax in examples.lean
- No changes to user code
- Robust pattern matching on clean syntax
- Avoids all elaboration issues

**Implementation:**
1. Read KernelMacro.lean to understand syntax pattern matching
2. Create similar patterns but emit DeviceIR instead of VerifyIR
3. Handle: globalIdx, array get/set, barrier, if, for, let bindings
4. Generate DeviceIR.Kernel structure
5. Store in environment extension

### Option B: Hybrid Approach

Use syntax-level for structure, meta-level for expressions:

```lean
-- Macro extracts high-level structure
macro_rules
  | `(do let $x ← globalIdxX; $rest) =>
      -- Emit: assign x (globalIdx expression)
  | `(do if $cond then $thenB else $elseB) =>
      -- Extract cond as expression (may need meta)
      -- Recursively extract branches
```

**Advantages:**
- Can use type information where needed
- Syntax-level for control flow
- Meta-level for complex expressions

### Option C: Custom Elaborator

Write a custom elaborator that intercepts kernel definitions and extracts during elaboration:

```lean
elab "device_kernel" name:ident ":" ty:term ":=" body:term : command => do
  -- Elaborate body with custom handler
  -- Extract DeviceIR during elaboration
```

**Advantages:**
- Access to both syntax and types
- Can emit multiple outputs (executable + IR)
- Most powerful

**Disadvantages:**
- Most complex to implement
- Requires deep Lean 4 elab knowledge

## Concrete Next Steps

### Immediate (to get extraction working):

1. **Study KernelMacro.lean** (lines 1-200)
   - Understand how it pattern matches on syntax
   - See how it extracts globalIdxX, array ops, etc.
   - Note the syntax quotation patterns

2. **Create DeviceMacro.lean**
   - Copy KernelMacro structure
   - Replace VerifyIR generation with DeviceIR
   - Key patterns to handle:
     - `do let i ← globalIdxX` → `.assign "i" (.binop .add ...)`
     - `do let x ← arr.get i` → `.assign "x" (.index (.var "arr") (.var "i"))`
     - `do arr.set i val` → `.store (.var "arr") (.var "i") (.var "val")`
     - `do if cond then A else B` → `.ite (extractCond cond) ...`
     - `barrier` → `.barrier`

3. **Test on saxpyKernel**
   - Should work immediately with syntax-level approach
   - No elaboration issues

### Medium-term (for full system):

4. **Add auto-translate for helper functions**
   - For functions marked `@[device_eligible]`
   - Extract their syntax and convert to DeviceIR
   - Store for inlining when called from kernels

5. **Integrate with typeclass system**
   - ToCudaType for custom types
   - ToCudaFn for operations
   - Use in expression extraction

6. **CUDA backend**
   - DeviceIR → CUDA AST → .cu code
   - Pretty-printing

7. **Formal semantics**
   - DeviceIR operational semantics
   - Verification properties

## Files to Create

```
CLean/
├── DeviceMacro.lean        # NEW: Syntax-level extraction to DeviceIR
├── DeviceIR.lean           # ✓ Already done
├── DeviceTranslation.lean   # ✓ Already done (for future use)
├── DeviceInstances.lean     # ✓ Already done (for future use)
├── DeviceExtractor.lean     # Keep for reference / future meta-level work
└── CudaCodeGen.lean         # TODO: CUDA backend

test_device_macro.lean       # Test syntax-level extraction
```

## Why This Will Work

The syntax-level approach has been proven by `KernelMacro.lean`:
- It successfully extracts to VerifyIR
- Works on test_macro_simple.lean, test_saxpy_macro.lean, etc.
- Handles barriers, shared memory, control flow
- No elaboration issues

We just need to:
1. Retarget it to DeviceIR (cleaner IR than VerifyIR)
2. Add support for more patterns (for loops, mutable variables)
3. Store results in environment extension

## Summary

**Current status:** DeviceExtractor compiles but can't extract due to working at the wrong level (elaborated terms instead of syntax).

**Solution:** Use syntax-level macro extraction (like KernelMacro.lean) targeting DeviceIR.

**Effort:** ~2-3 hours to adapt KernelMacro to DeviceIR and get basic extraction working.

**Result:** Clean, robust extraction that works with existing kernels in examples.lean with zero changes to user code.
