# cLean Codebase Organization

## File Categories

### 🎯 **CORE PRODUCTION FILES** (Keep These!)

#### **CLean/GPU.lean** (1080 lines)
**Purpose**: Core GPU DSL - KernelM monad, array operations, barriers, CPU simulation
**Status**: ✅ ESSENTIAL - Foundation of everything
**Contains**:
- `KernelM` monad (ReaderT/StateT wrapper)
- `GlobalArray`, `SharedArray` types
- `getArgs`, `globalIdxX`, `barrier` operations
- `kernelArgs` macro (generates argument structures)
- CPU runtime with barrier simulation
**Needed for**: All GPU kernel definitions

#### **CLean/DeviceIR.lean** (142 lines)
**Purpose**: Intermediate representation for GPU kernels
**Status**: ✅ ESSENTIAL - Target IR for extraction
**Contains**:
- `DExpr` (expressions: vars, binops, GPU indices)
- `DStmt` (statements: assign, store, if-then-else, for, barrier)
- `Kernel` structure (params, arrays, body)
- `MemorySpace` enum (global, shared, local)
**Needed for**: IR generation and CUDA codegen

#### **CLean/DeviceMacro.lean** (495 lines)
**Purpose**: Syntax-level extraction - `device_kernel` macro
**Status**: ✅ ESSENTIAL - Main extraction approach (WORKING!)
**Contains**:
- `device_kernel` macro
- Syntax pattern matching for:
  - GlobalArray/SharedArray operations
  - Barriers
  - Binary operations
  - If-then-else (with explicit do blocks)
  - For loops (with explicit do blocks)
- Generates both KernelM def AND DeviceIR
**Needed for**: test_working_features.lean, test_full_extraction.lean

#### **CLean/DeviceCodeGen.lean** (300 lines)
**Purpose**: CUDA C++ code generation from DeviceIR
**Status**: ✅ ESSENTIAL - Generates executable CUDA
**Contains**:
- Expression/statement translation to CUDA
- Shared memory declarations
- Barrier (`__syncthreads()`) generation
- Kernel signature generation
- Complete host program generation
- Launch configuration code
**Needed for**: Generating .cu files from kernels

---

### 📚 **WORKING TEST FILES** (Keep for reference)

#### **test_working_features.lean** (175 lines)
**Purpose**: Demonstrates all working features
**Status**: ✅ WORKING - Zero errors
**Contains**: 4 kernels:
1. Matrix transpose (shared memory + barrier)
2. Multi-barrier with 2 shared arrays
3. Complex arithmetic with shared memory
4. 3-point stencil (neighbor access)
**Use**: Reference for how to write kernels correctly

#### **test_cuda_generation.lean** (152 lines)
**Purpose**: CUDA code generation test suite
**Status**: ✅ WORKING - Generates 3 .cu files
**Contains**:
- SAXPY kernel
- Transpose kernel
- Stencil kernel
- Launch configurations
- File writing code
**Use**: Generate executable CUDA programs

#### **test_full_extraction.lean** (140 lines)
**Purpose**: Comprehensive extraction tests
**Status**: ✅ WORKING - Shows binary ops, SAXPY, complex compute
**Use**: Reference for extraction patterns

---

### 🔬 **ALTERNATIVE/EXPERIMENTAL FILES** (Optional - different approach)

#### **CLean/DeviceExtractor.lean** (459 lines)
**Purpose**: Elaboration-based extraction (alternative to DeviceMacro)
**Status**: ⚠️ ALTERNATIVE APPROACH - Works but uses elaboration
**Advantage**: Handles if-then-else and for loops without explicit do blocks
**Disadvantage**: Works on elaborated terms (more complex)
**Decision**: DeviceMacro is simpler; use DeviceExtractor only if you need better control flow

#### **CLean/KernelMacro.lean** (376 lines)
**Purpose**: Another macro-based extraction attempt
**Status**: ⚠️ EXPERIMENTAL - Similar to DeviceMacro but different approach
**Decision**: DeviceMacro supersedes this

#### **CLean/DeviceTranslation.lean**, **DeviceReflection.lean**, **DeviceInstances.lean**
**Purpose**: Support files for DeviceExtractor
**Status**: ⚠️ Only needed if using DeviceExtractor
**Decision**: Skip if using DeviceMacro

---

### 📊 **VERIFICATION FILES** (Keep if doing verification)

#### **CLean/VerifyIR.lean** (285 lines)
**Purpose**: Verification IR with semantics
**Status**: ✅ USEFUL for verification work
**Contains**:
- `VExpr`, `VStmt`, `VKernel` (verification IR)
- Semantic evaluation
- State tracking with barriers
**Needed for**: Formal verification proofs

#### **CLean/CodeGen.lean** (195 lines)
**Purpose**: CUDA codegen for VerifyIR (not DeviceIR)
**Status**: ⚠️ Different from DeviceCodeGen
**Decision**: Use DeviceCodeGen.lean instead (works with DeviceMacro output)

---

### 🗑️ **OBSOLETE/DEBUG TEST FILES** (Can Delete)

#### Test files that were debugging/experimental:
- `test_device_advanced.lean` - Has control flow issues, use test_working_features.lean instead
- `test_simple_features.lean` - Debugging file, not needed
- `test_device_clean.lean` - Early test, superseded
- `test_device_extraction.lean` - Testing DeviceExtractor
- `test_device_kernel_macro.lean` - Old macro tests
- `test_device_macro.lean` - Early DeviceMacro tests
- `test_extraction.lean` - Old extraction tests
- `test_extract_command.lean` - Command experiments
- `test_extract_letbindings.lean` - Debugging
- `test_kernelargs.lean` - kernelArgs debugging (fixed now)
- `test_check_saxpy.lean` - Early SAXPY test
- `test_macro.lean`, `test_macro_simple.lean`, `test_simple_macro.lean` - Early macro experiments
- `test_new_architecture.lean` - Architecture experiment
- `test_saxpy_macro.lean`, `test_transpose_macro.lean` - Specific kernel tests (superseded by test_working_features)
- `test_syntax_debug.lean` - Debugging
- `test_extraction_simple.lean` - Simple extraction test

#### Obsolete implementation files:
- `CLean/KernelMacro_simple.lean` - Simplified version, not used
- `CLean/KernelBuilder.lean` - Old builder approach
- `CLean/KernelOps.lean` - Old operations
- `CLean/Extract.lean` - Old extraction

---

### 📝 **DOCUMENTATION FILES** (Some are outdated)

#### Generated documentation (from this session):
- `DEVICE_MACRO_ADVANCED_FEATURES.md` - ✅ KEEP - Documents shared memory/barriers
- `CUDA_GENERATION_SUMMARY.md` - ✅ KEEP - Documents CUDA codegen
- `CODEBASE_ORGANIZATION.md` - ✅ THIS FILE

#### Old documentation:
- `NEW_ARCHITECTURE.md` - ⚠️ From earlier design, may be outdated
- `README.md` - Update needed
- `ffi_tutorial.md` - FFI tutorial (if doing C++ interop)
- `DEVICE_MACRO_STATUS.md`, `DEVICE_EXTRACTION_SUMMARY.md`, etc. - May be outdated

#### Directories with old logs:
- `logs/` - Old progress logs, can archive

---

### 📦 **GENERATED FILES** (Can delete, regenerate anytime)

- `saxpy.cu`, `transpose.cu`, `stencil.cu` - Generated CUDA programs
- `.lake/` - Build artifacts
- `*.olean`, `*.ilean`, `*.c` - Compiled artifacts

---

## 🎯 **MINIMAL WORKING SET**

For a **minimal working GPU verification system**, you need:

### Core (5 files):
```
CLean/
  ├── GPU.lean              # KernelM monad, array ops, barriers
  ├── DeviceIR.lean         # IR representation
  ├── DeviceMacro.lean      # device_kernel macro (extraction)
  └── DeviceCodeGen.lean    # CUDA code generation

CLean.lean                  # Module aggregator
```

### Optional but useful:
```
CLean/VerifyIR.lean        # For formal verification
test_working_features.lean  # Reference examples
test_cuda_generation.lean   # CUDA generation examples
```

### If you need elaboration-based extraction:
```
CLean/
  ├── DeviceExtractor.lean
  ├── DeviceTranslation.lean
  ├── DeviceReflection.lean
  └── DeviceInstances.lean
```

---

## 🗑️ **FILES TO DELETE**

### Test files (20+ files):
```bash
rm test_device_advanced.lean
rm test_simple_features.lean
rm test_device_clean.lean
rm test_device_extraction.lean
rm test_device_kernel_macro.lean
rm test_device_macro.lean
rm test_extraction.lean
rm test_extract_command.lean
rm test_extract_letbindings.lean
rm test_kernelargs.lean
rm test_check_saxpy.lean
rm test_macro.lean
rm test_macro_simple.lean
rm test_simple_macro.lean
rm test_new_architecture.lean
rm test_saxpy_macro.lean
rm test_transpose_macro.lean
rm test_syntax_debug.lean
rm test_extraction_simple.lean
```

### Obsolete implementation files:
```bash
rm CLean/KernelMacro_simple.lean
rm CLean/KernelBuilder.lean
rm CLean/KernelOps.lean
rm CLean/Extract.lean
rm CLean/KernelMacro.lean  # If keeping DeviceMacro
```

### Generated files:
```bash
rm *.cu  # Regenerate anytime from test_cuda_generation.lean
```

---

## 🔄 **DECISION TREE**

### "What extraction approach should I use?"

**DeviceMacro (Syntax-level)**:
- ✅ Simpler, faster
- ✅ Works great for most kernels
- ⚠️ Control flow needs explicit `do` blocks
- **Use for**: Standard GPU kernels with shared memory, barriers, arithmetic

**DeviceExtractor (Elaboration-level)**:
- ✅ Handles all control flow automatically
- ⚠️ More complex (works on elaborated terms)
- **Use for**: Kernels with complex control flow

**Recommendation**: Start with DeviceMacro, switch to DeviceExtractor only if needed.

---

## 📋 **CLEANUP CHECKLIST**

1. **Delete obsolete test files** (20+ files listed above)
2. **Delete obsolete implementation files** (4 files listed above)
3. **Keep minimal core** (5 essential files)
4. **Keep working tests** (2-3 reference files)
5. **Update documentation** (README.md with correct status)
6. **Archive logs/** (move to logs/archive/)

---

## 🎉 **WHAT YOU HAVE NOW**

### Working System:
✅ GPU DSL with shared memory and barriers
✅ Syntax-level extraction to DeviceIR
✅ CUDA code generation
✅ 4 working example kernels (transpose, stencil, etc.)
✅ Complete pipeline: Lean → DeviceIR → CUDA → Executable

### File Count:
- **Before cleanup**: 40+ files
- **After cleanup**: ~10-15 core files
- **Reduction**: 60-75% fewer files

### Lines of Code (core):
- GPU.lean: 1080 lines (monad + runtime)
- DeviceIR.lean: 142 lines (IR definition)
- DeviceMacro.lean: 495 lines (extraction)
- DeviceCodeGen.lean: 300 lines (CUDA codegen)
- **Total core**: ~2000 lines

**This is a lean, focused codebase ready for GPU verification work!** 🚀
