# CLean Declaration Hierarchy

This document maps the project bottom-up, from the foundational datatypes through executable semantics, proof infrastructure, PTX ingestion, and the SAXPY/matmul examples. It intentionally includes private fixtures because they are part of how the examples are constructed, but the main hierarchy is the public proof and execution spine.

Inventory from the current working tree: 499 `def`/`abbrev` declarations, 251 theorem/lemma declarations, 76 foundational `structure`/`inductive` declarations, and 55 anonymous examples across the Lean files. 20 theorem/lemma declarations contain non-comment `sorry` in their bodies and are marked **admitted/in progress** below.

## Bottom-Up Hierarchy

1. **Core foundations.** `CLean/Core/Types.lean` defines scalar types, values, addresses, special registers, grid dimensions, parameter/shared metadata, and syntactic read sets. `CLean/Core/IR.lean` builds the project IR on that vocabulary: expressions, instructions, guarded instructions, terminators, and blocks. `CLean/Core/State.lean` then defines byte memories, lane/warp/CTA/kernel/machine state, plus lookup/update and well-formedness APIs.

2. **Typing and executable semantics.** `CLean/Core/Typing.lean` gives byte widths, scalar compatibility, expression typing, alignment, and memory-access preconditions. `CLean/Semantics/Helpers.lean` is the executable semantics: special-register evaluation, expression evaluation, byte/scalar codecs, memory reads/writes, guard/participant selection, barriers, instruction stepping, and terminator stepping.

3. **Relational semantics and executable driver.** `CLean/Semantics/SmallStep.lean` wraps the executable helpers in proof-facing relations (`StepInstr`, `StepBlock`, `StepWarp`, `StepMachine`). `CLean/Proof/Lemmas.lean` proves or states the frame facts and executable-to-relational constructors needed by `CLean/Semantics/Execution.lean`, which defines `Reaches`, `step?`, `runN`, `runN?`, and `traceN`.

4. **Proof middle layer.** `Proof/Loops.lean` defines correctness predicates, abstract reachability, counted-loop specs, and vector/matrix postconditions. `Proof/Determinism.lean` proves that single-warp relational traces correspond to deterministic `runN` traces. `Proof/InstrCompute.lean`, `Proof/InstrValueCompute.lean`, and `Proof/IsSingleWarpPres.lean` supply the structural, value-tracking, and single-warp preservation lemmas needed to make concrete kernel proofs symbolic rather than just Boolean smoke tests. `Proof/LaneDecomposition.lean` states the per-lane independence/lifting layer.

5. **PTX front end.** `PTX/Ast.lean` models parsed PTX syntax. `PTX/Parser.lean` parses realistic PTX text into that AST. `PTX/Lowering.lean` checks names/types/labels and lowers PTX AST to executable IR. `PTX/Bridge.lean` combines parsing and checked lowering, while `Proof/Bridge.lean` states the proof-facing soundness surfaces for successful front-end ingestion.

6. **Examples and top-level case studies.** `Examples/Common.lean` provides reusable lane/memory checkers and memory initializers. `Examples/Semantic.lean`, `Examples/Lowering.lean`, and `Examples/Parser.lean` are regression layers. `Examples/Matmul.lean` and `Examples/Saxpy.lean` are the big examples: each has concrete executable correctness checks and general correctness theorems, with SAXPY currently assembled around the single-warp determinism bridge and explicit in-progress symbolic-execution obligations.

## SAXPY Proof Spine

The SAXPY path starts with `saxpyKernelText`, parsed into `saxpyKernel`, lowered into the `KernelEnv` inside `saxpyStateFor`, and seeded with global vectors and parameter bytes by `saxpyGlobalBytesFor` and `saxpyParamBytesFor`. The state has exactly one warp by `saxpyStateFor_isSingleWarp`. `step?_preserves_IsSingleWarp` keeps that invariant through executable stepping, while `terminal_eq_runN` turns any relational terminal state into an executable `runN` endpoint.

The concrete executable target is `saxpy_runN_satisfies_post`: it produces a fuel `K` such that `runN K (saxpyStateFor n alpha xs ys)` is final and satisfies `saxpyPost`. The `n = 0` branch is structurally proved. The positive branch factors through `saxpy_per_lane_writes` and `saxpy_termination_value_independent`, both admitted/in progress; those are the focused remaining obligations for universal SAXPY correctness. `saxpy_partial_correct` and `saxpy_total_correct` then use the determinism bridge and final-state uniqueness to lift that executable result to relational correctness.

## File-Level Declaration Catalog

### `lakefile.lean`

Imports: `Lake`.

This file is an import/configuration node in the project configuration/import layer and introduces no named Lean `def` or `theorem` declarations.

### `CLean.lean`

Imports: `CLean.Core.Types`, `CLean.Core.IR`, `CLean.Core.State`, `CLean.Core.Typing`, `CLean.Semantics.Helpers`, `CLean.Semantics.SmallStep`, `CLean.Proof.Lemmas`, `CLean.Semantics.Execution`, `CLean.PTX.Ast`, `CLean.PTX.Lowering`, `CLean.PTX.Parser`, `CLean.PTX.Bridge`, `CLean.Proof.Bridge`, `CLean.Proof.Loops`, `CLean.Proof.Determinism`, `CLean.Proof.InstrCompute`, `CLean.Proof.IsSingleWarpPres`, `CLean.Proof.LaneDecomposition`, `CLean.Proof.Automation`, `CLean.Examples`.

This file is an import/configuration node in the project configuration/import layer and introduces no named Lean `def` or `theorem` declarations.

### `CLean/Core/IR.lean`

Imports: `CLean.Core.Types`.

Layer role: intermediate representation grammar.

- `ScalarUnaryOp` ( inductive, line 5): the unary scalar operation grammar: move, negation, absolute value, bit-not, and conversion. Typing and evaluation both dispatch on these constructors. Signature: `inductive ScalarUnaryOp where`.

- `ScalarBinaryOp` ( inductive, line 13): the binary scalar operation grammar: arithmetic, wide multiply, min/max, bitwise operations, and shifts. It is the main arithmetic opcode layer for lowered PTX. Signature: `inductive ScalarBinaryOp where`.

- `ScalarTernaryOp` ( inductive, line 21): the ternary scalar operation grammar: multiply-add, fused multiply-add, and select-predicate. These model PTX-style three-input scalar instructions. Signature: `inductive ScalarTernaryOp where`.

- `CmpOp` ( inductive, line 27): the comparison operation grammar for equality, inequality, and ordering. Predicate assignment and branch conditions depend on it. Signature: `inductive CmpOp where`.

- `RValue` ( inductive, line 31): the recursive right-hand-side expression language. It can be an immediate, register, predicate, special register, or unary/binary/ternary expression tree. Signature: `inductive RValue where`.

- `CmpExpr` ( structure, line 41): a typed comparison expression with an operator and two `RValue` operands. `assignPred` and comparison typing/evaluation consume it. Signature: `structure CmpExpr where`.

- `Guard` ( structure, line 47): a predicate guard with optional negation. Guarded instructions use it to compute participating lanes. Signature: `structure Guard where`.

- `TypedAddr` ( structure, line 52): an address expression paired with expected address space and scalar type. Loads and stores use it as the bridge from expression evaluation to memory access. Signature: `structure TypedAddr where`.

- `ShflMode` ( inductive, line 58): the mode grammar for future warp shuffle operations. It is present in the IR even though warp-op execution is still unsupported. Signature: `inductive ShflMode where`.

- `WarpOp` ( inductive, line 62): the modeled warp-level instruction family: active-mask, shuffle-sync, and ballot-sync. The IR can represent these even where execution currently returns `none`. Signature: `inductive WarpOp where`.

- `AtomicOp` ( inductive, line 68): the modeled atomic operation family: add, exchange, and compare-and-swap. Atomic execution is represented in state but not fully implemented. Signature: `inductive AtomicOp where`.

- `MMAInstr` ( structure, line 72): the structured payload for an MMA instruction: destination and source fragment registers plus fragment type. This keeps MMA shape explicit in the IR. Signature: `structure MMAInstr where`.

- `Instr` ( inductive, line 80): the executable instruction grammar. It covers scalar assignments, predicate assignments, memory operations, address conversion, barriers, warp ops, atomics, and MMA. Signature: `inductive Instr where`.

- `GInstr` ( structure, line 94): a guarded instruction, combining an optional `Guard` with an `Instr`. Lockstep execution first computes its participants, then runs the instruction. Signature: `structure GInstr where`.

- `Terminator` ( inductive, line 99): the block terminator grammar: unconditional branch, uniform conditional branch, and termination. Terminator execution updates PCs or lane status. Signature: `inductive Terminator where`.

- `Block` ( structure, line 105): a basic block: label, array of guarded instructions, and terminator. Kernel environments are maps from labels to these blocks. Signature: `structure Block where`.

### `CLean/Core/State.lean`

Imports: `Std`, `CLean.Core.Types`, `CLean.Core.IR`.

Layer role: machine-state representation.

- `Byte` ( abbrev, line 7): the byte type, `UInt8`, used by all modeled memories. Signature: `abbrev Byte := UInt8`.

- `ByteMem` ( abbrev, line 8): the byte-addressed memory map, `Std.HashMap Nat Byte`. Global, shared, local, parameter, and constant memories all wrap this shape. Signature: `abbrev ByteMem := Std.HashMap Nat Byte`.

- `GlobalMem` ( structure, line 10): the top-level global byte memory wrapper. Global loads/stores read and update this field of `State`. Signature: `structure GlobalMem where`.

- `SharedMem` ( structure, line 14): the per-CTA shared byte memory wrapper. Shared accesses are scoped through the CTA state. Signature: `structure SharedMem where`.

- `LocalMem` ( structure, line 18): the per-lane local byte memory wrapper. Local accesses are scoped through CTA, warp, and lane. Signature: `structure LocalMem where`.

- `ParamMem` ( structure, line 22): the kernel parameter byte memory wrapper. PTX parameter loads read this top-level field. Signature: `structure ParamMem where`.

- `ConstMem` ( structure, line 26): the constant byte memory wrapper. Constant-memory loads read this top-level field. Signature: `structure ConstMem where`.

- `LaneStatus` ( inductive, line 30): the lane execution status: running, blocked at a barrier, or terminated. Runnable-lane discovery and finality depend on it. Signature: `inductive LaneStatus where`.

- `LaneState` ( structure, line 36): the full per-lane machine state: register file, predicate file, local memory, PC, and status. It is the unit updated by most non-memory instruction effects. Signature: `structure LaneState where`.

- `WarpState` ( structure, line 44): the per-warp state: exactly 32 lane slots plus active and exited masks. Lockstep semantics and lane-participant logic operate here. Signature: `structure WarpState where`.

- `BarrierInstance` ( structure, line 50): the state of one CTA barrier, with epoch, arrivals, and expected count. `stepBarrierCTA?` mutates this during barrier synchronization. Signature: `structure BarrierInstance where`.

- `BarrierState` ( structure, line 56): the CTA-local collection of barrier instances indexed by barrier id. Signature: `structure BarrierState where`.

- `AtomicEvent` ( structure, line 60): a log record for a future atomic operation: location, operands, before/after values, and the executing lane identity. Signature: `structure AtomicEvent where`.

- `AtomicState` ( structure, line 72): the top-level atomic event log. It gives atomics a state slot even though atomic instruction execution is not implemented yet. Signature: `structure AtomicState where`.

- `CTAState` ( structure, line 76): the state of one CTA: shared memory, warp map, and barrier state. Signature: `structure CTAState where`.

- `KernelEnv` ( structure, line 82): the static kernel environment: entry label, launch context, parameter/shared declarations, and executable block map. Signature: `structure KernelEnv where`.

- `State` ( structure, line 90): the complete machine state: kernel environment, top-level memories, CTA map, and atomic log. Signature: `structure State where`.

- `getLane?` ( def, line 101): looks up a lane either inside a `WarpState` or through the full `(CTA, warp, lane)` path in `State`. It is the read side of the lane-update API. Signature: `def getLane? (warp : WarpState) (lane : LaneId) : Option LaneState :=`.

- `setLane` ( def, line 104): updates a lane either inside a `WarpState` or through the full state path. It is the primitive most instruction-step functions eventually use. Signature: `def setLane (warp : WarpState) (lane : LaneId) (laneState : LaneState) : WarpState :=`.

- `wf?` ( def, line 107): computes a Boolean well-formedness check for the current namespace target. For warps it checks lane-array length; for CTAs/states/environments it recursively checks contained maps and block labels. Signature: `def wf? (warp : WarpState) : Bool :=`.

- `wf` ( def, line 110): turns the Boolean well-formedness check into a `Prop`. This lets relational semantics require well-formedness while still relying on executable checks. Signature: `def wf (warp : WarpState) : Prop :=`.

- `wf_iff_bool` ( theorem, line 113): states that the propositional well-formedness wrapper is definitionally equivalent to the Boolean check. The proof is `Iff.rfl`, making it a bridge for simplification. Signature: `theorem wf_iff_bool (warp : WarpState) : WarpState.wf warp ↔ warp.wf? = true := Iff.rfl`.

- `instance@115` ( instance, line 115): provides a typeclass instance needed by the machine-state representation. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (warp : WarpState) : Decidable (WarpState.wf warp) := by`.

- `wf?` ( def, line 123): computes a Boolean well-formedness check for the current namespace target. For warps it checks lane-array length; for CTAs/states/environments it recursively checks contained maps and block labels. Signature: `def wf? (cta : CTAState) : Bool :=`.

- `wf` ( def, line 126): turns the Boolean well-formedness check into a `Prop`. This lets relational semantics require well-formedness while still relying on executable checks. Signature: `def wf (cta : CTAState) : Prop :=`.

- `wf_iff_bool` ( theorem, line 129): states that the propositional well-formedness wrapper is definitionally equivalent to the Boolean check. The proof is `Iff.rfl`, making it a bridge for simplification. Signature: `theorem wf_iff_bool (cta : CTAState) : CTAState.wf cta ↔ cta.wf? = true := Iff.rfl`.

- `instance@131` ( instance, line 131): provides a typeclass instance needed by the machine-state representation. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (cta : CTAState) : Decidable (CTAState.wf cta) := by`.

- `wf?` ( def, line 139): computes a Boolean well-formedness check for the current namespace target. For warps it checks lane-array length; for CTAs/states/environments it recursively checks contained maps and block labels. Signature: `def wf? (env : KernelEnv) : Bool :=`.

- `wf` ( def, line 142): turns the Boolean well-formedness check into a `Prop`. This lets relational semantics require well-formedness while still relying on executable checks. Signature: `def wf (env : KernelEnv) : Prop :=`.

- `wf_iff_bool` ( theorem, line 145): states that the propositional well-formedness wrapper is definitionally equivalent to the Boolean check. The proof is `Iff.rfl`, making it a bridge for simplification. Signature: `theorem wf_iff_bool (env : KernelEnv) : KernelEnv.wf env ↔ env.wf? = true := Iff.rfl`.

- `instance@147` ( instance, line 147): provides a typeclass instance needed by the machine-state representation. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (env : KernelEnv) : Decidable (KernelEnv.wf env) := by`.

- `getCTA?` ( def, line 155): looks up a CTA in the state CTA map. All nested CTA/warp/lane lookups start here. Signature: `def getCTA? (st : State) (cta : CTAId) : Option CTAState :=`.

- `setCTA` ( def, line 158): updates a CTA entry in the state CTA map. It is the primitive below `setWarp`, shared-memory updates, and barrier updates. Signature: `def setCTA (st : State) (cta : CTAId) (ctaState : CTAState) : State :=`.

- `getWarp?` ( def, line 161): looks up a warp by first finding its CTA and then reading the CTA warp map. Most step relations require this lookup to succeed. Signature: `def getWarp? (st : State) (cta : CTAId) (warp : WarpId) : Option WarpState := do`.

- `setWarp` ( def, line 165): updates a warp in an existing CTA and returns `none` if the CTA does not exist. It is the full-state primitive below lane updates. Signature: `def setWarp (st : State) (cta : CTAId) (warp : WarpId) (warpState : WarpState) : Option State := do`.

- `getLane?` ( def, line 170): looks up a lane either inside a `WarpState` or through the full `(CTA, warp, lane)` path in `State`. It is the read side of the lane-update API. Signature: `def getLane? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : Option LaneState := do`.

- `setLane` ( def, line 174): updates a lane either inside a `WarpState` or through the full state path. It is the primitive most instruction-step functions eventually use. Signature: `def setLane (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (laneState : LaneState) : Option State := do`.

- `wf?` ( def, line 179): computes a Boolean well-formedness check for the current namespace target. For warps it checks lane-array length; for CTAs/states/environments it recursively checks contained maps and block labels. Signature: `def wf? (st : State) : Bool :=`.

- `wf` ( def, line 182): turns the Boolean well-formedness check into a `Prop`. This lets relational semantics require well-formedness while still relying on executable checks. Signature: `def wf (st : State) : Prop :=`.

- `wf_iff_bool` ( theorem, line 185): states that the propositional well-formedness wrapper is definitionally equivalent to the Boolean check. The proof is `Iff.rfl`, making it a bridge for simplification. Signature: `theorem wf_iff_bool (st : State) : State.wf st ↔ st.wf? = true := Iff.rfl`.

- `instance@187` ( instance, line 187): provides a typeclass instance needed by the machine-state representation. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (st : State) : Decidable (State.wf st) := by`.

### `CLean/Core/Types.lean`

Imports: `Std`.

Layer role: foundational type vocabulary.

- `ScalarTy` ( inductive, line 5): the scalar type universe for predicates, fixed-width unsigned/signed/bit integers, and floating-point scalar values. Everything from encoding bytes to PTX lowering uses this as the common type vocabulary. Signature: `inductive ScalarTy where`.

- `AddrSpace` ( inductive, line 13): the memory-space tags, including global, shared, local, parameter, constant, and generic addresses. It is the discriminator used by address resolution, memory access preconditions, and PTX `cvta`/`isspacep` semantics. Signature: `inductive AddrSpace where`.

- `CTAId` ( abbrev, line 17): the block/CTA identifier type, currently just `Nat`. It keeps state and semantic APIs explicit about which CTA is being inspected or stepped. Signature: `abbrev CTAId := Nat`.

- `WarpId` ( abbrev, line 18): the warp identifier type, currently `Nat`. It is paired with `CTAId` to locate a warp in the machine state. Signature: `abbrev WarpId := Nat`.

- `LaneId` ( abbrev, line 19): the lane identifier type, `Fin 32`, which bakes the 32-lane warp width into the type. This is the foundation for lane arrays, masks, and per-lane proofs. Signature: `abbrev LaneId := Fin 32`.

- `RegName` ( abbrev, line 20): the register-name type, represented as `String`. Register maps in lane states and type environments are keyed by this alias. Signature: `abbrev RegName := String`.

- `PredName` ( abbrev, line 21): the predicate-register-name type, represented as `String`. Predicate files, guards, and conditional branches use it. Signature: `abbrev PredName := String`.

- `BlockLabel` ( abbrev, line 22): the control-flow label type. Kernel blocks and PCs use labels to address basic blocks. Signature: `abbrev BlockLabel := String`.

- `PC` ( abbrev, line 23): the program-counter shape, a block label paired with an instruction index. This is the smallest control-flow unit tracked per lane. Signature: `abbrev PC := BlockLabel × Nat`.

- `FragTy` ( inductive, line 25): the currently modeled fragment/MMA tile types. It is present so `Value.frag` and `Instr.mma` have a typed payload even though MMA semantics are not yet implemented. Signature: `inductive FragTy where`.

- `Value` ( inductive, line 31): the runtime value universe. It mirrors scalar types, adds generic addresses, and includes fragments, making it the value domain used by registers, memory decoding, expression evaluation, and instruction execution. Signature: `inductive Value where`.

- `Addr` ( inductive, line 53): the resolved address universe. It records both address space and the extra scoping needed for shared and local memory, so memory helpers can choose the correct byte map. Signature: `inductive Addr where`.

- `SpecialReg` ( inductive, line 62): the modeled PTX special registers for thread/block coordinates and launch dimensions. `evalSpecial` maps these to runtime `u32` values. Signature: `inductive SpecialReg where`.

- `Dim3` ( structure, line 69): a CUDA-style three-dimensional extent with default `y` and `z` equal to one. Grid and block dimensions share this representation. Signature: `structure Dim3 where`.

- `GridCtx` ( structure, line 75): the launch context containing grid and block dimensions. Special-register evaluation and thread indexing consume it. Signature: `structure GridCtx where`.

- `ParamInfo` ( structure, line 80): metadata for a lowered kernel parameter: name, scalar type, pointer annotation, alignment, offset, and byte size. Lowering and parameter-memory setup share this structure. Signature: `structure ParamInfo where`.

- `SharedDecl` ( structure, line 90): metadata for a lowered shared-memory allocation. It records a name, size, alignment, and offset inside CTA shared memory. Signature: `structure SharedDecl where`.

- `ReadSet` ( structure, line 97): a lightweight syntactic dependency summary for registers, predicates, and special registers read by expressions/instructions. Signature: `structure ReadSet where`.

- `union` ( def, line 105): combines two `ReadSet`s by concatenating their register, predicate, and special-register lists. It is intentionally simple and preserves duplicate reads rather than normalizing them. Signature: `def union (a b : ReadSet) : ReadSet :=`.

- `space` ( def, line 112): projects the address space from a resolved `Addr`. Generic addresses return their stored concrete space, so memory preconditions can compare spaces uniformly. Signature: `def space : Addr → AddrSpace | .global _ => .global | .shared _ _ => .shared | .local _ _ _ _ => .local | .param _ => .param | .const _ => .const | .generic s _ => s`.

- `offset` ( def, line 120): projects the byte offset from a resolved `Addr`, ignoring the scoping fields used for shared and local memory. Signature: `def offset : Addr → Nat | .global o => o | .shared _ o => o | .local _ _ _ o => o | .param o => o | .const o => o | .generic _ o => o`.

### `CLean/Core/Typing.lean`

Imports: `CLean.Core.Types`, `CLean.Core.IR`.

Layer role: typing and access-precondition layer.

- `byteWidth?` ( def, line 8): maps every supported scalar type to its encoded byte width. Memory access, parameter layout, and shared-memory layout depend on it. Signature: `def byteWidth? : ScalarTy → Option Nat | .pred => some 1 | .u8 | .s8 | .b8 => some 1 | .u16 | .s16 | .b16 | .f16 | .bf16 => some 2 | .u32 | .s32 | .b32 | .f32 => some 4 | .u64 | .s64 | .b64 | .f64 => some 8`.

- `alignment?` ( def, line 15): uses scalar byte width as the default alignment. It is the current alignment policy used by access preconditions and layout code. Signature: `def alignment? (ty : ScalarTy) : Option Nat :=`.

- `scalarCodecSupported?` ( def, line 18): states which scalar types have byte encoders/decoders. Currently all scalar types in `ScalarTy` are supported by the codec layer. Signature: `def scalarCodecSupported? : ScalarTy → Bool | .pred | .u8 | .u16 | .u32 | .u64 | .s8 | .s16 | .s32 | .s64 | .b8 | .b16 | .b32 | .b64 | .f16 | .bf16 | .f32 | .f64 => true`.

- `cvtaSourceSupported?` ( def, line 25): recognizes scalar integer/bit types accepted as source operands for address conversion. It keeps `cvta` lowering and semantics from accepting arbitrary values. Signature: `def cvtaSourceSupported? : ScalarTy → Bool | .u32 | .u64 | .b32 | .b64 => true | _ => false`.

- `isGenericAddrValue?` ( def, line 29): recognizes runtime generic address values. It is a small predicate for address-typed runtime checks. Signature: `def isGenericAddrValue? : Value → Bool | .gaddr _ _ => true | _ => false`.

- `scalarCompatible?` ( def, line 33): implements the project’s permissive scalar compatibility relation, especially signed/unsigned/bit reinterpretation at 32 and 64 bits. Checked PTX lowering uses it before inserting conversions. Signature: `def scalarCompatible? (expected actual : ScalarTy) : Bool :=`.

- `valueHasType` ( def, line 44): the propositional value/type relation for plain scalar values. Generic addresses and fragments intentionally do not satisfy scalar types here. Signature: `def valueHasType : Value → ScalarTy → Prop | .pred _, .pred => True | .u8 _, .u8 => True | .u16 _, .u16 => True | .u32 _, .u32 => True | .u64 _, .u64 => True | .s8 _, .s8 => True | .s16 _, .s16 => True | .s32 _, .s32 => True | .s64 _, .s64 => True | .b8 _, .b8 => True`.

- `TypeEnv` ( structure, line 66): the checked-lowering/typechecking environment, tracking register types, predicate types, parameter metadata, and shared-memory metadata. Signature: `structure TypeEnv where`.

- `binarySameWidthInt?` ( private def, line 73): private helper recognizing same-width integer arithmetic pairs. It supports binary and comparison signature checks. Signature: `private def binarySameWidthInt? (lhs rhs : ScalarTy) : Bool :=`.

- `binaryFloat?` ( private def, line 78): private helper recognizing same-width floating-point pairs. It supports arithmetic and comparison signature checks. Signature: `private def binaryFloat? (lhs rhs : ScalarTy) : Bool :=`.

- `cvtSig?` ( private def, line 83): private helper giving the allowed scalar conversion signatures. `unarySig?` delegates conversion cases to it. Signature: `private def cvtSig? (dst src : ScalarTy) : Option ScalarTy :=`.

- `unarySig?` ( def, line 110): computes the result type of a unary operation on an input scalar type. It rejects unsupported operations by returning `none`. Signature: `def unarySig? : ScalarUnaryOp → ScalarTy → Option ScalarTy | .mov, ty => some ty | .neg, .s32 => some .s32 | .neg, .s64 => some .s64 | .neg, .f32 => some .f32 | .neg, .f64 => some .f64 | .abs, .s32 => some .s32 | .abs, .s64 => some .s64 | .abs, .f32 => some .f32`.

- `binarySig?` ( def, line 127): computes the result type of a binary operation on two scalar types. This is the checked-lowering gate for binary PTX opcodes. Signature: `def binarySig? : ScalarBinaryOp → ScalarTy → ScalarTy → Option ScalarTy | .mulWideS32, .s32, .s32 => some .s64 | .bitor, .pred, .pred => some .pred | .bitand, .pred, .pred => some .pred | .bitxor, .pred, .pred => some .pred | .add, a, b | .sub, a, b | .mul, a, b =>`.

- `ternarySig?` ( def, line 151): computes result types for ternary operations such as `mad`, `fma`, and `selp`. It enforces same-type arithmetic and predicate condition for `selp`. Signature: `def ternarySig? : ScalarTernaryOp → ScalarTy → ScalarTy → ScalarTy → Option ScalarTy | .mad, .u32, .u32, .u32 => some .u32 | .mad, .u64, .u64, .u64 => some .u64 | .mad, .s32, .s32, .s32 => some .s32 | .mad, .s64, .s64, .s64 => some .s64 | .fma, .f32, .f32, .f32 => some .f32`.

- `cmpSig?` ( def, line 161): computes whether a comparison between two types produces a predicate. It currently accepts same-width ints and floats. Signature: `def cmpSig? : CmpOp → ScalarTy → ScalarTy → Option ScalarTy | _, a, b => if a = b && (binarySameWidthInt? a b || binaryFloat? a b) then some .pred else none`.

- `rvalueTypeOf?` ( partial def, line 166): recursively infers the scalar type of an `RValue` using the current `TypeEnv`. It is mutual with value and special-register type inference. Signature: `partial def rvalueTypeOf? (env : TypeEnv) : RValue → Option ScalarTy | .imm v => valueType? v | .reg r => env.regs[r]? | .pred p => env.preds[p]? | .special s => specialType? s | .unop op a => do let ta <- rvalueTypeOf? env a unarySig? op ta | .binop op a b => do`.

- `valueType?` ( partial def, line 184): maps runtime scalar `Value` constructors back to `ScalarTy`, returning `none` for generic addresses and fragments. Lowering uses it for immediate operands. Signature: `partial def valueType? : Value → Option ScalarTy | .pred _ => some .pred | .u8 _ => some .u8 | .u16 _ => some .u16 | .u32 _ => some .u32 | .u64 _ => some .u64 | .s8 _ => some .s8 | .s16 _ => some .s16 | .s32 _ => some .s32 | .s64 _ => some .s64 | .b8 _ => some .b8`.

- `specialType?` ( partial def, line 205): assigns scalar types to special registers; currently every modeled special register is `u32`. Signature: `partial def specialType? : SpecialReg → Option ScalarTy | _ => some .u32 end`.

- `cmpExprWellTyped` ( def, line 209): the propositional well-typedness condition for a comparison expression in a type environment. It asks the operands to typecheck and `cmpSig?` to return `.pred`. Signature: `def cmpExprWellTyped (env : TypeEnv) (cmp : CmpExpr) : Prop :=`.

- `aligned` ( def, line 214): the propositional alignment predicate for typed memory access. It requires a positive alignment and offset divisibility. Signature: `def aligned (ty : ScalarTy) (addr : Addr) : Prop :=`.

- `aligned?` ( def, line 219): the executable Boolean version of `aligned` used by memory access functions. Signature: `def aligned? (ty : ScalarTy) (addr : Addr) : Bool :=`.

- `addrSpaceMatches` ( def, line 224): the propositional check that a resolved address belongs to the requested address space. Generic addresses compare their stored concrete space. Signature: `def addrSpaceMatches (space : AddrSpace) (addr : Addr) : Prop :=`.

- `addrSpaceMatches?` ( def, line 229): the executable Boolean version of `addrSpaceMatches`. Signature: `def addrSpaceMatches? (space : AddrSpace) (addr : Addr) : Bool :=`.

- `typedAccessPreconditions` ( def, line 234): the propositional memory-access precondition: codec support, known byte width, alignment, and address-space match. Signature: `def typedAccessPreconditions (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Prop :=`.

- `typedAccessPreconditions?` ( def, line 237): the executable Boolean memory-access precondition used by `readMem?` and `writeMem?`. Signature: `def typedAccessPreconditions? (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Bool :=`.

### `CLean/Semantics/Execution.lean`

Imports: `CLean.Proof.Lemmas`.

Layer role: executable driver and reachability layer.

- `Reaches` ( inductive, line 5): defines an algebraic datatype used in the executable driver and reachability layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive Reaches : State → State → Prop where`.

- `trans` ( theorem, line 14): proves a property in the executable driver and reachability layer. The name and signature identify the exact fact: `theorem trans {st₀ st₁ st₂ : State} : Reaches st₀ st₁ → Reaches st₁ st₂ → Reaches st₀ st₂ := by`. Signature: `theorem trans {st₀ st₁ st₂ : State} : Reaches st₀ st₁ → Reaches st₁ st₂ → Reaches st₀ st₂ := by`.

- `stepAt?` ( def, line 25): defines `stepAt?` in the executable driver and reachability layer. The signature is `def stepAt? (st : State) (cta : CTAId) (warp : WarpId) : Option State :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def stepAt? (st : State) (cta : CTAId) (warp : WarpId) : Option State :=`.

- `step?` ( def, line 30): defines `step?` in the executable driver and reachability layer. The signature is `def step? (st : State) : Option State :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def step? (st : State) : Option State :=`.

- `runN` ( def, line 33): defines `runN` in the executable driver and reachability layer. The signature is `def runN : Nat → State → State | 0, st => st | fuel + 1, st => match step? st with | some st' => runN fuel st' | none => st`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def runN : Nat → State → State | 0, st => st | fuel + 1, st => match step? st with | some st' => runN fuel st' | none => st`.

- `runN?` ( def, line 40): defines `runN?` in the executable driver and reachability layer. The signature is `def runN? : Nat → State → Option State | 0, st => some st | fuel + 1, st => do let st' <- step? st runN? fuel st'`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def runN? : Nat → State → Option State | 0, st => some st | fuel + 1, st => do let st' <- step? st runN? fuel st'`.

- `traceN` ( def, line 46): defines `traceN` in the executable driver and reachability layer. The signature is `def traceN : Nat → State → List State | 0, st => [st] | fuel + 1, st => match step? st with | some st' => st :: traceN fuel st' | none => [st]`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def traceN : Nat → State → List State | 0, st => [st] | fuel + 1, st => match step? st with | some st' => st :: traceN fuel st' | none => [st]`.

- `stepAt?_sound` ( theorem, line 53): proves that an executable computation corresponds to the relational semantics. This is a bridge from `Option`-returning functions to `StepMachine`/`Reaches` proofs. Signature: `theorem stepAt?_sound {st st' : State} {cta : CTAId} {warp : WarpId} (hstep : stepAt? st cta warp = some st') : StepMachine st st' := by`.

- `step?_sound` ( theorem, line 77): proves that an executable computation corresponds to the relational semantics. This is a bridge from `Option`-returning functions to `StepMachine`/`Reaches` proofs. Signature: `theorem step?_sound {st st' : State} (hstep : step? st = some st') : StepMachine st st' :=`.

- `runN_reaches` ( theorem, line 82): proves a property in the executable driver and reachability layer. The name and signature identify the exact fact: `theorem runN_reaches (fuel : Nat) (st : State) : Reaches st (runN fuel st) := by`. Signature: `theorem runN_reaches (fuel : Nat) (st : State) : Reaches st (runN fuel st) := by`.

- `runN?_sound` ( theorem, line 95): proves that an executable computation corresponds to the relational semantics. This is a bridge from `Option`-returning functions to `StepMachine`/`Reaches` proofs. Signature: `theorem runN?_sound {fuel : Nat} {st st' : State} (hrun : runN? fuel st = some st') : Reaches st st' := by`.

### `CLean/Semantics/Helpers.lean`

Imports: `CLean.Core.State`, `CLean.Core.Typing`.

Layer role: executable semantic helper layer.

- `intAbs` ( private def, line 10): defines `intAbs` in the executable semantic helper layer. The signature is `private def intAbs (x : Int) : Int :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def intAbs (x : Int) : Int :=`.

- `normalizeSigned` ( private def, line 13): defines `normalizeSigned` in the executable semantic helper layer. The signature is `private def normalizeSigned (bits : Nat) (x : Int) : Int :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def normalizeSigned (bits : Nat) (x : Int) : Int :=`.

- `intToFloat` ( private def, line 22): defines `intToFloat` in the executable semantic helper layer. The signature is `private def intToFloat (x : Int) : Float :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def intToFloat (x : Int) : Float :=`.

- `u32Xor` ( private def, line 28): defines `u32Xor` in the executable semantic helper layer. The signature is `private def u32Xor (a b : UInt32) : UInt32 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u32Xor (a b : UInt32) : UInt32 :=`.

- `u64Xor` ( private def, line 31): defines `u64Xor` in the executable semantic helper layer. The signature is `private def u64Xor (a b : UInt64) : UInt64 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u64Xor (a b : UInt64) : UInt64 :=`.

- `u32Shl` ( private def, line 34): defines `u32Shl` in the executable semantic helper layer. The signature is `private def u32Shl (a : UInt32) (n : Nat) : UInt32 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u32Shl (a : UInt32) (n : Nat) : UInt32 :=`.

- `u64Shl` ( private def, line 37): defines `u64Shl` in the executable semantic helper layer. The signature is `private def u64Shl (a : UInt64) (n : Nat) : UInt64 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u64Shl (a : UInt64) (n : Nat) : UInt64 :=`.

- `u32Shr` ( private def, line 40): defines `u32Shr` in the executable semantic helper layer. The signature is `private def u32Shr (a : UInt32) (n : Nat) : UInt32 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u32Shr (a : UInt32) (n : Nat) : UInt32 :=`.

- `u64Shr` ( private def, line 43): defines `u64Shr` in the executable semantic helper layer. The signature is `private def u64Shr (a : UInt64) (n : Nat) : UInt64 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def u64Shr (a : UInt64) (n : Nat) : UInt64 :=`.

- `bitSet` ( def, line 46): defines `bitSet` in the executable semantic helper layer. The signature is `def bitSet (mask : UInt32) (i : Nat) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def bitSet (mask : UInt32) (i : Nat) : Bool :=`.

- `nonzeroDim` ( def, line 49): defines `nonzeroDim` in the executable semantic helper layer. The signature is `def nonzeroDim (n : Nat) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def nonzeroDim (n : Nat) : Nat :=`.

- `dim3X` ( def, line 52): defines `dim3X` in the executable semantic helper layer. The signature is `def dim3X (linear : Nat) (dims : Dim3) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def dim3X (linear : Nat) (dims : Dim3) : Nat :=`.

- `dim3Y` ( def, line 55): defines `dim3Y` in the executable semantic helper layer. The signature is `def dim3Y (linear : Nat) (dims : Dim3) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def dim3Y (linear : Nat) (dims : Dim3) : Nat :=`.

- `dim3Z` ( def, line 58): defines `dim3Z` in the executable semantic helper layer. The signature is `def dim3Z (linear : Nat) (dims : Dim3) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def dim3Z (linear : Nat) (dims : Dim3) : Nat :=`.

- `blockLinearTid` ( def, line 63): defines `blockLinearTid` in the executable semantic helper layer. The signature is `def blockLinearTid (warp : WarpId) (lane : LaneId) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def blockLinearTid (warp : WarpId) (lane : LaneId) : Nat :=`.

- `blockLinearTidX` ( def, line 66): defines `blockLinearTidX` in the executable semantic helper layer. The signature is `def blockLinearTidX (warp : WarpId) (lane : LaneId) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def blockLinearTidX (warp : WarpId) (lane : LaneId) : Nat :=`.

- `threadIdxX` ( def, line 69): defines `threadIdxX` in the executable semantic helper layer. The signature is `def threadIdxX (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def threadIdxX (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`.

- `threadIdxY` ( def, line 72): defines `threadIdxY` in the executable semantic helper layer. The signature is `def threadIdxY (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def threadIdxY (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`.

- `threadIdxZ` ( def, line 75): defines `threadIdxZ` in the executable semantic helper layer. The signature is `def threadIdxZ (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def threadIdxZ (grid : GridCtx) (warp : WarpId) (lane : LaneId) : Nat :=`.

- `ctaIdxX` ( def, line 78): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `def ctaIdxX (grid : GridCtx) (cta : CTAId) : Nat :=`.

- `ctaIdxY` ( def, line 81): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `def ctaIdxY (grid : GridCtx) (cta : CTAId) : Nat :=`.

- `ctaIdxZ` ( def, line 84): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `def ctaIdxZ (grid : GridCtx) (cta : CTAId) : Nat :=`.

- `laneIds` ( def, line 87): defines `laneIds` in the executable semantic helper layer. The signature is `def laneIds : List LaneId :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def laneIds : List LaneId :=`.

- `readReg` ( def, line 90): defines `readReg` in the executable semantic helper layer. The signature is `def readReg (lane : LaneState) (r : RegName) : Option Value :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def readReg (lane : LaneState) (r : RegName) : Option Value :=`.

- `writeReg` ( def, line 93): defines `writeReg` in the executable semantic helper layer. The signature is `def writeReg (lane : LaneState) (r : RegName) (v : Value) : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def writeReg (lane : LaneState) (r : RegName) (v : Value) : LaneState :=`.

- `readPred` ( def, line 96): defines `readPred` in the executable semantic helper layer. The signature is `def readPred (lane : LaneState) (p : PredName) : Option Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def readPred (lane : LaneState) (p : PredName) : Option Bool :=`.

- `writePred` ( def, line 99): defines `writePred` in the executable semantic helper layer. The signature is `def writePred (lane : LaneState) (p : PredName) (b : Bool) : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def writePred (lane : LaneState) (p : PredName) (b : Bool) : LaneState :=`.

- `evalSpecial` ( def, line 102): defines `evalSpecial` in the executable semantic helper layer. The signature is `def evalSpecial (grid : GridCtx) (cta : CTAId) (warp : WarpId) (lane : LaneId) : SpecialReg → Value | .tidX => .u32 <| UInt32.ofNat (threadIdxX grid warp lane) | .tidY => .u32 <| UInt32.ofNat (threadIdxY grid warp lane) | .tidZ => .u32 <| UInt32.ofNat (threadIdxZ grid warp lane)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def evalSpecial (grid : GridCtx) (cta : CTAId) (warp : WarpId) (lane : LaneId) : SpecialReg → Value | .tidX => .u32 <| UInt32.ofNat (threadIdxX grid warp lane) | .tidY => .u32 <| UInt32.ofNat (threadIdxY grid warp lane) | .tidZ => .u32 <| UInt32.ofNat (threadIdxZ grid warp lane)`.

- `rvalueReadSet` ( partial def, line 117): defines `rvalueReadSet` in the executable semantic helper layer. The signature is `partial def rvalueReadSet : RValue → ReadSet | .imm _ => {} | .reg r => { regs := [r] }`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def rvalueReadSet : RValue → ReadSet | .imm _ => {} | .reg r => { regs := [r] }`.

- `cmpReadSet` ( partial def, line 126): defines `cmpReadSet` in the executable semantic helper layer. The signature is `partial def cmpReadSet (cmp : CmpExpr) : ReadSet :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def cmpReadSet (cmp : CmpExpr) : ReadSet :=`.

- `instrReadSet` ( def, line 130): defines `instrReadSet` in the executable semantic helper layer. The signature is `def instrReadSet : Instr → ReadSet | .assignReg _ rhs => rvalueReadSet rhs | .assignPred _ cmp => cmpReadSet cmp | .assignPredValue _ rhs => rvalueReadSet rhs | .load _ src => rvalueReadSet src.addr | .store dst value => ReadSet.union (rvalueReadSet dst.addr) (rvalueReadSet value)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrReadSet : Instr → ReadSet | .assignReg _ rhs => rvalueReadSet rhs | .assignPred _ cmp => cmpReadSet cmp | .assignPredValue _ rhs => rvalueReadSet rhs | .load _ src => rvalueReadSet src.addr | .store dst value => ReadSet.union (rvalueReadSet dst.addr) (rvalueReadSet value)`.

- `terminatorReadSet` ( def, line 150): defines `terminatorReadSet` in the executable semantic helper layer. The signature is `def terminatorReadSet : Terminator → ReadSet | .br _ => {} | .cbr cond _ _ => rvalueReadSet cond | .terminate => {}`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def terminatorReadSet : Terminator → ReadSet | .br _ => {} | .cbr cond _ _ => rvalueReadSet cond | .terminate => {}`.

- `ginstrReadSet` ( def, line 155): defines `ginstrReadSet` in the executable semantic helper layer. The signature is `def ginstrReadSet (gi : GInstr) : ReadSet :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ginstrReadSet (gi : GInstr) : ReadSet :=`.

- `valueToBool?` ( def, line 161): defines `valueToBool?` in the executable semantic helper layer. The signature is `def valueToBool? : Value → Option Bool | .pred b => some b | _ => none`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def valueToBool? : Value → Option Bool | .pred b => some b | _ => none`.

- `natToBytesLE` ( def, line 165): defines `natToBytesLE` in the executable semantic helper layer. The signature is `def natToBytesLE (n width : Nat) : List Byte :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natToBytesLE (n width : Nat) : List Byte :=`.

- `bytesToNatLE` ( def, line 168): defines `bytesToNatLE` in the executable semantic helper layer. The signature is `def bytesToNatLE (bs : List Byte) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def bytesToNatLE (bs : List Byte) : Nat :=`.

- `signedToNat` ( def, line 174): defines `signedToNat` in the executable semantic helper layer. The signature is `def signedToNat (bits : Nat) (x : Int) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def signedToNat (bits : Nat) (x : Int) : Nat :=`.

- `natToSigned` ( def, line 178): defines `natToSigned` in the executable semantic helper layer. The signature is `def natToSigned (bits : Nat) (n : Nat) : Int :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natToSigned (bits : Nat) (n : Nat) : Int :=`.

- `encodeScalar?` ( def, line 183): defines `encodeScalar?` in the executable semantic helper layer. The signature is `def encodeScalar? : ScalarTy → Value → Option (List Byte) | .pred, .pred b => some [if b then 1 else 0] | .u8, .u8 x | .b8, .b8 x => some [x] | .u16, .u16 x | .b16, .b16 x => some <| natToBytesLE x.toNat 2 | .u32, .u32 x | .b32, .b32 x => some <| natToBytesLE x.toNat 4`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def encodeScalar? : ScalarTy → Value → Option (List Byte) | .pred, .pred b => some [if b then 1 else 0] | .u8, .u8 x | .b8, .b8 x => some [x] | .u16, .u16 x | .b16, .b16 x => some <| natToBytesLE x.toNat 2 | .u32, .u32 x | .b32, .b32 x => some <| natToBytesLE x.toNat 4`.

- `decodeScalar?` ( def, line 198): defines `decodeScalar?` in the executable semantic helper layer. The signature is `def decodeScalar? : ScalarTy → List Byte → Option Value | .pred, [b] => some (.pred (b != 0)) | .u8, [b] => some (.u8 b) | .b8, [b] => some (.b8 b) | .u16, bs => if bs.length = 2 then some (.u16 (UInt16.ofNat <| bytesToNatLE bs)) else none | .b16, bs => if bs.length = 2 then some (.b16 (UInt16.ofNat <| bytesToNatLE bs)) else none`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def decodeScalar? : ScalarTy → List Byte → Option Value | .pred, [b] => some (.pred (b != 0)) | .u8, [b] => some (.u8 b) | .b8, [b] => some (.b8 b) | .u16, bs => if bs.length = 2 then some (.u16 (UInt16.ofNat <| bytesToNatLE bs)) else none | .b16, bs => if bs.length = 2 then some (.b16 (UInt16.ofNat <| bytesToNatLE bs)) else none`.

- `readBytes?` ( def, line 218): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `def readBytes? (mem : ByteMem) (offset width : Nat) : Option (List Byte) :=`.

- `writeBytes` ( def, line 228): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `def writeBytes (mem : ByteMem) (offset : Nat) (bytes : List Byte) : ByteMem :=`.

- `guardHolds?` ( def, line 234): defines `guardHolds?` in the executable semantic helper layer. The signature is `def guardHolds? (lane : LaneState) (guard? : Option Guard) : Option Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def guardHolds? (lane : LaneState) (guard? : Option Guard) : Option Bool :=`.

- `laneIsRunnable` ( def, line 241): defines `laneIsRunnable` in the executable semantic helper layer. The signature is `def laneIsRunnable (warp : WarpState) (lane : LaneId) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def laneIsRunnable (warp : WarpState) (lane : LaneId) : Bool :=`.

- `runnableLaneIds` ( def, line 246): defines `runnableLaneIds` in the executable semantic helper layer. The signature is `def runnableLaneIds (warp : WarpState) : List LaneId :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def runnableLaneIds (warp : WarpState) : List LaneId :=`.

- `currentRunnablePc?` ( def, line 249): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `def currentRunnablePc? (warp : WarpState) : Option PC := do`.

- `participatingRunnableLaneIds?` ( def, line 254): defines `participatingRunnableLaneIds?` in the executable semantic helper layer. The signature is `def participatingRunnableLaneIds? (warp : WarpState) (guard? : Option Guard) : Option (List LaneId) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def participatingRunnableLaneIds? (warp : WarpState) (guard? : Option Guard) : Option (List LaneId) := do`.

- `RunnablePc` ( def, line 266): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `def RunnablePc (warp : WarpState) (pc : PC) : Prop :=`.

- `ParticipatingRunnable` ( def, line 269): defines `ParticipatingRunnable` in the executable semantic helper layer. The signature is `def ParticipatingRunnable (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ParticipatingRunnable (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Prop :=`.

- `runnablePc_iff_bool` ( theorem, line 272): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `theorem runnablePc_iff_bool (warp : WarpState) (pc : PC) : Helpers.RunnablePc warp pc ↔ Helpers.currentRunnablePc? warp = some pc := Iff.rfl`.

- `instance@275` ( instance, line 275): provides a typeclass instance needed by the executable semantic helper layer. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (warp : WarpState) (pc : PC) : Decidable (Helpers.RunnablePc warp pc) := by`.

- `participatingRunnable_iff_bool` ( theorem, line 279): proves a property in the executable semantic helper layer. The name and signature identify the exact fact: `theorem participatingRunnable_iff_bool (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Helpers.ParticipatingRunnable warp guard? lanes ↔ Helpers.participatingRunnableLaneIds? warp guard? = some lanes := Iff.rfl`. Signature: `theorem participatingRunnable_iff_bool (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Helpers.ParticipatingRunnable warp guard? lanes ↔ Helpers.participatingRunnableLaneIds? warp guard? = some lanes := Iff.rfl`.

- `instance@283` ( instance, line 283): provides a typeclass instance needed by the executable semantic helper layer. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (warp : WarpState) (guard? : Option Guard) (lanes : List LaneId) : Decidable (Helpers.ParticipatingRunnable warp guard? lanes) := by`.

- `lockstepRunnable?` ( def, line 288): defines `lockstepRunnable?` in the executable semantic helper layer. The signature is `def lockstepRunnable? (warp : WarpState) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lockstepRunnable? (warp : WarpState) : Bool :=`.

- `lockstepRunnable` ( def, line 297): defines `lockstepRunnable` in the executable semantic helper layer. The signature is `def lockstepRunnable (warp : WarpState) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lockstepRunnable (warp : WarpState) : Prop :=`.

- `lockstepRunnable_iff_bool` ( theorem, line 300): proves a property in the executable semantic helper layer. The name and signature identify the exact fact: `theorem lockstepRunnable_iff_bool (warp : WarpState) : Helpers.lockstepRunnable warp ↔ Helpers.lockstepRunnable? warp = true := Iff.rfl`. Signature: `theorem lockstepRunnable_iff_bool (warp : WarpState) : Helpers.lockstepRunnable warp ↔ Helpers.lockstepRunnable? warp = true := Iff.rfl`.

- `instance@303` ( instance, line 303): provides a typeclass instance needed by the executable semantic helper layer. In practice this makes the associated predicate or datatype usable by Lean automation and decidability machinery. Signature: `instance (warp : WarpState) : Decidable (Helpers.lockstepRunnable warp) := by`.

- `getSpaceBaseMem?` ( def, line 307): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def getSpaceBaseMem? (st : State) (addr : Addr) : Option ByteMem :=`.

- `setSpaceBaseMem?` ( def, line 320): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def setSpaceBaseMem? (st : State) (addr : Addr) (bytes : ByteMem) : Option State :=`.

- `readMem?` ( def, line 335): defines `readMem?` in the executable semantic helper layer. The signature is `def readMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Option Value := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def readMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) : Option Value := do`.

- `writeMem?` ( def, line 344): defines `writeMem?` in the executable semantic helper layer. The signature is `def writeMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) (value : Value) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def writeMem? (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr) (value : Value) : Option State := do`.

- `evalCvta?` ( def, line 353): defines `evalCvta?` in the executable semantic helper layer. The signature is `def evalCvta? (space : AddrSpace) (value : Value) : Option Value :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def evalCvta? (space : AddrSpace) (value : Value) : Option Value :=`.

- `evalIsspacep?` ( def, line 383): defines `evalIsspacep?` in the executable semantic helper layer. The signature is `def evalIsspacep? (space : AddrSpace) (value : Value) : Option Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def evalIsspacep? (space : AddrSpace) (value : Value) : Option Bool :=`.

- `evalRValue?` ( partial def, line 389): defines `evalRValue?` in the executable semantic helper layer. The signature is `partial def evalRValue? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : RValue → Option Value | .imm v => some v | .reg r => do let laneState <- st.getLane? cta warp lane readReg laneState r | .pred p => do let laneState <- st.getLane? cta warp lane`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def evalRValue? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) : RValue → Option Value | .imm v => some v | .reg r => do let laneState <- st.getLane? cta warp lane readReg laneState r | .pred p => do let laneState <- st.getLane? cta warp lane`.

- `evalUnary?` ( partial def, line 412): defines `evalUnary?` in the executable semantic helper layer. The signature is `partial def evalUnary? : ScalarUnaryOp → Value → Option Value | .mov, v => some v | .neg, .s32 x => some (.s32 (normalizeSigned 32 (-x))) | .neg, .s64 x => some (.s64 (normalizeSigned 64 (-x))) | .neg, .f32 x => some (.f32 (-x)) | .neg, .f64 x => some (.f64 (-x))`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def evalUnary? : ScalarUnaryOp → Value → Option Value | .mov, v => some v | .neg, .s32 x => some (.s32 (normalizeSigned 32 (-x))) | .neg, .s64 x => some (.s64 (normalizeSigned 64 (-x))) | .neg, .f32 x => some (.f32 (-x)) | .neg, .f64 x => some (.f64 (-x))`.

- `evalBinary?` ( partial def, line 452): defines `evalBinary?` in the executable semantic helper layer. The signature is `partial def evalBinary? : ScalarBinaryOp → Value → Value → Option Value | .mulWideS32, .s32 a, .s32 b => some (.s64 (normalizeSigned 64 (a * b))) | .bitor, .pred a, .pred b => some (.pred (a || b)) | .bitand, .pred a, .pred b => some (.pred (a && b)) | .bitxor, .pred a, .pred b => some (.pred (a != b))`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def evalBinary? : ScalarBinaryOp → Value → Value → Option Value | .mulWideS32, .s32 a, .s32 b => some (.s64 (normalizeSigned 64 (a * b))) | .bitor, .pred a, .pred b => some (.pred (a || b)) | .bitand, .pred a, .pred b => some (.pred (a && b)) | .bitxor, .pred a, .pred b => some (.pred (a != b))`.

- `evalTernary?` ( partial def, line 515): defines `evalTernary?` in the executable semantic helper layer. The signature is `partial def evalTernary? : ScalarTernaryOp → Value → Value → Value → Option Value | .mad, .u32 a, .u32 b, .u32 c => some (.u32 (a * b + c)) | .mad, .u64 a, .u64 b, .u64 c => some (.u64 (a * b + c)) | .mad, .s32 a, .s32 b, .s32 c => some (.s32 (normalizeSigned 32 (a * b + c)))`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def evalTernary? : ScalarTernaryOp → Value → Value → Value → Option Value | .mad, .u32 a, .u32 b, .u32 c => some (.u32 (a * b + c)) | .mad, .u64 a, .u64 b, .u64 c => some (.u64 (a * b + c)) | .mad, .s32 a, .s32 b, .s32 c => some (.s32 (normalizeSigned 32 (a * b + c)))`.

- `evalCmp?` ( def, line 527): defines `evalCmp?` in the executable semantic helper layer. The signature is `def evalCmp? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (cmp : CmpExpr) : Option Bool := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def evalCmp? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (cmp : CmpExpr) : Option Bool := do`.

- `resolveAddr?` ( def, line 569): defines `resolveAddr?` in the executable semantic helper layer. The signature is `def resolveAddr? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (ta : TypedAddr) : Option Addr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def resolveAddr? (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (ta : TypedAddr) : Option Addr := do`.

- `uniformBranchDestination?` ( def, line 600): defines `uniformBranchDestination?` in the executable semantic helper layer. The signature is `def uniformBranchDestination? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId) (cond : RValue) (tLabel fLabel : BlockLabel) : Option PC := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def uniformBranchDestination? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId) (cond : RValue) (tLabel fLabel : BlockLabel) : Option PC := do`.

- `advancePcForLane` ( def, line 611): defines `advancePcForLane` in the executable semantic helper layer. The signature is `def advancePcForLane (laneState : LaneState) : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def advancePcForLane (laneState : LaneState) : LaneState :=`.

- `applyToLaneIds?` ( def, line 615): defines `applyToLaneIds?` in the executable semantic helper layer. The signature is `def applyToLaneIds? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId) (f : LaneId → LaneState → Option LaneState) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def applyToLaneIds? (st : State) (cta : CTAId) (warp : WarpId) (lanes : List LaneId) (f : LaneId → LaneState → Option LaneState) : Option State := do`.

- `advanceRunnablePcs?` ( def, line 624): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `def advanceRunnablePcs? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do`.

- `barrierArrivalPresent` ( def, line 633): defines `barrierArrivalPresent` in the executable semantic helper layer. The signature is `def barrierArrivalPresent (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def barrierArrivalPresent (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) : Bool :=`.

- `addBarrierArrival` ( def, line 636): defines `addBarrierArrival` in the executable semantic helper layer. The signature is `def addBarrierArrival (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) : List (WarpId × LaneId) :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addBarrierArrival (token : WarpId × LaneId) (arrived : List (WarpId × LaneId)) : List (WarpId × LaneId) :=`.

- `barrierExpectedCount` ( def, line 640): defines `barrierExpectedCount` in the executable semantic helper layer. The signature is `def barrierExpectedCount (st : State) (inst : BarrierInstance) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def barrierExpectedCount (st : State) (inst : BarrierInstance) : Nat :=`.

- `setBarrierInstance?` ( def, line 643): defines `setBarrierInstance?` in the executable semantic helper layer. The signature is `def setBarrierInstance? (st : State) (cta : CTAId) (barrierId : Nat) (inst : BarrierInstance) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def setBarrierInstance? (st : State) (cta : CTAId) (barrierId : Nat) (inst : BarrierInstance) : Option State := do`.

- `stepBarrierCTA?` ( def, line 649): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `def stepBarrierCTA? (st : State) (cta : CTAId) (warp : WarpId) (barrierId : Nat) (participants : List LaneId) : Option State := do`.

- `stepInstr?` ( def, line 683): defines `stepInstr?` in the executable semantic helper layer. The signature is `def stepInstr? (st : State) (cta : CTAId) (warp : WarpId) (gi : GInstr) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def stepInstr? (st : State) (cta : CTAId) (warp : WarpId) (gi : GInstr) : Option State := do`.

- `stepTerminator?` ( def, line 735): defines `stepTerminator?` in the executable semantic helper layer. The signature is `def stepTerminator? (st : State) (cta : CTAId) (warp : WarpId) (term : Terminator) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def stepTerminator? (st : State) (cta : CTAId) (warp : WarpId) (term : Terminator) : Option State := do`.

### `CLean/Semantics/SmallStep.lean`

Imports: `CLean.Semantics.Helpers`.

Layer role: relational small-step layer.

- `StepInstr` ( inductive, line 7): defines an algebraic datatype used in the relational small-step layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive StepInstr : State → CTAId → WarpId → GInstr → State → Prop where`.

- `StepBlock` ( inductive, line 17): defines an algebraic datatype used in the relational small-step layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive StepBlock : State → CTAId → WarpId → State → Prop where`.

- `StepWarp` ( inductive, line 39): defines an algebraic datatype used in the relational small-step layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive StepWarp : State → CTAId → WarpId → State → Prop where`.

- `StepMachine` ( inductive, line 45): defines an algebraic datatype used in the relational small-step layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive StepMachine : State → State → Prop where`.

### `CLean/PTX/Ast.lean`

Imports: `CLean.Core.IR`.

Layer role: PTX surface AST layer.

- `Operand` ( inductive, line 7): defines an algebraic datatype used in the PTX surface AST layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive Operand where`.

- `Instr` ( inductive, line 16): the executable instruction grammar. It covers scalar assignments, predicate assignments, memory operations, address conversion, barriers, warp ops, atomics, and MMA. Signature: `inductive Instr where`.

- `GInstr` ( structure, line 31): a guarded instruction, combining an optional `Guard` with an `Instr`. Lockstep execution first computes its participants, then runs the instruction. Signature: `structure GInstr where`.

- `Terminator` ( inductive, line 36): the block terminator grammar: unconditional branch, uniform conditional branch, and termination. Terminator execution updates PCs or lane status. Signature: `inductive Terminator where`.

- `Block` ( structure, line 42): a basic block: label, array of guarded instructions, and terminator. Kernel environments are maps from labels to these blocks. Signature: `structure Block where`.

- `RegDecl` ( structure, line 48): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure RegDecl where`.

- `PredDecl` ( structure, line 53): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure PredDecl where`.

- `ParamDecl` ( structure, line 57): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure ParamDecl where`.

- `SharedDecl` ( structure, line 65): metadata for a lowered shared-memory allocation. It records a name, size, alignment, and offset inside CTA shared memory. Signature: `structure SharedDecl where`.

- `ModuleDirective` ( inductive, line 72): defines an algebraic datatype used in the PTX surface AST layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive ModuleDirective where`.

- `ModuleMemorySpace` ( inductive, line 78): defines an algebraic datatype used in the PTX surface AST layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive ModuleMemorySpace where`.

- `ModuleMemoryDecl` ( structure, line 83): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure ModuleMemoryDecl where`.

- `Kernel` ( structure, line 91): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure Kernel where`.

- `Module` ( structure, line 101): defines a record used in the PTX surface AST layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure Module where`.

### `CLean/PTX/Bridge.lean`

Imports: `CLean.PTX.Parser`, `CLean.PTX.Lowering`.

Layer role: PTX ingestion bridge layer.

- `BridgeError` ( inductive, line 8): defines an algebraic datatype used in the PTX ingestion bridge layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive BridgeError where`.

- `CheckedKernel` ( structure, line 13): defines a record used in the PTX ingestion bridge layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure CheckedKernel where`.

- `CheckedModule` ( structure, line 18): defines a record used in the PTX ingestion bridge layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure CheckedModule where`.

- `lowerInstrSupported?` ( def, line 24): defines `lowerInstrSupported?` in the PTX ingestion bridge layer. The signature is `def lowerInstrSupported? (env : Typing.TypeEnv) (instr : Instr) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerInstrSupported? (env : Typing.TypeEnv) (instr : Instr) : Bool :=`.

- `lowerGInstrSupported?` ( def, line 30): defines `lowerGInstrSupported?` in the PTX ingestion bridge layer. The signature is `def lowerGInstrSupported? (env : Typing.TypeEnv) (gi : GInstr) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerGInstrSupported? (env : Typing.TypeEnv) (gi : GInstr) : Bool :=`.

- `lowerBlockSupported?` ( def, line 36): defines `lowerBlockSupported?` in the PTX ingestion bridge layer. The signature is `def lowerBlockSupported? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (block : Block) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerBlockSupported? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (block : Block) : Bool :=`.

- `lowerKernelSupported?` ( def, line 43): defines `lowerKernelSupported?` in the PTX ingestion bridge layer. The signature is `def lowerKernelSupported? (kernel : Kernel) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerKernelSupported? (kernel : Kernel) : Bool :=`.

- `parseAndLowerKernel?` ( def, line 49): defines `parseAndLowerKernel?` in the PTX ingestion bridge layer. The signature is `def parseAndLowerKernel? (input : String) : Except BridgeError CheckedKernel := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def parseAndLowerKernel? (input : String) : Except BridgeError CheckedKernel := do`.

- `parseAndLowerModule?` ( def, line 59): defines `parseAndLowerModule?` in the PTX ingestion bridge layer. The signature is `def parseAndLowerModule? (input : String) : Except BridgeError CheckedModule := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def parseAndLowerModule? (input : String) : Except BridgeError CheckedModule := do`.

- `parseAndLowerKernelOk?` ( def, line 72): defines `parseAndLowerKernelOk?` in the PTX ingestion bridge layer. The signature is `def parseAndLowerKernelOk? (input : String) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def parseAndLowerKernelOk? (input : String) : Bool :=`.

- `parseAndLowerModuleOk?` ( def, line 78): defines `parseAndLowerModuleOk?` in the PTX ingestion bridge layer. The signature is `def parseAndLowerModuleOk? (input : String) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def parseAndLowerModuleOk? (input : String) : Bool :=`.

### `CLean/PTX/Lowering.lean`

Imports: `CLean.Core.State`, `CLean.Core.Typing`, `CLean.PTX.Ast`.

Layer role: checked PTX-to-IR lowering layer.

- `LowerError` ( inductive, line 9): defines an algebraic datatype used in the checked PTX-to-IR lowering layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive LowerError where`.

- `LowerM` ( abbrev, line 24): introduces a naming alias in the checked PTX-to-IR lowering layer. The alias keeps signatures domain-specific without changing the underlying representation. Signature: `abbrev LowerM := Except LowerError`.

- `lowerOperand` ( def, line 26): defines `lowerOperand` in the checked PTX-to-IR lowering layer. The signature is `def lowerOperand : Operand → RValue | .reg r => .reg r | .pred p => .pred p | .symbol s => .reg s | .addr base 0 => lowerOperand base | .addr base (Int.ofNat n) => .binop .add (lowerOperand base) (.imm (.u64 (UInt64.ofNat n))) | .addr base (Int.negSucc n) => .binop .sub (lowerOperand base) (.imm (.u64 (UInt64.ofNat (n + 1))))`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerOperand : Operand → RValue | .reg r => .reg r | .pred p => .pred p | .symbol s => .reg s | .addr base 0 => lowerOperand base | .addr base (Int.ofNat n) => .binop .add (lowerOperand base) (.imm (.u64 (UInt64.ofNat n))) | .addr base (Int.negSucc n) => .binop .sub (lowerOperand base) (.imm (.u64 (UInt64.ofNat (n + 1))))`.

- `lowerParamOperand` ( def, line 36): defines `lowerParamOperand` in the checked PTX-to-IR lowering layer. The signature is `def lowerParamOperand (info : ParamInfo) : RValue :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerParamOperand (info : ParamInfo) : RValue :=`.

- `lowerSharedOperand` ( def, line 39): defines `lowerSharedOperand` in the checked PTX-to-IR lowering layer. The signature is `def lowerSharedOperand (info : CLean.SharedDecl) : RValue :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerSharedOperand (info : CLean.SharedDecl) : RValue :=`.

- `addressOffsetValue?` ( def, line 42): defines `addressOffsetValue?` in the checked PTX-to-IR lowering layer. The signature is `def addressOffsetValue? (ty : ScalarTy) (offset : Nat) : Option Value :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addressOffsetValue? (ty : ScalarTy) (offset : Nat) : Option Value :=`.

- `lowerOperandChecked?` ( def, line 50): defines `lowerOperandChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerOperandChecked? (env : Typing.TypeEnv) : Operand → LowerM RValue | .reg r => match env.regs[r]? with | some _ => pure (.reg r) | none => throw (.unknownReg r) | .pred p => match env.preds[p]? with | some .pred => pure (.pred p) | some ty => throw (.typeMismatch .pred ty)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerOperandChecked? (env : Typing.TypeEnv) : Operand → LowerM RValue | .reg r => match env.regs[r]? with | some _ => pure (.reg r) | none => throw (.unknownReg r) | .pred p => match env.preds[p]? with | some .pred => pure (.pred p) | some ty => throw (.typeMismatch .pred ty)`.

- `operandType?` ( def, line 65): defines `operandType?` in the checked PTX-to-IR lowering layer. The signature is `def operandType? (env : Typing.TypeEnv) : Operand → LowerM ScalarTy | .reg r => match env.regs[r]? with | some ty => pure ty | none => throw (.unknownReg r) | .pred p => match env.preds[p]? with | some .pred => pure .pred | some ty => throw (.typeMismatch .pred ty)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def operandType? (env : Typing.TypeEnv) : Operand → LowerM ScalarTy | .reg r => match env.regs[r]? with | some ty => pure ty | none => throw (.unknownReg r) | .pred p => match env.preds[p]? with | some .pred => pure .pred | some ty => throw (.typeMismatch .pred ty)`.

- `addressOperandType?` ( def, line 86): defines `addressOperandType?` in the checked PTX-to-IR lowering layer. The signature is `def addressOperandType? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM ScalarTy | .addr base _ => addressOperandType? env space base | .symbol s => match space with | .param => match env.params[s]? with | some _ => pure .u64 | none => throw (.unknownParam s)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addressOperandType? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM ScalarTy | .addr base _ => addressOperandType? env space base | .symbol s => match space with | .param => match env.params[s]? with | some _ => pure .u64 | none => throw (.unknownParam s)`.

- `lowerAddressOperandChecked?` ( def, line 101): defines `lowerAddressOperandChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerAddressOperandChecked? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM RValue | .addr base offset => do let baseTy <- addressOperandType? env space base let baseRv <- lowerAddressOperandChecked? env space base if offset = 0 then pure baseRv`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerAddressOperandChecked? (env : Typing.TypeEnv) (space : AddrSpace) : Operand → LowerM RValue | .addr base offset => do let baseTy <- addressOperandType? env space base let baseRv <- lowerAddressOperandChecked? env space base if offset = 0 then pure baseRv`.

- `expectOperandType` ( def, line 131): defines `expectOperandType` in the checked PTX-to-IR lowering layer. The signature is `def expectOperandType (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM Unit := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def expectOperandType (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM Unit := do`.

- `coerceOperandTo?` ( def, line 138): defines `coerceOperandTo?` in the checked PTX-to-IR lowering layer. The signature is `def coerceOperandTo? (expected actual : ScalarTy) (rv : RValue) : RValue :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def coerceOperandTo? (expected actual : ScalarTy) (rv : RValue) : RValue :=`.

- `lowerOperandCheckedAs?` ( def, line 152): defines `lowerOperandCheckedAs?` in the checked PTX-to-IR lowering layer. The signature is `def lowerOperandCheckedAs? (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM RValue := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerOperandCheckedAs? (env : Typing.TypeEnv) (expected : ScalarTy) (operand : Operand) : LowerM RValue := do`.

- `ensureCodecType` ( def, line 161): defines `ensureCodecType` in the checked PTX-to-IR lowering layer. The signature is `def ensureCodecType (ty : ScalarTy) : LowerM Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ensureCodecType (ty : ScalarTy) : LowerM Unit :=`.

- `ensureCvtaSourceType` ( def, line 167): defines `ensureCvtaSourceType` in the checked PTX-to-IR lowering layer. The signature is `def ensureCvtaSourceType (ty : ScalarTy) : LowerM Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ensureCvtaSourceType (ty : ScalarTy) : LowerM Unit :=`.

- `ensureGuard?` ( def, line 173): defines `ensureGuard?` in the checked PTX-to-IR lowering layer. The signature is `def ensureGuard? (env : Typing.TypeEnv) : Option Guard → LowerM Unit | none => pure () | some g => match env.preds[g.pred]? with | some .pred => pure () | some ty => throw (.typeMismatch .pred ty) | none => throw (.unknownPred g.pred)`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ensureGuard? (env : Typing.TypeEnv) : Option Guard → LowerM Unit | none => pure () | some g => match env.preds[g.pred]? with | some .pred => pure () | some ty => throw (.typeMismatch .pred ty) | none => throw (.unknownPred g.pred)`.

- `lowerInstr` ( def, line 181): defines `lowerInstr` in the checked PTX-to-IR lowering layer. The signature is `def lowerInstr : Instr → CLean.Instr | .mov _ dst src => .assignReg dst (lowerOperand src) | .unop op _ dst src => .assignReg dst (.unop op (lowerOperand src)) | .binop op _ dst lhs rhs => .assignReg dst (.binop op (lowerOperand lhs) (lowerOperand rhs)) | .predBinop op dst lhs rhs => .assignPredValue dst (.binop op (lowerOperand lhs) (lowerOperand rhs))`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerInstr : Instr → CLean.Instr | .mov _ dst src => .assignReg dst (lowerOperand src) | .unop op _ dst src => .assignReg dst (.unop op (lowerOperand src)) | .binop op _ dst lhs rhs => .assignReg dst (.binop op (lowerOperand lhs) (lowerOperand rhs)) | .predBinop op dst lhs rhs => .assignPredValue dst (.binop op (lowerOperand lhs) (lowerOperand rhs))`.

- `lowerInstrChecked?` ( def, line 195): defines `lowerInstrChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerInstrChecked? (env : Typing.TypeEnv) : Instr → LowerM (CLean.Instr × Typing.TypeEnv) | .mov ty dst src => do expectOperandType env ty src let src <- lowerOperandCheckedAs? env ty src pure (.assignReg dst src, { env with regs := env.regs.insert dst ty })`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerInstrChecked? (env : Typing.TypeEnv) : Instr → LowerM (CLean.Instr × Typing.TypeEnv) | .mov ty dst src => do expectOperandType env ty src let src <- lowerOperandCheckedAs? env ty src pure (.assignReg dst src, { env with regs := env.regs.insert dst ty })`.

- `lowerGInstr` ( def, line 287): defines `lowerGInstr` in the checked PTX-to-IR lowering layer. The signature is `def lowerGInstr (gi : GInstr) : CLean.GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerGInstr (gi : GInstr) : CLean.GInstr :=`.

- `lowerGInstrChecked?` ( def, line 290): defines `lowerGInstrChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerGInstrChecked? (env : Typing.TypeEnv) (gi : GInstr) : LowerM (CLean.GInstr × Typing.TypeEnv) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerGInstrChecked? (env : Typing.TypeEnv) (gi : GInstr) : LowerM (CLean.GInstr × Typing.TypeEnv) := do`.

- `lowerTerminator` ( def, line 295): defines `lowerTerminator` in the checked PTX-to-IR lowering layer. The signature is `def lowerTerminator : Terminator → CLean.Terminator | .bra label => .br label | .cbra pred false tLabel fLabel => .cbr (.pred pred) tLabel fLabel | .cbra pred true tLabel fLabel => .cbr (.pred pred) fLabel tLabel | .exit => .terminate`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerTerminator : Terminator → CLean.Terminator | .bra label => .br label | .cbra pred false tLabel fLabel => .cbr (.pred pred) tLabel fLabel | .cbra pred true tLabel fLabel => .cbr (.pred pred) fLabel tLabel | .exit => .terminate`.

- `blockLabels` ( def, line 301): defines `blockLabels` in the checked PTX-to-IR lowering layer. The signature is `def blockLabels (blocks : Array Block) : Std.HashMap BlockLabel Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def blockLabels (blocks : Array Block) : Std.HashMap BlockLabel Unit :=`.

- `requireBlockLabel` ( def, line 304): defines `requireBlockLabel` in the checked PTX-to-IR lowering layer. The signature is `def requireBlockLabel (labels : Std.HashMap BlockLabel Unit) (label : BlockLabel) : LowerM Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def requireBlockLabel (labels : Std.HashMap BlockLabel Unit) (label : BlockLabel) : LowerM Unit :=`.

- `lowerTerminatorChecked?` ( def, line 309): defines `lowerTerminatorChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerTerminatorChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) : Terminator → LowerM CLean.Terminator | .bra label => do requireBlockLabel labels label pure (.br label) | .cbra pred false tLabel fLabel => do requireBlockLabel labels tLabel`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerTerminatorChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) : Terminator → LowerM CLean.Terminator | .bra label => do requireBlockLabel labels label pure (.br label) | .cbra pred false tLabel fLabel => do requireBlockLabel labels tLabel`.

- `lowerBlock` ( def, line 330): defines `lowerBlock` in the checked PTX-to-IR lowering layer. The signature is `def lowerBlock (block : Block) : CLean.Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerBlock (block : Block) : CLean.Block :=`.

- `lowerGInstrsChecked?` ( def, line 335): defines `lowerGInstrsChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerGInstrsChecked? (env : Typing.TypeEnv) (body : Array GInstr) : LowerM (Array CLean.GInstr × Typing.TypeEnv) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerGInstrsChecked? (env : Typing.TypeEnv) (body : Array GInstr) : LowerM (Array CLean.GInstr × Typing.TypeEnv) := do`.

- `lowerBlockChecked?` ( def, line 345): defines `lowerBlockChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerBlockChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (block : Block) : LowerM (CLean.Block × Typing.TypeEnv) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerBlockChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (block : Block) : LowerM (CLean.Block × Typing.TypeEnv) := do`.

- `lowerBlocks` ( def, line 351): defines `lowerBlocks` in the checked PTX-to-IR lowering layer. The signature is `def lowerBlocks (blocks : Array Block) : Std.HashMap BlockLabel CLean.Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerBlocks (blocks : Array Block) : Std.HashMap BlockLabel CLean.Block :=`.

- `alignUp` ( def, line 354): defines `alignUp` in the checked PTX-to-IR lowering layer. The signature is `def alignUp (offset align : Nat) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def alignUp (offset align : Nat) : Nat :=`.

- `lowerParamsChecked?` ( def, line 360): defines `lowerParamsChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerParamsChecked? (params : Array ParamDecl) : LowerM (Array ParamInfo) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerParamsChecked? (params : Array ParamDecl) : LowerM (Array ParamInfo) := do`.

- `paramInfoMap` ( def, line 381): defines `paramInfoMap` in the checked PTX-to-IR lowering layer. The signature is `def paramInfoMap (params : Array ParamInfo) : Std.HashMap String ParamInfo :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def paramInfoMap (params : Array ParamInfo) : Std.HashMap String ParamInfo :=`.

- `lowerSharedsChecked?` ( def, line 384): defines `lowerSharedsChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerSharedsChecked? (shareds : Array SharedDecl) : LowerM (Array CLean.SharedDecl) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerSharedsChecked? (shareds : Array SharedDecl) : LowerM (Array CLean.SharedDecl) := do`.

- `sharedInfoMap` ( def, line 404): defines `sharedInfoMap` in the checked PTX-to-IR lowering layer. The signature is `def sharedInfoMap (shareds : Array CLean.SharedDecl) : Std.HashMap String CLean.SharedDecl :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def sharedInfoMap (shareds : Array CLean.SharedDecl) : Std.HashMap String CLean.SharedDecl :=`.

- `initialTypeEnv` ( def, line 407): defines `initialTypeEnv` in the checked PTX-to-IR lowering layer. The signature is `def initialTypeEnv (kernel : Kernel) (params : Array ParamInfo := #[])`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def initialTypeEnv (kernel : Kernel) (params : Array ParamInfo := #[])`.

- `lowerBlocksChecked?` ( def, line 413): defines `lowerBlocksChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerBlocksChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (blocks : Array Block) : LowerM (Std.HashMap BlockLabel CLean.Block) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerBlocksChecked? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv) (blocks : Array Block) : LowerM (Std.HashMap BlockLabel CLean.Block) := do`.

- `lowerKernelEnv` ( def, line 421): defines `lowerKernelEnv` in the checked PTX-to-IR lowering layer. The signature is `def lowerKernelEnv (kernel : Kernel) : KernelEnv :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerKernelEnv (kernel : Kernel) : KernelEnv :=`.

- `lowerKernelEnvChecked?` ( def, line 428): defines `lowerKernelEnvChecked?` in the checked PTX-to-IR lowering layer. The signature is `def lowerKernelEnvChecked? (kernel : Kernel) : LowerM KernelEnv := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerKernelEnvChecked? (kernel : Kernel) : LowerM KernelEnv := do`.

- `lowerKernelEnvCheckedD` ( def, line 442): defines `lowerKernelEnvCheckedD` in the checked PTX-to-IR lowering layer. The signature is `def lowerKernelEnvCheckedD (kernel : Kernel) (default : KernelEnv := {}) : KernelEnv :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lowerKernelEnvCheckedD (kernel : Kernel) (default : KernelEnv := {}) : KernelEnv :=`.

### `CLean/PTX/Parser.lean`

Imports: `Std.Internal.Parsec.String`, `CLean.PTX.Ast`.

Layer role: PTX parser layer.

- `Parser` ( abbrev, line 10): introduces a naming alias in the PTX parser layer. The alias keeps signatures domain-specific without changing the underlying representation. Signature: `abbrev Parser := Std.Internal.Parsec.String.Parser`.

- `ParseError` ( structure, line 12): defines a record used in the PTX parser layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure ParseError where`.

- `lineColumnFromOffset` ( def, line 19): defines `lineColumnFromOffset` in the PTX parser layer. The signature is `def lineColumnFromOffset (input : String) (offset : Nat) : Nat × Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lineColumnFromOffset (input : String) (offset : Nat) : Nat × Nat :=`.

- `runParser` ( def, line 32): defines `runParser` in the PTX parser layer. The signature is `def runParser (p : Parser α) (input : String) : Except ParseError α :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def runParser (p : Parser α) (input : String) : Except ParseError α :=`.

- `whitespace` ( def, line 40): defines `whitespace` in the PTX parser layer. The signature is `def whitespace : Parser Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def whitespace : Parser Unit :=`.

- `lineComment` ( def, line 43): defines `lineComment` in the PTX parser layer. The signature is `def lineComment : Parser Unit := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lineComment : Parser Unit := do`.

- `blockComment` ( partial def, line 48): defines `blockComment` in the PTX parser layer. The signature is `partial def blockComment : Parser Unit := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def blockComment : Parser Unit := do`.

- `trivia` ( partial def, line 58): defines `trivia` in the PTX parser layer. The signature is `partial def trivia : Parser Unit := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def trivia : Parser Unit := do`.

- `lexeme` ( def, line 63): defines `lexeme` in the PTX parser layer. The signature is `def lexeme (p : Parser α) : Parser α :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lexeme (p : Parser α) : Parser α :=`.

- `symbol` ( def, line 66): defines `symbol` in the PTX parser layer. The signature is `def symbol (s : String) : Parser Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def symbol (s : String) : Parser Unit :=`.

- `rawSymbol` ( def, line 69): defines `rawSymbol` in the PTX parser layer. The signature is `def rawSymbol (s : String) : Parser Unit :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def rawSymbol (s : String) : Parser Unit :=`.

- `optional?` ( def, line 72): defines `optional?` in the PTX parser layer. The signature is `def optional? (p : Parser α) : Parser (Option α) :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def optional? (p : Parser α) : Parser (Option α) :=`.

- `sepBy` ( def, line 75): defines `sepBy` in the PTX parser layer. The signature is `def sepBy (p : Parser α) (sep : Parser Unit) : Parser (Array α) :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def sepBy (p : Parser α) (sep : Parser Unit) : Parser (Array α) :=`.

- `charBetween` ( def, line 81): defines `charBetween` in the PTX parser layer. The signature is `def charBetween (lo hi c : Char) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def charBetween (lo hi c : Char) : Bool :=`.

- `identStart` ( def, line 84): defines `identStart` in the PTX parser layer. The signature is `def identStart (c : Char) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def identStart (c : Char) : Bool :=`.

- `identRest` ( def, line 87): defines `identRest` in the PTX parser layer. The signature is `def identRest (c : Char) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def identRest (c : Char) : Bool :=`.

- `tokenChar` ( def, line 90): defines `tokenChar` in the PTX parser layer. The signature is `def tokenChar (c : Char) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def tokenChar (c : Char) : Bool :=`.

- `NameRef` ( structure, line 93): defines a record used in the PTX parser layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure NameRef where`.

- `nameRef` ( def, line 98): defines `nameRef` in the PTX parser layer. The signature is `def nameRef : Parser NameRef := lexeme do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def nameRef : Parser NameRef := lexeme do`.

- `ident` ( def, line 104): defines `ident` in the PTX parser layer. The signature is `def ident : Parser String := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ident : Parser String := do`.

- `rawToken` ( def, line 107): defines `rawToken` in the PTX parser layer. The signature is `def rawToken : Parser String :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def rawToken : Parser String :=`.

- `natFromDigits` ( def, line 110): defines `natFromDigits` in the PTX parser layer. The signature is `def natFromDigits (digits : Array Char) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natFromDigits (digits : Array Char) : Nat :=`.

- `hexDigitValue?` ( def, line 113): defines `hexDigitValue?` in the PTX parser layer. The signature is `def hexDigitValue? (c : Char) : Option Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def hexDigitValue? (c : Char) : Option Nat :=`.

- `natFromHexDigits` ( def, line 119): defines `natFromHexDigits` in the PTX parser layer. The signature is `def natFromHexDigits (digits : Array Char) : Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natFromHexDigits (digits : Array Char) : Nat :=`.

- `decNatRaw` ( def, line 125): defines `decNatRaw` in the PTX parser layer. The signature is `def decNatRaw : Parser Nat := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def decNatRaw : Parser Nat := do`.

- `hexNatRaw` ( def, line 129): defines `hexNatRaw` in the PTX parser layer. The signature is `def hexNatRaw : Parser Nat := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def hexNatRaw : Parser Nat := do`.

- `hexFloat32Raw` ( def, line 134): defines `hexFloat32Raw` in the PTX parser layer. The signature is `def hexFloat32Raw : Parser Float := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def hexFloat32Raw : Parser Float := do`.

- `natRaw` ( def, line 139): defines `natRaw` in the PTX parser layer. The signature is `def natRaw : Parser Nat :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natRaw : Parser Nat :=`.

- `intLit` ( def, line 142): defines `intLit` in the PTX parser layer. The signature is `def intLit : Parser Int := lexeme do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def intLit : Parser Int := lexeme do`.

- `natLit` ( def, line 147): defines `natLit` in the PTX parser layer. The signature is `def natLit : Parser Nat := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def natLit : Parser Nat := do`.

- `scalarTySuffix` ( def, line 151): defines `scalarTySuffix` in the PTX parser layer. The signature is `def scalarTySuffix : Parser ScalarTy :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def scalarTySuffix : Parser ScalarTy :=`.

- `scalarTy` ( def, line 170): defines `scalarTy` in the PTX parser layer. The signature is `def scalarTy : Parser ScalarTy :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def scalarTy : Parser ScalarTy :=`.

- `addrSpaceSuffix` ( def, line 173): defines `addrSpaceSuffix` in the PTX parser layer. The signature is `def addrSpaceSuffix : Parser AddrSpace :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addrSpaceSuffix : Parser AddrSpace :=`.

- `addrSpace` ( def, line 181): defines `addrSpace` in the PTX parser layer. The signature is `def addrSpace : Parser AddrSpace :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addrSpace : Parser AddrSpace :=`.

- `cmpOp` ( def, line 184): defines `cmpOp` in the PTX parser layer. The signature is `def cmpOp : Parser CmpOp :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def cmpOp : Parser CmpOp :=`.

- `comma` ( def, line 192): defines `comma` in the PTX parser layer. The signature is `def comma : Parser Unit := symbol ","`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def comma : Parser Unit := symbol ","`.

- `semi` ( def, line 193): defines `semi` in the PTX parser layer. The signature is `def semi : Parser Unit := symbol ";"`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def semi : Parser Unit := symbol ";"`.

- `optionalSemi` ( def, line 194): defines `optionalSemi` in the PTX parser layer. The signature is `def optionalSemi : Parser Unit := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def optionalSemi : Parser Unit := do`.

- `colon` ( def, line 197): defines `colon` in the PTX parser layer. The signature is `def colon : Parser Unit := symbol ":"`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def colon : Parser Unit := symbol ":"`.

- `lbrace` ( def, line 198): defines `lbrace` in the PTX parser layer. The signature is `def lbrace : Parser Unit := symbol "{"`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lbrace : Parser Unit := symbol "{"`.

- `rbrace` ( def, line 199): defines `rbrace` in the PTX parser layer. The signature is `def rbrace : Parser Unit := symbol "}"`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def rbrace : Parser Unit := symbol "}"`.

- `lparen` ( def, line 200): defines `lparen` in the PTX parser layer. The signature is `def lparen : Parser Unit := symbol "("`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lparen : Parser Unit := symbol "("`.

- `rparen` ( def, line 201): defines `rparen` in the PTX parser layer. The signature is `def rparen : Parser Unit := symbol ")"`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def rparen : Parser Unit := symbol ")"`.

- `immediateValue?` ( def, line 203): defines `immediateValue?` in the PTX parser layer. The signature is `def immediateValue? (ty : ScalarTy) (n : Int) : Option Value :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def immediateValue? (ty : ScalarTy) (n : Int) : Option Value :=`.

- `specialReg?` ( def, line 222): defines `specialReg?` in the PTX parser layer. The signature is `def specialReg? : String → Option SpecialReg | "tid.x" => some .tidX | "tid.y" => some .tidY | "tid.z" => some .tidZ | "ctaid.x" => some .ctaidX | "ctaid.y" => some .ctaidY | "ctaid.z" => some .ctaidZ | "ntid.x" => some .ntidX | "ntid.y" => some .ntidY | "ntid.z" => some .ntidZ`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def specialReg? : String → Option SpecialReg | "tid.x" => some .tidX | "tid.y" => some .tidY | "tid.z" => some .tidZ | "ctaid.x" => some .ctaidX | "ctaid.y" => some .ctaidY | "ctaid.z" => some .ctaidZ | "ntid.x" => some .ntidX | "ntid.y" => some .ntidY | "ntid.z" => some .ntidZ`.

- `operandOfNameRef` ( def, line 237): defines `operandOfNameRef` in the PTX parser layer. The signature is `def operandOfNameRef (ref : NameRef) : Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def operandOfNameRef (ref : NameRef) : Operand :=`.

- `operandOfNameRefWithTy` ( def, line 242): defines `operandOfNameRefWithTy` in the PTX parser layer. The signature is `def operandOfNameRefWithTy (ty : ScalarTy) (ref : NameRef) : Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def operandOfNameRefWithTy (ty : ScalarTy) (ref : NameRef) : Operand :=`.

- `operandWithTy` ( def, line 247): defines `operandWithTy` in the PTX parser layer. The signature is `def operandWithTy (ty : ScalarTy) : Parser Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def operandWithTy (ty : ScalarTy) : Parser Operand :=`.

- `addrAtom` ( def, line 261): defines `addrAtom` in the PTX parser layer. The signature is `def addrAtom : Parser Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addrAtom : Parser Operand :=`.

- `signedOffset` ( def, line 267): defines `signedOffset` in the PTX parser layer. The signature is `def signedOffset : Parser Int :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def signedOffset : Parser Int :=`.

- `bracketAddrOperand` ( def, line 271): defines `bracketAddrOperand` in the PTX parser layer. The signature is `def bracketAddrOperand : Parser Operand := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def bracketAddrOperand : Parser Operand := do`.

- `addrOperand` ( def, line 278): defines `addrOperand` in the PTX parser layer. The signature is `def addrOperand : Parser Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addrOperand : Parser Operand :=`.

- `guard` ( def, line 281): defines `guard` in the PTX parser layer. The signature is `def guard : Parser Guard := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def guard : Parser Guard := do`.

- `instrMov` ( def, line 287): defines `instrMov` in the PTX parser layer. The signature is `def instrMov : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrMov : Parser Instr := do`.

- `unaryOpcode` ( def, line 296): defines `unaryOpcode` in the PTX parser layer. The signature is `def unaryOpcode : Parser ScalarUnaryOp :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def unaryOpcode : Parser ScalarUnaryOp :=`.

- `instrUnop` ( def, line 301): defines `instrUnop` in the PTX parser layer. The signature is `def instrUnop : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrUnop : Parser Instr := do`.

- `instrCvt` ( def, line 310): defines `instrCvt` in the PTX parser layer. The signature is `def instrCvt : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrCvt : Parser Instr := do`.

- `instrMadLo` ( def, line 320): defines `instrMadLo` in the PTX parser layer. The signature is `def instrMadLo : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrMadLo : Parser Instr := do`.

- `instrFmaRn` ( def, line 333): defines `instrFmaRn` in the PTX parser layer. The signature is `def instrFmaRn : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrFmaRn : Parser Instr := do`.

- `scalarBinaryOp` ( def, line 346): defines `scalarBinaryOp` in the PTX parser layer. The signature is `def scalarBinaryOp : Parser ScalarBinaryOp :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def scalarBinaryOp : Parser ScalarBinaryOp :=`.

- `instrMulLo` ( def, line 358): defines `instrMulLo` in the PTX parser layer. The signature is `def instrMulLo : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrMulLo : Parser Instr := do`.

- `instrMulWideS32` ( def, line 369): defines `instrMulWideS32` in the PTX parser layer. The signature is `def instrMulWideS32 : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrMulWideS32 : Parser Instr := do`.

- `predBinaryOp` ( def, line 379): defines `predBinaryOp` in the PTX parser layer. The signature is `def predBinaryOp : Parser ScalarBinaryOp :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def predBinaryOp : Parser ScalarBinaryOp :=`.

- `instrPredBinop` ( def, line 384): defines `instrPredBinop` in the PTX parser layer. The signature is `def instrPredBinop : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrPredBinop : Parser Instr := do`.

- `instrBinop` ( def, line 394): defines `instrBinop` in the PTX parser layer. The signature is `def instrBinop : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrBinop : Parser Instr := do`.

- `instrSetp` ( def, line 405): defines `instrSetp` in the PTX parser layer. The signature is `def instrSetp : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrSetp : Parser Instr := do`.

- `instrLd` ( def, line 417): defines `instrLd` in the PTX parser layer. The signature is `def instrLd : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrLd : Parser Instr := do`.

- `instrSt` ( def, line 427): defines `instrSt` in the PTX parser layer. The signature is `def instrSt : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrSt : Parser Instr := do`.

- `instrCvta` ( def, line 437): defines `instrCvta` in the PTX parser layer. The signature is `def instrCvta : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrCvta : Parser Instr := do`.

- `instrIsspacep` ( def, line 448): defines `instrIsspacep` in the PTX parser layer. The signature is `def instrIsspacep : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrIsspacep : Parser Instr := do`.

- `instrBarSync` ( def, line 457): defines `instrBarSync` in the PTX parser layer. The signature is `def instrBarSync : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrBarSync : Parser Instr := do`.

- `unsupportedOperand` ( def, line 463): defines `unsupportedOperand` in the PTX parser layer. The signature is `def unsupportedOperand : Parser Operand :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def unsupportedOperand : Parser Operand :=`.

- `opcodeParts` ( def, line 466): defines `opcodeParts` in the PTX parser layer. The signature is `def opcodeParts (tok : String) : String × Array String :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def opcodeParts (tok : String) : String × Array String :=`.

- `instrUnsupported` ( def, line 471): defines `instrUnsupported` in the PTX parser layer. The signature is `def instrUnsupported : Parser Instr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrUnsupported : Parser Instr := do`.

- `instr` ( def, line 477): defines `instr` in the PTX parser layer. The signature is `def instr : Parser Instr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instr : Parser Instr :=`.

- `gInstr` ( def, line 495): defines `gInstr` in the PTX parser layer. The signature is `def gInstr : Parser GInstr := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def gInstr : Parser GInstr := do`.

- `terminatorBra` ( def, line 500): defines `terminatorBra` in the PTX parser layer. The signature is `def terminatorBra : Parser Terminator := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def terminatorBra : Parser Terminator := do`.

- `terminatorCbra` ( def, line 505): defines `terminatorCbra` in the PTX parser layer. The signature is `def terminatorCbra : Parser Terminator := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def terminatorCbra : Parser Terminator := do`.

- `terminatorExit` ( def, line 514): defines `terminatorExit` in the PTX parser layer. The signature is `def terminatorExit : Parser Terminator := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def terminatorExit : Parser Terminator := do`.

- `terminator` ( def, line 518): defines `terminator` in the PTX parser layer. The signature is `def terminator : Parser Terminator :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def terminator : Parser Terminator :=`.

- `Stmt` ( inductive, line 521): defines an algebraic datatype used in the PTX parser layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive Stmt where`.

- `implicitFallthroughLabel` ( def, line 528): defines `implicitFallthroughLabel` in the PTX parser layer. The signature is `def implicitFallthroughLabel : BlockLabel :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def implicitFallthroughLabel : BlockLabel :=`.

- `stmt` ( def, line 531): defines `stmt` in the PTX parser layer. The signature is `def stmt : Parser Stmt := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def stmt : Parser Stmt := do`.

- `blockOfStmts` ( def, line 550): defines `blockOfStmts` in the PTX parser layer. The signature is `def blockOfStmts (label : BlockLabel) (stmts : Array Stmt) : Block := Id.run do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def blockOfStmts (label : BlockLabel) (stmts : Array Stmt) : Block := Id.run do`.

- `fallthroughLabel` ( def, line 561): defines `fallthroughLabel` in the PTX parser layer. The signature is `def fallthroughLabel (label : BlockLabel) (idx : Nat) : BlockLabel :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def fallthroughLabel (label : BlockLabel) (idx : Nat) : BlockLabel :=`.

- `blocksOfStmts` ( partial def, line 564): defines `blocksOfStmts` in the PTX parser layer. The signature is `partial def blocksOfStmts (label : BlockLabel) (stmts : Array Stmt) : Array Block := Id.run do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `partial def blocksOfStmts (label : BlockLabel) (stmts : Array Stmt) : Array Block := Id.run do`.

- `linkImplicitFallthroughs` ( def, line 583): defines `linkImplicitFallthroughs` in the PTX parser layer. The signature is `def linkImplicitFallthroughs (blocks : Array Block) : Array Block := Id.run do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def linkImplicitFallthroughs (blocks : Array Block) : Array Block := Id.run do`.

- `block` ( def, line 600): defines `block` in the PTX parser layer. The signature is `def block : Parser (Array Block) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def block : Parser (Array Block) := do`.

- `NameDecl` ( structure, line 606): defines a record used in the PTX parser layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure NameDecl where`.

- `nameDecl` ( def, line 611): defines `nameDecl` in the PTX parser layer. The signature is `def nameDecl : Parser NameDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def nameDecl : Parser NameDecl := do`.

- `expandNames` ( def, line 616): defines `expandNames` in the PTX parser layer. The signature is `def expandNames (decl : NameDecl) : Array String :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def expandNames (decl : NameDecl) : Array String :=`.

- `regDecl` ( def, line 621): defines `regDecl` in the PTX parser layer. The signature is `def regDecl : Parser (Array RegDecl) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def regDecl : Parser (Array RegDecl) := do`.

- `regPredDecl` ( def, line 628): defines `regPredDecl` in the PTX parser layer. The signature is `def regPredDecl : Parser (Array PredDecl) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def regPredDecl : Parser (Array PredDecl) := do`.

- `predDecl` ( def, line 638): defines `predDecl` in the PTX parser layer. The signature is `def predDecl : Parser (Array PredDecl) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def predDecl : Parser (Array PredDecl) := do`.

- `ptrAttr` ( def, line 644): defines `ptrAttr` in the PTX parser layer. The signature is `def ptrAttr : Parser (Option AddrSpace × Nat) := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def ptrAttr : Parser (Option AddrSpace × Nat) := do`.

- `paramDeclCore` ( def, line 651): defines `paramDeclCore` in the PTX parser layer. The signature is `def paramDeclCore : Parser ParamDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def paramDeclCore : Parser ParamDecl := do`.

- `paramDecl` ( def, line 662): defines `paramDecl` in the PTX parser layer. The signature is `def paramDecl : Parser ParamDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def paramDecl : Parser ParamDecl := do`.

- `sharedDeclCore` ( def, line 667): defines `sharedDeclCore` in the PTX parser layer. The signature is `def sharedDeclCore : Parser SharedDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def sharedDeclCore : Parser SharedDecl := do`.

- `sharedDecl` ( def, line 675): defines `sharedDecl` in the PTX parser layer. The signature is `def sharedDecl : Parser SharedDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def sharedDecl : Parser SharedDecl := do`.

- `moduleMemoryDecl` ( def, line 680): defines `moduleMemoryDecl` in the PTX parser layer. The signature is `def moduleMemoryDecl (space : ModuleMemorySpace) : Parser ModuleMemoryDecl := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def moduleMemoryDecl (space : ModuleMemorySpace) : Parser ModuleMemoryDecl := do`.

- `Decl` ( inductive, line 688): defines an algebraic datatype used in the PTX parser layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive Decl where`.

- `decl` ( def, line 695): defines `decl` in the PTX parser layer. The signature is `def decl : Parser Decl :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def decl : Parser Decl :=`.

- `splitDecls` ( def, line 702): defines `splitDecls` in the PTX parser layer. The signature is `def splitDecls (decls : Array Decl) : Array RegDecl × Array PredDecl × Array ParamDecl × Array SharedDecl := Id.run do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def splitDecls (decls : Array Decl) : Array RegDecl × Array PredDecl × Array ParamDecl × Array SharedDecl := Id.run do`.

- `withLeadingBlocks` ( def, line 715): defines `withLeadingBlocks` in the PTX parser layer. The signature is `def withLeadingBlocks (entry : BlockLabel) (leading : Array Stmt) (blocks : Array Block) : Array Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def withLeadingBlocks (entry : BlockLabel) (leading : Array Stmt) (blocks : Array Block) : Array Block :=`.

- `entryParamList` ( def, line 718): defines `entryParamList` in the PTX parser layer. The signature is `def entryParamList : Parser (Array ParamDecl) :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def entryParamList : Parser (Array ParamDecl) :=`.

- `kernelBody` ( def, line 721): defines `kernelBody` in the PTX parser layer. The signature is `def kernelBody (entry : BlockLabel) (entryParams : Array ParamDecl) : Parser Kernel := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def kernelBody (entry : BlockLabel) (entryParams : Array ParamDecl) : Parser Kernel := do`.

- `kernel` ( def, line 736): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `def kernel : Parser Kernel := do`.

- `versionDirective` ( def, line 751): defines `versionDirective` in the PTX parser layer. The signature is `def versionDirective : Parser ModuleDirective := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def versionDirective : Parser ModuleDirective := do`.

- `targetDirective` ( def, line 757): defines `targetDirective` in the PTX parser layer. The signature is `def targetDirective : Parser ModuleDirective := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def targetDirective : Parser ModuleDirective := do`.

- `addressSizeDirective` ( def, line 764): defines `addressSizeDirective` in the PTX parser layer. The signature is `def addressSizeDirective : Parser ModuleDirective := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def addressSizeDirective : Parser ModuleDirective := do`.

- `directive` ( def, line 770): defines `directive` in the PTX parser layer. The signature is `def directive : Parser ModuleDirective :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def directive : Parser ModuleDirective :=`.

- `ModuleItem` ( inductive, line 773): defines an algebraic datatype used in the PTX parser layer. Its constructors enumerate the cases that lower layers evaluate and upper layers prove by case analysis. Signature: `inductive ModuleItem where`.

- `moduleItem` ( def, line 780): defines `moduleItem` in the PTX parser layer. The signature is `def moduleItem : Parser ModuleItem :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def moduleItem : Parser ModuleItem :=`.

- `moduleOfItems` ( def, line 787): defines `moduleOfItems` in the PTX parser layer. The signature is `def moduleOfItems (items : Array ModuleItem) : Module := Id.run do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def moduleOfItems (items : Array ModuleItem) : Module := Id.run do`.

- `moduleParser` ( def, line 800): defines `moduleParser` in the PTX parser layer. The signature is `def moduleParser : Parser Module := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def moduleParser : Parser Module := do`.

- `parseModule` ( def, line 807): defines `parseModule` in the PTX parser layer. The signature is `def parseModule (input : String) : Except ParseError Module :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def parseModule (input : String) : Except ParseError Module :=`.

- `parseKernel` ( def, line 810): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `def parseKernel (input : String) : Except ParseError Kernel := do`.

### `CLean/Proof/Automation.lean`

Imports: `CLean.Semantics.Execution`.

Layer role: proof automation convenience layer.

- `HoldsPost` ( def, line 6): wraps a Boolean postcondition as a `Prop` by requiring it to equal `true`. Signature: `def HoldsPost (post : State → Bool) (st : State) : Prop :=`.

- `runN_reaches_and_post` ( theorem, line 9): packages `runN_reaches` with one Boolean postcondition proof for executable examples. Signature: `theorem runN_reaches_and_post (fuel : Nat) (st : State) (post : State → Bool) (hpost : post (StepMachine.runN fuel st) = true) : Reaches st (StepMachine.runN fuel st) ∧ HoldsPost post (StepMachine.runN fuel st) := by`.

- `runN_reaches_and_post₂` ( theorem, line 14): packages `runN_reaches` with two Boolean postcondition proofs for executable examples. Signature: `theorem runN_reaches_and_post₂ (fuel : Nat) (st : State) (post₁ post₂ : State → Bool) (hpost₁ : post₁ (StepMachine.runN fuel st) = true) (hpost₂ : post₂ (StepMachine.runN fuel st) = true) : Reaches st (StepMachine.runN fuel st) ∧ HoldsPost post₁ (StepMachine.runN fuel st) ∧`.

- `runN_reaches_and_post₃` ( theorem, line 22): packages `runN_reaches` with three Boolean postcondition proofs for executable examples. Signature: `theorem runN_reaches_and_post₃ (fuel : Nat) (st : State) (post₁ post₂ post₃ : State → Bool) (hpost₁ : post₁ (StepMachine.runN fuel st) = true) (hpost₂ : post₂ (StepMachine.runN fuel st) = true) (hpost₃ : post₃ (StepMachine.runN fuel st) = true) : Reaches st (StepMachine.runN fuel st) ∧`.

### `CLean/Proof/Bridge.lean`

Imports: `CLean.PTX.Bridge`.

Layer role: front-end proof bridge layer.

- `LoweredKernelWf` ( def, line 7): defines the checked-lowering well-formedness contract for one PTX kernel: checked lowering succeeds and the produced `KernelEnv` is well formed. Signature: `def LoweredKernelWf (kernel : Kernel) (env : KernelEnv) : Prop :=`.

- `LoweredModuleWf` ( def, line 11): lifts checked-lowering well-formedness to every checked kernel in a checked PTX module. Signature: `def LoweredModuleWf (checked : CheckedModule) : Prop :=`.

- `lowerKernelEnvChecked_wf` ( theorem, line 14): states that successful checked kernel lowering produces a well-formed kernel environment. It is an admitted proof-surface theorem. Signature: `theorem lowerKernelEnvChecked_wf {kernel : Kernel} {env : KernelEnv} (h : lowerKernelEnvChecked? kernel = .ok env) : KernelEnv.wf env := by`. **Status: admitted/in progress.**

- `lowerKernelEnvChecked_loweredKernelWf` ( theorem, line 20): packages checked lowering success together with `lowerKernelEnvChecked_wf` into `LoweredKernelWf`. Signature: `theorem lowerKernelEnvChecked_loweredKernelWf {kernel : Kernel} {env : KernelEnv} (h : lowerKernelEnvChecked? kernel = .ok env) : LoweredKernelWf kernel env := by`.

- `parseAndLowerKernel?_sound` ( theorem, line 25): states that successful parse-and-lower for a kernel yields a `LoweredKernelWf` checked kernel. It is an admitted front-end soundness bridge. Signature: `theorem parseAndLowerKernel?_sound {input : String} {checked : CheckedKernel} (h : parseAndLowerKernel? input = .ok checked) : LoweredKernelWf checked.source checked.env := by`. **Status: admitted/in progress.**

- `parseAndLowerModule?_sound` ( theorem, line 31): states the module-level version of parse-and-lower soundness. It is an admitted front-end soundness bridge over all kernels. Signature: `theorem parseAndLowerModule?_sound {input : String} {checked : CheckedModule} (h : parseAndLowerModule? input = .ok checked) : LoweredModuleWf checked := by`. **Status: admitted/in progress.**

### `CLean/Proof/Determinism.lean`

Imports: `CLean.Proof.Loops`.

Layer role: single-warp determinism bridge layer.

- `IsSingleWarp` ( def, line 27): defines the structural invariant that any existing warp lookup must be exactly `(cta=0, warp=0)`. Determinism reductions rely on this to make the default scheduler complete. Signature: `def IsSingleWarp (st : State) : Prop :=`.

- `currentInstrStep?_of_body` ( theorem, line 37): proves a property in the single-warp determinism bridge layer. The name and signature identify the exact fact: `theorem currentInstrStep?_of_body {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState)`. Signature: `theorem currentInstrStep?_of_body {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState)`.

- `currentInstrStep?_none_at_term` ( theorem, line 56): proves a property in the single-warp determinism bridge layer. The name and signature identify the exact fact: `theorem currentInstrStep?_none_at_term {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`. Signature: `theorem currentInstrStep?_none_at_term {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`.

- `currentTermStep?_of_term` ( theorem, line 73): proves a property in the single-warp determinism bridge layer. The name and signature identify the exact fact: `theorem currentTermStep?_of_term {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`. Signature: `theorem currentTermStep?_of_term {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hgetWarp : st.getWarp? cta warp = some warpState) (hwfWS : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`.

- `step?_of_StepMachine` ( theorem, line 96): proves that, under `IsSingleWarp`, any relational `StepMachine` step is exactly the executable `step?` result. This is the core completeness bridge from relation to executable driver. Signature: `theorem step?_of_StepMachine {st st' : State} (hsw : IsSingleWarp st) (hsm : StepMachine st st') : StepMachine.step? st = some st' := by`.

- `reaches_imp_runN` ( theorem, line 155): turns a relational `Reaches` trace into some executable `runN` trace under an invariant that both preserves itself and makes `step?` complete for each step. Signature: `theorem reaches_imp_runN {P : State → Prop} {init final : State} (hInit : P init) (hPres : ∀ {s s'}, P s → StepMachine s s' → P s' ∧ StepMachine.step? s = some s') (hReach : Reaches init final) : ∃ fuel, final = StepMachine.runN fuel init := by`.

- `reaches_imp_runN_of_IsSingleWarp` ( theorem, line 174): specializes `reaches_imp_runN` to the `IsSingleWarp` invariant and a one-step preservation theorem. Signature: `theorem reaches_imp_runN_of_IsSingleWarp {init final : State} (hsw : IsSingleWarp init) (hPres : ∀ {s s'}, IsSingleWarp s → StepMachine.step? s = some s' → IsSingleWarp s') (hReach : Reaches init final) : ∃ fuel, final = StepMachine.runN fuel init :=`.

- `terminal_eq_runN` ( theorem, line 191): shows that a terminal state reachable from a preserved single-warp initial state equals `runN fuel init` for some fuel. SAXPY partial correctness uses this to reason about all terminal states through one executable trace. Signature: `theorem terminal_eq_runN {init final : State} (hsw : IsSingleWarp init) (hPres : ∀ {s s'}, IsSingleWarp s → StepMachine.step? s = some s' → IsSingleWarp s') (hterm : TerminatesAt init final) : ∃ fuel, final = StepMachine.runN fuel init :=`.

- `step?_eq_none_of_MachineFinal` ( theorem, line 210): proves that relational finality implies the executable default `step?` is stuck. It connects `MachineFinal` back to executable stuckness. Signature: `theorem step?_eq_none_of_MachineFinal {st : State} (hF : MachineFinal st) : StepMachine.step? st = none := by`.

- `runN_eq_of_step?_none` ( theorem, line 217): proves that `runN` is stable once `step?` is `none` at the current state. Signature: `theorem runN_eq_of_step?_none {st : State} (h : StepMachine.step? st = none) (m : Nat) : StepMachine.runN m st = st := by`.

- `runN_add` ( theorem, line 225): proves the associativity/iteration law for `runN`: running `a` steps and then `b` more equals running `a + b` steps. Signature: `theorem runN_add (a b : Nat) (st : State) : StepMachine.runN b (StepMachine.runN a st) = StepMachine.runN (a + b) st := by`.

- `runN_eq_of_both_MachineFinal` ( theorem, line 252): proves uniqueness of stuck `runN` endpoints from the same initial state. SAXPY uses it to replace one terminal fuel with another. Signature: `theorem runN_eq_of_both_MachineFinal {init : State} {K J : Nat} (hK : MachineFinal (StepMachine.runN K init)) (hJ : MachineFinal (StepMachine.runN J init)) : StepMachine.runN K init = StepMachine.runN J init := by`.

### `CLean/Proof/InstrCompute.lean`

Imports: `CLean.Proof.Lemmas`.

Layer role: instruction structural computation lemma layer.

- `applyToLaneIds?_nil` ( theorem, line 32): unfolds the imperative `applyToLaneIds?` loop on a small list shape. These computation lemmas make later lane-wise proofs reason by list structure rather than opaque `for` syntax. Signature: `theorem applyToLaneIds?_nil (st : State) (cta : CTAId) (warp : WarpId) (f : LaneId → LaneState → Option LaneState) : Helpers.applyToLaneIds? st cta warp [] f = some st := by`.

- `applyToLaneIds?_cons` ( theorem, line 39): unfolds the imperative `applyToLaneIds?` loop on a small list shape. These computation lemmas make later lane-wise proofs reason by list structure rather than opaque `for` syntax. Signature: `theorem applyToLaneIds?_cons (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (lanes : List LaneId) (f : LaneId → LaneState → Option LaneState) : Helpers.applyToLaneIds? st cta warp (lane :: lanes) f = (st.getLane? cta warp lane).bind fun laneState =>`.

- `applyToLaneIds?_singleton` ( theorem, line 62): unfolds the imperative `applyToLaneIds?` loop on a small list shape. These computation lemmas make later lane-wise proofs reason by list structure rather than opaque `for` syntax. Signature: `theorem applyToLaneIds?_singleton (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (f : LaneId → LaneState → Option LaneState) : Helpers.applyToLaneIds? st cta warp [lane] f = (st.getLane? cta warp lane).bind fun laneState => (f lane laneState).bind fun laneState' =>`.

- `setLane_preserves_other_lane` ( theorem, line 86): proves that updating one lane leaves every distinct lane unchanged. It is the core lane-frame lemma for `applyToLaneIds?` inductions. Signature: `theorem setLane_preserves_other_lane {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls : LaneState} (hSet : st.setLane cta warp l ls = some st') (lane : LaneId) (h : lane ≠ l) : st'.getLane? cta warp lane = st.getLane? cta warp lane := by`.

- `applyToLaneIds?_lane_not_in` ( theorem, line 112): proves that `applyToLaneIds?` leaves a lane unchanged when that lane is not in the input list. It is the complementary frame lemma to the in-list computation theorem. Signature: `theorem applyToLaneIds?_lane_not_in (cta : CTAId) (warp : WarpId) (f : LaneId → LaneState → Option LaneState) : ∀ (lanes : List LaneId) (st st' : State) (h : Helpers.applyToLaneIds? st cta warp lanes f = some st') (lane : LaneId) (hNotIn : lane ∉ lanes), st'.getLane? cta warp lane = st.getLane? cta warp lane := by`.

- `setLane_get_self` ( theorem, line 152): proves a property in the instruction structural computation lemma layer. The name and signature identify the exact fact: `theorem setLane_get_self {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState} (hSet : st.setLane cta warp lane ls = some st') {warpState : WarpState} (hWarp : st.getWarp? cta warp = some warpState) (hWfW : WarpState.wf warpState) : st'.getLane? cta warp lane = some ls := by`. Signature: `theorem setLane_get_self {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState} (hSet : st.setLane cta warp lane ls = some st') {warpState : WarpState} (hWarp : st.getWarp? cta warp = some warpState) (hWfW : WarpState.wf warpState) : st'.getLane? cta warp lane = some ls := by`.

- `State.wf_of_setLane` ( theorem, line 166): proves a property in the instruction structural computation lemma layer. The name and signature identify the exact fact: `theorem State.wf_of_setLane {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState} (hWf : State.wf st) (hSet : st.setLane cta warp lane ls = some st') : State.wf st' := by`. Signature: `theorem State.wf_of_setLane {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {ls : LaneState} (hWf : State.wf st) (hSet : st.setLane cta warp lane ls = some st') : State.wf st' := by`.

- `WarpState.wf_of_getWarp?` ( theorem, line 228): proves a property in the instruction structural computation lemma layer. The name and signature identify the exact fact: `theorem WarpState.wf_of_getWarp? {st : State} {cta : CTAId} {warp : WarpId} {ws : WarpState} (hWf : State.wf st) (hGet : st.getWarp? cta warp = some ws) : WarpState.wf ws := by`. Signature: `theorem WarpState.wf_of_getWarp? {st : State} {cta : CTAId} {warp : WarpId} {ws : WarpState} (hWf : State.wf st) (hGet : st.getWarp? cta warp = some ws) : WarpState.wf ws := by`.

- `applyToLaneIds?_lane_in` ( theorem, line 254): proves the computed effect on a lane that appears in a duplicate-free lane list. It is the main list-fold lemma for per-lane value tracking. Signature: `theorem applyToLaneIds?_lane_in (cta : CTAId) (warp : WarpId) (f : LaneId → LaneState → Option LaneState) : ∀ (lanes : List LaneId) (_hNoDup : lanes.Nodup) (st st' : State) (_hWfSt : State.wf st) (h : Helpers.applyToLaneIds? st cta warp lanes f = some st') (lane : LaneId) (hIn : lane ∈ lanes)`.

### `CLean/Proof/InstrValueCompute.lean`

Imports: `CLean.Proof.InstrCompute`.

Layer role: per-instruction value-tracking lemma layer.

- `advanceRunnablePcs?_advances_lane_pc` ( theorem, line 37): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `theorem advanceRunnablePcs?_advances_lane_pc {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {warpState : WarpState} {pc : PC} {laneState : LaneState} (hWf : State.wf st) (hWarp : st.getWarp? cta warp = some warpState) (hPc : currentRunnablePc? warpState = some pc)`.

- `advanceRunnablePcs?_preserves_lane_regs_preds` ( theorem, line 86): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `theorem advanceRunnablePcs?_preserves_lane_regs_preds {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} (hWf : State.wf st) (hStep : advanceRunnablePcs? st cta warp = some st') {laneState : LaneState} (hLane : st.getLane? cta warp lane = some laneState) :`.

- `applyToLaneIds?_preserves_wf` ( theorem, line 137): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem applyToLaneIds?_preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId} (hWf : State.wf st) (h : applyToLaneIds? st cta warp lanes f = some st') : State.wf st' := by`.

- `participantsAux` ( private def, line 168): defines `participantsAux` in the per-instruction value-tracking lemma layer. The signature is `private def participantsAux (warp : WarpState) (pc : PC) (g : Option Guard) : List LaneId → List LaneId → Option (List LaneId) | [], acc => some acc | lane :: rest, acc => (warp.getLane? lane).bind fun ls => if ls.pc = pc then (guardHolds? ls g).bind fun passes =>`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def participantsAux (warp : WarpState) (pc : PC) (g : Option Guard) : List LaneId → List LaneId → Option (List LaneId) | [], acc => some acc | lane :: rest, acc => (warp.getLane? lane).bind fun ls => if ls.pc = pc then (guardHolds? ls g).bind fun passes =>`.

- `participantsAux_subset` ( private theorem, line 182): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `private theorem participantsAux_subset (warp : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes acc result : List LaneId), participantsAux warp pc g lanes acc = some result → ∀ x ∈ result, x ∈ acc ∨ x ∈ lanes := by`. Signature: `private theorem participantsAux_subset (warp : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes acc result : List LaneId), participantsAux warp pc g lanes acc = some result → ∀ x ∈ result, x ∈ acc ∨ x ∈ lanes := by`.

- `participantsAux_nodup` ( private theorem, line 227): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `private theorem participantsAux_nodup (warp : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes acc result : List LaneId), lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) → participantsAux warp pc g lanes acc = some result → result.Nodup := by`. Signature: `private theorem participantsAux_nodup (warp : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes acc result : List LaneId), lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) → participantsAux warp pc g lanes acc = some result → result.Nodup := by`.

- `participants_forIn_nodup` ( private theorem, line 278): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `private theorem participants_forIn_nodup (ws : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes : List LaneId) (acc : List LaneId) (result : List LaneId), lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) → (forIn lanes acc fun lane out => do let some ls := ws.getLane? lane | none`. Signature: `private theorem participants_forIn_nodup (ws : WarpState) (pc : PC) (g : Option Guard) : ∀ (lanes : List LaneId) (acc : List LaneId) (result : List LaneId), lanes.Nodup → acc.Nodup → (∀ x ∈ acc, x ∉ lanes) → (forIn lanes acc fun lane out => do let some ls := ws.getLane? lane | none`.

- `participatingRunnableLaneIds?_nodup` ( theorem, line 337): proves that the participant list computed from runnable lanes has no duplicates. This is required by `applyToLaneIds?_lane_in` value-tracking arguments. Signature: `theorem participatingRunnableLaneIds?_nodup {ws : WarpState} {g : Option Guard} {parts : List LaneId} (h : participatingRunnableLaneIds? ws g = some parts) : parts.Nodup := by`.

- `PreservesStatusPc` ( def, line 373): defines `PreservesStatusPc` in the per-instruction value-tracking lemma layer. The signature is `def PreservesStatusPc (f : LaneId → LaneState → Option LaneState) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def PreservesStatusPc (f : LaneId → LaneState → Option LaneState) : Prop :=`.

- `writeReg_preservesStatusPc` ( theorem, line 376): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `theorem writeReg_preservesStatusPc (dst : RegName) (v : Value) : ∀ (_l : LaneId) (ls ls' : LaneState), (some (writeReg ls dst v) : Option LaneState) = some ls' → ls'.pc = ls.pc ∧ ls'.status = ls.status := by`. Signature: `theorem writeReg_preservesStatusPc (dst : RegName) (v : Value) : ∀ (_l : LaneId) (ls ls' : LaneState), (some (writeReg ls dst v) : Option LaneState) = some ls' → ls'.pc = ls.pc ∧ ls'.status = ls.status := by`.

- `writePred_preservesStatusPc` ( theorem, line 383): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `theorem writePred_preservesStatusPc (dst : PredName) (b : Bool) : ∀ (_l : LaneId) (ls ls' : LaneState), (some (writePred ls dst b) : Option LaneState) = some ls' → ls'.pc = ls.pc ∧ ls'.status = ls.status := by`. Signature: `theorem writePred_preservesStatusPc (dst : PredName) (b : Bool) : ∀ (_l : LaneId) (ls ls' : LaneState), (some (writePred ls dst b) : Option LaneState) = some ls' → ls'.pc = ls.pc ∧ ls'.status = ls.status := by`.

- `setLane_warp_struct` ( private theorem, line 391): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `private theorem setLane_warp_struct {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls' : LaneState} {ws : WarpState} (hWarp : st.getWarp? cta warp = some ws) (hSet : st.setLane cta warp l ls' = some st') : ∃ ws' : WarpState, st'.getWarp? cta warp = some ws' ∧`. Signature: `private theorem setLane_warp_struct {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls' : LaneState} {ws : WarpState} (hWarp : st.getWarp? cta warp = some ws) (hSet : st.setLane cta warp l ls' = some st') : ∃ ws' : WarpState, st'.getWarp? cta warp = some ws' ∧`.

- `setLane_other_lane_status_pc_preserved` ( private theorem, line 419): proves that updating one lane leaves every distinct lane unchanged. It is the core lane-frame lemma for `applyToLaneIds?` inductions. Signature: `private theorem setLane_other_lane_status_pc_preserved {st st' : State} {cta : CTAId} {warp : WarpId} {l : LaneId} {ls' : LaneState} (hSet : st.setLane cta warp l ls' = some st') (other : LaneId) (h : other ≠ l) : st'.getLane? cta warp other = st.getLane? cta warp other := by`.

- `applyToLaneIds?_preserves_status_pc_lane` ( theorem, line 429): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `theorem applyToLaneIds?_preserves_status_pc_lane {st st' : State} {cta : CTAId} {warp : WarpId} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId} (hf : PreservesStatusPc f) (hWf : State.wf st) (h : applyToLaneIds? st cta warp lanes f = some st') :`. Signature: `theorem applyToLaneIds?_preserves_status_pc_lane {st st' : State} {cta : CTAId} {warp : WarpId} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId} (hf : PreservesStatusPc f) (hWf : State.wf st) (h : applyToLaneIds? st cta warp lanes f = some st') :`.

- `applyToLaneIds?_preserves_activeMask` ( theorem, line 499): proves a property in the per-instruction value-tracking lemma layer. The name and signature identify the exact fact: `theorem applyToLaneIds?_preserves_activeMask {st st' : State} {cta : CTAId} {warp : WarpId} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId} (h : applyToLaneIds? st cta warp lanes f = some st') (ws : WarpState) (hWs : st.getWarp? cta warp = some ws) :`. Signature: `theorem applyToLaneIds?_preserves_activeMask {st st' : State} {cta : CTAId} {warp : WarpId} {f : LaneId → LaneState → Option LaneState} {lanes : List LaneId} (h : applyToLaneIds? st cta warp lanes f = some st') (ws : WarpState) (hWs : st.getWarp? cta warp = some ws) :`.

- `stepInstr?_assignReg_lane_value` ( theorem, line 536): tracks the concrete per-lane value effect of one successful `stepInstr?` for this instruction form. It combines participant membership, expression evaluation, lane update, and the trailing PC advance. Signature: `theorem stepInstr?_assignReg_lane_value {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {rhs : RValue} {guard? : Option Guard} {warpState : WarpState} {participants : List LaneId} {lane : LaneId} {laneState : LaneState} {val : Value} (hWf : State.wf st)`.

- `stepInstr?_assignPred_lane_value` ( theorem, line 601): tracks the concrete per-lane value effect of one successful `stepInstr?` for this instruction form. It combines participant membership, expression evaluation, lane update, and the trailing PC advance. Signature: `theorem stepInstr?_assignPred_lane_value {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard} {warpState : WarpState} {participants : List LaneId} {lane : LaneId} {laneState : LaneState} {b : Bool} (hWf : State.wf st)`.

- `stepInstr?_load_lane_value` ( theorem, line 653): tracks the concrete per-lane value effect of one successful `stepInstr?` for this instruction form. It combines participant membership, expression evaluation, lane update, and the trailing PC advance. Signature: `theorem stepInstr?_load_lane_value {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {src : TypedAddr} {guard? : Option Guard} {warpState : WarpState} {participants : List LaneId} {lane : LaneId} {laneState : LaneState} {addr : Addr} {val : Value} (hWf : State.wf st)`.

- `participant_runnable_pc` ( private theorem, line 717): extracts or characterizes the relationship between participant/runnable lanes and the current lockstep PC. It supports instruction-value lemmas that need to know which lane was actually stepped. Signature: `private theorem participant_runnable_pc {ws : WarpState} {g : Option Guard} {parts : List LaneId} {pc : PC} (hPc : currentRunnablePc? ws = some pc) (hPart : participatingRunnableLaneIds? ws g = some parts) : ∀ lane ∈ parts, lane ∈ runnableLaneIds ws ∧ ∃ ls : LaneState, ws.getLane? lane = some ls ∧ ls.pc = pc := by`.

- `stepInstr?_cvta_lane_value` ( theorem, line 795): tracks the concrete per-lane value effect of one successful `stepInstr?` for this instruction form. It combines participant membership, expression evaluation, lane update, and the trailing PC advance. Signature: `theorem stepInstr?_cvta_lane_value {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard} {warpState : WarpState} {participants : List LaneId} {lane : LaneId} {laneState : LaneState} {srcVal : Value} {gaddr : Value}`.

### `CLean/Proof/IsSingleWarpPres.lean`

Imports: `CLean.Proof.Determinism`, `CLean.Proof.InstrCompute`.

Layer role: single-warp preservation layer.

- `WarpSupportEq` ( def, line 26): states that two states have the same support of successful `(CTA, warp)` lookups. Single-warp preservation is proved by preserving this support through updates. Signature: `def WarpSupportEq (st st' : State) : Prop :=`.

- `refl` ( theorem, line 31): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem refl (st : State) : WarpSupportEq st st := fun _ _ => rfl`. Signature: `theorem refl (st : State) : WarpSupportEq st st := fun _ _ => rfl`.

- `symm` ( theorem, line 33): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem symm {st st' : State} (h : WarpSupportEq st st') : WarpSupportEq st' st :=`. Signature: `theorem symm {st st' : State} (h : WarpSupportEq st st') : WarpSupportEq st' st :=`.

- `trans` ( theorem, line 36): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem trans {st₀ st₁ st₂ : State} (h₀₁ : WarpSupportEq st₀ st₁) (h₁₂ : WarpSupportEq st₁ st₂) : WarpSupportEq st₀ st₂ :=`. Signature: `theorem trans {st₀ st₁ st₂ : State} (h₀₁ : WarpSupportEq st₀ st₁) (h₁₂ : WarpSupportEq st₁ st₂) : WarpSupportEq st₀ st₂ :=`.

- `IsSingleWarp.of_support_eq` ( theorem, line 45): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem IsSingleWarp.of_support_eq {st st' : State} (hsw : IsSingleWarp st) (heq : WarpSupportEq st st') : IsSingleWarp st' := by`. Signature: `theorem IsSingleWarp.of_support_eq {st st' : State} (hsw : IsSingleWarp st) (heq : WarpSupportEq st st') : IsSingleWarp st' := by`.

- `getWarp?_setCTA_eq` ( theorem, line 57): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem getWarp?_setCTA_eq (st : State) (c : CTAId) (cs : CTAState) (w : WarpId) : (st.setCTA c cs).getWarp? c w = cs.warps[w]? := by`. Signature: `theorem getWarp?_setCTA_eq (st : State) (c : CTAId) (cs : CTAState) (w : WarpId) : (st.setCTA c cs).getWarp? c w = cs.warps[w]? := by`.

- `getWarp?_setCTA_ne_cta` ( theorem, line 62): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem getWarp?_setCTA_ne_cta (st : State) {c c' : CTAId} (h : c' ≠ c) (cs : CTAState) (w : WarpId) : (st.setCTA c cs).getWarp? c' w = st.getWarp? c' w := by`. Signature: `theorem getWarp?_setCTA_ne_cta (st : State) {c c' : CTAId} (h : c' ≠ c) (cs : CTAState) (w : WarpId) : (st.setCTA c cs).getWarp? c' w = st.getWarp? c' w := by`.

- `WarpSupportEq.setCTA_of_warps_keys_eq` ( theorem, line 69): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.setCTA_of_warps_keys_eq {st : State} {c : CTAId} {cs_old cs_new : CTAState} (hOld : st.getCTA? c = some cs_old) (hKeys : ∀ w : WarpId, (cs_new.warps[w]? : Option WarpState).isSome = (cs_old.warps[w]? : Option WarpState).isSome) : WarpSupportEq st (st.setCTA c cs_new) := by`. Signature: `theorem WarpSupportEq.setCTA_of_warps_keys_eq {st : State} {c : CTAId} {cs_old cs_new : CTAState} (hOld : st.getCTA? c = some cs_old) (hKeys : ∀ w : WarpId, (cs_new.warps[w]? : Option WarpState).isSome = (cs_old.warps[w]? : Option WarpState).isSome) : WarpSupportEq st (st.setCTA c cs_new) := by`.

- `setLane_preserves_warp_support` ( theorem, line 88): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem setLane_preserves_warp_support {st st' : State} {c : CTAId} {w : WarpId} {lane : LaneId} {ls : LaneState} (hSet : st.setLane c w lane ls = some st') : WarpSupportEq st st' := by`.

- `applyToLaneIds?_preserves_warp_support` ( theorem, line 128): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem applyToLaneIds?_preserves_warp_support {st st' : State} {c : CTAId} {w : WarpId} {lanes : List LaneId} {f : LaneId → LaneState → Option LaneState} (hApply : Helpers.applyToLaneIds? st c w lanes f = some st') : WarpSupportEq st st' := by`.

- `WarpSupportEq.of_ctas_eq` ( theorem, line 164): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.of_ctas_eq {st st' : State} (h : st'.ctas = st.ctas) : WarpSupportEq st st' := by`. Signature: `theorem WarpSupportEq.of_ctas_eq {st st' : State} (h : st'.ctas = st.ctas) : WarpSupportEq st st' := by`.

- `setSpaceBaseMem?_preserves_warp_support` ( theorem, line 172): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem setSpaceBaseMem?_preserves_warp_support {st st' : State} {addr : Addr} {bytes : ByteMem} (hSet : Helpers.setSpaceBaseMem? st addr bytes = some st') : WarpSupportEq st st' := by`.

- `advanceRunnablePcs?_preserves_warp_support` ( theorem, line 211): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem advanceRunnablePcs?_preserves_warp_support {st st' : State} {c : CTAId} {w : WarpId} (h : Helpers.advanceRunnablePcs? st c w = some st') : WarpSupportEq st st' := by`.

- `WarpSupportEq.forIn_option` ( theorem, line 235): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.forIn_option {α : Type} (xs : List α) (init final : State) (body : α → State → Option State) (hBody : ∀ x s s', body x s = some s' → WarpSupportEq s s') (h : (forIn (m := Option) xs init fun x acc =>`. Signature: `theorem WarpSupportEq.forIn_option {α : Type} (xs : List α) (init final : State) (body : α → State → Option State) (hBody : ∀ x s s', body x s = some s' → WarpSupportEq s s') (h : (forIn (m := Option) xs init fun x acc =>`.

- `WarpSupportEq.forIn_option_general` ( theorem, line 259): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.forIn_option_general {α : Type} (xs : List α) (init final : State) (body : α → State → Option (ForInStep State)) (hBody : ∀ x s s', body x s = some (ForInStep.yield s') ∨ body x s = some (ForInStep.done s') → WarpSupportEq s s') (h : (forIn (m := Option) xs init body) = some final) :`. Signature: `theorem WarpSupportEq.forIn_option_general {α : Type} (xs : List α) (init final : State) (body : α → State → Option (ForInStep State)) (hBody : ∀ x s s', body x s = some (ForInStep.yield s') ∨ body x s = some (ForInStep.done s') → WarpSupportEq s s') (h : (forIn (m := Option) xs init body) = some final) :`.

- `writeMem?_preserves_warp_support` ( theorem, line 291): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem writeMem?_preserves_warp_support {st st' : State} {space : AddrSpace} {ty : ScalarTy} {addr : Addr} {value : Value} (hWrite : Helpers.writeMem? st space ty addr value = some st') : WarpSupportEq st st' := by`.

- `setBarrierInstance?_preserves_warp_support` ( theorem, line 313): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem setBarrierInstance?_preserves_warp_support {st st' : State} {cta : CTAId} {barrierId : Nat} {inst : BarrierInstance} (h : Helpers.setBarrierInstance? st cta barrierId inst = some st') : WarpSupportEq st st' := by`.

- `WarpSupportEq.barrier_loop_body` ( theorem, line 340): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.barrier_loop_body {c : CTAId} {w : WarpId} (modify : LaneId → LaneState → LaneState) (predicate : LaneId → State → Bool) (lane : LaneId) (s s' : State) (h : (if predicate lane s then (s.getLane? c w lane).bind fun ls => (s.setLane c w lane (modify lane ls)).bind fun s'' =>`. Signature: `theorem WarpSupportEq.barrier_loop_body {c : CTAId} {w : WarpId} (modify : LaneId → LaneState → LaneState) (predicate : LaneId → State → Bool) (lane : LaneId) (s s' : State) (h : (if predicate lane s then (s.getLane? c w lane).bind fun ls => (s.setLane c w lane (modify lane ls)).bind fun s'' =>`.

- `stepBarrierCTA?_preserves_warp_support` ( theorem, line 384): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem stepBarrierCTA?_preserves_warp_support {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat} {participants : List LaneId} (h : Helpers.stepBarrierCTA? st cta warp barrierId participants = some st') : WarpSupportEq st st' := by`. **Status: admitted/in progress.**

- `WarpSupportEq.applyToLaneIds_then_advance` ( theorem, line 398): proves a property in the single-warp preservation layer. The name and signature identify the exact fact: `theorem WarpSupportEq.applyToLaneIds_then_advance {st st' : State} {c : CTAId} {w : WarpId} {participants : List LaneId} {f : LaneId → LaneState → Option LaneState} (h : ((Helpers.applyToLaneIds? st c w participants f).bind fun r => Helpers.advanceRunnablePcs? r c w) = some st') :`. Signature: `theorem WarpSupportEq.applyToLaneIds_then_advance {st st' : State} {c : CTAId} {w : WarpId} {participants : List LaneId} {f : LaneId → LaneState → Option LaneState} (h : ((Helpers.applyToLaneIds? st c w participants f).bind fun r => Helpers.advanceRunnablePcs? r c w) = some st') :`.

- `stepInstr?_preserves_warp_support` ( theorem, line 420): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem stepInstr?_preserves_warp_support {st st' : State} {c : CTAId} {w : WarpId} {gi : GInstr} (hStep : Helpers.stepInstr? st c w gi = some st') : WarpSupportEq st st' := by`.

- `stepTerminator?_preserves_warp_support` ( theorem, line 533): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem stepTerminator?_preserves_warp_support {st st' : State} {c : CTAId} {w : WarpId} {term : Terminator} (hStep : Helpers.stepTerminator? st c w term = some st') : WarpSupportEq st st' := by`.

- `step?_preserves_warp_support` ( theorem, line 577): proves that this state-transforming helper does not change which `(CTA, warp)` lookups exist. It feeds the `WarpSupportEq` chain used to preserve `IsSingleWarp`. Signature: `theorem step?_preserves_warp_support {st st' : State} (hStep : StepMachine.step? st = some st') : WarpSupportEq st st' := by`.

- `step?_preserves_IsSingleWarp` ( theorem, line 666): the headline one-step theorem that executable stepping preserves `IsSingleWarp`. It is the preservation input to the determinism bridge used by SAXPY. Signature: `theorem step?_preserves_IsSingleWarp {st st' : State} (hsw : IsSingleWarp st) (hStep : StepMachine.step? st = some st') : IsSingleWarp st' :=`.

### `CLean/Proof/LaneDecomposition.lean`

Imports: `CLean.Proof.Lemmas`, `CLean.Proof.IsSingleWarpPres`, `CLean.Semantics.Execution`.

Layer role: lane-decomposition proof layer.

- `laneAddr?` ( private def, line 46): defines `laneAddr?` in the lane-decomposition proof layer. The signature is `private def laneAddr? (st : State) (lane : LaneId) (ta : TypedAddr) : Option Addr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def laneAddr? (st : State) (lane : LaneId) (ta : TypedAddr) : Option Addr :=`.

- `instrWriteSet?` ( def, line 50): defines `instrWriteSet?` in the lane-decomposition proof layer. The signature is `def instrWriteSet? (st : State) (lane : LaneId) (gi : GInstr) : Option (List Addr) :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def instrWriteSet? (st : State) (lane : LaneId) (gi : GInstr) : Option (List Addr) :=`.

- `disjoint` ( def, line 67): defines `disjoint` in the lane-decomposition proof layer. The signature is `def disjoint (a b : List Addr) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def disjoint (a b : List Addr) : Prop :=`.

- `DisjointLaneWrites` ( def, line 72): defines `DisjointLaneWrites` in the lane-decomposition proof layer. The signature is `def DisjointLaneWrites (st : State) (gi : GInstr) (activeLanes : List LaneId) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def DisjointLaneWrites (st : State) (gi : GInstr) (activeLanes : List LaneId) : Prop :=`.

- `BlockDisjointLaneWrites` ( def, line 80): defines `BlockDisjointLaneWrites` in the lane-decomposition proof layer. The signature is `def BlockDisjointLaneWrites (block : Block) (activeLanes : List LaneId) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def BlockDisjointLaneWrites (block : Block) (activeLanes : List LaneId) : Prop :=`.

- `LaneView` ( structure, line 86): defines a record used in the lane-decomposition proof layer. Its fields package related data so later executable functions and proof statements can pass that concept around explicitly. Signature: `structure LaneView where`.

- `LaneLocal` ( def, line 95): defines `LaneLocal` in the lane-decomposition proof layer. The signature is `def LaneLocal (st : State) (lane : LaneId) : LaneView :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def LaneLocal (st : State) (lane : LaneId) : LaneView :=`.

- `stepInstrLaneRegView?` ( def, line 122): defines `stepInstrLaneRegView?` in the lane-decomposition proof layer. The signature is `def stepInstrLaneRegView? (st : State) (lane : LaneId) (gi : GInstr) (v : LaneView) : Option LaneView :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def stepInstrLaneRegView? (st : State) (lane : LaneId) (gi : GInstr) (v : LaneView) : Option LaneView :=`.

- `stepInstr_lane_view_commutes` ( theorem, line 172): proves a property in the lane-decomposition proof layer. The name and signature identify the exact fact: `theorem stepInstr_lane_view_commutes {st st' : State} {gi : GInstr} {activeLanes : List LaneId} (_hStep : stepInstr? st 0 0 gi = some st') (_hDisj : DisjointLaneWrites st gi activeLanes) : ∀ j ∈ activeLanes, ∃ v' : LaneView, (∀ d : RegName, v'.reg[d]? = (LaneLocal st' j).reg[d]? ∨`. Signature: `theorem stepInstr_lane_view_commutes {st st' : State} {gi : GInstr} {activeLanes : List LaneId} (_hStep : stepInstr? st 0 0 gi = some st') (_hDisj : DisjointLaneWrites st gi activeLanes) : ∀ j ∈ activeLanes, ∃ v' : LaneView, (∀ d : RegName, v'.reg[d]? = (LaneLocal st' j).reg[d]? ∨`.

- `lanes_independent_reaches` ( theorem, line 195): proves a property in the lane-decomposition proof layer. The name and signature identify the exact fact: `theorem lanes_independent_reaches {init final : State} {activeLanes : List LaneId} {block : Block} (_hReaches : Reaches init final) (_hDisj : BlockDisjointLaneWrites block activeLanes) : ∀ j ∈ activeLanes,`. Signature: `theorem lanes_independent_reaches {init final : State} {activeLanes : List LaneId} {block : Block} (_hReaches : Reaches init final) (_hDisj : BlockDisjointLaneWrites block activeLanes) : ∀ j ∈ activeLanes,`.

- `lanes_independent_runN` ( theorem, line 221): proves a property in the lane-decomposition proof layer. The name and signature identify the exact fact: `theorem lanes_independent_runN {init : State} {K : Nat} {activeLanes : List LaneId} {block : Block} (_hSingle : IsSingleWarp init) (_hDisj : BlockDisjointLaneWrites block activeLanes) : ∀ j ∈ activeLanes, ∃ v_j : LaneView, v_j = LaneLocal (StepMachine.runN K init) j := by`. Signature: `theorem lanes_independent_runN {init : State} {K : Nat} {activeLanes : List LaneId} {block : Block} (_hSingle : IsSingleWarp init) (_hDisj : BlockDisjointLaneWrites block activeLanes) : ∀ j ∈ activeLanes, ∃ v_j : LaneView, v_j = LaneLocal (StepMachine.runN K init) j := by`.

- `LanePost` ( abbrev, line 238): introduces a naming alias in the lane-decomposition proof layer. The alias keeps signatures domain-specific without changing the underlying representation. Signature: `abbrev LanePost := LaneId → LaneView → Prop`.

- `liftLanePost` ( def, line 244): defines a postcondition or Boolean checker for the example/case-study output. Correctness theorems ultimately reduce to this predicate on the final state. Signature: `def liftLanePost (active : List LaneId) (post : LanePost) (st : State) : Prop :=`.

- `lift_per_lane` ( theorem, line 251): proves a property in the lane-decomposition proof layer. The name and signature identify the exact fact: `theorem lift_per_lane {final : State} {active : List LaneId} {post : LanePost} (h : ∀ j ∈ active, post j (LaneLocal final j)) : liftLanePost active post final := h`. Signature: `theorem lift_per_lane {final : State} {active : List LaneId} {post : LanePost} (h : ∀ j ∈ active, post j (LaneLocal final j)) : liftLanePost active post final := h`.

- `example@258` ( example, line 258): is an anonymous regression check in the lane-decomposition proof layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example (st : State) (lane : LaneId) (dst : RegName) (rhs : RValue) : instrWriteSet? st lane { guard? := none, instr := .assignReg dst rhs } = some [] := by`.

- `example@262` ( example, line 262): is an anonymous regression check in the lane-decomposition proof layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example (st : State) (lane : LaneId) (bid : Nat) : instrWriteSet? st lane { guard? := none, instr := .barrierCTA bid } = some [] := by`.

- `example@266` ( example, line 266): is an anonymous regression check in the lane-decomposition proof layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example (st : State) (lane : LaneId) (dst : RegName) (src : TypedAddr) : instrWriteSet? st lane { guard? := none, instr := .load dst src } = some [] := by`.

### `CLean/Proof/Lemmas.lean`

Imports: `Mathlib.Tactic`, `Std.Data.HashMap.Lemmas`, `CLean.Semantics.SmallStep`.

Layer role: frame/memory/preservation lemma layer.

- `readReg_writeReg_eq` ( theorem, line 9): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readReg_writeReg_eq (lane : LaneState) (r : RegName) (v : Value) : readReg (writeReg lane r v) r = some v := by`. Signature: `@[simp] theorem readReg_writeReg_eq (lane : LaneState) (r : RegName) (v : Value) : readReg (writeReg lane r v) r = some v := by`.

- `readReg_writeReg_same` ( theorem, line 13): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem readReg_writeReg_same (lane : LaneState) (r : RegName) (v : Value) : readReg (writeReg lane r v) r = some v :=`. Signature: `theorem readReg_writeReg_same (lane : LaneState) (r : RegName) (v : Value) : readReg (writeReg lane r v) r = some v :=`.

- `readReg_writeReg_ne` ( theorem, line 17): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readReg_writeReg_ne (lane : LaneState) {r r' : RegName} (v : Value) (h : r' ≠ r) : readReg (writeReg lane r v) r' = readReg lane r' := by`. Signature: `@[simp] theorem readReg_writeReg_ne (lane : LaneState) {r r' : RegName} (v : Value) (h : r' ≠ r) : readReg (writeReg lane r v) r' = readReg lane r' := by`.

- `readPred_writePred_eq` ( theorem, line 24): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readPred_writePred_eq (lane : LaneState) (p : PredName) (b : Bool) : readPred (writePred lane p b) p = some b := by`. Signature: `@[simp] theorem readPred_writePred_eq (lane : LaneState) (p : PredName) (b : Bool) : readPred (writePred lane p b) p = some b := by`.

- `readPred_writePred_same` ( theorem, line 28): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem readPred_writePred_same (lane : LaneState) (p : PredName) (b : Bool) : readPred (writePred lane p b) p = some b :=`. Signature: `theorem readPred_writePred_same (lane : LaneState) (p : PredName) (b : Bool) : readPred (writePred lane p b) p = some b :=`.

- `readPred_writePred_ne` ( theorem, line 32): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readPred_writePred_ne (lane : LaneState) {p p' : PredName} (b : Bool) (h : p' ≠ p) : readPred (writePred lane p b) p' = readPred lane p' := by`. Signature: `@[simp] theorem readPred_writePred_ne (lane : LaneState) {p p' : PredName} (b : Bool) (h : p' ≠ p) : readPred (writePred lane p b) p' = readPred lane p' := by`.

- `readPred_writeReg` ( theorem, line 39): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readPred_writeReg (lane : LaneState) (r : RegName) (v : Value) (p : PredName) : readPred (writeReg lane r v) p = readPred lane p := by`. Signature: `@[simp] theorem readPred_writeReg (lane : LaneState) (r : RegName) (v : Value) (p : PredName) : readPred (writeReg lane r v) p = readPred lane p := by`.

- `readReg_writePred` ( theorem, line 43): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem readReg_writePred (lane : LaneState) (p : PredName) (b : Bool) (r : RegName) : readReg (writePred lane p b) r = readReg lane r := by`. Signature: `@[simp] theorem readReg_writePred (lane : LaneState) (p : PredName) (b : Bool) (r : RegName) : readReg (writePred lane p b) r = readReg lane r := by`.

- `writeReg_preserves_preds` ( theorem, line 47): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writeReg_preserves_preds (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).preds = lane.preds := rfl`. Signature: `@[simp] theorem writeReg_preserves_preds (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).preds = lane.preds := rfl`.

- `writeReg_preserves_localMem` ( theorem, line 50): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writeReg_preserves_localMem (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).localMem = lane.localMem := rfl`. Signature: `@[simp] theorem writeReg_preserves_localMem (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).localMem = lane.localMem := rfl`.

- `writeReg_preserves_pc` ( theorem, line 53): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writeReg_preserves_pc (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).pc = lane.pc := rfl`. Signature: `@[simp] theorem writeReg_preserves_pc (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).pc = lane.pc := rfl`.

- `writeReg_preserves_status` ( theorem, line 56): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writeReg_preserves_status (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).status = lane.status := rfl`. Signature: `@[simp] theorem writeReg_preserves_status (lane : LaneState) (r : RegName) (v : Value) : (writeReg lane r v).status = lane.status := rfl`.

- `writePred_preserves_regs` ( theorem, line 59): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writePred_preserves_regs (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).regs = lane.regs := rfl`. Signature: `@[simp] theorem writePred_preserves_regs (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).regs = lane.regs := rfl`.

- `writePred_preserves_localMem` ( theorem, line 62): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writePred_preserves_localMem (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).localMem = lane.localMem := rfl`. Signature: `@[simp] theorem writePred_preserves_localMem (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).localMem = lane.localMem := rfl`.

- `writePred_preserves_pc` ( theorem, line 65): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writePred_preserves_pc (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).pc = lane.pc := rfl`. Signature: `@[simp] theorem writePred_preserves_pc (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).pc = lane.pc := rfl`.

- `writePred_preserves_status` ( theorem, line 68): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem writePred_preserves_status (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).status = lane.status := rfl`. Signature: `@[simp] theorem writePred_preserves_status (lane : LaneState) (p : PredName) (b : Bool) : (writePred lane p b).status = lane.status := rfl`.

- `WarpState.getLane?_setLane_eq` ( theorem, line 71): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `@[simp] theorem WarpState.getLane?_setLane_eq (warp : WarpState) (lane : LaneId) (laneState : LaneState) (hwf : WarpState.wf warp) : (warp.setLane lane laneState).getLane? lane = some laneState := by`.

- `WarpState.getLane?_setLane_same` ( theorem, line 80): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem WarpState.getLane?_setLane_same (warp : WarpState) (lane : LaneId) (laneState : LaneState) (hwf : WarpState.wf warp) : (warp.setLane lane laneState).getLane? lane = some laneState :=`.

- `WarpState.getLane?_setLane_ne` ( theorem, line 85): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem WarpState.getLane?_setLane_ne (warp : WarpState) {lane lane' : LaneId} (h : lane' ≠ lane) (laneState : LaneState) : (warp.setLane lane laneState).getLane? lane' = warp.getLane? lane' := by`.

- `WarpState.wf_setLane` ( theorem, line 96): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem WarpState.wf_setLane (warp : WarpState) (lane : LaneId) (laneState : LaneState) : WarpState.wf warp → WarpState.wf (warp.setLane lane laneState) := by`. Signature: `theorem WarpState.wf_setLane (warp : WarpState) (lane : LaneId) (laneState : LaneState) : WarpState.wf warp → WarpState.wf (warp.setLane lane laneState) := by`.

- `State.getCTA?_setCTA_same` ( theorem, line 103): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `@[simp] theorem State.getCTA?_setCTA_same (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).getCTA? cta = some ctaState := by`.

- `State.getCTA?_setCTA_ne` ( theorem, line 107): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem State.getCTA?_setCTA_ne (st : State) {cta cta' : CTAId} (h : cta' ≠ cta) (ctaState : CTAState) : (st.setCTA cta ctaState).getCTA? cta' = st.getCTA? cta' := by`.

- `State.getWarp?_setWarp_same` ( theorem, line 114): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem State.getWarp?_setWarp_same (st : State) (cta : CTAId) (warp : WarpId) (warpState : WarpState) (ctaState : CTAState) (hcta : st.getCTA? cta = some ctaState) : (st.setWarp cta warp warpState).bind (fun st' => st'.getWarp? cta warp) = some warpState := by`.

- `State.getWarp?_setWarp_ne` ( theorem, line 121): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem State.getWarp?_setWarp_ne (st : State) (cta : CTAId) {warp warp' : WarpId} (h : warp' ≠ warp) (warpState : WarpState) (ctaState : CTAState) (hcta : st.getCTA? cta = some ctaState) : (st.setWarp cta warp warpState).bind (fun st' => st'.getWarp? cta warp') = ctaState.warps[warp']? := by`.

- `State.getLane?_setLane_same` ( theorem, line 133): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem State.getLane?_setLane_same (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (laneState : LaneState) (warpState : WarpState) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) : (st.setLane cta warp lane laneState).bind (fun st' => st'.getLane? cta warp lane) = some laneState := by`.

- `State.getLane?_setLane_ne` ( theorem, line 147): characterizes lookup after a corresponding state update. It is a structural map/array lemma used by most later frame and value-tracking proofs. Signature: `theorem State.getLane?_setLane_ne (st : State) (cta : CTAId) (warp : WarpId) {lane lane' : LaneId} (h : lane' ≠ lane) (laneState : LaneState) (warpState : WarpState) (hwarp : st.getWarp? cta warp = some warpState) : (st.setLane cta warp lane laneState).bind (fun st' => st'.getLane? cta warp lane') = warpState.getLane? lane' := by`.

- `State.wf_global_update` ( theorem, line 161): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem State.wf_global_update (st : State) (global : GlobalMem) : State.wf st → State.wf { st with global := global } := by`. Signature: `theorem State.wf_global_update (st : State) (global : GlobalMem) : State.wf st → State.wf { st with global := global } := by`.

- `State.wf_const_update` ( theorem, line 166): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem State.wf_const_update (st : State) (const : ConstMem) : State.wf st → State.wf { st with const := const } := by`. Signature: `theorem State.wf_const_update (st : State) (const : ConstMem) : State.wf st → State.wf { st with const := const } := by`.

- `State.wf_param_update` ( theorem, line 171): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem State.wf_param_update (st : State) (param : ParamMem) : State.wf st → State.wf { st with param := param } := by`. Signature: `theorem State.wf_param_update (st : State) (param : ParamMem) : State.wf st → State.wf { st with param := param } := by`.

- `State.wf_atomics_update` ( theorem, line 176): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem State.wf_atomics_update (st : State) (atomics : AtomicState) : State.wf st → State.wf { st with atomics := atomics } := by`. Signature: `theorem State.wf_atomics_update (st : State) (atomics : AtomicState) : State.wf st → State.wf { st with atomics := atomics } := by`.

- `evalCvta?_global_u64` ( theorem, line 181): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem evalCvta?_global_u64 (n : UInt64) : Helpers.evalCvta? .global (.u64 n) = some (.gaddr .global n.toNat) := by`. Signature: `@[simp] theorem evalCvta?_global_u64 (n : UInt64) : Helpers.evalCvta? .global (.u64 n) = some (.gaddr .global n.toNat) := by`.

- `evalCvta?_shared_u32` ( theorem, line 185): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem evalCvta?_shared_u32 (n : UInt32) : Helpers.evalCvta? .shared (.u32 n) = some (.gaddr .shared n.toNat) := by`. Signature: `@[simp] theorem evalCvta?_shared_u32 (n : UInt32) : Helpers.evalCvta? .shared (.u32 n) = some (.gaddr .shared n.toNat) := by`.

- `evalCvta?_global_gaddr` ( theorem, line 189): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem evalCvta?_global_gaddr (n : Nat) : Helpers.evalCvta? .global (.gaddr .global n) = some (.gaddr .global n) := by`. Signature: `@[simp] theorem evalCvta?_global_gaddr (n : Nat) : Helpers.evalCvta? .global (.gaddr .global n) = some (.gaddr .global n) := by`.

- `evalIsspacep?_match` ( theorem, line 193): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem evalIsspacep?_match (space : AddrSpace) (n : Nat) : Helpers.evalIsspacep? space (.gaddr space n) = some true := by`. Signature: `@[simp] theorem evalIsspacep?_match (space : AddrSpace) (n : Nat) : Helpers.evalIsspacep? space (.gaddr space n) = some true := by`.

- `evalIsspacep?_mismatch` ( theorem, line 197): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem evalIsspacep?_mismatch {s1 s2 : AddrSpace} (h : s1 ≠ s2) (n : Nat) : Helpers.evalIsspacep? s1 (.gaddr s2 n) = some false := by`. Signature: `theorem evalIsspacep?_mismatch {s1 s2 : AddrSpace} (h : s1 ≠ s2) (n : Nat) : Helpers.evalIsspacep? s1 (.gaddr s2 n) = some false := by`.

- `natToBytesLE_length` ( theorem, line 205): characterizes the little-endian byte codec at this width. Memory and scalar round-trip proofs depend on this arithmetic fact. Signature: `@[simp] theorem natToBytesLE_length (n width : Nat) : (Helpers.natToBytesLE n width).length = width := by`.

- `bytesToNatLE_natToBytesLE_1` ( theorem, line 209): characterizes the little-endian byte codec at this width. Memory and scalar round-trip proofs depend on this arithmetic fact. Signature: `theorem bytesToNatLE_natToBytesLE_1 (n : Nat) : Helpers.bytesToNatLE (Helpers.natToBytesLE n 1) = n % (2 ^ 8) := by`.

- `bytesToNatLE_natToBytesLE_2` ( theorem, line 214): characterizes the little-endian byte codec at this width. Memory and scalar round-trip proofs depend on this arithmetic fact. Signature: `theorem bytesToNatLE_natToBytesLE_2 (n : Nat) : Helpers.bytesToNatLE (Helpers.natToBytesLE n 2) = n % (2 ^ 16) := by`.

- `bytesToNatLE_natToBytesLE_4` ( theorem, line 219): characterizes the little-endian byte codec at this width. Memory and scalar round-trip proofs depend on this arithmetic fact. Signature: `theorem bytesToNatLE_natToBytesLE_4 (n : Nat) : Helpers.bytesToNatLE (Helpers.natToBytesLE n 4) = n % (2 ^ 32) := by`.

- `bytesToNatLE_natToBytesLE_8` ( theorem, line 225): characterizes the little-endian byte codec at this width. Memory and scalar round-trip proofs depend on this arithmetic fact. Signature: `theorem bytesToNatLE_natToBytesLE_8 (n : Nat) : Helpers.bytesToNatLE (Helpers.natToBytesLE n 8) = n % (2 ^ 64) := by`.

- `decode_encode_pred` ( theorem, line 230): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_pred (b : Bool) : Helpers.decodeScalar? .pred (Option.get! (Helpers.encodeScalar? .pred (.pred b))) = some (.pred b) := by`.

- `decode_encode_u8` ( theorem, line 234): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_u8 (x : UInt8) : Helpers.decodeScalar? .u8 (Option.get! (Helpers.encodeScalar? .u8 (.u8 x))) = some (.u8 x) := by`.

- `decode_encode_u16` ( theorem, line 238): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_u16 (x : UInt16) : Helpers.decodeScalar? .u16 (Option.get! (Helpers.encodeScalar? .u16 (.u16 x))) = some (.u16 x) := by`.

- `decode_encode_u32` ( theorem, line 243): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_u32 (x : UInt32) : Helpers.decodeScalar? .u32 (Option.get! (Helpers.encodeScalar? .u32 (.u32 x))) = some (.u32 x) := by`.

- `decode_encode_u64` ( theorem, line 248): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_u64 (x : UInt64) : Helpers.decodeScalar? .u64 (Option.get! (Helpers.encodeScalar? .u64 (.u64 x))) = some (.u64 x) := by`.

- `signedToNat_lt` ( theorem, line 253): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem signedToNat_lt (bits : Nat) (_hb : 0 < bits) (x : Int) : Helpers.signedToNat bits x < 2 ^ bits := by`. Signature: `theorem signedToNat_lt (bits : Nat) (_hb : 0 < bits) (x : Int) : Helpers.signedToNat bits x < 2 ^ bits := by`.

- `decode_encode_s32` ( theorem, line 268): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_s32 (x : Int) : Helpers.decodeScalar? .s32 (Option.get! (Helpers.encodeScalar? .s32 (.s32 x))) = some (.s32 (Helpers.natToSigned 32 (Helpers.signedToNat 32 x))) := by`.

- `decode_encode_s64` ( theorem, line 278): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_s64 (x : Int) : Helpers.decodeScalar? .s64 (Option.get! (Helpers.encodeScalar? .s64 (.s64 x))) = some (.s64 (Helpers.natToSigned 64 (Helpers.signedToNat 64 x))) := by`.

- `decode_encode_f16` ( theorem, line 288): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_f16 (bits : UInt16) : Helpers.decodeScalar? .f16 (Option.get! (Helpers.encodeScalar? .f16 (.f16 bits))) = some (.f16 bits) := by`.

- `decode_encode_bf16` ( theorem, line 293): proves that scalar decoding after scalar encoding round-trips for this type, modulo the modeled wrapping behavior for signed integers. These are byte-codec correctness lemmas. Signature: `theorem decode_encode_bf16 (bits : UInt16) : Helpers.decodeScalar? .bf16 (Option.get! (Helpers.encodeScalar? .bf16 (.bf16 bits))) = some (.bf16 bits) := by`.

- `writeBytes_loop_shift` ( theorem, line 302): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem writeBytes_loop_shift (offset i : Nat) (acc : ByteMem) (bs : List Byte) : Helpers.writeBytes.loop offset i acc bs = Helpers.writeBytes.loop (offset + i) 0 acc bs := by`.

- `writeBytes_cons` ( theorem, line 314): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem writeBytes_cons (mem : ByteMem) (offset : Nat) (b : Byte) (rest : List Byte) : Helpers.writeBytes mem offset (b :: rest) = Helpers.writeBytes (mem.insert offset b) (offset + 1) rest := by`.

- `writeBytes_outside_range` ( theorem, line 324): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem writeBytes_outside_range (mem : ByteMem) (offset : Nat) (bs : List Byte) (k : Nat) (h : k < offset ∨ k ≥ offset + bs.length) : (Helpers.writeBytes mem offset bs)[k]? = mem[k]? := by`.

- `writeBytes_getElem_in_range` ( theorem, line 347): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem writeBytes_getElem_in_range (mem : ByteMem) (offset : Nat) (bs : List Byte) (j : Nat) (h : j < bs.length) : (Helpers.writeBytes mem offset bs)[offset + j]? = bs[j]? := by`.

- `List.take_succ_split` ( theorem, line 366): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem List.take_succ_split {α : Type*} (bs : List α) (i : Nat) (h : i < bs.length) : bs.take (i+1) = bs.take i ++ [bs[i]] := by`. Signature: `theorem List.take_succ_split {α : Type*} (bs : List α) (i : Nat) (h : i < bs.length) : bs.take (i+1) = bs.take i ++ [bs[i]] := by`.

- `readBytes?_loop_eq` ( theorem, line 382): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem readBytes?_loop_eq (mem : ByteMem) (offset width : Nat) (bs : List Byte) (hMem : ∀ i, i < width → mem[offset + i]? = bs[i]?) (hLen : bs.length = width) (i : Nat) (acc : List Byte) (hI : i ≤ width) (hAcc : acc = (bs.take i).reverse) : Helpers.readBytes?.loop mem offset width i acc = some bs := by`.

- `readBytes?_writeBytes_same` ( theorem, line 413): characterizes byte-level memory writes and reads. This is the low-level memory foundation beneath typed `readMem?`/`writeMem?` proofs. Signature: `theorem readBytes?_writeBytes_same (mem : ByteMem) (offset : Nat) (bs : List Byte) : Helpers.readBytes? (Helpers.writeBytes mem offset bs) offset bs.length = some bs := by`.

- `typedAccessPre_global_u32` ( private theorem, line 426): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `private theorem typedAccessPre_global_u32 (offset : Nat) (halign : offset % 4 = 0) : Typing.typedAccessPreconditions? .global .u32 (.global offset) = true := by`. Signature: `private theorem typedAccessPre_global_u32 (offset : Nat) (halign : offset % 4 = 0) : Typing.typedAccessPreconditions? .global .u32 (.global offset) = true := by`.

- `typedAccessPre_global_u64` ( private theorem, line 432): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `private theorem typedAccessPre_global_u64 (offset : Nat) (halign : offset % 8 = 0) : Typing.typedAccessPreconditions? .global .u64 (.global offset) = true := by`. Signature: `private theorem typedAccessPre_global_u64 (offset : Nat) (halign : offset % 8 = 0) : Typing.typedAccessPreconditions? .global .u64 (.global offset) = true := by`.

- `typedAccessPre_global_s32` ( private theorem, line 438): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `private theorem typedAccessPre_global_s32 (offset : Nat) (halign : offset % 4 = 0) : Typing.typedAccessPreconditions? .global .s32 (.global offset) = true := by`. Signature: `private theorem typedAccessPre_global_s32 (offset : Nat) (halign : offset % 4 = 0) : Typing.typedAccessPreconditions? .global .s32 (.global offset) = true := by`.

- `typedAccessPre_global_s64` ( private theorem, line 444): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `private theorem typedAccessPre_global_s64 (offset : Nat) (halign : offset % 8 = 0) : Typing.typedAccessPreconditions? .global .s64 (.global offset) = true := by`. Signature: `private theorem typedAccessPre_global_s64 (offset : Nat) (halign : offset % 8 = 0) : Typing.typedAccessPreconditions? .global .s64 (.global offset) = true := by`.

- `readMem_writeMem_same_global_u32` ( theorem, line 450): proves a same-address read-after-write property for a typed global memory store. It combines typed access preconditions, byte write/read lemmas, and scalar encode/decode facts. Signature: `theorem readMem_writeMem_same_global_u32 (st st' : State) (offset : Nat) (x : UInt32) (halign : offset % 4 = 0) (hwrite : Helpers.writeMem? st .global .u32 (.global offset) (.u32 x) = some st') : Helpers.readMem? st' .global .u32 (.global offset) = some (.u32 x) := by`.

- `readMem_writeMem_same_global_u64` ( theorem, line 470): proves a same-address read-after-write property for a typed global memory store. It combines typed access preconditions, byte write/read lemmas, and scalar encode/decode facts. Signature: `theorem readMem_writeMem_same_global_u64 (st st' : State) (offset : Nat) (x : UInt64) (halign : offset % 8 = 0) (hwrite : Helpers.writeMem? st .global .u64 (.global offset) (.u64 x) = some st') : Helpers.readMem? st' .global .u64 (.global offset) = some (.u64 x) := by`.

- `readMem_writeMem_same_global_s32` ( theorem, line 490): proves a same-address read-after-write property for a typed global memory store. It combines typed access preconditions, byte write/read lemmas, and scalar encode/decode facts. Signature: `theorem readMem_writeMem_same_global_s32 (st st' : State) (offset : Nat) (x : Int) (halign : offset % 4 = 0) (hwrite : Helpers.writeMem? st .global .s32 (.global offset) (.s32 x) = some st') : Helpers.readMem? st' .global .s32 (.global offset) = some (.s32 (Helpers.natToSigned 32 (Helpers.signedToNat 32 x))) := by`.

- `readMem_writeMem_same_global_s64` ( theorem, line 514): proves a same-address read-after-write property for a typed global memory store. It combines typed access preconditions, byte write/read lemmas, and scalar encode/decode facts. Signature: `theorem readMem_writeMem_same_global_s64 (st st' : State) (offset : Nat) (x : Int) (halign : offset % 8 = 0) (hwrite : Helpers.writeMem? st .global .s64 (.global offset) (.s64 x) = some st') : Helpers.readMem? st' .global .s64 (.global offset) = some (.s64 (Helpers.natToSigned 64 (Helpers.signedToNat 64 x))) := by`.

- `TopMemEq` ( def, line 538): defines `TopMemEq` in the frame/memory/preservation lemma layer. The signature is `def TopMemEq (st st' : State) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def TopMemEq (st st' : State) : Prop :=`.

- `LaneRegFilesEq` ( def, line 541): defines `LaneRegFilesEq` in the frame/memory/preservation lemma layer. The signature is `def LaneRegFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def LaneRegFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=`.

- `LanePredFilesEq` ( def, line 547): defines `LanePredFilesEq` in the frame/memory/preservation lemma layer. The signature is `def LanePredFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def LanePredFilesEq (st st' : State) (cta : CTAId) (warp : WarpId) : Prop :=`.

- `State.setCTA_global` ( theorem, line 562): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem State.setCTA_global (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).global = st.global := rfl`. Signature: `@[simp] theorem State.setCTA_global (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).global = st.global := rfl`.

- `State.setCTA_const` ( theorem, line 565): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem State.setCTA_const (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).const = st.const := rfl`. Signature: `@[simp] theorem State.setCTA_const (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).const = st.const := rfl`.

- `State.setCTA_param` ( theorem, line 568): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem State.setCTA_param (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).param = st.param := rfl`. Signature: `@[simp] theorem State.setCTA_param (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).param = st.param := rfl`.

- `State.setCTA_kernelEnv` ( theorem, line 571): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem State.setCTA_kernelEnv (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).kernelEnv = st.kernelEnv := rfl`. Signature: `@[simp] theorem State.setCTA_kernelEnv (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).kernelEnv = st.kernelEnv := rfl`.

- `State.setCTA_atomics` ( theorem, line 574): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `@[simp] theorem State.setCTA_atomics (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).atomics = st.atomics := rfl`. Signature: `@[simp] theorem State.setCTA_atomics (st : State) (cta : CTAId) (ctaState : CTAState) : (st.setCTA cta ctaState).atomics = st.atomics := rfl`.

- `State.setWarp_preserves_top` ( theorem, line 577): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem State.setWarp_preserves_top {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} (h : st.setWarp cta warp warpState = some st') : st'.global = st.global ∧ st'.const = st.const ∧ st'.param = st.param ∧ st'.kernelEnv = st.kernelEnv ∧ st'.atomics = st.atomics := by`.

- `State.setLane_preserves_top` ( theorem, line 590): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem State.setLane_preserves_top {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId} {laneState : LaneState} (h : st.setLane cta warp lane laneState = some st') : st'.global = st.global ∧ st'.const = st.const ∧ st'.param = st.param ∧ st'.kernelEnv = st.kernelEnv ∧ st'.atomics = st.atomics := by`.

- `applyToLaneIds?_nil_local` ( private theorem, line 603): unfolds the imperative `applyToLaneIds?` loop on a small list shape. These computation lemmas make later lane-wise proofs reason by list structure rather than opaque `for` syntax. Signature: `private theorem applyToLaneIds?_nil_local (st : State) (cta : CTAId) (warp : WarpId) (f : LaneId → LaneState → Option LaneState) : Helpers.applyToLaneIds? st cta warp [] f = some st := by`.

- `applyToLaneIds?_cons_local` ( private theorem, line 611): unfolds the imperative `applyToLaneIds?` loop on a small list shape. These computation lemmas make later lane-wise proofs reason by list structure rather than opaque `for` syntax. Signature: `private theorem applyToLaneIds?_cons_local (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId) (lanes : List LaneId) (f : LaneId → LaneState → Option LaneState) : Helpers.applyToLaneIds? st cta warp (lane :: lanes) f = (st.getLane? cta warp lane).bind fun laneState =>`.

- `applyToLaneIds?_preserves_top` ( theorem, line 634): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem applyToLaneIds?_preserves_top {st st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {f : LaneId → LaneState → Option LaneState} (h : Helpers.applyToLaneIds? st cta warp lanes f = some st') : st'.global = st.global ∧ st'.const = st.const ∧ st'.param = st.param ∧`.

- `advanceRunnablePcs?_preserves_top` ( theorem, line 667): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem advanceRunnablePcs?_preserves_top {st st' : State} {cta : CTAId} {warp : WarpId} (h : Helpers.advanceRunnablePcs? st cta warp = some st') : st'.global = st.global ∧ st'.const = st.const ∧ st'.param = st.param ∧ st'.kernelEnv = st.kernelEnv ∧ st'.atomics = st.atomics := by`.

- `applyAdvance_preserves_top` ( private theorem, line 686): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `private theorem applyAdvance_preserves_top {st sMid st' : State} {cta : CTAId} {warp : WarpId} {lanes : List LaneId} {f : LaneId → LaneState → Option LaneState} (hApply : Helpers.applyToLaneIds? st cta warp lanes f = some sMid) (hAdv : Helpers.advanceRunnablePcs? sMid cta warp = some st') :`.

- `step_applyAdvance_chain_top` ( private theorem, line 701): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `private theorem step_applyAdvance_chain_top {st st' : State} {cta : CTAId} {warp : WarpId} {parts : List LaneId} {f : LaneId → LaneState → Option LaneState} (h : ((Helpers.applyToLaneIds? st cta warp parts f).bind fun y => Helpers.advanceRunnablePcs? y cta warp) = some st') :`. Signature: `private theorem step_applyAdvance_chain_top {st st' : State} {cta : CTAId} {warp : WarpId} {parts : List LaneId} {f : LaneId → LaneState → Option LaneState} (h : ((Helpers.applyToLaneIds? st cta warp parts f).bind fun y => Helpers.advanceRunnablePcs? y cta warp) = some st') :`.

- `stepInstr_assignReg_preserves_mem` ( theorem, line 714): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_assignReg_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {rhs : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } = some st') :`.

- `stepInstr_assignPred_preserves_mem` ( theorem, line 735): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_assignPred_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } = some st') :`.

- `stepInstr_load_preserves_mem` ( theorem, line 756): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_load_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {src : TypedAddr} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .load dst src } = some st') :`.

- `stepInstr_cvta_preserves_mem` ( theorem, line 777): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_cvta_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } = some st') :`.

- `stepInstr_isspacep_preserves_mem` ( theorem, line 798): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_isspacep_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {space : AddrSpace} {src : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } = some st') :`.

- `stepInstr_barrierCTA_preserves_mem` ( theorem, line 819): proves a top-level frame property: the relevant instruction/update does not change the top-level memory fields it is not supposed to touch. These facts separate register/predicate computation from global memory effects. Signature: `theorem stepInstr_barrierCTA_preserves_mem {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_assignReg_preserves_pred_files` ( theorem, line 826): states that the instruction preserves lane predicate files. This is a frame lemma used when the instruction affects registers, memory, barriers, or control rather than predicates. Signature: `theorem stepInstr_assignReg_preserves_pred_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {rhs : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignReg dst rhs } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_assignPred_preserves_reg_files` ( theorem, line 833): states that the instruction preserves lane register files. This is a frame lemma used when the instruction affects memory, predicates, barriers, or control rather than registers. Signature: `theorem stepInstr_assignPred_preserves_reg_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {cmp : CmpExpr} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .assignPred dst cmp } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_store_preserves_reg_files` ( theorem, line 840): states that the instruction preserves lane register files. This is a frame lemma used when the instruction affects memory, predicates, barriers, or control rather than registers. Signature: `theorem stepInstr_store_preserves_reg_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : TypedAddr} {value : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .store dst value } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_store_preserves_pred_files` ( theorem, line 847): states that the instruction preserves lane predicate files. This is a frame lemma used when the instruction affects registers, memory, barriers, or control rather than predicates. Signature: `theorem stepInstr_store_preserves_pred_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : TypedAddr} {value : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .store dst value } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_cvta_preserves_pred_files` ( theorem, line 854): states that the instruction preserves lane predicate files. This is a frame lemma used when the instruction affects registers, memory, barriers, or control rather than predicates. Signature: `theorem stepInstr_cvta_preserves_pred_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : RegName} {space : AddrSpace} {src : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .cvta dst space src } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_isspacep_preserves_reg_files` ( theorem, line 861): states that the instruction preserves lane register files. This is a frame lemma used when the instruction affects memory, predicates, barriers, or control rather than registers. Signature: `theorem stepInstr_isspacep_preserves_reg_files {st st' : State} {cta : CTAId} {warp : WarpId} {dst : PredName} {space : AddrSpace} {src : RValue} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .isspacep dst space src } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_barrierCTA_preserves_reg_files` ( theorem, line 868): states that the instruction preserves lane register files. This is a frame lemma used when the instruction affects memory, predicates, barriers, or control rather than registers. Signature: `theorem stepInstr_barrierCTA_preserves_reg_files {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :`. **Status: admitted/in progress.**

- `stepInstr_barrierCTA_preserves_pred_files` ( theorem, line 875): states that the instruction preserves lane predicate files. This is a frame lemma used when the instruction affects registers, memory, barriers, or control rather than predicates. Signature: `theorem stepInstr_barrierCTA_preserves_pred_files {st st' : State} {cta : CTAId} {warp : WarpId} {barrierId : Nat} {guard? : Option Guard} (hstep : Helpers.stepInstr? st cta warp { guard? := guard?, instr := .barrierCTA barrierId } = some st') :`. **Status: admitted/in progress.**

- `stepInstr?_preserves_wf` ( theorem, line 882): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem stepInstr?_preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr} (hwf : State.wf st) (hstep : Helpers.stepInstr? st cta warp gi = some st') : State.wf st' := by`. **Status: admitted/in progress.**

- `stepInstr?_preserves_kernelEnv` ( theorem, line 889): proves that the executable step leaves the static kernel environment, or a block lookup inside it, unchanged. This is a frame fact for chaining instruction and block proofs. Signature: `theorem stepInstr?_preserves_kernelEnv {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr} (hstep : Helpers.stepInstr? st cta warp gi = some st') : st'.kernelEnv = st.kernelEnv := by`. **Status: admitted/in progress.**

- `stepTerminator?_preserves_wf` ( theorem, line 895): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem stepTerminator?_preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} {term : Terminator} (hwf : State.wf st) (hstep : Helpers.stepTerminator? st cta warp term = some st') : State.wf st' := by`. **Status: admitted/in progress.**

- `StepInstr.preserves_wf` ( theorem, line 902): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem StepInstr.preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} {gi : GInstr} (hstep : StepInstr st cta warp gi st') : State.wf st' := by`.

- `StepBlock.preserves_wf` ( theorem, line 910): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem StepBlock.preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} (hstep : StepBlock st cta warp st') : State.wf st' := by`.

- `StepWarp.preserves_wf` ( theorem, line 920): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem StepWarp.preserves_wf {st st' : State} {cta : CTAId} {warp : WarpId} (hstep : StepWarp st cta warp st') : State.wf st' := by`.

- `StepMachine.preserves_wf` ( theorem, line 928): proves that a semantic step or helper preserves `State.wf`. This keeps relational steps inside the well-formed-state fragment required by the small-step constructors. Signature: `theorem StepMachine.preserves_wf {st st' : State} (hstep : StepMachine st st') : State.wf st' := by`.

- `StepInstr.exists_of_stepInstr?_isSome` ( theorem, line 936): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepInstr.exists_of_stepInstr?_isSome {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`. Signature: `theorem StepInstr.exists_of_stepInstr?_isSome {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`.

- `StepMachine.body_of_stepInstr?` ( theorem, line 952): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.body_of_stepInstr? {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState)`. Signature: `theorem StepMachine.body_of_stepInstr? {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState)`.

- `StepMachine.term_of_stepTerminator?` ( theorem, line 971): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.term_of_stepTerminator? {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`. Signature: `theorem StepMachine.term_of_stepTerminator? {st st' : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`.

- `StepMachine.body_of_stepInstr?_computed` ( theorem, line 987): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.body_of_stepInstr?_computed {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState)`. Signature: `theorem StepMachine.body_of_stepInstr?_computed {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} {gi : GInstr} {participants : List LaneId} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState)`.

- `StepMachine.term_of_stepTerminator?_computed` ( theorem, line 1015): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.term_of_stepTerminator?_computed {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`. Signature: `theorem StepMachine.term_of_stepTerminator?_computed {st : State} {cta : CTAId} {warp : WarpId} {warpState : WarpState} {pc : PC} {block : Block} (hwf : State.wf st) (hwarp : st.getWarp? cta warp = some warpState) (hwfWarp : WarpState.wf warpState) (hlock : Helpers.lockstepRunnable warpState)`.

- `stepInstr?_computed_preserves_kernelEnv` ( theorem, line 1040): proves that the executable step leaves the static kernel environment, or a block lookup inside it, unchanged. This is a frame fact for chaining instruction and block proofs. Signature: `theorem stepInstr?_computed_preserves_kernelEnv {st : State} {cta : CTAId} {warp : WarpId} {gi : GInstr} (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) : (match Helpers.stepInstr? st cta warp gi with | some st' => st' | none => st).kernelEnv = st.kernelEnv := by`.

- `stepInstr?_computed_preserves_block?` ( theorem, line 1052): proves that the executable step leaves the static kernel environment, or a block lookup inside it, unchanged. This is a frame fact for chaining instruction and block proofs. Signature: `theorem stepInstr?_computed_preserves_block? {st : State} {cta : CTAId} {warp : WarpId} {gi : GInstr} (label : BlockLabel) (hisSome : (Helpers.stepInstr? st cta warp gi).isSome = true) : (match Helpers.stepInstr? st cta warp gi with | some st' => st' | none => st).kernelEnv.blocks[label]? = st.kernelEnv.blocks[label]? := by`.

- `StepMachine.currentInstrStep?` ( def, line 1063): defines `StepMachine.currentInstrStep?` in the frame/memory/preservation lemma layer. The signature is `def StepMachine.currentInstrStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def StepMachine.currentInstrStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do`.

- `StepMachine.currentTermStep?` ( def, line 1079): defines `StepMachine.currentTermStep?` in the frame/memory/preservation lemma layer. The signature is `def StepMachine.currentTermStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def StepMachine.currentTermStep? (st : State) (cta : CTAId) (warp : WarpId) : Option State := do`.

- `StepMachine.body_of_currentInstrStep?_computed` ( theorem, line 1095): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.body_of_currentInstrStep?_computed {st : State} {cta : CTAId} {warp : WarpId} (hisSome : (StepMachine.currentInstrStep? st cta warp).isSome = true) : StepMachine st (match StepMachine.currentInstrStep? st cta warp with | some st' => st' | none => st) := by`. Signature: `theorem StepMachine.body_of_currentInstrStep?_computed {st : State} {cta : CTAId} {warp : WarpId} (hisSome : (StepMachine.currentInstrStep? st cta warp).isSome = true) : StepMachine st (match StepMachine.currentInstrStep? st cta warp with | some st' => st' | none => st) := by`.

- `StepMachine.term_of_currentTermStep?_computed` ( theorem, line 1144): proves a property in the frame/memory/preservation lemma layer. The name and signature identify the exact fact: `theorem StepMachine.term_of_currentTermStep?_computed {st : State} {cta : CTAId} {warp : WarpId} (hisSome : (StepMachine.currentTermStep? st cta warp).isSome = true) : StepMachine st (match StepMachine.currentTermStep? st cta warp with | some st' => st' | none => st) := by`. Signature: `theorem StepMachine.term_of_currentTermStep?_computed {st : State} {cta : CTAId} {warp : WarpId} (hisSome : (StepMachine.currentTermStep? st cta warp).isSome = true) : StepMachine st (match StepMachine.currentTermStep? st cta warp with | some st' => st' | none => st) := by`.

### `CLean/Proof/Loops.lean`

Imports: `CLean.Semantics.Execution`.

Layer role: loop/correctness specification layer.

- `MachineFinal` ( def, line 7): defines relational finality: no `StepMachine` transition can leave the state. This is the proof-facing termination notion, stronger than the executable default scheduler being stuck. Signature: `def MachineFinal (st : State) : Prop :=`.

- `PrimaryMachineStuck` ( def, line 12): defines stuckness for the default executable scheduler at CTA 0 / warp 0. It is useful for smoke tests but not the general relational finality notion. Signature: `def PrimaryMachineStuck (st : State) : Prop :=`.

- `TerminatesAt` ( def, line 15): packages reachability from an initial state together with finality of the reached state. Partial and total correctness are phrased in terms of this predicate. Signature: `def TerminatesAt (init final : State) : Prop :=`.

- `PartialCorrect` ( def, line 18): states that every terminal state reachable from `init` satisfies the postcondition. This is the main relational partial-correctness contract. Signature: `def PartialCorrect (init : State) (post : State → Prop) : Prop :=`.

- `TotalCorrect` ( def, line 21): states that some terminal state is reachable and satisfies the postcondition. It combines termination existence with the postcondition. Signature: `def TotalCorrect (init : State) (post : State → Prop) : Prop :=`.

- `SymbolicRunSummary` ( structure, line 30): packages a computed `runN` trace, its finality, its postcondition, and uniqueness among terminal states. It is an ergonomic bridge for examples while deeper proofs are built. Signature: `structure SymbolicRunSummary (init : State) (fuel : Nat) (post : State → Prop) : Prop where`.

- `cta_scheduler_independence` ( theorem, line 51): distributes a global postcondition over per-CTA postconditions using a caller-supplied combiner. The theorem is logical packaging, not a scheduler-independence axiom. Signature: `theorem cta_scheduler_independence (init : State) (post : State → Prop) (perCTAPost : CTAId → State → Prop) (hcombine : ∀ final, (∀ cta, perCTAPost cta final) → post final) (hperCTA : ∀ cta, PartialCorrect init (perCTAPost cta)) : PartialCorrect init post :=`.

- `TotalCorrect.partial` ( theorem, line 59): extracts the reachable final state and postcondition from a total-correctness proof, dropping the finality component. Signature: `theorem TotalCorrect.partial {init : State} {post : State → Prop} (h : TotalCorrect init post) : ∃ final, Reaches init final ∧ post final := by`.

- `preserves` ( theorem, line 67): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem preserves {P : State → Prop} {st st' : State} (hstep : ∀ {a b : State}, StepMachine a b → P a → P b) (hreach : Reaches st st') (hinit : P st) : P st' := by`. Signature: `theorem preserves {P : State → Prop} {st st' : State} (hstep : ∀ {a b : State}, StepMachine a b → P a → P b) (hreach : Reaches st st') (hinit : P st) : P st' := by`.

- `post_of_final_invariant` ( theorem, line 76): establishes a finality or final-state property used to close a correctness proof. It typically bridges executable stuckness, relational `MachineFinal`, and `runN` endpoints. Signature: `theorem post_of_final_invariant {P post : State → Prop} {st st' : State} (hstep : ∀ {a b : State}, StepMachine a b → P a → P b) (hexit : ∀ final, MachineFinal final → P final → post final) (hreach : Reaches st st') (hfinal : MachineFinal st') (hinit : P st) : post st' :=`.

- `step?_to_reaches` ( theorem, line 86): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem step?_to_reaches {st st' : State} (h : StepMachine.step? st = some st') : Reaches st st' :=`. Signature: `theorem step?_to_reaches {st st' : State} (h : StepMachine.step? st = some st') : Reaches st st' :=`.

- `runN_reaches'` ( theorem, line 93): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem runN_reaches' (fuel : Nat) (st : State) : Reaches st (StepMachine.runN fuel st) :=`. Signature: `theorem runN_reaches' (fuel : Nat) (st : State) : Reaches st (StepMachine.runN fuel st) :=`.

- `reaches_runN_step` ( theorem, line 101): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem reaches_runN_step {fuel : Nat} {st : State} (_h : ∀ k < fuel, (StepMachine.step? (StepMachine.runN k st)).isSome = true) : Reaches st (StepMachine.runN fuel st) :=`. Signature: `theorem reaches_runN_step {fuel : Nat} {st : State} (_h : ∀ k < fuel, (StepMachine.step? (StepMachine.runN k st)).isSome = true) : Reaches st (StepMachine.runN fuel st) :=`.

- `Reaches.compose` ( theorem, line 109): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem Reaches.compose {st₀ st₁ st₂ : State} (h01 : Reaches st₀ st₁) (h12 : Reaches st₁ st₂) : Reaches st₀ st₂ :=`. Signature: `theorem Reaches.compose {st₀ st₁ st₂ : State} (h01 : Reaches st₀ st₁) (h12 : Reaches st₁ st₂) : Reaches st₀ st₂ :=`.

- `RelReaches` ( inductive, line 118): defines reflexive/transitive closure for an arbitrary relation, independent of machine states. Counted loop proofs use it as an abstract reachability layer. Signature: `inductive RelReaches {σ : Type u} (step : σ → σ → Prop) : σ → σ → Prop where`.

- `preserves` ( theorem, line 127): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem preserves {σ : Type u} {step : σ → σ → Prop} {P : σ → Prop} {s s' : σ} (hstep : ∀ {a b : σ}, step a b → P a → P b) (hreach : RelReaches step s s') (hinit : P s) : P s' := by`. Signature: `theorem preserves {σ : Type u} {step : σ → σ → Prop} {P : σ → Prop} {s s' : σ} (hstep : ∀ {a b : σ}, step a b → P a → P b) (hreach : RelReaches step s s') (hinit : P s) : P s' := by`.

- `CountedLoopSpec` ( structure, line 140): packages a loop invariant, guard, body relation, variant, and postcondition for abstract counted-loop proofs. Signature: `structure CountedLoopSpec (σ : Type u) where`.

- `Exited` ( def, line 149): defines the exit condition for a `CountedLoopSpec`: invariant holds and guard is false. Signature: `def Exited {σ : Type u} (spec : CountedLoopSpec σ) (s : σ) : Prop :=`.

- `partial_correct` ( theorem, line 152): proves abstract counted-loop partial correctness from invariant preservation, an exit-to-post rule, reachability, initial invariant, and false final guard. Signature: `theorem partial_correct {σ : Type u} (spec : CountedLoopSpec σ) {init final : σ} (hpres : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.inv s') (hexit : ∀ s : σ, spec.Exited s → spec.post s) (hreach : RelReaches spec.body init final) (hinit : spec.inv init) (hnotGuard : ¬ spec.guard final) :`.

- `total_correct` ( theorem, line 166): proves abstract counted-loop total correctness from an invariant, body existence, preservation, strict variant decrease, and exit-to-post rule. Signature: `theorem total_correct {σ : Type u} (spec : CountedLoopSpec σ) (init : σ) (hinit : spec.inv init) (hbody : ∀ s : σ, spec.inv s → spec.guard s → ∃ s', spec.body s s') (hpres : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.inv s') (hdec : ∀ {s s' : σ}, spec.body s s' → spec.inv s → spec.guard s →`.

- `reachesBody` ( def, line 219): turns a state-to-state predicate into a relation that also carries a `Reaches` witness. It is a bridge from concrete machine traces to abstract loop-body relations. Signature: `def reachesBody (predicate : State → State → Prop) : State → State → Prop :=`.

- `reachesBody.fromReaches` ( theorem, line 222): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem reachesBody.fromReaches {predicate : State → State → Prop} {s s' : State} (hr : Reaches s s') (hp : predicate s s') : reachesBody predicate s s' :=`. Signature: `theorem reachesBody.fromReaches {predicate : State → State → Prop} {s s' : State} (hr : Reaches s s') (hp : predicate s s') : reachesBody predicate s s' :=`.

- `RelReaches.reachesBody_to_reaches` ( theorem, line 228): proves a property in the loop/correctness specification layer. The name and signature identify the exact fact: `theorem RelReaches.reachesBody_to_reaches {predicate : State → State → Prop} {init final : State} (h : RelReaches (reachesBody predicate) init final) : Reaches init final := by`. Signature: `theorem RelReaches.reachesBody_to_reaches {predicate : State → State → Prop} {init final : State} (h : RelReaches (reachesBody predicate) init final) : Reaches init final := by`.

- `globalF32At?` ( def, line 236): reads a global-memory `f32` cell at `base + index * 4` and projects the contained `Float` if decoding succeeds. Signature: `def globalF32At? (st : State) (base index : Nat) : Option Float := do`.

- `globalS32At?` ( def, line 241): reads a global-memory `s32` cell at `base + index * 4` and projects the contained `Int` if decoding succeeds. Signature: `def globalS32At? (st : State) (base index : Nat) : Option Int := do`.

- `listFloatGetD` ( def, line 246): gets a float list element with default `0.0` when the index is out of bounds. Matrix specs use it to avoid partial list indexing. Signature: `def listFloatGetD : List Float → Nat → Float | [], _ => 0.0 | x :: _, 0 => x | _ :: xs, i + 1 => listFloatGetD xs i`.

- `listIntGetD` ( def, line 251): gets an integer list element with default `0` when the index is out of bounds. SAXPY specs use it to define expected values for every lane index. Signature: `def listIntGetD : List Int → Nat → Int | [], _ => 0 | x :: _, 0 => x | _ :: xs, i + 1 => listIntGetD xs i`.

- `s32Wrap` ( def, line 256): wraps an integer through signed 32-bit encoding and decoding. It matches the project’s modeled `s32` arithmetic behavior. Signature: `def s32Wrap (x : Int) : Int :=`.

- `matrixIndex` ( def, line 259): converts row/column coordinates to a row-major flat index. Signature: `def matrixIndex (n row col : Nat) : Nat :=`.

- `matrixCellOffset` ( def, line 262): turns a row/column coordinate and base address into a byte offset for a 32-bit matrix cell. Signature: `def matrixCellOffset (n row col base : Nat) : Nat :=`.

- `vectorF32Post` ( def, line 265): states that the first `n` global `f32` cells at a base address match an expected function. Signature: `def vectorF32Post (base n : Nat) (expected : Nat → Float) (st : State) : Prop :=`.

- `vectorS32Post` ( def, line 268): states that the first `n` global `s32` cells at a base address match an expected function. Signature: `def vectorS32Post (base n : Nat) (expected : Nat → Int) (st : State) : Prop :=`.

- `saxpyExpectedAt` ( def, line 271): defines the SAXPY expected integer result at index `i`: `xs[i] * alpha + ys[i]`, wrapped as signed 32-bit. Signature: `def saxpyExpectedAt (alpha : Int) (xs ys : List Int) (i : Nat) : Int :=`.

- `saxpyPost` ( def, line 274): specializes the `s32` vector postcondition to the SAXPY output base and expected-value function. Signature: `def saxpyPost (base n : Nat) (alpha : Int) (xs ys : List Int) (st : State) : Prop :=`.

- `dotF32List` ( def, line 277): computes a row/column dot product from flat row-major input matrices using Lean `Float` arithmetic. Signature: `def dotF32List (n row col : Nat) (a b : List Float) : Float :=`.

- `matmulCellPost` ( def, line 282): states that one output matrix cell in global memory equals `dotF32List` for that row and column. Signature: `def matmulCellPost (n row col cBase : Nat) (a b : List Float) (st : State) : Prop :=`.

- `matmulPost` ( def, line 285): lifts `matmulCellPost` to every row and column in an `n x n` output matrix. Signature: `def matmulPost (n cBase : Nat) (a b : List Float) (st : State) : Prop :=`.

### `CLean/Examples.lean`

Imports: `CLean.Examples.All`.

This file is an import/configuration node in the example/regression layer and introduces no named Lean `def` or `theorem` declarations.

### `CLean/Examples/All.lean`

Imports: `CLean.Examples.Common`, `CLean.Examples.Semantic`, `CLean.Examples.Lowering`, `CLean.Examples.Parser`, `CLean.Examples.Saxpy`, `CLean.Examples.Matmul`.

This file is an import/configuration node in the example/regression layer and introduces no named Lean `def` or `theorem` declarations.

### `CLean/Examples/Common.lean`

Imports: `CLean.PTX.Parser`, `CLean.PTX.Lowering`, `CLean.Proof.Automation`, `CLean.Proof.Loops`, `Mathlib.Tactic`.

Layer role: example/regression layer.

- `lane0` ( def, line 11): defines `lane0` in the example/regression layer. The signature is `def lane0 : LaneId := ⟨0, by decide⟩`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0 : LaneId := ⟨0, by decide⟩`.

- `lane0HasRegU32` ( def, line 13): defines `lane0HasRegU32` in the example/regression layer. The signature is `def lane0HasRegU32 (reg : RegName) (value : UInt32) (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0HasRegU32 (reg : RegName) (value : UInt32) (st : State) : Bool :=`.

- `lane0HasRegS32` ( def, line 21): defines `lane0HasRegS32` in the example/regression layer. The signature is `def lane0HasRegS32 (reg : RegName) (value : Int) (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0HasRegS32 (reg : RegName) (value : Int) (st : State) : Bool :=`.

- `lane0HasR1Seven` ( def, line 29): defines `lane0HasR1Seven` in the example/regression layer. The signature is `def lane0HasR1Seven (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0HasR1Seven (st : State) : Bool :=`.

- `lane0Terminated` ( def, line 32): defines `lane0Terminated` in the example/regression layer. The signature is `def lane0Terminated (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0Terminated (st : State) : Bool :=`.

- `lane0HasGlobalAddr` ( def, line 37): defines `lane0HasGlobalAddr` in the example/regression layer. The signature is `def lane0HasGlobalAddr (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0HasGlobalAddr (st : State) : Bool :=`.

- `lane0PredQTrue` ( def, line 45): defines `lane0PredQTrue` in the example/regression layer. The signature is `def lane0PredQTrue (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0PredQTrue (st : State) : Bool :=`.

- `copySrcOffset` ( def, line 53): defines `copySrcOffset` in the example/regression layer. The signature is `def copySrcOffset : Nat := 0`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def copySrcOffset : Nat := 0`.

- `copyDstOffset` ( def, line 55): defines `copyDstOffset` in the example/regression layer. The signature is `def copyDstOffset : Nat := 4`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def copyDstOffset : Nat := 4`.

- `copyValue` ( def, line 57): defines `copyValue` in the example/regression layer. The signature is `def copyValue : UInt32 := 99`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def copyValue : UInt32 := 99`.

- `activeMaskPrefix` ( def, line 59): defines `activeMaskPrefix` in the example/regression layer. The signature is `def activeMaskPrefix (n : Nat) : UInt32 :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def activeMaskPrefix (n : Nat) : UInt32 :=`.

- `writeU32Bytes` ( def, line 62): encodes a scalar value into little-endian bytes and writes it into a byte memory. Example state constructors use it to seed global and parameter memory. Signature: `def writeU32Bytes (mem : ByteMem) (off : Nat) (x : UInt32) : ByteMem :=`.

- `writeU64Bytes` ( def, line 65): encodes a scalar value into little-endian bytes and writes it into a byte memory. Example state constructors use it to seed global and parameter memory. Signature: `def writeU64Bytes (mem : ByteMem) (off : Nat) (x : UInt64) : ByteMem :=`.

- `writeS32Bytes` ( def, line 68): encodes a scalar value into little-endian bytes and writes it into a byte memory. Example state constructors use it to seed global and parameter memory. Signature: `def writeS32Bytes (mem : ByteMem) (off : Nat) (x : Int) : ByteMem :=`.

- `writeF32Bytes` ( def, line 71): encodes a scalar value into little-endian bytes and writes it into a byte memory. Example state constructors use it to seed global and parameter memory. Signature: `def writeF32Bytes (mem : ByteMem) (off : Nat) (x : Float) : ByteMem :=`.

- `writeS32Vector` ( def, line 74): iterates the scalar byte helpers over a vector. It either builds initial memory contents or checks final memory contents element by element. Signature: `def writeS32Vector (mem : ByteMem) (base : Nat) (xs : List Int) : ByteMem :=`.

- `writeF32Vector` ( def, line 80): iterates the scalar byte helpers over a vector. It either builds initial memory contents or checks final memory contents element by element. Signature: `def writeF32Vector (mem : ByteMem) (base : Nat) (xs : List Float) : ByteMem :=`.

- `readGlobalS32?` ( def, line 86): reads and projects a typed value from global memory for examples and postconditions. It wraps the generic semantic memory reader with a concrete expected type. Signature: `def readGlobalS32? (st : State) (off : Nat) : Option Int := do`.

- `readGlobalF32?` ( def, line 91): reads and projects a typed value from global memory for examples and postconditions. It wraps the generic semantic memory reader with a concrete expected type. Signature: `def readGlobalF32? (st : State) (off : Nat) : Option Float := do`.

- `globalS32VectorMatches?` ( def, line 96): iterates the scalar byte helpers over a vector. It either builds initial memory contents or checks final memory contents element by element. Signature: `def globalS32VectorMatches? (st : State) (base : Nat) (expected : List Int) : Bool :=`.

- `globalF32VectorMatches?` ( def, line 105): iterates the scalar byte helpers over a vector. It either builds initial memory contents or checks final memory contents element by element. Signature: `def globalF32VectorMatches? (st : State) (base : Nat) (expected : List Float) : Bool :=`.

- `copyDstHasValue` ( def, line 114): defines `copyDstHasValue` in the example/regression layer. The signature is `def copyDstHasValue (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def copyDstHasValue (st : State) : Bool :=`.

- `copyLane0Terminated` ( def, line 119): defines `copyLane0Terminated` in the example/regression layer. The signature is `def copyLane0Terminated (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def copyLane0Terminated (st : State) : Bool :=`.

- `lane0RunningAt` ( def, line 122): defines `lane0RunningAt` in the example/regression layer. The signature is `def lane0RunningAt (pc : PC) (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def lane0RunningAt (pc : PC) (st : State) : Bool :=`.

- `barrier0Released` ( def, line 127): defines `barrier0Released` in the example/regression layer. The signature is `def barrier0Released (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `def barrier0Released (st : State) : Bool :=`.

### `CLean/Examples/Lowering.lean`

Imports: `CLean.Examples.Common`.

Layer role: example/regression layer.

- `example@6` ( example, line 6): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.lowerInstr (.mov .u32 "r1" (.imm (.u32 7))) = .assignReg "r1" (.imm (.u32 7)) := by`.

- `example@11` ( example, line 11): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.lowerInstr (.binop .add .u32 "r3" (.reg "r1") (.reg "r2")) = .assignReg "r3" (.binop .add (.reg "r1") (.reg "r2")) := by`.

- `example@16` ( example, line 16): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.lowerInstr (.barSync 0) = .barrierCTA 0 := by`.

- `example@20` ( example, line 20): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerInstrChecked? {} (.mov .u32 "r1" (.imm (.u32 7))) with | .ok (.assignReg "r1" (.imm (.u32 7)), env) => env.regs["r1"]? == some .u32 | _ => false) = true := by`.

- `example@26` ( example, line 26): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerInstrChecked? {} (.binop .add .u32 "r3" (.reg "r1") (.reg "r2")) with | .error _ => true | _ => false) = true := by`.

- `ptxAssignKernel` ( private def, line 32): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def ptxAssignKernel : PTX.Kernel :=`.

- `ptxAssignWarp0` ( private def, line 41): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def ptxAssignWarp0 : WarpState :=`.

- `ptxAssignState` ( private def, line 44): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxAssignState : State :=`.

- `ptxAssignFinalState` ( private def, line 49): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxAssignFinalState : State :=`.

- `ptx_assign_kernel_run_functional` ( theorem, line 52): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem ptx_assign_kernel_run_functional : Reaches ptxAssignState ptxAssignFinalState ∧ lane0HasR1Seven ptxAssignFinalState = true ∧ lane0Terminated ptxAssignFinalState = true := by`.

- `ptxCopyKernel` ( private def, line 60): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def ptxCopyKernel : PTX.Kernel :=`.

- `ptxCopyWarp0` ( private def, line 72): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def ptxCopyWarp0 : WarpState :=`.

- `ptxCopyState` ( private def, line 75): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxCopyState : State :=`.

- `ptxCopyFinalState` ( private def, line 81): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxCopyFinalState : State :=`.

- `ptx_copy_kernel_run_functional` ( theorem, line 84): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem ptx_copy_kernel_run_functional : Reaches ptxCopyState ptxCopyFinalState ∧ copyDstHasValue ptxCopyFinalState = true ∧ copyLane0Terminated ptxCopyFinalState = true := by`.

- `ptxBarrierKernel` ( private def, line 92): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def ptxBarrierKernel : PTX.Kernel :=`.

- `ptxBarrierWarp0` ( private def, line 104): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def ptxBarrierWarp0 : WarpState :=`.

- `ptxBarrierState` ( private def, line 107): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxBarrierState : State :=`.

- `ptxBarrierFinalState` ( private def, line 113): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxBarrierFinalState : State :=`.

- `ptx_barrier_kernel_run_functional` ( theorem, line 116): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem ptx_barrier_kernel_run_functional : Reaches ptxBarrierState ptxBarrierFinalState ∧ barrier0Released ptxBarrierFinalState = true ∧ lane0HasR1Seven ptxBarrierFinalState = true ∧ lane0Terminated ptxBarrierFinalState = true := by`.

- `ptxAddKernel` ( private def, line 126): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def ptxAddKernel : PTX.Kernel :=`.

- `ptxAddLane0` ( private def, line 139): defines `ptxAddLane0` in the example/regression layer. The signature is `private def ptxAddLane0 : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def ptxAddLane0 : LaneState :=`.

- `ptxAddWarp0` ( private def, line 145): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def ptxAddWarp0 : WarpState :=`.

- `ptxAddState` ( private def, line 148): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxAddState : State :=`.

- `ptxAddFinalState` ( private def, line 153): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def ptxAddFinalState : State :=`.

- `example@156` ( example, line 156): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? ptxAddKernel with | .ok _ => true | .error _ => false) = true := by`.

- `ptx_add_kernel_run_functional` ( theorem, line 162): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem ptx_add_kernel_run_functional : Reaches ptxAddState ptxAddFinalState ∧ lane0HasRegU32 "r3" 7 ptxAddFinalState = true ∧ lane0Terminated ptxAddFinalState = true := by`.

### `CLean/Examples/Matmul.lean`

Imports: `CLean.Examples.Common`, `CLean.PTX.Bridge`.

Layer role: matmul case-study layer.

- `matmulKernelText` ( private def, line 53): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def matmulKernelText : String :=`.

- `matmulKernel` ( private def, line 182): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def matmulKernel : PTX.Kernel :=`.

- `example@187` ( example, line 187): is an anonymous regression check in the matmul case-study layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.parseAndLowerKernelOk? matmulKernelText = true := by`.

- `example@190` ( example, line 190): is an anonymous regression check in the matmul case-study layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.lowerKernelSupported? matmulKernel = true := by`.

- `matmulABase` ( def, line 193): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def matmulABase : Nat := 0`.

- `matmulBBase` ( def, line 194): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def matmulBBase : Nat := 128`.

- `matmulCBase` ( def, line 195): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def matmulCBase : Nat := 256`.

- `matmulCTAIdForCell` ( private def, line 197): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `private def matmulCTAIdForCell (n row col : Nat) : CTAId :=`.

- `matmulParamBytesFor` ( private def, line 200): lays out kernel parameter bytes in `ParamMem` according to the lowered parameter offsets. Case-study state constructors use it to make `ld.param` instructions executable. Signature: `private def matmulParamBytesFor (n : Nat) : ByteMem :=`.

- `matmulGlobalBytesFor` ( private def, line 206): initializes global byte memory from input vectors or matrices. It is the data fixture read by modeled global loads. Signature: `private def matmulGlobalBytesFor (a b : List Float) : ByteMem :=`.

- `matmulKernelEnvFor` ( private def, line 210): defines `matmulKernelEnvFor` in the matmul case-study layer. The signature is `private def matmulKernelEnvFor (n : Nat) : KernelEnv :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def matmulKernelEnvFor (n : Nat) : KernelEnv :=`.

- `matmulWarpForCell` ( private def, line 214): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def matmulWarpForCell : WarpState :=`.

- `matmulStateForCell` ( def, line 217): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `def matmulStateForCell (n row col : Nat) (a b : List Float) : State :=`.

- `matmulOneCellPost` ( def, line 224): defines a postcondition or Boolean checker for the example/case-study output. Correctness theorems ultimately reduce to this predicate on the final state. Signature: `def matmulOneCellPost (n row col : Nat) (a b : List Float) (st : State) : Prop :=`.

- `matmulConcreteA` ( private def, line 230): defines `matmulConcreteA` in the matmul case-study layer. The signature is `private def matmulConcreteA : List Float := [3.0]`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def matmulConcreteA : List Float := [3.0]`.

- `matmulConcreteB` ( private def, line 231): defines `matmulConcreteB` in the matmul case-study layer. The signature is `private def matmulConcreteB : List Float := [4.0]`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def matmulConcreteB : List Float := [4.0]`.

- `matmulConcreteState` ( private def, line 233): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def matmulConcreteState : State :=`.

- `matmulConcreteFinalState` ( private def, line 236): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def matmulConcreteFinalState : State :=`.

- `matmulConcretePost?` ( private def, line 239): defines a postcondition or Boolean checker for the example/case-study output. Correctness theorems ultimately reduce to this predicate on the final state. Signature: `private def matmulConcretePost? (st : State) : Bool :=`.

- `example@244` ( example, line 244): is an anonymous regression check in the matmul case-study layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (dotF32List 1 0 0 matmulConcreteA matmulConcreteB == 12.0) = true := by`.

- `cached_matmul_n1_one_cell_functional` ( theorem, line 251): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem cached_matmul_n1_one_cell_functional : Reaches matmulConcreteState matmulConcreteFinalState ∧ matmulConcretePost? matmulConcreteFinalState = true := by`.

- `matmul_cell_partial_correct` ( theorem, line 281): states or proves partial correctness: every terminal state reachable from the initial state satisfies the specified postcondition. In case-study files, admitted versions identify the remaining proof obligation. Signature: `theorem matmul_cell_partial_correct (n row col : Nat) (_hn : n > 0) (_hrow : row < n) (_hcol : col < n) (a b : List Float) (_ha : a.length = n * n) (_hb : b.length = n * n) : PartialCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by`. **Status: admitted/in progress.**

- `matmul_cell_total_correct` ( theorem, line 293): states or proves total correctness: a terminal state exists and satisfies the specified postcondition. In case-study files, admitted versions identify the remaining termination-plus-post proof obligation. Signature: `theorem matmul_cell_total_correct (n row col : Nat) (_hn : n > 0) (_hrow : row < n) (_hcol : col < n) (a b : List Float) (_ha : a.length = n * n) (_hb : b.length = n * n) : TotalCorrect (matmulStateForCell n row col a b) (matmulOneCellPost n row col a b) := by`. **Status: admitted/in progress.**

### `CLean/Examples/Parser.lean`

Imports: `CLean.Examples.Common`.

Layer role: example/regression layer.

- `parsedAssignText` ( private def, line 6): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedAssignText : String :=`.

- `parsedAssignKernel` ( private def, line 14): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedAssignKernel : PTX.Kernel :=`.

- `example@19` ( example, line 19): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseKernel parsedAssignText with | .ok kernel => kernel.entry == "parsed_assign" && kernel.regs.size == 1 && kernel.blocks.size == 1 | .error _ => false) = true := by`.

- `parsedAssignWarp0` ( private def, line 25): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedAssignWarp0 : WarpState :=`.

- `parsedAssignState` ( private def, line 28): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedAssignState : State :=`.

- `parsedAssignFinalState` ( private def, line 33): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedAssignFinalState : State :=`.

- `parsed_ptx_assign_kernel_run_functional` ( theorem, line 36): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_assign_kernel_run_functional : Reaches parsedAssignState parsedAssignFinalState ∧ lane0HasR1Seven parsedAssignFinalState = true ∧ lane0Terminated parsedAssignFinalState = true := by`.

- `parsedAddText` ( private def, line 44): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedAddText : String :=`.

- `parsedAddKernel` ( private def, line 54): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedAddKernel : PTX.Kernel :=`.

- `example@59` ( example, line 59): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseKernel parsedAddText with | .ok kernel => kernel.entry == "parsed_add" && kernel.regs.size == 3 && kernel.blocks.size == 1 | .error _ => false) = true := by`.

- `parsedAddLane0` ( private def, line 65): defines `parsedAddLane0` in the example/regression layer. The signature is `private def parsedAddLane0 : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def parsedAddLane0 : LaneState :=`.

- `parsedAddWarp0` ( private def, line 71): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedAddWarp0 : WarpState :=`.

- `parsedAddState` ( private def, line 74): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedAddState : State :=`.

- `parsedAddFinalState` ( private def, line 79): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedAddFinalState : State :=`.

- `parsed_ptx_add_kernel_run_functional` ( theorem, line 82): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_add_kernel_run_functional : Reaches parsedAddState parsedAddFinalState ∧ lane0HasRegU32 "r3" 7 parsedAddFinalState = true ∧ lane0Terminated parsedAddFinalState = true := by`.

- `parsedCopyText` ( private def, line 90): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedCopyText : String :=`.

- `parsedCopyKernel` ( private def, line 99): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedCopyKernel : PTX.Kernel :=`.

- `parsedCopyWarp0` ( private def, line 104): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedCopyWarp0 : WarpState :=`.

- `parsedCopyState` ( private def, line 107): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedCopyState : State :=`.

- `parsedCopyFinalState` ( private def, line 113): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedCopyFinalState : State :=`.

- `parsed_ptx_copy_kernel_run_functional` ( theorem, line 116): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_copy_kernel_run_functional : Reaches parsedCopyState parsedCopyFinalState ∧ copyDstHasValue parsedCopyFinalState = true ∧ copyLane0Terminated parsedCopyFinalState = true := by`.

- `parsedParamCopyText` ( private def, line 124): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedParamCopyText : String :=`.

- `parsedParamCopyKernel` ( private def, line 139): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedParamCopyKernel : PTX.Kernel :=`.

- `example@144` ( example, line 144): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseKernel parsedParamCopyText with | .ok kernel => kernel.entry == "parsed_param_copy" && kernel.params.size == 2 && kernel.regs.size == 3 && kernel.blocks.size == 1 | .error _ => false) = true := by`.

- `example@151` ( example, line 151): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedParamCopyKernel with | .ok env => env.params.size == 2 && env.params[0]?.map (fun p => p.offset) == some 0 && env.params[1]?.map (fun p => p.offset) == some 8 | .error _ => false) = true := by`.

- `parsedParamCopyParamBytes` ( private def, line 160): lays out kernel parameter bytes in `ParamMem` according to the lowered parameter offsets. Case-study state constructors use it to make `ld.param` instructions executable. Signature: `private def parsedParamCopyParamBytes : ByteMem :=`.

- `parsedParamCopyWarp0` ( private def, line 164): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedParamCopyWarp0 : WarpState :=`.

- `parsedParamCopyState` ( private def, line 167): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedParamCopyState : State :=`.

- `parsedParamCopyFinalState` ( private def, line 174): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedParamCopyFinalState : State :=`.

- `parsed_ptx_param_copy_kernel_run_functional` ( theorem, line 177): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_param_copy_kernel_run_functional : Reaches parsedParamCopyState parsedParamCopyFinalState ∧ copyDstHasValue parsedParamCopyFinalState = true ∧ copyLane0Terminated parsedParamCopyFinalState = true := by`.

- `parsedBarrierText` ( private def, line 185): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedBarrierText : String :=`.

- `parsedBarrierKernel` ( private def, line 194): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedBarrierKernel : PTX.Kernel :=`.

- `parsedBarrierWarp0` ( private def, line 199): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedBarrierWarp0 : WarpState :=`.

- `parsedBarrierState` ( private def, line 202): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedBarrierState : State :=`.

- `parsedBarrierFinalState` ( private def, line 208): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedBarrierFinalState : State :=`.

- `parsed_ptx_barrier_kernel_run_functional` ( theorem, line 211): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_barrier_kernel_run_functional : Reaches parsedBarrierState parsedBarrierFinalState ∧ barrier0Released parsedBarrierFinalState = true ∧ lane0HasR1Seven parsedBarrierFinalState = true ∧ lane0Terminated parsedBarrierFinalState = true := by`.

- `parsedSharedText` ( private def, line 221): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedSharedText : String :=`.

- `parsedSharedKernel` ( private def, line 232): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedSharedKernel : PTX.Kernel :=`.

- `example@237` ( example, line 237): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseKernel parsedSharedText with | .ok kernel => kernel.entry == "parsed_shared" && kernel.shareds.size == 1 && kernel.regs.size == 1 && kernel.blocks.size == 1 | .error _ => false) = true := by`.

- `example@244` ( example, line 244): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedSharedKernel with | .ok env => env.sharedDecls.size == 1 && env.sharedDecls[0]?.map (fun decl => decl.offset) == some 0 && env.sharedDecls[0]?.map (fun decl => decl.size) == some 4 | .error _ => false) = true := by`.

- `parsedSharedWarp0` ( private def, line 253): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedSharedWarp0 : WarpState :=`.

- `parsedSharedState` ( private def, line 256): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedSharedState : State :=`.

- `parsedSharedFinalState` ( private def, line 262): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedSharedFinalState : State :=`.

- `parsed_ptx_shared_barrier_kernel_run_functional` ( theorem, line 265): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_shared_barrier_kernel_run_functional : Reaches parsedSharedState parsedSharedFinalState ∧ barrier0Released parsedSharedFinalState = true ∧ lane0HasR1Seven parsedSharedFinalState = true ∧ lane0Terminated parsedSharedFinalState = true := by`.

- `parsedBraText` ( private def, line 275): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedBraText : String :=`.

- `parsedBraKernel` ( private def, line 285): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedBraKernel : PTX.Kernel :=`.

- `example@290` ( example, line 290): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedBraKernel with | .ok env => env.blocks["parsed_bra"]?.isSome && env.blocks["target"]?.isSome | .error _ => false) = true := by`.

- `parsedBraWarp0` ( private def, line 296): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedBraWarp0 : WarpState :=`.

- `parsedBraState` ( private def, line 299): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedBraState : State :=`.

- `parsedBraFinalState` ( private def, line 304): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedBraFinalState : State :=`.

- `parsed_ptx_bra_kernel_run_functional` ( theorem, line 307): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_bra_kernel_run_functional : Reaches parsedBraState parsedBraFinalState ∧ lane0HasR1Seven parsedBraFinalState = true ∧ lane0Terminated parsedBraFinalState = true := by`.

- `parsedCbraText` ( private def, line 315): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedCbraText : String :=`.

- `parsedCbraKernel` ( private def, line 330): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedCbraKernel : PTX.Kernel :=`.

- `example@335` ( example, line 335): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedCbraKernel with | .ok env => env.blocks["parsed_cbra"]?.isSome && env.blocks["then_blk"]?.isSome && env.blocks["else_blk"]?.isSome | .error _ => false) = true := by`.

- `parsedCbraWarp0` ( private def, line 344): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedCbraWarp0 : WarpState :=`.

- `parsedCbraState` ( private def, line 347): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedCbraState : State :=`.

- `parsedCbraFinalState` ( private def, line 352): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedCbraFinalState : State :=`.

- `parsed_ptx_cbra_kernel_run_functional` ( theorem, line 355): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_cbra_kernel_run_functional : Reaches parsedCbraState parsedCbraFinalState ∧ lane0HasR1Seven parsedCbraFinalState = true ∧ lane0Terminated parsedCbraFinalState = true := by`.

- `parsedScalarOpsText` ( private def, line 363): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedScalarOpsText : String :=`.

- `parsedScalarOpsKernel` ( private def, line 387): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedScalarOpsKernel : PTX.Kernel :=`.

- `example@392` ( example, line 392): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseKernel parsedScalarOpsText with | .ok kernel => kernel.entry == "parsed_scalar_ops" && kernel.regs.size == 9 && kernel.blocks.size == 1 | .error _ => false) = true := by`.

- `parsedScalarOpsWarp0` ( private def, line 398): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedScalarOpsWarp0 : WarpState :=`.

- `parsedScalarOpsState` ( private def, line 401): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedScalarOpsState : State :=`.

- `parsedScalarOpsFinalState` ( private def, line 406): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedScalarOpsFinalState : State :=`.

- `parsed_ptx_scalar_ops_kernel_run_functional` ( theorem, line 409): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_scalar_ops_kernel_run_functional : Reaches parsedScalarOpsState parsedScalarOpsFinalState ∧ lane0HasRegU32 "r3" 7 parsedScalarOpsFinalState = true ∧ lane0HasRegU32 "r9" 6 parsedScalarOpsFinalState = true ∧ lane0Terminated parsedScalarOpsFinalState = true := by`.

- `parsedBadScalarText` ( private def, line 419): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedBadScalarText : String :=`.

- `parsedBadScalarKernel` ( private def, line 429): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedBadScalarKernel : PTX.Kernel :=`.

- `example@434` ( example, line 434): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedBadScalarKernel with | .error (.typeMismatch .u32 .u64) => true | _ => false) = true := by`.

- `parsedBadBraText` ( private def, line 440): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedBadBraText : String :=`.

- `parsedBadBraKernel` ( private def, line 446): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedBadBraKernel : PTX.Kernel :=`.

- `example@451` ( example, line 451): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedBadBraKernel with | .error (.unknownBlock "missing_target") => true | _ => false) = true := by`.

- `parsedRealModuleText` ( private def, line 457): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedRealModuleText : String :=`.

- `parsedRealModule` ( private def, line 477): defines `parsedRealModule` in the example/regression layer. The signature is `private def parsedRealModule : PTX.Module :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def parsedRealModule : PTX.Module :=`.

- `parsedRealKernel` ( private def, line 482): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedRealKernel : PTX.Kernel :=`.

- `example@487` ( example, line 487): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.Parser.parseModule parsedRealModuleText with | .ok m => m.directives.size == 3 && m.memories.size == 1 && m.kernels.size == 1 | .error _ => false) = true := by`.

- `example@493` ( example, line 493): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : parsedRealKernel.params.size = 2 ∧ parsedRealKernel.regs.size = 4 ∧ parsedRealKernel.blocks.size = 1 := by`.

- `parsedRealParamBytes` ( private def, line 499): lays out kernel parameter bytes in `ParamMem` according to the lowered parameter offsets. Case-study state constructors use it to make `ld.param` instructions executable. Signature: `private def parsedRealParamBytes : ByteMem :=`.

- `parsedRealWarp0` ( private def, line 503): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedRealWarp0 : WarpState :=`.

- `parsedRealState` ( private def, line 506): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedRealState : State :=`.

- `parsedRealFinalState` ( private def, line 513): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedRealFinalState : State :=`.

- `parsed_real_style_param_copy_kernel_run_functional` ( theorem, line 516): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_real_style_param_copy_kernel_run_functional : Reaches parsedRealState parsedRealFinalState ∧ copyDstHasValue parsedRealFinalState = true ∧ lane0Terminated parsedRealFinalState = true := by`.

- `parsedUnaryCvtText` ( private def, line 524): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedUnaryCvtText : String :=`.

- `parsedUnaryCvtKernel` ( private def, line 539): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedUnaryCvtKernel : PTX.Kernel :=`.

- `parsedUnaryCvtWarp0` ( private def, line 544): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def parsedUnaryCvtWarp0 : WarpState :=`.

- `parsedUnaryCvtState` ( private def, line 547): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedUnaryCvtState : State :=`.

- `parsedUnaryCvtFinalState` ( private def, line 553): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def parsedUnaryCvtFinalState : State :=`.

- `parsed_ptx_unary_cvt_kernel_run_functional` ( theorem, line 556): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem parsed_ptx_unary_cvt_kernel_run_functional : Reaches parsedUnaryCvtState parsedUnaryCvtFinalState ∧ lane0HasRegU32 "r1" copyValue parsedUnaryCvtFinalState = true ∧ lane0HasRegS32 "s1" 5 parsedUnaryCvtFinalState = true ∧ lane0HasRegS32 "s2" 5 parsedUnaryCvtFinalState = true ∧`.

- `parsedBadModifierText` ( private def, line 568): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def parsedBadModifierText : String :=`.

- `parsedBadModifierKernel` ( private def, line 576): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def parsedBadModifierKernel : PTX.Kernel :=`.

- `example@581` ( example, line 581): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (match PTX.lowerKernelEnvChecked? parsedBadModifierKernel with | .error (.unsupportedModifier "volatile") => true | _ => false) = true := by`.

### `CLean/Examples/Saxpy.lean`

Imports: `CLean.Examples.Common`, `CLean.PTX.Bridge`, `CLean.Proof.Determinism`, `CLean.Proof.IsSingleWarpPres`, `CLean.Proof.LaneDecomposition`.

Layer role: SAXPY case-study layer.

- `saxpyKernelText` ( private def, line 45): stores a PTX source string used by parser/lowering examples or case-study kernels. The surrounding tests parse this text into a PTX AST and lower it to executable IR. Signature: `private def saxpyKernelText : String :=`.

- `example@91` ( example, line 91): is an anonymous regression check in the SAXPY case-study layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.parseAndLowerKernelOk? saxpyKernelText = true := by`.

- `saxpyKernel` ( private def, line 94): stores the parsed PTX kernel AST, defaulting on parse failure. It is the static source object later lowered into a `KernelEnv` for execution. Signature: `private def saxpyKernel : PTX.Kernel :=`.

- `example@99` ( example, line 99): is an anonymous regression check in the SAXPY case-study layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : PTX.lowerKernelSupported? saxpyKernel = true := by`.

- `saxpyXBase` ( def, line 102): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def saxpyXBase : Nat := 0`.

- `saxpyYBase` ( def, line 103): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def saxpyYBase : Nat := 128`.

- `saxpyRBase` ( def, line 104): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `def saxpyRBase : Nat := 256`.

- `saxpyParamBytesFor` ( private def, line 106): lays out kernel parameter bytes in `ParamMem` according to the lowered parameter offsets. Case-study state constructors use it to make `ld.param` instructions executable. Signature: `private def saxpyParamBytesFor (n : Nat) (alpha : Int) : ByteMem :=`.

- `saxpyGlobalBytesFor` ( private def, line 113): initializes global byte memory from input vectors or matrices. It is the data fixture read by modeled global loads. Signature: `private def saxpyGlobalBytesFor (xs ys : List Int) : ByteMem :=`.

- `saxpyWarpFor` ( private def, line 117): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def saxpyWarpFor (n : Nat) : WarpState :=`.

- `saxpyStateFor` ( def, line 121): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `def saxpyStateFor (n : Nat) (alpha : Int) (xs ys : List Int) : State :=`.

- `saxpyStateFor_isSingleWarp` ( private theorem, line 132): proves that the SAXPY initial state contains only the `(cta=0, warp=0)` warp. This is the structural starting point for the single-warp determinism bridge used by the general SAXPY correctness theorems. Signature: `private theorem saxpyStateFor_isSingleWarp (n : Nat) (alpha : Int) (xs ys : List Int) : IsSingleWarp (saxpyStateFor n alpha xs ys) := by`.

- `saxpyFuel` ( private def, line 218): defines the fixed fuel budget used for executable symbolic execution. It is chosen large enough to cover the modeled straight-line path or case-study termination bound. Signature: `private def saxpyFuel : Nat := 64`.

- `saxpyActiveLanes` ( private def, line 221): computes the active lane list from a launch size and proof that it fits in one warp. Lane-decomposition and per-lane proof statements quantify over this list. Signature: `private def saxpyActiveLanes (n : Nat) (hn : n ≤ 32) : List LaneId :=`.

- `saxpy_lanes_independent` ( private theorem, line 233): specializes the lane-decomposition shape to SAXPY active lanes, currently as an existence/reflexivity witness for each lane view at the chosen fuel. It marks where the future disjoint-write proof will plug into the SAXPY proof spine. Signature: `private theorem saxpy_lanes_independent (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) : ∀ j ∈ saxpyActiveLanes n hn, ∃ v_j : LaneDecomposition.LaneView, v_j = LaneDecomposition.LaneLocal (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) j := by`.

- `saxpy_canonical_per_lane_writes` ( private theorem, line 248): proves the canonical zero-input SAXPY write result for every supported `n ≤ 32` by case-splitting `n` and using `native_decide` on each lane. It is a proved executable baseline for the later universal per-lane theorem. Signature: `private theorem saxpy_canonical_per_lane_writes (n : Nat) (hn : n ≤ 32) : ∀ j : Nat, j < n → globalS32At? (StepMachine.runN saxpyFuel (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0))) saxpyRBase j = some 0 := by`.

- `saxpy_per_lane_writes` ( private theorem, line 268): proves a property in the SAXPY case-study layer. The name and signature identify the exact fact: `private theorem saxpy_per_lane_writes (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : ∀ j : Nat, j < n → globalS32At? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) saxpyRBase j = some (saxpyExpectedAt alpha xs ys j) := by`. Signature: `private theorem saxpy_per_lane_writes (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : ∀ j : Nat, j < n → globalS32At? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) saxpyRBase j = some (saxpyExpectedAt alpha xs ys j) := by`. **Status: admitted/in progress.**

- `runN_preserves_IsSingleWarp` ( private theorem, line 280): proves that repeated executable stepping preserves the single-warp structural invariant by induction on fuel and `step?_preserves_IsSingleWarp`. This lets SAXPY reason about `runN saxpyFuel init` as still single-warp. Signature: `private theorem runN_preserves_IsSingleWarp (K : Nat) {init : State} (hsw : IsSingleWarp init) : IsSingleWarp (StepMachine.runN K init) := by`.

- `saxpy_canonical_terminates` ( private theorem, line 297): proves by finite `n ≤ 32` case analysis and `native_decide` that the canonical zero-input SAXPY execution is stuck after `saxpyFuel`. It is the proved termination baseline used by the value-independence theorem. Signature: `private theorem saxpy_canonical_terminates (n : Nat) (hn : n ≤ 32) : (StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true := by`.

- `saxpy_termination_value_independent` ( private theorem, line 314): proves a property in the SAXPY case-study layer. The name and signature identify the exact fact: `private theorem saxpy_termination_value_independent (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) : (StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys))).isNone = true ↔ (StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true := by`. Signature: `private theorem saxpy_termination_value_independent (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) : (StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys))).isNone = true ↔ (StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n 0 (List.replicate n 0) (List.replicate n 0)))).isNone = true := by`. **Status: admitted/in progress.**

- `saxpy_step?_none_at_fuel` ( private theorem, line 324): defines the fixed fuel budget used for executable symbolic execution. It is chosen large enough to cover the modeled straight-line path or case-study termination bound. Signature: `private theorem saxpy_step?_none_at_fuel (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) : StepMachine.step? (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) = none := by`.

- `saxpy_machinefinal` ( private theorem, line 338): establishes a finality or final-state property used to close a correctness proof. It typically bridges executable stuckness, relational `MachineFinal`, and `runN` endpoints. Signature: `private theorem saxpy_machinefinal (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) : MachineFinal (StepMachine.runN saxpyFuel (saxpyStateFor n alpha xs ys)) := by`.

- `saxpy_runN_satisfies_post_pos` ( private theorem, line 352): proves a property in the SAXPY case-study layer. The name and signature identify the exact fact: `private theorem saxpy_runN_satisfies_post_pos (n : Nat) (hn : n ≤ 32) (_hnpos : 1 ≤ n) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : ∃ K : Nat, MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧ saxpyPost saxpyRBase n alpha xs ys`. Signature: `private theorem saxpy_runN_satisfies_post_pos (n : Nat) (hn : n ≤ 32) (_hnpos : 1 ≤ n) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : ∃ K : Nat, MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧ saxpyPost saxpyRBase n alpha xs ys`.

- `saxpy_step?_none_n_zero` ( private theorem, line 368): proves a property in the SAXPY case-study layer. The name and signature identify the exact fact: `private theorem saxpy_step?_none_n_zero (alpha : Int) (xs ys : List Int) : StepMachine.step? (saxpyStateFor 0 alpha xs ys) = none := by`. Signature: `private theorem saxpy_step?_none_n_zero (alpha : Int) (xs ys : List Int) : StepMachine.step? (saxpyStateFor 0 alpha xs ys) = none := by`.

- `saxpy_runN_satisfies_post` ( theorem, line 415): defines a postcondition or Boolean checker for the example/case-study output. Correctness theorems ultimately reduce to this predicate on the final state. Signature: `theorem saxpy_runN_satisfies_post (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : ∃ K : Nat, MachineFinal (StepMachine.runN K (saxpyStateFor n alpha xs ys)) ∧ saxpyPost saxpyRBase n alpha xs ys (StepMachine.runN K (saxpyStateFor n alpha xs ys)) := by`.

- `saxpy_partial_correct` ( theorem, line 445): states or proves partial correctness: every terminal state reachable from the initial state satisfies the specified postcondition. In case-study files, admitted versions identify the remaining proof obligation. Signature: `theorem saxpy_partial_correct (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : PartialCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by`.

- `saxpy_total_correct` ( theorem, line 471): states or proves total correctness: a terminal state exists and satisfies the specified postcondition. In case-study files, admitted versions identify the remaining termination-plus-post proof obligation. Signature: `theorem saxpy_total_correct (n : Nat) (hn : n ≤ 32) (alpha : Int) (xs ys : List Int) (hxs : xs.length = n) (hys : ys.length = n) : TotalCorrect (saxpyStateFor n alpha xs ys) (saxpyPost saxpyRBase n alpha xs ys) := by`.

### `CLean/Examples/Semantic.lean`

Imports: `CLean.Examples.Common`.

Layer role: example/regression layer.

- `baseWarp` ( private def, line 6): defines a fixed byte base address used by examples or postconditions. Keeping bases named makes memory layout assumptions explicit. Signature: `private def baseWarp : WarpState :=`.

- `baseCTA` ( private def, line 9): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `private def baseCTA : CTAState :=`.

- `exampleBlock` ( private def, line 12): defines `exampleBlock` in the example/regression layer. The signature is `private def exampleBlock : Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def exampleBlock : Block :=`.

- `exampleAssign` ( private def, line 17): defines `exampleAssign` in the example/regression layer. The signature is `private def exampleAssign : GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def exampleAssign : GInstr :=`.

- `exampleState` ( private def, line 20): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def exampleState : State :=`.

- `afterAssignState` ( private def, line 24): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterAssignState : State :=`.

- `afterTerminateState` ( private def, line 29): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterTerminateState : State :=`.

- `cvtaLane0` ( private def, line 34): defines `cvtaLane0` in the example/regression layer. The signature is `private def cvtaLane0 : LaneState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def cvtaLane0 : LaneState :=`.

- `cvtaWarp` ( private def, line 37): defines `cvtaWarp` in the example/regression layer. The signature is `private def cvtaWarp : WarpState :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def cvtaWarp : WarpState :=`.

- `cvtaCTA` ( private def, line 40): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `private def cvtaCTA : CTAState :=`.

- `cvtaState` ( private def, line 43): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def cvtaState : State :=`.

- `afterCvtaState` ( private def, line 47): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterCvtaState : State :=`.

- `afterIsspacepState` ( private def, line 52): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterIsspacepState : State :=`.

- `example@57` ( example, line 57): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).regs = ["a", "b"] := by`.

- `example@60` ( example, line 60): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.rvalueReadSet (.triop .selp (.reg "a") (.reg "b") (.pred "p"))).preds = ["p"] := by`.

- `example@63` ( example, line 63): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.stepInstr? exampleState 0 0 { instr := .assignReg "r1" (.imm (.u32 7)) }).isSome = true := by`.

- `example@66` ( example, line 66): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.stepTerminator? afterAssignState 0 0 .terminate).isSome = true := by`.

- `example@69` ( example, line 69): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : State.wf? exampleState = true := by`.

- `example@72` ( example, line 72): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : Helpers.lockstepRunnable? baseWarp = true := by`.

- `example@75` ( example, line 75): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.stepInstr? cvtaState 0 0 { instr := .cvta "gp" .global (.reg "p") }).isSome = true := by`.

- `example@78` ( example, line 78): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (Helpers.stepInstr? afterCvtaState 0 0 { instr := .isspacep "q" .global (.reg "gp") }).isSome = true := by`.

- `example@81` ( example, line 81): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : ∃ st', StepInstr exampleState 0 0 exampleAssign st' := by`.

- `example_step_assign` ( private theorem, line 91): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem example_step_assign : StepMachine exampleState afterAssignState := by`. Signature: `private theorem example_step_assign : StepMachine exampleState afterAssignState := by`.

- `example@94` ( example, line 94): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : ∃ st', StepMachine exampleState st' :=`.

- `example_step_terminate` ( private theorem, line 97): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem example_step_terminate : StepMachine afterAssignState afterTerminateState := by`. Signature: `private theorem example_step_terminate : StepMachine afterAssignState afterTerminateState := by`.

- `toy_assign_kernel_functional` ( theorem, line 101): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_assign_kernel_functional : ∃ st1 st2, StepMachine exampleState st1 ∧ StepMachine st1 st2 ∧ lane0HasR1Seven st1 = true ∧ lane0Terminated st2 = true := by`.

- `exampleRunFinalState` ( private def, line 111): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def exampleRunFinalState : State :=`.

- `toy_assign_kernel_run_functional` ( theorem, line 114): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_assign_kernel_run_functional : Reaches exampleState exampleRunFinalState ∧ lane0HasR1Seven exampleRunFinalState = true ∧ lane0Terminated exampleRunFinalState = true := by`.

- `example@122` ( example, line 122): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.traceN 2 exampleState).length = 3 := by`.

- `example@125` ( example, line 125): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.runN? 2 exampleState).isSome = true := by`.

- `example@128` ( example, line 128): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0HasR1Seven afterAssignState = true := by`.

- `example@131` ( example, line 131): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0Terminated afterTerminateState = true := by`.

- `example@134` ( example, line 134): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0HasGlobalAddr afterCvtaState = true := by`.

- `example@137` ( example, line 137): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0PredQTrue afterIsspacepState = true := by`.

- `copyLoad` ( private def, line 139): defines `copyLoad` in the example/regression layer. The signature is `private def copyLoad : GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def copyLoad : GInstr :=`.

- `copyStore` ( private def, line 142): defines `copyStore` in the example/regression layer. The signature is `private def copyStore : GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def copyStore : GInstr :=`.

- `copyBlock` ( private def, line 146): defines `copyBlock` in the example/regression layer. The signature is `private def copyBlock : Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def copyBlock : Block :=`.

- `copyWarp0` ( private def, line 151): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def copyWarp0 : WarpState :=`.

- `copyCTA0` ( private def, line 154): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `private def copyCTA0 : CTAState :=`.

- `copyState` ( private def, line 157): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def copyState : State :=`.

- `copyAfterLoadState` ( private def, line 162): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def copyAfterLoadState : State :=`.

- `copyAfterStoreState` ( private def, line 167): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def copyAfterStoreState : State :=`.

- `copyAfterTerminateState` ( private def, line 172): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def copyAfterTerminateState : State :=`.

- `copyLane0Loaded` ( private def, line 177): defines `copyLane0Loaded` in the example/regression layer. The signature is `private def copyLane0Loaded (st : State) : Bool :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def copyLane0Loaded (st : State) : Bool :=`.

- `copy_step_load` ( private theorem, line 185): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem copy_step_load : StepMachine copyState copyAfterLoadState := by`. Signature: `private theorem copy_step_load : StepMachine copyState copyAfterLoadState := by`.

- `copy_step_store` ( private theorem, line 188): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem copy_step_store : StepMachine copyAfterLoadState copyAfterStoreState := by`. Signature: `private theorem copy_step_store : StepMachine copyAfterLoadState copyAfterStoreState := by`.

- `copy_step_terminate` ( private theorem, line 191): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem copy_step_terminate : StepMachine copyAfterStoreState copyAfterTerminateState := by`. Signature: `private theorem copy_step_terminate : StepMachine copyAfterStoreState copyAfterTerminateState := by`.

- `toy_copy_kernel_functional` ( theorem, line 194): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_copy_kernel_functional : ∃ st1 st2 st3, StepMachine copyState st1 ∧ StepMachine st1 st2 ∧ StepMachine st2 st3 ∧ copyLane0Loaded st1 = true ∧ copyDstHasValue st2 = true ∧ copyLane0Terminated st3 = true := by`.

- `copyRunFinalState` ( private def, line 208): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def copyRunFinalState : State :=`.

- `toy_copy_kernel_run_functional` ( theorem, line 211): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_copy_kernel_run_functional : Reaches copyState copyRunFinalState ∧ copyLane0Loaded copyRunFinalState = true ∧ copyDstHasValue copyRunFinalState = true ∧ copyLane0Terminated copyRunFinalState = true := by`.

- `example@221` ( example, line 221): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.traceN 3 copyState).length = 4 := by`.

- `example@224` ( example, line 224): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.runN? 3 copyState).isSome = true := by`.

- `example@227` ( example, line 227): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : copyLane0Loaded copyAfterLoadState = true := by`.

- `example@230` ( example, line 230): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : copyDstHasValue copyAfterStoreState = true := by`.

- `example@233` ( example, line 233): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : copyLane0Terminated copyAfterTerminateState = true := by`.

- `barrierInstr` ( private def, line 236): defines `barrierInstr` in the example/regression layer. The signature is `private def barrierInstr : GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def barrierInstr : GInstr :=`.

- `barrierAssign` ( private def, line 239): defines `barrierAssign` in the example/regression layer. The signature is `private def barrierAssign : GInstr :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def barrierAssign : GInstr :=`.

- `barrierBlock` ( private def, line 242): defines `barrierBlock` in the example/regression layer. The signature is `private def barrierBlock : Block :=`, and the declaration supplies a reusable executable or propositional component for the layers above it. Signature: `private def barrierBlock : Block :=`.

- `barrierWarp0` ( private def, line 247): constructs the initial warp used by an example or case study, setting lane PCs and active masks so the executable semantics starts at the kernel entry block. Signature: `private def barrierWarp0 : WarpState :=`.

- `barrierCTA0` ( private def, line 250): constructs or identifies the CTA-level component used by an example. It packages the warp map, shared memory, or barrier setup needed by the state fixture. Signature: `private def barrierCTA0 : CTAState :=`.

- `barrierState` ( private def, line 254): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def barrierState : State :=`.

- `afterBarrierState` ( private def, line 258): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterBarrierState : State :=`.

- `afterBarrierAssignState` ( private def, line 263): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterBarrierAssignState : State :=`.

- `afterBarrierTerminateState` ( private def, line 268): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def afterBarrierTerminateState : State :=`.

- `barrier_step_release` ( private theorem, line 273): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem barrier_step_release : StepMachine barrierState afterBarrierState := by`. Signature: `private theorem barrier_step_release : StepMachine barrierState afterBarrierState := by`.

- `barrier_step_assign` ( private theorem, line 276): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem barrier_step_assign : StepMachine afterBarrierState afterBarrierAssignState := by`. Signature: `private theorem barrier_step_assign : StepMachine afterBarrierState afterBarrierAssignState := by`.

- `barrier_step_terminate` ( private theorem, line 279): proves a property in the example/regression layer. The name and signature identify the exact fact: `private theorem barrier_step_terminate : StepMachine afterBarrierAssignState afterBarrierTerminateState := by`. Signature: `private theorem barrier_step_terminate : StepMachine afterBarrierAssignState afterBarrierTerminateState := by`.

- `toy_barrier_kernel_functional` ( theorem, line 282): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_barrier_kernel_functional : ∃ st1 st2 st3, StepMachine barrierState st1 ∧ StepMachine st1 st2 ∧ StepMachine st2 st3 ∧ lane0RunningAt ("barrier", 1) st1 = true ∧ barrier0Released st1 = true ∧ lane0HasR1Seven st2 = true ∧ lane0Terminated st3 = true := by`.

- `barrierRunFinalState` ( private def, line 298): constructs a concrete machine state for an example or case study. It installs the lowered kernel environment, initial memories, and the CTA/warp/lane structure needed by `runN`. Signature: `private def barrierRunFinalState : State :=`.

- `toy_barrier_kernel_run_functional` ( theorem, line 301): is an executable regression theorem for a concrete kernel or parsed PTX snippet. It runs the modeled machine for a fixed fuel and proves the expected register, memory, or termination postconditions, usually with `native_decide`. Signature: `theorem toy_barrier_kernel_run_functional : Reaches barrierState barrierRunFinalState ∧ barrier0Released barrierRunFinalState = true ∧ lane0HasR1Seven barrierRunFinalState = true ∧ lane0Terminated barrierRunFinalState = true := by`.

- `example@311` ( example, line 311): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.traceN 3 barrierState).length = 4 := by`.

- `example@314` ( example, line 314): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : (StepMachine.runN? 3 barrierState).isSome = true := by`.

- `example@317` ( example, line 317): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0RunningAt ("barrier", 1) afterBarrierState = true := by`.

- `example@320` ( example, line 320): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : barrier0Released afterBarrierState = true := by`.

- `example@323` ( example, line 323): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0HasR1Seven afterBarrierAssignState = true := by`.

- `example@326` ( example, line 326): is an anonymous regression check in the example/regression layer. It confirms a concrete parse/lower/execute/simplification fact without introducing a named API theorem. Signature: `example : lane0Terminated afterBarrierTerminateState = true := by`.
