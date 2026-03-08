Yes — that’s the implementation order I’d use.

**1. Define the concrete semantic state.**
**2. Define the LeanIR instruction/effect vocabulary that acts on that state.**
**3. Add supported ops one by one as semantics over that state.**
**4. Build a PTX parser that parses raw syntax into PTX AST.**
**5. Build a PTX→LeanIR translator that lowers supported PTX instructions into your semantic ops.**

That is the clean pipeline.

The PTX docs are especially useful for step 1 because PTX already organizes the machine around **state spaces** and pointer/state-space conversions. PTX distinguishes `.const`, `.global`, `.local`, `.shared`, and `.param`, and `cvta` / `isspacep` exist specifically to move between or test these spaces. Kernel parameters can also carry `.ptr` attributes telling you which state space they point into and what alignment they assume.   

So I’d make the design concrete like this.

## The state you should define now

You do **not** want a “PTX text interpreter.” You want a semantic machine state.

At minimum:

```text
State =
  { global    : GlobalMem
  , const     : ConstMem
  , param     : ParamMem
  , ctas      : CTAId → CTAState
  , kernelEnv : KernelEnv
  }
```

and then:

```text
CTAState =
  { shared    : SharedMem
  , warps     : WarpId → WarpState
  , barrier   : BarrierState
  }
```

and then:

```text
WarpState =
  { lanes      : LaneId → LaneState
  , activeMask : BitVec 32
  , exitedMask : BitVec 32
  }
```

and then:

```text
LaneState =
  { regs   : Reg → Value
  , preds  : PredReg → Bool
  , local  : LocalMem
  , pc     : BlockLabel
  , status : Running | Blocked | Exited
  }
```

That gives you exactly the ingredients suggested by PTX’s state-space model:

* per-lane register/predicate state,
* per-thread local memory,
* per-CTA shared memory,
* global/const/param spaces,
* and enough warp structure for `shfl.sync`-style instructions. PTX explicitly defines `shfl.sync` as warp-scoped, mask-participating, and waiting for the specified participating lanes.  

## The address model

This is the next critical thing. Don’t make addresses plain integers only. PTX itself treats state-space identity as semantically important, and `cvta` / `isspacep` are exactly about that. 

So define:

```text
Addr =
  | Global  offset
  | Shared  cta offset
  | Local   cta lane offset
  | Param   offset
  | Const   offset
  | Generic tag offset
```

For implementation, `Generic` can be a tagged generic address rather than a raw flat integer at first. That is much easier to reason about than trying to perfectly emulate PTX’s address windows from day one.

This also lines up with PTX kernel parameter attributes: if a parameter is declared `.ptr.global.align 16`, your initial `param` state should know that this parameter semantically denotes a global-space pointer with 16-byte alignment; if `.ptr` has no space, treat it as generic. 

## The value model

Your `Value` type needs to be rich enough for the PTX families you want:

```text
Value =
  | U32 | U64 | S32 | S64
  | F16 | BF16 | F32 | F64
  | Pred
  | GenericAddr
  | Frag fragTy
```

That’s enough for:

* scalar integer/float ops,
* predicates,
* generic/state-space addresses,
* tensor-core fragments.

This is the big thing you were circling around earlier: once `Value` and `State` are right, adding lots of PTX ops becomes routine.

## The instruction split

I would split LeanIR into **pure ops** and **effects**.

### Pure ops

These are the big scalar PTX families:

* integer arithmetic
* float arithmetic
* mixed precision arithmetic
* comparisons
* selections
* logic / bitwise
* shifts
* conversions / moves

These don’t need new state. They just compute values.

### Effects

These are the things that actually mutate or synchronize the machine:

* `Load`
* `Store`
* `BarrierCTA`
* `WarpOp`
* `Atomic`
* `MMA`
* `Exit`
* control flow

That way, most PTX instructions become:

* parse instruction,
* translate into either `Assign dst (PureOp ...)`
* or an effectful instruction.

That is the right abstraction boundary.

## What the parser stack should be

You were exactly right that there are really two parser layers.

### Layer 1: PTX parser

This is syntax only:

* directives
* declarations
* labels
* operands
* instruction opcode + modifiers

Example: parse

```text
shfl.sync.down.b32 d|p, a, b, c, membermask;
```

into a PTX AST node that still knows it is a PTX shuffle instruction with mode/modifiers.

PTX syntax and semantics for `shfl.sync` are explicit enough to do this cleanly. 

### Layer 2: PTX→LeanIR translator

This is where you decide:

* do we support this instruction?
* how does it lower into semantic primitives?

For example:

* `add.s32` → `Assign dst (BinOp IAdd src1 src2)`
* `setp.lt.f32` → `AssignPred p (CmpOp FLt a b)`
* `ld.global.f32` → `Load dst Global F32 addr`
* `shfl.sync.down.b32` → `WarpOp (Shfl Down ...)`

And for unsupported instructions, reject here with a good error.

## What “supporting an operation” means

For each new PTX op, you need three things:

1. **Parse shape**
   Recognize the opcode/modifiers/operand form.

2. **Typing rule**
   What source/destination types are legal?

3. **Semantic clause**
   How does it update state?

That’s it.

For scalar ops, the semantic clause is usually tiny. For example:

* read operand values from registers/immediates,
* compute result,
* write destination reg/pred.

For `shfl.sync`, it is warp-scoped and depends on member masks and lane IDs, which PTX spells out pretty explicitly: participating non-exited threads in the member mask wait together, each computes a source lane according to mode and operands, then copies the chosen value if valid.  

## What I would implement first

In this exact order:

### Phase 1

Define:

* `AddrSpace`
* `Addr`
* `Value`
* `State`
* `PureOp`
* `Instr`

Then implement:

* register read/write
* memory read/write
* one-step semantics for `Assign`, `Load`, `Store`, `Branch`, `Exit`

### Phase 2

Add PTX parser support for:

* registers, predicates, immediates
* labels
* scalar arithmetic / compare / move / cvt
* `ld/st`
* basic branches

At this point you can already execute a ton of PTX.

### Phase 3

Add:

* `.ptr` parameter metadata
* `cvta`
* `isspacep`

These are important because PTX explicitly uses them to connect generic and non-generic address spaces.  

### Phase 4

Add warp ops:

* `shfl.sync`
* ballot/vote later

This is where your warp-structured state pays off.

### Phase 5

Add atomics and MMA.

That’s a real system.

## The main caution

The only thing I’d push back on is the phrase “just a parser.”

The **syntax parser** is just a parser.
The **translator** is where the semantics lives.

So the real pipeline is:

```text
PTX text
  → PTX AST
  → normalized PTX AST
  → LeanIR
  → step semantics over State
```

That normalization stage is important because PTX surface syntax is more annoying than the semantics you actually want to reason about.

The shortest summary is:

> Yes: first define the state, then add supported ops for that state, then implement PTX parsing and translation.
> The PTX state-space model should directly inform your `State`, `Addr`, and parameter typing design.  

Next I’d write the actual Lean datatype sketch for `State`, `Addr`, `Value`, `PureOp`, and `Instr`, because once those are fixed, the rest becomes straightforward engineering.



Absolutely. Here’s the concrete sketch I’d use for the **first real implementation**.

The goal is:

* define a **semantic state** that is strong enough for

  * scalar PTX ops
  * memory state spaces
  * warp intrinsics
  * barriers
  * atomics
  * MMA / tensor-core-style primitives
* keep the instruction language **extensible**
* make PTX lowering mostly mechanical

I’ll write this in a Lean-ish style, but keep it slightly schematic so we can still change details.

---

# 1) Core types

## Scalar / element types

```lean
inductive ScalarTy where
  | pred
  | u8   | u16  | u32  | u64
  | s8   | s16  | s32  | s64
  | b8   | b16  | b32  | b64
  | f16  | bf16 | f32  | f64
  deriving DecidableEq, Repr
```

I’d include the smaller integer/bit types now even if you do not fully support every op on them yet.
That way your typing surface is future-proof.

---

## Address spaces

```lean
inductive AddrSpace where
  | global
  | shared
  | local
  | param
  | const
  | generic
  deriving DecidableEq, Repr
```

---

## Thread / warp / CTA identifiers

```lean
abbrev CTAId   := Nat
abbrev WarpId  := Nat
abbrev LaneId  := Fin 32
abbrev RegName := String
abbrev PredName := String
abbrev BlockLabel := String
```

Later you may want more structured IDs than `String`, but for the first implementation, `String` is fine.

---

# 2) Values

This is one of the most important design choices.

```lean
inductive FragTy where
  | mma_m16n16k16_f16_f16_f32
  | mma_m16n8k16_f16_f16_f32
  | mma_m16n16k16_bf16_bf16_f32
  deriving DecidableEq, Repr
```

```lean
inductive GenericAddr where
  | mk (space : AddrSpace) (offset : Nat)
  deriving DecidableEq, Repr
```

```lean
inductive Value where
  | pred  (b : Bool)

  | u8    (x : UInt8)
  | u16   (x : UInt16)
  | u32   (x : UInt32)
  | u64   (x : UInt64)

  | s8    (x : Int)
  | s16   (x : Int)
  | s32   (x : Int)
  | s64   (x : Int)

  | b8    (x : UInt8)
  | b16   (x : UInt16)
  | b32   (x : UInt32)
  | b64   (x : UInt64)

  | f16   (bits : UInt16)   -- store bit-pattern at first
  | bf16  (bits : UInt16)
  | f32   (x : Float)
  | f64   (x : Float)

  | gaddr (a : GenericAddr)

  | frag  (ty : FragTy) (payload : Array Value)
  deriving Repr
```

## Why this shape?

A few important points:

### `f16` / `bf16`

I would initially store these as **bit-patterns**, not mathematical values.
Then define helper functions:

* `decodeF16 : UInt16 → Float`
* `encodeF16 : Float → UInt16`
* same for BF16

That gives you flexibility:

* executable semantics can use approximate decoding/encoding
* proof semantics can later abstract them if needed

### `frag`

Tensor-core fragments are first-class values.
This is the cleanest way to keep MMA out of the “actual PTX register mess.”

---

# 3) Addresses

This is where the PTX state-space story gets baked into the semantics.

```lean
inductive Addr where
  | global  (offset : Nat)
  | shared  (cta : CTAId) (offset : Nat)
  | local   (cta : CTAId) (warp : WarpId) (lane : LaneId) (offset : Nat)
  | param   (offset : Nat)
  | const   (offset : Nat)
  | generic (space : AddrSpace) (offset : Nat)
  deriving DecidableEq, Repr
```

This is intentionally **semantic**, not raw machine physical addressing.

For the first version, I strongly prefer this over flattening everything into a raw `u64`.
You can still support PTX `cvta` / `isspacep` by converting to and from `Addr.generic`.

---

# 4) Memory

Use byte-addressed memory, not typed memory.

```lean
abbrev Byte := UInt8
abbrev ByteMem := Std.HashMap Nat Byte
```

Then define typed loads/stores as functions on `ByteMem`.

```lean
structure GlobalMem where
  bytes : ByteMem
  deriving Repr

structure SharedMem where
  bytes : ByteMem
  deriving Repr

structure LocalMem where
  bytes : ByteMem
  deriving Repr

structure ParamMem where
  bytes : ByteMem
  deriving Repr

structure ConstMem where
  bytes : ByteMem
  deriving Repr
```

---

# 5) Lane / warp / CTA state

## Lane state

```lean
inductive LaneStatus where
  | running
  | blockedBarrier
  | exited
  deriving DecidableEq, Repr
```

```lean
structure LaneState where
  regs   : Std.HashMap RegName Value
  preds  : Std.HashMap PredName Bool
  local  : LocalMem
  pc     : BlockLabel
  status : LaneStatus
  deriving Repr
```

---

## Warp state

```lean
structure WarpState where
  lanes      : Array LaneState          -- length 32
  activeMask : UInt32
  exitedMask : UInt32
  deriving Repr
```

You can enforce the array length invariant later if you want a dependent type.
For implementation, fixed-size-by-convention is easier at first.

---

## Barrier state

Since you are not doing async yet, keep CTA barriers simple.

```lean
structure BarrierInstance where
  epoch   : Nat
  arrived : Std.HashSet (WarpId × LaneId)
  expectedCount : Nat
  deriving Repr
```

```lean
structure BarrierState where
  bars : Std.HashMap Nat BarrierInstance   -- barrier id ↦ state
  deriving Repr
```

For now, barrier id can just be `Nat`.

---

## Atomic ghost state

This is optional for execution, but extremely useful for proving.

```lean
inductive AtomicOp where
  | add | min | max | exch | cas
  deriving DecidableEq, Repr
```

```lean
structure AtomicEvent where
  cta    : CTAId
  warp   : WarpId
  lane   : LaneId
  op     : AtomicOp
  addr   : Addr
  before : Option Value
  arg    : Option Value
  after  : Option Value
  deriving Repr
```

```lean
structure AtomicState where
  log : Array AtomicEvent
  deriving Repr
```

---

## CTA state

```lean
structure CTAState where
  shared  : SharedMem
  warps   : Std.HashMap WarpId WarpState
  barrier : BarrierState
  deriving Repr
```

---

# 6) Kernel environment and global runtime state

## Grid / special register context

```lean
structure Dim3 where
  x : Nat
  y : Nat
  z : Nat
  deriving DecidableEq, Repr
```

```lean
structure GridCtx where
  gridDim  : Dim3
  blockDim : Dim3
  deriving DecidableEq, Repr
```

---

## Kernel parameter metadata

This is important because PTX params may be pointers into specific spaces.

```lean
structure ParamInfo where
  name      : String
  ty        : ScalarTy
  isPtr     : Bool
  ptrSpace? : Option AddrSpace
  align     : Nat := 1
  offset    : Nat
  size      : Nat
  deriving Repr
```

---

## Shared declarations

```lean
structure SharedDecl where
  name   : String
  size   : Nat
  align  : Nat := 1
  offset : Nat
  deriving Repr
```

---

## Kernel environment

```lean
structure KernelEnv where
  entry       : String
  gridCtx     : GridCtx
  params      : Array ParamInfo
  sharedDecls : Array SharedDecl
  blocks      : Std.HashMap BlockLabel Block
  deriving Repr
```

---

## Whole machine state

```lean
structure State where
  kernelEnv : KernelEnv
  global    : GlobalMem
  const     : ConstMem
  param     : ParamMem
  ctas      : Std.HashMap CTAId CTAState
  atomics   : AtomicState
  deriving Repr
```

That is the concrete semantic state I’d start with.

---

# 7) Pure operations

This is where most PTX instructions live.

## Unary ops

```lean
inductive UnaryOp where
  | mov
  | neg
  | abs
  | bitnot
  | cvt (dst : ScalarTy)
  deriving DecidableEq, Repr
```

## Binary ops

```lean
inductive BinaryOp where
  | add | sub | mul | div
  | rem
  | min | max
  | and | or | xor
  | shl | shr
  | seteq | setne | setlt | setle | setgt | setge
  deriving DecidableEq, Repr
```

## Ternary ops

```lean
inductive TernaryOp where
  | mad
  | fma
  | selp
  deriving DecidableEq, Repr
```

Then:

```lean
inductive RValue where
  | imm  (v : Value)
  | reg  (r : RegName)
  | pred (p : PredName)
  | special_tid_x
  | special_tid_y
  | special_tid_z
  | special_ctaid_x
  | special_ctaid_y
  | special_ctaid_z
  | special_ntid_x
  | special_ntid_y
  | special_ntid_z
  | special_nctaid_x
  | special_nctaid_y
  | special_nctaid_z
  | unop  (op : UnaryOp) (a : RValue)
  | binop (op : BinaryOp) (a b : RValue)
  | triop (op : TernaryOp) (a b c : RValue)
  deriving Repr
```

This is the crucial split:

* most PTX ops become `RValue`
* only the interesting things remain effects

---

# 8) Effectful instructions

## Memory ops

```lean
inductive MemSpace where
  | global | shared | local | param | const
  deriving DecidableEq, Repr
```

```lean
structure TypedAddr where
  space : MemSpace
  ty    : ScalarTy
  addr  : RValue
  deriving Repr
```

---

## Warp operations

```lean
inductive ShflMode where
  | up | down | bfly | idx
  deriving DecidableEq, Repr
```

```lean
inductive WarpOp where
  | activemask (dst : RegName)
  | shflSync
      (mode : ShflMode)
      (dst : RegName)
      (src : RValue)
      (laneOrDelta : RValue)
      (clamp : RValue)
      (memberMask : RValue)
  | ballotSync
      (dst : RegName)
      (pred : RValue)
      (memberMask : RValue)
  | matchAnySync
      (dst : RegName)
      (src : RValue)
      (memberMask : RValue)
  | reduxSync
      (dst : RegName)
      (op : BinaryOp)
      (src : RValue)
      (memberMask : RValue)
  deriving Repr
```

This covers a lot already.

---

## MMA operations

```lean
structure MMASpec where
  fragTy : FragTy
  deriving Repr
```

```lean
structure MMAInstr where
  dst  : RegName
  a    : RegName
  b    : RegName
  c    : RegName
  spec : MMASpec
  deriving Repr
```

You may later want multiple destination regs, depending on how you encode fragments.
For now, one fragment-valued register is simpler.

---

## Control-flow terminators

```lean
inductive Terminator where
  | br    (label : BlockLabel)
  | cbr   (cond : RValue) (tLabel fLabel : BlockLabel)
  | exit
  deriving Repr
```

---

## Instruction type

```lean
inductive Instr where
  | assignReg  (dst : RegName)  (rhs : RValue)
  | assignPred (dst : PredName) (rhs : RValue)

  | load   (dst : RegName) (src : TypedAddr)
  | store  (dst : TypedAddr) (value : RValue)

  | barrierCTA (barId : Nat)

  | atomic
      (dst? : Option RegName)
      (op : AtomicOp)
      (addr : TypedAddr)
      (arg1 : RValue)
      (arg2? : Option RValue)   -- for CAS, etc.

  | warp   (op : WarpOp)

  | mma    (i : MMAInstr)
  deriving Repr
```

---

## Basic block

```lean
structure Block where
  label : BlockLabel
  body  : Array Instr
  term  : Terminator
  deriving Repr
```

This is the actual LeanIR core.

---

# 9) Semantic helper functions

You’ll want these immediately.

## Register / predicate access

```lean
def readReg (lane : LaneState) (r : RegName) : Option Value := ...
def writeReg (lane : LaneState) (r : RegName) (v : Value) : LaneState := ...

def readPred (lane : LaneState) (p : PredName) : Bool := ...
def writePred (lane : LaneState) (p : PredName) (b : Bool) : LaneState := ...
```

---

## Special register evaluation

```lean
def evalSpecial
  (grid : GridCtx) (cta : CTAId) (warp : WarpId) (lane : LaneId)
  : RValue → Option Value
```

This maps things like `special_tid_x` to concrete `u32` values.

---

## Address evaluation

```lean
def evalAddr
  (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId)
  (ta : TypedAddr) : Option Addr
```

This is where:

* the `RValue` computing the address is evaluated
* the appropriate state-space address is built

---

## Memory reads/writes

```lean
def readMem (st : State) (addr : Addr) (ty : ScalarTy) : Option Value := ...
def writeMem (st : State) (addr : Addr) (ty : ScalarTy) (v : Value) : Option State := ...
```

If you get only one thing right early, make it these two functions.

---

# 10) Step semantics

You do not need the full theorem-proving machinery first.
Just write executable semantics.

## Step result

```lean
inductive StepError where
  | badRegister
  | badPredicate
  | typeMismatch
  | invalidAddress
  | unsupported
  | barrierMismatch
  | mmaMismatch
  deriving Repr
```

```lean
abbrev SemM := Except StepError
```

---

## One-lane pure expression evaluation

```lean
def evalRValue
  (st : State) (cta : CTAId) (warp : WarpId) (lane : LaneId)
  (rv : RValue) : SemM Value := ...
```

This handles:

* immediates
* regs
* preds
* special registers
* unary/binary/ternary pure ops

---

## One instruction step

Because of warp ops, you really want instruction stepping at warp granularity.

```lean
def stepInstrWarp
  (st : State) (cta : CTAId) (warp : WarpId) (i : Instr) : SemM State := ...
```

### Why warp-level step?

Because:

* `shfl.sync`
* `ballotSync`
* `reduxSync`
* MMA-ish warp-cooperative operations

all naturally read multiple lanes together.

For plain scalar instructions, `stepInstrWarp` just maps a lane-local update across active lanes.

---

## Block stepping

```lean
def stepBlockWarp
  (st : State) (cta : CTAId) (warp : WarpId) : SemM State := ...
```

This:

* fetches the current block by `pc`
* executes instructions in order
* applies terminator
* updates PCs

You can initially assume all active lanes in a warp share the same `pc`.
That makes the implementation much simpler.

---

# 11) PTX parser boundary

Once the above is fixed, the PTX side becomes much cleaner.

You want:

## Raw PTX AST

Something like:

```lean
inductive PTXOperand ...
inductive PTXOpcode ...
structure PTXInstr ...
structure PTXBlock ...
structure PTXKernel ...
```

This is just syntax.

## PTX → LeanIR lowering

Then each supported PTX instruction lowers into one of:

* `assignReg`
* `assignPred`
* `load`
* `store`
* `barrierCTA`
* `atomic`
* `warp`
* `mma`

This is the point where you “imbue semantics.”

---

# 12) Suggested first implementation slice

I would build this in exactly this order:

## Slice A

Implement:

* `ScalarTy`, `Value`, `Addr`, `State`
* `RValue`
* `Instr.assignReg`, `load`, `store`
* `evalRValue`, `readMem`, `writeMem`

That already gives you:

* arithmetic
* compare/select
* conversions
* memory

## Slice B

Add:

* control flow / blocks / terminators
* warp-stepped execution
* branches

## Slice C

Add:

* `barrierCTA`
* `activemask`
* `shflSync`

## Slice D

Add:

* atomics
* MMA fragments + `mma`

That is already a serious system.

---

# 13) The main conceptual takeaway

This is the right decomposition:

* **State** models the semantic resources:

  * registers, preds, local/shared/global/param/const memory
  * warp structure
  * CTA barriers
  * atomic log
  * fragment values

* **RValue** handles the huge scalar PTX instruction surface.

* **Instr** handles the smaller set of stateful / collective effects.

That means once the state is fixed, adding more PTX instructions really does become mostly a matter of:

1. parse it,
2. type it,
3. lower it into `RValue` or `Instr`,
4. define the clause if it’s a new effect.

---

If you want, next I can do one of two things:

* write a more **Lean-compilable version** of these types, or
* take a few concrete PTX instructions (`add`, `setp`, `ld.global`, `shfl.sync.down`, `atom.add`, `mma`) and show exactly how they lower into this IR.
