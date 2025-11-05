import Lean
import Lean.Elab.Command
import Mathlib.Tactic
import Std.Data.HashMap
import SciLean.Data.DataArray
open Lean Lean.Elab Lean.Elab.Command Meta SciLean

namespace GpuDSL

universe u v

structure Dim3 where
  x : Nat
  y : Nat
  z : Nat
deriving Repr

-- structure GlobalArray (α : Type) where
--   data : Array α
-- deriving Repr

-- structure SharedArray (α : Type) where
--   data : Array α
-- deriving Repr

-- abbrev GlobalPtr (α : Type) := GlobalArray α

structure KernelCtx (Args : Type*) where
  threadIdx : Dim3
  blockIdx : Dim3
  blockDim : Dim3
  gridDim : Dim3
  args : Args


-- Heterogeneous value type for storing different types in KernelState
inductive KernelValue where
  | arrayFloat : Array Float → KernelValue
  | arrayInt : Array Int → KernelValue
  | arrayNat : Array Nat → KernelValue
  | float : Float → KernelValue
  | int : Int → KernelValue
  | nat : Nat → KernelValue
deriving Repr, Inhabited

structure KernelState where
  shared  : Std.HashMap Name KernelValue
  globals : Std.HashMap Name KernelValue

instance : Inhabited KernelState :=
  ⟨{ shared := ∅, globals := ∅ }⟩


abbrev KernelM (Args : Type) (α : Type) :=
  ReaderT (KernelCtx Args) (StateM KernelState) α


@[inline] def readCtx {Args} : KernelM Args (KernelCtx Args) :=
  fun ctx s => (ctx, s)

/-! ## Type class for converting to/from KernelValue -/

class ToKernelValue (α : Type) where
  toKernelValue : α → KernelValue

class FromKernelValue (α : Type) where
  fromKernelValue? : KernelValue → Option α

instance : ToKernelValue (Array Float) where
  toKernelValue := KernelValue.arrayFloat

instance : FromKernelValue (Array Float) where
  fromKernelValue?
    | .arrayFloat a => some a
    | _ => none

instance : ToKernelValue (Array Int) where
  toKernelValue := KernelValue.arrayInt

instance : FromKernelValue (Array Int) where
  fromKernelValue?
    | .arrayInt a => some a
    | _ => none

instance : ToKernelValue (Array Nat) where
  toKernelValue := KernelValue.arrayNat

instance : FromKernelValue (Array Nat) where
  fromKernelValue?
    | .arrayNat a => some a
    | _ => none

instance : ToKernelValue Float where
  toKernelValue := KernelValue.float

instance : FromKernelValue Float where
  fromKernelValue?
    | .float a => some a
    | _ => none

instance : ToKernelValue Int where
  toKernelValue := KernelValue.int

instance : FromKernelValue Int where
  fromKernelValue?
    | .int a => some a
    | _ => none

instance : ToKernelValue Nat where
  toKernelValue := KernelValue.nat

instance : FromKernelValue Nat where
  fromKernelValue?
    | .nat a => some a
    | _ => none

/-! ## Heaps: helpers to get/put arrays and scalars by name -/

@[inline] def setGlobal {Args α} [ToKernelValue α]
    (name : Name) (v : α) : KernelM Args Unit :=
  modify fun s => { s with globals := s.globals.insert name (ToKernelValue.toKernelValue v) }

@[inline] def getGlobal? {Args α} [FromKernelValue α]
    (name : Name) : KernelM Args (Option α) :=
  fun _ s =>
    match s.globals.get? name with
    | some val => (FromKernelValue.fromKernelValue? val, s)
    | none => (none, s)

@[inline] def setShared {Args α} [ToKernelValue α]
    (name : Name) (v : α) : KernelM Args Unit :=
  modify fun s => { s with shared := s.shared.insert name (ToKernelValue.toKernelValue v) }

@[inline] def getShared? {Args α} [FromKernelValue α]
    (name : Name) : KernelM Args (Option α) :=
  fun _ s =>
    match s.shared.get? name with
    | some val => (FromKernelValue.fromKernelValue? val, s)
    | none => (none, s)

/-- Read/write a *global* `Array α` element by name and index. -/
@[inline] def gReadAt {Args α} [Inhabited α] [FromKernelValue (Array α)]
    (name : Name) (i : Nat) : KernelM Args α := do
  let some (arr : Array α) ← getGlobal? name
    | panic! s!"global '{name}' not found or wrong type"
  pure <| arr[i]!

@[inline] def gWriteAt {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)]
    (name : Name) (i : Nat) (v : α) : KernelM Args Unit := do
  let some (arr : Array α) ← getGlobal? name
    | panic! s!"global '{name}' not found or wrong type"
  setGlobal name (arr.set! i v)

/-- Block-level barrier (no-op in the single-threaded CPU simulator). -/
@[inline] def barrier {Args} : KernelM Args Unit := pure ()

/-! ## Thread index helpers -/

@[inline] def globalIdxX {Args} : KernelM Args Nat := do
  let c ← readCtx
  pure (c.blockIdx.x * c.blockDim.x + c.threadIdx.x)

/-! ## Wrapper types for clean syntax -/

/-- Typed reference to a global array buffer -/
structure GlobalArray (α : Type) where
  name : Name

/-- Typed reference to a shared array buffer -/
structure SharedArray (α : Type) where
  name : Name

/-- Typed reference to a global scalar value -/
structure GlobalScalar (α : Type) where
  name : Name

/-- Typed reference to a shared scalar value -/
structure SharedScalar (α : Type) where
  name : Name

namespace GlobalArray

/-- Read from a global array at index i -/
@[inline] def get {Args α} [Inhabited α] [FromKernelValue (Array α)]
    (arr : GlobalArray α) (i : Nat) : KernelM Args α :=
  gReadAt arr.name i

/-- Write to a global array at index i -/
@[inline] def set {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)]
    (arr : GlobalArray α) (i : Nat) (v : α) : KernelM Args Unit :=
  gWriteAt arr.name i v

/-- Read entire global array -/
@[inline] def read {Args α} [FromKernelValue (Array α)]
    (arr : GlobalArray α) : KernelM Args (Array α) := do
  let some a ← getGlobal? arr.name
    | panic! s!"global '{arr.name}' not found"
  pure a

end GlobalArray

namespace SharedArray

/-- Read from a shared array at index i -/
@[inline] def get {Args α} [Inhabited α] [FromKernelValue (Array α)]
    (arr : SharedArray α) (i : Nat) : KernelM Args α := do
  let some (a : Array α) ← getShared? arr.name
    | panic! s!"shared '{arr.name}' not found"
  pure a[i]!

/-- Write to a shared array at index i -/
@[inline] def set {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)]
    (arr : SharedArray α) (i : Nat) (v : α) : KernelM Args Unit := do
  let some (a : Array α) ← getShared? arr.name
    | panic! s!"shared '{arr.name}' not found"
  setShared arr.name (a.set! i v)

end SharedArray

namespace GlobalScalar

/-- Read a global scalar value -/
@[inline] def get {Args α} [FromKernelValue α] [Inhabited α]
    (s : GlobalScalar α) : KernelM Args α := do
  let some v ← getGlobal? s.name
    | panic! s!"global scalar '{s.name}' not found"
  pure v

/-- Write a global scalar value -/
@[inline] def set {Args α} [ToKernelValue α]
    (s : GlobalScalar α) (v : α) : KernelM Args Unit :=
  setGlobal s.name v

end GlobalScalar

namespace SharedScalar

/-- Read a shared scalar value -/
@[inline] def get {Args α} [FromKernelValue α] [Inhabited α]
    (s : SharedScalar α) : KernelM Args α := do
  let some v ← getShared? s.name
    | panic! s!"shared scalar '{s.name}' not found"
  pure v

/-- Write a shared scalar value -/
@[inline] def set {Args α} [ToKernelValue α]
    (s : SharedScalar α) (v : α) : KernelM Args Unit :=
  setShared s.name v

end SharedScalar

/-! ## Helper functions for creating references -/

/-- Create a global array reference -/
@[inline] def global (name : Name) : GlobalArray α := ⟨name⟩

/-- Create a shared array reference -/
@[inline] def shared (name : Name) : SharedArray α := ⟨name⟩

/-- Create a global scalar reference -/
@[inline] def globalScalar (name : Name) : GlobalScalar α := ⟨name⟩

/-- Create a shared scalar reference -/
@[inline] def sharedScalar (name : Name) : SharedScalar α := ⟨name⟩

/-! ## CPU "runtime": grid/block interpreter -/

/-- Run one kernel body across the whole grid on CPU, threading state. -/
def runKernelCPU
    {Args : Type}
    (grid block : Dim3)
    (args : Args)
    (initState : KernelState)
    (body : KernelM Args Unit)
    : KernelState :=
  Id.run do
    let mut st := initState
    for bz in [0:grid.z] do
      for by_ in [0:grid.y] do
        for bx in [0:grid.x] do
          -- reset shared memory for each block
          st := { st with shared := ∅ }
          for tz in [0:block.z] do
            for ty_ in [0:block.y] do
              for tx in [0:block.x] do
                let ctx : KernelCtx Args :=
                  { threadIdx := ⟨tx, ty_, tz⟩
                    blockIdx  := ⟨bx, by_, bz⟩
                    blockDim  := block
                    gridDim   := grid
                    args }
                let (_, st') := (body ctx).run st
                st := st'
    st




-- /-! ## A fixed-length vector (for the host API) -/

-- def Vec (α : Type) (n : Nat) := Fin n → α
-- notation:100 α " ^[" n "]" => Vec α n

-- @[inline] def vecToArray {α} {n} (v : α^[n]) : Array α :=
--   Array.ofSubarray <| (List.ofFn (fun (i : Fin n) => v i)).toArray.toSubarray 0 n

-- @[inline] def arrayToVec {α} {n} (a : Array α) (h : a.size = n := by rfl) : α^[n] :=
--   fun i => a[i]

/-! ## Helper functions for cleaner syntax -/

/-- Helper to create a KernelState from arrays. Automatically wraps them in KernelValue. -/
def mkKernelState
    (globals : List (Name × KernelValue))
    (shared : List (Name × KernelValue) := []) : KernelState :=
  { globals := Std.HashMap.ofList globals
    shared := Std.HashMap.ofList shared }

/-- Helper to insert a Float array into globals -/
@[inline] def globalFloatArray (name : Name) (arr : Array Float) : Name × KernelValue :=
  (name, KernelValue.arrayFloat arr)

/-- Helper to insert an Int array into globals -/
@[inline] def globalIntArray (name : Name) (arr : Array Int) : Name × KernelValue :=
  (name, KernelValue.arrayInt arr)

/-- Helper to insert a scalar Float into globals -/
@[inline] def globalFloat (name : Name) (v : Float) : Name × KernelValue :=
  (name, KernelValue.float v)

/-! ## Notation for cleaner kernel access -/

/-- Notation to get args from context in one line -/
macro "getArgs" : term => `((·.args) <$> readCtx)

/-- Helper to create multiple GlobalArray references at once -/
def globals (names : List Name) : List (GlobalArray Float) :=
  names.map (fun n => global n)



end GpuDSL
