import Lean
import Lean.Elab.Command
import Lean.Data.Json
import Mathlib.Tactic.TypeStar
import Std.Data.HashMap

-- import SciLean.Data.DataArray
open Lean Lean.Elab Lean.Elab.Command Meta --SciLean

namespace GpuDSL

universe u v

structure Dim3 where
  x : Nat
  y : Nat
  z : Nat
deriving Repr, Inhabited

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

namespace KernelValue
/-- Merge two KernelValues element-wise for arrays, prefer v2 for scalars.
    For arrays: take max size and use v2[i] if available, else v1[i]. -/
def merge (v1 v2 : KernelValue) : KernelValue :=
  match v1, v2 with
  | arrayFloat a1, arrayFloat a2 =>
    -- Element-wise merge: prefer v2's elements when available
    let maxSize := max a1.size a2.size
    let result := Id.run do
      let mut arr := Array.mkEmpty maxSize
      for i in [:maxSize] do
        let val := if i < a2.size then a2[i]!
                  else if i < a1.size then a1[i]!
                  else 0.0
        arr := arr.push val
      return arr
    arrayFloat result
  | arrayInt a1, arrayInt a2 =>
    let maxSize := max a1.size a2.size
    let result := Id.run do
      let mut arr := Array.mkEmpty maxSize
      for i in [:maxSize] do
        let val := if i < a2.size then a2[i]!
                  else if i < a1.size then a1[i]!
                  else (0 : Int)
        arr := arr.push val
      return arr
    arrayInt result
  | arrayNat a1, arrayNat a2 =>
    let maxSize := max a1.size a2.size
    let result := Id.run do
      let mut arr := Array.mkEmpty maxSize
      for i in [:maxSize] do
        let val := if i < a2.size then a2[i]!
                  else if i < a1.size then a1[i]!
                  else 0
        arr := arr.push val
      return arr
    arrayNat result
  | _, scalar => scalar  -- For scalars, prefer v2
end KernelValue

/-- WriteBuffer collects all writes made by a thread during a barrier phase.
    For array elements: key is (arrayName, index), value is the element
    For scalars: key is (scalarName, 0), value is the scalar -/
structure WriteBuffer where
  sharedWrites : Std.HashMap (Name × Nat) KernelValue := ∅
  globalWrites : Std.HashMap (Name × Nat) KernelValue := ∅
deriving Inhabited

structure KernelState where
  shared  : Std.HashMap Name KernelValue
  globals : Std.HashMap Name KernelValue
  -- Write buffering for parallel execution simulation
  writeBuffer : WriteBuffer  -- Collects writes during phase execution
  isBuffering : Bool         -- If true, writes go to buffer; if false, writes go to state
  -- Barrier phase tracking for CPU simulation
  currentPhase : Nat      -- Which barrier phase we're currently executing
  threadBarrierCount : Nat  -- How many barriers this thread has hit in current execution
  hitBarrier : Bool        -- True if this thread hit a barrier beyond the current phase

instance : Inhabited KernelState :=
  ⟨{ shared := ∅, globals := ∅, writeBuffer := default, isBuffering := false,
     currentPhase := 0, threadBarrierCount := 0, hitBarrier := false }⟩


abbrev KernelM (Args : Type) (α : Type) :=
  ReaderT (KernelCtx Args) (StateT KernelState IO) α


@[inline] def readCtx {Args} : KernelM Args (KernelCtx Args) :=
  fun ctx s => pure (ctx, s)

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

attribute [ext] Dim3 KernelCtx WriteBuffer KernelState

/-! ## Heaps: helpers to get/put arrays and scalars by name -/

@[inline] def setGlobal {Args α} [ToKernelValue α]
    (name : Name) (v : α) : KernelM Args Unit := do
  let s ← get
  -- Skip operations if we've exceeded the current phase
  if s.hitBarrier then return ()

  if s.isBuffering then
    -- Only buffer writes in the active phase (snapshot already has previous phases)
    if s.threadBarrierCount == s.currentPhase then
      modify fun s =>
        { s with writeBuffer :=
          { s.writeBuffer with globalWrites :=
            s.writeBuffer.globalWrites.insert (name, 0) (ToKernelValue.toKernelValue v) } }
    -- else: skip (either replay section or beyond current phase)
  else
    -- Write directly to state (initialization, not buffering)
    modify fun s => { s with globals := s.globals.insert name (ToKernelValue.toKernelValue v) }

@[inline] def getGlobal? {Args α} [FromKernelValue α]
    (name : Name) : KernelM Args (Option α) :=
  fun _ s =>
    match s.globals.get? name with
    | some val => pure (FromKernelValue.fromKernelValue? val, s)
    | none => pure (none, s)

@[inline] def setShared {Args α} [ToKernelValue α]
    (name : Name) (v : α) : KernelM Args Unit := do
  let s ← get
  -- Skip operations if we've exceeded the current phase
  if s.hitBarrier then return ()

  if s.isBuffering then
    -- Only buffer writes in the active phase (snapshot already has previous phases)
    if s.threadBarrierCount == s.currentPhase then
      modify fun s =>
        { s with writeBuffer :=
          { s.writeBuffer with sharedWrites :=
            s.writeBuffer.sharedWrites.insert (name, 0) (ToKernelValue.toKernelValue v) } }
    -- else: skip (either replay section or beyond current phase)
  else
    -- Write directly to state (initialization, not buffering)
    modify fun s => { s with shared := s.shared.insert name (ToKernelValue.toKernelValue v) }

@[inline] def getShared? {Args α} [FromKernelValue α]
    (name : Name) : KernelM Args (Option α) :=
  fun _ s =>
    match s.shared.get? name with
    | some val => pure (FromKernelValue.fromKernelValue? val, s)
    | none => pure (none, s)

/-- Read/write a *global* `Array α` element by name and index. -/
@[inline] def gReadAt {Args α} [Inhabited α] [FromKernelValue (Array α)]
    (name : Name) (i : Nat) : KernelM Args α := do
  let some (arr : Array α) ← getGlobal? name
    | panic! s!"global '{name}' not found or wrong type"
  pure <| arr[i]!

@[inline] def gWriteAt {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)] [ToKernelValue α]
    (name : Name) (i : Nat) (v : α) : KernelM Args Unit := do
  let s ← get
  if s.hitBarrier then return ()  -- Skip if exceeded phase

  if s.isBuffering then
    -- Only buffer writes in the active phase (snapshot already has previous phases)
    if s.threadBarrierCount == s.currentPhase then
      modify fun s =>
        { s with writeBuffer :=
          { s.writeBuffer with globalWrites :=
            s.writeBuffer.globalWrites.insert (name, i) (ToKernelValue.toKernelValue v) } }
    -- else: skip (either replay section or beyond current phase)
  else
    -- Direct write to state (initialization, not buffering)
    let some (arr : Array α) ← getGlobal? name
      | panic! s!"global '{name}' not found or wrong type"
    setGlobal name (arr.set! i v)

/-- Block-level barrier. In CPU simulation, this marks synchronization points.
    Threads execute in phases to simulate barrier synchronization. In phase N,
    we execute until hitting barrier N+1. -/
@[inline] def barrier {Args} : KernelM Args Unit := do
  let s ← get
  let newCount := s.threadBarrierCount + 1
  set { s with threadBarrierCount := newCount }

  -- If we've hit the target barrier for this phase, mark it and stop
  if newCount > s.currentPhase then
    modify fun s => { s with hitBarrier := true }

/-! ## Thread index helpers -/

@[inline] def globalIdxX {Args} : KernelM Args Nat := do
  let c ← readCtx
  pure (c.blockIdx.x * c.blockDim.x + c.threadIdx.x)

@[inline] def globalIdxY {Args} : KernelM Args Nat := do
  let c ← readCtx
  pure (c.blockIdx.y * c.blockDim.y + c.threadIdx.y)

@[inline] def globalIdxZ {Args} : KernelM Args Nat := do
  let c ← readCtx
  pure (c.blockIdx.z * c.blockDim.z + c.threadIdx.z)


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

attribute [ext] GlobalArray GlobalScalar SharedArray SharedScalar

namespace GlobalArray

/-- Read from a global array at index i -/
@[inline] def get {Args α} [Inhabited α] [FromKernelValue (Array α)]
    (arr : GlobalArray α) (i : Nat) : KernelM Args α :=
  gReadAt arr.name i

/-- Write to a global array at index i -/
@[inline] def set {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)] [ToKernelValue α]
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
@[inline] def set {Args α} [FromKernelValue (Array α)] [ToKernelValue (Array α)] [ToKernelValue α]
    (arr : SharedArray α) (i : Nat) (v : α) : KernelM Args Unit := do
  let st ← MonadState.get
  if st.hitBarrier then return ()  -- Skip if exceeded phase

  if st.isBuffering then
    -- Only buffer writes in the active phase (snapshot already has previous phases)
    if st.threadBarrierCount == st.currentPhase then
      modify fun s =>
        { s with writeBuffer :=
          { s.writeBuffer with sharedWrites :=
            s.writeBuffer.sharedWrites.insert (arr.name, i) (ToKernelValue.toKernelValue v) } }
    -- else: skip (either replay section or beyond current phase)
  else
    -- Direct write to state (initialization, not buffering)
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
    (gs : GlobalScalar α) (v : α) : KernelM Args Unit :=
  setGlobal gs.name v

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
    (ss : SharedScalar α) (v : α) : KernelM Args Unit :=
  setShared ss.name v

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

/-- Run one kernel body across the whole grid on CPU with barrier synchronization.
    Uses write buffering to simulate parallel execution: all threads in a phase
    read the same memory state, and their writes are applied atomically at phase end. -/
def runKernelCPU
    {Args : Type}
    (grid block : Dim3)
    (args : Args)
    (initState : KernelState)
    (body : KernelM Args Unit)
    : IO KernelState := do
    let mut st := initState
    for bz in [0:grid.z] do
      for by_ in [0:grid.y] do
        for bx in [0:grid.x] do
          -- reset shared memory to initial state for each block
          st := { st with
            shared := initState.shared,
            writeBuffer := default,
            isBuffering := false,
            currentPhase := 0,
            threadBarrierCount := 0,
            hitBarrier := false
          }

          -- Execute threads in phases until all complete
          let mut phase : Nat := 0
          let mut anyBarrierHit := true

          while anyBarrierHit do
            anyBarrierHit := false

            -- Create snapshot for this phase (threads read from this)
            let snapshot := { st with
              writeBuffer := default,
              isBuffering := false
            }

            -- Collect write buffers from all threads
            let mut allBuffers := #[]

            -- Execute all threads for this phase
            for tz in [0:block.z] do
              for ty_ in [0:block.y] do
                for tx in [0:block.x] do
                  let ctx : KernelCtx Args :=
                    { threadIdx := ⟨tx, ty_, tz⟩
                      blockIdx  := ⟨bx, by_, bz⟩
                      blockDim  := block
                      gridDim   := grid
                      args }

                  -- Start with snapshot, enable buffering
                  let threadState := { snapshot with
                    writeBuffer := default,
                    isBuffering := true,
                    currentPhase := phase,
                    threadBarrierCount := 0,
                    hitBarrier := false
                  }

                  -- Execute thread body
                  let (_, finalState) ← (body ctx).run threadState

                  -- Check if thread hit a barrier this phase
                  if finalState.hitBarrier then
                    anyBarrierHit := true

                  -- Save this thread's write buffer
                  allBuffers := allBuffers.push finalState.writeBuffer

            -- Apply all write buffers to state
            -- Collect all writes: HashMap (Name × Nat) KernelValue contains individual writes
            -- Need to group by Name, rebuild arrays, then apply to state

            -- Collect all element writes per array
            let mut sharedElementWrites : Std.HashMap (Name × Nat) KernelValue := ∅
            let mut globalElementWrites : Std.HashMap (Name × Nat) KernelValue := ∅

            for wb in allBuffers do
              for ((name, idx), value) in wb.sharedWrites.toList do
                sharedElementWrites := sharedElementWrites.insert (name, idx) value
              for ((name, idx), value) in wb.globalWrites.toList do
                globalElementWrites := globalElementWrites.insert (name, idx) value

            -- Helper to apply element writes to an array
            let applyWrites (arrayKV : KernelValue) (writes : List (Nat × KernelValue)) : KernelValue :=
              match arrayKV with
              | .arrayFloat arr =>
                .arrayFloat <| Id.run do
                  let mut result := arr
                  for (idx, kv) in writes do
                    match kv with
                    | .float v => result := result.set! idx v
                    | _ => ()  -- Skip mismatched types
                  return result
              | .arrayInt arr =>
                .arrayInt <| Id.run do
                  let mut result := arr
                  for (idx, kv) in writes do
                    match kv with
                    | .int v => result := result.set! idx v
                    | _ => ()
                  return result
              | .arrayNat arr =>
                .arrayNat <| Id.run do
                  let mut result := arr
                  for (idx, kv) in writes do
                    match kv with
                    | .nat v => result := result.set! idx v
                    | _ => ()
                  return result
              | scalar =>
                -- For scalars, if index 0 is written, use that value
                match writes.head? with
                | some (0, v) => v
                | _ => scalar

            -- Apply shared memory writes
            let sharedNames := sharedElementWrites.toList.map (fun ((n, _), _) => n) |>.eraseDups
            for name in sharedNames do
              let writes := sharedElementWrites.toList.filter (fun ((n, _), _) => n == name)
                            |>.map (fun ((_, idx), v) => (idx, v))
              match st.shared.get? name with
              | some arrayKV =>
                let updated := applyWrites arrayKV writes
                st := { st with shared := st.shared.insert name updated }
              | none => pure ()  -- Array not found, skip

            -- Apply global memory writes
            let globalNames := globalElementWrites.toList.map (fun ((n, _), _) => n) |>.eraseDups
            for name in globalNames do
              let writes := globalElementWrites.toList.filter (fun ((n, _), _) => n == name)
                            |>.map (fun ((_, idx), v) => (idx, v))
              match st.globals.get? name with
              | some arrayKV =>
                let updated := applyWrites arrayKV writes
                st := { st with globals := st.globals.insert name updated }
              | none => pure ()  -- Array not found, skip

            -- If any thread hit a barrier, increment phase for next iteration
            if anyBarrierHit then
              phase := phase + 1

    return st



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
    shared := Std.HashMap.ofList shared
    writeBuffer := default
    isBuffering := false
    currentPhase := 0
    threadBarrierCount := 0
    hitBarrier := false }

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

/-! ## Kernel Macro for Auto-Generation -/

/-- Kernel Args structure generator.

Automatically generates a kernel arguments structure with scalar parameters
and Name fields for global/shared variable identifiers.

Example:
```
kernelArgs saxpy(N: Nat, alpha: Float)
  global[x y: Array Float]
  global[result: Array Int]
  shared[temp: Array Float]
  shared[sum: Nat]
```

This generates:
```
structure saxpyArgs where
  N : Nat
  alpha : Float
  x : Name
  y : Name
  result : Name
  temp : Name
  sum : Name
  deriving Repr
```

Then write your kernel manually using the generated structure:
```
def saxpyKernel : KernelM saxpyArgs Unit := do
  let args ← getArgs
  let N := args.N
  let alpha := args.alpha
  let x : GlobalArray Float := ⟨args.x⟩
  let y : GlobalArray Float := ⟨args.y⟩
  let result : GlobalArray Int := ⟨args.result⟩
  let temp : SharedArray Float := ⟨args.temp⟩
  let sum : SharedScalar Nat := ⟨args.sum⟩
  -- kernel body...
```
-/
syntax (name := kernelArgsCmd) "kernelArgs" ident
  "(" (ident ":" term),* ")"
  ("global" "[" ident+ ":" term "]")*
  ("shared" "[" ident+ ":" term "]")* : command

open Lean Elab Command Parser in
@[command_elab kernelArgsCmd] def elabKernelArgsCmd : CommandElab := fun stx => do
  -- Parse syntax components
  -- Syntax: "kernelArgs" ident "(" params ")" (global[...])* (shared[...])*
  let name := stx[1]
  let scalarParams := stx[3]

  let argsName := mkIdent (name.getId)--.appendAfter "Args")

  -- Build scalar fields
  let mut scalarFields := #[]
  for i in [:scalarParams.getArgs.size] do
    let arg := scalarParams.getArgs[i]!
    if arg.getNumArgs >= 3 then
      let sid := arg[0]
      let sty := arg[2]
      scalarFields := scalarFields.push (sid, sty)

  -- Collect all global and shared declarations
  let mut allMemFields := #[]

  -- The global/shared declarations start after the closing paren
  -- They're stored as separate syntax nodes in the getArgs array
  let globalDecls := stx[5]  -- This should be the ("global" "[" ident+ ":" term "]")* part
  let sharedDecls := stx[6]  -- This should be the ("shared" "[" ident+ ":" term "]")* part

  -- Process global declarations
  for globalDecl in globalDecls.getArgs do
    if globalDecl.getNumArgs >= 5 then
      let ids := globalDecl[2]  -- ident+ inside brackets
      -- Handle multiple identifiers
      let idArray := if ids.isOfKind `ident then #[ids] else ids.getArgs
      for id in idArray do
        if !id.isOfKind nullKind then
          allMemFields := allMemFields.push id

  -- Process shared declarations
  for sharedDecl in sharedDecls.getArgs do
    if sharedDecl.getNumArgs >= 5 then
      let ids := sharedDecl[2]  -- ident+ inside brackets
      -- Handle multiple identifiers
      let idArray := if ids.isOfKind `ident then #[ids] else ids.getArgs
      for id in idArray do
        if !id.isOfKind nullKind then
          allMemFields := allMemFields.push id

  -- Generate structure command as string
  let mut structStr := s!"structure {argsName.getId} where\n"
  for (sid, sty) in scalarFields do
    let styStr := Format.pretty (← liftCoreM <| PrettyPrinter.ppCategory `term sty)
    structStr := structStr ++ s!"  {sid.getId} : {styStr}\n"
  for id in allMemFields do
    structStr := structStr ++ s!"  {id.getId} : Lean.Name\n"
  structStr := structStr ++ "  deriving Repr"

  -- Parse and elaborate structure
  match runParserCategory (← getEnv) `command structStr with
  | Except.ok structSyntax => elabCommand structSyntax
  | Except.error err => throwError "Failed to parse structure: {err}"

  -- Generate response structure for global arrays only
  -- Response contains the actual array types from the global declarations
  let responseName := mkIdent (name.getId.appendAfter "Response")

  -- Collect global array field names with their types
  let mut globalFields := #[]
  for globalDecl in globalDecls.getArgs do
    if globalDecl.getNumArgs >= 5 then
      let ids := globalDecl[2]
      let typeSyntax := globalDecl[4]  -- The type after the colon
      let idArray := if ids.isOfKind `ident then #[ids] else ids.getArgs
      for id in idArray do
        if !id.isOfKind nullKind then
          globalFields := globalFields.push (id, typeSyntax)

  -- Generate response structure
  let mut responseStr := s!"structure {responseName.getId} where\n"
  for (id, typeSyntax) in globalFields do
    -- Extract element type from "Array T" syntax
    let typeStr ← liftCoreM <| do
      let fmt ← PrettyPrinter.ppCategory `term typeSyntax
      pure (Format.pretty fmt)
    responseStr := responseStr ++ s!"  {id.getId} : {typeStr}\n"
  responseStr := responseStr ++ "  deriving Repr"

  match runParserCategory (← getEnv) `command responseStr with
  | Except.ok responseSyntax => elabCommand responseSyntax
  | Except.error err => throwError "Failed to parse response structure: {err}"

  -- Generate FromJson instance for response structure
  let mut fromJsonStr := s!"instance : Lean.FromJson {responseName.getId} where\n"
  fromJsonStr := fromJsonStr ++ "  fromJson? json := do\n"
  fromJsonStr := fromJsonStr ++ "    -- Parse the nested JSON structure\n"
  fromJsonStr := fromJsonStr ++ "    let resultsObj ← json.getObjVal? \"results\"\n"

  -- Extract each global array field from JSON
  for (id, typeSyntax) in globalFields do
    let fieldName := id.getId.toString
    let typeStr ← liftCoreM <| do
      let fmt ← PrettyPrinter.ppCategory `term typeSyntax
      pure (Format.pretty fmt)
    fromJsonStr := fromJsonStr ++ s!"    let {id.getId}Json ← resultsObj.getObjVal? \"{fieldName}\"\n"
    fromJsonStr := fromJsonStr ++ s!"    let {id.getId} : {typeStr} ← Lean.fromJson? {id.getId}Json\n"

  -- Construct response object
  fromJsonStr := fromJsonStr ++ "    pure { "
  for i in [:globalFields.size] do
    let (id, _) := globalFields[i]!
    fromJsonStr := fromJsonStr ++ s!"{id.getId} := {id.getId}"
    if i < globalFields.size - 1 then
      fromJsonStr := fromJsonStr ++ ", "
  fromJsonStr := fromJsonStr ++ " }"

  match runParserCategory (← getEnv) `command fromJsonStr with
  | Except.ok fromJsonSyntax => elabCommand fromJsonSyntax
  | Except.error err => throwError "Failed to parse FromJson instance: {err}"


end GpuDSL
