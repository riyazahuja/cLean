import CLean.DeviceIR
import CLean.GPU
import Mathlib.Tactic

/-!
# Verification Semantics for DeviceIR

Pure functional semantics using Rat instead of Float for provable arithmetic.
Designed for functional correctness verification of GPU kernels.

## Key Design Decisions

1. **Rat instead of Float**: All floating-point operations use `Rat` for exact arithmetic
2. **Pure function memory**: `VMem := String → Nat → VVal` instead of HashMap for clean reasoning
3. **Structural recursion**: Uses structural recursion on `DStmt` for termination
4. **Loop unrolling**: For loops are unrolled to finite sequences for bounded verification
-/

namespace CLean.Verification

open DeviceIR
open GpuDSL

/-! ## Verification Values -/

/-- Values for verification: uses Rat instead of Float -/
inductive VVal where
  | rat : Rat → VVal
  | int : Int → VVal
  | nat : Nat → VVal    -- For indices and sizes - avoids Int/Nat conversion issues
  | bool : Bool → VVal
  deriving Repr, Inhabited, DecidableEq

namespace VVal

def toRat : VVal → Rat
  | rat q => q
  | int n => n
  | nat n => n
  | bool _ => 0

def toInt : VVal → Int
  | rat q => q.floor
  | int n => n
  | nat n => n
  | bool b => if b then 1 else 0

def toBool : VVal → Bool
  | rat q => q ≠ 0
  | int n => n ≠ 0
  | nat n => n ≠ 0
  | bool b => b

/-- Convert to Nat - direct for nat, otherwise via Int.toNat -/
def toNat (v : VVal) : Nat :=
  match v with
  | nat n => n
  | int n => n.toNat
  | rat q => q.floor.toNat
  | bool b => if b then 1 else 0

/-- Default value for uninitialized memory -/
def default : VVal := VVal.int 0

end VVal

/-! ## Verification Memory -/

/-- Memory as a pure function from (array name, index) to value.
    This representation is ideal for verification proofs. -/
def VMem := String → Nat → VVal

namespace VMem

/-- Empty memory (all zeros) -/
def empty : VMem := fun _ _ => VVal.int 0

/-- Get a value from memory -/
def get (mem : VMem) (arr : String) (idx : Nat) : VVal := mem arr idx

/-- Set a value in memory -/
def set (mem : VMem) (arr : String) (idx : Nat) (val : VVal) : VMem :=
  fun a i => if a = arr ∧ i = idx then val else mem a i

/-- Setting a value doesn't affect other indices -/
theorem set_ne (mem : VMem) (arr : String) (idx idx' : Nat) (val : VVal) (h : idx' ≠ idx) :
    (mem.set arr idx val) arr idx' = mem arr idx' := by
  simp only [set]
  simp only [h, and_false, ↓reduceIte]

/-- Setting a value doesn't affect other arrays -/
theorem set_ne_arr (mem : VMem) (arr arr' : String) (idx idx' : Nat) (val : VVal) (h : arr' ≠ arr) :
    (mem.set arr idx val) arr' idx' = mem arr' idx' := by
  simp only [set]
  simp only [h, false_and, ↓reduceIte]

/-- Get a rational value from memory -/
def getR (mem : VMem) (arr : String) (idx : Nat) : Rat := (mem.get arr idx).toRat

/-- Get an integer value from memory -/
def getI (mem : VMem) (arr : String) (idx : Nat) : Int := (mem.get arr idx).toInt

/-- Set a rational value in memory -/
def setR (mem : VMem) (arr : String) (idx : Nat) (val : Rat) : VMem :=
  mem.set arr idx (VVal.rat val)

/-- Set an integer value in memory -/
def setI (mem : VMem) (arr : String) (idx : Nat) (val : Int) : VMem :=
  mem.set arr idx (VVal.int val)

end VMem

/-! ## Memory Initialization -/

/-- Initialize memory from a list of (name, Rat array) pairs -/
def initMemR (arrays : List (String × Array Rat)) : VMem :=
  fun arr idx =>
    match arrays.find? (fun p => p.1 = arr) with
    | some (_, data) =>
        if h : idx < data.size then VVal.rat (data[idx])
        else VVal.int 0
    | none => VVal.int 0

/-- Initialize memory from a list of (name, Int array) pairs -/
def initMemI (arrays : List (String × Array Int)) : VMem :=
  fun arr idx =>
    match arrays.find? (fun p => p.1 = arr) with
    | some (_, data) =>
        if h : idx < data.size then VVal.int (data[idx])
        else VVal.int 0
    | none => VVal.int 0

/-! ## Thread Context -/

/-- Thread context for verification -/
structure VCtx where
  tid : Nat        -- thread index within block (linearized)
  tidX : Nat       -- thread index X
  tidY : Nat       -- thread index Y
  tidZ : Nat       -- thread index Z
  bid : Nat        -- block index (linearized)
  bidX : Nat       -- block index X
  bidY : Nat       -- block index Y
  bidZ : Nat       -- block index Z
  blockDimX : Nat
  blockDimY : Nat
  blockDimZ : Nat
  gridDimX : Nat
  gridDimY : Nat
  gridDimZ : Nat
  locals : String → VVal  -- local variable bindings
  deriving Inhabited

namespace VCtx

/-- Create a thread context from grid/block config and thread/block indices -/
def mk' (grid block : Dim3) (tidX tidY tidZ bidX bidY bidZ : Nat) : VCtx where
  tid := tidX + tidY * block.x + tidZ * block.x * block.y
  tidX := tidX
  tidY := tidY
  tidZ := tidZ
  bid := bidX + bidY * grid.x + bidZ * grid.x * grid.y
  bidX := bidX
  bidY := bidY
  bidZ := bidZ
  blockDimX := block.x
  blockDimY := block.y
  blockDimZ := block.z
  gridDimX := grid.x
  gridDimY := grid.y
  gridDimZ := grid.z
  locals := fun _ => VVal.int 0

/-- Get a local variable -/
def getLocal (ctx : VCtx) (name : String) : VVal := ctx.locals name

/-- Set a local variable -/
def setLocal (ctx : VCtx) (name : String) (val : VVal) : VCtx :=
  { ctx with locals := fun n => if n = name then val else ctx.locals n }

/-- Global thread index X = blockIdx.x * blockDim.x + threadIdx.x -/
def globalIdxX (ctx : VCtx) : Nat := ctx.bidX * ctx.blockDimX + ctx.tidX

/-- Global thread index Y = blockIdx.y * blockDim.y + threadIdx.y -/
def globalIdxY (ctx : VCtx) : Nat := ctx.bidY * ctx.blockDimY + ctx.tidY

/-- Global thread index Z = blockIdx.z * blockDim.z + threadIdx.z -/
def globalIdxZ (ctx : VCtx) : Nat := ctx.bidZ * ctx.blockDimZ + ctx.tidZ

end VCtx

/-! ## Binary Operations -/

/-- Evaluate a binary operation on VVal.
    Note: VVal.nat is promoted to VVal.nat for arithmetic,
    keeping results as Nat when both operands are Nat (for clean index arithmetic). -/
def vEvalBinOp (op : BinOp) (v1 v2 : VVal) : VVal :=
  match op with
  | BinOp.add =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.nat (n1 + n2)  -- Nat + Nat = Nat
      | VVal.rat q1, VVal.rat q2 => VVal.rat (q1 + q2)
      | VVal.int n1, VVal.int n2 => VVal.int (n1 + n2)
      | VVal.nat n, VVal.int m => VVal.int (n + m)
      | VVal.int n, VVal.nat m => VVal.int (n + m)
      | VVal.rat q, VVal.int n => VVal.rat (q + n)
      | VVal.int n, VVal.rat q => VVal.rat (n + q)
      | VVal.rat q, VVal.nat n => VVal.rat (q + n)
      | VVal.nat n, VVal.rat q => VVal.rat (n + q)
      | _, _ => VVal.int 0
  | BinOp.sub =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.int ((n1 : Int) - n2)  -- Sub may go negative
      | VVal.rat q1, VVal.rat q2 => VVal.rat (q1 - q2)
      | VVal.int n1, VVal.int n2 => VVal.int (n1 - n2)
      | VVal.nat n, VVal.int m => VVal.int (n - m)
      | VVal.int n, VVal.nat m => VVal.int (n - m)
      | VVal.rat q, VVal.int n => VVal.rat (q - n)
      | VVal.int n, VVal.rat q => VVal.rat (n - q)
      | VVal.rat q, VVal.nat n => VVal.rat (q - n)
      | VVal.nat n, VVal.rat q => VVal.rat (n - q)
      | _, _ => VVal.int 0
  | BinOp.mul =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.nat (n1 * n2)  -- Nat * Nat = Nat
      | VVal.rat q1, VVal.rat q2 => VVal.rat (q1 * q2)
      | VVal.int n1, VVal.int n2 => VVal.int (n1 * n2)
      | VVal.nat n, VVal.int m => VVal.int (n * m)
      | VVal.int n, VVal.nat m => VVal.int (n * m)
      | VVal.rat q, VVal.int n => VVal.rat (q * n)
      | VVal.int n, VVal.rat q => VVal.rat (n * q)
      | VVal.rat q, VVal.nat n => VVal.rat (q * n)
      | VVal.nat n, VVal.rat q => VVal.rat (n * q)
      | _, _ => VVal.int 0
  | BinOp.div =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => if n2 ≠ 0 then VVal.nat (n1 / n2) else VVal.int 0
      | VVal.rat q1, VVal.rat q2 => if q2 ≠ 0 then VVal.rat (q1 / q2) else VVal.int 0
      | VVal.int n1, VVal.int n2 => if n2 ≠ 0 then VVal.int (n1 / n2) else VVal.int 0
      | VVal.rat q, VVal.int n => if n ≠ 0 then VVal.rat (q / n) else VVal.int 0
      | VVal.int n, VVal.rat q => if q ≠ 0 then VVal.rat (n / q) else VVal.int 0
      | VVal.rat q, VVal.nat n => if n ≠ 0 then VVal.rat (q / n) else VVal.int 0
      | VVal.nat n, VVal.rat q => if q ≠ 0 then VVal.rat (n / q) else VVal.int 0
      | VVal.nat n, VVal.int m => if m ≠ 0 then VVal.int (n / m) else VVal.int 0
      | VVal.int n, VVal.nat m => if m ≠ 0 then VVal.int (n / m) else VVal.int 0
      | _, _ => VVal.int 0
  | BinOp.mod =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => if n2 ≠ 0 then VVal.nat (n1 % n2) else VVal.int 0
      | VVal.int n1, VVal.int n2 => if n2 ≠ 0 then VVal.int (n1 % n2) else VVal.int 0
      | _, _ => VVal.int 0
  | BinOp.lt =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 < n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) < m)
      | VVal.int n, VVal.nat m => VVal.bool (n < (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 < q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 < n2)
      | VVal.rat q, VVal.int n => VVal.bool (q < n)
      | VVal.int n, VVal.rat q => VVal.bool (n < q)
      | VVal.rat q, VVal.nat n => VVal.bool (q < n)
      | VVal.nat n, VVal.rat q => VVal.bool ((n : Rat) < q)
      | _, _ => VVal.bool false
  | BinOp.le =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 ≤ n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) ≤ m)
      | VVal.int n, VVal.nat m => VVal.bool (n ≤ (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 ≤ q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 ≤ n2)
      | VVal.rat q, VVal.int n => VVal.bool (q ≤ n)
      | VVal.int n, VVal.rat q => VVal.bool (n ≤ q)
      | VVal.rat q, VVal.nat n => VVal.bool (q ≤ n)
      | VVal.nat n, VVal.rat q => VVal.bool ((n : Rat) ≤ q)
      | _, _ => VVal.bool false
  | BinOp.gt =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 > n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) > m)
      | VVal.int n, VVal.nat m => VVal.bool (n > (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 > q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 > n2)
      | VVal.rat q, VVal.int n => VVal.bool (q > n)
      | VVal.int n, VVal.rat q => VVal.bool (n > q)
      | VVal.rat q, VVal.nat n => VVal.bool (q > n)
      | VVal.nat n, VVal.rat q => VVal.bool ((n : Rat) > q)
      | _, _ => VVal.bool false
  | BinOp.ge =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 ≥ n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) ≥ m)
      | VVal.int n, VVal.nat m => VVal.bool (n ≥ (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 ≥ q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 ≥ n2)
      | VVal.rat q, VVal.int n => VVal.bool (q ≥ n)
      | VVal.int n, VVal.rat q => VVal.bool (n ≥ q)
      | VVal.rat q, VVal.nat n => VVal.bool (q ≥ n)
      | VVal.nat n, VVal.rat q => VVal.bool ((n : Rat) ≥ q)
      | _, _ => VVal.bool false
  | BinOp.eq =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 = n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) = m)
      | VVal.int n, VVal.nat m => VVal.bool (n = (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 = q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 = n2)
      | VVal.bool b1, VVal.bool b2 => VVal.bool (b1 = b2)
      | _, _ => VVal.bool false
  | BinOp.ne =>
      match v1, v2 with
      | VVal.nat n1, VVal.nat n2 => VVal.bool (n1 ≠ n2)
      | VVal.nat n, VVal.int m => VVal.bool ((n : Int) ≠ m)
      | VVal.int n, VVal.nat m => VVal.bool (n ≠ (m : Int))
      | VVal.rat q1, VVal.rat q2 => VVal.bool (q1 ≠ q2)
      | VVal.int n1, VVal.int n2 => VVal.bool (n1 ≠ n2)
      | VVal.bool b1, VVal.bool b2 => VVal.bool (b1 ≠ b2)
      | _, _ => VVal.bool true
  | BinOp.and =>
      match v1, v2 with
      | VVal.bool b1, VVal.bool b2 => VVal.bool (b1 && b2)
      | _, _ => VVal.bool false
  | BinOp.or =>
      match v1, v2 with
      | VVal.bool b1, VVal.bool b2 => VVal.bool (b1 || b2)
      | _, _ => VVal.bool false

/-! ## Float to Rat Conversion -/

/-- Convert a Float to Rat for verification purposes.
    Note: This is an approximation since Float uses IEEE 754.
    For verification, we use a fixed-precision approximation. -/
def floatToRat (f : Float) : Rat :=
  -- Use a fixed-precision approximation
  let scale : Nat := 1000000
  let scaled := (f * Float.ofNat scale).toUInt64.toNat
  let sign : Int := if f < 0 then -1 else 1
  sign * (scaled : Rat) / scale

/-! ## Expression Evaluation -/

/-- Evaluate a DExpr to a VVal.
    Converts Float literals to Rat for verification.
    Thread/block indices use VVal.nat for cleaner proofs. -/
def vEvalExpr (e : DExpr) (ctx : VCtx) (mem : VMem) : VVal :=
  match e with
  | DExpr.intLit n => VVal.int n
  | DExpr.floatLit f => VVal.rat (floatToRat f)  -- Convert Float → Rat
  | DExpr.boolLit b => VVal.bool b
  | DExpr.var x => ctx.getLocal x
  | DExpr.threadIdx Dim.x => VVal.nat ctx.tidX
  | DExpr.threadIdx Dim.y => VVal.nat ctx.tidY
  | DExpr.threadIdx Dim.z => VVal.nat ctx.tidZ
  | DExpr.blockIdx Dim.x => VVal.nat ctx.bidX
  | DExpr.blockIdx Dim.y => VVal.nat ctx.bidY
  | DExpr.blockIdx Dim.z => VVal.nat ctx.bidZ
  | DExpr.blockDim Dim.x => VVal.nat ctx.blockDimX
  | DExpr.blockDim Dim.y => VVal.nat ctx.blockDimY
  | DExpr.blockDim Dim.z => VVal.nat ctx.blockDimZ
  | DExpr.gridDim Dim.x => VVal.nat ctx.gridDimX
  | DExpr.gridDim Dim.y => VVal.nat ctx.gridDimY
  | DExpr.gridDim Dim.z => VVal.nat ctx.gridDimZ
  | DExpr.binop op e1 e2 =>
      vEvalBinOp op (vEvalExpr e1 ctx mem) (vEvalExpr e2 ctx mem)
  | DExpr.unop UnOp.neg e' =>
      match vEvalExpr e' ctx mem with
      | VVal.rat q => VVal.rat (-q)
      | VVal.int n => VVal.int (-n)
      | v => v
  | DExpr.unop UnOp.not e' =>
      match vEvalExpr e' ctx mem with
      | VVal.bool b => VVal.bool (!b)
      | _ => VVal.bool false
  | DExpr.index (DExpr.var arrName) idx =>
      let idxVal := (vEvalExpr idx ctx mem).toNat
      mem.get arrName idxVal
  | DExpr.index _ _ => VVal.int 0  -- Nested indexing not supported
  | DExpr.field _ _ => VVal.int 0  -- Field access not yet supported

/-! ## Statement Evaluation (Structural Recursion)

For loop-free kernels (like SAXPY), we use direct structural recursion.
For kernels with bounded loops, use `vEvalStmtUnrolled` with explicit bounds.
-/

/-- Evaluate a loop-free statement using structural recursion on DStmt.
    For loops are treated as no-ops (use vEvalStmtUnrolled for loops). -/
def vEvalStmt (s : DStmt) (ctx : VCtx) (mem : VMem) : VCtx × VMem :=
  match s with
  | DStmt.skip => (ctx, mem)
  | DStmt.assign x e =>
      let val := vEvalExpr e ctx mem
      (ctx.setLocal x val, mem)
  | DStmt.store (DExpr.var arrName) idx val =>
      let idxVal := (vEvalExpr idx ctx mem).toNat
      let value := vEvalExpr val ctx mem
      (ctx, mem.set arrName idxVal value)
  | DStmt.store _ _ _ => (ctx, mem)
  | DStmt.seq s1 s2 =>
      let (ctx', mem') := vEvalStmt s1 ctx mem
      vEvalStmt s2 ctx' mem'
  | DStmt.ite cond sthen selse =>
      if (vEvalExpr cond ctx mem).toBool then
        vEvalStmt sthen ctx mem
      else
        vEvalStmt selse ctx mem
  | DStmt.for _ _ _ _ => (ctx, mem)  -- Loops require explicit unrolling
  | DStmt.whileLoop _ _ => (ctx, mem)  -- While loops not supported
  | DStmt.barrier => (ctx, mem)
  | DStmt.call _ _ => (ctx, mem)
  | DStmt.assert _ _ => (ctx, mem)

/-! ## Thread Execution -/

/-- Execute a statement for a single thread -/
def vExecThread (body : DStmt) (grid block : Dim3)
    (tidX tidY tidZ bidX bidY bidZ : Nat) (mem : VMem) : VMem :=
  let ctx := VCtx.mk' grid block tidX tidY tidZ bidX bidY bidZ
  let (_, mem') := vEvalStmt body ctx mem
  mem'

/-- Execute a statement for a single thread with initial locals (for kernel params) -/
def vExecThreadWithLocals (body : DStmt) (grid block : Dim3)
    (tidX tidY tidZ bidX bidY bidZ : Nat)
    (initLocals : String → VVal) (mem : VMem) : VMem :=
  let ctx : VCtx := {
    VCtx.mk' grid block tidX tidY tidZ bidX bidY bidZ with
    locals := initLocals
  }
  let (_, mem') := vEvalStmt body ctx mem
  mem'

/-- Execute a statement for a single thread (1D simplified version) -/
def vExecThread1D (body : DStmt) (blockSize gridSize : Nat)
    (tid bid : Nat) (mem : VMem) : VMem :=
  vExecThread body ⟨gridSize, 1, 1⟩ ⟨blockSize, 1, 1⟩ tid 0 0 bid 0 0 mem

/-- Execute a statement for a single thread (1D) with kernel params -/
def vExecThread1DWithParams (body : DStmt) (blockSize gridSize : Nat)
    (tid bid : Nat) (params : String → VVal) (mem : VMem) : VMem :=
  vExecThreadWithLocals body ⟨gridSize, 1, 1⟩ ⟨blockSize, 1, 1⟩ tid 0 0 bid 0 0 params mem

/-! ## Kernel Execution -/

/-- Execute kernel over all threads in 1D configuration -/
def vExecKernel1D (body : DStmt) (numBlocks blockSize : Nat) (mem₀ : VMem) : VMem :=
  let totalThreads := numBlocks * blockSize
  let rec loop (gid : Nat) (mem : VMem) : VMem :=
    if gid >= totalThreads then mem
    else
      let bid := gid / blockSize
      let tid := gid % blockSize
      let mem' := vExecThread1D body blockSize numBlocks tid bid mem
      loop (gid + 1) mem'
  termination_by totalThreads - gid
  loop 0 mem₀

/-- Execute kernel over all threads in 1D configuration with kernel params -/
def vExecKernel1DWithParams (body : DStmt) (numBlocks blockSize : Nat)
    (params : String → VVal) (mem₀ : VMem) : VMem :=
  let totalThreads := numBlocks * blockSize
  let rec loop (gid : Nat) (mem : VMem) : VMem :=
    if gid >= totalThreads then mem
    else
      let bid := gid / blockSize
      let tid := gid % blockSize
      let mem' := vExecThread1DWithParams body blockSize numBlocks tid bid params mem
      loop (gid + 1) mem'
  termination_by totalThreads - gid
  loop 0 mem₀

/-! ## Simp Lemmas for Memory Operations -/

@[simp]
theorem VMem.get_set_same (mem : VMem) (arr : String) (idx : Nat) (val : VVal) :
    (mem.set arr idx val).get arr idx = val := by
  simp [VMem.get, VMem.set]

@[simp]
theorem VMem.get_set_diff_arr (mem : VMem) (arr1 arr2 : String) (idx1 idx2 : Nat) (val : VVal)
    (h : arr1 ≠ arr2) :
    (mem.set arr1 idx1 val).get arr2 idx2 = mem.get arr2 idx2 := by
  simp only [VMem.get, VMem.set]
  simp only [ite_eq_right_iff, and_imp]
  intro h_eq _
  exact absurd h_eq.symm h

@[simp]
theorem VMem.get_set_diff_idx (mem : VMem) (arr : String) (idx1 idx2 : Nat) (val : VVal)
    (h : idx1 ≠ idx2) :
    (mem.set arr idx1 val).get arr idx2 = mem.get arr idx2 := by
  simp only [VMem.get, VMem.set]
  simp only [ite_eq_right_iff, and_imp]
  intro _ h_eq
  exact absurd h_eq.symm h

@[simp]
theorem VMem.getR_set_same (mem : VMem) (arr : String) (idx : Nat) (val : Rat) :
    (mem.setR arr idx val).getR arr idx = val := by
  simp [VMem.getR, VMem.setR, VMem.get_set_same, VVal.toRat]

@[simp]
theorem VMem.getI_set_same (mem : VMem) (arr : String) (idx : Nat) (val : Int) :
    (mem.setI arr idx val).getI arr idx = val := by
  simp [VMem.getI, VMem.setI, VMem.get_set_same, VVal.toInt]

/-! ## Total Thread Count -/

def totalThreads (grid block : Dim3) : Nat :=
  grid.x * grid.y * grid.z * block.x * block.y * block.z

def totalThreads1D (numBlocks blockSize : Nat) : Nat :=
  numBlocks * blockSize

/-! ## Global Thread Index -/

/-- Compute global thread index from block/thread indices (1D) -/
def globalIdx1D (blockSize bid tid : Nat) : Nat :=
  bid * blockSize + tid

/-- A thread is active if its global index is less than N -/
def threadActive1D (blockSize bid tid N : Nat) : Prop :=
  globalIdx1D blockSize bid tid < N

/-! ## Key Lemmas for Statement Evaluation -/

@[simp]
theorem vEvalStmt_skip (ctx : VCtx) (mem : VMem) :
    vEvalStmt DStmt.skip ctx mem = (ctx, mem) := by
  simp [vEvalStmt]

@[simp]
theorem vEvalStmt_assign (x : String) (e : DExpr) (ctx : VCtx) (mem : VMem) :
    vEvalStmt (DStmt.assign x e) ctx mem = (ctx.setLocal x (vEvalExpr e ctx mem), mem) := by
  simp [vEvalStmt]

@[simp]
theorem vEvalStmt_store (arrName : String) (idx val : DExpr) (ctx : VCtx) (mem : VMem) :
    vEvalStmt (DStmt.store (DExpr.var arrName) idx val) ctx mem =
      (ctx, mem.set arrName (vEvalExpr idx ctx mem).toNat (vEvalExpr val ctx mem)) := by
  simp [vEvalStmt]

theorem vEvalStmt_seq (s1 s2 : DStmt) (ctx : VCtx) (mem : VMem) :
    vEvalStmt (DStmt.seq s1 s2) ctx mem =
      let (ctx', mem') := vEvalStmt s1 ctx mem
      vEvalStmt s2 ctx' mem' := by
  simp [vEvalStmt]


theorem vEvalStmt_ite_true (cond : DExpr) (sthen selse : DStmt) (ctx : VCtx) (mem : VMem)
    (h_cond : (vEvalExpr cond ctx mem).toBool = true) :
    vEvalStmt (DStmt.ite cond sthen selse) ctx mem = vEvalStmt sthen ctx mem := by
  simp [vEvalStmt, h_cond]

theorem vEvalStmt_ite_false (cond : DExpr) (sthen selse : DStmt) (ctx : VCtx) (mem : VMem)
    (h_cond : (vEvalExpr cond ctx mem).toBool = false) :
    vEvalStmt (DStmt.ite cond sthen selse) ctx mem = vEvalStmt selse ctx mem := by
  simp [vEvalStmt, h_cond]

/-! ## Int/Nat Coercion Lemmas for Symbolic Proofs -/

/-- Match on Int.ofNat always takes the ofNat branch -/
@[simp]
theorem Int.ofNat_match_eq (n : Nat) :
    (match (Int.ofNat n) with | Int.ofNat m => m | Int.negSucc _ => 0) = n := rfl

/-- Key lemma: global index computation simplifies when all inputs are Nat -/
@[simp]
theorem globalIdx_match_simplify (bid blockSize tid : Nat) :
    (match (bid : Int) * (blockSize : Int) + (tid : Int) with
     | Int.ofNat m => m
     | Int.negSucc _ => 0) = bid * blockSize + tid := by
  have h : (bid : Int) * (blockSize : Int) + (tid : Int) = Int.ofNat (bid * blockSize + tid) := by
    simp only [Int.ofNat_eq_coe, Int.natCast_mul, Int.natCast_add]
  simp only [h, Int.ofNat_match_eq]

/-- Nat comparison through Int coercion -/
theorem Int.coe_nat_lt (a b : Nat) : (a : Int) < (b : Int) ↔ a < b := Int.ofNat_lt

end CLean.Verification
