import CLean.DeviceIR
import CLean.GPU
import Std.Data.HashMap

/-!
# Denotational Semantics for DeviceIR

Executable semantics using HashMap-based state representation.
-/

namespace CLean.Semantics

open DeviceIR
open GpuDSL
open Std (HashMap)

/-! ## Runtime Values -/

inductive Value where
  | int : Int → Value
  | float : Float → Value
  | bool : Bool → Value
  deriving Repr, Inhabited, BEq

def Value.toInt : Value → Int
  | Value.int n => n
  | _ => 0

def Value.toFloat : Value → Float
  | Value.float f => f
  | _ => 0.0

def Value.toBool : Value → Bool
  | Value.bool b => b
  | _ => false

def Value.toNat (v : Value) : Nat := v.toInt.toNat

/-! ## State Representation -/

structure ThreadContext where
  tid : Nat
  bid : Nat
  blockSize : Nat
  locals : HashMap String Value
  deriving Inhabited

structure Memory where
  arrays : HashMap String (HashMap Nat Value)
  deriving Inhabited

/-! ## State Helpers -/

def ThreadContext.getLocal (ctx : ThreadContext) (name : String) : Value :=
  ctx.locals.getD name (Value.int 0)

def ThreadContext.setLocal (ctx : ThreadContext) (name : String) (val : Value) : ThreadContext :=
  {ctx with locals := ctx.locals.insert name val}

def Memory.get (mem : Memory) (arrName : String) (idx : Nat) : Value :=
  match mem.arrays.get? arrName with
  | some arr => arr.getD idx (Value.int 0)
  | none => Value.int 0

def Memory.set (mem : Memory) (arrName : String) (idx : Nat) (val : Value) : Memory :=
  let arr := mem.arrays.getD arrName ∅
  let newArr := arr.insert idx val
  {arrays := mem.arrays.insert arrName newArr}

/-! ## Expression Evaluation -/

def evalBinOp (op : BinOp) (v1 v2 : Value) : Value :=
  match op, v1, v2 with
  | BinOp.add, Value.int n1, Value.int n2 => Value.int (n1 + n2)
  | BinOp.sub, Value.int n1, Value.int n2 => Value.int (n1 - n2)
  | BinOp.mul, Value.int n1, Value.int n2 => Value.int (n1 * n2)
  | BinOp.div, Value.int n1, Value.int n2 => Value.int (n1 / n2)
  | BinOp.add, Value.float f1, Value.float f2 => Value.float (f1 + f2)
  | BinOp.sub, Value.float f1, Value.float f2 => Value.float (f1 - f2)
  | BinOp.mul, Value.float f1, Value.float f2 => Value.float (f1 * f2)
  | BinOp.div, Value.float f1, Value.float f2 => Value.float (f1 / f2)
  | BinOp.lt, Value.int n1, Value.int n2 => Value.bool (n1 < n2)
  | BinOp.le, Value.int n1, Value.int n2 => Value.bool (n1 ≤ n2)
  | BinOp.and, Value.bool b1, Value.bool b2 => Value.bool (b1 && b2)
  | BinOp.or, Value.bool b1, Value.bool b2 => Value.bool (b1 || b2)
  | _, _, _ => Value.int 0

def evalExpr (e : DExpr) (ctx : ThreadContext) (mem : Memory) : Value :=
  match e with
  | DExpr.intLit n => Value.int n
  | DExpr.floatLit f => Value.float f
  | DExpr.boolLit b => Value.bool b
  | DExpr.var x => ctx.getLocal x
  | DExpr.threadIdx Dim.x => Value.int ctx.tid
  | DExpr.threadIdx _ => Value.int 0
  | DExpr.blockIdx _ => Value.int ctx.bid
  | DExpr.blockDim Dim.x => Value.int ctx.blockSize
  | DExpr.blockDim _ => Value.int 1
  | DExpr.gridDim _ => Value.int 1
  | DExpr.binop op e1 e2 =>
      evalBinOp op (evalExpr e1 ctx mem) (evalExpr e2 ctx mem)
  | DExpr.unop UnOp.neg (DExpr.intLit n) => Value.int (-n)
  | DExpr.unop UnOp.not e' => Value.bool (!(evalExpr e' ctx mem).toBool)
  | DExpr.index (DExpr.var arrName) idx =>
      mem.get arrName (evalExpr idx ctx mem).toNat
  | _ => Value.int 0

/-! ## Statement Execution -/

def evalStmt (s : DStmt) (ctx : ThreadContext) (mem : Memory) : ThreadContext × Memory :=
  match s with
  | DStmt.skip => (ctx, mem)
  | DStmt.assign x e =>
      (ctx.setLocal x (evalExpr e ctx mem), mem)
  | DStmt.store (DExpr.var arrName) idx val =>
      let idxVal := (evalExpr idx ctx mem).toNat
      let value := evalExpr val ctx mem
      (ctx, mem.set arrName idxVal value)
  | DStmt.seq s1 s2 =>
      let (ctx', mem') := evalStmt s1 ctx mem
      evalStmt s2 ctx' mem'
  | DStmt.ite cond sthen selse =>
      if (evalExpr cond ctx mem).toBool then
        evalStmt sthen ctx mem
      else
        evalStmt selse ctx mem
  | DStmt.barrier => (ctx, mem)
  | _ => (ctx, mem)

/-! ## Kernel Execution -/

def execThread (body : DStmt) (tid bid blockSize : Nat) (mem : Memory) : Memory :=
  let ctx : ThreadContext := {tid := tid, bid := bid, blockSize := blockSize, locals := ∅}
  let (_, mem') := evalStmt body ctx mem
  mem'

def execKernel (body : DStmt) (numThreads blockSize : Nat) (mem₀ : Memory) : Memory :=
  let rec loop (tid : Nat) (mem : Memory) : Memory :=
    if tid >= numThreads then mem
    else loop (tid + 1) (execThread body tid 0 blockSize mem)
  termination_by numThreads - tid
  loop 0 mem₀

/-! ## Helper Functions -/

def Memory.fromArray (name : String) (arr : Array Float) : Memory :=
  let hmap := arr.zipIdx.foldl (fun acc (v, i) => acc.insert i (Value.float v)) ∅
  {arrays := (∅ : HashMap String (HashMap Nat Value)).insert name hmap}

def Memory.toArray (mem : Memory) (name : String) (size : Nat) : Array Float :=
  Array.range size |>.map (fun i => (mem.get name i).toFloat)

end CLean.Semantics
