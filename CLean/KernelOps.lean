/-
  KernelOps Typeclass

  Tagless-final interface for writing GPU kernels that can have
  multiple interpretations (execution, IR building, etc.)
-/

import CLean.DeviceIR
import CLean.DeviceTranslation

open DeviceIR DeviceTranslation

namespace CLean

-- Typeclass defining the operations available inside kernels
class KernelOps (m : Type → Type _) where
  /-- Expression type used in this interpretation -/
  Expr : Type

  -- Lifting values to expressions
  natLit    : Nat → Expr
  intLit    : Int → Expr
  floatLit  : Float → Expr
  boolLit   : Bool → Expr

  -- Arithmetic operations
  add  : Expr → Expr → Expr
  sub  : Expr → Expr → Expr
  mul  : Expr → Expr → Expr
  div  : Expr → Expr → Expr
  mod  : Expr → Expr → Expr

  -- Comparison operations
  lt   : Expr → Expr → Expr
  le   : Expr → Expr → Expr
  gt   : Expr → Expr → Expr
  ge   : Expr → Expr → Expr
  eq   : Expr → Expr → Expr
  ne   : Expr → Expr → Expr

  -- Logical operations
  and  : Expr → Expr → Expr
  or   : Expr → Expr → Expr
  not  : Expr → Expr

  -- GPU thread/block intrinsics
  globalIdxX : m Expr
  globalIdxY : m Expr
  globalIdxZ : m Expr
  blockIdxX  : m Expr
  blockIdxY  : m Expr
  blockIdxZ  : m Expr
  threadIdxX : m Expr
  threadIdxY : m Expr
  threadIdxZ : m Expr
  blockDimX  : m Expr
  blockDimY  : m Expr
  blockDimZ  : m Expr
  gridDimX   : m Expr
  gridDimY   : m Expr
  gridDimZ   : m Expr

  -- Memory operations
  /-- Read from global array: let val ← globalGet "arrayName" index -/
  globalGet  : String → Expr → m Expr

  /-- Write to global array: globalSet "arrayName" index value -/
  globalSet  : String → Expr → Expr → m Unit

  /-- Read from shared array -/
  sharedGet  : String → Expr → m Expr

  /-- Write to shared array -/
  sharedSet  : String → Expr → Expr → m Unit

  -- Synchronization
  /-- Barrier synchronization -/
  barrier    : m Unit

  -- Control flow
  /-- Conditional: ifThenElse condition thenBranch elseBranch -/
  ifThenElse : Expr → m Unit → m Unit → m Unit

  /-- For loop: forLoop varName lo hi body -/
  forLoop    : String → Expr → Expr → (Expr → m Unit) → m Unit

-- Syntactic sugar for operations
namespace KernelOps

variable {m : Type → Type _} [Monad m] [KernelOps m]

-- Helper to allow using `<` operator with KernelOps.Expr
def Expr.lt [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.lt a b

def Expr.le [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.le a b

def Expr.add [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.add a b

def Expr.sub [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.sub a b

def Expr.mul [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.mul a b

def Expr.div [KernelOps m] (a b : KernelOps.Expr m) : KernelOps.Expr m :=
  KernelOps.div a b

end KernelOps

-- Notation for nicer kernel syntax
-- We'll add instances to make Expr work with standard operators

end CLean
