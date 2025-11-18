import Lean
import Std.Data.HashMap

/-! # Verification Intermediate Representation

This module defines the core IR for both CUDA code generation and verification.
-/

namespace CLean.VerifyIR

/-- Types in the verification IR -/
inductive VType where
  | float | int | nat | bool
  | array : VType → VType
deriving Repr, BEq, Inhabited

/-- Memory space classification -/
inductive MemorySpace where
  | global | shared | local
deriving Repr, BEq, Inhabited

/-- Uniformity classification for variables -/
inductive Uniformity where
  | uniform | nonUniform
deriving Repr, BEq, Inhabited

/-- Symbolic expressions -/
inductive VExpr where
  | threadIdX | threadIdY | threadIdZ
  | blockIdX | blockIdY | blockIdZ
  | blockDimX | blockDimY | blockDimZ
  | gridDimX | gridDimY | gridDimZ
  | constInt : Int → VExpr
  | constFloat : Float → VExpr
  | constBool : Bool → VExpr
  | var : Lean.Name → VExpr
  | add : VExpr → VExpr → VExpr
  | sub : VExpr → VExpr → VExpr
  | mul : VExpr → VExpr → VExpr
  | div : VExpr → VExpr → VExpr
  | mod : VExpr → VExpr → VExpr
  | shl : VExpr → VExpr → VExpr
  | shr : VExpr → VExpr → VExpr
  | band : VExpr → VExpr → VExpr
  | bor : VExpr → VExpr → VExpr
  | lt : VExpr → VExpr → VExpr
  | le : VExpr → VExpr → VExpr
  | eq : VExpr → VExpr → VExpr
  | ne : VExpr → VExpr → VExpr
  | gt : VExpr → VExpr → VExpr
  | ge : VExpr → VExpr → VExpr
  | land : VExpr → VExpr → VExpr
  | lor : VExpr → VExpr → VExpr
  | lnot : VExpr → VExpr
deriving Inhabited

instance : Repr VExpr := ⟨fun _ _ => "VExpr"⟩

/-- Memory location -/
structure MemLoc where
  array : Lean.Name
  index : VExpr
  memorySpace : MemorySpace

instance : Repr MemLoc := ⟨fun _ _ => "MemLoc"⟩
instance : Inhabited MemLoc := ⟨⟨`dummy, .constInt 0, .local⟩⟩

/-- Access mode -/
inductive AccessMode where
  | read | write | readWrite
deriving Repr, BEq

end CLean.VerifyIR

/-! Mutual types must be outside namespace -/
mutual
  /-- Statement kinds -/
  inductive CLean.VerifyIR.VStmtKind where
    | read : CLean.VerifyIR.MemLoc → Lean.Name → CLean.VerifyIR.VStmtKind
    | write : CLean.VerifyIR.MemLoc → CLean.VerifyIR.VExpr → CLean.VerifyIR.VStmtKind
    | assign : Lean.Name → CLean.VerifyIR.VExpr → CLean.VerifyIR.VStmtKind
    | barrier : CLean.VerifyIR.VStmtKind
    | seq : List CLean.VerifyIR.VStmt → CLean.VerifyIR.VStmtKind
    | ite : CLean.VerifyIR.VExpr → List CLean.VerifyIR.VStmt → List CLean.VerifyIR.VStmt → CLean.VerifyIR.VStmtKind
    | whileLoop : CLean.VerifyIR.VExpr → List CLean.VerifyIR.VStmt → CLean.VerifyIR.VStmtKind
    | forLoop : Lean.Name → CLean.VerifyIR.VExpr → CLean.VerifyIR.VExpr → CLean.VerifyIR.VExpr → List CLean.VerifyIR.VStmt → CLean.VerifyIR.VStmtKind

  /-- Predicated statement -/
  structure CLean.VerifyIR.VStmt where
    stmt : CLean.VerifyIR.VStmtKind
    predicate : CLean.VerifyIR.VExpr
end

namespace CLean.VerifyIR

-- Instances for mutual types
instance : Repr VStmtKind := ⟨fun _ _ => "VStmtKind"⟩
instance : Repr VStmt := ⟨fun _ _ => "VStmt"⟩
instance : Inhabited VStmtKind := ⟨.barrier⟩
instance : Inhabited VStmt := ⟨⟨.barrier, .constBool true⟩⟩

/-- Variable info -/
structure VarInfo where
  name : Lean.Name
  type : VType
  uniformity : Uniformity
  memorySpace : MemorySpace

instance : Repr VarInfo := ⟨fun _ _ => "VarInfo"⟩
instance : Inhabited VarInfo := ⟨⟨`dummy, .int, .uniform, .local⟩⟩

/-- Complete kernel representation -/
structure VKernel where
  name : Lean.Name
  params : List VarInfo
  locals : List VarInfo
  globalArrays : List VarInfo
  sharedArrays : List VarInfo
  body : List VStmt

instance : Repr VKernel := ⟨fun _ _ => "VKernel"⟩
instance : Inhabited VKernel := ⟨⟨`dummy, [], [], [], [], []⟩⟩

/-! ## Two-Thread Encoding -/

inductive ThreadId where
  | t1 | t2
deriving Repr, BEq, Inhabited

structure VExpr2 where
  e1 : VExpr
  e2 : VExpr
deriving Repr, Inhabited

structure VStmt2 where
  stmt1 : VStmt
  stmt2 : VStmt
deriving Repr, Inhabited

structure VKernel2 where
  original : VKernel
  varMap : List (Lean.Name × (Lean.Name × Lean.Name))
  body1 : List VStmt
  body2 : List VStmt
  assumptions : List VExpr

instance : Repr VKernel2 := ⟨fun _ _ => "VKernel2"⟩
instance : Inhabited VKernel2 := ⟨⟨default, [], [], [], []⟩⟩

/-! ## Memory Access Tracking -/

structure AccessRecord where
  location : MemLoc
  mode : AccessMode
  threadId : ThreadId
  predicate : VExpr

instance : Repr AccessRecord := ⟨fun _ _ => "AccessRecord"⟩
instance : Inhabited AccessRecord := ⟨⟨default, .read, .t1, .constBool true⟩⟩

structure BarrierPhase where
  phaseId : Nat
  statements : List VStmt
  accesses : List AccessRecord
  entryPredicate : VExpr2

instance : Repr BarrierPhase := ⟨fun _ _ => "BarrierPhase"⟩
instance : Inhabited BarrierPhase := ⟨⟨0, [], [], default⟩⟩

/-! ## Control Flow Graph -/

inductive CFGNode where
  | entry | exit
  | barrier : Nat → CFGNode
  | basic : List VStmt → CFGNode
deriving Repr, Inhabited

structure CFGEdge where
  source : Nat
  target : Nat
  condition : VExpr

instance : Repr CFGEdge := ⟨fun _ _ => "CFGEdge"⟩
instance : Inhabited CFGEdge := ⟨⟨0, 0, .constBool true⟩⟩

structure CFG where
  nodes : List (Nat × CFGNode)
  edges : List CFGEdge
  entry : Nat
  exit : Nat

instance : Repr CFG := ⟨fun _ _ => "CFG"⟩
instance : Inhabited CFG := ⟨⟨[], [], 0, 0⟩⟩

/-! ## Helper Functions -/

namespace VExpr

partial def dependsOnThreadId : VExpr → Bool
  | threadIdX => true
  | threadIdY => true
  | threadIdZ => true
  | blockIdX => true
  | blockIdY => true
  | blockIdZ => true
  | blockDimX => false
  | blockDimY => false
  | blockDimZ => false
  | gridDimX => false
  | gridDimY => false
  | gridDimZ => false
  | constInt _ => false
  | constFloat _ => false
  | constBool _ => false
  | var _ => false
  | add a b => dependsOnThreadId a || dependsOnThreadId b
  | sub a b => dependsOnThreadId a || dependsOnThreadId b
  | mul a b => dependsOnThreadId a || dependsOnThreadId b
  | div a b => dependsOnThreadId a || dependsOnThreadId b
  | mod a b => dependsOnThreadId a || dependsOnThreadId b
  | shl a b => dependsOnThreadId a || dependsOnThreadId b
  | shr a b => dependsOnThreadId a || dependsOnThreadId b
  | band a b => dependsOnThreadId a || dependsOnThreadId b
  | bor a b => dependsOnThreadId a || dependsOnThreadId b
  | lt a b => dependsOnThreadId a || dependsOnThreadId b
  | le a b => dependsOnThreadId a || dependsOnThreadId b
  | eq a b => dependsOnThreadId a || dependsOnThreadId b
  | ne a b => dependsOnThreadId a || dependsOnThreadId b
  | gt a b => dependsOnThreadId a || dependsOnThreadId b
  | ge a b => dependsOnThreadId a || dependsOnThreadId b
  | land a b => dependsOnThreadId a || dependsOnThreadId b
  | lor a b => dependsOnThreadId a || dependsOnThreadId b
  | lnot a => dependsOnThreadId a

partial def substituteThreadId (tid : ThreadId) : VExpr → VExpr
  | threadIdX => match tid with | .t1 => var `tid1_x | .t2 => var `tid2_x
  | threadIdY => match tid with | .t1 => var `tid1_y | .t2 => var `tid2_y
  | threadIdZ => match tid with | .t1 => var `tid1_z | .t2 => var `tid2_z
  | blockIdX => match tid with | .t1 => var `bid1_x | .t2 => var `bid2_x
  | blockIdY => match tid with | .t1 => var `bid1_y | .t2 => var `bid2_y
  | blockIdZ => match tid with | .t1 => var `bid1_z | .t2 => var `bid2_z
  | e@(blockDimX) => e
  | e@(blockDimY) => e
  | e@(blockDimZ) => e
  | e@(gridDimX) => e
  | e@(gridDimY) => e
  | e@(gridDimZ) => e
  | e@(constInt _) => e
  | e@(constFloat _) => e
  | e@(constBool _) => e
  | var n => match tid with
    | .t1 => var (n.appendAfter "_1")
    | .t2 => var (n.appendAfter "_2")
  | add a b => add (substituteThreadId tid a) (substituteThreadId tid b)
  | sub a b => sub (substituteThreadId tid a) (substituteThreadId tid b)
  | mul a b => mul (substituteThreadId tid a) (substituteThreadId tid b)
  | div a b => div (substituteThreadId tid a) (substituteThreadId tid b)
  | mod a b => mod (substituteThreadId tid a) (substituteThreadId tid b)
  | shl a b => shl (substituteThreadId tid a) (substituteThreadId tid b)
  | shr a b => shr (substituteThreadId tid a) (substituteThreadId tid b)
  | band a b => band (substituteThreadId tid a) (substituteThreadId tid b)
  | bor a b => bor (substituteThreadId tid a) (substituteThreadId tid b)
  | lt a b => lt (substituteThreadId tid a) (substituteThreadId tid b)
  | le a b => le (substituteThreadId tid a) (substituteThreadId tid b)
  | eq a b => eq (substituteThreadId tid a) (substituteThreadId tid b)
  | ne a b => ne (substituteThreadId tid a) (substituteThreadId tid b)
  | gt a b => gt (substituteThreadId tid a) (substituteThreadId tid b)
  | ge a b => ge (substituteThreadId tid a) (substituteThreadId tid b)
  | land a b => land (substituteThreadId tid a) (substituteThreadId tid b)
  | lor a b => lor (substituteThreadId tid a) (substituteThreadId tid b)
  | lnot a => lnot (substituteThreadId tid a)

end VExpr

namespace VStmt

partial def findBarriers (s : VStmt) : List Nat :=
  match s.stmt with
  | .barrier => [0]
  | .seq stmts => stmts.flatMap findBarriers
  | .ite _ thn els => (thn.flatMap findBarriers) ++ (els.flatMap findBarriers)
  | .whileLoop _ body => body.flatMap findBarriers
  | .forLoop _ _ _ _ body => body.flatMap findBarriers
  | _ => []

partial def findAccesses (s : VStmt) : List (MemLoc × AccessMode) :=
  match s.stmt with
  | .read loc _ => [(loc, .read)]
  | .write loc _ => [(loc, .write)]
  | .seq stmts => stmts.flatMap findAccesses
  | .ite _ thn els => (thn.flatMap findAccesses) ++ (els.flatMap findAccesses)
  | .whileLoop _ body => body.flatMap findAccesses
  | .forLoop _ _ _ _ body => body.flatMap findAccesses
  | _ => []

end VStmt

namespace VKernel

def extractBarrierPhases (k : VKernel) : List BarrierPhase :=
  [{ phaseId := 0
     statements := k.body
     accesses := []
     entryPredicate := ⟨.constBool true, .constBool true⟩ }]

def buildCFG (k : VKernel) : CFG :=
  { nodes := [(0, .entry), (1, .basic k.body), (2, .exit)]
    edges := [{ source := 0, target := 1, condition := .constBool true },
              { source := 1, target := 2, condition := .constBool true }]
    entry := 0
    exit := 2 }

end VKernel

end CLean.VerifyIR
