/-
  Device Intermediate Representation (DeviceIR)

  Core IR for GPU kernels that:
  - Is target of translation from Lean code
  - Can be interpreted for verification
  - Can be compiled to CUDA
-/

namespace DeviceIR

-- Grid dimension (x, y, or z)
inductive Dim
  | x | y | z
  deriving Repr, BEq, Inhabited

-- Binary operations
inductive BinOp
  -- Arithmetic
  | add | sub | mul | div | mod
  -- Comparison
  | lt | le | gt | ge | eq | ne
  -- Logical
  | and | or
  deriving Repr, BEq, Inhabited

-- Unary operations
inductive UnOp
  | neg  -- arithmetic negation
  | not  -- logical not
  deriving Repr, BEq, Inhabited

-- Device types (types that can exist on GPU)
inductive DType
  | int       -- signed integer (maps to int32_t)
  | nat       -- natural number (maps to uint32_t or int with bounds checks)
  | float     -- floating point (maps to float)
  | bool      -- boolean
  | array (elemTy : DType)
  | tuple (tys : List DType)
  | struct (name : String) (fields : List (String × DType))
  deriving Repr, BEq, Inhabited

-- Device expressions (pure computation, no side effects)
inductive DExpr
  -- Literals
  | intLit    : Int → DExpr
  | floatLit  : Float → DExpr
  | boolLit   : Bool → DExpr

  -- Variables
  | var       : String → DExpr

  -- Operations
  | binop     : BinOp → DExpr → DExpr → DExpr
  | unop      : UnOp → DExpr → DExpr

  -- Array/tuple indexing
  | index     : DExpr → DExpr → DExpr        -- A[i]
  | field     : DExpr → String → DExpr       -- s.field

  -- GPU-specific thread/block indices
  | threadIdx : Dim → DExpr
  | blockIdx  : Dim → DExpr
  | blockDim  : Dim → DExpr
  | gridDim   : Dim → DExpr
  deriving Repr, BEq, Inhabited

-- Device statements (effects: memory, control flow)
inductive DStmt
  | skip

  -- Assignment to local variable
  | assign    : String → DExpr → DStmt

  -- Store to array/memory
  -- store arr idx val   means   arr[idx] := val
  | store     : DExpr → DExpr → DExpr → DStmt

  -- Sequencing
  | seq       : DStmt → DStmt → DStmt

  -- Conditional
  | ite       : DExpr → DStmt → DStmt → DStmt

  -- For loop: for (var = lo; var < hi; var++) body
  | for       : String → DExpr → DExpr → DStmt → DStmt

  -- While loop
  | whileLoop : DExpr → DStmt → DStmt

  -- GPU barrier synchronization
  | barrier   : DStmt

  -- Function/procedure call
  | call      : String → List DExpr → DStmt

  -- Assertions for verification
  | assert    : DExpr → String → DStmt  -- assert cond, message
  deriving Repr, BEq, Inhabited

-- Variable declaration
structure VarDecl where
  name : String
  ty : DType
  deriving Repr, BEq, Inhabited

-- Memory space classification
inductive MemorySpace
  | local    -- thread-local (registers/local memory)
  | global   -- global device memory
  | shared   -- shared memory within block
  deriving Repr, BEq, Inhabited

-- Array declaration with memory space
structure ArrayDecl where
  name : String
  ty : DType
  space : MemorySpace
  deriving Repr, BEq, Inhabited

-- Complete kernel definition
structure Kernel where
  name : String

  -- Scalar parameters (e.g., N, alpha)
  params : List VarDecl

  -- Local variables
  locals : List VarDecl

  -- Global arrays (input/output)
  globalArrays : List ArrayDecl

  -- Shared arrays
  sharedArrays : List ArrayDecl

  -- Kernel body
  body : DStmt
  deriving Repr, Inhabited

end DeviceIR
