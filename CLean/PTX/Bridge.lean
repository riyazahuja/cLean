import CLean.PTX.Parser
import CLean.PTX.Lowering

namespace CLean
namespace PTX

/-- Errors at the trusted PTX ingestion boundary: syntax first, checked lowering second. -/
inductive BridgeError where
  | parse (err : Parser.ParseError)
  | lower (err : LowerError)
  deriving Repr

structure CheckedKernel where
  source : Kernel
  env : KernelEnv
  deriving Repr, Inhabited

structure CheckedModule where
  source : Module
  kernels : Array CheckedKernel
  deriving Repr, Inhabited

/-- Computable instruction supportedness for a particular type environment. -/
def lowerInstrSupported? (env : Typing.TypeEnv) (instr : Instr) : Bool :=
  match lowerInstrChecked? env instr with
  | .ok _ => true
  | .error _ => false

/-- Computable guarded-instruction supportedness for a particular type environment. -/
def lowerGInstrSupported? (env : Typing.TypeEnv) (gi : GInstr) : Bool :=
  match lowerGInstrChecked? env gi with
  | .ok _ => true
  | .error _ => false

/-- Computable block supportedness, including terminator label checks. -/
def lowerBlockSupported? (labels : Std.HashMap BlockLabel Unit) (env : Typing.TypeEnv)
    (block : Block) : Bool :=
  match lowerBlockChecked? labels env block with
  | .ok _ => true
  | .error _ => false

/-- Computable kernel supportedness at the checked lowering boundary. -/
def lowerKernelSupported? (kernel : Kernel) : Bool :=
  match lowerKernelEnvChecked? kernel with
  | .ok _ => true
  | .error _ => false

/-- Syntax parse plus checked lowering for a single kernel. -/
def parseAndLowerKernel? (input : String) : Except BridgeError CheckedKernel := do
  let kernel ← match Parser.parseKernel input with
    | .ok kernel => pure kernel
    | .error err => throw (.parse err)
  let env ← match lowerKernelEnvChecked? kernel with
    | .ok env => pure env
    | .error err => throw (.lower err)
  pure { source := kernel, env := env }

/-- Syntax parse plus checked lowering for every kernel in a module. -/
def parseAndLowerModule? (input : String) : Except BridgeError CheckedModule := do
  let moduleAst ← match Parser.parseModule input with
    | .ok moduleAst => pure moduleAst
    | .error err => throw (.parse err)
  let mut kernels : Array CheckedKernel := #[]
  for kernel in moduleAst.kernels do
    let env ← match lowerKernelEnvChecked? kernel with
      | .ok env => pure env
      | .error err => throw (.lower err)
    kernels := kernels.push { source := kernel, env := env }
  pure { source := moduleAst, kernels := kernels }

/-- Boolean wrapper for examples and regression tests. -/
def parseAndLowerKernelOk? (input : String) : Bool :=
  match parseAndLowerKernel? input with
  | .ok _ => true
  | .error _ => false

/-- Boolean wrapper for examples and regression tests. -/
def parseAndLowerModuleOk? (input : String) : Bool :=
  match parseAndLowerModule? input with
  | .ok _ => true
  | .error _ => false

end PTX
end CLean
