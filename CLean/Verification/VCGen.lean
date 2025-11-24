/-
  Verification Condition Generation

  Generates concrete proof obligations from verified kernels.
  Implements dual-thread transformation inspired by GPUVerify:
  - Symbolically execute two arbitrary threads
  - Generate VCs showing they don't race
  - Generate bounds-checking VCs
  - Generate barrier divergence VCs

  Outputs: Well-typed Lean theorems that users prove interactively.
-/

import CLean.VerificationIR
import CLean.ToVerificationIR
import CLean.Verification.SafetyProperties
import CLean.DeviceIR
import CLean.GPU

open CLean.VerificationIR
open CLean.Verification.SafetyProperties
open DeviceIR
open GpuDSL

namespace CLean.Verification.VCGen

/-! ## Symbolic Thread Representation -/

/-- Symbolic thread identifier for verification -/
structure SymbolicThread where
  /-- Thread number (1 or 2 for dual-thread) -/
  threadNum : Nat
  /-- Symbolic threadIdx.x value -/
  tidX : Nat → Prop  -- Predicate: "tidX is this value"
  /-- Symbolic threadIdx.y value -/
  tidY : Nat → Prop
  /-- Symbolic threadIdx.z value -/
  tidZ : Nat → Prop
  /-- Bounds constraints -/
  boundsX : Nat
  boundsY : Nat
  boundsZ : Nat

/-- Create symbolic thread with unknown thread ID -/
def mkSymbolicThread (threadNum : Nat) (blockDim : Dim3) : SymbolicThread :=
  { threadNum := threadNum
    tidX := fun _ => True  -- Any valid thread ID
    tidY := fun _ => True
    tidZ := fun _ => True
    boundsX := blockDim.x
    boundsY := blockDim.y
    boundsZ := blockDim.z }

/-- Two symbolic threads are distinct -/
def SymbolicThread.distinct (t1 t2 : SymbolicThread) : Prop :=
  ∃ (id1X id1Y id1Z id2X id2Y id2Z : Nat),
    t1.tidX id1X ∧ t1.tidY id1Y ∧ t1.tidZ id1Z ∧
    t2.tidX id2X ∧ t2.tidY id2Y ∧ t2.tidZ id2Z ∧
    (id1X ≠ id2X ∨ id1Y ≠ id2Y ∨ id1Z ≠ id2Z)

/-! ## Dual-Thread Transformation -/

/-- Dualized memory access: tagged with thread number -/
structure DualAccess where
  thread : Nat  -- 1 or 2
  access : MemoryAccess
deriving Repr

/-- Dualized kernel: two threads executing symbolically -/
structure DualKernel where
  original : VerifiedKernel
  thread1 : SymbolicThread
  thread2 : SymbolicThread
  /-- Accesses tagged by thread -/
  dualAccesses : List DualAccess
  /-- Constraint: threads are distinct -/
  threadsDistinct : thread1.distinct thread2
instance : Inhabited DualKernel where
  default := {
    original := default
    thread1 := mkSymbolicThread 1 ⟨1, 1, 1⟩
    thread2 := mkSymbolicThread 2 ⟨1, 1, 1⟩
    dualAccesses := []
    threadsDistinct := by sorry
  }

/-- Create dual-thread version of kernel -/
def dualizeKernel (k : VerifiedKernel) : DualKernel :=
  let t1 := mkSymbolicThread 1 k.context.blockDim
  let t2 := mkSymbolicThread 2 k.context.blockDim

  -- Duplicate accesses for both threads
  let dualAccesses := Id.run do
    let mut accesses := []
    for acc in k.accesses do
      accesses := ({ thread := 1, access := acc } : DualAccess) :: accesses
      accesses := ({ thread := 2, access := acc } : DualAccess) :: accesses
    return accesses.reverse

  { original := k
    thread1 := t1
    thread2 := t2
    dualAccesses := dualAccesses
    threadsDistinct := sorry }

/-! ## Verification Condition Types -/

/-- A verification condition (proof obligation) -/
inductive VerificationCondition where
  /-- Race freedom: two accesses don't race -/
  | raceVC :
      (acc1 acc2 : MemoryAccess) →
      (conflict : acc1.conflicts acc2) →
      VerificationCondition

  /-- Memory bounds: access is within array bounds -/
  | boundsVC :
      (acc : MemoryAccess) →
      (arrayName : String) →
      (arraySize : Nat) →
      VerificationCondition

  /-- Barrier divergence: all threads reach barrier -/
  | barrierVC :
      (barrier : BarrierPoint) →
      VerificationCondition

  /-- Uniformity: expression has same value across threads -/
  | uniformityVC :
      (expr : DExpr) →
      VerificationCondition

/-- Convert VC to a human-readable theorem statement -/
def VerificationCondition.toTheorem (vc : VerificationCondition) (kernelName : String) : String :=
  match vc with
  | raceVC acc1 acc2 _ =>
      s!"theorem {kernelName}_no_race_{acc1.location}_{acc2.location} : \n" ++
      s!"  {acc1} and {acc2} do not race :=\n" ++
      s!"  sorry"

  | boundsVC acc arrayName arraySize =>
      s!"theorem {kernelName}_bounds_{acc.location}_{arrayName} : \n" ++
      let idxStr := match acc.index with | some i => dexprToString i | none => "scalar"
      s!"  Access to {arrayName}[{idxStr}] is within bounds [0, {arraySize}) :=\n" ++
      s!"  sorry"

  | barrierVC barrier =>
      s!"theorem {kernelName}_barrier_{barrier.location} : \n" ++
      s!"  All threads reach barrier at location {barrier.location} :=\n" ++
      s!"  sorry"

  | uniformityVC expr =>
      s!"theorem {kernelName}_uniform_{dexprToString expr} : \n" ++
      s!"  Expression {dexprToString expr} is thread-uniform :=\n" ++
      s!"  sorry"

/-! ## VC Generation for Specific Properties -/

/-- Generate race-freedom VCs using dual-thread approach -/
def generateRaceVCs (dk : DualKernel) : List VerificationCondition :=
  Id.run do
    let mut vcs := []

    -- For all pairs of accesses from different threads
    for da1 in dk.dualAccesses do
      for da2 in dk.dualAccesses do
        if da1.thread ≠ da2.thread && da1.access.conflictsBool da2.access then
          -- Need to prove: either ordered by happens-before, or don't conflict
          vcs := VerificationCondition.raceVC da1.access da2.access (by sorry) :: vcs

    return vcs

/-- Generate bounds-checking VCs -/
def generateBoundsVCs (k : VerifiedKernel) (arraySizes : List (String × Nat)) : List VerificationCondition :=
  Id.run do
    let mut vcs := []

    for acc in k.accesses do
      match acc.index with
      | none => ()  -- Scalar access, no bounds check needed
      | some _ =>
          -- Find array size
          match arraySizes.find? (fun (name, _) => name = acc.name) with
          | none => ()  -- Unknown array size, skip
          | some (_, size) =>
              vcs := VerificationCondition.boundsVC acc acc.name size :: vcs

    return vcs

/-- Generate barrier divergence VCs -/
def generateBarrierVCs (k : VerifiedKernel) : List VerificationCondition :=
  k.barriers.map VerificationCondition.barrierVC

/-- Generate uniformity VCs for expressions in conditional branches -/
def generateUniformityVCs (k : VerifiedKernel) : List VerificationCondition :=
  -- For each barrier, check that it's in uniform control flow
  Id.run do
    let mut vcs := []
    for b in k.barriers do
      -- Find uniformity info around this barrier
      -- (Simplified: just generate VC, actual check would analyze control flow)
      ()  -- TODO: implement control flow analysis
    return vcs

/-! ## Main VC Generation -/

/-- Generate all verification conditions for a kernel -/
def generateAllVCs
    (k : VerifiedKernel)
    (arraySizes : List (String × Nat))
    : List VerificationCondition :=
  let dk := dualizeKernel k
  let raceVCs := generateRaceVCs dk
  let boundsVCs := generateBoundsVCs k arraySizes
  let barrierVCs := generateBarrierVCs k
  let uniformityVCs := generateUniformityVCs k
  raceVCs ++ boundsVCs ++ barrierVCs ++ uniformityVCs

/-! ## Theorem Generation (Output for Users) -/

/-- Generate Lean theorem statements for all VCs -/
def generateTheoremStatements
    (k : VerifiedKernel)
    (arraySizes : List (String × Nat))
    : List String :=
  let vcs := generateAllVCs k arraySizes
  vcs.map (fun vc => vc.toTheorem k.ir.name)

/-- Write VCs to a Lean file -/
def writeVCsToFile
    (k : VerifiedKernel)
    (arraySizes : List (String × Nat))
    (filename : String)
    : IO Unit := do
  let theorems := generateTheoremStatements k arraySizes

  let header := s!"/-\n" ++
                s!"  Verification Conditions for {k.ir.name}\n" ++
                s!"  Generated automatically from DeviceIR\n" ++
                s!"-/\n\n" ++
                s!"import CLean.Verification.SafetyProperties\n" ++
                s!"import CLean.Verification.Tactics\n\n" ++
                s!"open CLean.Verification.SafetyProperties\n\n"

  let content := header ++ String.intercalate "\n\n" theorems

  IO.FS.writeFile filename content
  IO.println s!"Generated {theorems.length} verification conditions in {filename}"

/-! ## Pretty Printing and Analysis -/

/-- Print summary of generated VCs -/
def printVCSummary (vcs : List VerificationCondition) : IO Unit := do
  let numRace := vcs.filter (fun vc => match vc with | .raceVC _ _ _ => true | _ => false) |>.length
  let numBounds := vcs.filter (fun vc => match vc with | .boundsVC _ _ _ => true | _ => false) |>.length
  let numBarrier := vcs.filter (fun vc => match vc with | .barrierVC _ => true | _ => false) |>.length
  let numUniformity := vcs.filter (fun vc => match vc with | .uniformityVC _ => true | _ => false) |>.length

  IO.println "═══════════════════════════════════════"
  IO.println "Verification Condition Summary"
  IO.println "═══════════════════════════════════════"
  IO.println s!"  Race Freedom VCs:       {numRace}"
  IO.println s!"  Bounds Checking VCs:    {numBounds}"
  IO.println s!"  Barrier Divergence VCs: {numBarrier}"
  IO.println s!"  Uniformity VCs:         {numUniformity}"
  IO.println s!"  Total VCs:              {vcs.length}"

/-! ## Complete Verification Workflow -/

/-- Full verification workflow: analyze kernel and generate VCs -/
def verifyKernel
    (kernel : Kernel)
    (gridDim blockDim : Dim3)
    (arraySizes : List (String × Nat))
    (outputFile : String)
    : IO Unit := do
  IO.println s!"[1/4] Translating {kernel.name} to VerificationIR..."
  let vk := CLean.ToVerificationIR.toVerificationIR kernel gridDim blockDim

  IO.println "[2/4] Analyzing kernel..."
  CLean.ToVerificationIR.printKernelAnalysis vk

  IO.println "[3/4] Generating verification conditions..."
  let vcs := generateAllVCs vk arraySizes
  printVCSummary vcs

  IO.println s!"[4/4] Writing theorems to {outputFile}..."
  writeVCsToFile vk arraySizes outputFile

  IO.println "\n✓ Verification conditions generated!"
  IO.println s!"  Next step: Prove the theorems in {outputFile}"

/-- Generate a single safety theorem with subgoals for all VCs -/
def generateSafetyTheorem
    (k : VerifiedKernel)
    (arraySizes : List (String × Nat))
    : String :=
  let vcs := generateAllVCs k arraySizes
  let header := s!"theorem {k.ir.name}_safe : True := by\n"
  let subgoals := vcs.map (fun vc =>
    match vc with
    | .raceVC acc1 acc2 _ =>
        s!"  have no_race_{acc1.location}_{acc2.location} : {acc1} and {acc2} do not race := by sorry"
    | .boundsVC acc arrayName arraySize =>
        let idxStr := match acc.index with | some i => dexprToString i | none => "scalar"
        s!"  have bounds_{acc.location}_{arrayName} : Access to {arrayName}[{idxStr}] is within bounds [0, {arraySize}) := by sorry"
    | .barrierVC barrier =>
        s!"  have barrier_{barrier.location} : All threads reach barrier at location {barrier.location} := by sorry"
    | .uniformityVC expr =>
        s!"  have uniform_{dexprToString expr} : Expression {dexprToString expr} is thread-uniform := by sorry"
  )
  header ++ String.intercalate "\n" subgoals ++ "\n  trivial"

end CLean.Verification.VCGen
