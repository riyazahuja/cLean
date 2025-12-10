/-
  Gröbner Basis Computation

  Implements the Buchberger algorithm and matrix-based F4-style reduction.
-/

import CLean.Polynomial
import CLean.FiniteField
import Std.Data.HashMap

namespace CLean.Groebner

open FiniteField
open Polynomial

/-- Compute S-polynomial of two polynomials -/
def spoly (f g : Poly) : Poly :=
  match f.leadTerm, g.leadTerm with
  | some ltf, some ltg =>
    let lcmMono := Monomial.lcm ltf.mono ltg.mono
    let mf := Monomial.div lcmMono ltf.mono
    let mg := Monomial.div lcmMono ltg.mono
    let cf_inv := inv ltf.coeff
    let cg_inv := inv ltg.coeff
    let termF : Term := { coeff := cf_inv, mono := mf }
    let termG : Term := { coeff := cg_inv, mono := mg }
    (f.mulTerm termF).sub (g.mulTerm termG)
  | _, _ => Poly.zero f.numVars

/-- Reduce polynomial f by polynomial g (one step).
    Returns (reduced, didReduce) -/
def reduceStep (f g : Poly) : Poly × Bool :=
  match g.leadTerm with
  | none => (f, false)
  | some ltg =>
    -- Find a term in f divisible by LT(g)
    let maybeIdx := f.terms.findIdx? fun t => Monomial.divides ltg.mono t.mono
    match maybeIdx with
    | none => (f, false)
    | some idx =>
      let t := f.terms[idx]!
      let quotMono := Monomial.div t.mono ltg.mono
      let quotCoeff := fdiv t.coeff ltg.coeff
      let quot : Term := { coeff := quotCoeff, mono := quotMono }
      let reduced := f.sub (g.mulTerm quot)
      (reduced, true)

/-- Fully reduce f by a set of polynomials G -/
partial def reduce (f : Poly) (G : Array Poly) : Poly :=
  if f.isZero then f
  else
    -- Try to reduce by each polynomial in G
    let rec tryReduce (f : Poly) (i : Nat) : Poly × Bool :=
      if i >= G.size then (f, false)
      else
        let (f', reduced) := reduceStep f G[i]!
        if reduced then (f', true)
        else tryReduce f (i + 1)

    let (f', reduced) := tryReduce f 0
    if reduced then reduce f' G
    else f

/-- Check if polynomial is in ideal (reduces to zero) -/
def inIdeal (f : Poly) (G : Array Poly) : Bool :=
  (reduce f G).isZero

/-- Buchberger's algorithm to compute Gröbner basis (CPU version) -/
partial def buchberger (F : Array Poly) : Array Poly := Id.run do
  if F.isEmpty then return #[]

  let mut G := F
  let mut pairs : List (Nat × Nat) := []

  -- Initialize pairs
  for i in [:G.size] do
    for j in [i+1:G.size] do
      pairs := (i, j) :: pairs

  while !pairs.isEmpty do
    let (i, j) := pairs.head!
    pairs := pairs.tail!

    if i < G.size && j < G.size then
      let s := spoly G[i]! G[j]!
      let r := reduce s G

      if !r.isZero then
        let newIdx := G.size
        G := G.push r
        -- Add new pairs
        for k in [:newIdx] do
          pairs := (k, newIdx) :: pairs

  G

/-- Minimal Gröbner basis (remove redundant elements) -/
def minimalBasis (G : Array Poly) : Array Poly := Id.run do
  let mut result : Array Poly := #[]
  for g in G do
    -- Check if LT(g) is divisible by LT of any other polynomial
    let dominated := result.any fun h =>
      match g.leadMono, h.leadMono with
      | some mg, some mh => Monomial.divides mh mg && !Monomial.eq mg mh
      | _, _ => false
    if !dominated then
      result := result.push g
  result

/-! ## Matrix-based reduction for F4 -/

/-- Dense Macaulay matrix for small problems -/
structure DenseMatrix where
  rows : Nat
  cols : Nat
  data : Array FpElem  -- Row-major
  deriving Repr, Inhabited

def DenseMatrix.empty : DenseMatrix := ⟨0, 0, #[]⟩

def DenseMatrix.get (m : DenseMatrix) (i j : Nat) : FpElem :=
  if i < m.rows && j < m.cols then
    m.data.getD (i * m.cols + j) 0
  else 0

def DenseMatrix.set (m : DenseMatrix) (i j : Nat) (v : FpElem) : DenseMatrix :=
  if i < m.rows && j < m.cols then
    { m with data := m.data.set! (i * m.cols + j) v }
  else m

def DenseMatrix.create (rows cols : Nat) : DenseMatrix :=
  ⟨rows, cols, Array.replicate (rows * cols) 0⟩

/-- Build a dense matrix from a list of polynomials.
    Returns (matrix, monomial_list) where columns correspond to monomials -/
def buildDenseMatrix (polys : Array Poly) : DenseMatrix × Array Monomial := Id.run do
  if polys.isEmpty then return (DenseMatrix.empty, #[])

  -- Collect all monomials and sort by GrevLex descending
  let mut allMonos : Array Monomial := #[]
  for p in polys do
    for t in p.terms do
      if !allMonos.any (Monomial.eq t.mono) then
        allMonos := allMonos.push t.mono
  allMonos := allMonos.qsort fun m1 m2 => Monomial.grevLexLt m2 m1

  -- Build monomial to column index map
  let mut monoToCol : Std.HashMap Monomial Nat := {}
  for (mono, idx) in allMonos.zipIdx do
    monoToCol := monoToCol.insert mono idx

  -- Build matrix
  let nrows := polys.size
  let ncols := allMonos.size
  let mut mat := DenseMatrix.create nrows ncols

  for (p, row) in polys.zipIdx do
    for t in p.terms do
      if let some col := monoToCol.get? t.mono then
        mat := mat.set row col t.coeff

  (mat, allMonos)

/-- CPU Gaussian elimination on dense matrix -/
def gaussianEliminationCPU (m : DenseMatrix) : DenseMatrix := Id.run do
  let mut mat := m
  let mut pivotRow := 0

  for col in [:mat.cols] do
    if pivotRow >= mat.rows then break

    -- Find pivot (non-zero entry in column)
    let mut found := false
    for row in [pivotRow:mat.rows] do
      if mat.get row col != 0 then
        -- Swap rows
        if row != pivotRow then
          for c in [:mat.cols] do
            let tmp := mat.get pivotRow c
            mat := mat.set pivotRow c (mat.get row c)
            mat := mat.set row c tmp
        found := true
        break

    if found then
      -- Scale pivot row to make leading coefficient 1
      let pivotVal := mat.get pivotRow col
      let pivotInv := inv pivotVal
      for c in [:mat.cols] do
        mat := mat.set pivotRow c (mul (mat.get pivotRow c) pivotInv)

      -- Eliminate column in other rows
      for row in [:mat.rows] do
        if row != pivotRow then
          let factor := mat.get row col
          if factor != 0 then
            for c in [:mat.cols] do
              let newVal := sub (mat.get row c) (mul factor (mat.get pivotRow c))
              mat := mat.set row c newVal

      pivotRow := pivotRow + 1

  mat

/-- Extract polynomials from reduced matrix -/
def extractPolys (mat : DenseMatrix) (monos : Array Monomial) (numVars : Nat) : Array Poly := Id.run do
  let mut result : Array Poly := #[]
  for row in [:mat.rows] do
    let mut terms : Array Term := #[]
    for col in [:mat.cols] do
      let coeff := mat.get row col
      if coeff != 0 && col < monos.size then
        terms := terms.push { coeff := coeff, mono := monos[col]! }
    if !terms.isEmpty then
      result := result.push (Poly.normalize terms numVars)
  result

/-- F4-style reduction using matrix operations (CPU baseline) -/
def f4ReductionCPU (polys : Array Poly) : Array Poly :=
  if polys.isEmpty then #[]
  else
    let (mat, monos) := buildDenseMatrix polys
    let reduced := gaussianEliminationCPU mat
    let numVars := polys[0]!.numVars
    extractPolys reduced monos numVars

/-! ## Ideal Membership Testing -/

/-- Test if polynomial f is in the ideal generated by G.
    First computes Gröbner basis, then reduces f. -/
def idealMembershipTest (f : Poly) (generators : Array Poly) : Bool :=
  let G := buchberger generators
  inIdeal f G

/-! ## Testing -/

-- Test: x - 2, y - 3, check if x + y - 5 is in ideal
#eval do
  let numVars := 2

  -- Generator 1: x - 2
  let g1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },      -- x
      { coeff := neg 2, mono := #[0, 0] }   -- -2
    ]
    numVars := numVars
  }

  -- Generator 2: y - 3
  let g2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[0, 1] },      -- y
      { coeff := neg 3, mono := #[0, 0] }   -- -3
    ]
    numVars := numVars
  }

  -- Test: x + y - 5 (should be in ideal since 2 + 3 = 5)
  let f : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },      -- x
      { coeff := 1, mono := #[0, 1] },      -- y
      { coeff := neg 5, mono := #[0, 0] }   -- -5
    ]
    numVars := numVars
  }

  IO.println s!"Testing: f = {f}"
  IO.println s!"Generators: g1 = {g1}, g2 = {g2}"

  let result := idealMembershipTest f #[g1, g2]
  IO.println s!"Is f in ideal? {result}"  -- Should be true

-- Test: x + y - 6 should NOT be in ideal
#eval do
  let numVars := 2

  let g1 : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },
      { coeff := neg 2, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  let g2 : Poly := {
    terms := #[
      { coeff := 1, mono := #[0, 1] },
      { coeff := neg 3, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  -- Test: x + y - 6 (should NOT be in ideal since 2 + 3 ≠ 6)
  let f : Poly := {
    terms := #[
      { coeff := 1, mono := #[1, 0] },
      { coeff := 1, mono := #[0, 1] },
      { coeff := neg 6, mono := #[0, 0] }
    ]
    numVars := numVars
  }

  IO.println s!"Testing: f = {f}"
  let result := idealMembershipTest f #[g1, g2]
  IO.println s!"Is f in ideal? {result}"  -- Should be false

end CLean.Groebner
