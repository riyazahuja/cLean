/-
  Multivariate Polynomial Representation for Gröbner Basis

  Polynomials over ℤ_p with GrevLex (graded reverse lexicographic) ordering.
-/

import CLean.FiniteField
import Std.Data.HashMap

namespace CLean.Polynomial

open FiniteField

/-- A monomial is represented by its exponent vector.
    For n variables, monomial x₀^e₀ * x₁^e₁ * ... * x_{n-1}^e_{n-1}
    is represented as #[e₀, e₁, ..., e_{n-1}] -/
abbrev Monomial := Array Nat

/-- Total degree of a monomial -/
def Monomial.degree (m : Monomial) : Nat := m.foldl (· + ·) 0

/-- Compare monomials using GrevLex ordering.
    First compare total degree, then compare exponents from right to left (reversed). -/
def Monomial.grevLexLt (m1 m2 : Monomial) : Bool :=
  let d1 := m1.degree
  let d2 := m2.degree
  if d1 != d2 then d1 < d2
  else
    -- Same degree: compare from last variable, smaller exponent is "larger" in grevlex
    let n := min m1.size m2.size
    Id.run do
      let mut result := false
      for i in [:n] do
        let idx := n - 1 - i
        let e1 := m1.getD idx 0
        let e2 := m2.getD idx 0
        if e1 != e2 then
          result := e1 > e2  -- Note: reversed for grevlex
          break
      result

/-- Check if monomials are equal -/
def Monomial.eq (m1 m2 : Monomial) : Bool :=
  let n := max m1.size m2.size
  Id.run do
    for i in [:n] do
      if m1.getD i 0 != m2.getD i 0 then return false
    true

/-- Multiply two monomials (add exponents) -/
def Monomial.mul (m1 m2 : Monomial) : Monomial :=
  let n := max m1.size m2.size
  Array.range n |>.map fun i => m1.getD i 0 + m2.getD i 0

/-- Check if m1 divides m2 (all exponents of m1 ≤ m2) -/
def Monomial.divides (m1 m2 : Monomial) : Bool :=
  let n := max m1.size m2.size
  (Array.range n).all fun i => m1.getD i 0 <= m2.getD i 0

/-- Divide m2 by m1 (subtract exponents, assumes m1 | m2) -/
def Monomial.div (m2 m1 : Monomial) : Monomial :=
  let n := max m1.size m2.size
  Array.range n |>.map fun i => m2.getD i 0 - m1.getD i 0

/-- LCM of two monomials (max of exponents) -/
def Monomial.lcm (m1 m2 : Monomial) : Monomial :=
  let n := max m1.size m2.size
  Array.range n |>.map fun i => max (m1.getD i 0) (m2.getD i 0)

/-- Hash function for monomials -/
def Monomial.hash (m : Monomial) : UInt64 :=
  m.foldl (fun h e => h * 31 + e.toUInt64) 0

instance : Hashable Monomial := ⟨Monomial.hash⟩

instance : BEq Monomial := ⟨Monomial.eq⟩

/-- A term is a coefficient-monomial pair -/
structure Term where
  coeff : FpElem
  mono : Monomial
  deriving Repr, BEq

instance : Inhabited Term := ⟨{ coeff := 0, mono := #[] }⟩

/-- A polynomial is a list of terms, sorted by monomial in descending GrevLex order -/
structure Poly where
  terms : Array Term
  numVars : Nat
  deriving Repr

instance : Inhabited Poly := ⟨⟨#[], 0⟩⟩

/-- The zero polynomial -/
def Poly.zero (n : Nat) : Poly := ⟨#[], n⟩

/-- Check if polynomial is zero -/
def Poly.isZero (p : Poly) : Bool := p.terms.isEmpty

/-- Leading term of a polynomial (highest in GrevLex) -/
def Poly.leadTerm (p : Poly) : Option Term :=
  if p.terms.isEmpty then none else some p.terms[0]!

/-- Leading monomial -/
def Poly.leadMono (p : Poly) : Option Monomial := p.leadTerm.map (·.mono)

/-- Leading coefficient -/
def Poly.leadCoeff (p : Poly) : Option FpElem := p.leadTerm.map (·.coeff)

/-- Normalize: sort terms by GrevLex descending, combine like terms, remove zeros -/
def Poly.normalize (terms : Array Term) (n : Nat) : Poly :=
  -- Sort by monomial descending
  let sorted := terms.qsort fun t1 t2 => Monomial.grevLexLt t2.mono t1.mono

  -- Combine like terms
  let combined := sorted.foldl (init := #[]) fun acc t =>
    if acc.isEmpty then
      if t.coeff != 0 then #[t] else #[]
    else
      let last := acc.back!
      if Monomial.eq last.mono t.mono then
        let newCoeff := add last.coeff t.coeff
        if newCoeff == 0 then acc.pop
        else acc.pop.push { last with coeff := newCoeff }
      else if t.coeff != 0 then acc.push t
      else acc

  ⟨combined, n⟩

/-- Add two polynomials -/
def Poly.add (p1 p2 : Poly) : Poly :=
  normalize (p1.terms ++ p2.terms) (max p1.numVars p2.numVars)

/-- Negate a polynomial -/
def Poly.neg (p : Poly) : Poly :=
  ⟨p.terms.map fun t => { t with coeff := FiniteField.neg t.coeff }, p.numVars⟩

/-- Subtract two polynomials -/
def Poly.sub (p1 p2 : Poly) : Poly := p1.add p2.neg

/-- Multiply polynomial by a term -/
def Poly.mulTerm (p : Poly) (t : Term) : Poly :=
  let newTerms := p.terms.map fun pt => {
    coeff := FiniteField.mul pt.coeff t.coeff
    mono := Monomial.mul pt.mono t.mono
  }
  normalize newTerms p.numVars

/-- Multiply two polynomials -/
def Poly.mul (p1 p2 : Poly) : Poly :=
  let allTerms := p1.terms.foldl (init := #[]) fun acc t1 =>
    acc ++ (p2.terms.map fun t2 => {
      coeff := FiniteField.mul t1.coeff t2.coeff
      mono := Monomial.mul t1.mono t2.mono
    })
  normalize allTerms (max p1.numVars p2.numVars)

/-- Create a polynomial from a single variable x_i -/
def Poly.var (i : Nat) (n : Nat) : Poly :=
  let mono : Monomial := Array.range n |>.map fun j => if j == i then 1 else 0
  ⟨#[{ coeff := 1, mono := mono }], n⟩

/-- Create a constant polynomial -/
def Poly.const (c : FpElem) (n : Nat) : Poly :=
  if c == 0 then zero n
  else ⟨#[{ coeff := c, mono := Array.replicate n 0 }], n⟩

instance : ToString Term where
  toString t :=
    let coeffStr := s!"{t.coeff}"
    let monoStrs := t.mono.zipIdx.filterMap fun (e, i) =>
      if e == 0 then none
      else if e == 1 then some s!"x{i}"
      else some s!"x{i}^{e}"
    if monoStrs.isEmpty then coeffStr
    else if t.coeff == 1 then String.intercalate "*" monoStrs.toList
    else coeffStr ++ "*" ++ String.intercalate "*" monoStrs.toList

instance : ToString Poly where
  toString p :=
    if p.terms.isEmpty then "0"
    else String.intercalate " + " (p.terms.map toString).toList

-- Test: Create polynomial x + y
#eval do
  let numVars := 2
  let x := Poly.var 0 numVars
  let y := Poly.var 1 numVars
  let p := x.add y
  IO.println s!"x + y = {p}"

-- Test: Create polynomial x² + y²
#eval do
  let numVars := 2
  let x := Poly.var 0 numVars
  let y := Poly.var 1 numVars
  let p := (x.mul x).add (y.mul y)
  IO.println s!"x² + y² = {p}"

end CLean.Polynomial
