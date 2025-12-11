/-
  Finite Field Arithmetic for Gröbner Basis Computation

  Implements efficient modular arithmetic over ℤ_p where p is a prime.
  Uses p = 65521 (largest prime < 2^16) to ensure u32 × u32 products don't overflow u64.
-/

namespace CLean.FiniteField

/-- The prime modulus for our finite field.
    65521 is the largest prime less than 2^16, which ensures:
    - Products of two field elements fit in u32 before reduction
    - Intermediate sums in dot products don't overflow u64 -/
def PRIME : UInt32 := 65521

/-- A finite field element is just a UInt32 < PRIME -/
abbrev FpElem := UInt32

/-- Reduce a value modulo PRIME -/
@[inline] def reduce (x : UInt32) : FpElem := x % PRIME

/-- Reduce a UInt64 value modulo PRIME (for products) -/
@[inline] def reduce64 (x : UInt64) : FpElem :=
  (x % PRIME.toUInt64).toUInt32

/-- Addition in ℤ_p -/
@[inline] def add (a b : FpElem) : FpElem :=
  let sum := a + b
  if sum >= PRIME then sum - PRIME else sum

/-- Subtraction in ℤ_p -/
@[inline] def sub (a b : FpElem) : FpElem :=
  if a >= b then a - b else a + PRIME - b

/-- Multiplication in ℤ_p -/
@[inline] def mul (a b : FpElem) : FpElem :=
  reduce64 (a.toUInt64 * b.toUInt64)

/-- Negation in ℤ_p -/
@[inline] def neg (a : FpElem) : FpElem :=
  if a == 0 then 0 else PRIME - a

/-- Extended Euclidean algorithm to compute modular inverse.
    Returns (gcd, x, y) where gcd = a*x + b*y -/
partial def extendedGcd (a b : Int) : Int × Int × Int :=
  if b == 0 then
    (a, 1, 0)
  else
    let (g, x, y) := extendedGcd b (a % b)
    (g, y, x - (a / b) * y)

/-- Modular multiplicative inverse using extended GCD.
    Returns a^(-1) mod p -/
def inv (a : FpElem) : FpElem :=
  if a == 0 then 0 else
    let (_, x, _) := extendedGcd (Int.ofNat a.toNat) (Int.ofNat PRIME.toNat)
    let result := x % (Int.ofNat PRIME.toNat)
    if result < 0 then (result + Int.ofNat PRIME.toNat).toNat.toUInt32
    else result.toNat.toUInt32

/-- Division in ℤ_p: a / b = a * b^(-1) -/
@[inline] def fdiv (a b : FpElem) : FpElem := mul a (inv b)

/-- Power by squaring -/
partial def pow (base : FpElem) (exp : Nat) : FpElem :=
  if exp == 0 then 1
  else if exp % 2 == 0 then
    let half := pow base (exp / 2)
    mul half half
  else
    mul base (pow base (exp - 1))

/-- Convert from Int to FpElem -/
def ofInt (n : Int) : FpElem :=
  let m := n % (Int.ofNat PRIME.toNat)
  if m < 0 then (m + Int.ofNat PRIME.toNat).toNat.toUInt32
  else m.toNat.toUInt32

/-- Convert from Nat to FpElem -/
def ofNat (n : Nat) : FpElem := (n % PRIME.toNat).toUInt32

/-- Convert to Int (as positive integer) -/
def toInt (a : FpElem) : Int := a.toNat

/-- Convert to Nat -/
def toNat (a : FpElem) : Nat := a.toNat

instance : ToString FpElem := ⟨fun x => toString x.toNat⟩

instance : Repr FpElem := ⟨fun x _ => toString x.toNat⟩

instance : Inhabited FpElem := ⟨0⟩

instance : BEq FpElem := inferInstance

-- Test the inverse
#eval do
  let a : FpElem := 12345
  let a_inv := inv a
  let product := mul a a_inv
  IO.println s!"a = {a}, a^(-1) = {a_inv}, a * a^(-1) = {product}"
  -- Should print 1

-- Test basic operations
#eval do
  let a : FpElem := 65520  -- PRIME - 1
  let b : FpElem := 2
  IO.println s!"({a} + {b}) mod p = {add a b}"  -- Should be 1
  IO.println s!"({a} * {b}) mod p = {mul a b}"  -- Should be 65519

end CLean.FiniteField
