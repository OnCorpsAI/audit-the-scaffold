import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/-- The unnormalized margin after T rounds. -/
def f (N : ℕ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) (i : Fin N) : ℝ :=
  ∑ t ∈ range T, α t * h t i

/-- The exponential loss after T rounds. 
    Note: In standard AdaBoost, labels `y i` and hypotheses `h t i` take values in `{-1, 1}`.
    Our formalization proves algebraic bounds that hold for any real values. -/
def exp_loss (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) : ℝ :=
  (1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * f N h α T i)

/-- The probability distribution over samples at round t. 
    Note: Lean's division by zero safely returns 0 if `exp_loss` is 0, 
    but mathematically we know `exp_loss > 0` (this is proved locally where D is used). -/
def D (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) (i : Fin N) : ℝ :=
  exp (-y i * f N h α t i) / (N * exp_loss N y h α t)

/-- The normalization factor Z at round t. -/
def Z (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ i : Fin N, D N y h α t i * exp (-y i * α t * h t i)

end
