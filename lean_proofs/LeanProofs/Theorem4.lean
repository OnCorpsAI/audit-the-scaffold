import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

theorem exp_half_log (x : ℝ) (hx : 0 < x) : exp ((1 / 2) * log x) = sqrt x :=
by
  have h1 : exp ((1 / 2) * log x) * exp ((1 / 2) * log x) = x := by
    rw [←exp_add]
    have h_add : (1 / 2 : ℝ) * log x + (1 / 2) * log x = log x := by ring
    rw [h_add]
    exact exp_log hx
  have h2 : sqrt (exp ((1 / 2) * log x) * exp ((1 / 2) * log x)) = sqrt x := congrArg sqrt h1
  have h3 : sqrt (exp ((1 / 2) * log x) * exp ((1 / 2) * log x)) = exp ((1 / 2) * log x) := by
    apply sqrt_mul_self
    exact le_of_lt (exp_pos _)
  rw [h3] at h2
  exact h2

/--
Theorem 4: Boosting as Greedy Coordinate Descent
We show that the optimal weight α minimizes the exponential loss objective.
-/
theorem thm4_fsam_optimal_alpha (ε : ℝ) (h_pos : 0 < ε) (h_lt_one : ε < 1) :
  let α := (1 / 2) * log ((1 - ε) / ε)
  (1 - ε) * exp (-α) + ε * exp α = 2 * sqrt (ε * (1 - ε)) :=
by
  intro α
  have h_eps_ratio_pos : 0 < (1 - ε) / ε := div_pos (by linarith) h_pos
  have h_exp1 : exp ((1 / 2) * log ((1 - ε) / ε)) = sqrt ((1 - ε) / ε) := exp_half_log ((1 - ε) / ε) h_eps_ratio_pos
  have h_exp2 : exp (-((1 / 2) * log ((1 - ε) / ε))) = sqrt (ε / (1 - ε)) := by
    rw [exp_neg, h_exp1]
    have h_div_inv : ((1 - ε) / ε)⁻¹ = ε / (1 - ε) := inv_div _ _
    rw [←sqrt_inv, h_div_inv]
  
  have part1 : (1 - ε) * (sqrt ε / sqrt (1 - ε)) = sqrt (ε * (1 - ε)) := by
    have h_sub_pos : 0 ≤ 1 - ε := by linarith
    have h_rw : 1 - ε = sqrt (1 - ε) * sqrt (1 - ε) := (mul_self_sqrt h_sub_pos).symm
    calc
      (1 - ε) * (sqrt ε / sqrt (1 - ε))
        = (sqrt (1 - ε) * sqrt (1 - ε)) * (sqrt ε / sqrt (1 - ε)) := by nth_rw 1 [h_rw]
      _ = sqrt (1 - ε) * sqrt ε * (sqrt (1 - ε) / sqrt (1 - ε)) := by ring
      _ = sqrt (1 - ε) * sqrt ε * 1 := by
        have h_ne : sqrt (1 - ε) ≠ 0 := ne_of_gt (sqrt_pos.mpr (by linarith))
        rw [div_self h_ne]
      _ = sqrt ((1 - ε) * ε) := by rw [mul_one, ←sqrt_mul h_sub_pos]
      _ = sqrt (ε * (1 - ε)) := by
        have h_comm : (1 - ε) * ε = ε * (1 - ε) := mul_comm _ _
        rw [h_comm]
  -- Mirror image of part1, for the ε·exp α half of the objective.
  have part2 : ε * (sqrt (1 - ε) / sqrt ε) = sqrt (ε * (1 - ε)) := by
    have h_eps_nonneg : 0 ≤ ε := le_of_lt h_pos
    have h_rw : ε = sqrt ε * sqrt ε := (mul_self_sqrt h_eps_nonneg).symm
    calc
      ε * (sqrt (1 - ε) / sqrt ε)
        = (sqrt ε * sqrt ε) * (sqrt (1 - ε) / sqrt ε) := by nth_rw 1 [h_rw]
      _ = sqrt ε * sqrt (1 - ε) * (sqrt ε / sqrt ε) := by ring
      _ = sqrt ε * sqrt (1 - ε) * 1 := by
        have h_ne : sqrt ε ≠ 0 := ne_of_gt (sqrt_pos.mpr h_pos)
        rw [div_self h_ne]
      _ = sqrt (ε * (1 - ε)) := by rw [mul_one, ←sqrt_mul h_eps_nonneg]
  -- Both halves equal √(ε(1-ε)), so at the optimal α the objective is twice it.
  calc
    (1 - ε) * exp (-α) + ε * exp α
      = (1 - ε) * exp (-((1 / 2) * log ((1 - ε) / ε))) + ε * exp ((1 / 2) * log ((1 - ε) / ε)) := by rfl
    _ = (1 - ε) * sqrt (ε / (1 - ε)) + ε * sqrt ((1 - ε) / ε) := by rw [h_exp1, h_exp2]
    _ = (1 - ε) * (sqrt ε / sqrt (1 - ε)) + ε * (sqrt (1 - ε) / sqrt ε) := by
        rw [sqrt_div (le_of_lt h_pos), sqrt_div (by linarith)]
    _ = sqrt (ε * (1 - ε)) + sqrt (ε * (1 - ε)) := by rw [part1, part2]
    _ = 2 * sqrt (ε * (1 - ε)) := by ring

/--
Theorem 4 (extension): The exponential loss objective is globally minimized.
-/
theorem thm4_fsam_global_minimum (ε : ℝ) (h_pos : 0 < ε) (h_lt_one : ε < 1) (α : ℝ) :
  2 * sqrt (ε * (1 - ε)) ≤ (1 - ε) * exp (-α) + ε * exp α :=
by
  have a_pos : 0 ≤ (1 - ε) * exp (-α) := mul_nonneg (by linarith) (le_of_lt (exp_pos _))
  have b_pos : 0 ≤ ε * exp α := by positivity
  
  have am_gm : 2 * sqrt (((1 - ε) * exp (-α)) * (ε * exp α)) ≤ (1 - ε) * exp (-α) + ε * exp α := by
    have h_sq : 0 ≤ (sqrt ((1 - ε) * exp (-α)) - sqrt (ε * exp α)) ^ 2 := sq_nonneg _
    have h_expand : (sqrt ((1 - ε) * exp (-α)) - sqrt (ε * exp α)) ^ 2 =
      ((1 - ε) * exp (-α)) - 2 * sqrt (((1 - ε) * exp (-α)) * (ε * exp α)) + (ε * exp α) := by
      calc
        (sqrt ((1 - ε) * exp (-α)) - sqrt (ε * exp α)) ^ 2
          = sqrt ((1 - ε) * exp (-α)) ^ 2 - 2 * (sqrt ((1 - ε) * exp (-α)) * sqrt (ε * exp α)) + sqrt (ε * exp α) ^ 2 := by ring
        _ = (1 - ε) * exp (-α) - 2 * (sqrt ((1 - ε) * exp (-α)) * sqrt (ε * exp α)) + ε * exp α := by rw [sq_sqrt a_pos, sq_sqrt b_pos]
        _ = (1 - ε) * exp (-α) - 2 * sqrt (((1 - ε) * exp (-α)) * (ε * exp α)) + ε * exp α := by rw [←sqrt_mul a_pos]
    linarith [h_expand]
  -- The product under AM-GM's square root collapses, since exp(-α)·exp α = 1.
  have h_prod : ((1 - ε) * exp (-α)) * (ε * exp α) = ε * (1 - ε) := by
    calc
      ((1 - ε) * exp (-α)) * (ε * exp α) = (1 - ε) * ε * (exp (-α) * exp α) := by ring
      _ = (1 - ε) * ε * exp (-α + α) := by rw [←exp_add]
      _ = (1 - ε) * ε * exp 0 := by ring_nf
      _ = (1 - ε) * ε * 1 := by rw [exp_zero]
      _ = ε * (1 - ε) := by ring
  -- Substituting that product into am_gm is exactly the goal.
  rw [h_prod] at am_gm
  exact am_gm

end
