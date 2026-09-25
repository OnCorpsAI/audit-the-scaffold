import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Setup
import LeanProofs.Theorem1
import LeanProofs.Theorem4
import LeanProofs.ZDecomposition

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

lemma z_le_sqrt_edge (ε γ : ℝ) (hε_pos : 0 < ε) (hε_edge : ε ≤ 1 / 2 - γ) (hγ_nonneg : 0 ≤ γ) :
  2 * sqrt (ε * (1 - ε)) ≤ sqrt (1 - 4 * γ^2) :=
by
  have h_eps_le : ε ≤ 1 / 2 := by linarith
  have h_eps_prod : ε * (1 - ε) = 1 / 4 - (1 / 2 - ε)^2 := by ring
  have h_gamma_sq : (1 / 2 - ε)^2 ≥ γ^2 := by
    have h1 : 1 / 2 - ε ≥ γ := by linarith
    nlinarith
  have h_eps_prod_le : ε * (1 - ε) ≤ 1 / 4 - γ^2 := by linarith
  have h_nonneg : 0 ≤ ε * (1 - ε) := mul_nonneg (le_of_lt hε_pos) (by linarith)
  have h_sqrt_le : sqrt (ε * (1 - ε)) ≤ sqrt (1 / 4 - γ^2) := sqrt_le_sqrt h_eps_prod_le
  calc
    2 * sqrt (ε * (1 - ε)) ≤ 2 * sqrt (1 / 4 - γ^2) := by linarith
    _ = sqrt 4 * sqrt (1 / 4 - γ^2) := by 
      have h4 : (2 : ℝ) = sqrt 4 := by norm_num
      rw [h4]
    _ = sqrt (4 * (1 / 4 - γ^2)) := by rw [←sqrt_mul zero_le_four]
    _ = sqrt (1 - 4 * γ^2) := by ring_nf

lemma sqrt_edge_le_exp (γ : ℝ) :
  sqrt (1 - 4 * γ^2) ≤ exp (-2 * γ^2) :=
by
  have h_cases : 1 - 4 * γ^2 ≤ 0 ∨ 0 ≤ 1 - 4 * γ^2 := le_total (1 - 4 * γ^2) 0
  rcases h_cases with h_nonpos | h_nonneg
  · rw [sqrt_eq_zero_of_nonpos h_nonpos]
    exact le_of_lt (exp_pos _)
  · have h_sq_left : sqrt (1 - 4 * γ^2) ^ 2 = 1 - 4 * γ^2 := sq_sqrt h_nonneg
    have h_sq_right : exp (-2 * γ^2) ^ 2 = exp (-4 * γ^2) := by
      rw [←exp_nat_mul]
      ring_nf
    have h_exp_bound := add_one_le_exp (-4 * γ^2)
    have h_sq_le : sqrt (1 - 4 * γ^2) ^ 2 ≤ exp (-2 * γ^2) ^ 2 := by
      rw [h_sq_left, h_sq_right]
      have h_eq : -4 * γ^2 + 1 = 1 - 4 * γ^2 := by ring
      rw [h_eq] at h_exp_bound
      exact h_exp_bound
    have h_mul_le : sqrt (1 - 4 * γ^2) * sqrt (1 - 4 * γ^2) ≤ exp (-2 * γ^2) * exp (-2 * γ^2) := by
      calc
        sqrt (1 - 4 * γ^2) * sqrt (1 - 4 * γ^2) = sqrt (1 - 4 * γ^2) ^ 2 := by ring
        _ ≤ exp (-2 * γ^2) ^ 2 := h_sq_le
        _ = exp (-2 * γ^2) * exp (-2 * γ^2) := by ring
    exact nonneg_le_nonneg_of_sq_le_sq (le_of_lt (exp_pos _)) h_mul_le

lemma z_bound_from_edge (ε γ : ℝ) (hε_pos : 0 < ε) (hε_edge : ε ≤ 1 / 2 - γ) (hγ_nonneg : 0 ≤ γ) :
  2 * sqrt (ε * (1 - ε)) ≤ exp (-2 * γ^2) :=
by
  exact le_trans (z_le_sqrt_edge ε γ hε_pos hε_edge hγ_nonneg) (sqrt_edge_le_exp γ)

/--
Corollary 2: Exponential Decay of Training Error
If each weak coding agent has a bounded edge γ, the training error decays exponentially.

We now formalize the connection between Z_t, the weighted error, and the optimal α_t
using Theorem 4 and the Z decomposition.
-/
theorem cor2_exponential_decay (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) (γ : ℝ)
  (h_binary : ∀ t ∈ range T, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
  (h_gamma_nonneg : 0 ≤ γ)
  (h_eps_pos : ∀ t ∈ range T, 0 < err_t N y h α t)
  (h_eps_lt_one : ∀ t ∈ range T, err_t N y h α t < 1)
  (h_alpha_opt : ∀ t ∈ range T, α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t))
  (h_edge : ∀ t ∈ range T, err_t N y h α t ≤ 1 / 2 - γ) :
  exp_loss N y h α T ≤ exp (-2 * γ^2 * T) :=
by
  rw [thm1_training_error_bound]
  have h_prod : ∏ t ∈ range T, Z N y h α t ≤ ∏ t ∈ range T, exp (-2 * γ^2) := by
    apply prod_le_prod
    · intro i _
      unfold Z
      apply sum_nonneg
      intro j _
      apply mul_nonneg
      · unfold D
        apply div_nonneg
        · positivity
        · apply mul_nonneg
          · positivity
          · unfold exp_loss
            apply mul_nonneg
            · positivity
            · apply sum_nonneg
              intro k _
              positivity
      · positivity
    · intro t ht
      have h_z_decomp := z_decomposition N y h α t (h_binary t ht)
      have h_opt := thm4_fsam_optimal_alpha (err_t N y h α t) (h_eps_pos t ht) (h_eps_lt_one t ht)
      have h_z_eq : Z N y h α t = 2 * sqrt (err_t N y h α t * (1 - err_t N y h α t)) := by
        calc
          Z N y h α t = (1 - err_t N y h α t) * exp (-α t) + err_t N y h α t * exp (α t) := h_z_decomp
          _ = (1 - err_t N y h α t) * exp (-((1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t))) + err_t N y h α t * exp ((1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t)) := by rw [h_alpha_opt t ht]
          _ = 2 * sqrt (err_t N y h α t * (1 - err_t N y h α t)) := h_opt
      rw [h_z_eq]
      exact z_bound_from_edge (err_t N y h α t) γ (h_eps_pos t ht) (h_edge t ht) h_gamma_nonneg
  have h_exp : ∏ t ∈ range T, exp (-2 * γ^2) = exp (-2 * γ^2 * T) := by
    rw [prod_const, card_range]
    have h_pow : exp (-2 * γ^2) ^ T = exp (-2 * γ^2 * T) := by
      rw [←exp_nat_mul]
      ring_nf
    exact h_pow
  rw [←h_exp]
  exact h_prod

end
