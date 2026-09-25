import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Setup

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.induction false

noncomputable section

/--
The exponential loss is always strictly positive.
Note: We avoid the `positivity` tactic for `0 < 1 / (N : ℝ)` because Mathlib4's `positivity`
currently struggles to automatically deduce strict positivity from `[NeZero N]` without 
explicit typeclass unbundling. The explicit proof using `one_div_pos` is more robust.
-/
lemma exp_loss_pos (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) :
  0 < exp_loss N y h α T :=
by
  unfold exp_loss
  apply mul_pos
  · have hN : 0 < (N : ℝ) := Nat.cast_pos.mpr (NeZero.pos N)
    exact one_div_pos.mpr hN
  · apply sum_pos
    · intro j _
      exact exp_pos (-y j * f N h α T j)
    · exact univ_nonempty

/--
Theorem 1: Training Error of Agentic Boosting
The exponential loss after T rounds is the product of the Z_t factors.
-/
theorem thm1_training_error_bound (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) :
  exp_loss N y h α T = ∏ t ∈ range T, Z N y h α t :=
by
  induction' T with T ih
  · unfold exp_loss f
    simp
  · calc
      exp_loss N y h α (T + 1)
        = (1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * f N h α (T + 1) i) := by rfl
      _ = (1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * (f N h α T i + α T * h T i)) := by
          have hf : ∀ i, f N h α (T + 1) i = f N h α T i + α T * h T i := by
            intro i
            unfold f
            rw [sum_range_succ]
          simp_rw [hf]
      _ = (1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * f N h α T i) * exp (-y i * α T * h T i) := by
          have h_exp : ∀ i, exp (-y i * (f N h α T i + α T * h T i)) = exp (-y i * f N h α T i) * exp (-y i * α T * h T i) := by
            intro i
            have h_mul : -y i * (f N h α T i + α T * h T i) = -y i * f N h α T i + -y i * α T * h T i := by ring
            rw [h_mul, exp_add]
          simp_rw [h_exp]
      _ = exp_loss N y h α T * Z N y h α T := by
          unfold Z D
          rw [mul_sum, mul_sum]
          apply sum_congr rfl
          intro i _
          have h_exp_pos : 0 < exp_loss N y h α T := exp_loss_pos N y h α T
          have h_eq : exp_loss N y h α T * (exp (-y i * f N h α T i) / (↑N * exp_loss N y h α T) * exp (-y i * α T * h T i)) = 1 / ↑N * (exp (-y i * f N h α T i) * exp (-y i * α T * h T i)) := by
            calc
              exp_loss N y h α T * (exp (-y i * f N h α T i) / (↑N * exp_loss N y h α T) * exp (-y i * α T * h T i))
                = (exp_loss N y h α T / exp_loss N y h α T) * (exp (-y i * f N h α T i) / ↑N * exp (-y i * α T * h T i)) := by ring
              _ = 1 * (exp (-y i * f N h α T i) / ↑N * exp (-y i * α T * h T i)) := by rw [div_self (ne_of_gt h_exp_pos)]
              _ = 1 / ↑N * (exp (-y i * f N h α T i) * exp (-y i * α T * h T i)) := by ring
          exact h_eq.symm
      _ = (∏ t ∈ range T, Z N y h α t) * Z N y h α T := by rw [ih]
      _ = ∏ t ∈ range (T + 1), Z N y h α t := by rw [prod_range_succ]

lemma indicator_le_exp (x : ℝ) : (if x ≤ 0 then (1 : ℝ) else 0) ≤ exp (-x) :=
by
  split_ifs with h
  · have h_neg : 0 ≤ -x := neg_nonneg.mpr h
    have h_one : (1 : ℝ) = exp 0 := exp_zero.symm
    rw [h_one]
    exact exp_monotone h_neg
  · exact le_of_lt (exp_pos _)

/-- The zero-one loss after T rounds. -/
def zero_one_loss (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) : ℝ :=
  (1 / (N : ℝ)) * ∑ i : Fin N, if y i * f N h α T i ≤ 0 then 1 else 0

/--
Theorem 1 (extension): 0-1 loss is bounded by the exponential loss.
-/
theorem thm1_zero_one_loss_bound (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (T : ℕ) :
  zero_one_loss N y h α T ≤ ∏ t ∈ range T, Z N y h α t :=
by
  have h_bound : zero_one_loss N y h α T ≤ exp_loss N y h α T := by
    unfold zero_one_loss exp_loss
    apply mul_le_mul_of_nonneg_left
    · apply sum_le_sum
      intro i _
      have h_ind := indicator_le_exp (y i * f N h α T i)
      have h_neg_mul : -(y i * f N h α T i) = -y i * f N h α T i := by ring
      rw [h_neg_mul] at h_ind
      exact h_ind
    · apply le_of_lt
      exact one_div_pos.mpr (Nat.cast_pos.mpr (NeZero.pos N))
  rw [thm1_training_error_bound] at h_bound
  exact h_bound

end
