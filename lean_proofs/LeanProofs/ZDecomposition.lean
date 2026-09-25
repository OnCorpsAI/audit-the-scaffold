import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import LeanProofs.Setup
import LeanProofs.Theorem1

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

lemma D_sum_eq_one (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) :
  ∑ i : Fin N, D N y h α t i = 1 :=
by
  unfold D
  rw [←sum_div]
  unfold exp_loss
  have h_eq : (N : ℝ) * ((1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * f N h α t i)) = ∑ i : Fin N, exp (-y i * f N h α t i) := by
    calc
      (N : ℝ) * ((1 / (N : ℝ)) * ∑ i : Fin N, exp (-y i * f N h α t i))
        = ((N : ℝ) * (1 / (N : ℝ))) * ∑ i : Fin N, exp (-y i * f N h α t i) := by ring
      _ = 1 * ∑ i : Fin N, exp (-y i * f N h α t i) := by
          have hN : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
          rw [mul_one_div_cancel hN]
      _ = ∑ i : Fin N, exp (-y i * f N h α t i) := by ring
  rw [h_eq]
  have h_sum_pos : 0 < ∑ i : Fin N, exp (-y i * f N h α t i) := by
    apply sum_pos
    · intro j _
      exact exp_pos _
    · exact univ_nonempty
  exact div_self (ne_of_gt h_sum_pos)

/-- Weighted error ε_t -/
def err_t (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i

lemma z_decomposition (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
  (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1) :
  Z N y h α t = (1 - err_t N y h α t) * exp (-α t) + err_t N y h α t * exp (α t) :=
by
  unfold Z err_t
  have h_univ : (univ : Finset (Fin N)) = filter (fun i => y i * h t i = 1) univ ∪ filter (fun i => y i * h t i = -1) univ := by
    ext a
    simp only [mem_univ, mem_union, mem_filter, true_and]
    exact iff_of_true trivial (h_binary a)
  have h_disjoint : Disjoint (filter (fun i => y i * h t i = 1) univ) (filter (fun i => y i * h t i = -1) univ) := by
    rw [disjoint_left]
    intro a ha hb
    simp only [mem_filter, mem_univ, true_and] at ha hb
    linarith
  have h_Z_split : ∑ i : Fin N, D N y h α t i * exp (-y i * α t * h t i) =
    (∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i * exp (-y i * α t * h t i)) +
    (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i * exp (-y i * α t * h t i)) := by
    have h_sum_univ : ∑ i : Fin N, D N y h α t i * exp (-y i * α t * h t i) =
                      ∑ i ∈ univ, D N y h α t i * exp (-y i * α t * h t i) := rfl
    rw [h_sum_univ]
    nth_rw 1 [h_univ]
    exact sum_union h_disjoint
  have h_sum1 : (∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i * exp (-y i * α t * h t i)) =
    (∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i) * exp (-α t) := by
    have h_eq : ∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i * exp (-y i * α t * h t i) =
                ∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i * exp (-α t) := by
      apply sum_congr rfl
      intro i hi
      simp only [mem_filter, mem_univ, true_and] at hi
      have h_exp : -y i * α t * h t i = -α t := by
        calc
          -y i * α t * h t i = -(y i * h t i) * α t := by ring
          _ = -(1) * α t := by rw [hi]
          _ = -α t := by ring
      rw [h_exp]
    rw [h_eq]
    rw [sum_mul]
  have h_sum2 : (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i * exp (-y i * α t * h t i)) =
    (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i) * exp (α t) := by
    have h_eq : ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i * exp (-y i * α t * h t i) =
                ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i * exp (α t) := by
      apply sum_congr rfl
      intro i hi
      simp only [mem_filter, mem_univ, true_and] at hi
      have h_exp : -y i * α t * h t i = α t := by
        calc
          -y i * α t * h t i = -(y i * h t i) * α t := by ring
          _ = -(-1) * α t := by rw [hi]
          _ = α t := by ring
      rw [h_exp]
    rw [h_eq]
    rw [sum_mul]
  rw [h_sum1, h_sum2] at h_Z_split
  have h_D_sum : (∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i) +
                 (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i) = 1 := by
    have h_sum_all := D_sum_eq_one N y h α t
    have h_sum_univ : ∑ i : Fin N, D N y h α t i = ∑ i ∈ univ, D N y h α t i := rfl
    rw [h_sum_univ] at h_sum_all
    rw [←sum_union h_disjoint]
    have h_eq : filter (fun i => y i * h t i = 1) univ ∪ filter (fun i => y i * h t i = -1) univ = univ := h_univ.symm
    rw [h_eq]
    exact h_sum_all
  have h_corr : (∑ i ∈ filter (fun i => y i * h t i = 1) univ, D N y h α t i) = 1 - (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i) := by linarith
  rw [h_Z_split, h_corr]
