import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.induction false

/--
Theorem 5: Monotone Improvement under Refinement (Refinement Game)
We model the expected quality of code artifacts over rounds.
-/
theorem thm5_refinement_game {X A : Type}
  (V : X → ℝ)
  (R : ℕ → X → A → X)
  (E_A : ℕ → X → (A → ℝ) → ℝ)
  (E_X : ℕ → (X → ℝ) → ℝ)
  (h_transition : ∀ t (f : X → ℝ), E_X (t + 1) f = E_X t (fun x => E_A t x (fun a => f (R t x a))))
  (h_E_X_mono : ∀ t f g, (∀ x, f x ≤ g x) → E_X t f ≤ E_X t g)
  (h_E_X_add : ∀ t f g, E_X t (fun x => f x + g x) = E_X t f + E_X t g)
  (h_E_X_const : ∀ t c, E_X t (fun _ => c) = c)
  (η : ℕ → ℝ)
  (h_local_improve : ∀ t x, E_A t x (fun a => V (R t x a)) ≥ V x + η t) :
  ∀ T, E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η t :=
by
  intro T
  induction' T with T ih
  · simp
  · have h_step : E_X (T + 1) V ≥ E_X T V + η T := by
      rw [h_transition T V]
      have h_mono := h_E_X_mono T (fun x => V x + η T) (fun x => E_A T x (fun a => V (R T x a))) (by
        intro x
        exact h_local_improve T x
      )
      rw [h_E_X_add] at h_mono
      rw [h_E_X_const] at h_mono
      exact h_mono
    calc
      E_X (T + 1) V ≥ E_X T V + η T := h_step
      _ ≥ (E_X 0 V + ∑ t ∈ range T, η t) + η T := by linarith [ih]
      _ = E_X 0 V + (∑ t ∈ range T, η t + η T) := by ring
      _ = E_X 0 V + ∑ t ∈ range (T + 1), η t := by rw [sum_range_succ]

/--
Theorem 5 (Saturation bound): If quality is bounded by 1, improvement is finite.
-/
theorem thm5_saturation {X A : Type}
  (V : X → ℝ)
  (R : ℕ → X → A → X)
  (E_A : ℕ → X → (A → ℝ) → ℝ)
  (E_X : ℕ → (X → ℝ) → ℝ)
  (h_transition : ∀ t (f : X → ℝ), E_X (t + 1) f = E_X t (fun x => E_A t x (fun a => f (R t x a))))
  (h_E_X_mono : ∀ t f g, (∀ x, f x ≤ g x) → E_X t f ≤ E_X t g)
  (h_E_X_add : ∀ t f g, E_X t (fun x => f x + g x) = E_X t f + E_X t g)
  (h_E_X_const : ∀ t c, E_X t (fun _ => c) = c)
  (η : ℕ → ℝ)
  (h_local_improve : ∀ t x, E_A t x (fun a => V (R t x a)) ≥ V x + η t)
  (h_bound : ∀ x, V x ≤ 1) :
  ∀ T, ∑ t ∈ range T, η t ≤ 1 - E_X 0 V :=
by
  intro T
  have h1 : E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η t := thm5_refinement_game V R E_A E_X h_transition h_E_X_mono h_E_X_add h_E_X_const η h_local_improve T
  have h2 : E_X T V ≤ 1 := by
    have h_const := h_E_X_const T 1
    have h_mono := h_E_X_mono T V (fun _ => 1) h_bound
    rw [←h_const]
    exact h_mono
  linarith
