import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Archimedean.Basic
import Mathlib.Tactic
import LeanProofs.Theorem5

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

/--
Proposition 8 (Stationarity Dichotomy for Self-Improvement), part (i):
If code quality is bounded by a *fixed* ceiling `B`, then the total improvement is
finite (`∑ η_t ≤ B - E[V(x_0)]`), forcing the per-round improvements `η_t` to decay to
zero. This is the diminishing-returns / stationary regime; it generalizes the saturation
bound `thm5_saturation` from the ceiling `1` to an arbitrary ceiling `B`.
-/
theorem prop8_saturation_forces_decay {X A : Type}
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
  (B : ℝ)
  (h_bound : ∀ x, V x ≤ B) :
  ∀ T, ∑ t ∈ range T, η t ≤ B - E_X 0 V :=
by
  intro T
  have h1 : E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η t :=
    thm5_refinement_game V R E_A E_X h_transition h_E_X_mono h_E_X_add h_E_X_const η h_local_improve T
  have h2 : E_X T V ≤ B := by
    have h_const := h_E_X_const T B
    have h_mono := h_E_X_mono T V (fun _ => B) h_bound
    rw [←h_const]
    exact h_mono
  linarith

/--
Proposition 8, part (ii):
If the effective hypothesis class expands enough to sustain a *uniform* positive edge
`η_t ≥ c > 0` for every round, then expected quality grows without bound
(`∀ B, ∃ T, E[V(x_T)] > B`). Hence no fixed ceiling can hold and improvement does not
saturate — the non-stationary / potential runaway regime. This is a direct consequence of
the refinement-game bound `thm5_refinement_game` (which assumes no upper bound on `V`)
together with the Archimedean property of ℝ.
-/
theorem prop8_sustained_edge_diverges {X A : Type}
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
  (c : ℝ) (hc : c > 0)
  (h_eta_min : ∀ t, η t ≥ c) :
  ∀ B : ℝ, ∃ T, E_X T V > B :=
by
  intro B
  obtain ⟨T, hT⟩ := exists_nat_gt ((B - E_X 0 V) / c)
  refine ⟨T, ?_⟩
  have h1 : E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η t :=
    thm5_refinement_game V R E_A E_X h_transition h_E_X_mono h_E_X_add h_E_X_const η h_local_improve T
  have h2 : ∑ t ∈ range T, (c : ℝ) ≤ ∑ t ∈ range T, η t := by
    apply sum_le_sum
    intro i _
    exact h_eta_min i
  have h3 : ∑ t ∈ range T, (c : ℝ) = (T : ℝ) * c := by
    rw [sum_const, card_range, nsmul_eq_mul]
  have h4 : B - E_X 0 V < (T : ℝ) * c := (div_lt_iff₀ hc).mp hT
  linarith

/--
Frozen-Weight Ceiling Corollary (§5.2 RSI discussion).
Model a self-improving system as a pair `(W, C)` with frozen weights `W` and mutable
scaffold `C`. Suppose that, with `W` fixed, every artifact any scaffold can reach has quality
bounded by an ultimate ceiling `Vstar := V*(W)` (`h_ceiling`). Then the total improvement is
finite (`∑ η_t ≤ Vstar - E[V(x_0)]`), which forces `η_t → 0`: the frozen-weight system saturates
at some value `≤ Vstar` — it cannot escape diminishing returns without raising the ceiling itself.

This is a direct instantiation of `prop8_saturation_forces_decay` with `B := Vstar`; the ceiling
`Vstar` plays the role of the generic bound `B`. Escaping this regime (unbounded RSI) is the
contrapositive: by `prop8_sustained_edge_diverges`, sustaining a uniform edge `η_t ≥ c > 0` makes
`E[V(x_T)]` exceed *every* real bound, so no fixed `Vstar` can hold — raising `V*(W)` (modifying
weights or architecture) is necessary. The existence of `Vstar` and the expandability of the
effective class are modeling inputs, not consequences of this corollary.
-/
theorem cor_frozen_weight_ceiling {X A : Type}
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
  (Vstar : ℝ)
  (h_ceiling : ∀ x, V x ≤ Vstar) :
  ∀ T, ∑ t ∈ range T, η t ≤ Vstar - E_X 0 V :=
  prop8_saturation_forces_decay V R E_A E_X h_transition h_E_X_mono
    h_E_X_add h_E_X_const η h_local_improve Vstar h_ceiling
