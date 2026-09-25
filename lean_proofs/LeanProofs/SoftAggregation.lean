import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Theorem1
import LeanProofs.Corollary2

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/--
Pure-analysis fact underlying the soft-aggregation extension of Prop. 15 (§F.3): the
AdaBoost-style exponential error bound `exp(-2γ²k)` eventually drops strictly below the
weak-learning threshold `1/2 - γ` that any single weak learner is only *guaranteed*, not
guaranteed to beat. This is what makes combining `k` diverse weak workers strictly stronger
than relying on any one of them, once `k` is large enough.
-/
theorem exists_ensemble_size_beating_weak_threshold (γ : ℝ) (hγ_pos : 0 < γ) (hγ_lt_half : γ < 1 / 2) :
    ∃ k₀ : ℕ, ∀ k : ℕ, k ≥ k₀ → exp (-2 * γ ^ 2 * k) < 1 / 2 - γ :=
by
  set δ := 1 / 2 - γ with hδ_def
  have hδ_pos : 0 < δ := by rw [hδ_def]; linarith
  have hγsq_pos : 0 < 2 * γ ^ 2 := by positivity
  obtain ⟨k₀, hk₀⟩ := exists_nat_gt (-(Real.log δ) / (2 * γ ^ 2))
  refine ⟨k₀, fun k hk => ?_⟩
  have hk' : (k₀ : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk
  have h1 : -(Real.log δ) / (2 * γ ^ 2) < (k : ℝ) := lt_of_lt_of_le hk₀ hk'
  have h2 : -(Real.log δ) < (k : ℝ) * (2 * γ ^ 2) := (div_lt_iff₀ hγsq_pos).mp h1
  have h3 : -2 * γ ^ 2 * (k : ℝ) < Real.log δ := by nlinarith
  exact (lt_log_iff_exp_lt hδ_pos).mp h3

/--
Soft-aggregation extension of Prop. 15 (§F.3): combine `k` frozen workers via the AdaBoost
weighted vote (Algorithm 1) rather than best-of-`k` hard selection --- treat the `k` workers as
the sequence `h_0, ..., h_{k-1}` incorporated in that order, each maintaining weighted error
`err_t ≤ 1/2 - γ` against the reweighted distribution current when it is incorporated
(`h_edge`, `t` ranges over *every* natural number here rather than only `range T` since we
quantify over all prefixes `k`). Then for large enough `k`, the ensemble's zero-one training
error is *strictly below* `1/2 - γ` --- the weak-learning threshold each individual worker is
only guaranteed to satisfy, not guaranteed to beat.

This answers the open question flagged in the paper (§F.3, "What this still does not resolve"): unlike
best-of-`k` hard selection (Prop. 15, `prop15_tight_ceiling_orchestration`, which provably ties
the best single worker and never exceeds it), weighted aggregation *can* achieve a genuine edge
over what any single worker is guaranteed to reach --- but only under the `h_edge` diversity
condition, which requires each worker to keep beating the *reweighted* distribution that
up-weights the points previous workers got wrong. Workers with correlated failure modes (e.g.
identical copies) fail this hypothesis, not via an extra side-condition bolted on afterward:
AdaBoost's own stopping rule (`ε_t ≥ 1/2 ⟹ α_t ≤ 0`, Algorithm 1) already gates them out, so
diversity is encoded in `h_edge` itself.
-/
theorem cor_soft_aggregation_beats_weak_baseline (N : ℕ) [NeZero N] (y : Fin N → ℝ)
    (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (γ : ℝ) (hγ_pos : 0 < γ) (hγ_lt_half : γ < 1 / 2)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (h_eps_pos : ∀ t, 0 < err_t N y h α t)
    (h_eps_lt_one : ∀ t, err_t N y h α t < 1)
    (h_alpha_opt : ∀ t, α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t))
    (h_edge : ∀ t, err_t N y h α t ≤ 1 / 2 - γ) :
    ∃ k₀ : ℕ, ∀ k : ℕ, k ≥ k₀ → zero_one_loss N y h α k < 1 / 2 - γ :=
by
  obtain ⟨k₀, hk₀⟩ := exists_ensemble_size_beating_weak_threshold γ hγ_pos hγ_lt_half
  refine ⟨k₀, fun k hk => ?_⟩
  have h1 : zero_one_loss N y h α k ≤ ∏ t ∈ range k, Z N y h α t :=
    thm1_zero_one_loss_bound N y h α k
  have h2 : (∏ t ∈ range k, Z N y h α t) = exp_loss N y h α k :=
    (thm1_training_error_bound N y h α k).symm
  have h3 : exp_loss N y h α k ≤ exp (-2 * γ ^ 2 * k) :=
    cor2_exponential_decay N y h α k γ (fun t _ => h_binary t) hγ_pos.le
      (fun t _ => h_eps_pos t) (fun t _ => h_eps_lt_one t) (fun t _ => h_alpha_opt t)
      (fun t _ => h_edge t)
  have h4 : zero_one_loss N y h α k ≤ exp (-2 * γ ^ 2 * k) :=
    calc zero_one_loss N y h α k ≤ ∏ t ∈ range k, Z N y h α t := h1
    _ = exp_loss N y h α k := h2
    _ ≤ exp (-2 * γ ^ 2 * k) := h3
  exact lt_of_le_of_lt h4 (hk₀ k hk)

end
