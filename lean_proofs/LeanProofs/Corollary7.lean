import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Tactic

open Real
open Finset
open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

set_option linter.style.header false
set_option linter.style.longLine false
set_option linter.style.induction false

omit [IsProbabilityMeasure μ] in
lemma prob_inter_bound (E : ℕ → Set Ω) (p : ℕ → ENNReal)
  (h_Pr_E : ∀ t, μ (E t)ᶜ ≤ p t) :
  ∀ T : ℕ, μ (⋂ t ∈ range T, E t)ᶜ ≤ ∑ t ∈ range T, p t :=
by
  intro T
  have h_eq : (⋂ t ∈ range T, E t)ᶜ = ⋃ t ∈ range T, (E t)ᶜ := by
    ext ω
    simp
  rw [h_eq]
  calc
    μ (⋃ t ∈ range T, (E t)ᶜ) ≤ ∑ t ∈ range T, μ (E t)ᶜ := measure_biUnion_finset_le _ _
    _ ≤ ∑ t ∈ range T, p t := sum_le_sum (fun t _ => h_Pr_E t)

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma sum_telescope (V : ℕ → Ω → ℝ) (ω : Ω) :
  ∀ T, ∑ t ∈ range T, (V (t + 1) ω - V t ω) = V T ω - V 0 ω :=
by
  intro T
  induction' T with T ih
  · simp
  · rw [sum_range_succ, ih]
    ring

omit [IsProbabilityMeasure μ] in
/--
Corollary 7: High-Probability Refinement
In practice, LLM coding agents may occasionally regress code quality. Suppose instead of a strict expectation, the refinement operators satisfy a probabilistic improvement property: for each round t, Pr[V(x_t) - V(x_{t-1}) >= \eta_t] >= 1 - p_t.
Then, after T rounds: Pr[V(x_T) >= V(x_0) + \sum_{t=1}^T \eta_t] >= 1 - \sum_{t=1}^T p_t.
-/
theorem corollary7_high_prob_refinement
  (V : ℕ → Ω → ℝ)
  (η : ℕ → ℝ)
  (p : ℕ → ENNReal)
  (h_Pr_improve : ∀ t, μ {ω | V (t + 1) ω - V t ω ≥ η t}ᶜ ≤ p t) :
  ∀ T : ℕ, μ {ω | V T ω ≥ V 0 ω + ∑ t ∈ range T, η t}ᶜ ≤ ∑ t ∈ range T, p t :=
by
  intro T
  let E := fun t => {ω | V (t + 1) ω - V t ω ≥ η t}
  have h_bound := prob_inter_bound μ E p h_Pr_improve T
  have h_subset : {ω | V T ω ≥ V 0 ω + ∑ t ∈ range T, η t}ᶜ ⊆ (⋂ t ∈ range T, E t)ᶜ := by
    intro ω h_not_sum
    simp only [Set.mem_compl_iff, Set.mem_iInter, mem_range, not_forall] at *
    simp only [Set.mem_setOf_eq] at h_not_sum
    by_contra h_all_E
    push Not at h_all_E
    have h_sum : ∑ t ∈ range T, (V (t + 1) ω - V t ω) = V T ω - V 0 ω := sum_telescope V ω T
    have h_sum_le : ∑ t ∈ range T, η t ≤ ∑ t ∈ range T, (V (t + 1) ω - V t ω) := by
      apply sum_le_sum
      intro i hi
      exact h_all_E i (mem_range.mp hi)
    linarith
  calc
    μ {ω | V T ω ≥ V 0 ω + ∑ t ∈ range T, η t}ᶜ ≤ μ (⋂ t ∈ range T, E t)ᶜ := measure_mono h_subset
    _ ≤ ∑ t ∈ range T, p t := h_bound
