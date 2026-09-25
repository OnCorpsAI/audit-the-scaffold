import Mathlib.Probability.Distributions.Bernoulli
import Mathlib.Probability.Independence.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic
import LeanProofs.Condorcet

open Real
open Finset
open MeasureTheory ProbabilityTheory unitInterval

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/-!
### Condorcet with independence *derived* rather than encoded

`Condorcet.lean` proves the majority-vote bound with the failure-pattern weight `wt` *defined* as a
product. That product form is the independence assumption, and its module docstring says so: the
word "independent" in `thm_condorcet_majority_bound` is a modelling choice, not a theorem.

This file discharges it. Workers become coordinates of a genuine product of Bernoulli measures on
`FailPattern k = Fin k → Bool`; independence is then a *consequence*, supplied by Mathlib's
`iIndepFun_pi` (`lem_coord_iIndepFun`). The bound itself is not re-proved: `lem_pattern_measure`
identifies the measure of a single failure pattern with `wt`, and `lem_failure_event_eq_voteErr`
identifies the measure of the majority-failure event with `voteErr`, after which
`thm_condorcet_majority_bound` transports verbatim.

**Why an identification rather than Hoeffding.** Mathlib's sub-Gaussian machinery
(`measure_sum_ge_le_of_iIndepFun`) would give the bound directly, at the cost of a centering step and
measurability side conditions throughout. Identifying `voteErr` with a measure is cheaper and yields
a strictly stronger reading of the existing theorem: the combinatorial statement was *already* the
measure-theoretic one, and now that is proved rather than asserted. The sample space is a finite
product of finite spaces, so every set is measurable and the side conditions the `Condorcet.lean`
docstring warned about do not arise.

**What is still encoded.** That the workers are independent *at all* --- as opposed to the pattern
weight merely being written as a product --- is now a theorem about this measure. What remains a
modelling choice is the decision to study a product measure in the first place, which is the Condorcet
jury setup and is not something a proof can discharge. Empirically it is the assumption Part B's
workers fail: independent failures could not put `m⋆` at its ceiling `k` on every round, which is
what the overlap measurement finds (`Diversity.lean`, §B). An earlier version of this comment cited
Part B's non-positive per-round edge here; that number is withdrawn in the paper as a back-filling
artifact, and it was never the statistic that bears on independence anyway.
-/

/-- A failure pattern: which of the `k` workers fail. `true` means "fails". -/
abbrev FailPattern (k : ℕ) : Type := ∀ _ : Fin k, Bool

/--
The product of `k` independent Bernoulli failure measures, worker `i` failing with probability
`q i`. Unlike `wt`, this is an honest measure on a product space, and the independence of the
coordinates is a derivable fact about it rather than a feature of how it was written.
-/
def failMeasure (k : ℕ) (q : Fin k → I) : Measure (FailPattern k) :=
  Measure.pi fun i => bernoulliMeasure true false (q i)

instance instIsProbabilityMeasureFailMeasure (k : ℕ) (q : Fin k → I) :
    IsProbabilityMeasure (failMeasure k q) := by
  unfold failMeasure
  infer_instance

/--
**Independence, derived.** The per-worker failure indicators are independent under `failMeasure`.
This is the declaration that upgrades the word "independent" in `thm_condorcet_majority_bound` from
a modelling choice to a theorem: nothing here is definitional, it is `iIndepFun_pi` applied to the
coordinate projections of a product measure.
-/
lemma lem_coord_iIndepFun (k : ℕ) (q : Fin k → I) :
    iIndepFun (fun (i : Fin k) (ω : FailPattern k) => ω i) (failMeasure k q) :=
  iIndepFun_pi fun _ => measurable_id.aemeasurable

/-- Worker `i` fails with probability exactly `q i`. -/
lemma lem_bernoulli_true (p : I) : (bernoulliMeasure true false p).real {true} = (p : ℝ) := by
  simp

/-- Worker `i` succeeds with probability exactly `1 - q i`. -/
lemma lem_bernoulli_false (p : I) :
    (bernoulliMeasure true false p).real {false} = 1 - (p : ℝ) := by
  simp

/--
**One failure pattern weighs exactly `wt`.** The measure of a single point of the product space is
`Condorcet.lean`'s pattern weight at the corresponding failure set. `Measure.pi_singleton` does the
work: the product form of `wt` is recovered as the product a product measure actually has, rather
than being posited.
-/
lemma lem_pattern_measure (k : ℕ) (q : Fin k → I) (ω : FailPattern k) :
    (failMeasure k q).real {ω}
      = wt k (fun i => (q i : ℝ)) (univ.filter fun i => ω i = true) :=
by
  classical
  have hpi : (failMeasure k q).real {ω}
      = ∏ i, (bernoulliMeasure true false (q i)).real {ω i} := by
    simp only [failMeasure, measureReal_def, Measure.pi_singleton, ENNReal.toReal_prod]
  have hcoord : ∀ i : Fin k, (bernoulliMeasure true false (q i)).real {ω i}
      = if ω i = true then (q i : ℝ) else 1 - (q i : ℝ) := by
    intro i
    cases ω i with
    | true => simp
    | false => simp
  rw [hpi, prod_congr rfl fun i _ => hcoord i]
  unfold wt
  rw [← prod_filter_mul_prod_filter_not (univ : Finset (Fin k)) (fun i => ω i = true)
    (fun i => if ω i = true then (q i : ℝ) else 1 - (q i : ℝ))]
  congr 1
  · exact prod_congr rfl fun i hi => by
      simp only [mem_filter] at hi
      rw [if_pos hi.2]
  · rw [← filter_not]
    exact prod_congr rfl fun i hi => by
      simp only [mem_filter] at hi
      rw [if_neg hi.2]

/-- Recovering a failure set from a failure pattern and back is the identity. -/
lemma lem_filter_decide (k : ℕ) (S : Finset (Fin k)) :
    (univ.filter fun i => (decide (i ∈ S)) = true) = S :=
by
  ext i
  simp

/--
**The majority-failure event weighs exactly `voteErr`.** Summing `lem_pattern_measure` over the
patterns in which at least half the workers fail, reindexed along the bijection between failure
patterns and failure sets. So `Condorcet.lean`'s `voteErr` *is* the probability of a majority
failure under a product of Bernoullis --- not merely an analogue of it.
-/
lemma lem_failure_event_eq_voteErr (k : ℕ) (q : Fin k → I) :
    (failMeasure k q).real
        {ω : FailPattern k | k ≤ 2 * (univ.filter fun i => ω i = true).card}
      = voteErr k (fun i => (q i : ℝ)) :=
by
  classical
  have hset : {ω : FailPattern k | k ≤ 2 * (univ.filter fun i => ω i = true).card}
      = ↑(univ.filter fun ω : FailPattern k =>
          k ≤ 2 * (univ.filter fun i => ω i = true).card) := by
    ext ω
    simp
  rw [hset, ← sum_measureReal_singleton]
  unfold voteErr
  refine sum_nbij' (fun ω : FailPattern k => univ.filter fun i => ω i = true)
    (fun S : Finset (Fin k) => fun i => decide (i ∈ S)) ?_ ?_ ?_ ?_ ?_
  · intro ω hω
    simp only [mem_filter, mem_univ, true_and, mem_powerset] at hω ⊢
    exact ⟨subset_univ _, hω⟩
  · intro S hS
    simp only [mem_filter, mem_univ, true_and, mem_powerset] at hS ⊢
    rw [lem_filter_decide]
    exact hS.2
  · intro ω _
    funext i
    simp
  · intro S _
    exact lem_filter_decide k S
  · intro ω _
    exact lem_pattern_measure k q ω

/--
**Condorcet bound with independence derived** (§F.3, open item (ii), closed). Exactly
`thm_condorcet_majority_bound`, but as a statement about the measure of an event under a genuine
product of Bernoulli measures whose coordinates are provably independent
(`lem_coord_iIndepFun`), rather than about a sum of posited product weights.

The bound is *not* re-proved: `lem_failure_event_eq_voteErr` shows the two quantities are equal, so
the combinatorial theorem transports verbatim. That is the point --- the earlier proof was already
about this measure; now that is a theorem.

**What this does not claim.** The converse framing is unchanged: this is the *unweighted* majority
vote under a product model, a sibling of Prop. 18's deterministic overlap condition rather than a
replacement, and it says nothing about AdaBoost's reweighted `err_t`. Choosing to model worker
failures by a product measure at all remains a modelling decision --- that is the Condorcet jury
setup, and no proof discharges it.
-/
theorem thm_condorcet_majority_bound_prob (m : ℕ) (γ : ℝ) (q : Fin (2 * m) → I)
    (hγ_pos : 0 < γ) (hγ_lt : γ < 1 / 2) (hq_edge : ∀ i, (q i : ℝ) ≤ 1 / 2 - γ) :
    (failMeasure (2 * m) q).real
        {ω : FailPattern (2 * m) | 2 * m ≤ 2 * (univ.filter fun i => ω i = true).card}
      ≤ exp (-2 * γ ^ 2 * (2 * m)) :=
by
  rw [lem_failure_event_eq_voteErr]
  exact thm_condorcet_majority_bound m γ (fun i => (q i : ℝ)) hγ_pos hγ_lt
    (fun i => (q i).2.1) hq_edge

/-- Non-vacuity, mirroring `exists_condorcet_instance`: the hypotheses are satisfiable in the
`unitInterval`-valued setting too, at the extreme where every worker sits exactly on the
weak-learning threshold. -/
lemma exists_condorcet_prob_instance (m : ℕ) (γ : ℝ) (hγ_pos : 0 < γ) (hγ_lt : γ < 1 / 2) :
    ∃ q : Fin (2 * m) → I, (∀ i, (q i : ℝ) = 1 / 2 - γ) ∧ (∀ i, (q i : ℝ) ≤ 1 / 2 - γ) :=
  ⟨fun _ => ⟨1 / 2 - γ, Set.mem_Icc.mpr ⟨by linarith, by linarith⟩⟩,
    fun _ => rfl, fun _ => le_refl _⟩

end
