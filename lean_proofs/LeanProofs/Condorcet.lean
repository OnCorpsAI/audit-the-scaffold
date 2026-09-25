import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Diversity

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/-!
### The probabilistic route to diversity, combinatorially

§F.3 asked for the Condorcet/Hoeffding statement: independent worker errors give a majority vote
whose error decays exponentially in the number of workers. `Diversity.lean` deliberately does not
attempt it, being deterministic and combinatorial.

This file supplies it *without leaving that setting*. Rather than build a probability space, the
failure-pattern weight `wt` is written down as an explicit product, and the majority-vote error
`voteErr` as an explicit `Finset` sum over the patterns in which at least half the workers fail.
`Finset.prod_add` then does double duty: it proves the weights sum to `1`, and it performs the
moment-generating-function factorization that yields the exponential bound.

**Independence is an assumption baked into the definition of `wt` as a product, not a theorem proved
in this file.** It is, however, a theorem: `CondorcetProb.lean` puts the workers on a `Measure.pi` of
Bernoullis, *derives* the independence of the failure indicators via `iIndepFun_pi`
(`lem_coord_iIndepFun`), and identifies the measure of the majority-failure event with `voteErr`
(`lem_failure_event_eq_voteErr`), so `thm_condorcet_majority_bound` below transports to that setting
verbatim (`thm_condorcet_majority_bound_prob`).

That identification, rather than a fresh derivation from Mathlib's sub-Gaussian Hoeffding lemmas, is
what earns the stronger reading of the word "independent" --- and it costs almost nothing, because the
sample space is a finite product of finite spaces. What no proof discharges is the modelling step a
Condorcet jury argument makes in the first place: studying a product measure at all.
-/

/--
The weight of one failure pattern `S`: the workers in `S` fail (each with probability `q i`) and the
rest succeed. This product form *is* the independence assumption.
-/
def wt (k : ℕ) (q : Fin k → ℝ) (S : Finset (Fin k)) : ℝ :=
  (∏ i ∈ S, q i) * ∏ i ∈ univ \ S, (1 - q i)

/--
The majority vote's error: the total weight of the patterns in which at least half the workers
fail. The `k ≤ 2 * S.card` convention counts a tie as an error, matching `zero_one_loss` and
`thm_overlap_boundary_sharp` (`Diversity.lean`).
-/
def voteErr (k : ℕ) (q : Fin k → ℝ) : ℝ :=
  ∑ S ∈ filter (fun S => k ≤ 2 * S.card) univ.powerset, wt k q S

/-- Pattern weights are nonnegative when the `q i` are probabilities. -/
lemma lem_wt_nonneg (k : ℕ) (q : Fin k → ℝ) (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∀ i, q i ≤ 1)
    (S : Finset (Fin k)) :
    0 ≤ wt k q S :=
by
  unfold wt
  refine mul_nonneg (prod_nonneg fun i _ => hq0 i) (prod_nonneg fun i _ => ?_)
  linarith [hq1 i]

/-- The pattern weights are a probability distribution: they sum to `1`. Pure `Finset.prod_add`. -/
lemma lem_wt_sum_one (k : ℕ) (q : Fin k → ℝ) :
    ∑ S ∈ univ.powerset, wt k q S = 1 :=
by
  unfold wt
  rw [← prod_add]
  simp

/--
The tilted sum, again by `Finset.prod_add` --- this is the moment-generating function of the number
of failures, factorized across workers. It is the whole analytic content of the bound below.
-/
lemma lem_wt_tilt_sum (k : ℕ) (q : Fin k → ℝ) (r : ℝ) :
    ∑ S ∈ univ.powerset, wt k q S * r ^ S.card = ∏ i, (q i * r + (1 - q i)) :=
by
  have hterm : ∀ S ∈ univ.powerset, wt k q S * r ^ S.card
      = (∏ i ∈ S, (q i * r)) * ∏ i ∈ univ \ S, (1 - q i) := by
    intro S _
    have hsplit : ∏ i ∈ S, (q i * r) = (∏ i ∈ S, q i) * r ^ S.card := by
      rw [prod_mul_distrib, prod_const]
    unfold wt
    rw [hsplit]
    ring
  rw [sum_congr rfl hterm, ← prod_add]

/--
**Condorcet bound for independent errors** (§F.3, open item (ii)). If `2m` workers fail
independently, each with probability at most `1/2 - γ`, then the majority vote's error is at most
`exp(-2γ²·2m)` --- exponential decay in the number of workers.

The constant is *exactly* the one in `exists_ensemble_size_beating_weak_threshold`
(`SoftAggregation.lean`), so this composes with Prop. 16 with no further analysis: the same `k₀`
works.

The even size `2m` avoids a half-integer exponent, the same device
`thm_overlap_boundary_sharp` uses; nothing about the argument needs it.

**What this does not claim.** Independence is encoded in `wt`'s product form rather than derived
*here* --- `thm_condorcet_majority_bound_prob` (`CondorcetProb.lean`) derives it and shows this bound
is the same statement about a genuine product measure (see the module docstring). And this is a
statement about the *unweighted* majority vote under a product model --- it is a sibling of Prop. 18's deterministic overlap condition, not a replacement
for it, and it says nothing about AdaBoost's reweighted `err_t`.
-/
theorem thm_condorcet_majority_bound (m : ℕ) (γ : ℝ) (q : Fin (2 * m) → ℝ)
    (hγ_pos : 0 < γ) (hγ_lt : γ < 1 / 2)
    (hq0 : ∀ i, 0 ≤ q i) (hq_edge : ∀ i, q i ≤ 1 / 2 - γ) :
    voteErr (2 * m) q ≤ exp (-2 * γ ^ 2 * (2 * m)) :=
by
  set γ' : ℝ := 1 / 2 - γ with hγ'_def
  have hγ'_pos : 0 < γ' := by rw [hγ'_def]; linarith
  have hone_sub_pos : 0 < 1 - γ' := by rw [hγ'_def]; linarith
  set r : ℝ := (1 - γ') / γ' with hr_def
  have hr_pos : 0 < r := div_pos hone_sub_pos hγ'_pos
  have hr_ge_one : 1 ≤ r := by
    rw [hr_def, le_div_iff₀ hγ'_pos]
    rw [hγ'_def]
    linarith
  have hq1 : ∀ i, q i ≤ 1 := fun i => by have := hq_edge i; rw [hγ'_def] at this; linarith
  have hrm_pos : 0 < r ^ m := pow_pos hr_pos m
  -- Step 1: relax the majority constraint into a tilt, then sum over all patterns.
  have hstep1 : voteErr (2 * m) q
      ≤ (∑ S ∈ univ.powerset, wt (2 * m) q S * r ^ S.card) / r ^ m := by
    rw [sum_div]
    unfold voteErr
    refine le_trans (sum_le_sum ?_) (sum_le_sum_of_subset_of_nonneg (filter_subset _ _) ?_)
    · intro S hS
      have hcard : m ≤ S.card := by
        have := (mem_filter.mp hS).2
        omega
      have hpow : r ^ m ≤ r ^ S.card := pow_le_pow_right₀ hr_ge_one hcard
      rw [le_div_iff₀ hrm_pos]
      exact mul_le_mul_of_nonneg_left hpow (lem_wt_nonneg _ q hq0 hq1 S)
    · intro S _ _
      exact div_nonneg (mul_nonneg (lem_wt_nonneg _ q hq0 hq1 S) (le_of_lt (pow_pos hr_pos _)))
        (le_of_lt hrm_pos)
  -- Step 2: the tilted sum factorizes and each factor is maximized at the edge.
  have hfactor : ∀ i, q i * r + (1 - q i) ≤ 2 * (1 - γ') := by
    intro i
    have hqi : q i ≤ γ' := by rw [hγ'_def]; exact hq_edge i
    have hmono : q i * r + (1 - q i) ≤ γ' * r + (1 - γ') := by nlinarith
    have hval : γ' * r + (1 - γ') = 2 * (1 - γ') := by
      rw [hr_def]
      field_simp
      ring
    linarith
  have hstep2 : (∑ S ∈ univ.powerset, wt (2 * m) q S * r ^ S.card) ≤ (2 * (1 - γ')) ^ (2 * m) := by
    rw [lem_wt_tilt_sum]
    calc ∏ i, (q i * r + (1 - q i)) ≤ ∏ _i : Fin (2 * m), (2 * (1 - γ')) := by
          refine prod_le_prod (fun i _ => ?_) (fun i _ => hfactor i)
          have := hq0 i
          have := hq1 i
          nlinarith
      _ = (2 * (1 - γ')) ^ (2 * m) := by rw [prod_const, card_univ, Fintype.card_fin]
  -- Step 3: the two pieces combine to (1 - 4γ²)^m.
  have hcombine : (2 * (1 - γ')) ^ (2 * m) / r ^ m = (1 - 4 * γ ^ 2) ^ m := by
    rw [pow_mul, ← div_pow]
    congr 1
    rw [hr_def, hγ'_def]
    field_simp
    ring
  -- Step 4: (1 - 4γ²)^m ≤ exp(-4γ²)^m = exp(-2γ²·2m).
  have hbase : 1 - 4 * γ ^ 2 ≤ exp (-4 * γ ^ 2) := by
    have h := add_one_le_exp (-4 * γ ^ 2)
    linarith
  have hbase_nonneg : (0 : ℝ) ≤ 1 - 4 * γ ^ 2 := by nlinarith
  have hpow_le : (1 - 4 * γ ^ 2) ^ m ≤ exp (-4 * γ ^ 2) ^ m :=
    pow_le_pow_left₀ hbase_nonneg hbase m
  have hexp_eq : exp (-4 * γ ^ 2) ^ m = exp (-2 * γ ^ 2 * (2 * m)) := by
    rw [← exp_nat_mul]
    congr 1
    ring
  calc voteErr (2 * m) q
      ≤ (∑ S ∈ univ.powerset, wt (2 * m) q S * r ^ S.card) / r ^ m := hstep1
    _ ≤ (2 * (1 - γ')) ^ (2 * m) / r ^ m := by
        exact div_le_div_of_nonneg_right hstep2 hrm_pos.le
    _ = (1 - 4 * γ ^ 2) ^ m := hcombine
    _ ≤ exp (-4 * γ ^ 2) ^ m := hpow_le
    _ = exp (-2 * γ ^ 2 * (2 * m)) := hexp_eq

/-- Non-vacuity: the hypotheses are satisfiable, at the extreme where every worker sits exactly on
the weak-learning threshold. -/
lemma exists_condorcet_instance (m : ℕ) (γ : ℝ) (hγ_pos : 0 < γ) (hγ_lt : γ < 1 / 2) :
    ∃ q : Fin (2 * m) → ℝ, (∀ i, 0 ≤ q i) ∧ (∀ i, q i ≤ 1 / 2 - γ) ∧ (∀ i, q i < 1 / 2) :=
  ⟨fun _ => 1 / 2 - γ, fun _ => by linarith, fun _ => le_refl _, fun _ => by linarith⟩

end
