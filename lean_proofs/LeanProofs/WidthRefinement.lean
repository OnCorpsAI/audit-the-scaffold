import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import LeanProofs.Theorem5

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

/--
Non-vacuity guard for the width-augmented refinement game (§F.3): an orchestrated refinement
operator satisfying `h_orch_dominates` below always exists. Given `k` worker refinement
operators `R : Fin k → ℕ → X → A → X`, hard (arg-max) selection over the `k` outputs produces an
`R_orch` whose value dominates every worker's at every `(t, x, a)`. This matters because the
main results below take `h_orch_dominates` as a *hypothesis* (matching `Proposition15.lean`'s
`h_ceiling` style); without this lemma those results could be vacuous.

Note the selector is chosen pointwise in `(t, x, a)`, which is exactly what best-of-`k`
orchestration does: the winning worker is allowed to vary with the round, the artifact, and the
realized action. No claim is made that a single worker wins uniformly --- indeed Prop. 15
(`prop15_tight_ceiling_orchestration`) shows the ceiling ties the best worker regardless.
-/
lemma exists_width_orchestrator {X A : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ) (R : Fin k → ℕ → X → A → X) :
    ∃ R_orch : ℕ → X → A → X, ∀ t x a i, V (R i t x a) ≤ V (R_orch t x a) :=
by
  have hsel : ∀ (t : ℕ) (x : X) (a : A), ∃ i₀ : Fin k, ∀ i, V (R i t x a) ≤ V (R i₀ t x a) := by
    intro t x a
    obtain ⟨i₀, -, hi₀⟩ := exists_mem_eq_sup' (univ_nonempty (α := Fin k)) (fun i => V (R i t x a))
    refine ⟨i₀, fun i => ?_⟩
    rw [← hi₀]
    exact le_sup' (fun i => V (R i t x a)) (mem_univ i)
  choose sel hsel using hsel
  exact ⟨fun t x a => R (sel t x a) t x a, fun t x a i => hsel t x a i⟩

/--
The load-bearing step of the width-augmented refinement game (§F.3): per-worker local
improvement lifts to orchestrated local improvement *at the best worker's edge*, not at the
average. From `h_local_improve : ∀ i t x, E_A t x (V ∘ R i t x) ≥ V x + η i t` we get the same
statement for `R_orch` with edge `max_i η i t`.

The proof picks the worker `i₀` attaining `max_i η i t`, then uses `h_orch_dominates` under
`h_E_A_mono`.

**Divergence from Theorem 5.** `thm5_refinement_game` assumes monotonicity of the *outer*
expectation `E_X` only (`h_E_X_mono`). This lemma additionally needs monotonicity of the *inner*
action-expectation `E_A` (`h_E_A_mono`), because the selection argument compares two different
integrands under the same `E_A t x`. This is the one place the width result asks for more than
the depth-only Theorem 5 did; it is satisfied by any genuine expectation or supremum, but it is
an extra hypothesis and is flagged as such rather than hidden.
-/
lemma width_local_improve {X A : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (R : Fin k → ℕ → X → A → X)
    (R_orch : ℕ → X → A → X)
    (E_A : ℕ → X → (A → ℝ) → ℝ)
    (h_E_A_mono : ∀ t x f g, (∀ a, f a ≤ g a) → E_A t x f ≤ E_A t x g)
    (h_orch_dominates : ∀ t x a i, V (R i t x a) ≤ V (R_orch t x a))
    (η : Fin k → ℕ → ℝ)
    (h_local_improve : ∀ i t x, E_A t x (fun a => V (R i t x a)) ≥ V x + η i t) :
    ∀ t x, E_A t x (fun a => V (R_orch t x a)) ≥ V x + univ.sup' univ_nonempty (fun i => η i t) :=
by
  intro t x
  obtain ⟨i₀, -, hi₀⟩ := exists_mem_eq_sup' (univ_nonempty (α := Fin k)) (fun i => η i t)
  have h1 : E_A t x (fun a => V (R i₀ t x a)) ≤ E_A t x (fun a => V (R_orch t x a)) :=
    h_E_A_mono t x _ _ (fun a => h_orch_dominates t x a i₀)
  have h2 := h_local_improve i₀ t x
  rw [hi₀]
  linarith

/--
Width-augmented refinement game (§F.3), closing the "no width-dimension analog of the refinement
game" gap for the **hard-selection** case. `k` frozen workers are consulted *every* round and the
orchestrator keeps the best output (`h_orch_dominates`), for `T` sequential rounds. The abstract
expectation setup (`E_A`, `E_X`, `h_transition`, `h_E_X_mono`, `h_E_X_add`, `h_E_X_const`) is
`thm5_refinement_game`'s verbatim, with `h_transition` stated for `R_orch`, so this result is
literal reuse of the depth-only telescoping argument at the orchestrated per-round edge
`max_i η i t`.

Combining width with depth therefore improves the per-round *rate*: the cumulative bound picks up
`max_i η i t` each round rather than any fixed worker's `η j t`. See
`cor_width_dominates_best_worker` for the comparison against individual workers and
`cor_width_saturation` for the budget that this rate improvement does *not* escape.

**What is not formalized.** Only hard (arg-max) selection. The weighted-vote case (Prop. 16,
`cor_soft_aggregation_beats_weak_baseline`) is *not* covered here: its diversity hypothesis
`h_edge` constrains each worker's error against the *reweighted* distribution `D_t`, and reusing
the same `k` workers on every round was once thought to threaten `h_edge` through round-to-round
self-correlation. That reading turned out to be the wrong diagnosis: what actually binds a reused
pool is its *capacity*, not correlation. `thm_pool_reuse_horizon` and `cor_no_perpetual_pool_edge`
(`PoolReuse.lean`) show the reused pool's margin never leaves the pool's nonnegative cone, so a
pool whose cone cannot fit the training data fails `h_edge` by a computable round --- while a pool
whose cone interpolates it is untouched by that argument. That complementary case is now settled, and
positively: `thm_interpolating_pool_sustains_edge` (`InterpolatingPool.lean`) shows an interpolating
pool satisfies `h_edge` against *every* distribution, at an edge fixed before the distribution is
chosen.

So what remains unformalized in the weighted-vote direction is narrower than "that case": it is the
multi-round *run* --- a schedule and confidence sequence exhibited outright, rather than the per-round
edge that would feed one. Constructing it needs a simultaneous recursion defining the schedule and
the confidences together, and needs to exclude a *perfect* pool member, since `cor2_exponential_decay`
assumes `0 < err_t` (§F.3).
-/
theorem thm_width_refinement_game {X A : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (R : Fin k → ℕ → X → A → X)
    (R_orch : ℕ → X → A → X)
    (E_A : ℕ → X → (A → ℝ) → ℝ)
    (E_X : ℕ → (X → ℝ) → ℝ)
    (h_transition : ∀ t (f : X → ℝ), E_X (t + 1) f = E_X t (fun x => E_A t x (fun a => f (R_orch t x a))))
    (h_E_X_mono : ∀ t f g, (∀ x, f x ≤ g x) → E_X t f ≤ E_X t g)
    (h_E_X_add : ∀ t f g, E_X t (fun x => f x + g x) = E_X t f + E_X t g)
    (h_E_X_const : ∀ t c, E_X t (fun _ => c) = c)
    (h_E_A_mono : ∀ t x f g, (∀ a, f a ≤ g a) → E_A t x f ≤ E_A t x g)
    (h_orch_dominates : ∀ t x a i, V (R i t x a) ≤ V (R_orch t x a))
    (η : Fin k → ℕ → ℝ)
    (h_local_improve : ∀ i t x, E_A t x (fun a => V (R i t x a)) ≥ V x + η i t) :
    ∀ T, E_X T V ≥ E_X 0 V + ∑ t ∈ range T, univ.sup' univ_nonempty (fun i => η i t) :=
  thm5_refinement_game V R_orch E_A E_X h_transition h_E_X_mono h_E_X_add h_E_X_const
    (fun t => univ.sup' univ_nonempty (fun i => η i t))
    (width_local_improve V R R_orch E_A h_E_A_mono h_orch_dominates η h_local_improve)

/--
The width payoff (§F.3): the orchestrated run's cumulative guarantee dominates *every*
individual worker's depth-only guarantee, not merely the average. For each fixed worker `j`,
running `thm5_refinement_game` on `j` alone yields `E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η j t`;
the orchestrated bound is at least that, for all `j` simultaneously.

This is the width-dimension counterpart of `prop15_orchestration_dominates`, which makes the
same "never worse than the best single worker" point about *ceilings*; here it is about the
*cumulative multi-round improvement*.
-/
theorem cor_width_dominates_best_worker {X A : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (R : Fin k → ℕ → X → A → X)
    (R_orch : ℕ → X → A → X)
    (E_A : ℕ → X → (A → ℝ) → ℝ)
    (E_X : ℕ → (X → ℝ) → ℝ)
    (h_transition : ∀ t (f : X → ℝ), E_X (t + 1) f = E_X t (fun x => E_A t x (fun a => f (R_orch t x a))))
    (h_E_X_mono : ∀ t f g, (∀ x, f x ≤ g x) → E_X t f ≤ E_X t g)
    (h_E_X_add : ∀ t f g, E_X t (fun x => f x + g x) = E_X t f + E_X t g)
    (h_E_X_const : ∀ t c, E_X t (fun _ => c) = c)
    (h_E_A_mono : ∀ t x f g, (∀ a, f a ≤ g a) → E_A t x f ≤ E_A t x g)
    (h_orch_dominates : ∀ t x a i, V (R i t x a) ≤ V (R_orch t x a))
    (η : Fin k → ℕ → ℝ)
    (h_local_improve : ∀ i t x, E_A t x (fun a => V (R i t x a)) ≥ V x + η i t)
    (j : Fin k) :
    ∀ T, E_X T V ≥ E_X 0 V + ∑ t ∈ range T, η j t :=
by
  intro T
  have h1 := thm_width_refinement_game V R R_orch E_A E_X h_transition h_E_X_mono h_E_X_add
    h_E_X_const h_E_A_mono h_orch_dominates η h_local_improve T
  have h2 : ∑ t ∈ range T, η j t ≤ ∑ t ∈ range T, univ.sup' univ_nonempty (fun i => η i t) :=
    sum_le_sum (fun t _ => le_sup' (fun i => η i t) (mem_univ j))
  linarith

/--
The punchline (§F.3): **width does not raise the improvement budget.** With quality bounded by 1,
`thm5_saturation` applied to the orchestrated edge gives
`∑ t ∈ range T, max_i η i t ≤ 1 - E_X 0 V` --- the same total-improvement budget a single agent
faces, now consumed by the *max* over workers rather than by one worker's edge.

Reading: consulting `k` workers every round improves the per-round *rate*
(`cor_width_dominates_best_worker`) but leaves the total *budget* unchanged. Depth × width
composition does not escape the frozen-weight ceiling; it only spends the same budget faster.
This is consistent with `cor_frozen_weight_ceiling` and with Prop. 15's reading that
orchestration is an instance of assumption (M1), not a new mechanism (§F.3, §F.5).
-/
theorem cor_width_saturation {X A : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (R : Fin k → ℕ → X → A → X)
    (R_orch : ℕ → X → A → X)
    (E_A : ℕ → X → (A → ℝ) → ℝ)
    (E_X : ℕ → (X → ℝ) → ℝ)
    (h_transition : ∀ t (f : X → ℝ), E_X (t + 1) f = E_X t (fun x => E_A t x (fun a => f (R_orch t x a))))
    (h_E_X_mono : ∀ t f g, (∀ x, f x ≤ g x) → E_X t f ≤ E_X t g)
    (h_E_X_add : ∀ t f g, E_X t (fun x => f x + g x) = E_X t f + E_X t g)
    (h_E_X_const : ∀ t c, E_X t (fun _ => c) = c)
    (h_E_A_mono : ∀ t x f g, (∀ a, f a ≤ g a) → E_A t x f ≤ E_A t x g)
    (h_orch_dominates : ∀ t x a i, V (R i t x a) ≤ V (R_orch t x a))
    (η : Fin k → ℕ → ℝ)
    (h_local_improve : ∀ i t x, E_A t x (fun a => V (R i t x a)) ≥ V x + η i t)
    (h_bound : ∀ x, V x ≤ 1) :
    ∀ T, ∑ t ∈ range T, univ.sup' univ_nonempty (fun i => η i t) ≤ 1 - E_X 0 V :=
  thm5_saturation V R_orch E_A E_X h_transition h_E_X_mono h_E_X_add h_E_X_const
    (fun t => univ.sup' univ_nonempty (fun i => η i t))
    (width_local_improve V R R_orch E_A h_E_A_mono h_orch_dominates η h_local_improve)
    h_bound
