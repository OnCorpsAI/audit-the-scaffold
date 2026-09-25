import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Setup
import LeanProofs.Theorem1
import LeanProofs.ZDecomposition
import LeanProofs.Corollary2
import LeanProofs.SelfReuse

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/--
**No per-round obstruction.** Prop. 19 (`thm_self_reuse_zero_edge`) shows the *reused* worker's
edge against the distribution it induces is exactly zero. That is a fact about the reused worker,
and it does not propagate: against *any* round's distribution --- including the one immediately
after a reuse --- a binary worker with error at most `1/N` exists, hence edge at least
`1/2 - 1/N`. The witness is wrong at a single minimum-weight point, and `D`'s normalization
(`D_sum_eq_one`) forces the minimum to sit at or below the average `1/N`.

**What this does not claim.** The constructed worker need not belong to any given pool. So this
removes a *per-round* obstruction to cross-worker reuse; it does **not** show that a pool can
sustain the edge hypothesis. What limits a reused pool is capacity, not the per-round edge ---
see `thm_pool_reuse_horizon` below.
-/
lemma exists_edge_worker_vs_any_round (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ)
    (α : ℕ → ℝ) (t : ℕ) (hy : ∀ i, y i = 1 ∨ y i = -1) :
    ∃ H' : Fin N → ℝ, (∀ i, y i * H' i = 1 ∨ y i * H' i = -1) ∧
      ∑ i ∈ filter (fun i => y i * H' i = -1) univ, D N y h α t i ≤ 1 / N :=
by
  obtain ⟨i₀, -, hmin⟩ := exists_min_image (univ : Finset (Fin N)) (D N y h α t) univ_nonempty
  have hsq : ∀ i, y i * y i = 1 := by
    intro i
    rcases hy i with h1 | h1 <;> rw [h1] <;> norm_num
  have hneg : ∀ i, y i * -y i = -1 := by
    intro i
    calc y i * -y i = -(y i * y i) := by ring
      _ = -1 := by rw [hsq i]
  refine ⟨fun i => if i = i₀ then -y i else y i, ?_, ?_⟩
  · intro i
    change y i * (if i = i₀ then -y i else y i) = 1 ∨ y i * (if i = i₀ then -y i else y i) = -1
    by_cases hi : i = i₀
    · exact Or.inr (by rw [if_pos hi]; exact hneg i)
    · exact Or.inl (by rw [if_neg hi]; exact hsq i)
  · change ∑ i ∈ filter (fun i => y i * (if i = i₀ then -y i else y i) = -1) univ,
      D N y h α t i ≤ 1 / N
    have hfilt : filter (fun i => y i * (if i = i₀ then -y i else y i) = -1) univ = {i₀} := by
      ext i
      simp only [mem_filter, mem_univ, true_and, mem_singleton]
      constructor
      · intro hcon
        by_contra hne
        rw [if_neg hne, hsq i] at hcon
        norm_num at hcon
      · intro hi
        rw [if_pos hi]
        exact hneg i
    rw [hfilt, sum_singleton]
    have hN : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
    have hsum : (N : ℝ) * D N y h α t i₀ ≤ ∑ j : Fin N, D N y h α t j := by
      have hnsmul := card_nsmul_le_sum (univ : Finset (Fin N)) (D N y h α t) (D N y h α t i₀)
        (fun j _ => hmin j (mem_univ j))
      rwa [card_univ, Fintype.card_fin, nsmul_eq_mul] at hnsmul
    rw [D_sum_eq_one] at hsum
    rw [le_div_iff₀ hN]
    linarith

/--
The total confidence a reused pool assigns to member `j` over the first `T` rounds: the sum of
`α t` over exactly the rounds the schedule `σ` sent to `j`.
-/
def poolCoeff (k : ℕ) (σ : ℕ → Fin k) (α : ℕ → ℝ) (T : ℕ) (j : Fin k) : ℝ :=
  ∑ t ∈ filter (fun t => σ t = j) (range T), α t

/--
The zero-one training loss of a single element of the pool's span, `∑ j A j * H j`. Same shape as
`zero_one_loss` (`Theorem1.lean`), but indexed by a coefficient vector rather than by a round
count --- which is what makes a `T`-independent floor expressible.
-/
def spanLoss (N k : ℕ) (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ) (A : Fin k → ℝ) : ℝ :=
  (1 / (N : ℝ)) * ∑ i, if y i * (∑ j, A j * H j i) ≤ 0 then 1 else 0

/--
**Span collapse.** Running a pool `H` of `k` workers under a schedule `σ` for `T` rounds produces
a margin lying in the span of the pool, with member `j`'s coefficient the total confidence
`poolCoeff` assigned to it. Grouping the rounds into fibers of `σ` is all it takes.

This is the structural fact behind everything below: the reachable class has dimension at most `k`
no matter how many rounds run.
-/
lemma lem_pool_span_collapse (N k : ℕ) (H : Fin k → Fin N → ℝ) (σ : ℕ → Fin k) (α : ℕ → ℝ)
    (T : ℕ) (i : Fin N) :
    f N (fun t => H (σ t)) α T i = ∑ j : Fin k, poolCoeff k σ α T j * H j i :=
by
  unfold f poolCoeff
  rw [← sum_fiberwise (range T) σ (fun t => α t * H (σ t) i)]
  refine sum_congr rfl fun j _ => ?_
  rw [sum_mul]
  refine sum_congr rfl fun t ht => ?_
  simp only [mem_filter] at ht
  rw [ht.2]

/-- The scheduled run's zero-one loss *is* the loss of the span element it lands on. -/
lemma zero_one_loss_eq_spanLoss (N k : ℕ) (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ)
    (σ : ℕ → Fin k) (α : ℕ → ℝ) (T : ℕ) :
    zero_one_loss N y (fun t => H (σ t)) α T = spanLoss N k y H (poolCoeff k σ α T) :=
by
  simp only [zero_one_loss, spanLoss, lem_pool_span_collapse]

/--
At the optimal confidence, an edge is the same thing as a nonnegative `α`: this is AdaBoost's
stopping rule (`ε ≥ ½ ⇒ α ≤ 0`) read forwards. It is what confines a reused pool's margin to the
pool's *nonnegative cone* rather than merely its span.
-/
lemma alpha_opt_nonneg (ε γ : ℝ) (hε_pos : 0 < ε) (hγ_pos : 0 < γ) (h_edge : ε ≤ 1 / 2 - γ) :
    0 ≤ (1 / 2) * log ((1 - ε) / ε) :=
by
  have h1 : 1 ≤ (1 - ε) / ε := by
    rw [le_div_iff₀ hε_pos]
    linarith
  have h2 : 0 ≤ log ((1 - ε) / ε) := Real.log_nonneg h1
  linarith

/-- Nonnegative confidences give nonnegative pool coefficients. -/
lemma poolCoeff_nonneg (k : ℕ) (σ : ℕ → Fin k) (α : ℕ → ℝ) (T : ℕ)
    (hα : ∀ t ∈ range T, 0 ≤ α t) (j : Fin k) :
    0 ≤ poolCoeff k σ α T j :=
by
  unfold poolCoeff
  exact sum_nonneg fun t ht => hα t (mem_of_mem_filter t ht)

/--
**The reused-pool horizon** (§F.3, open item (i), the effective negative half). Run AdaBoost with a
*fixed pool* `H` of `k` workers under an arbitrary schedule `σ`, at the optimal confidences, with
every scheduled worker maintaining edge `γ > 0` against the reweighted distribution current when it
is used. If no element of the pool's nonnegative cone drives the zero-one training loss below
`ε₀ > 0`, then `ε₀ ≤ exp(-2γ²T)`, so the number of rounds is capped:
`T ≤ log(1/ε₀) / (2γ²)`.

The bound is uniform over the schedule `σ` and the confidences `α`, and depends on the pool size
`k` only through `ε₀`. So a pool whose cone cannot fit the training data provably fails the edge
hypothesis at a finite, computable round --- the cross-worker obstruction is **capacity, not
correlation**. This is the stationarity dichotomy (Prop. 11) and its frozen-weight ceiling
(Cor. 12) instantiated in the weighted-vote setting.

**What this does not claim.**
* The *qualitative* form of this is classical, not new: AdaBoost's minimax duality already gives
  that a pool is `γ`-weak-learnable exactly when its convex hull separates every training point
  with margin `≥ 2γ`. What is contributed here is the *effective* negative half --- machine-checked,
  with an explicit horizon. The duality's converse direction (weak learnability ⟹ interpolation with
  margin) is **not** formalized.
* It says nothing when `ε₀ = 0`. A pool whose cone interpolates the training data is untouched by
  this argument, and such a pool *does* sustain the edge --- which is now machine-checked separately
  as `thm_interpolating_pool_sustains_edge` (`InterpolatingPool.lean`), and by averaging rather than
  by duality. The two results are mutually exclusive rather than overlapping:
  `cor_pool_horizon_floor_excludes_interpolation` derives `False` from `h_floor`, `hε₀_pos`, and an
  interpolating cone element, so `hε₀_pos` marks the boundary between the two cases rather than a
  limitation of the proof.
* It does not identify *which* round fails, only that one does by the stated bound.
* The proof itself is not reuse-specific: any sequence `h` factors on `range T` as
  `fun t => H (σ t)` with `k := T` and `σ := id`. Reuse-specificity lives in the hypothesis --- for
  a *fixed* pool one `ε₀` serves every `T`, which is what `cor_no_perpetual_pool_edge` exploits and
  what fresh workers cannot supply.
* `spanLoss` is training loss. Nothing here concerns generalization.
* Prop. 16 (`cor_soft_aggregation_beats_weak_baseline`) is untouched: its workers are each
  incorporated once, so no pool is reused there.
-/
theorem thm_pool_reuse_horizon (N k : ℕ) [NeZero N] (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ)
    (σ : ℕ → Fin k) (α : ℕ → ℝ) (T : ℕ) (γ ε₀ : ℝ)
    (hγ_pos : 0 < γ) (hε₀_pos : 0 < ε₀)
    (h_floor : ∀ A : Fin k → ℝ, (∀ j, 0 ≤ A j) → ε₀ ≤ spanLoss N k y H A)
    (h_binary : ∀ t ∈ range T, ∀ i : Fin N, y i * H (σ t) i = 1 ∨ y i * H (σ t) i = -1)
    (h_eps_pos : ∀ t ∈ range T, 0 < err_t N y (fun t => H (σ t)) α t)
    (h_eps_lt_one : ∀ t ∈ range T, err_t N y (fun t => H (σ t)) α t < 1)
    (h_alpha_opt : ∀ t ∈ range T, α t
      = (1 / 2) * log ((1 - err_t N y (fun t => H (σ t)) α t) / err_t N y (fun t => H (σ t)) α t))
    (h_edge : ∀ t ∈ range T, err_t N y (fun t => H (σ t)) α t ≤ 1 / 2 - γ) :
    ε₀ ≤ exp (-2 * γ ^ 2 * T) ∧ (T : ℝ) ≤ Real.log (1 / ε₀) / (2 * γ ^ 2) :=
by
  have hα_nonneg : ∀ t ∈ range T, 0 ≤ α t := by
    intro t ht
    rw [h_alpha_opt t ht]
    exact alpha_opt_nonneg _ γ (h_eps_pos t ht) hγ_pos (h_edge t ht)
  have hfloor : ε₀ ≤ zero_one_loss N y (fun t => H (σ t)) α T := by
    rw [zero_one_loss_eq_spanLoss]
    exact h_floor _ (poolCoeff_nonneg k σ α T hα_nonneg)
  have hle : ε₀ ≤ exp (-2 * γ ^ 2 * T) :=
    calc ε₀ ≤ zero_one_loss N y (fun t => H (σ t)) α T := hfloor
      _ ≤ ∏ t ∈ range T, Z N y (fun t => H (σ t)) α t :=
          thm1_zero_one_loss_bound N y (fun t => H (σ t)) α T
      _ = exp_loss N y (fun t => H (σ t)) α T :=
          (thm1_training_error_bound N y (fun t => H (σ t)) α T).symm
      _ ≤ exp (-2 * γ ^ 2 * T) :=
          cor2_exponential_decay N y (fun t => H (σ t)) α T γ h_binary hγ_pos.le
            h_eps_pos h_eps_lt_one h_alpha_opt h_edge
  refine ⟨hle, ?_⟩
  have hlog : Real.log ε₀ ≤ -2 * γ ^ 2 * T := by
    have hmono := Real.log_le_log hε₀_pos hle
    rwa [Real.log_exp] at hmono
  have hden : (0 : ℝ) < 2 * γ ^ 2 := by positivity
  have hinv : Real.log (1 / ε₀) = -Real.log ε₀ := by
    rw [one_div, Real.log_inv]
  rw [le_div_iff₀ hden, hinv]
  nlinarith [hlog]

/--
**No pool sustains the edge forever.** Same setup as `thm_pool_reuse_horizon` but with the edge
hypothesis imposed at *every* round: that is outright contradictory whenever the pool's cone has a
positive training-loss floor. The horizon bound holds for each `T`, while a fixed pool supplies one
`ε₀` for all of them, so a large enough `T` breaks it.

This is the reuse-specific statement. It has no fresh-worker analog: with a new worker each round
there is no single `(k, H)` to hold `ε₀` fixed as `T` grows, and the corollary admits no
`k`-grows-with-`T` form.
-/
theorem cor_no_perpetual_pool_edge (N k : ℕ) [NeZero N] (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ)
    (σ : ℕ → Fin k) (α : ℕ → ℝ) (γ ε₀ : ℝ)
    (hγ_pos : 0 < γ) (hε₀_pos : 0 < ε₀)
    (h_floor : ∀ A : Fin k → ℝ, (∀ j, 0 ≤ A j) → ε₀ ≤ spanLoss N k y H A)
    (h_binary : ∀ t, ∀ i : Fin N, y i * H (σ t) i = 1 ∨ y i * H (σ t) i = -1)
    (h_eps_pos : ∀ t, 0 < err_t N y (fun t => H (σ t)) α t)
    (h_eps_lt_one : ∀ t, err_t N y (fun t => H (σ t)) α t < 1)
    (h_alpha_opt : ∀ t, α t
      = (1 / 2) * log ((1 - err_t N y (fun t => H (σ t)) α t) / err_t N y (fun t => H (σ t)) α t))
    (h_edge : ∀ t, err_t N y (fun t => H (σ t)) α t ≤ 1 / 2 - γ) :
    False :=
by
  obtain ⟨T, hT⟩ := exists_nat_gt (Real.log (1 / ε₀) / (2 * γ ^ 2))
  have hbound := (thm_pool_reuse_horizon N k y H σ α T γ ε₀ hγ_pos hε₀_pos h_floor
    (fun t _ => h_binary t) (fun t _ => h_eps_pos t) (fun t _ => h_eps_lt_one t)
    (fun t _ => h_alpha_opt t) (fun t _ => h_edge t)).2
  linarith

/-- An indicator term is nonnegative. Small helper for the non-vacuity witnesses below. -/
lemma ite_one_zero_nonneg (P : Prop) [Decidable P] : (0 : ℝ) ≤ if P then 1 else 0 :=
by
  by_cases hP : P
  · rw [if_pos hP]
    norm_num
  · rw [if_neg hP]

/--
Pools with a positive cone floor exist, so `thm_pool_reuse_horizon`'s `h_floor` is satisfiable: a
single constant worker over two oppositely-labelled points has floor `1/2`, since every span
element assigns both points the same margin and one of the two labels must disagree with it.
-/
lemma exists_positive_span_floor :
    ∃ (y : Fin 2 → ℝ) (H : Fin 1 → Fin 2 → ℝ),
      ∀ A : Fin 1 → ℝ, (1 : ℝ) / 2 ≤ spanLoss 2 1 y H A :=
by
  refine ⟨fun i => if i = 0 then (1 : ℝ) else -1, fun _ _ => 1, fun A => ?_⟩
  unfold spanLoss
  rw [Fin.sum_univ_two]
  norm_num
  rcases le_total (A 0) 0 with hA | hA
  · rw [if_pos hA]
    linarith [ite_one_zero_nonneg ((0 : ℝ) ≤ A 0)]
  · rw [if_pos hA]
    linarith [ite_one_zero_nonneg (A 0 ≤ (0 : ℝ))]

/--
Non-vacuity for `thm_pool_reuse_horizon` proper, and the one that matters: `h_floor` and the edge
hypothesis are **jointly** satisfiable, so the theorem is not `False → anything`. One constant
worker over four points, three labelled `+1` and one `-1`: the cone floor is `1/4` (every span
element is constant, so it must disagree with one side), while round zero starts uniform
(`D_zero_eq_uniform`) and gives `err_t = 1/4 ≤ 1/2 - 1/8`, an edge of `γ = 1/8` that genuinely
survives the round. The resulting horizon is `log 4 / (2·(1/8)²) ≈ 44` rounds.
-/
lemma exists_pool_reuse_horizon_instance :
    ∃ (y : Fin 4 → ℝ) (H : Fin 1 → Fin 4 → ℝ) (σ : ℕ → Fin 1) (α : ℕ → ℝ),
      (∀ A : Fin 1 → ℝ, (∀ j, 0 ≤ A j) → (1 : ℝ) / 4 ≤ spanLoss 4 1 y H A) ∧
      (∀ t ∈ range 1, ∀ i : Fin 4, y i * H (σ t) i = 1 ∨ y i * H (σ t) i = -1) ∧
      (∀ t ∈ range 1, 0 < err_t 4 y (fun t => H (σ t)) α t) ∧
      (∀ t ∈ range 1, err_t 4 y (fun t => H (σ t)) α t < 1) ∧
      (∀ t ∈ range 1, α t = (1 / 2)
        * log ((1 - err_t 4 y (fun t => H (σ t)) α t) / err_t 4 y (fun t => H (σ t)) α t)) ∧
      (∀ t ∈ range 1, err_t 4 y (fun t => H (σ t)) α t ≤ 1 / 2 - 1 / 8) :=
by
  have herr : err_t 4 (fun i => if i = 3 then (-1 : ℝ) else 1) (fun _ _ => (1 : ℝ))
      (fun _ => (1 / 2) * log 3) 0 = 1 / 4 := by
    unfold err_t
    have hfilt : filter (fun i => (if i = 3 then (-1 : ℝ) else 1) * (1 : ℝ) = -1)
        (univ : Finset (Fin 4)) = {3} := by
      ext i
      fin_cases i <;> norm_num
    rw [hfilt, sum_singleton, D_zero_eq_uniform]
    norm_num
  refine ⟨fun i => if i = 3 then (-1 : ℝ) else 1, fun _ _ => 1, fun _ => 0,
    fun _ => (1 / 2) * log 3, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro A hA0
    unfold spanLoss
    rw [Fin.sum_univ_four]
    norm_num [Fin.ext_iff]
    rw [if_pos (hA0 0)]
    linarith [ite_one_zero_nonneg (A 0 ≤ (0 : ℝ))]
  · intro t _ i
    change (if i = 3 then (-1 : ℝ) else 1) * 1 = 1 ∨ (if i = 3 then (-1 : ℝ) else 1) * 1 = -1
    by_cases hi : i = 3
    · exact Or.inr (by rw [if_pos hi]; norm_num)
    · exact Or.inl (by rw [if_neg hi]; norm_num)
  · intro t ht
    simp only [mem_range, Nat.lt_one_iff] at ht
    subst ht
    rw [herr]
    norm_num
  · intro t ht
    simp only [mem_range, Nat.lt_one_iff] at ht
    subst ht
    rw [herr]
    norm_num
  · intro t ht
    simp only [mem_range, Nat.lt_one_iff] at ht
    subst ht
    rw [herr]
    norm_num
  · intro t ht
    simp only [mem_range, Nat.lt_one_iff] at ht
    subst ht
    rw [herr]
    norm_num

end
