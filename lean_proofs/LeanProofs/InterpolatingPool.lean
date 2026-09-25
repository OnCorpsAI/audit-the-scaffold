import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic
import LeanProofs.Setup
import LeanProofs.ZDecomposition
import LeanProofs.PoolReuse

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/-!
### The interpolating pool: the positive half of the reuse dichotomy

`PoolReuse.lean`'s `thm_pool_reuse_horizon` caps how long a reused pool can sustain the edge
hypothesis, but only under `hε₀_pos`: some positive floor `ε₀` below which no element of the pool's
nonnegative cone can drive the training loss. Its docstring records that the complementary case is
untouched --- a pool whose cone *interpolates* the training data is unconstrained by the horizon
argument --- and attributes the positive direction to AdaBoost's minimax duality.

This file supplies that positive direction, and it does **not** need duality. LP duality is required
only for the *hard* direction (weak learnability ⟹ interpolation with margin). The direction wanted
here is the easy one: a nonnegative combination that separates every training point with margin `θ`
forces *some* member of the pool to beat any distribution by `θ / (2 * ∑ j, A j)`, by averaging.
Concretely, a distribution-weighted average of a pointwise lower bound is a lower bound on a
nonnegative combination of per-worker edges, and a nonnegative combination cannot exceed its
coefficient sum times its largest term.

The resulting edge is **uniform in the distribution**, which is exactly what "the pool sustains the
edge" has to mean: it holds against every round's reweighted `D`, whatever the schedule and
confidences did beforehand.
-/

/--
**A zero cone loss is a positive margin floor.** `spanLoss` counts `y i * ⟪A, H⟫ ≤ 0` as an error,
so a cone element with zero loss separates *strictly* at every point; on the finite sample the
minimum of finitely many positive numbers supplies a uniform `θ > 0`.
-/
lemma lem_margin_floor_of_spanLoss_zero (N k : ℕ) [NeZero N] (y : Fin N → ℝ)
    (H : Fin k → Fin N → ℝ) (A : Fin k → ℝ) (h_fit : spanLoss N k y H A = 0) :
    ∃ θ : ℝ, 0 < θ ∧ ∀ i, θ ≤ y i * ∑ j, A j * H j i :=
by
  have hNne : (1 / (N : ℝ)) ≠ 0 := by
    have : (0 : ℝ) < 1 / (N : ℝ) := by
      have hN : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
      positivity
    exact ne_of_gt this
  have hsum : ∑ i : Fin N, (if y i * (∑ j, A j * H j i) ≤ 0 then (1 : ℝ) else 0) = 0 := by
    unfold spanLoss at h_fit
    exact (mul_eq_zero.mp h_fit).resolve_left hNne
  have hpos : ∀ i, 0 < y i * ∑ j, A j * H j i := by
    intro i
    have hterm := (sum_eq_zero_iff_of_nonneg
      (fun i (_ : i ∈ (univ : Finset (Fin N))) => ite_one_zero_nonneg
        (y i * (∑ j, A j * H j i) ≤ 0))).mp hsum i (mem_univ i)
    by_contra hcon
    rw [if_pos (not_lt.mp hcon)] at hterm
    norm_num at hterm
  obtain ⟨i₀, -, hmin⟩ := exists_min_image (univ : Finset (Fin N))
    (fun i => y i * ∑ j, A j * H j i) univ_nonempty
  exact ⟨y i₀ * ∑ j, A j * H j i₀, hpos i₀, fun i => hmin i (mem_univ i)⟩

/--
An interpolating cone element has a positive coefficient sum: if every `A j` vanished the margin
would be `0` at every point, contradicting `θ > 0`. This is what makes the normalization
`θ / (2 * ∑ j, A j)` legitimate.
-/
lemma lem_span_coeff_sum_pos (N k : ℕ) [NeZero N] (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ)
    (A : Fin k → ℝ) (hA : ∀ j, 0 ≤ A j) (θ : ℝ) (hθ : 0 < θ)
    (hmargin : ∀ i, θ ≤ y i * ∑ j, A j * H j i) :
    0 < ∑ j, A j :=
by
  rcases (sum_nonneg fun j (_ : j ∈ (univ : Finset (Fin k))) => hA j).lt_or_eq with hlt | heq
  · exact hlt
  · exfalso
    have hzero : ∀ j ∈ (univ : Finset (Fin k)), A j = 0 :=
      (sum_eq_zero_iff_of_nonneg fun j (_ : j ∈ (univ : Finset (Fin k))) => hA j).mp heq.symm
    obtain ⟨i, -⟩ := (univ_nonempty : (univ : Finset (Fin N)).Nonempty)
    have hzero_margin : y i * ∑ j, A j * H j i = 0 := by
      have hinner : ∑ j, A j * H j i = 0 := by
        refine sum_eq_zero fun j hj => ?_
        rw [hzero j hj, zero_mul]
      rw [hinner, mul_zero]
    linarith [hmargin i]

/--
**Weighted margin versus weighted error.** For a binary worker, the distribution-weighted margin
`∑ Dist i * (y i * H j i)` is `1 - 2ε` where `ε` is the worker's weighted error. Same split as
`z_decomposition` (`ZDecomposition.lean`), but indexed by a pool member and an arbitrary
distribution rather than by a round.
-/
lemma lem_weighted_margin_eq (N k : ℕ) (y : Fin N → ℝ) (H : Fin k → Fin N → ℝ) (j : Fin k)
    (Dist : Fin N → ℝ) (hD1 : ∑ i, Dist i = 1)
    (hb : ∀ i : Fin N, y i * H j i = 1 ∨ y i * H j i = -1) :
    ∑ i, Dist i * (y i * H j i)
      = 1 - 2 * ∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i :=
by
  classical
  have hsplit := sum_filter_add_sum_filter_not (univ : Finset (Fin N))
    (fun i => y i * H j i = -1) (fun i => Dist i * (y i * H j i))
  have hDsplit := sum_filter_add_sum_filter_not (univ : Finset (Fin N))
    (fun i => y i * H j i = -1) (fun i => Dist i)
  have hfail : ∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i * (y i * H j i)
      = -∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i := by
    calc ∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i * (y i * H j i)
        = ∑ i ∈ filter (fun i => y i * H j i = -1) univ, (-Dist i) :=
          sum_congr rfl fun i hi => by
            simp only [mem_filter] at hi
            rw [hi.2]; ring
      _ = -∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i := by simp
  have hpass : ∑ i ∈ filter (fun i => ¬ (y i * H j i = -1)) univ, Dist i * (y i * H j i)
      = ∑ i ∈ filter (fun i => ¬ (y i * H j i = -1)) univ, Dist i :=
    sum_congr rfl fun i hi => by
      simp only [mem_filter] at hi
      rcases hb i with h1 | h1
      · rw [h1]; ring
      · exact absurd h1 hi.2
  rw [hD1] at hDsplit
  rw [← hsplit, hfail, hpass]
  linarith

/--
**An interpolating pool sustains the edge** (§F.3, open item (i), the positive half). If some
nonnegative combination of the pool `H` drives the zero-one training loss to `0` --- i.e. the pool's
cone *interpolates* the sample --- then there is a single `γ > 0` such that against **every**
distribution on the sample some member of the pool has weighted error at most `1/2 - γ`.

The `γ` is `θ / (2 * ∑ j, A j)`, where `θ` is the interpolating combination's margin floor. Note the
order of quantifiers: `γ` is fixed *before* the distribution is chosen, so no schedule, confidence
sequence, or history can erode it. That uniformity is what `thm_pool_reuse_horizon`'s `hε₀_pos`
excludes and what makes the two results complementary halves of a dichotomy on `ε₀`.

**No duality is used.** AdaBoost's minimax duality is needed for the converse (weak learnability ⟹
interpolation with margin), which is *not* formalized here. The direction proved is the elementary
one: a distribution-weighted average of a pointwise margin bound lower-bounds a nonnegative
combination of per-worker margins, and such a combination cannot exceed its coefficient sum times its
largest term.

**What this does not claim.** It is a *per-round* statement: at every round some member has the
edge. It does not construct a schedule `σ` and confidences `α` realizing a full run, which would
require defining the two by simultaneous recursion (`α t` is pinned by the optimal-confidence rule to
a quantity depending on `D … α t`, which depends on `σ` and `α` below `t`). Nor does it supply
`0 < err_t`: a member with *zero* weighted error makes the optimal confidence infinite, and an
interpolating pool is exactly the regime where that can happen --- see
`cor_interpolating_pool_edge_every_round`.
-/
theorem thm_interpolating_pool_sustains_edge (N k : ℕ) [NeZero N] [NeZero k] (y : Fin N → ℝ)
    (H : Fin k → Fin N → ℝ) (A : Fin k → ℝ) (hA : ∀ j, 0 ≤ A j)
    (h_fit : spanLoss N k y H A = 0)
    (h_binary : ∀ j : Fin k, ∀ i : Fin N, y i * H j i = 1 ∨ y i * H j i = -1) :
    ∃ γ : ℝ, 0 < γ ∧ ∀ Dist : Fin N → ℝ, (∀ i, 0 ≤ Dist i) → (∑ i, Dist i = 1) →
      ∃ j : Fin k, ∑ i ∈ filter (fun i => y i * H j i = -1) univ, Dist i ≤ 1 / 2 - γ :=
by
  classical
  obtain ⟨θ, hθ, hmargin⟩ := lem_margin_floor_of_spanLoss_zero N k y H A h_fit
  have hS : 0 < ∑ j, A j := lem_span_coeff_sum_pos N k y H A hA θ hθ hmargin
  refine ⟨θ / (2 * ∑ j, A j), by positivity, ?_⟩
  intro Dist hD0 hD1
  have hstep1 : θ ≤ ∑ i, Dist i * (y i * ∑ j, A j * H j i) := by
    have hbase : θ = ∑ i, Dist i * θ := by
      rw [← sum_mul, hD1, one_mul]
    rw [hbase]
    exact sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hmargin i) (hD0 i)
  have hswap : ∑ i, Dist i * (y i * ∑ j, A j * H j i)
      = ∑ j, A j * ∑ i, Dist i * (y i * H j i) := by
    have hL : ∑ i : Fin N, Dist i * (y i * ∑ j, A j * H j i)
        = ∑ i : Fin N, ∑ j : Fin k, A j * (Dist i * (y i * H j i)) :=
      sum_congr rfl fun i _ => by
        rw [mul_sum, mul_sum]
        exact sum_congr rfl fun j _ => by ring
    have hR : ∑ j : Fin k, A j * ∑ i : Fin N, Dist i * (y i * H j i)
        = ∑ j : Fin k, ∑ i : Fin N, A j * (Dist i * (y i * H j i)) :=
      sum_congr rfl fun j _ => mul_sum _ _ _
    rw [hL, hR, sum_comm]
  obtain ⟨j₀, -, hmax⟩ := exists_max_image (univ : Finset (Fin k))
    (fun j => ∑ i, Dist i * (y i * H j i)) univ_nonempty
  have hbound : θ ≤ (∑ j, A j) * ∑ i, Dist i * (y i * H j₀ i) := by
    calc θ ≤ ∑ j, A j * ∑ i, Dist i * (y i * H j i) := by rw [← hswap]; exact hstep1
      _ ≤ ∑ j : Fin k, A j * ∑ i, Dist i * (y i * H j₀ i) :=
          sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hmax j (mem_univ j)) (hA j)
      _ = (∑ j, A j) * ∑ i, Dist i * (y i * H j₀ i) := by rw [sum_mul]
  rw [lem_weighted_margin_eq N k y H j₀ Dist hD1 (h_binary j₀)] at hbound
  refine ⟨j₀, ?_⟩
  have hkey : θ / (2 * ∑ j, A j)
      ≤ 1 / 2 - ∑ i ∈ filter (fun i => y i * H j₀ i = -1) univ, Dist i := by
    rw [div_le_iff₀ (by positivity : (0 : ℝ) < 2 * ∑ j, A j)]
    nlinarith [hbound]
  linarith [hkey]

/-- `D` is a softmax, hence nonnegative. Not previously needed anywhere, so not previously proved. -/
lemma lem_D_nonneg (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
    (i : Fin N) :
    0 ≤ D N y h α t i :=
by
  unfold D
  have hN : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  exact le_of_lt (div_pos (exp_pos _) (mul_pos hN (exp_loss_pos N y h α t)))

/--
**The edge survives every round, whatever the history.** Specializing
`thm_interpolating_pool_sustains_edge` to AdaBoost's own reweighted distribution: for an
interpolating pool there is one `γ > 0` such that at *every* round `t`, under *any* prior worker
sequence `h` and confidence sequence `α`, some pool member has weighted error at most `1/2 - γ`.

This is the in-pool counterpart of `exists_edge_worker_vs_any_round` (`PoolReuse.lean`), whose
docstring disclaims pool membership --- here the witness is a member of the given pool.

**Why this is not yet a run.** Turning a per-round edge into `zero_one_loss ≤ exp(-2γ²T)` via
`cor2_exponential_decay` needs an actual schedule `σ` and confidences `α`, which must be built by
joint recursion on `t`: `α t` is pinned by the optimal-confidence rule to a quantity reading
`D … α t`, which reads `σ` and `α` below `t`. That is mechanical but substantial, and it needs one
extra hypothesis this theorem does not supply: `cor2_exponential_decay` assumes `0 < err_t`, whereas
an interpolating pool is precisely the regime in which a member may be *perfect* on the current `D`
(`err_t = 0`). So the multi-round weighted-vote statement flagged in `WidthRefinement.lean` stays
open; what closes here is the `ε₀ = 0` case of `thm_pool_reuse_horizon`.
-/
theorem cor_interpolating_pool_edge_every_round (N k : ℕ) [NeZero N] [NeZero k] (y : Fin N → ℝ)
    (H : Fin k → Fin N → ℝ) (A : Fin k → ℝ) (hA : ∀ j, 0 ≤ A j)
    (h_fit : spanLoss N k y H A = 0)
    (h_binary : ∀ j : Fin k, ∀ i : Fin N, y i * H j i = 1 ∨ y i * H j i = -1) :
    ∃ γ : ℝ, 0 < γ ∧ ∀ (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ), ∃ j : Fin k,
      ∑ i ∈ filter (fun i => y i * H j i = -1) univ, D N y h α t i ≤ 1 / 2 - γ :=
by
  obtain ⟨γ, hγ, hmain⟩ := thm_interpolating_pool_sustains_edge N k y H A hA h_fit h_binary
  exact ⟨γ, hγ, fun h α t => hmain (D N y h α t) (fun i => lem_D_nonneg N y h α t i)
    (D_sum_eq_one N y h α t)⟩

/--
**The horizon's floor hypothesis and interpolation are mutually exclusive.** The sharpness companion
to `thm_pool_reuse_horizon`: its `h_floor` together with `hε₀_pos` outright contradicts the existence
of an interpolating element of the cone. So the horizon theorem's restriction to `ε₀ > 0` is not an
artifact of the proof --- it is exactly the boundary between the two halves of the dichotomy, the
other half being `thm_interpolating_pool_sustains_edge`.

Stated as a `False` conclusion deliberately. "`hε₀_pos` cannot be dropped" is a claim *about* the
Lean statement rather than a theorem inside it; the formalizable content is the incompatibility.
-/
theorem cor_pool_horizon_floor_excludes_interpolation (N k : ℕ) [NeZero N] (y : Fin N → ℝ)
    (H : Fin k → Fin N → ℝ) (A : Fin k → ℝ) (ε₀ : ℝ) (hA : ∀ j, 0 ≤ A j) (hε₀_pos : 0 < ε₀)
    (h_floor : ∀ A' : Fin k → ℝ, (∀ j, 0 ≤ A' j) → ε₀ ≤ spanLoss N k y H A')
    (h_interp : spanLoss N k y H A = 0) :
    False :=
by
  have hle := h_floor A hA
  rw [h_interp] at hle
  linarith

/--
Non-vacuity: interpolating pools exist, so `thm_interpolating_pool_sustains_edge`'s hypotheses are
jointly satisfiable. One worker that is right on both of two oppositely-labelled points, with
confidence `1`: margin `1` everywhere, coefficient sum `1`, hence `γ = 1/2` --- the largest edge
possible, as it must be for a pool that already classifies the sample perfectly.
-/
lemma exists_interpolating_pool_instance :
    ∃ (y : Fin 2 → ℝ) (H : Fin 1 → Fin 2 → ℝ) (A : Fin 1 → ℝ),
      (∀ j, 0 ≤ A j) ∧ (∀ j i, y i * H j i = 1 ∨ y i * H j i = -1) ∧
      spanLoss 2 1 y H A = 0 :=
by
  refine ⟨fun i => if i = 0 then (1 : ℝ) else -1, fun _ i => if i = 0 then (1 : ℝ) else -1,
    fun _ => 1, fun _ => zero_le_one, ?_, ?_⟩
  · intro j i
    fin_cases i <;> norm_num
  · unfold spanLoss
    norm_num [Fin.sum_univ_two, Fin.sum_univ_one]

end
