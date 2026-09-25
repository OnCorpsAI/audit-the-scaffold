import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic
import LeanProofs.Setup
import LeanProofs.Theorem1
import LeanProofs.ZDecomposition

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/--
Per-point failure multiplicity: how many of the first `k` workers misclassify point `i`. Uses the
same misclassification predicate as `err_t` (`ZDecomposition.lean`), namely `y i * h t i = -1`, so
the counting results below line up with the weighted-error machinery of Thm. 1 / Cor. 2.

This is the quantitative diversity parameter asked for in §F.3: `failMult` bounded by `m`
uniformly over points says "no single failure point is shared by more than `m` workers."
-/
def failMult (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (k : ℕ) (i : Fin N) : ℕ :=
  #{t ∈ range k | y i * h t i = -1}

/--
Bridge from *checkable* facts to Prop. 16's diversity hypothesis `h_edge` (§F.3, item (iii)).
Prop. 16 assumes `err_t ≤ 1/2 - γ` against the AdaBoost-*reweighted* distribution `D_t`, which
cannot be checked from primitive facts about the workers. This lemma discharges it for a single
round from two things one can actually inspect:

* `h_ratio`: the reweighted density is at most `ρ / N` pointwise, i.e. reweighting concentrates
  mass by at most a factor `ρ` versus uniform;
* `h_unif`: worker `t`'s error *under the uniform distribution* is at most `(1/2 - γ) / ρ`.

`ρ` is a cap on how far AdaBoost's reweighting can concentrate. Taken here as a hypothesis, it need
not be assumed: `cor_rho_from_confidences` derives `ρ ≤ exp (2 * ∑_{s<t} |α s|)` from the logged
confidences, and `cor_reweighting_edge_from_confidences` gives the resulting statement with no `ρ`
hypothesis at all. What remains true either way is that the uniform-error requirement tightens as
`ρ` grows --- a more concentrated `D_t` demands a proportionally better uniform-distribution
worker --- and that the derived `ρ` grows exponentially in the round count, so the trade is
*checkability*, not *weakness*.
-/
lemma lem_bounded_reweighting_edge (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ)
    (α : ℕ → ℝ) (t : ℕ) (γ ρ : ℝ) (hρ_pos : 0 < ρ)
    (h_ratio : ∀ i, D N y h α t i ≤ ρ / N)
    (h_unif : ((#{i | y i * h t i = -1} : ℕ) : ℝ) / N ≤ (1 / 2 - γ) / ρ) :
    err_t N y h α t ≤ 1 / 2 - γ :=
by
  have h_cancel : ρ * ((1 / 2 - γ) / ρ) = 1 / 2 - γ := by
    field_simp
  have h_scaled : ρ * (((#{i | y i * h t i = -1} : ℕ) : ℝ) / N) ≤ 1 / 2 - γ := by
    have := mul_le_mul_of_nonneg_left h_unif hρ_pos.le
    linarith
  calc err_t N y h α t
      = ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i := rfl
    _ ≤ ∑ _i ∈ filter (fun i => y i * h t i = -1) univ, ρ / N :=
        sum_le_sum (fun i _ => h_ratio i)
    _ = ((#{i | y i * h t i = -1} : ℕ) : ℝ) * (ρ / N) := by
        rw [sum_const, nsmul_eq_mul]
    _ = ρ * (((#{i | y i * h t i = -1} : ℕ) : ℝ) / N) := by ring
    _ ≤ 1 / 2 - γ := h_scaled

/--
The concentration cap `ρ` is not an extra assumption --- it is a consequence of any bound on the
margin spread. Since the repo's `D` is the closed-form softmax
`D t i = exp(-y i * f t i) / ∑ⱼ exp(-y j * f t j)` (`Setup.lean`), bounding every margin by `B`
bounds the numerator above by `exp B` and every denominator term below by `exp (-B)`, giving
`ρ ≤ exp (2 * B)`.

Stated for an arbitrary margin cap `B` so the two ingredients stay separate: this lemma is the
density estimate, `cor_rho_from_confidences` is the choice of `B`.
-/
lemma D_le_of_margin_bound (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ)
    (t : ℕ) (B : ℝ) (h_margin : ∀ i, |y i * f N h α t i| ≤ B) :
    ∀ i, D N y h α t i ≤ exp (2 * B) / N :=
by
  intro i
  have hN : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  have hden_eq : (N : ℝ) * exp_loss N y h α t = ∑ j : Fin N, exp (-y j * f N h α t j) := by
    unfold exp_loss
    field_simp
  have hterm : ∀ j : Fin N, exp (-B) ≤ exp (-y j * f N h α t j) := by
    intro j
    apply exp_le_exp.mpr
    have hab := abs_le.mp (h_margin j)
    have hring : -y j * f N h α t j = -(y j * f N h α t j) := by ring
    rw [hring]
    linarith [hab.2]
  have hlow : (N : ℝ) * exp (-B) ≤ ∑ j : Fin N, exp (-y j * f N h α t j) := by
    calc (N : ℝ) * exp (-B) = ∑ _j : Fin N, exp (-B) := by
          rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ ≤ ∑ j : Fin N, exp (-y j * f N h α t j) := sum_le_sum (fun j _ => hterm j)
  have hnum : exp (-y i * f N h α t i) ≤ exp B := by
    apply exp_le_exp.mpr
    have hab := abs_le.mp (h_margin i)
    have hring : -y i * f N h α t i = -(y i * f N h α t i) := by ring
    rw [hring]
    linarith [hab.1]
  have hSpos : (0 : ℝ) < ∑ j : Fin N, exp (-y j * f N h α t j) :=
    lt_of_lt_of_le (by positivity) hlow
  unfold D
  rw [hden_eq, div_le_div_iff₀ hSpos hN]
  calc exp (-y i * f N h α t i) * (N : ℝ) ≤ exp B * (N : ℝ) :=
        mul_le_mul_of_nonneg_right hnum (le_of_lt hN)
    _ = exp (2 * B) * ((N : ℝ) * exp (-B)) := by
        rw [show (2 : ℝ) * B = B + B by ring, exp_add, exp_neg]
        have hb : exp B ≠ 0 := ne_of_gt (exp_pos B)
        field_simp
    _ ≤ exp (2 * B) * ∑ j : Fin N, exp (-y j * f N h α t j) :=
        mul_le_mul_of_nonneg_left hlow (le_of_lt (exp_pos _))

/--
**`ρ` from the run's own logged confidences** (§F.3): with `±1` outputs, every margin is bounded by
`∑_{s<t} |α s|`, so `D_le_of_margin_bound` gives the explicit cap
`ρ ≤ exp (2 * ∑_{s<t} |α s|)`.

This closes the gap `lem_bounded_reweighting_edge` opened: `ρ` was a *run-level regularity
assumption*, and is now a quantity computable from confidences the algorithm already records.

**The honest remaining cost.** The bound degrades fast: `exp (2 * ∑ |α s|)` grows exponentially in
the number of rounds, so the uniform-error requirement it implies
(`cor_reweighting_edge_from_confidences`) tightens exponentially too, and is close to vacuous after
many rounds. What has changed is that `ρ` is now *checkable* rather than assumed; whether it is
*small* on a given run is a separate question this does not answer.
-/
lemma cor_rho_from_confidences (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ)
    (t : ℕ) (h_binary : ∀ s, ∀ i : Fin N, y i * h s i = 1 ∨ y i * h s i = -1) :
    ∀ i, D N y h α t i ≤ exp (2 * ∑ s ∈ range t, |α s|) / N :=
by
  apply D_le_of_margin_bound
  intro i
  have hexp : y i * f N h α t i = ∑ s ∈ range t, α s * (y i * h s i) := by
    unfold f
    rw [mul_sum]
    exact sum_congr rfl (fun s _ => by ring)
  rw [hexp]
  calc |∑ s ∈ range t, α s * (y i * h s i)| ≤ ∑ s ∈ range t, |α s * (y i * h s i)| :=
        abs_sum_le_sum_abs _ _
    _ = ∑ s ∈ range t, |α s| := by
        refine sum_congr rfl (fun s _ => ?_)
        rcases h_binary s i with hb | hb <;> rw [hb] <;> simp

/--
Prop. 16's reweighted-edge hypothesis discharged with **no `ρ` hypothesis at all**: only `±1`
outputs and a uniform-distribution error bound, with the concentration factor supplied by
`cor_rho_from_confidences`. Compare `lem_bounded_reweighting_edge`, which takes the cap as given.

Read the required uniform error honestly: it is `(1/2 - γ)` divided by `exp (2 * ∑ |α s|)`, so this
is a strong demand that grows stronger with each round. The point is that it is a demand about
*checkable quantities*, not that it is easy to meet.
-/
theorem cor_reweighting_edge_from_confidences (N : ℕ) [NeZero N] (y : Fin N → ℝ)
    (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) (γ : ℝ)
    (h_binary : ∀ s, ∀ i : Fin N, y i * h s i = 1 ∨ y i * h s i = -1)
    (h_unif : ((#{i | y i * h t i = -1} : ℕ) : ℝ) / N
      ≤ (1 / 2 - γ) / exp (2 * ∑ s ∈ range t, |α s|)) :
    err_t N y h α t ≤ 1 / 2 - γ :=
  lem_bounded_reweighting_edge N y h α t γ (exp (2 * ∑ s ∈ range t, |α s|)) (exp_pos _)
    (cor_rho_from_confidences N y h α t h_binary) h_unif

/--
Margin identity for the *unweighted* majority vote: with all confidences set to 1, the margin at
point `i` is exactly `k` minus twice the number of workers that get `i` wrong. Elementary, but it
is the algebraic core of both results below, so it is factored out.

Note `α := fun _ => 1` makes `f N h (fun _ => 1) k i = ∑ t ∈ range k, h t i` --- the repo's
existing `f` (`Setup.lean`) *is* the unweighted vote at that `α`, so no parallel definition of
majority voting is introduced.
-/
lemma vote_margin_eq (N : ℕ) (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (k : ℕ)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1) (i : Fin N) :
    y i * f N h (fun _ => 1) k i = (k : ℝ) - 2 * (failMult N y h k i : ℝ) :=
by
  have hf : f N h (fun _ => 1) k i = ∑ t ∈ range k, h t i := by
    unfold f
    exact sum_congr rfl (fun t _ => one_mul _)
  have hsplit := sum_filter_add_sum_filter_not (range k) (fun t => y i * h t i = -1)
    (fun t => y i * h t i)
  have hA : ∑ t ∈ filter (fun t => y i * h t i = -1) (range k), y i * h t i
      = -(failMult N y h k i : ℝ) := by
    rw [sum_congr rfl (fun t ht => (mem_filter.mp ht).2), sum_const, nsmul_eq_mul]
    unfold failMult
    ring
  have hB : ∑ t ∈ filter (fun t => ¬(y i * h t i = -1)) (range k), y i * h t i
      = ((#{t ∈ range k | ¬(y i * h t i = -1)} : ℕ) : ℝ) := by
    rw [sum_congr rfl (fun t ht => ?_), sum_const, nsmul_eq_mul, mul_one]
    have hmem := mem_filter.mp ht
    rcases h_binary t i with h1 | h1
    · exact h1
    · exact absurd h1 hmem.2
  have hcard : (failMult N y h k i) + (#{t ∈ range k | ¬(y i * h t i = -1)} : ℕ) = k := by
    unfold failMult
    rw [card_filter_add_card_filter_not]
    exact card_range k
  have hcardR : (failMult N y h k i : ℝ) + ((#{t ∈ range k | ¬(y i * h t i = -1)} : ℕ) : ℝ)
      = (k : ℝ) := by
    exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) hcard
  rw [hf, mul_sum]
  rw [hA, hB] at hsplit
  linarith [hsplit]

/--
Exact characterization of when the unweighted majority vote is right on everything (§F.3,
item (iii)): **zero training error iff every point's failure multiplicity is a strict minority.** Both
directions come off `vote_margin_eq`; the reverse direction is the sufficiency argument, the
forward direction reads it backwards through the fact that a vanishing sum of nonnegative
indicators forces each one to vanish.

Necessity is what is new here relative to the sufficiency-only reading §F.3 previously reported.
See `cor_tight_overlap_iff` for the same statement phrased with a single overlap parameter, and
`thm_overlap_boundary_sharp` for what happens at the boundary.
-/
theorem thm_overlap_iff_zero_error (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (k : ℕ)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1) :
    zero_one_loss N y h (fun _ => 1) k = 0 ↔ ∀ i, 2 * failMult N y h k i < k :=
by
  constructor
  · intro hloss i
    have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
    have hsum : ∑ j : Fin N, (if y j * f N h (fun _ => 1) k j ≤ 0 then (1 : ℝ) else 0) = 0 := by
      unfold zero_one_loss at hloss
      rcases mul_eq_zero.mp hloss with hone | hs
      · exfalso
        have : (0 : ℝ) < 1 / (N : ℝ) := one_div_pos.mpr hNR
        rw [hone] at this
        exact absurd this (lt_irrefl 0)
      · exact hs
    have hzero := (sum_eq_zero_iff_of_nonneg (fun j _ => by positivity)).mp hsum i (mem_univ i)
    have hpos : ¬(y i * f N h (fun _ => 1) k i ≤ 0) := by
      intro hc
      rw [if_pos hc] at hzero
      exact absurd hzero one_ne_zero
    rw [vote_margin_eq N y h k h_binary i, not_le] at hpos
    have hR : ((2 * failMult N y h k i : ℕ) : ℝ) < ((k : ℕ) : ℝ) := by
      push_cast
      linarith
    exact_mod_cast hR
  · intro hall
    have h_zero : ∀ i : Fin N, (if y i * f N h (fun _ => 1) k i ≤ 0 then (1 : ℝ) else 0) = 0 := by
      intro i
      have hltR : 2 * (failMult N y h k i : ℝ) < (k : ℝ) := by exact_mod_cast hall i
      have hpos : ¬(y i * f N h (fun _ => 1) k i ≤ 0) := by
        rw [vote_margin_eq N y h k h_binary i]
        linarith
      simp [hpos]
    unfold zero_one_loss
    simp [h_zero]

/--
Headline quantitative-diversity result (§F.3, item (iii)): **bounded overlap forces zero training
error under unweighted majority vote.** If no point is misclassified by more than `m` of the `k`
workers (`h_overlap`) and `m` is a strict minority (`2 * m < k`), the majority vote is exactly
right on every training point: `zero_one_loss = 0`.

Diversity is quantified as `m` --- the largest number of workers sharing any single failure point.
The condition `m < k/2` is stated in the tight form `2 * m < k` to avoid `ℕ`-division. Now a
one-line consequence of `thm_overlap_iff_zero_error`; kept as a named result because it is the form
cited in the paper.

**Relation to Prop. 16's `h_edge`.** This is a *sufficient* condition standing alongside `h_edge`,
and it is deterministic/combinatorial: no measure theory, no independence assumption. Its necessity
*as a characterization of zero error* is settled by `thm_overlap_iff_zero_error` /
`cor_tight_overlap_iff`. Its relation to `h_edge` is settled too, and negatively: the two conditions
are logically independent, neither implying the other (`thm_edge_overlap_independent`,
`EdgeVsOverlap.lean`). For the *probabilistic* counterpart of this result --- independent worker
errors rather than bounded overlap --- see `thm_condorcet_majority_bound` (`Condorcet.lean`), which
stays combinatorial by encoding independence in a product weight rather than deriving it.

**Which real setups plausibly satisfy `m < k/2`.** Workers drawn from genuinely different model
families, scaffolds, or prompts --- where failures are driven by different blind spots --- are the
plausible case. The implausible case is the same model resampled: its failures concentrate on the
same hard points, driving `m` toward `k`. That is exactly what Part B measures with this very
statistic --- `m⋆ = k` at every round, on one vendor's two tiers and three prompt variants, and
still `m⋆ = k` within either tier alone --- so the hypothesis is a real empirical constraint on
orchestration design, not a formality. (An earlier version of this comment cited Part B's
non-positive per-round edge instead. The paper withdraws that number as a back-filling artifact;
the overlap measurement is the one that carries the claim, and it does so directly.)
-/
theorem thm_bounded_overlap_zero_error (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ)
    (k m : ℕ)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (h_overlap : ∀ i, failMult N y h k i ≤ m)
    (h_minority : 2 * m < k) :
    zero_one_loss N y h (fun _ => 1) k = 0 :=
  (thm_overlap_iff_zero_error N y h k h_binary).mpr
    (fun i => by have := h_overlap i; omega)

/--
The overlap condition is **necessary as well as sufficient**, once `m` is the *tight* overlap
`m⋆ = max_i failMult i` rather than an arbitrary upper bound. Immediate from
`thm_overlap_iff_zero_error` and the defining properties of `Finset.sup'`.

This is the sense in which §F.3's earlier "`m < k/2` is sufficient, not necessary" understated the
situation: that reading is true only of a *loose* `m` (take `m = k` and the hypothesis fails while
the conclusion may hold), which is a remark about loose bounds rather than a gap in the condition.
At the tight overlap the characterization is exact.

Still **not** claimed *here*: any relation to Prop. 16's `h_edge`. That relation is settled
separately, and negatively --- `thm_edge_overlap_independent` (`EdgeVsOverlap.lean`) shows the tight
overlap condition `m⋆ < k/2` and the reweighted edge hypothesis are logically independent, neither
implying the other. So the two conditions do stand alongside each other, but not for want of an
answer: there is no equivalence to establish.
-/
theorem cor_tight_overlap_iff (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (k : ℕ)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1) :
    zero_one_loss N y h (fun _ => 1) k = 0
      ↔ 2 * univ.sup' univ_nonempty (failMult N y h k) < k :=
by
  rw [thm_overlap_iff_zero_error N y h k h_binary]
  constructor
  · intro hall
    obtain ⟨i₀, -, hi₀⟩ := exists_mem_eq_sup' (univ_nonempty (α := Fin N)) (failMult N y h k)
    rw [hi₀]
    exact hall i₀
  · intro hsup i
    have hle := le_sup' (failMult N y h k) (mem_univ i)
    omega

/--
**The bound is sharp, and fails maximally the moment it fails.** At the boundary `2 * m = k` --- the
very first violation of `2 * m < k` --- the zero-one loss is not merely nonzero but `1`: *every*
training point is misclassified. Stated for an arbitrary family whose overlap is exactly `k/2` at
every point, so no particular construction is privileged; `exists_boundary_overlap_family` supplies
a witness so the hypothesis is not vacuous.

Every margin is exactly `k - 2j = 0` --- a tie, which the repo's `zero_one_loss` convention
(`if y i * f i ≤ 0`) counts as an error, as it must, since a tied vote carries no information.

So `2 * m < k` cannot be weakened to `2 * m ≤ k`. The failure is not a technicality about one
awkward point: in the witness the workers are individually right half the time and evenly diverse,
and the vote is still worthless.
-/
theorem thm_overlap_boundary_sharp (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (j : ℕ)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (h_even : ∀ i, failMult N y h (2 * j) i = j) :
    zero_one_loss N y h (fun _ => 1) (2 * j) = 1 :=
by
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  have h_one : ∀ i : Fin N,
      (if y i * f N h (fun _ => 1) (2 * j) i ≤ 0 then (1 : ℝ) else 0) = 1 := by
    intro i
    have hmar : y i * f N h (fun _ => 1) (2 * j) i = 0 := by
      rw [vote_margin_eq N y h (2 * j) h_binary i, h_even i]
      push_cast
      ring
    rw [hmar]
    simp
  unfold zero_one_loss
  rw [sum_congr rfl (fun i _ => h_one i), sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one]
  field_simp

/--
Non-vacuity guard for `thm_overlap_boundary_sharp`: a family with overlap exactly `k/2` at every
point exists. The witness is the even split --- with `k = 2 * j` workers over two points, the first
`j` workers get point `0` right and point `1` wrong, and the last `j` do the reverse.
-/
lemma exists_boundary_overlap_family (j : ℕ) :
    ∃ (y : Fin 2 → ℝ) (h : ℕ → Fin 2 → ℝ),
      (∀ t, ∀ i : Fin 2, y i * h t i = 1 ∨ y i * h t i = -1) ∧
      (∀ i, failMult 2 y h (2 * j) i = j) :=
by
  have hcard_lt : #{t ∈ range (2 * j) | t < j} = j := by
    have hset : (range (2 * j)).filter (fun t => t < j) = range j := by
      ext t
      simp only [mem_filter, mem_range]
      omega
    rw [hset, card_range]
  have hcard_nlt : #{t ∈ range (2 * j) | ¬(t < j)} = j := by
    have hadd : #{t ∈ range (2 * j) | t < j} + #{t ∈ range (2 * j) | ¬(t < j)}
        = #(range (2 * j)) := card_filter_add_card_filter_not _
    rw [card_range, hcard_lt] at hadd
    omega
  refine ⟨fun _ => 1,
    fun t i => if i = 0 then (if t < j then (1 : ℝ) else -1) else (if t < j then -1 else 1),
    ?_, ?_⟩
  · intro t i
    by_cases hi : i = 0 <;> by_cases ht : t < j <;> simp [hi, ht]
  · intro i
    unfold failMult
    by_cases hi : i = 0
    · have hset : (range (2 * j)).filter (fun t => (1 : ℝ) *
          (if i = 0 then (if t < j then (1 : ℝ) else -1) else (if t < j then -1 else 1)) = -1)
          = (range (2 * j)).filter (fun t => ¬(t < j)) := by
        apply filter_congr
        intro t _
        by_cases ht : t < j <;> norm_num [hi, ht]
      rw [hset]
      exact hcard_nlt
    · have hset : (range (2 * j)).filter (fun t => (1 : ℝ) *
          (if i = 0 then (if t < j then (1 : ℝ) else -1) else (if t < j then -1 else 1)) = -1)
          = (range (2 * j)).filter (fun t => t < j) := by
        apply filter_congr
        intro t _
        by_cases ht : t < j <;> norm_num [hi, ht]
      rw [hset]
      exact hcard_lt

/--
Graded version, and the honest boundary of the counting argument (§F.3). Without any overlap cap,
double-counting plus Markov gives only
`zero_one_loss ≤ 2 * (average per-worker uniform error)`.

**State plainly what this is worth: nothing, on its own.** For workers with uniform error
`ε < 1/2`, the bound `2ε` is *weaker* than `ε` itself --- weaker than just using a single worker.
So counting alone buys no improvement: the `2 * m < k` overlap hypothesis in
`thm_bounded_overlap_zero_error` is load-bearing, and that is exactly the sense in which
*diversity*, not individual accuracy, is what the vote needs. A bound that improves on `ε` by
counting alone does not exist, because the adversarial case (all workers failing on the same `ε N`
points) genuinely has ensemble error `ε`.
-/
theorem cor_overlap_markov_bound (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ)
    (k : ℕ) (hk : 0 < k)
    (h_binary : ∀ t, ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1) :
    zero_one_loss N y h (fun _ => 1) k
      ≤ 2 * (1 / (k : ℝ)) * ∑ t ∈ range k, ((#{i | y i * h t i = -1} : ℕ) : ℝ) / N :=
by
  have hkR : (0 : ℝ) < k := Nat.cast_pos.mpr hk
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos N)
  have hind : ∀ i : Fin N, (if y i * f N h (fun _ => 1) k i ≤ 0 then (1 : ℝ) else 0)
      ≤ 2 * (failMult N y h k i : ℝ) / k := by
    intro i
    split_ifs with hcase
    · rw [vote_margin_eq N y h k h_binary i] at hcase
      rw [le_div_iff₀ hkR]
      linarith
    · positivity
  have hdc : ∑ i : Fin N, (failMult N y h k i : ℝ)
      = ∑ t ∈ range k, ((#{i | y i * h t i = -1} : ℕ) : ℝ) := by
    unfold failMult
    simp_rw [natCast_card_filter]
    exact sum_comm
  have hpull : ∑ i : Fin N, 2 * (failMult N y h k i : ℝ) / k
      = (2 / (k : ℝ)) * ∑ i : Fin N, (failMult N y h k i : ℝ) := by
    rw [mul_sum]
    exact sum_congr rfl (fun i _ => by ring)
  unfold zero_one_loss
  calc (1 / (N : ℝ)) * ∑ i : Fin N, (if y i * f N h (fun _ => 1) k i ≤ 0 then (1 : ℝ) else 0)
      ≤ (1 / (N : ℝ)) * ∑ i : Fin N, 2 * (failMult N y h k i : ℝ) / k := by
        exact mul_le_mul_of_nonneg_left (sum_le_sum (fun i _ => hind i)) (by positivity)
    _ = (1 / (N : ℝ)) * ((2 / (k : ℝ)) * ∑ i : Fin N, (failMult N y h k i : ℝ)) := by rw [hpull]
    _ = (1 / (N : ℝ)) * ((2 / (k : ℝ)) * ∑ t ∈ range k, ((#{i | y i * h t i = -1} : ℕ) : ℝ)) := by
        rw [hdc]
    _ = 2 * (1 / (k : ℝ)) * ∑ t ∈ range k, ((#{i | y i * h t i = -1} : ℕ) : ℝ) / N := by
        rw [← sum_div]
        ring

end
