import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import LeanProofs.Setup
import LeanProofs.Theorem1
import LeanProofs.ZDecomposition
import LeanProofs.SelfReuse
import LeanProofs.Diversity

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

noncomputable section

/-!
### Closed-form reweighting at the optimal confidence

`D_succ_eq` (`SelfReuse.lean`) gives the reweighting recursion with a `Z t` in the denominator and
an `exp` in the numerator. At the *optimal* confidence both collapse: `Z t = 2 ε_t e^{α_t}`, and the
tilt on a point is `e^{±α_t}`, so every `exp` cancels and the update is a division by a rational
function of `ε_t` alone. Concretely, AdaBoost's own balance property --- the failing half and the
passing half of `Z` are equal --- says the failing points collectively carry mass exactly `1/2`
after the update, and these two lemmas are the pointwise form of that.

This is what makes concrete multi-round examples checkable by `norm_num` with no transcendental
evaluation: pick rational `ε_t` and the whole distribution stays rational.
-/

/-- `Z` at the optimal confidence, written on the failing side. Mirrors the inline step in
`thm_self_reuse_zero_edge` (`SelfReuse.lean`). -/
lemma Z_eq_two_err_exp (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
    (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (hε_pos : 0 < err_t N y h α t) (hε_lt_one : err_t N y h α t < 1)
    (h_alpha : α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t)) :
    Z N y h α t = 2 * (err_t N y h α t * exp (α t)) :=
by
  have hbalance := optimal_alpha_balances (err_t N y h α t) (α t) hε_pos hε_lt_one h_alpha
  rw [z_decomposition N y h α t h_binary, hbalance]
  ring

/-- `Z` at the optimal confidence, written on the passing side. -/
lemma Z_eq_two_one_sub_err_exp (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ)
    (t : ℕ)
    (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (hε_pos : 0 < err_t N y h α t) (hε_lt_one : err_t N y h α t < 1)
    (h_alpha : α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t)) :
    Z N y h α t = 2 * ((1 - err_t N y h α t) * exp (-α t)) :=
by
  have hbalance := optimal_alpha_balances (err_t N y h α t) (α t) hε_pos hε_lt_one h_alpha
  rw [z_decomposition N y h α t h_binary, ← hbalance]
  ring

/-- **Failing points**: their weight is divided by `2 ε_t`. No `exp` survives. -/
lemma D_succ_fail (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
    (i : Fin N)
    (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (hε_pos : 0 < err_t N y h α t) (hε_lt_one : err_t N y h α t < 1)
    (h_alpha : α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t))
    (hi : y i * h t i = -1) :
    D N y h α (t + 1) i = D N y h α t i / (2 * err_t N y h α t) :=
by
  have htilt : -y i * α t * h t i = α t := by
    calc -y i * α t * h t i = -(y i * h t i) * α t := by ring
      _ = -(-1) * α t := by rw [hi]
      _ = α t := by ring
  have hexp : exp (α t) ≠ 0 := ne_of_gt (exp_pos _)
  have hε_ne : err_t N y h α t ≠ 0 := ne_of_gt hε_pos
  rw [D_succ_eq, htilt, Z_eq_two_err_exp N y h α t h_binary hε_pos hε_lt_one h_alpha]
  field_simp

/-- **Passing points**: their weight is divided by `2 (1 - ε_t)`. No `exp` survives. -/
lemma D_succ_pass (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
    (i : Fin N)
    (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (hε_pos : 0 < err_t N y h α t) (hε_lt_one : err_t N y h α t < 1)
    (h_alpha : α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t))
    (hi : y i * h t i = 1) :
    D N y h α (t + 1) i = D N y h α t i / (2 * (1 - err_t N y h α t)) :=
by
  have htilt : -y i * α t * h t i = -α t := by
    calc -y i * α t * h t i = -(y i * h t i) * α t := by ring
      _ = -1 * α t := by rw [hi]
      _ = -α t := by ring
  have hexp : exp (-α t) ≠ 0 := ne_of_gt (exp_pos _)
  have hone_ne : 1 - err_t N y h α t ≠ 0 := by
    have : 0 < 1 - err_t N y h α t := by linarith
    exact ne_of_gt this
  rw [D_succ_eq, htilt, Z_eq_two_one_sub_err_exp N y h α t h_binary hε_pos hε_lt_one h_alpha]
  field_simp

/-!
### The witness family for §F.3 open item (iii)

Four points, all labelled `+1`, and three workers, described by which points they fail:
worker `0` fails `{0}`, worker `1` fails `{1}`, worker `2` fails `{0, 2}`. Point `0` is therefore
failed by two of the three workers, so the overlap condition `2 m_i < k` breaks at `i = 0` --- yet
the reweighted errors are `1/4`, `1/6`, `2/5`, every one of them at most `1/2 - 1/10`.
-/

/-- Labels for the item-(iii) witness: all four points positive. -/
def ovY : Fin 4 → ℝ := fun _ => 1

/-- Three workers over four points: `0` fails point `0`, `1` fails point `1`, `2` fails `0` and `2`. -/
def ovH : ℕ → Fin 4 → ℝ := fun t i =>
  if t = 0 then (if i = 0 then -1 else 1)
  else if t = 1 then (if i = 1 then -1 else 1)
  else (if i = 0 then -1 else if i = 2 then -1 else 1)

/-- The optimal confidences for `ovH`, one per round: `½log 3`, `½log 5`, `½log(3/2)`. -/
def ovA : ℕ → ℝ := fun t =>
  if t = 0 then (1 / 2) * log 3
  else if t = 1 then (1 / 2) * log 5
  else (1 / 2) * log (3 / 2)

lemma ovH_binary : ∀ t, ∀ i : Fin 4, ovY i * ovH t i = 1 ∨ ovY i * ovH t i = -1 :=
by
  intro t i
  unfold ovY ovH
  split_ifs <;> norm_num

/-- Pointwise evaluation of worker `0`: it fails exactly point `0`. -/
lemma ov_h0 (i : Fin 4) : ovY i * ovH 0 i = if i = 0 then -1 else 1 :=
by
  unfold ovY ovH
  norm_num

/-- Pointwise evaluation of worker `1`: it fails exactly point `1`. -/
lemma ov_h1 (i : Fin 4) : ovY i * ovH 1 i = if i = 1 then -1 else 1 :=
by
  unfold ovY ovH
  norm_num

/-- Pointwise evaluation of worker `2`: it fails points `0` and `2`. -/
lemma ov_h2 (i : Fin 4) : ovY i * ovH 2 i = if i = 0 then -1 else if i = 2 then -1 else 1 :=
by
  unfold ovY ovH
  norm_num

/-- Round 0: the distribution is uniform, and worker `0` fails exactly point `0`. -/
lemma ov_err0 : err_t 4 ovY ovH ovA 0 = 1 / 4 :=
by
  unfold err_t
  have hfilt : filter (fun i => ovY i * ovH 0 i = -1) (univ : Finset (Fin 4)) = {0} := by
    ext i
    fin_cases i <;> norm_num [ovY, ovH]
  rw [hfilt, sum_singleton, D_zero_eq_uniform]
  norm_num

lemma ov_alpha0 : ovA 0 = (1 / 2) * log ((1 - err_t 4 ovY ovH ovA 0) / err_t 4 ovY ovH ovA 0) :=
by
  rw [ov_err0]
  norm_num [ovA]

/-- Worker `0`'s failure point absorbs half the mass, as AdaBoost's balance property requires. -/
lemma ov_D1_zero : D 4 ovY ovH ovA 1 0 = 1 / 2 :=
by
  rw [D_succ_fail 4 ovY ovH ovA 0 0 (ovH_binary 0) (by rw [ov_err0]; norm_num)
      (by rw [ov_err0]; norm_num) ov_alpha0 (by rw [ov_h0]; norm_num),
    D_zero_eq_uniform, ov_err0]
  norm_num

/-- The three points worker `0` gets right share the other half. -/
lemma ov_D1_other (i : Fin 4) (hi : i ≠ 0) : D 4 ovY ovH ovA 1 i = 1 / 6 :=
by
  rw [D_succ_pass 4 ovY ovH ovA 0 i (ovH_binary 0) (by rw [ov_err0]; norm_num)
      (by rw [ov_err0]; norm_num) ov_alpha0 (by rw [ov_h0]; norm_num [hi]),
    D_zero_eq_uniform, ov_err0]
  norm_num

/-- Round 1: worker `1` fails exactly point `1`, which now carries weight `1/6`. -/
lemma ov_err1 : err_t 4 ovY ovH ovA 1 = 1 / 6 :=
by
  unfold err_t
  have hfilt : filter (fun i => ovY i * ovH 1 i = -1) (univ : Finset (Fin 4)) = {1} := by
    ext i
    fin_cases i <;> norm_num [ovY, ovH]
  rw [hfilt, sum_singleton, ov_D1_other 1 (by norm_num)]

lemma ov_alpha1 : ovA 1 = (1 / 2) * log ((1 - err_t 4 ovY ovH ovA 1) / err_t 4 ovY ovH ovA 1) :=
by
  rw [ov_err1]
  norm_num [ovA]

lemma ov_D2_zero : D 4 ovY ovH ovA 2 0 = 3 / 10 :=
by
  rw [D_succ_pass 4 ovY ovH ovA 1 0 (ovH_binary 1) (by rw [ov_err1]; norm_num)
      (by rw [ov_err1]; norm_num) ov_alpha1 (by rw [ov_h1]; norm_num),
    ov_D1_zero, ov_err1]
  norm_num

lemma ov_D2_two : D 4 ovY ovH ovA 2 2 = 1 / 10 :=
by
  rw [D_succ_pass 4 ovY ovH ovA 1 2 (ovH_binary 1) (by rw [ov_err1]; norm_num)
      (by rw [ov_err1]; norm_num) ov_alpha1
      (by rw [ov_h1, if_neg (by decide : ¬(2 : Fin 4) = 1)]),
    ov_D1_other 2 (by decide), ov_err1]
  norm_num

/-- Round 2: worker `2` fails points `0` and `2`, of combined weight `3/10 + 1/10 = 2/5`. -/
lemma ov_err2 : err_t 4 ovY ovH ovA 2 = 2 / 5 :=
by
  unfold err_t
  have hfilt : filter (fun i => ovY i * ovH 2 i = -1) (univ : Finset (Fin 4)) = {0, 2} := by
    ext i
    fin_cases i <;> norm_num [ovY, ovH]
  rw [hfilt, sum_insert (by decide), sum_singleton, ov_D2_zero, ov_D2_two]
  norm_num

lemma ov_alpha2 : ovA 2 = (1 / 2) * log ((1 - err_t 4 ovY ovH ovA 2) / err_t 4 ovY ovH ovA 2) :=
by
  rw [ov_err2]
  norm_num [ovA]

/-- Point `0` is failed by workers `0` and `2`: two of the three, so `2 m_0 = 4 ≥ 3 = k`. -/
lemma ov_failMult_zero : failMult 4 ovY ovH 3 0 = 2 :=
by
  unfold failMult
  have hfilt : filter (fun t => ovY 0 * ovH t 0 = -1) (range 3) = {0, 2} := by
    ext t
    simp only [mem_filter, mem_range, mem_insert, mem_singleton]
    constructor
    · rintro ⟨ht, hval⟩
      interval_cases t
      · exact Or.inl rfl
      · rw [ov_h1] at hval
        norm_num at hval
      · exact Or.inr rfl
    · rintro (rfl | rfl)
      · exact ⟨by norm_num, by rw [ov_h0]; norm_num⟩
      · exact ⟨by norm_num, by rw [ov_h2]; norm_num⟩
  rw [hfilt]
  decide

/--
**The reweighted edge hypothesis does not imply bounded overlap.** Three workers over four points
satisfying every hypothesis of Prop. 16 --- binary outputs, errors strictly inside `(0,1)`, optimal
confidences, and edge `γ = 1/10` against the genuinely *reweighted* distribution at every round ---
and yet point `0` is failed by two of the three, so `2 m_0 = 4 ≥ 3 = k`: the overlap condition of
Prop. 18 fails, and with it the unweighted vote's zero training error.

Note this is checked against the real `D_t`, not against the uniform distribution: the errors
`1/4`, `1/6`, `2/5` are the reweighted ones, computed through `D_succ_fail`/`D_succ_pass`.
-/
theorem thm_edge_not_imply_overlap :
    (∀ t ∈ range 3, ∀ i : Fin 4, ovY i * ovH t i = 1 ∨ ovY i * ovH t i = -1) ∧
    (∀ t ∈ range 3, 0 < err_t 4 ovY ovH ovA t) ∧
    (∀ t ∈ range 3, err_t 4 ovY ovH ovA t < 1) ∧
    (∀ t ∈ range 3, ovA t
      = (1 / 2) * log ((1 - err_t 4 ovY ovH ovA t) / err_t 4 ovY ovH ovA t)) ∧
    (∀ t ∈ range 3, err_t 4 ovY ovH ovA t ≤ 1 / 2 - 1 / 10) ∧
    ¬ (∀ i : Fin 4, 2 * failMult 4 ovY ovH 3 i < 3) ∧
    zero_one_loss 4 ovY ovH (fun _ => 1) 3 ≠ 0 :=
by
  refine ⟨fun t _ i => ovH_binary t i, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro t ht
    simp only [mem_range] at ht
    interval_cases t
    · rw [ov_err0]; norm_num
    · rw [ov_err1]; norm_num
    · rw [ov_err2]; norm_num
  · intro t ht
    simp only [mem_range] at ht
    interval_cases t
    · rw [ov_err0]; norm_num
    · rw [ov_err1]; norm_num
    · rw [ov_err2]; norm_num
  · intro t ht
    simp only [mem_range] at ht
    interval_cases t
    · exact ov_alpha0
    · exact ov_alpha1
    · exact ov_alpha2
  · intro t ht
    simp only [mem_range] at ht
    interval_cases t
    · rw [ov_err0]; norm_num
    · rw [ov_err1]; norm_num
    · rw [ov_err2]; norm_num
  · intro hall
    have h0 := hall 0
    rw [ov_failMult_zero] at h0
    norm_num at h0
  · intro hzero
    have h0 := (thm_overlap_iff_zero_error 4 ovY ovH 3 ovH_binary).mp hzero 0
    rw [ov_failMult_zero] at h0
    norm_num at h0

/-!
### The converse witness

Worker `0` fails three of the four points; workers `1` and `2` are correct everywhere. Every point
is failed by at most one worker, so `2 m_i = 2 < 3` and the overlap condition holds with room to
spare --- but worker `0`'s error is `3/4`, so no positive `γ` makes the edge hypothesis true, even
at round zero where the distribution is still uniform.
-/

/-- Labels for the converse witness: all four points positive. -/
def ovY2 : Fin 4 → ℝ := fun _ => 1

/-- Worker `0` fails points `0,1,2`; every later worker is correct everywhere. -/
def ovH2 : ℕ → Fin 4 → ℝ := fun t i => if t = 0 then (if i = 3 then 1 else -1) else 1

lemma ovH2_binary : ∀ t, ∀ i : Fin 4, ovY2 i * ovH2 t i = 1 ∨ ovY2 i * ovH2 t i = -1 :=
by
  intro t i
  unfold ovY2 ovH2
  split_ifs <;> norm_num

/-- Only worker `0` ever fails, so no point is failed by more than one worker. -/
lemma ov2_failMult_le (i : Fin 4) : failMult 4 ovY2 ovH2 3 i ≤ 1 :=
by
  unfold failMult
  have hsub : filter (fun t => ovY2 i * ovH2 t i = -1) (range 3) ⊆ {0} := by
    intro t ht
    simp only [mem_filter, mem_range] at ht
    simp only [mem_singleton]
    by_contra hne
    have hone : ovY2 i * ovH2 t i = 1 := by
      unfold ovY2 ovH2
      rw [if_neg hne]
      norm_num
    rw [hone] at ht
    norm_num at ht
  calc #(filter (fun t => ovY2 i * ovH2 t i = -1) (range 3)) ≤ #({0} : Finset ℕ) :=
        card_le_card hsub
    _ = 1 := by decide

/-- Worker `0`'s error against the uniform round-zero distribution is `3/4`. -/
lemma ov2_err0 (α : ℕ → ℝ) : err_t 4 ovY2 ovH2 α 0 = 3 / 4 :=
by
  unfold err_t
  have hfilt : filter (fun i => ovY2 i * ovH2 0 i = -1) (univ : Finset (Fin 4)) = {0, 1, 2} := by
    ext i
    fin_cases i <;> norm_num [ovY2, ovH2, Fin.ext_iff]
  rw [hfilt, sum_insert (by decide), sum_insert (by decide), sum_singleton,
    D_zero_eq_uniform, D_zero_eq_uniform, D_zero_eq_uniform]
  norm_num

/--
**Bounded overlap does not imply the reweighted edge hypothesis.** The overlap condition holds
(hence the unweighted vote already has zero training error, by Prop. 18), yet worker `0`'s error is
`3/4`, so the edge hypothesis fails for every `γ > 0` --- and fails on the bound, not by degenerating
to `ε ∈ {0,1}`, since `3/4` is strictly inside `(0,1)`.
-/
theorem thm_overlap_not_imply_edge :
    (∀ t, ∀ i : Fin 4, ovY2 i * ovH2 t i = 1 ∨ ovY2 i * ovH2 t i = -1) ∧
    (∀ i : Fin 4, 2 * failMult 4 ovY2 ovH2 3 i < 3) ∧
    zero_one_loss 4 ovY2 ovH2 (fun _ => 1) 3 = 0 ∧
    (∀ α : ℕ → ℝ, 0 < err_t 4 ovY2 ovH2 α 0 ∧ err_t 4 ovY2 ovH2 α 0 < 1) ∧
    (∀ (α : ℕ → ℝ) (γ : ℝ), 0 < γ → ¬ (err_t 4 ovY2 ovH2 α 0 ≤ 1 / 2 - γ)) :=
by
  have hoverlap : ∀ i : Fin 4, 2 * failMult 4 ovY2 ovH2 3 i < 3 := by
    intro i
    have := ov2_failMult_le i
    omega
  refine ⟨ovH2_binary, hoverlap, ?_, ?_, ?_⟩
  · exact (thm_overlap_iff_zero_error 4 ovY2 ovH2 3 ovH2_binary).mpr hoverlap
  · intro α
    rw [ov2_err0]
    norm_num
  · intro α γ hγ hcon
    rw [ov2_err0] at hcon
    linarith

/--
**§F.3 open item (iii), settled negatively.** The bounded-overlap condition `m^* < k/2`
(Prop. 18) and the reweighted edge hypothesis (Prop. 16) are **logically independent**: neither
implies the other. Both witnesses live on four points with three workers.

**What this does not claim.** It does not say the two conditions are unrelated in practice --- both
still express forms of diversity, and correlated workers plausibly fail both at once. It does not
weaken either proposition: each remains sufficient on its own terms. And it says nothing about
whether some *third* condition implies both.
-/
theorem thm_edge_overlap_independent :
    (∃ (y : Fin 4 → ℝ) (h : ℕ → Fin 4 → ℝ) (α : ℕ → ℝ) (γ : ℝ), 0 < γ ∧
        (∀ t ∈ range 3, ∀ i : Fin 4, y i * h t i = 1 ∨ y i * h t i = -1) ∧
        (∀ t ∈ range 3, 0 < err_t 4 y h α t) ∧
        (∀ t ∈ range 3, err_t 4 y h α t < 1) ∧
        (∀ t ∈ range 3, α t = (1 / 2) * log ((1 - err_t 4 y h α t) / err_t 4 y h α t)) ∧
        (∀ t ∈ range 3, err_t 4 y h α t ≤ 1 / 2 - γ) ∧
        ¬ (∀ i : Fin 4, 2 * failMult 4 y h 3 i < 3)) ∧
    (∃ (y : Fin 4 → ℝ) (h : ℕ → Fin 4 → ℝ),
        (∀ t, ∀ i : Fin 4, y i * h t i = 1 ∨ y i * h t i = -1) ∧
        (∀ i : Fin 4, 2 * failMult 4 y h 3 i < 3) ∧
        ∀ (α : ℕ → ℝ) (γ : ℝ), 0 < γ → ¬ (err_t 4 y h α 0 ≤ 1 / 2 - γ)) :=
by
  obtain ⟨hb, hp, hl, ha, he, hno, -⟩ := thm_edge_not_imply_overlap
  obtain ⟨hb2, ho2, -, -, he2⟩ := thm_overlap_not_imply_edge
  exact ⟨⟨ovY, ovH, ovA, 1 / 10, by norm_num, hb, hp, hl, ha, he, hno⟩,
    ⟨ovY2, ovH2, hb2, ho2, he2⟩⟩

end
