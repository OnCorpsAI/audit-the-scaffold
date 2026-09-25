import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
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
One-step form of Thm. 1: the exponential loss picks up exactly one `Z` factor per round.
Obtained from `thm1_training_error_bound` at `t` and `t + 1` plus `Finset.prod_range_succ`, so
`Theorem1.lean` needs no change (the same equality appears there as an unnamed `calc` step).
-/
lemma exp_loss_succ (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) :
    exp_loss N y h α (t + 1) = exp_loss N y h α t * Z N y h α t :=
by
  rw [thm1_training_error_bound, thm1_training_error_bound, prod_range_succ]

/-- `Z` is strictly positive, since it is a ratio of two strictly positive exponential losses. -/
lemma Z_pos (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ) :
    0 < Z N y h α t :=
by
  have h1 : 0 < exp_loss N y h α t * Z N y h α t := by
    rw [← exp_loss_succ]
    exact exp_loss_pos N y h α (t + 1)
  have h2 : 0 < exp_loss N y h α t := exp_loss_pos N y h α t
  rcases mul_pos_iff.mp h1 with ⟨-, hz⟩ | ⟨hneg, -⟩
  · exact hz
  · linarith

/--
The AdaBoost reweighting recursion, recovered from the repo's *closed-form* `D` (`Setup.lean`)
rather than posited: the round-`t+1` distribution is the round-`t` one tilted by the round-`t`
worker and renormalized by `Z t`. This is the bridge that makes questions about *reuse* --- what
happens to a worker on the distribution its own incorporation produced --- expressible at all.
-/
lemma D_succ_eq (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ) (t : ℕ)
    (i : Fin N) :
    D N y h α (t + 1) i = D N y h α t i * exp (-y i * α t * h t i) / Z N y h α t :=
by
  have hN : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
  have hE : exp_loss N y h α t ≠ 0 := ne_of_gt (exp_loss_pos N y h α t)
  have hZ : Z N y h α t ≠ 0 := ne_of_gt (Z_pos N y h α t)
  have hf : f N h α (t + 1) i = f N h α t i + α t * h t i := by
    unfold f
    rw [sum_range_succ]
  have hsplit : exp (-y i * f N h α (t + 1) i)
      = exp (-y i * f N h α t i) * exp (-y i * α t * h t i) := by
    rw [hf]
    have hring : -y i * (f N h α t i + α t * h t i)
        = -y i * f N h α t i + -y i * α t * h t i := by ring
    rw [hring, exp_add]
  unfold D
  rw [hsplit, exp_loss_succ]
  field_simp

/--
The reused worker's error factorizes. Summing `D_succ_eq` over the points worker `t` gets wrong ---
where the tilt is the constant `exp (α t)`, the same rewrite `z_decomposition` performs --- gives
worker `t`'s weighted error against the *next* round's distribution as `err_t * exp (α t) / Z t`.

Stated as a literal sum rather than through a new definition: the repo's `err_t` pins the worker
index to the distribution index, and generalizing it is unnecessary for the single question asked
here.
-/
lemma reused_worker_err_eq (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ)
    (t : ℕ) :
    ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α (t + 1) i
      = err_t N y h α t * exp (α t) / Z N y h α t :=
by
  have hterm : ∀ i ∈ filter (fun i => y i * h t i = -1) univ,
      D N y h α (t + 1) i = D N y h α t i * exp (α t) / Z N y h α t := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi
    have htilt : -y i * α t * h t i = α t := by
      calc -y i * α t * h t i = -(y i * h t i) * α t := by ring
        _ = -(-1) * α t := by rw [hi]
        _ = α t := by ring
    rw [D_succ_eq, htilt]
  calc ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α (t + 1) i
      = ∑ i ∈ filter (fun i => y i * h t i = -1) univ,
          D N y h α t i * exp (α t) / Z N y h α t := sum_congr rfl hterm
    _ = (∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α t i)
          * exp (α t) / Z N y h α t := by
        rw [← sum_div, ← sum_mul]
    _ = err_t N y h α t * exp (α t) / Z N y h α t := rfl

/--
At the optimal confidence, the two halves of `Z` are exactly equal: the mass AdaBoost removes from
the points worker `t` got right equals the mass it adds to the points worker `t` got wrong. This is
the whole content of the reuse result, and it is why no square roots are needed --- the identity
`exp (α t) ^ 2 = (1 - ε) / ε` is enough on its own.
-/
lemma optimal_alpha_balances (ε α : ℝ) (hε_pos : 0 < ε) (hε_lt_one : ε < 1)
    (h_alpha : α = (1 / 2) * log ((1 - ε) / ε)) :
    (1 - ε) * exp (-α) = ε * exp α :=
by
  have hratio_pos : 0 < (1 - ε) / ε := div_pos (by linarith) hε_pos
  have hsq : exp α * exp α = (1 - ε) / ε := by
    rw [h_alpha, ← exp_add]
    have hhalf : (1 / 2 : ℝ) * log ((1 - ε) / ε) + (1 / 2) * log ((1 - ε) / ε)
        = log ((1 - ε) / ε) := by ring
    rw [hhalf]
    exact exp_log hratio_pos
  have hea : exp α ≠ 0 := ne_of_gt (exp_pos α)
  have hsq' : ε * (exp α * exp α) = 1 - ε := by
    rw [hsq]
    field_simp
  rw [exp_neg, inv_eq_one_div, mul_one_div, div_eq_iff hea]
  linear_combination -hsq'

/--
**Immediate reuse zeroes the edge** (the self-reuse half of §F.3 open item (i), settled
*negatively*; the cross-worker half is `PoolReuse.lean`). Run one AdaBoost
round with worker `h t` at the optimal confidence `α t = ½ log((1-ε)/ε)`. Then worker `h t`'s
weighted error against the distribution its own incorporation produced, `D (t+1)`, is exactly
`1/2` --- edge exactly zero, not merely non-positive.

So the naive multi-round scheme --- keep consulting the worker you just consulted --- provably
cannot sustain Prop. 16's edge hypothesis for even one further round. §F.3 previously called
round-to-round self-correlation a *threat* to that hypothesis and claimed nothing either way; this
makes the sharp form of the claim a theorem, and it is the exact instance of AdaBoost's own
stopping rule (`ε ≥ ½ ⇒ α ≤ 0`) rather than an assumption added on top of it.

**What this does not claim.**
* It does *not* contradict Prop. 16 (`cor_soft_aggregation_beats_weak_baseline`), whose workers
  `h 0, …, h (k-1)` are each incorporated once against the distribution current at that time. No
  worker is reused there, so the hypothesis is untouched.
* It settles only *self*-reuse. The cross-worker question --- whether a *pool* of `k` workers can
  collectively sustain the edge hypothesis when reused across rounds --- is not addressed here; it
  is taken up in `PoolReuse.lean`, where the answer turns out to be about the pool's capacity rather
  than about correlation: the negative half is `thm_pool_reuse_horizon`, and the complementary
  interpolating half is `thm_interpolating_pool_sustains_edge` (`InterpolatingPool.lean`). In particular this
  result does *not* propagate: a worker with an edge against the distribution a reuse produces
  always exists (`exists_edge_worker_vs_any_round`), so there is no per-round obstruction.
* The alignment with Part B is **suggestive only**, and weaker than an earlier version of this
  comment claimed. Part B's *pooled* per-round edge is negative from round 1 onward, but the paper
  withdraws that as an artifact of a harness that stops calling the model once a task passes and
  back-fills a perfect score; on cells where a call actually happened the strong tier's early
  intervals include zero. What Part B measures is a loop that increasingly *abstains*, not one with
  a demonstrated non-positive edge. Part B also runs sequential self-refinement, not AdaBoost
  reweighting, so this is an analogy about why reuse is structurally unpromising, not a prediction
  the experiment confirms.
-/
theorem thm_self_reuse_zero_edge (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ)
    (α : ℕ → ℝ) (t : ℕ)
    (h_binary : ∀ i : Fin N, y i * h t i = 1 ∨ y i * h t i = -1)
    (hε_pos : 0 < err_t N y h α t) (hε_lt_one : err_t N y h α t < 1)
    (h_alpha : α t = (1 / 2) * log ((1 - err_t N y h α t) / err_t N y h α t)) :
    ∑ i ∈ filter (fun i => y i * h t i = -1) univ, D N y h α (t + 1) i = 1 / 2 :=
by
  have hbalance : (1 - err_t N y h α t) * exp (-α t) = err_t N y h α t * exp (α t) :=
    optimal_alpha_balances (err_t N y h α t) (α t) hε_pos hε_lt_one h_alpha
  have hZ : Z N y h α t = 2 * (err_t N y h α t * exp (α t)) := by
    rw [z_decomposition N y h α t h_binary, hbalance]
    ring
  have hden : 2 * (err_t N y h α t * exp (α t)) ≠ 0 :=
    ne_of_gt (mul_pos two_pos (mul_pos hε_pos (exp_pos (α t))))
  rw [reused_worker_err_eq, hZ, div_eq_iff hden]
  ring

/-- Round zero starts uniform: with an empty margin, `exp_loss = 1` and `D 0` is `1/N`. -/
lemma D_zero_eq_uniform (N : ℕ) [NeZero N] (y : Fin N → ℝ) (h : ℕ → Fin N → ℝ) (α : ℕ → ℝ)
    (i : Fin N) :
    D N y h α 0 i = 1 / N :=
by
  have hloss : exp_loss N y h α 0 = 1 := by
    unfold exp_loss f
    simp
  unfold D f
  rw [hloss]
  simp

/--
Non-vacuity guard for `thm_self_reuse_zero_edge`, in the style of `exists_width_orchestrator`: its
hypotheses are jointly satisfiable, so the conclusion is not empty. One worker over two points,
right on one and wrong on the other, has `err_t = 1/2` at round zero (where `D` is uniform by
`D_zero_eq_uniform`), which is strictly between `0` and `1`, and whose optimal confidence is
`α = 0`.
-/
lemma exists_zero_edge_reuse_instance :
    ∃ (y : Fin 2 → ℝ) (h : ℕ → Fin 2 → ℝ) (α : ℕ → ℝ),
      (∀ i : Fin 2, y i * h 0 i = 1 ∨ y i * h 0 i = -1) ∧
      0 < err_t 2 y h α 0 ∧ err_t 2 y h α 0 < 1 ∧
      α 0 = (1 / 2) * log ((1 - err_t 2 y h α 0) / err_t 2 y h α 0) :=
by
  have herr : err_t 2 (fun _ => 1) (fun _ i => if i = 0 then (1 : ℝ) else -1) (fun _ => 0) 0
      = 1 / 2 := by
    unfold err_t
    have hfilt : (univ : Finset (Fin 2)).filter
        (fun i => (1 : ℝ) * (if i = 0 then (1 : ℝ) else -1) = -1) = {1} := by
      ext i
      fin_cases i <;> norm_num
    rw [hfilt, sum_singleton, D_zero_eq_uniform]
    norm_num
  refine ⟨fun _ => 1, fun _ i => if i = 0 then (1 : ℝ) else -1, fun _ => 0, ?_, ?_, ?_, ?_⟩
  · intro i
    by_cases hi : i = 0 <;> norm_num [hi]
  · rw [herr]; norm_num
  · rw [herr]; norm_num
  · rw [herr]; norm_num

end
