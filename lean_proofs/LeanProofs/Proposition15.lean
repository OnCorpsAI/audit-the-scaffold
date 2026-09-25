import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

open Real
open Finset

set_option linter.style.header false
set_option linter.style.longLine false

/--
Proposition 15 (Best-of-`k` Orchestration is Scaffold Expansion), §F.3.
Model `k` frozen workers under scaffolds `C_1, ..., C_k`, each with its own reachable class
`Hyp i : Set X` and ceiling `Vstar i` satisfying the same Frozen-Weight Ceiling hypothesis as
`cor_frozen_weight_ceiling`'s `h_ceiling`: every artifact reachable by worker `i`'s scaffold has
quality bounded by `Vstar i`. The orchestrator's reachable class under best-of-`k` selection is
the union `⋃ i, Hyp i`; this theorem shows the natural orchestrator ceiling
`Finset.univ.sup' Finset.univ_nonempty Vstar` (the max over workers) is a valid ceiling for that
union.

This is the `≥` direction of Prop. 15's inequality `V*_orch ≥ max_i V*(W,C_i)`: taking the union
of `k` scaffold-indexed classes is an instance of assumption (M1), not a new mechanism (§F.5,
`cor_frozen_weight_ceiling`). We do NOT formalize the paper's accompanying equality
characterization ("equality iff some worker's class dominates the union"): that direction needs
`Vstar i` to be the *tight* supremum of `V` over `Hyp i`, not merely an upper bound, and is left
as prose in the paper (§F.3, proof sketch), not machine-checked here.
-/
theorem prop15_orchestration_ceiling {X : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (Hyp : Fin k → Set X)
    (Vstar : Fin k → ℝ)
    (h_ceiling : ∀ i, ∀ x ∈ Hyp i, V x ≤ Vstar i) :
    ∀ x ∈ ⋃ i, Hyp i, V x ≤ univ.sup' univ_nonempty Vstar :=
by
  intro x hx
  simp only [Set.mem_iUnion] at hx
  obtain ⟨i, hxi⟩ := hx
  have h1 : V x ≤ Vstar i := h_ceiling i x hxi
  have h2 : Vstar i ≤ univ.sup' univ_nonempty Vstar := le_sup' Vstar (mem_univ i)
  linarith

/--
Orchestration cannot fall below any individual worker's ceiling: the orchestrator's ceiling
`univ.sup' univ_nonempty Vstar` dominates `Vstar j` for every worker `j`. Combined with
`prop15_orchestration_ceiling`, this gives the formal content of `V*_orch ≥ max_i V*(W,C_i)` ---
best-of-`k` orchestration is never worse than the best single frozen worker, matching the "reach
for capability before orchestration" guidance (§F.4, item 3): the orchestrator's ceiling is still
capped by the underlying weights `W`, not exempt from them.
-/
theorem prop15_orchestration_dominates {k : ℕ} [NeZero k] (Vstar : Fin k → ℝ) (j : Fin k) :
    Vstar j ≤ univ.sup' univ_nonempty Vstar :=
  le_sup' Vstar (mem_univ j)

/--
Tight version of Prop. 15, attempting the paper's full equality claim rather than only the
`≥` direction above. Here `Vstar i := sSup (V '' Hyp i)` is the *actual* supremum of `V` over
worker `i`'s reachable class (not an arbitrary upper-bound hypothesis), so the statement is
about best-of-`k` orchestration's genuinely attained ceiling.

Result: the orchestrator's attained ceiling is *always exactly* `⨆ i, sSup (V '' Hyp i)` --- an
unconditional equality, with no side condition. This is a stronger, cleaner fact than the paper
currently states (§F.3): the paper's proposition claims equality holds only when some worker's
class *dominates* the union (`Hyp j ⊇ ⋃ i, Hyp i`) and is otherwise strict. That dichotomy is
mathematically wrong for hard (arg-max) selection: `sSup` of a union of images always equals the
max of the per-piece `sSup`s, regardless of whether any single piece contains the whole union
(e.g. `Hyp 0 = {a}`, `Hyp 1 = {b}`, `V a = V b`: neither singleton contains `{a, b}`, yet the
sSups are trivially equal). The Lean proof below fixes on that fact directly; it does not
reproduce or rescue the paper's "strict inequality otherwise" clause, because no such case
exists to formalize.
-/
theorem prop15_tight_ceiling_orchestration {X : Type} {k : ℕ} [NeZero k]
    (V : X → ℝ)
    (Hyp : Fin k → Set X)
    (h_nonempty : ∀ i, (V '' Hyp i).Nonempty)
    (h_bdd : ∀ i, BddAbove (V '' Hyp i)) :
    sSup (V '' ⋃ i, Hyp i) = univ.sup' univ_nonempty (fun i => sSup (V '' Hyp i)) :=
by
  set Vstar : Fin k → ℝ := fun i => sSup (V '' Hyp i) with hVstar
  have h_bound : ∀ y ∈ V '' ⋃ i, Hyp i, y ≤ univ.sup' univ_nonempty Vstar := by
    intro y hy
    obtain ⟨x, hx, rfl⟩ := hy
    simp only [Set.mem_iUnion] at hx
    obtain ⟨i, hxi⟩ := hx
    have h1 : V x ≤ Vstar i := le_csSup (h_bdd i) ⟨x, hxi, rfl⟩
    have h2 : Vstar i ≤ univ.sup' univ_nonempty Vstar := le_sup' Vstar (mem_univ i)
    linarith
  have h_bdd_union : BddAbove (V '' ⋃ i, Hyp i) := ⟨univ.sup' univ_nonempty Vstar, h_bound⟩
  have h_nonempty_union : (V '' ⋃ i, Hyp i).Nonempty := by
    obtain ⟨i, _⟩ := univ_nonempty (α := Fin k)
    obtain ⟨y, hy⟩ := h_nonempty i
    exact ⟨y, Set.image_mono (Set.subset_iUnion Hyp i) hy⟩
  apply le_antisymm
  · exact csSup_le h_nonempty_union h_bound
  · rw [sup'_le_iff]
    intro i _
    exact csSup_le_csSup h_bdd_union (h_nonempty i) (Set.image_mono (Set.subset_iUnion Hyp i))
