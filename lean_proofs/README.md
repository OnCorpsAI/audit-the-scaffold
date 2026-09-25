# lean_proofs

Lean 4 machine-checked proofs for the paper "Audit the Scaffold, Not the Checkpoint: A
Stationarity Dichotomy for Recursive Self-Improvement in Agentic Coding."

## What is verified

All proofs are complete with **zero `sorry` and zero custom axioms**. `Theorem3.lean`
takes the margin/VC-dimension bound as a local hypothesis (a deliberate design choice
documented in that file) rather than formalizing full VC-theory inline; all other proofs
rely only on Lean's standard library and Mathlib.

**For the authoritative statement-to-declaration map, see the paper's Formalization
Index (Appendix H).** It lists every result against the declarations that carry it, and
`test_lean_index.py` in the repository root keeps the two in step: it fails if a cited
declaration name does not exist, if a `sorry` or `axiom` appears here, or if a result in
the paper has no index row.

The table below is the per-file view of the same development. It deliberately carries no
paper result *numbers*: file names were fixed against an early draft, numbering has since
shifted more than once, and a numeric column here was wrong for both reasons — it named
an older draft's numbers while claiming to be current. Descriptions and declaration
names do not drift the way numbers do.

| File | What it proves |
|---|---|
| `Theorem1.lean` | Training error of agentic boosting: `exp_loss = ∏ Z_t`, and 0-1 loss ≤ exponential loss (`thm1_zero_one_loss_bound` is the bound as the paper states it) |
| `Corollary2.lean` | Exponential decay of the training error under a uniform edge, `≤ exp(-2γ²T)` |
| `Theorem3.lean` | Generalization error, with the Schapire et al. margin bound as an explicit hypothesis (`MarginGeneralizationBound`) rather than derived |
| `Theorem4.lean` | Boosting as greedy coordinate descent: the optimal `α_t` and FSAM equivalence, including global minimality |
| `Theorem5.lean` | Monotone improvement under refinement, plus the saturation bound (partial sums only — the limit is not formalized) |
| `Corollary7.lean` | High-probability refinement, via a finite union bound over rounds against Mathlib's `IsProbabilityMeasure` |
| `Proposition8.lean` | The Stationarity Dichotomy (both parts), plus the Frozen-Weight Ceiling as a direct instantiation at `B := V*(W)` |
| `Proposition15.lean` | Best-of-`k` orchestration realizes the best worker's ceiling exactly (tight-equality and weaker upper-bound-hypothesis versions) |
| `SoftAggregation.lean` | Weighted aggregation beats the weak-learning baseline for large enough ensembles |
| `WidthRefinement.lean` | Multi-round best-of-`k` orchestration: width improves the per-round rate, not the total improvement budget |
| `Diversity.lean` | Bounded overlap (`2m < k`) characterizes a correct majority vote — necessary as well as sufficient, and sharp at the boundary — plus the counting-only Markov baseline and the density-ratio route to the reweighted edge hypothesis, with the concentration cap `ρ` derived from logged confidences |
| `SelfReuse.lean` | Immediate reuse zeroes the edge: a worker's weighted error against the distribution its own incorporation produced is exactly `1/2`; includes the derived AdaBoost reweighting recursion |
| `PoolReuse.lean` | Reused-pool horizon: a scheduled pool's margin never leaves the pool's nonnegative cone, so a pool that cannot fit the data fails the edge hypothesis by a computable round; includes the absence of any per-round obstruction |
| `InterpolatingPool.lean` | The complementary half of that dichotomy on `ε₀`: if the pool's cone fits the data, one `γ > 0` works against *every* distribution, so the edge survives every round. Proved by averaging and a pigeonhole, not by minimax duality |
| `EdgeVsOverlap.lean` | The bounded-overlap condition and the reweighted edge hypothesis are logically independent, neither implying the other; includes the exponential-free reweighting steps that keep concrete multi-round examples rational |
| `Condorcet.lean` | Condorcet/Hoeffding bound for independent worker errors, proved combinatorially (independence encoded in a product weight, not derived) |
| `CondorcetProb.lean` | The same bound with independence *derived*: workers become coordinates of a product of Bernoulli measures, independence follows from Mathlib's `iIndepFun_pi`, and the majority-failure event's measure is identified with `Condorcet.lean`'s `voteErr` |
| `Setup.lean` | Shared definitions: margin `f`, exponential loss, reweighting `D_t`, normalizer `Z_t` |
| `ZDecomposition.lean` | Auxiliary `Z_t` decomposition lemmas and the weighted error `err_t` |
| `AgenticBoostingProofs.lean` | Top-level module importing all of the above |

### Not formalized

The multi-round weighted-vote *run* — a schedule and confidence sequence exhibited
outright, as distinct from the per-round edge that feeds it — and the converse direction
of AdaBoost's minimax duality (weak learnability ⇒ interpolation with margin). Appendix
H records both.

## Running the proofs

```bash
cd lean_proofs
lake build
```

A clean build with no errors confirms all theorems are verified. Mathlib is fetched
automatically by `lake` on first build.

## Toolchain

The pinned Lean and Mathlib versions are in `lean-toolchain` and `lake-manifest.json`.
Using a different Mathlib revision may produce elaboration errors.
