#!/usr/bin/env python3
"""Part B diversity/edge statistics with task-cluster bootstrap CIs.

Two quantities the paper reports as point estimates, computed here with 95%
percentile CIs from a single resampling scheme -- a cluster bootstrap over the 55
tasks, carrying each task's 5 seed replicates together (the 5 seeds of a task are
not independent draws, so resampling the 275 rows individually would understate
the interval). Same case-resampling logic as ``part_b_anova_power`` -- see
``model_effect_bootstrap_ci`` there.

1. The per-round edge ``gamma_t`` (Section "The measured edge confirms the bounded
   regime"), in two forms. The pooled form averages over every cell including the
   rounds the experiment back-filled after a task's first pass, so it is reported
   only as a disclosed aggregate (``padding_stats`` quantifies the contamination:
   28-54% of cells). The form the paper relies on is ``headroom_edge_with_ci``,
   restricted to cells where a model call actually happened. ``abstention_bound``
   adds the confidence-rated reading, under which an unchanged round is an
   abstention rather than an error.

2. The failMult overlap estimator for Prop. 18 (Bounded Overlap Characterizes a
   Correct Vote): ``m_i`` = the number of workers that fail task i, the tight
   overlap ``m* = max_i m_i``, and the resulting unweighted-majority-vote error.
   Prop. 18 is a biconditional -- the vote has zero error iff ``2m_i < k`` for
   every i -- so measuring ``m_i`` on real workers directly checks a condition the
   appendix otherwise only asserts. Workers are the 30 (arm, seed) pairs of the
   6-arm K=55 sweep; a task is "failed" by a worker when that worker's
   running-best quality stays below tau, the same solve criterion used throughout
   (see ``compute_metrics`` and ``fig5_solve_comparison``).

Scope caveat, stated because the estimator is easy to over-read: the 30 workers
are one vendor, two capability tiers, and three feedback modes. This measures the
pessimistic half of the appendix's claim (a resampled same-family worker has high
failure overlap) and cannot test the optimistic half (that genuinely different
model families satisfy the condition). Part C cannot support this estimator at
all -- its sessions share no task instances across workers, so ``m_i`` is undefined
there.

Usage:
    uv run python scripts/part_b_edge_overlap.py [--boot-reps 9999] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(REPO, "data")

ARMS = [
    ("haiku", "independent", "haiku_independent"),
    ("haiku", "blind", "haiku_blind"),
    ("haiku", "diagnostic", "haiku_diagnostic"),
    ("sonnet", "independent", "sonnet_independent"),
    ("sonnet", "blind", "sonnet_blind"),
    ("sonnet", "diagnostic", "sonnet_diagnostic"),
]
TAU = 0.6
# SWE-bench's own "resolved" bar (every test passing). TAU=0.6 is a *fractional*
# passing threshold, so solve rates and the overlap statistic at TAU are not
# comparable to published SWE-bench resolve rates; both are computed so the paper can
# report the pair. The edge gamma_t and its headroom decomposition are tau-free --
# they compare q_t against q_{t-1} directly -- so only the overlap moves with tau.
TAU_RESOLVED = 1.0
CONFIDENCE = 0.95
CACHE_NAME = "part_b_edge_overlap.json"

# The attainable quality ceiling. Quality is ``fraction_passing`` over a task's
# FAIL_TO_PASS/PASS_TO_PASS set, so 1.0 is the maximum a row can reach and a row already
# at 1.0 cannot improve. Deliberately distinct from TAU: TAU=0.6 is the *solve* threshold
# used for the vote and the ensemble error, whereas a task at q=0.7 counts as solved yet
# still has headroom to 1.0. The headroom-conditional edge below turns on Q_MAX, not TAU.
Q_MAX = 1.0


def _percentile_ci(reps: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 + confidence) / 2 * 100
    return float(np.percentile(reps, lo_pct)), float(np.percentile(reps, hi_pct))


def _cluster_indices(n_clusters: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n_clusters) index matrix for a case-resampling bootstrap --
    the same construction as ``part_b_anova_power.model_effect_bootstrap_ci``."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_resamples, n_clusters))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_worker_cube() -> tuple[np.ndarray, list[str], list[str]]:
    """Load every (arm, seed) pair as a worker: ``(k, n_tasks, n_rounds)`` cube of
    *running-best* quality, plus the task ids and worker labels.

    Cannot reuse ``part_b_anova_power.load_raw_by_arm``: that returns only the
    final *raw* round, and both the vote and the per-round trajectory need the
    running best across all rounds -- quality does dip between rounds, on 2-12 of
    the 275 rows depending on the arm. That count is no longer asserted here: it is
    computed per arm as ``n_regressing_rows`` by ``edge_gamma_with_ci``.

    Tasks are aligned by ``instance_id`` rather than by row position, and the
    loader refuses to proceed unless all 6 arms share an identical task pool -- a
    silent mismatch would misalign workers and corrupt every ``m_i``.
    """
    cube_rows: list[np.ndarray] = []
    labels: list[str] = []
    reference_ids: list[str] | None = None

    for model, mode, tag in ARMS:
        path = os.path.join(DATADIR, f"llm_results_{tag}.json")
        with open(path) as f:
            r = json.load(f)
        k, n_seeds = r["config"]["k"], r["config"]["n_seeds"]
        ids = r["instance_ids"]
        q = np.asarray(r["quality_matrix"], dtype=float)
        if q.shape[0] != k * n_seeds:
            raise ValueError(
                f"{tag}: quality_matrix has {q.shape[0]} rows, expected k*n_seeds={k * n_seeds}"
            )

        # Rows are seed-major blocks: seed 0's k tasks, then seed 1's k tasks, and so on,
        # so a task's replicates sit k rows apart rather than adjacently. This loader is
        # immune to that either way because it indexes by instance_id, which is why the
        # ordering was never load-bearing here -- but ``edge_gamma_with_ci`` clusters by
        # arithmetic on the row index and does depend on it. See
        # ``test_real_arm_files_are_seed_major``, which pins the layout for both.
        running_best = np.maximum.accumulate(q, axis=1)
        by_task: dict[str, list[np.ndarray]] = {}
        for iid, row in zip(ids, running_best, strict=True):
            by_task.setdefault(iid, []).append(row)
        counts = {len(v) for v in by_task.values()}
        if counts != {n_seeds}:
            raise ValueError(
                f"{tag}: expected exactly n_seeds={n_seeds} rows per task, got counts={counts}"
            )

        if reference_ids is None:
            reference_ids = sorted(by_task)
        elif sorted(by_task) != reference_ids:
            raise ValueError(
                f"{tag}: arms do not share an identical task pool -- cannot align "
                "workers on a common set of points. Prop. 18's m_i is only defined "
                "when every worker is evaluated on the same tasks."
            )

        for seed_idx in range(n_seeds):
            cube_rows.append(np.stack([by_task[iid][seed_idx] for iid in reference_ids]))
            labels.append(f"{model}-{mode}-s{seed_idx}")

    if reference_ids is None:
        raise ValueError("no arms configured")
    return np.stack(cube_rows), reference_ids, labels


def load_pooled_quality(tag: str) -> tuple[np.ndarray, int]:
    """The ``(k*n_seeds, n_rounds)`` raw quality matrix for one arm, plus n_seeds --
    the pooled task x seed population the paper computes gamma_t over."""
    with open(os.path.join(DATADIR, f"llm_results_{tag}.json")) as f:
        r = json.load(f)
    return np.asarray(r["quality_matrix"], dtype=float), int(r["config"]["n_seeds"])


def load_solved_rounds(tag: str) -> list[int | None]:
    """The arm's ``solved_rounds``: for each row, the round at which the task first
    passed, or ``None`` if it never did.

    This is the provenance record for the *padding* described in ``padding_stats``.
    ``llm_simulations._run_trajectory`` breaks out of the round loop on the first pass and
    back-fills the remaining rounds with a hardcoded ``quality=1.0`` marked
    ``padded=True``, so rounds after ``solved_round`` are not measurements: no patch was
    generated and no evaluation ran. The matrix persisted to JSON keeps the values but not
    the per-record flag, so ``solved_rounds`` is the only surviving discriminator.
    """
    with open(os.path.join(DATADIR, f"llm_results_{tag}.json")) as f:
        raw = json.load(f)["solved_rounds"]
    return [None if x is None else int(x) for x in raw]


def load_instance_ids(tag: str) -> list[str]:
    """The arm's ``instance_ids`` in file order -- one entry per row of the matrix
    ``load_pooled_quality`` returns.

    Exposed so the seed-major row layout that ``edge_gamma_with_ci``'s clustering assumes
    can be asserted against the real files rather than trusted (see
    ``test_real_arm_files_are_seed_major``). The analysis path itself does not need this:
    ``load_worker_cube`` aligns by id and ``load_pooled_quality`` only needs the values.
    """
    with open(os.path.join(DATADIR, f"llm_results_{tag}.json")) as f:
        return [str(x) for x in json.load(f)["instance_ids"]]


# ---------------------------------------------------------------------------
# Item 5: per-round edge gamma_t with a cluster bootstrap CI
# ---------------------------------------------------------------------------


def _nan_to_none(values: np.ndarray) -> list[float | None]:
    """NaN -> None so the cache serialises to strict JSON (`null`, never bare `NaN`).

    NaN is the signal for "undefined for lack of a denominator", which happens for round 0
    (no predecessor) and for any round in which no row has headroom left. Emitting it as
    `null` keeps the distinction between "undefined" and a real 0.0 legible to any reader
    of the cache, and keeps the file parseable by non-Python consumers.
    """
    return [None if not np.isfinite(v) else float(v) for v in values]


def headroom_edge_with_ci(
    q: np.ndarray,
    n_seeds: int,
    n_resamples: int = 9999,
    seed: int = 42,
    confidence: float = CONFIDENCE,
) -> dict:
    """**The per-round edge on measured cells -- the estimator the paper reports.**

    The pooled ``gamma_t`` collapses two very different situations into one number: a row
    that *could* have improved and did not, and a row already at ``Q_MAX`` that *cannot*
    improve and so can only register as unimproved. This estimator removes the second by
    conditioning each round on ``q_{i,t-1} < Q_MAX``:

        ``gamma_headroom_t = (#improved and had headroom) / (#had headroom) - 1/2``

    An earlier version of this docstring called the split "a *descriptive decomposition* of
    ``gamma_t``, not a corrected estimate of it", on the reasoning that the conditioning
    variable is post-treatment. That reasoning does not survive contact with how the rows at
    ``Q_MAX`` arise. They are not rows the model was asked to improve and failed to: the
    experiment stops calling the model on the round a task passes and back-fills the rest
    (``padding_stats``, and ``llm_simulations._run_trajectory``). Those cells record no
    patch, no evaluation and no model call, so excluding them is not conditioning on an
    outcome -- it is declining to average over non-events. On this arm that is 28% (Haiku)
    to 54% (Sonnet) of the matrix. **This is therefore the corrected estimate and the pooled
    ``gamma_t`` is the contaminated aggregate**, which is the reverse of the earlier reading.

    Two honest limits remain, and neither is repaired by the above. The measured population
    still shrinks and hardens with ``t`` as the easy tasks solve and drop out (Sonnet:
    113 -> 70 -> 49 rows), so the decline in ``gamma_headroom`` across rounds is partly a
    change of subsample rather than of behaviour; it is the edge on cells that were still
    live, not a clean unconditional edge, which only a re-run without the early break would
    give. And the between-tier comparison the split licenses is the sharper reading either
    way: if one tier's edge moves a lot under conditioning and the other's barely moves, the
    pooled collapse has different causes in the two tiers.

    The denominator shrinks with ``t`` as rows reach ``Q_MAX``, so the interval matters more
    here than for ``gamma``. It comes from the same task-clustered resampling and the *same*
    ``seed``, hence the same cluster draws, so the two intervals are directly comparable;
    numerator and denominator are resampled together, which propagates the denominator's own
    sampling variation instead of holding it fixed. Rounds whose resample contains no
    headroom rows drop out of that round's percentile rather than contributing a NaN.
    """
    q = np.asarray(q, dtype=float)
    rows, n_rounds = q.shape
    if n_seeds < 1 or rows % n_seeds != 0:
        raise ValueError(f"n_seeds={n_seeds} does not divide {rows} rows")
    n_tasks = rows // n_seeds

    headroom = np.zeros_like(q)
    headroom[:, 1:] = (q[:, :-1] < Q_MAX).astype(float)
    improved_hr = np.zeros_like(q)
    improved_hr[:, 1:] = ((q[:, 1:] > q[:, :-1]) & (q[:, :-1] < Q_MAX)).astype(float)

    n_headroom = headroom.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma_hr = improved_hr.sum(axis=0) / n_headroom - 0.5
    gamma_hr[0] = np.nan  # round 0 has no predecessor, so headroom is undefined

    cl_num = improved_hr.reshape(n_seeds, n_tasks, n_rounds).transpose(1, 0, 2)
    cl_den = headroom.reshape(n_seeds, n_tasks, n_rounds).transpose(1, 0, 2)
    idx = _cluster_indices(n_tasks, n_resamples, seed)
    num = cl_num[idx].reshape(n_resamples, rows, n_rounds).sum(axis=1)
    den = cl_den[idx].reshape(n_resamples, rows, n_rounds).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = num / den - 0.5

    ci_lo: list[float | None] = []
    ci_hi: list[float | None] = []
    for t in range(n_rounds):
        finite = reps[np.isfinite(reps[:, t]), t]
        if t == 0 or finite.size == 0:
            ci_lo.append(None)
            ci_hi.append(None)
            continue
        lo, hi = _percentile_ci(finite, confidence)
        ci_lo.append(lo)
        ci_hi.append(hi)

    return {
        "gamma_headroom": _nan_to_none(gamma_hr),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_headroom": [int(x) for x in n_headroom],
        "q_max": Q_MAX,
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
        "n_pooled": rows,
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def padded_mask(solved_rounds: list[int | None], n_rounds: int) -> np.ndarray:
    """``(rows, n_rounds)`` boolean: True where the cell is back-filled, not measured.

    A cell is padded iff ``t > solved_round`` -- the experiment stops calling the model on
    the round a task passes, then writes a hardcoded 1.0 into every later round to keep the
    matrix rectangular (``llm_simulations.py``, ``_run_trajectory``).
    """
    mask = np.zeros((len(solved_rounds), n_rounds), dtype=bool)
    for i, s in enumerate(solved_rounds):
        if s is not None:
            mask[i, s + 1 :] = True
    return mask


def padding_stats(q: np.ndarray, solved_rounds: list[int | None]) -> dict:
    """How much of the pooled matrix is back-filled, and how much of ``gamma``'s tie mass
    that padding accounts for.

    This is the disclosure that makes the pooled ``gamma_t`` interpretable. ``gamma`` counts
    a padded round as a tie, and a tie scores ``-0.5`` on a strict ``>``; since a row is
    padded exactly because it *succeeded*, the pooled estimator is driven downward in
    proportion to the solve rate. The perverse consequence is worth stating numerically
    rather than in prose: a stronger tier pads more rows and so posts a *more* negative
    pooled edge than a weaker one, on the same data that shows it refining better once the
    padding is removed.

    The transition ``t-1 -> t`` is excluded from the measured population when ``t`` itself is
    padded. That set is identical to ``headroom_edge_with_ci``'s ``q_{t-1} < Q_MAX`` mask --
    a row sits at ``Q_MAX`` from its solve round onward and nowhere else -- so the two
    definitions agree cell for cell, and ``test_padded_mask_matches_headroom_mask`` pins
    that. Deriving the mask from ``solved_rounds`` rather than from ``q == Q_MAX`` is the
    principled form: it reads the experiment's own provenance record instead of inferring
    provenance from a value.
    """
    q = np.asarray(q, dtype=float)
    rows, n_rounds = q.shape
    padded = padded_mask(solved_rounds, n_rounds)

    tie = np.zeros_like(q, dtype=bool)
    tie[:, 1:] = q[:, 1:] == q[:, :-1]
    # Padded cells are ties by construction (a hardcoded 1.0 following a 1.0), so this is
    # the share of each round's tie mass that is an artifact rather than an observed no-op.
    with np.errstate(invalid="ignore", divide="ignore"):
        padded_share_of_tie = np.where(
            tie.sum(axis=0) > 0, (tie & padded).sum(axis=0) / tie.sum(axis=0), np.nan
        )

    return {
        "n_padded": int(padded.sum()),
        "n_cells": int(padded.size),
        "padded_share": float(padded.mean()),
        "padded_share_by_round": [float(x) for x in padded.mean(axis=0)],
        "padded_share_of_tie": _nan_to_none(padded_share_of_tie),
        "tie_total": [float(x) for x in tie.mean(axis=0)],
        "tie_genuine": [float(x) for x in (tie & ~padded).mean(axis=0)],
        "n_solved": int(sum(1 for s in solved_rounds if s is not None)),
        "n_rows": rows,
    }


def abstention_bound(q: np.ndarray) -> dict:
    """The confidence-rated (abstaining) boosting bound on measured cells.

    A refinement round that leaves quality unchanged is an *abstention*, not an error: it
    neither moves the artifact toward the target nor away from it. AdaBoost's binary edge
    ``gamma = Pr[improve] - 1/2`` has no representation for that -- it charges an abstention
    the same as a regression -- but \\citet{schapire1999improved}'s confidence-rated
    generalisation does. With per-round weights ``W_+`` (improved), ``W_-`` (regressed) and
    ``W_0`` (unchanged) over the measured population, the optimally-weighted round
    contributes a factor

        ``Z_t = W_0 + 2 * sqrt(W_+ * W_-)``

    to the training-error product bound. ``Z_t < 1`` whenever ``W_+`` exceeds ``W_-``, so a
    mostly-abstaining learner still drives the bound down, only slowly -- which is what
    saturation looks like in boosting's own terms, and is why the clipped-``gamma`` curve
    pinned at 1.0 is an artifact of the binary formulation rather than a property of the
    loop.

    Also returns the edge conditional on the round having changed anything at all,
    ``Pr[improve | improved or regressed] - 1/2``. That is a third conditioning, reported
    for completeness: it answers "when the loop acts, does it act correctly?" and is
    strongly positive throughout, but its denominator is small and selected on outcome.
    """
    q = np.asarray(q, dtype=float)
    _, n_rounds = q.shape

    measured = np.zeros_like(q, dtype=bool)
    measured[:, 1:] = q[:, :-1] < Q_MAX
    improved = np.zeros_like(q, dtype=bool)
    improved[:, 1:] = q[:, 1:] > q[:, :-1]
    regressed = np.zeros_like(q, dtype=bool)
    regressed[:, 1:] = q[:, 1:] < q[:, :-1]

    n_meas = measured.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_plus = (improved & measured).sum(axis=0) / n_meas
        w_minus = (regressed & measured).sum(axis=0) / n_meas
        w_zero = 1.0 - w_plus - w_minus
        z = w_zero + 2.0 * np.sqrt(w_plus * w_minus)
        moved = (improved | regressed) & measured
        n_moved = moved.sum(axis=0)
        edge_moved = np.where(
            n_moved > 0, (improved & measured).sum(axis=0) / n_moved - 0.5, np.nan
        )

    # Round 0 has no predecessor, so it contributes no factor and no conditional edge.
    z[0] = np.nan
    edge_moved[0] = np.nan
    z_product = float(np.prod(z[1:][np.isfinite(z[1:])])) if n_rounds > 1 else float("nan")

    return {
        "w_plus": _nan_to_none(w_plus),
        "w_minus": _nan_to_none(w_minus),
        "w_zero": _nan_to_none(w_zero),
        "z": _nan_to_none(z),
        "z_product": z_product,
        "edge_given_moved": _nan_to_none(edge_moved),
        "n_measured": [int(x) for x in n_meas],
        "q_max": Q_MAX,
    }


def edge_gamma_with_ci(
    q: np.ndarray,
    n_seeds: int,
    n_resamples: int = 9999,
    seed: int = 42,
    confidence: float = CONFIDENCE,
) -> dict:
    """Per-round edge ``gamma_t = mean_rows 1[q_t > q_{t-1}] - 1/2`` with a 95%
    percentile CI from a cluster bootstrap over tasks.

    **This is the contaminated aggregate, retained for disclosure rather than relied on.**
    Rounds after a task's first pass are back-filled, not measured (``padding_stats``), and
    ``gamma`` charges each one as a tie. Read ``headroom_edge_with_ci`` for the estimator on
    measured cells.

    ``q`` is the pooled ``(n_seeds * n_tasks, n_rounds)`` matrix in **seed-major**
    blocks (as written by the experiment: seed 0's k tasks, then seed 1's, so a task's
    replicates sit ``n_tasks`` rows apart); the resampling unit is the task, taken with
    all ``n_seeds`` of its replicates. Pass ``n_seeds=1`` to resample rows individually
    (only correct when there is one seed per task).

    The layout matters only for the bootstrap, and only for the interval: ``gamma`` is a
    mean over every row and so is order-invariant, whereas mis-grouping the clusters
    would resample across tasks and silently change the CI *width* while leaving the
    point estimate identical -- a failure with no visible symptom. The reshape below is
    therefore written seed-major to match the data, and
    ``test_real_arm_files_are_seed_major`` pins that assumption against the real files.

    Round 0's CI is the degenerate ``[0, 0]``, not a bootstrap: gamma_0 is fixed to
    0 because round 0 has no predecessor (matching ``compute_metrics``), so the raw
    formula's -0.5 is never a reported value and an interval around it would be
    an interval around a number the estimator does not produce.

    Also returns ``eta``, the mean per-round improvement
    ``eta_bar_t = mean_i (b_{i,t} - b_{i,t-1})`` on the running best
    ``b_{i,t} = max_{s<=t} q_{i,s}`` -- the quantity Thm. refinement bounds. It is
    computed here rather than read from the per-arm metrics files so that eta and
    gamma are guaranteed to come from the same pooled matrix, the same arm, and the
    same seed pooling; the paper quotes both side by side. Measuring eta on the
    running best rather than raw ``q`` is deliberate and matches
    ``llm_simulations.compute_metrics``: the cumulative sum then telescopes to
    ``b_{i,t} - q_{i,0} <= 1 - q_{i,0}``, which is the theorem's bound and holds for
    any path, whereas raw ``q`` would track upward variation and can exceed it.
    ``eta_0`` is 0 by construction, as with gamma.
    """
    q = np.asarray(q, dtype=float)
    rows, n_rounds = q.shape
    if n_seeds < 1 or rows % n_seeds != 0:
        raise ValueError(f"n_seeds={n_seeds} does not divide {rows} rows")
    n_tasks = rows // n_seeds

    improved = np.zeros_like(q)
    improved[:, 1:] = (q[:, 1:] > q[:, :-1]).astype(float)
    gamma = improved.mean(axis=0) - 0.5
    gamma[0] = 0.0

    # Split gamma's complement. `improved` uses a strict `>`, so "not improved" pools two
    # unlike cases: quality identical to the previous round, and quality actually lower.
    # Reporting them apart is what stops gamma's sign from being misread -- a converged,
    # never-regressing refiner scores the floor -0.5 on ties alone, which is a different
    # claim from AdaBoost's eps > 1/2 (a learner anti-correlated with the label).
    tie = np.zeros_like(q)
    tie[:, 1:] = (q[:, 1:] == q[:, :-1]).astype(float)
    regressed = np.zeros_like(q)
    regressed[:, 1:] = (q[:, 1:] < q[:, :-1]).astype(float)

    running_best = np.maximum.accumulate(q, axis=1)
    eta = np.zeros_like(q)
    eta[:, 1:] = running_best[:, 1:] - running_best[:, :-1]
    eta_bar = eta.mean(axis=0)

    # Rows whose quality never moves at all, and rows that dip at least once. The second
    # replaces a hand-asserted range in this module's own docstring; both are scalars over
    # the pooled (task, seed) population, not per-round rates.
    flat_fraction = float(np.all(np.diff(q, axis=1) == 0, axis=1).mean()) if n_rounds > 1 else 1.0
    n_regressing_rows = int((q != running_best).any(axis=1).sum())

    # (n_tasks, n_seeds, n_rounds): cluster = task, carrying all its seeds.
    clustered = improved.reshape(n_seeds, n_tasks, n_rounds).transpose(1, 0, 2)
    idx = _cluster_indices(n_tasks, n_resamples, seed)
    reps = clustered[idx].reshape(n_resamples, n_tasks * n_seeds, n_rounds).mean(axis=1) - 0.5

    ci_lo, ci_hi = [], []
    for t in range(n_rounds):
        if t == 0:
            ci_lo.append(0.0)
            ci_hi.append(0.0)
            continue
        lo, hi = _percentile_ci(reps[:, t], confidence)
        ci_lo.append(lo)
        ci_hi.append(hi)

    return {
        "gamma": [float(x) for x in gamma],
        "eta": [float(x) for x in eta_bar],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "tie": [float(x) for x in tie.mean(axis=0)],
        "regressed": [float(x) for x in regressed.mean(axis=0)],
        "flat_fraction": flat_fraction,
        "n_regressing_rows": n_regressing_rows,
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
        "n_pooled": rows,
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Item 4: the failMult overlap estimator for Prop. 18
# ---------------------------------------------------------------------------


def _overlap_stats(fail: np.ndarray) -> dict:
    """Per-round-slice statistics from a ``(k, n_tasks)`` boolean failure matrix.

    ``vote_wrong_i = 1[2 m_i >= k]``: Prop. 18's condition is ``2m_i < k``, and the
    boundary ``2m_i == k`` is a *tie*, which the zero-one loss counts as an error
    (the proposition's sharpness case) -- hence ``>=``, not ``>``.
    """
    k = fail.shape[0]
    m = fail.sum(axis=0)
    return {
        "k": int(k),
        "m": m.astype(int),
        "m_star": int(m.max()),
        "vote_wrong": (2 * m >= k).astype(float),
        "eps_per_task": fail.mean(axis=0),
    }


def _ci_block(
    vote_wrong: np.ndarray,
    eps_per_task: np.ndarray,
    idx: np.ndarray,
    confidence: float,
) -> dict:
    """Cluster-bootstrap intervals for one round slice, over a *shared* index matrix.

    ``idx`` is passed in rather than drawn here, and that is load-bearing twice over.
    It makes the rounds comparable to each other -- every round averages the same
    cluster draws, so a change across rounds is a change in the data and not in the
    resampling noise -- and it makes the final round's per-round intervals *identical*
    to the top-level ones by construction rather than by coincidence, which is what
    ``test_final_round_ci_matches_the_top_level_ci`` pins.
    """
    vote_reps = vote_wrong[idx].mean(axis=1)
    eps_reps = eps_per_task[idx].mean(axis=1)
    gap_reps = vote_reps - eps_reps
    eps_ci = _percentile_ci(eps_reps, confidence)
    return {
        "vote_err_ci": _percentile_ci(vote_reps, confidence),
        "eps_bar_ci": eps_ci,
        # A deterministic 2x of eps_bar's interval (cor_overlap_markov_bound), carried
        # explicitly so a consumer plotting the bound need not re-derive the factor.
        "markov_bound_ci": tuple(2 * x for x in eps_ci),
        "gap_ci": _percentile_ci(gap_reps, confidence),
        "gap_frac_positive": float((gap_reps > 0).mean()),
    }


def failmult_overlap(
    cube: np.ndarray,
    tau: float = TAU,
    n_resamples: int = 9999,
    seed: int = 42,
    confidence: float = CONFIDENCE,
) -> dict:
    """Measure Prop. 18's overlap condition on a ``(k, n_tasks, n_rounds)`` cube of
    running-best quality.

    Returns the tight overlap ``m*``, the fraction of tasks violating
    ``2m_i < k`` (which *is* the unweighted majority vote's zero-one error), the
    mean per-worker error ``eps_bar``, the Markov bound ``2*eps_bar`` from
    ``cor_overlap_markov_bound``, and the paired gap ``vote_err - eps_bar``, each
    with a 95% percentile CI from one cluster bootstrap over tasks. Per-round
    slices are reported too, since the appendix's claim is about the trajectory,
    and each carries the same four intervals -- Fig. 2's right panel draws them, and
    the round structure of ``gap_ci`` is itself reported (\\S app:orchestration): the
    gap excludes zero at the early rounds and includes it only at the last, so the
    top-level interval is a fact about the final round rather than about every round.

    ``m*`` is deliberately reported *without* a CI. It is a max over tasks, so a
    task-resampling bootstrap can never exceed the observed value -- the interval
    would be degenerate and one-sided by construction. Where ``m* == k`` it is
    additionally at its ceiling, where an interval carries no information at all.
    A CI on ``m*`` would require resampling workers, a different (and for this
    single-vendor pool, far less meaningful) question.
    """
    cube = np.asarray(cube, dtype=float)
    k, n_tasks, n_rounds = cube.shape

    final = _overlap_stats(cube[:, :, -1] < tau)
    vote_wrong, eps_per_task = final["vote_wrong"], final["eps_per_task"]

    idx = _cluster_indices(n_tasks, n_resamples, seed)
    final_ci = _ci_block(vote_wrong, eps_per_task, idx, confidence)

    vote_err = float(vote_wrong.mean())
    eps_bar = float(eps_per_task.mean())

    per_round = []
    for t in range(n_rounds):
        stats_t = _overlap_stats(cube[:, :, t] < tau)
        per_round.append(
            {
                "round": t,
                "m_star": stats_t["m_star"],
                "vote_err": float(stats_t["vote_wrong"].mean()),
                "eps_bar": float(stats_t["eps_per_task"].mean()),
                "gap": float(stats_t["vote_wrong"].mean() - stats_t["eps_per_task"].mean()),
                "n_violating": int(stats_t["vote_wrong"].sum()),
                **_ci_block(stats_t["vote_wrong"], stats_t["eps_per_task"], idx, confidence),
            }
        )

    return {
        "k": int(k),
        "n_tasks": int(n_tasks),
        "half_k": k / 2,
        "tau": tau,
        "m_star": final["m_star"],
        "m_histogram": np.bincount(final["m"], minlength=k + 1).tolist(),
        "n_violating": int(vote_wrong.sum()),
        "vote_err": vote_err,
        "vote_err_ci": final_ci["vote_err_ci"],
        "eps_bar": eps_bar,
        "eps_bar_ci": final_ci["eps_bar_ci"],
        "markov_bound": 2 * eps_bar,
        "markov_bound_ci": final_ci["markov_bound_ci"],
        "gap": vote_err - eps_bar,
        "gap_ci": final_ci["gap_ci"],
        "gap_frac_positive": final_ci["gap_frac_positive"],
        "per_round": per_round,
        "n_resamples": n_resamples,
        "confidence": confidence,
    }


def tier_overlap(
    cube: np.ndarray,
    labels: Sequence[str],
    tau: float = TAU,
    n_resamples: int = 9999,
    seed: int = 42,
    confidence: float = CONFIDENCE,
) -> dict[str, dict]:
    """The same overlap statistic restricted to one capability tier at a time.

    Separates the part of the primary k=30 result that survives tier restriction from
    the part that does not, because the two carry very different weight.

    ``m*`` survives, necessarily: it is a max over tasks, and a task that defeats all
    30 workers also defeats the 15 in either tier, so ``m* = k`` on both sides. That is
    the statistic Prop. 18 makes decisive, and this function is what lets the paper say
    so rather than leaving it as an inference.

    The gap ``vote_err - eps_bar`` does *not* survive, and reporting the pooled value
    alone would misattribute it. The k=30 pool mixes a strong tier with a weak one --
    not a pool any orchestrator would build -- and mixing inflates the majority-failure
    count while ``eps_bar`` averages the tiers instead. So a positive pooled gap can be
    a tier-pooling effect rather than evidence about voting, which is exactly what the
    real data turns out to show (see ``test_tier_overlap_can_reverse_the_pooled_gap_sign``
    for the mechanism on a hand-built fixture).

    Tiers are read off each label's model prefix rather than by slicing the worker axis
    positionally: ``load_worker_cube`` emits ``f"{model}-{mode}-s{seed}"`` in ARMS order,
    so a reordering of ARMS would silently mis-slice a positional split while still
    producing plausible-looking numbers.
    """
    cube = np.asarray(cube, dtype=float)
    if len(labels) != cube.shape[0]:
        raise ValueError(
            f"got {len(labels)} labels for {cube.shape[0]} workers -- labels must name "
            "every worker on the cube's first axis, or tiers group the wrong rows"
        )

    tiers: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        tiers.setdefault(label.split("-")[0], []).append(idx)

    return {
        tier: failmult_overlap(
            cube[rows], tau=tau, n_resamples=n_resamples, seed=seed, confidence=confidence
        )
        for tier, rows in tiers.items()
    }


def arm_level_overlap(cube: np.ndarray, n_seeds: int, tau: float = TAU) -> dict:
    """Companion at the coarser worker granularity: one worker per *arm*, solved by
    seed majority. Treats seeds as noise within a worker rather than as distinct
    workers -- a weaker test (k=6 gives a coarse m_i) reported as a robustness
    check on the k=30 primary, not as a co-equal result.
    """
    k_workers = cube.shape[0]
    if k_workers % n_seeds != 0:
        raise ValueError(f"n_seeds={n_seeds} does not divide {k_workers} workers")
    n_arms = k_workers // n_seeds
    solved = (cube[:, :, -1] >= tau).reshape(n_arms, n_seeds, -1)
    arm_fail = solved.mean(axis=1) <= 0.5  # (n_arms, n_tasks); tie -> not solved
    stats = _overlap_stats(arm_fail)
    return {
        "k": stats["k"],
        "half_k": stats["k"] / 2,
        "m_star": stats["m_star"],
        "m_histogram": np.bincount(stats["m"], minlength=stats["k"] + 1).tolist(),
        "n_violating": int(stats["vote_wrong"].sum()),
        "vote_err": float(stats["vote_wrong"].mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rounded(values: Sequence[float | None], digits: int = 3) -> list[float | None]:
    """Round a per-round series for console output, preserving ``None`` as ``None``.

    Kept distinct from a plain comprehension because ``None`` means "undefined for lack of a
    denominator" here and must not print as ``0.0`` (see ``_nan_to_none``).
    """
    return [None if v is None else round(v, digits) for v in values]


def _report_edge(tag: str, stats: dict, n_rows: int) -> None:
    """Print one tier's edge report. Split out of ``main`` to keep it under the
    complexity gate; pure output, so the numbers it prints are exactly the cached ones."""
    g, lo, hi = stats["gamma"], stats["ci_lo"], stats["ci_hi"]
    print(f"\ngamma_t ({tag}, diagnostic arm, N={stats['n_pooled']}):")
    for t, (gt, lt, ht) in enumerate(zip(g, lo, hi, strict=True)):
        note = "  (fixed: no predecessor)" if t == 0 else ""
        print(f"  t={t}: {gt:+.4f}  95% CI [{lt:+.4f}, {ht:+.4f}]{note}")
    print(f"eta_bar_t ({tag}, same matrix): {[round(x, 4) for x in stats['eta']]}")

    pad, abst, hr = stats["padding"], stats["abstention"], stats["headroom"]
    print(
        f"  padded (back-filled, not measured): {pad['n_padded']}/{pad['n_cells']} cells "
        f"= {pad['padded_share']:.1%}"
    )
    print(f"    padded share of each round's tie mass: {_rounded(pad['padded_share_of_tie'][1:])}")
    print(
        f"    tie total {_rounded(pad['tie_total'][1:])} -> "
        f"genuine {_rounded(pad['tie_genuine'][1:])}"
    )
    print(
        f"  abstaining bound Z_t (measured cells): {_rounded(abst['z'][1:])}  "
        f"product={abst['z_product']:.3f}"
    )
    print(f"    edge | round changed anything: {_rounded(abst['edge_given_moved'][1:])}")
    print(
        f"  ties/regressions ({tag}): tie={_rounded(stats['tie'][1:])}  "
        f"regressed={_rounded(stats['regressed'][1:])}"
    )
    print(
        f"  never-changing rows: {stats['flat_fraction']:.4f}   "
        f"rows dipping at least once: {stats['n_regressing_rows']}/{stats['n_pooled']}"
    )
    print(f"  gamma_t | headroom (q_{{t-1}} < {hr['q_max']}):")
    for t in range(1, len(hr["gamma_headroom"])):
        gh = hr["gamma_headroom"][t]
        if gh is None:
            print(f"    t={t}: undefined (no row had headroom)")
            continue
        lh, hh, nh = hr["ci_lo"][t], hr["ci_hi"][t], hr["n_headroom"][t]
        print(f"    t={t}: {gh:+.4f}  95% CI [{lh:+.4f}, {hh:+.4f}]  (n={nh} of {n_rows})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot-reps", type=int, default=9999, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed")
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU,
        help=f"primary solve threshold for the overlap statistic (default {TAU}). The "
        f"SWE-bench resolved bar {TAU_RESOLVED} is always reported alongside it. The edge "
        "gamma_t is tau-free and is unaffected by this flag.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Part B edge + overlap statistics (task-cluster bootstrap)")
    print(f"  boot_reps={args.boot_reps}  seed={args.seed}  tau={args.tau}")
    print("=" * 70)

    edge = {}
    for tag in ("haiku", "sonnet"):
        q, n_seeds = load_pooled_quality(f"{tag}_diagnostic")
        boot = {"n_resamples": args.boot_reps, "seed": args.seed}
        edge[tag] = edge_gamma_with_ci(q, n_seeds=n_seeds, **boot)
        # Re-measure the edge on rows that could still move, using the same seed as above so
        # both intervals rest on the same cluster draws; then quantify the back-filling that
        # makes the pooled gamma uninterpretable and give the confidence-rated reading.
        edge[tag]["headroom"] = headroom_edge_with_ci(q, n_seeds=n_seeds, **boot)
        edge[tag]["padding"] = padding_stats(q, load_solved_rounds(f"{tag}_diagnostic"))
        edge[tag]["abstention"] = abstention_bound(q)
        _report_edge(tag, edge[tag], len(q))

    worst_upper = max(hi for e in edge.values() for hi in e["ci_hi"][1:])
    print(f"\nWorst (closest-to-zero) upper bound over t>=1, both tiers: {worst_upper:+.4f}")

    cube, task_ids, labels = load_worker_cube()
    n_seeds = len(labels) // len(ARMS)
    overlap = failmult_overlap(cube, tau=args.tau, n_resamples=args.boot_reps, seed=args.seed)
    arm_level = arm_level_overlap(cube, n_seeds=n_seeds, tau=args.tau)
    # Same statistic at the resolved bar. Raising tau can only turn solved tasks into
    # failed ones, so m_i is monotone in tau and m*=k at the lower threshold already
    # forces m*=k here -- computed rather than asserted, and reported so the headline
    # reads as threshold-independent instead of resting on the looser bar.
    overlap_resolved = failmult_overlap(
        cube, tau=TAU_RESOLVED, n_resamples=args.boot_reps, seed=args.seed
    )
    # Same statistic within each tier. Reported because the two halves of the primary
    # result come apart here: m* stays at its ceiling, while the gap against eps_bar
    # does not survive, so the pooled gap's sign is a tier-pooling effect.
    by_tier = tier_overlap(cube, labels, tau=args.tau, n_resamples=args.boot_reps, seed=args.seed)

    print(f"\nfailMult overlap: k={overlap['k']} workers x {overlap['n_tasks']} tasks")
    print(f"  m* = {overlap['m_star']}  (condition needs m* < k/2 = {overlap['half_k']})")
    print(
        f"  violating tasks 2m_i >= k: {overlap['n_violating']}/{overlap['n_tasks']} "
        f"= {overlap['vote_err']:.4f}  95% CI "
        f"[{overlap['vote_err_ci'][0]:.4f}, {overlap['vote_err_ci'][1]:.4f}]"
    )
    print(
        f"  mean per-worker error eps_bar: {overlap['eps_bar']:.4f}  95% CI "
        f"[{overlap['eps_bar_ci'][0]:.4f}, {overlap['eps_bar_ci'][1]:.4f}]"
    )
    print(f"  Markov bound 2*eps_bar: {overlap['markov_bound']:.4f}")
    print(
        f"  gap (vote - single worker): {overlap['gap']:+.4f}  95% CI "
        f"[{overlap['gap_ci'][0]:+.4f}, {overlap['gap_ci'][1]:+.4f}]  "
        f"({overlap['gap_frac_positive']:.1%} of replicates positive)"
    )
    if overlap["gap_ci"][0] <= 0.0 <= overlap["gap_ci"][1]:
        print("  -> gap CI includes 0: directional only, NOT a significant difference.")
    print("  per round (m*, vote_err, eps_bar):")
    for entry in overlap["per_round"]:
        print(
            f"    t={entry['round']}: m*={entry['m_star']}  "
            f"vote_err={entry['vote_err']:.4f}  eps_bar={entry['eps_bar']:.4f}"
        )
    print(
        f"  arm-level companion (k={arm_level['k']}, seed majority): "
        f"m*={arm_level['m_star']}, violating {arm_level['n_violating']}/{overlap['n_tasks']}"
    )
    print(
        f"  at the resolved bar (tau={TAU_RESOLVED}): m*={overlap_resolved['m_star']}, "
        f"violating {overlap_resolved['n_violating']}/{overlap_resolved['n_tasks']} "
        f"= {overlap_resolved['vote_err']:.4f}"
    )
    print("  within one tier (m* survives; the gap against eps_bar does not):")
    for tier, res in sorted(by_tier.items()):
        flag = "" if res["m_star"] == res["k"] else "  <-- NOT at ceiling"
        print(
            f"    {tier}: k={res['k']}  m*={res['m_star']}{flag}  "
            f"vote_err={res['vote_err']:.4f}  eps_bar={res['eps_bar']:.4f}  "
            f"gap={res['gap']:+.4f}  95% CI "
            f"[{res['gap_ci'][0]:+.4f}, {res['gap_ci'][1]:+.4f}]"
        )

    out_path = os.path.join(DATADIR, CACHE_NAME)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "tau": args.tau,
                    "tau_resolved": TAU_RESOLVED,
                    "boot_reps": args.boot_reps,
                    "seed": args.seed,
                    "confidence": CONFIDENCE,
                    "arms": [tag for _m, _mo, tag in ARMS],
                    "workers": labels,
                    "n_tasks": len(task_ids),
                },
                "edge_gamma": edge,
                "edge_worst_upper": float(worst_upper),
                "overlap": overlap,
                "overlap_arm_level": arm_level,
                "overlap_resolved": overlap_resolved,
                "overlap_by_tier": by_tier,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
