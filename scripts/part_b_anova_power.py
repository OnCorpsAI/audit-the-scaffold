#!/usr/bin/env python3
"""Part B Phase 1: 2x3 repeated-measures ANOVA (model x feedback-mode) + a
case-resampling bootstrap power analysis, replicated on the K=55 harness-fixed sweep.

Same methodology as the prior K=50 pass documented in
tasks/part-b-phase1-stats-analysis.md (Sections 3-4) -- committed this time (that pass's
scripts were ad hoc). Differences from the prior pass, both intentional:
  - No harness-bug exclusion needed: all 55 tasks in this pool are gold-patch-verified
    clean under the fixed HarnessEvaluator (see tasks/part-b-scale-up-plan.md "Harness
    fixes landed"), vs. the prior pass's 23/50 excluded-for-infra-bugs subset.
  - K=55 instead of the prior pass's clean-27, so this IS (not merely estimates) a
    higher-powered replicate of the same design.

Design: task (n=55, repeated-measures subject) x model (Haiku/Sonnet) x feedback_mode
(independent/blind/diagnostic). One score per (task, model, mode) cell = final-round
(round index T-1) quality, averaged over the 5 seed replicates -- same DV definition as
the prior pass ("pivot data/llm_results_*.json final-round quality to task-level").

Usage:
    uv run python scripts/part_b_anova_power.py [--boot-reps 30000] [--k-grid 55,75,100,150]
        [--variance-filter]   # also run on tasks that are neither floor (all-0) nor
                               # ceiling (all-1) across all 6 arms x 5 seeds
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

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
MODELS = ("haiku", "sonnet")

# Hypothesized ordering (user's stated prior, corrected): diagnostic > blind >
# independent -- more carried-forward information should monotonically help:
# diagnostic has prior patch + critique (most informative), blind has prior patch only
# (some information -- still has something to refine even without explicit guidance),
# independent has no memory at all (least informative, can't accumulate any progress
# across rounds). Equally-spaced ranks -> standard orthogonal linear/quadratic
# polynomial contrast weights (dot product 0, confirmed): linear tests "does diagnostic
# beat independent, with blind in between", quadratic tests "is the pattern actually
# monotonic, or is blind an outlier relative to a straight-line fit between independent
# and diagnostic".
MODES_ORDERED = ["independent", "blind", "diagnostic"]
LINEAR_W = np.array([-1.0, 0.0, 1.0])
QUADRATIC_W = np.array([1.0, -2.0, 1.0])
if abs(float(np.dot(LINEAR_W, QUADRATIC_W))) >= 1e-12:
    raise ValueError("contrast weights must be orthogonal")


def load_raw_by_arm(
    use_running_best: bool = False,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Load every arm's raw (unaveraged) final-round quality, keyed
    ``raw[(model, mode)][task] = [5 per-seed values]``. Verifies all 6 arms share the
    exact same 55-task pool -- a silent mismatch here would silently corrupt every
    downstream number, however the arms are later subset/combined.

    ``use_running_best`` selects the robustness DV instead of the published one. The
    published DV is the last column of the matrix as stored, which treats two kinds of row
    differently without meaning to: a solved row was back-filled to 1.0 on the round it
    passed, so its final column *is* its running best, whereas an unsolved row carries
    whatever its last attempt scored, which may sit below its own earlier best (12/275 rows
    dip at least once on the strongest arm). Because the stronger tier solves more rows, the
    asymmetry runs in the direction of the tier effect. Passing True gives every row its
    running best and removes the asymmetry; ``main`` reports both so the tier effect is shown
    not to rest on it.
    """
    raw: dict[tuple[str, str], dict[str, list[float]]] = {}
    id_sets = []
    for model, mode, tag in ARMS:
        with open(os.path.join(DATADIR, f"llm_results_{tag}.json")) as f:
            r = json.load(f)
        ids = r["instance_ids"]
        id_sets.append(frozenset(ids))
        matrix = np.asarray(r["quality_matrix"], dtype=float)
        final_round = (
            np.maximum.accumulate(matrix, axis=1)[:, -1] if use_running_best else matrix[:, -1]
        )
        by_task: dict[str, list[float]] = {}
        for iid, val in zip(ids, final_round, strict=True):
            by_task.setdefault(iid, []).append(float(val))
        counts = {len(v) for v in by_task.values()}
        if counts != {r["config"]["n_seeds"]}:
            raise ValueError(
                f"{tag}: expected exactly n_seeds={r['config']['n_seeds']} "
                f"rows per task, got counts={counts}"
            )
        raw[(model, mode)] = by_task

    if len(set(id_sets)) != 1:
        pairwise_equal = ", ".join(str(a == b) for a, b in itertools.combinations(id_sets, 2))
        raise ValueError(
            "Arms do not share an identical task pool -- cannot build a "
            "balanced repeated-measures design. id_sets differ: "
            f"{[len(s) for s in id_sets]} tasks each, "
            f"pairwise-equal={{ {pairwise_equal} }}"
        )
    return raw


def build_table(
    raw: dict[tuple[str, str], dict[str, list[float]]], arms_subset: list[tuple[str, str]]
) -> pd.DataFrame:
    """One row per (task, model, mode) in ``arms_subset``, quality = mean over the 5
    seed replicates. Asserts the result is fully balanced (every task appears in every
    requested (model, mode) cell) -- an unbalanced subset would silently bias any
    repeated-measures F-test that assumes a complete design.
    """
    rows = []
    for model, mode in arms_subset:
        for iid, vals in raw[(model, mode)].items():
            rows.append(
                {"task": iid, "model": model, "mode": mode, "quality": float(np.mean(vals))}
            )
    df = pd.DataFrame(rows)
    n_tasks = df["task"].nunique()
    expected_rows = n_tasks * len(arms_subset)
    if len(df) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows (n_tasks={n_tasks} x "
            f"{len(arms_subset)} requested (model,mode) cells), got "
            f"{len(df)} -- subset is not fully balanced/crossed"
        )
    return df


def build_table_from_seeds(
    raw: dict[tuple[str, str], dict[str, list[float]]],
    arms_subset: list[tuple[str, str]],
    seed_indices: list[int],
) -> pd.DataFrame:
    """Like ``build_table``, but averaging only over ``seed_indices`` (positions into
    the 5-long per-seed list) instead of all 5 -- the basis for a seed-split gate: hold
    out one seed to decide floor/ceiling, test on the rest, so the gate is orthogonal to
    *mode* (unlike ``classify_tasks_by_variance`` scoped to a single mode) and usable
    when all modes are needed as data (e.g. the linear-trend contrast below).
    """
    rows = []
    for model, mode in arms_subset:
        for iid, vals in raw[(model, mode)].items():
            selected = [vals[i] for i in seed_indices]
            rows.append(
                {"task": iid, "model": model, "mode": mode, "quality": float(np.mean(selected))}
            )
    df = pd.DataFrame(rows)
    n_tasks = df["task"].nunique()
    expected_rows = n_tasks * len(arms_subset)
    if len(df) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows, got {len(df)} -- subset is not fully balanced/crossed"
        )
    return df


def pooled_raw_for_classification(
    raw: dict[tuple[str, str], dict[str, list[float]]], arms_subset: list[tuple[str, str]]
) -> dict[str, list[float]]:
    """Pool raw per-seed quality across the given (model, mode) arms, per task -- the
    basis for a floor/ceiling classification scoped to exactly those arms (e.g. only
    ``independent``, to gate task inclusion from a condition not part of the tested
    contrast -- see ``main``'s ``--independent-gate``).
    """
    out: dict[str, list[float]] = {}
    for model, mode in arms_subset:
        for iid, vals in raw[(model, mode)].items():
            out.setdefault(iid, []).extend(vals)
    return out


def classify_tasks_by_variance(
    raw_by_task: dict[str, list[float]], tol: float = 1e-9
) -> dict[str, str]:
    """Classify each task as ``floor`` (every one of its 30 raw seed-level final-round
    quality observations, pooled across all 6 arms, is exactly 0.0), ``ceiling`` (all
    exactly 1.0), or ``variable`` (anything else -- includes tasks that are e.g. flat 0.5,
    since a constant *non-0/1* profile is a data artifact worth surfacing separately, not
    silently bucketed as "variable" -- see the caller's reporting).

    Exact (tolerance-only) equality, not a near-floor/near-ceiling band, to match "all 1s"/
    "all 0s" literally -- a task that solves 4/5 seeds is retained as informative, not
    treated as ceiling.
    """
    out = {}
    for task, vals in raw_by_task.items():
        arr = np.asarray(vals)
        if np.all(np.abs(arr) < tol):
            out[task] = "floor"
        elif np.all(np.abs(arr - 1.0) < tol):
            out[task] = "ceiling"
        else:
            out[task] = "variable"
    return out


# ============================================================
# Classical two-way (both-within-subject) repeated-measures ANOVA,
# hand-derived, cross-checked against statsmodels.AnovaRM below.
# ============================================================


def two_way_rm_anova(y: np.ndarray) -> dict:
    """``y`` has shape ``(n_subjects, a_levels, b_levels)`` (fully balanced, one score
    per cell). Returns F/df/p/partial-eta2/Cohen's-f for the A main effect, B main
    effect, and A x B interaction, using the standard within-subjects SS decomposition
    (each effect tested against its own subject-interaction error term -- matches what
    ``AnovaRM(within=[A, B])`` computes for a two-within-factor design).
    """
    n, a, b = y.shape
    gm = y.mean()
    m_i = y.mean(axis=(1, 2))  # (n,)      subject means
    m_j = y.mean(axis=(0, 2))  # (a,)      A means
    m_k = y.mean(axis=(0, 1))  # (b,)      B means
    m_jk = y.mean(axis=0)  # (a, b)    A x B cell means
    m_ij = y.mean(axis=2)  # (n, a)    subject x A means
    m_ik = y.mean(axis=1)  # (n, b)    subject x B means

    ss_total = float(((y - gm) ** 2).sum())
    ss_a = n * b * float(((m_j - gm) ** 2).sum())
    ss_b = n * a * float(((m_k - gm) ** 2).sum())
    ss_ab = n * float(((m_jk - m_j[:, None] - m_k[None, :] + gm) ** 2).sum())
    ss_s = a * b * float(((m_i - gm) ** 2).sum())
    ss_sa = b * float(((m_ij - m_i[:, None] - m_j[None, :] + gm) ** 2).sum())
    ss_sb = a * float(((m_ik - m_i[:, None] - m_k[None, :] + gm) ** 2).sum())
    ss_sab = ss_total - ss_a - ss_b - ss_ab - ss_s - ss_sa - ss_sb

    df_a, df_sa = a - 1, (n - 1) * (a - 1)
    df_b, df_sb = b - 1, (n - 1) * (b - 1)
    df_ab, df_sab = (a - 1) * (b - 1), (n - 1) * (a - 1) * (b - 1)

    def _effect(ss_eff: float, df_eff: int, ss_err: float, df_err: int) -> dict[str, object]:
        ms_eff, ms_err = ss_eff / df_eff, ss_err / df_err
        f = ms_eff / ms_err
        p = float(stats.f.sf(f, df_eff, df_err))
        partial_eta2 = ss_eff / (ss_eff + ss_err)
        cohen_f = (
            float(np.sqrt(partial_eta2 / (1 - partial_eta2))) if partial_eta2 < 1 else float("inf")
        )
        return {
            "F": float(f),
            "df": (df_eff, df_err),
            "p": p,
            "partial_eta2": float(partial_eta2),
            "cohen_f": cohen_f,
        }

    return {
        "A": _effect(ss_a, df_a, ss_sa, df_sa),
        "B": _effect(ss_b, df_b, ss_sb, df_sb),
        "AB": _effect(ss_ab, df_ab, ss_sab, df_sab),
    }


def _vectorized_tier_effect_ss(
    y: np.ndarray, gm: np.ndarray, m_i: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized SS_A (tier main effect) / SS_S×A (tier×subject interaction) for a
    batch of resampled task cubes. ``y`` has shape ``(n_batch, n_obs, a, b)``; ``gm``
    (grand mean, shape ``(n_batch,)``) and ``m_i`` (subject means, shape ``(n_batch,
    n_obs)``) are passed in since both callers already compute them for other effects.
    Returns ``(ss_a_, ss_sa_)``, each shape ``(n_batch,)``. Shared by
    ``model_effect_bootstrap_ci`` (CI on the A/model effect itself) and
    ``bootstrap_power`` (needed there only as a piece of the ``ss_sab_`` residual).
    """
    b = y.shape[3]
    n_obs = y.shape[1]
    m_j = y.mean(axis=(1, 3))  # (n_batch, a)
    m_ij = y.mean(axis=3)  # (n_batch, n_obs, a)
    ss_a_ = n_obs * b * ((m_j - gm[:, None]) ** 2).sum(axis=1)
    ss_sa_ = b * (
        ((m_ij - m_i[:, :, None] - m_j[:, None, :] + gm[:, None, None]) ** 2).sum(axis=(1, 2))
    )
    return ss_a_, ss_sa_


def model_effect_bootstrap_ci(
    task_cube: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 42,
) -> dict:
    """Case-resampling bootstrap CI for the Model (A) effect's partial-eta2 and
    Cohen's f -- the huge effect size Table 2 reports for the capability effect,
    which (unlike Finding 2's capability-gap null) doesn't otherwise carry a CI.

    ``task_cube`` has shape ``(n_observed_tasks, 2, 3)`` (model tier x feedback
    mode), one quality score per cell per task. Resamples tasks with
    replacement (preserving each task's full 2x3 profile, i.e. case resampling,
    matching ``bootstrap_power``'s approach), recomputes the A-effect SS
    decomposition on each replicate (vectorized over the leading rep axis),
    and returns the percentile CI of partial-eta2 and Cohen's f across
    replicates around the actual (non-resampled) point estimate.

    Returns observed_partial_eta2, observed_cohen_f, eta2_ci_low, eta2_ci_high,
    cohen_f_ci_low, cohen_f_ci_high, n_tasks.
    """
    observed = two_way_rm_anova(task_cube)["A"]

    rng = np.random.default_rng(seed)
    n_obs = task_cube.shape[0]
    idx = rng.integers(0, n_obs, size=(n_resamples, n_obs))
    y = task_cube[idx]  # (n_resamples, n_obs, a, b)

    gm = y.mean(axis=(1, 2, 3))  # (n_resamples,)
    m_i = y.mean(axis=(2, 3))  # (n_resamples, n_obs)
    ss_a_, ss_sa_ = _vectorized_tier_effect_ss(y, gm, m_i)

    # A degenerate resample (ss_a_ + ss_sa_ == 0, e.g. every resampled task drawn from a
    # zero-variance subset) leaves partial_eta2/cohen_f undefined for that replicate;
    # mask it out rather than letting a 0/0 NaN silently corrupt the whole percentile CI.
    denom = ss_a_ + ss_sa_
    valid = denom > 0
    partial_eta2 = np.full_like(denom, np.nan)
    partial_eta2[valid] = ss_a_[valid] / denom[valid]
    finite = valid & (partial_eta2 < 1)
    cohen_f = np.full_like(partial_eta2, np.nan)
    cohen_f[finite] = np.sqrt(partial_eta2[finite] / (1 - partial_eta2[finite]))
    cohen_f[valid & ~finite] = np.inf

    lo_pct = (1 - confidence_level) / 2 * 100
    hi_pct = (1 + confidence_level) / 2 * 100

    return {
        "observed_partial_eta2": observed["partial_eta2"],
        "observed_cohen_f": observed["cohen_f"],
        "eta2_ci_low": float(np.nanpercentile(partial_eta2, lo_pct)),
        "eta2_ci_high": float(np.nanpercentile(partial_eta2, hi_pct)),
        "cohen_f_ci_low": float(np.nanpercentile(cohen_f, lo_pct)),
        "cohen_f_ci_high": float(np.nanpercentile(cohen_f, hi_pct)),
        "n_tasks": n_obs,
    }


def cross_check_with_statsmodels(df: pd.DataFrame) -> None:
    res = AnovaRM(df, depvar="quality", subject="task", within=["model", "mode"]).fit()
    print("\nstatsmodels AnovaRM cross-check:")
    print(res.anova_table)


# ============================================================
# Case-resampling bootstrap power analysis
# ============================================================


def bootstrap_power(
    task_cube: np.ndarray, k_grid: list[int], n_reps: int, rng: np.random.Generator
) -> dict:
    """``task_cube`` has shape ``(n_observed_tasks, 2, 3)``. For each K in ``k_grid``,
    resample K task-profiles with replacement (preserving each task's full 2x3 profile,
    i.e. case resampling, not cell-independent resampling) ``n_reps`` times and rerun
    ``two_way_rm_anova`` on each replicate, vectorized over reps for speed (the classical
    SS formulas above reduce to a handful of einsum-able reductions, batched over the
    leading rep axis) -- ``AnovaRM`` per replicate is far too slow for 30k reps x several
    K values, per the caveat in the prior pass's notes.
    """
    n_obs = task_cube.shape[0]
    power = {k: {"B": 0.0, "AB": 0.0} for k in k_grid}
    for k in k_grid:
        idx = rng.integers(0, n_obs, size=(n_reps, k))  # (n_reps, k) resampled task indices
        y = task_cube[idx]  # (n_reps, k, a=2, b=3)
        rep_p_b = np.empty(n_reps)
        rep_p_ab = np.empty(n_reps)
        # Vectorized classical SS decomposition, batched over the rep axis.
        n, a, b = k, y.shape[2], y.shape[3]
        gm = y.mean(axis=(1, 2, 3))  # (n_reps,)
        m_i = y.mean(axis=(2, 3))  # (n_reps, k)
        m_j = y.mean(axis=(1, 3))  # (n_reps, a)
        m_k = y.mean(axis=(1, 2))  # (n_reps, b)
        m_jk = y.mean(axis=1)  # (n_reps, a, b)
        m_ik = y.mean(axis=2)  # (n_reps, k, b)

        ss_total = ((y - gm[:, None, None, None]) ** 2).sum(axis=(1, 2, 3))
        ss_b_ = n * a * ((m_k - gm[:, None]) ** 2).sum(axis=1)
        ss_ab_ = n * (
            ((m_jk - m_j[:, :, None] - m_k[:, None, :] + gm[:, None, None]) ** 2).sum(axis=(1, 2))
        )
        ss_s_ = a * b * ((m_i - gm[:, None]) ** 2).sum(axis=1)
        ss_sb_ = a * (
            ((m_ik - m_i[:, :, None] - m_k[:, None, :] + gm[:, None, None]) ** 2).sum(axis=(1, 2))
        )
        ss_a_, ss_sa_ = _vectorized_tier_effect_ss(y, gm, m_i)
        ss_sab_ = ss_total - ss_a_ - ss_b_ - ss_ab_ - ss_s_ - ss_sa_ - ss_sb_

        df_b, df_sb = b - 1, (n - 1) * (b - 1)
        df_ab, df_sab = (a - 1) * (b - 1), (n - 1) * (a - 1) * (b - 1)

        f_b = (ss_b_ / df_b) / (ss_sb_ / df_sb)
        f_ab = (ss_ab_ / df_ab) / (ss_sab_ / df_sab)
        rep_p_b = stats.f.sf(f_b, df_b, df_sb)
        rep_p_ab = stats.f.sf(f_ab, df_ab, df_sab)

        power[k]["B"] = float(np.mean(rep_p_b < 0.05))
        power[k]["AB"] = float(np.mean(rep_p_ab < 0.05))
        print(f"  K={k}: power(mode)={power[k]['B']:.3f}  power(interaction)={power[k]['AB']:.3f}")
    return power


def run_anova_only(df: pd.DataFrame, label: str) -> tuple[dict, np.ndarray, list[str], list[str]]:
    """The ANOVA + statsmodels cross-check only, no bootstrap -- shared by every
    analysis variant below (full pool, variance-filter, single-mode gates) so a
    divergence in results is attributable to which tasks/modes were included, not to a
    second, differently-written analysis path. Returns ``(result, y, models, modes)``
    so callers that also want power (``run_analysis``) don't have to rebuild the pivot.
    """
    n_tasks = df["task"].nunique()
    tasks = sorted(df["task"].unique())
    models = sorted(df["model"].unique())
    modes = sorted(df["mode"].unique())
    pivot = df.pivot_table(index="task", columns=["model", "mode"], values="quality")
    pivot = pivot.reindex(index=tasks, columns=pd.MultiIndex.from_product([models, modes]))
    y = pivot.to_numpy().reshape(n_tasks, len(models), len(modes))
    if np.isnan(y).any():
        raise ValueError("missing cells in the task x model x mode pivot")

    print("\n" + "=" * 70)
    print(
        f"{len(models)}x{len(modes)} repeated-measures ANOVA -- {label} "
        f"(n={n_tasks} tasks, hand-derived, modes={modes})"
    )
    print("=" * 70)
    result = two_way_rm_anova(y)
    labels = {
        "A": "model (Haiku vs Sonnet)",
        "B": f"mode ({' vs '.join(modes)})",
        "AB": "model x mode (interaction)",
    }
    print(f"{'Effect':<28} {'F':>8} {'df':>12} {'p':>10} {'partial eta2':>14} {'Cohen f':>10}")
    for key, lbl in labels.items():
        e = result[key]
        print(
            f"{lbl:<28} {e['F']:>8.2f} {str(e['df']):>12} {e['p']:>10.4g} "
            f"{e['partial_eta2']:>14.3f} {e['cohen_f']:>10.3f}"
        )

    cross_check_with_statsmodels(df)
    return result, y, models, modes


def run_analysis(
    df: pd.DataFrame,
    label: str,
    k_grid: list[int],
    boot_reps: int,
    rng: np.random.Generator,
    ci_seed: int = 42,
) -> dict:
    """ANOVA + bootstrap power for one task subset (full 55, or a filtered subset)."""
    result, y, models, modes = run_anova_only(df, label)
    n_tasks = y.shape[0]
    tasks = sorted(df["task"].unique())

    print("\n" + "=" * 70)
    print(f"Case-resampling bootstrap power analysis -- {label} ({boot_reps} reps/K)")
    print("=" * 70)
    power = bootstrap_power(y, k_grid, boot_reps, rng)

    model_effect_ci = model_effect_bootstrap_ci(y, n_resamples=boot_reps, seed=ci_seed)

    return {
        "n_tasks_observed": n_tasks,
        "tasks": tasks,
        "anova": {k: result[k] for k in ["A", "B", "AB"]},
        "bootstrap_power": power,
        "model_effect_ci": model_effect_ci,
    }


def run_single_mode_gate(
    raw: dict[tuple[str, str], dict[str, list[float]]],
    gate_mode: str,
    models: tuple[str, ...] = MODELS,
) -> dict:
    """Leave-one-mode-out gate: classify floor/ceiling using ``gate_mode`` alone (pooled
    across both models x 5 seeds = 10 obs/task), then run the ANOVA-only pipeline on a
    model x {the other two modes} design restricted to the retained tasks. ``gate_mode``
    itself is dropped from the tested design entirely, not just from the gate -- this is
    what keeps the gate non-circular relative to the effect being tested.
    """
    test_modes = [m for m in ["independent", "blind", "diagnostic"] if m != gate_mode]
    gate_arms = [(m, gate_mode) for m in models]
    raw_gate = pooled_raw_for_classification(raw, gate_arms)
    classification = classify_tasks_by_variance(raw_gate)
    floor_ids = sorted(t for t, c in classification.items() if c == "floor")
    ceiling_ids = sorted(t for t, c in classification.items() if c == "ceiling")
    variable_ids = sorted(t for t, c in classification.items() if c == "variable")
    n_tasks = len(classification)

    print("\n" + "=" * 70)
    print(
        f"{gate_mode.upper()}-mode gate (raw per-seed quality, `{gate_mode}` arms only, "
        f"{len(models)} models x 5 seeds = {len(models) * 5} obs/task)"
    )
    print("=" * 70)
    print(f"  floor under {gate_mode} (n={len(floor_ids)}):    {floor_ids}")
    print(f"  ceiling under {gate_mode} (n={len(ceiling_ids)}): {ceiling_ids}")
    print(f"  variable under {gate_mode} (retained, n={len(variable_ids)})")
    excluded_frac = (len(floor_ids) + len(ceiling_ids)) / n_tasks
    print(
        f"  -> excluding {len(floor_ids) + len(ceiling_ids)}/{n_tasks} tasks "
        f"({excluded_frac:.0%}) based on `{gate_mode}` alone, then testing "
        f"model x {{{', '.join(test_modes)}}} only on the rest."
    )

    test_arms = [(m, mo) for m in models for mo in test_modes]
    df_test = build_table(raw, test_arms)
    df_gated = df_test[df_test["task"].isin(variable_ids)].reset_index(drop=True)
    gated_label = f"{gate_mode}-gated (model x {'/'.join(test_modes)})"
    result, _, _, _ = run_anova_only(df_gated, gated_label)
    return {
        "gate_mode": gate_mode,
        "test_modes": test_modes,
        "floor_ids": floor_ids,
        "ceiling_ids": ceiling_ids,
        "variable_ids": variable_ids,
        "anova": {k: result[k] for k in ["A", "B", "AB"]},
    }


def quadratic_contrast_test(y: np.ndarray) -> dict:
    """Single-df test of "is `blind` an outlier relative to a straight-line fit
    through independent and diagnostic" -- i.e. deviation from the linear/monotonic
    story, NOT covered by the linear contrast (which is exactly recovered by dropping
    `blind` and testing independent vs diagnostic directly -- see ``main``, no separate
    contrast code needed there since the omnibus 2-level ANOVA already IS that test).

    ``y`` must have modes on its last axis in ``MODES_ORDERED`` order (independent,
    blind, diagnostic). The quadratic weights ``[1, -2, 1]`` give a per-(task,
    model) score proportional to ``(independent + diagnostic)/2 - blind`` (up to a
    factor of -2, which flips sign but not significance) -- a self-contained,
    directly-interpretable one-sample/paired t-test on a well-defined quantity, not a
    derived decomposition of the omnibus SS that would need its own separate proof.
    """
    n, n_models, _ = y.shape
    score = y @ QUADRATIC_W  # (n_tasks, n_models)
    main = score.mean(axis=1)  # model-averaged midpoint-deviation, per task
    t_main, p_main = stats.ttest_1samp(main, 0.0)
    out = {
        "main_effect": {
            "t": float(t_main),
            "df": n - 1,
            "p": float(p_main),
            "mean": float(main.mean()),
            "cohen_d": float(main.mean() / main.std(ddof=1)),
        }
    }
    if n_models == 2:
        diff = score[:, 1] - score[:, 0]  # model B - model A
        t_int, p_int = stats.ttest_1samp(diff, 0.0)
        out["model_interaction"] = {
            "t": float(t_int),
            "df": n - 1,
            "p": float(p_int),
            "mean": float(diff.mean()),
            "cohen_d": float(diff.mean() / diff.std(ddof=1)),
        }
    return out


def print_trend_report(linear_result: dict, quadratic: dict, models: list[str]) -> None:
    print(
        "\n  Linear (diagnostic vs independent, blind excluded -- see the 2-level "
        "ANOVA above, which already IS this test):"
    )
    e = linear_result["B"]
    print(f"    F={e['F']:.3f} df={e['df']}  p={e['p']:.4g}  Cohen's f={e['cohen_f']:.3f}")
    e = linear_result["AB"]
    print(
        f"    model x linear: F={e['F']:.3f} df={e['df']}  p={e['p']:.4g}  "
        f"Cohen's f={e['cohen_f']:.3f}"
    )
    print("\n  Quadratic (is `blind` an outlier vs. the independent-diagnostic line?):")
    e = quadratic["main_effect"]
    print(
        f"    t={e['t']:.3f} df={e['df']}  p={e['p']:.4g}  Cohen's d={e['cohen_d']:.3f}  "
        f"(midpoint-deviation mean={e['mean']:.4f})"
    )
    if "model_interaction" in quadratic:
        e = quadratic["model_interaction"]
        print(
            f"    model x quadratic: t={e['t']:.3f} df={e['df']}  p={e['p']:.4g}  "
            f"Cohen's d={e['cohen_d']:.3f}  (between-model diff mean={e['mean']:.4f})"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot-reps", type=int, default=30000)
    parser.add_argument("--k-grid", type=str, default="55,75,100,150,200")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-power",
        action="store_true",
        help="ANOVA/contrasts only, skip the bootstrap power analysis "
        "entirely (no --boot-reps/--k-grid cost) -- use when only "
        "the effect estimates are needed, not a resampling curve.",
    )
    parser.add_argument(
        "--variance-filter",
        action="store_true",
        help="Also run the same analysis restricted to tasks that are "
        "neither floor (all-0) nor ceiling (all-1) across all 6 "
        "arms x 5 seeds -- reports the exclusion first, then the "
        "filtered-subset ANOVA/power alongside the full-pool one.",
    )
    parser.add_argument(
        "--gates",
        action="store_true",
        help="Run all 3 leave-one-mode-out gates (independent-gate, "
        "blind-gate, diagnostic-gate): classify floor/ceiling on "
        "one mode alone, test model x {the other two} on the rest. "
        "ANOVA only (bootstrap power not implemented for this path "
        "-- pass --skip-power, which is a no-op here regardless).",
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        help="Test the hypothesized ordering independent < blind < "
        "diagnostic via orthogonal linear + quadratic contrasts, "
        "gated on a HELD-OUT SEED (seed index 0 decides floor/"
        "ceiling per task across all 6 arms; the contrast is "
        "tested on the remaining 4 seeds only) -- orthogonal to "
        "mode, so usable even though the trend test needs all 3 "
        "modes as data. Also reports the ungated (all 5 seeds) "
        "version for comparison.",
    )
    parser.add_argument(
        "--trend-power",
        action="store_true",
        help="With --trend (and --skip-power NOT set): also run the "
        "case-resampling bootstrap power analysis for the linear "
        "contrast (independent vs diagnostic, blind excluded -- "
        "the same 2-level ANOVA the linear contrast already IS), "
        "for both the ungated and seed-0-gated task sets.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    raw = load_raw_by_arm()
    all_arms = [(m, mo) for m, mo, _ in ARMS]
    df = build_table(raw, all_arms)
    raw_by_task = pooled_raw_for_classification(raw, all_arms)
    n_tasks = df["task"].nunique()
    print(
        f"Loaded {n_tasks} tasks x 2 models x 3 modes = {len(df)} rows (final-round "
        f"quality, averaged over 5 seeds)."
    )

    k_grid = [int(k) for k in args.k_grid.split(",")]
    rng = np.random.default_rng(args.seed)

    if args.skip_power:
        result, y, *_ = run_anova_only(df, "full 55-task pool")
        full = {
            "n_tasks_observed": n_tasks,
            "anova": {k: result[k] for k in ["A", "B", "AB"]},
            "model_effect_ci": model_effect_bootstrap_ci(y, seed=args.seed),
        }
    else:
        full = run_analysis(df, "full 55-task pool", k_grid, args.boot_reps, rng, ci_seed=args.seed)
    out: dict = {"full_pool": full}

    if args.variance_filter:
        out["variance_filter"] = _variance_filter_block(df, raw_by_task, n_tasks, args, k_grid)

    if args.gates:
        out["gates"] = {}
        for gate_mode in ["independent", "blind", "diagnostic"]:
            out["gates"][gate_mode] = run_single_mode_gate(raw, gate_mode)

    if args.trend:
        out["trend"] = _trend_block(df, raw, all_arms, args, k_grid)

    # Robustness: the same ANOVA with every row contributing its running best, which removes
    # the solved/unsolved asymmetry in the published DV (see load_raw_by_arm). Cheap enough
    # to run unconditionally -- it is one more ANOVA on 55 rows, no bootstrap.
    rb_result, _rb_y, *_ = run_anova_only(
        build_table(load_raw_by_arm(use_running_best=True), all_arms),
        "full 55-task pool (running-best DV)",
    )
    out["running_best_dv"] = {k: rb_result[k] for k in ["A", "B", "AB"]}
    print(
        "\nRobustness -- running-best DV (removes the solved/unsolved asymmetry): "
        f"model partial eta^2 = {rb_result['A']['partial_eta2']:.4f}, "
        f"mode p={rb_result['B']['p']:.3f}, interaction p={rb_result['AB']['p']:.3f} "
        "(compare the published-DV row of the ANOVA table above)"
    )

    out_path = os.path.join(DATADIR, "part_b_k55_anova_power.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")

    task_level_path = os.path.join(DATADIR, "part_b_k55_task_level.csv")
    df.to_csv(task_level_path, index=False)
    print(f"Wrote {task_level_path}")


def _variance_filter_block(
    df: pd.DataFrame,
    raw_by_task: dict[str, list[float]],
    n_tasks: int,
    args: argparse.Namespace,
    k_grid: list[int],
) -> dict:
    """Re-run the analysis on tasks that are neither floor (all-0) nor ceiling (all-1).

    Reports the exclusion first, because it narrows the target population and the
    filtered results no longer generalize to the full K=55 pool.
    """
    classification = classify_tasks_by_variance(raw_by_task)
    floor_ids = sorted(t for t, c in classification.items() if c == "floor")
    ceiling_ids = sorted(t for t, c in classification.items() if c == "ceiling")
    variable_ids = sorted(t for t, c in classification.items() if c == "variable")
    print("\n" + "=" * 70)
    print("Floor/ceiling exclusion (raw per-seed final-round quality, all 6 arms x 5 seeds)")
    print("=" * 70)
    print(f"  floor (all-0, n={len(floor_ids)}):    {floor_ids}")
    print(f"  ceiling (all-1, n={len(ceiling_ids)}): {ceiling_ids}")
    print(f"  variable (retained, n={len(variable_ids)})")
    excluded_frac = (len(floor_ids) + len(ceiling_ids)) / n_tasks
    print(
        f"  -> excluding {len(floor_ids) + len(ceiling_ids)}/{n_tasks} tasks "
        f"({excluded_frac:.0%}); NOTE this narrows the target population to "
        f"'tasks where at least one (model, mode, seed) attempt differed from the "
        f"rest' -- results below no longer generalize to the full K=55 pool."
    )

    df_variable = df[df["task"].isin(variable_ids)].reset_index(drop=True)
    if args.skip_power:
        result, *_ = run_anova_only(df_variable, "variance-filtered subset")
        filtered = {
            "n_tasks_observed": len(variable_ids),
            "anova": {k: result[k] for k in ["A", "B", "AB"]},
        }
    else:
        rng2 = np.random.default_rng(args.seed + 1)
        filtered = run_analysis(
            df_variable,
            "variance-filtered subset",
            k_grid,
            args.boot_reps,
            rng2,
            ci_seed=args.seed + 1,
        )
    return {
        "floor_ids": floor_ids,
        "ceiling_ids": ceiling_ids,
        "variable_ids": variable_ids,
        **filtered,
    }


def _trend_block(
    df: pd.DataFrame,
    raw: dict,
    all_arms: list[tuple[str, str]],
    args: argparse.Namespace,
    k_grid: list[int],
) -> dict:
    """Test the hypothesized ordering independent < blind < diagnostic.

    Reported twice: ungated over all 5 seeds, and gated on a held-out seed (seed
    index 0 decides floor/ceiling per task across all 6 arms; the contrast is then
    tested on seeds 1-4 only, so seed 0 is excluded from the test data and not just
    from the gate).
    """
    print("\n" + "=" * 70)
    print("Ordering hypothesis: independent < blind < diagnostic")
    print("=" * 70)

    def _trend_report(
        df_mode_table: pd.DataFrame, label: str, rng_for_power: np.random.Generator | None
    ) -> dict:
        models = sorted(df_mode_table["model"].unique())
        id_diag = df_mode_table[
            df_mode_table["mode"].isin(["independent", "diagnostic"])
        ].reset_index(drop=True)
        print(f"\n-- {label} --")
        linear_result, y3, _, modes3 = run_anova_only(
            df_mode_table, f"{label} (all 3 modes, omnibus)"
        )
        # Reorder y3's mode axis to MODES_ORDERED for the quadratic contrast (modes3
        # comes back alphabetically sorted from run_anova_only's pivot).
        order_idx = [modes3.index(m) for m in MODES_ORDERED]
        y_ordered = y3[:, :, order_idx]
        quadratic = quadratic_contrast_test(y_ordered)
        linear_label = f"{label} (linear = independent vs diagnostic only)"
        if rng_for_power is not None:
            linear_analysis = run_analysis(
                id_diag, linear_label, k_grid, args.boot_reps, rng_for_power
            )
            linear_2level = {
                "A": linear_analysis["anova"]["A"],
                "B": linear_analysis["anova"]["B"],
                "AB": linear_analysis["anova"]["AB"],
            }
        else:
            linear_2level, *_ = run_anova_only(id_diag, linear_label)
            linear_analysis = None
        print_trend_report(linear_2level, quadratic, models)
        out_dict = {
            "omnibus_3mode": {k: linear_result[k] for k in ["A", "B", "AB"]},
            "linear_2level": {k: linear_2level[k] for k in ["A", "B", "AB"]},
            "quadratic": quadratic,
        }
        if linear_analysis is not None:
            out_dict["linear_2level_bootstrap_power"] = linear_analysis["bootstrap_power"]
        return out_dict

    run_power = args.trend_power and not args.skip_power
    power_rng_1 = np.random.default_rng(args.seed + 10) if run_power else None
    trend: dict = {"ungated_all_5_seeds": _trend_report(df, "ungated (all 5 seeds)", power_rng_1)}

    seed0_raw: dict[str, list[float]] = {}
    for model, mode in all_arms:
        for iid, vals in raw[(model, mode)].items():
            seed0_raw.setdefault(iid, []).append(vals[0])
    classification_seed0 = classify_tasks_by_variance(seed0_raw)
    floor_s = sorted(t for t, c in classification_seed0.items() if c == "floor")
    ceiling_s = sorted(t for t, c in classification_seed0.items() if c == "ceiling")
    variable_s = sorted(t for t, c in classification_seed0.items() if c == "variable")
    print(
        f"\nSeed-0 gate (raw quality at seed index 0 only, pooled across all 6 "
        f"arms, 6 obs/task): floor n={len(floor_s)} {floor_s}, ceiling "
        f"n={len(ceiling_s)} {ceiling_s}, variable (retained) n={len(variable_s)}. "
        f"Testing the trend on seeds 1-4 only (seed 0 excluded from the test data, "
        f"not just the gate)."
    )
    df_seeds1to4 = build_table_from_seeds(raw, all_arms, [1, 2, 3, 4])
    df_seeds1to4_gated = df_seeds1to4[df_seeds1to4["task"].isin(variable_s)].reset_index(drop=True)
    power_rng_2 = np.random.default_rng(args.seed + 11) if run_power else None
    trend["seed0_gated_seeds1to4"] = _trend_report(
        df_seeds1to4_gated, "seed-0-gated (tested on seeds 1-4)", power_rng_2
    )
    trend["seed0_gate_ids"] = {
        "floor": floor_s,
        "ceiling": ceiling_s,
        "variable": variable_s,
    }
    return trend


if __name__ == "__main__":
    main()
