"""Tests for scripts/part_b_edge_overlap.py.

Like test_part_b_anova_power.py, this targets only the pure, newly-added functions
-- the task-cluster bootstrap CI on the per-round edge gamma_t, and the failMult
overlap estimator for Prop. 18's condition m* < k/2. The module's data loader and
CLI driver are a one-off analysis path and are covered only where their invariants
are cheap to assert (the task-pool identity check).

The negative tests matter more than the happy path here, because both estimators
have a boundary that a naive implementation gets wrong:
  - gamma_0 is definitionally 0 (round 0 has no predecessor), so its CI must be
    exactly [0, 0], not a bootstrap of the raw formula (which would give
    [-0.5, -0.5] -- an interval around a value the estimator never reports).
  - Prop. 18 counts a *tie* (2m_i == k) as an error, so the vote's error must be
    1 at the sharpness boundary, not 0.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.part_b_edge_overlap as eo

# ---------------------------------------------------------------------------
# Fixtures: (n_workers, n_tasks, n_rounds) running-best cubes with hand-chosen
# failure patterns. TAU = 0.6, so 1.0 passes and 0.0 fails.
# ---------------------------------------------------------------------------

PASS, FAIL = 1.0, 0.0


def _cube(fail_sets: list[set[int]], n_tasks: int, n_rounds: int = 2) -> np.ndarray:
    """(k, n_tasks, n_rounds) cube where worker t fails exactly the tasks in
    ``fail_sets[t]`` (in every round). k = len(fail_sets)."""
    k = len(fail_sets)
    cube = np.full((k, n_tasks, n_rounds), PASS)
    for t, failed in enumerate(fail_sets):
        for i in failed:
            cube[t, i, :] = FAIL
    return cube


# ---------------------------------------------------------------------------
# failmult_overlap: Prop. 18's biconditional and its sharpness boundary
# ---------------------------------------------------------------------------


def test_disjoint_failures_give_zero_vote_error() -> None:
    """m* = 1 < k/2 = 1.5 with 3 workers failing disjoint tasks -> the vote is
    correct everywhere. This is the *sufficient* direction of Prop. 18; if the
    estimator reports a nonzero vote error here it is not measuring the
    proposition's condition."""
    cube = _cube([{0}, {1}, {2}], n_tasks=4)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert res["k"] == 3
    assert res["m_star"] == 1
    assert res["m_star"] * 2 < res["k"]
    assert res["vote_err"] == 0.0


def test_all_workers_fail_one_task_gives_m_star_k() -> None:
    """The extremal violation the appendix claims for same-family resampling:
    every worker fails the same task, so m* = k and the vote errs on it."""
    cube = _cube([{0}, {0}, {0}], n_tasks=4)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert res["m_star"] == 3 == res["k"]
    assert res["vote_err"] == pytest.approx(0.25)  # 1 of 4 tasks


def test_tie_boundary_counts_as_error() -> None:
    """Prop. 18's sharpness case: at 2m_i == k the margin is exactly 0, and the
    zero-one loss counts a tie as an error. A `>` instead of `>=` in the
    implementation would silently report 0 here."""
    cube = _cube([{0}, {0}, set(), set()], n_tasks=1)  # k=4, m_0=2, 2*2 == 4
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert res["k"] == 4
    assert res["m_star"] == 2
    assert 2 * res["m_star"] == res["k"]
    assert res["vote_err"] == 1.0


def test_markov_bound_holds_and_is_two_eps_bar() -> None:
    """cor_overlap_markov_bound: the vote's error is at most 2*eps_bar. Assert
    both that the reported bound is 2*eps_bar and that it actually holds."""
    cube = _cube([{0, 1}, {0, 2}, {0}], n_tasks=4)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert res["markov_bound"] == pytest.approx(2 * res["eps_bar"])
    assert res["vote_err"] <= res["markov_bound"] + 1e-12


def test_gap_is_vote_err_minus_eps_bar() -> None:
    cube = _cube([{0, 1}, {0, 2}, {0}], n_tasks=4)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert res["gap"] == pytest.approx(res["vote_err"] - res["eps_bar"])


def test_m_star_has_no_ci_reported() -> None:
    """m* is a max over tasks, so a task-resampling bootstrap can never exceed the
    observed value -- a degenerate one-sided interval. The estimator must not
    report one rather than reporting a misleading one."""
    cube = _cube([{0}, {1}, {2}], n_tasks=4)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert not any(key.startswith("m_star_ci") for key in res)


def test_per_round_statistics_have_one_entry_per_round() -> None:
    cube = _cube([{0}, {0, 1}, {1}], n_tasks=4, n_rounds=3)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=0)
    assert len(res["per_round"]) == 3
    for entry in res["per_round"]:
        assert set(entry) >= {"m_star", "vote_err", "eps_bar"}


def test_failmult_ci_is_deterministic_under_fixed_seed() -> None:
    cube = _cube([{0, 1}, {0, 2}, {0}], n_tasks=6)
    a = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=7)
    b = eo.failmult_overlap(cube, tau=0.6, n_resamples=199, seed=7)
    assert a["vote_err_ci"] == b["vote_err_ci"]
    assert a["gap_ci"] == b["gap_ci"]


def test_failmult_ci_brackets_the_observed_statistic() -> None:
    cube = _cube([{0, 1}, {0, 2}, {0}, {3}, {4}], n_tasks=10)
    res = eo.failmult_overlap(cube, tau=0.6, n_resamples=999, seed=3)
    lo, hi = res["vote_err_ci"]
    assert lo <= res["vote_err"] <= hi


# ---------------------------------------------------------------------------
# Per-round intervals: Fig. 2's right panel draws a CI on every bar, so every
# round has to carry one, and the last round's has to agree with the top-level
# figure the paper quotes.
# ---------------------------------------------------------------------------

PER_ROUND_CI_KEYS = ("vote_err_ci", "eps_bar_ci", "markov_bound_ci", "gap_ci")


def _improving_cube(n_tasks: int = 8, n_rounds: int = 4) -> np.ndarray:
    """A cube whose failures *shrink* over rounds, so the rounds are distinguishable.

    ``_cube`` holds the failure set fixed across rounds, which cannot catch a per-round
    statistic that silently reports one round's numbers for all of them.
    """
    k = 4
    cube = np.full((k, n_tasks, n_rounds), PASS)
    for t in range(n_rounds):
        # Round 0 fails the most tasks; each later round recovers one.
        failing = range(0, max(1, n_tasks - t * 2))
        for w in range(k):
            for i in failing:
                if (i + w) % 2 == 0:
                    cube[w, i, t] = FAIL
    return cube


def test_every_round_carries_all_four_intervals() -> None:
    """The figure whiskers three series at every round and the appendix quotes the
    per-round gap interval, so a missing key on any round is a broken figure."""
    res = eo.failmult_overlap(_improving_cube(), tau=0.6, n_resamples=299, seed=0)
    assert len(res["per_round"]) == 4
    for entry in res["per_round"]:
        for key in PER_ROUND_CI_KEYS:
            lo, hi = entry[key]
            assert lo <= hi, f"round {entry['round']} {key} is inverted"


def test_final_round_ci_matches_the_top_level_ci() -> None:
    """The invariant that pins ``_ci_block``'s shared index matrix.

    The top-level intervals *are* the final round's -- the paper quotes them as such.
    Drawing a fresh bootstrap per round would leave the two differing by resampling
    noise, which is exactly the bug this catches: the numbers would still look
    plausible, and Fig. 2's last-round whisker would silently stop matching the CI
    printed in the text beside it.
    """
    res = eo.failmult_overlap(_improving_cube(), tau=0.6, n_resamples=299, seed=11)
    last = res["per_round"][-1]
    for key in PER_ROUND_CI_KEYS:
        assert last[key] == res[key], f"final round's {key} diverged from the top level"
    assert last["vote_err"] == res["vote_err"]
    assert last["eps_bar"] == res["eps_bar"]
    assert last["gap"] == pytest.approx(res["gap"])


def test_per_round_intervals_bracket_their_own_estimates() -> None:
    """Each round's interval must bracket *that* round's point estimate -- not the
    final round's, which is what an off-by-one over the round axis would produce."""
    res = eo.failmult_overlap(_improving_cube(), tau=0.6, n_resamples=299, seed=5)
    for entry in res["per_round"]:
        assert entry["vote_err_ci"][0] <= entry["vote_err"] <= entry["vote_err_ci"][1]
        assert entry["eps_bar_ci"][0] <= entry["eps_bar"] <= entry["eps_bar_ci"][1]
        assert entry["gap_ci"][0] <= entry["gap"] <= entry["gap_ci"][1]


def test_per_round_markov_interval_is_exactly_twice_eps_bar_interval() -> None:
    """cor_overlap_markov_bound is 2*eps_bar, so its interval is a deterministic 2x --
    it carries no independent information, and the figure's docstring says so. If these
    ever came apart, one of the two was bootstrapped separately by mistake."""
    res = eo.failmult_overlap(_improving_cube(), tau=0.6, n_resamples=299, seed=2)
    for entry in res["per_round"]:
        assert entry["markov_bound_ci"] == pytest.approx(tuple(2 * x for x in entry["eps_bar_ci"]))


def test_per_round_intervals_are_deterministic_under_fixed_seed() -> None:
    """Same guarantee as test_failmult_ci_is_deterministic_under_fixed_seed, extended
    over the round axis: the figure is committed to the repo, so a re-run that moved
    the whiskers would produce a spurious diff on every regeneration."""
    cube = _improving_cube()
    a = eo.failmult_overlap(cube, tau=0.6, n_resamples=299, seed=7)
    b = eo.failmult_overlap(cube, tau=0.6, n_resamples=299, seed=7)
    for ea, eb in zip(a["per_round"], b["per_round"], strict=True):
        for key in PER_ROUND_CI_KEYS:
            assert ea[key] == eb[key]


# ---------------------------------------------------------------------------
# edge_gamma_with_ci: the round-0 boundary and the clustering unit
# ---------------------------------------------------------------------------


def _improving(n_tasks: int, n_seeds: int, n_rounds: int, frac: float) -> np.ndarray:
    """(n_seeds*n_tasks, n_rounds) quality matrix in which the first ``frac`` fraction
    of rows improves every round and the rest never do.

    Deliberately says nothing about task/seed grouping: which rows belong to which task
    is irrelevant to the point estimates these fixtures assert, so the fixture is
    layout-agnostic on purpose. The real files' layout is pinned separately by
    ``test_real_arm_files_are_seed_major``, because the clustering -- and only the
    clustering -- depends on it."""
    rows = n_seeds * n_tasks
    q = np.zeros((rows, n_rounds))
    n_improving = int(round(frac * rows))
    q[:n_improving] = np.arange(n_rounds) * 0.1
    return q


def test_edge_gamma_round0_ci_is_exactly_zero() -> None:
    """The gotcha: gamma_0 is fixed to 0 because round 0 has no predecessor, so
    its CI must be the degenerate [0, 0]. Bootstrapping the raw formula would give
    [-0.5, -0.5] -- an interval that excludes the value the paper reports."""
    q = _improving(n_tasks=10, n_seeds=5, n_rounds=4, frac=0.5)
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=199, seed=0)
    assert res["gamma"][0] == 0.0
    assert res["ci_lo"][0] == 0.0
    assert res["ci_hi"][0] == 0.0


def test_edge_gamma_matches_compute_metrics_formula() -> None:
    """Same definition as llm_simulations.compute_metrics: the mean over all rows
    of 1[q_t > q_{t-1}], minus 1/2, with round 0 zeroed."""
    q = _improving(n_tasks=10, n_seeds=5, n_rounds=4, frac=0.5)
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=199, seed=0)
    assert res["gamma"][1] == pytest.approx(0.0)  # exactly half the rows improve
    assert res["gamma"][2] == pytest.approx(0.0)


def test_eta_is_measured_on_the_running_best_not_raw_quality() -> None:
    """eta must track the running best, so a dip between rounds contributes 0 rather
    than a negative gain. Raw-quality differencing would give -0.4 at round 2 and
    break the Thm.-refinement bound the paper cites eta for.
    """
    q = np.array([[0.2, 0.6, 0.2, 0.7]])
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    assert res["eta"] == pytest.approx([0.0, 0.4, 0.0, 0.1])


def test_eta_cumulative_sum_respects_the_refinement_ceiling() -> None:
    """sum_t eta_t = b_T - q_0 <= 1 - q_0 for ANY path (Thm. refinement). This is the
    property that makes eta the quantity the theorem bounds."""
    rng = np.random.default_rng(0)
    q = rng.random((20, 5))
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    running_best = np.maximum.accumulate(q, axis=1)
    expected = (running_best[:, -1] - q[:, 0]).mean()
    assert sum(res["eta"]) == pytest.approx(expected)
    assert sum(res["eta"]) <= (1.0 - q[:, 0]).mean() + 1e-12


def test_eta_round_zero_is_zero_by_construction() -> None:
    q = np.array([[0.9, 0.1, 0.1], [0.3, 0.4, 0.5]])
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    assert res["eta"][0] == 0.0


def test_eta_and_gamma_come_from_the_same_matrix() -> None:
    """The paper quotes eta and gamma side by side per tier, so they must be the
    same length and derived from one call -- not stitched from two sources, which is
    how the hand-maintained Sonnet eta series drifted from its arm."""
    q = _improving(n_tasks=8, n_seeds=5, n_rounds=4, frac=0.5)
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=99, seed=0)
    assert len(res["eta"]) == len(res["gamma"]) == 4


def test_edge_gamma_all_improving_is_plus_half() -> None:
    q = _improving(n_tasks=10, n_seeds=5, n_rounds=3, frac=1.0)
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=199, seed=0)
    assert res["gamma"][1] == pytest.approx(0.5)
    assert res["ci_lo"][1] == pytest.approx(0.5)  # no variation to resample


# ---------------------------------------------------------------------------
# The row-layout invariant the cluster bootstrap silently depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag", [tag for _m, _mo, tag in eo.ARMS])
def test_real_arm_files_are_seed_major(tag: str) -> None:
    """``edge_gamma_with_ci`` groups clusters by arithmetic on the row index, reshaping
    ``(n_seeds, n_tasks, n_rounds)``. That is only the task-carries-its-seeds grouping the
    paper's CIs claim if the real files are laid out seed-major: seed 0's k tasks, then
    seed 1's k tasks, so a task's replicates sit k rows apart.

    Worth pinning because the failure is invisible. ``gamma`` is a mean over every row,
    so it is order-invariant; a task-major file would leave every published point estimate
    untouched and only change the CI *width* -- and the module's docstrings asserted the
    wrong layout ("task-major") for some time while the code did the right thing, so
    nothing but this test stands between a regenerated data file and a wrong interval.
    """
    q, n_seeds = eo.load_pooled_quality(tag)
    ids = eo.load_instance_ids(tag)
    n_tasks = len(q) // n_seeds

    assert len(ids) == len(q) == n_tasks * n_seeds
    assert len(set(ids[:n_tasks])) == n_tasks, "first block must be one full task pool"
    assert ids == ids[:n_tasks] * n_seeds, "rows are not seed-major blocks"
    # The negative half, stated so the intent survives a future refactor: adjacent rows
    # are different tasks, not the same task's replicates.
    assert ids[0] != ids[1], "adjacent rows are the same task -- file is task-major"


@pytest.mark.parametrize("tag", [tag for _m, _mo, tag in eo.ARMS])
def test_padded_mask_matches_headroom_mask(tag: str) -> None:
    """The two definitions of "this cell is not a measurement" must agree exactly.

    ``headroom_edge_with_ci`` -- the estimator the paper reports -- excludes a transition
    when ``q_{t-1} >= Q_MAX``. ``padding_stats`` excludes it when round ``t`` was back-filled
    after the task's first pass, read from the experiment's own ``solved_rounds`` record.
    These coincide only because a row sits at ``Q_MAX`` from its solve round onward and
    never before or after; if that ever stopped holding -- a task solvable at a fractional
    quality, or a pad value other than 1.0 -- the paper's primary edge would silently start
    averaging over non-events again, with no symptom other than a shifted point estimate.

    Asserting equality both ways rather than one containment, since either direction failing
    is a different bug: extra padded cells mean the estimator still includes non-events,
    missing ones mean it discards real measurements.
    """
    q, _n_seeds = eo.load_pooled_quality(tag)
    solved = eo.load_solved_rounds(tag)
    n_rounds = q.shape[1]

    padded = eo.padded_mask(solved, n_rounds)
    at_ceiling = np.zeros_like(q, dtype=bool)
    at_ceiling[:, 1:] = q[:, :-1] >= eo.Q_MAX

    assert padded.shape == at_ceiling.shape
    np.testing.assert_array_equal(padded, at_ceiling)
    # Round 0 is always a real measurement: nothing precedes it to have solved.
    assert not padded[:, 0].any()
    # And the pad is the hardcoded ceiling value, never something lower.
    assert np.all(q[padded] == eo.Q_MAX)


def test_padding_stats_tie_split_is_exhaustive() -> None:
    """Total tie share must decompose into padded + genuine, per round.

    Guards the disclosure arithmetic the paper quotes: if these stopped summing, the
    "genuine no-op" figure supporting "refinement turns idempotent" would drift from the
    total without any test noticing.
    """
    # Two rows: one solves at round 1 (so rounds 2-3 are padded), one never solves and
    # genuinely ties throughout.
    q = np.array([[0.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    res = eo.padding_stats(q, [1, None])

    assert res["n_padded"] == 2  # row 0, rounds 2 and 3
    assert res["padded_share"] == pytest.approx(2 / 8)
    for t in range(1, 4):
        assert res["tie_total"][t] == pytest.approx(
            res["tie_genuine"][t] + (res["tie_total"][t] * (res["padded_share_of_tie"][t] or 0.0))
        )
    # Round 1: row 0 improved, row 1 tied -> half the cells tie, none of it padding.
    assert res["tie_total"][1] == pytest.approx(0.5)
    assert res["padded_share_of_tie"][1] == pytest.approx(0.0)
    # Round 2: both rows tie, one of the two is padding.
    assert res["tie_total"][2] == pytest.approx(1.0)
    assert res["padded_share_of_tie"][2] == pytest.approx(0.5)


def test_abstention_bound_rewards_abstaining_over_regressing() -> None:
    """``Z_t = W_0 + 2*sqrt(W_+ W_-)`` must stay below 1 when improvements outweigh
    regressions, and must equal ``W_0`` when nothing regresses.

    This is the whole point of the confidence-rated reading: a round that mostly does
    nothing still drives the bound down, where the binary edge charges it as failure. A
    sign error or a swapped weight here would silently restore the flat-at-1.0 artifact.
    """
    # Four measured rows: 1 improves, 3 tie, none regress -> Z = W_0 = 0.75.
    q = np.array([[0.0, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    res = eo.abstention_bound(q)
    assert res["w_minus"][1] == pytest.approx(0.0)
    assert res["z"][1] == pytest.approx(0.75)
    assert res["z"][1] < 1.0
    # Round 0 contributes no factor.
    assert res["z"][0] is None
    # Conditional on moving, every mover improved.
    assert res["edge_given_moved"][1] == pytest.approx(0.5)

    # A pure abstainer: Z = 1 exactly, so the bound stops improving but never worsens.
    flat = np.zeros((4, 2))
    assert eo.abstention_bound(flat)["z"][1] == pytest.approx(1.0)


def test_abstention_bound_excludes_padded_cells() -> None:
    """The bound must be computed on measured cells only -- a row already at ``Q_MAX``
    contributes no weight, or its back-filled tie would inflate ``W_0`` toward 1 and make
    the bound look vacuous for exactly the strong-tier arms where it matters most.
    """
    # Row 0 is at the ceiling from round 0, so its later rounds are non-events.
    q = np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 0.5]])
    res = eo.abstention_bound(q)
    assert res["n_measured"][1] == 1
    assert res["n_measured"][2] == 1
    assert res["w_plus"][1] == pytest.approx(1.0)  # the one measured row improved


def test_edge_gamma_resamples_tasks_not_rows() -> None:
    """The 5 seeds of a task are not independent, so the bootstrap must resample
    the 55 task clusters (carrying all seeds together), not the 275 rows. Build a
    matrix where every seed of a task agrees but tasks differ: row-resampling
    would give a visibly tighter CI than cluster-resampling on the same data.
    """
    n_tasks, n_seeds, n_rounds = 20, 5, 2
    rng = np.random.default_rng(0)
    improves = rng.random(n_tasks) < 0.5  # per-task, shared by all its seeds
    q = np.zeros((n_seeds * n_tasks, n_rounds))
    for s in range(n_seeds):
        for i in range(n_tasks):
            if improves[i]:
                q[s * n_tasks + i, 1] = 1.0
    clustered = eo.edge_gamma_with_ci(q, n_seeds=n_seeds, n_resamples=2999, seed=1)
    naive = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=2999, seed=1)
    cluster_width = clustered["ci_hi"][1] - clustered["ci_lo"][1]
    row_width = naive["ci_hi"][1] - naive["ci_lo"][1]
    assert cluster_width > row_width, (
        "cluster bootstrap must be wider than row bootstrap when seeds are "
        f"perfectly correlated within task (got {cluster_width:.4f} vs {row_width:.4f})"
    )


def test_edge_gamma_rejects_non_divisible_seed_count() -> None:
    q = _improving(n_tasks=10, n_seeds=5, n_rounds=3, frac=0.5)  # 50 rows
    with pytest.raises(ValueError, match="n_seeds"):
        eo.edge_gamma_with_ci(q, n_seeds=3, n_resamples=99, seed=0)


def test_edge_gamma_ci_is_deterministic_under_fixed_seed() -> None:
    q = _improving(n_tasks=20, n_seeds=5, n_rounds=4, frac=0.4)
    a = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=499, seed=11)
    b = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=499, seed=11)
    assert a["ci_lo"] == b["ci_lo"]
    assert a["ci_hi"] == b["ci_hi"]


# ---------------------------------------------------------------------------
# Decomposing gamma's complement: ties vs regressions, and the headroom edge
# ---------------------------------------------------------------------------


def test_a_frozen_never_regressing_matrix_scores_the_gamma_floor_on_ties_alone() -> None:
    """The reason the paper cannot read gamma < 0 as AdaBoost's "worse than chance".

    A refiner that has fully converged -- every round identical, never a regression --
    attains gamma = -1/2, the floor, purely because the strict `>` counts a tie as a
    non-improvement. If this ever failed, the sign of gamma would be carrying information
    about correctness that it does not in fact carry.
    """
    q = np.full((20, 4), 0.7)
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=99, seed=0)
    assert res["gamma"][1:] == pytest.approx([-0.5, -0.5, -0.5])
    assert res["tie"][1:] == pytest.approx([1.0, 1.0, 1.0])
    assert res["regressed"][1:] == pytest.approx([0.0, 0.0, 0.0])
    assert res["flat_fraction"] == pytest.approx(1.0)
    assert res["n_regressing_rows"] == 0


def test_a_regression_is_counted_as_a_regression_not_a_tie() -> None:
    """tie and regressed must partition gamma's complement, not overlap or double-count.
    Pooling them is what the old single-`>` reporting did, and it is why a converged
    refiner and a degrading one were indistinguishable in the reported number."""
    q = np.array([[0.8, 0.5, 0.5, 0.9]])
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    assert res["regressed"][1] == pytest.approx(1.0)  # 0.8 -> 0.5 is a dip
    assert res["tie"][1] == pytest.approx(0.0)
    assert res["tie"][2] == pytest.approx(1.0)  # 0.5 -> 0.5 is a tie
    assert res["regressed"][2] == pytest.approx(0.0)
    assert res["gamma"][3] == pytest.approx(0.5)  # 0.5 -> 0.9 improves
    assert res["n_regressing_rows"] == 1
    assert res["flat_fraction"] == pytest.approx(0.0)


def test_tie_regressed_and_improved_partition_every_round() -> None:
    """The three rates must sum to exactly 1 for every t >= 1 (round 0 is excluded: gamma_0
    is fixed to 0 rather than measured, so it is not a rate). A gap would mean some
    transition is being classified into none of the three buckets."""
    rng = np.random.default_rng(3)
    q = np.round(rng.random((40, 5)), 1)  # rounding manufactures genuine ties
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    for t in range(1, 5):
        improved = res["gamma"][t] + 0.5
        assert improved + res["tie"][t] + res["regressed"][t] == pytest.approx(1.0)


def test_headroom_edge_excludes_saturated_rows_from_its_denominator_only() -> None:
    """The point of the decomposition. A row pinned at Q_MAX cannot improve, so it belongs
    in gamma's denominator (it is a row that did not improve) but not in the headroom
    edge's (it is a row that could not). Here one row of two is pinned and the other
    improves every round: pooled gamma is 0, the headroom edge is +1/2.
    """
    q = np.array([[1.0, 1.0, 1.0], [0.1, 0.2, 0.3]])
    res = eo.edge_gamma_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    hr = eo.headroom_edge_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    assert res["gamma"][1] == pytest.approx(0.0)  # 1 of 2 rows improved
    assert hr["gamma_headroom"][1] == pytest.approx(0.5)  # 1 of 1 eligible row improved
    assert hr["n_headroom"][1] == 1
    assert hr["n_headroom"][2] == 1


def test_headroom_edge_is_undefined_rather_than_zero_when_nothing_can_improve() -> None:
    """The empty-denominator trap. With every row at Q_MAX there is no eligible population,
    and 0/0 must surface as undefined (None in the cache, so `null` in JSON) rather than a
    spurious -0.5 that would read as a measured collapse. Round 0 is likewise undefined,
    having no predecessor to have left headroom.
    """
    q = np.ones((10, 3))
    hr = eo.headroom_edge_with_ci(q, n_seeds=5, n_resamples=99, seed=0)
    assert hr["gamma_headroom"][0] is None
    assert hr["gamma_headroom"][1] is None
    assert hr["gamma_headroom"][2] is None
    assert hr["ci_lo"][1] is None and hr["ci_hi"][1] is None
    assert hr["n_headroom"][1] == 0


def test_headroom_edge_turns_on_q_max_not_on_the_solve_threshold() -> None:
    """A task above TAU=0.6 counts as solved yet still has room to 1.0, so it must stay in
    the headroom denominator. Conditioning on TAU instead of Q_MAX would drop exactly the
    rows where late-round improvement is still possible, which is the opposite of the
    estimator's purpose."""
    q = np.array([[0.7, 0.8]])  # solved at both rounds, but not saturated
    hr = eo.headroom_edge_with_ci(q, n_seeds=1, n_resamples=99, seed=0)
    assert eo.TAU < 0.7 < eo.Q_MAX
    assert hr["n_headroom"][1] == 1
    assert hr["gamma_headroom"][1] == pytest.approx(0.5)


def test_headroom_edge_ci_is_wider_than_the_pooled_ci_on_the_same_data() -> None:
    """Conditioning shrinks the denominator, so its interval must be the wider of the two.
    The paper quotes both, and quoting a conditional point estimate with an interval no
    wider than the pooled one would understate exactly the uncertainty conditioning adds.
    Same seed on both calls, so the comparison is across estimators, not across draws.
    """
    rng = np.random.default_rng(7)
    q = np.round(rng.random((100, 3)), 1)
    q[:70] = 1.0  # most rows saturated, so the eligible population is small
    res = eo.edge_gamma_with_ci(q, n_seeds=5, n_resamples=999, seed=5)
    hr = eo.headroom_edge_with_ci(q, n_seeds=5, n_resamples=999, seed=5)
    pooled_width = res["ci_hi"][1] - res["ci_lo"][1]
    headroom_width = hr["ci_hi"][1] - hr["ci_lo"][1]
    assert headroom_width > pooled_width


def test_headroom_edge_rejects_non_divisible_seed_count() -> None:
    q = _improving(n_tasks=10, n_seeds=5, n_rounds=3, frac=0.5)  # 50 rows
    with pytest.raises(ValueError, match="n_seeds"):
        eo.headroom_edge_with_ci(q, n_seeds=3, n_resamples=99, seed=0)


def test_headroom_edge_is_deterministic_under_fixed_seed() -> None:
    q = _improving(n_tasks=20, n_seeds=5, n_rounds=4, frac=0.4)
    a = eo.headroom_edge_with_ci(q, n_seeds=5, n_resamples=499, seed=11)
    b = eo.headroom_edge_with_ci(q, n_seeds=5, n_resamples=499, seed=11)
    assert a["gamma_headroom"] == b["gamma_headroom"]
    assert a["ci_lo"] == b["ci_lo"]
    assert a["ci_hi"] == b["ci_hi"]


# ---------------------------------------------------------------------------
# Loader invariant
# ---------------------------------------------------------------------------


def test_worker_cube_rejects_mismatched_task_pool(tmp_path, monkeypatch) -> None:
    """A silent task-pool mismatch across arms would misalign workers and corrupt
    every m_i, so the loader must refuse rather than proceed."""
    import json

    monkeypatch.setattr(eo, "DATADIR", str(tmp_path))
    for idx, (_model, _mode, tag) in enumerate(eo.ARMS):
        ids = [f"task-{i}" for i in range(3)]
        if idx == len(eo.ARMS) - 1:
            ids[0] = "task-rogue"  # last arm has a different pool
        payload = {
            "config": {"k": 3, "n_seeds": 1, "tau": 0.6},
            "instance_ids": ids,
            "quality_matrix": [[0.0, 1.0]] * 3,
        }
        (tmp_path / f"llm_results_{tag}.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task pool"):
        eo.load_worker_cube()


def test_worker_cube_uses_running_best_not_final_raw(tmp_path, monkeypatch) -> None:
    """Quality can dip between rounds (it does, for 1-7 rows per real arm), and the
    paper's solve criterion is running-best >= tau everywhere. A loader reading the
    final raw round would mark a dipped task as failed."""
    import json

    monkeypatch.setattr(eo, "DATADIR", str(tmp_path))
    for _model, _mode, tag in eo.ARMS:
        payload = {
            "config": {"k": 2, "n_seeds": 1, "tau": 0.6},
            "instance_ids": ["task-0", "task-1"],
            "quality_matrix": [[1.0, 0.0], [0.0, 0.0]],  # task-0 peaks then dips
        }
        (tmp_path / f"llm_results_{tag}.json").write_text(json.dumps(payload))
    cube, task_ids, labels = eo.load_worker_cube()
    assert cube.shape == (len(eo.ARMS), 2, 2)
    assert task_ids == ["task-0", "task-1"]
    assert len(labels) == len(eo.ARMS)
    # running-best keeps the round-0 peak at round 1
    assert cube[0, 0, 1] == pytest.approx(1.0)
    assert cube[0, 1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# arm_level_overlap: the coarser one-worker-per-arm robustness companion, which
# feeds the reported statPBOverlapArmMStar / statPBOverlapArmViolate macros.
# ---------------------------------------------------------------------------


def test_arm_level_overlap_rejects_indivisible_seed_count() -> None:
    """Silently truncating here would reshape workers into the wrong arms and
    report a statistic for a grouping that does not exist."""
    cube = _cube([set(), set(), set()], n_tasks=2)  # k=3
    with pytest.raises(ValueError, match="does not divide"):
        eo.arm_level_overlap(cube, n_seeds=2)


def test_arm_level_overlap_treats_a_seed_split_as_arm_failure() -> None:
    """The tie rule again, one level up: an arm whose seeds split 1-1 on a task has
    solved-fraction exactly 0.5, and ``<= 0.5`` must count that arm as failing.
    A `<` would flip this arm to solved and hide the violation entirely.

    Layout: 4 workers, 2 seeds -> 2 arms. Arm 0 = workers 0,1 (worker 1 fails
    task 0, so arm 0 splits 1-1 there). Arm 1 = workers 2,3, both solve everything.
    So m = [1, 0] over 2 arms, m* = 1, and 2*1 >= 2 makes task 0 a vote error.
    """
    cube = _cube([set(), {0}, set(), set()], n_tasks=2)
    res = eo.arm_level_overlap(cube, n_seeds=2, tau=0.6)
    assert res["k"] == 2, "k must be the number of arms, not the number of workers"
    assert res["half_k"] == 1.0
    assert res["m_star"] == 1
    assert res["n_violating"] == 1
    assert res["vote_err"] == pytest.approx(0.5)  # 1 of 2 tasks


def test_arm_level_overlap_needs_a_seed_majority_not_a_single_seed() -> None:
    """With 3 seeds an arm fails only when a majority of its seeds fail. One failing
    seed out of three is 2/3 solved, which is above the boundary, so the arm counts
    as solved and no violation is reported."""
    cube = _cube([{0}, set(), set()], n_tasks=2)  # 3 workers, 1 arm of 3 seeds
    res = eo.arm_level_overlap(cube, n_seeds=3, tau=0.6)
    assert res["k"] == 1
    assert res["m_star"] == 0
    assert res["n_violating"] == 0
    assert res["vote_err"] == 0.0


def test_arm_level_overlap_histogram_spans_zero_to_k() -> None:
    """m_histogram is indexed by "number of arms failing", so it must have k+1 bins
    and sum to the task count -- otherwise a downstream index is off by one."""
    cube = _cube([{0}, {0}, set(), set()], n_tasks=3)
    res = eo.arm_level_overlap(cube, n_seeds=2, tau=0.6)
    assert len(res["m_histogram"]) == res["k"] + 1
    assert sum(res["m_histogram"]) == 3


# ---------------------------------------------------------------------------
# tier_overlap: the same statistic within one capability tier. Exists because the
# k=30 pool mixes tiers, and the paper's directional gap reading must be scoped to
# that mixing rather than stated as a fact about voting.
# ---------------------------------------------------------------------------

BOOT = {"n_resamples": 199, "seed": 7}


def test_tier_overlap_splits_on_the_label_prefix_not_position() -> None:
    """Tiers must come from the label prefix, not from slicing the cube in half.

    The labels here interleave the tiers, so a positional split would put one
    worker of each tier in the wrong group and report a different m* for both.
    Worker layout: strong-* solve everything, weak-* fail task 0.
    """
    cube = _cube([set(), {0}, set(), {0}], n_tasks=2)
    labels = ["strong-blind-s0", "weak-blind-s0", "strong-blind-s1", "weak-blind-s1"]
    res = eo.tier_overlap(cube, labels, tau=0.6, **BOOT)
    assert set(res) == {"strong", "weak"}
    assert res["strong"]["k"] == 2
    assert res["strong"]["m_star"] == 0, "positional split would have leaked a weak worker in"
    assert res["weak"]["m_star"] == 2
    assert res["weak"]["n_violating"] == 1


def test_tier_overlap_rejects_label_count_mismatch() -> None:
    """A labels list shorter than the worker axis would silently group only a prefix
    of the cube, reporting a tier statistic over the wrong workers."""
    cube = _cube([set(), {0}, set()], n_tasks=2)
    with pytest.raises(ValueError, match="labels"):
        eo.tier_overlap(cube, ["strong-blind-s0", "weak-blind-s0"], tau=0.6, **BOOT)


def test_tier_overlap_preserves_m_star_at_the_ceiling() -> None:
    """The headline the paper rests on must survive tier restriction: a task that
    defeats every worker defeats every worker of each tier, so m* stays at k on
    both sides. This is the property that makes m*=k robust while the error
    comparison below is not."""
    cube = _cube([{0}, {0}, {0}, {0}], n_tasks=3)
    labels = ["strong-blind-s0", "strong-blind-s1", "weak-blind-s0", "weak-blind-s1"]
    res = eo.tier_overlap(cube, labels, tau=0.6, **BOOT)
    for tier in ("strong", "weak"):
        assert res[tier]["m_star"] == res[tier]["k"], f"{tier}: m* must stay at its ceiling"


def test_tier_overlap_can_reverse_the_pooled_gap_sign() -> None:
    """The reason this function exists, as a test that fails if the promise is removed.

    Mixing a strong tier with a weak one can make the pooled gap
    ``vote_err - eps_bar`` positive while *neither* tier's own gap is, because the
    weak half supplies majority-failure mass that eps_bar averages away. Layout:
    4 tasks; the 2 strong workers fail only task 0, the 2 weak workers fail tasks
    0, 1 and 2. Pooled (k=4): m = [4, 2, 2, 0], so tasks 0-2 are vote errors
    (2m >= k) -> vote_err = 3/4, while eps_bar = (1 + 3)/2 / 4 = 1/2 -> gap = +1/4.
    Within the weak tier (k=2): m = [2, 2, 2, 0] -> vote_err = 3/4 = eps_bar, gap 0.
    Within the strong tier: vote_err = eps_bar = 1/4, gap 0.
    """
    cube = _cube([{0}, {0}, {0, 1, 2}, {0, 1, 2}], n_tasks=4)
    labels = ["strong-blind-s0", "strong-blind-s1", "weak-blind-s0", "weak-blind-s1"]
    pooled = eo.failmult_overlap(cube, tau=0.6, **BOOT)
    res = eo.tier_overlap(cube, labels, tau=0.6, **BOOT)

    assert pooled["gap"] == pytest.approx(0.25), "pooled gap must be positive for this fixture"
    for tier in ("strong", "weak"):
        assert res[tier]["gap"] <= 0.0 or res[tier]["gap"] == pytest.approx(0.0), (
            f"{tier}: within-tier gap must not inherit the pooled sign"
        )
        assert res[tier]["gap"] < pooled["gap"]
