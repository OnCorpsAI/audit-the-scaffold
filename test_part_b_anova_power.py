"""Tests for scripts/part_b_anova_power.py's bootstrap-CI addition.

The rest of the module (task loading, power analysis, CLI) is a one-off
analysis driver, not covered by tests; this file targets only the pure,
newly-added `model_effect_bootstrap_ci` function -- a percentile bootstrap CI
on the Model-effect partial-eta2 / Cohen's f reported in Table 2, addressing
the reviewer comment that only Finding 2's capability gap reports a CI while
Table 2's ANOVA effect sizes don't.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.part_b_anova_power as pb


def _make_task_cube(
    n_tasks: int,
    tier_a_mean: float,
    tier_b_mean: float,
    noise_sd: float,
    n_modes: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """(n_tasks, 2, n_modes) cube: tier A ~ N(tier_a_mean, noise_sd), tier B ~
    N(tier_b_mean, noise_sd), independent noise per task/mode cell."""
    rng = np.random.default_rng(seed)
    a = rng.normal(tier_a_mean, noise_sd, size=(n_tasks, n_modes))
    b = rng.normal(tier_b_mean, noise_sd, size=(n_tasks, n_modes))
    return np.stack([a, b], axis=1)


def test_model_effect_bootstrap_ci_returns_expected_keys() -> None:
    cube = _make_task_cube(55, 0.85, 0.45, 0.15, seed=1)
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    for key in (
        "observed_partial_eta2",
        "observed_cohen_f",
        "eta2_ci_low",
        "eta2_ci_high",
        "cohen_f_ci_low",
        "cohen_f_ci_high",
        "n_tasks",
    ):
        assert key in result
    assert result["n_tasks"] == 55


def test_model_effect_bootstrap_ci_matches_point_estimate() -> None:
    """The CI's point estimate matches two_way_rm_anova's A-effect on the same
    data -- the bootstrap supplies the interval, not a re-derived point estimate."""
    cube = _make_task_cube(55, 0.85, 0.45, 0.15, seed=1)
    observed = pb.two_way_rm_anova(cube)["A"]
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    assert result["observed_partial_eta2"] == pytest.approx(observed["partial_eta2"])
    assert result["observed_cohen_f"] == pytest.approx(observed["cohen_f"])


def test_model_effect_bootstrap_ci_large_separation_excludes_zero() -> None:
    """A clean tier separation (like the paper's 85-89% vs 44-49% solve rates)
    should give a partial-eta2 CI that excludes 0."""
    cube = _make_task_cube(55, 0.85, 0.45, 0.10, seed=2)
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    assert result["eta2_ci_low"] > 0.0, f"Expected CI to exclude 0, got {result}"


def test_model_effect_bootstrap_ci_no_separation_includes_near_zero() -> None:
    """NEGATIVE TEST: no tier difference should give a partial-eta2 CI whose
    lower bound sits near 0 (not a confidently-nonzero effect)."""
    cube = _make_task_cube(55, 0.6, 0.6, 0.15, seed=3)
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    assert result["eta2_ci_low"] < 0.05, f"Expected near-zero lower bound, got {result}"


def test_model_effect_bootstrap_ci_bounds_ordered() -> None:
    cube = _make_task_cube(55, 0.85, 0.45, 0.15, seed=1)
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    assert result["eta2_ci_low"] <= result["observed_partial_eta2"] <= result["eta2_ci_high"]
    assert result["cohen_f_ci_low"] <= result["observed_cohen_f"] <= result["cohen_f_ci_high"]


def test_model_effect_bootstrap_ci_degenerate_resamples_no_nan() -> None:
    """Most tasks here are exact constants (zero within-cell variance), so a sizeable
    fraction of bootstrap resamples draw only from those constant tasks, giving
    ss_a_ + ss_sa_ == 0 for that replicate. That must not let a NaN leak into the
    reported CI bounds via np.percentile."""
    constant_task = np.full((2, 3), 0.5)
    variable_task = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    cube = np.stack([constant_task, constant_task, constant_task, variable_task], axis=0)
    result = pb.model_effect_bootstrap_ci(cube, n_resamples=999, seed=42)
    for key in ("eta2_ci_low", "eta2_ci_high", "cohen_f_ci_low", "cohen_f_ci_high"):
        assert not np.isnan(result[key]), f"{key} is NaN: {result}"
