"""Tests for the advisory predicates in ``scripts/validate_diagnostic_28task.py``.

``main`` was CC 27 (radon rank D) with no tests, and it cannot be run from this repo:
it needs ``LITELLM_API_KEY``/``LITELLM_API_BASE`` behind the VPN plus a Modal sandbox.
Only the *reporting* half is exercisable without credentials -- and that is where the
complexity was, in the degenerate-score and solve-rate flags the script prints for a
human to read by eye.

Those flags are the reason the script exists (they are what says "the harness fix did
not hold"), so they are worth pinning even though the orchestration spine above them
stays untested. The module is not in the ``--cov=`` list; see the note in
``[tool.pytest.ini_options]``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scripts.validate_diagnostic_28task as v  # noqa: E402


def _recs(*rounds: tuple[int, float, bool]) -> list[dict]:
    return [
        {"round": r, "fraction_passing": fp, "quality": fp, "passed": passed}
        for r, fp, passed in rounds
    ]


# --------------------------------------------------------------------------
# _fp_sequences -- per-seed fraction_passing, ordered by round
# --------------------------------------------------------------------------


def test_fp_sequences_orders_by_round_not_insertion() -> None:
    by_instance = {"t1": [_recs((2, 0.9, True), (0, 0.1, False), (1, 0.5, False))]}
    assert v._fp_sequences(by_instance, "t1") == [[0.1, 0.5, 0.9]]


def test_fp_sequences_returns_one_list_per_seed() -> None:
    by_instance = {"t1": [_recs((0, 0.0, False)), _recs((0, 1.0, True))]}
    assert v._fp_sequences(by_instance, "t1") == [[0.0], [1.0]]


def test_fp_sequences_of_unknown_instance_is_empty() -> None:
    assert v._fp_sequences({}, "nope") == []


# --------------------------------------------------------------------------
# _is_degenerate -- the "harness scored everything flat" alarm
# --------------------------------------------------------------------------


def test_all_zero_across_rounds_and_seeds_is_degenerate() -> None:
    assert v._is_degenerate([[0.0, 0.0], [0.0, 0.0]]) is True


def test_all_one_across_rounds_and_seeds_is_degenerate() -> None:
    assert v._is_degenerate([[1.0, 1.0], [1.0, 1.0]]) is True


def test_any_variation_is_not_degenerate() -> None:
    assert v._is_degenerate([[0.0, 0.5], [1.0, 1.0]]) is False


def test_mixed_all_zero_and_all_one_seeds_is_not_degenerate() -> None:
    """Flat *within* each seed but differing between them is real signal, not a bug."""
    assert v._is_degenerate([[0.0, 0.0], [1.0, 1.0]]) is False


def test_single_observation_is_never_degenerate() -> None:
    """One value cannot evidence flatness, so the original required len(flat) > 1."""
    assert v._is_degenerate([[0.0]]) is False
    assert v._is_degenerate([[1.0]]) is False


def test_no_observations_is_not_degenerate() -> None:
    assert v._is_degenerate([]) is False


# --------------------------------------------------------------------------
# _solve_rate_flag -- implausible-extreme warnings
# --------------------------------------------------------------------------


def test_zero_solve_rate_warns() -> None:
    assert "WARN" in v._solve_rate_flag(0.0)
    assert "universally 0" in v._solve_rate_flag(0.0)


def test_perfect_solve_rate_warns() -> None:
    assert "WARN" in v._solve_rate_flag(1.0)
    assert "universally 1" in v._solve_rate_flag(1.0)


def test_intermediate_solve_rate_is_unflagged() -> None:
    for rate in (0.01, 0.25, 0.5, 0.87, 0.99):
        assert v._solve_rate_flag(rate) == ""
