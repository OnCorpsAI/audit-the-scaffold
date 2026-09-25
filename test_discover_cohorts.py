"""Tests for the pure helpers extracted from ``scripts/discover_cohorts.main``.

``main`` was CC 31 (radon rank E) and had no tests. It cannot be run from this repo at
all -- it needs a ``--root`` directory of repository clones whose paths were scrubbed
before publication -- so the refactor that brought it under the complexity ceiling
moved every decision out of the orchestration spine into pure functions, and those are
what is tested here. ``main`` itself keeps only argparse, the thread pools, and the
prints, matching the convention already declared in ``[tool.coverage.report]`` that CLI
entrypoints are not coverage-gated.

The module is not in the ``--cov=`` list; see the note in ``[tool.pytest.ini_options]``.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scripts.discover_cohorts as dc  # noqa: E402


def _scan(name: str, *, ai: int = 0, trailer: str = "", error: str = "") -> dc.RepoScan:
    return dc.RepoScan(
        path=f"/clones/{name}",
        name=name,
        total_commits=100,
        ai_commits=ai,
        earliest_ai_trailer=trailer,
        error=error,
    )


# --------------------------------------------------------------------------
# _count_summary -- the empty-list guard that used to be six inline ternaries
# --------------------------------------------------------------------------


def test_count_summary_returns_min_median_max() -> None:
    assert dc._count_summary([3, 1, 2, 5, 4]) == (1, 3, 5)


def test_count_summary_of_empty_is_all_zeroes() -> None:
    """The guard exists so an empty cohort prints 0/0/0 instead of raising IndexError."""
    assert dc._count_summary([]) == (0, 0, 0)


def test_count_summary_median_uses_upper_middle_for_even_length() -> None:
    """Pins the original ``counts[len(counts) // 2]`` indexing, not a true mean-median.

    For [1, 2, 3, 4] that is index 2 -> 3, not 2.5. Preserved deliberately: the
    published cohort summary was produced with this definition.
    """
    assert dc._count_summary([1, 2, 3, 4]) == (1, 3, 4)


def test_count_summary_does_not_require_sorted_input() -> None:
    assert dc._count_summary([9, 1, 5]) == dc._count_summary([1, 5, 9])


def test_count_summary_single_element() -> None:
    assert dc._count_summary([7]) == (7, 7, 7)


# --------------------------------------------------------------------------
# _derive_window -- cutoff and control window from the earliest AI trailer
# --------------------------------------------------------------------------


def test_derive_window_subtracts_margin_then_window() -> None:
    scans = [_scan("a", trailer="2024-06-01"), _scan("b", trailer="2025-01-01")]
    earliest, cutoff, since = dc._derive_window(scans, margin_years=3, window_days=315)
    assert earliest == "2024-06-01"
    assert cutoff == date(2024, 6, 1) - dc.timedelta(days=365 * 3)
    assert since == cutoff - dc.timedelta(days=315)


def test_derive_window_picks_the_earliest_trailer_anywhere() -> None:
    scans = [
        _scan("late", trailer="2025-12-31"),
        _scan("early", trailer="2023-02-14"),
        _scan("mid", trailer="2024-07-01"),
    ]
    earliest, _cutoff, _since = dc._derive_window(scans, margin_years=1, window_days=10)
    assert earliest == "2023-02-14"


def test_derive_window_ignores_repos_with_no_trailer() -> None:
    scans = [_scan("none"), _scan("has", trailer="2024-03-03")]
    earliest, _cutoff, _since = dc._derive_window(scans, margin_years=1, window_days=1)
    assert earliest == "2024-03-03"


def test_derive_window_exits_when_no_trailer_exists_anywhere() -> None:
    """Without a trailer there is no cutoff to derive, so this must not fall through.

    Falling through would silently produce a control window anchored on nothing.
    """
    with pytest.raises(SystemExit) as excinfo:
        dc._derive_window([_scan("a"), _scan("b")], margin_years=3, window_days=315)
    assert excinfo.value.code == 1


def test_derive_window_zero_margin_puts_cutoff_on_the_trailer_date() -> None:
    _earliest, cutoff, _since = dc._derive_window(
        [_scan("a", trailer="2024-06-01")], margin_years=0, window_days=0
    )
    assert cutoff == date(2024, 6, 1)


# --------------------------------------------------------------------------
# _split_cohorts -- AI tier vs control candidates
# --------------------------------------------------------------------------


def test_split_cohorts_applies_the_ai_commit_floor() -> None:
    scans = [_scan("big", ai=20), _scan("small", ai=3)]
    ai_tier, candidates = dc._split_cohorts(scans, min_ai_commits=15)
    assert [s.name for s in ai_tier] == ["big"]
    assert [s.name for s in candidates] == ["small"]


def test_split_cohorts_excludes_errored_repos_from_candidates() -> None:
    """An unreadable repo is not a control candidate -- it has no countable history."""
    scans = [_scan("ok", ai=0), _scan("broken", ai=0, error="not a git repo")]
    ai_tier, candidates = dc._split_cohorts(scans, min_ai_commits=15)
    assert ai_tier == []
    assert [s.name for s in candidates] == ["ok"]


def test_split_cohorts_never_puts_a_repo_in_both_cohorts() -> None:
    scans = [_scan("a", ai=99), _scan("b", ai=1)]
    ai_tier, candidates = dc._split_cohorts(scans, min_ai_commits=15)
    ai_paths = {s.path for s in ai_tier}
    assert ai_paths.isdisjoint({s.path for s in candidates})


def test_split_cohorts_errored_repo_still_joins_ai_tier_if_it_cleared_the_floor() -> None:
    """Pins existing behaviour: the AI-tier filter looks only at the commit count.

    A repo that errored partway can still carry enough counted Claude commits, and the
    original code applied no error filter on that side.
    """
    scans = [_scan("partial", ai=20, error="timeout")]
    ai_tier, candidates = dc._split_cohorts(scans, min_ai_commits=15)
    assert [s.name for s in ai_tier] == ["partial"]
    assert candidates == []
