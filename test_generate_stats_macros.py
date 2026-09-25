"""Tests for scripts/generate_stats_macros.py.

This script is the only thing standing between the JSON caches and the numbers the
paper prints. It rewrites `\\newcommand` values in `stats_macros.tex`, so a bug here
does not crash -- it silently publishes a wrong number, or silently *fails* to update
one, leaving a stale hand-transcribed value that no longer tracks the data. That
failure mode has already happened once in this repo (a stale `statPCPermCI` survived a
full re-derivation), which is why the script exists at all.

So the tests here target three things:

1. **The formatters**, since they encode LaTeX-specific escaping decisions (`{,}` for
   thousands, braced signs so TeX does not typeset a leading `-` as a binary operator)
   that are easy to "simplify" into subtly wrong output.
2. **`patch_macros`' byte-preservation contract**, especially that a value containing
   LaTeX backslashes survives. `re.sub` reads `\\1`-style sequences in a *string*
   replacement as group references, so the substitution must go through a callable --
   this is a regression guard on exactly that.
3. **`compute_macros` against the committed caches**, which is the real integration
   path, plus its two fail-loud branches. It must refuse to run on a params file that
   predates part of the schema rather than emit a half-updated macro set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.generate_stats_macros as gsm

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_fmt_p_switches_to_a_bound_below_the_reporting_floor() -> None:
    """Below 0.0001 the paper prints a bound, not a rounded zero -- "p = 0.0000"
    would claim more precision than the estimate supports."""
    assert gsm._fmt_p(0.00001) == "{<}0.0001"
    assert gsm._fmt_p(0.0001) == "0.0001"
    assert gsm._fmt_p(0.03) == "0.03"


def test_fmt_ci_honours_the_digit_count() -> None:
    assert gsm._fmt_ci(0.1234, 0.5678) == "$[0.123, \\, 0.568]$"
    assert gsm._fmt_ci(0.1234, 0.5678, digits=2) == "$[0.12, \\, 0.57]$"


def test_fmt_ci_pair_unpacks_a_two_element_sequence() -> None:
    """The JSON caches store CIs as [lo, hi] lists; this is the checkable spelling of
    `_fmt_ci(*ci, digits=2)`."""
    assert gsm._fmt_ci_pair([0.1234, 0.5678], digits=2) == gsm._fmt_ci(0.1234, 0.5678, digits=2)


def test_fmt_ci_pair_rejects_a_sequence_that_is_not_a_pair() -> None:
    """Loud failure beats a silently dropped bound."""
    with pytest.raises(ValueError):
        gsm._fmt_ci_pair([0.1, 0.2, 0.3])


def test_fmt_signed_braces_the_sign() -> None:
    """`{-}0.286`, not `-0.286`: unbraced, TeX sets the sign as a binary operator and
    spaces it wrongly. A zero must read as positive, not carry a stray minus."""
    assert gsm._fmt_signed(-0.286) == "{-}0.286"
    assert gsm._fmt_signed(0.062) == "{+}0.062"
    assert gsm._fmt_signed(0.0) == "{+}0.000"
    assert gsm._fmt_signed(-0.2857, digits=2) == "{-}0.29"


def test_fmt_signed_ci_braces_both_ends() -> None:
    assert gsm._fmt_signed_ci(-0.6, -0.2) == "$[{-}0.600, \\, {-}0.200]$"


def test_num_word_spells_counts_the_prose_spells() -> None:
    """The sentence reads "five distinct Opus versions", so the generated value has to
    be a word. Outside the covered range it falls back to digits rather than guessing
    a spelling -- wrong-but-obvious beats wrong-and-plausible in a paper."""
    assert gsm._num_word(5) == "five"
    assert gsm._num_word(8) == "eight"
    assert gsm._num_word(42) == "42"


def test_fmt_version_drops_a_trailing_zero() -> None:
    """Version 5.0 is written "5" in the prose; 4.5 keeps its decimal."""
    assert gsm._fmt_version(5.0) == "5"
    assert gsm._fmt_version(4.5) == "4.5"


def _inv(opus: list[float], sonnet: list[float], n_opus: int = 1, n_sonnet: int = 1) -> dict:
    return {
        "opus": {"n_trailer_strings": n_opus, "versions": opus},
        "sonnet": {"n_trailer_strings": n_sonnet, "versions": sonnet},
    }


def test_version_macros_emit_both_counts_per_family() -> None:
    m = gsm._version_macros(_inv([4.5, 4.6, 5.0], [4.5, 5.0], n_opus=8, n_sonnet=4))
    assert m["statPCOpusVersions"] == "three"
    assert m["statPCOpusTrailers"] == "eight"
    assert m["statPCSonnetVersions"] == "two"
    assert m["statPCSonnetTrailers"] == "four"


def test_version_span_claims_both_families_only_when_they_coincide() -> None:
    """NEGATIVE TEST: "in both families" is a claim about the data, not boilerplate. If
    one family gains a version the other lacks, the phrase must drop out rather than
    silently become false -- the failure mode a hand-written span would have."""
    shared = gsm._version_macros(_inv([4.5, 4.6, 5.0], [4.5, 4.6, 5.0]))
    assert shared["statPCVersionSpan"] == "generations~4.5 through~5 in both families"

    diverged = gsm._version_macros(_inv([4.5, 4.6, 5.0], [4.6, 4.8]))
    assert diverged["statPCVersionSpan"] == "generations~4.5 through~5"
    assert "both families" not in diverged["statPCVersionSpan"]


def test_fmt_thousands_uses_a_tex_safe_separator() -> None:
    """`{,}` keeps TeX from treating the comma as punctuation with trailing space."""
    assert gsm._fmt_thousands(30000) == "30{,}000"
    assert gsm._fmt_thousands(999) == "999"
    assert gsm._fmt_thousands(1234567) == "1{,}234{,}567"


def _round(t: int, lo: float, hi: float) -> dict:
    return {"round": t, "gap": (lo + hi) / 2, "gap_ci": (lo, hi)}


def test_gap_ci_boundary_returns_the_last_round_before_zero_is_straddled() -> None:
    """app:orchestration says the paired gap "excludes zero at every round through N".
    N is derived, so it must be the last round of the *initial run* that excludes zero --
    the boundary, not merely the last excluding round anywhere in the series."""
    per_round = [
        _round(0, 0.05, 0.18),
        _round(1, 0.04, 0.18),
        _round(2, 0.01, 0.17),
        _round(3, -0.02, 0.14),
    ]
    assert gsm._gap_ci_boundary(per_round)["round"] == 2


def test_gap_ci_boundary_stops_at_the_first_straddle_not_the_last() -> None:
    """A later round that happens to exclude zero again must not extend the range: the
    prose claims *every* round through N excludes it, so the run has to be contiguous.
    Returning round 3 here would make a true-sounding sentence false at round 1."""
    per_round = [
        _round(0, 0.05, 0.18),
        _round(1, -0.02, 0.14),
        _round(2, 0.01, 0.17),
        _round(3, 0.02, 0.19),
    ]
    assert gsm._gap_ci_boundary(per_round)["round"] == 0


def test_gap_ci_boundary_raises_when_round_zero_already_straddles_zero() -> None:
    """The negative case, and the reason this is a raise rather than a fallback: with no
    excluding round there is no range to describe, and emitting round "-1" (or silently
    reporting round 0) would put a false claim in the appendix. Failing the generator
    forces the sentence to be rewritten instead."""
    per_round = [_round(0, -0.01, 0.14), _round(1, 0.02, 0.19)]
    with pytest.raises(ValueError, match="includes zero at round 0"):
        gsm._gap_ci_boundary(per_round)


def test_gap_ci_boundary_treats_a_zero_endpoint_as_straddling() -> None:
    """An interval touching zero does not exclude it; a strict `<` here would claim
    significance for a boundary case."""
    per_round = [_round(0, 0.05, 0.18), _round(1, 0.0, 0.17), _round(2, 0.02, 0.19)]
    assert gsm._gap_ci_boundary(per_round)["round"] == 0


# ---------------------------------------------------------------------------
# patch_macros
# ---------------------------------------------------------------------------


def _macros_file(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "stats_macros.tex"
    p.write_text(body)
    return p


def test_patch_macros_reports_only_changed_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = _macros_file(
        tmp_path,
        "\\newcommand{\\statA}{old}\n\\newcommand{\\statB}{same}\n",
    )
    monkeypatch.setattr(gsm, "MACROS_PATH", str(p))
    changes = gsm.patch_macros({"statA": "new", "statB": "same"})
    assert changes == [("statA", "old", "new")]
    assert p.read_text() == "\\newcommand{\\statA}{new}\n\\newcommand{\\statB}{same}\n"


def test_patch_macros_preserves_latex_backslashes_in_the_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression guard for the substitution callable.

    Values legitimately contain LaTeX: `$[0.12, \\, 0.57]$`, `30{,}000`, `12.3\\%`.
    A string replacement would have `re.sub` interpret `\\,` and `\\%` as group
    references or escapes and corrupt the output. Written as a raw expectation so the
    assertion states the exact bytes the file must end up with.
    """
    p = _macros_file(tmp_path, "\\newcommand{\\statCI}{placeholder}\n")
    monkeypatch.setattr(gsm, "MACROS_PATH", str(p))
    value = "$[0.12, \\, 0.57]$ 30{,}000 12.3\\%"
    gsm.patch_macros({"statCI": value})
    assert p.read_text() == "\\newcommand{\\statCI}{" + value + "}\n"


def test_patch_macros_check_mode_does_not_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--check` is what CI and the pre-publish drift check use; if it wrote, running
    the check would itself destroy the evidence of drift."""
    original = "\\newcommand{\\statA}{old}\n"
    p = _macros_file(tmp_path, original)
    monkeypatch.setattr(gsm, "MACROS_PATH", str(p))
    changes = gsm.patch_macros({"statA": "new"}, check=True)
    assert changes == [("statA", "old", "new")]
    assert p.read_text() == original, "check mode must leave the file untouched"


def test_patch_macros_raises_on_a_macro_missing_from_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failing loud is the point: a macro the script computes but cannot place would
    otherwise stay hand-transcribed forever, silently diverging from the data."""
    p = _macros_file(tmp_path, "\\newcommand{\\statA}{old}\n")
    monkeypatch.setattr(gsm, "MACROS_PATH", str(p))
    with pytest.raises(ValueError, match=r"macro \\statMissing not found"):
        gsm.patch_macros({"statMissing": "x"})


def test_patch_macros_leaves_unrelated_lines_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Most of stats_macros.tex is still hand-maintained, so the rewrite must be
    surgical: comments, blank lines, and untouched macros survive exactly."""
    body = (
        "% header comment\n"
        "\\newcommand{\\handMade}{do not touch}\n"
        "\n"
        "\\newcommand{\\statA}{old}\n"
        "% trailing comment\n"
    )
    p = _macros_file(tmp_path, body)
    monkeypatch.setattr(gsm, "MACROS_PATH", str(p))
    gsm.patch_macros({"statA": "new"})
    assert p.read_text() == body.replace("{old}", "{new}")


# ---------------------------------------------------------------------------
# compute_macros: fail-loud branches
# ---------------------------------------------------------------------------


def test_compute_macros_names_the_command_that_produces_a_missing_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An actionable error, not a bare FileNotFoundError: the message must name the
    script that regenerates the cache."""
    monkeypatch.setattr(gsm, "DATADIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="company_simulations.py"):
        gsm.compute_macros()


def test_compute_macros_rejects_a_params_file_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A params file that predates (or post-processes away) part of the schema must
    fail loudly. Silently skipping would strand the Part C Finding-1 macros as stale
    hand-typed values -- the exact failure this script was written to end."""
    monkeypatch.setattr(gsm, "DATADIR", str(tmp_path))
    (tmp_path / "company_params.json").write_text(json.dumps({"decay_pct": 1.0}))
    (tmp_path / "part_b_k55_anova_power.json").write_text("{}")
    (tmp_path / "part_b_edge_overlap.json").write_text("{}")
    # Every cache compute_macros checks must exist, or its FileNotFoundError fires first
    # and masks the ValueError this test is about.
    (tmp_path / "part_b_k55_paper_numbers.json").write_text("{}")
    with pytest.raises(ValueError, match="decay_stats"):
        gsm.compute_macros()


# ---------------------------------------------------------------------------
# compute_macros: the real integration path, against the committed caches
# ---------------------------------------------------------------------------

_REQUIRED_CACHES = (
    "company_params.json",
    "part_b_k55_anova_power.json",
    "part_b_edge_overlap.json",
    "part_b_k55_paper_numbers.json",
    "part_c_control_stats.json",
)


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(gsm.DATADIR, n)) for n in _REQUIRED_CACHES),
    reason="committed JSON caches not present in this checkout",
)
def test_compute_macros_against_committed_caches_matches_the_checked_in_macros() -> None:
    """End-to-end on the real data, and the strongest assertion available: every macro
    the script computes must already equal what `stats_macros.tex` holds.

    A non-empty diff means the committed paper macros no longer match the committed
    caches -- i.e. the paper is printing numbers its own data no longer supports. This
    is the same invariant `--check` enforces, pinned as a test so it cannot be skipped.
    """
    macros = gsm.compute_macros()
    assert macros, "compute_macros returned nothing"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in macros.items())
    drift = gsm.patch_macros(macros, check=True)
    assert drift == [], f"stats_macros.tex has drifted from the caches: {drift}"


@pytest.mark.skipif(
    not all(os.path.exists(os.path.join(gsm.DATADIR, n)) for n in _REQUIRED_CACHES),
    reason="committed JSON caches not present in this checkout",
)
def test_computed_macro_values_are_latex_safe() -> None:
    """Every value lands inside `\\newcommand{...}{HERE}`, so an unescaped `%` would
    comment out the rest of the line and a newline would break the macro entirely."""
    for name, value in gsm.compute_macros().items():
        assert "\n" not in value, f"{name} contains a newline"
        assert value.count("{") == value.count("}"), f"{name} has unbalanced braces"
        for i, ch in enumerate(value):
            if ch == "%":
                assert i > 0 and value[i - 1] == "\\", f"{name} has an unescaped %: {value!r}"
