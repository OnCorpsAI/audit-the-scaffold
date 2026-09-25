#!/usr/bin/env python3
"""Patch the stats_macros.tex macros that are traceable to code output -- the Part C
Finding-1 block, the Part B Model-effect block, and the Part B edge / edge-CI /
edge-decomposition / failMult-overlap blocks -- from the JSON caches written by
``company_simulations.py``, ``scripts/part_b_anova_power.py``, and
``scripts/part_b_edge_overlap.py``.

Does a targeted per-macro ``\\newcommand{...}{...}`` line replacement; every other
macro in stats_macros.tex (still hand-transcribed, per the file's own header) is
left untouched.

Usage:
    uv run python company_simulations.py
    uv run python scripts/part_b_anova_power.py --boot-reps 30000 \\
        --k-grid 55,75,100,150,200,300,400,430,500,600,700,750,800,900,1000
    uv run python scripts/part_b_edge_overlap.py
    uv run python scripts/generate_stats_macros.py [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(REPO, "data")
MACROS_PATH = os.path.join(REPO, "stats_macros.tex")


def _fmt_p(p: float) -> str:
    return "{<}0.0001" if p < 0.0001 else f"{p:.2g}"


# Resamples used by the permutation tests in company_simulations. A two-sided
# scipy permutation test cannot report a p below 2/(n+1), so a value at that floor
# means "no resample beat the observed statistic", not "p equals this".
PERM_RESAMPLES = 9999


def _fmt_perm_p(p: float, n_resamples: int = PERM_RESAMPLES) -> str:
    """Permutation p-value, written as a bound when it sits at the resolution floor.

    Printing the floor as an equality overstates the precision of a test that simply
    never saw a resample exceed the observation, and it reads inconsistently beside the
    ``{<}``-form p-values elsewhere in the paper.
    """
    floor = 2.0 / (n_resamples + 1)
    if p <= floor:
        return f"{{<}}{floor:.4f}"
    return f"{p:.4f}"


def _fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"$[{lo:.{digits}f}, \\, {hi:.{digits}f}]$"


def _fmt_ci_pair(ci: Sequence[float], digits: int = 3) -> str:
    """`_fmt_ci` for a [lo, hi] pair loaded straight out of JSON.

    Spelling the two ends out keeps the call sites checkable: `_fmt_ci(*ci, digits=2)`
    on a JSON-loaded list gives mypy no length bound, so it cannot rule out the
    unpacking also filling `digits`.
    """
    lo, hi = ci
    return _fmt_ci(lo, hi, digits=digits)


def _fmt_signed(x: float, digits: int = 3) -> str:
    """LaTeX-safe signed number: {-}0.286 / {+}0.062 (braces keep TeX from setting the
    sign as a binary operator with the wrong spacing)."""
    return f"{{{'+' if x >= 0 else '-'}}}{abs(x):.{digits}f}"


def _fmt_signed_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"$[{_fmt_signed(lo, digits)}, \\, {_fmt_signed(hi, digits)}]$"


def _gap_ci_boundary(per_round: Sequence[dict]) -> dict:
    """The last round for which *every* round from 0 up to it excludes zero.

    The appendix says the paired vote-vs-worker gap "excludes zero at every round
    through N", so N is derived here rather than written into the prose: it is the
    largest N with no zero-straddling interval at or below it, which makes the claim
    true by construction and lets the sentence move with the data instead of going
    quietly stale. Returns that round's entry (round index, gap, interval).

    Raises if round 0 already includes zero -- the prose would then be asserting an
    empty range, and failing loudly beats emitting a macro that reads as round "-1".
    """
    boundary = None
    for entry in per_round:
        lo, hi = entry["gap_ci"]
        if lo <= 0.0 <= hi:
            break
        boundary = entry
    if boundary is None:
        raise ValueError(
            "the paired gap's interval includes zero at round 0, so there is no run of "
            "early rounds excluding it -- app:orchestration's 'through round N' sentence "
            "no longer describes this data and needs rewriting, not a regenerated macro"
        )
    return boundary


def _fmt_thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


# The prose spells small counts as words ("five distinct Opus versions"), so the
# generated value has to be a word too or the sentence would read "5 distinct Opus
# versions" mid-paragraph. Only the range these counts can plausibly occupy is
# covered; anything larger falls back to digits rather than inventing a spelling.
_NUM_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def _num_word(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


def _fmt_version(v: float) -> str:
    """Model version as the prose writes it: 4.5 stays 4.5, 5.0 becomes 5."""
    return f"{v:g}"


def _fmt_eta_series(eta: Sequence[float], digits: int = 3) -> str:
    """Per-round improvement series as an inline-math arrow chain.

    Round 0 is dropped: ``eta_0`` is 0 by construction (no predecessor), exactly as
    the gamma series drops round 0 from its quoted CIs. The result is wrapped in
    ``$...$`` because ``\\to`` is a math-mode command -- an earlier hand-written
    version of these macros omitted the delimiters and carried a comment warning
    that it could not be cited as-is.
    """
    body = r" \to ".join(f"{x:.{digits}f}" for x in eta[1:])
    return f"${body}$"


def _fmt_pct_range(values: Sequence[float], digits: int = 0) -> str:
    """Min--max of a set of proportions as a LaTeX percent range: ``80--95\\%``.

    Collapses to a single figure when the ends round to the same value, so the prose never
    reads "0.4--0.4\\%".
    """
    lo, hi = min(values), max(values)
    lo_s, hi_s = f"{lo:.{digits}%}", f"{hi:.{digits}%}"
    body = lo_s.rstrip("%") if lo_s == hi_s else f"{lo_s.rstrip('%')}--{hi_s.rstrip('%')}"
    return f"{body}\\%"


def _fmt_rate_series(rates: Sequence[float], digits: int = 3) -> str:
    """Per-round rate series, rounds 1+: ``(0.887, 0.945, 0.949)``.

    Round 0 is dropped for the same reason the gamma and eta series drop it -- there is no
    predecessor, so no transition to classify.
    """
    return f"({', '.join(f'{x:.{digits}f}' for x in rates[1:])})"


def _fmt_gamma_series(gamma: Sequence[float], digits: int = 2) -> str:
    """Per-round edge series as the prose quotes it: ``(0, -0.39, -0.46, -0.45)``.

    Round 0 is kept and printed as a bare ``0`` rather than ``0.00``, because it is fixed
    by definition rather than measured and the prose leans on that ("round~0 carries no
    interval because gamma_0 is fixed to 0"). Plain ``-`` rather than ``{-}`` is correct
    here: every call site sets this inside ``$...$``, where TeX already spaces the sign.
    """
    body = ", ".join(f"{x:.{digits}f}" for x in gamma[1:])
    return f"(0, {body})"


def _fmt_headroom_series(gamma: Sequence[float | None], digits: int = 3) -> str:
    """Headroom-conditional edge series, rounds 1+. Undefined rounds print as ``n/a``.

    ``None`` means the round had no eligible rows at all, which is a different statement
    from a measured 0 and must not be printed as one (see ``headroom_edge_with_ci``).
    """
    parts = [
        "n/a" if g is None else f"{g:+.{digits}f}".replace("+", "{+}").replace("-", "{-}")
        for g in gamma[1:]
    ]
    return f"({', '.join(parts)})"


# Keys this script reads out of company_params.json. Checked up front so a
# params file that predates (or post-processes away) part of the schema fails
# with an actionable message rather than a bare KeyError deep in the macro
# dict -- and, more importantly, so it fails *at all*: a params file missing
# these keys silently strands the Part C Finding-1 macros as hand-transcribed
# values that no longer track the data they were derived from.
COMPANY_REQUIRED_KEYS = ("decay_stats", "decay_ci", "decay_pct")
# Feedback modes in the order the paper reports them. Mirrors
# part_b_paper_numbers.MODES_ORDERED; duplicated rather than imported because this module
# reads that script's *cache*, not the script, and importing it would pull in pandas and
# the ANOVA path for two dict lookups.
MODES_ORDERED = ("independent", "blind", "diagnostic")


def _fmt_rate_count(entry: dict) -> str:
    """``{"k": 16, "n": 55, "rate": 0.29...}`` -> ``16/55 (29\\%)``.

    Same shape the Part B solve rates are already quoted in, so the resolved-bar rates
    read identically to the tau=0.6 ones beside them and the only visible difference is
    the number.
    """
    return f"{entry['k']}/{entry['n']} ({entry['rate']:.0%})".replace("%", "\\%")


def compute_macros() -> dict[str, str]:
    """Load the three pipeline caches and derive the in-scope macro values.

    Raises FileNotFoundError (with a message pointing at the generating command)
    if any cache is missing, or ValueError if company_params.json is present but
    does not carry the keys this script reads (see COMPANY_REQUIRED_KEYS).
    """
    company_path = os.path.join(DATADIR, "company_params.json")
    part_b_path = os.path.join(DATADIR, "part_b_k55_anova_power.json")
    edge_overlap_path = os.path.join(DATADIR, "part_b_edge_overlap.json")
    if not os.path.exists(company_path):
        raise FileNotFoundError(
            f"Missing {company_path} -- run: uv run python company_simulations.py"
        )
    if not os.path.exists(part_b_path):
        raise FileNotFoundError(
            f"Missing {part_b_path} -- run: uv run python scripts/part_b_anova_power.py "
            "--boot-reps 30000 --k-grid 55,75,100,150,200,300,400,430,500,600,700,750,800,900,1000"
        )
    if not os.path.exists(edge_overlap_path):
        raise FileNotFoundError(
            f"Missing {edge_overlap_path} -- run: uv run python scripts/part_b_edge_overlap.py"
        )
    paper_numbers_path = os.path.join(DATADIR, "part_b_k55_paper_numbers.json")
    if not os.path.exists(paper_numbers_path):
        raise FileNotFoundError(
            f"Missing {paper_numbers_path} -- run: uv run python scripts/part_b_paper_numbers.py"
        )

    with open(company_path) as f:
        company = json.load(f)
    missing = [k for k in COMPANY_REQUIRED_KEYS if k not in company]
    if missing:
        raise ValueError(
            f"{company_path} is missing {', '.join(missing)}. "
            "company_simulations.py's main() emits all of "
            f"{', '.join(COMPANY_REQUIRED_KEYS)}, so this file was not written by it "
            "(or was edited afterwards). Regenerate with: "
            "uv run python company_simulations.py"
        )
    with open(part_b_path) as f:
        part_b = json.load(f)
    with open(edge_overlap_path) as f:
        edge_overlap = json.load(f)
    with open(paper_numbers_path) as f:
        paper_numbers = json.load(f)

    decay_stats = company["decay_stats"]
    decay_ci = company["decay_ci"]
    model_effect = part_b["full_pool"]["anova"]["A"]
    model_ci = part_b["full_pool"]["model_effect_ci"]
    edge = edge_overlap["edge_gamma"]
    overlap = edge_overlap["overlap"]
    arm_level = edge_overlap["overlap_arm_level"]
    overlap_resolved = edge_overlap["overlap_resolved"]
    resolved_rates = paper_numbers["solve_rate_by_arm_resolved"]
    gap_boundary = _gap_ci_boundary(overlap["per_round"])

    macros = {
        "statPCDecayPct": f"{company['decay_pct']:.1f}\\%",
        "statPCMedianRatio": f"{decay_stats['observed_stat']:.2f}",
        "statPCPermCI": _fmt_ci(decay_ci["ci_low"], decay_ci["ci_high"]),
        "statPCPermP": _fmt_p(decay_stats["p_value"]),
        "statPCPermN": str(decay_stats["n_sessions"]),
        "statPBModelF": f"{model_effect['F']:.2f}",
        "statPBModelEta": f"{model_effect['partial_eta2']:.3f}",
        # The same effect on the running-best DV, which removes the published DV's
        # solved/unsolved asymmetry (see part_b_anova_power.load_raw_by_arm).
        "statPBModelEtaRunningBest": (f"{part_b['running_best_dv']['A']['partial_eta2']:.3f}"),
        "statPBModelCohenF": f"{model_effect['cohen_f']:.2f}",
        "statPBModelCohenFCI": _fmt_ci(model_ci["cohen_f_ci_low"], model_ci["cohen_f_ci_high"]),
        # Part B edge CIs. Rounds 1+ only: round 0's gamma is fixed to 0 (no
        # predecessor), so its interval is the degenerate [0,0] and is not quoted.
        "statPBEdgeCIReps": _fmt_thousands(edge["haiku"]["n_resamples"]),
        # Per-round improvement eta_bar_t on the running best, same pooled
        # diagnostic-arm matrix as gamma_t above. Round 0 dropped (eta_0 = 0 by
        # construction). These were hand-maintained until they were brought under
        # the generator; the Sonnet series was stale by ~0.009 at round 1 and
        # matched no arm exactly, which is what being outside --check permits.
        "statPBHaikuEta": _fmt_eta_series(edge["haiku"]["eta"]),
        "statPBSonnetEta": _fmt_eta_series(edge["sonnet"]["eta"]),
        # The gamma point-estimate series and the pooled N. Hand-transcribed until now
        # while being cited in the main text, so they sat outside --check.
        "statPBEdgeN": str(edge["haiku"]["n_pooled"]),
        "statPBHaikuEdge": _fmt_gamma_series(edge["haiku"]["gamma"]),
        "statPBSonnetEdge": _fmt_gamma_series(edge["sonnet"]["gamma"]),
        # Splitting gamma's complement. Ranges span rounds 1+ and both tiers, so the
        # prose can cite one figure for "ties dominate, regressions are rare".
        "statPBTieRange": _fmt_pct_range(
            [x for tier in ("haiku", "sonnet") for x in edge[tier]["tie"][1:]]
        ),
        # Maximum, not a range: one Haiku round has exactly zero regressions, so a range
        # would read "0.0--2.9\%" and invite the question of why the floor is zero. The
        # claim the prose makes is an upper bound ("regressions are rare"), so quote that.
        "statPBRegressMax": (
            f"{max(x for tier in ('haiku', 'sonnet') for x in edge[tier]['regressed'][1:]):.1%}"
        ).replace("%", "\\%"),
        "statPBHaikuFlat": f"{edge['haiku']['flat_fraction']:.1%}".replace("%", "\\%"),
        "statPBSonnetFlat": f"{edge['sonnet']['flat_fraction']:.1%}".replace("%", "\\%"),
        # Part B failMult overlap.
        "statPBOverlapK": str(overlap["k"]),
        "statPBOverlapHalfK": f"{overlap['half_k']:g}",
        "statPBOverlapMStar": str(overlap["m_star"]),
        "statPBOverlapViolate": (
            f"{overlap['n_violating']}/{overlap['n_tasks']} ({overlap['vote_err']:.0%})".replace(
                "%", "\\%"
            )
        ),
        "statPBOverlapVoteErr": f"{overlap['vote_err']:.3f}",
        "statPBOverlapVoteErrCI": _fmt_ci(*overlap["vote_err_ci"]),
        "statPBOverlapEpsBar": f"{overlap['eps_bar']:.3f}",
        "statPBOverlapEpsBarCI": _fmt_ci(*overlap["eps_bar_ci"]),
        "statPBOverlapMarkov": f"{overlap['markov_bound']:.3f}",
        "statPBOverlapGap": _fmt_signed(overlap["gap"]),
        "statPBOverlapGapCI": _fmt_signed_ci(*overlap["gap_ci"]),
        "statPBOverlapGapPosPct": f"{overlap['gap_frac_positive']:.0%}".replace("%", "\\%"),
        # The gap's sign is round-dependent, so the interval above describes the final
        # round only. Derive the boundary rather than hardcoding a round index: if the
        # data moves, the prose moves with it instead of going quietly stale.
        "statPBOverlapGapLastExclRound": str(gap_boundary["round"]),
        "statPBOverlapGapLastExcl": _fmt_signed(gap_boundary["gap"]),
        "statPBOverlapGapLastExclCI": _fmt_signed_ci(*gap_boundary["gap_ci"]),
        "statPBOverlapArms": str(len(edge_overlap["config"]["arms"])),
        "statPBOverlapArmMStar": str(arm_level["m_star"]),
        "statPBOverlapArmViolate": f"{arm_level['n_violating']}/{overlap['n_tasks']}",
        # ---- The SWE-bench resolved bar (every test passing) ----
        # tau=0.6 is a *fractional* passing threshold, so the solve rates it produces are
        # not comparable to published SWE-bench resolve rates. These let the paper quote
        # both bars from one source instead of inviting the comparison silently.
        # One decimal, not %g: the threshold reads as a quality value next to tau=0.6, and
        # %g would render it as a bare "1".
        "statPBTauResolved": f"{paper_numbers['tau_resolved']:.1f}",
        "statPBHaikuResolvedRange": _fmt_pct_range(
            [resolved_rates[f"haiku_{m}"]["rate"] for m in MODES_ORDERED]
        ),
        "statPBSonnetResolvedRange": _fmt_pct_range(
            [resolved_rates[f"sonnet_{m}"]["rate"] for m in MODES_ORDERED]
        ),
        "statPBHaikuResolvedIndep": _fmt_rate_count(resolved_rates["haiku_independent"]),
        "statPBHaikuResolvedDiag": _fmt_rate_count(resolved_rates["haiku_diagnostic"]),
        "statPBSonnetResolvedDiag": _fmt_rate_count(resolved_rates["sonnet_diagnostic"]),
        # The overlap statistic at the same bar. Raising tau can only turn a solved task
        # into a failed one, so m* is monotone in tau and this is expected to match the
        # tau=0.6 value -- generated rather than asserted so the paper's claim that the
        # m*=k headline is threshold-independent rests on a number.
        "statPBOverlapMStarResolved": str(overlap_resolved["m_star"]),
        "statPBOverlapViolateResolved": (
            f"{overlap_resolved['n_violating']}/{overlap_resolved['n_tasks']}"
        ),
    }
    for tier in ("haiku", "sonnet"):
        macros.update(_part_b_tier_macros(tier, edge[tier]))

    macros.update(_part_b_overlap_tier_macros(edge_overlap["overlap_by_tier"]))
    macros.update(_part_c_macros(company))
    return macros


def _part_b_tier_macros(tier: str, stats: dict) -> dict[str, str]:
    """Every per-tier Part B edge macro, for one tier.

    Split out of ``compute_macros`` to keep that function under the complexity gate. Pure
    formatting over one tier's block of ``data/part_b_edge_overlap.json``.
    """
    prefix = f"statPB{tier.capitalize()}"
    hr, pad, abst = stats["headroom"], stats["padding"], stats["abstention"]
    n_hr = hr["n_headroom"]
    hlo, hhi = hr["ci_lo"], hr["ci_hi"]
    movers = [x for x in abst["edge_given_moved"][1:] if x is not None]

    return {
        # Rounds 1..T-1 as a comma-separated list of intervals.
        f"{prefix}EdgeCI": ", ".join(
            _fmt_signed_ci(stats["ci_lo"][t], stats["ci_hi"][t])
            for t in range(1, len(stats["gamma"]))
        ),
        # Edge on measured cells: series, intervals, and the shrinking denominator.
        # Undefined rounds are skipped in the interval list rather than printed as a
        # degenerate [0,0] -- there is no eligible population to resample.
        f"{prefix}EdgeHeadroom": _fmt_headroom_series(hr["gamma_headroom"]),
        f"{prefix}EdgeHeadroomCI": ", ".join(
            _fmt_signed_ci(hlo[t], hhi[t])
            for t in range(1, len(hr["gamma_headroom"]))
            if hlo[t] is not None and hhi[t] is not None
        ),
        f"{prefix}HeadroomN": f"{n_hr[1]}$\\to${n_hr[-1]} of {hr['n_pooled']}",
        f"{prefix}HeadroomNLast": str(n_hr[-1]),
        # The back-filling that makes the pooled gamma uninterpretable: its overall share of
        # the matrix, its share of each round's tie mass, and the tie rate net of it.
        # Generated rather than transcribed because the whole point of quoting them is that
        # the pooled and corrected estimators differ by exactly this much.
        f"{prefix}PadShare": f"{pad['padded_share']:.1%}".replace("%", "\\%"),
        f"{prefix}PadTieRange": _fmt_pct_range(
            [x for x in pad["padded_share_of_tie"][1:] if x is not None]
        ),
        f"{prefix}GenuineTieRange": _fmt_pct_range(pad["tie_genuine"][1:]),
        # Confidence-rated (abstaining) bound on measured cells, and the edge conditional on
        # the round having changed anything.
        f"{prefix}AbstainZ": f"{abst['z_product']:.2f}",
        f"{prefix}MoversEdge": f"{_fmt_signed(min(movers))} to {_fmt_signed(max(movers))}",
        # Per-round tie/regression series and the count of rows that dip at least once. The
        # appendix quotes these individually rather than only as the pooled ranges, so they
        # are generated too -- transcribing a series is how the eta pair drifted.
        f"{prefix}TieSeries": _fmt_rate_series(stats["tie"]),
        f"{prefix}RegressSeries": _fmt_rate_series(stats["regressed"]),
        f"{prefix}RegressRows": str(stats["n_regressing_rows"]),
    }


def _part_b_overlap_tier_macros(by_tier: dict) -> dict[str, str]:
    """The overlap statistic restricted to one capability tier, for every tier.

    Generated rather than transcribed for the usual reason plus a specific one: these
    numbers exist to stop a *misreading* of the pooled gap, so a stale copy here would
    reintroduce exactly the error they were added to correct.

    ``statPBOverlapTierK`` is emitted once and asserted equal across tiers -- the
    appendix prose quotes a single k for both, which is only honest while the arms are
    balanced. An unbalanced sweep must fail here rather than silently print one tier's k
    for both.
    """
    ks = {res["k"] for res in by_tier.values()}
    if len(ks) != 1:
        raise ValueError(
            f"tiers have unequal worker counts {sorted(ks)}; statPBOverlapTierK assumes "
            "one k for both and the appendix prose quotes it once. Split the macro first."
        )

    macros = {"statPBOverlapTierK": str(ks.pop())}
    for tier, res in by_tier.items():
        prefix = f"statPBOverlap{tier.capitalize()}"
        macros[f"{prefix}MStar"] = str(res["m_star"])
        macros[f"{prefix}Violate"] = f"{res['n_violating']}/{res['n_tasks']}"
        macros[f"{prefix}VoteErr"] = f"{res['vote_err']:.3f}"
        macros[f"{prefix}EpsBar"] = f"{res['eps_bar']:.3f}"
        macros[f"{prefix}Gap"] = _fmt_signed(res["gap"])
        macros[f"{prefix}GapCI"] = _fmt_signed_ci(*res["gap_ci"])
    return macros


def _part_c_macros(company: dict) -> dict[str, str]:
    """Every remaining Part C macro, from company_params.json and the Part C
    control cache.

    These were hand-transcribed until now -- about 25 numbers copied by hand on
    every re-derivation of Part C. That is how a ``statPCPermCI`` from a superseded
    378-session sample survived into the paper: nothing recomputed it, so nothing
    could notice. Generating them makes ``--check`` a real gate over the whole Part
    C block rather than a fifth of it.
    """
    control_path = os.path.join(DATADIR, "part_c_control_stats.json")
    if not os.path.exists(control_path):
        raise FileNotFoundError(
            f"Missing {control_path} -- run: uv run python scripts/part_c_control_stats.py"
        )
    with open(control_path) as f:
        pc = json.load(f)

    gap = company["gap_stats"]
    ev = company["gap_evalue"]
    strat = company["stratified_gap"]
    decay = company["tier_decay"]
    mm = company["mixed_model"]
    oos = company["oos_stats"]
    fig3 = company["fig3_ks"]
    ctrl = pc["control"]
    comp = pc["composition"]
    rel = pc["release"]
    tiers = pc["tier_composition"]

    macros = {
        # Cohort scale. n_developers is the *cohort* count from the session data,
        # not the mixed model's filtered row count -- the two differ and the paper
        # previously quoted the filtered one as if it were the cohort.
        "statPCSessions": str(pc["ai_tier"]["n_sessions"]),
        "statPCCommits": _fmt_thousands(pc["ai_tier"]["n_commits"]),
        "statPCRepos": str(pc["ai_tier"]["n_repos"]),
        "statPCDevelopers": str(pc["ai_tier"]["n_developers"]),
        # Finding 2: the tier contrast.
        "statPCTauOpus": f"{company['tau_opus']:.2f}",
        "statPCTauSonnet": f"{company['tau_sonnet']:.2f}",
        "statPCNOpus": str(gap["n_opus"]),
        "statPCNSonnet": str(gap["n_sonnet"]),
        "statPCGap": _fmt_signed(gap["gap"]),
        "statPCGapCI": _fmt_signed_ci(*gap["ci_gap_95"]),
        "statPCGapPermP": f"{gap['p_value']:.2g}",
        "statPCGapMWUP": f"{gap['mwu_p']:.2g}",
        "statPCEvaluePoint": f"{ev['evalue_point']:.2f}",
        "statPCEvalueCI": f"{ev['evalue_ci']:.2f}",
        "statPCStratGap": _fmt_signed(strat["adjusted_gap"]),
        "statPCStratP": f"{strat['p_value']:.2g}",
        "statPCStrata": str(strat["n_strata_used"]),
        "statPCChurnRhoOpus": f"{decay['rho_opus']:.2f}",
        "statPCChurnRhoSonnet": f"{decay['rho_sonnet']:.2f}",
        "statPCChurnGap": f"{decay['rho_gap']:.3f}",
        "statPCChurnGapP": f"{decay['p_value']:.2g}",
        "statPCChurnGapN": str(decay["n_sessions"]),
        "statPCMultilevelCoef": _fmt_signed(mm["tier_coef"]),
        "statPCMultilevelCI": _fmt_signed_ci(*mm["tier_ci"]),
        "statPCMultilevelP": f"{mm['tier_p']:.2g}",
        "statPCVarRepo": f"{mm['vc_repo']:.3f}",
        "statPCVarDev": f"{mm['vc_developer']:.3f}",
        # Finding 3: the generative fit.
        "statPCRho": f"{company['rho']:.2f}",
        "statPCRhoCI": _fmt_ci_pair(company["rho_ci"], digits=2),
        "statPCSigma": f"{company['sigma']:.2f}",
        "statPCSigmaCI": _fmt_ci_pair(company["sigma_ci"], digits=2),
        "statPCPiPure": f"{company['pi_pure']:.3f}",
        "statPCTrainN": str(oos["n_train"]),
        "statPCTestN": str(oos["n_test"]),
        "statPCKSHeldout": f"{oos['ks_stat']:.2f}",
        "statPCKSHeldoutP": f"{oos['ks_p']:.3f}",
        "statPCKSInsample": f"{fig3['ks_stat']:.2f}",
        "statPCKSInsampleP": _fmt_p(fig3["ks_p"]),
        # Finding 4 / composition.
        "statPCSingleCommitPct": f"{comp['ai_tier']['single_commit_pct']:.1f}\\%",
        # Control cohort.
        "statPCCtrlSince": str(ctrl["since"]),
        "statPCCtrlUntil": str(ctrl["until"]),
        "statPCCtrlRepos": str(ctrl["n_repos"]),
        "statPCCtrlDevelopers": str(ctrl["n_developers"]),
        "statPCCtrlSessions": _fmt_thousands(ctrl["n_sessions"]),
        "statPCCtrlCommits": _fmt_thousands(ctrl["n_commits"]),
        "statPCCtrlTau": f"{ctrl['tau_human']:.2f}",
        "statPCCtrlRho": f"{ctrl['rho']:.2f}",
        "statPCCtrlRhoCI": _fmt_ci_pair(ctrl["rho_ci"], digits=2),
        # Control composition, so the singleton-heaviness objection is answerable
        # from generated numbers rather than a hand-computed aside.
        "statPCCtrlSingleCommitPct": f"{comp['control']['single_commit_pct']:.1f}\\%",
        "statPCCtrlTauMulti": f"{comp['control']['tau_multi']:.3f}",
        "statPCTauMulti": f"{comp['ai_tier']['tau_multi']:.3f}",
        # Release footprint. The Responsible Use Statement's privacy claim is about
        # everyone in the published datasets, which is strictly more people than the
        # control cohort -- so it gets its own macro rather than borrowing
        # statPCCtrlDevelopers, which counts only the pre-AI roster.
        "statPCReleasedDevelopers": str(rel["n_developers"]),
        "statPCReleasedDatasets": _num_word(rel["n_datasets"]),
        # Tier composition. Opus and Sonnet do not exhaust the cohort, so the residue
        # the contrast excludes is stated rather than left for a reader to infer from
        # the appendix's Haiku cell.
        "statPCHaikuSessions": str(tiers["by_tier"].get("haiku", 0)),
        "statPCUnresolvedSessions": str(tiers["by_tier"].get("unknown", 0)),
        "statPCTierExcluded": str(tiers["n_excluded"]),
        # Quoted where a permutation p sits at its resolution floor, so the floor the
        # prose names and the floor _fmt_perm_p compares against cannot drift apart.
        "statPCPermReps": _fmt_thousands(PERM_RESAMPLES),
    }
    for tier in ("opus", "sonnet"):
        g = pc[f"human_vs_{tier}"]
        suffix = tier.capitalize()
        macros[f"statPCCtrlGap{suffix}"] = _fmt_signed(g["gap"])
        macros[f"statPCCtrlGap{suffix}P"] = _fmt_perm_p(g["p_value"])
        macros[f"statPCCtrlML{suffix}Coef"] = _fmt_signed(g["mm_coef"])
        macros[f"statPCCtrlML{suffix}P"] = f"{g['mm_p']:.2g}"

    macros.update(_version_macros(pc["model_versions"]))
    macros.update(_within_era_macros(pc["within_era"]))
    macros.update(_paired_macros(pc["paired_within_developer"]))
    macros.update(_capability_macros(pc["capability"]))
    return macros


def _version_macros(inventory: dict) -> dict[str, str]:
    """Version heterogeneity behind the Opus and Sonnet labels.

    Both the trailer-string count and the released-version count are quoted in the
    prose as the reason Finding 2's tier contrast is coarse, so both are derived.
    Only the two families the tier contrast pools are emitted; Haiku sits outside
    both bins and the paper reports its single cell through the capability block.
    """
    macros: dict[str, str] = {}
    for family, suffix in (("opus", "Opus"), ("sonnet", "Sonnet")):
        entry = inventory[family]
        macros[f"statPC{suffix}Versions"] = _num_word(len(entry["versions"]))
        macros[f"statPC{suffix}Trailers"] = _num_word(entry["n_trailer_strings"])

    # "in both families" is a claim about the two spans coinciding, so it is asserted
    # from the data rather than written into the template: if a refresh gives one
    # family a version the other lacks, the phrase drops out instead of going false.
    spans = {
        f: (inventory[f]["versions"][0], inventory[f]["versions"][-1]) for f in ("opus", "sonnet")
    }
    lo, hi = min(s[0] for s in spans.values()), max(s[1] for s in spans.values())
    span = f"generations~{_fmt_version(lo)} through~{_fmt_version(hi)}"
    if all(s == (lo, hi) for s in spans.values()):
        span += " in both families"
    macros["statPCVersionSpan"] = span
    return macros


# floor_<n> key in the JSON -> macro infix. Spelled out rather than derived from the
# number so a floor label can never collide with a statistic name, and so changing
# the floor grid in part_c_control_stats.py surfaces as a missing-macro error here
# rather than as a table row silently retargeted at a different floor.
_FLOOR_INFIX = {"floor_0": "FZero", "floor_15": "FB", "floor_30": "FC", "floor_50": "FD"}


def _floor_row_macros(block: dict, prefix: str, floors: tuple[str, ...]) -> dict[str, str]:
    """One row of paired-comparison statistics per commit floor.

    ``wilcoxon_p`` is NaN wherever the floor leaves fewer than the six non-zero
    pairs the test needs, so it is emitted only where it is defined -- a NaN
    formatted into the paper would read as a computed value.
    """
    out: dict[str, str] = {}
    for key in floors:
        row, infix = block[key], _FLOOR_INFIX[key]
        out[f"{prefix}{infix}Pairs"] = str(row["n_pairs"])
        out[f"{prefix}{infix}Higher"] = str(row["n_b_greater"])
        out[f"{prefix}{infix}SignP"] = f"{row['sign_p']:.2g}"
        out[f"{prefix}{infix}Median"] = _fmt_signed(row["median_delta"])
        if row["wilcoxon_p"] == row["wilcoxon_p"]:  # not NaN
            out[f"{prefix}{infix}WilcoxP"] = f"{row['wilcoxon_p']:.2g}"
    return out


def _within_era_macros(we: dict) -> dict[str, str]:
    """The within-era AI-vs-non-AI contrast: same people, same window, split only on
    whether a commit carries a Claude trailer."""
    paired = we["paired"]
    top = paired["floor_0"]["top_volume_pair"]
    macros = {
        "statPCWithinSince": str(we["since"]),
        "statPCWithinUntil": str(we["until"]),
        "statPCWithinAISessions": _fmt_thousands(we["ai"]["n_sessions"]),
        "statPCWithinAICommits": _fmt_thousands(we["ai"]["n_commits"]),
        "statPCWithinNonAISessions": _fmt_thousands(we["nonai"]["n_sessions"]),
        "statPCWithinNonAICommits": _fmt_thousands(we["nonai"]["n_commits"]),
        "statPCWithinDevelopers": str(we["ai"]["n_developers"]),
        "statPCWithinTauAI": f"{we['ai']['tau']:.3f}",
        "statPCWithinTauNonAI": f"{we['nonai']['tau']:.3f}",
        "statPCWithinFloorB": str(paired["floor_15"]["min_commits_per_side"]),
        "statPCWithinFloorC": str(paired["floor_30"]["min_commits_per_side"]),
        "statPCWithinFloorD": str(paired["floor_50"]["min_commits_per_side"]),
        # The signed spread at the >=15 floor: what Wilcoxon reacts to and the sign
        # test does not.
        "statPCWithinFBDeltaMin": _fmt_signed(paired["floor_15"]["delta_min"]),
        "statPCWithinFBDeltaMax": _fmt_signed(paired["floor_15"]["delta_max"]),
        "statPCWithinTopVolNonAI": _fmt_thousands(top["n_commits_a"]),
        "statPCWithinTopVolAI": _fmt_thousands(top["n_commits_b"]),
        "statPCWithinTopVolDelta": _fmt_signed(top["delta"]),
    }
    macros.update(
        _floor_row_macros(paired, "statPCWithin", ("floor_0", "floor_15", "floor_30", "floor_50"))
    )
    return macros


def _paired_macros(paired: dict) -> dict[str, str]:
    """The pre-AI within-developer pairing. Reported at floor 0 and the >=50 floor
    only: it cuts against the between-group gap, and the >=50 row is where it does so
    most clearly, so the intermediate floors are not quoted."""
    top = paired["floor_0"]["top_volume_pair"]
    macros = {
        "statPCPairedFloorD": str(paired["floor_50"]["min_commits_per_side"]),
        "statPCPairedTopVolCtrl": _fmt_thousands(top["n_commits_a"]),
        "statPCPairedTopVolAI": _fmt_thousands(top["n_commits_b"]),
        "statPCPairedTopVolDelta": _fmt_signed(top["delta"]),
        "statPCPairedMaxRatio": f"{paired['floor_0']['max_volume_ratio']:.0f}",
    }
    rows = _floor_row_macros(paired, "statPCPaired", ("floor_0", "floor_50"))
    # Floor 0 is the headline row, so it carries no infix.
    for name, value in rows.items():
        macros[name.replace("statPCPairedFZero", "statPCPaired")] = value
    return macros


def _capability_macros(cap: dict) -> dict[str, str]:
    """Finding 2 re-tested as a monotone trend over an ordered capability scale,
    under both orderings. The adjusted CI is turned into the widest total swing it
    still permits across every rank step, which is what makes this a *bounded* null
    rather than a tight zero."""
    family_trend = cap["family"]["trend"]
    n_cells = family_trend["n_cells"]
    steps = n_cells - 1
    macros = {
        "statPCCapSessions": str(family_trend["n_sessions"]),
        "statPCCapCells": str(n_cells),
        "statPCCapSteps": str(steps),
        "statPCCapSwing": f"{steps * cap['family']['mixed']['cap_ci'][1]:.3f}",
    }
    for ordering, suffix in (("family", "Family"), ("version", "Version")):
        trend, mixed = cap[ordering]["trend"], cap[ordering]["mixed"]
        macros[f"statPCCap{suffix}Rho"] = _fmt_signed(trend["spearman_rho"])
        macros[f"statPCCap{suffix}P"] = f"{trend['spearman_p']:.2g}"
        macros[f"statPCCap{suffix}Coef"] = _fmt_signed(mixed["cap_coef"])
        macros[f"statPCCap{suffix}CoefP"] = f"{mixed['cap_p']:.2g}"
        macros[f"statPCCap{suffix}CI"] = _fmt_signed_ci(*mixed["cap_ci"])

    # The non-monotonicity that drives rho to zero, located structurally rather than
    # by hardcoding version numbers: Sonnet's newest cell against the best older one.
    sonnet = [c for c in cap["per_cell"] if c["family"] == "sonnet"]
    newest = max(sonnet, key=lambda c: c["version"])
    older_best = max(
        (c for c in sonnet if c["version"] < newest["version"]), key=lambda c: c["tau"]
    )
    haiku = max((c for c in cap["per_cell"] if c["family"] == "haiku"), key=lambda c: c["version"])
    macros["statPCCapTauSonnetMid"] = f"{older_best['tau']:.3f}"
    macros["statPCCapTauSonnetNew"] = f"{newest['tau']:.3f}"
    macros["statPCCapTauHaiku"] = f"{haiku['tau']:.3f}"
    macros["statPCCapNHaiku"] = str(haiku["n_sessions"])

    # Cell composition. The prose has to state this because the cell total coincides
    # with the pooled-version total it is quoted next to (five Opus + three Sonnet
    # versions, eight cells), so the arithmetic invites a reading the data does not
    # support. Emitted as words, matching how the surrounding prose spells counts.
    comp = cap["composition"]
    for family, suffix in (("opus", "Opus"), ("sonnet", "Sonnet"), ("haiku", "Haiku")):
        macros[f"statPCCapCells{suffix}"] = _num_word(comp["cells_by_family"].get(family, 0))
    # Which Opus version drops out of the scale for lack of a plurality session. A list
    # rather than a scalar in the data, so the single-element case is the only one the
    # prose can phrase; anything else must surface as an error rather than a silent pick.
    absent = comp["versions_without_a_cell"].get("opus", [])
    if len(absent) != 1:
        raise ValueError(
            "statPCCapAbsentOpusVersion assumes exactly one Opus version without a "
            f"capability cell; found {absent}. Update the appendix prose to match."
        )
    macros["statPCCapAbsentOpusVersion"] = _fmt_version(absent[0])
    return macros


def patch_macros(macros: dict[str, str], check: bool = False) -> list[tuple[str, str, str]]:
    """Replace each ``\\newcommand{\\name}{...}`` line's value in place, preserving
    every other line and macro byte-for-byte. Returns (name, old, new) for macros
    whose value actually changed."""
    with open(MACROS_PATH) as f:
        text = f.read()

    changes = []
    for name, new_value in macros.items():
        pattern = re.compile(r"(\\newcommand\{\\" + re.escape(name) + r"\}\{)([^\n]*)(\})")
        match = pattern.search(text)
        if not match:
            raise ValueError(f"macro \\{name} not found in {MACROS_PATH}")
        old_value = match.group(2)
        if old_value != new_value:
            changes.append((name, old_value, new_value))

        # A callable (not a replacement template) because the values carry LaTeX
        # backslashes that `sub` would otherwise read as group references. Spelled as
        # a named function rather than a lambda so it can be annotated for mypy while
        # still binding `new_value` as a default, which ruff's B023 requires.
        def _replace(m: re.Match[str], value: str = new_value) -> str:
            return m.group(1) + value + m.group(3)

        text = pattern.sub(_replace, text, count=1)

    if not check:
        with open(MACROS_PATH, "w") as f:
            f.write(text)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Report changes without writing the file."
    )
    args = parser.parse_args()

    macros = compute_macros()
    changes = patch_macros(macros, check=args.check)
    if not changes:
        print(f"No changes -- all {len(macros)} macros already match the computed values.")
        return
    verb = "Would update" if args.check else "Updated"
    for name, old, new in changes:
        print(f"{verb} \\{name}: {old!r} -> {new!r}")


if __name__ == "__main__":
    main()
