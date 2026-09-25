"""
Empirically-calibrated simulation of agentic coding session dynamics.

Data: data/company_sessions.json (git_session_extractor.py output).
IP-safe: only structural metrics — diff sizes, commit counts, model tier.
No code, no file paths, no commit messages.

Produces figures/emp_fig{1-4}.{pdf,png}.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Palette (from dataviz reference — blue slot 1, green slot 2)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
GREEN = "#008300"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK2 = "#52514e"
SURF = "#fcfcfb"

TIER_COLOR = {"opus": GREEN, "sonnet": BLUE}
TIER_LABEL = {"opus": "Opus", "sonnet": "Sonnet"}

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str = "data/company_sessions.json") -> tuple[list[dict], list[dict]]:
    d = json.loads(Path(path).read_text())
    return d["sessions"], d["commits"]


def model_tier(model: str | None) -> str:
    if model is None:
        # Control-mode (pre-AI-adoption) commits carry no trailer, so no model
        # string — this is the only source of a "human" tier.
        return "human"
    m = model.lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return "unknown"


_MODEL_CELL_RE = re.compile(r"Claude\s+(Opus|Sonnet|Haiku)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

# Family order for the capability scale. Deliberately coarse and explicit: this
# is an assumption the analysis rests on, not a measurement.
_FAMILY_RANK = {"haiku": 0, "sonnet": 1, "opus": 2}

CAPABILITY_ORDERINGS = ("family", "version")
"""Two defensible linear orders over ``(family, version)`` model cells.

``model_tier`` collapses eight distinct Opus trailer strings onto one label and
four Sonnet strings onto another, so a tier contrast pools Opus 4.5 with Opus 4.8
and Sonnet 4.5 with Sonnet 5. Within-bin capability spread is then plausibly
comparable to the between-bin contrast Finding 2 tries to detect, which would
attenuate a real gap toward the observed null. Ordering the cells on a single
capability axis tests the monotone prediction directly instead.

* ``"family"`` -- family first, version second: any Opus outranks any Sonnet
  outranks any Haiku. Keeps the paper's tier framing interpretable.
* ``"version"`` -- version first, family second: a newer generation outranks an
  older larger model, so Sonnet 5 outranks Opus 4.8.

Neither is provably right, which is why both are reported. They are a sensitivity
on each other.
"""


def model_cell(model: str | None) -> tuple[str, float] | None:
    """Parse a commit trailer into a ``(family, version)`` capability cell.

    Returns ``None`` for control-mode commits (no trailer), for bare ``"Claude"``
    with no version, and for anything unparseable — all of which must be excluded
    from an ordered analysis rather than silently ranked.

    The ``(1M context)`` suffix is deliberately collapsed: it is the same model
    with a larger context window, not a different capability level.
    """
    if model is None:
        return None
    m = _MODEL_CELL_RE.search(model)
    if not m:
        return None
    return (m.group(1).lower(), float(m.group(2)))


def capability_rank(cell: tuple[str, float], ordering: str = "family") -> tuple[float, float]:
    """Sort key placing ``cell`` on the chosen linear capability order."""
    family, version = cell
    if ordering == "family":
        return (_FAMILY_RANK[family], version)
    if ordering == "version":
        return (version, _FAMILY_RANK[family])
    raise ValueError(f"unknown ordering {ordering!r} — expected one of {CAPABILITY_ORDERINGS}")


def dominant_model_cell(session: dict, commits: list[dict]) -> tuple[str, float] | None:
    """Plurality ``(family, version)`` cell across a session's commits.

    Mirrors ``dominant_tier``: iterates the ordered list so ties resolve to the
    earliest-occurring cell rather than an arbitrary set ordering.
    """
    cells = [c for c in (model_cell(x["model"]) for x in _session_commits(session, commits)) if c]
    if not cells:
        return None
    return max(cells, key=cells.count)


def _session_commits(session: dict, commits: list[dict]) -> list[dict]:
    """Commits belonging to ``session``, in chronological order.

    Prefers the explicit ``session_id`` linkage baked into the public dataset
    (an integer per commit plus a ``seq`` order index), so commit-to-session
    membership and intra-session ordering survive the timestamp coarsening
    applied for public release. Falls back to the original timestamp-window
    match when commits carry no ``session_id`` — e.g. unit-test fixtures or
    freshly extracted data with full timestamps.
    """
    if any("session_id" in c for c in commits):
        sc = [
            c
            for c in commits
            if c["repo"] == session["repo"] and c.get("session_id") == session["session_id"]
        ]
        sc.sort(key=lambda c: c.get("seq", 0))
        return sc
    sc = [
        c
        for c in commits
        if c["repo"] == session["repo"]
        and session["start_ts"] <= c["timestamp"] <= session["end_ts"]
    ]
    sc.sort(key=lambda c: c["timestamp"])
    return sc


def dominant_tier(session: dict, commits: list[dict]) -> str:
    tiers = [model_tier(c["model"]) for c in _session_commits(session, commits)]
    if not tiers:
        return "unknown"
    # Iterate the ordered list (not a set): ``max`` returns the first element
    # achieving the maximal count, so an evenly-split session is attributed to
    # the tier of its earliest commit. Using ``set(tiers)`` here would make the
    # tie-break depend on set-iteration order (PYTHONHASHSEED), producing
    # non-reproducible fits across runs.
    return max(tiers, key=tiers.count)


def session_commit_churns(session: dict, commits: list[dict]) -> list[int]:
    return [c["churn"] for c in _session_commits(session, commits)]


def _nonpure_survivors(
    sessions: list[dict], commits: list[dict], tier: str | None = None
) -> list[float]:
    """Non-pure (survivor_ratio < 0.999) survivor ratios, optionally for one tier.

    ``tier=None`` pools every tier. Lives here, beside the other session-level
    primitives, because three call sites had each grown their own identical copy
    of this comprehension -- ``fit_params``, ``survivor_gap_stats``, and
    ``stratified_gap_stats`` (which reached it via the module-level definition
    this replaces).
    """
    return [
        s["survivor_ratio"]
        for s in sessions
        if s["survivor_ratio"] is not None
        and s["survivor_ratio"] < 0.999
        and (tier is None or dominant_tier(s, commits) == tier)
    ]


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------


def _log_churn_ratios(sessions: list[dict], commits: list[dict]) -> list[float]:
    """log(x_{t+1}/x_t) over consecutive commit pairs, for the churn-decay fit.

    Sessions under 3 commits give no usable pair; zero-churn commits are skipped
    because the ratio is undefined rather than because they carry no signal.
    """
    log_ratios: list[float] = []
    for s in sessions:
        if s["n_commits"] < 3:
            continue
        churns = session_commit_churns(s, commits)
        for a, b in zip(churns[:-1], churns[1:], strict=False):
            if a > 0 and b > 0:
                log_ratios.append(math.log(b / a))
    return log_ratios


def _decay_cis(log_ratios: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% percentile bootstrap CIs for (rho, sigma). Seed 42, 9999 resamples.

    The seed and resample count are load-bearing: `rho_ci` is reported in the
    paper, so changing either silently moves a published number.
    """
    if len(log_ratios) < 2:
        return (0.0, 0.0), (0.0, 0.0)

    lr_arr = np.array(log_ratios)

    def _ci(statistic: Callable[[np.ndarray], float]) -> tuple[float, float]:
        boot = scipy_stats.bootstrap(
            (lr_arr,),
            statistic,
            confidence_level=0.95,
            method="percentile",
            n_resamples=9999,
            random_state=42,
        )
        return float(boot.confidence_interval.low), float(boot.confidence_interval.high)

    return (
        _ci(lambda a: float(np.exp(np.mean(a)))),
        _ci(lambda a: float(np.std(a, ddof=1))),
    )


def _pi_pure(sessions: list[dict]) -> float:
    """Fraction of n>=2 sessions that are pure addition (survivor_ratio >= 0.999)."""
    multi_sr = [
        s["survivor_ratio"]
        for s in sessions
        if s["survivor_ratio"] is not None and s["n_commits"] >= 2
    ]
    if not multi_sr:
        return 0.0
    return sum(1 for sr in multi_sr if sr >= 0.999) / len(multi_sr)


@dataclass
class FittedParams:
    rho: float  # geometric churn-decay factor (consecutive commit pairs)
    sigma: float  # log-space noise std
    tau_opus: float  # mean survivor ratio — Opus sessions (non-pure only)
    tau_sonnet: float  # mean survivor ratio — Sonnet sessions (non-pure only)
    tau_pooled: float  # pooled (non-pure only)
    session_n_median: float
    session_n_mean: float
    n_sessions_fit: int  # sessions used for churn-decay fit
    pi_pure: float = 0.0  # fraction of n≥2 sessions with survivor_ratio ≥ 0.999
    rho_ci: tuple[float, float] = (0.0, 0.0)  # 95% bootstrap CI for rho
    sigma_ci: tuple[float, float] = (0.0, 0.0)  # 95% bootstrap CI for sigma


def fit_params(sessions: list[dict], commits: list[dict]) -> FittedParams:
    log_ratios = _log_churn_ratios(sessions, commits)

    # Geometric churn-decay: rho = exp(mean(log_ratios)), sigma = stdev(log_ratios)
    rho = math.exp(statistics.mean(log_ratios)) if log_ratios else 0.5
    sigma = statistics.stdev(log_ratios) if len(log_ratios) > 1 else 0.5
    rho_ci, sigma_ci = _decay_cis(log_ratios)

    # Re-fit tau on non-pure sessions so the mixture mean matches observed
    opus_sr = _nonpure_survivors(sessions, commits, "opus")
    sonnet_sr = _nonpure_survivors(sessions, commits, "sonnet")
    all_sr = _nonpure_survivors(sessions, commits)

    ns = [s["n_commits"] for s in sessions]

    return FittedParams(
        rho=rho,
        sigma=sigma,
        tau_opus=statistics.mean(opus_sr) if opus_sr else 0.70,
        tau_sonnet=statistics.mean(sonnet_sr) if sonnet_sr else 0.63,
        tau_pooled=statistics.mean(all_sr) if all_sr else 0.65,
        session_n_median=statistics.median(ns),
        session_n_mean=statistics.mean(ns),
        n_sessions_fit=sum(1 for s in sessions if s["n_commits"] >= 3),
        pi_pure=_pi_pure(sessions),
        rho_ci=rho_ci,
        sigma_ci=sigma_ci,
    )


def survivor_gap_stats(
    sessions: list[dict], commits: list[dict], *, tiers: tuple[str, str] = ("opus", "sonnet")
) -> dict:
    """Permutation test + bootstrap CI for the survivor-ratio gap between two tiers.

    Defaults to Opus/Sonnet (the paper's Finding 2), returning keys named for
    those two tiers: tau_opus_mean, tau_sonnet_mean, gap, p_value, significant,
    ci_opus_95, ci_sonnet_95, ci_gap_95, n_opus, n_sonnet. Passing e.g.
    ``tiers=("human", "opus")`` runs the same test between any other pair
    (the "human" tier comes from control-mode extraction — see
    ``git_session_extractor.py``'s ``require_ai_trailer=False`` path) and
    names the keys accordingly (tau_human_mean, n_human, ...).
    Returns {"error": "insufficient data"} if either tier has no non-pure sessions.
    """
    tier_a, tier_b = tiers

    opus_sr = _nonpure_survivors(sessions, commits, tier_a)
    sonnet_sr = _nonpure_survivors(sessions, commits, tier_b)
    if not opus_sr or not sonnet_sr:
        return {"error": "insufficient data"}

    a = np.array(opus_sr, dtype=float)
    b = np.array(sonnet_sr, dtype=float)

    def _diff_means(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(x) - np.mean(y))

    perm = scipy_stats.permutation_test(
        (a, b),
        _diff_means,
        permutation_type="independent",
        alternative="two-sided",
        n_resamples=9999,
        random_state=42,
    )
    ci_opus = scipy_stats.bootstrap(
        (a,), np.mean, confidence_level=0.95, method="percentile", n_resamples=9999, random_state=42
    ).confidence_interval
    ci_sonnet = scipy_stats.bootstrap(
        (b,), np.mean, confidence_level=0.95, method="percentile", n_resamples=9999, random_state=42
    ).confidence_interval
    ci_gap = scipy_stats.bootstrap(
        (a, b),
        _diff_means,
        confidence_level=0.95,
        method="percentile",
        n_resamples=9999,
        random_state=42,
    ).confidence_interval

    mwu = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")

    return {
        f"tau_{tier_a}_mean": float(np.mean(a)),
        f"tau_{tier_b}_mean": float(np.mean(b)),
        "gap": float(np.mean(a) - np.mean(b)),
        "p_value": float(perm.pvalue),
        "mwu_p": float(mwu.pvalue),
        "significant": bool(perm.pvalue < 0.05),
        f"ci_{tier_a}_95": [float(ci_opus.low), float(ci_opus.high)],
        f"ci_{tier_b}_95": [float(ci_sonnet.low), float(ci_sonnet.high)],
        "ci_gap_95": [float(ci_gap.low), float(ci_gap.high)],
        f"n_{tier_a}": len(opus_sr),
        f"n_{tier_b}": len(sonnet_sr),
    }


def _late_early_ratio(churns: list[int]) -> float | None:
    """late/early churn ratio for one session's per-commit churn sequence.

    late = mean churn in second half of session; early = mean churn in first
    half. Returns None if the session has fewer than 5 commits or a
    non-positive early-half mean (ratio undefined).
    """
    n = len(churns)
    if n < 5:
        return None
    mid = n // 2
    early_mean = statistics.mean(churns[:mid])
    if early_mean <= 0:
        return None
    return statistics.mean(churns[mid:]) / early_mean


def _collect_late_early_ratios(
    sessions: list[dict], commits: list[dict]
) -> tuple[list[float], list[list[int]]]:
    """Per-session late/early churn ratios and their underlying churn sequences,
    restricted to sessions with n_commits >= 5 and a defined ratio."""
    obs_ratios: list[float] = []
    session_churns: list[list[int]] = []
    for s in sessions:
        churns = session_commit_churns(s, commits)
        ratio = _late_early_ratio(churns)
        if ratio is not None:
            obs_ratios.append(ratio)
            session_churns.append(churns)
    return obs_ratios, session_churns


def churn_decay_permutation(
    sessions: list[dict],
    commits: list[dict],
    n_resamples: int = 9999,
    seed: int = 42,
) -> dict:
    """Permutation test for within-session churn decay.

    Statistic: median(late/early churn ratio) across sessions with n_commits >= 5.
    Null: shuffle commit order within each session (destroys temporal structure).

    Returns observed_stat, p_value, n_sessions.
    Returns {"error": "no sessions with n_commits >= 5"} if no eligible sessions.
    """
    rng = random.Random(seed)

    obs_ratios, session_churns = _collect_late_early_ratios(sessions, commits)
    if not obs_ratios:
        return {"error": "no sessions with n_commits >= 5"}

    observed_stat = statistics.median(obs_ratios)

    null_stats: list[float] = []
    for _ in range(n_resamples):
        shuffled_ratios: list[float] = []
        for churns in session_churns:
            shuffled = churns[:]
            rng.shuffle(shuffled)
            ratio = _late_early_ratio(shuffled)
            if ratio is not None:
                shuffled_ratios.append(ratio)
        if shuffled_ratios:
            null_stats.append(statistics.median(shuffled_ratios))

    # One-sided p: fraction of null statistics ≤ observed (decay = small ratio is the signal)
    p_value = float(np.mean([s <= observed_stat for s in null_stats])) if null_stats else 1.0

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "n_sessions": len(obs_ratios),
    }


def churn_ratio_bootstrap_ci(
    sessions: list[dict],
    commits: list[dict],
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for the median late/early churn ratio (Finding 1).

    Resamples sessions with replacement and recomputes the median ratio each
    time, giving a precision estimate to sit alongside churn_decay_permutation's
    p-value: the permutation test says the decay isn't an artifact of commit
    order; this CI says how tightly the ratio itself is pinned down.

    Returns observed_stat, ci_low, ci_high, n_sessions.
    Returns {"error": "no sessions with n_commits >= 5"} if no eligible sessions.
    """
    obs_ratios, _ = _collect_late_early_ratios(sessions, commits)
    if not obs_ratios:
        return {"error": "no sessions with n_commits >= 5"}

    arr = np.array(obs_ratios, dtype=float)
    ci = scipy_stats.bootstrap(
        (arr,),
        np.median,
        confidence_level=confidence_level,
        method="percentile",
        n_resamples=n_resamples,
        random_state=seed,
    ).confidence_interval

    return {
        "observed_stat": float(np.median(arr)),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "n_sessions": len(obs_ratios),
    }


def out_of_sample_fit(
    sessions: list[dict],
    commits: list[dict],
    seed: int = 42,
) -> dict:
    """Train/test split to validate survivor-ratio distribution out-of-sample.

    Fits params on 70% of eligible sessions (n_commits >= 2, non-None survivor_ratio),
    runs simulation with train params, then compares held-out empirical survivor ratios
    to simulated using ks_2samp.

    Note: survivor ratio is governed by the Beta/pi_pure mixture (not rho).
    This KS test validates the generative component for survivor ratios.

    Returns ks_stat, ks_p, n_train, n_test.
    """
    rng = random.Random(seed)

    eligible = [s for s in sessions if s["survivor_ratio"] is not None and s["n_commits"] >= 2]
    if len(eligible) < 10:
        return {"error": "insufficient data"}

    shuffled = eligible[:]
    rng.shuffle(shuffled)
    split = int(0.7 * len(shuffled))
    train = shuffled[:split]
    test = shuffled[split:]

    if len(train) < 5 or len(test) < 5:
        return {"error": "insufficient data"}

    train_params = fit_params(train, commits)
    sim = run_simulation(train_params, n_synthetic=500)
    sim_pooled = sim["opus"]["survivor_ratios"] + sim["sonnet"]["survivor_ratios"]

    test_sr = [s["survivor_ratio"] for s in test]

    ks = scipy_stats.ks_2samp(test_sr, sim_pooled, alternative="two-sided")

    return {
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "n_train": len(train),
        "n_test": len(test),
    }


# ---------------------------------------------------------------------------
# De-confounding robustness checks for Finding 2 (tier gap)
# ---------------------------------------------------------------------------
#
# Model tier is not randomly assigned: Opus skews to planning/architecture,
# Sonnet to execution/edits, and those task types carry systematically
# different churn signatures. The survivor ratio itself is |net|/churn — an
# additivity measure — so the confounder (task type) is collinear with the
# outcome. That rules out propensity-score adjustment (matching on
# outcome-collinear covariates is overadjustment). Instead we (a) quantify how
# little unmeasured confounding would explain the gap (E-value), (b) hold a
# coarse task-type proxy fixed and re-test (stratified permutation), and
# (c) test a discriminator less tautological with additivity (per-tier churn
# decay rho).


def _session_task_type(session: dict) -> str:
    """Coarse task-type label = the session's plurality commit type.

    A noisy proxy: ~30-50% of commits are non-conventional ('other'), so this
    bounds the planning/execution confound rather than removing it. Tie-break is
    deterministic (higher count, then lexical) so fits reproduce across runs.
    """
    cts = session.get("commit_types", {})
    if not cts:
        return "other"
    return max(cts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _evalue_threshold(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele & Ding 2017). RR<1 is inverted."""
    if rr <= 0:
        return float("inf")
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def gap_sensitivity_evalue(sessions: list[dict], commits: list[dict]) -> dict:
    """E-value sensitivity analysis for the Opus-Sonnet survivor-ratio gap.

    Quantifies how strongly an unmeasured tier-outcome confounder (e.g. the
    planning/execution task-type split) would have to be associated with *both*
    model tier and survivor ratio to fully explain the observed gap. Uses the
    standardized-mean-difference approximation of the EValue package
    (evalues.MD): d = gap / pooled_sd, RR ~= exp(0.91 d),
    E = RR + sqrt(RR (RR - 1)). ``evalue_ci`` is the E-value for the CI bound
    nearest the null; when the gap CI already includes the null it is 1.0 —
    i.e. no unmeasured confounding at all is required for a null effect.

    Returns cohens_d, rr_approx, rr_ci, evalue_point, evalue_ci, n_opus, n_sonnet.
    """
    opus = _nonpure_survivors(sessions, commits, "opus")
    sonnet = _nonpure_survivors(sessions, commits, "sonnet")
    if len(opus) < 2 or len(sonnet) < 2:
        return {"error": "insufficient data"}

    a = np.array(opus, dtype=float)
    b = np.array(sonnet, dtype=float)
    n1, n2 = len(a), len(b)
    gap = float(a.mean() - b.mean())

    # Pooled SD (Cohen's d denominator)
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = gap / sp if sp > 0 else 0.0
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))

    rr = math.exp(0.91 * d)
    rr_lo = math.exp(0.91 * d - 1.78 * se_d)
    rr_hi = math.exp(0.91 * d + 1.78 * se_d)

    e_point = _evalue_threshold(rr)
    if rr_lo <= 1.0 <= rr_hi:
        e_ci = 1.0  # CI includes the null: no confounding needed
    else:
        bound = rr_lo if rr > 1.0 else rr_hi
        e_ci = _evalue_threshold(bound)

    return {
        "cohens_d": d,
        "rr_approx": rr,
        "rr_ci": [rr_lo, rr_hi],
        "evalue_point": e_point,
        "evalue_ci": e_ci,
        "n_opus": n1,
        "n_sonnet": n2,
    }


def stratified_gap_stats(
    sessions: list[dict],
    commits: list[dict],
    n_resamples: int = 9999,
    seed: int = 42,
) -> dict:
    """Task-type-stratified Opus-Sonnet gap, holding the confound proxy fixed.

    Each non-pure, tier-assigned session is labelled by its plurality commit
    type (``_session_task_type``). We report per-stratum gaps and a
    Mantel-Haenszel-style size-weighted pooled ('adjusted') gap over strata
    containing both tiers, with a stratified permutation test that shuffles tier
    labels *within* each stratum (so task-type composition is held fixed under
    the null). commit type is a coarse proxy, so this bounds — not removes — the
    planning/execution confound.

    Returns adjusted_gap, naive_gap, p_value, n_strata_used, strata (breakdown).
    """
    from collections import defaultdict

    strata: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s in sessions:
        sr = s["survivor_ratio"]
        if sr is None or sr >= 0.999:
            continue
        tier = dominant_tier(s, commits)
        if tier not in ("opus", "sonnet"):
            continue
        strata[_session_task_type(s)].append((tier, sr))

    def _weighted_gap(strata_map: dict[str, list[tuple[str, float]]]) -> tuple[float, dict]:
        num = den = 0.0
        breakdown: dict[str, dict] = {}
        for tt, items in strata_map.items():
            o = [sr for (t, sr) in items if t == "opus"]
            s = [sr for (t, sr) in items if t == "sonnet"]
            if o and s:
                w = len(o) * len(s) / (len(o) + len(s))  # MH-style weight
                g = statistics.mean(o) - statistics.mean(s)
                num += w * g
                den += w
                breakdown[tt] = {
                    "n_opus": len(o),
                    "n_sonnet": len(s),
                    "tau_opus": statistics.mean(o),
                    "tau_sonnet": statistics.mean(s),
                    "gap": g,
                }
        return (num / den if den > 0 else float("nan")), breakdown

    obs_gap, breakdown = _weighted_gap(strata)
    if math.isnan(obs_gap):
        return {"error": "no stratum contains both tiers"}

    # Naive (unstratified) gap over the same sessions, for reference
    all_opus = [sr for items in strata.values() for (t, sr) in items if t == "opus"]
    all_sonnet = [sr for items in strata.values() for (t, sr) in items if t == "sonnet"]
    naive_gap = statistics.mean(all_opus) - statistics.mean(all_sonnet)

    rng = random.Random(seed)
    count = valid = 0
    for _ in range(n_resamples):
        permuted: dict[str, list[tuple[str, float]]] = {}
        for tt, items in strata.items():
            tiers = [t for (t, _) in items]
            srs = [sr for (_, sr) in items]
            rng.shuffle(tiers)
            permuted[tt] = list(zip(tiers, srs, strict=False))
        g, _ = _weighted_gap(permuted)
        if not math.isnan(g):
            valid += 1
            if abs(g) >= abs(obs_gap) - 1e-12:
                count += 1
    p_value = count / valid if valid else 1.0

    return {
        "adjusted_gap": obs_gap,
        "naive_gap": naive_gap,
        "p_value": p_value,
        "n_strata_used": len(breakdown),
        "strata": breakdown,
    }


def tier_decay_stats(
    sessions: list[dict],
    commits: list[dict],
    n_resamples: int = 9999,
    seed: int = 42,
) -> dict:
    """Per-tier geometric churn-decay rho — a discriminator less tautological
    with additivity than the survivor ratio.

    rho is the within-session decay *rate* of consecutive-commit churn, fitted
    exactly as in ``fit_params`` but split by dominant tier. If
    rho_Opus ~= rho_Sonnet, the two tiers converge at the same rate and the
    capability-ceiling reading of the survivor-ratio gap is weakened: the tau
    gap is then more consistent with a task-type (additivity) artifact than a
    capability-driven difference in convergence dynamics. The permutation test
    shuffles tier labels at the *session* level, respecting within-session
    correlation of log-ratios.

    Returns rho_opus, rho_sonnet, rho_gap, p_value, n_pairs_opus,
    n_pairs_sonnet, n_sessions.
    """
    sess_lr: list[tuple[str, list[float]]] = []
    for s in sessions:
        if s["n_commits"] < 3:
            continue
        tier = dominant_tier(s, commits)
        if tier not in ("opus", "sonnet"):
            continue
        churns = session_commit_churns(s, commits)
        lrs = [
            math.log(b / a)
            for a, b in zip(churns[:-1], churns[1:], strict=False)
            if a > 0 and b > 0
        ]
        if lrs:
            sess_lr.append((tier, lrs))

    labels = [t for (t, _) in sess_lr]

    def _mean_logratio(labs: list[str], tier: str) -> float | None:
        pool = [
            lr for (_, lrs), lab in zip(sess_lr, labs, strict=False) if lab == tier for lr in lrs
        ]
        return statistics.mean(pool) if pool else None

    m_opus = _mean_logratio(labels, "opus")
    m_sonnet = _mean_logratio(labels, "sonnet")
    if m_opus is None or m_sonnet is None:
        return {"error": "insufficient data"}

    obs = m_opus - m_sonnet  # difference of mean log-ratios (log-space rho gap)

    rng = random.Random(seed)
    count = valid = 0
    shuffled = labels[:]
    for _ in range(n_resamples):
        rng.shuffle(shuffled)
        mo = _mean_logratio(shuffled, "opus")
        ms = _mean_logratio(shuffled, "sonnet")
        if mo is not None and ms is not None:
            valid += 1
            if abs(mo - ms) >= abs(obs) - 1e-12:
                count += 1
    p_value = count / valid if valid else 1.0

    n_pairs_opus = sum(len(lrs) for (t, lrs) in sess_lr if t == "opus")
    n_pairs_sonnet = sum(len(lrs) for (t, lrs) in sess_lr if t == "sonnet")

    return {
        "rho_opus": math.exp(m_opus),
        "rho_sonnet": math.exp(m_sonnet),
        "rho_gap": math.exp(m_opus) - math.exp(m_sonnet),
        "p_value": p_value,
        "n_pairs_opus": n_pairs_opus,
        "n_pairs_sonnet": n_pairs_sonnet,
        "n_sessions": len(sess_lr),
    }


def mixed_model_gap_stats(
    sessions: list[dict], commits: list[dict], *, tiers: tuple[str, str] = ("sonnet", "opus")
) -> dict:
    """Multilevel de-confound of the survivor-ratio gap between two tiers.

    ``tiers[0]`` is the reference/treatment baseline (``Treatment(...)`` in the
    formula); ``tiers[1]`` is the tier compared against it — defaults to
    Opus-vs-Sonnet (the paper's Finding 2). Pass e.g. ``tiers=("human", "opus")``
    to compare the pre-AI-adoption control tier against Opus instead; the
    reported ``tier_coef``/``tier_p``/``tier_ci`` then describe that pair.

    Adjusts for task-type (``task_type``, a churn-independent dir-category
    proxy — see ``git_session_extractor._dir_category``) as a fixed effect,
    and models repo and developer as *crossed* random intercepts (a developer
    can appear in more than one repo, so they are not nested). statsmodels'
    ``MixedLM`` has no native crossed-random-effects syntax, so we use the
    standard workaround: a single constant group (``groups="grp"``, no
    per-group random effect via ``re_formula="0"``) plus both random
    intercepts declared as variance components via ``vc_formula``.

    The outcome is a bounded [0, 1] ratio, so this is a linear approximation
    for interpretability — consistent with the raw-survivor gap analysis
    elsewhere in this module, not a beta/logit model. Report only, never
    tune toward significance.

    Returns tier_coef, tier_p, tier_ci, vc_repo, vc_developer, n_sessions,
    n_repos, n_developers, converged. On insufficient data (< 2 repos or a
    singular fit) returns converged=False with a diagnostic-only payload
    (no coef/p/ci) rather than forcing a fit.
    """
    reference, treatment = tiers
    rows = []
    for s in sessions:
        sr = s["survivor_ratio"]
        if sr is None or sr >= 0.999:
            continue
        tier = dominant_tier(s, commits)
        if tier not in tiers:
            continue
        rows.append(
            {
                "survivor": sr,
                "tier": tier,
                "task_type": s.get("task_type", "other"),
                "repo": s["repo"],
                "developer": s.get("developer", "unknown"),
            }
        )

    n_sessions = len(rows)
    n_repos = len({r["repo"] for r in rows})
    n_developers = len({r["developer"] for r in rows})

    if n_sessions < 4 or n_repos < 2:
        return {
            "converged": False,
            "error": "insufficient data (<4 sessions or <2 repos)",
            "n_sessions": n_sessions,
            "n_repos": n_repos,
            "n_developers": n_developers,
        }

    df = pd.DataFrame(rows)
    df["grp"] = 1

    try:
        md = smf.mixedlm(
            f"survivor ~ C(tier, Treatment('{reference}')) + C(task_type)",
            df,
            groups="grp",
            re_formula="0",
            vc_formula={"repo": "0 + C(repo)", "developer": "0 + C(developer)"},
        )
        res = md.fit(reml=True, method=["lbfgs", "cg"])
    except (np.linalg.LinAlgError, ValueError):
        return {
            "converged": False,
            "error": "singular fit",
            "n_sessions": n_sessions,
            "n_repos": n_repos,
            "n_developers": n_developers,
        }

    tier_key = f"C(tier, Treatment('{reference}'))[T.{treatment}]"
    if not res.converged or tier_key not in res.params.index:
        return {
            "converged": False,
            "error": "did not converge"
            if tier_key in res.params.index
            else "tier coefficient missing",
            "n_sessions": n_sessions,
            "n_repos": n_repos,
            "n_developers": n_developers,
        }

    ci = res.conf_int()
    return {
        "converged": True,
        "tier_coef": float(res.params[tier_key]),
        "tier_p": float(res.pvalues[tier_key]),
        "tier_ci": [float(ci.loc[tier_key, 0]), float(ci.loc[tier_key, 1])],
        "vc_repo": float(res.params["repo Var"]),
        "vc_developer": float(res.params["developer Var"]),
        "n_sessions": n_sessions,
        "n_repos": n_repos,
        "n_developers": n_developers,
    }


def namespace_repos(data: dict, prefix: str) -> tuple[list[dict], list[dict]]:
    """Prefix ``repo`` for cross-dataset merging; leave ``developer`` untouched.

    Repository labels are positional -- ``_repo_label_map`` letters them per
    extraction -- so ``repo_A`` denotes a different repository in each dataset and
    merging without a prefix would fuse unrelated repositories into one
    random-effect level.

    Developer labels are the opposite. They are ``HMAC(salt, canonical_identity)``,
    so under one salt and one curated roster the same person yields the same label
    in both eras. Prefixing would split that person into two crossed-random-effect
    levels and discard the only within-person signal available against the era
    confound (see :func:`paired_developer_stats`). Under different salts the labels
    are simply incomparable and the prefix would remove nothing.
    """
    sessions = [dict(s, repo=f"{prefix}_{s['repo']}") for s in data["sessions"]]
    commits = [dict(c, repo=f"{prefix}_{c['repo']}") for c in data["commits"]]
    for row in (*sessions, *commits):
        row.setdefault("developer", "unknown")
    return sessions, commits


def _developer_tau(sessions: list[dict]) -> dict[str, tuple[int, float]]:
    """Per-developer ``(n_non_pure_sessions, mean_tau)`` over non-pure sessions."""
    per: dict[str, list[float]] = {}
    for s in sessions:
        sr = s["survivor_ratio"]
        if sr is None or sr >= 0.999:
            continue
        per.setdefault(s.get("developer", "unknown"), []).append(sr)
    return {d: (len(v), statistics.mean(v)) for d, v in per.items()}


def paired_developer_stats(
    sessions_a: list[dict],
    commits_a: list[dict],
    sessions_b: list[dict],
    commits_b: list[dict],
    *,
    min_commits_per_side: int = 0,
) -> dict:
    """Paired within-developer comparison of survivor ratio across two datasets.

    Pairs on the ``developer`` label, which is only meaningful when both datasets
    were extracted with the **same salt and the same curated roster** — the label
    is ``HMAC(salt, canonical_identity)``, so a different salt or a differently
    spelled canonical label yields no overlap and this function silently returns
    ``n_pairs=0`` rather than erroring. Check ``n_pairs`` against the roster
    overlap you expect.

    Deliberately reports a **sign test on the paired direction**, not an effect
    size with a confidence interval. Two reasons, both structural rather than
    merely statistical:

    * The pair count is small (13 on this org's data, falling to 6 at a 50-commit
      floor), so an interval would imply precision the design cannot support.
    * The two datasets need not have equivalent inclusion rules. Where dataset A
      keeps all of a developer's commits and dataset B keeps only their
      AI-assisted ones, the comparison is between *different slices* of the same
      person's work. That is legitimate for a per-session shape metric like the
      survivor ratio, but it layers a task-selection difference on top of whatever
      else separates the datasets, and it makes any volume or productivity read
      uninterpretable. ``n_commits`` per side is returned so that asymmetry stays
      visible rather than being averaged away.

    ``min_commits_per_side`` drops developers with too little data on either side.

    Returns n_pairs, n_b_greater, sign_p, median_delta, wilcoxon_p, and a
    per-developer table. ``median_delta`` is descriptive only.

    Four summaries of the per-developer table are returned alongside it, because the
    honest reading of this comparison depends on them and a caller that only saw the
    aggregates would have to re-derive them by hand:

    * ``delta_min``/``delta_max`` -- the signed spread, recomputed over the floored
      population. This is what separates the two tests reported here: Wilcoxon can
      reach significance purely because the positive deltas are larger in magnitude
      than the negative ones, an assumption the sign test declines to make. Quoting
      the significant p-value without the spread would hide which assumption is
      doing the work.
    * ``top_volume_pair`` -- the pair with the most commits on *both* sides
      (maximizing the smaller side, since that is what bounds the precision of the
      delta), so "the direction is not an artifact of thin data" is checkable rather
      than asserted.
    * ``max_volume_ratio`` -- the worst per-developer imbalance, ``max/min`` over the
      two sides. Under unequal inclusion rules this can be large, and it is the
      reason volume reads are uninterpretable.

    All four are omitted entirely when there are no pairs.
    """
    tau_a, tau_b = _developer_tau(sessions_a), _developer_tau(sessions_b)
    commits_by_dev_a: dict[str, int] = {}
    commits_by_dev_b: dict[str, int] = {}
    for commits, sink in ((commits_a, commits_by_dev_a), (commits_b, commits_by_dev_b)):
        for c in commits:
            dev = c.get("developer", "unknown")
            sink[dev] = sink.get(dev, 0) + 1

    pairs: list[dict] = []
    for dev in sorted(set(tau_a) & set(tau_b)):
        n_ca, n_cb = commits_by_dev_a.get(dev, 0), commits_by_dev_b.get(dev, 0)
        if n_ca < min_commits_per_side or n_cb < min_commits_per_side:
            continue
        (n_sa, ta), (n_sb, tb) = tau_a[dev], tau_b[dev]
        pairs.append(
            {
                "developer": dev,
                "n_commits_a": n_ca,
                "n_commits_b": n_cb,
                "n_sessions_a": n_sa,
                "n_sessions_b": n_sb,
                "tau_a": ta,
                "tau_b": tb,
                "delta": tb - ta,
            }
        )

    base = {
        "n_pairs": len(pairs),
        "min_commits_per_side": min_commits_per_side,
        "per_developer": pairs,
    }
    if not pairs:
        return {**base, "error": "no developers present in both datasets"}

    # Annotated because ``pairs`` is list[dict], so every value it yields is typed
    # `object` -- without this the comparisons, median, and min/max below are all
    # uncheckable.
    deltas: list[float] = [float(p["delta"]) for p in pairs]
    n_greater = sum(1 for d in deltas if d > 0)
    n_nonzero = sum(1 for d in deltas if d != 0)
    if n_nonzero:
        sign_p = float(scipy_stats.binomtest(n_greater, n_nonzero, 0.5).pvalue)
    else:
        sign_p = float("nan")
    # Wilcoxon needs at least a handful of non-zero pairs to mean anything; it is
    # reported alongside the sign test, not instead of it.
    try:
        wilcoxon_p = float(scipy_stats.wilcoxon(deltas).pvalue) if n_nonzero >= 6 else float("nan")
    except ValueError:
        wilcoxon_p = float("nan")

    # Best-measured pair: maximize the *smaller* side. Total or single-side volume
    # would crown a developer with 900 commits on one side and 1 on the other, which
    # is precisely the thin-data case this summary exists to rule out.
    def _smaller_side(pair: dict) -> int:
        return min(int(pair["n_commits_a"]), int(pair["n_commits_b"]))

    def _larger_side(pair: dict) -> int:
        return max(int(pair["n_commits_a"]), int(pair["n_commits_b"]))

    top = max(pairs, key=lambda p: (_smaller_side(p), str(p["developer"])))
    ratios = [_larger_side(p) / _smaller_side(p) for p in pairs if _smaller_side(p) > 0]

    return {
        **base,
        "n_b_greater": n_greater,
        "n_nonzero": n_nonzero,
        "sign_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "median_delta": statistics.median(deltas),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "top_volume_pair": {
            "developer": top["developer"],
            "n_commits_a": top["n_commits_a"],
            "n_commits_b": top["n_commits_b"],
            "delta": top["delta"],
        },
        "max_volume_ratio": max(ratios) if ratios else float("nan"),
    }


def capability_trend_stats(
    sessions: list[dict],
    commits: list[dict],
    *,
    ordering: str = "family",
    include_haiku: bool = True,
) -> dict:
    """Rank correlation between model capability and survivor ratio.

    Tests Finding 2's prediction as a monotone trend over a linear capability
    order instead of a two-bin contrast, so version pooling cannot mask a real
    effect (see ``CAPABILITY_ORDERINGS``). Sessions are ranked by their plurality
    ``(family, version)`` cell; those with no parseable cell are excluded.

    ``include_haiku`` defaults to True: excluding the weakest model from a
    capability test discards the low end of the range that gives a monotone test
    its power. Haiku's cell count is small, so the per-cell table is returned for
    the reader to judge rather than only the summary statistic.

    Returns spearman_rho, spearman_p, n_sessions, ordering, the ranked cell
    labels, and a per-cell table of (family, version, n_sessions, tau).
    """
    if ordering not in CAPABILITY_ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r} — expected one of {CAPABILITY_ORDERINGS}")

    rows: list[tuple[tuple[str, float], float]] = []
    for s in sessions:
        sr = s["survivor_ratio"]
        if sr is None or sr >= 0.999:
            continue
        cell = dominant_model_cell(s, commits)
        if cell is None or (not include_haiku and cell[0] == "haiku"):
            continue
        rows.append((cell, sr))

    cells = sorted({c for c, _ in rows}, key=lambda c: capability_rank(c, ordering))
    if len(cells) < 2:
        return {
            "error": "insufficient data (<2 distinct model cells)",
            "n_sessions": len(rows),
            "ordering": ordering,
        }

    rank_of = {c: i for i, c in enumerate(cells)}
    result = scipy_stats.spearmanr([rank_of[c] for c, _ in rows], [sr for _, sr in rows])

    per_cell: dict[tuple[str, float], list[float]] = {}
    for c, sr in rows:
        per_cell.setdefault(c, []).append(sr)

    return {
        "spearman_rho": float(result.statistic),
        "spearman_p": float(result.pvalue),
        "n_sessions": len(rows),
        "n_cells": len(cells),
        "ordering": ordering,
        "include_haiku": include_haiku,
        "ranked_cells": [f"{f}{v:g}" for f, v in cells],
        "per_cell": [
            {
                "family": f,
                "version": v,
                "n_sessions": len(per_cell[(f, v)]),
                "tau": statistics.mean(per_cell[(f, v)]),
            }
            for f, v in cells
        ],
    }


def capability_mixed_model_stats(
    sessions: list[dict],
    commits: list[dict],
    *,
    ordering: str = "family",
    include_haiku: bool = True,
) -> dict:
    """Capability rank as a continuous fixed effect, repo/developer crossed.

    The de-confounded counterpart to ``capability_trend_stats``: the same crossed
    random-intercept structure as ``mixed_model_gap_stats`` (see its docstring for
    why ``vc_formula`` rather than nested groups), with capability rank replacing
    the two-level tier factor. Without this, the ordered result would be the only
    Part C capability check left unadjusted for repository and developer.

    A rank is ordinal and is treated here as equally spaced; the
    ``family``/``version`` ordering sensitivity is partly there to probe how much
    that assumption carries. Report only, never tune toward significance.
    """
    if ordering not in CAPABILITY_ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r} — expected one of {CAPABILITY_ORDERINGS}")

    prepared: list[tuple[tuple[str, float], dict]] = []
    for s in sessions:
        sr = s["survivor_ratio"]
        if sr is None or sr >= 0.999:
            continue
        cell = dominant_model_cell(s, commits)
        if cell is None or (not include_haiku and cell[0] == "haiku"):
            continue
        prepared.append(
            (
                cell,
                {
                    "survivor": sr,
                    "task_type": s.get("task_type", "other"),
                    "repo": s["repo"],
                    "developer": s.get("developer", "unknown"),
                },
            )
        )

    cells = sorted({c for c, _ in prepared}, key=lambda c: capability_rank(c, ordering))
    rank_of = {c: i for i, c in enumerate(cells)}
    rows = [{**row, "cap_rank": rank_of[cell]} for cell, row in prepared]

    n_sessions = len(rows)
    n_repos = len({r["repo"] for r in rows})
    base = {
        "n_sessions": n_sessions,
        "n_repos": n_repos,
        "n_developers": len({r["developer"] for r in rows}),
        "n_cells": len(cells),
        "ordering": ordering,
        "include_haiku": include_haiku,
    }

    if n_sessions < 4 or n_repos < 2 or len(cells) < 2:
        return {
            **base,
            "converged": False,
            "error": "insufficient data (<4 sessions, <2 repos, or <2 model cells)",
        }

    df = pd.DataFrame(rows)
    df["grp"] = 1
    try:
        md = smf.mixedlm(
            "survivor ~ cap_rank + C(task_type)",
            df,
            groups="grp",
            re_formula="0",
            vc_formula={"repo": "0 + C(repo)", "developer": "0 + C(developer)"},
        )
        res = md.fit(reml=True, method=["lbfgs", "cg"])
    except (np.linalg.LinAlgError, ValueError):
        return {**base, "converged": False, "error": "singular fit"}

    if not res.converged or "cap_rank" not in res.params.index:
        return {
            **base,
            "converged": False,
            "error": "did not converge"
            if "cap_rank" in res.params.index
            else "cap_rank coefficient missing",
        }

    ci = res.conf_int()
    return {
        **base,
        "converged": True,
        "cap_coef": float(res.params["cap_rank"]),
        "cap_p": float(res.pvalues["cap_rank"]),
        "cap_ci": [float(ci.loc["cap_rank", 0]), float(ci.loc["cap_rank", 1])],
        "vc_repo": float(res.params["repo Var"]),
        "vc_developer": float(res.params["developer Var"]),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_session(
    n_commits: int,
    rho: float,
    sigma: float,
    tau: float,
    x0: float = 500.0,
    rng: random.Random | None = None,
    pi_pure: float = 0.0,
) -> tuple[list[float], float]:
    """
    Simulate a single session. Returns (normalised_churn_trajectory, survivor_ratio).
    Churn is drawn i.i.d. log-normal: log(x) ~ N(log(rho), sigma^2), where rho is
    the geometric churn-decay factor (mean ratio of consecutive commit churns).
    Survivor ratio uses a mixture: with prob pi_pure the session is pure-addition
    (sr=1.0); otherwise drawn from Beta calibrated to tau (non-pure mean).
    """
    if rng is None:
        rng = random.Random()

    log_x = math.log(x0)
    log_rho = math.log(rho) if rho > 0 else -0.5
    churns: list[float] = []
    for _ in range(n_commits):
        churns.append(math.exp(log_x))
        log_x = log_rho + sigma * rng.gauss(0, 1)  # stationary AR(1) in log space

    norm = [c / churns[0] if churns[0] > 0 else 0.0 for c in churns]

    # Survivor ratio: mixture of pure-addition spike and Beta component
    if pi_pure > 0.0 and rng.random() < pi_pure:
        sr = 1.0
    else:
        a = tau * 5
        b = (1 - tau) * 5
        sr = min(1.0, max(0.0, rng.betavariate(a, b)))
    return norm, sr


def run_simulation(
    params: FittedParams,
    n_synthetic: int = 500,
    session_lengths: list[int] | None = None,
) -> dict:
    """Generate synthetic sessions for Opus and Sonnet tiers."""
    rng = random.Random(RANDOM_SEED)

    results: dict[str, dict[str, list]] = {
        "opus": {"trajectories": [], "survivor_ratios": [], "n_commits": []},
        "sonnet": {"trajectories": [], "survivor_ratios": [], "n_commits": []},
    }

    for tier, tau in [("opus", params.tau_opus), ("sonnet", params.tau_sonnet)]:
        bucket = results[tier]
        for _ in range(n_synthetic):
            if session_lengths:
                n = rng.choice(session_lengths)
            else:
                n = max(1, round(rng.lognormvariate(math.log(2.5), 0.9)))
            traj, sr = simulate_session(
                n, params.rho, params.sigma, tau, rng=rng, pi_pure=params.pi_pure
            )
            bucket["trajectories"].append(traj)
            bucket["survivor_ratios"].append(sr)
            bucket["n_commits"].append(n)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURF)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.grid(axis="y", which="minor", color=GRID, linewidth=0.25, zorder=0)
    ax.set_axisbelow(True)


def fig1_churn_trajectories(sessions: list[dict], commits: list[dict]) -> None:
    """Normalised within-session churn trajectories for multi-commit sessions."""
    multi = [s for s in sessions if s["n_commits"] >= 5]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor=SURF)
    _style_axes(ax)

    Y_CAP = 3.0  # cap normalised churn; sessions with spike > cap still shown but clipped

    # Darkness encodes commit count: more commits -> darker line (grayscale).
    ncs = [s["n_commits"] for s in multi]
    n_lo, n_hi = (min(ncs), max(ncs)) if ncs else (0, 1)

    # Individual session curves — grayscale shade scaled by n_commits
    for s in multi:
        churns = session_commit_churns(s, commits)
        peak = max(churns) if churns else 0
        if peak == 0:
            continue
        norm = [min(c / peak, Y_CAP) for c in churns]
        xs = list(range(len(norm)))
        frac = (s["n_commits"] - n_lo) / (n_hi - n_lo) if n_hi > n_lo else 1.0
        g = 0.72 - 0.60 * frac  # 0=black (most commits), lighter for fewer
        ax.plot(xs, norm, color=(g, g, g), linewidth=0.6, alpha=0.55, zorder=1)

    # Compute median trajectory at each commit position (up to position 9)
    # Normalise each session by its own peak churn
    max_pos = 10
    medians: list[float | None] = []
    for pos in range(max_pos):
        vals = []
        for s in multi:
            churns = session_commit_churns(s, commits)
            peak = max(churns) if churns else 0
            if pos < len(churns) and peak > 0:
                vals.append(churns[pos] / peak)
        medians.append(statistics.median(vals) if vals else None)

    xs_med = [i for i, v in enumerate(medians) if v is not None]
    ys_med = [v for v in medians if v is not None]
    ax.plot(xs_med, ys_med, color=BLUE, linewidth=2.0, zorder=3, label="Median")
    ax.axhline(1.0, color=MUTED, linewidth=0.75, linestyle="--", zorder=2)

    ax.set_xlabel("Commit within session", fontsize=9, color=INK2)
    ax.set_ylabel("Normalised churn (session peak = 1)", fontsize=9, color=INK2)
    ax.set_title(
        "Within-session churn convergence\n(real company agentic sessions, n≥5 commits)",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.set_xlim(-0.3, max_pos - 0.7)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    _save(fig, "emp_fig1_churn_trajectories")
    print("Saved emp_fig1_churn_trajectories")


def fig2_survivor_by_tier(sessions: list[dict], commits: list[dict]) -> None:
    """Survivor ratio by model tier — Opus vs Sonnet."""
    tiers = ["sonnet", "opus"]
    data: dict[str, list[float]] = {t: [] for t in tiers}
    for s in sessions:
        if s["survivor_ratio"] is None or s["survivor_ratio"] >= 0.999:
            continue
        t = dominant_tier(s, commits)
        if t in data:
            data[t].append(s["survivor_ratio"])

    fig, ax = plt.subplots(figsize=(4.0, 3.5), facecolor=SURF)
    _style_axes(ax)

    for i, tier in enumerate(tiers):
        vals = data[tier]
        if not vals:
            continue
        color = TIER_COLOR[tier]
        n = len(vals)

        # Jitter strip
        xs = [i + 0.05 * (hash(v) % 100 - 50) / 100 for v in vals]
        ax.scatter(xs, vals, color=color, alpha=0.35, s=12, zorder=2, linewidths=0)

        # Box (IQR) with median bar; solid dot marks the MEAN (the statistic
        # reported as tau in the text, Finding 2).
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        mean = float(np.mean(vals))
        ax.vlines(i, q1, q3, color=color, linewidth=4, alpha=0.5, zorder=3)
        ax.hlines(med, i - 0.06, i + 0.06, color=color, linewidth=1.5, alpha=0.7, zorder=4)
        ax.scatter([i], [mean], color=color, s=44, zorder=5, linewidths=0)

        # Direct label — report the mean (= tau) and median so the figure and
        # the caption cite the same statistic.
        ax.text(
            i,
            q3 + 0.03,
            f"mean {mean:.2f}\nmedian {med:.2f}\nn={n}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
        )

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([TIER_LABEL[t] for t in tiers], fontsize=9)
    ax.set_ylabel("Session survivor ratio", fontsize=9, color=INK2)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Survivor ratio by model tier\n(real company sessions)", fontsize=9, color=INK, loc="left"
    )

    fig.tight_layout()
    _save(fig, "emp_fig2_survivor_by_tier")
    print("Saved emp_fig2_survivor_by_tier")


def fig3_empirical_vs_simulated(
    sessions: list[dict],
    sim_results: dict,
) -> dict:
    """Empirical survivor ratio distribution vs calibrated simulation."""
    # Filter to multi-commit sessions — single-commit sessions trivially get survivor≈1
    # and aren't relevant to refinement dynamics
    empirical = [
        s["survivor_ratio"]
        for s in sessions
        if s["survivor_ratio"] is not None and s["n_commits"] >= 2
    ]
    sim_pooled = sim_results["opus"]["survivor_ratios"] + sim_results["sonnet"]["survivor_ratios"]

    fig, ax = plt.subplots(figsize=(5.0, 3.5), facecolor=SURF)
    _style_axes(ax)

    # .tolist(): Axes.hist types bins as int | Sequence[float] | str | None, which an
    # ndarray does not satisfy. Same edges, same plot -- verified rather than assumed:
    # all four emp_fig PDFs were re-rendered after this change and rasterized at
    # 120 dpi, giving byte-identical images. The PDFs themselves differ in exactly 6
    # bytes, the CreationDate, which is why the committed figures were left alone.
    bins = np.linspace(0, 1, 21).tolist()
    ax.hist(
        empirical,
        bins=bins,
        density=True,
        color=BLUE,
        alpha=0.55,
        label=f"Empirical, n≥2 commits (n={len(empirical)})",
        zorder=2,
    )
    ax.hist(
        sim_pooled,
        bins=bins,
        density=True,
        color=GREEN,
        alpha=0.45,
        label=f"Simulated (n={len(sim_pooled)})",
        zorder=2,
    )

    ks = scipy_stats.ks_2samp(empirical, sim_pooled)

    ax.set_xlabel("Survivor ratio", fontsize=9, color=INK2)
    ax.set_ylabel("Density", fontsize=9, color=INK2)
    ax.set_title(
        "Empirical vs calibrated-simulation survivor ratio", fontsize=9, color=INK, loc="left"
    )
    ax.legend(fontsize=8, frameon=False)
    ax.text(
        0.97,
        0.95,
        f"KS={ks.statistic:.3f}, p={ks.pvalue:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK2,
    )

    fig.tight_layout()
    _save(fig, "emp_fig3_empirical_vs_simulated")
    print(f"Saved emp_fig3_empirical_vs_simulated  [KS={ks.statistic:.3f}, p={ks.pvalue:.3f}]")
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def fig4_session_length_dist(sessions: list[dict]) -> dict:
    """Session length distribution — empirical + log-normal fit."""
    ns = [s["n_commits"] for s in sessions]

    fig, ax = plt.subplots(figsize=(5.0, 3.5), facecolor=SURF)
    _style_axes(ax)

    max_n = max(ns)  # show full tail (incl. the 45-commit session cited in the text)
    bins = np.arange(0.5, max_n + 1.5, 1).tolist()
    ax.hist(ns, bins=bins, color=BLUE, alpha=0.7, density=True, zorder=2, label="Observed sessions")

    # Log-normal overlay (fit to log of ns) -- undefined for constant-length
    # data (zero log-variance); real session lengths are heavy-tailed and
    # never hit this, but the fit is skipped rather than dividing by zero.
    log_ns = [math.log(n) for n in ns]
    mu_ln = statistics.mean(log_ns)
    sigma_ln = statistics.stdev(log_ns) if len(ns) > 1 else 0.0
    if sigma_ln > 0:
        xs = np.linspace(0.5, max_n + 0.5, 200)
        pdf = np.array(
            [
                math.exp(-0.5 * ((math.log(x) - mu_ln) / sigma_ln) ** 2)
                / (x * sigma_ln * math.sqrt(2 * math.pi))
                if x > 0
                else 0
                for x in xs
            ]
        )
        ax.plot(xs, pdf, color=GREEN, linewidth=1.5, zorder=3, label="Log-normal fit")

    ax.set_xlabel("Commits per session", fontsize=9, color=INK2)
    ax.set_ylabel("Density", fontsize=9, color=INK2)
    ax.set_title(
        "Session length distribution (company agentic sessions)", fontsize=9, color=INK, loc="left"
    )
    ax.set_xlim(0, max_n + 1)
    ax.legend(fontsize=8, frameon=False)

    ax.text(
        0.97,
        0.95,
        f"n={len(ns)}  median={statistics.median(ns):.0f}  mean={statistics.mean(ns):.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK2,
    )

    ks_stat: float | None = None
    ks_p: float | None = None
    if sigma_ln > 0:
        ks = scipy_stats.kstest(
            ns, lambda x: scipy_stats.lognorm.cdf(x, s=sigma_ln, scale=math.exp(mu_ln))
        )
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        ax.text(
            0.97,
            0.55,
            f"KS={ks_stat:.3f}, p={ks_p:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK2,
        )

    fig.tight_layout()
    _save(fig, "emp_fig4_session_length_dist")
    if ks_stat is not None:
        print(f"Saved emp_fig4_session_length_dist  [KS={ks_stat:.3f}, p={ks_p:.3f}]")
    else:
        print(
            "Saved emp_fig4_session_length_dist  [log-normal fit skipped: constant session length]"
        )
    return {"ks_stat": ks_stat, "ks_p": ks_p}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover -- CLI entrypoint: loads real data, calls
    # already-unit-tested functions, prints/writes results. No branching logic
    # of its own to verify; see llm_simulations.py for the same convention.
    sessions, commits = load_data()
    print(f"Loaded {len(sessions)} sessions, {len(commits)} commits")

    params = fit_params(sessions, commits)
    print("\nFitted parameters:")
    rlo, rhi = params.rho_ci
    slo, shi = params.sigma_ci
    print(f"  rho (geometric churn-decay factor): {params.rho:.3f}  95% CI [{rlo:.3f}, {rhi:.3f}]")
    print(f"  sigma (log noise): {params.sigma:.3f}  95% CI [{slo:.3f}, {shi:.3f}]")
    print(f"  tau (Opus):        {params.tau_opus:.3f}")
    print(f"  tau (Sonnet):      {params.tau_sonnet:.3f}")
    print(f"  tau (pooled):      {params.tau_pooled:.3f}")
    print(f"  pi_pure:           {params.pi_pure:.3f}")
    print(f"  sessions used:     {params.n_sessions_fit}")

    obs_lengths = [s["n_commits"] for s in sessions]
    sim_results = run_simulation(params, n_synthetic=500, session_lengths=obs_lengths)

    print("\nGenerating figures...")
    fig1_churn_trajectories(sessions, commits)
    fig2_survivor_by_tier(sessions, commits)
    fig3_stats = fig3_empirical_vs_simulated(sessions, sim_results)
    fig4_stats = fig4_session_length_dist(sessions)

    # Permutation test + bootstrap CI for the Opus/Sonnet gap (incl. MWU cross-check)
    gap_stats = survivor_gap_stats(sessions, commits)
    print("\nOpus vs Sonnet gap statistics:")
    if "error" in gap_stats:
        print(f"  {gap_stats['error']}")
    else:
        sig = "SIGNIFICANT" if gap_stats["significant"] else "not significant"
        print(
            f"  gap (Opus - Sonnet):  {gap_stats['gap']:+.3f}"
            f"  [{sig}, perm p={gap_stats['p_value']:.4f}, MWU p={gap_stats['mwu_p']:.4f}]"
        )
        lo_o, hi_o = gap_stats["ci_opus_95"]
        lo_s, hi_s = gap_stats["ci_sonnet_95"]
        lo_g, hi_g = gap_stats["ci_gap_95"]
        print(f"  Opus  95% CI:         [{lo_o:.3f}, {hi_o:.3f}]  (n={gap_stats['n_opus']})")
        print(f"  Sonnet 95% CI:        [{lo_s:.3f}, {hi_s:.3f}]  (n={gap_stats['n_sonnet']})")
        print(f"  Gap 95% CI:           [{lo_g:+.3f}, {hi_g:+.3f}]")

    # Churn-decay permutation null baseline
    decay_stats = churn_decay_permutation(sessions, commits)
    print("\nChurn-decay permutation test:")
    if "error" in decay_stats:
        print(f"  {decay_stats['error']}")
    else:
        print(
            f"  median late/early ratio: {decay_stats['observed_stat']:.3f}"
            f"  p={decay_stats['p_value']:.4f}"
            f"  (n={decay_stats['n_sessions']} sessions)"
        )

    decay_ci = churn_ratio_bootstrap_ci(sessions, commits)
    if "error" not in decay_ci:
        print(f"  95% CI (bootstrap):      [{decay_ci['ci_low']:.3f}, {decay_ci['ci_high']:.3f}]")

    obs_ratios, _ = _collect_late_early_ratios(sessions, commits)
    decay_pct = (
        100 * sum(1 for r in obs_ratios if r < 1.0) / len(obs_ratios) if obs_ratios else None
    )
    if decay_pct is not None:
        print(f"  sessions with decreasing churn: {decay_pct:.1f}% (of n={len(obs_ratios)})")

    # --- De-confounding robustness checks for Finding 2 (tier gap) ---
    gap_evalue = gap_sensitivity_evalue(sessions, commits)
    print("\nTier-gap sensitivity (E-value):")
    if "error" in gap_evalue:
        print(f"  {gap_evalue['error']}")
    else:
        print(
            f"  Cohen's d={gap_evalue['cohens_d']:.3f}, RR~={gap_evalue['rr_approx']:.3f}"
            f"  E-value(point)={gap_evalue['evalue_point']:.2f}"
            f", E-value(CI)={gap_evalue['evalue_ci']:.2f}"
        )

    stratified_gap = stratified_gap_stats(sessions, commits)
    print("\nTask-type-stratified gap (commit-type proxy):")
    if "error" in stratified_gap:
        print(f"  {stratified_gap['error']}")
    else:
        print(
            f"  naive gap={stratified_gap['naive_gap']:+.3f}"
            f"  adjusted gap={stratified_gap['adjusted_gap']:+.3f}"
            f"  perm p={stratified_gap['p_value']:.4f}"
            f"  ({stratified_gap['n_strata_used']} strata)"
        )

    tier_decay = tier_decay_stats(sessions, commits)
    print("\nPer-tier churn-decay rho (additivity-robust discriminator):")
    if "error" in tier_decay:
        print(f"  {tier_decay['error']}")
    else:
        print(
            f"  rho_Opus={tier_decay['rho_opus']:.3f}  rho_Sonnet={tier_decay['rho_sonnet']:.3f}"
            f"  gap={tier_decay['rho_gap']:+.3f}  perm p={tier_decay['p_value']:.4f}"
        )

    mixed_model = mixed_model_gap_stats(sessions, commits)
    print("\nMultilevel model (tier + task_type fixed, repo/developer crossed random):")
    if not mixed_model["converged"]:
        print(
            f"  {mixed_model['error']} (n_sessions={mixed_model['n_sessions']},"
            f" n_repos={mixed_model['n_repos']}, n_developers={mixed_model['n_developers']})"
        )
    else:
        lo, hi = mixed_model["tier_ci"]
        print(
            f"  tier (Opus) coef={mixed_model['tier_coef']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
            f"  p={mixed_model['tier_p']:.4f}"
        )
        print(
            f"  vc_repo={mixed_model['vc_repo']:.4f}"
            f"  vc_developer={mixed_model['vc_developer']:.4f}"
            f"  (n_sessions={mixed_model['n_sessions']}, n_repos={mixed_model['n_repos']},"
            f" n_developers={mixed_model['n_developers']})"
        )

    # Out-of-sample validation of survivor-ratio distribution
    oos_stats = out_of_sample_fit(sessions, commits)
    print("\nOut-of-sample KS test (survivor ratio):")
    if "error" in oos_stats:
        print(f"  {oos_stats['error']}")
    else:
        print(
            f"  KS stat={oos_stats['ks_stat']:.3f}, p={oos_stats['ks_p']:.3f}"
            f"  (train n={oos_stats['n_train']}, test n={oos_stats['n_test']})"
        )

    # Persist fitted params for paper
    out = {
        "rho": params.rho,
        "rho_ci": list(params.rho_ci),
        "sigma": params.sigma,
        "sigma_ci": list(params.sigma_ci),
        "tau_opus": params.tau_opus,
        "tau_sonnet": params.tau_sonnet,
        "tau_pooled": params.tau_pooled,
        "pi_pure": params.pi_pure,
        "n_sessions": len(sessions),
        "n_commits": len(commits),
        "n_sessions_fit": params.n_sessions_fit,
        "gap_stats": gap_stats,
        "gap_evalue": gap_evalue,
        "stratified_gap": stratified_gap,
        "tier_decay": tier_decay,
        "mixed_model": mixed_model,
        "decay_stats": decay_stats,
        "decay_ci": decay_ci,
        "decay_pct": decay_pct,
        "oos_stats": oos_stats,
        "fig3_ks": fig3_stats,
        "fig4_ks": fig4_stats,
    }
    Path("data/company_params.json").write_text(json.dumps(out, indent=2))
    print("\nWrote data/company_params.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
