"""
part_c_control_stats.py
-----------------------
Compute every Part C statistic that the paper reports but no script produced.

Before this existed, tau_human, the two human-versus-AI gaps, the two multilevel
coefficients, the control cohort's counts, and the churn-decay fit were derived ad
hoc and hand-transcribed into ``stats_macros.tex`` and into assertions in
``test_company.py``. That is ~25 numbers copied by hand every time Part C is
re-derived, which is where transcription errors live -- and it is why a stale
``statPCPermCI`` survived a full re-derivation of the dataset.

This writes ``data/part_c_control_stats.json``, from which
``scripts/generate_stats_macros.py`` formats the macros, so the paper and the tests
read from one computed source.

The control cohort's developer labels are **not** namespaced here. Both datasets
are extracted with one salt and one curated roster, so the same person carries the
same ``dev_*`` label in both eras; prefixing would split them into two
crossed-random-effect levels and discard the only within-person signal available
against the era confound. Repository labels *are* namespaced -- they are positional,
so ``repo_A`` denotes a different repository in each file.

Usage
-----
    python3 scripts/part_c_control_stats.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import company_simulations as sim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATADIR = REPO / "data"
AI_PATH = DATADIR / "company_sessions.json"
CONTROL_PATH = DATADIR / "company_control_sessions.json"
OUT_PATH = DATADIR / "part_c_control_stats.json"
WITHIN_AI_PATH = DATADIR / "company_within_era_ai_sessions.json"
WITHIN_NONAI_PATH = DATADIR / "company_within_era_nonai_sessions.json"


def _non_pure(sessions: list[dict]) -> list[float]:
    return [
        s["survivor_ratio"]
        for s in sessions
        if s["survivor_ratio"] is not None and s["survivor_ratio"] < 0.999
    ]


def _strata(rows: list[dict]) -> dict:
    """Session-length composition for one cohort.

    The control cohort is far more singleton-heavy than the AI tier, so the gap is
    reported both over all sessions and restricted to multi-commit ones rather than
    leaving a reviewer to wonder whether length explains it.
    """
    multi = [s for s in rows if s["n_commits"] >= 2]
    nonpure_all = _non_pure(rows)
    nonpure_multi = _non_pure(multi)
    return {
        "n_sessions": len(rows),
        "single_commit_pct": 100.0 * sum(1 for s in rows if s["n_commits"] == 1) / len(rows),
        "tau_all": statistics.mean(nonpure_all) if nonpure_all else None,
        "tau_multi": statistics.mean(nonpure_multi) if nonpure_multi else None,
    }


def _paired_blocks(
    base_sessions: list[dict],
    base_commits: list[dict],
    ai_sessions: list[dict],
    ai_commits: list[dict],
) -> dict:
    """Within-developer pairing at each commit floor, minus the per-developer detail.

    Used for both the pre-AI control pairing and the within-era pairing, which had
    grown identical copies of this comprehension.
    """
    return {
        f"floor_{floor}": {
            k: v
            for k, v in sim.paired_developer_stats(
                base_sessions,
                base_commits,
                ai_sessions,
                ai_commits,
                min_commits_per_side=floor,
            ).items()
            if k != "per_developer"
        }
        for floor in (0, 15, 30, 50)
    }


def _model_versions_block(ai_commits: list[dict]) -> dict:
    """Version inventory behind the two tier labels.

    The paper quotes both how many trailer strings each label collapses and how many
    released versions those are, and it quoted them as hand-counted words until now --
    the last Part C numbers outside the generator, and the ones most likely to drift
    silently, since a cohort refresh can add a model version without changing any
    other statistic. ``sim.model_cell`` owns the parsing, including collapsing the
    "(1M context)" suffix (the same version at a longer context window, not a
    separate capability level).
    """
    inventory: dict[str, dict[str, set]] = {}
    for c in ai_commits:
        trailer = c.get("model")
        cell = sim.model_cell(trailer)
        if cell is None:
            continue
        family, version = cell
        entry = inventory.setdefault(family, {"trailers": set(), "versions": set()})
        entry["trailers"].add(trailer)
        entry["versions"].add(version)
    return {
        family: {
            "n_trailer_strings": len(entry["trailers"]),
            "versions": sorted(entry["versions"]),
        }
        for family, entry in sorted(inventory.items())
    }


def _release_block(released: dict[str, dict]) -> dict:
    """How many distinct people appear anywhere in the published datasets.

    Deliberately *not* the control cohort's headcount (``control.n_developers``),
    which the Responsible Use Statement used to quote: the AI-era cohorts contain
    people the pre-AI window never saw, so the control count understates who is
    actually in the release. Developer labels are stable HMAC hashes across cohorts,
    so a union over labels is a union over people. Commits are scanned as well as
    sessions because a developer can survive into a commit whose session was filtered
    out, and the release exposes both tables.
    """
    developers: set[str] = set()
    for data in released.values():
        for row in (*data["sessions"], *data["commits"]):
            label = row.get("developer")
            if label:
                developers.add(label)
    return {
        "datasets": sorted(released),
        "n_datasets": len(released),
        "n_developers": len(developers),
    }


def _capability_block(ai: dict, model_versions: dict) -> dict:
    """Capability-ordering results, plus how the cells break down by family.

    Both orderings are reported, each the other's sensitivity analysis.

    The composition sub-block exists because without it the appendix invites a wrong
    reading: it says five Opus and three Sonnet versions are pooled, then gives the
    cell count, and 5 + 3 happens to equal the total -- so a reader reconstructs "the
    cells are exactly the Opus and Sonnet versions" and is wrong twice over. A family
    whose newest version is the plurality cell of no session contributes no cell, and
    Haiku contributes one the tier contrast never sees.
    """
    block: dict = {
        ordering: {
            "trend": {
                k: v
                for k, v in sim.capability_trend_stats(
                    ai["sessions"], ai["commits"], ordering=ordering
                ).items()
                if k != "per_cell"
            },
            "mixed": sim.capability_mixed_model_stats(
                ai["sessions"], ai["commits"], ordering=ordering
            ),
        }
        for ordering in sim.CAPABILITY_ORDERINGS
    }
    per_cell = sim.capability_trend_stats(ai["sessions"], ai["commits"], ordering="family")[
        "per_cell"
    ]
    block["per_cell"] = per_cell

    versions_by_family = {
        family: sorted(entry["versions"]) for family, entry in model_versions.items()
    }
    block["composition"] = {
        "cells_by_family": dict(sorted(Counter(c["family"] for c in per_cell).items())),
        # Versions present in the cohort's trailers but carried by no session's plurality
        # cell, so absent from the ordered scale. Named so the appendix can say which.
        "versions_without_a_cell": {
            family: [
                v
                for v in versions
                if not any(c["family"] == family and c["version"] == v for c in per_cell)
            ]
            for family, versions in versions_by_family.items()
        },
    }
    return block


def compute() -> dict:
    ai = json.loads(AI_PATH.read_text())
    ctrl = json.loads(CONTROL_PATH.read_text())
    ai_s, ai_c = sim.namespace_repos(ai, "ai")
    ctrl_s, ctrl_c = sim.namespace_repos(ctrl, "ctrl")
    sessions, commits = ai_s + ctrl_s, ai_c + ctrl_c

    # Every dataset this repo publishes, keyed by released filename. The release
    # footprint below is a union over these, so adding a dataset to the release
    # must mean adding it here or the reported headcount silently understates it.
    released: dict[str, dict] = {AI_PATH.name: ai, CONTROL_PATH.name: ctrl}

    human = [s for s in sessions if sim.dominant_tier(s, commits) == "human"]
    human_nonpure = _non_pure(human)

    out: dict = {
        "control": {
            "since": ctrl.get("control_since"),
            "until": ctrl.get("control_until"),
            "n_repos": len(ctrl["repos"]),
            "n_developers": len({s.get("developer") for s in ctrl["sessions"]}),
            "n_sessions": len(ctrl["sessions"]),
            "n_commits": len(ctrl["commits"]),
            "tau_human": statistics.mean(human_nonpure),
            "n_human_nonpure": len(human_nonpure),
        },
        "ai_tier": {
            "n_repos": len(ai["repos"]),
            "n_developers": len({s.get("developer") for s in ai["sessions"]}),
            "n_sessions": len(ai["sessions"]),
            "n_commits": len(ai["commits"]),
        },
    }

    # Churn-decay fit on the control cohort alone, for the shape comparison
    # ("is geometric front-loading AI-specific?").
    ctrl_fit = sim.fit_params(ctrl["sessions"], ctrl["commits"])
    out["control"]["rho"] = ctrl_fit.rho
    out["control"]["rho_ci"] = list(ctrl_fit.rho_ci)
    out["control"]["sigma"] = ctrl_fit.sigma

    for tier in ("opus", "sonnet"):
        gap = sim.survivor_gap_stats(sessions, commits, tiers=("human", tier))
        mm = sim.mixed_model_gap_stats(sessions, commits, tiers=("human", tier))
        out[f"human_vs_{tier}"] = {
            "gap": gap.get("gap"),
            "p_value": gap.get("p_value"),
            "tau_tier": gap.get(f"tau_{tier}_mean"),
            "n_tier": gap.get(f"n_{tier}"),
            "n_human": gap.get("n_human"),
            "mm_coef": mm.get("tier_coef"),
            "mm_p": mm.get("tier_p"),
            "mm_ci": mm.get("tier_ci"),
            "mm_vc_repo": mm.get("vc_repo"),
            "mm_vc_developer": mm.get("vc_developer"),
            "mm_converged": mm.get("converged"),
        }

    out["composition"] = {
        "control": _strata(ctrl["sessions"]),
        "ai_tier": _strata(ai["sessions"]),
    }

    # Tier composition of the AI-tier cohort. Finding 2 contrasts Opus against Sonnet,
    # but those two labels do not exhaust the cohort: a small Haiku residue and a few
    # sessions whose trailer does not resolve to a tier are dropped from the contrast.
    # The appendix already reports the Haiku cell through the capability block, so the
    # main text has to name the residue rather than imply only two tiers exist.
    tier_counts = Counter(sim.dominant_tier(s, ai["commits"]) for s in ai["sessions"])
    contrasted = tier_counts["opus"] + tier_counts["sonnet"]
    out["tier_composition"] = {
        "n_sessions": len(ai["sessions"]),
        "by_tier": dict(sorted(tier_counts.items())),
        "n_contrasted": contrasted,
        "n_excluded": len(ai["sessions"]) - contrasted,
    }

    # Within-developer pairing: only meaningful because both cohorts now share a
    # salt and a roster. Direction-only by construction -- see paired_developer_stats.
    out["paired_within_developer"] = _paired_blocks(
        ctrl["sessions"], ctrl["commits"], ai["sessions"], ai["commits"]
    )

    # Within-era contrast: the same people, the same window, split only on whether
    # a commit carries a Claude trailer. This is the one design here that removes
    # both the person and the era confounds; what remains is task selection.
    ai_era = WITHIN_AI_PATH
    nonai_era = WITHIN_NONAI_PATH
    if ai_era.exists() and nonai_era.exists():
        wa = json.loads(ai_era.read_text())
        wn = json.loads(nonai_era.read_text())
        released[ai_era.name] = wa
        released[nonai_era.name] = wn
        out["within_era"] = {
            "since": wa.get("within_era_since"),
            "until": wa.get("within_era_until"),
            "ai": {
                "n_sessions": len(wa["sessions"]),
                "n_commits": len(wa["commits"]),
                "n_developers": len({s["developer"] for s in wa["sessions"]}),
                "tau": statistics.mean(_non_pure(wa["sessions"])),
            },
            "nonai": {
                "n_sessions": len(wn["sessions"]),
                "n_commits": len(wn["commits"]),
                "n_developers": len({s["developer"] for s in wn["sessions"]}),
                "tau": statistics.mean(_non_pure(wn["sessions"])),
            },
            "paired": _paired_blocks(wn["sessions"], wn["commits"], wa["sessions"], wa["commits"]),
        }

    # Version inventory behind the two tier labels. The paper quotes both how many
    # trailer strings each label collapses and how many released versions those are,
    # and it quoted them as hand-counted words until now -- the last Part C numbers
    # outside the generator, and the ones most likely to drift silently, since a
    # cohort refresh can add a model version without changing any other statistic.
    # sim.model_cell owns the parsing, including collapsing the "(1M context)" suffix
    # (the same version at a longer context window, not a separate capability level).
    out["model_versions"] = _model_versions_block(ai["commits"])

    # Release footprint: how many distinct people appear anywhere in the published
    # datasets. This is deliberately *not* the control cohort's headcount
    # (``control.n_developers``), which the Responsible Use Statement used to quote:
    # the AI-era cohorts contain people the pre-AI window never saw, so the control
    # count understates who is actually in the release. Developer labels are stable
    # HMAC hashes across cohorts, so a union over labels is a union over people.
    # Commits are scanned as well as sessions because a developer can survive into a
    # commit whose session was filtered out, and the release exposes both tables.
    out["release"] = _release_block(released)

    out["capability"] = _capability_block(ai, out["model_versions"])

    return out


def main() -> None:
    stats = compute()
    OUT_PATH.write_text(json.dumps(stats, indent=2))
    c = stats["control"]
    print(
        f"Control: {c['n_repos']} repos, {c['n_developers']} developers, "
        f"{c['n_sessions']} sessions, {c['n_commits']} commits\n"
        f"  tau_human = {c['tau_human']:.4f} (n={c['n_human_nonpure']} non-pure)\n"
        f"  rho_human = {c['rho']:.4f}",
        file=sys.stderr,
    )
    for tier in ("opus", "sonnet"):
        g = stats[f"human_vs_{tier}"]
        print(
            f"  vs {tier}: gap={g['gap']:+.4f} p={g['p_value']} "
            f"adjusted={g['mm_coef']:+.4f} p={g['mm_p']:.4g}",
            file=sys.stderr,
        )
    print(f"\nWrote {OUT_PATH.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
