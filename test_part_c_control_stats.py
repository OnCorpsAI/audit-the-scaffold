"""Tests for scripts/part_c_control_stats.py.

This module is thin assembly over ``company_simulations``: it loads the cohort JSON
caches, calls the heavy estimators, and shapes the result into the dict that
``scripts/generate_stats_macros.py`` formats into paper macros. The estimators
themselves are tested in test_company.py (94% covered there), so every ``sim.*``
call is stubbed here and what gets asserted is *this* module's own logic --
which tier is pooled into which statistic, which keys survive into the payload,
and which optional block appears.

That split matters because the bugs this file can actually have are assembly bugs,
and they are silent: a tau computed over the wrong tier subset, or a payload that
quietly drops a key the macro generator reads, still produces a plausible number.

The negative tests are the point:
  - ``tau_human`` must pool *only* human-tier non-pure sessions. The fixtures are
    built so that pooling any wider (all tiers) or any narrower (dropping the
    non-pure filter) yields a visibly different mean, so a regression cannot pass.
  - ``survivor_ratio >= 0.999`` is "pure" and must be excluded; 0.999 exactly is
    the boundary and must be *out*, not in.
  - ``per_developer`` must be stripped from every paired block. It is a per-person
    payload that has no business in a committed stats file, so its presence is a
    privacy regression, not a formatting one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import company_simulations as sim
import scripts.part_c_control_stats as pcs

# ---------------------------------------------------------------------------
# Fixtures. survivor_ratio values are chosen so the tier subsets have distinct
# means: human non-pure = mean(0.2, 0.4) = 0.3, opus non-pure = 0.8.
# ---------------------------------------------------------------------------


def _session(
    ratio: float | None, n_commits: int, developer: str, tier: str = "human"
) -> dict[str, Any]:
    return {
        "survivor_ratio": ratio,
        "n_commits": n_commits,
        "developer": developer,
        "tier": tier,
        "repo": "repo_A",
    }


def _cohort(sessions: list[dict], **extra: Any) -> dict[str, Any]:
    return {
        "repos": ["r1", "r2"],
        "sessions": sessions,
        "commits": [{"session": i} for i in range(len(sessions))],
        **extra,
    }


CONTROL_SESSIONS = [
    _session(0.2, 3, "dev_a"),  # human, non-pure, multi-commit
    _session(0.4, 2, "dev_b"),  # human, non-pure, multi-commit
    _session(1.0, 4, "dev_c"),  # human but PURE -> excluded from tau
    _session(None, 1, "dev_d"),  # no ratio -> excluded
    _session(0.999, 1, "dev_e"),  # boundary: pure, excluded
]
AI_SESSIONS = [
    _session(0.8, 5, "dev_a", tier="opus"),
    _session(0.9, 1, "dev_b", tier="opus"),
]

HUMAN_NONPURE_MEAN = 0.3  # mean(0.2, 0.4)

# Stand-in for capability_trend_stats' per_cell payload. Deliberately mirrors the real
# cohort's awkward shape: three families, and an opus version (4.9) that appears in the
# trailers but is the plurality cell of no session, so it must show up as a version
# without a cell rather than being silently counted.
PER_CELL = [
    {"family": "haiku", "version": 4.5, "n_sessions": 3, "tau": 0.35},
    {"family": "sonnet", "version": 4.5, "n_sessions": 8, "tau": 0.52},
    {"family": "sonnet", "version": 5.0, "n_sessions": 6, "tau": 0.45},
    {"family": "opus", "version": 4.8, "n_sessions": 9, "tau": 0.56},
]


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace every heavy estimator with a recorder. Returns the call log."""
    calls: dict[str, list] = {"namespace": [], "paired": [], "trend": [], "cap_mixed": []}

    def _namespace(data: dict, prefix: str) -> tuple[list[dict], list[dict]]:
        calls["namespace"].append(prefix)
        return data["sessions"], data["commits"]

    # _sa/_ca are underscore-prefixed because the stub only exists to match
    # paired_developer_stats' positional signature; only the keyword is recorded.
    def _paired(_sa: list, _ca: list, sb: list, cb: list, *, min_commits_per_side: int = 0) -> dict:
        calls["paired"].append(min_commits_per_side)
        # `per_developer` is per-person data the caller must strip.
        return {"n_pairs": 2, "direction": "b_lower", "per_developer": {"dev_a": 0.1}}

    def _trend(sessions: list, commits: list, *, ordering: str = "family", **kw: Any) -> dict:
        calls["trend"].append(ordering)
        # per_cell entries must mirror the real capability_trend_stats contract
        # (family/version/n_sessions/tau) -- compute() reads `family` and `version`
        # off them to derive the cell composition, so a placeholder shape here would
        # only prove the stub is self-consistent.
        return {"rho_spearman": 0.5, "ordering": ordering, "per_cell": list(PER_CELL)}

    def _cap_mixed(sessions: list, commits: list, *, ordering: str = "family", **kw: Any) -> dict:
        calls["cap_mixed"].append(ordering)
        return {"rank_coef": 0.02, "converged": True}

    monkeypatch.setattr(pcs.sim, "namespace_repos", _namespace)
    monkeypatch.setattr(pcs.sim, "dominant_tier", lambda s, commits: s["tier"])
    monkeypatch.setattr(
        pcs.sim,
        "fit_params",
        lambda sessions, commits: sim.FittedParams(
            rho=0.55,
            sigma=0.31,
            tau_opus=0.8,
            tau_sonnet=0.7,
            tau_pooled=0.75,
            session_n_median=3.0,
            session_n_mean=3.5,
            n_sessions_fit=len(sessions),
            rho_ci=(0.5, 0.6),
        ),
    )
    monkeypatch.setattr(
        pcs.sim,
        "survivor_gap_stats",
        lambda s, c, *, tiers: {
            "gap": -0.5,
            "p_value": 0.01,
            f"tau_{tiers[1]}_mean": 0.8,
            f"n_{tiers[1]}": 7,
            "n_human": 11,
        },
    )
    monkeypatch.setattr(
        pcs.sim,
        "mixed_model_gap_stats",
        lambda s, c, *, tiers: {
            "tier_coef": -0.42,
            "tier_p": 0.002,
            "tier_ci": [-0.6, -0.2],
            "vc_repo": 0.01,
            "vc_developer": 0.02,
            "converged": True,
        },
    )
    monkeypatch.setattr(pcs.sim, "paired_developer_stats", _paired)
    monkeypatch.setattr(pcs.sim, "capability_trend_stats", _trend)
    monkeypatch.setattr(pcs.sim, "capability_mixed_model_stats", _cap_mixed)
    return calls


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list]:
    """Point the module at fixture JSON and stub the estimators. No within-era files."""
    ai = tmp_path / "ai.json"
    ctrl = tmp_path / "ctrl.json"
    ai.write_text(json.dumps(_cohort(AI_SESSIONS)))
    ctrl.write_text(
        json.dumps(_cohort(CONTROL_SESSIONS, control_since="2024-01", control_until="2024-06"))
    )
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", tmp_path / "absent_ai.json")
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", tmp_path / "absent_nonai.json")
    return _install_stubs(monkeypatch)


# ---------------------------------------------------------------------------
# _non_pure: the "pure session" exclusion rule and its boundary
# ---------------------------------------------------------------------------


def test_non_pure_drops_none_and_pure_sessions() -> None:
    rows = [_session(0.2, 1, "d"), _session(None, 1, "d"), _session(1.0, 1, "d")]
    assert pcs._non_pure(rows) == [0.2]


def test_non_pure_excludes_exactly_the_threshold() -> None:
    """0.999 is pure and must be out; anything below it is in. A `<=` typo in the
    comparison would flip this and silently pull pure sessions into every tau."""
    assert pcs._non_pure([_session(0.999, 1, "d")]) == []
    assert pcs._non_pure([_session(0.9989, 1, "d")]) == [0.9989]


# ---------------------------------------------------------------------------
# compute(): tier pooling
# ---------------------------------------------------------------------------


def test_tau_human_pools_only_human_non_pure_sessions(wired: dict[str, list]) -> None:
    """The fixtures make every wrong subset detectable: pooling all tiers would
    include the opus 0.8/0.9 rows and raise the mean well above 0.3, and dropping
    the non-pure filter would pull in 1.0 and 0.999."""
    out = pcs.compute()
    assert out["control"]["tau_human"] == pytest.approx(HUMAN_NONPURE_MEAN)
    assert out["control"]["n_human_nonpure"] == 2


def test_cohort_counts_come_from_the_raw_cohort_not_the_merged_pool(
    wired: dict[str, list],
) -> None:
    """n_sessions/n_commits must describe each cohort alone. Reading them off the
    merged pool would report the same total twice."""
    out = pcs.compute()
    assert out["control"]["n_sessions"] == len(CONTROL_SESSIONS)
    assert out["ai_tier"]["n_sessions"] == len(AI_SESSIONS)
    assert out["control"]["n_repos"] == 2
    assert out["control"]["since"] == "2024-01"
    assert out["control"]["until"] == "2024-06"


def test_repos_are_namespaced_per_cohort(wired: dict[str, list]) -> None:
    """Repo labels are positional, so both cohorts must be namespaced with distinct
    prefixes before merging -- otherwise unrelated repos fuse into one level."""
    pcs.compute()
    assert wired["namespace"] == ["ai", "ctrl"]


def test_control_churn_fit_is_recorded(wired: dict[str, list]) -> None:
    out = pcs.compute()
    assert out["control"]["rho"] == pytest.approx(0.55)
    assert out["control"]["rho_ci"] == [0.5, 0.6]
    assert out["control"]["sigma"] == pytest.approx(0.31)


# ---------------------------------------------------------------------------
# compute(): composition strata
# ---------------------------------------------------------------------------


def test_single_commit_pct_and_multi_commit_tau(wired: dict[str, list]) -> None:
    """tau_multi must restrict to n_commits >= 2. In the control fixture the only
    non-pure multi-commit rows are 0.2 and 0.4, so tau_all and tau_multi coincide
    at 0.3; in the AI fixture the 0.9 row is a singleton, so tau_multi = 0.8 while
    tau_all = 0.85 -- a test that would pass under a broken filter only if the
    filter were ignored entirely."""
    comp = pcs.compute()["composition"]
    assert comp["control"]["single_commit_pct"] == pytest.approx(100.0 * 2 / 5)
    assert comp["control"]["tau_multi"] == pytest.approx(0.3)
    assert comp["ai_tier"]["tau_all"] == pytest.approx(0.85)
    assert comp["ai_tier"]["tau_multi"] == pytest.approx(0.8)


def test_tau_multi_is_none_when_no_multi_commit_session_is_non_pure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The `else None` branch: a cohort whose only multi-commit sessions are pure
    must report None rather than raising on an empty mean."""
    rows = [_session(0.3, 1, "d1"), _session(1.0, 4, "d2")]
    ai = tmp_path / "ai.json"
    ctrl = tmp_path / "ctrl.json"
    ai.write_text(json.dumps(_cohort(rows)))
    ctrl.write_text(json.dumps(_cohort(rows, control_since="x", control_until="y")))
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", tmp_path / "no.json")
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", tmp_path / "no2.json")
    _install_stubs(monkeypatch)
    comp = pcs.compute()["composition"]
    assert comp["control"]["tau_multi"] is None
    assert comp["control"]["tau_all"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute(): payload hygiene and the optional within-era block
# ---------------------------------------------------------------------------


def test_paired_blocks_strip_per_developer(wired: dict[str, list]) -> None:
    """per_developer is per-person data; it must never reach the committed stats
    file. Every floor block must drop it while keeping the aggregate keys."""
    paired = pcs.compute()["paired_within_developer"]
    assert set(paired) == {"floor_0", "floor_15", "floor_30", "floor_50"}
    for block in paired.values():
        assert "per_developer" not in block
        assert block["n_pairs"] == 2
    assert wired["paired"] == [0, 15, 30, 50]


def test_within_era_block_absent_when_either_cache_is_missing(wired: dict[str, list]) -> None:
    assert "within_era" not in pcs.compute()


def test_within_era_block_present_when_both_caches_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The within-era contrast is the only design free of both confounds, so its
    presence is conditional on both halves existing -- never on one."""
    ai = tmp_path / "ai.json"
    ctrl = tmp_path / "ctrl.json"
    ai.write_text(json.dumps(_cohort(AI_SESSIONS)))
    ctrl.write_text(json.dumps(_cohort(CONTROL_SESSIONS, control_since="a", control_until="b")))
    wa = tmp_path / "wa.json"
    wn = tmp_path / "wn.json"
    wa.write_text(
        json.dumps(
            _cohort(
                [_session(0.5, 2, "dev_a", tier="opus")],
                within_era_since="2025-01",
                within_era_until="2025-06",
            )
        )
    )
    wn.write_text(json.dumps(_cohort([_session(0.7, 2, "dev_a")])))
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", wa)
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", wn)
    calls = _install_stubs(monkeypatch)

    we = pcs.compute()["within_era"]
    assert we["since"] == "2025-01"
    assert we["until"] == "2025-06"
    assert we["ai"]["tau"] == pytest.approx(0.5)
    assert we["nonai"]["tau"] == pytest.approx(0.7)
    assert we["ai"]["n_developers"] == 1
    for block in we["paired"].values():
        assert "per_developer" not in block
    # 4 floors for the cross-era pairing + 4 for the within-era pairing.
    assert calls["paired"] == [0, 15, 30, 50, 0, 15, 30, 50]


# ---------------------------------------------------------------------------
# compute(): release footprint
#
# This block backs the Responsible Use Statement's privacy claim, which used to
# quote the *control* cohort's headcount. That undercounts: the AI-era cohorts
# contain people the pre-AI window never saw. The tests below fix the union
# semantics in place, since the failure mode is a plausible-looking smaller number.
# ---------------------------------------------------------------------------


def _dev_cohort(sessions: list[dict], commit_devs: list[str]) -> dict[str, Any]:
    """A cohort whose commits carry their own developer labels."""
    return {
        "repos": ["r1"],
        "sessions": sessions,
        "commits": [{"session": i, "developer": d} for i, d in enumerate(commit_devs)],
    }


def _wire_four(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, within: bool) -> dict[str, list]:
    """Wire all four datasets (or just the two mandatory ones) with disjoint rosters."""
    ai = tmp_path / "ai.json"
    ctrl = tmp_path / "ctrl.json"
    ai.write_text(json.dumps(_dev_cohort([_session(0.8, 2, "dev_a", tier="opus")], ["dev_a"])))
    ctrl.write_text(
        json.dumps(
            {
                **_dev_cohort([_session(0.2, 2, "dev_b")], ["dev_b"]),
                "control_since": "2024-01",
                "control_until": "2024-06",
            }
        )
    )
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    wa, wn = tmp_path / "wa.json", tmp_path / "wn.json"
    if within:
        wa.write_text(
            json.dumps(
                {
                    **_dev_cohort([_session(0.5, 2, "dev_c", tier="opus")], ["dev_c"]),
                    "within_era_since": "2025-01",
                    "within_era_until": "2025-06",
                }
            )
        )
        wn.write_text(json.dumps(_dev_cohort([_session(0.7, 2, "dev_d")], ["dev_d"])))
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", wa)
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", wn)
    return _install_stubs(monkeypatch)


def test_release_footprint_unions_developers_across_all_four_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each of the four datasets contributes one developer nobody else has, so the
    union must be 4. Counting any single cohort -- including the control, which is
    what the statement used to quote -- yields 1 and fails."""
    _wire_four(monkeypatch, tmp_path, within=True)
    rel = pcs.compute()["release"]
    assert rel["n_developers"] == 4
    assert rel["n_datasets"] == 4
    # Recorded by released filename, sorted, so the audit trail names what was counted.
    assert rel["datasets"] == ["ai.json", "ctrl.json", "wa.json", "wn.json"]


def test_release_footprint_shrinks_when_a_dataset_is_not_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The negative proof for the union: drop the two within-era files and the count
    must fall to 2. A release block that hardcoded the cohort list, or read only the
    control cohort, would report the same number either way."""
    _wire_four(monkeypatch, tmp_path, within=False)
    rel = pcs.compute()["release"]
    assert rel["n_datasets"] == 2
    assert rel["n_developers"] == 2


def test_release_footprint_counts_developers_present_only_in_commits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A developer can survive into a released commit whose session was filtered out
    -- this is true of the real AI-tier dataset (21 developers in commits, 20 in
    sessions). The release exposes both tables, so both must be scanned."""
    ai = tmp_path / "ai.json"
    ctrl = tmp_path / "ctrl.json"
    # dev_ghost appears in a commit but in no session.
    ai.write_text(
        json.dumps(_dev_cohort([_session(0.8, 2, "dev_a", tier="opus")], ["dev_a", "dev_ghost"]))
    )
    ctrl.write_text(
        json.dumps(
            {
                **_dev_cohort([_session(0.2, 2, "dev_b")], ["dev_b"]),
                "control_since": "a",
                "control_until": "b",
            }
        )
    )
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", tmp_path / "absent_a.json")
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", tmp_path / "absent_n.json")
    _install_stubs(monkeypatch)

    out = pcs.compute()
    assert out["release"]["n_developers"] == 3
    # And it must exceed the control cohort's own headcount, which is the whole point.
    assert out["release"]["n_developers"] > out["control"]["n_developers"]


def test_human_vs_tier_blocks_cover_both_tiers(wired: dict[str, list]) -> None:
    out = pcs.compute()
    for tier in ("opus", "sonnet"):
        block = out[f"human_vs_{tier}"]
        assert block["tau_tier"] == pytest.approx(0.8)
        assert block["n_tier"] == 7
        assert block["mm_coef"] == pytest.approx(-0.42)
        assert block["mm_converged"] is True


def test_capability_reports_both_orderings_and_hoists_per_cell(wired: dict[str, list]) -> None:
    """Both orderings are reported as each other's sensitivity check, and per_cell
    is hoisted out of the trend payload exactly once."""
    cap = pcs.compute()["capability"]
    assert set(cap) == {"family", "version", "per_cell", "composition"}
    for ordering in sim.CAPABILITY_ORDERINGS:
        assert "per_cell" not in cap[ordering]["trend"]
        assert cap[ordering]["trend"]["rho_spearman"] == pytest.approx(0.5)
        assert cap[ordering]["mixed"]["converged"] is True
    # Hoisted from the family ordering specifically, not whichever ran last.
    assert cap["per_cell"] == PER_CELL
    assert cap["family"]["trend"]["ordering"] == "family"


def test_capability_composition_counts_cells_per_family(wired: dict[str, list]) -> None:
    """The appendix states this breakdown because the cell total coincides with the
    pooled-version total beside it. PER_CELL has 1 haiku + 2 sonnet + 1 opus, so a
    count that collapsed families or counted rows would give 3 or 4, not this."""
    comp = pcs.compute()["capability"]["composition"]
    assert comp["cells_by_family"] == {"haiku": 1, "opus": 1, "sonnet": 2}


def test_capability_composition_flags_versions_carrying_no_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A version can appear in the trailers and still be the plurality cell of no
    session -- true of Opus 5 in the real cohort, which is why the appendix has to say
    the cells are not simply the released versions. Trailers here carry Opus 4.8 *and*
    4.9 while PER_CELL has a cell only for 4.8, so 4.9 must be reported as uncelled."""
    ai = tmp_path / "ai_trailers.json"
    cohort = _cohort(AI_SESSIONS)
    cohort["commits"] = [{"model": "Claude Opus 4.8"}, {"model": "Claude Opus 4.9"}]
    ai.write_text(json.dumps(cohort))
    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text(json.dumps(_cohort(CONTROL_SESSIONS, control_since="a", control_until="b")))
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    monkeypatch.setattr(pcs, "CONTROL_PATH", ctrl)
    monkeypatch.setattr(pcs, "WITHIN_AI_PATH", tmp_path / "absent_a.json")
    monkeypatch.setattr(pcs, "WITHIN_NONAI_PATH", tmp_path / "absent_n.json")
    _install_stubs(monkeypatch)

    comp = pcs.compute()["capability"]["composition"]
    assert comp["versions_without_a_cell"]["opus"] == [4.9]


# ---------------------------------------------------------------------------
# model_versions: the version inventory behind the two tier labels
# ---------------------------------------------------------------------------


def _ai_cohort_with_trailers(trailers: list[str | None]) -> dict[str, Any]:
    cohort = _cohort(AI_SESSIONS)
    cohort["commits"] = [{"model": t} for t in trailers]
    return cohort


def _inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trailers: list[str | None]) -> dict:
    ai = tmp_path / "ai_trailers.json"
    ai.write_text(json.dumps(_ai_cohort_with_trailers(trailers)))
    monkeypatch.setattr(pcs, "AI_PATH", ai)
    return pcs.compute()["model_versions"]


def test_model_versions_counts_trailer_strings_and_released_versions(
    wired: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two numbers per family, and they differ: the paper cites the trailer count as
    the size of the pooling and the version count as what it pools onto."""
    inv = _inventory(
        monkeypatch,
        tmp_path,
        ["Claude Opus 4.6", "Claude Opus 4.6 (1M context)", "Claude Opus 4.8"],
    )
    assert inv["opus"]["n_trailer_strings"] == 3
    assert inv["opus"]["versions"] == [4.6, 4.8]


def test_model_versions_collapses_the_context_window_suffix(
    wired: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NEGATIVE TEST: "(1M context)" is the same model at a longer window. Counting it
    as its own version would overstate the capability spread the paper reports, which
    is the whole quantity Finding 2's coarseness argument rests on."""
    inv = _inventory(monkeypatch, tmp_path, ["Claude Sonnet 4.6", "Claude Sonnet 4.6 (1M context)"])
    assert inv["sonnet"]["versions"] == [4.6]
    assert inv["sonnet"]["n_trailer_strings"] == 2


def test_model_versions_excludes_unparseable_and_absent_trailers(
    wired: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control-mode commits carry no trailer and bare "Claude" carries no version.
    Neither can be placed on a capability scale, so neither may reach the inventory."""
    inv = _inventory(monkeypatch, tmp_path, [None, "Claude", "not a model", "Claude Opus 4.7"])
    assert inv == {"opus": {"n_trailer_strings": 1, "versions": [4.7]}}


def test_model_versions_is_absent_for_a_family_with_no_commits(
    wired: dict[str, list], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inv = _inventory(monkeypatch, tmp_path, ["Claude Haiku 4.5"])
    assert set(inv) == {"haiku"}


# ---------------------------------------------------------------------------
# main(): writes the cache the macro generator reads
# ---------------------------------------------------------------------------


def test_main_writes_the_stats_cache(
    wired: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "part_c_control_stats.json"
    monkeypatch.setattr(pcs, "OUT_PATH", out_path)
    monkeypatch.setattr(pcs, "REPO", tmp_path)

    pcs.main()

    written = json.loads(out_path.read_text())
    assert written["control"]["tau_human"] == pytest.approx(HUMAN_NONPURE_MEAN)
    assert "within_era" not in written
    # The summary goes to stderr so stdout stays free for piping.
    err = capsys.readouterr().err
    assert "tau_human" in err
    assert "vs opus" in err
