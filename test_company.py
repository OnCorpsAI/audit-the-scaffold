"""Tests for git_session_extractor and company_simulations.

Pure in-memory, no git calls — except the ``_dominant_branch``/``_remote_branches``
tests, which build a throwaway local git repo under ``tmp_path`` (no external or
client data touched)."""

from __future__ import annotations

import csv
import io
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

import company_simulations as sim
import git_session_extractor as gse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = UTC


def _commit(
    *,
    sha: str = "abc123",
    ts: str,
    model: str = "Claude Sonnet 4.6",
    commit_type: str = "feat",
    files: int = 3,
    insertions: int = 10,
    deletions: int = 5,
    dir_category: str = "code",
    developer: str = "dev_00000000",
) -> gse.CommitMetrics:
    return gse.CommitMetrics(
        sha=sha,
        timestamp=datetime.fromisoformat(ts),
        model=model,
        commit_type=commit_type,
        files_changed=files,
        insertions=insertions,
        deletions=deletions,
        churn=insertions + deletions,
        dir_category=dir_category,
        developer=developer,
    )


def _session_dict(
    *,
    repo: str = "repo-a",
    start_ts: str = "2024-01-01T10:00:00",
    end_ts: str = "2024-01-01T11:00:00",
    n_commits: int = 5,
    survivor_ratio: float | None = 0.5,
    task_type: str = "code",
    developer: str = "dev_00000000",
) -> dict:
    return {
        "repo": repo,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_commits": n_commits,
        "survivor_ratio": survivor_ratio,
        "task_type": task_type,
        "developer": developer,
    }


def _commit_dict(
    *,
    repo: str = "repo-a",
    timestamp: str = "2024-01-01T10:30:00",
    model: str | None = "Claude Sonnet 4.6",
    churn: int = 20,
    dir_category: str = "code",
    developer: str = "dev_00000000",
) -> dict:
    return {
        "repo": repo,
        "timestamp": timestamp,
        "model": model,
        "churn": churn,
        "dir_category": dir_category,
        "developer": developer,
    }


# ---------------------------------------------------------------------------
# git_session_extractor: _parse_shortstat
# ---------------------------------------------------------------------------


def test_parse_shortstat_full() -> None:
    stat = " 3 files changed, 10 insertions(+), 5 deletions(-)"
    assert gse._parse_shortstat(stat) == (3, 10, 5)


def test_parse_shortstat_insertions_only() -> None:
    stat = " 1 file changed, 7 insertions(+)"
    assert gse._parse_shortstat(stat) == (1, 7, 0)


def test_parse_shortstat_deletions_only() -> None:
    stat = " 2 files changed, 3 deletions(-)"
    assert gse._parse_shortstat(stat) == (2, 0, 3)


def test_parse_shortstat_empty() -> None:
    assert gse._parse_shortstat("") == (0, 0, 0)


def test_parse_shortstat_singular_file() -> None:
    stat = " 1 file changed, 1 insertion(+), 1 deletion(-)"
    assert gse._parse_shortstat(stat) == (1, 1, 1)


# ---------------------------------------------------------------------------
# git_session_extractor: CLAUDE_CO_AUTHOR_RE and COMMIT_TYPE_RE
# ---------------------------------------------------------------------------


def test_co_author_re_matches_standard_format() -> None:
    body = "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    m = gse.CLAUDE_CO_AUTHOR_RE.search(body)
    assert m is not None
    assert "Sonnet" in m.group(1)


def test_co_author_re_case_insensitive() -> None:
    body = "co-authored-by: claude opus 4.8 <noreply@anthropic.com>"
    assert gse.CLAUDE_CO_AUTHOR_RE.search(body) is not None


def test_co_author_re_no_match_without_anthropic_domain() -> None:
    body = "Co-Authored-By: Some Human <human@example.com>"
    assert gse.CLAUDE_CO_AUTHOR_RE.search(body) is None


def test_commit_type_re_matches_conventional_prefixes() -> None:
    for prefix in ("feat", "fix", "refactor", "docs", "test", "chore"):
        assert gse.COMMIT_TYPE_RE.match(f"{prefix}: add something") is not None


def test_commit_type_re_no_match_free_form() -> None:
    assert gse.COMMIT_TYPE_RE.match("Add something") is None


# ---------------------------------------------------------------------------
# git_session_extractor: _summarise
# ---------------------------------------------------------------------------


def test_summarise_survivor_ratio_formula() -> None:
    c1 = _commit(ts="2024-01-01T10:00:00+00:00", insertions=100, deletions=60)
    c2 = _commit(ts="2024-01-01T10:30:00+00:00", insertions=20, deletions=40)
    s = gse._summarise("repo-a", 0, [c1, c2])
    # net = (100+20) - (60+40) = 20; churn = 220; survivor = 20/220 ≈ 0.091
    assert s.total_churn == 220
    assert s.survivor_ratio is not None
    assert abs(s.survivor_ratio - round(20 / 220, 3)) < 1e-9


def test_summarise_pure_addition_survivor_close_to_one() -> None:
    # Only insertions → net = insertions, survivor = insertions / insertions = 1.0
    c1 = _commit(ts="2024-01-01T10:00:00+00:00", insertions=50, deletions=0)
    c2 = _commit(ts="2024-01-01T10:05:00+00:00", insertions=30, deletions=0)
    s = gse._summarise("repo-a", 0, [c1, c2])
    assert s.survivor_ratio == 1.0


def test_summarise_zero_churn_survivor_is_none() -> None:
    c1 = _commit(ts="2024-01-01T10:00:00+00:00", insertions=0, deletions=0)
    s = gse._summarise("repo-a", 0, [c1])
    assert s.survivor_ratio is None


def test_summarise_single_commit_duration_zero() -> None:
    c1 = _commit(ts="2024-01-01T10:00:00+00:00")
    s = gse._summarise("repo-a", 0, [c1])
    assert s.duration_minutes == 0.0
    assert s.n_commits == 1


# ---------------------------------------------------------------------------
# git_session_extractor: group_into_sessions
# ---------------------------------------------------------------------------


def test_group_into_sessions_same_session_within_4h() -> None:
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00")
    c2 = _commit(sha="b", ts="2024-01-01T13:00:00+00:00")  # 3h gap — same session
    sessions = gse.group_into_sessions("repo-a", [c1, c2])
    assert len(sessions) == 1
    assert sessions[0].n_commits == 2


def test_group_into_sessions_splits_at_4h_boundary() -> None:
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00")
    c2 = _commit(sha="b", ts="2024-01-01T14:01:00+00:00")  # > 4h gap → new session
    sessions = gse.group_into_sessions("repo-a", [c1, c2])
    assert len(sessions) == 2


def test_group_into_sessions_exactly_4h_is_same_session() -> None:
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00")
    c2 = _commit(sha="b", ts="2024-01-01T14:00:00+00:00")  # exactly 4h — NOT a split
    sessions = gse.group_into_sessions("repo-a", [c1, c2])
    assert len(sessions) == 1


def test_group_into_sessions_empty_returns_empty() -> None:
    assert gse.group_into_sessions("repo-a", []) == []


def test_group_into_sessions_three_sessions() -> None:
    commits = [
        _commit(sha="a", ts="2024-01-01T10:00:00+00:00"),
        _commit(sha="b", ts="2024-01-01T11:00:00+00:00"),
        _commit(sha="c", ts="2024-01-01T17:00:00+00:00"),  # 6h gap
        _commit(sha="d", ts="2024-01-01T18:00:00+00:00"),
        _commit(sha="e", ts="2024-01-02T08:00:00+00:00"),  # 14h gap
    ]
    sessions = gse.group_into_sessions("repo-a", commits)
    assert len(sessions) == 3
    assert sessions[0].n_commits == 2
    assert sessions[1].n_commits == 2
    assert sessions[2].n_commits == 1


def test_group_into_sessions_default_does_not_split_on_developer_change() -> None:
    """Regression guard: same_developer_only defaults to False, so AI-session
    grouping (which never set this) is unaffected by this parameter's addition."""
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00", developer="dev_aaaaaaaa")
    c2 = _commit(sha="b", ts="2024-01-01T11:00:00+00:00", developer="dev_bbbbbbbb")
    sessions = gse.group_into_sessions("repo-a", [c1, c2])
    assert len(sessions) == 1
    assert sessions[0].n_commits == 2


def test_group_into_sessions_same_developer_only_splits_on_developer_change() -> None:
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00", developer="dev_aaaaaaaa")
    c2 = _commit(sha="b", ts="2024-01-01T11:00:00+00:00", developer="dev_bbbbbbbb")  # 1h gap
    sessions = gse.group_into_sessions("repo-a", [c1, c2], same_developer_only=True)
    assert len(sessions) == 2
    assert sessions[0].n_commits == 1
    assert sessions[1].n_commits == 1


def test_group_into_sessions_same_developer_only_keeps_same_developer_together() -> None:
    c1 = _commit(sha="a", ts="2024-01-01T10:00:00+00:00", developer="dev_aaaaaaaa")
    c2 = _commit(sha="b", ts="2024-01-01T11:00:00+00:00", developer="dev_aaaaaaaa")
    sessions = gse.group_into_sessions("repo-a", [c1, c2], same_developer_only=True)
    assert len(sessions) == 1
    assert sessions[0].n_commits == 2


# ---------------------------------------------------------------------------
# company_simulations: model_tier and dominant_tier
# ---------------------------------------------------------------------------


def test_model_tier_opus() -> None:
    assert sim.model_tier("Claude Opus 4.8") == "opus"


def test_model_tier_sonnet() -> None:
    assert sim.model_tier("Claude Sonnet 4.6 (1M context)") == "sonnet"


def test_model_tier_haiku() -> None:
    assert sim.model_tier("claude-haiku-4-5-20251001") == "haiku"


def test_model_tier_unknown() -> None:
    assert sim.model_tier("gpt-4o") == "unknown"


def test_dominant_tier_majority_wins() -> None:
    sess = _session_dict(repo="r", start_ts="2024-01-01T10:00:00", end_ts="2024-01-01T12:00:00")
    commits = [
        _commit_dict(repo="r", timestamp="2024-01-01T10:30:00", model="Claude Sonnet 4.6"),
        _commit_dict(repo="r", timestamp="2024-01-01T11:00:00", model="Claude Sonnet 4.6"),
        _commit_dict(repo="r", timestamp="2024-01-01T11:30:00", model="Claude Opus 4.8"),
    ]
    assert sim.dominant_tier(sess, commits) == "sonnet"


def test_dominant_tier_no_matching_commits_returns_unknown() -> None:
    sess = _session_dict(repo="r", start_ts="2024-01-01T10:00:00", end_ts="2024-01-01T11:00:00")
    assert sim.dominant_tier(sess, []) == "unknown"


# ---------------------------------------------------------------------------
# company_simulations: session_commit_churns
# ---------------------------------------------------------------------------


def test_session_commit_churns_sorted_by_timestamp() -> None:
    sess = _session_dict(repo="r", start_ts="2024-01-01T10:00:00", end_ts="2024-01-01T12:00:00")
    commits = [
        _commit_dict(repo="r", timestamp="2024-01-01T11:00:00", churn=30),
        _commit_dict(repo="r", timestamp="2024-01-01T10:30:00", churn=10),
        _commit_dict(repo="r", timestamp="2024-01-01T11:30:00", churn=20),
    ]
    churns = sim.session_commit_churns(sess, commits)
    assert churns == [10, 30, 20]


def test_session_commit_churns_filters_by_repo() -> None:
    sess = _session_dict(repo="r", start_ts="2024-01-01T10:00:00", end_ts="2024-01-01T12:00:00")
    commits = [
        _commit_dict(repo="r", timestamp="2024-01-01T10:30:00", churn=10),
        _commit_dict(repo="other-repo", timestamp="2024-01-01T10:45:00", churn=99),
    ]
    assert sim.session_commit_churns(sess, commits) == [10]


# ---------------------------------------------------------------------------
# company_simulations: fit_params
# ---------------------------------------------------------------------------


def _make_sessions_and_commits(
    n: int,
    survivor: float,
    model: str = "Claude Sonnet 4.6",
    n_commits: int = 5,
) -> tuple[list[dict], list[dict]]:
    sessions = []
    commits = []
    for i in range(n):
        start = f"2024-01-{i + 1:02d}T10:00:00"
        end = f"2024-01-{i + 1:02d}T12:00:00"
        sessions.append(
            _session_dict(
                repo="r",
                start_ts=start,
                end_ts=end,
                n_commits=n_commits,
                survivor_ratio=survivor,
            )
        )
        for j in range(n_commits):
            churn = 100 * (0.8**j)  # decaying series for AR1 estimation
            ts_h = 10 + j
            commits.append(
                _commit_dict(
                    repo="r",
                    timestamp=f"2024-01-{i + 1:02d}T{ts_h:02d}:00:00",
                    model=model,
                    churn=int(churn),
                )
            )
    return sessions, commits


def test_fit_params_rho_sigma_computed() -> None:
    sessions, commits = _make_sessions_and_commits(10, survivor=0.6)
    params = sim.fit_params(sessions, commits)
    assert 0.0 < params.rho < 1.5
    assert params.sigma >= 0.0
    assert params.n_sessions_fit == 10


def test_fit_params_tau_per_tier() -> None:
    def _make(n: int, sr: float, model: str, repo: str) -> tuple[list[dict], list[dict]]:
        sessions = [
            _session_dict(
                repo=repo,
                start_ts=f"2024-01-{i + 1:02d}T10:00:00",
                end_ts=f"2024-01-{i + 1:02d}T12:00:00",
                n_commits=5,
                survivor_ratio=sr,
            )
            for i in range(n)
        ]
        commits = [
            _commit_dict(
                repo=repo, timestamp=f"2024-01-{i + 1:02d}T10:30:00", model=model, churn=50
            )
            for i in range(n)
        ]
        return sessions, commits

    s_sessions, s_commits = _make(6, 0.6, "Claude Sonnet 4.6", "sonnet-repo")
    o_sessions, o_commits = _make(4, 0.75, "Claude Opus 4.8", "opus-repo")
    params = sim.fit_params(s_sessions + o_sessions, s_commits + o_commits)
    assert abs(params.tau_sonnet - 0.6) < 0.01
    assert abs(params.tau_opus - 0.75) < 0.01


def test_fit_params_pi_pure_computed() -> None:
    sessions = [
        _session_dict(n_commits=3, survivor_ratio=1.0),  # pure
        _session_dict(n_commits=3, survivor_ratio=1.0),  # pure
        _session_dict(n_commits=3, survivor_ratio=0.5),  # not pure
        _session_dict(n_commits=3, survivor_ratio=0.5),  # not pure
        _session_dict(n_commits=3, survivor_ratio=0.5),  # not pure
        _session_dict(n_commits=3, survivor_ratio=0.5),  # not pure
    ]
    # Need commits for dominant_tier — use minimal set
    commits = [
        _commit_dict(repo="repo-a", timestamp="2024-01-01T10:30:00", churn=50),
        _commit_dict(repo="repo-a", timestamp="2024-01-01T10:35:00", churn=40),
    ]
    params = sim.fit_params(sessions, commits)
    # 2 pure out of 6 n≥2 sessions → pi_pure = 2/6 ≈ 0.333
    assert abs(params.pi_pure - 2 / 6) < 0.01


def test_fit_params_pi_pure_zero_when_no_pure_sessions() -> None:
    sessions, commits = _make_sessions_and_commits(5, survivor=0.5)
    params = sim.fit_params(sessions, commits)
    assert params.pi_pure == 0.0


def test_fit_params_tau_fitted_on_nonpure_sessions_only() -> None:
    # Mix: some pure (sr=1.0), some non-pure (sr=0.5)
    # tau should reflect only the non-pure sessions' mean
    sessions = [
        _session_dict(n_commits=3, survivor_ratio=1.0),
        _session_dict(n_commits=3, survivor_ratio=1.0),
        _session_dict(n_commits=3, survivor_ratio=0.5),
        _session_dict(n_commits=3, survivor_ratio=0.5),
    ]
    commits: list[dict] = []
    params = sim.fit_params(sessions, commits)
    # Only non-pure sessions used for tau_pooled; mean of [0.5, 0.5] = 0.5
    assert abs(params.tau_pooled - 0.5) < 0.01


# ---------------------------------------------------------------------------
# company_simulations: simulate_session
# ---------------------------------------------------------------------------


def test_simulate_session_survivor_in_unit_interval() -> None:
    rng = random.Random(42)
    for _ in range(100):
        _, sr = sim.simulate_session(5, rho=0.84, sigma=0.5, tau=0.6, rng=rng)
        assert 0.0 <= sr <= 1.0


def test_simulate_session_pi_pure_zero_never_returns_exactly_one_from_beta() -> None:
    # With pi_pure=0, sr is Beta-drawn; with tau=0.9 it's rare but possible to hit 1.0
    # Just check sr is in [0, 1] range
    rng = random.Random(0)
    srs = [sim.simulate_session(3, 0.84, 0.5, 0.6, rng=rng, pi_pure=0.0)[1] for _ in range(200)]
    assert all(0.0 <= sr <= 1.0 for sr in srs)


def test_simulate_session_pi_pure_one_always_returns_one() -> None:
    rng = random.Random(7)
    for _ in range(50):
        _, sr = sim.simulate_session(3, 0.84, 0.5, 0.6, rng=rng, pi_pure=1.0)
        assert sr == 1.0


def test_simulate_session_pi_pure_mixture_mass_approximately_correct() -> None:
    rng = random.Random(99)
    pi = 0.3
    srs = [sim.simulate_session(3, 0.84, 0.5, 0.6, rng=rng, pi_pure=pi)[1] for _ in range(1000)]
    pure_fraction = sum(1 for sr in srs if sr == 1.0) / len(srs)
    # With 1000 draws the fraction should be within ±0.05 of pi_pure
    assert abs(pure_fraction - pi) < 0.05


def test_simulate_session_trajectory_length_equals_n_commits() -> None:
    rng = random.Random(1)
    traj, _ = sim.simulate_session(7, 0.84, 0.5, 0.6, rng=rng)
    assert len(traj) == 7


def test_simulate_session_trajectory_normalized_to_one_at_start() -> None:
    rng = random.Random(2)
    traj, _ = sim.simulate_session(5, 0.84, 0.5, 0.6, rng=rng)
    assert traj[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# company_simulations: run_simulation
# ---------------------------------------------------------------------------


def test_run_simulation_output_shapes() -> None:
    params = sim.FittedParams(
        rho=0.84,
        sigma=0.5,
        tau_opus=0.68,
        tau_sonnet=0.61,
        tau_pooled=0.63,
        session_n_median=2,
        session_n_mean=4,
        n_sessions_fit=47,
        pi_pure=0.05,
    )
    results = sim.run_simulation(params, n_synthetic=50)
    for tier in ("opus", "sonnet"):
        assert len(results[tier]["survivor_ratios"]) == 50
        assert len(results[tier]["trajectories"]) == 50
        assert len(results[tier]["n_commits"]) == 50


def test_run_simulation_survivor_ratios_in_unit_interval() -> None:
    params = sim.FittedParams(
        rho=0.84,
        sigma=0.5,
        tau_opus=0.68,
        tau_sonnet=0.61,
        tau_pooled=0.63,
        session_n_median=2,
        session_n_mean=4,
        n_sessions_fit=47,
        pi_pure=0.0,
    )
    results = sim.run_simulation(params, n_synthetic=100)
    for tier in ("opus", "sonnet"):
        for sr in results[tier]["survivor_ratios"]:
            assert 0.0 <= sr <= 1.0


def test_run_simulation_with_pi_pure_produces_exact_one_mass() -> None:
    params = sim.FittedParams(
        rho=0.84,
        sigma=0.5,
        tau_opus=0.68,
        tau_sonnet=0.61,
        tau_pooled=0.63,
        session_n_median=3,
        session_n_mean=4,
        n_sessions_fit=47,
        pi_pure=1.0,
    )
    results = sim.run_simulation(params, n_synthetic=20)
    for tier in ("opus", "sonnet"):
        assert all(sr == 1.0 for sr in results[tier]["survivor_ratios"])


# ---------------------------------------------------------------------------
# company_simulations: survivor_gap_stats
# ---------------------------------------------------------------------------


def _gap_data(
    n_opus: int,
    sr_opus: float,
    n_sonnet: int,
    sr_sonnet: float,
) -> tuple[list[dict], list[dict]]:
    """Build synthetic sessions + commits with two clearly-separated tiers."""
    sessions: list[dict] = []
    commits: list[dict] = []
    for i in range(n_opus):
        repo = f"opus-repo-{i}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2024-01-{i + 1:02d}T10:00:00",
                end_ts=f"2024-01-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=sr_opus,
            )
        )
        commits.append(
            _commit_dict(
                repo=repo,
                timestamp=f"2024-01-{i + 1:02d}T10:30:00",
                model="Claude Opus 4.8",
                churn=20,
            )
        )
    for i in range(n_sonnet):
        repo = f"sonnet-repo-{i}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2024-02-{i + 1:02d}T10:00:00",
                end_ts=f"2024-02-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=sr_sonnet,
            )
        )
        commits.append(
            _commit_dict(
                repo=repo,
                timestamp=f"2024-02-{i + 1:02d}T10:30:00",
                model="Claude Sonnet 4.6",
                churn=20,
            )
        )
    return sessions, commits


def test_survivor_gap_stats_returns_expected_keys() -> None:
    sessions, commits = _gap_data(10, 0.7, 10, 0.4)
    result = sim.survivor_gap_stats(sessions, commits)
    for key in (
        "tau_opus_mean",
        "tau_sonnet_mean",
        "gap",
        "p_value",
        "significant",
        "ci_opus_95",
        "ci_sonnet_95",
        "ci_gap_95",
        "n_opus",
        "n_sonnet",
    ):
        assert key in result, f"missing key: {key}"


def test_survivor_gap_stats_significant_for_separated_groups() -> None:
    """p < 0.05 when groups are clearly separated (Opus sr=0.85, Sonnet sr=0.20)."""
    sessions, commits = _gap_data(25, 0.85, 25, 0.20)
    result = sim.survivor_gap_stats(sessions, commits)
    assert result["p_value"] < 0.05, f"expected p<0.05, got {result['p_value']}"
    assert result["significant"] is True


def test_survivor_gap_stats_not_significant_for_identical_groups() -> None:
    """NEGATIVE TEST: p is large when both groups have the same survivor ratios."""
    sessions, commits = _gap_data(20, 0.60, 20, 0.60)
    result = sim.survivor_gap_stats(sessions, commits)
    # Identical groups → permutation test cannot reject; p should be large
    assert result["p_value"] > 0.05, (
        f"expected p>0.05 for identical groups, got {result['p_value']}; "
        "significance logic is broken if this fails"
    )
    assert result["significant"] is False


def test_survivor_gap_stats_bootstrap_ci_brackets_known_mean() -> None:
    """Bootstrap 95% CI for the Opus tier contains the sample mean."""
    # Use varied survivor ratios so the CI is non-degenerate
    rng_local = random.Random(7)
    sessions: list[dict] = []
    commits: list[dict] = []
    center = 0.70
    for i in range(30):
        repo = f"opus-v-{i}"
        sr = max(0.01, min(0.98, center + rng_local.uniform(-0.15, 0.15)))
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2024-01-{i + 1:02d}T10:00:00",
                end_ts=f"2024-01-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=sr,
            )
        )
        commits.append(
            _commit_dict(
                repo=repo,
                timestamp=f"2024-01-{i + 1:02d}T10:30:00",
                model="Claude Opus 4.8",
                churn=20,
            )
        )
    for i in range(15):
        repo = f"sonnet-v-{i}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2024-02-{i + 1:02d}T10:00:00",
                end_ts=f"2024-02-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=0.50,
            )
        )
        commits.append(
            _commit_dict(
                repo=repo,
                timestamp=f"2024-02-{i + 1:02d}T10:30:00",
                model="Claude Sonnet 4.6",
                churn=20,
            )
        )
    result = sim.survivor_gap_stats(sessions, commits)
    lo, hi = result["ci_opus_95"]
    # The CI must bracket the sample mean (fundamental property of bootstrap CIs)
    sample_mean = result["tau_opus_mean"]
    assert lo <= sample_mean <= hi, (
        f"CI [{lo:.3f}, {hi:.3f}] does not bracket sample mean {sample_mean:.3f}"
    )
    # And the CI should be non-trivial (wide enough to reflect real variance)
    assert hi > lo, "CI should be non-degenerate with varied input data"


def test_survivor_gap_stats_gap_direction() -> None:
    """Gap is positive when Opus > Sonnet and tau means are ordered correctly."""
    sessions, commits = _gap_data(20, 0.8, 20, 0.5)
    result = sim.survivor_gap_stats(sessions, commits)
    assert result["gap"] > 0
    assert result["tau_opus_mean"] > result["tau_sonnet_mean"]


def test_survivor_gap_stats_n_counts_correct() -> None:
    sessions, commits = _gap_data(15, 0.7, 25, 0.5)
    result = sim.survivor_gap_stats(sessions, commits)
    assert result["n_opus"] == 15
    assert result["n_sonnet"] == 25


# ---------------------------------------------------------------------------
# survivor_gap_stats / mixed_model_gap_stats: generalized ``tiers`` param
# (Part C human-control robustness check — "human" tier comes from
# control-mode extraction, i.e. commits with model=None)
# ---------------------------------------------------------------------------


def _human_vs_tier_data(
    n_human: int,
    sr_human: float,
    tier_model: str,
    n_tier: int,
    sr_tier: float,
) -> tuple[list[dict], list[dict]]:
    """Synthetic sessions + commits: a no-model ("human") group vs one AI tier."""
    sessions: list[dict] = []
    commits: list[dict] = []
    for i in range(n_human):
        repo = f"human-repo-{i}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2022-01-{i + 1:02d}T10:00:00",
                end_ts=f"2022-01-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=sr_human,
            )
        )
        commits.append(
            _commit_dict(repo=repo, timestamp=f"2022-01-{i + 1:02d}T10:30:00", model=None, churn=20)
        )
    for i in range(n_tier):
        repo = f"tier-repo-{i}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts=f"2024-01-{i + 1:02d}T10:00:00",
                end_ts=f"2024-01-{i + 1:02d}T12:00:00",
                n_commits=2,
                survivor_ratio=sr_tier,
            )
        )
        commits.append(
            _commit_dict(
                repo=repo,
                timestamp=f"2024-01-{i + 1:02d}T10:30:00",
                model=tier_model,
                churn=20,
            )
        )
    return sessions, commits


def test_model_tier_none_is_human() -> None:
    assert sim.model_tier(None) == "human"


def test_dominant_tier_all_none_model_commits_is_human() -> None:
    sessions, commits = _human_vs_tier_data(3, 0.6, "Claude Opus 4.8", 3, 0.3)
    assert sim.dominant_tier(sessions[0], commits) == "human"


def test_survivor_gap_stats_tiers_param_names_keys_for_human_vs_opus() -> None:
    sessions, commits = _human_vs_tier_data(10, 0.7, "Claude Opus 4.8", 10, 0.4)
    result = sim.survivor_gap_stats(sessions, commits, tiers=("human", "opus"))
    for key in (
        "tau_human_mean",
        "tau_opus_mean",
        "gap",
        "p_value",
        "significant",
        "ci_human_95",
        "ci_opus_95",
        "ci_gap_95",
        "n_human",
        "n_opus",
    ):
        assert key in result, f"missing key: {key}"
    assert result["n_human"] == 10
    assert result["n_opus"] == 10


def test_survivor_gap_stats_tiers_param_default_unchanged() -> None:
    """Regression guard: omitting tiers still reproduces the exact Opus/Sonnet keys."""
    sessions, commits = _gap_data(10, 0.7, 10, 0.4)
    result = sim.survivor_gap_stats(sessions, commits)
    assert "tau_opus_mean" in result
    assert "tau_sonnet_mean" in result
    assert "tau_human_mean" not in result


def _mm_human_vs_tier_data(
    tier_model: str = "Claude Opus 4.8",
    n_repos: int = 4,
    n_per_repo: int = 6,
    sr_human: float = 0.6,
    sr_tier: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """Human (model=None) vs one AI tier, spread across repos/developers/task
    types — mirrors ``_mm_data``'s structure so the multilevel model has
    enough variance to fit (a single session per repo/developer combination
    makes repo and developer variance components unidentifiable)."""
    sessions: list[dict] = []
    commits: list[dict] = []
    rng = random.Random(11)
    task_types = ["code", "docs", "other"]
    for r in range(n_repos):
        repo = f"repo_{r}"
        for j in range(n_per_repo):
            is_human = j % 2 == 0
            model = None if is_human else tier_model
            base_sr = sr_human if is_human else sr_tier
            developer = f"dev_{j % 3}"
            task_type = task_types[j % len(task_types)]
            sr = max(0.0, min(0.998, base_sr + 0.05 * rng.random()))
            dd = (j % 28) + 1
            sessions.append(
                {
                    "repo": repo,
                    "start_ts": f"2024-{r + 1:02d}-{dd:02d}T10:00:00",
                    "end_ts": f"2024-{r + 1:02d}-{dd:02d}T12:00:00",
                    "n_commits": 2,
                    "survivor_ratio": sr,
                    "task_type": task_type,
                    "developer": developer,
                }
            )
            commits.append(
                {
                    "repo": repo,
                    "timestamp": f"2024-{r + 1:02d}-{dd:02d}T10:30:00",
                    "model": model,
                    "churn": 20,
                }
            )
    return sessions, commits


def test_mixed_model_gap_stats_tiers_param_human_vs_opus() -> None:
    sessions, commits = _mm_human_vs_tier_data()
    result = sim.mixed_model_gap_stats(sessions, commits, tiers=("human", "opus"))
    assert result["converged"] is True
    assert "tier_coef" in result
    assert "tier_p" in result


def test_mixed_model_gap_stats_tiers_param_default_unchanged() -> None:
    """Regression guard: omitting tiers still reproduces the Sonnet-reference,
    Opus-treatment formula (tier_key = ...Treatment('sonnet'))[T.opus])."""
    sessions, commits = _mm_data(n_repos=4, n_per_repo=6)
    result = sim.mixed_model_gap_stats(sessions, commits)
    assert result["converged"] is True


# ---------------------------------------------------------------------------
# fit_params: bootstrap CI for rho (Piece 1)
# ---------------------------------------------------------------------------


def test_fit_params_rho_ci_brackets_point_estimate() -> None:
    """Bootstrap 95% CI for rho contains the rho point estimate."""
    # Varied churn factors per session give genuine spread in log_ratios
    # so the bootstrap CI is non-degenerate.
    sessions = []
    commits = []
    factors = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        1.05,
        1.15,
        1.25,
        1.35,
        1.45,
    ]
    for idx, factor in enumerate(factors):
        repo = f"ci-repo-{idx:02d}"
        sessions.append(
            _session_dict(
                repo=repo, start_ts="2024-01-01T09:00:00", end_ts="2024-01-01T20:00:00", n_commits=5
            )
        )
        for j in range(5):
            churn = max(1, int(100 * (factor**j)))
            commits.append(
                _commit_dict(repo=repo, timestamp=f"2024-01-01T{10 + j:02d}:30:00", churn=churn)
            )
    params = sim.fit_params(sessions, commits)
    lo, hi = params.rho_ci
    assert lo <= params.rho <= hi, (
        f"rho_ci [{lo:.3f}, {hi:.3f}] does not bracket rho={params.rho:.3f}"
    )
    assert hi > lo, "CI should be non-degenerate (log_ratios must vary)"


# ---------------------------------------------------------------------------
# survivor_gap_stats: MWU cross-check (Piece 6)
# ---------------------------------------------------------------------------


def test_survivor_gap_stats_includes_mannwhitney() -> None:
    """survivor_gap_stats result includes mwu_p key with a valid probability."""
    sessions, commits = _gap_data(20, 0.8, 20, 0.5)
    result = sim.survivor_gap_stats(sessions, commits)
    assert "mwu_p" in result
    assert 0.0 <= result["mwu_p"] <= 1.0


def test_survivor_gap_stats_mwu_significant_for_separated_groups() -> None:
    """MWU p < 0.05 when Opus sr=0.85 and Sonnet sr=0.20."""
    sessions, commits = _gap_data(30, 0.85, 30, 0.20)
    result = sim.survivor_gap_stats(sessions, commits)
    assert result["mwu_p"] < 0.05, f"Expected MWU p<0.05, got {result['mwu_p']:.4f}"


# ---------------------------------------------------------------------------
# churn_decay_permutation (Piece 3)
# ---------------------------------------------------------------------------


def _make_decay_sessions(
    n: int,
    churns: list[int],
    repo_prefix: str = "decay-repo",
) -> tuple[list[dict], list[dict]]:
    """Build n sessions, each with the given per-commit churn list."""
    sessions: list[dict] = []
    commits: list[dict] = []
    n_commits = len(churns)
    for i in range(n):
        repo = f"{repo_prefix}-{i:03d}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts="2024-01-01T09:00:00",
                end_ts="2024-01-01T23:00:00",
                n_commits=n_commits,
            )
        )
        for j, churn in enumerate(churns):
            commits.append(
                _commit_dict(
                    repo=repo,
                    timestamp=f"2024-01-01T{10 + j:02d}:30:00",
                    churn=churn,
                )
            )
    return sessions, commits


def test_churn_decay_permutation_decreasing_beats_null() -> None:
    """Strictly decreasing churn sessions get a low permutation p-value."""
    # Churn halves each commit across 6 commits — strong decay signal
    sessions, commits = _make_decay_sessions(20, [256, 128, 64, 32, 16, 8])
    result = sim.churn_decay_permutation(sessions, commits, n_resamples=999, seed=42)
    assert "error" not in result
    obs = result["observed_stat"]
    assert obs < 0.5, f"Median ratio should be << 1, got {obs:.3f}"
    assert result["p_value"] < 0.05, f"Expected p<0.05, got {result['p_value']:.4f}"


def test_churn_decay_permutation_flat_churn_does_not_reject() -> None:
    """NEGATIVE TEST: flat churn should not show decay — p should be large."""
    # Constant churn — no decay signal; median late/early ratio ≈ 1.0
    sessions, commits = _make_decay_sessions(20, [50, 50, 50, 50, 50, 50])
    result = sim.churn_decay_permutation(sessions, commits, n_resamples=999, seed=42)
    assert "error" not in result
    assert result["p_value"] > 0.10, f"Expected p>0.10 for flat churn, got {result['p_value']:.4f}"


def test_churn_decay_permutation_insufficient_data_returns_error() -> None:
    """Returns error dict when no session has n_commits >= 5."""
    sessions = [_session_dict(n_commits=3)]
    commits = [_commit_dict(churn=20)] * 3
    result = sim.churn_decay_permutation(sessions, commits)
    assert "error" in result


def test_churn_decay_permutation_returns_expected_keys() -> None:
    """Result contains observed_stat, p_value, n_sessions."""
    sessions, commits = _make_decay_sessions(10, [200, 100, 50, 25, 12])
    result = sim.churn_decay_permutation(sessions, commits, n_resamples=99, seed=0)
    assert "error" not in result
    assert "observed_stat" in result
    assert "p_value" in result
    assert "n_sessions" in result
    assert result["n_sessions"] == 10
    assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# churn_ratio_bootstrap_ci
# ---------------------------------------------------------------------------


def test_churn_ratio_bootstrap_ci_returns_expected_keys() -> None:
    """Result contains observed_stat, ci_low, ci_high, n_sessions."""
    sessions, commits = _make_decay_sessions(20, [256, 128, 64, 32, 16, 8])
    result = sim.churn_ratio_bootstrap_ci(sessions, commits, n_resamples=999, seed=42)
    assert "error" not in result
    assert "observed_stat" in result
    assert "ci_low" in result
    assert "ci_high" in result
    assert result["n_sessions"] == 20
    assert result["ci_low"] <= result["observed_stat"] <= result["ci_high"]


def test_churn_ratio_bootstrap_ci_matches_permutation_observed_stat() -> None:
    """The CI's point estimate matches churn_decay_permutation's observed_stat
    on the same data -- both derive from the same median late/early ratio."""
    sessions, commits = _make_decay_sessions(20, [256, 128, 64, 32, 16, 8])
    perm = sim.churn_decay_permutation(sessions, commits, n_resamples=999, seed=42)
    ci = sim.churn_ratio_bootstrap_ci(sessions, commits, n_resamples=999, seed=42)
    assert ci["observed_stat"] == pytest.approx(perm["observed_stat"])


def test_churn_ratio_bootstrap_ci_strong_decay_excludes_one() -> None:
    """A strong, consistent decay signal should give a CI that excludes 1.0
    (the no-decay baseline), not just a low point estimate."""
    sessions, commits = _make_decay_sessions(30, [256, 128, 64, 32, 16, 8])
    result = sim.churn_ratio_bootstrap_ci(sessions, commits, n_resamples=999, seed=42)
    assert "error" not in result
    assert result["ci_high"] < 1.0, f"Expected CI to exclude 1.0, got {result}"


def test_churn_ratio_bootstrap_ci_flat_churn_includes_one() -> None:
    """NEGATIVE TEST: flat churn (no decay) should give a CI spanning 1.0."""
    sessions, commits = _make_decay_sessions(20, [50, 50, 50, 50, 50, 50])
    result = sim.churn_ratio_bootstrap_ci(sessions, commits, n_resamples=999, seed=42)
    assert "error" not in result
    assert result["ci_low"] <= 1.0 <= result["ci_high"]


def test_churn_ratio_bootstrap_ci_insufficient_data_returns_error() -> None:
    """Returns error dict when no session has n_commits >= 5."""
    sessions = [_session_dict(n_commits=3)]
    commits = [_commit_dict(churn=20)] * 3
    result = sim.churn_ratio_bootstrap_ci(sessions, commits)
    assert "error" in result


# ---------------------------------------------------------------------------
# out_of_sample_fit (Piece 4)
# ---------------------------------------------------------------------------


def test_out_of_sample_fit_returns_expected_keys() -> None:
    """Result contains ks_stat, ks_p, n_train, n_test."""
    sessions, commits = _make_sessions_and_commits(40, 0.6, n_commits=5)
    result = sim.out_of_sample_fit(sessions, commits, seed=42)
    assert "error" not in result, result
    assert "ks_stat" in result
    assert "ks_p" in result
    assert "n_train" in result
    assert "n_test" in result
    assert 0.0 <= result["ks_stat"] <= 1.0
    assert 0.0 <= result["ks_p"] <= 1.0


def test_out_of_sample_fit_train_test_sizes() -> None:
    """n_train + n_test covers all eligible sessions."""
    sessions, commits = _make_sessions_and_commits(40, 0.6, n_commits=5)
    # All sessions eligible: n_commits=5 >= 2 and survivor_ratio is not None
    result = sim.out_of_sample_fit(sessions, commits, seed=42)
    assert "error" not in result
    assert result["n_train"] + result["n_test"] == 40


def test_out_of_sample_fit_insufficient_data_returns_error() -> None:
    """Returns error when fewer than 10 eligible sessions are provided."""
    sessions, commits = _make_sessions_and_commits(5, 0.6, n_commits=5)
    result = sim.out_of_sample_fit(sessions, commits)
    assert "error" in result


def test_out_of_sample_fit_ks_outputs_are_valid_probabilities() -> None:
    """ks_stat is in [0,1] and ks_p is in [0,1]."""
    sessions, commits = _make_sessions_and_commits(40, 0.6, n_commits=5)
    result = sim.out_of_sample_fit(sessions, commits, seed=42)
    assert "error" not in result
    assert 0.0 <= result["ks_stat"] <= 1.0
    assert 0.0 <= result["ks_p"] <= 1.0


def test_out_of_sample_fit_respects_seed_for_split() -> None:
    """NEGATIVE TEST: different seeds produce different train/test splits."""
    # Use sessions with distinct survivor_ratios so the split actually matters
    sessions = []
    commits = []
    for i in range(40):
        sr = 0.1 + (i / 40) * 0.8  # survivor ratios from 0.1 to 0.9
        repo = f"oos-repo-{i:02d}"
        sessions.append(
            _session_dict(
                repo=repo,
                start_ts="2024-01-01T09:00:00",
                end_ts="2024-01-01T15:00:00",
                n_commits=5,
                survivor_ratio=sr,
            )
        )
        for j in range(5):
            commits.append(
                _commit_dict(repo=repo, timestamp=f"2024-01-01T{10 + j:02d}:30:00", churn=100)
            )
    r1 = sim.out_of_sample_fit(sessions, commits, seed=42)
    r2 = sim.out_of_sample_fit(sessions, commits, seed=99)
    assert "error" not in r1 and "error" not in r2
    # Different seeds → different splits → different test sets
    # (at least the n_train/n_test are stable; split ordering differs)
    assert r1["n_train"] + r1["n_test"] == 40
    assert r2["n_train"] + r2["n_test"] == 40


# ---------------------------------------------------------------------------
# Headline-number regression tests — pin paper-reported statistics
# Actual values are read from data/company_sessions.json and recomputed each
# run so the tests catch any drift in the underlying data or analysis code.
#
# Paper-reported values (Section 3, Findings 1–3; re-derivation on the
# 23-repo, dominant-branch-selected dataset, 2026-08-12):
#   ρ ≈ 0.76  (geometric churn-decay factor)       → actual 0.7637, tol ±0.005
#   σ = 2.39  (log-noise std)                       → actual 2.3858, tol ±0.05
#   τ_Opus  ≈ 0.55  (mean survivor ratio, Opus)     → actual 0.5494, tol ±0.01
#   τ_Sonnet ≈ 0.55 (mean survivor ratio, Sonnet)   → actual 0.5510, tol ±0.01
#   n_sessions = 401                                → exact
# ---------------------------------------------------------------------------

_SESSIONS_JSON = Path(__file__).parent / "data" / "company_sessions.json"


@pytest.fixture(scope="module")
def real_data() -> tuple[list[dict], list[dict]]:
    """Load the real company sessions and commits from disk."""
    d = json.loads(_SESSIONS_JSON.read_text())
    return d["sessions"], d["commits"]


@pytest.fixture(scope="module")
def fitted_params(real_data: tuple[list[dict], list[dict]]) -> sim.FittedParams:
    sessions, commits = real_data
    return sim.fit_params(sessions, commits)


class TestHeadlineNumbers:
    """Regression tests that pin the headline statistics reported in the paper.

    Any code or data change that shifts a headline number outside its tolerance
    must be explicitly reviewed and the tolerances updated with justification.
    """

    def test_n_sessions_exact(self, real_data: tuple[list[dict], list[dict]]) -> None:
        sessions, _ = real_data
        assert len(sessions) == 401, f"Expected 401 sessions, got {len(sessions)}"

    def test_rho_matches_paper(self, fitted_params: sim.FittedParams) -> None:
        # Paper: ρ ≈ 0.76; computed: 0.7637
        assert abs(fitted_params.rho - 0.773) < 0.005, (
            f"rho={fitted_params.rho:.5f} deviates from paper value 0.773 by more than ±0.005"
        )

    def test_sigma_matches_paper(self, fitted_params: sim.FittedParams) -> None:
        # Paper: σ = 2.39; computed: 2.3858
        assert abs(fitted_params.sigma - 2.39) < 0.05, (
            f"sigma={fitted_params.sigma:.4f} deviates from paper value 2.39 by more than ±0.05"
        )

    def test_tau_opus_matches_paper(self, fitted_params: sim.FittedParams) -> None:
        # Paper: τ_Opus ≈ 0.55; computed: 0.5494
        assert abs(fitted_params.tau_opus - 0.5494) < 0.01, (
            f"tau_opus={fitted_params.tau_opus:.4f} deviates from paper value 0.5494 "
            "by more than ±0.01"
        )

    def test_tau_sonnet_matches_paper(self, fitted_params: sim.FittedParams) -> None:
        # Paper: τ_Sonnet ≈ 0.55; computed: 0.5510
        assert abs(fitted_params.tau_sonnet - 0.5510) < 0.01, (
            f"tau_sonnet={fitted_params.tau_sonnet:.4f} deviates from paper value 0.5510 "
            "by more than ±0.01"
        )

    def test_rho_perturbation_changes_significantly(
        self,
        real_data: tuple[list[dict], list[dict]],
    ) -> None:
        """NEGATIVE TEST: flattening all commit churns to a constant must shift rho
        outside the paper tolerance, confirming the regression tests would catch
        data corruption.

        When all churns are equal, every log(b/a) = 0, so rho = exp(0) = 1.0 —
        more than 0.2 away from the paper value of 0.773.  If this test ever
        passes with the paper assertion (|rho - 0.773| < 0.005), the positive
        tests above have lost their discriminative power.
        """
        sessions, commits = real_data
        flat_commits = [dict(c, churn=100) for c in commits]
        perturbed = sim.fit_params(sessions, flat_commits)
        # Flat churns → rho ≈ 1.0; must differ from 0.773 by more than the tolerance
        assert abs(perturbed.rho - 0.773) >= 0.005, (
            f"Flat-churn rho={perturbed.rho:.4f} unexpectedly close to paper value — "
            "positive headline tests would miss data corruption"
        )


# ---------------------------------------------------------------------------
# Part C robustness check: pre-AI-adoption human control set vs AI tiers.
# Pins the headline numbers from the paper's "Robustness check" paragraph.
# Repo labels are namespaced before merging (they are positional -- repo_A means
# a different repository in each file -- so all 23 AI-tier labels collide with
# control labels). Developer labels are deliberately NOT namespaced; see
# _namespace.
# ---------------------------------------------------------------------------

_CONTROL_SESSIONS_JSON = Path(__file__).parent / "data" / "company_control_sessions.json"


_namespace = sim.namespace_repos


@pytest.fixture(scope="module")
def merged_human_ai_data() -> tuple[list[dict], list[dict]]:
    """Union of the AI-tier and human-control datasets, repo-namespaced only."""
    if not _CONTROL_SESSIONS_JSON.exists():
        pytest.skip("data/company_control_sessions.json not present in this checkout")
    ai = json.loads(_SESSIONS_JSON.read_text())
    ctrl = json.loads(_CONTROL_SESSIONS_JSON.read_text())
    ai_s, ai_c = _namespace(ai, "ai")
    ctrl_s, ctrl_c = _namespace(ctrl, "ctrl")
    return ai_s + ctrl_s, ai_c + ctrl_c


def test_namespace_prefixes_repo_but_not_developer() -> None:
    """The asymmetry is deliberate and load-bearing in both directions.

    Repo labels are positional and must be prefixed or unrelated repositories
    fuse. Developer labels are salted hashes of a canonical identity and must
    *not* be prefixed, or one person spanning both eras becomes two
    random-effect levels and the within-person signal is lost.
    """
    data = {
        "sessions": [{"repo": "repo_A", "developer": "dev_abc"}],
        "commits": [{"repo": "repo_A", "developer": "dev_abc"}],
    }
    sessions, commits = _namespace(data, "ctrl")
    for row in (sessions[0], commits[0]):
        assert row["repo"] == "ctrl_repo_A"
        assert row["developer"] == "dev_abc", "developer must not be namespaced"


def test_namespace_defaults_missing_developer() -> None:
    data = {"sessions": [{"repo": "repo_A"}], "commits": [{"repo": "repo_A"}]}
    sessions, commits = _namespace(data, "ai")
    assert sessions[0]["developer"] == "unknown"
    assert commits[0]["developer"] == "unknown"


class TestControlHeadlineNumbers:
    """Regression tests pinning the paper's human-baseline robustness-check numbers."""

    @pytest.mark.slow
    def test_tau_human_matches_paper(
        self, merged_human_ai_data: tuple[list[dict], list[dict]]
    ) -> None:
        # Paper: tau_human ~ 0.38
        sessions, commits = merged_human_ai_data
        human_sessions = [s for s in sessions if sim.dominant_tier(s, commits) == "human"]
        nonpure = [
            s["survivor_ratio"]
            for s in human_sessions
            if s["survivor_ratio"] is not None and s["survivor_ratio"] < 0.999
        ]
        tau_human = statistics.mean(nonpure)
        assert abs(tau_human - 0.3896) < 0.01, (
            f"tau_human={tau_human:.4f} deviates from paper value 0.3896 by more than ±0.01"
        )

    def test_control_cohort_counts_match_paper(self) -> None:
        """Previously unpinned: the cohort counts the appendix quotes were checked
        by nobody, so a re-extraction could change them silently. The developer
        count is the one the curated roster fixed (100 -> 62)."""
        if not _CONTROL_SESSIONS_JSON.exists():
            pytest.skip("data/company_control_sessions.json not present in this checkout")
        ctrl = json.loads(_CONTROL_SESSIONS_JSON.read_text())
        assert len(ctrl["repos"]) == 122
        assert len({s["developer"] for s in ctrl["sessions"]}) == 62
        assert len(ctrl["sessions"]) == 9395
        assert len(ctrl["commits"]) == 17694
        assert ctrl["control_since"] == "2021-11-08"
        assert ctrl["control_until"] == "2022-09-19"
        assert ctrl["same_developer_only"] is True

    @pytest.mark.slow
    def test_rho_human_matches_paper(self) -> None:
        """Also previously unpinned. rho_human carries the claim that geometric
        front-loading is not AI-specific, so it needs a guard like rho does."""
        if not _CONTROL_SESSIONS_JSON.exists():
            pytest.skip("data/company_control_sessions.json not present in this checkout")
        ctrl = json.loads(_CONTROL_SESSIONS_JSON.read_text())
        fitted = sim.fit_params(ctrl["sessions"], ctrl["commits"])
        assert abs(fitted.rho - 0.863) < 0.005, (
            f"rho_human={fitted.rho:.4f} deviates from paper value 0.863 by more than ±0.005"
        )

    def test_within_era_sides_differ_in_exactly_one_respect(self) -> None:
        """The within-era contrast is only interpretable if the two sides share a
        window, a grouping rule, and a roster, and differ solely on whether a commit
        carries a Claude trailer. Anything else and the comparison silently differs
        in more than one way."""
        ai_path = Path(__file__).parent / "data" / "company_within_era_ai_sessions.json"
        nonai_path = Path(__file__).parent / "data" / "company_within_era_nonai_sessions.json"
        if not (ai_path.exists() and nonai_path.exists()):
            pytest.skip("within-era datasets not present in this checkout")
        ai = json.loads(ai_path.read_text())
        nonai = json.loads(nonai_path.read_text())

        assert ai["within_era_side"] == "ai"
        assert nonai["within_era_side"] == "nonai"
        for key in ("within_era_since", "within_era_until", "session_gap_hours"):
            assert ai[key] == nonai[key], f"{key} differs between the two sides"
        assert ai["same_developer_only"] is True
        assert nonai["same_developer_only"] is True

        # The defining difference: every AI-side commit carries a model, no
        # non-AI-side commit does.
        assert all(c["model"] for c in ai["commits"])
        assert not any(c["model"] for c in nonai["commits"])

        # Same roster, so the developer label spaces must coincide for pairing.
        assert {s["developer"] for s in ai["sessions"]} == {
            s["developer"] for s in nonai["sessions"]
        }

    def test_every_commit_is_linked_to_a_session(self) -> None:
        """Unlinked commits are dropped from per-session churn by
        _session_commits, so they are silent data loss. Both committed datasets
        previously carried some (124 control, 15 AI tier) from a timestamp-string
        comparison that mis-ordered across the cohort's 11 UTC offsets."""
        for path in (_SESSIONS_JSON, _CONTROL_SESSIONS_JSON):
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            unlinked = [c for c in data["commits"] if c.get("session_id") is None]
            assert not unlinked, f"{path.name}: {len(unlinked)} commits left unlinked"

    @pytest.mark.slow
    def test_human_vs_opus_gap_significant(
        self, merged_human_ai_data: tuple[list[dict], list[dict]]
    ) -> None:
        # Paper: gap = -0.171, perm p = 0.0002
        sessions, commits = merged_human_ai_data
        result = sim.survivor_gap_stats(sessions, commits, tiers=("human", "opus"))
        assert result["significant"] is True
        assert result["gap"] < -0.15, f"gap={result['gap']:.4f} weaker than paper's -0.171"

    @pytest.mark.slow
    def test_human_vs_sonnet_gap_significant(
        self, merged_human_ai_data: tuple[list[dict], list[dict]]
    ) -> None:
        # Paper: gap = -0.173, perm p = 0.0002
        sessions, commits = merged_human_ai_data
        result = sim.survivor_gap_stats(sessions, commits, tiers=("human", "sonnet"))
        assert result["significant"] is True
        assert result["gap"] < -0.15, f"gap={result['gap']:.4f} weaker than paper's -0.173"

    @pytest.mark.slow
    def test_mixed_model_human_vs_opus_tier_effect_positive_and_significant(
        self, merged_human_ai_data: tuple[list[dict], list[dict]]
    ) -> None:
        # Paper: adjusted coefficient = +0.166, p = 0.0004 (Opus higher than human)
        sessions, commits = merged_human_ai_data
        result = sim.mixed_model_gap_stats(sessions, commits, tiers=("human", "opus"))
        assert result["converged"] is True
        assert result["tier_coef"] > 0.1
        assert result["tier_p"] < 0.01


# ---------------------------------------------------------------------------
# Deconfounding stats: builders (varied within-tier spread for realistic SDs)
# ---------------------------------------------------------------------------


def _tier_sessions(
    opus_srs: list[float],
    sonnet_srs: list[float],
    *,
    commit_type: str = "other",
) -> tuple[list[dict], list[dict]]:
    """Sessions with per-session survivor ratios (genuine within-tier spread).

    Unlike ``_gap_data`` (constant sr per tier → zero within-group variance),
    this lets each tier carry a distribution, so Cohen's d and pooled SD are
    well-defined for the E-value.
    """
    sessions: list[dict] = []
    commits: list[dict] = []
    for month, model, srs in (
        (1, "Claude Opus 4.8", opus_srs),
        (2, "Claude Sonnet 4.6", sonnet_srs),
    ):
        for j, sr in enumerate(srs):
            repo = f"r{month}-{j}"
            dd = (j % 28) + 1
            sessions.append(
                {
                    "repo": repo,
                    "start_ts": f"2024-{month:02d}-{dd:02d}T10:00:00",
                    "end_ts": f"2024-{month:02d}-{dd:02d}T12:00:00",
                    "n_commits": 2,
                    "survivor_ratio": sr,
                    "commit_types": {commit_type: 2},
                }
            )
            commits.append(
                {
                    "repo": repo,
                    "timestamp": f"2024-{month:02d}-{dd:02d}T10:30:00",
                    "model": model,
                    "churn": 20,
                }
            )
    return sessions, commits


def _strat_data(
    spec: list[tuple[str, str, int, float]],
) -> tuple[list[dict], list[dict]]:
    """Build sessions per (task_type, model, n, survivor_ratio) group.

    ``commit_types`` fixes each session's stratum, so ``stratified_gap_stats``
    can hold task-type composition constant.
    """
    sessions: list[dict] = []
    commits: list[dict] = []
    for k, (tt, model, n, sr) in enumerate(spec):
        for j in range(n):
            repo = f"s{k}-{j}"
            mm = (k % 12) + 1
            dd = (j % 28) + 1
            sessions.append(
                {
                    "repo": repo,
                    "start_ts": f"2024-{mm:02d}-{dd:02d}T10:00:00",
                    "end_ts": f"2024-{mm:02d}-{dd:02d}T12:00:00",
                    "n_commits": 2,
                    "survivor_ratio": sr,
                    "commit_types": {tt: 2},
                }
            )
            commits.append(
                {
                    "repo": repo,
                    "timestamp": f"2024-{mm:02d}-{dd:02d}T10:30:00",
                    "model": model,
                    "churn": 20,
                }
            )
    return sessions, commits


def _decay_data(
    opus_seqs: list[list[int]],
    sonnet_seqs: list[list[int]],
) -> tuple[list[dict], list[dict]]:
    """Build multi-commit sessions with prescribed per-commit churn sequences.

    Each sequence becomes one session (n_commits = len(seq)); consecutive
    churns set the geometric decay the session contributes to per-tier rho.
    """
    sessions: list[dict] = []
    commits: list[dict] = []
    for k, (seqs, model) in enumerate(
        [(opus_seqs, "Claude Opus 4.8"), (sonnet_seqs, "Claude Sonnet 4.6")]
    ):
        for j, churns in enumerate(seqs):
            repo = f"d{k}-{j}"
            mm = (k % 12) + 1
            dd = (j % 28) + 1
            sessions.append(
                {
                    "repo": repo,
                    "start_ts": f"2024-{mm:02d}-{dd:02d}T10:00:00",
                    "end_ts": f"2024-{mm:02d}-{dd:02d}T13:00:00",
                    "n_commits": len(churns),
                    "survivor_ratio": 0.5,
                    "commit_types": {"other": len(churns)},
                }
            )
            for i, ch in enumerate(churns):
                commits.append(
                    {
                        "repo": repo,
                        "timestamp": f"2024-{mm:02d}-{dd:02d}T10:{10 + i * 5:02d}:00",
                        "model": model,
                        "churn": ch,
                    }
                )
    return sessions, commits


# ---------------------------------------------------------------------------
# gap_sensitivity_evalue
# ---------------------------------------------------------------------------


def test_gap_evalue_returns_expected_keys() -> None:
    sessions, commits = _tier_sessions([0.75, 0.8, 0.85, 0.82, 0.78], [0.2, 0.25, 0.15, 0.3, 0.22])
    result = sim.gap_sensitivity_evalue(sessions, commits)
    for key in (
        "cohens_d",
        "rr_approx",
        "rr_ci",
        "evalue_point",
        "evalue_ci",
        "n_opus",
        "n_sonnet",
    ):
        assert key in result, f"missing key {key}"


def test_gap_evalue_insufficient_data_errors() -> None:
    sessions, commits = _tier_sessions([0.5], [0.5])
    assert "error" in sim.gap_sensitivity_evalue(sessions, commits)


def test_gap_evalue_identical_tiers_needs_no_confounding() -> None:
    """NEGATIVE TEST: when the two tiers share the same survivor-ratio
    distribution the gap is zero, so the point E-value is exactly 1.0 (no
    confounding needed) and the CI E-value is 1.0 (the CI includes the null).

    If this ever returns an E-value materially above 1.0 for identical groups
    the sensitivity analysis is fabricating confounding that is not there.
    """
    dist = [0.4, 0.5, 0.6, 0.55, 0.45, 0.5]
    sessions, commits = _tier_sessions(dist, list(dist))
    result = sim.gap_sensitivity_evalue(sessions, commits)
    assert abs(result["cohens_d"]) < 1e-9
    assert result["evalue_point"] == pytest.approx(1.0, abs=1e-9)
    assert result["evalue_ci"] == pytest.approx(1.0, abs=1e-9)


def test_gap_evalue_separated_tiers_need_strong_confounding() -> None:
    """A large, clean separation demands a large E-value (point > 1.5) and,
    because the CI excludes the null, a CI E-value strictly above 1.0."""
    sessions, commits = _tier_sessions(
        [0.80, 0.85, 0.82, 0.88, 0.79, 0.83],
        [0.18, 0.22, 0.15, 0.25, 0.20, 0.17],
    )
    result = sim.gap_sensitivity_evalue(sessions, commits)
    assert result["cohens_d"] > 0
    assert result["evalue_point"] > 1.5
    assert result["evalue_ci"] > 1.0


def test_gap_evalue_point_dominates_ci_bound() -> None:
    """PROPERTY: the point E-value is never below the CI E-value — the CI bound
    is nearer the null by construction, so it can never require *more*
    confounding than the point estimate."""
    sessions, commits = _tier_sessions([0.7, 0.75, 0.72, 0.78, 0.68], [0.4, 0.45, 0.38, 0.5, 0.42])
    result = sim.gap_sensitivity_evalue(sessions, commits)
    assert result["evalue_point"] >= result["evalue_ci"]


def test_gap_evalue_monotone_in_separation() -> None:
    """Wider tier separation ⇒ larger point E-value."""
    small_s, small_c = _tier_sessions([0.58, 0.62, 0.6, 0.64, 0.56], [0.48, 0.52, 0.5, 0.54, 0.46])
    large_s, large_c = _tier_sessions(
        [0.80, 0.85, 0.82, 0.88, 0.79], [0.18, 0.22, 0.15, 0.25, 0.20]
    )
    e_small = sim.gap_sensitivity_evalue(small_s, small_c)["evalue_point"]
    e_large = sim.gap_sensitivity_evalue(large_s, large_c)["evalue_point"]
    assert e_large > e_small


# ---------------------------------------------------------------------------
# stratified_gap_stats
# ---------------------------------------------------------------------------


def test_stratified_gap_returns_expected_keys() -> None:
    spec = [
        ("feat", "Claude Opus 4.8", 6, 0.7),
        ("feat", "Claude Sonnet 4.6", 6, 0.5),
    ]
    sessions, commits = _strat_data(spec)
    result = sim.stratified_gap_stats(sessions, commits, n_resamples=200)
    for key in ("adjusted_gap", "naive_gap", "p_value", "n_strata_used", "strata"):
        assert key in result, f"missing key {key}"


def test_stratified_gap_single_stratum_matches_naive() -> None:
    """With one task type (all 'other'), the size-weighted adjusted gap must
    equal the naive gap — stratification is a no-op when there is nothing to
    stratify on."""
    sessions, commits = _gap_data(10, 0.7, 10, 0.4)
    result = sim.stratified_gap_stats(sessions, commits, n_resamples=200)
    assert result["n_strata_used"] == 1
    assert result["adjusted_gap"] == pytest.approx(result["naive_gap"], abs=1e-9)


def test_stratified_gap_removes_composition_driven_gap() -> None:
    """KEY TEST: a naive gap that is *purely* a task-type composition artifact
    must collapse under stratification.

    Opus concentrates in the high-survivor 'feat' stratum and Sonnet in the
    low-survivor 'fix' stratum, but within each stratum the tiers are identical.
    The naive (unstratified) gap is therefore positive, while the
    task-type-adjusted gap is ~0 and the stratified permutation p-value is large
    — exactly the honest-demotion signal the paper relies on.
    """
    spec = [
        ("feat", "Claude Opus 4.8", 10, 0.8),
        ("feat", "Claude Sonnet 4.6", 5, 0.8),
        ("fix", "Claude Opus 4.8", 5, 0.4),
        ("fix", "Claude Sonnet 4.6", 10, 0.4),
    ]
    sessions, commits = _strat_data(spec)
    result = sim.stratified_gap_stats(sessions, commits, n_resamples=2000)
    assert result["n_strata_used"] == 2
    assert result["naive_gap"] > 0.1, "composition should create a spurious naive gap"
    assert abs(result["adjusted_gap"]) < 1e-9, "within-stratum gap is zero"
    assert result["p_value"] > 0.5, "adjusted gap must not be significant"


def test_stratified_gap_no_shared_stratum_errors() -> None:
    """When no task type contains both tiers, there is nothing to compare."""
    spec = [
        ("feat", "Claude Opus 4.8", 5, 0.8),
        ("fix", "Claude Sonnet 4.6", 5, 0.4),
    ]
    sessions, commits = _strat_data(spec)
    assert "error" in sim.stratified_gap_stats(sessions, commits, n_resamples=100)


# ---------------------------------------------------------------------------
# tier_decay_stats
# ---------------------------------------------------------------------------


def test_tier_decay_returns_expected_keys() -> None:
    sessions, commits = _decay_data([[64, 16, 4]] * 4, [[100, 90, 81]] * 4)
    result = sim.tier_decay_stats(sessions, commits, n_resamples=200)
    for key in (
        "rho_opus",
        "rho_sonnet",
        "rho_gap",
        "p_value",
        "n_pairs_opus",
        "n_pairs_sonnet",
        "n_sessions",
    ):
        assert key in result, f"missing key {key}"


def test_tier_decay_insufficient_data_errors() -> None:
    """All sessions shorter than 3 commits ⇒ no decay pairs ⇒ error."""
    sessions, commits = _decay_data([[100, 50]] * 3, [[100, 50]] * 3)
    assert "error" in sim.tier_decay_stats(sessions, commits, n_resamples=100)


def test_tier_decay_equal_rates_not_significant() -> None:
    """NEGATIVE TEST: identical per-tier decay sequences give rho_gap ≈ 0 and a
    non-significant p-value. If the permutation test flags identical decay
    dynamics as different, it has lost its discriminative power."""
    seq = [128, 64, 32, 16]  # constant ratio 0.5 ⇒ rho = 0.5
    sessions, commits = _decay_data([seq] * 6, [seq] * 6)
    result = sim.tier_decay_stats(sessions, commits, n_resamples=2000)
    assert result["rho_opus"] == pytest.approx(0.5, abs=1e-9)
    assert result["rho_sonnet"] == pytest.approx(0.5, abs=1e-9)
    assert abs(result["rho_gap"]) < 1e-9
    assert result["p_value"] > 0.5


def test_tier_decay_different_rates_detected() -> None:
    """Cleanly separated decay rates (Opus 0.25, Sonnet 0.9) give the correct
    sign and a small permutation p-value."""
    opus_seq = [64, 16, 4]  # ratio 0.25
    sonnet_seq = [100, 90, 81]  # ratio 0.9
    sessions, commits = _decay_data([opus_seq] * 6, [sonnet_seq] * 6)
    result = sim.tier_decay_stats(sessions, commits, n_resamples=4000)
    assert result["rho_opus"] == pytest.approx(0.25, abs=1e-9)
    assert result["rho_sonnet"] == pytest.approx(0.9, abs=1e-9)
    assert result["rho_gap"] < 0
    assert result["p_value"] < 0.05


# ---------------------------------------------------------------------------
# git_session_extractor: _dir_category
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("top_dirs", "expected"),
    [
        (["src"], "code"),
        (["lib"], "code"),
        (["app"], "code"),
        (["packages"], "code"),
        (["tests"], "test"),
        (["test"], "test"),
        (["spec"], "test"),
        (["__tests__"], "test"),
        (["terraform"], "infra"),
        (["k8s"], "infra"),
        (["infra"], "infra"),
        ([".github"], "config"),
        (["ci"], "config"),
        (["config"], "config"),
        (["docs"], "docs"),
        (["README.md"], "other"),  # no top-level dir (repo-root file)
        ([], "other"),
    ],
)
def test_dir_category_maps_known_prefixes(top_dirs: list[str], expected: str) -> None:
    assert gse._dir_category(top_dirs) == expected


def test_dir_category_plurality_wins_over_minority() -> None:
    # 2x "src" (code) vs 1x "docs" — code should win
    assert gse._dir_category(["src", "src", "docs"]) == "code"


def test_top_level_dir_extracts_first_segment() -> None:
    assert gse._top_level_dir("src/app/main.py") == "src"


def test_top_level_dir_repo_root_file_returns_empty() -> None:
    assert gse._top_level_dir("README.md") == ""


# ---------------------------------------------------------------------------
# git_session_extractor: _hash_id / _get_salt (developer anonymization)
# ---------------------------------------------------------------------------


def test_hash_id_format() -> None:
    h = gse._hash_id("some-salt", "person@example.com", prefix="dev")
    assert h.startswith("dev_")
    assert len(h) == len("dev_") + 8


def test_hash_id_deterministic_for_same_input() -> None:
    a = gse._hash_id("salt1", "person@example.com", prefix="dev")
    b = gse._hash_id("salt1", "person@example.com", prefix="dev")
    assert a == b


def test_hash_id_differs_for_different_emails() -> None:
    a = gse._hash_id("salt1", "alice@example.com", prefix="dev")
    b = gse._hash_id("salt1", "bob@example.com", prefix="dev")
    assert a != b


def test_hash_id_differs_for_different_salts() -> None:
    """NEGATIVE TEST: without the salt, the same email always hashes to the same
    id across different deployments/datasets — the salt is what prevents
    cross-dataset re-identification via a rainbow-table attack."""
    a = gse._hash_id("salt1", "person@example.com", prefix="dev")
    b = gse._hash_id("salt2", "person@example.com", prefix="dev")
    assert a != b


def test_hash_id_never_contains_raw_email() -> None:
    h = gse._hash_id("some-salt", "person@example.com", prefix="dev")
    assert "person" not in h
    assert "@" not in h


def test_get_salt_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPANY_HASH_SALT", raising=False)
    with pytest.raises(OSError, match="COMPANY_HASH_SALT"):
        gse._get_salt()


def test_get_salt_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPANY_HASH_SALT", "test-salt-value")
    assert gse._get_salt() == "test-salt-value"


# ---------------------------------------------------------------------------
# git_session_extractor: extraction-mode env config.
#
# These four helpers were lifted out of ``main`` to bring it under the complexity
# ceiling, and the move revealed that none of this was ever covered -- it had been
# sitting inside a function that ``[tool.coverage.report].exclude_lines`` exempts as a
# CLI entrypoint, so the validation below was unmeasured rather than tested. It decides
# which cohort gets extracted and which roster is applied, so a silent change here
# would mislabel a whole dataset.
# ---------------------------------------------------------------------------


def _clear_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "COMPANY_WITHIN_ERA_MODE",
        "COMPANY_WITHIN_ERA_SINCE",
        "COMPANY_WITHIN_ERA_UNTIL",
        "COMPANY_CONTROL_MODE",
        "COMPANY_CONTROL_SINCE",
        "COMPANY_CONTROL_UNTIL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_within_era_config_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mode_env(monkeypatch)
    assert gse._within_era_config() == ("", None, None)


def test_within_era_config_reads_side_and_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_WITHIN_ERA_MODE", "ai")
    monkeypatch.setenv("COMPANY_WITHIN_ERA_SINCE", "2024-01-01")
    monkeypatch.setenv("COMPANY_WITHIN_ERA_UNTIL", "2024-11-11")
    assert gse._within_era_config() == ("ai", "2024-01-01", "2024-11-11")


@pytest.mark.parametrize("side", ["ai", "nonai"])
def test_within_era_config_accepts_both_sides(monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_WITHIN_ERA_MODE", side)
    monkeypatch.setenv("COMPANY_WITHIN_ERA_SINCE", "2024-01-01")
    monkeypatch.setenv("COMPANY_WITHIN_ERA_UNTIL", "2024-11-11")
    assert gse._within_era_config()[0] == side


def test_within_era_config_rejects_unknown_side(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGATIVE TEST: a typo'd side must fail loudly, not pick a default.

    Both sides must differ in exactly one respect; silently treating an unrecognised
    value as one side or the other would produce an incomparable pair of artifacts.
    """
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_WITHIN_ERA_MODE", "AI")
    monkeypatch.setenv("COMPANY_WITHIN_ERA_SINCE", "2024-01-01")
    monkeypatch.setenv("COMPANY_WITHIN_ERA_UNTIL", "2024-11-11")
    with pytest.raises(OSError, match="expected 'ai' or 'nonai'"):
        gse._within_era_config()


@pytest.mark.parametrize("present", ["COMPANY_WITHIN_ERA_SINCE", "COMPANY_WITHIN_ERA_UNTIL"])
def test_within_era_config_requires_both_window_bounds(
    monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    """NEGATIVE TEST: half a window is not a window.

    The window is what makes the two sides comparable, so it is required rather than
    defaulted -- with only one bound set the run must refuse.
    """
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_WITHIN_ERA_MODE", "ai")
    monkeypatch.setenv(present, "2024-01-01")
    with pytest.raises(OSError, match="COMPANY_WITHIN_ERA_SINCE"):
        gse._within_era_config()


def test_control_config_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mode_env(monkeypatch)
    assert gse._control_config() == (False, None, None)


def test_control_config_reads_window_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_CONTROL_MODE", "1")
    monkeypatch.setenv("COMPANY_CONTROL_SINCE", "2021-11-08")
    monkeypatch.setenv("COMPANY_CONTROL_UNTIL", "2022-09-19")
    assert gse._control_config() == (True, "2021-11-08", "2022-09-19")


def test_control_config_only_1_enables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGATIVE TEST: the flag is an exact "1", so "true"/"0" leave control mode off."""
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_CONTROL_MODE", "true")
    assert gse._control_config()[0] is False


def test_control_config_requires_until_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEGATIVE TEST: without a cutoff there is no pre-AI window to extract."""
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_CONTROL_MODE", "1")
    with pytest.raises(OSError, match="COMPANY_CONTROL_UNTIL"):
        gse._control_config()


def test_control_config_since_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mode_env(monkeypatch)
    monkeypatch.setenv("COMPANY_CONTROL_MODE", "1")
    monkeypatch.setenv("COMPANY_CONTROL_UNTIL", "2022-09-19")
    assert gse._control_config() == (True, None, "2022-09-19")


@pytest.mark.parametrize(
    ("within_era", "control_mode", "expected"),
    [
        ("", False, "data/company_sessions.json"),
        ("", True, "data/company_control_sessions.json"),
        ("ai", False, "data/company_within_era_ai_sessions.json"),
        ("nonai", False, "data/company_within_era_nonai_sessions.json"),
        # within-era wins over control mode: it is checked first.
        ("ai", True, "data/company_within_era_ai_sessions.json"),
    ],
)
def test_output_path_defaults_per_mode(
    monkeypatch: pytest.MonkeyPatch, within_era: str, control_mode: bool, expected: str
) -> None:
    for name in (
        "COMPANY_RAW_OUTPUT",
        "COMPANY_CONTROL_RAW_OUTPUT",
        "COMPANY_WITHIN_ERA_RAW_OUTPUT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert gse._output_path(within_era, control_mode) == Path(expected)


@pytest.mark.parametrize(
    ("within_era", "control_mode", "env_var"),
    [
        ("", False, "COMPANY_RAW_OUTPUT"),
        ("", True, "COMPANY_CONTROL_RAW_OUTPUT"),
        ("ai", False, "COMPANY_WITHIN_ERA_RAW_OUTPUT"),
    ],
)
def test_output_path_each_mode_has_its_own_override(
    monkeypatch: pytest.MonkeyPatch, within_era: str, control_mode: bool, env_var: str
) -> None:
    """Each mode reads a *different* override, so one cohort cannot overwrite another."""
    # Never opened -- _output_path only builds a Path, so this needs no real directory.
    monkeypatch.setenv(env_var, "elsewhere/override.json")
    assert gse._output_path(within_era, control_mode) == Path("elsewhere/override.json")


def test_summary_json_counts_per_repo_label() -> None:
    sessions = [{"repo": "repo_A"}, {"repo": "repo_A"}, {"repo": "repo_B"}]
    commits = [{"repo": "repo_A"}, {"repo": "repo_B"}, {"repo": "repo_B"}]
    summary = json.loads(gse._summary_json(sessions, commits, {"/x": "repo_A", "/y": "repo_B"}))
    assert summary["total_sessions"] == 3
    assert summary["total_claude_commits"] == 3
    assert summary["by_repo"] == {
        "repo_A": {"sessions": 2, "commits": 1},
        "repo_B": {"sessions": 1, "commits": 2},
    }


def test_summary_json_leaks_no_repo_paths() -> None:
    """The summary goes to stdout, so it must carry anonymized labels only."""
    labels = {"/private/clones/secret-project": "repo_A"}
    summary = gse._summary_json([{"repo": "repo_A"}], [{"repo": "repo_A"}], labels)
    assert "secret-project" not in summary
    assert "/private" not in summary


def test_commit_metrics_has_no_email_field() -> None:
    """Structural guard: CommitMetrics must never grow an email/author field —
    only the pre-hashed ``developer`` id."""
    field_names = {f for f in gse.CommitMetrics.__dataclass_fields__}
    assert "email" not in field_names
    assert "author_email" not in field_names
    assert "developer" in field_names


# ---------------------------------------------------------------------------
# git_session_extractor: _remote_branches / _dominant_branch (local temp repo)
# ---------------------------------------------------------------------------


def _init_repo_with_branches(tmp_path: Path, branches: dict[str, int]) -> str:
    """Create a throwaway local git repo (no external data) with one commit
    per entry in ``branches``, ``n`` of which are Claude co-authored, on each
    named branch. Returns the repo path as a string."""
    import subprocess

    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
    (tmp_path / "repo" / "f.txt").write_text("seed\n")
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "seed"], check=True)

    for branch, n_claude in branches.items():
        if branch != "main":
            subprocess.run(["git", "-C", repo, "checkout", "-q", "-b", branch, "main"], check=True)
        else:
            subprocess.run(["git", "-C", repo, "checkout", "-q", "main"], check=True)
        for i in range(n_claude):
            (tmp_path / "repo" / "f.txt").write_text(f"{branch}-{i}\n")
            subprocess.run(["git", "-C", repo, "add", "."], check=True)
            msg = f"work {i}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
            subprocess.run(["git", "-C", repo, "commit", "-q", "-m", msg], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "-q", "main"], check=True)

    # Simulate "already fetched" remote-tracking refs without a real remote:
    # the extractor only ever reads local refs, never fetches.
    for branch in branches:
        subprocess.run(
            [
                "git",
                "-C",
                repo,
                "update-ref",
                f"refs/remotes/origin/{branch}",
                f"refs/heads/{branch}",
            ],
            check=True,
        )
    return repo


def test_remote_branches_lists_names_without_origin_prefix(tmp_path: Path) -> None:
    repo = _init_repo_with_branches(tmp_path, {"main-like": 1, "other": 1})
    assert set(gse._remote_branches(repo)) == {"main-like", "other"}


def test_remote_branches_skips_the_origin_head_symbolic_ref(tmp_path: Path) -> None:
    """``git branch -r`` prints ``origin/HEAD -> origin/main`` for the symbolic ref.
    That is a pointer, not a branch: counting it would add a phantom "HEAD" branch
    that duplicates whichever branch it targets, double-counting its commits in
    ``_dominant_branch``. Any real clone has this ref, so the skip is load-bearing
    everywhere except the synthetic repos the other tests build."""
    import subprocess

    repo = _init_repo_with_branches(tmp_path, {"main": 1, "other": 1})
    subprocess.run(
        ["git", "-C", repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
        check=True,
    )
    raw = gse._git(repo, "branch", "-r")
    assert "->" in raw, "precondition: git must print the symbolic ref with an arrow"
    assert set(gse._remote_branches(repo)) == {"main", "other"}
    assert "HEAD" not in gse._remote_branches(repo)


# ---------------------------------------------------------------------------
# git_session_extractor: _get_repos / _repo_label_map (config + repo
# anonymization). No real repos: _git is stubbed so the label ordering is
# determined entirely by the salt.
# ---------------------------------------------------------------------------


def test_get_repos_requires_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing closed matters here: silently returning [] would make the extractor
    write an empty dataset that looks like a successful run."""
    monkeypatch.delenv("COMPANY_REPOS", raising=False)
    with pytest.raises(OSError, match="COMPANY_REPOS"):
        gse._get_repos()


def test_get_repos_splits_and_strips_and_drops_empties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPANY_REPOS", " /a/one , ,/b/two,  ")
    assert gse._get_repos() == ["/a/one", "/b/two"]


def _stub_remote_urls(monkeypatch: pytest.MonkeyPatch, urls: dict[str, str]) -> None:
    def _fake_git(repo: str, *args: str) -> str:
        assert args == ("config", "--get", "remote.origin.url")
        return urls[repo]

    monkeypatch.setattr(gse, "_git", _fake_git)


def test_repo_label_map_assigns_sequential_letters(monkeypatch: pytest.MonkeyPatch) -> None:
    repos = ["/r/alpha", "/r/beta", "/r/gamma"]
    _stub_remote_urls(monkeypatch, {r: f"https://example.com{r}.git" for r in repos})
    labels = gse._repo_label_map(repos, salt="s3cr3t")
    assert sorted(labels.values()) == ["repo_A", "repo_B", "repo_C"]
    assert len(set(labels.values())) == 3


def test_repo_label_order_follows_the_salt_not_the_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering must come from HMAC(salt, remote_url), so changing only the salt
    must be able to permute the labels. If labels tracked input order (or size, or
    alphabetical name) instead, the letters would leak repo identity -- the whole
    point of the indirection."""
    repos = ["/r/alpha", "/r/beta", "/r/gamma", "/r/delta"]
    _stub_remote_urls(monkeypatch, {r: f"https://example.com{r}.git" for r in repos})
    one = gse._repo_label_map(repos, salt="salt-one")
    two = gse._repo_label_map(repos, salt="salt-two")
    assert one != two, "a different salt must be able to reorder the labels"
    # Same salt, shuffled input -> identical mapping (order-independent).
    assert gse._repo_label_map(list(reversed(repos)), salt="salt-one") == one


def test_repo_label_map_rolls_over_past_z(monkeypatch: pytest.MonkeyPatch) -> None:
    """27 repos must produce a 27th distinct label, not a collision with repo_A.
    The rollover is bijective base-26 (A..Z, then AA), so the count of unique
    labels is the assertion that matters."""
    repos = [f"/r/{i:03d}" for i in range(27)]
    _stub_remote_urls(monkeypatch, {r: f"https://example.com{r}.git" for r in repos})
    labels = gse._repo_label_map(repos, salt="s")
    assert len(set(labels.values())) == 27
    assert all(v.startswith("repo_") for v in labels.values())
    two_letter = [v for v in labels.values() if len(v) == len("repo_AA")]
    assert two_letter == ["repo_AA"]


def test_dominant_branch_picks_branch_with_most_claude_commits(tmp_path: Path) -> None:
    """Regression guard for the bug this Phase 2 pass fixed: some repos' real
    working branch is not named ``main`` and is not ``origin/HEAD`` — the
    dominant-commit-count criterion must find it regardless of naming."""
    repo = _init_repo_with_branches(tmp_path, {"main": 0, "1.x": 5, "scratch": 1})
    assert gse._dominant_branch(repo) == "1.x"


def test_dominant_branch_falls_back_to_checked_out_head_with_no_remote_branches(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = str(tmp_path / "solo")
    subprocess.run(["git", "init", "-q", "-b", "trunk", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
    (tmp_path / "solo" / "f.txt").write_text("seed\n")
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "seed"], check=True)
    assert gse._dominant_branch(repo) == "trunk"


# ---------------------------------------------------------------------------
# git_session_extractor: extract_claude_commits control-mode parameters
# (since/until/require_ai_trailer/exclude_ai_trailer) — local temp repo, no
# external or client data touched.
# ---------------------------------------------------------------------------

CLAUDE_TRAILER = "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
COPILOT_TRAILER = "Co-Authored-By: GitHub Copilot <copilot@github.com>"


def _init_repo_with_commits(tmp_path: Path, commits: list[tuple[str, str]]) -> str:
    """Create a throwaway local git repo (no external or client data) with one
    commit per ``(iso_timestamp, message)`` pair on the default branch, at the
    given timestamps (via ``GIT_AUTHOR_DATE``/``GIT_COMMITTER_DATE``, not wall
    clock, so the test is deterministic)."""
    import os
    import subprocess

    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
    for i, (ts, msg) in enumerate(commits):
        (tmp_path / "repo" / "f.txt").write_text(f"line {i}\n")
        subprocess.run(["git", "-C", repo, "add", "."], check=True)
        env = {**os.environ, "GIT_AUTHOR_DATE": ts, "GIT_COMMITTER_DATE": ts}
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", msg], check=True, env=env)
    subprocess.run(
        ["git", "-C", repo, "update-ref", "refs/remotes/origin/main", "refs/heads/main"],
        check=True,
    )
    return repo


def _init_repo_with_authored_commits(tmp_path: Path, commits: list[tuple[str, str, str]]) -> str:
    """Like ``_init_repo_with_commits`` but with a per-commit author email
    (``(iso_timestamp, message, email)`` triples), for testing bot-author
    filtering — no external or client data touched."""
    import os
    import subprocess

    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
    for i, (ts, msg, email) in enumerate(commits):
        (tmp_path / "repo" / "f.txt").write_text(f"line {i}\n")
        subprocess.run(["git", "-C", repo, "add", "."], check=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": ts,
            "GIT_COMMITTER_DATE": ts,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_EMAIL": email,
        }
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", msg], check=True, env=env)
    subprocess.run(
        ["git", "-C", repo, "update-ref", "refs/remotes/origin/main", "refs/heads/main"],
        check=True,
    )
    return repo


def test_extract_claude_commits_excludes_bot_author_by_email(tmp_path: Path) -> None:
    """A commit from a recognizable CI/bot service account must be excluded,
    even with an otherwise-valid AI trailer — bots aren't developers."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            (
                "2022-01-01T10:00:00+00:00",
                f"work\n\n{CLAUDE_TRAILER}",
                "29139614+renovate[bot]@users.noreply.github.com",
            ),
            (
                "2022-01-02T10:00:00+00:00",
                f"work\n\n{CLAUDE_TRAILER}",
                "human@example.com",
            ),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt")
    assert len(commits) == 1
    assert commits[0].timestamp.day == 2


def test_extract_claude_commits_excludes_bot_author_in_control_mode(tmp_path: Path) -> None:
    """Same exclusion applies in control mode — a bot's pre-cutoff commits must
    not be misclassified as part of the "fully human" baseline."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            (
                "2022-01-01T10:00:00+00:00",
                "chore: dependency bump",
                "dependabot[bot]@users.noreply.github.com",
            ),
            (
                "2022-01-02T10:00:00+00:00",
                "plain human commit, no trailer",
                "human@example.com",
            ),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert len(commits) == 1
    assert commits[0].timestamp.day == 2


def test_extract_claude_commits_does_not_exclude_generic_github_noreply(
    tmp_path: Path,
) -> None:
    """Regression guard: plain 'noreply@github.com' (web-UI edits, and GitHub's
    own email-privacy feature for real humans) must NOT be excluded — only
    the '[bot]' convention and named automation accounts are bot signals."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}", "noreply@github.com")],
    )
    commits = gse.extract_claude_commits(repo, "salt")
    assert len(commits) == 1


def test_extract_claude_commits_default_behaviour_unchanged(tmp_path: Path) -> None:
    """Regression guard: calling with no new kwargs must behave exactly as before."""
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
            ("2022-01-02T10:00:00+00:00", "plain human commit, no trailer"),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt")
    assert len(commits) == 1
    assert commits[0].model is not None
    assert "Sonnet" in commits[0].model


# ---------------------------------------------------------------------------
# extract_claude_commits: identity_resolver (alias-merged developer identity)
# ---------------------------------------------------------------------------


def test_extract_claude_commits_identity_resolver_merges_distinct_pairs(tmp_path: Path) -> None:
    """Two commits with different (name, email) pairs must hash to the same
    developer when the resolver maps both to one canonical identity."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            (
                "2022-01-01T10:00:00+00:00",
                "plain human commit",
                "handle123@users.noreply.github.com",
            ),
            ("2022-01-02T10:00:00+00:00", "plain human commit two", "real.name@example.com"),
        ],
    )

    def resolver(name: str, email: str) -> str | None:
        return "canonical person" if email.endswith("example.com") or "handle123" in email else None

    commits = gse.extract_claude_commits(
        repo, "salt", require_ai_trailer=False, identity_resolver=resolver
    )
    assert len(commits) == 2
    assert commits[0].developer == commits[1].developer


def test_extract_claude_commits_identity_resolver_none_return_falls_back_to_name(
    tmp_path: Path,
) -> None:
    """A resolver that returns None for a given (name, email) must fall back to
    the default casefold(name) behaviour for that commit, not crash or merge it
    with something else."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [("2022-01-01T10:00:00+00:00", "plain human commit", "someone@example.com")],
    )
    commits_with_resolver = gse.extract_claude_commits(
        repo, "salt", require_ai_trailer=False, identity_resolver=lambda name, email: None
    )
    commits_without = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert commits_with_resolver[0].developer == commits_without[0].developer


def test_extract_claude_commits_omitting_identity_resolver_unchanged(tmp_path: Path) -> None:
    """Regression guard: not passing identity_resolver at all (the AI-tier
    extraction path) must reproduce today's casefold(name) hashing exactly."""
    repo = _init_repo_with_commits(
        tmp_path, [("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}")]
    )
    commits = gse.extract_claude_commits(repo, "salt")
    expected = gse._hash_id("salt", "test".casefold(), prefix="dev")
    assert commits[0].developer == expected


def test_extract_claude_commits_until_excludes_commits_after_cutoff(tmp_path: Path) -> None:
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
            ("2023-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt", until="2022-06-01")
    assert len(commits) == 1
    assert commits[0].timestamp.year == 2022


def test_extract_claude_commits_since_excludes_commits_before_window(tmp_path: Path) -> None:
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
            ("2023-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt", since="2022-06-01")
    assert len(commits) == 1
    assert commits[0].timestamp.year == 2023


def test_extract_claude_commits_require_ai_trailer_false_keeps_human_commits(
    tmp_path: Path,
) -> None:
    repo = _init_repo_with_commits(
        tmp_path, [("2022-01-01T10:00:00+00:00", "plain human commit, no trailer")]
    )
    commits = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert len(commits) == 1
    assert commits[0].model is None


def test_extract_claude_commits_exclude_ai_trailer_drops_claude_even_when_not_required(
    tmp_path: Path,
) -> None:
    """Defense in depth: control mode (require_ai_trailer=False) must still drop
    any commit carrying an AI co-author trailer, even if the date cutoff should
    already guarantee none exist."""
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}"),
            ("2022-01-02T10:00:00+00:00", "plain human commit, no trailer"),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert len(commits) == 1
    assert commits[0].timestamp.day == 2


def test_extract_claude_commits_exclude_ai_trailer_drops_non_claude_ai_tools(
    tmp_path: Path,
) -> None:
    """The exclusion net must cover AI tools other than Claude (e.g. Copilot) —
    not just the CLAUDE_CO_AUTHOR_RE pattern used for inclusion."""
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{COPILOT_TRAILER}"),
            ("2022-01-02T10:00:00+00:00", "plain human commit, no trailer"),
        ],
    )
    commits = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert len(commits) == 1
    assert commits[0].timestamp.day == 2


def test_extract_claude_commits_exclude_ai_trailer_false_keeps_ai_commits_when_asked(
    tmp_path: Path,
) -> None:
    """exclude_ai_trailer=False is an explicit opt-out — verifies the flag actually
    does something, rather than the exclusion being hardcoded on."""
    repo = _init_repo_with_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", f"work\n\n{COPILOT_TRAILER}"),
            ("2022-01-02T10:00:00+00:00", "plain human commit, no trailer"),
        ],
    )
    commits = gse.extract_claude_commits(
        repo, "salt", require_ai_trailer=False, exclude_ai_trailer=False
    )
    assert len(commits) == 2


def test_extract_claude_commits_all_branches_ignores_git_notes(tmp_path: Path) -> None:
    """``use_all_branches=True`` must not ingest ``refs/notes/*`` commits as developer
    commits.

    This is the control cohort's exposure specifically. ``control_mode`` and the non-AI
    within-era side both call this with ``require_ai_trailer=False``, so the trailer
    requirement that shields the AI tier does not apply, and ``BOT_AUTHOR_RE`` does not
    match typical note-writer identities. Without the ref exclusion a note commit is
    counted as a pre-cutoff human commit and lands in ``tau_human``.
    """
    import os
    import subprocess

    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "dev@corp.example"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Dev"], check=True)
    (tmp_path / "repo" / "f.txt").write_text("work\n")
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repo, "commit", "-q", "-m", "human work"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2024-01-01T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-01T10:00:00+00:00",
        },
    )
    subprocess.run(
        ["git", "-C", repo, "notes", "--ref=ai", "add", "-m", "annotation", "HEAD"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "note-bot",
            "GIT_AUTHOR_EMAIL": "note-bot@local",
            "GIT_COMMITTER_NAME": "note-bot",
            "GIT_COMMITTER_EMAIL": "note-bot@local",
            "GIT_AUTHOR_DATE": "2024-01-02T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-02T10:00:00+00:00",
        },
    )

    commits = gse.extract_claude_commits(
        repo,
        salt="s",
        since="2023-01-01",
        until="2025-01-01",
        require_ai_trailer=False,
        use_all_branches=True,
    )
    assert len(commits) == 1, "the note commit must not be counted as a developer commit"
    # The surviving commit is the real one: dated 2024-01-01, not the note's 2024-01-02.
    assert commits[0].timestamp.date().isoformat() == "2024-01-01"
    assert {c.developer for c in commits} == {gse._hash_id("s", "dev", prefix="dev")}


def test_extract_claude_commits_use_all_branches_finds_history_on_non_dominant_branch(
    tmp_path: Path,
) -> None:
    """When a repo has zero Claude commits anywhere, _dominant_branch's pick is
    an arbitrary tie-break (every branch scores 0), not where the real history
    lives. use_all_branches=True must find pre-cutoff human commits on any
    branch regardless of that tie-break — the bug this guards against would
    otherwise silently drop most of a legacy repo's history in control mode."""
    import os
    import subprocess

    repo = str(tmp_path / "repo")
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)

    def _commit(ts: str, content: str) -> None:
        (tmp_path / "repo" / "f.txt").write_text(content + "\n")
        subprocess.run(["git", "-C", repo, "add", "."], check=True)
        env = {**os.environ, "GIT_AUTHOR_DATE": ts, "GIT_COMMITTER_DATE": ts}
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "work"], check=True, env=env)

    _commit("2024-01-01T10:00:00+00:00", "recent, post-cutoff, irrelevant")
    subprocess.run(["git", "-C", repo, "checkout", "-q", "-b", "legacy", "main"], check=True)
    _commit("2020-01-01T10:00:00+00:00", "legacy-0")
    _commit("2020-01-02T10:00:00+00:00", "legacy-1")
    subprocess.run(["git", "-C", repo, "checkout", "-q", "main"], check=True)
    for branch in ("main", "legacy"):
        subprocess.run(
            [
                "git",
                "-C",
                repo,
                "update-ref",
                f"refs/remotes/origin/{branch}",
                f"refs/heads/{branch}",
            ],
            check=True,
        )

    commits = gse.extract_claude_commits(
        repo, "salt", until="2022-09-19", require_ai_trailer=False, use_all_branches=True
    )
    assert len(commits) == 2


def test_extract_claude_commits_use_all_branches_defaults_false(tmp_path: Path) -> None:
    """Regression guard: omitting use_all_branches must not change AI-mode behavior."""
    repo = _init_repo_with_commits(
        tmp_path, [("2022-01-01T10:00:00+00:00", f"work\n\n{CLAUDE_TRAILER}")]
    )
    commits = gse.extract_claude_commits(repo, "salt")
    assert len(commits) == 1


# ---------------------------------------------------------------------------
# scripts/anonymize_sessions: sanitize() removes repo names, emails, raw SHAs
# ---------------------------------------------------------------------------


def _load_anonymize_sessions() -> ModuleType:
    """Import scripts/anonymize_sessions.py, falling back to a by-path load.

    A real ``scripts`` package from an unrelated site-packages install can shadow the
    local ``scripts/`` directory, in which case a plain
    ``import scripts.anonymize_sessions`` resolves to the wrong module. So import
    normally but *verify* the module came from this repo, and only load by path when
    it did not. The normal import is preferred because a by-path load registers the
    module under an alias name, which stops ``--cov=scripts.anonymize_sessions`` from
    resolving it and silently drops the file from the coverage gate.
    """
    import importlib.util

    path = (Path(__file__).parent / "scripts" / "anonymize_sessions.py").resolve()
    try:
        import scripts.anonymize_sessions as mod

        if Path(mod.__file__ or "").resolve() == path:
            return mod
    except ImportError:
        pass

    spec = importlib.util.spec_from_file_location("_anonymize_sessions_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_mapping_returns_empty_when_no_env_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty mapping must be a legitimate answer, not an error: it is the normal
    case when the caller relies on sanitize() alone. Raising here would break the
    default path."""
    anon = _load_anonymize_sessions()
    monkeypatch.delenv("COMPANY_REPO_MAP", raising=False)
    monkeypatch.delenv("COMPANY_REPO_MAP_FILE", raising=False)
    assert anon.load_mapping() == {}


def test_load_mapping_reads_inline_json(monkeypatch: pytest.MonkeyPatch) -> None:
    anon = _load_anonymize_sessions()
    monkeypatch.setenv("COMPANY_REPO_MAP", '{"real-repo": "repo_A"}')
    monkeypatch.delenv("COMPANY_REPO_MAP_FILE", raising=False)
    assert anon.load_mapping() == {"real-repo": "repo_A"}


def test_load_mapping_reads_file_when_inline_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    anon = _load_anonymize_sessions()
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"other-repo": "repo_B"}))
    monkeypatch.delenv("COMPANY_REPO_MAP", raising=False)
    monkeypatch.setenv("COMPANY_REPO_MAP_FILE", str(p))
    assert anon.load_mapping() == {"other-repo": "repo_B"}


def test_load_mapping_prefers_inline_over_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Documented precedence. If it silently inverted, a stale on-disk map would
    override the one the caller passed explicitly."""
    anon = _load_anonymize_sessions()
    p = tmp_path / "map.json"
    p.write_text(json.dumps({"real-repo": "FROM_FILE"}))
    monkeypatch.setenv("COMPANY_REPO_MAP", '{"real-repo": "FROM_INLINE"}')
    monkeypatch.setenv("COMPANY_REPO_MAP_FILE", str(p))
    assert anon.load_mapping() == {"real-repo": "FROM_INLINE"}


def test_apply_name_mapping_is_a_noop_for_an_empty_mapping() -> None:
    """Must leave the payload byte-identical rather than, say, rewriting `repos` to
    an empty list -- which is what a missing early return would do."""
    anon = _load_anonymize_sessions()
    data = {"repos": ["a"], "sessions": [{"repo": "a"}], "commits": [{"repo": "a"}]}
    anon.apply_name_mapping(data, {})
    assert data == {"repos": ["a"], "sessions": [{"repo": "a"}], "commits": [{"repo": "a"}]}


def test_apply_name_mapping_rewrites_mapped_names_and_leaves_others() -> None:
    """Partial mappings are the realistic case: only the repos in the map are
    renamed, and an unmapped repo must survive unchanged rather than becoming None."""
    anon = _load_anonymize_sessions()
    data = {
        "repos": ["mapped", "unmapped"],
        "sessions": [{"repo": "mapped"}, {"repo": "unmapped"}],
        "commits": [{"repo": "mapped"}, {"repo": "unmapped"}, {}],
    }
    anon.apply_name_mapping(data, {"mapped": "repo_A"})
    assert data["repos"] == ["repo_A", "unmapped"]
    assert [s["repo"] for s in data["sessions"]] == ["repo_A", "unmapped"]
    assert [c.get("repo") for c in data["commits"]] == ["repo_A", "unmapped", None]


def test_anonymize_sanitize_leaves_no_real_repo_name_or_email_or_raw_sha() -> None:
    anon = _load_anonymize_sessions()
    raw = {
        "repos": ["real-secret-repo"],
        "sessions": [
            {
                "repo": "real-secret-repo",
                "session_id": 0,
                "start_ts": "2022-01-01T10:00:00+00:00",
                "end_ts": "2022-01-01T10:30:00+00:00",
            }
        ],
        "commits": [
            {
                "repo": "real-secret-repo",
                "sha": "deadbeef1234",
                "timestamp": "2022-01-01T10:15:00+00:00",
            }
        ],
    }
    sanitized = anon.sanitize(raw, {"real-secret-repo": "repo_A"})
    serialized = json.dumps(sanitized)
    assert "real-secret-repo" not in serialized
    assert "deadbeef1234" not in serialized
    assert "@" not in serialized
    assert sanitized["sessions"][0]["start_ts"] == "2022-01-01"
    assert sanitized["commits"][0]["sha"].startswith("c")


def _overlapping_two_developer_payload(*, same_developer_only: bool) -> dict:
    """Two developers committing in one repo inside overlapping session windows.

    Developer-split grouping (control mode) would never place these four commits
    in two sessions of two developers each unless the linkage honours
    ``developer`` — the windows overlap, so a timestamp-only match is ambiguous.
    """
    return {
        "same_developer_only": same_developer_only,
        "repos": ["repo_A"],
        "sessions": [
            {
                "repo": "repo_A",
                "session_id": 0,
                "developer": "dev_aaaa",
                "start_ts": "2022-01-01T10:00:00+00:00",
                "end_ts": "2022-01-01T12:00:00+00:00",
            },
            {
                "repo": "repo_A",
                "session_id": 1,
                "developer": "dev_bbbb",
                "start_ts": "2022-01-01T10:30:00+00:00",
                "end_ts": "2022-01-01T11:30:00+00:00",
            },
        ],
        "commits": [
            {
                "repo": "repo_A",
                "sha": "a1",
                "developer": "dev_aaaa",
                "timestamp": "2022-01-01T10:15:00+00:00",
            },
            {
                "repo": "repo_A",
                "sha": "a2",
                "developer": "dev_aaaa",
                "timestamp": "2022-01-01T11:00:00+00:00",
            },
            {
                "repo": "repo_A",
                "sha": "b1",
                "developer": "dev_bbbb",
                "timestamp": "2022-01-01T10:45:00+00:00",
            },
            {
                "repo": "repo_A",
                "sha": "b2",
                "developer": "dev_bbbb",
                "timestamp": "2022-01-01T11:15:00+00:00",
            },
        ],
    }


def test_link_commits_respects_developer_when_sessions_are_developer_split() -> None:
    """Negative test for the developer condition in link_commits_to_sessions.

    Fails if the ``c.get("developer") == s.get("developer")`` clause is removed:
    without it both of dev_bbbb's commits land in session 0 (whose window
    contains them), producing a session with two distinct developers -- which
    ``same_developer_only=True`` grouping makes impossible by construction.
    """
    anon = _load_anonymize_sessions()
    data = _overlapping_two_developer_payload(same_developer_only=True)
    anon.link_commits_to_sessions(data)

    by_session: dict[int, set[str]] = {}
    for c in data["commits"]:
        assert c["session_id"] is not None, f"commit {c['sha']} left unlinked"
        by_session.setdefault(c["session_id"], set()).add(c["developer"])

    assert by_session == {0: {"dev_aaaa"}, 1: {"dev_bbbb"}}, (
        f"developer-split sessions must not mix developers, got {by_session}"
    )
    for session in data["sessions"]:
        seqs = sorted(c["seq"] for c in data["commits"] if c["session_id"] == session["session_id"])
        assert seqs == [0, 1], f"session {session['session_id']} seq not contiguous: {seqs}"


def test_link_commits_ignores_developer_when_sessions_are_not_split() -> None:
    """The AI-tier path must keep its developer-blind behaviour.

    AI-tier sessions are grouped with ``same_developer_only=False``, so a session
    legitimately spans several developers and its ``developer`` is the plurality.
    Applying the developer condition there would drop minority-developer commits,
    so the flag must gate it rather than always being on.
    """
    anon = _load_anonymize_sessions()
    data = _overlapping_two_developer_payload(same_developer_only=False)
    anon.link_commits_to_sessions(data)

    # Session 1's window is a subset of session 0's, so the later-matching
    # session wins every commit inside it — developer is not consulted at all.
    linked = {c["sha"]: c["session_id"] for c in data["commits"]}
    assert linked["a1"] == 0, "commit outside session 1's window should stay in session 0"
    assert linked["b1"] == 1
    assert linked["a2"] == 1, (
        "developer-blind matching must still place dev_aaaa's in-window commit in "
        "session 1 — gating on same_developer_only changed AI-tier behaviour"
    )


def test_link_commits_compares_instants_not_timestamp_strings() -> None:
    """NEGATIVE TEST: mixed UTC offsets must not break the window match.

    Fails if the comparison reverts to lexical string ordering. The commit occurs
    inside the session in absolute time, but its "+05:30" literal sorts after the
    session's "-04:00" end string, so a string comparison leaves it unlinked. This
    org's control cohort spans 11 distinct offsets and 382 commits were unlinked
    for exactly this reason before the fix.
    """
    anon = _load_anonymize_sessions()
    data = {
        "same_developer_only": True,
        "repos": ["repo_A"],
        "sessions": [
            {
                "repo": "repo_A",
                "session_id": 0,
                "developer": "dev_aaaa",
                # 13:00 to 21:00 UTC
                "start_ts": "2022-01-01T09:00:00-04:00",
                "end_ts": "2022-01-01T17:00:00-04:00",
            }
        ],
        "commits": [
            # 14:30 UTC — inside the window, but "20:00...+05:30" > "17:00...-04:00"
            # as a string, so lexical comparison rejects it.
            {
                "repo": "repo_A",
                "sha": "tz1",
                "developer": "dev_aaaa",
                "timestamp": "2022-01-01T20:00:00+05:30",
            },
        ],
    }
    anon.link_commits_to_sessions(data)
    assert data["commits"][0]["session_id"] == 0, (
        "commit inside the session in absolute time was not linked — "
        "the window test is comparing strings, not instants"
    )


def test_link_commits_orders_seq_by_instant_across_offsets() -> None:
    """seq must follow real chronology, since it is the intra-session order the
    simulation relies on after timestamps are coarsened to dates."""
    anon = _load_anonymize_sessions()
    data = {
        "same_developer_only": True,
        "repos": ["repo_A"],
        "sessions": [
            {
                "repo": "repo_A",
                "session_id": 0,
                "developer": "dev_aaaa",
                "start_ts": "2022-01-01T00:00:00+00:00",
                "end_ts": "2022-01-01T23:00:00+00:00",
            }
        ],
        "commits": [
            # 15:30 UTC, but its literal starts with "21"
            {
                "repo": "repo_A",
                "sha": "later_string_earlier_instant",
                "developer": "dev_aaaa",
                "timestamp": "2022-01-01T21:00:00+05:30",
            },
            # 18:00 UTC, literal starts with "18"
            {
                "repo": "repo_A",
                "sha": "earlier_string_later_instant",
                "developer": "dev_aaaa",
                "timestamp": "2022-01-01T18:00:00+00:00",
            },
        ],
    }
    anon.link_commits_to_sessions(data)
    order = {c["sha"]: c["seq"] for c in data["commits"]}
    assert order["later_string_earlier_instant"] == 0
    assert order["earlier_string_later_instant"] == 1


def test_link_commits_tolerates_naive_timestamps() -> None:
    """Existing fixtures and any offset-less extraction must keep working."""
    anon = _load_anonymize_sessions()
    data = _overlapping_two_developer_payload(same_developer_only=True)
    anon.link_commits_to_sessions(data)
    assert all(c["session_id"] is not None for c in data["commits"])


def test_link_commits_defaults_to_developer_blind_when_flag_absent() -> None:
    """Datasets predating the flag (the committed AI-tier file has no
    ``same_developer_only`` key) must keep their existing linkage semantics."""
    anon = _load_anonymize_sessions()
    data = _overlapping_two_developer_payload(same_developer_only=False)
    del data["same_developer_only"]
    anon.link_commits_to_sessions(data)
    assert {c["sha"]: c["session_id"] for c in data["commits"]}["a2"] == 1


# ---------------------------------------------------------------------------
# scripts/resolve_author_aliases: cluster_authors (pure, no git/subprocess)
# ---------------------------------------------------------------------------


def _load_resolve_author_aliases() -> ModuleType:
    """Load scripts/resolve_author_aliases.py by file path — same reason as
    ``_load_anonymize_sessions``: an unrelated ``scripts`` package on
    site-packages would otherwise shadow the local ``scripts/`` directory.

    Registers the module in ``sys.modules`` *before* executing it: the
    module's ``@dataclass``-decorated classes use postponed annotations
    (``from __future__ import annotations``), and dataclass field-type
    resolution looks the defining module up via ``sys.modules[cls.__module__]``
    — omitting this step raises ``AttributeError`` on class creation.
    """
    import importlib.util
    import sys

    path = Path(__file__).parent / "scripts" / "resolve_author_aliases.py"
    spec = importlib.util.spec_from_file_location("_resolve_author_aliases_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raa() -> ModuleType:
    return _load_resolve_author_aliases()


def test_cluster_authors_groups_same_name_different_email(raa: ModuleType) -> None:
    """Layer 1: the common case — same display name, multiple emails."""
    pairs = {
        ("Alan Turing", "alan@example.com"),
        ("Alan Turing", "12345+alan@users.noreply.github.com"),
    }
    result = raa.cluster_authors(pairs)
    canonicals = set(result.canonical.values())
    assert len(canonicals) == 1


def test_cluster_authors_keeps_distinct_names_separate(raa: ModuleType) -> None:
    pairs = {("Alice Smith", "alice@example.com"), ("Bob Jones", "bob@example.com")}
    result = raa.cluster_authors(pairs)
    canonicals = {result.canonical[p] for p in pairs}
    assert len(canonicals) == 2


def test_cluster_authors_github_id_union_merges_across_names(raa: ModuleType) -> None:
    """Layer 2: one numeric GitHub-noreply ID under two different display names is
    a provable same-account merge -- the case a renamed account produces.

    The fixture is built so that *only* Layer 2 can produce the merge, which is
    what makes this a test of Layer 2 rather than of the pipeline as a whole:
    the two casefolded names differ, so Layer 1 leaves them apart, and both names
    contain a space, so ``_layer3_merge_handles`` skips them outright and no fuzzy
    score is ever computed. The layer counts pin that attribution -- 2 clusters
    standing after Layer 1, 1 after Layer 2 -- so this test starts failing if the
    union is dropped, instead of quietly passing on a Layer 3 rescue.
    """
    pairs = {
        ("Ada Lovelace", "24680+ada@users.noreply.github.com"),
        ("Ada Byron", "24680+ada-byron@users.noreply.github.com"),
    }
    result = raa.cluster_authors(pairs)
    assert len({result.canonical[p] for p in pairs}) == 1
    assert result.layer_cluster_counts == {
        "layer1_casefold_name": 2,
        "layer2_github_id_union": 1,
        "layer3_fuzzy_handle_merge": 1,
    }
    assert not result.accepted_merges
    assert not result.unresolved


def test_cluster_authors_bare_noreply_without_numeric_id_not_unioned(raa: ModuleType) -> None:
    """The bare ``username@users.noreply.github.com`` form (no numeric ID
    prefix) has no stable per-account guarantee and must not trigger a
    Layer-2 merge on its own."""
    pairs = {
        ("some name", "somename@users.noreply.github.com"),
        ("some name", "othername@users.noreply.github.com"),
    }
    result = raa.cluster_authors(pairs)
    # Same casefold(name) already merges these at Layer 1 -- this test only
    # confirms Layer 2 doesn't additionally merge a *different*-name pair
    # sharing the bare noreply domain, which the fixture below checks.
    assert len({result.canonical[p] for p in pairs}) == 1


def test_cluster_authors_fuzzy_merges_confident_handle(raa: ModuleType) -> None:
    """Layer 3: a handle name equal to the email local-part, confidently
    similar to exactly one real name in the pool, gets merged."""
    pairs = {
        ("barbara-liskov", "13579+barbara-liskov@users.noreply.github.com"),
        ("Barbara Liskov", "barbara@example.com"),
    }
    result = raa.cluster_authors(pairs)
    assert len({result.canonical[p] for p in pairs}) == 1
    assert len(result.accepted_merges) == 1


def test_cluster_authors_fuzzy_rejects_below_threshold(raa: ModuleType) -> None:
    """A handle with no confidently-similar real name must stay unmerged and
    show up in the unresolved log, not get force-matched to the closest
    (still-dissimilar) candidate."""
    pairs = {
        ("obscurehandle99", "obscurehandle99@example.com"),
        ("Completely Different", "cd@example.com"),
    }
    result = raa.cluster_authors(pairs)
    assert len({result.canonical[p] for p in pairs}) == 2
    assert not result.accepted_merges
    assert len(result.unresolved) == 1


def test_cluster_authors_fuzzy_rejects_ambiguous_match(raa: ModuleType) -> None:
    """NEGATIVE TEST: when two real names are both plausibly similar to a
    handle (within the ambiguity margin of each other), the merge must be
    rejected rather than guessing -- ambiguity is a reason to abstain, not to
    pick the marginally-higher-scoring candidate."""
    pairs = {
        ("ansmith", "ansmith@example.com"),
        ("Ann Smith", "ann.smith@example.com"),
        ("Ana Smith", "ana.smith@example.com"),
    }
    result = raa.cluster_authors(pairs)
    handle_canonical = result.canonical[("ansmith", "ansmith@example.com")]
    other_canonicals = {
        result.canonical[("Ann Smith", "ann.smith@example.com")],
        result.canonical[("Ana Smith", "ana.smith@example.com")],
    }
    assert handle_canonical not in other_canonicals


def test_cluster_authors_real_name_pair_never_treated_as_handle(raa: ModuleType) -> None:
    """A normal 'First Last' name must never enter the handle-matching path,
    even if its email happens to look unusual."""
    pairs = {("Edsger Dijkstra", "edsger@Edsgers-MacBook-Pro.local")}
    result = raa.cluster_authors(pairs)
    assert not result.accepted_merges
    assert not result.unresolved
    assert len(result.canonical) == 1


# ---------------------------------------------------------------------------
# scripts/resolve_author_aliases: cluster_authors, cases added when the function
# was split into per-layer helpers to bring it under the complexity ceiling.
# The tests above already covered the happy path of each layer; these pin the
# behaviours the split could have quietly changed and nothing was asserting --
# the reported per-layer cluster counts, the canonical-label choice, the two
# guard branches, and order-independence.
# ---------------------------------------------------------------------------


def test_cluster_authors_reports_per_layer_cluster_counts(raa: ModuleType) -> None:
    """The three layer counts are the audit trail for how much each layer merged."""
    pairs = {
        ("Ada Lovelace", "ada@example.com"),
        ("Ada Lovelace", "ada2@example.com"),
        ("Grace Hopper", "grace@example.com"),
    }
    counts = raa.cluster_authors(pairs).layer_cluster_counts
    assert counts == {
        "layer1_casefold_name": 2,
        "layer2_github_id_union": 2,
        "layer3_fuzzy_handle_merge": 2,
    }


def test_cluster_authors_name_key_ignores_case_and_surrounding_space(raa: ModuleType) -> None:
    pairs = {("Ada Lovelace", "a@example.com"), ("  ada lovelace  ", "b@example.com")}
    assert len(set(raa.cluster_authors(pairs).canonical.values())) == 1


def test_cluster_authors_different_github_numeric_ids_stay_separate(raa: ModuleType) -> None:
    """Layer 2 must merge on a *shared* ID only -- two IDs are two accounts."""
    pairs = {
        ("Ada Lovelace", "12345+ada@users.noreply.github.com"),
        ("Grace Hopper", "67890+grace@users.noreply.github.com"),
    }
    assert len(set(raa.cluster_authors(pairs).canonical.values())) == 2


def test_cluster_authors_bare_noreply_does_not_union_different_names(raa: ModuleType) -> None:
    """NEGATIVE TEST: the case the existing bare-noreply test explicitly defers.

    That test's fixture shares one display name, so Layer 1 merges it before Layer 2
    is consulted. Two *different* names sharing the bare noreply domain is the real
    check: widening Layer 2 to the bare form would merge strangers.
    """
    pairs = {
        ("Ada Lovelace", "shared@users.noreply.github.com"),
        ("Grace Hopper", "shared@users.noreply.github.com"),
    }
    assert len(set(raa.cluster_authors(pairs).canonical.values())) == 2


def test_cluster_authors_ambiguous_match_records_no_merge_at_all(raa: ModuleType) -> None:
    """NEGATIVE TEST: strengthens the ambiguity check to the merge list itself.

    The existing ambiguity test asserts the handle lands outside both real names'
    clusters. This asserts nothing was accepted, which is what actually fails if
    FUZZY_AMBIGUITY_MARGIN is dropped -- verified by setting it to 0.0, at which point
    'onsmith' merges into 'Ron Smith' on a 0.9333 score that 'Jon Smith' ties exactly.
    """
    pairs = {
        ("Jon Smith", "jon@example.com"),
        ("Ron Smith", "ron@example.com"),
        ("onsmith", "onsmith@example.com"),
    }
    result = raa.cluster_authors(pairs)
    assert result.accepted_merges == []
    assert len(result.unresolved) == 1
    assert "2nd=" in result.unresolved[0][2]


def test_cluster_authors_strips_handle_separators_before_scoring(raa: ModuleType) -> None:
    """``-``, ``_`` and ``.`` are removed from the handle, so they cost no similarity."""
    pairs = {("Ada Lovelace", "ada@example.com"), ("ada-love_lace", "ada-love_lace@example.com")}
    assert len(set(raa.cluster_authors(pairs).canonical.values())) == 1


def test_cluster_authors_handle_with_no_real_names_in_pool_is_unresolved(raa: ModuleType) -> None:
    """The empty-candidate branch: a distinct reason string, not a low-score rejection."""
    result = raa.cluster_authors({("adalovelace", "adalovelace@example.com")})
    assert result.accepted_merges == []
    assert result.unresolved == [
        ("adalovelace", "adalovelace@example.com", "no real-name candidates in pool")
    ]


def test_cluster_authors_canonical_label_prefers_a_full_name(raa: ModuleType) -> None:
    """A cluster containing a "First Last" member is labelled with it, casefolded."""
    pairs = {("Ada Lovelace", "ada@example.com"), ("adalovelace", "adalovelace@example.com")}
    assert set(raa.cluster_authors(pairs).canonical.values()) == {"ada lovelace"}


def test_cluster_authors_canonical_covers_every_input_pair(raa: ModuleType) -> None:
    pairs = {
        ("Ada Lovelace", "ada@example.com"),
        ("adalovelace", "adalovelace@example.com"),
        ("Grace Hopper", "grace@example.com"),
        ("obscurehandle99", "obscurehandle99@example.com"),
    }
    assert set(raa.cluster_authors(pairs).canonical) == pairs


def test_cluster_authors_is_independent_of_set_iteration_order(raa: ModuleType) -> None:
    """NEGATIVE TEST: the roster must not depend on PYTHONHASHSEED.

    ``cluster_authors`` iterates a set, so any order-sensitive step inside it would
    make the published headcount vary between runs. Same pairs, two insertion orders,
    identical result.
    """
    forward = {
        ("Ada Lovelace", "ada@example.com"),
        ("adalovelace", "adalovelace@example.com"),
        ("Grace Hopper", "grace@example.com"),
    }
    backward = set(reversed(sorted(forward)))
    assert raa.cluster_authors(forward).canonical == raa.cluster_authors(backward).canonical


def test_cluster_authors_empty_input_returns_empty_resolution(raa: ModuleType) -> None:
    result = raa.cluster_authors(set())
    assert result.canonical == {}
    assert result.accepted_merges == []
    assert result.unresolved == []
    assert result.layer_cluster_counts == {
        "layer1_casefold_name": 0,
        "layer2_github_id_union": 0,
        "layer3_fuzzy_handle_merge": 0,
    }


# ---------------------------------------------------------------------------
# scripts/resolve_author_aliases: curated-roster round trip (pure, no git)
# ---------------------------------------------------------------------------


def _candidate_stats(raa: ModuleType) -> dict[tuple[str, str], object]:
    """Three raw identities: one real name with two emails, plus a bot-free
    one-off contributor the reviewer will drop."""
    stats: dict[tuple[str, str], object] = {}
    for pair, dates, repo in [
        (("Ada Lovelace", "ada@corp.example"), ["2022-01-05", "2022-03-02"], "repo_A"),
        (("Ada Lovelace", "ada@personal.example"), ["2022-02-10"], "repo_B"),
        (("driveby", "driveby@corp.example"), ["2022-04-01"], "repo_A"),
    ]:
        st = raa.AuthorStats()
        for d in dates:
            st.observe(d, repo)
        stats[pair] = st
    return stats


def test_emit_candidates_has_no_outcome_column_and_blank_decisions(raa: ModuleType) -> None:
    """The blinding rule is structural, not procedural: the reviewer cannot see
    survivor ratios because the table has no column for them."""
    stats = _candidate_stats(raa)
    resolution = raa.cluster_authors(set(stats))
    tsv = raa.emit_candidates(stats, resolution)

    header = tsv.splitlines()[0].split("\t")
    assert header == list(raa.AUDIT_COLUMNS)
    for forbidden in ("survivor", "ratio", "churn", "tau", "insertions", "deletions"):
        assert forbidden not in tsv.casefold(), f"audit table leaks outcome data: {forbidden}"

    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    assert len(rows) == 3
    assert all(r["decision"] == "" and r["person"] == "" for r in rows), (
        "decision/person must ship blank for the human to fill in"
    )
    # Highest-volume cluster first, so the decisions that move the headcount lead.
    assert rows[0]["cluster"] == "ada lovelace"
    ada = [r for r in rows if r["name"] == "Ada Lovelace"]
    assert {r["email"] for r in ada} == {"ada@corp.example", "ada@personal.example"}
    assert [r["n_commits"] for r in ada] == ["2", "1"]
    assert ada[0]["first_commit"] == "2022-01-05" and ada[0]["last_commit"] == "2022-03-02"


def test_build_reviewed_map_round_trip_drops_and_forces_merges(raa: ModuleType) -> None:
    """End-to-end: emit → edit → build. A dropped row leaves the lookup entirely
    (so the extractor excludes it); a ``person`` label overrides the cluster."""
    stats = _candidate_stats(raa)
    resolution = raa.cluster_authors(set(stats))
    rows = list(csv.DictReader(io.StringIO(raa.emit_candidates(stats, resolution)), delimiter="\t"))

    for row in rows:
        if row["name"] == "driveby":
            row["decision"] = "drop"
        else:
            row["person"] = "Ada L"  # force the two emails onto one label

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(raa.AUDIT_COLUMNS), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    payload = raa.build_reviewed_map(buf.getvalue())

    assert payload["strict"] is True, "curated map must be strict or it only merges"
    assert payload["audit"] == {
        "raw_identities": 3,
        "after_drop": 2,
        "dropped": 1,
        "after_merge": 1,
    }
    assert set(payload["lookup"].values()) == {"ada l"}
    assert "driveby\x1edriveby@corp.example" not in payload["lookup"]


def test_collect_author_stats_claude_only_scopes_to_ai_commits(
    raa: ModuleType, tmp_path: Path
) -> None:
    """The AI-tier roster must cover exactly the commits its extraction keeps.

    Without this scoping the roster collects every author who ever touched the
    AI-tier repositories -- 269 identities against the ~39 the extraction uses on
    this org's data -- so nearly all of the review pass would be spent on rows
    that can never match a commit.
    """
    trailer = "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            ("2024-01-01T10:00:00+00:00", f"ai work\n\n{trailer}", "ai@corp.example"),
            ("2024-01-02T10:00:00+00:00", "plain human work", "human@corp.example"),
        ],
    )

    everything: dict = {}
    raa._collect_author_stats(repo, "2000-01-01", "2026-12-31", everything)
    assert {e for _n, e in everything} == {"ai@corp.example", "human@corp.example"}

    claude_only: dict = {}
    raa._collect_author_stats(repo, "2000-01-01", "2026-12-31", claude_only, claude_only=True)
    assert {e for _n, e in claude_only} == {"ai@corp.example"}
    assert next(iter(claude_only.values())).n_commits == 1


def test_collect_author_stats_returns_quietly_when_git_fails(
    raa: ModuleType, tmp_path: Path
) -> None:
    """A path that is not a repo must leave the accumulator untouched rather than
    raise. The roster is built by looping over ~122 repositories, so one unreadable
    checkout must not abort the whole pass -- but it must also not invent rows."""
    into: dict = {}
    raa._collect_author_stats(str(tmp_path / "not-a-repo"), "2000-01-01", "2026-12-31", into)
    assert into == {}


def test_collect_author_stats_skips_bot_authors(raa: ModuleType, tmp_path: Path) -> None:
    """Bots are not developers. The roster feeds a manual review pass, so every bot
    row admitted is reviewer time spent on something that can never be a person."""
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            ("2024-01-01T10:00:00+00:00", "human work", "dev@corp.example"),
            ("2024-01-02T10:00:00+00:00", "bot work", "dependabot[bot]@users.noreply.github.com"),
        ],
    )
    into: dict = {}
    raa._collect_author_stats(repo, "2000-01-01", "2026-12-31", into)
    assert {e for _n, e in into} == {"dev@corp.example"}


def test_collect_author_stats_claude_only_rejects_other_ai_coauthors(
    raa: ModuleType, tmp_path: Path
) -> None:
    """The git-level narrowing is deliberately broad (``--grep=Co-Authored-By:``) and
    the extractor's own regex does the deciding, so the roster cannot drift from what
    extraction keeps. A Copilot trailer passes the grep and must then be rejected in
    Python -- if it were not, the AI-tier roster would include authors whose commits
    the Claude-only extraction never keeps.
    """
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [("2024-01-01T10:00:00+00:00", f"work\n\n{COPILOT_TRAILER}", "copilot-user@corp.example")],
    )
    into: dict = {}
    raa._collect_author_stats(repo, "2000-01-01", "2026-12-31", into, claude_only=True)
    assert into == {}


def test_report_resolution_prints_the_reviewable_audit_trail(
    raa: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fuzzy merges are the part of clustering a human must be able to audit, so
    each accepted merge must report its score *and* its runner-up: a merge accepted at
    0.91 with a 0.90 second-best is a coin flip and needs different scrutiny than one
    at 0.98 with a 0.40 runner-up. The counts must go to stderr, leaving stdout free
    for the TSV/JSON payload.
    """
    resolution = raa.AliasResolution(
        canonical={("Ada L", "ada@corp.example"): "ada l"},
        accepted_merges=[
            raa.AliasMerge(
                handle_name="adal",
                handle_email="adal@corp.example",
                matched_name="Ada L",
                score=0.912,
                second_best_score=0.400,
            )
        ],
        unresolved=[("xyz", "xyz@corp.example", "no real-name candidates in pool")],
        layer_cluster_counts={"layer1": 5, "layer3": 3},
    )
    raa._report_resolution(resolution)
    captured = capsys.readouterr()
    assert captured.out == "", "the audit trail must not pollute stdout"
    err = captured.err
    assert "layer1: 5 clusters" in err
    assert "layer3: 3 clusters" in err
    assert "1 accepted fuzzy merges" in err
    assert "'adal'" in err and "'Ada L'" in err
    assert "score=0.912" in err
    assert "2nd-best=0.400" in err
    assert "1 handle-like rows left unresolved" in err


def test_collect_author_stats_ignores_git_notes_authors(raa: ModuleType, tmp_path: Path) -> None:
    """Note commits must not enter the roster.

    ``_collect_author_stats`` walks ``--all``, which includes ``refs/notes/*``. A note
    commit is authored by whatever wrote the note -- a review bot, a CI annotator, or
    local git tooling -- and such an identity has authored no code, so counting it
    inflates the roster with rows no commit can ever match.

    This is a real regression guard, not a hypothetical: the machine this was written
    on runs a ``git-ai`` daemon that writes ``refs/notes/ai`` into repositories as git
    runs, which made this test fail intermittently (an extra ``git-ai@local``
    identity) depending on whether the daemon had touched the fixture repo yet.
    """
    import os
    import subprocess

    repo = _init_repo_with_authored_commits(
        tmp_path,
        [("2024-01-01T10:00:00+00:00", "plain work", "dev@corp.example")],
    )
    subprocess.run(
        ["git", "-C", repo, "notes", "--ref=ai", "add", "-m", "annotation", "HEAD"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "note-bot",
            "GIT_AUTHOR_EMAIL": "note-bot@local",
            "GIT_COMMITTER_NAME": "note-bot",
            "GIT_COMMITTER_EMAIL": "note-bot@local",
        },
    )
    # Precondition: the note ref really exists, so a pass cannot be vacuous.
    refs = subprocess.run(
        ["git", "-C", repo, "for-each-ref", "--format=%(refname)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "refs/notes/ai" in refs

    found: dict = {}
    raa._collect_author_stats(repo, "2000-01-01", "2026-12-31", found)
    assert {e for _n, e in found} == {"dev@corp.example"}


def test_build_reviewed_map_rejects_unrecognized_decision(raa: ModuleType) -> None:
    """A typo in the decision column must fail loudly rather than silently
    defaulting to keep — the roster is the paper's cohort definition."""
    tsv = "\t".join(raa.AUDIT_COLUMNS) + "\nc\tn\te\t1\t2022-01-01\t2022-01-01\t1\tx\t\t\tmaybe\t\n"
    with pytest.raises(ValueError, match="unrecognized decision"):
        raa.build_reviewed_map(tsv)


def test_build_reviewed_map_rejects_missing_columns(raa: ModuleType) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        raa.build_reviewed_map("name\temail\n a\tb\n")


def test_refuses_to_write_real_identities_inside_the_repo(tmp_path: Path) -> None:
    """The audit table, roster map, and cohort manifest all carry real names,
    emails, or repository paths. The rule that they live outside the repo was
    documented but unenforced, and .gitignore's ``data/*.json`` pattern only
    covers one of the places they could land."""
    repo_root = str(Path(__file__).parent)
    for inside in (f"{repo_root}/roster.json", f"{repo_root}/data/candidates.tsv", repo_root):
        with pytest.raises(SystemExit) as exc:
            gse.require_outside_repo(inside, "the alias map")
        assert exc.value.code == 1

    # A path outside the repo is accepted, and a sibling directory whose name
    # merely starts with the repo path is not mistaken for being inside it.
    gse.require_outside_repo(str(tmp_path / "roster.json"), "the alias map")
    gse.require_outside_repo(f"{repo_root}-elsewhere/roster.json", "the alias map")


# ---------------------------------------------------------------------------
# git_session_extractor: strict roster maps exclude non-members
# ---------------------------------------------------------------------------


def _write_map(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_alias_resolver_strict_excludes_unlisted(tmp_path: Path) -> None:
    path = _write_map(tmp_path, {"lookup": {"Ada\x1eada@x": "ada"}, "strict": True})
    resolve = gse.load_alias_resolver(path)
    assert resolve("Ada", "ada@x") == "ada"
    assert resolve("Stranger", "s@x") is gse.EXCLUDE_IDENTITY


def test_load_alias_resolver_non_strict_falls_back(tmp_path: Path) -> None:
    """Maps without ``strict`` keep merge-only semantics, so every alias map
    written before curation existed behaves exactly as it did."""
    path = _write_map(tmp_path, {"lookup": {"Ada\x1eada@x": "ada"}})
    resolve = gse.load_alias_resolver(path)
    assert resolve("Ada", "ada@x") == "ada"
    assert resolve("Stranger", "s@x") is None


def test_strict_resolver_drops_commits_of_unlisted_authors(tmp_path: Path) -> None:
    """The whole point of strict mode: an unlisted author contributes no commits
    and no ``dev_`` label at all.

    ``_init_repo_with_authored_commits`` fixes ``user.name`` to ``Test``, so the
    two identities differ by email — which is exactly the aliasing shape the
    roster exists to resolve.
    """
    repo = _init_repo_with_authored_commits(
        tmp_path,
        [
            ("2022-01-01T10:00:00+00:00", "listed author", "ada@corp.example"),
            ("2022-01-02T10:00:00+00:00", "unlisted author", "stranger@corp.example"),
        ],
    )
    both = gse.extract_claude_commits(repo, "salt", require_ai_trailer=False)
    assert len(both) == 2, "fixture should produce two human commits"

    resolve = gse.load_alias_resolver(
        _write_map(
            tmp_path, {"lookup": {"Test\x1eada@corp.example": "ada lovelace"}, "strict": True}
        )
    )
    kept = gse.extract_claude_commits(
        repo, "salt", require_ai_trailer=False, identity_resolver=resolve
    )
    assert len(kept) == 1, "the unlisted author's commit must be dropped, not merged"
    assert kept[0].developer == gse._hash_id("salt", "ada lovelace", prefix="dev")
    # And no label was minted from the excluded author's fallback identity.
    assert kept[0].developer != gse._hash_id("salt", "test", prefix="dev")


# ---------------------------------------------------------------------------
# mixed_model_gap_stats
# ---------------------------------------------------------------------------


def _mm_data(
    n_repos: int = 4,
    n_per_repo: int = 6,
) -> tuple[list[dict], list[dict]]:
    """Sessions spread across multiple repos/developers/task-types so the
    multilevel model has enough structure to fit variance components for."""
    sessions: list[dict] = []
    commits: list[dict] = []
    rng = random.Random(7)
    task_types = ["code", "docs", "other"]
    for r in range(n_repos):
        repo = f"repo_{r}"
        for j in range(n_per_repo):
            model = "Claude Opus 4.8" if j % 2 == 0 else "Claude Sonnet 4.6"
            developer = f"dev_{j % 3}"
            task_type = task_types[j % len(task_types)]
            sr = 0.5 + 0.05 * rng.random()
            dd = (j % 28) + 1
            sessions.append(
                {
                    "repo": repo,
                    "start_ts": f"2024-{r + 1:02d}-{dd:02d}T10:00:00",
                    "end_ts": f"2024-{r + 1:02d}-{dd:02d}T12:00:00",
                    "n_commits": 2,
                    "survivor_ratio": sr,
                    "task_type": task_type,
                    "developer": developer,
                }
            )
            commits.append(
                {
                    "repo": repo,
                    "timestamp": f"2024-{r + 1:02d}-{dd:02d}T10:30:00",
                    "model": model,
                    "churn": 20,
                }
            )
    return sessions, commits


def test_mixed_model_returns_converged_result_with_enough_data() -> None:
    sessions, commits = _mm_data(n_repos=4, n_per_repo=6)
    result = sim.mixed_model_gap_stats(sessions, commits)
    for key in (
        "converged",
        "tier_coef",
        "tier_p",
        "tier_ci",
        "vc_repo",
        "vc_developer",
        "n_sessions",
        "n_repos",
        "n_developers",
    ):
        assert key in result, f"missing key {key}"
    assert result["converged"] is True
    assert result["n_repos"] == 4


def test_mixed_model_insufficient_data_single_repo() -> None:
    """NEGATIVE TEST: a single repo gives no repo-level variance to estimate —
    the function must report converged=False rather than forcing a fit."""
    sessions, commits = _mm_data(n_repos=1, n_per_repo=6)
    result = sim.mixed_model_gap_stats(sessions, commits)
    assert result["converged"] is False
    assert result["n_repos"] == 1
    assert "error" in result


def test_mixed_model_insufficient_data_too_few_sessions() -> None:
    sessions, commits = _tier_sessions([0.6], [0.5])
    result = sim.mixed_model_gap_stats(sessions, commits)
    assert result["converged"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Capability-ordered analysis: model_cell / capability_rank / trend / mixed model
# ---------------------------------------------------------------------------


def _dev_side(taus_by_dev: dict[str, list[float]], n_commits: int = 20) -> tuple[list, list]:
    """Sessions and commits for one side of a paired comparison."""
    sessions, commits = [], []
    for dev, taus in taus_by_dev.items():
        for t in taus:
            sessions.append(
                {
                    "repo": "repo_A",
                    "session_id": len(sessions),
                    "developer": dev,
                    "survivor_ratio": t,
                    "n_commits": 2,
                    "start_ts": "2024-01-01T09:00:00",
                    "end_ts": "2024-01-01T13:00:00",
                }
            )
        for _ in range(n_commits):
            commits.append({"repo": "repo_A", "developer": dev, "model": None, "churn": 10})
    return sessions, commits


def test_paired_developer_stats_pairs_only_shared_developers() -> None:
    a_s, a_c = _dev_side({"dev_1": [0.3, 0.4], "dev_2": [0.5], "dev_only_a": [0.9]})
    b_s, b_c = _dev_side({"dev_1": [0.6, 0.7], "dev_2": [0.8], "dev_only_b": [0.1]})
    r = sim.paired_developer_stats(a_s, a_c, b_s, b_c)
    assert r["n_pairs"] == 2
    assert {p["developer"] for p in r["per_developer"]} == {"dev_1", "dev_2"}


def test_paired_developer_stats_sign_test_detects_consistent_direction() -> None:
    """Eight developers all higher on side B: the sign test must fire."""
    a = {f"dev_{i}": [0.30 + 0.01 * i] for i in range(8)}
    b = {f"dev_{i}": [0.60 + 0.01 * i] for i in range(8)}
    r = sim.paired_developer_stats(*_dev_side(a), *_dev_side(b))
    assert r["n_pairs"] == 8
    assert r["n_b_greater"] == 8
    assert r["sign_p"] < 0.01
    assert r["median_delta"] > 0.25


def test_paired_developer_stats_reports_no_direction_when_mixed() -> None:
    a = {f"dev_{i}": [0.5] for i in range(8)}
    b = {f"dev_{i}": [0.6 if i % 2 == 0 else 0.4] for i in range(8)}
    r = sim.paired_developer_stats(*_dev_side(a), *_dev_side(b))
    assert r["n_b_greater"] == 4
    assert r["sign_p"] > 0.5


def test_paired_developer_stats_commit_floor_drops_thin_developers() -> None:
    a_s, a_c = _dev_side({"dev_busy": [0.3]}, n_commits=50)
    a_s2, a_c2 = _dev_side({"dev_thin": [0.3]}, n_commits=5)
    b_s, b_c = _dev_side({"dev_busy": [0.6]}, n_commits=50)
    b_s2, b_c2 = _dev_side({"dev_thin": [0.6]}, n_commits=5)
    both = sim.paired_developer_stats(a_s + a_s2, a_c + a_c2, b_s + b_s2, b_c + b_c2)
    floored = sim.paired_developer_stats(
        a_s + a_s2, a_c + a_c2, b_s + b_s2, b_c + b_c2, min_commits_per_side=15
    )
    assert both["n_pairs"] == 2
    assert floored["n_pairs"] == 1
    assert floored["per_developer"][0]["developer"] == "dev_busy"


def test_paired_developer_stats_no_overlap_returns_error_not_crash() -> None:
    """A salt or canonical-label mismatch yields zero overlap. That must surface
    as n_pairs=0 with an error, not a misleading empty-but-successful result."""
    r = sim.paired_developer_stats(*_dev_side({"dev_x": [0.5]}), *_dev_side({"dev_y": [0.5]}))
    assert r["n_pairs"] == 0
    assert "error" in r


def test_paired_developer_stats_keeps_commit_asymmetry_visible() -> None:
    """The inclusion rules of the two datasets can differ (all commits vs only
    AI-assisted ones), so per-side commit counts must be reported rather than
    averaged away — that asymmetry is what makes volume reads uninterpretable."""
    a_s, a_c = _dev_side({"dev_1": [0.4]}, n_commits=269)
    b_s, b_c = _dev_side({"dev_1": [0.5]}, n_commits=1)
    r = sim.paired_developer_stats(a_s, a_c, b_s, b_c)
    p = r["per_developer"][0]
    assert (p["n_commits_a"], p["n_commits_b"]) == (269, 1)
    # Summarized for the prose as well, so the worst imbalance is quotable without
    # re-deriving it from the per-developer table.
    assert r["max_volume_ratio"] == pytest.approx(269.0)


def test_paired_developer_stats_reports_delta_extremes() -> None:
    """The signed range of the deltas, which is what separates the sign test from
    Wilcoxon: the latter can reach significance on magnitudes the former ignores."""
    a = {"dev_1": [0.50], "dev_2": [0.50], "dev_3": [0.50]}
    b = {"dev_1": [0.45], "dev_2": [0.60], "dev_3": [0.90]}
    r = sim.paired_developer_stats(*_dev_side(a), *_dev_side(b))
    assert r["delta_min"] == pytest.approx(-0.05)
    assert r["delta_max"] == pytest.approx(0.40)


def test_paired_developer_stats_delta_extremes_recomputed_per_floor() -> None:
    """NEGATIVE TEST: the extremes must describe the *floored* population. A thin
    developer carrying the widest delta must vanish from the range once the floor
    excludes them, or the prose would quote a spread the reported test never saw."""
    a_s, a_c = _dev_side({"dev_busy": [0.50]}, n_commits=50)
    a_s2, a_c2 = _dev_side({"dev_thin": [0.50]}, n_commits=5)
    b_s, b_c = _dev_side({"dev_busy": [0.55]}, n_commits=50)
    b_s2, b_c2 = _dev_side({"dev_thin": [0.99]}, n_commits=5)
    args = (a_s + a_s2, a_c + a_c2, b_s + b_s2, b_c + b_c2)
    assert sim.paired_developer_stats(*args)["delta_max"] == pytest.approx(0.49)
    floored = sim.paired_developer_stats(*args, min_commits_per_side=15)
    assert floored["delta_max"] == pytest.approx(0.05)


def test_paired_developer_stats_top_volume_pair_needs_volume_on_both_sides() -> None:
    """NEGATIVE TEST: 'best-measured pair' means most data on *both* sides. A
    developer with a huge side A and a single commit on side B is not well
    measured, and must lose to a smaller pair that is balanced -- otherwise the
    claim that the direction does not rest on thin data would rest on thin data."""
    lopsided_a, lopsided_ac = _dev_side({"dev_lopsided": [0.40]}, n_commits=900)
    lopsided_b, lopsided_bc = _dev_side({"dev_lopsided": [0.90]}, n_commits=1)
    even_a, even_ac = _dev_side({"dev_even": [0.40]}, n_commits=100)
    even_b, even_bc = _dev_side({"dev_even": [0.50]}, n_commits=80)
    r = sim.paired_developer_stats(
        lopsided_a + even_a,
        lopsided_ac + even_ac,
        lopsided_b + even_b,
        lopsided_bc + even_bc,
    )
    top = r["top_volume_pair"]
    assert (top["n_commits_a"], top["n_commits_b"]) == (100, 80)
    assert top["delta"] == pytest.approx(0.10)


def test_paired_developer_stats_summaries_absent_when_no_overlap() -> None:
    """No pairs means no extremes to report; the keys must be absent rather than
    present-and-meaningless (a None would read as 'computed, came out empty')."""
    r = sim.paired_developer_stats(*_dev_side({"dev_x": [0.5]}), *_dev_side({"dev_y": [0.5]}))
    for key in ("delta_min", "delta_max", "top_volume_pair", "max_volume_ratio"):
        assert key not in r


@pytest.mark.parametrize(
    ("trailer", "expected"),
    [
        ("Claude Opus 4.8", ("opus", 4.8)),
        # The (1M context) suffix is the same model with a bigger window, so it
        # must collapse onto the same cell or one model splits into two ranks.
        ("Claude Opus 4.8 (1M context)", ("opus", 4.8)),
        ("Claude Sonnet 4.6 (1M context)", ("sonnet", 4.6)),
        ("Claude Haiku 4.5", ("haiku", 4.5)),
        ("Claude Opus 5", ("opus", 5.0)),
        ("Claude Sonnet 5", ("sonnet", 5.0)),
        # Unversioned and non-model strings must be excluded, not guessed at.
        ("Claude", None),
        ("Claude Opus", None),
        (None, None),
        ("GitHub Copilot", None),
    ],
)
def test_model_cell_parses_family_and_version(
    trailer: str | None, expected: tuple[str, float] | None
) -> None:
    assert sim.model_cell(trailer) == expected


def test_capability_rank_family_ordering_puts_any_opus_above_any_sonnet() -> None:
    cells = [("opus", 4.5), ("sonnet", 5.0), ("haiku", 4.5), ("opus", 4.8)]
    ranked = sorted(cells, key=lambda c: sim.capability_rank(c, "family"))
    assert ranked == [("haiku", 4.5), ("sonnet", 5.0), ("opus", 4.5), ("opus", 4.8)]


def test_capability_rank_version_ordering_puts_newer_generation_first() -> None:
    """The two orderings must actually disagree, or the sensitivity is vacuous."""
    cells = [("opus", 4.8), ("sonnet", 5.0)]
    assert sorted(cells, key=lambda c: sim.capability_rank(c, "family")) == [
        ("sonnet", 5.0),
        ("opus", 4.8),
    ]
    assert sorted(cells, key=lambda c: sim.capability_rank(c, "version")) == [
        ("opus", 4.8),
        ("sonnet", 5.0),
    ]


def test_capability_rank_rejects_unknown_ordering() -> None:
    with pytest.raises(ValueError, match="unknown ordering"):
        sim.capability_rank(("opus", 5.0), "vibes")


def test_dominant_model_cell_takes_plurality() -> None:
    commits = [
        {"repo": "r", "timestamp": "2024-01-01T10:00:00", "model": "Claude Opus 4.7"},
        {"repo": "r", "timestamp": "2024-01-01T10:30:00", "model": "Claude Opus 4.7 (1M context)"},
        {"repo": "r", "timestamp": "2024-01-01T11:00:00", "model": "Claude Sonnet 4.6"},
    ]
    session = {"repo": "r", "start_ts": "2024-01-01T09:00:00", "end_ts": "2024-01-01T12:00:00"}
    # The two Opus 4.7 variants collapse to one cell and therefore win 2-1.
    assert sim.dominant_model_cell(session, commits) == ("opus", 4.7)


def _capability_data(taus_by_cell: dict[str, float]) -> tuple[list[dict], list[dict]]:
    """Sessions across 4 repos and 3 developers, one model trailer per cell."""
    sessions: list[dict] = []
    commits: list[dict] = []
    rng = random.Random(11)
    for i, (trailer, tau) in enumerate(taus_by_cell.items()):
        for r in range(4):
            for j in range(3):
                dd = (i * 3 + j) % 28 + 1
                ts = f"2024-{r + 1:02d}-{dd:02d}"
                sessions.append(
                    {
                        "repo": f"repo_{r}",
                        "start_ts": f"{ts}T09:00:00",
                        "end_ts": f"{ts}T13:00:00",
                        "n_commits": 2,
                        "survivor_ratio": tau + 0.01 * rng.random(),
                        "task_type": ["code", "docs", "other"][j % 3],
                        "developer": f"dev_{j}",
                    }
                )
                commits.append(
                    {
                        "repo": f"repo_{r}",
                        "timestamp": f"{ts}T10:00:00",
                        "model": trailer,
                        "churn": 20,
                    }
                )
    return sessions, commits


def test_capability_trend_detects_a_monotone_trend() -> None:
    """Positive control: if survivor ratio really does rise with capability, the
    ordered test must find it — otherwise a null tells us nothing."""
    sessions, commits = _capability_data(
        {
            "Claude Haiku 4.5": 0.30,
            "Claude Sonnet 4.6": 0.45,
            "Claude Opus 4.6": 0.60,
            "Claude Opus 4.8": 0.75,
        }
    )
    result = sim.capability_trend_stats(sessions, commits, ordering="family")
    assert result["spearman_rho"] > 0.9, result
    assert result["spearman_p"] < 1e-6
    assert result["n_cells"] == 4
    assert result["ranked_cells"] == ["haiku4.5", "sonnet4.6", "opus4.6", "opus4.8"]
    assert [c["tau"] for c in result["per_cell"]] == sorted(c["tau"] for c in result["per_cell"])


def test_capability_trend_flat_data_gives_null() -> None:
    sessions, commits = _capability_data(
        {"Claude Haiku 4.5": 0.5, "Claude Sonnet 4.6": 0.5, "Claude Opus 4.8": 0.5}
    )
    result = sim.capability_trend_stats(sessions, commits)
    assert abs(result["spearman_rho"]) < 0.3
    assert result["spearman_p"] > 0.05


def test_capability_trend_include_haiku_false_drops_the_low_end() -> None:
    sessions, commits = _capability_data(
        {"Claude Haiku 4.5": 0.30, "Claude Sonnet 4.6": 0.45, "Claude Opus 4.8": 0.75}
    )
    with_haiku = sim.capability_trend_stats(sessions, commits, include_haiku=True)
    without = sim.capability_trend_stats(sessions, commits, include_haiku=False)
    assert with_haiku["n_cells"] == 3
    assert without["n_cells"] == 2
    assert "haiku4.5" not in without["ranked_cells"]
    assert without["n_sessions"] < with_haiku["n_sessions"]


def test_capability_trend_needs_two_cells() -> None:
    sessions, commits = _capability_data({"Claude Opus 4.8": 0.5})
    result = sim.capability_trend_stats(sessions, commits)
    assert "error" in result
    assert "spearman_rho" not in result


def test_capability_trend_excludes_pure_and_unparseable_sessions() -> None:
    sessions, commits = _capability_data(
        {"Claude Haiku 4.5": 0.3, "Claude Opus 4.8": 0.7, "Claude": 0.5}
    )
    result = sim.capability_trend_stats(sessions, commits)
    # The bare "Claude" cell has no version and must not be ranked at all.
    assert result["n_cells"] == 2
    assert result["n_sessions"] == 24


def test_capability_mixed_model_converges_and_reports_rank_effect() -> None:
    sessions, commits = _capability_data(
        {
            "Claude Haiku 4.5": 0.30,
            "Claude Sonnet 4.6": 0.45,
            "Claude Opus 4.6": 0.60,
            "Claude Opus 4.8": 0.75,
        }
    )
    result = sim.capability_mixed_model_stats(sessions, commits)
    assert result["converged"] is True, result
    for key in ("cap_coef", "cap_p", "cap_ci", "vc_repo", "vc_developer", "n_cells"):
        assert key in result
    assert result["cap_coef"] > 0.1, "should recover the planted positive slope"
    assert result["cap_p"] < 0.01


def test_capability_mixed_model_insufficient_data() -> None:
    sessions, commits = _capability_data({"Claude Opus 4.8": 0.5})
    result = sim.capability_mixed_model_stats(sessions, commits)
    assert result["converged"] is False
    assert "error" in result


def test_capability_mixed_model_rejects_unknown_ordering() -> None:
    sessions, commits = _capability_data({"Claude Opus 4.8": 0.5, "Claude Haiku 4.5": 0.4})
    with pytest.raises(ValueError, match="unknown ordering"):
        sim.capability_mixed_model_stats(sessions, commits, ordering="nope")


# ---------------------------------------------------------------------------
# Figures (smoke tests) -- these are plotting functions; a numeric assertion
# can't meaningfully verify a rendered chart, but the data-processing logic
# inside each one (peak normalisation, tier filtering, KS fits) is real and
# can crash on edge cases (a zero-churn session, an empty tier). Each test
# redirects FIG_DIR to tmp_path so it never touches the real figures/ dir,
# and checks the function runs without error, writes the expected files, and
# (for fig3/fig4, which also compute and return KS statistics) returns the
# expected keys.
# ---------------------------------------------------------------------------


def test_fig1_churn_trajectories_writes_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    # Two different session lengths so the darkness-by-n_commits branch
    # (n_hi > n_lo) is exercised, not just the degenerate n_hi == n_lo case.
    short, short_commits = _make_decay_sessions(5, [200, 100, 50, 25, 12], repo_prefix="short")
    long, long_commits = _make_decay_sessions(
        5, [200, 150, 100, 75, 50, 25, 12], repo_prefix="long"
    )
    sim.fig1_churn_trajectories(short + long, short_commits + long_commits)
    assert (tmp_path / "emp_fig1_churn_trajectories.png").exists()
    assert (tmp_path / "emp_fig1_churn_trajectories.pdf").exists()


def test_fig1_churn_trajectories_handles_zero_peak_session(tmp_path, monkeypatch) -> None:
    """NEGATIVE TEST: a session with all-zero churn (peak == 0) must be
    skipped by the normalisation step (division by peak), not crash it."""
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions, commits = _make_decay_sessions(1, [0, 0, 0, 0, 0])
    sim.fig1_churn_trajectories(sessions, commits)
    assert (tmp_path / "emp_fig1_churn_trajectories.png").exists()


def test_fig2_survivor_by_tier_writes_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions, commits = _gap_data(10, 0.8, 10, 0.5)
    sim.fig2_survivor_by_tier(sessions, commits)
    assert (tmp_path / "emp_fig2_survivor_by_tier.png").exists()


def test_fig2_survivor_by_tier_handles_empty_tier(tmp_path, monkeypatch) -> None:
    """NEGATIVE TEST: a tier with no non-pure (survivor_ratio < 0.999) sessions
    must be skipped, not crash on an empty ``vals`` list."""
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions, commits = _gap_data(5, 0.999, 5, 0.6)
    sim.fig2_survivor_by_tier(sessions, commits)
    assert (tmp_path / "emp_fig2_survivor_by_tier.png").exists()


def test_fig3_empirical_vs_simulated_returns_ks_stats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions, _ = _make_sessions_and_commits(20, survivor=0.6, n_commits=3)
    params = sim.FittedParams(
        rho=0.8,
        sigma=0.5,
        tau_opus=0.6,
        tau_sonnet=0.6,
        tau_pooled=0.6,
        session_n_median=3,
        session_n_mean=3,
        n_sessions_fit=20,
        pi_pure=0.0,
    )
    sim_results = sim.run_simulation(params, n_synthetic=50)
    result = sim.fig3_empirical_vs_simulated(sessions, sim_results)
    assert "ks_stat" in result
    assert "ks_p" in result
    assert (tmp_path / "emp_fig3_empirical_vs_simulated.png").exists()


def test_fig4_session_length_dist_returns_ks_stats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions = [
        _session_dict(repo="r", start_ts="2024-01-01T10:00:00", n_commits=n)
        for n in [1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 8, 10, 12, 15, 20, 30, 40, 45]
    ]
    result = sim.fig4_session_length_dist(sessions)
    assert "ks_stat" in result
    assert "ks_p" in result
    assert result["ks_stat"] is not None
    assert result["ks_p"] is not None
    assert (tmp_path / "emp_fig4_session_length_dist.png").exists()


def test_fig4_session_length_dist_handles_constant_length(tmp_path, monkeypatch) -> None:
    """NEGATIVE TEST: every session having the same commit count gives zero
    log-variance, for which a log-normal fit is undefined -- must skip the
    fit/KS test gracefully (ks_stat/ks_p=None), not divide by zero."""
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path)
    sessions = [
        _session_dict(repo="r", start_ts="2024-01-01T10:00:00", n_commits=4) for _ in range(10)
    ]
    result = sim.fig4_session_length_dist(sessions)
    assert result["ks_stat"] is None
    assert result["ks_p"] is None
    assert (tmp_path / "emp_fig4_session_length_dist.png").exists()
