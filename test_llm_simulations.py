"""Tests for ``llm_simulations`` (the real-LLM SWE-bench refinement experiment).

These tests are written TDD-first and exercise the parts of the pipeline that do
*not* require network access or AWS credentials: the evaluator seam, the caching
layer, the leakage guard, the metric computations, the offline (stub) agent
runner, the orchestration loop, and the plotting functions. The live
mini-swe-agent / Bedrock path is thin by construction and excluded from coverage.

Run: ``uv run pytest`` (coverage gate: ``--cov-fail-under=90`` in pyproject).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

# litellm is imported here (not lazily) for its side effect: caches the module
# in sys.modules before any test runs. `_is_budget_error` lazily imports litellm
# inside an exception handler; litellm's own first import takes ~2s, and paying
# that cost mid-race in a concurrency test starves the raising thread of GIL
# time long enough for several other (near-instant, unrelated) threads to
# finish first -- a test artifact, since production always imports litellm
# well before any error handling runs (MiniSweAgentRunner's first real round
# pulls it in transitively).
import litellm  # noqa: F401
import numpy as np
import pytest

import llm_simulations as sim

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOLD = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n"
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-def add(a, b):\n"
    "-    return a - b\n"
    "+def add(a, b):\n"
    "+    return a + b\n"
)


def make_instance(idx: int = 0, gold: str = GOLD) -> dict:
    """A minimal SWE-bench-Lite-shaped record."""
    return {
        "instance_id": f"test__repo-{idx}",
        "problem_statement": "add() returns the difference instead of the sum.",
        "repo": "test/repo",
        "base_commit": "0" * 40,
        "patch": gold,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults(monkeypatch):
    for var in list(os.environ):
        if var.startswith("LLM_SIM_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    cfg = sim.Config.from_env()
    assert cfg.k == 5
    assert cfg.t == 4
    assert cfg.tau == 0.6
    assert cfg.evaluator == "proxy"
    assert cfg.agent == "mini"
    assert cfg.step_limit == 40
    # The default weak learner routes through the internal LiteLLM proxy.
    assert cfg.model == sim.DEFAULT_MODEL and cfg.model.startswith("litellm_proxy/")
    assert cfg.api_base == sim.DEFAULT_API_BASE
    # Panel evaluator defaults.
    assert cfg.judge_model == "litellm_proxy/claude-opus-4-8"
    assert cfg.panel_primary == "file_f1"
    assert cfg.apply_check is False
    # Variance / seed layer defaults.
    assert cfg.n_seeds == 1
    assert cfg.seed_base == 0
    # Weak-learner generation temperature (judges stay at temperature=0).
    assert cfg.temperature == 0.25
    # Feedback ablation: default reproduces today's (pre-ablation) behavior.
    assert cfg.feedback_mode == "diagnostic"
    # Tier assignment: default reproduces today's (pre-Phase-2) behavior.
    assert cfg.tier_policy == "fixed"
    assert cfg.tier_models == ()


def test_config_temperature_env_override(monkeypatch):
    monkeypatch.setenv("LLM_SIM_TEMPERATURE", "0.7")
    cfg = sim.Config.from_env()
    assert cfg.temperature == 0.7


def test_config_feedback_mode_env_override(monkeypatch):
    monkeypatch.setenv("LLM_SIM_FEEDBACK", "blind")
    cfg = sim.Config.from_env()
    assert cfg.feedback_mode == "blind"


def test_config_tier_policy_env_override(monkeypatch):
    monkeypatch.setenv("LLM_SIM_TIER_POLICY", "randomized")
    cfg = sim.Config.from_env()
    assert cfg.tier_policy == "randomized"


def test_config_tier_models_env_override_parses_comma_list_with_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_SIM_TIER_MODELS", " model-a , model-b ,model-c")
    cfg = sim.Config.from_env()
    assert cfg.tier_models == ("model-a", "model-b", "model-c")


def test_config_exclude_ids_env_override_parses_comma_list_with_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_SIM_EXCLUDE_IDS", " django__django-10914 , sympy__sympy-11870")
    cfg = sim.Config.from_env()
    assert cfg.exclude_ids == ("django__django-10914", "sympy__sympy-11870")


def test_config_exclude_ids_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("LLM_SIM_EXCLUDE_IDS", raising=False)
    cfg = sim.Config.from_env()
    assert cfg.exclude_ids == ()


def test_config_proxy_env_override(monkeypatch):
    monkeypatch.setenv("LITELLM_API_BASE", "https://proxy.example.test")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-123")
    cfg = sim.Config.from_env()
    assert cfg.api_base == "https://proxy.example.test"
    assert cfg.api_key == "sk-test-123"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_SIM_K", "2")
    monkeypatch.setenv("LLM_SIM_T", "3")
    monkeypatch.setenv("LLM_SIM_TAU", "0.75")
    monkeypatch.setenv("LLM_SIM_EVALUATOR", "harness")
    monkeypatch.setenv("LLM_SIM_AGENT", "stub")
    monkeypatch.setenv("LLM_SIM_MODEL", "bedrock/some-model")
    monkeypatch.setenv("LLM_SIM_STEP_LIMIT", "12")
    cfg = sim.Config.from_env()
    assert cfg.k == 2
    assert cfg.t == 3
    assert cfg.tau == 0.75
    assert cfg.evaluator == "harness"
    assert cfg.agent == "stub"
    assert cfg.model == "bedrock/some-model"
    assert cfg.step_limit == 12


def test_config_concurrency_default_and_env(monkeypatch):
    monkeypatch.delenv("LLM_SIM_CONCURRENCY", raising=False)
    assert sim.Config.from_env().concurrency == 1

    monkeypatch.setenv("LLM_SIM_CONCURRENCY", "4")
    assert sim.Config.from_env().concurrency == 4


# ---------------------------------------------------------------------------
# LiteLLM proxy routing seam
# ---------------------------------------------------------------------------


def test_litellm_kwargs_both_set():
    cfg = _cfg(api_base="https://proxy.example.test", api_key="sk-abc")
    assert sim._litellm_kwargs(cfg) == {
        "api_base": "https://proxy.example.test",
        "api_key": "sk-abc",
    }


def test_litellm_kwargs_none_set():
    # Negative test: with no proxy credentials, nothing is injected -- litellm falls
    # back to its own resolution (e.g. AWS env for a bedrock/ model).
    cfg = _cfg(api_base=None, api_key=None)
    assert sim._litellm_kwargs(cfg) == {}


def test_litellm_kwargs_partial():
    # Fails if forwarding is hardcoded rather than driven by which field is set.
    assert sim._litellm_kwargs(_cfg(api_base="https://p.test", api_key=None)) == {
        "api_base": "https://p.test"
    }
    assert sim._litellm_kwargs(_cfg(api_base=None, api_key="sk-x")) == {"api_key": "sk-x"}


def test_judge_forwards_proxy_kwargs(monkeypatch):
    # Negative test: the judge MUST forward api_base/api_key to litellm.completion,
    # else a proxy-only environment cannot reach any model.
    captured: dict = {}

    class _Msg:
        content = "0.75"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    judge = sim.LLMJudgeEvaluator(
        model="litellm_proxy/claude-haiku-4-5-20251001",
        tau=0.6,
        api_base="https://proxy.example.test",
        api_key="sk-judge",
    )
    result = judge.evaluate("some patch", make_instance())
    assert captured["api_base"] == "https://proxy.example.test"
    assert captured["api_key"] == "sk-judge"
    assert captured["model"] == "litellm_proxy/claude-haiku-4-5-20251001"
    assert result.quality == pytest.approx(0.75)
    assert result.passed is True


def test_judge_handles_unparseable_output(monkeypatch):
    # A non-numeric grade must clamp to 0.0, not raise.
    class _Msg:
        content = "not a number"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    monkeypatch.setattr(litellm, "completion", lambda **_: _Resp())
    judge = sim.LLMJudgeEvaluator(model="litellm_proxy/m", tau=0.6)
    result = judge.evaluate("p", make_instance())
    assert result.quality == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Evaluator seam
# ---------------------------------------------------------------------------


def test_proxy_evaluator_exact_match_is_quality_one():
    ev = sim.ProxyEditDistanceEvaluator(tau=0.6)
    res = ev.evaluate(GOLD, make_instance())
    assert isinstance(res, sim.PatchEvaluation)
    assert res.quality == pytest.approx(1.0, abs=1e-9)
    assert res.passed is True


def test_proxy_evaluator_empty_patch_is_low_quality():
    ev = sim.ProxyEditDistanceEvaluator(tau=0.6)
    res = ev.evaluate("", make_instance())
    assert 0.0 <= res.quality < 0.6
    assert res.passed is False


def test_proxy_evaluator_partial_patch_between_zero_and_one():
    ev = sim.ProxyEditDistanceEvaluator(tau=0.6)
    half = GOLD[: len(GOLD) // 2]
    res = ev.evaluate(half, make_instance())
    assert 0.0 < res.quality < 1.0


def test_proxy_evaluator_threshold_boundary():
    ev = sim.ProxyEditDistanceEvaluator(tau=0.0)
    # tau=0 => everything passes (quality >= 0 always).
    assert ev.evaluate("", make_instance()).passed is True


def test_harness_image_name(tmp_path):
    # No digest lockfile -> falls back to :latest with a warning.
    import unittest.mock as mock

    ev = sim.HarnessEvaluator()
    ev._digests = {}  # empty digest map (no lockfile entries)
    with mock.patch("builtins.print"):  # suppress the warning line
        assert ev.image_name("pallets__flask-4045") == (
            "swebench/sweb.eval.x86_64.pallets_1776_flask-4045:latest"
        )
        assert ev.image_name("psf__requests-1963") == (
            "swebench/sweb.eval.x86_64.psf_1776_requests-1963:latest"
        )


def _junit_case(node_id: str, *, failed: bool = False, skipped: bool = False) -> str:
    """Build one <testcase> element for a pytest --junit-xml report, given a full
    pytest node id (``path/to/test.py::Class::test[param]``). Matches production's
    ``_node_id_signature`` transform for classname/name construction."""
    file_part, _, rest = node_id.partition("::")
    parts = rest.split("::")
    name = parts[-1]
    cls_parts = parts[:-1]
    module = file_part[: -len(".py")].replace("/", ".") if file_part.endswith(".py") else file_part
    classname = module + ("." + ".".join(cls_parts) if cls_parts else "")
    body = ""
    if failed:
        body = '<failure message="boom">boom</failure>'
    elif skipped:
        body = "<skipped/>"
    return (
        f'<testcase classname="{classname}" name="{name}" file="{file_part}" line="1">'
        f"{body}</testcase>"
    )


def _junit_xml_blob(
    *,
    passed: Sequence[str] = (),
    failed: Sequence[str] = (),
    skipped: Sequence[str] = (),
) -> str:
    """Minimal pytest --junit-xml report blob, shaped like real output (this is
    what the eval script's ``cat /patches/report.xml`` appends after the console
    text) -- used to feed ``_junit_test_ids``/``_parse_junit_xml`` in tests."""
    cases = "".join(
        [
            *(_junit_case(t) for t in passed),
            *(_junit_case(t, failed=True) for t in failed),
            *(_junit_case(t, skipped=True) for t in skipped),
        ]
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites name="pytest tests"><testsuite name="pytest">{cases}</testsuite></testsuites>'
    )


def test_harness_junit_ids_all_pass():
    ftp = ["tests/test_foo.py::test_a", "tests/test_foo.py::test_b"]
    ptp = ["tests/test_foo.py::test_c"]
    xml = _junit_xml_blob(passed=[*ftp, *ptp])
    result = sim.HarnessEvaluator._parse_junit_xml(xml, ftp, ptp)
    assert result.quality == 1.0
    assert result.passed is True
    assert result.fraction_passing == pytest.approx(1.0)
    assert "ftp=2/2" in result.detail
    assert "ptp_reg=0" in result.detail


def test_harness_junit_ids_partial_ftp():
    ftp = ["tests/test_foo.py::test_a", "tests/test_foo.py::test_b"]
    ptp: list[str] = []
    xml = _junit_xml_blob(
        passed=["tests/test_foo.py::test_a"], failed=["tests/test_foo.py::test_b"]
    )
    result = sim.HarnessEvaluator._parse_junit_xml(xml, ftp, ptp)
    assert result.quality == pytest.approx(0.5)
    assert result.fraction_passing == pytest.approx(0.5)
    assert result.passed is False


def test_harness_junit_ids_ptp_regression_halves_quality():
    ftp = ["tests/test_foo.py::test_a"]
    ptp = ["tests/test_foo.py::test_c"]
    xml = _junit_xml_blob(
        passed=["tests/test_foo.py::test_a"], failed=["tests/test_foo.py::test_c"]
    )
    result = sim.HarnessEvaluator._parse_junit_xml(xml, ftp, ptp)
    assert result.quality == pytest.approx(0.5)  # 1.0 * 0.5 regression penalty
    assert result.fraction_passing == pytest.approx(1.0)  # penalty-free: ftp_pass/len(ftp)
    assert result.passed is False


def test_harness_junit_ids_all_fail():
    ftp = ["tests/test_foo.py::test_a"]
    ptp: list[str] = []
    xml = _junit_xml_blob(failed=["tests/test_foo.py::test_a"])
    result = sim.HarnessEvaluator._parse_junit_xml(xml, ftp, ptp)
    assert result.quality == 0.0
    assert result.fraction_passing == 0.0
    assert result.passed is False


def test_harness_junit_ids_empty_output():
    ftp = ["tests/test_foo.py::test_a"]
    result = sim.HarnessEvaluator._parse_junit_xml("", ftp, [])
    assert result.quality == 0.0
    assert result.fraction_passing == 0.0
    assert result.passed is False


def test_harness_junit_ids_immune_to_ansi_color_console_text():
    """The XML report is appended after the colorized -v console text; ANSI
    escapes in that preceding text must not affect XML-based scoring at all --
    this is the actual bug being fixed (naive ' PASSED' substring matching broke
    on ANSI-wrapped tokens)."""
    ftp = ["tests/test_foo.py::test_a"]
    ansi_console = "\x1b[32mtests/test_foo.py::test_a \x1b[0mPASSED\x1b[0m\n"
    raw = ansi_console + _junit_xml_blob(passed=ftp)
    result = sim.HarnessEvaluator._parse_junit_xml(raw, ftp, [])
    assert result.quality == 1.0
    assert result.passed is True


def test_harness_junit_ids_nested_class_and_parametrize():
    """Signature matching must handle a nested Test class and a parametrize
    suffix, not just flat module-level functions."""
    node_id = "tests/test_foo.py::TestGroup::test_param[1]"
    xml = _junit_xml_blob(passed=[node_id])
    passed_ids, failed_ids = sim.HarnessEvaluator._junit_test_ids(xml)
    assert sim.HarnessEvaluator._node_id_signature(node_id) in passed_ids
    assert not failed_ids


def test_harness_node_id_signature_matches_decorator_shadowed_file_attr():
    """A decorated test's <testcase file=...> can point at the *decorator's* own
    definition site, not the test's module (confirmed live: matplotlib's
    @image_comparison-wrapped tests report file=matplotlib/testing/decorators.py).
    Matching must rely on classname/name (computed from the collected item's own
    parent chain), not on reconstructing a node id from file+classname --
    negative test: a file-attribute-trusting implementation fails this."""
    node_id = "lib/mpl_toolkits/tests/test_mplot3d.py::test_invisible_axes[png]"
    classname, name = sim.HarnessEvaluator._node_id_signature(node_id)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest">'
        f'<testcase classname="{classname}" name="{name}" '
        'file="lib/matplotlib/testing/decorators.py" line="453" />'
        "</testsuite></testsuites>"
    )
    result = sim.HarnessEvaluator._parse_junit_xml(xml, [node_id], [])
    assert result.fraction_passing == 1.0
    assert result.passed is True


def test_harness_junit_ids_skipped_tests_excluded():
    """A skipped FTP test is neither a pass nor a fail -- must not silently
    count toward fraction_passing as if it ran."""
    ftp = ["tests/test_foo.py::test_a"]
    xml = _junit_xml_blob(skipped=ftp)
    result = sim.HarnessEvaluator._parse_junit_xml(xml, ftp, [])
    assert result.fraction_passing == 0.0
    passed_ids, failed_ids = sim.HarnessEvaluator._junit_test_ids(xml)
    assert not passed_ids
    assert not failed_ids


# ---------------------------------------------------------------------------
# Fuzzy parametrize-ID matching (stale FAIL_TO_PASS/PASS_TO_PASS node IDs)
# ---------------------------------------------------------------------------


def test_closest_test_id_exact_match_passthrough():
    collected = {"tests/test_foo.py::test_a"}
    assert (
        sim.HarnessEvaluator._closest_test_id("tests/test_foo.py::test_a", collected)
        == "tests/test_foo.py::test_a"
    )


def test_closest_test_id_parametrize_drift_matches():
    """A parametrize repr changed (e.g. '1.0' -> '1') but the prefix before the
    bracket is unique -- must resolve to the currently-collected id. This is the
    negative test for the fix itself: reverting _closest_test_id to always
    return ``requested`` unchanged makes this fail."""
    requested = "tests/test_foo.py::test_x[1.0]"
    collected = {"tests/test_foo.py::test_x[1]"}
    assert (
        sim.HarnessEvaluator._closest_test_id(requested, collected)
        == "tests/test_foo.py::test_x[1]"
    )


def test_closest_test_id_no_confident_match_falls_back_to_literal():
    """No collected id is even remotely similar -- must return the literal
    requested id unchanged (never worse than the un-corrected behavior)."""
    requested = "tests/test_foo.py::test_totally_unrelated"
    collected = {"tests/other.py::test_x"}
    assert sim.HarnessEvaluator._closest_test_id(requested, collected) == requested


def test_closest_test_id_no_prefix_match_never_fuzzes_against_whole_collection():
    """Zero candidates share the pre-'[' prefix -- must fall back to the literal
    id, never fuzzy-match against the entire collected set. Confirmed live on
    pylint-7080: a truncated/malformed recorded PASS_TO_PASS string (a
    SWE-bench dataset artifact, not parametrize drift) has no real match, and
    whole-collection difflib similarity picked an unrelated real test in the
    same large test class (test_stdin[...] -> test_stdin_missing_modulename),
    turning a harmless no-op into a false PASS_TO_PASS regression -- worse than
    the uncorrected behavior. Negative test: restoring the whole-collection
    fallback makes this fail."""
    requested = "tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]"
    collected = {
        "tests/test_self.py::TestRunTC::test_stdin_missing_modulename",  # no '[' prefix match
        "tests/test_self.py::TestRunTC::test_all",
    }
    assert sim.HarnessEvaluator._closest_test_id(requested, collected) == requested


def test_closest_test_id_multiple_prefix_matches_falls_back_to_literal():
    """Multiple candidates share the pre-'[' prefix -- ambiguous, so this must
    fall back to the literal id rather than guess via similarity. Confirmed
    live on pylint-7080: with exactly 2 same-prefix candidates, similarity-
    based disambiguation still picked the wrong one (see _closest_test_id's
    docstring) -- there is no candidate-pool size where that guess is safe."""
    requested = "tests/test_foo.py::test_x[1.0]"
    collected = {
        "tests/test_foo.py::test_x[1]",
        "tests/test_foo.py::test_x[2]",
    }
    assert sim.HarnessEvaluator._closest_test_id(requested, collected) == requested


def test_resolve_test_ids_substitutes_and_caches(monkeypatch):
    """_resolve_test_ids must call the backend exactly once per instance_id
    (memoized across a task's T rounds) and substitute drifted parametrize IDs."""
    ev = sim.HarnessEvaluator()
    calls: list[str] = []

    def fake_run_backend(self, _instance_id, _image, _test_patch, _candidate, _test_ids, script):
        calls.append(script)
        return "tests/test_foo.py::test_x[1]\n\n1 test collected in 0.01s\n"

    monkeypatch.setattr(sim.HarnessEvaluator, "_run_backend", fake_run_backend)
    ftp = ["tests/test_foo.py::test_x[1.0]"]

    resolved_ftp1, _, _ = ev._resolve_test_ids("img", "inst-1", "diff", ftp, [])
    resolved_ftp2, _, _ = ev._resolve_test_ids("img", "inst-1", "diff", ftp, [])

    assert resolved_ftp1 == ["tests/test_foo.py::test_x[1]"]
    assert resolved_ftp2 == resolved_ftp1
    assert len(calls) == 1, (
        "second call must hit the per-instance_id cache, not re-run collect-only"
    )
    assert "--collect-only" in calls[0]


def test_resolve_test_ids_runnable_ids_excludes_unresolvable_entries(monkeypatch):
    """An FTP/PTP id with no confident resolution must be excluded from
    runnable_ids (never fed to pytest) -- a single such id in the xargs'd list
    fails collection for the *entire* batch, not just itself (confirmed live
    on pylint-7080's malformed PASS_TO_PASS entries), which would zero out
    every other test in the same run including FTP. It must still appear
    unchanged in the returned (resolved_ftp, resolved_ptp) used for scoring,
    so it correctly counts as not-passing rather than being silently excused."""
    ev = sim.HarnessEvaluator()
    monkeypatch.setattr(
        sim.HarnessEvaluator,
        "_run_backend",
        lambda self, *a, **k: "tests/test_foo.py::test_a\n\n1 test collected in 0.01s\n",
    )
    ftp = ["tests/test_foo.py::test_a"]
    ptp = ["tests/test_foo.py::test_truncated[unresolvable"]

    resolved_ftp, resolved_ptp, runnable_ids = ev._resolve_test_ids(
        "img", "inst-3", "diff", ftp, ptp
    )

    assert resolved_ftp == ftp
    assert resolved_ptp == ptp, "unresolvable entry stays literal for scoring purposes"
    assert "tests/test_foo.py::test_a" in runnable_ids
    assert "tests/test_foo.py::test_truncated[unresolvable" not in runnable_ids


def test_resolve_test_ids_runnable_ids_includes_exact_and_substituted(monkeypatch):
    """Both an already-exact id and a confidently-substituted id must be
    included in runnable_ids -- only genuinely unresolvable ids are dropped."""
    ev = sim.HarnessEvaluator()
    monkeypatch.setattr(
        sim.HarnessEvaluator,
        "_run_backend",
        lambda self, *a, **k: (
            "tests/test_foo.py::test_a\ntests/test_foo.py::test_x[1]\n\n2 tests collected\n"
        ),
    )
    ftp = ["tests/test_foo.py::test_a", "tests/test_foo.py::test_x[1.0]"]

    _, _, runnable_ids = ev._resolve_test_ids("img", "inst-4", "diff", ftp, [])

    assert runnable_ids == frozenset({"tests/test_foo.py::test_a", "tests/test_foo.py::test_x[1]"})


def test_resolve_test_ids_scopes_collect_only_to_referenced_files(monkeypatch):
    ev = sim.HarnessEvaluator()
    captured_scripts: list[str] = []

    def fake_run_backend(self, _instance_id, _image, _test_patch, _candidate, _test_ids, script):
        captured_scripts.append(script)
        return ""

    monkeypatch.setattr(sim.HarnessEvaluator, "_run_backend", fake_run_backend)
    ftp = ["tests/test_foo.py::test_a"]
    ptp = ["tests/test_bar.py::test_b"]
    ev._resolve_test_ids("img", "inst-2", "diff", ftp, ptp)

    script = captured_scripts[0]
    assert "tests/test_foo.py" in script
    assert "tests/test_bar.py" in script
    assert "--collect-only" in script


def test_evaluate_uses_resolved_test_ids(monkeypatch):
    """evaluate() must grade against the IDs _resolve_test_ids returns, not the
    raw (possibly stale) FAIL_TO_PASS/PASS_TO_PASS strings from the instance --
    proves the substitution is actually wired into the scoring path, not just
    computed and discarded."""
    ev = sim.HarnessEvaluator()
    monkeypatch.setattr(
        sim.HarnessEvaluator,
        "_resolve_test_ids",
        lambda self, image, instance_id, test_patch, ftp, ptp: (
            ["tests/foo.py::resolved"],
            ptp,
            frozenset({"tests/foo.py::resolved", *ptp}),
        ),
    )

    def fake_run(cmd, **_kwargs):
        class R:
            stdout = _junit_xml_blob(passed=["tests/foo.py::resolved"])
            stderr = ""
            returncode = 0

        return R()

    monkeypatch.setattr(sim.subprocess, "run", fake_run)
    instance = {
        "instance_id": "psf__requests-1963",
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',  # the stale literal id
        "PASS_TO_PASS": "[]",
        "test_patch": "",
    }
    result = ev.evaluate("diff --git a/f.py b/f.py\n", instance)

    assert result.passed is True, "must score against the resolved id, not the stale literal"


def test_harness_script_terminates_options_before_xargs_test_ids(monkeypatch):
    """The pytest invocation must end its own fixed options with `--` before
    xargs appends the test-id positional arguments -- guards against a test id
    that happens to look like a CLI flag being misparsed as one. (Historically
    also required because an `-o junit_family=legacy` override, since removed,
    was declared nargs='*' on pytest 3.3.1 and greedily swallowed the appended
    test ids as more override values without `--`; confirmed live against the
    astropy-6938 pinned image.) Negative test: removing `--` from the script
    makes this fail."""
    ev = sim.HarnessEvaluator()
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured_cmds.append(list(cmd))

        class R:
            stdout = _junit_xml_blob(passed=["tests/foo.py::test_a"])
            stderr = ""
            returncode = 0

        return R()

    monkeypatch.setattr(sim.subprocess, "run", fake_run)
    instance = {
        "instance_id": "psf__requests-1963",
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "",
    }
    ev.evaluate("diff --git a/f.py b/f.py\n", instance)

    docker_cmd = next(
        c for c in captured_cmds if "docker" in c[0] and "run" in c and "--junit-xml" in c[-1]
    )
    script = docker_cmd[-1]
    assert "--junit-xml=/patches/report.xml -- " in script, (
        "-- must directly follow the pytest invocation's fixed options, before xargs's "
        "appended test ids"
    )


def test_harness_evaluate_pytest_command_no_quiet_flag(monkeypatch):
    """Ensure the pytest command uses -v without -q so per-test PASSED/FAILED lines
    appear, and scores via JUnit-XML rather than the old --no-header text flag
    (rejected by some pinned older pytest images -- see _junit_test_ids)."""
    ev = sim.HarnessEvaluator()
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured_cmds.append(list(cmd))

        class R:
            stdout = _junit_xml_blob(passed=["tests/foo.py::test_a"])
            stderr = ""
            returncode = 0

        return R()

    monkeypatch.setattr(sim.subprocess, "run", fake_run)
    instance = {
        "instance_id": "psf__requests-1963",
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "",
    }
    result = ev.evaluate("diff --git a/f.py b/f.py\n", instance)
    assert result.passed is True

    # Two docker invocations happen per evaluate() call now: a one-time
    # _resolve_test_ids collect-only probe, then the real eval run -- select
    # the real run specifically (it's the only one with --junit-xml).
    docker_cmd = next(
        c for c in captured_cmds if "docker" in c[0] and "run" in c and "--junit-xml" in c[-1]
    )
    script = docker_cmd[-1]
    assert "-q" not in script.split(), "pytest must not use -q — it suppresses per-test output"
    assert "-v" in script.split()
    assert "--no-header" not in script, "--no-header is rejected by some pinned older pytest"
    assert "--junit-xml=/patches/report.xml" in script


def test_harness_script_omits_allow_empty(monkeypatch):
    """git apply --allow-empty is invalid on the image's git 2.34.1 and silently
    swallows both the test_patch and candidate — the all-zeros bug. The script must
    use plain git apply."""
    ev = sim.HarnessEvaluator()
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        captured_cmds.append(list(cmd))

        class R:
            stdout = _junit_xml_blob(passed=["tests/foo.py::test_a"])
            stderr = ""
            returncode = 0

        return R()

    monkeypatch.setattr(sim.subprocess, "run", fake_run)
    instance = {
        "instance_id": "psf__requests-1963",
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "diff --git a/t.py b/t.py\n",
    }
    ev.evaluate("diff --git a/f.py b/f.py\n", instance)

    # Two docker invocations happen per evaluate() call now: a one-time
    # _resolve_test_ids collect-only probe, then the real eval run -- select
    # the real run specifically (it's the only one with --junit-xml).
    docker_cmd = next(
        c for c in captured_cmds if "docker" in c[0] and "run" in c and "--junit-xml" in c[-1]
    )
    script = docker_cmd[-1]
    assert "--allow-empty" not in script, "git apply --allow-empty breaks on git 2.34.1"
    # Candidate must be guarded so an empty patch is skipped, not fed to git apply.
    assert "-s /patches/candidate.diff" in script, "candidate apply must be guarded by [ -s ]"


def test_harness_raises_when_test_patch_fails_to_apply(monkeypatch):
    """A test_patch that cannot apply is an infrastructure failure — it must raise
    (so the task is loudly SKIPPED) rather than silently score 0."""
    ev = sim.HarnessEvaluator()

    def fake_run(cmd, **_kwargs):
        class R:
            stdout = "__HARNESS_TEST_PATCH_FAILED__\n"
            stderr = ""
            returncode = 3

        return R()

    monkeypatch.setattr(sim.subprocess, "run", fake_run)
    instance = {
        "instance_id": "psf__requests-1963",
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "diff --git a/t.py b/t.py\n",
    }
    with pytest.raises(RuntimeError, match="test_patch"):
        ev.evaluate("diff --git a/f.py b/f.py\n", instance)


def test_harness_teardown_noop_when_auto_remove_off(monkeypatch):
    calls: list[str] = []
    monkeypatch.delenv("LLM_HARNESS_AUTO_REMOVE", raising=False)
    ev = sim.HarnessEvaluator()
    monkeypatch.setattr(sim.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    ev.teardown("pallets__flask-4045")
    assert calls == [], "teardown should not call docker rmi when auto_remove is off"


def test_harness_teardown_calls_rmi_when_enabled(monkeypatch):
    monkeypatch.setenv("LLM_HARNESS_AUTO_REMOVE", "1")
    ev = sim.HarnessEvaluator()
    captured: list[list[str]] = []
    monkeypatch.setattr(
        sim.subprocess,
        "run",
        lambda cmd, **_kwargs: captured.append(list(cmd)),
    )
    ev.teardown("pallets__flask-4045")
    assert any("rmi" in cmd for cmd in captured[0]), "teardown should call docker rmi"
    assert any("pallets_1776_flask-4045" in c for cmd in captured for c in cmd)


# ---------------------------------------------------------------------------
# Modal eval backend (LLM_SIM_HARNESS_BACKEND)
# ---------------------------------------------------------------------------


def test_harness_backend_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LLM_SIM_HARNESS_BACKEND", raising=False)
    ev = sim.HarnessEvaluator()
    assert ev.backend == "local"


def test_harness_backend_env_override(monkeypatch):
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    ev = sim.HarnessEvaluator()
    assert ev.backend == "modal"


def test_harness_modal_timeout_default(monkeypatch):
    monkeypatch.delenv("LLM_HARNESS_MODAL_TIMEOUT", raising=False)
    ev = sim.HarnessEvaluator()
    assert ev.modal_timeout == 120


def test_harness_modal_timeout_env_override(monkeypatch):
    monkeypatch.setenv("LLM_HARNESS_MODAL_TIMEOUT", "45")
    ev = sim.HarnessEvaluator()
    assert ev.modal_timeout == 45


def _harness_instance(instance_id: str = "psf__requests-1963") -> dict:
    return {
        "instance_id": instance_id,
        "FAIL_TO_PASS": '["tests/foo.py::test_a"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "diff --git a/t.py b/t.py\n",
    }


def test_harness_modal_backend_skips_docker(monkeypatch):
    """Negative test: with backend=modal, evaluate() must never touch
    sim.subprocess.run — proves the local docker path is skipped, not just added
    alongside it."""
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    ev = sim.HarnessEvaluator()
    monkeypatch.setattr(
        sim.HarnessEvaluator,
        "_run_modal",
        lambda *a, **k: _junit_xml_blob(passed=["tests/foo.py::test_a"]),
    )

    def fail_run(*_a, **_k):
        pytest.fail("modal backend must not call subprocess.run (the docker path)")

    monkeypatch.setattr(sim.subprocess, "run", fail_run)
    result = ev.evaluate("diff --git a/f.py b/f.py\n", _harness_instance())
    assert result.passed is True


class _FakeStdinWriter:
    def __init__(self) -> None:
        self.written = ""
        self.eof = False

    def write(self, data: str) -> None:
        self.written += data

    def write_eof(self) -> None:
        self.eof = True

    def drain(self) -> None:
        pass


class _FakeExecProc:
    def __init__(self, stdout: str = _junit_xml_blob(passed=["tests/foo.py::test_a"])) -> None:
        self._stdout = stdout
        self.returncode = 0
        self.stdin = _FakeStdinWriter()

    def wait(self) -> int:
        return self.returncode

    @property
    def stdout(self):
        class _Stream:
            def __init__(self, text: str) -> None:
                self._text = text

            def read(self) -> str:
                return self._text

        return _Stream(self._stdout)

    @property
    def stderr(self):
        class _Stream:
            def read(self) -> str:
                return ""

        return _Stream()


class _FakeSandbox:
    instances: list[_FakeSandbox] = []

    def __init__(self) -> None:
        self.execs: list[tuple[tuple, dict]] = []
        self.terminated = False
        _FakeSandbox.instances.append(self)

    def exec(self, *args, **kwargs):
        self.execs.append((args, kwargs))
        return _FakeExecProc()

    def terminate(self) -> None:
        self.terminated = True


class _FakeImage:
    @staticmethod
    def from_registry(*_a, **_k):
        return object()


class _FakeApp:
    @staticmethod
    def lookup(*_a, **_k):
        return object()


class _FakeModal:
    App = _FakeApp
    Image = _FakeImage

    class Sandbox:
        @staticmethod
        def create(*_a, **_k):
            return _FakeSandbox()


@pytest.fixture
def fake_modal(monkeypatch):
    """Inject a fake ``modal`` module into sys.modules so the lazy ``import modal``
    inside HarnessEvaluator resolves without the real package -- the standard
    technique for testing an optional, lazily-imported dependency."""
    import sys

    _FakeSandbox.instances = []
    fake = _FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    return fake


@pytest.mark.usefixtures("fake_modal")
def test_harness_modal_reuses_sandbox_across_rounds(monkeypatch):
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    monkeypatch.setenv("LLM_HARNESS_MODAL_TIMEOUT", "45")
    ev = sim.HarnessEvaluator()
    instance = _harness_instance()

    ev.evaluate("diff round0\n", instance)
    ev.evaluate("diff round1\n", instance)

    assert len(_FakeSandbox.instances) == 1, "Sandbox.create must run once per task, not per round"
    sb = _FakeSandbox.instances[0]
    reset_execs = [e for e in sb.execs if "git checkout -- ." in " ".join(e[0])]
    # 2 real rounds + 1 one-time _resolve_test_ids collect-only probe (cached
    # after round 0, so it never fires again on round 1).
    assert len(reset_execs) == 3, "each round must reset the reused container's working tree"


@pytest.mark.usefixtures("fake_modal")
def test_harness_modal_exec_passes_timeout(monkeypatch):
    """Negative test: the eval exec MUST carry the per-episode timeout kwarg -- fails
    if the gvisor-stall guard is dropped."""
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    monkeypatch.setenv("LLM_HARNESS_MODAL_TIMEOUT", "45")
    ev = sim.HarnessEvaluator()
    ev.evaluate("diff --git a/f.py b/f.py\n", _harness_instance())

    sb = _FakeSandbox.instances[0]
    eval_execs = [e for e in sb.execs if "pytest" in " ".join(e[0])]
    assert eval_execs, "expected a pytest exec call"
    assert eval_execs[0][1].get("timeout") == 45


@pytest.mark.usefixtures("fake_modal")
def test_harness_modal_teardown_terminates_sandbox(monkeypatch):
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    ev = sim.HarnessEvaluator()
    instance = _harness_instance()
    ev.evaluate("diff --git a/f.py b/f.py\n", instance)

    sb = _FakeSandbox.instances[0]
    ev.teardown(instance["instance_id"])
    assert sb.terminated is True
    assert instance["instance_id"] not in ev._modal_sandboxes


def test_harness_modal_app_lock_single_lookup(monkeypatch):
    """Concurrent threads with distinct instance_ids racing on the FIRST-EVER
    sandbox creation must call ``App.lookup`` exactly once -- the lock guards
    only that singleton init, not per-instance sandbox creation."""
    import sys

    lookup_calls: list[int] = []

    class _CountingApp:
        @staticmethod
        def lookup(*_a, **_k):
            lookup_calls.append(1)
            time.sleep(0.02)  # widen the race window
            return object()

    class _CountingModal:
        App = _CountingApp
        Image = _FakeImage

        class Sandbox:
            @staticmethod
            def create(*_a, **_k):
                return _FakeSandbox()

    monkeypatch.setitem(sys.modules, "modal", _CountingModal())
    monkeypatch.setenv("LLM_SIM_HARNESS_BACKEND", "modal")
    _FakeSandbox.instances = []

    ev = sim.HarnessEvaluator()
    barrier = threading.Barrier(4)

    def worker(idx: int) -> None:
        barrier.wait()
        ev._get_or_create_modal_sandbox(f"instance-{idx}", "fake-image")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert lookup_calls == [1]  # App.lookup called exactly once despite 4 racing threads
    assert len(_FakeSandbox.instances) == 4  # each distinct instance_id still got its own sandbox


def test_harness_build_feedback_lists_failing_tests():
    ftp = ["tests/test_foo.py::test_a", "tests/test_foo.py::test_b"]
    output = "tests/test_foo.py::test_a PASSED\ntests/test_foo.py::test_b FAILED\n"
    passed_ids = {sim.HarnessEvaluator._node_id_signature("tests/test_foo.py::test_a")}
    feedback = sim.HarnessEvaluator._build_harness_feedback(output, ftp, passed_ids)
    assert "1/2" in feedback
    assert "tests/test_foo.py::test_b" in feedback
    assert "Revise" in feedback


def test_harness_build_feedback_includes_failure_hints():
    ftp = ["tests/test_foo.py::test_b"]
    output = (
        "tests/test_foo.py::test_b FAILED\n"
        "FAILED tests/test_foo.py::test_b - AssertionError: got 0\n"
    )
    feedback = sim.HarnessEvaluator._build_harness_feedback(output, ftp, set())
    assert "AssertionError: got 0" in feedback


def test_harness_build_feedback_no_revise_when_all_pass() -> None:
    """_build_harness_feedback must omit the 'Revise' prompt when all required tests pass."""
    ftp = ["tests/test_foo.py::test_a", "tests/test_foo.py::test_b"]
    output = "tests/test_foo.py::test_a PASSED\ntests/test_foo.py::test_b PASSED\n"
    passed_ids = {sim.HarnessEvaluator._node_id_signature(t) for t in ftp}
    feedback = sim.HarnessEvaluator._build_harness_feedback(output, ftp, passed_ids)
    assert "2/2" in feedback
    assert "Revise" not in feedback


def test_harness_make_critique_uses_feedback():
    ev = sim.HarnessEvaluator()
    evaluation = sim.PatchEvaluation(quality=0.0, passed=False, feedback="Use the feedback.")
    result = ev.make_critique("diff ...", evaluation, {})
    assert result == "Use the feedback."


def test_harness_make_critique_falls_back_when_no_feedback():
    ev = sim.HarnessEvaluator()
    evaluation = sim.PatchEvaluation(quality=0.0, passed=False, feedback="")
    patch = "diff --git a/f.py b/f.py\n@@\n+x\n"
    result = ev.make_critique(patch, evaluation, {})
    assert "patch" in result.lower() or "diff" in result.lower()


def test_proxy_evaluator_make_critique_uses_patch_fallback():
    ev = sim.ProxyEditDistanceEvaluator(tau=0.6)
    evaluation = sim.PatchEvaluation(quality=0.5, passed=False, feedback="should be ignored")
    patch = "diff --git a/f.py b/f.py\n@@\n+x\n"
    result = ev.make_critique(patch, evaluation, {})
    # Base class ignores feedback — returns patch-structure critique
    assert "should be ignored" not in result


def test_get_evaluator_dispatch():
    assert isinstance(sim.get_evaluator(_cfg(evaluator="proxy")), sim.ProxyEditDistanceEvaluator)
    assert isinstance(sim.get_evaluator(_cfg(evaluator="harness")), sim.HarnessEvaluator)
    assert isinstance(sim.get_evaluator(_cfg(evaluator="judge")), sim.LLMJudgeEvaluator)


def test_get_evaluator_unknown_raises():
    with pytest.raises(ValueError):
        sim.get_evaluator(_cfg(evaluator="nope"))


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def test_leakage_guard_passes_clean_prompt():
    inst = make_instance()
    # Describes the problem only; contains none of the gold diff's content lines.
    sim.assert_no_leakage("Please fix the addition helper so the reported issue is resolved.", inst)


def test_leakage_guard_noop_when_no_gold():
    inst = make_instance(gold="")
    sim.assert_no_leakage("anything at all, even return a + b", inst)  # no gold => no raise


def test_leakage_guard_detects_gold_diff_body():
    inst = make_instance()
    leaky = "Here is the fix:\n" + GOLD
    with pytest.raises(sim.LeakageError):
        sim.assert_no_leakage(leaky, inst)


def test_leakage_guard_detects_gold_added_line():
    inst = make_instance()
    # A single distinctive added line from the gold diff must not appear.
    with pytest.raises(sim.LeakageError):
        sim.assert_no_leakage("hint: return a + b", inst)


def test_leakage_guard_ignores_gold_lines_already_public_in_problem_statement():
    # SWE-bench problem statements routinely quote code (repros, imports, tracebacks)
    # that coincides with lines the gold patch adds. That overlap is PUBLIC task
    # input the agent receives verbatim -- not a harness-introduced leak -- and must
    # not trip the guard (this is the astropy-14182 false positive).
    inst = {
        "instance_id": "pub__overlap-1",
        "problem_statement": "Repro:\n>>> from astropy.table import QTable\nfails with TypeError.",
        "patch": (
            "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n"
            "+>>> from astropy.table import QTable\n+guard_value = True\n"
        ),
    }
    # Prompt is just the (public) problem statement: the overlapping line is allowed.
    sim.assert_no_leakage(inst["problem_statement"], inst)


def test_leakage_guard_still_flags_gold_only_line_not_in_problem_statement():
    # Negative test: a distinctive gold line that is NOT public must still be caught
    # if harness-added text (a critique) surfaces it. Fails if the fix over-broadly
    # disables the guard.
    inst = {
        "instance_id": "sec__leak-1",
        "problem_statement": "Something is broken.",
        "patch": (
            "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n"
            "+guard_value = compute_secret_fix()\n"
        ),
    }
    with pytest.raises(sim.LeakageError):
        sim.assert_no_leakage(
            "Something is broken.\nhint: guard_value = compute_secret_fix()", inst
        )


def test_harness_injected_context_is_gold_free_even_with_gold_like_prior():
    # The reflexion loop feeds the agent's OWN prior patch back; that patch may
    # legitimately resemble gold as it converges. Only the harness-injected context
    # (problem statement + critique) is leakage-checked -- and it stays gold-free.
    inst = make_instance()
    ctx = sim.harness_injected_context(inst, critique="refine the edit")
    sim.assert_no_leakage(ctx, inst)  # no raise, even though gold == GOLD


def test_build_agent_prompt_includes_prior_patch_only_after_round_0():
    inst = make_instance()
    p0 = sim.build_agent_prompt(inst, round_idx=0, prior_patch="PRIOR", critique="CRIT")
    assert "PRIOR" not in p0
    p1 = sim.build_agent_prompt(inst, round_idx=1, prior_patch="PRIOR", critique="CRIT")
    assert "PRIOR" in p1 and "CRIT" in p1


def test_build_agent_prompt_omits_critique_section_when_critique_empty():
    # ``blind`` feedback mode feeds a prior patch but no critique. The prompt must
    # not emit a "Critique of the previous attempt" section (which would be
    # nonsensical when empty) or an instruction that references a critique that
    # doesn't exist -- it should fall back to a mode-neutral revise instruction.
    inst = make_instance()
    p = sim.build_agent_prompt(inst, round_idx=1, prior_patch="PRIOR", critique="")
    assert "PRIOR" in p
    assert "Critique of the previous attempt" not in p
    assert "address the critique" not in p
    assert "Revise your patch to improve it." in p


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_prompt_cache_key_is_deterministic_and_sensitive():
    base = sim.prompt_cache_key("m", "id", 0, "p")
    assert base == sim.prompt_cache_key("m", "id", 0, "p")  # deterministic
    assert len(base) == 64  # sha256 hex
    # Sensitive to EVERY component of the key (model, instance, round, prompt).
    assert base != sim.prompt_cache_key("m2", "id", 0, "p")
    assert base != sim.prompt_cache_key("m", "id2", 0, "p")
    assert base != sim.prompt_cache_key("m", "id", 1, "p")
    assert base != sim.prompt_cache_key("m", "id", 0, "p2")


def test_make_critique_is_independent_of_evaluator_output():
    # The critique must key ONLY on the agent's own patch, never on evaluator output
    # (surfacing the pass/fail bit would leak a thresholded gold-similarity signal).
    empty = sim._make_critique("")
    nondiff = sim._make_critique("just some prose, not a patch")
    valid = sim._make_critique("diff --git a/x b/x\n@@\n+ok\n")
    assert empty != valid and nondiff != valid
    assert "no usable diff" in empty
    # It takes exactly one argument (the patch): no PatchEvaluation is accepted.
    import inspect

    assert list(inspect.signature(sim._make_critique).parameters) == ["patch"]


def test_cache_roundtrip(tmp_path: Path):
    cache = sim.ResponseCache(tmp_path)
    key = "abc123"
    assert cache.get(key) is None
    cache.put(key, "the-response")
    assert cache.get(key) == "the-response"
    # A fresh cache object over the same dir still sees it (persisted to disk).
    assert sim.ResponseCache(tmp_path).get(key) == "the-response"


def test_response_cache_put_is_atomic_and_get_tolerates_corruption(tmp_path: Path):
    cache = sim.ResponseCache(tmp_path)
    key = "corrupt-key"
    path = tmp_path / f"{key}.json"
    path.write_text("{not valid json")  # simulates a process killed mid-write

    assert cache.get(key) is None  # a torn file reads as a miss...
    assert not path.exists()  # ...and is unlinked, not a permanent poison pill

    cache.put(key, "clean-response")
    assert cache.get(key) == "clean-response"
    assert list(tmp_path.glob("*.tmp-*")) == []  # no tmp artifacts survive a put


# ---------------------------------------------------------------------------
# Eval cache (makes Modal/harness evaluation cost resumable)
# ---------------------------------------------------------------------------


def test_eval_cache_key_varies_on_patch_evaluator_seed_round():
    base = sim.eval_cache_key("inst-1", 0, 0, "patch-A", "proxy::0.6")
    assert base == sim.eval_cache_key("inst-1", 0, 0, "patch-A", "proxy::0.6")
    assert base != sim.eval_cache_key("inst-1", 0, 0, "patch-B", "proxy::0.6")  # patch
    assert base != sim.eval_cache_key("inst-1", 0, 0, "patch-A", "harness:modal:0.6")  # evaluator
    assert base != sim.eval_cache_key("inst-1", 0, 1, "patch-A", "proxy::0.6")  # seed
    assert base != sim.eval_cache_key("inst-1", 1, 0, "patch-A", "proxy::0.6")  # round
    assert base != sim.eval_cache_key("inst-2", 0, 0, "patch-A", "proxy::0.6")  # instance


def test_evaluator_signature_version_marker_present():
    """_evaluator_signature must carry a version marker so bumping it invalidates
    every eval cached under the pre-fix HarnessEvaluator (see eval_cache_key) --
    negative test: reverting the marker makes this fail."""
    cfg = _cfg(evaluator="harness")
    ev = sim.HarnessEvaluator()
    assert sim._evaluator_signature(cfg, ev).endswith(":v2")


def test_evaluator_signature_bump_changes_cache_key():
    """Same (instance, round, seed, patch) but a different evaluator_signature
    version marker must produce a different eval_cache_key -- proves a version
    bump actually invalidates old cache entries rather than being cosmetic."""
    key_v1 = sim.eval_cache_key("inst-1", 0, 0, "patch-A", "harness::0.6")
    key_v2 = sim.eval_cache_key("inst-1", 0, 0, "patch-A", "harness::0.6:v2")
    assert key_v1 != key_v2


def test_run_task_uses_eval_cache_hit_skips_evaluate(tmp_path: Path):
    cfg = _cfg(t=3, agent="stub", evaluator="proxy")
    eval_cache = sim.EvalCache(tmp_path)
    instance = make_instance()

    class _CountingEvaluator(sim.PatchEvaluator):
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            self.calls += 1
            return sim.PatchEvaluation(quality=0.3, passed=False, fraction_passing=0.3)

    ev1 = _CountingEvaluator()
    records1 = sim.run_task(instance, cfg, sim.StubAgentRunner(), ev1, eval_cache)
    assert ev1.calls == cfg.t  # first run: every round is a genuine miss

    ev2 = _CountingEvaluator()
    records2 = sim.run_task(instance, cfg, sim.StubAgentRunner(), ev2, eval_cache)
    assert ev2.calls == 0  # second run: fully served from cache, no Modal/harness call
    assert records1 == records2


def test_run_task_eval_cache_miss_populates_then_hits(tmp_path: Path):
    cfg = _cfg(t=2, agent="stub", evaluator="harness")
    eval_cache = sim.EvalCache(tmp_path)
    instance = make_instance()

    class _RichEvaluator(sim.PatchEvaluator):
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            self.calls += 1
            return sim.PatchEvaluation(
                quality=0.75,
                passed=False,
                detail="harness:ftp=3/4,ptp_reg=0",
                feedback="Test harness: 3/4 required tests passed.",
                fraction_passing=0.75,
            )

    ev1 = _RichEvaluator()
    records1 = sim.run_task(instance, cfg, sim.StubAgentRunner(), ev1, eval_cache)
    assert ev1.calls == cfg.t  # miss on every round -> evaluate called once per round

    ev2 = _RichEvaluator()
    records2 = sim.run_task(instance, cfg, sim.StubAgentRunner(), ev2, eval_cache)
    assert ev2.calls == 0  # hit on every round
    assert records1 == records2

    # The cached PatchEvaluation round-trips ALL fields (quality, passed, detail,
    # feedback, fraction_passing) -- not just the subset that lands in records.
    runner = sim.StubAgentRunner()
    patch0 = runner.run(instance, round_idx=0, prior_patch="", critique="")
    key0 = sim.eval_cache_key(
        str(instance["instance_id"]), 0, 0, patch0, sim._evaluator_signature(cfg, ev1)
    )
    assert eval_cache.get(key0) == sim.PatchEvaluation(
        quality=0.75,
        passed=False,
        detail="harness:ftp=3/4,ptp_reg=0",
        feedback="Test harness: 3/4 required tests passed.",
        fraction_passing=0.75,
    )


# ---------------------------------------------------------------------------
# Stub agent runner (offline, deterministic)
# ---------------------------------------------------------------------------


def test_stub_agent_is_deterministic_and_leak_free():
    runner = sim.StubAgentRunner()
    inst = make_instance()
    p1 = runner.run(inst, round_idx=0, prior_patch="", critique="")
    p2 = runner.run(inst, round_idx=0, prior_patch="", critique="")
    assert p1 == p2
    assert isinstance(p1, str)
    # The stub must not reproduce the gold patch verbatim (no leakage by fixture).
    assert GOLD not in p1


def test_get_agent_runner_dispatch():
    assert isinstance(sim.get_agent_runner(_cfg(agent="stub")), sim.StubAgentRunner)
    assert isinstance(sim.get_agent_runner(_cfg(agent="mini")), sim.MiniSweAgentRunner)


def test_get_agent_runner_unknown_raises():
    with pytest.raises(ValueError):
        sim.get_agent_runner(_cfg(agent="nope"))


# ---------------------------------------------------------------------------
# Clone caching (MiniSweAgentRunner.prepare / _get_or_create_clone / teardown)
# ---------------------------------------------------------------------------


def _install_fake_clone_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch MiniSweAgentRunner._clone_repo to build a real, tiny local git
    repo (no network) instead of fetching from GitHub -- so the real
    ``git clone --local`` in ``_get_or_create_clone`` has a genuine template to
    clone.

    Every git subprocess (the fake's and the runner's real clone) ignores global and
    system config. A trace2-wired daemon (git-ai) otherwise writes ``refs/notes/ai``
    and ``.git/ai/`` into these throwaway repos, racing ``cleanup()``'s
    ``rmtree(ignore_errors=True)``, which then silently leaves a template behind."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def fake_clone_repo(instance: dict, workdir: str) -> Path:
        repo_dir = Path(workdir) / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        for args in (
            ["init", "--quiet"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "test"],
        ):
            subprocess.run(["git", "-C", str(repo_dir), *args], check=True)  # noqa: S603, S607
        (repo_dir / "f.txt").write_text("hello\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True)  # noqa: S603, S607
        subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "commit", "--quiet", "-m", "init"],  # noqa: S607
            check=True,
        )
        return repo_dir

    monkeypatch.setattr(sim.MiniSweAgentRunner, "_clone_repo", staticmethod(fake_clone_repo))


def test_mini_swe_agent_runner_prepare_clones_template_once_per_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _install_fake_clone_repo(monkeypatch)  # replaces _clone_repo with the fake FIRST
    fake_clone_repo = sim.MiniSweAgentRunner._clone_repo

    def counting_clone_repo(instance: dict, workdir: str) -> Path:
        calls.append(instance["instance_id"])
        return fake_clone_repo(instance, workdir)

    monkeypatch.setattr(
        sim.MiniSweAgentRunner,
        "_clone_repo",
        staticmethod(counting_clone_repo),
    )
    cfg = _cfg(agent="mini")
    runner = sim.MiniSweAgentRunner(cfg, sim.ResponseCache(tmp_path))
    # instance 0 repeated: it must clone only once total.
    instances = [make_instance(0), make_instance(1), make_instance(0)]

    runner.prepare(instances)

    assert calls == ["test__repo-0", "test__repo-1"]


def test_mini_swe_agent_runner_get_or_create_clone_uses_local_clone_not_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_clone_repo(monkeypatch)
    cfg = _cfg(agent="mini")
    runner = sim.MiniSweAgentRunner(cfg, sim.ResponseCache(tmp_path))
    instance = make_instance(0)
    runner.prepare([instance])

    clone_repo_calls: list[str] = []
    real = sim.MiniSweAgentRunner._clone_repo

    def counting_clone_repo(instance: dict, workdir: str) -> Path:
        clone_repo_calls.append(instance["instance_id"])
        return real(instance, workdir)

    monkeypatch.setattr(sim.MiniSweAgentRunner, "_clone_repo", staticmethod(counting_clone_repo))

    clone_seed0 = runner._get_or_create_clone(instance, seed=0)
    clone_seed1 = runner._get_or_create_clone(instance, seed=1)

    assert clone_repo_calls == []  # neither seed re-fetched the template (no network)
    assert clone_seed0 != clone_seed1
    assert (clone_seed0 / "f.txt").exists()
    assert (clone_seed1 / "f.txt").exists()


def test_mini_swe_agent_runner_teardown_removes_clone_and_allows_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_clone_repo(monkeypatch)
    cfg = _cfg(agent="mini")
    runner = sim.MiniSweAgentRunner(cfg, sim.ResponseCache(tmp_path))
    instance = make_instance(0)
    runner.prepare([instance])

    clone_seed0 = runner._get_or_create_clone(instance, seed=0)
    clone_seed1 = runner._get_or_create_clone(instance, seed=1)
    assert clone_seed0.exists()
    assert clone_seed1.exists()

    runner.teardown(instance["instance_id"], seed=0)

    assert not clone_seed0.exists()
    assert clone_seed1.exists()  # a different seed's clone is untouched

    recreated = runner._get_or_create_clone(instance, seed=0)
    assert recreated.exists()
    assert recreated != clone_seed0  # a fresh workdir, not a reuse of the removed one


def test_mini_swe_agent_runner_cleanup_removes_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_clone_repo(monkeypatch)
    cfg = _cfg(agent="mini")
    runner = sim.MiniSweAgentRunner(cfg, sim.ResponseCache(tmp_path))
    instances = [make_instance(0), make_instance(1)]
    runner.prepare(instances)
    templates = list(runner._repo_templates.values())
    assert templates and all(t.exists() for t in templates)

    runner.cleanup()

    assert not any(t.exists() for t in templates)
    assert runner._repo_templates == {}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_task_produces_per_round_records():
    cfg = _cfg(t=3, agent="stub", evaluator="proxy")
    runner = sim.StubAgentRunner()
    ev = sim.ProxyEditDistanceEvaluator(tau=cfg.tau)
    records = sim.run_task(make_instance(), cfg, runner, ev)
    assert len(records) == cfg.t
    for r_idx, rec in enumerate(records):
        assert rec["round"] == r_idx
        assert 0.0 <= rec["quality"] <= 1.0
        assert isinstance(rec["passed"], bool)


def test_run_task_leakage_guard_is_wired_in(monkeypatch):
    # The guard must fire through the orchestration path (not just when called in
    # isolation) when HARNESS-SIDE text surfaces distinctive gold content. The only
    # harness-generated text is the critique, so we make it leak a gold added-line
    # body; gold that is merely public in the problem statement is intentionally
    # allowed (see test_leakage_guard_ignores_gold_lines_already_public...).
    cfg = _cfg(t=2, agent="stub", evaluator="proxy")
    inst = make_instance()  # "return a + b" is NOT in this problem statement
    monkeypatch.setattr(sim, "_make_critique", lambda _patch: "the fix is: return a + b")
    runner = sim.StubAgentRunner()
    ev = sim.ProxyEditDistanceEvaluator(tau=cfg.tau)
    with pytest.raises(sim.LeakageError):
        sim.run_task(inst, cfg, runner, ev)


def test_run_task_early_stop() -> None:
    """Early-stop: evaluator not called after solve; padded rounds have quality=1.0."""
    cfg = _cfg(t=4, agent="stub", evaluator="proxy")
    instance = make_instance()
    call_count = [0]

    class _SolveAtRound1(sim.PatchEvaluator):
        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            n = call_count[0]
            call_count[0] += 1
            if n >= 1:
                return sim.PatchEvaluation(quality=1.0, passed=True, fraction_passing=1.0)
            return sim.PatchEvaluation(quality=0.3, passed=False, fraction_passing=0.3)

    records = sim.run_task(instance, cfg, sim.StubAgentRunner(), _SolveAtRound1())

    assert call_count[0] == 2  # called only for rounds 0 and 1
    assert len(records) == cfg.t  # rectangular

    real_recs = [r for r in records if not r["padded"]]
    pad_recs = [r for r in records if r["padded"]]
    assert len(real_recs) == 2
    assert len(pad_recs) == 2

    assert real_recs[0]["round"] == 0
    assert real_recs[0]["quality"] == pytest.approx(0.3)
    assert real_recs[0]["solved_round"] == 1

    assert real_recs[1]["round"] == 1
    assert real_recs[1]["quality"] == pytest.approx(1.0)
    assert real_recs[1]["solved_round"] == 1

    for pr in pad_recs:
        assert pr["quality"] == pytest.approx(1.0)
        assert pr["passed"] is True
        assert pr["padded"] is True
        assert pr["solved_round"] == 1


def test_run_task_keep_best_patch() -> None:
    """Keep-best: a regressed round must not become the base for the next attempt."""
    cfg = _cfg(t=4, agent="stub", evaluator="proxy")
    instance = make_instance()
    received_priors: list[str] = []

    class _TrackingRunner(sim.AgentRunner):
        def run(
            self,
            instance: dict,
            round_idx: int,
            prior_patch: str,
            critique: str,
            seed: int = 0,
            model: str | None = None,
        ) -> str:
            received_priors.append(prior_patch)
            return f"patch_round_{round_idx}"

    class _Regression(sim.PatchEvaluator):
        _qualities = [0.2, 0.8, 0.1, 0.5]

        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            idx = len(received_priors) - 1
            q = self._qualities[idx]
            return sim.PatchEvaluation(quality=q, passed=False, fraction_passing=q)

    records = sim.run_task(instance, cfg, _TrackingRunner(), _Regression())

    assert len(records) == cfg.t
    assert received_priors[0] == ""  # round 0: no prior
    # Round 2 and 3 must both receive the round-1 patch (best), not the regressed round-2 patch
    assert received_priors[2] == "patch_round_1"
    assert received_priors[3] == "patch_round_1"


def test_run_task_early_stop_negative() -> None:
    """Negative guard: runner called only solved_round+1 times, not T times.

    This test FAILS if the early-stop ``break`` is removed from run_task.
    """
    cfg = _cfg(t=4, agent="stub", evaluator="proxy")
    instance = make_instance()
    runner_calls = [0]

    class _CountingRunner(sim.AgentRunner):
        def run(
            self,
            instance: dict,
            round_idx: int,
            prior_patch: str,
            critique: str,
            seed: int = 0,
            model: str | None = None,
        ) -> str:
            runner_calls[0] += 1
            return "patch"

    class _SolveAtRound1(sim.PatchEvaluator):
        def __init__(self) -> None:
            self._n = 0

        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            p = self._n >= 1
            q = 1.0 if p else 0.3
            self._n += 1
            return sim.PatchEvaluation(quality=q, passed=p, fraction_passing=q)

    records = sim.run_task(instance, cfg, _CountingRunner(), _SolveAtRound1())
    solved = next(r["solved_round"] for r in records if r["solved_round"] is not None)
    assert runner_calls[0] == solved + 1
    assert runner_calls[0] < cfg.t  # guard fired — not T calls


# ---------------------------------------------------------------------------
# Feedback ablation (independent / blind / diagnostic)
# ---------------------------------------------------------------------------


class _RecordingRunner(sim.AgentRunner):
    """Records the (round_idx, prior_patch, critique, model) tuple it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str, str | None]] = []

    def run(
        self,
        instance: dict,
        round_idx: int,
        prior_patch: str,
        critique: str,
        seed: int = 0,
        model: str | None = None,
    ) -> str:
        self.calls.append((round_idx, prior_patch, critique, model))
        return f"patch_round_{round_idx}"


class _NeverPassEvaluator(sim.PatchEvaluator):
    """Never solves, so all ``T`` rounds run; ``make_critique`` returns a
    distinctive, non-empty string so tests can assert on its presence/absence."""

    def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
        return sim.PatchEvaluation(quality=0.5, passed=False, fraction_passing=0.5)

    def make_critique(self, patch: str, evaluation: sim.PatchEvaluation, instance: dict) -> str:
        return "DISTINCTIVE_CRITIQUE_TEXT"


def test_run_task_diagnostic_mode_feeds_prior_patch_and_critique():
    cfg = _cfg(t=3, agent="stub", evaluator="proxy", feedback_mode="diagnostic")
    runner = _RecordingRunner()
    records = sim.run_task(make_instance(), cfg, runner, _NeverPassEvaluator())
    assert len(records) == cfg.t
    assert runner.calls[0] == (0, "", "", cfg.model)
    for _round_idx, prior_patch, critique, _model in runner.calls[1:]:
        assert prior_patch != ""
        assert critique == "DISTINCTIVE_CRITIQUE_TEXT"


def test_run_task_blind_mode_feeds_prior_patch_without_critique():
    # Negative test: fails if the diagnostic critique leaks through when blind.
    cfg = _cfg(t=3, agent="stub", evaluator="proxy", feedback_mode="blind")
    runner = _RecordingRunner()
    records = sim.run_task(make_instance(), cfg, runner, _NeverPassEvaluator())
    assert len(records) == cfg.t
    assert runner.calls[0] == (0, "", "", cfg.model)
    for _round_idx, prior_patch, critique, _model in runner.calls[1:]:
        assert prior_patch != ""
        assert critique == ""


def test_run_task_independent_mode_feeds_neither_prior_patch_nor_critique():
    # Negative test: fails if prior_patch/critique leak through when independent.
    cfg = _cfg(t=3, agent="stub", evaluator="proxy", feedback_mode="independent")
    runner = _RecordingRunner()
    records = sim.run_task(make_instance(), cfg, runner, _NeverPassEvaluator())
    assert len(records) == cfg.t
    for _round_idx, prior_patch, critique, _model in runner.calls:
        assert prior_patch == ""
        assert critique == ""


def test_run_task_invalid_feedback_mode_raises():
    cfg = _cfg(agent="stub", evaluator="proxy", feedback_mode="nope")
    with pytest.raises(ValueError):
        sim.run_task(
            make_instance(), cfg, sim.StubAgentRunner(), sim.ProxyEditDistanceEvaluator(tau=cfg.tau)
        )


@pytest.mark.parametrize("mode", ["independent", "blind", "diagnostic"])
def test_run_task_no_leakage_across_feedback_modes(mode: str):
    # End-to-end with the real (leakage-guarded) StubAgentRunner + evaluator: no
    # LeakageError in any of the three arms.
    cfg = _cfg(t=3, agent="stub", evaluator="proxy", feedback_mode=mode)
    records = sim.run_task(
        make_instance(), cfg, sim.StubAgentRunner(), sim.ProxyEditDistanceEvaluator(tau=cfg.tau)
    )
    assert len(records) == cfg.t


# ---------------------------------------------------------------------------
# Tier assignment (capability-gap deconfound)
# ---------------------------------------------------------------------------


def test_assign_tier_model_fixed_always_returns_config_model():
    cfg = _cfg(model="model-fixed", tier_policy="fixed", tier_models=("model-a", "model-b"))
    for instance_id, seed in [("task-1", 0), ("task-2", 3), ("task-1", 7)]:
        assert sim.assign_tier_model(cfg, instance_id, seed) == "model-fixed"


def test_assign_tier_model_randomized_is_deterministic():
    cfg = _cfg(tier_policy="randomized", tier_models=("model-a", "model-b"))
    first = sim.assign_tier_model(cfg, "task-42", 3)
    for _ in range(5):
        assert sim.assign_tier_model(cfg, "task-42", 3) == first


def test_assign_tier_model_randomized_distributes_across_tier_models():
    cfg = _cfg(tier_policy="randomized", tier_models=("model-a", "model-b"))
    picks = {sim.assign_tier_model(cfg, f"task-{i}", 0) for i in range(20)}
    assert picks == {"model-a", "model-b"}


def test_assign_tier_model_randomized_empty_tier_models_falls_back_to_config_model():
    cfg = _cfg(model="model-fixed", tier_policy="randomized", tier_models=())
    assert sim.assign_tier_model(cfg, "task-1", 0) == "model-fixed"


def test_assign_tier_model_unknown_policy_raises():
    cfg = _cfg(tier_policy="nope")
    with pytest.raises(ValueError):
        sim.assign_tier_model(cfg, "task-1", 0)


def test_run_task_randomized_tier_is_consistent_across_rounds_and_matches_assignment():
    # Every round of ONE task must carry the SAME assigned model (tier is a property
    # of the task attempt, not the round), and it must match assign_tier_model's own
    # output for that (instance_id, seed) -- not config.model (tier_models excludes
    # it, so this assertion is load-bearing: it fails if run_task were hardcoded to
    # config.model instead of threading the assigned tier).
    cfg = _cfg(
        t=3,
        agent="stub",
        evaluator="proxy",
        feedback_mode="diagnostic",
        tier_policy="randomized",
        tier_models=("model-a", "model-b"),
    )
    instance = make_instance()
    seed = 5
    expected_model = sim.assign_tier_model(cfg, str(instance["instance_id"]), seed)
    assert expected_model != cfg.model  # tier_models excludes config.model

    runner = _RecordingRunner()
    records = sim.run_task(instance, cfg, runner, _NeverPassEvaluator(), seed=seed)

    assert len(records) == cfg.t
    for rec in records:
        assert rec["model"] == expected_model
    for _round_idx, _prior_patch, _critique, model in runner.calls:
        assert model == expected_model


def test_run_experiment_shapes():
    cfg = _cfg(k=3, t=4, agent="stub", evaluator="proxy")
    instances = [make_instance(i) for i in range(cfg.k)]
    q, ok, sigs, _, _, _ = sim.run_experiment(instances, cfg)
    assert q.shape == (cfg.k, cfg.t)
    assert [i["instance_id"] for i in ok] == [i["instance_id"] for i in instances]
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert isinstance(sigs, dict)


def test_run_experiment_randomized_tier_returns_model_per_task():
    cfg = _cfg(
        k=4,
        t=2,
        agent="stub",
        evaluator="proxy",
        tier_policy="randomized",
        tier_models=("model-a", "model-b"),
    )
    instances = [make_instance(i) for i in range(cfg.k)]
    _, ok, _, _, _, assigned_models = sim.run_experiment(instances, cfg)
    assert len(assigned_models) == len(ok)
    assert all(m in cfg.tier_models for m in assigned_models)


def test_run_experiment_skips_failing_tasks():
    # A single task whose agent raises must NOT abort the whole experiment: it is
    # skipped and the run continues (essential for a fragile real-repo agent).
    cfg = _cfg(k=3, t=2, agent="stub", evaluator="proxy")
    instances = [make_instance(i) for i in range(3)]
    real_run = sim.StubAgentRunner.run

    def flaky_run(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        if instance["instance_id"] == "test__repo-1":
            raise RuntimeError("simulated agent crash")
        return real_run(self, instance, round_idx, prior_patch, critique)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", flaky_run)
        q, ok, _, _, _, _ = sim.run_experiment(instances, cfg)
    assert q.shape == (2, cfg.t)  # one of three tasks skipped
    assert [i["instance_id"] for i in ok] == ["test__repo-0", "test__repo-2"]


def test_run_experiment_stops_on_budget_error():
    # Budget exhaustion must STOP the loop (not skip and continue), and set the flag.
    cfg = _cfg(k=3, t=2, agent="stub", evaluator="proxy")
    instances = [make_instance(i) for i in range(3)]
    real_run = sim.StubAgentRunner.run

    def budget_explodes(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        if instance["instance_id"] == "test__repo-1":
            raise Exception("BudgetExceeded: budget exceeded $300")
        return real_run(self, instance, round_idx, prior_patch, critique)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", budget_explodes)
        q, ok, _, budget_stopped, _, _ = sim.run_experiment(instances, cfg)
    assert budget_stopped is True
    assert q.shape[0] == 1  # only task 0 completed before the budget hit
    assert ok[0]["instance_id"] == "test__repo-0"


# ---------------------------------------------------------------------------
# Opt-in concurrency (seed-outer, task-inner)
# ---------------------------------------------------------------------------


def test_run_experiment_concurrent_overlaps_in_time():
    cfg = _cfg(k=6, t=1, agent="stub", evaluator="proxy", concurrency=3)
    instances = [make_instance(i) for i in range(cfg.k)]

    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    class _SlowEvaluator(sim.PatchEvaluator):
        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return sim.PatchEvaluation(quality=0.5, passed=False, fraction_passing=0.5)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim, "get_evaluator", lambda config: _SlowEvaluator())
        q, ok, _, _, _, _ = sim.run_experiment(instances, cfg)

    assert q.shape[0] == cfg.k
    assert len(ok) == cfg.k
    assert state["peak"] > 1  # tasks genuinely overlapped, not run one-at-a-time


def test_run_experiment_concurrent_budget_stop_bounded_overrun():
    cfg = _cfg(k=6, t=1, agent="stub", evaluator="proxy", concurrency=3)
    instances = [make_instance(i) for i in range(cfg.k)]
    real_run = sim.StubAgentRunner.run

    def budget_explodes(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        if instance["instance_id"] == "test__repo-2":
            raise Exception("BudgetExceeded: budget exceeded $300")
        return real_run(self, instance, round_idx, prior_patch, critique)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", budget_explodes)
        q, _ok, _, budget_stopped, _, _ = sim.run_experiment(instances, cfg)

    assert budget_stopped is True
    # Bounded overrun: at most `concurrency` task-seeds complete after the stop.
    assert q.shape[0] <= cfg.concurrency


def test_run_experiment_concurrent_preserves_input_order():
    cfg = _cfg(k=4, t=1, agent="stub", evaluator="proxy", concurrency=4)
    instances = [make_instance(i) for i in range(cfg.k)]
    real_run = sim.StubAgentRunner.run

    def staggered_run(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        # Task 0 is the slowest, so later-submitted tasks (1..3) finish first --
        # output order must still follow the ORIGINAL instance order.
        if instance["instance_id"] == "test__repo-0":
            time.sleep(0.08)
        return real_run(self, instance, round_idx, prior_patch, critique)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", staggered_run)
        _, ok, _, _, _, _ = sim.run_experiment(instances, cfg)

    assert [i["instance_id"] for i in ok] == [f"test__repo-{i}" for i in range(cfg.k)]


def test_run_experiment_concurrent_skip_preserves_order():
    cfg = _cfg(k=4, t=1, agent="stub", evaluator="proxy", concurrency=4)
    instances = [make_instance(i) for i in range(cfg.k)]
    real_run = sim.StubAgentRunner.run

    def staggered_with_skip(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        if instance["instance_id"] == "test__repo-1":
            raise RuntimeError("simulated agent crash")
        if instance["instance_id"] == "test__repo-0":
            time.sleep(0.08)  # finishes last despite being submitted first
        return real_run(self, instance, round_idx, prior_patch, critique)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", staggered_with_skip)
        _, ok, _, _, _, _ = sim.run_experiment(instances, cfg)

    assert [i["instance_id"] for i in ok] == ["test__repo-0", "test__repo-2", "test__repo-3"]


def test_is_budget_error_detects_budget_exceeded_error():
    try:
        import litellm

        exc = litellm.BudgetExceededError(current_cost=300.0, max_budget=300.0)
        assert sim._is_budget_error(exc) is True
    except ImportError:
        pytest.skip("litellm not installed")


def test_is_budget_error_false_for_plain_exception():
    assert sim._is_budget_error(RuntimeError("something broke")) is False


@pytest.mark.parametrize(
    "message",
    [
        "Error: out of credits",
        "quota exceeded for workspace",
        "insufficient funds for this operation",
        "Payment Required",
        "402 Client Error",
    ],
)
def test_is_infra_budget_error_matches_modal_keywords_not_test_patch_failure(message: str):
    assert sim._is_infra_budget_error(RuntimeError(message)) is True
    # The pre-existing per-instance test_patch-apply-failure RuntimeError is
    # task-specific infra, not a global outage -- it must NOT trip this matcher.
    test_patch_failure = RuntimeError(
        "harness: test_patch failed to apply for test__repo-0 "
        "(infrastructure error, not a patch quality signal)"
    )
    assert sim._is_infra_budget_error(test_patch_failure) is False


def test_is_infra_budget_error_false_for_plain_exception():
    assert sim._is_infra_budget_error(RuntimeError("something broke")) is False


def test_select_diverse_tasks_picks_one_per_repo():
    rows = [
        {"instance_id": "astropy__a-2", "repo": "astropy/astropy"},
        {"instance_id": "astropy__a-1", "repo": "astropy/astropy"},
        {"instance_id": "django__d-1", "repo": "django/django"},
        {"instance_id": "sympy__s-1", "repo": "sympy/sympy"},
    ]
    picked = sim.select_diverse_tasks(rows, k=3)
    assert [p["repo"] for p in picked] == ["astropy/astropy", "django/django", "sympy/sympy"]
    # Deterministic tie-break within a repo: lowest instance id wins.
    assert picked[0]["instance_id"] == "astropy__a-1"


def test_select_diverse_tasks_exclude_ids_drops_denylisted_tasks():
    """A denylisted instance_id must be dropped from selection entirely, and the
    round-robin fills k from the remaining pool instead -- never returning fewer
    than k just because some were excluded (as long as enough remain)."""
    rows = [
        {"instance_id": "astropy__a-1", "repo": "astropy/astropy"},
        {"instance_id": "astropy__a-2", "repo": "astropy/astropy"},
        {"instance_id": "django__d-1", "repo": "django/django"},
        {"instance_id": "sympy__s-1", "repo": "sympy/sympy"},
    ]
    picked = sim.select_diverse_tasks(rows, k=3, exclude_ids=frozenset({"astropy__a-1"}))
    picked_ids = {p["instance_id"] for p in picked}
    assert "astropy__a-1" not in picked_ids
    assert len(picked) == 3
    assert "astropy__a-2" in picked_ids  # denylisted repo's next task fills the slot


def test_select_diverse_tasks_no_exclude_ids_unchanged():
    """Negative test: an empty denylist must be a no-op -- reverting exclude_ids
    filtering to always drop something (or never apply the default) would break
    either this test or the one above."""
    rows = [
        {"instance_id": "astropy__a-1", "repo": "astropy/astropy"},
        {"instance_id": "django__d-1", "repo": "django/django"},
    ]
    assert sim.select_diverse_tasks(rows, k=2) == sim.select_diverse_tasks(
        rows, k=2, exclude_ids=frozenset()
    )


def test_select_diverse_tasks_capped():
    rows = [{"instance_id": f"r{i}__x-1", "repo": f"r{i}"} for i in range(10)]
    picked = sim.select_diverse_tasks(rows, k=4)
    assert len(picked) == 4
    assert [p["repo"] for p in picked] == ["r0", "r1", "r2", "r3"]


def test_select_diverse_tasks_round_robin_when_k_exceeds_repos() -> None:
    """When k > n_repos, round-robin picks second tasks per repo."""
    # 3 repos with 2 tasks each (ids 1 and 2 per repo)
    rows = [
        {"instance_id": "r0__x-1", "repo": "r0"},
        {"instance_id": "r0__x-2", "repo": "r0"},
        {"instance_id": "r1__x-1", "repo": "r1"},
        {"instance_id": "r1__x-2", "repo": "r1"},
        {"instance_id": "r2__x-1", "repo": "r2"},
        {"instance_id": "r2__x-2", "repo": "r2"},
    ]
    picked = sim.select_diverse_tasks(rows, k=6)
    assert len(picked) == 6
    # First 3: one per repo (lowest id)
    assert [p["instance_id"] for p in picked[:3]] == ["r0__x-1", "r1__x-1", "r2__x-1"]
    # Second 3: second task per repo in same order
    assert [p["instance_id"] for p in picked[3:]] == ["r0__x-2", "r1__x-2", "r2__x-2"]


def test_select_diverse_tasks_first_six_unchanged_after_round_robin() -> None:
    """Existing first 6 picks must be identical before and after round-robin extension."""
    # 6 repos with 3 tasks each
    rows = []
    for i in range(6):
        for j in range(1, 4):
            rows.append({"instance_id": f"r{i}__x-{j}", "repo": f"r{i}"})

    first_six = sim.select_diverse_tasks(rows, k=6)
    all_twelve = sim.select_diverse_tasks(rows, k=12)

    assert first_six == all_twelve[:6], "First 6 picks must be identical regardless of k"


def test_select_diverse_tasks_k_greater_than_all_tasks_returns_all() -> None:
    """When k exceeds total tasks, return all tasks without error."""
    rows = [
        {"instance_id": "r0__x-1", "repo": "r0"},
        {"instance_id": "r1__x-1", "repo": "r1"},
    ]
    picked = sim.select_diverse_tasks(rows, k=10)
    assert len(picked) == 2


def test_select_diverse_tasks_spread_across_repos() -> None:
    """Tasks are spread across repos in round-robin order."""
    rows = [{"instance_id": f"r{i}__x-{j}", "repo": f"r{i}"} for i in range(3) for j in range(1, 5)]
    picked = sim.select_diverse_tasks(rows, k=9)
    repos = [p["repo"] for p in picked]
    # First 3 picks span all repos; picks 4-6 also span all repos
    assert repos[:3] == ["r0", "r1", "r2"]
    assert repos[3:6] == ["r0", "r1", "r2"]
    assert repos[6:9] == ["r0", "r1", "r2"]


# ---------------------------------------------------------------------------
# Metrics (map to the theorems)
# ---------------------------------------------------------------------------


def _toy_q() -> np.ndarray:
    # 3 tasks, 4 rounds; deliberately includes a dip to test running-best.
    return np.array(
        [
            [0.2, 0.5, 0.4, 0.7],
            [0.1, 0.3, 0.6, 0.6],
            [0.5, 0.5, 0.5, 0.9],
        ]
    )


def test_metrics_running_best_is_monotone():
    m = sim.compute_metrics(_toy_q(), tau=0.6)
    rb = np.array(m["running_best"])
    assert np.all(np.diff(rb, axis=1) >= -1e-12)


def test_metrics_saturation_respects_ceiling():
    q = _toy_q()
    m = sim.compute_metrics(q, tau=0.6)
    sat = np.array(m["saturation"])
    ceiling = 1.0 - q[:, 0]
    # S_{i,t} <= 1 - q_{i,0} for every task and round (Thm. refinement bound).
    assert np.all(sat <= ceiling[:, None] + 1e-9)


def test_metrics_saturation_ceiling_holds_on_non_monotone_paths():
    # A real agent may dip between rounds. Cumulative improvement must STILL respect
    # the 1 - q0 ceiling (it is measured on the running best, not raw upward variation).
    q = np.array([[0.2, 0.9, 0.3, 0.9], [0.5, 0.1, 0.6, 0.55]])
    m = sim.compute_metrics(q, tau=0.6)
    sat = np.array(m["saturation"])
    ceiling = 1.0 - q[:, 0]
    assert np.all(sat <= ceiling[:, None] + 1e-9)
    # Final cumulative improvement equals running-best gain.
    rb = np.array(m["running_best"])
    assert np.allclose(sat[:, -1], rb[:, -1] - q[:, 0])


def test_metrics_edge_gamma_round0_is_zero():
    m = sim.compute_metrics(_toy_q(), tau=0.6)
    assert m["edge_gamma"][0] == 0.0  # round 0 has no predecessor


def test_metrics_edge_alpha_epsilon_values():
    # Two tasks, both improve at round 1 -> gamma_1 = 1 - 1/2 = 1/2, eps=0, alpha large.
    q = np.array([[0.1, 0.9], [0.2, 0.8]])
    m = sim.compute_metrics(q, tau=0.6)
    assert m["edge_gamma"][1] == pytest.approx(0.5)
    assert m["epsilon"][1] == pytest.approx(0.0, abs=1e-9)
    assert m["alpha"][1] > 5.0  # 1/2 ln((1-eps)/eps) with eps clipped -> large
    # Convergence overlay uses only positive edges: monotone non-increasing.
    cb = np.array(m["convergence_bound"])
    assert np.all(np.diff(cb) <= 1e-12)


def test_metrics_edge_and_alpha_lengths():
    q = _toy_q()
    m = sim.compute_metrics(q, tau=0.6)
    t = q.shape[1]
    assert len(m["edge_gamma"]) == t
    assert len(m["epsilon"]) == t
    assert len(m["alpha"]) == t
    assert len(m["ensemble_err"]) == t
    assert len(m["eta_bar"]) == t


def test_metrics_ensemble_error_non_increasing():
    # Running-best only rises, so the fraction below tau can only fall.
    m = sim.compute_metrics(_toy_q(), tau=0.6)
    err = np.array(m["ensemble_err"])
    assert np.all(np.diff(err) <= 1e-12)


def test_metrics_margin_matches_definition():
    q = _toy_q()
    m = sim.compute_metrics(q, tau=0.6)
    assert np.allclose(np.array(m["margin"]), 2.0 * q - 1.0)


# ---------------------------------------------------------------------------
# Plotting + full offline pipeline
# ---------------------------------------------------------------------------


def test_plots_and_dumps_written(tmp_path: Path, monkeypatch):
    figdir = tmp_path / "figures"
    datadir = tmp_path / "data"
    figdir.mkdir()
    datadir.mkdir()
    monkeypatch.setattr(sim, "FIGDIR", str(figdir))
    monkeypatch.setattr(sim, "DATADIR", str(datadir))

    q = _toy_q()
    metrics = sim.compute_metrics(q, tau=0.6)
    sim.experiment_llm_1_pass_convergence(metrics)
    sim.experiment_llm_2_quality_edge(q, metrics)
    sim.experiment_llm_3_saturation(q, metrics)
    sim.experiment_llm_4_margin_distribution(q, metrics)

    for name in (
        "llm_fig1_pass_convergence.pdf",
        "llm_fig2_quality_edge.pdf",
        "llm_fig3_saturation.pdf",
        "llm_fig4_margin_distribution.pdf",
    ):
        assert (figdir / name).exists()


def test_main_survives_zero_completed_task_seeds(tmp_path: Path, monkeypatch):
    """A budget-stop before any task-seed completes yields q.shape == (0, T) --
    live failure mode: compute_metrics's margin.tolist() collapses a (0, T) array to
    ``[]`` (losing the round dimension), and experiment_llm_4_margin_distribution then
    crashes recovering it via margin.shape[1] (IndexError: tuple index out of range).
    main() must skip figure generation rather than crash when nothing completed."""
    figdir = tmp_path / "figures"
    datadir = tmp_path / "data"
    figdir.mkdir()
    datadir.mkdir()
    monkeypatch.setattr(sim, "FIGDIR", str(figdir))
    monkeypatch.setattr(sim, "DATADIR", str(datadir))
    monkeypatch.setenv("LLM_SIM_AGENT", "stub")
    monkeypatch.setenv("LLM_SIM_K", "3")
    monkeypatch.setenv("LLM_SIM_T", "2")

    def budget_explodes(self, instance, round_idx, prior_patch, critique, seed=0, model=None):
        raise Exception("BudgetExceeded: budget exceeded $300")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim.StubAgentRunner, "run", budget_explodes)
        sim.main()  # must not raise

    with open(datadir / "llm_metrics.json") as f:
        metrics = json.load(f)
    assert metrics["margin"] == []
    assert list(figdir.iterdir()) == []  # nothing to plot -- no figures written


def test_main_survives_figure_generation_crash(tmp_path: Path, monkeypatch):
    """Live failure mode: the mini-swe-agent's own `pip install -e .` inside a cloned
    SWE-bench repo (e.g. a matplotlib-repo task) can shadow the shared venv's real
    matplotlib with a path that later disappears (task cleanup), so a *later* arm's
    figure generation raises ModuleNotFoundError well after the actual results/metrics
    JSON was already written. Figures are best-effort -- a crash there must not stop
    the results/metrics write from having already landed, nor propagate out of main()
    and kill the sweep runner's cp-to-tagged-filename step for a completed arm."""
    figdir = tmp_path / "figures"
    datadir = tmp_path / "data"
    figdir.mkdir()
    datadir.mkdir()
    monkeypatch.setattr(sim, "FIGDIR", str(figdir))
    monkeypatch.setattr(sim, "DATADIR", str(datadir))
    monkeypatch.setenv("LLM_SIM_AGENT", "stub")
    monkeypatch.setenv("LLM_SIM_K", "3")
    monkeypatch.setenv("LLM_SIM_T", "2")

    def broken_backend(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'matplotlib.backends.backend_agg'")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sim, "experiment_llm_1_pass_convergence", broken_backend)
        sim.main()  # must not raise

    with open(datadir / "llm_metrics.json") as f:
        metrics = json.load(f)
    assert metrics["margin"]  # real data was written despite the later figure crash
    assert list(figdir.iterdir()) == []  # figure generation aborted, nothing partial


def test_main_offline_end_to_end(tmp_path: Path, monkeypatch):
    figdir = tmp_path / "figures"
    datadir = tmp_path / "data"
    figdir.mkdir()
    datadir.mkdir()
    monkeypatch.setattr(sim, "FIGDIR", str(figdir))
    monkeypatch.setattr(sim, "DATADIR", str(datadir))
    monkeypatch.setenv("LLM_SIM_K", "3")
    monkeypatch.setenv("LLM_SIM_T", "3")
    monkeypatch.setenv("LLM_SIM_AGENT", "stub")
    monkeypatch.setenv("LLM_SIM_EVALUATOR", "proxy")
    # Offline stub path uses make_demo_instances (no network); load_swe_tasks is not
    # called. Guard against accidental network access if that ever changes.
    monkeypatch.setattr(
        sim, "load_swe_tasks", lambda k: pytest.fail("stub run must not hit the network")
    )

    sim.main()

    results = json.loads((datadir / "llm_results.json").read_text())
    metrics = json.loads((datadir / "llm_metrics.json").read_text())
    assert "quality_matrix" in results
    assert "signals_matrices" in results
    assert np.array(results["quality_matrix"]).shape == (3, 3)
    assert "running_best" in metrics
    assert "signal_agreement" in metrics
    assert (figdir / "llm_fig1_pass_convergence.pdf").exists()


def test_demo_pipeline_quality_rises():
    # The illustrative offline configuration should show a rising mean-quality
    # trajectory (net improvement) and the theory's saturation bound.
    cfg = _cfg(k=6, t=5, agent="stub", evaluator="proxy")
    instances = sim.make_demo_instances(cfg.k)
    q, _, _sigs, _, solved_rounds, _ = sim.run_experiment(instances, cfg)
    mean_traj = q.mean(axis=0)
    assert mean_traj[-1] > mean_traj[0]  # net improvement
    metrics = sim.compute_metrics(q, tau=cfg.tau)
    # Per-task cumulative improvement never exceeds the 1 - q0 ceiling.
    sat = np.array(metrics["saturation"])
    ceiling = 1.0 - q[:, 0]
    assert np.all(sat <= ceiling[:, None] + 1e-9)
    # Per-round running-best gain is non-negative (monotone running-best invariant).
    eta_bar = np.array(metrics["eta_bar"])
    assert np.all(eta_bar >= -1e-9)
    # With early-stop the stub tasks solve before T rounds; post-solve padded rounds
    # contribute zero gain — eta_bar must be 0 at the final round.
    assert all(sr is not None for sr in solved_rounds), "all demo tasks should solve"
    assert eta_bar[-1] == pytest.approx(0.0)


def test_make_demo_instances_gold_matches_stub_target():
    insts = sim.make_demo_instances(2)
    assert len(insts) == 2
    for inst in insts:
        # Gold equals the public target the stub converges toward.
        assert inst["patch"] == sim._demo_target(inst["instance_id"])


# ---------------------------------------------------------------------------
# NormalizedEditSimEvaluator
# ---------------------------------------------------------------------------


def test_norm_edit_sim_content_denoising():
    # Two diffs identical in *content* lines but with different context lines
    # must score high (≈1). Different context should NOT drive score apart.
    shared_content = "+def add(a, b):\n+    return a + b\n-def add(a, b):\n-    return a - b\n"
    diff_a = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,3 @@\n"
        " some context\n" + shared_content + " other context\n"
    )
    diff_b = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -10,4 +10,4 @@\n"
        " completely different context\n" + shared_content + " more different context\n"
    )
    ev = sim.NormalizedEditSimEvaluator(tau=0.5)
    inst_a = make_instance(gold=diff_b)
    result = ev.evaluate(diff_a, inst_a)
    assert result.quality > 0.9  # identical content → near-perfect score


def test_norm_edit_sim_scaffolding_difference_not_inflated():
    # Identical scaffolding but different *content* → low score (scaffolding stripped).
    gold = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n"
        "+    return a + b\n-    return a - b\n"
    )
    wrong = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n"
        "+    return a * b\n-    return a - b\n"
    )
    ev = sim.NormalizedEditSimEvaluator(tau=0.5)
    inst = make_instance(gold=gold)
    result = ev.evaluate(wrong, inst)
    # Different content line (a*b vs a+b): must be strictly less than the exact-match score.
    exact = ev.evaluate(gold, inst)
    assert result.quality < exact.quality


def test_norm_edit_sim_exact_match():
    ev = sim.NormalizedEditSimEvaluator(tau=0.5)
    inst = make_instance()
    result = ev.evaluate(GOLD, inst)
    assert result.quality == pytest.approx(1.0)


def test_norm_edit_sim_passed_uses_tau():
    ev = sim.NormalizedEditSimEvaluator(tau=0.9)
    inst = make_instance()
    # Empty patch → quality near 0 → should not pass
    result = ev.evaluate("", inst)
    assert not result.passed


# ---------------------------------------------------------------------------
# LocalizationEvaluator
# ---------------------------------------------------------------------------


_GOLD_LOC = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n"
    "--- a/pkg/mod.py\n"
    "+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-def add(a, b):\n"
    "+def add(a, b):  # fixed\n"
    " pass\n"
)

_WRONG_FILE = (
    "diff --git a/other/thing.py b/other/thing.py\n"
    "--- a/other/thing.py\n"
    "+++ b/other/thing.py\n"
    "@@ -5,1 +5,1 @@\n"
    "-x = 1\n"
    "+x = 2\n"
)


def test_localization_disjoint_files_zero_f1():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_WRONG_FILE, inst)
    subs = json.loads(result.detail)
    assert subs["file_f1"] == pytest.approx(0.0)


def test_localization_identical_files_perfect_f1():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_GOLD_LOC, inst)
    subs = json.loads(result.detail)
    assert subs["file_f1"] == pytest.approx(1.0)


def test_localization_hunk_jaccard_same_diff():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_GOLD_LOC, inst)
    subs = json.loads(result.detail)
    assert subs["hunk_jaccard"] == pytest.approx(1.0)


def test_localization_hunk_jaccard_no_overlap():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_WRONG_FILE, inst)
    subs = json.loads(result.detail)
    assert subs["hunk_jaccard"] == pytest.approx(0.0)


def test_localization_primary_is_file_f1():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_GOLD_LOC, inst)
    subs = json.loads(result.detail)
    assert result.quality == pytest.approx(subs["file_f1"])


def test_localization_empty_patch_zero():
    ev = sim.LocalizationEvaluator(tau=0.5)
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate("", inst)
    assert result.quality == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ReferenceJudgeEvaluator
# ---------------------------------------------------------------------------


def test_ref_judge_gold_in_prompt_and_model_used(monkeypatch):
    class _Msg:
        content = "0.85"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _Resp()

    import litellm as _litellm

    monkeypatch.setattr(_litellm, "completion", fake_completion)

    ev = sim.ReferenceJudgeEvaluator(
        judge_model="litellm_proxy/claude-opus-4-8",
        tau=0.5,
        api_base="http://proxy",
        api_key="sk-test",
    )
    inst = make_instance()
    result = ev.evaluate(GOLD, inst)

    assert captured["model"] == "litellm_proxy/claude-opus-4-8"
    assert inst["patch"] in captured["messages"][0]["content"]  # gold in prompt
    assert result.quality == pytest.approx(0.85)
    assert result.passed is True


def test_ref_judge_unparseable_clamps_to_zero(monkeypatch):
    class _Msg:
        content = "not-a-number"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    import litellm as _litellm

    monkeypatch.setattr(_litellm, "completion", lambda **_kw: _Resp())

    ev = sim.ReferenceJudgeEvaluator(judge_model="m", tau=0.5)
    result = ev.evaluate("patch", make_instance())
    assert result.quality == pytest.approx(0.0)


def test_ref_judge_clamps_above_one(monkeypatch):
    class _Msg:
        content = "1.5"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    import litellm as _litellm

    monkeypatch.setattr(_litellm, "completion", lambda **_kw: _Resp())

    ev = sim.ReferenceJudgeEvaluator(judge_model="m", tau=0.5)
    result = ev.evaluate("patch", make_instance())
    assert result.quality == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CompositeEvaluator
# ---------------------------------------------------------------------------


def test_composite_returns_primary_as_quality():
    ev = sim.CompositeEvaluator(
        evaluators=[
            ("norm_edit_sim", sim.NormalizedEditSimEvaluator(tau=0.5)),
            ("localization", sim.LocalizationEvaluator(tau=0.5)),
        ],
        primary="file_f1",
        tau=0.5,
    )
    inst = make_instance(gold=_GOLD_LOC)
    result = ev.evaluate(_GOLD_LOC, inst)
    sigs = json.loads(result.detail)
    assert result.quality == pytest.approx(sigs["file_f1"])


def test_composite_detail_carries_all_signals():
    ev = sim.CompositeEvaluator(
        evaluators=[
            ("norm_edit_sim", sim.NormalizedEditSimEvaluator(tau=0.5)),
            ("localization", sim.LocalizationEvaluator(tau=0.5)),
        ],
        primary="file_f1",
        tau=0.5,
    )
    inst = make_instance()
    result = ev.evaluate(GOLD, inst)
    sigs = json.loads(result.detail)
    assert "norm_edit_sim" in sigs
    assert "file_f1" in sigs
    assert "hunk_jaccard" in sigs


def test_composite_failing_sub_evaluator_degrades_to_zero():
    class _BrokenEv(sim.PatchEvaluator):
        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            raise RuntimeError("boom")

    ev = sim.CompositeEvaluator(
        evaluators=[
            ("broken", _BrokenEv()),
            ("norm_edit_sim", sim.NormalizedEditSimEvaluator(tau=0.5)),
        ],
        primary="norm_edit_sim",
        tau=0.5,
    )
    inst = make_instance()
    result = ev.evaluate(GOLD, inst)
    sigs = json.loads(result.detail)
    assert sigs["broken"] == pytest.approx(0.0)
    assert result.quality > 0.0  # norm_edit_sim still ran


# ---------------------------------------------------------------------------
# get_evaluator panel dispatch
# ---------------------------------------------------------------------------


def test_get_evaluator_panel_dispatch():
    cfg = _cfg(evaluator="panel")
    ev = sim.get_evaluator(cfg)
    assert isinstance(ev, sim.CompositeEvaluator)


def test_get_evaluator_panel_with_apply_check():
    cfg = _cfg(evaluator="panel", apply_check=True)
    ev = sim.get_evaluator(cfg)
    assert isinstance(ev, sim.CompositeEvaluator)
    names = [n for n, _ in ev.evaluators]
    assert "apply_check" in names


# ---------------------------------------------------------------------------
# compute_signal_agreement
# ---------------------------------------------------------------------------


def test_signal_agreement_known_values():
    # Perfectly correlated pair → spearman ≈ 1; uncorrelated pair → 0 or small.
    sigs = {
        "a": [[0.1, 0.5, 0.9]],
        "b": [[0.1, 0.5, 0.9]],  # identical → spearman = 1
    }
    result = sim.compute_signal_agreement(sigs)
    assert result["spearman"]["a_vs_b"] == pytest.approx(1.0, abs=1e-9)
    assert result["mean_abs_diff"]["a_vs_b"] == pytest.approx(0.0, abs=1e-9)


def test_signal_agreement_returns_expected_keys():
    sigs: dict[str, list[list[float]]] = {"x": [[0.2, 0.4]], "y": [[0.8, 0.6]]}
    result = sim.compute_signal_agreement(sigs)
    assert "signal_names" in result
    assert "spearman" in result
    assert "mean_abs_diff" in result
    assert "x_vs_y" in result["spearman"]


def test_signal_agreement_empty():
    result = sim.compute_signal_agreement({})
    assert result["spearman"] == {}
    assert result["mean_abs_diff"] == {}


# ---------------------------------------------------------------------------
# signals persistence in run_experiment
# ---------------------------------------------------------------------------


def test_run_experiment_signals_matrices_shape():
    cfg = _cfg(k=2, t=3, agent="stub", evaluator="proxy")
    instances = [make_instance(i) for i in range(2)]
    _, _, sigs, _, _, _ = sim.run_experiment(instances, cfg)
    assert isinstance(sigs, dict)
    # proxy evaluator detail is not valid JSON → falls back to {"primary": quality}
    assert "primary" in sigs
    for matrix in sigs.values():
        assert len(matrix) == 2  # 2 instances
        assert len(matrix[0]) == 3  # 3 rounds


def test_run_experiment_panel_signals_matrices_names():
    # panel evaluator should populate file_f1, hunk_jaccard, norm_edit_sim, ref_judge
    # but we only check that it runs and populates the known non-API signals.
    cfg = _cfg(k=1, t=2, agent="stub", evaluator="panel", apply_check=False)
    instances = [make_instance(0)]

    class _FakeJudge(sim.PatchEvaluator):
        def evaluate(self, patch: str, instance: dict) -> sim.PatchEvaluation:
            return sim.PatchEvaluation(quality=0.5, passed=True, detail="ref-judge")

    import unittest.mock as mock

    with mock.patch.object(sim.ReferenceJudgeEvaluator, "evaluate", _FakeJudge().evaluate):
        _, _, sigs, _, _, _ = sim.run_experiment(instances, cfg)

    assert "file_f1" in sigs
    assert "norm_edit_sim" in sigs


# ---------------------------------------------------------------------------
# Negative test: seed partitions cache keys
# ---------------------------------------------------------------------------


def test_prompt_cache_key_different_seeds_produce_different_keys():
    """Negative test: two distinct seeds MUST produce different cache keys for the
    same (model, instance_id, round, prompt). Fails if seed is dropped from the hash."""
    args = ("m", "repo__pkg-1", 0, "the prompt")
    key0 = sim.prompt_cache_key(*args, seed=0)
    key1 = sim.prompt_cache_key(*args, seed=1)
    assert key0 != key1, "seed=0 and seed=1 must produce distinct cache keys"


def test_prompt_cache_key_seed_default_matches_explicit_zero():
    """seed=0 default must be backward-compatible: same key as explicit seed=0."""
    args = ("m", "repo__pkg-1", 0, "the prompt")
    assert sim.prompt_cache_key(*args) == sim.prompt_cache_key(*args, seed=0)


# ---------------------------------------------------------------------------
# Negative test: temperature partitions cache keys
# ---------------------------------------------------------------------------


def test_prompt_cache_key_different_temperatures_produce_different_keys():
    """Negative test: two distinct temperatures MUST produce different cache keys.
    Fails if temperature is dropped from the hash."""
    args = ("m", "repo__pkg-1", 0, "the prompt")
    key_default = sim.prompt_cache_key(*args, temperature=0.0)
    key_hot = sim.prompt_cache_key(*args, temperature=0.25)
    assert key_default != key_hot, "temperature=0.0 and temperature=0.25 must differ"


def test_prompt_cache_key_temperature_default_matches_explicit_zero():
    """temperature=0.0 must be backward-compatible: same key as pre-temperature callers
    (existing pinned cache entries, all written at the old hardcoded temperature=0, stay
    valid)."""
    args = ("m", "repo__pkg-1", 0, "the prompt")
    assert sim.prompt_cache_key(*args) == sim.prompt_cache_key(*args, temperature=0.0)


# ---------------------------------------------------------------------------
# Negative test: image_name returns @sha256: when lockfile has entry
# ---------------------------------------------------------------------------


def test_harness_image_name_returns_digest_when_pinned():
    """Negative test: image_name MUST return @sha256: when the lockfile has a digest.
    Fails if digest pinning is removed from image_name."""
    ev = sim.HarnessEvaluator()
    ev._digests = {"psf__requests-1963": "abc123def456"}
    name = ev.image_name("psf__requests-1963")
    assert "@sha256:" in name, "image_name must include @sha256: when digest is set"
    assert "abc123def456" in name


def test_harness_image_name_fallback_to_latest_when_no_digest():
    """When no digest is in the lockfile, fall back to :latest (with a warning)."""
    import unittest.mock as mock

    ev = sim.HarnessEvaluator()
    ev._digests = {}
    with mock.patch("builtins.print"):
        name = ev.image_name("psf__requests-1963")
    assert name.endswith(":latest")
    assert "@sha256:" not in name


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cfg(
    *,
    model: str = "bedrock/test-model",
    k: int = 5,
    t: int = 4,
    tau: float = 0.6,
    evaluator: str = "proxy",
    agent: str = "stub",
    cache_dir: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    step_limit: int = 40,
    judge_model: str = "litellm_proxy/claude-opus-4-8",
    panel_primary: str = "file_f1",
    apply_check: bool = False,
    n_seeds: int = 1,
    seed_base: int = 0,
    temperature: float = 0.25,
    feedback_mode: str = "diagnostic",
    tier_policy: str = "fixed",
    tier_models: tuple[str, ...] = (),
    concurrency: int = 1,
    exclude_ids: tuple[str, ...] = (),
) -> sim.Config:
    return sim.Config(
        model=model,
        k=k,
        t=t,
        tau=tau,
        evaluator=evaluator,
        agent=agent,
        cache_dir=cache_dir,
        api_base=api_base,
        api_key=api_key,
        step_limit=step_limit,
        judge_model=judge_model,
        panel_primary=panel_primary,
        apply_check=apply_check,
        n_seeds=n_seeds,
        seed_base=seed_base,
        temperature=temperature,
        feedback_mode=feedback_mode,
        tier_policy=tier_policy,
        tier_models=tier_models,
        concurrency=concurrency,
        exclude_ids=exclude_ids,
    )


# ---------------------------------------------------------------------------
# Secret-redaction tests
# ---------------------------------------------------------------------------


def test_results_json_never_leaks_api_key_or_hostname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() must never write api_key starting with 'sk-' or the internal hostname."""
    figdir = tmp_path / "figures"
    datadir = tmp_path / "data"
    figdir.mkdir()
    datadir.mkdir()
    monkeypatch.setattr(sim, "FIGDIR", str(figdir))
    monkeypatch.setattr(sim, "DATADIR", str(datadir))
    monkeypatch.setenv("LLM_SIM_K", "2")
    monkeypatch.setenv("LLM_SIM_T", "2")
    monkeypatch.setenv("LLM_SIM_AGENT", "stub")
    monkeypatch.setenv("LLM_SIM_EVALUATOR", "proxy")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-ShouldNotAppear123")
    monkeypatch.setenv("LITELLM_API_BASE", "https://proxy.example.com")
    monkeypatch.setattr(
        sim, "load_swe_tasks", lambda k: pytest.fail("stub run must not hit the network")
    )

    sim.main()

    content = (datadir / "llm_results.json").read_text()
    assert "sk-ShouldNotAppear123" not in content, "api_key leaked into results JSON"
    assert "proxy.example.com" not in content, "api_base leaked into results JSON"
