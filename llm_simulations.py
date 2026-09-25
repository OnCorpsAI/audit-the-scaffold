#!/usr/bin/env python3
"""Real-LLM SWE-bench refinement experiment for
"Audit the Scaffold, Not the Checkpoint".

This is the empirical counterpart to the stylized decision-stump experiments in
``simulations.py`` / ``swe_bench_simulations.py``. Here an *actual coding agent*
(`mini-swe-agent`, litellm-native, on AWS Bedrock) plays the role of the weak
learner, wrapped in a boosting/refinement **outer loop**: for each SWE-bench Lite
task we re-invoke the agent for ``T`` rounds, seeding round ``t`` with the prior
patch plus a critique (a Reflexion-style loop). Each round's patch ``p_t`` is
scored, per-round metrics are recorded, and the critique is fed back. ``T`` rounds
per task is the refinement game of Theorem 5 (``thm:refinement``).

The scaffold -- outer loop + metrics + a pluggable *evaluator seam* -- is what
makes this a test of the theory; the agent and the evaluator are swappable behind
thin adapters selected by environment variables.

Design constraints (see ``tasks/llm-swebench-experiment.md``):

* **Leakage guard (critical).** The agent NEVER sees the gold ``instance['patch']``,
  the similarity score, or evaluator output -- enforced by construction. Only the
  evaluator reads gold.
* **Reproducibility.** Every model response is cached under ``data/llm_cache/``
  keyed by SHA-256 of ``(model, instance_id, round, prompt)``. The cache -- not the
  (stochastic) model -- is the determinism contract; it is gitignored.
* **Fidelity now, harness later.** The default evaluator is a proxy
  (edit-distance-to-gold); a real apply-patch + pytest-in-Docker
  ``HarnessEvaluator`` is stubbed as a one-env-var drop-in.

Conventions mirror ``simulations.py``: ``RNG_SEED``, ``FIGDIR``, ``DATADIR``,
``matplotlib.use("Agg")``, dual ``.pdf`` + ``.png dpi=150`` savefig, JSON dumps to
``data/``.

Env-var configuration (all optional; documented defaults):
    LLM_SIM_MODEL       litellm/Bedrock model id (default: a cheap Haiku id)
    LLM_SIM_K           number of tasks            (default: 5)
    LLM_SIM_T           refinement rounds per task (default: 4)
    LLM_SIM_TAU         pass threshold on quality  (default: 0.6)
    LLM_SIM_EVALUATOR   proxy | judge | harness    (default: proxy)
    LLM_SIM_AGENT       mini | stub                (default: mini)
    LLM_SIM_STEP_LIMIT  agent steps per round      (default: 40)
    LLM_SIM_TEMPERATURE weak-learner sampling temp (default: 0.25; judges stay at 0)
    LITELLM_API_BASE    LiteLLM proxy URL          (required for the real agent; no default)
    LITELLM_API_KEY     LiteLLM proxy key (the default credential path; never hardcoded)
    AWS creds               still needed for the legacy ``bedrock/<model>`` path
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the shared reproducibility seed from the stylized experiments. This is the
# one (allowed) dependency of `llm_simulations` on `simulations`, enforced by tach.
from simulations import RNG_SEED

# ---- Reproducibility ----
np.random.seed(RNG_SEED)

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
DATADIR = os.path.join(os.path.dirname(__file__), "data")
_DIGEST_LOCKFILE = os.path.join(DATADIR, "swebench_image_digests.json")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# Matplotlib with the headless Agg backend (mirrors simulations.py).
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The weak learner routes through a LiteLLM proxy (an OpenAI-compatible gateway). The
# ``litellm_proxy/`` prefix tells the litellm SDK to treat ``api_base`` as such a
# gateway; set the endpoint via ``LITELLM_API_BASE`` and credentials via
# ``LITELLM_API_KEY`` (never hardcoded). No endpoint is baked in — with the proxy
# unset, litellm falls back to its own resolution (e.g. a ``bedrock/<model>`` id +
# AWS env creds). Haiku is the deliberate choice: the weakest coder leaves the most
# refinement headroom across rounds (a stronger model that one-shots round 0 flattens
# the diminishing-returns curve the refinement theorem predicts).
DEFAULT_API_BASE = None  # set via LITELLM_API_BASE; no endpoint baked into source
DEFAULT_MODEL = "litellm_proxy/claude-haiku-4-5-20251001"


# ============================================================
# 0. Configuration
# ============================================================


@dataclass(frozen=True)
class Config:
    """Experiment configuration, normally built from the environment."""

    model: str
    k: int
    t: int
    tau: float
    evaluator: str
    agent: str
    cache_dir: str | None
    # Per-round inner-loop budget for the mini-swe-agent weak learner. Bounds the
    # agent's steps within ONE refinement round (the outer loop is ``t`` rounds).
    step_limit: int
    # Weak-learner sampling temperature. Judges (LLMJudgeEvaluator,
    # ReferenceJudgeEvaluator) stay hardcoded at temperature=0 regardless of this
    # value -- only the agent producing candidate patches gets the diversity knob.
    temperature: float
    # LiteLLM proxy routing. ``api_base``/``api_key`` are forwarded to every litellm
    # call (weak learner and judge). Credentials flow through the proxy only.
    api_base: str | None
    api_key: str | None
    # Panel evaluator knobs (LLM_SIM_EVALUATOR=panel).
    judge_model: str  # model for ReferenceJudgeEvaluator
    panel_primary: str  # which panel signal drives q (default: file_f1)
    apply_check: bool  # enable ApplyCheckEvaluator (requires network)
    # Variance / seed layer. n_seeds > 1 runs the task loop n_seeds times, each with
    # a distinct seed offset, yielding independent replicates for CI estimation.
    n_seeds: int
    seed_base: int
    # Feedback ablation: what the weak learner sees between refinement rounds.
    # "diagnostic" (default): prior best patch + evaluator-derived critique (today's
    # behavior). "blind": prior best patch, no critique. "independent": neither --
    # each round is a fresh, independent draw. See run_task.
    feedback_mode: str
    # Capability-tier assignment: which model runs a given (task, seed). "fixed"
    # (default): always config.model -- current/pre-Phase-2 behavior. "randomized":
    # picks uniformly from tier_models, decorrelated from task content, so a
    # measured capability gap across tiers is not confounded with which task each
    # tier happens to get. See assign_tier_model.
    tier_policy: str
    tier_models: tuple[str, ...]
    # Opt-in intra-seed parallelism (seed-outer, task-inner -- see
    # _run_experiment_concurrent). <= 1 takes the exact sequential path,
    # byte-identical to pre-Phase-B behavior.
    concurrency: int
    # Denylist of instance_ids to drop from task selection before picking k
    # (e.g. tasks confirmed broken at the harness/infrastructure level, not a
    # model-quality signal -- see select_diverse_tasks/load_swe_tasks).
    exclude_ids: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            model=os.environ.get("LLM_SIM_MODEL", DEFAULT_MODEL),
            k=int(os.environ.get("LLM_SIM_K", "5")),
            t=int(os.environ.get("LLM_SIM_T", "4")),
            tau=float(os.environ.get("LLM_SIM_TAU", "0.6")),
            evaluator=os.environ.get("LLM_SIM_EVALUATOR", "proxy"),
            agent=os.environ.get("LLM_SIM_AGENT", "mini"),
            cache_dir=None,
            step_limit=int(os.environ.get("LLM_SIM_STEP_LIMIT", "40")),
            temperature=float(os.environ.get("LLM_SIM_TEMPERATURE", "0.25")),
            api_base=os.environ.get("LITELLM_API_BASE", DEFAULT_API_BASE),
            api_key=os.environ.get("LITELLM_API_KEY"),
            judge_model=os.environ.get("LLM_SIM_JUDGE_MODEL", "litellm_proxy/claude-opus-4-8"),
            panel_primary=os.environ.get("LLM_SIM_PANEL_PRIMARY", "file_f1"),
            apply_check=os.environ.get("LLM_SIM_APPLY_CHECK", "0") == "1",
            n_seeds=int(os.environ.get("LLM_SIM_SEEDS", "1")),
            seed_base=int(os.environ.get("LLM_SIM_SEED_BASE", "0")),
            feedback_mode=os.environ.get("LLM_SIM_FEEDBACK", "diagnostic"),
            tier_policy=os.environ.get("LLM_SIM_TIER_POLICY", "fixed"),
            tier_models=tuple(
                m.strip() for m in os.environ.get("LLM_SIM_TIER_MODELS", "").split(",") if m.strip()
            ),
            concurrency=int(os.environ.get("LLM_SIM_CONCURRENCY", "1")),
            exclude_ids=tuple(
                x.strip() for x in os.environ.get("LLM_SIM_EXCLUDE_IDS", "").split(",") if x.strip()
            ),
        )


def _litellm_kwargs(config: Config) -> dict[str, str]:
    """The proxy-routing kwargs to splat into every litellm call.

    The single place credentials enter the litellm layer. Only non-``None`` fields
    are included so an unset proxy leaves litellm to its own resolution (e.g. AWS env
    for a ``bedrock/`` model). The key is passed through, never logged.
    """
    kwargs: dict[str, str] = {}
    if config.api_base is not None:
        kwargs["api_base"] = config.api_base
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    return kwargs


# ============================================================
# 1. Evaluator seam (enables the real test harness later)
# ============================================================


@dataclass
class PatchEvaluation:
    """The result of scoring one patch. ``quality`` in [0, 1]; ``passed`` is the
    boolean success criterion the boosting analysis consumes.

    ``feedback`` carries harness-generated text that is safe to show the agent
    (test names and failure details from pytest execution, NOT gold-patch content).
    Left empty by non-harness evaluators; populated by ``HarnessEvaluator``.
    """

    quality: float
    passed: bool
    detail: str = ""
    feedback: str = ""
    fraction_passing: float = 0.0


class PatchEvaluator(ABC):
    """Scores a candidate patch for one task.

    Implementations are the *only* place gold data is read. ``evaluate`` receives
    the full SWE-bench instance (which contains the gold ``patch``); the agent side
    of the pipeline never touches it.
    """

    @abstractmethod
    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        raise NotImplementedError

    def make_critique(
        self,
        patch: str,
        evaluation: PatchEvaluation,
        instance: dict,  # noqa: ARG002
    ) -> str:
        """Generate a leakage-free critique for the next refinement round.

        Default: keys only on the agent's own patch structure (no evaluator output).
        ``HarnessEvaluator`` overrides this to include actual test results, which are
        safe because pytest output comes from test execution, NOT the gold patch.
        """
        return _make_critique(patch)

    def teardown(self, instance_id: str) -> None:  # noqa: ARG002, B027
        """Called by run_experiment after all rounds for one task complete.

        Default no-op. HarnessEvaluator overrides this to optionally remove
        the per-instance Docker image and reclaim disk.
        """


class ProxyEditDistanceEvaluator(PatchEvaluator):
    """Default evaluator: normalized edit-distance similarity of the candidate diff
    to the gold diff (``rapidfuzz``). ``passed = quality >= tau``.

    NOTE (proxy validity): edit-distance-to-gold measures *textual proximity*, not
    functional correctness. It is a stand-in for the Docker test harness
    (``HarnessEvaluator``) and results should be read as evidence about *refinement
    dynamics*, not solve rates.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        from rapidfuzz.distance import Levenshtein

        gold = instance.get("patch", "") or ""
        quality = float(Levenshtein.normalized_similarity(patch or "", gold))
        return PatchEvaluation(
            quality=quality,
            passed=quality >= self.tau,
            detail="proxy:edit-distance-to-gold",
        )


class LLMJudgeEvaluator(PatchEvaluator):
    """Optional evaluator: an LLM scores the patch against the *problem statement*
    (never the gold patch), returning a quality in [0, 1]. Kept leakage-free by
    construction -- gold is not passed to the judge."""

    def __init__(
        self, model: str, tau: float, api_base: str | None = None, api_key: str | None = None
    ) -> None:
        self.model = model
        self.tau = tau
        self.api_base = api_base
        self.api_key = api_key

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        import litellm

        prompt = (
            "You are grading a proposed code patch against a problem statement.\n"
            "Return ONLY a float in [0,1] for how well the patch resolves it.\n\n"
            f"PROBLEM:\n{instance.get('problem_statement', '')}\n\n"
            f"PATCH:\n{patch}\n"
        )
        proxy_kwargs = {}
        if self.api_base is not None:
            proxy_kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            proxy_kwargs["api_key"] = self.api_key
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **proxy_kwargs,
        )
        text = resp.choices[0].message.content or "0"
        try:
            quality = float(text.strip().split()[0])
        except (ValueError, IndexError):
            quality = 0.0
        quality = min(1.0, max(0.0, quality))
        return PatchEvaluation(quality=quality, passed=quality >= self.tau, detail="llm-judge")


# Sentinels echoed by the in-container script to distinguish an infrastructure
# failure (test_patch won't apply -> raise, task SKIPPED) from an agent-patch
# failure (candidate won't apply -> legitimate 0 with actionable feedback).
_TEST_PATCH_FAILED = "__HARNESS_TEST_PATCH_FAILED__"
_CANDIDATE_FAILED = "__HARNESS_CANDIDATE_APPLY_FAILED__"


class HarnessEvaluator(PatchEvaluator):
    """Real test-execution evaluator using the SWE-bench Docker image.

    For each instance, pulls ``swebench/sweb.eval.x86_64.{instance_id}:latest``
    (skipped if already local), applies the test_patch and candidate patch inside
    the container, runs the FAIL_TO_PASS and PASS_TO_PASS tests with pytest, and
    returns the fraction of FAIL_TO_PASS tests that now pass.

    Images are x86_64; on Apple Silicon they run via Rosetta 2
    (``--platform linux/amd64``).  The same image is reused for all T rounds of one
    task. After the final round, ``teardown(instance_id)`` removes the image when
    ``LLM_HARNESS_AUTO_REMOVE=1`` is set.

    Env vars:
        LLM_HARNESS_PLATFORM      docker --platform flag (default: linux/amd64)
        LLM_HARNESS_PULL_TIMEOUT  seconds allowed for docker pull (default: 600)
        LLM_HARNESS_EVAL_TIMEOUT  seconds allowed for docker run per round (default: 300)
        LLM_HARNESS_AUTO_REMOVE   set to 1 to rmi image in teardown (default: 0)
        LLM_SIM_HARNESS_BACKEND   local | modal -- where the eval container runs (default: local)
        LLM_HARNESS_MODAL_TIMEOUT seconds allowed for one Modal exec (default: 120; guards
                                  against the gvisor network-stall tail measured empirically)

    The ``modal`` backend runs the *same* eval script this class always has; only "how the
    container is executed" changes (``_run_local`` vs ``_run_modal``), so ``run_task`` and
    ``make_critique`` see an identical result shape from either backend. Because a Modal
    Sandbox is reused across a task's T rounds (pay boot once per task, not once per round --
    unlike the ephemeral ``docker run --rm`` local path), each round resets the container's
    working tree before applying that round's patches.
    """

    def __init__(self) -> None:
        self.platform = os.environ.get("LLM_HARNESS_PLATFORM", "linux/amd64")
        self.pull_timeout = int(os.environ.get("LLM_HARNESS_PULL_TIMEOUT", "600"))
        self.eval_timeout = int(os.environ.get("LLM_HARNESS_EVAL_TIMEOUT", "300"))
        self.auto_remove = os.environ.get("LLM_HARNESS_AUTO_REMOVE", "0") == "1"
        self.backend = os.environ.get("LLM_SIM_HARNESS_BACKEND", "local")
        self.modal_timeout = int(os.environ.get("LLM_HARNESS_MODAL_TIMEOUT", "120"))
        # Sandbox lifetime must cover all T rounds of one task (boot once, reuse across
        # rounds -- see class docstring). 900s was sized for sequential execution; under
        # LLM_SIM_CONCURRENCY > 1, contention on the shared proxy/Modal API can stretch a
        # task's wall-clock span well past its compute time, expiring the sandbox mid-task
        # (surfaces as modal.exception.NotFoundError on a later round's exec).
        self.sandbox_timeout = int(os.environ.get("LLM_HARNESS_SANDBOX_TIMEOUT", "900"))
        self._modal_app: Any = None
        self._modal_sandboxes: dict[str, Any] = {}
        # Guards ONLY the lazy _modal_app singleton init. Seed-outer concurrency
        # (Phase B2) makes each instance_id single-threaded, so _modal_sandboxes
        # itself needs no locking -- but two different instance_ids' first-ever
        # sandbox creation can race on "is _modal_app still None?" concurrently.
        self._modal_lock = threading.Lock()
        # instance_id -> (resolved_ftp, resolved_ptp, runnable_ids), memoizing
        # _resolve_test_ids across a task's T rounds (the correction is a fixed
        # property of the pinned image + test_patch, not of the round or
        # candidate).
        self._resolved_test_ids: dict[str, tuple[list[str], list[str], frozenset[str]]] = {}
        self._digests: dict[str, str] = {}
        if os.path.exists(_DIGEST_LOCKFILE):
            import json as _json

            raw = _json.loads(Path(_DIGEST_LOCKFILE).read_text())
            self._digests = {k: v for k, v in raw.items() if k != "_comment" and v}

    def image_name(self, instance_id: str) -> str:
        """Return a digest-pinned image ref if the lockfile has an entry, else :latest.

        Populate data/swebench_image_digests.json by running:
          docker pull swebench/sweb.eval.x86_64.<tag>:latest
          docker inspect --format '{{index .RepoDigests 0}}' <image>
        A missing or empty digest falls back to :latest with a loud warning so
        unpinned runs are never silent.
        """
        # SWE-bench dataset uses org__repo-issue; DockerHub uses org_1776_repo-issue.
        tag = instance_id.replace("__", "_1776_", 1)
        base = f"swebench/sweb.eval.x86_64.{tag}"
        digest = self._digests.get(instance_id, "")
        if digest:
            return f"{base}@sha256:{digest.removeprefix('sha256:')}"
        print(
            f"    [harness] WARNING: no digest pinned for {instance_id}; "
            "using :latest (image drift risk). Populate data/swebench_image_digests.json.",
            flush=True,
        )
        return f"{base}:latest"

    def _ensure_pulled(self, image: str) -> None:
        check = subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", image],  # noqa: S607
            capture_output=True,
            timeout=15,
            check=False,
        )
        if check.returncode != 0:
            print(f"    [harness] pulling {image} ...", flush=True)
            subprocess.run(  # noqa: S603
                ["docker", "pull", "--platform", self.platform, image],  # noqa: S607
                check=True,
                timeout=self.pull_timeout,
            )

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        import json as _json

        instance_id: str = instance["instance_id"]
        image = self.image_name(instance_id)

        ftp: list[str] = _json.loads(instance.get("FAIL_TO_PASS") or "[]")
        ptp: list[str] = _json.loads(instance.get("PASS_TO_PASS") or "[]")
        test_patch: str = instance.get("test_patch") or ""

        if not ftp:
            return PatchEvaluation(quality=0.0, passed=False, detail="harness:no-ftp-tests")

        candidate = patch or ""
        ftp, ptp, runnable_ids = self._resolve_test_ids(image, instance_id, test_patch, ftp, ptp)
        # Newline-joined, unquoted: a large PASS_TO_PASS suite (seen live: 1690 tests for
        # one xarray task) blows Modal's 65536-byte exec CMD-argument limit if embedded
        # directly in `script` below (ARG_MAX InvalidError, deterministic per task). Written
        # to a file instead and split by `xargs -d` inside the container, which also sidesteps
        # having to shell-quote test IDs that themselves contain spaces. `runnable_ids` (not
        # `ftp + ptp` directly) excludes any id _resolve_test_ids couldn't confidently resolve
        # -- feeding those to pytest would fail collection for the whole batch, not just
        # themselves (see _resolve_test_ids); dropped ids still score correctly via `ftp`/`ptp`
        # below since they'll never appear in the JUnit report either way.
        test_ids = "\n".join(sorted(runnable_ids))
        # test_patch first (adds the regression tests), then candidate (the fix).
        # bash -lc sources /root/.bashrc which activates the conda testbed env.
        #
        # Plain ``git apply`` -- NOT ``--allow-empty``: the eval images ship git
        # 2.34.1, which predates that flag (git 2.44) and rejects it, silently
        # dropping BOTH patches. The test_patch is infrastructure and must apply,
        # so we surface a sentinel + non-zero exit on failure. The candidate is
        # agent output that may be empty (agent gave up) or malformed; we guard
        # with ``[ -s ]`` and tolerate an apply failure (it legitimately scores 0).
        # Determinism env vars reduce eval-side noise from hash-seed and locale
        # variation. Residual limitation: QEMU/Rosetta timing flakiness on Apple
        # Silicon (--platform linux/amd64) is NOT cured by env vars; the robust
        # fix is a native x86 Linux eval host.
        #
        # Scoring is JUnit-XML-based (not text-scraped): `--no-header` is rejected
        # as an unrecognized flag by some pinned older pytest versions, and even
        # where it isn't, ANSI color escapes around " PASSED"/" FAILED" in `-v`
        # output silently broke naive substring matching. Matching is done via
        # each <testcase>'s `classname`+`name` (see `_node_id_signature`), NOT
        # `file`+`classname` reconstruction -- `file` can point at a decorator's
        # own definition site rather than the test's module for wrapped tests
        # (confirmed live: matplotlib's `@image_comparison`-wrapped tests report
        # file=matplotlib/testing/decorators.py), which broke reconstruction.
        # `classname`/`name` exist in both the pytest xunit1 and xunit2 junit-xml
        # schema variants, so no `-o junit_family=...` override is needed either.
        # `-v --tb=line` stays only for the human-readable failure hints surfaced
        # in agent feedback.
        #
        # The trailing `--` guards against a test id that happens to look like a
        # CLI flag being misparsed as one -- confirmed load-bearing on pytest
        # 3.3.1 (the oldest pinned version seen live, e.g. astropy-6938) when an
        # `-o` override was still in this command: `-o` there is declared with
        # nargs='*' and, without `--`, greedily swallows the xargs-appended
        # test-id positional arguments as more override values, erroring with
        # "-o/--override-ini expects option=value style" the moment it hits one
        # that isn't shaped like `key=value`. Kept even after removing `-o` as
        # cheap, zero-risk defense against the same class of issue recurring.
        script = (
            "export PYTHONHASHSEED=0 TZ=UTC LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1; "
            "set -o pipefail; "
            "cd /testbed; "
            "git apply /patches/test_patch.diff || "
            f"{{ echo '{_TEST_PATCH_FAILED}'; exit 3; }}; "
            "if [ -s /patches/candidate.diff ]; then "
            f"git apply /patches/candidate.diff 2>&1 || echo '{_CANDIDATE_FAILED}'; "
            "fi; "
            "xargs -d '\\n' -a /patches/test_ids.txt "
            "python -m pytest --tb=line -v "
            "--junit-xml=/patches/report.xml -- 2>&1 || true; "
            "cat /patches/report.xml 2>/dev/null || true"
        )

        raw = self._run_backend(instance_id, image, test_patch, candidate, test_ids, script)

        if _TEST_PATCH_FAILED in raw:
            raise RuntimeError(
                f"harness: test_patch failed to apply for {instance_id} "
                "(infrastructure error, not a patch quality signal)"
            )
        result = self._parse_junit_xml(raw, ftp, ptp)
        passed_ids, _failed_ids = self._junit_test_ids(raw)
        result.feedback = self._build_harness_feedback(raw, ftp, passed_ids)
        if _CANDIDATE_FAILED in raw:
            result.feedback = (
                "Your patch did not apply cleanly (git apply failed). "
                "Ensure the diff targets the correct file paths and line context.\n"
                + result.feedback
            )
        return result

    def _run_backend(
        self,
        instance_id: str,
        image: str,
        test_patch: str,
        candidate: str,
        test_ids: str,
        script: str,
    ) -> str:
        """Dispatch ``script`` to whichever backend is configured -- shared by the
        main eval run and ``_resolve_test_ids``'s collect-only probe."""
        if self.backend == "modal":
            return self._run_modal(instance_id, image, test_patch, candidate, test_ids, script)
        self._ensure_pulled(image)
        return self._run_local(image, test_patch, candidate, test_ids, script)

    def _resolve_test_ids(
        self,
        image: str,
        instance_id: str,
        test_patch: str,
        ftp: list[str],
        ptp: list[str],
    ) -> tuple[list[str], list[str], frozenset[str]]:
        """Correct stale FAIL_TO_PASS/PASS_TO_PASS node IDs before the real run.

        pytest auto-generates parametrize-test IDs from ``repr()``'d argument
        values, and that generation is not stable across pytest/library versions
        -- the exact ID string recorded in the SWE-bench dataset can stop
        matching what the *currently* pinned image's pytest collects. Runs
        ``pytest --collect-only -q`` (scoped to just the referenced files) to
        get the real, currently-valid node IDs, then substitutes the unique
        same-prefix match for any requested ID that isn't already exact (see
        ``_closest_test_id``) -- falling back to the literal requested ID when
        no confident match exists, so scoring is never worse than the
        un-corrected behavior.

        Returns ``(resolved_ftp, resolved_ptp, runnable_ids)``. The first two
        are for *scoring* -- same length as the inputs, used exactly like the
        raw FTP/PTP lists were before this fix. ``runnable_ids`` is the
        (possibly smaller) set actually safe to hand to pytest: a single
        unresolvable node-id string in the xargs'd test-id list makes the
        *entire* collection fail with "not found" (confirmed live on
        pylint-7080's PASS_TO_PASS list, which contains several genuinely
        truncated/malformed strings -- a SWE-bench dataset artifact, not
        pytest-version drift), which would zero out every other test in the
        same run, including FTP. Excluding an unresolvable id from
        ``runnable_ids`` means pytest never emits a testcase for it, so it
        correctly scores as "not passing" (FTP) or "not detected as
        regressed" (PTP) -- both the safe, conservative outcome, and no worse
        than before this fix existed.

        Cached per ``instance_id``: the correction is a fixed property of the
        pinned image + test_patch, not of the round or candidate, so rounds
        1..T-1 reuse round 0's result.
        """
        cached = self._resolved_test_ids.get(instance_id)
        if cached is not None:
            return cached

        files = sorted({t.partition("::")[0] for t in ftp + ptp if t})
        if not files:
            result = (ftp, ptp, frozenset(ftp + ptp))
            self._resolved_test_ids[instance_id] = result
            return result

        import shlex

        quoted_files = " ".join(shlex.quote(f) for f in files)
        collect_script = (
            "cd /testbed; "
            "git apply /patches/test_patch.diff >/dev/null 2>&1 || true; "
            f"python -m pytest --collect-only -q {quoted_files} 2>&1 || true"
        )
        raw = self._run_backend(instance_id, image, test_patch, "", "", collect_script)
        collected = {line.strip() for line in raw.splitlines() if "::" in line.strip()}

        resolved_ftp = [self._closest_test_id(t, collected) for t in ftp]
        resolved_ptp = [self._closest_test_id(t, collected) for t in ptp]
        runnable_ids = frozenset(
            resolved
            for original, resolved in zip(ftp + ptp, resolved_ftp + resolved_ptp, strict=True)
            if resolved != original or original in collected
        )
        result = (resolved_ftp, resolved_ptp, runnable_ids)
        self._resolved_test_ids[instance_id] = result
        return result

    @staticmethod
    def _closest_test_id(requested: str, collected: set[str]) -> str:
        """Exact match passes through unchanged. Otherwise, substitute the
        collected node ID sharing the same pre-``[...]`` prefix (handles
        parametrize-repr drift) *only if exactly one candidate shares that
        prefix* -- an unambiguous, high-confidence signal. With zero or
        multiple prefix-sharing candidates, return ``requested`` unchanged.

        Deliberately does not fuzzy-match (``difflib`` or otherwise) among
        multiple same-prefix candidates, even though that sounds like a safe
        restriction. Confirmed live on pylint-7080, whose recorded
        PASS_TO_PASS list contains several genuinely truncated/malformed
        node-id strings (a SWE-bench dataset artifact, not pytest-version
        drift): with only 2 real same-prefix candidates
        (``test_stdin[/testbed/tests/mymodule.py-mymodule-...]`` vs.
        ``test_stdin[mymodule.py-mymodule-mymodule.py]``), textual similarity
        against the truncated ``test_stdin[/mymodule.py]`` still picked the
        wrong one -- there is no size of "restricted enough" candidate pool
        that makes similarity-based disambiguation reliable here, since the
        truncation destroys exactly the information needed to tell candidates
        apart. A wrong substitution risks a false PASS_TO_PASS regression
        (quality penalty), which is worse than the uncorrected "not found,
        silently ignored" behavior this fix must never regress past.
        """
        if requested in collected:
            return requested
        prefix = requested.split("[", 1)[0]
        same_prefix = [c for c in collected if c.split("[", 1)[0] == prefix]
        return same_prefix[0] if len(same_prefix) == 1 else requested

    def _run_local(
        self, image: str, test_patch: str, candidate: str, test_ids: str, script: str
    ) -> str:
        """Run ``script`` in a fresh, ``--rm`` local Docker container (the default
        backend). One container per round -- the repo is always pristine, so no reset
        is needed between rounds."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "candidate.diff").write_text(candidate)
            (Path(tmp) / "test_patch.diff").write_text(test_patch)
            (Path(tmp) / "test_ids.txt").write_text(test_ids)
            proc = subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    self.platform,
                    "-v",
                    f"{tmp}:/patches:ro",
                    image,
                    "bash",
                    "-lc",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=self.eval_timeout,
            )
        return proc.stdout + proc.stderr

    def _get_or_create_modal_sandbox(self, instance_id: str, image: str) -> Any:
        """Get this task's warm Modal Sandbox, booting it on first use. Reused across
        all T rounds of ``instance_id`` -- boot is paid once per task, not once per
        round -- and torn down by ``teardown(instance_id)``."""
        sb = self._modal_sandboxes.get(instance_id)
        if sb is not None:
            return sb
        import modal

        with self._modal_lock:
            if self._modal_app is None:
                self._modal_app = modal.App.lookup("llm-sim-harness-eval", create_if_missing=True)
        img = modal.Image.from_registry(image, add_python="3.11")
        sb = modal.Sandbox.create(
            image=img, app=self._modal_app, cpu=2.0, memory=2048, timeout=self.sandbox_timeout
        )
        sb.exec("bash", "-c", "true").wait()  # block until the container has booted
        self._modal_sandboxes[instance_id] = sb
        return sb

    @staticmethod
    def _write_file_modal(sb: Any, path: str, content: str) -> None:
        """Stream ``content`` to the sandbox over stdin rather than embedding it in the
        command line: a base64-inlined ``echo`` blows Modal's 65536-byte CMD-argument
        limit for any patch beyond a few tens of KB (seen live as ``InvalidError:
        ... ARG_MAX``), and that failure is deterministic -- the same oversized patch
        crashes on every retry, including cache-replayed reruns of a resumed sweep."""
        import shlex

        proc = sb.exec("bash", "-c", f"cat > {shlex.quote(path)}")
        proc.stdin.write(content)
        proc.stdin.write_eof()
        proc.stdin.drain()
        proc.wait()

    def _run_modal(
        self,
        instance_id: str,
        image: str,
        test_patch: str,
        candidate: str,
        test_ids: str,
        script: str,
    ) -> str:
        """Run ``script`` in this task's warm Modal Sandbox.

        Unlike the local ``--rm`` container, the Sandbox persists across rounds, so the
        repo must be reset to a pristine checkout before each round's patches are
        (re-)applied -- otherwise round 2 would apply on top of round 1's leftover diff.
        """
        sb = self._get_or_create_modal_sandbox(instance_id, image)
        sb.exec("bash", "-lc", "cd /testbed && git checkout -- . && git clean -fdq").wait()
        sb.exec("bash", "-c", "mkdir -p /patches").wait()
        self._write_file_modal(sb, "/patches/test_patch.diff", test_patch)
        self._write_file_modal(sb, "/patches/candidate.diff", candidate)
        self._write_file_modal(sb, "/patches/test_ids.txt", test_ids)
        proc = sb.exec("bash", "-lc", script, timeout=self.modal_timeout)
        proc.wait()
        return proc.stdout.read() + proc.stderr.read()

    @staticmethod
    def _build_harness_feedback(
        raw_output: str, ftp: list[str], passed_sigs: set[tuple[str, str]]
    ) -> str:
        """Format pytest results as actionable agent feedback for the next round.

        Test names (FAIL_TO_PASS) are public instance metadata, not gold — safe to
        show the agent. ``passed_sigs`` comes from the JUnit-XML report (immune to
        ANSI color codes), matched via ``_node_id_signature`` -- see
        ``_junit_test_ids``. With ``--tb=line``, pytest appends one-line failure
        reasons to the short summary section; we surface the first five straight
        from the console text (supplementary hints only, not the pass/fail
        determination itself).
        """
        ftp_pass = sum(1 for t in ftp if HarnessEvaluator._node_id_signature(t) in passed_sigs)
        failed_ftp = [t for t in ftp if HarnessEvaluator._node_id_signature(t) not in passed_sigs]

        parts = [f"Test harness: {ftp_pass}/{len(ftp)} required tests passed."]
        if failed_ftp:
            parts.append("These required tests still fail (must go FAIL → PASS):")
            parts.extend(f"  {t}" for t in failed_ftp[:8])
            if len(failed_ftp) > 8:
                parts.append(f"  ... and {len(failed_ftp) - 8} more")
        # One-line failure hints from --tb=line summary ("FAILED path::test - ErrorType: msg")
        hints = [
            ln.strip()
            for ln in raw_output.splitlines()
            if ln.strip().startswith("FAILED ") and " - " in ln
        ][:5]
        if hints:
            parts.append("Failure details:")
            parts.extend(f"  {h}" for h in hints)
        if failed_ftp:
            parts.append("Revise your patch to make the failing required tests pass.")
        return "\n".join(parts)

    def make_critique(
        self,
        patch: str,
        evaluation: PatchEvaluation,
        instance: dict,  # noqa: ARG002
    ) -> str:
        """Use pytest output as critique — test names are public instance metadata."""
        return evaluation.feedback if evaluation.feedback else _make_critique(patch)

    @staticmethod
    def _node_id_signature(node_id: str) -> tuple[str, str]:
        """Forward-transform a pytest node id (``path/to/test.py::Class::test[p]``)
        into the ``(classname, name)`` pair pytest's own junit-xml plugin would
        report for it, for matching against ``_junit_test_ids``'s parsed
        signatures. Deliberately NOT the inverse direction (reconstructing a node
        id from a `<testcase>`'s attributes): a decorated test's ``file``/``line``
        can point at the *decorator's* definition site rather than the test's own
        module (confirmed live: matplotlib's ``@image_comparison``-wrapped tests
        report ``file=matplotlib/testing/decorators.py``), which breaks any
        reconstruction that trusts ``file``. ``classname`` doesn't have this
        problem -- pytest computes it from the collected item's own parent chain,
        not from the wrapped function's ``__code__.co_filename``.
        """
        file_part, _, rest = node_id.partition("::")
        parts = rest.split("::")
        name = parts[-1]
        cls_parts = parts[:-1]
        if file_part.endswith(".py"):
            module = file_part[: -len(".py")].replace("/", ".")
        else:
            module = file_part
        classname = module + ("." + ".".join(cls_parts) if cls_parts else "")
        return classname, name

    @staticmethod
    def _junit_test_ids(raw_output: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """Parse ``<testcase>`` elements out of a pytest ``--junit-xml`` report
        embedded somewhere in ``raw_output`` (the console text precedes it; the
        eval script ``cat``s the report file after the pytest run) into
        ``(passed_signatures, failed_signatures)``, each a set of ``(classname,
        name)`` pairs -- match against these via ``_node_id_signature``, not by
        reconstructing a node-id string from ``file``/``classname`` (see
        ``_node_id_signature``'s docstring for why). Immune to ANSI escapes and
        wording differences across pytest versions, since ``classname``/``name``
        attributes exist in both the xunit1 and xunit2 junit-xml schema variants.
        Skipped tests are excluded from both sets (neither a pass nor a fail
        signal for FTP/PTP).
        """
        import xml.etree.ElementTree as ET

        passed: set[tuple[str, str]] = set()
        failed: set[tuple[str, str]] = set()
        start = raw_output.find("<?xml")
        if start == -1:
            return passed, failed
        try:
            # S314: this XML is our own pytest --junit-xml report from a container
            # we already git-apply arbitrary diffs into and execute test code in --
            # not a boundary where an external/untrusted document could arrive.
            root = ET.fromstring(raw_output[start:])  # noqa: S314
        except ET.ParseError:
            return passed, failed
        for tc in root.iter("testcase"):
            if tc.find("skipped") is not None:
                continue
            sig = (tc.get("classname", ""), tc.get("name", ""))
            has_failure = tc.find("failure") is not None or tc.find("error") is not None
            (failed if has_failure else passed).add(sig)
        return passed, failed

    @staticmethod
    def _parse_junit_xml(raw_output: str, ftp: list[str], ptp: list[str]) -> PatchEvaluation:
        """Score FAIL_TO_PASS/PASS_TO_PASS against a JUnit-XML report -- see
        ``_junit_test_ids``/``_node_id_signature`` for the matching this relies on."""
        passed_sigs, failed_sigs = HarnessEvaluator._junit_test_ids(raw_output)

        ftp_pass = sum(1 for t in ftp if HarnessEvaluator._node_id_signature(t) in passed_sigs)
        ptp_fail = sum(1 for t in ptp if HarnessEvaluator._node_id_signature(t) in failed_sigs)

        fraction_passing = ftp_pass / len(ftp) if ftp else 0.0
        quality = fraction_passing
        if ptp_fail:
            quality *= 0.5  # regression penalty

        passed = ftp_pass == len(ftp) and ptp_fail == 0
        detail = f"harness:ftp={ftp_pass}/{len(ftp)},ptp_reg={ptp_fail}"
        return PatchEvaluation(
            quality=quality, passed=passed, detail=detail, fraction_passing=fraction_passing
        )

    def teardown(self, instance_id: str) -> None:
        sb = self._modal_sandboxes.pop(instance_id, None)
        if sb is not None:
            sb.terminate()
        if not self.auto_remove:
            return
        image = self.image_name(instance_id)
        subprocess.run(  # noqa: S603
            ["docker", "rmi", image],  # noqa: S607
            capture_output=True,
            check=False,
        )
        print(f"    [harness] removed {image}", flush=True)


class NormalizedEditSimEvaluator(PatchEvaluator):
    """Edit-distance similarity after stripping diff scaffolding.

    Keeps only added/removed content lines (drops ---/+++/@@/index/diff headers and
    unchanged context lines), normalises whitespace, then computes rapidfuzz
    normalized similarity. Fixes the shared-scaffolding inflation in the raw proxy.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau

    @staticmethod
    def _content_lines(diff: str) -> str:
        lines = []
        for line in diff.splitlines():
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                lines.append(" ".join(line.split()))
        return "\n".join(lines)

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        from rapidfuzz.distance import Levenshtein

        gold = instance.get("patch", "") or ""
        candidate_content = self._content_lines(patch or "")
        gold_content = self._content_lines(gold)
        quality = float(Levenshtein.normalized_similarity(candidate_content, gold_content))
        return PatchEvaluation(
            quality=quality,
            passed=quality >= self.tau,
            detail="norm-edit-sim",
        )


class LocalizationEvaluator(PatchEvaluator):
    """File F1 and hunk Jaccard vs gold.

    Uses ``unidiff`` to extract changed file paths and line-ranges. Returns
    quality = file_f1 (most robust non-API signal); sub-signals exposed in detail.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau

    @staticmethod
    def _parse(diff: str) -> dict[str, set[int]]:
        """Map filename → set of changed line numbers (added+removed)."""
        import unidiff  # type: ignore[import-untyped]

        file_lines: dict[str, set[int]] = {}
        try:
            patch_set = unidiff.PatchSet(diff or "")
        except Exception:  # noqa: BLE001
            return file_lines
        for patched_file in patch_set:
            lines: set[int] = set()
            for hunk in patched_file:
                for line in hunk:
                    if line.is_added or line.is_removed:
                        lines.add(line.source_line_no or line.target_line_no or 0)
            file_lines[patched_file.path] = lines
        return file_lines

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        gold = instance.get("patch", "") or ""
        cand_files = self._parse(patch or "")
        gold_files = self._parse(gold)

        cand_set = set(cand_files)
        gold_set = set(gold_files)

        if not gold_set:
            file_f1 = 0.0
        else:
            tp = len(cand_set & gold_set)
            precision = tp / len(cand_set) if cand_set else 0.0
            recall = tp / len(gold_set)
            file_f1 = (
                2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            )

        # Hunk Jaccard over shared files
        shared = cand_set & gold_set
        if shared:
            union_lines = 0
            inter_lines = 0
            for fname in shared:
                c = cand_files[fname]
                g = gold_files[fname]
                inter_lines += len(c & g)
                union_lines += len(c | g)
            hunk_jaccard = inter_lines / union_lines if union_lines > 0 else 0.0
        else:
            hunk_jaccard = 0.0

        detail = json.dumps({"file_f1": file_f1, "hunk_jaccard": hunk_jaccard})
        return PatchEvaluation(
            quality=file_f1,
            passed=file_f1 >= self.tau,
            detail=detail,
        )


class ReferenceJudgeEvaluator(PatchEvaluator):
    """Reference-based LLM judge using the gold patch as a reference.

    Breaks Haiku-grading-Haiku circularity by defaulting to Opus, and includes
    the gold patch in the prompt to ground the rubric. Quality is gold-reading
    by design (evaluators are explicitly allowed to read gold).
    """

    def __init__(
        self,
        judge_model: str,
        tau: float,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.judge_model = judge_model
        self.tau = tau
        self.api_base = api_base
        self.api_key = api_key

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        import litellm

        gold = instance.get("patch", "") or ""
        prompt = (
            "You are grading a proposed code patch. "
            "Return ONLY a float in [0,1].\n\n"
            "Rubric: Does the patch address the right file(s)? "
            "Does it apply the correct logic? "
            "Would it plausibly pass the targeted tests?\n\n"
            f"PROBLEM:\n{instance.get('problem_statement', '')}\n\n"
            f"GOLD REFERENCE PATCH:\n{gold}\n\n"
            f"CANDIDATE PATCH:\n{patch}\n"
        )
        proxy_kwargs: dict[str, str] = {}
        if self.api_base is not None:
            proxy_kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            proxy_kwargs["api_key"] = self.api_key
        resp = litellm.completion(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **proxy_kwargs,
        )
        text = resp.choices[0].message.content or "0"
        try:
            quality = float(text.strip().split()[0])
        except (ValueError, IndexError):
            quality = 0.0
        quality = min(1.0, max(0.0, quality))
        return PatchEvaluation(quality=quality, passed=quality >= self.tau, detail="ref-judge")


class ApplyCheckEvaluator(PatchEvaluator):
    """Binary signal: does ``git apply --check`` accept the patch?

    Clones the repo at ``base_commit`` into a per-instance checkout cache (fetch
    once, reused across rounds/signals). Quality ∈ {0, 1}.
    Requires network; gate with ``LLM_SIM_APPLY_CHECK=1``.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau
        self._checkout_cache: dict[str, Path] = {}

    def _get_checkout(self, instance: dict) -> Path:  # pragma: no cover
        key = f"{instance.get('repo', '')}@{instance.get('base_commit', '')}"
        if key not in self._checkout_cache:
            workdir = tempfile.mkdtemp(prefix="apply_check_")
            self._checkout_cache[key] = MiniSweAgentRunner._clone_repo(instance, workdir)
        return self._checkout_cache[key]

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:  # pragma: no cover
        try:
            repo_dir = self._get_checkout(instance)
            result = subprocess.run(  # noqa: S603
                ["git", "apply", "--check", "--allow-empty"],  # noqa: S607
                input=patch or "",
                capture_output=True,
                text=True,
                cwd=str(repo_dir),
            )
            applies = 1.0 if result.returncode == 0 else 0.0
        except Exception:  # noqa: BLE001
            applies = 0.0
        return PatchEvaluation(
            quality=applies,
            passed=applies >= self.tau,
            detail=f"apply-check:{'ok' if applies else 'fail'}",
        )


class CompositeEvaluator(PatchEvaluator):
    """Run a named panel of evaluators; expose primary signal as quality.

    Sub-evaluator exceptions degrade to quality=0.0 for that signal without
    aborting the round. All signal scores are serialised into ``detail`` as JSON.
    """

    def __init__(
        self,
        evaluators: list[tuple[str, PatchEvaluator]],
        primary: str,
        tau: float,
    ) -> None:
        self.evaluators = evaluators
        self.primary = primary
        self.tau = tau

    def evaluate(self, patch: str, instance: dict) -> PatchEvaluation:
        signals: dict[str, float] = {}
        for name, ev in self.evaluators:
            try:
                result = ev.evaluate(patch, instance)
                # LocalizationEvaluator stores sub-signals in its detail JSON.
                if name == "localization":
                    try:
                        sub = json.loads(result.detail)
                        signals["file_f1"] = float(sub.get("file_f1", result.quality))
                        signals["hunk_jaccard"] = float(sub.get("hunk_jaccard", 0.0))
                    except (ValueError, KeyError):
                        signals[name] = result.quality
                else:
                    signals[name] = result.quality
            except Exception:  # noqa: BLE001
                if name == "localization":
                    signals["file_f1"] = 0.0
                    signals["hunk_jaccard"] = 0.0
                else:
                    signals[name] = 0.0

        primary_q = signals.get(self.primary, 0.0)
        return PatchEvaluation(
            quality=primary_q,
            passed=primary_q >= self.tau,
            detail=json.dumps(signals),
        )


def get_evaluator(config: Config) -> PatchEvaluator:
    """Dispatch on ``config.evaluator`` -- swapping the evaluator is one env var."""
    if config.evaluator == "proxy":
        return ProxyEditDistanceEvaluator(tau=config.tau)
    if config.evaluator == "judge":
        return LLMJudgeEvaluator(
            model=config.model,
            tau=config.tau,
            api_base=config.api_base,
            api_key=config.api_key,
        )
    if config.evaluator == "harness":
        return HarnessEvaluator()
    if config.evaluator == "panel":
        subs: list[tuple[str, PatchEvaluator]] = [
            ("norm_edit_sim", NormalizedEditSimEvaluator(tau=config.tau)),
            ("localization", LocalizationEvaluator(tau=config.tau)),
            (
                "ref_judge",
                ReferenceJudgeEvaluator(
                    judge_model=config.judge_model,
                    tau=config.tau,
                    api_base=config.api_base,
                    api_key=config.api_key,
                ),
            ),
        ]
        if config.apply_check:
            subs.append(("apply_check", ApplyCheckEvaluator(tau=config.tau)))
        return CompositeEvaluator(evaluators=subs, primary=config.panel_primary, tau=config.tau)
    raise ValueError(f"Unknown evaluator: {config.evaluator!r}")


# ============================================================
# 2. Leakage guard
# ============================================================


class LeakageError(RuntimeError):
    """Raised when a prompt about to be sent to the agent contains gold-patch
    content -- a construction-time guarantee that the agent cannot cheat."""


#  Bare single-keyword statement lines ("+    pass", "-    return", ...) carry no
# information about the actual fix -- they're near-universal Python boilerplate (a
# removed/added no-op stub, an early return, a loop-control statement) and are
# virtually guaranteed to coincidentally recur elsewhere in a large document (seen
# live: a gold "pass" line matched inside a bracketed parametrized test ID like
# "test_foo[pass]", where the brackets create word boundaries on both sides).
# Excluding them outright is more robust than chasing every possible collision a
# generic short token could have with unrelated text.
_TRIVIAL_GOLD_KEYWORDS = frozenset({"pass", "break", "continue", "else:", "try:", "finally:"})


def _gold_signal_lines(gold: str) -> list[str]:
    """Distinctive lines of the gold diff whose appearance in a prompt would leak
    the answer: the added ('+') and removed ('-') content lines, minus diff noise."""
    signals: list[str] = []
    for raw in gold.splitlines():
        line = raw.strip()
        if len(line) < 4:
            continue
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if line.startswith(("+", "-")):
            body = line[1:].strip()
            if len(body) >= 4 and body not in _TRIVIAL_GOLD_KEYWORDS:
                signals.append(body)
    return signals


def _contains_signal(text: str, signal: str) -> bool:
    """Whole-token containment: a naive substring check false-positives on any short
    common signal that happens to prefix an unrelated word -- seen live, a gold
    ``pass`` placeholder line matched inside our own critique text's "...tests
    passed.", which is not a leak. A ``\\b`` anchor is only added on a side whose
    edge character is itself a word character: many gold lines end in punctuation
    (``compute_secret_fix()``), where ``\\b`` between two non-word characters never
    matches and would produce a false NEGATIVE (missing a real leak) instead."""
    prefix = r"\b" if re.match(r"\w", signal[0]) else ""
    suffix = r"\b" if re.match(r"\w", signal[-1]) else ""
    return re.search(f"{prefix}{re.escape(signal)}{suffix}", text) is not None


def assert_no_leakage(prompt: str, instance: dict) -> None:
    """Fail closed if ``prompt`` contains the gold diff or any distinctive gold
    content line. Called on every prompt before it reaches the agent."""
    gold = instance.get("patch", "") or ""
    if not gold:
        return
    if gold.strip() and gold.strip() in prompt:
        raise LeakageError(f"Gold patch leaked into prompt for {instance.get('instance_id')}")
    # A gold line that ALSO appears in the (public) problem statement is not a leak
    # the harness can introduce: the agent receives that statement verbatim as its
    # task input (SWE-bench statements routinely quote repros, imports and tracebacks
    # that coincide with the fix). We only flag distinctive gold content that is NOT
    # already public -- i.e. content that could only have entered the prompt via
    # harness-side text such as the critique.
    public = instance.get("problem_statement", "") or ""
    for signal in _gold_signal_lines(gold):
        if _contains_signal(public, signal):
            continue
        if _contains_signal(prompt, signal):
            raise LeakageError(
                f"Gold content line leaked into prompt for {instance.get('instance_id')}: "
                f"{signal!r}"
            )


# ============================================================
# 3. Response cache (the reproducibility contract)
# ============================================================


def prompt_cache_key(
    model: str,
    instance_id: str,
    round_idx: int,
    prompt: str,
    seed: int = 0,
    temperature: float = 0.0,
) -> str:
    """SHA-256 over ``(model, instance_id, round, [seed,] [temperature,] prompt)``.

    ``seed`` partitions each replicate to a distinct cache slot so distinct seeds
    force independent live model samples rather than replaying the same cached
    response. ``seed=0`` omits the seed field entirely so existing cache entries
    (written before seed partitioning was added) remain valid.

    ``temperature`` follows the same back-compat pattern: ``temperature=0.0`` omits
    the field so pinned cache entries written before the temperature knob existed
    (all generated at the old hardcoded temperature=0) stay valid; any other value
    (e.g. the new default 0.25) is hashed in, forcing fresh live samples rather than
    replaying a temp-0 response under a temp-0.25 key.
    """
    h = hashlib.sha256()
    prefix = f"{model}\x00{instance_id}\x00{round_idx}\x00"
    if seed != 0:
        prefix += f"{seed}\x00"
    if temperature != 0.0:
        prefix += f"{temperature:.4g}\x00"
    h.update(prefix.encode())
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via a unique tmp file + ``os.replace``.

    ``os.replace`` is atomic on POSIX, so a process killed mid-write (e.g. a
    Ctrl-C after credits die) can never leave a torn/partial ``path`` behind. The
    tmp suffix includes pid+thread-id so concurrent writers (Phase B) never
    clobber each other's tmp file even when writing the same final ``path``.
    """
    tmp = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


class ResponseCache:
    """A tiny content-addressed cache of model responses on disk."""

    def __init__(self, cache_dir: str | os.PathLike[str]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())["response"]
        except (ValueError, OSError, KeyError):
            # A torn file from a process killed mid-write is a miss, not a
            # permanent poison pill: unlink it so the next put() rewrites it clean.
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, response: str) -> None:
        _atomic_write_json(self._path(key), {"response": response})


def eval_cache_key(
    instance_id: str,
    round_idx: int,
    seed: int,
    candidate_patch: str,
    evaluator_signature: str,
) -> str:
    """SHA-256 over ``(instance_id, round, seed, sha256(candidate_patch),
    evaluator_signature)`` -- mirrors ``prompt_cache_key``'s shape.

    Keyed on the candidate's content hash, not just ``round_idx``, because a
    generation-cache invalidation can change which patch a given round actually
    produced; the cached evaluation must match the *actual* patch that was scored.
    ``evaluator_signature`` (evaluator name + backend + tau) ensures switching
    evaluator/backend/threshold never returns a stale hit.
    """
    candidate_digest = hashlib.sha256(candidate_patch.encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    h.update(
        f"{instance_id}\x00{round_idx}\x00{seed}\x00{candidate_digest}\x00"
        f"{evaluator_signature}".encode()
    )
    return h.hexdigest()


def _evaluator_signature(config: Config, evaluator: PatchEvaluator) -> str:
    """``config.evaluator`` + backend + tau, for ``eval_cache_key``.

    ``backend`` distinguishes ``HarnessEvaluator``'s local vs Modal execution
    (only that evaluator carries a ``backend`` attribute; others default to "").

    ``:v2`` invalidates every eval cached under the pre-fix ``HarnessEvaluator``
    (--no-header/ANSI text-scraping + un-corrected stale parametrize IDs) without
    touching the separate agent-generation cache (``prompt_cache_key``) -- so a
    re-run replays already-generated patches for free and only re-evaluates.
    Bump again if ``HarnessEvaluator``'s eval semantics change again.
    """
    backend = getattr(evaluator, "backend", "")
    return f"{config.evaluator}:{backend}:{config.tau}:v2"


class EvalCache:
    """Content-addressed cache of ``PatchEvaluation`` results on disk.

    Makes evaluation cost (Modal Sandbox boot + pytest) resumable: a re-run that
    hits the same ``eval_cache_key`` skips ``evaluator.evaluate`` entirely instead
    of re-paying for a real harness execution.
    """

    def __init__(self, cache_dir: str | os.PathLike[str]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> PatchEvaluation | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return PatchEvaluation(**data)
        except (ValueError, OSError, KeyError, TypeError):
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, evaluation: PatchEvaluation) -> None:
        _atomic_write_json(self._path(key), asdict(evaluation))


# ============================================================
# 4. Agent adapter seam (the weak learner)
# ============================================================


class AgentRunner(ABC):
    """Produces a candidate patch (unified diff) for one refinement round.

    Implementations must NOT read gold (``instance['patch']``); they see only the
    problem statement, the prior patch, and a critique derived from non-gold signals.
    """

    @abstractmethod
    def run(
        self,
        instance: dict,
        round_idx: int,
        prior_patch: str,
        critique: str,
        seed: int = 0,
        model: str | None = None,
    ) -> str:
        raise NotImplementedError

    def prepare(self, instances: list[dict]) -> None:  # noqa: ARG002, B027
        """Called once by run_experiment before any rounds run, single-threaded.

        Default no-op. ``MiniSweAgentRunner`` overrides this to fetch each unique
        task's repo exactly once (see ``_get_or_create_template``), so opt-in
        concurrency never races on the initial GitHub fetch.
        """

    def teardown(self, instance_id: str, seed: int = 0) -> None:  # noqa: ARG002, B027
        """Called by run_experiment after one ``(instance_id, seed)`` attempt
        completes. Default no-op; mirrors ``PatchEvaluator.teardown``'s shape.
        ``MiniSweAgentRunner`` overrides this to remove that attempt's local
        working clone.
        """

    def cleanup(self) -> None:  # noqa: B027
        """Called once by run_experiment after all work completes.

        Default no-op. ``MiniSweAgentRunner`` overrides this to remove every
        cloned template (disk hygiene across a multi-day, many-repo sweep).
        """


def harness_injected_context(instance: dict, critique: str) -> str:
    """The parts of a prompt that the *harness* controls (problem statement +
    critique). This is what the leakage guard must vet -- NOT the agent's own prior
    patch, which may legitimately resemble gold as the agent converges."""
    return f"{instance.get('problem_statement', '')}\n{critique}"


def build_agent_prompt(instance: dict, round_idx: int, prior_patch: str, critique: str) -> str:
    """Compose the task prompt for a refinement round.

    The prior patch is the agent's *own* previous output (Reflexion-style feedback);
    only the harness-injected context is leakage-checked (see the callers)."""
    parts = [
        "You are resolving a software issue by producing a unified git diff.",
        f"\n## Problem statement\n{instance.get('problem_statement', '')}",
    ]
    if round_idx > 0 and prior_patch:
        parts.append(f"\n## Your previous attempt (patch)\n{prior_patch}")
        if critique:
            parts.append(f"\n## Critique of the previous attempt\n{critique}")
            parts.append("\nRevise your patch to address the critique.")
        else:
            parts.append("\nRevise your patch to improve it.")
    return "\n".join(parts)


def _demo_target(instance_id: str) -> str:
    """A deterministic, PUBLIC pseudo-diff derived from the (public) ``instance_id``.

    Used only by the offline illustrative demo (``StubAgentRunner`` +
    ``make_demo_instances``). It is *not* the gold ``patch`` field, so the stub can
    converge toward it without reading gold -- the leakage guard remains honest.
    """
    seed = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    body = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz \n"), size=240))
    return f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@\n+{body}\n"


class StubAgentRunner(AgentRunner):
    """Offline, deterministic agent for tests and for producing the committed
    *illustrative* figures without network access or credentials.

    It emits an increasingly complete prefix of a target diff derived from the
    (public) ``instance_id`` -- so quality rises across rounds with diminishing
    returns, exercising the full agent -> evaluator -> metrics -> plots path. It
    NEVER reads the gold ``patch`` field; the target is recomputed from the public
    id, so the leakage guard stays honest and any resemblance to real gold is
    coincidental. This is a faithful test of the *machinery*, not of solve rates.
    """

    def run(
        self,
        instance: dict,
        round_idx: int,
        prior_patch: str,
        critique: str,
        seed: int = 0,
        model: str | None = None,  # noqa: ARG002 -- stub is model-agnostic by design
    ) -> str:
        assert_no_leakage(harness_injected_context(instance, critique), instance)
        target = _demo_target(instance.get("instance_id", ""))
        # Concave (geometric) reveal schedule: the fraction of the target revealed
        # approaches 1 with a shrinking gap, so per-round gains diminish -- the
        # saturation the refinement theorem predicts (Thm. refinement).
        frac = 1.0 - (1.0 - 0.35) * (0.55**round_idx)
        cut = max(1, int(len(target) * frac))
        return target[:cut]


class MiniSweAgentRunner(AgentRunner):
    """Real weak learner: `mini-swe-agent` (litellm on Bedrock) editing a local
    clone of ``repo`` at ``base_commit``. Round 0 runs on the problem statement;
    later rounds are seeded with the prior patch + critique (Reflexion-style).

    Heavy, network- and credential-bound; lazy-imported and excluded from coverage.
    """

    def __init__(self, config: Config, cache: ResponseCache) -> None:
        self.config = config
        self.cache = cache
        # One real GitHub fetch per unique task (populated by prepare(), before any
        # concurrency starts -- see AgentRunner.prepare).
        self._repo_templates: dict[str, Path] = {}
        # Per-(instance_id, seed) local working clone -- cheap, no network (hardlinks
        # the template's objects). _clone_lock is defensive: in the seed-outer
        # concurrency shape each key is touched by exactly one worker thread, but
        # locking is cheap insurance against future interleaving.
        self._clone_dirs: dict[tuple[str, int], Path] = {}
        self._clone_lock = threading.Lock()

    def run(
        self,
        instance: dict,
        round_idx: int,
        prior_patch: str,
        critique: str,
        seed: int = 0,
        model: str | None = None,
    ) -> str:  # pragma: no cover
        assert_no_leakage(harness_injected_context(instance, critique), instance)
        resolved_model = model or self.config.model
        prompt = build_agent_prompt(instance, round_idx, prior_patch, critique)
        key = prompt_cache_key(
            resolved_model,
            instance["instance_id"],
            round_idx,
            prompt,
            seed,
            self.config.temperature,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        patch = self._run_agent(instance, prompt, resolved_model, seed)
        self.cache.put(key, patch)
        return patch

    def prepare(self, instances: list[dict]) -> None:  # pragma: no cover
        """Clone each unique task's repo exactly once, upfront, single-threaded."""
        for instance in instances:
            self._get_or_create_template(instance)

    def _get_or_create_template(self, instance: dict) -> Path:  # pragma: no cover
        instance_id = str(instance["instance_id"])
        template = self._repo_templates.get(instance_id)
        if template is not None:
            return template
        template = self._clone_repo(instance, tempfile.mkdtemp(prefix="llm_sim_template_"))
        self._repo_templates[instance_id] = template
        return template

    def _get_or_create_clone(self, instance: dict, seed: int) -> Path:  # pragma: no cover
        """Return this ``(instance_id, seed)``'s local working clone, creating it
        via a network-free ``git clone --local`` off the cached template on first
        use."""
        instance_id = str(instance["instance_id"])
        key = (instance_id, seed)
        with self._clone_lock:
            clone_dir = self._clone_dirs.get(key)
            if clone_dir is not None:
                return clone_dir
            template = self._get_or_create_template(instance)
            clone_dir = Path(tempfile.mkdtemp(prefix="llm_sim_clone_")) / "repo"
            subprocess.run(  # noqa: S603
                ["git", "clone", "--quiet", "--local", str(template), str(clone_dir)],  # noqa: S607
                check=True,
            )
            self._clone_dirs[key] = clone_dir
            return clone_dir

    def teardown(self, instance_id: str, seed: int = 0) -> None:  # pragma: no cover
        with self._clone_lock:
            clone_dir = self._clone_dirs.pop((instance_id, seed), None)
        if clone_dir is not None:
            shutil.rmtree(clone_dir.parent, ignore_errors=True)

    def cleanup(self) -> None:  # pragma: no cover
        for template in self._repo_templates.values():
            shutil.rmtree(template.parent, ignore_errors=True)
        self._repo_templates.clear()

    def _run_agent(
        self, instance: dict, task: str, model: str, seed: int
    ) -> str:  # pragma: no cover
        import yaml
        from minisweagent import package_dir
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model

        repo_dir = self._get_or_create_clone(instance, seed)
        # The clone persists across this task-seed's T rounds (unlike the old
        # per-round TemporaryDirectory), so reset the working tree before each
        # round's edits -- the agent only ever edits the tree, never commits (see
        # _git_diff), so a checkout+clean is a full, safe reset.
        subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "checkout", "--quiet", "--", "."],  # noqa: S607
            check=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "clean", "-fdq"],  # noqa: S607
            check=True,
        )
        cfg_path = package_dir / "config" / "benchmarks" / "swebench.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        model_obj = get_model(
            config={
                **cfg["model"],
                "model_name": model,
                # Splat proxy routing (api_base/api_key) into model_kwargs so
                # mini-swe-agent's litellm call hits the configured proxy, not AWS.
                # ``_skip_mcp_handler`` bypasses litellm's MCP tool-bridge import
                # path (needs fastapi, unused here) which otherwise fires whenever
                # the agent's BASH_TOOL is present in the completion call.
                # ``tool_choice="auto"`` is required for the proxy's Bedrock
                # backend: without it the tool-calling request is rejected
                # ("tool_choice.type: Field required"); "auto" lets the agent
                # choose between calling bash and submitting.
                "model_kwargs": {
                    **cfg["model"].get("model_kwargs", {}),
                    "temperature": self.config.temperature,
                    "_skip_mcp_handler": True,
                    "tool_choice": "auto",
                    **_litellm_kwargs(self.config),
                },
                "cost_tracking": "ignore_errors",
            }
        )
        # Build a MINIMAL local environment config. The stock swebench.yaml
        # environment block targets the SWE-bench Docker image (cwd /testbed,
        # BASH_ENV=/root/.bashrc, `conda activate testbed`); splatting it into a
        # LocalEnvironment would point at paths that do not exist here. We keep
        # only the fields a local shell needs.
        env = get_environment({"environment_class": "local", "cwd": str(repo_dir), "timeout": 60})
        # cost_limit=0 DISABLES the cost gate (guard is `0 < cost_limit <= cost`);
        # combined with cost_tracking="ignore_errors" (Bedrock cost calc is
        # unreliable), the run is bounded by the config's step_limit instead.
        agent = DefaultAgent(
            model_obj,
            env,
            **{**cfg["agent"], "cost_limit": 0, "step_limit": self.config.step_limit},
        )
        result = agent.run(task)
        if result.get("exit_status") == "Submitted" and result.get("submission"):
            return str(result["submission"])
        return self._git_diff(repo_dir)

    @staticmethod
    def _clone_repo(instance: dict, workdir: str) -> Path:  # pragma: no cover
        repo_dir = Path(workdir) / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{instance['repo']}.git"
        base = instance["base_commit"]
        # Shallow-fetch ONLY the base commit (GitHub allows fetching an arbitrary
        # SHA). No other history is downloaded, so the gold-fix commit is never
        # present on disk -- the filesystem leakage guard holds by construction,
        # large repos fetch in seconds, and there is no full-clone ``.git`` to
        # delete (avoiding the rmtree/background-gc race). ``git diff`` still
        # captures the agent's edits against the (detached) base HEAD.
        steps = (
            ["init", "--quiet"],
            ["remote", "add", "origin", url],
            ["fetch", "--depth", "1", "--quiet", "origin", base],
            ["checkout", "--quiet", "FETCH_HEAD"],
        )
        for args in steps:
            subprocess.run(  # noqa: S603
                ["git", "-C", str(repo_dir), *args],  # noqa: S607
                check=True,
            )
        return repo_dir

    @staticmethod
    def _git_diff(repo_dir: Path) -> str:  # pragma: no cover
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "diff"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout


def get_agent_runner(config: Config, cache: ResponseCache | None = None) -> AgentRunner:
    """Dispatch on ``config.agent`` -- the weak learner is swappable behind this."""
    if config.agent == "stub":
        return StubAgentRunner()
    if config.agent == "mini":
        return MiniSweAgentRunner(config, cache or ResponseCache(_cache_dir(config)))
    raise ValueError(f"Unknown agent: {config.agent!r}")


def _cache_dir(config: Config) -> str:
    return config.cache_dir or os.path.join(DATADIR, "llm_cache")


def _eval_cache_dir(config: Config) -> str:
    return os.path.join(_cache_dir(config), "eval")


# ============================================================
# 5. Data loading
# ============================================================


def select_diverse_tasks(
    rows: list[dict], k: int, exclude_ids: frozenset[str] = frozenset()
) -> list[dict]:
    """Pick ``k`` tasks spread across repositories, deterministically.

    First pass: one lowest-instance-id task per repo (repos in that id order).
    If k > n_repos, additional passes round-robin across the same ordered repos,
    taking the next lowest-id task per repo each pass.
    This preserves the first min(k, n_repos) picks unchanged (cache reuse intact).

    ``exclude_ids`` drops denylisted instance_ids (e.g. tasks confirmed broken
    at the harness/infrastructure level, not a model-quality signal) before
    selection, so the round-robin naturally fills the k slots from the
    remaining pool instead.
    """
    if exclude_ids:
        rows = [r for r in rows if r["instance_id"] not in exclude_ids]
    by_id = sorted(rows, key=lambda r: r["instance_id"])

    # Build per-repo task lists (ordered by instance_id within each repo)
    repo_tasks: dict[str, list[dict]] = {}
    repo_order: list[str] = []
    for row in by_id:
        repo = row.get("repo", "")
        if repo not in repo_tasks:
            repo_tasks[repo] = []
            repo_order.append(repo)
        repo_tasks[repo].append(row)

    picked: list[dict] = []
    pass_idx = 0
    while len(picked) < k:
        added_this_pass = False
        for repo in repo_order:
            if len(picked) >= k:
                break
            tasks = repo_tasks[repo]
            if pass_idx < len(tasks):
                picked.append(tasks[pass_idx])
                added_this_pass = True
        if not added_this_pass:
            break  # all repos exhausted
        pass_idx += 1

    return picked


def load_swe_tasks(
    k: int, exclude_ids: frozenset[str] = frozenset()
) -> list[dict]:  # pragma: no cover
    """Load ``k`` SWE-bench Lite test tasks from distinct repositories,
    deterministically. Network-bound; injected/monkeypatched in tests."""
    import datasets

    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return [dict(r) for r in select_diverse_tasks([dict(x) for x in ds], k, exclude_ids)]


def make_demo_instances(k: int) -> list[dict]:
    """Synthetic tasks for the offline illustrative run (``LLM_SIM_AGENT=stub``).

    Each task's gold ``patch`` is the full ``_demo_target(instance_id)`` that the
    stub agent converges toward -- a self-contained simulation (analogous to the
    stylized experiments in ``simulations.py``), NOT measured LLM output.
    """
    instances = []
    for i in range(k):
        instance_id = f"demo__task-{i:03d}"
        instances.append(
            {
                "instance_id": instance_id,
                "problem_statement": f"Synthetic offline demo task {i}.",
                "repo": "demo/repo",
                "base_commit": "0" * 40,
                "patch": _demo_target(instance_id),
            }
        )
    return instances


# ============================================================
# 6. Orchestration
# ============================================================


def _make_critique(patch: str) -> str:
    """Build a leakage-free critique from AGENT-SIDE signals only.

    Critically, this must NOT depend on the evaluator's output -- not even the
    pass/fail bit -- because ``passed`` is a thresholded function of the gold
    similarity, and surfacing it would leak one bit of the (forbidden) reward signal
    to the agent every round. The critique therefore keys only on properties of the
    agent's own patch (did it produce a usable diff?), which the agent already knows.
    """
    if not patch.strip():
        return (
            "Your previous attempt produced no usable diff. Produce a concrete unified "
            "diff that addresses the problem statement."
        )
    if not (patch.lstrip().startswith("diff ") or "@@" in patch):
        return (
            "Your previous output did not look like a unified diff. Re-express your "
            "change as a valid patch and reconsider whether it fully addresses the issue."
        )
    return (
        "Review your previous patch against the problem statement: check for missed "
        "cases, incorrect edits, or incomplete changes, and refine it."
    )


def assign_tier_model(config: Config, instance_id: str, seed: int) -> str:
    """Pick the model that runs one (task, seed) attempt.

    ``"fixed"``: always ``config.model`` (the current/pre-Phase-2 behavior).
    ``"randomized"``: pick uniformly from ``config.tier_models`` (or ``(config.model,)``
    if unset) via a hash of ``(instance_id, seed)`` -- deterministic and reproducible
    across re-runs, but decorrelated from task content, which is what breaks the
    task-type <-> tier confound in the capability-gap analysis.
    """
    if config.tier_policy == "fixed":
        return config.model
    if config.tier_policy == "randomized":
        models = config.tier_models or (config.model,)
        digest = hashlib.sha256(f"{instance_id}\x00{seed}".encode()).hexdigest()
        idx = int(digest, 16) % len(models)
        return models[idx]
    raise ValueError(f"Unknown tier_policy: {config.tier_policy!r}")


def _evaluate_cached(
    eval_cache: EvalCache | None,
    config: Config,
    evaluator: PatchEvaluator,
    instance: dict,
    patch: str,
    round_idx: int,
    seed: int,
) -> PatchEvaluation:
    """Evaluate ``patch``, consulting ``eval_cache`` first when one is given.

    ``eval_cache=None`` (the default for every call site except ``run_experiment``)
    disables caching entirely -- ``evaluator.evaluate`` runs every time, exactly as
    before this cache existed. This keeps every pre-existing test (which passes
    fake evaluators and never an eval_cache) byte-identical, and avoids fake
    evaluators of different tests colliding on the same on-disk cache key.
    """
    if eval_cache is None:
        return evaluator.evaluate(patch, instance)
    key = eval_cache_key(
        str(instance.get("instance_id", "")),
        round_idx,
        seed,
        patch,
        _evaluator_signature(config, evaluator),
    )
    cached = eval_cache.get(key)
    if cached is not None:
        return cached
    evaluation = evaluator.evaluate(patch, instance)
    eval_cache.put(key, evaluation)
    return evaluation


def run_task(
    instance: dict,
    config: Config,
    runner: AgentRunner,
    evaluator: PatchEvaluator,
    eval_cache: EvalCache | None = None,
    seed: int = 0,
) -> list[dict]:
    """Run the ``T``-round refinement loop for one task, returning per-round records.

    The agent sees only the problem statement, its prior patch, and a non-gold
    critique. Only ``evaluator`` reads gold.

    ``config.feedback_mode`` controls what carries over between rounds; each mode
    still loops ``config.t`` times (unless it solves early), so the three ablation
    arms spend an identical sampling budget per task -- only what the weak learner
    sees differs. ``diagnostic`` requires ``config.temperature > 0`` to be a
    meaningful ablation against ``independent``: at temperature 0 every round would
    be an identical greedy draw.
    """
    if config.feedback_mode not in ("independent", "blind", "diagnostic"):
        raise ValueError(f"Unknown feedback_mode: {config.feedback_mode!r}")

    model = assign_tier_model(config, str(instance.get("instance_id", "")), seed)

    records: list[dict] = []
    prior_patch = ""
    critique = ""
    best_patch = ""
    best_quality = -1.0
    # Sentinel: any real evaluation has quality >= 0 > -1, so this is always replaced.
    best_evaluation: PatchEvaluation = PatchEvaluation(quality=-1.0, passed=False)
    solved_round: int | None = None

    for round_idx in range(config.t):
        tag = f"[seed={seed} {instance.get('instance_id')}] round {round_idx + 1}/{config.t}"
        print(f"    {tag}: generating patch...", flush=True)
        patch = runner.run(instance, round_idx, prior_patch, critique, seed=seed, model=model)
        print(f"    {tag}: evaluating...", flush=True)
        evaluation = _evaluate_cached(
            eval_cache, config, evaluator, instance, patch, round_idx, seed
        )
        try:
            signals: dict[str, float] = json.loads(evaluation.detail)
            if not isinstance(signals, dict):
                raise ValueError
        except (ValueError, TypeError):
            signals = {"primary": evaluation.quality}

        if evaluation.quality > best_quality:
            best_quality = evaluation.quality
            best_patch = patch
            best_evaluation = evaluation

        records.append(
            {
                "instance_id": instance.get("instance_id"),
                "round": round_idx,
                "model": model,
                "quality": evaluation.quality,
                "fraction_passing": evaluation.fraction_passing,
                "passed": evaluation.passed,
                "detail": evaluation.detail,
                "signals": signals,
                "padded": False,
                "solved_round": None,  # back-filled after loop
            }
        )

        if evaluation.passed:
            solved_round = round_idx
            # Pad remaining rounds so the T-wide quality matrix stays rectangular.
            # A solved task contributes zero further edge — padded rounds show η_t → 0.
            try:
                pad_signals: dict[str, float] = json.loads(best_evaluation.detail)
                if not isinstance(pad_signals, dict):
                    raise ValueError
            except (ValueError, TypeError):
                pad_signals = {"primary": 1.0}
            for pad_idx in range(round_idx + 1, config.t):
                records.append(
                    {
                        "instance_id": instance.get("instance_id"),
                        "round": pad_idx,
                        "model": model,
                        "quality": 1.0,
                        "fraction_passing": 1.0,
                        "passed": True,
                        "detail": best_evaluation.detail,
                        "signals": pad_signals,
                        "padded": True,
                        "solved_round": solved_round,
                    }
                )
            break

        # Non-harness evaluators key only on the agent's own patch (no gold data).
        # HarnessEvaluator adds test results (safe: pytest output ≠ gold-patch content).
        # Feed the best patch seen so far — a regressed round must not erase a correct solution.
        if config.feedback_mode == "diagnostic":
            critique = evaluator.make_critique(best_patch, best_evaluation, instance)
            prior_patch = best_patch
        elif config.feedback_mode == "blind":
            critique = ""
            prior_patch = best_patch
        else:  # independent
            critique = ""
            prior_patch = ""

    for rec in records:
        if not rec.get("padded", False):
            rec["solved_round"] = solved_round

    return records


def _is_budget_error(exc: Exception) -> bool:
    """True if ``exc`` indicates the LiteLLM proxy's daily spend limit is exhausted.

    Checks three layers in order of specificity:
    1. ``litellm.BudgetExceededError`` -- the canonical class (status 429).
    2. ``litellm.RateLimitError`` with "budget" in the message -- the proxy may
       surface budget exhaustion through the rate-limit path.
    3. String-based fallback for wrapped exceptions propagating through
       mini-swe-agent's own try/except layers.
    """
    try:
        import litellm as _litellm

        if isinstance(exc, _litellm.BudgetExceededError):
            return True
        if isinstance(exc, _litellm.RateLimitError) and "budget" in str(exc).lower():
            return True
    except ImportError:
        pass
    combined = (str(exc) + str(getattr(exc, "__cause__", "") or "")).lower()
    return "budget" in combined and any(w in combined for w in ("exceed", "exhaust", "limit"))


def _is_infra_budget_error(exc: Exception) -> bool:
    """True if ``exc`` looks like Modal credit/quota exhaustion.

    Best-effort keyword match over the exception chain: as of writing, Modal does
    not expose a canonical billing-exhaustion exception class to match on (unlike
    ``litellm.BudgetExceededError``), so this deliberately fails toward STOPPING --
    the recoverable direction (a clean ``budget_stopped`` halt + resume message) --
    rather than letting an unrecognized global-outage error fall through to the
    per-instance ``SKIPPED``/``continue`` path, which would silently truncate the
    q-matrix with no infra-failure signal.

    Must NOT match the pre-existing per-instance ``test_patch``-apply-failure
    ``RuntimeError`` raised by ``HarnessEvaluator.evaluate`` (task-specific
    infrastructure, not a global outage): that message carries none of these
    billing keywords.
    """
    combined = (str(exc) + str(getattr(exc, "__cause__", "") or "")).lower()
    keywords = ("out of credits", "quota", "insufficient", "payment required", "402")
    return any(kw in combined for kw in keywords)


def _run_experiment_sequential(
    instances: list[dict],
    config: Config,
    runner: AgentRunner,
    evaluator: PatchEvaluator,
    eval_cache: EvalCache | None,
) -> tuple[list[list[dict]], list[dict], bool]:
    """The pre-Phase-B sequential loop (``concurrency <= 1`` takes this exact path,
    byte-identical). A task whose agent raises is SKIPPED (logged), not fatal -- the
    real-repo agent is fragile, and the SHA cache lets a re-run resume completed
    rounds -- so a single crash never discards the rest of the experiment.

    Budget exhaustion is treated differently: the loop stops immediately and the
    partial results are returned. The SHA-keyed response cache means the next
    invocation with the same arguments resumes from the last completed round, so
    no work is lost."""
    all_ok_records: list[list[dict]] = []
    all_ok_instances: list[dict] = []
    budget_stopped = False

    seeds = [config.seed_base + s for s in range(config.n_seeds)]
    for seed_idx, seed in enumerate(seeds):
        if config.n_seeds > 1:
            print(f"[seed {seed_idx + 1}/{config.n_seeds}  seed_val={seed}]", flush=True)
        for i, instance in enumerate(instances):
            print(f"[task {i + 1}/{len(instances)}] {instance.get('instance_id')}", flush=True)
            try:
                records = run_task(instance, config, runner, evaluator, eval_cache, seed=seed)
            except Exception as exc:  # noqa: BLE001 -- isolate one fragile task from the run
                if _is_budget_error(exc) or _is_infra_budget_error(exc):
                    print(
                        f"    BUDGET EXHAUSTED ({type(exc).__name__}: {str(exc)[:120]})",
                        flush=True,
                    )
                    print(
                        "    Stopping run. Re-run with the same command to resume from cache.",
                        flush=True,
                    )
                    budget_stopped = True
                    break
                print(f"    SKIPPED ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
                continue
            finally:
                evaluator.teardown(instance.get("instance_id", ""))
                runner.teardown(instance.get("instance_id", ""), seed)
            all_ok_records.append(records)
            all_ok_instances.append(instance)
            q_vals = [round(r["quality"], 3) for r in records]
            fp_vals = [round(r["fraction_passing"], 3) for r in records]
            print(f"    quality by round:           {q_vals}", flush=True)
            print(f"    fraction_passing by round:  {fp_vals}", flush=True)
        if budget_stopped:
            break

    return all_ok_records, all_ok_instances, budget_stopped


def _run_one_task_concurrent(
    i: int,
    instance: dict,
    config: Config,
    runner: AgentRunner,
    evaluator: PatchEvaluator,
    eval_cache: EvalCache | None,
    seed: int,
    seed_idx: int,
    stop_event: threading.Event,
) -> tuple[int, dict, list[dict] | None, bool]:
    """Run one ``(seed, task)`` attempt for the concurrent path.

    Returns ``(i, instance, records, tripped_budget_stop)``; ``records`` is
    ``None`` for a skipped, not-yet-started-after-stop, or budget-stopped task.
    """
    instance_id = str(instance.get("instance_id", ""))
    if stop_event.is_set():
        return i, instance, None, False
    print(f"[seed={seed_idx + 1} {instance_id}] starting", flush=True)
    try:
        records = run_task(instance, config, runner, evaluator, eval_cache, seed=seed)
    except Exception as exc:  # noqa: BLE001 -- isolate one fragile task from the run
        if _is_budget_error(exc) or _is_infra_budget_error(exc):
            print(
                f"    BUDGET EXHAUSTED ({type(exc).__name__}: {str(exc)[:120]})",
                flush=True,
            )
            stop_event.set()
            return i, instance, None, True
        print(f"    SKIPPED ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return i, instance, None, False
    finally:
        evaluator.teardown(instance_id)
        runner.teardown(instance_id, seed)
    q_vals = [round(r["quality"], 3) for r in records]
    fp_vals = [round(r["fraction_passing"], 3) for r in records]
    print(f"[seed={seed_idx + 1} {instance_id}] quality by round:           {q_vals}", flush=True)
    print(f"[seed={seed_idx + 1} {instance_id}] fraction_passing by round:  {fp_vals}", flush=True)
    return i, instance, records, False


def _run_experiment_concurrent(
    instances: list[dict],
    config: Config,
    runner: AgentRunner,
    evaluator: PatchEvaluator,
    eval_cache: EvalCache | None,
) -> tuple[list[list[dict]], list[dict], bool]:
    """Seed-outer, task-inner concurrency: seeds run sequentially; within a seed,
    up to ``config.concurrency`` tasks run at once via a thread pool.

    Seed-outer makes cross-thread Modal-sandbox / clone-cache collisions
    structurally impossible (within a seed, each ``instance_id`` is touched by
    exactly one worker thread). On a budget/infra-budget error, in-flight workers
    finish and not-yet-started ones skip immediately (bounded overrun: at most
    ``concurrency`` task-seeds complete after a stop is detected); results are
    resorted back into ``(seed_idx, i)`` order before returning.
    """
    seeds = [config.seed_base + s for s in range(config.n_seeds)]
    ordered_results: list[tuple[dict, list[dict] | None]] = []
    budget_stopped = False

    for seed_idx, seed in enumerate(seeds):
        if config.n_seeds > 1:
            print(f"[seed {seed_idx + 1}/{config.n_seeds}  seed_val={seed}]", flush=True)
        stop_event = threading.Event()
        seed_results: list[tuple[int, dict, list[dict] | None, bool]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            next_idx = 0
            pending: dict[concurrent.futures.Future, int] = {}

            def _submit_next(
                seed: int = seed,
                seed_idx: int = seed_idx,
                stop_event: threading.Event = stop_event,
                pending: dict[concurrent.futures.Future, int] = pending,
            ) -> None:
                nonlocal next_idx
                if next_idx >= len(instances) or stop_event.is_set():
                    return
                i = next_idx
                fut = pool.submit(
                    _run_one_task_concurrent,
                    i,
                    instances[i],
                    config,
                    runner,
                    evaluator,
                    eval_cache,
                    seed,
                    seed_idx,
                    stop_event,
                )
                pending[fut] = i
                next_idx += 1

            # Bulk-submitting all `len(instances)` futures upfront let GIL scheduling
            # race far more than `concurrency` fast (stub-agent-speed) tasks to
            # completion before a failing task's exception ever set stop_event --
            # the checked-at-body-start guard in `_run_one_task_concurrent` only
            # helps a task that hasn't started yet, and bulk submission starts nearly
            # all of them immediately. Keeping at most `concurrency` futures pending
            # and re-checking `stop_event` at each submission decision (only once a
            # slot actually frees up) bounds the overrun for real.
            for _ in range(config.concurrency):
                _submit_next()
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    del pending[fut]
                    seed_results.append(fut.result())
                    _submit_next()
        seed_results.sort(key=lambda r: r[0])
        ordered_results.extend(
            (instance, records) for _i, instance, records, _tripped in seed_results
        )
        if stop_event.is_set():
            budget_stopped = True
            break

    all_ok_records: list[list[dict]] = []
    all_ok_instances: list[dict] = []
    for instance, records in ordered_results:
        if records is not None:
            all_ok_records.append(records)
            all_ok_instances.append(instance)

    return all_ok_records, all_ok_instances, budget_stopped


def run_experiment(
    instances: list[dict], config: Config
) -> tuple[np.ndarray, list[dict], dict[str, list[list[float]]], bool, list[int | None], list[str]]:
    """Run the tasks and return the quality matrix ``q`` (shape ``(n_ok, T)``) and the
    list of instances that completed. ``config.concurrency <= 1`` dispatches to
    ``_run_experiment_sequential`` (the exact pre-Phase-B path); otherwise to
    ``_run_experiment_concurrent`` (seed-outer, task-inner thread pool)."""
    cache = ResponseCache(_cache_dir(config))
    runner = get_agent_runner(config, cache)
    evaluator = get_evaluator(config)
    # Eval-caching only pays for itself against real infra cost (a Modal Sandbox
    # boot + pytest run); proxy/judge/panel evaluators are local, cheap, and
    # already-deterministic re-computations, so skip the cache for them entirely
    # (None disables caching in _evaluate_cached -- see run_task).
    eval_cache = EvalCache(_eval_cache_dir(config)) if config.evaluator == "harness" else None

    runner.prepare(instances)
    try:
        if config.concurrency > 1:
            all_ok_records, all_ok_instances, budget_stopped = _run_experiment_concurrent(
                instances, config, runner, evaluator, eval_cache
            )
        else:
            all_ok_records, all_ok_instances, budget_stopped = _run_experiment_sequential(
                instances, config, runner, evaluator, eval_cache
            )
    finally:
        runner.cleanup()

    q = np.zeros((len(all_ok_records), config.t), dtype=float)
    for i, records in enumerate(all_ok_records):
        for rec in records:
            q[i, rec["round"]] = rec["quality"]

    # Collect all signal names across all records.
    signal_names: list[str] = []
    for records in all_ok_records:
        for rec in records:
            for name in rec.get("signals", {}):
                if name not in signal_names:
                    signal_names.append(name)

    signals_matrices: dict[str, list[list[float]]] = {name: [] for name in signal_names}
    for records in all_ok_records:
        row: dict[str, list[float]] = {name: [0.0] * config.t for name in signal_names}
        for rec in records:
            for name in signal_names:
                row[name][rec["round"]] = rec.get("signals", {}).get(name, 0.0)
        for name in signal_names:
            signals_matrices[name].append(row[name])

    solved_rounds: list[int | None] = [
        records[0].get("solved_round") if records else None for records in all_ok_records
    ]
    assigned_models: list[str] = [
        records[0].get("model", config.model) if records else config.model
        for records in all_ok_records
    ]
    return q, all_ok_instances, signals_matrices, budget_stopped, solved_rounds, assigned_models


# ============================================================
# 7. Metrics (map to the theorems)
# ============================================================


def compute_metrics(
    q: np.ndarray,
    tau: float,
    solved_rounds: list[int | None] | None = None,
    n_seeds: int = 1,
) -> dict:
    """Compute the theory-facing metrics from the quality matrix ``q[i, t]``.

    Raw vs. cummax, and the two differ by statistic -- do not read one convention across
    both.  ``eta`` is measured on the running-best ``b[i,t]``, because Theorem refinement
    bounds the improving process and its cumulative sum must telescope to
    ``b_{i,T} - q_{i,0} <= 1 - q_{i,0}``, which holds for any path only under cummax.
    ``gamma`` is measured on the *raw* ``q[i,t]`` (see the strict ``>`` below), so a round
    that dips counts as unimproved rather than being erased.  One consequence worth naming,
    since it governs how gamma's sign should be read: on raw ``q`` the complement of
    "improved" pools exact ties with genuine regressions, and in this data it is
    overwhelmingly ties -- a converged, never-regressing refiner therefore scores the floor
    ``gamma = -1/2``.  ``scripts/part_b_edge_overlap.py`` splits that complement apart and
    re-measures the edge on rows that still have headroom.  Raw ``q`` is also what feeds
    plotting and margin computation.

    * Running-best ``b[i,t] = max_{s<=t} q[i,s]`` (monotone by construction).
    * Edge ``gamma_t = mean_i 1[q_{i,t} > q_{i,t-1}] - 1/2``; ``eps_t = 1/2 - gamma_t``;
      ``alpha_t = 1/2 ln((1-eps_t)/eps_t)``.
    * Ensemble error ``err_t = mean_i 1[b_{i,t} < tau]`` (running-best; monotone).
    * Convergence overlay ``exp(-2 * cumsum(max(0, gamma_s)^2))`` (only positive
      edges contribute guaranteed progress).
    * Saturation ``eta_{i,t} = b_{i,t} - b_{i,t-1} >= 0`` (running-best gain),
      ``S_{i,t} = sum eta = b_{i,t} - q_{i,0} <= 1 - q_{i,0}`` (Thm. refinement bound,
      valid for any path); ``eta_bar_t = mean_i eta_{i,t}``.
    * Margin ``m_{i,t} = 2 q_{i,t} - 1``.
    * Hazard/survival: from ``solved_rounds`` (the first round where running-best >= tau),
      compute round-specific hazard rate h_t and survival S_t.  Censoring-correct: tasks
      not solved by T are right-censored.
    * Across-seed CI (n_seeds > 1): per-task solve rate + Wilson 95% CI +
      initial-condition-sensitivity scalar (std of per-seed solve outcomes per task).
    """
    q = np.asarray(q, dtype=float)
    k, t = q.shape

    running_best = np.maximum.accumulate(q, axis=1)

    improved = np.zeros((k, t), dtype=float)
    improved[:, 1:] = (q[:, 1:] > q[:, :-1]).astype(float)
    edge_gamma = improved.mean(axis=0) - 0.5
    edge_gamma[0] = 0.0  # round 0 has no predecessor
    epsilon = 0.5 - edge_gamma
    eps_clipped = np.clip(epsilon, 1e-10, 1 - 1e-10)
    alpha = 0.5 * np.log((1 - eps_clipped) / eps_clipped)

    ensemble_err = (running_best < tau).mean(axis=0)
    convergence_bound = np.exp(-2.0 * np.cumsum(np.maximum(edge_gamma, 0.0) ** 2))

    # Per-round improvement is measured on the running best, so cumulative
    # improvement S_{i,t} = b_{i,t} - q_{i,0} <= 1 - q_{i,0} holds for ANY quality
    # path, not only monotone ones (a real agent may dip between rounds). Measuring
    # eta on the raw sequence would instead track upward variation, which can exceed
    # the ceiling. See Theorem refinement: it bounds the improving process.
    eta = np.zeros((k, t), dtype=float)
    eta[:, 1:] = running_best[:, 1:] - running_best[:, :-1]
    saturation = np.cumsum(eta, axis=1)
    ceiling = 1.0 - q[:, 0]
    eta_bar = eta.mean(axis=0)

    margin = 2.0 * q - 1.0

    # Hazard / survival from persisted solved_rounds (censoring-correct).
    # at_risk[t] = number of tasks not yet solved before round t.
    # events[t] = number of tasks first solved at round t.
    # hazard[t] = events[t] / at_risk[t] (0 if at_risk==0).
    # survival[t] = prod_{s<=t} (1 - hazard[s]).
    hazard: list[float] = []
    survival: list[float] = []
    if solved_rounds is not None and len(solved_rounds) > 0:
        at_risk = k
        surv = 1.0
        for rnd in range(t):
            events = sum(1 for sr in solved_rounds if sr == rnd)
            h = events / at_risk if at_risk > 0 else 0.0
            surv *= 1.0 - h
            hazard.append(h)
            survival.append(surv)
            at_risk -= events

    # Across-seed CI: when n_seeds > 1, q is stacked as (n_seeds * n_tasks, t).
    # Reshape to (n_seeds, n_tasks, t), compute per-task solve rate ± Wilson CI.
    solve_rate_per_task: list[float] = []
    solve_rate_ci_lo: list[float] = []
    solve_rate_ci_hi: list[float] = []
    initial_condition_sensitivity: float = 0.0
    if n_seeds > 1 and k % n_seeds == 0:
        n_tasks = k // n_seeds
        q_by_seed = running_best.reshape(n_seeds, n_tasks, t)
        solved_by_seed = q_by_seed.max(axis=2) >= tau  # (n_seeds, n_tasks)
        for task_idx in range(n_tasks):
            x = int(solved_by_seed[:, task_idx].sum())
            n = n_seeds
            # Wilson score interval
            z = 1.96
            p_hat = x / n
            denom = 1 + z**2 / n
            centre = (p_hat + z**2 / (2 * n)) / denom
            half = z * (p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5 / denom
            solve_rate_per_task.append(p_hat)
            solve_rate_ci_lo.append(max(0.0, centre - half))
            solve_rate_ci_hi.append(min(1.0, centre + half))
        # Initial-condition sensitivity: mean std across tasks of solved_by_seed.
        initial_condition_sensitivity = float(solved_by_seed.astype(float).std(axis=0).mean())

    result: dict = {
        "tau": tau,
        "quality_matrix": q.tolist(),
        "running_best": running_best.tolist(),
        "edge_gamma": edge_gamma.tolist(),
        "epsilon": epsilon.tolist(),
        "alpha": alpha.tolist(),
        "ensemble_err": ensemble_err.tolist(),
        "convergence_bound": convergence_bound.tolist(),
        "eta": eta.tolist(),
        "saturation": saturation.tolist(),
        "ceiling": ceiling.tolist(),
        "eta_bar": eta_bar.tolist(),
        "margin": margin.tolist(),
    }
    if hazard:
        result["hazard"] = hazard
        result["survival"] = survival
    if solve_rate_per_task:
        result["solve_rate_per_task"] = solve_rate_per_task
        result["solve_rate_ci_lo"] = solve_rate_ci_lo
        result["solve_rate_ci_hi"] = solve_rate_ci_hi
        result["initial_condition_sensitivity"] = initial_condition_sensitivity
    return result


# ============================================================
# 8. Figures (mirror the four existing swe_fig*)
# ============================================================


def compute_signal_agreement(
    signals_matrices: dict[str, list[list[float]]],
) -> dict[str, Any]:
    """Pairwise Spearman rank correlation + mean absolute level difference.

    Flattens each signal's i×t matrix to a 1-D vector and computes all pairwise
    statistics. Returns a dict with ``spearman`` and ``mean_abs_diff`` sub-dicts
    (keys = "signalA_vs_signalB") plus ``signal_names`` for reference.
    """

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        n = len(a)
        if n < 3:
            return 0.0
        rank_a = np.argsort(np.argsort(a)).astype(float)
        rank_b = np.argsort(np.argsort(b)).astype(float)
        d = rank_a - rank_b
        denom = n * (n * n - 1)
        if denom == 0:
            return 0.0
        r = 1.0 - 6.0 * float(np.sum(d**2)) / denom
        return float(np.clip(r, -1.0, 1.0))

    names = list(signals_matrices.keys())
    flat: dict[str, np.ndarray] = {}
    for name, matrix in signals_matrices.items():
        arr = np.array(matrix, dtype=float).ravel()
        flat[name] = arr

    spearman: dict[str, float] = {}
    mean_abs_diff: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_vs_{b}"
            fa, fb = flat[a], flat[b]
            spearman[key] = _spearman(fa, fb)
            mean_abs_diff[key] = float(np.mean(np.abs(fa - fb)))

    return {
        "signal_names": names,
        "spearman": spearman,
        "mean_abs_diff": mean_abs_diff,
    }


def _savefig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{stem}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{stem}.png"), dpi=150)
    plt.close(fig)


def experiment_llm_1_pass_convergence(metrics: dict) -> None:
    """Fig 1: running-best pass error ``err_t`` vs round + dashed convergence overlay
    (Cor. convergence)."""
    err = np.array(metrics["ensemble_err"])
    bound = np.array(metrics["convergence_bound"])
    rounds = np.arange(len(err))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, err, "-o", linewidth=2, label="Empirical pass-error (running best)")
    ax.plot(rounds, bound, "--", linewidth=1.5, alpha=0.7, label=r"$\exp(-2\sum \gamma_s^2)$")
    ax.set_xlabel("Refinement Round (t)")
    ax.set_ylabel(r"Fraction unsolved  $\mathrm{err}_t$")
    ax.set_title("LLM Refinement: Pass Convergence vs Round")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _savefig(fig, "llm_fig1_pass_convergence")


def experiment_llm_2_quality_edge(q: np.ndarray, metrics: dict) -> None:
    """Fig 2: L: per-task quality trajectories + mean; R: per-round edge bars."""
    q = np.asarray(q, dtype=float)
    rounds = np.arange(q.shape[1])
    gamma = np.array(metrics["edge_gamma"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i in range(q.shape[0]):
        axes[0].plot(rounds, q[i], "-", alpha=0.4, linewidth=1)
    axes[0].plot(rounds, q.mean(axis=0), "-o", color="black", linewidth=2.5, label="Mean quality")
    axes[0].set_xlabel("Refinement Round (t)")
    axes[0].set_ylabel("Patch Quality  $q_{i,t}$")
    axes[0].set_title("Per-task Quality Trajectories")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(rounds, gamma, color="steelblue", alpha=0.8)
    axes[1].axhline(y=0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Refinement Round (t)")
    axes[1].set_ylabel(r"Edge  $\gamma_t$")
    axes[1].set_title("Per-round Improvement Edge")
    axes[1].grid(True, alpha=0.3)
    _savefig(fig, "llm_fig2_quality_edge")


def experiment_llm_3_saturation(q: np.ndarray, metrics: dict) -> None:
    """Fig 3: cumulative improvement ``S_{i,t}`` vs round + per-task ceiling +
    ``eta_bar_t`` decay (Thm. refinement)."""
    q = np.asarray(q, dtype=float)
    rounds = np.arange(q.shape[1])
    saturation = np.array(metrics["saturation"])
    ceiling = np.array(metrics["ceiling"])
    eta_bar = np.array(metrics["eta_bar"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i in range(q.shape[0]):
        axes[0].plot(rounds, saturation[i], "-", alpha=0.5, linewidth=1.2)
        axes[0].axhline(y=ceiling[i], color="gray", linestyle=":", alpha=0.4)
    axes[0].set_xlabel("Refinement Round (t)")
    axes[0].set_ylabel(r"Cumulative improvement  $S_{i,t}=\sum \eta_{i,s}$")
    axes[0].set_title(r"Saturation vs Ceiling $1-q_{i,0}$")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rounds, eta_bar, "-o", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Refinement Round (t)")
    axes[1].set_ylabel(r"Mean per-round gain  $\bar\eta_t$")
    axes[1].set_title("Diminishing Returns")
    axes[1].grid(True, alpha=0.3)
    _savefig(fig, "llm_fig3_saturation")


def experiment_llm_4_margin_distribution(q: np.ndarray, metrics: dict) -> None:
    """Fig 4: overlaid margin histograms at rounds 0 / mid / final (rightward shift)."""
    margin = np.array(metrics["margin"])
    t = margin.shape[1]
    rounds_to_show = sorted({0, t // 2, t - 1})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rounds_to_show:
        ax.hist(margin[:, r], bins=20, alpha=0.5, density=True, label=f"Round {r}")
    ax.axvline(
        x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Decision boundary"
    )
    ax.set_xlabel(r"Margin  $m_{i,t}=2q_{i,t}-1$")
    ax.set_ylabel("Density")
    ax.set_title("LLM Refinement: Margin Distribution Over Rounds")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _savefig(fig, "llm_fig4_margin_distribution")


# ============================================================
# 9. Main
# ============================================================


def main() -> None:
    config = Config.from_env()
    print("=" * 60)
    print("LLM SWE-bench Refinement Experiment")
    print(f"  model={config.model}  K={config.k}  T={config.t}  tau={config.tau}")
    print(f"  agent={config.agent}  evaluator={config.evaluator}")
    print(f"  tier_policy={config.tier_policy}  tier_models={config.tier_models}")
    print("=" * 60)

    if config.agent == "stub":
        print("  [offline demo] using synthetic tasks; figures are ILLUSTRATIVE, not")
        print("  measured LLM output. Set LLM_SIM_AGENT=mini + LITELLM_API_KEY for a real run.")
        instances = make_demo_instances(config.k)
    else:
        instances = load_swe_tasks(config.k, exclude_ids=frozenset(config.exclude_ids))
    q, ok_instances, signals_matrices, budget_stopped, solved_rounds, assigned_models = (
        run_experiment(instances, config)
    )
    metrics = compute_metrics(q, tau=config.tau)
    agreement = compute_signal_agreement(signals_matrices)

    with open(os.path.join(DATADIR, "llm_results.json"), "w") as f:
        _REDACT = {"api_key", "api_base"}
        json.dump(
            {
                "config": {
                    **{k: "REDACTED" if k in _REDACT else v for k, v in asdict(config).items()},
                    "cache_dir": os.path.relpath(_cache_dir(config), os.path.dirname(__file__)),
                },
                "instance_ids": [inst.get("instance_id") for inst in ok_instances],
                "models": assigned_models,
                "quality_matrix": q.tolist(),
                "solved_rounds": solved_rounds,
                "signals_matrices": signals_matrices,
            },
            f,
            indent=2,
        )
    with open(os.path.join(DATADIR, "llm_metrics.json"), "w") as f:
        json.dump({**metrics, "signal_agreement": agreement}, f, indent=2)

    if q.shape[0] > 0:
        try:
            experiment_llm_1_pass_convergence(metrics)
            experiment_llm_2_quality_edge(q, metrics)
            experiment_llm_3_saturation(q, metrics)
            experiment_llm_4_margin_distribution(q, metrics)
        except Exception as exc:  # noqa: BLE001 -- figures are best-effort; the
            # results/metrics JSON above is the run's actual output and is already
            # on disk. A broken plotting environment (e.g. a shared venv shadowed by
            # an agent-run `pip install -e .` inside a cloned SWE-bench repo) must
            # not discard a completed arm or block a sweep runner's next step.
            print(
                f"\nFigure generation failed ({type(exc).__name__}: {exc}) -- "
                "skipping figures, results/metrics were already written."
            )
    else:
        print("\nNo task-seeds completed -- skipping figure generation.")

    if budget_stopped:
        print("\nBUDGET EXHAUSTED — partial results written.")
        print("Re-run with the same command tomorrow; completed rounds are cached.")
    else:
        print("\nDone. Wrote data/llm_results.json, data/llm_metrics.json, figures/llm_*.pdf")


if __name__ == "__main__":
    main()
