#!/usr/bin/env python3
"""Scoped real-pipeline validation of the two ``HarnessEvaluator`` fixes (JUnit-XML
scoring, stale-parametrize-ID resolution) against exactly 28 SWE-bench Lite tasks:
the 13 tasks previously broken by those bugs, plus the 15 tasks backfilled into the
Part B pool to replace the 10 dropped Django/SymPy/network tasks. See
``tasks/part-b-scale-up-plan.md`` ("Harness fixes landed (2026-07-30)") for the full
history and ``scripts/gold_patch_diagnostic.py`` for the harness-only check this
complements -- that script proves the harness scores a known-correct patch as
``fraction_passing == 1.0`` in isolation; this one drives the *real* pipeline (Haiku
generation + ``diagnostic`` feedback + Modal harness eval, ``T=4`` rounds) to check
the fixes hold up when the agent's own imperfect patches are being scored round over
round, not just a gold patch.

This is a sanity check gating the full 6-arm K=55x5-seed sweep
(``tasks/execute-modal-experiment.md``) -- it does NOT launch that sweep.

Usage:
    uv run python scripts/validate_diagnostic_28task.py

Required env (see tasks/execute-modal-experiment.md "Prerequisites"):
    LITELLM_API_KEY, LITELLM_API_BASE  -- internal proxy credentials (VPN required)
    LLM_SIM_MODEL, LLM_SIM_FEEDBACK, LLM_SIM_EVALUATOR, LLM_SIM_AGENT, LLM_SIM_T,
    LLM_SIM_SEEDS, LLM_SIM_HARNESS_BACKEND, LLM_SIM_CONCURRENCY,
    LLM_HARNESS_SANDBOX_TIMEOUT -- see the run command in the handoff plan/tasks/todo.md

Exit code 0 if the run completed with no hard errors (regardless of what the
solve-rate/degenerate-score flags below report -- those are advisory, read by eye),
non-zero on any exception or missing-task_id/missing-credential failure.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIXED_13 = [
    "astropy__astropy-6938",
    "pytest-dev__pytest-5103",
    "pytest-dev__pytest-5221",
    "astropy__astropy-14182",
    "astropy__astropy-14365",
    "astropy__astropy-14995",
    "matplotlib__matplotlib-22835",
    "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23314",
    "pylint-dev__pylint-7080",
    "sphinx-doc__sphinx-10451",
    "pytest-dev__pytest-11143",
    "pallets__flask-5063",
]

BACKFILLED_15 = [
    "astropy__astropy-7746",
    "matplotlib__matplotlib-23476",
    "matplotlib__matplotlib-23562",
    "psf__requests-3362",
    "pydata__xarray-5131",
    "pylint-dev__pylint-7228",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-5413",
    "pytest-dev__pytest-5495",
    "pytest-dev__pytest-5692",
    "scikit-learn__scikit-learn-11281",
    "scikit-learn__scikit-learn-12471",
    "scikit-learn__scikit-learn-13142",
    "sphinx-doc__sphinx-7738",
    "sphinx-doc__sphinx-7975",
]

_overlap = set(FIXED_13) & set(BACKFILLED_15)
if _overlap:
    raise ValueError(f"FIXED_13/BACKFILLED_15 overlap: {_overlap}")
ALL_28 = FIXED_13 + BACKFILLED_15
if len(ALL_28) != 28:
    raise ValueError(f"expected 28 ids, got {len(ALL_28)}")

OUTPUT_PATH = os.path.join(REPO, "data", "validation_28task_diagnostic.json")


def _check_credentials() -> None:
    missing = [name for name in ("LITELLM_API_KEY", "LITELLM_API_BASE") if not os.environ.get(name)]
    if missing:
        print(
            f"MISSING required env var(s): {', '.join(missing)}. "
            "This run needs the internal LiteLLM proxy (VPN required) -- "
            "see tasks/execute-modal-experiment.md 'Prerequisites'. "
            "If only LITELLM_API_BASE is missing, source your shell profile "
            "first (non-interactive shells don't inherit it).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _load_instances(instance_ids: list[str]) -> list[dict]:
    import datasets

    print(
        f"Loading princeton-nlp/SWE-bench_Lite (test split) to resolve "
        f"{len(instance_ids)} instance_id(s) ..."
    )
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {row["instance_id"]: dict(row) for row in ds}

    missing = [i for i in instance_ids if i not in by_id]
    if missing:
        print(f"NOT_FOUND in SWE-bench_Lite test split: {missing}", file=sys.stderr)
        raise SystemExit(1)

    return [by_id[i] for i in instance_ids]


def _fp_sequences(by_instance: dict[str, list[list[dict]]], iid: str) -> list[list[float]]:
    """Per-seed ``fraction_passing`` sequences for one task, ordered by round."""
    return [
        [r["fraction_passing"] for r in sorted(recs, key=lambda r: r["round"])]
        for recs in by_instance.get(iid, [])
    ]


def _is_degenerate(seqs: list[list[float]]) -> bool:
    """True when every score across every round and seed is flat 0.0 or flat 1.0.

    That pattern is the signature the two ``HarnessEvaluator`` fixes were meant to
    remove -- a harness that scores nothing, or everything, regardless of the patch. A
    single observation cannot evidence flatness, so it never counts.
    """
    flat = [v for seq in seqs for v in seq]
    if len(flat) <= 1:
        return False
    return all(v == 0.0 for v in flat) or all(v == 1.0 for v in flat)


def _solve_rate_flag(rate: float) -> str:
    """Advisory warning for an implausible extreme; empty string when plausible."""
    if rate == 0.0:
        return "  <-- WARN: universally 0 (implausible for Haiku on a mixed set)"
    if rate == 1.0:
        return "  <-- WARN: universally 1 (implausible for Haiku on a mixed set)"
    return ""


def _report_fixed_13(by_instance: dict[str, list[list[dict]]], completed_ids: set[str]) -> None:
    print("\n" + "=" * 60)
    print(f"PREVIOUSLY BROKEN, NOW FIXED ({len(FIXED_13)})")
    print("=" * 60)
    for iid in FIXED_13:
        if iid not in completed_ids:
            print(f"  [MISSING] {iid} -- did not complete (skipped or errored)")
            continue
        seqs = _fp_sequences(by_instance, iid)
        flag = (
            "  <-- WARN: degenerate (flat 0.0/1.0 every round/seed)" if _is_degenerate(seqs) else ""
        )
        print(f"  {iid}: fraction_passing by seed = {seqs}{flag}")


def _report_backfilled_15(
    by_instance: dict[str, list[list[dict]]], completed_ids: set[str]
) -> list[bool]:
    print("\n" + "=" * 60)
    print(f"NEWLY BACKFILLED ({len(BACKFILLED_15)})")
    print("=" * 60)
    backfilled_passed = []
    for iid in BACKFILLED_15:
        if iid not in completed_ids:
            print(f"  [MISSING] {iid} -- did not complete (skipped or errored)")
            continue
        seqs = _fp_sequences(by_instance, iid)
        any_passed = any(r["passed"] for recs in by_instance.get(iid, []) for r in recs)
        backfilled_passed.append(any_passed)
        print(f"  {iid}: fraction_passing by seed = {seqs}  passed_any_round={any_passed}")
    return backfilled_passed


def main() -> int:
    _check_credentials()

    import llm_simulations as sim

    config = sim.Config.from_env()
    print("=" * 60)
    print("28-task diagnostic-feedback-mode validation")
    print(f"  model={config.model}  feedback_mode={config.feedback_mode}")
    print(f"  evaluator={config.evaluator}  agent={config.agent}  T={config.t}")
    print(f"  n_seeds={config.n_seeds}  concurrency={config.concurrency}")
    print(f"  harness_backend={os.environ.get('LLM_SIM_HARNESS_BACKEND', 'local')}")
    print("=" * 60)

    instances = _load_instances(ALL_28)

    cache = sim.ResponseCache(sim._cache_dir(config))
    runner = sim.get_agent_runner(config, cache)
    evaluator = sim.get_evaluator(config)
    eval_cache = (
        sim.EvalCache(sim._eval_cache_dir(config)) if config.evaluator == "harness" else None
    )

    runner.prepare(instances)
    try:
        if config.concurrency > 1:
            all_ok_records, all_ok_instances, budget_stopped = sim._run_experiment_concurrent(
                instances, config, runner, evaluator, eval_cache
            )
        else:
            all_ok_records, all_ok_instances, budget_stopped = sim._run_experiment_sequential(
                instances, config, runner, evaluator, eval_cache
            )
    finally:
        runner.cleanup()

    q = __import__("numpy").zeros((len(all_ok_records), config.t), dtype=float)
    for i, records in enumerate(all_ok_records):
        for rec in records:
            q[i, rec["round"]] = rec["quality"]

    metrics = sim.compute_metrics(q, tau=config.tau) if q.shape[0] > 0 else {}

    by_instance: dict[str, list[list[dict]]] = {}
    for instance, records in zip(all_ok_instances, all_ok_records, strict=True):
        iid = str(instance.get("instance_id", ""))
        by_instance.setdefault(iid, []).append(records)

    with open(OUTPUT_PATH, "w") as f:
        _REDACT = {"api_key", "api_base"}
        from dataclasses import asdict

        json.dump(
            {
                "config": {
                    k: ("REDACTED" if k in _REDACT else v) for k, v in asdict(config).items()
                },
                "budget_stopped": budget_stopped,
                "metrics": metrics,
                "by_instance": by_instance,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {OUTPUT_PATH}")

    requested_ids = set(ALL_28)
    completed_ids = set(by_instance.keys())
    missing_ids = requested_ids - completed_ids
    if budget_stopped:
        print(
            "\nBUDGET EXHAUSTED — the proxy's daily spend throttle tripped mid-run. "
            "Partial results only; re-run the same command to resume from cache."
        )

    _report_fixed_13(by_instance, completed_ids)
    backfilled_passed = _report_backfilled_15(by_instance, completed_ids)

    if backfilled_passed:
        rate = sum(backfilled_passed) / len(backfilled_passed)
        print(
            f"\n  Backfilled-15 solve rate (any round, either seed): "
            f"{rate:.2f}{_solve_rate_flag(rate)}"
        )

    if missing_ids:
        print(
            f"\n[MISSING overall] {len(missing_ids)}/28 task(s) did not complete: "
            f"{sorted(missing_ids)}"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
