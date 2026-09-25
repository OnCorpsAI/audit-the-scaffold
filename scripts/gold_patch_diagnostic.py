#!/usr/bin/env python3
"""Gold-patch diagnostic: apply the ground-truth reference patch for one or more
SWE-bench Lite instance_ids through ``HarnessEvaluator`` and report whether the
harness scores it ``fraction_passing == 1.0``.

Since the gold patch is known-correct by construction, anything less than
1.0 here is a harness/infrastructure bug, not a model-quality signal --
this is how the Part B feedback-ablation sweep's 23/50 infra-broken tasks
were first confirmed (see tasks/part-b-scale-up-plan.md).

Usage:
    uv run python scripts/gold_patch_diagnostic.py <instance_id> [<instance_id> ...]
    uv run python scripts/gold_patch_diagnostic.py --backend local astropy__astropy-6938

Exit code 0 if every requested instance scores fraction_passing == 1.0, non-zero
otherwise, so this can gate "is this task/fix actually fixed?" in a script as
well as be read by eye.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_ids", nargs="+", help="SWE-bench Lite instance_id(s)")
    parser.add_argument(
        "--backend",
        choices=["modal", "local"],
        default="modal",
        help="eval backend (default: modal, matching the real sweep's harness backend)",
    )
    args = parser.parse_args()

    os.environ["LLM_SIM_HARNESS_BACKEND"] = args.backend

    import datasets

    import llm_simulations as sim

    print(
        f"Loading princeton-nlp/SWE-bench_Lite (test split) to resolve {len(args.instance_ids)} "
        "instance_id(s) ..."
    )
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {row["instance_id"]: dict(row) for row in ds}

    evaluator = sim.HarnessEvaluator()
    results: list[tuple[str, str]] = []  # (instance_id, status)
    all_clean = True

    for instance_id in args.instance_ids:
        instance = by_id.get(instance_id)
        if instance is None:
            print(f"[{instance_id}] NOT_FOUND in SWE-bench_Lite test split")
            results.append((instance_id, "NOT_FOUND"))
            all_clean = False
            continue

        gold_patch = instance.get("patch") or ""
        try:
            result = evaluator.evaluate(gold_patch, instance)
        except RuntimeError as exc:
            print(f"[{instance_id}] INFRA_ERROR: {exc}")
            results.append((instance_id, "INFRA_ERROR"))
            all_clean = False
            continue
        finally:
            evaluator.teardown(instance_id)

        clean = result.fraction_passing == 1.0
        status = "CLEAN" if clean else "BROKEN"
        all_clean = all_clean and clean
        print(
            f"[{instance_id}] {status}: fraction_passing={result.fraction_passing:.3f} "
            f"quality={result.quality:.3f} passed={result.passed} detail={result.detail}"
        )
        results.append((instance_id, status))

    print("\n" + "=" * 60)
    print("Gold-patch diagnostic summary")
    print("=" * 60)
    for instance_id, status in results:
        print(f"  {status:12s} {instance_id}")

    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
