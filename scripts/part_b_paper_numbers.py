#!/usr/bin/env python3
"""Part B Phase 1: emit the exact numbers the paper.tex refresh needs, computed
from source data -- not hand-transcribed from tasks/part-b-phase1-stats-analysis.md.

Reuses part_b_anova_power.py's loaders/ANOVA (imported, not duplicated) for the
2x3 repeated-measures ANOVA and adds the pieces that script doesn't emit:
  - per-arm (model x mode) solve rate with Wilson 95% CI (final-round running-best
    quality >= TAU, averaged over seeds then thresholded per task, matching the
    paper's existing solve-rate definition in Table tab:swebench), reported at both
    TAU=0.6 (fractional test passing) and TAU_RESOLVED=1.0 (SWE-bench's own resolved
    criterion) so the two are never confused for each other
  - per-model marginal mean quality by mode (the Haiku-monotonic / Sonnet-crossover
    table, Section 8.2 of the stats doc)
  - temperature-noise numbers: seed-to-seed SD of final-round quality at a fixed
    (task, model, mode), vs. the mean |diagnostic - blind| per-task signal
  - 80%-power crossing K for the mode and interaction effects, by linear
    interpolation on the existing bootstrap cache (data/part_b_k55_anova_power.json,
    written by ``uv run python scripts/part_b_anova_power.py``) -- does not rerun
    the expensive bootstrap; if the cache is missing, tells you to generate it first.

Usage:
    uv run python scripts/part_b_paper_numbers.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from part_b_anova_power import ARMS, MODELS, build_table, load_raw_by_arm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(REPO, "data")
TAU = 0.6
# The SWE-bench "resolved" bar: every FAIL_TO_PASS/PASS_TO_PASS test passing, i.e.
# fraction_passing == 1.0. Reported alongside TAU because TAU=0.6 is a *fractional*
# passing threshold and is therefore NOT comparable to published SWE-bench resolve
# rates -- a task with 60% of its tests passing counts as solved at TAU and as
# unresolved by SWE-bench's own criterion. Both are emitted from one run so the
# paper can quote the pair without a second invocation drifting from the first.
TAU_RESOLVED = 1.0
MODES_ORDERED = ["independent", "blind", "diagnostic"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def solve_rate_by_arm(raw: dict[tuple[str, str], dict[str, list[float]]], tau: float = TAU) -> dict:
    """Per-(model,mode) solve rate: a task 'solves' if its mean-over-5-seeds
    final-round quality >= ``tau`` -- same DV (seed-averaged final-round quality) the
    ANOVA itself tests, just thresholded, so this is a solve-rate view of the exact
    same numbers, not a different quantity computed from raw per-seed data.

    ``tau`` is a parameter rather than the module constant because the paper reports
    two thresholds: TAU=0.6 (fractional test passing, the threshold used throughout
    Part B) and TAU_RESOLVED=1.0 (SWE-bench's own resolved criterion). Thresholding
    the *seed mean* means a task can sit strictly between the two even when no single
    seed does, so the 1.0 rate is not simply a count of all-seeds-perfect tasks.

    Note the ANOVA itself is tau-free -- its DV is continuous quality -- so varying
    tau moves the solve rates and nothing else in the ANOVA table.
    """
    out = {}
    for model, mode, _ in ARMS:
        by_task = raw[(model, mode)]
        n = len(by_task)
        k = sum(1 for vals in by_task.values() if float(np.mean(vals)) >= tau)
        lo, hi = wilson_ci(k, n)
        out[f"{model}_{mode}"] = {"k": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi}
    return out


def marginal_means_by_mode(raw: dict[tuple[str, str], dict[str, list[float]]]) -> dict:
    """Per-model mean final-round quality (averaged over seeds, then over tasks) by
    mode, in MODES_ORDERED order -- the crossover table (stats doc Section 8.2).
    """
    out: dict[str, dict[str, float]] = {}
    for model in MODELS:
        out[model] = {}
        for mode in MODES_ORDERED:
            by_task = raw[(model, mode)]
            task_means = [float(np.mean(vals)) for vals in by_task.values()]
            out[model][mode] = float(np.mean(task_means))
    return out


def temperature_noise(raw: dict[tuple[str, str], dict[str, list[float]]]) -> dict:
    """Seed-to-seed SD of final-round quality at a fixed (task, model, mode),
    averaged across all task/arm cells, vs. the mean |diagnostic - blind| per-task
    signal (both models pooled) -- reproduces stats doc Section 9's noise-vs-signal
    comparison directly from source data.
    """
    seed_sds = []
    for model, mode, _ in ARMS:
        for vals in raw[(model, mode)].values():
            seed_sds.append(float(np.std(vals, ddof=1)))
    mean_seed_sd = float(np.mean(seed_sds))

    diffs = []
    for model in MODELS:
        blind = raw[(model, "blind")]
        diag = raw[(model, "diagnostic")]
        for task in blind:
            b = float(np.mean(blind[task]))
            d = float(np.mean(diag[task]))
            diffs.append(abs(d - b))
    mean_abs_diff = float(np.mean(diffs))

    return {
        "mean_seed_to_seed_sd": mean_seed_sd,
        "mean_abs_diagnostic_minus_blind": mean_abs_diff,
        "noise_to_signal_ratio": mean_seed_sd / mean_abs_diff,
    }


def interpolate_crossing(
    power_by_k: dict[str, dict], effect_key: str, target: float = 0.80
) -> float | None:
    """Linear interpolation of the K at which bootstrap power for ``effect_key``
    ('B' for mode, 'AB' for interaction) first crosses ``target``. Returns None if
    the cached k_grid never reaches ``target``.
    """
    ks = sorted(int(k) for k in power_by_k)
    for k_lo, k_hi in zip(ks, ks[1:], strict=False):
        p_lo = power_by_k[str(k_lo)][effect_key]
        p_hi = power_by_k[str(k_hi)][effect_key]
        if p_lo < target <= p_hi:
            frac = (target - p_lo) / (p_hi - p_lo)
            return k_lo + frac * (k_hi - k_lo)
    return None


def _print_rates(rates: dict) -> None:
    for key, v in rates.items():
        print(
            f"  {key:<22} {v['k']}/{v['n']} = {v['rate']:.1%}  "
            f"95% Wilson CI [{v['ci_lo']:.1%}, {v['ci_hi']:.1%}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU,
        help=f"primary solve threshold (default {TAU}). The SWE-bench resolved bar "
        f"{TAU_RESOLVED} is always reported alongside it.",
    )
    args = parser.parse_args()

    raw = load_raw_by_arm()
    all_arms = [(m, mo) for m, mo, _ in ARMS]
    df = build_table(raw, all_arms)
    n_tasks = df["task"].nunique()
    print(f"Loaded {n_tasks} tasks x 2 models x 3 modes.\n")

    print("=" * 70)
    print(f"Per-arm solve rate (mean-over-5-seeds final-round quality >= {args.tau})")
    print("=" * 70)
    rates = solve_rate_by_arm(raw, tau=args.tau)
    _print_rates(rates)

    print("\n" + "=" * 70)
    print(
        f"Same, at the SWE-bench resolved bar (>= {TAU_RESOLVED}: every test passing). "
        "NOT comparable to the rates above."
    )
    print("=" * 70)
    rates_resolved = solve_rate_by_arm(raw, tau=TAU_RESOLVED)
    _print_rates(rates_resolved)

    print("\n" + "=" * 70)
    print("Marginal mean quality by mode (independent -> blind -> diagnostic order)")
    print("=" * 70)
    means = marginal_means_by_mode(raw)
    for model, by_mode in means.items():
        vals = " / ".join(f"{mo}={by_mode[mo]:.3f}" for mo in MODES_ORDERED)
        print(f"  {model:<8} {vals}")

    print("\n" + "=" * 70)
    print("Temperature-noise vs. between-mode signal")
    print("=" * 70)
    tn = temperature_noise(raw)
    print(f"  mean seed-to-seed SD (fixed task/model/mode): {tn['mean_seed_to_seed_sd']:.4f}")
    print(
        f"  mean |diagnostic - blind| per task:           "
        f"{tn['mean_abs_diagnostic_minus_blind']:.4f}"
    )
    print(f"  noise:signal ratio:                           {tn['noise_to_signal_ratio']:.2f}x")

    cache_path = os.path.join(DATADIR, "part_b_k55_anova_power.json")
    model_effect_ci = None
    print("\n" + "=" * 70)
    print("80%-power crossing K (interpolated from cached bootstrap, full 55-task pool)")
    print("=" * 70)
    if not os.path.exists(cache_path):
        print(f"  MISSING: {cache_path}")
        print(
            "  Generate it first: uv run python scripts/part_b_anova_power.py "
            "--boot-reps 30000 "
            "--k-grid 55,75,100,150,200,300,400,430,500,600,700,750,800,900,1000"
        )
    else:
        with open(cache_path) as f:
            cached = json.load(f)
        power_by_k = cached["full_pool"]["bootstrap_power"]
        k_mode = interpolate_crossing(power_by_k, "B")
        k_interaction = interpolate_crossing(power_by_k, "AB")
        print(
            f"  mode:        K~{k_mode:.0f}" if k_mode else "  mode: not reached in cached k_grid"
        )
        print(
            f"  interaction: K~{k_interaction:.0f}"
            if k_interaction
            else "  interaction: not reached in cached k_grid"
        )
        anova = cached["full_pool"]["anova"]
        print(f"\n  ANOVA (from cache, source of truth for Table): {json.dumps(anova, indent=2)}")
        model_effect_ci = cached["full_pool"].get("model_effect_ci")
        if model_effect_ci:
            print(
                f"\n  Model-effect Cohen's f 95% CI (from cache): "
                f"[{model_effect_ci['cohen_f_ci_low']:.3f}, "
                f"{model_effect_ci['cohen_f_ci_high']:.3f}]"
            )

    out = {
        "tau": args.tau,
        "tau_resolved": TAU_RESOLVED,
        "solve_rate_by_arm": rates,
        "solve_rate_by_arm_resolved": rates_resolved,
        "marginal_means_by_mode": means,
        "temperature_noise": tn,
        "model_effect_ci": model_effect_ci,
    }
    out_path = os.path.join(DATADIR, "part_b_k55_paper_numbers.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
