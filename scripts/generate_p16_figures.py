#!/usr/bin/env python3
"""P1.6: Generate Part B comparison figures for Haiku vs Sonnet.

The edge-collapse and overlap figures (the ones the paper uses) are built from the six
K=55 arm files. ``fig1``/``fig3``/``fig5`` are built from the legacy K=20 single-seed
``*_t4_pinned.json`` pilot and are not referenced by the paper.

Overwrites:
  figures/llm_fig1_pass_convergence.{pdf,png}  — dual-model + Wilson CI bands
  figures/llm_fig3_saturation.{pdf,png}         — dual-model + ±SEM bands
Creates:
  figures/llm_fig5_solve_comparison.{pdf,png}   — per-task grid + aggregate CI bars

Usage:
    uv run python scripts/generate_p16_figures.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from part_b_anova_power import load_raw_by_arm  # noqa: E402
from part_b_edge_overlap import CACHE_NAME as EDGE_OVERLAP_CACHE  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, "figures")
TAU = 0.6
COLORS = {"Haiku": "steelblue", "Sonnet": "darkorange"}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _running_best(q: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(q, axis=1)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _savefig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{stem}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{stem}.png"), dpi=150)
    plt.close(fig)


def fig1_pass_convergence(datasets: dict[str, np.ndarray]) -> None:
    """Pass-error err_t vs round for both models, with 95% Wilson CI shading."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, q in datasets.items():
        rb = _running_best(q)
        K, T = rb.shape
        rounds = np.arange(T)
        n_solved = np.array([(rb[:, t] >= TAU).sum() for t in rounds])
        err = 1.0 - n_solved / K
        lo = np.array([1.0 - _wilson_ci(int(n_solved[t]), K)[1] for t in rounds])
        hi = np.array([1.0 - _wilson_ci(int(n_solved[t]), K)[0] for t in rounds])
        c = COLORS[label]
        ax.plot(rounds, err, "-o", linewidth=2, color=c, label=label)
        ax.fill_between(rounds, lo, hi, alpha=0.15, color=c)
    ax.set_xlabel("Refinement Round (t)")
    ax.set_ylabel(r"Fraction unsolved  $\mathrm{err}_t$")
    ax.set_title("LLM Refinement: Pass Convergence vs Round (95% Wilson CI)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _savefig(fig, "llm_fig1_pass_convergence")


def fig3_saturation(datasets: dict[str, np.ndarray]) -> None:
    """Mean saturation S_t ±SEM and mean eta_bar_t ±SEM for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, q in datasets.items():
        rb = _running_best(q)
        K, T = rb.shape
        rounds = np.arange(T)
        c = COLORS[label]
        # Left: mean S_i,t = best_t - q_i,0, ±SEM envelope
        sat = rb - rb[:, 0:1]
        sat_mean = sat.mean(axis=0)
        sat_sem = sat.std(axis=0, ddof=1) / math.sqrt(K)
        axes[0].plot(rounds, sat_mean, "-o", linewidth=2, color=c, label=label)
        axes[0].fill_between(rounds, sat_mean - sat_sem, sat_mean + sat_sem, alpha=0.15, color=c)
        # Right: eta_bar_t = mean marginal gain, ±SEM
        delta = np.diff(rb, axis=1)  # (K, T-1), per-task per-round gain
        eta_bar = delta.mean(axis=0)
        eta_sem = delta.std(axis=0, ddof=1) / math.sqrt(K)
        r_eta = np.arange(1, T)
        axes[1].plot(r_eta, eta_bar, "-o", linewidth=2, color=c, label=label)
        axes[1].fill_between(r_eta, eta_bar - eta_sem, eta_bar + eta_sem, alpha=0.15, color=c)
    axes[0].set_xlabel("Refinement Round (t)")
    axes[0].set_ylabel(r"Mean cumulative improvement  $\bar{S}_t$ (±1 SEM)")
    axes[0].set_title(r"Saturation vs Ceiling $1-q_{i,0}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Refinement Round (t)")
    axes[1].set_ylabel(r"Mean per-round gain  $\bar\eta_t$ (±1 SEM)")
    axes[1].set_title("Diminishing Returns")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _savefig(fig, "llm_fig3_saturation")


def fig5_solve_comparison(
    haiku_q: np.ndarray,
    instance_ids: list[str],
    sonnet_q: np.ndarray,
) -> None:
    """Per-task solve grid (left) + aggregate solve rates with Wilson CI bars (right)."""
    K = len(instance_ids)
    short = [iid.split("__")[-1][:20] for iid in instance_ids]
    haiku_solved = (_running_best(haiku_q)[:, -1] >= TAU).astype(int)
    sonnet_solved = (_running_best(sonnet_q)[:, -1] >= TAU).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-task binary heat grid
    grid = np.stack([haiku_solved, sonnet_solved], axis=0)  # (2, K)
    im = axes[0].imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_xticks(np.arange(K))
    axes[0].set_xticklabels(short, rotation=90, fontsize=6)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Haiku", "Sonnet"], fontsize=9)
    axes[0].set_title("Per-task solve (green = pass, red = fail)")
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.04, ticks=[0, 1])

    # Right: aggregate solve rates ±Wilson CI
    models = ["Haiku", "Sonnet"]
    counts = [int(haiku_solved.sum()), int(sonnet_solved.sum())]
    rates = [c / K for c in counts]
    cis = [_wilson_ci(c, K) for c in counts]
    lo_err = [r - ci[0] for r, ci in zip(rates, cis, strict=True)]
    hi_err = [ci[1] - r for r, ci in zip(rates, cis, strict=True)]
    xs = np.arange(len(models))
    axes[1].bar(
        xs,
        rates,
        color=[COLORS[m] for m in models],
        alpha=0.8,
        yerr=[lo_err, hi_err],
        capsize=10,
        error_kw={"linewidth": 2},
    )
    for i, (r, c, hi) in enumerate(zip(rates, counts, hi_err, strict=True)):
        axes[1].text(  # noqa: E501
            i,
            r + hi + 0.03,
            f"{c}/{K} = {r:.0%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].set_ylabel(r"Solve rate (running-best $\geq \tau$)")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Aggregate solve rate ± 95% Wilson CI")
    axes[1].grid(True, alpha=0.3, axis="y")
    _savefig(fig, "llm_fig5_solve_comparison")


def fig_edge(datasets: dict[str, np.ndarray], edge_ci: dict[str, dict]) -> None:
    """Per-round edge on measured cells, and the abstaining convergence bound.

    Left: bars are gamma_t restricted to cells where a model call actually happened
    (``headroom_edge_with_ci``), with 95% bootstrap CI whiskers. Grey crosses are the
    pooled gamma_t over every cell, shown so the gap between them is visible rather
    than argued: rounds after a task's first pass are back-filled with a hardcoded
    1.0 and no model call, and the pooled estimator charges each one as a tie, which
    drags it down in proportion to the solve rate. Round 0 has no predecessor and so
    no bar.

    Right: running-best pass error err_t (solid) against two versions of the
    transferred convergence bound. The dashed line is the confidence-rated bound
    ``prod_s Z_s``, ``Z_s = W_0 + 2 sqrt(W_+ W_-)``, which treats an unchanged round
    as an *abstention* -- it descends, slowly, because abstentions leave the bound
    alone rather than breaking it. The faint dotted line is the binary form
    ``exp(-2 sum max(0,gamma_s)^2)``, which pins at 1.0; that flatness is an artifact
    of a formulation with no representation for abstention, not a property of the
    loop, and it is drawn only to make the contrast explicit.

    ``datasets`` supplies the seed-averaged (K, T) matrix used for the running-best
    error curve. ``edge_ci`` supplies gamma_t, its CI, the headroom block and the
    abstention block per label, precomputed by ``part_b_edge_overlap`` -- read rather
    than recomputed here so the figure and the paper's macros cannot drift.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    labels = list(datasets.keys())
    t_max = max(len(edge_ci[lab]["gamma"]) for lab in labels)
    rounds = np.arange(t_max)
    width = 0.38

    for j, lab in enumerate(labels):
        hr = edge_ci[lab]["headroom"]
        # None marks a round with no eligible population; plot it as absent, not as 0.
        g = np.array([np.nan if v is None else v for v in hr["gamma_headroom"]])
        lo = np.array([np.nan if v is None else v for v in hr["ci_lo"]])
        hi = np.array([np.nan if v is None else v for v in hr["ci_hi"]])
        x = np.arange(len(g)) + (j - 0.5) * width
        axes[0].bar(
            x,
            np.nan_to_num(g),
            width=width,
            color=COLORS[lab],
            alpha=0.85,
            label=f"{lab}: measured cells",
            yerr=[np.nan_to_num(g - lo), np.nan_to_num(hi - g)],
            capsize=5,
            error_kw={"linewidth": 1.5},
        )
        pooled = np.array(edge_ci[lab]["gamma"])
        axes[0].plot(
            x[1:],
            pooled[1:],
            "x",
            color="dimgray",
            markersize=8,
            markeredgewidth=2,
            label="pooled (incl. back-filled)" if j == 0 else None,
            zorder=5,
        )
    axes[0].axhline(y=0.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_xlabel("Refinement round $t$")
    axes[0].set_ylabel(r"Per-round edge  $\gamma_t$")
    axes[0].set_title(r"Edge on measured cells vs. pooled (95% CI)")
    axes[0].set_xticks(rounds)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, (lab, q) in enumerate(datasets.items()):
        rb = _running_best(q)
        err = (rb < TAU).mean(axis=0)
        r = np.arange(len(err))
        # Abstaining bound: cumulative product of Z_s, with round 0 contributing no factor.
        z = [1.0 if v is None else v for v in edge_ci[lab]["abstention"]["z"]]
        abstain = np.cumprod(z)
        binary = np.exp(-2.0 * np.cumsum(np.maximum(np.array(edge_ci[lab]["gamma"]), 0.0) ** 2))
        axes[1].plot(
            r, err, "-o", linewidth=2, color=COLORS[lab], label=f"{lab}: $\\mathrm{{err}}_t$"
        )
        axes[1].plot(
            r,
            abstain[: len(r)],
            "--",
            linewidth=1.5,
            color=COLORS[lab],
            alpha=0.8,
            label=f"{lab}: $\\prod_s Z_s$",
        )
        # Both tiers' binary bounds sit at 1.0, so label it once rather than twice.
        axes[1].plot(
            r,
            binary[: len(r)],
            ":",
            linewidth=1.2,
            color="dimgray",
            alpha=0.6,
            label=r"binary $e^{-2\sum\max(0,\gamma_s)^2}$" if i == 0 else None,
        )
    axes[1].set_xlabel("Refinement round $t$")
    axes[1].set_ylabel(r"Running-best pass error  $\mathrm{err}_t$")
    axes[1].set_title(r"Abstaining bound descends; binary form pins at 1")
    axes[1].set_xticks(rounds)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    _savefig(fig, "llm_fig_edge_collapse")


def fig_overlap(overlap: dict) -> None:
    """The failMult overlap estimator for Prop. 18 (bounded overlap <-> correct vote).

    Left: how many of the k workers fail each task (the m_i distribution over the 55
    tasks), with a dashed line at k/2. Prop. 18 needs *every* task strictly left of
    that line; the mass at or right of it is exactly the unweighted majority vote's
    zero-one error, and the spike at m=k is the total-overlap case the appendix
    predicts for same-family resampling.
    Right: per round, the vote's error against the mean per-worker error eps_bar and
    the Markov bound 2*eps_bar (cor_overlap_markov_bound) -- the bound holds but sits
    far above both, and the vote's error stays above eps_bar at every round. Every bar
    carries its cluster-bootstrap interval, read from the cache's per_round block, which
    shares one index matrix across rounds so the trajectory is comparable round to round
    (part_b_edge_overlap._ci_block). The Markov whisker is a deterministic 2x of
    eps_bar's, drawn for completeness rather than as independent information.

    These are *marginal* intervals and they overlap at every round, which is the weaker
    of the two comparisons: the paired gap actually excludes zero at the early rounds and
    straddles it only at the last (see app:orchestration). So the panel must not be
    titled as though overlap settled the question -- it names which comparison it shows
    and leaves the paired test to the appendix, where check (iii) attributes the excess
    to tier pooling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    k, n_tasks = overlap["k"], overlap["n_tasks"]
    hist = np.array(overlap["m_histogram"])
    ms = np.arange(len(hist))
    violating = 2 * ms >= k
    axes[0].bar(
        ms[~violating], hist[~violating], color="seagreen", alpha=0.85, label="vote correct"
    )
    axes[0].bar(
        ms[violating],
        hist[violating],
        color="firebrick",
        alpha=0.85,
        label=r"vote errs ($2m_i\geq k$)",
    )
    axes[0].axvline(
        x=overlap["half_k"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$k/2={overlap['half_k']:g}$",
    )
    axes[0].annotate(
        rf"$m^\star={overlap['m_star']}=k$",
        xy=(overlap["m_star"], hist[overlap["m_star"]]),
        xytext=(overlap["m_star"] - 7, max(hist) * 0.55),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        fontsize=10,
        fontweight="bold",
    )
    axes[0].set_xlabel(rf"Workers failing a task  $m_i$  (of $k={k}$)")
    axes[0].set_ylabel(f"Tasks (of {n_tasks})")
    axes[0].set_title(
        rf"Overlap $m^\star<k/2$ fails: {overlap['n_violating']}/{n_tasks} tasks violate it"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    per_round = overlap["per_round"]
    xs = np.arange(len(per_round))
    width = 0.27
    series = [
        ("vote error", "vote_err", "vote_err_ci", "firebrick"),
        (r"mean single worker $\bar{\epsilon}$", "eps_bar", "eps_bar_ci", "steelblue"),
        (r"Markov bound $2\bar{\epsilon}$", "markov_bound", "markov_bound_ci", "silver"),
    ]
    for j, (label, key, ci_key, color) in enumerate(series):
        # markov_bound is not stored per round (it is 2*eps_bar by definition), so derive
        # it here while reading the interval straight from the cache.
        vals = [2 * e["eps_bar"] if key == "markov_bound" else e[key] for e in per_round]
        offsets = xs + (j - 1) * width
        axes[1].bar(offsets, vals, width=width, color=color, alpha=0.85, label=label)
        # A whisker on every bar of every round, not just the round the paper quotes: the
        # panel compares two error levels across the trajectory, so showing uncertainty
        # for one round out of four invited reading the other three as exact.
        los = [obs - e[ci_key][0] for obs, e in zip(vals, per_round, strict=True)]
        his = [e[ci_key][1] - obs for obs, e in zip(vals, per_round, strict=True)]
        axes[1].errorbar(
            offsets,
            vals,
            yerr=[los, his],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )
    # The Markov bound exceeds 1 at round 0 (2*eps_bar > 1): keep the axis above it so
    # the bound is visibly vacuous there rather than silently clipped at the frame. The
    # ceiling is set by that bar's CI top (~1.27), not the bar (~1.11) -- a clipped
    # whisker reads as a shorter interval, which is worse than no whisker at all.
    axes[1].axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("Refinement round $t$")
    axes[1].set_ylabel("Zero-one error")
    # Name the comparison rather than pronouncing on it. The old title read "(CIs
    # overlap)", which invites "so there is no difference" -- but that is the marginal
    # comparison, and the paired gap excludes zero at rounds 0..2 (`gap_ci` per round in
    # the cache). The paired test belongs to the appendix; this panel shows the levels.
    axes[1].set_title("Per-round error, 95% CIs (marginal; paired gap in text)")
    axes[1].set_ylim(0, 1.35)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3, axis="y")
    _savefig(fig, "llm_fig_overlap")


def fig_feedback_ablation(raw: dict[tuple[str, str], dict[str, list[float]]]) -> None:
    """Mean final-round quality by feedback mode, one line per model, mode on the
    x-axis in the hypothesized independent -> blind -> diagnostic order, +-SEM
    across the 55 tasks. Visualizes the K=55 ablation's most narratively
    interesting result: Haiku improves monotonically with more carried-forward
    context, Sonnet's `blind` is its worst arm (a crossover) -- point estimates
    only, no effect in this 2x3 design reaches significance (see paper text).
    """
    modes = ["independent", "blind", "diagnostic"]
    model_labels = {"haiku": "Haiku", "sonnet": "Sonnet"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(modes))
    for model, label in model_labels.items():
        means, sems = [], []
        for mode in modes:
            task_means = np.array([np.mean(v) for v in raw[(model, mode)].values()])
            means.append(task_means.mean())
            sems.append(task_means.std(ddof=1) / math.sqrt(len(task_means)))
        means_arr, sems_arr = np.array(means), np.array(sems)
        c = COLORS[label]
        ax.plot(x, means_arr, "-o", linewidth=2, color=c, label=label)
        ax.fill_between(x, means_arr - sems_arr, means_arr + sems_arr, alpha=0.15, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(["independent", "blind", "diagnostic"])
    ax.set_xlabel("Feedback mode")
    ax.set_ylabel("Mean final-round quality (55 tasks, 5 seeds, ±1 SEM)")
    ax.set_title("Feedback-mode ablation: no effect reaches significance (2×3 RM-ANOVA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, "llm_fig6_feedback_ablation")


def _load_diagnostic_k55(tag: str) -> np.ndarray:
    """Load the K=55, 5-seed ``diagnostic``-mode arm for one model as the
    seed-averaged (K, T) quality matrix (task-major blocks averaged over seeds --
    the DV the paper's ANOVA also uses), for the running-best error curve.

    The pooled (K*n_seeds, T) matrix that gamma_t is measured over is no longer
    built here: ``part_b_edge_overlap`` computes gamma_t and its bootstrap CI from
    the same results file, and ``fig_edge`` reads that instead.
    """
    d = _load(os.path.join(REPO, "data", f"llm_results_{tag}_diagnostic.json"))
    q = np.array(d["quality_matrix"])
    k, n_seeds = d["config"]["k"], d["config"]["n_seeds"]
    q3 = q.reshape(n_seeds, k, q.shape[1])  # task-major blocks repeated per seed
    return q3.mean(axis=0)


def main() -> None:
    haiku = _load(os.path.join(REPO, "data", "llm_results_haiku_t4_pinned.json"))
    sonnet = _load(os.path.join(REPO, "data", "llm_results_sonnet_t4_pinned.json"))

    if haiku["instance_ids"] != sonnet["instance_ids"]:
        raise ValueError("instance_ids mismatch between Haiku and Sonnet result files")
    instance_ids = haiku["instance_ids"]

    haiku_q = np.array(haiku["quality_matrix"])
    sonnet_q = np.array(sonnet["quality_matrix"])
    datasets = {"Haiku": haiku_q, "Sonnet": sonnet_q}

    print(
        f"Loaded (K=20 pinned, fig1/fig3/fig5 only): Haiku {haiku_q.shape}, "
        f"Sonnet {sonnet_q.shape}, K={len(instance_ids)}"
    )

    print("Generating fig1 (pass convergence, dual-model + Wilson CI)...")
    fig1_pass_convergence(datasets)

    print("Generating fig3 (saturation + eta_bar, dual-model + SEM)...")
    fig3_saturation(datasets)

    print("Generating fig5 (per-task solve comparison + aggregate CI bars)...")
    fig5_solve_comparison(haiku_q, instance_ids, sonnet_q)

    haiku_disp = _load_diagnostic_k55("haiku")
    sonnet_disp = _load_diagnostic_k55("sonnet")
    cache_path = os.path.join(REPO, "data", EDGE_OVERLAP_CACHE)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Missing {cache_path} -- run: uv run python scripts/part_b_edge_overlap.py"
        )
    edge_overlap = _load(cache_path)
    edge_ci = {
        "Haiku": edge_overlap["edge_gamma"]["haiku"],
        "Sonnet": edge_overlap["edge_gamma"]["sonnet"],
    }
    print(
        f"Loaded (K=55, 5-seed diagnostic arm, fig_edge): Haiku display "
        f"{haiku_disp.shape}, Sonnet display {sonnet_disp.shape}; gamma_t + "
        f"{edge_ci['Haiku']['n_resamples']}-rep bootstrap CI from {EDGE_OVERLAP_CACHE}"
    )
    print("Generating fig_edge (measured-cell edge + CI + abstaining bound, K=55 diagnostic)...")
    fig_edge({"Haiku": haiku_disp, "Sonnet": sonnet_disp}, edge_ci)

    print("Generating fig_overlap (failMult overlap vs Prop. 18's m* < k/2, 6 arms x 5 seeds)...")
    fig_overlap(edge_overlap["overlap"])

    print("Generating fig6 (feedback-mode ablation, K=55, crossover pattern)...")
    raw = load_raw_by_arm()
    fig_feedback_ablation(raw)

    print(f"Done. Figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
