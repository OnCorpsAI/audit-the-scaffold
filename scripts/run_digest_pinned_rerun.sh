#!/usr/bin/env bash
# Part 2e: Full digest-pinned re-run (Haiku + Sonnet, K=20, T=4, harness evaluator).
# Runs both models sequentially; copies named outputs after each.
# Usage: bash scripts/run_digest_pinned_rerun.sh 2>&1 | tee data/runs/digest_pinned_rerun.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="data/runs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Part 2e: Digest-pinned re-run"
echo "Started: $(date)"
echo "========================================"

# Shared settings
export LLM_SIM_K=20
export LLM_SIM_T=4
export LLM_SIM_EVALUATOR=harness
export LLM_SIM_AGENT=mini
export LLM_SIM_STEP_LIMIT=40
# LITELLM_API_KEY already in env

run_model() {
    local model="$1"
    local label="$2"
    echo ""
    echo "--- ${label} (${model}) ---"
    echo "Started: $(date)"
    LLM_SIM_MODEL="$model" uv run python llm_simulations.py
    # Copy outputs to named files (main() overwrites the generic names)
    cp data/llm_results.json  "data/llm_results_${label}_t4_pinned.json"
    cp data/llm_metrics.json  "data/llm_metrics_${label}_t4_pinned.json"
    for n in 1 2 3 4; do
        cp "figures/llm_fig${n}_pass_convergence.pdf"   "figures/llm_fig${n}_${label}_t4_pinned.pdf"  2>/dev/null || true
        cp "figures/llm_fig${n}_quality_edge.pdf"        "figures/llm_fig${n}_${label}_t4_pinned.pdf"  2>/dev/null || true
        cp "figures/llm_fig${n}_saturation.pdf"          "figures/llm_fig${n}_${label}_t4_pinned.pdf"  2>/dev/null || true
        cp "figures/llm_fig${n}_margin_distribution.pdf" "figures/llm_fig${n}_${label}_t4_pinned.pdf"  2>/dev/null || true
    done
    # Copy the four actual figure files
    cp figures/llm_fig1_pass_convergence.pdf    "data/runs/llm_fig1_${label}_t4_pinned_${TIMESTAMP}.pdf" 2>/dev/null || true
    cp figures/llm_fig2_quality_edge.pdf        "data/runs/llm_fig2_${label}_t4_pinned_${TIMESTAMP}.pdf" 2>/dev/null || true
    cp figures/llm_fig3_saturation.pdf          "data/runs/llm_fig3_${label}_t4_pinned_${TIMESTAMP}.pdf" 2>/dev/null || true
    cp figures/llm_fig4_margin_distribution.pdf "data/runs/llm_fig4_${label}_t4_pinned_${TIMESTAMP}.pdf" 2>/dev/null || true
    echo "Finished ${label}: $(date)"
}

run_model "litellm_proxy/claude-haiku-4-5-20251001" "haiku"
run_model "litellm_proxy/claude-sonnet-5"            "sonnet"

echo ""
echo "========================================"
echo "Part 2e complete: $(date)"
echo "Results: data/llm_results_{haiku,sonnet}_t4_pinned.json"
echo "========================================"
