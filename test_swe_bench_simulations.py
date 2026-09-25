"""Smoke tests for swe_bench_simulations.py and the agentic boosting core.

swe_bench_simulations.py loads the SWE-bench HuggingFace dataset at module
level, so importing it in CI without a local cache triggers a heavy download.
The tests here are split into two layers:

  1. Core-logic smoke tests (always run): exercise agentic_boosting from
     simulations.py with tiny synthetic data (N=10, T=3) and validate the
     schema of the committed artefact data/swe_exp1_results.json.

  2. Module-level smoke test (run when SWE-bench data is available): imports
     swe_bench_simulations with a mocked dataset, calls generate_swe_bench_dataset
     with N=10, and checks output shape.  Skipped automatically when the
     HuggingFace cache is absent and no mock is injected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent
_SWE_EXP1_JSON = _REPO_ROOT / "data" / "swe_exp1_results.json"

# Expected edge keys written by experiment_1()
_EXP1_EDGE_KEYS = {"0.05", "0.1", "0.2", "0.3"}
_EXP1_T_MAX = 100  # rounds written per edge in the committed JSON


def _make_synthetic_xy(n: int = 10, d: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    true_w = rng.randn(d)
    y = np.where(X @ true_w > 0, 1.0, -1.0)
    return X, y


# ---------------------------------------------------------------------------
# 1. Core agentic_boosting smoke tests (no SWE-bench required)
# ---------------------------------------------------------------------------


def test_agentic_boosting_return_structure_small() -> None:
    """agentic_boosting returns a 4-tuple with the right lengths for T=3."""
    from simulations import agentic_boosting

    X, y = _make_synthetic_xy(n=10, d=10)
    agents, alphas, train_errors, margin_history = agentic_boosting(X, y, T=3)

    assert len(agents) == 3, "Expected one agent per round"
    assert len(alphas) == 3, "Expected one alpha per round"
    assert len(train_errors) == 3, "Expected one train_error per round"
    assert len(margin_history) == 3, "Expected one margin distribution per round"


def test_agentic_boosting_train_errors_in_unit_interval() -> None:
    """All training errors must lie in [0, 1]."""
    from simulations import agentic_boosting

    X, y = _make_synthetic_xy(n=10, d=10)
    _, _, train_errors, _ = agentic_boosting(X, y, T=3)

    for t, err in enumerate(train_errors):
        assert 0.0 <= err <= 1.0, f"train_errors[{t}]={err:.4f} outside [0, 1]"


def test_agentic_boosting_margin_shape() -> None:
    """margin_history[t] is a 1-D array with one entry per sample."""
    from simulations import agentic_boosting

    N = 10
    X, y = _make_synthetic_xy(n=N, d=10)
    _, _, _, margin_history = agentic_boosting(X, y, T=3)

    for t, margins in enumerate(margin_history):
        assert len(margins) == N, f"margin_history[{t}] has {len(margins)} entries, expected {N}"


def test_agentic_boosting_alphas_positive() -> None:
    """Alpha weights must be finite and positive (well-calibrated weak learner)."""
    from simulations import agentic_boosting

    X, y = _make_synthetic_xy(n=20, d=10, seed=1)
    _, alphas, _, _ = agentic_boosting(X, y, T=3)

    for t, alpha in enumerate(alphas):
        assert np.isfinite(alpha), f"alpha[{t}]={alpha} is not finite"
        assert alpha > 0, f"alpha[{t}]={alpha} is not positive"


# ---------------------------------------------------------------------------
# 2. swe_exp1_results.json schema tests (no SWE-bench required)
# ---------------------------------------------------------------------------


def test_swe_exp1_results_exists() -> None:
    assert _SWE_EXP1_JSON.exists(), f"Missing committed artefact: {_SWE_EXP1_JSON}"


def test_swe_exp1_results_schema() -> None:
    """exp1_results JSON has the expected edge keys and list structure."""
    data: dict[str, Any] = json.loads(_SWE_EXP1_JSON.read_text())

    # Must contain all four edge keys written by experiment_1()
    missing = _EXP1_EDGE_KEYS - data.keys()
    assert not missing, f"swe_exp1_results.json missing keys: {missing}"

    for key in _EXP1_EDGE_KEYS:
        values = data[key]
        assert isinstance(values, list), f"Key '{key}' value is not a list"
        assert len(values) == _EXP1_T_MAX, (
            f"Key '{key}' has {len(values)} entries; expected {_EXP1_T_MAX}"
        )
        for i, v in enumerate(values):
            assert isinstance(v, (int, float)), (
                f"Key '{key}'[{i}] is type {type(v).__name__}, expected numeric"
            )
            assert 0.0 <= v <= 1.0, f"Key '{key}'[{i}]={v:.4f} outside [0, 1]"


def test_swe_exp1_results_overall_decay_for_large_edge() -> None:
    """For the largest edge (0.3) the training error must end substantially lower
    than it starts.  The edge-calibrated learner introduces per-round randomness
    so strict per-round monotonicity is not guaranteed, but the boosting bound
    guarantees aggregate decay: the final-10-round average must beat the first-10.
    """
    data: dict[str, Any] = json.loads(_SWE_EXP1_JSON.read_text())
    errors = data["0.3"]
    first_avg = sum(errors[:10]) / 10
    last_avg = sum(errors[-10:]) / 10
    assert last_avg < first_avg, (
        f"No aggregate decay for edge=0.3: first-10 avg={first_avg:.4f}, last-10 avg={last_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Module-level smoke test with mocked SWE-bench dataset
# ---------------------------------------------------------------------------


def _build_mock_dataset(n_rows: int = 300) -> MagicMock:
    """Build a minimal mock that satisfies swe_bench_simulations' module-level code.

    The mock must yield enough unique non-stop-word tokens for TfidfVectorizer
    to produce max_features=50 columns from the resulting vocabulary.
    """
    import random as _random

    rng = _random.Random(0)
    # Build a corpus of varied pseudo-technical words so TF-IDF finds ≥50 features
    base_words = [
        "authentication",
        "database",
        "migration",
        "serializer",
        "endpoint",
        "middleware",
        "decorator",
        "validator",
        "scheduler",
        "pipeline",
        "transaction",
        "rollback",
        "commit",
        "branch",
        "deployment",
        "container",
        "registry",
        "orchestration",
        "kubernetes",
        "terraform",
        "async",
        "concurrent",
        "thread",
        "process",
        "executor",
        "callback",
        "promise",
        "observable",
        "reactive",
        "stream",
        "pagination",
        "cursor",
        "filter",
        "aggregate",
        "projection",
        "encryption",
        "signature",
        "certificate",
        "token",
        "payload",
        "compression",
        "parsing",
        "tokenizer",
        "lexer",
        "grammar",
        "inference",
        "embedding",
        "attention",
        "transformer",
        "gradient",
        "optimizer",
        "checkpoint",
        "evaluation",
        "benchmark",
        "profiler",
        "allocation",
        "garbage",
        "collector",
        "heap",
        "stack",
    ]
    texts = [" ".join(rng.sample(base_words, k=min(20, len(base_words)))) for _ in range(n_rows)]
    mock_ds = MagicMock()
    mock_ds.__getitem__ = MagicMock(return_value=texts)
    return mock_ds


@pytest.fixture(scope="module")
def swe_module() -> Any:
    """Import swe_bench_simulations with a mocked HuggingFace dataset.

    The module runs dataset loading at import time, so we patch
    datasets.load_dataset BEFORE the first import and ensure any prior
    import is evicted from sys.modules.
    """
    # Evict prior import if present (e.g., from a previous test run in-process)
    for key in list(sys.modules.keys()):
        if "swe_bench_simulations" in key:
            del sys.modules[key]

    mock_ds = _build_mock_dataset(n_rows=300)

    # Patch at the datasets namespace that swe_bench_simulations imports from
    with patch("datasets.load_dataset", return_value=mock_ds):
        import swe_bench_simulations as swe  # noqa: PLC0415

    return swe


def test_swe_module_x_all_shape(swe_module: Any) -> None:
    """After mock-import, X_all must be a 2-D standardised float array.

    With max_features=50 and the varied-vocabulary mock corpus the vectorizer
    returns up to 50 features; with the real SWE-bench dataset it is exactly 50.
    We enforce ndim==2 and that at least 1 feature column exists.
    """
    X_all = swe_module.X_all
    assert X_all.ndim == 2, f"X_all.ndim={X_all.ndim}, expected 2"
    assert X_all.shape[0] > 0, "X_all has no rows"
    assert X_all.shape[1] >= 1, "X_all has no feature columns"


def test_generate_swe_bench_dataset_shape(swe_module: Any) -> None:
    """generate_swe_bench_dataset(N=10) returns arrays with consistent shapes.

    We use n_features=X_all.shape[1] to adapt to whatever the mocked corpus
    produced; the real corpus always yields exactly 50 features.
    """
    n_features = swe_module.X_all.shape[1]
    X, y, true_w = swe_module.generate_swe_bench_dataset(n_samples=10, n_features=n_features)
    assert X.shape == (10, n_features), f"X.shape={X.shape}, expected (10, {n_features})"
    assert y.shape == (10,), f"y.shape={y.shape}, expected (10,)"
    assert true_w.shape == (n_features,), f"true_w.shape={true_w.shape}"


def test_generate_swe_bench_dataset_binary_labels(swe_module: Any) -> None:
    """Labels must be in {-1, +1}."""
    n_features = swe_module.X_all.shape[1]
    _, y, _ = swe_module.generate_swe_bench_dataset(n_samples=10, n_features=n_features)
    assert set(y).issubset({-1.0, 1.0}), f"Unexpected label values: {set(y)}"
