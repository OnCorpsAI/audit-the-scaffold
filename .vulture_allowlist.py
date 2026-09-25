"""Vulture allowlist: symbols that are part of the public API / plug-in seams and
are intentionally not referenced from within the module itself.

Vulture is run as ``vulture $(PY_SOURCES) .vulture_allowlist.py --min-confidence 80``
(see the ``check`` target in the Makefile) over every top-level module, every test
module and ``scripts/``. Names listed here are treated as "used" so that dead-code
detection does not flag evaluator/agent-adapter extension points that exist to be
swapped in via the ``LLM_SIM_EVALUATOR`` / ``LLM_SIM_AGENT`` environment variables, or
dataclass fields consumed only through serialization.

**This file currently suppresses nothing, and has not for some time.** Measured when
the gate was widened from one file to the whole tree: ``vulture`` reports zero findings
at ``--min-confidence 80`` with the allowlist *removed*, both at the widened scope and
at the original ``llm_simulations.py``-only scope. Every seam below is in fact
referenced from inside ``llm_simulations.py`` by the factory functions that dispatch on
the env vars -- ``get_evaluator`` names ``LLMJudgeEvaluator`` and ``CompositeEvaluator``
directly, ``get_agent_runner`` names ``MiniSweAgentRunner``.

It is kept rather than deleted because the references it duplicates are incidental: a
refactor of those factories into a string-keyed registry would erase every static
reference at once while leaving the classes fully reachable through
``LLM_SIM_EVALUATOR``, and that is precisely the shape of change where vulture would
otherwise report a live plug-in seam as dead code. Treat the list as a statement of
intent about which names are dynamically dispatched, not as an active suppression --
and do not read an entry here as evidence that vulture would flag it today.
"""

# Evaluator seam implementations selected at runtime by env var.
LLMJudgeEvaluator  # noqa: F821
HarnessEvaluator  # noqa: F821
NormalizedEditSimEvaluator  # noqa: F821
LocalizationEvaluator  # noqa: F821
ReferenceJudgeEvaluator  # noqa: F821
ApplyCheckEvaluator  # noqa: F821
CompositeEvaluator  # noqa: F821
compute_signal_agreement  # noqa: F821

# Agent-adapter implementations selected at runtime by env var.
MiniSweAgentRunner  # noqa: F821

# Public entry points invoked from __main__ / external callers.
main  # noqa: F821

# Config field consumed only via ``asdict`` serialization in ``main`` (and only
# meaningful for the legacy ``bedrock/<model>`` + AWS path; unused by the proxy).
region  # noqa: F821
