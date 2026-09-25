"""Guard the stats-macro surface: the numbers the paper's prose reaches through macros.

Every statistic in the paper arrives through a ``\\stat*`` macro generated into
``stats_macros.tex``, so a number reaches the reader only if some sentence still cites its
macro. Nothing checked that. ``test_generate_stats_macros.py`` checks the generator that
*produces* the values, and ``pdflatex`` fails loudly on an undefined macro, but neither
notices the failure that a prose revision actually causes: cutting or rewriting a sentence
drops the last citation of a macro, the value stops reaching the reader, and the build stays
green because an unused ``\\newcommand`` is perfectly legal LaTeX.

``scripts/check_macro_usage.py`` already computes all three sets. It was reachable only by
hand, and its ``--strict`` flag only fails on used-but-undefined -- the one direction
``pdflatex`` already catches. These tests wire it in, and cover the direction that matters
when prose is being cut.

Three invariants:

1. every macro the paper cites is defined, so no sentence renders as a raw command;
2. no macro falls out of use, checked as exact equality against a one-name allowlist -- so
   a *new* orphan trips the gate and retiring the documented one also has to be deliberate;
3. no macro is reachable only from the abstract. An abstract is the most-rewritten
   paragraph in any paper, and a statistic cited nowhere else is one edit from silent
   removal.

What this does not reach is whether a number is cited in the *right* sentence, or whether
the surrounding prose describes it correctly. Invariant 3 narrows where that can go wrong;
it does not check it. That needs a human reading both.
"""

from __future__ import annotations

from pathlib import Path

import scripts.check_macro_usage as cmu

ROOT = Path(__file__).parent

# The final-round eligible count for the weak tier. stats_macros.tex:150-160 documents why
# it is generated but never cited: 158 already reaches the reader inside
# \statPBHaikuHeadroomN ("194->158 of 275"), a second mention would be redundant prose, and
# deleting the declaration would make the generator's patch_macros raise on a key it still
# emits. Held as an allowlist rather than an exemption -- see the exact-equality note below.
DELIBERATELY_UNUSED = frozenset({"statPBHaikuHeadroomNLast"})


def test_every_macro_the_paper_uses_is_defined() -> None:
    """A macro cited but never declared renders as a raw command, not a number."""
    used = cmu.used_macros(ROOT)
    undefined = (used["paper.tex"] | used["paper_appendix.tex"]) - cmu.defined_macros(ROOT)
    assert undefined == set(), (
        f"cited in the paper but absent from stats_macros.tex: {sorted(undefined)}. "
        "Declare the \\newcommand first -- the generator patches existing lines and "
        "cannot create one."
    )


def test_no_macro_falls_out_of_use() -> None:
    """A dropped citation silently removes a reported number; the build stays green.

    Exact equality, not a subset check: a macro that stops being cited is a number that
    stopped reaching the reader, and the fix is to restore the mention rather than to widen
    this list. Retiring the documented orphan must edit the allowlist on purpose.
    """
    used = cmu.used_macros(ROOT)
    unused = cmu.defined_macros(ROOT) - (used["paper.tex"] | used["paper_appendix.tex"])
    assert unused == set(DELIBERATELY_UNUSED), (
        f"defined-but-uncited: {sorted(unused)}, expected {sorted(DELIBERATELY_UNUSED)}. "
        "A newly orphaned macro means a prose edit dropped a number -- restore the "
        "citation instead of allowlisting it."
    )


def test_no_number_reaches_the_reader_only_through_the_abstract() -> None:
    """The abstract is the most-rewritten paragraph; a statistic held only there is fragile."""
    orphan_risk = cmu.abstract_only(ROOT, cmu.used_macros(ROOT)["paper_appendix.tex"])
    assert orphan_risk == set(), (
        f"cited only inside the abstract: {sorted(orphan_risk)}. Cite each one in the body "
        "or the appendix too, so rewriting the abstract cannot drop it."
    )
