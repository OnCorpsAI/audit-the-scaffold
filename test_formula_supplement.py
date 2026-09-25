"""Guard the formula supplement against the paper it claims to decode.

``formula_supplement.tex`` is a standalone companion: it does not ``\\input`` the paper,
so nothing in the build couples the two. That independence is what let it drift. Found
2026-08-18: eleven of its twelve numbered cross-references pointed at an older draft's
numbering -- "Definition~2 (Strong Coding System)" when result 2 is Thm. 2 (Monotone
Improvement), "Proposition~12 (Stationarity Dichotomy)" when the dichotomy is Prop. 3 --
because the paper moved five theorem environments into the appendix and the supplement
was never resynced. A reader following those pointers lands on the wrong result.

Two invariants, both derived from the sources rather than from a build artifact
(``paper.aux`` is generated and gitignored, so the tests may not depend on it):

1. every ``Kind~N`` cross-reference in the supplement names the right *kind* for result
   ``N``, and where the reference carries a parenthesized title, that title agrees with
   the theorem environment's own bracketed name;
2. the supplement covers every result the *main body* states, so a proposition promoted
   into the body cannot stay undecoded (this is how Props. 5-7 -- the orchestration,
   voting and pool-reuse results the abstract leads with -- were found missing).

Numbering is derived the way LaTeX derives it: ``\\newtheorem{theorem}{Theorem}`` and
``[theorem]`` for every other environment means one shared counter, incremented in
document order across ``paper.tex`` and then ``paper_appendix.tex``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
PAPER = ROOT / "paper.tex"
APPENDIX = ROOT / "paper_appendix.tex"
SUPPLEMENT = ROOT / "formula_supplement.tex"

# Environments sharing the `theorem` counter, per paper.tex's preamble.
SHARED_COUNTER = ("theorem", "corollary", "definition", "lemma", "proposition", "remark")

ENV_START = re.compile(
    r"\\begin\{(" + "|".join(SHARED_COUNTER) + r")\}(?:\[([^\]]*)\])?", re.MULTILINE
)
LABEL_AFTER = re.compile(r"\\label\{([^}]*)\}")

# "Theorem~10 (Training Error of Agentic Boosting)" inside an \xref, and the bare
# "Prop.~18" / "Props.~20 and~21" short forms the newer sections use.
XREF_LONG = re.compile(
    r"\b(Theorem|Corollary|Definition|Lemma|Proposition|Remark)~(\d+)\s*\(([^)]*)\)"
)
XREF_SHORT = re.compile(r"\b(Thm|Cor|Def|Prop|Props|Rem)\.~(\d+)")
SHORT_TO_KIND = {
    "Thm": "theorem",
    "Cor": "corollary",
    "Def": "definition",
    "Prop": "proposition",
    "Props": "proposition",
    "Rem": "remark",
}


def _results() -> list[tuple[str, str, str]]:
    """(kind, bracketed_name, label) for every shared-counter environment, in order.

    Index i in this list is result number i+1.
    """
    out: list[tuple[str, str, str]] = []
    for path in (PAPER, APPENDIX):
        text = path.read_text()
        for m in ENV_START.finditer(text):
            kind, name = m.group(1), (m.group(2) or "")
            tail = text[m.end() : m.end() + 200]
            label_match = LABEL_AFTER.search(tail)
            out.append((kind, name, label_match.group(1) if label_match else ""))
    return out


def _supplement_xrefs() -> list[tuple[str, int, str]]:
    """(kind, number, title) for each numbered cross-reference; title may be ''."""
    text = SUPPLEMENT.read_text()
    found = [(m.group(1).lower(), int(m.group(2)), m.group(3)) for m in XREF_LONG.finditer(text)]
    long_spans = {(m.start(), m.end()) for m in XREF_LONG.finditer(text)}
    for m in XREF_SHORT.finditer(text):
        if any(s <= m.start() < e for s, e in long_spans):
            continue
        found.append((SHORT_TO_KIND[m.group(1)], int(m.group(2)), ""))
    return found


def _normalise(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+|[{}$\\~]", " ", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def test_the_supplement_cites_paper_results_at_all() -> None:
    """A parser that silently matches nothing would make every other test vacuous."""
    refs = _supplement_xrefs()
    assert len(refs) >= 12, f"only {len(refs)} cross-references parsed; the regex has drifted"


def test_result_numbering_derivation_is_sane() -> None:
    """Anchor the derived numbering on three results whose numbers are load-bearing."""
    results = _results()
    by_label = {label: i + 1 for i, (_, _, label) in enumerate(results) if label}
    assert by_label.get("def:weak-agent") == 1, by_label.get("def:weak-agent")
    assert by_label.get("thm:refinement") == 2, by_label.get("thm:refinement")
    # The body states seven results; the eighth is the first appendix one.
    assert by_label.get("def:strong-agent") == 8, by_label.get("def:strong-agent")


@pytest.mark.parametrize("ref", _supplement_xrefs())
def test_supplement_cross_reference_matches_the_paper(ref: tuple[str, int, str]) -> None:
    kind, number, title = ref
    results = _results()
    assert 1 <= number <= len(results), (
        f"supplement cites {kind.title()}~{number}, but the paper has {len(results)} results"
    )
    actual_kind, actual_name, label = results[number - 1]
    assert actual_kind == kind, (
        f"supplement calls result {number} a {kind}; in the paper it is a {actual_kind} "
        f"({label or 'unlabelled'}: {actual_name!r}). Renumber the \\xref."
    )
    if title and actual_name:
        got, want = _normalise(title), _normalise(actual_name)
        assert got in want or want in got, (
            f"supplement titles result {number} {title!r}; the paper calls it "
            f"{actual_name!r} ({label})."
        )


def test_every_main_body_result_is_decoded_by_the_supplement() -> None:
    """The supplement's stated job is every defining and result formula in the paper.

    Restricted to the main body: appendix results are decoded selectively and on purpose,
    but a result the body states -- and the abstract leads with -- must not be missing.
    Props. 5-7 (orchestration, bounded overlap, reused pool) were absent when this test
    was written, which is why it exists.
    """
    body_count = len(list(ENV_START.finditer(PAPER.read_text())))
    cited = {number for _, number, _ in _supplement_xrefs()}
    missing = sorted(set(range(1, body_count + 1)) - cited)
    assert not missing, (
        f"main-body results with no supplement section: {missing} "
        f"(of {body_count} in paper.tex). Add a section or cite the result in an \\xref."
    )
