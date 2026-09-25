"""Guard the Lean citation surface: the paper's claims *about* its own formalization.

The Lean development checks the algebra against hallucinated derivations, and
``test_references.py`` checks the bibliography. Nothing checked the layer between them --
the prose that names Lean declarations. That layer has already produced a real defect
(Prop.~8's prose overstated what its Lean theorem proves -- the declarations still carry
their draft-era ``prop15_`` prefix), and it is invisible to every
other gate: ``lake build`` proves the Lean is sound but says nothing about what the paper
claims it shows, and ``pdflatex`` renders ``\\path{thm_does_not_exist}`` without complaint.

Four invariants, all offline and deterministic:

1. every Lean identifier cited in the paper resolves to a real declaration in
   ``lean_proofs/LeanProofs/`` -- so a renamed or deleted lemma cannot leave a dangling
   citation in the Formalization Index;
2. the two allowlists (Mathlib names, non-Lean identifiers) stay honest in the other
   direction -- an entry that becomes a local declaration must leave the list rather than
   hold a permanent exemption;
3. ``lean_proofs/LeanProofs/`` contains no ``sorry`` and no ``axiom``, which Appendix~H
   and the limitations list both assert outright;
4. every theorem/corollary/proposition in either ``.tex`` appears in a Formalization Index
   row. Appendix~H opens "Every result above is machine-checked ... these two tables map
   each one to the declarations that carry it", so a result with no row makes that
   sentence false and leaves a reader going statement->proof term at a dead end.

Invariant 4 is the one that found something: Prop.~10 (the reused-pool dichotomy, the main
text's sharpest result) had no row, while all 18 other results did.

What none of this reaches is *semantics* -- whether a row's description ("tight equality",
"takes X as a hypothesis", "for non-vacuity") matches what the named declaration actually
proves. That needs a human reading both, and no test replaces it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SOURCES = ("paper.tex", "paper_appendix.tex")
LEAN_DIR = ROOT / "lean_proofs" / "LeanProofs"

# Lean declarations sit at column 0 after an optional `noncomputable section` (which is a
# section, not a modifier -- there is no `noncomputable def` in this development). `def`
# and `abbrev` are included because the index cites `wt`, `voteErr` and `failMeasure`,
# which are definitions rather than proofs.
DECLARATION = re.compile(
    r"^(?:theorem|lemma|def|abbrev|instance)\s+([A-Za-z_][A-Za-z0-9_']*)",
    re.MULTILINE,
)

# The paper cites Lean names two ways: `\path{...}` in the Appendix~H tables (chosen so
# underscores need no escaping) and `\texttt{...}` in running prose (where each underscore
# is escaped). Both are collected and normalised to bare identifiers.
PATH_CITATION = re.compile(r"\\path\{([^}]*)\}")
TEXTTT_CITATION = re.compile(r"\\texttt\{([^}]*)\}")

# Mathlib declarations, correctly attributed as such in the prose. They are cited to say
# what the local proofs are built *on*, so they must not be required to exist locally.
MATHLIB_NAMES = frozenset(
    {
        "iIndepFun_pi",  # independence of coordinates of a product measure
        "IsProbabilityMeasure",  # the measure class Cor. 5 is proved against
    }
)

# Identifiers that look Lean-shaped but are not declarations of any kind.
NON_LEAN_IDENTIFIERS = frozenset(
    {
        "h_binary",  # a hypothesis *binder* inside several theorems, not a declaration
        "dev_",  # anonymisation prefix for hashed developer ids (Part C)
        "repo_A",  # placeholder repository label in the sanitisation audit (Part C)
    }
)

# Definitions and remarks are not results, so Appendix~H makes no claim about them.
RESULT_ENVIRONMENTS = ("theorem", "corollary", "proposition")

RESULT_LABEL = re.compile(
    r"\\begin\{(" + "|".join(RESULT_ENVIRONMENTS) + r")\}(?:\[[^\]]*\])?\s*\n?\s*\\label\{([^}]+)\}"
)

# Results deliberately absent from the Formalization Index, each with the reason. Held
# empty: Prop. 10's absence was a defect, not an exemption, and the right fix was a row.
UNFORMALIZED_RESULTS: tuple[str, ...] = ()


def _source_text() -> str:
    return "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SOURCES)


def _strip_comments(source: str) -> str:
    """Blank out Lean comments, preserving line structure so line numbers stay usable.

    Both forms have to go, and block comments have to be handled properly rather than by
    line: `Theorem3.lean`'s docstring explains that the Schapire margin bound is taken as a
    hypothesis *instead of* a global `axiom`, so a naive scan reports the word `axiom` from
    the very prose that says it was avoided. Lean block comments nest, hence the depth
    counter rather than a regex.
    """
    out: list[str] = []
    depth = 0
    index = 0
    while index < len(source):
        two = source[index : index + 2]
        if two == "/-":
            depth += 1
            out.append("  ")
            index += 2
        elif two == "-/" and depth:
            depth -= 1
            out.append("  ")
            index += 2
        elif depth:
            char = source[index]
            out.append(char if char == "\n" else " ")
            index += 1
        elif two == "--":
            end = source.find("\n", index)
            end = len(source) if end == -1 else end
            out.append(" " * (end - index))
            index = end
        else:
            out.append(source[index])
            index += 1
    return "".join(out)


def _lean_declarations() -> set[str]:
    decls: set[str] = set()
    for path in sorted(LEAN_DIR.glob("*.lean")):
        source = _strip_comments(path.read_text(encoding="utf-8"))
        decls.update(DECLARATION.findall(source))
    return decls


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_']*\Z")


def _is_lean_shaped(token: str) -> bool:
    """Identify a citation as a Lean identifier rather than a path, flag, or constant.

    A bare identifier, underscore-bearing and not SCREAMING_CASE. That excludes
    `\\path{lean_proofs/...}` and file names (not identifiers), the config flags the
    appendix cites in `\\texttt{}` (`require_ai_trailer=False`), and the SWE-bench harness
    constants (`FAIL_TO_PASS`, `MAP_REPO_TO_PARSER`). Everything left is either a Lean name
    or one of the three documented exceptions above.
    """
    if "_" not in token or not IDENTIFIER.match(token):
        return False
    return token != token.upper()


def _cited_lean_names() -> set[str]:
    text = _source_text()
    raw = PATH_CITATION.findall(text) + [
        match.replace(r"\_", "_") for match in TEXTTT_CITATION.findall(text)
    ]
    return {token for token in raw if _is_lean_shaped(token)}


def _formalization_index() -> str:
    """Appendix~H, from its label to the end of the file."""
    appendix = (ROOT / "paper_appendix.tex").read_text(encoding="utf-8")
    start = appendix.index(r"\label{app:lean}")
    return appendix[start:]


def test_the_paper_cites_lean_names_at_all() -> None:
    """Fail loudly if the citation regexes stop matching, rather than passing vacuously on
    an empty set the way the three tests below otherwise would."""
    cited = _cited_lean_names()
    assert len(cited) > 50, (
        f"only {len(cited)} Lean citations found across {' and '.join(SOURCES)}; "
        "the extraction regexes have probably drifted from the paper's markup"
    )


def test_every_cited_lean_name_is_a_declaration() -> None:
    """A citation that names nothing is worse than no citation: it reads as a proof term a
    reviewer can look up, and renders identically to one that exists."""
    dangling = sorted(
        _cited_lean_names() - _lean_declarations() - MATHLIB_NAMES - NON_LEAN_IDENTIFIERS
    )
    assert not dangling, (
        f"Lean names cited by the paper but declared nowhere in {LEAN_DIR}: {dangling}. "
        "Fix the citation, or add the name to MATHLIB_NAMES / NON_LEAN_IDENTIFIERS "
        "with a reason."
    )


@pytest.mark.parametrize("allowlist", ["MATHLIB_NAMES", "NON_LEAN_IDENTIFIERS"])
def test_allowlist_has_no_stale_entries(allowlist: str) -> None:
    """Keep the exemptions honest in the other direction: a name that has since become a
    local declaration should leave the list, so it is checked like every other citation."""
    names = {"MATHLIB_NAMES": MATHLIB_NAMES, "NON_LEAN_IDENTIFIERS": NON_LEAN_IDENTIFIERS}[
        allowlist
    ]
    stale = sorted(names & _lean_declarations())
    assert not stale, f"{allowlist} entries that are now local declarations, remove them: {stale}"


@pytest.mark.parametrize("forbidden", ["sorry", "axiom"])
def test_lean_development_has_no_sorry_and_no_axiom(forbidden: str) -> None:
    """Appendix~H and the limitations list both state this outright ("no `axiom` and no
    `sorry`"), which makes it a claim in the paper rather than a repo convention."""
    pattern = re.compile(rf"(?<![A-Za-z_]){forbidden}(?![A-Za-z_])")
    offenders = []
    for path in sorted(LEAN_DIR.glob("*.lean")):
        code = _strip_comments(path.read_text(encoding="utf-8"))
        for number, line in enumerate(code.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        f"`{forbidden}` appears in the Lean development, contradicting the paper: {offenders}"
    )


def test_every_result_appears_in_the_formalization_index() -> None:
    """Appendix~H claims to map every result to the declarations carrying it. A result with
    no row makes that sentence false and strands a reader looking for the proof term."""
    index = _formalization_index()
    labels = [label for _, label in RESULT_LABEL.findall(_source_text())]
    assert len(labels) > 15, f"only {len(labels)} results parsed; the label regex has drifted"

    missing = sorted(
        label
        for label in labels
        if label not in UNFORMALIZED_RESULTS and rf"\ref{{{label}}}" not in index
    )
    assert not missing, (
        f"results with no Formalization Index row: {missing}. Add a row to "
        "tab:lean-core or tab:lean-orchestration, or list the label in "
        "UNFORMALIZED_RESULTS with a reason."
    )


def test_unformalized_results_list_has_no_stale_entries() -> None:
    """Same honesty check for the fourth invariant's escape hatch."""
    index = _formalization_index()
    stale = sorted(label for label in UNFORMALIZED_RESULTS if rf"\ref{{{label}}}" in index)
    assert not stale, f"UNFORMALIZED_RESULTS labels that now have a row, remove them: {stale}"
